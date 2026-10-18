"""Clip boundary refinement: trim leading fillers, then snap cuts into nearby silences; trim
idle (no speech, still picture) stretches off a clip's ends."""

import logging
import math
import re
from itertools import pairwise

from hookline.config import Settings
from hookline.models import ClipPick, Word
from hookline.pipeline.layout import COVER_MIN_S
from hookline.pipeline.signals import (
    LOW_MOTION,
    SCREEN_MARKER_MIN_S,
    STATIC_MIN_S,
    Signals,
    loud_seconds,
    low_motion,
    scare_seconds,
    scare_times,
    screen_runs,
    static_runs,
)

log = logging.getLogger(__name__)

FILLER_OPENERS: tuple[str, ...] = (
    "and",
    "so",
    "but",
    "because",
    "like",
    "um",
    "uh",
    "yeah",
    "okay",
    "well",
    "i mean",
    "you know",
    "anyway",
)

_MAX_FILLER_WORDS = 3
_MIN_MAIN_S = 15.0
_SNAP_MARGIN_S = 0.05
_FILLER_PAUSE_S = 0.12  # a filler must be followed by this much gap (or a silence) to go
_LEAD_IN_S = 0.10  # breath before the first word when no silence snap placed the start
_PUNCTUATION = ",.!?…\"'-:;–—‘’“”"


def _norm(word: Word) -> str:
    return word.text.lower().strip().strip(_PUNCTUATION)


def _paused_after(word: Word, nxt: Word, silences: list[tuple[float, float]]) -> bool:
    """True when a word gap >= 0.12 s or a silence separates `word` from `nxt`."""
    if nxt.start - word.end >= _FILLER_PAUSE_S - 1e-6:
        return True
    return any(a <= nxt.start and b >= word.end for a, b in silences)


def _trim_fillers(
    start: float,
    end: float,
    clip_words: list[Word],
    silences: list[tuple[float, float]],
    loud_seconds: set[int] | None = None,
) -> float:
    i = 0
    while i < len(clip_words) and i < _MAX_FILLER_WORDS:
        step = 0
        if (
            i + 1 < len(clip_words)
            and i + 2 <= _MAX_FILLER_WORDS
            and f"{_norm(clip_words[i])} {_norm(clip_words[i + 1])}" in FILLER_OPENERS
        ):
            step = 2
        if not step and _norm(clip_words[i]) in FILLER_OPENERS:
            step = 1
        if not step or i + step >= len(clip_words):
            break
        if loud_seconds and any(int(w.start) in loud_seconds for w in clip_words[i : i + step]):
            break  # a shouted opener ("YEAH!") is the reaction, not filler
        if not _paused_after(clip_words[i + step - 1], clip_words[i + step], silences):
            break  # the "filler" runs straight into the next word: it is part of the sentence
        new_start = clip_words[i + step].start
        if end - new_start < _MIN_MAIN_S:
            break
        start = new_start
        i += step
    return start


def refine_boundaries(
    pick: ClipPick,
    words: list[Word],
    silences: list[tuple[float, float]],
    window: float = 0.4,
    loud_seconds: set[int] | None = None,
) -> ClipPick:
    """Trim up to 3 leading filler words (only those followed by a pause), then snap
    start/end into silences within `window`. Without a start snap, the start gets a 0.10 s
    lead-in before the first word (never before the previous word's end).

    A filler word whose second is in `loud_seconds` is never trimmed.

    Never moves `start` past the first remaining word's start, nor `end` before the last
    word's end. Pure: no I/O."""
    start, end = pick.start, pick.end
    idxs = [i for i, w in enumerate(words) if start - 1e-6 <= w.start < end]
    start = _trim_fillers(start, end, [words[i] for i in idxs], silences, loud_seconds)

    first_idx = next((i for i in idxs if words[i].start >= start - 1e-6), None)
    last_idx = idxs[-1] if idxs else None

    lo, hi = 0.0, float("inf")  # a snap may only move into the gap between words
    first_start = last_end = None
    if first_idx is not None and last_idx is not None:
        first_start, last_end = words[first_idx].start, words[last_idx].end
        if first_idx > 0:
            lo = words[first_idx - 1].end
        if last_idx + 1 < len(words):
            hi = words[last_idx + 1].start

    near = _nearest_silence(silences, start, window)
    if near:
        start = near[1] - _SNAP_MARGIN_S
    elif first_start is not None:
        start = max(lo, first_start - _LEAD_IN_S)
    near = _nearest_silence(silences, end, window)
    if near:
        end = near[0] + _SNAP_MARGIN_S

    if first_start is not None and last_end is not None:
        start = min(max(start, lo), first_start)
        end = max(min(end, hi), last_end)
    return pick.model_copy(update={"start": start, "end": end})


def _nearest_silence(
    silences: list[tuple[float, float]], boundary: float, window: float
) -> tuple[float, float] | None:
    """The silence overlapping [boundary - window, boundary + window] closest to `boundary`."""
    hits = [s for s in silences if s[0] < boundary + window and s[1] > boundary - window]
    return min(hits, key=lambda s: max(s[0] - boundary, boundary - s[1], 0.0), default=None)


IDLE_MIN_S = 3.0  # a no-speech stretch longer than this at a clip's end may be idle
IDLE_TAIL_S = 1.0  # an idle tail is cut to this long after the last word
IDLE_LEAD_S = 0.3  # an idle lead is cut to this long before the first word (Q6: speech by 0.5 s)
# Per-second loudness (`Signals.loudness`, RMS dBFS) under this is quiet: the default
# silence level (`Settings.silence_db`); callers pass the configured one.
QUIET_DB: float = Settings.model_fields["silence_db"].default
INTERIOR_IDLE_MIN_S = 2.0  # an idle stretch inside a clip longer than this is cut ...
INTERIOR_IDLE_KEEP_S = 0.8  # ... down to this much (half kept on each side)
_CLOSERS = "\"'”’)"


def _spoken(pick: ClipPick, words: list[Word]) -> list[Word]:
    """The pick's words in its main span, outside its drops, in order."""
    return sorted(
        (
            w
            for w in words
            if pick.start - 1e-6 <= w.start < pick.end
            and not any(a <= w.start and w.end <= b for a, b in pick.drops)
        ),
        key=lambda w: w.start,
    )


def _quiet(signals: Signals, start: float, end: float, quiet_db: float) -> bool:
    """Every second from round(start) to round(end) (exclusive) is under `quiet_db`."""
    lo, hi = round(start), round(end)
    if lo < 0 or hi > len(signals.loudness):
        return False
    return all(db < quiet_db for db in signals.loudness[lo:hi])


def _out_of_drops(t: float, drops: list[tuple[float, float]], end: bool) -> float:
    """`t` moved out of a drop it lands inside (to the drop's start for an end, its end for a
    start): a bound inside a drop would disable it (`ClipPick.segments`)."""
    for a, b in drops:
        if a < t < b:
            return a if end else b
    return t


def trim_idle(
    pick: ClipPick, words: list[Word], signals: Signals, quiet_db: float = QUIET_DB
) -> ClipPick:
    """Cut an idle tail or lead off the clip's main span: more than IDLE_MIN_S without speech
    whose picture is still (`signals.low_motion`: a death, loading or menu screen, a pause),
    whose audio is quiet (every second under `quiet_db`: a long untranscribed laugh or game
    audio is not idle) and that holds no scare. The clip then ends IDLE_TAIL_S after its last
    word, or starts IDLE_LEAD_S before its first (never inside a drop). Words in dropped
    ranges are not speech. A no-op without motion signals (cached before them) or without
    words in the span. Logs each trim. Pure apart from logging."""
    if not signals.motion:
        return pick
    spoken = _spoken(pick, words)
    if not spoken:
        return pick
    scares = scare_seconds(signals)
    start, end = pick.start, pick.end

    def idle(a: float, b: float) -> bool:
        calm = not any(round(a) <= s < round(b) for s in scares)
        return (
            b - a > IDLE_MIN_S
            and calm
            and low_motion(signals.motion, a, b)
            and _quiet(signals, a, b, quiet_db)
        )

    last = max(w.end for w in spoken)
    if idle(last, end):
        end = _out_of_drops(last + IDLE_TAIL_S, pick.drops, end=True)
        log.info(
            "clip %r: idle tail %.1f-%.1fs (no speech, still picture); end moved to %.2f",
            pick.title, last, pick.end, end,
        )  # fmt: skip
    first = min(w.start for w in spoken)
    if idle(start, first):
        start = _out_of_drops(first - IDLE_LEAD_S, pick.drops, end=False)
        log.info(
            "clip %r: idle lead %.1f-%.1fs (no speech, still picture); start moved to %.2f",
            pick.title, pick.start, first, start,
        )  # fmt: skip
    if (start, end) == (pick.start, pick.end):
        return pick
    return pick.model_copy(update={"start": round(start, 3), "end": round(end, 3)})


def _idle_second(signals: Signals, k: int, quiet_db: float, scares: list[float]) -> bool:
    return (
        signals.motion is not None
        and 0 <= k < min(len(signals.motion), len(signals.loudness))
        and signals.motion[k] < LOW_MOTION
        and signals.loudness[k] < quiet_db
        and not any(k - 0.5 <= t < k + 1.5 for t in scares)
    )


def idle_gaps(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    quiet_db: float = QUIET_DB,
    quiet: bool = False,
) -> list[tuple[float, float]]:
    """Interior idle stretches to cut (source seconds): between two of the clip's words, the
    whole seconds with a still picture (`signals.LOW_MOTION`) and quiet audio (under
    `quiet_db`), in a pause holding no scare (the silence before one is its build-up), running
    more than INTERIOR_IDLE_MIN_S; each is cut down to
    INTERIOR_IDLE_KEEP_S (half kept on each side). Empty without motion signals. Logs each
    (unless `quiet`)."""
    if not signals.motion:
        return []
    spoken = _spoken(pick, words)
    scares = scare_times(signals)
    out: list[tuple[float, float]] = []
    for w, nxt in pairwise(spoken):
        a, b = w.end, nxt.start
        if b - a <= INTERIOR_IDLE_MIN_S or any(a <= t < b for t in scares):
            continue  # short, or the silence builds up to a scare
        secs = range(math.ceil(a - 1e-6), math.floor(b + 1e-6))  # whole seconds inside
        runs: list[list[int]] = []
        for k in secs:
            if not _idle_second(signals, k, quiet_db, scares):
                continue
            if runs and runs[-1][-1] == k - 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            lo, hi = float(run[0]), float(run[-1] + 1)
            if hi - lo <= INTERIOR_IDLE_MIN_S + 1e-6:
                continue
            half = INTERIOR_IDLE_KEEP_S / 2
            gap = (round(lo + half, 3), round(hi - half, 3))
            if any(gap[0] < y and x < gap[1] for x, y in pick.drops):
                continue
            out.append(gap)
            if quiet:
                continue
            log.info(
                "clip %r: interior idle %.1f-%.1fs (no speech, still picture, quiet); "
                "cut to %.1f s", pick.title, lo, hi, INTERIOR_IDLE_KEEP_S,
            )  # fmt: skip
    return out


CAUSE_LOOKBACK_S = 4.0  # a scare this close before the clip's first word is what it reacts to
CAUSE_REACT_S = 0.2  # words from this long before a scare's onset on are reactions to it
CAUSE_MAX_WORDS = 4  # the clip's start moves back by at most this many words ...
CAUSE_MAX_S = 6.0  # ... and at most this many seconds
_SENTENCE_END = (".", "!", "?", "…")
_SENTENCE_GAP_S = 1.0  # a pause this long before a word also starts a sentence


def _ends_sentence(word: Word) -> bool:
    return word.text.rstrip(_CLOSERS).endswith(_SENTENCE_END)


def _sentence_start(prev: Word | None, word: Word) -> bool:
    return prev is None or _ends_sentence(prev) or word.start - prev.end >= _SENTENCE_GAP_S


def _ends_on_punchline(spoken: list[Word]) -> bool:
    """`spoken` (in order) ends on a finished sentence that is not a shout (an exclamation of
    at most PAYOFF_SHOUT_WORDS words)."""
    if not spoken or not _ends_sentence(spoken[-1]):
        return False
    n = 1
    for prev, word in zip(spoken[-2::-1], spoken[::-1]):
        if _sentence_start(prev, word):
            break
        n += 1
    shout = spoken[-1].text.rstrip(_CLOSERS).endswith("!") and n <= PAYOFF_SHOUT_WORDS
    return not shout


def _filler_start(ws: list[Word]) -> bool:
    """The words open on a filler (`FILLER_OPENERS`) that refinement may trim."""
    two = " ".join(_norm(w) for w in ws[:2])
    return bool(ws) and (_norm(ws[0]) in FILLER_OPENERS or two in FILLER_OPENERS)


def _cause_block(before: list[Word], lo: float, room: int) -> list[Word]:
    """The longest run of whole sentences ending at `before`'s end, starting at or after
    `lo`, at most `room` words, not opening on a filler; [] when there is none."""
    new = [w for w in before if w.start >= lo]
    if not new or not _ends_sentence(new[-1]):
        return []  # nothing said, or the clip itself opens mid-sentence
    offset = len(before) - len(new)
    for j, w in enumerate(new):
        if len(new) - j > room:
            continue
        prev = before[offset + j - 1] if offset + j > 0 else None
        if _sentence_start(prev, w) and not _filler_start(new[j:]):
            return new[j:]
    return []


def include_cause(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """Start a clip that opens just after a scare (`signals.scare_times`: a game-audio
    transient or a flash with a sound, at its onset, within CAUSE_LOOKBACK_S before its first
    word) on the reaction said after that scare instead (the "Oh."s between the sound and the
    reaction line), so the viewer hears and sees the cause. Only whole sentences are added
    (else nothing), never one opening on a filler (refinement would trim it again); repeats
    while another scare precedes the new first word; at most CAUSE_MAX_WORDS earlier words and
    CAUSE_MAX_S. Skipped when the opener guard moved the start (`opener_moved`): the words
    before it are the weak opener it rejected. The added words are prepended to `first_words`
    and `declared_first_words`, so Q6 checks the opening the viewer now hears. Logs."""
    if pick.opener_moved is not None:
        return pick
    scares = scare_times(signals)
    if not scares:
        return pick
    before = sorted((w for w in words if w.start < pick.start - 1e-6), key=lambda w: w.start)
    inside = [w for w in words if pick.start - 1e-6 <= w.start < pick.end]
    if not inside:
        return pick
    first = min(w.start for w in inside)
    added: list[Word] = []
    while len(added) < CAUSE_MAX_WORDS:
        cause = next((t for t in scares if first - CAUSE_LOOKBACK_S < t < first), None)
        if cause is None:
            break
        lo = max(cause - CAUSE_REACT_S, pick.start - CAUSE_MAX_S)
        earlier = [w for w in before if w.start < first]
        block = _cause_block(earlier, lo, CAUSE_MAX_WORDS - len(added))
        if not block:
            break
        added = block + added
        first = block[0].start
    if not added:
        return pick
    log.info(
        "clip %r: a scare just before its first word; start moved %.2f -> %.2f (%s)",
        pick.title, pick.start, first, " ".join(w.text for w in added),
    )  # fmt: skip
    said = " ".join(w.text for w in added)
    update: dict = {"start": round(first, 3)}
    for key in ("first_words", "declared_first_words"):
        if getattr(pick, key):
            update[key] = f"{said} {getattr(pick, key)}"
    return pick.model_copy(update=update)


def _length(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    quiet_db: float,
    voiceless_s: float | None = None,
) -> float:
    """The pick's on-screen length before dead-air cuts and the end tail: its segments minus
    its interior edit gaps (`edit_gaps`: idle, static screens, voiceless stretches over
    `voiceless_s`, default VOICELESS_MAX_S)."""
    limit = VOICELESS_MAX_S if voiceless_s is None else voiceless_s
    gaps = edit_gaps(pick, words, signals, quiet_db, quiet=True, voiceless_s=limit)
    return pick.duration - sum(b - a for a, b in gaps)


def _front_trimmed(pick: ClipPick, spoken: list[Word], i: int) -> ClipPick:
    """`pick` starting on `spoken[i]`, its opening sentence as `first_words`."""
    start = spoken[i].start
    end_i = next((j for j in range(i, len(spoken)) if _ends_sentence(spoken[j])), len(spoken) - 1)
    return pick.model_copy(
        update={
            "start": round(start, 3),
            "drops": [d for d in pick.drops if d[1] > start],
            "first_words": " ".join(w.text for w in spoken[i : end_i + 1]),
            "declared_first_words": "",
        }
    )


def clamp_length(
    pick: ClipPick,
    original: ClipPick,
    words: list[Word],
    signals: Signals,
    max_s: float,
    quiet_db: float = QUIET_DB,
    voiceless_s: float | None = None,
) -> ClipPick:
    """The final hard length clamp before render: `pick` (fitted from `original`) lasting at
    most `max_s` (max_clip_s plus the payoff allowance), counted without its interior idle
    gaps and the end tail. Over it, the scare-cause words `include_cause` added go first (the
    original start, idle-trimmed); still over, the start moves to the earliest later sentence
    start (not a filler opener, not past the cold-open teaser) that fits, else the earliest
    later word that fits. `first_words` becomes the new opening sentence and
    `declared_first_words` is cleared (Claude's opening is no longer in the clip). Lengths
    count voiceless stretches over `voiceless_s` cut (`edit_gaps`). Logs."""
    full = _length(pick, words, signals, quiet_db, voiceless_s)
    if full <= max_s + 1e-6:
        return pick
    if pick.start < original.start - 1e-6:
        uncaused = trim_idle(original, words, signals, quiet_db)
        log.info(
            "clip %r: length %.1fs over %.0fs; scare-cause words dropped (start %.2f -> %.2f)",
            pick.title, full, max_s, pick.start, uncaused.start,
        )  # fmt: skip
        pick = uncaused
        full = _length(pick, words, signals, quiet_db, voiceless_s)
        if full <= max_s + 1e-6:
            return pick
    spoken = _spoken(pick, words)
    teaser = pick.cold_open[0] if pick.cold_open else math.inf
    later = [
        i for i in range(1, len(spoken)) if pick.start + 1e-6 < spoken[i].start <= teaser + 1e-6
    ]
    clean = [
        i
        for i in later
        if _sentence_start(spoken[i - 1], spoken[i]) and not _filler_start(spoken[i:])
    ]
    for kind, cands in (("sentence", clean), ("word", later)):
        for i in cands:
            out = _front_trimmed(pick, spoken, i)
            length = _length(out, words, signals, quiet_db, voiceless_s)
            if length <= max_s + 1e-6:
                log.info(
                    "clip %r: length %.1fs over %.0fs; trimmed from the front to a %s start "
                    "(%.2f -> %.2f, %.1fs)",
                    pick.title, full, max_s, kind, pick.start, out.start, length,
                )  # fmt: skip
                return out
    log.warning(
        "clip %r: length %.1fs over %.0fs and no later start fits; kept", pick.title, full, max_s
    )
    return pick


def fit_to_signals(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    max_s: float | None = None,
    quiet_db: float = QUIET_DB,
    voiceless_s: float | None = None,
    gameplay: bool = False,
    min_side_s: float | None = None,
) -> ClipPick:
    """`keep_payoff_side` (a clip never spans a death screen; sides shorter than `min_side_s`
    (default SIDE_MIN_S) are no clip; a clip it would reject is kept as given: `run` rejects those before
    packaging), `include_cause` (else, on `gameplay` -- a stream with a facecam --
    `include_cause_sound`), then `trim_idle`, then `avoid_static`: the clip shows what its
    reaction reacts to and neither idles on a still screen at either end nor starts or ends
    on a static screen. A start moved off `pick`'s has its opener re-checked
    (`recheck_opener`) -- not one the cause moved onto its reaction. With `max_s`, then
    `extend_payoff` and `clamp_length` (its start re-checked, forward only): the payoff scare
    just after the end is shown, and the clip lasts at most `max_s` (its interior edit gaps,
    voiceless stretches over `voiceless_s` -- default VOICELESS_GAME_S on `gameplay`, else
    VOICELESS_MAX_S -- not counted)."""
    if voiceless_s is None:
        voiceless_s = VOICELESS_GAME_S if gameplay else VOICELESS_MAX_S
    side = keep_payoff_side(pick, words, signals, min_s=min_side_s or SIDE_MIN_S) or pick
    caused = include_cause(side, words, signals)
    if caused is side and gameplay:
        caused = include_cause_sound(side, words, signals)
    out = trim_idle(caused, words, signals, quiet_db)
    out = avoid_static(out, words, signals)
    on_cause = caused is not side and abs(out.start - caused.start) < 1e-6
    if abs(out.start - pick.start) > 1e-6 and not on_cause:
        out = recheck_opener(out, words, signals)
    if max_s is not None:
        out = extend_payoff(out, words, signals, max_s, quiet_db, voiceless_s)
        clamped = clamp_length(out, side, words, signals, max_s, quiet_db, voiceless_s)
        if abs(clamped.start - out.start) > 1e-6:
            clamped = recheck_opener(clamped, words, signals, back=False)
        out = clamped
    return out


# --- stream editing: static screens, voiceless stretches, the payoff after the end ---------

STATIC_WORD_PAD_S = 0.05  # a static cut starts / ends this far clear of a word it would split
STATIC_MIN_GAP_S = 0.25  # a static cut shorter than this is not worth a jump cut
VOICELESS_MAX_S = 6.0  # a stretch without speech longer than this inside a clip is trimmed:
# each scare in it keeps [scare - VOICELESS_SCARE_PRE_S, scare + VOICELESS_SCARE_POST_S]
# (never cut); of the rest, at most VOICELESS_CAP_S is kept: VOICELESS_LEAD_S after the last
# word, then up to VOICELESS_KEEP_S of build-up before the next speech (or the stretch's
# closing scare), then the remainder before each scare's window, latest first. Voice is the
# words as captioned: held ends clamped to the voice (`render.voiced_words`), like Q-voiceless.
VOICELESS_KEEP_S = 3.2
VOICELESS_LEAD_S = 0.3
VOICELESS_CAP_S = 4.0
VOICELESS_SCARE_PRE_S = 0.5
VOICELESS_SCARE_POST_S = 1.5
# ... on gameplay (a stream with a facecam: `run_job`) the limit is VOICELESS_GAME_S, and the
# kept build-up at most that long too; talk sources keep VOICELESS_MAX_S
VOICELESS_GAME_S = 3.0
# A scare this soon after a clip's end is its payoff (the clip ends just before it): the end
# moves to PAYOFF_REACT_S after it, or to the end of a sentence said by then (at most
# PAYOFF_SAID_MAX_S after it), never into a static screen. 8.5, not 8: measured from the
# transcript's last word end, which runs late on a held word.
PAYOFF_SCARE_S = 8.5
PAYOFF_REACT_S = 2.5
PAYOFF_SAID_MAX_S = 6.0
# A clip already ending on a punchline -- a finished sentence ("...make him run?") -- is
# extended only to a scare at most PAYOFF_PUNCHLINE_S after its last spoken word, and never when
# that would add a new spoken sentence; a shout (an exclamation of at most PAYOFF_SHOUT_WORDS
# words: "No!" mid-chase) is no punchline and keeps the full payoff window.
PAYOFF_PUNCHLINE_S = 1.5
PAYOFF_SHOUT_WORDS = 4
# The catch: an extension keeps at least PAYOFF_CATCH_S after the scare (the reaction), or is
# not made. When a death screen follows the scare, a voiced reaction (a word within
# PAYOFF_CATCH_S of it) still gets PAYOFF_CATCH_S, and runs a whole cover (layout.COVER_MIN_S)
# into the screen: the cam shows the reaction over it. Without one the clip ends before the
# screen, never less than PAYOFF_MIN_CATCH_S after the scare (else no extension): a death
# screen too short to cover is never shown.
PAYOFF_CATCH_S = 1.5
PAYOFF_MIN_CATCH_S = 0.2
# A screen run starts somewhere in its first flagged 4 fps frame: an end stays this far clear.
STATIC_EDGE_S = 0.25


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def static_gaps(
    pick: ClipPick, words: list[Word], signals: Signals, quiet: bool = False
) -> list[tuple[float, float]]:
    """Static screens (`signals.static_runs`) inside the clip's main span, to cut (source s):
    a word the screen's edge would split is kept whole (the cut starts / ends
    STATIC_WORD_PAD_S clear of it); words wholly on the screen go with it. Logs each (unless
    `quiet`). A screen the clip ends on right after its closing scare or with the streamer
    reacting over it (`extend_payoff`'s catch) is not cut: the cam shows the reaction."""
    out = []
    scares = scare_times(signals)
    for lo, hi in static_runs(signals, words):
        if hi >= pick.end - 1e-6 and _closing_catch(lo, pick.end, words, scares):
            continue
        a, b = max(lo, pick.start), min(hi, pick.end)
        for w in words:
            if w.start < a < w.end:
                a = w.end + STATIC_WORD_PAD_S
            if w.start < b < w.end:
                b = w.start - STATIC_WORD_PAD_S
        if b - a < STATIC_MIN_GAP_S:
            continue
        if any(a < y and x < b for x, y in pick.drops):
            continue
        out.append((round(a, 3), round(b, 3)))
        if not quiet:
            log.info(
                "clip %r: static screen %.2f-%.2fs (death / loading screen or fade); cut",
                pick.title, a, b,
            )  # fmt: skip
    return out


def _closing_catch(lo: float, end: float, words: list[Word], scares: list[float]) -> bool:
    """A screen starting at `lo` that a clip ending at `end` closes on is the catch's tail: a
    scare within PAYOFF_CATCH_S before it, or words said over it before the end."""
    return any(lo - PAYOFF_CATCH_S <= t <= lo + 1e-6 for t in scares) or any(
        lo - 1e-6 <= w.start < end for w in words
    )


def avoid_static(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """A clip never starts or ends on a static screen or fade (`signals.static_runs`): a start
    on one moves to the first sentence start after it (its opening becomes `first_words`); an
    end on one moves STATIC_EDGE_S before the screen's start -- or, with a word running into
    the screen, to min(the word's end, the screen's start): pass voice-clamped words
    (`render.voiced_words`), a DTW-held end is not voice; `last_words` follows. An end after
    words said over the screen is the reaction the side before a death screen keeps
    (`keep_payoff_side`, `_closing_catch`): left alone, the cam covers the screen. Logs."""
    out = pick
    for lo, hi in static_runs(signals, words):
        if lo - 1e-6 <= out.start < hi:
            spoken = _spoken(out, words)
            later = [
                i
                for i, w in enumerate(spoken)
                if w.start >= hi - 1e-6 and _sentence_start(spoken[i - 1] if i else None, w)
            ]
            if later:
                new = _front_trimmed(out, spoken, later[0])
                log.info(
                    "clip %r: starts on a static screen / fade %.2f-%.2fs; start %.2f -> %.2f",
                    pick.title, lo, hi, out.start, new.start,
                )  # fmt: skip
                out = new
        over = [w.end for w in words if lo - 1e-6 <= w.start < out.end]
        reacting = bool(over) and out.end - max(over) <= PAYOFF_CATCH_S + 1e-6
        if lo < out.end <= hi + 1e-6 and lo > out.start and not reacting:
            end = lo - STATIC_EDGE_S
            for w in words:
                if w.start < lo - STATIC_EDGE_S < w.end:
                    end = min(w.end, lo)
            log.info(
                "clip %r: ends on a static screen %.2f-%.2fs; end %.2f -> %.2f",
                pick.title, lo, hi, out.end, end,
            )  # fmt: skip
            out = _ended(out, words, round(end, 3))
    return out


def _ended(pick: ClipPick, words: list[Word], end: float) -> ClipPick:
    """`pick` ending at `end`, its last sentence there as `last_words` (unchanged when no word
    is left), its drops past the end gone."""
    update: dict = {"end": end, "drops": [d for d in pick.drops if d[1] <= end]}
    kept = [w for w in _spoken(pick, words) if w.start < end - 1e-6]
    if kept:
        update["last_words"] = _sentence_text(_last_sentence(kept))
    return pick.model_copy(update=update)


def _voiceless_keep(
    a: float,
    b: float,
    scares: list[float],
    screens: list[tuple[float, float]] | None = None,
    cap: float = VOICELESS_CAP_S,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(the scare windows, the kept build-up spans) of the voiceless stretch [a, b]. A scare
    inside a screen run (`signals.screen_runs`: a death / loading screen) has no window, and a
    window stops STATIC_EDGE_S into the next screen run: a jumpscare's death screen is not
    held as its reaction. At most `cap` s of build-up is kept."""
    screens = screens or []

    def window(t: float) -> tuple[float, float]:
        nxt = next((lo for lo, _ in screens if lo > t), math.inf)
        return max(t - VOICELESS_SCARE_PRE_S, a), min(
            t + VOICELESS_SCARE_POST_S, b, nxt + STATIC_EDGE_S
        )

    windows = _merge(
        [window(t) for t in scares if a < t <= b and not any(lo <= t < hi for lo, hi in screens)]
    )
    free, cur = [], a
    for x, y in windows:
        if x > cur:
            free.append((cur, x))
        cur = max(cur, y)
    if b > cur:
        free.append((cur, b))
    kept: list[tuple[float, float]] = []
    budget = cap
    if free and free[0][0] == a:  # a moment after the last word
        lead = min(VOICELESS_LEAD_S, free[0][1] - a)
        kept.append((a, a + lead))
        budget -= lead
    for i, (x, y) in enumerate(reversed(free)):  # build-ups, the next speech's first
        floor = kept[0][1] if kept and kept[0][0] == x else x
        take = min(y - floor, budget, VOICELESS_KEEP_S if i == 0 else budget)
        if take > 0:
            kept.append((y - take, y))
            budget -= take
    return windows, _merge(kept)


def voiceless_gaps(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    quiet: bool = False,
    limit: float = VOICELESS_MAX_S,
) -> list[tuple[float, float]]:
    """Voiceless stretches inside the clip to cut (source s): between two of its words (or
    from its last word to the last scare before its end) longer than VOICELESS_MAX_S, each
    scare's window kept (`scare_times`; a scare is never cut) and at most VOICELESS_CAP_S of
    the rest (VOICELESS_LEAD_S after the word, VOICELESS_KEEP_S before the next speech, the
    remainder before the scares: the build-up). Voice is `words` as given: pass them
    voice-clamped (`render.voiced_words`, the captions' measure), a game bed is not voice.
    Logs each (unless `quiet`). `limit` replaces VOICELESS_MAX_S (gameplay:
    VOICELESS_GAME_S), the kept build-up then at most `limit` too."""
    spoken = _spoken(pick, words)
    if not spoken:
        return []
    scares = scare_times(signals)
    screens = screen_runs(signals)
    stretches = [(w.end, nxt.start) for w, nxt in pairwise(spoken)]
    last = max(w.end for w in spoken)
    tail = [t for t in scares if last < t < pick.end]
    if tail:
        stretches.append((last, tail[-1]))
    out = []
    for a, b in stretches:
        if b - a <= limit:
            continue
        windows, kept = _voiceless_keep(a, b, scares, screens, cap=min(VOICELESS_CAP_S, limit))
        held = _merge([*windows, *kept])
        cur, cuts = a, []
        for x, y in held:
            if x - cur >= STATIC_MIN_GAP_S:
                cuts.append((round(cur, 3), round(x, 3)))
            cur = max(cur, y)
        if b - cur >= STATIC_MIN_GAP_S:
            cuts.append((round(cur, 3), round(b, 3)))
        out += cuts
        if cuts and not quiet:
            log.info(
                "clip %r: voiceless %.1f-%.1fs (no speech); %.1f s cut, %d scare(s) kept",
                pick.title, a, b, sum(y - x for x, y in cuts), len(windows),
            )  # fmt: skip
    return out


def edit_gaps(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    quiet_db: float = QUIET_DB,
    quiet: bool = False,
    voiceless_s: float = VOICELESS_MAX_S,
) -> list[tuple[float, float]]:
    """Every interior cut the signals ask for (source s, sorted, merged): `idle_gaps`,
    `static_gaps` and `voiceless_gaps` (stretches over `voiceless_s`)."""
    return _merge(
        [
            *idle_gaps(pick, words, signals, quiet_db, quiet=quiet),
            *static_gaps(pick, words, signals, quiet=quiet),
            *voiceless_gaps(pick, words, signals, quiet=quiet, limit=voiceless_s),
        ]
    )


def _off_words(target: float, words: list[Word], screen: float | None, earlier: bool) -> float:
    """`target` moved out of a word it splits: to the word's end (`earlier`: its start), or its
    start when its end runs into the `screen`."""
    for w in words:
        if w.start < target < w.end:
            return w.start if earlier or (screen is not None and w.end > screen) else w.end
    return target


def extend_payoff(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    max_s: float,
    quiet_db: float = QUIET_DB,
    voiceless_s: float | None = None,
) -> ClipPick:
    """A clip ending just before its payoff: the first scare (`scare_times`; none on a screen
    already running, `screen_runs`: a stinger over the death screen is no payoff) within
    PAYOFF_SCARE_S after its end, with no static screen starting between (from the end on),
    moves the end to PAYOFF_REACT_S after the scare -- or to the end of the sentence said in
    that time (at most PAYOFF_SAID_MAX_S after the scare) -- never mid-word. A death screen
    after the scare (`screen_runs` of at least STATIC_MIN_S): with the streamer reacting (a
    word within PAYOFF_CATCH_S of the scare) the end runs on over it at least PAYOFF_CATCH_S
    after the scare and COVER_MIN_S + STATIC_EDGE_S into the screen (the render covers the
    screen with the cam); without one the clip ends STATIC_EDGE_S before the screen (at the
    latest on its first flagged frame) -- not extended when that leaves less than
    PAYOFF_MIN_CATCH_S after the scare. A clip already ending on a punchline
    (`_ends_on_punchline`) is extended only to a scare at most PAYOFF_PUNCHLINE_S after its
    last spoken word, and never when the extension adds a new spoken sentence. Shortened in
    0.5 s steps (each moved off a word, earlier) to fit `max_s` (counted without
    `edit_gaps`, voiceless stretches over `voiceless_s`); not made when even the catch does
    not fit. Logs."""
    screens = screen_runs(signals)
    scares = [t for t in scare_times(signals) if not any(lo <= t < hi for lo, hi in screens)]
    scare = next((t for t in scares if pick.end < t <= pick.end + PAYOFF_SCARE_S), None)
    if scare is None:
        return pick
    spoken = _spoken(pick, words)
    last = max((w.end for w in spoken), default=pick.end)
    punchline = _ends_on_punchline(spoken)
    if punchline and scare - last > PAYOFF_PUNCHLINE_S:
        log.info(
            "clip %r: ends on a punchline; payoff scare at %.2fs (%.1f s after its last word) "
            "not added", pick.title, scare, scare - last,
        )  # fmt: skip
        return pick
    runs = static_runs(signals, words)
    if any(pick.end - 1e-6 <= lo < scare for lo, _ in runs):
        return pick
    target = scare + PAYOFF_REACT_S
    said = sorted((w for w in words if scare <= w.start < target), key=lambda w: w.start)
    if said:
        later = sorted((w for w in words if w.start >= said[-1].start), key=lambda w: w.start)
        end_w = next((w for w in later if _ends_sentence(w)), later[-1])
        target = min(max(target, end_w.end), scare + PAYOFF_SAID_MAX_S)
    deaths = [lo for lo, _ in screen_runs(signals, min_s=STATIC_MIN_S)]
    screen = min((lo for lo in [*deaths, *(lo for lo, _ in runs)] if lo >= scare), default=None)
    floor = scare + PAYOFF_CATCH_S
    if screen is not None and screen < target:
        if any(scare - CAUSE_REACT_S <= w.start <= scare + PAYOFF_CATCH_S for w in words):
            # the streamer's reaction, on the cam over the screen: a whole cover of it
            floor = max(floor, screen + COVER_MIN_S + STATIC_EDGE_S)
            target = max(target, floor)
            screen = None
        else:
            target = min(target, screen - STATIC_EDGE_S)
            if target < scare + PAYOFF_MIN_CATCH_S:
                target = min(screen, scare + PAYOFF_REACT_S)  # its first flagged frame
            if target < scare + PAYOFF_MIN_CATCH_S - 1e-6:
                log.info(
                    "clip %r: payoff scare at %.2fs runs straight into a death screen; not "
                    "extended", pick.title, scare,
                )  # fmt: skip
                return pick
            floor = target
    target = round(_off_words(target, words, screen, earlier=False), 3)
    if punchline and any(pick.end < w.start < target for w in words):
        log.info(
            "clip %r: ends on a punchline; the payoff scare at %.2fs would add a new sentence; "
            "not added", pick.title, scare,
        )  # fmt: skip
        return pick
    while target >= floor - 1e-6:
        out = pick.model_copy(update={"end": target})
        if _length(out, words, signals, quiet_db, voiceless_s) <= max_s + 1e-6:
            log.info(
                "clip %r: payoff scare at %.2fs just after the end; end %.2f -> %.2f",
                pick.title, scare, pick.end, target,
            )  # fmt: skip
            return out
        target = round(_off_words(target - 0.5, words, screen, earlier=True), 3)
    log.info(
        "clip %r: payoff scare at %.2fs and its catch after the end do not fit %.0fs; not "
        "extended", pick.title, scare, max_s,
    )  # fmt: skip
    return pick


# --- death screens inside a clip, reaction openers, openers after a start move ------------------

# A screen run (`signals.screen_runs`, talked over or not) at least this long wholly inside a
# clip is a death / respawn: the clip keeps one side of it. A side shorter than SIDE_MIN_S
# (from its first word, or the clip's start, to its end) is no clip.
DEATH_SCREEN_MIN_S = SCREEN_MARKER_MIN_S  # the selector's screen markers: the same deaths
SIDE_MIN_S = 10.0
# A reaction said over the death screen (its first word within this of the screen's start,
# each next word within this of the one before) stays with the side before it.
SIDE_REACT_GAP_S = 1.5
SIDE_PAYOFF_SCORE = 2.0  # the side holding the clip's end holds its declared payoff ...
SIDE_LOUD_SCORE = 0.1  # ... each scare scores 1, each loud second this


def _sentence_text(ws: list[Word]) -> str:
    return " ".join(w.text for w in ws)


def _last_sentence(spoken: list[Word]) -> list[Word]:
    """The last sentence of `spoken` (in order)."""
    i = len(spoken) - 1
    while i > 0 and not _sentence_start(spoken[i - 1], spoken[i]):
        i -= 1
    return spoken[i:]


def _first_sentence(spoken: list[Word], i: int = 0) -> list[Word]:
    """The sentence of `spoken` (in order) starting at index `i`."""
    end = next((j for j in range(i, len(spoken)) if _ends_sentence(spoken[j])), len(spoken) - 1)
    return spoken[i : end + 1]


def _sides(
    pick: ClipPick, spoken: list[Word], deaths: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """The clip's spans between its death screens: a side before a screen keeps the reaction
    said over it (SIDE_REACT_GAP_S), a side after one starts where the screen ends."""
    sides: list[tuple[float, float]] = []
    cur = pick.start
    for lo, hi in deaths:
        end = lo
        for w in (w for w in spoken if lo - 1e-6 <= w.start < hi):
            if w.start - end > SIDE_REACT_GAP_S + 1e-6:
                break
            end = w.end
        sides.append((cur, end))
        cur = hi
    sides.append((cur, pick.end))
    return sides


_BEAT_TIME_RE = re.compile(r"@\s*(-?\d+(?:\.\d+)?)\s*s")


def _beats_within(beats: list[str], start: float, end: float) -> list[str]:
    """The `edit_beats` ("zoom: 'word' @ 52.3s") timed inside [start, end] (untimed kept)."""
    out = []
    for b in beats:
        m = _BEAT_TIME_RE.search(b)
        if m is None or start - 1e-6 <= float(m.group(1)) <= end + 1e-6:
            out.append(b)
    return out


def keep_payoff_side(
    pick: ClipPick, words: list[Word], signals: Signals, min_s: float = SIDE_MIN_S
) -> ClipPick | None:
    """A clip spanning a death / respawn screen (`screen_runs` of at least DEATH_SCREEN_MIN_S
    wholly inside it) splices two attempts: keep only the side (between screens) holding the
    payoff -- scored by its scares (1 each, none on a screen), loud seconds (SIDE_LOUD_SCORE
    each) and, for the side holding the clip's end and its last words, SIDE_PAYOFF_SCORE; a
    tie goes to the later side -- among the sides lasting at least `min_s` (`run` passes
    max(SIDE_MIN_S, min_clip_s)) with words. A side after a screen starts on its first
    sentence start (a sentence begun over the screen is skipped; `first_words` follows,
    `declared_first_words` is cleared) and has its opener re-checked (`recheck_opener`,
    forward: no reply opener); a side before one ends with the reaction said over it
    (`last_words` follows), at least COVER_MIN_S + STATIC_EDGE_S into the screen (a whole
    cam cover of it); a cold open outside the kept side is dropped, the opener guard's move
    (`opener_moved`) cleared and the `edit_beats` outside it dropped. None (logged) when no
    side is viable: the clip is rejected. `pick` itself when it spans no death screen. Logs."""
    runs = screen_runs(signals)
    deaths = [
        (lo, hi)
        for lo, hi in runs
        if hi - lo >= DEATH_SCREEN_MIN_S - 1e-6 and pick.start < lo and hi < pick.end
    ]
    if not deaths:
        return pick
    spoken = _spoken(pick, words)
    scares = [t for t in scare_times(signals) if not any(lo <= t < hi for lo, hi in runs)]
    loud = loud_seconds(signals.loudness)
    sides = _sides(pick, spoken, deaths)
    best: tuple[float, float, int] | None = None
    for k, (a, b) in enumerate(sides):
        ws = [w for w in spoken if a - 1e-6 <= w.start < b]
        if k and ws:  # after a screen: from its first sentence start
            j = spoken.index(ws[0])
            prev = spoken[j - 1] if j else None
            i = next(
                (i for i, w in enumerate(ws) if _sentence_start(ws[i - 1] if i else prev, w)),
                None,
            )
            ws = ws[i:] if i is not None else []
        begin = ws[0].start if k and ws else a
        if not ws or b - begin < min_s - 1e-6:
            continue
        score = sum(1.0 for t in scares if begin <= t < b)
        score += SIDE_LOUD_SCORE * sum(1 for sec in loud if begin <= sec < b)
        score += SIDE_PAYOFF_SCORE if b >= pick.end - 1e-6 else 0.0
        if best is None or score >= best[0]:
            best = (score, begin, k)
    where = ", ".join(f"{lo:.1f}-{hi:.1f}s" for lo, hi in deaths)
    if best is None:
        log.warning(
            "clip %r: spans a death screen (%s) and no side lasts %.0fs with speech; rejected",
            pick.title, where, min_s,
        )  # fmt: skip
        return None
    _, begin, k = best
    end = sides[k][1]
    if k < len(deaths) and end > deaths[k][0] + 1e-6:  # a reaction over the screen
        end = max(end, min(deaths[k][0] + COVER_MIN_S + STATIC_EDGE_S, deaths[k][1]))
    out = pick.model_copy(update={"opener_moved": None})
    if k:
        out = _front_trimmed(out, spoken, next(i for i, w in enumerate(spoken) if w.start == begin))
    if end < pick.end - 1e-6:
        kept = [w for w in spoken if out.start - 1e-6 <= w.start < end]
        update = {"end": round(end, 3), "drops": [d for d in out.drops if d[1] <= end]}
        out = out.model_copy(update={**update, "last_words": _sentence_text(_last_sentence(kept))})
    out = out.model_copy(update={"edit_beats": _beats_within(out.edit_beats, out.start, out.end)})
    teaser = out.cold_open
    if teaser and not (out.start - 1e-6 <= teaser[0] and teaser[1] <= out.end + 1e-6):
        out = out.model_copy(update={"cold_open": None})
    log.info(
        "clip %r: spans a death screen (%s); kept the side %.2f-%.2fs holding the payoff",
        pick.title, where, out.start, out.end,
    )  # fmt: skip
    if k:
        out = recheck_opener(out, words, signals, back=False)
    return out


# A clip opening on a reaction phrase with no scare marker reacts to a sound the viewer must
# hear: its start backs off to the latest audio onset (`Signals.onset_times`) from
# CAUSE_LOOKBACK_S to CAUSE_SOUND_MIN_S before the first word, CAUSE_SOUND_LEAD_S before it.
REACTION_OPENERS = ("what was that", "the fuck", "oh my god", "what the", "what is that")
REACTION_OPENERS += ("the hell",)
CAUSE_SOUND_MIN_S = 0.3  # an onset this close to the first word is the word itself
CAUSE_SOUND_LEAD_S = 0.1


def _reaction_opener(ws: list[Word]) -> bool:
    return " ".join(_norm(w) for w in ws[:4]).startswith(REACTION_OPENERS)


def include_cause_sound(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """A clip opening on a reaction ("What was that?", "The fuck...", "Oh my god") with no
    scare within CAUSE_LOOKBACK_S before it (`include_cause` handles those) starts
    CAUSE_SOUND_LEAD_S before the latest audio onset from CAUSE_LOOKBACK_S to
    CAUSE_SOUND_MIN_S before its first word: the cause sound. Words said from there on are
    added only as whole sentences (else nothing changes) and prepended to `first_words` and
    `declared_first_words`. Skipped when the opener guard moved the start. Logs."""
    if pick.opener_moved is not None or not signals.onset_times:
        return pick
    spoken = _spoken(pick, words)
    if not spoken or not _reaction_opener(_first_sentence(spoken)):
        return pick
    first = spoken[0].start
    if any(first - CAUSE_LOOKBACK_S < t <= first + CAUSE_REACT_S for t in scare_times(signals)):
        return pick
    lo, hi = first - CAUSE_LOOKBACK_S, first - CAUSE_SOUND_MIN_S
    onsets = [t for t in signals.onset_times if lo - 1e-6 <= t <= hi + 1e-6]
    if not onsets:
        return pick
    onset = max(onsets)
    start = onset - CAUSE_SOUND_LEAD_S
    ordered = sorted(words, key=lambda w: w.start)
    added = [w for w in ordered if w.end > start and w.start < first - 1e-6]
    if added:
        k = ordered.index(added[0])
        prev = ordered[k - 1] if k else None
        if not _sentence_start(prev, added[0]) or not _ends_sentence(added[-1]):
            return pick
        start = min(start, added[0].start)
    log.info(
        "clip %r: opens on a reaction (%s); start %.2f -> %.2f, onto its cause sound at %.2fs",
        pick.title, _sentence_text(_first_sentence(spoken)), pick.start, start, onset,
    )  # fmt: skip
    update: dict = {"start": round(start, 3)}
    if added:
        for key in ("first_words", "declared_first_words"):
            if getattr(pick, key):
                update[key] = f"{_sentence_text(added)} {getattr(pick, key)}"
    return pick.model_copy(update=update)


# A clip whose start moved (static avoidance, cause inclusion, the length clamp) never opens on a
# reply or correction ("No, game time!") or a context opener: it moves back to the turn the
# reply answers when that starts within OPENER_BACK_S, else forward to the next good sentence
# start within OPENER_FORWARD_S that leaves OPENER_MIN_S of clip.
REPLY_WORDS = ("no", "nope", "yeah", "yes", "okay", "ok", "right")
OPENER_BACK_S = 3.0
OPENER_FORWARD_S = 12.0
OPENER_MIN_S = 10.0


def _weak_opening(ws: list[Word]) -> bool:
    from hookline.pipeline.select import _bad_opener  # select imports this module

    if not ws:
        return False
    if _norm(ws[0]) in REPLY_WORDS and ws[0].text.rstrip().endswith(","):
        return True
    return _bad_opener(ws)


def recheck_opener(
    pick: ClipPick, words: list[Word], signals: Signals, back: bool = True
) -> ClipPick:
    """`pick` opening on a reply / correction ("No,", "Yeah,", "Okay," with its comma) or a
    context opener (`select._bad_opener`) moved: back onto the sentence before it (the turn it
    answers) when that starts within OPENER_BACK_S and off a static screen (with `back`),
    else forward to the next sentence start within OPENER_FORWARD_S that opens cleanly and
    leaves OPENER_MIN_S of clip. Kept (logged) when neither exists. An opening said from
    CAUSE_REACT_S before a scare to PAYOFF_CATCH_S after it ("No, no, no, no!") is the
    reaction to it, never a reply: kept. Logs each move."""
    spoken = _spoken(pick, words)
    if not spoken:
        return pick
    opening = _first_sentence(spoken)
    if not _weak_opening(opening):
        return pick
    first = spoken[0].start
    if any(t - CAUSE_REACT_S <= first <= t + PAYOFF_CATCH_S for t in scare_times(signals)):
        return pick
    text = _sentence_text(opening)
    before = sorted((w for w in words if w.start < spoken[0].start - 1e-6), key=lambda w: w.start)
    prev = _last_sentence(before) if back and before else []
    if (
        prev
        and _ends_sentence(prev[-1])
        and spoken[0].start - prev[0].start <= OPENER_BACK_S + 1e-6
        and not any(lo - 1e-6 <= prev[0].start < hi for lo, hi in static_runs(signals, words))
        and not _weak_opening(prev)
    ):
        said = _sentence_text(prev)
        log.info("clip %r: opens on a reply (%s); start moved back onto %r", pick.title, text, said)
        update: dict = {"start": round(prev[0].start, 3)}
        for key in ("first_words", "declared_first_words"):
            if getattr(pick, key):
                update[key] = f"{said} {getattr(pick, key)}"
        return pick.model_copy(update=update)
    for i in range(len(opening), len(spoken)):
        w = spoken[i]
        if w.start - spoken[0].start > OPENER_FORWARD_S + 1e-6 or pick.end - w.start < OPENER_MIN_S:
            break
        if _sentence_start(spoken[i - 1], w) and not _weak_opening(_first_sentence(spoken, i)):
            log.info(
                "clip %r: opens on a reply or context opener (%s); start %.2f -> %.2f (%r)",
                pick.title, text, pick.start, w.start, _sentence_text(_first_sentence(spoken, i)),
            )  # fmt: skip
            return _front_trimmed(pick, spoken, i)
    log.warning("clip %r: opens on a reply or context opener (%s); kept", pick.title, text)
    return pick


def cover_first_word(pick: ClipPick, raw: list[Word], near: list[Word]) -> ClipPick:
    """`pick` starting no later than its first word as refined against the audio: `near`
    (`raw` with onsets snapped, word for word) may move a word starting at the clip's start
    earlier than it, and the clip would lose it ("The" of "The fuck was that?"). A cold open
    is covered the same way: the teaser is the clip's very first audio ("-talism" for
    "capitalism" when the onset moved before the teaser's start)."""

    def covered(start: float, end: float) -> float:
        starts = [n.start for r, n in zip(raw, near, strict=True) if start - 1e-6 <= r.start < end]
        if not starts or min(starts) >= start - 1e-6:
            return start
        return round(min(starts), 3)

    update: dict = {}
    if (start := covered(pick.start, pick.end)) != pick.start:
        update["start"] = start
    if pick.cold_open is not None:
        a, b = pick.cold_open
        if (first := covered(a, b)) != a:
            update["cold_open"] = (first, b)
    return pick.model_copy(update=update) if update else pick
