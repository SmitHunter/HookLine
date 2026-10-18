"""Clip boundary refinement: trim leading fillers, then snap cuts into nearby silences; trim
idle (no speech, still picture) stretches off a clip's ends."""

import logging
import math
from itertools import pairwise

from hookline.models import ClipPick, Word
from hookline.pipeline.signals import (
    LOW_MOTION,
    Signals,
    low_motion,
    scare_seconds,
    scare_times,
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
QUIET_DB = -32.0
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


def _length(pick: ClipPick, words: list[Word], signals: Signals, quiet_db: float) -> float:
    """The pick's on-screen length before dead-air cuts and the end tail: its segments minus
    its interior idle gaps."""
    gaps = idle_gaps(pick, words, signals, quiet_db, quiet=True)
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
) -> ClipPick:
    """The final hard length clamp before render: `pick` (fitted from `original`) lasting at
    most `max_s` (max_clip_s plus the payoff allowance), counted without its interior idle
    gaps and the end tail. Over it, the scare-cause words `include_cause` added go first (the
    original start, idle-trimmed); still over, the start moves to the earliest later sentence
    start (not a filler opener, not past the cold-open teaser) that fits, else the earliest
    later word that fits. `first_words` becomes the new opening sentence and
    `declared_first_words` is cleared (Claude's opening is no longer in the clip). Logs."""
    full = _length(pick, words, signals, quiet_db)
    if full <= max_s + 1e-6:
        return pick
    if pick.start < original.start - 1e-6:
        uncaused = trim_idle(original, words, signals, quiet_db)
        log.info(
            "clip %r: length %.1fs over %.0fs; scare-cause words dropped (start %.2f -> %.2f)",
            pick.title, full, max_s, pick.start, uncaused.start,
        )  # fmt: skip
        pick = uncaused
        full = _length(pick, words, signals, quiet_db)
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
            length = _length(out, words, signals, quiet_db)
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
) -> ClipPick:
    """`include_cause`, then `trim_idle`: the clip shows what its reaction reacts to and does
    not idle on a still screen at either end. With `max_s`, then `clamp_length`: the clip
    lasts at most `max_s` (its interior idle gaps not counted)."""
    out = trim_idle(include_cause(pick, words, signals), words, signals, quiet_db)
    if max_s is not None:
        out = clamp_length(out, pick, words, signals, max_s, quiet_db)
    return out
