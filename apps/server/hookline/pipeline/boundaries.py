"""Clip boundary refinement: trim leading fillers, then snap cuts into nearby silences; trim
idle (no speech, still picture) stretches off a clip's ends."""

import logging

from hookline.models import ClipPick, Word
from hookline.pipeline.signals import Signals, low_motion, scare_seconds

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


def trim_idle(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """Cut an idle tail or lead off the clip's main span: more than IDLE_MIN_S without speech
    whose picture is still (`signals.low_motion`: a death, loading or menu screen, a pause) and
    holds no scare. The clip then ends IDLE_TAIL_S after its last word, or starts IDLE_LEAD_S
    before its first. Words in dropped ranges are not speech. A no-op without motion signals
    (cached before them) or without words in the span. Logs each trim. Pure apart from logging."""
    if not signals.motion:
        return pick
    spoken = [
        w
        for w in words
        if pick.start - 1e-6 <= w.start < pick.end
        and not any(a <= w.start and w.end <= b for a, b in pick.drops)
    ]
    if not spoken:
        return pick
    scares = scare_seconds(signals)
    start, end = pick.start, pick.end

    def idle(a: float, b: float) -> bool:
        quiet = not any(round(a) <= s < round(b) for s in scares)
        return b - a > IDLE_MIN_S and quiet and low_motion(signals.motion, a, b)

    last = max(w.end for w in spoken)
    if idle(last, end):
        end = last + IDLE_TAIL_S
        log.info(
            "clip %r: idle tail %.1f-%.1fs (no speech, still picture); end moved to %.2f",
            pick.title, last, pick.end, end,
        )  # fmt: skip
    first = min(w.start for w in spoken)
    if idle(start, first):
        start = first - IDLE_LEAD_S
        log.info(
            "clip %r: idle lead %.1f-%.1fs (no speech, still picture); start moved to %.2f",
            pick.title, pick.start, first, start,
        )  # fmt: skip
    if (start, end) == (pick.start, pick.end):
        return pick
    return pick.model_copy(update={"start": round(start, 3), "end": round(end, 3)})


CAUSE_LOOKBACK_S = 4.0  # a scare this close before the clip's first word is what it reacts to
CAUSE_REACT_S = 0.5  # words from this long before the scare's second on are reactions to it
CAUSE_MAX_WORDS = 4  # the clip's start moves back by at most this many words ...
CAUSE_MAX_S = 6.0  # ... and at most this many seconds


def include_cause(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """Start a clip that opens just after a scare (`signals.scare_seconds`: a game-audio
    transient or a flash with a sound, within CAUSE_LOOKBACK_S before its first word) on the
    first word said after that scare instead (the "Oh."s between the sound and the reaction
    line), so the viewer hears and sees the cause. Repeats while another scare precedes the new
    first word; at most CAUSE_MAX_WORDS earlier words and CAUSE_MAX_S. Only spoken words move
    the start (a start before every word would be refined back onto the first one). The added
    words are prepended to `first_words` and `declared_first_words`, so Q6 checks the opening
    the viewer now hears. Logs."""
    scares = sorted(scare_seconds(signals))
    if not scares:
        return pick
    before = sorted((w for w in words if w.start < pick.start - 1e-6), key=lambda w: w.start)
    inside = [w for w in words if pick.start - 1e-6 <= w.start < pick.end]
    if not inside:
        return pick
    first = min(w.start for w in inside)
    added: list[Word] = []
    while True:
        cause = next((s for s in scares if s + 1 > first - CAUSE_LOOKBACK_S and s < first), None)
        if cause is None:
            break
        lo = max(cause - CAUSE_REACT_S, pick.start - CAUSE_MAX_S)
        new = [w for w in before if lo <= w.start < first]
        new = new[-(CAUSE_MAX_WORDS - len(added)) :] if len(added) < CAUSE_MAX_WORDS else []
        if not new:
            break
        added = new + added
        first = new[0].start
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


def fit_to_signals(pick: ClipPick, words: list[Word], signals: Signals) -> ClipPick:
    """`include_cause`, then `trim_idle`: the clip shows what its reaction reacts to and does
    not idle on a still screen at either end."""
    return trim_idle(include_cause(pick, words, signals), words, signals)
