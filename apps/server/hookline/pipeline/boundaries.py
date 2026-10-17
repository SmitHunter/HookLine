"""Clip boundary refinement: trim leading fillers, then snap cuts into nearby silences."""

from hookline.models import ClipPick, Word

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
