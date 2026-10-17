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
_PUNCTUATION = ",.!?…\"'"


def _norm(word: Word) -> str:
    return word.text.lower().strip().strip(_PUNCTUATION)


def _trim_fillers(start: float, end: float, clip_words: list[Word]) -> float:
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
) -> ClipPick:
    """Trim up to 3 leading filler words, then snap start/end into silences within `window`.

    Never moves `start` past the first remaining word's start, nor `end` before the last
    word's end. Pure: no I/O."""
    start, end = pick.start, pick.end
    clip_words = [w for w in words if start - 1e-6 <= w.start < end]
    start = _trim_fillers(start, end, clip_words)

    first_start = next((w.start for w in clip_words if w.start >= start - 1e-6), None)
    last_end = max((w.end for w in clip_words), default=None)

    for s_start, s_end in silences:
        if s_start < start + window and s_end > start - window:
            start = s_end - _SNAP_MARGIN_S
            break
    for s_start, s_end in silences:
        if s_start < end + window and s_end > end - window:
            end = s_start + _SNAP_MARGIN_S
            break

    if first_start is not None:
        start = min(start, first_start)
    if last_end is not None:
        end = max(end, last_end)
    return pick.model_copy(update={"start": start, "end": end})
