"""Word-list hygiene shared by every transcription backend."""

import re

from hookline.models import Word

_MIN_RUN, _MAX_RUN = 2, 12
_NEAR_START_S = 0.3
_SAME_START_S = 0.01
_MIN_WORD_S = 0.08


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def _is_repeat(prev: list[Word], run: list[Word]) -> bool:
    if [_norm(w.text) for w in prev] != [_norm(w.text) for w in run]:
        return False
    if not any(_norm(w.text) for w in run):
        return False
    return abs(run[0].start - prev[0].start) <= _NEAR_START_S or run[0].start < prev[-1].end


def dedupe_segments(words: list[Word]) -> list[Word]:
    """Drop Whisper repeat-hallucinations: a 2-12 word run repeating the preceding run
    at (nearly) the same time. Repeats until stable, so x3 collapses to x1."""
    out: list[Word] = []
    i = 0
    while i < len(words):
        for n in range(min(_MAX_RUN, len(out), len(words) - i), _MIN_RUN - 1, -1):
            if _is_repeat(out[-n:], words[i : i + n]):
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


def spread_collapsed(words: list[Word]) -> list[Word]:
    """Spread runs of words sharing a start evenly up to the next distinct start."""
    out = list(words)
    i = 0
    while i < len(out):
        s = out[i].start
        j = i + 1
        while j < len(out) and abs(out[j].start - s) < _SAME_START_S:
            j += 1
        k = j - i
        if k > 1 and j < len(out):
            n = out[j].start
            if n - s >= _MIN_WORD_S * k:
                step = (n - s) / k
                for m in range(k):
                    start = s + m * step
                    if m < k - 1:
                        end = start + step
                    else:
                        end = min(max(out[i + m].end, start + _MIN_WORD_S), n)
                    out[i + m] = out[i + m].model_copy(update={"start": start, "end": end})
        i = j
    return out
