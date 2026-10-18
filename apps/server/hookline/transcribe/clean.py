"""Word-list hygiene shared by every transcription backend."""

import re

from hookline.models import Word

_MIN_RUN, _MAX_RUN = 2, 12
_NEAR_START_S = 0.1
_COLLAPSED_S = 0.02
_MAX_STEP_S = 0.6
_SAME_START_S = 0.01
_MIN_WORD_S = 0.08


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def _is_repeat(prev: list[Word], run: list[Word]) -> bool:
    if [_norm(w.text) for w in prev] != [_norm(w.text) for w in run]:
        return False
    if not any(_norm(w.text) for w in run):
        return False
    if all(abs(a.start - b.start) <= _NEAR_START_S for a, b in zip(prev, run, strict=True)):
        return True
    # collapsed DTW tail: every word of the copy starts at one instant (impossible for speech)
    starts = [w.start for w in run]
    return max(starts) - min(starts) <= _COLLAPSED_S


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


def spread_collapsed(words: list[Word], end_limit: float | None = None) -> list[Word]:
    """Spread runs of words sharing a start evenly up to the next distinct start.

    A trailing run uses `end_limit` (e.g. the transcript duration) as the next start.
    Each step is capped at 0.6 s so a run before a long pause does not stretch across it."""
    out = list(words)
    i = 0
    while i < len(out):
        s = out[i].start
        j = i + 1
        while j < len(out) and abs(out[j].start - s) < _SAME_START_S:
            j += 1
        k = j - i
        n = out[j].start if j < len(out) else end_limit
        if k > 1 and n is not None and n - s >= _MIN_WORD_S * k:
            step = min((n - s) / k, _MAX_STEP_S)
            for m in range(k):
                start = s + m * step
                if m < k - 1:
                    end = start + step
                else:
                    end = min(max(out[i + m].end, start + _MIN_WORD_S), n)
                out[i + m] = out[i + m].model_copy(update={"start": start, "end": end})
        i = j
    return out
