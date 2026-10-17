"""Audience/audio reaction signals and the heat markers shown to Claude in the transcript."""

from statistics import median

from pydantic import BaseModel

_CHAT_WINDOW_S = 10
_CHAT_BASELINE_S = 300
_CHAT_RATIO = 2.5
_CHAT_MIN_BASELINE = 1.0
_LOUD_OVER_MEDIAN_DB = 9.0
_LOUD_FLOOR_DB = -30.0


class Signals(BaseModel):
    duration: float
    loudness: list[float]  # dB per second
    chat: list[float] | None = None  # weighted messages per second


def _chat_markers(chat: list[float]) -> dict[int, str]:
    n = len(chat)
    prefix = [0.0]
    for v in chat:
        prefix.append(prefix[-1] + v)
    before = _CHAT_WINDOW_S // 2
    # 10 s rolling sum centred on each second, truncated at the array ends.
    sums = [
        prefix[min(n, i + _CHAT_WINDOW_S - before)] - prefix[max(0, i - before)] for i in range(n)
    ]
    half = _CHAT_BASELINE_S // 2
    out: dict[int, str] = {}
    prev_hot = False
    for i, s in enumerate(sums):
        baseline = max(median(sums[max(0, i - half) : i + half + 1]), _CHAT_MIN_BASELINE)
        hot = s >= _CHAT_RATIO * baseline
        if hot and not prev_hot:
            out[i] = f"🔥chat×{round(s / baseline)}"
        prev_hot = hot
    return out


def _loud_markers(loudness: list[float]) -> dict[int, str]:
    if not loudness:
        return {}
    threshold = median(loudness) + _LOUD_OVER_MEDIAN_DB
    out: dict[int, str] = {}
    prev_hot = False
    for i, db in enumerate(loudness):
        hot = db >= threshold and db > _LOUD_FLOOR_DB
        if hot and not prev_hot:
            out[i] = "🔊loud"
        prev_hot = hot
    return out


def heat_markers(signals: Signals) -> dict[int, list[str]]:
    """Second -> markers, only at the first second of each run of heat."""
    found = [_chat_markers(signals.chat)] if signals.chat else []
    found.append(_loud_markers(signals.loudness))
    markers: dict[int, list[str]] = {}
    for source in found:
        for sec, label in source.items():
            markers.setdefault(sec, []).append(label)
    return dict(sorted(markers.items()))
