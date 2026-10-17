"""Audience/audio reaction signals and the heat markers shown to Claude in the transcript."""

from statistics import median

from pydantic import BaseModel

_CHAT_WINDOW_S = 10
_CHAT_BASELINE_S = 300
_CHAT_RATIO = 2.5
_CHAT_MIN_BASELINE = 1.0
_LOUD_OVER_MEDIAN_DB = 9.0
_LOUD_FLOOR_DB = -30.0
_KIND_EMOJI = {"laugh": "😂", "hype": "🏆", "fail": "💀", "shock": "😱"}
_KIND_SHARE = 0.4
_CLIP_MIN_REQUESTS = 3
CLIP_MARKER = "\u2702\ufe0fclip"


class Signals(BaseModel):
    duration: float
    loudness: list[float]  # dB per second
    chat: list[float] | None = None  # weighted messages per second
    # second -> {kind: keyword hits}; absent in signals cached before reaction kinds existed
    chat_kinds: dict[int, dict[str, int]] | None = None


def _window_kinds(kinds: dict[int, dict[str, int]], i: int) -> dict[str, int]:
    before = _CHAT_WINDOW_S // 2
    total: dict[str, int] = {}
    for sec in range(i - before, i + _CHAT_WINDOW_S - before):
        for k, v in kinds.get(sec, {}).items():
            total[k] = total.get(k, 0) + v
    return total


def _dominant_emoji(window: dict[str, int]) -> str:
    hits = sum(window.values())
    if not hits:
        return ""
    kind, count = max(
        ((k, v) for k, v in window.items() if k in _KIND_EMOJI),
        key=lambda kv: kv[1],
        default=("", 0),
    )
    return f" {_KIND_EMOJI[kind]}" if kind and count >= _KIND_SHARE * hits else ""


def _clip_markers(kinds: dict[int, dict[str, int]], n: int) -> dict[int, str]:
    out: dict[int, str] = {}
    prev_hot = False
    for i in range(n):
        hot = _window_kinds(kinds, i).get("clip", 0) >= _CLIP_MIN_REQUESTS
        if hot and not prev_hot:
            out[i] = CLIP_MARKER
        prev_hot = hot
    return out


def _chat_hot(chat: list[float]) -> list[float | None]:
    """Per second: the spike ratio vs the local baseline when that second is hot, else None."""
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
    out: list[float | None] = []
    for i, s in enumerate(sums):
        baseline = max(median(sums[max(0, i - half) : i + half + 1]), _CHAT_MIN_BASELINE)
        out.append(s / baseline if s >= _CHAT_RATIO * baseline else None)
    return out


def _chat_markers(
    chat: list[float], kinds: dict[int, dict[str, int]] | None = None
) -> dict[int, str]:
    out: dict[int, str] = {}
    prev_hot = False
    for i, ratio in enumerate(_chat_hot(chat)):
        if ratio is not None and not prev_hot:
            emoji = _dominant_emoji(_window_kinds(kinds, i)) if kinds else ""
            out[i] = f"\U0001f525chat\u00d7{round(ratio)}{emoji}"
        prev_hot = ratio is not None
    return out


def _loud_hot(loudness: list[float], over_median_db: float) -> list[bool]:
    if not loudness:
        return []
    threshold = median(loudness) + over_median_db
    return [db >= threshold and db > _LOUD_FLOOR_DB for db in loudness]


def loud_seconds(loudness: list[float], over_median_db: float = _LOUD_OVER_MEDIAN_DB) -> set[int]:
    """Seconds at least `over_median_db` above the median and above the absolute floor."""
    return {i for i, hot in enumerate(_loud_hot(loudness, over_median_db)) if hot}


def _loud_markers(loudness: list[float]) -> dict[int, str]:
    out: dict[int, str] = {}
    prev_hot = False
    for i, hot in enumerate(_loud_hot(loudness, _LOUD_OVER_MEDIAN_DB)):
        if hot and not prev_hot:
            out[i] = "\U0001f50aloud"
        prev_hot = hot
    return out


def heat_seconds(signals: Signals) -> set[int]:
    """Every second inside a loud run or a chat-spike run (not only each run's first second)."""
    secs = loud_seconds(signals.loudness)
    if signals.chat:
        secs |= {i for i, r in enumerate(_chat_hot(signals.chat)) if r is not None}
    return secs


def heat_markers(signals: Signals) -> dict[int, list[str]]:
    """Second -> markers, only at the first second of each run of heat."""
    found = [_chat_markers(signals.chat, signals.chat_kinds)] if signals.chat else []
    if signals.chat_kinds and signals.chat:
        found.append(_clip_markers(signals.chat_kinds, len(signals.chat)))
    found.append(_loud_markers(signals.loudness))
    markers: dict[int, list[str]] = {}
    for source in found:
        for sec, label in source.items():
            markers.setdefault(sec, []).append(label)
    return dict(sorted(markers.items()))
