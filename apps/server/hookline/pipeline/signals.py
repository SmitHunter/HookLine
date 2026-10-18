"""Audience/audio/visual reaction signals and the heat markers shown to Claude in the
transcript."""

from statistics import median

import numpy as np
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
SCARE_MARKER = "\u26a1scare"
# signals.json format; a cached file below this version has its onsets recomputed (2: wave
# G-select added motion, onsets and game-audio transients; 3: exact onset times, 0.3 s speech
# pad). The VOD motion decode is kept from any cache that has it.
SIGNALS_VERSION = 3
# Audio onsets (short-term loudness every 0.1 s, `media.audio.momentary_db`): a jump of
# >= 10 dB within 0.5 s, landing >= 12 dB over the VOD's median level and over -35 dBFS.
ONSET_JUMP_DB = 10.0
ONSET_WITHIN_S = 0.5
ONSET_OVER_MEDIAN_DB = 12.0
ONSET_FLOOR_DB = -35.0
SPEECH_PAD_S = 0.3  # an onset this close to a transcribed word is speech, not game audio
# Visual flashes (`media.motion.frame_motion`, mean abs frame difference, 0-255): >= 40 and
# >= 4x the median motion of the surrounding +-10 s, so busy gameplay is not a flash.
FLASH_MIN = 40.0
FLASH_OVER_LOCAL = 4.0
FLASH_LOCAL_S = 10
FLASH_SOUND_S = 1  # a flash is a scare only with an audio onset within +-1 s
SCARE_MERGE_S = 3  # scare seconds this close together are one marker
LOW_MOTION = 3.0  # per-second motion below this is a still frame (death screen, menu, pause)


class Signals(BaseModel):
    duration: float
    loudness: list[float]  # dB per second
    chat: list[float] | None = None  # weighted messages per second
    # second -> {kind: keyword hits}; absent in signals cached before reaction kinds existed
    chat_kinds: dict[int, dict[str, int]] | None = None
    # Wave G-select (absent, and version 1, in signals cached before it): per-second motion,
    # seconds with an audio onset, and those onsets away from speech (game-audio transients).
    version: int = 1
    motion: list[float] | None = None
    onsets: list[int] | None = None
    transients: list[int] | None = None
    # version 3: the same onsets and transients at their exact times (seconds)
    onset_times: list[float] | None = None
    transient_times: list[float] | None = None


def onset_times(
    db: np.ndarray | list[float], hop: float, speech: list[tuple[float, float]]
) -> tuple[list[float], list[float]]:
    """(times of every audio onset, times of those away from `speech`) in seconds, from a
    short-term loudness envelope `db` sampled every `hop` s (`media.audio.momentary_db`): a
    frame at least ONSET_JUMP_DB above the quietest frame of the preceding ONSET_WITHIN_S,
    landing ONSET_OVER_MEDIAN_DB over the envelope's median and over ONSET_FLOOR_DB; a run of
    such frames is one onset, at its first frame. An onset within SPEECH_PAD_S of a speech span
    is the speaker, not the game."""
    db = np.asarray(db, dtype=np.float64)
    k = max(1, round(ONSET_WITHIN_S / hop))
    if len(db) <= k:
        return [], []
    prior = np.lib.stride_tricks.sliding_window_view(db[:-1], k).min(axis=1)  # frames i-k..i-1
    level = db[k:]
    floor = max(ONSET_FLOOR_DB, float(np.median(db)) + ONSET_OVER_MEDIAN_DB)
    hits = np.flatnonzero((level - prior >= ONSET_JUMP_DB) & (level >= floor)) + k
    firsts = [int(i) for j, i in enumerate(hits) if j == 0 or i - hits[j - 1] > 1]
    talking = np.zeros(len(db), dtype=bool)
    for a, b in speech:
        lo, hi = int((a - SPEECH_PAD_S) / hop), int((b + SPEECH_PAD_S) / hop) + 1
        talking[max(lo, 0) : max(hi, 0)] = True
    every = [round(i * hop, 3) for i in firsts]
    game = [round(i * hop, 3) for i in firsts if not talking[i]]
    return every, game


def audio_onsets(
    db: np.ndarray | list[float], hop: float, speech: list[tuple[float, float]]
) -> tuple[list[int], list[int]]:
    """`onset_times` as whole seconds: (seconds with an audio onset, seconds with one away from
    `speech`, i.e. game-audio transients)."""
    every, game = onset_times(db, hop, speech)
    return seconds_of(every), seconds_of(game)


def seconds_of(times: list[float]) -> list[int]:
    return sorted({int(t + 1e-9) for t in times})


def flash_seconds(motion: list[float]) -> set[int]:
    """Seconds whose motion is a flash or hard visual jump: >= FLASH_MIN and >= FLASH_OVER_LOCAL
    times the median motion of the surrounding +-FLASH_LOCAL_S seconds."""
    m = np.asarray(motion, dtype=np.float64)
    if not len(m):
        return set()
    padded = np.pad(m, FLASH_LOCAL_S, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * FLASH_LOCAL_S + 1)
    local = np.median(window, axis=1)
    return {int(i) for i in np.flatnonzero((m >= FLASH_MIN) & (m >= FLASH_OVER_LOCAL * local))}


def scare_seconds(signals: Signals) -> set[int]:
    """Game-audio transients, plus flashes with an audio onset within FLASH_SOUND_S (a silent
    flash is a cut or a menu, not a scare). Empty for signals cached before scares existed."""
    out = set(signals.transients or ())
    if signals.motion and signals.onsets is not None:
        onsets = set(signals.onsets)
        out |= {
            s
            for s in flash_seconds(signals.motion)
            if any(s + d in onsets for d in range(-FLASH_SOUND_S, FLASH_SOUND_S + 1))
        }
    return out


def scare_times(signals: Signals) -> list[float]:
    """`scare_seconds` at their exact times: each transient's onset time, each flash at the
    time of its sound's onset (within FLASH_SOUND_S). Signals without exact times give each
    scare's second."""
    if signals.transient_times is None or signals.onset_times is None:
        return sorted(float(s) for s in scare_seconds(signals))
    out = set(signals.transient_times)
    if signals.motion:
        for s in flash_seconds(signals.motion):
            out |= {t for t in signals.onset_times if abs(int(t + 1e-9) - s) <= FLASH_SOUND_S}
    return sorted(out)


def low_motion(motion: list[float] | None, start: float, end: float) -> bool:
    """True when every second from round(start) to round(end) (exclusive) is below LOW_MOTION.
    False without motion, or when the span reaches past it (unknown is not still)."""
    if not motion:
        return False
    lo, hi = round(start), round(end)
    if lo < 0 or hi > len(motion):
        return False
    return all(m < LOW_MOTION for m in motion[lo:hi])


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


def _scare_markers(signals: Signals) -> dict[int, str]:
    out: dict[int, str] = {}
    last = None
    for sec in sorted(scare_seconds(signals)):
        if last is None or sec - last > SCARE_MERGE_S:
            out[sec] = SCARE_MARKER
        last = sec
    return out


def heat_seconds(signals: Signals) -> set[int]:
    """Every second inside a loud run or a chat-spike run (not only each run's first second),
    and every scare second."""
    secs = loud_seconds(signals.loudness) | scare_seconds(signals)
    if signals.chat:
        secs |= {i for i, r in enumerate(_chat_hot(signals.chat)) if r is not None}
    return secs


def heat_markers(signals: Signals) -> dict[int, list[str]]:
    """Second -> markers, only at the first second of each run of heat."""
    found = [_chat_markers(signals.chat, signals.chat_kinds)] if signals.chat else []
    if signals.chat_kinds and signals.chat:
        found.append(_clip_markers(signals.chat_kinds, len(signals.chat)))
    found.append(_loud_markers(signals.loudness))
    found.append(_scare_markers(signals))
    markers: dict[int, list[str]] = {}
    for source in found:
        for sec, label in source.items():
            markers.setdefault(sec, []).append(label)
    return dict(sorted(markers.items()))
