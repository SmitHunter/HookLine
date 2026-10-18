"""Audience/audio/visual reaction signals and the heat markers shown to Claude in the
transcript."""

from statistics import median

import numpy as np
from pydantic import BaseModel

from hookline.media.motion import MOTION_FPS

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
# pad; 4: wave H per-frame screen flags; 5: wave H round 1 screen chroma, cam masked). The VOD
# motion decode is kept from any cache that has it, the screen flags and their chroma for the
# current cam mask (`pipeline.run._signals`).
SIGNALS_VERSION = 5
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
# Wave H: a static screen (death / loading / menu screen, a fade) is a run of screen frames
# (`media.motion.screen_frames`: near uniform or near white), single busy frames bridged,
# lasting at least STATIC_MIN_S, with words over at most STATIC_VOICED_MAX of it (the
# streamer's line over a death screen goes with it; dark gameplay he talks over stays). 3 s: a
# horror stream's death / respawn sequences ran 16 s, its pitch-black corridors (with a
# reaction "Oh," and a scare in them) ~2 s.
STATIC_MIN_S = 3.0
STATIC_VOICED_MAX = 0.4
STATIC_BRIDGE_FRAMES = 1
# ... and (round 1) grey: the run's mean chroma (`Signals.screen_chroma`, the stream cam masked)
# under STATIC_CHROMA_MAX. Measured on the horror stream, cam masked: death / loading / black
# screens 0.03-0.08, dark gameplay (the GrabPack HUD at 1556.75-1560.25, pitch-black
# corridors) 1.47-2.3. (Unmasked the cam adds ~1.5: death screens ~1.5 vs gameplay 3.0-3.7.)
STATIC_CHROMA_MAX = 0.8


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
    # version 4 (wave H): per second, a bitmask of its MOTION_FPS frames that are screen frames
    # (`media.motion.frame_stats`)
    screen: list[int] | None = None
    # version 5 (wave H round 1): per second, the mean chroma (`media.motion.frame_chroma`) of
    # its screen frames (None without any), over the frame outside `screen_mask` (normalized
    # x0, y0, x1, y1 boxes: the stream profile's cam and labels; None = the whole frame)
    screen_chroma: list[float | None] | None = None
    screen_mask: list[list[float]] | None = None
    # the source and signals version (`run.motion_key`) whose motion decode failed or timed
    # out: not retried until either changes
    motion_failed: str | None = None


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


def screen_flags(signals: Signals, fps: int = MOTION_FPS) -> np.ndarray:
    """Per frame (`fps` per second) of the VOD: a screen frame. Empty without screen data."""
    if not signals.screen:
        return np.zeros(0, bool)
    masks = np.asarray(signals.screen, dtype=np.int64)
    return ((masks[:, None] >> np.arange(fps)) & 1).astype(bool).ravel()


def _voiced_s(starts: np.ndarray, ends: np.ndarray, lo: float, hi: float) -> float:
    """Seconds of [lo, hi] under at least one word span (overlaps counted once)."""
    sel = (ends > lo) & (starts < hi)
    total, cur = 0.0, lo
    for a, b in sorted(zip(np.maximum(starts[sel], lo), np.minimum(ends[sel], hi), strict=True)):
        a = max(a, cur)
        if b > a:
            total += b - a
            cur = b
    return float(total)


# static_runs per (screen flags, chroma, words list): computed once per job / clip, not for
# every boundary candidate (a 10 h VOD has ~144k screen frames). Entries hold their inputs, so
# an id is never reused while cached.
_STATIC_CACHE: dict[tuple[int, int, int, int], tuple[object, ...]] = {}
_STATIC_CACHE_MAX = 16


def static_runs(signals: Signals, words: list, fps: int = MOTION_FPS) -> list[tuple[float, float]]:
    """Static screens (source s, frame-precise): runs of screen frames (`screen_flags`; gaps of
    up to STATIC_BRIDGE_FRAMES busy frames bridged) lasting at least STATIC_MIN_S, with
    `words` (the transcript; voice-clamped by the pipeline) covering at most
    STATIC_VOICED_MAX of the run and, with `Signals.screen_chroma`, of mean chroma under
    STATIC_CHROMA_MAX (dark gameplay is coloured). Empty without screen data. Cached per
    (signals' screen data, `words` list): the caller must not mutate `words` in place."""
    key = (id(signals.screen), id(signals.screen_chroma), id(words), fps)
    hit = _STATIC_CACHE.get(key)
    if hit is not None and hit[0] is signals.screen and hit[1] is signals.screen_chroma:
        return list(hit[3])  # type: ignore[arg-type]
    runs = _static_runs(signals, words, fps)
    if len(_STATIC_CACHE) >= _STATIC_CACHE_MAX:
        _STATIC_CACHE.pop(next(iter(_STATIC_CACHE)))
    _STATIC_CACHE[key] = (signals.screen, signals.screen_chroma, words, runs)
    return list(runs)


def _static_runs(signals: Signals, words: list, fps: int) -> list[tuple[float, float]]:
    flags = screen_flags(signals, fps)
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    runs: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for k in idx[1:]:
        if k - runs[-1][1] <= STATIC_BRIDGE_FRAMES + 1:
            runs[-1][1] = int(k)
        else:
            runs.append([int(k), int(k)])
    starts = np.array([w.start for w in words], dtype=np.float64)
    ends = np.array([w.end for w in words], dtype=np.float64)
    chroma = signals.screen_chroma
    out = []
    for a, b in runs:
        lo, hi = a / fps, (b + 1) / fps
        if hi - lo < STATIC_MIN_S - 1e-6:
            continue
        if _voiced_s(starts, ends, lo, hi) > STATIC_VOICED_MAX * (hi - lo):
            continue
        if chroma is not None:
            seen = [c for c in chroma[a // fps : b // fps + 1] if c is not None]
            if seen and sum(seen) / len(seen) >= STATIC_CHROMA_MAX:
                continue
        out.append((round(lo, 3), round(hi, 3)))
    return out


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
