import json
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hookline.media.ffmpeg import FFmpegError, _run

_FLOOR_DB = -100.0
_PTS_RE = re.compile(r"pts_time:(\S+)")
_RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(\S+)")
_SIL_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")
ENVELOPE_RATE = 16000
ENVELOPE_HOP = 0.01
# Voice likelihood per envelope frame (Q3b: game sound and music are not uncaptioned speech):
# over a VOICE_WINDOW-sample window, the voice band's (VOICE_BAND Hz) share of the energy in
# VOICE_BROAD Hz is >= VOICE_SHARE and the spectral flatness (geometric / arithmetic mean of
# the power spectrum: ~1 for noise, near 0 for harmonic sound) <= VOICE_FLATNESS.
VOICE_WINDOW = 512
VOICE_BAND = (250.0, 4000.0)
VOICE_BROAD = (60.0, 8000.0)
VOICE_SHARE = 0.5
VOICE_FLATNESS = 0.35


@dataclass(frozen=True)
class Envelope:
    """Short-time RMS loudness: `db[k]` (dBFS, floored at -100) covers source seconds
    [t0 + k * hop, t0 + (k + 1) * hop)."""

    t0: float
    hop: float
    db: np.ndarray
    voice: np.ndarray | None = None  # per frame: voice-like spectrum (`voice_frames`)

    def index(self, t: float) -> int:
        """The frame containing time `t`, clamped to the envelope."""
        return min(max(int((t - self.t0) / self.hop + 1e-9), 0), len(self.db) - 1)

    def time(self, k: int) -> float:
        """Start time of frame `k`."""
        return self.t0 + k * self.hop

    def level(self, t: float) -> float:
        return float(self.db[self.index(t)])


def energy_envelope(wav: Path, start: float, end: float, hop: float = ENVELOPE_HOP) -> Envelope:
    """RMS loudness of mono 16 kHz audio over [start, end] in `hop`-second frames."""
    start = max(start, 0.0)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(wav),
        "-ac", "1", "-ar", str(ENVELOPE_RATE), "-f", "s16le", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, check=False, capture_output=True)
    except FileNotFoundError as e:
        raise FFmpegError("ffmpeg not found on PATH") from e
    if res.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with code {res.returncode}", res.stderr.decode("utf-8", "replace")
        )
    pcm = np.frombuffer(res.stdout, np.int16).astype(np.float64) / 32768.0
    n = round(hop * ENVELOPE_RATE)
    frames = pcm[: len(pcm) // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(axis=1)) if len(frames) else np.zeros(0)
    db = np.maximum(20 * np.log10(np.maximum(rms, 1e-10)), _FLOOR_DB)
    return Envelope(t0=start, hop=hop, db=db, voice=voice_frames(pcm, n, len(db)))


def voice_frames(pcm: np.ndarray, hop: int, count: int, rate: int = ENVELOPE_RATE) -> np.ndarray:
    """Per `hop`-sample frame (`count` of them) of mono `pcm`: whether its spectrum (a
    VOICE_WINDOW Hann window centred on the frame) looks like a voice -- most of its energy in
    the voice band and harmonic rather than noisy (VOICE_SHARE, VOICE_FLATNESS)."""
    if count == 0:
        return np.zeros(0, bool)
    half = VOICE_WINDOW // 2
    padded = np.pad(pcm, (half, half + VOICE_WINDOW))
    starts = np.arange(count) * hop + hop // 2
    idx = starts[:, None] + np.arange(VOICE_WINDOW)[None, :]
    spec = np.abs(np.fft.rfft(padded[idx] * np.hanning(VOICE_WINDOW), axis=1)) ** 2
    freqs = np.fft.rfftfreq(VOICE_WINDOW, 1 / rate)
    broad = (freqs >= VOICE_BROAD[0]) & (freqs <= VOICE_BROAD[1])
    band = (freqs >= VOICE_BAND[0]) & (freqs <= VOICE_BAND[1])
    p = spec[:, broad] + 1e-12
    total = p.sum(axis=1)
    share = spec[:, band].sum(axis=1) / total
    flat = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
    return (share >= VOICE_SHARE) & (flat <= VOICE_FLATNESS)


def _sample_rate(wav: Path) -> int:
    res = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "json",
            str(wav),
        ]
    )
    return int(json.loads(res.stdout)["streams"][0]["sample_rate"])


def rms_per_second(wav: Path) -> list[float]:
    """RMS loudness in dBFS for each 1 s window; digital silence (-inf) is clamped to -100."""
    rate = _sample_rate(wav)
    res = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(wav),
            "-af",
            f"asetnsamples=n={rate}:p=0,astats=metadata=1:reset=1,ametadata=print:file=-",
            "-f",
            "null",
            "-",
        ]
    )
    frames: list[tuple[float, float]] = []
    pts = 0.0
    for line in res.stdout.splitlines():
        if m := _PTS_RE.search(line):
            pts = float(m.group(1))
        elif m := _RMS_RE.search(line):
            try:
                db = float(m.group(1))
            except ValueError:
                db = _FLOOR_DB
            frames.append((pts, max(db, _FLOOR_DB)))
    frames.sort(key=lambda f: f[0])
    return [db for _, db in frames]


def detect_silences(
    wav: Path, start: float, end: float, noise_db: float = -32.0, min_s: float = 0.35
) -> list[tuple[float, float]]:
    """Silent spans as absolute source times, clipped to [start, end]."""
    res = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(wav),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_s}",
            "-f",
            "null",
            "-",
        ]
    )
    spans: list[tuple[float, float]] = []
    open_at: float | None = None
    for line in res.stderr.splitlines():
        if m := _SIL_START_RE.search(line):
            open_at = start + float(m.group(1))
        elif (m := _SIL_END_RE.search(line)) and open_at is not None:
            spans.append((open_at, start + float(m.group(1))))
            open_at = None
    if open_at is not None:
        spans.append((open_at, end))
    clipped = [(max(a, start), min(b, end)) for a, b in spans]
    return [(a, b) for a, b in clipped if b > a]


MOMENTARY_HOP = 0.1
MOMENTARY_WINDOW = 0.4
_BLOCK_HOPS = 600  # hops read per block: a multi-hour VOD never sits in memory at once


def momentary_db(
    wav: Path, hop: float = MOMENTARY_HOP, window: float = MOMENTARY_WINDOW
) -> np.ndarray:
    """Short-term loudness of a 16-bit PCM WAV every `hop` s: `db[k]` (dBFS mean power,
    floored at -100) is the `window` s ending at (k + 1) * hop (shorter at the start). Read in
    blocks with the `wave` module, so it costs well under a second per hour of 16 kHz audio."""
    with wave.open(str(wav), "rb") as w:
        if w.getsampwidth() != 2:
            raise FFmpegError(f"{wav} is not 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        n = max(1, round(hop * rate))
        powers: list[np.ndarray] = []
        while raw := w.readframes(n * _BLOCK_HOPS):
            x = np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0
            if channels > 1:
                x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
            m = len(x) // n
            powers.append((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
    power = np.concatenate(powers) if powers else np.zeros(0)
    k = max(1, round(window / hop))
    csum = np.concatenate([[0.0], np.cumsum(power)])
    ends = np.arange(1, len(power) + 1)
    starts = np.maximum(ends - k, 0)
    mean = (csum[ends] - csum[starts]) / (ends - starts) if len(power) else power
    return np.maximum(10 * np.log10(np.maximum(mean, 1e-10)), _FLOOR_DB)


def digital_zero_runs(path: Path, min_s: float = 0.2) -> list[tuple[float, float]]:
    """(start s, length s) of each run of exact digital silence (every sample 0, e.g. a
    source's noise gate) at least `min_s` long in the first audio stream (downmixed to mono
    at its own rate); [] without audio."""
    rate = _sample_rate(path)
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
         "-ac", "1", "-f", "s16le", "-"],
        check=False, capture_output=True,
    )  # fmt: skip
    if res.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with code {res.returncode}", res.stderr.decode("utf-8", "replace")
        )
    zero = np.frombuffer(res.stdout, np.int16) == 0
    if not zero.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], zero.astype(np.int8), [0]))))
    need = round(min_s * rate)
    return [
        (round(a / rate, 3), round((b - a) / rate, 3))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if b - a >= need
    ]
