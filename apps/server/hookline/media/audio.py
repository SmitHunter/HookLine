import json
import re
import subprocess
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


@dataclass(frozen=True)
class Envelope:
    """Short-time RMS loudness: `db[k]` (dBFS, floored at -100) covers source seconds
    [t0 + k * hop, t0 + (k + 1) * hop)."""

    t0: float
    hop: float
    db: np.ndarray

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
    return Envelope(t0=start, hop=hop, db=db)


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
