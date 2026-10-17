import json
import re
from pathlib import Path

from hookline.media.ffmpeg import _run

_FLOOR_DB = -100.0
_PTS_RE = re.compile(r"pts_time:(\S+)")
_RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(\S+)")
_SIL_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


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
