"""Two-pass EBU R128 loudness normalisation helpers (target -14 LUFS, TP -1 dBTP)."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from hookline.media.ffmpeg import FFmpegError, _run

TARGET_I = -14.0
TARGET_TP = -1.0
TARGET_LRA = 11.0
# AAC overshoots the limiter ceiling by ~0.3 dB, so pass 2 aims below the -1 dBTP bar.
FINAL_TP = -2.0
MAX_GAIN_DB = 20.0
SILENCE_FLOOR_LUFS = -50.0
COMPRESSOR = "acompressor=threshold=-21dB:ratio=3:attack=5:release=120:makeup=2"


@dataclass(frozen=True)
class Loudness:
    i: float
    tp: float
    lra: float
    thresh: float
    offset: float


def measure_loudness(path: Path) -> Loudness:
    """Pass 1 of loudnorm over `path`'s audio: integrated loudness, true peak, LRA."""
    res = _run(
        [
            "ffmpeg", "-hide_banner", "-v", "info", "-nostats", "-i", str(path), "-vn",
            "-af", f"loudnorm=I={TARGET_I:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}:print_format=json",
            "-f", "null", "-",
        ]
    )  # fmt: skip
    blocks = re.findall(r"\{[^{}]*\}", res.stderr)
    if not blocks:
        raise FFmpegError("no loudnorm statistics in ffmpeg output", res.stderr)
    d = json.loads(blocks[-1])
    try:
        return Loudness(
            i=_num(d["input_i"]),
            tp=_num(d["input_tp"]),
            lra=_num(d["input_lra"]),
            thresh=_num(d["input_thresh"]),
            offset=_num(d["target_offset"]),
        )
    except KeyError as e:
        raise FFmpegError(f"loudnorm statistics missing {e}", res.stderr) from e


def _num(v: str | float) -> float:
    x = float(v)
    return -70.0 if x == float("-inf") else x


def loudnorm_filter(m: Loudness) -> str:
    """Audio filter chain for the final encode: gentle compression, then loudnorm pass 2.

    Gain is capped at +20 dB (target I = m.i + 20 when -14 would need more) so noise floors
    are not blown up; near-silent audio (< -50 LUFS) is left untouched."""
    if m.i < SILENCE_FLOOR_LUFS:
        return "anull"
    target = min(TARGET_I, m.i + MAX_GAIN_DB)
    return (
        f"{COMPRESSOR},"
        f"loudnorm=I={target:g}:TP={FINAL_TP:g}:LRA={TARGET_LRA:g}"
        f":measured_I={m.i:g}:measured_TP={m.tp:g}:measured_LRA={m.lra:g}"
        f":measured_thresh={m.thresh:g}:offset={m.offset:g}:linear=true:print_format=json"
    )
