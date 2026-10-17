"""Two-pass EBU R128 loudness normalisation helpers (target -14 LUFS, TP -1 dBTP)."""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from hookline.media.ffmpeg import FFmpegError, _run, has_audio

TARGET_I = -14.0
TARGET_TP = -1.0
TARGET_LRA = 11.0
# AAC overshoots the limiter ceiling by ~0.3 dB, so pass 2 aims below the -1 dBTP bar.
FINAL_TP = -2.0
MAX_GAIN_DB = 20.0
SILENCE_FLOOR_LUFS = -50.0
# makeup is a linear factor (1 = unity). Loudnorm supplies the gain, so no blind boost here.
COMPRESSOR = "acompressor=threshold=-21dB:ratio=3:attack=5:release=120:makeup=1"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Loudness:
    i: float
    tp: float
    lra: float
    thresh: float
    offset: float


def parse_loudnorm_json(stderr: str) -> dict:
    """The last JSON object in ffmpeg's stderr (loudnorm prints its stats there)."""
    blocks = re.findall(r"\{[^{}]*\}", stderr)
    if not blocks:
        raise FFmpegError("no loudnorm statistics in ffmpeg output", stderr)
    return json.loads(blocks[-1])


def measure_loudness(path: Path, prefilter: str | None = None) -> Loudness | None:
    """Pass 1 of loudnorm over `path`'s audio (after `prefilter`, e.g. COMPRESSOR, when given):
    integrated loudness, true peak, LRA. None when the file has no audio stream."""
    if not has_audio(path):
        return None
    chain = f"{prefilter}," if prefilter else ""
    res = _run(
        [
            "ffmpeg", "-hide_banner", "-v", "info", "-nostats", "-i", str(path), "-vn",
            "-af",
            f"{chain}loudnorm=I={TARGET_I:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}:print_format=json",
            "-f", "null", "-",
        ]
    )  # fmt: skip
    d = parse_loudnorm_json(res.stderr)
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


def loudnorm_mode(m: Loudness, target_i: float | None = None) -> str:
    """ "skip" (near-silent), "linear" or "dynamic". `m` is measured AFTER the prefilter;
    `target_i` is loudnorm's I (default `_target_i(m)`).

    Mirrors loudnorm's own rule: linear gain only works when the clip's LRA is nonzero and
    fits the target and the gained true peak stays under the ceiling; otherwise loudnorm would silently fall
    back to dynamic mode, so we ask for dynamic explicitly."""
    if m.i < SILENCE_FLOOR_LUFS:
        return "skip"
    gain = (_target_i(m) if target_i is None else target_i) - m.i
    if 0 < m.lra <= TARGET_LRA and m.tp + gain <= FINAL_TP:
        return "linear"
    return "dynamic"


def _target_i(m: Loudness) -> float:
    return min(TARGET_I, m.i + MAX_GAIN_DB)


def loudnorm_filter(m: Loudness, pre: str = COMPRESSOR, target_i: float | None = None) -> str:
    """Audio filter chain for the final encode: `pre` (the prefilter `m` was measured after),
    then loudnorm pass 2 aiming at `target_i` (default `_target_i(m)`).

    Gain is capped at +20 dB (target I = m.i + 20 when -14 would need more) so noise floors
    are not blown up; near-silent audio (< -50 LUFS) is left untouched."""
    target = _target_i(m) if target_i is None else target_i
    mode = loudnorm_mode(m, target)
    log.info("audio: loudnorm %s, measured I=%.1f TP=%.1f LRA=%.1f", mode, m.i, m.tp, m.lra)
    if mode == "skip":
        return "anull"
    base = f"loudnorm=I={target:g}:TP={FINAL_TP:g}:LRA={TARGET_LRA:g}"
    if mode == "dynamic":
        return f"{pre},{base}:print_format=json"
    return (
        f"{pre},{base}"
        f":measured_I={m.i:g}:measured_TP={m.tp:g}:measured_LRA={m.lra:g}"
        f":measured_thresh={m.thresh:g}:offset={m.offset:g}:linear=true:print_format=json"
    )


# Peaks are limited BEFORE loudnorm, at the target level. Left to loudnorm, a high-crest clip
# (true peak more than target I - FINAL_TP = 12 dB above its loudness, e.g. plosives in quiet
# speech) forces dynamic mode, whose limiter then has to take several dB off the peaks and
# the integrated loudness undershoots (real job e87398ed46 clip 04: crest 17 dB -> -15.0
# LUFS). The compressor cannot help there: it runs before any gain, so on -30 LUFS speech it
# sits below its threshold. With the peaks held at LIMIT_DB and the level near the target,
# loudnorm can go linear: an exact gain of a fraction of a dB.
LIMIT_DB = -3.0
LIMITER = "alimiter=limit={lim:.4f}:attack=1:release=50:level=false:latency=true"
PREGAIN_PASSES = 3  # limiting costs loudness, so the pre-gain is re-aimed by the measured miss


def prefilter(gain_db: float) -> str:
    """COMPRESSOR, then `gain_db` of gain, then a lookahead limiter at LIMIT_DB."""
    return f"{COMPRESSOR},volume={gain_db:.2f}dB," + LIMITER.format(lim=10 ** (LIMIT_DB / 20))


@dataclass(frozen=True)
class AudioPlan:
    af: str  # the final encode's audio filter
    mode: str  # loudnorm_mode: "skip", "linear" or "dynamic"


def plan_loudness(path: Path) -> AudioPlan | None:
    """Measure `path`'s audio and plan its normalisation to `_target_i` (None without audio):
    the compressed loudness sets a pre-gain to the target; the limited result is re-measured
    and the pre-gain corrected by the miss (up to PREGAIN_PASSES measurements) until
    loudnorm's remaining gain fits under the true-peak ceiling, so it can go linear."""
    c = measure_loudness(path, COMPRESSOR)
    if c is None:
        return None
    if loudnorm_mode(c) == "skip":
        return AudioPlan(af=loudnorm_filter(c), mode="skip")
    target = _target_i(c)
    gain = target - c.i
    for _ in range(PREGAIN_PASSES):
        pre = prefilter(gain)
        m = measure_loudness(path, pre)
        miss = target - m.i
        log.info("audio: pre-gain %.2f dB -> I=%.2f TP=%.2f", gain, m.i, m.tp)
        if m.tp + miss <= FINAL_TP:
            break
        gain += miss
    return AudioPlan(af=loudnorm_filter(m, pre, target), mode=loudnorm_mode(m, target))


TP_TRIM_ABOVE = -1.2
TP_TRIM_TARGET = -1.5


def ensure_true_peak(dst: Path) -> float | None:
    """Post-encode safety net: AAC can overshoot the limiter ceiling. When `dst`'s true peak
    is above -1.2 dBTP, re-encode only its audio with a volume trim to land at about -1.5.
    Returns the trim in dB (None when nothing was done)."""
    m = measure_loudness(dst)
    if m is None or m.tp <= TP_TRIM_ABOVE:
        return None
    trim = TP_TRIM_TARGET - m.tp
    tmp = dst.with_suffix(".trim" + dst.suffix)
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(dst),
            "-map", "0:v", "-map", "0:a", "-c:v", "copy",
            "-af", f"volume={trim:.2f}dB",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(tmp),
        ]
    )  # fmt: skip
    os.replace(tmp, dst)
    after = measure_loudness(dst)
    log.info("audio: true-peak trim %.2f dB -> TP %s", trim, f"{after.tp:.2f}" if after else "?")
    return trim
