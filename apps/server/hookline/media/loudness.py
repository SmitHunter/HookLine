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


# Wave G-select: the clip's loudness range is held to MAX_LRA by a leveler (a slow
# compressor after the gain, which has put the speech at TARGET_I, so its threshold sits
# relative to the speech level) before the limiter. LEVELERS is a ladder from gentle to firm,
# climbed only while the range is over MAX_LRA or the speech sits more than SPEECH_SWAY_DB
# under the clip: a podcast whose range is already natural gets none (no pumping); game
# stingers over a quiet voice get squashed. Gain corrections after a rung go AFTER the
# leveler (its makeup), so they neither move its threshold nor get compressed by it.
MAX_LRA = 8.0
# (threshold vs TARGET_I in dB, ratio), gentle to firm
LEVELERS = (
    (4.0, 2.0),
    (0.0, 3.0),
    (-4.0, 4.0),
    (-8.0, 6.0),
    (-12.0, 8.0),
)
LEVELER = (
    "acompressor=threshold={thr:g}dB:ratio={ratio:g}:attack=20:release={rel:g}:knee={knee:g}"
    ":makeup=1"
)
# Rungs from FIRM_RUNG on (up to +10 dB of makeup) release slowly over a soft knee, so the game
# bed and noise floor do not swell back up in the gaps between words (pumping).
FIRM_RUNG = 3
GENTLE_RELEASE_MS, FIRM_RELEASE_MS = 300, 1200
GENTLE_KNEE_DB, FIRM_KNEE_DB = 2.83, 6.0
# The final loudnorm aims the whole clip so the speech lands on TARGET_I, but never more than
# this far from TARGET_I (Q1: -14 +/- 1 LUFS for the whole clip).
SPEECH_SWAY_DB = 0.5
# Measurement decodes one plan may spend (the initial one or two included); typical clips
# take 3-8, a pathological ladder stops here with the best chain so far.
MAX_DECODES = 24
# A chain within this of its aim is at its final level: its loudness range can be judged.
SETTLED_DB = 1.0
DYNAMIC_PASSES = 2  # gain corrections bringing a dynamic plan's chain onto the target ...
DYNAMIC_TOL_DB = 0.3  # ... until it is this close
MIN_SPEECH_S = 3.0  # less speech than this is not gated (too little to measure)
_SPEECH_JOIN_S = 0.3  # word spans closer than this are measured as one stretch
# The limiter runs at 192 kHz in the final encode too. loudnorm's measuring (dynamic) pass
# forces 192 kHz on everything before it, its linear pass does not: the limiter measured at
# 192 kHz then ran at 48 kHz in the encode, letting through intersample peaks the plan never
# saw (a limited white-noise stinger: +1.4 LU and +5 dBTP over the measurement).
LIMIT_RATE = 192000


def prefilter(gain_db: float, level: int = 0, makeup_db: float = 0.0) -> str:
    """COMPRESSOR, then `gain_db` of gain, then leveler rung `level` (0: none; see LEVELERS)
    followed by `makeup_db` of gain, then a lookahead limiter at LIMIT_DB, at LIMIT_RATE (so
    it limits intersample peaks)."""
    leveler = ""
    if level:
        off, ratio = LEVELERS[level - 1]
        thr = TARGET_I + off
        firm = level >= FIRM_RUNG
        rel = FIRM_RELEASE_MS if firm else GENTLE_RELEASE_MS
        knee = FIRM_KNEE_DB if firm else GENTLE_KNEE_DB
        leveler = LEVELER.format(thr=thr, ratio=ratio, rel=rel, knee=knee)
        leveler += f",volume={makeup_db:.2f}dB,"
    lim = LIMITER.format(lim=10 ** (LIMIT_DB / 20))
    return f"{COMPRESSOR},volume={gain_db:.2f}dB,{leveler}aresample={LIMIT_RATE},{lim}"


def _rung_makeup(level: int) -> float:
    """Roughly what rung `level` takes off speech sitting at its threshold offset's level."""
    off, ratio = LEVELERS[level - 1]
    return -off * (1 - 1 / ratio) if off < 0 else 0.0


def speech_filter(spans: list[tuple[float, float]] | None) -> str | None:
    """An `aselect` keeping only the speech `spans` (seconds on the file's timeline; spans
    closer than 0.3 s merged), for a speech-gated measurement. None under MIN_SPEECH_S."""
    merged: list[list[float]] = []
    for a, b in sorted(spans or []):
        if merged and a - merged[-1][1] < _SPEECH_JOIN_S:
            merged[-1][1] = max(merged[-1][1], b)
        elif b > a:
            merged.append([a, b])
    if sum(b - a for a, b in merged) < MIN_SPEECH_S:
        return None
    terms = "+".join(f"between(t\\,{a:.3f}\\,{b:.3f})" for a, b in merged)
    return f"aselect='{terms}',asetpts=N/SR/TB"


@dataclass(frozen=True)
class AudioPlan:
    af: str  # the final encode's audio filter
    mode: str  # loudnorm_mode: "skip", "linear" or "dynamic"


def _gated(pre: str | None, gate: str | None) -> str | None:
    return ",".join(f for f in (pre, gate) if f) or None


def _aim(target: float, m: Loudness, s: Loudness | None) -> float:
    """The whole clip's loudnorm target: `target` moved so the speech (`s`, measured through
    the same chain as `m`) lands on it, by at most SPEECH_SWAY_DB."""
    if s is None or s.i < SILENCE_FLOOR_LUFS:
        return target
    return min(max(target - (s.i - m.i), target - SPEECH_SWAY_DB), target + SPEECH_SWAY_DB)


def plan_loudness(path: Path, speech: list[tuple[float, float]] | None = None) -> AudioPlan | None:
    """Measure `path`'s audio and plan its normalisation (None without audio).

    The pre-gain puts the speech (the `speech` spans, seconds on `path`'s timeline, measured
    gated; the whole clip without them) at the target `_target_i`; the chain (gain, leveler,
    limiter) is re-measured and corrected by the miss until loudnorm's remaining gain fits
    under the true-peak ceiling, so it can go linear. While the clip's loudness range is over
    MAX_LRA, or its speech sits more than SPEECH_SWAY_DB under the whole clip (voice under
    game), the leveler climbs one rung of LEVELERS. loudnorm then aims the whole clip so the
    speech lands on the target, within SPEECH_SWAY_DB of it (`_aim`). Each measurement costs
    one decode of the clip's audio (twice when gated), at most MAX_DECODES in all.

    Each rung climbed gets a fresh set of PREGAIN_PASSES; makeup corrections made on a rung
    carry over to the next (on top of its own rung makeup). When loudnorm will still run in
    dynamic mode, it aims at the target itself (no speech sway): dynamic mode lands the
    whole clip on its target, and the sway would only eat Q1's margin."""
    c = measure_loudness(path, COMPRESSOR)
    if c is None:
        return None
    if loudnorm_mode(c) == "skip":
        return AudioPlan(af=loudnorm_filter(c), mode="skip")
    target = _target_i(c)
    gate = speech_filter(speech)
    sc = measure_loudness(path, _gated(COMPRESSOR, gate)) if gate else None
    decodes = 2 if gate else 1
    per_pass = 2 if gate else 1
    gain = target - (sc.i if sc and sc.i >= SILENCE_FLOOR_LUFS else c.i)
    gain = min(gain, target - c.i + MAX_GAIN_DB / 2)  # a near-silent voice: no huge boost
    level, correction, passes = 0, 0.0, 0
    while True:
        makeup = _rung_makeup(level) + correction if level else 0.0
        pre = prefilter(gain, level, makeup)
        m = measure_loudness(path, pre)
        s = measure_loudness(path, _gated(pre, gate)) if gate else None
        decodes += per_pass
        aim = _aim(target, m, s)
        log.info(
            "audio: pre-gain %.2f dB, leveler %d (+%.2f dB) -> I=%.2f TP=%.2f LRA=%.1f%s",
            gain, level, makeup, m.i, m.tp, m.lra, f" speech I={s.i:.2f}" if s else "",
        )  # fmt: skip
        budget = decodes + per_pass <= MAX_DECODES
        audible = s is not None and s.i >= SILENCE_FLOOR_LUFS
        buried = audible and m.i - s.i > SPEECH_SWAY_DB
        passes += 1
        miss = aim - m.i
        # the range is only judged near the final level: the noise floor's share of it
        # moves with the gain (LRA's absolute gate)
        settled = abs(miss) <= SETTLED_DB or passes >= PREGAIN_PASSES
        climb = (m.lra > MAX_LRA or buried) and level < len(LEVELERS) and budget
        if climb and settled:
            level += 1
            passes = 0
            continue
        if not climb and (m.tp + miss <= FINAL_TP or passes >= PREGAIN_PASSES or not budget):
            break
        if level:
            correction += miss
        else:
            gain += miss
    if loudnorm_mode(m, aim) == "dynamic":
        # dynamic loudnorm lands the whole clip on its target (no speech sway, which would
        # only eat Q1's margin), but undershoots when it must also add gain into its limiter:
        # the chain itself is brought to the target first
        aim = target
        for _ in range(DYNAMIC_PASSES):
            miss = aim - m.i
            if abs(miss) <= DYNAMIC_TOL_DB or decodes + 1 > MAX_DECODES:
                break
            if level:
                correction += miss
            else:
                gain += miss
            makeup = _rung_makeup(level) + correction if level else 0.0
            pre = prefilter(gain, level, makeup)
            m = measure_loudness(path, pre)
            decodes += 1
            log.info(
                "audio: dynamic: pre-gain %.2f dB, leveler %d (+%.2f dB) -> I=%.2f TP=%.2f",
                gain, level, makeup, m.i, m.tp,
            )  # fmt: skip
    log.info("audio: loudness plan took %d decodes", decodes)
    return AudioPlan(af=loudnorm_filter(m, pre, aim), mode=loudnorm_mode(m, aim))


TP_TRIM_ABOVE = -1.2
TP_TRIM_TARGET = -1.5
# The safety net's first try: a lookahead limiter at this ceiling (at LIMIT_RATE) on the
# decoded audio takes off only the overshooting peaks -- AAC rings on a lone transient (a game
# hit: stream clip 01 went 1.9 dB over its plan) -- so the clip keeps its loudness.
TP_LIMIT_DB = -2.5


def _reencode_audio(src: Path, dst: Path, af: str) -> None:
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(src),
            "-map", "0:v", "-map", "0:a", "-c:v", "copy",
            "-af", af,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(dst),
        ]
    )  # fmt: skip


def ensure_true_peak(dst: Path) -> float | None:
    """Post-encode safety net: AAC can overshoot the limiter ceiling. When `dst`'s true peak
    is above -1.2 dBTP, re-encode only its audio: first through a peak limiter at TP_LIMIT_DB
    (the loudness stays); if that still leaves the peak over -1.2 dBTP, with a volume trim
    instead (from the original) to land at about -1.5. Returns the loudness change in dB
    (None when nothing was done)."""
    m = measure_loudness(dst)
    if m is None or m.tp <= TP_TRIM_ABOVE:
        return None
    tmp = dst.with_suffix(".trim" + dst.suffix)
    lim = LIMITER.format(lim=10 ** (TP_LIMIT_DB / 20))
    _reencode_audio(dst, tmp, f"aresample={LIMIT_RATE},{lim}")
    after = measure_loudness(tmp)
    if after is not None and after.tp <= TP_TRIM_ABOVE:
        os.replace(tmp, dst)
        log.info(
            "audio: true-peak limit %.2f -> TP %.2f, I %.2f -> %.2f", m.tp, after.tp, m.i, after.i
        )
        return round(after.i - m.i, 2)
    trim = TP_TRIM_TARGET - m.tp
    _reencode_audio(dst, tmp, f"volume={trim:.2f}dB")
    os.replace(tmp, dst)
    after = measure_loudness(dst)
    log.info("audio: true-peak trim %.2f dB -> TP %s", trim, f"{after.tp:.2f}" if after else "?")
    return trim
