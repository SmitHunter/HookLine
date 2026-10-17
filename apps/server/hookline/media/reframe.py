"""9:16 reframing helpers."""

import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from hookline.media.ffmpeg import _run, probe_duration, probe_size
from hookline.pipeline.layout import SceneLayout

OUT_W, OUT_H = 1080, 1920
STACK_CAM_H = 640
STACK_GAME_H = OUT_H - STACK_CAM_H  # 1280
SPLIT_H = OUT_H // 2  # 960
TRACK_ALPHA = 0.3
TRACK_DEADBAND = 0.05
BLUR = "boxblur=20:2"

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
_CROP_SAMPLES = 5
_MIN_REMOVED = 0.02


def detect_crop(video: Path) -> tuple[int, int, int, int] | None:
    """Black-bar crop (w, h, x, y) agreed by most of 5 evenly spaced samples, or None
    when it would remove less than 2% of both width and height."""
    width, height = probe_size(video)
    duration = probe_duration(video)
    found: list[tuple[int, int, int, int]] = []
    for i in range(_CROP_SAMPLES):
        t = duration * (i + 0.5) / _CROP_SAMPLES
        res = _run(
            [
                "ffmpeg", "-hide_banner", "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "3", "-vf", "cropdetect=limit=24:round=2", "-f", "null", "-",
            ]
        )  # fmt: skip
        matches = _CROP_RE.findall(res.stderr)
        if matches:
            w, h, x, y = (int(v) for v in matches[-1])
            found.append((w, h, x, y))
    if not found:
        return None
    crop = Counter(found).most_common(1)[0][0]
    if width - crop[0] < _MIN_REMOVED * width and height - crop[1] < _MIN_REMOVED * height:
        return None
    return crop


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


def _aspect_crop(in_w: int, in_h: int, aw: int, ah: int) -> tuple[int, int]:
    """Largest even (w, h) with aspect aw:ah that fits inside in_w x in_h."""
    if in_w * ah >= in_h * aw:
        return _even(in_h * aw / ah), _even(in_h)
    return _even(in_w), _even(in_w * ah / aw)


def _fill(w: int, h: int) -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def _smooth(keyframes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """EMA (alpha 0.3) that only moves when the raw value differs by more than the deadband."""
    out: list[tuple[float, float]] = []
    s = 0.0
    for i, (t, x) in enumerate(sorted(keyframes)):
        if i == 0:
            s = x
        elif abs(x - s) > TRACK_DEADBAND:
            s += TRACK_ALPHA * (x - s)
        out.append((t, s))
    return out


def track_x_expr(keyframes: list[tuple[float, float]], in_w: int, crop_w: int) -> str:
    """ffmpeg expression (in `t`) for the crop's left edge: piecewise-linear over smoothed,
    clamped keyframes (normalized centre x). Commas are NOT escaped."""
    hi = max(in_w - crop_w, 0)
    pts = [(t, min(max(s * in_w - crop_w / 2, 0.0), hi)) for t, s in _smooth(keyframes)]
    if not pts:
        return _fmt(hi / 2)
    expr = _fmt(pts[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(pairwise(pts))):
        if t1 <= t0:
            continue
        lerp = f"{_fmt(x0)}+({_fmt(x1 - x0)})*(t-{_fmt(t0)})/{_fmt(t1 - t0)}"
        expr = f"if(lt(t,{_fmt(t1)}),{lerp},{expr})"
    return f"if(lt(t,{_fmt(pts[0][0])}),{_fmt(pts[0][1])},{expr})"


def _fit_blur(label_in: str, label_out: str) -> str:
    bg, fg, bgo, fgo = (f"[{label_out}_{n}]" for n in ("bg", "fg", "bgo", "fgo"))
    return (
        f"[{label_in}]split=2{bg}{fg};"
        f"{bg}{_fill(OUT_W, OUT_H)},{BLUR}{bgo};"
        f"{fg}scale={OUT_W}:-2{fgo};"
        f"{bgo}{fgo}overlay=(W-w)/2:(H-h)/2,setsar=1[{label_out}]"
    )


def scene_filter(layout: SceneLayout, in_w: int, in_h: int, label_in: str, label_out: str) -> str:
    """Filter chain [label_in] (already trimmed to the scene) -> [label_out] at 1080x1920, SAR 1."""
    kind = layout.kind
    if (kind == "SPLIT" and layout.split_x is None) or (
        kind == "STACKED" and layout.facecam is None
    ):
        kind = "FIT_BLUR"
    a, b = f"[{label_out}_a]", f"[{label_out}_b]"
    if kind == "STACKED":
        assert layout.facecam is not None
        fx, fy, fw, fh = layout.facecam
        cw, ch = _even(fw * in_w), _even(fh * in_h)
        cx = min(max(int(fx * in_w), 0), in_w - cw)
        cy = min(max(int(fy * in_h), 0), in_h - ch)
        gw, gh = _aspect_crop(in_w, in_h, OUT_W, STACK_GAME_H)
        cam, game = f"[{label_out}_cam]", f"[{label_out}_game]"
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={cw}:{ch}:{cx}:{cy},{_fill(OUT_W, STACK_CAM_H)}{cam};"
            f"{b}crop={gw}:{gh}:{(in_w - gw) // 2}:{(in_h - gh) // 2},"
            f"scale={OUT_W}:{STACK_GAME_H}{game};"
            f"{cam}{game}vstack,setsar=1[{label_out}]"
        )
    if kind == "FACE_TRACK":
        cw = min(_even(in_h * 9 / 16), _even(in_w))
        rel = [(t - layout.start, x) for t, x in layout.track_x]
        x = track_x_expr(rel, in_w, cw).replace(",", r"\,")
        return (
            f"[{label_in}]crop={cw}:{_even(in_h)}:{x}:0,scale={OUT_W}:{OUT_H},setsar=1[{label_out}]"
        )
    if kind == "SPLIT":
        assert layout.split_x is not None
        sw, sh = _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)
        y = (in_h - sh) // 2
        xs = [min(max(int(c * in_w - sw / 2), 0), in_w - sw) for c in layout.split_x]
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={sw}:{sh}:{xs[0]}:{y},scale={OUT_W}:{SPLIT_H}[{label_out}_top];"
            f"{b}crop={sw}:{sh}:{xs[1]}:{y},scale={OUT_W}:{SPLIT_H}[{label_out}_bot];"
            f"[{label_out}_top][{label_out}_bot]vstack,setsar=1[{label_out}]"
        )
    return _fit_blur(label_in, label_out)


def reframe_graph(
    layouts: list[SceneLayout],
    in_w: int,
    in_h: int,
    crop: tuple[int, int, int, int] | None = None,
) -> str:
    """Full -filter_complex: [0:v] (crop) -> split -> per-scene trim + scene_filter -> concat [vout].
    When `crop` is given, in_w/in_h are the cropped size."""
    n = len(layouts)
    pre = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else ""
    parts = [f"[0:v]{pre}split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, lay in enumerate(layouts):
        parts.append(
            f"[s{i}]trim=start={_fmt(lay.start)}:end={_fmt(lay.end)},setpts=PTS-STARTPTS[t{i}]"
        )
        parts.append(scene_filter(lay, in_w, in_h, f"t{i}", f"v{i}"))
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]")
    return ";".join(parts)
