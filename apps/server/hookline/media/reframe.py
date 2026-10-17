"""9:16 reframing helpers."""

import math
import re
from collections import Counter
from collections.abc import Sequence
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
MIN_SCENE = 1 / 30

# Pacing zooms: alternate 1.0 / 1.12 at cuts and at least every 4.5 s; punch to 1.2 on
# emphasis beats. Steps ease linearly over 0.2 s; a punch holds 1.2 s and releases over 0.3 s.
ZOOM_IN = 1.12
ZOOM_PUNCH = 1.2
ZOOM_EASE = 0.2
PUNCH_HOLD = 1.2
PUNCH_RELEASE = 0.3
ZOOM_MAX_GAP = 4.5
ZOOM_MIN_STEP = 0.5  # no two steps closer than this
BOUNDARY_SNAP = 1.0  # a filler step moves to a sentence end within this of its nominal time
SHARPEN = "unsharp=5:5:0.6:5:5:0.0"
SHARPEN_ABOVE = 1.3  # sharpen panels upscaled more than this

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
    """EMA (alpha 0.3) chasing a deadbanded target: the target jumps to the raw value only
    when it differs by more than 0.05; the output eases toward the target every step."""
    out: list[tuple[float, float]] = []
    target = smoothed = 0.0
    for i, (t, x) in enumerate(sorted(keyframes)):
        if i == 0:
            target = smoothed = x
        else:
            if abs(x - target) > TRACK_DEADBAND:
                target = x
            smoothed += TRACK_ALPHA * (target - smoothed)
        out.append((t, smoothed))
    return out


def _pwl_expr(pts: list[tuple[float, float]]) -> str:
    """Piecewise-linear ffmpeg expression in `t` through (t, value) points, held constant
    before the first and after the last. Commas are NOT escaped."""
    expr = _fmt(pts[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(pairwise(pts))):
        dt = _fmt(t1 - t0)
        if t1 <= t0 or dt == "0":
            continue  # degenerate segment: would divide by zero once formatted
        if _fmt(x1 - x0) == "0":
            seg = _fmt(x0)
        else:
            seg = f"{_fmt(x0)}+({_fmt(x1 - x0)})*(t-{_fmt(t0)})/{dt}"
        expr = f"if(lt(t,{_fmt(t1)}),{seg},{expr})"
    return f"if(lt(t,{_fmt(pts[0][0])}),{_fmt(pts[0][1])},{expr})"


def track_x_expr(keyframes: list[tuple[float, float]], in_w: int, crop_w: int) -> str:
    """ffmpeg expression (in `t`) for the crop's left edge: piecewise-linear over smoothed,
    clamped keyframes (normalized centre x). Commas are NOT escaped."""
    hi = max(in_w - crop_w, 0)
    pts = [(t, min(max(s * in_w - crop_w / 2, 0.0), hi)) for t, s in _smooth(keyframes)]
    if not pts:
        return _fmt(hi / 2)
    return _pwl_expr(pts)


def _fill_gap(a: float, b: float, max_gap: float, boundaries: Sequence[float]) -> list[float]:
    """Filler step times splitting (a, b) into pieces no longer than `max_gap`; each sits at
    its even-split nominal time, or at the nearest sentence end within BOUNDARY_SNAP of it."""
    out: list[float] = []
    cur = a
    while b - cur > max_gap + 1e-9:
        rest = b - cur
        nominal = cur + rest / math.ceil(rest / max_gap - 1e-9)
        lo = max(nominal - BOUNDARY_SNAP, cur + ZOOM_MIN_STEP)
        hi = min(nominal + BOUNDARY_SNAP, cur + max_gap)
        near = [x for x in boundaries if lo <= x <= hi and b - x >= ZOOM_MIN_STEP]
        cur = min(near, key=lambda x: abs(x - nominal)) if near else nominal
        out.append(cur)
    return out


def zoom_schedule(
    duration: float,
    cut_times: Sequence[float],
    emphasis_times: Sequence[float],
    max_gap: float = ZOOM_MAX_GAP,
    boundaries: Sequence[float] = (),
) -> list[tuple[float, float]]:
    """Zoom steps `(t, scale)` over an output timeline of `duration` s, starting `(0, 1.0)`.

    Consecutive steps always differ: the base level alternates 1.0 / ZOOM_IN at each cut and
    at filler times keeping every gap <= `max_gap` (fillers prefer `boundaries`, i.e. sentence
    ends, within BOUNDARY_SNAP s of their nominal time). An emphasis time punches to ZOOM_PUNCH
    and returns to the base level PUNCH_HOLD s after its ease completes. Punches overlapping an
    earlier punch, and cuts within ZOOM_MIN_STEP s of a punch or of each other, are dropped."""
    punch_len = ZOOM_EASE + PUNCH_HOLD
    punches: list[float] = []
    for e in sorted(emphasis_times):
        if 0 < e < duration and (not punches or e >= punches[-1] + punch_len + ZOOM_MIN_STEP):
            punches.append(e)
    busy = [(p - ZOOM_MIN_STEP, p + punch_len + ZOOM_MIN_STEP) for p in punches]
    cuts: list[float] = []
    for c in sorted(cut_times):
        if not ZOOM_MIN_STEP <= c <= duration - ZOOM_MIN_STEP:
            continue
        if any(lo < c < hi for lo, hi in busy) or (cuts and c - cuts[-1] < ZOOM_MIN_STEP):
            continue
        cuts.append(c)
    events: list[tuple[float, str]] = [(c, "toggle") for c in cuts]
    events += [(p, "punch") for p in punches] + [(p + punch_len, "release") for p in punches]
    anchors = [0.0, *sorted(t for t, _ in events), duration]
    for a, b in pairwise(anchors):
        events += [(f, "toggle") for f in _fill_gap(a, b, max_gap, boundaries)]

    base = 1.0
    steps = [(0.0, 1.0)]
    for t, kind in sorted(events):
        if kind == "toggle":
            base = ZOOM_IN if base == 1.0 else 1.0
        steps.append((round(t, 3), ZOOM_PUNCH if kind == "punch" else base))
    return steps


def zoom_expr(
    steps: Sequence[Sequence[float]], offset: float = 0.0, length: float | None = None
) -> str:
    """Eased scale expression S(t) for `steps`, in scene time (`t` = output time - `offset`).
    A step eases linearly from the previous scale over ZOOM_EASE s (PUNCH_RELEASE s when
    leaving a punch). With `length`, only points that matter for [0, length] are emitted.
    Commas are NOT escaped."""
    if not steps:
        return "1"
    pts = [(float(steps[0][0]), float(steps[0][1]))]
    for (_, s0), (t1, s1) in pairwise(steps):
        d = PUNCH_RELEASE if s0 == ZOOM_PUNCH else ZOOM_EASE
        pts += [(t1, s0), (t1 + d, s1)]
    pts = [(t - offset, v) for t, v in pts]
    if length is not None:
        first = max((i for i, (t, _) in enumerate(pts) if t <= 0), default=0)
        last = min((i for i, (t, _) in enumerate(pts) if t >= length), default=len(pts) - 1)
        pts = pts[first : last + 1]
    if len({_fmt(v) for _, v in pts}) == 1:
        return _fmt(pts[0][1])
    return _pwl_expr(pts)


def _zoom(expr: str, w: int, h: int, ay: float | None = None) -> str:
    """Scale-then-crop zoom of a w x h panel by S(t) = `expr`, keeping (w/2, ay) fixed
    (ay defaults to the centre). Empty when S is constant 1. crop's iw/ih stay at their
    configure-time values when the input size changes per frame, so x/y come from S itself."""
    if expr == "1":
        return ""
    s = "(" + expr.replace(",", r"\,") + ")"
    sw, sh = f"trunc({w}*{s}/2)*2", f"trunc({h}*{s}/2)*2"
    ay = h / 2 if ay is None else min(max(ay, 0.0), float(h))
    x = rf"min({_fmt(w / 2)}*({s}-1)\,{sw}-{w})"
    y = rf"min({_fmt(ay)}*({s}-1)\,{sh}-{h})"
    return f",scale=w='{sw}':h='{sh}':eval=frame,crop={w}:{h}:x='{x}':y='{y}'"


def _sharpen(factor: float) -> str:
    return f",{SHARPEN}" if factor > SHARPEN_ABOVE else ""


def _fit_blur(label_in: str, label_out: str, in_w: int, in_h: int, zoom: str = "1") -> str:
    bg, fg, bgo, fgo = (f"[{label_out}_{n}]" for n in ("bg", "fg", "bgo", "fgo"))
    fh = _even(OUT_W * in_h / in_w)
    return (
        f"[{label_in}]split=2{bg}{fg};"
        f"{bg}{_fill(OUT_W, OUT_H)},{BLUR}{bgo};"
        f"{fg}scale={OUT_W}:{fh}{_sharpen(OUT_W / in_w)}{_zoom(zoom, OUT_W, fh)}{fgo};"
        f"{bgo}{fgo}overlay=(W-w)/2:(H-h)/2,setsar=1[{label_out}]"
    )


def _kind(layout: SceneLayout) -> str:
    """The layout kind actually rendered: SPLIT/STACKED without geometry fall back to FIT_BLUR."""
    if (layout.kind == "SPLIT" and layout.split_x is None) or (
        layout.kind == "STACKED" and layout.facecam is None
    ):
        return "FIT_BLUR"
    return layout.kind


def _cam_crop(layout: SceneLayout, in_w: int, in_h: int) -> tuple[int, int, int, int]:
    assert layout.facecam is not None
    fx, fy, fw, fh = layout.facecam
    cw, ch = _even(fw * in_w), _even(fh * in_h)
    cx = min(max(int(fx * in_w), 0), in_w - cw)
    cy = min(max(int(fy * in_h), 0), in_h - ch)
    return cw, ch, cx, cy


def _track_w(in_w: int, in_h: int) -> int:
    return min(_even(in_h * 9 / 16), _even(in_w))


def scene_upscale(layout: SceneLayout, in_w: int, in_h: int) -> float:
    """Largest source-to-output upscale factor among the scene's panels (zoom excluded)."""
    kind = _kind(layout)
    if kind == "STACKED":
        cw, ch, _, _ = _cam_crop(layout, in_w, in_h)
        gw, _ = _aspect_crop(in_w, in_h, OUT_W, STACK_GAME_H)
        return max(OUT_W / cw, STACK_CAM_H / ch, OUT_W / gw)
    if kind == "FACE_TRACK":
        return max(OUT_W / _track_w(in_w, in_h), OUT_H / _even(in_h))
    if kind == "SPLIT":
        return OUT_W / _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)[0]
    return OUT_W / in_w


def scene_filter(
    layout: SceneLayout,
    in_w: int,
    in_h: int,
    label_in: str,
    label_out: str,
    zoom_steps: Sequence[Sequence[float]] | None = None,
) -> str:
    """Filter chain [label_in] (already trimmed to the scene) -> [label_out] at 1080x1920, SAR 1.

    `zoom_steps` (output time, from `zoom_schedule`) zoom the speaker panel after the layout:
    FACE_TRACK whole frame (anchored on the face centre), STACKED facecam panel, both SPLIT
    panels, FIT_BLUR foreground. Panels upscaled more than SHARPEN_ABOVE are sharpened."""
    kind = _kind(layout)
    zoom = (
        zoom_expr(zoom_steps, offset=layout.start, length=layout.end - layout.start)
        if zoom_steps
        else "1"
    )
    a, b = f"[{label_out}_a]", f"[{label_out}_b]"
    if kind == "STACKED":
        cw, ch, cx, cy = _cam_crop(layout, in_w, in_h)
        gw, gh = _aspect_crop(in_w, in_h, OUT_W, STACK_GAME_H)
        cam, game = f"[{label_out}_cam]", f"[{label_out}_game]"
        cam_up = max(OUT_W / cw, STACK_CAM_H / ch)
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={cw}:{ch}:{cx}:{cy},{_fill(OUT_W, STACK_CAM_H)}{_sharpen(cam_up)}"
            f"{_zoom(zoom, OUT_W, STACK_CAM_H)}{cam};"
            f"{b}crop={gw}:{gh}:{(in_w - gw) // 2}:{(in_h - gh) // 2},"
            f"scale={OUT_W}:{STACK_GAME_H}{_sharpen(OUT_W / gw)}{game};"
            f"{cam}{game}vstack,setsar=1[{label_out}]"
        )
    if kind == "FACE_TRACK":
        cw = _track_w(in_w, in_h)
        rel = [(t - layout.start, x) for t, x in layout.track_x]
        x = track_x_expr(rel, in_w, cw).replace(",", r"\,")
        ay = None
        if layout.face_box is not None:
            fy, fh = layout.face_box
            ay = (fy + fh / 2) * OUT_H
        return (
            f"[{label_in}]crop={cw}:{_even(in_h)}:{x}:0,scale={OUT_W}:{OUT_H}"
            f"{_sharpen(scene_upscale(layout, in_w, in_h))}{_zoom(zoom, OUT_W, OUT_H, ay)},"
            f"setsar=1[{label_out}]"
        )
    if kind == "SPLIT":
        assert layout.split_x is not None
        sw, sh = _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)
        y = (in_h - sh) // 2
        xs = [min(max(int(c * in_w - sw / 2), 0), in_w - sw) for c in layout.split_x]
        post = f"{_sharpen(OUT_W / sw)}{_zoom(zoom, OUT_W, SPLIT_H)}"
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={sw}:{sh}:{xs[0]}:{y},scale={OUT_W}:{SPLIT_H}{post}[{label_out}_top];"
            f"{b}crop={sw}:{sh}:{xs[1]}:{y},scale={OUT_W}:{SPLIT_H}{post}[{label_out}_bot];"
            f"[{label_out}_top][{label_out}_bot]vstack,setsar=1[{label_out}]"
        )
    return _fit_blur(label_in, label_out, in_w, in_h, zoom)


def _merge_short(layouts: list[SceneLayout]) -> list[SceneLayout]:
    """Fold scenes shorter than MIN_SCENE into a neighbour without leaving gaps: a short scene
    extends the previous scene's end; leading short scenes pull the next scene's start back."""
    merged: list[SceneLayout] = []
    lead: float | None = None
    for lay in layouts:
        if lay.end - lay.start < MIN_SCENE:
            if merged:
                merged[-1] = merged[-1].model_copy(update={"end": lay.end})
            elif lead is None:
                lead = lay.start
            continue
        if lead is not None:
            lay = lay.model_copy(update={"start": lead})
            lead = None
        merged.append(lay)
    if not merged and layouts and layouts[-1].end - layouts[0].start >= MIN_SCENE:
        merged = [layouts[0].model_copy(update={"end": layouts[-1].end})]
    if not merged:
        raise ValueError("no layouts")
    return merged


def reframe_graph(
    layouts: list[SceneLayout],
    in_w: int,
    in_h: int,
    crop: tuple[int, int, int, int] | None = None,
    zoom_steps: Sequence[Sequence[float]] | None = None,
) -> str:
    """Full -filter_complex: [0:v] (crop) -> split -> per-scene trim + scene_filter -> concat [vout].
    When `crop` is given, in_w/in_h are the cropped size. Scenes shorter than 1/30 s are merged
    into a neighbour so the timeline keeps its full length. `zoom_steps` (output time) are
    applied per scene (see `scene_filter`)."""
    layouts = _merge_short(layouts)
    n = len(layouts)
    pre = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else ""
    parts = [f"[0:v]{pre}split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, lay in enumerate(layouts):
        parts.append(
            f"[s{i}]trim=start={_fmt(lay.start)}:end={_fmt(lay.end)},setpts=PTS-STARTPTS[t{i}]"
        )
        parts.append(scene_filter(lay, in_w, in_h, f"t{i}", f"v{i}", zoom_steps))
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]")
    return ";".join(parts)
