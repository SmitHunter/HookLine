"""9:16 reframing helpers."""

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

from hookline.media.ffmpeg import FFmpegError, _run, probe_duration, probe_size
from hookline.pipeline.layout import ZOOM_TOP, SceneLayout
from hookline.vision.faces import sample_frames

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
STACK_CAM_H = 640
STACK_GAME_H = OUT_H - STACK_CAM_H  # 1280
SPLIT_H = OUT_H // 2  # 960
TRACK_ALPHA = 0.3
TRACK_DEADBAND = 0.05
TRACK_MARGIN = 0.1  # the face stays this share of the crop width inside the crop's edges
BLUR = "boxblur=20:2"
MIN_SCENE = 1 / 30

# Pacing zooms: alternate 1.0 / 1.12 at every cut and at least every 4.5 s; punch to 1.2 on
# emphasis beats. Steps ease (smoothstep) over 0.2 s; a punch holds 1.2 s and releases over 0.3 s.
ZOOM_IN = 1.12
ZOOM_PUNCH = ZOOM_TOP  # 1.2
ZOOM_EASE = 0.2
PUNCH_HOLD = 1.2
PUNCH_RELEASE = 0.3
ZOOM_MAX_GAP = 4.5
ZOOM_MIN_STEP = 0.5  # no two steps closer than this
BOUNDARY_SNAP = 1.0  # a filler step moves to a sentence end within this of its nominal time
SHARPEN = "unsharp=5:5:0.6:5:5:0.0"
SHARPEN_ABOVE = 1.3  # sharpen panels upscaled more than this
# Per-scene zoom plan (plan_zoom). Soft sources: a scene upscaled more than UPSCALE_SOFT keeps
# every zoom level (punches included) at or under ZOOM_IN and scales with lanczos.
UPSCALE_SOFT = 2.0
LANCZOS = ":flags=lanczos"
# Small faces: a FACE_TRACK face under SMALL_FACE of the output height gets a base zoom toward
# FACE_TARGET, within MAX_UPSCALE total upscale and MAX_ZOOM total zoom.
SMALL_FACE = 0.20
FACE_TARGET = 0.22
MAX_UPSCALE = 2.5
MAX_ZOOM = 1.4
HEADROOM = 0.04  # the head top (face top - half a face height) stays this share below the top
_CAP_STEP = 0.005

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


# Burned-in overlays (lower-thirds, alert boxes, sub goals): edge-touching bands of rows whose
# edges stay put while the rest of the frame moves. Measured on 12 grayscale samples, 320 px wide.
_OVERLAY_SAMPLES = 12
_OVERLAY_W = 320
_EDGE_T = 20  # gradient (max of |dx|, |dy|) above this is an edge pixel
_EDGE_TOL = 0.3  # an edge is static when within 30% of the median frame's gradient there
_EDGE_DENSITY = 0.08  # a row is static in a sample when > 8% of its pixels are static edges
_STATIC_SHARE = 0.9  # ... in at least 90% of the samples
_BAND_MARGIN = 0.08  # a band's outermost static row lies within 8% of the frame edge
_BAND_GAP = 0.03  # flat rows inside a band (between text lines) bridged up to 3% of height
_BAND_SLIT = 0.015  # ... and any rows (descenders, a border) up to 1.5%
_BAND_MIN = 0.03  # at least 3% of the height in static rows
_BAND_PAD = 0.01  # cut 1% past the last static row: borders/shadows whose contrast varies
_BORDER_DENSITY = 0.2  # a row with edges across >= 20% of the width is a box border
_BOX_T = 8  # a vertical step this big marks a box's border (faint on dim footage) ...
_BOX_STEADY = 0.5  # ... when present in >= half the samples (semi-transparent boxes fade)
_BOX_SPAN = 0.6  # ... along one run (gaps <= 2% of the width bridged) of >= 60% of the width
_BOX_GAP = 0.02
_BOX_BORDER = 0.5  # ... of which a visible border covers >= 50% of the width
_BOX_FILL_SPREAD = 30  # a column inside the box whose brightness spread (p90-p10) stays under this
_BOX_CLEAN = 0.5  # >= half the rows inside the box carry no live (non-static) edges
# The two 30% limits have different bases: a band is measured against the sampled frame (the
# letterbox crop); combine_crop re-checks the total against that crop's height, and also keeps
# >= 60% of the original (uncropped) frame height.
_MAX_BAND = 0.30  # never more than 30% of the height, per band and in total
_MIN_KEPT = 0.6  # letterbox + bands must leave >= 60% of the original height
_MIN_MOTION = 2.0  # 90th-percentile row motion outside the bands; below it nothing moves


def _gradient(gray: np.ndarray) -> np.ndarray:
    """max(|dx|, |dy|) per pixel; a vertical step is attributed to the row below it, so a
    box's top border belongs to the box."""
    grad = np.zeros_like(gray)
    grad[..., :, :-1] = np.abs(np.diff(gray, axis=-1))
    grad[..., 1:, :] = np.maximum(grad[..., 1:, :], np.abs(np.diff(gray, axis=-2)))
    return grad


def _row_flags(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row (n, H, W -> H): `static` -- in >= 90% of samples > 8% of its pixels are edges
    matching the median gradient; `flat` -- the median frame has almost no edges there (the
    plain fill between an overlay's text lines); `strong` -- an edge across >= 20% of the median
    frame's width (a box border); plus `live` (n, H, W): edges that don't match the median
    gradient, i.e. moving picture. Gradients, not intensities: a semi-transparent band shifts
    in brightness with what is behind it, but its text edges stay put."""
    grad = _gradient(gray)
    med = np.median(grad, axis=0)
    edge = grad > _EDGE_T
    match = edge & (med > _EDGE_T) & (np.abs(grad - med) < _EDGE_TOL * np.maximum(grad, med))
    static = ((match.mean(axis=2) > _EDGE_DENSITY).mean(axis=0)) >= _STATIC_SHARE
    density = (med > _EDGE_T).mean(axis=1)
    return static, density < _EDGE_DENSITY / 2, density >= _BORDER_DENSITY, edge & ~match


def _longest_run(mask: np.ndarray, gap: int) -> tuple[int, int]:
    """[lo, hi) of the longest run of True in `mask`, bridging False gaps up to `gap`."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0, 0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > gap + 1) + 1)
    best = max(runs, key=lambda r: r[-1] - r[0])
    return int(best[0]), int(best[-1]) + 1


def _boxed(gray: np.ndarray, live: np.ndarray, band: int, start: int) -> bool:
    """Whether a band (rows [0, band) of edge-first (n, H, W) arrays; outermost static row
    `start`) is shaped like a boxed overlay: its inner edge is a border row -- a vertical step
    present in >= half the samples (its contrast varies with what is behind it), or a column of
    steady fill below it -- running along >= 60% of the width in one piece, the visible border
    covering >= 50% of the width, and >= half the rows between `start`
    and that border carry no live edges within the box's columns (text on a fill, not a
    picture). Separate HUD bars (two health bars + a timer) and cam frames fail."""
    _, h, w = gray.shape
    dy = np.zeros_like(gray)
    dy[:, 1:] = np.abs(np.diff(gray, axis=1))
    reach = math.ceil((_BAND_GAP + _BAND_PAD) * h) + 2
    best = (0, 0, 0)  # (length, lo, row)
    for r in range(max(band - reach, start + 1), min(band + 1, h - 1)):
        steady = (dy[:, r - 1 : r + 2].max(axis=1) > _BOX_T).mean(axis=0) >= _BOX_STEADY
        # Where the border fades into a same-brightness background, the box still shows as
        # columns whose interior barely changes (a fill), unlike a moving picture between bars.
        inside = gray[:, start:r]
        spread = np.percentile(inside, 90, axis=0) - np.percentile(inside, 10, axis=0)
        boxy = np.median(spread, axis=0) < _BOX_FILL_SPREAD
        lo, hi = _longest_run(steady | boxy, math.ceil(_BOX_GAP * w))
        if steady[lo:hi].sum() >= _BOX_BORDER * w and hi - lo > best[0]:
            best = (hi - lo, lo, r)
    length, lo, row = best
    if length < _BOX_SPAN * w:
        return False
    rows = np.median(live[:, start:row, lo : lo + length].mean(axis=2), axis=0)
    return bool(rows.size) and bool((rows < _EDGE_DENSITY / 2).mean() >= _BOX_CLEAN)


def _edge_band(static: np.ndarray, flat: np.ndarray, strong: np.ndarray) -> tuple[int, int]:
    """(height in rows, outermost static row) of the band touching index 0 (flags ordered edge
    first), or (0, 0).

    Candidate runs start at each static row within the margin (live content may sit between
    the band and the frame edge) and extend over static rows; gaps up to _BAND_SLIT are
    bridged, up to _BAND_GAP only when flat (a wider gap with edges that aren't static is live
    content: the band ends). The longest run wins. Its end moves out to a box border (`strong`
    row) within _BAND_GAP, whose contrast varies with what is behind it, plus _BAND_PAD. Over
    _MAX_BAND it is not a band."""
    h = len(static)
    hits = np.flatnonzero(static)
    best_last, best_count, best_start = -1, 0, 0
    for start in hits[hits <= _BAND_MARGIN * h]:
        last, count = int(start), 0
        for i in hits[hits >= start]:
            gap = i - last - 1
            if gap > _BAND_GAP * h or (gap > _BAND_SLIT * h and not flat[last + 1 : i].all()):
                break
            last, count = int(i), count + 1
        if count > best_count:
            best_last, best_count, best_start = last, count, int(start)
    if best_count < _BAND_MIN * h:
        return 0, 0
    window = np.flatnonzero(strong[best_last + 1 : best_last + 1 + math.ceil(_BAND_GAP * h)])
    if len(window):
        best_last += int(window[0]) + 1
    band = min(best_last + 1 + math.ceil(_BAND_PAD * h), h)
    return (band, best_start) if band <= _MAX_BAND * h else (0, 0)


def detect_overlay_bands(
    video: Path, crop: tuple[int, int, int, int] | None = None
) -> tuple[int, int] | None:
    """Burned-in overlay bands at the top/bottom edge as `(top_cut_px, bottom_cut_px)` in
    (cropped) source pixels, or None. A band is a run of rows whose edges are static in >= 90%
    of 12 samples while the rest of the frame moves, touching the frame edge, <= 30% of the
    height, shaped like a boxed overlay (`_boxed`). A paused video (nothing moves), a static
    frame-sized region, a corner minimap, separate HUD bars or a cam frame is not a band. An
    unreadable video gives None (logged)."""
    try:
        duration = probe_duration(video)
        times = [duration * (i + 0.5) / _OVERLAY_SAMPLES for i in range(_OVERLAY_SAMPLES)]
        frames = sample_frames(video, times, width=_OVERLAY_W, crop=crop)
        src_h = crop[1] if crop else probe_size(video)[1]
    except FFmpegError as e:
        log.warning("overlay detection skipped for %s: %s", video.name, e)
        return None
    gray = np.stack(
        [f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299 for f in frames]
    ).astype(np.float32)
    h = gray.shape[1]
    static, flat, strong, live = _row_flags(gray)
    top, t0 = _edge_band(static, flat, strong)
    bottom, b0 = _edge_band(static[::-1], flat[::-1], strong[::-1])
    top = top if top and _boxed(gray, live, top, t0) else 0
    bottom = bottom if bottom and _boxed(gray[:, ::-1], live[:, ::-1], bottom, b0) else 0
    if not top and not bottom:
        return None
    if top + bottom > _MAX_BAND * h:
        log.info("overlay bands %d+%d of %d rows exceed 30%%; not cropped", top, bottom, h)
        return None
    motion = np.median(np.abs(gray - np.median(gray, axis=0)).mean(axis=2), axis=0)
    if np.percentile(motion[top : h - bottom], 90) < _MIN_MOTION:
        return None  # nothing moves: an overlay can't be told from the picture
    return _even_up(top * src_h / h) if top else 0, _even_up(bottom * src_h / h) if bottom else 0


def _even_up(v: float) -> int:
    return math.ceil(v / 2) * 2


def combine_crop(
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    bands: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    """The letterbox `crop` (or the full `size` frame) shrunk in height to exclude overlay
    `bands` (top, bottom px in crop space), even dimensions. Band removal is refused (logged),
    returning the letterbox crop unchanged, when it would remove more than 30% of the crop's
    height or leave less than 60% of the original frame's height."""
    if not bands or not any(bands):
        return crop
    w, h, x, y = crop or (size[0], size[1], 0, 0)
    top, bottom = _even_up(bands[0]), bands[1]
    if top + bottom > _MAX_BAND * h or h - top - bottom < _MIN_KEPT * size[1]:
        log.info(
            "overlay bands %s on %d of %d px rows removed too much; skipped", bands, h, size[1]
        )
        return crop
    return w, _even(h - top - bottom), x, y + top


def box_to_crop(
    box: tuple[float, float, float, float],
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
) -> tuple[float, float, float, float]:
    """A box (x, y, w, h as fractions of the full `size` frame) in `crop` space, clipped to it:
    x' = (x * W - crop_x) / crop_w, and likewise for y, w, h."""
    if crop is None:
        return box
    cw, ch, cx, cy = crop
    x0 = min(max((box[0] * size[0] - cx) / cw, 0.0), 1.0)
    y0 = min(max((box[1] * size[1] - cy) / ch, 0.0), 1.0)
    x1 = min(max(((box[0] + box[2]) * size[0] - cx) / cw, 0.0), 1.0)
    y1 = min(max(((box[1] + box[3]) * size[1] - cy) / ch, 0.0), 1.0)
    return x0, y0, x1 - x0, y1 - y0


def box_hits_bands(
    box: tuple[float, float, float, float],
    size: tuple[int, int],
    letterbox: tuple[int, int, int, int] | None,
    bands: tuple[int, int],
) -> bool:
    """Whether a box (fractions of the full `size` frame) overlaps the rows overlay `bands`
    (top, bottom px in `letterbox` space) would remove."""
    _, h, _, y = letterbox or (size[0], size[1], 0, 0)
    y0, y1 = box[1] * size[1], (box[1] + box[3]) * size[1]
    top, bottom = bands
    hits_top = top > 0 and y0 < y + top and y1 > y
    hits_bottom = bottom > 0 and y1 > y + h - bottom and y0 < y + h
    return hits_top or hits_bottom


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


def _aspect_crop(in_w: int, in_h: int, aw: int, ah: int) -> tuple[int, int]:
    """Largest even (w, h) with aspect aw:ah that fits inside in_w x in_h."""
    if in_w * ah >= in_h * aw:
        return _even(in_h * aw / ah), _even(in_h)
    return _even(in_w), _even(in_w * ah / aw)


def _fill(w: int, h: int, flags: str = "") -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=increase{flags},crop={w}:{h}"


def _smooth(
    keyframes: list[tuple[float, float]], allow: float | None = None
) -> list[tuple[float, float]]:
    """EMA (alpha 0.3) chasing a deadbanded target: the target jumps to the raw value only
    when it differs by more than 0.05; the output eases toward the target every step. With
    `allow`, the output never lags more than that behind the raw value (the face stays in the
    crop when it moves fast, e.g. a lean)."""
    out: list[tuple[float, float]] = []
    target = smoothed = 0.0
    for i, (t, x) in enumerate(sorted(keyframes)):
        if i == 0:
            target = smoothed = x
        else:
            if abs(x - target) > TRACK_DEADBAND:
                target = x
            smoothed += TRACK_ALPHA * (target - smoothed)
        if allow is not None:
            smoothed = min(max(smoothed, x - allow), x + allow)
        out.append((t, smoothed))
    return out


def _pwl_expr(pts: list[tuple[float, float]], smooth: bool = False) -> str:
    """Piecewise ffmpeg expression in `t` through (t, value) points, held constant before the
    first and after the last. Segments are linear, or smoothstep (u^2 (3 - 2u)) when `smooth`.
    Commas are NOT escaped."""
    expr = _fmt(pts[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(pairwise(pts))):
        dt = _fmt(t1 - t0)
        if t1 <= t0 or dt == "0":
            continue  # degenerate segment: would divide by zero once formatted
        if _fmt(x1 - x0) == "0":
            seg = _fmt(x0)
        elif smooth:
            u = f"((t-{_fmt(t0)})/{dt})"
            seg = f"{_fmt(x0)}+({_fmt(x1 - x0)})*{u}*{u}*(3-2*{u})"
        else:
            seg = f"{_fmt(x0)}+({_fmt(x1 - x0)})*(t-{_fmt(t0)})/{dt}"
        expr = f"if(lt(t,{_fmt(t1)}),{seg},{expr})"
    return f"if(lt(t,{_fmt(pts[0][0])}),{_fmt(pts[0][1])},{expr})"


def track_x_expr(
    keyframes: list[tuple[float, float]], in_w: int, crop_w: int, face_w: float | None = None
) -> str:
    """ffmpeg expression (in `t`) for the crop's left edge: piecewise-linear over smoothed,
    clamped keyframes (normalized centre x). The smoothing never lets the face (`face_w`,
    normalized; 0 when unknown) of a keyframe come closer than TRACK_MARGIN of the crop width
    to the crop's edge. Commas are NOT escaped."""
    hi = max(in_w - crop_w, 0)
    room = crop_w / 2 - (face_w or 0.0) * in_w / 2 - TRACK_MARGIN * crop_w
    allow = max(room, 0.0) / in_w
    pts = [(t, min(max(s * in_w - crop_w / 2, 0.0), hi)) for t, s in _smooth(keyframes, allow)]
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

    Every cut in (0, duration) gets a step at its exact time, so no cut is a bare jump cut.
    The base level alternates 1.0 / ZOOM_IN at each cut and at filler times keeping every gap
    <= `max_gap` (fillers prefer `boundaries`, i.e. sentence ends, within BOUNDARY_SNAP s of
    their nominal time). An emphasis time punches to ZOOM_PUNCH and returns to the base level
    PUNCH_HOLD s after its ease completes. A punch within ZOOM_MIN_STEP s of a cut moves onto
    the cut (the punch is that cut's framing change); a cut during a punch (or within
    ZOOM_MIN_STEP s after its release) ends the punch there, at 1.0, the level furthest from
    the punch. Punches overlapping an earlier punch, or before ZOOM_MIN_STEP s, are dropped.
    Consecutive steps always differ."""
    punch_len = ZOOM_EASE + PUNCH_HOLD
    cuts = sorted({round(c, 3) for c in cut_times if 0 < c < duration})
    punches: list[float] = []
    for e in sorted(emphasis_times):
        if not ZOOM_MIN_STEP <= e < duration:
            continue
        near = [c for c in cuts if abs(c - e) < ZOOM_MIN_STEP]
        if near:
            e = min(near, key=lambda c: abs(c - e))
        if not punches or e >= punches[-1] + punch_len + ZOOM_MIN_STEP:
            punches.append(e)
    events: list[tuple[float, str]] = []
    taken: set[float] = set()
    for p in punches:
        events.append((p, "punch"))
        taken.add(p)
        ender = [c for c in cuts if p < c < p + punch_len + ZOOM_MIN_STEP]
        if ender:
            events.append((ender[0], "cut_release"))
            taken.add(ender[0])
        else:
            events.append((p + punch_len, "release"))
    events += [(c, "toggle") for c in cuts if c not in taken]
    anchors = sorted({0.0, *(t for t, _ in events), duration})
    for a, b in pairwise(anchors):
        events += [(f, "toggle") for f in _fill_gap(a, b, max_gap, boundaries)]

    base = 1.0
    steps = [(0.0, 1.0)]
    for t, kind in sorted(events):
        if kind == "toggle":
            base = ZOOM_IN if base == 1.0 else 1.0
        elif kind == "cut_release":
            base = 1.0
        steps.append((round(t, 3), ZOOM_PUNCH if kind == "punch" else base))
    return [st for st in steps if st[0] < duration]  # e.g. a release past the end


def zoom_expr(
    steps: Sequence[Sequence[float]],
    offset: float = 0.0,
    length: float | None = None,
    level: Callable[[float], float] | None = None,
) -> str:
    """Eased scale expression S(t) for `steps`, in scene time (`t` = output time - `offset`).
    A step eases (smoothstep) from the previous scale over ZOOM_EASE s (PUNCH_RELEASE s when
    leaving a punch), cut short when the next step comes sooner. `level` maps a step's scale
    to the rendered zoom (a scene's `SceneLayout.zoom_level`). With `length`, only points that
    matter for [0, length] are emitted. Commas are NOT escaped."""
    if not steps:
        return "1"
    lv = level or (lambda v: v)
    seq = [(float(t), float(v)) for t, v in steps]
    pts = [(seq[0][0], lv(seq[0][1]))]
    for i, ((_, s0), (t1, s1)) in enumerate(pairwise(seq)):
        d = PUNCH_RELEASE if s0 == ZOOM_PUNCH else ZOOM_EASE
        nxt = seq[i + 2][0] if i + 2 < len(seq) else math.inf
        pts += [(t1, lv(s0)), (min(t1 + d, nxt), lv(s1))]
    pts = [(t - offset, v) for t, v in pts]
    if length is not None:
        first = max((i for i, (t, _) in enumerate(pts) if t <= 0), default=0)
        last = min((i for i, (t, _) in enumerate(pts) if t >= length), default=len(pts) - 1)
        pts = pts[first : last + 1]
    if len({_fmt(v) for _, v in pts}) == 1:
        return _fmt(pts[0][1])
    return _pwl_expr(pts, smooth=True)


def _zoom(expr: str, w: int, h: int, ay: float | None = None, flags: str = "") -> str:
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
    return f",scale=w='{sw}':h='{sh}':eval=frame{flags},crop={w}:{h}:x='{x}':y='{y}'"


def _sharpen(factor: float) -> str:
    return f",{SHARPEN}" if factor > SHARPEN_ABOVE else ""


def _fit_blur(
    label_in: str, label_out: str, in_w: int, in_h: int, zoom: str = "1", flags: str = ""
) -> str:
    bg, fg, bgo, fgo = (f"[{label_out}_{n}]" for n in ("bg", "fg", "bgo", "fgo"))
    fh = _even(OUT_W * in_h / in_w)
    return (
        f"[{label_in}]split=2{bg}{fg};"
        f"{bg}{_fill(OUT_W, OUT_H)},{BLUR}{bgo};"
        f"{fg}scale={OUT_W}:{fh}{flags}{_zoom(zoom, OUT_W, fh, flags=flags)}"
        f"{_sharpen(OUT_W / in_w)}{fgo};"
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

    `zoom_steps` (output time, from `zoom_schedule`) zoom the speaker panel after the layout,
    each level mapped through the scene's plan (`plan_zoom`, `SceneLayout.zoom_level`) and
    scaled about its anchor row (`SceneLayout.anchor_px`): FACE_TRACK whole frame, STACKED
    facecam panel, both SPLIT panels, FIT_BLUR foreground. Panels upscaled more than
    SHARPEN_ABOVE are sharpened after the zoom, so the final pixels are the sharpened ones;
    scenes upscaled more than UPSCALE_SOFT scale with lanczos."""
    kind = _kind(layout)
    if zoom_steps:
        zoom = zoom_expr(
            zoom_steps,
            offset=layout.start,
            length=layout.end - layout.start,
            level=layout.zoom_level,
        )
    else:
        zoom = _fmt(layout.zoom_level(1.0))
    flags = LANCZOS if scene_upscale(layout, in_w, in_h) > UPSCALE_SOFT else ""
    a, b = f"[{label_out}_a]", f"[{label_out}_b]"
    if kind == "STACKED":
        cw, ch, cx, cy = _cam_crop(layout, in_w, in_h)
        gw, gh = _aspect_crop(in_w, in_h, OUT_W, STACK_GAME_H)
        cam, game = f"[{label_out}_cam]", f"[{label_out}_game]"
        cam_up = max(OUT_W / cw, STACK_CAM_H / ch)
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={cw}:{ch}:{cx}:{cy},{_fill(OUT_W, STACK_CAM_H, flags)}"
            f"{_zoom(zoom, OUT_W, STACK_CAM_H, flags=flags)}{_sharpen(cam_up)}{cam};"
            f"{b}crop={gw}:{gh}:{(in_w - gw) // 2}:{(in_h - gh) // 2},"
            f"scale={OUT_W}:{STACK_GAME_H}{flags}{_sharpen(OUT_W / gw)}{game};"
            f"{cam}{game}vstack,setsar=1[{label_out}]"
        )
    if kind == "FACE_TRACK":
        cw = _track_w(in_w, in_h)
        rel = [(t - layout.start, x) for t, x in layout.track_x]
        x = track_x_expr(rel, in_w, cw, layout.face_w).replace(",", r"\,")
        ay = layout.anchor_px(OUT_H)
        return (
            f"[{label_in}]crop={cw}:{_even(in_h)}:{x}:0,scale={OUT_W}:{OUT_H}{flags}"
            f"{_zoom(zoom, OUT_W, OUT_H, ay, flags)}{_sharpen(scene_upscale(layout, in_w, in_h))},"
            f"setsar=1[{label_out}]"
        )
    if kind == "SPLIT":
        assert layout.split_x is not None
        sw, sh = _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)
        y = (in_h - sh) // 2
        xs = [min(max(int(c * in_w - sw / 2), 0), in_w - sw) for c in layout.split_x]
        post = f"{_zoom(zoom, OUT_W, SPLIT_H, layout.anchor_px(SPLIT_H), flags)}"
        post += _sharpen(OUT_W / sw)
        scale = f"scale={OUT_W}:{SPLIT_H}{flags}"
        return (
            f"[{label_in}]split=2{a}{b};"
            f"{a}crop={sw}:{sh}:{xs[0]}:{y},{scale}{post}[{label_out}_top];"
            f"{b}crop={sw}:{sh}:{xs[1]}:{y},{scale}{post}[{label_out}_bot];"
            f"[{label_out}_top][{label_out}_bot]vstack,setsar=1[{label_out}]"
        )
    return _fit_blur(label_in, label_out, in_w, in_h, zoom, flags)


def _panel_y(layout: SceneLayout, y: float, in_w: int, in_h: int) -> tuple[float, float]:
    """(unzoomed panel px, panel height) of normalized source row `y`: the full frame height
    for FACE_TRACK, the vertically centred crop for SPLIT."""
    if _kind(layout) == "SPLIT":
        sh = _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)[1]
        return (y * in_h - (in_h - sh) // 2) / sh * SPLIT_H, SPLIT_H
    return y * OUT_H, OUT_H


def _headroom(layout: SceneLayout, in_w: int, in_h: int) -> tuple[float, float] | None:
    """(largest zoom <= the scene's top level, anchor px) that keeps the head top (the
    scene's highest, `head_top`) HEADROOM below the panel top and the median chin inside the
    panel at every level up to it, the anchor as close to the face centre as allowed. None
    without face geometry. A head already within HEADROOM unzoomed (the source framing cut it)
    anchors at the top: zooming in then only moves it down."""
    if layout.face_box is None or layout.head_top is None:
        return None
    if _kind(layout) not in ("FACE_TRACK", "SPLIT"):
        return None
    fy, fh = layout.face_box
    hp, ph = _panel_y(layout, layout.head_top, in_w, in_h)
    fc, _ = _panel_y(layout, fy + fh / 2, in_w, in_h)
    fb, _ = _panel_y(layout, fy + fh, in_w, in_h)
    m = HEADROOM * ph
    s = layout.zoom_level(ZOOM_TOP)
    while s > 1.0 + 1e-9:
        # zoom s about anchor a maps row y to a + s (y - a): head top >= m and chin <= ph
        hi = max(hp + (hp - m) / (s - 1), 0.0)
        lo = max((s * fb - ph) / (s - 1), 0.0)
        if lo <= hi:
            return s, min(max(fc, lo), hi)
        s -= _CAP_STEP
    return 1.0, min(max(fc, 0.0), ph)


def plan_zoom(layout: SceneLayout, in_w: int, in_h: int) -> SceneLayout:
    """The scene with its zoom plan (base_zoom, zoom_cap, zoom_anchor) filled in:

    - soft source (upscale > UPSCALE_SOFT): every level, punches included, <= ZOOM_IN;
    - small FACE_TRACK face (< SMALL_FACE of the output height): a base zoom toward
      FACE_TARGET, with every level <= MAX_ZOOM total zoom and MAX_UPSCALE total upscale,
      leaving room for the full step range above the base (cut steps and punches keep their
      size);
    - headroom: the anchor row (and if need be a lower cap) keeps the head top >= HEADROOM of
      the panel height below its top at every level (`_headroom`)."""
    up = scene_upscale(layout, in_w, in_h)
    cap: float | None = ZOOM_IN if up > UPSCALE_SOFT else None
    base = 1.0
    face_h = layout.face_box[1] if layout.face_box else 0.0
    if _kind(layout) == "FACE_TRACK" and 0 < face_h < SMALL_FACE:
        room = min(MAX_ZOOM, MAX_UPSCALE / up, cap or MAX_ZOOM)
        base = max(1.0, min(FACE_TARGET / face_h, room / ZOOM_PUNCH))
        if base > 1.0:
            cap = room
    plan = layout.model_copy(update={"base_zoom": math.floor(base * 1e4) / 1e4, "zoom_cap": cap})
    head = _headroom(plan, in_w, in_h)
    if head is None:
        return plan
    top, anchor = head
    if top < plan.zoom_level(ZOOM_TOP) - 1e-9:
        base, cap = max(1.0, min(base, top / ZOOM_PUNCH)), top
    ph = _panel_y(layout, 0.0, in_w, in_h)[1]
    return plan.model_copy(
        update={
            "base_zoom": math.floor(base * 1e4) / 1e4,
            "zoom_cap": cap,
            "zoom_anchor": anchor / ph,
        }
    )


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
