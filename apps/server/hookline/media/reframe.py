"""9:16 reframing helpers."""

import logging
import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.pipeline.layout import (
    FACE_CORE_TOP,
    STACK_CAM_H,
    STACK_GAME_H,
    ZOOM_TOP,
    SceneLayout,
)
from hookline.pipeline.profile import vod_bars
from hookline.vision.faces import sample_frames

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
SPLIT_H = OUT_H // 2  # 960
TRACK_ALPHA = 0.3  # EMA weight per TRACK_ALPHA_DT s of track (scaled to the sample spacing)
TRACK_ALPHA_DT = 0.5
TRACK_DEADBAND = 0.05
# Edge guard: when the face box comes within GUARD_MARGIN of the visible crop's width of its
# edge (visible at the scene's highest zoom), the crop catches up at once (no deadband or EMA
# lag) and the zoom holds at the base level around those samples.
GUARD_MARGIN = 0.08
GUARD_MERGE_S = 1.0  # guard spans closer than this merge (no zoom pumping)
GUARD_JITTER = 0.02  # a guarded face leaves this much (source width) each side for jitter
GUARD_PASSES = 5  # guard_zoom iterates to a fixed point in at most this many passes
BEAT_SLIDE = 1.5  # a beat's crop pulse may start up to this much later where it fits ...
SAMPLE_STEP = 0.25  # ... tried at this spacing (the face sampling interval)
MAX_CADENCE = 12  # stretches over ZOOM_MAX_GAP filled after the guard, at most this many
MERGE_S = 0.1  # a crop pulse this close to a zoom step moves onto it: one framing change
FIT_SLACK = 0.98  # plan_zoom's face-fit cap stays this far inside the guard's limit
FACE_CORE = 0.8  # share of the face box width kept inside the crop for unguarded close-ups
KEYFRAME_TOL = 0.5  # px: track keyframes this close to the line through their neighbours go
# Reframe snaps (fix wave F): the rendered crop never pans faster than MAX_PAN_PX source px/s
# at a PAN_REF_W-wide source (scaled with the width). A move that needs more becomes an
# instant reframe (a framing cut) at the moment the face would leave the margin, moved back
# to a word gap within SNAP_GAP_S when the new framing already fits there; it frames the face
# for up to SNAP_LOOKAHEAD s ahead. SNAP_EPS is the expression's one-step ramp.
MAX_PAN_PX = 150.0
PAN_REF_W = 1920
MAX_PAN = MAX_PAN_PX / PAN_REF_W  # share of the source width per second
SNAP_GAP_S = 0.3
SNAP_LOOKAHEAD = 1.0
SNAP_EPS = 0.001
SNAP_SETTLED = 0.01  # a face moving less than this over the lookahead has settled (centred)
RESYNC = 0.25  # an off-plan crop rejoins the plan within this share of a sample's reach
PAN_TOL = 2 * KEYFRAME_TOL / PAN_REF_W  # held / limited crops keep this inside the margin
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
CUT_QUIET = 1.0  # no beat (non-cut) step within this of a cut or scene change
INSTANT_LEAD = 0.005  # a cut's zoom change lands this early, so the cut frame shows it
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
MAX_BASE = 1.25
MIN_STEP_RANGE = 0.1  # the base leaves at least this much zoom above it for the steps
HEADROOM = 0.04  # the head top (face top - half a face height) stays this share below the top
# A FACE_TRACK scene whose zoom levels the headroom flattens (top <= base + FLAT_ZOOM) shifts
# its crop sideways by STEP_SHIFT of the crop width at each step instead (never over the face).
FLAT_ZOOM = 0.03
STEP_SHIFT = 0.04
_CAP_STEP = 0.005

_CROP_SAMPLES = 15
# Full-cam reaction beats (wave G-layout): the cam's chin lands at most here, so a caption fits
# below it inside Q4 (caption line 1180 - 39 px visible above it - a 40 px face gap).
BEAT_CHIN_Y = 1100
PATCH_BLUR = "gblur=sigma=16"  # an overlay inside the gameplay panel, blurred over


def detect_crop(
    video: Path, samples: int = _CROP_SAMPLES, protect: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int] | None:
    """Static black-bar crop (w, h, x, y) over `samples` evenly spaced frames: the union of
    every frame's content (`profile.vod_bars`: lines with pixels above near-pure black), grown
    to keep `protect` (x0, y0, x1, y1 px; a facecam); None when the bars are under 2% of both
    dimensions. A job computes this once over the whole VOD (the layout profile); a dark
    stretch of a game is content somewhere in the VOD, so it is never cropped."""
    width, _ = probe_size(video)
    duration = probe_duration(video)
    times = [duration * (i + 0.5) / samples for i in range(samples)]
    try:
        frames = sample_frames(video, times, width=width)
    except FFmpegError as e:
        log.warning("black-bar detection skipped for %s: %s", video.name, e)
        return None
    gray = np.stack(
        [
            (f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299).astype(np.uint8)
            for f in frames
        ]
    )
    return vod_bars(gray, protect=protect)


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


def _median3(xs: list[float]) -> list[float]:
    """Median of each value and its neighbours. The first and last use Tukey's end-point
    rule, the median of the end value, its neighbour's median and the line through the next
    two medians extrapolated to the end: a stray value at either end is ignored too, while a
    real move that ends the scene (a lean) is still followed."""
    n = len(xs)
    if n < 3:
        return list(xs)

    def mid(a: float, b: float, c: float) -> float:
        return sorted((a, b, c))[1]

    med = [xs[0]] + [mid(*xs[i - 1 : i + 2]) for i in range(1, n - 1)] + [xs[-1]]
    first = mid(xs[0], med[1], 3 * med[1] - 2 * med[2])
    med[-1] = mid(xs[-1], med[-2], 3 * med[-2] - 2 * med[-3])
    med[0] = first
    return med


def _contain(c: float, lo: float, hi: float, room: float) -> float:
    """Crop centre nearest `c` that keeps the face span [lo, hi] inside [c - room, c + room];
    the face centre when the face is wider than that."""
    if hi - lo > 2 * room:
        return (lo + hi) / 2
    return min(max(c, hi - room), lo + room)


ZoomAt = float | Callable[[float], float]
OffsetAt = Callable[[float], float] | None  # extra rendered crop offset (share of src width)
Gaps = Sequence[tuple[float, float]]


def _zoom_fn(zoom: ZoomAt) -> Callable[[float], float]:
    return zoom if callable(zoom) else (lambda _t: float(zoom))


class _Track(NamedTuple):
    """`_track_plan`'s result: per sample (t, plan centre) and guard flag, and the reframe
    snaps as (scene time, index of the first sample after it)."""

    points: list[tuple[float, float]]
    flags: list[bool]
    snaps: list[tuple[float, int]]


def _track_faces(
    keyframes: list[tuple[float, float]], face_w: float | None, widths: list[float] | None
) -> tuple[list[float], list[float], list[float]]:
    """(times, face centres, face widths) in time order: centres median-of-3 filtered
    (`_median3`), widths this sample's (else `face_w`, else 0) median-of-3 so one stray
    oversize box is ignored too."""
    order = sorted(range(len(keyframes)), key=lambda i: keyframes[i][0])
    ts = [keyframes[i][0] for i in order]
    xs = _median3([keyframes[i][1] for i in order])
    if widths is not None and len(widths) == len(keyframes):
        ws = _median3([widths[i] for i in order])
    else:
        ws = [face_w or 0.0] * len(order)
    return ts, xs, ws


def _guardable(w: float, room: float) -> bool:
    """Whether a face `w` wide fits the guard room with GUARD_JITTER to spare each side; a
    wider face (a tight close-up) keeps the deadband/EMA path with containment instead."""
    return w + 2 * GUARD_JITTER <= 2 * room


def _track_plan(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
    gaps: Gaps = (),
) -> tuple[list[tuple[float, float]], list[bool]]:
    """((t, crop centre), guard flag) per sample: `_plan_track` without the snap list."""
    plan = _plan_track(keyframes, in_w, crop_w, face_w, widths, lead, zoom_top, offset, gaps)
    return plan.points, plan.flags


def _plan_track(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
    gaps: Gaps = (),
) -> _Track:
    """Per sample (t, crop centre) and guard flag, normalized to the source width, plus the
    reframe snaps.

    The centre is an EMA (TRACK_ALPHA per TRACK_ALPHA_DT s) chasing a deadbanded target over
    the median-of-3 face position (`_track_faces`: single stray detections are ignored); the
    target jumps only when the face moves more than TRACK_DEADBAND. `zoom_top` is the rendered
    zoom at each sample (a constant, or a function of the keyframe time): the crop as seen at
    that zoom is what the face must stay inside. `offset` (a function of the keyframe time,
    share of the source width) is a crop offset the renderer adds on top (a flat scene's step
    shift): the guard and the frame bounds see the crop where it renders; the returned
    centres exclude it.

    Edge guard: when the face box comes within GUARD_MARGIN of the visible crop's edge, the
    centre moves at once to hold that margin and the EMA restarts there; the sample is
    flagged. A face too wide for that (`_guardable`: a tight close-up) is never guarded: the
    EMA keeps its deadband and the centre only moves as far as keeping the face core
    (FACE_CORE of its width) inside the visible crop. Then `lead` (signed share of the crop
    width: gaze lead room) shifts the centre, never past the guard margin, and the frame
    bounds clamp it.

    Finally the path is limited to MAX_PAN (`_limit_pans`): a faster move becomes a reframe
    snap (preferring a word gap in `gaps`, keyframe time). The hard constraint holds on the
    final path: at every sample the face (core) keeps the margin at least as well as on the
    guarded centre."""
    ts, xs, ws = _track_faces(keyframes, face_w, widths)
    zoom = _zoom_fn(zoom_top)
    cw = crop_w / in_w
    lo_c, hi_c = (cw / 2, 1 - cw / 2) if cw < 1 else (0.5, 0.5)
    out: list[float] = []
    flags: list[bool] = []
    target = smoothed = 0.0
    offs = [offset(t) if offset else 0.0 for t in ts]
    for i, (t, x, w) in enumerate(zip(ts, xs, ws, strict=True)):
        off = offs[i]
        if i == 0:
            target = smoothed = x - off
        else:
            if abs(x - off - target) > TRACK_DEADBAND:
                target = x - off
            alpha = 1 - (1 - TRACK_ALPHA) ** (max(t - ts[i - 1], 0.0) / TRACK_ALPHA_DT)
            smoothed += alpha * (target - smoothed)
        visible = cw / max(zoom(t), 1e-6)
        room = visible * (0.5 - GUARD_MARGIN)
        lo, hi = x - w / 2, x + w / 2
        c = min(max(smoothed + off, lo_c), hi_c)  # where the crop renders
        guard = _guardable(w, room)
        if guard:
            guard = lo < c - room - 1e-9 or hi > c + room + 1e-9
            if guard:
                smoothed, target = _contain(c, lo, hi, room) - off, x - off
        else:  # a tight close-up: only the face core is kept inside the visible crop
            lo, hi = x - w * FACE_CORE / 2, x + w * FACE_CORE / 2
            room = visible / 2
            smoothed = _contain(c, lo, hi, room) - off
        c = min(max(smoothed + off, lo_c), hi_c)
        if lead and hi - lo <= 2 * room:
            # toward the gaze, never backwards, leaving GUARD_JITTER before a guarded margin
            slack = GUARD_JITTER if _guardable(w, visible * (0.5 - GUARD_MARGIN)) else 0.0
            shift = lead * cw
            if shift > 0:
                shift = min(shift, max(lo + room - slack - c, 0.0))
            else:
                shift = max(shift, min(hi - room + slack - c, 0.0))
            c = min(max(c + shift, lo_c), hi_c)
        out.append(c)
        flags.append(guard)
    rooms = [_room(w, cw / max(zoom(t), 1e-6)) for t, w in zip(ts, ws, strict=True)]
    path, snaps = _limit_pans(ts, out, xs, rooms, gaps, (lo_c, hi_c))
    points = [(t, c - off) for t, c, off in zip(ts, path, offs, strict=True)]
    return _Track(points, flags, snaps)


def _room(w: float, visible: float) -> tuple[float, float, float]:
    """(face width the guard keeps in the margin, half-room, face core width) at a sample: a
    guardable face and the guard margin (its core: FACE_CORE of it), else a close-up's face
    core inside the whole visible crop (both widths the core)."""
    room = visible * (0.5 - GUARD_MARGIN)
    if _guardable(w, room):
        return w, room, w * FACE_CORE
    return w * FACE_CORE, visible / 2, w * FACE_CORE


def _over(c: float, x: float, w: float, room: float) -> float:
    """How far (share of the source width) the face [x - w/2, x + w/2] pokes out of
    [c - room, c + room]; 0 when inside."""
    return max(0.0, (c - room) - (x - w / 2), (x + w / 2) - (c + room))


class _PanFit:
    """The face (filtered centres; (width, half-room, core width) per sample) against the
    guarded plan, at fractional sample indices (linear between samples)."""

    def __init__(
        self,
        ts: list[float],
        plan: list[float],
        xs: list[float],
        rooms: list[tuple[float, float, float]],
        frame: tuple[float, float] = (-math.inf, math.inf),
    ):
        self.ts, self.plan, self.xs, self.rooms, self.frame = ts, plan, xs, rooms, frame

    def at(self, i: float, core: bool = False) -> tuple[float, float, float, float]:
        """(face x, width -- the core's with `core` --, half-room, plan centre) at
        fractional sample index `i`."""
        k = min(max(math.floor(i), 0), len(self.xs) - 1)
        u = i - k
        n = 2 if core else 0
        r0, x0, p0 = self.rooms[k], self.xs[k], self.plan[k]
        if u <= 1e-9 or k + 1 >= len(self.xs):
            return x0, r0[n], r0[1], p0
        r1, x1, p1 = self.rooms[k + 1], self.xs[k + 1], self.plan[k + 1]

        def mix(a: float, b: float) -> float:
            return a + (b - a) * u

        return mix(x0, x1), mix(r0[n], r1[n]), mix(r0[1], r1[1]), mix(p0, p1)

    def fits(self, c: float, i: float, core: bool = False) -> bool:
        """Whether centre `c` keeps the face (its core: the hard constraint) inside the
        margin at index `i`, never worse than the plan does (where the plan pokes out -- a
        face against the frame edge -- that much is allowed) and inside the frame. Off the
        plan, PAN_TOL to spare: the drawn path may round by that."""
        x, w, room, ref = self.at(i, core)
        lo, hi = self.frame
        if not lo - 1e-9 <= c <= hi + 1e-9:
            return False
        return _over(c, x, w, room - PAN_TOL) <= _over(ref, x, w, room) + 1e-9

    def index(self, t: float) -> float:
        """Fractional sample index of time `t` (clamped to the samples)."""
        ts = self.ts
        if t <= ts[0]:
            return 0.0
        for k in range(len(ts) - 1):
            if ts[k] <= t <= ts[k + 1]:
                span = ts[k + 1] - ts[k]
                return k + ((t - ts[k]) / span if span > 0 else 0.0)
        return float(len(ts) - 1)


def _limit_pans(
    ts: list[float],
    plan: list[float],
    xs: list[float],
    rooms: list[tuple[float, float, float]],
    gaps: Gaps = (),
    frame: tuple[float, float] = (-math.inf, math.inf),
) -> tuple[list[float], list[tuple[float, int]]]:
    """The rendered crop centres limited to MAX_PAN between samples, with as few reframe snaps
    as the hard constraint allows, and the snaps.

    Each sample has a band of centres keeping the face core inside the guard margin (`_band`:
    never worse than the guarded `plan`, inside the frame). A forward pass carries the centres
    reachable at MAX_PAN through the bands; where none is left, a snap starts a new stretch
    (the fewest snaps: each stretch runs as long as one speed-capped path fits). A backward
    pass then picks, per stretch, the path nearest the plan (the guarded EMA: smooth
    following, the whole face in the margin wherever the plan has it). Snap times:
    `_snap_time`."""
    if not plan:
        return [], []
    fit = _PanFit(ts, plan, xs, rooms, frame)
    bands = [_band(fit, j) for j in range(len(plan))]
    starts = [0]
    reach_sets: list[tuple[float, float]] = [bands[0]]
    for j in range(1, len(plan)):
        reach = MAX_PAN * max(ts[j] - ts[j - 1], 0.0) + 1e-9
        a, b = reach_sets[-1]
        lo, hi = max(a - reach, bands[j][0]), min(b + reach, bands[j][1])
        if lo > hi + 1e-12:
            starts.append(j)
            reach_sets.append(bands[j])
        else:
            reach_sets.append((lo, hi))
    path = [0.0] * len(plan)
    bounds = [*starts[1:], len(plan)]
    for s, e in zip(starts, bounds, strict=True):
        # backward: the centres from which the rest of the stretch stays reachable
        ok = list(reach_sets[s:e])
        for j in range(e - 2, s - 1, -1):
            reach = MAX_PAN * max(ts[j + 1] - ts[j], 0.0) + 1e-9
            a, b = ok[j + 1 - s]
            ok[j - s] = (max(ok[j - s][0], a - reach), min(ok[j - s][1], b + reach))
        # forward: a snap lands as near the old framing as it can (the smallest jump, which
        # also lets it move back to a word gap); then the path heads for the plan at MAX_PAN
        for j in range(s, e):
            lo, hi = ok[j - s]
            if j > s:
                reach = MAX_PAN * max(ts[j] - ts[j - 1], 0.0) + 1e-9
                lo, hi = max(lo, path[j - 1] - reach), min(hi, path[j - 1] + reach)
            pref = path[s - 1] if j == s and s > 0 else plan[j]
            path[j] = min(max(pref, lo), hi)
    snaps: list[tuple[float, int]] = []
    for j in starts[1:]:
        floor = snaps[-1][0] if snaps else -math.inf
        when, first = _snap_time(j, fit, path[j - 1], path[j], gaps, floor)
        if snaps and first <= snaps[-1][1]:  # never moved back over the previous snap
            when, first = _snap_time(j, fit, path[j - 1], path[j], (), floor)
        for k in range(first, j):
            path[k] = path[j]
        snaps.append((round(when, 3), first))
    return path, snaps


def _band(fit: _PanFit, j: int) -> tuple[float, float]:
    """Centres keeping the face core inside the margin at sample `j` (`fit.fits` with the
    core, PAN_TOL to spare), within the frame; just the plan when there are none."""
    x, w, room, ref = fit.at(j, core=True)
    allow = _over(ref, x, w, room)
    r = room - PAN_TOL
    lo = max(x + w / 2 - r - allow, fit.frame[0])
    hi = min(x - w / 2 + r + allow, fit.frame[1])
    return (lo, hi) if lo <= hi else (ref, ref)


def _snap_time(
    i: int, fit: _PanFit, prev: float, target: float, gaps: Gaps, floor: float = -math.inf
) -> tuple[float, int]:
    """(snap time, index of the first sample showing `target`) for a snap that must have
    happened by sample `i`: about the moment the held crop `prev` stops holding the face core
    (linear between samples, after sample i - 1; no earlier than `target` holds it), moved
    back to the latest point of a word gap within SNAP_GAP_S before it (never within MERGE_S
    of the previous snap, `floor`) when `target` holds the core from there on (samples it
    passes take `target`)."""
    ts = fit.ts
    t0, t1 = ts[i - 1], ts[i]
    grid = [t0 + (t1 - t0) * k / 20 for k in range(1, 21)]  # after the last shown sample
    when = next((t for t in grid if not fit.fits(prev, fit.index(t), core=True)), t1)
    when = next((t for t in grid if t >= when and fit.fits(target, fit.index(t), True)), t1)
    best = None
    for a, b in gaps:
        g = min(b, when)
        if g < a - 1e-9 or g < max(when - SNAP_GAP_S, floor + MERGE_S) - 1e-9 or g >= when - 1e-9:
            continue
        span = [g, *(t for t in ts if g < t <= t1)]
        if all(fit.fits(target, fit.index(u), core=True) for u in span):
            best = g if best is None else max(best, g)
    if best is not None:
        when = best
    first = next(k for k in range(len(ts)) if ts[k] >= when - 1e-9)
    return when, min(first, i)


def _drop_collinear(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """`pts` without the points lying within `tol` of the straight line between the kept
    points around them (every dropped point is checked, so slow curves don't drift)."""
    if len(pts) <= 2:
        return list(pts)
    kept = [pts[0]]
    run: list[tuple[float, float]] = []  # dropped since the last kept point
    for i in range(1, len(pts) - 1):
        (t0, v0), (t1, v1) = kept[-1], pts[i + 1]
        cand = [*run, pts[i]]
        if t1 > t0 and all(
            abs(v - (v0 + (v1 - v0) * (t - t0) / (t1 - t0))) <= tol for t, v in cand
        ):
            run = cand
        else:
            kept.append(pts[i])
            run = []
    kept.append(pts[-1])
    return kept


def _pwl_expr(pts: list[tuple[float, float]], smooth: bool = False) -> str:
    """Piecewise ffmpeg expression in `t` through (t, value) points, held constant before the
    first and after the last. Segments are linear, or smoothstep (u^2 (3 - 2u)) when `smooth`.
    A balanced tree of if(lt(t, ...)) picks the segment, so the nesting depth grows with
    log2 of the point count (ffmpeg's expression parser gives up at ~100 levels). One value
    throughout is just that value. Commas are NOT escaped."""
    if len({_fmt(v) for _, v in pts}) == 1:
        return _fmt(pts[0][1])
    segs: list[tuple[float, str]] = [(-math.inf, _fmt(pts[0][1]))]
    for (t0, x0), (t1, x1) in pairwise(pts):
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
        segs.append((t0, seg))
    segs.append((pts[-1][0], _fmt(pts[-1][1])))

    def build(lo: int, hi: int) -> str:
        if hi - lo == 1:
            return segs[lo][1]
        mid = (lo + hi) // 2
        return f"if(lt(t,{_fmt(segs[mid][0])}),{build(lo, mid)},{build(mid, hi)})"

    return build(0, len(segs))


def track_x_expr(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None = None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
    gaps: Gaps = (),
) -> str:
    """ffmpeg expression (in `t`) for the crop's left edge: piecewise-linear over the smoothed,
    edge-guarded, lead-shifted, pan-limited and clamped keyframes (normalized centre x;
    `_plan_track`), each reframe snap a one-step jump, keyframes on a straight line dropped.
    `widths` are the faces' per-keyframe widths (normalized; else `face_w`, else 0);
    `zoom_top` is the rendered zoom (a constant, or a function of the keyframe time:
    `zoom_at_fn`); the guard keeps the face inside the crop as seen there. With `offset` (the
    renderer adds it, then clamps) the edge is left unclamped. `gaps`: word gaps (keyframe
    time) a snap prefers. Commas are NOT escaped."""
    pts = _track_lefts(keyframes, in_w, crop_w, face_w, widths, lead, zoom_top, offset, gaps)
    if not pts:
        return _fmt(max(in_w - crop_w, 0) / 2)
    return _pwl_expr(_drop_collinear(pts, KEYFRAME_TOL))


def track_guard_spans(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None = None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
) -> list[tuple[float, float]]:
    """When the edge guard acts (keyframe time base), to hold the zoom at its base: each run
    of guarded samples, from ZOOM_EASE s before its first (the zoom is at base by then) to one
    sample spacing after its last; spans closer than GUARD_MERGE_S merge."""
    plan, flags = _track_plan(keyframes, in_w, crop_w, face_w, widths, lead, zoom_top, offset)
    ts = [t for t, _ in plan]
    spans: list[tuple[float, float]] = []
    for i, flag in enumerate(flags):
        if not flag:
            continue
        step = ts[i + 1] - ts[i] if i + 1 < len(ts) else (ts[i] - ts[i - 1] if i else 0.0)
        a, b = max(ts[i] - ZOOM_EASE, 0.0), ts[i] + step
        if spans and a - spans[-1][1] < GUARD_MERGE_S:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def track_snaps(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None = None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
    gaps: Gaps = (),
) -> list[float]:
    """Times (keyframe time base) of the track's reframe snaps (`_limit_pans`)."""
    plan = _plan_track(keyframes, in_w, crop_w, face_w, widths, lead, zoom_top, offset, gaps)
    return [t for t, _ in plan.snaps]


def _step_offset(
    layout: SceneLayout, steps: Sequence[Sequence[float]] | None, in_w: int, in_h: int
) -> OffsetAt:
    """A flat FACE_TRACK scene's step shift as rendered (`_shifted`): `step_shift` of the crop
    width (as a share of the source width) after every other zoom step, eased (smoothstep over
    ZOOM_EASE) except at the scene's cuts (`layout.cuts`), as a function of scene time; None
    when the scene has none."""
    if not layout.step_shift or not steps:
        return None
    shift = layout.step_shift * _track_w(in_w, in_h) / in_w
    times = sorted(float(st[0]) for st in steps)

    def at(t: float) -> float:
        u = t + layout.start
        n = sum(1 for a in times if a <= u + 1e-9)
        now = shift if n and (n - 1) % 2 else 0.0
        if n < 2:
            return now
        since = u - times[n - 1]
        if since >= ZOOM_EASE or any(abs(times[n - 1] - c) < 1e-3 for c in layout.cuts):
            return now
        before = shift if (n - 2) % 2 else 0.0
        k = since / ZOOM_EASE
        return before + (now - before) * k * k * (3 - 2 * k)

    return at


def _track_kw(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]] | None
) -> dict:
    """`_plan_track` arguments for a FACE_TRACK scene rendered under zoom `steps` (scene time
    base): its keyframes, face widths, lead, the zoom it renders at, its step shift and the
    word gaps inside it."""
    return {
        "keyframes": [(t - layout.start, x) for t, x in layout.track_x],
        "in_w": in_w,
        "crop_w": _track_w(in_w, in_h),
        "face_w": layout.face_w,
        "widths": layout.track_w or None,
        "lead": layout.lead,
        "zoom_top": zoom_at_fn(steps or [(0.0, 1.0)], layout),
        "offset": _step_offset(layout, steps, in_w, in_h),
        "gaps": [(a - layout.start, b - layout.start) for a, b in layout.gaps],
    }


def scene_snaps(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]]
) -> list[float]:
    """Output times of a FACE_TRACK scene's reframe snaps as rendered under `steps` (framing
    changes: the sidecar records them for Q7); [] for other scenes."""
    if _kind(layout) != "FACE_TRACK" or not layout.track_x:
        return []
    return [round(t + layout.start, 3) for t in track_snaps(**_track_kw(layout, in_w, in_h, steps))]


def guarded_steps(
    steps: Sequence[Sequence[float]], spans: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Zoom `steps` held at level 1.0 (the base) inside each span: a step to 1.0 at its start
    and back to the schedule's level at its end; schedule steps inside it, and repeats of the
    level just before, dropped."""
    seq = [(float(st[0]), float(st[1])) for st in steps]
    if not spans:
        return seq

    def level_at(t: float) -> float:
        return max((st for st in seq if st[0] <= t + 1e-9), default=(0.0, 1.0))[1]

    out = [st for st in seq if not any(a <= st[0] < b for a, b in spans)]
    for a, b in spans:
        out += [(a, 1.0), (b, level_at(b))]
    merged: list[tuple[float, float]] = []
    for t, v in sorted(out):
        if not merged or v != merged[-1][1]:
            merged.append((t, v))
    return merged


def zoom_at_fn(steps: Sequence[Sequence[float]], layout: SceneLayout) -> Callable[[float], float]:
    """Rendered zoom of `layout` at scene time t under `steps` (output time): the highest of
    the level in effect, any step easing in within ZOOM_EASE s (an ease toward a zoom-in
    already narrows the crop) and, within PUNCH_RELEASE s after a step, the level before it
    (a zoom-out is still easing)."""
    seq = sorted((float(t), float(s)) for t, s in steps)

    def at(t: float) -> float:
        u = t + layout.start
        past = [st for st in seq if st[0] <= u + 1e-9]
        levels = [past[-1][1] if past else 1.0, *(s for v, s in seq if u < v <= u + ZOOM_EASE)]
        if len(past) > 1 and u - past[-1][0] < PUNCH_RELEASE:
            levels.append(past[-2][1])
        return max(layout.zoom_level(s) for s in levels)

    return at


def _dedupe(steps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for t, v in sorted(steps):
        if out and abs(t - out[-1][0]) < 1e-9:
            out[-1] = (t, v)  # the later entry for one time wins
        elif not out or v != out[-1][1]:
            out.append((t, v))
    return out


def _level_at(steps: Sequence[tuple[float, float]], t: float) -> float:
    return max((st for st in steps if st[0] <= t + 1e-9), default=(0.0, 1.0))[1]


def guard_zoom(
    layouts: Sequence[SceneLayout],
    in_w: int,
    in_h: int,
    steps: Sequence[Sequence[float]],
    cut_times: Sequence[float] = (),
) -> tuple[list[tuple[float, float]], list[SceneLayout]]:
    """(zoom steps, layouts with `shift_steps`) with the edge guard applied to the pre-guard
    schedule `steps`, computed once for the render AND the sidecar (any `shift_steps` already
    on the layouts are ignored: the result depends on the schedule alone).

    Per FACE_TRACK scene: hold the base level inside the guard spans, recompute the spans at
    the zoom that renders, until they settle (GUARD_PASSES at most; unsettled, the scene
    holds base from its first span to its end and a warning is logged). Then one pass
    (`_hold_spans`, checked against that final schedule) places the returns to the schedule
    and the STEP_SHIFT crop pulses that keep every cut and step a framing change."""
    held = _dedupe([(float(t), float(s)) for t, s in steps])
    out: list[SceneLayout] = []
    for lay in layouts:
        lay = lay.model_copy(  # the scene's cuts: crop changes there render instantly
            update={"cuts": [float(c) for c in cut_times if lay.start <= c <= lay.end]}
        )
        if _kind(lay) != "FACE_TRACK" or not lay.track_x or not held:
            out.append(lay)
            continue
        held, shifts = _guard_scene(lay, in_w, in_h, held, cut_times)
        out.append(lay.model_copy(update={"shift_steps": shifts}))
    return held, out


class _SceneFit:
    """Face geometry of one FACE_TRACK scene and the crop path it renders under `steps`, for
    `guard_zoom`'s safety checks."""

    def __init__(self, lay: SceneLayout, in_w: int, in_h: int, steps: list[tuple[float, float]]):
        self.lay, self.in_w = lay, in_w
        self.zoom = zoom_at_fn(steps, lay)
        self.cw = _track_w(in_w, in_h)
        kw = _track_kw(lay, in_w, in_h, steps)
        off = kw["offset"] or (lambda _t: 0.0)
        # the rendered centres: the plan plus a flat scene's step shift
        self.plan = [(t, c + off(t)) for t, c in _plan_track(**kw).points]
        self.ts, self.xs, self.ws = _track_faces(kw["keyframes"], lay.face_w, kw["widths"])
        half = self.cw / in_w / 2
        self.frame = (half, 1 - half) if half < 0.5 else (0.5, 0.5)

    def _samples(self, t0: float, t1: float) -> list[int]:
        a, b = t0 - self.lay.start, t1 - self.lay.start
        idx = [i for i, t in enumerate(self.ts) if a <= t <= b]
        before = [i for i, t in enumerate(self.ts) if t < a]
        return idx + before[-1:]

    def fits(
        self, level: float | None, t0: float, t1: float, shift: float = 0.0, path: bool = False
    ) -> bool:
        """Whether the face stays inside the guard margin from output time t0 to t1 at
        schedule `level` (None: the zoom that renders then, eases included): with the crop
        free to follow it, or, with `shift` (share of the crop width) or `path`, on the
        rendered crop path (shifted by that much) without the guard having to catch up."""
        lo_c, hi_c = self.frame
        for i in self._samples(t0, t1):
            z = self.zoom(self.ts[i]) if level is None else self.lay.zoom_level(level)
            room = self.cw / self.in_w / z * (0.5 - GUARD_MARGIN)
            x, w = self.xs[i], self.ws[i]
            lo, hi = x - w / 2, x + w / 2
            if shift or path:
                c = min(max(self.plan[i][1] + shift * self.cw / self.in_w, lo_c), hi_c)
                if lo < c - room - 1e-9 or hi > c + room + 1e-9:
                    return False
            elif not _guardable(w, room) or max(hi - room, lo_c) > min(lo + room, hi_c):
                return False
        return True

    def direction(self, t0: float, t1: float) -> float:
        """+1 / -1: the crop shift that moves the face toward the crop centre, from the mean
        face offset over output times t0..t1 (toward the frame centre when it is zero)."""
        idx = self._samples(t0, t1)
        off = sum(self.xs[i] - self.plan[i][1] for i in idx) / max(len(idx), 1)
        if abs(off) < 1e-6:
            off = 0.5 - (self.plan[idx[0]][1] if idx else 0.5)
        return 1.0 if off >= 0 else -1.0


def _join_spans(
    spans: list[tuple[float, float]],
    times: Sequence[float],
    rises: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    """Spans sorted and merged where they overlap, each start pulled back to a step or cut
    within CUT_QUIET before it (no zoom-in just before a hold). `rises` are (beat punch time,
    time the level before it rose above base): a span starting at most CUT_QUIET after such a
    beat starts at that rise instead, so the beat never becomes a zoom-out."""
    out: list[tuple[float, float]] = []
    for a0, b in sorted(spans):
        a = min([a0, *(t for t in times if a0 - CUT_QUIET <= t < a0)])
        a = min([a, *(r for p, r in rises if a0 - CUT_QUIET - 1e-9 <= p <= a0 + 1e-9)])
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in out]


def _beat_rises(
    sched: list[tuple[float, float]], scene: tuple[float, float]
) -> list[tuple[float, float]]:
    """(punch time, time the level before it rose above base) for each beat punch in the
    scene that a base hold starting at it would turn into a zoom-out (the scene start when the
    level was already up there)."""
    out = []
    for i, (t, v) in enumerate(sched):
        if v != ZOOM_PUNCH or not scene[0] < t < scene[1]:
            continue
        if not i or sched[i - 1][1] <= 1.0:
            continue  # a punch from base: a hold there zooms nothing out
        base = [j for j in range(i) if sched[j][1] <= 1.0]
        rise = sched[base[-1] + 1][0] if base else sched[0][0]
        out.append((t, max(rise, scene[0])))
    return out


def _inside(t: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in spans)


def _hold_base(
    sched: list[tuple[float, float]], spans: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """`sched` with an explicit base (1.0) step at each span's start and its steps inside
    spans dropped (not deduplicated: a span start stays a reset even where the level is
    already base, so nothing placed before it can run through the hold)."""
    return sorted(
        [*(st for st in sched if not _inside(st[0], spans)), *((a, 1.0) for a, _ in spans)]
    )


def _guard_scene(
    lay: SceneLayout,
    in_w: int,
    in_h: int,
    sched: list[tuple[float, float]],
    cut_times: Sequence[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """`guard_zoom` for one FACE_TRACK scene: (steps, its shift_steps). The guard sees the
    crop where it renders: a flat scene's step shift included (`_step_offset`)."""
    cuts = sorted(c for c in cut_times if lay.start < c < lay.end)
    times = sorted({*(t for t, _ in sched if lay.start < t < lay.end), *cuts})
    rises = _beat_rises(sched, (lay.start, lay.end))

    def spans_at(steps: list[tuple[float, float]]) -> list[tuple[float, float]]:
        kw = _track_kw(lay, in_w, in_h, steps)
        del kw["gaps"]
        return [(a + lay.start, b + lay.start) for a, b in track_guard_spans(**kw)]

    spans: list[tuple[float, float]] = []
    held = sched
    for _ in range(GUARD_PASSES):
        found = [(a, min(b, lay.end)) for a, b in spans_at(held) if a < lay.end]
        grown = _join_spans(spans + found, times, rises)
        if grown == spans:
            break
        spans = grown
        held = _dedupe(_hold_base(sched, spans))
    else:
        log.warning(
            "edge guard did not settle in the scene at %.2f-%.2f s; holding base from %.2f s",
            lay.start, lay.end, spans[0][0],
        )  # fmt: skip
        spans = [(spans[0][0], lay.end)]

    return _hold_spans(
        sched, spans, cuts, lambda steps: _SceneFit(lay, in_w, in_h, steps),
        (lay.start, lay.end), spans_at, lay.gaps,
    )  # fmt: skip


def _hold_spans(
    sched: list[tuple[float, float]],
    spans: list[tuple[float, float]],
    cuts: Sequence[float],
    make_fit: Callable[[list[tuple[float, float]]], "_SceneFit"],
    scene: tuple[float, float],
    spans_at: Callable[[list[tuple[float, float]]], list[tuple[float, float]]] | None = None,
    word_gaps: Sequence[tuple[float, float]] = (),
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The placement pass of `guard_zoom` for a scene (output times in `scene`):
    (steps, shift_steps). `make_fit` builds a `_SceneFit` for a schedule.

    1. Levels: base from every span's start (an explicit reset) to its end; after a span the
    schedule's level returns once CUT_QUIET has passed since the span's start and its last
    cut or step, unless the next span, a step, a cut or the scene end comes within CUT_QUIET;
    a cut outside the spans that lost its change takes the other level. Each zoom-in placed
    here holds only until the next level step or span start, and stays only if the guard
    (`spans_at`, at the zoom the final schedule renders) never fires while it is in effect.
    2. Pulses, checked against that final schedule: every cut or step inside a span (its
    start too, when the hold changes nothing) and every cut or step outside the spans still
    without a change gets a STEP_SHIFT crop pulse (`pulse`)."""
    start, end = scene
    times = sorted({*(t for t, _ in sched if start < t < end), *cuts})
    starts = [a for a, _ in spans]
    levels = _hold_base(sched, spans)

    def limit(t: float) -> float:
        """Where a level placed at `t` stops: the next level step or span start."""
        nxt = [u for u, _ in levels if u > t + 1e-9] + [a for a in starts if a > t + 1e-9]
        return min([*nxt, end])

    placed: list[tuple[float, float]] = []  # zoom-ins placed here, to verify on the final path
    for k, (a, b) in enumerate(spans):
        inner = [t for t in times if a <= t < b]
        r = max(b, a + CUT_QUIET, inner[-1] + CUT_QUIET if inner else b)
        nxt = min(
            starts[k + 1] if k + 1 < len(starts) else math.inf,
            min((t for t in times if t >= b - 1e-9), default=math.inf),
            end,
        )
        level = _level_at(sched, r)
        if nxt >= r + CUT_QUIET and level != 1.0:
            placed.append((r, level))
    levels = sorted([*levels, *placed])
    for c in cuts:  # a cut outside the holds keeps a level change
        if _inside(c, spans):
            continue
        prev = _level_at([st for st in levels if st[0] < c - 1e-9], c)
        if _level_at(levels, c) == prev:
            levels = sorted(
                [
                    *(st for st in levels if abs(st[0] - c) >= 1e-9),
                    (c, ZOOM_IN if prev == 1.0 else 1.0),
                ]
            )
            if prev == 1.0:
                placed.append((c, ZOOM_IN))
    for _ in range(GUARD_PASSES):  # keep only zoom-ins the guard never fires under
        fresh = spans_at(_dedupe(levels)) if spans_at else []
        bad = [(t, v) for t, v in placed if any(a < limit(t) and t < b for a, b in fresh)]
        if not bad:
            break
        placed = [st for st in placed if st not in bad]
        levels = [st for st in levels if st not in bad]
    if spans_at is not None:
        levels = _punch_in(levels, sched, cuts, scene, spans_at)
    beats = [t for t, v in sched if v == ZOOM_PUNCH and start < t < end]
    held = _no_beat_zoom_out(_dedupe(levels), beats)
    if spans_at is not None:
        held = _fill_cadence(held, cuts, scene, spans_at, word_gaps, spans)
    fit = make_fit(held)
    pulses: list[tuple[float, float, float]] = []  # (start, end, sign)
    shifts: list[tuple[float, float]] = []

    def pulse(t: float, sign: float, free: bool) -> bool:
        """A crop pulse at `t` (moved onto a level step within MERGE_S, one framing change):
        STEP_SHIFT for CUT_QUIET, else ZOOM_MIN_STEP, ended early by a cut, never shorter than
        ZOOM_MIN_STEP and never switching off within ZOOM_EASE of the scene end; starts
        CUT_QUIET apart, ZOOM_MIN_STEP between one's end and the next, and pulses within
        CUT_QUIET of each other share a direction. `free`: the other direction may be
        tried when `sign` is not safe. Checked through its ease-out on the final schedule."""
        t = min((u for u, _ in held if abs(u - t) < MERGE_S), key=lambda u: abs(u - t), default=t)
        if any(abs(t - a) < CUT_QUIET or a <= t < b + ZOOM_MIN_STEP for a, b, _ in pulses):
            return False
        # a pulse ending within CUT_QUIET before, or starting soon after, sets the direction
        near = [s for a, b, s in pulses if b <= t < b + CUT_QUIET or t < a < t + 2 * CUT_QUIET]
        if len(set(near)) > 1:
            return False  # between two opposite pulses: any direction would flip-flop
        signs = near[:1] if near else ([sign, -sign] if free else [sign])
        for s in signs:
            for dur in (CUT_QUIET, ZOOM_MIN_STEP):
                off = min([t + dur, *(c for c in cuts if t < c < t + dur)])
                if off > end - ZOOM_EASE:
                    off = end
                if off - t < ZOOM_MIN_STEP - 1e-9 or any(
                    t < a < off + ZOOM_MIN_STEP for a, _, _ in pulses
                ):
                    continue
                if fit.fits(None, t, min(off + ZOOM_EASE, end), shift=s * STEP_SHIFT):
                    pulses.append((t, off, s))
                    shifts.append((t, s * STEP_SHIFT))
                    if off < end:
                        shifts.append((round(off, 3), 0.0))
                    return True
        return False

    def changes(t: float) -> bool:
        return _level_at(held, t) != _level_at([st for st in held if st[0] < t - 1e-9], t)

    for a, b in spans:
        before = _level_at([st for st in held if st[0] < a - 1e-9], a)
        sign = fit.direction(a, b)
        for t in (t for t in times if a <= t < b and (t > a + 1e-9 or before == 1.0)):
            if t > a + 1e-9 and changes(t):
                continue  # a punch the guard allowed (`_punch_in`) is its framing change
            pulse(t, sign, free=False)
    for t in times:  # outside the holds, every cut and step keeps a framing change
        if _inside(t, spans) or any(a - 1e-9 <= t < b for a, b, _ in pulses):
            continue
        if _level_at(held, t) != _level_at([st for st in held if st[0] < t - 1e-9], t):
            continue
        nxt = min((u for u in times if u > t + 1e-9), default=end)
        slide = 0.0 if t in cuts else min(BEAT_SLIDE, max(nxt - CUT_QUIET - t, 0.0))
        sign = fit.direction(t, t + CUT_QUIET)
        tries = [t + k * SAMPLE_STEP for k in range(int(slide / SAMPLE_STEP + 1e-9) + 1)]
        any(pulse(u, sign, free=True) for u in tries)  # a beat may slide later to fit
    # the cadence: a stretch still over ZOOM_MAX_GAP without a framing change gets a pulse at
    # a word gap in it (the level changes `_fill_cadence` could place are already in)
    for _ in range(MAX_CADENCE):
        marks = _marks(held, cuts, scene, [t for t, _ in shifts])
        long = [(a, b) for a, b in pairwise(marks) if b - a > ZOOM_MAX_GAP + 1e-9]
        if not any(
            any(pulse(u, fit.direction(a, b), free=True) for u in _cadence_times(a, b, word_gaps))
            for a, b in long
        ):
            break
    return held, sorted(shifts)


def _marks(
    steps: Sequence[tuple[float, float]],
    cuts: Sequence[float],
    scene: tuple[float, float],
    extra: Sequence[float] = (),
) -> list[float]:
    """The scene's framing changes (as Q7 counts them): its bounds, cuts, zoom steps and
    `extra` (crop shifts) inside it, sorted."""
    a, b = scene
    inside = [*cuts, *(t for t, _ in steps), *extra]
    return sorted({a, b, *(t for t in inside if a < t < b)})


def _cadence_times(a: float, b: float, word_gaps: Sequence[tuple[float, float]]) -> list[float]:
    """Where to try a framing change splitting a stretch (a, b) that runs over ZOOM_MAX_GAP:
    mid-points of the word gaps in it, at least CUT_QUIET from either end -- those leaving
    both pieces within ZOOM_MAX_GAP first, longer pauses first, then nearest the middle --
    then even times around the middle."""
    lo, hi, mid = a + CUT_QUIET, b - CUT_QUIET, (a + b) / 2

    def splits(t: float) -> bool:
        return t - a <= ZOOM_MAX_GAP + 1e-9 and b - t <= ZOOM_MAX_GAP + 1e-9

    pauses = [(gb - ga, round((ga + gb) / 2, 3)) for ga, gb in word_gaps]
    pauses = [(n, t) for n, t in pauses if lo <= t <= hi]
    pauses.sort(key=lambda p: (not splits(p[1]), -round(p[0], 2), abs(p[1] - mid)))
    even = [round(mid + k * SAMPLE_STEP * s, 3) for k in range(7) for s in (1, -1)]
    seen: list[float] = []
    for t in [t for _, t in pauses] + [t for t in even if lo <= t <= hi]:
        if t not in seen:
            seen.append(t)
    return seen


def _violations(
    steps: list[tuple[float, float]],
    spans_at: Callable[[list[tuple[float, float]]], list[tuple[float, float]]],
) -> set[tuple[float, float]]:
    """Guard spans (`spans_at` under `steps`) inside which a level above base is in effect."""
    bad = set()
    for a, b in spans_at(steps):
        if _level_at(steps, a) != 1.0 or any(v != 1.0 for u, v in steps if a < u < b):
            bad.add((a, b))
    return bad


def _fill_cadence(
    steps: list[tuple[float, float]],
    cuts: Sequence[float],
    scene: tuple[float, float],
    spans_at: Callable[[list[tuple[float, float]]], list[tuple[float, float]]],
    word_gaps: Sequence[tuple[float, float]] = (),
    holds: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    """`steps` (deduplicated) with a level change in every stretch of the scene longer than
    ZOOM_MAX_GAP without a framing change, outside the guard's `holds` (there a pulse does
    it), at a word gap (`_cadence_times`): the other level
    from there on, else a ZOOM_IN for CUT_QUIET * 2 s back to it, kept only where the guard
    fires nowhere a level above base is in effect that it did not before."""
    out = list(steps)
    end = scene[1]
    for _ in range(MAX_CADENCE):
        long = [(a, b) for a, b in pairwise(_marks(out, cuts, scene)) if b - a > ZOOM_MAX_GAP]
        current = _violations(out, spans_at) if long else set()
        placed = False
        for a, b in long:
            for t in _cadence_times(a, b, word_gaps):
                if _inside(t, holds):
                    continue
                before = _level_at([st for st in out if st[0] < t - 1e-9], t)
                tries = [[(t, ZOOM_IN if before == 1.0 else 1.0)]]
                back = round(t + 2 * CUT_QUIET, 3)
                if before == 1.0 and back < min(b, end) - CUT_QUIET:
                    tries.append([(t, ZOOM_IN), (back, 1.0)])
                for add in tries:
                    trial = _dedupe([*out, *add])
                    if _violations(trial, spans_at) <= current:
                        out, placed = trial, True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            break
    return out


def _no_beat_zoom_out(
    steps: list[tuple[float, float]], beats: Sequence[float]
) -> list[tuple[float, float]]:
    """`steps` (deduplicated) where no beat is a zoom-out: while a beat shows a lower level
    than just before it, the latest zoom-in before it is dropped (that earlier change gives
    way to the beat; levels only ever come down, so no hold is broken)."""
    out = list(steps)
    for t in sorted(beats):
        for _ in range(len(out)):
            before = [st for st in out if st[0] < t - 1e-9]
            if _level_at(out, t) >= _level_at(before, t) or len(before) < 2:
                break
            rise = next(
                (k for k in range(len(before) - 1, 0, -1) if before[k][1] > before[k - 1][1]),
                None,
            )
            if rise is None:
                break
            del out[rise]
            out = _dedupe(out)
    return out


def _punch_in(
    levels: list[tuple[float, float]],
    sched: list[tuple[float, float]],
    cuts: Sequence[float],
    scene: tuple[float, float],
    spans_at: Callable[[list[tuple[float, float]]], list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """`levels` with a punch-in at every beat (a ZOOM_PUNCH step of `sched`) that is not one
    yet and every cut without a level change, where the guard allows it: ZOOM_PUNCH, else
    ZOOM_IN, above the level before it, for PUNCH_HOLD + ZOOM_EASE s (ended early by the next
    level step, cut or the scene end, where the level returns), kept only if the guard
    (`spans_at`, on the whole new schedule) fires nowhere a level above base is in effect
    that it did not before. Never a zoom-out."""
    start, end = scene
    beats = {t for t, v in sched if v == ZOOM_PUNCH and start <= t < end}
    out = sorted(levels)

    def violations(steps: list[tuple[float, float]]) -> set[tuple[float, float]]:
        return _violations(steps, spans_at)

    current: set[tuple[float, float]] | None = None
    for t in [*sorted(beats), *sorted(c for c in cuts if c not in beats)]:  # beats first
        lv = _dedupe(out)
        before = _level_at([st for st in lv if st[0] < t - 1e-9], t)
        now = _level_at(lv, t)
        if (now > before) if t in beats else (now != before):
            continue  # already a punch-in (a beat) or a framing change (a cut)
        later = [u for u, _ in lv if u > t + 1e-9] + [c for c in cuts if c > t + 1e-9]
        stop = min([*later, t + ZOOM_EASE + PUNCH_HOLD, end])
        if stop - t < ZOOM_MIN_STEP - 1e-9:
            continue
        back = _level_at(lv, stop - 1e-6)
        for lvl in (ZOOM_PUNCH, ZOOM_IN):
            if lvl <= before + 1e-9:
                continue
            if any(abs(stop - b) < 1e-9 for b in beats) and lvl >= _level_at(lv, stop) - 1e-9:
                continue  # its return would land on a beat: the beat must still rise
            trial = [st for st in out if abs(st[0] - t) >= 1e-9] + [(t, lvl)]
            if not any(abs(u - stop) < 1e-9 for u, _ in out):
                # the return; at the scene end it lands on the camera change, in one frame
                # (`scene_filter` makes steps at scene bounds instant): never carried over
                trial.append((round(stop, 3), back))
            sd = _dedupe(trial)
            if current is None:
                current = violations(lv)
            if violations(sd) <= current:
                out, current = sorted(trial), None
                break
    return out


def _track_lefts(
    keyframes: list[tuple[float, float]],
    in_w: int,
    crop_w: int,
    face_w: float | None,
    widths: list[float] | None = None,
    lead: float = 0.0,
    zoom_top: ZoomAt = 1.0,
    offset: OffsetAt = None,
    gaps: Gaps = (),
) -> list[tuple[float, float]]:
    """(t, crop left px) of `track_x_expr`, before collinear keyframes are dropped: each snap
    holds the previous left until SNAP_EPS before it, then jumps. Clamped to the frame unless
    the renderer adds an `offset` (it clamps after)."""
    hi = max(in_w - crop_w, 0)
    plan = _plan_track(keyframes, in_w, crop_w, face_w, widths, lead, zoom_top, offset, gaps)

    def left(c: float) -> float:
        px = c * in_w - crop_w / 2
        return px if offset else min(max(px, 0.0), hi)

    pts = [(t, left(c)) for t, c in plan.points]
    out: list[tuple[float, float]] = []
    snaps = {first: when for when, first in plan.snaps}
    for k, pt in enumerate(pts):
        when = snaps.get(k)
        if when is not None and k > 0 and when - SNAP_EPS > pts[k - 1][0] + 1e-9:
            out += [(when - SNAP_EPS, pts[k - 1][1]), (when, pt[1])]
        out.append(pt)
    return out


def _fill_gap(
    a: float,
    b: float,
    max_gap: float,
    boundaries: Sequence[float],
    word_gaps: Sequence[tuple[float, float]] = (),
) -> list[float]:
    """Filler step times splitting (a, b) into pieces no longer than `max_gap`; each sits at
    the nearest sentence end within BOUNDARY_SNAP of its even-split nominal time, else in the
    middle of the longest word gap there (a pause in an unpunctuated run), else at the nominal
    time (never within CUT_QUIET of either end: a beat step next to a cut reads as fidgety)."""
    out: list[float] = []
    cur = a
    while b - cur > max_gap + 1e-9:
        rest = b - cur
        nominal = cur + rest / math.ceil(rest / max_gap - 1e-9)
        lo = max(nominal - BOUNDARY_SNAP, cur + CUT_QUIET)
        hi = min(nominal + BOUNDARY_SNAP, cur + max_gap)
        near = [x for x in boundaries if lo <= x <= hi and b - x >= CUT_QUIET]
        pauses = [
            (gb - ga, round((ga + gb) / 2, 3))
            for ga, gb in word_gaps
            if lo <= (ga + gb) / 2 <= hi and b - (ga + gb) / 2 >= CUT_QUIET
        ]
        if near:
            cur = min(near, key=lambda x: abs(x - nominal))
        elif pauses:
            cur = max(pauses, key=lambda p: (round(p[0], 2), -abs(p[1] - nominal)))[1]
        else:
            cur = nominal
        out.append(cur)
    return out


def zoom_schedule(
    duration: float,
    cut_times: Sequence[float],
    emphasis_times: Sequence[float],
    max_gap: float = ZOOM_MAX_GAP,
    boundaries: Sequence[float] = (),
    changes: Sequence[float] = (),
    word_gaps: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    """Zoom steps `(t, scale)` over an output timeline of `duration` s, starting `(0, 1.0)`.

    Every cut in (0, duration) gets a step at its exact time, so no cut is a bare jump cut.
    The base level alternates 1.0 / ZOOM_IN at each cut and at filler times keeping every gap
    (cuts and scene `changes` included) <= `max_gap` (fillers prefer `boundaries`, i.e.
    sentence ends, within BOUNDARY_SNAP s of their nominal time, else the longest `word_gaps`
    pause there). An emphasis time punches to
    ZOOM_PUNCH and returns to the base level PUNCH_HOLD s after its ease completes.

    No beat step (punch, release, filler) lands within CUT_QUIET s of a cut or scene change:
    a punch that close moves onto it (the punch is that cut's framing change; dropped when that
    is before ZOOM_MIN_STEP s), and a cut or change during a punch, or within CUT_QUIET s after
    its release, ends the punch there, at 1.0, the level furthest from the punch. Punches
    overlapping an earlier punch, or before ZOOM_MIN_STEP s, are dropped. Consecutive steps
    always differ."""
    punch_len = ZOOM_EASE + PUNCH_HOLD
    cuts = sorted({round(c, 3) for c in cut_times if 0 < c < duration})
    marks = sorted({*cuts, *(round(c, 3) for c in changes if 0 < c < duration)})
    punches: list[float] = []
    for e in sorted(emphasis_times):
        if not 0.0 <= e < duration:
            continue
        near = [c for c in marks if abs(c - e) < CUT_QUIET]
        if near:
            e = min(near, key=lambda c: abs(c - e))
            if e < ZOOM_MIN_STEP:
                continue  # its cut is too close to the start: that cut's own step stands
        elif e < ZOOM_MIN_STEP:
            e = 0.0  # a beat on the opening words punches from the first frame
        if not punches or e >= punches[-1] + punch_len + ZOOM_MIN_STEP:
            punches.append(e)
    events: list[tuple[float, str]] = []
    taken: set[float] = set()
    for p in punches:
        events.append((p, "punch"))
        taken.add(p)
        ender = [c for c in marks if p < c < p + punch_len + CUT_QUIET]
        if ender and ender[0] in punches and ender[0] - CUT_QUIET >= p + ZOOM_MIN_STEP:
            # the next beat punches at that cut: release a quiet second before it, so the
            # beat is a punch-in again (never punch-on-punch, which reads as nothing)
            events.append((round(ender[0] - CUT_QUIET, 3), "release"))
        elif ender:
            events.append((ender[0], "cut_release"))
            taken.add(ender[0])
        else:
            events.append((p + punch_len, "release"))
    events += [(c, "toggle") for c in cuts if c not in taken]
    anchors = sorted({0.0, *(t for t, _ in events), *marks, duration})
    for a, b in pairwise(anchors):
        events += [(f, "toggle") for f in _fill_gap(a, b, max_gap, boundaries, word_gaps)]

    base = 1.0
    steps = [(0.0, 1.0)]
    for t, kind in sorted(events):
        if kind == "toggle":
            base = ZOOM_IN if base == 1.0 else 1.0
        elif kind == "cut_release":
            base = 1.0
        steps.append((round(t, 3), ZOOM_PUNCH if kind == "punch" else base))
    if len(steps) > 1 and steps[1][0] == 0.0:
        steps = steps[1:]  # an opening punch replaces the base start
    return [st for st in steps if st[0] < duration]  # e.g. a release past the end


def zoom_expr(
    steps: Sequence[Sequence[float]],
    offset: float = 0.0,
    length: float | None = None,
    level: Callable[[float], float] | None = None,
    instant: Sequence[float] = (),
) -> str:
    """Eased scale expression S(t) for `steps`, in scene time (`t` = output time - `offset`).
    A step eases (smoothstep) from the previous scale over ZOOM_EASE s (PUNCH_RELEASE s when
    leaving a punch), cut short when the next step comes sooner. A step at an `instant` time
    (output s: the cuts) changes in one frame, INSTANT_LEAD s early so the cut's first frame
    already shows it. `level` maps a step's scale to the rendered zoom (a scene's
    `SceneLayout.zoom_level`). With `length`, only points that matter for [0, length] are
    emitted. Commas are NOT escaped."""
    if not steps:
        return "1"
    lv = level or (lambda v: v)
    seq = [(float(t), float(v)) for t, v in steps]
    cut = [any(abs(t - c) < 1e-3 for c in instant) for t, _ in seq]
    at = [t - INSTANT_LEAD if c else t for (t, _), c in zip(seq, cut, strict=True)]
    pts = [(at[0], lv(seq[0][1]))]
    for i, ((_, s0), (_, s1)) in enumerate(pairwise(seq)):
        t1 = at[i + 1]
        d = 0.0 if cut[i + 1] else PUNCH_RELEASE if s0 == ZOOM_PUNCH else ZOOM_EASE
        nxt = at[i + 2] if i + 2 < len(seq) else math.inf
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


def _box_px(
    box: tuple[float, float, float, float], in_w: int, in_h: int
) -> tuple[int, int, int, int]:
    """A normalized (x, y, w, h) box as an even ffmpeg crop (w, h, x, y) inside in_w x in_h."""
    w, h = min(_even(box[2] * in_w), _even(in_w)), min(_even(box[3] * in_h), _even(in_h))
    x = min(max(round(box[0] * in_w), 0), in_w - w)
    y = min(max(round(box[1] * in_h), 0), in_h - h)
    return w, h, x, y


def full_cam_geometry(layout: SceneLayout, in_w: int, in_h: int) -> tuple[float, float]:
    """(height, top y) in output px of a full-cam beat's cam picture: the whole cam
    (`cam_rect`) at the full output width, centred, raised so the chin (the head box's
    bottom) lands at BEAT_CHIN_Y at most -- the caption goes under the chin."""
    assert layout.cam_rect is not None
    cw, ch, _, cy = _box_px(layout.cam_rect, in_w, in_h)
    fg_h = _even(OUT_W * ch / cw)
    y = (OUT_H - fg_h) / 2
    if layout.head_box is not None:
        _, hy, _, hh = layout.head_box
        chin = ((hy + hh) * in_h - cy) / ch * fg_h
        y = min(y, BEAT_CHIN_Y - chin)
    return fg_h, float(max(int(y), 0))


def _track_w(in_w: int, in_h: int) -> int:
    return min(_even(in_h * 9 / 16), _even(in_w))


def scene_upscale(layout: SceneLayout, in_w: int, in_h: int) -> float:
    """Largest source-to-output upscale factor among the scene's panels (zoom excluded)."""
    kind = _kind(layout)
    if kind == "STACKED":
        cw, ch, _, _ = _cam_crop(layout, in_w, in_h)
        if layout.game_box is not None:
            gw, gh, _, _ = _box_px(layout.game_box, in_w, in_h)
            game = OUT_W / gw if not layout.game_fit else min(OUT_W / gw, STACK_GAME_H / gh)
        else:
            game = OUT_W / _aspect_crop(in_w, in_h, OUT_W, STACK_GAME_H)[0]
        return max(OUT_W / cw, STACK_CAM_H / ch, game)
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
    cut_times: Sequence[float] = (),
) -> str:
    """Filter chain [label_in] (already trimmed to the scene) -> [label_out] at 1080x1920, SAR 1.

    `zoom_steps` (output time, from `zoom_schedule`) zoom the speaker panel after the layout,
    each level mapped through the scene's plan (`plan_zoom`, `SceneLayout.zoom_level`) and
    scaled about its anchor row (`SceneLayout.anchor_px`): FACE_TRACK whole frame, STACKED
    facecam panel, both SPLIT panels, FIT_BLUR foreground. Steps at `cut_times` (output s)
    and at the scene's own bounds (camera changes) land in one frame, the others ease. Pass steps already through `guard_zoom` (the edge
    guard's holds; the same steps go to the sidecar). A FACE_TRACK crop follows its track with
    the edge guard and gaze lead (`_track_plan`, the rendered zoom at each moment) plus the
    guard's `shift_steps` (instant at cuts). Panels upscaled more than SHARPEN_ABOVE are
    sharpened after the zoom, so the final pixels are the sharpened ones; scenes upscaled
    more than UPSCALE_SOFT scale with lanczos."""
    kind = _kind(layout)
    track: dict = {}
    if kind == "FACE_TRACK":
        if cut_times and not layout.cuts:
            layout = layout.model_copy(update={"cuts": [float(c) for c in cut_times]})
        track = _track_kw(layout, in_w, in_h, zoom_steps)
    if zoom_steps:
        zoom = zoom_expr(
            zoom_steps,
            offset=layout.start,
            length=layout.end - layout.start,
            level=layout.zoom_level,
            # a step on a camera change is like one on a cut: no ease across it
            instant=[*cut_times, layout.start, layout.end],
        )
    else:
        zoom = _fmt(layout.zoom_level(1.0))
    flags = LANCZOS if scene_upscale(layout, in_w, in_h) > UPSCALE_SOFT else ""
    a, b = f"[{label_out}_a]", f"[{label_out}_b]"
    if kind == "STACKED" and layout.game_box is not None:
        return _template_filter(layout, in_w, in_h, label_in, label_out, zoom, flags)
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
        cw = track["crop_w"]
        x = track_x_expr(**track)
        if layout.step_shift and zoom_steps:
            x = _shifted(x, layout, zoom_steps, in_w, cw, cut_times)
        if layout.shift_steps:
            alt = [(layout.start, 0.0), *((t, v * cw) for t, v in layout.shift_steps)]
            off = zoom_expr(
                alt, offset=layout.start, length=layout.end - layout.start, instant=cut_times
            )
            x = f"min(max(({x})+({off}),0),{_fmt(max(in_w - cw, 0))})"
        x = x.replace(",", r"\,")
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


def _template_filter(
    layout: SceneLayout, in_w: int, in_h: int, label_in: str, label_out: str, zoom: str, flags: str
) -> str:
    """A stream profile's fixed STACKED template: the cam panel (the cam window, zoomed about
    its anchor row), over the gameplay panel (its source box with the overlay `patches` in it
    blurred; `game_fit`: shown whole over a blurred fill), and during `cam_beats` the whole
    cam full-width over its own blurred fill (`full_cam_geometry`)."""
    o = label_out
    beats = layout.cam_beats
    cw, ch, cx, cy = _cam_crop(layout, in_w, in_h)
    gw, gh, gx, gy = _box_px(layout.game_box, in_w, in_h)  # type: ignore[arg-type]
    parts = [f"[{label_in}]split={3 if beats else 2}[{o}_a][{o}_b]" + (f"[{o}_c]" if beats else "")]
    cam_up = max(OUT_W / cw, STACK_CAM_H / ch)
    parts.append(
        f"[{o}_a]crop={cw}:{ch}:{cx}:{cy},{_fill(OUT_W, STACK_CAM_H, flags)}"
        f"{_zoom(zoom, OUT_W, STACK_CAM_H, layout.anchor_px(STACK_CAM_H), flags)}"
        f"{_sharpen(cam_up)}[{o}_cam]"
    )
    chain, cur = f"[{o}_b]crop={gw}:{gh}:{gx}:{gy}", f"{o}_g0"
    parts.append(f"{chain}[{cur}]")
    for k, patch in enumerate(layout.patches):
        pw, ph, px, py = _box_px(patch, in_w, in_h)
        px, py = min(max(px - gx, 0), max(gw - pw, 0)), min(max(py - gy, 0), max(gh - ph, 0))
        pw, ph = min(pw, gw), min(ph, gh)
        nxt = f"{o}_g{k + 1}"
        parts.append(
            f"[{cur}]split=2[{cur}a][{cur}b];[{cur}b]crop={pw}:{ph}:{px}:{py},{PATCH_BLUR}[{cur}p];"
            f"[{cur}a][{cur}p]overlay={px}:{py}[{nxt}]"
        )
        cur = nxt
    if layout.game_fit:
        fh = _even(OUT_W * gh / gw)
        fw = OUT_W
        if fh > STACK_GAME_H:
            fw, fh = _even(STACK_GAME_H * gw / gh), STACK_GAME_H
        parts.append(
            f"[{cur}]split=2[{o}_fb][{o}_ff];[{o}_fb]{_fill(OUT_W, STACK_GAME_H)},{BLUR}[{o}_fbo];"
            f"[{o}_ff]scale={fw}:{fh}{flags}{_sharpen(fw / gw)}[{o}_ffo];"
            f"[{o}_fbo][{o}_ffo]overlay=(W-w)/2:(H-h)/2[{o}_game]"
        )
    else:
        parts.append(f"[{cur}]scale={OUT_W}:{STACK_GAME_H}{flags}{_sharpen(OUT_W / gw)}[{o}_game]")
    if not beats:
        parts.append(f"[{o}_cam][{o}_game]vstack,setsar=1[{o}]")
        return ";".join(parts)
    parts.append(f"[{o}_cam][{o}_game]vstack[{o}_st]")
    rw, rh, rx, ry = _box_px(layout.cam_rect, in_w, in_h)  # type: ignore[arg-type]
    fg_h, top = full_cam_geometry(layout, in_w, in_h)
    when = "+".join(
        f"between(t\\,{_fmt(a - layout.start)}\\,{_fmt(b - layout.start)})" for a, b in beats
    )
    parts.append(
        f"[{o}_c]crop={rw}:{rh}:{rx}:{ry},split=2[{o}_cb][{o}_cf];"
        f"[{o}_cb]{_fill(OUT_W, OUT_H)},{BLUR}[{o}_cbo];"
        f"[{o}_cf]scale={OUT_W}:{_even(fg_h)}{LANCZOS}{_sharpen(OUT_W / rw)}[{o}_cfo];"
        f"[{o}_cbo][{o}_cfo]overlay=0:{int(top)}[{o}_full];"
        f"[{o}_st][{o}_full]overlay=0:0:enable='{when}',setsar=1[{o}]"
    )
    return ";".join(parts)


def _shifted(
    x: str,
    layout: SceneLayout,
    steps: Sequence[Sequence[float]],
    in_w: int,
    cw: int,
    cut_times: Sequence[float] = (),
) -> str:
    """Crop-left expression `x` plus a sideways shift that alternates 0 / `step_shift` (signed)
    of the crop width at every zoom step, eased by smoothstep over ZOOM_EASE s both ways (a
    pixel offset is never ZOOM_PUNCH, so PUNCH_RELEASE never applies; instant at
    `cut_times`), clamped to the frame."""
    shift = layout.step_shift * cw
    alt = [(float(t), shift if i % 2 else 0.0) for i, (t, _) in enumerate(steps)]
    off = zoom_expr(alt, offset=layout.start, length=layout.end - layout.start, instant=cut_times)
    return f"min(max(({x})+({off}),0),{_fmt(max(in_w - cw, 0))})"


def _rendered_track(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]] | None
) -> tuple[list[float], list[float], list[int]]:
    """A FACE_TRACK scene's samples as rendered under `steps`: (scene times, crop centres
    (share of the source width) with every crop offset in effect -- flat-scene step shift and
    guard `shift_steps` -- clamped to the frame, indices into `layout.track_x` sorted by
    time)."""
    kw = _track_kw(layout, in_w, in_h, steps)
    plan = _plan_track(**kw).points
    off = kw["offset"] or (lambda _t: 0.0)
    cwn = kw["crop_w"] / in_w
    lo_c, hi_c = (cwn / 2, 1 - cwn / 2) if cwn < 1 else (0.5, 0.5)
    shifts = sorted((float(a), float(v)) for a, v in layout.shift_steps)

    def pulse(u: float) -> float:
        on = [v for a, v in shifts if a <= u + layout.start + 1e-9]
        return (on[-1] if on else 0.0) * cwn

    order = sorted(range(len(layout.track_x)), key=lambda i: layout.track_x[i][0])
    centres = [min(max(c + off(t) + pulse(t), lo_c), hi_c) for t, c in plan]
    return [t for t, _ in plan], centres, order


def face_offset(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]]
) -> float | None:
    """A FACE_TRACK scene's rendered face centre: the median offset of the (filtered) face
    from the crop centre over its samples, as a share of the crop width (+ right), with the
    crop shifts in effect at each sample (guard `shift_steps`, flat-scene `step_shift`)
    applied; None for other scenes."""
    if _kind(layout) != "FACE_TRACK" or not layout.track_x:
        return None
    cwn = _track_w(in_w, in_h) / in_w
    _, centres, _ = _rendered_track(layout, in_w, in_h, steps)
    rel = [(t - layout.start, x) for t, x in layout.track_x]
    _, xs, _ = _track_faces(rel, layout.face_w, layout.track_w or None)
    offs = sorted((x - c) / cwn for x, c in zip(xs, centres, strict=True))
    return round(offs[len(offs) // 2], 4)


def face_track_px(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]]
) -> list[list[float | None]]:
    """Per face sample of a FACE_TRACK scene with per-sample face rows (`track_face`), the
    rendered face in output px: [output t, x0, core top, x1, bottom, eye line y (None
    without landmarks)], mapped through the crop path (snaps, step shift, guard shifts), the
    zoom in effect (eases counted: `zoom_at_fn`, the larger face) about the scene's anchor
    row; the core top is the detector box minus its top FACE_CORE_TOP (hairline), which
    placement and Q4b use when there is no eye line. [] for other scenes."""
    if _kind(layout) != "FACE_TRACK" or not layout.track_x or not layout.track_face:
        return []
    if len(layout.track_face) != len(layout.track_x):
        return []
    cw = _track_w(in_w, in_h)
    ts, centres, order = _rendered_track(layout, in_w, in_h, steps)
    rel = [(t - layout.start, x) for t, x in layout.track_x]
    _, xs, ws = _track_faces(rel, layout.face_w, layout.track_w or None)
    rows = [layout.track_face[i] for i in order]
    ys = _median3([r[0] for r in rows])
    hs = _median3([r[1] for r in rows])
    raw_eyes = [r[2] for r in rows]  # None: no landmarks (the caption keeps the 20% core)
    known = all(e is not None for e in raw_eyes)
    eyes = _median3(raw_eyes) if known else raw_eyes  # type: ignore[arg-type]
    zoom = zoom_at_fn(steps or [(0.0, 1.0)], layout)
    ay = layout.anchor_px(OUT_H)
    out = []
    for t, c, x, w, y, h, eye in zip(ts, centres, xs, ws, ys, hs, eyes, strict=True):
        z = zoom(t)
        left = c * in_w - cw / 2

        def px(v: float, z: float = z, left: float = left) -> float:
            return OUT_W / 2 + z * ((v * in_w - left) / cw * OUT_W - OUT_W / 2)

        def py(v: float, z: float = z) -> float:
            return ay + z * (v * OUT_H - ay)

        top = y + FACE_CORE_TOP * h
        out.append(
            [round(t + layout.start, 3), round(px(x - w / 2)), round(py(top)),
             round(px(x + w / 2)), round(py(y + h)), None if eye is None else round(py(eye))]
        )  # fmt: skip
    return out


def stacked_face_px(
    layout: SceneLayout, in_w: int, in_h: int, steps: Sequence[Sequence[float]]
) -> list[list[float | None]]:
    """A template STACKED scene's head (`head_box`: top of the hair to the chin) as rendered,
    every SAMPLE_STEP s: [output t, x0, top, x1, bottom, None] in output px -- in the cam
    panel at the zoom in effect (eases counted, about the anchor row; clipped to the panel),
    during a full-cam beat on the full-cam picture. Captions and Q4b keep off it (the whole
    head, not just the eyes-to-mouth core: the cam panel is small). [] for other scenes."""
    if _kind(layout) != "STACKED" or layout.head_box is None or layout.facecam is None:
        return []
    hx, hy, hw, hh = layout.head_box
    fx, fy, fw, fh = layout.facecam
    zoom = zoom_at_fn(steps or [(0.0, 1.0)], layout)
    ay = layout.anchor_px(STACK_CAM_H)
    full = None
    if layout.cam_beats and layout.cam_rect is not None:
        full = full_cam_geometry(layout, in_w, in_h)
        fg_h, top = full
        rx, ry, rw, rh = layout.cam_rect
    out: list[list[float | None]] = []
    n = max(math.ceil((layout.end - layout.start) / SAMPLE_STEP - 1e-9), 1)
    for k in range(n + 1):
        t = min(layout.start + k * SAMPLE_STEP, layout.end)
        if full is not None and any(a <= t < b for a, b in layout.cam_beats):
            row = [
                (hx - rx) / rw * OUT_W, top + (hy - ry) / rh * fg_h,
                (hx + hw - rx) / rw * OUT_W, top + (hy + hh - ry) / rh * fg_h,
            ]  # fmt: skip
        else:
            z = zoom(t - layout.start)

            def px(v: float, z: float = z) -> float:
                return min(max(OUT_W / 2 + z * ((v - fx) / fw * OUT_W - OUT_W / 2), 0.0), OUT_W)

            def py(v: float, z: float = z) -> float:
                return min(max(ay + z * ((v - fy) / fh * STACK_CAM_H - ay), 0.0), STACK_CAM_H)

            row = [px(hx), py(hy), px(hx + hw), py(hy + hh)]
        out.append([round(t, 3), *(round(v) for v in row), None])
    return out


def rendered_zoom_steps(
    layouts: Sequence[SceneLayout], steps: Sequence[Sequence[float]]
) -> list[tuple[float, float]]:
    """`steps` (schedule levels) as rendered: each level through the zoom plan of the scene
    on screen at its time (`SceneLayout.zoom_level`; soft-source caps, base zooms), 4 dp."""
    out = []
    for t, s in steps:
        lay = next((x for x in layouts if x.start - 1e-6 <= t < x.end), layouts[-1])
        out.append((float(t), round(lay.zoom_level(float(s)), 4)))
    return out


def _panel_y(layout: SceneLayout, y: float, in_w: int, in_h: int) -> tuple[float, float]:
    """(unzoomed panel px, panel height) of normalized source row `y`: the full frame height
    for FACE_TRACK, the vertically centred crop for SPLIT."""
    if _kind(layout) == "SPLIT":
        sh = _aspect_crop(in_w, in_h, OUT_W, SPLIT_H)[1]
        return (y * in_h - (in_h - sh) // 2) / sh * SPLIT_H, SPLIT_H
    if _kind(layout) == "STACKED" and layout.facecam is not None:  # the cam panel
        _, fy, _, fh = layout.facecam
        return (y - fy) / fh * STACK_CAM_H, STACK_CAM_H
    return y * OUT_H, OUT_H


def _headroom(layout: SceneLayout, in_w: int, in_h: int) -> tuple[float, float] | None:
    """(largest zoom <= the scene's top level, anchor px) that keeps the head top (the
    scene's highest, `head_top`) HEADROOM below the panel top and the median chin inside the
    panel at every level up to it, the anchor as close to the face centre as allowed. None
    without face geometry. A head already within HEADROOM unzoomed (the source framing cut it)
    anchors at the top: zooming in then only moves it down."""
    if layout.face_box is None or layout.head_top is None:
        return None
    template = _kind(layout) == "STACKED" and layout.head_box is not None
    if _kind(layout) not in ("FACE_TRACK", "SPLIT") and not template:
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


def _shift_sign(layout: SceneLayout, in_w: int, in_h: int) -> int:
    """Direction (+1 crop right, -1 left, 0 none) of a FACE_TRACK scene's step shift: the face
    moves towards the crop centre (a crop pinned at a frame edge holds its face off-centre
    towards that edge), else, centred, towards the side with more room (median track
    position). 0 when the shift would move the crop over any part of the face core (`face_w`)
    at any track point (only what the unshifted crop showed counts), or can't move it at all."""
    cw = _track_w(in_w, in_h)
    hi = max(in_w - cw, 0)
    plan, _ = _track_plan(
        layout.track_x, in_w, cw, layout.face_w, layout.track_w or None, layout.lead,
        layout.zoom_level(ZOOM_TOP),
    )  # fmt: skip
    pts = [(t, min(max(c * in_w - cw / 2, 0.0), hi)) for t, c in plan]  # one per sample
    if not pts:
        return 0
    med = _median3([x for _, x in sorted(layout.track_x)])
    i = len(pts) // 2
    off = med[i] * in_w - (pts[i][1] + cw / 2)  # face centre from crop centre, px
    if abs(off) > 1:
        sign = 1 if off > 0 else -1
    else:
        sign = -1 if sorted(med)[len(med) // 2] > 0.5 else 1
    half = (layout.face_w or 0.0) * in_w / 2
    moved = False
    for (_, left), x in zip(pts, med, strict=True):
        new = min(max(left + sign * STEP_SHIFT * cw, 0.0), hi)
        lo_px, hi_px = max(x * in_w - half, left), min(x * in_w + half, left + cw)
        if new > lo_px + 1e-6 or hi_px > new + cw + 1e-6:
            return 0
        moved = moved or abs(new - left) >= 1
    return sign if moved else 0


def plan_zoom(layout: SceneLayout, in_w: int, in_h: int) -> SceneLayout:
    """The scene with its zoom plan (base_zoom, zoom_cap, zoom_anchor) filled in:

    - soft source (upscale > UPSCALE_SOFT): every level, punches included, <= ZOOM_IN;
    - small FACE_TRACK face (< SMALL_FACE of the output height) on a sharper source (upscale
      <= UPSCALE_SOFT; soft sources get none): a base zoom toward FACE_TARGET, at most
      MAX_BASE and leaving MIN_STEP_RANGE above it for the steps, with the step levels
      compressed into [base, MAX_ZOOM] and total upscale <= MAX_UPSCALE;
    - headroom: the anchor row (and if need be a lower cap) keeps the head top >= HEADROOM of
      the panel height below its top at every level (`_headroom`);
    - a FACE_TRACK scene whose levels end up (nearly) flat (top level <= base + FLAT_ZOOM)
      gets a signed `step_shift` (`_shift_sign`; none when it would trim the face): its steps
      shift the crop sideways instead, so cuts still change framing."""
    up = scene_upscale(layout, in_w, in_h)
    cap: float | None = ZOOM_IN if up > UPSCALE_SOFT else None
    base = 1.0
    face_h = layout.face_box[1] if layout.face_box else 0.0
    if _kind(layout) == "FACE_TRACK" and 0 < face_h < SMALL_FACE and up <= UPSCALE_SOFT:
        room = min(MAX_ZOOM, MAX_UPSCALE / up)
        base = max(1.0, min(FACE_TARGET / face_h, MAX_BASE, room - MIN_STEP_RANGE))
        if base > 1.0:
            cap = room
    if _kind(layout) == "FACE_TRACK" and layout.face_w:
        # the face (its wider moments, with jitter room) must fit the guard margin at the top
        # level, with FIT_SLACK to spare so the guard never flips to the close-up path there
        widths = sorted(_median3(layout.track_w)) if layout.track_w else []
        face = max(layout.face_w, widths[int(0.9 * (len(widths) - 1))] if widths else 0.0)
        fit = _track_w(in_w, in_h) / in_w * (1 - 2 * GUARD_MARGIN) * FIT_SLACK
        fit /= face + 2 * GUARD_JITTER
        if fit < (cap if cap is not None else ZOOM_TOP * base):
            cap = max(fit, base)
    plan = layout.model_copy(update={"base_zoom": math.floor(base * 1e4) / 1e4, "zoom_cap": cap})
    head = _headroom(plan, in_w, in_h)
    if head is None:
        return plan
    top, anchor = head
    if top < plan.zoom_level(ZOOM_TOP) - 1e-9:
        base, cap = max(1.0, min(base, top / ZOOM_IN)), top
    ph = _panel_y(layout, 0.0, in_w, in_h)[1]
    plan = plan.model_copy(
        update={
            "base_zoom": math.floor(base * 1e4) / 1e4,
            "zoom_cap": cap,
            "zoom_anchor": anchor / ph,
        }
    )
    if (
        _kind(layout) == "FACE_TRACK"
        and plan.zoom_level(ZOOM_TOP) <= plan.zoom_level(1.0) + FLAT_ZOOM
        and (sign := _shift_sign(plan, in_w, in_h))
    ):
        plan = plan.model_copy(update={"step_shift": sign * STEP_SHIFT})
    return plan


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
    cut_times: Sequence[float] = (),
) -> str:
    """Full -filter_complex: [0:v] (crop) -> split -> per-scene trim + scene_filter -> concat [vout].
    When `crop` is given, in_w/in_h are the cropped size. Scenes shorter than 1/30 s are merged
    into a neighbour so the timeline keeps its full length. `zoom_steps` (output time) are
    applied per scene, instant at `cut_times` (see `scene_filter`)."""
    layouts = _merge_short(layouts)
    n = len(layouts)
    pre = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else ""
    parts = [f"[0:v]{pre}split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i, lay in enumerate(layouts):
        parts.append(
            f"[s{i}]trim=start={_fmt(lay.start)}:end={_fmt(lay.end)},setpts=PTS-STARTPTS[t{i}]"
        )
        parts.append(scene_filter(lay, in_w, in_h, f"t{i}", f"v{i}", zoom_steps, cut_times))
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]")
    return ";".join(parts)
