"""VOD-level stream layout profile (wave G-layout), computed once per job and cached.

A streamer's facecam overlay is a fixed rectangle: the same small face in the same place for
most of the VOD. The profile finds it once -- sampling PROFILE_SAMPLES frames across the VOD
after the intro, clustering the face detections (`cam_cluster`), snapping to the overlay's
static border (`snap_cam`) and noting the label strip next to it (`label_masks`) -- so every
clip of the job renders the same fixed STACKED template instead of re-detecting a cam per
scene. The same frames give the VOD's static black bars (`vod_bars`). No profile cam (podcasts,
full-cam Just Chatting) leaves the per-scene layouts as they were."""

import json
import logging
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from hookline.fsutil import atomic_write_text
from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.vision.faces import Face, sample_frames

log = logging.getLogger(__name__)

PROFILE_VERSION = 3  # bump when the profile changes shape or meaning: cached ones re-run
PROFILE_FILE = "layout_profile.json"
PROFILE_SAMPLES = 150
INTRO_SKIP_S = 180.0  # the first minutes ("starting soon") are skipped ...
INTRO_SKIP_SHARE = 0.1  # ... but never more than this share of the VOD
DETECT_W = 640  # faces are detected on frames this wide (narrower sources as they are)
# The VOD is analysed on frames at most ANALYSIS_W wide (a 1440p / 4K source's 150 full-size
# frames would take GBs); a wider source's cam is then re-snapped on REFINE_SAMPLES full-size
# crops around it, so its boxes and border ring are exact source px.
ANALYSIS_W = 1280
REFINE_SAMPLES = 40
REFINE_PAD = 0.3  # the crop: the coarse cam grown this much of its size each way ...
REFINE_LABEL = 0.8  # ... and this much of its height above and below (its label strip)
MIN_SCORE = 0.6
# The cam face: a cluster of detections (members within CLUSTER_RADIUS face sizes of its
# median, each axis) present in >= CLUSTER_SHARE of the samples, its spread (pstdev) under
# CLUSTER_SPREAD face sizes and its face smaller than CAM_FACE_MAX of the frame height (a
# full-frame speaker is no cam). Face-relative, so a 1440p cam and a tiny one cluster alike.
CLUSTER_RADIUS = 0.8
CLUSTER_SPREAD = 0.35
CLUSTER_SHARE = 0.6
CAM_FACE_MAX = 0.25
# A corner cam, not a webcam shot: its outer box at most CAM_AREA_MAX of the frame, its face
# at least CAM_OFF_CENTRE face widths from the frame centre.
CAM_AREA_MAX = 0.35
CAM_OFF_CENTRE = 1.5
# Snapping: the cam rectangle is searched within SNAP_REACH face heights of the face, growing
# through the SNAP_REACHES until one finds it (a 16:9 cam reaches ~3 face heights sideways).
# A pixel is a static edge when its gradient (> EDGE_T) matches the median frame's in
# >= STATIC_SHARE of the samples -- or a still/live edge: a still pixel (mean |frame - median|
# under STILL_T: a smooth wall, a border) next to a live one (over LIVE_T: gameplay) within 2
# px, a borderless cam's edge. A border line has such edges along >= LINE_SHARE of a span
# centred on the face (SPAN of the face width either side) and the rectangle's 4 sides
# >= PERIM_SHARE. The frame's own edge is a side too (a cam flush in a corner) when the
# rectangle's band along it is still in >= STILL_SIDE of it (not a strip of gameplay).
SNAP_REACH = 2.5
SNAP_REACHES = (2.5, 4.0, 6.0, 9.0)
STILL_T = 6.0
LIVE_T = 10.0
STILL_SIDE = 0.6
EDGE_T = 20
EDGE_TOL = 0.3
STATIC_SHARE = 0.7
LINE_SHARE = 0.6
SPAN = 0.6
PERIM_SHARE = 0.7
MIN_GROW = 1.3  # the rectangle is at least this many face sizes each way
OUTSIDE_LIVE = 4.0  # mean |frame - median| just outside the rectangle: the picture moves
OUTSIDE_BAND = 6  # px
# Trimming inward past the border: up to TRIM_MAX uniform lines (p90 - p10 of the median
# frame along the line under TRIM_UNIFORM: the border and its shadow), then TRIM_SAFETY px.
TRIM_MAX = 5
TRIM_UNIFORM = 25
TRIM_SAFETY = 1
MASK_PAD = 2  # the cam keep-out mask is the outer rectangle grown by this (px)
# Label strip (name / alert text under or over the cam): rows within LABEL_REACH of the cam's
# height from its edge where >= LABEL_DENSITY of the columns have an edge in >= LABEL_PRESENT
# of the samples (a static frame, text changing), gaps up to LABEL_GAP px bridged, at most
# LABEL_MAX of the cam height tall, padded by LABEL_PAD px.
LABEL_REACH = 0.25
LABEL_PRESENT = 0.3
LABEL_DENSITY = 0.05
LABEL_GAP = 6
LABEL_MAX = 0.5
LABEL_PAD = 4
LABEL_WIDEN = 0.25  # the strip may be this much of the cam width wider each side
# VOD-level black bars: a line is content when > BAR_CONTENT of its pixels exceed BAR_LIMIT
# in some sample; bars under BAR_MIN of both dimensions are not cropped.
BAR_LIMIT = 16
BAR_CONTENT = 0.01
BAR_MIN = 0.02
# Cam visibility: RING_POINTS points per side on the border's outer RING_DEPTH lines; a frame
# shows the cam when >= RING_MATCH of them are within RING_TOL of the profile's median frame
# AND the frame's values there follow the reference's pattern (normalised correlation
# >= RING_CORR): a uniformly dark frame matches a dim reference by value alone. A reference
# with no pattern (std under RING_FLAT) counts only when it is not dark (mean > RING_DARK).
RING_POINTS = 24
RING_DEPTH = 3
RING_TOL = 40
RING_MATCH = 0.7
RING_CORR = 0.5
RING_FLAT = 6.0
RING_DARK = 48.0
# ... or the cam's own picture does: the same check RING_INSET px inside the trimmed cam (a
# white flash behind the overlay blends the border's edge pixels away, not the picture)
RING_INSET = 3
HAIR_SHARE = 0.5  # head top ~ face box top - half a face height (hair, a cap)
HEAD_WIDEN = 0.25  # the head is this much of the face width wider each side (hair, headset)

PxBox = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), source px


class _Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Face]: ...


class LayoutProfile(BaseModel):
    """Source-px geometry of a VOD's layout. `bars`: the static black-bar crop (w, h, x, y)
    or None. `cam`: the facecam's picture inside its border; `outer`: the border's outside;
    `head`: the streamer's head (top of hair to chin) in the cam; `masks`: overlay strips next
    to the cam (labels); `ring` / `ring_in`: [x, y, value] points of the median frame on the
    border and just inside the cam's picture (the cam is shown where either matches). All
    None / empty without a cam."""

    size: tuple[int, int]
    bars: tuple[int, int, int, int] | None = None
    cam: PxBox | None = None
    outer: PxBox | None = None
    head: PxBox | None = None
    face: PxBox | None = None
    masks: list[PxBox] = []
    ring: list[tuple[int, int, int]] = []
    ring_in: list[tuple[int, int, int]] = []
    share: float | None = None  # share of the samples showing the cam face

    def keep_out(self) -> list[PxBox]:
        """What the gameplay panel must not show: the cam (outer, grown) and its labels."""
        if self.outer is None:
            return []
        w, h = self.size
        x0, y0, x1, y1 = self.outer
        grown = (
            max(x0 - MASK_PAD, 0), max(y0 - MASK_PAD, 0),
            min(x1 + MASK_PAD, w), min(y1 + MASK_PAD, h),
        )  # fmt: skip
        return [grown, *self.masks]


def cam_cluster(samples: Sequence[Sequence[Face]]) -> tuple[Face, float] | None:
    """The cam face: the median face of the densest detection cluster (one member per sample,
    the nearest), with the share of samples it is in; None when no cluster is in
    >= CLUSTER_SHARE of the samples with a spread under CLUSTER_SPREAD face sizes and a small
    face. Distances are in face sizes (the seed's width / height)."""
    dets = [(i, f) for i, faces in enumerate(samples) for f in faces if f.score >= MIN_SCORE]
    if not samples or not dets:
        return None
    best: list[tuple[int, Face]] = []
    for _, seed in dets:
        cx, cy, members = seed.cx, seed.cy, []
        rx, ry = CLUSTER_RADIUS * seed.w, CLUSTER_RADIUS * seed.h
        for _ in range(3):  # shift to the neighbourhood's median, then settle
            members = [(i, f) for i, f in dets if abs(f.cx - cx) <= rx and abs(f.cy - cy) <= ry]
            cx = statistics.median(f.cx for _, f in members)
            cy = statistics.median(f.cy for _, f in members)
        per: dict[int, Face] = {}
        for i, f in members:
            if i not in per or abs(f.cx - cx) + abs(f.cy - cy) < abs(per[i].cx - cx) + abs(
                per[i].cy - cy
            ):
                per[i] = f
        if len(per) > len(best):
            best = list(per.items())
    share = len(best) / len(samples)
    faces = [f for _, f in best]
    if share < CLUSTER_SHARE or len(faces) < 2:
        return None
    fw, fh = statistics.median(f.w for f in faces), statistics.median(f.h for f in faces)
    spread = max(
        statistics.pstdev(f.cx for f in faces) / max(fw, 1e-9),
        statistics.pstdev(f.cy for f in faces) / max(fh, 1e-9),
    )
    if spread >= CLUSTER_SPREAD:
        return None
    med = Face(
        statistics.median(f.x for f in faces),
        statistics.median(f.y for f in faces),
        statistics.median(f.w for f in faces),
        statistics.median(f.h for f in faces),
        statistics.median(f.score for f in faces),
        None,
        statistics.median(f.eye_y for f in faces) if all(f.eye_y for f in faces) else None,
    )
    if med.h >= CAM_FACE_MAX:
        return None
    return med, round(share, 3)


def _gradient(gray: np.ndarray) -> np.ndarray:
    """max(|dx|, |dy|) per pixel (a vertical step belongs to the row below, a horizontal one
    to the pixel left of it), as in `reframe._gradient`."""
    grad = np.zeros_like(gray)
    grad[..., :, :-1] = np.abs(np.diff(gray, axis=-1))
    grad[..., 1:, :] = np.maximum(grad[..., 1:, :], np.abs(np.diff(gray, axis=-2)))
    return grad


def _static(gray: np.ndarray, motion: bool = False) -> np.ndarray:
    """(H, W) bool: an edge matching the median frame's gradient in >= STATIC_SHARE of the
    (n, H, W) samples; with `motion` also a still pixel within 2 px of a live one along a row
    or column (the edge of a borderless cam over a smooth wall, gameplay moving next to it)."""
    g = gray.astype(np.float32)
    grad = _gradient(g)
    med = np.median(grad, axis=0)
    match = (
        (grad > EDGE_T) & (med > EDGE_T) & (np.abs(grad - med) < EDGE_TOL * np.maximum(grad, med))
    )
    static = match.mean(axis=0) >= STATIC_SHARE
    if not motion:
        return static
    del grad, match
    still, live = _motion_masks(g)
    edge = np.zeros_like(still)
    for d in (1, 2):
        edge[:, :-d] |= still[:, :-d] & live[:, d:]  # live to the right
        edge[:, d:] |= still[:, d:] & live[:, :-d]  # live to the left
        edge[:-d, :] |= still[:-d, :] & live[d:, :]  # live below
        edge[d:, :] |= still[d:, :] & live[:-d, :]  # live above
    return static | edge


def _motion_masks(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(still, live) (H, W) masks of float samples: mean |frame - median frame| under STILL_T
    / over LIVE_T."""
    mad = np.abs(g - np.median(g, axis=0)).mean(axis=0)
    return mad < STILL_T, mad > LIVE_T


def _outer_ends(
    share: np.ndarray, lo: int, hi: int, reverse: bool, at_edge: bool = False
) -> list[int]:
    """Indices in [lo, hi) where a run of lines with `share` >= LINE_SHARE starts, seen from
    the outside (from index lo upward, or from hi downward when `reverse`), the line before
    it (outside) not one: the outer edge of a border line. `at_edge`: the region's first line
    (seen from outside) is the frame's edge, an outer edge whatever its share (a cam flush
    with the frame; `snap_cam` checks that side differently)."""
    idx = range(hi - 1, lo - 1, -1) if reverse else range(lo, hi)
    out, prev = [], True  # the region's first line has no outside: never an outer edge ...
    for k, i in enumerate(idx):
        on = bool(share[i] >= LINE_SHARE)
        if on and not prev:
            out.append(i)
        prev = on
        if k == 0 and at_edge:
            out.append(i)  # ... unless it is the frame's edge
    return sorted(set(out))


def _uniform(line: np.ndarray) -> bool:
    return len(line) > 0 and float(np.percentile(line, 90) - np.percentile(line, 10)) < (
        TRIM_UNIFORM
    )


def snap_cam(
    gray: np.ndarray,
    face: tuple[float, float, float, float],
    origin: tuple[int, int] = (0, 0),
    frame: tuple[int, int] | None = None,
) -> tuple[PxBox, PxBox] | None:
    """(outer, inner) px boxes of the overlay rectangle around the cam `face` (x, y, w, h px)
    in the (n, H, W) gray samples: the largest rectangle containing the face, at least
    MIN_GROW face sizes each way, whose 4 sides are static border lines (static edges along
    >= PERIM_SHARE of each; a side on the frame's edge: a still band instead), the picture
    just outside it moving. `inner` is trimmed past the border (uniform lines, then
    TRIM_SAFETY px). The search reaches SNAP_REACHES face heights from the face, the nearest
    reach that finds one wins. None when there is no such rectangle, or only ones covering
    more than CAM_AREA_MAX of the frame.

    `gray` may be a crop of the frame: `origin` is its top-left and `frame` the frame's size
    (px; default: the crop is the frame); `face` and the boxes are in frame px."""
    _, height, width = gray.shape
    ox, oy = origin
    size = frame or (width + ox, height + oy)
    local = (face[0] - ox, face[1] - oy, face[2], face[3])
    # border lines first (a label strip's still box must not extend the cam), then still/live
    # edges (a borderless cam)
    for motion, mult in [(m, r) for m in (False, True) for r in SNAP_REACHES]:
        found = _snap_within(gray, local, mult * face[3], origin, size, motion)
        if found is not None:
            outer, inner = found
            return (
                (outer[0] + ox, outer[1] + oy, outer[2] + ox, outer[3] + oy),
                (inner[0] + ox, inner[1] + oy, inner[2] + ox, inner[3] + oy),
            )
    return None


def _snap_within(
    gray: np.ndarray,
    face: tuple[float, float, float, float],
    reach: float,
    origin: tuple[int, int],
    frame: tuple[int, int],
    motion: bool,
) -> tuple[PxBox, PxBox] | None:
    """`snap_cam` within `reach` px of the face centre (`motion`: still/live edges count as
    border lines too); boxes in `gray` px."""
    _, height, width = gray.shape
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    rx0, rx1 = max(int(cx - reach), 0), min(math.ceil(cx + reach), width)
    ry0, ry1 = max(int(cy - reach), 0), min(math.ceil(cy + reach), height)
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return None
    region = gray[:, ry0:ry1, rx0:rx1]
    s = _static(region, motion)
    still, _ = _motion_masks(region.astype(np.float32))
    # the region's sides that are the frame's own edges
    e_left, e_top = rx0 + origin[0] == 0, ry0 + origin[1] == 0
    e_right, e_bottom = rx1 + origin[0] == frame[0], ry1 + origin[1] == frame[1]
    # border-line shares over a span centred on the face (inside any cam around it)
    sx0, sx1 = int(cx - SPAN * fw) - rx0, math.ceil(cx + SPAN * fw) - rx0
    sy0, sy1 = int(cy - SPAN * fh) - ry0, math.ceil(cy + SPAN * fh) - ry0
    rows = s[:, max(sx0, 0) : max(sx1, 1)].mean(axis=1)
    cols = s[max(sy0, 0) : max(sy1, 1), :].mean(axis=0)
    f_x0, f_x1 = int(fx) - rx0, math.ceil(fx + fw) - rx0
    f_y0, f_y1 = int(fy) - ry0, math.ceil(fy + fh) - ry0
    max_area = CAM_AREA_MAX * frame[0] * frame[1]
    d = RING_DEPTH
    around = ((e_left, e_top, e_right, e_bottom), (rx0, ry0, rx1, ry1))

    def side(line: np.ndarray) -> float:
        return float(line.mean()) if line.size else 0.0

    def sides_ok(t: int, b: int, le: int, r: int) -> bool:
        sx, sy = slice(le, r + 1), slice(t, b + 1)
        bands = (
            (t == 0 and e_top, s[t : t + d, sx].any(axis=0), still[t : t + d, sx].all(axis=0)),
            (b == len(rows) - 1 and e_bottom, s[max(b - d + 1, 0) : b + 1, sx].any(axis=0),
             still[max(b - d + 1, 0) : b + 1, sx].all(axis=0)),
            (le == 0 and e_left, s[sy, le : le + d].any(axis=1), still[sy, le : le + d].all(axis=1)),
            (r == len(cols) - 1 and e_right, s[sy, max(r - d + 1, 0) : r + 1].any(axis=1),
             still[sy, max(r - d + 1, 0) : r + 1].all(axis=1)),
        )  # fmt: skip
        return all(
            side(st) >= STILL_SIDE if edge else side(line) >= PERIM_SHARE
            for edge, line, st in bands
        )

    best: tuple[int, PxBox] | None = None
    # border lines only first; the frame's edges as sides only when that finds nothing (a
    # still strip of gameplay between a cam and the frame edge must not join the cam)
    for edges in ((False,) * 4, (e_top, e_bottom, e_left, e_right)):
        if best is not None:
            break
        tops = _outer_ends(rows, 0, max(f_y0, 0), reverse=False, at_edge=edges[0])
        bottoms = _outer_ends(rows, min(f_y1, len(rows)), len(rows), reverse=True, at_edge=edges[1])
        lefts = _outer_ends(cols, 0, max(f_x0, 0), reverse=False, at_edge=edges[2])
        rights = _outer_ends(cols, min(f_x1, len(cols)), len(cols), reverse=True, at_edge=edges[3])
        for t in tops:
            for b in bottoms:
                if b - t + 1 < MIN_GROW * fh:
                    continue
                for le in lefts:
                    for r in rights:
                        if r - le + 1 < MIN_GROW * fw:
                            continue
                        area = (b - t + 1) * (r - le + 1)
                        if area > max_area or (best is not None and area <= best[0]):
                            continue
                        box = (le + rx0, t + ry0, r + 1 + rx0, b + 1 + ry0)
                        # a borderless edge needs gameplay all round: not the streamer's
                        # outline on a still wall
                        if sides_ok(t, b, le, r) and (
                            not motion or _live_around(gray, box, around)
                        ):
                            best = (area, box)
    if best is None:
        return None
    outer = best[1]
    if _outside_motion(gray, outer) < OUTSIDE_LIVE:
        return None
    return outer, _trim(np.median(gray, axis=0), outer)


def _live_around(
    gray: np.ndarray, box: PxBox, around: tuple[tuple[bool, bool, bool, bool], PxBox]
) -> bool:
    """Whether the picture just outside every side of `box` (a band RING_DEPTH..OUTSIDE_BAND
    px out; sides on the frame's edge skipped) is live (mean |frame - median| over LIVE_T)
    along >= LINE_SHARE of it: a borderless cam's edge has gameplay all round it."""
    _, height, width = gray.shape
    edges, region = around
    x0, y0, x1, y1 = box
    a, b = RING_DEPTH, OUTSIDE_BAND
    at_edge = (
        x0 == region[0] and edges[0], y0 == region[1] and edges[1],
        x1 == region[2] and edges[2], y1 == region[3] and edges[3],
    )  # fmt: skip
    bands = (
        (slice(y0, y1), slice(max(x0 - b, 0), max(x0 - a, 0))),
        (slice(max(y0 - b, 0), max(y0 - a, 0)), slice(x0, x1)),
        (slice(y0, y1), slice(min(x1 + a, width), min(x1 + b, width))),
        (slice(min(y1 + a, height), min(y1 + b, height)), slice(x0, x1)),
    )
    for skip, (sy, sx) in zip(at_edge, bands, strict=True):
        if skip:
            continue
        band = gray[:, sy, sx].astype(np.float32)
        if band.size == 0:
            return False
        live = np.abs(band - np.median(band, axis=0)).mean(axis=0) > LIVE_T
        if float(live.mean()) < LINE_SHARE:
            return False
    return True


def _outside_motion(gray: np.ndarray, box: PxBox) -> float:
    """Mean |frame - median frame| in a band OUTSIDE_BAND px wide around `box`."""
    _, height, width = gray.shape
    x0, y0, x1, y1 = box
    b = OUTSIDE_BAND
    mask = np.zeros((height, width), bool)
    mask[max(y0 - b, 0) : min(y1 + b, height), max(x0 - b, 0) : min(x1 + b, width)] = True
    mask[y0:y1, x0:x1] = False
    if not mask.any():
        return 0.0
    vals = gray[:, mask].astype(np.float32)
    return float(np.abs(vals - np.median(vals, axis=0)).mean())


def _trim(med: np.ndarray, outer: PxBox) -> PxBox:
    """`outer` moved inward past its border: up to TRIM_MAX uniform lines per side (border,
    shadow), then TRIM_SAFETY px."""
    x0, y0, x1, y1 = outer
    t, b, le, r = y0, y1 - 1, x0, x1 - 1
    for _ in range(TRIM_MAX):
        if _uniform(med[t, le : r + 1]):
            t += 1
        if _uniform(med[b, le : r + 1]):
            b -= 1
        if _uniform(med[t : b + 1, le]):
            le += 1
        if _uniform(med[t : b + 1, r]):
            r -= 1
    k = TRIM_SAFETY
    return le + k, t + k, r + 1 - k, b + 1 - k


def label_masks(gray: np.ndarray, outer: PxBox) -> list[PxBox]:
    """Overlay strips touching the cam (`outer` px box) below or above it: a run of rows within
    LABEL_REACH of the cam's height from its edge where edges sit at fixed places in a good
    share of the samples (a static label, its text changing). Each mask runs from the cam's
    edge to the strip's far end (+ LABEL_PAD) across the cam's and the strip's columns."""
    _, height, width = gray.shape
    x0, y0, x1, y1 = outer
    cam_w, cam_h = x1 - x0, y1 - y0
    wx0, wx1 = (
        max(int(x0 - LABEL_WIDEN * cam_w), 0),
        min(math.ceil(x1 + LABEL_WIDEN * cam_w), width),
    )
    depth = math.ceil((LABEL_REACH + LABEL_MAX) * cam_h)
    out = []
    for below in (True, False):
        lo, hi = (y1, min(y1 + depth, height)) if below else (max(y0 - depth, 0), y0)
        if hi - lo < 2:
            continue
        region = gray[:, lo:hi, wx0:wx1].astype(np.float32)
        present = (_gradient(region) > EDGE_T).mean(axis=0) >= LABEL_PRESENT  # (rows, cols)
        flagged = present.mean(axis=1) >= LABEL_DENSITY
        order = list(range(hi - lo)) if below else list(range(hi - lo - 1, -1, -1))
        reach = math.ceil(LABEL_REACH * cam_h)
        start = next((k for k, i in enumerate(order) if flagged[i]), None)
        if start is None or start > reach:
            continue
        last, gap = start, 0
        for k in range(start + 1, len(order)):
            if flagged[order[k]]:
                last, gap = k, 0
            else:
                gap += 1
                if gap > LABEL_GAP:
                    break
        if last + 1 > math.ceil(LABEL_MAX * cam_h) + start:
            continue
        rows = [order[k] for k in range(start, last + 1)]
        # a strip right at the cam edge that is only the border's own shadow is no label
        if len([r for r in rows if flagged[r]]) < 3:
            continue
        cols = np.flatnonzero(present[min(rows) : max(rows) + 1].any(axis=0))
        mx0 = min(x0, wx0 + int(cols[0])) if len(cols) else x0
        mx1 = max(x1, wx0 + int(cols[-1]) + 1) if len(cols) else x1
        far = lo + (max(rows) + 1 + LABEL_PAD if below else min(rows) - LABEL_PAD)
        if below:
            out.append((mx0, y1, mx1, min(far, height)))
        else:
            out.append((mx0, max(far, 0), mx1, y0))
    return out


def vod_bars(gray: np.ndarray, protect: PxBox | None = None) -> tuple[int, int, int, int] | None:
    """Static black bars over the (n, H, W) samples: the crop (w, h, x, y) to the union of
    every sample's content (a line is content when > BAR_CONTENT of it is brighter than
    BAR_LIMIT), grown to contain `protect` (the facecam), even-aligned; None when nothing is
    content or the bars are under BAR_MIN of both dimensions."""
    _, height, width = gray.shape
    lit = gray > BAR_LIMIT
    rows = (lit.mean(axis=2) > BAR_CONTENT).any(axis=0)
    cols = (lit.mean(axis=1) > BAR_CONTENT).any(axis=0)
    if not rows.any() or not cols.any():
        return None
    ys, xs = np.flatnonzero(rows), np.flatnonzero(cols)
    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1
    if protect is not None:
        x0, y0 = min(x0, protect[0]), min(y0, protect[1])
        x1, y1 = max(x1, protect[2]), max(y1, protect[3])
    x0, y0 = x0 // 2 * 2, y0 // 2 * 2
    x1, y1 = min(math.ceil(x1 / 2) * 2, width), min(math.ceil(y1 / 2) * 2, height)
    w, h = x1 - x0, y1 - y0
    if width - w < BAR_MIN * width and height - h < BAR_MIN * height:
        return None
    return w, h, x0, y0


def ring_reference(med: np.ndarray, outer: PxBox) -> list[tuple[int, int, int]]:
    """[x, y, value] of the median frame `med` at RING_POINTS points per side on the outer
    RING_DEPTH lines of the border (`outer`)."""
    x0, y0, x1, y1 = outer
    pts = []
    for d in range(RING_DEPTH):
        for k in range(RING_POINTS):
            u = (k + 0.5) / RING_POINTS
            x, y = int(x0 + d + u * (x1 - x0 - 2 * d - 1)), int(y0 + d + u * (y1 - y0 - 2 * d - 1))
            pts += [(x, y0 + d), (x, y1 - 1 - d), (x0 + d, y), (x1 - 1 - d, y)]
    return [(x, y, int(med[y, x])) for x, y in pts]


def ring_visible(
    gray: np.ndarray, ring: Sequence[Sequence[int]], origin: tuple[int, int] = (0, 0)
) -> bool:
    """Whether a gray frame (its top-left at source px `origin`) shows the cam border: >=
    RING_MATCH of the `ring` points within RING_TOL of the reference, their values following
    its pattern (RING_CORR; a flat reference: only when it isn't dark, RING_DARK)."""
    if not ring:
        return False
    ox, oy = origin
    h, w = gray.shape
    ref, got = [], []
    for x, y, v in ring:
        if 0 <= y - oy < h and 0 <= x - ox < w:
            ref.append(float(v))
            got.append(float(gray[y - oy, x - ox]))
    if not ref:
        return False
    r, g = np.asarray(ref), np.asarray(got)
    if int((np.abs(g - r) <= RING_TOL).sum()) < RING_MATCH * len(ring):
        return False
    if r.std() < RING_FLAT:
        return float(r.mean()) > RING_DARK
    if g.std() < RING_FLAT / 2:
        return False  # a flat frame (black, a white flash) has no border pattern
    return float(np.corrcoef(r, g)[0, 1]) >= RING_CORR


def _times(duration: float, n: int) -> list[float]:
    skip = min(INTRO_SKIP_S, INTRO_SKIP_SHARE * duration)
    return [skip + (duration - skip) * (i + 0.5) / n for i in range(n)]


def build_profile(src: Path, detector: _Detector | None) -> LayoutProfile:
    """Sample PROFILE_SAMPLES frames after the intro, at most ANALYSIS_W wide; the bars from
    all of them; with a `detector`, the cam (cluster, snap, labels, border ring). A source
    wider than ANALYSIS_W gets its cam re-snapped on full-size crops around it (`_refine`),
    so every box is in source px."""
    import cv2

    width, height = probe_size(src)
    aw = min(width, ANALYSIS_W)
    ah = max(2, round(aw * height / width / 2) * 2) if aw < width else height
    times = _times(probe_duration(src), PROFILE_SAMPLES)
    grays, faces, kept = [], [], []
    for t in times:
        try:
            (bgr,) = sample_frames(src, [t], width=aw)
        except FFmpegError:
            continue  # a sample past a broken tail: the others decide
        grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        kept.append(t)
        if detector is not None:
            small = bgr
            if aw > DETECT_W:
                small = cv2.resize(bgr, (DETECT_W, max(2, round(DETECT_W * ah / aw))))
            faces.append(detector.detect(small))
    if not grays:
        return LayoutProfile(size=(width, height))
    gray = np.stack(grays)
    del grays
    sx, sy = width / gray.shape[2], height / gray.shape[1]
    cam = _find_cam(gray, faces) if detector is not None else None
    if cam is not None and (sx, sy) != (1.0, 1.0):
        cam = _refine(src, kept, cam, (width, height), (sx, sy))
    protect = None
    if cam is not None and cam.outer is not None:
        x0, y0, x1, y1 = cam.outer
        protect = (int(x0 / sx), int(y0 / sy), math.ceil(x1 / sx), math.ceil(y1 / sy))
    bars = _scale_bars(vod_bars(gray, protect=protect), (sx, sy), (width, height))
    if cam is None:
        return LayoutProfile(size=(width, height), bars=bars)
    return cam.model_copy(update={"bars": bars})


def _scale_bars(
    bars: tuple[int, int, int, int] | None, scale: tuple[float, float], size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Analysis-size bars (w, h, x, y) in source px, grown outward (never into content),
    even-aligned."""
    if bars is None or scale == (1.0, 1.0):
        return bars
    w, h, x, y = bars
    sx, sy = scale
    x0, y0 = int(x * sx) // 2 * 2, int(y * sy) // 2 * 2
    x1 = min(math.ceil((x + w) * sx / 2) * 2, size[0])
    y1 = min(math.ceil((y + h) * sy / 2) * 2, size[1])
    return x1 - x0, y1 - y0, x0, y0


def _find_cam(
    gray: np.ndarray,
    faces: list[list[Face]],
    origin: tuple[int, int] = (0, 0),
    frame: tuple[int, int] | None = None,
    found: tuple[Face, float] | None = None,
) -> LayoutProfile | None:
    """The cam profile from (n, H, W) gray samples (a crop at `origin` of a `frame`-sized
    picture, default the whole frame) and their face detections (or the cluster `found`);
    boxes in frame px. None when no fixed small face, no snapped overlay, or the overlay is a
    webcam shot rather than a corner cam (> CAM_AREA_MAX of the frame, or the face within
    CAM_OFF_CENTRE face widths of the frame centre)."""
    _, h_px, w_px = gray.shape
    width, height = frame or (w_px, h_px)
    found = found or cam_cluster(faces)
    if found is None:
        log.info("layout profile: no fixed facecam face in %d samples", len(faces))
        return None
    face, share = found
    box = (face.x * width, face.y * height, face.w * width, face.h * height)
    off = math.hypot((face.cx - 0.5) * width, (face.cy - 0.5) * height)
    if off < CAM_OFF_CENTRE * box[2]:
        log.info("layout profile: the fixed face at %s px is central: a webcam shot", box)
        return None
    snapped = snap_cam(gray, box, origin, (width, height))
    if snapped is None:
        log.info("layout profile: cam face at %s px has no static overlay border", box)
        return None
    outer, inner = snapped
    head_top = face.y - HAIR_SHARE * face.h
    head = (
        max(int((face.x - HEAD_WIDEN * face.w) * width), inner[0]),
        max(int(head_top * height), inner[1]),
        min(math.ceil((face.x + face.w * (1 + HEAD_WIDEN)) * width), inner[2]),
        min(math.ceil((face.y + face.h) * height), inner[3]),
    )
    fpx = (
        int(box[0]), int(box[1]), math.ceil(box[0] + box[2]), math.ceil(box[1] + box[3])
    )  # fmt: skip
    ox, oy = origin

    def local(b: PxBox) -> PxBox:
        return b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy

    def shift(b: PxBox) -> PxBox:
        return b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy

    med = np.median(gray, axis=0)
    k = RING_INSET
    inset = (inner[0] + k, inner[1] + k, inner[2] - k, inner[3] - k)
    return LayoutProfile(
        size=(width, height),
        cam=inner,
        outer=outer,
        head=head,
        face=fpx,
        masks=[shift(m) for m in label_masks(gray, local(outer))],
        ring=[(x + ox, y + oy, v) for x, y, v in ring_reference(med, local(outer))],
        ring_in=[(x + ox, y + oy, v) for x, y, v in ring_reference(med, local(inset))],
        share=share,
    )


def _refine(
    src: Path,
    times: list[float],
    coarse: LayoutProfile,
    size: tuple[int, int],
    scale: tuple[float, float],
) -> LayoutProfile | None:
    """`coarse` (found on downscaled samples) re-found on REFINE_SAMPLES full-size crops of the
    source around its cam (REFINE_PAD, REFINE_LABEL): source-px boxes and a border ring that
    matches full-size frames. None when the crops can't be read or the cam doesn't snap."""
    import cv2

    assert coarse.outer is not None and coarse.face is not None
    sx, sy = scale
    x0, y0, x1, y1 = coarse.outer
    w, h = (x1 - x0) * sx, (y1 - y0) * sy
    cx0 = max(int(x0 * sx - REFINE_PAD * w), 0) // 2 * 2
    cy0 = max(int(y0 * sy - REFINE_LABEL * h), 0) // 2 * 2
    cx1 = min(math.ceil(x1 * sx + REFINE_PAD * w), size[0])
    cy1 = min(math.ceil(y1 * sy + REFINE_LABEL * h), size[1])
    cw, ch = (cx1 - cx0) // 2 * 2, (cy1 - cy0) // 2 * 2
    step = max(len(times) // REFINE_SAMPLES, 1)
    grays = []
    for t in times[::step][:REFINE_SAMPLES]:
        try:
            (bgr,) = sample_frames(src, [t], width=cw, crop=(cw, ch, cx0, cy0))
        except FFmpegError:
            continue
        grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if len(grays) < 2:
        return None
    fx0, fy0, fx1, fy1 = coarse.face
    face = Face(
        fx0 * sx / size[0], fy0 * sy / size[1], (fx1 - fx0) * sx / size[0],
        (fy1 - fy0) * sy / size[1], 1.0, None, None,
    )  # fmt: skip
    return _find_cam(np.stack(grays), [], (cx0, cy0), size, found=(face, coarse.share or 0.0))


def _key(src: Path, detector: _Detector | None) -> dict:
    return {
        "version": PROFILE_VERSION,
        "source": src.name,
        "size": list(probe_size(src)),
        "duration": round(probe_duration(src), 3),
        "faces": detector is not None,
    }


def layout_profile(job_dir: Path, src: Path, detector: _Detector | None) -> LayoutProfile:
    """The job's layout profile, cached in `job_dir`/layout_profile.json keyed by
    PROFILE_VERSION, the source and whether faces were looked for (a cache made without a
    detector is redone when one is available)."""
    path = job_dir / PROFILE_FILE
    key = _key(src, detector)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["key"] == key:
                return LayoutProfile.model_validate(data["profile"])
        except (ValueError, KeyError, TypeError):
            log.warning("layout profile cache %s unreadable; recomputing", path)
    prof = build_profile(src, detector)
    atomic_write_text(path, json.dumps({"key": key, "profile": prof.model_dump(mode="json")}))
    if prof.cam is not None:
        log.info(
            "layout profile: facecam %s (outer %s), masks %s, bars %s, in %.0f%% of samples",
            prof.cam, prof.outer, prof.masks, prof.bars, 100 * (prof.share or 0),
        )  # fmt: skip
    else:
        log.info("layout profile: no facecam; bars %s", prof.bars)
    return prof
