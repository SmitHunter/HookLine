"""VOD-level stream layout profile, computed once per job and cached.

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
from typing import NamedTuple, Protocol

import numpy as np
from pydantic import BaseModel

from hookline.fsutil import atomic_write_text
from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.vision.faces import Face, sample_frames

log = logging.getLogger(__name__)

PROFILE_VERSION = 4  # bump when the profile changes shape or meaning: cached ones re-run
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
# Snapping: the cam rectangle is searched within SNAP_REACHES[0] face heights of the face,
# growing through the SNAP_REACHES until one finds it (a 16:9 cam reaches ~3 face heights sideways).
# A pixel is a static edge when its gradient (> EDGE_T) matches the median frame's in
# >= STATIC_SHARE of the samples -- or a still/live edge: a still pixel (mean |frame - median|
# under STILL_T: a smooth wall, a border) next to a live one (over LIVE_T: gameplay) within 2
# px, a borderless cam's edge. A border line has such edges along >= LINE_SHARE of a span
# centred on the face (SPAN of the face width either side) and the rectangle's 4 sides
# >= PERIM_SHARE. The frame's own edge is a side too (a cam flush in a corner) when the
# rectangle's band along it is still in >= STILL_SIDE of it (not a strip of gameplay).
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
# A static HUD (a minimap) next to a borderless cam: a still band whose median frame is
# textured (>= HUD_TEXTURE of a line's pixels with a gradient over EDGE_T) and at most
# HUD_LIVE of it live, at least HUD_MIN px deep. Outside a borderless cam it counts as a side
# with gameplay (the cam's edge is still/live elsewhere); inside a snapped rectangle, on the
# side away from the face, it is cut off (the face then more central) -- either way it
# becomes a mask, never cam. Looked for up to HUD_REACH of the cam's size from its edge.
HUD_TEXTURE = 0.3
HUD_LIVE = 0.05
HUD_MIN = 6
HUD_REACH = 1.0
# The cam's picture moves: in >= HEAD_LIVE_SHARE of the samples showing it, >= HEAD_MOVED of
# the head region differs from its median by over HEAD_DIFF gray (a person, or a PNGtuber's
# periodic mouth / blink); a static in-game portrait in a bordered HUD box never does.
HEAD_DIFF = 12
HEAD_MOVED = 0.01
HEAD_LIVE_SHARE = 0.1
# Per-pixel masks are computed in row chunks of about MASK_CHUNK sample pixels (int16 /
# float32), once per cam over the widest snap reach: memory stays bounded (150 x 720p
# samples would be ~0.5 GB as one float32 copy per reach).
MASK_CHUNK = 1_000_000
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
    found = _cluster(samples)
    return None if found is None else found[:2]


def _cluster(samples: Sequence[Sequence[Face]]) -> tuple[Face, float, list[int]] | None:
    """`cam_cluster` plus the indices of the samples in the cluster (those showing the cam)."""
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
    if share < CLUSTER_SHARE - 1e-9 or len(faces) < 2:
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
    return med, round(share, 3), sorted(i for i, _ in best)


def _gradient(gray: np.ndarray) -> np.ndarray:
    """max(|dx|, |dy|) per pixel (a vertical step belongs to the row below, a horizontal one
    to the pixel left of it), as in `reframe._gradient`."""
    grad = np.zeros_like(gray)
    grad[..., :, :-1] = np.abs(np.diff(gray, axis=-1))
    grad[..., 1:, :] = np.maximum(grad[..., 1:, :], np.abs(np.diff(gray, axis=-2)))
    return grad


Idx = list[int] | None  # the samples that count (those showing the cam); None: all


def _take(gray: np.ndarray, idx: Idx, ys: slice, xs: slice) -> np.ndarray:
    """gray[idx, ys, xs]: only the window is copied (a view without `idx`)."""
    return gray[:, ys, xs] if idx is None else gray[idx, ys, xs]


class _Masks(NamedTuple):
    """Per-pixel masks of a window of the samples (its top-left at gray px x0, y0): static
    border-line edges, still/live edges (a still pixel within 2 px of a live one along a row
    or column: a borderless cam's edge), still / live pixels (mean |frame - median| under
    STILL_T / over LIVE_T) and the median frame."""

    x0: int
    y0: int
    static: np.ndarray
    edge: np.ndarray
    still: np.ndarray
    live: np.ndarray
    med: np.ndarray

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "_Masks":
        """The masks of the gray-px window (x0, y0, x1, y1) inside this one."""
        sl = (slice(y0 - self.y0, y1 - self.y0), slice(x0 - self.x0, x1 - self.x0))
        return _Masks(
            x0, y0, self.static[sl], self.edge[sl], self.still[sl], self.live[sl], self.med[sl]
        )


def _masks(gray: np.ndarray, box: PxBox, idx: Idx = None) -> _Masks:
    """`_Masks` of the gray-px window `box` over the samples `idx`, in row chunks of about
    MASK_CHUNK sample pixels (int16 / float32 temporaries: memory stays bounded). A pixel is
    a static edge when its gradient (> EDGE_T) matches the median frame's in >= STATIC_SHARE
    of the samples."""
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    n = len(gray) if idx is None else len(idx)
    static = np.zeros((h, w), bool)
    still = np.zeros((h, w), bool)
    live = np.zeros((h, w), bool)
    med = np.zeros((h, w), np.float32)
    step = max(1, MASK_CHUNK // max(n * w, 1))
    for r0 in range(y0, y1, step):
        r1 = min(r0 + step, y1)
        top = max(r0 - 1, 0)  # the row above: the chunk's first vertical step
        g = _take(gray, idx, slice(top, r1), slice(x0, x1)).astype(np.int16)
        grad = _gradient(g)[:, r0 - top :].astype(np.float32)
        gm = np.median(grad, axis=0)
        match = (grad > EDGE_T) & (gm > EDGE_T)
        match &= np.abs(grad - gm) < EDGE_TOL * np.maximum(grad, gm)
        del grad
        rows = slice(r0 - y0, r1 - y0)
        static[rows] = match.mean(axis=0) >= STATIC_SHARE
        del match
        g = g[:, r0 - top :]
        m = np.median(g, axis=0).astype(np.float32)
        mad = np.abs(g - m).mean(axis=0)
        still[rows], live[rows], med[rows] = mad < STILL_T, mad > LIVE_T, m
    edge = np.zeros_like(still)
    for d in (1, 2):
        edge[:, :-d] |= still[:, :-d] & live[:, d:]  # live to the right
        edge[:, d:] |= still[:, d:] & live[:, :-d]  # live to the left
        edge[:-d, :] |= still[:-d, :] & live[d:, :]  # live below
        edge[d:, :] |= still[d:, :] & live[:-d, :]  # live above
    return _Masks(x0, y0, static, edge, still, live, med)


def static_edges(gray: np.ndarray, box: PxBox) -> np.ndarray:
    """(h, w) bool static edges of the (n, H, W) samples inside the px box (x0, y0, x1, y1):
    a gradient over EDGE_T matching the median frame's in >= STATIC_SHARE of them, on a still
    pixel (an opaque overlay; busy gameplay matches by chance, never still)."""
    m = _masks(gray, box)
    return m.static & m.still


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
    more than CAM_AREA_MAX of the frame. (`_snap` also returns the HUD masks found.)

    `gray` may be a crop of the frame: `origin` is its top-left and `frame` the frame's size
    (px; default: the crop is the frame); `face` and the boxes are in frame px."""
    found = _snap(gray, face, origin, frame)
    return None if found is None else found[:2]


def _snap(
    gray: np.ndarray,
    face: tuple[float, float, float, float],
    origin: tuple[int, int] = (0, 0),
    frame: tuple[int, int] | None = None,
    idx: Idx = None,
) -> tuple[PxBox, PxBox, list[PxBox]] | None:
    """`snap_cam` over the samples `idx`, with the static HUD masks next to a borderless cam
    (frame px). The masks are computed once, over the widest reach."""
    _, height, width = gray.shape
    ox, oy = origin
    size = frame or (width + ox, height + oy)
    local = (face[0] - ox, face[1] - oy, face[2], face[3])
    cx, cy = local[0] + local[2] / 2, local[1] + local[3] / 2
    far = SNAP_REACHES[-1] * face[3]
    big = (
        max(int(cx - far), 0), max(int(cy - far), 0),
        min(math.ceil(cx + far), width), min(math.ceil(cy + far), height),
    )  # fmt: skip
    if big[2] - big[0] < 4 or big[3] - big[1] < 4:
        return None
    masks = _masks(gray, big, idx)
    # border lines first (a label strip's still box must not extend the cam), then still/live
    # edges (a borderless cam)
    for motion, mult in [(m, r) for m in (False, True) for r in SNAP_REACHES]:
        found = _snap_within(gray, local, mult * face[3], origin, size, motion, masks, idx)
        if found is not None:
            outer, inner, huds = found

            def shift(b: PxBox) -> PxBox:
                return b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy

            return shift(outer), shift(inner), [shift(m) for m in huds]
    return None


def _snap_within(
    gray: np.ndarray,
    face: tuple[float, float, float, float],
    reach: float,
    origin: tuple[int, int],
    frame: tuple[int, int],
    motion: bool,
    masks: _Masks,
    idx: Idx = None,
) -> tuple[PxBox, PxBox, list[PxBox]] | None:
    """`snap_cam` within `reach` px of the face centre (`motion`: still/live edges count as
    border lines too, and a static HUD next to the cam is kept out of it); boxes in `gray`
    px, with the HUD masks."""
    _, height, width = gray.shape
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    rx0, rx1 = max(int(cx - reach), 0), min(math.ceil(cx + reach), width)
    ry0, ry1 = max(int(cy - reach), 0), min(math.ceil(cy + reach), height)
    rx0, ry0 = max(rx0, masks.x0), max(ry0, masks.y0)
    rx1 = min(rx1, masks.x0 + masks.static.shape[1])
    ry1 = min(ry1, masks.y0 + masks.static.shape[0])
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return None
    mk = masks.crop(rx0, ry0, rx1, ry1)
    s = mk.static | mk.edge if motion else mk.static
    still = mk.still
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

    best: tuple[int, PxBox, list[int]] | None = None
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
                        if not sides_ok(t, b, le, r):
                            continue
                        huds: list[int] = []
                        if motion:
                            # a borderless edge needs gameplay (or a static HUD) all round:
                            # not the streamer's outline on a still wall
                            ok, huds = _live_around(gray, box, around, idx)
                            if not ok:
                                continue
                        best = (area, box, huds)
    if best is None:
        return None
    outer, hud_sides = best[1], best[2]
    if _outside_motion(gray, outer, idx) < OUTSIDE_LIVE:
        return None
    huds_out: list[PxBox] = []
    if motion:
        outer, cut = _cut_hud(outer, face, masks)
        huds_out += cut
        huds_out += [m for k in hud_sides if (m := _hud_mask(gray, outer, k, idx)) is not None]
    x0, y0, x1, y1 = outer
    inner = _trim(masks.med, (x0 - masks.x0, y0 - masks.y0, x1 - masks.x0, y1 - masks.y0))
    inner = (inner[0] + masks.x0, inner[1] + masks.y0, inner[2] + masks.x0, inner[3] + masks.y0)
    return outer, inner, huds_out


def _bands(box: PxBox, size: tuple[int, int], a: int, b: int) -> list[tuple[slice, slice]]:
    """(rows, cols) of the bands a..b px outside each side of `box` (left, top, right,
    bottom), clipped to a `size` (w, h) frame."""
    width, height = size
    x0, y0, x1, y1 = box
    return [
        (slice(y0, y1), slice(max(x0 - b, 0), max(x0 - a, 0))),
        (slice(max(y0 - b, 0), max(y0 - a, 0)), slice(x0, x1)),
        (slice(y0, y1), slice(min(x1 + a, width), min(x1 + b, width))),
        (slice(min(y1 + a, height), min(y1 + b, height)), slice(x0, x1)),
    ]


def _textured(med: np.ndarray) -> np.ndarray:
    """Pixels of a median frame with a gradient over EDGE_T (texture: a HUD, text)."""
    return _gradient(med.astype(np.float32)) > EDGE_T


def _live_around(
    gray: np.ndarray,
    box: PxBox,
    around: tuple[tuple[bool, bool, bool, bool], PxBox],
    idx: Idx = None,
) -> tuple[bool, list[int]]:
    """Whether the picture just outside every side of `box` (a band RING_DEPTH..OUTSIDE_BAND
    px out; sides on the frame's edge skipped) is live (mean |frame - median| over LIVE_T)
    along >= LINE_SHARE of it -- a borderless cam's edge has gameplay all round it -- or,
    there, live or a static textured HUD; with the sides (0 left, 1 top, 2 right, 3 bottom)
    that needed the HUD."""
    _, height, width = gray.shape
    edges, region = around
    x0, y0, x1, y1 = box
    at_edge = (
        x0 == region[0] and edges[0], y0 == region[1] and edges[1],
        x1 == region[2] and edges[2], y1 == region[3] and edges[3],
    )  # fmt: skip
    huds = []
    bands = _bands(box, (width, height), RING_DEPTH, OUTSIDE_BAND)
    for k, (skip, (sy, sx)) in enumerate(zip(at_edge, bands, strict=True)):
        if skip:
            continue
        band = _take(gray, idx, sy, sx).astype(np.int16)
        if band.size == 0:
            return False, []
        med = np.median(band, axis=0).astype(np.float32)
        mad = np.abs(band - med).mean(axis=0)
        live = mad > LIVE_T
        if float(live.mean()) >= LINE_SHARE:
            continue
        hud = (mad < STILL_T) & _textured(med)
        if float((live | hud).mean()) < LINE_SHARE:
            return False, []
        huds.append(k)
    return True, huds


def _cut_hud(
    outer: PxBox, face: tuple[float, float, float, float], masks: _Masks
) -> tuple[PxBox, list[PxBox]]:
    """`outer` (a borderless cam's snap, gray px) with a static textured HUD band inside one of
    its sides cut off -- lines (across the box) at least HUD_TEXTURE textured, at most HUD_LIVE
    live, HUD_MIN+ deep from that side, clear of the face -- when that leaves the face more
    central and the box at least MIN_GROW face sizes; with the bands cut (masks)."""
    fx, fy, fw, fh = face
    fcx, fcy = fx + fw / 2, fy + fh / 2
    cut: list[PxBox] = []
    for k in range(4):  # left, top, right, bottom
        x0, y0, x1, y1 = outer
        mk = masks.crop(x0, y0, x1, y1)
        tex = _textured(mk.med)
        axis = 1 if k in (1, 3) else 0  # lines across the box: rows for top / bottom
        hud = (tex.mean(axis=axis) >= HUD_TEXTURE) & (mk.live.mean(axis=axis) <= HUD_LIVE)
        seq = hud[::-1] if k in (2, 3) else hud
        run = int(np.argmin(seq)) if not seq.all() else len(seq)
        if run < HUD_MIN:
            continue
        lo, hi = (y0, y1) if axis == 1 else (x0, x1)
        c, f0, f1, size = (fcy, fy, fy + fh, fh) if axis == 1 else (fcx, fx, fx + fw, fw)
        new = (lo + run, hi) if k in (0, 1) else (lo, hi - run)
        if new[0] > f0 or new[1] < f1 or new[1] - new[0] < MIN_GROW * size:
            continue
        if abs((c - new[0]) / (new[1] - new[0]) - 0.5) >= abs((c - lo) / (hi - lo) - 0.5):
            continue
        band = {0: (x0, y0, new[0], y1), 1: (x0, y0, x1, new[0]),
                2: (new[1], y0, x1, y1), 3: (x0, new[1], x1, y1)}[k]  # fmt: skip
        outer = {0: (new[0], y0, x1, y1), 1: (x0, new[0], x1, y1),
                 2: (x0, y0, new[1], y1), 3: (x0, y0, x1, new[1])}[k]  # fmt: skip
        cut.append(band)
    return outer, cut


def _hud_mask(gray: np.ndarray, outer: PxBox, side: int, idx: Idx = None) -> PxBox | None:
    """The static HUD outside `side` (0 left, 1 top, 2 right, 3 bottom) of a borderless cam
    (gray px): lines within HUD_REACH of its size whose pixels are still and textured in
    >= HUD_TEXTURE of them, starting within OUTSIDE_BAND + 2 px of the side, gaps up to
    LABEL_GAP bridged; the mask runs from the side to their far end (+ LABEL_PAD) across the
    cam's span. None when there is none."""
    _, height, width = gray.shape
    x0, y0, x1, y1 = outer
    vertical = side in (1, 3)
    depth = math.ceil(HUD_REACH * ((y1 - y0) if vertical else (x1 - x0)))
    if side == 0:
        sy, sx = slice(y0, y1), slice(max(x0 - depth, 0), x0)
    elif side == 1:
        sy, sx = slice(max(y0 - depth, 0), y0), slice(x0, x1)
    elif side == 2:
        sy, sx = slice(y0, y1), slice(x1, min(x1 + depth, width))
    else:
        sy, sx = slice(y1, min(y1 + depth, height)), slice(x0, x1)
    strip = _take(gray, idx, sy, sx).astype(np.int16)
    if strip.size == 0:
        return None
    med = np.median(strip, axis=0).astype(np.float32)
    hud = (np.abs(strip - med).mean(axis=0) < STILL_T) & _textured(med)
    del strip
    flagged = hud.mean(axis=1 if vertical else 0) >= HUD_TEXTURE
    order = list(range(len(flagged)))
    if side in (0, 1):
        order.reverse()  # from the cam outward
    start = next((k for k, i in enumerate(order) if flagged[i]), None)
    if start is None or start > OUTSIDE_BAND + 2:
        return None
    last, gap = start, 0
    for k in range(start + 1, len(order)):
        if flagged[order[k]]:
            last, gap = k, 0
        else:
            gap += 1
            if gap > LABEL_GAP:
                break
    if last - start + 1 < HUD_MIN:
        return None
    far = last + 1 + LABEL_PAD
    if side == 0:
        return max(x0 - far, 0), y0, x0, y1
    if side == 1:
        return x0, max(y0 - far, 0), x1, y0
    if side == 2:
        return x1, y0, min(x1 + far, width), y1
    return x0, y1, x1, min(y1 + far, height)


def _outside_motion(gray: np.ndarray, box: PxBox, idx: Idx = None) -> float:
    """Mean |frame - median frame| in a band OUTSIDE_BAND px wide around `box`."""
    _, height, width = gray.shape
    x0, y0, x1, y1 = box
    b = OUTSIDE_BAND
    gx0, gx1 = max(x0 - b, 0), min(x1 + b, width)
    rects = [
        (slice(max(y0 - b, 0), y0), slice(gx0, gx1)),  # above (corners included)
        (slice(y1, min(y1 + b, height)), slice(gx0, gx1)),  # below
        (slice(y0, y1), slice(gx0, x0)),  # left
        (slice(y0, y1), slice(x1, gx1)),  # right
    ]
    total, count = 0.0, 0
    for sy, sx in rects:
        vals = _take(gray, idx, sy, sx).astype(np.int16)
        if vals.size == 0:
            continue
        med = np.median(vals, axis=0).astype(np.float32)
        total += float(np.abs(vals - med).sum())
        count += vals.size
    return total / count if count else 0.0


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


def _head_moves(gray: np.ndarray, head: PxBox, idx: Idx = None) -> bool:
    """Whether the cam's picture moves in its head region (gray px): in >= HEAD_LIVE_SHARE of
    the samples >= HEAD_MOVED of it differs from its median by over HEAD_DIFF (a person; a
    PNGtuber's periodic mouth). A static in-game portrait never does."""
    x0, y0, x1, y1 = head
    g = _take(gray, idx, slice(y0, y1), slice(x0, x1)).astype(np.int16)
    if g.size == 0 or len(g) < 2:
        return False
    med = np.median(g, axis=0).astype(np.float32)
    moved = (np.abs(g - med) > HEAD_DIFF).mean(axis=(1, 2))
    return float((moved >= HEAD_MOVED).mean()) >= HEAD_LIVE_SHARE


def label_masks(gray: np.ndarray, outer: PxBox, idx: Idx = None) -> list[PxBox]:
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
        region = _take(gray, idx, slice(lo, hi), slice(wx0, wx1)).astype(np.int16)
        present = (_gradient(region) > EDGE_T).mean(axis=0) >= LABEL_PRESENT  # (rows, cols)
        del region
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
    rows, cols = np.zeros(height, bool), np.zeros(width, bool)
    for f in gray:  # one sample at a time: no (n, H, W) temporaries
        lit = f > BAR_LIMIT
        rows |= lit.mean(axis=1) > BAR_CONTENT
        cols |= lit.mean(axis=0) > BAR_CONTENT
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
    all of them; with a `detector`, the cam (cluster, snap, labels, border ring) from the
    samples showing it (its face in the cluster). A source wider than ANALYSIS_W gets its cam
    re-snapped on full-size crops around it (`_refine`), so every box is in source px; when
    that fails, the coarse cam scaled up stands (`_scaled`)."""
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
    cam = None
    if detector is not None:
        found = _cluster(faces)
        if found is None:
            log.info("layout profile: no fixed facecam face in %d samples", len(faces))
        else:
            face, share, members = found
            cam = _find_cam(gray, [], found=(face, share), idx=members)
            if cam is not None and (sx, sy) != (1.0, 1.0):
                fine = _refine(src, [kept[i] for i in members], cam, (width, height), (sx, sy))
                if fine is None:
                    log.info("layout profile: full-size re-snap failed; the coarse cam scaled")
                cam = fine or _scaled(cam, (sx, sy), (width, height))
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


def _scaled(
    coarse: LayoutProfile, scale: tuple[float, float], size: tuple[int, int]
) -> LayoutProfile:
    """`coarse` (analysis px) in source px: `outer`, `face` and masks grown outward to whole
    px, `cam` and `head` inward (never past the border), ring points scaled (their reference
    values kept)."""
    sx, sy = scale
    w, h = size

    def out(b: PxBox) -> PxBox:
        return (
            max(int(b[0] * sx), 0), max(int(b[1] * sy), 0),
            min(math.ceil(b[2] * sx), w), min(math.ceil(b[3] * sy), h),
        )  # fmt: skip

    def inward(b: PxBox) -> PxBox:
        return (
            math.ceil(b[0] * sx), math.ceil(b[1] * sy),
            min(int(b[2] * sx), w), min(int(b[3] * sy), h),
        )  # fmt: skip

    def pts(ring: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        return [(min(round(x * sx), w - 1), min(round(y * sy), h - 1), v) for x, y, v in ring]

    assert coarse.cam and coarse.outer and coarse.head and coarse.face
    return coarse.model_copy(
        update={
            "size": size,
            "cam": inward(coarse.cam),
            "outer": out(coarse.outer),
            "head": inward(coarse.head),
            "face": out(coarse.face),
            "masks": [out(b) for b in coarse.masks],
            "ring": pts(coarse.ring),
            "ring_in": pts(coarse.ring_in),
        }
    )


def _find_cam(
    gray: np.ndarray,
    faces: list[list[Face]],
    origin: tuple[int, int] = (0, 0),
    frame: tuple[int, int] | None = None,
    found: tuple[Face, float] | None = None,
    idx: Idx = None,
) -> LayoutProfile | None:
    """The cam profile from (n, H, W) gray samples (a crop at `origin` of a `frame`-sized
    picture, default the whole frame) and their face detections (or the cluster `found` and
    the samples `idx` showing it); boxes in frame px. Only the samples whose face is in the
    cluster count (a cam hidden for a stretch of the VOD: its gameplay is no overlay). None
    when no fixed small face, no snapped overlay, the overlay is a webcam shot rather than a
    corner cam (> CAM_AREA_MAX of the frame, or the face within CAM_OFF_CENTRE face widths of
    the frame centre), or its picture never moves (`_head_moves`: an in-game portrait)."""
    _, h_px, w_px = gray.shape
    width, height = frame or (w_px, h_px)
    if found is None:
        clustered = _cluster(faces)
        if clustered is None:
            log.info("layout profile: no fixed facecam face in %d samples", len(faces))
            return None
        face, share, idx = clustered
        found = (face, share)
    face, share = found
    if idx is not None and len(idx) >= len(gray):
        idx = None  # every sample: no copies
    box = (face.x * width, face.y * height, face.w * width, face.h * height)
    off = math.hypot((face.cx - 0.5) * width, (face.cy - 0.5) * height)
    if off < CAM_OFF_CENTRE * box[2]:
        log.info("layout profile: the fixed face at %s px is central: a webcam shot", box)
        return None
    snapped = _snap(gray, box, origin, (width, height), idx)
    if snapped is None:
        log.info("layout profile: cam face at %s px has no static overlay border", box)
        return None
    outer, inner, huds = snapped
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

    if not _head_moves(gray, local(head), idx):
        log.info("layout profile: the overlay at %s px never moves: no facecam", outer)
        return None
    lx0, ly0, lx1, ly1 = local(outer)
    med = np.median(_take(gray, idx, slice(ly0, ly1), slice(lx0, lx1)), axis=0)

    def ring(b: PxBox) -> list[tuple[int, int, int]]:
        x0, y0, x1, y1 = local(b)
        pts = ring_reference(med, (x0 - lx0, y0 - ly0, x1 - lx0, y1 - ly0))
        return [(x + lx0 + ox, y + ly0 + oy, v) for x, y, v in pts]

    k = RING_INSET
    inset = (inner[0] + k, inner[1] + k, inner[2] - k, inner[3] - k)
    return LayoutProfile(
        size=(width, height),
        cam=inner,
        outer=outer,
        head=head,
        face=fpx,
        masks=[*(shift(m) for m in label_masks(gray, local(outer), idx)), *huds],
        ring=ring(outer),
        ring_in=ring(inset),
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
    matches full-size frames, from `times` (samples showing the cam). None when the crops
    can't be read or the cam doesn't snap (the caller then scales `coarse` up: `_scaled`)."""
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
