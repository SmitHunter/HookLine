"""Per-scene 9:16 layout choice: STACKED facecam, FACE_TRACK, SPLIT, or FIT_BLUR."""

import logging
import math
import statistics
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.vision.faces import Crop, Face, sample_frames

LayoutKind = Literal["STACKED", "FACE_TRACK", "SPLIT", "FIT_BLUR"]
log = logging.getLogger(__name__)
Box = tuple[float, float, float, float]  # normalized x, y, w, h

MIN_SCORE = 0.6
CAM_MAX_H = 0.15
CAM_MAX_AREA = 0.20  # grown facecam box (normalized w*h); larger = a webcam shot, not a cam
CAM_MIN_SHARE = 0.6
CAM_MAX_STD = 0.03
CAM_GROW_W = 3.0
CAM_GROW_H = 2.6
SPLIT_MIN_H = 0.15
SPLIT_MIN_SHARE = 0.5
TRACK_MIN_SHARE = 0.5
CAM_VETO_SHARE = 0.3  # share of cam samples with a comparable face outside the cam box
CAM_VETO_MIN_H = 0.5  # ... "comparable" = at least half the cam face's height
CAM_MIN_PX = 160  # grown cam box must be this many source pixels tall
GROUP_MIN_SHARE = 0.5  # share of samples showing a small-face group -> wide shot (FIT_BLUR)
GROUP_MAX_RATIO = 2.0  # faces within 2x height of the largest are "comparable"
TRACK_HOLD_S = 3.0  # a face lost for up to this long: the crop holds its last good x ...
TRACK_EASE_S = 1.0  # ... beyond it, eases to the scene's median face x over this long
HAIR_SHARE = 0.5  # head top ~ face box top - half a face height (hair)
HEAD_QUANTILE = 0.1  # the head top budgeted for: this low quantile of the samples (highest)
ZOOM_TOP = 1.2  # the highest zoom step level (a punch); see reframe.ZOOM_PUNCH
SAMPLE_EVERY = 0.25  # face samples every 0.25 s (4 fps): the tracking edge guard needs them
# Gaze lead room: a FACE_TRACK crop shifts up to LEAD of its width toward where the subject
# looks (median landmark yaw over the scene): none under YAW_DEAD, the full lead from YAW_FULL.
LEAD = 0.08
YAW_DEAD = 0.1
YAW_FULL = 0.3


class SceneLayout(BaseModel):
    start: float
    end: float
    kind: LayoutKind
    facecam: Box | None = None
    track_x: list[tuple[float, float]] = []
    track_w: list[float] = []  # face width per track_x keyframe, normalized (edge guard)
    lead: float = 0.0  # gaze lead room: signed share of crop width the crop shifts (+ right)
    # Edge-guard framing changes (reframe.guard_zoom): (output s, signed share of crop width)
    # sideways crop offsets, instant at cuts; and the rendered face centre's median offset from
    # the crop centre (share of crop width, + right), for caption placement and the audit.
    shift_steps: list[tuple[float, float]] = []
    face_dx: float | None = None
    split_x: tuple[float, float] | None = None
    face_box: tuple[float, float] | None = None  # median face (y, h), normalized; captions
    face_w: float | None = None  # median face width (FACE_TRACK), normalized to source width
    head_top: float | None = None  # highest head top (HEAD_QUANTILE), normalized source y
    # Zoom plan (reframe.plan_zoom): step level s shows as zoom_level(s); zooms keep the panel
    # row zoom_anchor (normalized panel y; None = face centre / panel centre) fixed.
    base_zoom: float = 1.0
    zoom_cap: float | None = None
    zoom_anchor: float | None = None
    step_shift: float = 0.0  # flat-zoom FACE_TRACK: signed share of crop width steps shift by

    def zoom_level(self, s: float) -> float:
        """Rendered zoom for schedule level `s` (1.0 / ZOOM_IN / ZOOM_TOP): base_zoom times
        the step, the step part compressed (never clipped flat) so ZOOM_TOP lands on zoom_cap
        when the full range would pass it; every distinct level stays distinct."""
        k = 1.0
        if self.zoom_cap is not None:
            k = min(1.0, max(self.zoom_cap / self.base_zoom - 1.0, 0.0) / (ZOOM_TOP - 1.0))
        return self.base_zoom * (1.0 + (s - 1.0) * k)

    def zoom_y(self, y: float, s: float, panel_h: float) -> float:
        """Where panel row `y` (px, unzoomed) lands at schedule level `s`: zooms scale about
        the anchor row (zoom_anchor, else the face centre, else the panel centre)."""
        a = self.anchor_px(panel_h)
        return a + self.zoom_level(s) * (y - a)

    def anchor_px(self, panel_h: float) -> float:
        if self.zoom_anchor is not None:
            return self.zoom_anchor * panel_h
        if self.kind == "FACE_TRACK" and self.face_box is not None:
            return (self.face_box[0] + self.face_box[1] / 2) * panel_h
        return panel_h / 2


class _Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Face]: ...


def _in_corner(f: Face) -> bool:
    return (f.cx < 0.35 or f.cx > 0.65) and (f.cy < 0.4 or f.cy > 0.6)


def _grow_inside(cx: float, cy: float, w: float, h: float) -> Box:
    """Box of size w x h centred on (cx, cy), shifted (shrunk only if larger) to fit [0, 1]."""
    w, h = min(w, 1.0), min(h, 1.0)
    x = min(max(cx - w / 2, 0.0), 1.0 - w)
    y = min(max(cy - h / 2, 0.0), 1.0 - h)
    return (x, y, w, h)


def _inside(box: Box, f: Face) -> bool:
    x, y, w, h = box
    return x <= f.cx <= x + w and y <= f.cy <= y + h


def _facecam(samples: list[list[Face]], src_h: int | None = None) -> Box | None:
    """Fixed small corner face -> its grown cam box. Faces in `samples` already passed
    MIN_SCORE. Vetoed when, in >= 30% of the cam samples, another comparable face sits outside
    the box (a wide shot with someone near a corner, not a facecam overlay), and when the box
    would be under 160 source px tall."""
    cams: list[tuple[Face, list[Face]]] = []
    for faces in samples:
        corner = [f for f in faces if f.h < CAM_MAX_H and _in_corner(f)]
        if corner:
            cams.append((max(corner, key=lambda f: f.w * f.h), faces))
    if not samples or len(cams) < CAM_MIN_SHARE * len(samples):
        return None
    cxs, cys = [c.cx for c, _ in cams], [c.cy for c, _ in cams]
    if statistics.pstdev(cxs) >= CAM_MAX_STD or statistics.pstdev(cys) >= CAM_MAX_STD:
        return None
    box = _grow_inside(
        statistics.median(cxs),
        statistics.median(cys),
        statistics.median(c.w for c, _ in cams) * CAM_GROW_W,
        statistics.median(c.h for c, _ in cams) * CAM_GROW_H,
    )
    if box[2] * box[3] >= CAM_MAX_AREA:
        return None
    if src_h is not None and box[3] * src_h < CAM_MIN_PX:
        return None
    vetoed = sum(
        any(f is not c and f.h >= CAM_VETO_MIN_H * c.h and not _inside(box, f) for f in faces)
        for c, faces in cams
    )
    return None if vetoed >= CAM_VETO_SHARE * len(cams) else box


def _small_group(faces: list[Face]) -> bool:
    """>= 2 comparable faces (within 2x height), all under SPLIT_MIN_H: a wide group shot."""
    if len(faces) < 2:
        return False
    tallest = max(f.h for f in faces)
    comparable = [f for f in faces if f.h * GROUP_MAX_RATIO >= tallest]
    return len(comparable) >= 2 and tallest < SPLIT_MIN_H


def classify_scene(
    samples: list[tuple[float, list[Face]]],
    start: float,
    end: float,
    facecam_override: Box | None = None,
    src_h: int | None = None,
) -> SceneLayout:
    """`src_h` (source height in px, when known) rejects facecam boxes under 160 px."""
    if facecam_override is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=facecam_override)

    kept = [(t, [f for f in faces if f.score >= MIN_SCORE]) for t, faces in samples]
    n = len(kept)

    box = _facecam([faces for _, faces in kept], src_h)
    if box is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=box)

    if n and sum(_small_group(faces) for _, faces in kept) >= GROUP_MIN_SHARE * n:
        return SceneLayout(start=start, end=end, kind="FIT_BLUR")

    pairs = []
    pair_boxes = []
    for _, faces in kept:
        big = sorted((f for f in faces if f.h >= SPLIT_MIN_H), key=lambda f: f.w * f.h)
        if len(big) >= 2:
            pairs.append(sorted((big[-1].cx, big[-2].cx)))
            pair_boxes.append((min(big[-1].y, big[-2].y), max(big[-1].h, big[-2].h)))
    if n and len(pairs) >= SPLIT_MIN_SHARE * n:
        left = statistics.median(p[0] for p in pairs)
        right = statistics.median(p[1] for p in pairs)
        return SceneLayout(
            start=start,
            end=end,
            kind="SPLIT",
            split_x=(min(left, right), max(left, right)),
            face_box=_median_box(pair_boxes),
            head_top=_quantile([y - HAIR_SHARE * h for y, h in pair_boxes], HEAD_QUANTILE),
        )

    largest = [(t, max(faces, key=lambda f: f.w * f.h)) for t, faces in kept if faces]
    if n and len(largest) >= TRACK_MIN_SHARE * n:
        track = _track_keyframes(kept)
        return SceneLayout(
            start=start,
            end=end,
            kind="FACE_TRACK",
            track_x=[(t, x) for t, x, _ in track],
            track_w=[w for _, _, w in track],
            lead=gaze_lead([f.yaw for _, f in largest]),
            face_box=_median_box([(f.y, f.h) for _, f in largest]),
            face_w=statistics.median(f.w for _, f in largest),
            head_top=_head_top([f for _, f in largest]),
        )

    return SceneLayout(start=start, end=end, kind="FIT_BLUR")


def _quantile(values: list[float], q: float) -> float:
    """Nearest-rank `q` quantile (q=0.1: a low value that ignores a stray outlier or two)."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def _head_top(faces: list[Face]) -> float:
    return _quantile([f.y - HAIR_SHARE * f.h for f in faces], HEAD_QUANTILE)


def gaze_lead(yaws: list[float | None]) -> float:
    """Signed lead room (share of crop width; + = the crop shifts right) from the scene's
    median face yaw: 0 below YAW_DEAD, ramping to LEAD at YAW_FULL. The median ignores a
    stray estimate; one value per scene keeps the crop steady."""
    known = [y for y in yaws if y is not None]
    if not known:
        return 0.0
    yaw = statistics.median(known)
    ramp = min(max((abs(yaw) - YAW_DEAD) / (YAW_FULL - YAW_DEAD), 0.0), 1.0)
    return math.copysign(LEAD * ramp, yaw) if ramp else 0.0


def _track_keyframes(kept: list[tuple[float, list[Face]]]) -> list[tuple[float, float, float]]:
    """(t, centre x, width) of the largest face per sample. A sample with no face (none passed
    MIN_SCORE) HOLDS the last good x (and width) rather than following whatever else is in
    frame; after TRACK_HOLD_S without a face it eases to the scene's median face x over
    TRACK_EASE_S. Samples before the first face add nothing (the first x holds back to the
    scene start), so the track only interpolates between valid samples and hold points."""
    xs = [max(faces, key=lambda f: f.w * f.h).cx for _, faces in kept if faces]
    median = statistics.median(xs) if xs else 0.5
    out: list[tuple[float, float, float]] = []
    last: tuple[float, float, float] | None = None
    for t, faces in sorted(kept, key=lambda s: s[0]):
        if faces:
            face = max(faces, key=lambda f: f.w * f.h)
            last = (t, face.cx, face.w)
            out.append(last)
        elif last is not None:
            ease = min(max((t - last[0] - TRACK_HOLD_S) / TRACK_EASE_S, 0.0), 1.0)
            out.append((t, last[1] + (median - last[1]) * ease, last[2]))
    return out


def _median_box(boxes: list[tuple[float, float]]) -> tuple[float, float]:
    return (statistics.median(b[0] for b in boxes), statistics.median(b[1] for b in boxes))


def _scenes(video: Path) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, detect

    found = detect(str(video), ContentDetector())
    if not found:
        return [(0.0, probe_duration(video))]
    return [(a.seconds, b.seconds) for a, b in found]


def _sample_times(start: float, end: float, every: float) -> list[float]:
    """Midpoints of `every`-second slots, so no sample lands on a scene edge or past the end."""
    n = int((end - start) // every)
    if n < 1:
        return [(start + end) / 2]
    return [start + (k + 0.5) * every for k in range(n)]


def plan_layouts(
    video: Path,
    detector: _Detector | None,
    facecam_override: Box | None = None,
    sample_every: float = SAMPLE_EVERY,
    crop: Crop | None = None,
) -> list[SceneLayout]:
    """One layout per scene. With no detector (and no override) the whole video is a single
    FIT_BLUR scene: without faces every scene would be FIT_BLUR anyway."""
    if detector is None and facecam_override is None:
        return [SceneLayout(start=0.0, end=probe_duration(video), kind="FIT_BLUR")]
    src_h = None
    if facecam_override is None:
        src_h = crop[1] if crop else probe_size(video)[1]
    layouts = []
    for start, end in _scenes(video):
        if facecam_override is not None:
            layouts.append(classify_scene([], start, end, facecam_override))
            continue
        times = _sample_times(start, end, sample_every)
        assert detector is not None
        try:
            frames = sample_frames(video, times, crop=crop)
            samples = [(t, detector.detect(f)) for t, f in zip(times, frames, strict=True)]
        except FFmpegError as e:
            # unreadable scene: fall back to FIT_BLUR rather than abort the plan, never silently
            log.warning(
                "face sampling failed for %s scene %.2f-%.2f s (%s); the scene uses FIT_BLUR",
                video.name, start, end, (str(e).splitlines() or ["error"])[0],
            )  # fmt: skip
            samples = []
        layouts.append(classify_scene(samples, start, end, src_h=src_h))
    return layouts
