"""Per-scene 9:16 layout choice: STACKED facecam, FACE_TRACK, SPLIT, or FIT_BLUR."""

import statistics
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.vision.faces import Crop, Face, sample_frames

LayoutKind = Literal["STACKED", "FACE_TRACK", "SPLIT", "FIT_BLUR"]
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


class SceneLayout(BaseModel):
    start: float
    end: float
    kind: LayoutKind
    facecam: Box | None = None
    track_x: list[tuple[float, float]] = []
    split_x: tuple[float, float] | None = None
    face_box: tuple[float, float] | None = None  # median face (y, h), normalized; captions


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
        )

    largest = [(t, max(faces, key=lambda f: f.w * f.h)) for t, faces in kept if faces]
    if n and len(largest) >= TRACK_MIN_SHARE * n:
        return SceneLayout(
            start=start,
            end=end,
            kind="FACE_TRACK",
            track_x=[(t, f.cx) for t, f in largest],
            face_box=_median_box([(f.y, f.h) for _, f in largest]),
        )

    return SceneLayout(start=start, end=end, kind="FIT_BLUR")


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
    sample_every: float = 1.0,
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
        except FFmpegError:
            samples = []  # unreadable scene: fall back to FIT_BLUR rather than abort the plan
        layouts.append(classify_scene(samples, start, end, src_h=src_h))
    return layouts
