"""Per-scene 9:16 layout choice: STACKED facecam, FACE_TRACK, SPLIT, or FIT_BLUR."""

import statistics
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from hookline.media.ffmpeg import FFmpegError, probe_duration
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


class SceneLayout(BaseModel):
    start: float
    end: float
    kind: LayoutKind
    facecam: Box | None = None
    track_x: list[tuple[float, float]] = []
    split_x: tuple[float, float] | None = None


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


def _facecam(samples: list[list[Face]]) -> Box | None:
    cams = []
    for faces in samples:
        corner = [f for f in faces if f.h < CAM_MAX_H and _in_corner(f)]
        if corner:
            cams.append(max(corner, key=lambda f: f.w * f.h))
    if not samples or len(cams) < CAM_MIN_SHARE * len(samples):
        return None
    cxs, cys = [f.cx for f in cams], [f.cy for f in cams]
    if statistics.pstdev(cxs) >= CAM_MAX_STD or statistics.pstdev(cys) >= CAM_MAX_STD:
        return None
    box = _grow_inside(
        statistics.median(cxs),
        statistics.median(cys),
        statistics.median(f.w for f in cams) * CAM_GROW_W,
        statistics.median(f.h for f in cams) * CAM_GROW_H,
    )
    return box if box[2] * box[3] < CAM_MAX_AREA else None


def classify_scene(
    samples: list[tuple[float, list[Face]]],
    start: float,
    end: float,
    facecam_override: Box | None = None,
) -> SceneLayout:
    if facecam_override is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=facecam_override)

    kept = [(t, [f for f in faces if f.score >= MIN_SCORE]) for t, faces in samples]
    n = len(kept)

    box = _facecam([faces for _, faces in kept])
    if box is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=box)

    pairs = []
    for _, faces in kept:
        big = sorted((f for f in faces if f.h >= SPLIT_MIN_H), key=lambda f: f.w * f.h)
        if len(big) >= 2:
            pairs.append(sorted((big[-1].cx, big[-2].cx)))
    if n and len(pairs) >= SPLIT_MIN_SHARE * n:
        left = statistics.median(p[0] for p in pairs)
        right = statistics.median(p[1] for p in pairs)
        return SceneLayout(
            start=start, end=end, kind="SPLIT", split_x=(min(left, right), max(left, right))
        )

    track = [(t, max(faces, key=lambda f: f.w * f.h).cx) for t, faces in kept if faces]
    if n and len(track) >= TRACK_MIN_SHARE * n:
        return SceneLayout(start=start, end=end, kind="FACE_TRACK", track_x=track)

    return SceneLayout(start=start, end=end, kind="FIT_BLUR")


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
        layouts.append(classify_scene(samples, start, end))
    return layouts
