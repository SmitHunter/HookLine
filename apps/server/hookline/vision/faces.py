import math
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hookline.media.ffmpeg import FFmpegError, probe_fps, probe_size

Crop = tuple[int, int, int, int]  # ffmpeg crop=w:h:x:y, in source pixels
Point = tuple[float, float]

MIN_EYE_PX = 2.0  # eyes closer than this (px) give no usable yaw
BATCH_MIN = 4  # sample this many times or more in one decode pass instead of one seek each
BATCH_MAX = 40  # frames per pass: a longer select sum overflows ffmpeg's expression parser


def face_yaw(right_eye: Point, left_eye: Point, nose: Point) -> float | None:
    """Horizontal head turn from YuNet's landmarks: how far the nose tip sits from the eye
    midpoint, in eye distances; > 0 when the nose points image-right (the subject looks
    frame-right), < 0 frame-left, about 0 facing the camera. None when the eyes overlap."""
    dist = abs(left_eye[0] - right_eye[0])
    if dist < MIN_EYE_PX:
        return None
    return (nose[0] - (right_eye[0] + left_eye[0]) / 2) / dist


class Face(NamedTuple):
    """A detected face, normalized 0..1 relative to the frame (x, y = top-left); `yaw` from
    its landmarks (`face_yaw`) and `eye_y` the eye line, None when unknown."""

    x: float
    y: float
    w: float
    h: float
    score: float
    yaw: float | None = None
    eye_y: float | None = None  # the eye line (mean of the eye landmarks), normalized

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class FaceDetector:
    """YuNet (cv2.FaceDetectorYN) wrapper returning normalized faces, best score first."""

    def __init__(self, model: Path, score_threshold: float = 0.6) -> None:
        import cv2

        self._size = (320, 320)
        self._det = cv2.FaceDetectorYN.create(
            str(model), "", self._size, score_threshold, 0.3, 5000
        )

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._det.detect(bgr)
        if rows is None:
            return []
        faces = []
        for r in rows:
            x0, y0 = max(0.0, float(r[0])), max(0.0, float(r[1]))
            x1, y1 = min(float(w), float(r[0] + r[2])), min(float(h), float(r[1] + r[3]))
            if x1 <= x0 or y1 <= y0:
                continue
            yaw = face_yaw((r[4], r[5]), (r[6], r[7]), (r[8], r[9]))
            eye = min(max((float(r[5]) + float(r[7])) / 2 / h, 0.0), 1.0)
            box = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
            faces.append(Face(*box, float(r[14]), yaw, eye))
        return sorted(faces, key=lambda f: f.score, reverse=True)


def sample_frames(
    video: Path, times: list[float], width: int = 640, crop: Crop | None = None
) -> list[np.ndarray]:
    """One BGR frame per time, cropped (if given) then scaled to `width` (even height): the
    first frame at or after each time. BATCH_MIN or more times are read in decode passes of
    up to BATCH_MAX frames (a frame-number select) instead of one seek per time. Raises
    FFmpegError when a time has no frame (past the end)."""
    if crop is not None:
        src_w, src_h = crop[0], crop[1]
    else:
        src_w, src_h = probe_size(video)
    height = max(2, round(width * src_h / src_w / 2) * 2)
    vf = f"scale={width}:{height}"
    if crop is not None:
        vf = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," + vf
    if len(times) >= BATCH_MIN:
        return _sample_batched(video, times, vf, width, height)
    return [_grab(video, ["-ss", f"{t:.3f}"], vf, width, height, 1, f"{t:.3f}s")[0] for t in times]


def _grab(
    video: Path, seek: list[str], vf: str, width: int, height: int, n: int, what: str
) -> list[np.ndarray]:
    """`n` raw BGR frames from one ffmpeg run (`seek` args before the input)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", str(video),
        "-frames:v", str(n), "-vf", vf, "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, check=False, capture_output=True)
    except FileNotFoundError as e:
        raise FFmpegError("ffmpeg not found on PATH") from e
    stderr = res.stderr.decode("utf-8", errors="replace")
    if res.returncode != 0:
        raise FFmpegError(f"ffmpeg exited with code {res.returncode}", stderr)
    size = width * height * 3
    if len(res.stdout) < n * size:
        raise FFmpegError(f"no frame at {what} in {video.name}", stderr)
    buf = np.frombuffer(res.stdout[: n * size], dtype=np.uint8)
    return [f.copy() for f in buf.reshape(n, height, width, 3)]


def _sample_batched(
    video: Path, times: list[float], vf: str, width: int, height: int
) -> list[np.ndarray]:
    """Frame k/fps sits at index k; time t maps to the first frame at or after it."""
    fps = probe_fps(video)
    index = [max(math.ceil(t * fps - 1e-6), 0) for t in times]
    uniq = sorted(set(index))
    by_index: dict[int, np.ndarray] = {}
    for i in range(0, len(uniq), BATCH_MAX):
        chunk = uniq[i : i + BATCH_MAX]
        first = chunk[0]
        select = "+".join(f"eq(n\\,{k - first})" for k in chunk)
        seek = ["-ss", f"{max((first - 0.5) / fps, 0.0):.4f}"]
        what = f"{chunk[-1] / fps:.3f}s"
        frames = _grab(video, seek, f"select='{select}',{vf}", width, height, len(chunk), what)
        by_index.update(zip(chunk, frames, strict=True))
    return [by_index[k] for k in index]
