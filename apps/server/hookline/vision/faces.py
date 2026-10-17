import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hookline.media.ffmpeg import FFmpegError, probe_size

Crop = tuple[int, int, int, int]  # ffmpeg crop=w:h:x:y, in source pixels


class Face(NamedTuple):
    """A detected face, normalized 0..1 relative to the frame (x, y = top-left)."""

    x: float
    y: float
    w: float
    h: float
    score: float

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
            faces.append(Face(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, float(r[14])))
        return sorted(faces, key=lambda f: f.score, reverse=True)


def sample_frames(
    video: Path, times: list[float], width: int = 640, crop: Crop | None = None
) -> list[np.ndarray]:
    """One BGR frame per time, cropped (if given) then scaled to `width` (even height)."""
    if crop is not None:
        src_w, src_h = crop[0], crop[1]
    else:
        src_w, src_h = probe_size(video)
    height = max(2, round(width * src_h / src_w / 2) * 2)
    vf = f"scale={width}:{height}"
    if crop is not None:
        vf = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," + vf
    frame_bytes = width * height * 3
    frames = []
    for t in times:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]  # fmt: skip
        try:
            res = subprocess.run(cmd, check=False, capture_output=True)
        except FileNotFoundError as e:
            raise FFmpegError("ffmpeg not found on PATH") from e
        stderr = res.stderr.decode("utf-8", errors="replace")
        if res.returncode != 0:
            raise FFmpegError(f"ffmpeg exited with code {res.returncode}", stderr)
        if len(res.stdout) < frame_bytes:
            raise FFmpegError(f"no frame at {t:.3f}s in {video.name}", stderr)
        buf = np.frombuffer(res.stdout[:frame_bytes], dtype=np.uint8)
        frames.append(buf.reshape(height, width, 3).copy())
    return frames
