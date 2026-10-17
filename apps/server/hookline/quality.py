"""Measurements on rendered frames, for the M2.5 quality bar."""

import subprocess
from pathlib import Path
from typing import Literal

import numpy as np

from hookline.media.subtitles import fonts_dir, subtitles_filter

OUT_W, OUT_H = 1080, 1920
TextColor = Literal["white", "yellow", "any"]


def render_ass_frame(ass_path: Path, t: float) -> np.ndarray:
    """Burn `ass_path` over a black 1080x1920 frame and return the frame at `t` s (HxWx3 RGB)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=black:s={OUT_W}x{OUT_H}:r=25:d={t + 1:.3f}",
        "-vf", subtitles_filter(Path(ass_path), fonts_dir()),
        "-ss", f"{t:.3f}", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    res = subprocess.run(cmd, check=True, capture_output=True)
    return np.frombuffer(res.stdout, np.uint8).reshape(OUT_H, OUT_W, 3)


def text_mask(frame: np.ndarray, color: TextColor = "any") -> np.ndarray:
    """Boolean mask of text-fill pixels: white (all channels high), yellow (R,G high, B low),
    or any bright pixel."""
    rgb = frame.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "white":
        return (r > 200) & (g > 200) & (b > 200)
    if color == "yellow":
        return (r > 200) & (g > 200) & (b < 90)
    return rgb.max(axis=2) > 128


def text_bbox(
    frame: np.ndarray, color: TextColor = "any", region: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) inclusive bounding box of `color` text pixels, or None. `region`
    (x0, y0, x1, y1) limits the search to part of the frame."""
    mask = text_mask(frame, color)
    if region is not None:
        x0, y0, x1, y1 = region
        clip = np.zeros_like(mask)
        clip[y0 : y1 + 1, x0 : x1 + 1] = True
        mask &= clip
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
