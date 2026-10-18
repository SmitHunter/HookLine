"""Per-second visual motion of a whole VOD from a cheap low-resolution decode."""

import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from hookline.media.ffmpeg import FFmpegError

MOTION_FPS = 4
MOTION_W, MOTION_H = 64, 36
_BLOCK_FRAMES = 4 * 600  # frames read per block (~10 minutes at 4 fps)
# A 1 h 720p VOD decodes in ~20 s (a 1080p60 one in a few minutes on a slow CPU); a decode
# still running after `motion_timeout` is stuck (a damaged file). The limit scales with the
# source so a long VOD on a slow machine is never cut off: half real time, at least 10 min.
MOTION_TIMEOUT_MIN_S = 600.0
MOTION_TIMEOUT_RATE = 0.5
# Wave H: a "screen" frame (death / loading / menu screen, a fade) of the same decode: near
# uniform -- its 90th and 10th percentile grey levels at most SCREEN_SPREAD apart (a black
# screen with small text; a fade's flat colour) -- or near white (10th percentile >=
# SCREEN_WHITE). Measured on a horror stream: death / loading screens and fades spread 0-7,
# the darkest gameplay 7-60 (with voice: see `signals.static_runs`), a talking head 60+.
SCREEN_SPREAD = 8.0
SCREEN_WHITE = 230.0


Rect = tuple[float, float, float, float]  # normalized x0, y0, x1, y1


def motion_timeout(duration: float) -> float:
    """The motion decode's time limit (s) for a `duration`-second source."""
    return max(MOTION_TIMEOUT_MIN_S, duration * MOTION_TIMEOUT_RATE)


def frame_motion(
    src: Path, fps: int = MOTION_FPS, timeout: float = MOTION_TIMEOUT_MIN_S
) -> list[float]:
    """`frame_stats`' per-second motion."""
    return frame_stats(src, fps, timeout)[0]


def frame_chroma(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per frame (rows of U and V samples, 0-255): the mean chroma magnitude
    sqrt((U-128)^2 + (V-128)^2) -- about 0 for a grey / black / white screen, a few units for
    dark but coloured gameplay."""
    if not len(u):
        return np.zeros(0)
    du = u.astype(np.float64) - 128.0
    dv = v.astype(np.float64) - 128.0
    return np.sqrt(du * du + dv * dv).mean(axis=1)


def cell_mask(rects: list[Rect] | None, w: int = MOTION_W, h: int = MOTION_H) -> np.ndarray:
    """Flattened `w` x `h` cells outside every `rects` box (a cell any box touches is out)."""
    keep = np.ones((h, w), bool)
    for x0, y0, x1, y1 in rects or []:
        c0, c1 = int(np.floor(x0 * w)), int(np.ceil(x1 * w))
        r0, r1 = int(np.floor(y0 * h)), int(np.ceil(y1 * h))
        keep[max(r0, 0) : min(r1, h), max(c0, 0) : min(c1, w)] = False
    return keep.ravel()


def screen_frames(frames: np.ndarray) -> np.ndarray:
    """Per greyscale frame (rows of pixels): a screen frame (SCREEN_SPREAD, SCREEN_WHITE)."""
    if not len(frames):
        return np.zeros(0, bool)
    p10, p90 = np.percentile(frames, [10, 90], axis=1)
    return (p90 - p10 <= SCREEN_SPREAD) | (p10 >= SCREEN_WHITE)


def frame_stats(
    src: Path,
    fps: int = MOTION_FPS,
    timeout: float = MOTION_TIMEOUT_MIN_S,
    mask: list[Rect] | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[list[float], list[int], list[float | None]]:
    """(motion, screen, chroma) per second of `src`. Motion: per second of `src`, the largest mean absolute difference (0-255 grey levels) between
    consecutive frames of a `fps`-frame/s, 64x36 greyscale decode: about 0 for a still frame,
    single digits for a talking head or a calm scene, tens for busy gameplay, near 100+ for a
    flash or a hard cut. Second i takes the differences into its frames (the change from the
    last frame of second i - 1 included). Streamed: a 1 h 720p VOD takes ~20 s.

    ffmpeg's stderr goes to a temporary file, never a pipe: a damaged VOD can log megabytes
    of concealment errors, which would fill an unread pipe (~4 KB on Windows) and stall the
    decode. A decode still running after `timeout` s is killed (FFmpegError). `progress`, when
    given, is called after each block of frames with the source seconds decoded so far.

    Screen: per second, a bitmask of its `fps` frames (bit k = frame k) that are screen
    frames (`screen_frames`), from the same decode. Chroma (wave H round 1): per second, the
    mean `frame_chroma` of its screen frames over the cells outside `mask` (a stream's cam and
    labels: normalized boxes), None for a second without screen frames."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-i", str(src), "-an", "-sn",
        "-vf", f"fps={fps},scale={MOTION_W}:{MOTION_H}:flags=area,format=yuv444p",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    size = MOTION_W * MOTION_H
    keep = cell_mask(mask)
    diffs: list[np.ndarray] = []
    screens: list[np.ndarray] = []
    chromas: list[np.ndarray] = []
    prev: np.ndarray | None = None
    n_read = 0
    with tempfile.TemporaryFile() as errf:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
        except FileNotFoundError as e:
            raise FFmpegError("ffmpeg not found on PATH") from e
        assert proc.stdout is not None
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(timeout, kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            while chunk := proc.stdout.read(3 * size * _BLOCK_FRAMES):
                planes = np.frombuffer(chunk[: len(chunk) // (3 * size) * 3 * size], np.uint8)
                planes = planes.reshape(-1, 3, size)
                frames = planes[:, 0].astype(np.int16)
                screens.append(screen_frames(frames))
                chromas.append(frame_chroma(planes[:, 1, keep], planes[:, 2, keep]))
                if prev is not None:
                    frames = np.vstack([prev, frames])
                if len(frames) > 1:
                    diffs.append(np.abs(np.diff(frames, axis=0)).mean(axis=1))
                prev = frames[-1:]
                n_read += len(planes)
                if progress is not None:
                    progress(n_read / fps)
            code = proc.wait()
        finally:
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace")
    if timed_out.is_set():
        raise FFmpegError(f"motion analysis timed out after {timeout:g} s", err)
    if code != 0:
        raise FFmpegError(f"ffmpeg exited with code {code}", err)
    d = np.concatenate(diffs) if diffs else np.zeros(0)
    n_frames = len(d) + (1 if prev is not None else 0)
    seconds = -(-n_frames // fps)
    out = []
    for i in range(seconds):
        # differences into frames i*fps .. i*fps+fps-1, i.e. d[i*fps-1 .. i*fps+fps-2]
        part = d[max(i * fps - 1, 0) : i * fps + fps - 1]
        out.append(round(float(part.max()), 2) if len(part) else 0.0)
    flags = np.concatenate(screens) if screens else np.zeros(0, bool)
    chroma = np.concatenate(chromas) if chromas else np.zeros(0)
    screen, tint = [], []
    for i in range(seconds):
        sl = slice(i * fps, (i + 1) * fps)
        screen.append(int(sum(1 << k for k, f in enumerate(flags[sl]) if f)))
        c = chroma[sl][flags[sl]]
        tint.append(round(float(c.mean()), 3) if len(c) else None)
    return out, screen, tint
