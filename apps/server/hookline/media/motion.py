"""Per-second visual motion of a whole VOD from a cheap low-resolution decode."""

import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from hookline.media.ffmpeg import FFmpegError

MOTION_FPS = 4
MOTION_W, MOTION_H = 64, 36
_BLOCK_FRAMES = 4 * 600  # frames read per block (~10 minutes at 4 fps)
# A 1 h 720p VOD decodes in ~20 s; a decode still running after this is stuck (a damaged file).
MOTION_TIMEOUT_S = 1800.0
# Wave H: a "screen" frame (death / loading / menu screen, a fade) of the same decode: near
# uniform -- its 90th and 10th percentile grey levels at most SCREEN_SPREAD apart (a black
# screen with small text; a fade's flat colour) -- or near white (10th percentile >=
# SCREEN_WHITE). Measured on a horror stream: death / loading screens and fades spread 0-7,
# the darkest gameplay 7-60 (with voice: see `signals.static_runs`), a talking head 60+.
SCREEN_SPREAD = 8.0
SCREEN_WHITE = 230.0


def frame_motion(
    src: Path, fps: int = MOTION_FPS, timeout: float = MOTION_TIMEOUT_S
) -> list[float]:
    """`frame_stats`' per-second motion."""
    return frame_stats(src, fps, timeout)[0]


def screen_frames(frames: np.ndarray) -> np.ndarray:
    """Per greyscale frame (rows of pixels): a screen frame (SCREEN_SPREAD, SCREEN_WHITE)."""
    if not len(frames):
        return np.zeros(0, bool)
    p10, p90 = np.percentile(frames, [10, 90], axis=1)
    return (p90 - p10 <= SCREEN_SPREAD) | (p10 >= SCREEN_WHITE)


def frame_stats(
    src: Path, fps: int = MOTION_FPS, timeout: float = MOTION_TIMEOUT_S
) -> tuple[list[float], list[int]]:
    """(motion, screen) per second of `src`. Motion: per second of `src`, the largest mean absolute difference (0-255 grey levels) between
    consecutive frames of a `fps`-frame/s, 64x36 greyscale decode: about 0 for a still frame,
    single digits for a talking head or a calm scene, tens for busy gameplay, near 100+ for a
    flash or a hard cut. Second i takes the differences into its frames (the change from the
    last frame of second i - 1 included). Streamed: a 1 h 720p VOD takes ~20 s.

    ffmpeg's stderr goes to a temporary file, never a pipe: a damaged VOD can log megabytes
    of concealment errors, which would fill an unread pipe (~4 KB on Windows) and stall the
    decode. A decode still running after `timeout` s is killed (FFmpegError).

    Screen: per second, a bitmask of its `fps` frames (bit k = frame k) that are screen
    frames (`screen_frames`), from the same decode."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-i", str(src), "-an", "-sn",
        "-vf", f"fps={fps},scale={MOTION_W}:{MOTION_H}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    size = MOTION_W * MOTION_H
    diffs: list[np.ndarray] = []
    screens: list[np.ndarray] = []
    prev: np.ndarray | None = None
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
            while chunk := proc.stdout.read(size * _BLOCK_FRAMES):
                frames = np.frombuffer(chunk[: len(chunk) // size * size], np.uint8)
                frames = frames.reshape(-1, size).astype(np.int16)
                screens.append(screen_frames(frames))
                if prev is not None:
                    frames = np.vstack([prev, frames])
                if len(frames) > 1:
                    diffs.append(np.abs(np.diff(frames, axis=0)).mean(axis=1))
                prev = frames[-1:]
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
    screen = [
        int(sum(1 << k for k, f in enumerate(flags[i * fps : (i + 1) * fps]) if f))
        for i in range(seconds)
    ]
    return out, screen
