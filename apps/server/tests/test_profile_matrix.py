"""The stream profile on a synthetic matrix of cam geometries and
source sizes (16:9 / 4:3 / square cams, a borderless cam flush in a corner, 1080p and 1440p,
a full-frame webcam that must not profile)."""

import numpy as np
import pytest

from hookline.pipeline import profile as profile_mod
from hookline.pipeline.profile import build_profile, ring_visible, snap_cam
from hookline.vision.faces import Face

N = 24


def _scene(
    size: tuple[int, int],
    cam: tuple[float, float, float, float],
    border: bool = True,
    smooth_room: bool = False,
    face_share: float = 0.32,
    n: int = N,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(n, H, W) gray frames of a stream at `size`: moving gameplay, a static facecam overlay
    at `cam` (normalized x, y, w, h: its outer box) with a 2 px light border (or none) round
    a room (textured or a smooth wall) and the streamer (a live patch); and the face box (px,
    x y w h) of height `face_share` of the cam."""
    width, height = size
    rng = np.random.default_rng(seed)
    x0, y0 = round(cam[0] * width), round(cam[1] * height)
    x1, y1 = round((cam[0] + cam[2]) * width), round((cam[1] + cam[3]) * height)
    cw, ch = x1 - x0, y1 - y0
    yy, xx = np.mgrid[0:height, 0:width]
    if smooth_room:
        room = (70 + 30 * (np.mgrid[0:ch, 0:cw][1] / max(cw, 1))).astype(np.uint8)
    else:
        room = rng.integers(40, 200, size=(ch, cw)).astype(np.uint8)
    fh = face_share * ch
    fw = 0.75 * fh
    fx, fy = x0 + cw / 2 - fw / 2, y0 + 0.45 * ch - fh / 2
    out = np.empty((n, height, width), np.uint8)
    unit = width / 320
    for i in range(n):
        f = ((xx * 0.4 / unit + yy * 0.3 / unit + i * 17) % 110 + 20).astype(np.uint8)
        bw, bh = int(30 * unit), int(20 * unit)
        bx, by = rng.integers(0, width - bw), rng.integers(0, height - bh)
        f[by : by + bh, bx : bx + bw] = 200
        f[y0:y1, x0:x1] = room
        if border:
            b = max(2, round(unit * 0.5))
            f[y0 : y0 + b, x0:x1] = f[y1 - b : y1, x0:x1] = 235
            f[y0:y1, x0 : x0 + b] = f[y0:y1, x1 - b : x1] = 235
        # the streamer: a live head-and-shoulders patch around the face
        px0, px1 = int(fx - 0.3 * fw), int(fx + 1.3 * fw)
        py0, py1 = int(fy - 0.4 * fh), min(int(fy + 2.2 * fh), y1 - 3)
        dy, dx = i % 3, i % 4
        f[py0 + dy : py1, px0 + dx : px1 + dx] = rng.integers(
            0, 255, size=(py1 - py0 - dy, px1 - px0)
        )
        out[i] = f
    return out, (fx, fy, fw, fh)


# (name, size, outer cam box normalized, border, smooth room)
CASES = [
    ("16:9 cam 1080p", (1920, 1080), (0.03, 0.62, 150 * 1.5 / 1920, 84 * 1.5 / 1080), True, False),
    ("16:9 cam smooth wall", (1280, 720), (0.70, 0.05, 240 / 1280, 135 / 720), True, True),
    ("4:3 cam", (1280, 720), (0.04, 0.55, 200 / 1280, 150 / 720), True, False),
    ("square cam", (1280, 720), (0.04, 0.50, 201 / 1280, 200 / 720), True, False),
    ("borderless flush corner", (1280, 720), (0.0, 1 - 160 / 720, 284 / 1280, 160 / 720), False, True),
    ("bordered flush top-right", (1280, 720), (1 - 220 / 1280, 0.0, 220 / 1280, 165 / 720), True, False),
]  # fmt: skip


@pytest.mark.parametrize(("name", "size", "cam", "border", "smooth"), CASES)
def test_snap_cam_matrix(name, size, cam, border, smooth):
    gray, face = _scene(size, cam, border=border, smooth_room=smooth)
    snapped = snap_cam(gray, face)
    assert snapped is not None, name
    outer, inner = snapped
    w, h = size
    want = (cam[0] * w, cam[1] * h, (cam[0] + cam[2]) * w, (cam[1] + cam[3]) * h)
    assert outer == pytest.approx(want, abs=3), name
    assert all(o <= i for o, i in zip(outer[:2], inner[:2], strict=True))
    assert all(i <= o for o, i in zip(outer[2:], inner[2:], strict=True))


def _detector_for(face_norm: tuple[float, float, float, float]):
    class _Det:
        def detect(self, bgr):
            x, y, w, h = face_norm
            return [Face(x, y, w, h, 0.9, 0.0, y + h / 3)]

    return _Det()


def _mock_video(monkeypatch, size, frames_fn):
    """build_profile reads `frames_fn(width, crop)` (BGR) instead of decoding a video; records
    the widths asked for."""
    asked: list[tuple[int, bool]] = []  # (width, whole frame)

    def fake_sample(video, times, width=640, crop=None):
        asked.append((width, crop is None))
        return frames_fn(width, crop, len(times))

    monkeypatch.setattr(profile_mod, "sample_frames", fake_sample)
    monkeypatch.setattr(profile_mod, "probe_size", lambda p: size)
    monkeypatch.setattr(profile_mod, "probe_duration", lambda p: 3600.0)
    return asked


def _frames_fn(size, cam, **kw):
    """Frames of the scene at the requested width (rendered at that size: a source decoded
    at a smaller width), optionally cropped (source px); successive calls read successive
    frames."""
    cache: dict = {}
    calls = [0]

    def fn(width, crop, n):
        k = calls[0]
        calls[0] += n
        if crop is not None:
            if "full" not in cache:
                cache["full"] = _scene(size, cam, n=N, **kw)[0]
            cw, ch, cx, cy = crop
            assert width == cw  # crops are read at full resolution
            g = [cache["full"][(k + i) % N, cy : cy + ch, cx : cx + cw] for i in range(n)]
        else:
            height = round(width * size[1] / size[0] / 2) * 2
            if width not in cache:
                cache[width] = _scene((width, height), cam, n=N, **kw)[0]
            g = [cache[width][(k + i) % N] for i in range(n)]
        return [np.repeat(f[..., None], 3, axis=2) for f in g]

    return fn


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440)])
def test_build_profile_analyses_at_most_1280_wide_and_maps_back(monkeypatch, size):
    monkeypatch.setattr(profile_mod, "PROFILE_SAMPLES", 24)
    cam = (0.04, 0.55, 0.16, 0.28)
    _, face = _scene(size, cam, n=1)
    w, h = size
    face_norm = (face[0] / w, face[1] / h, face[2] / w, face[3] / h)
    asked = _mock_video(monkeypatch, size, _frames_fn(size, cam))
    prof = build_profile(None, _detector_for(face_norm))  # type: ignore[arg-type]
    assert all(wd <= 1280 for wd, whole in asked if whole)  # whole frames: downscaled
    assert prof.cam is not None and prof.size == size
    want = (cam[0] * w, cam[1] * h, (cam[0] + cam[2]) * w, (cam[1] + cam[3]) * h)
    assert prof.outer == pytest.approx(want, abs=3)
    # the ring reference matches a full-resolution frame of the source
    full = _scene(size, cam, n=2)[0]
    assert ring_visible(full[0], prof.ring) or ring_visible(full[0], prof.ring_in)


def test_build_profile_rejects_a_full_frame_webcam(monkeypatch):
    """A Just Chatting webcam filling the frame (a still room, the streamer live in the
    middle) is no corner cam: no profile cam, whatever the snap finds."""
    monkeypatch.setattr(profile_mod, "PROFILE_SAMPLES", 24)
    size = (1280, 720)
    cam = (0.0, 0.0, 1.0, 1.0)
    _, face = _scene(size, cam, n=1, face_share=0.2)
    face_norm = (face[0] / 1280, face[1] / 720, face[2] / 1280, face[3] / 720)
    _mock_video(monkeypatch, size, _frames_fn(size, cam, face_share=0.2))
    prof = build_profile(None, _detector_for(face_norm))  # type: ignore[arg-type]
    assert prof.cam is None


def test_build_profile_rejects_a_huge_or_centred_cam(monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILE_SAMPLES", 24)
    size = (1280, 720)
    # > 35% of the frame (its face small enough for a cam face); centred
    for cam, share in [((0.02, 0.05, 0.62, 0.9), 0.2), ((0.4, 0.38, 0.2, 0.24), 0.32)]:
        _, face = _scene(size, cam, n=1, face_share=share)
        face_norm = (face[0] / 1280, face[1] / 720, face[2] / 1280, face[3] / 720)
        _mock_video(monkeypatch, size, _frames_fn(size, cam, face_share=share))
        prof = build_profile(None, _detector_for(face_norm))  # type: ignore[arg-type]
        assert prof.cam is None, cam


def test_cam_cluster_radius_scales_with_the_face():
    """A big cam face leaning around more than a fixed 0.04 of the frame (but well within its
    own size) is still one cluster; a tiny face jittering by its whole width is not."""
    rng = np.random.default_rng(5)
    big = [
        [Face(0.1 + rng.normal(0, 0.012), 0.5 + rng.normal(0, 0.02), 0.1, 0.2, 0.9, 0.0, None)]
        for _ in range(60)
    ]
    assert profile_mod.cam_cluster(big) is not None
    tiny = [
        [Face(0.1 + rng.normal(0, 0.012), 0.5 + rng.normal(0, 0.012), 0.02, 0.035, 0.9, 0.0, None)]
        for _ in range(60)
    ]
    assert profile_mod.cam_cluster(tiny) is None
