import json
import subprocess

import numpy as np
import pytest

from hookline.pipeline import profile as profile_mod
from hookline.pipeline.profile import (
    LayoutProfile,
    cam_cluster,
    label_masks,
    layout_profile,
    ring_visible,
    snap_cam,
    vod_bars,
)
from hookline.vision.faces import Face

W, H = 320, 180
CAM = (20, 90, 72, 72)  # outer x, y, w, h (px): a 2 px white border around a square cam


def _face(cx: float, cy: float, h: float = 0.12, w: float = 0.06, score: float = 0.9) -> Face:
    return Face(cx - w / 2, cy - h / 2, w, h, score, 0.0, cy - h / 6)


def test_cam_cluster_finds_a_fixed_small_face_in_most_samples():
    rng = np.random.default_rng(0)
    samples = []
    for i in range(150):
        faces = []
        if i % 10 < 7:  # 70% of the samples
            faces.append(_face(0.12 + rng.normal(0, 0.004), 0.66 + rng.normal(0, 0.004)))
        if i % 3 == 0:  # an in-game face, moving around
            faces.append(_face(rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.8), h=0.2))
        samples.append(faces)
    found = cam_cluster(samples)
    assert found is not None
    face, share = found
    assert face.cx == pytest.approx(0.12, abs=0.005) and face.cy == pytest.approx(0.66, abs=0.005)
    assert share == pytest.approx(0.7, abs=0.01)


def test_cam_cluster_none_when_scattered_rare_or_big():
    rng = np.random.default_rng(1)
    wander = [[_face(0.12 + rng.normal(0, 0.03), 0.66 + rng.normal(0, 0.03))] for _ in range(150)]
    assert cam_cluster(wander) is None  # spread >= 0.02
    rare = [[_face(0.12, 0.66)] if i % 2 else [] for i in range(150)]
    assert cam_cluster(rare) is None  # 50% < 60%
    # a still full-frame speaker (podcast / Just Chatting): not a facecam
    assert cam_cluster([[_face(0.5, 0.4, h=0.45, w=0.25)] for _ in range(150)]) is None


def _frames(n: int = 40, cam: tuple[int, int, int, int] | None = CAM, label: bool = True):
    """Gray frames: moving noise gameplay, a static cam (bordered) whose middle changes (the
    streamer), and under it a label strip whose text changes between two names."""
    rng = np.random.default_rng(2)
    out = np.empty((n, H, W), np.uint8)
    room = rng.integers(40, 200, size=(H, W)).astype(np.uint8)  # static cam background
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        # dim smooth gameplay that changes every frame, plus a bright object moving around
        f = ((xx * 0.4 + yy * 0.3 + i * 17) % 110).astype(np.uint8)
        bx, by = rng.integers(0, W - 30), rng.integers(0, H - 20)
        f[by : by + 20, bx : bx + 30] = 200
        if cam is not None:
            x, y, w, h = cam
            f[y : y + h, x : x + w] = 0  # dark outer line
            f[y + 1 : y + h - 1, x + 1 : x + w - 1] = 255  # 2 px white border
            f[y + 3 : y + h - 3, x + 3 : x + w - 3] = room[y + 3 : y + h - 3, x + 3 : x + w - 3]
            fy, fx = y + h // 2 + (i % 3), x + w // 2 + (i % 4)
            f[fy - 12 : fy + 12, fx - 9 : fx + 9] = rng.integers(0, 255, size=(24, 18))
            if label:
                ly = y + h + 6
                f[ly : ly + 8, x + 10 : x + w - 10] = 0
                for k in range(x + 12, x + w - 12, 4 if i % 2 else 6):
                    f[ly + 2 : ly + 6, k] = 230  # "text" strokes
        out[i] = f
    return out


def test_snap_cam_finds_the_bordered_rectangle_and_trims_the_border():
    gray = _frames()
    x, y, w, h = CAM
    face = (x + w / 2 - 9, y + h / 2 - 12, 18, 26)  # px
    snapped = snap_cam(gray, face)
    assert snapped is not None
    outer, inner = snapped
    assert outer == pytest.approx((x, y, x + w, y + h), abs=1)
    # inside the white border (rows/cols x+1..x+2 are border): none of it left
    assert x + 3 <= inner[0] <= x + 5 and y + 3 <= inner[1] <= y + 5
    assert x + w - 5 <= inner[2] <= x + w - 3 and y + h - 5 <= inner[3] <= y + h - 3


def test_snap_cam_none_without_a_static_border():
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 255, size=(30, H, W)).astype(np.uint8)
    assert snap_cam(gray, (40, 110, 18, 26)) is None


def test_snap_cam_none_when_nothing_outside_moves():
    """A still podcast frame: static edges everywhere, the outside of any box doesn't change."""
    gray = _frames()
    still = np.repeat(gray[:1], len(gray), axis=0)
    x, y, w, h = CAM
    still[:, y + 30 : y + 42, x + 28 : x + 46] = gray[:, y + 30 : y + 42, x + 28 : x + 46]
    assert snap_cam(still, (x + w / 2 - 9, y + h / 2 - 12, 18, 26)) is None


def test_label_masks_cover_the_strip_under_the_cam():
    gray = _frames()
    x, y, w, h = CAM
    masks = label_masks(gray, (x, y, x + w, y + h))
    assert len(masks) == 1
    mx0, my0, mx1, my1 = masks[0]
    assert my0 == y + h and y + h + 12 <= my1 <= y + h + 20
    assert mx0 <= x + 12 and mx1 >= x + w - 12


def test_label_masks_none_without_a_label():
    gray = _frames(label=False)
    x, y, w, h = CAM
    assert label_masks(gray, (x, y, x + w, y + h)) == []


def test_vod_bars_union_of_content_over_the_vod():
    gray = np.zeros((10, H, W), np.uint8)
    gray[:, 20:160, :] = 90  # letterbox bars: rows 0-19 and 160-179 black in every frame
    gray[3, 20:160, :] = 5  # a near-black (dark game) frame adds nothing
    assert vod_bars(gray) == (W, 140, 0, 20)
    gray[4, 10:20, 100:200] = 60  # content reaching into the bar in one frame only
    assert vod_bars(gray) == (W, 150, 0, 10)


def test_vod_bars_never_crop_the_facecam_and_ignore_small_bars():
    gray = np.zeros((10, H, W), np.uint8)
    gray[:, :, 40:280] = 90  # pillarbox
    assert vod_bars(gray) == (240, H, 40, 0)
    assert vod_bars(gray, protect=(10, 60, 80, 120)) == (270, H, 10, 0)
    gray[:, :, 2:318] = 90
    assert vod_bars(gray) is None  # < 2% of each dimension


def test_ring_visible_matches_the_border_reference():
    gray = _frames(n=4)
    x, y, w, h = CAM
    ring = profile_mod.ring_reference(np.median(gray, axis=0), (x, y, x + w, y + h))
    assert ring_visible(gray[0], ring)
    hidden = _frames(n=4, cam=None)
    assert not ring_visible(hidden[0], ring)


class _CamDetector:
    """Detects the streamer's face in the cam (frames are 640 wide when sampled)."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, bgr):
        self.calls += 1
        x, y, w, h = CAM
        return [_face((x + w / 2) / W, (y + h / 2) / H, h=26 / H, w=18 / W)]


@pytest.fixture(scope="module")
def stream_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("stream") / "stream.mp4"
    frames = _frames(n=60)
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "2", "-i", "-",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    proc.communicate(frames.tobytes())
    assert proc.returncode == 0
    return path


def test_layout_profile_detects_cam_and_caches(stream_video, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILE_SAMPLES", 24)
    det = _CamDetector()
    prof = layout_profile(tmp_path, stream_video, det)
    assert isinstance(prof, LayoutProfile) and prof.cam is not None
    x, y, w, h = CAM
    assert prof.outer == pytest.approx((x, y, x + w, y + h), abs=1)
    assert prof.masks and prof.masks[0][1] == y + h
    assert prof.size == (W, H) and prof.ring and prof.ring_in
    data = json.loads((tmp_path / "layout_profile.json").read_text(encoding="utf-8"))
    assert data["key"]["version"] == profile_mod.PROFILE_VERSION
    calls = det.calls
    again = layout_profile(tmp_path, stream_video, det)
    assert again == prof and det.calls == calls  # cached: no re-sampling


def test_layout_profile_without_detector_has_bars_only(stream_video, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILE_SAMPLES", 12)
    prof = layout_profile(tmp_path, stream_video, None)
    assert prof.cam is None and prof.bars is None
    # a later run with a detector recomputes (the cache was made without faces)
    assert layout_profile(tmp_path, stream_video, _CamDetector()).cam is not None
