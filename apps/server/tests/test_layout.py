import subprocess
from pathlib import Path

import pytest

from hookline.pipeline.layout import SceneLayout, classify_scene, plan_layouts
from hookline.vision.faces import Face


def test_corner_cam_is_stacked():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)
    s = classify_scene([(t, [cam]) for t in range(10)], 0, 10)
    assert s.kind == "STACKED" and s.facecam[0] < 0.78 and s.facecam[0] + s.facecam[2] <= 1.0


def test_big_face_tracks():
    face = Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9)
    assert classify_scene([(t, [face]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_two_faces_split():
    f1 = Face(x=0.1, y=0.2, w=0.15, h=0.3, score=0.9)
    f2 = Face(x=0.7, y=0.2, w=0.15, h=0.3, score=0.9)
    assert classify_scene([(t, [f1, f2]) for t in range(10)], 0, 10).kind == "SPLIT"


def test_no_faces_fit_blur():
    assert classify_scene([(t, []) for t in range(10)], 0, 10).kind == "FIT_BLUR"


def test_override_forces_stacked():
    s = classify_scene([(0, [])], 0, 5, facecam_override=(0, 0, 0.3, 0.3))
    assert s.kind == "STACKED" and s.facecam == (0, 0, 0.3, 0.3)


def test_stacked_box_grows_around_centre_and_shifts_inside_frame():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)
    s = classify_scene([(t, [cam]) for t in range(10)], 0, 10)
    x, y, w, h = s.facecam
    assert w == pytest.approx(0.24) and h == pytest.approx(0.364)
    assert x == pytest.approx(0.82 - 0.12)  # centred on face, already inside horizontally
    assert y == pytest.approx(0.0)  # would start at -0.062: shifted down, not shrunk


def test_stacked_box_in_bottom_right_corner_is_shifted_left_and_up():
    cam = Face(x=0.9, y=0.85, w=0.08, h=0.14, score=0.9)
    s = classify_scene([(t, [cam]) for t in range(10)], 0, 10)
    x, y, w, h = s.facecam
    assert s.kind == "STACKED"
    assert x + w == pytest.approx(1.0) and y + h == pytest.approx(1.0)
    assert w == pytest.approx(0.24) and h == pytest.approx(0.364)


def test_moving_corner_face_is_not_stacked():
    # Small corner face whose centre wanders more than 0.03: not a fixed facecam.
    samples = [
        (t, [Face(x=0.66 + 0.04 * (t % 5), y=0.05, w=0.08, h=0.14, score=0.9)]) for t in range(10)
    ]
    assert classify_scene(samples, 0, 10).kind != "STACKED"


def test_facecam_needs_sixty_percent_of_samples():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)
    samples = [(t, [cam] if t < 5 else []) for t in range(10)]
    assert classify_scene(samples, 0, 10).kind != "STACKED"
    samples = [(t, [cam] if t < 6 else []) for t in range(10)]
    assert classify_scene(samples, 0, 10).kind == "STACKED"


def test_centre_face_small_is_not_facecam():
    small_centre = Face(x=0.45, y=0.45, w=0.08, h=0.14, score=0.9)
    assert classify_scene([(t, [small_centre]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_low_score_faces_are_ignored():
    weak = Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.5)
    assert classify_scene([(t, [weak]) for t in range(10)], 0, 10).kind == "FIT_BLUR"


def test_split_x_is_sorted_median_centres():
    left = Face(x=0.1, y=0.2, w=0.2, h=0.3, score=0.9)
    right = Face(x=0.6, y=0.2, w=0.2, h=0.3, score=0.9)
    s = classify_scene([(t, [right, left]) for t in range(10)], 0, 10)
    assert s.kind == "SPLIT"
    assert s.split_x == pytest.approx((0.2, 0.7))


def test_two_small_faces_are_not_split():
    f1 = Face(x=0.3, y=0.4, w=0.06, h=0.1, score=0.9)
    f2 = Face(x=0.6, y=0.4, w=0.06, h=0.1, score=0.9)
    assert classify_scene([(t, [f1, f2]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_track_x_follows_largest_face_per_sample():
    small = Face(x=0.0, y=0.4, w=0.05, h=0.1, score=0.9)
    samples = [
        (float(t), [small, Face(x=0.1 * t, y=0.2, w=0.2, h=0.4, score=0.9)]) for t in range(5)
    ]
    s = classify_scene(samples, 0, 5)
    assert s.kind == "FACE_TRACK"
    assert s.track_x == pytest.approx([(float(t), 0.1 * t + 0.1) for t in range(5)])


def test_face_track_needs_half_the_samples():
    face = Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9)
    assert (
        classify_scene([(t, [face] if t < 4 else []) for t in range(10)], 0, 10).kind == "FIT_BLUR"
    )
    assert (
        classify_scene([(t, [face] if t < 5 else []) for t in range(10)], 0, 10).kind
        == "FACE_TRACK"
    )


def test_scene_layout_carries_times():
    s = classify_scene([(2.5, [])], 2.0, 3.0)
    assert isinstance(s, SceneLayout) and (s.start, s.end) == (2.0, 3.0)
    assert s.facecam is None and s.track_x == [] and s.split_x is None


class _RecordingDetector:
    def __init__(self, faces=None):
        self.frames = []
        self.faces = faces or []

    def detect(self, bgr):
        self.frames.append(bgr)
        return list(self.faces)


def test_plan_layouts_without_cuts_is_one_scene(tmp_path):
    video = tmp_path / "flat.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=gray:size=320x180:rate=25:duration=4", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    det = _RecordingDetector()
    layouts = plan_layouts(video, det)
    assert len(layouts) == 1
    assert layouts[0].start == 0 and layouts[0].end == pytest.approx(4.0, abs=0.1)
    assert layouts[0].kind == "FIT_BLUR"
    assert len(det.frames) == 4  # one sample per second
    assert det.frames[0].shape == (360, 640, 3)


def test_plan_layouts_splits_on_hard_cut(tmp_path):
    video = tmp_path / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=25:duration=3",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=3",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    layouts = plan_layouts(video, _RecordingDetector())
    assert len(layouts) == 2
    assert layouts[0].end == pytest.approx(3.0, abs=0.1) == layouts[1].start


def test_plan_layouts_override_and_crop_pass_through(tmp_path):
    video = tmp_path / "flat.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=gray:size=320x240:rate=25:duration=2", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    det = _RecordingDetector()
    layouts = plan_layouts(video, det, sample_every=1.0, crop=(320, 180, 0, 30))
    assert [s.kind for s in layouts] == ["FIT_BLUR"]
    assert det.frames[0].shape == (360, 640, 3)  # cropped to 16:9 before scaling

    det = _RecordingDetector()
    layouts = plan_layouts(video, det, facecam_override=(0.7, 0, 0.3, 0.3))
    assert [(s.kind, s.facecam) for s in layouts] == [("STACKED", (0.7, 0, 0.3, 0.3))]
    assert det.frames == []  # override needs no detection


@pytest.mark.integration
def test_plan_layouts_finds_facecam_in_synthetic_stream(synthetic_stream: Path):
    from hookline.assets import ensure_yunet
    from hookline.vision.faces import FaceDetector

    layouts = plan_layouts(synthetic_stream, FaceDetector(ensure_yunet()), sample_every=1.0)
    assert [s.kind for s in layouts] == ["STACKED"]
    x, y, w, _ = layouts[0].facecam
    assert x > 0.5 and y < 0.2 and x + w <= 1.0
