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


def test_large_off_centre_face_is_face_track_not_stacked():
    big = Face(x=0.05, y=0.05, w=0.12, h=0.25, score=0.9)  # in corner region, but too tall
    assert classify_scene([(t, [big]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_mid_size_off_centre_face_is_face_track_not_stacked():
    # Just Chatting webcam, streamer sat back and off-centre: cx 0.2, cy 0.25, h 0.18.
    mid = Face(x=0.15, y=0.16, w=0.10, h=0.18, score=0.9)
    assert classify_scene([(t, [mid]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_facecam_height_ceiling_is_exclusive():
    just_under = Face(x=0.78, y=0.05, w=0.08, h=0.149, score=0.9)
    at_ceiling = Face(x=0.78, y=0.05, w=0.08, h=0.15, score=0.9)
    assert classify_scene([(t, [just_under]) for t in range(10)], 0, 10).kind == "STACKED"
    assert classify_scene([(t, [at_ceiling]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def test_facecam_box_area_must_be_under_a_fifth():
    # h=0.14 passes the height ceiling, but grown box 0.6 x 0.364 = 0.218 >= 0.20.
    wide = Face(x=0.75, y=0.05, w=0.20, h=0.14, score=0.9)
    assert classify_scene([(t, [wide]) for t in range(10)], 0, 10).kind == "FACE_TRACK"
    # w=0.18: 0.54 x 0.364 = 0.197 < 0.20 -> still a facecam.
    ok = Face(x=0.75, y=0.05, w=0.18, h=0.14, score=0.9)
    assert classify_scene([(t, [ok]) for t in range(10)], 0, 10).kind == "STACKED"


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


def test_two_small_faces_are_a_wide_shot_not_split():
    # Two comparable faces, both under SPLIT_MIN_H: a wide group shot -> FIT_BLUR.
    f1 = Face(x=0.3, y=0.4, w=0.06, h=0.1, score=0.9)
    f2 = Face(x=0.6, y=0.4, w=0.06, h=0.1, score=0.9)
    assert classify_scene([(t, [f1, f2]) for t in range(10)], 0, 10).kind == "FIT_BLUR"


def test_small_face_with_tiny_background_face_still_tracks():
    main = Face(x=0.3, y=0.4, w=0.06, h=0.12, score=0.9)
    tiny = Face(x=0.6, y=0.4, w=0.03, h=0.05, score=0.9)  # not comparable (h > 2x apart)
    assert classify_scene([(t, [main, tiny]) for t in range(10)], 0, 10).kind == "FACE_TRACK"


def _row_of_three(cy=0.39, h=0.11):
    return [Face(x=cx - 0.03, y=cy - h / 2, w=0.06, h=h, score=0.9) for cx in (0.15, 0.5, 0.85)]


def test_three_faces_in_a_row_are_fit_blur_not_stacked():
    # Wide group shot: the outer faces sit in "corners" (cy < 0.4) but a comparable face
    # outside the grown box vetoes STACKED; all faces are small -> FIT_BLUR.
    s = classify_scene([(t, _row_of_three()) for t in range(10)], 0, 10, src_h=1080)
    assert s.kind == "FIT_BLUR"


def test_facecam_veto_needs_thirty_percent_of_cam_samples():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)
    other = Face(x=0.3, y=0.5, w=0.06, h=0.08, score=0.9)  # h >= 0.5 x cam.h, outside box
    tiny = Face(x=0.3, y=0.5, w=0.03, h=0.05, score=0.9)  # too small to veto
    weak = Face(x=0.3, y=0.5, w=0.06, h=0.08, score=0.5)  # below MIN_SCORE
    two = [(t, [cam, other] if t < 2 else [cam]) for t in range(10)]
    assert classify_scene(two, 0, 10).kind == "STACKED"  # 20% < 30%
    three = [(t, [cam, other] if t < 3 else [cam]) for t in range(10)]
    assert classify_scene(three, 0, 10).kind != "STACKED"
    assert classify_scene([(t, [cam, tiny]) for t in range(10)], 0, 10).kind == "STACKED"
    assert classify_scene([(t, [cam, weak]) for t in range(10)], 0, 10).kind == "STACKED"


def test_facecam_box_must_be_160_source_px_tall():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)  # grown h 0.364
    samples = [(t, [cam]) for t in range(10)]
    assert classify_scene(samples, 0, 10, src_h=1080).kind == "STACKED"  # 393 px
    assert classify_scene(samples, 0, 10, src_h=360).kind != "STACKED"  # 131 px


def test_face_box_is_median_face_y_h():
    face = Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9)
    s = classify_scene([(t, [face]) for t in range(10)], 0, 10)
    assert s.kind == "FACE_TRACK" and s.face_box == pytest.approx((0.2, 0.4))
    f1 = Face(x=0.1, y=0.3, w=0.15, h=0.3, score=0.9)
    f2 = Face(x=0.7, y=0.2, w=0.15, h=0.2, score=0.9)
    s = classify_scene([(t, [f1, f2]) for t in range(10)], 0, 10)
    assert s.kind == "SPLIT" and s.face_box == pytest.approx((0.2, 0.3))  # topmost y, max h
    assert classify_scene([(t, []) for t in range(10)], 0, 10).face_box is None


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
    assert len(det.frames) == 16  # four samples per second (the edge guard needs >= 4 fps)
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


def test_plan_layouts_failed_scene_falls_back_to_fit_blur(tmp_path, monkeypatch, caplog):
    from hookline.media.ffmpeg import FFmpegError
    from hookline.pipeline import layout

    video = tmp_path / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=25:duration=3",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=3",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    real = layout.sample_frames

    def flaky(v, times, *a, **k):
        if times[0] < 3.0:
            raise FFmpegError("no frame")
        return real(v, times, *a, **k)

    monkeypatch.setattr(layout, "sample_frames", flaky)
    face = Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9)
    with caplog.at_level("WARNING"):
        layouts = plan_layouts(video, _RecordingDetector([face]))
    assert [s.kind for s in layouts] == ["FIT_BLUR", "FACE_TRACK"]
    # never silent: the warning names the video, the scene and the fallback layout
    (msg,) = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "cut.mp4" in msg and "0.00-3.00" in msg and "FIT_BLUR" in msg and "no frame" in msg


@pytest.mark.integration
def test_plan_layouts_finds_facecam_in_synthetic_stream(synthetic_stream: Path):
    from hookline.assets import ensure_yunet
    from hookline.vision.faces import FaceDetector

    layouts = plan_layouts(synthetic_stream, FaceDetector(ensure_yunet()), sample_every=1.0)
    assert [s.kind for s in layouts] == ["STACKED"]
    x, y, w, _ = layouts[0].facecam
    assert x > 0.5 and y < 0.2 and x + w <= 1.0


def test_plan_layouts_without_detector_is_fit_blur(fixture_video):
    layouts = plan_layouts(fixture_video, None)
    assert [lay.kind for lay in layouts] == ["FIT_BLUR"]
    assert layouts[0].start == 0 and layouts[0].end == pytest.approx(150, abs=0.1)


# --- tracking holds through face gaps; head top; zoom plan fields ---------------------------------


def _face_at(cx: float, y: float = 0.2, h: float = 0.4, score: float = 0.9) -> Face:
    return Face(x=cx - 0.1, y=y, w=0.2, h=h, score=score)


def test_track_holds_last_good_x_through_a_short_face_gap():
    # face at 0.3 until 2.0 s, gone for 1.5 s (only a low-score "body" at 0.7), back at 0.6
    samples = []
    for i in range(13):
        t = i * 0.5
        if t <= 2.0:
            faces = [_face_at(0.3)]
        elif t < 4.0:
            faces = [_face_at(0.7, score=0.4)]
        else:
            faces = [_face_at(0.6)]
        samples.append((t, faces))
    s = classify_scene(samples, 0, 6.5)
    assert s.kind == "FACE_TRACK"
    track = dict(s.track_x)
    for t in (2.5, 3.0, 3.5):  # held, not re-centred and not following the body
        assert track[t] == pytest.approx(0.3)
    assert track[4.0] == pytest.approx(0.6)


def test_track_long_face_gap_eases_to_median_x():
    # face at 0.3 for 0-2 s, gone 3-12 s, at 0.9 from 13 s: median of the valid x is 0.9
    samples = [(float(t), [_face_at(0.3)] if t < 3 else []) for t in range(13)]
    samples += [(float(t), [_face_at(0.9)]) for t in range(13, 20)]
    s = classify_scene(samples, 0, 20)
    track = dict(s.track_x)
    assert track[3.0] == pytest.approx(0.3) and track[5.0] == pytest.approx(0.3)  # <= 3 s: hold
    assert track[6.0] == pytest.approx(0.9)  # 4 s without a face: eased to the median x
    assert track[12.0] == pytest.approx(0.9)


def test_track_leading_gap_adds_no_keyframes():
    samples = [(float(t), [] if t < 3 else [_face_at(0.4)]) for t in range(8)]
    s = classify_scene(samples, 0, 8)
    assert [t for t, _ in s.track_x] == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_face_track_records_face_width_and_highest_head_top():
    samples = [(float(t), [_face_at(0.5, y=0.3, h=0.4)]) for t in range(9)]
    samples.append((9.0, [_face_at(0.5, y=0.1, h=0.4)]))  # one high sample of ten
    s = classify_scene(samples, 0, 10)
    assert s.face_w == pytest.approx(0.2)
    assert s.head_top == pytest.approx(0.1 - 0.2)  # 10% quantile of ten = the highest one


def test_zoom_level_maps_steps_through_base_and_cap():
    lay = SceneLayout(start=0, end=1, kind="FACE_TRACK")
    assert [lay.zoom_level(s) for s in (1.0, 1.12, 1.2)] == pytest.approx([1.0, 1.12, 1.2])
    capped = lay.model_copy(update={"zoom_cap": 1.12})
    levels = [capped.zoom_level(s) for s in (1.0, 1.12, 1.2)]
    assert levels[0] == 1.0 and levels[2] == pytest.approx(1.12) and 1.0 < levels[1] < 1.12
    based = lay.model_copy(update={"base_zoom": 1.25, "zoom_cap": 1.4})
    assert based.zoom_level(1.0) == pytest.approx(1.25) and based.zoom_level(1.2) == pytest.approx(
        1.4
    )


# --- per-sample face widths, gaze lead ------------------------------------------------------------


def test_track_records_face_width_per_keyframe():
    samples = [
        (t * 0.25, [Face(x=0.4, y=0.2, w=0.1 + 0.01 * t, h=0.4, score=0.9)]) for t in range(6)
    ]
    samples.insert(3, (0.6, []))  # a gap holds the last width too
    s = classify_scene(samples, 0, 2)
    assert len(s.track_w) == len(s.track_x)
    assert s.track_w == pytest.approx([0.1, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15])


@pytest.mark.parametrize(
    ("yaws", "lead"),
    [
        ([-0.4] * 10, -0.08),  # looking frame-left: room on the left
        ([0.4] * 10, 0.08),
        ([0.03] * 10, 0.0),  # facing the camera: centred
        ([None] * 10, 0.0),  # no landmarks
        ([-0.4] * 9 + [0.9], -0.08),  # one stray estimate does not flip it
    ],
)
def test_face_track_gaze_lead_from_median_yaw(yaws, lead):
    samples = [
        (t * 0.25, [Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9, yaw=y)]) for t, y in enumerate(yaws)
    ]
    s = classify_scene(samples, 0, 3)
    assert s.lead == pytest.approx(lead)


# --- per-sample face rows (caption placement over time) -------------------------------------------


def test_face_track_records_each_samples_face_row_and_eye_line():
    samples = [
        (t * 0.25, [Face(x=0.4, y=0.2 + 0.02 * t, w=0.2, h=0.4, score=0.9, eye_y=0.35 + 0.02 * t)])
        for t in range(6)
    ]
    samples.insert(3, (0.6, []))  # a lost face holds the last row
    s = classify_scene(samples, 0.0, 2.0)
    assert len(s.track_face) == len(s.track_x)
    assert s.track_face[0] == pytest.approx((0.2, 0.4, 0.35))
    assert s.track_face[3] == pytest.approx(s.track_face[2])
    assert Face(x=0, y=0, w=1, h=1, score=1).eye_y is None
