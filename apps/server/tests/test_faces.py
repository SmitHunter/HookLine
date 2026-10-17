from pathlib import Path

import numpy as np
import pytest

from hookline import assets
from hookline.media.ffmpeg import FFmpegError
from hookline.vision.faces import Face, FaceDetector, face_yaw, sample_frames


def test_face_is_normalized_record():
    f = Face(x=0.1, y=0.2, w=0.3, h=0.4, score=0.9)
    assert (f.x, f.y, f.w, f.h, f.score) == (0.1, 0.2, 0.3, 0.4, 0.9)


def test_sample_frames_returns_bgr_at_requested_width(fixture_video):
    frames = sample_frames(fixture_video, [1.0, 2.5])
    assert len(frames) == 2
    assert all(f.shape == (360, 640, 3) and f.dtype == np.uint8 for f in frames)
    assert not np.array_equal(frames[0], frames[1])  # testsrc2 moves


def test_sample_frames_custom_width(fixture_video):
    (f,) = sample_frames(fixture_video, [0.0], width=320)
    assert f.shape == (180, 320, 3)


def test_sample_frames_crops_before_scaling(fixture_video):
    (f,) = sample_frames(fixture_video, [0.0], width=200, crop=(160, 160, 80, 10))
    assert f.shape == (200, 200, 3)


def test_sample_frames_past_end_raises(fixture_video):
    with pytest.raises(FFmpegError):
        sample_frames(fixture_video, [10_000.0])


def test_ensure_yunet_fetches_into_tools_models(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(assets, "tools_dir", lambda: tmp_path)
    monkeypatch.setattr(assets, "fetch", lambda url, dst: calls.append((url, dst)) or Path(dst))
    p = assets.ensure_yunet()
    assert p == tmp_path / "models" / "face_detection_yunet_2023mar.onnx"
    url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    )
    assert calls == [(url, p)]


@pytest.mark.integration
def test_detector_finds_face_in_synthetic_stream_frame(synthetic_stream):
    (frame,) = sample_frames(synthetic_stream, [1.0])
    faces = FaceDetector(assets.ensure_yunet()).detect(frame)
    assert faces and faces[0].x > 0.6 and faces[0].y < 0.4
    assert all(0 <= v <= 1 for v in (faces[0].x, faces[0].y, faces[0].w, faces[0].h))
    assert faces[0].score >= 0.6


@pytest.mark.integration
def test_detector_finds_no_face_in_test_pattern(fixture_video):
    (frame,) = sample_frames(fixture_video, [1.0])
    assert FaceDetector(assets.ensure_yunet()).detect(frame) == []


@pytest.mark.integration
def test_detector_handles_changing_frame_sizes(synthetic_stream):
    det = FaceDetector(assets.ensure_yunet())
    (big,) = sample_frames(synthetic_stream, [1.0], width=960)
    (small,) = sample_frames(synthetic_stream, [1.0], width=640)
    a, b = det.detect(big), det.detect(small)
    assert a and b and abs(a[0].x - b[0].x) < 0.02


# --- wave D: yaw from landmarks, batched sampling ---------------------------------------


def test_face_yaw_from_eyes_and_nose():
    assert face_yaw((100, 50), (140, 50), (120, 70)) == pytest.approx(0.0)
    assert face_yaw((100, 50), (140, 50), (130, 70)) == pytest.approx(0.25)  # nose right
    assert face_yaw((100, 50), (140, 50), (106, 70)) == pytest.approx(-0.35)  # nose left
    assert face_yaw((100, 50), (100.5, 50), (120, 70)) is None  # eyes on top of each other


def test_face_defaults_to_no_yaw():
    assert Face(x=0.1, y=0.2, w=0.3, h=0.4, score=0.9).yaw is None


def test_sample_frames_batch_matches_single_seeks(fixture_video):
    times = [0.25 * k + 0.125 for k in range(20)]
    batch = sample_frames(fixture_video, times, width=160)
    assert len(batch) == len(times)
    for t, frame in zip(times[::6], batch[::6], strict=True):
        (single,) = sample_frames(fixture_video, [t], width=160)
        assert np.abs(frame.astype(int) - single.astype(int)).mean() < 2.0


def test_sample_frames_batch_past_end_raises(fixture_video):
    with pytest.raises(FFmpegError):
        sample_frames(fixture_video, [0.5 * k for k in range(8)] + [10_000.0])


def test_sample_frames_many_times_in_one_scene(fixture_video):
    # 4 fps over 40 s: one select sum of 160 terms overflows ffmpeg's expression parser
    times = [0.25 * k + 0.125 for k in range(160)]
    frames = sample_frames(fixture_video, times, width=64)
    assert len(frames) == 160 and all(f.shape == (36, 64, 3) for f in frames)


@pytest.mark.integration
def test_detector_reports_yaw_from_landmarks(synthetic_stream):
    (frame,) = sample_frames(synthetic_stream, [1.0])
    faces = FaceDetector(assets.ensure_yunet()).detect(frame)
    assert faces and faces[0].yaw is not None and -1.0 < faces[0].yaw < 1.0
