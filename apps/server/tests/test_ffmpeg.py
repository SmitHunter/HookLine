import pytest

from hookline.media import ffmpeg as ffmpeg_mod
from hookline.media.ffmpeg import (
    FFmpegError,
    cut_segments,
    encoder_args,
    extract_audio,
    pick_encoder,
    probe_duration,
)
from tests.conftest import ffprobe_stream


def test_probe_duration(fixture_video):
    assert probe_duration(fixture_video) == pytest.approx(150.0, abs=0.1)


def test_extract_audio_is_16k_mono(fixture_video, tmp_path):
    wav = extract_audio(fixture_video, tmp_path / "a.wav")
    info = ffprobe_stream(wav)
    assert info["sample_rate"] == "16000" and info["channels"] == 1


def test_pick_encoder_returns_known():
    assert pick_encoder() in {"h264_nvenc", "h264_amf", "libx264"}


def test_cut_segments_concatenates(fixture_video, tmp_path):
    out = cut_segments(fixture_video, [(100.0, 102.0), (10.0, 20.0)], tmp_path / "c.mp4", "libx264")
    assert probe_duration(out) == pytest.approx(12.0, abs=0.15)


def test_cut_segments_raises_ffmpeg_error_on_bad_input(tmp_path):
    with pytest.raises(FFmpegError):
        cut_segments(tmp_path / "missing.mp4", [(0, 1)], tmp_path / "x.mp4", "libx264")


def test_encoder_args_table():
    assert encoder_args("libx264") == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    assert encoder_args("h264_amf") == [
        "-c:v",
        "h264_amf",
        "-rc",
        "cqp",
        "-qp_i",
        "22",
        "-qp_p",
        "24",
        "-qp_b",
        "26",
    ]
    assert encoder_args("h264_nvenc") == [
        "-c:v",
        "h264_nvenc",
        "-rc",
        "vbr",
        "-cq",
        "21",
        "-b:v",
        "0",
    ]
    assert encoder_args("unknown_enc") == ["-c:v", "unknown_enc"]


def test_cut_segments_passes_encoder_args(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def fake_run(cmd):
        seen.append(cmd)
        (tmp_path / "o.tmp.mp4").write_bytes(b"x")

    monkeypatch.setattr(ffmpeg_mod, "_run", fake_run)
    cut_segments(tmp_path / "in.mp4", [(0.0, 1.0)], tmp_path / "o.mp4", "h264_amf")
    cmd = seen[0]
    i = cmd.index("-c:v")
    assert cmd[i : i + len(encoder_args("h264_amf"))] == encoder_args("h264_amf")


def test_libx264_cut_has_sane_bitrate(fixture_video, tmp_path):
    out = cut_segments(fixture_video, [(10.0, 50.0)], tmp_path / "c.mp4", "libx264")
    bits = out.stat().st_size * 8 / probe_duration(out)
    assert bits < 2_000_000


def test_amf_cut_has_sane_bitrate_when_available(fixture_video, tmp_path):
    if pick_encoder() != "h264_amf":
        pytest.skip("h264_amf not usable here")
    out = cut_segments(fixture_video, [(10.0, 50.0)], tmp_path / "c.mp4", "h264_amf")
    assert out.stat().st_size * 8 / probe_duration(out) < 2_000_000


def test_cut_segments_quality_replaces_rate_control(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def fake_run(cmd):
        seen.append(cmd)
        (tmp_path / "o.tmp.mp4").write_bytes(b"x")

    monkeypatch.setattr(ffmpeg_mod, "_run", fake_run)
    q = ["-preset", "veryfast", "-crf", "12"]
    cut_segments(tmp_path / "in.mp4", [(0.0, 1.0)], tmp_path / "o.mp4", "libx264", quality=q)
    cmd = seen[0]
    i = cmd.index("-c:v")
    assert cmd[i : i + 6] == ["-c:v", "libx264", *q]
    assert cmd.count("-crf") == 1
