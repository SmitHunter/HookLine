from pathlib import Path

import pytest

from hookline.media import ffmpeg as ffmpeg_mod
from hookline.media.ffmpeg import (
    FFmpegError,
    cut_segments,
    encoder_args,
    extract_audio,
    filter_graph_args,
    pick_encoder,
    probe_duration,
    probe_fps,
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
    out = cut_segments(fixture_video, [(100.0, 102.0), (10.0, 20.0)], tmp_path / "c.mkv", "libx264")
    assert probe_duration(out) == pytest.approx(12.0, abs=0.15)


def test_cut_segments_raises_ffmpeg_error_on_bad_input(tmp_path):
    with pytest.raises(FFmpegError):
        cut_segments(tmp_path / "missing.mp4", [(0, 1)], tmp_path / "x.mkv", "libx264")


def test_encoder_args_table(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "amf_b_frames", lambda: False)
    assert encoder_args("libx264") == [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-maxrate", "10M", "-bufsize", "20M",
    ]  # fmt: skip
    assert encoder_args("h264_amf") == [
        "-c:v", "h264_amf", "-rc", "cqp", "-qp_i", "16", "-qp_p", "18", "-qp_b", "20",
    ]  # fmt: skip
    assert encoder_args("h264_nvenc") == [
        "-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "18", "-b:v", "0",
    ]  # fmt: skip
    assert encoder_args("unknown_enc") == ["-c:v", "unknown_enc"]


def test_cut_segments_passes_encoder_args(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def fake_run(cmd):
        seen.append(cmd)
        (tmp_path / "o.tmp.mkv").write_bytes(b"x")

    monkeypatch.setattr(ffmpeg_mod, "_run", fake_run)
    monkeypatch.setattr(ffmpeg_mod, "has_audio", lambda p: True)
    monkeypatch.setattr(ffmpeg_mod, "amf_b_frames", lambda: True)
    cut_segments(tmp_path / "in.mp4", [(0.0, 1.0)], tmp_path / "o.mkv", "h264_amf", fps=25)
    cmd = seen[0]
    i = cmd.index("-c:v")
    assert cmd[i : i + len(encoder_args("h264_amf"))] == encoder_args("h264_amf")


def test_libx264_cut_has_sane_bitrate(fixture_video, tmp_path):
    out = cut_segments(fixture_video, [(10.0, 50.0)], tmp_path / "c.mkv", "libx264")
    bits = out.stat().st_size * 8 / probe_duration(out)
    assert bits < 2_000_000


def test_amf_cut_has_sane_bitrate_when_available(fixture_video, tmp_path):
    if pick_encoder() != "h264_amf":
        pytest.skip("h264_amf not usable here")
    out = cut_segments(fixture_video, [(10.0, 50.0)], tmp_path / "c.mkv", "h264_amf")
    assert out.stat().st_size * 8 / probe_duration(out) < 2_000_000


def test_cut_segments_quality_replaces_rate_control(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def fake_run(cmd):
        seen.append(cmd)
        (tmp_path / "o.tmp.mkv").write_bytes(b"x")

    monkeypatch.setattr(ffmpeg_mod, "_run", fake_run)
    monkeypatch.setattr(ffmpeg_mod, "has_audio", lambda p: True)
    q = ["-preset", "veryfast", "-crf", "12"]
    cut_segments(
        tmp_path / "in.mp4", [(0.0, 1.0)], tmp_path / "o.mkv", "libx264", quality=q, fps=25
    )
    cmd = seen[0]
    i = cmd.index("-c:v")
    assert cmd[i : i + 6] == ["-c:v", "libx264", *q]
    assert cmd.count("-crf") == 1


def test_probe_fps(fixture_video):
    assert probe_fps(fixture_video) == pytest.approx(25.0)


def _frame(video, t):
    from hookline.vision.faces import sample_frames

    return sample_frames(video, [t], width=320)[0].astype("int32")


def test_cut_segments_exact_with_many_short_ranges(fixture_video, tmp_path):
    import numpy as np

    from hookline.models import Word
    from hookline.pipeline.deadair import remap_words, snap_ranges

    fps = 25.0
    raw = [(10.013 + 1.37 * i, 10.013 + 1.37 * i + 0.41 + 0.07 * i) for i in range(12)]
    keep = snap_ranges(raw, fps)
    assert len(keep) == 12 and all(round(t * fps, 6) == round(t * fps) for r in keep for t in r)
    out = cut_segments(fixture_video, keep, tmp_path / "c.mkv", "libx264")
    expected = sum(e - s for s, e in keep)
    assert probe_duration(out) == pytest.approx(expected, abs=1 / fps)

    k_start, _ = keep[7]
    word = Word(start=k_start + 4 / fps, end=k_start + 8 / fps, text="x")
    (mapped,) = remap_words([word], keep)
    assert mapped.start == pytest.approx(sum(e - s for s, e in keep[:7]) + 4 / fps, abs=1e-6)
    eps = 0.25 / fps
    got = _frame(out, mapped.start + eps)
    diffs = {
        k: np.abs(got - _frame(fixture_video, word.start + k / fps + eps)).mean()
        for k in (-1, 0, 1)
    }
    assert diffs[0] < min(diffs[-1], diffs[1]), diffs


def test_filter_graph_args_inline_when_short(tmp_path):
    script = tmp_path / "g.txt"
    assert filter_graph_args("[0:v]null[v]", script) == ["-filter_complex", "[0:v]null[v]"]
    assert not script.exists()


def test_filter_graph_args_script_when_long_or_forced(tmp_path):
    script = tmp_path / "g.txt"
    forced = filter_graph_args("[0:v]null[v]", script, force=True)
    assert forced == ["-/filter_complex", str(script)]
    assert script.read_text(encoding="utf-8") == "[0:v]null[v]"
    long = "[0:v]null" + ",null" * 5000 + "[v]"
    assert filter_graph_args(long, script)[0] == "-/filter_complex"
    assert script.read_text(encoding="utf-8") == long


def test_synthetic_40k_char_graph_renders(tmp_path):
    # ~40K chars: past the Windows command-line limit if it were passed inline
    graph = "[0:v]null" + ",scale=w='64+0*(iw-1)':h=64" * 1600 + "[v]"
    assert len(graph) > 40_000
    out = tmp_path / "o.mp4"
    ffmpeg_mod._run(
        ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:d=1",
         *filter_graph_args(graph, tmp_path / "g.txt"), "-map", "[v]", str(out)]
    )  # fmt: skip
    assert ffprobe_stream(out)["width"] == 64


def test_cut_segments_long_graph_goes_through_script(monkeypatch, tmp_path):
    seen: list[tuple[list[str], str]] = []

    def fake_run(cmd):
        i = cmd.index("-/filter_complex")
        seen.append((cmd, Path(cmd[i + 1]).read_text(encoding="utf-8")))
        (tmp_path / "o.tmp.mkv").write_bytes(b"x")

    monkeypatch.setattr(ffmpeg_mod, "_run", fake_run)
    monkeypatch.setattr(ffmpeg_mod, "has_audio", lambda p: True)
    segs = [(i * 2.0, i * 2.0 + 1.0) for i in range(200)]
    cut_segments(tmp_path / "in.mp4", segs, tmp_path / "o.mkv", "libx264", fps=25)
    ((cmd, graph),) = seen
    assert "-filter_complex" not in cmd and "concat=n=200" in graph
    assert not list(tmp_path.glob("*.graph.txt"))  # script removed after the cut


# --- AMF QP 16/18/20 with B-frames when the encoder takes them ------------------------------------


def test_amf_args_add_b_frames_only_when_supported(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "amf_b_frames", lambda: True)
    assert encoder_args("h264_amf") == [
        "-c:v", "h264_amf", "-rc", "cqp", "-qp_i", "16", "-qp_p", "18", "-qp_b", "20", "-bf", "2",
    ]  # fmt: skip
    monkeypatch.setattr(ffmpeg_mod, "amf_b_frames", lambda: False)
    assert encoder_args("h264_amf") == [
        "-c:v", "h264_amf", "-rc", "cqp", "-qp_i", "16", "-qp_p", "18", "-qp_b", "20",
    ]  # fmt: skip


def test_amf_b_frame_probe_falls_back_cleanly(monkeypatch):
    def boom(cmd):
        raise ffmpeg_mod.FFmpegError("h264_amf exited with code 1")

    ffmpeg_mod.amf_b_frames.cache_clear()
    monkeypatch.setattr(ffmpeg_mod, "_run", boom)
    try:
        assert ffmpeg_mod.amf_b_frames() is False
    finally:
        ffmpeg_mod.amf_b_frames.cache_clear()


def test_amf_b_frame_probe_reads_the_encoded_stream(monkeypatch):
    class Res:
        def __init__(self, out=""):
            self.stdout = out

    def fake(cmd):
        return Res("2\n") if cmd[0] == "ffprobe" else Res()

    ffmpeg_mod.amf_b_frames.cache_clear()
    monkeypatch.setattr(ffmpeg_mod, "_run", fake)
    try:
        assert ffmpeg_mod.amf_b_frames() is True
    finally:
        ffmpeg_mod.amf_b_frames.cache_clear()


def test_probe_video_kbps(fixture_video, tmp_path):
    kbps = ffmpeg_mod.probe_video_kbps(fixture_video)
    assert kbps is not None and kbps > 0
    assert ffmpeg_mod.probe_video_kbps(tmp_path / "missing.mp4") is None


def test_amf_b_frame_probe_does_not_cache_a_transient_failure(monkeypatch):
    class Res:
        def __init__(self, out=""):
            self.stdout = out

    calls = []

    def flaky(cmd):
        calls.append(cmd[0])
        if len(calls) == 1:
            raise ffmpeg_mod.FFmpegError("h264_amf exited with code 1 (device busy)")
        return Res("2\n") if cmd[0] == "ffprobe" else Res()

    ffmpeg_mod.amf_b_frames.cache_clear()
    monkeypatch.setattr(ffmpeg_mod, "_run", flaky)
    try:
        assert ffmpeg_mod.amf_b_frames() is False  # this time
        assert ffmpeg_mod.amf_b_frames() is True  # re-probed, and the success is kept
        n = len(calls)
        assert ffmpeg_mod.amf_b_frames() is True and len(calls) == n
    finally:
        ffmpeg_mod.amf_b_frames.cache_clear()


def test_amf_b_frame_probe_caches_a_definitive_no(monkeypatch):
    class Res:
        def __init__(self, out=""):
            self.stdout = out

    calls = []

    def no_b(cmd):
        calls.append(cmd[0])
        return Res("0\n") if cmd[0] == "ffprobe" else Res()

    ffmpeg_mod.amf_b_frames.cache_clear()
    monkeypatch.setattr(ffmpeg_mod, "_run", no_b)
    try:
        assert ffmpeg_mod.amf_b_frames() is False and ffmpeg_mod.amf_b_frames() is False
        assert len(calls) == 2  # probed once
    finally:
        ffmpeg_mod.amf_b_frames.cache_clear()
