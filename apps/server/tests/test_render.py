import json
import os
import subprocess
from itertools import pairwise
from pathlib import Path
from unittest import mock

import pytest

from hookline.config import Settings
from hookline.media import ffmpeg as ffmpeg_mod
from hookline.media.ffmpeg import extract_audio, probe_duration, probe_size
from hookline.models import ClipPick, RubricScores, Transcript, Word
from hookline.pipeline.render import cut_times_for, emphasis_times, render_clip
from hookline.pipeline.run import _face_detector
from tests.conftest import NoFaces
from tests.test_pipeline import FakeTranscriber


def _graph(cmd: list[str]) -> str:
    if "-/filter_complex" in cmd:
        return Path(cmd[cmd.index("-/filter_complex") + 1]).read_text(encoding="utf-8")
    return cmd[cmd.index("-filter_complex") + 1]


@pytest.fixture(scope="module")
def fixture_wav(fixture_video, tmp_path_factory):
    return extract_audio(fixture_video, tmp_path_factory.mktemp("wav") / "audio.wav")


@pytest.fixture(scope="module")
def fixture_words(fixture_wav) -> list[Word]:
    return FakeTranscriber(duration=150).transcribe(fixture_wav).words()


def test_render_clip_vertical_with_captions(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    final_argvs,
):
    dst = tmp_path / "out" / "01-sample-clip.mp4"
    extras = render_clip(
        fixture_video,
        fixture_wav,
        sample_pick,
        fixture_words,
        tmp_path / "render",
        dst,
        settings,
        NoFaces(),
    )
    assert dst.exists() and probe_size(dst) == (1080, 1920)
    assert extras["layout"][0]["kind"] == "FIT_BLUR"
    kept = sum(e - s for s, e in extras["keep_ranges"])
    assert probe_duration(dst) == pytest.approx(kept, abs=0.2)
    assert extras["removed_s"] >= 0
    assert (tmp_path / "render" / "01.ass").exists()
    assert not (tmp_path / "render" / "01.mkv").exists()  # intermediate removed after success
    (final,) = final_argvs
    assert "subtitles=" in _graph(final)
    i = final.index("-c:a")
    assert final[i : i + 6] == ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    af = final[final.index("-af") + 1]
    assert af.startswith("acompressor=") and ",loudnorm=I=-14:TP=-2:" in af
    json.dumps(extras)  # sidecar-serialisable


def test_render_clip_horizontal_no_captions(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    final_argvs,
):
    s = settings.model_copy(update={"vertical": False, "captions": False})
    dst = tmp_path / "02-h.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, s, None
    )
    w, h = probe_size(dst)
    assert w / h == pytest.approx(320 / 180, abs=0.02)
    assert extras["layout"] == []
    assert not (tmp_path / "r" / "02.ass").exists()
    assert "subtitles=" not in _graph(final_argvs[0])


def test_render_clip_silence_window_uses_original_segments(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    from hookline.pipeline import render as render_mod

    seen: list[tuple[float, float]] = []
    real = render_mod.detect_silences

    def spy(wav, start, end, *a, **k):
        seen.append((start, end))
        return real(wav, start, end, *a, **k)

    monkeypatch.setattr(render_mod, "detect_silences", spy)
    pick = sample_pick.model_copy(update={"cold_open": (60.0, 63.0)})
    s = settings.model_copy(update={"vertical": False, "captions": False})
    render_clip(
        fixture_video, fixture_wav, pick, fixture_words, tmp_path, tmp_path / "03-x.mp4", s, None
    )
    assert seen == [(29.0, 64.0)]


@pytest.mark.parametrize("protect", [True, False])
def test_render_clip_word_protection_flag(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch, protect
):
    from hookline.pipeline import render as render_mod

    seen = []
    real = render_mod.keep_ranges

    def spy(*a, **k):
        seen.append(k.get("words"))
        return real(*a, **k)

    monkeypatch.setattr(render_mod, "keep_ranges", spy)
    s = settings.model_copy(update={"vertical": False, "captions": False})
    render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, tmp_path / "04-p.mp4",
        s, None, protect_words=protect,
    )  # fmt: skip
    assert (seen[0] is not None and len(seen[0]) > 0) if protect else seen[0] is None


def test_cut_times_are_output_time_keep_boundaries():
    keep = [(10.0, 12.0), (13.0, 16.5), (20.0, 21.0)]
    assert cut_times_for(keep) == pytest.approx([2.0, 5.5])
    assert cut_times_for([(0.0, 5.0)]) == []


def test_emphasis_times_map_zoom_beats_through_keep_ranges():
    keep = [(10.0, 12.0), (13.0, 16.5)]
    beats = [
        "zoom: 'knives' @ 11.5s",  # -> 1.5
        "zoom: 'gone' @ 12.5s",  # inside a removed range: dropped
        "zoom: 'late' @ 14.0s",  # -> 2 + 1 = 3.0
        "cut: 'x' @ 15.0s",  # not a zoom beat
        "zoom: no time",
    ]
    assert emphasis_times(beats, keep) == pytest.approx([1.5, 3.0])


def _video_kbps(path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=bit_rate", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return int(res.stdout.strip()) / 1000


_ZOOM_PICK = ClipPick(
    start=30.0, end=55.0, cold_open=None, title="Zoom", hook_text="Wait for it",
    reason="test", scores=RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    ),
    edit_beats=["zoom: 'something' @ 40.0s"],
)  # fmt: skip


@pytest.fixture(scope="module")
def zoom_render(fixture_video, fixture_wav, fixture_words, tmp_path_factory):
    d = tmp_path_factory.mktemp("zoom")
    dst = d / "05-z.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, _ZOOM_PICK, fixture_words, d / "r", dst, Settings(), NoFaces()
    )
    return dst, extras


def test_render_clip_zoom_cadence(zoom_render):
    dst, extras = zoom_render
    kept = sum(e - s for s, e in extras["keep_ranges"])
    assert extras["cut_times"] == pytest.approx(cut_times_for(extras["keep_ranges"]))
    steps = extras["zoom_steps"]
    times = [t for t, _ in steps] + [kept]
    assert steps[0] == [0.0, 1.0] and max(b - a for a, b in pairwise(times)) <= 4.5 + 1e-6
    assert all(t < kept for t, _ in steps)
    assert any(s == 1.2 for _, s in steps)  # the emphasis punch
    assert max(s for _, s in steps) <= 1.2
    assert probe_size(dst) == (1080, 1920)
    json.dumps(extras)


_OLD_RATE_CONTROL = {  # pre-M2.5 values
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    "h264_amf": ["-rc", "cqp", "-qp_i", "22", "-qp_p", "24", "-qp_b", "26"],
    "h264_nvenc": ["-rc", "vbr", "-cq", "21", "-b:v", "0"],
}


def test_render_clip_bitrate_rises_vs_previous_settings(
    zoom_render, fixture_video, fixture_wav, fixture_words, tmp_path
):
    dst_new, _ = zoom_render
    dst_old = tmp_path / "06-old.mp4"
    with mock.patch.dict(ffmpeg_mod._ENCODER_QUALITY_ARGS, _OLD_RATE_CONTROL):
        render_clip(
            fixture_video, fixture_wav, _ZOOM_PICK, fixture_words, tmp_path / "r", dst_old,
            Settings(), NoFaces(),
        )  # fmt: skip
    assert _video_kbps(dst_new) > 1.1 * _video_kbps(dst_old)


def _real_job() -> Path | None:
    """A cached real podcast job (HOOKLINE_REAL_JOB, else the local workspace copy)."""
    default = Path(__file__).parents[1] / "workspace" / "e87398ed46"
    job = Path(os.environ.get("HOOKLINE_REAL_JOB", default))
    needed = ("source.mp4", "audio.wav", "transcript.json")
    return job if all((job / n).exists() for n in needed) else None


@pytest.mark.skipif(_real_job() is None, reason="no cached real footage in this environment")
def test_real_footage_bitrate_at_least_5mbps(tmp_path):
    job = _real_job()
    assert job is not None
    transcript = Transcript.model_validate_json(
        (job / "transcript.json").read_text(encoding="utf-8")
    )
    pick = _ZOOM_PICK.model_copy(update={"start": 1000.0, "end": 1015.0, "edit_beats": []})
    s = Settings()
    dst = tmp_path / "07-real.mp4"
    render_clip(
        job / "source.mp4", job / "audio.wav", pick, transcript.words(), tmp_path / "r", dst, s,
        _face_detector(s, print),
    )  # fmt: skip
    assert _video_kbps(dst) >= 5000
