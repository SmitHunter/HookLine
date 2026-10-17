import json

import pytest

from hookline.media.ffmpeg import extract_audio, probe_duration, probe_size
from hookline.models import Word
from hookline.pipeline.render import render_clip
from tests.test_pipeline import FakeTranscriber


class NoFaces:
    def detect(self, bgr):
        return []


@pytest.fixture(scope="module")
def fixture_wav(fixture_video, tmp_path_factory):
    return extract_audio(fixture_video, tmp_path_factory.mktemp("wav") / "audio.wav")


@pytest.fixture(scope="module")
def fixture_words(fixture_wav) -> list[Word]:
    return FakeTranscriber(duration=150).transcribe(fixture_wav).words()


def test_render_clip_vertical_with_captions(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path
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
    assert (tmp_path / "render" / "01.mp4").exists() and (tmp_path / "render" / "01.ass").exists()
    json.dumps(extras)  # sidecar-serialisable


def test_render_clip_horizontal_no_captions(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path
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
