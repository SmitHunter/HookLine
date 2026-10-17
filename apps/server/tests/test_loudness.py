import pytest

from hookline.media.ffmpeg import _run
from hookline.media.loudness import Loudness, loudnorm_filter, measure_loudness
from hookline.pipeline.render import render_clip
from tests.test_render import NoFaces, _final_argvs


def _m(i: float) -> Loudness:
    return Loudness(i=i, tp=-3.0, lra=7.0, thresh=i - 10, offset=0.0)


def test_measure_loudness_reads_fixture(quiet_speech_video):
    m = measure_loudness(quiet_speech_video)
    assert m.i == pytest.approx(-30.0, abs=1.0)
    assert m.lra > 1


def test_loudnorm_filter_targets_minus_14():
    f = loudnorm_filter(_m(-30.0))
    assert f.startswith("acompressor=threshold=-21dB:ratio=3:attack=5:release=120:makeup=2,")
    for frag in ("I=-14", "TP=-2", "LRA=11", "measured_I=-30", "linear=true", "print_format=json"):
        assert frag in f


def test_loudnorm_filter_caps_gain_at_20db():
    f = loudnorm_filter(_m(-45.0))
    assert "loudnorm=I=-25" in f  # -45 + 20, not -14


def test_loudnorm_filter_skips_below_minus_50():
    assert loudnorm_filter(_m(-60.0)) == "anull"


def _render(video, tmp_path, sample_pick, settings, monkeypatch):
    from hookline.media.ffmpeg import extract_audio
    from hookline.models import Word

    wav = extract_audio(video, tmp_path / "a.wav")
    words = [Word(start=float(t), end=t + 0.5, text="hi") for t in range(20)]
    pick = sample_pick.model_copy(update={"start": 2.0, "end": 16.0})
    s = settings.model_copy(update={"captions": False})
    argvs = _final_argvs(monkeypatch)
    dst = tmp_path / "out" / "01-x.mp4"
    render_clip(video, wav, pick, words, tmp_path / "r", dst, s, NoFaces())
    return dst, argvs


def test_quiet_clip_renders_to_minus_14(
    quiet_speech_video, sample_pick, settings, tmp_path, monkeypatch
):
    dst, argvs = _render(quiet_speech_video, tmp_path, sample_pick, settings, monkeypatch)
    m = measure_loudness(dst)
    print(f"quiet -> I={m.i} TP={m.tp} LRA={m.lra}")
    assert m.i == pytest.approx(-14.0, abs=1.0)
    assert m.tp <= -1.0
    (final,) = argvs
    assert "-af" in final and "loudnorm=" in final[final.index("-af") + 1]
    i = final.index("-c:a")
    assert final[i : i + 6] == ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    streams = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,bit_rate",
            "-of",
            "json",
            str(dst),
        ]
    ).stdout
    assert '"aac"' in streams and '"48000"' in streams


def test_near_silent_clip_not_boosted(
    near_silent_video, sample_pick, settings, tmp_path, monkeypatch
):
    dst, _ = _render(near_silent_video, tmp_path, sample_pick, settings, monkeypatch)
    m = measure_loudness(dst)
    print(f"silent -> I={m.i} TP={m.tp}")
    assert m.i <= -40.0


def test_intermediate_is_mkv_pcm(quiet_speech_video, sample_pick, settings, tmp_path, monkeypatch):
    from hookline.pipeline import render as render_mod

    seen = []
    real = render_mod.cut_segments

    def spy(src, segs, dst, *a, **k):
        out = real(src, segs, dst, *a, **k)
        seen.append(out)
        probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(out),
            ]
        ).stdout.strip()
        seen.append(probe)
        return out

    monkeypatch.setattr(render_mod, "cut_segments", spy)
    _render(quiet_speech_video, tmp_path, sample_pick, settings, monkeypatch)
    assert seen[0].suffix == ".mkv" and seen[1] == "pcm_s16le"
    assert not seen[0].exists()  # removed after success
