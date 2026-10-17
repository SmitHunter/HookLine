import pytest

from hookline.media.ffmpeg import _run
from hookline.media.loudness import (
    COMPRESSOR,
    Loudness,
    ensure_true_peak,
    loudnorm_filter,
    loudnorm_mode,
    measure_loudness,
    parse_loudnorm_json,
)
from hookline.pipeline.render import render_clip
from tests.conftest import NoFaces


def _m(i: float, tp: float = -20.0, lra: float = 7.0) -> Loudness:
    return Loudness(i=i, tp=tp, lra=lra, thresh=i - 10, offset=0.0)


def test_measure_loudness_reads_fixture(quiet_speech_video):
    m = measure_loudness(quiet_speech_video)
    assert m.i == pytest.approx(-30.0, abs=1.0)
    assert m.lra > 1


def test_loudnorm_filter_targets_minus_14():
    f = loudnorm_filter(_m(-30.0))
    assert f.startswith(f"{COMPRESSOR},")
    assert "makeup=1" in COMPRESSOR
    for frag in ("I=-14", "TP=-2", "LRA=11", "measured_I=-30", "linear=true", "print_format=json"):
        assert frag in f


def test_loudnorm_mode_rules():
    assert loudnorm_mode(_m(-30.0, tp=-20.0, lra=7.0)) == "linear"
    assert loudnorm_mode(_m(-30.0, tp=-5.0, lra=7.0)) == "dynamic"  # -5 + 16 > -2
    assert loudnorm_mode(_m(-30.0, tp=-20.0, lra=15.0)) == "dynamic"  # LRA above target
    assert loudnorm_mode(_m(-30.0, tp=-20.0, lra=0.0)) == "dynamic"  # loudnorm rejects LRA 0
    assert loudnorm_mode(_m(-60.0)) == "skip"
    f = loudnorm_filter(_m(-30.0, lra=15.0))
    assert "linear=true" not in f and "measured_I" not in f and "loudnorm=I=-14" in f


def test_measure_loudness_no_audio_returns_none(tmp_path):
    out = tmp_path / "v.mp4"
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i",
          "testsrc2=size=64x64:rate=25:duration=1", "-c:v", "libx264", str(out)])  # fmt: skip
    assert measure_loudness(out) is None


def test_loudnorm_filter_caps_gain_at_20db():
    f = loudnorm_filter(_m(-45.0))
    assert "loudnorm=I=-25" in f  # -45 + 20, not -14


def test_loudnorm_filter_skips_below_minus_50():
    assert loudnorm_filter(_m(-60.0)) == "anull"


def test_plan_loudness_regains_limiter_loss_and_keeps_gain_cap(monkeypatch, tmp_path):
    """The pre-gain aims the limited signal at the (capped) target and is re-aimed by the
    loudness the limiter took; loudnorm then targets the source's capped I, not -14."""
    from hookline.media import loudness as loud_mod

    calls = []

    def fake(path, pre=None):
        calls.append(pre)
        if pre == loud_mod.COMPRESSOR:
            return _m(-45.0, tp=-30.0)  # capped: target -25
        gain = float(pre.split("volume=")[1].split("dB")[0])
        lost = 1.5 if len(calls) == 2 else 0.4  # loudness the limiter takes
        return _m(-45.0 + gain - lost, tp=-3.0)  # peaks held at the limiter ceiling

    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    plan = loud_mod.plan_loudness(tmp_path / "x.mkv")
    assert [c.split(",")[1] for c in calls[1:]] == ["volume=20.00dB", "volume=21.50dB"]
    assert "alimiter=" in calls[1] and plan.af.startswith(calls[2])
    assert "loudnorm=I=-25:" in plan.af and plan.mode == "linear"


def _render(video, tmp_path, sample_pick, settings, final_argvs):
    from hookline.media.ffmpeg import extract_audio
    from hookline.models import Word

    wav = extract_audio(video, tmp_path / "a.wav")
    words = [Word(start=float(t), end=t + 0.5, text="hi") for t in range(20)]
    pick = sample_pick.model_copy(update={"start": 2.0, "end": 16.0})
    s = settings.model_copy(update={"captions": False})
    dst = tmp_path / "out" / "01-x.mp4"
    render_clip(video, wav, pick, words, tmp_path / "r", dst, s, NoFaces())
    return dst, final_argvs


def _check_render(video, tmp_path, sample_pick, settings, final_argvs, expected_mode):
    dst, argvs = _render(video, tmp_path, sample_pick, settings, final_argvs)
    m = measure_loudness(dst)
    print(f"{expected_mode}: I={m.i} TP={m.tp} LRA={m.lra}")
    assert m.i == pytest.approx(-14.0, abs=1.0)
    assert m.tp <= -1.0
    (final,) = argvs
    i = final.index("-c:a")
    assert final[i : i + 6] == ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    streams = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                    "stream=codec_name,sample_rate", "-of", "json", str(dst)]).stdout  # fmt: skip
    assert '"aac"' in streams and '"48000"' in streams
    # Pass 1 as the pipeline ran it, on the intermediate's audio, decides the mode.
    af = final[final.index("-af") + 1]
    assert ("linear=true" in af) == (expected_mode == "linear")
    reported = parse_loudnorm_json(argvs.stderrs[0])["normalization_type"]
    print(f"{expected_mode}: loudnorm reported normalization_type={reported}")
    assert reported == expected_mode


def test_low_crest_clip_is_linear_and_minus_14(
    steady_quiet_video, sample_pick, settings, tmp_path, final_argvs
):
    _check_render(steady_quiet_video, tmp_path, sample_pick, settings, final_argvs, "linear")


def test_speech_like_clip_is_dynamic_and_minus_14(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs
):
    _check_render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs, "dynamic")


def test_high_crest_clip_lands_on_target(
    high_crest_video, sample_pick, settings, tmp_path, final_argvs
):
    """Regression (e87398ed46 clip 04, -15.0 LUFS): when dynamic loudnorm's limiter has to
    take several dB off the peaks, its output undershoots the target. The render must check
    what the chain actually produces and correct for it."""
    src = measure_loudness(high_crest_video)
    print(f"source: I={src.i} TP={src.tp} crest={src.tp - src.i:.1f}")
    assert src.tp - src.i > 16  # the limiter must take > 4 dB off at -14 / -2 dBTP
    dst, _ = _render(high_crest_video, tmp_path, sample_pick, settings, final_argvs)
    m = measure_loudness(dst)
    print(f"high crest -> I={m.i} TP={m.tp}")
    assert m.i == pytest.approx(-14.0, abs=0.5)
    assert m.tp <= -1.0


def test_near_silent_clip_not_boosted(
    near_silent_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.pipeline import render as render_mod

    trims = []
    monkeypatch.setattr(render_mod, "ensure_true_peak", lambda p: trims.append(p))
    dst, _ = _render(near_silent_video, tmp_path, sample_pick, settings, final_argvs)
    assert trims == []  # near-silent audio is skipped, so no true-peak check either
    m = measure_loudness(dst)
    print(f"silent -> I={m.i} TP={m.tp}")
    assert m.i <= -40.0


def test_intermediate_is_mkv_pcm(
    quiet_speech_video, sample_pick, settings, tmp_path, monkeypatch, final_argvs
):
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
    _render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs)
    assert seen[0].suffix == ".mkv" and seen[1] == "pcm_s16le"
    assert not seen[0].exists()  # removed after success


def _hot_clip(src, dst, tp_db=-0.4):
    """Copy of `src` with audio gained so its true peak is about `tp_db`."""
    gain = tp_db - measure_loudness(src).tp
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(src), "-af",
          f"volume={gain:.2f}dB", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(dst)])  # fmt: skip
    return dst


def test_ensure_true_peak_trims_hot_file(steady_quiet_video, tmp_path):
    hot = _hot_clip(steady_quiet_video, tmp_path / "hot.mp4")
    before = measure_loudness(hot)
    assert before.tp > -1.2
    trim = ensure_true_peak(hot)
    after = measure_loudness(hot)
    print(f"trim {trim:.2f} dB: TP {before.tp} -> {after.tp}, I {before.i} -> {after.i}")
    assert trim < 0 and after.tp <= -1.0
    assert not hot.with_suffix(".trim.mp4").exists()
    streams = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                    "-of", "csv=p=0", str(hot)]).stdout  # fmt: skip
    assert "aac" in streams and "h264" in streams


def test_ensure_true_peak_leaves_safe_file_alone(steady_quiet_video, tmp_path, monkeypatch):
    from hookline.media import loudness as loud_mod

    calls = []
    monkeypatch.setattr(loud_mod, "_run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(loud_mod, "measure_loudness", lambda p: _m(-14.0, tp=-1.8))
    before = steady_quiet_video.read_bytes()
    assert ensure_true_peak(steady_quiet_video) is None
    assert calls == [] and steady_quiet_video.read_bytes() == before


def test_ensure_true_peak_runs_on_hot_measurement(steady_quiet_video, tmp_path, monkeypatch):
    from hookline.media import loudness as loud_mod

    copy = tmp_path / "c.mp4"
    copy.write_bytes(steady_quiet_video.read_bytes())
    tps = iter([-0.4, -1.9])
    monkeypatch.setattr(loud_mod, "measure_loudness", lambda p: _m(-14.0, tp=next(tps)))
    cmds = []
    real = loud_mod._run
    monkeypatch.setattr(loud_mod, "_run", lambda cmd: (cmds.append(cmd), real(cmd))[1])
    trim = ensure_true_peak(copy)
    assert trim == pytest.approx(-1.1)
    (cmd,) = cmds
    assert cmd[cmd.index("-c:v") + 1] == "copy" and "volume=-1.10dB" in cmd


def test_render_runs_true_peak_check(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.pipeline import render as render_mod

    checked = []
    monkeypatch.setattr(render_mod, "ensure_true_peak", lambda p: checked.append(p))
    dst, _ = _render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs)
    assert checked == [dst]


def test_render_without_audio_stream(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs
):
    from hookline.media.ffmpeg import extract_audio

    silent = tmp_path / "noaudio.mp4"
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i",
          "testsrc2=size=320x180:rate=25:duration=20", "-c:v", "libx264",
          "-pix_fmt", "yuv420p", str(silent)])  # fmt: skip
    wav = extract_audio(quiet_speech_video, tmp_path / "a.wav")  # timing source only
    from hookline.models import Word

    words = [Word(start=float(t), end=t + 0.5, text="hi") for t in range(20)]
    pick = sample_pick.model_copy(update={"start": 2.0, "end": 16.0})
    s = settings.model_copy(update={"captions": False})
    dst = tmp_path / "out" / "01-x.mp4"
    render_clip(silent, wav, pick, words, tmp_path / "r", dst, s, NoFaces())
    assert dst.exists()
    (final,) = final_argvs
    assert "-af" not in final
    assert _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                 "stream=index", "-of", "csv=p=0", str(dst)]).stdout.strip() == ""  # fmt: skip
