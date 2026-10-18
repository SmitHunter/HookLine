from pathlib import Path

import pytest

from hookline.media.ffmpeg import _run
from hookline.media.loudness import (
    COMPRESSOR,
    SPEECH_SWAY_DB,
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


def test_speech_like_clip_is_levelled_linear_and_minus_14(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs
):
    """Wave G-select: its 16 dB swings (LRA ~16) are levelled to <= 8 LU, so loudnorm goes
    linear (before the leveler it had to go dynamic)."""
    _check_render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs, "linear")
    m = measure_loudness(tmp_path / "out" / "01-x.mp4")
    assert m.lra <= 8.0


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
    # wave G-select: the whole clip may sit up to SPEECH_SWAY_DB off -14 so its speech lands
    # on -14 (its "words" are the loud bursts here). Round 1: the firm rungs' 1.2 s release
    # cannot level under full-scale clicks every 0.43 s, so this clip ends on the dynamic path
    # (no sway, but dynamic loudnorm's own range control lands ~0.5 dB under): +-1 kept
    assert m.i == pytest.approx(-14.0, abs=0.5 + SPEECH_SWAY_DB)
    assert m.tp <= -1.0


def test_near_silent_clip_not_boosted(
    near_silent_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.pipeline import render as render_mod

    trims = []
    monkeypatch.setattr(render_mod, "ensure_true_peak", lambda p, **k: trims.append(p))
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
    assert ensure_true_peak(copy) == pytest.approx(0.0)  # the peak limiter did it
    (cmd,) = cmds
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert "alimiter=limit=" in cmd[cmd.index("-af") + 1]
    # the limiter not enough (nor a deeper one): a volume trim from the original instead
    tps = iter([-0.4, -1.0, -1.0, -1.9])
    cmds.clear()
    trim = ensure_true_peak(copy)
    assert trim == pytest.approx(-1.1)
    assert len(cmds) == 3 and "volume=-1.10dB" in cmds[2]
    lims = [c[c.index("-af") + 1] for c in cmds[:2]]
    assert lims[0] != lims[1]  # the second limiter sits lower


def test_ensure_true_peak_acts_when_the_trim_still_overshoots(
    steady_quiet_video, tmp_path, monkeypatch
):
    """Round 1: the volume-trim fallback is re-measured; a peak still over -1.2 dBTP after it
    (AAC ringing again) gets a second trim instead of being left hot."""
    from hookline.media import loudness as loud_mod

    copy = tmp_path / "c.mp4"
    copy.write_bytes(steady_quiet_video.read_bytes())
    tps = iter([-0.4, -1.0, -1.0, -1.0, -1.6])
    monkeypatch.setattr(loud_mod, "measure_loudness", lambda p: _m(-14.0, tp=next(tps)))
    cmds = []
    real = loud_mod._run
    monkeypatch.setattr(loud_mod, "_run", lambda cmd: (cmds.append(cmd), real(cmd))[1])
    trim = ensure_true_peak(copy)
    # round 2: the whole correction re-applied to the original audio, not a trim of the trim
    assert len(cmds) == 4 and "volume=-1.10dB" in cmds[2] and "volume=-1.60dB" in cmds[3]
    assert trim == pytest.approx(-1.6)


def test_render_runs_true_peak_check(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.pipeline import render as render_mod

    checked = []
    monkeypatch.setattr(render_mod, "ensure_true_peak", lambda p, **k: checked.append(p))
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


# --- wave G-select: speech-gated loudness and a loudness range <= 8 LU -------------------------

GAME_SPEECH = [(2.0 * k, 2.0 * k + 1.6) for k in range(15) if k not in (4, 10)]


@pytest.fixture(scope="module")
def game_clip(tmp_path_factory) -> Path:
    """30 s: speech-like bursts (1.6 s on / 0.4 s off) at about -30 LUFS, with two 1.5 s game
    stingers (at 8 s and 20 s) 14 dB above the speech: voice under game, LRA well over 8."""
    out = tmp_path_factory.mktemp("game") / "game.mkv"
    speech = r"if(lt(mod(t\,2)\,1.6)\,1\,0)*(1-between(t\,8\,10)-between(t\,20\,22))"
    sting = r"8*(between(t\,8\,9.5)+between(t\,20\,21.5))"
    graph = (
        f"[1:a]volume='{speech}':eval=frame[s];[2:a]volume='{sting}':eval=frame[g];"
        "[s][g]amix=inputs=2:normalize=0[a]"
    )
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
          "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=30",
          "-f", "lavfi", "-i", "anoisesrc=color=pink:seed=3:amplitude=0.2:sample_rate=48000:duration=30",
          "-f", "lavfi", "-i", "anoisesrc=color=white:seed=5:amplitude=0.03:sample_rate=48000:duration=30",
          "-filter_complex",
          graph,
          "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "pcm_s16le", str(out)])  # fmt: skip
    return out


def _apply(src: Path, af: str, dst: Path) -> Path:
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(src), "-vn", "-af", af,
          "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(dst)])  # fmt: skip
    return dst


def test_game_clip_fixture_has_voice_under_game(game_clip):
    from hookline.media.loudness import speech_filter

    whole = measure_loudness(game_clip)
    speech = measure_loudness(game_clip, speech_filter(GAME_SPEECH))
    print(f"fixture: I={whole.i} LRA={whole.lra} speech I={speech.i}")
    assert whole.lra > 8 and whole.i - speech.i > 4


def test_speech_lands_near_the_target_and_the_range_is_compressed(game_clip, tmp_path):
    from hookline.media.loudness import MAX_LRA, plan_loudness, speech_filter

    plan = plan_loudness(game_clip, speech=GAME_SPEECH)
    out = _apply(game_clip, plan.af, tmp_path / "out.m4a")
    m = measure_loudness(out)
    s = measure_loudness(out, speech_filter(GAME_SPEECH))
    print(f"speech-gated: I={m.i} TP={m.tp} LRA={m.lra} speech I={s.i} ({plan.mode})")
    assert MAX_LRA == 8.0
    assert m.i == pytest.approx(-14.0, abs=1.0) and m.tp <= -1.0
    assert m.lra <= 8.0
    assert s.i >= -15.5  # the voice is no longer buried under the game


def test_speech_filter_selects_the_spans_and_merges_close_ones():
    from hookline.media.loudness import speech_filter

    f = speech_filter([(1.0, 1.5), (1.6, 3.0), (5.0, 6.25)])
    assert f.startswith("aselect='") and f.endswith(",asetpts=N/SR/TB")
    assert "between(t\\,1.000\\,3.000)+between(t\\,5.000\\,6.250)'" in f
    assert speech_filter([(1.0, 2.0)]) is None  # under 3 s of speech: not gated


def test_plan_without_speech_spans_keeps_the_whole_clip_target(monkeypatch, tmp_path):
    from hookline.media import loudness as loud_mod

    calls = []

    def fake(path, pre=None):
        calls.append(pre)
        return _m(-20.0, tp=-8.0, lra=6.0)

    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    plan = loud_mod.plan_loudness(tmp_path / "x.mkv")
    assert "loudnorm=I=-14:" in plan.af and all("aselect" not in (c or "") for c in calls)
    assert all((c or "").count("acompressor") <= 1 for c in calls)  # LRA 6: no leveler


def test_render_passes_the_spoken_spans_to_the_loudness_plan(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.pipeline import render as render_mod

    seen = []
    real = render_mod.plan_loudness
    monkeypatch.setattr(
        render_mod, "plan_loudness", lambda p, speech=None: seen.append(speech) or real(p, speech)
    )
    _render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs)
    (spans,) = seen
    assert spans and all(b > a for a, b in spans) and spans[0][0] < 1.0


# --- wave G-select round 1: leveler release/knee, ladder bookkeeping --------------------------


def test_firm_leveler_rungs_release_slowly_with_a_soft_knee():
    import re

    from hookline.media.loudness import prefilter

    def opts(level):
        leveler = prefilter(0.0, level).split("volume=", 1)[1]  # after the base compressor
        m = re.search(r"acompressor=threshold=[^,]*", leveler)
        return dict(kv.split("=") for kv in m.group(0).split(":")[1:])

    for level in (3, 4, 5):
        assert float(opts(level)["release"]) >= 1000 and float(opts(level)["knee"]) >= 6
    assert float(opts(1)["release"]) == 300  # gentle rungs: unchanged


def _ladder_fake(loud_mod, calls, lra_for, i_miss=1.0, s_i=None):
    """A measure_loudness stand-in: the chain always lands `i_miss` dB under -14 with a hot
    peak (so the pre-gain keeps being corrected); `lra_for(level, nth)` the range."""
    import re

    seen: dict[int, int] = {}

    def fake(path, pre=None):
        calls.append(pre)
        if pre == loud_mod.COMPRESSOR:
            return _m(-30.0, tp=-20.0, lra=12.0)
        if "aselect" in (pre or ""):
            return _m(s_i if s_i is not None else -14.0 - i_miss, tp=-3.0)
        m = re.search(r"threshold=(-?[\d.]+)dB:ratio=([\d.]+)", pre.split("volume=", 1)[1])
        level = 0
        if m:
            level = 1 + [round(-14 + o, 3) for o, _ in loud_mod.LEVELERS].index(float(m[1]))
        seen[level] = seen.get(level, 0) + 1
        return _m(-14.0 - i_miss, tp=-2.5, lra=lra_for(level, seen[level]))

    return fake


def test_ladder_resets_passes_on_a_climb_and_keeps_makeup_corrections(monkeypatch, tmp_path):
    from hookline.media import loudness as loud_mod

    calls = []
    # level 0: natural first, then over the range once the gain moved; level 1: over after
    # its first correction; level 2: fine
    lra = {0: [7.0, 9.0], 1: [7.0, 9.0]}
    fake = _ladder_fake(loud_mod, calls, lambda lv, n: lra.get(lv, [7.0, 7.0])[min(n, 2) - 1])
    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    loud_mod.plan_loudness(tmp_path / "x.mkv")
    lv1 = [c for c in calls if c and "threshold=-10dB" in c]
    lv2 = [c for c in calls if c and "threshold=-14dB" in c]
    # a full set of passes after the climb (then the hot peak sends loudnorm dynamic, whose
    # chain gets its own gain corrections)
    assert len(lv1) == 2 and len(lv2) == loud_mod.PREGAIN_PASSES + loud_mod.DYNAMIC_PASSES
    # level 1's +1 dB correction is kept on level 2 (its rung makeup is 0)
    assert "volume=1.00dB,aresample" in lv2[0]


def test_ladder_decodes_are_capped(monkeypatch, tmp_path):
    from hookline.media import loudness as loud_mod

    calls = []
    # every rung looks fine once, then over the range: the worst case for the ladder
    fake = _ladder_fake(loud_mod, calls, lambda lv, n: 7.0 if n == 1 else 9.0)
    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    speech = [(k * 2.0, k * 2.0 + 1.5) for k in range(10)]
    loud_mod.plan_loudness(tmp_path / "x.mkv", speech=speech)
    assert len(calls) <= loud_mod.MAX_DECODES <= 24


def test_silent_gated_speech_does_not_climb_the_ladder(monkeypatch, tmp_path):
    from hookline.media import loudness as loud_mod

    calls = []
    fake = _ladder_fake(loud_mod, calls, lambda lv, n: 6.0, i_miss=0.0, s_i=-60.0)
    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    speech = [(k * 2.0, k * 2.0 + 1.5) for k in range(10)]
    loud_mod.plan_loudness(tmp_path / "x.mkv", speech=speech)
    assert not any("threshold=-10dB" in (c or "") for c in calls)  # not "buried"


def test_dynamic_loudnorm_aims_the_whole_clip_at_the_target(monkeypatch, tmp_path):
    from hookline.media import loudness as loud_mod

    calls = []
    # the range never fits loudnorm's linear mode; the speech sits 2 dB under the clip
    fake = _ladder_fake(loud_mod, calls, lambda lv, n: 12.0, i_miss=0.0, s_i=-16.0)
    monkeypatch.setattr(loud_mod, "measure_loudness", fake)
    speech = [(k * 2.0, k * 2.0 + 1.5) for k in range(10)]
    plan = loud_mod.plan_loudness(tmp_path / "x.mkv", speech=speech)
    assert plan.mode == "dynamic" and "loudnorm=I=-14:" in plan.af  # no speech sway


def test_ensure_true_peak_limits_a_lone_transient_without_losing_loudness(tmp_path):
    """Wave G-layout: AAC rang 1.9 dB over the plan on one game transient (stream clip 01);
    a whole-clip volume trim then cost Q1 (-15.1 LUFS). Only the peak is limited now."""
    dst = tmp_path / "click.mp4"
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
          "-f", "lavfi", "-i", "color=c=gray:size=64x64:rate=25:duration=6",
          "-f", "lavfi", "-i",
          "aevalsrc='0.25*sin(2*PI*220*t)+0.7*between(t,3.000,3.0005)':s=48000:d=6",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
          "-shortest", str(dst)])  # fmt: skip
    before = measure_loudness(dst)
    assert before.tp > -1.2
    ensure_true_peak(dst)
    after = measure_loudness(dst)
    assert after.tp <= -1.0
    assert after.i == pytest.approx(before.i, abs=0.2)  # the loudness stays


def test_non_finite_loudnorm_numbers_never_reach_the_plan():
    """loudnorm prints "-inf" for silence and can print "nan" (an all-zero gated segment):
    neither may leak into a filter string."""
    import math

    from hookline.media.loudness import _num

    assert _num("-inf") == -70.0
    assert math.isfinite(_num("nan")) and math.isfinite(_num("inf"))
    assert _num("-23.5") == -23.5


def test_plan_loudness_probes_the_audio_stream_once(quiet_speech_video, tmp_path, monkeypatch):
    """The audio-stream probe is per file, not per loudnorm pass (a plan runs up to
    MAX_DECODES passes)."""
    import shutil

    from hookline.media import ffmpeg as ffmpeg_mod
    from hookline.media import loudness as loud_mod

    clip = tmp_path / "clip.mp4"
    shutil.copy(quiet_speech_video, clip)
    probes = []
    real = ffmpeg_mod._run

    def counting(argv, *a, **k):
        if argv[0] == "ffprobe":
            probes.append(argv)
        return real(argv, *a, **k)

    monkeypatch.setattr(ffmpeg_mod, "_run", counting)
    monkeypatch.setattr(loud_mod, "_run", counting)
    plan = loud_mod.plan_loudness(clip, speech=[(0.5, 4.0), (5.0, 9.0)])
    assert plan is not None and len(probes) == 1
