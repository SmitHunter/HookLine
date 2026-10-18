"""Wave H (stream editing): static / death / loading screens, payoff extension, voiceless
stretches, the cold-open seam, caption holds, frame-0 pre-roll and the new audit checks."""

import itertools
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hookline.media.audio import Envelope
from hookline.models import ClipPick, RubricScores, Word
from hookline.pipeline.signals import SIGNALS_VERSION, Signals

FULL = 0b1111  # every frame of a second is a screen frame (4 fps)


def pick(start, end, **kw):
    scores = RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    )
    return ClipPick(start=start, end=end, title="t", hook_text="h", reason="r", scores=scores, **kw)


def W(start, end, text="word"):
    return Word(start=start, end=end, text=text)


def sentence(start, text, step=0.3):
    toks = text.split()
    return [W(start + i * step, start + i * step + step * 0.8, t) for i, t in enumerate(toks)]


def sig(n=120, screen=(), scares=(), motion=20.0):
    """`n` s of busy gameplay; `screen` (lo, hi) second ranges are full screen seconds."""
    mask = [0] * n
    for lo, hi in screen:
        for s in range(lo, hi):
            mask[s] = FULL
    return Signals(
        duration=n,
        loudness=[-20.0] * n,
        version=SIGNALS_VERSION,
        motion=[motion] * n,
        onsets=[int(t) for t in scares],
        transients=[int(t) for t in scares],
        onset_times=list(scares),
        transient_times=list(scares),
        screen=mask,
    )


# --- 1. static-screen detection ---------------------------------------------------------------


@pytest.fixture(scope="module")
def screen_video(tmp_path_factory) -> Path:
    """2 s busy test pattern, 2 s black with a small white box (death screen text), 2 s
    white (a fade), 2 s test pattern."""
    out = tmp_path_factory.mktemp("screen") / "screen.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=8",
         "-vf", ("drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,2,3.99)',"
                "drawbox=x=150:y=85:w=16:h=6:color=white:t=fill:enable='between(t,2,3.99)',"
                "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,4,5.99)'"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


def test_frame_stats_marks_black_and_white_screen_frames(screen_video):
    from hookline.media.motion import frame_motion, frame_stats

    motion, screen = frame_stats(screen_video)
    assert len(motion) == len(screen) == 8
    assert screen[0] == screen[1] == screen[6] == screen[7] == 0  # busy picture
    assert screen[2] == screen[3] == screen[4] == screen[5] == FULL  # black / white screens
    assert frame_motion(screen_video) == motion


def test_signals_version_bumped_for_screens():
    assert SIGNALS_VERSION >= 4
    assert Signals(duration=1, loudness=[0.0]).screen is None


def test_static_runs_need_screen_frames_mostly_without_words():
    from hookline.pipeline.signals import static_runs

    s = sig(screen=[(10, 26), (40, 44), (60, 62), (80, 84)])
    words = [
        *sentence(17.6, "I'm trying not to."),  # 1 s of words in a 16 s death screen
        *sentence(40.0, "one two three four five six seven eight nine ten eleven"),  # voiced
    ]
    runs = static_runs(s, words)
    # 10-26: mostly voiceless; 40-44: voiced; 60-62: under 3 s; 80-84: voiceless
    assert runs == [(10.0, 26.0), (80.0, 84.0)]


def test_static_runs_are_frame_precise_and_bridge_one_frame():
    from hookline.pipeline.signals import static_runs

    s = sig(screen=[(10, 14)])
    mask = list(s.screen)
    mask[9] = 0b1100  # the last two frames of second 9 are already black
    mask[12] = 0b1101  # one busy frame inside: bridged
    s = s.model_copy(update={"screen": mask})
    assert static_runs(s, []) == [(9.5, 14.0)]
    assert static_runs(sig(n=20), []) == []
    assert static_runs(Signals(duration=5, loudness=[0.0] * 5), []) == []


def test_static_gaps_cut_the_screen_inside_a_clip_but_never_half_a_word(caplog):
    from hookline.pipeline.boundaries import static_gaps

    s = sig(screen=[(10, 26)])
    words = [*sentence(5, "Oh no"), W(9.5, 10.4, "oh!"), *sentence(17.6, "I'm trying not to."),
             *sentence(30, "We are back now.")]  # fmt: skip
    with caplog.at_level("INFO"):
        gaps = static_gaps(pick(5, 33), words, s)
    assert gaps == [(pytest.approx(10.45), 26.0)]  # after "oh!"; "I'm trying not to." goes
    assert any("static" in r.getMessage() for r in caplog.records)
    assert static_gaps(pick(30, 33), words, s) == []  # no screen inside


def test_a_clip_never_starts_or_ends_on_a_static_screen(caplog):
    from hookline.pipeline.boundaries import avoid_static

    s = sig(screen=[(53, 69), (110, 115)])
    words = [*sentence(67.6, "Game plan! Game plan!"), *sentence(69.5, "No, game time!"),
             *sentence(80, "I don't want a hug!"), W(108.0, 110.6, "No!")]  # fmt: skip
    with caplog.at_level("INFO"):
        p = avoid_static(pick(67.6, 112.0, first_words="Game plan!"), words, s)
    assert p.start == pytest.approx(69.5)  # the next sentence after the fade
    assert p.first_words.startswith("No, game time!")
    assert p.end == pytest.approx(110.6)  # the word running into the screen ends it
    assert any("static" in r.getMessage() for r in caplog.records)
    clean = pick(70, 100)
    assert avoid_static(clean, words, s) is clean


# --- 2. payoff extension ------------------------------------------------------------------------


def test_payoff_scare_just_after_the_end_extends_the_clip(caplog):
    from hookline.pipeline.boundaries import extend_payoff

    words = [*sentence(10, "We are running away now"), W(28.0, 30.0, "No!")]
    s = sig(scares=[37.8])
    with caplog.at_level("INFO"):
        p = extend_payoff(pick(10, 30.0), words, s, max_s=60)
    assert p.end == pytest.approx(37.8 + 2.5)
    assert any("payoff" in r.getMessage() for r in caplog.records)
    # a reaction said after the scare is kept to its sentence end
    said = [*words, *sentence(38.5, "Oh my god he got me.", step=0.5)]
    assert extend_payoff(pick(10, 30.0), said, s, max_s=60).end == pytest.approx(said[-1].end)


def test_payoff_extension_limits():
    from hookline.pipeline.boundaries import extend_payoff

    words = [*sentence(10, "We are running away now"), W(28.0, 30.0, "No!")]
    base = pick(10, 30.0)
    assert extend_payoff(base, words, sig(scares=[39.5]), max_s=60) is base  # too late
    assert extend_payoff(base, words, sig(scares=[20.0]), max_s=60) is base  # inside the clip
    # a static screen right after the scare: the clip ends where the screen starts
    p = extend_payoff(base, words, sig(scares=[37.8], screen=[(39, 50)]), max_s=60)
    assert p.end == pytest.approx(38.75)  # a motion frame (0.25 s) clear of the screen
    # a static screen before the scare: no extension
    assert extend_payoff(base, words, sig(scares=[37.8], screen=[(33, 36)]), max_s=60) is base
    # over the maximum length (voiceless cuts counted): no extension
    assert extend_payoff(base, words, sig(scares=[37.8]), max_s=9.0) is base


# --- 3. voiceless stretches -------------------------------------------------------------------


def test_voiceless_stretches_are_trimmed_keeping_the_build_up(caplog):
    from hookline.pipeline.boundaries import VOICELESS_KEEP_S, voiceless_gaps

    words = [*sentence(0, "Game time"), *sentence(20, "I don't want a hug!")]
    last = words[1].end
    with caplog.at_level("INFO"):
        gaps = voiceless_gaps(pick(0, 23), words, sig())
    assert gaps == [(pytest.approx(last + 0.3), pytest.approx(20 - VOICELESS_KEEP_S))]
    assert any("voiceless" in r.getMessage() for r in caplog.records)
    # a scare inside is never cut: the stretch splits there
    gaps = voiceless_gaps(pick(0, 23), words, sig(scares=[10.0]))
    assert gaps == [
        (pytest.approx(last + 0.3), pytest.approx(10 - VOICELESS_KEEP_S)),
        (pytest.approx(11.5), pytest.approx(20 - VOICELESS_KEEP_S)),
    ]
    assert all(not (a <= 10.0 <= b) for a, b in gaps)
    # 6 s or less: kept whole
    assert voiceless_gaps(pick(0, 23), [*sentence(0, "a b"), *sentence(6, "c d")], sig()) == []


def test_voiceless_stretch_counts_a_held_word_as_at_most_1_5_s():
    from hookline.pipeline.boundaries import voiceless_gaps

    words = [*sentence(0, "Run run"), W(1.0, 4.4, "No!")]  # DTW held "No!" 3.4 s
    gaps = voiceless_gaps(pick(0, 13.5), words, sig(scares=[11.0]))
    assert gaps == [(pytest.approx(2.8), pytest.approx(11.0 - 3.2))]


def test_voiceless_gap_before_a_payoff_scare_after_the_last_word():
    from hookline.pipeline.boundaries import voiceless_gaps

    words = [*sentence(0, "Run run run"), W(2.0, 3.0, "No!")]
    gaps = voiceless_gaps(pick(0, 13.5), words, sig(scares=[11.0]))
    assert gaps == [(pytest.approx(3.3), pytest.approx(11.0 - 3.2))]


def test_edit_gaps_merge_idle_static_and_voiceless_cuts():
    from hookline.pipeline.boundaries import edit_gaps

    words = [*sentence(0, "Game time"), *sentence(30, "I don't want a hug!")]
    s = sig(screen=[(5, 12)])
    gaps = edit_gaps(pick(0, 33), words, s)
    assert gaps == sorted(gaps)
    assert all(b0 < a1 for (_, b0), (a1, _) in itertools.pairwise(gaps))
    cut = sum(b - a for a, b in gaps)
    assert cut == pytest.approx(30 - 3.2 - (words[1].end + 0.3), abs=0.05)


def test_fit_to_signals_moves_off_screens_and_extends_the_payoff():
    from hookline.pipeline.boundaries import fit_to_signals

    s = sig(n=200, screen=[(53, 69), (111, 126)], scares=[109.8])
    words = [*sentence(67.6, "Game plan! Game plan!"), *sentence(69.5, "No, game time!"),
             *sentence(89, "I don't want a hug!"), W(98.2, 101.6, "No!")]  # fmt: skip
    p = fit_to_signals(pick(67.6, 101.6), words, s, max_s=50.0)
    assert p.start == pytest.approx(69.5)
    assert p.end == pytest.approx(110.75)  # the catch, up to the death screen


# --- 4. cold-open seam -------------------------------------------------------------------------


def _env(levels: list[tuple[float, float, float]], length=3.0, t0=0.0, hop=0.01) -> Envelope:
    db = np.full(round(length / hop), -55.0)
    for a, b, lv in levels:
        db[round((a - t0) / hop) : round((b - t0) / hop)] = lv
    return Envelope(t0=t0, hop=hop, db=db)


def test_cause_onset_just_before_a_segment_start():
    from hookline.pipeline.timing import cause_onset

    env = _env([(1.0, 1.15, -18.0), (1.4, 2.0, -15.0)])  # a bang, then the reaction
    assert cause_onset(env, 1.47, []) == pytest.approx(0.98)
    assert cause_onset(env, 2.5, []) is None  # nothing within 0.6 s before
    # an earlier word owns the sound: not pulled in (no fragment)
    assert cause_onset(env, 1.47, [W(0.7, 1.2, "girl.")]) is None


def test_seam_join_and_dark_flash():
    from hookline.pipeline.render import seam_flash, seam_join

    keep = [(45.0, 46.0), (46.5, 48.0), (30.0, 40.0), (41.0, 55.0)]
    assert seam_join(keep, (45.0, 48.0)) == 1
    assert seam_join(keep, None) is None
    assert "color=white" in seam_flash(2.0, 30.0)
    dark = seam_flash(2.0, 30.0, dark=True)
    assert "color=black" in dark and "white" not in dark


def _pcm(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "48000",
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.int16).astype(float) / 32768


def test_cut_segments_crossfades_the_seam_over_40_ms(tmp_path):
    from hookline.media.ffmpeg import cut_segments

    src = tmp_path / "steps.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
         "-f", "lavfi", "-i",
         "aevalsrc='(0.02+0.5*gte(t,2))*sin(2*PI*440*t)':s=48000:d=4",
         "-c:v", "libx264", "-c:a", "pcm_s16le", "-shortest", str(src)],
        check=True,
    )  # fmt: skip
    keep = [(0.0, 1.0), (round(2.5 * 30) / 30, 3.5)]  # quiet, then loud
    out = cut_segments(src, keep, tmp_path / "x.mkv", "libx264", fps=30.0, crossfade_at=0)
    a = _pcm(out)
    assert len(a) == pytest.approx(2.0 * 48000, abs=48 * 3)  # the timeline is unchanged
    i = 48000  # the seam
    env = [np.sqrt(np.mean(a[k : k + 240] ** 2)) for k in range(i - 1200, i + 2400, 240)]
    db = 20 * np.log10(np.maximum(env, 1e-6))
    assert max(np.diff(db)) < 9.0  # a ramp over ~40 ms, not a step or a dip-and-jump
    assert abs(a[i + 2400 : i + 4800]).max() > 0.4  # full level after the ramp


# --- 5. caption holds ------------------------------------------------------------------------


def test_inflated_word_ends_are_clamped_to_the_voice():
    from hookline.pipeline.timing import clamp_word_ends

    env = _env([(1.0, 1.4, -15.0), (3.0, 4.5, -15.0)], length=8.0)
    ws = [W(1.0, 5.0, "please!"), W(3.0, 4.6, "noooo"), W(6.0, 6.3, "ok")]
    out = clamp_word_ends(ws, env, -32.0)
    assert out[0].end == pytest.approx(1.5, abs=0.02)  # voice ends at 1.4
    assert out[1].end == pytest.approx(4.6)  # voice continues to 4.5: kept
    assert out[2] == ws[2]  # short words untouched
    silent = clamp_word_ends([W(6.0, 8.0, "hmm")], env, -32.0)
    assert silent[0].end == pytest.approx(6.8)  # no voice at all: at most 0.8 s
    # a game bed over the voice threshold: the voice is what stays near the word's peak
    bed = _env([(1.0, 1.3, -17.0), (1.3, 6.0, -37.0)], length=8.0)
    held = clamp_word_ends([W(1.0, 5.0, "please!")], bed, -44.0)
    assert held[0].end == pytest.approx(1.4, abs=0.02)


def test_hook_ends_at_the_seam_and_earlier_tag_starts_there():
    from hookline.captions.ass import build_ass

    ass = build_ass([W(0.2, 0.5, "hey")], "Hook here", 20, seam=1.5)
    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 1,")]
    hook = next(ln for ln in lines if "EARLIER" not in ln)
    tag = next(ln for ln in lines if "EARLIER" in ln)
    assert hook.startswith("Dialogue: 1,0:00:00.00,0:00:01.50,")
    assert tag.startswith("Dialogue: 1,0:00:01.50,0:00:02.70,")


# --- 6. frame-0 pre-roll ---------------------------------------------------------------------


def test_pre_roll_three_frames_when_speech_starts_at_frame_0():
    from hookline.pipeline.timing import pre_roll

    env = _env([(1.0, 1.5, -15.0)])
    assert pre_roll(env, 1.0, [W(1.0, 1.3)], 30.0) == pytest.approx(0.9)
    assert pre_roll(env, 0.8, [W(1.0, 1.3)], 30.0) == 0.8  # silence leads already
    # the previous word ends just before: start in the quietest frame between them
    env2 = _env([(0.5, 0.95, -15.0), (0.95, 0.97, -45.0), (0.97, 1.0, -30.0), (1.0, 1.5, -15.0)])
    out = pre_roll(env2, 1.0, [W(0.5, 0.97, "girl."), W(1.0, 1.3, "This")], 30.0)
    assert out == pytest.approx(0.95)


# --- 7. audit checks ---------------------------------------------------------------------------


def test_q_static_fails_on_voiceless_screen_frames():
    from hookline.quality import _check_static

    flags = [False] * 40
    flags[8:12] = [True] * 4  # 1 s of black at 4 fps
    r = _check_static(flags, 4, [])
    assert r.id == "Q-static" and not r.ok and r.value == pytest.approx(1.0)
    assert _check_static(flags[:10], 4, []).ok  # 0.5 s: within the bar
    # the same dark frames under speech are dark gameplay, not a death screen
    assert _check_static(flags, 4, [W(1.9, 3.1, "talking")]).ok


def test_q_voiceless_warns_on_long_wordless_spans():
    from hookline.quality import _check_voiceless

    r = _check_voiceless([W(0.0, 1.0), W(8.5, 9.0)], 10.0)
    assert r.id == "Q-voiceless" and r.ok and r.status == "warn"
    assert _check_voiceless([W(0.0, 1.0), W(5.0, 9.0)], 10.0).status == "pass"


def test_q_seam_fails_on_an_audio_step_at_the_seam():
    from hookline.quality import _check_seam

    rate = 48000
    t = np.arange(2 * rate) / rate
    level = np.where(t < 1.0, 0.03, 0.6)
    step = level * np.sin(2 * np.pi * 440 * t)
    r = _check_seam(step, rate, 1.0)
    assert r.id == "Q-seam" and not r.ok
    ramp = np.interp(t, [0, 0.98, 1.02, 2], [0.03, 0.03, 0.6, 0.6]) * np.sin(2 * np.pi * 440 * t)
    assert _check_seam(ramp, rate, 1.0).ok
    assert _check_seam(step, rate, None).ok  # no cold open


def test_q_beat_fails_when_a_full_cam_beat_covers_a_scare():
    from hookline.quality import _check_beat

    side = {"layout": [{"cam_beats": [[5.0, 7.5]]}], "scare_times": [7.0]}
    r = _check_beat(side)
    assert r.id == "Q-beat" and not r.ok
    assert _check_beat({**side, "scare_times": [3.0, 9.0]}).ok
    assert _check_beat({"layout": [{"cam_beats": [[5.0, 7.5]]}]}).ok  # not recorded


def test_q5_accepts_a_hook_that_ends_at_an_early_seam():
    from hookline.quality import _check_hook, _Event

    hook = _Event(0.0, 1.5, "Hook", "Hook here")
    assert _check_hook([hook], [[70], [70]], 20.0, seam=1.5).ok
    assert not _check_hook([hook], [[70], [70]], 20.0).ok


def test_words_ending_inside_the_pre_roll_are_not_captioned():
    from hookline.pipeline.render import drop_lead_fragments

    ws = [W(0.0, 0.06, "girl."), W(0.06, 0.3, "This"), W(0.3, 0.5, "makes")]
    assert [w.text for w in drop_lead_fragments(ws, 0.06)] == ["This", "makes"]
    assert drop_lead_fragments(ws, 0.0) == ws
