import subprocess

import numpy as np
import pytest

from hookline.media.audio import energy_envelope
from hookline.models import Word
from hookline.pipeline.timing import (
    Envelope,
    refine_word_ends,
    snap_onsets,
    tail_end,
)

HOP = 0.01
THR = -32.0


def _env(spans: list[tuple[float, float]], length: float = 5.0, t0: float = 0.0) -> Envelope:
    """-60 dB everywhere except -15 dB inside `spans` (absolute seconds)."""
    n = round(length / HOP)
    db = np.full(n, -60.0)
    for a, b in spans:
        db[round((a - t0) / HOP) : round((b - t0) / HOP)] = -15.0
    return Envelope(t0=t0, hop=HOP, db=db)


def _w(start, end, text="w"):
    return Word(start=start, end=end, text=text)


def test_energy_envelope_follows_a_tone_burst(tmp_path):
    wav = tmp_path / "burst.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "aevalsrc='0.5*sin(2*PI*440*t)*between(t,1.0,1.5)':s=16000:d=3",
         "-ac", "1", str(wav)],
        check=True,
    )  # fmt: skip
    env = energy_envelope(wav, 0.5, 2.5)
    assert env.t0 == pytest.approx(0.5) and env.hop == pytest.approx(0.01)
    assert len(env.db) == pytest.approx(200, abs=2)
    assert env.level(1.2) > -15 and env.level(0.8) < -60 and env.level(2.0) < -60
    # the rise and fall land within a hop of the burst edges
    voiced = np.flatnonzero(env.db > THR)
    assert env.t0 + voiced[0] * HOP == pytest.approx(1.0, abs=0.011)
    assert env.t0 + (voiced[-1] + 1) * HOP == pytest.approx(1.5, abs=0.011)


def test_refine_word_ends_extends_synthetic_dtw_ends_to_the_energy_fall():
    # DTW gives every single-token word end = start + 0.15; the speech runs longer
    env = _env([(1.0, 1.42), (1.6, 2.3)])
    words = [_w(1.0, 1.15, "special"), _w(1.6, 1.75, "about"), _w(1.9, 2.05, "this")]
    out = refine_word_ends(words, env, THR)
    assert out[0].end == pytest.approx(1.42, abs=0.011)  # to where the energy falls
    assert out[1].end == pytest.approx(1.9)  # voiced through to the next word: capped there
    assert out[2].end == pytest.approx(2.3, abs=0.011)
    assert [w.start for w in out] == [1.0, 1.6, 1.9]  # starts untouched


def test_refine_word_ends_never_shrinks_and_bridges_short_dips():
    env = _env([(1.0, 1.2), (1.23, 1.5)])  # a 30 ms stop-consonant dip inside the word
    (out,) = refine_word_ends([_w(1.0, 1.15)], env, THR)
    assert out.end == pytest.approx(1.5, abs=0.011)
    quiet = _env([])
    (same,) = refine_word_ends([_w(1.0, 1.4)], quiet, THR)
    assert same.end == 1.4


def test_refine_word_ends_caps_a_word_in_endless_noise():
    env = _env([(0.0, 5.0)])
    (out,) = refine_word_ends([_w(1.0, 1.15)], env, THR)
    assert out.end <= 1.0 + 1.5 + 1e-9


def test_snap_onsets_moves_a_late_post_pause_onset_to_the_energy_rise():
    # clip 04: silence ends 1.80 s, DTW put "And," at 1.98 s
    env = _env([(0.5, 1.2), (1.80, 2.6)])
    words = [_w(0.5, 1.2, "alone."), _w(1.98, 2.3, "And,"), _w(2.3, 2.6, "I")]
    out = snap_onsets(words, env, THR)
    assert out[1].start == pytest.approx(1.80, abs=0.011)
    assert out[1].end == 2.3 and out[2].start == 2.3  # a word without a pause is untouched
    assert out[0].start == pytest.approx(0.5)


def test_snap_onsets_ignores_words_without_a_pause_and_far_rises():
    env = _env([(0.5, 1.0), (1.05, 1.3), (2.0, 2.5)])
    words = [_w(0.5, 1.0), _w(1.1, 1.3)]  # 0.1 s gap: not a pause
    assert snap_onsets(words, env, THR) == words
    far = [_w(0.5, 1.0), _w(2.5, 2.8)]  # wave F: a rise 0.5 s before, after a pause: taken
    assert snap_onsets(far, _env([(0.5, 1.0), (2.0, 2.8)]), THR)[1].start == pytest.approx(2.0)
    after = [_w(0.5, 1.0), _w(1.5, 1.8)]  # a rise 0.5 s after the start: out of reach
    assert snap_onsets(after, _env([(0.5, 1.0), (2.0, 2.8)]), THR)[1].start == 1.5


def test_snap_onsets_never_crosses_the_previous_word_or_its_own_end():
    env = _env([(0.5, 1.0), (1.05, 1.6)])
    words = [_w(0.5, 1.0), _w(1.25, 1.6)]  # rise at 1.05 lies before ... the prev end + gap
    out = snap_onsets(words, env, THR)
    assert out[1].start >= out[0].end
    early = [_w(0.5, 0.7), _w(1.2, 1.25)]  # rise 0.25 s after the start: past the word's end
    out = snap_onsets(early, _env([(0.5, 0.7), (1.45, 1.6)]), THR)
    assert out[1].start == 1.2


def test_snap_onsets_snaps_an_early_onset_forward_too():
    env = _env([(0.5, 1.0), (1.7, 2.2)])
    out = snap_onsets([_w(0.5, 1.0), _w(1.55, 2.2)], env, THR)
    assert out[1].start == pytest.approx(1.7, abs=0.011)


def test_tail_end_snaps_past_the_last_word_into_silence():
    env = _env([(1.0, 2.12)])  # the word's energy runs 0.12 s past its transcribed end
    end = tail_end(2.0, 1.95, 10.0, env, THR)
    assert 2.12 <= end <= 2.0 + 0.35 + 1e-9
    assert end == pytest.approx(2.12 + 0.2, abs=0.011)  # wave F: 0.2 s into the silence


def test_tail_end_never_reaches_the_next_word_nor_shortens():
    env = _env([(1.0, 2.15), (2.2, 2.6)])  # the next word's onset is audible at 2.2
    end = tail_end(2.0, 2.0, 2.2, env, THR)
    assert end <= 2.2 - 0.05 + 1e-9
    assert tail_end(2.0, 2.3, 10.0, _env([(1.0, 2.0)]), THR) == 2.3  # already past: kept
    capped = tail_end(2.0, 2.0, 10.0, _env([(1.0, 3.0)]), THR)
    assert capped <= 2.35 + 1e-9  # no silence in reach: at most 0.35 s past the last word


# --- wave D review round 1: noisy sources, bounded extensions, audible next-word onsets ----


def _noisy(spans: list[tuple[float, float]], bed: float = -38.0, length: float = 6.0) -> Envelope:
    env = _env(spans, length)
    db = np.where(env.db < -20, bed, env.db)
    return Envelope(t0=env.t0, hop=env.hop, db=db)


def test_voice_threshold_follows_the_local_noise_floor():
    from hookline.pipeline.timing import voice_threshold

    assert voice_threshold(_env([(1.0, 2.0)]), -32.0) == pytest.approx(-44.0)  # clean: -32 - 12
    assert voice_threshold(_noisy([(1.0, 2.0)]), -32.0) is None  # -38 + 9 > -32: too noisy
    mid = _noisy([(1.0, 2.0)], bed=-50.0)
    assert voice_threshold(mid, -32.0) == pytest.approx(-41.0)  # floor + 9


def test_noise_bed_between_words_still_ends_words_and_snaps_onsets():
    from hookline.pipeline.timing import voice_threshold

    env = _noisy([(1.0, 1.42), (2.0, 2.6)], bed=-45.0)  # a -45 dB bed between the words
    thr = voice_threshold(env, -32.0)
    words = [_w(1.0, 1.15, "one"), _w(2.15, 2.3, "two")]
    ends = refine_word_ends(words, env, thr)
    assert ends[0].end == pytest.approx(1.42, abs=0.011)  # not stretched to the next word
    onsets = snap_onsets(ends, env, thr)
    assert onsets[1].start == pytest.approx(2.0, abs=0.011)
    assert tail_end(2.6, 2.6, 10.0, env, thr) == pytest.approx(2.80, abs=0.011)  # wave F: +0.2


def test_refine_word_ends_without_an_energy_fall_extends_at_most_0_6_s():
    env = _env([(0.0, 5.0)])
    (out,) = refine_word_ends([_w(1.0, 1.15)], env, THR)
    assert out.end == pytest.approx(1.75)
    later = refine_word_ends([_w(1.0, 1.15), _w(3.0, 3.2)], _env([(1.0, 2.2)]), THR)
    assert later[0].end == pytest.approx(2.2, abs=0.011)  # a real fall 1.05 s later counts


def test_protected_words_cap_the_refined_end():
    from hookline.pipeline.timing import protected_words

    raw = [_w(1.0, 1.15, "a"), _w(2.0, 2.15, "b")]
    refined = [_w(1.0, 1.75, "a"), _w(2.0, 2.3, "b")]
    out = protected_words(raw, refined)
    assert [w.end for w in out] == pytest.approx([1.45, 2.3])


def test_tail_ignores_a_next_word_start_without_an_audible_onset():
    # clip 01: DTW put "I" at 362.74 inside the sound of "alone." (voiced to ~362.87)
    env = _env([(1.0, 2.25)])
    assert tail_end(2.0, 2.0, 2.1, env, THR) == pytest.approx(2.45, abs=0.011)  # fall + 0.2
    # ... but an audible onset of the next word still stops it
    heard = _env([(1.0, 2.05), (2.15, 2.6)])
    assert tail_end(2.0, 2.0, 2.15, heard, THR) <= 2.10 + 1e-9


# --- wave D review round 2 ---------------------------------------------------------------


def test_voice_threshold_gives_up_on_a_bed_near_the_silence_level():
    from hookline.pipeline.timing import voice_threshold

    rng = np.random.default_rng(1)
    for bed, jitter in ((-31.0, 2.0), (-38.0, 0.0)):  # the reviewer's -31 +- 2 dB probe
        env = _noisy([(1.0, 1.3), (1.4, 1.7)], bed=bed)
        db = np.where(env.db == bed, bed + rng.uniform(-jitter, jitter, len(env.db)), env.db)
        assert voice_threshold(Envelope(t0=0.0, hop=HOP, db=db), -32.0) is None
    assert voice_threshold(_noisy([(1.0, 2.0)], bed=-45.0), -32.0) == pytest.approx(-36.0)


def test_tail_never_swallows_a_whole_audible_next_word():
    # "alone. I" where the short "I" (2.1-2.2) lies wholly inside the continuing sound: the
    # tail stops at its start instead of swallowing it
    env = _env([(1.0, 2.25)])
    end = tail_end(2.0, 2.0, 2.1, env, THR, next_end=2.2)
    assert end <= 2.1 - 0.05 + 1e-9
    # a next word running past the fall (DTW put its start inside the last word) is ignored
    assert tail_end(2.0, 2.0, 2.1, env, THR, next_end=2.6) == pytest.approx(2.45, abs=0.011)


# --- fix wave F: onsets after pauses, final and silent words, splices, tail ---------------


def test_snap_onsets_reanchors_a_late_onset_to_the_first_voiced_frame_within_0_7_s():
    # clip 03: "reveal." is voiced from 1.72 s, DTW put it at 2.30 s (580 ms late)
    env = _env([(0.5, 1.0), (1.72, 2.4)])
    words = [_w(0.5, 1.0, "the"), _w(2.30, 2.45, "reveal.")]
    out = snap_onsets(words, env, THR)
    assert out[1].start == pytest.approx(1.72, abs=0.011)
    assert out[1].end == 2.45
    # never before the previous word's end, nor past 0.7 s back
    near = [_w(0.5, 1.6, "the"), _w(2.30, 2.45, "reveal.")]
    assert snap_onsets(near, _env([(0.5, 2.4)]), THR)[1].start >= 1.6
    far = [_w(0.5, 1.0), _w(2.8, 3.0)]
    assert snap_onsets(far, _env([(0.5, 1.0), (1.9, 3.0)]), THR)[1].start == 2.8


def test_anchor_words_clamps_the_final_words_to_the_last_voiced_frame():
    from hookline.pipeline.timing import anchor_words

    # clip 02: "direct action." -- audio ends 1.9 s, DTW put "action." at 2.1 s
    env = _env([(0.5, 1.9)])
    words = [_w(0.5, 1.45, "take"), _w(1.45, 2.1, "direct"), _w(2.1, 2.3, "action.")]
    out = anchor_words(words, env, THR, clip_end=2.4)
    assert out[-1].end <= 1.9 + 0.011 and out[-1].start < out[-1].end
    assert out[-2].end <= out[-1].start + 1e-9 and out[-2].start == 1.45
    assert out[0] == words[0]


def test_anchor_words_moves_a_word_over_silence_to_the_nearest_voiced_run():
    from hookline.pipeline.timing import anchor_words

    env = _env([(0.5, 1.0), (1.6, 2.0), (3.0, 3.5)])
    words = [_w(0.5, 1.0, "a"), _w(1.1, 1.4, "b"), _w(3.0, 3.5, "c")]  # "b": 0% voiced
    out = anchor_words(words, env, THR, clip_end=4.0)
    assert out[1].start == pytest.approx(1.6, abs=0.011) and out[1].end <= 2.0 + 0.011
    assert out[1].end - out[1].start == pytest.approx(0.3, abs=0.011)
    assert out[0] == words[0] and out[2] == words[2]
    # mid-clip, nothing voiced between its neighbours: left alone (clip 04 "had ... literally":
    # taking a share of the next word's sound made "literally" 150 ms late)
    mid = [_w(0.5, 1.0, "a"), _w(1.1, 1.4, "had"), _w(1.5, 2.0, "literally"), _w(2.2, 2.6, "x")]
    env = _env([(0.5, 1.0), (1.5, 2.0), (2.2, 2.6)])
    assert anchor_words(mid, env, THR, clip_end=4.0)[1:3] == mid[1:3]
    # the clip's last spoken word: a silent word right before it shares its sound
    lone = [_w(0.5, 1.0, "a"), _w(1.1, 1.4, "b"), _w(1.5, 2.0, "c")]
    out = anchor_words(lone, _env([(0.5, 1.0), (1.5, 2.0)]), THR, clip_end=4.0)
    assert (out[1].start, out[1].end, out[2].start) == pytest.approx((1.5, 1.75, 1.75))


def test_uncaptioned_speech_finds_voiced_stretches_without_words():
    from hookline.pipeline.timing import uncaptioned_speech

    env = _env([(0.5, 1.0), (1.3, 2.8), (3.0, 3.2)])
    words = [_w(0.5, 1.0), _w(2.5, 2.8), _w(3.0, 3.2)]
    assert uncaptioned_speech(words, env, THR, [(0.0, 4.0)]) == [
        (pytest.approx(1.3, abs=0.011), pytest.approx(2.45, abs=0.06))
    ]
    # a short breath (< 0.3 s) is not speech; outside the kept ranges nothing counts
    assert uncaptioned_speech(words, _env([(1.3, 1.5)]), THR, [(0.0, 4.0)]) == []
    assert uncaptioned_speech(words, env, THR, [(0.0, 1.2), (2.5, 4.0)]) == []


def test_snap_splices_moves_a_range_start_out_of_a_dropped_word_tail():
    from hookline.pipeline.timing import snap_splices

    # clip 01: the drop ends at 2.5 s but "alone." (dropped) sounds until 2.6; the first kept
    # word starts at 2.8: the kept range now starts in the quiet between
    env = _env([(1.0, 1.9), (2.1, 2.6), (2.8, 3.5)])
    words = [_w(1.0, 1.9, "kept."), _w(2.1, 2.6, "alone."), _w(2.8, 3.5, "And")]
    keep = [(0.5, 2.0), (2.5, 4.0)]
    out = snap_splices(keep, words, env, THR, fps=30)
    (a0, e0), (s1, e1) = out
    assert (a0, e1) == (0.5, 4.0)  # the clip's own bounds stay
    assert 2.6 <= s1 < 2.8 and abs(s1 * 30 - round(s1 * 30)) < 1e-6
    assert 1.9 <= e0 <= 2.1 and abs(e0 * 30 - round(e0 * 30)) < 1e-6
    # never further than 0.15 s, never into a kept word's sound
    tight = snap_splices([(0.5, 2.0), (2.5, 4.0)], words, _env([(1.0, 1.95), (2.05, 3.5)]),
                         THR, fps=30)  # fmt: skip
    assert abs(tight[1][0] - 2.5) <= 0.15 + 1e-9 and tight[0][1] >= 1.95 - 1e-9


def test_tail_end_leaves_0_2_to_0_3_s_after_the_last_word():
    env = _env([(1.0, 2.02)])  # the word falls silent at once
    end = tail_end(2.0, 2.0, 10.0, env, THR)
    assert 2.02 + 0.2 - 0.011 <= end <= 2.02 + 0.3 + 1e-9
    capped = tail_end(2.0, 2.0, 10.0, _env([(1.0, 3.0)]), THR)
    assert capped <= 2.3 + 1e-9  # no silence in reach: at most 0.3 s past the last word


# --- fix wave F, real-clip follow-ups: fragments past the window, ramps, shared words -------


def test_snap_splices_skips_a_short_dropped_fragment_and_moves_a_misplaced_kept_word():
    from hookline.pipeline.timing import snap_splices

    # clip 01: the kept range starts at 2.67 inside the tail of the dropped "alone." (sounding
    # 2.56-2.86); DTW put the kept "I" mostly past that sound (2.80-3.05), its real onset is
    # 3.14 (a kept word mostly inside the burst is that burst and keeps it -- unless it has its
    # own onset just after it, see round 2; this "I" is not mostly inside: the burst goes)
    env = _env([(1.0, 1.9), (2.0, 2.42), (2.56, 2.86), (3.14, 3.5)])
    words = [_w(1.0, 1.9, "kept."), _w(2.28, 2.6, "alone."), _w(2.80, 3.05, "I"),
             _w(3.3, 3.5, "put")]  # fmt: skip
    keep, out = snap_splices([(0.5, 2.0), (2.67, 4.0)], words, env, THR, fps=30, adjust=True)
    assert 2.86 <= keep[1][0] < 3.14
    i_word = next(w for w in out if w.text == "I")
    assert i_word.start >= keep[1][0] and i_word.start == pytest.approx(3.14, abs=0.011)
    # a long continuing run is the kept speech itself: never skipped
    long_env = _env([(1.0, 1.9), (2.0, 3.6)])
    keep2, _ = snap_splices([(0.5, 2.0), (2.67, 4.0)], words, long_env, THR, fps=30, adjust=True)
    assert abs(keep2[1][0] - 2.67) <= 0.15 + 1e-9


def test_snap_onsets_reanchors_through_a_ramped_onset():
    # clip 02 "mainstream": the onset ramps (-46, -44, -47, then -32): no clean 50 ms of quiet
    # right before it, still the first voiced run in the window
    env = _env([(0.5, 1.0), (3.6, 4.4)])
    db = env.db.copy()
    db[358], db[359], db[360], db[361], db[362] = -46.0, -31.0, -33.0, -31.0, -15.0
    env = Envelope(t0=0.0, hop=HOP, db=db)
    words = [_w(0.5, 1.0, "So,"), _w(3.9, 4.05, "mainstream")]
    assert snap_onsets(words, env, THR)[1].start == pytest.approx(3.59, abs=0.011)


def test_anchor_words_shares_the_next_words_sound_when_none_is_between():
    from hookline.pipeline.timing import anchor_words

    # clip 02 end: "direct" (DTW 1.58-1.73) sits over silence; "direct action." is the one
    # burst 1.80-2.0 that DTW gave "action." alone
    env = _env([(0.5, 1.48), (1.8, 2.0)])
    words = [_w(0.5, 1.48, "community,"), _w(1.58, 1.73, "direct"), _w(1.8, 2.45, "action.")]
    out = anchor_words(words, env, THR, clip_end=2.45)
    assert out[1].start == pytest.approx(1.8, abs=0.011)
    assert out[1].end == pytest.approx(out[2].start) and out[2].end <= 2.0 + 0.011
    assert out[2].start - out[1].start == pytest.approx(0.2 * 6 / 13, abs=0.02)


def test_uncaptioned_speech_bridges_syllable_gaps():
    from hookline.pipeline.timing import uncaptioned_speech

    # clip 02: three short bursts between "So," and "mainstream" (gaps < 0.2 s): one stretch
    env = _env([(0.5, 1.0), (1.44, 1.66), (1.8, 1.83), (2.04, 2.14), (2.6, 3.0)])
    words = [_w(0.5, 1.0), _w(2.6, 3.0)]
    (span,) = uncaptioned_speech(words, env, THR, [(0.0, 4.0)])
    assert span[0] == pytest.approx(1.44, abs=0.011) and span[1] == pytest.approx(2.14, abs=0.011)


# --- fix wave F review round 1: reviewer probes as regressions -----------------------------


def test_snap_splices_keeps_a_kept_word_its_burst_starts_with():
    from hookline.pipeline.timing import snap_splices

    # case A: the dropped "um" runs straight into the kept "so" (one sound 0.7-1.25)
    env = _env([(0.3, 0.6), (0.7, 1.25), (1.45, 2.0)])
    words = [_w(0.3, 0.6, "okay"), _w(0.7, 0.95, "um"), _w(1.0, 1.25, "so"),
             _w(1.45, 2.0, "what")]  # fmt: skip
    keep, out = snap_splices([(0.0, 0.65), (1.0, 3.0)], words, env, THR, fps=30, adjust=True)
    assert keep[1][0] <= 1.0 + 1e-9 and out[2] == words[2]
    # case B: the kept range starts at its short word's own sound (0.98-1.2); the dropped
    # "um" has no sound of its own there
    env = _env([(0.3, 0.6), (0.98, 1.2), (1.4, 2.0)])
    words = [_w(0.3, 0.6, "okay"), _w(0.7, 0.9, "um"), _w(1.0, 1.2, "I"), _w(1.4, 2.0, "think")]
    keep, out = snap_splices([(0.0, 0.65), (1.0, 3.0)], words, env, THR, fps=30, adjust=True)
    assert keep[1][0] <= 1.0 + 1e-9 and out[2] == words[2]


def test_snap_onsets_takes_the_run_leading_into_the_word_not_a_click():
    # reviewer probe: a 40 ms click at 1.30, the word's sound from 1.85, DTW start 1.9
    env = _env([(0.5, 1.0), (1.3, 1.34), (1.85, 2.3)])
    out = snap_onsets([_w(0.5, 1.0, "a"), _w(1.9, 2.3, "word")], env, THR)
    assert out[1].start == pytest.approx(1.85, abs=0.011)


def test_anchor_words_keeps_words_in_their_own_ranges_sound():
    from hookline.pipeline.timing import anchor_words

    # the last kept range (2.0-3.0) is quiet: its word must not be squashed into the earlier
    # range's speech
    env = _env([(0.5, 1.4)])
    words = [_w(0.5, 1.4, "said."), _w(2.1, 2.4, "Yes.")]
    out = anchor_words(words, env, THR, clip_end=3.0, ranges=[(0.0, 1.5), (2.0, 3.0)])
    assert out == words


def test_snap_onsets_chains_back_over_a_short_gap_inside_the_word():
    # clip 03 "reveal.": sound 3.34-3.85, a 50 ms stop, 3.90-4.02; DTW start 3.98 (640 ms
    # late): the onset is the start of the whole utterance, not of its last syllable
    env = _env([(2.5, 3.05), (3.34, 3.85), (3.90, 4.02)])
    out = snap_onsets([_w(2.5, 3.05, "really"), _w(3.98, 4.5, "reveal.")], env, THR)
    assert out[1].start == pytest.approx(3.34, abs=0.011)


# --- fix wave F review round 2: burst-skip tie-break ---------------------------------------


def test_snap_splices_skips_a_burst_whose_kept_word_has_its_own_onset_after_it():
    from hookline.pipeline.timing import snap_splices

    # clip 01 28.57: DTW put the kept "I" (2.62-2.92) 80% inside the dropped "alone."'s sound
    # (2.56-2.86); the real "I" sounds from 3.14, within its span end + 0.3 s: the burst is
    # skipped and "I" moves to its onset
    env = _env([(1.0, 1.9), (2.0, 2.42), (2.56, 2.86), (3.14, 3.25), (3.4, 3.7)])
    words = [_w(1.0, 1.9, "kept."), _w(2.28, 2.6, "alone."), _w(2.62, 2.92, "I"),
             _w(3.4, 3.7, "put")]  # fmt: skip
    keep, out = snap_splices([(0.5, 2.0), (2.67, 4.0)], words, env, THR, fps=30, adjust=True)
    assert 2.86 <= keep[1][0] < 3.14  # no "...lone" fragment
    i_word = next(w for w in out if w.text == "I")
    assert i_word.start == pytest.approx(3.14, abs=0.011) and i_word.start >= keep[1][0]
    # its only onset after the burst is the next kept word's: the burst is that word (kept)
    late = [*words[:3], _w(3.14, 3.5, "put")]
    keep2, out2 = snap_splices([(0.5, 2.0), (2.67, 4.0)], late, env, THR, fps=30, adjust=True)
    assert keep2[1][0] <= 2.67 + 1e-9 and out2[2] == late[2]


def test_uncaptioned_speech_needs_a_voice_like_spectrum(tmp_path):
    """Q3b (wave G-layout): a loud noise burst (game sound), a low rumble or a sustained tone
    (a scream, chase music) with no word is not uncaptioned speech; a harmonic stretch with
    formant-like energy in the voice band, rising and falling in syllables, is."""
    import wave

    from hookline.pipeline.timing import uncaptioned_speech

    rate = 16000
    t = np.arange(6 * rate) / rate
    rng = np.random.default_rng(0)
    sig = np.zeros_like(t)
    noise = (t >= 0.5) & (t < 1.3)
    sig[noise] = 0.3 * rng.standard_normal(noise.sum())
    rumble = (t >= 1.5) & (t < 2.2)
    sig[rumble] = 0.5 * np.sin(2 * np.pi * 60 * t[rumble])
    voiced = (t >= 2.6) & (t < 3.4)
    sustained = (t >= 4.0) & (t < 4.8)
    for k in range(1, 30):  # f0 140 Hz, harmonics shaped by formants near 600 and 1500 Hz
        f = 140 * k
        gain = np.exp(-(((f - 600) / 250) ** 2)) + 0.6 * np.exp(-(((f - 1500) / 300) ** 2))
        tone = 0.1 * gain * np.sin(2 * np.pi * f * t)
        syllables = np.abs(np.sin(2 * np.pi * 2.0 * t)) ** 2  # ~4 syllables a second
        sig[voiced] += (tone * syllables)[voiced]
        sig[sustained] += tone[sustained]
    wav = tmp_path / "mix.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    env = energy_envelope(wav, 0.0, 6.0)
    assert env.voice is not None and len(env.voice) == len(env.db)
    (span,) = uncaptioned_speech([], env, THR, [(0.0, 6.0)])
    assert span == (pytest.approx(2.6, abs=0.1), pytest.approx(3.4, abs=0.1))
