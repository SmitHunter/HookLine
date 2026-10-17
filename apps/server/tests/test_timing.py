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
    far = [_w(0.5, 1.0), _w(2.5, 2.8)]  # the rise at 2.0 is 0.5 s away (> 0.25 s window)
    assert snap_onsets(far, _env([(0.5, 1.0), (2.0, 2.8)]), THR)[1].start == 2.5


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
    assert end == pytest.approx(2.12 + 0.08, abs=0.011)  # settles a little into the silence


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
    assert tail_end(2.6, 2.6, 10.0, env, thr) == pytest.approx(2.68, abs=0.011)


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
    assert tail_end(2.0, 2.0, 2.1, env, THR) == pytest.approx(2.33, abs=0.011)
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
    assert tail_end(2.0, 2.0, 2.1, env, THR, next_end=2.6) == pytest.approx(2.33, abs=0.011)
