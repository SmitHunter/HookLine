import numpy as np
import pytest

from hookline.pipeline.signals import (
    CLIP_MARKER,
    SCARE_MARKER,
    SIGNALS_VERSION,
    Signals,
    audio_onsets,
    flash_seconds,
    heat_markers,
    heat_seconds,
    loud_seconds,
    low_motion,
    scare_seconds,
)


def test_chat_spike_marked_once():
    chat = [1.0] * 600
    chat[300:310] = [8.0] * 10
    m = heat_markers(Signals(duration=600, loudness=[-25.0] * 600, chat=chat))
    keys = [k for k, v in m.items() if any(x.startswith("🔥chat") for x in v)]
    assert len(keys) == 1 and 295 <= keys[0] <= 301


def test_loud_marked_and_no_chat_ok():
    loud = [-30.0] * 100
    loud[50] = -15.0
    m = heat_markers(Signals(duration=100, loudness=loud, chat=None))
    assert m == {50: ["🔊loud"]}


def test_loud_run_marks_only_first_second():
    loud = [-40.0] * 100
    loud[20:25] = [-10.0] * 5
    assert heat_markers(Signals(duration=100, loudness=loud, chat=None)) == {20: ["🔊loud"]}


def test_quiet_absolute_floor_not_marked():
    loud = [-80.0] * 100
    loud[50] = -35.0  # +45 dB over median but below -30 dB
    assert heat_markers(Signals(duration=100, loudness=loud, chat=None)) == {}


def test_chat_ratio_in_label_and_minimum_baseline():
    chat = [0.0] * 600
    chat[300:310] = [1.0] * 10  # sum 10 vs baseline floor 1.0
    m = heat_markers(Signals(duration=600, loudness=[], chat=chat))
    labels = [x for v in m.values() for x in v]
    assert len(labels) == 1 and labels[0].startswith("🔥chat×")


def _spike(kind_counts, start=300):
    chat = [1.0] * 600
    chat[start : start + 10] = [8.0] * 10
    kinds = {start + 2: dict(kind_counts)}
    return Signals(duration=600, loudness=[], chat=chat, chat_kinds=kinds)


def test_spike_labelled_with_dominant_kind():
    for kind, emoji in [("laugh", "😂"), ("hype", "🏆"), ("fail", "💀"), ("shock", "😱")]:
        m = heat_markers(_spike({kind: 6, "hype" if kind != "hype" else "laugh": 1}))
        labels = [x for v in m.values() for x in v]
        assert len(labels) == 1 and labels[0].startswith("🔥chat×") and labels[0].endswith(emoji)


def test_spike_without_dominant_kind_has_no_emoji():
    m = heat_markers(_spike({"laugh": 3, "hype": 3, "fail": 3}))
    (label,) = [x for v in m.values() for x in v]
    assert label.startswith("🔥chat×") and " " not in label


def test_clip_kind_never_labels_a_spike():
    m = heat_markers(_spike({"clip": 2}))
    (label,) = [x for v in m.values() for x in v if x.startswith("🔥")]
    assert " " not in label


def test_clip_requests_add_standalone_marker():
    kinds = {100: {"clip": 1}, 103: {"clip": 1}, 106: {"clip": 1}}
    sig = Signals(duration=300, loudness=[], chat=[0.0] * 300, chat_kinds=kinds)
    m = heat_markers(sig)
    clips = {k: v for k, v in m.items() if "✂️clip" in v}
    assert len(clips) == 1 and 95 <= next(iter(clips)) <= 106
    sig = Signals(
        duration=300,
        loudness=[],
        chat=[0.0] * 300,
        chat_kinds={100: {"clip": 1}, 120: {"clip": 1}, 140: {"clip": 2}},
    )
    assert not any("✂️clip" in v for v in heat_markers(sig).values())


def test_chat_kinds_round_trip_through_json():
    sig = _spike({"laugh": 5})
    back = Signals.model_validate_json(sig.model_dump_json())
    assert back.chat_kinds == sig.chat_kinds


def test_dominant_kind_share_boundary_is_40_percent():
    at = heat_markers(_spike({"laugh": 4, "hype": 3, "fail": 3}))
    assert next(x for v in at.values() for x in v).endswith("😂")
    below = heat_markers(_spike({"laugh": 4, "hype": 3, "fail": 3, "shock": 1}))
    assert " " not in next(x for v in below.values() for x in v)


def test_clip_marker_constant_is_shared_with_prompt():
    from hookline.pipeline.select import SYSTEM_PROMPT

    assert CLIP_MARKER == "✂️clip"
    assert CLIP_MARKER in SYSTEM_PROMPT


def test_loud_seconds_has_absolute_floor():
    quiet = [-50.0] * 100
    quiet[10:13] = [-40.0] * 3  # above median + 6 dB, but below -30 dB
    assert loud_seconds(quiet, 6.0) == set()
    ok = [-40.0] * 100
    ok[10:13] = [-20.0] * 3
    assert loud_seconds(ok, 6.0) == {10, 11, 12}


def test_heat_seconds_cover_whole_runs():
    loud = [-40.0] * 100
    loud[20:25] = [-10.0] * 5
    chat = [1.0] * 600
    chat[300:310] = [8.0] * 10
    heat = heat_seconds(Signals(duration=600, loudness=loud + [-40.0] * 500, chat=chat))
    assert set(range(20, 25)) <= heat
    assert 305 in heat and 100 not in heat and 450 not in heat


# --- wave G-select: game-audio transients, visual flashes, scare markers ----------------------


def _env(n_s: int, base: float = -45.0) -> np.ndarray:
    return np.full(n_s * 10, base)  # 0.1 s hop


def test_audio_onset_needs_a_10db_jump_within_half_a_second():
    db = _env(60)
    db[200:215] = -18.0  # sharp: -45 -> -18 at 20.0 s
    db[400:430] = np.linspace(-45, -18, 30)  # a slow 3 s swell: under 5 dB per 0.5 s
    db[430:440] = -18.0
    every, game = audio_onsets(db, 0.1, [])
    assert every == [20] and game == [20]


def test_audio_onset_must_be_loud_against_the_vod():
    db = _env(60)
    db[200:215] = -34.0  # an 11 dB jump, but under median + 12 (-33)
    db[450:460] = -38.0
    assert audio_onsets(db, 0.1, []) == ([], [])
    quiet = _env(60, base=-80.0)
    quiet[200:215] = -50.0  # +30 dB jump, but under the -35 dBFS floor
    assert audio_onsets(quiet, 0.1, []) == ([], [])


def test_onsets_inside_speech_are_not_game_transients():
    db = _env(60)
    db[200:215] = -15.0  # a shout at 20.0 s
    db[300:315] = -15.0  # a stinger at 30.0 s, no words near it
    every, game = audio_onsets(db, 0.1, [(20.05, 20.6)])
    assert every == [20, 30] and game == [30]


def test_flash_needs_a_big_jump_over_the_local_motion():
    motion = [2.0] * 90
    motion[20] = 80.0  # a flash in a calm scene
    motion[40:80] = [45.0] * 40  # busy gameplay: big diffs, but nothing stands out
    motion[60] = 90.0
    assert flash_seconds(motion) == {20}
    assert flash_seconds([1.0] * 30) == set() and flash_seconds([]) == set()


def _sig(**kw) -> Signals:
    base = {"duration": 100, "loudness": [-40.0] * 100, "version": SIGNALS_VERSION}
    return Signals(**{**base, **kw})


def test_scares_are_game_transients_and_flashes_with_a_sound():
    motion = [2.0] * 100
    motion[30] = motion[70] = 90.0
    sig = _sig(motion=motion, onsets=[12, 31], transients=[12])
    # the transient at 12; the flash at 30 has a sound 1 s later; the flash at 70 is silent
    assert scare_seconds(sig) == {12, 30}
    assert scare_seconds(_sig()) == set()  # signals cached before scares existed


def test_scare_marker_once_per_burst_and_in_heat():
    sig = _sig(motion=[2.0] * 100, onsets=[10, 12, 14, 40], transients=[10, 12, 14, 40])
    m = heat_markers(sig)
    assert SCARE_MARKER == "⚡scare"
    assert {k: v for k, v in m.items() if SCARE_MARKER in v} == {
        10: [SCARE_MARKER],
        40: [SCARE_MARKER],
    }
    assert {10, 12, 14, 40} <= heat_seconds(sig)


def test_scare_marker_is_described_in_the_prompt():
    from hookline.pipeline.select import SYSTEM_PROMPT

    assert SCARE_MARKER in SYSTEM_PROMPT


def test_low_motion_spans():
    motion = [1.0] * 20 + [12.0] * 5 + [1.0] * 20
    assert low_motion(motion, 2.0, 15.0)
    assert not low_motion(motion, 15.0, 30.0)
    assert not low_motion(motion, 40.0, 60.0)  # past the end: unknown, not low
    assert not low_motion(None, 2.0, 15.0)


def test_signals_without_new_fields_are_an_old_version():
    old = Signals.model_validate_json('{"duration": 5, "loudness": [-30.0]}')
    assert old.version < SIGNALS_VERSION and old.motion is None and old.transients is None


# --- wave G-select round 1: speech pad, exact onset times -------------------------------------


def test_speech_pad_is_point_three_seconds():
    from hookline.pipeline.signals import SPEECH_PAD_S

    assert SPEECH_PAD_S == 0.3
    db = _env(60)
    db[203:215] = -15.0  # 20.3 s: 0.25 s after a word that ended at 20.05 s
    every, game = audio_onsets(db, 0.1, [(19.5, 20.05)])
    assert every == [20] and game == []


def test_onset_times_are_exact_and_one_per_jump():
    from hookline.pipeline.signals import onset_times

    db = _env(60)
    db[203:215] = -15.0  # 20.3 s, several frames over the jump
    db[507:515] = -15.0  # 50.7 s, inside speech
    every, game = onset_times(db, 0.1, [(50.5, 51.0)])
    assert every == [pytest.approx(20.3), pytest.approx(50.7)]
    assert game == [pytest.approx(20.3)]


def test_scare_times_prefer_exact_onsets():
    from hookline.pipeline.signals import scare_times

    motion = [2.0] * 100
    motion[30] = 90.0
    sig = _sig(
        motion=motion,
        onsets=[12, 31],
        transients=[12],
        onset_times=[12.4, 31.2],
        transient_times=[12.4],
    )
    # the transient at 12.4; the flash at 30 takes its sound's onset time
    assert scare_times(sig) == [pytest.approx(12.4), pytest.approx(31.2)]
    # signals without exact times fall back to the seconds
    old = _sig(motion=motion, onsets=[12, 31], transients=[12])
    assert scare_times(old) == [12.0, 30.0]
