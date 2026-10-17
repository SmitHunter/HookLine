from hookline.pipeline.signals import (
    CLIP_MARKER,
    Signals,
    heat_markers,
    heat_seconds,
    loud_seconds,
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
