from hookline.pipeline.signals import Signals, heat_markers


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
