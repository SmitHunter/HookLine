from hookline.models import ClipPick, Sentence, Word
from hookline.transcript_text import compact_transcript, format_timestamp, split_sentences


def w(s, e, t):
    return Word(start=s, end=e, text=t)


def test_split_on_punctuation():
    words = [w(0, 0.4, "Hi."), w(0.5, 0.9, "How"), w(1, 1.3, "are"), w(1.4, 1.8, "you?")]
    assert [s.text for s in split_sentences(words)] == ["Hi.", "How are you?"]


def test_split_falls_back_to_pauses_without_punctuation():
    words = [
        w(0, 0.4, "no"),
        w(0.5, 0.9, "punctuation"),
        w(2.0, 2.4, "here"),
        w(2.5, 2.9, "either"),
    ]
    sents = split_sentences(words)
    assert [s.text for s in sents] == ["no punctuation", "here either"]
    assert (sents[1].start, sents[1].end) == (2.0, 2.9)


def test_split_caps_runaway_sentences():
    words = [w(i, i + 0.5, "yap") for i in range(60)]  # 60s, no punctuation, gaps 0.5
    assert all(s.end - s.start <= 25.0 for s in split_sentences(words))


def test_split_ignores_trailing_quotes_and_brackets():
    words = [w(0, 0.4, 'said "stop."'), w(0.5, 0.9, "Next")]
    assert [s.text for s in split_sentences(words)] == ['said "stop."', "Next"]


def test_split_empty():
    assert split_sentences([]) == []


def test_compact_format_and_hours():
    out = compact_transcript(
        [
            Sentence(start=65.25, end=70, text="Hello there."),
            Sentence(start=3725.0, end=3726, text="Late."),
        ]
    )
    assert out.splitlines() == ["[01:05.2] Hello there.", "[1:02:05.0] Late."]


def test_format_timestamp_is_public():
    assert format_timestamp(65.25) == "01:05.2"
    assert format_timestamp(3725.0) == "1:02:05.0"


def test_clip_pick_segments_put_cold_open_first(sample_pick):
    p = sample_pick.model_copy(update={"cold_open": (50.0, 52.5)})
    assert p.segments() == [(50.0, 52.5), (30.0, 55.0)]


def test_clip_pick_without_cold_open_and_duration(sample_pick):
    assert sample_pick.segments() == [(30.0, 55.0)]
    assert sample_pick.duration == 25.0


def test_clip_pick_json_roundtrip_with_total(sample_pick):
    p = sample_pick.model_copy(update={"cold_open": (50.0, 52.5)})
    dumped = p.model_dump()
    assert dumped["scores"]["total"] == 0
    back = ClipPick.model_validate_json(p.model_dump_json())
    assert back.cold_open == (50.0, 52.5)
    assert back == p


def test_compact_appends_markers():
    out = compact_transcript([Sentence(start=50, end=53, text="No way.")], {51: ["🔊loud"]})
    assert out == "[00:50.0] No way. ⟨🔊loud⟩"


def test_compact_marker_window_is_half_open_and_ordered_deduped():
    sents = [Sentence(start=10, end=12, text="A."), Sentence(start=12, end=14, text="B.")]
    out = compact_transcript(sents, {10: ["🔥chat×4"], 11: ["🔊loud", "🔥chat×4"], 12: ["🔊loud"]})
    assert out == "[00:10.0] A. ⟨🔥chat×4 🔊loud⟩\n[00:12.0] B. ⟨🔊loud⟩"


def test_compact_no_markers_unchanged():
    assert compact_transcript([Sentence(start=1, end=2, text="Hi.")], {}) == "[00:01.0] Hi."
