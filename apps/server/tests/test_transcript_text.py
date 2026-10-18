import pytest

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
    assert out.splitlines() == ["L0 [01:05.2] Hello there.", "L1 [1:02:05.0] Late."]


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
    assert out == "L0 [00:50.0] No way. ⟨🔊loud⟩"


def test_compact_marker_window_is_half_open_and_ordered_deduped():
    sents = [Sentence(start=10, end=12, text="A."), Sentence(start=12, end=14, text="B.")]
    out = compact_transcript(sents, {10: ["🔥chat×4"], 11: ["🔊loud", "🔥chat×4"], 12: ["🔊loud"]})
    assert out == "L0 [00:10.0] A. ⟨🔥chat×4 🔊loud⟩\nL1 [00:12.0] B. ⟨🔊loud⟩"


def test_compact_no_markers_unchanged():
    assert compact_transcript([Sentence(start=1, end=2, text="Hi.")], {}) == "L0 [00:01.0] Hi."


def test_compact_pause_markers_attach_to_next_sentence():
    sents = [Sentence(start=10, end=12, text="A."), Sentence(start=15, end=17, text="B.")]
    out = compact_transcript(sents, {11: ["x"], 13: ["pause"], 20: ["late"]})
    assert out == "L0 [00:10.0] A. ⟨x⟩\nL1 [00:15.0] B. ⟨pause late⟩"


def test_compact_markers_before_first_sentence_attach_to_it():
    sents = [Sentence(start=10, end=12, text="A.")]
    assert compact_transcript(sents, {3: ["early"]}) == "L0 [00:10.0] A. ⟨early⟩"


def test_compact_lines_are_numbered_by_sentence_index():
    sents = [Sentence(start=i * 3, end=i * 3 + 2, text=f"S{i}.") for i in range(12)]
    lines = compact_transcript(sents).splitlines()
    assert lines[0] == "L0 [00:00.0] S0." and lines[11] == "L11 [00:33.0] S11."


def test_clip_pick_new_fields_default_so_old_caches_validate(sample_pick):
    new = {"first_words", "last_words", "hook_type", "verdict", "post_caption"}
    new |= {"hashtags", "search_title", "category", "comment_risk"}
    old = sample_pick.model_dump(mode="json", exclude=new)
    back = ClipPick.model_validate(old)
    assert back.first_words == "" and back.hashtags == [] and back.post_caption == ""
    assert back.hook_type is None and back.comment_risk is None and back.verdict == "maybe"
    assert new <= set(back.model_dump())


# --- selection lines: terminal punctuation only, pause split only for 25 s runs -------------


def test_selection_lines_ignore_pauses_inside_a_sentence():
    from hookline.transcript_text import selection_lines

    words = [w(0, 0.3, "So"), w(1.5, 1.8, "I"), w(1.9, 2.2, "think"), w(3.5, 3.8, "to")]
    words += [w(3.9, 4.3, "win."), w(4.5, 4.9, "Next"), w(5.0, 5.4, "one!")]
    assert [s.text for s in selection_lines(words)] == ["So I think to win.", "Next one!"]


def test_selection_lines_split_on_closing_quotes():
    from hookline.transcript_text import selection_lines

    words = [w(0, 0.4, "He"), w(0.5, 0.9, 'said "stop."'), w(1.0, 1.4, "Next?”")]
    words += [w(1.5, 1.9, "More")]
    assert [s.text for s in selection_lines(words)] == ['He said "stop."', "Next?”", "More"]


def test_selection_lines_split_a_long_unpunctuated_run_at_its_longest_gap():
    from hookline.transcript_text import selection_lines

    # 0.5 s words every 0.6 s, except a 2 s gap after word 20 (ends 12.5, next starts 14.5)
    words = [w(round(i * 0.6, 2), round(i * 0.6 + 0.5, 2), f"w{i}") for i in range(21)]
    words += [
        w(round(14.5 + i * 0.6, 2), round(14.5 + i * 0.6 + 0.5, 2), f"v{i}") for i in range(30)
    ]
    lines = selection_lines(words)
    assert lines[0].text.split()[-1] == "w20" and lines[0].end == 12.5
    assert lines[1].start == 14.5
    assert all(s.end - s.start <= 25.0 for s in lines)
    assert sum(len(s.text.split()) for s in lines) == len(words)


def test_selection_lines_cap_every_run_at_25s():
    from hookline.transcript_text import selection_lines

    words = [w(i, i + 0.5, "yap") for i in range(80)]  # uniform gaps
    lines = selection_lines(words)
    assert all(s.end - s.start <= 25.0 for s in lines) and len(lines) >= 4
    assert selection_lines([]) == []


# --- ellipses, abbreviations, lightly punctuated (Twitch) speech ----------------------------------


def _texts(words):
    from hookline.transcript_text import selection_lines

    return [s.text for s in selection_lines(words)]


@pytest.mark.parametrize("trail", ["going...", "going…"])
def test_selection_lines_ellipsis_is_not_terminal(trail):
    words = [w(0, 0.3, "I"), w(0.4, 0.7, "was"), w(0.8, 1.1, trail), w(2.0, 2.3, "home.")]
    assert _texts(words) == [f"I was {trail} home."]


@pytest.mark.parametrize(
    "abbr", ["Mr.", "Mrs.", "Ms.", "Dr.", "St.", "vs.", "etc.", "e.g.", "i.e."]
)
def test_selection_lines_abbreviations_are_not_terminal(abbr):
    words = [w(0, 0.3, "Ask"), w(0.4, 0.7, abbr), w(1.2, 1.5, "Smith"), w(1.6, 1.9, "now.")]
    assert _texts(words) == [f"Ask {abbr} Smith now."]


def test_dotted_acronym_ends_a_sentence_only_before_a_capital_after_a_pause():
    def run(nxt, gap):
        words = [w(0, 0.3, "in"), w(0.4, 0.8, "the"), w(0.9, 1.3, "U.S.")]
        return _texts([*words, w(1.3 + gap, 1.6 + gap, nxt), w(1.7 + gap, 2.0 + gap, "go.")])

    assert run("And", 0.5) == ["in the U.S.", "And go."]
    assert run("And", 0.1) == ["in the U.S. And go."]
    assert run("economy", 0.5) == ["in the U.S. economy go."]


def _stream(seconds=60.0):
    """Unpunctuated chatter: 0.25 s words every 0.35 s, with a 1.2 s pause every 3-8 s."""
    words, t, k, since = [], 0.0, 0, 0.0
    pauses = [3.0, 5.5, 8.0, 4.0, 6.5, 3.5, 7.0, 5.0]
    while t < seconds:
        words.append(w(round(t, 3), round(t + 0.25, 3), f"word{len(words)}"))
        t += 0.35
        since += 0.35
        if since >= pauses[k % len(pauses)]:
            t += 1.2
            since, k = 0.0, k + 1
    return words


def test_unpunctuated_stream_splits_into_10_to_20s_lines():
    from hookline.transcript_text import selection_lines

    lines = selection_lines(_stream())
    lengths = [s.end - s.start for s in lines[:-1]]  # the last line is whatever is left
    assert all(d <= 25.0 for d in lengths) and len(lengths) >= 2
    assert sum(10.0 <= d <= 20.0 for d in lengths) / len(lengths) >= 0.8, lengths


def test_soft_split_never_cuts_at_a_short_gap():
    from hookline.transcript_text import selection_lines

    words = [w(round(i * 0.6, 2), round(i * 0.6 + 0.5, 2), f"w{i}") for i in range(30)]  # 18 s
    assert len(selection_lines(words)) == 1  # gaps of 0.1 s: under 25 s nothing splits
