import itertools
import logging
import re

import pytest

from hookline.llm import LLMError
from hookline.models import ClipPick, RubricScores
from hookline.pipeline.select import (
    SELECTION_SCHEMA,
    SYSTEM_PROMPT,
    build_prompt,
    dedupe_overlaps,
    polish_pick,
    select_clips,
    snap_to_sentences,
)

_SCORE_KEYS = ("hook", "context", "payoff", "emotion", "length_fit", "pacing", "loop", "safety")


def raw(start, end, total=50, cold_open=None, **kw):
    """One candidate as the LLM returns it (SELECTION_SCHEMA item)."""
    hook = min(total, 25)
    rest = total - hook
    scores = {"hook": hook, "context": min(rest, 15), "payoff": max(0, rest - 15)}
    scores.update({k: 0 for k in _SCORE_KEYS if k not in scores})
    item = {
        "start": start,
        "end": end,
        "cold_open_start": cold_open[0] if cold_open else None,
        "cold_open_end": cold_open[1] if cold_open else None,
        "title": f"Clip at {start}",
        "hook_text": "You will not believe this",
        "reason": "test",
        "scores": scores,
        "edit_beats": [f"zoom: 'wow' @ {start + 1}s"],
    }
    item.update(kw)
    return item


def make_payload(n, extra=(), cold_open=None):
    """n valid, non-overlapping 30s candidates with descending scores, plus `extra` raw items."""
    clips = [raw(30 + i * 60, 60 + i * 60, total=40 - i, cold_open=cold_open) for i in range(n)]
    return {"clips": clips + list(extra)}


EMPTY = {"clips": []}  # follow-up response with no further candidates


def pick(start=30.0, end=55.0, total=0):
    return ClipPick(
        start=start,
        end=end,
        title="t",
        hook_text="h",
        reason="r",
        scores=RubricScores(
            hook=total, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
        ),
    )


# --- brief tests ---------------------------------------------------------------------------


def test_select_returns_top_n_sorted_by_start(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))  # 8 valid non-overlapping candidates, descending scores
    picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 5 and [p.start for p in picks] == sorted(p.start for p in picks)
    assert fake_llm.calls[0]["model"] == "claude-opus-5-5"


def test_snap_moves_to_sentence_bounds(long_sentences):
    p = snap_to_sentences(pick(start=31.2, end=54.1), long_sentences, duration=600)
    assert p.start in {s.start for s in long_sentences} and p.end in {s.end for s in long_sentences}


def test_impossible_timestamps_are_dropped(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(6, extra=[raw(start=50, end=40), raw(start=9999, end=10020)]))
    picks = select_clips(long_transcript, fake_llm, settings)
    assert all(p.end > p.start and p.end <= long_transcript.duration for p in picks)


def test_overlaps_deduped_keeping_higher_score():
    a, b = pick(0, 40, total=90), pick(10, 45, total=70)
    assert dedupe_overlaps([b, a]) == [a]


def test_retries_once_then_raises(long_transcript, fake_llm, settings):
    fake_llm.queue(LLMError("boom"), {"clips": "garbage"})
    with pytest.raises(LLMError):
        select_clips(long_transcript, fake_llm, settings)
    assert len(fake_llm.calls) == 2


def test_retry_recovers(long_transcript, fake_llm, settings):
    fake_llm.queue(LLMError("boom"), make_payload(8))
    assert len(select_clips(long_transcript, fake_llm, settings)) == 5


def test_short_input_skips_llm(short_transcript, fake_llm, settings):
    assert len(select_clips(short_transcript, fake_llm, settings)) == 1 and fake_llm.calls == []


def test_chunking_when_over_budget(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue_forever(make_payload(5))
    select_clips(long_transcript, fake_llm, s)
    assert len(fake_llm.calls) > 1


def test_bad_cold_open_is_removed(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8, cold_open=(0.0, 10.0)))  # outside clip / too long
    assert all(p.cold_open is None for p in select_clips(long_transcript, fake_llm, settings))


# --- additional coverage -------------------------------------------------------------------


def test_requests_clips_plus_three_candidates(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings)
    call = fake_llm.calls[0]
    assert "Return the 8 best clip candidates" in call["prompt"]
    assert call["system"] == SYSTEM_PROMPT
    assert call["schema"] == SELECTION_SCHEMA


def test_returns_highest_scoring_picks(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    picks = select_clips(long_transcript, fake_llm, settings)
    # make_payload scores descend with start, so the 5 best are the first 5 by start
    assert [round(p.start) for p in picks] == [30, 90, 150, 210, 270]


def test_valid_cold_open_is_kept(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(45.0, 48.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open == (45.0, 47.3)  # snapped: line at 45.0 ends at 47.3


def test_cold_open_longer_than_limit_is_removed(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(39.0, 45.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open is None


def test_length_out_of_range_dropped(long_transcript, fake_llm, settings):
    too_short, too_long, ok = raw(30, 40), raw(100, 190), raw(300, 330)
    fake_llm.queue({"clips": [too_short, too_long, ok]}, EMPTY)
    picks = select_clips(long_transcript, fake_llm, settings)
    assert [round(p.start) for p in picks] == [300]


def test_scores_are_clamped(long_transcript, fake_llm, settings):
    item = raw(30, 60)
    item["scores"] = {k: 99 for k in _SCORE_KEYS} | {"safety": -4}
    fake_llm.queue({"clips": [item]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.scores.model_dump(exclude={"total"}) == {
        "hook": 25,
        "context": 15,
        "payoff": 15,
        "emotion": 15,
        "length_fit": 10,
        "pacing": 10,
        "loop": 5,
        "safety": 0,
    }


def test_fewer_than_n_logs_warning(long_transcript, fake_llm, settings, caplog):
    fake_llm.queue(make_payload(2), {"clips": [raw(300, 330)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 3 and len(fake_llm.calls) == 2
    assert "only 3 valid clip(s) selected (wanted 5)" in caplog.text


def test_chunks_cover_whole_transcript_with_overlap(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue_forever(make_payload(5))
    select_clips(long_transcript, fake_llm, s)
    prompts = [c["prompt"] for c in fake_llm.calls]
    assert "line0 says" in prompts[0] and "line199 says" in prompts[-1]
    # each chunk after the first repeats the previous chunk's final line (2-minute overlap)
    for prev, cur in itertools.pairwise(prompts):
        last_prev = re.findall(r"\] line\d+ says", prev)[-1]
        assert last_prev in cur and cur.index(last_prev) < cur.rindex("] line")
    assert all(f"Return the {s.clips} best clip candidates" in p for p in prompts)
    assert "This is an excerpt covering 00:00.0 to " in prompts[0]
    assert re.search(r"excerpt covering \S+ to 09:59\.3 of a longer video", prompts[-1])


def test_build_prompt_contents():
    p = build_prompt("[00:01.0] hello there.", n=8, duration=3723.0, exclude=[(30.0, 60.0)])
    assert "[00:01.0] hello there." in p and "Return the 8 best clip candidates" in p
    assert "Video duration: 1:02:03.0 (3723.0 seconds)." in p
    assert "excerpt" not in p
    assert "do not pick moments overlapping these ranges" in p.lower()
    assert "00:30.0" in p and "01:00.0" in p
    assert "overlapping" not in build_prompt("x", n=5, duration=100.0, exclude=[]).lower()


def test_exclude_reaches_prompt(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings, exclude=[(120.0, 150.0)])
    assert "02:00.0" in fake_llm.calls[0]["prompt"]


def test_snap_clamps_and_falls_back(long_sentences):
    p = snap_to_sentences(pick(start=-5.0, end=700.0), long_sentences, duration=600)
    assert p.start == long_sentences[0].start and p.end == long_sentences[-1].end
    p = snap_to_sentences(pick(start=10.0, end=900.0), long_sentences, duration=500)
    assert p.end == 500


def test_dedupe_keeps_small_overlaps():
    a, b = pick(0, 40, total=90), pick(30, 70, total=70)  # 10s / 40s = 25% overlap
    assert dedupe_overlaps([b, a]) == [a, b]


def test_polish_pick_covers_whole_input(short_transcript):
    p = polish_pick(short_transcript)
    assert (p.start, p.end) == (0.0, short_transcript.duration)
    assert p.hook_text == "line0 says something here."
    assert p.scores.total == 0


def test_polish_pick_truncates_hook_to_eight_words():
    from hookline.models import Segment, Transcript, Word

    texts = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten."]
    words = [Word(start=i, end=i + 0.5, text=t) for i, t in enumerate(texts)]
    tr = Transcript(
        language="en",
        duration=30.0,
        segments=[Segment(start=0, end=9.5, text=" ".join(texts), words=words)],
    )
    assert polish_pick(tr).hook_text == "one two three four five six seven eight"


def test_schema_is_strict_and_has_no_numeric_bounds():
    def walk(node):
        if isinstance(node, dict):
            assert not {"minimum", "maximum", "minItems", "maxItems", "pattern"} & node.keys()
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(SELECTION_SCHEMA)
    item = SELECTION_SCHEMA["properties"]["clips"]["items"]
    assert set(item["properties"]["scores"]["properties"]) == set(_SCORE_KEYS)
    nullable = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert item["properties"]["cold_open_start"] == nullable
    assert item["properties"]["cold_open_end"] == nullable


def test_system_prompt_covers_brief():
    for needle in (
        (
            "Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · "
            "Length fit 10 · Pacing 10 · Loop potential 5 · Platform safety 5"
        ),
        "20–45s usually wins; 60s only if tension holds",
        "first 3 seconds",
        "cold_open",
        "8 words",
        "best first",
        "timestamp of the line after",
        "short phrase or sentence",
        "at most 5 seconds",
        '"And", "So", "But", "Because", "Which"',
        "introduces the subject",
        "score `hook` and `context` lower",
        "⟨🔥chat×4 🔊loud⟩",
        "audience and audio reactions",
        "not a requirement",
    ):
        assert needle in SYSTEM_PROMPT


def test_snap_end_given_as_next_line_start_adds_no_extra_sentence(long_sentences):
    # sentences start every 3s and last 2.3s: [30.0-32.3], [33.0-35.3], ...
    p = snap_to_sentences(pick(start=30.0, end=33.0), long_sentences, duration=600)
    assert (p.start, p.end) == (30.0, 32.3)


def test_snap_end_mid_sentence_extends_to_sentence_end(long_sentences):
    p = snap_to_sentences(pick(start=30.0, end=31.0), long_sentences, duration=600)
    assert p.end == 32.3


def test_snap_end_before_any_sentence_uses_first_sentence_from_start(long_sentences):
    late = [s for s in long_sentences if s.start >= 300]
    p = snap_to_sentences(pick(start=100.0, end=200.0), late, duration=600)
    assert p.end == late[0].end


def test_picks_overlapping_exclude_are_dropped(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    picks = select_clips(long_transcript, fake_llm, settings, exclude=[(100.0, 110.0)])
    assert len(picks) == 5 and all(not (p.start < 110.0 and p.end > 100.0) for p in picks)
    assert 90 not in [round(p.start) for p in picks]


def test_pick_touching_exclude_edge_is_kept(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60)]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings, exclude=[(0.0, 30.0)])
    assert p.start == 30.0


def test_chunk_failure_names_the_chunk(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue(make_payload(5), LLMError("boom"), LLMError("boom"))
    with pytest.raises(LLMError, match=r"chunk 2/\d+"):
        select_clips(long_transcript, fake_llm, s)


def test_build_prompt_excerpt():
    p = build_prompt("x", n=5, duration=7200.0, exclude=[], excerpt=(3600.0, 5400.0))
    assert "This is an excerpt covering 1:00:00.0 to 1:30:00.0 of a longer video." in p


def test_cold_open_teasing_final_line_is_snapped_and_kept(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(57.0, 60.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert (p.start, p.end) == (30.0, 59.3)
    assert p.cold_open == (57.0, 59.3)  # ends at the final sentence's end, not mid-gap


def test_cold_open_snapped_longer_than_limit_is_removed(long_transcript, fake_llm, settings):
    # 45.0-51.0 snaps to 45.0-50.3 (5.3s)
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(45.0, 51.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open is None


def test_cold_open_snaps_to_words_and_allows_5s(long_transcript, fake_llm, settings):
    # words: 45.0-45.5, 45.6-46.1, 46.2-46.7, 46.8-47.3, 48.0-48.5 ...
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(45.5, 48.4))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open == (45.6, 48.5)  # mid-sentence word boundaries, 2.9s
    fake_llm.queue({"clips": [raw(30, 60, cold_open=(45.0, 50.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open == (45.0, 49.7)  # 4.7s: more than the old 3.5s cap


def test_cold_open_pushing_over_max_length_is_dropped_pick_kept(
    long_transcript, fake_llm, settings
):
    # main clip 30.0-104.3 = 74.3s (<= 75s); with a 2.3s cold open it would be 76.6s
    fake_llm.queue({"clips": [raw(30, 103, cold_open=(57.0, 60.0))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert (p.start, p.end, p.cold_open) == (30.0, 104.3, None)


def test_all_invalid_first_response_triggers_follow_up(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(50, 40), raw(9999, 10020)]}, make_payload(8))
    picks = select_clips(long_transcript, fake_llm, settings, exclude=[(500.0, 520.0)])
    assert len(picks) == 5 and len(fake_llm.calls) == 2
    follow_up = fake_llm.calls[1]["prompt"]
    assert "Return the 8 best clip candidates" in follow_up
    assert "- 08:20.0 to 08:40.0" in follow_up


def test_follow_up_fills_missing_picks(long_transcript, fake_llm, settings):
    more = {"clips": [raw(200 + i * 60, 230 + i * 60, total=30 - i) for i in range(4)]}
    fake_llm.queue(make_payload(2), more)
    picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 5 and len(fake_llm.calls) == 2
    follow_up = fake_llm.calls[1]["prompt"]
    assert "Return the 6 best clip candidates" in follow_up  # 3 missing + 3
    assert "- 00:30.0 to 00:59.3" in follow_up and "- 01:30.0 to 01:59.3" in follow_up


def test_follow_up_failure_keeps_first_picks(long_transcript, fake_llm, settings, caplog):
    fake_llm.queue(make_payload(2), LLMError("boom"), LLMError("boom"))
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 2 and "only 2 valid clip(s) selected (wanted 5)" in caplog.text


def test_enough_picks_makes_no_follow_up(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(5))
    assert len(select_clips(long_transcript, fake_llm, settings)) == 5
    assert len(fake_llm.calls) == 1


def _cold_open_warnings(caplog, llm, transcript, settings, item):
    llm.queue({"clips": [item]}, EMPTY)
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        (p,) = select_clips(transcript, llm, settings)
    assert p.cold_open is None
    return [r.getMessage() for r in caplog.records if "dropped cold open" in r.getMessage()]


def test_dropped_cold_open_outside_clip_logs_reason(long_transcript, fake_llm, settings, caplog):
    msgs = _cold_open_warnings(
        caplog, fake_llm, long_transcript, settings, raw(30, 60, cold_open=(0.0, 2.0))
    )
    assert len(msgs) == 1 and "Clip at 30" in msgs[0] and "outside the clip" in msgs[0]


def test_dropped_cold_open_too_long_logs_reason(long_transcript, fake_llm, settings, caplog):
    msgs = _cold_open_warnings(
        caplog, fake_llm, long_transcript, settings, raw(30, 60, cold_open=(45.0, 51.0))
    )
    assert len(msgs) == 1 and "Clip at 30" in msgs[0] and "exceeds the 5.0s cap" in msgs[0]


def test_dropped_cold_open_over_max_length_logs_reason(long_transcript, fake_llm, settings, caplog):
    msgs = _cold_open_warnings_keep(
        caplog, fake_llm, long_transcript, settings, raw(30, 103, cold_open=(57.0, 60.0))
    )
    assert len(msgs) == 1 and "maximum length" in msgs[0]


def _cold_open_warnings_keep(caplog, llm, transcript, settings, item):
    llm.queue({"clips": [item]}, EMPTY)
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        select_clips(transcript, llm, settings)
    return [r.getMessage() for r in caplog.records if "dropped cold open" in r.getMessage()]


def test_raw_sink_receives_every_payload(long_transcript, fake_llm, settings):
    first = {"clips": [raw(30, 60)]}
    fake_llm.queue(first, EMPTY)
    sink: list[dict] = []
    select_clips(long_transcript, fake_llm, settings, raw_sink=sink)
    assert sink == [first, EMPTY]


def test_cold_open_without_whole_words_logs_reason(long_transcript, fake_llm, settings, caplog):
    # 46.75-46.78 falls between words: snapped start (46.8) is after snapped end (46.7)
    msgs = _cold_open_warnings(
        caplog, fake_llm, long_transcript, settings, raw(30, 60, cold_open=(46.75, 46.78))
    )
    assert len(msgs) == 1 and "no whole words fit" in msgs[0]


def test_markers_reach_the_prompt(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings, markers={31: ["🔊loud"]})
    assert "⟨🔊loud⟩" in fake_llm.calls[0]["prompt"]


def test_markers_reach_chunk_prompts(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue_forever(make_payload(5))
    select_clips(long_transcript, fake_llm, s, markers={31: ["🔊loud"], 400: ["🔊loud"]})
    assert len(fake_llm.calls) > 1
    assert sum("⟨🔊loud⟩" in c["prompt"] for c in fake_llm.calls) >= 2


def test_chunk_prompts_keep_markers_on_their_own_lines(long_transcript, fake_llm, settings):
    # a marker at the very end belongs to the transcript's last sentence only; a chunk ending
    # earlier must not pull it onto its own last line
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue_forever(make_payload(5))
    select_clips(long_transcript, fake_llm, s, markers={599: ["🔥end"]})
    prompts = [c["prompt"] for c in fake_llm.calls]
    assert len(prompts) > 1
    assert all("🔥end" not in p for p in prompts[:-1]) and "🔥end" in prompts[-1]
