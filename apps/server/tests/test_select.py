import itertools
import logging
import math
import re

import pytest

from hookline.llm import LLMError
from hookline.models import ClipPick, RubricScores, Segment, Transcript, Word
from hookline.pipeline.select import (
    SELECTION_SCHEMA,
    SYSTEM_PROMPT,
    NoClipsSelectedError,
    build_prompt,
    dedupe_overlaps,
    polish_pick,
    select_clips,
)

_SCORE_KEYS = ("hook", "context", "payoff", "emotion", "length_fit", "pacing", "loop", "safety")


def raw(start, end, total=50, cold_open_words=None, verdict="post", **kw):
    """One candidate as the LLM returns it (SELECTION_SCHEMA item). `start`/`end` are seconds
    on the fixtures' 3 s sentence grid (sentence n starts at 3n): start_line is the sentence
    starting at `start`, end_line the last sentence starting before `end`."""
    hook = min(total, 25)
    rest = total - hook
    scores = {"hook": hook, "context": min(rest, 15), "payoff": max(0, rest - 15)}
    scores.update({k: 0 for k in _SCORE_KEYS if k not in scores})
    start_line, end_line = round(start / 3), math.ceil(end / 3) - 1
    item = {
        "start_line": start_line,
        "end_line": end_line,
        "cold_open_words": cold_open_words,
        "first_words": f"line{start_line} says something here.",
        "last_words": f"line{end_line} says something here.",
        "hook_type": "bold_claim",
        "verdict": verdict,
        "title": f"Clip at {start}",
        "hook_text": "Three hundred covers at nineteen",
        "reason": "test",
        "scores": scores,
        "post_caption": "He ran the whole kitchen alone. Could you?",
        "hashtags": ["linecook", "kitchenlife", "chefstories"],
        "search_title": "Running a restaurant kitchen alone at 19",
        "category": "food",
        "comment_risk": "low",
        "edit_beats": [f"zoom: 'says' @ {start + 1}s"],
    }
    item.update(kw)
    return item


def make_payload(n, extra=(), cold_open_words=None):
    """n valid, non-overlapping 30s candidates with descending scores, plus `extra` raw items."""
    clips = [
        raw(30 + i * 60, 60 + i * 60, total=40 - i, cold_open_words=cold_open_words)
        for i in range(n)
    ]
    return {"clips": clips + list(extra)}


EMPTY = {"clips": []}  # follow-up response with no further candidates


def _L(n):  # the verbatim text of long_transcript line n
    return f"line{n} says something here."


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
    fake_llm.queue(make_payload(8, cold_open_words=_L(0)))  # outside every clip
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
    fake_llm.queue({"clips": [raw(30, 60, cold_open_words=_L(15))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open == (45.0, 47.3)  # the whole sentence L15


def test_length_out_of_range_dropped(long_transcript, fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    too_short, too_long, ok = raw(30, 40), raw(100, 190), raw(300, 330)
    fake_llm.queue({"clips": [too_short, too_long, ok]}, EMPTY)
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.select"):
        picks = select_clips(long_transcript, fake_llm, settings)
    # the 92 s candidate is trimmed from the front (L33 -> L44, 59.3 s); the 11 s one is dropped
    assert [round(p.start) for p in picks] == [132, 300]
    assert "dropped clip 'Clip at 30': 11.3s is outside 15.0-75.0s" in caplog.text
    assert "trimmed from the front to fit (92.3s -> 59.3s)" in caplog.text


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


def test_dedupe_keeps_small_overlaps():
    a, b = pick(0, 40, total=90), pick(30, 70, total=70)  # 10s / 40s = 25% overlap
    assert dedupe_overlaps([b, a]) == [a, b]


def test_polish_pick_covers_whole_input(short_transcript):
    p = polish_pick(short_transcript)
    assert (p.start, p.end) == (0.0, short_transcript.duration)
    assert p.hook_text == "line0 says something here."
    assert p.scores.total == 0


def test_polish_pick_truncates_hook_to_eight_words():
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
    props = item["properties"]
    assert set(props["scores"]["properties"]) == set(_SCORE_KEYS)
    assert props["start_line"] == {"type": "integer"} and props["end_line"] == {"type": "integer"}
    assert props["cold_open_words"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "cold_open_line" not in props
    assert props["hook_type"]["enum"] == [
        "bold_claim",
        "surprising_fact",
        "conflict",
        "confession",
        "question",
        "in_progress_action",
        "reaction_peak",
        "story_peak",
    ]
    assert props["verdict"]["enum"] == ["post", "maybe", "skip"]
    assert props["comment_risk"]["enum"] == ["low", "medium", "high"]
    assert props["hashtags"] == {"type": "array", "items": {"type": "string"}}
    for k in ("first_words", "last_words", "post_caption", "category"):
        assert props[k] == {"type": "string"}
    assert props["search_title"]["type"] == "string"  # plus a description of its limit
    assert not {"start", "end", "cold_open_start", "cold_open_end"} & props.keys()


def test_system_prompt_covers_brief():
    for needle in (
        (
            "Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · "
            "Length fit 10 · Pacing 10 · Loop potential 5 · Platform safety 5"
        ),
        # line-indexed transcript and picks
        "L{n}",
        "`start_line`",
        "`end_line`",
        "`cold_open_words`",
        "at most 6 seconds",
        # feed editor
        "feed editor, not a documentary editor",
        "stranger mid-scroll",
        # first_words
        "`first_words`",
        "verbatim",
        "move the start or use a cold open",
        # hook archetypes, one idea
        "bold_claim",
        "surprising_fact",
        "conflict",
        "confession",
        "question",
        "in_progress_action",
        "reaction_peak",
        "story_peak",
        "one idea per clip",
        # ending
        "payoff line",
        '"yeah"',
        '"mm-hmm"',
        "backchannel",
        "echo the opening",
        # length
        "20–40 seconds",
        # hook_text
        "3–7 words",
        "literally true",
        '"Then this"',
        '"Wait for it"',
        '"You won\'t believe"',
        '"Watch till the end"',
        # hot takes
        "hot takes",
        "slurs, sexual content and gore",
        # calibration
        "`hook` 20 or higher only if",
        "1–4 postable clips",
        "`skip`",
        # post copy
        "`post_caption`",
        "ends with a question",
        "#fyp",
        "`search_title`",
        "searchable topic",
        # live streams
        "dead air",
        "3–8 seconds earlier",
        "answering chat",
        # comment risk
        "`comment_risk`",
        "appearance",
        # markers stay explained
        "⟨🔥chat×4 🔊loud⟩",
        "😂",
        "✂️clip",
        "not a requirement",
        # review round 1
        "a conversation the viewer never saw",
        "The cause line is then your `first_words`",
        "must still work cold",
        "Everything in packaging must be literally true to the clip",
        "A good pick (37 seconds)",
    ):
        assert needle in SYSTEM_PROMPT, needle
    assert not re.search(r"\b[A-Z]{4,}\b", SYSTEM_PROMPT)  # calm brief, no shouting


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


def test_cold_open_teasing_final_line_is_kept(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60, cold_open_words=_L(19))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert (p.start, p.end) == (30.0, 59.3)
    assert p.cold_open == (57.0, 59.3)


def test_cold_open_on_the_first_line_is_dropped(long_transcript, fake_llm, settings):
    fake_llm.queue({"clips": [raw(30, 60, cold_open_words=_L(10))]}, EMPTY)
    (p,) = select_clips(long_transcript, fake_llm, settings)
    assert p.cold_open is None


def _long_sentence_transcript():
    """Sentence 0..9: 4 short words each (3 s grid); sentence 10 at 30 s is 16 words of
    0.5 s with 0.1 s gaps (30.0-39.5); then short sentences again from 42 s."""
    segments = []
    for n in range(40):
        if n == 10:
            texts = [f"w{j}" for j in range(15)] + ["end."]
            t0, step = 30.0, 0.6
        else:
            texts = [f"line{n}", "says", "something", "here."]
            t0, step = (3.0 * n if n < 10 else 42.0 + 3.0 * (n - 11)), 0.6
        words = [
            Word(start=round(t0 + j * step, 3), end=round(t0 + j * step + 0.5, 3), text=t)
            for j, t in enumerate(texts)
        ]
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=" ".join(texts), words=words)
        )
    return Transcript(language="en", duration=200.0, segments=segments)


def test_cold_open_that_cannot_end_a_sentence_within_6s_is_dropped(fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    tr = _long_sentence_transcript()  # L10: 30.0-39.5, its only terminal word is "end."
    item = raw(0, 60, cold_open_words="w0 w1 w2 w3", first_words="line0 says something here.")
    item["last_words"] = "line15 says something here."
    item["end_line"] = 15  # L15 starts at 54.0 and ends at 56.3
    fake_llm.queue({"clips": [item]}, EMPTY)
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        (p,) = select_clips(tr, fake_llm, settings)
    assert (p.start, p.end, p.cold_open) == (0.0, 56.3, None)
    assert "dropped cold open" in caplog.text and "6" in caplog.text


def test_cold_open_pushing_over_max_length_is_dropped_pick_kept(
    long_transcript, fake_llm, settings
):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    # main clip 30.0-104.3 = 74.3s: the teaser on L11 leaves no later start, so it stays
    # untrimmed (<= 75 s); with the 2.3 s cold open it would be 76.6 s
    fake_llm.queue({"clips": [raw(30, 103, cold_open_words=_L(11))]}, EMPTY)
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
        caplog, fake_llm, long_transcript, settings, raw(30, 60, cold_open_words=_L(0))
    )
    assert len(msgs) == 1 and "Clip at 30" in msgs[0] and "not found in the clip" in msgs[0]


def test_dropped_cold_open_not_in_transcript_logs_reason(
    long_transcript, fake_llm, settings, caplog
):
    msgs = _cold_open_warnings(
        caplog, fake_llm, long_transcript, settings, raw(30, 60, cold_open_words="nobody said this")
    )
    assert len(msgs) == 1 and "not found in the clip" in msgs[0]


def test_dropped_cold_open_over_max_length_logs_reason(long_transcript, fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    msgs = _cold_open_warnings_keep(
        caplog, fake_llm, long_transcript, settings, raw(30, 103, cold_open_words=_L(11))
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


# --- selection v2: line-indexed picks, verdicts, post copy ----------------------------------


def _one(transcript, llm, settings, item, caplog=None):
    llm.queue({"clips": [item]}, EMPTY)
    if caplog is None:
        return select_clips(transcript, llm, settings)
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.select"):
        return select_clips(transcript, llm, settings)


def test_lines_map_to_sentence_times(long_transcript, fake_llm, settings):
    item = raw(0, 0, start_line=10, end_line=19, first_words="line10 says something here.")
    (p,) = _one(long_transcript, fake_llm, settings, item)
    assert (p.start, p.end) == (30.0, 59.3)  # L10 starts at 30.0, L19 ends at 59.3


def test_prompt_lines_are_numbered(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings)
    prompt = fake_llm.calls[0]["prompt"]
    assert "L0 [00:00.0] line0 says something here." in prompt
    assert "L199 [09:57.0] line199 says something here." in prompt


def test_out_of_range_lines_fall_back_to_nearest_and_log_once(
    long_transcript, fake_llm, settings, caplog
):
    item = raw(0, 0, start_line=-3, end_line=9, first_words="line0 says something here.")
    (p,) = _one(long_transcript, fake_llm, settings, item, caplog)
    assert (p.start, p.end) == (0.0, 29.3)
    msgs = [r.getMessage() for r in caplog.records if "out of range" in r.getMessage()]
    assert len(msgs) == 1
    caplog.clear()
    item = raw(0, 0, start_line=190, end_line=4000, first_words="line190 says something here.")
    (p,) = _one(long_transcript, fake_llm, settings, item, caplog)
    assert (p.start, p.end) == (570.0, 599.3)  # clamped to the last line (L199)
    assert sum("out of range" in r.getMessage() for r in caplog.records) == 1


def _distinct_transcript() -> Transcript:
    """Like long_transcript (sentence n at 3n s, 2.3 s long) but every word is unique:
    "alphaN bravoN charlieN deltaN."."""
    segments = []
    for n in range(200):
        texts = [f"alpha{n}", f"bravo{n}", f"charlie{n}", f"delta{n}."]
        words = [
            Word(start=round(3.0 * n + j * 0.6, 3), end=round(3.0 * n + j * 0.6 + 0.5, 3), text=t)
            for j, t in enumerate(texts)
        ]
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=" ".join(texts), words=words)
        )
    return Transcript(language="en", duration=600.0, segments=segments)


def test_first_words_reanchor_start_within_three_lines(fake_llm, settings):
    item = raw(30, 60, first_words="Alpha12, bravo12 charlie12 delta12.")  # start_line 10
    (p,) = _one(_distinct_transcript(), fake_llm, settings, item)
    assert p.start == 36.0 and p.first_words == "alpha12 bravo12 charlie12 delta12."


def test_first_words_may_run_into_the_next_line(fake_llm, settings, caplog):
    item = raw(30, 60, first_words="alpha10 bravo10 charlie10 delta10. alpha11 bravo11")
    (p,) = _one(_distinct_transcript(), fake_llm, settings, item, caplog)
    assert p.start == 30.0 and "first_words" not in caplog.text


def test_first_words_fuzzy_match_keeps_line(fake_llm, settings, caplog):
    item = raw(30, 60, first_words="alpha10 bravo10 charlie1O")  # 2 of 3 tokens (an OCR-ish O)
    (p,) = _one(_distinct_transcript(), fake_llm, settings, item, caplog)
    assert p.start == 30.0 and "first_words" not in caplog.text


def test_first_words_without_match_keeps_line_and_logs(fake_llm, settings, caplog):
    item = raw(30, 60, first_words="Nobody expected the raccoon.")
    (p,) = _one(_distinct_transcript(), fake_llm, settings, item, caplog)
    assert p.start == 30.0
    assert sum("first_words" in r.getMessage() for r in caplog.records) == 1


def test_first_words_match_beyond_three_lines_is_ignored(fake_llm, settings):
    item = raw(30, 60, first_words="alpha14 bravo14 charlie14 delta14.")  # 4 lines away
    (p,) = _one(_distinct_transcript(), fake_llm, settings, item)
    assert p.start == 30.0


def test_reversed_lines_are_dropped(long_transcript, fake_llm, settings):
    item = raw(0, 0, start_line=20, end_line=10, first_words="line20 says something here.")
    assert _one(long_transcript, fake_llm, settings, item) == []


def test_skip_verdicts_are_dropped(long_transcript, fake_llm, settings):
    payload = make_payload(8)
    for c in payload["clips"][:3]:
        c["verdict"] = "skip"
    payload["clips"][3]["verdict"] = "maybe"
    fake_llm.queue(payload)
    picks = select_clips(long_transcript, fake_llm, settings)
    # 8 returned (>= 5 wanted) -> no follow-up even though only 5 survive the skips
    assert len(fake_llm.calls) == 1
    assert [round(p.start) for p in picks] == [210, 270, 330, 390, 450]
    assert all(p.verdict in {"post", "maybe"} for p in picks)


def test_fewer_kept_after_skips_makes_no_follow_up(long_transcript, fake_llm, settings):
    payload = make_payload(6)
    for c in payload["clips"][2:]:
        c["verdict"] = "skip"
    fake_llm.queue(payload)
    picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 2 and len(fake_llm.calls) == 1


def test_post_ranks_above_maybe(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"clips": 1})
    hi = raw(30, 60, total=90, verdict="maybe")
    lo = raw(300, 330, total=40, verdict="post")
    fake_llm.queue({"clips": [hi, lo]})
    (p,) = select_clips(long_transcript, fake_llm, s)
    assert round(p.start) == 300


def test_every_candidate_skip_raises_clear_error(long_transcript, fake_llm, settings):
    payload = make_payload(8)
    for c in payload["clips"]:
        c["verdict"] = "skip"
    fake_llm.queue(payload)
    with pytest.raises(NoClipsSelectedError, match="rated every candidate as skip"):
        select_clips(long_transcript, fake_llm, settings)


def test_every_candidate_skip_after_follow_up_raises(long_transcript, fake_llm, settings):
    first = {"clips": [raw(30, 60, verdict="skip")]}
    more = {"clips": [raw(300, 330, verdict="skip")]}
    fake_llm.queue(first, more)
    with pytest.raises(NoClipsSelectedError, match="may not have strong moments"):
        select_clips(long_transcript, fake_llm, settings)
    assert len(fake_llm.calls) == 2


def test_edit_beats_outside_clip_are_dropped(long_transcript, fake_llm, settings):
    beats = [
        "zoom: 'line4' @ 12.0s",
        "zoom: 'line10' @ 31.5s",
        "emoji 😱 @ 59.3s",
        "zoom: 'line20' @ 61.0s",
        "emphasis without a time",
    ]
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, edit_beats=beats))
    assert p.edit_beats == ["zoom: 'line10' @ 30.0s", "emoji 😱 @ 59.3s"]


def test_post_copy_reaches_the_pick(long_transcript, fake_llm, settings):
    item = raw(
        30,
        60,
        hook_type="confession",
        hashtags=["#LineCook", "fyp", " kitchenlife ", "", "chefstories"],
        search_title="Running a restaurant kitchen alone at nineteen when the head chef vanished",
        comment_risk="high",
    )
    (p,) = _one(long_transcript, fake_llm, settings, item)
    assert p.hook_type == "confession" and p.verdict == "post" and p.comment_risk == "high"
    assert p.post_caption == "He ran the whole kitchen alone. Could you?"
    assert p.hashtags == ["LineCook", "kitchenlife", "chefstories"]
    assert len(p.search_title) <= 70 and p.search_title.startswith("Running a restaurant")
    assert p.category == "food" and p.last_words == "line19 says something here."


def test_chunk_mode_uses_global_line_numbers(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    item = raw(300, 330)  # L100-L109
    fake_llm.queue_forever({"clips": [item]})
    picks = select_clips(long_transcript, fake_llm, s)
    prompts = [c["prompt"] for c in fake_llm.calls]
    assert len(prompts) > 1
    assert "L199 [09:57.0] line199" in prompts[-1] and "L0 [" not in prompts[-1]
    later_first = re.search(r"^L(\d+) \[", prompts[1], re.MULTILINE)
    assert later_first and int(later_first.group(1)) > 0
    assert [(p.start, p.end) for p in picks] == [(300.0, 329.3)]
    assert "line numbers" in prompts[1].lower()


# --- review round 1: line-ownership anchoring, length-safe re-anchoring, enforced endings ----

_FILLER = "filler{n} words keep going here."
_SETUP = "And one night the head chef just doesn't show up."
_CLAIM = "I ran a three-hundred-cover Saturday on my own when I was nineteen."


def _lines_transcript(overrides: dict[int, str], count: int = 40) -> Transcript:
    """`count` lines on a 4 s grid (line i starts at 4i); words 0.2 s long every 0.25 s, so a
    gap of >= 1 s always separates lines. Line i is `_FILLER` unless overridden."""
    segments = []
    for i in range(count):
        texts = overrides.get(i, _FILLER.format(n=i)).split()
        words = [
            Word(start=round(4.0 * i + j * 0.25, 3), end=round(4.0 * i + j * 0.25 + 0.2, 3), text=t)
            for j, t in enumerate(texts)
        ]
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=" ".join(texts), words=words)
        )
    return Transcript(language="en", duration=4.0 * count, segments=segments)


def _line_item(start_line, end_line, first_words, last_words, **kw):
    item = raw(0, 3, first_words=first_words, last_words=last_words, edit_beats=[], **kw)
    item.update(start_line=start_line, end_line=end_line)
    return item


def _end_of(tr: Transcript, line: int) -> float:
    return tr.segments[line].words[-1].end


def test_start_does_not_anchor_on_a_backchannel_before_first_words(fake_llm, settings):
    tr = _lines_transcript({10: _SETUP, 11: "Yeah, so.", 12: _CLAIM})
    item = _line_item(11, 19, _CLAIM, _FILLER.format(n=19))
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 48.0  # L12, not the "Yeah, so." line that merely runs into it


def test_start_reanchors_to_an_earlier_line(fake_llm, settings):
    tr = _lines_transcript({12: _CLAIM})
    item = _line_item(14, 20, _CLAIM, _FILLER.format(n=20))
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 48.0


def test_start_reanchor_never_crosses_end_line(fake_llm, settings, caplog):
    s = settings.model_copy(update={"min_clip_s": 4})
    tr = _lines_transcript({12: _CLAIM})
    item = _line_item(10, 11, _CLAIM, _FILLER.format(n=11))
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert (p.start, p.end) == (40.0, _end_of(tr, 11))  # kept, not dropped as reversed
    assert "first_words" in caplog.text and "keeping L10" in caplog.text


def test_start_reanchor_breaking_length_falls_back_to_given_line(fake_llm, settings, caplog):
    s = settings.model_copy(update={"min_clip_s": 32})  # min length 24 s
    tr = _lines_transcript({13: _CLAIM})
    item = _line_item(10, 17, _CLAIM, _FILLER.format(n=17))  # L13-L17 would be ~18 s
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert p.start == 40.0 and "back to L10" in caplog.text


def test_end_reanchors_from_last_words_both_ways(fake_llm, settings):
    tr = _lines_transcript({19: "And that is how I got the job."})
    item = _line_item(10, 17, _FILLER.format(n=10), "And that is how I got the job.")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.end == _end_of(tr, 19)
    tr = _lines_transcript({17: "And that is how I got the job."})
    item = _line_item(10, 19, _FILLER.format(n=10), "And that is how I got the job.")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.end == _end_of(tr, 17)


def test_backchannel_last_line_steps_back(fake_llm, settings, caplog):
    tr = _lines_transcript({18: "Mm-hmm."})
    item = _line_item(10, 18, _FILLER.format(n=10), "Mm-hmm.")
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.end == _end_of(tr, 17) and "backchannel" in caplog.text


def _end_item(last_text, **kw):
    return _line_item(10, 18, _FILLER.format(n=10), last_text, **kw)


@pytest.mark.parametrize("word", ["Wow.", "Exactly."])
def test_wow_and_exactly_are_not_backchannels(fake_llm, settings, word):
    tr = _lines_transcript({18: word})
    (p,) = _one(tr, fake_llm, settings, _end_item(word))
    assert p.end == _end_of(tr, 18)


def test_backchannel_kept_for_reaction_peak(fake_llm, settings):
    tr = _lines_transcript({18: "Mm-hmm."})
    (p,) = _one(tr, fake_llm, settings, _end_item("Mm-hmm.", hook_type="reaction_peak"))
    assert p.end == _end_of(tr, 18)


@pytest.mark.parametrize("marker", ["🔊loud", "🔥chat×4 😂"])
def test_backchannel_kept_when_last_sentence_is_marked(fake_llm, settings, marker):
    tr = _lines_transcript({18: "Mm-hmm."})  # L18 spans 72.0-72.45
    fake_llm.queue({"clips": [_end_item("Mm-hmm.")]}, EMPTY)
    (p,) = select_clips(tr, fake_llm, settings, markers={72: [marker]})
    assert p.end == _end_of(tr, 18)


def test_backchannel_kept_when_loud_run_began_earlier(fake_llm, settings):
    tr = _lines_transcript({18: "Yeah."})  # L18 starts at 72.0
    fake_llm.queue({"clips": [_end_item("Yeah.")]}, EMPTY)
    (p,) = select_clips(tr, fake_llm, settings, heat_seconds={70, 71, 72})
    assert p.end == _end_of(tr, 18)
    fake_llm.queue({"clips": [_end_item("Yeah.")]}, EMPTY)
    (p,) = select_clips(tr, fake_llm, settings, heat_seconds={73})  # next line's second
    assert p.end == _end_of(tr, 17)


def test_is_marked_ignores_the_next_sentences_second():
    from hookline.models import Sentence
    from hookline.pipeline.select import _is_marked

    s = Sentence(start=72.0, end=73.0, text="Yeah.")
    assert not _is_marked(s, {73})
    assert _is_marked(s, {72})
    assert _is_marked(Sentence(start=72.4, end=73.2, text="Yeah."), {73})


def test_backchannel_still_steps_back_when_marker_is_elsewhere(fake_llm, settings):
    tr = _lines_transcript({18: "Mm-hmm."})
    fake_llm.queue({"clips": [_end_item("Mm-hmm.")]}, EMPTY)
    (p,) = select_clips(tr, fake_llm, settings, markers={40: ["🔊loud"]})
    assert p.end == _end_of(tr, 17)


def _fix_src(texts: dict[int, str], settings, count: int = 40):
    """A _Source whose sentences are exactly the given lines (bypassing selection_lines, which
    would merge an unfinished line into the next): line i at 4i s, 0.2 s words every 0.25 s."""
    from hookline.models import Sentence
    from hookline.pipeline.select import _Source, _tokens

    sents, words = [], []
    for i in range(count):
        text = texts.get(i, _FILLER.format(n=i))
        ws = [
            Word(start=4.0 * i + j * 0.25, end=4.0 * i + j * 0.25 + 0.2, text=t)
            for j, t in enumerate(text.split())
        ]
        sents.append(Sentence(start=ws[0].start, end=ws[-1].end, text=text))
        words += ws
    return _Source(
        sentences=sents,
        tokens=[_tokens(x.text) for x in sents],
        words=words,
        duration=4.0 * count,
        settings=settings,
        exclude=[],
    )


def _fix(texts, settings, last=17):
    from hookline.pipeline.select import _fix_ending, _RawClip

    item = _line_item(10, last, _FILLER.format(n=10), texts.get(last, ""))
    return _fix_ending(_RawClip.model_validate(item), 10, last, _fix_src(texts, settings))


def test_unfinished_last_line_extends_to_a_finished_one(settings, caplog):
    caplog.set_level(logging.INFO, logger="hookline.pipeline.select")
    assert _fix({17: "and the news was not going to...", 18: "really tell it."}, settings) == 18
    assert "extended to L18" in caplog.text


def test_unfinished_last_line_extends_up_to_two_lines(settings):
    texts = {17: "so the thing is...", 18: "that we never...", 19: "got paid for it."}
    assert _fix(texts, settings) == 19


def test_unfinished_ending_without_completion_is_kept_and_logged(settings, caplog):
    # the next line is a backchannel, the one after is unfinished too: no extension
    texts = {17: "so the thing is...", 18: "Yeah.", 19: "and then..."}
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        assert _fix(texts, settings) == 17
    assert "could not be completed" in caplog.text
    assert _fix({17: "so the thing is...", 18: "that we...", 19: "never got..."}, settings) == 17


def test_unfinished_ending_never_extends_past_max_clip_s(settings):
    # L10-L18 is 32.45 s: more than the 5 s a payoff may run over max_clip_s
    s = settings.model_copy(update={"max_clip_s": 25})
    assert _fix({17: "so the thing is...", 18: "it worked."}, s) == 17


def test_hashtags_strip_punctuation_ban_more_and_log_when_few(
    long_transcript, fake_llm, settings, caplog
):
    tags = ["#Line-Cook!", "#ForYourPage", "reels", "tiktok", "explore", "shorts", "fypage"]
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, hashtags=tags), caplog)
    assert p.hashtags == ["LineCook"]
    assert "only 1 usable hashtag" in caplog.text


def test_dropped_edit_beats_are_logged(long_transcript, fake_llm, settings, caplog):
    caplog.set_level(logging.INFO, logger="hookline.pipeline.select")
    beats = ["zoom: 'x' @ 5.0s", "no time here", "zoom: 'says' @ 31.0s"]
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, edit_beats=beats))
    assert p.edit_beats == ["zoom: 'says' @ 30.6s"]
    assert "dropped 2 edit beat(s)" in caplog.text


# --- M2.5 wave A: sentence lines, verbatim cold opens, opener guard, packaging checks -------


def test_prompt_lines_are_whole_sentences_not_pause_fragments(fake_llm, settings):
    # "So" ... (1 s pause) ... "the rent doubled in a year." become one selection line
    tr = _lines_transcript({15: "So", 16: "the rent doubled in a year."})
    fake_llm.queue({"clips": [_line_item(10, 17, _FILLER.format(n=10), _FILLER.format(n=18))]})
    fake_llm.queue(EMPTY)
    select_clips(tr, fake_llm, settings)
    prompt = fake_llm.calls[0]["prompt"]
    assert "L15 [01:00.0] So the rent doubled in a year." in prompt
    assert "L16 [01:08.0] filler17 words keep going here." in prompt


_TEASE = "And so I think the thing you can predict is that capitalism"


def _tease_transcript(rest="will eat every public good."):
    # selection lines: L0-L14 fillers, L15 = original lines 15+16 (60.0-...), L16 = line 17 ...
    return _lines_transcript({15: _TEASE, 16: rest})


def test_cold_open_words_strip_lead_ins_and_extend_to_the_sentence_end(fake_llm, settings):
    tr = _tease_transcript()
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=19), cold_open_words=_TEASE)
    (p,) = _one(tr, fake_llm, settings, item)
    # "And so I think" stripped: starts on "the" (word 4 -> 61.0); ends on "good." of line 16
    assert p.cold_open == (61.0, _end_of(tr, 16))


def test_cold_open_words_match_fuzzily_and_in_order(fake_llm, settings):
    tr = _tease_transcript()
    words = "thing you can predict is that capitalism will eat every public good"
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=19), cold_open_words=words)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.cold_open == (61.25, _end_of(tr, 16))


def test_cold_open_dropped_when_no_sentence_end_within_6s(fake_llm, settings, caplog):
    rest = " ".join(["on"] * 12) + " and forever."  # ends 67.45 s, 6.45 s after 61.0
    tr = _tease_transcript(rest)
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=19), cold_open_words=_TEASE)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.cold_open is None
    assert "dropped cold open" in caplog.text and "6" in caplog.text


def test_cold_open_in_the_clips_first_sentence_is_dropped(fake_llm, settings, caplog):
    tease = "the thing you can predict is that capitalism"  # a clean opener: no guard move
    tr = _lines_transcript({15: tease, 16: "will eat every public good."})
    first = tease + " will eat every public good."
    item = _line_item(15, 22, first, _FILLER.format(n=23), cold_open_words=tease)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.cold_open is None and "first sentence" in caplog.text


def test_cold_open_never_reaches_past_the_clip_end(fake_llm, settings, caplog):
    s = settings.model_copy(update={"min_clip_s": 4})
    tr = _tease_transcript()
    # clip L10-L14 ends before the teaser sentence (L15): words beyond the clip are not used
    item = _line_item(10, 14, _FILLER.format(n=10), _FILLER.format(n=14), cold_open_words=_TEASE)
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert p.cold_open is None and "not found in the clip" in caplog.text


def _guard_item(tr_overrides, first_words, **kw):
    tr = _lines_transcript(tr_overrides)
    return tr, _line_item(10, 18, first_words, _FILLER.format(n=18), **kw)


@pytest.mark.parametrize(
    "opener",
    [
        "So the point is this.",
        "And that changed everything.",
        "Um, but here is the thing.",
        "I was like, no way.",
        "He goes, absolutely not.",
        "They said we could not do it.",
        '"Get out of my office," she said.',
    ],
)
def test_bad_opener_moves_start_forward_one_sentence(fake_llm, settings, caplog, opener):
    tr, item = _guard_item({10: opener}, opener)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 44.0 and p.first_words == _FILLER.format(n=11)
    assert "opener" in caplog.text


def test_bad_opener_moves_up_to_two_sentences(fake_llm, settings):
    tr, item = _guard_item({10: "But anyway.", 11: "Because he goes, what?"}, "But anyway.")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 48.0


def test_bad_opener_kept_and_logged_when_moving_breaks_min_length(fake_llm, settings, caplog):
    s = settings.model_copy(update={"min_clip_s": 42})  # min 31.5 s; L10-L18 33.2, L11- 29.2
    tr, item = _guard_item({10: "So the point is this."}, "So the point is this.")
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert p.start == 40.0 and "opener" in caplog.text and "keeping" in caplog.text


def test_three_bad_openers_in_a_row_keep_the_start(fake_llm, settings, caplog):
    bad = {10: "So yes.", 11: "And no.", 12: "But maybe."}
    tr, item = _guard_item(bad, "So yes.")
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 40.0 and "keeping" in caplog.text


def test_clean_opener_is_not_moved(fake_llm, settings):
    tr, item = _guard_item({10: _CLAIM}, _CLAIM)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 40.0 and p.first_words == _CLAIM


def test_first_and_last_words_are_the_actual_sentences(fake_llm, settings):
    tr = _lines_transcript({})
    item = _line_item(10, 17, "something nobody said", "nor this either")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.first_words == _FILLER.format(n=10) and p.last_words == _FILLER.format(n=17)


def test_reaction_peak_may_be_as_short_as_12s(fake_llm, settings):
    tr = _lines_transcript({})
    short = _line_item(10, 13, _FILLER.format(n=10), _FILLER.format(n=13))  # ~13.2 s
    assert _one(tr, fake_llm, settings, short) == []  # 15 s minimum for other hook types
    short["hook_type"] = "reaction_peak"
    (p,) = _one(tr, fake_llm, settings, short)
    assert p.start == 40.0


def test_quoted_title_kept_when_verbatim_in_the_clip(fake_llm, settings):
    tr = _lines_transcript({12: _CLAIM})
    title = 'Chef: "a three-hundred-cover Saturday on my own"'
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=18), title=title)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.title == title


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ('Chef: "I ran the whole restaurant alone"', "Chef: I ran the whole restaurant alone"),
        ("Chef: “I ran it alone”", "Chef: I ran it alone"),
    ],
)
def test_paraphrased_quote_in_title_loses_its_quote_marks(fake_llm, settings, title, expected):
    tr = _lines_transcript({12: _CLAIM})
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=18), title=title)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.title == expected


@pytest.mark.parametrize(
    "caption",
    [
        "Big call from the chef. Agree or disagree?",
        "She stood her ground. Thoughts?",
        "Has a story ever flipped your view?",
        "What do you think?",
    ],
)
def test_template_caption_question_is_logged_and_kept(
    long_transcript, fake_llm, settings, caplog, caption
):
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, post_caption=caption), caplog)
    assert p.post_caption == caption and "template caption" in caplog.text


def test_specific_caption_question_is_not_flagged(long_transcript, fake_llm, settings, caplog):
    _one(long_transcript, fake_llm, settings, raw(30, 60), caplog)
    assert "template caption" not in caplog.text


def test_edit_beat_snaps_to_nearest_occurrence_within_3s(long_transcript, fake_llm, settings):
    beats = [
        "zoom: 'says' @ 35.0s",  # 'says' at 33.6 and 36.6 -> 33.6 is 1.4 s away, 36.6 is 1.6
        "zoom: 'line12' @ 40.5s",  # 4.5 s away, but its only occurrence in the clip -> 36.0
        "zoom: 'Line15,' @ 44.0s",  # case and punctuation normalised -> 45.0
        "zoom: 'here' @ 10.0s",  # many occurrences, none within 3 s -> dropped
    ]
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, edit_beats=beats))
    assert p.edit_beats == [
        "zoom: 'says' @ 33.6s",
        "zoom: 'line12' @ 36.0s",
        "zoom: 'Line15,' @ 45.0s",
    ]


def test_same_category_pick_ranks_below_other_categories(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"clips": 2})
    a = raw(30, 60, total=60, category="Politics")
    b = raw(90, 120, total=55, category="politics ")
    c = raw(150, 180, total=40, category="media")
    fake_llm.queue({"clips": [a, b, c]})
    picks = select_clips(long_transcript, fake_llm, s)
    assert [round(p.start) for p in picks] == [30, 150]


def test_same_category_pick_still_fills_when_nothing_else(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"clips": 2})
    fake_llm.queue({"clips": [raw(30, 60, total=60), raw(90, 120, total=55)]})
    assert [round(p.start) for p in select_clips(long_transcript, fake_llm, s)] == [30, 90]


def test_system_prompt_covers_wave_a():
    for needle in (
        "`cold_open_words`",
        "never mark it skip only because of its opening",
        "never a topic label",
        "never promise visuals",
        "complement",
        "famous people, brands or institutions",
        "read the next 20 seconds",
        "20–40 seconds",
        "start later, closer to the turn",
        "two plausible answers",
        '"Agree or disagree?"',
        '"Thoughts?"',
        '"What do you think?"',
        "same theme",
        "scores 12",
        "scores 22",
        "as short as 12 seconds",
    ):
        assert needle in SYSTEM_PROMPT, needle


# --- fix round 1: length bounds, front-trim, opener refinements -------------------------------


def test_build_prompt_states_the_length_bounds(long_transcript, fake_llm, settings):
    p = build_prompt("x", n=5, duration=900.0, exclude=[], min_s=20, max_s=60)
    assert (
        "Each clip lasts 20–60 seconds. Never exceed 60 s; if the story needs more, start "
        "later, closer to the turn." in p
    )
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings)
    assert "Never exceed 50 s" in fake_llm.calls[0]["prompt"]  # the 50 s default


def _trim_item(overrides=None, **kw):
    tr = _lines_transcript(overrides or {})
    return tr, _line_item(10, 31, _FILLER.format(n=10), _FILLER.format(n=31), **kw)


def test_86s_candidate_is_trimmed_from_the_front(fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    tr, item = _trim_item()  # L10-L31: 40.0-125.2 = 85.2 s
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert (p.start, p.end) == (68.0, _end_of(tr, 31))  # L17: earliest start within 60 s
    assert "trimmed from the front to fit (85.2s -> 57.2s)" in caplog.text


def test_front_trim_skips_weak_openers_and_backchannels(fake_llm, settings):
    settings = settings.model_copy(update={"max_clip_s": 60})  # L17 (57.2 s) fits 60 s
    tr, item = _trim_item({17: "So yes it was.", 18: "Yeah."})
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 76.0  # L17 opens on "So", L18 is a backchannel


def test_front_trim_never_cuts_into_the_cold_open(fake_llm, settings):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    teaser = "Rent doubled in one year for everyone."  # L16, 1.7 s
    tr, item = _trim_item({16: teaser}, cold_open_words=teaser)
    (p,) = _one(tr, fake_llm, settings, item)
    # starts must end before the teaser (L16 at 64.0): none fits 60 s; L13 fits 1.25 x 60
    assert p.start == 52.0 and p.cold_open == (64.0, _end_of(tr, 16))


def test_untrimmable_long_candidate_is_dropped_with_a_log(fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    tr = _lines_transcript({})
    item = _line_item(10, 39, _FILLER.format(n=10), _FILLER.format(n=39))
    item["cold_open_words"] = _FILLER.format(n=12)  # no start may pass L11: 113 s minimum
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        assert _one(tr, fake_llm, settings, item) == []
    assert "dropped clip 'Clip at 0': 117.2s is over 60s" in caplog.text


def _words(*spec):
    """Word list from (start, text) pairs; each word lasts 0.2 s."""
    return [Word(start=t, end=t + 0.2, text=x) for t, x in spec]


def _seq(text, t0=0.0, step=0.25):
    return _words(*((t0 + i * step, x) for i, x in enumerate(text.split())))


@pytest.mark.parametrize(
    ("words", "bad"),
    [
        (_seq("I like pizza a lot."), False),
        (_seq("I like, no way."), True),
        (_seq("She was like, absolutely not."), True),
        (_seq("I said, Hi, I'm here to borrow your equipment."), False),  # self-quote: allowed
        (_seq("I was like, so you think it is worth the risk?"), False),  # >= 4 quoted words
        (_seq("We were like, there is no way."), False),
        (_seq("He was like, so you think it is worth the risk?"), True),  # not first person
        (_seq("So the rent doubled."), True),
        (_words((0.0, "So,"), (0.6, "the"), (0.85, "rent"), (1.1, "doubled.")), False),
        (_words((0.0, "So,"), (0.6, "and"), (0.85, "then"), (1.1, "this.")), True),
        (_words((0.0, "So,"), (0.6, "I"), (0.85, "was"), (1.1, "like,"), (1.35, "no.")), True),
    ],
)
def test_bad_opener_rules(words, bad):
    from hookline.pipeline.select import _bad_opener

    assert _bad_opener(words) is bad


def test_reaction_peak_is_exempt_from_the_opener_guard(fake_llm, settings):
    tr, item = _guard_item({10: "So the point is this."}, "So the point is this.")
    item["hook_type"] = "reaction_peak"
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 40.0


def _timed_transcript(lines: list[tuple[float, str]], duration: float) -> Transcript:
    segments = []
    for t0, text in lines:
        ws = [
            Word(start=round(t0 + j * 0.25, 3), end=round(t0 + j * 0.25 + 0.2, 3), text=t)
            for j, t in enumerate(text.split())
        ]
        segments.append(Segment(start=ws[0].start, end=ws[-1].end, text=text, words=ws))
    return Transcript(language="en", duration=duration, segments=segments)


def test_opener_guard_moves_at_most_12s(fake_llm, settings, caplog):
    settings = settings.model_copy(update={"max_clip_s": 60})  # arithmetic assumes 60 s
    lines = [(4.0 * i, _FILLER.format(n=i)) for i in range(10)]
    lines += [(40.0, "So that was it."), (53.0, "Rent doubled in a year.")]
    lines += [(57.0 + 4.0 * i, _FILLER.format(n=i + 12)) for i in range(12)]
    tr = _timed_transcript(lines, 150.0)
    item = _line_item(10, 20, "So that was it.", _FILLER.format(n=20))
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 40.0 and "keeping it" in caplog.text  # L11 is 13 s later


def test_cold_open_strips_quoted_conversation_lead_ins(fake_llm, settings):
    tr = _lines_transcript({15: "I was like, you really think it is worth it for you?"})
    item = _line_item(
        10, 18, _FILLER.format(n=10), _FILLER.format(n=18), cold_open_words=tr.segments[15].text
    )
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.cold_open == (60.75, _end_of(tr, 15))  # starts on "you"


def test_diversify_works_within_verdict_tiers(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"clips": 2})
    a = raw(30, 60, total=60, category="Politics")
    b = raw(90, 120, total=55, category="politics")
    c = raw(150, 180, total=70, category="media", verdict="maybe")
    fake_llm.queue({"clips": [a, b, c]})
    assert [round(p.start) for p in select_clips(long_transcript, fake_llm, s)] == [30, 90]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Chef: 'I ran it alone'", "Chef: I ran it alone"),
        (
            "Chef: 'a three-hundred-cover Saturday' story",
            "Chef: 'a three-hundred-cover Saturday' story",
        ),
        ("The chef's 'big' night", "The chef's 'big' night"),  # one word: left alone
    ],
)
def test_straight_single_quoted_title_spans(fake_llm, settings, title, expected):
    tr = _lines_transcript({12: _CLAIM})
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=18), title=title)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.title == expected


def test_opener_guard_never_moves_into_the_cold_open(fake_llm, settings, caplog):
    # real case: "And, of course, I'm very snarky." / "I was like, so, ...?" / teaser on L12
    bad = {10: "And, of course, I'm very snarky.", 11: "I was like, so, is it worth it?"}
    teaser = "Like, your husband might die over there tomorrow."  # 1.7 s without "Like,"
    tr, item = _guard_item({**bad, 12: teaser}, bad[10])
    item["cold_open_words"] = teaser
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    # wave C: L11 is a first-person self-quote with content, so the guard may land on it
    assert p.start == 44.0 and p.cold_open == (48.25, _end_of(tr, 12))
    assert "start moved to L11" in caplog.text


# --- fix round 2: 50 s default maximum, 1.5 s minimum teaser ----------------------------------


def test_prompt_length_bound_follows_settings(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))
    select_clips(long_transcript, fake_llm, settings)
    assert "Each clip lasts 20–50 seconds. Never exceed 50 s" in fake_llm.calls[0]["prompt"]


_SHORT = "We all knew that your husband might die."  # 8 words: 60.0 ... "die." ends 61.95


def test_short_teaser_extends_back_within_its_sentence_to_1s(fake_llm, settings):
    tr = _lines_transcript({15: _SHORT})
    item = _line_item(
        10, 18, _FILLER.format(n=10), _FILLER.format(n=18), cold_open_words="might die."
    )
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.cold_open == (60.75, 61.95)  # "that your husband might die." = 1.2 s


def _teaser_line(timed_words):
    """_lines_transcript with line 15 replaced by explicitly timed (start, end, text) words."""
    tr = _lines_transcript({})
    ws = [Word(start=a, end=b, text=t) for a, b, t in timed_words]
    text = " ".join(t for _, _, t in timed_words)
    tr.segments[15] = Segment(start=ws[0].start, end=ws[-1].end, text=text, words=ws)
    return tr


def test_real_1_2s_teaser_after_a_lead_in_is_kept(fake_llm, settings):
    # the strongest real teaser: "Like, your husband might die." -> "your ... die." is 1.2 s
    tr = _teaser_line(
        [(60.0, 60.3, "Like,"), (60.4, 60.6, "your"), (60.65, 60.95, "husband"),
         (61.0, 61.2, "might"), (61.3, 61.6, "die.")]
    )  # fmt: skip
    words = "Like, your husband might die."
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=18), cold_open_words=words)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.cold_open == (60.4, 61.6)


def test_0_8s_teaser_without_an_extendable_start_is_dropped(fake_llm, settings, caplog):
    tr = _teaser_line(
        [(60.0, 60.3, "Like,"), (60.4, 60.6, "he"), (60.65, 60.9, "might"), (60.95, 61.2, "die.")]
    )  # "he might die." is 0.8 s; only the lead-in "Like," precedes it
    item = _line_item(
        10, 18, _FILLER.format(n=10), _FILLER.format(n=18), cold_open_words="he might die."
    )
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.cold_open is None and "under 1.0s" in caplog.text


# --- fix wave C: self-quotes, filler starts, drop_lines, mid-line starts, endings, titles ----


@pytest.mark.parametrize(
    "opener",
    [
        "I was like, so you think it is worth the risk?",
        "We were like, there is no way this works.",
        "I said, you cannot keep doing this to us.",
    ],
)
def test_first_person_self_quote_with_content_is_not_moved(fake_llm, settings, caplog, opener):
    # real case: "I was like, so, you think it's worth it to go to war?" lost its hook line
    tr, item = _guard_item({10: opener}, opener)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 40.0 and "weak opener" not in caplog.text


def test_opener_guard_never_lands_on_a_filler_led_bad_opener(fake_llm, settings, caplog):
    bad = {10: "So the point is this.", 11: "Like, so the rent doubled in a year."}
    tr, item = _guard_item(bad, bad[10])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 48.0 and "start moved to L12" in caplog.text


def test_opener_guard_lands_on_a_line_whose_only_problem_is_a_comma_filler(
    fake_llm, settings, caplog
):
    bad = {10: "So the point is this.", 11: "Like, the rent doubled in a year."}
    tr, item = _guard_item(bad, bad[10])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 44.25 and p.first_words == "the rent doubled in a year."
    assert "start moved to L11" in caplog.text


def test_leading_filler_is_stripped_from_the_clip_start(fake_llm, settings, caplog):
    line = "Um, like, my landlord raised the rent forty percent."
    tr, item = _guard_item({10: line}, line)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 40.5 and p.first_words == "my landlord raised the rent forty percent."
    assert "filler" in caplog.text


@pytest.mark.parametrize("line", ["You know what she did next?", "Like a lot of people, I paid."])
def test_filler_words_without_a_comma_are_kept(fake_llm, settings, line):
    tr, item = _guard_item({10: line}, line)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 40.0 and p.first_words == line


def test_declared_first_words_are_persisted(fake_llm, settings):
    tr, item = _guard_item({10: "So the point is this."}, "So the point is this.")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 44.0 and p.declared_first_words == "So the point is this."


_MID = "As soon as it was, I mean, the original investors walked out of the room."


def test_start_moves_mid_line_to_the_declared_first_words(fake_llm, settings, caplog):
    tr, item = _guard_item({10: _MID}, "the original investors walked out of the room.")
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 41.75 and p.first_words == "the original investors walked out of the room."
    assert "mid-line" in caplog.text and "match none" not in caplog.text


def test_drop_lines_in_the_schema():
    item = SELECTION_SCHEMA["properties"]["clips"]["items"]
    assert item["properties"]["drop_lines"] == {"type": "array", "items": {"type": "integer"}}
    assert "drop_lines" in item["required"]


def test_drop_lines_become_cuts_inside_the_clip(fake_llm, settings, caplog):
    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[12, 13])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    # cut from midway between L11's end (45.2) and L12's start (48.0) to midway between
    # L13's end (53.2) and L14's start (56.0)
    assert p.segments() == [(40.0, 46.6), (54.6, _end_of(tr, 18))]
    assert p.duration == pytest.approx(_end_of(tr, 18) - 40.0 - 8.0)
    assert "dropped L12" in caplog.text and "dropped L13" in caplog.text


def test_invalid_drop_lines_are_ignored_with_a_warning(fake_llm, settings, caplog):
    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[10, 18, 25, 3, 12])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.segments() == [(40.0, 46.6), (50.6, _end_of(tr, 18))]
    assert "drop_lines" in caplog.text and "L10" in caplog.text and "L25" in caplog.text


def test_at_most_three_drop_lines(fake_llm, settings, caplog):
    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[12, 14, 16, 17])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert len(p.segments()) == 4 and "at most 3" in caplog.text


def test_drop_lines_never_remove_the_cold_open(fake_llm, settings, caplog):
    teaser = _FILLER.format(n=15)
    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[15], cold_open_words=teaser)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.cold_open is not None and len(p.segments()) == 2  # teaser + one main piece
    assert "cold open" in caplog.text and "L15" in caplog.text


def test_drops_count_toward_the_length_bounds(fake_llm, settings):
    tr = _lines_transcript({})
    # L10-L24 spans 57.2 s (over the 50 s max, even with the 5 s payoff allowance); dropping
    # L12 and L14 (4 s each) leaves 49.2 s
    item = _line_item(10, 24, _FILLER.format(n=10), _FILLER.format(n=24), drop_lines=[12, 14])
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 40.0 and p.duration == pytest.approx(49.2)
    item["drop_lines"] = []
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start > 40.0  # without drops it is trimmed from the front


def test_unfinished_ending_extends_across_one_backchannel(settings, caplog):
    caplog.set_level(logging.INFO, logger="hookline.pipeline.select")
    texts = {17: "and what we found was...", 18: "Yeah.", 19: "the money was never there."}
    assert _fix(texts, settings) == 19
    assert "extended to L19" in caplog.text


def test_backchannel_crossing_needs_the_next_line_to_finish(settings):
    assert _fix({17: "and what we found was...", 18: "Yeah.", 19: "Right."}, settings) == 17
    texts = {17: "and what we found was...", 18: "Yeah.", 19: "that...", 20: "it was."}
    assert _fix(texts, settings) == 17


def test_backchannel_crossing_stays_within_max_length(settings):
    s = settings.model_copy(update={"max_clip_s": 31})  # L10-L19 is 37.2 s (over 31 + 5)
    texts = {17: "and what we found was...", 18: "Yeah.", 19: "the money was never there."}
    assert _fix(texts, s) == 17


def test_backchannel_crossing_stays_within_20s(settings):
    from hookline.models import Sentence
    from hookline.pipeline.select import _fix_ending, _RawClip

    texts = {17: "and what we found was...", 18: "Yeah.", 19: "the money was never there."}
    src = _fix_src(texts, settings.model_copy(update={"max_clip_s": 120}))
    late = src.sentences[19]
    src.sentences[19] = Sentence(start=late.start + 20, end=late.end + 20, text=late.text)
    item = _line_item(10, 17, _FILLER.format(n=10), texts[17])
    assert _fix_ending(_RawClip.model_validate(item), 10, 17, src) == 17


def test_system_prompt_has_no_examples_from_the_test_video():
    low = SYSTEM_PROMPT.lower()
    for needle in (
        "war worth it",
        "a little rude",
        "come to mock",
        "activists versus big money",
        "library",
        "public good",
    ):
        assert needle not in low, needle


def test_system_prompt_covers_wave_c():
    for needle in (
        "`drop_lines`",
        "scores about 14",
        "the payoff is after the interviewer's",
        "I said",
        "mid-line",
    ):
        assert needle in SYSTEM_PROMPT, needle


@pytest.mark.parametrize(
    "text",
    [
        "Public Access Television Explained: Why Small Towns Fought Cable Giants for Air",
        "How One Line Cook Ran a Three Hundred Cover Saturday Service Alone and the",
        "The Real Reason Small Restaurants Close in Their First Year of Business With",
    ],
)
def test_search_title_never_ends_on_a_function_word(text):
    from hookline.pipeline.select import _clip_title

    out = _clip_title(text)
    assert len(out) <= 70 and text.startswith(out)
    assert out.split()[-1].lower() not in {
        "for", "to", "of", "and", "the", "a", "an", "with", "in", "on", "at", "by", "or",
    }  # fmt: skip


def test_search_title_schema_states_the_limit():
    props = SELECTION_SCHEMA["properties"]["clips"]["items"]["properties"]
    assert "60" in props["search_title"].get("description", "")  # code still caps at 70


# --- fix wave C review round 1 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "drops"),
    [({11: "Yeah."}, [11]), ({11: "Yeah."}, []), ({}, [11])],
    ids=["dropped-backchannel", "backchannel", "dropped"],
)
def test_opener_guard_never_lands_on_a_dropped_line_or_backchannel(
    fake_llm, settings, caplog, overrides, drops
):
    tr, item = _guard_item(
        {10: "So the point is this.", **overrides}, "So the point is this.", drop_lines=drops
    )
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 48.0 and "start moved to L12" in caplog.text


def test_guard_move_is_recorded_on_the_pick(fake_llm, settings):
    tr, item = _guard_item({10: "So the point is this."}, "So the point is this.")
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.opener_moved is not None
    assert (p.opener_moved.from_line, p.opener_moved.to_line) == (10, 11)
    assert "weak opener" in p.opener_moved.reason
    tr, item = _guard_item({10: _CLAIM}, _CLAIM)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.opener_moved is None


@pytest.mark.parametrize("filler", ["Okay,", "Well,", "Yeah,", "Oh,", "Anyway,"])
def test_comma_followed_opener_fillers_are_stripped_from_the_start(fake_llm, settings, filler):
    line = f"{filler} my landlord raised the rent forty percent."
    tr, item = _guard_item({10: line}, line)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.start == 40.25 and p.first_words == "my landlord raised the rent forty percent."


@pytest.mark.parametrize(
    ("line", "first_words", "min_clip_s"),
    [
        (_MID, "the original investors walked out of the room.", 32),  # min 24.0 s
        ("Um, like, my landlord raised the rent forty percent.", None, 33.4),  # min 25.05 s
    ],
    ids=["mid-line", "filler-strip"],
)
def test_start_falls_back_to_the_whole_line_when_the_cut_start_breaks_min_length(
    fake_llm, settings, caplog, line, first_words, min_clip_s
):
    s = settings.model_copy(update={"min_clip_s": min_clip_s})
    tr = _lines_transcript({10: line})
    item = _line_item(10, 16, first_words or line, _FILLER.format(n=16))  # L10-L16: 25.2 s
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert p.start == 40.0 and p.first_words == line
    assert "whole line" in caplog.text


def test_length_estimate_keeps_drops_restored_for_the_teaser(fake_llm, settings, caplog):
    # L10-L24 is 57.2 s; dropping L15 would make it 53.2 s, but L15 holds the teaser, so the
    # drop is restored: the clip is over 50 + 5 s and must be trimmed from the front
    teaser = "Rent doubled in one year for everyone."
    tr = _lines_transcript({15: teaser})
    item = _line_item(
        10, 24, _FILLER.format(n=10), _FILLER.format(n=24), drop_lines=[15], cold_open_words=teaser
    )
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 48.0 and p.drops == [] and p.cold_open == (60.0, _end_of(tr, 15))
    assert p.end - p.start <= settings.max_clip_s


_QUESTION = {13: "Did I do this alone?", 14: "No, I did not do this alone."}


@pytest.mark.parametrize(
    "answer",
    ["No, I did not do this alone.", "Absolutely, every single day.", "Nope, never."],
)
def test_a_question_whose_answer_is_kept_is_never_dropped(fake_llm, settings, caplog, answer):
    tr, item = _guard_item({**_QUESTION, 14: answer}, _FILLER.format(n=10), drop_lines=[13])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.drops == [] and "L13" in caplog.text and "answer" in caplog.text


def test_a_question_without_a_kept_answer_may_be_dropped(fake_llm, settings):
    over = {13: "Did I do this alone?", 14: "The weather turned bad that week."}
    tr, item = _guard_item(over, _FILLER.format(n=10), drop_lines=[13])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1
    tr, item = _guard_item(_QUESTION, _FILLER.format(n=10), drop_lines=[13, 14])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1  # the answer is dropped too, so the question may go


def test_system_prompt_never_drops_a_question_whose_answer_is_kept():
    assert "never drop a question" in SYSTEM_PROMPT


def _piece_transcript():
    """Filler lines on the 4 s grid, except L12-L14: an aside, a 0.2 s "Okay." and another
    aside packed into 50.0-52.3 s, so dropping L12 and L14 leaves a 0.55 s piece between."""
    lines = [(4.0 * i, _FILLER.format(n=i)) for i in range(12)]
    lines += [(50.0, "aside one here."), (51.0, "Okay."), (51.6, "aside two here.")]
    lines += [(4.0 * i - 2.5, _FILLER.format(n=i)) for i in range(15, 40)]
    return _timed_transcript(lines, 160.0)


def test_a_drop_leaving_a_short_spoken_piece_is_ignored(fake_llm, settings, caplog):
    tr = _piece_transcript()
    item = _line_item(10, 20, _FILLER.format(n=10), _FILLER.format(n=20), drop_lines=[12, 14])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.drops == [(pytest.approx(47.6), pytest.approx(50.85))]
    assert "L14" in caplog.text and "1.5" in caplog.text


def test_drop_ranges_swallow_a_short_wordless_piece(settings, caplog):
    import dataclasses

    from hookline.pipeline.select import _merge_wordless_pieces

    words = [Word(start=0.0, end=0.2, text="a"), Word(start=5.0, end=5.2, text="b")]
    src = dataclasses.replace(_fix_src({}, settings), words=words)
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.select"):
        out = _merge_wordless_pieces([(1.0, 2.0), (2.5, 3.0), (5.5, 6.0)], src, "t")
    assert out == [(1.0, 3.0), (5.5, 6.0)] and "swallow" in caplog.text


# --- fix wave C review round 2 ----------------------------------------------------------------


def test_a_question_answered_by_its_content_words_is_never_dropped(fake_llm, settings, caplog):
    over = {13: "Did the landlord raise the rent?", 14: "The landlord raised the rent again."}
    tr, item = _guard_item(over, _FILLER.format(n=10), drop_lines=[13])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.drops == [] and "answers" in caplog.text  # "landlord" and "rent" repeated


def test_one_shared_content_word_is_not_an_answer(fake_llm, settings):
    over = {13: "Did I go?", 14: "We go there on Sundays with the kids."}
    tr, item = _guard_item(over, _FILLER.format(n=10), drop_lines=[13])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1


def test_a_drop_leaving_a_short_spoken_opening_is_ignored(fake_llm, settings, caplog):
    tr, item = _guard_item({10: "Look."}, "Look.", drop_lines=[11])
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.drops == [] and "L11" in caplog.text and "opening" in caplog.text


def test_a_drop_leaving_a_short_spoken_ending_is_ignored(fake_llm, settings, caplog):
    tr, item = _guard_item({18: "Exactly right."}, _FILLER.format(n=10), drop_lines=[17])
    item["last_words"] = "Exactly right."
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.drops == [] and "L17" in caplog.text and "closing" in caplog.text


def test_a_drop_next_to_a_long_enough_edge_piece_stands(fake_llm, settings):
    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[12, 16])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 2  # L10-L11 and L17-L18 each hold 5.2 s of speech


def test_whole_line_fallback_logs_each_drop_once(fake_llm, settings, caplog):
    s = settings.model_copy(update={"min_clip_s": 32})  # min 24.0 s
    tr = _lines_transcript({10: _MID})
    item = _line_item(
        10,
        17,
        "the original investors walked out of the room.",
        _FILLER.format(n=17),
        drop_lines=[13, 25],
    )  # L10-L17 is 29.2 s, 25.2 s without L13; 23.45 s from the mid-line start
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert p.start == 40.0 and "whole line" in caplog.text
    assert caplog.text.count("dropped L13") == 1
    assert caplog.text.count("L25 ignored") == 1


# --- fix wave E: search titles, payoff allowance, prompt calibration --------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (
            "Rent Control, Zoning and Parking Minimums: What Small Landlords Keep Missing",
            "Rent Control, Zoning and Parking Minimums",
        ),
        (
            "How a Night-Shift Baker Built a Bakery Empire — From One Oven to Twelve Stores",
            "How a Night-Shift Baker Built a Bakery Empire",
        ),
        (
            "Night Shift Bakers Who Quit Before Sunrise | Flour Prices and Twelve Stores",
            "Night Shift Bakers Who Quit Before Sunrise",
        ),
    ],
)
def test_search_title_over_the_cap_cuts_at_the_last_clause_boundary(text, want):
    from hookline.pipeline.select import _search_title

    assert len(text) > 70
    assert _search_title(text, "Selection Title That Fits") == want


def test_search_title_without_a_clause_boundary_falls_back_then_word_cuts(caplog):
    from hookline.pipeline.select import _search_title

    # the only boundary leaves an 18-character stub: not a title
    text = "Night Shift Bakers, Flour Prices and the Twelve-Store Bakery Chain Nobody Expected"
    assert _search_title(text, "Why Night Shift Bakers Quit") == "Why Night Shift Bakers Quit"
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.select"):
        out = _search_title(text, "", label="bakers")
    assert len(out) <= 70 and text.startswith(out) and out.split()[-1] != "the"
    assert "search_title" in caplog.text and "bakers" in caplog.text
    assert _search_title("  Short  Title ", "x") == "Short Title"  # under the cap: as given


def test_selection_search_title_uses_the_clause_cut(long_transcript, fake_llm, settings):
    long = "Rent Control, Zoning and Parking Minimums: What Small Landlords Keep Missing"
    (p,) = _one(long_transcript, fake_llm, settings, raw(30, 60, search_title=long))
    assert p.search_title == "Rent Control, Zoning and Parking Minimums"


def test_search_title_limit_is_60_in_the_prompt_and_schema():
    props = SELECTION_SCHEMA["properties"]["clips"]["items"]["properties"]
    assert "60" in props["search_title"]["description"]
    assert "at most 60 characters" in SYSTEM_PROMPT


def test_unfinished_ending_may_run_5s_over_max_for_a_payoff_within_8s(settings, caplog):
    caplog.set_level(logging.INFO, logger="hookline.pipeline.select")
    s = settings.model_copy(update={"max_clip_s": 30})  # L10-L18 is 32.45 s
    assert _fix({17: "so the thing is...", 18: "it worked."}, s) == 18
    assert "over the 30s maximum" in caplog.text
    s = settings.model_copy(update={"max_clip_s": 27})  # 32.45 s is more than 5 s over
    assert _fix({17: "so the thing is...", 18: "it worked."}, s) == 17


def test_clip_over_max_keeps_its_start_when_only_the_payoff_line_runs_over(
    fake_llm, settings, caplog
):
    tr = _lines_transcript({})  # L10-L21 is 45.2 s; L22 ends 4 s later (49.2 s)
    item = _line_item(10, 22, _FILLER.format(n=10), _FILLER.format(n=22))
    s = settings.model_copy(update={"max_clip_s": 46})
    (p,) = _one(tr, fake_llm, s, item, caplog)
    assert (p.start, p.end) == (40.0, _end_of(tr, 22)) and "payoff" in caplog.text
    s = settings.model_copy(update={"max_clip_s": 43})  # L10-L21 alone is over: trimmed
    (p,) = _one(tr, fake_llm, s, item)
    assert p.start > 40.0 and p.end == _end_of(tr, 22)


def test_build_prompt_states_the_payoff_allowance():
    p = build_prompt("x", n=5, duration=900.0, exclude=[], min_s=20, max_s=50)
    assert "up to 5 s over" in p and "within 8 s" in p


def test_system_prompt_covers_wave_e():
    for needle in (
        "5–15 seconds earlier",
        "outside knowledge",
        "about 14",
        '"She says…"',
        "`comment_risk`",
        "quotable",
        "up to 5 seconds over",
    ):
        assert needle in SYSTEM_PROMPT, needle


def test_wave_e_prompts_use_no_phrases_from_the_test_video():
    from hookline.pipeline.package import PACKAGING_SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, PACKAGING_SYSTEM_PROMPT):
        low = prompt.lower()
        for needle in (
            "worth it to go to war",
            "your husband",
            "united fruit",
            "cable company",
            "public access",
            "uvm",
            "send-off",
            "devour",
            "making media",
            "forces of capital",
            "president's office",
        ):
            assert needle not in low, needle


# --- fix wave E review round 1 ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Why First-Time Buyers Now Pay $1,000,000 for a Studio Apartment in the City Centre",
        "The Bakery Line That Starts at 3:30 Every Saturday Morning Wraps Around the Block Twice",
    ],
)
def test_clause_cut_ignores_number_commas_and_clock_times(text):
    from hookline.pipeline.select import _clause_cut

    assert _clause_cut(text) == ""  # no whitespace after the "," / ":": not a clause


def test_clause_cut_counts_an_unspaced_em_dash():
    from hookline.pipeline.select import _clause_cut

    text = "How a Night-Shift Baker Built a Bakery Empire—From One Oven to Twelve Stores Overnight"
    assert _clause_cut(text) == "How a Night-Shift Baker Built a Bakery Empire"


def test_clause_cut_strips_a_dangling_function_word():
    from hookline.pipeline.select import _clause_cut

    text = "Inside the Fight Over Parking Minimums and the Future of: Small Landlords, Big Rents"
    assert _clause_cut(text) == "Inside the Fight Over Parking Minimums and the Future"


def test_fix_ending_never_extends_a_clip_already_over_the_cap(settings):
    # L10-L17 is 28.95 s, already over 28; L18 (32.45 s) would fit 28 + 5 within 8 s
    s = settings.model_copy(update={"max_clip_s": 28})
    assert _fix({17: "so the thing is...", 18: "it worked."}, s) == 17


def test_length_cap_skips_a_backchannel_before_the_payoff(settings, caplog):
    from hookline.pipeline.select import _length_cap, _RawClip

    caplog.set_level(logging.INFO, logger="hookline.pipeline.select")
    s = settings.model_copy(update={"max_clip_s": 46})
    src = _fix_src({21: "Yeah."}, s)  # L10-L20 is 41.2 s; L22 ends 8 s after L20
    item = _line_item(10, 22, _FILLER.format(n=10), _FILLER.format(n=22))
    assert _length_cap(_RawClip.model_validate(item), 10, 22, src) == 51
    assert "payoff L22" in caplog.text


def test_length_cap_measures_from_the_last_kept_line_not_a_dropped_one(settings):
    from hookline.pipeline.select import _length_cap, _RawClip

    # L21 is dropped (-4 s): the clip is 45.95 s. The payoff L22 ends 4.75 s after L21
    # (dropped) but 8.75 s after L20, the last kept line: no allowance
    s = settings.model_copy(update={"max_clip_s": 45.5})
    texts = {22: "and that is how we finally got paid."}
    src = _fix_src(texts, s)
    item = _line_item(10, 22, _FILLER.format(n=10), texts[22], drop_lines=[21])
    assert _length_cap(_RawClip.model_validate(item), 10, 22, src) == 45.5


def test_hard_length_bound_is_the_larger_of_125_percent_and_cap_plus_5(fake_llm, settings):
    # max 17.5: 1.25 x = 21.875 s, cap + 5 = 22.5 s; L10-L15 is 21.95 s and only the payoff
    # line L15 runs over the cap
    s = settings.model_copy(update={"max_clip_s": 17.5, "min_clip_s": 10})
    tr = _lines_transcript({15: "and that is how we finally got paid."})
    item = _line_item(10, 15, _FILLER.format(n=10), "and that is how we finally got paid.")
    (p,) = _one(tr, fake_llm, s, item)
    assert (p.start, p.end) == (40.0, _end_of(tr, 15))


# Wave-E prompt examples must not mirror the test video's line structures either: these
# function-word skeletons came from its lines ("the original investors in ... included ...",
# "we're not making media just to make media ... we're making media so that ...") or map one
# to one onto its clips (whose company / which school / which war).
_TEST_VIDEO_SKELETONS = (
    r"\bincluded an? \w+",
    r"\binvestors in\b",
    r"\bnot \w+ \w+ just to\b",
    r"\bdon't \w+ \w+ to \w+ \w+;",
    r"; we \w+ (?:\w+ )?so\b",
    r"whose company",
    r"which school",
    r"which war",
)


def test_prompt_examples_avoid_the_test_videos_sentence_skeletons():
    from hookline.pipeline.package import PACKAGING_SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, PACKAGING_SYSTEM_PROMPT):
        for pattern in _TEST_VIDEO_SKELETONS:
            assert not re.search(pattern, prompt, re.IGNORECASE), pattern


def test_packaging_rules_have_no_stray_line_break():
    from hookline.pipeline.select import PACKAGING_RULES

    first = PACKAGING_RULES.splitlines()[0]
    assert first.endswith("mouth."), first


# --- wave G-select: gameplay cause and payoff ---------------------------------------------------


def test_system_prompt_covers_wave_g_select():
    for needle in (
        "⚡scare",
        "2–4 seconds of the cause",
        "within 3 seconds after the payoff",
        "death, loading or menu screen",
        "new attempt",
    ):
        assert needle in SYSTEM_PROMPT, needle


def test_wave_g_select_prompt_uses_no_phrases_from_the_test_stream():
    low = SYSTEM_PROMPT.lower()
    for needle in (
        "poppy",
        "huggy",
        "wuggy",
        "innovation is key",
        "thank you science",
        "we got him",
        "chinese food",
        "the fuck was that",
        "i don't want a hug",
        "run button",
        "just go",
    ):
        assert needle not in low, needle


# --- final fixes: opener guard on the untrimmed fallback, ellipses, edge pieces, drop logs ---


def test_untrimmed_long_clip_still_runs_the_opener_guard(fake_llm, settings, caplog):
    """A clip over max_clip_s with no clean later start is kept untrimmed; its weak opener is
    still judged by the opener guard (here no start within the guard's reach: kept, logged)."""
    settings = settings.model_copy(update={"max_clip_s": 60})
    teaser = "Rent doubled in one year for everyone."  # L11: no start may pass it
    tr = _lines_transcript({10: "So yes it was.", 11: teaser})
    item = _line_item(
        10, 27, "So yes it was.", _FILLER.format(n=27), cold_open_words=teaser
    )  # L10-L27: 69.2 s, over 60 s, within 75 s
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.start == 40.0 and "kept untrimmed" in caplog.text
    assert "L10 is a weak opener" in caplog.text


def test_a_cold_open_runs_past_an_ellipsis_to_the_sentence_end(fake_llm, settings):
    """A trailing "..." is not a sentence end for the teaser, as for selection lines."""
    tr = _tease_transcript("will eat every... public good.")
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=19), cold_open_words=_TEASE)
    (p,) = _one(tr, fake_llm, settings, item)
    assert p.cold_open == (61.0, _end_of(tr, 16))


def test_an_edge_piece_needs_only_0_8s_of_speech(fake_llm, settings):
    """Next to a jump cut an opening or closing piece needs MIN_EDGE_PIECE_S (0.8 s) of
    speech; a piece between two drops still needs MIN_PIECE_S."""
    tr, item = _guard_item({10: "Look at this now."}, "Look at this now.", drop_lines=[11])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1  # 0.95 s of speech before the cut
    tr, item = _guard_item({18: "Exactly right, it is."}, _FILLER.format(n=10), drop_lines=[17])
    item["last_words"] = "Exactly right, it is."
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1


def test_the_opening_piece_is_measured_from_the_clip_start(settings):
    """A mid-line start leaves less speech before the first jump cut than the whole line."""
    from hookline.pipeline.select import _drop_lines, _RawClip

    src = _fix_src({10: "Yes and look at this now."}, settings)
    item = _line_item(10, 18, "look at this now.", _FILLER.format(n=18), drop_lines=[11])
    clip = _RawClip.model_validate(item)
    assert _drop_lines(clip, 10, 18, src, None, log=False) == [11]  # 1.45 s from 40.0
    assert _drop_lines(clip, 10, 18, src, None, log=False, start=40.75) == []  # 0.7 s


def test_drop_decisions_are_logged_only_for_a_kept_clip(settings, caplog):
    import dataclasses

    from hookline.pipeline.select import _RawClip, _to_pick

    src = dataclasses.replace(_fix_src({}, settings), exclude=[(50.0, 52.0)])
    item = _line_item(10, 18, _FILLER.format(n=10), _FILLER.format(n=18), drop_lines=[13])
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.select"):
        assert _to_pick(_RawClip.model_validate(item), src) is None  # overlaps an exclude
    assert "dropped L13" not in caplog.text
