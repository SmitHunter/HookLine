import logging

import pytest

from hookline.llm import LLMError
from hookline.models import CaptionFix, ClipPick, RubricScores, Word
from hookline.pipeline.package import (
    PACKAGING_SCHEMA,
    PACKAGING_SYSTEM_PROMPT,
    apply_caption_fixes,
    clip_words,
    package_clips,
)
from hookline.pipeline.select import SYSTEM_PROMPT


def _seq(text, t0, step=0.5):
    return [
        Word(start=round(t0 + i * step, 3), end=round(t0 + i * step + 0.4, 3), text=x)
        for i, x in enumerate(text.split())
    ]


# 10.0 s: main starts; 15.0 s: an aside (dropped as DROP); 20.0 s: the teaser line
WORDS = (
    _seq("Before any of this I was a dishwasher.", 0.0)
    + _seq("My boss handed me the keys on day three.", 10.0)
    + _seq("Wait, was it day three? Maybe four.", 15.0)
    + _seq("Nobody had ever trusted me with a kitchen.", 20.0)
    + _seq("Afterwards we never talked about it.", 40.0)
)


def _pick(**kw) -> ClipPick:
    base = {
        "start": 10.0,
        "end": 24.0,
        "title": "Keys on day three",
        "hook_text": "Dishwasher to head chef",
        "reason": "r",
        "scores": RubricScores(
            hook=20, context=10, payoff=10, emotion=10, length_fit=5, pacing=5, loop=2, safety=5
        ),
        "first_words": "My boss handed me the keys on day three.",
        "verdict": "post",
        "post_caption": "Handed the keys on day three. Trust or desperation?",
        "hashtags": ["kitchenlife", "linecook", "chefstories"],
        "search_title": "Dishwasher Gets Restaurant Keys on Day Three",
        "category": "food",
    }
    base.update(kw)
    return ClipPick(**base)


def _answer(n=1, **kw):
    item = {
        "clip": n,
        "hook_text": "Handed the keys on day three",
        "title": "The day-three keys",
        "post_caption": "His boss gave him the keys on day three. Trust or desperation?",
        "hashtags": ["#kitchenlife", "fyp", "restaurantstories", "linecook"],
        "search_title": "Dishwasher Handed the Restaurant Keys on Day Three",
        "verdict": "post",
        "caption_fixes": [],
    }
    item.update(kw)
    return item


DROP = (14.7, 19.5)
TEASER = ["Nobody", "had", "ever", "trusted", "me", "with", "a", "kitchen."]
MAIN = ["My", "boss", "handed", "me", "the", "keys", "on", "day", "three."]


def test_clip_words_follow_segments_teaser_first_and_skip_drops():
    p = _pick(cold_open=(20.0, 23.9), drops=[DROP])
    texts = [w.text for w in clip_words(p, WORDS)]
    assert texts == [*TEASER, *MAIN, *TEASER]


def test_prompt_numbers_final_words_and_flags_low_probability_ones(fake_llm, settings):
    words = [w.model_copy(update={"prob": 0.2}) if w.text == "keys" else w for w in WORDS]
    fake_llm.queue({"clips": [_answer()]})
    package_clips([_pick(drops=[DROP])], words, fake_llm, settings)
    (call,) = fake_llm.calls
    assert call["schema"] is PACKAGING_SCHEMA and call["system"] == PACKAGING_SYSTEM_PROMPT
    prompt = call["prompt"]
    assert "0:My 1:boss 2:handed" in prompt and "5:keys(?)" in prompt
    assert "Maybe" not in prompt and "dishwasher." not in prompt
    assert "Dishwasher to head chef" in prompt  # selection-time packaging shown for reference


def test_cold_open_teaser_words_come_first_in_the_prompt(fake_llm, settings):
    fake_llm.queue({"clips": [_answer()]})
    package_clips([_pick(cold_open=(20.0, 23.9))], WORDS, fake_llm, settings)
    prompt = fake_llm.calls[0]["prompt"]
    assert prompt.index("0:Nobody") < prompt.index("8:My")


def test_packaging_replaces_copy_and_cleans_it(fake_llm, settings):
    fake_llm.queue({"clips": [_answer(verdict="maybe")]})
    (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.hook_text == "Handed the keys on day three"
    assert p.title == "The day-three keys"
    assert p.post_caption.endswith("Trust or desperation?")
    assert p.hashtags == ["kitchenlife", "restaurantstories", "linecook"]
    assert p.search_title == "Dishwasher Handed the Restaurant Keys on Day Three"
    assert p.verdict == "maybe"
    assert p.start == 10.0 and p.end == 24.0  # boundaries untouched


def test_caption_fixes_are_recorded_with_source_times(fake_llm, settings):
    fixes = [
        {"index": 5, "original": "keys", "text": "keys,"},
        {"index": 8, "original": "three.", "text": "three!"},
    ]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.caption_fixes == [
        CaptionFix(start=12.5, end=12.9, original="keys", text="keys,"),
        CaptionFix(start=14.0, end=14.4, original="three.", text="three!"),
    ]


def test_caption_fix_keeps_the_original_trailing_punctuation(fake_llm, settings):
    fake_llm.queue(
        {"clips": [_answer(caption_fixes=[{"index": 8, "original": "three", "text": "tree"}])]}
    )
    (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.caption_fixes == [CaptionFix(start=14.0, end=14.4, original="three.", text="tree.")]


def test_invalid_caption_fixes_are_ignored(fake_llm, settings, caplog):
    fixes = [
        {"index": 99, "original": "x", "text": "x"},  # out of range
        {"index": 2, "original": "handed", "text": "gave us all the"},  # 4 tokens: an insert
        {"index": 3, "original": "me", "text": ""},  # empty
        {"index": 1, "original": "Boss", "text": "manager"},  # fine (case-insensitive)
    ]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert [f.text for f in p.caption_fixes] == ["manager"]
    assert "caption fix" in caplog.text


def test_too_many_caption_fixes_reject_them_all(fake_llm, settings, caplog):
    # 24 words in the clip; 4 fixes is over 15%
    fixes = [{"index": i, "original": MAIN[i], "text": "word"} for i in (0, 1, 2, 3)]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.caption_fixes == [] and "15%" in caplog.text


@pytest.mark.parametrize("failure", [LLMError("boom"), {"nope": 1}])
def test_packaging_failure_keeps_selection_copy(fake_llm, settings, caplog, failure):
    fake_llm.queue(failure, failure)
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        out = package_clips([_pick()], WORDS, fake_llm, settings)
    assert out == [_pick()] and "packaging" in caplog.text


@pytest.mark.parametrize(
    "numbers", [[2], [0, 1], [1, 1], [1, 2, 3]], ids=["missing", "0-based", "duplicate", "extra"]
)
def test_clip_numbers_other_than_1_to_n_fail_the_whole_pass(fake_llm, settings, caplog, numbers):
    fake_llm.queue({"clips": [_answer(n) for n in numbers]})
    picks = [_pick(), _pick(start=40.0, end=43.0, title="Other")]
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        out = package_clips(picks, WORDS, fake_llm, settings)
    assert out is picks and "clip numbers" in caplog.text


def test_overlong_hook_text_keeps_the_selection_hook(fake_llm, settings, caplog):
    long_hook = "His boss handed him all the keys on day three"
    fake_llm.queue({"clips": [_answer(hook_text=long_hook)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.hook_text == "Dishwasher to head chef" and "7 words" in caplog.text


def test_packaging_skip_drops_the_clip_unless_it_is_the_last(fake_llm, settings, caplog):
    a, b = _pick(), _pick(start=40.0, end=43.0, title="Other")
    fake_llm.queue({"clips": [_answer(1, verdict="skip"), _answer(2)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        out = package_clips([a, b], WORDS, fake_llm, settings)
    assert [p.title for p in out] == ["The day-three keys"] and "skip" in caplog.text
    fake_llm.queue({"clips": [_answer(1, verdict="skip"), _answer(2, verdict="skip")]})
    out = package_clips([a, b], WORDS, fake_llm, settings)
    assert len(out) == 2  # never drops every clip


def test_search_title_is_clipped_without_a_dangling_word(fake_llm, settings):
    long = "Dishwasher Handed the Restaurant Keys on Day Three by a Desperate Owner for"
    fake_llm.queue({"clips": [_answer(search_title=long)]})
    (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert len(p.search_title) <= 70 and not p.search_title.endswith((" for", " a", " by"))


def test_apply_caption_fixes_replaces_matching_words_only():
    fixes = [CaptionFix(start=12.5, end=12.9, original="keys", text="keys,")]
    out = apply_caption_fixes(WORDS, fixes)
    assert [w.text for w in out if 12 <= w.start < 13] == ["the", "keys,"]
    assert len(out) == len(WORDS) and out[0] == WORDS[0]
    stale = [CaptionFix(start=12.5, end=12.9, original="locks", text="x")]
    assert apply_caption_fixes(WORDS, stale) == WORDS  # text changed underneath: not applied


def test_packaging_rules_are_shared_with_selection():
    for needle in ('"Agree or disagree?"', "#fyp", "two plausible answers", "keyword-first"):
        assert needle in PACKAGING_SYSTEM_PROMPT and needle in SYSTEM_PROMPT, needle
    for needle in ("give away the payoff", "true of the final words", "1:1"):
        assert needle in PACKAGING_SYSTEM_PROMPT, needle


# --- fix wave C review round 1 ----------------------------------------------------------------


def test_caption_fix_schema_names_the_original_word():
    item = PACKAGING_SCHEMA["properties"]["clips"]["items"]["properties"]["caption_fixes"]
    assert set(item["items"]["required"]) == {"index", "original", "text"}
    assert "`original`" in PACKAGING_SYSTEM_PROMPT


def test_off_by_one_caption_fix_is_rejected(fake_llm, settings, caplog):
    # "keys" is word 5; Claude numbered it 6 ("on"): the index and the original disagree
    fixes = [{"index": 6, "original": "keys", "text": "locks"}]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.package"):
        (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.caption_fixes == [] and "'on'" in caplog.text and "keys" in caplog.text


def test_two_words_sharing_a_start_keep_both_fixes(fake_llm, settings):
    words = [*WORDS[:13], WORDS[13].model_copy(update={"start": WORDS[12].start})]
    words += WORDS[14:]  # "the" and "keys" now both start at 12.0
    fixes = [
        {"index": 4, "original": "the", "text": "da"},
        {"index": 5, "original": "keys", "text": "kays"},
    ]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    (p,) = package_clips([_pick()], words, fake_llm, settings)
    assert [f.text for f in p.caption_fixes] == ["da", "kays"]


def test_caption_fix_keeps_the_original_leading_quote(fake_llm, settings):
    words = [w.model_copy(update={"text": "“keys"}) if w.text == "keys" else w for w in WORDS]
    fixes = [{"index": 5, "original": "keys", "text": "locks"}]
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    (p,) = package_clips([_pick()], words, fake_llm, settings)
    assert [f.text for f in p.caption_fixes] == ["“locks"]


def test_transcript_without_word_confidence_is_logged_once(fake_llm, settings, caplog):
    fake_llm.queue({"clips": [_answer(1), _answer(2)]})
    picks = [_pick(), _pick(start=40.0, end=43.0, title="Other")]
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.package"):
        package_clips(picks, WORDS, fake_llm, settings)
    assert caplog.text.count("no word confidence in this transcript") == 1
    caplog.clear()
    words = [w.model_copy(update={"prob": 0.9}) if w.text == "keys" else w for w in WORDS]
    fake_llm.queue({"clips": [_answer(1), _answer(2)]})
    with caplog.at_level(logging.INFO, logger="hookline.pipeline.package"):
        package_clips(picks, words, fake_llm, settings)
    assert "no word confidence" not in caplog.text


def test_caption_fix_original_matches_ignoring_case_and_punctuation(fake_llm, settings):
    fixes = [{"index": 8, "original": "Three", "text": "tree"}]  # the word is "three."
    fake_llm.queue({"clips": [_answer(caption_fixes=fixes)]})
    (p,) = package_clips([_pick()], WORDS, fake_llm, settings)
    assert p.caption_fixes == [CaptionFix(start=14.0, end=14.4, original="three.", text="tree.")]
