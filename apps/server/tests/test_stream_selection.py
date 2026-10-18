"""Stream selection: death / respawn screens shown to the selector and the packager, clips kept
on one side of a death screen, spread across encounters, punchline and catch endings, opener
re-checks after a start move, reaction openers backed onto their cause sound, cold-open
repeats, hook rules and unsupported packaging specifics."""

import logging

import pytest

from hookline.models import ClipPick, RubricScores, Word
from hookline.pipeline.signals import SIGNALS_VERSION, Signals

FULL = 0b1111  # every frame of a second is a screen frame (4 fps)


def pick(start, end, **kw):
    scores = RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    )
    base = {"title": "t", "hook_text": "h", "reason": "r", "scores": scores}
    return ClipPick(start=start, end=end, **{**base, **kw})


def W(start, end, text="word"):
    return Word(start=start, end=end, text=text)


def sentence(start, text, step=0.3):
    toks = text.split()
    return [W(start + i * step, start + i * step + step * 0.8, t) for i, t in enumerate(toks)]


def sig(n=200, screen=(), scares=(), onsets=None, loud=()):
    """`n` s of busy gameplay; `screen` (lo, hi) second ranges are full screen seconds;
    `onsets` every audio onset (default: the scares); `loud` loud seconds."""
    mask = [0] * n
    for lo, hi in screen:
        for s in range(lo, hi):
            mask[s] = FULL
    every = list(scares) if onsets is None else sorted({*onsets, *scares})
    loudness = [-20.0] * n
    for s in loud:
        loudness[s] = 0.0
    return Signals(
        duration=n,
        loudness=loudness,
        version=SIGNALS_VERSION,
        motion=[20.0] * n,
        onsets=[int(t) for t in every],
        transients=[int(t) for t in scares],
        onset_times=every,
        transient_times=list(scares),
        screen=mask,
    )


# --- the selector sees death screens ----------------------------------------------------------


def test_screen_runs_are_marked_on_the_transcript():
    from hookline.pipeline.signals import SCREEN_MARKER, heat_markers

    m = heat_markers(sig(screen=[(30, 46), (80, 82)]))
    assert m[30] == [f"{SCREEN_MARKER} 16s"]
    assert 80 not in m  # a 2 s flicker is no death screen
    assert SCREEN_MARKER == "⬛screen"


def test_selection_prompt_covers_deaths_encounters_and_climax_cold_opens():
    from hookline.pipeline.select import SYSTEM_PROMPT
    from hookline.pipeline.signals import SCREEN_MARKER

    assert SCREEN_MARKER in SYSTEM_PROMPT
    for needle in (
        "Never span a death or respawn",
        "at most one clip per encounter",
        "ends in success or the climax",
        "use the payoff line itself as `cold_open_words`",
    ):
        assert needle in SYSTEM_PROMPT, needle


def _ranked(*spans, verdict="maybe"):
    return [pick(a, b, category="gaming", verdict=verdict, title=f"c{a}") for a, b in spans]


def test_diversify_demotes_a_pick_within_180s_of_a_higher_one():
    from hookline.pipeline.select import diversify

    a, b, c = _ranked((1000, 1030), (1100, 1130), (2000, 2030))
    assert diversify([a, b, c]) == [a, c, b]


def test_diversify_demotes_a_pick_in_the_same_stretch_of_repeated_deaths():
    from hookline.pipeline.select import diversify

    a, b, c = _ranked((1000, 1030), (1300, 1330), (3000, 3030))
    deaths = [(1050.0, 1066.0), (1120.0, 1136.0), (1190.0, 1206.0), (1260.0, 1276.0)]
    assert diversify([a, b, c], deaths) == [a, c, b]
    assert diversify([a, b, c]) == [a, b, c]  # 270 s apart and no deaths between: distinct


# --- a death screen inside a clip -----------------------------------------------------------------


def _two_attempts():
    """Attempt one (10-29 s), a 10 s death screen (30-40), attempt two (41-60)."""
    words = [*sentence(10, "Okay here we go again."), *sentence(14, "Run run run run!")]
    words += [*sentence(20, "He is right behind me now."), *sentence(26, "Oh no no no!")]
    words += [*sentence(41, "Alright second try now."), *sentence(46, "Left then right.")]
    words += [*sentence(52, "I made it out of there."), *sentence(57, "Never again.")]
    return words


def test_a_death_screen_inside_a_clip_keeps_the_side_holding_the_payoff(caplog):
    from hookline.pipeline.boundaries import keep_payoff_side

    words = _two_attempts()
    p = pick(10.0, 58.0, first_words="Okay here we go again.", declared_first_words="x")
    with caplog.at_level(logging.INFO):
        out = keep_payoff_side(p, words, sig(screen=[(30, 40)]))
    assert out.start == pytest.approx(41.0) and out.end == 58.0
    assert out.first_words == "Alright second try now." and out.declared_first_words == ""
    assert "death" in caplog.text
    # the scares and the heat are on attempt one, and it ends on the death: that side
    hot = sig(screen=[(30, 40)], scares=[22.0, 27.0, 28.5], loud=range(20, 30))
    out = keep_payoff_side(p, words, hot)
    assert (out.start, out.end) == (10.0, 30.0)
    assert out.last_words == "Oh no no no!"


def test_a_death_screen_with_no_viable_side_rejects_the_clip(caplog):
    from hookline.pipeline.boundaries import keep_payoff_side

    words = [*sentence(10, "Okay go."), *sentence(40, "Again.")]
    with caplog.at_level(logging.WARNING):
        assert keep_payoff_side(pick(10.0, 45.0), words, sig(screen=[(14, 38)])) is None
    assert "rejected" in caplog.text


def test_short_screens_and_screens_at_the_edges_leave_the_clip_alone():
    from hookline.pipeline.boundaries import keep_payoff_side

    words = _two_attempts()
    p = pick(10.0, 58.0)
    assert keep_payoff_side(p, words, sig(screen=[(30, 34)])) is p  # 4 s: not a death
    assert keep_payoff_side(p, words, sig(screen=[(55, 70)])) is p  # runs past the end


def test_a_reaction_said_over_the_death_screen_stays_with_the_side_before():
    from hookline.pipeline.boundaries import keep_payoff_side

    words = [w for w in _two_attempts() if w.start < 30] + sentence(30.4, "Are you kidding me?")
    words += sentence(41, "Alright.")
    hot = sig(screen=[(30, 40)], scares=[22.0, 27.0, 29.6])
    out = keep_payoff_side(pick(10.0, 42.0), words, hot)
    assert out.start == 10.0 and out.end == pytest.approx(words[-2].end)


def test_packaging_is_told_where_the_death_screens_are(fake_llm, settings):
    from hookline.pipeline.package import PACKAGING_SYSTEM_PROMPT, package_clips

    words = _two_attempts()
    fake_llm.queue({"clips": []})
    package_clips([pick(10.0, 58.0)], words, fake_llm, settings, screens=[(30.0, 40.0)])
    prompt = fake_llm.calls[0]["prompt"]
    clip = next(x for x in prompt.splitlines() if x.startswith("Clip:"))
    assert "no!" in clip.split("[death screen]")[0].lower()
    assert "Alright" in clip.split("[death screen]")[1]
    assert "`[death screen]`" in PACKAGING_SYSTEM_PROMPT


# --- spreading picks across a stream -------------------------------------------------------------


def test_select_clips_demotes_picks_near_each_other(fake_llm, settings):
    from hookline.pipeline.select import select_clips
    from tests.conftest import _synthetic_transcript
    from tests.test_select import raw  # 3 s sentence grid

    tr = _synthetic_transcript(900.0)
    s = settings.model_copy(update={"clips": 2})
    a = raw(30, 60, total=60, category="gaming")
    b = raw(120, 150, total=55, category="gaming")
    c = raw(600, 630, total=40, category="gaming")
    fake_llm.queue({"clips": [a, b, c]})
    assert [round(p.start) for p in select_clips(tr, fake_llm, s)] == [30, 600]


# --- punchline and catch endings -------------------------------------------------------------------


def test_punchline_guard_measures_from_the_last_word_not_the_padded_end():
    from hookline.pipeline.boundaries import extend_payoff

    words = sentence(10, "You think the run button would make him run?")
    last = words[-1].end
    padded = last + 0.4
    # the scare 1.8 s after the last word (1.4 s after the padded end): no extension
    assert extend_payoff(pick(10, padded), words, sig(scares=[last + 1.8]), 60).end == padded
    # within 1.5 s of the last word and nothing new said: extended to keep the catch
    out = extend_payoff(pick(10, padded), words, sig(scares=[last + 1.2]), 60)
    assert out.end >= last + 1.2 + 1.5 - 1e-6


def test_punchline_never_extends_into_a_new_sentence():
    from hookline.pipeline.boundaries import extend_payoff

    words = sentence(10, "You think the run button would make him run?")
    last = words[-1].end
    more = [*words, *sentence(last + 1.0, "Who? Who? Dodge!")]
    p = pick(10, last + 0.2)
    assert extend_payoff(p, more, sig(scares=[last + 1.2]), 60) is p


def test_a_closing_scare_keeps_its_catch():
    from hookline.pipeline.boundaries import extend_payoff

    words = [*sentence(10, "We are running away now"), W(28.0, 30.0, "nooo")]  # a shout
    # a death screen right after the scare, nobody reacting: none of it (uncovered)
    out = extend_payoff(pick(10, 30.0), words, sig(scares=[37.8], screen=[(38, 50)]), 60)
    assert out.end == pytest.approx(38.0)
    # the streamer reacts over the death screen: the cam shows it, >= 1.5 s kept
    react = [*words, W(38.0, 38.9, "Nooo!")]
    out = extend_payoff(pick(10, 30.0), react, sig(scares=[37.8], screen=[(38, 50)]), 60)
    assert out.end >= 37.8 + 1.5 - 1e-6
    # never a 0.3 s sliver of the catch: when 1.5 s after it does not fit, no extension
    dense = [W(10 + i * 0.5, 10 + i * 0.5 + 0.4, "w") for i in range(40)]  # to 29.9 s
    base = pick(10, 30.0)
    assert extend_payoff(base, dense, sig(scares=[32.0]), max_s=22.6) is base
    assert extend_payoff(base, dense, sig(scares=[32.0]), max_s=24.0).end >= 33.5 - 1e-6


def test_a_voiced_closing_reaction_over_a_death_screen_is_not_cut():
    from hookline.pipeline.boundaries import static_gaps

    words = [*sentence(10, "We are running away now"), W(38.0, 38.9, "Nooo!")]
    s = sig(scares=[37.8], screen=[(38, 50)])
    assert static_gaps(pick(10, 39.6), words, s) == []
    # an interior screen is still cut
    assert static_gaps(pick(10, 60), [*words, *sentence(52, "Okay again.")], s)


# --- opener re-check after a start move ----------------------------------------------------------


def test_a_start_moved_onto_a_reply_moves_to_the_next_good_opener(caplog):
    from hookline.pipeline.boundaries import fit_to_signals

    s = sig(screen=[(53, 69)])
    words = [*sentence(67.6, "Game plan! Game plan!"), *sentence(69.5, "No, game time!")]
    words += [*sentence(70.6, "Game time!"), *sentence(80, "I do not want a hug!")]
    words += [*sentence(92, "He is right there.")]
    with caplog.at_level(logging.INFO):
        p = fit_to_signals(pick(67.6, 94.0), words, s, max_s=50.0)
    assert p.start == pytest.approx(70.6) and p.first_words == "Game time!"
    assert "reply" in caplog.text


def test_a_reply_opener_moves_back_to_its_question_within_3s():
    from hookline.pipeline.boundaries import recheck_opener

    words = [*sentence(20, "Did you call her?"), *sentence(21.6, "No, I never did.")]
    words += [*sentence(24, "She called me first."), *sentence(40, "That was it.")]
    p = recheck_opener(pick(21.6, 41.0, first_words="No, I never did."), words, sig())
    assert p.start == pytest.approx(20.0)
    assert p.first_words.startswith("Did you call her?")
    # a clean opener is left alone
    clean = pick(24.0, 41.0)
    assert recheck_opener(clean, words, sig()) is clean


def test_reply_openers_are_weak_at_selection():
    from hookline.pipeline.select import _bad_opener
    from tests.test_select import _seq

    assert _bad_opener(_seq("No, game time!"))
    assert _bad_opener(_seq("Nope, not this one."))
    assert not _bad_opener(_seq("No one saw it coming."))


# --- reaction openers back onto their cause sound -----------------------------------------------


def test_a_reaction_opener_backs_off_to_the_previous_audio_onset(caplog):
    from hookline.pipeline.boundaries import include_cause_sound

    words = [*sentence(10, "Okay let me look around."), *sentence(17.1, "Hey.")]
    words += [*sentence(19.1, "The fuck was that?"), *sentence(22, "I hate this place.")]
    p = pick(19.1, 45.0, first_words="The fuck was that?", declared_first_words="The fuck")
    with caplog.at_level(logging.INFO):
        out = include_cause_sound(p, words, sig(onsets=[16.9]))
    assert out.start == pytest.approx(16.8)
    assert out.first_words == "Hey. The fuck was that?"
    assert "cause sound" in caplog.text
    # a sound with no words after it: the clip starts just before it
    out = include_cause_sound(p, words[:5] + words[6:], sig(onsets=[17.5]))
    assert out.start == pytest.approx(17.4) and out.first_words == "The fuck was that?"


def test_a_reaction_opener_without_a_nearby_onset_or_with_a_scare_is_left():
    from hookline.pipeline.boundaries import include_cause_sound

    words = [*sentence(10, "Okay let me look around."), *sentence(19.1, "What the hell?")]
    p = pick(19.1, 45.0)
    assert include_cause_sound(p, words, sig(onsets=[14.0])) is p  # over 4 s before
    assert include_cause_sound(p, words, sig(scares=[17.0])) is p  # include_cause's job
    calm = pick(10.0, 45.0)
    assert include_cause_sound(calm, words, sig(onsets=[8.0])) is calm  # not a reaction


# --- cold opens, drop lines, hook caps ---------------------------------------------------------------


def test_cold_open_dropped_when_its_line_recurs_in_the_body_within_10s(fake_llm, settings, caplog):
    from tests.test_select import _FILLER, _guard_item, _one

    line = "I really do not want a hug!"
    tr, item = _guard_item({15: line, 16: line}, _FILLER.format(n=10), cold_open_words=line)
    (p,) = _one(tr, fake_llm, settings, item, caplog)
    assert p.cold_open is None and "recurs" in caplog.text


def test_surprising_fact_with_comment_risk_caps_the_hook_at_15(fake_llm, settings):
    from tests.test_select import _FILLER, _guard_item, _one

    for risk, hook in (("medium", 15), ("high", 15), ("low", 22)):
        tr, item = _guard_item({}, _FILLER.format(n=10), hook_type="surprising_fact")
        item.update(comment_risk=risk)
        item["scores"]["hook"] = 22
        (p,) = _one(tr, fake_llm, settings, item)
        assert p.scores.hook == hook, risk


def test_ignored_drop_lines_are_recorded_with_the_reason(fake_llm, settings):
    from tests.test_select import _FILLER, _guard_item, _one

    tr, item = _guard_item({}, _FILLER.format(n=10), drop_lines=[10, 25])
    (p,) = _one(tr, fake_llm, settings, item)
    assert any("L10" in n and "not inside" in n for n in p.ignored_drops)


def test_a_yes_word_does_not_answer_a_wh_question(fake_llm, settings):
    from tests.test_select import _FILLER, _guard_item, _one

    over = {13: "What is the time?", 14: "Yeah, that is not scary at all."}
    tr, item = _guard_item(over, _FILLER.format(n=10), drop_lines=[13])
    (p,) = _one(tr, fake_llm, settings, item)
    assert len(p.drops) == 1 and p.ignored_drops == []


# --- hooks and packaging specifics ---------------------------------------------------------------


def _pkg_words():
    words = sentence(0.0, "I think Grimbo is behind the vent.", step=0.5)
    words += sentence(10.0, "Okay we go left at the big door.", step=0.5)
    words += sentence(20.0, "He grabbed me right at the end.", step=0.5)
    words += sentence(40.0, "Grimbo is back and he is fast.", step=0.5)
    words += sentence(50.0, "Run to the lift right now.", step=0.5)
    return words


def _answer(n, **kw):
    item = {
        "clip": n,
        "hook_text": "One door left to reach",
        "alt_hook_text": "",
        "title": "Left at the big door",
        "post_caption": "He went left at the door. Right call or panic?",
        "hashtags": ["horrorgame", "grimbo", "jumpscare"],
        "search_title": "Grimbo Chase Left at the Big Door",
        "verdict": "post",
        "caption_fixes": [],
    }
    item.update(kw)
    return item


def _pkg(fake_llm, settings, answers, picks=None):
    from hookline.pipeline.package import package_clips

    picks = picks or [
        pick(10.0, 25.0, hook_text="Grabbed at the last door", verdict="post"),
        pick(40.0, 54.0, hook_text="He is back and faster", verdict="post"),
    ]
    fake_llm.queue({"clips": answers})
    return package_clips(picks, _pkg_words(), fake_llm, settings)


def test_a_name_is_used_in_at_most_one_hook_per_batch(fake_llm, settings):
    a = _answer(1, hook_text="Grimbo waits behind the door")
    b = _answer(2, hook_text="Grimbo closes in fast", alt_hook_text="Forty metres to the lift")
    one, two = _pkg(fake_llm, settings, [a, b])
    assert one.hook_text == "Grimbo waits behind the door"
    assert two.hook_text == "Forty metres to the lift"


def test_an_entity_plus_label_hook_falls_back_to_the_selection_hook(fake_llm, settings):
    a = _answer(1, hook_text="Grimbo's vent ad: the last line", alt_hook_text="Grimbo: the door")
    b = _answer(2, hook_text="Nobody makes it to the lift")
    one, two = _pkg(fake_llm, settings, [a, b])
    assert one.hook_text == "Grabbed at the last door"
    assert two.hook_text == "Nobody makes it to the lift"


def test_unsupported_specifics_in_packaging_fall_back(fake_llm, settings, caplog):
    a = _answer(
        1,
        hook_text="The scariest door in 1998",
        alt_hook_text="He goes left at the door",
        title="First Grimbo escape ever",
        search_title="First Grimbo Escape in the Vents, 1998",
        post_caption="The worst door in the game. Left or right?",
    )
    with caplog.at_level(logging.WARNING):
        one, _ = _pkg(fake_llm, settings, [a, _answer(2, hook_text="Nobody makes the lift")])
    assert one.hook_text == "He goes left at the door"
    assert one.title == "t" and one.search_title == "" and one.post_caption == ""
    assert "unsupported" in caplog.text


def test_supported_specifics_are_kept():
    from hookline.pipeline.package import unsupported_specifics

    source = [
        "this",
        "was",
        "our",
        "first",
        "deployment",
        "to",
        "the",
        "gulf",
        "in",
        "1991",
        "the",
        "worst",
        "year",
    ]
    assert unsupported_specifics("Their first deployment, 1991", source) == []
    assert unsupported_specifics("The worst year", source) == []
    assert unsupported_specifics("First Gulf War deployment", source) == ["first"]
    assert unsupported_specifics("An honest interest in the forest", source) == []


def test_packaging_prompt_states_the_hook_and_specifics_rules():
    from hookline.pipeline.package import PACKAGING_SYSTEM_PROMPT

    for needle in ("generic label", "never lead two hooks", "ordinal"):
        assert needle in PACKAGING_SYSTEM_PROMPT, needle


# --- the first declared word is never clipped -------------------------------------------------------


def test_render_start_covers_a_first_word_whose_onset_moved_earlier():
    from hookline.pipeline.boundaries import cover_first_word

    raw = [W(17.1, 17.25, "Hey."), W(19.08, 19.23, "The"), W(19.48, 19.63, "fuck")]
    near = [W(17.1, 17.25, "Hey."), W(19.0, 19.48, "The"), W(19.48, 19.76, "fuck")]
    p = pick(19.08, 30.0)
    assert cover_first_word(p, raw, near).start == pytest.approx(19.0)
    assert cover_first_word(pick(19.0, 30.0), raw, near).start == 19.0


def test_q6_fails_when_the_first_word_is_clipped():
    from hookline.quality import _check_opening

    words = [W(0.1, 0.3, "fuck"), W(0.4, 0.6, "was"), W(0.7, 1.2, "that?")]
    res = _check_opening(words, "The fuck was that?", None, actual_first_words="The fuck was that?")
    assert not res.ok and "clipped" in res.detail
    good = [W(0.0, 0.1, "The"), *words]
    assert _check_opening(good, "The fuck was that?", None).ok


def test_run_keeps_one_side_of_each_death_screen_before_packaging():
    from hookline.pipeline.run import _one_side_of_deaths

    words = _two_attempts()
    keep = pick(10.0, 58.0)
    gone = pick(10.0, 45.0)
    logs: list[str] = []
    s = sig(screen=[(14, 38)])  # wholly inside both; only the first has a viable side
    out = _one_side_of_deaths([keep, gone], words, s, logs.append)
    assert [(p.start, p.end) for p in out] == [(41.0, 58.0)]
    assert any("dropped" in x for x in logs)
    # never drops every pick
    assert _one_side_of_deaths([gone], words, s, logs.append) == [gone]
    noted = pick(10.0, 58.0, ignored_drops=["L48 ignored: it is a question"])
    _one_side_of_deaths([noted], words, sig(), logs.append)
    assert any("L48 ignored" in x for x in logs)


# --- a death screen inside a clip: what the fit keeps ----------------------------------------------


def _reaction_over_screen():
    words = [w for w in _two_attempts() if w.start < 30] + sentence(30.4, "Are you kidding me?")
    words += sentence(41, "Alright.")
    return words, sig(screen=[(30, 40)], scares=[22.0, 27.0, 29.6])


def test_the_reaction_kept_over_a_death_screen_survives_the_fit():
    """The side before a screen keeps the reaction said over it (10-31.54 s); the fit's
    static-end check cut it again to 29.75 s while packaging had described it."""
    from hookline.pipeline.boundaries import fit_to_signals, keep_payoff_side

    words, hot = _reaction_over_screen()
    side = keep_payoff_side(pick(10.0, 42.0), words, hot)
    out = fit_to_signals(pick(10.0, 42.0), words, hot, max_s=65)
    assert out.end == side.end and out.end >= words[-2].end - 1e-6
    assert out.last_words == side.last_words == "Are you kidding me?"
    again = fit_to_signals(out, words, hot, max_s=65)  # the render fits the side again
    assert (again.start, again.end) == (out.start, out.end)


def test_an_end_moved_off_a_silent_screen_updates_last_words():
    from hookline.pipeline.boundaries import avoid_static

    words = [*sentence(10, "Run run run!"), *sentence(25, "Oh no.")]
    out = avoid_static(pick(10.0, 35.0, last_words="x"), words, sig(screen=[(30, 40)]))
    assert out.end < 30 and out.last_words == "Oh no."


def test_the_side_after_a_screen_gets_its_opener_rechecked():
    from hookline.pipeline.boundaries import fit_to_signals, keep_payoff_side

    words = sentence(10, "Okay here we go now.") + sentence(14, "Run run run!")
    words += sentence(20, "Oh no.") + sentence(41, "No, not again.")
    words += sentence(44, "Alright second try now.") + sentence(50, "I made it out of there.")
    words += sentence(55, "Never again.")
    s = sig(screen=[(25, 40)], scares=[52.0])
    side = keep_payoff_side(pick(10.0, 57.0), words, s)
    assert side.start == pytest.approx(44.0) and side.first_words == "Alright second try now."
    out = fit_to_signals(pick(10.0, 57.0), words, s, max_s=55)
    assert out.start == pytest.approx(44.0)


def test_the_side_after_a_screen_starts_on_a_sentence_start():
    """A sentence begun over the screen ("I swear this game is / so unfair to me."): the side
    starts on the next sentence, not on "so"."""
    from hookline.pipeline.boundaries import keep_payoff_side

    words = sentence(10, "Okay here we go now.") + sentence(14, "Run run run!")
    words += sentence(20, "Oh no.") + sentence(38.5, "I swear this game is")
    words += sentence(40.3, "so unfair to me.") + sentence(44, "Alright second try now.")
    words += sentence(50, "I made it out of there.") + sentence(55, "Never again.")
    out = keep_payoff_side(pick(10.0, 57.0), words, sig(screen=[(25, 40)], scares=[52.0]))
    assert out.start == pytest.approx(44.0) and out.first_words == "Alright second try now."


def test_the_cause_reaction_is_not_undone_by_the_opener_check():
    """'No, no, no, no!' right after a scare is the reaction include_cause pulled in, not a
    reply opener."""
    from hookline.pipeline.boundaries import fit_to_signals

    words = sentence(5, "Let me check the vent.") + sentence(12.2, "No, no, no, no!")
    words += sentence(14, "I hate this game so much.") + sentence(20, "Where do I go now?")
    words += sentence(30, "Okay this way.")
    p = pick(14.0, 34.0, first_words="I hate this game so much.")
    out = fit_to_signals(p, words, sig(scares=[12.0]), max_s=50)
    assert out.start == pytest.approx(12.2) and out.first_words.startswith("No, no, no, no!")


def test_keeping_a_side_clears_the_opener_move_and_beats_outside_it():
    from hookline.models import OpenerMove
    from hookline.pipeline.boundaries import keep_payoff_side

    moved = OpenerMove(from_line=3, to_line=4, reason="reply")
    beats = ["zoom: 'Run' @ 14.2s", "zoom: 'made' @ 52.3s"]
    p = pick(10.0, 58.0, opener_moved=moved, edit_beats=beats)
    out = keep_payoff_side(p, _two_attempts(), sig(screen=[(30, 40)]))
    assert out.start == pytest.approx(41.0)
    assert out.opener_moved is None and out.edit_beats == ["zoom: 'made' @ 52.3s"]


def test_a_side_shorter_than_the_minimum_clip_is_no_clip():
    from hookline.pipeline.boundaries import keep_payoff_side
    from hookline.pipeline.run import _one_side_of_deaths

    words = _two_attempts()
    p = pick(10.0, 58.0)
    s = sig(screen=[(30, 40)])
    assert keep_payoff_side(p, words, s, min_s=21.0) is None  # sides of 20 s and 17 s
    assert keep_payoff_side(p, words, s, min_s=17.0) is not None
    logs: list[str] = []
    out = _one_side_of_deaths([p, pick(70.0, 100.0)], words, s, logs.append, min_s=21.0)
    assert [x.start for x in out] == [70.0]


def test_screen_markers_are_death_screens():
    from hookline.pipeline.boundaries import DEATH_SCREEN_MIN_S
    from hookline.pipeline.signals import SCREEN_MARKER_MIN_S

    assert SCREEN_MARKER_MIN_S == DEATH_SCREEN_MIN_S


def test_a_reaction_opener_backs_onto_its_cause_sound_on_gameplay_only():
    from hookline.pipeline.boundaries import fit_to_signals

    words = sentence(10, "And then he told me the price.")
    words += sentence(14.5, "Oh my god, that is insane.") + sentence(18, "Right?")
    p = pick(14.5, 40.0, first_words="Oh my god, that is insane.")
    assert fit_to_signals(p, words, sig(onsets=[13.0])).start == 14.5  # a talk show
    assert fit_to_signals(p, words, sig(onsets=[13.0]), gameplay=True).start < 14.5


# --- cold-open repeats, answers ------------------------------------------------------------------


def test_cold_open_repeats_count_only_inside_the_clip():
    from hookline.pipeline.select import _recurs

    line = "I really do not want a hug!"
    words = sentence(14, line) + sentence(17, "Okay.") + sentence(20, line) + sentence(30, "Go.")
    assert _recurs(words, 20.0, 22.0)  # said 6 s before
    assert not _recurs(words, 20.0, 22.0, lo=17.0, hi=31.0)  # ... outside the clip


def test_a_climax_teaser_is_kept_when_its_body_repeat_is_3_s_away():
    from hookline.pipeline.select import _recurs

    line = "I did it, I made it out!"
    words = sentence(10, "Okay go go go.") + sentence(30, line) + sentence(36, line)
    words += sentence(38, "Unbelievable.")
    # the payoff line at the clip's end said 6 s before: heard three times, but far apart
    assert not _recurs(words, 36.0, 37.9, lo=10.0, hi=39.0)
    close = sentence(10, "Okay go go go.") + sentence(34.5, line) + sentence(36.5, line)
    assert _recurs(close, 36.5, 38.3, lo=10.0, hi=39.0)  # back to back: dropped


def test_answers_wh_test_reads_the_question_word():
    from hookline.pipeline.select import _answers, _tokens

    assert _answers("Whoa, did you see that?", _tokens("Yeah, I did."))
    assert _answers("Whatever, are you in?", _tokens("Yes, I am."))
    assert not _answers("So what did you do there?", _tokens("Yeah, it was great."))
    assert not _answers("What's the time?", _tokens("Yeah, that is not scary."))


# --- Q6: the declared first words ----------------------------------------------------------------


def _said(text: str) -> list:
    return [W(i * 0.5, i * 0.5 + 0.4, t) for i, t in enumerate(text.split())]


def test_q6_first_word_ignores_selection_fillers():
    from hookline.quality import first_word_clipped

    assert first_word_clipped("Oh, what was that?", _said("What was that?")) is None
    assert first_word_clipped("Umm the thing is", _said("the thing is")) is None
    assert first_word_clipped("Erm, okay, go!", _said("go!")) is None
    assert first_word_clipped("The fuck was that?", _said("fuck was that?")) == "the"


def test_q6_first_word_allows_contractions_and_near_spellings():
    from hookline.quality import first_word_clipped

    assert first_word_clipped("What is that thing", _said("What's that thing")) is None
    assert first_word_clipped("Hugie is coming", _said("Huggie is coming")) is None


def test_q6_compares_after_the_caption_fixes():
    """first_words quote the transcript ("Kimble"); the captions show the fix ("Grimbo")."""
    from hookline.quality import _check_opening

    words = _said("Grimbo is coming for me right now")
    assert not _check_opening(words, "Kimble is coming for me", None).ok
    fixes = [{"start": 3.0, "end": 3.2, "original": "Kimble", "text": "Grimbo"}]
    assert _check_opening(words, "Kimble is coming for me", None, fixes=fixes).ok


# --- packaging: names and specifics ----------------------------------------------------------------


def test_proper_names_need_two_capitalised_uses_and_no_lowercase_one():
    """Unpunctuated transcripts: every word is 'mid-sentence'."""
    from hookline.pipeline.package import _proper_names

    text = "so then Grimbo came out and Then I'm like It's over and Grimbo got me then we ran Left"
    assert _proper_names(_said(text)) == {"grimbo"}


def test_viewer_questions_and_number_words_are_no_unsupported_specifics():
    from hookline.pipeline.package import unsupported_specifics
    from hookline.pipeline.select import _tokens

    src = _tokens("this is my fifth attempt and it feels like the nineties again")
    assert unsupported_specifics("Best scare or worst?", src) == []
    assert unsupported_specifics("His fifth attempt", src) == []
    assert unsupported_specifics("A 5th attempt", src) == []
    assert unsupported_specifics("Back in the 1990s", src) == []
    assert unsupported_specifics("The worst attempt. Agree?", src) == ["worst"]


def test_fallback_packaging_is_checked_for_specifics_too():
    from hookline.pipeline.package import _supported
    from hookline.pipeline.select import _tokens

    src = _tokens("we made it out of the vent")
    out = _supported("title", "The first vent escape", "The fastest vent escape", src, "t")
    assert out == "The vent escape"
