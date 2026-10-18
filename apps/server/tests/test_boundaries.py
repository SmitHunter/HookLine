import pytest

from hookline.models import ClipPick, RubricScores, Word
from hookline.pipeline.boundaries import (
    FILLER_OPENERS,
    fit_to_signals,
    include_cause,
    refine_boundaries,
    trim_idle,
)
from hookline.pipeline.signals import SIGNALS_VERSION, Signals


def pick(start, end, **kw):
    scores = RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    )
    return ClipPick(start=start, end=end, title="t", hook_text="h", reason="r", scores=scores, **kw)


def words_from(text, start, step=0.5, until=None):
    """Words at `step` spacing from `start` (gaps of 0.3*step, i.e. a filler-trim pause at the
    default step); the last word ends at `until` if given."""
    toks = text.split()
    out = [
        Word(start=start + i * step, end=start + i * step + step * 0.7, text=t)
        for i, t in enumerate(toks)
    ]
    if until is not None:
        out[-1] = out[-1].model_copy(update={"end": until})
    return out


def test_filler_list_is_verbatim():
    assert FILLER_OPENERS == (
        "and", "so", "but", "because", "like", "um", "uh", "yeah", "okay", "well",
        "i mean", "you know", "anyway",
    )  # fmt: skip


def test_trims_leading_filler():
    ws = words_from("and so the city never told us", start=10)
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == pytest.approx(ws[2].start - 0.10)  # lead-in, after the trimmed "so"


def test_trims_punctuated_and_two_word_fillers():
    ws = words_from("Okay, I mean, you know the city", start=10)
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == pytest.approx(ws[3].start - 0.10)  # "okay" + "i mean"; capped at 3


def test_trims_at_most_three_words():
    ws = words_from("and so but like the city", start=10)
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == pytest.approx(ws[3].start - 0.10)


def test_no_filler_trim_below_15s_floor():
    ws = words_from("and so the city", start=10)
    p = refine_boundaries(pick(10, 25.0), ws, [])  # trimming would leave < 15s
    assert p.start == pytest.approx(9.9)  # untrimmed; only the lead-in
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == pytest.approx(ws[2].start - 0.10)


def test_floor_stops_partway():
    ws = words_from("and so the city", start=10)
    p = refine_boundaries(pick(10, 25.8), ws, [])  # first trim ok (15.3), second not
    assert p.start == pytest.approx(ws[1].start - 0.10)


def test_snaps_start_into_preceding_silence():
    ws = words_from("the city", start=10.2)
    p = refine_boundaries(pick(10.2, 40), ws, [(9.5, 10.1)])
    assert p.start == pytest.approx(10.05)


def test_snaps_end_into_following_silence():
    ws = words_from("the city", start=10, until=39.8)
    p = refine_boundaries(pick(10, 40), ws, [(40.1, 41.0)])
    assert p.end == pytest.approx(40.15)


def test_no_silence_adds_lead_in_only():
    p = refine_boundaries(pick(10, 40), words_from("the city", start=10), [])
    assert (p.start, p.end) == (pytest.approx(9.9), 40)


def test_start_never_passes_first_word():
    ws = words_from("the city", start=10.0)
    p = refine_boundaries(pick(10.0, 40), ws, [(9.8, 11.5)])  # silence ends after the word
    assert p.start == 10.0


def test_end_never_before_last_word_end():
    ws = words_from("the city", start=10, until=39.9)
    p = refine_boundaries(pick(10, 40), ws, [(39.5, 39.9)])
    assert p.end == 39.9


def W(start, end, text):
    return Word(start=start, end=end, text=text)


def test_start_snap_does_not_cross_previous_word():
    # previous sentence's trailing word 9.7-9.95; silence ends at 9.7 -> would cut it in half
    ws = [W(9.7, 9.95, "end."), W(10.0, 10.4, "the"), W(10.5, 10.9, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [(9.2, 9.7)])
    assert p.start >= 9.95


def test_end_snap_does_not_cross_next_word():
    ws = [W(10.0, 10.4, "the"), W(39.5, 39.9, "city."), W(40.0, 40.3, "Next")]
    p = refine_boundaries(pick(10.0, 40.0), ws, [(40.3, 41.0)])  # snap would be 40.35
    assert p.end <= 40.0
    p = refine_boundaries(pick(10.0, 40.0), ws, [(39.9, 40.3)])  # inside the gap: fine
    assert p.end <= 40.0 and p.end >= 39.9


def test_snap_does_not_reinclude_trimmed_filler():
    ws = [W(10.0, 10.3, "um"), W(10.7, 10.85, "so"), W(11.0, 11.4, "the"), W(11.5, 11.9, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [(10.3, 10.7)])
    assert p.start >= 10.85  # "um" and "so" stay trimmed even though a silence sits before "so"


def test_nearest_silence_wins():
    ws = words_from("the city", start=10.0)
    p = refine_boundaries(pick(10.0, 40), ws, [(9.4, 9.7), (9.8, 9.95)])
    assert p.start == pytest.approx(9.9)


def test_curly_quotes_and_dashes_stripped_from_fillers():
    ws = words_from("“So” — the city", start=10)
    ws[1] = ws[1].model_copy(update={"text": "—"})
    ws = [ws[0], ws[2], ws[3]]
    ws[0] = ws[0].model_copy(update={"text": "“So,”"})
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == pytest.approx(ws[1].start - 0.10)


def test_empty_words_and_silences():
    assert (refine_boundaries(pick(10, 40), [], []).start, 10) == (10, 10)
    p = refine_boundaries(pick(10, 40), [], [(9.8, 10.1)])
    assert p.end == 40 and p.start == pytest.approx(10.05)
    p = refine_boundaries(pick(10, 40), words_from("the city", start=10), [])
    assert (p.start, p.end) == (pytest.approx(9.9), 40)


def test_filler_run_into_next_word_is_kept():
    # "so" flows straight into "the" (0.05 s gap, no silence): it is speech, not an opener
    ws = [W(10.0, 10.25, "so"), W(10.3, 10.6, "the"), W(10.7, 11.0, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [])
    assert p.start == pytest.approx(9.9)


def test_filler_trimmed_when_silence_separates_it():
    ws = [W(10.0, 10.25, "so"), W(10.3, 10.6, "the"), W(10.7, 11.0, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [(10.24, 10.31)])
    assert p.start >= 10.25 and p.start <= 10.3


def test_filler_trimmed_on_word_gap_of_0_12s():
    ws = [W(10.0, 10.25, "so"), W(10.37, 10.6, "the"), W(10.7, 11.0, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [])
    assert p.start == pytest.approx(10.27)  # lead-in 0.10 s, capped at the filler's end


def test_lead_in_capped_at_previous_word_end():
    ws = [W(9.7, 9.95, "end."), W(10.0, 10.4, "the"), W(10.5, 10.9, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [])
    assert p.start == pytest.approx(9.95)


def test_no_lead_in_after_silence_snap():
    ws = words_from("the city", start=10.2)
    p = refine_boundaries(pick(10.2, 40), ws, [(9.5, 10.1)])
    assert p.start == pytest.approx(10.05)


def test_loud_opener_is_kept_and_quiet_one_trimmed():
    ws = [W(10.0, 10.25, "yeah"), W(10.4, 10.7, "the"), W(10.8, 11.1, "city")]
    quiet = refine_boundaries(pick(10.0, 40), ws, [], loud_seconds=set())
    assert quiet.start > 10.2
    loud = refine_boundaries(pick(10.0, 40), ws, [], loud_seconds={10})
    assert loud.start <= 10.0
    assert refine_boundaries(pick(10.0, 40), ws, [], loud_seconds=None).start > 10.2


# --- wave G-select: idle (no speech, low motion) stretches at a clip's ends -------------------


def _signals(motion, transients=()):
    return Signals(
        duration=len(motion),
        loudness=[-40.0] * len(motion),
        version=SIGNALS_VERSION,
        motion=motion,
        onsets=list(transients),
        transients=list(transients),
    )


def test_idle_tail_is_trimmed_to_a_second_after_the_last_word(caplog):
    ws = words_from("we finally got him out", start=10, until=13.0)
    sig = _signals([20.0] * 13 + [1.0] * 30)  # gameplay, then a still death screen
    with caplog.at_level("INFO"):
        p = trim_idle(pick(10, 25), ws, sig)
    assert p.start == 10 and p.end == pytest.approx(14.0)
    assert any("idle" in r.getMessage() and "end" in r.getMessage() for r in caplog.records)


def test_tail_with_motion_a_scare_or_a_short_gap_is_kept():
    ws = words_from("we finally got him out", start=10, until=13.0)
    busy = [20.0] * 40
    assert trim_idle(pick(10, 25), ws, _signals(busy)).end == 25  # the action plays on
    still = [1.0] * 40
    assert trim_idle(pick(10, 25), ws, _signals(still, [20])).end == 25  # a stinger at 20 s
    assert trim_idle(pick(10, 15.5), ws, _signals(still)).end == 15.5  # 2.5 s: not > 3 s


def test_idle_lead_is_trimmed_to_just_before_the_first_word(caplog):
    ws = words_from("okay here it comes now", start=16, until=19.0)
    with caplog.at_level("INFO"):
        p = trim_idle(pick(10, 19.5), ws, _signals([1.0] * 30))
    assert p.start == pytest.approx(15.7) and p.end == 19.5
    assert any("idle" in r.getMessage() and "start" in r.getMessage() for r in caplog.records)


def test_trim_idle_without_motion_or_words_is_a_no_op():
    ws = words_from("we finally got him out", start=10, until=13.0)
    old = Signals(duration=40, loudness=[-40.0] * 40)
    assert trim_idle(pick(10, 25), ws, old) == pick(10, 25)
    assert trim_idle(pick(10, 25), [], _signals([1.0] * 40)) == pick(10, 25)


def test_words_in_dropped_ranges_do_not_count_as_speech():
    ws = words_from("we got him", start=10, until=11.5) + words_from("aside", start=17)
    p = trim_idle(pick(10, 25, drops=[(16.5, 18.0)]), ws, _signals([1.0] * 40))
    assert p.end == pytest.approx(12.5)


def test_a_scare_just_before_the_first_word_pulls_in_the_reaction_to_it(caplog):
    # "Oh." 4 s before the clip's "What was that?", a stinger at 13 s, another at 15.5 s
    ws = [
        *words_from("we head down", start=5, until=6.5),
        *words_from("Oh.", start=13.9),
        *words_from("Hey.", start=16.1),
        *words_from("what was that", start=18.0, until=19.5),
    ]
    sig = _signals([20.0] * 40, transients=[13, 15])
    with caplog.at_level("INFO"):
        p = include_cause(
            pick(18.0, 30, first_words="what was that", declared_first_words="what was that"),
            ws,
            sig,
        )
    assert p.start == pytest.approx(13.9) and p.end == 30
    # Q6 checks the opening the viewer now hears
    assert p.first_words == p.declared_first_words == "Oh. Hey. what was that"
    assert any("scare" in r.getMessage() for r in caplog.records)


def test_no_scare_or_no_reaction_words_keeps_the_start():
    ws = [*words_from("hey", start=16.1), *words_from("what was that", start=18.0)]
    assert include_cause(pick(18.0, 30), ws, _signals([20.0] * 40)).start == 18.0
    far = _signals([20.0] * 40, transients=[11])  # 7 s before the first word: not its cause
    assert include_cause(pick(18.0, 30), ws, far).start == 18.0
    quiet = [*words_from("what was that", start=18.0)]  # the scare, but nothing said after it
    assert include_cause(pick(18.0, 30), quiet, _signals([20.0] * 40, [16])).start == 18.0


def test_cause_lead_in_is_capped_to_whole_sentences():
    # round 1: the last 4 words of an unpunctuated run opened the clip mid-sentence
    ws = [*words_from("one two three four five six", start=14.2, step=0.6)]
    ws += words_from("what was that", start=17.8)
    p = include_cause(pick(17.8, 30), ws, _signals([20.0] * 40, transients=[14]))
    assert p.start == 17.8  # no whole sentence: nothing added
    ws[5] = ws[5].model_copy(update={"text": "six."})
    p = include_cause(pick(17.8, 30), ws, _signals([20.0] * 40, transients=[14]))
    assert p.start == 17.8  # one sentence of 6 words: over the 4-word cap, nothing added
    ws = [*words_from("No no. Get away now.", start=14.2, step=0.6)]
    ws += words_from("what was that", start=17.8)
    p = include_cause(pick(17.8, 30, first_words="what"), ws, _signals([20.0] * 40, [14]))
    assert p.start == pytest.approx(ws[2].start)  # "Get away now." (3); +"No no." would be 5
    assert p.first_words == "Get away now. what"


def test_cause_is_skipped_when_the_opener_guard_moved_the_start():
    from hookline.models import OpenerMove

    ws = [*words_from("Oh.", start=15.0), *words_from("what was that", start=17.0)]
    moved = OpenerMove(from_line=3, to_line=4, reason="weak opener")
    p = pick(17.0, 30, opener_moved=moved)
    assert include_cause(p, ws, _signals([20.0] * 40, transients=[14])) == p


def test_cause_never_prepends_pure_fillers():
    ws = [*words_from("Okay.", start=14.6), *words_from("Uh, yeah.", start=15.4)]
    ws += words_from("what was that", start=17.0)
    p = include_cause(pick(17.0, 30), ws, _signals([20.0] * 40, transients=[14]))
    assert p.start == 17.0
    ws = [*words_from("Okay.", start=14.6), *words_from("Oh.", start=15.4)]
    ws += words_from("what was that", start=17.0)
    p = include_cause(pick(17.0, 30), ws, _signals([20.0] * 40, transients=[14]))
    assert p.start == pytest.approx(15.4)  # "Oh." is the reaction; "Okay." would be trimmed


def test_cause_uses_exact_onset_times():
    # a stinger at 14.8 s (second 14): "Hm." at 14.2 was said before it, not in reaction
    ws = [*words_from("Hm.", start=14.2), *words_from("Oh.", start=15.0)]
    ws += words_from("what was that", start=17.0)
    sig = _signals([20.0] * 40, transients=[14]).model_copy(
        update={"onset_times": [14.8], "transient_times": [14.8]}
    )
    p = include_cause(pick(17.0, 30), ws, sig)
    assert p.start == pytest.approx(15.0)


def test_fit_to_signals_pulls_in_the_cause_then_trims_the_idle_tail():
    ws = [*words_from("Oh.", start=15.0), *words_from("what was that", start=17.0, until=18.5)]
    motion = [20.0] * 18 + [1.0] * 21
    p = fit_to_signals(pick(17.0, 30), ws, _signals(motion, transients=[14]))
    assert p.start == pytest.approx(15.0) and p.end == pytest.approx(19.5)


def test_idle_needs_quiet_audio_so_a_long_laugh_is_kept():
    ws = words_from("we finally got him out", start=10, until=13.0)
    sig = _signals([20.0] * 13 + [1.0] * 30)
    loud = sig.model_copy(update={"loudness": [-40.0] * 13 + [-20.0] * 30})  # laughing
    assert trim_idle(pick(10, 25), ws, loud).end == 25
    assert trim_idle(pick(10, 25), ws, sig).end == pytest.approx(14.0)
    # the quiet level is the caller's silence level
    assert trim_idle(pick(10, 25), ws, loud, quiet_db=-15.0).end == pytest.approx(14.0)


def test_trimmed_end_is_clamped_to_a_trailing_drops_start():
    # the last kept word ends at 11.5; a drop runs 12.0-26 past the clip's end: a tail end at
    # 12.5 would land inside the drop and segments() would bring the dropped audio back
    ws = words_from("we got him", start=10, until=11.5) + words_from("aside", start=12.2)
    p = trim_idle(pick(10, 25, drops=[(12.0, 26.0)]), ws, _signals([1.0] * 40))
    assert p.end == pytest.approx(12.0)
    assert all(b <= p.end for _, b in p.segments())


# --- wave G-select round 1: hard length clamp, interior idle gaps ---------------------------


def test_clamp_drops_the_cause_words_first():
    from hookline.pipeline.boundaries import clamp_length

    ws = [*words_from("Oh. No.", start=14.0), *words_from("what was that.", start=17.0)]
    ws += words_from("more talk here.", start=19.0, until=71.0)  # no voiceless stretch
    orig = pick(17.0, 71.0, first_words="what was that.")
    caused = include_cause(orig, ws, _signals([20.0] * 80, transients=[14]))
    assert caused.start == pytest.approx(14.0)
    out = clamp_length(caused, orig, ws, _signals([20.0] * 80, transients=[14]), max_s=55.0)
    assert out.start == pytest.approx(17.0) and out.first_words == "what was that."


def test_clamp_front_trims_to_a_sentence_start_and_updates_the_opening(caplog):
    from hookline.pipeline.boundaries import clamp_length

    ws = [
        *words_from("So we went in.", start=10.0),
        *words_from("It was dark.", start=14.0),
        *words_from("Then the thing jumped out.", start=20.0),
        *words_from("and that was it.", start=24.0, until=66.0),  # no voiceless stretch
    ]
    p = pick(10.0, 66.0, first_words="So we went in.", declared_first_words="So we went in.")
    sig = _signals([20.0] * 80)
    with caplog.at_level("INFO"):
        out = clamp_length(p, p, ws, sig, max_s=55.0)
    assert out.start == pytest.approx(14.0) and out.end == 66.0  # 52 s: "It was dark." on
    assert out.first_words == "It was dark." and out.declared_first_words == ""
    assert any("length" in r.getMessage() for r in caplog.records)
    assert clamp_length(p, p, ws, sig, max_s=60.0) == p  # 56 s fits


def test_clamp_counts_interior_idle_gaps_as_removed():
    from hookline.pipeline.boundaries import clamp_length

    ws = [*words_from("Here we go.", start=10.0), *words_from("Got it.", start=30.0)]
    ws += words_from("Done.", start=60.0, until=66.0)
    motion = [20.0] * 12 + [1.0] * 17 + [20.0] * 50  # still and quiet 12-29
    p = pick(10.0, 66.0)
    assert clamp_length(p, p, ws, _signals(motion), max_s=55.0) == p  # 56 - ~16 s idle


def test_interior_idle_gaps_leave_point_eight_seconds(caplog):
    from hookline.pipeline.boundaries import idle_gaps

    ws = [*words_from("Here we go.", start=10.0, until=11.5), *words_from("Oh.", start=20.0)]
    ws += words_from("Got it.", start=25.0)
    motion = [20.0] * 12 + [1.0] * 8 + [20.0] * 30  # still 12-20
    with caplog.at_level("INFO"):
        gaps = idle_gaps(pick(10.0, 27.0), ws, _signals(motion))
    assert gaps == [pytest.approx((12.4, 19.6))]  # 12-20 still, quiet, no words: 0.8 s left
    assert any("idle" in r.getMessage() for r in caplog.records)


def test_interior_idle_gap_needs_still_quiet_scareless_and_over_two_seconds():
    from hookline.pipeline.boundaries import idle_gaps

    ws = [*words_from("Here we go.", start=10.0, until=11.5), *words_from("Oh.", start=20.0)]
    still = [20.0] * 12 + [1.0] * 8 + [20.0] * 30
    p = pick(10.0, 22.0)
    assert idle_gaps(p, ws, _signals([20.0] * 50)) == []  # the picture moves
    assert idle_gaps(p, ws, _signals(still, transients=[16])) == []  # a stinger inside
    loud = _signals(still).model_copy(update={"loudness": [-20.0] * 50})
    assert idle_gaps(p, ws, loud) == []  # a laugh, game audio: not quiet
    short = [20.0] * 12 + [1.0] * 2 + [20.0] * 40
    assert idle_gaps(p, ws, _signals(short)) == []  # 2 s: not over 2 s
    assert idle_gaps(p, ws, Signals(duration=50, loudness=[-40.0] * 50)) == []  # no motion
