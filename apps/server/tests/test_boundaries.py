import pytest

from hookline.models import ClipPick, RubricScores, Word
from hookline.pipeline.boundaries import FILLER_OPENERS, refine_boundaries


def pick(start, end, **kw):
    scores = RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    )
    return ClipPick(start=start, end=end, title="t", hook_text="h", reason="r", scores=scores, **kw)


def words_from(text, start, step=0.5, until=None):
    """Words at `step` spacing from `start`; the last word ends at `until` if given."""
    toks = text.split()
    out = [
        Word(start=start + i * step, end=start + i * step + step * 0.8, text=t)
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
    assert p.start == ws[2].start


def test_trims_punctuated_and_two_word_fillers():
    ws = words_from("Okay, I mean, you know the city", start=10)
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == ws[3].start  # "okay" + "i mean"... capped at 3 words


def test_trims_at_most_three_words():
    ws = words_from("and so but like the city", start=10)
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == ws[3].start


def test_no_filler_trim_below_15s_floor():
    ws = words_from("and so the city", start=10)
    p = refine_boundaries(pick(10, 25.0), ws, [])  # trimming would leave < 15s
    assert p.start == 10
    p = refine_boundaries(pick(10, 40), ws, [])
    assert p.start == ws[2].start


def test_floor_stops_partway():
    ws = words_from("and so the city", start=10)
    p = refine_boundaries(pick(10, 25.8), ws, [])  # first trim ok (15.3), second not
    assert p.start == ws[1].start


def test_snaps_start_into_preceding_silence():
    ws = words_from("the city", start=10.2)
    p = refine_boundaries(pick(10.2, 40), ws, [(9.5, 10.1)])
    assert p.start == pytest.approx(10.05)


def test_snaps_end_into_following_silence():
    ws = words_from("the city", start=10, until=39.8)
    p = refine_boundaries(pick(10, 40), ws, [(40.1, 41.0)])
    assert p.end == pytest.approx(40.15)


def test_no_silence_leaves_bounds():
    p = refine_boundaries(pick(10, 40), words_from("the city", start=10), [])
    assert (p.start, p.end) == (10, 40)


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
    ws = [W(10.0, 10.3, "um"), W(10.7, 10.9, "so"), W(11.0, 11.4, "the"), W(11.5, 11.9, "city")]
    p = refine_boundaries(pick(10.0, 40), ws, [(10.3, 10.7)])
    assert p.start >= 10.9  # "um" and "so" stay trimmed even though a silence sits before "so"


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
    assert p.start == ws[1].start


def test_empty_words_and_silences():
    assert (refine_boundaries(pick(10, 40), [], []).start, 10) == (10, 10)
    p = refine_boundaries(pick(10, 40), [], [(9.8, 10.1)])
    assert p.end == 40 and p.start == pytest.approx(10.05)
    p = refine_boundaries(pick(10, 40), words_from("the city", start=10), [])
    assert (p.start, p.end) == (10, 40)
