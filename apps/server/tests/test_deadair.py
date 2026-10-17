import pytest

from hookline.models import Word
from hookline.pipeline.deadair import keep_ranges, remap_words


def w(s, e, t):
    return Word(start=s, end=e, text=t)


def test_cuts_long_pause_keeps_padding():
    assert keep_ranges([(10, 20)], [(12, 13)]) == [(10, 12.08), (12.92, 20)]


def test_short_pause_kept():
    assert keep_ranges([(10, 20)], [(12, 12.3)]) == [(10, 20)]


def test_no_silences_is_identity():
    assert keep_ranges([(10, 20)], []) == [(10, 20)]


def test_refuses_to_eat_most_of_clip():
    assert keep_ranges([(0, 10)], [(1, 9)]) == [(0, 10)]


def test_cold_open_segments_processed_independently():
    assert keep_ranges([(50, 52), (30, 55)], [(40, 41)]) == [
        (50, 52),
        (30, 40.08),
        (40.92, 55),
    ]


def test_remap_shifts_words_after_cut():
    keep = [(10, 12.08), (12.92, 20)]
    out = remap_words([w(11, 11.5, "a"), w(13, 13.4, "b")], keep)
    assert [(x.start, x.end) for x in out] == [(1.0, 1.5), pytest.approx((2.16, 2.56))]


def test_remap_duplicates_words_in_cold_open():
    out = remap_words([w(50.5, 51, "x")], [(50, 52), (30, 55)])
    assert [round(x.start, 2) for x in out] == [0.5, 22.5]
