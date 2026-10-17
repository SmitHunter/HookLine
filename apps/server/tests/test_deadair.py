import pytest

from hookline.models import Word
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges


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


def test_snap_ranges_snaps_to_frame_grid():
    assert snap_ranges([(1.01, 2.03)], 25) == [(1.0, 2.04)]


def test_snap_ranges_merges_small_gaps_only_forward():
    keep = [(1.0, 2.0), (2.1, 3.0), (3.48, 4.0), (0.2, 0.8)]
    assert snap_ranges(keep, 25) == [(1.0, 3.0), (3.48, 4.0), (0.2, 0.8)]


def test_snap_ranges_drops_empty():
    assert snap_ranges([(1.0, 1.01), (2.0, 3.0)], 25) == [(2.0, 3.0)]


def test_cut_never_lands_inside_a_word():
    # silence 2.0-4.0 overlaps "didn't" (1.5-2.6): only the word-free part (padded) is cut
    words = [w(1.5, 2.6, "didn't")]
    assert keep_ranges([(0, 10)], [(2.0, 4.0)]) == [(0, 2.08), (3.92, 10)]
    keep = keep_ranges([(0, 10)], [(2.0, 4.0)], words=words)
    assert keep == [(0, 2.76), (3.92, 10)]


def test_silence_split_by_word_only_cuts_long_pieces():
    # padded word 4.92-5.28 splits the silence: 4.7-4.92 (0.22 s) stays, 5.28-7.0 is cut
    words = [w(5.0, 5.2, "uh")]
    keep = keep_ranges([(0, 10)], [(4.7, 7.0)], words=words)
    assert keep == [(0, 5.36), (6.92, 10)]


def test_remap_keeps_word_clipped_to_a_sliver():
    out = remap_words([w(1.0, 1.5, "didn't")], [(0.0, 1.013)])
    assert len(out) == 1 and out[0].end - out[0].start == pytest.approx(0.05)
    assert out[0].end <= 1.013 + 1e-9


def test_remap_sliver_at_range_start_extends_forward_within_range():
    out = remap_words([w(0.5, 1.007, "x")], [(1.0, 2.0)])
    assert (out[0].start, out[0].end) == pytest.approx((0.0, 0.05))


def test_remap_does_not_stretch_uncut_short_words():
    out = remap_words([w(1.0, 1.02, "a")], [(0.0, 2.0)])
    assert out[0].end - out[0].start == pytest.approx(0.02)
