import pytest

from hookline.models import Word
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges


def w(s, e, t):
    return Word(start=s, end=e, text=t)


def test_cuts_long_pause_keeps_padding():
    assert keep_ranges([(10, 20)], [(12, 13)]) == [(10, 12.12), (12.88, 20)]


def test_short_pause_kept():
    assert keep_ranges([(10, 20)], [(12, 12.3)]) == [(10, 20)]
    assert keep_ranges([(10, 20)], [(12, 12.6)]) == [(10, 20)]  # default max_pause 0.6


def test_cut_leaving_short_piece_drops_smaller_cut():
    # cuts 2.12-2.88 (0.76 s) and 3.92-4.88 (0.96 s) would leave a 1.04 s piece between them
    assert keep_ranges([(0, 10)], [(2.0, 3.0), (3.8, 5.0)]) == [(0, 3.92), (4.88, 10)]


def test_cut_leaving_short_edge_piece_is_dropped():
    assert keep_ranges([(0, 10)], [(0.5, 1.5)]) == [(0, 10)]  # 0.62 s head piece
    assert keep_ranges([(0, 10)], [(8.5, 9.6)]) == [(0, 10)]  # 0.52 s tail piece


def test_min_piece_iterates_until_stable():
    # three cuts 1.2 s apart: dropping one still leaves a short piece, so another goes too
    silences = [(2.0, 3.0), (3.9, 5.2), (6.2, 7.0)]
    keep = keep_ranges([(0, 12)], silences)
    pieces = [b - a for a, b in keep]
    assert all(p >= 1.5 for p in pieces)
    assert keep == [(0, 4.02), (5.08, 12)]
    assert keep_ranges([(0, 12)], silences, min_piece_s=0.5) == [
        (0, 2.12), (2.88, 4.02), (5.08, 6.32), (6.88, 12),
    ]  # fmt: skip


def test_dead_air_settings_defaults():
    from hookline.config import Settings

    s = Settings()
    assert s.max_pause_s == 0.6 and s.dead_air_pad_s == 0.12


def test_no_silences_is_identity():
    assert keep_ranges([(10, 20)], []) == [(10, 20)]


def test_refuses_to_eat_most_of_clip():
    assert keep_ranges([(0, 10)], [(1, 9)]) == [(0, 10)]


def test_cold_open_segments_processed_independently():
    assert keep_ranges([(50, 52), (30, 55)], [(40, 41)]) == [
        (50, 52),
        (30, 40.12),
        (40.88, 55),
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
    assert keep_ranges([(0, 10)], [(2.0, 4.0)]) == [(0, 2.12), (3.88, 10)]
    keep = keep_ranges([(0, 10)], [(2.0, 4.0)], words=words)
    assert keep == [(0, 2.84), (3.88, 10)]


def test_silence_split_by_word_only_cuts_long_pieces():
    # padded word 4.88-5.32 splits the silence: 4.7-4.88 (0.18 s) stays, 5.32-7.0 is cut
    words = [w(5.0, 5.2, "uh")]
    keep = keep_ranges([(0, 10)], [(4.7, 7.0)], words=words)
    assert keep == [(0, 5.44), (6.88, 10)]


def test_remap_keeps_word_clipped_to_a_sliver():
    # 40 ms kept (over one 30 fps frame): stretched to 50 ms, inside the range
    out = remap_words([w(1.0, 1.5, "didn't")], [(0.0, 1.04)])
    assert len(out) == 1 and out[0].end - out[0].start == pytest.approx(0.05)
    assert out[0].end <= 1.04 + 1e-9


def test_remap_sliver_at_range_start_extends_forward_within_range():
    out = remap_words([w(0.5, 1.04, "x")], [(1.0, 2.0)])
    assert (out[0].start, out[0].end) == pytest.approx((0.0, 0.05))


@pytest.mark.parametrize(
    ("word", "fps"),
    [((1.0, 1.5), None), ((1.0, 1.5), 30.0), ((0.5, 1.007), None), ((1.0, 1.5), 60.0)],
)
def test_remap_drops_a_word_clipped_to_under_one_frame(word, fps):
    # real case: "order." kept for 3 ms at a cold-open seam showed as a 0.05 s sliver
    keep = [(0.0, 1.013)] if word[0] == 1.0 else [(1.0, 2.0)]
    assert remap_words([w(*word, "order.")], keep, fps) == []


def test_remap_one_frame_threshold_follows_the_fps():
    # 25 ms kept: under a 30 fps frame (33 ms) but over a 60 fps frame (17 ms)
    keep = [(0.0, 1.025)]
    assert remap_words([w(1.0, 1.5, "x")], keep, 30.0) == []
    (out,) = remap_words([w(1.0, 1.5, "x")], keep, 60.0)
    assert out.end - out.start == pytest.approx(0.05)


def test_remap_does_not_stretch_uncut_short_words():
    out = remap_words([w(1.0, 1.02, "a")], [(0.0, 2.0)])
    assert out[0].end - out[0].start == pytest.approx(0.02)


def test_snap_ranges_never_merges_an_explicit_drop():
    keep = [(1.0, 2.0), (2.12, 3.0), (3.04, 4.0)]
    assert snap_ranges(keep, 25, keep_gaps=[(2.0, 2.12)]) == [(1.0, 2.0), (2.12, 4.0)]
