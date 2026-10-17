import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from hookline.captions.ass import build_ass, caption_y_for, group_words, hook_y_for
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import Word
from hookline.pipeline.layout import SceneLayout


def w(s, e, t):
    return Word(start=s, end=e, text=t)


def _events(ass):
    return [ln for ln in ass.splitlines() if ln.startswith("Dialogue")]


def test_group_breaks_on_punctuation_and_gap():
    g = group_words([w(0, 0.3, "No"), w(0.3, 0.6, "way."), w(0.7, 1, "I"), w(2, 2.3, "swear")])
    assert [[x.text for x in grp] for grp in g] == [["No", "way."], ["I"], ["swear"]]


def test_group_breaks_on_word_and_char_limits():
    ws = [w(i * 0.2, i * 0.2 + 0.2, "ab") for i in range(4)]
    assert [len(x) for x in group_words(ws)] == [3, 1]
    long = [w(0, 0.2, "abcdefghij"), w(0.2, 0.4, "klmnopqrs")]
    assert [len(x) for x in group_words(long)] == [1, 1]


def test_ass_highlights_each_word_once():
    ass = build_ass([w(0, 0.3, "no"), w(0.3, 0.6, "way")], hook_text="Wait for it", duration=5)
    events = _events(ass)
    assert (
        len(events) == 3
        and "{\\c&H00FFFF&}NO{\\c&HFFFFFF&} WAY" in events[1]
        and "NO {\\c&H00FFFF&}WAY" in events[2]
    )


def test_ass_header_and_styles():
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=640)
    for s in ("PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes"):
        assert s in ass
    assert "Style: Cap,Anton,136," in ass and "Style: Hook,Anton,136," in ass
    assert "{\\pos(474,700)}" in ass and "{\\pos(474,360)\\fad" in ass  # 640 clamped to 700
    hook = next(e for e in _events(ass) if ",Hook," in e)
    assert hook.startswith("Dialogue: 1,0:00:00.00,0:00:03.00,")


def test_hook_clamped_to_duration_and_wrapped():
    ass = build_ass(
        [w(0, 0.3, "no")], hook_text="this is a very long hook title here", duration=1.5
    )
    hook = next(e for e in _events(ass) if ",Hook," in e)
    assert ",0:00:01.50," in hook
    assert hook.count("\\N") == 1
    assert "THIS IS A VERY" in hook


def test_ass_escapes_special_chars():
    ass = build_ass([w(0, 0.5, "{hi}\\")], hook_text="x", duration=1)
    assert "\\{HI\\}\\" + chr(0x2060) in ass  # a word joiner stops "\" escaping what follows


def test_normalises_slivers_and_split_fragments():
    ws = [
        w(0, 0.02, "tiny"),
        w(0, 0.4, "hello"),
        w(0.4, 0.8, "hello"),
        w(0.82, 1.0, "hello"),
        w(1.0, 1.4, "x"),
    ]
    ass = build_ass(ws, hook_text="h", duration=5)
    cap = [e for e in _events(ass) if ",Cap," in e]
    assert len(cap) == 2
    assert "TINY" not in ass
    assert cap[0].count("HELLO") == 1 and ",0:00:00.00," in cap[0]
    # merged fragments: first start .. last end (1.0)
    assert "0:00:01.00" in cap[0]


def test_min_duration_and_centisecond_format():
    ass = build_ass([w(1.0, 1.06, "a"), w(1.06, 1.06, "b")], hook_text="h", duration=5)
    for e in _events(ass):
        _, start, end, _ = e.split(",", 3)
        assert start != end and start < end


def test_caption_y_for():
    st = SceneLayout(start=0, end=5, kind="STACKED")
    ft = SceneLayout(start=0, end=5, kind="FIT_BLUR")
    assert caption_y_for([st, ft]) == 700
    assert caption_y_for([ft, st]) == 1180
    assert caption_y_for([]) == 1180


def test_subtitles_filter_escaping():
    f = subtitles_filter(Path("C:/a b/it's/c.ass"), Path("C:/fonts dir"))
    assert f.startswith("subtitles='") and "C\\:/a b/it'\\\\\\''s/c.ass" in f
    assert ":fontsdir='C\\:/fonts dir'" in f


def test_fonts_dir_has_anton():
    assert (fonts_dir() / "Anton-Regular.ttf").exists()


@pytest.mark.integration
def test_burns_captions_in_emoji_space_dir(fixture_video, tmp_path):
    d = tmp_path / "it's my clips 🎮"
    d.mkdir()
    ass = d / "c.ass"
    ass.write_text(build_ass([w(0, 0.5, "héllo"), w(0.5, 1, "😱")], "Hook!", 2), encoding="utf-8")
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-t",
            "2",
            "-i",
            str(fixture_video),
            "-vf",
            subtitles_filter(ass, fonts_dir()),
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert r.returncode == 0, r.stderr[-1500:]


def test_word_clipped_to_13ms_by_a_cut_still_captioned():
    from hookline.pipeline.deadair import remap_words

    words = [w(0.0, 0.4, "But"), w(0.5, 0.9, "I"), w(1.0, 1.5, "didn't")]
    cap = remap_words(words, [(0.0, 1.013), (1.6, 2.0)])
    assert "DIDN'T" in build_ass(cap, "", 1.5)


def _lay(kind, face_box=None):
    return SceneLayout(start=0, end=5, kind=kind, face_box=face_box)


def test_hook_and_caption_placement_per_layout():
    assert (hook_y_for([]), caption_y_for([])) == (360, 1180)
    assert (hook_y_for([_lay("FIT_BLUR")]), caption_y_for([_lay("FIT_BLUR")])) == (360, 1180)
    stacked = [_lay("STACKED"), _lay("FIT_BLUR")]
    assert (hook_y_for(stacked), caption_y_for(stacked)) == (900, 700)  # below the seam
    assert hook_y_for([_lay("FACE_TRACK")]) == 360  # no face box: default


def test_face_track_hook_moves_off_the_face():
    low = [_lay("FACE_TRACK", (0.3, 0.25))]  # face 576-1056 px: clear of hook and captions
    assert (hook_y_for(low), caption_y_for(low)) == (360, 1180)
    high = [_lay("FACE_TRACK", (0.1, 0.3))]  # face top 192 px: under the hook band
    assert (hook_y_for(high), caption_y_for(high)) == (920, 1180)


def test_face_track_captions_move_off_a_face_covering_them():
    small = [_lay("FACE_TRACK", (0.5, 0.12))]  # face 960-1190 px covers the 1180 line
    assert caption_y_for(small) == 840  # no room below (max 1180): above the face
    clear = [_lay("FACE_TRACK", (0.25, 0.3))]  # face 480-1056 px: clear of the 1180 line
    assert caption_y_for(clear) == 1180
    big = [_lay("FACE_TRACK", (0.3, 0.5))]  # face 576-1536 px: no room either side
    assert caption_y_for(big) == 1180 and hook_y_for(big) == 360


def test_split_hook_uses_top_panel_face():
    assert hook_y_for([_lay("SPLIT", (0.1, 0.5))]) == 920  # 96 px into the 960 px panel
    assert hook_y_for([_lay("SPLIT", (0.55, 0.3))]) == 360  # 528 px: below the band


def test_build_ass_hook_position():
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=900, hook_y=860)
    assert "{\\pos(474,860)\\fad" in ass and "{\\pos(474,900)}" in ass


# --- M2.5: rendered size, coverage, safe zones (Q2-Q5) ---------------------------------

Q4_X = (60, 888)
Q4_Y = (290, 1250)


def _render(tmp_path, ass_text, t):
    from hookline.quality import render_ass_frame

    p = tmp_path / "m.ass"
    p.write_text(ass_text, encoding="utf-8")
    return render_ass_frame(p, t)


def _inside_q4(box, pad=0):
    x0, y0, x1, y1 = box
    return (
        Q4_X[0] <= x0 - pad and x1 + pad <= Q4_X[1] and Q4_Y[0] <= y0 - pad and y1 + pad <= Q4_Y[1]
    )


def test_text_bbox_on_synthetic_frame():
    import numpy as np

    from hookline.quality import text_bbox

    f = np.zeros((1920, 1080, 3), np.uint8)
    f[100:150, 200:260] = (255, 255, 0)  # yellow
    f[300:310, 400:420] = (255, 255, 255)
    assert text_bbox(f, "yellow") == (200, 100, 259, 149)
    assert text_bbox(f, "white") == (400, 300, 419, 309)
    assert text_bbox(f, "any") == (200, 100, 419, 309)
    assert text_bbox(np.zeros_like(f)) is None


def test_rendered_caption_cap_height(tmp_path):
    from hookline.quality import text_bbox

    frame = _render(tmp_path, build_ass([w(0, 1, "test")], "", 2), 0.5)
    box = text_bbox(frame, "yellow")  # the active word
    assert box is not None
    assert 64 <= box[3] - box[1] + 1 <= 72


def test_rendered_hook_cap_height(tmp_path):
    from hookline.quality import text_bbox

    frame = _render(tmp_path, build_ass([], "test", 4), 0.5)
    box = text_bbox(frame, "white")
    assert box is not None and box[3] - box[1] + 1 >= 64
    assert _inside_q4(box)


def _cap_intervals(ass):
    out = []
    for e in _events(ass):
        if ",Cap," not in e:
            continue
        _, start, end, _ = e.split(",", 3)
        out.append((_secs(start), _secs(end)))
    return sorted(out)


def _secs(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_coverage_holds_groups_across_short_gaps():
    words = [w(round(i * 0.7, 2), round(i * 0.7 + 0.4, 2), f"w{i}") for i in range(20)]
    iv = _cap_intervals(build_ass(words, "", 20))
    for (_, e0), (s1, _) in pairwise(iv):
        assert e0 <= s1 + 1e-9  # never two captions at once
    span = (words[0].start, words[-1].end)
    covered = sum(min(e, span[1]) - max(s, span[0]) for s, e in iv if e > span[0] and s < span[1])
    assert covered / (span[1] - span[0]) >= 0.95


def test_coverage_long_gap_ends_shortly_after_last_word():
    words = [w(0, 0.4, "one"), w(0.5, 0.9, "two."), w(2.5, 2.9, "three")]
    iv = _cap_intervals(build_ass(words, "", 10))
    assert iv[1][1] == pytest.approx(1.2)  # "two." ends 0.9 + 0.3; the 1.6 s gap is not held
    assert iv[-1][1] == pytest.approx(3.2)  # last group: word end + 0.3
    held = _cap_intervals(build_ass([w(0, 0.4, "a."), w(1.3, 1.6, "b")], "", 10))
    assert held[0][1] == pytest.approx(1.3)  # 0.9 s gap: held until the next group


def test_held_last_word_stays_highlighted():
    ass = build_ass([w(0, 0.4, "one"), w(0.5, 0.9, "two."), w(1.5, 2, "three")], "", 5)
    cap = [e for e in _events(ass) if ",Cap," in e]
    assert "0:00:01.50" in cap[1].split(",")[2]
    assert "{\\c&H00FFFF&}TWO." in cap[1]


def test_hook_at_most_two_lines_of_18_chars():
    for text in ("Parks. Open space. All under siege.", "x", "This hook is long but fits"):
        hook = next(e for e in _events(build_ass([], text, 5)) if ",Hook," in e)
        body = hook.split("}")[-1]
        lines = body.split("\\N")
        assert len(lines) <= 2 and all(len(x) <= 18 for x in lines), lines
    too_long = "this is a very long hook title that keeps on going"  # > 2 x 18 chars
    hook = next(e for e in _events(build_ass([], too_long, 5)))
    assert hook.count("\\N") == 1 and "\\fscx" in hook  # condensed, never a third line


def test_hook_style_box_and_fade():
    ass = build_ass([], "Hi there", 5)
    style = next(ln for ln in ass.splitlines() if ln.startswith("Style: Hook"))
    f = style.split(",")
    assert f[3] == "&H00FFFFFF" and f[5] == "&H26000000" and f[15] == "3"  # 85% black box
    hook = next(e for e in _events(ass) if ",Hook," in e)
    assert "\\fad(0,250)" in hook and ",0:00:03.00," in hook


def test_caption_style_outline_and_shadow():
    ass = build_ass([w(0, 1, "x")], "", 2)
    f = next(ln for ln in ass.splitlines() if ln.startswith("Style: Cap")).split(",")
    assert (f[16], f[17]) == ("10", "4") and f[6] == "&H80000000"  # semi-transparent shadow


def test_caption_y_clamped_to_safe_band():
    assert "{\\pos(474,1180)}" in build_ass([w(0, 1, "x")], "", 2, caption_y=1600)
    assert "{\\pos(474,700)}" in build_ass([w(0, 1, "x")], "", 2, caption_y=400)


def test_rendered_boxes_inside_q4_at_band_extremes(tmp_path):
    from hookline.quality import text_bbox

    words = [w(0, 1, "INCREDIBLE"), w(1, 2, "TEST")]
    for cy, hy in ((1180, 360), (700, 360), (1180, 920)):
        ass = build_ass(words, "Parks. Open space. All under siege.", 3, caption_y=cy, hook_y=hy)
        box = text_bbox(_render(tmp_path, ass, 0.5), "any")
        assert box is not None and _inside_q4(box, pad=10), (cy, hy, box)  # 10 px outline


def test_wide_caption_group_shrinks_to_828(tmp_path):
    from hookline.quality import text_bbox

    ass = build_ass([w(0, 1, "SUPERCALIFRAGILISTICEXPIALIDOCIOUS")], "", 2)
    assert "\\fscx" in ass and "\\fscy" in ass
    box = text_bbox(_render(tmp_path, ass, 0.5), "any")
    assert box is not None and box[2] - box[0] + 1 + 20 <= 828 and _inside_q4(box, pad=10)


def test_trailing_backslash_does_not_break_highlight_reset(tmp_path):
    from hookline.quality import text_bbox

    ass = build_ass([w(0, 1, "foo\\"), w(1, 2, "bar")], "", 3)
    frame = _render(tmp_path, ass, 0.5)
    yellow, white = text_bbox(frame, "yellow"), text_bbox(frame, "white")
    assert yellow is not None and white is not None
    assert yellow[2] < white[0]  # "BAR" is white again, right of the yellow "FOO\"


def test_newline_in_word_text_becomes_space():
    ass = build_ass([w(0, 1, "a\nb"), w(1, 2, "c\r\nd")], "", 3)
    assert all("\n" not in e for e in _events(ass))
    assert "A B" in ass and "C D" in ass


def test_group_breaks_after_closing_quote():
    for end in ('."', '!"', '?"'):
        g = group_words([w(0, 0.3, "said" + end), w(0.3, 0.6, "Then")])
        assert len(g) == 2, end
