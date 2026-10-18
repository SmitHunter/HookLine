import re
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from hookline.captions.ass import (
    CAPTION_ABOVE,
    CAPTION_BELOW,
    CAPTION_LATE_MIN_Y,
    CAPTION_MIN_Y,
    build_ass,
    caption_y_for,
    face_rects,
    group_words,
    hook_y_for,
)
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
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=600)
    for s in ("PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes"):
        assert s in ass
    assert "Style: Cap,Anton,152," in ass and "Style: Hook,Anton,136," in ass
    assert "{\\pos(474,689)}" in ass  # 600 clamped to 689 (48 px below a two-line hook)
    assert "{\\pos(474,378)\\fad" in ass  # 360 clamped: one-line box top at 290
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


def test_word_clipped_to_40ms_by_a_cut_still_captioned():
    from hookline.pipeline.deadair import remap_words

    words = [w(0.0, 0.4, "But"), w(0.5, 0.9, "I"), w(1.0, 1.5, "didn't")]
    cap = remap_words(words, [(0.0, 1.04), (1.6, 2.0)])
    assert "DIDN'T" in build_ass(cap, "", 1.5)


def _lay(kind, face_box=None):
    return SceneLayout(start=0, end=5, kind=kind, face_box=face_box)


def test_hook_and_caption_placement_per_layout():
    assert (hook_y_for([]), caption_y_for([])) == (360, 1180)
    assert (hook_y_for([_lay("FIT_BLUR")]), caption_y_for([_lay("FIT_BLUR")])) == (360, 1180)
    stacked = [_lay("STACKED"), _lay("FIT_BLUR")]
    assert (hook_y_for(stacked), caption_y_for(stacked)) == (965, 700)  # below the seam
    assert hook_y_for([_lay("FACE_TRACK")]) == 360  # no face box: default


def _hook_box(tmp_path, layouts, text="Parks. Open space. All under siege."):
    return _grey(tmp_path, build_ass([], text, 3, caption_y_for(layouts), hook_y_for(layouts)))


def _cap_box(tmp_path, layouts):
    return _grey(tmp_path, build_ass([w(0, 2, "INCREDIBLE")], "", 3, caption_y_for(layouts)))


def _face_share(box, rect):
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = rect
    ov = max(0, min(x1, fx1) - max(x0, fx0)) * max(0, min(y1, fy1) - max(y0, fy0))
    return ov / ((x1 - x0 + 1) * (y1 - y0 + 1))


def test_face_track_hook_moves_off_the_face(tmp_path):
    low = [_lay("FACE_TRACK", (0.32, 0.23))]  # face 614-1056 px: clear of hook and captions
    assert (hook_y_for(low), caption_y_for(low)) == (360, 1180)
    high = [_lay("FACE_TRACK", (0.1, 0.3))]  # face 192-768 px (1.2x: 134-826) under the hook
    hy, cy = hook_y_for(high), caption_y_for(high)
    assert cy == 1180 and hy > 900  # below the face, still above the captions
    hook, cap = _hook_box(tmp_path, high), _cap_box(tmp_path, high)
    assert cap[1] - hook[3] >= 48  # visible separation
    ((rect,),) = [face_rects(lay) for lay in high]
    assert _face_share(hook, rect) == 0.0  # Q4b for the hook box


def test_face_track_captions_move_off_a_face_covering_them():
    small = [_lay("FACE_TRACK", (0.5, 0.12))]  # face 960-1190 px covers the 1180 line
    y = caption_y_for(small)  # 1180 is the band bottom: move above the face core
    # 1.2x punch-in: box 937-1213, core (top 20% removed) from 992
    assert 760 <= y and y + CAPTION_BELOW + 40 <= 1075 - 138 + 0.2 * 276
    clear = [_lay("FACE_TRACK", (0.25, 0.3))]  # face 480-1056 px (1.2x: 422-1114): clear
    assert caption_y_for(clear) == 1180
    big = [_lay("FACE_TRACK", (0.32, 0.5))]  # face 614-1574 px covers every line
    assert hook_y_for(big) == 360
    # late line: lowest one clear of the 1.2x face core (box 518-1670, core from 748)
    assert caption_y_for(big, late=True) + CAPTION_BELOW + 40 <= 748


def test_close_up_captions_go_above_the_head_clear_of_the_hook(tmp_path):
    # a real podcast close-up: face 687-1332 px covers 1180 and reaches past the safe zone
    close = [_lay("FACE_TRACK", (0.358, 0.336))]
    # 1.2x punch-in: box 622-1397, face core (top 20% = hairline removed) from 777
    core_top = 1009.5 - 1.2 * 322.5 + 0.2 * 2 * 1.2 * 322.5
    y = caption_y_for(close)
    # the face covers every line: the highest one the 48 px hook separation allows, its
    # visible text still above the core at a full punch-in (the 40 px face gap gives way)
    assert y == CAPTION_MIN_Y and y + CAPTION_BELOW <= core_top
    hook, cap = _hook_box(tmp_path, close), _cap_box(tmp_path, close)
    assert hook is not None and cap is not None and cap[1] - hook[3] >= 48
    # after the hook: the same rule from 553 up; the lowest clear line is still above the core
    late = caption_y_for(close, late=True)
    assert CAPTION_LATE_MIN_Y <= late and late + CAPTION_BELOW + 40 <= core_top
    from hookline.quality import text_bbox

    box = text_bbox(
        _render(tmp_path, build_ass([w(0, 2, "HUGGING")], "", 3, 900, 360, None, late), 1.0)
    )
    ((rect,),) = [face_rects(lay, 1.2) for lay in close]
    assert _face_share(box, rect) == 0.0  # Q4b even at full punch-in


def test_late_caption_line_used_only_after_hook_and_tag():
    ws = [w(0.2, 0.6, "ONE."), w(3.2, 3.6, "TWO."), w(5.0, 5.4, "THREE."), w(7.0, 7.4, "FOUR.")]
    ass = build_ass(ws, "Hook here", 9, caption_y=900, seam=4.8, caption_y_late=600)
    caps = [e for e in _events(ass) if ",Cap," in e]
    ys = [int(e.split("\\pos(474,")[1].split(")")[0]) for e in caps]
    assert ys == [900, 600, 900, 600]  # hook up; free; EARLIER tag 4.8-6.0 up; free
    # ONE ends 0.6 then TWO at 3.2 (> 1 s): tail; TWO. ends 3.6, held until the seam (4.8)
    assert caps[1].startswith("Dialogue: 0,0:00:03.13,0:00:03.90,")  # 2 frames early
    ass = build_ass([w(4.0, 4.4, "BEFORE"), w(5.0, 5.4, "AFTER")], "", 9, seam=4.6)
    caps = [e for e in _events(ass) if ",Cap," in e]
    assert caps[0].startswith("Dialogue: 0,0:00:03.93,0:00:04.60,")  # not bleeding past seam


def test_caption_line_weighs_every_face_scene_by_duration():
    long_close = SceneLayout(start=0, end=30, kind="FACE_TRACK", face_box=(0.358, 0.336))
    short_clear = SceneLayout(start=30, end=33, kind="FACE_TRACK", face_box=(0.2, 0.2))
    assert caption_y_for([short_clear, long_close], late=True) < 700  # the close-up decides
    split = [SceneLayout(start=0, end=5, kind="SPLIT", face_box=(0.3, 0.5))]
    # faces 288-768 and 1248-1728 px; at 1.2x the bottom face core starts at 1315, so the
    # default line clears both cores
    assert caption_y_for(split) == 1180
    tall = [SceneLayout(start=0, end=5, kind="SPLIT", face_box=(0.1, 0.5))]
    # bottom face 1056-1536, 1.2x core from 1123: the line moves up above that core
    y = caption_y_for(tall)
    assert y < 1180 and y + CAPTION_BELOW + 40 <= 1123


def test_split_hook_uses_top_panel_face():
    # face 96-576 px in the top panel (1.2x: 48-624): the hook moves below it, above captions
    split = [_lay("SPLIT", (0.1, 0.5))]
    hy, cy = hook_y_for(split), caption_y_for(split)
    assert 600 < hy and hy + 156 + 24 <= cy - CAPTION_ABOVE
    assert hook_y_for([_lay("SPLIT", (0.65, 0.3))]) == 360  # 624 px: below the band


def test_build_ass_hook_position():
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=900, hook_y=860)
    assert "{\\pos(474,860)\\fad" in ass and "{\\pos(474,900)}" in ass


# --- M2.5: rendered size, coverage, safe zones (Q2-Q5) ---------------------------------

Q4_X = (60, 888)
Q4_Y = (290, 1250)


def _render(tmp_path, ass_text, t, bg="black"):
    from hookline.quality import render_ass_frame

    p = tmp_path / "m.ass"
    p.write_text(ass_text, encoding="utf-8")
    return render_ass_frame(p, t, bg)


def _grey(tmp_path, ass_text, t=0.5):
    """Everything visible (text, outline, shadow, hook box) over a mid-grey canvas."""
    from hookline.quality import GREY, box_bbox

    return box_bbox(_render(tmp_path, ass_text, t, GREY))


def _first_line_cap_h(frame):
    """Height of the first band of white text rows (the first hook line's caps)."""
    from hookline.quality import text_mask

    rows = text_mask(frame, "white").any(axis=1).nonzero()[0]
    assert len(rows)
    end = rows[0]
    while end + 1 in set(rows):
        end += 1
    return int(end - rows[0] + 1)


def _hook(ass):
    return next(e for e in _events(ass) if ",Hook," in e)


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
    assert 72 <= box[3] - box[1] + 1 <= 80  # wave D: ~12% larger (75-76 px)


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
    # 0.9 s gap: held until the next group appears (2 frames before its word at 1.3)
    assert held[0][1] == pytest.approx(1.3 - 2 / 30, abs=0.006)


def test_held_last_word_stays_highlighted():
    ass = build_ass([w(0, 0.4, "one"), w(0.5, 0.9, "two."), w(1.5, 2, "three")], "", 5)
    cap = [e for e in _events(ass) if ",Cap," in e]
    assert "0:00:01.43" in cap[1].split(",")[2]  # until THREE's page appears
    assert "{\\c&H00FFFF&}TWO." in cap[1]


def test_hook_at_most_two_lines_of_18_chars():
    for text in ("Parks. Open space. All under siege.", "x", "This hook is long but fits"):
        hook = next(e for e in _events(build_ass([], text, 5)) if ",Hook," in e)
        body = hook.split("}")[-1]
        lines = body.split("\\N")
        assert len(lines) <= 2 and all(len(x) <= 18 for x in lines), lines
    too_long = "this is a very long hook title that keeps on going"  # > 2 x 18 chars
    hook = next(e for e in _events(build_ass([], too_long, 5)))
    assert hook.count("\\N") == 1  # two balanced lines, never a third


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
    assert "{\\pos(474,689)}" in build_ass([w(0, 1, "x")], "", 2, caption_y=400)
    late = build_ass([w(0, 1, "x")], "", 2, caption_y=400, caption_y_late=300)
    assert "{\\pos(474,553)}" in late  # the late line may rise to 553, no further


def test_rendered_boxes_inside_q4_at_band_extremes(tmp_path):
    words = [w(0, 1, "INCREDIBLE"), w(1, 2, "TEST")]
    for cy, hy in ((1180, 360), (700, 360), (1180, 920), (1180, 2000), (700, 0)):
        ass = build_ass(words, "Parks. Open space. All under siege.", 3, caption_y=cy, hook_y=hy)
        box = _grey(tmp_path, ass)  # includes the hook's background box and caption outline
        assert box is not None and _inside_q4(box), (cy, hy, box)


def test_two_line_hook_box_top_stays_in_q4():
    hook = _hook(build_ass([], "Parks. Open space. All under siege.", 4))
    assert "\\pos(474,446)" in hook  # 290 + 2 lines x 68 + 20 px box pad


def _tag(event, name):
    import re

    m = re.search(rf"\\{name}(\d+)", event)
    return int(m.group(1)) if m else None


def test_seven_long_words_hook_steps_size_not_squash(tmp_path):
    ass = build_ass([], "Women remember moments when summer mornings mattered", 4)
    hook = _hook(ass)
    assert (_tag(hook, "fscx") or 100) >= 80
    assert _tag(hook, "fs") is not None  # 80% alone was not enough: the size stepped down
    assert "…" not in hook  # 7 words: never truncated
    assert _first_line_cap_h(_render(tmp_path, ass, 0.5)) >= 60
    box = _grey(tmp_path, ass)
    assert box is not None and _inside_q4(box), box


def test_ten_word_hook_too_wide_is_cut_to_seven_words(tmp_path):
    text = "everybody remembers the moment the internet stopped being free forever"
    ass = build_ass([], text, 4)
    hook = _hook(ass)
    body = hook.split("}")[-1].replace("\\N", " ")
    assert body == "EVERYBODY REMEMBERS THE MOMENT THE INTERNET STOPPED…"
    assert (_tag(hook, "fscx") or 100) >= 80
    box = _grey(tmp_path, ass)
    assert box is not None and _inside_q4(box), box


def test_worst_case_wide_caption_group_inside_q4(tmp_path):
    ass = build_ass([w(0, 1, "WWWWWW"), w(1, 2, "MMMMM"), w(2, 3, "WWWWW")], "", 4)
    cap = next(e for e in _events(ass) if ",Cap," in e)
    assert "WWWWWW MMMMM WWWWW" in cap.replace("{\\c&H00FFFF&}", "").replace("{\\c&HFFFFFF&}", "")
    assert (_tag(cap, "fscx") or 100) >= 80
    box = _grey(tmp_path, ass)
    assert box is not None and _inside_q4(box), box


def test_over_long_caption_word_keeps_min_size(tmp_path):
    from hookline.quality import text_bbox

    ass = build_ass([w(0, 1, "SUPERCALIFRAGILISTICEXPIALIDOCIOUS")], "", 2)
    cap = next(e for e in _events(ass) if ",Cap," in e)
    assert _tag(cap, "fscx") >= 80 and _tag(cap, "fs") is not None
    box = text_bbox(_render(tmp_path, ass, 0.5), "yellow")
    assert box is not None and box[3] - box[1] + 1 >= 55  # Q2 holds even when too wide


def test_stacked_hook_and_captions_do_not_overlap(tmp_path):
    stacked = [_lay("STACKED"), _lay("FIT_BLUR")]
    cy, hy = caption_y_for(stacked), hook_y_for(stacked)
    hook_text = "Parks. Open space. All under siege."
    cap = _grey(tmp_path, build_ass([w(0, 2, "INCREDIBLE"), w(2, 3, "TEST")], "", 3, cy, hy))
    hook = _grey(tmp_path, build_ass([], hook_text, 3, cy, hy))
    assert cap is not None and hook is not None
    assert cap[3] < hook[1] or hook[3] < cap[1], (cap, hook)
    assert _inside_q4(cap) and _inside_q4(hook)


def test_render_failure_reports_ffmpeg_stderr(tmp_path):
    from hookline.quality import render_ass_frame

    with pytest.raises(RuntimeError, match=r"m\.ass failed: \S"):  # ffmpeg's stderr included
        render_ass_frame(tmp_path / "missing" / "m.ass", 0.5)


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


def test_early_until_covers_hook_tag_and_groups_shown_with_them():
    from hookline.captions.ass import early_until

    assert early_until([], "Hook", 20) == 3.0
    assert early_until([], "Hook", 20, seam=5.0) == pytest.approx(6.2)  # 1.2 s EARLIER tag
    # a group starting under the hook (2.8 s) holds until the next one appears (3.6 s less
    # 2 frames, gap <= 1 s)
    ws = [w(2.6, 2.8, "SO."), w(2.8, 3.4, "LONG."), w(3.6, 4.0, "NEXT.")]
    assert early_until(ws, "Hook", 20) == pytest.approx(3.6 - 2 / 30)


# --- M2.5 wave B: caption pages break at cuts and the seam; face rects follow the zoom plan --


def _secs(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _cap_spans(ass):
    out = []
    for e in _events(ass):
        f = e.split(",", 9)
        if f[3] == "Cap":
            out.append((_secs(f[1]), _secs(f[2]), f[9]))
    return out


def test_no_caption_event_spans_a_cut_or_the_seam():
    # clip 05: "CAPITALISM THE" was one page across the cold-open seam at 3.67-4.49 s
    ws = [
        w(3.2, 3.66, "capitalism"),
        w(3.67, 3.9, "the"),
        w(3.9, 4.49, "market"),
        w(6.0, 6.3, "and"),
        w(6.3, 6.9, "then"),
        w(7.0, 7.4, "we"),
        w(7.45, 7.9, "won"),
    ]
    cuts, seam = [3.667, 6.95], 3.667
    ass = build_ass(ws, "Hook", 10, seam=seam, breaks=cuts)
    spans = _cap_spans(ass)
    for b in {*cuts, seam}:
        for start, end, text in spans:
            assert not (start < b - 0.01 and end > b + 0.01), (b, start, end, text)
    pages = [text for _, _, text in spans]
    assert not any("CAPITALISM" in p and "THE" in p for p in pages)
    assert not any("THEN" in p and "WE" in p for p in pages)


def test_group_words_breaks_pages_at_break_times():
    ws = [w(0, 0.3, "one"), w(0.3, 0.6, "two"), w(0.6, 0.9, "three")]
    assert [len(g) for g in group_words(ws)] == [3]
    assert [len(g) for g in group_words(ws, breaks=[0.6])] == [2, 1]
    assert [len(g) for g in group_words(ws, breaks=[0.595])] == [2, 1]  # cut a frame early


def test_held_page_ends_at_the_next_cut():
    spans = _cap_spans(build_ass([w(1.0, 1.5, "hold"), w(2.2, 2.5, "next")], "", 5, breaks=[1.8]))
    assert spans[0][1] == pytest.approx(1.8)  # would hold to 2.2 without the cut


def test_face_rects_follow_the_planned_zoom():
    lay = SceneLayout(start=0, end=4, kind="FACE_TRACK", face_box=(0.3, 0.2))
    ((_, _, _, y1),) = face_rects(lay, 1.2)  # unplanned: about the face centre
    assert y1 == pytest.approx(0.4 * 1920 + 1.2 * 0.1 * 1920)  # chin, 1.2x about the centre
    planned = lay.model_copy(update={"base_zoom": 1.1, "zoom_cap": 1.3, "zoom_anchor": 0.1})
    ((_, p0, _, p1),) = face_rects(planned, 1.0)
    a = 0.1 * 1920
    assert p1 == pytest.approx(a + 1.1 * (0.5 * 1920 - a))  # chin mapped about the anchor
    assert p1 - p0 == pytest.approx(1.1 * 0.2 * 1920 * 0.8)  # core = bottom 80%, base zoom


def test_word_running_past_a_cut_opens_the_next_page():
    ws = [w(1.0, 1.4, "before"), w(1.5, 2.3, "straddles"), w(2.4, 2.8, "after")]
    spans = _cap_spans(build_ass(ws, "", 5, breaks=[2.0]))
    page = min(sp for sp in spans if "STRADDLES" in sp[2])
    assert page[0] == pytest.approx(2.0) and page[1] == pytest.approx(2.4)  # still captioned
    assert all(not (st < 1.99 and en > 2.01) for st, en, _ in spans)
    assert not any("BEFORE" in t and "STRADDLES" in t for _, _, t in spans)
    # a short overrun (<= 0.15 s) stays on its page, clipped at the cut
    short = _cap_spans(build_ass([w(1.5, 2.1, "brief"), w(2.4, 2.8, "next")], "", 5, breaks=[2.0]))
    assert short[0][0] == pytest.approx(1.5 - 2 / 30, abs=0.006)
    assert short[0][1] == pytest.approx(2.0)


# --- wave D: pages lead speech by 2 frames; larger captions ------------------------------


def test_pages_appear_two_frames_before_their_first_word():
    ws = [w(1.0, 1.4, "one"), w(1.4, 1.8, "two."), w(3.0, 3.4, "three"), w(3.4, 3.9, "four.")]
    spans = _cap_spans(build_ass(ws, "", 6))
    assert spans[0][0] == pytest.approx(0.93, abs=0.006)  # 1.0 - 2/30, centisecond grid
    page2 = next(sp for sp in spans if "THREE" in sp[2])
    assert page2[0] == pytest.approx(2.93, abs=0.006)
    # the first word lights up with its page; the next word still at its own start
    assert "{\\c&H00FFFF&}ONE" in spans[0][2]
    assert next(sp for sp in spans if "{\\c&H00FFFF&}TWO" in sp[2])[0] == pytest.approx(1.4)
    for (_, e0, _), (s1, _, _) in pairwise(spans):
        assert e0 <= s1 + 1e-9  # never two captions at once


def test_page_lead_never_crosses_a_cut_the_start_or_the_previous_word():
    at_cut = _cap_spans(build_ass([w(0.5, 0.9, "a."), w(2.03, 2.4, "b")], "", 4, breaks=[2.0]))
    assert at_cut[-1][0] == pytest.approx(2.0)  # not before the cut
    first = _cap_spans(build_ass([w(0.03, 0.4, "go")], "", 2))
    assert first[0][0] == 0.0
    tight = _cap_spans(build_ass([w(0.0, 0.5, "end."), w(0.52, 0.9, "next")], "", 2))
    assert tight[0][1] > tight[0][0] and tight[1][0] >= 0.0 + 0.05  # previous word keeps a beat


def test_caption_font_is_larger_than_the_hook_and_stays_inside_q4(tmp_path):
    from hookline.captions.ass import CAP_FONT_SIZE, FONT_SIZE

    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5)
    assert f"Style: Cap,Anton,{CAP_FONT_SIZE}," in ass and f"Style: Hook,Anton,{FONT_SIZE}," in ass
    assert 1.10 * 136 <= CAP_FONT_SIZE <= 1.15 * 136
    for words in (
        [w(0, 1, "WWWWWW"), w(1, 2, "MMMMM"), w(2, 3, "WWWWW")],
        [w(0, 1, "INCREDIBLE"), w(1, 2, "TEST")],
        [w(0, 1, "abcdefghij"), w(1, 2, "klmnopq")],
    ):
        for cy in (1180, 689, 553):
            text = build_ass(words, "", 4, caption_y=cy, caption_y_late=cy)
            box = _grey(tmp_path, text)
            assert box is not None and _inside_q4(box), (cy, box)
            from hookline.quality import line_heights

            assert len(line_heights(_render(tmp_path, text, 0.5), dense=0.4)) == 1  # one line


def test_hook_line_choice_uses_the_page_start():
    # a page whose word starts 0.03 s after the hook ends appears 2 frames earlier, while the
    # hook is still up: it must use the early (hook-clear) line
    ass = build_ass([w(3.03, 3.4, "after")], "Hook", 6, caption_y=900, caption_y_late=600)
    (cap,) = [e for e in _events(ass) if ",Cap," in e]
    assert "\\pos(474,900)" in cap


def test_face_rects_use_the_measured_face_offset_and_zoom():
    lay = SceneLayout(
        start=0, end=4, kind="FACE_TRACK", face_box=(0.3, 0.2), face_dx=0.1, lead=-0.08
    )
    ((x0, _, x1, _),) = face_rects(lay, 1.0)
    assert (x0 + x1) / 2 == pytest.approx(540 + 0.1 * 1080)
    zoomed = lay.model_copy(update={"zoom_cap": 1.12})
    ((z0, _, z1, _),) = face_rects(zoomed, 1.2)
    assert (z0 + z1) / 2 == pytest.approx(540 + 0.1 * 1080 * 1.12)


# --- fix wave F: captions clear the face at every sample of their page --------------------


def _page_y(text: str) -> list[tuple[float, float, int]]:
    out = []
    for line in text.splitlines():
        if line.startswith("Dialogue:") and ",Cap," in line:
            parts = line.split(",", 9)
            y = int(re.search(r"\\pos\(\d+,(\d+)\)", parts[9]).group(1))
            out.append((_secs(parts[1]), _secs(parts[2]), y))
    return out


def _secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_build_ass_moves_only_the_pages_shown_over_a_head_dip():
    from hookline.captions.ass import CAPTION_ABOVE, CAPTION_BELOW

    words = [Word(start=0.2 + 0.4 * i, end=0.5 + 0.4 * i, text=f"w{i}.") for i in range(12)]
    faces = []
    for k in range(24):
        t = k * 0.25 + 0.125
        low = 2.0 <= t <= 2.4  # the head dips: the core reaches the 1180 caption band
        y0, y1 = (1090, 1480) if low else (600, 980)
        faces.append((t, 300.0, float(y0), 780.0, float(y1), y0 + 70.0))
    text = build_ass(words, "", 6.0, faces=faces)
    pages = _page_y(text)
    moved = [(a, b, y) for a, b, y in pages if y != 1180]
    assert moved and all(b > 1.9 and a < 2.5 for a, b, _ in moved)
    for a, b, y in pages:
        for t, _, y0, _, y1, eye in faces:
            if a - 0.125 <= t <= b + 0.125:
                top, bottom = y - CAPTION_ABOVE, y + CAPTION_BELOW
                assert bottom <= min(y0, eye) or top >= y1, (a, b, y, t)


def test_caption_y_for_counts_the_per_sample_faces():
    lay = SceneLayout(
        start=0, end=6, kind="FACE_TRACK", face_box=(0.3, 0.2), track_x=[(0, 0.5)],
        face_px=[[t / 4, 300, 1000, 780, 1400, 1080] for t in range(24)],
    )  # fmt: skip
    # the median box (576-960 px) leaves 1180 free, the per-sample faces (1000-1400) do not
    assert caption_y_for([lay], late=True) != 1180
