import subprocess
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
        and "{\\c&H00FFFF&}NO{\\r} WAY" in events[1]
        and "NO {\\c&H00FFFF&}WAY" in events[2]
    )


def test_ass_header_and_styles():
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=640)
    for s in ("PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes"):
        assert s in ass
    assert "Style: Cap,Anton,96" in ass and "Style: Hook,Anton,84" in ass
    assert "{\\pos(540,640)}" in ass and "{\\pos(540,260)}" in ass
    hook = next(e for e in _events(ass) if ",Hook," in e)
    assert hook.startswith("Dialogue: 1,0:00:00.00,0:00:03.00,")


def test_hook_clamped_to_duration_and_wrapped():
    ass = build_ass(
        [w(0, 0.3, "no")], hook_text="this is a very long hook title here", duration=1.5
    )
    hook = next(e for e in _events(ass) if ",Hook," in e)
    assert ",0:00:01.50," in hook
    assert "\\N" in hook and hook.count("\\N") <= 2
    assert "THIS IS A VERY" in hook


def test_ass_escapes_special_chars():
    ass = build_ass([w(0, 0.5, "{hi}\\")], hook_text="x", duration=1)
    assert "\\{HI\\}\\\\" in ass


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
    assert caption_y_for([st, ft]) == 640
    assert caption_y_for([ft, st]) == 1300
    assert caption_y_for([]) == 1300


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
    assert (hook_y_for([]), caption_y_for([])) == (260, 1300)
    assert (hook_y_for([_lay("FIT_BLUR")]), caption_y_for([_lay("FIT_BLUR")])) == (260, 1300)
    stacked = [_lay("STACKED"), _lay("FIT_BLUR")]
    assert (hook_y_for(stacked), caption_y_for(stacked)) == (860, 640)  # below the seam
    assert hook_y_for([_lay("FACE_TRACK")]) == 260  # no face box: default


def test_face_track_hook_moves_off_the_face():
    low = [_lay("FACE_TRACK", (0.3, 0.3))]  # face 576-1152 px: clear of the hook band
    assert (hook_y_for(low), caption_y_for(low)) == (260, 1300)
    high = [_lay("FACE_TRACK", (0.1, 0.3))]  # face top 192 px: under the hook band
    assert (hook_y_for(high), caption_y_for(high)) == (1040, 1300)


def test_face_track_captions_move_below_a_face_covering_them():
    big = [_lay("FACE_TRACK", (0.4, 0.4))]  # face 768-1536 px covers the 1300 caption line
    assert caption_y_for(big) == 1600 and hook_y_for(big) == 260


def test_split_hook_uses_top_panel_face():
    assert hook_y_for([_lay("SPLIT", (0.1, 0.5))]) == 1040  # 96 px into the 960 px panel
    assert hook_y_for([_lay("SPLIT", (0.5, 0.3))]) == 260  # 480 px: below the band


def test_build_ass_hook_position():
    ass = build_ass([w(0, 0.3, "no")], hook_text="Hi", duration=5, caption_y=640, hook_y=860)
    assert "{\\pos(540,860)}" in ass and "{\\pos(540,640)}" in ass
