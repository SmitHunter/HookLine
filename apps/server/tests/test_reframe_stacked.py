"""Rendering the fixed STACKED template (cam panel, gameplay panel, patches,
pillarbox fit, full-cam beats) and its head geometry."""

import re
import subprocess

import pytest

from hookline.media.ffmpeg import probe_duration, probe_size
from hookline.media.reframe import (
    BEAT_CHIN_Y,
    OUT_H,
    OUT_W,
    ZOOM_IN,
    detect_crop,
    full_cam_geometry,
    plan_zoom,
    reframe_graph,
    scene_filter,
    scene_upscale,
    stacked_face_px,
)
from hookline.pipeline.layout import STACK_CAM_H, SceneLayout, stack_template
from hookline.pipeline.profile import LayoutProfile

SIZE = (1280, 720)
PROFILE = LayoutProfile(
    size=SIZE,
    cam=(46, 362, 247, 562),
    outer=(41, 360, 250, 568),
    head=(105, 390, 208, 521),
    face=(122, 433, 191, 521),
    masks=[(41, 568, 250, 597)],
)


def _template(start=0.0, end=4.0, **kw) -> SceneLayout:
    t = stack_template(PROFILE, None, SIZE, fit=kw.pop("fit", None))
    return SceneLayout(start=start, end=end, kind="STACKED", **{**t, **kw})


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    dst = tmp_path_factory.mktemp("src") / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=1280x720:rate=25:duration=4", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )  # fmt: skip
    return dst


def _render(src, graph, dst):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-filter_complex", graph, "-map", "[vout]",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )  # fmt: skip
    return dst


def test_template_panels_crop_the_cam_window_and_the_centred_gameplay():
    lay = _template()
    f = scene_filter(lay, 1280, 720, "a", "b")
    cw, ch, cx, cy = (int(v) for v in re.search(r"_a\]crop=(\d+):(\d+):(\d+):(\d+)", f).groups())
    assert (cx, cw) == (46, 200) and 362 <= cy and cy + ch <= 562
    gw, gh, gx, gy = (int(v) for v in re.search(r"_b\]crop=(\d+):(\d+):(\d+):(\d+)", f).groups())
    assert gh == 720 and gy == 0 and gx + gw / 2 == pytest.approx(640, abs=2)
    assert f"scale={OUT_W}:{OUT_H - STACK_CAM_H}" in f
    assert "gblur" not in f and "enable=" not in f


def test_template_patches_fit_and_beats_render(src, tmp_path):
    fit = (150 / 1280, 0.0, 980 / 1280, 1.0)
    patch = (150 / 1280, 360 / 720, 100 / 1280, 240 / 720)  # an overlay it could not trim
    lays = [
        _template(0.0, 1.5, fit=fit).model_copy(update={"patches": [patch]}),
        _template(1.5, 4.0, cam_beats=[(2.0, 3.5)]),
    ]
    graph = reframe_graph(lays, 1280, 720, zoom_steps=[(0.0, 1.0), (2.5, ZOOM_IN)])
    assert graph.count("gblur") == 1
    assert "between(t\\,0.5\\,2)" in graph  # beat times relative to its scene
    out = _render(src, graph, tmp_path / "t.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def _beat(cam, head):
    """A template layout of a cam (x0, y0, x1, y1 px of 1280x720) with that head box."""
    prof = PROFILE.model_copy(
        update={"cam": cam, "outer": (cam[0] - 2, cam[1] - 2, cam[2] + 2, cam[3] + 2),
                "head": head, "face": (head[0] + 10, head[1] + 20, head[2] - 10, head[3]),
                "masks": []}
    )  # fmt: skip
    t = stack_template(prof, None, SIZE)
    return SceneLayout(start=0.0, end=4.0, kind="STACKED", **t)


@pytest.mark.parametrize(
    ("cam", "head"),
    [
        ((46, 362, 247, 562), (105, 390, 208, 521)),  # the audited square cam
        ((900, 20, 1260, 222), (1040, 40, 1130, 200)),  # a 16:9 cam
        ((1000, 100, 1200, 400), (1050, 180, 1150, 330)),  # a portrait cam
        ((60, 400, 300, 535), (130, 402, 230, 534)),  # a head taller than a 0.8 window
    ],
)
def test_full_cam_geometry_large_head_centred_window(cam, head):
    """A beat shows a head-centred window of the cam at least 1080x1350 (a wide cam is
    cropped round the head, never shrunk into letterbox bands), the whole head in it, the
    chin at most BEAT_CHIN_Y, the picture inside the frame."""
    lay = _beat(cam, head)
    (ww, wh, wx, wy), (pw, ph), top = full_cam_geometry(lay, 1280, 720)
    assert cam[0] <= wx and wx + ww <= cam[2] + 1 and cam[1] <= wy and wy + wh <= cam[3] + 1
    assert wx <= head[0] + 1 and head[2] <= wx + ww + 1  # the whole head width ...
    left = (OUT_W - pw) / 2
    assert left + (head[0] - wx) / ww * pw >= 0 and left + (head[2] - wx) / ww * pw <= OUT_W
    assert wy <= head[1] + 1 and head[3] <= wy + wh + 1  # top of the hair to the chin
    wide = (cam[2] - cam[0]) / (cam[3] - cam[1]) > 0.8
    if wide and (head[2] - head[0]) * 1.1 <= 0.8 * (cam[3] - cam[1]):
        assert ww / wh == pytest.approx(0.8, abs=0.02)  # 1080x1350
    if (head[3] - head[1]) / wh * 1350 <= BEAT_CHIN_Y - 40:  # else the head fills it: smaller
        assert pw >= OUT_W or ph == OUT_H  # 1080 wide (sides past the frame when widened)
        assert ph >= 1350 or ph == OUT_H
    chin = top + (head[3] - wy) / wh * ph
    assert chin <= BEAT_CHIN_Y + 2
    assert top + (head[1] - wy) / wh * ph >= 0  # the head top on screen


def test_beat_render_vignette_and_edge_fill(src, tmp_path):
    lay = _template(0.0, 4.0, cam_beats=[(1.0, 3.5)])
    f = scene_filter(lay, 1280, 720, "a", "b")
    assert "vignette" in f
    out = _render(src, reframe_graph([lay], 1280, 720), tmp_path / "beat.mp4")
    assert probe_size(out) == (1080, 1920)


def test_stacked_face_px_head_in_the_cam_panel_and_in_beats():
    lay = plan_zoom(_template(0.0, 4.0, cam_beats=[(2.0, 3.0)]), 1280, 720)
    rows = stacked_face_px(lay, 1280, 720, [(0.0, 1.0), (1.0, ZOOM_IN)])
    assert rows and rows[0][0] == pytest.approx(0.0, abs=0.13)
    stack = [r for r in rows if r[0] < 2.0]
    beat = [r for r in rows if 2.0 <= r[0] < 3.0]
    for _, x0, top, x1, bottom, eye in stack:
        assert eye is None and 0 <= top < bottom <= STACK_CAM_H  # head inside the cam panel
        assert 0 <= x0 < x1 <= OUT_W
    assert beat
    for _, x0, top, x1, bottom, _ in beat:  # the big full-cam head, the picture at the top
        assert bottom <= BEAT_CHIN_Y + 2 and 0 <= top < STACK_CAM_H / 2
        assert x1 - x0 > 0.5 * OUT_W  # a large head


def test_plan_zoom_template_punches_the_gameplay_when_the_cam_cannot():
    """The audited cam (its head fills the panel, a 5x upscale) has no headroom for a visible
    zoom (1.02-1.035 before): the steps zoom the gameplay panel instead, >= 1.08, about its
    centre, recorded as such; the cam panel and its head stay put."""
    from hookline.media.reframe import STACK_PUNCH_MIN, rendered_zoom_steps

    lay = plan_zoom(_template(), 1280, 720)
    assert lay.zoom_panel == "game" and lay.zoom_anchor is None
    assert lay.zoom_level(ZOOM_IN) >= STACK_PUNCH_MIN >= 1.08
    assert lay.zoom_level(1.2) >= lay.zoom_level(ZOOM_IN)
    steps = [(0.0, 1.0), (1.0, 1.2), (2.0, ZOOM_IN)]
    assert [z for _, z in rendered_zoom_steps([lay], steps)][1:] == [
        pytest.approx(lay.zoom_level(1.2)), pytest.approx(lay.zoom_level(ZOOM_IN))
    ]  # fmt: skip
    still = stacked_face_px(lay, 1280, 720, [(0.0, 1.0)])
    zoomed = stacked_face_px(lay, 1280, 720, [(0.0, 1.2)])
    assert still == zoomed  # the cam panel does not zoom
    f = scene_filter(lay, 1280, 720, "a", "b", zoom_steps=steps)
    cam_chain = f.split("[b_cam]")[0].split(";")[-1]
    game_chain = f.split("[b_game]")[0].split(";")[-1]
    assert "eval=frame" not in cam_chain and "eval=frame" in game_chain


def test_plan_zoom_template_keeps_cam_zoom_with_headroom():
    """A big cam with a small head has room to punch the cam panel itself (>= 1.08)."""
    prof = PROFILE.model_copy(
        update={"cam": (20, 300, 420, 600), "outer": (18, 298, 422, 602),
                "head": (190, 380, 250, 450), "face": (200, 400, 240, 450), "masks": []}
    )  # fmt: skip
    t = stack_template(prof, None, SIZE)
    lay = plan_zoom(SceneLayout(start=0.0, end=4.0, kind="STACKED", **t), 1280, 720)
    assert lay.zoom_panel == "speaker" and lay.zoom_level(1.2) >= 1.08
    rows = stacked_face_px(lay, 1280, 720, [(0.0, 1.2)])
    assert all(r[2] >= 0 and r[4] <= STACK_CAM_H for r in rows)


def test_scene_upscale_template_uses_the_cam_window():
    assert scene_upscale(_template(), 1280, 720) == pytest.approx(1080 / 200, rel=0.01)


def test_detect_crop_ignores_dark_content_next_to_a_bar(tmp_path):
    """A dark stretch of the game is not a bar: only lines near pure black in every sample
    of the video are (the union of the samples' content)."""
    dst = tmp_path / "dark.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=1280x720:rate=25:duration=5",
         "-vf", ("drawbox=x=0:y=0:w=200:h=720:color=black:t=fill:enable='lt(t,4)',"
                "drawbox=x=1240:y=0:w=40:h=720:color=black:t=fill"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )  # fmt: skip
    assert detect_crop(dst) == (1240, 720, 0, 0)
