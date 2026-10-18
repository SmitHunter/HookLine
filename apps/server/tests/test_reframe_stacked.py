"""Wave G-layout: rendering the fixed STACKED template (cam panel, gameplay panel, patches,
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


def test_full_cam_geometry_keeps_the_chin_above_the_caption_line():
    lay = _template()
    fg_h, y = full_cam_geometry(lay, 1280, 720)
    assert fg_h == pytest.approx(1080 * 200 / 201, abs=8)  # even px crop
    chin = y + (521 - 362) / 200 * fg_h
    assert chin <= BEAT_CHIN_Y + 1 and y >= 0


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
    for _, _, top, _, bottom, _ in beat:  # the big full-cam head, its chin on BEAT_CHIN_Y
        assert bottom == pytest.approx(BEAT_CHIN_Y, abs=2) and top > STACK_CAM_H / 2


def test_plan_zoom_template_keeps_the_head_in_the_cam_panel():
    lay = plan_zoom(_template(), 1280, 720)
    assert lay.zoom_anchor is not None
    top = lay.zoom_level(1.2)
    assert top <= ZOOM_IN + 1e-6  # a soft (5x upscaled) cam: zooms stay small
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
