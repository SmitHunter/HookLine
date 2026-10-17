import math
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from hookline.media.ffmpeg import probe_duration, probe_size
from hookline.media.reframe import (
    OUT_H,
    OUT_W,
    ZOOM_IN,
    ZOOM_PUNCH,
    box_hits_bands,
    box_to_crop,
    combine_crop,
    detect_crop,
    detect_overlay_bands,
    plan_zoom,
    reframe_graph,
    scene_filter,
    scene_upscale,
    track_x_expr,
    zoom_expr,
    zoom_schedule,
)
from hookline.pipeline.layout import SceneLayout


def _make(dst, vf=None, duration=5):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc2=size=1280x720:rate=25:duration={duration}"]  # fmt: skip
    if vf:
        cmd += ["-vf", vf]
    subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def _render(src, graph, dst):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-filter_complex", graph, "-map", "[vout]",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )  # fmt: skip
    return dst


def evaluate(expr: str, t: float) -> float:
    """Evaluate an ffmpeg if/lt/arithmetic expression at time t."""

    def _if(cond, a, b):
        return a if cond else b

    def lt(a, b):
        return 1 if a < b else 0

    py = expr.replace("if(", "_if(")
    env = {"_if": _if, "lt": lt, "t": t, "min": min, "max": max}
    return float(eval(py, {"__builtins__": {}}, env))


def test_detect_crop_finds_letterbox(tmp_path):
    crop = detect_crop(_make(tmp_path / "boxed.mp4", "pad=1280:920:0:100"))
    assert crop is not None
    for got, want in zip(crop, (1280, 720, 0, 100), strict=True):
        assert got == pytest.approx(want, abs=4)


def test_detect_crop_none_without_bars(tmp_path):
    assert detect_crop(_make(tmp_path / "plain.mp4")) is None


def test_output_size_constant():
    assert (OUT_W, OUT_H) == (1080, 1920)


def test_track_expression_clamped_and_deadbanded():
    expr = track_x_expr([(0, 0.5), (1, 0.52), (2, 0.9)], in_w=1920, crop_w=607)
    assert "if(lt(t," in expr and evaluate(expr, t=0.5) == pytest.approx(evaluate(expr, t=0))
    assert 0 <= evaluate(expr, t=2) <= 1920 - 607


def test_track_expression_clamps_edges_and_interpolates():
    expr = track_x_expr([(0, 0.0), (1, 1.0), (2, 1.0), (3, 1.0)], in_w=1920, crop_w=607)
    assert evaluate(expr, t=0) == pytest.approx(0)
    for t in (0, 0.5, 1, 1.5, 2, 2.5, 3, 10):
        assert 0 <= evaluate(expr, t=t) <= 1920 - 607
    assert evaluate(expr, t=0) < evaluate(expr, t=0.5) < evaluate(expr, t=1)


def _centre(expr: str, t: float, in_w: int = 1000, crop_w: int = 100) -> float:
    """Normalized centre x back from an evaluated left edge."""
    return (evaluate(expr, t=t) + crop_w / 2) / in_w


def test_track_ema_converges_to_steady_target():
    kfs = [(0, 0.5)] + [(i, 0.9) for i in range(1, 16)]
    expr = track_x_expr(kfs, in_w=1000, crop_w=100)
    assert _centre(expr, t=15) == pytest.approx(0.9, abs=0.01)


def test_track_jitter_inside_deadband_holds_still():
    kfs = [(i, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(1, 12)]
    expr = track_x_expr([(0, 0.5), *kfs], in_w=1000, crop_w=100)
    for t in (0, 0.5, 3, 7.5, 11):
        assert _centre(expr, t=t) == pytest.approx(0.5)


def test_track_tiny_time_delta_has_no_division_by_zero():
    expr = track_x_expr([(0, 0.2), (0.0003, 0.8), (1, 0.8)], in_w=1000, crop_w=100)
    assert "/0)" not in expr and "/0," not in expr and not expr.endswith("/0")
    for t in (0, 0.0001, 0.0003, 0.5, 1):
        v = evaluate(expr, t=t)
        assert math.isfinite(v) and 0 <= v <= 900


def test_reframe_graph_rejects_empty_and_all_tiny():
    with pytest.raises(ValueError, match="no layouts"):
        reframe_graph([], 1280, 720)
    with pytest.raises(ValueError, match="no layouts"):
        reframe_graph([SceneLayout(start=1, end=1.01, kind="FIT_BLUR")], 1280, 720)


_TINY_MIX = [
    SceneLayout(start=0, end=0.02, kind="STACKED", facecam=(0.75, 0.0, 0.25, 0.3)),
    SceneLayout(start=0.02, end=2, kind="FIT_BLUR"),
    SceneLayout(start=2, end=2.02, kind="SPLIT", split_x=(0.25, 0.75)),
    SceneLayout(start=2.02, end=4, kind="FACE_TRACK", track_x=[(2.02, 0.3), (3, 0.7), (4, 0.5)]),
]


def _trim_spans(graph: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in re.findall(r"trim=start=([\d.]+):end=([\d.]+)", graph)]


def test_reframe_graph_merges_tiny_scenes_without_gaps():
    graph = reframe_graph(_TINY_MIX, 1280, 720)
    spans = _trim_spans(graph)
    assert spans == [(0, 2.02), (2.02, 4)]  # tiny lead pulled into next, tiny middle into prev
    assert sum(b - a for a, b in spans) == pytest.approx(4.0)
    assert "concat=n=2:" in graph


def test_render_merged_tiny_scenes_keeps_duration(src, tmp_path):
    out = _render(src, reframe_graph(_TINY_MIX, 1280, 720), tmp_path / "merged.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def test_track_expression_empty_is_centre():
    assert evaluate(track_x_expr([], in_w=1280, crop_w=404), t=1) == pytest.approx((1280 - 404) / 2)


def test_scene_filter_labels():
    chain = scene_filter(SceneLayout(start=0, end=1, kind="FIT_BLUR"), 1280, 720, "a", "b")
    assert chain.startswith("[a]") and chain.endswith("[b]")


_LAYOUTS = {
    "STACKED": SceneLayout(start=0, end=4, kind="STACKED", facecam=(0.75, 0.0, 0.25, 0.3)),
    "FACE_TRACK": SceneLayout(
        start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.2), (1.5, 0.6), (3, 0.9), (3.5, 0.3)]
    ),
    "SPLIT": SceneLayout(start=0, end=4, kind="SPLIT", split_x=(0.25, 0.75)),
    "FIT_BLUR": SceneLayout(start=0, end=4, kind="FIT_BLUR"),
}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("src") / "src.mp4", duration=4)


@pytest.mark.parametrize("kind", list(_LAYOUTS))
def test_render_each_kind(src, tmp_path, kind):
    out = _render(src, reframe_graph([_LAYOUTS[kind]], 1280, 720), tmp_path / "out.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def test_render_two_scenes(src, tmp_path):
    layouts = [
        SceneLayout(start=0, end=2, kind="STACKED", facecam=(0.0, 0.7, 0.3, 0.3)),
        SceneLayout(start=2, end=4, kind="FIT_BLUR"),
    ]
    out = _render(src, reframe_graph(layouts, 1280, 720), tmp_path / "two.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def test_render_track_offset_scene_and_fallbacks(src, tmp_path):
    layouts = [
        SceneLayout(start=0, end=1, kind="SPLIT"),  # split_x None -> FIT_BLUR
        SceneLayout(start=1, end=2, kind="STACKED"),  # facecam None -> FIT_BLUR
        SceneLayout(start=2, end=4, kind="FACE_TRACK", track_x=[(2, 0.1), (3, 0.5), (4, 0.9)]),
    ]
    out = _render(src, reframe_graph(layouts, 1280, 720), tmp_path / "mix.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def test_render_with_crop(tmp_path):
    boxed = _make(tmp_path / "boxed.mp4", "pad=1280:920:0:100", duration=4)
    graph = reframe_graph([_LAYOUTS["SPLIT"]], 1280, 720, crop=(1280, 720, 0, 100))
    assert graph.startswith("[0:v]crop=1280:720:0:100")
    out = _render(boxed, graph, tmp_path / "out.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def test_reframe_graph_all_tiny_but_long_enough_becomes_one_scene():
    tiny = [SceneLayout(start=i * 0.02, end=(i + 1) * 0.02, kind="FIT_BLUR") for i in range(5)]
    assert _trim_spans(reframe_graph(tiny, 1280, 720)) == [(0, 0.1)]


# --- pacing zooms -------------------------------------------------------------------------


def _gaps(steps: list[tuple[float, float]], duration: float) -> list[float]:
    times = [t for t, _ in steps] + [duration]
    return [b - a for a, b in pairwise(times)]


def test_zoom_schedule_long_take_no_gap_over_max_and_alternates():
    steps = zoom_schedule(30.0, [], [])
    assert steps[0] == (0.0, 1.0)
    assert max(_gaps(steps, 30.0)) <= 4.5 + 1e-9
    assert all(a[1] != b[1] for a, b in pairwise(steps))
    assert {s for _, s in steps} == {1.0, ZOOM_IN}
    assert all(b[0] - a[0] >= 0.5 for a, b in pairwise(steps))


def test_zoom_schedule_steps_at_cuts():
    steps = zoom_schedule(12.0, [2.0, 7.5], [])
    times = [t for t, _ in steps]
    assert 2.0 in times and 7.5 in times
    assert max(_gaps(steps, 12.0)) <= 4.5 + 1e-9
    assert all(a[1] != b[1] for a, b in pairwise(steps))


def test_zoom_schedule_emphasis_punch_held_then_released():
    steps = zoom_schedule(20.0, [], [6.0])
    assert (6.0, ZOOM_PUNCH) in steps
    i = steps.index((6.0, ZOOM_PUNCH))
    t_back, s_back = steps[i + 1]
    assert t_back == pytest.approx(6.0 + 0.2 + 1.2) and s_back in (1.0, ZOOM_IN)
    assert max(s for _, s in steps) <= 1.2
    assert max(_gaps(steps, 20.0)) <= 4.5 + 1e-9


def test_zoom_schedule_drops_out_of_range_and_overlapping_punches():
    steps = zoom_schedule(10.0, [], [-1.0, 3.0, 3.5, 12.0])
    assert [t for t, s in steps if s == ZOOM_PUNCH] == [3.0]


def test_zoom_schedule_early_punch_dropped_and_nothing_at_or_after_end():
    assert [s for _, s in zoom_schedule(10.0, [], [0.3])].count(ZOOM_PUNCH) == 0
    steps = zoom_schedule(10.0, [], [9.0])  # release would land at 10.4
    assert (9.0, ZOOM_PUNCH) in steps and all(t < 10.0 for t, _ in steps)
    assert all(t < 9.0 for t, _ in zoom_schedule(9.0, [], []))


def test_zoom_schedule_prefers_sentence_boundaries():
    steps = zoom_schedule(9.0, [], [], boundaries=[3.9])
    assert [t for t, _ in steps][:2] == [0.0, 3.9]
    assert max(_gaps(steps, 9.0)) <= 4.5
    far = zoom_schedule(9.0, [], [], boundaries=[3.4])  # > 1 s from nominal 4.5: ignored
    assert [t for t, _ in far] == [0.0, 4.5]


def test_zoom_expr_eases_between_steps():
    steps = [(0.0, 1.0), (2.0, ZOOM_IN), (5.0, ZOOM_PUNCH), (6.4, 1.0)]
    expr = zoom_expr(steps)
    assert evaluate(expr, 1.0) == pytest.approx(1.0)
    assert evaluate(expr, 2.1) == pytest.approx(1.06)  # halfway through the 0.2 s ease
    assert evaluate(expr, 2.05) == pytest.approx(1 + 0.12 * 0.15625)  # smoothstep
    assert evaluate(expr, 3.0) == pytest.approx(ZOOM_IN)
    assert evaluate(expr, 5.5) == pytest.approx(ZOOM_PUNCH)
    assert evaluate(expr, 6.55) == pytest.approx(1.1)  # 0.3 s release from the punch
    assert evaluate(expr, 9.0) == pytest.approx(1.0)
    # scene-relative: the scene starting at 2.0 sees the step at its t=0
    assert evaluate(zoom_expr(steps, offset=2.0), 1.0) == pytest.approx(ZOOM_IN)


def test_scene_upscale_factors():
    assert scene_upscale(_LAYOUTS["FACE_TRACK"], 1280, 720) == pytest.approx(1920 / 720, rel=0.01)
    assert scene_upscale(_LAYOUTS["FIT_BLUR"], 1920, 1080) == pytest.approx(1080 / 1920)
    assert scene_upscale(_LAYOUTS["SPLIT"], 3840, 2160) < 1.3


_STEPS = [(0.0, 1.0), (1.5, ZOOM_IN), (3.0, 1.0)]


def test_scene_filter_zoom_and_sharpen_per_kind():
    track = scene_filter(_LAYOUTS["FACE_TRACK"], 1280, 720, "a", "b", zoom_steps=_STEPS)
    assert track.count("eval=frame") == 1 and "unsharp=5:5:0.6:5:5:0.0" in track
    assert track.index("unsharp") > track.index("eval=frame")  # sharpen final pixels
    stacked = scene_filter(_LAYOUTS["STACKED"], 1280, 720, "a", "b", zoom_steps=_STEPS)
    assert stacked.count("eval=frame") == 1  # facecam panel only
    assert stacked.index("eval=frame") < stacked.index("_cam]")
    split = scene_filter(_LAYOUTS["SPLIT"], 1280, 720, "a", "b", zoom_steps=_STEPS)
    assert split.count("eval=frame") == 2
    blur = scene_filter(_LAYOUTS["FIT_BLUR"], 1920, 1080, "a", "b", zoom_steps=_STEPS)
    assert blur.count("eval=frame") == 1 and "unsharp" not in blur
    assert blur.index("eval=frame") > blur.index("_fg]")  # foreground only
    plain = scene_filter(_LAYOUTS["FACE_TRACK"], 1280, 720, "a", "b")
    assert "eval=frame" not in plain


def test_scene_filter_flat_schedule_skips_zoom():
    flat = scene_filter(_LAYOUTS["FACE_TRACK"], 1280, 720, "a", "b", zoom_steps=[(0.0, 1.0)])
    assert "eval=frame" not in flat


@pytest.mark.parametrize("kind", list(_LAYOUTS))
def test_render_each_kind_with_zoom(src, tmp_path, kind):
    graph = reframe_graph([_LAYOUTS[kind]], 1280, 720, zoom_steps=_STEPS)
    out = _render(src, graph, tmp_path / "z.mp4")
    assert probe_size(out) == (1080, 1920)
    assert probe_duration(out) == pytest.approx(4.0, abs=0.1)


def _gray(path, t: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(OUT_H, OUT_W).astype(int)


_STATIC_SRC = (
    "color=c=yellow:s=1280x720:r=25:d=12,drawbox=x=640:y=0:w=640:h=720:c=blue:t=fill,"
    "drawbox=x=0:y=300:w=1280:h=60:c=red:t=fill"
)


def _top_edge(col: np.ndarray) -> int:
    return int(np.nonzero(np.abs(np.diff(col)) > 40)[0][0])


def test_rendered_face_track_zoom_changes_crop_on_static_source(tmp_path):
    static = tmp_path / "static.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         _STATIC_SRC,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(static)],
        check=True,
    )  # fmt: skip
    lay = SceneLayout(start=0, end=12, kind="FACE_TRACK", track_x=[(0, 0.5), (12, 0.5)])
    steps = zoom_schedule(12.0, [], [])
    assert evaluate(zoom_expr(steps), 1.0) == 1.0 and evaluate(zoom_expr(steps), 6.0) != 1.0
    out = _render(static, reframe_graph([lay], 1280, 720, zoom_steps=steps), tmp_path / "z.mp4")
    a, b = _gray(out, 1.0), _gray(out, 6.0)
    assert np.abs(a - b).mean() > 0.5  # the crop region differs
    # centre-anchored zoom-in pushes the band edge away from the centre
    assert _top_edge(b[:, 100]) < _top_edge(a[:, 100]) - 10


def _gray_frames(path, start: float, end: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf",
         f"trim=start={start}:end={end}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, OUT_H, OUT_W).astype(int)


def test_zoom_step_across_scene_boundary_has_no_jump(tmp_path):
    static = tmp_path / "static.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "color=c=yellow:s=1280x720:r=25:d=4,drawbox=x=0:y=100:w=1280:h=60:c=red:t=fill",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(static)],
        check=True,
    )  # fmt: skip
    lays = [
        SceneLayout(start=0, end=2, kind="FACE_TRACK", track_x=[(0, 0.5)]),
        SceneLayout(start=2, end=4, kind="FACE_TRACK", track_x=[(2, 0.5)]),
    ]
    steps = [(0.0, 1.0), (1.9, ZOOM_IN)]  # eases over 1.9-2.1, across the cut at 2.0
    out = _render(static, reframe_graph(lays, 1280, 720, zoom_steps=steps), tmp_path / "x.mp4")
    edges = [_top_edge(f[:, 100]) for f in _gray_frames(out, 1.8, 2.32)]
    moves = [b - a for a, b in pairwise(edges)]
    assert edges[0] - edges[-1] > 60  # the band edge moved outward with the zoom
    assert all(m <= 1 for m in moves)  # monotonic: scene 2 continues, never restarts
    assert max(-m for m in moves) <= 30  # no jump at the scene boundary


# --- burned-in overlay bands -------------------------------------------------------------

_FONT = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "Anton-Regular.ttf"
_REAL_SRC = Path(__file__).resolve().parents[1] / "workspace" / "e87398ed46" / "source.mp4"
_LOWER_THIRD = (
    "drawbox=x=0:y=612:w=1280:h=108:color=0x6a1fb0:t=fill,"
    "drawtext=fontfile=Anton-Regular.ttf:text='The History of Public Access - Episode 1':"
    "x=40:y=622:fontsize=40:fontcolor=white,"
    "drawtext=fontfile=Anton-Regular.ttf:text='Harrison Goldstein, Myles Jewell 10/21/2022':"
    "x=40:y=670:fontsize=34:fontcolor=white"
)


def _make_overlay(
    dst: Path, vf: str, duration: float = 12, still: bool = False, pan: bool = False
) -> Path:
    """testsrc2 1280x720 with `vf`; `still` freezes the first frame (a paused video), `pan`
    scrolls the picture sideways (a moving game camera). Runs in dst's folder so drawtext can
    name the font file without Windows path escaping."""
    shutil.copy(_FONT, dst.parent / _FONT.name)
    src = f"testsrc2=size=1280x720:rate=25:duration={duration}"
    if pan:
        src = f"testsrc2=size=2560x720:rate=25:duration={duration}"
        vf = f"crop=1280:720:x='t*100':y=0,{vf}"
    if still:
        vf = f"trim=end_frame=1,loop=loop=-1:size=1,setpts=N/25/TB,trim=duration={duration},{vf}"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src,
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", dst.name],
        check=True, cwd=dst.parent,
    )  # fmt: skip
    return dst


def test_overlay_bottom_lower_third_detected(tmp_path):
    bands = detect_overlay_bands(_make_overlay(tmp_path / "lt.mp4", _LOWER_THIRD))
    assert bands is not None
    top, bottom = bands
    assert top == 0
    assert bottom / 720 == pytest.approx(0.15, abs=0.02)
    assert bottom % 2 == 0


def test_overlay_moving_ticker_at_bottom_is_none(tmp_path):
    ticker = (
        "drawbox=x=0:y=612:w=1280:h=108:color=0x6a1fb0:t=fill,"
        "drawtext=fontfile=Anton-Regular.ttf:text='BREAKING  NEWS  TICKER  SCROLLS  BY  FAST':"
        r"x='w-mod(t*400\,w+tw)':y=640:fontsize=48:fontcolor=white"
    )
    assert detect_overlay_bands(_make_overlay(tmp_path / "ticker.mp4", ticker)) is None


def test_overlay_corner_minimap_is_game_content(tmp_path):
    # A static HUD/minimap touching the bottom edge but covering a corner, not the width.
    hud = (
        "drawbox=x=0:y=500:w=300:h=220:color=0x204020:t=fill,"
        + ",".join(
            f"drawtext=fontfile=Anton-Regular.ttf:text='{t}':x=12:y={y}:fontsize=40:fontcolor=white"
            for t, y in (("MAP  N  12", 510), ("HP 100  AMMO 30", 560), ("X 512 Y 77", 610))
        )
        + ",drawtext=fontfile=Anton-Regular.ttf:text='SCORE 4521':x=12:y=660:fontsize=40:"
        "fontcolor=white"
    )
    assert detect_overlay_bands(_make_overlay(tmp_path / "hud.mp4", hud, pan=True)) is None
    # a full-width lower-third over the same moving camera is an overlay
    bands = detect_overlay_bands(_make_overlay(tmp_path / "strip.mp4", _LOWER_THIRD, pan=True))
    assert bands is not None and bands[1] / 720 == pytest.approx(0.15, abs=0.02)


def test_overlay_fighting_game_health_bars_are_game_content(tmp_path):
    # Two static bordered health bars with names + a centre timer box, touching the top.
    parts = [
        "drawbox=x=40:y=20:w=520:h=40:color=0xc02020:t=fill",
        "drawbox=x=40:y=20:w=520:h=40:color=white:t=4",
        "drawbox=x=720:y=20:w=520:h=40:color=0xc02020:t=fill",
        "drawbox=x=720:y=20:w=520:h=40:color=white:t=4",
        "drawbox=x=600:y=8:w=80:h=70:color=black:t=fill",
        "drawbox=x=600:y=8:w=80:h=70:color=white:t=4",
        "drawtext=fontfile=Anton-Regular.ttf:text='99':x=618:y=14:fontsize=48:fontcolor=white",
        "drawtext=fontfile=Anton-Regular.ttf:text='RYU':x=52:y=24:fontsize=28:fontcolor=white",
        "drawtext=fontfile=Anton-Regular.ttf:text='KEN':x=1180:y=24:fontsize=28:fontcolor=white",
    ]
    hud = ",".join(parts)
    bands = detect_overlay_bands(_make_overlay(tmp_path / "fight.mp4", hud, pan=True))
    assert bands is None or bands[0] == 0


def test_overlay_wide_bottom_cam_frame_is_not_cropped(tmp_path):
    # A wide webcam frame at the bottom: static border, moving picture inside.
    cam = "drawbox=x=80:y=540:w=1120:h=170:color=white:t=6"
    assert detect_overlay_bands(_make_overlay(tmp_path / "cam.mp4", cam, pan=True)) is None


def test_overlay_unreadable_video_is_none(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert detect_overlay_bands(bad) is None


def test_overlay_paused_video_is_none(tmp_path):
    paused = _make_overlay(tmp_path / "paused.mp4", _LOWER_THIRD, still=True)
    assert detect_overlay_bands(paused) is None


def test_overlay_plain_testsrc_is_none(tmp_path):
    assert detect_overlay_bands(_make_overlay(tmp_path / "plain.mp4", "null")) is None


def test_overlay_bands_are_in_cropped_space(tmp_path):
    boxed = _make_overlay(tmp_path / "boxed.mp4", f"{_LOWER_THIRD},pad=1280:920:0:100")
    bands = detect_overlay_bands(boxed, crop=(1280, 720, 0, 100))
    assert bands is not None
    assert bands[1] / 720 == pytest.approx(0.15, abs=0.02)


def test_combine_crop_shrinks_letterbox_crop_to_exclude_bands():
    assert combine_crop((1920, 1080), None, None) is None
    assert combine_crop((1920, 1080), None, (0, 200)) == (1920, 880, 0, 0)
    assert combine_crop((1920, 1080), (1920, 800, 0, 140), (20, 100)) == (1920, 680, 0, 160)
    # odd cuts keep even dimensions and offsets
    w, h, _, y = combine_crop((1280, 720), None, (13, 101))
    assert (w, h % 2, y % 2) == (1280, 0, 0)
    assert y >= 13 and y + h <= 720 - 101


def test_box_to_crop_maps_full_frame_fractions_into_crop_space():
    assert box_to_crop((0.7, 0.0, 0.3, 0.25), (320, 180), None) == (0.7, 0.0, 0.3, 0.25)
    x, y, w, h = box_to_crop((0.7, 0.0, 0.3, 0.25), (320, 180), (320, 140, 0, 0))
    assert (x, y, w, h) == pytest.approx((0.7, 0.0, 0.3, 45 / 140))
    # letterbox at y=100 of 920: a box at the top of the picture maps to y=0
    x, y, w, h = box_to_crop((0.0, 100 / 920, 0.5, 0.2), (1280, 920), (1280, 720, 0, 100))
    assert (x, y, w, h) == pytest.approx((0.0, 0.0, 0.5, 184 / 720))
    # a box sticking out of the crop is clipped to it
    _, y, _, h = box_to_crop((0.0, 0.7, 0.3, 0.3), (320, 180), (320, 140, 0, 0))
    assert y == pytest.approx(126 / 140) and y + h == pytest.approx(1.0)


def test_box_hits_bands():
    size = (320, 180)
    assert box_hits_bands((0.0, 0.7, 0.3, 0.3), size, None, (0, 40))  # rows 126..180
    assert not box_hits_bands((0.7, 0.0, 0.3, 0.25), size, None, (0, 40))
    assert box_hits_bands((0.7, 0.0, 0.3, 0.25), size, None, (20, 0))
    # bands are in letterbox-crop space
    assert not box_hits_bands((0, 0.0, 0.3, 0.09), (320, 220), (320, 180, 0, 20), (10, 0))
    assert box_hits_bands((0, 0.1, 0.3, 0.1), (320, 220), (320, 180, 0, 20), (10, 0))


def test_combine_crop_refuses_to_remove_too_much():
    # > 30% of the height in total: band removal skipped, letterbox crop kept
    assert combine_crop((1920, 1080), None, (200, 200)) is None
    assert combine_crop((1920, 1080), (1920, 800, 0, 140), (0, 300)) == (1920, 800, 0, 140)
    # letterbox + bands leaving < 60% of the original height: skipped too
    assert combine_crop((1920, 1080), (1920, 800, 0, 140), (0, 200)) == (1920, 800, 0, 140)


@pytest.mark.skipif(not _REAL_SRC.exists(), reason="real source not present")
def test_overlay_real_source_lower_third(tmp_path):
    # The purple lower-third is on screen for the whole VOD (630/630 samples, every 5 s).
    clip = tmp_path / "window.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "600", "-t", "20",
             "-i", str(_REAL_SRC), "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
             str(clip)],
            check=True,
        )  # fmt: skip
        bands = detect_overlay_bands(clip)
        assert bands is not None
        assert bands[0] == 0
        assert 0.15 <= bands[1] / 1080 <= 0.22
        # Without the lower-third in the frame (cropped above it), the talking-head shot
        # (static curtain, moving speaker) has no overlay band.
        assert detect_overlay_bands(clip, crop=(1920, 860, 0, 0)) is None
    finally:
        clip.unlink(missing_ok=True)


# --- M2.5 wave B: tracking holds, every cut steps, soft-source caps, small faces, headroom ---

_FPS = 30


def test_track_holds_crop_x_through_a_face_gap():
    from hookline.pipeline.layout import classify_scene
    from hookline.vision.faces import Face

    def face(cx, score=0.9):
        return Face(x=cx - 0.08, y=0.2, w=0.16, h=0.4, score=score)

    samples = []
    for i in range(13):
        t = i * 0.5
        if t <= 2.0:
            faces = [face(0.3)]
        elif t < 4.0:  # 1.5 s without a face; only a low-score body box elsewhere
            faces = [face(0.75, score=0.4)]
        else:
            faces = [face(0.3)]
        samples.append((t, faces))
    lay = classify_scene(samples, 0, 6.5)
    expr = track_x_expr(lay.track_x, in_w=1280, crop_w=404, face_w=lay.face_w)
    held = evaluate(expr, t=2.0)
    for t in (2.1, 2.5, 3.0, 3.5, 3.9, 4.5, 6.0):
        assert evaluate(expr, t=t) == pytest.approx(held)


def test_track_keeps_a_leaning_face_inside_the_crop():
    # clip 04: she leans from 0.57 to 0.77 in one sample; the EMA alone lagged ~0.14 behind
    kfs = [(0, 0.57), (1, 0.57), (2, 0.77), (3, 0.776), (4, 0.62), (5, 0.565)]
    face_w, in_w, crop_w = 0.17, 1280, 404
    expr = track_x_expr(kfs, in_w=in_w, crop_w=crop_w, face_w=face_w)
    for t, x in kfs:
        left = evaluate(expr, t=t)
        margin = 0.1 * crop_w - 1e-6
        assert left + margin <= (x - face_w / 2) * in_w
        assert (x + face_w / 2) * in_w <= left + crop_w - margin


def _cut_has_step(steps, c):
    return any(abs(t - c) <= 1 / _FPS + 1e-9 for t, _ in steps)


@pytest.mark.parametrize(
    ("duration", "cuts", "emph"),
    [
        (34.0, [14.667], [2.0, 14.867]),  # clip 04: the cut 0.2 s before a punch
        (20.0, [3.93, 13.67], [3.5, 13.0]),  # cuts inside punch windows
        (20.0, [5.0, 5.3, 5.45, 12.0], []),  # cuts closer than ZOOM_MIN_STEP
        (20.0, [0.2, 19.8], [0.6]),  # cuts near both ends
        (30.0, [6.4, 7.6, 14.83, 26.63], [6.0, 14.5, 26.2]),
    ],
)
def test_zoom_schedule_steps_at_every_cut(duration, cuts, emph):
    steps = zoom_schedule(duration, cuts, emph)
    for c in cuts:
        assert _cut_has_step(steps, c), (c, steps)
    assert all(a[1] != b[1] for a, b in pairwise(steps))
    assert all(a[0] < b[0] for a, b in pairwise(steps))
    assert max(_gaps(steps, duration)) <= 4.5 + 1e-9


def test_zoom_schedule_cut_inside_a_punch_ends_it_at_the_cut():
    steps = zoom_schedule(20.0, [6.8], [6.0])
    assert (6.0, ZOOM_PUNCH) in steps
    assert steps[steps.index((6.0, ZOOM_PUNCH)) + 1] == (6.8, 1.0)
    # a punch just after a cut moves onto it: the punch is the cut's framing change
    steps = zoom_schedule(20.0, [6.0], [6.3])
    assert (6.0, ZOOM_PUNCH) in steps and all(t != 6.3 for t, _ in steps)


def test_zoom_expr_back_to_back_steps_stay_monotonic_in_time():
    steps = [(0.0, 1.0), (5.0, ZOOM_IN), (5.1, 1.0), (5.2, ZOOM_IN)]
    expr = zoom_expr(steps)
    assert evaluate(expr, 5.1) == pytest.approx(ZOOM_IN)  # ease cut short by the next step
    assert evaluate(expr, 5.2) == pytest.approx(1.0)
    assert evaluate(expr, 6.0) == pytest.approx(ZOOM_IN)


_SOFT_TRACK = SceneLayout(
    start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)], face_box=(0.3, 0.3), head_top=0.15
)


def test_plan_zoom_soft_source_caps_every_level_and_uses_lanczos():
    plan = plan_zoom(_SOFT_TRACK, 1280, 720)  # 720p FACE_TRACK: 2.67x upscale
    assert scene_upscale(plan, 1280, 720) > 2.0
    levels = [plan.zoom_level(s) for s in (1.0, ZOOM_IN, ZOOM_PUNCH)]
    assert max(levels) <= 1.12 + 1e-9 and len(set(levels)) == 3  # capped, still distinct
    chain = scene_filter(plan, 1280, 720, "a", "b", zoom_steps=[(0, 1.0), (1, ZOOM_PUNCH)])
    assert "scale=1080:1920:flags=lanczos" in chain and "eval=frame:flags=lanczos" in chain
    assert chain.index("unsharp") > chain.index("eval=frame")  # sharpen after the zoom
    expr = zoom_expr([(0, 1.0), (1, ZOOM_PUNCH)], level=plan.zoom_level)
    assert evaluate(expr, 3.0) == pytest.approx(1.12)
    sharp = plan_zoom(_SOFT_TRACK, 3840, 2160)  # 4K: no cap, no lanczos
    assert sharp.zoom_cap is None and sharp.zoom_level(ZOOM_PUNCH) == pytest.approx(1.2)
    assert "lanczos" not in scene_filter(sharp, 3840, 2160, "a", "b", zoom_steps=_STEPS)


def test_plan_zoom_small_face_gets_a_base_zoom_within_limits():
    small = SceneLayout(
        start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)], face_box=(0.4, 0.15),
        head_top=0.3,
    )  # fmt: skip
    plan = plan_zoom(small, 1920, 1080)  # 1.78x upscale
    up = scene_upscale(plan, 1920, 1080)
    assert plan.base_zoom == pytest.approx(1.25)  # 15% -> 18.75% at the base level ...
    assert plan.zoom_level(ZOOM_PUNCH) * 0.15 >= 0.21 - 1e-9  # ... ~21% at the top level
    levels = [plan.zoom_level(s) for s in (1.0, ZOOM_IN, ZOOM_PUNCH)]
    assert levels == sorted(set(levels))  # steps compressed into [base, 1.4], still distinct
    assert levels[0] == pytest.approx(1.25) and levels[-1] <= 1.4 + 1e-9
    assert all(v * up <= 2.5 + 1e-9 for v in levels)
    assert plan.model_dump()["base_zoom"] == plan.base_zoom  # recorded in the sidecar
    # a big enough face, or a soft source (> 2x upscale), gets none: 720p unchanged
    assert plan_zoom(small.model_copy(update={"face_box": (0.3, 0.25)}), 1920, 1080).base_zoom == 1
    soft = plan_zoom(small, 1280, 720)
    assert soft.base_zoom == 1.0 and soft.zoom_cap == pytest.approx(1.12)
    flat = scene_filter(plan, 1920, 1080, "a", "b")  # no steps: the base zoom still applies
    assert "eval=frame" in flat


@pytest.mark.parametrize("face_y", [0.12, 0.2, 0.3, 0.45])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
def test_plan_zoom_keeps_head_top_inside_the_frame_at_every_level(face_y, size):
    fh = 0.3
    lay = SceneLayout(
        start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)], face_box=(face_y, fh),
        head_top=face_y - 0.5 * fh,
    )  # fmt: skip
    plan = plan_zoom(lay, *size)
    head = lay.head_top * OUT_H
    for s in (1.0, ZOOM_IN, ZOOM_PUNCH):
        if head >= 0.04 * OUT_H:  # never cuts above the head top: >= 4% headroom
            assert plan.zoom_y(head, s, OUT_H) >= 0.04 * OUT_H - 1e-6, s
        assert plan.zoom_y((face_y + fh) * OUT_H, s, OUT_H) <= OUT_H + 1e-6  # chin in frame
    if head < 0.04 * OUT_H:  # the source framing already cut it: zoom about the top row
        assert plan.anchor_px(OUT_H) == 0
    else:  # anchor rendered as planned
        chain = scene_filter(plan, *size, "a", "b", zoom_steps=_STEPS)
        assert f"min({plan.anchor_px(OUT_H):.3f}".rstrip("0").rstrip(".") in chain


def test_plan_zoom_caps_the_level_when_head_and_chin_cannot_both_fit():
    big = SceneLayout(
        start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)], face_box=(0.3, 0.62),
        head_top=0.0,
    )  # fmt: skip
    plan = plan_zoom(big, 3840, 2160)
    assert plan.zoom_cap is not None and plan.zoom_cap < ZOOM_PUNCH
    for s in (1.0, ZOOM_IN, ZOOM_PUNCH):
        assert plan.zoom_y(0.92 * OUT_H, s, OUT_H) <= OUT_H + 1e-6


# --- wave B review round 1 --------------------------------------------------------------


def test_track_ignores_a_single_outlier_sample():
    # another face for one 0.5 s sample: the crop must not pan towards it
    kfs = (
        [(i * 0.5, 0.4) for i in range(6)] + [(3.0, 0.8)] + [(3.5 + i * 0.5, 0.4) for i in range(6)]
    )
    for face_w in (None, 0.1):
        expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=face_w)
        for t in (2.5, 3.0, 3.25, 3.5, 4.0, 6.0):
            assert evaluate(expr, t=t) == pytest.approx(evaluate(expr, t=0))


def test_track_wide_close_up_face_keeps_the_deadband():
    # face_w 0.26 of a 1280 frame (333 px) leaves no room in a 404 px crop: no containment,
    # so sub-deadband jitter must not move the crop sample by sample
    kfs = [(0.0, 0.5)] + [(i * 0.5, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(1, 16)]
    expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=0.26)
    vals = {round(evaluate(expr, t=t), 3) for t, _ in kfs}
    assert len(vals) == 1


def test_track_containment_never_tighter_than_half_the_deadband():
    # a face nearly as wide as the room left: jitter of +-0.02 (< deadband / 2) holds still
    kfs = [(0.0, 0.5)] + [(i * 0.5, 0.5 + (0.02 if i % 2 else -0.02)) for i in range(1, 16)]
    expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=0.24)
    assert len({round(evaluate(expr, t=t), 3) for t, _ in kfs}) == 1


def test_zoom_schedule_punch_not_moved_onto_a_cut_before_min_step():
    steps = zoom_schedule(10.0, [0.3], [0.6])
    assert (0.3, ZOOM_IN) in steps  # the cut keeps its own step
    assert all(s != ZOOM_PUNCH for _, s in steps)  # the punch would land at 0.3 s: dropped
    assert all(t == 0 or t >= 0.3 for t, _ in steps)


_FLAT = SceneLayout(
    start=0, end=6, kind="FACE_TRACK", track_x=[(0, 0.3), (6, 0.3)], face_box=(0.3, 0.69),
    head_top=0.05,
)  # fmt: skip


def test_plan_zoom_flat_headroom_scene_shifts_crop_at_each_step():
    plan = plan_zoom(_FLAT, 3840, 2160)
    assert plan.zoom_level(ZOOM_PUNCH) <= plan.zoom_level(1.0) + 0.03  # headroom flattened it
    assert plan.step_shift == pytest.approx(0.04)
    steps = [(0.0, 1.0), (2.0, ZOOM_IN), (4.0, 1.0)]
    chain = scene_filter(plan, 3840, 2160, "a", "b", zoom_steps=steps)
    x = chain.split("crop=", 1)[1].split(":", 3)[2].replace("\\,", ",")
    cw = 1214  # _track_w(3840, 2160)
    a, b, c = (evaluate(x, t) for t in (1.0, 3.0, 5.0))
    assert b - a == pytest.approx(0.04 * cw, abs=1)  # track at 0.3: shifts right (more room)
    assert c == pytest.approx(a)
    assert "step_shift" in plan.model_dump()
    # a scene with room to zoom gets no shift
    assert plan_zoom(_SOFT_TRACK, 3840, 2160).step_shift == 0.0
