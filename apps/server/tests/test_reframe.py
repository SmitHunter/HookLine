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
    kfs = [(-1, 0.0), (0, 0.0), (1, 1.0), (2, 1.0), (3, 1.0)]  # a lone first 0.0 is a stray
    expr = track_x_expr(kfs, in_w=1920, crop_w=607)
    assert evaluate(expr, t=0) == pytest.approx(0)
    for t in (0, 0.5, 1, 1.5, 2, 2.5, 3, 10):
        assert 0 <= evaluate(expr, t=t) <= 1920 - 607
    # fix wave F: a move this fast (the whole frame in 1 s) is a reframe snap, not a pan
    assert evaluate(expr, t=0) <= evaluate(expr, t=0.5) <= evaluate(expr, t=1)
    assert evaluate(expr, t=1) == pytest.approx(1920 - 607)


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


def test_zoom_schedule_early_punch_from_zero_and_nothing_at_or_after_end():
    # fix wave F: a beat on the opening words punches from the first frame (was dropped)
    assert zoom_schedule(10.0, [], [0.3])[0] == (0.0, ZOOM_PUNCH)
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
        margin = 0.08 * crop_w - 1e-6  # wave D: the edge guard's margin
        face_w = 0.17 * 0.8  # fix wave F: the hard constraint is the face core
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
    steps = zoom_schedule(20.0, [7.3], [6.0])
    assert (6.0, ZOOM_PUNCH) in steps
    assert steps[steps.index((6.0, ZOOM_PUNCH)) + 1] == (7.3, 1.0)
    # wave D: a punch within CUT_QUIET (1 s) before a cut lands on the cut instead
    assert (6.8, ZOOM_PUNCH) in zoom_schedule(20.0, [6.8], [6.0])
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
    # ... nor at the first or last sample, where the median window would clamp to itself
    rest = [(i * 0.5, 0.4) for i in range(1, 12)]
    for kfs in ([(0.0, 0.8), *rest], [(0.0, 0.4), *rest, (6.0, 0.8)]):
        for face_w in (None, 0.1):
            expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=face_w)
            for t, _ in kfs:
                assert evaluate(expr, t=t) == pytest.approx(0.4 * 1280 - 202), (kfs[0], t)


def test_track_close_up_jitter_inside_the_guard_margin_holds_still():
    # face_w 0.2 of a 1280 frame (256 px) in a 404 px crop: +-0.03 jitter keeps the face more
    # than 8% inside the crop, so the deadband holds the crop still sample by sample
    kfs = [(0.0, 0.5)] + [(i * 0.5, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(1, 16)]
    expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=0.2)
    vals = {round(evaluate(expr, t=t), 3) for t, _ in kfs}
    assert len(vals) == 1


def test_track_close_up_that_fits_the_margin_stays_inside_it():
    # a face that just fits inside the crop with the 8% margins is kept there by the guard
    kfs = [(0.0, 0.5)] + [(i * 0.5, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(1, 16)]
    from hookline.media.reframe import _median3

    med = _median3([x for _, x in kfs])  # the guard follows the outlier-filtered position
    for face_w in (0.2, 0.22):
        expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=face_w)
        for (t, _), x in zip(kfs, med, strict=True):
            left = evaluate(expr, t=t)
            assert left + 0.08 * 404 - 1e-6 <= (x - face_w / 2) * 1280
            assert (x + face_w / 2) * 1280 <= left + 404 - 0.08 * 404 + 1e-6


def test_track_close_up_wider_than_the_margin_keeps_the_deadband_with_containment():
    # review round 1: a face wider than the guard room is not guarded (no 1:1 jitter following,
    # no zoom hold); the deadband/EMA path only moves the crop as far as keeping it inside
    from hookline.media.reframe import _median3, track_guard_spans

    kfs = [(0.0, 0.5)] + [(i * 0.5, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(1, 16)]
    med = _median3([x for _, x in kfs])
    face_w = 0.3  # 384 px of a 404 px crop
    expr = track_x_expr(kfs, in_w=1280, crop_w=404, face_w=face_w)
    lefts = [evaluate(expr, t=t) for t, _ in kfs]
    core = 0.8 * face_w  # the face core: the detector box without ears/cheek edges
    for left, x in zip(lefts, med, strict=True):
        assert left - 1e-6 <= (x - core / 2) * 1280  # the face core stays inside the crop
        assert (x + core / 2) * 1280 <= left + 404 + 1e-6
    face_moves = max(abs(b - a) for a, b in pairwise(med)) * 1280  # 76.8 px
    assert max(abs(b - a) for a, b in pairwise(lefts)) < face_moves * 0.5  # not 1:1
    assert track_guard_spans(kfs, 1280, 404, face_w) == []


def test_whole_scene_webcam_close_up_caps_the_zoom_and_keeps_steps():
    # a full-screen webcam face ~0.8 of the crop width: the zoom is capped so the face still
    # fits with the margin, the scene is never guarded, and its cut steps survive
    from hookline.media.reframe import guard_zoom, track_guard_spans

    in_w, in_h, cw = 1920, 1080, 606
    fw = 0.8 * cw / in_w
    kfs = [(i * 0.25, 0.5 + (0.004 if i % 2 else -0.004)) for i in range(40)]
    lay = SceneLayout(
        start=0, end=10, kind="FACE_TRACK", track_x=kfs, track_w=[fw] * 40, face_w=fw,
        face_box=(0.25, 0.5), head_top=0.0,
    )  # fmt: skip
    plan = plan_zoom(lay, in_w, in_h)
    top = plan.zoom_level(ZOOM_PUNCH)
    assert top * 0.8 <= 1 - 2 * 0.08 + 1e-6  # the face fits the visible crop with margins
    assert track_guard_spans(kfs, in_w, cw, fw, [fw] * 40, zoom_top=top) == []
    steps = [(0.0, 1.0), (2.0, ZOOM_IN), (5.0, 1.0), (7.0, ZOOM_PUNCH)]
    held, lays = guard_zoom([plan], in_w, in_h, steps, cut_times=[2.0, 5.0])
    assert held == steps and lays[0].shift_steps == []


def _lean_layout() -> SceneLayout:
    xs = [0.5] * 8 + [0.6, 0.7, 0.75, 0.75, 0.75, 0.75, 0.7, 0.6] + [0.5] * 8
    return SceneLayout(
        start=0, end=6, kind="FACE_TRACK", track_x=[(i * 0.25, x) for i, x in enumerate(xs)],
        face_w=0.13, face_box=(0.3, 0.3), head_top=0.15,
    )  # fmt: skip


def test_track_guard_width_ignores_one_stray_wide_detection():
    from hookline.media.reframe import track_guard_spans

    kfs = [(i * 0.25, 0.5) for i in range(20)]
    widths = [0.13] * 20
    widths[9] = 0.4  # one oversize box (a hand in front of the face)
    assert track_guard_spans(kfs, 1920, 486, 0.13, widths) == []


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


# --- wave B review round 2 --------------------------------------------------------------


@pytest.mark.parametrize("size", [(1706, 960), (1752, 986), (1800, 1012)])
def test_plan_zoom_small_face_base_leaves_step_range(size):
    # ~1.9-2.0x upscale: MAX_UPSCALE leaves ~1.25-1.3 total zoom; the base must not eat it all
    small = SceneLayout(
        start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)], face_box=(0.4, 0.15),
        head_top=0.3,
    )  # fmt: skip
    plan = plan_zoom(small, *size)
    up = scene_upscale(plan, *size)
    assert 1.9 < up <= 2.0
    levels = [plan.zoom_level(s) for s in (1.0, ZOOM_IN, ZOOM_PUNCH)]
    assert plan.base_zoom > 1.0
    assert levels[-1] - levels[0] >= 0.1 - 1e-3  # steps keep >= 0.1 above the base
    assert levels == sorted(set(levels)) and levels[-1] * up <= 2.5 + 1e-9
    assert plan.step_shift == 0.0


def _shift_x(plan, size, steps):
    chain = scene_filter(plan, *size, "a", "b", zoom_steps=steps)
    return chain.split("crop=", 1)[1].split(":", 3)[2].replace("\\,", ",")


@pytest.mark.parametrize(
    ("track", "face_w"), [(0.1, 0.2), (0.9, 0.2), (0.5, 0.3), (0.05, 0.08), (0.95, 0.08)]
)
def test_step_shift_never_trims_the_face(track, face_w):
    # crop pinned at a frame edge (track 0.1/0.9), or a face too wide to shift past: the shift
    # must never move the crop over the face; when it shifts, the face moves towards the centre
    lay = _FLAT.model_copy(update={"track_x": [(0, track), (6, track)], "face_w": face_w})
    size, cw = (3840, 2160), 1214
    plan = plan_zoom(lay, *size)
    assert plan.zoom_level(ZOOM_PUNCH) <= plan.zoom_level(1.0) + 0.03
    steps = [(0.0, 1.0), (2.0, ZOOM_IN), (4.0, 1.0)]
    x = _shift_x(plan, size, steps)
    fc = track * size[0]
    rest = evaluate(x, 1.0)
    for t in (1.0, 2.1, 3.0, 4.1, 5.0):
        left = evaluate(x, t)
        assert left <= fc - face_w * size[0] / 2 + 1e-6 or left <= 0
        assert fc + face_w * size[0] / 2 <= left + cw + 1e-6 or left >= size[0] - cw
        # never further off-centre than unshifted
        assert abs(fc - (left + cw / 2)) <= abs(fc - (rest + cw / 2)) + 1e-6 or left == rest
    if evaluate(x, 3.0) == pytest.approx(rest):
        assert plan.step_shift == 0.0  # the sidecar records only a shift that happens


def test_step_shift_only_recorded_where_rendered():
    split = SceneLayout(
        start=0, end=6, kind="SPLIT", split_x=(0.3, 0.7), face_box=(0.3, 0.69), head_top=0.05
    )
    plan = plan_zoom(split, 3840, 2160)
    assert plan.zoom_level(ZOOM_PUNCH) <= plan.zoom_level(1.0) + 0.03  # flat, like _FLAT
    assert plan.step_shift == 0.0  # SPLIT never renders a shift


# --- wave D: edge guard, gaze lead, instant cut zooms, quiet zone around cuts ------------


def _nesting(expr: str) -> int:
    depth = best = 0
    for c in expr:
        depth += c == "("
        depth -= c == ")"
        best = max(best, depth)
    return best


def test_track_expression_stays_shallow_with_many_keyframes(tmp_path):
    # 4 fps over 60 s: a nested if() per keyframe passes ffmpeg's ~100-level parser limit
    kfs = [(i * 0.25, 0.5 + 0.2 * math.sin(i / 7)) for i in range(240)]
    expr = track_x_expr(kfs, in_w=1920, crop_w=486, face_w=0.1)
    assert _nesting(expr) <= 40
    for t in (0.0, 7.3, 31.0, 59.75, 80.0):
        assert 0 <= evaluate(expr, t) <= 1920 - 486
    graph = tmp_path / "g.txt"
    graph.write_text(
        "color=black:s=1920x864:r=25:d=60,crop=486:864:x='"
        + expr.replace(",", r"\,")
        + "':y=0[vout]",
        encoding="utf-8",
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-filter_complex_script", str(graph),
         "-map", "[vout]", "-frames:v", "3", "-f", "null", "-"],
        check=True,
    )  # fmt: skip


def test_track_expression_drops_collinear_keyframes():
    from hookline.media.reframe import _drop_collinear

    kfs = [(i * 0.25, 0.5) for i in range(200)]
    assert track_x_expr(kfs, in_w=1920, crop_w=486) == "717"  # one value: a constant
    line = [(i * 0.25, 100 + 3 * i) for i in range(50)]
    assert _drop_collinear(line, 0.5) == [line[0], line[-1]]
    curve = [(i * 0.25, 400 + 80 * math.sin(i / 9)) for i in range(120)]
    kept = _drop_collinear(curve, 0.5)
    assert len(kept) < len(curve) / 2
    for t, v in curve:  # every dropped point within the tolerance of the kept polyline
        (t0, v0), (t1, v1) = next((a, b) for a, b in pairwise(kept) if a[0] <= t <= b[0])
        assert abs(v - (v0 + (v1 - v0) * (t - t0) / (t1 - t0))) <= 0.5 + 1e-9


def _inside(left: float, cw: float, x: float, fw: float, in_w: float, margin: float) -> bool:
    lo, hi = (x - fw / 2) * in_w, (x + fw / 2) * in_w
    return left + margin * cw - 1e-6 <= lo and hi <= left + cw - margin * cw + 1e-6


def test_track_edge_guard_keeps_a_fast_lean_inside_the_crop():
    # clip 05: track 0.52 -> 0.80 -> 0.55 within 3 s on a 1920 px frame, 486 px crop, the face
    # growing as she leans in; 4 fps samples
    xs = (
        [0.52] * 8 + [0.56, 0.62, 0.70, 0.77, 0.80, 0.80, 0.79, 0.74, 0.66, 0.58, 0.55] + [0.55] * 8
    )
    ws = (
        [0.13] * 8 + [0.14, 0.15, 0.16, 0.17, 0.17, 0.17, 0.16, 0.15, 0.14, 0.13, 0.13] + [0.13] * 8
    )
    kfs = [(i * 0.25, x) for i, x in enumerate(xs)]
    in_w, cw = 1920, 486
    expr = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.13, widths=ws)
    for (t, x), w in zip(kfs, ws, strict=True):
        # fix wave F: the hard constraint is the face core (FACE_CORE of the box) + margin
        assert _inside(evaluate(expr, t), cw, x, w * 0.8, in_w, 0.08), t


def test_track_edge_guard_still_ignores_a_single_stray_sample():
    kfs = [(i * 0.25, 0.5) for i in range(12)]
    kfs[6] = (1.5, 0.85)  # another face for one sample
    expr = track_x_expr(kfs, in_w=1920, crop_w=486, face_w=0.13)
    assert {round(evaluate(expr, t), 3) for t, _ in kfs} == {round(0.5 * 1920 - 243, 3)}


def test_track_guard_reports_spans_and_accounts_for_the_zoom():
    from hookline.media.reframe import track_guard_spans

    xs = [0.5] * 8 + [0.6, 0.7, 0.75, 0.75, 0.7, 0.6] + [0.5] * 8
    kfs = [(i * 0.25, x) for i, x in enumerate(xs)]
    flat = track_guard_spans(kfs, in_w=1920, crop_w=486, face_w=0.13)
    zoomed = track_guard_spans(kfs, in_w=1920, crop_w=486, face_w=0.13, zoom_top=1.12)
    assert flat and all(0.0 < a < b <= 5.5 for a, b in flat)
    # a zoomed-in crop is narrower: the guard engages at least as long
    assert sum(b - a for a, b in zoomed) >= sum(b - a for a, b in flat)
    assert track_guard_spans([(i * 0.25, 0.5) for i in range(20)], 1920, 486, 0.13) == []


def test_track_lead_biases_the_crop_toward_the_gaze_within_containment():
    from hookline.media.reframe import track_guard_spans

    kfs = [(i * 0.25, 0.5) for i in range(8)]
    in_w, cw = 1920, 486
    centre = 0.5 * in_w - cw / 2
    left = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.1, lead=-0.08)
    right = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.1, lead=0.08)
    assert evaluate(left, 1.0) == pytest.approx(centre - 0.08 * cw, abs=0.5)  # room on the left
    assert evaluate(right, 1.0) == pytest.approx(centre + 0.08 * cw, abs=0.5)
    # the widest guarded face still takes the lead and stays 8% inside
    snug = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.17, lead=-0.08)
    assert _inside(evaluate(snug, 1.0), cw, 0.5, 0.17, in_w, 0.08)
    assert evaluate(snug, 1.0) < centre  # leans the way she looks
    # a tight close-up (wider than the guard room) keeps its face core inside the crop
    wide = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.24, lead=-0.08)
    assert _inside(evaluate(wide, 1.0), cw, 0.5, 0.24 * 0.8, in_w, 0.0)
    assert evaluate(wide, 1.0) < centre
    # the lead alone never engages the guard (no zoom hold for a steady, well-framed face)
    assert track_guard_spans(kfs, in_w, cw, 0.2, lead=-0.08) == []


def test_scene_filter_holds_zoom_at_base_while_the_guard_is_active():
    from hookline.media.reframe import guarded_steps

    steps = [(0.0, 1.0), (1.0, ZOOM_IN), (3.0, ZOOM_PUNCH), (6.0, 1.0)]
    held = guarded_steps(steps, [(2.0, 4.0)])
    expr = zoom_expr(held)
    assert evaluate(expr, 1.5) == pytest.approx(ZOOM_IN)
    for t in (2.25, 3.0, 3.5, 3.95):
        assert evaluate(expr, t) == pytest.approx(1.0)  # held at the base level
    assert evaluate(expr, 4.5) == pytest.approx(ZOOM_PUNCH)  # back to the schedule after
    assert evaluate(expr, 7.0) == pytest.approx(1.0)
    assert guarded_steps(steps, []) == steps


def test_scene_filter_face_track_guard_holds_the_zoom():
    from hookline.media.reframe import guard_zoom

    lay = _lean_layout()
    steps = [(0.0, 1.0), (0.5, ZOOM_IN), (5.5, 1.0)]
    held, lays = guard_zoom([lay], 1920, 864, steps, cut_times=[])
    chain = scene_filter(lays[0], 1920, 864, "a", "b", zoom_steps=held)
    s = chain.split("scale=w='trunc(1080*(", 1)[1].split(")/2)*2'", 1)[0].replace("\\,", ",")
    assert evaluate(s, 1.2) > 1.0  # zoomed before the lean
    assert evaluate(s, 2.9) == pytest.approx(1.0)  # held at base during it
    # a zoom-in within CUT_QUIET before the hold joins it: no in-then-out pumping
    near, _ = guard_zoom([lay], 1920, 864, [(0.0, 1.0), (1.2, ZOOM_IN)], cut_times=[])
    assert (1.2, ZOOM_IN) not in near and evaluate(zoom_expr(near), 2.9) == 1.0


def test_guard_zoom_keeps_a_framing_change_at_a_cut_inside_a_span():
    # review round 1: a cut inside a guard span still changes the framing instantly, either
    # with a zoom step that keeps the face inside the margin or a shift toward the centre
    from hookline.media.reframe import guard_zoom, track_guard_spans

    lay = _lean_layout()
    rel = lay.track_x
    spans = track_guard_spans(rel, 1920, 486, 0.13, zoom_top=1.12)
    (a, b), *_ = spans
    cut = round((a + b) / 2, 3)
    steps = [(0.0, 1.0), (1.0, ZOOM_IN), (cut, 1.0), (5.5, ZOOM_IN)]
    held, lays = guard_zoom([lay], 1920, 864, steps, cut_times=[cut])
    zoom_at_cut = [v for t, v in held if t == cut]
    shifts = [t for t, _ in lays[0].shift_steps]
    assert zoom_at_cut or cut in shifts
    if zoom_at_cut:
        before = [v for t, v in held if t < cut][-1]
        assert zoom_at_cut[0] != before
    # the return to the schedule never lands within CUT_QUIET of the cut
    for t, _ in held:
        if cut < t < cut + 1.0 - 1e-9:
            raise AssertionError((t, held))


def test_guard_zoom_return_step_moves_onto_a_nearby_cut():
    from hookline.media.reframe import guard_zoom, track_guard_spans

    lay = _lean_layout()
    (_, b), *_ = track_guard_spans(lay.track_x, 1920, 486, 0.13, zoom_top=1.12)
    cut = round(b + 0.5, 3)
    steps = [(0.0, 1.0), (1.0, ZOOM_IN), (cut, 1.0)]
    held, _ = guard_zoom([lay], 1920, 864, steps, cut_times=[cut])
    assert all(not (b - 1e-9 <= t < cut) for t, _ in held), held  # no return 0.5 s before it


def test_scene_filter_applies_shift_steps_instantly_at_cuts():
    lay = SceneLayout(
        start=0, end=6, kind="FACE_TRACK", track_x=[(0, 0.5), (6, 0.5)], face_w=0.1,
        shift_steps=[(2.0, 0.04), (4.0, 0.0)],
    )  # fmt: skip
    chain = scene_filter(lay, 1920, 864, "a", "b", zoom_steps=[(0.0, 1.0)], cut_times=[2.0])
    x = chain.split("crop=", 1)[1].split(":", 3)[2].replace("\\,", ",")
    base = 960 - 243
    assert evaluate(x, 1.9) == pytest.approx(base)
    assert evaluate(x, 2.0) == pytest.approx(base + 0.04 * 486, abs=0.5)  # instant at the cut
    assert evaluate(x, 5.0) == pytest.approx(base)


def test_zoom_expr_steps_at_cuts_are_instant():
    steps = [(0.0, 1.0), (2.0, ZOOM_IN), (5.0, 1.0)]
    expr = zoom_expr(steps, instant=[2.0])
    assert evaluate(expr, 1.99) == pytest.approx(1.0)
    assert evaluate(expr, 2.0 - 1 / 240) == pytest.approx(ZOOM_IN)  # the cut frame is zoomed
    assert evaluate(expr, 2.0) == pytest.approx(ZOOM_IN)
    assert evaluate(expr, 5.1) == pytest.approx(1.06)  # a beat still eases over 0.2 s
    # scene-relative: an instant step is matched in output time
    rel = zoom_expr(steps, offset=1.0, instant=[2.0])
    assert evaluate(rel, 0.99) == pytest.approx(1.0) and evaluate(rel, 1.0) == ZOOM_IN


def test_reframe_graph_cut_zoom_is_instant(tmp_path):
    static = tmp_path / "static.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "color=c=yellow:s=1280x720:r=25:d=4,drawbox=x=0:y=100:w=1280:h=60:c=red:t=fill",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(static)],
        check=True,
    )  # fmt: skip
    lay = SceneLayout(start=0, end=4, kind="FACE_TRACK", track_x=[(0, 0.5)])
    steps = [(0.0, 1.0), (2.0, ZOOM_IN)]
    graph = reframe_graph([lay], 1280, 720, zoom_steps=steps, cut_times=[2.0])
    out = _render(static, graph, tmp_path / "x.mp4")
    edges = [_top_edge(f[:, 100]) for f in _gray_frames(out, 1.88, 2.2)]
    assert edges[0] == edges[2] and edges[3] == edges[-1]  # frames 47-49, then 50 on
    assert edges[2] - edges[3] > 20  # the whole step lands on the cut frame (t = 2.0)


def test_zoom_schedule_no_beat_step_within_a_second_of_a_cut():
    # clip 03: punch 1.4, camera change 2.43, release 2.8, cut 3.93 -> fidgety
    steps = zoom_schedule(40.0, [3.933, 25.533], [1.4, 3.3, 26.3], changes=[2.433, 6.567])
    quiet = [3.933, 25.533, 2.433, 6.567]
    for t, _ in steps:
        if t == 0.0 or any(abs(t - c) < 1e-6 for c in quiet):
            continue
        assert all(abs(t - c) >= 1.0 - 1e-6 for c in quiet), (t, steps)
    assert (3.933, ZOOM_PUNCH) in steps  # a punch 0.63 s before the cut lands on it
    assert (25.533, ZOOM_PUNCH) in steps  # ... and 0.77 s after it
    times = [t for t, _ in steps] + [40.0]
    marks = sorted({*times, *quiet})
    assert max(b - a for a, b in pairwise(marks)) <= 4.5 + 1e-9


def test_zoom_schedule_release_near_a_cut_happens_at_the_cut():
    steps = zoom_schedule(20.0, [8.0], [6.0])  # release would land at 7.4, 0.6 s before the cut
    i = steps.index((6.0, ZOOM_PUNCH))
    assert steps[i + 1] == (8.0, 1.0)


def test_rendered_zoom_steps_map_levels_through_each_scene():
    from hookline.media.reframe import rendered_zoom_steps

    soft = plan_zoom(_SOFT_TRACK.model_copy(update={"end": 3.0}), 1280, 720)
    sharp = SceneLayout(start=3.0, end=6.0, kind="FIT_BLUR")
    steps = [(0.0, 1.0), (1.0, ZOOM_IN), (2.0, ZOOM_PUNCH), (4.0, ZOOM_IN)]
    got = rendered_zoom_steps([soft, sharp], steps)
    assert got[0] == (0.0, 1.0)
    assert got[1][1] == pytest.approx(soft.zoom_level(ZOOM_IN)) and got[1][1] < ZOOM_IN
    assert got[2][1] == pytest.approx(1.12)  # the soft scene's punch renders at its cap
    assert got[3] == (4.0, pytest.approx(ZOOM_IN))


# --- wave D review round 2: guard fixed point, holds hold, shifts checked as rendered ------


def _lean_at(start: float, end: float, xs: list[float], fw: float = 0.13) -> SceneLayout:
    return SceneLayout(
        start=start, end=end, kind="FACE_TRACK",
        track_x=[(start + i * 0.25, x) for i, x in enumerate(xs)],
        face_w=fw, face_box=(0.3, 0.3), head_top=0.15,
    )  # fmt: skip


_LEAN = [0.5] * 8 + [0.6, 0.7, 0.75, 0.75, 0.75, 0.75, 0.7, 0.6] + [0.5] * 8
_WOBBLE = [0.5 + (0.12 if (i // 2) % 2 else -0.12) for i in range(40)]
_TWO_HOLDS = [0.5] * 4 + [0.75] * 4 + [0.5] * 5 + [0.25] * 4 + [0.5] * 8 + [0.75] * 4 + [0.5] * 10
_GUARD_CASES = {  # the round-2 reviewer's probes (probeD/probe1.py, probe3.py)
    "cut inside hold": (
        [_lean_at(0, 6, _LEAN)],
        [(0.0, 1.0), (1.0, ZOOM_IN), (3.0, 1.0), (5.5, ZOOM_IN)],
        [3.0],
    ),
    "beat inside hold": (
        [_lean_at(0, 6, _LEAN)],
        [(0.0, 1.0), (1.0, ZOOM_IN), (3.0, 1.0), (5.5, ZOOM_IN)],
        [],
    ),
    "whole-scene hold": (
        [_lean_at(0, 10, _WOBBLE)],
        [(0.0, 1.0), (2.0, ZOOM_IN), (4.0, 1.0), (6.0, ZOOM_IN), (8.0, ZOOM_PUNCH)],
        [2.0, 4.0, 6.0, 8.0],
    ),
    "face at frame edge": (
        [_lean_at(0, 10, [0.93] * 40)],
        [(0.0, 1.0), (2.0, ZOOM_IN), (4.0, 1.0), (6.0, ZOOM_IN)],
        [2.0, 4.0, 6.0],
    ),
    "back-to-back holds": (
        [_lean_at(0, 10, _TWO_HOLDS)],
        [(0.0, 1.0), (0.5, ZOOM_IN), (2.6, 1.0), (4.5, ZOOM_IN), (8.0, 1.0)],
        [2.6, 4.5],
    ),
    "return inside the next span": (
        [_lean_at(0, 10, _TWO_HOLDS)],
        [(0.0, 1.0), (0.5, ZOOM_IN), (1.9, 1.0), (3.3, ZOOM_IN), (8.0, 1.0)],
        [1.9, 3.3],
    ),
    "cut pulls span earlier": (
        [_lean_at(0, 10, _TWO_HOLDS)],
        [(0.0, 1.0), (0.5, ZOOM_IN), (4.65, 1.0), (5.1, ZOOM_IN), (8.5, 1.0)],
        [4.65, 5.1],
    ),
    "hold across a camera change": (
        [
            _lean_at(0, 3, [0.5] * 8 + [0.62, 0.7, 0.75, 0.75]),
            _lean_at(3, 7, [0.75, 0.75, 0.7, 0.6] + [0.5] * 12),
        ],
        [(0.0, 1.0), (1.0, ZOOM_IN), (3.0, 1.0), (5.0, ZOOM_IN)],
        [3.0],
    ),
}


def _final_spans(steps, lay, in_w=1920, in_h=864):
    from hookline.media.reframe import _track_w, track_guard_spans, zoom_at_fn

    rel = [(t - lay.start, x) for t, x in lay.track_x]
    spans = track_guard_spans(
        rel, in_w, _track_w(in_w, in_h), lay.face_w, lay.track_w or None, lay.lead,
        zoom_top=zoom_at_fn(steps, lay),
    )  # fmt: skip
    return [(a + lay.start, b + lay.start) for a, b in spans]


@pytest.mark.parametrize("name", list(_GUARD_CASES))
def test_guard_zoom_is_a_fixed_point_and_holds_hold(name):
    from hookline.media.reframe import guard_zoom

    lays, steps, cuts = _GUARD_CASES[name]
    lays = [plan_zoom(lay, 1920, 864) for lay in lays]
    held, out = guard_zoom(lays, 1920, 864, steps, cuts)
    # the same pre-guard schedule gives the same result, whatever shifts the layouts carry
    again, out2 = guard_zoom(out, 1920, 864, steps, cuts)
    assert again == held and [o.shift_steps for o in out2] == [o.shift_steps for o in out]
    for lay in out:  # the guard does not fire again at the zoom it renders
        for a, b in _final_spans(held, lay):
            inside = [(t, v) for t, v in held if a <= t < b and v != 1.0]
            assert not inside, (name, (a, b), inside)


def test_hold_spans_never_zooms_in_inside_a_hold_nor_returns_into_the_next():
    from hookline.media.reframe import _hold_spans

    class Fit:
        def __init__(self, ok):
            self.ok = ok

        def fits(self, level, t0, t1, shift=0.0, path=False):
            return self.ok(level, shift)

        def direction(self, t0, t1):
            return 1.0

    def run(steps, spans, cuts, ok, scene=(0, 20)):
        held, shifts = _hold_spans(list(steps), spans, cuts, lambda _: Fit(ok), scene)
        for a, b in spans:
            assert not [(t, v) for t, v in held if a <= t < b and v != 1.0], (held, spans)
        return held, shifts

    yes = lambda lvl, sh: True
    held, _ = run([(0, 1.0), (1, ZOOM_IN), (3.9, 1.0), (4.4, ZOOM_IN), (9, 1.0)],
                  [(2, 4), (5.1, 7)], [3.9, 4.4], yes)  # fmt: skip
    assert not [t for t, _ in held if 4.9 < t < 5.1]  # no return 0.2 s before the next hold
    held, shifts = run([(0, 1.0), (1, ZOOM_IN), (5.8, 1.0)], [(2, 6)], [5.8], yes, scene=(0, 6))
    assert all(t < 6 for t, _ in held) and all(t < 6 for t, _ in shifts)  # not past the scene
    held, shifts = run([(0, 1.0), (3, ZOOM_IN), (5, 1.0)], [(2, 8)], [3, 5],
                       lambda lvl, sh: not sh)  # fmt: skip
    assert shifts == []  # no safe change: the cuts alone change the framing (pinned edge)


def test_guard_shifts_are_checked_at_the_rendered_zoom():
    # a shift outside a hold, where the crop renders zoomed in, is validated at that zoom
    # (a 0.13 face: a 0.17 one now caps the zoom flat so it fits the guard margin)
    from hookline.media.reframe import _SceneFit

    lay = plan_zoom(_lean_at(0, 6, [0.5] * 24, fw=0.13), 1920, 864)
    steps = [(0.0, ZOOM_IN)]
    fit = _SceneFit(lay, 1920, 864, steps)
    room_base = 486 * 0.42 / lay.zoom_level(1.0)
    room_zoomed = 486 * 0.42 / lay.zoom_level(ZOOM_IN)
    # a shift that fits the base view but not the zoomed one
    shift = (room_base - 0.13 * 1920 / 2) / 486 * 0.99
    assert shift * 486 > room_zoomed - 0.13 * 1920 / 2
    assert not fit.fits(None, 1.0, 2.0, shift=shift)


def test_face_offset_includes_the_guard_shifts():
    from hookline.media.reframe import face_offset

    lay = _lean_at(0, 6, [0.5] * 24, fw=0.1)
    plain = face_offset(lay, 1920, 864, [(0.0, 1.0)])
    shifted = lay.model_copy(update={"shift_steps": [(0.0, 0.04)]})
    assert face_offset(shifted, 1920, 864, [(0.0, 1.0)]) == pytest.approx(plain - 0.04, abs=1e-3)


# --- wave D review round 3: one placement pass, spaced pulses, eases, lead slack ---------


def test_guard_zoom_unsettled_scene_holds_base_to_its_end(monkeypatch, caplog):
    from hookline.media import reframe

    monkeypatch.setattr(reframe, "GUARD_PASSES", 1)  # no pass can confirm the spans settled
    lays, steps, cuts = _GUARD_CASES["back-to-back holds"]
    lay = plan_zoom(lays[0], 1920, 864)
    with caplog.at_level("WARNING"):
        held, _ = reframe.guard_zoom([lay], 1920, 864, steps, cuts)
    assert any("did not settle" in r.getMessage() for r in caplog.records)
    # the face first moves at 1.0 s: from the first hold on, base to the end (no return)
    assert all(v == 1.0 for t, v in held if t >= 1.0)


def test_guard_pulses_are_spaced_and_keep_one_direction_per_span():
    from hookline.media.reframe import _hold_spans

    class Fit:
        def fits(self, level, t0, t1, shift=0.0, path=False):
            return True

        def direction(self, t0, t1):  # the nearest sample would flip half-way through
            return 1.0 if t0 < 3 else -1.0

    sched = [(0.0, 1.0), (2.0, ZOOM_IN), (2.4, 1.0), (4.0, ZOOM_IN), (4.3, 1.0), (6.0, ZOOM_IN)]
    cuts = [2.0, 4.0, 4.3, 6.0, 6.5]
    _, shifts = _hold_spans(sched, [(1.0, 9.0)], cuts, lambda _: Fit(), (0, 12))
    ons = [(t, v) for t, v in shifts if v]
    # 2.4 and 6.5 are under 1 s after a pulse; one at 4.0 would be cut to 0.3 s by the cut at
    # 4.3 (no blinks), so 4.3 takes it
    assert [t for t, _ in ons] == [2.0, 4.3, 6.0]
    assert {v for _, v in ons} == {0.04}  # the span's one direction


def test_zoom_at_fn_counts_a_zoom_out_while_it_eases():
    from hookline.media.reframe import zoom_at_fn

    lay = SceneLayout(start=0, end=6, kind="FACE_TRACK")
    at = zoom_at_fn([(0.0, ZOOM_IN), (2.0, 1.0)], lay)
    assert at(2.1) == pytest.approx(ZOOM_IN)  # still easing out of the zoom-in
    assert at(2.5) == pytest.approx(1.0)


def test_no_pulse_switches_off_just_before_a_scene_end():
    from hookline.media.reframe import _hold_spans

    class Fit:
        def fits(self, level, t0, t1, shift=0.0, path=False):
            return True

        def direction(self, t0, t1):
            return -1.0

    _, shifts = _hold_spans(
        [(0.0, 1.0), (1.4, ZOOM_IN)], [(0.5, 2.433)], [1.4], lambda _: Fit(), (0, 2.433)
    )
    assert shifts == [(1.4, -0.04)]  # 1.4-2.4 would end 0.03 s before the camera change


def test_gaze_lead_leaves_jitter_slack_before_the_guard_margin():
    kfs = [(i * 0.25, 0.5) for i in range(8)]
    in_w, cw = 1920, 486
    expr = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.17, lead=-0.08)
    left = evaluate(expr, 1.0)
    room_edge = left + cw - 0.08 * cw  # the guard margin on the side the face is pushed to
    assert (0.5 + 0.17 / 2) * in_w <= room_edge - 0.02 * in_w + 1e-6


def test_hold_spans_later_pulse_does_not_block_an_earlier_one_and_no_pumping():
    from hookline.media.reframe import _hold_spans

    class Fit:
        def fits(self, level, t0, t1, shift=0.0, path=False):
            return True

        def direction(self, t0, t1):
            return 1.0

    # a beat at 14.9 lost its change to a hold (too close for a return); a pulse at 25.267
    # (inside a later hold) must not stop it getting its own pulse
    sched = [(0.0, 1.0), (13.256, ZOOM_IN), (14.9, 1.0), (25.267, ZOOM_PUNCH)]
    held, shifts = _hold_spans(
        sched, [(13.256, 14.1), (25.267, 26.3)], [], lambda _: Fit(), (0, 30)
    )
    # (fix wave F: cadence pulses may fill the long stretches around them too)
    assert {13.256, 14.9, 25.267} <= {t for t, v in shifts if v}
    # a hold that starts on a zoom step never returns within CUT_QUIET of its start
    held, _ = _hold_spans(
        [(0.0, 1.0), (7.992, ZOOM_IN)], [(7.992, 8.44)], [], lambda _: Fit(), (0, 20)
    )
    assert all(not (7.992 < t < 8.992) for t, _ in held), held


# --- wave D review round 4: the reviewer's fuzz as a property test ------------------------


def _fuzz_scene(seed: int):
    import random

    from hookline.media import reframe as R

    rnd = random.Random(seed)
    dur = rnd.uniform(6, 20)
    x, xs = 0.5, []
    for _ in range(int(dur / 0.25)):
        x += rnd.gauss(0, 0.03) + (0.5 - x) * 0.05
        if rnd.random() < 0.05:
            x += rnd.choice([-1, 1]) * rnd.uniform(0.1, 0.25)
        x = min(max(x, 0.08), 0.92)
        xs.append(x)
    lay = SceneLayout(
        start=0, end=dur, kind="FACE_TRACK", track_x=[(i * 0.25, v) for i, v in enumerate(xs)],
        face_w=rnd.uniform(0.09, 0.2), face_box=(0.3, 0.3), head_top=0.15,
        lead=rnd.choice([0, 0, 0.05, -0.05]),
    )  # fmt: skip
    n = rnd.randint(0, int(dur / 2.5))
    cuts = sorted(round(rnd.uniform(0.5, dur - 0.3), 3) for _ in range(n))
    emph = [rnd.uniform(0.5, dur) for _ in range(rnd.randint(0, 5))]
    return R.plan_zoom(lay, 1920, 864), R.zoom_schedule(dur, cuts, emph), cuts


@pytest.mark.parametrize("block", range(6))
def test_guard_property_over_random_scenes(block):
    # 300 seeded scenes (6 x 50): base in effect inside every span the guard finds on the
    # final schedule; every pulse fits that schedule; the face keeps the margin (less jitter)
    # on the rendered crop and zoom, wherever the crop is free and the face guardable
    from hookline.media import reframe as R

    w, h = 1920, 864
    cw = R._track_w(w, h)
    for seed in range(block * 50, block * 50 + 50):
        lay0, sched, cuts = _fuzz_scene(seed)
        held, (lay,) = R.guard_zoom([lay0], w, h, sched, cuts)
        rel = [(t - lay.start, x) for t, x in lay.track_x]
        zoom = R.zoom_at_fn(held, lay)
        for a, b in R.track_guard_spans(rel, w, cw, lay.face_w, None, lay.lead, zoom_top=zoom):
            for k in range(int((b - a) / 0.05)):
                u = a + k * 0.05
                assert R._level_at(held, u) == 1.0, (seed, (a, b), u, held)
        fit = R._SceneFit(lay, w, h, held)
        sh = sorted(lay.shift_steps)
        for t, v in sh:
            if v:
                off = next((u for u, vv in sh if u > t and not vv), lay.end)
                assert fit.fits(None, t, min(off + R.ZOOM_EASE, lay.end), shift=v), (seed, t)
        chain = scene_filter(lay, w, h, "a", "b", zoom_steps=held, cut_times=cuts)
        xe = chain.split("crop=", 1)[1].split(":", 3)[2].replace(r"\,", ",")
        ze = zoom_expr(held, offset=0, length=lay.end, level=lay.zoom_level, instant=cuts)
        ts, xs, ws = R._track_faces(rel, lay.face_w, None)
        for t, x, fw in zip(ts, xs, ws, strict=True):
            left, z = evaluate(xe, t), evaluate(ze, t)
            vis = cw / z
            vl = left + cw / 2 - vis / 2
            free = 1 < left < w - cw - 1 and 1 < vl and vl + vis < w - 1
            if not free or not R._guardable(fw, cw / w / z * (0.5 - R.GUARD_MARGIN)):
                continue
            core = fw * R.FACE_CORE / 2  # fix wave F: the face core keeps the margin
            m = min(x * w - core * w - vl, vl + vis - x * w - core * w) / vis
            assert m >= R.GUARD_MARGIN - R.GUARD_JITTER, (seed, t, m)


# --- fix wave F: reframe snaps instead of whip pans; step shift modelled; beats punch ------

_LEAN05 = (
    [0.52] * 8 + [0.56, 0.62, 0.70, 0.77, 0.80, 0.80, 0.79, 0.74, 0.66, 0.58, 0.55] + [0.55] * 8
)
_LEAN05_W = (
    [0.13] * 8 + [0.14, 0.15, 0.16, 0.17, 0.17, 0.17, 0.16, 0.15, 0.14, 0.13, 0.13] + [0.13] * 8
)


def _per_frame_moves(expr: str, t1: float, fps: int = 30) -> list[tuple[float, float]]:
    vals = [(k / fps, evaluate(expr, k / fps)) for k in range(int(t1 * fps) + 1)]
    return [(b[0], b[1] - a[1]) for a, b in pairwise(vals)]


@pytest.mark.parametrize("in_w", [1920, 1280])
def test_track_fast_lean_snaps_instead_of_whip_panning(in_w):
    # clip 05: the lean needs ~600 src px/s; the crop either pans at <= 150 px/s (scaled with
    # the source width) or reframes in one frame, and keeps the face in the margin throughout
    kfs = [(i * 0.25, x) for i, x in enumerate(_LEAN05)]
    cw = round(in_w * 486 / 1920 / 2) * 2
    expr = track_x_expr(kfs, in_w=in_w, crop_w=cw, face_w=0.13, widths=_LEAN05_W)
    limit = 150 * in_w / 1920 / 30 + 0.5  # px per frame
    moves = _per_frame_moves(expr, kfs[-1][0])
    jumps = [(t, d) for t, d in moves if abs(d) > limit]
    assert 1 <= len(jumps) <= 3, jumps  # out and back: reframes, not a whip pan
    for (t, x), w in zip(kfs, _LEAN05_W, strict=True):
        assert _inside(evaluate(expr, t), cw, x, w * 0.8, in_w, 0.08), t  # the core


def test_track_snaps_are_reported_and_prefer_a_word_gap():
    from hookline.media.reframe import track_snaps

    kfs = [(i * 0.25, x) for i, x in enumerate(_LEAN05)]
    kw = {"in_w": 1920, "crop_w": 486, "face_w": 0.13, "widths": _LEAN05_W}
    snaps = track_snaps(kfs, **kw)
    assert snaps and all(0 < t < kfs[-1][0] for t in snaps)
    first = snaps[0]
    gap = (round(first - 0.2, 3), round(first - 0.12, 3))
    expr = track_x_expr(kfs, gaps=[gap], **kw)
    for (t, x), w in zip(kfs, _LEAN05_W, strict=True):
        assert _inside(evaluate(expr, t), 486, x, w * 0.8, 1920, 0.08), t  # the core
    # a gap further back than SNAP_GAP_S is ignored
    assert track_snaps(kfs, gaps=[(first - 1.0, first - 0.6)], **kw)[0] == first


def test_track_slow_drift_still_pans_without_snaps():
    from hookline.media.reframe import track_snaps

    kfs = [(i * 0.25, 0.4 + 0.02 * i * 0.25) for i in range(40)]  # 38 px/s
    expr = track_x_expr(kfs, in_w=1920, crop_w=486, face_w=0.1)
    assert track_snaps(kfs, in_w=1920, crop_w=486, face_w=0.1) == []
    assert all(abs(d) <= 150 / 30 + 0.5 for _, d in _per_frame_moves(expr, 9.75))
    assert evaluate(expr, 9.75) > evaluate(expr, 0) + 50  # it does follow


def test_track_single_stray_detection_never_snaps():
    from hookline.media.reframe import track_snaps

    kfs = [(i * 0.25, 0.5) for i in range(12)]
    kfs[6] = (1.5, 0.85)
    assert track_snaps(kfs, in_w=1920, crop_w=486, face_w=0.13) == []


def test_scene_snaps_are_on_the_output_timeline():
    from hookline.media.reframe import guard_zoom, scene_snaps

    lay = plan_zoom(_lean_at(3.0, 3.0 + 0.25 * len(_LEAN05), _LEAN05, fw=0.13), 1920, 864)
    lay = lay.model_copy(update={"track_w": list(_LEAN05_W)})
    held, (lay,) = guard_zoom([lay], 1920, 864, [(0.0, 1.0)], [])
    snaps = scene_snaps(lay, 1920, 864, held)
    assert snaps and all(lay.start < t < lay.end for t in snaps)


# fuzz seeds where a flat scene's step shift broke the guard margin (reviewer round 4)
_STEP_SHIFT_SEEDS = [325, 361, 428, 454, 456, 468, 551, 678, 732, 791, 854, 868, 896, 972,
                     1049, 1052, 1102, 1165, 1194, 1324]  # fmt: skip


def _check_guard_property(seed: int) -> None:
    from hookline.media import reframe as R

    w, h = 1920, 864
    cw = R._track_w(w, h)
    lay0, sched, cuts = _fuzz_scene(seed)
    held, (lay,) = R.guard_zoom([lay0], w, h, sched, cuts)
    rel = [(t - lay.start, x) for t, x in lay.track_x]
    kw = R._track_kw(lay, w, h, held)  # the crop as rendered: a flat scene's step shift too
    del kw["gaps"]
    for a, b in R.track_guard_spans(**kw):
        for k in range(int((b - a) / 0.05)):
            u = a + k * 0.05
            assert R._level_at(held, u) == 1.0, (seed, (a, b), u, held)
    # a beat never becomes a zoom-out
    for t, v in sched:
        if v == ZOOM_PUNCH and t > 0:
            before = R._level_at([st for st in held if st[0] < t - 1e-9], t)
            assert R._level_at(held, t) >= before, (seed, t, held)
    fit = R._SceneFit(lay, w, h, held)
    sh = sorted(lay.shift_steps)
    for t, v in sh:
        if v:
            off = next((u for u, vv in sh if u > t and not vv), lay.end)
            assert fit.fits(None, t, min(off + R.ZOOM_EASE, lay.end), shift=v), (seed, t)
    chain = scene_filter(lay, w, h, "a", "b", zoom_steps=held, cut_times=cuts)
    xe = chain.split("crop=", 1)[1].split(":", 3)[2].replace(r"\,", ",")
    ze = zoom_expr(held, offset=0, length=lay.end, level=lay.zoom_level, instant=cuts)
    ts, xs, ws = R._track_faces(rel, lay.face_w, None)
    for t, x, fw in zip(ts, xs, ws, strict=True):
        left, z = evaluate(xe, t), evaluate(ze, t)
        vis = cw / z
        vl = left + cw / 2 - vis / 2
        free = 1 < left < w - cw - 1 and 1 < vl and vl + vis < w - 1
        if not free or not R._guardable(fw, cw / w / z * (0.5 - R.GUARD_MARGIN)):
            continue
        core = fw * R.FACE_CORE / 2  # the face core keeps the margin
        m = min(x * w - core * w - vl, vl + vis - x * w - core * w) / vis
        assert m >= R.GUARD_MARGIN - 0.002, (seed, t, m)  # the hard constraint, rounding aside


@pytest.mark.parametrize("seed", _STEP_SHIFT_SEEDS)
def test_guard_property_holds_for_step_shift_scenes(seed):
    lay0, _, _ = _fuzz_scene(seed)
    assert lay0.zoom_level(ZOOM_PUNCH) <= lay0.zoom_level(1.0) + 0.03  # flat scenes
    _check_guard_property(seed)


@pytest.mark.parametrize("block", range(6))
def test_guard_property_beats_never_zoom_out(block):
    for seed in range(block * 50, block * 50 + 50):
        _check_guard_property(seed)


def test_zoom_schedule_beat_at_the_clip_start_punches_from_zero():
    steps = zoom_schedule(10.0, [], [0.0])
    assert steps[0] == (0.0, ZOOM_PUNCH)
    assert (1.4, 1.0) in steps


def test_cut_in_a_shrunken_placement_span_gets_a_level_change():
    # reviewer round 4, seed 89: the placement span 2.8-4.75 shrinks to 2.8-3.75 on the final
    # schedule; the cut at 3.965 then sits outside every hold and gets a framing change
    from hookline.media import reframe as R

    lay0, sched, cuts = _fuzz_scene(89)
    assert 3.965 in cuts
    held, (lay,) = R.guard_zoom([lay0], 1920, 864, sched, cuts)
    changed = R._level_at(held, 3.965) != R._level_at(
        [st for st in held if st[0] < 3.965 - 1e-9], 3.965
    )
    if changed or any(abs(t - 3.965) < 1e-6 and v for t, v in lay.shift_steps):
        return
    # no change: only because none is safe (the guard fires under a zoom-in there and no
    # crop pulse fits) -- the cut was given its attempt against the final spans

    def spans_at(steps):
        kw = R._track_kw(lay, 1920, 864, steps)
        del kw["gaps"]
        return R.track_guard_spans(**kw)

    assert R._violations(R._dedupe([*held, (3.965, ZOOM_IN)]), spans_at)
    fit = R._SceneFit(lay, 1920, 864, held)
    assert not any(fit.fits(None, 3.965, 4.665, shift=s) for s in (0.04, -0.04))


def test_beat_inside_a_hold_punches_when_the_guard_allows():
    # a lean early in the scene holds the base; by the beat (4.6 s, in the hold's tail) the
    # face is back and steady: the beat punches in (a zoom change upward), not a pulse
    from hookline.media.reframe import guard_zoom

    xs = [0.5] * 6 + [0.62, 0.72, 0.76, 0.76, 0.7, 0.6] + [0.5] * 28
    lay = plan_zoom(_lean_at(0, 10, xs, fw=0.1), 1920, 864)
    sched = zoom_schedule(10.0, [], [4.6])
    held, _ = guard_zoom([lay], 1920, 864, sched, [])
    p = next(t for t, v in sched if v == ZOOM_PUNCH)
    before = max((v for t, v in held if t < p - 1e-9), default=1.0)
    at = [v for t, v in held if abs(t - p) < 1e-6]
    assert at and at[0] > 1.0, held
    assert at[0] >= before


# --- fix wave F item 9: a framing change at least every ZOOM_MAX_GAP s ----------------------


def _framing_changes(held, lay, cuts, end):
    times = {0.0, end, *cuts, *(t for t, _ in held), *(t for t, _ in lay.shift_steps)}
    times |= set(lay.snaps)
    return sorted(t for t in times if 0 <= t <= end)


def test_zoom_schedule_fillers_prefer_the_largest_word_gap_without_punctuation():
    # a 15 s unpunctuated sentence: fillers sit at its biggest pauses near the even split
    gaps = [(3.3, 3.35), (4.0, 4.3), (7.0, 7.05), (7.6, 7.7), (11.9, 11.95)]
    steps = zoom_schedule(15.0, [], [], word_gaps=gaps)
    fillers = [t for t, _ in steps[1:]]
    assert all(b - a <= 4.5 + 1e-9 for a, b in pairwise([0.0, *fillers, 15.0]))
    assert fillers[0] == pytest.approx(4.15)  # the longer pause beats the nearer one
    assert all(any(a <= t <= b for a, b in gaps) for t in fillers)


def test_guard_keeps_a_framing_change_every_max_gap_through_a_long_hold():
    # wave E clip 03: a cut at 8.6 s, then one long unpunctuated sentence the guard holds at
    # base (she keeps leaning); the fillers inside the hold are dropped, so the guard must put
    # a framing change (level step or pulse) back at a word gap at least every ZOOM_MAX_GAP
    from hookline.media.reframe import ZOOM_MAX_GAP, guard_zoom

    xs = [0.5 + 0.1 * math.sin(i / 3.0) for i in range(4 * 24)]
    lay = plan_zoom(_lean_at(0, 24, xs, fw=0.13), 1920, 864)
    word_gaps = [(round(8.6 + k * 0.7, 2), round(8.6 + k * 0.7 + 0.08, 2)) for k in range(20)]
    lay = lay.model_copy(update={"gaps": word_gaps})
    cuts = [8.6]
    sched = zoom_schedule(24.0, cuts, [], word_gaps=word_gaps)
    held, (out,) = guard_zoom([lay], 1920, 864, sched, cuts)
    marks = _framing_changes(held, out, cuts, 24.0)
    worst = max(b - a for a, b in pairwise(marks))
    assert worst <= ZOOM_MAX_GAP + 1e-6, (marks, held, out.shift_steps)


def test_face_track_px_maps_each_sample_through_crop_and_zoom():
    from hookline.media.reframe import face_track_px

    lay = _lean_at(0, 4, [0.5] * 16, fw=0.1).model_copy(
        update={"track_face": [(0.3, 0.3, 0.42)] * 15 + [(0.5, 0.3, 0.62)]}
    )
    px = face_track_px(lay, 1920, 864, [(0.0, 1.0)])
    assert len(px) == 16
    t, x0, top, x1, bottom, eye = px[0]
    assert t == 0.0 and x0 + x1 == pytest.approx(1080, abs=2)  # centred face, centred crop
    assert (x1 - x0) == pytest.approx(0.1 * 1920 / 486 * 1080, abs=2)
    assert top == pytest.approx((0.3 + 0.2 * 0.3) * 1920, abs=1)  # the core: top 20% off
    assert bottom == pytest.approx(0.6 * 1920, abs=1) and eye == pytest.approx(0.42 * 1920, abs=1)
    zoomed = face_track_px(lay, 1920, 864, [(0.0, 1.12)])
    assert zoomed[0][3] - zoomed[0][1] > x1 - x0  # a zoom-in grows the face
    assert face_track_px(lay.model_copy(update={"track_face": []}), 1920, 864, []) == []


# --- fix wave F review round 1 ------------------------------------------------------------


def test_track_snap_moves_back_into_a_word_gap_where_the_new_framing_fits():
    # a 0.18 jump: one snap; the new framing already holds the face core a little before
    # the moment the old one stops holding it, so a word gap there takes the snap
    from hookline.media.reframe import track_snaps

    kfs = [(i * 0.25, x) for i, x in enumerate([0.5] * 8 + [0.68] * 20)]
    kw = {"in_w": 1920, "crop_w": 486, "face_w": 0.1}
    (first,) = track_snaps(kfs, **kw)
    gap = (round(first - 0.055, 3), round(first - 0.025, 3))
    (moved,) = track_snaps(kfs, gaps=[gap], **kw)
    assert moved != first and gap[0] - 1e-6 <= moved <= gap[1] + 1e-6
    expr = track_x_expr(kfs, gaps=[gap], **kw)
    for t, x in kfs:
        assert _inside(evaluate(expr, t), 486, x, 0.08, 1920, 0.08), t


def test_track_swaying_subject_pans_instead_of_snapping():
    # reviewer probe: +-0.1 sway with a 4 s period needs ~100 px/s of crop motion: a
    # speed-capped pan that anticipates it, at most one snap per 2.5 s
    from hookline.media.reframe import track_snaps

    xs = [0.5 + 0.1 * math.sin(2 * math.pi * i * 0.25 / 4.0) for i in range(120)]
    kfs = [(i * 0.25, x) for i, x in enumerate(xs)]
    snaps = track_snaps(kfs, in_w=1920, crop_w=486, face_w=0.1)
    assert len(snaps) <= 30 / 2.5
    expr = track_x_expr(kfs, in_w=1920, crop_w=486, face_w=0.1)
    for t, x in kfs:
        assert _inside(evaluate(expr, t), 486, x, 0.08, 1920, 0.08), t


def test_step_offset_models_the_eased_shift():
    from hookline.media.reframe import ZOOM_EASE, _step_offset

    lay = _FLAT.model_copy(update={"step_shift": 0.04})
    steps = [(0.0, 1.0), (2.0, ZOOM_IN)]
    shift = 0.04 * 1214 / 3840
    eased = _step_offset(lay, steps, 3840, 2160)
    assert eased(1.9) == 0.0 and eased(2.0 + ZOOM_EASE + 0.01) == pytest.approx(shift)
    assert eased(2.0 + ZOOM_EASE / 2) == pytest.approx(shift / 2, rel=1e-3)  # mid-ease
    at_cut = _step_offset(lay.model_copy(update={"cuts": [2.0]}), steps, 3840, 2160)
    assert at_cut(2.0 + ZOOM_EASE / 2) == pytest.approx(shift)  # a cut's shift is instant


def test_punch_in_returns_exactly_on_the_camera_change():
    from hookline.media.reframe import _punch_in

    calls = []

    def spans_at(steps):
        calls.append(steps)
        return []

    levels = [(0.0, 1.0)]
    out = _punch_in(levels, [(0.0, 1.0), (4.5, ZOOM_PUNCH)], [], (0.0, 5.0), spans_at)
    assert (4.5, ZOOM_PUNCH) in out
    # round 3: the return sits on the camera change (instant there, `scene_filter`), so the
    # punch never carries into the next scene; nothing else past the punch
    assert (5.0, 1.0) in out and all(t <= 5.0 for t, _ in out)


def test_face_track_px_leaves_a_missing_eye_line_unset():
    from hookline.media.reframe import face_track_px

    lay = _lean_at(0, 2, [0.5] * 8, fw=0.1).model_copy(
        update={"track_face": [(0.3, 0.3, None)] * 8}
    )
    px = face_track_px(lay, 1920, 864, [(0.0, 1.0)])
    assert px and all(row[5] is None for row in px)  # the caption falls back to the 20% core


# --- fix wave F review round 3: a punch never carries across a camera change -------------


def test_punch_in_returns_at_the_scene_end_instantly():
    from hookline.media.reframe import _level_at, _punch_in

    sched = [(0.0, 1.0), (9.0, ZOOM_IN)]
    out = _punch_in(sched, sched, cuts=[4.4], scene=(0, 5), spans_at=lambda _s: [])
    assert _level_at(out, 4.5) > 1.0  # the cut got its punch ...
    assert (5.0, 1.0) in out and _level_at(out, 6.0) == 1.0  # ... which ends at the scene end
    # the next scene starts on the returned level in its first frame: no ease across the change
    lay = SceneLayout(start=5, end=12, kind="FIT_BLUR")
    chain = scene_filter(lay, 1920, 864, "a", "b", zoom_steps=out, cut_times=[4.4])
    s = chain.split("scale=w='trunc(1080*(", 1)[1].split(")/2)*2'", 1)[0].replace("\\,", ",")
    assert evaluate(s, 0.0) == pytest.approx(1.0) and evaluate(s, 0.1) == pytest.approx(1.0)


def test_a_box_inside_the_letterbox_maps_to_nothing():
    assert box_to_crop((0.0, 0.0, 0.5, 0.05), (320, 180), (320, 140, 0, 20)) is None
    assert box_to_crop((0.0, 0.0, 0.5, 0.3), (320, 180), (320, 140, 0, 20)) is not None


def test_a_facecam_override_inside_the_letterbox_is_ignored_with_a_warning(
    monkeypatch, settings, caplog
):
    import logging

    from hookline.pipeline import render as render_mod

    seen = []
    monkeypatch.setattr(render_mod, "detect_overlay_bands", lambda inter, crop=None: None)
    monkeypatch.setattr(
        render_mod,
        "plan_layouts",
        lambda inter, det, override, crop=None: seen.append(override) or [],
    )
    s = settings.model_copy(update={"facecam": (0.0, 0.0, 0.5, 0.05)})
    with caplog.at_level(logging.WARNING, logger="hookline.pipeline.render"):
        render_mod._plan_vertical(Path("x.mp4"), None, s, (320, 180), (320, 140, 0, 20))
    assert seen == [None] and "letterbox" in caplog.text
