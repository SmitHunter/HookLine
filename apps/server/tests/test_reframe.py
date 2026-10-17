import math
import subprocess

import pytest

from hookline.media.ffmpeg import probe_duration, probe_size
from hookline.media.reframe import (
    OUT_H,
    OUT_W,
    detect_crop,
    reframe_graph,
    scene_filter,
    track_x_expr,
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
    return float(eval(py, {"__builtins__": {}}, {"_if": _if, "lt": lt, "t": t}))


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


def test_reframe_graph_rejects_empty_and_drops_tiny_scenes():
    with pytest.raises(ValueError, match="no layouts"):
        reframe_graph([], 1280, 720)
    with pytest.raises(ValueError, match="no layouts"):
        reframe_graph([SceneLayout(start=1, end=1.01, kind="FIT_BLUR")], 1280, 720)
    graph = reframe_graph(
        [
            SceneLayout(start=0, end=2, kind="FIT_BLUR"),
            SceneLayout(start=2, end=2.02, kind="FIT_BLUR"),
            SceneLayout(start=2.02, end=4, kind="FIT_BLUR"),
        ],
        1280,
        720,
    )
    assert "split=2[" in graph and "concat=n=2:" in graph


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
