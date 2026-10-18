"""Stream render: full-cam covers over talked-over screens, frame-filling full-cam framings,
the near-black gameplay lift, the face-track opening check, loud-splice fades and the tail."""

import itertools
import re
import subprocess

import numpy as np
import pytest

from hookline.media.audio import Envelope
from hookline.media.reframe import (
    BLUR,
    LIFT_BELOW,
    LIFT_MAX,
    LIFT_RATE,
    MAX_UPSCALE,
    OUT_H,
    OUT_W,
    full_cam_fill,
    full_cam_geometry,
    guard_zoom,
    lift_gamma,
    lift_keyframes,
    lift_lut,
    plan_zoom,
    reframe_graph,
    scene_filter,
    stacked_face_px,
)
from hookline.models import Word
from hookline.pipeline.layout import (
    COVER_MIN_S,
    SceneLayout,
    plan_cam_beats,
    plan_cam_covers,
    stack_template,
)
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
BIG = LayoutProfile(  # a big cam (a 900 px tall window fits 9:16 within MAX_UPSCALE)
    size=(1920, 1080),
    cam=(1000, 60, 1900, 1060),
    outer=(996, 56, 1904, 1064),
    head=(1330, 180, 1570, 470),
    face=(1360, 260, 1540, 470),
    masks=[],
)


def W(start, end, text="word"):
    return Word(start=start, end=end, text=text)


def _template(start=0.0, end=10.0, profile=PROFILE, **kw) -> SceneLayout:
    t = stack_template(profile, None, profile.size)
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


def _frame(path, t, w=OUT_W, h=OUT_H):
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-ss", f"{t}", "-i", str(path),
         "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"],
        check=True, capture_output=True,
    )  # fmt: skip
    return np.frombuffer(res.stdout, np.uint8).reshape(h, w)


# --- covers over talked-over screens -----------------------------------------------------------


def test_cam_covers_span_the_screens_in_cam_scenes_only():
    lays = [
        _template(0.0, 10.0),
        SceneLayout(start=10.0, end=20.0, kind="FIT_BLUR"),  # the cam hidden: screen kept
    ]
    screens = [(3.9, 6.0), (6.2, 9.0), (12.0, 15.0), (1.0, 1.0 + COVER_MIN_S - 0.2)]
    out = plan_cam_covers(screens, lays, 20.0)
    assert out[0].cam_covers == [(3.8, 9.1)]  # joined over 0.2 s, padded; the short one not
    assert out[1].cam_covers == []
    edge = plan_cam_covers([(0.3, 3.0), (8.0, 9.7)], [_template(0.0, 10.0)], 10.0)
    assert edge[0].cam_covers == [(0.0, 3.1), (7.9, 10.0)]  # run on to the scene edges


def test_cam_beats_keep_clear_of_covers():
    lay = _template(0.0, 20.0, cam_covers=[(5.0, 9.0)])
    out = plan_cam_beats([(8.0, 10.0), (15.0, 9.0)], [lay], [], 20.0)
    beats = out[0].cam_beats
    assert beats and all(b <= 5.0 - 2.0 or a >= 9.0 + 2.0 for a, b in beats)


def test_cover_renders_the_cam_with_no_blur_and_no_gameplay(src, tmp_path):
    """The cam picture pinned to the top, a plain black band under it (no blurred cam copy,
    no screen); the head inside the frame."""
    lay = _template(0.0, 4.0, cam_covers=[(1.0, 3.0)])
    f = scene_filter(lay, 1280, 720, "a", "b")
    assert BLUR not in f.split("[b_st]")[1]  # nothing blurred in the full-cam framing
    out = _render(src, reframe_graph([lay], 1280, 720), tmp_path / "cover.mp4")
    (_, _wh, _, _), (_, ph), top = full_cam_geometry(lay, 1280, 720)
    assert top <= 0
    fr = _frame(out, 2.0)
    band = fr[int(top + ph) + 8 :, :]
    assert band.mean() < 20 and band.std() < 4  # black under the cam picture
    assert fr[: int(top + ph) - 8, :].std() > 10  # the cam picture (testsrc2: busy)


def test_beat_band_shows_the_gameplay_panel_not_a_blurred_cam(src, tmp_path):
    lay = _template(0.0, 4.0, cam_beats=[(1.0, 3.0)])
    (_, _, _, _), (_, ph), top = full_cam_geometry(lay, 1280, 720)
    assert top <= 0 and top + ph < OUT_H  # the audited cam cannot fill: a band under it
    f = scene_filter(lay, 1280, 720, "a", "b")
    assert BLUR not in f.split("[b_st]")[1]
    out = _render(src, reframe_graph([lay], 1280, 720), tmp_path / "beat.mp4")
    band = _frame(out, 2.0)[int(top + ph) + 4 :, :]
    assert band.std() > 20  # gameplay (testsrc2's pattern), not black, not a blur


def test_full_cam_fills_the_frame_when_the_cam_is_big_enough():
    lay = _template(0.0, 10.0, profile=BIG, cam_beats=[(3.0, 5.5)])
    win = full_cam_fill(lay, 1920, 1080)
    assert win is not None
    ww, wh, wx, wy = win
    assert ww / wh == pytest.approx(9 / 16, abs=0.01) and OUT_H / wh <= MAX_UPSCALE + 1e-9
    cam = BIG.cam
    assert cam[0] <= wx and wx + ww <= cam[2] and cam[1] <= wy and wy + wh <= cam[3]
    head = BIG.head
    assert wx <= head[0] and head[2] <= wx + ww  # the whole head
    chin = (head[3] - wy) / wh * OUT_H
    assert chin <= 1100 + 2 and (head[1] - wy) / wh * OUT_H >= 0
    f = scene_filter(lay, 1920, 1080, "a", "b")
    assert f"scale={OUT_W}:{OUT_H}" in f.split("[b_st]")[1] and BLUR not in f.split("[b_st]")[1]
    assert full_cam_fill(_template(), 1280, 720) is None  # the audited cam: 9.6x to fill


def test_stacked_face_px_follows_the_cover():
    lay = plan_zoom(_template(0.0, 4.0, cam_covers=[(2.0, 3.0)]), 1280, 720)
    rows = stacked_face_px(lay, 1280, 720, [(0.0, 1.0)])
    cover = [r for r in rows if 2.0 <= r[0] < 3.0]
    assert cover and all(r[3] - r[1] > 0.5 * OUT_W for r in cover)  # the large head


# --- near-black gameplay lift ------------------------------------------------------------------


def test_lift_gamma_only_for_near_black_and_gentle():
    assert lift_gamma(LIFT_BELOW) == 1.0 and lift_gamma(120.0) == 1.0
    assert 1.0 < lift_gamma(LIFT_BELOW - 8) < lift_gamma(20.0) <= LIFT_MAX <= 1.5


def test_lift_lut_keeps_black_and_white_and_raises_the_shadows():
    g = LIFT_MAX
    assert abs(lift_lut(16, g) - 16) <= 1 and abs(lift_lut(235, g) - 235) <= 3
    assert lift_lut(30, g) >= 30 + 8  # the shadows lifted
    lut = [lift_lut(v, g) for v in range(16, 236)]
    assert all(b >= a for a, b in itertools.pairwise(lut))  # monotone: no banding flips


def test_lift_keyframes_smooth_and_rate_limited_instant_at_cuts():
    # 2 samples / s: dark 0-5 s, a flicker of bright frames at 2.5 s, bright 5-10 s after a cut
    times = [k / 2 for k in range(20)]
    luma = [20.0 if t < 5 else 90.0 for t in times]
    luma[5] = 60.0
    pts = lift_keyframes(times, luma, cuts=[5.0])
    assert pts
    pre = [g for t, g in pts if t < 5.0 - 1e-6]
    post = [g for t, g in pts if t >= 5.0]
    assert min(pre) > 1.2  # the single bright frame does not drop the lift
    assert all(g == 1.0 for g in post)  # after the cut: none, at once
    for (t0, g0), (t1, g1) in itertools.pairwise(pts):
        if t1 - t0 > 0.01:
            assert abs(g1 - g0) / (t1 - t0) <= LIFT_RATE + 1e-6  # no flicker


def test_lift_renders_on_the_gameplay_panel_only(tmp_path):
    dark = tmp_path / "dark.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=1280x720:rate=25:duration=2",
         "-vf", "eq=brightness=-0.42", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dark)],
        check=True,
    )  # fmt: skip
    plain = _template(0.0, 2.0)
    lifted = plain.model_copy(update={"game_lift": [(0.0, LIFT_MAX)]})
    f = scene_filter(lifted, 1280, 720, "a", "b")
    assert re.search(r"_b\]crop=[^;]*eq=gamma=", f) and "eq=" not in f.split("[b_cam]")[0]
    a = _frame(_render(dark, reframe_graph([plain], 1280, 720), tmp_path / "a.mp4"), 1.0)
    b = _frame(_render(dark, reframe_graph([lifted], 1280, 720), tmp_path / "b.mp4"), 1.0)
    game = slice(800, 1900)
    assert b[game].mean() > a[game].mean() + 5
    assert abs(float(b[:700].mean()) - float(a[:700].mean())) < 1.0  # the cam untouched


# --- face-track opening -------------------------------------------------------------------------


def _track(x: float, lead: float = 0.0) -> SceneLayout:
    keys = [(0.125 + k * 0.25, x) for k in range(16)]
    return SceneLayout(
        start=0.0, end=4.0, kind="FACE_TRACK", track_x=keys, track_w=[0.12] * 16,
        track_face=[(0.3, 0.34, 0.45)] * 16, lead=lead, face_box=(0.3, 0.34), face_w=0.12,
        head_top=0.14,
    )  # fmt: skip


def test_opening_never_punched_in_with_the_head_cut():
    """The gaze lead leaves the back of the head against the crop edge at the opening punch:
    the clip opens at base instead (later steps kept)."""
    lay = plan_zoom(_track(0.545, lead=-0.08), 1280, 720)
    steps = [(0.0, 1.2), (1.4, 1.0), (2.5, 1.12)]
    held, _ = guard_zoom([lay], 1280, 720, steps)
    assert held[0] == (0.0, 1.0)
    assert any(t == pytest.approx(2.5) for t, _ in held)
    centred = plan_zoom(_track(0.5), 1280, 720)
    held, _ = guard_zoom([centred], 1280, 720, steps)
    assert held[0] == (0.0, 1.2)  # room enough: the opening punch stays


# --- loud splices, the end tail -----------------------------------------------------------------


def _env(spans: list[tuple[float, float, float]], length: float = 10.0) -> Envelope:
    db = np.full(round(length / 0.01), -60.0)
    for a, b, lv in spans:
        db[round(a / 0.01) : round(b / 0.01)] = lv
    return Envelope(t0=0.0, hop=0.01, db=db)


def test_splice_fades_longer_on_loud_audio():
    from hookline.media.ffmpeg import SPLICE_FADE_S
    from hookline.pipeline.render import LOUD_SPLICE_FADE_S, splice_fades

    assert 0.015 <= LOUD_SPLICE_FADE_S <= 0.02 and SPLICE_FADE_S == 0.008
    env = _env([(1.0, 3.0, -12.0), (6.0, 8.0, -40.0)])
    keep = [(0.0, 2.0), (4.0, 6.5), (7.5, 9.0)]
    assert splice_fades(keep, env) == [LOUD_SPLICE_FADE_S, SPLICE_FADE_S]  # loud before the 1st


def test_cut_segments_uses_the_given_splice_fades(monkeypatch, tmp_path):
    from hookline.media import ffmpeg

    seen = {}

    def grab(graph, script, force=False):
        seen["graph"] = graph
        return ["-filter_complex", graph]

    monkeypatch.setattr(ffmpeg, "filter_graph_args", grab)
    monkeypatch.setattr(ffmpeg, "_run", lambda cmd: None)
    monkeypatch.setattr(ffmpeg, "has_audio", lambda p: True)
    monkeypatch.setattr(ffmpeg.os, "replace", lambda a, b: None)
    ffmpeg.cut_segments(
        tmp_path / "s.mp4", [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], tmp_path / "o.mkv",
        "libx264", fps=30.0, splice_fades=[0.018, 0.008],
    )  # fmt: skip
    g = seen["graph"]
    assert g.count("d=0.018") == 2 and g.count("d=0.008") == 2


def test_tail_targets_0_35_to_0_45_s_after_the_last_word():
    from hookline.pipeline.timing import tail_end

    env = _env([(1.0, 2.02, -15.0)])
    end = tail_end(2.0, 2.0, 10.0, env, -32.0)
    assert 2.02 + 0.35 - 0.011 <= end <= 2.02 + 0.45 + 1e-9
    near = tail_end(2.0, 2.0, 2.3, _env([(1.0, 2.02, -15.0), (2.3, 2.8, -15.0)]), -32.0)
    assert near <= 2.3 - 0.05 + 1e-9  # the audible next word still stops it


# --- the audit measures what is shown -----------------------------------------------------------


def test_q7_counts_cover_edges_as_framing_changes():
    from hookline.quality import _check_pacing

    lay = {"start": 0.0, "end": 14.0, "kind": "STACKED", "cam_covers": [[4.0, 8.5]]}
    side = {"cut_times": [], "layout": [lay], "zoom_steps": [[0.0, 1.0], [12.0, 1.05]]}
    assert _check_pacing(side, 14.0).ok  # 0 -> 4 -> 8.5 -> 12
    assert not _check_pacing({**side, "layout": [{**lay, "cam_covers": []}]}, 14.0).ok


def test_q_static_skips_screen_frames_under_a_cover():
    from hookline.quality import _check_static, _stream_hidden

    flags = [False] * 8 + [True] * 24 + [False] * 8  # a 6 s screen at 2-8 s (4 fps)
    assert not _check_static(flags, 4, []).ok
    assert _check_static(flags, 4, [], hidden=[(1.9, 8.1)]).ok
    side = {"layout": [{"start": 0.0, "end": 10.0, "cam_covers": [[1.9, 8.1]],
                        "cam_beats": [[8.5, 9.5]]}]}  # fmt: skip
    assert _stream_hidden(side) == [(1.9, 8.1)]  # a beat's band still shows the gameplay


# --- covers: cuts, the hook and tag, punches on the cam ------------------------------------------


def test_cover_edges_snap_to_cuts_never_padding_across_one():
    """Stream clip 05: a screen split at a jump cut at 13.3 s was covered to 13.4 s (the full
    cam showed the new footage, then switched), and from 3.533 s, 0.33 s before a cut at
    3.867 s (two framing changes a third of a second apart)."""
    lay = _template(0.0, 20.0)
    out = plan_cam_covers([(3.633, 13.3)], [lay], 20.0, cuts=[3.867, 13.3])
    assert out[0].cam_covers == [(3.867, 13.3)]
    # a cut just inside the padding: the edge lands on it, not 0.1 s past it
    out = plan_cam_covers([(5.0, 9.0)], [lay], 20.0, cuts=[4.95, 9.04])
    assert out[0].cam_covers == [(4.95, 9.04)]
    # no cut near: padded as before
    out = plan_cam_covers([(5.0, 9.0)], [lay], 20.0, cuts=[2.0, 12.0])
    assert out[0].cam_covers == [(4.9, 9.1)]


def test_covers_keep_out_of_the_hook_and_the_earlier_tag():
    lay = _template(0.0, 20.0)
    hook, tag = (0.0, 3.0), (9.5, 10.7)
    out = plan_cam_covers([(1.0, 6.0), (10.0, 14.0)], [lay], 20.0, avoid=[hook, tag])
    assert out[0].cam_covers == [(3.0, 6.1), (10.7, 14.1)]  # each starts after the text
    # what is left after the hook is too short to cover: no cover
    out = plan_cam_covers([(1.0, 3.6)], [lay], 20.0, avoid=[hook])
    assert out[0].cam_covers == []


def test_text_windows_are_the_hook_and_the_earlier_tag():
    from hookline.captions.ass import SEAM_TAG_S, text_windows

    assert text_windows("A hook", 30.0, None) == [(0.0, 3.0)]
    assert text_windows("A hook", 30.0, 2.0) == [(0.0, 2.0), (2.0, 2.0 + SEAM_TAG_S)]
    assert text_windows("", 30.0, None) == []


def _cover_chain(f: str) -> str:
    """The cover's full-cam picture chain of a template scene filter (fills or pinned)."""
    return next(part for part in f.split(";") if part.endswith("_fc]"))


def test_covers_carry_the_punches_on_the_full_cam_picture(src, tmp_path):
    """Jump cuts under a cover get a framing change: the zoom steps play on the full-cam
    picture during a cover (a beat's picture stays still)."""
    from hookline.media.reframe import ZOOM_IN

    steps = [(0.0, 1.0), (1.5, ZOOM_IN), (2.6, 1.0)]
    covered = plan_cam_covers([(1.0, 3.0)], [_template(0.0, 4.0)], 4.0)[0]
    assert covered.cover_zoom
    f = scene_filter(covered, 1280, 720, "a", "b", zoom_steps=steps)
    assert "eval=frame" in _cover_chain(f)
    still = covered.model_copy(update={"cover_zoom": False})  # a render before it did
    assert "eval=frame" not in _cover_chain(scene_filter(still, 1280, 720, "a", "b", steps))
    beat = _template(0.0, 4.0, cam_beats=[(1.0, 3.0)])
    plain = _template(0.0, 4.0)
    n = scene_filter(plain, 1280, 720, "a", "b", steps).count("eval=frame")
    assert scene_filter(beat, 1280, 720, "a", "b", steps).count("eval=frame") == n
    big = plan_cam_covers([(1.0, 3.0)], [_template(0.0, 4.0, profile=BIG)], 4.0)[0]
    assert "eval=frame" in _cover_chain(scene_filter(big, 1920, 1080, "a", "b", steps))
    out = _render(src, reframe_graph([covered], 1280, 720, zoom_steps=steps), tmp_path / "z.mp4")
    assert _frame(out, 2.0)[:600].std() > 10  # renders: the cam picture under the cover


def test_stacked_face_px_follows_a_punch_under_a_cover():
    from hookline.media.reframe import ZOOM_PUNCH

    lay = plan_zoom(plan_cam_covers([(1.0, 4.0)], [_template(0.0, 5.0)], 5.0)[0], 1280, 720)
    flat = stacked_face_px(lay, 1280, 720, [(0.0, 1.0)])
    punched = stacked_face_px(lay, 1280, 720, [(0.0, 1.0), (2.0, ZOOM_PUNCH)])
    at = next(i for i, r in enumerate(flat) if r[0] == pytest.approx(3.0))
    assert punched[at][3] - punched[at][1] > flat[at][3] - flat[at][1] + 4  # the head grew


def test_q7_counts_only_framing_changes_the_viewer_sees():
    """Stream clip 05: zoom steps under a cover whose picture held still were counted (4.4 s
    reported, 6.03 s seen). Steps under a beat are hidden; under a cover only when the cover
    carries them (`cover_zoom`)."""
    from hookline.quality import _check_pacing

    steps = [[0.0, 1.0], [3.0, 1.12], [5.5, 1.0], [8.0, 1.12], [11.0, 1.0]]
    lay = {"start": 0.0, "end": 14.0, "kind": "STACKED", "cam_covers": [[2.0, 9.5]]}
    side = {"cut_times": [], "layout": [lay], "zoom_steps": steps}
    res = _check_pacing(side, 14.0)
    assert not res.ok and res.value == pytest.approx(7.5)  # 2.0 -> 9.5: nothing seen to change
    zoomed = {**side, "layout": [{**lay, "cover_zoom": True}]}
    assert _check_pacing(zoomed, 14.0).ok
    beat = {"start": 0.0, "end": 14.0, "kind": "STACKED", "cam_beats": [[2.5, 5.0]],
            "cover_zoom": True}  # fmt: skip
    res = _check_pacing({**side, "layout": [beat], "zoom_steps": [[0.0, 1.0], [3.0, 1.12],
                                                                  [9.0, 1.0]]}, 14.0)  # fmt: skip
    assert res.value == pytest.approx(5.0)  # 0 -> 2.5 -> 5.0 -> 9.0 -> 14.0: 3.0 hidden


def test_q_voiceless_uses_the_sources_limit_and_skips_scare_windows():
    from hookline.quality import _check_voiceless

    words = [W(0.0, 1.0), W(5.0, 6.0), W(6.2, 9.0)]
    assert _check_voiceless(words, 9.0).status == "pass"  # talk: 4 s is fine
    assert _check_voiceless(words, 9.0, limit=3.0).status == "warn"  # gameplay: 3 s
    # a scare at 2.5 s: its window [2.0, 4.0] is the payoff, not dead footage
    assert _check_voiceless(words, 9.0, limit=3.0, scares=[2.5]).status == "pass"


def test_stream_checks_read_the_recorded_voiceless_limit(monkeypatch, tmp_path):
    import hookline.quality as q

    monkeypatch.setattr(q, "screen_frame_flags", lambda mp4: ([], []))
    words = [W(0.0, 1.0), W(5.0, 6.0)]
    side = {"voiceless_limit_s": 3.0, "scare_times": []}
    res = next(r for r in q._stream_checks(tmp_path / "x.mp4", side, words, 6.0)
               if r.id == "Q-voiceless")  # fmt: skip
    assert res.status == "warn" and "3" in res.threshold
