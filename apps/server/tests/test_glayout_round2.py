"""Wave G-layout review round 2: the reviewer's adversarial cases as regressions -- startles
need a game transient, borderless cams stop at a static HUD, a cam picture must move, cams
hidden for a third of the VOD still profile, sit-back Just Chatting, bounded game punches, the
seam hook with a large head, a memory-bounded snap, the coarse-cam fallback and the true-peak
trim from the pre-AAC master."""

import re
import tracemalloc

import numpy as np
import pytest

from hookline.media.audio import Envelope
from hookline.models import Word
from hookline.pipeline import profile as P
from hookline.pipeline.layout import SceneLayout, plan_cam_beats, stack_template
from hookline.pipeline.profile import LayoutProfile
from hookline.vision.faces import Face
from tests.test_profile_matrix import _detector_for, _mock_video, _scene

W, H, N = 1280, 720, 40


def _px(cam):
    return (
        round(cam[0] * W), round(cam[1] * H),
        round((cam[0] + cam[2]) * W), round((cam[1] + cam[3]) * H),
    )  # fmt: skip


def _faces(face, n, hidden=()):
    fn = (face[0] / W, face[1] / H, face[2] / W, face[3] / H)
    return [[] if i in hidden else [Face(*fn, 0.9, 0.0, None)] for i in range(n)]


# --- 1. startles need a game transient; the peak's own word never blocks its beat ---------


def _startle_env():
    db = np.full(3000, -40.0)
    db[500:540] = -15.0  # a sudden "Oh!" at 5.0 s out of quiet
    db[1200:1240] = -15.0  # ... and a post-pause "Hey" at 12.0 s
    return Envelope(t0=0.0, hop=0.01, db=db)


def test_startle_needs_a_game_transient_or_spike_at_the_onset():
    from hookline.pipeline.render import startle_times

    env = _startle_env()
    words = [Word(start=5.0, end=5.4, text="Oh!"), Word(start=12.0, end=12.4, text="Hey")]
    keep = [(0.0, 30.0)]
    peaks = [5.0, 12.0]
    # a sharp onset alone (any word after a pause) is no startle
    assert startle_times(words, env, keep, peaks) == []
    assert startle_times(words, env, keep, peaks, scares=[], spikes=[]) == []
    # a game transient within 0.3 s of the onset: that word is startled
    assert startle_times(words, env, keep, peaks, scares=[5.2]) == [5.0]
    assert startle_times(words, env, keep, peaks, scares=[5.5]) == []  # 0.5 s off
    # ... or a flash / motion spike there
    assert startle_times(words, env, keep, peaks, spikes=[(11.5, 11.8)]) == [12.0]
    assert startle_times(words, env, keep, peaks, spikes=[(13.0, 14.0)]) == []


def _stacked(start, end, **kw):
    prof = LayoutProfile(
        size=(W, H), cam=(46, 362, 247, 562), outer=(41, 360, 250, 568),
        head=(105, 390, 208, 521), face=(122, 433, 191, 521), masks=[(41, 568, 250, 597)],
    )  # fmt: skip
    return SceneLayout(
        start=start, end=end, kind="STACKED", **{**stack_template(prof, None, (W, H)), **kw}
    )


def test_beat_starts_before_its_reaction_word_despite_its_own_startle():
    """Real stream wavegl-3126: every beat started exactly 1.0 s after its reaction word (the
    word's own startle span). The peak's own startle never blocks it: BEAT_LEAD before."""
    from hookline.pipeline.layout import BEAT_LEAD

    lays = [_stacked(0.0, 40.0)]
    (out,) = plan_cam_beats([(16.37, 9.0)], lays, cuts=[], duration=40.0, startles=[16.37])
    ((a, b),) = out.cam_beats
    assert a == pytest.approx(16.37 - BEAT_LEAD, abs=1e-3) and b - a == pytest.approx(2.5)
    # another (startled) word's span still keeps the beat off it
    (out,) = plan_cam_beats([(16.37, 9.0)], lays, cuts=[], duration=40.0, startles=[16.37, 16.8])
    ((a, _),) = out.cam_beats
    assert a == pytest.approx(16.8 + 1.0, abs=1e-3)
    # scares still apply to the peak's own beat
    (out,) = plan_cam_beats(
        [(16.37, 9.0)], lays, cuts=[], duration=40.0, avoid=[(16.0, 17.2)], startles=[16.37]
    )
    ((a, _),) = out.cam_beats
    assert a == pytest.approx(17.2, abs=1e-3)
    # avoid spans as render builds them (lists) mix with the startle spans
    (out,) = plan_cam_beats(
        [(16.37, 9.0)], lays, cuts=[], duration=40.0, avoid=[[16.0, 17.2]], startles=[16.8]
    )
    ((a, _),) = out.cam_beats
    assert a == pytest.approx(17.8, abs=1e-3)


# --- 2. a borderless cam stops at a static textured HUD (the HUD becomes a mask) -----------


def _hud_scene(gap):
    cam = (0.0, 0.40, 284 / W, 160 / H)
    g, face = _scene((W, H), cam, border=False, smooth_room=True, n=N)
    o = _px(cam)
    rng = np.random.default_rng(1)
    if gap:
        g[:, o[3] + gap : o[3] + gap + 110, 10 : o[2] - 10] = rng.integers(
            30, 220, size=(110, o[2] - o[0] - 20)
        ).astype(np.uint8)
    else:
        g[:, o[3] : o[3] + 110, 0 : o[2]] = rng.integers(30, 220, size=(110, o[2])).astype(np.uint8)
    return g, face, o


@pytest.mark.parametrize("gap", [0, 4])
def test_borderless_cam_stops_at_a_static_hud_below_it(gap):
    g, face, o = _hud_scene(gap)
    prof = P._find_cam(g, _faces(face, N))
    assert prof is not None, "a HUD next to a borderless cam must not cost the profile"
    assert prof.outer == pytest.approx(o, abs=3)  # (0, 288, 284, 448): not down to 558
    # the HUD is kept out of the gameplay panel like a label
    keep = prof.keep_out()
    assert any(m[1] <= o[3] + gap + 4 and m[3] >= o[3] + gap + 100 for m in keep[1:]), keep


# --- 3. the cam picture must move (a static in-game portrait is no facecam) -----------------


def test_a_static_bordered_portrait_is_not_a_facecam():
    cam = (0.02, 0.80, 100 / W, 120 / H)
    g, face = _scene((W, H), cam, border=True, n=N)
    o = _px(cam)
    portrait = np.random.default_rng(9).integers(40, 200, size=(o[3] - o[1] - 6, o[2] - o[0] - 6))
    g[:, o[1] + 3 : o[3] - 3, o[0] + 3 : o[2] - 3] = portrait.astype(np.uint8)
    assert P._find_cam(g, _faces(face, N)) is None


def test_a_pngtuber_toggling_two_images_is_a_facecam():
    """Periodic motion (a PNGtuber's mouth open / closed) counts as a live cam."""
    cam = (0.04, 0.55, 200 / W, 150 / H)
    g, face = _scene((W, H), cam, border=True, n=N)
    o = _px(cam)
    rng = np.random.default_rng(4)
    inside = (slice(o[1] + 3, o[3] - 3), slice(o[0] + 3, o[2] - 3))
    still = rng.integers(40, 200, size=(o[3] - o[1] - 6, o[2] - o[0] - 6)).astype(np.uint8)
    mouth = still.copy()
    fx, fy, fw, fh = (int(v) for v in face)
    my, mx = fy + int(0.7 * fh) - inside[0].start, fx + int(0.3 * fw) - inside[1].start
    mouth[my : my + int(0.2 * fh), mx : mx + int(0.4 * fw)] = 10
    for i in range(N):
        g[i][inside] = mouth if (i // 3) % 2 else still
    prof = P._find_cam(g, _faces(face, N))
    assert prof is not None and prof.outer == pytest.approx(o, abs=3)


# --- 4. cams hidden for a third of the VOD still profile ------------------------------------


@pytest.mark.parametrize("hide", [0.35, 0.4])
@pytest.mark.parametrize("border", [True, False])
def test_cam_hidden_in_a_third_of_the_vod_still_profiles(hide, border):
    if border:
        cam = (0.04, 0.55, 200 / W, 150 / H)
    else:
        cam = (0.0, 1 - 160 / H, 284 / W, 160 / H)
    g, face = _scene((W, H), cam, border=border, smooth_room=not border, n=N)
    plain, _ = _scene((W, H), (0.6, 0.0, 0.01, 0.01), border=False, n=N, seed=3)
    idx = set(np.random.default_rng(2).permutation(N)[: round(hide * N)].tolist())
    for i in idx:
        g[i] = plain[i]
    prof = P._find_cam(g, _faces(face, N, hidden=idx))
    assert prof is not None
    assert prof.outer == pytest.approx(_px(cam), abs=3)
    # the ring reference is the cam's, not a blend with the hidden frames
    shown = [P.ring_visible(g[i], prof.ring) for i in range(N) if i not in idx]
    assert sum(shown) >= 0.9 * len(shown)


# --- 5. sit-back Just Chatting: a 15-19% face is the streamer full frame --------------------


@pytest.mark.parametrize("fh", [0.15, 0.19])
def test_centred_webcam_with_a_small_face_tracks_the_streamer(fh):
    from hookline.pipeline.layout import hidden_scene_layout

    samples = [
        (t * 0.25, [Face(0.45, 0.3 + 0.002 * (t % 5), fh * 9 / 16 * 1.2, fh, 0.9, 0.0, 0.3 + fh / 3)])
        for t in range(240)
    ]  # fmt: skip
    assert hidden_scene_layout(samples, 0.0, 60.0, src_h=720).kind == "FACE_TRACK"


# --- 7. game-panel punches: within MAX_UPSCALE, never cropping a static edge overlay --------


def test_game_punch_never_passes_max_upscale_on_a_480p_source():
    from hookline.media.reframe import MAX_UPSCALE, plan_zoom

    size = (854, 480)
    k = 480 / 720
    prof = LayoutProfile(
        size=size,
        cam=tuple(round(v * k) for v in (46, 362, 247, 562)),
        outer=tuple(round(v * k) for v in (41, 360, 250, 568)),
        head=tuple(round(v * k) for v in (105, 390, 208, 521)),
        face=tuple(round(v * k) for v in (122, 433, 191, 521)),
    )  # fmt: skip
    t = stack_template(prof, None, size)
    lay = plan_zoom(SceneLayout(start=0.0, end=4.0, kind="STACKED", **t), *size)
    gw = t["game_box"][2] * size[0]
    # 1080 / 447 = 2.42x already: GAME_ZOOM_MIN (1.14) would make it 2.75x -- no game punch
    assert 1080 / gw > MAX_UPSCALE / 1.14
    assert lay.zoom_panel == "speaker"
    # a 720p source of the same layout still punches the gameplay, within MAX_UPSCALE
    lay = plan_zoom(_stacked(0.0, 4.0), W, H)
    gw = lay.game_box[2] * W
    assert lay.zoom_panel == "game" and 1080 / gw * lay.zoom_level(1.2) <= MAX_UPSCALE + 1e-6


def test_game_punch_keeps_a_static_overlay_at_the_panel_edge_whole():
    """Real clip 01: the game's "VIDRES" overlay at the gameplay panel's bottom right nearly
    cut by the 1.2 punch. A punch is limited so a static overlay stays inside the zoomed panel
    (with a margin); one right at the edge leaves the punches on the cam."""
    from hookline.media.reframe import plan_zoom

    base = _stacked(0.0, 4.0)
    gx, gy, gw, gh = base.game_box
    plain = plan_zoom(base, W, H)
    assert plain.zoom_panel == "game" and plain.zoom_level(1.2) == pytest.approx(1.2, abs=0.01)
    # an overlay ending 9% of the panel width from its right edge (VIDRES: 0.903)
    vid = (gx + 0.70 * gw, gy + 0.80 * gh, 0.21 * gw, 0.06 * gh)
    lay = plan_zoom(base.model_copy(update={"game_static": [vid]}), W, H)
    assert lay.zoom_panel == "game"
    z = lay.zoom_level(1.2)
    assert 1.08 <= z < 1.2
    assert 0.5 + 0.5 / z >= 0.91 + 0.015  # its right edge inside the zoomed panel, a margin
    # right at the edge: no visible punch would keep it whole -- the gameplay doesn't zoom
    edge = (gx + 0.80 * gw, gy + 0.80 * gh, 0.19 * gw, 0.06 * gh)
    lay = plan_zoom(base.model_copy(update={"game_static": [edge]}), W, H)
    assert lay.zoom_panel == "speaker"


def test_profile_scene_finds_a_static_overlay_in_the_gameplay_panel(monkeypatch):
    from hookline.pipeline import layout as layout_mod
    from hookline.pipeline.layout import PILLAR_W, classify_profile_scene, game_overlays

    prof = LayoutProfile(
        size=(W, H), cam=(46, 362, 247, 562), outer=(41, 360, 250, 568),
        head=(105, 390, 208, 521), face=(122, 433, 191, 521), masks=[(41, 568, 250, 597)],
    )  # fmt: skip
    ph = PILLAR_W * H // W
    n = 12
    rng = np.random.default_rng(3)
    wide = rng.integers(20, 200, size=(n, ph, PILLAR_W)).astype(np.uint8)  # moving gameplay
    text = rng.integers(0, 2, size=(8, 30)).astype(np.uint8) * 230  # a static "VIDRES"
    wide[:, 158:166, 200:230] = text
    tpl = stack_template(prof, None, (W, H))
    (box,) = game_overlays(wide, tpl["game_box"], [])
    bx, by, bw, _ = box
    assert bx == pytest.approx(200 / PILLAR_W, abs=0.02) and by == pytest.approx(158 / ph, abs=0.03)
    assert bx + bw == pytest.approx(230 / PILLAR_W, abs=0.02)
    # moving gameplay alone has none
    plain = rng.integers(20, 200, size=(n, ph, PILLAR_W)).astype(np.uint8)
    assert game_overlays(plain, tpl["game_box"], []) == []
    # a profiled scene records them for the punch planner
    monkeypatch.setattr(layout_mod, "ring_visible", lambda *a, **k: True)
    ring = np.zeros((n, 10, 10), np.uint8)
    lay = classify_profile_scene(0.0, 10.0, ring, (0, 0), wide, prof, None)
    assert lay.kind == "STACKED" and lay.game_static == [box]


def test_game_punch_keeps_the_cam_base_zoom():
    from hookline.media.reframe import plan_zoom, scene_filter, stacked_face_px

    lay = plan_zoom(_stacked(0.0, 4.0), W, H)
    assert lay.zoom_panel == "game" and lay.cam_zoom == pytest.approx(1.0)
    held = lay.model_copy(update={"cam_zoom": 1.1})
    rows0 = stacked_face_px(lay, W, H, [(0.0, 1.2)])
    rows1 = stacked_face_px(held, W, H, [(0.0, 1.2)])
    assert (rows1[0][3] - rows1[0][1]) > (rows0[0][3] - rows0[0][1])  # the head shown larger
    f = scene_filter(held, W, H, "a", "b", zoom_steps=[(0.0, 1.0), (1.0, 1.2)])
    cam_chain = f.split("[b_cam]")[0].split(";")[-1]
    assert "scale=1188:836" in cam_chain and "eval=frame" not in cam_chain  # a static zoom


def test_scene_filter_docstring_names_the_gameplay_panel():
    from hookline.media.reframe import scene_filter

    assert "gameplay panel" in (scene_filter.__doc__ or "")


# --- 8. the seam hook with a large head; one y per caption page -----------------------------


def _chin_layout(chin):
    rows = [[t / 4, 300.0, 30.0, 780.0, float(chin), None] for t in range(13)]
    return SceneLayout(start=0, end=3, kind="STACKED", face_px=rows)


def test_seam_hook_straddles_the_seam_or_sits_just_below_it():
    from hookline.captions.ass import (
        CAPTION_ABOVE,
        CAPTION_FACE_GAP,
        HOOK_HALF,
        TEXT_SEPARATION,
        caption_y_for,
        hook_y_for,
    )
    from hookline.pipeline.layout import STACK_CAM_H

    # a chin leaving room: the box straddles the seam, clear of the chin
    hy = hook_y_for([_chin_layout(600)])
    assert (
        hy - HOOK_HALF >= 600 + CAPTION_FACE_GAP and hy - HOOK_HALF < STACK_CAM_H < hy + HOOK_HALF
    )
    # a large head (chin 732, real stream): no room to straddle -- just below the seam, not
    # pushed further down over the gameplay (928 before)
    hy = hook_y_for([_chin_layout(732)])
    assert hy - HOOK_HALF == STACK_CAM_H
    assert caption_y_for([_chin_layout(732)]) - CAPTION_ABOVE == hy + HOOK_HALF + TEXT_SEPARATION


def test_a_caption_page_keeps_one_y_when_the_hook_ends_under_it():
    from hookline.captions.ass import build_ass, caption_y_for, hook_y_for

    lay = _chin_layout(732)
    words = [
        Word(start=0.2, end=0.8, text="this"), Word(start=2.6, end=2.9, text="and"),
        Word(start=2.9, end=3.3, text="that"), Word(start=3.3, end=3.6, text="is"),
        Word(start=6.0, end=6.4, text="later"),
    ]  # fmt: skip
    ass = build_ass(
        words, "A hook title here", 8.0, caption_y_for([lay]), hook_y_for([lay]),
        caption_y_late=caption_y_for([lay], late=True),
    )  # fmt: skip
    caps = [ln for ln in ass.splitlines() if ",Cap," in ln]
    pages: dict[str, set[int]] = {}
    for ln in caps:
        text = re.sub(r"\{[^}]*\}", "", ln.split(",,")[-1])
        pages.setdefault(text, set()).add(int(re.search(r"\\pos\(\d+,(\d+)\)", ln).group(1)))
    assert any("AND" in t for t in pages)
    assert all(len(ys) == 1 for ys in pages.values()), pages


# --- 9. the snap is memory-bounded ----------------------------------------------------------


def test_find_cam_memory_stays_bounded_on_150_samples():
    cam = (0.0, 1 - 160 / H, 284 / W, 160 / H)  # borderless: every reach, both routes
    g, face = _scene((W, H), cam, border=False, smooth_room=True, n=150)
    tracemalloc.start()
    try:
        prof = P._find_cam(g, _faces(face, 150))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert prof is not None
    assert peak < 0.6 * g.nbytes, f"peak {peak / 1e6:.0f} MB for {g.nbytes / 1e6:.0f} MB of samples"


# --- 10. a failed full-size refine falls back to the scaled coarse cam ----------------------


def test_refine_failure_falls_back_to_the_scaled_coarse_cam(monkeypatch):
    from hookline.media.ffmpeg import FFmpegError
    from tests.test_profile_matrix import _frames_fn

    monkeypatch.setattr(P, "PROFILE_SAMPLES", 24)
    size = (1920, 1080)
    cam = (0.04, 0.55, 0.16, 0.28)
    _, face = _scene(size, cam, n=1)
    fn = _frames_fn(size, cam)

    def frames(width, crop, n):
        if crop is not None:
            raise FFmpegError("crop read failed")
        return fn(width, crop, n)

    _mock_video(monkeypatch, size, frames)
    norm = (face[0] / size[0], face[1] / size[1], face[2] / size[0], face[3] / size[1])
    prof = P.build_profile(None, _detector_for(norm))  # type: ignore[arg-type]
    assert prof.cam is not None and prof.size == size
    want = (cam[0] * 1920, cam[1] * 1080, (cam[0] + cam[2]) * 1920, (cam[1] + cam[3]) * 1080)
    assert prof.outer == pytest.approx(want, abs=5)
    assert prof.ring and all(0 <= x < 1920 and 0 <= y < 1080 for x, y, _ in prof.ring)


def test_profile_version_bumped():
    assert P.PROFILE_VERSION >= 4


# --- 11. the true-peak trim re-encodes from the pre-AAC master ------------------------------


def test_true_peak_trims_from_the_master_not_the_aac(steady_quiet_video, tmp_path, monkeypatch):
    from hookline.media import loudness as loud_mod
    from tests.test_loudness import _m

    copy = tmp_path / "c.mp4"
    copy.write_bytes(steady_quiet_video.read_bytes())
    master = tmp_path / "master.wav"
    loud_mod._run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(steady_quiet_video),
         "-vn", "-c:a", "pcm_s24le", str(master)]
    )  # fmt: skip
    built = []

    def build():
        built.append(master)
        return master

    tps = iter([-0.4, -1.0, -1.0, -1.0, -1.6])
    monkeypatch.setattr(loud_mod, "measure_loudness", lambda p: _m(-14.0, tp=next(tps)))
    cmds = []
    real = loud_mod._run
    monkeypatch.setattr(loud_mod, "_run", lambda cmd: (cmds.append(cmd), real(cmd))[1])
    trim = loud_mod.ensure_true_peak(copy, master=build)
    assert built == [master]  # built once, lazily
    assert len(cmds) == 4
    for cmd in cmds:  # every re-encode takes its audio from the master
        assert str(master) in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    # the second trim is the whole correction applied to the master once, not a trim of AAC
    assert "volume=-1.10dB" in " ".join(cmds[2]) and "volume=-1.60dB" in " ".join(cmds[3])
    assert trim == pytest.approx(-1.6)


def test_render_hands_the_true_peak_check_a_pcm_master(
    quiet_speech_video, sample_pick, settings, tmp_path, final_argvs, monkeypatch
):
    from hookline.media.ffmpeg import _run, probe_duration
    from hookline.pipeline import render as render_mod
    from tests.test_loudness import _render

    seen = []

    def fake(dst, master=None):
        assert master is not None
        path = master()
        codec = _run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", str(path)]
        ).stdout.strip()  # fmt: skip
        seen.append((codec, probe_duration(path), probe_duration(dst)))

    monkeypatch.setattr(render_mod, "ensure_true_peak", fake)
    _render(quiet_speech_video, tmp_path, sample_pick, settings, final_argvs)
    ((codec, master_s, clip_s),) = seen
    assert codec.startswith("pcm_") and master_s == pytest.approx(clip_s, abs=0.1)
