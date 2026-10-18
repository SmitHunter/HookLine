"""The fixed STACKED template from a VOD's stream profile."""

import numpy as np
import pytest

from hookline.pipeline.layout import (
    STACK_CAM_H,
    STACK_GAME_H,
    SceneLayout,
    classify_profile_scene,
    merge_same,
    pillarbox,
    plan_cam_beats,
    stack_template,
)
from hookline.pipeline.profile import LayoutProfile, ring_reference

SIZE = (1280, 720)
# the audited stream: a square cam left of centre, its name label under it
PROFILE = LayoutProfile(
    size=SIZE,
    cam=(46, 362, 247, 562),
    outer=(41, 360, 250, 568),
    head=(105, 390, 208, 521),
    face=(122, 433, 191, 521),
    masks=[(41, 568, 250, 597)],
    share=0.92,
)


def _px(box, crop=None, size=SIZE):
    cw, ch, cx, cy = crop or (size[0], size[1], 0, 0)
    x, y, w, h = box
    return x * cw + cx, y * ch + cy, w * cw, h * ch


def test_stack_panels_fill_the_frame():
    assert STACK_CAM_H + STACK_GAME_H == 1920
    assert 740 <= STACK_CAM_H <= 820


def test_stack_template_cam_panel_shows_the_whole_cam_width_and_the_head():
    t = stack_template(PROFILE, None, SIZE)
    x, y, w, h = _px(t["facecam"])
    assert w / h == pytest.approx(1080 / STACK_CAM_H, rel=0.01)
    assert x == pytest.approx(46, abs=1) and w == pytest.approx(201, abs=1)  # whole width
    assert 362 - 0.5 <= y and y + h <= 562 + 0.5  # inside the cam: no border lines
    assert y <= 390 and y + h >= 521  # top of the head to the chin
    assert t["cam_rect"] == pytest.approx((46 / 1280, 362 / 720, 201 / 1280, 200 / 720))
    _, hy, _, hh = _px(t["head_box"])
    assert (hy, hy + hh) == pytest.approx((390, 521), abs=1)


def test_stack_template_gameplay_full_height_centred_without_the_cam():
    t = stack_template(PROFILE, None, SIZE)
    x, y, w, h = _px(t["game_box"])
    assert (y, h) == pytest.approx((0, 720), abs=1)
    assert w / h == pytest.approx(1080 / STACK_GAME_H, rel=0.01)
    assert x + w / 2 == pytest.approx(640, abs=1)  # the crosshair stays centred
    assert t["patches"] == [] and t["game_fit"] is False


def test_stack_template_shifts_gameplay_off_a_central_cam():
    prof = PROFILE.model_copy(
        update={"cam": (905, 40, 1095, 200), "outer": (900, 35, 1100, 205),
                "head": (950, 60, 1050, 190), "masks": []}
    )  # fmt: skip
    t = stack_template(prof, None, SIZE)
    x, _, w, _ = _px(t["game_box"])
    assert x + w <= 898 and x + w / 2 > 560  # moved just clear of the cam
    assert t["patches"] == []


def test_stack_template_blurs_a_cam_it_cannot_avoid():
    prof = PROFILE.model_copy(
        update={"cam": (500, 300, 780, 560), "outer": (300, 295, 980, 565),
                "head": (560, 320, 720, 540), "masks": []}
    )  # fmt: skip
    t = stack_template(prof, None, SIZE)
    x, _, w, _ = _px(t["game_box"])
    assert x + w / 2 == pytest.approx(640, abs=1)
    ((px, py, pw, ph),) = t["patches"]
    assert _px((px, py, pw, ph))[1] == pytest.approx(295 - 2, abs=1)


def test_stack_template_fit_trims_an_edge_overlay_off_the_pillarboxed_picture():
    fit = (150 / 1280, 0.0, 980 / 1280, 1.0)
    t = stack_template(PROFILE, None, SIZE, fit=fit)
    assert t["game_fit"] is True and t["patches"] == []
    # the cam (outer x 41-250, grown 2 px) reaches 102 px into the picture's left edge: <= 15%
    # of it, so the picture starts after it instead of carrying a blurred cam-shaped patch
    x, y, w, h = _px(t["game_box"])
    assert x == pytest.approx(252, abs=1) and x + w == pytest.approx(1130, abs=1)
    assert (y, h) == pytest.approx((0, 720), abs=1)


def test_stack_template_fit_blurs_an_overlay_it_cannot_trim():
    fit = (150 / 1280, 0.0, 980 / 1280, 1.0)
    prof = PROFILE.model_copy(update={"outer": (41, 360, 450, 568), "masks": []})
    t = stack_template(prof, None, SIZE, fit=fit)
    assert t["game_box"] == pytest.approx(fit)  # 302 px of 980 is too much to trim
    ((px, _, pw, _),) = t["patches"]
    assert _px((px, 0, pw, 1))[0] == pytest.approx(150, abs=1)


def test_stack_template_maps_into_a_crop():
    crop = (1280, 680, 0, 20)
    t = stack_template(PROFILE, crop, SIZE)
    x, y, _, h = _px(t["facecam"], crop)
    assert x == pytest.approx(46, abs=1) and y >= 362 - 0.5 and y + h <= 562 + 0.5


def test_pillarbox_finds_symmetric_black_bars_outside_the_masks():
    frames = np.zeros((4, 180, 320), np.uint8)
    frames[:, :, 38:282] = 80
    frames[:, 90:140, 10:62] = 200  # the cam, sitting over the left bar
    keep_out = [(10 / 320, 0.5, 52 / 320, 50 / 180)]
    assert pillarbox(frames, keep_out) == pytest.approx((38 / 320, 0, 244 / 320, 1))
    assert pillarbox(frames, []) is None  # the cam counts as content: not symmetric
    full = np.full((4, 180, 320), 80, np.uint8)
    assert pillarbox(full, keep_out) is None
    dark_left = np.full((4, 180, 320), 80, np.uint8)
    dark_left[:, :, :60] = 0  # a dark area on one side only is not a pillarbox
    assert pillarbox(dark_left, keep_out) is None


def _ring_frames(visible: bool, n: int = 4):
    med = np.full((720, 1280), 30, np.uint8)
    x0, y0, x1, y1 = PROFILE.outer
    med[y0 : y0 + 2, x0:x1] = med[y1 - 2 : y1, x0:x1] = 250
    med[y0:y1, x0 : x0 + 2] = med[y0:y1, x1 - 2 : x1] = 250
    ring = ring_reference(med, PROFILE.outer)
    frames = np.repeat(med[None], n, axis=0) if visible else np.full((n, 720, 1280), 90, np.uint8)
    return ring, frames


def test_classify_profile_scene_template_when_the_cam_border_shows():
    ring, frames = _ring_frames(True)
    prof = PROFILE.model_copy(update={"ring": ring})
    wide = np.full((4, 180, 320), 80, np.uint8)
    lay = classify_profile_scene(0.0, 5.0, frames, (0, 0), wide, prof, None)
    assert lay.kind == "STACKED" and lay.facecam is not None and lay.game_box is not None
    assert lay.cam_rect is not None and lay.head_box is not None and not lay.game_fit
    assert lay.face_box is not None and lay.head_top == pytest.approx(390 / 720, abs=0.002)


def test_classify_profile_scene_fit_blur_when_the_cam_is_hidden():
    ring, frames = _ring_frames(False)
    prof = PROFILE.model_copy(update={"ring": ring})
    wide = np.full((4, 180, 320), 80, np.uint8)
    lay = classify_profile_scene(0.0, 5.0, frames, (0, 0), wide, prof, None)
    assert lay.kind == "FIT_BLUR" and lay.facecam is None  # never FACE_TRACK


def test_classify_profile_scene_fits_a_pillarboxed_scene():
    ring, frames = _ring_frames(True)
    prof = PROFILE.model_copy(update={"ring": ring})
    wide = np.zeros((4, 180, 320), np.uint8)
    wide[:, :, 38:282] = 80
    wide[:, 90:142, 10:63] = 200  # the cam (and label) over the left bar
    lay = classify_profile_scene(0.0, 5.0, frames, (0, 0), wide, prof, None)
    assert lay.kind == "STACKED" and lay.game_fit
    assert lay.fit_src[0] == pytest.approx(38 / 320, abs=0.01)
    assert lay.game_box[0] >= lay.fit_src[0]  # the cam at its edge trimmed off


def test_merge_same_joins_identical_neighbours_only():
    a = SceneLayout(start=0, end=2, kind="STACKED", facecam=(0.1, 0.5, 0.1, 0.1))
    b = a.model_copy(update={"start": 2, "end": 5})
    c = SceneLayout(start=5, end=7, kind="FIT_BLUR")
    d = a.model_copy(update={"start": 7, "end": 9})
    merged = merge_same([a, b, c, d])
    assert [(m.start, m.end, m.kind) for m in merged] == [
        (0, 5, "STACKED"), (5, 7, "FIT_BLUR"), (7, 9, "STACKED")
    ]  # fmt: skip


def _stacked(start, end, **kw):
    t = stack_template(PROFILE, None, SIZE)
    return SceneLayout(start=start, end=end, kind="STACKED", **{**t, **kw})


def test_plan_cam_beats_picks_the_strongest_peaks_with_holds_and_gaps():
    lays = [_stacked(0.0, 40.0)]
    peaks = [(10.0, 9.0), (11.0, 12.0), (20.0, 8.0), (30.0, 7.0), (1.0, 20.0), (39.0, 20.0)]
    (out,) = plan_cam_beats(peaks, lays, cuts=[], duration=40.0)
    beats = out.cam_beats
    assert len(beats) == 2  # at most 2 per clip
    for a, b in beats:
        assert 2.5 - 1e-6 <= b - a <= 3.0
        assert a >= 3.0 and b <= 38.0  # never under the hook, never at the very end
    assert beats[0][0] < 11.0 < beats[0][1]  # the strongest usable peak, its onset inside
    (_, b0), (a1, _) = sorted(beats)
    assert a1 - b0 >= 2.0  # >= 2 s between switches


def test_plan_cam_beats_skip_cuts_hidden_cams_and_fit_scenes():
    lays = [_stacked(0.0, 12.0), SceneLayout(start=12.0, end=20.0, kind="FIT_BLUR"),
            _stacked(20.0, 40.0, game_fit=True)]  # fmt: skip
    peaks = [(8.0, 10.0), (15.0, 10.0), (30.0, 10.0)]
    out = plan_cam_beats(peaks, lays, cuts=[8.5], duration=40.0)
    assert all(not lay.cam_beats for lay in out)
    assert plan_cam_beats([], lays, cuts=[], duration=40.0) == lays


def test_classify_profile_scene_cam_shown_over_a_white_flash():
    """A white flash behind the cam blends its border's edge pixels away (the real stream:
    the outer lines went from ~85 to ~229): the cam's own picture just inside the border
    (`ring_in`) still shows it, so the template holds (no FIT_BLUR flicker)."""
    ox0, oy0, ox1, oy1 = PROFILE.outer
    x0, y0, x1, y1 = PROFILE.cam

    def frame(bg: int, blend: int) -> np.ndarray:
        f = np.full((720, 1280), bg, np.uint8)
        f[oy0:oy1, ox0:ox1] = blend  # the border's anti-aliased outer pixels
        f[oy0 + 1 : oy1 - 1, ox0 + 1 : ox1 - 1] = 250  # the white border
        f[y0:y1, x0:x1] = 120  # the cam's picture
        return f

    med = frame(30, 130)
    ring = ring_reference(med, PROFILE.outer)
    ring_in = ring_reference(med, (x0 + 3, y0 + 3, x1 - 3, y1 - 3))
    prof = PROFILE.model_copy(update={"ring": ring, "ring_in": ring_in})
    wide = np.full((4, 180, 320), 250, np.uint8)
    flash = np.stack([frame(250, 250)] * 4)
    assert classify_profile_scene(0.0, 5.0, flash, (0, 0), wide, prof, None).kind == "STACKED"
    hidden = np.full_like(flash, 250)  # the cam gone too
    assert classify_profile_scene(0.0, 5.0, hidden, (0, 0), wide, prof, None).kind == "FIT_BLUR"


def test_classify_profile_scene_reuses_a_near_identical_previous_fit():
    """Consecutive scenes of one cutscene keep one picture box (no framing jitter between
    them), so they merge into one layout."""
    ring, frames = _ring_frames(True)
    prof = PROFILE.model_copy(update={"ring": ring})
    wide = np.zeros((4, 180, 320), np.uint8)
    wide[:, :, 40:280] = 80
    prev = (38 / 320, 0.0, 244 / 320, 1.0)
    lay = classify_profile_scene(0.0, 5.0, frames, (0, 0), wide, prof, None, prev_fit=prev)
    assert lay.fit_src == pytest.approx(prev)
    far = (60 / 320, 0.0, 200 / 320, 1.0)
    lay = classify_profile_scene(0.0, 5.0, frames, (0, 0), wide, prof, None, prev_fit=far)
    assert lay.fit_src == pytest.approx((40 / 320, 0.0, 240 / 320, 1.0))


# --- synthetic matrix (layout side) ---------------------------------------------------------------


def _textured_profile(seed: int = 0):
    """A profile whose border ring (dark outer line, white border) and cam picture (a dim
    room: values 0-60, i.e. mostly <= 40 gray) have structure, and its median frame."""
    rng = np.random.default_rng(seed)
    med = np.full((720, 1280), 30, np.uint8)
    ox0, oy0, ox1, oy1 = PROFILE.outer
    x0, y0, x1, y1 = PROFILE.cam
    med[oy0:oy1, ox0:ox1] = 20
    med[oy0 + 1 : oy1 - 1, ox0 + 1 : ox1 - 1] = 230
    med[y0:y1, x0:x1] = rng.integers(0, 60, size=(y1 - y0, x1 - x0))
    ring = ring_reference(med, PROFILE.outer)
    ring_in = ring_reference(med, (x0 + 3, y0 + 3, x1 - 3, y1 - 3))
    return PROFILE.model_copy(update={"ring": ring, "ring_in": ring_in}), med


def test_classify_profile_scene_dark_frames_do_not_show_the_cam():
    """A black / near-black screen (the cam hidden, a dark loading screen) matches the dim
    points of the reference by value alone; with no structure it is not the cam."""
    prof, med = _textured_profile()
    wide = np.full((4, 180, 320), 12, np.uint8)
    for level in (0, 12, 25):
        dark = np.full((4, 720, 1280), level, np.uint8)
        lay = classify_profile_scene(0.0, 5.0, dark, (0, 0), wide, prof, None)
        assert lay.kind == "FIT_BLUR", level
    # the cam itself, with sensor noise, still shows
    rng = np.random.default_rng(1)
    noisy = np.clip(med[None].astype(int) + rng.integers(-6, 7, size=(4, 720, 1280)), 0, 255)
    lay = classify_profile_scene(0.0, 5.0, noisy.astype(np.uint8), (0, 0), wide, prof, None)
    assert lay.kind == "STACKED"


def test_hidden_scene_layout_tracks_a_just_chatting_face():
    """A Just Chatting stretch of a profiled VOD (the overlay off, the streamer full frame):
    a large face in most samples is followed (FACE_TRACK), not blurred-fitted; an in-game
    character's small face or no face stays FIT_BLUR."""
    from hookline.pipeline.layout import hidden_scene_layout
    from hookline.vision.faces import Face

    big = [(t * 0.25, [Face(0.42, 0.2, 0.16, 0.3, 0.95, 0.0, 0.3)]) for t in range(20)]
    assert hidden_scene_layout(big, 0.0, 5.0, src_h=720).kind == "FACE_TRACK"
    small = [(t * 0.25, [Face(0.6, 0.4, 0.04, 0.08, 0.9, 0.0, None)]) for t in range(20)]
    assert hidden_scene_layout(small, 0.0, 5.0, src_h=720).kind == "FIT_BLUR"
    assert hidden_scene_layout([(0.0, [])], 0.0, 5.0, src_h=720).kind == "FIT_BLUR"


def test_pillarbox_rejects_a_dark_symmetric_vignette():
    """A dark scene whose edges fade to black evenly (a vignette) has no hard picture edge:
    not a pillarboxed cutscene."""
    xx = np.arange(320)
    ramp = np.clip(np.minimum(xx, 319 - xx) * 0.6 - 10, 0, 60).astype(np.uint8)
    frames = np.repeat(np.repeat(ramp[None, None, :], 180, axis=1), 4, axis=0)
    assert pillarbox(frames, []) is None
    hard = np.zeros((4, 180, 320), np.uint8)
    hard[:, :, 40:280] = 50
    assert pillarbox(hard, []) == pytest.approx((40 / 320, 0, 240 / 320, 1))


def test_plan_cam_beats_never_cover_a_scare_and_land_on_the_reaction_after():
    """A beat never shows during [scare - 0.5 s, scare + 1 s] (the payoff is the gameplay)
    nor over a gameplay spike: a reaction peak right at the scare gets its beat just after
    it (the reaction continuing), up to BEAT_SHIFT_MAX later; else none."""
    from hookline.pipeline.layout import BEAT_SHIFT_MAX

    lays = [_stacked(0.0, 40.0)]
    (out,) = plan_cam_beats([(10.2, 12.0)], lays, cuts=[], duration=40.0, avoid=[(9.5, 11.0)])
    ((a, b),) = out.cam_beats
    assert a >= 11.0 - 1e-6 and a - 10.2 <= BEAT_SHIFT_MAX + 1e-6 and b - a == pytest.approx(2.5)
    # a long spike: shifting past it would leave the reaction behind -- no beat
    (out,) = plan_cam_beats([(10.2, 12.0)], lays, cuts=[], duration=40.0, avoid=[(9.5, 14.0)])
    assert out.cam_beats == []
    # a spike inside the would-be beat pushes it past the spike too
    (out,) = plan_cam_beats([(20.0, 9.0)], lays, cuts=[], duration=40.0, avoid=[(21.0, 21.4)])
    ((a, _),) = out.cam_beats
    assert a >= 21.4 - 1e-6
