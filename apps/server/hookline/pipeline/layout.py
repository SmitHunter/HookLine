"""Per-scene 9:16 layout choice: STACKED facecam, FACE_TRACK, SPLIT, or FIT_BLUR."""

import logging
import math
import statistics
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from hookline.media.ffmpeg import FFmpegError, probe_duration, probe_size
from hookline.pipeline.profile import BAR_CONTENT, BAR_LIMIT, LayoutProfile, PxBox, ring_visible
from hookline.vision.faces import Crop, Face, sample_frames

LayoutKind = Literal["STACKED", "FACE_TRACK", "SPLIT", "FIT_BLUR"]
log = logging.getLogger(__name__)
Box = tuple[float, float, float, float]  # normalized x, y, w, h

MIN_SCORE = 0.6
CAM_MAX_H = 0.15
CAM_MAX_AREA = 0.20  # grown facecam box (normalized w*h); larger = a webcam shot, not a cam
CAM_MIN_SHARE = 0.6
CAM_MAX_STD = 0.03
CAM_GROW_W = 3.0
CAM_GROW_H = 2.6
SPLIT_MIN_H = 0.15
SPLIT_MIN_SHARE = 0.5
TRACK_MIN_SHARE = 0.5
CAM_VETO_SHARE = 0.3  # share of cam samples with a comparable face outside the cam box
CAM_VETO_MIN_H = 0.5  # ... "comparable" = at least half the cam face's height
CAM_MIN_PX = 160  # grown cam box must be this many source pixels tall
GROUP_MIN_SHARE = 0.5  # share of samples showing a small-face group -> wide shot (FIT_BLUR)
GROUP_MAX_RATIO = 2.0  # faces within 2x height of the largest are "comparable"
TRACK_HOLD_S = 3.0  # a face lost for up to this long: the crop holds its last good x ...
TRACK_EASE_S = 1.0  # ... beyond it, eases to the scene's median face x over this long
HAIR_SHARE = 0.5  # head top ~ face box top - half a face height (hair)
HEAD_QUANTILE = 0.1  # the head top budgeted for: this low quantile of the samples (highest)
ZOOM_TOP = 1.2  # the highest zoom step level (a punch); see reframe.ZOOM_PUNCH
FACE_CORE_TOP = 0.2  # top share of a face box (hairline/forehead) outside the face core
SAMPLE_EVERY = 0.25  # face samples every 0.25 s (4 fps): the tracking edge guard needs them
# Gaze lead room: a FACE_TRACK crop shifts up to LEAD of its width toward where the subject
# looks (median landmark yaw over the scene): none under YAW_DEAD, the full lead from YAW_FULL.
LEAD = 0.08
YAW_DEAD = 0.1
YAW_FULL = 0.3
# STACKED panels (1080 wide): the cam on top, the gameplay under it.
STACK_W = 1080
STACK_CAM_H = 760
STACK_GAME_H = 1920 - STACK_CAM_H  # 1160
# Wave G-layout: a stream profile's fixed STACKED template. The gameplay panel may shift up to
# GAME_SHIFT_MAX of the frame width to keep the cam and its label out; else they're blurred.
GAME_SHIFT_MAX = 0.25
# Per-scene checks: samples every PROFILE_SAMPLE_S s; the cam shows in a scene when its border
# matches in >= CAM_SHOWN of them. A pillarbox: dark side bars each >= PILLAR_MIN of the width,
# within PILLAR_SYMMETRY of each other (a cutscene), outside the cam and its label.
PROFILE_SAMPLE_S = 1.0
CAM_SHOWN = 0.5
PILLAR_MIN = 0.05
PILLAR_SYMMETRY = 0.03
PILLAR_W = 320
FIT_TRIM = 0.15  # an overlay at a fitted picture's side edge is cut off when it costs <= this
# Full-cam reaction beats: the cam alone (whole, blurred background) for BEAT_S at the
# strongest reaction peaks, starting BEAT_LEAD before the peak; at most MAX_BEATS per clip,
# never before BEAT_AFTER (the hook title), BEAT_GAP clear of other switches (scene changes,
# other beats, the clip's end), never across a cut.
BEAT_S = 2.5
BEAT_LEAD = 0.3
MAX_BEATS = 2
BEAT_AFTER = 3.0
BEAT_GAP = 2.0


class SceneLayout(BaseModel):
    start: float
    end: float
    kind: LayoutKind
    facecam: Box | None = None
    track_x: list[tuple[float, float]] = []
    track_w: list[float] = []  # face width per track_x keyframe, normalized (edge guard)
    lead: float = 0.0  # gaze lead room: signed share of crop width the crop shifts (+ right)
    # Edge-guard framing changes (reframe.guard_zoom): (output s, signed share of crop width)
    # sideways crop offsets, instant at cuts; and the rendered face centre's median offset from
    # the crop centre (share of crop width, + right), for caption placement and the audit.
    shift_steps: list[tuple[float, float]] = []
    face_dx: float | None = None
    # Fix wave F: per track keyframe the face's (top y, height, eye line y) normalized (eye
    # None when unknown); the reframe snaps (output s, framing changes for Q7); the rendered
    # face core per sample [output s, x0, top, x1, bottom, eye y] in output px (caption
    # placement and Q4b); word gaps (output s) a snap prefers -- render-time only, not saved.
    track_face: list[tuple[float, float, float | None]] = []
    snaps: list[float] = []
    face_px: list[list[float | None]] = []
    gaps: list[tuple[float, float]] = Field(default=[], exclude=True)
    cuts: list[float] = Field(default=[], exclude=True)  # output s: changes there are instant
    split_x: tuple[float, float] | None = None
    face_box: tuple[float, float] | None = None  # median face (y, h), normalized; captions
    face_w: float | None = None  # median face width (FACE_TRACK), normalized to source width
    head_top: float | None = None  # highest head top (HEAD_QUANTILE), normalized source y
    # Zoom plan (reframe.plan_zoom): step level s shows as zoom_level(s); zooms keep the panel
    # row zoom_anchor (normalized panel y; None = face centre / panel centre) fixed.
    base_zoom: float = 1.0
    zoom_cap: float | None = None
    zoom_anchor: float | None = None
    step_shift: float = 0.0  # flat-zoom FACE_TRACK: signed share of crop width steps shift by
    # Wave G-layout (a stream profile's fixed STACKED template; `facecam` is then the cam
    # panel's window): the cam's picture inside its border (full-cam beats), the head in it
    # (top of hair to chin: cam-panel zoom, captions, Q4b), the gameplay panel's source region
    # (`game_fit`: shown whole, letterboxed -- a pillarboxed cutscene) with the overlay parts in
    # it blurred over (`patches`), and the full-cam reaction beats (output s). Normalized boxes.
    cam_rect: Box | None = None
    head_box: Box | None = None
    game_box: Box | None = None
    game_fit: bool = False
    patches: list[Box] = []
    cam_beats: list[tuple[float, float]] = []
    fit_src: Box | None = Field(default=None, exclude=True)  # the pillarbox picture found

    def zoom_level(self, s: float) -> float:
        """Rendered zoom for schedule level `s` (1.0 / ZOOM_IN / ZOOM_TOP): base_zoom times
        the step, the step part compressed (never clipped flat) so ZOOM_TOP lands on zoom_cap
        when the full range would pass it; every distinct level stays distinct."""
        k = 1.0
        if self.zoom_cap is not None:
            k = min(1.0, max(self.zoom_cap / self.base_zoom - 1.0, 0.0) / (ZOOM_TOP - 1.0))
        return self.base_zoom * (1.0 + (s - 1.0) * k)

    def zoom_y(self, y: float, s: float, panel_h: float) -> float:
        """Where panel row `y` (px, unzoomed) lands at schedule level `s`: zooms scale about
        the anchor row (zoom_anchor, else the face centre, else the panel centre)."""
        a = self.anchor_px(panel_h)
        return a + self.zoom_level(s) * (y - a)

    def anchor_px(self, panel_h: float) -> float:
        if self.zoom_anchor is not None:
            return self.zoom_anchor * panel_h
        if self.kind == "FACE_TRACK" and self.face_box is not None:
            return (self.face_box[0] + self.face_box[1] / 2) * panel_h
        return panel_h / 2


class _Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Face]: ...


def _in_corner(f: Face) -> bool:
    return (f.cx < 0.35 or f.cx > 0.65) and (f.cy < 0.4 or f.cy > 0.6)


def _grow_inside(cx: float, cy: float, w: float, h: float) -> Box:
    """Box of size w x h centred on (cx, cy), shifted (shrunk only if larger) to fit [0, 1]."""
    w, h = min(w, 1.0), min(h, 1.0)
    x = min(max(cx - w / 2, 0.0), 1.0 - w)
    y = min(max(cy - h / 2, 0.0), 1.0 - h)
    return (x, y, w, h)


def _inside(box: Box, f: Face) -> bool:
    x, y, w, h = box
    return x <= f.cx <= x + w and y <= f.cy <= y + h


def _facecam(samples: list[list[Face]], src_h: int | None = None) -> Box | None:
    """Fixed small corner face -> its grown cam box. Faces in `samples` already passed
    MIN_SCORE. Vetoed when, in >= 30% of the cam samples, another comparable face sits outside
    the box (a wide shot with someone near a corner, not a facecam overlay), and when the box
    would be under 160 source px tall."""
    cams: list[tuple[Face, list[Face]]] = []
    for faces in samples:
        corner = [f for f in faces if f.h < CAM_MAX_H and _in_corner(f)]
        if corner:
            cams.append((max(corner, key=lambda f: f.w * f.h), faces))
    if not samples or len(cams) < CAM_MIN_SHARE * len(samples):
        return None
    cxs, cys = [c.cx for c, _ in cams], [c.cy for c, _ in cams]
    if statistics.pstdev(cxs) >= CAM_MAX_STD or statistics.pstdev(cys) >= CAM_MAX_STD:
        return None
    box = _grow_inside(
        statistics.median(cxs),
        statistics.median(cys),
        statistics.median(c.w for c, _ in cams) * CAM_GROW_W,
        statistics.median(c.h for c, _ in cams) * CAM_GROW_H,
    )
    if box[2] * box[3] >= CAM_MAX_AREA:
        return None
    if src_h is not None and box[3] * src_h < CAM_MIN_PX:
        return None
    vetoed = sum(
        any(f is not c and f.h >= CAM_VETO_MIN_H * c.h and not _inside(box, f) for f in faces)
        for c, faces in cams
    )
    return None if vetoed >= CAM_VETO_SHARE * len(cams) else box


def _small_group(faces: list[Face]) -> bool:
    """>= 2 comparable faces (within 2x height), all under SPLIT_MIN_H: a wide group shot."""
    if len(faces) < 2:
        return False
    tallest = max(f.h for f in faces)
    comparable = [f for f in faces if f.h * GROUP_MAX_RATIO >= tallest]
    return len(comparable) >= 2 and tallest < SPLIT_MIN_H


def classify_scene(
    samples: list[tuple[float, list[Face]]],
    start: float,
    end: float,
    facecam_override: Box | None = None,
    src_h: int | None = None,
) -> SceneLayout:
    """`src_h` (source height in px, when known) rejects facecam boxes under 160 px."""
    if facecam_override is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=facecam_override)

    kept = [(t, [f for f in faces if f.score >= MIN_SCORE]) for t, faces in samples]
    n = len(kept)

    box = _facecam([faces for _, faces in kept], src_h)
    if box is not None:
        return SceneLayout(start=start, end=end, kind="STACKED", facecam=box)

    if n and sum(_small_group(faces) for _, faces in kept) >= GROUP_MIN_SHARE * n:
        return SceneLayout(start=start, end=end, kind="FIT_BLUR")

    pairs = []
    pair_boxes = []
    for _, faces in kept:
        big = sorted((f for f in faces if f.h >= SPLIT_MIN_H), key=lambda f: f.w * f.h)
        if len(big) >= 2:
            pairs.append(sorted((big[-1].cx, big[-2].cx)))
            pair_boxes.append((min(big[-1].y, big[-2].y), max(big[-1].h, big[-2].h)))
    if n and len(pairs) >= SPLIT_MIN_SHARE * n:
        left = statistics.median(p[0] for p in pairs)
        right = statistics.median(p[1] for p in pairs)
        return SceneLayout(
            start=start,
            end=end,
            kind="SPLIT",
            split_x=(min(left, right), max(left, right)),
            face_box=_median_box(pair_boxes),
            head_top=_quantile([y - HAIR_SHARE * h for y, h in pair_boxes], HEAD_QUANTILE),
        )

    largest = [(t, max(faces, key=lambda f: f.w * f.h)) for t, faces in kept if faces]
    if n and len(largest) >= TRACK_MIN_SHARE * n:
        track = _track_keyframes(kept)
        return SceneLayout(
            start=start,
            end=end,
            kind="FACE_TRACK",
            track_x=[(t, x) for t, x, *_ in track],
            track_w=[w for _, _, w, *_ in track],
            track_face=[(y, h, eye) for *_, y, h, eye in track],
            lead=gaze_lead([f.yaw for _, f in largest]),
            face_box=_median_box([(f.y, f.h) for _, f in largest]),
            face_w=statistics.median(f.w for _, f in largest),
            head_top=_head_top([f for _, f in largest]),
        )

    return SceneLayout(start=start, end=end, kind="FIT_BLUR")


def _quantile(values: list[float], q: float) -> float:
    """Nearest-rank `q` quantile (q=0.1: a low value that ignores a stray outlier or two)."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def _head_top(faces: list[Face]) -> float:
    return _quantile([f.y - HAIR_SHARE * f.h for f in faces], HEAD_QUANTILE)


def gaze_lead(yaws: list[float | None]) -> float:
    """Signed lead room (share of crop width; + = the crop shifts right) from the scene's
    median face yaw: 0 below YAW_DEAD, ramping to LEAD at YAW_FULL. The median ignores a
    stray estimate; one value per scene keeps the crop steady."""
    known = [y for y in yaws if y is not None]
    if not known:
        return 0.0
    yaw = statistics.median(known)
    ramp = min(max((abs(yaw) - YAW_DEAD) / (YAW_FULL - YAW_DEAD), 0.0), 1.0)
    return math.copysign(LEAD * ramp, yaw) if ramp else 0.0


Keyframe = tuple[float, float, float, float, float, float | None]  # t, x, w, y, h, eye y


def _track_keyframes(kept: list[tuple[float, list[Face]]]) -> list[Keyframe]:
    """(t, centre x, width, top y, height, eye line y) of the largest face per sample. A
    sample with no face (none passed MIN_SCORE) HOLDS the last good face rather than following
    whatever else is in frame; after TRACK_HOLD_S without a face its x eases to the scene's
    median face x over TRACK_EASE_S. Samples before the first face add nothing (the first x
    holds back to the scene start), so the track only interpolates between valid samples and
    hold points."""
    xs = [max(faces, key=lambda f: f.w * f.h).cx for _, faces in kept if faces]
    median = statistics.median(xs) if xs else 0.5
    out: list[Keyframe] = []
    last: Keyframe | None = None
    for t, faces in sorted(kept, key=lambda s: s[0]):
        if faces:
            face = max(faces, key=lambda f: f.w * f.h)
            last = (t, face.cx, face.w, face.y, face.h, face.eye_y)
            out.append(last)
        elif last is not None:
            ease = min(max((t - last[0] - TRACK_HOLD_S) / TRACK_EASE_S, 0.0), 1.0)
            out.append((t, last[1] + (median - last[1]) * ease, *last[2:]))
    return out


def _median_box(boxes: list[tuple[float, float]]) -> tuple[float, float]:
    return (statistics.median(b[0] for b in boxes), statistics.median(b[1] for b in boxes))


def _scenes(video: Path) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, detect

    found = detect(str(video), ContentDetector())
    if not found:
        return [(0.0, probe_duration(video))]
    return [(a.seconds, b.seconds) for a, b in found]


def _sample_times(start: float, end: float, every: float) -> list[float]:
    """Midpoints of `every`-second slots, so no sample lands on a scene edge or past the end."""
    n = int((end - start) // every)
    if n < 1:
        return [(start + end) / 2]
    return [start + (k + 0.5) * every for k in range(n)]


def plan_layouts(
    video: Path,
    detector: _Detector | None,
    facecam_override: Box | None = None,
    sample_every: float = SAMPLE_EVERY,
    crop: Crop | None = None,
) -> list[SceneLayout]:
    """One layout per scene. With no detector (and no override) the whole video is a single
    FIT_BLUR scene: without faces every scene would be FIT_BLUR anyway."""
    if detector is None and facecam_override is None:
        return [SceneLayout(start=0.0, end=probe_duration(video), kind="FIT_BLUR")]
    src_h = None
    if facecam_override is None:
        src_h = crop[1] if crop else probe_size(video)[1]
    layouts = []
    for start, end in _scenes(video):
        if facecam_override is not None:
            layouts.append(classify_scene([], start, end, facecam_override))
            continue
        times = _sample_times(start, end, sample_every)
        assert detector is not None
        try:
            frames = sample_frames(video, times, crop=crop)
            samples = [(t, detector.detect(f)) for t, f in zip(times, frames, strict=True)]
        except FFmpegError as e:
            # unreadable scene: fall back to FIT_BLUR rather than abort the plan, never silently
            log.warning(
                "face sampling failed for %s scene %.2f-%.2f s (%s); the scene uses FIT_BLUR",
                video.name, start, end, (str(e).splitlines() or ["error"])[0],
            )  # fmt: skip
            samples = []
        layouts.append(classify_scene(samples, start, end, src_h=src_h))
    return layouts


# --- Wave G-layout: the fixed STACKED template of a stream profile ------------------------


def _norm(box: PxBox, crop: Crop | None, size: tuple[int, int]) -> Box:
    """A source-px (x0, y0, x1, y1) box as a normalized (x, y, w, h) box of `crop` (or the
    full `size` frame), clipped to it."""
    cw, ch, cx, cy = crop or (size[0], size[1], 0, 0)
    x0 = min(max((box[0] - cx) / cw, 0.0), 1.0)
    y0 = min(max((box[1] - cy) / ch, 0.0), 1.0)
    x1 = min(max((box[2] - cx) / cw, 0.0), 1.0)
    y1 = min(max((box[3] - cy) / ch, 0.0), 1.0)
    return (x0, y0, x1 - x0, y1 - y0)


def _overlap(a: Box, b: Box) -> Box | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 + 1e-9 and y1 > y0 + 1e-9 else None


def _cam_window(cam: Box, head: Box, aspect: float) -> Box:
    """The largest `aspect` (w/h in normalized units) window inside `cam`: the cam's whole
    width (only background above/below cropped; a wider cam: its whole height), centred on the
    head, shifted to stay inside the cam."""
    x, y, w, h = cam
    hx, hy, hw, hh = head
    if w / h <= aspect:
        win_h = w / aspect
        wy = min(max(hy + hh / 2 - win_h / 2, y), y + h - win_h)
        return (x, wy, w, win_h)
    win_w = h * aspect
    wx = min(max(hx + hw / 2 - win_w / 2, x), x + w - win_w)
    return (wx, y, win_w, h)


def stack_template(
    profile: LayoutProfile, crop: Crop | None, size: tuple[int, int], fit: Box | None = None
) -> dict:
    """SceneLayout fields of the profile's fixed STACKED template in `crop` space: the cam
    panel window (`facecam`: the cam's whole width, background above/below cropped, the head
    centred; never the border), `cam_rect`, `head_box`, `face_box` / `head_top` (zoom
    headroom), and the gameplay panel (`game_box`): the full frame height centred on the
    screen centre, shifted up to GAME_SHIFT_MAX of the width when that keeps the cam and its
    label out, else those blurred over (`patches`); with `fit` (a pillarboxed scene's
    picture) that picture whole, less the overlays at its side edges (when they take at most
    FIT_TRIM of its width), the other overlay parts on it blurred."""
    assert profile.cam is not None and profile.head is not None
    cw, ch, _, _ = crop or (size[0], size[1], 0, 0)
    cam = _norm(profile.cam, crop, size)
    head = _norm(profile.head, crop, size)
    # normalized units: a w/h ratio in pixels is (w * cw) / (h * ch)
    window = _cam_window(cam, head, STACK_W / STACK_CAM_H * ch / cw)
    keep_out = [_norm(b, crop, size) for b in profile.keep_out()]
    fields: dict = {
        "facecam": window,
        "cam_rect": cam,
        "head_box": head,
        "head_top": head[1],
        "game_fit": fit is not None,
    }
    if profile.face is not None:
        face = _norm(profile.face, crop, size)
        fields["face_box"] = (face[1], face[3])
    if fit is not None:
        box = _trim_fit(fit, keep_out)
        fields["game_box"] = box
        fields["patches"] = [p for b in keep_out if (p := _overlap(b, box)) is not None]
        return fields
    gh = 1.0
    gw = STACK_W / STACK_GAME_H * ch / cw
    if gw > 1.0:
        gw, gh = 1.0, STACK_GAME_H / STACK_W * cw / ch
    gy = (1.0 - gh) / 2
    centre = (1.0 - gw) / 2

    def clear(x: float) -> bool:
        return all(_overlap((x, gy, gw, gh), b) is None for b in keep_out)

    options = [centre]
    for b in keep_out:
        options += [b[0] + b[2], b[0] - gw]
    ok = [
        x
        for x in options
        if -1e-9 <= x <= 1.0 - gw + 1e-9 and abs(x - centre) <= GAME_SHIFT_MAX and clear(x)
    ]
    gx = min(max(min(ok, key=lambda x: abs(x - centre)) if ok else centre, 0.0), 1.0 - gw)
    game = (gx, gy, gw, gh)
    fields["game_box"] = game
    fields["patches"] = [] if ok else [p for b in keep_out if (p := _overlap(b, game))]
    return fields


def _trim_fit(fit: Box, keep_out: list[Box]) -> Box:
    """`fit` with the overlays touching its left / right edge cut off, each side only when
    that costs at most FIT_TRIM of its width (else they stay, to be blurred)."""
    x0, x1 = fit[0], fit[0] + fit[2]
    left, right = x0, x1
    for b in keep_out:
        ov = _overlap(b, fit)
        if ov is None:
            continue
        if ov[0] <= x0 + 1e-9:
            left = max(left, ov[0] + ov[2])
        elif ov[0] + ov[2] >= x1 - 1e-9:
            right = min(right, ov[0])
    if left - x0 > FIT_TRIM * fit[2]:
        left = x0
    if x1 - right > FIT_TRIM * fit[2]:
        right = x1
    return (left, fit[1], right - left, fit[3])


def pillarbox(frames: np.ndarray, keep_out: list[Box]) -> Box | None:
    """The picture of a pillarboxed scene (normalized (x, 0, w, 1)) from its gray (n, H, W)
    samples, the `keep_out` boxes (cam, label) ignored: content columns (> BAR_CONTENT of a
    column brighter than BAR_LIMIT in some sample) leave dark bars each >= PILLAR_MIN of the
    width on BOTH sides, within PILLAR_SYMMETRY of each other. None otherwise."""
    _, h, w = frames.shape
    lit = frames > BAR_LIMIT
    for x, y, bw, bh in keep_out:
        lit[:, int(y * h) : math.ceil((y + bh) * h), int(x * w) : math.ceil((x + bw) * w)] = False
    cols = np.flatnonzero((lit.mean(axis=1) > BAR_CONTENT).any(axis=0))
    if not len(cols):
        return None
    left, right = int(cols[0]), w - 1 - int(cols[-1])
    if min(left, right) < PILLAR_MIN * w or abs(left - right) > PILLAR_SYMMETRY * w:
        return None
    return (left / w, 0.0, (w - left - right) / w, 1.0)


def classify_profile_scene(
    start: float,
    end: float,
    ring_frames: np.ndarray,
    origin: tuple[int, int],
    wide: np.ndarray,
    profile: LayoutProfile,
    crop: Crop | None,
    prev_fit: Box | None = None,
) -> SceneLayout:
    """A scene of a profiled stream: the fixed STACKED template when the cam's border (or its
    picture just inside it) shows in >= CAM_SHOWN of the `ring_frames` (gray crops of the
    source at px `origin`), its
    gameplay fitted when the `wide` samples (gray, `crop` space) are pillarboxed; the cam
    hidden (BRB, a full-screen cutscene): FIT_BLUR of the gameplay. Never FACE_TRACK: the
    only face that matters is the streamer's, in the cam. A pillarbox picture within
    PILLAR_SYMMETRY of the previous scene's (`prev_fit`) is taken as that one: one cutscene,
    one framing."""
    shown = sum(
        ring_visible(f, profile.ring, origin) or ring_visible(f, profile.ring_in, origin)
        for f in ring_frames
    )
    if not len(ring_frames) or shown < CAM_SHOWN * len(ring_frames):
        return SceneLayout(start=start, end=end, kind="FIT_BLUR")
    keep_out = [_norm(b, crop, profile.size) for b in profile.keep_out()]
    fit = pillarbox(wide, keep_out)
    near = fit is not None and prev_fit is not None
    if near and all(abs(a - b) <= PILLAR_SYMMETRY for a, b in zip(fit, prev_fit, strict=True)):  # type: ignore[arg-type]
        fit = prev_fit
    tpl = stack_template(profile, crop, profile.size, fit=fit)
    return SceneLayout(start=start, end=end, kind="STACKED", fit_src=fit, **tpl)


def _gray(frames: list[np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            (f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299).astype(np.uint8)
            for f in frames
        ]
    )


def _ring_box(profile: LayoutProfile) -> Crop:
    """The even-aligned source-px crop (w, h, x, y) holding every border ring point."""
    xs = [x for x, _, _ in [*profile.ring, *profile.ring_in]]
    ys = [y for _, y, _ in [*profile.ring, *profile.ring_in]]
    x0, y0 = min(xs) // 2 * 2, min(ys) // 2 * 2
    return max((max(xs) + 2 - x0) // 2 * 2, 2), max((max(ys) + 2 - y0) // 2 * 2, 2), x0, y0


def plan_profile_layouts(
    video: Path,
    profile: LayoutProfile,
    crop: Crop | None = None,
    sample_every: float = PROFILE_SAMPLE_S,
) -> list[SceneLayout]:
    """One layout per scene of a profiled stream (`classify_profile_scene`), identical
    neighbours merged: the cam is never re-detected, only checked for showing (its border)
    and the gameplay for pillarboxing. `video` has the profile's source size; `crop` is the
    letterbox / overlay crop the reframe applies."""
    rw, rh, rx, ry = _ring_box(profile)
    layouts = []
    for start, end in _scenes(video):
        times = _sample_times(start, end, sample_every)
        try:
            ring = _gray(sample_frames(video, times, width=rw, crop=(rw, rh, rx, ry)))
            wide = _gray(sample_frames(video, times, width=PILLAR_W, crop=crop))
        except FFmpegError as e:
            log.warning(
                "profile checks failed for %s scene %.2f-%.2f s (%s); the scene keeps the "
                "STACKED template", video.name, start, end, (str(e).splitlines() or ["error"])[0],
            )  # fmt: skip
            tpl = stack_template(profile, crop, profile.size)
            layouts.append(SceneLayout(start=start, end=end, kind="STACKED", **tpl))
            continue
        prev = layouts[-1].fit_src if layouts else None
        layouts.append(
            classify_profile_scene(start, end, ring, (rx, ry), wide, profile, crop, prev)
        )
    return merge_same(layouts)


_GEOMETRY = ("kind", "facecam", "cam_rect", "head_box", "game_box", "game_fit", "patches")


def merge_same(layouts: list[SceneLayout]) -> list[SceneLayout]:
    """Consecutive scenes with the same kind and geometry joined into one: one framing, no
    camera change (nor its instant zoom step) at a gameplay scene cut."""
    out: list[SceneLayout] = []
    for lay in layouts:
        prev = out[-1] if out else None
        if prev is not None and all(getattr(prev, k) == getattr(lay, k) for k in _GEOMETRY):
            out[-1] = prev.model_copy(update={"end": lay.end})
        else:
            out.append(lay)
    return out


def plan_cam_beats(
    peaks: list[tuple[float, float]],
    layouts: list[SceneLayout],
    cuts: list[float],
    duration: float,
    avoid: list[tuple[float, float]] | None = None,
) -> list[SceneLayout]:
    """`layouts` with full-cam reaction beats (`cam_beats`, output s) at the strongest
    `peaks` (output s, score): BEAT_S long from BEAT_LEAD before the peak, inside one
    template STACKED scene (cam shown, gameplay cropped) at least BEAT_GAP from its inner
    bounds, from BEAT_AFTER on and BEAT_GAP before the end, no cut inside, BEAT_GAP from
    other beats and clear of the `avoid` spans; at most MAX_BEATS."""
    chosen: list[tuple[int, float, float]] = []
    for t, _ in sorted(peaks, key=lambda p: -p[1]):
        if len(chosen) >= MAX_BEATS:
            break
        a = round(max(t - BEAT_LEAD, 0.0), 3)
        b = round(a + BEAT_S, 3)
        if a < BEAT_AFTER or b > duration - BEAT_GAP:
            continue
        i = next((k for k, lay in enumerate(layouts) if lay.start <= a < lay.end), None)
        if i is None:
            continue
        lay = layouts[i]
        if lay.kind != "STACKED" or lay.cam_rect is None or lay.game_fit:
            continue
        near_start = lay.start > 1e-6 and a - lay.start < BEAT_GAP
        near_end = lay.end < duration - 1e-6 and lay.end - b < BEAT_GAP
        if b > lay.end or near_start or near_end:
            continue
        if any(a + 1e-6 < c < b - 1e-6 for c in cuts):
            continue
        if any(a < q + BEAT_GAP and b > p - BEAT_GAP for _, p, q in chosen):
            continue
        if any(a < q and b > p for p, q in avoid or []):
            continue
        chosen.append((i, a, b))
    out = list(layouts)
    for i, a, b in chosen:
        out[i] = out[i].model_copy(update={"cam_beats": sorted([*out[i].cam_beats, (a, b)])})
    return out
