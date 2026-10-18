"""Measurements on rendered frames, and the automated quality bar (`hookline audit`)."""

import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from hookline.captions.ass import face_rects, face_top, held_face_px
from hookline.fsutil import atomic_write_text
from hookline.media.audio import SEAM_FRAME_S, seam_jump_db
from hookline.media.ffmpeg import probe_duration, probe_video_kbps
from hookline.media.loudness import measure_loudness
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import Word
from hookline.pipeline.boundaries import FILLER_OPENERS
from hookline.pipeline.ingest import find_source
from hookline.pipeline.layout import SCARE_POST_S, SCARE_PRE_S, SceneLayout
from hookline.pipeline.signals import STATIC_CHROMA_MAX, STATIC_VOICED_MAX
from hookline.pipeline.timing import UNCAPTIONED_S  # Q3b: what the render reports

OUT_W, OUT_H = 1080, 1920
GREY = "0x808080"  # canvas on which black outlines, shadows and hook boxes are visible
TextColor = Literal["white", "yellow", "any"]
Box = tuple[int, int, int, int]


def render_ass_frame(ass_path: Path, t: float, bg: str = "black") -> np.ndarray:
    """Burn `ass_path` over a plain 1080x1920 `bg` frame (an ffmpeg colour, e.g. "black" or
    GREY) and return the frame at `t` s (HxWx3 RGB). Raises RuntimeError with ffmpeg's stderr."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color={bg}:s={OUT_W}x{OUT_H}:r=25:d={t + 1:.3f}",
        "-vf", subtitles_filter(Path(ass_path), fonts_dir()),
        "-ss", f"{t:.3f}", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"rendering {ass_path} failed: {stderr}") from e
    return np.frombuffer(res.stdout, np.uint8).reshape(OUT_H, OUT_W, 3)


def text_mask(frame: np.ndarray, color: TextColor = "any") -> np.ndarray:
    """Boolean mask of text-fill pixels: white (all channels high), yellow (R,G high, B low),
    or any bright pixel."""
    rgb = frame.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "white":
        return (r > 200) & (g > 200) & (b > 200)
    if color == "yellow":
        return (r > 200) & (g > 200) & (b < 90)
    return rgb.max(axis=2) > 128


def _bbox(mask: np.ndarray, region: Box | None) -> Box | None:
    if region is not None:
        x0, y0, x1, y1 = region
        clip = np.zeros_like(mask)
        clip[y0 : y1 + 1, x0 : x1 + 1] = True
        mask = mask & clip
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def text_bbox(frame: np.ndarray, color: TextColor = "any", region: Box | None = None) -> Box | None:
    """(x0, y0, x1, y1) inclusive bounding box of `color` text pixels, or None. `region`
    (x0, y0, x1, y1) limits the search to part of the frame."""
    return _bbox(text_mask(frame, color), region)


def box_bbox(
    frame: np.ndarray, bg: tuple[int, int, int] = (128, 128, 128), region: Box | None = None
) -> Box | None:
    """Bounding box of everything visibly drawn over a plain `bg` canvas: text, outline,
    shadow and hook background boxes."""
    diff = np.abs(frame.astype(np.int16) - np.array(bg, np.int16)).max(axis=2)
    return _bbox(diff > 24, region)


# --- The quality bar (Q1-Q8, plus Q4b: captions and hook off the face) -------------------

LUFS_TARGET, LUFS_TOL, TP_MAX = -14.0, 1.0, -1.0
CAP_MIN_PX = 55
HOOK_MIN_PX = 60
HOOK_MAX_LINES, HOOK_MAX_WORDS, HOOK_UNTIL = 2, 7, 3.0
COVERAGE_MIN = 0.95
SAFE_X, SAFE_Y = (60, 888), (290, 1250)
FACE_OVERLAP_MAX = 0.10
PAGE_TOUCH_S = 0.05  # a caption group's events touch (word highlights); a gap starts another
FACE_SAMPLE_HALF_S = 0.125  # a per-sample face (4 fps) stands for this long either side
SPEECH_START_MAX = 0.5
FIRST_WORDS_MIN = 0.6
PACE_MAX_GAP = 5.0
MIN_VIDEO_KBPS = 5000
# Q8 bar for a soft source: 2.5x its video bitrate, within [2000, MIN_VIDEO_KBPS] kb/s
SOURCE_KBPS_FACTOR, FLOOR_VIDEO_KBPS = 2.5, 2000
# AAC is encoded at 192k; the measured average undershoots on simple audio (real clips 176-189k).
AUDIO_KBPS, AUDIO_KBPS_RANGE = 192, (160, 200)
MAX_CAPTION_SAMPLES = 400  # one per caption group; beyond this the widest/smallest groups win
FRAME_BATCH = 24  # frames rendered per ffmpeg run (~150 MB of RGB), measured, then dropped
DOMINANT_ROWS = 0.4  # Q2: a line's cap band = rows with >= 40% of its densest row's pixels
SAMPLE_FPS = 25
EARLIER_TAG = "EARLIER"  # the cold-open seam tag (hook style), not the hook title

_TIME_RE = re.compile(r"(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_TAG_RE = re.compile(r"\{[^}]*\}")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_FS_RE = re.compile(r"\\fs(\d+)")
_FSCX_RE = re.compile(r"\\fscx(\d+)")


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome. A warning (`warn`) passes (`ok`) but is reported as WARN."""

    id: str
    ok: bool
    value: float | str | None
    threshold: str
    detail: str = ""
    warn: bool = False

    @property
    def status(self) -> Literal["pass", "warn", "fail"]:
        return "fail" if not self.ok else "warn" if self.warn else "pass"

    def as_dict(self) -> dict:
        return {**asdict(self), "status": self.status}


@dataclass(frozen=True)
class _Event:
    start: float
    end: float
    style: str
    text: str  # override tags stripped; "\N" kept as the line break
    raw: str = ""  # the Dialogue text field, override tags included

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def render_ass_frames(ass_path: Path, times: list[float], bg: str = "black") -> list[np.ndarray]:
    """`render_ass_frame` for several times in one ffmpeg run. Frames are picked on a 25 fps
    grid before libass draws them, so long clips stay cheap. Returned in `times` order."""
    if not times:
        return []
    idx = [int(t * SAMPLE_FPS + 1e-6) for t in times]
    uniq = sorted(set(idx))
    select = "+".join(f"eq(n\\,{n})" for n in uniq)
    dur = uniq[-1] / SAMPLE_FPS + 1
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color={bg}:s={OUT_W}x{OUT_H}:r={SAMPLE_FPS}:d={dur:.3f}",
        "-vf", f"select='{select}',{subtitles_filter(Path(ass_path), fonts_dir())}",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"rendering {ass_path} failed: {stderr}") from e
    frames = np.frombuffer(res.stdout, np.uint8).reshape(-1, OUT_H, OUT_W, 3)
    if len(frames) != len(uniq):
        raise RuntimeError(f"rendering {ass_path}: got {len(frames)} of {len(uniq)} frames")
    by_n = dict(zip(uniq, frames, strict=True))
    return [by_n[n] for n in idx]


def _measure_frames[T](
    ass_path: Path, times: list[float], fn: Callable[[np.ndarray], T], bg: str = "black"
) -> list[T]:
    """`fn` of each frame at `times`, rendered FRAME_BATCH at a time so memory stays bounded."""
    out: list[T] = []
    for i in range(0, len(times), FRAME_BATCH):
        out += [fn(f) for f in render_ass_frames(ass_path, times[i : i + FRAME_BATCH], bg)]
    return out


def line_heights(frame: np.ndarray, color: TextColor = "any", dense: float = 0.05) -> list[int]:
    """Height (px) of each text line in `frame`: row runs of text pixels separated by empty
    rows, counting only rows with at least `dense` x the run's densest row (the default drops
    comma tails; DOMINANT_ROWS keeps just the cap band)."""
    rows = text_mask(frame, color).sum(axis=1)
    heights, y = [], 0
    while y < len(rows):
        if rows[y] == 0:
            y += 1
            continue
        y1 = y
        while y1 < len(rows) and rows[y1] > 0:
            y1 += 1
        run = rows[y:y1]
        keep = np.nonzero(run >= max(1, dense * run.max()))[0]
        heights.append(int(keep[-1] - keep[0] + 1))
        y = y1
    return heights


def _secs(ts: str) -> float:
    m = _TIME_RE.fullmatch(ts.strip())
    if not m:
        raise ValueError(f"bad ASS time {ts!r}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _split_ass(text: str) -> tuple[str, list[tuple[str, _Event]]]:
    """(header, [(raw Dialogue line, parsed event)])."""
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.startswith("Dialogue:")), len(lines))
    events = []
    for ln in lines[first:]:
        if ln.startswith("Dialogue:"):
            parts = ln.split(",", 9)
            body = _TAG_RE.sub("", parts[9]).replace("\u2060", "")
            ev = _Event(_secs(parts[1]), _secs(parts[2]), parts[3].strip(), body, parts[9])
            events.append((ln, ev))
    return "\n".join(lines[:first]), events


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


def _covered(spans: list[tuple[float, float]], cover: list[tuple[float, float]]) -> float:
    """Share of the (disjoint) `spans` time inside the (disjoint) `cover` intervals."""
    total = sum(b - a for a, b in spans)
    hit = sum(max(0.0, min(b, d) - max(a, c)) for a, b in spans for c, d in cover)
    return hit / total if total else 1.0


def _visible(e: _Event) -> bool:
    return any(c.isalnum() for c in e.text) and e.end - e.start >= 0.08


def _priority(e: _Event) -> tuple[int, int, int]:
    """Riskiest first: smallest \\fs, then most condensed \\fscx, then the longest text."""
    fs = _FS_RE.search(e.raw)
    sx = _FSCX_RE.search(e.raw)
    return (int(fs.group(1)) if fs else 10_000, int(sx.group(1)) if sx else 100, -len(e.text))


def caption_samples(caps: list[_Event], layouts: list[dict]) -> list[float]:
    """Sample times for caption frames: each caption group's final event (consecutive events
    with the same text are one group; its words light up in turn), plus one inside every
    FACE_TRACK/SPLIT scene that has captions but no sample yet. Above MAX_CAPTION_SAMPLES the
    scene samples stay and the smallest/most condensed/widest groups are kept."""
    groups: list[list[_Event]] = []
    for e in caps:
        if groups and groups[-1][-1].text == e.text:
            groups[-1].append(e)
        else:
            groups.append([e])
    # the last event long enough to land on a sampled frame
    finals = [vis[-1] for g in groups if (vis := [e for e in g if _visible(e)])]
    times = {e.mid for e in finals}
    scene_times = set()
    for lay in layouts:
        if lay.get("kind") not in ("FACE_TRACK", "SPLIT"):
            continue
        a, b = lay["start"], lay["end"]
        if any(a <= t < b for t in times):
            continue
        inside = [e for e in caps if _visible(e) and e.start < b and e.end > a]
        if inside:
            e = inside[0]
            scene_times.add((max(e.start, a) + min(e.end, b)) / 2)
    if len(finals) + len(scene_times) > MAX_CAPTION_SAMPLES:
        room = max(MAX_CAPTION_SAMPLES - len(scene_times), 0)
        finals = sorted(finals, key=_priority)[:room]
    return sorted({e.mid for e in finals} | scene_times)


def _excursion(box: Box) -> int:
    """How far (px) a box pokes out of the Q4 safe zone; 0 when inside."""
    x0, y0, x1, y1 = box
    return max(0, SAFE_X[0] - x0, x1 - SAFE_X[1], SAFE_Y[0] - y0, y1 - SAFE_Y[1])


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("\u2019", "'"))


def _strip_fillers(tokens: list[str]) -> list[str]:
    """Drop leading FILLER_OPENERS (one- or two-word, within the first 3 words): the filler
    trim may legitimately remove them from the clip."""
    i = 0
    while i < min(len(tokens), 3):
        if i + 1 < len(tokens) and f"{tokens[i]} {tokens[i + 1]}" in FILLER_OPENERS:
            i += 2
        elif tokens[i] in FILLER_OPENERS:
            i += 1
        else:
            break
    return tokens[i:]


def _layout_at(layouts: list[dict], t: float) -> SceneLayout | None:
    for lay in layouts:
        if lay["start"] - 1e-6 <= t < lay["end"]:
            return SceneLayout.model_validate(lay)
    return None


def zoom_at(steps: list, t: float) -> float:
    """The zoom step scale in effect at output time `t` (easing ignored); 1.0 without steps."""
    scale = 1.0
    for st, s in sorted((float(a), float(b)) for a, b in steps or []):
        if st <= t + 1e-9:
            scale = s
    return scale


def zoom_levels(sidecar: dict) -> list:
    """The clip's zoom schedule levels, which Q4b maps through each scene's plan: the sidecar's
    `zoom_levels`, else `zoom_steps` (older sidecars recorded the levels there; newer ones
    record the rendered zooms)."""
    return sidecar.get("zoom_levels") or sidecar.get("zoom_steps") or []


def _probe(mp4: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return json.loads(res.stdout)


def _check_loudness(mp4: Path) -> CheckResult:
    thr = f"{LUFS_TARGET:g} +/- {LUFS_TOL:g} LUFS, TP <= {TP_MAX:g} dBTP"
    m = measure_loudness(mp4)
    if m is None:
        return CheckResult("Q1", False, None, thr, "no audio stream")
    ok = abs(m.i - LUFS_TARGET) <= LUFS_TOL and m.tp <= TP_MAX
    return CheckResult("Q1", ok, round(m.i, 2), thr, f"I={m.i:.1f} LUFS, TP={m.tp:.1f} dBTP")


def min_video_kbps(source_kbps: float | None) -> float:
    """Q8's video bitrate bar: min(5000, max(2000, 2.5 x the source's video bitrate)) kb/s; a
    low-bitrate source (soft, dark 720p) has no detail for 5 Mb/s to keep. 5000 when the
    source bitrate is unknown (sidecars from before it was recorded)."""
    if not source_kbps:
        return float(MIN_VIDEO_KBPS)
    return min(MIN_VIDEO_KBPS, max(FLOOR_VIDEO_KBPS, SOURCE_KBPS_FACTOR * source_kbps))


def _check_encode(info: dict, source_kbps: float | None = None) -> CheckResult:
    bar = min_video_kbps(source_kbps)
    thr = f"1080x1920 h264 yuv420p, video >= {bar:.0f} kb/s, aac {AUDIO_KBPS}k"
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        return CheckResult("Q8", False, None, thr, "no video stream")
    a_kbps = int(a.get("bit_rate", 0)) / 1000 if a else 0.0
    v_bps = v.get("bit_rate") or int(info.get("format", {}).get("bit_rate", 0)) - a_kbps * 1000
    v_kbps = float(v_bps) / 1000
    size = f"{v.get('width')}x{v.get('height')}"
    problems = []
    if size != f"{OUT_W}x{OUT_H}":
        problems.append(f"size {size}")
    if v.get("codec_name") != "h264" or v.get("pix_fmt") != "yuv420p":
        problems.append(f"{v.get('codec_name')} {v.get('pix_fmt')}")
    if v_kbps < bar:
        problems.append(f"video {v_kbps:.0f} kb/s")
    lo, hi = AUDIO_KBPS_RANGE
    if a is None or a.get("codec_name") != "aac" or not lo <= a_kbps <= hi:
        problems.append(f"audio {a.get('codec_name') if a else 'none'} {a_kbps:.0f}k")
    detail = f"{size} {v.get('codec_name')} {v.get('pix_fmt')} {v_kbps:.0f} kb/s, aac {a_kbps:.0f}k"
    if problems:
        detail += " | " + "; ".join(problems)
    return CheckResult("Q8", not problems, round(v_kbps), thr, detail)


def _check_caption_size(heights_per_frame: list[list[int]], has_speech: bool) -> CheckResult:
    thr = f"cap height >= {CAP_MIN_PX}px"
    # captions are one line: the tallest run is the line, smaller runs are lifted accents
    heights = [max(hs) for hs in heights_per_frame if hs]
    if not heights:
        return CheckResult("Q2", not has_speech, None, thr, "no caption frames")
    cap = min(heights)
    detail = f"min {cap}px, median {int(np.median(heights))}px over {len(heights)} samples"
    return CheckResult("Q2", cap >= CAP_MIN_PX, cap, thr, detail)


def _check_coverage(caps: list[_Event], words: list[Word] | None) -> CheckResult:
    thr = f">= {COVERAGE_MIN:.0%} of speech captioned"
    if words is None:
        return CheckResult("Q3", False, None, thr, "no caption_words in sidecar (re-render)")
    spans = _union([(w.start, w.end) for w in words])
    if not spans:
        return CheckResult("Q3", True, 1.0, thr, "no speech")
    cov = _covered(spans, _union([(e.start, e.end) for e in caps]))
    speech = sum(b - a for a, b in spans)
    detail = f"{cov:.1%} of {speech:.1f}s speech"
    return CheckResult("Q3", cov >= COVERAGE_MIN, round(cov, 4), thr, detail)


def _check_uncaptioned(side: dict) -> CheckResult:
    """Q3b (a warning, never a failure): voiced stretches the render found with no word in
    the transcript (`uncaptioned_speech`, output s) -- speech the viewer hears uncaptioned."""
    thr = f"no voiced stretch > {UNCAPTIONED_S:g}s without a caption"
    spans = side.get("uncaptioned_speech")
    if spans is None:
        return CheckResult("Q3b", True, None, thr, "not recorded (an older render)")
    if not spans:
        return CheckResult("Q3b", True, 0, thr, "none")
    where = ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in spans)
    total = round(sum(b - a for a, b in spans), 3)
    return CheckResult("Q3b", True, total, thr, f"uncaptioned speech at {where}", warn=True)


def _check_safe_zone(boxes: list[Box | None]) -> CheckResult:
    thr = f"x in {list(SAFE_X)}, y in {list(SAFE_Y)}"
    drawn = [b for b in boxes if b is not None]
    worst = max(drawn, key=_excursion, default=None)
    if worst is None:
        return CheckResult("Q4", True, 0, thr, "nothing drawn")
    out = _excursion(worst)
    return CheckResult("Q4", out == 0, out, thr, f"{len(drawn)} boxes; worst {worst}")


def _face_share(
    box: Box,
    layouts: list[dict],
    steps: list,
    t: float,
    window: tuple[float, float] | None = None,
) -> float:
    """Largest share of `box` covered by a face core (detector box minus its top 20%,
    hairline/forehead; see `face_rects`): every rendered face sample (`face_px`, eye line
    counted) of the scenes shown during `window` (the box's time on screen; default `t`
    alone), else the median face of the scene at `t` scaled by the zoom step in effect."""
    a, b = window or (t, t)
    rects = [
        (s[1], face_top(s), s[3], s[4])
        for lay in layouts
        if lay["start"] - 1e-6 <= b and a < lay["end"]
        for s in held_face_px(lay.get("face_px") or [], lay["start"], lay["end"])
        if a - FACE_SAMPLE_HALF_S <= s[0] <= b + FACE_SAMPLE_HALF_S
    ]
    if not rects:
        lay = _layout_at(layouts, t)
        if lay is None or lay.face_px:
            return 0.0
        rects = face_rects(lay, zoom_at(steps, t))
    x0, y0, x1, y1 = box
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    share = 0.0
    for fx0, fy0, fx1, fy1 in rects:
        ov = max(0.0, min(x1, fx1) - max(x0, fx0)) * max(0.0, min(y1, fy1) - max(y0, fy0))
        share = max(share, ov / area)
    return share


def _event_window(events: list[_Event], t: float) -> tuple[float, float] | None:
    """When the caption group (consecutive, touching events with one text: its words light up
    in turn) or hook event shown at `t` is on screen; None when nothing is. A later page with
    the same text after a gap is another group."""
    hit = next((i for i, e in enumerate(events) if e.start <= t < e.end), None)
    if hit is None:
        return None

    def joined(a: _Event, b: _Event) -> bool:
        return a.text == b.text and b.start - a.end <= PAGE_TOUCH_S

    lo = hi = hit
    while lo > 0 and joined(events[lo - 1], events[lo]):
        lo -= 1
    while hi + 1 < len(events) and joined(events[hi], events[hi + 1]):
        hi += 1
    return events[lo].start, events[hi].end


def _check_face(
    samples: list[tuple[str, float, Box | None, tuple[float, float] | None]],
    layouts: list[dict],
    steps: list,
) -> CheckResult:
    """Q4b: the share of each caption text box (cap band), hook box and EARLIER-tag box
    (what, t, box, time on screen) that covers the face: every per-sample rendered face
    while the box is up, else the median face zoomed as at t."""
    worst, where = 0.0, None
    for what, t, box, window in samples:
        if box is None:
            continue
        share = _face_share(box, layouts, steps, t, window)
        if share > worst:
            worst, where = share, f"{what} at {t:.1f}s"
    detail = f"worst {worst:.0%} of a box on the face core ({where})" if where else "no overlap"
    thr = f"caption/hook face-core overlap <= {FACE_OVERLAP_MAX:.0%}"
    return CheckResult("Q4b", worst <= FACE_OVERLAP_MAX, round(worst, 3), thr, detail)


def _check_hook(
    hooks: list[_Event], per_frame: list[list[int]], duration: float, seam: float | None = None
) -> CheckResult:
    """Q5 from the hook event at 0 s and its line heights rendered over black at 0.1/2.7 s (the
    hook ends at an earlier cold-open `seam`, where the EARLIER tag takes over)."""
    thr = (
        f"<= {HOOK_MAX_LINES} lines, <= {HOOK_MAX_WORDS} words, cap >= {HOOK_MIN_PX}px, "
        f"0-{HOOK_UNTIL:g}s"
    )
    hook = next((e for e in hooks if e.start <= 0.01 and EARLIER_TAG not in e.text), None)
    if hook is None:
        return CheckResult("Q5", False, None, thr, "no hook title at 0 s")
    lines = [ln for ln in hook.text.split("\\N") if ln.strip()]
    n_words = len(" ".join(lines).split())
    # each line's height; runs under half the tallest are lifted accents, not lines
    cap = min((h for hs in per_frame for h in hs if h >= max(hs) / 2), default=0)
    problems = []
    if len(lines) > HOOK_MAX_LINES:
        problems.append(f"{len(lines)} lines")
    if n_words > HOOK_MAX_WORDS:
        problems.append(f"{n_words} words")
    if cap < HOOK_MIN_PX:
        problems.append(f"cap {cap}px")
    until = min(HOOK_UNTIL, duration)
    if seam is not None and 0 < seam < until:
        until = seam
    if hook.end < until - 0.01 or not all(per_frame):
        problems.append("not visible for the first 3 s")
    detail = f"{len(lines)} line(s), {n_words} words, cap {cap}px, {hook.start:.1f}-{hook.end:.1f}s"
    if problems:
        detail += " | " + "; ".join(problems)
    return CheckResult("Q5", not problems, cap, thr, detail)


def first_words_match(first_words: str, spoken: str) -> float:
    """Order-aware similarity (SequenceMatcher ratio) of the declared first words and the
    first len(want) + min(2, len(want)) spoken tokens (a one-word declaration, "Go!", is not
    diluted by the reaction after it), leading filler openers stripped from both."""
    want = _strip_fillers(_tokens(first_words))
    if not want:
        return 0.0
    got = _strip_fillers(_tokens(spoken))[: len(want) + min(2, len(want))]
    return SequenceMatcher(None, want, got, autojunk=False).ratio()


def first_word_clipped(first_words: str, main: list[Word]) -> str | None:
    """The first word of `first_words` (leading filler openers stripped) when the clip's
    opening does not start on it: the render cut it ("fuck was that?" for "The fuck was
    that?"); None when it is heard."""
    want = _strip_fillers(_tokens(first_words))
    got = _strip_fillers(_tokens(" ".join(w.text for w in main)))
    if not want or not got or got[0] == want[0]:
        return None
    return want[0]


def _check_opening(
    words: list[Word] | None,
    first_words: str,
    seam: float | None,
    declared: bool = False,
    moved: dict | None = None,
    actual_first_words: str = "",
) -> CheckResult:
    """Q6: speech by 0.5 s, and the opening (after a cold-open seam, where the declared first
    sentence begins) matches `first_words` in order. `declared` marks Claude's own first_words
    (`declared_first_words`), not the start the boundary code moved to. When the opener guard
    recorded a move (`moved`: from_line, to_line, reason), the opening is checked against
    `actual_first_words` (where the clip now starts) and a pass is a WARN naming the move."""
    if moved:
        res = _check_opening(words, actual_first_words, seam)
        if not res.ok:
            return res
        move = f"L{moved.get('from_line')} -> L{moved.get('to_line')}"
        detail = f"{res.detail}; opener guard moved the start {move}: {moved.get('reason', '')}"
        return replace(res, detail=detail, warn=True)
    thr = f"speech <= {SPEECH_START_MAX:g}s, first_words match >= {FIRST_WORDS_MIN:.0%}"
    if words is None:
        return CheckResult("Q6", False, None, thr, "no caption_words in sidecar (re-render)")
    if not words:
        return CheckResult("Q6", False, None, thr, "no speech")
    start = words[0].start
    main = [w for w in words if seam is None or w.start >= seam - 0.05]
    match = first_words_match(first_words, " ".join(w.text for w in main))
    # the clip's very first word (a cold open's main span may start on a lead-in)
    clipped = first_word_clipped(first_words, main) if seam is None else None
    ok = start <= SPEECH_START_MAX and match >= FIRST_WORDS_MIN and clipped is None
    label = "declared first_words" if declared else "first_words"
    detail = f"speech at {start:.2f}s; {label} match {match:.0%}"
    if clipped is not None:
        detail += f"; first word {clipped!r} clipped"
    if seam is not None:
        detail += f" after the seam at {seam:.2f}s"
    if not first_words.strip():
        detail += " (no first_words declared)"
    return CheckResult("Q6", ok, round(start, 3), thr, detail)


def _check_pacing(side: dict, duration: float) -> CheckResult:
    """Q7: longest stretch without a cut, zoom step (as rendered, edge-guard holds applied),
    edge-guard crop shift, reframe snap, scene (layout) change or full-cam beat / cover edge."""
    thr = f"framing change every <= {PACE_MAX_GAP:g}s"
    changes = {float(t) for t in side.get("cut_times") or []}
    changes |= {float(st[0]) for st in side.get("zoom_steps") or []}
    changes |= {float(lay["start"]) for lay in side.get("layout") or []}
    changes |= {
        float(st[0]) for lay in side.get("layout") or [] for st in lay.get("shift_steps") or []
    }
    changes |= {float(t) for lay in side.get("layout") or [] for t in lay.get("snaps") or []}
    changes |= {  # full-cam reaction beats and screen covers: the cut to the cam and back
        float(t)
        for lay in side.get("layout") or []
        for b in [*(lay.get("cam_beats") or []), *(lay.get("cam_covers") or [])]
        for t in b
    }
    pts = sorted({0.0, duration} | {t for t in changes if 0 < t < duration})
    gap, at = max((b - a, a) for a, b in pairwise(pts))
    detail = f"longest {gap:.1f}s from {at:.1f}s"
    return CheckResult("Q7", gap <= PACE_MAX_GAP + 1e-6, round(gap, 3), thr, detail)


def cap_band_bbox(frame: np.ndarray) -> Box | None:
    """Caption text box with its vertical extent cut to the cap band (rows with at least
    DOMINANT_ROWS of the densest row), so comma tails and lifted accents don't count."""
    box = text_bbox(frame)
    if box is None:
        return None
    rows = text_mask(frame).sum(axis=1)
    band = np.nonzero(rows >= DOMINANT_ROWS * rows.max())[0]
    return box[0], int(band[0]), box[2], int(band[-1])


def _cap_black(frame: np.ndarray) -> tuple[list[int], Box | None]:
    return line_heights(frame, dense=DOMINANT_ROWS), cap_band_bbox(frame)


def _hook_black(frame: np.ndarray) -> list[int]:
    return line_heights(frame, "white")


def audit_clip(
    mp4: Path, sidecar: dict, ass_path: Path | None, words: list[Word] | None = None
) -> list[CheckResult]:
    """Q1-Q8 plus Q4b (caption, hook and tag off the face) for one rendered clip. `words` are
    the clip's output-time spoken words; by default the sidecar's `caption_words`.
    Caption and hook events are rendered separately over plain canvases (black for text
    measurement, grey for boxes), so neither hides or inflates the other."""
    if words is None and sidecar.get("caption_words") is not None:
        words = [Word(start=s, end=e, text=t) for s, e, t in sidecar["caption_words"]]
    duration = probe_duration(mp4)
    layouts = sidecar.get("layout") or []
    steps = zoom_levels(sidecar)
    caps: list[_Event] = []
    hooks: list[_Event] = []
    cap_black: list[tuple[list[int], Box | None]] = []
    cap_grey: list[Box | None] = []
    hook_heights: list[list[int]] = []
    hook_grey: list[Box | None] = []
    cap_times: list[float] = []
    hook_times: list[float] = []
    if ass_path is not None and Path(ass_path).exists():
        header, events = _split_ass(Path(ass_path).read_text(encoding="utf-8"))
        caps = [e for _, e in events if e.style == "Cap"]
        hooks = [e for _, e in events if e.style == "Hook"]
        cap_times = caption_samples(caps, layouts)
        seam = sidecar.get("seam")
        until = min(2.7, seam - 0.2) if seam is not None and 0.3 < seam < 3.0 else 2.7
        title_times = [0.1, min(until, max(duration - 0.2, 0.1))]
        hook_times = [*title_times, *(e.mid for e in hooks if EARLIER_TAG in e.text)]
        with tempfile.TemporaryDirectory(prefix="hookline-audit-") as tmp:
            cap_ass, hook_ass = Path(tmp) / "cap.ass", Path(tmp) / "hook.ass"
            for path, style in ((cap_ass, "Cap"), (hook_ass, "Hook")):
                body = [ln for ln, e in events if e.style == style]
                path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
            cap_black = _measure_frames(cap_ass, cap_times, _cap_black)
            cap_grey = _measure_frames(cap_ass, cap_times, box_bbox, GREY)
            if hooks:
                hook_heights = _measure_frames(hook_ass, title_times, _hook_black)
                hook_grey = _measure_frames(hook_ass, hook_times, box_bbox, GREY)
    face_samples = [
        ("caption", t, box, _event_window(caps, t))
        for t, (_, box) in zip(cap_times, cap_black, strict=True)
    ]
    face_samples += [
        ("hook", t, box, _event_window(hooks, t))
        for t, box in zip(hook_times, hook_grey, strict=True)
    ]
    return [
        _check_loudness(mp4),
        _check_caption_size([hs for hs, _ in cap_black], bool(words)),
        _check_coverage(caps, words),
        _check_uncaptioned(sidecar),
        _check_safe_zone([*cap_grey, *hook_grey]),
        _check_face(face_samples, layouts, steps),
        _check_hook(hooks, hook_heights, duration, sidecar.get("seam")),
        _check_opening(
            words,
            sidecar.get("declared_first_words") or sidecar.get("first_words") or "",
            sidecar.get("seam"),
            declared=bool(sidecar.get("declared_first_words")),
            moved=sidecar.get("opener_moved"),
            actual_first_words=sidecar.get("first_words") or "",
        ),
        _check_pacing(sidecar, duration),
        _check_encode(_probe(mp4), sidecar.get("source_video_kbps")),
        *_stream_checks(mp4, sidecar, words or [], duration),
    ]


# --- stream editing checks ----------------------------------------------------------------------

STATIC_MAX_S = 0.5  # Q-static: voiceless static / death / loading screen frames allowed
STATIC_FPS = 4
STATIC_ROWS = 0.45  # the frame from this far down (a STACKED clip's gameplay panel)
VOICELESS_WARN_S = 6.0  # Q-voiceless: a wordless span longer than this
# Q-seam: the output's rise at the seam (`media.audio.seam_jump_db`) over the source's own
# rise at the same source point (the sidecar's `seam_source_db`) more than this: a step / pop
SEAM_JUMP_DB = 12.0
# Q-beat: no full-cam beat over [scare - SCARE_PRE_S, scare + SCARE_POST_S] (layout's beat
# avoidance spans) nor over a recorded gameplay spike (the sidecar's `beat_spikes`)
BEAT_BEFORE_S, BEAT_AFTER_S = SCARE_PRE_S, SCARE_POST_S


def screen_frame_flags(mp4: Path, fps: int = STATIC_FPS) -> tuple[list[bool], list[float]]:
    """Per frame of `mp4` at `fps`, in the frame's lower part (from STATIC_ROWS down: a
    STACKED clip's gameplay): (a screen frame -- `motion.screen_frames`: near uniform or near
    white --, its mean chroma `motion.frame_chroma`)."""
    from hookline.media.motion import frame_chroma, screen_frames

    w, h = 36, 64
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(mp4), "-an",
         "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=yuv444p", "-f", "rawvideo", "-"],
        check=False, capture_output=True,
    )  # fmt: skip
    if res.returncode != 0:
        return [], []
    size = 3 * w * h
    raw = np.frombuffer(res.stdout[: len(res.stdout) // size * size], np.uint8)
    top = round(STATIC_ROWS * h)
    planes = raw.reshape(-1, 3, h, w)[:, :, top:, :].reshape(-1, 3, (h - top) * w)
    grey = planes[:, 0].astype(np.int16)
    chroma = frame_chroma(planes[:, 1], planes[:, 2])
    return [bool(f) for f in screen_frames(grey)], [float(c) for c in chroma]


def _covered_s(words: list[Word], lo: float, hi: float) -> float:
    """Seconds of [lo, hi] under at least one word (overlapping words counted once)."""
    total, cur = 0.0, lo
    for a, b in sorted((max(w.start, lo), min(w.end, hi)) for w in words if w.end > lo):
        a = max(a, cur)
        if b > a:
            total += b - a
            cur = b
    return total


def _check_static(
    flags: list[bool],
    fps: float,
    words: list[Word],
    chroma: list[float] | None = None,
    hidden: list[tuple[float, float]] | None = None,
) -> CheckResult:
    """Q-static: seconds of static / death / loading screen frames (`screen_frame_flags`; runs,
    single frames bridged, under words for at most STATIC_VOICED_MAX of their length and,
    with `chroma`, of mean chroma under STATIC_CHROMA_MAX: dark gameplay is coloured) at most
    STATIC_MAX_S. Frames inside the `hidden` spans (output s: full-cam covers, the screen not
    shown) are not counted."""
    thr = f"<= {STATIC_MAX_S:g}s of voiceless screen frames"
    runs: list[list[int]] = []
    for i, f in enumerate(flags):
        if not f or any(a <= (i + 0.5) / fps < b for a, b in hidden or []):
            continue
        if runs and i - runs[-1][1] <= 2:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    total, where = 0.0, []
    for a, b in runs:
        lo, hi = a / fps, (b + 1) / fps
        if _covered_s(words, lo, hi) > STATIC_VOICED_MAX * (hi - lo):
            continue
        if chroma and float(np.mean(chroma[a : b + 1])) >= STATIC_CHROMA_MAX:
            continue
        total += hi - lo
        where.append(f"{lo:.2f}-{hi:.2f}")
    detail = f"screens at {', '.join(where)}" if where else "none"
    return CheckResult("Q-static", total <= STATIC_MAX_S + 1e-6, round(total, 3), thr, detail)


def _check_voiceless(words: list[Word], duration: float) -> CheckResult:
    """Q-voiceless (a warning): the longest span without a caption word (clip start and end
    included) is at most VOICELESS_WARN_S."""
    thr = f"<= {VOICELESS_WARN_S:g}s without speech"
    if not words:
        return CheckResult("Q-voiceless", True, None, thr, "no speech")
    spans, cur = [], 0.0
    for w in sorted(words, key=lambda w: w.start):
        if w.start > cur:
            spans.append((cur, w.start))
        cur = max(cur, w.end)
    if duration > cur:
        spans.append((cur, duration))
    a, b = max(spans, key=lambda g: g[1] - g[0], default=(0.0, 0.0))
    detail = f"longest {b - a:.1f}s at {a:.2f}-{b:.2f}"
    warn = b - a > VOICELESS_WARN_S
    return CheckResult("Q-voiceless", True, round(b - a, 3), thr, detail, warn=warn)


def _check_seam(
    pcm: np.ndarray, rate: int, seam: float | None, source_db: float | None = None
) -> CheckResult:
    """Q-seam: no audio step or pop at the cold-open seam: the output's rise there
    (`seam_jump_db`) over `source_db`, the source's own rise at the same source point (a bang
    starting just after the cut, a bass bed's ripple), at most SEAM_JUMP_DB. Without
    `source_db` (a render before it was recorded), the output's rise alone."""
    thr = f"<= {SEAM_JUMP_DB:g} dB rise per {SEAM_FRAME_S * 1000:g} ms over the source's"
    if seam is None:
        return CheckResult("Q-seam", True, None, thr, "no cold open")
    jump = seam_jump_db(pcm, rate, seam)
    excess = max(jump - (source_db or 0.0), 0.0)
    src = "" if source_db is None else f" (source {source_db:.1f} dB)"
    detail = f"{jump:.1f} dB at {seam:.3f}s{src}"
    return CheckResult("Q-seam", excess <= SEAM_JUMP_DB, round(excess, 2), thr, detail)


def _check_beat(side: dict) -> CheckResult:
    """Q-beat: no full-cam beat (`layout[].cam_beats`, output s) over a scare's window
    [scare - BEAT_BEFORE_S, scare + BEAT_AFTER_S] (`scare_times`: game transients and flash
    scares, output s) or a gameplay spike (`beat_spikes`, output spans, already padded)."""
    thr = f"no cam beat within -{BEAT_BEFORE_S:g}/+{BEAT_AFTER_S:g}s of a scare or on a spike"
    scares = side.get("scare_times")
    if scares is None:
        return CheckResult("Q-beat", True, None, thr, "scares not recorded (an older render)")
    spikes = side.get("beat_spikes") or []
    beats = [b for lay in side.get("layout") or [] for b in lay.get("cam_beats") or []]
    hits = [
        f"{a:.2f}-{b:.2f} over scare {s:.2f}"
        for a, b in beats
        for s in scares
        if a < s + BEAT_AFTER_S and b > s - BEAT_BEFORE_S
    ]
    hits += [
        f"{a:.2f}-{b:.2f} over spike {x:.2f}-{y:.2f}"
        for a, b in beats
        for x, y in spikes
        if a < y and b > x
    ]
    summary = f"{len(beats)} beats, {len(scares)} scares, {len(spikes)} spikes"
    return CheckResult("Q-beat", not hits, len(hits), thr, "; ".join(hits) or summary)


def _clip_pcm(mp4: Path, rate: int = 48000) -> np.ndarray:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1",
         "-ar", str(rate), "-f", "s16le", "-"],
        check=False, capture_output=True,
    )  # fmt: skip
    return np.frombuffer(res.stdout, np.int16).astype(np.float64) / 32768.0


def _stream_hidden(side: dict) -> list[tuple[float, float]]:
    """Output spans whose gameplay is not shown: the full-cam covers over static screens (a
    beat's band still shows the gameplay)."""
    return [
        (float(a), float(b))
        for lay in side.get("layout") or []
        for a, b in lay.get("cam_covers") or []
    ]


def _stream_checks(mp4: Path, side: dict, words: list[Word], duration: float) -> list[CheckResult]:
    """Q-static, Q-voiceless, Q-seam and Q-beat: the stream editing checks."""
    seam = side.get("seam")
    pcm = _clip_pcm(mp4) if seam is not None else np.zeros(0)
    flags, chroma = screen_frame_flags(mp4)
    return [
        _check_static(flags, STATIC_FPS, words, chroma, hidden=_stream_hidden(side)),
        _check_voiceless(words, duration),
        _check_seam(pcm, 48000, seam, side.get("seam_source_db")),
        _check_beat(side),
    ]


def _ass_for(mp4: Path, side: dict, work_dir: Path) -> tuple[Path | None, CheckResult | None]:
    """The clip's ASS file (the sidecar's recorded `ass`, else `render/NN.ass`) and, when the
    sidecar records a hash, a failed "ASS" check if the file is missing or changed."""
    name = side.get("ass") or f"{mp4.stem.split('-', 1)[0]}.ass"
    ass = work_dir / "render" / name
    want = side.get("ass_sha1")
    if not ass.exists():
        bad = CheckResult("ASS", False, None, "recorded ASS present", f"{ass} missing")
        return None, bad if want else None
    if want and hashlib.sha1(ass.read_bytes()).hexdigest() != want:
        detail = f"{ass.name} changed since the render (another clip or job overwrote it?)"
        return ass, CheckResult("ASS", False, None, "ASS matches the render's sha1", detail)
    return ass, None


def audit_job(out_dir: Path, work_dir: Path) -> dict:
    """Audit every `NN-*.mp4` in `out_dir` against its sidecar and the ASS file the sidecar
    records (verified by sha1; `render/NN.ass` for older sidecars), and write the report to
    `out_dir/audit.json`. A clip whose audit raises is reported as a failed "ERR" check.
    A sidecar without `source_video_kbps` (rendered before Q8 followed the source) gets the
    job's workspace source probed for it (the 5000 kb/s bar when there is none)."""
    clips = []
    source_kbps: list[float | None] = []  # probed once, on the first sidecar that needs it
    for mp4 in sorted(out_dir.glob("*.mp4")):
        if mp4.name.endswith(".tmp.mp4"):
            continue
        ass: Path | None = None
        try:
            side_path = mp4.with_suffix(".json")
            side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}
            if not side.get("source_video_kbps"):
                if not source_kbps:
                    src = find_source(work_dir) if work_dir.is_dir() else None
                    source_kbps.append(probe_video_kbps(src) if src else None)
                side = {**side, "source_video_kbps": source_kbps[0]}
            ass, ass_check = _ass_for(mp4, side, work_dir)
            results = ([ass_check] if ass_check else []) + audit_clip(mp4, side, ass)
        except Exception as e:  # noqa: BLE001 - one broken clip must not stop the audit
            msg = (str(e).splitlines() or [""])[0][:300]
            results = [
                CheckResult("ERR", False, None, "audit completes", f"{type(e).__name__}: {msg}")
            ]
        clips.append(
            {
                "clip": mp4.name,
                "ass": str(ass) if ass else None,
                "ok": all(r.ok for r in results),
                "warnings": sum(r.status == "warn" for r in results),
                "checks": [r.as_dict() for r in results],
            }
        )
    report = {
        "ok": bool(clips) and all(c["ok"] for c in clips),
        "warnings": sum(c["warnings"] for c in clips),
        "clips": clips,
    }
    atomic_write_text(out_dir / "audit.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report
