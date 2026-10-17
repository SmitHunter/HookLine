"""Measurements on rendered frames, and the automated M2.5 quality bar (`hookline audit`)."""

import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from hookline.captions.ass import face_rects
from hookline.fsutil import atomic_write_text
from hookline.media.ffmpeg import probe_duration
from hookline.media.loudness import measure_loudness
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import Word
from hookline.pipeline.boundaries import FILLER_OPENERS
from hookline.pipeline.layout import SceneLayout

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


# --- The quality bar (Q1-Q8, plus Q4b: captions off the face) -----------------------------

LUFS_TARGET, LUFS_TOL, TP_MAX = -14.0, 1.0, -1.0
CAP_MIN_PX = 55
HOOK_MIN_PX = 60
HOOK_MAX_LINES, HOOK_MAX_WORDS, HOOK_UNTIL = 2, 7, 3.0
COVERAGE_MIN = 0.95
SAFE_X, SAFE_Y = (60, 888), (290, 1250)
FACE_OVERLAP_MAX = 0.10
SPEECH_START_MAX = 0.5
FIRST_WORDS_MIN = 0.6
PACE_MAX_GAP = 5.0
MIN_VIDEO_KBPS = 5000
# AAC is encoded at 192k; the measured average undershoots on simple audio (real clips 176-189k).
AUDIO_KBPS, AUDIO_KBPS_RANGE = 192, (160, 200)
MAX_CAPTION_SAMPLES = 12
SAMPLE_FPS = 25
EARLIER_TAG = "EARLIER"  # the cold-open seam tag (hook style), not the hook title

_TIME_RE = re.compile(r"(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_TAG_RE = re.compile(r"\{[^}]*\}")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class CheckResult:
    id: str
    ok: bool
    value: float | str | None
    threshold: str
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Event:
    start: float
    end: float
    style: str
    text: str  # override tags stripped; "\N" kept as the line break

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


def line_heights(frame: np.ndarray, color: TextColor = "any") -> list[int]:
    """Cap height (px) of each text line in `frame`: row runs of text pixels separated by
    empty rows, ignoring sparse rows (comma tails) under 5% of the run's densest row."""
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
        dense = np.nonzero(run >= max(1, 0.05 * run.max()))[0]
        heights.append(int(dense[-1] - dense[0] + 1))
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
            body = _TAG_RE.sub("", parts[9]).replace("⁠", "")
            events.append((ln, _Event(_secs(parts[1]), _secs(parts[2]), parts[3].strip(), body)))
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


def _spread[T](items: list[T], n: int) -> list[T]:
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _excursion(box: Box) -> int:
    """How far (px) a box pokes out of the Q4 safe zone; 0 when inside."""
    x0, y0, x1, y1 = box
    return max(0, SAFE_X[0] - x0, x1 - SAFE_X[1], SAFE_Y[0] - y0, y1 - SAFE_Y[1])


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


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


def _check_encode(info: dict) -> CheckResult:
    thr = f"1080x1920 h264 yuv420p, video >= {MIN_VIDEO_KBPS} kb/s, aac {AUDIO_KBPS}k"
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
    if v_kbps < MIN_VIDEO_KBPS:
        problems.append(f"video {v_kbps:.0f} kb/s")
    if (
        a is None
        or a.get("codec_name") != "aac"
        or not AUDIO_KBPS_RANGE[0] <= a_kbps <= AUDIO_KBPS_RANGE[1]
    ):
        problems.append(f"audio {a.get('codec_name') if a else 'none'} {a_kbps:.0f}k")
    detail = f"{size} {v.get('codec_name')} {v.get('pix_fmt')} {v_kbps:.0f} kb/s, aac {a_kbps:.0f}k"
    if problems:
        detail += " | " + "; ".join(problems)
    return CheckResult("Q8", not problems, round(v_kbps), thr, detail)


def _check_caption_size(frames: list[np.ndarray], has_speech: bool) -> CheckResult:
    thr = f"cap height >= {CAP_MIN_PX}px"
    heights = [min(hs) for hs in (line_heights(f) for f in frames) if hs]
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


def _check_safe_zone(frames: list[np.ndarray]) -> CheckResult:
    thr = f"x in {list(SAFE_X)}, y in {list(SAFE_Y)}"
    boxes = [b for b in (box_bbox(f) for f in frames) if b is not None]
    worst = max(boxes, key=_excursion, default=None)
    if worst is None:
        return CheckResult("Q4", True, 0, thr, "nothing drawn")
    out = _excursion(worst)
    return CheckResult("Q4", out == 0, out, thr, f"{len(boxes)} boxes; worst {worst}")


def _check_face(times: list[float], frames: list[np.ndarray], layouts: list[dict]) -> CheckResult:
    """Q4b: share of the caption text box covering the layout's face box (output px)."""
    worst, worst_t = 0.0, None
    for t, frame in zip(times, frames, strict=True):
        box = text_bbox(frame)
        lay = _layout_at(layouts, t)
        if box is None or lay is None:
            continue
        x0, y0, x1, y1 = box
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        for fx0, fy0, fx1, fy1 in face_rects(lay):
            ov = max(0.0, min(x1, fx1) - max(x0, fx0)) * max(0.0, min(y1, fy1) - max(y0, fy0))
            if ov / area > worst:
                worst, worst_t = ov / area, t
    detail = f"worst {worst:.0%} at {worst_t:.1f}s" if worst_t is not None else "no overlap"
    thr = f"caption/face overlap <= {FACE_OVERLAP_MAX:.0%}"
    return CheckResult("Q4b", worst <= FACE_OVERLAP_MAX, round(worst, 3), thr, detail)


def _check_hook(hooks: list[_Event], frames: list[np.ndarray], duration: float) -> CheckResult:
    """Q5 from the hook event at 0 s and its frames rendered over black at 0.1 s and 2.7 s."""
    thr = (
        f"<= {HOOK_MAX_LINES} lines, <= {HOOK_MAX_WORDS} words, cap >= {HOOK_MIN_PX}px, "
        f"0-{HOOK_UNTIL:g}s"
    )
    hook = next((e for e in hooks if e.start <= 0.01 and EARLIER_TAG not in e.text), None)
    if hook is None:
        return CheckResult("Q5", False, None, thr, "no hook title at 0 s")
    lines = [ln for ln in hook.text.split("\\N") if ln.strip()]
    n_words = len(" ".join(lines).split())
    per_frame = [line_heights(f, "white") for f in frames]
    cap = min((h for hs in per_frame for h in hs), default=0)
    problems = []
    if len(lines) > HOOK_MAX_LINES:
        problems.append(f"{len(lines)} lines")
    if n_words > HOOK_MAX_WORDS:
        problems.append(f"{n_words} words")
    if cap < HOOK_MIN_PX:
        problems.append(f"cap {cap}px")
    if hook.end < min(HOOK_UNTIL, duration) - 0.01 or not all(per_frame):
        problems.append("not visible for the first 3 s")
    detail = f"{len(lines)} line(s), {n_words} words, cap {cap}px, {hook.start:.1f}-{hook.end:.1f}s"
    if problems:
        detail += " | " + "; ".join(problems)
    return CheckResult("Q5", not problems, cap, thr, detail)


def _check_opening(words: list[Word] | None, first_words: str, seam: float | None) -> CheckResult:
    """Q6: speech by 0.5 s, and the opening (after a cold-open seam, where the declared first
    sentence begins) matches `first_words`, leading filler openers ignored on both sides."""
    thr = f"speech <= {SPEECH_START_MAX:g}s, first_words overlap >= {FIRST_WORDS_MIN:.0%}"
    if words is None:
        return CheckResult("Q6", False, None, thr, "no caption_words in sidecar (re-render)")
    if not words:
        return CheckResult("Q6", False, None, thr, "no speech")
    start = words[0].start
    main = [w for w in words if seam is None or w.start >= seam - 0.05]
    want = _strip_fillers(_tokens(first_words))
    have = _strip_fillers(_tokens(" ".join(w.text for w in main)))[: len(want) + 2]
    overlap = sum((Counter(want) & Counter(have)).values()) / len(want) if want else 0.0
    ok = start <= SPEECH_START_MAX and overlap >= FIRST_WORDS_MIN
    detail = f"speech at {start:.2f}s; first_words overlap {overlap:.0%}"
    if seam is not None:
        detail += f" after the seam at {seam:.2f}s"
    if not want:
        detail += " (no first_words declared)"
    return CheckResult("Q6", ok, round(start, 3), thr, detail)


def _check_pacing(side: dict, duration: float) -> CheckResult:
    """Q7: longest stretch without a cut, zoom step or scene (layout) change."""
    thr = f"framing change every <= {PACE_MAX_GAP:g}s"
    changes = {float(t) for t in side.get("cut_times") or []}
    changes |= {float(st[0]) for st in side.get("zoom_steps") or []}
    changes |= {float(lay["start"]) for lay in side.get("layout") or []}
    pts = sorted({0.0, duration} | {t for t in changes if 0 < t < duration})
    gap, at = max((b - a, a) for a, b in pairwise(pts))
    detail = f"longest {gap:.1f}s from {at:.1f}s"
    return CheckResult("Q7", gap <= PACE_MAX_GAP + 1e-6, round(gap, 3), thr, detail)


def audit_clip(
    mp4: Path, sidecar: dict, ass_path: Path | None, words: list[Word] | None = None
) -> list[CheckResult]:
    """Q1-Q8 plus Q4b (captions off the face) for one rendered clip. `words` are the clip's
    output-time spoken words; by default the sidecar's `caption_words` ([start, end, text]).
    Caption and hook events are rendered separately over plain canvases (black for text
    measurement, grey for boxes), so neither hides or inflates the other."""
    if words is None and sidecar.get("caption_words") is not None:
        words = [Word(start=s, end=e, text=t) for s, e, t in sidecar["caption_words"]]
    duration = probe_duration(mp4)
    caps: list[_Event] = []
    hooks: list[_Event] = []
    cap_black: list[np.ndarray] = []
    cap_grey: list[np.ndarray] = []
    hook_black: list[np.ndarray] = []
    hook_grey: list[np.ndarray] = []
    cap_times: list[float] = []
    if ass_path is not None and Path(ass_path).exists():
        header, events = _split_ass(Path(ass_path).read_text(encoding="utf-8"))
        caps = [e for _, e in events if e.style == "Cap"]
        hooks = [e for _, e in events if e.style == "Hook"]
        samples = _spread(
            [e for e in caps if any(c.isalnum() for c in e.text) and e.end - e.start >= 0.08],
            MAX_CAPTION_SAMPLES,
        )
        cap_times = [e.mid for e in samples]
        hook_times = [0.1, min(2.7, max(duration - 0.2, 0.1))]
        tag_times = [e.mid for e in hooks if EARLIER_TAG in e.text]
        with tempfile.TemporaryDirectory(prefix="hookline-audit-") as tmp:
            cap_ass, hook_ass = Path(tmp) / "cap.ass", Path(tmp) / "hook.ass"
            for path, style in ((cap_ass, "Cap"), (hook_ass, "Hook")):
                body = [ln for ln, e in events if e.style == style]
                path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
            cap_black = render_ass_frames(cap_ass, cap_times)
            cap_grey = render_ass_frames(cap_ass, cap_times, GREY)
            if hooks:
                hook_black = render_ass_frames(hook_ass, hook_times)
                hook_grey = render_ass_frames(hook_ass, [*hook_times, *tag_times], GREY)
    return [
        _check_loudness(mp4),
        _check_caption_size(cap_black, bool(words)),
        _check_coverage(caps, words),
        _check_safe_zone([*cap_grey, *hook_grey]),
        _check_face(cap_times, cap_black, sidecar.get("layout") or []),
        _check_hook(hooks, hook_black, duration),
        _check_opening(words, sidecar.get("first_words") or "", sidecar.get("seam")),
        _check_pacing(sidecar, duration),
        _check_encode(_probe(mp4)),
    ]


def audit_job(out_dir: Path, work_dir: Path) -> dict:
    """Audit every `NN-*.mp4` in `out_dir` against its sidecar and `work_dir/render/NN.ass`,
    and write the report to `out_dir/audit.json`."""
    clips = []
    for mp4 in sorted(out_dir.glob("*.mp4")):
        if mp4.name.endswith(".tmp.mp4"):
            continue
        side_path = mp4.with_suffix(".json")
        side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}
        ass = work_dir / "render" / f"{mp4.stem.split('-', 1)[0]}.ass"
        results = audit_clip(mp4, side, ass if ass.exists() else None)
        clips.append(
            {
                "clip": mp4.name,
                "ass": str(ass) if ass.exists() else None,
                "ok": all(r.ok for r in results),
                "checks": [r.as_dict() for r in results],
            }
        )
    report = {"ok": bool(clips) and all(c["ok"] for c in clips), "clips": clips}
    atomic_write_text(out_dir / "audit.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report
