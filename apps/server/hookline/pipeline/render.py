"""Per-clip render: refine boundaries, drop dead air, reframe to 9:16 and burn captions."""

import hashlib
import logging
import os
import re
from pathlib import Path

from hookline.captions.ass import build_ass, caption_y_for, hook_y_for
from hookline.config import Settings
from hookline.fsutil import atomic_write_text
from hookline.media.audio import detect_silences
from hookline.media.ffmpeg import (
    _run,
    cut_segments,
    encoder_args,
    filter_graph_args,
    pick_encoder,
    probe_duration,
    probe_fps,
    probe_size,
)
from hookline.media.loudness import (
    COMPRESSOR,
    ensure_true_peak,
    loudnorm_filter,
    loudnorm_mode,
    measure_loudness,
)
from hookline.media.reframe import (
    box_hits_bands,
    box_to_crop,
    combine_crop,
    detect_crop,
    detect_overlay_bands,
    reframe_graph,
    zoom_schedule,
)
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import ClipPick, Word
from hookline.pipeline.boundaries import refine_boundaries
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges
from hookline.pipeline.layout import SceneLayout, plan_layouts
from hookline.vision.faces import Crop, FaceDetector

INTERMEDIATE_QUALITY = ["-preset", "veryfast", "-crf", "12"]
MAX_HORIZONTAL_W = 1920
_ZOOM_BEAT_RE = re.compile(r"^\s*zoom\s*:.*@\s*(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?")
CAM_EDGE_LO, CAM_EDGE_HI = 0.02, 0.98  # a cam box this close to a cropped edge was cut by it
SEAM_FLASH = (0.6, 1.0, 0.6)  # white opacity of the 3 frames around the cold-open seam
SEAM_SLACK_S = 0.1  # frame snapping may move a cold-open range bound by up to a frame

log = logging.getLogger(__name__)


def cut_times_for(keep: list[tuple[float, float]]) -> list[float]:
    """Output-time cut points: the joins between consecutive keep ranges."""
    out, t = [], 0.0
    for start, end in keep[:-1]:
        t += end - start
        out.append(round(t, 3))
    return out


def seam_time(
    keep: list[tuple[float, float]], cold_open: tuple[float, float] | None
) -> float | None:
    """Output time where the cold open ends: the length of the leading keep ranges that lie
    inside the cold open's source span. None without a cold open, or when nothing of it is
    kept or nothing follows it."""
    if cold_open is None:
        return None
    lo, hi = cold_open[0] - SEAM_SLACK_S, cold_open[1] + SEAM_SLACK_S
    t, n = 0.0, 0
    for start, end in keep:
        if not (lo <= start and end <= hi):
            break
        t += end - start
        n += 1
    return round(t, 3) if 0 < n < len(keep) else None


def seam_flash(seam: float, fps: float) -> str:
    """3-frame white flash centred on the frame at `seam`: fade in, full white, fade out."""
    boxes = []
    for k, alpha in zip((-1, 0, 1), SEAM_FLASH, strict=True):
        a, b = seam + (k - 0.5) / fps, seam + (k + 0.5) / fps
        boxes.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white@{alpha:g}:t=fill"
            f":enable='between(t\\,{a:.4f}\\,{b:.4f})'"
        )
    return ",".join(boxes)


def _to_output_time(t: float, keep: list[tuple[float, float]]) -> float | None:
    offset = 0.0
    for start, end in keep:
        if start <= t < end:
            return round(offset + t - start, 3)
        offset += end - start
    return None


def emphasis_times(beats: list[str], keep: list[tuple[float, float]]) -> list[float]:
    """Output times of `zoom: '<word>' @ <t>s` beats (absolute source seconds), mapped through
    the keep ranges; beats in removed ranges, without a time, or of other kinds are dropped."""
    out = []
    for beat in beats:
        m = _ZOOM_BEAT_RE.match(beat)
        t = _to_output_time(float(m.group(1)), keep) if m else None
        if t is not None:
            out.append(t)
    return out


def _horizontal_graph(crop: tuple[int, int, int, int] | None) -> str:
    pre = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else ""
    return f"[0:v]{pre}scale='trunc(min({MAX_HORIZONTAL_W},iw)/2)*2':-2,setsar=1[vout]"


def _plan_vertical(
    inter: Path,
    detector: FaceDetector | None,
    settings: Settings,
    size: tuple[int, int],
    letterbox: Crop | None,
) -> tuple[list[SceneLayout], Crop | None, tuple[int, int] | None]:
    """(layouts, crop, overlay bands) for the vertical reframe. Burned-in lower-thirds/edge
    overlays (would show cut mid-word) are cropped away unless disabled, refused by
    `combine_crop`, or they'd cut into the facecam: an override box overlapping a band, or an
    auto-detected STACKED cam touching the cropped edge (it continues into the band), keeps
    the letterbox-only crop. The facecam override (full-frame fractions) is mapped into the
    crop's space."""
    bands = detect_overlay_bands(inter, crop=letterbox) if settings.overlay_crop else None
    crop = combine_crop(size, letterbox, bands)
    if crop == letterbox:
        bands = None  # none found, or removal refused (too much of the height)
    cam = settings.facecam
    if cam is not None and bands and box_hits_bands(cam, size, letterbox, bands):
        log.info("facecam %s overlaps overlay bands %s; bands kept", cam, bands)
        crop, bands = letterbox, None
    override = box_to_crop(cam, size, crop) if cam is not None else None
    layouts = plan_layouts(inter, detector, override, crop=crop)
    if bands and cam is None and any(_cam_on_cut(lay, bands) for lay in layouts):
        log.info("facecam touches the overlay crop edge; re-planned without bands %s", bands)
        crop, bands = letterbox, None
        layouts = plan_layouts(inter, detector, None, crop=crop)
    return layouts, crop, bands


def _cam_on_cut(layout: SceneLayout, bands: tuple[int, int]) -> bool:
    """A STACKED cam box touching an edge where an overlay band was cropped away."""
    if layout.kind != "STACKED" or layout.facecam is None:
        return False
    _, y, _, h = layout.facecam
    return (bands[1] > 0 and y + h >= CAM_EDGE_HI) or (bands[0] > 0 and y <= CAM_EDGE_LO)


def render_clip(
    src: Path,
    wav: Path,
    pick: ClipPick,
    words: list[Word],
    work_dir: Path,
    dst: Path,
    settings: Settings,
    detector: FaceDetector | None,
    protect_words: bool = True,
    encoder: str | None = None,
    loud_seconds: set[int] | None = None,
) -> dict:
    """Write the finished clip to `dst`; returns sidecar extras (segments, layout, keep_ranges,
    removed_s, cut_times, zoom_steps, overlay_bands, seam, ass, ass_sha1, caption_words). Cut
    and zoom times are on the output timeline; zooms are applied before captions, so captions
    never zoom (vertical only; empty when horizontal).
    `overlay_bands` is the [top, bottom] px of burned-in overlay cropped away (vertical only),
    or None. `seam` is the output time where a cold open ends (None without one); it gets a
    3-frame white flash and the "EARLIER…" tag. `ass`/`ass_sha1` name the burned caption file
    in `work_dir` and its hash (None without captions). `caption_words` are the spoken words
    on the output timeline, [start, end, text], so `hookline audit` needs no transcript.

    The intermediate cut (`work_dir/NN.mkv`, near-lossless libx264 + PCM audio, deleted after success)
    and the ASS file live in `work_dir`. Video is lossily encoded once (the final encode);
    audio once (AAC 192k, after compression and two-pass loudnorm to -14 LUFS).

    `protect_words` keeps dead-air cuts out of word spans; pass False for
    segment-interpolated word times, which tile the timeline with no gaps. `encoder` is the
    final video encoder (resolved by `pick_encoder()` when None; pass it to avoid re-probing)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    name = dst.stem.split("-", 1)[0]

    segs = pick.segments()
    lo = max(0.0, min(s for s, _ in segs) - 1)
    hi = max(e for _, e in segs) + 1
    silences = detect_silences(wav, lo, hi, settings.silence_db)
    near = [w for w in words if w.end > lo and w.start < hi]
    refined = refine_boundaries(pick, near, silences, loud_seconds=loud_seconds)
    fps = probe_fps(src)
    # Frame-grid bounds make the cut timeline exact, so remapped caption times stay in sync.
    keep = snap_ranges(
        keep_ranges(
            refined.segments(),
            silences,
            settings.max_pause_s,
            settings.dead_air_pad_s,
            words=near if protect_words else None,
        ),
        fps,
    )

    inter = cut_segments(
        src, keep, work_dir / f"{name}.mkv", "libx264", quality=INTERMEDIATE_QUALITY, fps=fps
    )
    loudness = measure_loudness(inter, COMPRESSOR)
    audio_args = ["-af", loudnorm_filter(loudness)] if loudness else []
    letterbox = detect_crop(inter)
    size = probe_size(inter)
    crop, bands = letterbox, None

    duration = probe_duration(inter)
    out_words = remap_words(near, keep)
    cuts = cut_times_for(keep)
    layouts: list[SceneLayout] = []
    steps: list[tuple[float, float]] = []
    if settings.vertical:
        layouts, crop, bands = _plan_vertical(inter, detector, settings, size, letterbox)
        in_w, in_h = (crop[0], crop[1]) if crop else size
        steps = zoom_schedule(
            duration,
            cuts,
            emphasis_times(pick.edit_beats, keep),
            boundaries=[w.end for w in out_words if w.text.rstrip().endswith(_SENTENCE_END)],
        )
        graph = reframe_graph(layouts, in_w, in_h, crop=crop, zoom_steps=steps)
    else:
        graph = _horizontal_graph(crop)

    seam = seam_time(keep, refined.cold_open)
    out_label = "[vout]"
    if seam is not None:
        graph += f";[vout]{seam_flash(seam, fps)}[vseam]"
        out_label = "[vseam]"
    ass_ref: dict = {"ass": None, "ass_sha1": None}
    if settings.captions:
        ass = work_dir / f"{name}.ass"
        text = build_ass(
            out_words,
            pick.hook_text,
            duration,
            caption_y_for(layouts),
            hook_y_for(layouts),
            seam=seam,
            caption_y_late=caption_y_for(layouts, late=True),
        )
        atomic_write_text(ass, text)
        ass_ref = {"ass": ass.name, "ass_sha1": hashlib.sha1(ass.read_bytes()).hexdigest()}
        graph += f";{out_label}{subtitles_filter(ass, fonts_dir())}[vfinal]"
        out_label = "[vfinal]"

    tmp = dst.with_suffix(".tmp.mp4")
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(inter),
            # always a script: zoom/track expressions can outgrow the Windows command line
            *filter_graph_args(graph, work_dir / f"{name}.graph.txt", force=True),
            "-map", out_label, "-map", "0:a?",
            *encoder_args(encoder or pick_encoder()), "-pix_fmt", "yuv420p",
            *audio_args,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",  # the one lossy audio encode
            "-movflags", "+faststart",
            str(tmp),
        ]
    )  # fmt: skip
    os.replace(tmp, dst)
    if loudness and loudnorm_mode(loudness) != "skip":
        ensure_true_peak(dst)
    inter.unlink(missing_ok=True)

    kept = sum(e - s for s, e in keep)
    return {
        "segments": [list(s) for s in refined.segments()],
        "layout": [lay.model_dump(mode="json") for lay in layouts],
        "keep_ranges": [list(k) for k in keep],
        "removed_s": round(max(refined.duration - kept, 0.0), 3),
        "cut_times": cuts,
        "zoom_steps": [list(st) for st in steps],
        "overlay_bands": list(bands) if bands else None,
        "seam": seam,
        **ass_ref,
        "caption_words": [[w.start, w.end, w.text] for w in out_words],
    }
