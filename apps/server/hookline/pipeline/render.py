"""Per-clip render: refine boundaries, drop dead air, reframe to 9:16 and burn captions."""

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
from hookline.media.reframe import detect_crop, reframe_graph, zoom_schedule
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import ClipPick, Word
from hookline.pipeline.boundaries import refine_boundaries
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges
from hookline.pipeline.layout import SceneLayout, plan_layouts
from hookline.vision.faces import FaceDetector

INTERMEDIATE_QUALITY = ["-preset", "veryfast", "-crf", "12"]
MAX_HORIZONTAL_W = 1920
_ZOOM_BEAT_RE = re.compile(r"^\s*zoom\s*:.*@\s*(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?")


def cut_times_for(keep: list[tuple[float, float]]) -> list[float]:
    """Output-time cut points: the joins between consecutive keep ranges."""
    out, t = [], 0.0
    for start, end in keep[:-1]:
        t += end - start
        out.append(round(t, 3))
    return out


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
    removed_s, cut_times, zoom_steps). Cut and zoom times are on the output timeline; zooms are
    applied before captions, so captions never zoom (vertical only; empty when horizontal).

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
    crop = detect_crop(inter)
    in_w, in_h = (crop[0], crop[1]) if crop else probe_size(inter)

    duration = probe_duration(inter)
    out_words = remap_words(near, keep)
    cuts = cut_times_for(keep)
    layouts: list[SceneLayout] = []
    steps: list[tuple[float, float]] = []
    if settings.vertical:
        layouts = plan_layouts(inter, detector, settings.facecam, crop=crop)
        steps = zoom_schedule(
            duration,
            cuts,
            emphasis_times(pick.edit_beats, keep),
            boundaries=[w.end for w in out_words if w.text.rstrip().endswith(_SENTENCE_END)],
        )
        graph = reframe_graph(layouts, in_w, in_h, crop=crop, zoom_steps=steps)
    else:
        graph = _horizontal_graph(crop)

    out_label = "[vout]"
    if settings.captions:
        ass = work_dir / f"{name}.ass"
        text = build_ass(
            out_words,
            pick.hook_text,
            duration,
            caption_y_for(layouts),
            hook_y_for(layouts),
        )
        atomic_write_text(ass, text)
        graph += f";[vout]{subtitles_filter(ass, fonts_dir())}[vfinal]"
        out_label = "[vfinal]"

    tmp = dst.with_suffix(".tmp.mp4")
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(inter),
            "-filter_complex", graph,
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
    }
