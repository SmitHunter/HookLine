"""Per-clip render: refine boundaries, drop dead air, reframe to 9:16 and burn captions."""

import os
from pathlib import Path

from hookline.captions.ass import build_ass, caption_y_for
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
from hookline.media.reframe import detect_crop, reframe_graph
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import ClipPick, Word
from hookline.pipeline.boundaries import refine_boundaries
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges
from hookline.pipeline.layout import SceneLayout, plan_layouts
from hookline.vision.faces import FaceDetector

INTERMEDIATE_QUALITY = ["-preset", "veryfast", "-crf", "12"]
MAX_HORIZONTAL_W = 1920


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
) -> dict:
    """Write the finished clip to `dst`; returns sidecar extras (layout, keep_ranges, removed_s).

    The intermediate cut (`work_dir/NN.mp4`, near-lossless libx264, deleted after success)
    and the ASS file live in `work_dir`. Video is lossily encoded once (the final encode);
    audio once (the intermediate's AAC, copied into the final file)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    name = dst.stem.split("-", 1)[0]

    segs = pick.segments()
    lo = max(0.0, min(s for s, _ in segs) - 1)
    hi = max(e for _, e in segs) + 1
    silences = detect_silences(wav, lo, hi, settings.silence_db)
    near = [w for w in words if w.end > lo and w.start < hi]
    refined = refine_boundaries(pick, near, silences)
    fps = probe_fps(src)
    # Frame-grid bounds make the cut timeline exact, so remapped caption times stay in sync.
    keep = snap_ranges(keep_ranges(refined.segments(), silences, settings.max_pause_s), fps)

    inter = cut_segments(
        src, keep, work_dir / f"{name}.mp4", "libx264", quality=INTERMEDIATE_QUALITY, fps=fps
    )
    crop = detect_crop(inter)
    in_w, in_h = (crop[0], crop[1]) if crop else probe_size(inter)

    layouts: list[SceneLayout] = []
    if settings.vertical:
        layouts = plan_layouts(inter, detector, settings.facecam, crop=crop)
        graph = reframe_graph(layouts, in_w, in_h, crop=crop)
    else:
        graph = _horizontal_graph(crop)

    out_label = "[vout]"
    if settings.captions:
        ass = work_dir / f"{name}.ass"
        cap_words = remap_words(near, keep)
        text = build_ass(cap_words, pick.hook_text, probe_duration(inter), caption_y_for(layouts))
        atomic_write_text(ass, text)
        graph += f";[vout]{subtitles_filter(ass, fonts_dir())}[vfinal]"
        out_label = "[vfinal]"

    tmp = dst.with_suffix(".tmp.mp4")
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(inter),
            "-filter_complex", graph,
            "-map", out_label, "-map", "0:a?",
            *encoder_args(pick_encoder()), "-pix_fmt", "yuv420p",
            "-c:a", "copy",  # the intermediate's AAC 192k is the one lossy audio encode
            "-movflags", "+faststart",
            str(tmp),
        ]
    )  # fmt: skip
    os.replace(tmp, dst)
    inter.unlink(missing_ok=True)

    kept = sum(e - s for s, e in keep)
    return {
        "segments": [list(s) for s in refined.segments()],
        "layout": [lay.model_dump(mode="json") for lay in layouts],
        "keep_ranges": [list(k) for k in keep],
        "removed_s": round(max(refined.duration - kept, 0.0), 3),
    }
