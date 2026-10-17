"""Per-clip render: refine boundaries, drop dead air, reframe to 9:16 and burn captions."""

import hashlib
import logging
import os
import re
from pathlib import Path

from hookline.captions.ass import build_ass, caption_y_for, early_until, hook_y_for
from hookline.config import Settings
from hookline.fsutil import atomic_write_text
from hookline.media.audio import Envelope, detect_silences, energy_envelope
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
    face_offset,
    guard_zoom,
    plan_zoom,
    reframe_graph,
    rendered_zoom_steps,
    zoom_schedule,
)
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import ClipPick, Word
from hookline.pipeline.boundaries import refine_boundaries
from hookline.pipeline.deadair import keep_ranges, remap_words, snap_ranges
from hookline.pipeline.layout import SceneLayout, plan_layouts
from hookline.pipeline.timing import (
    protected_words,
    refine_word_ends,
    snap_onsets,
    tail_end,
    voice_threshold,
)
from hookline.vision.faces import Crop, FaceDetector

INTERMEDIATE_QUALITY = ["-preset", "veryfast", "-crf", "12"]
MAX_HORIZONTAL_W = 1920
_ZOOM_BEAT_RE = re.compile(r"^\s*zoom\s*:.*@\s*(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?")
CAM_EDGE_LO, CAM_EDGE_HI = 0.02, 0.98  # a cam box this close to a cropped edge was cut by it
SEAM_FLASH = (0.35, 0.6, 0.35)  # white opacity of the 3 frames around the cold-open seam
SEAM_SLACK_S = 0.1  # frame snapping may move a cold-open range bound by up to a frame
FADE_IN_S, FADE_OUT_S = 0.02, 0.12  # audio fades at the clip's start and end (no clicks)

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
    """3-frame white flash centred on the frame at `seam`: fade in, 60% white, fade out."""
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


def audio_fades(duration: float) -> str:
    """FADE_IN_S fade-in at the start and FADE_OUT_S fade-out ending at `duration`."""
    return (
        f"afade=t=in:st=0:d={FADE_IN_S:g},"
        f"afade=t=out:st={max(duration - FADE_OUT_S, 0.0):.3f}:d={FADE_OUT_S:g}"
    )


def _with_tail(
    keep: list[tuple[float, float]], words: list[Word], env: Envelope, voice_db: float
) -> list[tuple[float, float]]:
    """`keep` with its last range ending a beat after the last word (`timing.tail_end`):
    into the silence after it, at most MAX_TAIL_S past it, never into the next word. The clip
    may then run up to MAX_TAIL_S over max_clip_s."""
    if not keep or not words:
        return keep
    start, end = keep[-1]
    spoken = [w for w in words if w.start < end and w.end > start]
    if not spoken:
        return keep
    last = max(w.end for w in spoken)
    later = min((w for w in words if w.start >= end), key=lambda w: w.start, default=None)
    nxt = (later.start, later.end) if later else (float("inf"), float("inf"))
    new_end = tail_end(last, end, nxt[0], env, voice_db, next_end=nxt[1])
    return [*keep[:-1], (start, round(new_end, 3))]


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
    removed_s, cut_times, zoom_steps, zoom_levels, overlay_bands, seam, ass, ass_sha1,
    caption_words). Cut and zoom times are on the output timeline; `zoom_steps` are the zooms
    as rendered (each scene's plan applied), `zoom_levels` the schedule they come from. Zooms
    are applied before captions, so captions never zoom (vertical only; empty when
    horizontal). Each `layout` scene carries its zoom plan (`base_zoom`, `zoom_cap`,
    `zoom_anchor`; see `reframe.plan_zoom`).

    Word ends and post-pause onsets are refined against the audio envelope first
    (`pipeline.timing`); the clip ends a beat after its last word (`_with_tail`, up to 0.35 s,
    so a clip may run that much over max_clip_s) and its audio fades in over 0.02 s and out
    over 0.12 s.
    `overlay_bands` is the [top, bottom] px of burned-in overlay cropped away (vertical only),
    or None. `seam` is the output time where a cold open ends (None without one); it gets a
    3-frame 60% white flash and the "EARLIER…" tag. `ass`/`ass_sha1` name the burned caption file
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
    # word ends/onsets from the audio (DTW gives single-token words a synthetic 0.15 s end)
    env = energy_envelope(wav, lo, hi + 1)
    voice_db = voice_threshold(env, settings.silence_db)  # from the local noise floor
    raw = [w for w in words if w.end > lo and w.start < hi]
    near = raw
    if voice_db is None:
        log.warning(
            "%s: background noise near the silence level; word timings and end tail kept "
            "from the transcript", name,
        )  # fmt: skip
    else:
        near = snap_onsets(refine_word_ends(raw, env, voice_db), env, voice_db)
    refined = refine_boundaries(pick, near, silences, loud_seconds=loud_seconds)
    fps = probe_fps(src)
    # Frame-grid bounds make the cut timeline exact, so remapped caption times stay in sync.
    untailed = keep_ranges(
        refined.segments(),
        silences,
        settings.max_pause_s,
        settings.dead_air_pad_s,
        # dead air trusts a refined word end only so far past the transcript's
        words=protected_words(raw, near) if protect_words else None,
    )
    tailed = untailed if voice_db is None else _with_tail(untailed, near, env, voice_db)
    keep = snap_ranges(tailed, fps)
    # words starting in the tail (no audible onset: see `tail_end`) are not captioned
    spoken = [w for w in near if w.start < untailed[-1][1] - 1e-6] if untailed else near

    inter = cut_segments(
        src, keep, work_dir / f"{name}.mkv", "libx264", quality=INTERMEDIATE_QUALITY, fps=fps
    )
    loudness = measure_loudness(inter, COMPRESSOR)
    duration = probe_duration(inter)
    audio_args: list[str] = []
    if loudness is not None:  # the clip has audio
        audio_args = ["-af", f"{loudnorm_filter(loudness)},{audio_fades(duration)}"]
    letterbox = detect_crop(inter)
    size = probe_size(inter)
    crop, bands = letterbox, None

    out_words = remap_words(spoken, keep)
    cuts = cut_times_for(keep)
    layouts: list[SceneLayout] = []
    steps: list[tuple[float, float]] = []
    if settings.vertical:
        layouts, crop, bands = _plan_vertical(inter, detector, settings, size, letterbox)
        in_w, in_h = (crop[0], crop[1]) if crop else size
        # per-scene zoom plan (soft-source cap, small-face base zoom, headroom); the layouts
        # carry it into the sidecar and to caption placement
        layouts = [plan_zoom(lay, in_w, in_h) for lay in layouts]
        steps = zoom_schedule(
            duration,
            cuts,
            emphasis_times(pick.edit_beats, keep),
            boundaries=[w.end for w in out_words if w.text.rstrip().endswith(_SENTENCE_END)],
            changes=[lay.start for lay in layouts[1:]],  # no beat step next to a camera change
        )
        # the edge guard's holds/shifts, once: the graph and the sidecar get the same steps
        steps, layouts = guard_zoom(layouts, in_w, in_h, steps, cuts)
        layouts = [
            lay.model_copy(update={"face_dx": face_offset(lay, in_w, in_h, steps)})
            for lay in layouts
        ]
        graph = reframe_graph(layouts, in_w, in_h, crop=crop, zoom_steps=steps, cut_times=cuts)
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
        # the early line and the hook/tag position see every scene shown with hook or tag
        until = early_until(out_words, pick.hook_text, duration, seam, breaks=cuts)
        text = build_ass(
            out_words,
            pick.hook_text,
            duration,
            caption_y_for(layouts, until=until),
            hook_y_for(layouts, until=until),
            seam=seam,
            caption_y_late=caption_y_for(layouts, late=True),
            breaks=cuts,  # caption pages never span a cut (nor the seam)
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
    segments = [list(sg) for sg in refined.segments()]
    if segments and keep and keep[-1][1] > segments[-1][1]:
        segments[-1][1] = keep[-1][1]  # the end tail is part of the clip's span
    return {
        "segments": segments,
        "layout": [lay.model_dump(mode="json") for lay in layouts],
        "keep_ranges": [list(k) for k in keep],
        "removed_s": round(max(sum(b - a for a, b in segments) - kept, 0.0), 3),
        "cut_times": cuts,
        # as rendered (guard holds applied, each scene's zoom plan); `zoom_levels` the levels
        "zoom_steps": [list(st) for st in rendered_zoom_steps(layouts, steps)] if layouts else [],
        "zoom_levels": [list(st) for st in steps],
        "overlay_bands": list(bands) if bands else None,
        "seam": seam,
        **ass_ref,
        "caption_words": [[w.start, w.end, w.text] for w in out_words],
    }
