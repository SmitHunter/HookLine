"""Per-clip render: refine boundaries, drop dead air, reframe to 9:16 and burn captions."""

import hashlib
import logging
import os
import re
from itertools import pairwise
from pathlib import Path

from hookline.captions.ass import (
    SEAM_TAG_S,
    build_ass,
    caption_words,
    caption_y_for,
    early_until,
    face_samples,
    hook_y_for,
)
from hookline.config import Settings
from hookline.fsutil import atomic_write_text
from hookline.media.audio import Envelope, detect_silences, digital_zero_runs, energy_envelope
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
from hookline.media.loudness import ensure_true_peak, plan_loudness
from hookline.media.reframe import (
    box_hits_bands,
    box_to_crop,
    combine_crop,
    detect_crop,
    detect_overlay_bands,
    face_offset,
    face_track_px,
    guard_zoom,
    plan_zoom,
    reframe_graph,
    rendered_zoom_steps,
    scene_snaps,
    stacked_face_px,
    zoom_schedule,
)
from hookline.media.subtitles import fonts_dir, subtitles_filter
from hookline.models import ClipPick, Word
from hookline.pipeline.boundaries import refine_boundaries
from hookline.pipeline.deadair import cut_gaps, keep_ranges, remap_words, snap_ranges
from hookline.pipeline.layout import (
    SceneLayout,
    plan_cam_beats,
    plan_layouts,
    plan_profile_layouts,
)
from hookline.pipeline.profile import LayoutProfile
from hookline.pipeline.timing import (
    anchor_words,
    last_voiced,
    protected_words,
    refine_word_ends,
    snap_onsets,
    snap_splices,
    tail_end,
    uncaptioned_speech,
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
FADE_IN_S, FADE_OUT_S = 0.02, 0.2  # audio fades at the clip's start and end (no clicks)
# ... the end fade shortens to this at least to spare the last word; on a bed too noisy for
# the end tail the clip still runs this long past the last word (`min_tail`), so a loud game
# bed fades out instead of stopping dead
MIN_FADE_OUT_S = 0.15
# Room tone: a pink-noise bed this loud (dBFS RMS) under clips whose cut exposes digital
# silence (a source noise gate's exact zeros) at least ZERO_RUN_S long; too quiet to move
# the loudness. PINK_RMS_DB: anoisesrc pink noise at amplitude 1 (measured).
ROOM_TONE_DB = -62.0
PINK_RMS_DB = -14.3
ZERO_RUN_S = 0.2
CUT_SLIVER_S = 0.08  # a caption word this short next to a cut is a splice remnant: not shown
CUT_SLIVER_NEAR_S = 0.2  # ... its mid-point this close to the cut (further on: a real word)
MIX_RATE = 48000  # the room-tone mix runs at the encode's rate (see render_clip)
# Reaction peaks (full-cam beats): a word whose loudest frame is REACT_DB over the clip's
# median word, or REACT_NEAR_DB when it starts within SCARE_NEAR_S after a scare (a game
# transient); scored by that lift, plus SCARE_BONUS after a scare.
REACT_DB = 8.0
REACT_NEAR_DB = 4.0
SCARE_NEAR_S = 2.0
SCARE_BONUS = 4.0
# ... levels from voice-like frames only (a game bang under a word is no reaction); in a clip
# of shouting (its median word a shout) a word at least REACT_LOUD_DB (source dBFS) within
# REACT_TOP_DB of the clip's loudest word is a peak too, scored REACT_DB + its excess
REACT_LOUD_DB = -16.0
REACT_TOP_DB = 3.0
# Beats stay out of [scare - SCARE_PRE_S, scare + SCARE_POST_S] (the payoff is the gameplay)
# and the same round a startled reaction word: a peak word whose level jumps >= STARTLE_DB
# within STARTLE_RISE_S of its start over the quietest of the STARTLE_BEFORE_S before it (the
# game's sting and the shout land together; the cause is on screen then)
SCARE_PRE_S, SCARE_POST_S = 0.5, 1.0
STARTLE_DB = 10.0
STARTLE_RISE_S = 0.15
STARTLE_BEFORE_S = 0.5

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


def audio_fades(duration: float, last_word_end: float | None = None) -> str:
    """FADE_IN_S fade-in at the start and a FADE_OUT_S fade-out ending at `duration`, starting
    no earlier than `last_word_end` (output s; the fade shortens, to MIN_FADE_OUT_S at least,
    rather than dim the last word)."""
    start = max(duration - FADE_OUT_S, 0.0)
    if last_word_end is not None:
        start = min(max(start, last_word_end), max(duration - MIN_FADE_OUT_S, 0.0))
    return (
        f"afade=t=in:st=0:d={FADE_IN_S:g},"
        f"afade=t=out:st={start:.3f}:d={round(duration - start, 3):g}"
    )


def min_tail(
    keep: list[tuple[float, float]], words: list[Word], min_s: float = MIN_FADE_OUT_S
) -> list[tuple[float, float]]:
    """`keep` with its last range running at least `min_s` past its last word (never into
    the next word): the end fade's room when the bed is too noisy for the end tail."""
    if not keep:
        return keep
    start, end = keep[-1]
    last = max((w.end for w in words if start < w.end <= end + 1e-6), default=None)
    if last is None or end - last >= min_s:
        return keep
    nxt = min((w.start for w in words if w.start >= end - 1e-6 and w.end > end), default=None)
    new_end = last + min_s if nxt is None else min(last + min_s, max(nxt, end))
    return [*keep[:-1], (start, round(max(new_end, end), 3))]


def reaction_peaks(
    words: list[Word],
    env: Envelope,
    keep: list[tuple[float, float]],
    scares: list[float] | None = None,
) -> list[tuple[float, float]]:
    """(output s, score) of the streamer's reaction peaks among the spoken `words` (source
    time): a word's loudest voice-like envelope frame (`Envelope.voice`, when known; a word
    with none is no reaction) REACT_DB over the clip's median word, or REACT_NEAR_DB when it
    starts within SCARE_NEAR_S after a `scares` time, or a loud shout (REACT_LOUD_DB, within
    REACT_TOP_DB of the loudest word); time order."""
    levels = []
    for w in words:
        i, j = env.index(w.start), max(env.index(w.end), env.index(w.start) + 1)
        db = env.db[i:j]
        if env.voice is not None and len(env.voice) == len(env.db):
            db = db[env.voice[i:j]]
        levels.append(float(db.max()) if len(db) else -100.0)
    heard = [lv for lv in levels if lv > -100.0]
    if not heard:
        return []
    median = sorted(heard)[len(heard) // 2]
    top = max(heard)
    out = []
    for w, lv in zip(words, levels, strict=True):
        if lv <= -100.0:
            continue
        lift = lv - median
        near = any(0.0 <= w.start - s <= SCARE_NEAR_S for s in scares or [])
        score = lift + (SCARE_BONUS if near else 0.0)
        shout = lv >= REACT_LOUD_DB and lv >= top - REACT_TOP_DB
        if shout:
            score = max(score, REACT_DB + lv - REACT_LOUD_DB)
        if lift >= REACT_DB or (near and lift >= REACT_NEAR_DB) or shout:
            t = _to_output_time(w.start, keep)
            if t is not None:
                out.append((t, round(score, 3)))
    return sorted(out)


def startle_times(
    words: list[Word], env: Envelope, keep: list[tuple[float, float]], peaks: list[float]
) -> list[float]:
    """Output times among the reaction `peaks` (output s) whose word (source time) starts
    with a sharp onset: STARTLE_DB over the quietest frame of the STARTLE_BEFORE_S before it,
    within STARTLE_RISE_S."""
    out = []
    for w in words:
        t = _to_output_time(w.start, keep)
        if t is None or not any(abs(t - p) < 1e-3 for p in peaks):
            continue
        before = env.db[env.index(w.start - STARTLE_BEFORE_S) : env.index(w.start)]
        rise = env.db[env.index(w.start) : env.index(w.start + STARTLE_RISE_S) + 1]
        if len(before) and len(rise) and float(rise.max() - before.min()) >= STARTLE_DB:
            out.append(t)
    return out


def beat_avoid(
    seam: float | None,
    scares: list[float],
    spikes: list[tuple[float, float]],
    startles: list[float],
    keep: list[tuple[float, float]],
) -> list[list[float]]:
    """Output spans a full-cam beat stays out of: the EARLIER tag after the `seam`, each
    scare (source s) from SCARE_PRE_S before to SCARE_POST_S after, the gameplay `spikes`
    (source spans) and each startled reaction (`startles`, output s) like a scare."""
    out = [[seam, round(seam + SEAM_TAG_S, 3)]] if seam is not None else []
    near = [(s - SCARE_PRE_S, s + SCARE_POST_S) for s in scares]
    out += _to_output_spans([*near, *spikes], keep)
    out += [[round(t - SCARE_PRE_S, 3), round(t + SCARE_POST_S, 3)] for t in startles]
    return sorted(out)


def mix_format(path: Path) -> str:
    """aformat pinning a room-tone mix input to float planar at MIX_RATE in `path`'s audio
    channel layout: mono stays mono, stereo and wider sources (5.1, 7.1) mix in stereo."""
    res = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)]
    )  # fmt: skip
    layout = "mono" if res.stdout.strip() == "1" else "stereo"
    return f"aformat=sample_fmts=fltp:sample_rates={MIX_RATE}:channel_layouts={layout}"


def room_tone(duration: float) -> str:
    """lavfi source: `duration` s of pink noise at ROOM_TONE_DB RMS (a constant room-tone
    bed; anoisesrc's pink noise at amplitude 1 measures PINK_RMS_DB)."""
    amp = 10 ** ((ROOM_TONE_DB - PINK_RMS_DB) / 20)
    return (
        f"anoisesrc=color=pink:amplitude={amp:.6f}:sample_rate=48000:duration={duration:.3f}:seed=1"
    )


def drop_cut_slivers(
    words: list[Word], cuts: list[float], min_s: float = CUT_SLIVER_S
) -> list[Word]:
    """`words` (output time) without those shorter than `min_s` next to a cut: the last word
    before it or the first after it (by their mid-points), its mid-point within
    CUT_SLIVER_NEAR_S of the cut -- the remnant of a word the splice cut through, a caption
    flash. A short word further away is a real word and keeps its caption."""
    drop: set[int] = set()
    for c in cuts:
        before = [i for i, w in enumerate(words) if (w.start + w.end) / 2 < c]
        after = [i for i, w in enumerate(words) if (w.start + w.end) / 2 >= c]
        for i in [*before[-1:], *after[:1]]:
            w = words[i]
            near = abs((w.start + w.end) / 2 - c) <= CUT_SLIVER_NEAR_S
            if near and w.end - w.start < min_s:
                drop.add(i)
    return [w for i, w in enumerate(words) if i not in drop]


def _word_gaps(words: list[Word]) -> list[tuple[float, float]]:
    """Gaps (zero-length at touching boundaries) between consecutive words, output time."""
    return [(round(a.end, 3), round(max(b.start, a.end), 3)) for a, b in pairwise(words)]


def _to_output_spans(
    spans: list[tuple[float, float]], keep: list[tuple[float, float]]
) -> list[list[float]]:
    """Source spans mapped onto the output timeline of the kept ranges (split at cuts)."""
    out, offset = [], 0.0
    for k0, k1 in keep:
        for a, b in spans:
            lo, hi = max(a, k0), min(b, k1)
            if hi > lo:
                out.append([round(offset + lo - k0, 3), round(offset + hi - k0, 3)])
        offset += k1 - k0
    return sorted(out)


def _with_tail(
    keep: list[tuple[float, float]], words: list[Word], env: Envelope, voice_db: float
) -> list[tuple[float, float]]:
    """`keep` with its last range ending a beat after the speech (`timing.tail_end`): into the
    silence after the range's last voiced frame (the last word's end when the envelope has
    none), at most MAX_TAIL_S past it, never into the next word -- earlier than the range's
    end when the transcript's last word ran on over silence. The clip may then run up to
    MAX_TAIL_S over max_clip_s."""
    if not keep or not words:
        return keep
    start, end = keep[-1]
    spoken = [w for w in words if w.start < end and w.end > start]
    if not spoken:
        return keep
    last = max(w.end for w in spoken)
    voiced = last_voiced(env, start, end, voice_db)
    if voiced is not None:
        last = voiced  # the speech ends there, whatever the transcript says
    later = min((w for w in words if w.start >= end), key=lambda w: w.start, default=None)
    nxt = (later.start, later.end) if later else (float("inf"), float("inf"))
    new_end = tail_end(last, last, nxt[0], env, voice_db, next_end=nxt[1])
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
    profile: LayoutProfile | None = None,
) -> tuple[list[SceneLayout], Crop | None, tuple[int, int] | None]:
    """(layouts, crop, overlay bands) for the vertical reframe. Burned-in lower-thirds/edge
    overlays (would show cut mid-word) are cropped away unless disabled, refused by
    `combine_crop`, or they'd cut into the facecam: an override box overlapping a band, or an
    auto-detected STACKED cam touching the cropped edge (it continues into the band), keeps
    the letterbox-only crop. The facecam override (full-frame fractions) is mapped into the
    crop's space.

    With a stream `profile` cam (and no --facecam override) every scene gets the profile's
    fixed STACKED template (`plan_profile_layouts`); bands over the cam are kept."""
    bands = detect_overlay_bands(inter, crop=letterbox) if settings.overlay_crop else None
    crop = combine_crop(size, letterbox, bands)
    if crop == letterbox:
        bands = None  # none found, or removal refused (too much of the height)
    if settings.facecam is None and profile is not None and profile.outer is not None:
        x0, y0, x1, y1 = profile.outer
        box = (x0 / size[0], y0 / size[1], (x1 - x0) / size[0], (y1 - y0) / size[1])
        if bands and box_hits_bands(box, size, letterbox, bands):
            log.info("profile facecam %s overlaps overlay bands %s; bands kept", box, bands)
            crop, bands = letterbox, None
        return plan_profile_layouts(inter, profile, crop=crop, detector=detector), crop, bands
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
    idle_gaps: list[tuple[float, float]] | None = None,
    profile: LayoutProfile | None = None,
    scares: list[float] | None = None,
    spikes: list[tuple[float, float]] | None = None,
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
    over 0.2 s (sparing the last word).
    `overlay_bands` is the [top, bottom] px of burned-in overlay cropped away (vertical only),
    or None. `seam` is the output time where a cold open ends (None without one); it gets a
    3-frame 60% white flash and the "EARLIER…" tag. `ass`/`ass_sha1` name the burned caption file
    in `work_dir` and its hash (None without captions). `caption_words` are the spoken words
    on the output timeline, [start, end, text], so `hookline audit` needs no transcript.

    The intermediate cut (`work_dir/NN.mkv`, near-lossless libx264 + PCM audio, deleted after success)
    and the ASS file live in `work_dir`. Video is lossily encoded once (the final encode);
    audio once (AAC 192k, after compression, gain + peak limiting and two-pass
    loudnorm to -14 LUFS; see `plan_loudness`).

    `protect_words` keeps dead-air cuts out of word spans; pass False for
    segment-interpolated word times, which tile the timeline with no gaps. `encoder` is the
    final video encoder (resolved by `pick_encoder()` when None; pass it to avoid re-probing).
    `idle_gaps` (source seconds, `boundaries.idle_gaps`) are cut like dead air.

    `profile` (wave G-layout: the job's `profile.layout_profile`, computed once over the VOD)
    gives the static black bars (no per-clip detection) and, for a stream with a facecam
    overlay (unless `settings.facecam` overrides), the fixed STACKED template for every scene,
    with up to two full-cam reaction beats at the streamer's reaction peaks (`scares`: the
    VOD's scare times, source s; beats stay clear of them and of the gameplay `spikes`,
    source spans: `beat_avoid`); the sidecar's `layout_profile` records its cam."""
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
    # dead air trusts a refined word end only so far past the transcript's
    protected = protected_words(raw, near)
    untailed = keep_ranges(
        refined.segments(),
        silences,
        settings.max_pause_s,
        settings.dead_air_pad_s,
        words=protected if protect_words else None,
    )
    untailed = cut_gaps(untailed, idle_gaps or [])
    timed, tailed = near, untailed
    if voice_db is not None and untailed:
        # final words onto the last voiced frame, words over silence onto the speech; the
        # tail follows the speech, every drop / dead-air / idle splice lands in silence
        timed = anchor_words(near, env, voice_db, clip_end=untailed[-1][1], ranges=untailed)
        tailed = _with_tail(untailed, timed, env, voice_db)
        bounds = list(refined.cold_open) if refined.cold_open else []
        tailed, timed = snap_splices(tailed, timed, env, voice_db, fps, fixed=bounds, adjust=True)  # type: ignore[misc]
    elif untailed:
        tailed = min_tail(untailed, near)  # no tail on a noisy bed: still room for the fade
    keep = snap_ranges(tailed, fps, keep_gaps=[*refined.drops, *(idle_gaps or [])])
    # words starting in the tail (no audible onset: see `tail_end`) are not captioned
    heard = [i for i, w in enumerate(timed) if not untailed or w.start < untailed[-1][1] - 1e-6]
    spoken = [timed[i] for i in heard]
    # a word's survival of a cut is judged on the span dead air protected (moved along with
    # the word when anchoring / splice snapping moved it), not on an audio-refined end that
    # may run far past it (anchor_words / snap_splices keep the words' order and count)
    judge = [min(timed[i].end, protected[i].end + timed[i].start - near[i].start) for i in heard]
    uncaptioned: list[list[float]] = []
    if voice_db is not None and untailed:
        uncaptioned = _to_output_spans(uncaptioned_speech(spoken, env, voice_db, keep), keep)
        if uncaptioned:
            log.warning(
                "%s: speech without captions at %s (output s; the transcript has no words "
                "there)", name, ", ".join(f"{a:.2f}-{b:.2f}" for a, b in uncaptioned),
            )  # fmt: skip

    inter = cut_segments(
        src, keep, work_dir / f"{name}.mkv", "libx264", quality=INTERMEDIATE_QUALITY, fps=fps
    )
    duration = probe_duration(inter)
    size = probe_size(inter)
    if profile is not None and tuple(profile.size) != tuple(size):
        log.warning(
            "%s: layout profile is for %s, the cut is %s; profile ignored", name, profile.size, size
        )
        profile = None
    letterbox = profile.bars if profile is not None else detect_crop(inter)
    crop, bands = letterbox, None

    seam = seam_time(keep, refined.cold_open)
    cuts = cut_times_for(keep)
    # punctuation-only tokens never captioned (nor shielding a splice remnant from
    # drop_cut_slivers); the first word (and the first after the seam) capitalised once the
    # remnants are gone
    out_words = remap_words(spoken, keep, fps, judge_ends=judge)
    out_words = caption_words(drop_cut_slivers(caption_words(out_words), cuts), seam)
    # loudness is normalised to the speech: the captioned words' spans on the clip's timeline
    audio = plan_loudness(inter, speech=[(w.start, w.end) for w in out_words])
    # a source noise gate's digital silence exposed by the cut gets a room-tone bed (below)
    bed = bool(audio) and bool(digital_zero_runs(inter, ZERO_RUN_S))
    gaps = _word_gaps(out_words)
    layouts: list[SceneLayout] = []
    steps: list[tuple[float, float]] = []
    if settings.vertical:
        layouts, crop, bands = _plan_vertical(inter, detector, settings, size, letterbox, profile)
        in_w, in_h = (crop[0], crop[1]) if crop else size
        # per-scene zoom plan (soft-source cap, small-face base zoom, headroom); the layouts
        # carry it into the sidecar and to caption placement; word gaps steer reframe snaps
        layouts = [
            plan_zoom(lay, in_w, in_h).model_copy(
                update={"gaps": [g for g in gaps if lay.start <= g[0] < lay.end]}
            )
            for lay in layouts
        ]
        steps = zoom_schedule(
            duration,
            cuts,
            emphasis_times(pick.edit_beats, keep),
            boundaries=[w.end for w in out_words if w.text.rstrip().endswith(_SENTENCE_END)],
            changes=[lay.start for lay in layouts[1:]],  # no beat step next to a camera change
            word_gaps=gaps,
        )
        # the edge guard's holds/shifts, once: the graph and the sidecar get the same steps
        steps, layouts = guard_zoom(layouts, in_w, in_h, steps, cuts)
        if any(lay.cam_rect is not None for lay in layouts):
            # full-cam reaction beats, never under the EARLIER tag nor over a scare / spike
            peaks = reaction_peaks(spoken, env, keep, scares)
            startles = startle_times(spoken, env, keep, [t for t, _ in peaks])
            avoid = beat_avoid(seam, scares or [], spikes or [], startles, keep)
            layouts = plan_cam_beats(peaks, layouts, cuts, duration, avoid=avoid)  # type: ignore[arg-type]
        layouts = [
            lay.model_copy(
                update={
                    "face_dx": face_offset(lay, in_w, in_h, steps),
                    "snaps": scene_snaps(lay, in_w, in_h, steps),
                    # STACKED: the cam panel's head region (full-cam beats included)
                    "face_px": stacked_face_px(lay, in_w, in_h, steps)
                    if lay.kind == "STACKED"
                    else face_track_px(lay, in_w, in_h, steps),
                }
            )
            for lay in layouts
        ]
        graph = reframe_graph(layouts, in_w, in_h, crop=crop, zoom_steps=steps, cut_times=cuts)
    else:
        graph = _horizontal_graph(crop)

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
            faces=face_samples(layouts),  # each page clears the faces shown with it
        )
        atomic_write_text(ass, text)
        ass_ref = {"ass": ass.name, "ass_sha1": hashlib.sha1(ass.read_bytes()).hexdigest()}
        graph += f";{out_label}{subtitles_filter(ass, fonts_dir())}[vfinal]"
        out_label = "[vfinal]"

    audio_args: list[str] = []
    last_end = max((w.end for w in out_words), default=None)  # the end fade spares it
    audio_map = ["-map", "0:a?"]
    if audio is not None and bed:
        # digital silence in the cut: a room-tone bed under everything, after the loudness
        # chain (it adds nothing measurable to the loudness), faded with the clip. Both mix
        # inputs are pinned to one format, rate and the clip's channel layout: left to amix's
        # negotiation (the mono bed against the stereo clip) the planned loudness came out
        # ~0.7 dB low (podcast clips at -14.9 LUFS) and the peaks ~3 dB over the plan
        fmt = mix_format(inter)
        graph += (
            f";[0:a]{audio.af},{fmt}[amain];{room_tone(duration)},{fmt}[abed];"
            f"[amain][abed]amix=inputs=2:duration=first:normalize=0,"
            f"{audio_fades(duration, last_end)}[aout]"
        )
        audio_map = ["-map", "[aout]"]
        log.info("%s: digital silence in the cut; room tone at %g dBFS", name, ROOM_TONE_DB)
    elif audio is not None:  # the clip has audio
        audio_args = ["-af", f"{audio.af},{audio_fades(duration, last_end)}"]

    tmp = dst.with_suffix(".tmp.mp4")
    _run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(inter),
            # always a script: zoom/track expressions can outgrow the Windows command line
            *filter_graph_args(graph, work_dir / f"{name}.graph.txt", force=True),
            "-map", out_label, *audio_map,
            *encoder_args(encoder or pick_encoder()), "-pix_fmt", "yuv420p",
            *audio_args,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",  # the one lossy audio encode
            "-movflags", "+faststart",
            str(tmp),
        ]
    )  # fmt: skip
    os.replace(tmp, dst)
    if audio and audio.mode != "skip":
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
        "uncaptioned_speech": uncaptioned,
        "room_tone": bed,
        "layout_profile": _profile_summary(profile, layouts),
    }


def _profile_summary(profile: LayoutProfile | None, layouts: list[SceneLayout]) -> dict | None:
    """The stream profile's cam as the clip used it (source px), for the audit; None when no
    scene used the profile's template."""
    if profile is None or not any(lay.cam_rect is not None for lay in layouts):
        return None
    return profile.model_dump(mode="json", include={"cam", "outer", "head", "masks", "bars"})
