import hashlib
import json
import os
import traceback
import wave
from collections.abc import Callable
from pathlib import Path

from hookline.assets import AssetError, ensure_yunet
from hookline.config import Settings
from hookline.fsutil import atomic_write_text, slugify
from hookline.llm import LLMBackend
from hookline.media.audio import MOMENTARY_HOP, momentary_db, rms_per_second
from hookline.media.ffmpeg import (
    FFmpegError,
    extract_audio,
    has_audio,
    pick_encoder,
    probe_video_kbps,
)
from hookline.media.motion import Rect, frame_stats, motion_timeout
from hookline.models import ClipPick, Transcript, Word
from hookline.pipeline.boundaries import (
    CAUSE_MAX_S,
    PAYOFF_SAID_MAX_S,
    PAYOFF_SCARE_S,
    SIDE_MIN_S,
    VOICELESS_GAME_S,
    VOICELESS_MAX_S,
    edit_gaps,
    fit_to_signals,
    keep_payoff_side,
)
from hookline.pipeline.chat import chat_keywords, chat_rate, try_fetch_chat
from hookline.pipeline.ingest import IngestError, find_source, ingest, job_id_for
from hookline.pipeline.layout import SCARE_POST_S, SCARE_PRE_S
from hookline.pipeline.package import apply_caption_fixes, package_clips
from hookline.pipeline.profile import LayoutProfile, layout_profile
from hookline.pipeline.render import render_clip, voiced_words
from hookline.pipeline.select import NoClipsSelectedError, select_clips
from hookline.pipeline.signals import (
    SCREEN_MARKER_MIN_S,
    SIGNALS_VERSION,
    Signals,
    flash_seconds,
    heat_markers,
    heat_seconds,
    loud_seconds,
    onset_times,
    scare_times,
    screen_runs,
    seconds_of,
)
from hookline.transcribe import Transcriber
from hookline.transcribe.base import words_to_transcript
from hookline.vision.faces import FaceDetector

PICKS_SCHEMA = 9  # bump when picks change shape or meaning, so old picks.json re-select
PACKAGE_SCHEMA = 5  # bump when the packaging pass changes, so old package.json re-runs
PAYOFF_ALLOWANCE_S = 5.0  # a clip may run this far over max_clip_s (select's payoff extension)


def transcript_key(transcriber: Transcriber) -> dict:
    """What a cached transcript depends on: backend, model and DTW preset."""
    key = getattr(transcriber, "cache_key", None)
    if isinstance(key, dict):
        return {"backend": key.get("backend"), "model": key.get("model"), "dtw": key.get("dtw")}
    return {"backend": type(transcriber).__name__, "model": None, "dtw": None}


_LOUD_FILLER_DB = 6.0


def _picks_key(settings: Settings, tkey: dict) -> dict:
    return {
        "schema": PICKS_SCHEMA,
        "clips": settings.clips,
        "select_model": settings.select_model,
        "min_clip_s": settings.min_clip_s,
        "max_clip_s": settings.max_clip_s,
        "transcript": tkey,
    }


def _migrate_old_transcript(
    tpath: Path, old: Transcript, key: dict, log: Callable[[str], None]
) -> Transcript | None:
    """Re-key an old-format DTW transcript for a DTW whisper.cpp backend (no re-transcribe)."""
    if old.timing != "dtw" or key.get("backend") != "whisper-cpp" or not key.get("dtw"):
        return None
    cleaned = words_to_transcript(old.words(), old.language, old.duration, timing="dtw")
    atomic_write_text(
        tpath,
        json.dumps({"key": key, "transcript": cleaned.model_dump(mode="json")}),
    )
    log("transcript: migrated cached DTW transcript")
    return cleaned


def _load_transcript(tpath: Path, key: dict, log: Callable[[str], None]) -> Transcript | None:
    """Cached transcript when transcript.json was made with `key`; otherwise None."""
    if not tpath.exists():
        return None
    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data and "key" not in data:
            migrated = _migrate_old_transcript(tpath, Transcript.model_validate(data), key, log)
            if migrated is not None:
                return migrated
            log("[3/6] transcribe: transcript.json is in the old format; re-transcribing")
            return None
        cached_key = data["key"]
        transcript = Transcript.model_validate(data["transcript"])
    except (ValueError, TypeError, KeyError):
        log("warning: transcript.json is corrupt; deleting and re-running transcription")
        tpath.unlink()
        return None
    changed = sorted(k for k in key if cached_key.get(k) != key[k])
    if changed:
        log(f"[3/6] transcribe: transcriber changed ({', '.join(changed)}); re-transcribing")
        return None
    return transcript


def _load_picks(ppath: Path, key: dict, log: Callable[[str], None]) -> list[ClipPick] | None:
    """Cached picks when picks.json matches `key`; otherwise None (and deletes a corrupt file)."""
    if not ppath.exists():
        return None
    try:
        data = json.loads(ppath.read_text(encoding="utf-8"))
        if isinstance(data, list):
            log("[5/6] select: picks.json is in the old format; re-selecting")
            return None
        cached_key = data["key"]
        picks = [ClipPick.model_validate(d) for d in data["picks"]]
    except (ValueError, TypeError, KeyError):
        log("warning: picks.json is corrupt; deleting and re-running selection")
        ppath.unlink()
        return None
    changed = sorted(k for k in key if cached_key.get(k) != key[k])
    if changed:
        log(f"[5/6] select: settings changed ({', '.join(changed)}); re-selecting")
        return None
    return picks


def _package(
    job_dir: Path,
    picks: list[ClipPick],
    words: list[Word],
    llm: LLMBackend,
    settings: Settings,
    log: Callable[[str], None],
    screens: list[tuple[float, float]] | None = None,
) -> list[ClipPick]:
    """Packaged picks: cached in package.json, keyed by the exact picks it was made from, so a
    re-run neither re-calls Claude nor reuses packaging for different picks. A failed pass
    keeps the selection-time packaging and is retried on the next run. `screens` (death /
    loading screens) are marked in the words the packager reads."""
    ppath = job_dir / "package.json"
    dump = [p.model_dump(mode="json") for p in picks]
    digest = hashlib.sha1(json.dumps(dump, sort_keys=True).encode()).hexdigest()
    key = {"schema": PACKAGE_SCHEMA, "select_model": settings.select_model, "picks": digest}
    if ppath.exists():
        try:
            data = json.loads(ppath.read_text(encoding="utf-8"))
            if data["key"] == key:
                log("[5/6] select: packaging skipped (cached)")
                return [ClipPick.model_validate(d) for d in data["picks"]]
        except (ValueError, TypeError, KeyError):
            log("warning: package.json is corrupt; re-running packaging")
    log("[5/6] select: packaging started")
    raw: list[dict] = []
    packaged = package_clips(picks, words, llm, settings, raw_sink=raw, screens=screens)
    atomic_write_text(job_dir / "package_raw.json", json.dumps(raw, indent=2, ensure_ascii=False))
    if packaged is picks:
        log("warning: packaging pass failed; using the selection-time packaging")
        return picks
    atomic_write_text(
        ppath,
        json.dumps(
            {"key": key, "picks": [p.model_dump(mode="json") for p in packaged]},
            indent=2,
            ensure_ascii=False,
        ),
    )
    fixes = sum(len(p.caption_fixes) for p in packaged)
    log(f"[5/6] select: packaging done ({len(packaged)} clips, {fixes} caption fixes)")
    return packaged


def _signals(
    job_dir: Path,
    src: Path,
    wav: Path,
    transcript: Transcript,
    log: Callable[[str], None],
    mask: list[Rect] | None = None,
) -> Signals:
    """signals.json, recomputing only what is missing or stale: chat when chat.json exists but
    the cache has no chat (or no reaction kinds), the audio onsets when the cache is below
    SIGNALS_VERSION, the VOD motion decode (the slow part; motion, screen frames and their
    chroma) only when the cache lacks any of them or measured the chroma with another cam
    `mask` (`screen_mask`: the stream profile's cam and labels, `screen_mask_for`). A failed
    onset analysis is stored as missing, so the next run retries it; a failed or timed-out
    motion decode (slow: up to `motion_timeout`) is recorded (`motion_failed`) and retried only
    for a changed source file or SIGNALS_VERSION."""
    spath = job_dir / "signals.json"
    cpath = job_dir / "chat.json"
    cached: Signals | None = None
    if spath.exists():
        try:
            cached = Signals.model_validate_json(spath.read_text(encoding="utf-8"))
        except ValueError:
            log("warning: signals.json is corrupt; recomputing")
    chat_stale = cpath.exists() and (
        cached is None or cached.chat is None or cached.chat_kinds is None
    )
    onsets_stale = cached is None or cached.version < SIGNALS_VERSION or cached.onset_times is None
    want_mask = [list(b) for b in mask] if mask else None
    mkey = motion_key(src)
    motion_stale = (cached is None or cached.motion_failed != mkey) and (
        cached is None
        or cached.motion is None
        or cached.screen is None
        or cached.screen_chroma is None
        or cached.screen_mask != want_mask
    )
    if cached is not None and not (chat_stale or onsets_stale or motion_stale):
        log("[4/6] signals: skipped (cached)")
        return cached
    log("[4/6] signals: started")
    duration = transcript.duration
    update: dict = {"version": SIGNALS_VERSION}
    if cached is None:
        update |= {"chat": None, "chat_kinds": None}
    if chat_stale:
        try:
            raw = json.loads(cpath.read_text(encoding="utf-8"))
            raw = raw if isinstance(raw, dict) else {}
            update["chat"] = chat_rate(raw, duration)
            update["chat_kinds"] = {s: dict(c) for s, c in chat_keywords(raw, duration).items()}
        except (ValueError, OSError) as e:
            log(f"warning: chat.json unreadable ({e}); continuing without chat signals")
    if onsets_stale:
        speech = [(w.start, w.end) for w in transcript.words()]
        try:
            every, game = onset_times(momentary_db(wav), MOMENTARY_HOP, speech)
            update |= {
                "onset_times": every,
                "transient_times": game,
                "onsets": seconds_of(every),
                "transients": seconds_of(game),
            }
        except (FFmpegError, wave.Error, EOFError) as e:
            log(f"warning: audio onset analysis failed ({e}); continuing without scare signals")
            update |= {"onset_times": None, "transient_times": None}
            update |= {"onsets": None, "transients": None}
    if motion_stale:
        long_scan = duration >= MOTION_PROGRESS_MIN_S
        if long_scan:
            log(f"[4/6] signals: motion scan of {duration / 60:.0f} min started")

        def scanned(done: float) -> None:
            if long_scan:
                log(f"[4/6] signals: motion scan {done / 60:.0f} of {duration / 60:.0f} min")

        try:
            motion, screen, chroma = frame_stats(
                src, mask=mask, timeout=motion_timeout(duration), progress=scanned
            )
            update |= {"motion": motion, "screen": screen, "screen_chroma": chroma}
            update |= {"screen_mask": want_mask, "motion_failed": None}
        except FFmpegError as e:
            log(
                f"warning: motion analysis failed ({e}); continuing without visual signals "
                "(not retried until the source changes)"
            )
            update |= {"motion": None, "screen": None, "screen_chroma": None}
            update["motion_failed"] = mkey
    if cached is None:
        signals = Signals(duration=duration, loudness=rms_per_second(wav), **update)
    else:
        signals = cached.model_copy(update=update)
    atomic_write_text(spath, signals.model_dump_json())
    log("[4/6] signals: done")
    return signals


# a motion scan of a source this long (s) logs its progress
MOTION_PROGRESS_MIN_S = 1800.0


def motion_key(src: Path) -> str:
    """The source file (size, modification time) and SIGNALS_VERSION a motion decode ran on:
    a failed decode is retried only when this changes."""
    st = src.stat()
    return f"v{SIGNALS_VERSION}:{st.st_size}:{st.st_mtime_ns}"


def _loud_seconds(signals: Signals) -> set[int]:
    """Seconds 6 dB above the median (and over the -30 dB floor): shouted words."""
    return loud_seconds(signals.loudness, _LOUD_FILLER_DB)


def _face_detector(settings: Settings, log: Callable[[str], None]) -> FaceDetector | None:
    if not settings.vertical or settings.facecam is not None:
        return None
    try:
        model = ensure_yunet()
    except AssetError as e:
        log(f"warning: face model unavailable ({e}); every scene uses FIT_BLUR")
        return None
    try:
        return FaceDetector(model)
    except Exception as e:  # noqa: BLE001 - cv2 raises its own error types; never abort the job
        log(f"warning: face detector unavailable ({e}); every scene uses FIT_BLUR")
        return None


def beat_spikes(signals: Signals) -> list[tuple[float, float]]:
    """Gameplay motion spikes (`signals.flash_seconds`: a flash, a hard visual jump) as source
    spans a full-cam beat stays out of: the spike's second from SCARE_PRE_S before to
    SCARE_POST_S after it ends."""
    return [
        (s - SCARE_PRE_S, s + 1 + SCARE_POST_S) for s in sorted(flash_seconds(signals.motion or []))
    ]


def screen_mask_for(profile: LayoutProfile | None) -> list[Rect] | None:
    """The stream profile's cam and its labels (`LayoutProfile.keep_out`) as normalized x0, y0,
    x1, y1 boxes: masked out of the static-screen chroma (a cam is coloured, a death screen
    behind it is not). None without a cam."""
    if profile is None or profile.cam is None:
        return None
    w, h = profile.size
    return [
        (round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4))
        for x0, y0, x1, y1 in profile.keep_out()
    ]


# a fitted clip may start this much before its pick (the scare's cause) and end this much after
# it (the payoff scare's reaction): the voice-clamped words cover that
_FIT_BEFORE_S = CAUSE_MAX_S + 1.0
_FIT_AFTER_S = PAYOFF_SCARE_S + PAYOFF_SAID_MAX_S + 1.0


def voiceless_limit(gameplay: bool) -> float:
    """The longest voiceless stretch a clip keeps: VOICELESS_GAME_S on gameplay (a stream with
    a facecam), VOICELESS_MAX_S on talk; recorded in the sidecar for the audit's Q-voiceless."""
    return VOICELESS_GAME_S if gameplay else VOICELESS_MAX_S


def side_min_s(settings: Settings) -> float:
    """The shortest side of a death screen that is still a clip: SIDE_MIN_S, at least the
    configured min_clip_s."""
    return max(SIDE_MIN_S, settings.min_clip_s)


def fit_pick(
    pick: ClipPick,
    words: list[Word],
    signals: Signals,
    wav: Path,
    settings: Settings,
    gameplay: bool = False,
) -> tuple[ClipPick, list[tuple[float, float]], list[Word]]:
    """(the pick fitted to the signals, its interior edit gaps, the words they were judged
    on). The fit and the gaps see the words as the render captions them: held ends clamped to
    the voice (`render.voiced_words`), so the voiceless trimmer and the audit's Q-voiceless
    measure the same voice. `gameplay` (a stream with a facecam) cuts voiceless stretches
    over VOICELESS_GAME_S instead of VOICELESS_MAX_S -- in the fit's length estimates too --
    and backs reaction openers onto their cause sound (`include_cause_sound`)."""
    voiced = voiced_words(
        wav, words, pick.start - _FIT_BEFORE_S, pick.end + _FIT_AFTER_S, settings.silence_db
    )
    limit = voiceless_limit(gameplay)
    fitted = fit_to_signals(
        pick,
        voiced,
        signals,
        max_s=settings.max_clip_s + PAYOFF_ALLOWANCE_S,
        quiet_db=settings.silence_db,
        voiceless_s=limit,
        gameplay=gameplay,
        min_side_s=side_min_s(settings),
    )
    gaps = edit_gaps(fitted, voiced, signals, quiet_db=settings.silence_db, voiceless_s=limit)
    return fitted, gaps, voiced


def _layout_profile(
    job_dir: Path, src: Path, detector: FaceDetector | None, log: Callable[[str], None]
) -> LayoutProfile | None:
    """The VOD's layout profile (black bars; a stream's facecam overlay), once per job and
    cached; None (per-clip detection) when it cannot be computed."""
    try:
        prof = layout_profile(job_dir, src, detector)
    except Exception as e:  # noqa: BLE001 - an optional analysis (e.g. MemoryError on 4K)
        log(
            f"warning: layout profile failed ({type(e).__name__}: {e}); layouts are planned per clip"
        )
        return None
    if prof.cam is not None:
        log(f"layout: stream facecam at {prof.cam} px (fixed STACKED template)")
    return prof


def _one_side_of_deaths(
    picks: list[ClipPick],
    words: list[Word],
    signals: Signals,
    log: Callable[[str], None],
    min_s: float = SIDE_MIN_S,
) -> list[ClipPick]:
    """Each pick on one side of any death screen it spans (`boundaries.keep_payoff_side`;
    sides shorter than `min_s` are no clip), before packaging describes it; a pick with no
    viable side is dropped (all kept as they are when that would drop every one). Logs the
    drop_lines selection ignored, and each change."""
    out: list[ClipPick] = []
    for i, pick in enumerate(picks, 1):
        for note in pick.ignored_drops:
            log(f"clip {i:02d}: {note}")
        side = keep_payoff_side(pick, words, signals, min_s=min_s)
        if side is None:
            log(f"clip {i:02d}: spans a death screen with no viable side; dropped")
            continue
        if side is not pick:
            log(
                f"clip {i:02d}: spans a death screen; kept {side.start:.2f}-{side.end:.2f} "
                f"(was {pick.start:.2f}-{pick.end:.2f})"
            )
        out.append(side)
    if not out:
        log("warning: every pick spans a death screen with no viable side; all kept")
        return picks
    return out


def run_job(
    source: str,
    settings: Settings,
    llm: LLMBackend,
    transcriber: Transcriber,
    log: Callable[[str], None] = print,
) -> list[Path]:
    job_id = job_id_for(source)
    job_dir = settings.workspace_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    src = find_source(job_dir)
    if src:
        log("[1/6] ingest: skipped (cached)")
    else:
        log("[1/6] ingest: started")
        src = ingest(source, job_dir, settings)
        log(f"[1/6] ingest: done ({src.name})")
    if settings.chat:
        try_fetch_chat(source, job_dir, log)

    wav = job_dir / "audio.wav"
    if wav.exists():
        log("[2/6] audio: skipped (cached)")
    else:
        log("[2/6] audio: started")
        if not has_audio(src):
            # clips are chosen from what is said: nothing to transcribe, nothing to pick
            raise IngestError(
                f"{src.name} has no audio track — Hookline picks clips from speech, so the "
                "source needs its audio"
            )
        tmp = job_dir / "audio.wav.tmp.wav"
        extract_audio(src, tmp)
        os.replace(tmp, wav)
        log("[2/6] audio: done")

    tpath = job_dir / "transcript.json"
    tkey = transcript_key(transcriber)
    transcript = _load_transcript(tpath, tkey, log)
    if transcript is not None:
        log("[3/6] transcribe: skipped (cached)")
    else:
        log("[3/6] transcribe: started")
        last = [-1]

        def on_progress(frac: float) -> None:
            pct = int(frac * 100) // 10 * 10
            if pct > last[0]:
                last[0] = pct
                log(f"[3/6] transcribe: {pct}%")

        transcript = transcriber.transcribe(wav, on_progress)
        atomic_write_text(
            tpath,
            json.dumps({"key": tkey, "transcript": transcript.model_dump(mode="json")}),
        )
        log("[3/6] transcribe: done")

    # the stream profile first: its cam is masked out of the static-screen chroma
    detector = _face_detector(settings, log)
    profile = _layout_profile(job_dir, src, detector, log)
    signals = _signals(job_dir, src, wav, transcript, log, mask=screen_mask_for(profile))
    if not settings.chat:
        signals = signals.model_copy(update={"chat": None, "chat_kinds": None})

    # death screens: marked for the selector and the packager, and no pick spans one
    death_screens = screen_runs(signals, min_s=SCREEN_MARKER_MIN_S)
    ppath = job_dir / "picks.json"
    key = _picks_key(settings, tkey)
    picks = _load_picks(ppath, key, log)
    if picks is not None:
        log("[5/6] select: skipped (using cached picks.json - delete it to re-select)")
    else:
        log("[5/6] select: started")
        raw_payloads: list[dict] = []
        try:
            picks = select_clips(
                transcript,
                llm,
                settings,
                raw_sink=raw_payloads,
                markers=heat_markers(signals),
                heat_seconds=heat_seconds(signals),
                screens=death_screens,
            )
        finally:  # kept even when selection fails, for debugging what Claude returned
            atomic_write_text(
                job_dir / "select_raw.json",
                json.dumps(raw_payloads, indent=2, ensure_ascii=False),
            )
        if not picks:
            raise NoClipsSelectedError(
                f"No clips were selected — see workspace/{job_id}/select_raw.json"
            )
        atomic_write_text(
            ppath,
            json.dumps(
                {"key": key, "picks": [p.model_dump(mode="json") for p in picks]},
                indent=2,
                ensure_ascii=False,
            ),
        )
        log(f"[5/6] select: done ({len(picks)} clips)")

    words = transcript.words()
    picks = _one_side_of_deaths(picks, words, signals, log, min_s=side_min_s(settings))
    picks = _package(job_dir, picks, words, llm, settings, log, death_screens)

    log(f"[6/6] render: started ({len(picks)} clips)")
    out_dir = settings.output_dir / job_id
    work_dir = job_dir / "render"
    # full-cam reaction beats and the audit's Q-beat: every scare (game transients and flash
    # scares) and the gameplay's motion spikes
    scares, spikes = scare_times(signals), beat_spikes(signals)
    # every static / death screen run, talked over or not: covered by the full cam in the render
    cover_screens = screen_runs(signals)
    gameplay = profile is not None and profile.cam is not None  # a stream: 3 s voiceless
    # Segment-interpolated word times are contiguous (no gaps), so protecting them from cuts
    # would remove no dead air at all; only real word timings (DTW / whisper) are protected.
    protect = transcript.timing != "segment"
    encoder = pick_encoder()
    loud = _loud_seconds(signals)
    source_kbps = probe_video_kbps(src)  # the audit's Q8 bitrate bar follows the source
    clips: list[Path] = []
    for i, pick in enumerate(picks, 1):
        dst = out_dir / f"{i:02d}-{slugify(pick.title)}.mp4"
        try:
            # the cause of an opening reaction pulled in; no death-screen / menu tails; the
            # final hard length clamp (max_clip_s + the payoff allowance, the end tail not
            # counted); still, quiet, wordless stretches, static screens and long voiceless
            # stretches inside the clip are cut like dead air (`gaps`)
            trimmed, gaps, _ = fit_pick(pick, words, signals, wav, settings, gameplay)
            if trimmed != pick:
                log(
                    f"clip {i:02d}: fitted to signals {pick.start:.2f}-{pick.end:.2f} -> "
                    f"{trimmed.start:.2f}-{trimmed.end:.2f}"
                )
                pick = trimmed
            extras = render_clip(
                src,
                wav,
                pick,
                apply_caption_fixes(words, pick.caption_fixes),
                work_dir,
                dst,
                settings,
                detector,
                protect_words=protect,
                encoder=encoder,
                loud_seconds=loud,
                idle_gaps=gaps,
                profile=profile,
                scares=scares,
                spikes=spikes,
                screens=cover_screens,
            )
        except Exception as e:  # noqa: BLE001 - one bad clip never aborts the job
            reason = (str(e).splitlines() or [type(e).__name__])[0]
            if isinstance(e, FFmpegError) and e.stderr:
                reason += f" ({e.stderr.splitlines()[-1].strip()})"
            log(f"warning: clip {i:02d} failed: {reason}")
            if not isinstance(e, (FFmpegError, ValueError, OSError)):
                # an unexpected error is a bug: keep its traceback for the report
                log(traceback.format_exc().rstrip())
            continue
        post_caption = pick.post_caption or pick.hook_text
        sidecar = {
            **pick.model_dump(mode="json"),
            "post_caption": post_caption,
            "source_video_kbps": source_kbps,
            "voiceless_limit_s": voiceless_limit(gameplay),  # the audit's Q-voiceless
            **extras,
        }
        atomic_write_text(
            dst.with_suffix(".json"), json.dumps(sidecar, indent=2, ensure_ascii=False)
        )
        kinds = ",".join(lay["kind"] for lay in extras["layout"]) or "HORIZONTAL"
        log(f"clip {i:02d}: layouts=[{kinds}] removed={extras['removed_s']:.1f}s → {dst.name}")
        clips.append(dst)
    if not clips:
        raise FFmpegError(f"no clips rendered ({len(picks)} failed) — see warnings above")
    log(f"[6/6] render: done ({len(clips)} of {len(picks)} clips)")
    return clips
