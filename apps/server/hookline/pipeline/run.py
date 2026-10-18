import hashlib
import json
import os
import wave
from collections.abc import Callable
from pathlib import Path

from hookline.assets import AssetError, ensure_yunet
from hookline.config import Settings
from hookline.fsutil import atomic_write_text, slugify
from hookline.llm import LLMBackend
from hookline.media.audio import MOMENTARY_HOP, momentary_db, rms_per_second
from hookline.media.ffmpeg import FFmpegError, extract_audio, pick_encoder, probe_video_kbps
from hookline.media.motion import frame_motion
from hookline.models import ClipPick, Transcript, Word
from hookline.pipeline.boundaries import fit_to_signals, idle_gaps
from hookline.pipeline.chat import chat_keywords, chat_rate, try_fetch_chat
from hookline.pipeline.ingest import find_source, ingest, job_id_for
from hookline.pipeline.package import apply_caption_fixes, package_clips
from hookline.pipeline.render import render_clip
from hookline.pipeline.select import NoClipsSelectedError, select_clips
from hookline.pipeline.signals import (
    SIGNALS_VERSION,
    Signals,
    heat_markers,
    heat_seconds,
    loud_seconds,
    onset_times,
    seconds_of,
)
from hookline.transcribe import Transcriber
from hookline.transcribe.base import words_to_transcript
from hookline.vision.faces import FaceDetector

STAGES = 6
PICKS_SCHEMA = 7  # bump when picks change shape or meaning, so old picks.json re-select
PACKAGE_SCHEMA = 3  # bump when the packaging pass changes, so old package.json re-runs
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
) -> list[ClipPick]:
    """Packaged picks: cached in package.json, keyed by the exact picks it was made from, so a
    re-run neither re-calls Claude nor reuses packaging for different picks. A failed pass
    keeps the selection-time packaging and is retried on the next run."""
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
    packaged = package_clips(picks, words, llm, settings, raw_sink=raw)
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
    job_dir: Path, src: Path, wav: Path, transcript: Transcript, log: Callable[[str], None]
) -> Signals:
    """signals.json, recomputing only what is missing or stale: chat when chat.json exists but
    the cache has no chat (or no reaction kinds), the audio onsets when the cache is below
    SIGNALS_VERSION, the VOD motion decode (the slow part) only when the cache has none. A
    failed onset or motion analysis is stored as missing, so the next run retries it."""
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
    motion_stale = cached is None or cached.motion is None
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
        try:
            update["motion"] = frame_motion(src)
        except FFmpegError as e:
            log(f"warning: motion analysis failed ({e}); continuing without visual signals")
            update["motion"] = None
    if cached is None:
        signals = Signals(duration=duration, loudness=rms_per_second(wav), **update)
    else:
        signals = cached.model_copy(update=update)
    atomic_write_text(spath, signals.model_dump_json())
    log("[4/6] signals: done")
    return signals


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

    signals = _signals(job_dir, src, wav, transcript, log)
    if not settings.chat:
        signals = signals.model_copy(update={"chat": None, "chat_kinds": None})

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
    picks = _package(job_dir, picks, words, llm, settings, log)

    log(f"[6/6] render: started ({len(picks)} clips)")
    out_dir = settings.output_dir / job_id
    work_dir = job_dir / "render"
    detector = _face_detector(settings, log)
    # Segment-interpolated word times are contiguous (no gaps), so protecting them from cuts
    # would remove no dead air at all; only real word timings (DTW / whisper) are protected.
    protect = transcript.timing != "segment"
    encoder = pick_encoder()
    loud = _loud_seconds(signals)
    source_kbps = probe_video_kbps(src)  # the audit's Q8 bitrate bar follows the source
    clips: list[Path] = []
    for i, pick in enumerate(picks, 1):
        dst = out_dir / f"{i:02d}-{slugify(pick.title)}.mp4"
        # the cause of an opening reaction pulled in; no death-screen / menu tails; the final
        # hard length clamp (max_clip_s + the payoff allowance, the end tail not counted)
        trimmed = fit_to_signals(
            pick,
            words,
            signals,
            max_s=settings.max_clip_s + PAYOFF_ALLOWANCE_S,
            quiet_db=settings.silence_db,
        )
        if trimmed != pick:
            log(
                f"clip {i:02d}: fitted to signals {pick.start:.2f}-{pick.end:.2f} -> "
                f"{trimmed.start:.2f}-{trimmed.end:.2f}"
            )
            pick = trimmed
        # still, quiet, wordless stretches inside the clip are cut like dead air
        gaps = idle_gaps(pick, words, signals, quiet_db=settings.silence_db)
        try:
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
            )
        except (FFmpegError, ValueError, OSError) as e:
            reason = (str(e).splitlines() or [type(e).__name__])[0]
            if isinstance(e, FFmpegError) and e.stderr:
                reason += f" ({e.stderr.splitlines()[-1].strip()})"
            log(f"warning: clip {i:02d} failed: {reason}")
            continue
        post_caption = pick.post_caption or pick.hook_text
        sidecar = {
            **pick.model_dump(mode="json"),
            "post_caption": post_caption,
            "source_video_kbps": source_kbps,
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
