import json
import os
from collections.abc import Callable
from pathlib import Path

from hookline.assets import AssetError, ensure_yunet
from hookline.config import Settings
from hookline.fsutil import atomic_write_text, slugify
from hookline.llm import LLMBackend
from hookline.media.audio import rms_per_second
from hookline.media.ffmpeg import FFmpegError, extract_audio
from hookline.models import ClipPick, Transcript
from hookline.pipeline.chat import chat_rate, try_fetch_chat
from hookline.pipeline.ingest import find_source, ingest, job_id_for
from hookline.pipeline.render import render_clip
from hookline.pipeline.select import select_clips
from hookline.pipeline.signals import Signals, heat_markers
from hookline.transcribe import Transcriber
from hookline.vision.faces import FaceDetector

STAGES = 6


class NoClipsSelectedError(Exception):
    """Selection returned zero clips; nothing was cached so a re-run re-selects."""


def transcript_key(transcriber: Transcriber) -> dict:
    """What a cached transcript depends on: backend, model and DTW preset."""
    key = getattr(transcriber, "cache_key", None)
    if isinstance(key, dict):
        return {"backend": key.get("backend"), "model": key.get("model"), "dtw": key.get("dtw")}
    return {"backend": type(transcriber).__name__, "model": None, "dtw": None}


def _picks_key(settings: Settings, tkey: dict) -> dict:
    return {
        "clips": settings.clips,
        "select_model": settings.select_model,
        "min_clip_s": settings.min_clip_s,
        "max_clip_s": settings.max_clip_s,
        "transcript": tkey,
    }


def _load_transcript(tpath: Path, key: dict, log: Callable[[str], None]) -> Transcript | None:
    """Cached transcript when transcript.json was made with `key`; otherwise None."""
    if not tpath.exists():
        return None
    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data and "key" not in data:
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


def _signals(
    job_dir: Path, wav: Path, transcript: Transcript, log: Callable[[str], None]
) -> Signals:
    spath = job_dir / "signals.json"
    cpath = job_dir / "chat.json"
    if spath.exists():
        try:
            cached = Signals.model_validate_json(spath.read_text(encoding="utf-8"))
            if cached.chat is not None or not cpath.exists():
                log("[4/6] signals: skipped (cached)")
                return cached
        except ValueError:
            log("warning: signals.json is corrupt; recomputing")
    log("[4/6] signals: started")
    duration = transcript.duration
    chat = None
    if cpath.exists():
        try:
            raw = json.loads(cpath.read_text(encoding="utf-8"))
            chat = chat_rate(raw if isinstance(raw, dict) else {}, duration)
        except (ValueError, OSError) as e:
            log(f"warning: chat.json unreadable ({e}); continuing without chat signals")
    signals = Signals(duration=duration, loudness=rms_per_second(wav), chat=chat)
    atomic_write_text(spath, signals.model_dump_json())
    log("[4/6] signals: done")
    return signals


def _face_detector(settings: Settings, log: Callable[[str], None]) -> FaceDetector | None:
    if not settings.vertical or settings.facecam is not None:
        return None
    try:
        return FaceDetector(ensure_yunet())
    except AssetError as e:
        log(f"warning: face model unavailable ({e}); every scene uses FIT_BLUR")
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

    signals = _signals(job_dir, wav, transcript, log)
    if not settings.chat:
        signals = signals.model_copy(update={"chat": None})

    ppath = job_dir / "picks.json"
    key = _picks_key(settings, tkey)
    picks = _load_picks(ppath, key, log)
    if picks is not None:
        log("[5/6] select: skipped (using cached picks.json - delete it to re-select)")
    else:
        log("[5/6] select: started")
        raw_payloads: list[dict] = []
        picks = select_clips(
            transcript, llm, settings, raw_sink=raw_payloads, markers=heat_markers(signals)
        )
        atomic_write_text(
            job_dir / "select_raw.json", json.dumps(raw_payloads, indent=2, ensure_ascii=False)
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

    log(f"[6/6] render: started ({len(picks)} clips)")
    out_dir = settings.output_dir / job_id
    work_dir = job_dir / "render"
    detector = _face_detector(settings, log)
    words = transcript.words()
    # Segment-interpolated word times are contiguous (no gaps), so protecting them from cuts
    # would remove no dead air at all; only real word timings (DTW / whisper) are protected.
    protect = transcript.timing != "segment"
    clips: list[Path] = []
    for i, pick in enumerate(picks, 1):
        dst = out_dir / f"{i:02d}-{slugify(pick.title)}.mp4"
        try:
            extras = render_clip(
                src, wav, pick, words, work_dir, dst, settings, detector, protect_words=protect
            )
        except FFmpegError as e:
            reason = str(e).splitlines()[0]
            if e.stderr:
                reason += f" ({e.stderr.splitlines()[-1].strip()})"
            log(f"warning: clip {i:02d} failed: {reason}")
            continue
        sidecar = {**pick.model_dump(mode="json"), "post_caption": pick.hook_text, **extras}
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
