import json
import os
from collections.abc import Callable
from pathlib import Path

from hookline.config import Settings
from hookline.fsutil import atomic_write_text
from hookline.llm import LLMBackend
from hookline.media.ffmpeg import extract_audio, pick_encoder
from hookline.models import ClipPick, Transcript
from hookline.pipeline.cut import cut_clips
from hookline.pipeline.ingest import find_source, ingest, job_id_for
from hookline.pipeline.select import select_clips
from hookline.transcribe import Transcriber


class NoClipsSelectedError(Exception):
    """Selection returned zero clips; nothing was cached so a re-run re-selects."""


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
        log("[1/5] ingest: skipped (cached)")
    else:
        log("[1/5] ingest: started")
        src = ingest(source, job_dir, settings)
        log(f"[1/5] ingest: done ({src.name})")

    wav = job_dir / "audio.wav"
    if wav.exists():
        log("[2/5] audio: skipped (cached)")
    else:
        log("[2/5] audio: started")
        tmp = job_dir / "audio.wav.tmp.wav"
        extract_audio(src, tmp)
        os.replace(tmp, wav)
        log("[2/5] audio: done")

    tpath = job_dir / "transcript.json"
    transcript = None
    if tpath.exists():
        try:
            transcript = Transcript.model_validate_json(tpath.read_text(encoding="utf-8"))
            log("[3/5] transcribe: skipped (cached)")
        except ValueError:
            log("warning: transcript.json is corrupt; deleting and re-running transcription")
            tpath.unlink()
    if transcript is None:
        log("[3/5] transcribe: started")
        last = [-1]

        def on_progress(frac: float) -> None:
            pct = int(frac * 100) // 10 * 10
            if pct > last[0]:
                last[0] = pct
                log(f"[3/5] transcribe: {pct}%")

        transcript = transcriber.transcribe(wav, on_progress)
        atomic_write_text(tpath, transcript.model_dump_json())
        log("[3/5] transcribe: done")

    ppath = job_dir / "picks.json"
    picks = None
    if ppath.exists():
        try:
            picks = [
                ClipPick.model_validate(d) for d in json.loads(ppath.read_text(encoding="utf-8"))
            ]
            log("[4/5] select: skipped (using cached picks.json - delete it to re-select)")
        except (ValueError, TypeError):
            log("warning: picks.json is corrupt; deleting and re-running selection")
            ppath.unlink()
    if picks is None:
        log("[4/5] select: started")
        raw_payloads: list[dict] = []
        picks = select_clips(transcript, llm, settings, raw_sink=raw_payloads)
        atomic_write_text(
            job_dir / "select_raw.json", json.dumps(raw_payloads, indent=2, ensure_ascii=False)
        )
        if not picks:
            raise NoClipsSelectedError(
                f"No clips were selected — see workspace/{job_id}/select_raw.json"
            )
        atomic_write_text(
            ppath,
            json.dumps([p.model_dump(mode="json") for p in picks], indent=2, ensure_ascii=False),
        )
        log(f"[4/5] select: done ({len(picks)} clips)")

    log("[5/5] cut: started")
    clips = cut_clips(src, picks, settings.output_dir / job_id, pick_encoder())
    log(f"[5/5] cut: done ({len(clips)} clips)")
    return clips
