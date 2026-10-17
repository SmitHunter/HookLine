import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from hookline.config import Settings, parse_facecam
from hookline.doctor import run_doctor
from hookline.llm import LLMError, make_llm
from hookline.media.ffmpeg import FFmpegError
from hookline.pipeline.ingest import IngestError
from hookline.pipeline.run import NoClipsSelectedError, run_job
from hookline.transcribe import TranscriptionError, get_transcriber

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _backend_label(transcriber: object) -> str:
    name = type(transcriber).__name__
    if name == "WhisperCppBackend":
        return "whisper.cpp (Vulkan GPU)"
    if name == "FasterWhisperBackend":
        return f"faster-whisper ({getattr(transcriber, 'device', 'cpu')})"
    return name


def _safe_streams() -> None:
    """Replace characters the console encoding can't show (e.g. "→" on a cp1252 pipe)
    instead of crashing mid-run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.callback()
def _main() -> None:
    """Turn a long video into short clips."""
    _safe_streams()


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="Local video file or URL")],
    clips: Annotated[int | None, typer.Option("--clips", help="Number of clips")] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="claude-code or api")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Selection model")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output directory")] = None,
    horizontal: Annotated[
        bool, typer.Option("--horizontal", help="Keep the source aspect (no 9:16 reframe)")
    ] = False,
    no_captions: Annotated[
        bool, typer.Option("--no-captions", help="Do not burn in captions")
    ] = False,
    no_chat: Annotated[
        bool, typer.Option("--no-chat", help="Skip Twitch chat download and chat signals")
    ] = False,
    keep_overlays: Annotated[
        bool,
        typer.Option(
            "--keep-overlays", help="Keep burned-in lower-thirds/edge overlays (no overlay crop)"
        ),
    ] = False,
    facecam: Annotated[
        str | None,
        typer.Option("--facecam", help="Facecam box X,Y,W,H as 0-1 fractions of the frame"),
    ] = None,
) -> None:
    """Cut SOURCE into short clips."""
    overrides: dict = {}
    if facecam is not None:
        try:
            overrides["facecam"] = parse_facecam(facecam)
        except ValueError as e:
            err_console.print(f"Error: --facecam: {e}", style="red", highlight=False, markup=False)
            raise typer.Exit(1) from e
    if horizontal:
        overrides["vertical"] = False
    if no_captions:
        overrides["captions"] = False
    if no_chat:
        overrides["chat"] = False
    if keep_overlays:
        overrides["overlay_crop"] = False
    try:
        if clips is not None:
            overrides["clips"] = clips
        if backend is not None:
            overrides["llm_backend"] = backend
        if model is not None:
            overrides["select_model"] = model
        if out is not None:
            overrides["output_dir"] = out
        settings = Settings.model_validate({**Settings.from_env().model_dump(), **overrides})
        log = lambda m: console.print(m, highlight=False, markup=False)
        transcriber = get_transcriber(settings, log)
        console.print(f"Transcriber: {_backend_label(transcriber)}", markup=False)
        paths = run_job(source, settings, make_llm(settings), transcriber, log=log)
    except (
        LLMError,
        TranscriptionError,
        FFmpegError,
        IngestError,
        FileNotFoundError,
        ValidationError,
        NoClipsSelectedError,
    ) as e:
        err_console.print(
            f"Error: {e}".replace("\n", " "), style="red", highlight=False, markup=False
        )
        raise typer.Exit(1) from e
    for p in paths:
        console.print(str(p), highlight=False, markup=False)
        side = p.with_suffix(".json")
        if side.exists():
            d = json.loads(side.read_text(encoding="utf-8"))
            console.print("  ", end="")
            console.print(d["title"], style="bold", end=" ", highlight=False, markup=False)
            console.print(f"{d['scores']['total']}/100", style="cyan", end=" ", markup=False)
            console.print(f"- {d['hook_text']}", highlight=False, markup=False)


@app.command()
def doctor() -> None:
    """Check that required tools are installed."""
    checks = run_doctor()
    table = Table("Check", "Status", "Detail")
    for c in checks:
        if c.ok:
            status = "[green]OK[/green]"
        elif c.required:
            status = "[red]MISSING[/red]"
        else:
            status = "[yellow]optional[/yellow]"
        table.add_row(
            c.name,
            status,
            c.detail,
        )
    console.print(table)
    if any(c.required and not c.ok for c in checks):
        raise typer.Exit(1)
