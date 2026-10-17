import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from hookline.config import Settings
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


@app.callback()
def _main() -> None:
    """Turn a long video into short clips."""


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="Local video file or URL")],
    clips: Annotated[int | None, typer.Option("--clips", help="Number of clips")] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="claude-code or api")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Selection model")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output directory")] = None,
) -> None:
    """Cut SOURCE into short clips."""
    try:
        overrides: dict = {}
        if clips is not None:
            overrides["clips"] = clips
        if backend is not None:
            overrides["llm_backend"] = backend
        if model is not None:
            overrides["select_model"] = model
        if out is not None:
            overrides["output_dir"] = out
        settings = Settings.model_validate({**Settings.from_env().model_dump(), **overrides})
        transcriber = get_transcriber(settings)
        console.print(f"Transcriber: {_backend_label(transcriber)}", markup=False)
        paths = run_job(
            source,
            settings,
            make_llm(settings),
            transcriber,
            log=lambda m: console.print(m, highlight=False, markup=False),
        )
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
