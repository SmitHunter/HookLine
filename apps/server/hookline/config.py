import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


def repo_root() -> Path:
    """Repo root, found by walking up from this file to the build script or .git."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "build-whisper-vulkan.ps1").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]


Box = tuple[float, float, float, float]


def parse_facecam(value: str | tuple | list) -> Box:
    """`X,Y,W,H` (normalized 0..1, box inside the frame) -> tuple; ValueError on bad input."""
    parts = value.split(",") if isinstance(value, str) else list(value)
    try:
        box = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        raise ValueError(f"facecam must be X,Y,W,H numbers, got {value!r}") from None
    if len(box) != 4:
        raise ValueError(f"facecam must be 4 numbers X,Y,W,H, got {len(box)}")
    x, y, w, h = box
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("facecam X,Y must be in [0,1) and W,H in (0,1]")
    if x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
        raise ValueError("facecam box must fit inside the frame (X+W <= 1, Y+H <= 1)")
    return (x, y, w, h)


class Settings(BaseModel):
    workspace_dir: Path = Path("workspace")
    output_dir: Path = Path("output")
    llm_backend: Literal["claude-code", "api"] = "claude-code"
    select_model: str = "claude-opus-5-5"
    transcriber: Literal["auto", "faster-whisper", "whisper-cpp"] = "auto"
    whisper_model: str = "large-v3-turbo"
    whisper_cpp_bin: Path | None = None
    whisper_cpp_model: Path | None = None
    whisper_device: int | None = None  # whisper.cpp `-dev` (Vulkan device index)
    clips: int = 5
    min_clip_s: float = 20
    max_clip_s: float = 60
    short_input_s: float = 120
    max_prompt_tokens: int = 600_000
    cookies_from_browser: str | None = None
    vertical: bool = True
    captions: bool = True
    max_pause_s: float = 0.4
    silence_db: float = -32.0
    facecam: Box | None = None
    chat: bool = True

    @field_validator("facecam", mode="before")
    @classmethod
    def _parse_facecam(cls, v: object) -> object:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, (str, tuple, list)):
            return parse_facecam(v)
        return v

    @model_validator(mode="after")
    def _discover_whisper_cpp(self) -> "Settings":
        root = repo_root()
        exe = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
        if self.whisper_cpp_bin is None:
            cand = root / "tools" / "bin" / exe
            if cand.exists():
                self.whisper_cpp_bin = cand
        if self.whisper_cpp_model is None:
            cand = root / "tools" / "models" / "ggml-large-v3-turbo.bin"
            if cand.exists():
                self.whisper_cpp_model = cand
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        values = {
            name: os.environ[f"HOOKLINE_{name.upper()}"]
            for name in cls.model_fields
            if f"HOOKLINE_{name.upper()}" in os.environ
        }
        return cls.model_validate(values)
