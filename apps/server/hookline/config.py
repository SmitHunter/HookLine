import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator


def repo_root() -> Path:
    """Repo root, found by walking up from this file to the build script or .git."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "build-whisper-vulkan.ps1").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]


class Settings(BaseModel):
    workspace_dir: Path = Path("workspace")
    output_dir: Path = Path("output")
    llm_backend: Literal["claude-code", "api"] = "claude-code"
    select_model: str = "claude-opus-5-5"
    transcriber: Literal["auto", "faster-whisper", "whisper-cpp"] = "auto"
    whisper_model: str = "large-v3-turbo"
    whisper_cpp_bin: Path | None = None
    whisper_cpp_model: Path | None = None
    clips: int = 5
    min_clip_s: float = 20
    max_clip_s: float = 60
    short_input_s: float = 120
    max_prompt_tokens: int = 600_000
    cookies_from_browser: str | None = None

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
