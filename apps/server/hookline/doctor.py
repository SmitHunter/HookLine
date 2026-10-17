import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from hookline.assets import TWITCH_DL_EXE, YUNET_URL, tools_dir
from hookline.config import Settings
from hookline.media.subtitles import fonts_dir


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str


def _ffmpeg_output(run: Callable, *args: str) -> str:
    try:
        res = run(
            ["ffmpeg", "-hide_banner", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (res.stdout or "") + (res.stderr or "")


def _lists(output: str, name: str) -> bool:
    """True when a `-filters`/`-encoders` table row has `name` as its second column."""
    return any(line.split()[1:2] == [name] for line in output.splitlines())


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # noqa: BLE001 - any import/driver failure means no CUDA
        return False


def run_doctor(
    which: Callable[[str], str | None] = shutil.which,
    run: Callable = subprocess.run,
) -> list[Check]:
    checks: list[Check] = []
    ffmpeg = which("ffmpeg")
    checks.append(Check("ffmpeg", bool(ffmpeg), True, ffmpeg or "not found on PATH"))

    has_libass = False
    if ffmpeg:
        has_libass = _lists(_ffmpeg_output(run, "-filters"), "subtitles") or (
            "--enable-libass" in _ffmpeg_output(run, "-version")
        )
    checks.append(
        Check("libass", has_libass, True, "subtitles filter available" if has_libass else "missing")
    )

    claude = which("claude")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(Check("claude CLI", bool(claude), False, claude or "not found on PATH"))
    checks.append(
        Check(
            "LLM backend",
            bool(claude or has_key),
            True,
            claude or ("ANTHROPIC_API_KEY set" if has_key else "no claude CLI or API key"),
        )
    )

    ytdlp = importlib.util.find_spec("yt_dlp") is not None
    checks.append(Check("yt-dlp", ytdlp, True, "importable" if ytdlp else "not installed"))

    encoders = _ffmpeg_output(run, "-encoders") if ffmpeg else ""
    for enc in ("h264_nvenc", "h264_amf"):
        found = _lists(encoders, enc)
        checks.append(Check(enc, found, False, "available" if found else "not available"))

    try:
        s = Settings.from_env()
    except ValueError as e:
        checks.append(Check("settings", False, True, str(e).splitlines()[0]))
        s = Settings()
    cpp_ok = bool(
        s.whisper_cpp_bin
        and s.whisper_cpp_model
        and s.whisper_cpp_bin.exists()
        and s.whisper_cpp_model.exists()
    )
    checks.append(
        Check(
            "whisper.cpp",
            cpp_ok,
            False,
            "configured (DTW on)"
            if cpp_ok
            else "not configured (set HOOKLINE_WHISPER_CPP_BIN/_MODEL)",
        )
    )
    font = fonts_dir() / "Anton-Regular.ttf"
    checks.append(
        Check("Anton font", font.exists(), True, str(font) if font.exists() else "missing")
    )
    yunet = tools_dir() / "models" / YUNET_URL.rsplit("/", 1)[-1]
    checks.append(
        Check(
            "YuNet model",
            yunet.exists(),
            False,
            "cached" if yunet.exists() else "downloads on first vertical render",
        )
    )
    tdl = tools_dir() / "bin" / TWITCH_DL_EXE
    checks.append(
        Check(
            "TwitchDownloaderCLI",
            tdl.exists(),
            False,
            "cached" if tdl.exists() else "downloads on first Twitch VOD (chat signals)",
        )
    )
    cuda = _cuda_available()
    checks.append(Check("CUDA", cuda, False, "available" if cuda else "not available"))
    return checks
