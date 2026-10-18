import importlib.util
import os
import re
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


# renders pass their filter graph as a file (`-/filter_complex`), new in ffmpeg 7.0
MIN_FFMPEG_MAJOR = 7


def ffmpeg_major(version_output: str) -> tuple[int | None, str]:
    """(major release, version string) from `ffmpeg -version`; major None for a git build
    ("N-118000-g...") or an unreadable banner."""
    m = re.search(r"ffmpeg version (\S+)", version_output)
    if not m:
        return None, ""
    version = m.group(1)
    major = re.match(r"n?(\d+)\.", version)
    return (int(major.group(1)) if major else None), version


def _ffmpeg_check(ffmpeg: str | None, run: Callable) -> Check:
    if not ffmpeg:
        return Check("ffmpeg", False, True, "not found on PATH")
    major, version = ffmpeg_major(_ffmpeg_output(run, "-version"))
    if major is not None and major < MIN_FFMPEG_MAJOR:
        return Check(
            "ffmpeg",
            False,
            True,
            f"ffmpeg {version} at {ffmpeg} is too old: Hookline needs ffmpeg "
            f"{MIN_FFMPEG_MAJOR} or newer (filter scripts via -/filter_complex)",
        )
    return Check("ffmpeg", True, True, f"{ffmpeg} ({version or 'version unknown'})")


def parse_vulkaninfo(summary: str) -> list[tuple[str, str]]:
    """`vulkaninfo --summary` text -> [(deviceName, type)], type like DISCRETE_GPU."""
    devices: list[tuple[str, str]] = []
    kind = None
    for line in summary.splitlines():
        key, _, value = line.strip().partition("=")
        key, value = key.strip(), value.strip()
        if key == "deviceType":
            kind = value.removeprefix("PHYSICAL_DEVICE_TYPE_")
        elif key == "deviceName" and kind is not None:
            devices.append((value, kind))
            kind = None
    return devices


def _gpu_check(which: Callable[[str], str | None], run: Callable) -> Check:
    name = "GPU (whisper.cpp)"
    exe = which("vulkaninfo")
    if not exe:
        return Check(name, False, False, "vulkaninfo not on PATH; GPU not checked")
    try:
        res = run(
            [exe, "--summary"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        devices = parse_vulkaninfo((res.stdout or "") + (res.stderr or ""))
    except (OSError, subprocess.SubprocessError):
        devices = []
    listed = ", ".join(f"{n} ({k.lower()})" for n, k in devices) or "no Vulkan devices"
    discrete = any(k == "DISCRETE_GPU" for _, k in devices)
    detail = listed if discrete else f"discrete GPU not visible — check driver ({listed})"
    return Check(name, discrete, False, detail)


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
    checks.append(_ffmpeg_check(ffmpeg, run))

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
    checks.append(_gpu_check(which, run))
    cuda = _cuda_available()
    checks.append(Check("CUDA", cuda, False, "available" if cuda else "not available"))
    return checks
