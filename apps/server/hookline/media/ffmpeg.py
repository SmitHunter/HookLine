import functools
import json
import os
import re
import subprocess
from pathlib import Path

_ENCODER_PREFERENCE = ("h264_nvenc", "h264_amf", "libx264")

# Rate control per encoder: without it hardware encoders default to very high bitrates
# (h264_amf ~19 Mbps), producing huge clips from modest sources.
_ENCODER_QUALITY_ARGS: dict[str, list[str]] = {
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    # The requested QVBR mode (-rc qvbr / hqvbr) fails to initialise on this AMD driver (error 4),
    # so use constant-QP: quality-driven like CRF, bitrate scales with the content.
    "h264_amf": ["-rc", "cqp", "-qp_i", "22", "-qp_p", "24", "-qp_b", "26"],
    "h264_nvenc": ["-rc", "vbr", "-cq", "21", "-b:v", "0"],
}


def encoder_args(encoder: str) -> list[str]:
    """`-c:v <encoder>` plus that encoder's rate-control options."""
    return ["-c:v", encoder, *_ENCODER_QUALITY_ARGS.get(encoder, [])]


class FFmpegError(Exception):
    """An ffmpeg/ffprobe invocation failed; carries the last 20 lines of stderr."""

    def __init__(self, message: str, stderr: str = "") -> None:
        self.stderr = "\n".join(stderr.strip().splitlines()[-20:])
        super().__init__(f"{message}\n{self.stderr}" if self.stderr else message)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        res = subprocess.run(
            cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as e:
        raise FFmpegError(f"{cmd[0]} not found on PATH") from e
    if res.returncode != 0:
        raise FFmpegError(f"{cmd[0]} exited with code {res.returncode}", res.stderr)
    return res


def probe_duration(path: Path) -> float:
    res = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return float(json.loads(res.stdout)["format"]["duration"])


def extract_audio(src: Path, dst: Path) -> Path:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


@functools.cache
def pick_encoder() -> str:
    listing = _run(["ffmpeg", "-hide_banner", "-encoders"]).stdout
    available = set(re.findall(r"^\s*V\S+\s+(\S+)", listing, re.MULTILINE))
    for name in _ENCODER_PREFERENCE:
        if name not in available:
            continue
        try:
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=256x256:rate=25",
                    "-frames:v",
                    "1",
                    "-c:v",
                    name,
                    "-f",
                    "null",
                    "-",
                ]
            )
        except FFmpegError:
            continue
        return name
    return "libx264"


def cut_segments(src: Path, segments: list[tuple[float, float]], dst: Path, encoder: str) -> Path:
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for start, end in segments:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src)]
    n = len(segments)
    inputs = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    cmd += [
        "-filter_complex",
        f"{inputs}concat=n={n}:v=1:a=1[v][a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        *encoder_args(encoder),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    tmp = dst.with_suffix(".tmp.mp4")
    _run([*cmd, str(tmp)])
    os.replace(tmp, dst)
    return dst
