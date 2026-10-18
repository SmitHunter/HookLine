import functools
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

_ENCODER_PREFERENCE = ("h264_nvenc", "h264_amf", "libx264")

# Rate control per encoder: without it hardware encoders default to very high bitrates
# (h264_amf ~19 Mbps), producing huge clips from modest sources.
_ENCODER_QUALITY_ARGS: dict[str, list[str]] = {
    # CRF 17 capped at 10 Mb/s (VBV) keeps detail on upscaled faces without runaway sizes.
    "libx264": ["-preset", "veryfast", "-crf", "17", "-maxrate", "10M", "-bufsize", "20M"],
    # The requested QVBR mode (-rc qvbr / hqvbr) fails to initialise on this AMD driver (error 4),
    # so use constant-QP: quality-driven like CRF, bitrate scales with the content. QP 16/18/20
    # (wave G-select): soft dark sources posterise at 18/20/22; B-frames added when supported.
    "h264_amf": ["-rc", "cqp", "-qp_i", "16", "-qp_p", "18", "-qp_b", "20"],
    "h264_nvenc": ["-rc", "vbr", "-cq", "18", "-b:v", "0"],
}


# Windows caps a whole command line at ~32K chars; graphs longer than this go via a script file.
INLINE_GRAPH_MAX = 8000


def filter_graph_args(graph: str, script: Path, force: bool = False) -> list[str]:
    """`-filter_complex <graph>` inline, or (when longer than INLINE_GRAPH_MAX, or `force`)
    `-/filter_complex <script>` with the graph written to `script` (ffmpeg >= 7 syntax)."""
    if not force and len(graph) <= INLINE_GRAPH_MAX:
        return ["-filter_complex", graph]
    tmp = script.with_name(script.name + ".tmp")
    tmp.write_text(graph, encoding="utf-8")
    os.replace(tmp, script)
    return ["-/filter_complex", str(script)]


def encoder_args(encoder: str) -> list[str]:
    """`-c:v <encoder>` plus that encoder's rate-control options (AMF: `-bf 2` when the
    hardware takes B-frames, see `amf_b_frames`)."""
    extra = ["-bf", "2"] if encoder == "h264_amf" and amf_b_frames() else []
    return ["-c:v", encoder, *_ENCODER_QUALITY_ARGS.get(encoder, []), *extra]


@functools.cache
def amf_b_frames() -> bool:
    """Whether h264_amf really writes B-frames with `-bf 2` (probed once: a 10-frame encode
    whose stream must report a B-frame delay). False on any failure: no AMF, an older GPU
    that refuses B-frames, or one that silently drops them."""
    with tempfile.TemporaryDirectory(prefix="hookline-bf-") as tmp:
        out = Path(tmp) / "bf.mp4"
        try:
            _run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=25",
                    "-frames:v", "10", "-c:v", "h264_amf", *_ENCODER_QUALITY_ARGS["h264_amf"],
                    "-bf", "2", str(out),
                ]
            )  # fmt: skip
            res = _run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=has_b_frames", "-of", "csv=p=0", str(out)]
            )  # fmt: skip
            return int(res.stdout.strip() or 0) > 0
        except (FFmpegError, ValueError):
            return False


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


def probe_size(path: Path) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    res = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(res.stdout).get("streams") or []
    if not streams:
        raise FFmpegError(f"no video stream in {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def has_audio(path: Path) -> bool:
    res = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)]
    )  # fmt: skip
    return bool(res.stdout.strip())


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


def probe_fps(path: Path) -> float:
    """Frame rate of the first video stream (r_frame_rate, falling back to avg_frame_rate)."""
    res = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(path),
        ]
    )  # fmt: skip
    streams = json.loads(res.stdout).get("streams") or []
    if not streams:
        raise FFmpegError(f"no video stream in {path}")
    for key in ("r_frame_rate", "avg_frame_rate"):
        num, _, den = str(streams[0].get(key, "0/0")).partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 1 <= fps <= 240:
            return fps
    raise FFmpegError(f"unknown frame rate for {path}")


_SEEK_PREROLL_S = 1.0


def cut_segments(
    src: Path,
    segments: list[tuple[float, float]],
    dst: Path,
    encoder: str,
    quality: list[str] | None = None,
    fps: float | None = None,
) -> Path:
    """Concatenate `segments` of `src` into `dst` (an intermediate: use `.mkv`, audio is
    uncompressed `pcm_s16le` so the final encode is the only lossy audio step). `quality`, when given, replaces the
    encoder's default rate-control args.

    Each piece is decoded from a seek point ~1 s early and cut with trim/atrim, so the output
    timeline is exactly the sum of the segments when their bounds lie on the source frame
    grid: video takes the frames whose timestamps fall in [start, end) (bounds shifted half a
    frame to absorb rounding), audio is sample-trimmed to the same span."""
    if fps is None:
        fps = probe_fps(src)
    half = 0.5 / fps
    with_audio = has_audio(src)
    video = encoder_args(encoder) if quality is None else ["-c:v", encoder, *quality]
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    parts = []
    for i, (start, end) in enumerate(segments):
        seek = max(0.0, round((start - _SEEK_PREROLL_S) * fps) / fps)
        a, b = start - seek, end - seek
        cmd += ["-ss", f"{seek:.6f}", "-t", f"{b + 1:.6f}", "-i", str(src)]
        parts.append(
            f"[{i}:v]trim=start={max(a - half, 0):.6f}:end={b - half:.6f},"
            f"setpts=PTS-STARTPTS[v{i}]"
            + (
                f";[{i}:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[a{i}]"
                if with_audio
                else ""
            )
        )
    n = len(segments)
    inputs = "".join(f"[v{i}][a{i}]" if with_audio else f"[v{i}]" for i in range(n))
    outs = "[v][a]" if with_audio else "[v]"
    script = dst.with_suffix(".graph.txt")
    graph = ";".join(parts) + f";{inputs}concat=n={n}:v=1:a={int(with_audio)}{outs}"
    cmd += [
        *filter_graph_args(graph, script),
        "-map",
        "[v]",
        *(["-map", "[a]"] if with_audio else []),
        *video,
        "-pix_fmt",
        "yuv420p",
        *(["-c:a", "pcm_s16le"] if with_audio else []),
    ]
    tmp = dst.with_suffix(f".tmp{dst.suffix}")
    try:
        _run([*cmd, str(tmp)])
    finally:
        script.unlink(missing_ok=True)
    os.replace(tmp, dst)
    return dst


def probe_video_kbps(path: Path) -> float | None:
    """The first video stream's bitrate in kb/s (the container's total minus its audio when
    the stream has none recorded, e.g. Matroska); None when it cannot be read."""
    try:
        res = _run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,bit_rate:format=bit_rate",
             "-of", "json", str(path)]
        )  # fmt: skip
        info = json.loads(res.stdout)
    except (FFmpegError, ValueError):
        return None
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    try:
        if video.get("bit_rate"):
            return round(int(video["bit_rate"]) / 1000, 1)
        total = int(info.get("format", {}).get("bit_rate", 0))
        audio = sum(int(s.get("bit_rate") or 0) for s in streams if s.get("codec_type") == "audio")
    except ValueError:
        return None
    return round((total - audio) / 1000, 1) if total > audio else None
