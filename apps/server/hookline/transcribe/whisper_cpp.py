import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from hookline.media.ffmpeg import probe_duration
from hookline.models import Transcript, Word
from hookline.transcribe.base import TranscriptionError, words_to_transcript

_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")
# Whole-token annotations like [BLANK_AUDIO], [_BEG_], (laughs), *sighs* are not speech.
_NON_SPEECH = re.compile(r"^(\[[^\]]*\]|\([^)]*\)|\*[^*]*\*)$")


# whisper.cpp's built-in -dtw alignment-head presets.
DTW_PRESETS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large.v1",
    "large.v2",
    "large.v3",
    "large.v3.turbo",
)
_QUANT_SUFFIX = re.compile(r"-(q\d_\d|q\d_k|f16|f32)$", re.IGNORECASE)
# "ggml_vulkan: 0 = AMD Radeon RX 7800 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | ..."
_VK_DEVICE = re.compile(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|\s*uma:\s*([01])")


def dtw_preset(model: Path) -> str | None:
    """ggml-large-v3-turbo-q5_0.bin -> large.v3.turbo (whisper.cpp's -dtw preset name), by
    longest-prefix match against the known presets; None when no preset fits."""
    name = _QUANT_SUFFIX.sub("", model.stem.removeprefix("ggml-")).replace("-", ".").lower()
    hits = [p for p in DTW_PRESETS if name == p or name.startswith(p + ".")]
    return max(hits, key=len, default=None)


def parse_vulkan_devices(lines: list[str]) -> dict[int, tuple[str, bool]]:
    """`ggml_vulkan: <idx> = <name> ... | uma: <0|1>` lines -> {idx: (name, is_uma)}."""
    devices: dict[int, tuple[str, bool]] = {}
    for line in lines:
        if m := _VK_DEVICE.search(line):
            devices[int(m.group(1))] = (m.group(2), m.group(3) == "1")
    return devices


def device_report(devices: dict[int, tuple[str, bool]], used: int | None) -> list[str]:
    """Log lines for the Vulkan device whisper.cpp runs on (`used`, default 0)."""
    idx = used if used is not None else 0
    if idx not in devices:
        return []
    name, uma = devices[idx]
    lines = [f"transcribe: device {name}"]
    if uma:
        lines.append(
            "warning: transcribe: running on an integrated GPU (uma) — check your discrete GPU driver"
        )
        discrete = [i for i, (_, u) in sorted(devices.items()) if not u]
        if discrete:
            lines.append(
                f"hint: set HOOKLINE_WHISPER_DEVICE={discrete[0]} to use {devices[discrete[0]][0]}"
            )
    return lines


_WORDISH = re.compile(r"\w")


def _is_speech(text: str) -> bool:
    text = text.strip()
    return bool(text) and not text.startswith("[")


def parse_whisper_cpp_json(data: dict, duration: float) -> Transcript:
    # Each entry: (text, offset_start, offset_end, dtw_start | None, dtw_last | None, prob)
    entries: list[tuple[str, float, float, float | None, float | None, float]] = []
    for e in data.get("transcription", []):
        text = e.get("text", "").strip()
        if not text or _NON_SPEECH.match(text):
            continue
        dtws = [
            tok["t_dtw"] / 100
            for tok in e.get("tokens", [])
            if _is_speech(tok.get("text", "")) and tok.get("t_dtw", -1) >= 0
        ]
        # Ignore DTW unless the first speech token (not just a later one) has it.
        first_tok = next((t for t in e.get("tokens", []) if _is_speech(t.get("text", ""))), None)
        has_first = first_tok is not None and first_tok.get("t_dtw", -1) >= 0
        # the word's confidence: its least likely speech token (punctuation tokens excluded)
        probs = [
            float(tok["p"])
            for tok in e.get("tokens", [])
            if _is_speech(tok.get("text", "")) and "p" in tok and _WORDISH.search(tok["text"])
        ]
        entries.append(
            (
                text,
                e["offsets"]["from"] / 1000,
                e["offsets"]["to"] / 1000,
                dtws[0] if has_first else None,
                dtws[-1] if dtws else None,
                min(probs, default=1.0),
            )
        )
    language = (data.get("result") or {}).get("language", "en")
    n_dtw = sum(1 for en in entries if en[3] is not None)
    use_dtw = bool(entries) and n_dtw * 2 >= len(entries)
    if not use_dtw:
        words = [Word(start=s, end=t, text=x, prob=p) for x, s, t, _, _, p in entries]
        return words_to_transcript(words, language, duration)
    starts: list[float] = []
    prev = 0.0
    for _, off_from, _, d_start, _, _ in entries:
        prev = max(d_start if d_start is not None else off_from, prev)
        starts.append(prev)
    words = []
    for i, (text, _, off_to, _, d_last, prob) in enumerate(entries):
        start = starts[i]
        if i + 1 < len(entries):
            end = starts[i + 1]
            if d_last is not None:
                end = min(end, d_last + 0.15)
        else:
            end = off_to
        words.append(Word(start=start, end=max(end, start + 0.05), text=text, prob=prob))
    return words_to_transcript(words, language, duration, timing="dtw")


class WhisperCppBackend:
    def __init__(
        self,
        bin: Path,
        model: Path,
        device: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.bin = bin
        self.model = model
        self.device = device
        self.log = log or (lambda _m: None)

    @property
    def cache_key(self) -> dict:
        """What the transcript depends on; a cached transcript with another key is stale."""
        return {"backend": "whisper-cpp", "model": self.model.name, "dtw": dtw_preset(self.model)}

    def transcribe(
        self, wav: Path, on_progress: Callable[[float], None] | None = None
    ) -> Transcript:
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "out"
            cmd = [
                str(self.bin),
                "-m",
                str(self.model),
                "-f",
                str(wav),
                "-ojf",
                "-of",
                str(stem),
                "-ml",
                "1",
                "-sow",
                "-pp",
            ]
            preset = dtw_preset(self.model)
            if preset:
                cmd += ["-nfa", "-dtw", preset]
            else:
                self.log(f"transcribe: no DTW preset for {self.model.name}; using segment timing")
            if self.device is not None:
                cmd += ["-dev", str(self.device)]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as e:
                raise TranscriptionError(f"could not run whisper.cpp ({self.bin}): {e}") from e
            tail: list[str] = []
            vk_lines: list[str] = []
            reported = False
            assert proc.stderr is not None
            for line in proc.stderr:
                tail = [*tail[-19:], line.rstrip()]
                if _VK_DEVICE.search(line):
                    vk_lines.append(line)
                elif vk_lines and not reported:
                    reported = self._report_device(vk_lines)
                if on_progress and (m := _PROGRESS.search(line)):
                    on_progress(int(m.group(1)) / 100)
            code = proc.wait()
            if not reported:
                self._report_device(vk_lines)
            if code != 0:
                raise TranscriptionError(f"whisper.cpp exited with code {code}\n" + "\n".join(tail))
            try:
                data = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise TranscriptionError(
                    f"whisper.cpp produced no readable JSON output: {e}"
                ) from e
        if on_progress:
            on_progress(1.0)
        return parse_whisper_cpp_json(data, probe_duration(wav))

    def _report_device(self, vk_lines: list[str]) -> bool:
        for msg in device_report(parse_vulkan_devices(vk_lines), self.device):
            self.log(msg)
        return True
