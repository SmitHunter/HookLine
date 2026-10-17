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


def dtw_preset(model: Path) -> str:
    """ggml-large-v3-turbo.bin -> large.v3.turbo (whisper.cpp's -dtw preset name)."""
    return model.stem.removeprefix("ggml-").replace("-", ".")


def _is_speech(text: str) -> bool:
    text = text.strip()
    return bool(text) and not text.startswith("[")


def parse_whisper_cpp_json(data: dict, duration: float) -> Transcript:
    # Each entry: (text, offset_start, offset_end, dtw_start | None, dtw_last | None)
    entries: list[tuple[str, float, float, float | None, float | None]] = []
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
        entries.append(
            (
                text,
                e["offsets"]["from"] / 1000,
                e["offsets"]["to"] / 1000,
                dtws[0] if has_first else None,
                dtws[-1] if dtws else None,
            )
        )
    language = (data.get("result") or {}).get("language", "en")
    n_dtw = sum(1 for en in entries if en[3] is not None)
    use_dtw = bool(entries) and n_dtw * 2 >= len(entries)
    if not use_dtw:
        words = [Word(start=s, end=t, text=x) for x, s, t, _, _ in entries]
        return words_to_transcript(words, language, duration)
    starts: list[float] = []
    prev = 0.0
    for _, off_from, _, d_start, _ in entries:
        prev = max(d_start if d_start is not None else off_from, prev)
        starts.append(prev)
    words = []
    for i, (text, _, off_to, _, d_last) in enumerate(entries):
        start = starts[i]
        if i + 1 < len(entries):
            end = starts[i + 1]
            if d_last is not None:
                end = min(end, d_last + 0.15)
        else:
            end = off_to
        words.append(Word(start=start, end=max(end, start + 0.05), text=text))
    return words_to_transcript(words, language, duration, timing="dtw")


class WhisperCppBackend:
    def __init__(self, bin: Path, model: Path) -> None:
        self.bin = bin
        self.model = model

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
                "-nfa",
                "-dtw",
                dtw_preset(self.model),
            ]
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
            assert proc.stderr is not None
            for line in proc.stderr:
                tail = [*tail[-19:], line.rstrip()]
                if on_progress and (m := _PROGRESS.search(line)):
                    on_progress(int(m.group(1)) / 100)
            code = proc.wait()
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
