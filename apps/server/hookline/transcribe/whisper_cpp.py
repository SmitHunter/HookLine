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


def parse_whisper_cpp_json(data: dict, duration: float) -> Transcript:
    words = [
        Word(
            start=e["offsets"]["from"] / 1000,
            end=e["offsets"]["to"] / 1000,
            text=e["text"].strip(),
        )
        for e in data.get("transcription", [])
        if e.get("text", "").strip() and not _NON_SPEECH.match(e["text"].strip())
    ]
    language = (data.get("result") or {}).get("language", "en")
    return words_to_transcript(words, language, duration)


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
