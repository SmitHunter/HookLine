import json
import subprocess
import sys
from pathlib import Path

import pytest

from hookline.config import Settings
from hookline.media.ffmpeg import extract_audio
from hookline.models import ClipPick, RubricScores, Segment, Sentence, Transcript, Word
from hookline.transcript_text import split_sentences


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("my clips 🎮")
    out = d / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25:duration=150",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=150",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out


def ffprobe_stream(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(res.stdout)["streams"][0]


@pytest.fixture
def sample_pick() -> ClipPick:
    return ClipPick(
        start=30.0,
        end=55.0,
        cold_open=None,
        title="Sample clip",
        hook_text="Wait for it",
        reason="test",
        scores=RubricScores(
            hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
        ),
        edit_beats=[],
    )


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "win32":
        pytest.skip("speech fixture needs Windows SAPI")
    d = tmp_path_factory.mktemp("speech")
    raw = d / "raw.wav"
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{raw}'); "
        "$s.Speak('Welcome to hookline. This is a test.'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    return extract_audio(raw, d / "speech.wav")


@pytest.fixture
def fake_anthropic_client_factory():
    from contextlib import contextmanager
    from types import SimpleNamespace

    def factory(stop_reason: str = "end_turn", text: str = "{}"):
        calls: list[dict] = []

        @contextmanager
        def stream(**kw):
            calls.append(kw)
            msg = SimpleNamespace(
                stop_reason=stop_reason,
                content=[SimpleNamespace(type="text", text=text)],
            )
            yield SimpleNamespace(get_final_message=lambda: msg)

        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
        client.calls = calls
        return client

    return factory


class FakeLLM:
    """LLMBackend double: returns queued dicts / raises queued exceptions, in order."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._queue: list[dict | Exception] = []
        self._forever: dict | Exception | None = None

    def queue(self, *items: dict | Exception) -> None:
        self._queue.extend(items)

    def queue_forever(self, item: dict | Exception) -> None:
        self._forever = item

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "model": model})
        if self._queue:
            item = self._queue.pop(0)
        elif self._forever is not None:
            item = self._forever
        else:
            raise AssertionError("FakeLLM called with nothing queued")
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings() -> Settings:
    return Settings()


def _synthetic_transcript(duration: float, every: float = 3.0) -> Transcript:
    """4-word sentences ending in "." starting every `every` seconds."""
    segments = []
    t = 0.0
    n = 0
    while t + 2.3 <= duration:
        texts = [f"line{n}", "says", "something", "here."]
        words = [
            Word(start=round(t + j * 0.6, 3), end=round(t + j * 0.6 + 0.5, 3), text=txt)
            for j, txt in enumerate(texts)
        ]
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=" ".join(texts), words=words)
        )
        t = round(t + every, 3)
        n += 1
    return Transcript(language="en", duration=duration, segments=segments)


@pytest.fixture
def long_transcript() -> Transcript:
    return _synthetic_transcript(600.0)


@pytest.fixture
def long_sentences(long_transcript: Transcript) -> list[Sentence]:
    return split_sentences(long_transcript.words())


@pytest.fixture
def short_transcript() -> Transcript:
    return _synthetic_transcript(60.0)


def _lavfi_wav(dst: Path, *sources: str) -> Path:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for s in sources:
        cmd += ["-f", "lavfi", "-i", s]
    n = len(sources)
    cmd += ["-filter_complex", f"{''.join(f'[{i}:a]' for i in range(n))}concat=n={n}:v=0:a=1[a]"]
    cmd += ["-map", "[a]", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


@pytest.fixture(scope="session")
def tone_gap_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """16 kHz mono: 1 s tone, 1 s silence, 2 s tone."""
    return _lavfi_wav(
        tmp_path_factory.mktemp("audio") / "tone_gap.wav",
        "sine=frequency=440:sample_rate=16000:duration=1",
        "anullsrc=r=16000:cl=mono:d=1",
        "sine=frequency=440:sample_rate=16000:duration=2",
    )


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """16 kHz mono: 3 s continuous tone."""
    return _lavfi_wav(
        tmp_path_factory.mktemp("audio") / "tone.wav",
        "sine=frequency=440:sample_rate=16000:duration=3",
    )
