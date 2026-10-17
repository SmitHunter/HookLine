import json
import os
from pathlib import Path

import pytest

from hookline.config import Settings
from hookline.transcribe import get_transcriber
from hookline.transcribe.base import TranscriptionError
from hookline.transcribe.faster_whisper import FasterWhisperBackend
from hookline.transcribe.whisper_cpp import WhisperCppBackend, dtw_preset, parse_whisper_cpp_json

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_whisper_cpp_json():
    data = json.loads((FIXTURES / "whisper_cpp_sample.json").read_text(encoding="utf-8"))
    t = parse_whisper_cpp_json(data, duration=5.0)
    words = t.words()
    assert [x.text for x in words][:3] == ["Okay", "chat,", "watch"]
    assert words[0].start == pytest.approx(0.32) and words[0].end == pytest.approx(0.61)
    assert all(x.text.strip() for x in words)
    assert len(words) == 8
    assert t.language == "en" and t.duration == 5.0


def test_parse_groups_words_into_sentence_segments():
    data = json.loads((FIXTURES / "whisper_cpp_sample.json").read_text(encoding="utf-8"))
    t = parse_whisper_cpp_json(data, duration=5.0)
    assert [s.text for s in t.segments] == ["Okay chat, watch this.", "No way that worked!"]
    assert [len(s.words) for s in t.segments] == [4, 4]
    assert t.segments[1].start == pytest.approx(2.9)


def test_get_transcriber_auto_falls_back_to_faster_whisper(tmp_path):
    s = Settings(whisper_cpp_bin=tmp_path / "nope.exe", whisper_cpp_model=tmp_path / "nope.bin")
    assert isinstance(get_transcriber(s), FasterWhisperBackend)


def test_get_transcriber_auto_prefers_whisper_cpp_when_present(tmp_path):
    (tmp_path / "w.exe").write_bytes(b"")
    (tmp_path / "m.bin").write_bytes(b"")
    s = Settings(whisper_cpp_bin=tmp_path / "w.exe", whisper_cpp_model=tmp_path / "m.bin")
    assert isinstance(get_transcriber(s), WhisperCppBackend)


@pytest.mark.integration
def test_faster_whisper_tiny_on_speech(speech_wav):
    t = FasterWhisperBackend("tiny.en", "cpu", "int8").transcribe(speech_wav)
    joined = "".join(w.text.lower() for w in t.words()).replace(" ", "")
    assert "hookline" in joined


def test_parse_drops_non_speech_annotations():
    data = json.loads((FIXTURES / "whisper_cpp_sample.json").read_text(encoding="utf-8"))
    t = parse_whisper_cpp_json(data, duration=5.0)
    texts = [w.text for w in t.words()]
    assert not any(x[0] in "[(*" for x in texts)
    assert len(texts) == 8


def test_whisper_cpp_missing_output_json_raises_transcription_error(tmp_path):
    fake = tmp_path / "fake.cmd"
    fake.write_text("@echo off\r\nexit /b 0\r\n")
    with pytest.raises(TranscriptionError):
        WhisperCppBackend(fake, tmp_path / "m.bin").transcribe(tmp_path / "a.wav")


@pytest.mark.integration
def test_whisper_cpp_vulkan_on_speech(speech_wav):
    bin_ = os.environ.get("HOOKLINE_WHISPER_CPP_BIN")
    model = os.environ.get("HOOKLINE_WHISPER_CPP_MODEL")
    if not (bin_ and model and Path(bin_).is_file() and Path(model).is_file()):
        pytest.skip("HOOKLINE_WHISPER_CPP_BIN/MODEL not set")
    t = WhisperCppBackend(Path(bin_), Path(model)).transcribe(speech_wav)
    joined = " ".join(w.text.lower() for w in t.words())
    assert "welcome" in joined and "test" in joined


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_dtw_preset():
    assert dtw_preset(Path("ggml-large-v3-turbo.bin")) == "large.v3.turbo"
    assert dtw_preset(Path("x/ggml-base.en.bin")) == "base.en"


def test_parse_prefers_dtw_and_matches_real_pauses():
    t = parse_whisper_cpp_json(_load("whisper_cpp_dtw_sample.json"), duration=60.0)
    words = t.words()
    assert t.timing == "dtw"
    assert 1.0 <= words[0].start <= 1.4  # audio: silence until 1.11s (offsets said 0.00)
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    assert all(w.end > w.start for w in words)
    after_pause = next(w for w in words if w.start > 5.5)  # silence 5.20-6.22 in the audio
    assert 6.1 <= after_pause.start <= 6.7


def test_parse_falls_back_to_offsets_without_dtw():
    t = parse_whisper_cpp_json(_load("whisper_cpp_sample.json"), duration=5.0)
    assert t.timing == "segment" and t.words()[0].start == pytest.approx(0.32)


def test_backend_passes_dtw_flags(monkeypatch, tmp_path):
    argv: list[str] = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            argv.extend(cmd)
            stem = Path(cmd[cmd.index("-of") + 1])
            Path(f"{stem}.json").write_text(
                json.dumps({"result": {"language": "en"}, "transcription": []}),
                encoding="utf-8",
            )
            self.stderr = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr("hookline.transcribe.whisper_cpp.subprocess.Popen", FakePopen)
    monkeypatch.setattr("hookline.transcribe.whisper_cpp.probe_duration", lambda p: 1.0)
    WhisperCppBackend(tmp_path / "w.exe", tmp_path / "ggml-large-v3-turbo.bin").transcribe(
        tmp_path / "a.wav"
    )
    assert "-nfa" in argv and argv[argv.index("-dtw") + 1] == "large.v3.turbo"
