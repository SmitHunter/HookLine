import json
import os
from pathlib import Path

import pytest

from hookline.config import Settings
from hookline.transcribe import get_transcriber
from hookline.transcribe.base import TranscriptionError
from hookline.transcribe.faster_whisper import FasterWhisperBackend
from hookline.transcribe.whisper_cpp import (
    WhisperCppBackend,
    dtw_preset,
    parse_vulkan_devices,
    parse_whisper_cpp_json,
)

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
    assert dtw_preset(Path("ggml-large-v3-turbo-q5_0.bin")) == "large.v3.turbo"
    assert dtw_preset(Path("ggml-medium.en-q8_0.bin")) == "medium.en"
    assert dtw_preset(Path("ggml-large-v3.bin")) == "large.v3"
    assert dtw_preset(Path("ggml-small-f16.bin")) == "small"
    assert dtw_preset(Path("ggml-tiny.en-q4_k.bin")) == "tiny.en"
    assert dtw_preset(Path("custom.bin")) is None


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


def _fake_popen(argv: list[str], stderr_lines: list[str] | None = None):
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            argv.extend(cmd)
            stem = Path(cmd[cmd.index("-of") + 1])
            Path(f"{stem}.json").write_text(
                json.dumps({"result": {"language": "en"}, "transcription": []}),
                encoding="utf-8",
            )
            self.stderr = iter(stderr_lines or [])

        def wait(self):
            return 0

    return FakePopen


def test_backend_without_dtw_preset_omits_dtw_flags(monkeypatch, tmp_path):
    argv: list[str] = []
    logs: list[str] = []
    monkeypatch.setattr("hookline.transcribe.whisper_cpp.subprocess.Popen", _fake_popen(argv))
    monkeypatch.setattr("hookline.transcribe.whisper_cpp.probe_duration", lambda p: 1.0)
    WhisperCppBackend(tmp_path / "w.exe", tmp_path / "custom.bin", log=logs.append).transcribe(
        tmp_path / "a.wav"
    )
    assert "-dtw" not in argv and "-nfa" not in argv
    assert "transcribe: no DTW preset for custom.bin; using segment timing" in logs


VK_STDERR = [
    "ggml_vulkan: Found 2 Vulkan devices:\n",
    "ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64\n",
    "ggml_vulkan: 1 = AMD Radeon RX 7800 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 64\n",
    "whisper_init_with_params_no_state: use gpu    = 1\n",
    "whisper_print_progress_callback: progress =  50%\n",
]


def test_parse_vulkan_devices():
    devices = parse_vulkan_devices(VK_STDERR)
    assert devices == {
        0: ("AMD Radeon(TM) Graphics (AMD proprietary driver)", True),
        1: ("AMD Radeon RX 7800 XT (AMD proprietary driver)", False),
    }


def test_backend_logs_device_and_warns_on_uma(monkeypatch, tmp_path):
    argv: list[str] = []
    logs: list[str] = []
    monkeypatch.setattr(
        "hookline.transcribe.whisper_cpp.subprocess.Popen", _fake_popen(argv, VK_STDERR)
    )
    monkeypatch.setattr("hookline.transcribe.whisper_cpp.probe_duration", lambda p: 1.0)
    model = tmp_path / "ggml-large-v3-turbo.bin"
    WhisperCppBackend(tmp_path / "w.exe", model, log=logs.append).transcribe(tmp_path / "a.wav")
    assert "-dev" not in argv
    assert "transcribe: device AMD Radeon(TM) Graphics (AMD proprietary driver)" in logs
    assert any("integrated GPU (uma)" in m and "discrete GPU driver" in m for m in logs)
    assert any("HOOKLINE_WHISPER_DEVICE=1" in m for m in logs)


def test_backend_passes_device_and_logs_discrete(monkeypatch, tmp_path):
    argv: list[str] = []
    logs: list[str] = []
    monkeypatch.setattr(
        "hookline.transcribe.whisper_cpp.subprocess.Popen", _fake_popen(argv, VK_STDERR)
    )
    monkeypatch.setattr("hookline.transcribe.whisper_cpp.probe_duration", lambda p: 1.0)
    model = tmp_path / "ggml-large-v3-turbo.bin"
    WhisperCppBackend(tmp_path / "w.exe", model, device=1, log=logs.append).transcribe(
        tmp_path / "a.wav"
    )
    assert argv[argv.index("-dev") + 1] == "1"
    assert logs == ["transcribe: device AMD Radeon RX 7800 XT (AMD proprietary driver)"]


def test_whisper_device_setting_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOOKLINE_WHISPER_DEVICE", "1")
    (tmp_path / "w.exe").write_bytes(b"")
    (tmp_path / "m.bin").write_bytes(b"")
    s = Settings.from_env().model_copy(
        update={"whisper_cpp_bin": tmp_path / "w.exe", "whisper_cpp_model": tmp_path / "m.bin"}
    )
    assert s.whisper_device == 1
    t = get_transcriber(s)
    assert isinstance(t, WhisperCppBackend) and t.device == 1


def test_backend_cache_keys(tmp_path):
    cpp = WhisperCppBackend(tmp_path / "w.exe", tmp_path / "ggml-medium.en-q8_0.bin")
    assert cpp.cache_key == {
        "backend": "whisper-cpp",
        "model": "ggml-medium.en-q8_0.bin",
        "dtw": "medium.en",
    }
    fw = FasterWhisperBackend("large-v3-turbo", "cpu", "int8")
    assert fw.cache_key == {"backend": "faster-whisper", "model": "large-v3-turbo", "dtw": None}


def _w(text: str, start: float, end: float | None = None):
    from hookline.models import Word

    return Word(start=start, end=end if end is not None else start + 0.3, text=text)


def test_dedupe_removes_repeated_run():
    from hookline.transcribe.clean import dedupe_segments

    run = ["We", "were", "a", "little", "older"]
    words = [_w(t, 1817.4 + i * 0.3) for i, t in enumerate(run)]
    words += [_w(t, 1817.4 + i * 0.3) for i, t in enumerate(run)]
    words += [_w("then", 1820.0)]
    out = dedupe_segments(words)
    assert [w.text for w in out] == [*run, "then"]


def test_dedupe_handles_triple_and_punctuation():
    from hookline.transcribe.clean import dedupe_segments

    words = []
    for _ in range(3):
        words += [_w("It's", 10.0), _w("epic.", 10.3)]
    assert [w.text for w in dedupe_segments(words)] == ["It's", "epic."]


def test_dedupe_keeps_genuine_repeat_far_apart():
    from hookline.transcribe.clean import dedupe_segments

    words = [_w("no", 1.0), _w("way", 1.3), _w("no", 5.0), _w("way", 5.3)]
    assert len(dedupe_segments(words)) == 4


def test_spread_collapsed_words():
    from hookline.transcribe.clean import spread_collapsed

    words = [_w("a", 1817.40, 1817.45), _w("b", 1817.40, 1817.45), _w("c", 1817.40, 1817.45)]
    words.append(_w("d", 1818.0, 1818.3))
    out = spread_collapsed(words)
    starts = [w.start for w in out]
    assert starts == pytest.approx([1817.40, 1817.60, 1817.80, 1818.0])
    assert out[0].end == pytest.approx(1817.60) and out[2].end == pytest.approx(1817.88)
    assert all(w.end - w.start >= 0.08 - 1e-9 for w in out)


def test_spread_leaves_run_when_no_room():
    from hookline.transcribe.clean import spread_collapsed

    words = [_w("a", 5.0, 5.05), _w("b", 5.0, 5.05), _w("c", 5.1, 5.3)]
    out = spread_collapsed(words)
    assert [w.start for w in out] == [5.0, 5.0, 5.1]


def test_words_to_transcript_applies_cleaning():
    from hookline.transcribe.base import words_to_transcript

    words = [_w("hi", 1.0, 1.05), _w("there.", 1.0, 1.05), _w("ok", 2.0, 2.3)]
    t = words_to_transcript(words, "en", 5.0, timing="dtw")
    assert [w.start for w in t.words()] == pytest.approx([1.0, 1.5, 2.0])
