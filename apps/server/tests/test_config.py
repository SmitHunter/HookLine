from pathlib import Path

import pytest

from hookline import config
from hookline.config import Settings


def _fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_tools: bool) -> Path:
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(config.sys, "platform", "win32")
    bin_path = tmp_path / "tools" / "bin" / "whisper-cli.exe"
    model_path = tmp_path / "tools" / "models" / "ggml-large-v3-turbo.bin"
    if with_tools:
        bin_path.parent.mkdir(parents=True)
        model_path.parent.mkdir(parents=True)
        bin_path.write_bytes(b"")
        model_path.write_bytes(b"")
    return tmp_path


def test_defaults_to_repo_tools_when_files_exist(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path, monkeypatch, with_tools=True)
    s = Settings()
    assert s.whisper_cpp_bin == root / "tools" / "bin" / "whisper-cli.exe"
    assert s.whisper_cpp_model == root / "tools" / "models" / "ggml-large-v3-turbo.bin"


def test_stays_none_when_tools_missing(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, with_tools=False)
    s = Settings()
    assert s.whisper_cpp_bin is None and s.whisper_cpp_model is None


def test_explicit_values_win(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, with_tools=True)
    s = Settings(whisper_cpp_bin=tmp_path / "x.exe")
    assert s.whisper_cpp_bin == tmp_path / "x.exe"
    assert s.whisper_cpp_model == tmp_path / "tools" / "models" / "ggml-large-v3-turbo.bin"


def test_env_overrides_default(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, with_tools=True)
    monkeypatch.setenv("HOOKLINE_WHISPER_CPP_BIN", str(tmp_path / "other.exe"))
    s = Settings.from_env()
    assert s.whisper_cpp_bin == tmp_path / "other.exe"


def test_non_windows_binary_name(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, with_tools=False)
    monkeypatch.setattr(config.sys, "platform", "linux")
    (tmp_path / "tools" / "bin").mkdir(parents=True)
    (tmp_path / "tools" / "models").mkdir(parents=True)
    (tmp_path / "tools" / "bin" / "whisper-cli").write_bytes(b"")
    (tmp_path / "tools" / "models" / "ggml-large-v3-turbo.bin").write_bytes(b"")
    assert Settings().whisper_cpp_bin == tmp_path / "tools" / "bin" / "whisper-cli"


def test_repo_root_points_at_repo():
    root = config.repo_root()
    assert (root / "scripts" / "build-whisper-vulkan.ps1").exists()


def test_backend_label():
    from hookline.cli import _backend_label
    from hookline.transcribe.faster_whisper import FasterWhisperBackend
    from hookline.transcribe.whisper_cpp import WhisperCppBackend

    assert "whisper.cpp" in _backend_label(WhisperCppBackend(Path("a"), Path("b")))
    assert _backend_label(FasterWhisperBackend("tiny", "cpu", "int8")) == "faster-whisper (cpu)"
