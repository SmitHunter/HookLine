from hookline.config import Settings
from hookline.transcribe.base import Transcriber, TranscriptionError
from hookline.transcribe.faster_whisper import FasterWhisperBackend
from hookline.transcribe.whisper_cpp import WhisperCppBackend, parse_whisper_cpp_json

__all__ = [
    "FasterWhisperBackend",
    "Transcriber",
    "TranscriptionError",
    "WhisperCppBackend",
    "get_transcriber",
    "parse_whisper_cpp_json",
]


def get_transcriber(settings: Settings) -> Transcriber:
    cpp_ready = bool(
        settings.whisper_cpp_bin
        and settings.whisper_cpp_model
        and settings.whisper_cpp_bin.exists()
        and settings.whisper_cpp_model.exists()
    )
    if settings.transcriber == "whisper-cpp" or (settings.transcriber == "auto" and cpp_ready):
        if not (settings.whisper_cpp_bin and settings.whisper_cpp_model):
            raise TranscriptionError("whisper-cpp selected but whisper_cpp_bin/model not set")
        return WhisperCppBackend(settings.whisper_cpp_bin, settings.whisper_cpp_model)
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        return FasterWhisperBackend(settings.whisper_model, "cuda", "float16")
    return FasterWhisperBackend(settings.whisper_model, "cpu", "int8")
