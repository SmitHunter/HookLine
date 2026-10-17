from collections.abc import Callable
from pathlib import Path

from hookline.models import Transcript, Word
from hookline.transcribe.base import words_to_transcript


class FasterWhisperBackend:
    def __init__(self, model: str, device: str, compute_type: str) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self, wav: Path, on_progress: Callable[[float], None] | None = None
    ) -> Transcript:
        segments, info = self._load().transcribe(str(wav), word_timestamps=True, vad_filter=True)
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(start=w.start, end=w.end, text=w.word, prob=w.probability))
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))
        return words_to_transcript(words, info.language, info.duration, timing="whisper")
