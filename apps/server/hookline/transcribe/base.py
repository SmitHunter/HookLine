from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from hookline.models import Segment, Transcript, Word
from hookline.transcript_text import split_sentences


class TranscriptionError(Exception):
    """A transcription backend failed."""


class Transcriber(Protocol):
    def transcribe(
        self, wav: Path, on_progress: Callable[[float], None] | None = None
    ) -> Transcript: ...


def words_to_transcript(
    words: list[Word],
    language: str,
    duration: float,
    timing: Literal["segment", "dtw", "whisper"] = "segment",
) -> Transcript:
    """Regroup a flat word list into sentence-shaped segments (shared by all backends)."""
    words = [w.model_copy(update={"text": w.text.strip()}) for w in words if w.text.strip()]
    sentences = split_sentences(words)
    buckets: list[list[Word]] = [[] for _ in sentences]
    idx = 0
    for w in words:
        while idx + 1 < len(sentences) and w.start >= sentences[idx + 1].start:
            idx += 1
        buckets[idx].append(w)
    segments = [
        Segment(start=s.start, end=s.end, text=s.text, words=b)
        for s, b in zip(sentences, buckets, strict=True)
    ]
    return Transcript(language=language, duration=duration, segments=segments, timing=timing)
