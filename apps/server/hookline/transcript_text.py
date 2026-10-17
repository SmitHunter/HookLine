from hookline.models import Sentence, Word

_TRAILING = "\"'”’)]}»"


def _ends_sentence(text: str) -> bool:
    return text.rstrip(_TRAILING).endswith((".", "?", "!"))


def _make(buf: list[Word]) -> Sentence:
    return Sentence(
        start=buf[0].start,
        end=buf[-1].end,
        text=" ".join(w.text.strip() for w in buf),
    )


def split_sentences(
    words: list[Word], max_gap: float = 0.7, max_len_s: float = 25.0
) -> list[Sentence]:
    sentences: list[Sentence] = []
    buf: list[Word] = []
    for i, word in enumerate(words):
        buf.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (
            nxt is None
            or _ends_sentence(word.text)
            or nxt.start - word.end > max_gap
            or nxt.end - buf[0].start > max_len_s
        ):
            sentences.append(_make(buf))
            buf = []
    return sentences


def format_timestamp(t: float) -> str:
    tenths = int(t * 10)
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    s = rem / 10
    if h:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


def compact_transcript(sentences: list[Sentence]) -> str:
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in sentences)
