from bisect import bisect_left

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


def compact_transcript(
    sentences: list[Sentence], markers: dict[int, list[str]] | None = None
) -> str:
    """One `L{n} [mm:ss.s] sentence` line each, where n is the sentence's index in
    `sentences` (pass the whole transcript's sentences so line numbers stay global), with
    heat markers appended as ` ⟨a b⟩` (ascending
    second order, de-duplicated). A sentence owns markers in [previous sentence's end, its
    end): markers in a pause attach to the NEXT sentence; markers before the first sentence
    go to the first, and markers after the last sentence go to the last."""
    markers = markers or {}
    keys = sorted(markers)
    lines = []
    for i, s in enumerate(sentences):
        lo = sentences[i - 1].end if i else float("-inf")
        hi = s.end if i + 1 < len(sentences) else float("inf")
        found: list[str] = []
        for sec in keys[bisect_left(keys, lo) : bisect_left(keys, hi)]:
            found.extend(m for m in markers[sec] if m not in found)
        suffix = f" ⟨{' '.join(found)}⟩" if found else ""
        lines.append(f"L{i} [{format_timestamp(s.start)}] {s.text}{suffix}")
    return "\n".join(lines)
