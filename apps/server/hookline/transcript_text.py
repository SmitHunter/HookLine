import re
from bisect import bisect_left

from hookline.models import Sentence, Word

_TRAILING = "\"'”’)]}»"


def ends_sentence(text: str) -> bool:
    """True when `text` ends in terminal punctuation (.!?), ignoring closing quotes/brackets."""
    return text.rstrip(_TRAILING).endswith((".", "?", "!"))


def ends_statement(text: str) -> bool:
    """`ends_sentence`, except a trailing ellipsis ("...", "…"): the speaker trails off and the
    thought continues."""
    text = text.rstrip(_TRAILING)
    return ends_sentence(text) and not text.endswith(("...", "…"))


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
            or ends_sentence(word.text)
            or nxt.start - word.end > max_gap
            or nxt.end - buf[0].start > max_len_s
        ):
            sentences.append(_make(buf))
            buf = []
    return sentences


_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e."}
_DOTTED_RE = re.compile(r"^(?:[a-z]\.){2,}$", re.IGNORECASE)  # U.S., U.K.
_ACRONYM_PAUSE_S = 0.3
_SOFT_MIN_S = 10.0  # a soft split leaves a line of at least this long ...
_SOFT_DECIDE_S = 20.0  # ... and is decided once the run would pass this length
_SOFT_GAP_S = 1.0  # ... at a word gap of at least this long


def _ends_line(word: Word, nxt: Word | None) -> bool:
    """Terminal punctuation for selection lines: not an ellipsis, not an abbreviation; a
    dotted acronym (U.S.) ends a sentence only before a capitalised word after a 0.3 s pause."""
    text = word.text.strip().rstrip(_TRAILING)
    if not ends_statement(text):
        return False
    if text.lower().lstrip("\"'“‘(") in _ABBREVIATIONS:
        return False
    if _DOTTED_RE.match(text.lstrip("\"'“‘(")):
        return (
            nxt is not None
            and nxt.text.strip().lstrip("\"'“‘(")[:1].isupper()
            and nxt.start - word.end >= _ACRONYM_PAUSE_S
        )
    return True


def _longest_gap_cut(buf: list[Word], nxt: Word, min_left_s: float, min_gap_s: float) -> int:
    """Cut index (words before it form a line) at the longest gap (latest on a tie) whose
    left part lasts >= `min_left_s` and which is >= `min_gap_s`; 0 when there is none."""
    best, best_gap = 0, min_gap_s
    for k, (a, b) in enumerate(zip(buf, [*buf[1:], nxt], strict=True)):
        gap = b.start - a.end
        if a.end - buf[0].start >= min_left_s and gap >= best_gap:
            best, best_gap = k + 1, gap
    return best


def selection_lines(words: list[Word], max_len_s: float = 25.0) -> list[Sentence]:
    """Sentences for clip selection. Split on terminal punctuation (.!? before any closing
    quotes or brackets; not "..." / "…", abbreviations such as "Mr." or "e.g.", or a dotted
    acronym unless a capitalised word follows a 0.3 s pause), never on an ordinary pause.

    Lightly punctuated speech (Twitch): a run that would pass 20 s without punctuation is
    split at its longest gap >= 1.0 s that leaves a line of >= 10 s, so such lines run
    ~10-20 s. Hard cap: a run that would exceed `max_len_s` is split at its longest word gap
    (the latest one on a tie)."""
    lines: list[Sentence] = []
    buf: list[Word] = []
    for i, word in enumerate(words):
        buf.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or _ends_line(word, nxt):
            lines.append(_make(buf))
            buf = []
            continue
        if nxt.end - buf[0].start > _SOFT_DECIDE_S:
            cut = _longest_gap_cut(buf, nxt, _SOFT_MIN_S, _SOFT_GAP_S)
            if cut:
                lines.append(_make(buf[:cut]))
                buf = buf[cut:]
        while buf and nxt.end - buf[0].start > max_len_s:
            cut = _longest_gap_cut(buf, nxt, 0.0, float("-inf"))
            lines.append(_make(buf[:cut]))
            buf = buf[cut:]
    return lines


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
