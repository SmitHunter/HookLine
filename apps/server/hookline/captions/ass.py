from hookline.models import Word
from hookline.pipeline.layout import SceneLayout

MIN_WORD_S = 0.05
MERGE_GAP_S = 0.05
HOOK_S = 3.0
HOOK_LINE_CHARS = 16
HOOK_MAX_LINES = 3
YELLOW = "{\\c&H00FFFF&}"
RESET = "{\\r}"
SENTENCE_END = (".", "!", "?", "…")

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,8,0,5,40,40,40,1
Style: Hook,Anton,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,16,0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _normalise(words: list[Word]) -> list[Word]:
    """Drop slivers and merge adjacent same-text fragments (from remap_words cuts)."""
    out: list[Word] = []
    for word in words:
        if word.end - word.start < MIN_WORD_S - 1e-6:  # tolerate 3-dp rounding
            continue
        if out and out[-1].text == word.text and word.start - out[-1].end <= MERGE_GAP_S:
            out[-1] = out[-1].model_copy(update={"end": word.end})
        else:
            out.append(word)
    return out


def group_words(
    words: list[Word], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.5
) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for word in words:
        if cur:
            chars = len(" ".join(x.text for x in cur)) + 1 + len(word.text)
            if (
                cur[-1].text.endswith(SENTENCE_END)
                or word.start - cur[-1].end > max_gap
                or len(cur) >= max_words
                or chars > max_chars
            ):
                groups.append(cur)
                cur = []
        cur.append(word)
    if cur:
        groups.append(cur)
    return groups


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ts(t: float) -> str:
    cs = max(0, round(t * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _dialogue(layer: int, start: float, end: float, style: str, text: str) -> str:
    cs_start = round(start * 100)
    cs_end = max(round(end * 100), cs_start + 1)
    return f"Dialogue: {layer},{_ts(cs_start / 100)},{_ts(cs_end / 100)},{style},,0,0,0,,{text}"


def _wrap_hook(text: str) -> str:
    lines: list[str] = []
    for word in text.upper().split():
        if lines and len(lines[-1]) + 1 + len(word) <= HOOK_LINE_CHARS:
            lines[-1] += " " + word
        elif len(lines) < HOOK_MAX_LINES:
            lines.append(word)
        else:
            lines[-1] += " " + word
    return "\\N".join(_esc(line) for line in lines)


def build_ass(words: list[Word], hook_text: str, duration: float, caption_y: int = 1300) -> str:
    lines = [HEADER.rstrip("\n")]
    hook = _wrap_hook(hook_text)
    if hook:
        lines.append(_dialogue(1, 0.0, min(HOOK_S, duration), "Hook", f"{{\\pos(540,260)}}{hook}"))
    pos = f"{{\\pos(540,{caption_y})}}"
    for group in group_words(_normalise(words)):
        for i, active in enumerate(group):
            end = group[i + 1].start if i + 1 < len(group) else active.end
            parts = []
            for j, word in enumerate(group):
                txt = _esc(word.text.upper())
                parts.append(f"{YELLOW}{txt}{RESET}" if j == i else txt)
            lines.append(_dialogue(0, active.start, end, "Cap", pos + " ".join(parts)))
    return "\n".join(lines) + "\n"


def caption_y_for(layouts: list[SceneLayout]) -> int:
    """Caption height: 640 (the seam) when the first scene is stacked, else 1300."""
    return 640 if layouts and layouts[0].kind == "STACKED" else 1300
