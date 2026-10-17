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

OUT_H = 1920
SPLIT_PANEL_H = OUT_H // 2
CAPTION_Y = 1300
CAPTION_HALF_H = 60  # half the caption line's height
CAPTION_FACE_GAP = 120  # caption centre below a face that would cover the 1300 line
CAPTION_MAX_Y = 1600
STACKED_CAPTION_Y = 640  # the cam/game seam
HOOK_Y = 260
HOOK_BAND_BOTTOM = 440  # up to 3 hook lines around y=260 reach about this far down
HOOK_ABOVE_CAPTION = 260
STACKED_HOOK_Y = 860  # below the seam captions, over the top of the game

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


def build_ass(
    words: list[Word],
    hook_text: str,
    duration: float,
    caption_y: int = CAPTION_Y,
    hook_y: int = HOOK_Y,
) -> str:
    lines = [HEADER.rstrip("\n")]
    hook = _wrap_hook(hook_text)
    if hook:
        pos = f"{{\\pos(540,{hook_y})}}"
        lines.append(_dialogue(1, 0.0, min(HOOK_S, duration), "Hook", pos + hook))
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


def _face_px(layout: SceneLayout) -> tuple[float, float] | None:
    """Top and bottom (output px) of the layout's median face; for SPLIT, in the top panel."""
    if layout.face_box is None or layout.kind not in ("FACE_TRACK", "SPLIT"):
        return None
    scale = OUT_H if layout.kind == "FACE_TRACK" else SPLIT_PANEL_H
    y, h = layout.face_box
    return y * scale, (y + h) * scale


def caption_y_for(layouts: list[SceneLayout]) -> int:
    """Caption line for the first scene: 640 (the seam) when STACKED; below the face when a
    FACE_TRACK face covers the 1300 line; else 1300."""
    if not layouts:
        return CAPTION_Y
    first = layouts[0]
    if first.kind == "STACKED":
        return STACKED_CAPTION_Y
    face = _face_px(first) if first.kind == "FACE_TRACK" else None
    if face and face[0] < CAPTION_Y + CAPTION_HALF_H and face[1] > CAPTION_Y - CAPTION_HALF_H:
        return min(round(face[1]) + CAPTION_FACE_GAP, CAPTION_MAX_Y)
    return CAPTION_Y


def hook_y_for(layouts: list[SceneLayout]) -> int:
    """Hook title centre for the first scene: 260 (top) unless that covers a face, then
    260 px above the captions; 860 when STACKED (below the 640 seam captions)."""
    if not layouts:
        return HOOK_Y
    first = layouts[0]
    if first.kind == "STACKED":
        return STACKED_HOOK_Y
    face = _face_px(first)
    if face and face[0] < HOOK_BAND_BOTTOM:
        return caption_y_for(layouts) - HOOK_ABOVE_CAPTION
    return HOOK_Y
