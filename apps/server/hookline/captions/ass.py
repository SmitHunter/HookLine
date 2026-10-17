import math
import re

from hookline.models import Word
from hookline.pipeline.layout import SceneLayout

MIN_WORD_S = 0.05
MERGE_GAP_S = 0.05
HOOK_S = 3.0
HOOK_FADE_MS = 250
HOOK_LINE_CHARS = 18
HOOK_MAX_LINES = 2
HOOK_MAX_WORDS = 7  # renderer safety net only; selection owns Q5's word limit
ELLIPSIS = "…"
YELLOW = "{\\c&H00FFFF&}"
WHITE = "{\\c&HFFFFFF&}"  # back to the style colour; `{\r}` would also drop \fs/\fscx
SENTENCE_END = (".", "!", "?", "…", '."', '!"', '?"', '…"', ".”", "!”", "?”", "…”")
HOLD_GAP_S = 1.0  # a group stays up until the next one if the silence is at most this
TAIL_S = 0.3  # otherwise it ends this long after its last word
WORD_JOINER = "⁠"  # invisible; stops a literal "\" escaping the next character

# Anton's OS/2 metrics (2048 upm): usWinAscent 2876, usWinDescent 674, sCapHeight 1760.
# libass sizes a font so that winAscent + winDescent = Fontsize px, so Anton's cap height is
# only 1760 / 3550 = 0.496 x Fontsize (Fontsize 96 rendered 48 px caps). 136 -> 67-69 px.
FONT_SIZE = 136
_EM = 2876 + 674
_CAP_FRAC = 1760 / _EM
HOOK_MIN_FS = math.ceil(60 / _CAP_FRAC)  # 122: hook caps stay >= 60 px (Q5)
CAP_MIN_FS = math.ceil(55 / _CAP_FRAC)  # 111: caption caps stay >= 55 px (Q2)
MIN_SCALE_X = 80  # never condense horizontally below this; step the size down instead
# Anton advance widths (font units, hmtx) for the characters captions use; px = adv / _EM * fs.
# Checked on a render: 10 x "W" at Fontsize 100 is 408 px (1458 / 3550 * 100 = 41.1 each).
_ADV = {
    **dict(zip("ABCDEFGHIJKLM", (994, 980, 971, 1010, 843, 817, 993, 1022, 464, 955, 967, 814, 1528), strict=True)),
    **dict(zip("NOPQRSTUVWXYZ", (1020, 996, 967, 1011, 976, 945, 810, 970, 961, 1458, 991, 914, 840), strict=True)),
    **dict.fromkeys("0234567890", 1012), "1": 677,
    " ": 480, ".": 468, ",": 484, "!": 469, "?": 1008, "'": 438, '"': 878, "-": 637,
    ":": 495, ";": 502, "&": 1065, "%": 2164, "$": 946, "(": 596, ")": 596, "/": 830,
    ELLIPSIS: 1450, "‘": 481, "’": 475, "“": 950, "”": 948,
}  # fmt: skip
_ADV_DEFAULT = 1100  # anything else (fallback-font glyphs): a little wider than average
CAP_OUTLINE = 10
CAP_SHADOW = 4
CAP_SIDE = CAP_OUTLINE + CAP_SHADOW  # visible extra each side of the glyphs (shadow is right)
HOOK_PAD = 20  # BorderStyle 3 box padding
HOOK_SIDE = HOOK_PAD + 2  # the box measured ~2 px wider than advance + pad

OUT_H = 1920
SPLIT_PANEL_H = OUT_H // 2
# Q4 safe zone for everything visible (text, outline, hook box): x in [60, 888], y in [290, 1250].
TEXT_X = 474  # centre of [60, 888]
MAX_TEXT_W = 828
SAFE_TOP = 290
SAFE_BOTTOM = 1250
CAPTION_MIN_Y = 700
CAPTION_MAX_Y = 1180
CAPTION_Y = CAPTION_MAX_Y
# Visible caption extent around its \pos y at Fontsize 136, outline and shadow included
# (measured over grey: y-36 .. y+56; the glyphs sit low in Anton's tall line box).
CAPTION_ABOVE = 36
CAPTION_BELOW = 56
CAPTION_FACE_GAP = 120  # caption centre this far above a face that covers the caption line
STACKED_CAPTION_Y = 700  # just below the cam/game seam (640), inside the safe band
HOOK_Y = 360  # requested centre; clamped so the box stays in Q4 (1 line: 378, 2 lines: 446)
HOOK_BAND_BOTTOM = 600  # a two-line hook box at the top reaches about this far down
HOOK_ABOVE_CAPTION = 260
STACKED_HOOK_Y = 940  # a two-line hook box clears the 700 captions (visible to 756)

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{CAP_OUTLINE},{CAP_SHADOW},5,40,40,40,1
Style: Hook,Anton,{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H26000000,&H00000000,0,0,0,0,100,100,0,0,3,{HOOK_PAD},0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalise(words: list[Word]) -> list[Word]:
    """Flatten whitespace (newlines) in word text, drop slivers and empty words, and merge
    adjacent same-text fragments (from remap_words cuts)."""
    out: list[Word] = []
    for word in words:
        text = _clean(word.text)
        if not text or word.end - word.start < MIN_WORD_S - 1e-6:  # tolerate 3-dp rounding
            continue
        if out and out[-1].text == text and word.start - out[-1].end <= MERGE_GAP_S:
            out[-1] = out[-1].model_copy(update={"end": word.end})
        else:
            out.append(word.model_copy(update={"text": text}))
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
    # libass reads `\{` and `\}` as literal braces; a bare `\` would escape whatever follows
    # (e.g. the `{` of a colour tag), so it is followed by an invisible word joiner.
    text = text.replace("\\", "\\" + WORD_JOINER)
    return text.replace("{", "\\{").replace("}", "\\}")


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


def _line_w(text: str, fs: int) -> float:
    """Estimated rendered glyph width (px) of an uppercase Anton line at Fontsize `fs`."""
    return sum(_ADV.get(c, _ADV_DEFAULT) for c in text) / _EM * fs


def _fit(lines: list[str], side: int, min_fs: int) -> tuple[int, int] | None:
    """Largest (Fontsize, \\fscx percent) with fscx >= MIN_SCALE_X and fs >= min_fs that keeps
    every line plus `side` px each side within MAX_TEXT_W; None if none does."""
    avail = MAX_TEXT_W - 2 * side
    for fs in range(FONT_SIZE, min_fs - 1, -1):
        w = max(_line_w(line, fs) for line in lines)
        scale = 100 if w <= avail else math.floor(100 * avail / w)
        if scale >= MIN_SCALE_X:
            return fs, scale
    return None


def _size_tags(fs: int, scale: int) -> str:
    return (f"\\fs{fs}" if fs != FONT_SIZE else "") + (f"\\fscx{scale}" if scale < 100 else "")


def _hook_lines(words: list[str]) -> list[str]:
    """Greedy wrap to lines of at most HOOK_LINE_CHARS; if that needs more than
    HOOK_MAX_LINES, split the words into two balanced lines instead (sized by _fit)."""
    if not words:
        return []
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= HOOK_LINE_CHARS:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) <= HOOK_MAX_LINES:
        return lines
    best = min(
        range(1, len(words)),
        key=lambda i: max(len(" ".join(words[:i])), len(" ".join(words[i:]))),
    )
    return [" ".join(words[:best]), " ".join(words[best:])]


def _layout_hook(text: str) -> tuple[list[str], int, int]:
    """Hook lines, Fontsize and \\fscx: condense to 80%, then step the size down to the
    60 px-cap minimum; if it still does not fit, keep the first 7 words plus an ellipsis; as a
    last resort condense below 80% so the box stays inside Q4."""
    words = _clean(text).upper().split()
    lines = _hook_lines(words)
    if not lines:
        return [], FONT_SIZE, 100
    fit = _fit(lines, HOOK_SIDE, HOOK_MIN_FS)
    if fit is None and len(words) > HOOK_MAX_WORDS:
        words = words[:HOOK_MAX_WORDS]
        words[-1] = words[-1].rstrip(".,;:!?…") + ELLIPSIS
        lines = _hook_lines(words)
        fit = _fit(lines, HOOK_SIDE, HOOK_MIN_FS)
    if fit is None:
        w = max(_line_w(line, HOOK_MIN_FS) for line in lines)
        fit = HOOK_MIN_FS, math.floor(100 * (MAX_TEXT_W - 2 * HOOK_SIDE) / w)
    return lines, *fit


def _clamp_hook_y(y: int, n_lines: int, fs: int) -> int:
    """Keep the hook's background box (line boxes plus pad) inside Q4's y range."""
    half = n_lines * fs / 2 + HOOK_PAD
    return min(max(y, math.ceil(SAFE_TOP + half)), math.floor(SAFE_BOTTOM - half))


def _hold_ends(groups: list[list[Word]], duration: float) -> list[float]:
    ends = []
    for i, group in enumerate(groups):
        last = group[-1].end
        if i + 1 < len(groups):
            nxt = groups[i + 1][0].start
            ends.append(nxt if nxt - last <= HOLD_GAP_S else min(last + TAIL_S, nxt))
        else:
            ends.append(min(last + TAIL_S, max(duration, last)))
    return ends


def build_ass(
    words: list[Word],
    hook_text: str,
    duration: float,
    caption_y: int = CAPTION_Y,
    hook_y: int = HOOK_Y,
) -> str:
    lines = [HEADER.rstrip("\n")]
    hook, fs, scale = _layout_hook(hook_text)
    if hook:
        y = _clamp_hook_y(hook_y, len(hook), fs)
        tags = f"\\pos({TEXT_X},{y})\\fad(0,{HOOK_FADE_MS})" + _size_tags(fs, scale)
        body = "\\N".join(_esc(line) for line in hook)
        lines.append(_dialogue(1, 0.0, min(HOOK_S, duration), "Hook", f"{{{tags}}}{body}"))
    cy = min(max(caption_y, CAPTION_MIN_Y), CAPTION_MAX_Y)
    groups = group_words(_normalise(words))
    for group, hold_end in zip(groups, _hold_ends(groups, duration), strict=True):
        texts = [word.text.upper() for word in group]
        # A single over-long word that cannot fit even at the minimum size keeps that size.
        fs, scale = _fit([" ".join(texts)], CAP_SIDE, CAP_MIN_FS) or (CAP_MIN_FS, MIN_SCALE_X)
        tags = f"\\pos({TEXT_X},{cy})" + _size_tags(fs, scale)
        for i, active in enumerate(group):
            end = group[i + 1].start if i + 1 < len(group) else hold_end
            parts = [
                f"{YELLOW}{_esc(t)}{WHITE}" if j == i else _esc(t) for j, t in enumerate(texts)
            ]
            lines.append(_dialogue(0, active.start, end, "Cap", f"{{{tags}}}" + " ".join(parts)))
    return "\n".join(lines) + "\n"


def _face_px(layout: SceneLayout) -> tuple[float, float] | None:
    """Top and bottom (output px) of the layout's median face; for SPLIT, in the top panel."""
    if layout.face_box is None or layout.kind not in ("FACE_TRACK", "SPLIT"):
        return None
    scale = OUT_H if layout.kind == "FACE_TRACK" else SPLIT_PANEL_H
    y, h = layout.face_box
    return y * scale, (y + h) * scale


def caption_y_for(layouts: list[SceneLayout]) -> int:
    """Caption line for the first scene, always inside [CAPTION_MIN_Y, CAPTION_MAX_Y]: just
    below the seam when STACKED; above a FACE_TRACK face that covers the default line (the
    band bottom) if there is room, else the default."""
    if not layouts:
        return CAPTION_Y
    first = layouts[0]
    if first.kind == "STACKED":
        return STACKED_CAPTION_Y
    face = _face_px(first) if first.kind == "FACE_TRACK" else None
    if face and face[0] < CAPTION_Y + CAPTION_BELOW and face[1] > CAPTION_Y - CAPTION_ABOVE:
        above = round(face[0]) - CAPTION_FACE_GAP
        if above >= CAPTION_MIN_Y:
            return above
    return CAPTION_Y


def hook_y_for(layouts: list[SceneLayout]) -> int:
    """Hook title centre for the first scene: 360 (top) unless that covers a face, then
    260 px above the captions; 940 when STACKED (below the seam captions)."""
    if not layouts:
        return HOOK_Y
    first = layouts[0]
    if first.kind == "STACKED":
        return STACKED_HOOK_Y
    face = _face_px(first)
    if face and face[0] < HOOK_BAND_BOTTOM:
        return caption_y_for(layouts) - HOOK_ABOVE_CAPTION
    return HOOK_Y
