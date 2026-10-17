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
SEAM_TAG = "EARLIER…"  # hook-style tag marking the end of a cold open
SEAM_TAG_S = 0.8
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

OUT_W, OUT_H = 1080, 1920
SPLIT_PANEL_H = OUT_H // 2
FACE_W_PER_H = 0.8  # face detector boxes are about this wide relative to their height
FACE_CORE_TOP = 0.2  # top share of a face box (hairline/forehead) outside the face core
# Q4 safe zone for everything visible (text, outline, hook box): x in [60, 888], y in [290, 1250].
TEXT_X = 474  # centre of [60, 888]
MAX_TEXT_W = 828
SAFE_TOP = 290
SAFE_BOTTOM = 1250
# Visible caption extent around its \pos y at Fontsize 136, outline and shadow included
# (measured over grey: y-36 .. y+56; the glyphs sit low in Anton's tall line box).
CAPTION_ABOVE = 36
CAPTION_BELOW = 56
HOOK_HALF = FONT_SIZE + HOOK_PAD  # half-height of a two-line hook box (2 x 136 / 2 + pad)
HOOK_TOP_Y = SAFE_TOP + HOOK_HALF  # 446: a two-line hook clamped to the top of Q4
TAG_HALF = FONT_SIZE // 2 + HOOK_PAD  # half-height of the one-line EARLIER tag box
TEXT_SEPARATION = 24  # visible px between a hook/tag box and the caption below it
# Highest caption line while the hook is up: visible top clears a two-line top hook box (602).
CAPTION_MIN_Y = HOOK_TOP_Y + HOOK_HALF + TEXT_SEPARATION + CAPTION_ABOVE  # 662
# Highest line once hook and tag are gone: clears the space a top one-line tag box uses (466).
CAPTION_LATE_MIN_Y = SAFE_TOP + 2 * TAG_HALF + TEXT_SEPARATION + CAPTION_ABOVE  # 526
CAPTION_MAX_Y = 1180
CAPTION_Y = CAPTION_MAX_Y
CAPTION_FACE_GAP = 40  # clearance (px) kept between a face and the caption's visible extent
CAPTION_STEP = 4  # caption / hook line candidates, px apart
FACE_ZOOM_BUDGET = 1.2  # ZOOM_PUNCH: a punch-in grows the face about its centre this much
STACKED_CAPTION_Y = 700  # just below the cam/game seam (640), inside the safe band
HOOK_Y = 360  # requested centre; clamped so the box stays in Q4 (1 line: 378, 2 lines: 446)
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


def _hold_ends(groups: list[list[Word]], duration: float, seam: float | None = None) -> list[float]:
    """When each group leaves the screen (see HOLD_GAP_S/TAIL_S). A group spoken before a
    cold-open `seam` never holds past it."""
    ends = []
    for i, group in enumerate(groups):
        last = group[-1].end
        if i + 1 < len(groups):
            nxt = groups[i + 1][0].start
            end = nxt if nxt - last <= HOLD_GAP_S else min(last + TAIL_S, nxt)
        else:
            end = min(last + TAIL_S, max(duration, last))
        if seam is not None and last <= seam + 1e-6:
            end = min(end, max(seam, last))
        ends.append(end)
    return ends


def _windows(has_hook: bool, duration: float, seam: float | None) -> tuple[float, float, float]:
    """(hook end, EARLIER tag start, tag end); both tag times equal the hook end when there
    is no tag (no seam, or a seam at or past the end)."""
    hook_end = min(HOOK_S, duration) if has_hook else 0.0
    if seam is not None and max(seam, hook_end) < duration:
        start = max(seam, hook_end)
        return hook_end, start, min(start + SEAM_TAG_S, duration)
    return hook_end, hook_end, hook_end


def _early(
    groups: list[list[Word]], holds: list[float], windows: tuple[float, float, float]
) -> list[bool]:
    """Per group: on screen with the hook or the EARLIER tag, so it uses the early line."""
    hook_end, tag_start, tag_end = windows
    return [
        g[0].start < hook_end or (tag_end > tag_start and g[0].start < tag_end and h > tag_start)
        for g, h in zip(groups, holds, strict=True)
    ]


def early_until(
    words: list[Word], hook_text: str, duration: float, seam: float | None = None
) -> float:
    """When the early caption line and the hook/tag position stop mattering: the end of the
    hook, of the EARLIER tag and of every caption group shown with them (a group starting
    under the hook may hold into a later scene). Pass it as `until` to caption_y_for and
    hook_y_for so they consider every scene on screen until then."""
    windows = _windows(bool(_layout_hook(hook_text)[0]), duration, seam)
    groups = group_words(_normalise(words))
    holds = _hold_ends(groups, duration, seam)
    shown = [h for h, early in zip(holds, _early(groups, holds, windows), strict=True) if early]
    return max([HOOK_S, windows[2], *shown])


def build_ass(
    words: list[Word],
    hook_text: str,
    duration: float,
    caption_y: int = CAPTION_Y,
    hook_y: int = HOOK_Y,
    seam: float | None = None,
    caption_y_late: int | None = None,
) -> str:
    """`seam` (output s; cold opens only) adds SEAM_TAG for SEAM_TAG_S s at `hook_y`. When the
    seam falls inside the hook title's 3 s, the tag waits for the title to finish; caption
    groups spoken before the seam never hold past it.

    `caption_y_late` (default `caption_y`) is the caption line for groups that start once the
    hook and tag are off screen; it may sit higher (down to CAPTION_LATE_MIN_Y), e.g. above a
    close-up's head where the hook needed that space."""
    lines = [HEADER.rstrip("\n")]
    hook, fs, scale = _layout_hook(hook_text)
    windows = _windows(bool(hook), duration, seam)
    hook_end, tag_start, tag_end = windows
    if hook:
        y = _clamp_hook_y(hook_y, len(hook), fs)
        tags = f"\\pos({TEXT_X},{y})\\fad(0,{HOOK_FADE_MS})" + _size_tags(fs, scale)
        body = "\\N".join(_esc(line) for line in hook)
        lines.append(_dialogue(1, 0.0, hook_end, "Hook", f"{{{tags}}}{body}"))
    if tag_end > tag_start:
        tag = f"{{\\pos({TEXT_X},{_clamp_hook_y(hook_y, 1, FONT_SIZE)})}}{SEAM_TAG}"
        lines.append(_dialogue(1, tag_start, tag_end, "Hook", tag))
    cy = min(max(caption_y, CAPTION_MIN_Y), CAPTION_MAX_Y)
    cy_late = cy
    if caption_y_late is not None:
        cy_late = min(max(caption_y_late, CAPTION_LATE_MIN_Y), CAPTION_MAX_Y)
    groups = group_words(_normalise(words))
    holds = _hold_ends(groups, duration, seam)
    for group, hold_end, early in zip(groups, holds, _early(groups, holds, windows), strict=True):
        texts = [word.text.upper() for word in group]
        # A single over-long word that cannot fit even at the minimum size keeps that size.
        fs, scale = _fit([" ".join(texts)], CAP_SIDE, CAP_MIN_FS) or (CAP_MIN_FS, MIN_SCALE_X)
        tags = f"\\pos({TEXT_X},{cy if early else cy_late})" + _size_tags(fs, scale)
        for i, active in enumerate(group):
            end = group[i + 1].start if i + 1 < len(group) else hold_end
            parts = [
                f"{YELLOW}{_esc(t)}{WHITE}" if j == i else _esc(t) for j, t in enumerate(texts)
            ]
            lines.append(_dialogue(0, active.start, end, "Cap", f"{{{tags}}}" + " ".join(parts)))
    return "\n".join(lines) + "\n"


def face_rects(layout: SceneLayout, zoom: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Output-px (x0, y0, x1, y1) of the layout's median FACE CORE(s), scaled by `zoom` about
    the face centre: one for FACE_TRACK, one per panel for SPLIT.

    The face core is the detector box with its top FACE_CORE_TOP (20%) removed: that strip is
    hairline/upper forehead, where text is acceptable short-form practice; the core (eyes,
    nose, mouth, chin) is what captions and hooks must stay off. Placement here and the Q4b
    audit both use the core, so they agree. The layout stores only the face's (y, h); tracking
    and split crops centre the face horizontally, and a detector box is about FACE_W_PER_H as
    wide as tall, so x is estimated from that."""
    face = _face_px(layout)
    if face is None:
        return []
    top, bottom = face
    cy, half_h = (top + bottom) / 2, (bottom - top) / 2 * zoom
    half_w = half_h * FACE_W_PER_H
    core_top = cy - half_h + FACE_CORE_TOP * 2 * half_h
    panels = (0, SPLIT_PANEL_H) if layout.kind == "SPLIT" else (0,)
    return [(OUT_W / 2 - half_w, core_top + p, OUT_W / 2 + half_w, cy + half_h + p) for p in panels]


def _face_px(layout: SceneLayout) -> tuple[float, float] | None:
    """Top and bottom (output px) of the layout's median face; for SPLIT, in the top panel."""
    if layout.face_box is None or layout.kind not in ("FACE_TRACK", "SPLIT"):
        return None
    scale = OUT_H if layout.kind == "FACE_TRACK" else SPLIT_PANEL_H
    y, h = layout.face_box
    return y * scale, (y + h) * scale


Faces = list[tuple[float, float, float]]  # (weight s, top px, bottom px)


def _faces(layouts: list[SceneLayout], until: float | None = None) -> Faces:
    """Zoom-budgeted face-core spans of the scenes (those starting before `until`), weighted
    by their time on screen (before `until`, when given)."""
    return [
        (max((lay.end if until is None else min(lay.end, until)) - lay.start, 1e-3), y0, y1)
        for lay in layouts
        if until is None or lay.start < until
        for _, y0, _, y1 in face_rects(lay, FACE_ZOOM_BUDGET)
    ]


def _overlap(top: float, bottom: float, faces: Faces) -> float:
    """Duration-weighted px of vertical overlap between [top, bottom] and the faces."""
    return sum(wt * max(0.0, min(bottom, fb) - max(top, ft)) for wt, ft, fb in faces)


def _caption_cost(y: int, faces: Faces) -> float:
    gap = CAPTION_FACE_GAP
    return _overlap(y - CAPTION_ABOVE - gap, y + CAPTION_BELOW + gap, faces)


def caption_y_for(
    layouts: list[SceneLayout], late: bool = False, until: float | None = None
) -> int:
    """The caption line. STACKED first scene: just below the seam. Otherwise the lowest line
    clear of every FACE_TRACK/SPLIT face, budgeting for a 1.2x punch-in (default 1180, i.e.
    below the chin when the face ends higher; else just above the head); when none is clear,
    the line with the least face overlap weighted by scene duration.

    The default (`late=False`) line is for groups shown with the hook or the EARLIER tag:
    scenes starting before `until` (`early_until`: hook, tag and those groups; the hook's 3 s
    by default), lines in [CAPTION_MIN_Y, CAPTION_MAX_Y] so a top hook box stays 24 px
    clear. `late=True` is for the rest: every scene, lines from CAPTION_LATE_MIN_Y."""
    if not layouts:
        return CAPTION_Y
    if layouts[0].kind == "STACKED":
        return STACKED_CAPTION_Y
    faces = _faces(layouts, None if late else (until or HOOK_S))
    if not faces:
        return CAPTION_Y
    lo = CAPTION_LATE_MIN_Y if late else CAPTION_MIN_Y
    lines = [*range(CAPTION_MAX_Y, lo, -CAPTION_STEP), lo]
    return min(lines, key=lambda y: _caption_cost(y, faces))  # ties: the lowest line


def hook_y_for(layouts: list[SceneLayout], until: float | None = None) -> int:
    """Hook title (and EARLIER tag) centre: 940 when STACKED (below the seam captions).
    Otherwise 360 (the top of Q4) unless a face core in a scene starting before `until`
    (`early_until`; the hook's 3 s by default; punch-in budgeted) reaches the hook band; then
    the centre in [HOOK_TOP_Y, caption visible top - TEXT_SEPARATION - two-line half height]
    with the least face overlap (ties: the highest)."""
    if not layouts:
        return HOOK_Y
    if layouts[0].kind == "STACKED":
        return STACKED_HOOK_Y
    faces = _faces(layouts, until or HOOK_S)
    gap = CAPTION_FACE_GAP

    def cost(y: int) -> float:
        return _overlap(y - HOOK_HALF - gap, y + HOOK_HALF + gap, faces)

    hi = caption_y_for(layouts, until=until) - CAPTION_ABOVE - TEXT_SEPARATION - HOOK_HALF
    if not faces or cost(HOOK_TOP_Y) == 0 or hi <= HOOK_TOP_Y:
        return HOOK_Y
    best = min(range(HOOK_TOP_Y, hi + 1, CAPTION_STEP), key=cost)
    return HOOK_Y if best == HOOK_TOP_Y else best
