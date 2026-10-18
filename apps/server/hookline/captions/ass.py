import math
import re
from collections.abc import Sequence

from hookline.models import Word
from hookline.pipeline.layout import STACK_CAM_H, STACK_GAME_H, SceneLayout

MIN_WORD_S = 0.05
MERGE_GAP_S = 0.05
HOOK_S = 3.0
HOOK_FADE_MS = 250
HOOK_LINE_CHARS = 18
HOOK_MAX_LINES = 2
HOOK_MAX_WORDS = 7  # renderer safety net only; selection owns Q5's word limit
SEAM_TAG = "EARLIER…"  # hook-style tag marking the end of a cold open
SEAM_TAG_S = 1.2
ELLIPSIS = "…"
YELLOW = "{\\c&H00FFFF&}"
WHITE = "{\\c&HFFFFFF&}"  # back to the style colour; `{\r}` would also drop \fs/\fscx
SENTENCE_END = (".", "!", "?", "…", '."', '!"', '?"', '…"', ".”", "!”", "?”", "…”")
HOLD_GAP_S = 1.0  # a group stays up until the next one if the silence is at most this
TAIL_S = 0.3  # otherwise it ends this long after its last word
PAGE_LEAD_S = 2 / 30  # a page appears 2 frames before its first word is spoken
WORD_JOINER = "⁠"  # invisible; stops a literal "\" escaping the next character

# Anton's OS/2 metrics (2048 upm): usWinAscent 2876, usWinDescent 674, sCapHeight 1760.
# libass sizes a font so that winAscent + winDescent = Fontsize px, so Anton's cap height is
# only 1760 / 3550 = 0.496 x Fontsize (Fontsize 96 rendered 48 px caps). 136 -> 67-69 px.
FONT_SIZE = 136  # the hook title and EARLIER tag
CAP_FONT_SIZE = 152  # captions, ~12% larger: 75-76 px caps, still one line inside Q4
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
# Visible caption extent around its \pos y at CAP_FONT_SIZE, outline and shadow included
# (measured over grey: y-39 .. y+61, a Q tail or comma to y+70 -- still inside Q4 at the
# lowest line, 1180; the glyphs sit low in Anton's tall line box).
CAPTION_ABOVE = 39
CAPTION_BELOW = 61
HOOK_HALF = FONT_SIZE + HOOK_PAD  # half-height of a two-line hook box (2 x 136 / 2 + pad)
HOOK_TOP_Y = SAFE_TOP + HOOK_HALF  # 446: a two-line hook clamped to the top of Q4
TAG_HALF = FONT_SIZE // 2 + HOOK_PAD  # half-height of the one-line EARLIER tag box
TEXT_SEPARATION = 48  # visible px between a hook/tag box and the caption below it
# Highest caption line while the hook is up: visible top clears a two-line top hook box (602).
CAPTION_MIN_Y = HOOK_TOP_Y + HOOK_HALF + TEXT_SEPARATION + CAPTION_ABOVE  # 689
# Highest line once hook and tag are gone: clears the space a top one-line tag box uses (466).
CAPTION_LATE_MIN_Y = SAFE_TOP + 2 * TAG_HALF + TEXT_SEPARATION + CAPTION_ABOVE  # 553
CAPTION_MAX_Y = 1180
CAPTION_Y = CAPTION_MAX_Y
CAPTION_FACE_GAP = 40  # clearance (px) kept between a face and the caption's visible extent
CAPTION_STEP = 4  # caption / hook line candidates, px apart
FACE_ZOOM_BUDGET = 1.2  # ZOOM_PUNCH: a punch-in grows the face about its centre this much
# STACKED captions sit on the seam: the line's visible extent reaches SEAM_OVERLAP of the
# gameplay panel into it (the rest over the bottom of the cam panel, below the chin: placement
# moves a page down off the head); 757 with the 760 px cam panel.
SEAM_OVERLAP = 0.05
STACKED_CAPTION_Y = STACK_CAM_H + round(SEAM_OVERLAP * STACK_GAME_H) - CAPTION_BELOW
# ... moved down to clear a streamer's chin in the cam panel, at most this far (829)
STACKED_CAPTION_MAX_Y = SAFE_BOTTOM - 2 * HOOK_HALF - TEXT_SEPARATION - CAPTION_BELOW
# The STACKED hook title (and EARLIER tag) sits over the cam panel's lower band and the seam,
# centred on it (a two-line box 604-916), lowered to clear the chin (CAPTION_FACE_GAP) while
# it still straddles the seam; a chin too low for that (a large head) puts the box just
# below the seam (its top on it: 916), never further down over the gameplay. The captions
# shown with it go TEXT_SEPARATION under its box (<= 1159), the later ones back on the seam.
STACKED_HOOK_Y = STACK_CAM_H
STACKED_HOOK_MAX_Y = STACK_CAM_H + HOOK_HALF
STACKED_CAPTION_EARLY_Y = STACKED_HOOK_Y + HOOK_HALF + TEXT_SEPARATION + CAPTION_ABOVE
HOOK_Y = 360  # requested centre; clamped so the box stays in Q4 (1 line: 378, 2 lines: 446)
# A rendered face sample (`SceneLayout.face_px`): output s, core x0, top, x1, bottom, eye y px
FaceSample = tuple[float, float, float, float, float, float | None]
SAMPLE_HALF_S = 0.125  # a face sample stands for this long either side (4 fps)

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,{CAP_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{CAP_OUTLINE},{CAP_SHADOW},5,40,40,40,1
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


# Symbols without letters or digits that are still spoken ("rock & roll"): kept as words.
_SPOKEN_SYMBOLS = {"&", "%", "$", "+", "=", "@", "#"}
_SEAM_SLACK_S = 0.01


def _unspoken(text: str) -> bool:
    """A token with no letter or digit ("--", "—", "…", "«", "¿", "/", "♪"): punctuation a
    transcriber emitted as a word of its own, never a caption word."""
    t = text.strip()
    return t not in _SPOKEN_SYMBOLS and not any(c.isalnum() for c in t)


def _capitalised(text: str) -> str:
    """`text` with its first letter upper-cased, unless the word is mixed case ("iPhone") or
    starts with a digit ("3rd")."""
    i = next((k for k, c in enumerate(text) if c.isalnum()), None)
    if i is None or not text[i].isalpha() or any(c.isupper() for c in text[i + 1 :]):
        return text
    return text[:i] + text[i].upper() + text[i + 1 :]


def caption_words(words: list[Word], seam: float | None = None) -> list[Word]:
    """The words as captioned: punctuation-only tokens dropped (the other words keep their
    times), and the clip's first word and the first word at or after the cold-open `seam`
    capitalised (a clip may start mid-sentence).

    A stutter split by a dropped token ("I -- I") stays two words: when the pair is close
    enough for `_normalise` to merge it as one cut fragment, the second word starts just
    past that gap (it moves by at most MERGE_GAP_S)."""
    out: list[Word] = []
    removed = False
    for w in words:
        if _unspoken(w.text):
            removed = bool(out)
            continue
        if removed and _clean(w.text) == _clean(out[-1].text):
            start = round(out[-1].end + MERGE_GAP_S + 0.001, 3)
            if w.start - out[-1].end <= MERGE_GAP_S and start < w.end:
                w = w.model_copy(update={"start": start})
        out.append(w)
        removed = False
    firsts = {0} if out else set()
    if seam is not None:
        after = next((k for k, w in enumerate(out) if w.start >= seam - _SEAM_SLACK_S), None)
        if after is not None:
            firsts.add(after)
    return [
        w.model_copy(update={"text": _capitalised(w.text)}) if k in firsts else w
        for k, w in enumerate(out)
    ]


BREAK_SLACK_S = 0.01  # a word starting this little before a break still starts after it
BREAK_CARRY_S = 0.15  # a word running further past a break moves to the page after it


def _crosses(a: float, b: float, breaks: Sequence[float]) -> bool:
    """A break (cut or seam) between word starts `a` and `b`."""
    return any(a < x <= b + BREAK_SLACK_S for x in breaks)


def _next_break(t: float, breaks: Sequence[float]) -> float:
    return min((x for x in breaks if x > t + BREAK_SLACK_S), default=math.inf)


def _at_breaks(words: list[Word], breaks: Sequence[float]) -> list[Word]:
    """A word running more than BREAK_CARRY_S past a break it straddles starts at the break,
    so it opens the new page there (still captioned) instead of being clipped off the old
    one. A shorter overrun stays on the old page, clipped at the break."""
    out = []
    for word in words:
        nb = _next_break(word.start, breaks)
        if word.end - nb > BREAK_CARRY_S:
            word = word.model_copy(update={"start": nb})
        out.append(word)
    return out


def group_words(
    words: list[Word],
    max_words: int = 3,
    max_chars: int = 18,
    max_gap: float = 0.5,
    breaks: Sequence[float] = (),
) -> list[list[Word]]:
    """Caption pages. A page also ends at every `breaks` time (cuts and the cold-open seam):
    no page shows words from both sides of a cut."""
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
                or _crosses(cur[-1].start, word.start, breaks)
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


def _fit(
    lines: list[str], side: int, min_fs: int, max_fs: int = FONT_SIZE
) -> tuple[int, int] | None:
    """Largest (Fontsize <= max_fs, \\fscx percent) with fscx >= MIN_SCALE_X and fs >= min_fs
    that keeps every line plus `side` px each side within MAX_TEXT_W; None if none does."""
    avail = MAX_TEXT_W - 2 * side
    for fs in range(max_fs, min_fs - 1, -1):
        w = max(_line_w(line, fs) for line in lines)
        scale = 100 if w <= avail else math.floor(100 * avail / w)
        if scale >= MIN_SCALE_X:
            return fs, scale
    return None


def _size_tags(fs: int, scale: int, style_fs: int = FONT_SIZE) -> str:
    return (f"\\fs{fs}" if fs != style_fs else "") + (f"\\fscx{scale}" if scale < 100 else "")


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


def _breaks(breaks: Sequence[float], seam: float | None) -> list[float]:
    return sorted({*breaks, *([seam] if seam is not None else [])})


def _page_starts(groups: list[list[Word]], cuts: Sequence[float]) -> list[float]:
    """When each page appears: PAGE_LEAD_S before its first word, but never before 0, the
    break (cut or seam) it follows, or MIN_WORD_S after the previous page's last word starts
    (that word keeps its highlight)."""
    starts = []
    for i, group in enumerate(groups):
        first = group[0].start
        lo = groups[i - 1][-1].start + MIN_WORD_S if i else 0.0
        lo = max([lo, *(x for x in cuts if x <= first + BREAK_SLACK_S)])
        starts.append(min(max(first - PAGE_LEAD_S, lo), first))
    return starts


def _hold_ends(
    groups: list[list[Word]],
    duration: float,
    seam: float | None = None,
    breaks: Sequence[float] = (),
) -> list[float]:
    """When each group leaves the screen (see HOLD_GAP_S/TAIL_S): never after the next page
    appears (`_page_starts`), nor past the next break (a cut, or the cold-open `seam`) after
    its last word starts."""
    cuts = _breaks(breaks, seam)
    starts = _page_starts(groups, cuts)
    ends = []
    for i, group in enumerate(groups):
        last = group[-1].end
        if i + 1 < len(groups):
            nxt = starts[i + 1]
            end = nxt if nxt - last <= HOLD_GAP_S else min(last + TAIL_S, nxt)
        else:
            end = min(last + TAIL_S, max(duration, last))
        ends.append(min(end, _next_break(group[-1].start, cuts)))
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
    starts: list[float], holds: list[float], windows: tuple[float, float, float]
) -> list[bool]:
    """Per page (appearing at `starts`, see `_page_starts`; leaving at `holds`): on screen
    with the hook or the EARLIER tag, so it uses the early line."""
    hook_end, tag_start, tag_end = windows
    return [
        s < hook_end or (tag_end > tag_start and s < tag_end and h > tag_start)
        for s, h in zip(starts, holds, strict=True)
    ]


def early_until(
    words: list[Word],
    hook_text: str,
    duration: float,
    seam: float | None = None,
    breaks: Sequence[float] = (),
) -> float:
    """When the early caption line and the hook/tag position stop mattering: the end of the
    hook, of the EARLIER tag and of every caption group shown with them (a group starting
    under the hook may hold into a later scene). Pass it as `until` to caption_y_for and
    hook_y_for so they consider every scene on screen until then. `breaks` as for build_ass."""
    windows = _windows(bool(_layout_hook(hook_text)[0]), duration, seam)
    cuts = _breaks(breaks, seam)
    groups = group_words(_at_breaks(_normalise(words), cuts), breaks=cuts)
    holds = _hold_ends(groups, duration, seam, breaks)
    starts = _page_starts(groups, cuts)
    shown = [h for h, early in zip(holds, _early(starts, holds, windows), strict=True) if early]
    return max([HOOK_S, windows[2], *shown])


def build_ass(
    words: list[Word],
    hook_text: str,
    duration: float,
    caption_y: int = CAPTION_Y,
    hook_y: int = HOOK_Y,
    seam: float | None = None,
    caption_y_late: int | None = None,
    breaks: Sequence[float] = (),
    faces: Sequence[FaceSample] = (),
) -> str:
    """`seam` (output s; cold opens only) adds SEAM_TAG for SEAM_TAG_S s at `hook_y`. When the
    seam falls inside the hook title's 3 s, the tag waits for the title to finish.

    `breaks` (output s: the cut times, i.e. keep-range joins) and the seam force a caption
    page break: no caption event spans one (a page never shows across a cut).

    `caption_y_late` (default `caption_y`) is the caption line for groups that start once the
    hook and tag are off screen; it may sit higher (down to CAPTION_LATE_MIN_Y), e.g. above a
    close-up's head where the hook needed that space.

    `faces` (`face_samples`: the rendered face core per sample) move a page whose line would
    overlap a face core or eye line shown with it to the nearest clear line (`_page_y`),
    never into the hook box (nor its TEXT_SEPARATION) while that is up: below a top hook,
    above a STACKED hook (which sits under the seam captions)."""
    lines = [HEADER.rstrip("\n")]
    hook, fs, scale = _layout_hook(hook_text)
    windows = _windows(bool(hook), duration, seam)
    hook_end, tag_start, tag_end = windows
    lo_early, hi_early = CAPTION_MIN_Y, CAPTION_MAX_Y
    if hook:
        y = _clamp_hook_y(hook_y, len(hook), fs)
        box_top = y - len(hook) * fs / 2 - HOOK_PAD
        box_bottom = y + len(hook) * fs / 2 + HOOK_PAD
        if box_top > caption_y:  # STACKED: the hook sits under the seam captions
            hi_early = math.floor(box_top - TEXT_SEPARATION - CAPTION_BELOW)
        else:
            lo_early = max(lo_early, math.ceil(box_bottom + TEXT_SEPARATION + CAPTION_ABOVE))
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
    cuts = _breaks(breaks, seam)
    groups = group_words(_at_breaks(_normalise(words), cuts), breaks=cuts)
    holds = _hold_ends(groups, duration, seam, breaks)
    shown = _page_starts(groups, cuts)
    pages = zip(groups, shown, holds, _early(shown, holds, windows), strict=True)
    for group, page_start, hold_end, early in pages:
        texts = [word.text.upper() for word in group]
        # A single over-long word that cannot fit even at the minimum size keeps that size.
        fit = _fit([" ".join(texts)], CAP_SIDE, CAP_MIN_FS, CAP_FONT_SIZE)
        fs, scale = fit or (CAP_MIN_FS, MIN_SCALE_X)
        y = cy if early else cy_late
        if faces:
            width = _line_w(" ".join(texts), fs) * scale / 100 + 2 * CAP_SIDE
            lo = min(lo_early, y) if early else min(CAPTION_LATE_MIN_Y, y)
            hi = max(hi_early, y) if early else CAPTION_MAX_Y
            y = _page_y(y, lo, page_start, hold_end, faces, width, hi)
        tags = f"\\pos({TEXT_X},{y})" + _size_tags(fs, scale, CAP_FONT_SIZE)
        for i, active in enumerate(group):
            start = page_start if i == 0 else active.start  # the page leads its first word
            end = group[i + 1].start if i + 1 < len(group) else hold_end
            end = min(end, _next_break(active.start, cuts))  # a word straddling a cut
            parts = [
                f"{YELLOW}{_esc(t)}{WHITE}" if j == i else _esc(t) for j, t in enumerate(texts)
            ]
            lines.append(_dialogue(0, start, end, "Cap", f"{{{tags}}}" + " ".join(parts)))
    return "\n".join(lines) + "\n"


def face_rects(layout: SceneLayout, zoom: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Output-px (x0, y0, x1, y1) of the layout's median FACE CORE(s) at zoom step level
    `zoom`, mapped as the render zooms it (`SceneLayout.zoom_y`: the scene's base zoom and
    cap, about its anchor row; the face centre unless planned otherwise): one for FACE_TRACK,
    one per panel for SPLIT.

    The face core is the detector box with its top FACE_CORE_TOP (20%) removed: that strip is
    hairline/upper forehead, where text is acceptable short-form practice; the core (eyes,
    nose, mouth, chin) is what captions and hooks must stay off. Placement here and the Q4b
    audit both use the core, so they agree. The layout stores only the face's (y, h); tracking
    and split crops centre the face horizontally (a FACE_TRACK scene records the rendered
    face's median offset from the crop centre, `face_dx`: gaze lead and guard), and a detector
    box is about FACE_W_PER_H as wide as tall, so x is estimated from that."""
    face = _face_px(layout)
    if face is None:
        return []
    panel_h = SPLIT_PANEL_H if layout.kind == "SPLIT" else OUT_H
    top, bottom = (layout.zoom_y(y, zoom, panel_h) for y in face)
    half_w = (bottom - top) / 2 * FACE_W_PER_H
    core_top = top + FACE_CORE_TOP * (bottom - top)
    panels = (0, SPLIT_PANEL_H) if layout.kind == "SPLIT" else (0,)
    cx = OUT_W / 2
    if layout.kind == "FACE_TRACK" and layout.face_dx is not None:
        cx += layout.face_dx * OUT_W * layout.zoom_level(zoom)  # zooms scale about the centre
    return [(cx - half_w, core_top + p, cx + half_w, bottom + p) for p in panels]


def _face_px(layout: SceneLayout) -> tuple[float, float] | None:
    """Top and bottom (output px) of the layout's median face; for SPLIT, in the top panel."""
    if layout.face_box is None or layout.kind not in ("FACE_TRACK", "SPLIT"):
        return None
    scale = OUT_H if layout.kind == "FACE_TRACK" else SPLIT_PANEL_H
    y, h = layout.face_box
    return y * scale, (y + h) * scale


Faces = list[tuple[float, float, float]]  # (weight s, top px, bottom px)


def face_top(sample: Sequence[float | None]) -> float:
    """Where a face sample's core starts (px): its eye line when the landmarks gave one (text
    above the eyes sits on forehead/hair: accepted), else the box minus its top
    FACE_CORE_TOP."""
    eye = sample[5]
    return float(eye) if eye is not None else float(sample[2])  # type: ignore[arg-type]


def held_face_px(
    rows: Sequence[Sequence[float | None]], start: float, end: float
) -> list[list[float | None]]:
    """A scene's face samples with the first held back to its start and the last held on to
    its end (copies every 2 * SAMPLE_HALF_S): before the first sample and after the last the
    face is where it was nearest."""
    rows = sorted((list(r) for r in rows), key=lambda r: r[0])  # type: ignore[arg-type, return-value]
    if not rows:
        return []
    step = 2 * SAMPLE_HALF_S
    out: list[list[float | None]] = []
    t = rows[0][0] - step  # type: ignore[operator]
    while t > start - SAMPLE_HALF_S:
        out.insert(0, [max(t, start), *rows[0][1:]])
        t -= step
    out += rows
    t = rows[-1][0] + step  # type: ignore[operator]
    while t < end + SAMPLE_HALF_S:
        out.append([min(t, end), *rows[-1][1:]])
        t += step
    return out


def face_samples(layouts: Sequence[SceneLayout]) -> list[FaceSample]:
    """Every scene's rendered per-sample face cores (`face_px`, held to the scene's bounds:
    `held_face_px`), in time order; the last field the eye line, or None without landmarks."""
    return sorted(
        (float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), s[5])  # type: ignore[arg-type,misc]
        for lay in layouts
        for s in held_face_px(lay.face_px, lay.start, lay.end)
    )


def _faces(layouts: list[SceneLayout], until: float | None = None) -> Faces:
    """Face-core spans of the scenes (those starting before `until`), weighted by their time
    on screen (before `until`, when given): each rendered face sample (`face_px`: the real
    zoom and position, the eye line counted) where the scene has them, else the median face
    budgeted for a punch-in."""
    out: Faces = []
    for lay in layouts:
        if until is not None and lay.start >= until:
            continue
        if lay.face_px:
            out += [
                (2 * SAMPLE_HALF_S, face_top(s), s[4])
                for s in held_face_px(lay.face_px, lay.start, lay.end)
                if until is None or s[0] < until
            ]
            continue
        wt = max((lay.end if until is None else min(lay.end, until)) - lay.start, 1e-3)
        out += [(wt, y0, y1) for _, y0, _, y1 in face_rects(lay, FACE_ZOOM_BUDGET)]
    return out


def _page_y(
    default: int,
    lo: int,
    t0: float,
    t1: float,
    faces: Sequence[FaceSample],
    text_w: float,
    hi: int = CAPTION_MAX_Y,
) -> int:
    """The caption line for a page shown t0..t1: `default` unless a face sample shown with it
    (its core, or its eye line, under the text's columns) overlaps the caption's visible
    extent; then the line in [lo, hi] nearest `default` that clears every such sample with
    CAPTION_FACE_GAP to spare, else with none, else the least overlap."""
    x0, x1 = TEXT_X - text_w / 2, TEXT_X + text_w / 2
    near = [
        f
        for f in faces
        if t0 - SAMPLE_HALF_S <= f[0] <= t1 + SAMPLE_HALF_S and f[3] >= x0 and f[1] <= x1
    ]

    def overlap(y: int, gap: float) -> float:
        top, bottom = y - CAPTION_ABOVE - gap, y + CAPTION_BELOW + gap
        return sum(max(0.0, min(bottom, f[4]) - max(top, face_top(f))) for f in near)

    if not near or overlap(default, 0) == 0:
        return default
    lines = [*range(hi, lo, -CAPTION_STEP), lo]
    for gap in (CAPTION_FACE_GAP, 0):
        ok = [y for y in lines if overlap(y, gap) == 0]
        if ok:
            return min(ok, key=lambda y: abs(y - default))
    return min(lines, key=lambda y: (overlap(y, 0), abs(y - default)))


def _overlap(top: float, bottom: float, faces: Faces) -> float:
    """Duration-weighted px of vertical overlap between [top, bottom] and the faces."""
    return sum(wt * max(0.0, min(bottom, fb) - max(top, ft)) for wt, ft, fb in faces)


def _caption_cost(y: int, faces: Faces) -> float:
    gap = CAPTION_FACE_GAP
    return _overlap(y - CAPTION_ABOVE - gap, y + CAPTION_BELOW + gap, faces)


def _stacked_chin(layouts: list[SceneLayout], until: float | None) -> float | None:
    """The lowest chin (px) of the cam panel's head (`face_px` rows inside the panel, of
    STACKED scenes starting before `until`; full-cam beat rows sit below the panel and don't
    count); None without one."""
    chins = [
        float(r[4])  # type: ignore[arg-type]
        for lay in layouts
        if lay.kind == "STACKED" and (until is None or lay.start < until)
        for r in lay.face_px
        if (until is None or r[0] < until) and r[4] is not None and r[4] <= STACK_CAM_H + 1  # type: ignore[operator]
    ]
    return max(chins) if chins else None


def _stacked_caption_y(layouts: list[SceneLayout], until: float | None) -> int:
    """The STACKED seam caption line: STACKED_CAPTION_Y, lowered to clear (CAPTION_FACE_GAP)
    the cam panel's lowest chin (`_stacked_chin`), at most STACKED_CAPTION_MAX_Y."""
    chin = _stacked_chin(layouts, until)
    if chin is None:
        return STACKED_CAPTION_Y
    need = math.ceil(chin + CAPTION_FACE_GAP + CAPTION_ABOVE)
    return min(max(STACKED_CAPTION_Y, need), STACKED_CAPTION_MAX_Y)


def _stacked_hook_y(layouts: list[SceneLayout], until: float | None) -> int:
    """The STACKED hook centre: on the seam (STACKED_HOOK_Y), lowered so its two-line box
    clears the cam panel's chin by CAPTION_FACE_GAP while straddling the seam; at most
    STACKED_HOOK_MAX_Y (the box just below the seam: the panel edge parts it from the chin)."""
    chin = _stacked_chin(layouts, until)
    if chin is None:
        return STACKED_HOOK_Y
    need = math.ceil(chin + CAPTION_FACE_GAP + HOOK_HALF)
    return min(max(STACKED_HOOK_Y, need), STACKED_HOOK_MAX_Y)


def caption_y_for(
    layouts: list[SceneLayout], late: bool = False, until: float | None = None
) -> int:
    """The caption line. STACKED first scene: while the hook / tag is up (`late=False`) under
    the hook box on the seam (`hook_y_for`), later on the seam, lowered to clear the cam
    panel's chin (`_stacked_caption_y`). Otherwise the lowest line
    clear of every FACE_TRACK/SPLIT face, budgeting for a 1.2x punch-in (default 1180, i.e.
    below the chin when the face ends higher; else just above the head); when none is clear,
    the line with the least face overlap weighted by scene duration.

    The default (`late=False`) line is for groups shown with the hook or the EARLIER tag:
    scenes starting before `until` (`early_until`: hook, tag and those groups; the hook's 3 s
    by default), lines in [CAPTION_MIN_Y, CAPTION_MAX_Y] so a top hook box stays 48 px
    clear. `late=True` is for the rest: every scene, lines from CAPTION_LATE_MIN_Y."""
    if not layouts:
        return CAPTION_Y
    if layouts[0].kind == "STACKED":
        if late:
            return _stacked_caption_y(layouts, None)
        hook = _stacked_hook_y(layouts, until or HOOK_S)
        return hook + HOOK_HALF + TEXT_SEPARATION + CAPTION_ABOVE
    faces = _faces(layouts, None if late else (until or HOOK_S))
    if not faces:
        return CAPTION_Y
    lo = CAPTION_LATE_MIN_Y if late else CAPTION_MIN_Y
    lines = [*range(CAPTION_MAX_Y, lo, -CAPTION_STEP), lo]
    return min(lines, key=lambda y: _caption_cost(y, faces))  # ties: the lowest line


def hook_y_for(layouts: list[SceneLayout], until: float | None = None) -> int:
    """Hook title (and EARLIER tag) centre when STACKED: over the cam panel's lower band and
    the seam, clear of the chin (`_stacked_hook_y`; the captions shown with it go under it).
    Otherwise 360 (the top of Q4) unless a face core in a scene starting before `until`
    (`early_until`; the hook's 3 s by default; punch-in budgeted) reaches the hook band; then
    the centre in [HOOK_TOP_Y, caption visible top - TEXT_SEPARATION - two-line half height]
    with the least face overlap (ties: the highest)."""
    if not layouts:
        return HOOK_Y
    if layouts[0].kind == "STACKED":
        return _stacked_hook_y(layouts, until or HOOK_S)
    faces = _faces(layouts, until or HOOK_S)
    gap = CAPTION_FACE_GAP

    def cost(y: int) -> float:
        return _overlap(y - HOOK_HALF - gap, y + HOOK_HALF + gap, faces)

    hi = caption_y_for(layouts, until=until) - CAPTION_ABOVE - TEXT_SEPARATION - HOOK_HALF
    if not faces or cost(HOOK_TOP_Y) == 0 or hi <= HOOK_TOP_Y:
        return HOOK_Y
    best = min(range(HOOK_TOP_Y, hi + 1, CAPTION_STEP), key=cost)
    return HOOK_Y if best == HOOK_TOP_Y else best
