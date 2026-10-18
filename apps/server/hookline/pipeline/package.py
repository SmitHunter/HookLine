"""Packaging pass: one batched Claude call over every clip's FINAL spoken words.

Selection writes packaging before the boundaries settle (re-anchoring, the opener guard, the
ending fix, front trims, cold opens and dropped lines all move them), so its hook, title and
caption can describe words that were cut. This pass rewrites them from the words the viewer
actually hears (with up to 60 s of the source before the clip as context only, so the copy
can name who/where/what the cut leaves out), re-checks the verdict, and fixes clearly
misheard caption words 1:1 (no inserts or deletes, so caption timing is untouched). A failed
call keeps the selection copy.
"""

import logging
import re
from collections import Counter

from pydantic import BaseModel, ValidationError

from hookline.config import Settings
from hookline.llm.base import LLMBackend, LLMError
from hookline.models import CaptionFix, ClipPick, Sentence, Verdict, Word
from hookline.pipeline.select import (
    _BANNED_HASHTAGS,
    _FUNCTION_WORDS,
    _QUOTED_SPAN_RE,
    _STRING,
    PACKAGING_RULES,
    SEARCH_TITLE,
    _check_caption,
    _check_title,
    _clean_hashtags,
    _flat_tokens,
    _object,
    _search_title,
    _tokens,
    _word_norm,
)
from hookline.transcript_text import ends_sentence, selection_lines

logger = logging.getLogger(__name__)

PACKAGING_SYSTEM_PROMPT = (
    """\
You are the packaging editor for vertical short-form clips (TikTok, YouTube Shorts, Reels). \
The clips are already cut. For each clip you get its final spoken words, numbered, in the \
order the viewer hears them: a cold-open teaser (if any) plays first, then the clip; \
`[cut]` marks a jump cut where lines were removed. A word marked `(?)` is one the \
transcriber was unsure of. Lines marked CONTEXT ONLY are source text from just before the \
clip (and, for a teaser, the full sentence it is cut from): the viewer never hears them, so \
use them to understand who is speaking, where and about what, never as words of the clip. \
You also get the packaging written when the clip was chosen, before its boundaries were \
adjusted: it may describe words that are no longer in the clip.

Write each clip's packaging so it is true of the source and not contradicted by the cut:
- Every concrete claim in `hook_text`, `title` and `post_caption` (a name, number, event, \
quote or outcome) must be true of the source (the final words or the context) and not \
contradicted by what the cut leaves in or out. Only quote words that are in the final words.
- When the final words alone leave a stranger unsure who is speaking or acting, where it \
happens or what it is about, `hook_text` or `title` must supply that from the context, so the \
viewer knows what they are watching. Keep the earlier packaging's specific names and places \
when they are true of the source.
- `hook_text` must not give away the payoff: the viewer should want to hear the last line. \
It must not repeat the first two spoken sentences either, quoted or paraphrased: the viewer \
is hearing them as they read it. Add what the opening does not say.
- `alt_hook_text`: a second, different `hook_text` under the same rules, used when the first \
one is rejected.
- `verdict`: re-check it on the final words. `post` only if you would publish it as it \
stands; `maybe` if the hook, ending or context is soft (an abstract policy or jargon claim \
is a `maybe` at best); `skip` if the cut no longer works.

"""
    + PACKAGING_RULES
    + """
## Caption fixes

`caption_fixes` corrects transcription errors in the burned-in captions. Each fix is a 1:1 \
replacement: `index` is a word number from the list, `original` is that word exactly as \
listed (without the `(?)` flag), and `text` replaces that one word (1–3 words; keep its \
punctuation and capitalisation). A fix whose `original` is not the word at `index` is \
discarded. You cannot insert or delete words, so the timing stays exact. Fix only words \
that are clearly misheard in context (a wrong homophone, a garbled name, a word that makes \
the sentence nonsense) or flagged `(?)` and wrong. Never restyle, censor, tidy grammar or \
paraphrase; most clips need no fixes. Read every sentence for sense: a stray word inside a \
repeated phrase, a missing little word ("we're all going the shop" is "going to the shop": \
replace "going" with "going to"), or an idiom with one word wrong is misheard. When a \
phrase is misheard across several words, fix every one of them, not just the last. For \
example (invented), in "4:for 5:all 6:intensive 7:purposes, 8:the 9:deal 10:was 11:done." \
the fix is `{"index": 6, "original": "intensive", "text": "intents and"}`, and in \
"20:it's 21:a 22:dog 23:eat 24:dot 25:world." it is \
`{"index": 24, "original": "dot", "text": "dog"}`.

Return exactly one entry per clip, with `clip` set to the clip's number as given (1, 2, \
3, …).
"""
)

PACKAGING_SCHEMA: dict = _object(
    {
        "clips": {
            "type": "array",
            "items": _object(
                {
                    "clip": {"type": "integer"},
                    "hook_text": _STRING,
                    "alt_hook_text": _STRING,
                    "title": _STRING,
                    "post_caption": _STRING,
                    "hashtags": {"type": "array", "items": _STRING},
                    "search_title": SEARCH_TITLE,
                    "verdict": {"type": "string", "enum": ["post", "maybe", "skip"]},
                    "caption_fixes": {
                        "type": "array",
                        "items": _object(
                            {"index": {"type": "integer"}, "original": _STRING, "text": _STRING}
                        ),
                    },
                }
            ),
        }
    }
)

LOW_PROB = 0.5  # words the transcriber gave less than this are flagged `(?)`
_HOOK_MAX_WORDS = 7
_FIX_MAX_TOKENS = 3
_FIX_MAX_SHARE = 0.15  # more fixes than this share of a clip's words: none are applied
_CONTEXT_S = 60.0  # source seconds before the clip shown to the packager as context
_HOOK_REPEAT_SHARE = 0.7  # a hook sharing this much of its content with the opening repeats it
_OPENING_SENTENCES = 2
_OPENING_MAX_WORDS = 25  # ... or this many words, whichever is shorter (unpunctuated speech)
_GROUND_MIN_LETTERS = 5  # a source word grounding a carried-over tag has at least 5 letters
_GROUND_MIN_SHARE = 0.5  # ... and covers at least half the tag
# Broad tags that say nothing about the clip: never carried over, and ranked last.
_GENERIC_HASHTAGS = {"history", "life", "news", "politics", "podcast", "podcasts", "interview"}
_GENERIC_HASHTAGS |= {"story", "stories", "storytime", "motivation", "motivational", "funny"}
_GENERIC_HASHTAGS |= {"inspiration", "inspirational", "comedy", "love", "education", "facts"}
_GENERIC_HASHTAGS |= {"culture", "society", "truth", "media", "america", "usa", "community"}
_GENERIC_HASHTAGS |= {"video", "clips", "clip", "mindset", "success", "lifestyle"}
_MAX_HASHTAGS = 5
_TRAILING_PUNCT_RE = re.compile(r"[.,!?;:…]+$")
_LEADING_PUNCT_RE = re.compile(r'^["“‘«(\[¿¡]+')  # opening quotes; never an apostrophe


class _RawFix(BaseModel):
    index: int
    original: str = ""
    text: str


class _RawPackage(BaseModel):
    clip: int
    hook_text: str
    alt_hook_text: str = ""
    title: str
    post_caption: str
    hashtags: list[str]
    search_title: str
    verdict: Verdict
    caption_fixes: list[_RawFix] = []


class _RawPackaging(BaseModel):
    clips: list[_RawPackage]


def _in(w: Word, seg: tuple[float, float]) -> bool:
    return seg[0] - 1e-6 <= w.start < seg[1]


def clip_words(pick: ClipPick, words: list[Word]) -> list[Word]:
    """The words the viewer hears, in order: each of the pick's segments (the cold-open
    teaser first, then the main span without its dropped lines)."""
    return [w for seg in pick.segments() for w in words if _in(w, seg)]


def _segment_texts(pick: ClipPick, words: list[Word]) -> list[tuple[str, list[str]]]:
    """(label, numbered words) per segment, numbering continuing across segments."""
    out: list[tuple[str, list[str]]] = []
    i = 0
    for k, seg in enumerate(pick.segments()):
        label = "Teaser (plays first)" if pick.cold_open and k == 0 else "Clip"
        numbered = []
        for w in (w for w in words if _in(w, seg)):
            flag = "(?)" if w.prob < LOW_PROB else ""
            numbered.append(f"{i}:{w.text}{flag}")
            i += 1
        if out and label == out[-1][0] == "Clip":
            out[-1][1].extend(["[cut]", *numbered])
        else:
            out.append((label, numbered))
    return out


def _text(words: list[Word]) -> str:
    return " ".join(w.text.strip() for w in words)


def context_words(pick: ClipPick, words: list[Word], sentences: list[Sentence]) -> list[Word]:
    """The source words before the clip's main span: from the first sentence starting within
    60 s before it up to its start (so a mid-line start shows the line's earlier words)."""
    t0 = pick.start
    first = next((s for s in sentences if t0 - _CONTEXT_S <= s.start < t0 - 1e-6), None)
    if first is None:
        return []
    return [w for w in words if first.start - 1e-6 <= w.start and w.start < t0 - 1e-6]


def _teaser_source(pick: ClipPick, words: list[Word], sentences: list[Sentence]) -> list[Word]:
    """The whole sentence(s) the cold-open teaser is cut from; empty without a cold open."""
    if pick.cold_open is None:
        return []
    a, b = pick.cold_open
    spans = [(s.start, s.end) for s in sentences if s.start < b and s.end > a]
    return [w for w in words if any(lo - 1e-6 <= w.start <= hi for lo, hi in spans)]


def build_packaging_prompt(
    picks: list[ClipPick], words: list[Word], sentences: list[Sentence] | None = None
) -> str:
    if sentences is None:
        sentences = selection_lines(words)
    parts = [f"{len(picks)} clips. Rewrite each clip's packaging from its final words."]
    for n, p in enumerate(picks, 1):
        lines = [
            f"## Clip {n}",
            f"Length: {p.duration:.1f} s. Hook type: {p.hook_type or 'unknown'}.",
            "Packaging written at selection (may be untrue now):",
            f"- hook_text: {p.hook_text}",
            f"- title: {p.title}",
            f"- post_caption: {p.post_caption}",
            f"- hashtags: {', '.join(p.hashtags)}",
            f"- search_title: {p.search_title}",
            f"- verdict: {p.verdict}",
        ]
        if before := context_words(p, words, sentences):
            lines.append(
                "Context before the clip (CONTEXT ONLY: not in the clip, the viewer never hears "
                f"it): {_text(before)}"
            )
        if source := _teaser_source(p, words, sentences):
            lines.append(
                "Teaser source (CONTEXT ONLY: the full sentence the teaser is cut from): "
                f"{_text(source)}"
            )
        lines.append("Final words:")
        lines += [f"{label}: {' '.join(ws)}" for label, ws in _segment_texts(p, words)]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _with_punct(original: str, text: str) -> str:
    """`text` with the original word's leading (opening quote) and trailing punctuation when
    the fix dropped it."""
    m = _TRAILING_PUNCT_RE.search(original)
    if m and not _TRAILING_PUNCT_RE.search(text):
        text += m.group(0)
    m = _LEADING_PUNCT_RE.search(original)
    if m and not _LEADING_PUNCT_RE.search(text):
        text = m.group(0) + text
    return text


def _caption_fixes(raw: list[_RawFix], final: list[Word], label: str) -> list[CaptionFix]:
    """Valid fixes keyed to source words: index in range, `original` (normalised) the word at
    that index, 1-3 tokens, an actual change. When they would change more than 15% of the
    clip's words, none are applied (logged)."""
    good: dict[tuple[float, int], CaptionFix] = {}
    for f in raw:
        text = " ".join(f.text.split())
        if not 0 <= f.index < len(final) or not text or len(text.split()) > _FIX_MAX_TOKENS:
            logger.warning("clip %r: caption fix %r ignored (bad index or length)", label, f)
            continue
        w = final[f.index]
        if _word_norm(f.original) != _word_norm(w.text):
            logger.warning(
                "clip %r: caption fix %r ignored: word %d is %r, not %r",
                label,
                f.text,
                f.index,
                w.text,
                f.original,
            )
            continue
        text = _with_punct(w.text, text)
        if text != w.text:
            fix = CaptionFix(start=w.start, end=w.end, original=w.text, text=text)
            good[(w.start, f.index)] = fix
    if len(good) > _FIX_MAX_SHARE * len(final):
        logger.warning(
            "clip %r: %d caption fixes would change more than %.0f%% of %d words; none applied",
            label,
            len(good),
            _FIX_MAX_SHARE * 100,
            len(final),
        )
        return []
    for fix in good.values():
        logger.info("clip %r: caption %r -> %r", label, fix.original, fix.text)
    return [good[t] for t in sorted(good)]


def _opening_tokens(final: list[Word]) -> list[str]:
    """Normalised tokens of the first two sentences the viewer hears, or of the first 25
    words when that is shorter (speech without punctuation)."""
    out: list[str] = []
    ends = 0
    for w in final[:_OPENING_MAX_WORDS]:
        out += _tokens(w.text)
        if ends_sentence(w.text):
            ends += 1
            if ends >= _OPENING_SENTENCES:
                break
    return out


def _repeats_opening(hook: str, opening: list[str]) -> bool:
    """Whether `hook` repeats the opening: its words (or a quoted span of >= 2 words) appear
    there verbatim (normalised), or >= 70% of its content words (all words if it has none)
    are spoken there."""
    toks = _tokens(hook)
    if not toks or not opening:
        return False
    said = f" {' '.join(opening)} "
    if f" {' '.join(toks)} " in said:
        return True
    for m in _QUOTED_SPAN_RE.finditer(hook):
        inner = _tokens(next(g for g in m.groups() if g is not None))
        if len(inner) >= 2 and f" {' '.join(inner)} " in said:
            return True
    content = [t for t in toks if t not in _FUNCTION_WORDS] or toks
    shared = sum((Counter(content) & Counter(opening)).values())
    return shared / len(content) >= _HOOK_REPEAT_SHARE - 1e-9


def _choose_hook(pick: ClipPick, raw: _RawPackage, final: list[Word]) -> str:
    """The packaged hook_text, else its alternative, else the selection-time hook: the first
    that is 1-7 words and does not repeat the first two spoken sentences (each rejection
    logged). When every one repeats, the selection-time hook is kept (logged)."""
    label = pick.title
    opening = _opening_tokens(final)
    for name, text in (("hook_text", raw.hook_text), ("alt_hook_text", raw.alt_hook_text)):
        hook = " ".join(text.split())
        if not hook:
            log = logger.warning if name == "hook_text" else logger.info
            log("clip %r: packaged %s is empty", label, name)
            continue
        if len(hook.split()) > _HOOK_MAX_WORDS:
            logger.warning(
                "clip %r: packaged %s %r is over %d words",
                label,
                name,
                hook,
                _HOOK_MAX_WORDS,
            )
        elif _repeats_opening(hook, opening):
            logger.warning(
                "clip %r: packaged %s %r repeats the first spoken sentences", label, name, hook
            )
        else:
            return hook
    if _repeats_opening(pick.hook_text, opening):
        logger.warning(
            "clip %r: the selection hook_text %r repeats the opening too; kept",
            label,
            pick.hook_text,
        )
    else:
        logger.info("clip %r: kept the selection hook_text %r", label, pick.hook_text)
    return pick.hook_text


def _grounded(tag: str, source: set[str]) -> bool:
    """Whether a source word of >= 5 letters that covers at least half the (lowercase) tag
    appears inside it ("estate" grounds "realestate"; "real" or "new" ground nothing)."""
    return any(
        len(t) >= _GROUND_MIN_LETTERS and len(t) >= _GROUND_MIN_SHARE * len(tag) and t in tag
        for t in source
    )


def _merge_hashtags(
    packaged: list[str], selected: list[str], source: set[str], label: str
) -> list[str]:
    """The packager's tags in its own order, then the selection's tags that are true of the
    source (grounded in a source word, not generic, not banned, not already there), at most 5
    (logged when any are carried over)."""
    out = list(packaged)
    seen = {t.lower() for t in out}
    carried = []
    for tag in selected:
        key = tag.lower()
        if key in seen or key in _BANNED_HASHTAGS or key in _GENERIC_HASHTAGS:
            continue
        if _grounded(key, source):
            seen.add(key)
            carried.append(tag)
    out = (out + carried)[:_MAX_HASHTAGS]
    if kept := [t for t in carried if t in out]:
        logger.info("clip %r: kept the selection hashtags %s", label, kept)
    return out


def _apply(pick: ClipPick, raw: _RawPackage, final: list[Word], context: list[Word]) -> ClipPick:
    label = pick.title
    hook = _choose_hook(pick, raw, final)
    title = _check_title(raw.title, _flat_tokens(final)[0], label) or pick.title
    source = {t for w in [*context, *final] for t in _tokens(w.text)}
    packaged = _clean_hashtags(raw.hashtags, label)
    tags = _merge_hashtags(packaged, pick.hashtags, source, label) or pick.hashtags
    caption = _check_caption(raw.post_caption, label) or pick.post_caption
    search = _search_title(raw.search_title, pick.search_title, label) or pick.search_title
    return pick.model_copy(
        update={
            "hook_text": hook,
            "title": title,
            "post_caption": caption,
            "hashtags": tags,
            "search_title": search,
            "verdict": raw.verdict,
            "caption_fixes": _caption_fixes(raw.caption_fixes, final, label),
        }
    )


def _ask(llm: LLMBackend, prompt: str, settings: Settings, raw_sink: list[dict] | None):
    for attempt in (1, 2):
        try:
            payload = llm.complete_json(
                PACKAGING_SYSTEM_PROMPT, prompt, PACKAGING_SCHEMA, settings.select_model
            )
            if raw_sink is not None:
                raw_sink.append(payload)
            return _RawPackaging.model_validate(payload)
        except (LLMError, ValidationError) as e:
            if attempt == 2:
                raise
            logger.warning("packaging attempt failed, retrying: %s", e)
    raise AssertionError("unreachable")


def package_clips(
    picks: list[ClipPick],
    words: list[Word],
    llm: LLMBackend,
    settings: Settings,
    raw_sink: list[dict] | None = None,
) -> list[ClipPick]:
    """Re-package every pick from its final words in one call. Returns a new list (clips the
    pass rates `skip` removed, unless that would remove them all); on any failure, logs a
    warning and returns `picks` itself, so callers can tell nothing was packaged."""
    if not picks:
        return picks
    if words and all(w.prob >= 1.0 for w in words):
        logger.info("no word confidence in this transcript; caption fixes use context only")
    finals = [clip_words(p, words) for p in picks]
    sentences = selection_lines(words)
    contexts = [
        context_words(p, words, sentences) + _teaser_source(p, words, sentences) for p in picks
    ]
    try:
        answer = _ask(llm, build_packaging_prompt(picks, words, sentences), settings, raw_sink)
    except Exception as e:  # noqa: BLE001 - packaging is polish; it must never fail the job
        logger.warning("packaging pass failed; keeping the selection-time packaging: %s", e)
        return picks
    numbers = sorted(a.clip for a in answer.clips)
    if numbers != list(range(1, len(picks) + 1)):
        logger.warning(
            "packaging pass failed: clip numbers %s are not 1-%d; keeping the selection-time "
            "packaging",
            numbers,
            len(picks),
        )
        return picks
    by_clip = {a.clip: a for a in answer.clips}
    out = [
        _apply(p, by_clip[n], f, c)
        for n, (p, f, c) in enumerate(zip(picks, finals, contexts, strict=True), 1)
    ]
    kept = [p for p in out if p.verdict != "skip"]
    for p in out:
        if p.verdict == "skip":
            logger.warning(
                "packaging re-rated clip %r skip; %s", p.title, "dropped" if kept else "kept"
            )
    if kept:
        return kept
    # never drop every clip: keep them all with their selection verdicts
    return [o.model_copy(update={"verdict": p.verdict}) for o, p in zip(out, picks, strict=True)]


def apply_caption_fixes(words: list[Word], fixes: list[CaptionFix]) -> list[Word]:
    """`words` with each fix applied to the word starting at its time whose text is still the
    fix's `original` (a stale fix changes nothing)."""
    if not fixes:
        return words
    by_key = {(round(f.start, 3), f.original): f.text for f in fixes}
    return [
        w.model_copy(update={"text": by_key[k]})
        if (k := (round(w.start, 3), w.text)) in by_key
        else w
        for w in words
    ]
