"""Packaging pass: one batched Claude call over every clip's FINAL spoken words.

Selection writes packaging before the boundaries settle (re-anchoring, the opener guard, the
ending fix, front trims, cold opens and dropped lines all move them), so its hook, title and
caption can describe words that were cut. This pass rewrites them from the words the viewer
actually hears, re-checks the verdict, and fixes clearly misheard caption words 1:1 (no
inserts or deletes, so caption timing is untouched). A failed call keeps the selection copy.
"""

import logging
import re

from pydantic import BaseModel, ValidationError

from hookline.config import Settings
from hookline.llm.base import LLMBackend, LLMError
from hookline.models import CaptionFix, ClipPick, Verdict, Word
from hookline.pipeline.select import (
    _STRING,
    PACKAGING_RULES,
    SEARCH_TITLE,
    _check_caption,
    _check_title,
    _clean_hashtags,
    _clip_title,
    _flat_tokens,
    _object,
)

logger = logging.getLogger(__name__)

PACKAGING_SYSTEM_PROMPT = (
    """\
You are the packaging editor for vertical short-form clips (TikTok, YouTube Shorts, Reels). \
The clips are already cut. For each clip you get its final spoken words, numbered, in the \
order the viewer hears them: a cold-open teaser (if any) plays first, then the clip; \
`[cut]` marks a jump cut where lines were removed. A word marked `(?)` is one the \
transcriber was unsure of. You also get the packaging written when the clip was chosen, \
before its boundaries were adjusted: it may describe words that are no longer in the clip.

Write each clip's packaging so it is true of the final words:
- Every concrete claim in `hook_text`, `title` and `post_caption` (a name, number, event, \
quote or outcome) must be said in the final words. If the earlier packaging mentions \
something that was cut, leave it out.
- `hook_text` must not give away the payoff: the viewer should want to hear the last line.
- `verdict`: re-check it on the final words. `post` only if you would publish it as it \
stands; `maybe` if the hook, ending or context is soft (an abstract policy or jargon claim \
is a `maybe` at best); `skip` if the cut no longer works.

"""
    + PACKAGING_RULES
    + """
## Caption fixes

`caption_fixes` corrects transcription errors in the burned-in captions. Each fix is a 1:1 \
replacement: `index` is a word number from the list and `text` replaces that one word (1–3 \
words; keep its punctuation and capitalisation). You cannot insert or delete words, so the \
timing stays exact. Fix only words that are clearly misheard in context (a wrong homophone, \
a garbled name, a word that makes the sentence nonsense) or flagged `(?)` and wrong. Never \
restyle, censor, tidy grammar or paraphrase; most clips need no fixes. Read every sentence \
for sense: a stray word inside a repeated phrase, a missing little word ("we're all going \
the shop" is "going to the shop": replace "going" with "going to"), or an idiom with one \
word wrong is misheard. When a phrase is misheard across several words, fix every one of \
them, not just the last. For example (invented), in "4:for 5:all 6:intensive 7:purposes, \
8:the 9:deal 10:was 11:done." the fix is `{"index": 6, "text": "intents and"}`, and in \
"20:it's 21:a 22:dog 23:eat 24:dot 25:world." it is `{"index": 24, "text": "dog"}`.

Return one entry per clip, with `clip` set to the clip's number.
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
                    "title": _STRING,
                    "post_caption": _STRING,
                    "hashtags": {"type": "array", "items": _STRING},
                    "search_title": SEARCH_TITLE,
                    "verdict": {"type": "string", "enum": ["post", "maybe", "skip"]},
                    "caption_fixes": {
                        "type": "array",
                        "items": _object({"index": {"type": "integer"}, "text": _STRING}),
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
_TRAILING_PUNCT_RE = re.compile(r"[.,!?;:…]+$")


class _RawFix(BaseModel):
    index: int
    text: str


class _RawPackage(BaseModel):
    clip: int
    hook_text: str
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


def build_packaging_prompt(picks: list[ClipPick], words: list[Word]) -> str:
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
            "Final words:",
        ]
        lines += [f"{label}: {' '.join(ws)}" for label, ws in _segment_texts(p, words)]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _with_punct(original: str, text: str) -> str:
    """`text` with the original word's trailing punctuation when the fix dropped it."""
    m = _TRAILING_PUNCT_RE.search(original)
    if m and not _TRAILING_PUNCT_RE.search(text):
        return text + m.group(0)
    return text


def _caption_fixes(raw: list[_RawFix], final: list[Word], label: str) -> list[CaptionFix]:
    """Valid fixes keyed to source words: index in range, 1-3 tokens, an actual change. When
    they would change more than 15% of the clip's words, none are applied (logged)."""
    good: dict[float, CaptionFix] = {}
    for f in raw:
        text = " ".join(f.text.split())
        if not 0 <= f.index < len(final) or not text or len(text.split()) > _FIX_MAX_TOKENS:
            logger.warning("clip %r: caption fix %r ignored (bad index or length)", label, f)
            continue
        w = final[f.index]
        text = _with_punct(w.text, text)
        if text != w.text:
            good[w.start] = CaptionFix(start=w.start, end=w.end, original=w.text, text=text)
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


def _apply(pick: ClipPick, raw: _RawPackage, final: list[Word]) -> ClipPick:
    label = pick.title
    hook = " ".join(raw.hook_text.split())
    if not hook or len(hook.split()) > _HOOK_MAX_WORDS:
        logger.warning(
            "clip %r: packaged hook_text %r is empty or over %d words; kept %r",
            label,
            hook,
            _HOOK_MAX_WORDS,
            pick.hook_text,
        )
        hook = pick.hook_text
    title = _check_title(raw.title, _flat_tokens(final)[0], label) or pick.title
    tags = _clean_hashtags(raw.hashtags, label) or pick.hashtags
    caption = _check_caption(raw.post_caption, label) or pick.post_caption
    return pick.model_copy(
        update={
            "hook_text": hook,
            "title": title,
            "post_caption": caption,
            "hashtags": tags,
            "search_title": _clip_title(raw.search_title) or pick.search_title,
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
    finals = [clip_words(p, words) for p in picks]
    try:
        answer = _ask(llm, build_packaging_prompt(picks, words), settings, raw_sink)
    except Exception as e:  # noqa: BLE001 - packaging is polish; it must never fail the job
        logger.warning("packaging pass failed; keeping the selection-time packaging: %s", e)
        return picks
    by_clip = {a.clip: a for a in answer.clips}
    out: list[ClipPick] = []
    for n, (pick, final) in enumerate(zip(picks, finals, strict=True), 1):
        raw = by_clip.get(n)
        if raw is None:
            logger.warning("packaging: clip %d missing from the answer; kept its packaging", n)
            out.append(pick)
        else:
            out.append(_apply(pick, raw, final))
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
