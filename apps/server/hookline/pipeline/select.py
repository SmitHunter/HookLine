"""Clip selection: Claude reads the whole transcript and picks the best short-form moments.

Claude answers in transcript line numbers (`L{n}` = sentence n of the whole transcript), never
in seconds; this module maps lines back to sentence times, re-anchors a start whose declared
`first_words` sit on a neighbouring line, and drops candidates Claude itself rated `skip`.
"""

import itertools
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import get_args

from pydantic import BaseModel, ValidationError

from hookline.config import Settings
from hookline.llm.base import LLMBackend, LLMError
from hookline.models import (
    ClipPick,
    CommentRisk,
    HookType,
    RubricScores,
    Sentence,
    Transcript,
    Verdict,
    Word,
)
from hookline.pipeline.boundaries import FILLER_OPENERS
from hookline.pipeline.signals import CLIP_MARKER
from hookline.transcript_text import (
    compact_transcript,
    ends_sentence,
    format_timestamp,
    selection_lines,
)

logger = logging.getLogger(__name__)


class NoClipsSelectedError(Exception):
    """Selection produced zero usable clips; nothing is cached so a re-run re-selects."""


ALL_SKIP_MESSAGE = "Claude rated every candidate as skip — the source may not have strong moments"

SYSTEM_PROMPT = """\
You are an elite short-form video editor. You cut long recordings into vertical clips for \
TikTok, YouTube Shorts and Instagram Reels. Think like a feed editor, not a documentary editor: \
every clip is shown to a stranger mid-scroll who has never seen this creator and has no idea \
what happened earlier, so the first sentence has to work for that stranger on its own, and the \
clip has to make them feel something by the end.

## What you are given

A transcript of one long video. Most inputs are multi-hour live streams (Twitch VODs: gameplay \
with a facecam, or Just Chatting); some are podcasts, interviews or films. Each line is one \
sentence, written as `L{n} [timestamp] sentence`. `L{n}` is the line number: you say where a \
clip starts and ends by line number, so you never convert timestamps yourself. The timestamp \
(`mm:ss.s`, or `h:mm:ss.s` past the first hour) is when the sentence starts, measured from the \
start of the video; use it to judge pace, pauses and length. You only have the words, so read \
energy from them: exclamations, laughter, sudden reversals, arguments, reveals, fails, wins, \
confessions and strong opinions.

Some lines end with markers in angle brackets, such as `⟨🔥chat×4 🔊loud⟩`. They are audience \
and audio reactions inside that sentence: `🔥chat×N` means chat activity ran about N times the \
usual rate, and `🔊loud` means the audio got far louder than normal. An emoji after \
`🔥chat×N` says what chat was reacting with (😂 laughing, 🏆 hype, 💀 a fail, 😱 shock), and \
`<<CLIP_MARKER>>` means chat is asking for a clip. Treat them as evidence of \
emotional intensity, not a requirement: a moment without markers can still be great, and a \
marked one can still be a poor clip.

## Choosing a moment

- Keep one idea per clip, built on one hook archetype: the reason a stranger stops. Declare it \
as `hook_type`:
  - `bold_claim`: a strong, arguable statement.
  - `surprising_fact`: a number or fact the viewer did not expect.
  - `conflict`: disagreement, confrontation, someone pushing back.
  - `confession`: the speaker admits something personal, risky or embarrassing.
  - `question`: a question the viewer immediately wants answered.
  - `in_progress_action`: something is already happening (a clutch, a fail, a build) and the \
viewer joins mid-action.
  - `reaction_peak`: a big reaction (shock, rage, laughter) to something that just happened.
  - `story_peak`: the turning point of a story, told so the slow setup can be skipped.
- Stand alone. Live streams are full of running jokes, chat references, raid shout-outs and \
callbacks to earlier in the stream. Favour moments that make complete sense with zero context. \
A slightly smaller moment that anyone can follow beats a big one that needs backstory.
- Open on the hook. `first_words` is the verbatim first sentence the viewer hears, copied from \
the `start_line` line. Read it as a stranger mid-scroll would. If it needs earlier context (it \
begins with "And", "So", "But", "Because" or "Which", answers a question the viewer never \
heard, quotes a conversation the viewer never saw ("I was like…", "he goes…"), or leans on a \
pronoun from earlier), move the start or use a cold open. Setup lines \
such as "So I've been thinking about this" or "Okay, so basically" are not hooks. If a strong \
moment has a messy first line, move `start_line` forward; never mark it skip only because of \
its opening.
- Cold opens. When a clip genuinely needs its setup but its strongest line comes later, set \
`cold_open_words` to that later line, copied verbatim from the transcript: a complete thought \
that finishes its sentence, without lead-ins such as "I think" or "the thing is". It plays \
first as a teaser of at most 6 seconds, then the clip runs from `start_line`; a teaser that \
cannot finish its sentence within 6 seconds is dropped. Otherwise leave it null. It must lie \
inside the clip and must not be in the clip's first sentence.
- End on the payoff line: the punchline, reveal, outcome or reaction that rewards watching. \
Never end on "yeah", "mm-hmm", "right" or any other interviewer backchannel, on dead air, or \
on the start of the next topic. Before you fix `end_line`, read the next 20 seconds after \
it: if a more concrete or emotional line follows (the example, the number, the punchline), \
extend the clip to include it. `last_words` is the verbatim last sentence heard, copied from \
the `end_line` line. Prefer endings that echo the opening, so a replay loops naturally.
- Length: prefer 20–40 seconds for talking heads; go longer, up to the maximum stated with \
the request, only if the tension holds the whole way. Never exceed that maximum: if the \
story needs more, start later, closer to the turn. A `reaction_peak` clip may be as short \
as 12 seconds.
- Keep candidates distinct: they must not overlap each other in time, and avoid two clips on \
the same theme; spread them across categories.
- Moments that name famous people, brands or institutions are high-value for search; include \
the best one among your candidates.
- Controversy, hot takes and confrontations are assets, not risks. `safety` penalises only \
slurs, sexual content and gore.

## Live streams

- Chat-reading, sub and raid thank-yous and donation alerts are dead air. Cut around them.
- A loud reaction is the payoff, not the opening: start 3–8 seconds earlier on the cause, so \
the viewer sees what the streamer is reacting to. The cause line is then your `first_words` \
and must still work cold.
- When the streamer is answering chat, the viewer never saw the question. Put the question in \
`hook_text` so the answer lands.

## Scoring rubric (0–100)

Rubric: Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · Length fit 10 · \
Pacing 10 · Loop potential 5 · Platform safety 5.

Score each candidate with whole numbers, each within its range:
- `hook` (0–25): how hard the first sentence grabs a cold viewer.
- `context` — Standalone context (0–15): how well it works with no knowledge of the creator \
or what came before.
- `payoff` (0–15): how satisfying the ending is.
- `emotion` — Emotional intensity (0–15): hype, rage, shock, laughter, tension, awkwardness \
or heart.
- `length_fit` (0–10): how well the length suits the material; full marks for a tight \
20–40 second cut.
- `pacing` (0–10): density of speech and events; no long pauses, rambling or filler.
- `loop` — Loop potential (0–5): whether the ending flows back into the opening.
- `safety` — Platform safety (0–5): lose points only for slurs, sexual content and gore.

Calibration matters more than generosity. Give `hook` 20 or higher only if a stranger would \
stop scrolling on the first sentence alone. Most sources yield 1–4 postable clips, so most \
candidates should not score like hits; use the whole range. For example, a clip opening on \
"So I want to talk about the housing market for a second." is a setup opener and scores 12 \
for `hook`; one opening on "My landlord raised the rent 40 percent and told me to sell my \
car." puts the stake first and scores 22.

## Verdict

`verdict` is your call as the editor who has to put their name on the post:
- `post`: you would publish it as it stands.
- `maybe`: worth a look, but something is soft (the hook, the ending, the context).
- `skip`: not worth posting. Return it as `skip` rather than inflating it.
Use `skip` honestly; a source with no strong moments should get mostly skips.

## Packaging

Everything in packaging must be literally true to the clip.

- `title`: short and specific, under 60 characters.
- `hook_text`: on-screen text over the first seconds. 3–7 words that name the stake or the \
tension as a claim or a question (for example "Asked her: was the war worth it?"), never a \
topic label such as "Activists versus big money". Build it from concrete nouns or numbers in \
the clip, and keep it literally true: never exaggerate what happens (if someone was "a \
little rude", they did not "come to mock" anyone), and never promise visuals or footage the \
clip does not show; the viewer sees only the speakers. For a cold-open clip, `hook_text` must \
complement the teaser line, not contradict or repeat it. Tease the payoff without spoiling \
it, and do not just repeat the first spoken line. Never use template phrases such as "Then \
this", "Wait for it", "You won't believe" or "Watch till the end".
- `post_caption`: the caption under the post, 1–2 lines, that ends with a question inviting \
comments: a specific question with two plausible answers, so viewers pick a side. Template \
questions are banned: "Agree or disagree?", "Thoughts?", "What do you think?" and \
"Has X ever…?".
- `hashtags`: 3–5 specific tags, without the `#`, about the clip's actual topic, people or \
game. Generic reach tags such as #fyp, #foryou or #viral are banned.
- `search_title`: a keyword-first title for search, at most 70 characters, that leads with the \
searchable topic rather than a tease.
- `category`: one or two words for the content area, such as "gaming" or "relationships".
- `comment_risk`: `high` when the clip invites creepy or harassing comments about the \
speaker's appearance, `medium` when that is plausible, otherwise `low`.
- `reason`: one or two sentences on why this moment works and what the payoff is.
- `edit_beats`: a few editing cues at moments that matter, written exactly as \
`zoom: '<word>' @ <t>s` or `emoji <emoji> @ <t>s`, where `<t>` is absolute seconds from the \
start of the video and lies inside the clip (`14:44.0` is 884.0 seconds).

## A worked example (invented)

Transcript excerpt:
L212 [14:03.5] So I'd been a line cook for maybe two years at that point.
L213 [14:07.9] And one night the head chef just doesn't show up.
L214 [14:10.2] I ran a three-hundred-cover Saturday on my own when I was nineteen.
…
L226 [14:44.0] He walked in at close, looked at the tickets, and handed me his knives.
L227 [14:48.3] Yeah.

A good pick (37 seconds): `start_line` 214, `end_line` 226, `cold_open_words` null.
- `first_words`: "I ran a three-hundred-cover Saturday on my own when I was nineteen." It works \
cold; L212 and L213 are setup, so the clip starts after them.
- `last_words`: "He walked in at close, looked at the tickets, and handed me his knives." The \
payoff; the backchannel "Yeah." on L227 is left out.
- `hook_type`: `bold_claim`. `verdict`: `post`.
- `hook_text`: "Head chef walked in at close" (it teases the payoff rather than restating \
the first line)
- `post_caption`: "Nineteen, no head chef, a full Saturday rush. Was handing over his knives \
a promotion or a test?"
- `hashtags`: ["linecook", "kitchenlife", "restaurantstories", "cheflife"]
- `search_title`: "Running a 300-Cover Restaurant Service Alone at 19"
- `edit_beats`: ["zoom: 'knives' @ 887.2s"]

## Ranking

Return the number of candidates requested, ranked best first, each with its verdict. If the \
material is thin, still return them, mark the weak ones `skip` and score them honestly. Never \
pick a moment that overlaps a range you are told to avoid.
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("<<CLIP_MARKER>>", CLIP_MARKER)

_SCORE_MAX = {
    "hook": 25,
    "context": 15,
    "payoff": 15,
    "emotion": 15,
    "length_fit": 10,
    "pacing": 10,
    "loop": 5,
    "safety": 5,
}


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}

SELECTION_SCHEMA: dict = _object(
    {
        "clips": {
            "type": "array",
            "items": _object(
                {
                    "start_line": {"type": "integer"},
                    "first_words": _STRING,
                    "end_line": {"type": "integer"},
                    "last_words": _STRING,
                    "cold_open_words": {"anyOf": [_STRING, {"type": "null"}]},
                    "hook_type": {"type": "string", "enum": list(get_args(HookType))},
                    "title": _STRING,
                    "hook_text": _STRING,
                    "reason": _STRING,
                    "scores": _object({k: {"type": "integer"} for k in _SCORE_MAX}),
                    "verdict": {"type": "string", "enum": list(get_args(Verdict))},
                    "post_caption": _STRING,
                    "hashtags": {"type": "array", "items": _STRING},
                    "search_title": _STRING,
                    "category": _STRING,
                    "comment_risk": {"type": "string", "enum": list(get_args(CommentRisk))},
                    "edit_beats": {"type": "array", "items": _STRING},
                }
            ),
        }
    }
)

_COLD_OPEN_MAX_S = 6.0
_COLD_OPEN_MIN_S = 1.0  # shorter teasers extend back within their sentence, else drop
_COLD_OPEN_MIN_MATCH = 0.8
_BEAT_SNAP_S = 3.0
_REACTION_PEAK_MIN_S = 12.0
_CHUNK_OVERLAP_S = 120.0
_WORDS_MIN_OVERLAP = 0.6
_REANCHOR_RADIUS = 3
_EXTEND_MAX_LINES = 2
_EXTEND_MAX_S = 8.0
_SEARCH_TITLE_MAX = 70
_BANNED_HASHTAGS = {
    "fyp",
    "foryou",
    "foryoupage",
    "foryourpage",
    "fypage",
    "fy",
    "viral",
    "viralvideo",
    "trending",
    "explore",
    "reels",
    "shorts",
    "tiktok",
}
# Tokens of a backchannel line ("Yeah.", "Mm-hmm.", "Uh-huh, right."); hyphenated forms split.
_BACKCHANNEL = {"yeah", "yep", "mm", "hmm", "mm-hmm", "uh", "huh", "uh-huh", "right", "okay"}
_BACKCHANNEL |= {"sure"}  # "wow" / "exactly" are often the payoff, not a nod
_UNFINISHED_ENDINGS = ("...", "…", "--", "—", "-")
_TRAILING_QUOTES = "\"'”’)]}»"
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_BEAT_TIME_RE = re.compile(r"@\s*(\d+(?:\.\d+)?)\s*s?\s*$")
_BEAT_WORD_RE = re.compile(r"\s*zoom:\s*['\"‘“](.+?)['\"’”]\s*@")
_WORD_PUNCT = ",.!?…\"'-:;–—‘’“”()[]"
_QUOTE_CHARS = "\"“”‘'"
_QUOTED_SPAN_RE = re.compile(
    r'"([^"]+)"|“([^”]+)”|(?<!\w)‘([^’]+)’(?!\w)'
    r"|(?<!\w)'(\S[^\n]*?\s[^\n]*?\S)'(?!\w)"  # straight single quotes: >= 2 words
)
# Lead-ins stripped from the start of a cold-open teaser.
_LEAD_INS = (*FILLER_OPENERS, "i think", "the thing is", "i mean", "you know", "i would say")
_LEAD_INS += ("i was like", "he goes", "she goes", "they said", "he said", "she said")
# Fillers skipped before judging a first sentence's opener ("Um, so..." opens on "so").
_OPENER_FILLERS = ("um", "uh", "yeah", "okay", "ok", "well", "like", "oh", "anyway")
_OPENER_FILLERS += ("i mean", "you know")
_CONTEXT_OPENERS = {"and", "so", "but", "because", "which"}
# Matched on the lowercased text after fillers. "I like pizza" is fine; "I was like, ..." and
# "I like, ..." quote a conversation. A self-quote with content ("I said, Hi, I'm here to
# borrow your equipment.") is deliberately allowed: it works cold.
_QUOTED_OPENER_RE = re.compile(
    r"(?:i|he|she|they|we)(?:(?: was| were|'m| am| is| are) like\b| like[,\"“”'‘’])"
    r"|(?:he|she|they) (?:goes|go|went|said|says)\b"
)
_CONJ_PAUSE_S = 0.3  # a bare "So" / "And" followed by this pause is the filler trimmer's job
_GUARD_MAX_MOVE_S = 12.0
_TEMPLATE_QUESTION_RE = re.compile(
    r"agree or disagree|^thoughts\s*\?|^what do you think\s*\?|^(?:has|have)\b.*\bever\b"
)


class _RawScores(BaseModel):
    hook: int
    context: int
    payoff: int
    emotion: int
    length_fit: int
    pacing: int
    loop: int
    safety: int


class _RawClip(BaseModel):
    start_line: int
    end_line: int
    cold_open_words: str | None = None
    first_words: str = ""
    last_words: str = ""
    hook_type: HookType | None = None
    verdict: Verdict
    title: str
    hook_text: str
    reason: str
    scores: _RawScores
    post_caption: str = ""
    hashtags: list[str] = []
    search_title: str = ""
    category: str = ""
    comment_risk: CommentRisk | None = None
    edit_beats: list[str] = []


class _RawSelection(BaseModel):
    clips: list[_RawClip]


@dataclass(frozen=True)
class _Source:
    """What `_to_pick` needs to turn a line-indexed candidate into a timed ClipPick."""

    sentences: list[Sentence]
    tokens: list[list[str]]  # per-sentence normalised tokens, for first_words matching
    words: list[Word]
    duration: float
    settings: Settings
    exclude: list[tuple[float, float]]
    heat: frozenset[int] | None = None  # every second inside a loud / chat-spike run


def build_prompt(
    compact: str,
    n: int,
    duration: float,
    exclude: list[tuple[float, float]],
    excerpt: tuple[float, float] | None = None,
    min_s: float | None = None,
    max_s: float | None = None,
) -> str:
    parts = [f"Video duration: {format_timestamp(duration)} ({duration:.1f} seconds)."]
    if excerpt:
        parts.append(
            f"This is an excerpt covering {format_timestamp(excerpt[0])} to "
            f"{format_timestamp(excerpt[1])} of a longer video. Line numbers and timestamps "
            "are still those of the full video."
        )
    parts.append(
        f"Return the {n} best clip candidates, ranked best first, each with an honest verdict."
    )
    if min_s is not None and max_s is not None:
        parts.append(
            f"Each clip lasts {min_s:.0f}–{max_s:.0f} seconds. Never exceed {max_s:.0f} s; if "
            "the story needs more, start later, closer to the turn."
        )
    if exclude:
        ranges = "\n".join(f"- {format_timestamp(s)} to {format_timestamp(e)}" for s, e in exclude)
        parts.append(
            "These moments are already used. Do not pick moments overlapping these ranges:\n"
            + ranges
        )
    parts.append(f"Transcript:\n<transcript>\n{compact}\n</transcript>")
    return "\n\n".join(parts)


def _overlap(a: ClipPick, b: ClipPick) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _rank(p: ClipPick) -> tuple[bool, int]:
    """`post` beats `maybe`; within a verdict, the higher rubric total wins."""
    return p.verdict == "post", p.scores.total


def dedupe_overlaps(picks: list[ClipPick], max_overlap: float = 0.5) -> list[ClipPick]:
    kept: list[ClipPick] = []
    for p in sorted(picks, key=_rank, reverse=True):
        if all(
            _overlap(p, k) / max(min(p.end - p.start, k.end - k.start), 1e-9) <= max_overlap
            for k in kept
        ):
            kept.append(p)
    return kept


def diversify(ranked: list[ClipPick]) -> list[ClipPick]:
    """Within each verdict tier (`post` first, then the rest), keep rank order but move every
    pick whose `category` an earlier pick of that tier already has (case-insensitive) below
    the tier's picks of not-yet-seen categories."""
    post = [p for p in ranked if p.verdict == "post"]
    return _diversify_tier(post) + _diversify_tier([p for p in ranked if p.verdict != "post"])


def _diversify_tier(ranked: list[ClipPick]) -> list[ClipPick]:
    seen: set[str] = set()
    fresh: list[ClipPick] = []
    repeats: list[ClipPick] = []
    for p in ranked:
        key = p.category.strip().lower()
        (repeats if key and key in seen else fresh).append(p)
        seen.add(key)
    return fresh + repeats


def polish_pick(transcript: Transcript) -> ClipPick:
    sentences = selection_lines(transcript.words())
    first = sentences[0].text if sentences else ""
    hook = " ".join(first.split()[:8])
    return ClipPick(
        start=0.0,
        end=transcript.duration,
        title=hook or "Full clip",
        hook_text=hook,
        reason="Input is short, so the whole video is used as a single clip.",
        scores=RubricScores(**{k: 0 for k in _SCORE_MAX}),
        first_words=first,
        last_words=sentences[-1].text if sentences else "",
        verdict="post",
    )


def _clamped_scores(raw: _RawScores) -> RubricScores:
    return RubricScores(**{k: min(max(getattr(raw, k), 0), hi) for k, hi in _SCORE_MAX.items()})


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


def _window_overlap(want: list[str], tokens: list[list[str]], line: int, *, forward: bool) -> float:
    """Share of `want` found (as a multiset) in the len(want) transcript tokens that start at
    `line` (forward) or end at `line` (backward). Reading past the line lets declared words
    run into the neighbouring sentence."""
    have: list[str] = []
    span = tokens[line:] if forward else tokens[line::-1]
    for toks in span:
        if len(have) >= len(want):
            break
        have = have + toks if forward else toks + have
    have = have[: len(want)] if forward else have[-len(want) :]
    return sum((Counter(want) & Counter(have)).values()) / len(want)


def _line_belongs(want: list[str], toks: list[str], *, forward: bool) -> bool:
    """Whether the line itself is part of the declared words, not a short setup or
    backchannel line that merely runs into them: the declared first (last) word is among the
    line's first (last) 2 tokens, or >= 60% of the line's own tokens appear in `want`."""
    if not toks:
        return False
    edge = toks[:2] if forward else toks[-2:]
    if (want[0] if forward else want[-1]) in edge:
        return True
    return sum((Counter(toks) & Counter(want)).values()) / len(toks) >= _WORDS_MIN_OVERLAP


def _matches(want: list[str], tokens: list[list[str]], line: int, *, forward: bool) -> bool:
    return _window_overlap(
        want, tokens, line, forward=forward
    ) >= _WORDS_MIN_OVERLAP and _line_belongs(want, tokens[line], forward=forward)


def _reanchor(
    raw: _RawClip, line: int, lo: int, hi: int, tokens: list[list[str]], *, forward: bool
) -> int:
    """`line`, or the line in [lo, hi] (within ±3 of it) that the declared first_words
    (forward) or last_words (backward) actually match. Never drops; logs a move or a miss."""
    field = "first_words" if forward else "last_words"
    want = _tokens(raw.first_words if forward else raw.last_words)
    if not want or _matches(want, tokens, line, forward=forward):
        return line
    lo, hi = max(lo, line - _REANCHOR_RADIUS), min(hi, line + _REANCHOR_RADIUS)
    found = [
        (j, _window_overlap(want, tokens, j, forward=forward))
        for j in range(lo, hi + 1)
        if _matches(want, tokens, j, forward=forward)
    ]
    if found:  # best overlap wins; ties go to the line nearest the one Claude gave
        best = max(found, key=lambda f: (f[1], -abs(f[0] - line)))[0]
        logger.info("clip %r: %s found on L%d, not L%d; moved", raw.title, field, best, line)
        return best
    logger.warning(
        "clip %r: %s %r match none of L%d-L%d; keeping L%d",
        raw.title,
        field,
        raw.first_words if forward else raw.last_words,
        lo,
        hi,
        line,
    )
    return line


def _is_backchannel(toks: list[str]) -> bool:
    return 0 < len(toks) <= 2 and all(t in _BACKCHANNEL for t in toks)


def _is_finished(text: str) -> bool:
    t = text.rstrip().rstrip(_TRAILING_QUOTES).rstrip()
    return not t.endswith(_UNFINISHED_ENDINGS) and t.endswith((".", "!", "?"))


def _is_marked(sentence: Sentence, heat: frozenset[int] | set[int] | None) -> bool:
    """True when a loud or chat-spike second falls in [floor(start), ceil(end))."""
    if not heat:
        return False
    lo = math.floor(sentence.start)
    hi = max(math.ceil(sentence.end), lo + 1)
    return any(sec in heat for sec in range(lo, hi))


def _fix_ending(raw: _RawClip, first: int, last: int, src: _Source) -> int:
    """Never end on a backchannel (step back one line) unless the clip is a reaction peak or
    the line carries a loud / chat-spike marker (the nod is the payoff), and finish an unfinished last
    sentence by extending up to 2 lines / 8 s (within max_clip_s) to the first line with
    terminal punctuation. Logs each adjustment."""
    sents, toks = src.sentences, src.tokens
    if (
        last > first
        and _is_backchannel(toks[last])
        and raw.hook_type != "reaction_peak"
        and not _is_marked(sents[last], src.heat)
    ):
        logger.info("clip %r: L%d is a backchannel; ending on L%d", raw.title, last, last - 1)
        last -= 1
    if _is_finished(sents[last].text):
        return last
    for j in range(last + 1, min(last + _EXTEND_MAX_LINES, len(sents) - 1) + 1):
        too_long = sents[j].end - sents[last].end > _EXTEND_MAX_S
        if too_long or _is_backchannel(toks[j]):
            break
        if sents[j].end - sents[first].start > src.settings.max_clip_s:
            break
        if _is_finished(sents[j].text):
            logger.info("clip %r: L%d is unfinished; extended to L%d", raw.title, last, j)
            return j
    logger.warning("clip %r: L%d is unfinished and could not be completed", raw.title, last)
    return last


def _drop_cold_open(raw: _RawClip, reason: str) -> None:
    logger.warning("dropped cold open for clip %r: %s", raw.title, reason)


def _word_norm(text: str) -> str:
    return text.lower().replace("’", "'").strip(_WORD_PUNCT)


def _clip_words(src: _Source, start: float, end: float) -> list[Word]:
    return [w for w in src.words if w.start >= start - 1e-6 and w.end <= end + 1e-6]


def _flat_tokens(words: list[Word]) -> tuple[list[str], list[int]]:
    """Every normalised token of `words`, with the index of the word it came from."""
    toks: list[str] = []
    owner: list[int] = []
    for i, w in enumerate(words):
        for t in _tokens(w.text):
            toks.append(t)
            owner.append(i)
    return toks, owner


def _locate(want: list[str], words: list[Word]) -> tuple[int, int] | None:
    """(first, last) word index of the best order-aware fuzzy match of `want` in `words`: a
    window starting on one of want's first 3 tokens whose in-order matches cover >= 80% of
    `want`, spanning its first to its last matched token; the earliest best window wins. None
    when nothing matches well enough."""
    toks, owner = _flat_tokens(words)
    best: tuple[float, int, int] | None = None
    for i, tok in enumerate(toks):
        if tok not in want[:3]:
            continue
        window = toks[i : i + len(want) + 2]
        blocks = [
            b
            for b in SequenceMatcher(None, want, window, autojunk=False).get_matching_blocks()
            if b.size
        ]
        score = sum(b.size for b in blocks) / len(want)
        if score >= _COLD_OPEN_MIN_MATCH and (best is None or score > best[0]):
            last_tok = i + blocks[-1].b + blocks[-1].size - 1
            best = (score, owner[i + blocks[0].b], owner[last_tok])
    return (best[1], best[2]) if best else None


def _phrase_at(words: list[Word], i: int, phrases: tuple[str, ...]) -> int:
    """Length in words of the longest phrase in `phrases` that `words` spell from index i."""
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        parts = phrase.split()
        if [_word_norm(w.text) for w in words[i : i + len(parts)]] == parts:
            return len(parts)
    return 0


def _teaser(
    raw: _RawClip, first: int, last: int, src: _Source
) -> tuple[list[Word], int, int] | None:
    """(clip words, first, last word index) of the declared `cold_open_words` inside L{first}..
    L{last}, after stripping leading fillers and lead-ins; None when absent or not found."""
    want = _tokens((raw.cold_open_words or "").strip())
    if not want:
        return None
    words = _clip_words(src, src.sentences[first].start, src.sentences[last].end)
    loc = _locate(want, words)
    if loc is None:
        return None
    i, j = loc
    while i < j and (n := _phrase_at(words, i, _LEAD_INS)) and i + n <= j:
        i += n
    return words, i, j


def _extend_teaser_back(words: list[Word], i: int, k: int, src: _Source) -> int | None:
    """The latest earlier start inside the sentence holding words[i] that, after stripping
    lead-ins from it, makes the teaser words[start..k] last >= _COLD_OPEN_MIN_S (and <= 6 s); None."""
    lo = i
    line_start = max(s.start for s in src.sentences if s.start <= words[i].start + 1e-6)
    while lo > 0 and words[lo - 1].start >= line_start - 1e-6:
        lo -= 1
    for s in range(i - 1, lo - 1, -1):
        if words[k].end - words[s].start > _COLD_OPEN_MAX_S + 1e-6:
            return None
        start = s
        while start < i and (n := _phrase_at(words, start, _LEAD_INS)):
            start += n
        if start < i and words[k].end - words[start].start >= _COLD_OPEN_MIN_S - 1e-6:
            return start
    return None


def _cold_open(raw: _RawClip, first: int, last: int, src: _Source) -> tuple[float, float] | None:
    """Where the verbatim `cold_open_words` are spoken inside the clip, without leading fillers
    and lead-ins ("I think", "the thing is"), ending on the first sentence end at or after the
    declared words, all within 6 s (else on an earlier sentence end within 6 s). None (with a
    logged reason) when it cannot be used."""
    text = (raw.cold_open_words or "").strip()
    if not _tokens(text):
        return None
    sents = src.sentences
    found = _teaser(raw, first, last, src)
    if found is None:
        _drop_cold_open(raw, f"{text!r} not found in the clip (L{first}-L{last})")
        return None
    words, i, j = found
    if words[i].start < sents[first].end - 1e-6:
        _drop_cold_open(raw, f"{text!r} starts in the clip's first sentence (L{first})")
        return None
    t0 = words[i].start
    ends = [
        k
        for k in range(i, len(words))
        if ends_sentence(words[k].text) and words[k].end - t0 <= _COLD_OPEN_MAX_S + 1e-6
    ]
    after = [k for k in ends if k >= j]
    if after:
        k = after[0]
    elif ends:
        k = ends[-1]
        logger.info("clip %r: cold open cut to the sentence ending %r", raw.title, words[k].text)
    else:
        _drop_cold_open(raw, f"{text!r} does not finish a sentence within {_COLD_OPEN_MAX_S:.0f}s")
        return None
    if words[k].end - t0 < _COLD_OPEN_MIN_S:
        back = _extend_teaser_back(words, i, k, src)
        if back is None or words[back].start < sents[first].end - 1e-6:
            _drop_cold_open(
                raw,
                f"{text!r} is under {_COLD_OPEN_MIN_S}s and its sentence has no earlier words "
                "(after lead-ins) to reach it",
            )
            return None
        t0 = words[back].start
    return t0, words[k].end


def _bad_opener(words: list[Word]) -> bool:
    """A first sentence that leans on context the viewer never saw: it opens on a quote mark,
    or (after fillers such as "um" / "yeah", and after a bare And/So followed by a 0.3 s pause,
    which the filler trimmer removes) on And/So/But/Because/Which or a quoted conversation
    ("I was like, ...", "he goes ...", "they said ...")."""
    text = " ".join(w.text.strip() for w in words).strip()
    if not text:
        return False
    if text[:1] in _QUOTE_CHARS:
        return True
    norms = [_word_norm(w.text) for w in words]
    i = 0
    while i < len(words):
        if not norms[i]:
            i += 1
        elif i + 1 < len(words) and f"{norms[i]} {norms[i + 1]}" in _OPENER_FILLERS:
            i += 2
        elif norms[i] in _OPENER_FILLERS or (
            norms[i] in _CONTEXT_OPENERS
            and i + 1 < len(words)
            and words[i + 1].start - words[i].end >= _CONJ_PAUSE_S - 1e-6
        ):
            i += 1
        else:
            break
    if i >= len(words):
        return False
    if norms[i] in _CONTEXT_OPENERS:
        return True
    rest = " ".join(w.text.strip() for w in words[i:]).lower().replace("’", "'")
    return bool(_QUOTED_OPENER_RE.match(rest))


def _sentence_words(src: _Source, line: int) -> list[Word]:
    s = src.sentences[line]
    return _clip_words(src, s.start, s.end)


def _weak_start(src: _Source, line: int) -> bool:
    return _bad_opener(_sentence_words(src, line)) or _is_backchannel(src.tokens[line])


def _guard_opener(raw: _RawClip, first: int, last: int, src: _Source, min_s: float) -> int:
    """Move a weak opening sentence forward by 1-2 sentences (at most 12 s, never onto or past
    the cold-open teaser's sentence) while the clip keeps its minimum length; otherwise keep it
    and log. Reaction peaks are exempt."""
    sents = src.sentences
    if raw.hook_type == "reaction_peak" or not _bad_opener(_sentence_words(src, first)):
        return first
    found = _teaser(raw, first, last, src)
    teaser_at = found[0][found[1]].start if found else math.inf
    for k in (1, 2):
        cand = first + k
        if (
            cand > last
            or sents[last].end - sents[cand].start < min_s
            or sents[cand].start - sents[first].start > _GUARD_MAX_MOVE_S
            or sents[cand].end > teaser_at + 1e-6  # never start on or past the teaser
        ):
            break
        if not _bad_opener(_sentence_words(src, cand)):
            logger.info(
                "clip %r: L%d is a weak opener (%r); start moved to L%d",
                raw.title,
                first,
                sents[first].text,
                cand,
            )
            return cand
    logger.warning(
        "clip %r: L%d is a weak opener (%r) with no clean start within 2 sentences / %.0fs; "
        "keeping it",
        raw.title,
        first,
        sents[first].text,
        _GUARD_MAX_MOVE_S,
    )
    return first


def _trim_front(raw: _RawClip, first: int, last: int, src: _Source, min_s: float) -> int | None:
    """For a span longer than max_clip_s: the earliest later start L{c} (keeping `last`, the
    payoff) whose clip lasts <= max_clip_s (else <= 1.25 x max_clip_s) and >= `min_s`, that is
    not a weak opener or a backchannel, and that does not cut into the cold-open teaser. If
    there is none, the untrimmed span is kept when it is within 1.25 x max_clip_s (the old
    tolerance); otherwise None (logged)."""
    sents = src.sentences
    cap = src.settings.max_clip_s
    span = sents[last].end - sents[first].start
    found = _teaser(raw, first, last, src)
    teaser_at = found[0][found[1]].start if found else math.inf
    for limit in (cap, cap * 1.25):
        for c in range(first + 1, last + 1):
            dur = sents[last].end - sents[c].start
            if dur > limit:
                continue
            if dur < min_s or sents[c].end > teaser_at + 1e-6:
                break
            if _weak_start(src, c):
                continue
            logger.info(
                "clip %r: trimmed from the front to fit (%.1fs -> %.1fs): L%d -> L%d",
                raw.title,
                span,
                dur,
                first,
                c,
            )
            return c
    if span <= cap * 1.25:
        logger.warning(
            "clip %r: %.1fs is over %.0fs and no clean later start fits; kept untrimmed",
            raw.title,
            span,
            cap,
        )
        return first
    logger.warning(
        "dropped clip %r: %.1fs is over %.0fs and no clean later start fits", raw.title, span, cap
    )
    return None


def _contains(toks: list[str], want: list[str]) -> bool:
    n = len(want)
    return any(toks[i : i + n] == want for i in range(len(toks) - n + 1))


def _check_title(raw: _RawClip, clip_toks: list[str]) -> str:
    """Keep a quoted span only when it is verbatim (normalised) in the clip; otherwise drop
    its quote marks, since a quote that was never said is a misquote."""

    def unquote(m: re.Match) -> str:
        inner = next(g for g in m.groups() if g is not None)
        want = _tokens(inner)
        if want and _contains(clip_toks, want):
            return m.group(0)
        logger.info(
            "clip %r: quoted %r is not verbatim in the clip; quotes removed", raw.title, inner
        )
        return inner

    return " ".join(_QUOTED_SPAN_RE.sub(unquote, raw.title).split())


def _check_caption(raw: _RawClip) -> str:
    caption = raw.post_caption.strip()
    questions = [q for q in re.split(r"(?<=[.!?])\s+", caption) if q.endswith("?")]
    if any(_TEMPLATE_QUESTION_RE.search(q.lower()) for q in questions):
        logger.warning("clip %r: template caption question in %r; kept", raw.title, caption)
    return caption


def _snap_beats(raw: _RawClip, words: list[Word], start: float, end: float) -> list[str]:
    """Beats with a readable `@ <t>s`: a quoted-word beat (`zoom: '<word>' @ <t>s`) is snapped
    to the nearest start of that word in the clip within ±3 s, else to its only occurrence in
    the clip (dropped if it occurs nowhere or several times far away);
    any other beat must have its time in [start, end]. Logs the ones dropped."""
    toks, owner = _flat_tokens(words)
    kept, dropped = [], []
    for beat in raw.edit_beats:
        m = _BEAT_TIME_RE.search(beat)
        if not m:
            dropped.append(beat)
            continue
        t = float(m.group(1))
        quoted = _BEAT_WORD_RE.match(beat)
        if quoted:
            want = _tokens(quoted.group(1))
            n = len(want)
            hits = [
                words[owner[i]].start
                for i in range(len(toks) - n + 1)
                if n and toks[i : i + n] == want
            ]
            near = [h for h in hits if abs(h - t) <= _BEAT_SNAP_S]
            if not near and len(hits) == 1:  # off by more than 3 s, but unambiguous
                logger.info("clip %r: beat %r snapped to its only occurrence", raw.title, beat)
                near = hits
            if not near:
                dropped.append(beat)
                continue
            snapped = min(near, key=lambda h: abs(h - t))
            kept.append(f"{beat[: m.start()]}@ {snapped:.1f}s")
        elif start <= t <= end:
            kept.append(beat)
        else:
            dropped.append(beat)
    if dropped:
        logger.info(
            "clip %r: dropped %d edit beat(s) without a time inside %.1f-%.1fs or a matching "
            "word within %.0fs: %s",
            raw.title,
            len(dropped),
            start,
            end,
            _BEAT_SNAP_S,
            dropped,
        )
    return kept


def _clean_hashtags(raw: _RawClip) -> list[str]:
    out: list[str] = []
    for tag in raw.hashtags:
        t = re.sub(r"[^\w]", "", tag)
        if t and t.lower() not in _BANNED_HASHTAGS and t.lower() not in map(str.lower, out):
            out.append(t)
    if len(out) < 3:
        logger.warning(
            "clip %r: only %d usable hashtag(s) from %r", raw.title, len(out), raw.hashtags
        )
    return out[:5]


def _clip_title(text: str, limit: int = _SEARCH_TITLE_MAX) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
    return cut.rstrip(" ,;:-–—")


def _to_pick(raw: _RawClip, src: _Source) -> ClipPick | None:
    n = len(src.sentences)
    if not n:
        return None
    lines = {"start_line": raw.start_line, "end_line": raw.end_line}
    bad = [f"{k}=L{v}" for k, v in lines.items() if not 0 <= v < n]
    if bad:
        logger.warning(
            "clip %r: %s out of range (L0-L%d); using the nearest line",
            raw.title,
            ", ".join(bad),
            n - 1,
        )
    given_first = min(max(raw.start_line, 0), n - 1)
    given_last = min(max(raw.end_line, 0), n - 1)
    if given_last < given_first:
        logger.warning(
            "dropped clip %r: ends (L%d) before it starts (L%d)", raw.title, given_last, given_first
        )
        return None
    first = _reanchor(raw, given_first, 0, given_last, src.tokens, forward=True)
    last = _reanchor(raw, given_last, first, n - 1, src.tokens, forward=False)
    last = _fix_ending(raw, first, last, src)
    min_s, max_s = src.settings.min_clip_s * 0.75, src.settings.max_clip_s * 1.25
    if raw.hook_type == "reaction_peak":
        min_s = min(min_s, _REACTION_PEAK_MIN_S)

    def fits(a: int, b: int) -> bool:
        return min_s <= src.sentences[b].end - src.sentences[a].start <= max_s

    # Moving a boundary to match first_words / last_words must never cost the pick.
    if not fits(first, last) and first != given_first:
        logger.warning(
            "clip %r: start moved to L%d breaks the length bounds; back to L%d",
            raw.title,
            first,
            given_first,
        )
        first = given_first
    if not fits(first, last) and last != given_last and given_last >= first:
        logger.warning(
            "clip %r: end moved to L%d breaks the length bounds; back to L%d",
            raw.title,
            last,
            given_last,
        )
        last = given_last
    if src.sentences[last].end - src.sentences[first].start > src.settings.max_clip_s:
        trimmed = _trim_front(raw, first, last, src, min_s)
        if trimmed is None:
            return None
        first = trimmed
    else:
        first = _guard_opener(raw, first, last, src, min_s)
    start = min(max(src.sentences[first].start, 0.0), src.duration)
    end = min(max(src.sentences[last].end, 0.0), src.duration)
    words = _clip_words(src, start, end)
    pick = ClipPick(
        start=start,
        end=end,
        title=_check_title(raw, _flat_tokens(words)[0]),
        hook_text=raw.hook_text,
        reason=raw.reason,
        scores=_clamped_scores(raw.scores),
        edit_beats=_snap_beats(raw, words, start, end),
        first_words=src.sentences[first].text,
        last_words=src.sentences[last].text,
        hook_type=raw.hook_type,
        verdict=raw.verdict,
        post_caption=_check_caption(raw),
        hashtags=_clean_hashtags(raw),
        search_title=_clip_title(raw.search_title),
        category=raw.category.strip(),
        comment_risk=raw.comment_risk,
    )
    if not min_s <= pick.duration <= max_s:
        logger.warning(
            "dropped clip %r: %.1fs is outside %.1f-%.1fs", raw.title, pick.duration, min_s, max_s
        )
        return None
    cold_open = _cold_open(raw, first, last, src)
    if cold_open:
        if pick.duration + (cold_open[1] - cold_open[0]) <= max_s:
            pick = pick.model_copy(update={"cold_open": cold_open})
        else:
            _drop_cold_open(raw, f"would push the clip over the {max_s:.0f}s maximum length")
    if any(min(pick.end, e) - max(pick.start, s) > 0 for s, e in src.exclude):
        return None
    return pick


def _ask(
    llm: LLMBackend,
    prompt: str,
    settings: Settings,
    label: str = "",
    raw_sink: list[dict] | None = None,
) -> _RawSelection:
    """One LLM call with a single retry on backend or payload-validation failure."""
    for attempt in (1, 2):
        try:
            payload = llm.complete_json(
                SYSTEM_PROMPT, prompt, SELECTION_SCHEMA, settings.select_model
            )
            if raw_sink is not None:
                raw_sink.append(payload)
            return _RawSelection.model_validate(payload)
        except (LLMError, ValidationError) as e:
            if attempt == 2:
                raise LLMError(f"clip selection{label} failed after retry: {e}") from e
            logger.warning("clip selection%s attempt failed, retrying: %s", label, e)
    raise AssertionError("unreachable")


def _chunks(
    sentences: list[Sentence], max_chars: int, markers: dict[int, list[str]] | None = None
) -> list[list[Sentence]]:
    """Split sentences (one compact line each) into chunks of at most `max_chars` of new lines;
    each chunk after the first is prefixed with the previous chunk's final 2 minutes."""
    primaries: list[list[Sentence]] = []
    cur: list[Sentence] = []
    size = 0
    # Line lengths from the whole transcript, so markers attach exactly as in the full text.
    lines = compact_transcript(sentences, markers).split("\n")
    for s, line in zip(sentences, lines, strict=True):
        line_len = len(line) + 1
        if cur and size + line_len > max_chars:
            primaries.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += line_len
    if cur:
        primaries.append(cur)
    chunks = [primaries[0]] if primaries else []
    for prev, primary in itertools.pairwise(primaries):
        cutoff = prev[-1].end - _CHUNK_OVERLAP_S
        chunks.append([s for s in prev if s.start >= cutoff] + primary)
    return chunks


def select_clips(
    transcript: Transcript,
    llm: LLMBackend,
    settings: Settings,
    exclude: list[tuple[float, float]] = [],  # noqa: B006 - never mutated
    raw_sink: list[dict] | None = None,
    markers: dict[int, list[str]] | None = None,
    heat_seconds: set[int] | None = None,
) -> list[ClipPick]:
    """Pick up to `settings.clips` post/maybe clips, sorted by start. If `raw_sink` is given,
    each raw LLM payload is appended to it. `markers` (second -> heat markers) annotate the
    transcript lines Claude reads. Raises NoClipsSelectedError if every candidate was `skip`."""
    if transcript.duration < settings.short_input_s:
        return [polish_pick(transcript)]

    heat = heat_seconds
    if heat is None and markers:  # fall back to each run's first second
        heat = {
            sec
            for sec, ms in markers.items()
            if any(m.startswith(("\U0001f50a", "\U0001f525")) for m in ms)
        }
    words = transcript.words()
    sentences = selection_lines(words)
    src = _Source(
        sentences=sentences,
        tokens=[_tokens(s.text) for s in sentences],
        words=words,
        duration=transcript.duration,
        settings=settings,
        exclude=exclude,
        heat=frozenset(heat) if heat is not None else None,
    )
    compact = compact_transcript(sentences, markers)
    returned: list[_RawClip] = []

    def valid(raws: list[_RawClip]) -> list[ClipPick]:
        returned.extend(raws)
        return [
            p for raw in raws if raw.verdict != "skip" and (p := _to_pick(raw, src)) is not None
        ]

    if len(compact) / 4 > settings.max_prompt_tokens:
        chunks = _chunks(sentences, settings.max_prompt_tokens * 4, markers)
        # Chunk lines are cut from the full text: re-rendering a chunk alone would renumber its
        # lines and hand its last sentence every marker after it.
        line_of = dict(zip(map(id, sentences), compact.split("\n"), strict=True))
        logger.info("transcript over prompt budget; selecting across %d chunks", len(chunks))
        raws = []
        for i, chunk in enumerate(chunks, 1):
            label = (
                f" (chunk {i}/{len(chunks)}, "
                f"{format_timestamp(chunk[0].start)}-{format_timestamp(chunk[-1].end)})"
            )
            prompt = build_prompt(
                "\n".join(line_of[id(s)] for s in chunk),
                settings.clips,
                transcript.duration,
                exclude,
                excerpt=(chunk[0].start, chunk[-1].end),
                min_s=settings.min_clip_s,
                max_s=settings.max_clip_s,
            )
            raws.extend(_ask(llm, prompt, settings, label, raw_sink).clips)
        best = diversify(dedupe_overlaps(valid(raws)))[: settings.clips]
    else:
        prompt = build_prompt(
            compact,
            settings.clips + 3,
            transcript.duration,
            exclude,
            min_s=settings.min_clip_s,
            max_s=settings.max_clip_s,
        )
        first = _ask(llm, prompt, settings, raw_sink=raw_sink).clips
        best = diversify(dedupe_overlaps(valid(first)))[: settings.clips]
        if len(first) < settings.clips:  # too few returned (not: too few kept after skips)
            best = _follow_up(best, compact, llm, settings, src, valid, raw_sink)

    if returned and all(r.verdict == "skip" for r in returned):
        raise NoClipsSelectedError(ALL_SKIP_MESSAGE)
    if len(best) < settings.clips:
        logger.warning("only %d valid clip(s) selected (wanted %d)", len(best), settings.clips)
    return sorted(best, key=lambda p: p.start)


def _follow_up(
    best: list[ClipPick],
    compact: str,
    llm: LLMBackend,
    settings: Settings,
    src: _Source,
    valid: Callable[[list[_RawClip]], list[ClipPick]],
    raw_sink: list[dict] | None = None,
) -> list[ClipPick]:
    """One extra call for the missing picks, steering away from what is already accepted."""
    missing = settings.clips - len(best)
    logger.warning(
        "only %d valid clip(s) in the first response (wanted %d); asking for more",
        len(best),
        settings.clips,
    )
    taken = [*src.exclude, *((p.start, p.end) for p in best)]
    prompt = build_prompt(
        compact,
        missing + 3,
        src.duration,
        taken,
        min_s=settings.min_clip_s,
        max_s=settings.max_clip_s,
    )
    try:
        more = _ask(llm, prompt, settings, " (follow-up)", raw_sink).clips
    except LLMError as e:
        logger.warning("follow-up clip selection failed; keeping %d clip(s): %s", len(best), e)
        return best
    return diversify(dedupe_overlaps(best + valid(more)))[: settings.clips]
