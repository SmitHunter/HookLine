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
from hookline.pipeline.signals import CLIP_MARKER
from hookline.transcript_text import compact_transcript, format_timestamp, split_sentences

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
such as "So I've been thinking about this" or "Okay, so basically" are not hooks.
- Cold opens. When a clip genuinely needs its setup but its strongest line comes later, set \
`cold_open_line` to that later line. It plays first as a teaser (at most 5 seconds; a longer \
line is trimmed to its first 5 seconds), then the clip runs from `start_line`. Otherwise leave \
it null. It must lie inside the clip and must not be the clip's first line.
- End on the payoff line: the punchline, reveal, outcome or reaction that rewards watching. \
Never end on "yeah", "mm-hmm", "right" or any other interviewer backchannel, on dead air, or \
on the start of the next topic. If the payoff lands within 15 seconds after where you would \
stop, extend the clip to include it. `last_words` is the verbatim last sentence heard, copied \
from the `end_line` line. Prefer endings that echo the opening, so a replay loops naturally.
- Length: prefer 20–40 seconds for talking heads. Go up to 60 seconds only if the tension \
holds the whole way.
- Keep candidates distinct: they must not overlap each other in time.
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
candidates should not score like hits; use the whole range.

## Verdict

`verdict` is your call as the editor who has to put their name on the post:
- `post`: you would publish it as it stands.
- `maybe`: worth a look, but something is soft (the hook, the ending, the context).
- `skip`: not worth posting. Return it as `skip` rather than inflating it.
Use `skip` honestly; a source with no strong moments should get mostly skips.

## Packaging

Everything in packaging must be literally true to the clip.

- `title`: short and specific, under 60 characters.
- `hook_text`: on-screen text over the first seconds. 3–7 words, built from concrete nouns or \
numbers in the clip, and literally true: never exaggerate what happens (if someone was "a \
little rude", they did not "come to mock" anyone). Tease the payoff without spoiling it, and \
do not just repeat the first spoken line. Never use template phrases such as "Then this", \
"Wait for it", "You won't believe" or "Watch till the end".
- `post_caption`: the caption under the post, 1–2 lines, that ends with a question inviting \
comments.
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

A good pick (37 seconds): `start_line` 214, `end_line` 226, `cold_open_line` null.
- `first_words`: "I ran a three-hundred-cover Saturday on my own when I was nineteen." It works \
cold; L212 and L213 are setup, so the clip starts after them.
- `last_words`: "He walked in at close, looked at the tickets, and handed me his knives." The \
payoff; the backchannel "Yeah." on L227 is left out.
- `hook_type`: `bold_claim`. `verdict`: `post`.
- `hook_text`: "Head chef walked in at close" (it teases the payoff rather than restating \
the first line)
- `post_caption`: "Nineteen, no head chef, a full Saturday rush. What's the hardest shift you've \
ever covered?"
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
                    "cold_open_line": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
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

_COLD_OPEN_MAX_S = 5.0
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
    cold_open_line: int | None = None
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


def polish_pick(transcript: Transcript) -> ClipPick:
    sentences = split_sentences(transcript.words())
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


def _cold_open(raw: _RawClip, first: int, last: int, src: _Source) -> tuple[float, float] | None:
    """The `cold_open_line` sentence's span; a sentence over 5 s is trimmed to the words that
    end within its first 5 s. None (with a logged reason) when it cannot be used."""
    line = raw.cold_open_line
    if line is None:
        return None
    if not 0 <= line < len(src.sentences):
        _drop_cold_open(raw, f"L{line} does not exist")
        return None
    if line == first:
        _drop_cold_open(raw, f"L{line} is the clip's first line")
        return None
    if not first < line <= last:
        _drop_cold_open(raw, f"L{line} is outside the clip (L{first}-L{last})")
        return None
    s = src.sentences[line]
    if s.end - s.start <= _COLD_OPEN_MAX_S:
        return s.start, s.end
    inside = [w for w in src.words if w.start >= s.start - 1e-6 and w.end <= s.end + 1e-6]
    kept = [w for w in inside if w.end - s.start <= _COLD_OPEN_MAX_S]
    if not kept:
        _drop_cold_open(raw, f"L{line} has no words within its first {_COLD_OPEN_MAX_S}s")
        return None
    return kept[0].start, kept[-1].end


def _beats_inside(raw: _RawClip, start: float, end: float) -> list[str]:
    """Beats whose `@ <t>s` time lies in [start, end]; logs the ones dropped (no readable
    time, or outside the clip)."""
    kept, dropped = [], []
    for beat in raw.edit_beats:
        m = _BEAT_TIME_RE.search(beat)
        (kept if m and start <= float(m.group(1)) <= end else dropped).append(beat)
    if dropped:
        logger.info(
            "clip %r: dropped %d edit beat(s) without a time inside %.1f-%.1fs: %s",
            raw.title,
            len(dropped),
            start,
            end,
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
    start = min(max(src.sentences[first].start, 0.0), src.duration)
    end = min(max(src.sentences[last].end, 0.0), src.duration)
    pick = ClipPick(
        start=start,
        end=end,
        title=raw.title,
        hook_text=raw.hook_text,
        reason=raw.reason,
        scores=_clamped_scores(raw.scores),
        edit_beats=_beats_inside(raw, start, end),
        first_words=raw.first_words,
        last_words=raw.last_words,
        hook_type=raw.hook_type,
        verdict=raw.verdict,
        post_caption=raw.post_caption.strip(),
        hashtags=_clean_hashtags(raw),
        search_title=_clip_title(raw.search_title),
        category=raw.category.strip(),
        comment_risk=raw.comment_risk,
    )
    cold_open = _cold_open(raw, first, last, src)
    if cold_open:
        if pick.duration + (cold_open[1] - cold_open[0]) <= max_s:
            pick = pick.model_copy(update={"cold_open": cold_open})
        else:
            _drop_cold_open(raw, f"would push the clip over the {max_s:.0f}s maximum length")
    if not min_s <= pick.duration <= max_s:
        return None
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
    sentences = split_sentences(words)
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
            )
            raws.extend(_ask(llm, prompt, settings, label, raw_sink).clips)
        best = dedupe_overlaps(valid(raws))[: settings.clips]
    else:
        prompt = build_prompt(compact, settings.clips + 3, transcript.duration, exclude)
        first = _ask(llm, prompt, settings, raw_sink=raw_sink).clips
        best = dedupe_overlaps(valid(first))[: settings.clips]
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
    prompt = build_prompt(compact, missing + 3, src.duration, taken)
    try:
        more = _ask(llm, prompt, settings, " (follow-up)", raw_sink).clips
    except LLMError as e:
        logger.warning("follow-up clip selection failed; keeping %d clip(s): %s", len(best), e)
        return best
    return dedupe_overlaps(best + valid(more))[: settings.clips]
