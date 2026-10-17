"""Clip selection: Claude reads the whole transcript and picks the best short-form moments.

Claude answers in transcript line numbers (`L{n}` = sentence n of the whole transcript), never
in seconds; this module maps lines back to sentence times, re-anchors a start whose declared
`first_words` sit on a neighbouring line, and drops candidates Claude itself rated `skip`.
"""

import itertools
import logging
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
usual rate, and `🔊loud` means the audio got far louder than normal. Treat them as evidence of \
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
heard, or leans on a pronoun from earlier), move the start or use a cold open. Setup lines \
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
the viewer sees what the streamer is reacting to.
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
start of the video and lies inside the clip (`14:52.0` is 892.0 seconds).

## A worked example (invented)

Transcript excerpt:
L212 [14:03.5] So I'd been a line cook for maybe two years at that point.
L213 [14:07.9] And one night the head chef just doesn't show up.
L214 [14:10.2] I ran a three-hundred-cover Saturday on my own when I was nineteen.
…
L226 [14:52.0] He walked in at close, looked at the tickets, and handed me his knives.
L227 [14:56.1] Yeah.

A good pick: `start_line` 214, `end_line` 226, `cold_open_line` null.
- `first_words`: "I ran a three-hundred-cover Saturday on my own when I was nineteen." It works \
cold; L212 and L213 are setup, so the clip starts after them.
- `last_words`: "He walked in at close, looked at the tickets, and handed me his knives." The \
payoff; the backchannel "Yeah." on L227 is left out.
- `hook_type`: `bold_claim`. `verdict`: `post`.
- `hook_text`: "Nineteen, alone, 300 covers"
- `post_caption`: "Nineteen, no head chef, a full Saturday rush. What's the hardest shift you've \
ever covered?"
- `hashtags`: ["linecook", "kitchenlife", "restaurantstories", "cheflife"]
- `search_title`: "Running a 300-Cover Restaurant Service Alone at 19"
- `edit_beats`: ["zoom: 'knives' @ 895.4s"]

## Ranking

Return the number of candidates requested, ranked best first, each with its verdict. If the \
material is thin, still return them, mark the weak ones `skip` and score them honestly. Never \
pick a moment that overlaps a range you are told to avoid.
"""

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
_FIRST_WORDS_MIN_OVERLAP = 0.6
_REANCHOR_RADIUS = 3
_SEARCH_TITLE_MAX = 70
_BANNED_HASHTAGS = {"fyp", "foryou", "foryoupage", "fy", "viral", "trending"}
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


def _first_words_overlap(want: list[str], tokens: list[list[str]], line: int) -> float:
    """Share of `want` found (as a multiset) in the transcript's first len(want) tokens from
    `line` on. Reading on past the line lets `first_words` run into the next sentence."""
    have: list[str] = []
    for toks in tokens[line:]:
        if len(have) >= len(want):
            break
        have.extend(toks)
    common = Counter(want) & Counter(have[: len(want)])
    return sum(common.values()) / len(want)


def _anchor_start(raw: _RawClip, line: int, tokens: list[list[str]]) -> int:
    """`line`, or the line within ±3 that `first_words` actually matches. Never drops."""
    want = _tokens(raw.first_words)
    if not want or _first_words_overlap(want, tokens, line) >= _FIRST_WORDS_MIN_OVERLAP:
        return line
    lo, hi = max(0, line - _REANCHOR_RADIUS), min(len(tokens) - 1, line + _REANCHOR_RADIUS)
    # best overlap wins; ties go to the line nearest the one Claude gave
    best = max(
        range(lo, hi + 1), key=lambda j: (_first_words_overlap(want, tokens, j), -abs(j - line))
    )
    if _first_words_overlap(want, tokens, best) >= _FIRST_WORDS_MIN_OVERLAP:
        logger.info(
            "clip %r: first_words found on L%d, not L%d; moved start", raw.title, best, line
        )
        return best
    logger.warning(
        "clip %r: first_words %r match none of L%d-L%d; keeping L%d",
        raw.title,
        raw.first_words,
        lo,
        hi,
        line,
    )
    return line


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


def _beats_inside(beats: list[str], start: float, end: float) -> list[str]:
    """Beats whose `@ <t>s` time lies in [start, end]; beats without a time are dropped."""
    kept = []
    for beat in beats:
        m = _BEAT_TIME_RE.search(beat)
        if m and start <= float(m.group(1)) <= end:
            kept.append(beat)
    return kept


def _clean_hashtags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        t = "".join(tag.split()).lstrip("#")
        if t and t.lower() not in _BANNED_HASHTAGS and t.lower() not in map(str.lower, out):
            out.append(t)
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
    first = _anchor_start(raw, min(max(raw.start_line, 0), n - 1), src.tokens)
    last = min(max(raw.end_line, 0), n - 1)
    if last < first:
        logger.warning("dropped clip %r: ends (L%d) before it starts (L%d)", raw.title, last, first)
        return None
    start = min(max(src.sentences[first].start, 0.0), src.duration)
    end = min(max(src.sentences[last].end, 0.0), src.duration)
    pick = ClipPick(
        start=start,
        end=end,
        title=raw.title,
        hook_text=raw.hook_text,
        reason=raw.reason,
        scores=_clamped_scores(raw.scores),
        edit_beats=_beats_inside(raw.edit_beats, start, end),
        first_words=raw.first_words,
        last_words=raw.last_words,
        hook_type=raw.hook_type,
        verdict=raw.verdict,
        post_caption=raw.post_caption.strip(),
        hashtags=_clean_hashtags(raw.hashtags),
        search_title=_clip_title(raw.search_title),
        category=raw.category.strip(),
        comment_risk=raw.comment_risk,
    )
    min_s, max_s = src.settings.min_clip_s * 0.75, src.settings.max_clip_s * 1.25
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
) -> list[ClipPick]:
    """Pick up to `settings.clips` post/maybe clips, sorted by start. If `raw_sink` is given,
    each raw LLM payload is appended to it. `markers` (second -> heat markers) annotate the
    transcript lines Claude reads. Raises NoClipsSelectedError if every candidate was `skip`."""
    if transcript.duration < settings.short_input_s:
        return [polish_pick(transcript)]

    words = transcript.words()
    sentences = split_sentences(words)
    src = _Source(
        sentences=sentences,
        tokens=[_tokens(s.text) for s in sentences],
        words=words,
        duration=transcript.duration,
        settings=settings,
        exclude=exclude,
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
