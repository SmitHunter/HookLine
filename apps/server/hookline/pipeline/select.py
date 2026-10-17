"""Clip selection: Claude reads the whole transcript and picks the best short-form moments."""

import itertools
import logging
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from hookline.config import Settings
from hookline.llm.base import LLMBackend, LLMError
from hookline.models import ClipPick, RubricScores, Sentence, Transcript
from hookline.transcript_text import compact_transcript, format_timestamp, split_sentences

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert short-form video editor. You cut long recordings into vertical clips for \
TikTok, YouTube Shorts and Instagram Reels, and you have a sharp instinct for the moments that \
make a stranger stop scrolling and watch to the end.

## What you are given

A timestamped transcript of one long video. Most inputs are multi-hour live streams (Twitch \
VODs: gameplay with a facecam, or Just Chatting); some are podcasts, interviews or films. Each \
line is `[timestamp] sentence`, where the timestamp is when the sentence starts, written as \
`mm:ss.s` (or `h:mm:ss.s` past the first hour) from the start of the video. You only have the \
words, so read energy from them: exclamations, laughter, sudden reversals, arguments, reveals, \
fails, wins, confessions and strong opinions.

## Your task

Find the strongest self-contained moments and return them as clip candidates, ranked best \
first. Each clip will be watched by someone who has never seen this creator and has no idea \
what happened earlier. They should understand it immediately and feel something by the end.

## How to choose a moment

- Stand alone. Live streams are full of running jokes, chat references, raid shout-outs and \
callbacks to earlier in the stream. Favour moments that make complete sense with zero stream \
context. A slightly smaller moment that anyone can follow beats a big one that needs backstory.
- Hook within the first 3 seconds. The first line the viewer hears must create a question, \
tension, surprise or a bold claim that makes them need the rest.
- Open on the strongest line. Start the clip at its most gripping sentence whenever the clip \
still makes sense from there. If the natural opening is weak but the clip needs it for context, \
use a cold open instead (see `cold_open_start` / `cold_open_end` below).
- Never begin mid-thought. Avoid openings such as "so anyway", "and then", "but yeah", "like I \
said", or an answer to a question the viewer never heard. A sentence that begins with "And", \
"So", "But", "Because", "Which", or a pronoun referring to something said earlier usually \
continues a previous thought. Prefer starting the clip at the sentence that introduces the \
subject. When the opening still depends on earlier context, score `hook` and `context` lower.
- Start and end on complete sentences. Begin on the first word of a sentence and end on the \
last word of one; never cut a sentence in half at either end.
- Land the payoff. End just after the punchline, reveal, reaction or resolution. Do not trail \
off into dead air, chat-reading or the next topic.
- Length: 20–60 seconds. 20–45s usually wins; 60s only if tension holds the whole way.
- Keep candidates distinct. They must not overlap each other in time.

## Scoring rubric (0–100)

Rubric: Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · Length fit 10 · \
Pacing 10 · Loop potential 5 · Platform safety 5.

Score each candidate honestly with whole numbers, each within its range:
- `hook` (0–25): how hard the first 3 seconds grab a cold viewer.
- `context` — Standalone context (0–15): how well it works with no knowledge of the stream, \
the creator or what came before.
- `payoff` (0–15): how satisfying the ending is — a punchline, reveal, outcome or reaction that \
rewards watching.
- `emotion` — Emotional intensity (0–15): how strong the feeling is: hype, rage, shock, \
laughter, tension, awkwardness or heart.
- `length_fit` (0–10): how well the length suits the material; full marks for a tight 20–45s \
cut, fewer as it drifts toward 60s without the tension to carry it.
- `pacing` (0–10): density of the speech and events; no long pauses, rambling or filler.
- `loop` — Loop potential (0–5): whether the ending flows back into the opening so a replay \
feels natural.
- `safety` — Platform safety (0–5): free of slurs, explicit sexual content, graphic violence or \
anything likely to be suppressed or removed by the platforms.

## Fields to return for each candidate

- `start`, `end`: absolute times in seconds from the start of the video, as numbers (convert \
`1:02:03.5` to `3723.5`). `start` is the timestamp of the clip's first line. For `end`, give \
the timestamp of the line after the clip's last line; the clip is cut where its last line \
finishes, so that following line is not included. If the clip's last line is the final line of \
the transcript, set `end` to the video duration.
- `cold_open_start`, `cold_open_end`: when the natural opening is weak, set these to ONE WHOLE \
SENTENCE from inside the clip (between `start` and `end`) that lasts at most 3 seconds; only \
short, punchy sentences qualify. That sentence plays first as a teaser, then the clip runs from \
`start`. Use it only when the clip needs its setup and the strongest line comes later. If no \
such sentence exists, leave both fields null. Use the same convention as `start` and \
`end`: `cold_open_start` is the timestamp of the teaser line, and `cold_open_end` is the \
timestamp of the line after it (the video duration if the teaser is the transcript's final \
line). The teaser may be the clip's own last line.
- `title`: a short, specific title for the upload, under 60 characters.
- `hook_text`: on-screen text shown over the first seconds, at most 8 words. It should tease \
the payoff without spoiling it, and not simply repeat the first spoken line.
- `reason`: one or two sentences on why this moment works and what the payoff is.
- `scores`: the eight rubric scores above.
- `edit_beats`: a few short editing cues tied to the moments that matter, each with an \
absolute timestamp in seconds, such as "zoom: 'no way' @ 12.4s", "emoji 😱 @ 8.0s" or \
"emphasis: 'never' @ 30.2s".

## Ranking

Return the number of candidates requested, ranked best first by total score. If the material \
is thin, still return the requested number and score the weaker ones honestly; low scores tell \
the editor which ones to skip. Never pick a moment that overlaps a range you are told to avoid.
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


_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}

SELECTION_SCHEMA: dict = _object(
    {
        "clips": {
            "type": "array",
            "items": _object(
                {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "cold_open_start": _NULLABLE_NUMBER,
                    "cold_open_end": _NULLABLE_NUMBER,
                    "title": {"type": "string"},
                    "hook_text": {"type": "string"},
                    "reason": {"type": "string"},
                    "scores": _object({k: {"type": "integer"} for k in _SCORE_MAX}),
                    "edit_beats": {"type": "array", "items": {"type": "string"}},
                }
            ),
        }
    }
)

_COLD_OPEN_MAX_S = 3.5
_CHUNK_OVERLAP_S = 120.0
_SNAP_TOLERANCE_S = 0.5


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
    start: float
    end: float
    cold_open_start: float | None = None
    cold_open_end: float | None = None
    title: str
    hook_text: str
    reason: str
    scores: _RawScores
    edit_beats: list[str] = []


class _RawSelection(BaseModel):
    clips: list[_RawClip]


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
            f"{format_timestamp(excerpt[1])} of a longer video. Timestamps are still measured "
            "from the start of the full video."
        )
    parts.append(f"Return the {n} best clip candidates, ranked best first.")
    if exclude:
        ranges = "\n".join(f"- {format_timestamp(s)} to {format_timestamp(e)}" for s, e in exclude)
        parts.append(
            "These moments are already used. Do not pick moments overlapping these ranges:\n"
            + ranges
        )
    parts.append(f"Transcript:\n<transcript>\n{compact}\n</transcript>")
    return "\n\n".join(parts)


def snap_to_sentences(pick: ClipPick, sentences: list[Sentence], duration: float) -> ClipPick:
    """Snap a pick onto whole sentences, then clamp both ends to [0, duration].

    - start: the latest sentence start <= pick.start + 0.5 (else the first sentence's start).
    - end: the end of the last sentence that starts before pick.end, so an `end` given as the
      next line's timestamp (or falling mid-sentence) finishes that sentence and adds no extra
      one. If no sentence starts before pick.end, the end of the first sentence starting at or
      after pick.start.
    """
    start, end = _snap_span(pick.start, pick.end, sentences, duration)
    return pick.model_copy(update={"start": start, "end": end})


def _snap_span(
    start: float, end: float, sentences: list[Sentence], duration: float
) -> tuple[float, float]:
    """The snapping rules of `snap_to_sentences`, for any (start, end) span."""
    if sentences:
        starts = [s.start for s in sentences if s.start <= start + _SNAP_TOLERANCE_S]
        new_start = starts[-1] if starts else sentences[0].start
        before = [s for s in sentences if s.start < end]
        after = [s for s in sentences if s.start >= start]
        end = before[-1].end if before else (after[0] if after else sentences[-1]).end
        start = new_start
    return min(max(start, 0.0), duration), min(max(end, 0.0), duration)


def _overlap(a: ClipPick, b: ClipPick) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def dedupe_overlaps(picks: list[ClipPick], max_overlap: float = 0.5) -> list[ClipPick]:
    kept: list[ClipPick] = []
    for p in sorted(picks, key=lambda p: p.scores.total, reverse=True):
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
    )


def _clamped_scores(raw: _RawScores) -> RubricScores:
    return RubricScores(**{k: min(max(getattr(raw, k), 0), hi) for k, hi in _SCORE_MAX.items()})


def _drop_cold_open(raw: _RawClip, reason: str) -> None:
    logger.warning("dropped cold open for clip %r: %s", raw.title, reason)


def _cold_open(
    raw: _RawClip, pick: ClipPick, sentences: list[Sentence], duration: float
) -> tuple[float, float] | None:
    """Snap the raw cold open like the clip itself; None unless it then lies inside the
    (snapped) clip and is at most 3.5s long. Logs a warning naming the reason when dropped."""
    if raw.cold_open_start is None or raw.cold_open_end is None:
        return None
    if raw.cold_open_end <= raw.cold_open_start:
        _drop_cold_open(raw, "cold_open_end is not after cold_open_start")
        return None
    co_s, co_e = _snap_span(raw.cold_open_start, raw.cold_open_end, sentences, duration)
    if not pick.start <= co_s < co_e <= pick.end:
        _drop_cold_open(raw, f"outside the clip ({co_s:.1f}-{co_e:.1f}s)")
        return None
    if co_e - co_s > _COLD_OPEN_MAX_S:
        _drop_cold_open(
            raw, f"snapped length {co_e - co_s:.1f}s exceeds the {_COLD_OPEN_MAX_S}s cap"
        )
        return None
    return co_s, co_e


def _to_pick(
    raw: _RawClip,
    sentences: list[Sentence],
    duration: float,
    settings: Settings,
    exclude: list[tuple[float, float]],
) -> ClipPick | None:
    if raw.end <= raw.start or raw.start >= duration:
        return None
    pick = snap_to_sentences(
        ClipPick(
            start=raw.start,
            end=raw.end,
            title=raw.title,
            hook_text=raw.hook_text,
            reason=raw.reason,
            scores=_clamped_scores(raw.scores),
            edit_beats=raw.edit_beats,
        ),
        sentences,
        duration,
    )
    min_s, max_s = settings.min_clip_s * 0.75, settings.max_clip_s * 1.25
    cold_open = _cold_open(raw, pick, sentences, duration)
    if cold_open:
        if pick.duration + (cold_open[1] - cold_open[0]) <= max_s:
            pick = pick.model_copy(update={"cold_open": cold_open})
        else:
            _drop_cold_open(raw, f"would push the clip over the {max_s:.0f}s maximum length")
    if not min_s <= pick.duration <= max_s:
        return None
    if any(min(pick.end, e) - max(pick.start, s) > 0 for s, e in exclude):
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


def _chunks(sentences: list[Sentence], max_chars: int) -> list[list[Sentence]]:
    """Split sentences (one compact line each) into chunks of at most `max_chars` of new lines;
    each chunk after the first is prefixed with the previous chunk's final 2 minutes."""
    primaries: list[list[Sentence]] = []
    cur: list[Sentence] = []
    size = 0
    for s in sentences:
        line_len = len(compact_transcript([s])) + 1
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
) -> list[ClipPick]:
    """Pick the best clips. If `raw_sink` is given, each raw LLM payload is appended to it."""
    if transcript.duration < settings.short_input_s:
        return [polish_pick(transcript)]

    duration = transcript.duration
    sentences = split_sentences(transcript.words())
    compact = compact_transcript(sentences)

    def valid(raws: list[_RawClip]) -> list[ClipPick]:
        return [
            p
            for raw in raws
            if (p := _to_pick(raw, sentences, duration, settings, exclude)) is not None
        ]

    if len(compact) / 4 > settings.max_prompt_tokens:
        chunks = _chunks(sentences, settings.max_prompt_tokens * 4)
        logger.info("transcript over prompt budget; selecting across %d chunks", len(chunks))
        raws = []
        for i, chunk in enumerate(chunks, 1):
            label = (
                f" (chunk {i}/{len(chunks)}, "
                f"{format_timestamp(chunk[0].start)}-{format_timestamp(chunk[-1].end)})"
            )
            prompt = build_prompt(
                compact_transcript(chunk),
                settings.clips,
                duration,
                exclude,
                excerpt=(chunk[0].start, chunk[-1].end),
            )
            raws.extend(_ask(llm, prompt, settings, label, raw_sink).clips)
        best = dedupe_overlaps(valid(raws))[: settings.clips]
    else:
        prompt = build_prompt(compact, settings.clips + 3, duration, exclude)
        best = dedupe_overlaps(valid(_ask(llm, prompt, settings, raw_sink=raw_sink).clips))[
            : settings.clips
        ]
        if len(best) < settings.clips:
            best = _follow_up(best, compact, llm, settings, duration, exclude, valid, raw_sink)

    if len(best) < settings.clips:
        logger.warning("only %d valid clip(s) selected (wanted %d)", len(best), settings.clips)
    return sorted(best, key=lambda p: p.start)


def _follow_up(
    best: list[ClipPick],
    compact: str,
    llm: LLMBackend,
    settings: Settings,
    duration: float,
    exclude: list[tuple[float, float]],
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
    taken = [*exclude, *((p.start, p.end) for p in best)]
    prompt = build_prompt(compact, missing + 3, duration, taken)
    try:
        more = _ask(llm, prompt, settings, " (follow-up)", raw_sink).clips
    except LLMError as e:
        logger.warning("follow-up clip selection failed; keeping %d clip(s): %s", len(best), e)
        return best
    return dedupe_overlaps(best + valid(more))[: settings.clips]
