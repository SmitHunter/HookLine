"""Word and clip-end timing refined against the audio's loudness envelope.

whisper.cpp's DTW gives one timestamp per token, so a single-token word's end is a guess:
`parse_whisper_cpp_json` sets it to the token time + 0.15 s (two thirds of the podcast's words
end exactly 0.15 s after they start). These helpers move ends to where the speech actually
falls silent, snap post-pause onsets to where it rises, and find a clean clip end."""

import numpy as np

from hookline.media.audio import Envelope
from hookline.models import Word

__all__ = [
    "Envelope",
    "protected_words",
    "refine_word_ends",
    "snap_onsets",
    "tail_end",
    "voice_threshold",
]

# Voice threshold (`voice_threshold`): FLOOR_MARGIN_DB over the envelope's FLOOR_QUANTILE (the
# local noise floor), kept between VOICE_MARGIN_DB below the dead-air silence threshold
# (-32 -> -44 dB on a clean source: speech sits around -30 dB and dips under -32 inside words)
# and that silence threshold itself (a noisy source's bed must never count as voice).
FLOOR_QUANTILE = 10
FLOOR_MARGIN_DB = 9.0
VOICE_MARGIN_DB = 12.0
MAX_WORD_S = 1.5  # a word end is never extended past this long after its start ...
MAX_EXTEND_S = 0.6  # ... nor more than this past its transcribed end without an energy fall
PROTECT_S = 0.3  # dead-air protection trusts a refined end at most this past the transcript's
MIN_QUIET_S = 0.05  # a dip must last this long to count as silence (stops bridge)
MIN_VOICED_S = 0.03  # ... and a rise must stay voiced this long to count as an onset
PAUSE_S = 0.2  # a word this long after the previous one's end follows a pause
ONSET_WINDOW_S = 0.25  # a post-pause onset snaps to an energy rise within this of it
MIN_WORD_S = 0.05  # a snapped onset leaves the word at least this long
SNAP_EPS_S = 0.005
MAX_TAIL_S = 0.35  # the clip may end this long after its last word ...
TAIL_PAD_S = 0.08  # ... settling this far into the silence after it ...
TAIL_GUARD_S = 0.05  # ... and never closer than this to the next word


def _frames(env: Envelope, seconds: float) -> int:
    return max(round(seconds / env.hop), 1)


def _covers(env: Envelope, t: float) -> bool:
    return len(env.db) > 0 and env.t0 <= t < env.time(len(env.db))


def _quiet_from(env: Envelope, start: float, limit: float, threshold_db: float) -> float | None:
    """Start of the first silence (>= MIN_QUIET_S below `threshold_db`) in [start, limit), or
    None; a silence still running at `limit` counts from where it began."""
    need = _frames(env, MIN_QUIET_S)
    run_start, run = None, 0
    k = env.index(start)
    while k < len(env.db) and env.time(k) < limit:
        if env.db[k] < threshold_db:
            run_start = k if run == 0 else run_start
            run += 1
            if run >= need:
                return env.time(run_start)
        else:
            run = 0
        k += 1
    return env.time(run_start) if run and run_start is not None else None


def voice_threshold(env: Envelope, silence_db: float) -> float | None:
    """dB above which the envelope counts as voice: the local noise floor (its FLOOR_QUANTILE
    percentile) + FLOOR_MARGIN_DB, at least silence_db - VOICE_MARGIN_DB. None when that
    would be louder than silence_db: the bed is too close to the speech to tell word ends,
    onsets or a clean tail from it (the caller keeps the transcript's timings)."""
    if not len(env.db):
        return silence_db - VOICE_MARGIN_DB
    floor = float(np.percentile(env.db, FLOOR_QUANTILE))
    if floor + FLOOR_MARGIN_DB > silence_db:
        return None
    return max(floor + FLOOR_MARGIN_DB, silence_db - VOICE_MARGIN_DB)


def protected_words(raw: list[Word], refined: list[Word]) -> list[Word]:
    """`refined` words (same order as `raw`) whose ends trust the refinement at most
    PROTECT_S past the transcript's end: what dead-air cutting keeps out of its cuts."""
    return [
        r.model_copy(update={"end": min(r.end, round(w.end + PROTECT_S, 3))})
        for w, r in zip(raw, refined, strict=True)
    ]


def _rise_from(env: Envelope, start: float, threshold_db: float) -> float:
    """Time of the first frame from `start` that stays at or above `threshold_db` for
    MIN_VOICED_S (an audible onset); infinity when there is none in the envelope."""
    voiced = _frames(env, MIN_VOICED_S)
    for k in range(env.index(start), len(env.db)):
        if (env.db[k : k + voiced] >= threshold_db).all():
            return env.time(k)
    return np.inf


def refine_word_ends(
    words: list[Word], env: Envelope, threshold_db: float, max_word_s: float = MAX_WORD_S
) -> list[Word]:
    """Each word's end moved later to where the energy falls below `threshold_db` for
    MIN_QUIET_S (shorter dips, e.g. stop consonants, are bridged), never past the next
    word's start nor `max_word_s` after its own start; with no such fall in reach, at most
    MAX_EXTEND_S past its transcribed end. Ends only ever grow; words outside the envelope
    are untouched."""
    out = list(words)
    for i, word in enumerate(words):
        limit = word.start + max_word_s
        if i + 1 < len(words):
            limit = min(limit, words[i + 1].start)
        if word.end >= limit or not _covers(env, word.end):
            continue
        quiet = _quiet_from(env, word.end, limit, threshold_db)
        end = min(limit, word.end + MAX_EXTEND_S) if quiet is None else quiet
        if end > word.end:
            out[i] = word.model_copy(update={"end": round(end, 3)})
    return out


def snap_onsets(
    words: list[Word],
    env: Envelope,
    threshold_db: float,
    min_gap: float = PAUSE_S,
    window: float = ONSET_WINDOW_S,
) -> list[Word]:
    """Start of each word that follows a pause (>= `min_gap` after the previous word's end,
    or the first word) moved to the nearest energy rise within `window` of it: a frame at or
    above `threshold_db` after MIN_QUIET_S below it, staying up for MIN_VOICED_S. Never before
    the previous word's end, nor leaving the word shorter than MIN_WORD_S."""
    quiet, voiced = _frames(env, MIN_QUIET_S), _frames(env, MIN_VOICED_S)
    db = env.db
    out = list(words)
    for i, word in enumerate(words):
        prev_end = words[i - 1].end if i else -np.inf
        if word.start - prev_end < min_gap or not _covers(env, word.start):
            continue
        rises = [
            env.time(k)
            for k in range(env.index(word.start - window), env.index(word.start + window) + 1)
            if k >= quiet
            and (db[k : k + voiced] >= threshold_db).all()
            and (db[k - quiet : k] < threshold_db).all()
        ]
        ok = [t for t in rises if prev_end <= t <= word.end - MIN_WORD_S]
        if not ok:
            continue
        onset = min(ok, key=lambda t: abs(t - word.start))
        if abs(onset - word.start) > SNAP_EPS_S:
            out[i] = word.model_copy(update={"start": round(onset, 3)})
    return out


def tail_end(
    last_end: float,
    end: float,
    next_start: float,
    env: Envelope,
    threshold_db: float,
    max_tail: float = MAX_TAIL_S,
    next_end: float = np.inf,
) -> float:
    """Clip end (source s) after the last word (ending `last_end`) for a current end `end`:
    a little (TAIL_PAD_S) into the first silence after the word, within `max_tail` of it and
    TAIL_GUARD_S short of the next word; with no silence in reach, the quietest point there.
    A transcript next word (`next_start`-`next_end`) starting inside the last word's
    continuing sound but running past where it falls silent (no audible onset of its own
    before that: DTW misplaced it) is replaced by the next audible onset after the fall; a
    next word lying wholly inside that sound is audible, so the tail stops at its start (a
    whole word is never swallowed). Never earlier than `end`."""
    if _covers(env, last_end):
        fall = _quiet_from(env, last_end, last_end + max_tail, threshold_db)
        if fall is not None and next_start < fall < next_end:
            next_start = _rise_from(env, fall, threshold_db)
    hi = min(last_end + max_tail, next_start - TAIL_GUARD_S)
    if hi <= end or not _covers(env, last_end):
        return end
    quiet = _quiet_from(env, last_end, hi, threshold_db)
    if quiet is not None:
        return max(end, min(quiet + TAIL_PAD_S, hi))
    lo_k, hi_k = env.index(last_end), env.index(hi)
    k = lo_k + int(np.argmin(env.db[lo_k : hi_k + 1]))
    return max(end, min(env.time(k), hi))
