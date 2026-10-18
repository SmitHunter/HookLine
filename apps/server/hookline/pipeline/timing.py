"""Word and clip-end timing refined against the audio's loudness envelope.

whisper.cpp's DTW gives one timestamp per token, so a single-token word's end is a guess:
`parse_whisper_cpp_json` sets it to the token time + 0.15 s (two thirds of the podcast's words
end exactly 0.15 s after they start). These helpers move ends to where the speech actually
falls silent, snap post-pause onsets to where it rises, and find a clean clip end."""

import math
from collections.abc import Sequence

import numpy as np

from hookline.media.audio import Envelope
from hookline.models import Word

__all__ = [
    "Envelope",
    "anchor_words",
    "last_voiced",
    "protected_words",
    "refine_word_ends",
    "snap_onsets",
    "snap_splices",
    "tail_end",
    "uncaptioned_speech",
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
REANCHOR_GAP_S = 0.25  # after a gap this long, an onset is searched for ...
REANCHOR_S = 0.7  # ... this far before the transcribed start (DTW runs late after pauses)
ONSET_CHAIN_S = 0.1  # ... through stops this short between the voiced runs of one word
MAX_TAIL_S = 0.3  # the clip ends at most this long after its last word falls silent ...
MIN_TAIL_S = 0.2  # ... and at least this long after (when the next word allows) ...
TAIL_GUARD_S = 0.05  # ... and never closer than this to the next word
QUIET_WORD_SHARE = 0.15  # a word with fewer voiced frames than this sits over silence
WORD_PAD_S = 0.05  # speech this close to a word belongs to it (uncaptioned-speech check)
UNCAPTIONED_S = 0.3  # a voiced stretch this long with no word is uncaptioned speech
SPLICE_S = 0.15  # a splice edge moves at most this far to a silent point ...
ONSET_PAD_S = 0.02  # ... and starts at least this far before the first kept word's onset
SPLICE_TIE_DB = 3.0  # points within this of the quietest are as good: the nearest wins
OWN_ONSET_S = 0.3  # a kept word DTW put in a dropped burst may own an onset this far on
SPLICE_MAX_S = 0.4  # a burst of the dropped side's sound around an edge is skipped within this ...
MAX_FRAGMENT_S = 0.35  # ... when at most this much of it lies on the kept side ...
SPLICE_LEAD_S = 0.12  # ... the range then starts this long before the next onset
UNCAPTIONED_BRIDGE_S = 0.25  # syllable gaps up to this long inside one uncaptioned stretch


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
    or the first word) moved to where its sound starts. After a gap over REANCHOR_GAP_S, to
    the start of the voiced run leading into the transcribed start (at least MIN_VOICED_S at
    or above `threshold_db`, dips under MIN_QUIET_S bridged, reaching that start; chained back
    over stops up to ONSET_CHAIN_S, so an earlier click or breath is never it) when it begins
    within REANCHOR_S before it (DTW runs late after pauses); otherwise, or without
    one, to the nearest audible onset (a frame at or above `threshold_db` after MIN_QUIET_S
    below it, staying up for MIN_VOICED_S) within `window` of it. Never before the previous word's end, nor leaving the word shorter than
    MIN_WORD_S."""
    quiet, voiced = _frames(env, MIN_QUIET_S), _frames(env, MIN_VOICED_S)
    db = env.db

    def rise(k: int) -> bool:
        return (
            k >= quiet
            and (db[k : k + voiced] >= threshold_db).all()
            and (db[k - quiet : k] < threshold_db).all()
        )

    out = list(words)
    for i, word in enumerate(words):
        prev_end = words[i - 1].end if i else -np.inf
        if word.start - prev_end < min_gap or not _covers(env, word.start):
            continue
        onset = None
        if word.start - prev_end > REANCHOR_GAP_S:
            # the first voiced run starting inside the window (dips under MIN_QUIET_S, e.g. a
            # ramped onset, bridged); one already sounding at its start is earlier speech
            lo = max(prev_end, word.start - REANCHOR_S)
            runs = _voiced_runs(env, lo, word.start + MIN_VOICED_S + env.hop, threshold_db)
            # the run leading into the word: it reaches the transcribed start (gaps under
            # MIN_QUIET_S bridged), so an earlier click or breath in the window is not it
            k = next(
                (
                    k
                    for k in range(len(runs) - 1, -1, -1)
                    if runs[k][0] <= word.start + env.hop / 2
                    and runs[k][1] >= word.start - MIN_QUIET_S
                ),
                None,
            )
            # ... chained back over the short stops inside the word (ONSET_CHAIN_S at most)
            while k is not None and k > 0 and runs[k][0] - runs[k - 1][1] <= ONSET_CHAIN_S:
                k -= 1
            onset = runs[k][0] if k is not None and runs[k][0] > lo + env.hop / 2 else None
            if onset is not None and not prev_end <= onset <= word.end - MIN_WORD_S:
                onset = None
        if onset is None:
            lo_k, hi_k = env.index(word.start - window), env.index(word.start + window)
            rises = [env.time(k) for k in range(lo_k, hi_k + 1) if rise(k)]
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
    MIN_TAIL_S after the word falls silent (the first silence after it), within `max_tail` of
    that and TAIL_GUARD_S short of the next word; with no silence in reach, at least
    MIN_TAIL_S after the word, the quietest point there otherwise. A transcript next word
    (`next_start`-`next_end`) starting inside the last word's continuing sound but running
    past where it falls silent (no audible onset of its own before that: DTW misplaced it)
    is replaced by the next audible onset after the fall; a next word lying wholly inside that
    sound is audible, so the tail stops at its start (a whole word is never swallowed). Never
    earlier than `end`."""
    if not _covers(env, last_end):
        return end
    fall = _quiet_from(env, last_end, last_end + max_tail, threshold_db)
    if fall is not None and next_start < fall < next_end:
        next_start = _rise_from(env, fall, threshold_db)
    spoken = max(last_end, fall) if fall is not None else last_end
    hi = min(spoken + max_tail, next_start - TAIL_GUARD_S)
    if hi <= end:
        return end
    if fall is not None:
        return max(end, min(fall + MIN_TAIL_S, hi))
    lo_k, hi_k = env.index(last_end + MIN_TAIL_S), env.index(hi)
    if lo_k >= hi_k:
        return max(end, hi)
    k = lo_k + int(np.argmin(env.db[lo_k : hi_k + 1]))
    return max(end, min(env.time(k), hi))


def _voiced_runs(
    env: Envelope, lo: float, hi: float, threshold_db: float, bridge_s: float = MIN_QUIET_S
) -> list[tuple[float, float]]:
    """Voiced runs (at or above `threshold_db`, dips under `bridge_s` bridged, at least
    MIN_VOICED_S long) inside [lo, hi), source times."""
    if not len(env.db) or hi <= lo:
        return []
    a_k, b_k = env.index(lo), env.index(hi)
    voiced = env.db[a_k : b_k + 1] >= threshold_db
    runs: list[tuple[int, int]] = []
    start = None
    for k, v in enumerate(voiced):
        if v and start is None:
            start = k
        elif not v and start is not None:
            runs.append((start, k))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))
    bridge = _frames(env, bridge_s)
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < bridge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    need = _frames(env, MIN_VOICED_S)
    return [
        (max(env.time(a_k + a), lo), min(env.time(a_k + b), hi)) for a, b in merged if b - a >= need
    ]


def last_voiced(env: Envelope, lo: float, hi: float, threshold_db: float) -> float | None:
    """End (source s) of the last voiced run in [lo, hi), or None (none, or outside the
    envelope)."""
    if not _covers(env, lo):
        return None
    runs = _voiced_runs(env, lo, hi, threshold_db)
    return runs[-1][1] if runs else None


def _voiced_share(env: Envelope, start: float, end: float, threshold_db: float) -> float:
    a, b = env.index(start), env.index(end)
    span = env.db[a : max(b, a + 1)]
    return float((span >= threshold_db).mean()) if len(span) else 0.0


def anchor_words(
    words: list[Word],
    env: Envelope,
    threshold_db: float,
    clip_end: float,
    ranges: Sequence[tuple[float, float]] | None = None,
) -> list[Word]:
    """Words whose transcript times sit over silence moved onto the speech:

    - the final words: the last voiced frame of the last kept range with words (before
      `clip_end`) bounds them; words starting after it, and words over silence right before
      the last spoken word (no voice of their own), share that word's span up to it, by text
      length, and no word ends past it;
    - any other word with under QUIET_WORD_SHARE of its frames voiced moves (keeping its
      length where it fits) onto the nearest voiced run between its neighbours (left alone
      when there is none).
    With `ranges` (the kept source ranges) a word only ever moves within its own range's
    sound. Words outside the envelope are untouched."""
    out = list(words)
    inside = [i for i, w in enumerate(out) if _covers(env, w.start) and w.start < clip_end]
    if not inside:
        return out

    def home(t: float) -> tuple[float, float]:
        """The kept range holding time `t` (the nearest one), or everything."""
        if not ranges:
            return -np.inf, np.inf
        return min(ranges, key=lambda r: max(r[0] - t, t - r[1], 0.0))

    a0, b0 = home(out[inside[-1]].start)
    first = [i for i in inside if home(out[i].start) == (a0, b0)]
    lo0 = max(out[first[0]].start - REANCHOR_S, a0)
    runs = _voiced_runs(env, lo0, min(clip_end, b0), threshold_db)
    fixed: set[int] = set()
    if runs:
        last = runs[-1][1]
        trail = [i for i in first if out[i].start >= last - MIN_WORD_S]
        lead = [i for i in first if out[i].start < last - MIN_WORD_S]
        if lead:
            p = lead[-1]
            # words over silence right before it (no voice of their own, none between them
            # and it: DTW gave "direct action." to "action." alone) share its span too
            before: list[int] = []
            for q in reversed(lead[:-1]):
                if _voiced_share(env, out[q].start, out[q].end, threshold_db) >= QUIET_WORD_SHARE:
                    break
                if _voiced_runs(env, out[q].end, out[p].start, threshold_db):
                    break
                before.insert(0, q)
            group = [*before, p, *trail]
            t0 = out[p].start
            if len(group) > 1:
                sizes = [max(len(out[i].text.strip()), 1) for i in group]
                span, t = last - t0, t0
                for i, size in zip(group, sizes, strict=True):
                    nxt = t + span * size / sum(sizes)
                    out[i] = out[i].model_copy(update={"start": round(t, 3), "end": round(nxt, 3)})
                    t = nxt
                fixed.update(group)
            for i in lead:
                if out[i].end > last + env.hop and last - out[i].start >= MIN_WORD_S:
                    out[i] = out[i].model_copy(update={"end": round(last, 3)})
    for i in inside:
        w = out[i]
        if i in fixed or _voiced_share(env, w.start, w.end, threshold_db) >= QUIET_WORD_SHARE:
            continue
        ra, rb = home(w.start)
        lo = max(out[i - 1].end if i else -np.inf, ra)
        hi = min(out[i + 1].start if i + 1 < len(out) else clip_end, rb)
        near = [
            (a, b)
            for a, b in _voiced_runs(env, max(lo, env.t0), hi, threshold_db)
            if b - a >= MIN_WORD_S
        ]
        if not near:
            continue  # nothing voiced between its neighbours: left alone
        a, b = min(near, key=lambda r: max(r[0] - w.end, w.start - r[1], 0.0))
        dur = w.end - w.start
        if b <= w.start:
            start, end = max(a, b - dur), b
        else:
            start, end = max(a, lo), min(b, max(a, lo) + dur)
        if end - start >= MIN_WORD_S:
            out[i] = w.model_copy(update={"start": round(start, 3), "end": round(end, 3)})
    return out


def uncaptioned_speech(
    words: list[Word],
    env: Envelope,
    threshold_db: float,
    ranges: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Voiced stretches (source s) inside the kept `ranges` that no word covers (words
    widened by WORD_PAD_S), at least UNCAPTIONED_S long: speech the transcript missed."""
    out = []
    spans = sorted((w.start - WORD_PAD_S, w.end + WORD_PAD_S) for w in words)
    for lo, hi in ranges:
        for a, b in _voiced_runs(env, lo, hi, threshold_db, UNCAPTIONED_BRIDGE_S):
            for p, q in _uncovered((a, b), spans):
                if q - p >= UNCAPTIONED_S:
                    out.append((round(p, 3), round(q, 3)))
    return out


def _uncovered(
    span: tuple[float, float], covers: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces, (cursor, end) = [], span
    for a, b in covers:
        if b <= cursor or a >= end:
            continue
        if a > cursor:
            pieces.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _dropped_burst(
    burst: tuple[float, float],
    words: list[Word],
    mine: list[int],
    theirs: list[int],
    end: float,
    start: float,
    onset: float = np.inf,
) -> bool:
    """Whether a voiced burst around a range start (`start`; the previous range ends at
    `end`) is the tail of the dropped side's sound: at most MAX_FRAGMENT_S of it past the
    start, overlapping a dropped word (one between the ranges), and no kept word of the range
    (`theirs`) with most of its transcript span inside it -- that word is the burst -- unless
    that word has its own voiced onset after the burst (`onset`: the next one) within
    OWN_ONSET_S of its transcript end and before the next kept word starts (DTW put it in the
    dropped word's sound; it is moved there)."""
    a, b = burst
    if b - start > MAX_FRAGMENT_S:
        return False
    keep = set(mine) | set(theirs)
    dropped = [
        w
        for i, w in enumerate(words)
        if i not in keep and end - 1e-9 <= (w.start + w.end) / 2 < start + 1e-9
    ]
    if not any(min(w.end, b) > max(w.start, a) for w in dropped):
        return False
    for n, i in enumerate(theirs):
        w = words[i]
        inside = max(0.0, min(w.end, b) - max(w.start, a))
        if inside > (w.end - w.start) / 2:
            nxt = words[theirs[n + 1]].start if n + 1 < len(theirs) else np.inf
            if not (onset <= w.end + OWN_ONSET_S and onset < nxt - ONSET_PAD_S):
                return False
    return True


def snap_splices(
    keep: list[tuple[float, float]],
    words: list[Word],
    env: Envelope,
    threshold_db: float,
    fps: float,
    fixed: Sequence[float] = (),
    adjust: bool = False,
) -> list[tuple[float, float]] | tuple[list[tuple[float, float]], list[Word]]:
    """`keep` with every interior splice edge (a drop or a dead-air cut; not the clip's own
    start and end, nor edges within a frame of `fixed` times, e.g. a cold open's bounds) moved
    to a frame-grid point where the audio is silent (below `threshold_db`), within SPLICE_S of
    it: a range end into the first quiet run after its last kept word's sound, a range start
    into the last quiet run before its first kept word's sound (so a dropped word's tail or
    head never survives the splice); with no quiet point there, the quietest one. Never into
    a kept word (`words`: refined times), never crossing the neighbouring edge.

    An edge inside a short burst of the dropped side's sound (a word's tail running on past a
    range start, or a head starting before a range end: at most MAX_FRAGMENT_S of it on the
    kept side, all within SPLICE_MAX_S) moves past the whole burst; a kept word DTW placed
    inside that burst is misplaced and moves to the next audible onset (with `adjust`, the
    moved words are returned too: (keep, words))."""
    out = [list(r) for r in keep]
    words = list(words)
    if len(keep) < 2 or not len(env.db):
        return (list(keep), words) if adjust else list(keep)

    def level(t: float) -> float:
        return max(env.level(t - env.hop), env.level(t))

    def grid(lo: float, hi: float) -> list[float]:
        return [k / fps for k in range(math.ceil(lo * fps - 1e-9), math.floor(hi * fps + 1e-9) + 1)]

    def pick(cands: list[float], edge: float, first: bool) -> float | None:
        if not cands:
            return None
        quiet = [t for t in cands if level(t) < threshold_db]
        if quiet:
            runs: list[list[float]] = [[quiet[0]]]
            for t in quiet[1:]:
                if t - runs[-1][-1] <= 1.5 / fps:
                    runs[-1].append(t)
                else:
                    runs.append([t])
            cands = runs[0] if first else runs[-1]
        low = min(level(t) for t in cands)
        best = [t for t in cands if level(t) <= low + SPLICE_TIE_DB]
        return min(best, key=lambda t: abs(t - edge))

    def held(t: float) -> bool:
        return any(abs(t - f) < 1.0 / fps for f in fixed)

    def burst(t: float) -> tuple[float, float] | None:
        """The voiced run around `t` (strictly inside it), within SPLICE_MAX_S of it."""
        runs = _voiced_runs(env, t - SPLICE_MAX_S, t + SPLICE_MAX_S, threshold_db)
        return next(((a, b) for a, b in runs if a < t - env.hop / 2 < t + env.hop / 2 < b), None)

    for k in range(len(out) - 1):
        (a0, e), (s, b1) = out[k], out[k + 1]
        mine = [i for i, w in enumerate(words) if a0 <= (w.start + w.end) / 2 < e]
        theirs = [i for i, w in enumerate(words) if s <= (w.start + w.end) / 2 < b1]
        lower = max([e - SPLICE_S, *(words[i].end for i in mine)])
        if not held(e):
            head = burst(e)
            new = None
            if head and head[0] >= lower and e - head[0] <= MAX_FRAGMENT_S:
                # a dropped word's head already sounds before the range end: end before it
                new = pick(grid(max(lower, head[0] - SPLICE_S), head[0]), head[0], first=False)
                for i in mine:
                    if words[i].end > (new if new is not None else e):
                        words[i] = words[i].model_copy(update={"end": round(max(
                            words[i].start + MIN_WORD_S, new or e), 3)})  # fmt: skip
            if new is None:
                new = pick(grid(lower, e + SPLICE_S), e, first=True)
            if new is not None and a0 < new < s:
                out[k][1] = new
        if not held(s):
            tail = burst(s)
            new = None
            after = _rise_from(env, tail[1], threshold_db) if tail else np.inf
            if tail and _dropped_burst(tail, words, mine, theirs, e, s, after):
                # a dropped word's tail runs on past the range start: start after it
                onset = _rise_from(env, tail[1], threshold_db)
                hi = min(onset - ONSET_PAD_S, tail[1] + SPLICE_MAX_S)
                lead = max(tail[1], min(onset, hi) - SPLICE_LEAD_S)
                new = pick(grid(lead, hi), lead, first=False)
                if new is not None:
                    for n, i in enumerate(theirs):
                        w = words[i]
                        if w.start < new and onset < np.inf:
                            later = words[theirs[n + 1]].start if n + 1 < len(theirs) else b1
                            end = min(max(w.end, onset + (w.end - w.start)), later)
                            if end - onset >= MIN_WORD_S:
                                words[i] = w.model_copy(
                                    update={"start": round(onset, 3), "end": round(end, 3)}
                                )
            if new is None:
                upper = min([s + SPLICE_S, *(words[i].start - ONSET_PAD_S for i in theirs)])
                new = pick(grid(s - SPLICE_S, upper), s, first=False)
            if new is not None and out[k][1] < new < b1:
                out[k + 1][0] = new
    ranges = [(a, b) for a, b in out]
    return (ranges, words) if adjust else ranges
