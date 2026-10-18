from hookline.models import Word

MIN_RANGE = 0.05
MIN_PIECE_S = 1.5  # a cut may not leave a kept piece shorter than this (choppy jump cuts)
MIN_WORD_S = 0.05  # captions drop shorter words; edge-clipped survivors are stretched to this
DEFAULT_FPS = 30.0  # for the one-frame sliver threshold when the clip's fps is unknown
MIN_SURVIVING = 0.5  # an edge-clipped word keeping less than this share of itself is dropped


def _word_free(
    silence: tuple[float, float], spans: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """`silence` minus every span in `spans` (sorted by start)."""
    pieces = []
    cursor, end = silence
    for a, b in spans:
        if b <= cursor or a >= end:
            continue
        if a > cursor:
            pieces.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def keep_ranges(
    segments: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    max_pause: float = 0.6,
    pad: float = 0.12,
    min_keep_ratio: float = 0.5,
    words: list[Word] | None = None,
    min_piece_s: float = MIN_PIECE_S,
) -> list[tuple[float, float]]:
    """Cut long silences out of each segment, keeping `pad` on each side.

    When a cut would leave a kept piece shorter than `min_piece_s`, the cut with the smaller
    removal next to that piece is dropped, repeated until every piece is long enough (or has
    no cuts left to drop).

    With `words`, each word's span (widened by `pad`) is first removed from the silences, so
    a cut never lands inside a transcribed word; only word-free pieces longer than
    `max_pause` are cut."""
    if words:
        spans = sorted((x.start - pad, x.end + pad) for x in words)
        silences = [p for sil in silences for p in _word_free(sil, spans)]
    kept: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        inside = sorted(
            (s, e) for s, e in silences if s >= seg_start and e <= seg_end and e - s > max_pause
        )
        cuts = [(s + pad, e - pad) for s, e in inside if e - pad > s + pad]
        cuts = _drop_short_pieces(seg_start, seg_end, cuts, min_piece_s)
        cursor = seg_start
        for cut_start, cut_end in cuts:
            kept.append((cursor, cut_start))
            cursor = cut_end
        kept.append((cursor, seg_end))
    kept = [(round(a, 3), round(b, 3)) for a, b in kept if b - a >= MIN_RANGE]
    original = sum(e - s for s, e in segments)
    if sum(e - s for s, e in kept) < min_keep_ratio * original:
        return list(segments)
    return kept


def _drop_short_pieces(
    seg_start: float, seg_end: float, cuts: list[tuple[float, float]], min_piece: float
) -> list[tuple[float, float]]:
    """Drop cuts until no kept piece between them is shorter than `min_piece`."""
    cuts = list(cuts)
    while cuts:
        bounds = [seg_start, *(x for c in cuts for x in c), seg_end]
        pieces = list(zip(bounds[::2], bounds[1::2], strict=True))
        short = next((i for i, (a, b) in enumerate(pieces) if b - a < min_piece - 1e-9), None)
        if short is None:
            break
        # piece i lies between cut i-1 (before) and cut i (after)
        around = [j for j in (short - 1, short) if 0 <= j < len(cuts)]
        del cuts[min(around, key=lambda j: cuts[j][1] - cuts[j][0])]
    return cuts


MERGE_GAP_S = 0.15


def snap_ranges(
    keep: list[tuple[float, float]],
    fps: float,
    merge_gap: float = MERGE_GAP_S,
    keep_gaps: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Merge each range into the previous one when it starts less than `merge_gap` after it
    ends (forward gaps only), unless the gap overlaps one of `keep_gaps` (explicit drops are
    intentional, however short), then snap every bound to the source frame grid (k / fps) and
    drop ranges that become empty. The output timeline is then exactly sum(end - start)."""
    merged: list[tuple[float, float]] = []
    for start, end in keep:
        gap = (merged[-1][1], start) if merged else (0.0, 0.0)
        explicit = any(min(gap[1], b) - max(gap[0], a) > 1e-9 for a, b in keep_gaps or [])
        if merged and 0 <= start - merged[-1][1] < merge_gap and not explicit:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    out = []
    for start, end in merged:
        a, b = round(start * fps), round(end * fps)
        if b > a:
            out.append((a / fps, b / fps))
    return out


def _survives(word: Word, judge_end: float | None, k_start: float, k_end: float) -> bool:
    """Whether at least half of the word's judged span (start to `judge_end`, the end dead
    air protected, when given and inside the word) lies in [k_start, k_end]."""
    end = word.end if judge_end is None else min(word.end, max(judge_end, word.start))
    if end - word.start <= 1e-9:
        return True
    kept = min(end, k_end) - max(word.start, k_start)
    return kept >= MIN_SURVIVING * (end - word.start) - 1e-9


def remap_words(
    words: list[Word],
    keep: list[tuple[float, float]],
    fps: float | None = None,
    judge_ends: list[float] | None = None,
) -> list[Word]:
    """Move word timings onto the output timeline of the kept ranges. A word a range edge
    clips to less than half of its duration, or to less than one frame (1 / fps; 30 fps when
    unknown), is left out of that range: it belongs to the speech the cut removed.

    `judge_ends` (one per word) are the ends dead-air cutting protected (the transcript end
    plus PROTECT_S): the half is judged on start..that end, so an audio-refined end inflated
    past the protected span cannot cost a kept word its caption."""
    frame = 1 / (fps or DEFAULT_FPS)
    out: list[Word] = []
    offset = 0.0
    for k_start, k_end in keep:
        for i, word in enumerate(words):
            start = max(word.start, k_start)
            end = min(word.end, k_end)
            if end <= start:
                continue
            clipped = word.start < k_start or word.end > k_end
            judge = judge_ends[i] if judge_ends is not None else None
            if clipped and (
                end - start < frame - 1e-9 or not _survives(word, judge, k_start, k_end)
            ):
                continue  # mostly cut away at the seam (a dropped line's last word): no flash
            if end - start < MIN_WORD_S and clipped:
                # A cut clipped this word to a sliver: keep it visible, inside the range.
                if word.end > k_end:
                    start = max(k_start, end - MIN_WORD_S)
                else:
                    end = min(k_end, start + MIN_WORD_S)
            out.append(
                word.model_copy(
                    update={
                        "start": round(offset + start - k_start, 3),
                        "end": round(offset + end - k_start, 3),
                    }
                )
            )
        offset += k_end - k_start
    return out
