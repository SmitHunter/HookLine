from hookline.models import Word

MIN_RANGE = 0.05


def keep_ranges(
    segments: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    max_pause: float = 0.4,
    pad: float = 0.08,
    min_keep_ratio: float = 0.5,
) -> list[tuple[float, float]]:
    """Cut long silences out of each segment, keeping `pad` on each side."""
    kept: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        cursor = seg_start
        inside = sorted(
            (s, e) for s, e in silences if s >= seg_start and e <= seg_end and e - s > max_pause
        )
        for s, e in inside:
            cut_start, cut_end = s + pad, e - pad
            if cut_end <= cut_start:
                continue
            kept.append((cursor, cut_start))
            cursor = cut_end
        kept.append((cursor, seg_end))
    kept = [(round(a, 3), round(b, 3)) for a, b in kept if b - a >= MIN_RANGE]
    original = sum(e - s for s, e in segments)
    if sum(e - s for s, e in kept) < min_keep_ratio * original:
        return list(segments)
    return kept


MERGE_GAP_S = 0.15


def snap_ranges(
    keep: list[tuple[float, float]], fps: float, merge_gap: float = MERGE_GAP_S
) -> list[tuple[float, float]]:
    """Merge each range into the previous one when it starts less than `merge_gap` after it
    ends (forward gaps only), then snap every bound to the source frame grid (k / fps) and
    drop ranges that become empty. The output timeline is then exactly sum(end - start)."""
    merged: list[tuple[float, float]] = []
    for start, end in keep:
        if merged and 0 <= start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    out = []
    for start, end in merged:
        a, b = round(start * fps), round(end * fps)
        if b > a:
            out.append((a / fps, b / fps))
    return out


def remap_words(words: list[Word], keep: list[tuple[float, float]]) -> list[Word]:
    """Move word timings onto the output timeline of the kept ranges."""
    out: list[Word] = []
    offset = 0.0
    for k_start, k_end in keep:
        for word in words:
            start = max(word.start, k_start)
            end = min(word.end, k_end)
            if end <= start:
                continue
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
