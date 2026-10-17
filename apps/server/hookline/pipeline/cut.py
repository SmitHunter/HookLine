import json
from pathlib import Path

from hookline.fsutil import atomic_write_text, slugify
from hookline.media.ffmpeg import cut_segments
from hookline.models import ClipPick


def cut_clips(src: Path, picks: list[ClipPick], out_dir: Path, encoder: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, pick in enumerate(picks, 1):
        dst = out_dir / f"{i:02d}-{slugify(pick.title)}.mp4"
        cut_segments(src, pick.segments(), dst, encoder)
        sidecar = {**pick.model_dump(mode="json"), "post_caption": pick.hook_text}
        atomic_write_text(
            dst.with_suffix(".json"), json.dumps(sidecar, indent=2, ensure_ascii=False)
        )
        paths.append(dst)
    return paths
