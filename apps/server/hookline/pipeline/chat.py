import math
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from hookline.assets import AssetError, ensure_twitch_downloader

_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_EMOTE_WEIGHT = 0.5


class ChatError(Exception):
    pass


def twitch_vod_id(url: str) -> str | None:
    m = _VOD_RE.search(url)
    return m.group(1) if m else None


def download_chat(vod_id: str, dst: Path, exe: Path) -> Path:
    dst = Path(dst)
    tmp = dst.with_suffix(".part.json")
    try:
        proc = subprocess.run(
            [str(exe), "chatdownload", "--id", vod_id, "-o", str(tmp)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ChatError(f"chatdownload exit {proc.returncode}: {msg[-1] if msg else ''}")
        os.replace(tmp, dst)
    except ChatError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ChatError(f"chatdownload failed: {e}") from e
    return dst


def chat_rate(chat_json: dict, duration: float) -> list[float]:
    """Per-second chat activity: 1.0 per comment plus 0.5 per emote fragment."""
    out = [0.0] * math.ceil(duration)
    for c in chat_json.get("comments", []):
        t = c.get("content_offset_seconds")
        if t is None or not (0 <= t < duration):
            continue
        frags = (c.get("message") or {}).get("fragments") or []
        emotes = sum(1 for f in frags if f.get("emoticon") is not None)
        out[int(t)] += 1.0 + _EMOTE_WEIGHT * emotes
    return out


def try_fetch_chat(source: str, job_dir: Path, log: Callable[[str], None]) -> Path | None:
    vod_id = twitch_vod_id(source)
    if vod_id is None:
        return None
    dst = Path(job_dir) / "chat.json"
    if dst.exists():
        return dst
    try:
        exe = ensure_twitch_downloader()
        return download_chat(vod_id, dst, exe)
    except (AssetError, ChatError, OSError) as e:
        log(f"warning: chat unavailable ({e}) — continuing without chat signals")
        return None
