import itertools
import math
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from hookline.assets import AssetError, ensure_twitch_downloader

_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_EMOTE_WEIGHT = 0.5
CHAT_TIMEOUT_S = 600


# Emote names / single-letter spam are exact, case-sensitive tokens; words and phrases are
# matched case-insensitively.
_EMOTES: dict[str, str] = {}
for _kind, _names in {
    "laugh": ("KEKW", "LUL", "LULW", "OMEGALUL", "ICANT"),
    "hype": ("Pog", "PogChamp", "POGGERS", "W", "LETSGO"),
    "fail": ("L", "F", "Sadge"),
    "shock": ("WTF", "monkaS", "D:"),
}.items():
    _EMOTES.update(dict.fromkeys(_names, _kind))
_WORDS: dict[str, str] = {
    "lmao": "laugh",
    "lol": "laugh",
    "letsgo": "hype",
    "wtf": "shock",
    "!clip": "clip",
}
_PHRASES: dict[str, str] = {
    "lets go": "hype",
    "clip it": "clip",
    "clip that": "clip",
}
_SUBSTRINGS: dict[str, str] = {"😂": "laugh"}
_TOKEN_STRIP = ".,!?:;\"'()"


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
            timeout=CHAT_TIMEOUT_S,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ChatError(f"chatdownload exit {proc.returncode}: {msg[-1] if msg else ''}")
        os.replace(tmp, dst)
    except ChatError:
        tmp.unlink(missing_ok=True)
        raise
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise ChatError(f"chatdownload timed out after {CHAT_TIMEOUT_S}s") from e
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ChatError(f"chatdownload failed: {e}") from e
    return dst


def chat_rate(chat_json: dict, duration: float) -> list[float]:
    """Per-second chat activity: 1.0 per comment plus 0.5 per emote fragment."""
    out = [0.0] * math.ceil(duration)
    for c in chat_json.get("comments", []):
        if not isinstance(c, dict):
            continue
        t = c.get("content_offset_seconds")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not (0 <= t < duration):
            continue
        msg = c.get("message")
        frags = msg.get("fragments") if isinstance(msg, dict) else None
        if not isinstance(frags, list):
            frags = []
        emotes = sum(1 for f in frags if isinstance(f, dict) and f.get("emoticon") is not None)
        out[int(t)] += 1.0 + _EMOTE_WEIGHT * emotes
    return out


def _message_kinds(body: str) -> list[str]:
    """Reaction kind of each keyword hit in one chat message body."""
    hits: list[str] = []
    toks = body.split()
    for tok in toks:
        exact = _EMOTES.get(tok) or _EMOTES.get(tok.strip(_TOKEN_STRIP))
        if exact:
            hits.append(exact)
            continue
        word = tok.lower().strip(_TOKEN_STRIP.replace("!", ""))
        kind = _WORDS.get(word) or _WORDS.get(tok.lower().strip(_TOKEN_STRIP))
        if kind:
            hits.append(kind)
    lowered = [t.lower().strip(_TOKEN_STRIP) for t in toks]
    for a, b in itertools.pairwise(lowered):
        kind = _PHRASES.get(f"{a} {b}")
        if kind:
            hits.append(kind)
    for sub, kind in _SUBSTRINGS.items():
        if sub in body:
            hits.append(kind)
    return hits


def chat_keywords(chat_json: dict, duration: float) -> dict[int, Counter]:
    """Per-second counts of reaction keywords (laugh / hype / fail / clip / shock) in chat."""
    out: dict[int, Counter] = {}
    for c in chat_json.get("comments", []):
        if not isinstance(c, dict):
            continue
        t = c.get("content_offset_seconds")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not (0 <= t < duration):
            continue
        msg = c.get("message")
        if not isinstance(msg, dict):
            continue
        body = msg.get("body")
        if not isinstance(body, str):
            continue
        hits = _message_kinds(body)
        if hits:
            out.setdefault(int(t), Counter()).update(hits)
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
