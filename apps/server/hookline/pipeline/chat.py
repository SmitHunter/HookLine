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


# Single-letter spam stays an exact uppercase token; emote names of 2+ chars match
# case-insensitively; words and phrases are lowercase.
_LETTERS = {"W": "hype", "L": "fail", "F": "fail"}
_EMOTES: dict[str, str] = {}
for _kind, _names in {
    "laugh": ("kekw", "lul", "lulw", "omegalul", "icant"),
    "hype": ("pog", "pogchamp", "poggers", "letsgo"),
    "fail": ("sadge",),
    "shock": ("wtf", "monkas", "d:"),
}.items():
    _EMOTES.update(dict.fromkeys(_names, _kind))
_STRETCHED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"lma+o+"), "laugh"),
    (re.compile(r"lo+l+"), "laugh"),
)
_PHRASES: dict[str, str] = {"lets go": "hype", "clip it": "clip", "clip that": "clip"}
_PUNCT = '.,?:;"()'
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_DOUBLE_RE = re.compile(r"(.)\1+")


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


def _message_kinds(body: str) -> set[str]:
    """The reaction kinds present in one chat message (each kind counted once)."""
    text = body.replace("\u2019", "'")
    kinds: set[str] = set()
    cleaned: list[str] = []
    for tok in text.split():
        stripped = tok.strip(_PUNCT)
        core = stripped.strip("!")
        if core in _LETTERS:
            kinds.add(_LETTERS[core])
        low = core.lower()
        if low == "clip" and "!" in stripped:
            kinds.add("clip")
        if tok.lower() == "d:":
            kinds.add("shock")
        word = _REPEAT_RE.sub(r"\1\1", low)
        # an emote as typed, or stretched ("kekwww", "letsgoooo"): repeats collapsed to two
        # letters, then to one
        emote = next((e for e in (low, word, _DOUBLE_RE.sub(r"\1", low)) if e in _EMOTES), None)
        if emote is not None:
            kinds.add(_EMOTES[emote])
        for pat, kind in _STRETCHED:
            if pat.fullmatch(word):
                kinds.add(kind)
        cleaned.append(low.replace("'", ""))
    for a, b in itertools.pairwise(cleaned):
        if (kind := _PHRASES.get(f"{a} {b}")) is not None:
            kinds.add(kind)
    if "\U0001f602" in text:
        kinds.add("laugh")
    return kinds


def chat_keywords(chat_json: dict, duration: float) -> dict[int, Counter]:
    """Per-second counts of reaction keywords (laugh / hype / fail / clip / shock) in chat;
    a kind counts at most once per message so one spammer cannot set a spike's character."""
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
            frags = msg.get("fragments")
            if not isinstance(frags, list):
                continue
            body = " ".join(
                f["text"] for f in frags if isinstance(f, dict) and isinstance(f.get("text"), str)
            )
        kinds = _message_kinds(body)
        if kinds:
            out.setdefault(int(t), Counter()).update(kinds)
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
