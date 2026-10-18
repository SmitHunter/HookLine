import json
import subprocess
from pathlib import Path

import pytest

from hookline.assets import AssetError
from hookline.pipeline import chat
from hookline.pipeline.chat import (
    ChatError,
    chat_keywords,
    chat_rate,
    download_chat,
    try_fetch_chat,
    twitch_vod_id,
)

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def raise_(exc):
    def f(*a, **k):
        raise exc

    return f


def test_twitch_vod_id():
    assert twitch_vod_id("https://www.twitch.tv/videos/2280112233?t=1h") == "2280112233"
    assert twitch_vod_id("https://youtube.com/watch?v=x") is None


def test_chat_rate_weights_emotes():
    r = chat_rate(load("twitch_chat_sample.json"), duration=10)
    assert len(r) == 10 and r[3] == pytest.approx(2.5)


def test_chat_rate_ignores_out_of_range_and_ceils():
    r = chat_rate(load("twitch_chat_sample.json"), duration=9.5)
    assert len(r) == 10
    assert r[0] == 2.0  # the -1.0 comment is dropped
    assert r[9] == 1.0  # 9.9 is >= duration, 9.3 kept


def test_try_fetch_chat_non_twitch_returns_none(tmp_path):
    assert try_fetch_chat("https://youtube.com/watch?v=x", tmp_path, log=print) is None


def test_try_fetch_chat_failure_warns_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ensure_twitch_downloader", raise_(AssetError("offline")))
    logs = []
    assert try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, logs.append) is None
    assert any("chat unavailable" in m for m in logs)


def test_try_fetch_chat_cached(tmp_path, monkeypatch):
    (tmp_path / "chat.json").write_text("{}")
    monkeypatch.setattr(chat, "ensure_twitch_downloader", raise_(AssertionError("no")))
    assert (
        try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, print) == tmp_path / "chat.json"
    )


def test_try_fetch_chat_success(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ensure_twitch_downloader", lambda: Path("x.exe"))

    def fake_dl(vod_id, dst, exe):
        dst.write_text("{}")
        return dst

    monkeypatch.setattr(chat, "download_chat", fake_dl)
    assert (
        try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, print) == tmp_path / "chat.json"
    )


def test_download_chat_nonzero_exit(tmp_path, monkeypatch):
    def fake_run(cmd, **k):
        return subprocess.CompletedProcess(cmd, 1, "", "VOD not found")

    monkeypatch.setattr(chat.subprocess, "run", fake_run)
    with pytest.raises(ChatError, match="VOD not found"):
        download_chat("1", tmp_path / "chat.json", Path("x.exe"))


def test_download_chat_success_replaces(tmp_path, monkeypatch):
    def fake_run(cmd, **k):
        assert cmd[1:4] == ["chatdownload", "--id", "1"]
        Path(cmd[cmd.index("-o") + 1]).write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(chat.subprocess, "run", fake_run)
    dst = tmp_path / "chat.json"
    assert download_chat("1", dst, Path("x.exe")) == dst
    assert dst.exists() and not list(tmp_path.glob("*.part*"))


def test_try_fetch_chat_timeout_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ensure_twitch_downloader", lambda: Path("x.exe"))

    def boom(cmd, **k):
        assert k["timeout"] == chat.CHAT_TIMEOUT_S
        Path(cmd[cmd.index("-o") + 1]).write_text("partial")
        raise subprocess.TimeoutExpired(cmd, k["timeout"])

    monkeypatch.setattr(chat.subprocess, "run", boom)
    logs = []
    assert try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, logs.append) is None
    assert len([m for m in logs if "chat unavailable" in m]) == 1
    assert not list(tmp_path.glob("*.part*"))


def test_try_fetch_chat_exit0_no_file_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ensure_twitch_downloader", lambda: Path("x.exe"))
    monkeypatch.setattr(
        chat.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", "")
    )
    logs = []
    assert try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, logs.append) is None
    assert len([m for m in logs if "chat unavailable" in m]) == 1


def test_chat_rate_tolerates_malformed():
    data = {
        "comments": [
            "junk",
            None,
            {"content_offset_seconds": "3"},
            {"content_offset_seconds": None},
            {"content_offset_seconds": 1.0, "message": None},
            {"content_offset_seconds": 2.0, "message": {"fragments": ["x", {"emoticon": {}}]}},
            {"content_offset_seconds": 2.5, "message": {"fragments": "bad"}},
        ]
    }
    assert chat_rate(data, 4) == [0.0, 1.0, 2.5, 0.0]


def test_chat_keywords_on_fixture():
    k = chat_keywords(load("twitch_chat_sample.json"), duration=10)
    assert k[3] == {"hype": 1}
    assert k[5] == {"laugh": 2}  # one per message, not per token
    assert 0 not in k and 4 not in k


def _msgs(*items):
    return {
        "comments": [
            {"content_offset_seconds": t, "message": {"body": body, "fragments": []}}
            for t, body in items
        ]
    }


def test_chat_keywords_kinds_and_case_rules():
    k = chat_keywords(
        _msgs(
            (0.1, "LMAO that was great"),
            (0.2, "W"),
            (0.3, "w in chat"),  # lowercase w is not the emote
            (1.1, "F"),
            (1.2, "L"),
            (1.3, "lets go!!"),
            (2.1, "clip it"),
            (2.2, "!clip"),
            (2.3, "CLIP THAT"),
            (3.1, "WTF"),
            (3.2, "monkaS"),
            (3.3, "D:"),
            (4.1, "😂😂"),
            (4.2, "hello there"),
        ),
        duration=6,
    )
    assert k[0] == {"laugh": 1, "hype": 1}
    assert k[1] == {"fail": 2, "hype": 1}
    assert k[2] == {"clip": 3}
    assert k[3] == {"shock": 3}
    assert k[4] == {"laugh": 1}
    assert 5 not in k


def test_chat_keywords_tolerates_malformed():
    bad = {"comments": [None, {"content_offset_seconds": "x"}, {"content_offset_seconds": 1}]}
    assert chat_keywords(bad, duration=5) == {}
    assert chat_keywords({}, duration=5) == {}


def _kinds_of(body, frags=None):
    msg = {"fragments": frags or []}
    if body is not None:
        msg["body"] = body
    return chat_keywords({"comments": [{"content_offset_seconds": 1, "message": msg}]}, 5)


def test_spam_message_counts_each_kind_once():
    assert _kinds_of("KEKW KEKW KEKW KEKW")[1] == {"laugh": 1}
    assert _kinds_of("!clip !clip !clip")[1] == {"clip": 1}
    assert _kinds_of("!clip it")[1] == {"clip": 1}


def test_emote_names_case_insensitive_but_single_letters_exact():
    assert _kinds_of("kekw")[1] == {"laugh": 1}
    assert _kinds_of("Lul")[1] == {"laugh": 1}
    assert _kinds_of("pog")[1] == {"hype": 1}
    assert _kinds_of("OmegaLUL")[1] == {"laugh": 1}
    assert 1 not in _kinds_of("w l f")


def test_stretched_words_and_apostrophes():
    assert _kinds_of("lmaooooo")[1] == {"laugh": 1}
    assert _kinds_of("lolllll")[1] == {"laugh": 1}
    assert _kinds_of("let's go")[1] == {"hype": 1}
    assert _kinds_of("let’s go!")[1] == {"hype": 1}


def test_clip_with_bangs_and_emote_inside_a_sentence():
    assert _kinds_of("!clip!")[1] == {"clip": 1}
    assert _kinds_of("clip!")[1] == {"clip": 1}
    assert 1 not in _kinds_of("that clip was long")
    assert _kinds_of("omg that was so KEKW honestly")[1] == {"laugh": 1}


def test_body_missing_uses_fragments():
    k = _kinds_of(None, [{"text": "nice", "emoticon": None}, {"text": "KEKW"}])
    assert k[1] == {"laugh": 1}


def test_stretched_emotes_match_their_name():
    assert _kinds_of("kekwww")[1] == {"laugh": 1}
    assert _kinds_of("KEKWWWW")[1] == {"laugh": 1}
    assert _kinds_of("letsgoooo")[1] == {"hype": 1}
    assert _kinds_of("pogggg")[1] == {"hype": 1}
