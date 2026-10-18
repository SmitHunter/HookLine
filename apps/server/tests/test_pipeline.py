import json
from pathlib import Path

import pytest

from hookline.media.ffmpeg import probe_duration, probe_size
from hookline.models import Segment, Transcript, Word
from hookline.pipeline import package as package_mod
from hookline.pipeline import run as run_mod
from hookline.pipeline.ingest import job_id_for
from hookline.pipeline.run import NoClipsSelectedError, run_job
from tests.test_select import raw


@pytest.fixture(autouse=True)
def _no_packaging(monkeypatch):
    """The packaging pass is its own LLM call; tests here queue only selection payloads, so
    it is a pass-through unless a test restores it (`real_packaging`)."""
    monkeypatch.setattr(run_mod, "package_clips", lambda picks, *a, **k: picks)


@pytest.fixture
def real_packaging(monkeypatch):
    monkeypatch.setattr(run_mod, "package_clips", package_mod.package_clips)


class FakeTranscriber:
    """Contiguous 3.0s sentences across `duration`; counts calls."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.calls = 0

    def transcribe(self, wav: Path, on_progress=None) -> Transcript:
        self.calls += 1
        assert wav.name == "audio.wav" and wav.exists()
        segments = []
        for k in range(int(self.duration // 3)):
            texts = [f"line{k}", "says", "something", "here."]
            words = [
                Word(start=3.0 * k + j * 0.75, end=3.0 * k + (j + 1) * 0.75, text=t)
                for j, t in enumerate(texts)
            ]
            segments.append(
                Segment(start=words[0].start, end=words[-1].end, text=" ".join(texts), words=words)
            )
            if on_progress:
                on_progress((k + 1) / int(self.duration // 3))
        return Transcript(language="en", duration=self.duration, segments=segments)


def make_payload_for(duration: float, n: int, length: float) -> dict:
    """n non-overlapping clips, starts on sentence starts, `length` a multiple of 3."""
    gap = (duration - 2 * 30) / max(n, 1)
    clips = []
    for i in range(n):
        start = float(30 + 3 * round(i * gap / 3))
        clips.append(raw(start, start + length, total=50 - i))
    return {"clips": clips}


@pytest.fixture
def settings_tmp(settings, tmp_path):
    return settings.model_copy(
        update={"workspace_dir": tmp_path / "ws", "output_dir": tmp_path / "out"}
    )


def test_run_job_end_to_end_with_fakes(fixture_video, fake_llm, settings_tmp, monkeypatch):
    from hookline.pipeline import run as run_mod

    rms_calls: list[Path] = []
    real_rms = run_mod.rms_per_second
    monkeypatch.setattr(run_mod, "rms_per_second", lambda w: rms_calls.append(w) or real_rms(w))
    payload = make_payload_for(150, 2, 21)
    for c in payload["clips"]:
        c["post_caption"] = "📰😂 He ran the whole kitchen alone. Could you?"
    fake_llm.queue(payload)
    s = settings_tmp.model_copy(update={"clips": 2})
    logs: list[str] = []
    t = FakeTranscriber(duration=150)
    clips = run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(clips) == 2 and all(c.exists() and c.name.isascii() for c in clips)
    assert all(probe_size(c) == (1080, 1920) for c in clips)
    assert all(probe_duration(c) == pytest.approx(21, abs=1.0) for c in clips)
    side = [json.loads(c.with_suffix(".json").read_text(encoding="utf-8")) for c in clips]
    assert all(d["post_caption"] == "📰😂 He ran the whole kitchen alone. Could you?" for d in side)
    # emoji are written as real UTF-8 (not ASCII escapes, not cp1252 mojibake) and read back
    job = s.workspace_dir / job_id_for(str(fixture_video))
    for f in [*(c.with_suffix(".json") for c in clips), job / "picks.json"]:
        assert "📰😂".encode() in f.read_bytes(), f.name
    assert all(d["hashtags"] and d["search_title"] and d["verdict"] == "post" for d in side)
    assert all(d["first_words"] and d["hook_type"] and d["comment_risk"] for d in side)
    assert all(d["layout"] and "keep_ranges" in d and "removed_s" in d for d in side)
    assert any(m.startswith("clip 01: layouts=[") and "removed=" in m for m in logs)
    signals = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert len(signals["loudness"]) >= 149 and signals["chat"] is None
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(rms_calls) == 1  # signals.json reused on the second run


def test_run_job_names_and_sidecar(fixture_video, fake_llm, settings_tmp):
    payload = make_payload_for(150, 1, 21)
    payload["clips"][0]["title"] = "Sample clip"
    payload["clips"][0]["post_caption"] = ""  # empty: the sidecar falls back to hook_text
    fake_llm.queue(payload)
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert [p.name for p in clips] == ["01-sample-clip.mp4"]
    data = json.loads(clips[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert data["title"] == "Sample clip" and data["post_caption"] == data["hook_text"]
    assert data["category"] == "food" and data["hashtags"] == [
        "linecook",
        "kitchenlife",
        "chefstories",
    ]
    assert data["layout"] == []


def test_rerun_uses_cache(fixture_video, fake_llm, settings_tmp):
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2})
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    run_job(str(fixture_video), s, fake_llm, t)
    assert t.calls == 1 and len(fake_llm.calls) == 1


def test_partial_artifact_is_not_trusted(fixture_video, fake_llm, settings_tmp):
    job = settings_tmp.workspace_dir / job_id_for(str(fixture_video))
    job.mkdir(parents=True)
    (job / "transcript.json.tmp").write_text("{trunc", encoding="utf-8")
    (job / "audio.wav.tmp.wav").write_bytes(b"junk")
    fake_llm.queue(make_payload_for(150, 2, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), settings_tmp.model_copy(update={"clips": 2}), fake_llm, t)
    assert t.calls == 1


def test_job_id_stable_for_url():
    url = "https://www.twitch.tv/videos/1"
    assert job_id_for(url) == job_id_for(url) and len(job_id_for(url)) == 10


def test_job_id_changes_with_file_content(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    a = job_id_for(str(f))
    f.write_bytes(b"xx")
    assert job_id_for(str(f)) != a


def test_corrupt_cached_artifacts_are_redone(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 2})
    fake_llm.queue(make_payload_for(150, 2, 21), make_payload_for(150, 2, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    (job / "transcript.json").write_text("{trunc", encoding="utf-8")
    (job / "picks.json").write_text("[{bad", encoding="utf-8")
    logs: list[str] = []
    clips = run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert t.calls == 2 and len(clips) == 2
    assert any("corrupt" in m for m in logs)


def test_cached_picks_hint(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 2})
    fake_llm.queue(make_payload_for(150, 2, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert any("delete it to re-select" in m for m in logs)


def test_run_job_writes_select_raw(fixture_video, fake_llm, settings_tmp):
    payload = make_payload_for(150, 2, 21)
    fake_llm.queue(payload, {"clips": []})
    s = settings_tmp.model_copy(update={"clips": 2})
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    job = settings_tmp.workspace_dir / job_id_for(str(fixture_video))
    assert json.loads((job / "select_raw.json").read_text(encoding="utf-8"))[0] == payload


def test_empty_selection_writes_no_picks_and_raises(fixture_video, fake_llm, settings_tmp):
    fake_llm.queue_forever({"clips": []})
    with pytest.raises(NoClipsSelectedError, match="select_raw.json"):
        run_job(str(fixture_video), settings_tmp, fake_llm, FakeTranscriber(duration=150))
    job = settings_tmp.workspace_dir / job_id_for(str(fixture_video))
    assert not (job / "picks.json").exists() and (job / "select_raw.json").exists()


def test_picks_cache_key_mismatch_reselects(fixture_video, fake_llm, settings_tmp):
    fast = {"clips": 1, "vertical": False, "captions": False}
    s = settings_tmp.model_copy(update=fast)
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    saved = json.loads((job / "picks.json").read_text(encoding="utf-8"))
    assert saved["key"] == {
        "schema": 7,
        "clips": 1,
        "select_model": s.select_model,
        "min_clip_s": s.min_clip_s,
        "max_clip_s": s.max_clip_s,
        "transcript": {"backend": "FakeTranscriber", "model": None, "dtw": None},
    }
    logs: list[str] = []
    run_job(str(fixture_video), s.model_copy(update={"max_clip_s": 45}), fake_llm, t, logs.append)
    assert len(fake_llm.calls) == 2
    assert any("re-select" in m and "max_clip_s" in m for m in logs)


def test_old_list_picks_json_reselects(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    saved = json.loads((job / "picks.json").read_text(encoding="utf-8"))
    (job / "picks.json").write_text(json.dumps(saved["picks"]), encoding="utf-8")
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(fake_llm.calls) == 2 and any("old format" in m for m in logs)


def test_failed_clip_is_skipped_and_logged(fixture_video, fake_llm, settings_tmp, monkeypatch):
    from hookline.media.ffmpeg import FFmpegError
    from hookline.pipeline import run as run_mod

    real = run_mod.render_clip
    calls = [0]

    def flaky(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            raise FFmpegError("ffmpeg exited with code 1", "boom line")
        return real(*a, **k)

    monkeypatch.setattr(run_mod, "render_clip", flaky)
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2, "vertical": False, "captions": False})
    logs: list[str] = []
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150), logs.append)
    assert [c.name[:2] for c in clips] == ["02"]
    assert any(m.startswith("warning: clip 01 failed: ffmpeg exited with code 1") for m in logs)


def test_all_clips_failing_raises(fixture_video, fake_llm, settings_tmp, monkeypatch):
    from hookline.media.ffmpeg import FFmpegError
    from hookline.pipeline import run as run_mod

    def boom(*a, **k):
        raise FFmpegError("ffmpeg exited with code 1")

    monkeypatch.setattr(run_mod, "render_clip", boom)
    fake_llm.queue(make_payload_for(150, 1, 21))
    s = settings_tmp.model_copy(update={"clips": 1})
    with pytest.raises(FFmpegError, match="no clips rendered"):
        run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))


@pytest.mark.parametrize(("timing", "protect"), [("segment", False), ("dtw", True)])
def test_word_protection_follows_transcript_timing(
    fixture_video, fake_llm, settings_tmp, monkeypatch, timing, protect
):
    from hookline.pipeline import run as run_mod

    seen = []
    real = run_mod.render_clip

    def spy(*a, **k):
        seen.append(k.get("protect_words"))
        return real(*a, **k)

    class Timed(FakeTranscriber):
        def transcribe(self, wav, on_progress=None):
            return super().transcribe(wav, on_progress).model_copy(update={"timing": timing})

    monkeypatch.setattr(run_mod, "render_clip", spy)
    fake_llm.queue(make_payload_for(150, 1, 21))
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    run_job(str(fixture_video), s, fake_llm, Timed(duration=150))
    assert seen == [protect]


class KeyedTranscriber(FakeTranscriber):
    def __init__(self, duration: float, model: str, dtw: str | None) -> None:
        super().__init__(duration)
        self.cache_key = {"backend": "whisper-cpp", "model": model, "dtw": dtw}


def test_transcript_cache_is_keyed_by_transcriber(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    a = KeyedTranscriber(150, "ggml-large-v3-turbo.bin", "large.v3.turbo")
    run_job(str(fixture_video), s, fake_llm, a)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    saved = json.loads((job / "transcript.json").read_text(encoding="utf-8"))
    assert saved["key"] == a.cache_key and saved["transcript"]["segments"]
    run_job(str(fixture_video), s, fake_llm, a)
    assert a.calls == 1 and len(fake_llm.calls) == 1  # same key: both caches reused

    b = KeyedTranscriber(150, "custom.bin", None)
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, b, logs.append)
    assert b.calls == 1
    assert any("transcriber changed (dtw, model)" in m and "re-transcribing" in m for m in logs)
    assert len(fake_llm.calls) == 2  # picks depend on the transcript key too
    assert any("re-select" in m and "transcript" in m for m in logs)


def test_old_format_transcript_is_retranscribed(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    tpath = job / "transcript.json"
    saved = json.loads(tpath.read_text(encoding="utf-8"))
    tpath.write_text(json.dumps(saved["transcript"]), encoding="utf-8")
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert t.calls == 2 and any("old format" in m and "re-transcribing" in m for m in logs)


@pytest.mark.parametrize("exc", [ValueError("no layouts"), OSError("disk full")])
def test_clip_value_and_os_errors_are_skipped(
    fixture_video, fake_llm, settings_tmp, monkeypatch, exc
):
    from hookline.pipeline import run as run_mod

    real = run_mod.render_clip
    calls = [0]

    def flaky(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            raise exc
        return real(*a, **k)

    monkeypatch.setattr(run_mod, "render_clip", flaky)
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2, "vertical": False, "captions": False})
    logs: list[str] = []
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150), logs.append)
    assert [c.name[:2] for c in clips] == ["02"]
    assert any(m.startswith(f"warning: clip 01 failed: {exc}") for m in logs)


def test_face_detector_failure_falls_back(settings_tmp, monkeypatch):
    from hookline.pipeline import run as run_mod

    class Cv2Error(Exception):
        pass

    def broken(model):
        raise Cv2Error("cv2.error: bad model")

    monkeypatch.setattr(run_mod, "ensure_yunet", lambda: Path("yunet.onnx"))
    monkeypatch.setattr(run_mod, "FaceDetector", broken)
    logs: list[str] = []
    assert run_mod._face_detector(settings_tmp, logs.append) is None
    assert any("face detector unavailable" in m and "FIT_BLUR" in m for m in logs)


def test_encoder_is_resolved_once_per_job(fixture_video, fake_llm, settings_tmp, monkeypatch):
    from hookline.pipeline import run as run_mod

    picks = [0]
    seen = []
    real = run_mod.render_clip

    def counting():
        picks[0] += 1
        return "libx264"

    def spy(*a, **k):
        seen.append(k.get("encoder"))
        return real(*a, **k)

    monkeypatch.setattr(run_mod, "pick_encoder", counting)
    monkeypatch.setattr(run_mod, "render_clip", spy)
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2, "vertical": False, "captions": False})
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert picks[0] == 1 and seen == ["libx264", "libx264"]


def _old_format(fixture_video, fake_llm, s, timing: str):
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    tpath = s.workspace_dir / job_id_for(str(fixture_video)) / "transcript.json"
    saved = json.loads(tpath.read_text(encoding="utf-8"))
    old = saved["transcript"]
    old["timing"] = timing
    for w in old["segments"][0]["words"][:3]:  # collapsed DTW starts that need spreading
        w["start"], w["end"] = 0.0, 0.05
    tpath.write_text(json.dumps(old), encoding="utf-8")
    return tpath


def test_old_format_dtw_transcript_is_migrated(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    tpath = _old_format(fixture_video, fake_llm, s, "dtw")
    b = KeyedTranscriber(150, "ggml-large-v3-turbo.bin", "large.v3.turbo")
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, b, logs.append)
    assert b.calls == 0
    assert any("transcript: migrated cached DTW transcript" in m for m in logs)
    saved = json.loads(tpath.read_text(encoding="utf-8"))
    assert saved["key"] == b.cache_key and saved["transcript"]["timing"] == "dtw"
    starts = [w["start"] for w in saved["transcript"]["segments"][0]["words"][:3]]
    assert len(set(starts)) == 3  # cleaning was applied during migration
    assert not tpath.with_name("transcript.json.tmp").exists()


def test_old_format_segment_transcript_with_dtw_backend_retranscribes(
    fixture_video, fake_llm, settings_tmp
):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    _old_format(fixture_video, fake_llm, s, "segment")
    b = KeyedTranscriber(150, "ggml-large-v3-turbo.bin", "large.v3.turbo")
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, b, logs.append)
    assert b.calls == 1 and any("old format" in m for m in logs)


def test_picks_from_before_selection_v2_reselect(fixture_video, fake_llm, settings_tmp):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    fake_llm.queue(make_payload_for(150, 1, 21), make_payload_for(150, 1, 21))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    saved = json.loads((job / "picks.json").read_text(encoding="utf-8"))
    del saved["key"]["schema"]
    (job / "picks.json").write_text(json.dumps(saved), encoding="utf-8")
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(fake_llm.calls) == 2 and any("re-select" in m and "schema" in m for m in logs)


def test_all_skip_selection_raises_and_keeps_select_raw(fixture_video, fake_llm, settings_tmp):
    payload = make_payload_for(150, 5, 21)
    for c in payload["clips"]:
        c["verdict"] = "skip"
    fake_llm.queue(payload)
    with pytest.raises(NoClipsSelectedError, match="rated every candidate as skip"):
        run_job(str(fixture_video), settings_tmp, fake_llm, FakeTranscriber(duration=150))
    job = settings_tmp.workspace_dir / job_id_for(str(fixture_video))
    assert not (job / "picks.json").exists()
    assert json.loads((job / "select_raw.json").read_text(encoding="utf-8")) == [payload]


def _job_dir(settings, video):
    d = settings.workspace_dir / job_id_for(str(video))
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_signals_gain_chat_kinds_and_old_cache_is_recomputed(fixture_video, fake_llm, settings_tmp):
    chat = {
        "comments": [
            {"content_offset_seconds": 5.0, "message": {"body": "KEKW lol", "fragments": []}}
        ]
    }
    job = _job_dir(settings_tmp, fixture_video)
    (job / "chat.json").write_text(json.dumps(chat), encoding="utf-8")
    s = settings_tmp.model_copy(update={"clips": 1, "chat": True})
    fake_llm.queue(make_payload_for(150, 1, 21))
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    sig = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert sig["chat_kinds"] == {"5": {"laugh": 1}}
    # an old cache (chat but no chat_kinds) is recomputed while chat.json exists
    del sig["chat_kinds"]
    (job / "signals.json").write_text(json.dumps(sig), encoding="utf-8")
    fake_llm.queue(make_payload_for(150, 1, 21))
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    sig = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert sig["chat_kinds"] == {"5": {"laugh": 1}}


def test_run_job_passes_loud_seconds_to_render(fixture_video, fake_llm, settings_tmp, monkeypatch):
    from hookline.pipeline import run as run_mod

    seen = []
    real = run_mod.render_clip

    def spy(*a, **k):
        seen.append(k.get("loud_seconds"))
        return real(*a, **k)

    monkeypatch.setattr(run_mod, "render_clip", spy)
    monkeypatch.setattr(
        run_mod, "rms_per_second", lambda w: [-40.0] * 20 + [-20.0] * 5 + [-40.0] * 125
    )
    fake_llm.queue(make_payload_for(150, 1, 21))
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert seen == [set(range(20, 25))]


def test_picks_schema_is_7_and_schema_6_picks_are_reselected(settings, tmp_path):
    from hookline.pipeline.run import PACKAGE_SCHEMA, PICKS_SCHEMA, _load_picks, _picks_key

    # wave G-select: scare markers and the gameplay cause/payoff rules change the picks
    assert PICKS_SCHEMA == 7 and _picks_key(settings, {})["schema"] == 7
    assert PACKAGE_SCHEMA == 3  # wave E: context, hook/hashtag/search-title checks
    key = _picks_key(settings, {})
    old = {**key, "schema": 6}
    ppath = tmp_path / "picks.json"
    ppath.write_text(json.dumps({"key": old, "picks": []}), encoding="utf-8")
    logs: list[str] = []
    assert _load_picks(ppath, key, logs.append) is None
    assert any("schema" in x and "re-selecting" in x for x in logs)
    ppath.write_text(json.dumps({"key": key, "picks": []}), encoding="utf-8")
    assert _load_picks(ppath, key, logs.append) == []


def test_packaging_pass_runs_once_and_fixes_captions(
    fixture_video, fake_llm, settings_tmp, real_packaging
):
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False})
    answer = {
        "clip": 1,
        "hook_text": "Kitchen keys at nineteen",
        "title": "Keys at nineteen",
        "post_caption": "Handed the keys at nineteen. Trust or a test?",
        "hashtags": ["linecook", "kitchenlife", "chefstories"],
        "search_title": "Line Cook Handed the Kitchen Keys at 19",
        "verdict": "post",
        "caption_fixes": [{"index": 1, "original": "says", "text": "sails"}],
    }
    fake_llm.queue(make_payload_for(150, 1, 21), {"clips": [answer]})
    t = FakeTranscriber(duration=150)
    logs: list[str] = []
    (clip,) = run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert "[5/6] select: packaging started" in logs
    assert any(x.startswith("[5/6] select: packaging done") for x in logs)
    assert not any("package:" in x for x in logs)  # one stage, one label
    side = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["hook_text"] == "Kitchen keys at nineteen"
    assert side["search_title"] == "Line Cook Handed the Kitchen Keys at 19"
    (fix,) = side["caption_fixes"]
    assert (fix["original"], fix["text"], fix["start"]) == ("says", "sails", 30.75)
    texts = [w[2] for w in side["caption_words"]]
    assert texts[:2] == ["Line10", "sails"] and texts.count("says") == 6
    job = s.workspace_dir / job_id_for(str(fixture_video))
    assert (job / "package.json").exists() and (job / "package_raw.json").exists()
    run_job(str(fixture_video), s, fake_llm, t)  # cached: neither selection nor packaging
    assert len(fake_llm.calls) == 2


def test_failed_packaging_keeps_selection_copy_and_retries_next_run(
    fixture_video, fake_llm, settings_tmp, real_packaging
):
    from hookline.llm import LLMError

    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    payload = make_payload_for(150, 1, 21)
    fake_llm.queue(payload, LLMError("down"), LLMError("down"))
    logs: list[str] = []
    (clip,) = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150), logs.append)
    side = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["hook_text"] == payload["clips"][0]["hook_text"] and side["caption_fixes"] == []
    job = s.workspace_dir / job_id_for(str(fixture_video))
    assert not (job / "package.json").exists()


# --- wave G-select ------------------------------------------------------------------------------


def test_signals_gain_motion_and_transients_and_old_cache_is_recomputed(
    fixture_video, fake_llm, settings_tmp
):
    from hookline.pipeline.signals import SIGNALS_VERSION

    job = _job_dir(settings_tmp, fixture_video)
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    fake_llm.queue(make_payload_for(150, 1, 21))
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    sig = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert sig["version"] == SIGNALS_VERSION >= 2
    assert len(sig["motion"]) >= 149 and sig["transients"] == [] and sig["onsets"] == []
    # signals cached before wave G-select (no version) are recomputed
    for k in ("version", "motion", "transients", "onsets"):
        del sig[k]
    (job / "signals.json").write_text(json.dumps(sig), encoding="utf-8")
    fake_llm.queue(make_payload_for(150, 1, 21))
    logs: list[str] = []
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150), logs.append)
    sig = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert sig["version"] == SIGNALS_VERSION and "motion" in sig
    assert "[4/6] signals: started" in logs


def test_run_job_trims_idle_ends_before_render_and_records_the_source_bitrate(
    fixture_video, fake_llm, settings_tmp, monkeypatch
):
    from hookline.pipeline import run as run_mod

    seen = []
    real = run_mod.render_clip

    seen_kw = []

    def spy(src, wav, pick, *a, **k):
        seen.append(pick)
        seen_kw.append(k)
        return real(src, wav, pick, *a, **k)

    trims = []
    monkeypatch.setattr(run_mod, "render_clip", spy)
    monkeypatch.setattr(
        run_mod,
        "fit_to_signals",
        lambda pick, words, signals, **k: (
            trims.append(pick) or kws.append(k) or pick.model_copy(update={"end": pick.end - 3})
        ),
    )
    gaps = []
    monkeypatch.setattr(
        run_mod, "idle_gaps", lambda pick, words, signals, **k: gaps.append(pick) or [(25.0, 27.0)]
    )
    kws = []
    fake_llm.queue(make_payload_for(150, 1, 21))
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    (clip,) = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert len(trims) == 1 and seen[0].end == pytest.approx(trims[0].end - 3)
    # round 1: the hard length clamp (payoff allowance) and the quiet level reach the fit;
    # interior idle gaps of the fitted pick reach the render
    assert kws[0] == {"max_s": s.max_clip_s + 5.0, "quiet_db": s.silence_db}
    assert gaps == [seen[0]] and seen_kw[0]["idle_gaps"] == [(25.0, 27.0)]
    side = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["source_video_kbps"] > 0


# --- wave G-select round 1: signals cache -------------------------------------------------------


@pytest.fixture
def signals_inputs(fixture_video, tmp_path):
    from hookline.media.ffmpeg import extract_audio

    wav = extract_audio(fixture_video, tmp_path / "audio.wav")
    transcript = FakeTranscriber(duration=150).transcribe(wav)
    job = tmp_path / "job"
    job.mkdir()
    return job, fixture_video, wav, transcript


def test_signals_cache_keeps_motion_and_recomputes_only_the_onsets(signals_inputs, monkeypatch):
    from hookline.pipeline import run as run_mod
    from hookline.pipeline.signals import SIGNALS_VERSION

    job, src, wav, transcript = signals_inputs
    run_mod._signals(job, src, wav, transcript, lambda m: None)
    sig = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert sig["version"] == SIGNALS_VERSION >= 3 and sig["onset_times"] == []
    # a version-2 cache (no exact onset times) keeps its VOD motion decode
    sig.update({"version": 2, "motion": [7.0] * 150})
    del sig["onset_times"], sig["transient_times"]
    (job / "signals.json").write_text(json.dumps(sig), encoding="utf-8")

    def no_decode(src):
        raise AssertionError("motion re-decoded")

    monkeypatch.setattr(run_mod, "frame_motion", no_decode)
    out = run_mod._signals(job, src, wav, transcript, lambda m: None)
    assert out.motion == [7.0] * 150 and out.version == SIGNALS_VERSION
    assert out.onset_times == [] and out.transient_times == []
    # a chat.json appearing later does not re-decode the motion either
    chat = {"comments": [{"content_offset_seconds": 5.0, "message": {"body": "lol"}}]}
    (job / "chat.json").write_text(json.dumps(chat), encoding="utf-8")
    out = run_mod._signals(job, src, wav, transcript, lambda m: None)
    assert out.chat is not None and out.motion == [7.0] * 150


def test_a_failed_motion_analysis_is_retried_next_run(signals_inputs, monkeypatch):
    from hookline.media.ffmpeg import FFmpegError
    from hookline.pipeline import run as run_mod

    job, src, wav, transcript = signals_inputs
    real = run_mod.frame_motion

    def broken(src):
        raise FFmpegError("decode failed")

    monkeypatch.setattr(run_mod, "frame_motion", broken)
    assert run_mod._signals(job, src, wav, transcript, lambda m: None).motion is None
    monkeypatch.setattr(run_mod, "frame_motion", real)
    logs: list[str] = []
    out = run_mod._signals(job, src, wav, transcript, logs.append)
    assert out.motion and len(out.motion) >= 149
    assert "[4/6] signals: skipped (cached)" not in logs


def test_run_job_builds_the_layout_profile_once_and_passes_it_with_scares(
    fixture_video, fake_llm, settings_tmp, monkeypatch
):
    from hookline.pipeline import run as run_mod
    from hookline.pipeline.profile import LayoutProfile

    built, seen, spikes = [], [], []
    prof = LayoutProfile(size=(320, 180))

    def fake_profile(job_dir, src, detector):
        built.append((job_dir.name, src.name))
        return prof

    def spy(*a, **k):
        seen.append((k.get("profile"), k.get("scares")))
        spikes.append(k.get("spikes"))
        a[5].parent.mkdir(parents=True, exist_ok=True)  # dst
        return {"layout": [], "removed_s": 0.0}

    monkeypatch.setattr(run_mod, "layout_profile", fake_profile)
    monkeypatch.setattr(run_mod, "render_clip", spy)
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2, "captions": False})
    run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert len(built) == 1 and built[0][1].startswith("source")
    assert len(seen) == 2 and all(p is prof for p, _ in seen)
    assert all(isinstance(sc, list) for _, sc in seen)  # the VOD's scares
    assert all(isinstance(sp, list) for sp in spikes)  # gameplay motion spikes (source spans)


def test_beat_spikes_are_the_motion_flashes():
    from hookline.pipeline.signals import Signals

    motion = [5.0] * 60
    motion[30] = 80.0  # a flash / hard visual jump
    sig = Signals(duration=60.0, loudness=[-30.0] * 60, motion=motion)
    assert run_mod.beat_spikes(sig) == [(29.5, 32.0)]
    assert run_mod.beat_spikes(Signals(duration=60.0, loudness=[-30.0] * 60)) == []


def test_layout_profile_failure_of_any_kind_falls_back_to_per_clip(tmp_path, monkeypatch):
    """The profile is optional analysis: a MemoryError (a 4K VOD) or any bug in it logs a
    warning and the clips are planned per clip, never a failed job."""
    from hookline.pipeline import run as run_mod

    def boom(job_dir, src, detector):
        raise MemoryError("Unable to allocate 2.4 GiB")

    monkeypatch.setattr(run_mod, "layout_profile", boom)
    logs: list[str] = []
    assert run_mod._layout_profile(tmp_path, tmp_path / "src.mp4", None, logs.append) is None
    assert any("layout profile failed" in m and "MemoryError" in m for m in logs)
