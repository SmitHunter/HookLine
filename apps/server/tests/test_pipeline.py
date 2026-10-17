import json
from pathlib import Path

import pytest

from hookline.media.ffmpeg import probe_duration, probe_size
from hookline.models import Segment, Transcript, Word
from hookline.pipeline.ingest import job_id_for
from hookline.pipeline.run import NoClipsSelectedError, run_job
from tests.test_select import raw


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
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2})
    logs: list[str] = []
    t = FakeTranscriber(duration=150)
    clips = run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(clips) == 2 and all(c.exists() and c.name.isascii() for c in clips)
    assert all(probe_size(c) == (1080, 1920) for c in clips)
    assert all(probe_duration(c) == pytest.approx(21, abs=1.0) for c in clips)
    side = [json.loads(c.with_suffix(".json").read_text(encoding="utf-8")) for c in clips]
    assert all(d["post_caption"] == d["hook_text"] for d in side)
    assert all(d["layout"] and "keep_ranges" in d and "removed_s" in d for d in side)
    assert any(m.startswith("clip 01: layouts=[") and "removed=" in m for m in logs)
    job = s.workspace_dir / job_id_for(str(fixture_video))
    signals = json.loads((job / "signals.json").read_text(encoding="utf-8"))
    assert len(signals["loudness"]) >= 149 and signals["chat"] is None
    run_job(str(fixture_video), s, fake_llm, t, logs.append)
    assert len(rms_calls) == 1  # signals.json reused on the second run


def test_run_job_names_and_sidecar(fixture_video, fake_llm, settings_tmp):
    payload = make_payload_for(150, 1, 21)
    payload["clips"][0]["title"] = "Sample clip"
    fake_llm.queue(payload)
    s = settings_tmp.model_copy(update={"clips": 1, "vertical": False, "captions": False})
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert [p.name for p in clips] == ["01-sample-clip.mp4"]
    data = json.loads(clips[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert data["title"] == "Sample clip" and data["post_caption"] == data["hook_text"]
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
