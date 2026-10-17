import json
from pathlib import Path

import pytest

from hookline.media.ffmpeg import probe_duration
from hookline.models import Segment, Transcript, Word
from hookline.pipeline.cut import cut_clips
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


def test_run_job_end_to_end_with_fakes(fixture_video, fake_llm, settings_tmp):
    fake_llm.queue(make_payload_for(150, 2, 21))
    s = settings_tmp.model_copy(update={"clips": 2})
    logs: list[str] = []
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150), logs.append)
    assert len(clips) == 2 and all(c.exists() and c.name.isascii() for c in clips)
    assert all(probe_duration(c) == pytest.approx(21, abs=0.6) for c in clips)
    side = [json.loads(c.with_suffix(".json").read_text(encoding="utf-8")) for c in clips]
    assert all(d["post_caption"] == d["hook_text"] for d in side)
    assert logs


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


def test_cut_clips_names_and_sidecar(fixture_video, sample_pick, tmp_path):
    out = cut_clips(fixture_video, [sample_pick], tmp_path / "o", "libx264")
    assert [p.name for p in out] == ["01-sample-clip.mp4"]
    data = json.loads(out[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert data["post_caption"] == "Wait for it" and data["title"] == "Sample clip"


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
