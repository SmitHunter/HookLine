import json
import os
import subprocess
from itertools import pairwise
from pathlib import Path
from unittest import mock

import pytest

from hookline.config import Settings
from hookline.media import ffmpeg as ffmpeg_mod
from hookline.media.ffmpeg import extract_audio, probe_duration, probe_size
from hookline.models import ClipPick, RubricScores, Transcript, Word
from hookline.pipeline.render import cut_times_for, emphasis_times, render_clip
from hookline.pipeline.run import _face_detector
from tests.conftest import NoFaces
from tests.test_pipeline import FakeTranscriber


def _graph(cmd: list[str]) -> str:
    if "-/filter_complex" in cmd:
        return Path(cmd[cmd.index("-/filter_complex") + 1]).read_text(encoding="utf-8")
    return cmd[cmd.index("-filter_complex") + 1]


@pytest.fixture(scope="module")
def fixture_wav(fixture_video, tmp_path_factory):
    return extract_audio(fixture_video, tmp_path_factory.mktemp("wav") / "audio.wav")


@pytest.fixture(scope="module")
def fixture_words(fixture_wav) -> list[Word]:
    return FakeTranscriber(duration=150).transcribe(fixture_wav).words()


def test_render_clip_vertical_with_captions(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    final_argvs,
):
    dst = tmp_path / "out" / "01-sample-clip.mp4"
    extras = render_clip(
        fixture_video,
        fixture_wav,
        sample_pick,
        fixture_words,
        tmp_path / "render",
        dst,
        settings,
        NoFaces(),
    )
    assert dst.exists() and probe_size(dst) == (1080, 1920)
    assert extras["layout"][0]["kind"] == "FIT_BLUR"
    kept = sum(e - s for s, e in extras["keep_ranges"])
    assert probe_duration(dst) == pytest.approx(kept, abs=0.2)
    assert extras["removed_s"] >= 0
    assert (tmp_path / "render" / "01.ass").exists()
    assert not (tmp_path / "render" / "01.mkv").exists()  # intermediate removed after success
    (final,) = final_argvs
    assert "subtitles=" in _graph(final)
    i = final.index("-c:a")
    assert final[i : i + 6] == ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    af = final[final.index("-af") + 1]
    assert af.startswith("acompressor=") and ",loudnorm=I=-14:TP=-2:" in af
    json.dumps(extras)  # sidecar-serialisable


def test_render_clip_crops_overlay_bands(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    seen = []

    def fake_bands(video, crop=None):
        seen.append(crop)
        return (0, 40)

    monkeypatch.setattr(render_mod, "detect_overlay_bands", fake_bands)
    dst = tmp_path / "03-o.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, settings,
        NoFaces(),
    )  # fmt: skip
    assert seen == [None]  # no letterbox on the fixture
    assert extras["overlay_bands"] == [0, 40]
    assert _graph(final_argvs[0]).startswith("[0:v]crop=320:140:0:0,")
    assert probe_size(dst) == (1080, 1920)
    json.dumps(extras)


def _render_with(fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, settings,
                 detector=None):  # fmt: skip
    dst = tmp_path / "04-f.mp4"
    return render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, settings,
        detector,
    )  # fmt: skip


def _cam_rows(graph: str, pre_y: int = 0) -> tuple[int, int]:
    """Full-frame rows [y0, y1) of the STACKED cam panel crop in `graph`."""
    import re

    _, ch, _, cy = (int(v) for v in re.search(r"_a\]crop=(\d+):(\d+):(\d+):(\d+)", graph).groups())
    return pre_y + cy, pre_y + cy + ch


def test_render_clip_overlay_crop_disabled(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    def boom(video, crop=None):
        raise AssertionError("detection must not run")

    monkeypatch.setattr(render_mod, "detect_overlay_bands", boom)
    s = settings.model_copy(update={"overlay_crop": False})
    extras = _render_with(fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, s)
    assert extras["overlay_bands"] is None


def test_render_clip_facecam_override_over_band_skips_band_removal(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    monkeypatch.setattr(render_mod, "detect_overlay_bands", lambda video, crop=None: (0, 40))
    s = settings.model_copy(update={"facecam": (0.0, 0.7, 0.3, 0.3)})  # rows 126..180 of 180
    extras = _render_with(fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, s)
    assert extras["overlay_bands"] is None
    graph = _graph(final_argvs[0])
    assert not graph.startswith("[0:v]crop=")
    y0, y1 = _cam_rows(graph)
    assert 124 <= y0 and y1 <= 180


def test_render_clip_facecam_override_converted_to_band_crop(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    monkeypatch.setattr(render_mod, "detect_overlay_bands", lambda video, crop=None: (0, 40))
    s = settings.model_copy(update={"facecam": (0.7, 0.0, 0.3, 0.25)})  # rows 0..45 of 180
    extras = _render_with(fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, s)
    assert extras["overlay_bands"] == [0, 40]
    graph = _graph(final_argvs[0])
    assert graph.startswith("[0:v]crop=320:140:0:0,")
    y0, y1 = _cam_rows(graph)
    assert y0 == 0 and 42 <= y1 <= 46  # the original box, not 25% of the shorter crop


def test_render_clip_auto_cam_on_band_edge_replans_without_bands(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod
    from hookline.pipeline.layout import SceneLayout

    monkeypatch.setattr(render_mod, "detect_overlay_bands", lambda video, crop=None: (0, 40))
    crops = []

    def fake_plan(video, detector, facecam_override=None, crop=None):
        crops.append(crop)
        end = probe_duration(video)
        if crop is not None:  # the cam is cut off by the band crop: it touches the bottom
            return [SceneLayout(start=0, end=end, kind="STACKED", facecam=(0.7, 0.75, 0.3, 0.25))]
        return [SceneLayout(start=0, end=end, kind="STACKED", facecam=(0.7, 0.6, 0.3, 0.3))]

    monkeypatch.setattr(render_mod, "plan_layouts", fake_plan)
    extras = _render_with(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, settings, NoFaces()
    )
    assert crops == [(320, 140, 0, 0), None]
    assert extras["overlay_bands"] is None
    assert not _graph(final_argvs[0]).startswith("[0:v]crop=")


def test_render_clip_horizontal_no_captions(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    final_argvs,
):
    s = settings.model_copy(update={"vertical": False, "captions": False})
    dst = tmp_path / "02-h.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, s, None
    )
    w, h = probe_size(dst)
    assert w / h == pytest.approx(320 / 180, abs=0.02)
    assert extras["layout"] == []
    assert extras["overlay_bands"] is None
    assert not (tmp_path / "r" / "02.ass").exists()
    assert "subtitles=" not in _graph(final_argvs[0])


def test_render_clip_silence_window_uses_original_segments(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    from hookline.pipeline import render as render_mod

    seen: list[tuple[float, float]] = []
    real = render_mod.detect_silences

    def spy(wav, start, end, *a, **k):
        seen.append((start, end))
        return real(wav, start, end, *a, **k)

    monkeypatch.setattr(render_mod, "detect_silences", spy)
    pick = sample_pick.model_copy(update={"cold_open": (60.0, 63.0)})
    s = settings.model_copy(update={"vertical": False, "captions": False})
    render_clip(
        fixture_video, fixture_wav, pick, fixture_words, tmp_path, tmp_path / "03-x.mp4", s, None
    )
    assert seen == [(29.0, 64.0)]


@pytest.mark.parametrize("protect", [True, False])
def test_render_clip_word_protection_flag(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch, protect
):
    from hookline.pipeline import render as render_mod

    seen = []
    real = render_mod.keep_ranges

    def spy(*a, **k):
        seen.append(k.get("words"))
        return real(*a, **k)

    monkeypatch.setattr(render_mod, "keep_ranges", spy)
    s = settings.model_copy(update={"vertical": False, "captions": False})
    render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, tmp_path / "04-p.mp4",
        s, None, protect_words=protect,
    )  # fmt: skip
    assert (seen[0] is not None and len(seen[0]) > 0) if protect else seen[0] is None


def test_cut_times_are_output_time_keep_boundaries():
    keep = [(10.0, 12.0), (13.0, 16.5), (20.0, 21.0)]
    assert cut_times_for(keep) == pytest.approx([2.0, 5.5])
    assert cut_times_for([(0.0, 5.0)]) == []


def test_emphasis_times_map_zoom_beats_through_keep_ranges():
    keep = [(10.0, 12.0), (13.0, 16.5)]
    beats = [
        "zoom: 'knives' @ 11.5s",  # -> 1.5
        "zoom: 'gone' @ 12.5s",  # inside a removed range: dropped
        "zoom: 'late' @ 14.0s",  # -> 2 + 1 = 3.0
        "cut: 'x' @ 15.0s",  # not a zoom beat
        "zoom: no time",
    ]
    assert emphasis_times(beats, keep) == pytest.approx([1.5, 3.0])


def _video_kbps(path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=bit_rate", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return int(res.stdout.strip()) / 1000


_ZOOM_PICK = ClipPick(
    start=30.0, end=55.0, cold_open=None, title="Zoom", hook_text="Wait for it",
    reason="test", scores=RubricScores(
        hook=0, context=0, payoff=0, emotion=0, length_fit=0, pacing=0, loop=0, safety=0
    ),
    edit_beats=["zoom: 'something' @ 40.0s"],
)  # fmt: skip


@pytest.fixture(scope="module")
def zoom_render(fixture_video, fixture_wav, fixture_words, tmp_path_factory):
    d = tmp_path_factory.mktemp("zoom")
    dst = d / "05-z.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, _ZOOM_PICK, fixture_words, d / "r", dst, Settings(), NoFaces()
    )
    return dst, extras


def test_render_clip_zoom_cadence(zoom_render):
    dst, extras = zoom_render
    kept = sum(e - s for s, e in extras["keep_ranges"])
    assert extras["cut_times"] == pytest.approx(cut_times_for(extras["keep_ranges"]))
    steps = extras["zoom_steps"]
    times = [t for t, _ in steps] + [kept]
    assert steps[0] == [0.0, 1.0] and max(b - a for a, b in pairwise(times)) <= 4.5 + 1e-6
    assert all(t < kept for t, _ in steps)
    levels = extras["zoom_levels"]  # the schedule; zoom_steps are the rendered zooms
    assert [t for t, _ in levels] == [t for t, _ in steps]
    assert any(s == 1.2 for _, s in levels)  # the emphasis punch
    # 320 px fixture: a soft (> 2x) source, every rendered level capped at 1.12
    assert max(s for _, s in steps) <= 1.12 + 1e-9 and steps[0] == [0.0, 1.0]
    assert len({s for _, s in steps}) == 3  # still three distinct rendered levels
    assert probe_size(dst) == (1080, 1920)
    json.dumps(extras)


_OLD_RATE_CONTROL = {  # pre-M2.5 values
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    "h264_amf": ["-rc", "cqp", "-qp_i", "22", "-qp_p", "24", "-qp_b", "26"],
    "h264_nvenc": ["-rc", "vbr", "-cq", "21", "-b:v", "0"],
}


def test_render_clip_bitrate_rises_vs_previous_settings(
    zoom_render, fixture_video, fixture_wav, fixture_words, tmp_path
):
    dst_new, _ = zoom_render
    dst_old = tmp_path / "06-old.mp4"
    with mock.patch.dict(ffmpeg_mod._ENCODER_QUALITY_ARGS, _OLD_RATE_CONTROL):
        render_clip(
            fixture_video, fixture_wav, _ZOOM_PICK, fixture_words, tmp_path / "r", dst_old,
            Settings(), NoFaces(),
        )  # fmt: skip
    assert _video_kbps(dst_new) > 1.1 * _video_kbps(dst_old)


def _real_job() -> Path | None:
    """A cached real podcast job (HOOKLINE_REAL_JOB, else the local workspace copy)."""
    default = Path(__file__).parents[1] / "workspace" / "e87398ed46"
    job = Path(os.environ.get("HOOKLINE_REAL_JOB", default))
    needed = ("source.mp4", "audio.wav", "transcript.json")
    return job if all((job / n).exists() for n in needed) else None


@pytest.mark.skipif(_real_job() is None, reason="no cached real footage in this environment")
def test_real_footage_bitrate_at_least_5mbps(tmp_path):
    job = _real_job()
    assert job is not None
    raw = json.loads((job / "transcript.json").read_text(encoding="utf-8"))
    transcript = Transcript.model_validate(raw.get("transcript", raw))  # keyed cache format
    pick = _ZOOM_PICK.model_copy(update={"start": 1000.0, "end": 1015.0, "edit_beats": []})
    s = Settings()
    dst = tmp_path / "07-real.mp4"
    render_clip(
        job / "source.mp4", job / "audio.wav", pick, transcript.words(), tmp_path / "r", dst, s,
        _face_detector(s, print),
    )  # fmt: skip
    assert _video_kbps(dst) >= 5000


# --- M2.5 Task 8: cold-open seam marker, caption words in the sidecar --------------------


def _frame_at(path: Path, n: int):
    import numpy as np

    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"select=eq(n\\,{n})", "-fps_mode", "passthrough", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )  # fmt: skip
    return np.frombuffer(res.stdout, np.uint8).reshape(1920, 1080, 3)


def test_seam_time_counts_the_cold_open_ranges():
    from hookline.pipeline.render import seam_time

    keep = [(45.0, 46.2), (46.8, 48.0), (30.0, 40.0), (41.0, 55.0)]
    assert seam_time(keep, (45.0, 48.0)) == pytest.approx(2.4)
    assert seam_time(keep, None) is None
    assert seam_time([(30.0, 55.0)], (45.0, 48.0)) is None  # cold open fully cut away


def test_render_clip_cold_open_seam_flash_and_tag(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, final_argvs
):
    pick = sample_pick.model_copy(update={"cold_open": (45.0, 48.0)})
    dst = tmp_path / "out" / "02-cold.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, pick, fixture_words, tmp_path / "r", dst, settings, NoFaces()
    )
    seam = extras["seam"]
    assert seam == pytest.approx(3.0, abs=0.05)
    assert seam in extras["cut_times"]
    graph = _graph(final_argvs[0])
    assert graph.count("drawbox=") == 3 and "color=white" in graph
    ass = (tmp_path / "r" / "02.ass").read_text(encoding="utf-8")
    (tag,) = [ln for ln in ass.splitlines() if "EARLIER" in ln]
    assert ",Hook," in tag and tag.startswith("Dialogue: 1,0:00:03.00,0:00:04.20,")
    fps = 25
    corner = (slice(0, 60), slice(0, 60))
    n = round(seam * fps)
    peak, ramp, after = (_frame_at(dst, k)[corner].astype(int).mean() for k in (n, n - 1, n + 10))
    assert peak > after + 60  # a visible flash ...
    assert peak < 245  # ... at 60% white, not blown out
    assert after < ramp < peak  # ramp in


def test_render_clip_sidecar_caption_words_and_no_seam(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, final_argvs
):
    dst = tmp_path / "out" / "03-plain.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, settings,
        NoFaces(),
    )  # fmt: skip
    assert extras["seam"] is None and "drawbox=" not in _graph(final_argvs[0])
    words = extras["caption_words"]
    assert words and words[0][0] <= 0.5 and words[0][2] == "line10"
    assert all(len(w) == 3 and w[0] < w[1] for w in words)
    ass = tmp_path / "r" / "03.ass"
    assert "EARLIER" not in ass.read_text(encoding="utf-8")
    import hashlib

    assert extras["ass"] == "03.ass"
    assert extras["ass_sha1"] == hashlib.sha1(ass.read_bytes()).hexdigest()


# --- wave D: audio fades, end tail ---------------------------------------------------------


def _pcm_tail_rms(path, seconds: float) -> tuple[float, float]:
    """(RMS of the middle second, RMS of the last `seconds`) of the decoded audio."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "48000",
         "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    a = np.frombuffer(raw, np.int16).astype(float)
    mid = a[len(a) // 2 : len(a) // 2 + 48000]
    tail = a[-int(seconds * 48000) :]
    return float(np.sqrt((mid**2).mean())), float(np.sqrt((tail**2).mean()))


def test_render_clip_fades_audio_in_and_out(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, final_argvs
):
    dst = tmp_path / "out" / "06-fade.mp4"
    s = settings.model_copy(update={"vertical": False, "captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, s, None
    )
    af = final_argvs[0][final_argvs[0].index("-af") + 1]
    kept = sum(e - st for st, e in extras["keep_ranges"])
    assert "afade=t=in:st=0:d=0.02" in af
    assert f"afade=t=out:st={kept - 0.2:.3f}:d=0.2" in af  # wave F: 0.2 s
    mid, tail = _pcm_tail_rms(dst, 0.03)
    assert tail < 0.3 * mid  # the steady sine is faded out at the end (no click)


def test_render_clip_extends_the_last_range_by_the_tail(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    from hookline.pipeline import render as render_mod

    seen = []

    def fake_tail(last_end, end, next_start, env, threshold_db, **_):
        seen.append((last_end, end, next_start))
        return end + 0.3

    monkeypatch.setattr(render_mod, "tail_end", fake_tail)
    # the fixture is a steady tone: as a bed it reads "too noisy"; force a voice level
    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    s = settings.model_copy(update={"vertical": False, "captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, tmp_path / "07-t.mp4",
        s, None,
    )  # fmt: skip
    (last_end, end, next_start) = seen[0]
    assert last_end <= end <= next_start  # the fixture's sine is voiced up to the next word
    assert extras["keep_ranges"][-1][1] == pytest.approx(end + 0.3, abs=1 / 25)


# --- wave D review round 1 ------------------------------------------------------------------


def test_render_clip_sidecar_zoom_steps_are_the_rendered_guarded_steps(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    from hookline.media.reframe import rendered_zoom_steps
    from hookline.pipeline import render as render_mod
    from hookline.pipeline.layout import SceneLayout

    def lean_plan(video, detector, facecam_override=None, crop=None):
        end = probe_duration(video)
        n = int(end / 0.25)
        xs = [0.5 + 0.3 * (i % 40 >= 20) for i in range(n)]  # leans to 0.8 and back
        kfs = [(i * 0.25, x) for i, x in enumerate(xs)]
        return [
            SceneLayout(
                start=0,
                end=end,
                kind="FACE_TRACK",
                track_x=kfs,
                face_w=0.12,
                face_box=(0.3, 0.3),
                head_top=0.15,
            )
        ]

    graphs = []
    real_graph = render_mod.reframe_graph

    def spy(layouts, *a, **k):
        graphs.append((layouts, k["zoom_steps"]))
        return real_graph(layouts, *a, **k)

    monkeypatch.setattr(render_mod, "plan_layouts", lean_plan)
    monkeypatch.setattr(render_mod, "reframe_graph", spy)
    s = settings.model_copy(update={"captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, tmp_path / "08-g.mp4",
        s, NoFaces(),
    )  # fmt: skip
    ((layouts, rendered_steps),) = graphs
    assert extras["zoom_levels"] == [list(st) for st in rendered_steps]
    assert extras["zoom_steps"] == [list(st) for st in rendered_zoom_steps(layouts, rendered_steps)]
    assert extras["layout"][0]["face_dx"] is not None


def test_render_clip_fades_use_the_probed_duration_and_sidecar_includes_the_tail(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    final_argvs,
):
    from hookline.pipeline import render as render_mod

    probed = []
    real_probe = render_mod.probe_duration

    def spy_probe(path):
        d = real_probe(path)
        if str(path).endswith(".mkv"):
            probed.append(d)
        return d

    monkeypatch.setattr(render_mod, "probe_duration", spy_probe)
    monkeypatch.setattr(render_mod, "tail_end", lambda last, end, nxt, env, thr, **_: end + 0.3)
    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    s = settings.model_copy(update={"vertical": False, "captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, tmp_path / "09-f.mp4",
        s, None,
    )  # fmt: skip
    af = final_argvs[0][final_argvs[0].index("-af") + 1]
    assert f"afade=t=out:st={probed[0] - 0.2:.3f}:d=0.2" in af
    keep, segs = extras["keep_ranges"], extras["segments"]
    assert segs[-1][1] == pytest.approx(keep[-1][1])  # the tail is part of the clip's span
    kept = sum(e - st for st, e in keep)
    assert extras["removed_s"] == pytest.approx(sum(e - st for st, e in segs) - kept, abs=0.002)


def test_render_clip_skips_word_refinement_on_a_noisy_bed(
    fixture_video,
    fixture_wav,
    fixture_words,
    sample_pick,
    settings,
    tmp_path,
    monkeypatch,
    caplog,
):
    from hookline.pipeline import render as render_mod

    calls = []
    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: None)
    monkeypatch.setattr(render_mod, "refine_word_ends", lambda *a, **k: calls.append(1))
    s = settings.model_copy(update={"vertical": False, "captions": False})
    with caplog.at_level("WARNING"):
        render_clip(
            fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path,
            tmp_path / "10-n.mp4", s, None,
        )  # fmt: skip
    assert calls == []
    assert any("noise" in r.getMessage() for r in caplog.records)


# --- fix wave F: splice fades, sliver words at cuts, end fade, room tone, sidecar --------


def _pcm(path, rate=48000):
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate),
         "-f", "s16le", "-"], check=True, capture_output=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.int16).astype(float) / 32768


def test_cut_segments_fades_the_audio_at_every_splice(tmp_path):
    from hookline.media.ffmpeg import cut_segments

    src = tmp_path / "tone.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
         "-af", "volume=6dB", "-c:v", "libx264", "-c:a", "pcm_s16le", "-shortest", str(src)],
        check=True,
    )  # fmt: skip
    keep = [(0.0, 1.0), (1.5 + 1 / 60 + 1 / 30, 3.0)]  # the sine's phase jumps at the splice
    keep = [(round(a * 30) / 30, round(b * 30) / 30) for a, b in keep]
    out = cut_segments(src, keep, tmp_path / "cut.mkv", "libx264", fps=30.0)
    a = _pcm(out)
    i = round((keep[0][1] - keep[0][0]) * 48000)
    assert abs(a[i - 24 : i + 24]).max() < 0.05  # faded to ~0 around the splice: no click
    assert abs(a[i - 2400 : i - 1200]).max() > 0.2  # full level (0.25) 25-50 ms away
    assert abs(a[i + 1200 : i + 2400]).max() > 0.2


def test_drop_cut_slivers_removes_short_words_touching_a_cut():
    from hookline.pipeline.render import drop_cut_slivers

    words = [Word(start=0.0, end=0.5, text="kept"), Word(start=0.5, end=0.55, text="alone."),
             Word(start=0.7, end=0.75, text="ok"), Word(start=2.0, end=2.06, text="up."),
             Word(start=2.1, end=2.5, text="And")]  # fmt: skip
    out = drop_cut_slivers(words, cuts=[0.55, 2.06])
    assert [w.text for w in out] == ["kept", "ok", "And"]  # 50/60 ms at a cut: gone


def test_digital_zero_runs_finds_noise_gate_silence(tmp_path):
    from hookline.media.audio import digital_zero_runs

    wav = tmp_path / "gate.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "aevalsrc='0.3*sin(2*PI*300*t)*(lt(t,1)+gte(t,1.3))':s=48000:d=2", str(wav)],
        check=True,
    )  # fmt: skip
    runs = digital_zero_runs(wav)
    assert len(runs) == 1 and runs[0][0] == pytest.approx(1.0, abs=0.01)
    assert runs[0][1] == pytest.approx(0.3, abs=0.01)
    assert digital_zero_runs(wav, min_s=0.5) == []


def test_room_tone_bed_sits_at_minus_62_dbfs(tmp_path):
    import numpy as np

    from hookline.pipeline.render import ROOM_TONE_DB, room_tone

    dst = tmp_path / "bed.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", room_tone(5.0), str(dst)], check=True,
    )  # fmt: skip
    a = _pcm(dst)
    rms = 20 * np.log10(np.sqrt((a**2).mean()))
    assert ROOM_TONE_DB == -62.0 and rms == pytest.approx(-62.0, abs=1.5)


def test_render_clip_sidecar_reports_uncaptioned_speech_snaps_and_face_samples(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    from hookline.pipeline import render as render_mod

    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    s = settings.model_copy(update={"captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words[::3], tmp_path,
        tmp_path / "11-u.mp4", s, NoFaces(),
    )  # fmt: skip
    # the fixture's tone is voiced throughout: dropping 2 of 3 words leaves it uncaptioned
    assert extras["uncaptioned_speech"]
    assert all(0 <= a < b for a, b in extras["uncaptioned_speech"])
    lay = extras["layout"][0]
    assert "snaps" in lay and "face_px" in lay and "gaps" not in lay


def test_with_tail_ends_0_2_to_0_3_s_after_the_speech_even_before_the_old_end():
    # clip 02: the transcript's "action." ran 0.45 s on over silence, so the clip ended
    # 0.66 s after the voice; the end now follows the last voiced frame
    import numpy as np

    from hookline.media.audio import Envelope
    from hookline.pipeline.render import _with_tail

    db = np.full(400, -60.0)
    db[100:220] = -15.0  # voiced 1.0-2.2 s
    env = Envelope(t0=0.0, hop=0.01, db=db)
    words = [Word(start=1.0, end=1.8, text="direct"), Word(start=1.8, end=2.9, text="action.")]
    ((_, end),) = _with_tail([(0.5, 3.0)], words, env, -32.0)
    assert 2.2 + 0.2 - 0.011 <= end <= 2.2 + 0.3 + 1e-9
    # speech after the last word (uncaptioned) is never cut off
    db[240:260] = -15.0  # 2.4-2.6 s
    ((_, end2),) = _with_tail([(0.5, 3.0)], words[:1], Envelope(0.0, 0.01, db), -32.0)
    assert end2 >= 2.6 + 0.2 - 0.011
