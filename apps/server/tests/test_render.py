import json
import os
import subprocess
from itertools import pairwise
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from hookline.config import Settings
from hookline.media import ffmpeg as ffmpeg_mod
from hookline.media.ffmpeg import extract_audio, probe_duration, probe_size
from hookline.models import ClipPick, RubricScores, Transcript, Word
from hookline.pipeline.layout import SceneLayout
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
    assert seam == pytest.approx(3.14, abs=0.03)  # speech on frame 0 -> 3-frame pre-roll
    assert seam in extras["cut_times"]
    graph = _graph(final_argvs[0])
    assert graph.count("drawbox=") == 3 and "color=white" in graph
    ass = (tmp_path / "r" / "02.ass").read_text(encoding="utf-8")
    (tag,) = [ln for ln in ass.splitlines() if "EARLIER" in ln]
    t0, t1 = f"0:00:{seam:05.2f}", f"0:00:{seam + 1.2:05.2f}"
    assert ",Hook," in tag and tag.startswith(f"Dialogue: 1,{t0},{t1},")
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
    assert words and words[0][0] <= 0.5 and words[0][2] == "Line10"  # capitalised
    assert all(len(w) == 3 and w[0] < w[1] for w in words)
    ass = tmp_path / "r" / "03.ass"
    assert "EARLIER" not in ass.read_text(encoding="utf-8")
    import hashlib

    assert extras["ass"] == "03.ass"
    assert extras["ass_sha1"] == hashlib.sha1(ass.read_bytes()).hexdigest()


# --- audio fades, end tail ------------------------------------------------------------------------


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
    from hookline.pipeline.render import audio_fades

    last = max(e for _, e, _ in extras["caption_words"])  # the fade spares the last word
    assert audio_fades(kept, last).split(",")[1] in af
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


def test_render_cuts_interior_idle_gaps(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path
):
    s = settings.model_copy(update={"vertical": False, "captions": False})
    dst = tmp_path / "out" / "01-x.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, s,
        NoFaces(), idle_gaps=[(40.4, 44.6)],
    )  # fmt: skip
    keep = extras["keep_ranges"]
    assert not any(a < 44.5 and b > 40.5 for a, b in keep)
    assert any(abs(b - 40.4) < 0.05 for _, b in keep) and extras["removed_s"] >= 4.1


# --- splice fades, sliver words at cuts, end fade, room tone, sidecar -----------------------------


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
    # 50/60 ms next to a cut: gone ("ok" is the first word after the 0.55 cut)
    assert [w.text for w in out] == ["kept", "And"]


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
    from hookline.pipeline import timing as timing_mod

    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    # a steady tone has no syllables (the Q3b voice check): take it as speech here
    monkeypatch.setattr(timing_mod, "UNCAPTIONED_SWING_DB", 0.0)
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


def test_audio_fades_never_start_before_the_last_word_ends():
    from hookline.pipeline.render import audio_fades

    assert "afade=t=out:st=9.800:d=0.2" in audio_fades(10.0, last_word_end=9.5)
    assert "afade=t=out:st=9.850:d=0.15" in audio_fades(10.0, last_word_end=9.85)
    # at least 0.15 s, even when the last word ends right at the clip's end
    assert "afade=t=out:st=9.850:d=0.15" in audio_fades(10.0, last_word_end=9.97)
    assert "afade=t=out:st=9.800:d=0.2" in audio_fades(10.0)


def test_render_clip_room_tone_bed_mixes_into_aout(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    monkeypatch.setattr(render_mod, "digital_zero_runs", lambda path, min_s: [(1.0, 0.3)])
    s = settings.model_copy(update={"vertical": False, "captions": False})
    dst = tmp_path / "12-bed.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, dst, s, None
    )
    (final,) = final_argvs
    graph = _graph(final)
    assert extras["room_tone"] is True and "-af" not in final
    assert "anoisesrc=color=pink" in graph and "amix=inputs=2" in graph and "[aout]" in graph
    # both mix inputs pinned to one format, rate and the clip's channel layout: left to amix's
    # negotiation the mix lost ~0.7 dB and the limiter's work (podcast clips at -14.9 LUFS)
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=mono"  # the sine fixture
    assert f",{fmt}[amain]" in graph and f",{fmt}[abed]" in graph
    assert final[final.index("[aout]") - 1] == "-map"
    assert dst.exists() and probe_duration(dst) == pytest.approx(
        sum(e - st for st, e in extras["keep_ranges"]), abs=0.1
    )


def test_render_clip_splices_keep_cuts_and_captions_consistent(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch
):
    # a drop through the pick: the snapped splice edges are what the cut times, the remapped
    # captions and the sidecar all use
    from hookline.pipeline import render as render_mod

    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    pick = sample_pick.model_copy(update={"drops": [(40.0, 43.0)]})
    s = settings.model_copy(update={"vertical": False, "captions": False})
    extras = render_clip(
        fixture_video, fixture_wav, pick, fixture_words, tmp_path, tmp_path / "13-s.mp4", s, None
    )
    keep = extras["keep_ranges"]
    assert extras["cut_times"] == cut_times_for([tuple(k) for k in keep])
    assert any(a < 40.2 and 42.8 < b for (_, a), (b, _) in pairwise(keep))  # the drop is cut
    total = sum(b - a for a, b in keep)
    for start, end, _ in extras["caption_words"]:
        assert 0 <= start < end <= total + 1e-6
        # no caption word runs across a cut
        assert not any(start < c - 0.01 and c + 0.01 < end for c in extras["cut_times"])


def test_drop_cut_slivers_removes_the_short_word_next_to_a_cut_even_a_little_off_it():
    # clip 04: the remnant "up." (75 ms) was re-anchored onto sound 80 ms after the cut: it is
    # still the first word after the cut, a flash; a short word further on stays
    from hookline.pipeline.render import drop_cut_slivers

    words = [Word(start=0.0, end=0.5, text="kept"), Word(start=0.63, end=0.705, text="up."),
             Word(start=0.75, end=1.1, text="And"), Word(start=1.15, end=1.2, text="a"),
             Word(start=1.25, end=1.6, text="lot")]  # fmt: skip
    out = drop_cut_slivers(words, cuts=[0.55])
    assert [w.text for w in out] == ["kept", "And", "a", "lot"]


def test_drop_cut_slivers_keeps_a_short_word_away_from_the_cut():
    from hookline.pipeline.render import drop_cut_slivers

    # a short "I" 0.5 s after a cut is a real word, not a splice remnant
    words = [Word(start=0.0, end=1.0, text="kept"), Word(start=1.5, end=1.56, text="I"),
             Word(start=1.6, end=2.0, text="put")]  # fmt: skip
    assert [w.text for w in drop_cut_slivers(words, cuts=[1.0])] == ["kept", "I", "put"]
    near = [Word(start=0.0, end=1.0, text="kept"), Word(start=1.05, end=1.11, text="up."),
            Word(start=1.6, end=2.0, text="put")]  # fmt: skip
    assert [w.text for w in drop_cut_slivers(near, cuts=[1.0])] == ["kept", "put"]


def test_min_tail_gives_the_end_fade_room_on_a_noisy_bed():
    from hookline.pipeline.render import MIN_FADE_OUT_S, min_tail

    words = [Word(start=9.0, end=9.98, text="end."), Word(start=10.6, end=11.0, text="next")]
    keep = [(0.0, 5.0), (6.0, 10.0)]
    assert min_tail(keep, words) == [(0.0, 5.0), (6.0, pytest.approx(9.98 + MIN_FADE_OUT_S))]
    # never into the next word
    close = [words[0], Word(start=10.05, end=10.4, text="next")]
    assert min_tail(keep, close) == [(0.0, 5.0), (6.0, 10.05)]
    roomy = [Word(start=9.0, end=9.5, text="end.")]
    assert min_tail(keep, roomy) == keep


def test_reaction_peaks_loud_words_and_scare_reactions():
    from hookline.media.audio import Envelope
    from hookline.pipeline.render import reaction_peaks

    db = np.full(2000, -30.0)  # 20 s of speech at -30 dB
    db[500:540] = -18.0  # a shout at 5.0 s (+12 dB)
    db[1200:1240] = -25.0  # +5 dB at 12.0 s, 1 s after a scare
    env = Envelope(t0=0.0, hop=0.01, db=db)
    words = [Word(start=t, end=t + 0.4, text="w") for t in (1.0, 3.0, 5.0, 8.0, 12.0, 15.0)]
    keep = [(0.0, 20.0)]
    peaks = reaction_peaks(words, env, keep, scares=[11.0])
    assert [round(t, 2) for t, _ in peaks] == [5.0, 12.0]
    assert reaction_peaks(words, env, keep, scares=[]) == [(5.0, pytest.approx(12.0))]
    # mapped onto the output timeline
    assert reaction_peaks(words, env, [(4.0, 20.0)], scares=[])[0][0] == pytest.approx(1.0)


def test_reaction_peaks_use_voice_frames_and_loud_shouts():
    """Peaks are measured on voice-like frames only (a game bang under a word is no
    reaction); in a clip of shouting (the median word already a shout) the loud shouts still
    count (`REACT_LOUD_DB`)."""
    from hookline.media.audio import Envelope
    from hookline.pipeline.render import reaction_peaks

    db = np.full(2000, -30.0)
    db[500:540] = -12.0  # a game bang at 5.0 s ...
    voice = np.ones(2000, bool)
    voice[500:540] = False  # ... not voice-like
    db[800:840] = -16.0  # a real shout at 8.0 s
    env = Envelope(t0=0.0, hop=0.01, db=db, voice=voice)
    words = [Word(start=t, end=t + 0.4, text="w") for t in (1.0, 3.0, 5.0, 8.0, 12.0)]
    assert [round(t, 2) for t, _ in reaction_peaks(words, env, [(0.0, 20.0)])] == [8.0]
    # every word a shout (-12 dB): none lifts over the median, the loudest still count
    loud = np.full(2000, -12.0)
    loud[1200:1240] = -10.0
    env = Envelope(t0=0.0, hop=0.01, db=loud, voice=np.ones(2000, bool))
    peaks = reaction_peaks(words, env, [(0.0, 20.0)])
    assert peaks and max(peaks, key=lambda p: p[1])[0] == pytest.approx(12.0)


def _template_profile():
    from hookline.pipeline.profile import LayoutProfile

    # the 320x180 fixture: a cam at the left, rows 90-162
    return LayoutProfile(
        size=(320, 180), bars=(316, 180, 2, 0), cam=(22, 92, 90, 160), outer=(20, 90, 92, 162),
        head=(36, 100, 76, 150), face=(42, 112, 70, 150), masks=[(20, 162, 92, 172)],
        ring=[(20, 90, 255)], share=0.9,
    )  # fmt: skip


def test_render_clip_with_a_profile_uses_the_template_and_the_vod_bars(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod
    from hookline.pipeline.layout import stack_template

    prof = _template_profile()

    def boom(*a, **k):
        raise AssertionError("per-clip black-bar detection must not run with a profile")

    planned = []

    def fake_plan(video, profile, crop=None, detector=None):
        planned.append(crop)
        end = probe_duration(video)
        tpl = stack_template(profile, crop, profile.size)
        return [SceneLayout(start=0.0, end=end, kind="STACKED", **tpl)]

    monkeypatch.setattr(render_mod, "detect_crop", boom)
    monkeypatch.setattr(render_mod, "detect_overlay_bands", lambda video, crop=None: None)
    monkeypatch.setattr(render_mod, "plan_profile_layouts", fake_plan)
    monkeypatch.setattr(render_mod, "voice_threshold", lambda env, db: -44.0)
    dst = tmp_path / "out" / "05-p.mp4"
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r", dst, settings,
        NoFaces(), profile=prof, scares=[],
    )  # fmt: skip
    assert planned == [(316, 180, 2, 0)]
    graph = _graph(final_argvs[0])
    assert graph.startswith("[0:v]crop=316:180:2:0,")
    (lay,) = extras["layout"]
    assert lay["kind"] == "STACKED" and lay["game_box"] is not None and lay["face_px"]
    assert all(r[5] is None for r in lay["face_px"])  # the head region, no eye line
    assert extras["layout_profile"]["cam"] == [22, 92, 90, 160]
    assert probe_size(dst) == (1080, 1920)
    json.dumps(extras)


def test_render_clip_facecam_override_beats_the_profile(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch,
    final_argvs,
):  # fmt: skip
    from hookline.pipeline import render as render_mod

    def boom(*a, **k):
        raise AssertionError("the --facecam override wins over the profile")

    monkeypatch.setattr(render_mod, "plan_profile_layouts", boom)
    s = settings.model_copy(update={"facecam": (0.0, 0.5, 0.3, 0.3)})
    extras = _render_with(fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path, s)
    assert extras["layout"][0]["facecam"] == [0.0, 0.5, 0.3, 0.3]
    extras = render_clip(
        fixture_video, fixture_wav, sample_pick, fixture_words, tmp_path / "r2",
        tmp_path / "06-o.mp4", s, None, profile=_template_profile(),
    )  # fmt: skip
    # the override (mapped into the profile's black-bar crop), not the profile's template
    assert extras["layout"][0]["facecam"] == pytest.approx([0.0, 0.5, 94 / 316, 0.3], abs=0.004)
    assert extras["layout"][0]["game_box"] is None and extras["layout_profile"] is None


def test_beat_avoid_spans_cover_scares_spikes_and_the_tag():
    """Output spans a full-cam beat must stay out of: the EARLIER tag, [scare - 0.5 s,
    scare + 1 s] and gameplay spikes (source spans), all through the keep ranges (startled
    words go to plan_cam_beats apart: never blocking their own beat)."""
    from hookline.pipeline.render import beat_avoid

    keep = [(100.0, 110.0), (120.0, 130.0)]
    spans = beat_avoid(
        seam=None, scares=[105.0, 125.0, 115.0], spikes=[(121.0, 123.0)], keep=keep
    )  # fmt: skip
    assert [4.5, 6.0] in spans and [14.5, 16.0] in spans  # scares (one in a cut: dropped)
    assert [11.0, 13.0] in spans  # the spike
    assert all(not (a < 10.0 < b and b - a > 5) for a, b in spans)
    assert [1.2, 2.4] in beat_avoid(seam=1.2, scares=[], spikes=[], keep=keep)


def test_startle_times_sharp_onsets_at_peak_words():
    from hookline.media.audio import Envelope
    from hookline.pipeline.render import startle_times

    db = np.full(2000, -40.0)
    db[500:540] = -15.0  # a sudden "Oh!" at 5.0 s out of quiet
    db[790:840] = -16.0
    db[700:790] = -20.0  # a shout at 8.0 s that builds up (no jump)
    env = Envelope(t0=0.0, hop=0.01, db=db)
    words = [Word(start=5.0, end=5.4, text="Oh!"), Word(start=8.0, end=8.4, text="no")]
    scares = [5.1, 8.1]  # a game sting at each (no sting, no startle)
    assert startle_times(words, env, [(0.0, 20.0)], [5.0, 8.0], scares) == [5.0]
    assert startle_times(words, env, [(2.0, 20.0)], [3.0], scares) == [3.0]  # output s


def test_mix_format_pins_every_channel_count_to_a_layout(monkeypatch):
    """The room-tone mix inputs get one layout: mono stays mono, stereo and anything wider
    (5.1 / 7.1 sources) a stereo mix, never left to amix's negotiation."""
    import subprocess as sp

    from hookline.pipeline import render as render_mod

    for channels, layout in (("1", "mono"), ("2", "stereo"), ("6", "stereo"), ("8", "stereo")):
        monkeypatch.setattr(
            render_mod, "_run", lambda cmd, c=channels: sp.CompletedProcess(cmd, 0, c + "\n", "")
        )
        assert render_mod.mix_format(Path("x.mkv")).endswith(f"channel_layouts={layout}")


@pytest.mark.parametrize("fail", [False, True])
def test_render_clip_removes_its_filter_script_after_the_encode(
    fixture_video, fixture_wav, fixture_words, sample_pick, settings, tmp_path, monkeypatch, fail
):
    from hookline.media.ffmpeg import FFmpegError
    from hookline.pipeline import render as render_mod

    real = render_mod._run

    def run(cmd):
        if fail and "-/filter_complex" in cmd:
            raise FFmpegError("ffmpeg exited with code 1")
        return real(cmd)

    monkeypatch.setattr(render_mod, "_run", run)
    s = settings.model_copy(update={"vertical": False, "captions": False})
    work = tmp_path / "r"
    if fail:
        with pytest.raises(FFmpegError):
            render_clip(
                fixture_video,
                fixture_wav,
                sample_pick,
                fixture_words,
                work,
                tmp_path / "x.mp4",
                s,
                None,
            )
    else:
        render_clip(
            fixture_video,
            fixture_wav,
            sample_pick,
            fixture_words,
            work,
            tmp_path / "x.mp4",
            s,
            None,
        )
    assert not list(work.glob("*.graph.txt"))
