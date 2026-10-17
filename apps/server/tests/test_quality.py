"""`hookline audit`: each quality-bar check passes on a compliant synthetic clip and fails on
a crafted bad one."""

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hookline import cli
from hookline.captions.ass import build_ass
from hookline.media.loudness import measure_loudness
from hookline.models import Word
from hookline.quality import (
    DOMINANT_ROWS,
    CheckResult,
    _split_ass,
    audit_clip,
    audit_job,
    caption_samples,
    first_words_match,
    line_heights,
    render_ass_frames,
)

DUR = 8.0
FIRST = "This is the opening line."
SENTENCES = [
    FIRST,
    "Then something big happens.",
    "Nobody saw it coming.",
    "And that was the end.",
]
CHECK_IDS = ["Q1", "Q2", "Q3", "Q4", "Q4b", "Q5", "Q6", "Q7", "Q8"]


def _words(t0: float = 0.2) -> list[Word]:
    """Four sentences, 0.3 s per word with 0.05 s gaps and 0.3 s between sentences."""
    out, t = [], t0
    for sentence in SENTENCES:
        for tok in sentence.split():
            out.append(Word(start=round(t, 3), end=round(t + 0.3, 3), text=tok))
            t += 0.35
        t += 0.3
    return out


def _make_mp4(path: Path, size: str, vbit: str, lufs: float, abit: str) -> Path:
    """testsrc2 video at a constant `vbit` plus limited pink noise at about `lufs` integrated loudness."""

    def encode(dst: Path, gain_db: float) -> None:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={DUR}",
             "-f", "lavfi", "-i", f"anoisesrc=color=pink:seed=7:duration={DUR}:sample_rate=48000",
             "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.7:level=false",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-b:v", vbit, "-minrate", vbit, "-maxrate", vbit, "-bufsize", vbit,
             "-x264-params", "nal-hrd=cbr",
             "-c:a", "aac", "-b:a", abit, "-ar", "48000", "-shortest", str(dst)],
            check=True,
        )  # fmt: skip

    gain = 0.0
    for _ in range(4):  # the limiter (keeps true peak < -1 dBTP) bends the gain slightly
        encode(path, gain)
        m = measure_loudness(path)
        assert m is not None
        if abs(m.i - lufs) <= 0.3:
            break
        gain += lufs - m.i
    assert abs(m.i - lufs) <= 0.5, m
    return path


@pytest.fixture(scope="module")
def good_mp4(tmp_path_factory) -> Path:
    return _make_mp4(
        tmp_path_factory.mktemp("good") / "01-good.mp4", "1080x1920", "6M", -14, "192k"
    )


@pytest.fixture(scope="module")
def bad_mp4(tmp_path_factory) -> Path:
    return _make_mp4(tmp_path_factory.mktemp("bad") / "01-bad.mp4", "720x1280", "1M", -30, "96k")


def _sidecar(**over) -> dict:
    words = _words()
    side = {
        "title": "Synthetic",
        "first_words": FIRST,
        "caption_words": [[w.start, w.end, w.text] for w in words],
        "cut_times": [4.0],
        "zoom_steps": [[0.0, 1.0], [2.5, 1.1], [4.0, 1.0], [6.5, 1.1]],
        "layout": [{"start": 0.0, "end": DUR, "kind": "FIT_BLUR", "face_box": None}],
        "seam": None,
    }
    return {**side, **over}


def _ass(tmp_path: Path, text: str | None = None, name: str = "01.ass") -> Path:
    path = tmp_path / name
    path.write_text(text or build_ass(_words(), "Wait for the twist", DUR), encoding="utf-8")
    return path


def _by_id(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.id: r for r in results}


def test_compliant_clip_passes_every_check(good_mp4, tmp_path):
    res = audit_clip(good_mp4, _sidecar(), _ass(tmp_path))
    assert [r.id for r in res] == CHECK_IDS
    failed = [r for r in res if not r.ok]
    assert not failed, failed
    by = _by_id(res)
    assert by["Q2"].value >= 55 and by["Q5"].value >= 60
    assert by["Q3"].value >= 0.95
    json.dumps([r.as_dict() for r in res])


def test_quiet_small_low_bitrate_file_fails_q1_and_q8(bad_mp4, tmp_path):
    by = _by_id(audit_clip(bad_mp4, _sidecar(), _ass(tmp_path)))
    assert not by["Q1"].ok and by["Q1"].value == pytest.approx(-30, abs=1.5)
    assert not by["Q8"].ok and "720x1280" in by["Q8"].detail


def test_small_caption_font_fails_q2(good_mp4, tmp_path):
    text = build_ass(_words(), "Wait for the twist", DUR).replace(
        "Style: Cap,Anton,152", "Style: Cap,Anton,70"
    )
    by = _by_id(audit_clip(good_mp4, _sidecar(), _ass(tmp_path, text)))
    assert not by["Q2"].ok and by["Q2"].value < 55
    assert by["Q3"].ok  # still on screen


def test_gappy_captions_fail_q3(good_mp4, tmp_path):
    every_other = [w for i, w in enumerate(_words()) if i % 2 == 0]
    text = build_ass(every_other, "Wait for the twist", DUR)
    # hold rule would bridge the gaps; force each Cap event to end at its word's end
    lines = []
    caps = iter(every_other)
    for ln in text.splitlines():
        if ln.startswith("Dialogue: 0,"):
            w = next(caps)
            parts = ln.split(",", 9)
            parts[2] = f"0:00:{w.end:05.2f}"
            ln = ",".join(parts)
        lines.append(ln)
    by = _by_id(audit_clip(good_mp4, _sidecar(), _ass(tmp_path, "\n".join(lines) + "\n")))
    assert not by["Q3"].ok and by["Q3"].value < 0.95


def test_caption_outside_safe_zone_fails_q4(good_mp4, tmp_path):
    text = build_ass(_words(), "Wait for the twist", DUR).replace(
        "\\pos(474,1180)", "\\pos(474,1400)"
    )
    by = _by_id(audit_clip(good_mp4, _sidecar(), _ass(tmp_path, text)))
    assert not by["Q4"].ok


def test_caption_on_the_face_fails_q4b(good_mp4, tmp_path):
    face = [{"start": 0.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [0.55, 0.15]}]
    by = _by_id(audit_clip(good_mp4, _sidecar(layout=face), _ass(tmp_path)))
    assert not by["Q4b"].ok and by["Q4b"].value > 0.10
    clear = [{"start": 0.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [0.25, 0.25]}]
    assert _by_id(audit_clip(good_mp4, _sidecar(layout=clear), _ass(tmp_path)))["Q4b"].ok


def test_hook_on_the_face_fails_q4b(good_mp4, tmp_path):
    # face 230-614 px: clear of the 1180 captions, but under the top hook box (290-466)
    high = [{"start": 0.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [0.12, 0.2]}]
    by = _by_id(audit_clip(good_mp4, _sidecar(layout=high), _ass(tmp_path)))
    assert not by["Q4b"].ok and "hook" in by["Q4b"].detail


def test_zoomed_face_overlap_detected_q4b(good_mp4, tmp_path):
    # face 1240-1320 px misses the 1180 caption text (~1155-1221) unzoomed; a 4x step
    # (centre 1280, half 160: box 1120-1440, face core from 1184) reaches it
    low = [{"start": 0.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [1240 / 1920, 80 / 1920]}]
    flat = _sidecar(layout=low, zoom_steps=[[0.0, 1.0]], cut_times=[2.5, 5.0])
    assert _by_id(audit_clip(good_mp4, flat, _ass(tmp_path)))["Q4b"].ok
    zoomed = _sidecar(layout=low, zoom_steps=[[0.0, 1.0], [4.0, 4.0]], cut_times=[2.5])
    by = _by_id(audit_clip(good_mp4, zoomed, _ass(tmp_path)))
    assert not by["Q4b"].ok and by["Q4b"].value > 0.10


@pytest.mark.parametrize(
    ("top", "ok"),
    [(1140, True), (1000, False)],
    ids=["hairline-only", "eyes-and-mouth"],
)
def test_q4b_measures_the_face_core_not_the_hairline(good_mp4, tmp_path, top, ok):
    # 400 px face box. top=1140: the 1180 caption band (~1155-1221) lies in the box's top 20%
    # (1140-1220, hairline/upper forehead): acceptable. top=1000: core from 1080, so the
    # band sits on the eyes/nose: not acceptable.
    lay = [{"start": 0.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [top / 1920, 400 / 1920]}]
    side = _sidecar(layout=lay, zoom_steps=[[0.0, 1.0]])
    q4b = _by_id(audit_clip(good_mp4, side, _ass(tmp_path)))["Q4b"]
    assert q4b.ok is ok
    if not ok:
        assert "face core" in q4b.detail and "face-core" in q4b.threshold


def test_one_overflowing_group_among_many_fails_q4(good_mp4, tmp_path):
    # 30 one-word groups; the 2nd is a word too long even at CAP_MIN_FS / MIN_SCALE_X. An
    # evenly spread 12-sample audit (indices 0, 3, 5, ...) never looked at it.
    words = [Word(start=0.2 + 0.25 * i, end=0.4 + 0.25 * i, text=f"W{i}.") for i in range(30)]
    words[1] = Word(start=words[1].start, end=words[1].end, text="W" * 30)
    side = _sidecar(caption_words=[[w.start, w.end, w.text] for w in words], first_words="W0")
    by = _by_id(audit_clip(good_mp4, side, _ass(tmp_path, build_ass(words, "Wait", DUR))))
    assert not by["Q4"].ok and by["Q4"].value > 0


def test_caption_samples_one_per_group_and_every_face_scene():
    ass = build_ass(_words(), "Wait for the twist", DUR)
    _, events = _split_ass(ass)
    caps = [e for _, e in events if e.style == "Cap"]
    groups = len({e.text for e in caps})
    scenes = [
        {"start": 0.0, "end": 7.2, "kind": "FIT_BLUR"},
        {"start": 7.2, "end": 7.3, "kind": "FACE_TRACK"},  # no group ends in here
    ]
    times = caption_samples(caps, scenes)
    assert len(times) == groups + 1 and any(7.2 <= t < 7.3 for t in times)


def test_q2_measures_the_cap_band_not_comma_tails_or_accents(tmp_path):
    ass = tmp_path / "c.ass"
    ass.write_text(build_ass([Word(start=0, end=2, text="É,Ü,À,Q,")], "", 2), encoding="utf-8")
    frame = render_ass_frames(ass, [1.0])[0]
    loose = line_heights(frame)
    dense = line_heights(frame, dense=DOMINANT_ROWS)
    assert len(dense) == 2  # the accents float above the line as their own row run
    assert 72 <= max(dense) <= 80 < max(loose)  # comma tails inflate the loose measure


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace(
            "Dialogue: 1,0:00:00.00,0:00:03.00", "Dialogue: 1,0:00:00.00,0:00:01.50"
        ),
        lambda t: t.replace("WAIT FOR THE TWIST", "WAIT\\NFOR THE\\NTWIST"),
        lambda t: t.replace("WAIT FOR THE TWIST", "WAIT FOR THE TWIST\\NAND SEE WHAT HAPPENS"),
    ],
    ids=["short-lived", "three-lines", "nine-words"],
)
def test_bad_hook_fails_q5(good_mp4, tmp_path, edit):
    base = build_ass(_words(), "Wait for the twist", DUR)
    assert "Dialogue: 1,0:00:00.00,0:00:03.00" in base and "WAIT FOR THE TWIST" in base
    by = _by_id(audit_clip(good_mp4, _sidecar(), _ass(tmp_path, edit(base))))
    assert not by["Q5"].ok


def test_small_hook_fails_q5(good_mp4, tmp_path):
    base = build_ass(_words(), "Wait for the twist", DUR)
    text = base.replace("Style: Hook,Anton,136", "Style: Hook,Anton,80")
    by = _by_id(audit_clip(good_mp4, _sidecar(), _ass(tmp_path, text)))
    assert not by["Q5"].ok and by["Q5"].value < 60


def test_late_speech_fails_q6(good_mp4, tmp_path):
    late = _words(t0=1.2)
    side = _sidecar(caption_words=[[w.start, w.end, w.text] for w in late])
    ass = _ass(tmp_path, build_ass(late, "Wait for the twist", DUR))
    by = _by_id(audit_clip(good_mp4, side, ass))
    assert not by["Q6"].ok and by["Q6"].value == pytest.approx(1.2)


def test_wrong_first_words_fail_q6(good_mp4, tmp_path):
    by = _by_id(audit_clip(good_mp4, _sidecar(first_words="Nobody saw it coming."), _ass(tmp_path)))
    assert not by["Q6"].ok and "match" in by["Q6"].detail


@pytest.mark.parametrize(
    ("want", "got", "ok"),
    [
        ("the reason we did that was money", "we did it because the money was there", False),
        ("the reason we did that was money", "the reason we did that was money and", True),
        ("the reason we did that was money", "um the reason we did that was the money", True),
        ("Nobody saw it coming.", "coming it saw nobody", False),
    ],
)
def test_first_words_match_is_order_aware(want, got, ok):
    assert (first_words_match(want, got) >= 0.6) is ok


def test_first_words_match_after_trimmed_filler_opener(good_mp4, tmp_path):
    side = _sidecar(first_words="So, you know, this is the opening line.")
    assert _by_id(audit_clip(good_mp4, side, _ass(tmp_path)))["Q6"].ok


def test_first_words_checked_after_the_cold_open_seam(good_mp4, tmp_path):
    side = _sidecar(first_words="Then something big happens.", seam=2.5)
    assert _by_id(audit_clip(good_mp4, side, _ass(tmp_path)))["Q6"].ok


def test_no_framing_change_fails_q7(good_mp4, tmp_path):
    by = _by_id(
        audit_clip(good_mp4, _sidecar(cut_times=[], zoom_steps=[[0.0, 1.0]]), _ass(tmp_path))
    )
    assert not by["Q7"].ok and by["Q7"].value == pytest.approx(DUR, abs=0.1)
    scenes = [
        {"start": 0.0, "end": 4.0, "kind": "FIT_BLUR", "face_box": None},
        {"start": 4.0, "end": DUR, "kind": "FIT_BLUR", "face_box": None},
    ]
    side = _sidecar(cut_times=[], zoom_steps=[[0.0, 1.0]], layout=scenes)
    assert _by_id(audit_clip(good_mp4, side, _ass(tmp_path)))["Q7"].ok  # scene change at 4 s


def test_missing_caption_words_fail_q3_and_q6(good_mp4, tmp_path):
    side = _sidecar()
    del side["caption_words"]
    by = _by_id(audit_clip(good_mp4, side, _ass(tmp_path)))
    assert not by["Q3"].ok and not by["Q6"].ok and "caption_words" in by["Q3"].detail


# --- `hookline audit JOB_ID` ------------------------------------------------------------


def _job(tmp_path: Path, mp4: Path) -> tuple[Path, Path]:
    out, ws = tmp_path / "output", tmp_path / "workspace"
    (out / "job1").mkdir(parents=True)
    (ws / "job1" / "render").mkdir(parents=True)
    dst = out / "job1" / "01-good.mp4"
    dst.write_bytes(mp4.read_bytes())
    dst.with_suffix(".json").write_text(json.dumps(_sidecar()), encoding="utf-8")
    _ass(ws / "job1" / "render")
    return out, ws


def test_cli_audit_exit_codes_and_report(good_mp4, tmp_path, monkeypatch):
    out, ws = _job(tmp_path, good_mp4)
    monkeypatch.setenv("HOOKLINE_OUTPUT_DIR", str(out))
    monkeypatch.setenv("HOOKLINE_WORKSPACE_DIR", str(ws))
    res = CliRunner().invoke(cli.app, ["audit", "job1"])
    assert res.exit_code == 0, res.output
    assert "01-good" in res.output and "Q4b" in res.output
    report = json.loads((out / "job1" / "audit.json").read_text(encoding="utf-8"))
    assert report["ok"] is True and report["clips"][0]["clip"] == "01-good.mp4"

    side = out / "job1" / "01-good.json"
    side.write_text(json.dumps(_sidecar(cut_times=[], zoom_steps=[[0.0, 1.0]])), encoding="utf-8")
    res = CliRunner().invoke(cli.app, ["audit", "job1"])
    assert res.exit_code == 1 and "FAIL" in res.output
    assert json.loads((out / "job1" / "audit.json").read_text(encoding="utf-8"))["ok"] is False


def test_cli_audit_unknown_job_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("HOOKLINE_OUTPUT_DIR", str(tmp_path))
    res = CliRunner().invoke(cli.app, ["audit", "nope"])
    assert res.exit_code == 1 and "nope" in res.output


def test_audit_job_pairs_clips_with_their_ass(good_mp4, tmp_path):
    out, ws = _job(tmp_path, good_mp4)
    report = audit_job(out / "job1", ws / "job1")
    assert report["clips"][0]["ass"].endswith("01.ass")


def test_audit_job_uses_the_recorded_ass_and_verifies_its_hash(good_mp4, tmp_path):
    import hashlib

    out, ws = _job(tmp_path, good_mp4)
    real = _ass(ws / "job1" / "render", name="01-abc.ass")
    side = _sidecar(ass="01-abc.ass", ass_sha1=hashlib.sha1(real.read_bytes()).hexdigest())
    (out / "job1" / "01-good.json").write_text(json.dumps(side), encoding="utf-8")
    report = audit_job(out / "job1", ws / "job1")
    assert report["ok"] and report["clips"][0]["ass"].endswith("01-abc.ass")
    real.write_text(real.read_text(encoding="utf-8") + "; edited", encoding="utf-8")
    checks = audit_job(out / "job1", ws / "job1")["clips"][0]["checks"]
    assert checks[0]["id"] == "ASS" and not checks[0]["ok"]


def test_audit_job_reports_a_broken_clip_and_continues(good_mp4, tmp_path):
    out, ws = _job(tmp_path, good_mp4)
    (out / "job1" / "00-broken.mp4").write_bytes(b"not a video")
    report = audit_job(out / "job1", ws / "job1")
    broken, good = report["clips"]
    assert broken["clip"] == "00-broken.mp4" and broken["checks"][0]["id"] == "ERR"
    assert good["ok"] and report["ok"] is False


def test_seam_into_a_close_up_keeps_tag_and_early_captions_off_the_face(good_mp4, tmp_path):
    from hookline.captions.ass import caption_y_for, early_until, hook_y_for
    from hookline.pipeline.layout import SceneLayout

    # wide shot 0-5 s, then (from the cold-open seam at 5 s) a close-up with a high face:
    # box 230-806 px, face core from 345 px
    scenes = [
        {"start": 0.0, "end": 5.0, "kind": "FIT_BLUR", "face_box": None},
        {"start": 5.0, "end": DUR, "kind": "FACE_TRACK", "face_box": [0.12, 0.3]},
    ]
    lays = [SceneLayout.model_validate(s) for s in scenes]
    words, seam = _words(), 5.0
    until = early_until(words, "Wait for the twist", DUR, seam)
    assert until >= seam + 0.8  # the tag's window counts as "early"

    def audit(caption_y: int, hook_y: int) -> CheckResult:
        ass = build_ass(words, "Wait for the twist", DUR, caption_y, hook_y, seam=seam)
        side = _sidecar(layout=scenes, seam=seam, zoom_steps=[[0.0, 1.0]], cut_times=[2.5, 5.0])
        return _by_id(audit_clip(good_mp4, side, _ass(tmp_path, ass)))["Q4b"]

    # first-3-s-only placement: the EARLIER tag lands at the top, on the close-up's face
    stale = audit(caption_y_for(lays), hook_y_for(lays))
    assert not stale.ok and "hook" in stale.detail
    fixed = audit(caption_y_for(lays, until=until), hook_y_for(lays, until=until))
    assert fixed.ok, fixed
    assert hook_y_for(lays, until=until) > 806  # tag below the face, above the captions


def test_face_check_uses_the_schedule_levels_not_the_rendered_zooms():
    from hookline.quality import zoom_levels

    side = {"zoom_steps": [[0.0, 1.0], [2.0, 1.072]], "zoom_levels": [[0.0, 1.0], [2.0, 1.12]]}
    assert zoom_levels(side) == [[0.0, 1.0], [2.0, 1.12]]
    old = {"zoom_steps": [[0.0, 1.0], [2.0, 1.12]]}  # pre-wave-D sidecars recorded levels
    assert zoom_levels(old) == [[0.0, 1.0], [2.0, 1.12]]
    assert zoom_levels({}) == []


def test_q7_counts_guard_shift_steps_and_reads_the_guarded_zoom_steps():
    from hookline.quality import _check_pacing

    lay = {"start": 0.0, "end": 20.0, "kind": "FACE_TRACK", "shift_steps": [[9.0, 0.04]]}
    side = {"cut_times": [], "layout": [lay], "zoom_steps": [[0.0, 1.0], [4.5, 1.072]]}
    held = _check_pacing({**side, "layout": [{**lay, "shift_steps": []}]}, 13.0)
    assert not held.ok and held.value == pytest.approx(8.5)  # a guard hold, honestly reported
    assert _check_pacing(side, 13.0).ok  # the shift at 9.0 is a framing change
