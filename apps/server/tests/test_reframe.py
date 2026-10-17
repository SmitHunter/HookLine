import subprocess

import pytest

from hookline.media.reframe import detect_crop


def _make(dst, vf=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", "testsrc2=size=1280x720:rate=25:duration=5"]  # fmt: skip
    if vf:
        cmd += ["-vf", vf]
    subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def test_detect_crop_finds_letterbox(tmp_path):
    crop = detect_crop(_make(tmp_path / "boxed.mp4", "pad=1280:920:0:100"))
    assert crop is not None
    for got, want in zip(crop, (1280, 720, 0, 100), strict=True):
        assert got == pytest.approx(want, abs=4)


def test_detect_crop_none_without_bars(tmp_path):
    assert detect_crop(_make(tmp_path / "plain.mp4")) is None
