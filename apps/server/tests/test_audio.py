import subprocess
from pathlib import Path

import pytest

from hookline.media.audio import detect_silences, rms_per_second


def test_rms_per_second_marks_silence(tone_gap_wav):
    rms = rms_per_second(tone_gap_wav)
    assert len(rms) == 4 and rms[1] < -60 and rms[0] > -30 and rms[2] > -30


def test_rms_per_second_clamps_neg_inf(tone_gap_wav):
    assert min(rms_per_second(tone_gap_wav)) >= -100.0


def test_rms_per_second_non_16k_rate(tmp_path: Path):
    wav = tmp_path / "s8k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=8000:duration=2", str(wav)],
        check=True,
    )  # fmt: skip
    assert len(rms_per_second(wav)) == 2


def test_detect_silences_absolute_and_clipped(tone_gap_wav):
    s = detect_silences(tone_gap_wav, 0.5, 4.0)
    assert (
        len(s) == 1
        and s[0][0] == pytest.approx(1.0, abs=0.05)
        and s[0][1] == pytest.approx(2.0, abs=0.05)
    )


def test_detect_silences_open_silence_closes_at_end(tone_gap_wav):
    s = detect_silences(tone_gap_wav, 0.0, 1.6)
    assert len(s) == 1 and s[0][1] == pytest.approx(1.6, abs=0.01)


def test_detect_silences_none_in_continuous_audio(tone_wav):
    assert detect_silences(tone_wav, 0, 3) == []
