import subprocess
from pathlib import Path

import pytest

from hookline.media.motion import frame_motion


@pytest.fixture(scope="module")
def flash_video(tmp_path_factory) -> Path:
    """6 s of a still grey frame with a 1 s white flash at 3 s (no audio)."""
    out = tmp_path_factory.mktemp("motion") / "flash.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=gray:size=320x180:rate=25:duration=6",
         "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,3,3.99)'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


def test_frame_motion_is_per_second_and_spikes_on_a_flash(flash_video):
    motion = frame_motion(flash_video)
    assert len(motion) == 6
    assert max(motion[0], motion[1], motion[5]) < 1.0  # a still frame
    assert motion[3] > 60 and motion[4] > 60  # into the flash (3 s) and out of it (4 s)


# --- a damaged VOD floods stderr; a stuck decode times out ----------------------------------------

_FLOOD = (
    "import sys; sys.stderr.buffer.write(b'concealing errors in P frame ' * 20000); "
    "sys.stderr.flush(); sys.stdout.buffer.write(bytes(3 * 64 * 36 * 8)); sys.stdout.flush()"  # yuv444p
)


def _fake_popen(monkeypatch, script):
    import sys
    import threading

    from hookline.media import motion as motion_mod

    procs = []
    real = subprocess.Popen

    def fake(cmd, **kw):
        procs.append(real([sys.executable, "-c", script], **kw))
        return procs[-1]

    monkeypatch.setattr(motion_mod.subprocess, "Popen", fake)
    return procs, threading


def test_a_stderr_flood_does_not_deadlock(monkeypatch, tmp_path):
    procs, threading = _fake_popen(monkeypatch, _FLOOD)
    out = []
    t = threading.Thread(target=lambda: out.append(frame_motion(tmp_path / "x.mp4")), daemon=True)
    t.start()
    t.join(30)
    try:
        assert not t.is_alive(), "frame_motion deadlocked on a full stderr pipe"
        assert out == [[0.0, 0.0]]  # 8 identical frames at 4 fps
    finally:
        for p in procs:
            p.kill()


def test_a_stuck_decode_times_out(monkeypatch, tmp_path):
    from hookline.media.ffmpeg import FFmpegError

    procs, _ = _fake_popen(monkeypatch, "import time; time.sleep(60)")
    try:
        with pytest.raises(FFmpegError, match="timed out"):
            frame_motion(tmp_path / "x.mp4", timeout=1.0)
    finally:
        for p in procs:
            p.kill()


def test_the_motion_timeout_scales_with_the_source_duration():
    from hookline.media.motion import MOTION_TIMEOUT_MIN_S, motion_timeout

    assert motion_timeout(0) == motion_timeout(60.0) == MOTION_TIMEOUT_MIN_S
    assert motion_timeout(10 * 3600.0) >= 5 * 3600.0  # a 10 h VOD gets hours, not 30 min


def test_frame_stats_reports_progress_in_decoded_seconds(flash_video):
    from hookline.media.motion import frame_stats

    done: list[float] = []
    frame_stats(flash_video, progress=done.append)
    assert done and done[-1] == pytest.approx(6.0, abs=0.5)
    assert done == sorted(done)
