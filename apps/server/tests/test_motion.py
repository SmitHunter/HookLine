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
