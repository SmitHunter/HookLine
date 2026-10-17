import http.client
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from hookline.config import repo_root

TWITCH_DL_VERSION = "1.56.5"
TWITCH_DL_URL = (
    "https://github.com/lay295/TwitchDownloader/releases/download/"
    f"{TWITCH_DL_VERSION}/TwitchDownloaderCLI-{TWITCH_DL_VERSION}-Windows-x64.zip"
)
TWITCH_DL_EXE = "TwitchDownloaderCLI.exe"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
_TIMEOUT_S = 60
_CHUNK = 1024 * 256


class AssetError(Exception):
    pass


def tools_dir() -> Path:
    return repo_root() / "tools"


def fetch(url: str, dst: Path) -> Path:
    """Download url to dst atomically (via .part). Returns dst; cached if it exists."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp, open(part, "wb") as f:
            shutil.copyfileobj(resp, f, _CHUNK)
            expected = resp.headers.get("Content-Length")
            if expected is not None and f.tell() != int(expected):
                raise OSError(f"truncated download ({f.tell()} of {expected} bytes)")
        os.replace(part, dst)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        part.unlink(missing_ok=True)
        raise AssetError(f"download failed for {url}: {e}") from e
    return dst


def ensure_twitch_downloader() -> Path:
    if sys.platform != "win32":
        raise AssetError("TwitchDownloaderCLI auto-install is Windows-only")
    tools = tools_dir()
    exe = tools / "bin" / TWITCH_DL_EXE
    if exe.exists():
        return exe
    zip_path = fetch(TWITCH_DL_URL, tools / "downloads" / Path(TWITCH_DL_URL).name)
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(exe.suffix + ".part")
    try:
        with zipfile.ZipFile(zip_path) as z:
            member = next(
                (n for n in z.namelist() if Path(n.replace("\\", "/")).name == TWITCH_DL_EXE),
                None,
            )
            if member is None:
                raise AssetError(f"{TWITCH_DL_EXE} not found in {zip_path.name}")
            with z.open(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
        os.replace(tmp, exe)
    except zipfile.BadZipFile as e:
        tmp.unlink(missing_ok=True)
        zip_path.unlink(missing_ok=True)  # corrupt cache: re-download next run
        raise AssetError(f"corrupt download {zip_path.name}: {e}") from e
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise AssetError(f"could not extract {TWITCH_DL_EXE}: {e}") from e
    return exe


def ensure_yunet() -> Path:
    """YuNet face-detection model (MIT, ~230 KB), cached under tools/models/."""
    return fetch(YUNET_URL, tools_dir() / "models" / Path(YUNET_URL).name)
