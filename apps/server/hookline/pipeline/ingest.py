import hashlib
import os
import re
import shutil
from pathlib import Path

from hookline.config import Settings

_PARTIAL_MARKERS = (".part", ".ytdl", ".tmp")
_FORMAT_PART = re.compile(r"\.f\d+\.")


class IngestError(Exception):
    """Downloading the source failed."""


def _is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def job_id_for(source: str) -> str:
    """Stable 10-hex id: sha1 of the URL, or of abs path + size + mtime for a local file."""
    if not Path(source).exists():
        if _is_url(source):
            return hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
        raise FileNotFoundError(f"No such file or URL: {source}")
    p = Path(source).resolve()
    st = p.stat()
    key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def find_source(job_dir: Path) -> Path | None:
    """The finished `source.*` file (top level only), ignoring partial/intermediate files."""
    for p in sorted(job_dir.glob("source.*")):
        if (
            p.is_file()
            and not any(m in p.name for m in _PARTIAL_MARKERS)
            and not _FORMAT_PART.search(p.name)
        ):
            return p
    return None


def ingest(source: str, job_dir: Path, settings: Settings) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    existing = find_source(job_dir)
    if existing:
        return existing

    if Path(source).exists():
        src = Path(source).resolve()
        dst = job_dir / f"source{src.suffix.lower() or '.mp4'}"
        tmp = job_dir / (dst.name + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    if not _is_url(source):
        raise FileNotFoundError(f"No such file or URL: {source}")

    import yt_dlp
    import yt_dlp.utils

    tmp_dir = job_dir / "download.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()
    opts: dict = {
        "outtmpl": str(tmp_dir / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
    }
    if settings.cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([source])
    except yt_dlp.utils.DownloadError as e:
        raise IngestError(f"Download failed for {source}: {e}") from e
    found = find_source(tmp_dir)
    if not found:
        raise IngestError(f"yt-dlp produced no merged file for {source}")
    dst = job_dir / f"source{found.suffix.lower()}"
    os.replace(found, dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return dst
