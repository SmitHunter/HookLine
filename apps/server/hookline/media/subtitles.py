from pathlib import Path

from hookline.config import repo_root


def fonts_dir() -> Path:
    return repo_root() / "assets" / "fonts"


def _esc(path: Path) -> str:
    p = path.resolve().as_posix() if not path.is_absolute() else path.as_posix()
    return p.replace(":", "\\:").replace("'", "'\\\\\\''")


def subtitles_filter(ass_path: Path, fonts_dir: Path) -> str:
    """ffmpeg `subtitles` filter string (built for argv use, no shell)."""
    return f"subtitles='{_esc(ass_path)}':fontsdir='{_esc(fonts_dir)}'"
