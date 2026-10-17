import sys
import types

import pytest
import yt_dlp.utils

from hookline.pipeline.ingest import IngestError, ingest


def test_ingest_local_file_links_or_copies(fixture_video, tmp_path, settings):
    job = tmp_path / "job"
    job.mkdir()
    dst = ingest(str(fixture_video), job, settings)
    assert dst == job / "source.mp4" and dst.stat().st_size == fixture_video.stat().st_size
    assert ingest(str(fixture_video), job, settings) == dst  # skipped


def test_ingest_ignores_partial_files(fixture_video, tmp_path, settings):
    job = tmp_path / "job"
    job.mkdir()
    (job / "source.mp4.part").write_bytes(b"x")
    dst = ingest(str(fixture_video), job, settings)
    assert dst.name == "source.mp4" and dst.stat().st_size == fixture_video.stat().st_size


def test_ingest_missing_raises(tmp_path, settings):
    with pytest.raises(FileNotFoundError):
        ingest(str(tmp_path / "nope.mp4"), tmp_path, settings)


def test_ingest_url_uses_ytdlp(tmp_path, settings, monkeypatch):
    seen = {}

    class FakeYDL:
        def __init__(self, opts):
            seen["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            seen["urls"] = urls
            out = tmp_path / "download.tmp"
            assert not (out / "source.f137.mp4.stale").exists()
            out.mkdir(exist_ok=True)
            (out / "source.f137.mp4").write_bytes(b"vonly")
            (out / "source.mp4.part").write_bytes(b"x")
            (out / "source.mp4").write_bytes(b"video")

    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL, utils=yt_dlp.utils)
    )
    s = settings.model_copy(update={"cookies_from_browser": "firefox"})
    dst = ingest("https://example.com/v/1", tmp_path, s)
    assert dst == tmp_path / "source.mp4" and seen["urls"] == ["https://example.com/v/1"]
    assert seen["opts"]["merge_output_format"] == "mp4"
    assert seen["opts"]["noplaylist"] is True
    assert not (tmp_path / "download.tmp").exists()
    assert [p.name for p in tmp_path.glob("source*")] == ["source.mp4"]
    assert seen["opts"]["cookiesfrombrowser"] == ("firefox",)


def test_ingest_stale_intermediate_is_redownloaded(tmp_path, settings, monkeypatch):
    stale = tmp_path / "download.tmp"
    stale.mkdir()
    (stale / "source.f137.mp4.stale").write_bytes(b"old")
    (stale / "source.f137.mp4").write_bytes(b"old")
    called = []

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            called.append(1)
            (tmp_path / "download.tmp").mkdir(exist_ok=True)
            (tmp_path / "download.tmp" / "source.mp4").write_bytes(b"video")

    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL, utils=yt_dlp.utils)
    )
    dst = ingest("https://example.com/v/1", tmp_path, settings)
    assert called and dst.read_bytes() == b"video"


def test_ingest_download_error_wrapped(tmp_path, settings, monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            raise yt_dlp.utils.DownloadError("boom")

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
    with pytest.raises(IngestError):
        ingest("https://example.com/v/1", tmp_path, settings)
