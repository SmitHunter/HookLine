import http.server
import threading
from pathlib import Path

import pytest

from hookline import assets
from hookline.assets import AssetError, fetch, tools_dir


@pytest.fixture
def local_http_server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    (root / "blob.bin").write_bytes(b"hello-bytes" * 1000)
    counts = {"n": 0}

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def do_GET(self):
            counts["n"] += 1
            super().do_GET()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", counts
    srv.shutdown()
    srv.server_close()


def test_tools_dir_under_repo_root():
    assert tools_dir().name == "tools"


def test_fetch_is_atomic_and_cached(tmp_path, local_http_server):
    base, counts = local_http_server
    dst = tmp_path / "out" / "blob.bin"
    assert fetch(f"{base}/blob.bin", dst) == dst
    assert dst.read_bytes() == b"hello-bytes" * 1000
    assert counts["n"] == 1
    assert fetch(f"{base}/blob.bin", dst) == dst
    assert counts["n"] == 1
    assert not list(dst.parent.glob("*.part"))


def test_fetch_http_error_raises_and_leaves_no_part(tmp_path, local_http_server):
    base, _ = local_http_server
    dst = tmp_path / "missing.bin"
    with pytest.raises(AssetError):
        fetch(f"{base}/nope.bin", dst)
    assert not dst.exists()
    assert not list(tmp_path.glob("*.part"))


def test_fetch_connection_refused_raises(tmp_path):
    with pytest.raises(AssetError):
        fetch("http://127.0.0.1:1/x.bin", tmp_path / "x.bin")


def test_ensure_twitch_downloader_non_windows(monkeypatch):
    monkeypatch.setattr(assets.sys, "platform", "linux")
    with pytest.raises(AssetError, match="Windows-only"):
        assets.ensure_twitch_downloader()


def test_ensure_twitch_downloader_extracts_exe(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setattr(assets.sys, "platform", "win32")
    monkeypatch.setattr(assets, "tools_dir", lambda: tmp_path)

    def fake_fetch(url, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("../evil/TwitchDownloaderCLI.exe", b"MZ")
        return dst

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    exe = assets.ensure_twitch_downloader()
    assert exe == tmp_path / "bin" / "TwitchDownloaderCLI.exe"
    assert exe.read_bytes() == b"MZ"


def test_fetch_midstream_failure_removes_part(tmp_path):
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n" + b"x" * 10)
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    dst = tmp_path / "m.bin"
    with pytest.raises(AssetError):
        fetch(f"http://127.0.0.1:{srv.getsockname()[1]}/m.bin", dst)
    srv.close()
    assert not dst.exists()
    assert not list(tmp_path.glob("*.part"))


def test_ensure_twitch_downloader_corrupt_zip_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.sys, "platform", "win32")
    monkeypatch.setattr(assets, "tools_dir", lambda: tmp_path)

    def fake_fetch(url, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("<html>not a zip</html>")
        return dst

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    with pytest.raises(AssetError):
        assets.ensure_twitch_downloader()
    assert not list((tmp_path / "downloads").glob("*.zip"))
