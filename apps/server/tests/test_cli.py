import pytest
from typer.testing import CliRunner

from hookline import cli
from hookline.config import parse_facecam


@pytest.mark.parametrize(
    "bad", ["1,2", "a,b,c,d", "0.5,0.5,0.6,0.1", "0,0,0,0.2", "-0.1,0,0.2,0.2"]
)
def test_cli_bad_facecam_exits_1(bad):
    res = CliRunner().invoke(cli.app, ["run", "x.mp4", "--facecam", bad])
    assert res.exit_code == 1 and "--facecam" in res.output and "Traceback" not in res.output
    assert len([ln for ln in res.output.splitlines() if ln.strip()]) == 1


def test_cli_flags_reach_settings(monkeypatch):
    seen = {}

    def fake_run_job(source, settings, llm, transcriber, log):
        seen["s"] = settings
        return []

    monkeypatch.setattr(cli, "run_job", fake_run_job)
    monkeypatch.setattr(cli, "make_llm", lambda s: None)
    monkeypatch.setattr(cli, "get_transcriber", lambda s, log=None: None)
    args = ["run", "x.mp4", "--horizontal", "--no-captions", "--no-chat"]
    res = CliRunner().invoke(cli.app, [*args, "--facecam", "0.7,0,0.3,0.25"])
    assert res.exit_code == 0, res.output
    s = seen["s"]
    assert (s.vertical, s.captions, s.chat) == (False, False, False)
    assert s.facecam == (0.7, 0.0, 0.3, 0.25)


def test_parse_facecam_ok():
    assert parse_facecam(" 0.1, 0.2 ,0.3,0.4") == (0.1, 0.2, 0.3, 0.4)


def test_facecam_from_env(monkeypatch):
    from hookline.config import Settings

    monkeypatch.setenv("HOOKLINE_FACECAM", "0.7,0,0.3,0.25")
    monkeypatch.setenv("HOOKLINE_VERTICAL", "false")
    s = Settings.from_env()
    assert s.facecam == (0.7, 0.0, 0.3, 0.25) and s.vertical is False


def test_log_survives_non_utf8_stdout(monkeypatch):
    import io
    import sys

    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    cli._safe_streams()
    cli.console.print("clip 01: layouts=[FIT_BLUR] removed=1.0s → a.mp4", markup=False)
    stream.flush()
    assert b"removed=1.0s ? a.mp4" in buf.getvalue()
