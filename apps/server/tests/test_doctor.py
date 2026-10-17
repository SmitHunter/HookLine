import subprocess

from typer.testing import CliRunner

from hookline import cli
from hookline.doctor import Check, run_doctor


def fake_completed(stdout: str):
    return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")


def test_doctor_flags_missing_ffmpeg():
    checks = run_doctor(which=lambda name: None, run=fake_completed(""))
    assert any(c.name == "ffmpeg" and not c.ok and c.required for c in checks)


def test_doctor_all_good():
    out = (
        " .. subtitles  V->V  Render text subtitles\n"
        " V....D h264_nvenc  NVENC\n"
        " V....D h264_amf  AMF\n"
    )
    checks = run_doctor(which=lambda name: f"/bin/{name}", run=fake_completed(out))
    by = {c.name: c for c in checks}
    assert by["ffmpeg"].ok and by["libass"].ok and by["claude CLI"].ok
    assert by["h264_nvenc"].ok and not by["h264_nvenc"].required


def test_cli_doctor_exit_codes(monkeypatch):
    r = CliRunner()
    monkeypatch.setattr(cli, "run_doctor", lambda: [Check("ffmpeg", False, True, "missing")])
    res = r.invoke(cli.app, ["doctor"])
    assert res.exit_code == 1 and "ffmpeg" in res.output
    monkeypatch.setattr(cli, "run_doctor", lambda: [Check("ffmpeg", True, True, "ok")])
    assert r.invoke(cli.app, ["doctor"]).exit_code == 0


def test_cli_run_missing_file(tmp_path):
    res = CliRunner().invoke(cli.app, ["run", str(tmp_path / "nope.mp4"), "--out", str(tmp_path)])
    assert res.exit_code == 1 and "nope.mp4" in res.output and "Traceback" not in res.output


def test_cli_run_ingest_error(monkeypatch, tmp_path):
    from hookline.pipeline.ingest import IngestError

    def boom(*a, **k):
        raise IngestError("download failed")

    monkeypatch.setattr(cli, "run_job", boom)
    monkeypatch.setattr(cli, "make_llm", lambda s: None)
    monkeypatch.setattr(cli, "get_transcriber", lambda s: None)
    res = CliRunner().invoke(cli.app, ["run", "https://x.test/v"])
    assert res.exit_code == 1 and "download failed" in res.output
    assert "Traceback" not in res.output


def test_cli_run_bad_backend():
    res = CliRunner().invoke(cli.app, ["run", "x.mp4", "--backend", "nope"])
    assert res.exit_code == 1 and "Traceback" not in res.output


def test_doctor_settings_failure_is_check(monkeypatch):
    monkeypatch.setenv("HOOKLINE_CLIPS", "notanint")
    checks = run_doctor(which=lambda n: None, run=fake_completed(""))
    assert any(c.name == "settings" and not c.ok and c.required for c in checks)


def test_cli_run_no_clips_selected(monkeypatch):
    from hookline.pipeline.run import NoClipsSelectedError

    def boom(*a, **k):
        raise NoClipsSelectedError("No clips were selected — see workspace/abc/select_raw.json")

    monkeypatch.setattr(cli, "run_job", boom)
    monkeypatch.setattr(cli, "make_llm", lambda s: None)
    monkeypatch.setattr(cli, "get_transcriber", lambda s: None)
    res = CliRunner().invoke(cli.app, ["run", "x.mp4"])
    assert res.exit_code == 1 and "No clips were selected" in res.output
    assert "workspace/abc/select_raw.json" in res.output and "Traceback" not in res.output
