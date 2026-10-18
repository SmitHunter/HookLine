import subprocess

from typer.testing import CliRunner

from hookline import cli
from hookline.doctor import Check, parse_vulkaninfo, run_doctor


def fake_completed(stdout: str):
    return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")


def test_doctor_flags_missing_ffmpeg():
    checks = run_doctor(which=lambda name: None, run=fake_completed(""))
    assert any(c.name == "ffmpeg" and not c.ok and c.required for c in checks)


def test_doctor_all_good():
    out = (
        "ffmpeg version 7.1 Copyright (c) 2000-2024\n"
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
    monkeypatch.setattr(cli, "get_transcriber", lambda s, log=None: None)
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
    monkeypatch.setattr(cli, "get_transcriber", lambda s, log=None: None)
    res = CliRunner().invoke(cli.app, ["run", "x.mp4"])
    assert res.exit_code == 1 and "No clips were selected" in res.output
    assert "workspace/abc/select_raw.json" in res.output and "Traceback" not in res.output


def test_doctor_m2_asset_rows():
    checks = run_doctor(which=lambda name: f"/bin/{name}", run=fake_completed(""))
    by = {c.name: c for c in checks}
    assert by["Anton font"].required and by["Anton font"].ok
    assert not by["YuNet model"].required and not by["TwitchDownloaderCLI"].required
    assert "DTW on" in by["whisper.cpp"].detail or not by["whisper.cpp"].ok


VULKANINFO_IGPU_ONLY = """Devices:
========
GPU0:
	apiVersion         = 1.3.280
	vendorID           = 0x1002
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = AMD Radeon(TM) Graphics
GPU1:
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
"""

VULKANINFO_DISCRETE = (
    VULKANINFO_IGPU_ONLY
    + """GPU2:
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = AMD Radeon RX 7800 XT
"""
)


def test_parse_vulkaninfo():
    assert parse_vulkaninfo(VULKANINFO_DISCRETE) == [
        ("AMD Radeon(TM) Graphics", "INTEGRATED_GPU"),
        ("llvmpipe (LLVM 17.0.6, 256 bits)", "CPU"),
        ("AMD Radeon RX 7800 XT", "DISCRETE_GPU"),
    ]


def _gpu_row(which, text):
    checks = run_doctor(which=which, run=fake_completed(text))
    return next(c for c in checks if c.name == "GPU (whisper.cpp)")


def test_doctor_gpu_row_flags_missing_discrete_gpu():
    row = _gpu_row(lambda name: f"/bin/{name}", VULKANINFO_IGPU_ONLY)
    assert not row.ok and not row.required
    assert row.detail.startswith("discrete GPU not visible — check driver")
    assert "AMD Radeon(TM) Graphics" in row.detail


def test_doctor_gpu_row_ok_with_discrete_gpu():
    row = _gpu_row(lambda name: f"/bin/{name}", VULKANINFO_DISCRETE)
    assert row.ok and "AMD Radeon RX 7800 XT (discrete_gpu)" in row.detail


def test_doctor_gpu_row_without_vulkaninfo():
    row = _gpu_row(lambda name: None if name == "vulkaninfo" else f"/bin/{name}", "")
    assert not row.ok and not row.required and "vulkaninfo" in row.detail


def _versioned(version_line: str):
    tables = " .. subtitles  V->V  Render text subtitles\n V....D h264_nvenc  NVENC\n"
    return fake_completed(f"{version_line}\n{tables}")


def test_doctor_requires_ffmpeg_7_for_filter_scripts():
    """Renders pass their filter graph as a file (`-/filter_complex`, ffmpeg 7+)."""
    which = lambda name: f"/bin/{name}"
    old = {c.name: c for c in run_doctor(which=which, run=_versioned("ffmpeg version 6.1.1"))}
    assert not old["ffmpeg"].ok and old["ffmpeg"].required
    assert "6.1.1" in old["ffmpeg"].detail and "7" in old["ffmpeg"].detail
    for line in ("ffmpeg version 7.1-full_build-www.gyan.dev", "ffmpeg version n8.0"):
        assert {c.name: c for c in run_doctor(which=which, run=_versioned(line))}["ffmpeg"].ok
    # a git build ("N-118000-g...") has no release number: assumed current
    git = run_doctor(which=which, run=_versioned("ffmpeg version N-118000-gabc Copyright"))
    assert {c.name: c for c in git}["ffmpeg"].ok
