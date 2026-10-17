import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from hookline.llm.base import LLMError


def _tail(text: str | None, n: int = 500) -> str:
    return (text or "").strip()[-n:]


def _result_detail(payload: dict) -> str:
    result = payload.get("result")
    return _tail(result if isinstance(result, str) else "") or "no result text"


class ClaudeCodeBackend:
    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout_s: int = 900,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.runner = runner
        self.timeout_s = timeout_s
        self.which = which

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict:
        exe = self.which("claude")
        if not exe:
            raise LLMError("claude CLI not found on PATH")
        args = [
            exe,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
            "--model",
            model,
            "--system-prompt-file",
            "",
            "--tools",
            "",
        ]
        with tempfile.TemporaryDirectory() as cwd:
            sys_file = Path(cwd) / "system-prompt.txt"
            sys_file.write_text(system, encoding="utf-8")
            args[args.index("--system-prompt-file") + 1] = str(sys_file)
            try:
                res = self.runner(
                    args,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_s,
                    input=prompt,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired as e:
                raise LLMError(f"claude CLI timed out after {self.timeout_s}s") from e
            except OSError as e:
                raise LLMError(f"claude CLI could not be run: {e}") from e
        if res.returncode != 0:
            raise LLMError(
                f"claude CLI exited with code {res.returncode}: "
                f"{_tail(res.stderr) or _tail(res.stdout)}"
            )
        try:
            payload = json.loads(res.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude CLI returned non-JSON output: {_tail(res.stdout)}") from e
        if not isinstance(payload, dict):
            raise LLMError(f"claude CLI returned unexpected output: {_tail(res.stdout)}")
        if payload.get("is_error"):
            raise LLMError(f"claude CLI reported an error: {_result_detail(payload)}")
        out = payload.get("structured_output")
        if not isinstance(out, dict):
            raise LLMError(
                f"claude CLI returned no structured_output object: {_result_detail(payload)}"
            )
        return out
