import json
import subprocess
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from hookline.config import Settings
from hookline.llm import AnthropicApiBackend, ClaudeCodeBackend, LLMError, make_llm


def fake_run(stdout, code=0, stderr=""):
    calls = []

    def run(args, **kw):
        calls.append((args, kw))
        return subprocess.CompletedProcess(args, code, stdout, stderr)

    return run, calls


def capturing_run(stdout):
    seen = {}

    def run(args, **kw):
        seen["args"] = args
        i = args.index("--system-prompt-file")
        with open(args[i + 1], encoding="utf-8") as f:
            seen["system"] = f.read()
        return subprocess.CompletedProcess(args, 0, stdout, "")

    return run, seen


def which_ok(name):
    return "C:/bin/claude.cmd"


def test_claude_code_sends_prompt_on_stdin_and_returns_structured_output():
    run, calls = fake_run(json.dumps({"is_error": False, "structured_output": {"clips": []}}))
    out = ClaudeCodeBackend(runner=run, which=which_ok).complete_json(
        "sys", "BIG PROMPT", {"type": "object"}, "claude-opus-5-5"
    )
    args, kw = calls[0]
    assert out == {"clips": []}
    assert kw["input"] == "BIG PROMPT" and "BIG PROMPT" not in args
    assert kw["encoding"] == "utf-8" and kw["cwd"]
    assert args[0] == "C:/bin/claude.cmd"
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--tools") + 1] == ""


@pytest.mark.parametrize(
    "stdout,code",
    [("not json", 0), (json.dumps({"is_error": True, "result": "x"}), 0), ("{}", 1)],
)
def test_claude_code_errors(stdout, code):
    run, _ = fake_run(stdout, code)
    with pytest.raises(LLMError):
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "p", {}, "m")


def test_claude_code_missing_structured_output():
    run, _ = fake_run(json.dumps({"is_error": False, "result": "hi"}))
    with pytest.raises(LLMError):
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "p", {}, "m")


def test_claude_code_cli_missing():
    run, _ = fake_run("{}")
    with pytest.raises(LLMError, match="not found on PATH"):
        ClaudeCodeBackend(runner=run, which=lambda n: None).complete_json("s", "p", {}, "m")


def test_claude_code_timeout_becomes_llm_error():
    def run(args, **kw):
        raise subprocess.TimeoutExpired(args, 1)

    with pytest.raises(LLMError, match="timed out"):
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "SECRET", {}, "m")


def test_claude_code_error_detail_excludes_prompt():
    run, _ = fake_run("", 2, stderr="boom happened")
    with pytest.raises(LLMError, match="boom happened") as ei:
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "SECRET", {}, "m")
    assert "SECRET" not in str(ei.value)


def test_api_backend_refusal_raises(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="refusal", text="")
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "claude-opus-5-5")


def test_api_backend_max_tokens_raises(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="max_tokens", text="{")
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m")


def test_api_backend_bad_json_raises(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="end_turn", text="nope")
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m")


def test_api_backend_parses_text(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="end_turn", text='{"clips": [1]}')
    assert AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m") == {"clips": [1]}


def test_api_backend_request_shape(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(text="{}")
    AnthropicApiBackend(client=client).complete_json("sys", "p", {"type": "object"}, "claude-x")
    kw = client.calls[0]
    assert kw["model"] == "claude-x"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["format"]["schema"] == {"type": "object"}
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw and "temperature" not in kw


def test_api_backend_construction_without_key_does_not_raise(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = AnthropicApiBackend()
    assert backend._client is None


def test_make_llm_selects_backend():
    assert isinstance(make_llm(Settings(llm_backend="claude-code")), ClaudeCodeBackend)
    assert isinstance(make_llm(Settings(llm_backend="api")), AnthropicApiBackend)


def test_claude_code_system_prompt_goes_via_file():
    system = "line1\n<angle> & %PATH% | x\nline3 é"
    run, seen = capturing_run(json.dumps({"structured_output": {"a": 1}}))
    out = ClaudeCodeBackend(runner=run, which=which_ok).complete_json(
        system, "p", {"type": "object", "properties": {}}, "m"
    )
    assert out == {"a": 1}
    args = seen["args"]
    assert "--system-prompt" not in args
    assert system not in args
    assert seen["system"] == system
    schema_arg = args[args.index("--json-schema") + 1]
    assert schema_arg == '{"type":"object","properties":{}}'


def test_claude_code_is_error_includes_result_text():
    run, _ = fake_run(json.dumps({"is_error": True, "result": "rate limited"}))
    with pytest.raises(LLMError, match="rate limited"):
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "p", {}, "m")


def test_claude_code_no_literal_none_in_message():
    run, _ = fake_run(json.dumps({"is_error": False}))
    with pytest.raises(LLMError) as ei:
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "p", {}, "m")
    assert "None" not in str(ei.value)


def test_claude_code_structured_output_must_be_dict():
    run, _ = fake_run(json.dumps({"structured_output": [1, 2]}))
    with pytest.raises(LLMError):
        ClaudeCodeBackend(runner=run, which=which_ok).complete_json("s", "p", {}, "m")


def test_api_backend_json_must_be_object(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(text="[1]")
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m")


def test_api_backend_no_text_block_raises():
    from contextlib import contextmanager

    @contextmanager
    def stream(**kw):
        msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking")])
        yield SimpleNamespace(get_final_message=lambda: msg)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    with pytest.raises(LLMError, match="no text"):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m")


@pytest.mark.parametrize("kind", ["status", "connection"])
def test_api_backend_sdk_errors_become_llm_error(kind):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def stream(**kw):
        if kind == "status":
            raise anthropic.APIStatusError(
                "bad", response=httpx.Response(500, request=req), body=None
            )
        raise anthropic.APIConnectionError(request=req)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m")
