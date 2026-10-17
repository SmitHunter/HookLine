from hookline.config import Settings
from hookline.llm.anthropic_api import AnthropicApiBackend
from hookline.llm.base import LLMBackend, LLMError
from hookline.llm.claude_code import ClaudeCodeBackend

__all__ = ["AnthropicApiBackend", "ClaudeCodeBackend", "LLMBackend", "LLMError", "make_llm"]


def make_llm(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "api":
        return AnthropicApiBackend()
    return ClaudeCodeBackend()
