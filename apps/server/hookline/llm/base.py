from typing import Protocol


class LLMError(Exception):
    """Raised when a backend fails to produce a structured JSON answer."""


class LLMBackend(Protocol):
    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict: ...
