import json

import anthropic

from hookline.llm.base import LLMError


class AnthropicApiBackend:
    def __init__(self, client: anthropic.Anthropic | None = None, effort: str = "high") -> None:
        self._client = client
        self.effort = effort

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict:
        try:
            with self.client.beta.messages.stream(
                model=model,
                max_tokens=64000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                msg = stream.get_final_message()
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        if msg.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"Anthropic API stopped early: {msg.stop_reason}")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Anthropic API returned no text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Anthropic API returned invalid JSON: {text[-200:]}") from e
        if not isinstance(data, dict):
            raise LLMError(f"Anthropic API returned JSON that is not an object: {text[-200:]}")
        return data
