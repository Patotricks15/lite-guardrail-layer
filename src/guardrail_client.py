"""Framework-agnostic client for Lite Guardrail Layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


Message = Mapping[str, Any]


class GuardrailBlockedError(ValueError):
    """Raised when a message set violates a Lite Guardrail policy."""


class GuardrailClient:
    """Evaluate OpenAI-style messages independently of an LLM framework."""

    def __init__(self, api_base: str = "http://localhost:8000", timeout: float = 5.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    async def evaluate(self, messages: Sequence[Message]) -> dict[str, Any]:
        system_prompt, user_prompt = _prompts(messages)
        if not user_prompt:
            return {"decision": "safe", "skipped": True}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base}/v1/predict/base",
                json={"system_prompt": system_prompt or "You are a helpful assistant.", "user_prompt": user_prompt},
            )
            response.raise_for_status()
            return response.json()

    async def enforce(self, messages: Sequence[Message]) -> dict[str, Any]:
        """Return the verdict or raise when the messages must not reach an LLM."""
        verdict = await self.evaluate(messages)
        if verdict.get("decision") == "blocked":
            raise GuardrailBlockedError("Lite Guardrail Layer blocked this request")
        return verdict


def _prompts(messages: Sequence[Message]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        content = _text(message.get("content"))
        if message.get("role") == "system":
            system_parts.append(content)
        elif message.get("role") == "user":
            user_parts.append(content)
    return "\n".join(system_parts).strip(), "\n".join(user_parts).strip()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            _text(part.get("text", "")) for part in content if isinstance(part, Mapping)
        )
    return ""