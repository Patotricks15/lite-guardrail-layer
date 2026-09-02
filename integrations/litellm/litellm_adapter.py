"""Optional LiteLLM adapter for the framework-agnostic GuardrailClient."""

from typing import Any, Literal

from guardrail_client import GuardrailClient
from litellm.integrations.custom_guardrail import CustomGuardrail


class LiteGuardrail(CustomGuardrail):
    def __init__(self, api_base: str = "http://lite-guardrail-api:8000", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = GuardrailClient(api_base)

    async def apply_guardrail(
        self,
        inputs: dict[str, Any],
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
        logging_obj: Any = None,
    ) -> dict[str, Any]:
        if input_type == "request":
            await self.client.enforce(inputs.get("structured_messages", []))
        return inputs