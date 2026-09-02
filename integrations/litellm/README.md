# LiteLLM Adapter

This is an optional adapter over the framework-agnostic `GuardrailClient`. Mount `src/guardrail_client.py`, `integrations/litellm/litellm_adapter.py`, and `config.yaml` in the LiteLLM Proxy. Install LiteLLM only in the Proxy environment.

Use `pre_call` for enforcement before a model provider receives input. `post_call` is deliberately not configured because the current classifier evaluates user prompts, not model output.