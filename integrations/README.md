# Framework Integrations

The core integration contract is framework-neutral: pass OpenAI-style messages to `GuardrailClient.enforce(messages)` before sending them to an LLM. It returns the classifier verdict or raises `GuardrailBlockedError`.

## LiteLLM

Install the optional extra with `pip install litellm`, mount `src/guardrail_client.py` and `integrations/litellm/litellm_adapter.py` in the LiteLLM Proxy, then use `litellm_adapter.LiteGuardrail` from `integrations/litellm/config.yaml`.

## LangChain and LangGraph

Call `await client.enforce(messages)` in a LangChain runnable before the model, or in a LangGraph node immediately before a model node. Both frameworks expose message role and content, which map directly to this contract.

## Other clients

Use the same client around OpenAI SDK, Vercel AI SDK, direct provider clients, or custom HTTP calls. No framework dependency is required by the service or core client.