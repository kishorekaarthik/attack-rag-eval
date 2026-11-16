"""One interface for every LLM. Nothing outside this file knows which vendor is in use.

Switching vendors is a config value (LLM_PROVIDER=gemini|claude), so the RAG
pipeline and the eval harness run unchanged against either model. That is
what makes a fair model-vs-model comparison possible later.
"""
import os
from typing import Protocol

import httpx

from copilot import config


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str: ...


class MissingKeyError(RuntimeError):
    """The API key for the selected provider is not set."""


class LLMError(RuntimeError):
    """The provider returned an error."""


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = config.GEMINI_MODEL, client=None):
        self.model = model
        if client is None:
            from google import genai  # imported here so tests and Claude-only runs don't need it

            client = genai.Client(api_key=api_key)
        self._client = client

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        from google.genai import types

        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
            )
        except Exception as exc:  # the SDK raises several unrelated error types
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return resp.text or ""


class ClaudeProvider:
    """Calls the Anthropic Messages API over plain HTTP.

    No SDK on purpose: the SDK pulls in a compiled dependency that Windows
    Smart App Control blocked on the development machine, and the API is one
    POST request.
    """
    name = "claude"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = config.CLAUDE_MODEL, client: httpx.Client | None = None):
        self.model = model
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        resp = self._client.post(
            self.URL,
            headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code != 200:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text[:200]
            raise LLMError(f"Claude request failed ({resp.status_code}): {detail}")
        return "".join(block.get("text", "") for block in resp.json()["content"] if block.get("type") == "text")


_PROVIDERS = {
    "gemini": (GeminiProvider, "GEMINI_API_KEY"),
    "claude": (ClaudeProvider, "ANTHROPIC_API_KEY"),
}


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or config.LLM_PROVIDER).lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose one of: {', '.join(_PROVIDERS)}.")
    cls, env_var = _PROVIDERS[name]
    key = (os.getenv(env_var) or "").strip()
    if not key:
        raise MissingKeyError(f"{env_var} is not set. Add it to your .env file (see .env.example).")
    return cls(api_key=key)
