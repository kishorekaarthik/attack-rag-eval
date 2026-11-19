"""One interface for every LLM. Nothing outside this file knows which vendor is in use.

Switching vendors is a config value (LLM_PROVIDER=gemini|claude), so the RAG
pipeline and the eval harness run unchanged against either model. That is
what makes a fair model-vs-model comparison possible later.
"""
import os
import time
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


def _is_timeout(exc: Exception) -> bool:
    """True for a timeout from any HTTP library.

    google-genai raises timeouts from its own bundled copy of httpx
    (httpx2.ReadTimeout), which is not an httpx class, so match by type name.
    """
    return isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(exc).__name__.lower()


class GeminiProvider:
    name = "gemini"
    # Seconds to wait after each rate-limit reply. The free tier allows 15
    # requests a minute, so eval runs of 100+ calls must pace themselves.
    RETRY_WAITS = (15, 30, 60, 60)
    RETRY_CODES = (429, 500, 503, 504)  # rate limited, internal error, overloaded, server deadline
    # Without a timeout a reply that never arrives blocks the eval forever (seen in practice).
    TIMEOUT_SECONDS = 120

    def __init__(self, api_key: str, model: str = config.GEMINI_MODEL, client=None, sleep=time.sleep,
                 clock=time.perf_counter):
        self.model = model
        self._sleep, self._clock = sleep, clock
        # Tokens and request time of the last successful call (retry waits excluded), for cost reporting.
        self.last_usage: dict | None = None
        if client is None:
            from google import genai  # imported here so tests and Claude-only runs don't need it
            from google.genai import types

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=self.TIMEOUT_SECONDS * 1000))
        self._client = client

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        from google.genai import types

        request = dict(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        for attempt, wait in enumerate((*self.RETRY_WAITS, None), start=1):
            start = self._clock()
            try:
                resp = self._client.models.generate_content(**request)
                seconds = self._clock() - start
                usage = getattr(resp, "usage_metadata", None)
                self.last_usage = {
                    "input_tokens": getattr(usage, "prompt_token_count", None) or 0,
                    "output_tokens": getattr(usage, "candidates_token_count", None) or 0,
                    "seconds": seconds, "attempts": attempt,
                }
                return resp.text or ""
            except Exception as exc:  # the SDK raises several unrelated error types
                self._clock()  # keep the clock sequence predictable: one reading at the end of every attempt
                retryable = getattr(exc, "code", None) in self.RETRY_CODES or _is_timeout(exc)
                if retryable and wait is not None:
                    self._sleep(wait)
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
        raise AssertionError("unreachable")


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


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    name = (name or config.LLM_PROVIDER).lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose one of: {', '.join(_PROVIDERS)}.")
    cls, env_var = _PROVIDERS[name]
    key = (os.getenv(env_var) or "").strip()
    if not key:
        raise MissingKeyError(f"{env_var} is not set. Add it to your .env file (see .env.example).")
    return cls(api_key=key, model=model) if model else cls(api_key=key)
