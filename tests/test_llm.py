import json

import httpx
import pytest

from copilot import config
from copilot.llm import ClaudeProvider, GeminiProvider, LLMError, MissingKeyError, get_provider


def test_unknown_provider_name_is_rejected():
    with pytest.raises(ValueError, match="mistral"):
        get_provider("mistral")


@pytest.mark.parametrize("name,env_var", [("gemini", "GEMINI_API_KEY"), ("claude", "ANTHROPIC_API_KEY")])
def test_missing_key_error_names_the_env_var(monkeypatch, name, env_var):
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(MissingKeyError, match=env_var):
        get_provider(name)


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(MissingKeyError):
        get_provider("gemini")


def test_default_provider_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert get_provider().name == "claude"


def test_provider_reports_its_pinned_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    provider = get_provider("gemini")
    assert (provider.name, provider.model) == ("gemini", config.GEMINI_MODEL)


def claude_with(handler):
    return ClaudeProvider(api_key="secret-key", model="claude-test",
                          client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_claude_sends_messages_request_and_returns_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "It is "}, {"type": "text", "text": "T1059."}]})

    assert claude_with(handler).generate("be brief", "what is it?", temperature=0.0) == "It is T1059."
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-test"
    assert seen["body"]["system"] == "be brief"
    assert seen["body"]["messages"] == [{"role": "user", "content": "what is it?"}]
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] > 0


def test_claude_api_error_is_readable_and_hides_the_key():
    def handler(request):
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})

    with pytest.raises(LLMError) as err:
        claude_with(handler).generate("s", "p")
    assert "401" in str(err.value) and "invalid x-api-key" in str(err.value)
    assert "secret-key" not in str(err.value)


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"text": "answer"})()


def test_gemini_passes_model_system_prompt_and_temperature():
    models = FakeGeminiModels()
    client = type("Client", (), {"models": models})()
    provider = GeminiProvider(api_key="k", model="gemini-test", client=client)
    assert provider.generate("be brief", "what is it?", temperature=0.2) == "answer"
    [call] = models.calls
    assert call["model"] == "gemini-test"
    assert call["contents"] == "what is it?"
    assert call["config"].system_instruction == "be brief"
    assert call["config"].temperature == 0.2


def test_gemini_empty_reply_becomes_empty_string():
    models = FakeGeminiModels()
    models.generate_content = lambda **kw: type("Resp", (), {"text": None})()
    provider = GeminiProvider(api_key="k", model="m", client=type("Client", (), {"models": models})())
    assert provider.generate("s", "p") == ""


class RateLimited(Exception):
    code = 429


def flaky_gemini(failures, error=RateLimited("quota")):
    models = FakeGeminiModels()
    state = {"left": failures}

    def generate_content(**kwargs):
        models.calls.append(kwargs)
        if state["left"] > 0:
            state["left"] -= 1
            raise error
        return type("Resp", (), {"text": "answer"})()

    models.generate_content = generate_content
    waits = []
    provider = GeminiProvider(api_key="k", model="m", client=type("Client", (), {"models": models})(), sleep=waits.append)
    return provider, models, waits


def test_gemini_waits_and_retries_when_rate_limited():
    provider, models, waits = flaky_gemini(failures=2)
    assert provider.generate("s", "p") == "answer"
    assert len(models.calls) == 3
    assert len(waits) == 2 and waits[0] > 0 and waits[1] > waits[0]


def test_gemini_gives_up_after_repeated_rate_limits():
    provider, models, waits = flaky_gemini(failures=99)
    with pytest.raises(LLMError, match="quota"):
        provider.generate("s", "p")
    assert len(models.calls) == 5


def test_gemini_does_not_retry_other_errors():
    provider, models, waits = flaky_gemini(failures=1, error=ValueError("bad request"))
    with pytest.raises(LLMError, match="bad request"):
        provider.generate("s", "p")
    assert len(models.calls) == 1 and waits == []


class Overloaded(Exception):
    code = 503


def test_gemini_retries_when_the_service_is_temporarily_overloaded():
    provider, models, waits = flaky_gemini(failures=1, error=Overloaded("high demand"))
    assert provider.generate("s", "p") == "answer"
    assert len(models.calls) == 2 and len(waits) == 1


def test_gemini_records_tokens_and_request_time_excluding_retry_waits():
    usage = type("Usage", (), {"prompt_token_count": 1200, "candidates_token_count": 40})()
    models = FakeGeminiModels()
    state = {"fail": 1}

    def generate_content(**kwargs):
        if state["fail"]:
            state["fail"] -= 1
            raise RateLimited("quota")
        return type("Resp", (), {"text": "T1001", "usage_metadata": usage})()

    models.generate_content = generate_content
    ticks = iter([0.0, 0.5, 10.0, 10.8])          # failed attempt 0.5 s, wait, successful attempt 0.8 s
    provider = GeminiProvider(api_key="k", model="m", client=type("Client", (), {"models": models})(),
                              sleep=lambda s: None, clock=lambda: next(ticks))
    provider.generate("s", "p")
    assert provider.last_usage == {"input_tokens": 1200, "output_tokens": 40, "seconds": pytest.approx(0.8), "attempts": 2}
