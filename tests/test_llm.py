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
