"""Drives the real Streamlit app headlessly; only the LLM and the index are replaced."""
import pytest
from streamlit.testing.v1 import AppTest

from copilot import config, index, llm
from copilot.index import Hit

HITS = [
    Hit(chunk_id="T1486:description:0", technique_id="T1486", name="Data Encrypted for Impact",
        section="description", text="T1486 Data Encrypted for Impact (tactics: impact)\nFiles are encrypted.", score=0.71),
]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def generate(self, system, prompt, temperature=0.0):
        return "This is ransomware encryption [T1486]."


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(index, "index_exists", lambda *a, **k: True)
    monkeypatch.setattr(index, "vector_search", lambda *a, **kw: HITS)
    monkeypatch.setattr(config, "APP_RETRIEVER", "vector")
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30)


def test_question_gets_answer_with_citation_and_sources(app, monkeypatch):
    monkeypatch.setattr(llm, "get_provider", lambda name=None: FakeProvider())
    app.run()
    app.chat_input[0].set_value("files were encrypted").run()

    assert not app.exception
    messages = [m.markdown[0].value for m in app.chat_message]
    assert messages[0] == "files were encrypted"
    assert "This is ransomware encryption [T1486]." in messages[1]
    page = " ".join(el.value for el in app.markdown)
    assert "T1486" in page and "Data Encrypted for Impact" in page
    assert "attack.mitre.org/techniques/T1486" in page


def test_missing_key_shows_instructions_instead_of_crashing(app, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("copilot.config.LLM_PROVIDER", "gemini")
    app.run()

    assert not app.exception
    assert any("GEMINI_API_KEY" in e.value for e in app.error)
    assert len(app.chat_input) == 0 or app.chat_input[0].disabled
