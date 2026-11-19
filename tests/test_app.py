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
    import streamlit as st

    st.cache_resource.clear()  # fresh daily counter for every test
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


def test_app_serves_the_final_pipeline_by_default():
    from copilot.retrieval import PIPELINES

    assert config.APP_RETRIEVER == "hybrid_proc_llm"
    assert PIPELINES[config.APP_RETRIEVER] == dict(corpus="procedures", bm25=True, llm_rerank=True)


def test_first_start_builds_the_index_the_pipeline_searches(monkeypatch):
    built = {}
    monkeypatch.setattr(index, "index_exists", lambda *a, **k: False)
    monkeypatch.setattr(index, "build_index", lambda chunks, **kw: built.update(kw, n=len(chunks),
                        sections={c.section for c in chunks}))
    monkeypatch.setattr(config, "APP_RETRIEVER", "hybrid_proc_llm")
    monkeypatch.setattr(llm, "get_provider", lambda name=None: FakeProvider())
    import streamlit as st

    st.cache_resource.clear()  # prepare_index() is cached across AppTest runs in one process
    app = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    assert built["corpus"] == "procedures" and "procedures" in built["sections"]


def test_api_key_can_come_from_streamlit_secrets(app, monkeypatch):
    # On Streamlit Community Cloud the key is entered as a secret, not in a .env file.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("copilot.config.LLM_PROVIDER", "gemini")
    app.secrets["GEMINI_API_KEY"] = "from-secrets"
    app.run()
    assert not app.exception
    assert not any("GEMINI_API_KEY" in e.value for e in app.error)


class CountingProvider(FakeProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt, temperature=0.0):
        self.calls += 1
        return super().generate(system, prompt, temperature)


def fresh_app():
    import streamlit as st

    st.cache_resource.clear()  # the daily counter lives in the resource cache
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30)


def test_session_limit_stops_questions_after_the_limit(app, monkeypatch):
    provider = CountingProvider()
    monkeypatch.setattr(llm, "get_provider", lambda name=None: provider)
    monkeypatch.setattr(config, "DEMO_SESSION_LIMIT", 2)
    monkeypatch.setattr(config, "DEMO_DAILY_LIMIT", 100)
    app = fresh_app()
    app.run()
    for question in ("one", "two", "three"):
        app.chat_input[0].set_value(question).run()
    assert not app.exception
    assert provider.calls == 2
    assert any("limit of 2 questions for this session" in w.value for w in app.warning)
    assert app.chat_input[0].disabled


def test_daily_limit_is_shared_across_sessions(monkeypatch):
    provider = CountingProvider()
    monkeypatch.setattr(index, "index_exists", lambda *a, **k: True)
    monkeypatch.setattr(index, "vector_search", lambda *a, **kw: HITS)
    monkeypatch.setattr(config, "APP_RETRIEVER", "vector")
    monkeypatch.setattr(llm, "get_provider", lambda name=None: provider)
    monkeypatch.setattr(config, "DEMO_SESSION_LIMIT", 10)
    monkeypatch.setattr(config, "DEMO_DAILY_LIMIT", 1)

    first = fresh_app()
    first.run()
    first.chat_input[0].set_value("first visitor").run()

    second = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30)  # same process, same counter
    second.run()  # the input is already disabled, so the second visitor cannot ask

    assert provider.calls == 1
    assert any("daily question limit" in w.value for w in second.warning)
    assert second.chat_input[0].disabled


def test_demo_limits_are_well_below_the_free_quota():
    calls_per_question = 2  # rerank + answer
    assert config.DEMO_SESSION_LIMIT == 10
    assert config.DEMO_DAILY_LIMIT * calls_per_question <= 0.5 * config.GEMINI_FREE_REQUESTS_PER_DAY
