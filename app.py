"""Streamlit chat UI.  Run:  .venv/Scripts/python -m streamlit run app.py"""
import os

import streamlit as st

from copilot import config, index, llm, retrieval
from copilot.limits import DailyLimit
from copilot.rag import answer

st.set_page_config(page_title="ATT&CK Copilot", page_icon="🛡️")


def secrets_to_env() -> None:
    """On Streamlit Community Cloud the API key is a secret, not a .env entry.

    Copy top-level string secrets into the environment (without overriding
    anything already set), so copilot.llm reads keys the same way everywhere.
    """
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets file: local development with .env
        return
    for name, value in items.items():
        if isinstance(value, str) and value.strip() and not os.environ.get(name, "").strip():
            os.environ[name] = value


secrets_to_env()


PIPELINE = config.APP_RETRIEVER
EMBED_MODEL = retrieval.PIPELINES[PIPELINE].get("embed_model", config.EMBED_MODEL)
CORPUS = retrieval.PIPELINES[PIPELINE].get("corpus", "base")


@st.cache_resource(show_spinner="Building the ATT&CK index (first run only, a few minutes)...")
def prepare_index() -> bool:
    """Build the index once per server process if it is not on disk yet.

    On Streamlit Community Cloud the disk starts empty, so the first visitor
    after a restart triggers the build; everyone after that reuses it.
    """
    if not index.index_exists(embed_model=EMBED_MODEL, corpus=CORPUS):
        from copilot.attack import bundle_path, load_techniques
        from copilot.chunking import chunk_technique

        chunks = [c for t in load_techniques(bundle_path()) for c in chunk_technique(t, procedures=CORPUS == "procedures")]
        index.build_index(chunks, embed_model=EMBED_MODEL, corpus=CORPUS)
    return True


def render_sources(hits) -> None:
    with st.expander(f"Retrieved context ({len(hits)} chunks)"):
        for i, h in enumerate(hits, start=1):
            url = "https://attack.mitre.org/techniques/" + h.technique_id.replace(".", "/")
            st.markdown(f"**[{i}] [{h.technique_id} {h.name}]({url})** · {h.section} · score {h.score:.2f}")
            st.caption(h.text.split("\n", 1)[-1])


st.title("ATT&CK Copilot")
st.caption("Answers come only from retrieved MITRE ATT&CK passages, with technique IDs cited.")

try:
    provider = llm.get_provider()
except (llm.MissingKeyError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown(f"**Model:** `{provider.model}`")
    st.markdown(f"**ATT&CK:** enterprise v{config.ATTACK_VERSION}")
    st.markdown("**Retrieval:** BM25 + vector hybrid over technique text and procedure examples, "
                f"reranked by `{config.GEMINI_MODEL}`; top {config.CONTEXT_TECHNIQUES} techniques")

prepare_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("hits"):
            render_sources(msg["hits"])

@st.cache_resource
def daily_limit() -> DailyLimit:
    """One counter for every visitor to this app process."""
    return DailyLimit(config.DEMO_DAILY_LIMIT)


SESSION_LIMIT_MESSAGE = (
    f"You've reached the limit of {config.DEMO_SESSION_LIMIT} questions for this session. "
    "This demo runs on a free API quota; to keep exploring, run it locally with your own key (see the README)."
)
DAILY_LIMIT_MESSAGE = (
    "The demo has reached its daily question limit, which keeps it within a free API quota. "
    "It resets at 00:00 UTC. You can also run it locally with your own key (see the README)."
)

asked = st.session_state.setdefault("questions_asked", 0)
blocked = None
if asked >= config.DEMO_SESSION_LIMIT:
    blocked = SESSION_LIMIT_MESSAGE
elif daily_limit().remaining() <= 0:
    blocked = DAILY_LIMIT_MESSAGE

question = st.chat_input("Describe what you saw, or ask about a technique", disabled=blocked is not None)
if blocked:
    st.warning(blocked)
elif question and not daily_limit().try_acquire():  # another visitor took the last slot
    st.warning(DAILY_LIMIT_MESSAGE)
elif question:
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching ATT&CK..."):
                result = answer(question, retrieval.get_retriever(PIPELINE), provider)
        except llm.LLMError as exc:
            st.error(str(exc))
        else:
            st.markdown(result.text)
            render_sources(result.hits)
            st.session_state.messages.append({"role": "assistant", "text": result.text, "hits": result.hits})
