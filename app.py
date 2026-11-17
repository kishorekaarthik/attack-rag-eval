"""Streamlit chat UI.  Run:  .venv/Scripts/python -m streamlit run app.py"""
import streamlit as st

from copilot import config, index, llm, retrieval
from copilot.rag import answer

st.set_page_config(page_title="ATT&CK Copilot", page_icon="🛡️")


PIPELINE = config.APP_RETRIEVER
EMBED_MODEL = retrieval.PIPELINES[PIPELINE].get("embed_model", config.EMBED_MODEL)


@st.cache_resource(show_spinner="Building the ATT&CK index (first run only, about two minutes)...")
def prepare_index() -> bool:
    """Build the index once per server process if it is not on disk yet.

    On Streamlit Community Cloud the disk starts empty, so the first visitor
    after a restart triggers the build; everyone after that reuses it.
    """
    if not index.index_exists(embed_model=EMBED_MODEL):
        from copilot.attack import bundle_path, load_techniques
        from copilot.chunking import chunk_technique

        chunks = [c for t in load_techniques(bundle_path()) for c in chunk_technique(t)]
        index.build_index(chunks, embed_model=EMBED_MODEL)
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
    st.markdown(f"**Retrieval:** `{PIPELINE}`, top {config.CONTEXT_TECHNIQUES} techniques")

prepare_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("hits"):
            render_sources(msg["hits"])

if question := st.chat_input("Describe what you saw, or ask about a technique"):
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
