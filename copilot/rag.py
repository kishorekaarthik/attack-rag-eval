"""Retrieve chunks, then have the LLM answer from those chunks only.

The retriever is passed in as a function, so the baseline vector search, the
hybrid search and the reranker (Phase 4) all plug into the same answer code.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass

from copilot.index import Hit
from copilot.llm import LLMProvider

SYSTEM_PROMPT = """You are a security analyst assistant grounded in MITRE ATT&CK.

Rules:
- Answer using only the context passages. Do not use outside knowledge.
- Each passage is labelled with its technique ID in square brackets. Cite that
  technique ID, e.g. [T1059.001], for every claim. Never cite by passage number.
- Only cite technique IDs that appear in the context.
- If the context does not cover the question, say so plainly and do not guess.
- Be concise: a short direct answer, then detection guidance if the context has any."""

NO_CONTEXT = "No relevant ATT&CK content was retrieved for this question."

# Lookarounds stop T1059 from matching inside T1059.001 or T10599.
_TECHNIQUE_ID = re.compile(r"(?<![\w.])T\d{4}(?:\.\d{3})?(?![\d])")


@dataclass
class Answer:
    text: str
    cited_ids: list[str]
    hits: list[Hit]


def build_prompt(question: str, hits: list[Hit]) -> str:
    # Each passage is labelled with the ID to cite. Numbering them [1], [2] made
    # the model cite the numbers instead of the technique.
    context = "\n\n".join(f"[{h.technique_id}]\n{h.text}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def cited_ids(reply: str, hits: list[Hit]) -> list[str]:
    """Technique IDs the reply cites that were actually retrieved, in first-mention order.

    Anything cited but not retrieved is dropped: it cannot have come from the
    context, so the UI must not present it as a grounded citation.
    """
    allowed = {h.technique_id for h in hits}
    seen: list[str] = []
    for match in _TECHNIQUE_ID.findall(reply):
        match = match.rstrip(".")
        if match in allowed and match not in seen:
            seen.append(match)
    return seen


def answer(question: str, retriever: Callable[[str, int], list[Hit]], provider: LLMProvider, k: int = 5) -> Answer:
    hits = retriever(question, k)
    if not hits:
        return Answer(text=NO_CONTEXT, cited_ids=[], hits=[])
    # Temperature 0: same question, same context, same answer. Needed for repeatable evals.
    text = provider.generate(SYSTEM_PROMPT, build_prompt(question, hits), temperature=0.0)
    return Answer(text=text, cited_ids=cited_ids(text, hits), hits=hits)
