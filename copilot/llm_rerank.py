"""LLM reranking: let the model choose the best techniques from a shortlist.

First-stage retrieval finds a shortlist of candidate techniques. The LLM then
reads the question next to each candidate's ID, name and one-line summary and
returns its top picks in order. Unlike the embedding model and the small
cross-encoder, it knows the security domain and can tell sibling techniques
apart.

Responses are cached on disk keyed by model and prompt, so re-running an
eval costs nothing and gives the same numbers.
"""
import hashlib
import json
import re
from pathlib import Path

from copilot import config
from copilot.index import Hit
from copilot.llm import LLMProvider

CACHE_PATH = config.ROOT / "data" / "cache" / "llm_rerank.json"

SYSTEM = """You are a MITRE ATT&CK expert. You will get a list of candidate techniques and an
observation or question from a security analyst.

Choose the 5 candidates that best match, most likely first. Prefer the most specific
technique that fits: a sub-technique over its parent when the details support it.

Reply with exactly 5 technique IDs from the list, one per line, best first. No other text."""

_ID = re.compile(r"(?<![\w.])T\d{4}(?:\.\d{3})?(?!\d)")
_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)\.(?=\s|$)")


def summary_line(description: str, limit: int = 200) -> str:
    """First sentence of a description, capped at `limit` characters."""
    text = " ".join(description.split())
    match = _SENTENCE_END.search(text)
    sentence = text[:match.end()] if match else text
    return sentence if len(sentence) <= limit else sentence[:limit - 3].rstrip() + "..."


def build_rerank_prompt(question: str, candidates: list[tuple[str, str, str]]) -> str:
    lines = "\n".join(f"{tid} | {name} | {summary}" for tid, name, summary in candidates)
    return f"Candidate techniques (ID | name | summary):\n{lines}\n\nAnalyst's observation or question:\n{question}"


def parse_ranking(reply: str, allowed: set[str]) -> list[str]:
    """Technique IDs from the reply that were actually offered, in order, without repeats."""
    ranked: list[str] = []
    for tid in _ID.findall(reply):
        if tid in allowed and tid not in ranked:
            ranked.append(tid)
    return ranked


class LLMReranker:
    def __init__(self, provider: LLMProvider, catalog: dict[str, tuple[str, str]], cache_path: Path = CACHE_PATH,
                 top_n: int = config.LLM_RERANK_CANDIDATES, keep: int = config.CONTEXT_TECHNIQUES):
        """`catalog` maps technique ID -> (name, one-line summary)."""
        self._provider, self._catalog = provider, catalog
        self._cache_path, self._top_n, self._keep = Path(cache_path), top_n, keep
        self._cache = json.loads(self._cache_path.read_text(encoding="utf-8")) if self._cache_path.exists() else {}

    def _ask(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self._provider.model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._provider.generate(SYSTEM, prompt, temperature=0.0)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=0, sort_keys=True), encoding="utf-8")
        return self._cache[key]

    def __call__(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Reorder `hits` so the model's chosen techniques come first; nothing is dropped."""
        order: list[str] = []
        for h in hits:
            if h.technique_id not in order:
                order.append(h.technique_id)
        shortlist = order[:self._top_n]
        if not shortlist:
            return hits
        candidates = [(tid, *self._catalog.get(tid, (tid, ""))) for tid in shortlist]
        chosen = parse_ranking(self._ask(build_rerank_prompt(question, candidates)), set(shortlist))[:self._keep]
        final = chosen + [tid for tid in order if tid not in chosen]
        position = {tid: i for i, tid in enumerate(final)}
        return sorted(hits, key=lambda h: position[h.technique_id])  # stable: chunk order within a technique is kept


def provider_for(spec) -> LLMProvider:
    """`True` -> the configured default; "gemini" / "claude" -> that vendor's default model;
    "gemini:<model>" -> that vendor and model."""
    from copilot.llm import get_provider

    if spec is True:
        return get_provider()
    name, _, model = str(spec).partition(":")
    return get_provider(name, model=model or None)


def default_reranker(provider: LLMProvider | None = None, spec=True) -> LLMReranker:
    """Reranker with the technique catalog from the shipped snapshot, or from the bundle if there is none.

    Both give identical names and summaries (a test checks this); the snapshot
    spares the deployed app from parsing the 55 MB bundle.
    """
    from copilot import snapshot

    catalog = snapshot.load_catalog(config.EMBED_MODEL, "procedures")
    if catalog is None:
        from copilot.attack import bundle_path, load_techniques

        catalog = {t.id: (t.name, summary_line(t.description)) for t in load_techniques(bundle_path())}
    return LLMReranker(provider or provider_for(spec), catalog)
