"""Retrievers. Each is a function (query, k) -> ranked chunks.

Three stages, each optional, composed by `build_retriever`:

1. Vector search: finds chunks that mean the same thing as the question.
2. BM25 keyword search, fused with (1) by reciprocal rank fusion: recovers
   exact terms that embeddings blur (tool names, event IDs, port numbers).
3. Cross-encoder reranker: re-scores the fused candidates by reading the
   question and each chunk together, which is slower but more accurate than
   comparing two independently made vectors.

The app, the answer code and the eval harness pick a retriever by name, so
what gets measured is what gets shipped.
"""
import re
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from copilot import config, index
from copilot.chunking import Chunk
from copilot.index import Hit

Retriever = Callable[[str, int], list[Hit]]
Scorer = Callable[[list[tuple[str, str]]], list[float]]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Keyword index over the same chunks the vector index holds."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Hit(chunk_id=c.id, technique_id=c.technique_id, name=c.name, section=c.section, text=c.text,
                score=float(scores[i]))
            for i in top if scores[i] > 0 for c in [self._chunks[i]]
        ]


@lru_cache(maxsize=4)
def _bm25_index(persist_dir: Path, embed_model: str, corpus: str = "base", section: str | None = None) -> BM25Index:
    chunks = index.all_chunks(persist_dir, embed_model, corpus)
    return BM25Index([c for c in chunks if section is None or c.section == section])


def rrf(rankings: list[list[str]], k: int = 60, with_scores: bool = False):
    """Reciprocal rank fusion: score(item) = sum over lists of 1 / (k + rank).

    It uses only ranks, never raw scores, so cosine similarities and BM25
    scores (different scales) can be combined without normalising either.
    `k` damps the advantage of the very top ranks; 60 is the value from the
    original paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])  # stable: ties keep first-seen order
    return ordered if with_scores else [item for item, _ in ordered]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def cross_encoder_scores(pairs: list[tuple[str, str]]) -> list[float]:
    return [float(s) for s in _cross_encoder().predict(pairs, batch_size=config.RERANK_BATCH_SIZE)]


def rerank(query: str, hits: list[Hit], k: int, scorer: Scorer = cross_encoder_scores) -> list[Hit]:
    if not hits:
        return []
    scores = scorer([(query, h.text) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: -pair[1])
    return [replace(h, score=float(s)) for h, s in ranked[:k]]


def build_retriever(embed_model: str = config.EMBED_MODEL, bm25: bool = False, candidates: int = config.FUSION_CANDIDATES,
                    rrf_k: int = config.RRF_K, rerank_candidates: int = 0, scorer: Scorer = cross_encoder_scores,
                    persist_dir: Path = config.CHROMA_DIR, corpus: str = "base", section: str | None = None) -> Retriever:
    """Compose a retriever. `rerank_candidates=0` means no reranking stage."""

    def first_stage(query: str, n: int) -> list[Hit]:
        if not bm25:
            return index.vector_search(query, n, persist_dir, embed_model, corpus, section)
        depth = max(n, candidates)
        vec = index.vector_search(query, depth, persist_dir, embed_model, corpus, section)
        kw = _bm25_index(persist_dir, embed_model, corpus, section).search(query, depth)
        by_id = {h.chunk_id: h for h in vec + kw}
        fused = rrf([[h.chunk_id for h in vec], [h.chunk_id for h in kw]], k=rrf_k, with_scores=True)
        return [replace(by_id[cid], score=score) for cid, score in fused[:n]]

    def retrieve(query: str, k: int = 5) -> list[Hit]:
        if not rerank_candidates:
            return first_stage(query, k)
        return rerank_head(query, first_stage(query, max(k, rerank_candidates)), rerank_candidates, scorer)[:k]

    return retrieve


def rerank_head(query: str, hits: list[Hit], n: int, scorer: Scorer = cross_encoder_scores) -> list[Hit]:
    """Re-score only the first `n` hits; the rest follow in their original order."""
    return rerank(query, hits[:n], n, scorer) + hits[n:]


# Named configurations. Tuned values (config.py) were chosen on the dev split only.
# `llm_rerank` is applied after everything else (see copilot/llm_rerank.py).
PIPELINES: dict[str, dict] = {
    "vector": dict(),
    "hybrid": dict(bm25=True),
    "hybrid_rerank": dict(bm25=True, rerank_candidates=config.RERANK_CANDIDATES),
    "vector_bge": dict(embed_model="bge-small"),
    "hybrid_rerank_bge": dict(embed_model="bge-small", bm25=True, rerank_candidates=config.RERANK_CANDIDATES),
    "vector_desc": dict(section="description"),
    "vector_proc": dict(corpus="procedures"),
    "hybrid_proc": dict(corpus="procedures", bm25=True),
    "vector_llm": dict(llm_rerank=True),
    "vector_proc_llm": dict(corpus="procedures", llm_rerank=True),
    "hybrid_proc_llm": dict(corpus="procedures", bm25=True, llm_rerank=True),
}


def get_retriever(name: str) -> Retriever:
    if name not in PIPELINES:
        raise ValueError(f"Unknown retriever '{name}'. Choose one of: {', '.join(PIPELINES)}.")
    settings = dict(PIPELINES[name])
    if not settings.pop("llm_rerank", False):
        return build_retriever(**settings)

    from copilot.llm_rerank import default_reranker

    first, state = build_retriever(**settings), {}

    def retrieve(query: str, k: int = 5) -> list[Hit]:
        if "reranker" not in state:  # built on first use: it needs an API key
            state["reranker"] = default_reranker()
        return state["reranker"](query, first(query, max(k, config.LLM_RERANK_DEPTH)))[:k]

    return retrieve
