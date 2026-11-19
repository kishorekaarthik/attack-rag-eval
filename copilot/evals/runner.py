"""Run a retrieval pipeline over a question set and score it."""
import gc
from collections.abc import Callable

from copilot import config, index, retrieval
from copilot.evals.dataset import Question
from copilot.evals.metrics import ranked_technique_ids, score_question, summarise
from copilot.index import Hit


def evaluate_hits(questions: list[Question], hits_by_id: dict[str, list[Hit]], k: int = 5) -> dict:
    """Score already-retrieved results: per-question rows plus summaries overall and per source."""
    rows = []
    for q in questions:
        hits = hits_by_id[q.id]
        ranked = ranked_technique_ids(hits)
        rows.append({
            "id": q.id, "source": q.source, "split": q.split, "gold": q.technique_id,
            **score_question(hits, q.technique_id, k),
            "rank": ranked.index(q.technique_id) + 1 if q.technique_id in ranked else None,
            "retrieved": ranked[:10],
        })
    summary = {"overall": summarise(rows)}
    for source in sorted({q.source for q in questions}):
        summary[source] = summarise([r for r in rows if r["source"] == source])
    return {"summary": summary, "questions": rows}


def evaluate(questions: list[Question], retriever: Callable[[str, int], list[Hit]], k: int = 5, depth: int = 30,
             tick: Callable[[], object] | None = None) -> dict:
    """Retrieve and score every question.

    `depth` is how many chunks to fetch so that at least `k` distinct
    techniques come back; the metrics then look at the top `k` techniques.
    `tick` runs before each question (used as a free-memory check).
    """
    hits_by_id = {}
    for q in questions:
        if tick:
            tick()
        hits_by_id[q.id] = retriever(q.question, depth)
    return evaluate_hits(questions, hits_by_id, k)


def release_models() -> None:
    """Drop cached embedding and reranking models so their memory can be reclaimed."""
    index.embedder.cache_clear()
    retrieval._cross_encoder.cache_clear()
    gc.collect()


def retrieve_all(questions: list[Question], settings: dict, depth: int = 30,
                 scorer: retrieval.Scorer = retrieval.cross_encoder_scores,
                 tick: Callable[[], object] | None = None,
                 llm_reranker: Callable[[str, list[Hit]], list[Hit]] | None = None) -> dict[str, list[Hit]]:
    """Run a pipeline in passes so only one model is in memory at a time.

    Pass 1 runs the first stage (vector, or vector + BM25) for every question,
    then the embedding model is released. Pass 2 re-scores the top candidates
    with the cross-encoder, if the pipeline has one. Pass 3 applies the LLM
    reranker, if the pipeline has one. The result is identical to calling the
    composed retriever per question.
    """
    settings = dict(settings)
    n = settings.pop("rerank_candidates", 0)
    use_llm = settings.pop("llm_rerank", False)
    settings.pop("scorer", None)
    first_depth = max(depth, n, config.LLM_RERANK_DEPTH if use_llm else 0)
    first_stage = retrieval.build_retriever(**settings)
    hits = {}
    for q in questions:
        if tick:
            tick()
        hits[q.id] = first_stage(q.question, first_depth)
    release_models()
    if n:
        for q in questions:
            if tick:
                tick()
            hits[q.id] = retrieval.rerank_head(q.question, hits[q.id], n, scorer)
        release_models()
    if use_llm:
        if llm_reranker is None:
            from copilot.llm_rerank import default_reranker

            llm_reranker = default_reranker(spec=use_llm)
        for q in questions:
            hits[q.id] = llm_reranker(q.question, hits[q.id])
    return {qid: h[:depth] for qid, h in hits.items()}
