"""Tune retrieval settings on the dev split only. The test split is never loaded here.

  .venv/Scripts/python scripts/tune_retrieval.py

Step 1 picks the fusion settings for hybrid search. Step 2 fixes those and
picks how many fused candidates the reranker reads. The winners are copied by
hand into copilot/config.py, and the full grid is saved for the record.

Only one model is in memory at a time: all first-stage retrieval is done with
the embedding model, which is then released before the reranker is loaded.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evals.dataset import load_questions  # noqa: E402
from copilot.evals.runner import evaluate_hits, release_models, retrieve_all  # noqa: E402
from copilot.retrieval import rerank  # noqa: E402
from copilot.sysmem import LowMemoryError, require_memory  # noqa: E402

FUSION_GRID = [(c, k) for c in (20, 50, 100) for k in (10, 60)]
RERANK_GRID = (20, 30, 50)
K = config.CONTEXT_TECHNIQUES
DEPTH = config.RETRIEVAL_DEPTH


def row(settings: dict, questions, hits_by_id) -> dict:
    s = evaluate_hits(questions, hits_by_id, k=K)["summary"]["overall"]
    out = {**settings, "hits": s["hits"], "n": s["n"], "hit_rate": s["hit_rate"], "mrr_at_5": s["mrr"]}
    label = "  ".join(f"{k}={v}" for k, v in settings.items())
    print(f"  {label:<44} hit@5 {s['hits']}/{s['n']} ({s['hit_rate']:.3f})  MRR@5 {s['mrr']:.3f}", flush=True)
    return out


def main() -> int:
    questions = load_questions(split="dev")
    print(f"dev questions: {len(questions)}\n\nStep 1: hybrid fusion settings", flush=True)
    try:
        fusion = [row(dict(candidates=c, rrf_k=k), questions,
                      retrieve_all(questions, dict(bm25=True, candidates=c, rrf_k=k), DEPTH, tick=require_memory))
                  for c, k in FUSION_GRID]
        best = max(fusion, key=lambda r: (r["hit_rate"], r["mrr_at_5"]))
        print(f"  -> best: candidates={best['candidates']} rrf_k={best['rrf_k']}\n\nStep 2: reranker candidates", flush=True)

        deepest = max(RERANK_GRID)
        pool = retrieve_all(questions, dict(bm25=True, candidates=max(best["candidates"], deepest), rrf_k=best["rrf_k"]),
                            deepest, tick=require_memory)
        release_models()  # embedding model out before the reranker comes in
        start = time.time()
        scored = {}
        for q in questions:
            require_memory()
            scored[q.id] = rerank(q.question, pool[q.id], deepest)
        seconds = (time.time() - start) / len(questions)
        release_models()

        # Cross-encoder scores are per (question, chunk) pair, so reranking the top
        # N candidates is the same as keeping the scored chunks that were in the top N.
        reranked = []
        for n in RERANK_GRID:
            allowed = {q.id: {h.chunk_id for h in pool[q.id][:n]} for q in questions}
            hits = {q.id: [h for h in scored[q.id] if h.chunk_id in allowed[q.id]][:DEPTH] for q in questions}
            reranked.append(row(dict(rerank_candidates=n), questions, hits))
        best_rerank = max(reranked, key=lambda r: (r["hit_rate"], r["mrr_at_5"]))
        print(f"  -> best: rerank_candidates={best_rerank['rerank_candidates']}")
        print(f"  reranking {deepest} candidates took {seconds:.2f}s per question on this CPU")
    except LowMemoryError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2

    out = config.ROOT / "results" / "tuning_dev.json"
    out.write_text(json.dumps({"split": "dev", "n": len(questions), "fusion": fusion, "rerank": reranked,
                               "rerank_seconds_per_question_at_50": seconds}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
