"""Tune retrieval settings on the dev split only. The test split is never loaded here.

  .venv/Scripts/python scripts/tune_retrieval.py

Step 1 picks the fusion settings for hybrid search; step 2 fixes those and
picks how many fused candidates the reranker reads. The winners are copied by
hand into copilot/config.py, and the full grid is saved for the record.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evals.dataset import load_questions  # noqa: E402
from copilot.evals.runner import evaluate  # noqa: E402
from copilot.retrieval import build_retriever  # noqa: E402

FUSION_GRID = [(c, k) for c in (20, 50, 100) for k in (10, 60)]
RERANK_GRID = (20, 30, 50)


def run(questions, **settings) -> dict:
    start = time.time()
    s = evaluate(questions, build_retriever(**settings), k=config.CONTEXT_TECHNIQUES,
                 depth=config.RETRIEVAL_DEPTH)["summary"]["overall"]
    return {**settings, "hits": s["hits"], "n": s["n"], "hit_rate": s["hit_rate"], "mrr": s["mrr"],
            "seconds_per_question": (time.time() - start) / s["n"]}


def show(row: dict) -> None:
    settings = "  ".join(f"{k}={v}" for k, v in row.items()
                         if k not in ("hits", "n", "hit_rate", "mrr", "seconds_per_question", "bm25"))
    print(f"  {settings:<50} hit@5 {row['hits']}/{row['n']} ({row['hit_rate']:.3f})  MRR {row['mrr']:.3f}  "
          f"{row['seconds_per_question']:.2f}s/q")


def main() -> None:
    questions = load_questions(split="dev")
    print(f"dev questions: {len(questions)}\n\nStep 1: hybrid fusion settings")
    fusion = [run(questions, bm25=True, candidates=c, rrf_k=k) for c, k in FUSION_GRID]
    for row in fusion:
        show(row)
    best = max(fusion, key=lambda r: (r["hit_rate"], r["mrr"]))
    print(f"  -> best: candidates={best['candidates']} rrf_k={best['rrf_k']}\n\nStep 2: reranker candidates")
    reranked = [run(questions, bm25=True, candidates=best["candidates"], rrf_k=best["rrf_k"], rerank_candidates=n)
                for n in RERANK_GRID]
    for row in reranked:
        show(row)
    best_rerank = max(reranked, key=lambda r: (r["hit_rate"], r["mrr"]))
    print(f"  -> best: rerank_candidates={best_rerank['rerank_candidates']}")

    out = config.ROOT / "results" / "tuning_dev.json"
    out.write_text(json.dumps({"split": "dev", "n": len(questions), "fusion": fusion, "rerank": reranked}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
