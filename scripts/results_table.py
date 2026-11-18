"""Print the retrieval results tables (Markdown) from results/retrieval_*.json.

  .venv/Scripts/python scripts/results_table.py

The README tables are pasted from this output, so every number there traces
back to a results file written by scripts/eval_retrieval.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402

ORDER = ["vector", "vector_desc", "hybrid", "hybrid_rerank", "vector_proc", "hybrid_proc", "hybrid_proc_llm",
         "vector_bge", "hybrid_rerank_bge"]
LABELS = {
    "vector": "Vector search, MiniLM (baseline)",
    "vector_desc": "Vector, description chunks only",
    "hybrid": "+ BM25 hybrid (RRF)",
    "hybrid_rerank": "+ BM25 hybrid + cross-encoder reranker",
    "vector_proc": "Vector + procedure examples",
    "hybrid_proc": "BM25 hybrid + procedure examples",
    "hybrid_proc_llm": "BM25 hybrid + procedure examples + Gemini reranker",
    "vector_bge": "Vector search, bge-small (embedding swap only)",
    "hybrid_rerank_bge": "bge-small + BM25 hybrid + cross-encoder",
}
SETS = [("overall", "all"), ("independent", "independent set"), ("generated", "generated set")]


def cell(count: int, n: int, rate: float, ci=None) -> str:
    text = f"{count}/{n} ({rate:.3f})"
    return f"{text} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else text


def main() -> None:
    reports = {}
    for path in sorted((config.ROOT / "results").glob("retrieval_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[(report["config"]["retriever"], report["config"]["split"])] = report["summary"]

    for split in ("dev", "test"):
        rows = [(r, reports[(r, split)]) for r in ORDER if (r, split) in reports]
        if not rows:
            continue
        print(f"\n**{split} split**\n")
        print("| Pipeline | Set | hit rate@5 [95% CI] | MRR@5 [95% CI] | hit rate@5, parent level |")
        print("|---|---|---|---|---|")
        for retriever, summary in rows:
            for key, label in SETS:
                if key not in summary:
                    continue
                s = summary[key]
                print(f"| {LABELS[retriever]} | {label} | {cell(s['hits'], s['n'], s['hit_rate'], s['hit_rate_ci'])} | "
                      f"{s['mrr']:.3f} [{s['mrr_ci'][0]:.3f}, {s['mrr_ci'][1]:.3f}] | "
                      f"{cell(s['hits_parent'], s['n'], s['hit_rate_parent'])} |")


if __name__ == "__main__":
    main()
