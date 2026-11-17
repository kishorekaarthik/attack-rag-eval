"""Print the retrieval results table (Markdown) from results/retrieval_*.json.

  .venv/Scripts/python scripts/results_table.py

The README table is pasted from this output, so every number there traces
back to a results file written by scripts/eval_retrieval.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402

ORDER = ["vector", "hybrid", "hybrid_rerank"]
LABELS = {"vector": "Vector search (baseline)", "hybrid": "+ BM25 hybrid", "hybrid_rerank": "+ cross-encoder reranker"}


def main() -> None:
    reports = {}
    for path in sorted((config.ROOT / "results").glob("retrieval_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        cfg = report["config"]
        reports[(cfg["retriever"], cfg["split"])] = report["summary"]

    for split in ("dev", "test"):
        rows = [(r, reports[(r, split)]) for r in ORDER if (r, split) in reports]
        if not rows:
            continue
        print(f"\n**{split} split**\n")
        print("| Pipeline | Set | n | hit rate@5 | MRR | hit rate@5 (parent) | MRR (parent) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for retriever, summary in rows:
            for name in ("overall", "owner", "generated"):
                if name not in summary:
                    continue
                s = summary[name]
                label = {"overall": "all", "owner": "hand-written", "generated": "generated"}[name]
                print(f"| {LABELS[retriever]} | {label} | {s['n']} | {s['hit_rate']:.3f} | {s['mrr']:.3f} | "
                      f"{s['hit_rate_parent']:.3f} | {s['mrr_parent']:.3f} |")


if __name__ == "__main__":
    main()
