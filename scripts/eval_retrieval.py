"""Measure retrieval: hit rate@5 and MRR@5 over the top 5 distinct techniques.

  .venv/Scripts/python scripts/eval_retrieval.py --retriever vector --split dev

Reports raw counts, rates and 95% bootstrap confidence intervals, overall and
per question source. The run refuses to start if any answer-key ID is
revoked, deprecated, unknown, or leaked by its question.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques, retired_ids, revoked_successors  # noqa: E402
from copilot.evals.dataset import QUESTIONS_PATH, DatasetError, load_questions, validate  # noqa: E402
from copilot.evals.runner import evaluate_hits, retrieve_all  # noqa: E402
from copilot.retrieval import PIPELINES  # noqa: E402
from copilot.sysmem import LowMemoryError, require_memory  # noqa: E402

K = config.CONTEXT_TECHNIQUES
SET_LABELS = {"overall": "all", "independent": "independent set", "generated": "generated set"}


def print_summary(summary: dict) -> None:
    print(f"{'set':<16} {'hit@5':>9} {'rate':>6} {'95% CI':>15}   {'MRR@5':>6} {'95% CI':>15}   {'hit@5 parent':>13}")
    for name, s in summary.items():
        print(f"{SET_LABELS.get(name, name):<16} {str(s['hits']) + '/' + str(s['n']):>9} {s['hit_rate']:>6.3f} "
              f"[{s['hit_rate_ci'][0]:.3f}, {s['hit_rate_ci'][1]:.3f}]   {s['mrr']:>6.3f} "
              f"[{s['mrr_ci'][0]:.3f}, {s['mrr_ci'][1]:.3f}]   "
              f"{str(s['hits_parent']) + '/' + str(s['n']):>8} {s['hit_rate_parent']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="vector", choices=sorted(PIPELINES))
    parser.add_argument("--split", default="dev", choices=["dev", "test", "recent"])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the report (default: results/retrieval_<retriever>_<split>.json)")
    args = parser.parse_args()

    path = bundle_path()
    everything = load_questions(args.questions)
    try:  # validate the whole file, not just the split being run
        validate(everything, load_techniques(path), retired_ids(path), revoked_successors(path))
    except DatasetError as exc:
        print(f"ERROR: eval run aborted. {exc}", file=sys.stderr)
        return 1

    questions = [q for q in everything if q.split == args.split]
    try:  # one model in memory at a time; stop if the machine runs short
        hits = retrieve_all(questions, PIPELINES[args.retriever], config.RETRIEVAL_DEPTH, tick=require_memory)
    except LowMemoryError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    report = evaluate_hits(questions, hits, k=K)

    embed_key = PIPELINES[args.retriever].get("embed_model", config.EMBED_MODEL)
    print(f"retriever={args.retriever}  split={args.split}  ATT&CK v{config.ATTACK_VERSION}  "
          f"embeddings={config.EMBED_MODELS[embed_key][0].split('/')[-1]}")
    print_summary(report["summary"])

    report["config"] = {
        "retriever": args.retriever, "split": args.split, "k": K, "depth": config.RETRIEVAL_DEPTH,
        "pipeline": PIPELINES[args.retriever], "embed_model": config.EMBED_MODELS[embed_key][0],
        "rerank_model": config.RERANK_MODEL if PIPELINES[args.retriever].get("rerank_candidates") else None,
        "fusion_candidates": config.FUSION_CANDIDATES, "rrf_k": config.RRF_K,
        "attack_version": config.ATTACK_VERSION, "date": date.today().isoformat(),
    }
    out = args.out or config.ROOT / "results" / f"retrieval_{args.retriever}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
