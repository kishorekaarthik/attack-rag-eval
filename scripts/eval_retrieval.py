"""Measure retrieval: hit rate@5 and MRR over the top 5 distinct techniques.

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
from copilot.evals.runner import evaluate  # noqa: E402
from copilot.retrieval import PIPELINES, get_retriever  # noqa: E402

K = config.CONTEXT_TECHNIQUES
SET_LABELS = {"overall": "all", "independent": "independent set", "generated": "generated set"}


def print_summary(summary: dict) -> None:
    print(f"{'set':<16} {'hit@5':>9} {'rate':>6} {'95% CI':>15}   {'MRR':>6} {'95% CI':>15}   {'hit@5 parent':>13}")
    for name, s in summary.items():
        print(f"{SET_LABELS[name]:<16} {str(s['hits']) + '/' + str(s['n']):>9} {s['hit_rate']:>6.3f} "
              f"[{s['hit_rate_ci'][0]:.3f}, {s['hit_rate_ci'][1]:.3f}]   {s['mrr']:>6.3f} "
              f"[{s['mrr_ci'][0]:.3f}, {s['mrr_ci'][1]:.3f}]   "
              f"{str(s['hits_parent']) + '/' + str(s['n']):>8} {s['hit_rate_parent']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="vector", choices=sorted(PIPELINES))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args()

    path = bundle_path()
    everything = load_questions(args.questions)
    try:  # validate the whole file, not just the split being run
        validate(everything, load_techniques(path), retired_ids(path), revoked_successors(path))
    except DatasetError as exc:
        print(f"ERROR: eval run aborted. {exc}", file=sys.stderr)
        return 1

    questions = [q for q in everything if q.split == args.split]
    report = evaluate(questions, get_retriever(args.retriever), k=K, depth=config.RETRIEVAL_DEPTH)

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
    out = config.ROOT / "results" / f"retrieval_{args.retriever}_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(config.ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
