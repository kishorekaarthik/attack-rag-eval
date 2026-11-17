"""Phase 3: measure retrieval. Reports hit rate@5 and MRR, overall and per question source.

  .venv/Scripts/python scripts/eval_retrieval.py --retriever vector --split dev

The run refuses to start if any answer-key ID is revoked, deprecated, unknown,
or leaked by its question.
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
from copilot.retrieval import RETRIEVERS, get_retriever  # noqa: E402

K = 5        # chunks handed to the LLM
DEPTH = 20   # how far down the ranking MRR looks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="vector", choices=sorted(RETRIEVERS))
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args()

    path = bundle_path()
    everything = load_questions(args.questions)
    try:  # validate the whole file, not just the split being run
        validate(everything, load_techniques(path), retired_ids(path), revoked_successors(path))
    except DatasetError as exc:
        print(f"ERROR: eval run aborted. {exc}", file=sys.stderr)
        return 1

    questions = [q for q in everything if args.split == "all" or q.split == args.split]
    report = evaluate(questions, get_retriever(args.retriever), k=K, depth=DEPTH)

    print(f"retriever={args.retriever}  split={args.split}  ATT&CK v{config.ATTACK_VERSION}  "
          f"embeddings={config.EMBED_MODEL.split('/')[-1]}")
    print(f"{'set':<10} {'n':>3}  {'hit@5':>6}  {'MRR':>6}  {'hit@5 (parent)':>14}  {'MRR (parent)':>12}")
    for name, s in report["summary"].items():
        print(f"{name:<10} {s['n']:>3}  {s['hit_rate']:>6.3f}  {s['mrr']:>6.3f}  "
              f"{s['hit_rate_parent']:>14.3f}  {s['mrr_parent']:>12.3f}")

    report["config"] = {
        "retriever": args.retriever, "split": args.split, "k": K, "depth": DEPTH,
        "attack_version": config.ATTACK_VERSION, "embed_model": config.EMBED_MODEL, "date": date.today().isoformat(),
    }
    out = config.ROOT / "results" / f"retrieval_{args.retriever}_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
