"""Phase 5: closed-book baseline (no retrieval) compared with the final RAG pipeline.

  .venv/Scripts/python scripts/eval_closed_book.py --model gemini-3.1-flash-lite --split test
  .venv/Scripts/python scripts/eval_closed_book.py --model gemini-3.5-flash-lite --split recent

Each model names its top 5 technique IDs for each question from its own
knowledge. Scored like retrieval (hit@5: the correct ID is among the five),
plus invalid, retired and stale IDs. Paired against the final pipeline's
results on the same questions (results/retrieval_hybrid_proc_llm_<split>.json).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques, retired_ids, revoked_successors  # noqa: E402
from copilot.evals.closed_book import ClosedBook, score_predictions, summarise_closed_book  # noqa: E402
from copilot.evals.dataset import QUESTIONS_PATH, load_questions  # noqa: E402
from copilot.evals.metrics import paired_comparison  # noqa: E402
from copilot.llm import get_provider  # noqa: E402

FINAL = "hybrid_proc_llm"
RECENT_PATH = config.ROOT / "data" / "eval" / "recent_questions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.GEMINI_MODEL)
    parser.add_argument("--split", default="test", choices=["test", "recent"])
    args = parser.parse_args()

    questions = load_questions(RECENT_PATH if args.split == "recent" else QUESTIONS_PATH, split=args.split)
    path = bundle_path()
    active = {t.id for t in load_techniques(path)}
    retired, successors = retired_ids(path), revoked_successors(path)

    closed_book = ClosedBook(get_provider("gemini", model=args.model))
    predictions = {}
    for n, q in enumerate(questions, start=1):
        predictions[q.id] = closed_book(q.question)
        if n % 20 == 0:
            print(f"  {n}/{len(questions)}", flush=True)
    rows = score_predictions(questions, predictions, active, retired, successors)
    summary = {"overall": summarise_closed_book(rows)}
    for source in sorted({q.source for q in questions}):
        summary[source] = summarise_closed_book([r for r in rows if r["source"] == source])

    final_path = config.ROOT / "results" / f"retrieval_{FINAL}_{args.split}.json"
    final = {r["id"]: r["hit"] for r in json.loads(final_path.read_text(encoding="utf-8"))["questions"]}
    ids = [q.id for q in questions]
    paired = paired_comparison([final[i] for i in ids], [r["hit"] for r in rows])  # "after" = closed book

    s = summary["overall"]
    print(f"closed book, {args.model}, split={args.split}, n={s['n']}")
    print(f"  hit@5 {s['hits']}/{s['n']} ({s['hit_rate']:.3f})   final pipeline {paired['hits_before']}/{s['n']}")
    print(f"  vs final pipeline: closed book fixed {paired['fixed']}, broke {paired['broke']}; "
          f"difference {paired['diff']:+.3f} [{paired['diff_ci'][0]:+.3f}, {paired['diff_ci'][1]:+.3f}], "
          f"McNemar p={paired['p_value']:.3g}")
    print(f"  IDs given: {s['ids_total']}; invalid {s['ids_invalid']} ({s['invalid_id_rate']:.1%}), "
          f"retired {s['ids_retired']} ({s['retired_id_rate']:.1%})")
    print(f"  questions with an invalid ID: {s['questions_with_invalid']}/{s['n']}; with a retired ID: "
          f"{s['questions_with_retired']}/{s['n']}; stale (retired ID whose successor is correct): "
          f"{s['questions_stale']}/{s['n']}")

    out = config.ROOT / "results" / f"closed_book_{args.model}_{args.split}.json"
    out.write_text(json.dumps({"model": args.model, "split": args.split, "summary": summary,
                               "paired_vs_final": paired, "questions": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
