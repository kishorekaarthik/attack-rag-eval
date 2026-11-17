"""Error analysis: what did one pipeline fix relative to another, and what does it still miss?

  .venv/Scripts/python scripts/compare_runs.py vector hybrid --split dev
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.evals.dataset import load_questions  # noqa: E402


def load(retriever: str, split: str) -> dict:
    path = config.ROOT / "results" / f"retrieval_{retriever}_{split}.json"
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["questions"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    names = {t.id: t.name for t in load_techniques(bundle_path())}
    questions = {q.id: q.question for q in load_questions(split=args.split)}
    before, after = load(args.before, args.split), load(args.after, args.split)

    def label(tid):
        return f"{tid} {names.get(tid, '?')}" if tid else "(nothing retrieved)"

    fixed = [i for i in after if after[i]["hit"] and not before[i]["hit"]]
    broken = [i for i in after if not after[i]["hit"] and before[i]["hit"]]
    missed = [i for i in after if not after[i]["hit"]]
    print(f"{args.before} -> {args.after} ({args.split}): fixed {len(fixed)}, broke {len(broken)}, "
          f"still missing {len(missed)} of {len(after)}\n")

    print(f"Fixed by {args.after} (first {min(args.show, len(fixed))}):")
    for i in fixed[:args.show]:
        b = before[i]
        print(f"  {i}  {questions[i][:110]}")
        print(f"        correct: {label(after[i]['gold'])}; now rank {after[i]['rank']}, "
              f"was {'rank ' + str(b['rank']) if b['rank'] else 'not retrieved'}; "
              f"{args.before} ranked first: {label(b['retrieved'][0] if b['retrieved'] else None)}")

    print(f"\nStill missed by {args.after} (first {min(args.show, len(missed))}):")
    for i in missed[:args.show]:
        a = after[i]
        print(f"  {i}  {questions[i][:110]}")
        print(f"        correct: {label(a['gold'])} ({'rank ' + str(a['rank']) if a['rank'] else 'not retrieved'}); "
              f"ranked first instead: {label(a['retrieved'][0] if a['retrieved'] else None)}")


if __name__ == "__main__":
    main()
