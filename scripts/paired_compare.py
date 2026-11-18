"""Paired comparison of every pipeline against a baseline, on the same questions.

  .venv/Scripts/python scripts/paired_compare.py --split dev
  .venv/Scripts/python scripts/paired_compare.py --split dev --markdown

For each pipeline: questions it fixed and broke relative to the baseline, the
difference in hit@5 with a paired bootstrap 95% CI, and an exact McNemar
p-value. Reads results/retrieval_<pipeline>_<split>.json; runs no models.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evals.metrics import paired_comparison  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_table import LABELS, ORDER  # noqa: E402


def load(name: str, split: str) -> dict[str, bool] | None:
    path = config.ROOT / "results" / f"retrieval_{name}_{split}.json"
    if not path.exists():
        return None
    return {r["id"]: r["hit"] for r in json.loads(path.read_text(encoding="utf-8"))["questions"]}


def compare(split: str, baseline: str = "vector") -> list[tuple[str, dict]]:
    base = load(baseline, split)
    if base is None:
        raise SystemExit(f"no results for baseline '{baseline}' on {split}")
    ids = sorted(base)
    rows = []
    for name in ORDER:
        other = load(name, split)
        if name == baseline or other is None:
            continue
        if sorted(other) != ids:
            raise SystemExit(f"{name} was not run on the same {split} questions as {baseline}")
        rows.append((name, paired_comparison([base[i] for i in ids], [other[i] for i in ids])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--baseline", default="vector")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = compare(args.split, args.baseline)
    out = {name: r for name, r in rows}
    (config.ROOT / "results" / f"paired_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.markdown:
        print(f"| Pipeline vs {LABELS[args.baseline]} | hit@5 | fixed | broke | difference [paired 95% CI] | McNemar p |")
        print("|---|---|---:|---:|---|---:|")
    for name, r in rows:
        diff = f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]"
        if args.markdown:
            print(f"| {LABELS[name]} | {r['hits_after']}/{r['n']} | {r['fixed']} | {r['broke']} | {diff} | {r['p_value']:.3f} |")
        else:
            print(f"{name:<18} {r['hits_before']:>3} -> {r['hits_after']:>3}/{r['n']}  fixed {r['fixed']:>3}  "
                  f"broke {r['broke']:>3}  diff {diff}  McNemar p={r['p_value']:.3f}")


if __name__ == "__main__":
    main()
