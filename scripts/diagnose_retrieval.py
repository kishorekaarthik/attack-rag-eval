"""Diagnose retrieval on the dev split only. The test split is never loaded here.

  .venv/Scripts/python scripts/diagnose_retrieval.py recall      # recall@10/20/50: retrieval miss or ranking miss?
  .venv/Scripts/python scripts/diagnose_retrieval.py detection   # how useful are the detection chunks?
  .venv/Scripts/python scripts/diagnose_retrieval.py misses      # 20 random baseline misses with the top 5 returned
  .venv/Scripts/python scripts/diagnose_retrieval.py reranker    # reranker model, input limit, truncation
  .venv/Scripts/python scripts/diagnose_retrieval.py prefix      # is the bge query instruction applied?
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config, index  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.chunking import chunk_technique  # noqa: E402
from copilot.evals.dataset import load_questions  # noqa: E402
from copilot.evals.metrics import ranked_technique_ids  # noqa: E402
from copilot.evals.runner import release_models  # noqa: E402
from copilot.retrieval import build_retriever  # noqa: E402
from copilot.sysmem import require_memory  # noqa: E402

RESULTS = config.ROOT / "results"


def recall() -> None:
    """Where is the correct technique in a deep ranking? Top 5 = hit; 6-50 = ranking miss; beyond = retrieval miss."""
    questions = load_questions(split="dev")
    depth_chunks, ks = 400, (5, 10, 20, 50)
    out = {}
    for name, settings in (("vector", dict()), ("hybrid", dict(bm25=True))):
        retriever = build_retriever(**settings)
        ranks = []
        for q in questions:
            require_memory()
            ranked = ranked_technique_ids(retriever(q.question, depth_chunks))
            ranks.append(ranked.index(q.technique_id) + 1 if q.technique_id in ranked else None)
        n = len(ranks)
        counts = {k: sum(r is not None and r <= k for r in ranks) for k in ks}
        out[name] = {"n": n, **{f"recall@{k}": counts[k] for k in ks},
                     "techniques_per_question_median": None}
        print(f"{name:<7} " + "  ".join(f"recall@{k} {counts[k]}/{n} ({counts[k] / n:.3f})" for k in ks))
        print(f"        of {n - counts[5]} misses at 5: {counts[50] - counts[5]} are ranked 6-50 (ranking misses), "
              f"{n - counts[50]} are beyond 50 or absent (retrieval misses)")
    (RESULTS / "diagnosis_recall_dev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def detection() -> None:
    techniques = load_techniques(bundle_path())
    chunks = [c for t in techniques for c in chunk_technique(t)]
    det = [c for c in chunks if c.section == "detection"]
    bodies = [c.text.split("\n", 1)[1] for c in det]
    lengths = sorted(len(b) for b in bodies)
    print(f"techniques: {len(techniques)}; with no detection text: {sum(not t.detection for t in techniques)}")
    print(f"detection chunks: {len(det)}; body length min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")
    print(f"near-empty (< 150 chars): {sum(n < 150 for n in lengths)}")

    # Boilerplate: analytic lines repeated word for word across different techniques.
    lines = Counter()
    owners: dict[str, set] = {}
    for c, body in zip(det, bodies):
        for line in body.split("\n"):
            line = line.strip()
            if len(line) > 40:
                lines[line] += 1
                owners.setdefault(line, set()).add(c.technique_id)
    shared = {line for line, ids in owners.items() if len(ids) > 1}
    print(f"distinct analytic lines: {len(lines)}; lines shared by more than one technique: {len(shared)}")
    boiler = sum(all(l.strip() in shared for l in body.split("\n") if len(l.strip()) > 40) for body in bodies)
    print(f"detection chunks made up only of lines shared with another technique: {boiler}")
    firsts = Counter(" ".join(b.split()[:4]) for b in bodies)
    print("most common opening words:", firsts.most_common(5))
    print("\nthree sample detection chunks:")
    rng = random.Random(13)
    for c in rng.sample(det, 3):
        print(" -", c.text[:330].replace("\n", " | "))


def misses() -> None:
    names = {t.id: t.name for t in load_techniques(bundle_path())}
    questions = {q.id: q for q in load_questions(split="dev")}
    rows = json.loads((RESULTS / "retrieval_vector_dev.json").read_text(encoding="utf-8"))["questions"]
    missed = [r for r in rows if not r["hit"]]
    sample = random.Random(13).sample(missed, 20)
    print(f"{len(missed)} baseline misses on dev; 20 chosen at random (seed 13)\n")
    for r in sample:
        print(f"{r['id']}  {questions[r['id']].question}")
        print(f"   expected: {r['gold']} {names[r['gold']]}  ({'rank ' + str(r['rank']) if r['rank'] else 'not in top 30 chunks'})")
        for i, tid in enumerate(r["retrieved"][:5], start=1):
            print(f"   {i}. {tid} {names.get(tid, '?')}")
        print()


def reranker() -> None:
    from sentence_transformers import CrossEncoder

    questions = load_questions(split="dev")
    first_stage = build_retriever(bm25=True)
    pairs = []
    for q in questions:
        require_memory()
        pairs.extend((q.question, h.text) for h in first_stage(q.question, config.RERANK_CANDIDATES))
    release_models()
    model = CrossEncoder(config.RERANK_MODEL)
    limit = model.max_length or model.tokenizer.model_max_length
    lengths = [len(model.tokenizer(a, b)["input_ids"]) for a, b in pairs]
    print(f"model: {config.RERANK_MODEL}")
    print(f"max input length: {limit} tokens (question + passage together)")
    print(f"pairs scored on dev: {len(pairs)}; tokens per pair median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")
    print(f"pairs truncated: {sum(n > limit for n in lengths)}")


def prefix() -> None:
    seen = []
    model = index.embedder("bge-small")
    original = model.encode
    model.encode = lambda texts, **kw: (seen.extend(texts), original(texts, **kw))[1]
    index.vector_search("files were renamed with a strange extension", k=1, embed_model="bge-small")
    model.encode = original
    print("text sent to bge-small for a query:")
    print("  ", repr(seen[0]))
    print("documents are embedded without the prefix:", not index.all_chunks(embed_model="bge-small")[0].text.startswith("Represent"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["recall", "detection", "misses", "reranker", "prefix"])
    {"recall": recall, "detection": detection, "misses": misses, "reranker": reranker, "prefix": prefix}[
        parser.parse_args().what]()
