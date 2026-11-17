"""Retrieval metrics, computed over distinct techniques.

A retriever returns chunks, and one technique often owns several of the top
chunks. Scoring chunks would let a technique crowd out its neighbours and
count twice. So the chunk list is first reduced to distinct techniques in
order of first appearance, and both metrics look at the top k of that list:

- hit@k: the correct technique is among the top k distinct techniques.
- reciprocal rank: 1 / its position in that top k, 0 if it is not there.
  The mean over questions is MRR@5 (cut off at rank 5).

The answer pipeline builds its context from the same top k techniques, so
hit@k is exactly "was the right technique in front of the LLM".

Each metric is computed twice: exact ID match, and "parent level", where
T1059.001 and T1059 count as one family. Exact is the headline; parent level
shows how many misses are near misses within the right family.
"""
import numpy as np

from copilot.index import Hit


def parent(technique_id: str) -> str:
    return technique_id.split(".")[0]


def _key(technique_id: str, parent_level: bool) -> str:
    return parent(technique_id) if parent_level else technique_id


def ranked_technique_ids(hits: list[Hit], parent_level: bool = False) -> list[str]:
    """Distinct technique IDs (or families) in the order they first appear."""
    ranked: list[str] = []
    for h in hits:
        key = _key(h.technique_id, parent_level)
        if key not in ranked:
            ranked.append(key)
    return ranked


def hit_at_k(hits: list[Hit], gold: str, k: int = 5, parent_level: bool = False) -> bool:
    return _key(gold, parent_level) in ranked_technique_ids(hits, parent_level)[:k]


def reciprocal_rank(hits: list[Hit], gold: str, k: int = 5, parent_level: bool = False) -> float:
    top = ranked_technique_ids(hits, parent_level)[:k]
    key = _key(gold, parent_level)
    return 1.0 / (top.index(key) + 1) if key in top else 0.0


def score_question(hits: list[Hit], gold: str, k: int = 5) -> dict:
    return {
        "hit": hit_at_k(hits, gold, k),
        "rr": reciprocal_rank(hits, gold, k),
        "hit_parent": hit_at_k(hits, gold, k, parent_level=True),
        "rr_parent": reciprocal_rank(hits, gold, k, parent_level=True),
    }


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean.

    Resample the questions with replacement many times and see how much the
    mean moves. It needs no assumption about the shape of the scores, which
    matters for reciprocal ranks (they are not bell-shaped).
    """
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(n_resamples, len(data)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hit = [float(r["hit"]) for r in rows]
    rr = [r["rr"] for r in rows]
    return {
        "n": n,
        "hits": int(sum(hit)),
        "hit_rate": sum(hit) / n,
        "hit_rate_ci": bootstrap_ci(hit),
        "mrr": sum(rr) / n,
        "mrr_ci": bootstrap_ci(rr),
        "hits_parent": sum(bool(r["hit_parent"]) for r in rows),
        "hit_rate_parent": sum(r["hit_parent"] for r in rows) / n,
        "mrr_parent": sum(r["rr_parent"] for r in rows) / n,
    }
