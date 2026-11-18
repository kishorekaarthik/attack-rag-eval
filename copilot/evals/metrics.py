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


# --- paired comparison of two pipelines on the same questions -------------------------------
#
# Two pipelines scored on the same questions are not independent samples, so
# comparing their separate confidence intervals is the wrong test: questions
# both get right (or both get wrong) say nothing about which is better. Only
# the discordant questions do, the ones fixed or broken by the change.

def mcnemar_exact(fixed: int, broke: int) -> float:
    """Two-sided exact McNemar p-value.

    Under "no difference", each discordant question is equally likely to be
    a fix or a break, so `fixed` ~ Binomial(fixed + broke, 0.5).
    """
    from math import comb

    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_comparison(before: list[bool], after: list[bool], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Compare per-question hits of two pipelines run on the same questions in the same order.

    The bootstrap resamples questions (keeping each question's pair of
    outcomes together) and records the difference in hit rate each time.
    """
    if len(before) != len(after):
        raise ValueError("paired comparison needs results for the same questions in the same order")
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    n = len(b)
    fixed = int(((a == 1) & (b == 0)).sum())
    broke = int(((a == 0) & (b == 1)).sum())
    delta = a - b
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": n, "hits_before": int(b.sum()), "hits_after": int(a.sum()),
        "fixed": fixed, "broke": broke, "diff": float(delta.mean()),
        "diff_ci": (float(lo), float(hi)), "p_value": mcnemar_exact(fixed, broke),
    }
