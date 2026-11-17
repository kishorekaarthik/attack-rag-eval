"""Retrieval metrics.

Two numbers per question, each answering a different thing:

- hit@k: is the correct technique among the first k *chunks*? Those k chunks
  are exactly what the LLM is given, so this is "could the LLM have seen the
  right answer at all".
- reciprocal rank: 1 / position of the correct technique in the list of
  *distinct techniques*, 0 if absent. One technique often owns several of the
  top chunks; counting them separately would push the next technique's rank
  down for no real reason. The mean over questions is MRR.

Each is computed twice: exact ID match, and "parent level", where T1059.001
and T1059 count as the same family. Exact is the headline; parent level shows
how many misses are near misses within the right family.
"""
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
    return _key(gold, parent_level) in {_key(h.technique_id, parent_level) for h in hits[:k]}


def reciprocal_rank(hits: list[Hit], gold: str, parent_level: bool = False) -> float:
    ranked = ranked_technique_ids(hits, parent_level)
    key = _key(gold, parent_level)
    return 1.0 / (ranked.index(key) + 1) if key in ranked else 0.0


def score_question(hits: list[Hit], gold: str, k: int = 5) -> dict:
    return {
        "hit": hit_at_k(hits, gold, k),
        "rr": reciprocal_rank(hits, gold),
        "hit_parent": hit_at_k(hits, gold, k, parent_level=True),
        "rr_parent": reciprocal_rank(hits, gold, parent_level=True),
    }


def summarise(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "hit_rate": sum(s["hit"] for s in scores) / n,
        "mrr": sum(s["rr"] for s in scores) / n,
        "hit_rate_parent": sum(s["hit_parent"] for s in scores) / n,
        "mrr_parent": sum(s["rr_parent"] for s in scores) / n,
    }
