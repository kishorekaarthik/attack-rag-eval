"""Run a retriever over a question set and score it."""
from collections.abc import Callable

from copilot.evals.dataset import Question
from copilot.evals.metrics import ranked_technique_ids, score_question, summarise
from copilot.index import Hit


def evaluate(questions: list[Question], retriever: Callable[[str, int], list[Hit]], k: int = 5, depth: int = 30) -> dict:
    """Score every question.

    `depth` is how many chunks to fetch so that at least `k` distinct
    techniques come back; the metrics then look at the top `k` techniques.
    """
    rows = []
    for q in questions:
        hits = retriever(q.question, depth)
        scores = score_question(hits, q.technique_id, k)
        ranked = ranked_technique_ids(hits)
        rows.append({
            "id": q.id, "source": q.source, "split": q.split, "gold": q.technique_id, **scores,
            "rank": ranked.index(q.technique_id) + 1 if q.technique_id in ranked else None,
            "retrieved": ranked[:10],
        })
    summary = {"overall": summarise(rows)}
    for source in sorted({q.source for q in questions}):
        summary[source] = summarise([r for r in rows if r["source"] == source])
    return {"summary": summary, "questions": rows}
