"""Closed-book baseline: the model names technique IDs from its own knowledge, with no retrieval.

This is the comparison the RAG pipeline has to justify itself against. Beyond
accuracy it measures the failure modes retrieval is meant to prevent:

- invalid IDs: well-formed IDs that do not exist in the pinned release;
- retired IDs: IDs that exist but are revoked or deprecated;
- stale answers: a retired ID whose successor is the correct answer, i.e.
  the model knows the technique but under an out-of-date number.

The RAG pipeline cannot produce any of these: it only returns techniques
that are in the index, and the index holds active techniques only.
"""
import hashlib
import json
import re
from pathlib import Path

from copilot import config
from copilot.evals.dataset import Question
from copilot.llm import LLMProvider

CACHE_PATH = config.ROOT / "data" / "cache" / "closed_book.json"

SYSTEM = """You are a MITRE ATT&CK expert. Using your own knowledge only, identify which
ATT&CK Enterprise techniques best match the analyst's observation or question.

Reply with the 5 most likely technique IDs, most likely first, one per line, in the
form T1234 or T1234.001. Prefer the most specific sub-technique the details support.
No other text."""

_ID = re.compile(r"(?<![\w.])T\d{4}(?:\.\d{3})?(?!\d)", re.IGNORECASE)


def parse_ids(reply: str, limit: int = 5) -> list[str]:
    """Well-formed technique IDs in the reply, in order, without repeats, at most `limit`."""
    ids: list[str] = []
    for match in _ID.findall(reply):
        tid = match.upper()
        if tid not in ids:
            ids.append(tid)
    return ids[:limit]


def classify_id(tid: str, active: set[str], retired: dict[str, str]) -> str:
    if tid in active:
        return "valid"
    return "retired" if tid in retired else "invalid"


def score_predictions(questions: list[Question], predictions: dict[str, list[str]], active: set[str],
                      retired: dict[str, str], successors: dict[str, str]) -> list[dict]:
    rows = []
    for q in questions:
        preds = predictions[q.id]
        kinds = [classify_id(p, active, retired) for p in preds]
        rows.append({
            "id": q.id, "source": q.source, "split": q.split, "gold": q.technique_id,
            "predicted": preds, "kinds": kinds,
            "hit": q.technique_id in preds,
            "rank": preds.index(q.technique_id) + 1 if q.technique_id in preds else None,
            "n_invalid": kinds.count("invalid"),
            "n_retired": kinds.count("retired"),
            "stale": any(kind == "retired" and successors.get(p) == q.technique_id for p, kind in zip(preds, kinds)),
        })
    return rows


def summarise_closed_book(rows: list[dict]) -> dict:
    n = len(rows)
    ids_total = sum(len(r["predicted"]) for r in rows)
    return {
        "n": n,
        "hits": sum(r["hit"] for r in rows),
        "hit_rate": sum(r["hit"] for r in rows) / n if n else 0.0,
        "ids_total": ids_total,
        "ids_invalid": sum(r["n_invalid"] for r in rows),
        "ids_retired": sum(r["n_retired"] for r in rows),
        "invalid_id_rate": sum(r["n_invalid"] for r in rows) / ids_total if ids_total else 0.0,
        "retired_id_rate": sum(r["n_retired"] for r in rows) / ids_total if ids_total else 0.0,
        "questions_with_invalid": sum(r["n_invalid"] > 0 for r in rows),
        "questions_with_retired": sum(r["n_retired"] > 0 for r in rows),
        "questions_stale": sum(r["stale"] for r in rows),
    }


class ClosedBook:
    """Asks a model for its top 5 technique IDs. Responses are cached by model and prompt."""

    def __init__(self, provider: LLMProvider, cache_path: Path = CACHE_PATH):
        self._provider, self._cache_path = provider, Path(cache_path)
        self._cache = json.loads(self._cache_path.read_text(encoding="utf-8")) if self._cache_path.exists() else {}

    def __call__(self, question: str) -> list[str]:
        prompt = f"Analyst's observation or question:\n{question}"
        key = hashlib.sha256(f"{self._provider.model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._provider.generate(SYSTEM, prompt, temperature=0.0)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=0, sort_keys=True), encoding="utf-8")
        return parse_ids(self._cache[key])
