import json

import pytest

from copilot.evals.closed_book import ClosedBook, classify_id, parse_ids, score_predictions, summarise_closed_book
from copilot.evals.dataset import Question

ACTIVE = {"T1059", "T1059.001", "T1685", "T1486"}
RETIRED = {"T1562.001": "revoked", "T1070.001": "revoked", "T1001": "deprecated"}
SUCCESSORS = {"T1562.001": "T1685", "T1070.001": "T1685.005"}


def test_parse_ids_keeps_order_drops_repeats_and_caps_at_five():
    reply = "1. T1059.001 (PowerShell)\n2. T1059\n3. t1486\nT1059.001 again\n4. T9999\n5. T1562.001\n6. T1003"
    assert parse_ids(reply) == ["T1059.001", "T1059", "T1486", "T9999", "T1562.001"]


def test_parse_ids_of_a_reply_with_no_ids_is_empty():
    assert parse_ids("I cannot determine the technique.") == []


@pytest.mark.parametrize("tid,expected", [
    ("T1059.001", "valid"), ("T1562.001", "retired"), ("T1001", "retired"), ("T9999", "invalid"), ("T1059.999", "invalid"),
])
def test_classify_id_against_the_pinned_release(tid, expected):
    assert classify_id(tid, ACTIVE, RETIRED) == expected


def q(qid, gold):
    return Question(id=qid, question=f"question {qid}", technique_id=gold, source="generated", split="test")


def test_score_counts_hits_invalid_and_stale_ids():
    questions = [q("a", "T1685"), q("b", "T1486"), q("c", "T1059")]
    preds = {
        "a": ["T1562.001", "T1059"],              # stale: revoked ID whose successor is the answer
        "b": ["T9999", "T1486"],                  # one invalid ID, hit at rank 2
        "c": ["T1070.001", "T1001"],              # retired but not stale (successors are not the answer)
    }
    rows = {r["id"]: r for r in score_predictions(questions, preds, ACTIVE, RETIRED, SUCCESSORS)}
    assert (rows["a"]["hit"], rows["a"]["stale"], rows["a"]["n_retired"], rows["a"]["n_invalid"]) == (False, True, 1, 0)
    assert (rows["b"]["hit"], rows["b"]["rank"], rows["b"]["n_invalid"]) == (True, 2, 1)
    assert (rows["c"]["stale"], rows["c"]["n_retired"]) == (False, 2)


def test_summary_reports_rates_per_question_and_per_id():
    questions = [q("a", "T1685"), q("b", "T1486"), q("c", "T1059")]
    preds = {"a": ["T1562.001", "T1059"], "b": ["T9999", "T1486"], "c": ["T1070.001", "T1001"]}
    s = summarise_closed_book(score_predictions(questions, preds, ACTIVE, RETIRED, SUCCESSORS))
    assert (s["n"], s["hits"], s["ids_total"]) == (3, 1, 6)
    assert (s["ids_invalid"], s["ids_retired"]) == (1, 3)
    assert (s["questions_with_invalid"], s["questions_stale"]) == (1, 1)
    assert s["invalid_id_rate"] == pytest.approx(1 / 6)


class Provider:
    name, model = "fake", "fake-1"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, system, prompt, temperature=0.0):
        self.calls.append((system, prompt, temperature))
        return self.reply


def test_closed_book_sends_only_the_question_and_caches_by_model(tmp_path):
    cache = tmp_path / "cb.json"
    provider = Provider("T1486\nT1490")
    assert ClosedBook(provider, cache_path=cache)("files were encrypted") == ["T1486", "T1490"]
    [(system, prompt, temperature)] = provider.calls
    assert prompt.strip().endswith("files were encrypted") and "Context" not in prompt and temperature == 0.0
    assert "own knowledge" in system.lower()

    again = Provider("unused")
    assert ClosedBook(again, cache_path=cache)("files were encrypted") == ["T1486", "T1490"] and again.calls == []
    other = Provider("T1059")
    other.model = "fake-2"
    ClosedBook(other, cache_path=cache)("files were encrypted")
    assert len(other.calls) == 1 and len(json.loads(cache.read_text(encoding="utf-8"))) == 2
