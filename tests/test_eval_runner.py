from copilot.attack import Technique
from copilot.evals.dataset import Question
from copilot.evals.generate import make_question, sample_techniques
from copilot.evals.runner import evaluate
from copilot.index import Hit


def hits(*ids):
    return [Hit(chunk_id=f"{t}:description:{i}", technique_id=t, name=t, section="description", text="", score=0.9)
            for i, t in enumerate(ids)]


QUESTIONS = [
    Question(id="i001", question="q1", technique_id="T1486", source="independent", split="dev"),
    Question(id="g001", question="q2", technique_id="T1059.001", source="generated", split="dev"),
    Question(id="g002", question="q3", technique_id="T1003", source="generated", split="test"),
]
RESULTS = {
    "q1": hits("T1486", "T1490"),          # top hit
    "q2": hits("T1059.003", "T1059.001"),  # second technique
    "q3": hits("T1055"),                   # miss
}


def test_evaluate_reports_overall_and_per_source_summaries():
    calls = []

    def retriever(query, k):
        calls.append(k)
        return RESULTS[query]

    report = evaluate(QUESTIONS, retriever, k=5, depth=30)

    assert calls == [30, 30, 30]
    assert report["summary"]["overall"]["n"] == 3
    assert report["summary"]["overall"]["hit_rate"] == 2 / 3
    assert report["summary"]["overall"]["mrr"] == (1 + 0.5 + 0) / 3
    assert report["summary"]["independent"]["hits"] == 1 and report["summary"]["independent"]["hit_rate"] == 1.0
    assert report["summary"]["generated"]["hit_rate"] == 0.5
    assert report["summary"]["generated"]["mrr"] == 0.25


def test_evaluate_keeps_per_question_detail_for_error_analysis():
    report = evaluate(QUESTIONS, lambda q, k: RESULTS[q], k=5, depth=30)
    miss = next(r for r in report["questions"] if r["id"] == "g002")
    assert miss["gold"] == "T1003" and miss["hit"] is False and miss["rank"] is None
    assert miss["retrieved"] == ["T1055"]
    second = next(r for r in report["questions"] if r["id"] == "g001")
    assert second["rank"] == 2


# --- question generation ----------------------------------------------------

def tech(tid, name, tactic):
    return Technique(id=tid, name=name, description=f"Adversaries may do {name}.", tactics=[tactic], detection="",
                     is_subtechnique="." in tid, created="2020-01-01T00:00:00.000Z")


POOL = [tech(f"T{1000 + i}", f"Technique Number {i}", f"tactic-{i % 4}") for i in range(40)] + [tech("T2000", "At", "tactic-0")]


def test_sample_is_seeded_unique_and_respects_exclusions():
    a = sample_techniques(POOL, n=12, exclude={"T1000", "T1001"}, seed=7)
    assert [t.id for t in a] == [t.id for t in sample_techniques(POOL, n=12, exclude={"T1000", "T1001"}, seed=7)]
    assert len({t.id for t in a}) == 12
    assert not {"T1000", "T1001"} & {t.id for t in a}


def test_sample_spreads_evenly_across_tactics():
    counts = {}
    for t in sample_techniques(POOL, n=12, exclude=set(), seed=7):
        counts[t.tactics[0]] = counts.get(t.tactics[0], 0) + 1
    assert counts == {"tactic-0": 3, "tactic-1": 3, "tactic-2": 3, "tactic-3": 3}


def test_sample_skips_names_too_short_to_leak_check():
    assert "T2000" not in {t.id for t in sample_techniques(POOL, n=40, exclude=set(), seed=7)}


class ScriptedProvider:
    name, model = "fake", "fake-1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, system, prompt, temperature=0.0):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_make_question_returns_clean_single_line_text():
    provider = ScriptedProvider('  "A server began\nbehaving oddly overnight."  ')
    assert make_question(POOL[3], provider) == "A server began behaving oddly overnight."
    assert "Adversaries may do Technique Number 3." in provider.prompts[0]


def test_make_question_retries_when_the_reply_leaks_the_name():
    provider = ScriptedProvider("This is Technique Number 3 in action.", "Something odd happened on the host.")
    assert make_question(POOL[3], provider) == "Something odd happened on the host."
    assert len(provider.prompts) == 2


def test_make_question_gives_up_after_retries():
    provider = ScriptedProvider(*["Technique Number 3 again."] * 3)
    assert make_question(POOL[3], provider, retries=3) is None
