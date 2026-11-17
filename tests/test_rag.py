from copilot.index import Hit
from copilot.rag import SYSTEM_PROMPT, answer, build_prompt, cited_ids


def hit(tid, name, body, section="description"):
    return Hit(chunk_id=f"{tid}:{section}:0", technique_id=tid, name=name, section=section,
               text=f"{tid} {name} (tactics: x)\n{body}", score=0.5)


HITS = [
    hit("T1059.001", "PowerShell", "Adversaries abuse PowerShell."),
    hit("T1053.005", "Scheduled Task", "Adversaries abuse the Task Scheduler."),
]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt, temperature=0.0):
        self.calls.append((system, prompt, temperature))
        return self.reply


def test_prompt_labels_each_chunk_with_its_technique_id_not_a_number():
    # Numbered passages ([1], [2]) led the live model to cite the numbers instead of technique IDs.
    prompt = build_prompt("How is PowerShell abused?", HITS)
    assert "[T1059.001]\nT1059.001 PowerShell (tactics: x)\nAdversaries abuse PowerShell." in prompt
    assert "[T1053.005]\nT1053.005 Scheduled Task (tactics: x)\nAdversaries abuse the Task Scheduler." in prompt
    assert "[1]" not in prompt and "[2]" not in prompt
    assert prompt.rstrip().endswith("Question: How is PowerShell abused?")


def test_cited_ids_keeps_only_ids_that_were_retrieved_in_order_once():
    reply = "Use [T1053.005] and [T1059.001], again [T1053.005]. Also T1486 which was never retrieved."
    assert cited_ids(reply, HITS) == ["T1053.005", "T1059.001"]


def test_parent_id_is_not_confused_with_its_subtechnique():
    assert cited_ids("See T1059.", HITS) == []
    assert cited_ids("See T1059.001.", HITS) == ["T1059.001"]


def test_answer_retrieves_k_then_asks_provider_with_grounding_rules():
    asked = {}

    def retriever(query, k):
        asked["args"] = (query, k)
        return HITS

    provider = FakeProvider("PowerShell abuse is T1059.001 [T1059.001].")
    result = answer("How is PowerShell abused?", retriever, provider)

    assert asked["args"] == ("How is PowerShell abused?", 30)
    [(system, prompt, temperature)] = provider.calls
    assert system == SYSTEM_PROMPT
    assert prompt == build_prompt("How is PowerShell abused?", HITS)
    assert temperature == 0.0
    assert result.text == "PowerShell abuse is T1059.001 [T1059.001]."
    assert result.cited_ids == ["T1059.001"]
    assert result.hits == HITS


def test_system_prompt_states_the_grounding_contract():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered and "context" in lowered
    assert "technique id" in lowered
    assert "do not" in lowered or "don't" in lowered


def test_no_hits_answers_without_calling_the_model():
    provider = FakeProvider("should not be used")
    result = answer("anything", lambda q, k: [], provider)
    assert provider.calls == []
    assert result.cited_ids == [] and result.hits == []
    assert "no relevant" in result.text.lower()
