import json

from copilot.index import Hit
from copilot.llm_rerank import LLMReranker, build_rerank_prompt, parse_ranking, summary_line


def hit(tid, n=0, section="description"):
    return Hit(chunk_id=f"{tid}:{section}:{n}", technique_id=tid, name=f"Name {tid}", section=section, text=f"text {tid}", score=0.5)


CATALOG = {f"T{1000 + i}": (f"Name T{1000 + i}", f"Summary of T{1000 + i}.") for i in range(30)}
HITS = [hit(f"T{1000 + i}") for i in range(25)] + [hit("T1000", n=1), hit("T1003", section="detection")]


class Provider:
    name, model = "fake", "fake-1"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, system, prompt, temperature=0.0):
        self.calls.append((system, prompt, temperature))
        return self.reply


def test_summary_line_is_the_first_sentence_capped():
    assert summary_line("Adversaries may abuse PowerShell. It is everywhere. More text.") == "Adversaries may abuse PowerShell."
    assert len(summary_line("word " * 200)) <= 200
    assert summary_line("e.g. is not a sentence end here. Next.") == "e.g. is not a sentence end here."


def test_prompt_lists_candidates_with_id_name_and_summary_then_the_question():
    prompt = build_rerank_prompt("what happened?", [("T1001", "Name A", "Summary A."), ("T1002", "Name B", "Summary B.")])
    assert "T1001 | Name A | Summary A." in prompt and "T1002 | Name B | Summary B." in prompt
    assert prompt.rstrip().endswith("what happened?")


def test_parse_ranking_keeps_valid_ids_in_order_without_repeats():
    reply = "1. T1005\n2. T1002.001\n3. T9999 (made up)\n4. T1005\n5. T1001"
    assert parse_ranking(reply, allowed={"T1001", "T1002.001", "T1005"}) == ["T1005", "T1002.001", "T1001"]


def test_reranker_sends_top_20_techniques_and_applies_the_models_order(tmp_path):
    provider = Provider("T1019\nT1003\nT1000\nT1007\nT1011")
    reranker = LLMReranker(provider, CATALOG, cache_path=tmp_path / "cache.json", top_n=20, keep=5)
    out = reranker("what happened?", HITS)

    [(system, prompt, temperature)] = provider.calls
    assert temperature == 0.0
    assert "T1019 | Name T1019 | Summary of T1019." in prompt
    assert "T1020" not in prompt  # 21st technique is not offered

    ranked = []
    for h in out:
        if h.technique_id not in ranked:
            ranked.append(h.technique_id)
    assert ranked[:5] == ["T1019", "T1003", "T1000", "T1007", "T1011"]
    assert ranked[5:8] == ["T1001", "T1002", "T1004"]            # the rest keep their original order
    assert len(out) == len(HITS) and {h.chunk_id for h in out} == {h.chunk_id for h in HITS}
    assert [h.chunk_id for h in out if h.technique_id == "T1003"] == ["T1003:description:0", "T1003:detection:0"]


def test_reranker_ignores_ids_that_were_not_offered_and_falls_back_to_original_order(tmp_path):
    provider = Provider("T9999\nT1024\nnonsense")   # T1024 exists but was not in the top 20
    out = LLMReranker(provider, CATALOG, cache_path=tmp_path / "c.json")("q", HITS)
    assert [h.chunk_id for h in out] == [h.chunk_id for h in sorted(HITS, key=lambda h: HITS.index(
        next(x for x in HITS if x.technique_id == h.technique_id)))]


def test_responses_are_cached_on_disk_so_reruns_make_no_calls(tmp_path):
    cache = tmp_path / "cache.json"
    first = Provider("T1019\nT1003")
    a = LLMReranker(first, CATALOG, cache_path=cache)("what happened?", HITS)
    assert len(first.calls) == 1 and len(json.loads(cache.read_text(encoding="utf-8"))) == 1

    second = Provider("should not be called")
    b = LLMReranker(second, CATALOG, cache_path=cache)("what happened?", HITS)
    assert second.calls == [] and [h.chunk_id for h in a] == [h.chunk_id for h in b]


def test_cache_is_keyed_by_model_so_two_models_do_not_share_answers(tmp_path):
    cache = tmp_path / "cache.json"
    LLMReranker(Provider("T1019"), CATALOG, cache_path=cache)("q", HITS)
    other = Provider("T1003")
    other.model = "fake-2"
    LLMReranker(other, CATALOG, cache_path=cache)("q", HITS)
    assert len(other.calls) == 1


def test_reranker_spec_picks_provider_and_model(monkeypatch):
    from copilot import llm
    from copilot.llm_rerank import provider_for

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert provider_for(True).model == llm.config.GEMINI_MODEL
    p = provider_for("gemini:gemini-3.5-flash-lite")
    assert (p.name, p.model) == ("gemini", "gemini-3.5-flash-lite")
    p = provider_for("claude")
    assert (p.name, p.model) == ("claude", llm.config.CLAUDE_MODEL)


def test_named_pipelines_for_other_reranker_models_exist():
    from copilot.retrieval import PIPELINES

    assert PIPELINES["hybrid_proc_llm35"]["llm_rerank"] == "gemini:gemini-3.5-flash-lite"
    assert PIPELINES["hybrid_proc_llm_claude"]["llm_rerank"] == "claude"
