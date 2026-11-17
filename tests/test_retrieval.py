import pytest

from copilot import config
from copilot.chunking import Chunk
from copilot.index import Hit, build_index, query_text
from copilot.rag import select_context
from copilot.retrieval import BM25Index, build_retriever, get_retriever, rerank, rrf, tokenize


def chunk(tid, name, body, section="description", n=0):
    return Chunk(id=f"{tid}:{section}:{n}", text=f"{tid} {name} (tactics: x)\n{body}",
                 technique_id=tid, name=name, tactics="x", section=section)


CHUNKS = [
    chunk("T1059.001", "PowerShell", "Adversaries may abuse PowerShell commands and scripts for execution."),
    chunk("T1105", "Ingress Tool Transfer", "Tools such as certutil can download files from an external system."),
    chunk("T1486", "Data Encrypted for Impact", "Adversaries may encrypt files to extort a ransom payment."),
    chunk("T1566.001", "Spearphishing Attachment", "Adversaries may send emails with a malicious attachment."),
]


def hit(tid, score=0.5, section="description", n=0):
    return Hit(chunk_id=f"{tid}:{section}:{n}", technique_id=tid, name=tid, section=section, text=f"text of {tid}", score=score)


# --- tokenizer ---------------------------------------------------------------

def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Certutil.exe -urlcache, C$ share!") == ["certutil", "exe", "urlcache", "c", "share"]


# --- reciprocal rank fusion ----------------------------------------------------

def test_rrf_prefers_item_ranked_well_in_both_lists_over_top_of_one():
    assert rrf([["a", "b", "c"], ["d", "b", "e"]])[0] == "b"


def test_rrf_keeps_items_that_appear_in_only_one_list():
    assert set(rrf([["a", "b"], ["c"]])) == {"a", "b", "c"}


def test_rrf_order_is_deterministic_on_ties():
    assert rrf([["a", "b"], ["b", "a"]]) == rrf([["a", "b"], ["b", "a"]])
    assert rrf([["a"], ["b"]]) == ["a", "b"]  # tie: earlier list wins


def test_rrf_scores_follow_the_formula():
    scored = rrf([["a", "b"], ["b"]], k=60, with_scores=True)
    assert scored[0] == ("b", pytest.approx(1 / 62 + 1 / 61))
    assert scored[1] == ("a", pytest.approx(1 / 61))


# --- BM25 ------------------------------------------------------------------------

def test_bm25_finds_chunk_by_a_rare_exact_term():
    hits = BM25Index(CHUNKS).search("someone ran certutil on the host", k=2)
    assert hits[0].technique_id == "T1105"
    assert hits[0].chunk_id == "T1105:description:0" and hits[0].text == CHUNKS[1].text


def test_bm25_returns_nothing_when_no_term_matches():
    assert BM25Index(CHUNKS).search("zzzz qqqq", k=5) == []


# --- reranker ----------------------------------------------------------------------

def test_rerank_reorders_by_scorer_and_truncates():
    candidates = [hit("T1001"), hit("T1002"), hit("T1003")]
    scores = {"text of T1001": 0.1, "text of T1002": 0.9, "text of T1003": 0.5}
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [scores[text] for _, text in pairs]

    out = rerank("the query", candidates, k=2, scorer=scorer)
    assert [h.technique_id for h in out] == ["T1002", "T1003"]
    assert [h.score for h in out] == [0.9, 0.5]
    assert seen[0] == ("the query", "text of T1001") and len(seen) == 3


def test_rerank_of_nothing_is_nothing():
    assert rerank("q", [], k=5, scorer=lambda pairs: []) == []


# --- composed retrievers on a real toy index -----------------------------------------

@pytest.fixture
def toy_index(tmp_path):
    build_index(CHUNKS, persist_dir=tmp_path)
    return tmp_path


def test_hybrid_ranks_first_the_chunk_both_searches_agree_on(toy_index):
    retriever = build_retriever(bm25=True, persist_dir=toy_index)
    hits = retriever("ransomware will encrypt files and demand payment", 3)
    assert hits[0].technique_id == "T1486"
    assert len(hits) == 3 and len({h.chunk_id for h in hits}) == 3


def test_hybrid_surfaces_an_exact_term_match(toy_index):
    hits = build_retriever(bm25=True, persist_dir=toy_index)("certutil", 2)
    assert "T1105" in [h.technique_id for h in hits]


def test_rerank_stage_gets_candidates_and_decides_final_order(toy_index):
    def scorer(pairs):  # favour the phishing chunk regardless of the query
        return [1.0 if "Spearphishing" in text else 0.0 for _, text in pairs]

    retriever = build_retriever(bm25=True, rerank_candidates=4, scorer=scorer, persist_dir=toy_index)
    assert retriever("ransomware encrypt files", 2)[0].technique_id == "T1566.001"


def test_named_retrievers_exist_and_unknown_name_is_rejected():
    for name in ("vector", "hybrid", "hybrid_rerank", "vector_bge", "hybrid_rerank_bge"):
        assert callable(get_retriever(name))
    with pytest.raises(ValueError, match="nope"):
        get_retriever("nope")


# --- embedding model handling ----------------------------------------------------------

def test_query_prefix_is_added_only_for_models_that_need_one():
    assert query_text("minilm", "what happened?") == "what happened?"
    assert query_text("bge-small", "what happened?") == config.EMBED_MODELS["bge-small"][1] + "what happened?"
    assert config.EMBED_MODELS["bge-small"][1].startswith("Represent this sentence")


# --- context selection for the LLM --------------------------------------------------------

def test_context_is_top_k_techniques_with_at_most_two_chunks_each():
    hits = [hit("T1001", n=0), hit("T1001", n=1), hit("T1001", n=2), hit("T1002"), hit("T1003"),
            hit("T1004"), hit("T1005"), hit("T1006"), hit("T1002", section="detection")]
    context = select_context(hits, k=5, per_technique=2)
    assert [h.chunk_id for h in context] == [
        "T1001:description:0", "T1001:description:1", "T1002:description:0", "T1003:description:0",
        "T1004:description:0", "T1005:description:0", "T1002:detection:0",
    ]
    assert "T1006" not in {h.technique_id for h in context}


# --- staged retrieval (one model in memory at a time) -------------------------------------

def test_staged_retrieval_matches_the_composed_retriever(toy_index):
    from copilot.evals.dataset import Question
    from copilot.evals.runner import retrieve_all

    def scorer(pairs):
        return [float(len(text) % 7) for _, text in pairs]

    questions = [Question(id="g001", question="ransomware encrypt files", technique_id="T1486", source="generated"),
                 Question(id="g002", question="certutil download", technique_id="T1105", source="generated")]
    settings = dict(bm25=True, rerank_candidates=3, persist_dir=toy_index)
    composed = build_retriever(scorer=scorer, **settings)
    staged = retrieve_all(questions, settings, depth=3, scorer=scorer)
    for q in questions:
        assert [(h.chunk_id, h.score) for h in staged[q.id]] == [(h.chunk_id, h.score) for h in composed(q.question, 3)]


def test_staged_retrieval_without_reranker_is_plain_first_stage(toy_index):
    from copilot.evals.dataset import Question
    from copilot.evals.runner import retrieve_all

    q = Question(id="g001", question="ransomware encrypt files", technique_id="T1486", source="generated")
    staged = retrieve_all([q], dict(persist_dir=toy_index), depth=2)
    assert [h.chunk_id for h in staged["g001"]] == [h.chunk_id for h in build_retriever(persist_dir=toy_index)(q.question, 2)]
