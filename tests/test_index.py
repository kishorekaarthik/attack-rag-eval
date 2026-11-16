import pytest

from copilot.chunking import Chunk
from copilot.index import all_chunks, build_index, vector_search


def chunk(tid, name, body):
    return Chunk(id=f"{tid}:description:0", text=f"{tid} {name} (tactics: execution)\n{body}",
                 technique_id=tid, name=name, tactics="execution", section="description")


CHUNKS = [
    chunk("T1059.001", "PowerShell", "Adversaries may abuse PowerShell commands and scripts for execution."),
    chunk("T1566.001", "Spearphishing Attachment", "Adversaries may send emails with a malicious attachment."),
    chunk("T1486", "Data Encrypted for Impact", "Adversaries may encrypt files to extort a ransom payment."),
]


@pytest.fixture
def index_dir(tmp_path):
    build_index(CHUNKS, persist_dir=tmp_path)
    return tmp_path


def test_search_returns_semantically_closest_chunk(index_dir):
    [hit] = vector_search("ransomware locked all our documents", k=1, persist_dir=index_dir)
    assert hit.technique_id == "T1486"
    assert hit.chunk_id == "T1486:description:0"
    assert hit.name == "Data Encrypted for Impact"
    assert hit.section == "description"
    assert hit.text == CHUNKS[2].text


def test_scores_are_cosine_similarity_in_descending_order(index_dir):
    hits = vector_search("malicious email attachment", k=3, persist_dir=index_dir)
    assert hits[0].technique_id == "T1566.001"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_rebuilding_replaces_rather_than_duplicates(index_dir):
    build_index(CHUNKS[:2], persist_dir=index_dir)
    assert sorted(c.id for c in all_chunks(persist_dir=index_dir)) == ["T1059.001:description:0", "T1566.001:description:0"]


def test_all_chunks_round_trips_chunk_fields(index_dir):
    got = {c.id: c for c in all_chunks(persist_dir=index_dir)}
    assert got == {c.id: c for c in CHUNKS}


def test_index_exists_only_after_build(tmp_path, index_dir):
    from copilot.index import index_exists

    assert index_exists(persist_dir=index_dir) is True
    empty = tmp_path / "empty"
    empty.mkdir()
    assert index_exists(persist_dir=empty) is False


def test_k_larger_than_collection_returns_everything(index_dir):
    assert len(vector_search("anything", k=10, persist_dir=index_dir)) == 3
