import numpy as np
import pytest

from copilot import config, index
from copilot.chunking import Chunk
from copilot.index import all_chunks, build_index, vector_search
from copilot.snapshot import export_snapshot, load_snapshot, restore_index


def chunk(tid, name, body, section="description"):
    return Chunk(id=f"{tid}:{section}:0", text=f"{tid} {name} (tactics: x)\n{body}",
                 technique_id=tid, name=name, tactics="x", section=section)


CHUNKS = [
    chunk("T1059.001", "PowerShell", "Adversaries may abuse PowerShell commands and scripts for execution."),
    chunk("T1486", "Data Encrypted for Impact", "Adversaries may encrypt files to extort a ransom payment."),
    chunk("T1486", "Data Encrypted for Impact", "Watch for mass file renames.", section="detection"),
]
CATALOG = {"T1059.001": ("PowerShell", "Adversaries may abuse PowerShell."), "T1486": ("Data Encrypted for Impact", "Encrypts files.")}


@pytest.fixture
def snapshot(tmp_path):
    src = tmp_path / "src"
    build_index(CHUNKS, persist_dir=src)
    path = tmp_path / "snap.npz"
    export_snapshot(path, CATALOG, persist_dir=src)
    return src, path


def test_restore_reproduces_the_index_without_embedding_anything(snapshot, tmp_path, monkeypatch):
    src, path = snapshot
    dst = tmp_path / "dst"
    monkeypatch.setattr(index, "embed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedded during restore")))
    assert restore_index(path, persist_dir=dst) is True
    monkeypatch.undo()
    assert sorted(all_chunks(persist_dir=dst), key=lambda c: c.id) == sorted(all_chunks(persist_dir=src), key=lambda c: c.id)
    for query in ("ransom note appeared", "encoded script ran"):
        a = vector_search(query, k=3, persist_dir=src)
        b = vector_search(query, k=3, persist_dir=dst)
        assert [h.chunk_id for h in a] == [h.chunk_id for h in b]
        assert [h.score for h in a] == pytest.approx([h.score for h in b], abs=1e-6)


def test_snapshot_keeps_float32_vectors_and_needs_no_pickle(snapshot):
    _, path = snapshot
    with np.load(path, allow_pickle=False) as data:
        assert data["vectors"].dtype == np.float32 and data["vectors"].shape == (3, 384)
    meta, chunks, vectors, catalog = load_snapshot(path)
    assert meta["attack_version"] == config.ATTACK_VERSION and meta["count"] == 3
    assert meta["embed_model"] == config.EMBED_MODELS[config.EMBED_MODEL][0]
    assert catalog == CATALOG and len(chunks) == 3


def test_restore_refuses_a_snapshot_from_another_release(snapshot, tmp_path, monkeypatch):
    _, path = snapshot
    monkeypatch.setattr(config, "ATTACK_VERSION", "99.0")
    assert restore_index(path, persist_dir=tmp_path / "dst") is False


def test_restore_refuses_a_missing_file(tmp_path):
    assert restore_index(tmp_path / "nope.npz", persist_dir=tmp_path / "dst") is False


SHIPPED = config.ROOT / "data" / "index" / f"attack-v{config.ATTACK_VERSION}-minilm-procedures.npz"
BUNDLE = config.RAW_DIR / f"enterprise-attack-{config.ATTACK_VERSION}.json"


@pytest.mark.skipif(not (SHIPPED.exists() and BUNDLE.exists()), reason="needs the shipped snapshot and the bundle")
def test_shipped_snapshot_matches_what_the_code_builds_today():
    from copilot.attack import load_techniques
    from copilot.chunking import chunk_technique
    from copilot.llm_rerank import summary_line

    meta, chunks, vectors, catalog = load_snapshot(SHIPPED)
    techniques = load_techniques(BUNDLE)
    expected = {c.id: c for t in techniques for c in chunk_technique(t, procedures=True)}
    assert {c.id: c for c in chunks} == expected
    assert catalog == {t.id: (t.name, summary_line(t.description)) for t in techniques}
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert SHIPPED.stat().st_size < 100 * 1024 * 1024  # GitHub's per-file limit


def test_reranker_takes_its_catalog_from_the_snapshot_when_one_matches(monkeypatch):
    import copilot.llm_rerank as llm_rerank
    import copilot.snapshot as snapshot

    monkeypatch.setattr(snapshot, "load_catalog", lambda *a, **k: CATALOG)
    monkeypatch.setattr("copilot.attack.load_techniques", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed bundle")))

    class P:
        name, model = "fake", "fake-1"

    reranker = llm_rerank.default_reranker(provider=P())
    assert reranker._catalog == CATALOG


def test_export_keeps_insertion_order(tmp_path):
    # HNSW and BM25 depend on insertion order; sorting changed rankings in practice.
    unsorted = list(reversed(CHUNKS))
    build_index(unsorted, persist_dir=tmp_path / "src")
    export_snapshot(tmp_path / "s.npz", CATALOG, persist_dir=tmp_path / "src")
    _, chunks, _, _ = load_snapshot(tmp_path / "s.npz")
    assert [c.id for c in chunks] == [c.id for c in unsorted] != sorted(c.id for c in unsorted)
