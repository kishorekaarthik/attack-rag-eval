"""Ship a prebuilt index so the deployed app never has to embed or parse ATT&CK.

Embedding 3,811 chunks on a small cloud instance takes minutes and most of its
memory, and parsing the 55 MB ATT&CK bundle costs a few hundred MB more. The
snapshot holds the chunks, their float32 vectors (the precision the evals
used) and the reranker's technique catalog, compressed into one file. The app
loads it straight into ChromaDB.

The file uses plain arrays and JSON, never pickle, so loading it cannot run
code. Its header records what it was built from; a snapshot from a different
ATT&CK release, embedding model or chunk set is refused, and the app falls
back to building the index from the bundle.
"""
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from copilot import config, index
from copilot.chunking import Chunk

SNAPSHOT_DIR = config.ROOT / "data" / "index"


def snapshot_path(embed_model: str = config.EMBED_MODEL, corpus: str = "base") -> Path:
    return SNAPSHOT_DIR / f"attack-v{config.ATTACK_VERSION}-{embed_model}-{corpus}.npz"


def _expected_meta(embed_model: str, corpus: str) -> dict:
    return {"attack_version": config.ATTACK_VERSION, "embed_model": config.EMBED_MODELS[embed_model][0], "corpus": corpus}


def _encode(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def _decode(arr: np.ndarray):
    return json.loads(arr.tobytes().decode("utf-8"))


def export_snapshot(path: Path, catalog: dict[str, tuple[str, str]], persist_dir: Path = config.CHROMA_DIR,
                    embed_model: str = config.EMBED_MODEL, corpus: str = "base") -> dict:
    """Write the collection's chunks and stored vectors (no re-embedding) plus the catalog."""
    collection = index._client(persist_dir).get_collection(index._collection_name(embed_model, corpus))
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    # Keep the stored (insertion) order. ChromaDB's HNSW graph and the BM25 index
    # both depend on insertion order, so reordering would change rankings.
    chunks = [asdict(Chunk(id=i, text=d, **m)) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
    vectors = np.asarray(got["embeddings"], dtype=np.float32)
    meta = {**_expected_meta(embed_model, corpus), "count": len(chunks), "dim": int(vectors.shape[1]),
            "created": date.today().isoformat()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=_encode(meta), chunks=_encode(chunks), vectors=vectors,
                        catalog=_encode({k: list(v) for k, v in catalog.items()}))
    return meta


def load_snapshot(path: Path) -> tuple[dict, list[Chunk], np.ndarray, dict[str, tuple[str, str]]]:
    with np.load(path, allow_pickle=False) as data:
        meta = _decode(data["meta"])
        chunks = [Chunk(**c) for c in _decode(data["chunks"])]
        vectors = data["vectors"].copy()
        catalog = {k: tuple(v) for k, v in _decode(data["catalog"]).items()}
    return meta, chunks, vectors, catalog


def matches(meta: dict, embed_model: str = config.EMBED_MODEL, corpus: str = "base") -> bool:
    return all(meta.get(k) == v for k, v in _expected_meta(embed_model, corpus).items())


def restore_index(path: Path, persist_dir: Path = config.CHROMA_DIR, embed_model: str = config.EMBED_MODEL,
                  corpus: str = "base") -> bool:
    """Load a snapshot into ChromaDB. False if the file is missing or was built for something else."""
    path = Path(path)
    if not path.exists():
        return False
    meta, chunks, vectors, _ = load_snapshot(path)
    if not matches(meta, embed_model, corpus) or len(chunks) != len(vectors):
        return False
    index.build_index(chunks, persist_dir, embed_model, corpus, vectors=vectors.tolist())
    return True


def load_catalog(embed_model: str = config.EMBED_MODEL, corpus: str = "base") -> dict[str, tuple[str, str]] | None:
    """The technique catalog from a matching snapshot, or None."""
    path = snapshot_path(embed_model, corpus)
    if not path.exists():
        return None
    meta, _, _, catalog = load_snapshot(path)
    return catalog if matches(meta, embed_model, corpus) else None
