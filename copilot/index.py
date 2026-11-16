"""ChromaDB index: build it from chunks, query it by meaning.

We compute embeddings ourselves and hand vectors to Chroma instead of letting
Chroma embed. That keeps one model object in the process, shared by indexing,
search and (later) the eval harness, so they cannot drift apart.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

from copilot import config
from copilot.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    technique_id: str
    name: str
    section: str
    text: str
    score: float  # cosine similarity, higher is closer


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer  # slow import; only when needed

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    # Normalised vectors: cosine similarity becomes a plain dot product.
    return embedder().encode(texts, normalize_embeddings=True, batch_size=64).tolist()


def _client(persist_dir: Path):
    return chromadb.PersistentClient(path=str(persist_dir))


def index_exists(persist_dir: Path = config.CHROMA_DIR) -> bool:
    client = _client(persist_dir)
    return any(c.name == config.COLLECTION for c in client.list_collections()) and \
        client.get_collection(config.COLLECTION).count() > 0


def build_index(chunks: list[Chunk], persist_dir: Path = config.CHROMA_DIR) -> None:
    """(Re)build the collection from scratch so stale chunks never linger."""
    client = _client(persist_dir)
    if any(c.name == config.COLLECTION for c in client.list_collections()):
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    vectors = embed([c.text for c in chunks])
    for start in range(0, len(chunks), 1000):  # stay under Chroma's max batch size
        batch = chunks[start:start + 1000]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[start:start + 1000],
            metadatas=[{k: v for k, v in asdict(c).items() if k not in ("id", "text")} for c in batch],
        )


def vector_search(query: str, k: int = 5, persist_dir: Path = config.CHROMA_DIR) -> list[Hit]:
    collection = _client(persist_dir).get_collection(config.COLLECTION)
    res = collection.query(query_embeddings=embed([query]), n_results=min(k, collection.count()))
    return [
        Hit(chunk_id=cid, technique_id=meta["technique_id"], name=meta["name"], section=meta["section"],
            text=doc, score=1.0 - dist)
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def all_chunks(persist_dir: Path = config.CHROMA_DIR) -> list[Chunk]:
    """Every stored chunk; the keyword (BM25) index is built from these in Phase 4."""
    res = _client(persist_dir).get_collection(config.COLLECTION).get(include=["documents", "metadatas"])
    return [Chunk(id=cid, text=doc, **meta) for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]
