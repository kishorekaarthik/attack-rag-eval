"""ChromaDB index: build it from chunks, query it by meaning.

We compute embeddings ourselves and hand vectors to Chroma instead of letting
Chroma embed. That keeps one model object in the process, shared by indexing,
search and the eval harness, so they cannot drift apart.

Each embedding model gets its own collection, so two models can be compared
side by side without rebuilding.
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
    score: float  # higher is better; scale depends on the retriever


@lru_cache(maxsize=2)
def embedder(embed_model: str = config.EMBED_MODEL):
    from sentence_transformers import SentenceTransformer  # slow import; only when needed

    return SentenceTransformer(config.EMBED_MODELS[embed_model][0])


def query_text(embed_model: str, query: str) -> str:
    """Some models are trained with an instruction in front of queries (not documents)."""
    return config.EMBED_MODELS[embed_model][1] + query


def embed(texts: list[str], embed_model: str = config.EMBED_MODEL) -> list[list[float]]:
    # Normalised vectors: cosine similarity becomes a plain dot product.
    return embedder(embed_model).encode(texts, normalize_embeddings=True, batch_size=config.EMBED_BATCH_SIZE).tolist()


def _client(persist_dir: Path):
    return chromadb.PersistentClient(path=str(persist_dir))


def _collection_name(embed_model: str, corpus: str = "base") -> str:
    name = f"{config.COLLECTION}_{embed_model.replace('-', '_')}"
    return name if corpus == "base" else f"{name}_{corpus}"


def index_exists(persist_dir: Path = config.CHROMA_DIR, embed_model: str = config.EMBED_MODEL,
                 corpus: str = "base") -> bool:
    client = _client(persist_dir)
    name = _collection_name(embed_model, corpus)
    return any(c.name == name for c in client.list_collections()) and client.get_collection(name).count() > 0


def build_index(chunks: list[Chunk], persist_dir: Path = config.CHROMA_DIR, embed_model: str = config.EMBED_MODEL,
                corpus: str = "base", vectors: list[list[float]] | None = None) -> None:
    """(Re)build the collection from scratch so stale chunks never linger.

    `corpus` names a variant of the chunk set (e.g. "procedures" adds
    procedure-example chunks), kept in its own collection so variants can be
    compared without rebuilding. `vectors`, if given, are used as they are
    (a prebuilt snapshot) instead of embedding the chunks.
    """
    client = _client(persist_dir)
    name = _collection_name(embed_model, corpus)
    if any(c.name == name for c in client.list_collections()):
        client.delete_collection(name)
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    if vectors is None:
        vectors = embed([c.text for c in chunks], embed_model)
    for start in range(0, len(chunks), 1000):  # stay under Chroma's max batch size
        batch = chunks[start:start + 1000]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[start:start + 1000],
            metadatas=[{k: v for k, v in asdict(c).items() if k not in ("id", "text")} for c in batch],
        )


def vector_search(query: str, k: int = 5, persist_dir: Path = config.CHROMA_DIR,
                  embed_model: str = config.EMBED_MODEL, corpus: str = "base", section: str | None = None) -> list[Hit]:
    """Nearest chunks by cosine similarity; `section` limits the search to one kind of chunk."""
    collection = _client(persist_dir).get_collection(_collection_name(embed_model, corpus))
    res = collection.query(query_embeddings=embed([query_text(embed_model, query)], embed_model),
                           n_results=min(k, collection.count()),
                           where={"section": section} if section else None)
    return [
        Hit(chunk_id=cid, technique_id=meta["technique_id"], name=meta["name"], section=meta["section"],
            text=doc, score=1.0 - dist)
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def all_chunks(persist_dir: Path = config.CHROMA_DIR, embed_model: str = config.EMBED_MODEL,
               corpus: str = "base") -> list[Chunk]:
    """Every stored chunk; the keyword (BM25) index is built from these."""
    res = _client(persist_dir).get_collection(_collection_name(embed_model, corpus)).get(include=["documents", "metadatas"])
    return [Chunk(id=cid, text=doc, **meta) for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]
