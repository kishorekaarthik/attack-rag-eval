"""Sanity-check the vector indexes: same chunks, right model, cosine distance, normalised vectors.

  .venv/Scripts/python scripts/check_indexes.py

For each embedding model it checks that
- the collection holds exactly the chunks the chunker produces today,
- no stored passage carries the query instruction prefix,
- the collection is configured for cosine distance,
- stored vectors have length 1,
- stored vectors match what the collection's own model produces for the same
  text (and do not match the other model), so the collection was built with
  the model it is searched with.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config, index  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.chunking import chunk_technique  # noqa: E402
from copilot.evals.runner import release_models  # noqa: E402

SAMPLE = 40


def main() -> int:
    expected = {c.id: c.text for t in load_techniques(bundle_path()) for c in chunk_technique(t)}
    ok = True
    stored = {}
    for key in config.EMBED_MODELS:
        collection = index._client(config.CHROMA_DIR).get_collection(index._collection_name(key))
        got = collection.get(include=["documents", "embeddings"])
        docs = dict(zip(got["ids"], got["documents"]))
        vectors = np.asarray(got["embeddings"], dtype=float)
        stored[key] = (got["ids"], vectors)
        prefix = config.EMBED_MODELS[key][1]
        norms = np.linalg.norm(vectors, axis=1)
        checks = {
            "same chunk ids as the chunker": set(docs) == set(expected),
            "same chunk text as the chunker": all(docs[i] == expected[i] for i in docs if i in expected),
            "no passage starts with the query prefix": not prefix or not any(d.startswith(prefix) for d in docs.values()),
            "vectors are unit length": bool(np.allclose(norms, 1.0, atol=1e-3)),
        }
        print(f"\n{key}: {config.EMBED_MODELS[key][0]}")
        print(f"  collection: {collection.name}  count={collection.count()}  dim={vectors.shape[1]}")
        print(f"  metadata: {collection.metadata}")
        print(f"  configuration: {getattr(collection, 'configuration_json', None) or getattr(collection, 'configuration', None)}")
        print(f"  vector norms: min {norms.min():.4f} max {norms.max():.4f}")
        for name, passed in checks.items():
            print(f"  [{'ok' if passed else 'FAIL'}] {name}")
            ok &= passed

    # Was each collection built with its own model? Re-embed a sample with each model and compare.
    ids = stored["minilm"][0][:SAMPLE]
    texts = [expected[i] for i in ids]
    print()
    for model_key in config.EMBED_MODELS:
        fresh = np.asarray(index.embed(texts, model_key))
        release_models()
        for coll_key, (coll_ids, vectors) in stored.items():
            pos = {cid: n for n, cid in enumerate(coll_ids)}
            sims = [float(fresh[n] @ vectors[pos[cid]]) for n, cid in enumerate(ids)]
            same = model_key == coll_key
            passed = (min(sims) > 0.999) if same else (max(sims) < 0.9)
            ok &= passed
            print(f"  [{'ok' if passed else 'FAIL'}] {model_key} model vs {coll_key} collection: "
                  f"mean cosine {np.mean(sims):.3f} (expect {'~1.0' if same else 'well below 1'})")

    # Distance check: does Chroma's reported distance equal 1 - cosine for a real query?
    for key in config.EMBED_MODELS:
        q = np.asarray(index.embed([index.query_text(key, "credentials were dumped from memory")], key)[0])
        release_models()
        collection = index._client(config.CHROMA_DIR).get_collection(index._collection_name(key))
        res = collection.query(query_embeddings=[q.tolist()], n_results=3, include=["embeddings", "distances"])
        cos = [float(q @ np.asarray(v)) for v in res["embeddings"][0]]
        passed = bool(np.allclose([1 - d for d in res["distances"][0]], cos, atol=1e-3))
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL'}] {key}: Chroma distance equals 1 - cosine "
              f"(distances {[round(d, 3) for d in res['distances'][0]]}, 1-cos {[round(1 - c, 3) for c in cos]})")

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
