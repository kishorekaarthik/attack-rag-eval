"""Download ATT&CK, parse techniques, chunk, index in ChromaDB.

  .venv/Scripts/python scripts/ingest.py                         # default embedding model
  .venv/Scripts/python scripts/ingest.py --embed-model bge-small
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.chunking import chunk_technique  # noqa: E402
from copilot.index import build_index, embedder, vector_search  # noqa: E402

SAMPLE_QUERIES = [
    "attacker dumped credentials from LSASS memory",
    "how do I detect malicious scheduled tasks?",
    "employee received an email with a macro-enabled Word document",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default=config.EMBED_MODEL, choices=sorted(config.EMBED_MODELS))
    parser.add_argument("--corpus", default="base", choices=["base", "procedures"],
                        help="procedures: also index real-world usage examples for each technique")
    args = parser.parse_args()
    embed_model, corpus = args.embed_model, args.corpus

    techniques = load_techniques(bundle_path())
    subs = sum(t.is_subtechnique for t in techniques)
    print(f"ATT&CK enterprise v{config.ATTACK_VERSION}")
    print(f"  techniques: {len(techniques)}  ({len(techniques) - subs} parents, {subs} sub-techniques)")
    print(f"  with detection text: {sum(bool(t.detection) for t in techniques)}")

    chunks = [c for t in techniques for c in chunk_technique(t, procedures=corpus == "procedures")]
    by_section = Counter(c.section for c in chunks)
    print(f"  chunks: {len(chunks)}  {dict(by_section)}")

    # The embedding model reads at most max_seq_length tokens and silently
    # drops the rest, so check how many chunks actually fit.
    model = embedder(embed_model)
    lengths = [len(ids) for ids in model.tokenizer([c.text for c in chunks])["input_ids"]]
    over = sum(n > model.max_seq_length for n in lengths)
    print(f"  tokens per chunk: median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}; "
          f"{over} of {len(chunks)} exceed the model's {model.max_seq_length}-token window")

    start = time.time()
    build_index(chunks, embed_model=embed_model, corpus=corpus)
    print(f"  indexed into {config.CHROMA_DIR.name}/ in {time.time() - start:.0f}s\n")

    for query in SAMPLE_QUERIES:
        print(f"Q: {query}")
        for hit in vector_search(query, k=5, embed_model=embed_model, corpus=corpus):
            print(f"   {hit.score:.3f}  {hit.technique_id:<10} {hit.name}  [{hit.section}]")
        print()


if __name__ == "__main__":
    main()
