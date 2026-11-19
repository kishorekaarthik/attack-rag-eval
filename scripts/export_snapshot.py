"""Write the prebuilt index the deployed app loads instead of embedding at startup.

  .venv/Scripts/python scripts/ingest.py --corpus procedures   # build the index locally first
  .venv/Scripts/python scripts/export_snapshot.py

Writes data/index/attack-v<version>-minilm-procedures.npz from the local
ChromaDB collection (stored vectors, nothing re-embedded) plus the reranker's
technique catalog.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.llm_rerank import summary_line  # noqa: E402
from copilot.snapshot import export_snapshot, snapshot_path  # noqa: E402

CORPUS = "procedures"


def main() -> None:
    catalog = {t.id: (t.name, summary_line(t.description)) for t in load_techniques(bundle_path())}
    path = snapshot_path(config.EMBED_MODEL, CORPUS)
    meta = export_snapshot(path, catalog, embed_model=config.EMBED_MODEL, corpus=CORPUS)
    size = path.stat().st_size / 1024 ** 2
    print(f"{path.relative_to(config.ROOT)}: {meta['count']} chunks x {meta['dim']} dims, "
          f"{len(catalog)} catalog entries, {size:.1f} MB")
    if size >= 100:
        raise SystemExit("snapshot is over GitHub's 100 MB per-file limit")


if __name__ == "__main__":
    main()
