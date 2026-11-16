"""Named retrievers. Each is a function (query, k) -> ranked chunks.

The app, the answer code and the eval harness all pick a retriever by name,
so what gets measured is what gets shipped.
"""
from collections.abc import Callable

from copilot import index
from copilot.index import Hit

Retriever = Callable[[str, int], list[Hit]]


def vector(query: str, k: int = 5) -> list[Hit]:
    return index.vector_search(query, k)


RETRIEVERS: dict[str, Retriever] = {"vector": vector}


def get_retriever(name: str) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever '{name}'. Choose one of: {', '.join(RETRIEVERS)}.")
    return RETRIEVERS[name]
