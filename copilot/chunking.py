"""Turn a technique into retrieval chunks.

One technique yields separate description and detection chunks because they
answer different questions ("what is this?" vs "how do I spot it?"). Every
chunk starts with a header naming the technique, so a chunk cut from the
middle of a long description still says what it is about, both to the
embedding model and to the LLM that later has to cite an ID.
"""
from dataclasses import dataclass

from copilot.attack import Technique


@dataclass(frozen=True)
class Chunk:
    id: str  # "T1059.001:description:0"
    text: str
    technique_id: str
    name: str
    tactics: str  # comma-joined; Chroma metadata values must be scalars
    section: str  # "description" | "detection"


def _split_words(text: str, max_chars: int, overlap: int) -> list[str]:
    """Sliding word window; each piece repeats up to `overlap` trailing chars of the last."""
    pieces, current = [], []
    for word in text.split():
        if current and len(" ".join(current)) + 1 + len(word) > max_chars:
            pieces.append(" ".join(current))
            tail: list[str] = []
            while current and len(" ".join([current[-1]] + tail)) <= overlap:
                tail.insert(0, current.pop())
            current = tail
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack whole paragraphs up to `max_chars`; only an oversized paragraph is cut mid-text.

    Paragraph breaks are natural topic boundaries, so chunks that end on one
    need no overlap. Overlap applies only where we are forced to cut inside a
    paragraph, so a sentence straddling the cut is whole in one of the pieces.
    """
    pieces, current = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_words(para, max_chars, overlap))
        elif current and len(current) + 2 + len(para) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_technique(t: Technique, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    tactics = ", ".join(t.tactics)
    base = f"{t.id} {t.name} (tactics: {tactics})"
    chunks = []
    for section, text, header in (
        ("description", t.description, base),
        ("detection", t.detection, f"{base} - detection"),
    ):
        for i, piece in enumerate(split_text(text, max_chars, overlap)):
            chunks.append(Chunk(
                id=f"{t.id}:{section}:{i}",
                text=f"{header}\n{piece}",
                technique_id=t.id,
                name=t.name,
                tactics=",".join(t.tactics),
                section=section,
            ))
    return chunks
