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
    section: str  # "description" | "detection" | "procedures"


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


def pack_lines(lines: list[str], max_chars: int) -> list[str]:
    """Pack whole lines into pieces of at most `max_chars`; an oversized line is cut by words."""
    pieces, current = [], ""
    for line in lines:
        if len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_words(line, max_chars, 0))
        elif current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def chunk_technique(t: Technique, max_chars: int = 800, overlap: int = 120, procedures: bool = False,
                    max_procedure_chunks: int = 4) -> list[Chunk]:
    """Chunks for one technique.

    With `procedures=True`, real-world usage examples are added as extra
    chunks. They are capped per technique because a few very common techniques
    have hundreds of examples and would otherwise swamp the index.
    """
    tactics = ", ".join(t.tactics)
    base = f"{t.id} {t.name} (tactics: {tactics})"
    sections = [
        ("description", split_text(t.description, max_chars, overlap), base),
        ("detection", split_text(t.detection, max_chars, overlap), f"{base} - detection"),
    ]
    if procedures:
        sections.append(("procedures", pack_lines(t.procedures, max_chars)[:max_procedure_chunks],
                         f"{base} - procedure examples"))
    chunks = []
    for section, pieces, header in sections:
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=f"{t.id}:{section}:{i}",
                text=f"{header}\n{piece}",
                technique_id=t.id,
                name=t.name,
                tactics=",".join(t.tactics),
                section=section,
            ))
    return chunks
