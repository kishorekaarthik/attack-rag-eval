from copilot.attack import Technique
from copilot.chunking import chunk_technique


def tech(description="Short description.", detection="", tactics=("execution",)):
    return Technique(id="T1059.001", name="PowerShell", description=description, tactics=list(tactics),
                     detection=detection, is_subtechnique=True, created="2020-01-01T00:00:00.000Z")


HEADER = "T1059.001 PowerShell (tactics: execution)\n"


def test_short_technique_gives_one_description_chunk():
    [c] = chunk_technique(tech())
    assert c.id == "T1059.001:description:0"
    assert c.text == HEADER + "Short description."
    assert (c.technique_id, c.name, c.tactics, c.section) == ("T1059.001", "PowerShell", "execution", "description")


def test_detection_gets_its_own_chunk():
    chunks = chunk_technique(tech(detection="Watch script block logs."))
    assert [c.section for c in chunks] == ["description", "detection"]
    assert chunks[1].text == "T1059.001 PowerShell (tactics: execution) - detection\nWatch script block logs."


def test_empty_sections_produce_no_chunks():
    assert chunk_technique(tech(description="  ", detection="")) == []


def test_long_text_splits_on_paragraphs_within_limit():
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(8)]
    chunks = chunk_technique(tech(description="\n\n".join(paras)), max_chars=700, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith(HEADER)
        assert len(c.text) - len(HEADER) <= 700
    body = " ".join(c.text[len(HEADER):] for c in chunks)
    for i in range(8):
        assert f"Paragraph {i} " in body


def test_single_oversized_paragraph_is_still_split():
    chunks = chunk_technique(tech(description="word " * 600), max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) - len(HEADER) <= 500 for c in chunks)


def test_overlap_repeats_tail_of_previous_chunk():
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_technique(tech(description=" ".join(words)), max_chars=500, overlap=60)
    first, second = (c.text[len(HEADER):].split() for c in chunks[:2])
    shared = [w for w in second if w in first]
    assert shared and shared == second[:len(shared)] == first[-len(shared):]
    assert len(" ".join(shared)) <= 60
    assert [w for i, c in enumerate(chunks) for w in c.text[len(HEADER):].split()
            if i == 0 or w not in chunks[i - 1].text.split()] == words


def test_chunk_ids_are_unique():
    chunks = chunk_technique(tech(description="word " * 600, detection="det " * 600), max_chars=500)
    assert len({c.id for c in chunks}) == len(chunks)


def test_multiple_tactics_are_comma_joined():
    [c] = chunk_technique(tech(tactics=("execution", "persistence")))
    assert c.tactics == "execution,persistence"
    assert c.text.startswith("T1059.001 PowerShell (tactics: execution, persistence)\n")
