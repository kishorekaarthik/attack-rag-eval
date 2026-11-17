"""The eval question set: load, validate, split.

Validation runs before every eval. An answer key that points at a revoked or
deprecated technique can never be retrieved (those are not indexed), so the
question would silently count as a miss and drag every score down. Failing
loudly is better than a quietly wrong number.
"""
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from copilot import config
from copilot.attack import Technique

QUESTIONS_PATH = config.ROOT / "data" / "eval" / "questions.jsonl"
MIN_NAME_LENGTH_FOR_LEAK_CHECK = 5


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    technique_id: str
    source: str  # "owner" | "generated"
    split: str | None = None  # "dev" | "test"
    note: str = ""


def leaks(question: str, t: Technique) -> bool:
    """True if the question gives the answer away by naming the technique or its ID.

    A leaking question can be answered by keyword match alone, which would
    flatter keyword search and say nothing about understanding.
    """
    if re.search(rf"(?<![\w.]){re.escape(t.id)}(?!\d)", question):
        return True
    if len(t.name) < MIN_NAME_LENGTH_FOR_LEAK_CHECK:
        return False  # names like "At" are ordinary words
    return re.search(rf"\b{re.escape(t.name)}\b", question, flags=re.IGNORECASE) is not None


def validate(questions: list[Question], techniques: list[Technique],
             retired: dict[str, str], successors: dict[str, str]) -> None:
    """Raise DatasetError listing every problem; return None if the set is sound."""
    active = {t.id: t for t in techniques}
    problems, seen = [], set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate question id")
        seen.add(q.id)
        if q.technique_id in retired:
            fix = f"; replaced by {successors[q.technique_id]}" if q.technique_id in successors else ""
            problems.append(f"{q.id}: answer key {q.technique_id} is {retired[q.technique_id]}{fix}")
        elif q.technique_id not in active:
            problems.append(f"{q.id}: answer key {q.technique_id} is not in ATT&CK v{config.ATTACK_VERSION}")
        elif leaks(q.question, active[q.technique_id]):
            problems.append(f"{q.id}: question leaks the name or ID of {q.technique_id}")
    if problems:
        raise DatasetError(f"{len(problems)} problem(s) in the eval set:\n  " + "\n  ".join(problems))


def assign_splits(questions: list[Question], test_fraction: float = 0.3, seed: int = 13) -> list[Question]:
    """Stratified dev/test split: each source contributes the same share to test.

    Order within a source comes from a hash of (seed, question id), so the
    split does not depend on file order and adding a question later does not
    reshuffle the others.
    """
    def order(q: Question) -> str:
        return hashlib.sha256(f"{seed}:{q.id}".encode()).hexdigest()

    sources = sorted({q.source for q in questions})
    total_test = round(len(questions) * test_fraction)
    exact = {s: sum(q.source == s for q in questions) * test_fraction for s in sources}
    quota = {s: int(exact[s]) for s in sources}
    # Hand out the remainder to the largest fractional parts (ties: alphabetical source).
    for s in sorted(sources, key=lambda s: (-(exact[s] - quota[s]), s))[: total_test - sum(quota.values())]:
        quota[s] += 1

    test_ids = set()
    for s in sources:
        ranked = sorted((q for q in questions if q.source == s), key=order)
        test_ids.update(q.id for q in ranked[: quota[s]])
    return [replace(q, split="test" if q.id in test_ids else "dev") for q in questions]


def read_csv(path: Path, source: str) -> list[Question]:
    """Read `question,technique_id[,note]` rows; ids are o001.. or g001.. by row order."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [Question(id=f"{source[0]}{i:03d}", question=r["question"].strip(), technique_id=r["technique_id"].strip(),
                     source=source, note=(r.get("note") or "").strip())
            for i, r in enumerate(rows, start=1)]


def save_questions(questions: list[Question], path: Path = QUESTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in sorted(questions, key=lambda q: q.id):
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load_questions(path: Path = QUESTIONS_PATH, split: str | None = None, source: str | None = None) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        questions = [Question(**json.loads(line)) for line in f if line.strip()]
    return [q for q in questions if (split is None or q.split == split) and (source is None or q.source == source)]
