"""Generate scenario-style eval questions from technique text with an LLM.

The independent set (25 questions written by a different model without access
to the indexed corpus) is the stricter test, but it is small.
Generated questions add volume and coverage across tactics. They are reported
separately because a model writing questions from the same text we index may
echo its wording, which makes them easier for retrieval than real questions.
"""
import random
from collections import defaultdict

from copilot.attack import Technique
from copilot.evals.dataset import MIN_NAME_LENGTH_FOR_LEAK_CHECK, leaks
from copilot.llm import LLMProvider

SYSTEM = """You write evaluation questions for a security knowledge system.
Given a MITRE ATT&CK technique, write ONE short incident scenario (one or two
sentences) describing what a defender would observe when this technique is used.

Rules:
- Never state the technique's name or ID, and avoid its distinctive title words.
- Describe concrete observable behaviour in plain language, in your own words;
  do not copy phrases from the description.
- Be specific enough that this technique, not a sibling, is the best match.
- Output only the scenario text."""


def sample_techniques(techniques: list[Technique], n: int, exclude: set[str], seed: int = 13) -> list[Technique]:
    """Pick n techniques, taking turns across tactics so no tactic dominates."""
    rng = random.Random(seed)
    by_tactic: dict[str, list[Technique]] = defaultdict(list)
    for t in sorted(techniques, key=lambda t: t.id):
        if t.id in exclude or len(t.name) < MIN_NAME_LENGTH_FOR_LEAK_CHECK or not t.tactics:
            continue
        by_tactic[t.tactics[0]].append(t)
    for group in by_tactic.values():
        rng.shuffle(group)

    picked: list[Technique] = []
    tactics = sorted(by_tactic)
    while len(picked) < n and any(by_tactic[t] for t in tactics):
        for tactic in tactics:
            if by_tactic[tactic] and len(picked) < n:
                picked.append(by_tactic[tactic].pop())
    return picked


def make_question(t: Technique, provider: LLMProvider, retries: int = 3) -> str | None:
    """One leak-free scenario for `t`, or None if the model keeps naming the technique."""
    prompt = f"Technique: {t.id} {t.name}\nTactics: {', '.join(t.tactics)}\n\nDescription:\n{t.description[:1500]}"
    for _ in range(retries):
        text = " ".join(provider.generate(SYSTEM, prompt, temperature=0.7).split()).strip('"').strip()
        if text and not leaks(text, t):
            return text
    return None
