"""Cost and latency of the Gemini reranker, per question, on a sample of dev questions.

  .venv/Scripts/python scripts/measure_llm_rerank.py

Calls Gemini directly (bypassing the response cache) so every call is real.
Latency is the request time only; time spent waiting out rate limits is
excluded and the number of retries is reported separately. Cost uses the
paid-tier price pinned in config.py, even though this project runs on the
free tier.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.attack import bundle_path, load_techniques  # noqa: E402
from copilot.evals.dataset import load_questions  # noqa: E402
from copilot.evals.runner import release_models  # noqa: E402
from copilot.llm import get_provider  # noqa: E402
from copilot.llm_rerank import SYSTEM, build_rerank_prompt, summary_line  # noqa: E402
from copilot.retrieval import PIPELINES, build_retriever  # noqa: E402
from copilot.sysmem import require_memory  # noqa: E402

SAMPLE = 20


def main() -> None:
    questions = random.Random(13).sample(load_questions(split="dev"), SAMPLE)
    catalog = {t.id: (t.name, summary_line(t.description)) for t in load_techniques(bundle_path())}
    settings = {k: v for k, v in PIPELINES["hybrid_proc_llm"].items() if k != "llm_rerank"}
    first_stage = build_retriever(**settings)

    prompts, first_seconds = [], []
    for q in questions:
        require_memory()
        start = time.perf_counter()
        hits = first_stage(q.question, config.LLM_RERANK_DEPTH)
        first_seconds.append(time.perf_counter() - start)
        shortlist = []
        for h in hits:
            if h.technique_id not in shortlist:
                shortlist.append(h.technique_id)
        prompts.append(build_rerank_prompt(q.question, [(t, *catalog[t]) for t in shortlist[:config.LLM_RERANK_CANDIDATES]]))
    release_models()

    provider = get_provider("gemini")
    usage = []
    for prompt in prompts:
        provider.generate(SYSTEM, prompt, temperature=0.0)
        usage.append(provider.last_usage)

    def stats(values):
        s = sorted(values)
        return {"median": s[len(s) // 2], "p90": s[int(len(s) * 0.9) - 1], "max": s[-1]}

    price_in, price_out = config.GEMINI_PRICE_PER_M_TOKENS
    tokens_in = sum(u["input_tokens"] for u in usage) / len(usage)
    tokens_out = sum(u["output_tokens"] for u in usage) / len(usage)
    cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
    report = {
        "model": provider.model, "sample": SAMPLE, "split": "dev",
        "first_stage_seconds": stats(first_seconds),
        "gemini_seconds": stats([u["seconds"] for u in usage]),
        "gemini_retries": sum(u["attempts"] - 1 for u in usage),
        "input_tokens_mean": tokens_in, "output_tokens_mean": tokens_out,
        "price_per_m_tokens_usd": {"input": price_in, "output": price_out},
        "cost_per_question_usd": cost, "cost_per_1000_questions_usd": cost * 1000,
    }
    (config.ROOT / "results" / "llm_rerank_cost_dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"model {provider.model}, {SAMPLE} dev questions")
    print(f"first stage (hybrid + procedures): median {report['first_stage_seconds']['median']:.2f}s")
    g = report["gemini_seconds"]
    print(f"Gemini rerank call: median {g['median']:.2f}s, p90 {g['p90']:.2f}s, max {g['max']:.2f}s "
          f"(retries excluded; {report['gemini_retries']} retries in total)")
    print(f"tokens per question: {tokens_in:.0f} in, {tokens_out:.0f} out")
    print(f"cost at ${price_in}/${price_out} per 1M tokens: ${cost:.6f} per question, ${cost * 1000:.3f} per 1,000")


if __name__ == "__main__":
    main()
