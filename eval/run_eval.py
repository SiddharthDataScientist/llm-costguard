"""
Eval harness: runs the labeled test set through the real pipeline
(cache -> router -> OpenAI client) and reports:
  - routing accuracy vs. your human-assigned labels
  - cache hit rate
  - total cost vs. a no-cache/always-gpt-4o baseline

Run with: python -m eval.run_eval
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import check_cache, add_to_cache
from app.router import route
from app.openai_client import get_completion, _compute_cost


def load_test_set(path: str = "eval/test_set.jsonl") -> list:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_eval():
    cases = load_test_set()

    results = []
    total_actual_cost = 0.0
    total_baseline_cost = 0.0  # cost if every query always hit gpt-4o with no caching
    routing_correct = 0
    routing_total = 0  # only counted on cache misses, since cache hits skip routing
    cache_hits = 0

    for case in cases:
        prompt = case["prompt"]
        expected_model = case["expected_model"]

        cached = check_cache(prompt)
        if cached is not None:
            cache_hits += 1
            actual_model = cached.model_used
            cost = 0.0
            was_cache_hit = True
        else:
            decision = route(prompt)
            actual_model = decision["model"]
            result = get_completion(prompt, model=actual_model)
            add_to_cache(prompt, result.text, actual_model)
            cost = result.cost_usd
            was_cache_hit = False

            routing_total += 1
            if actual_model == expected_model:
                routing_correct += 1

        # Baseline: what would this have cost with no caching, always using gpt-4o?
        baseline_result = get_completion(prompt, model="gpt-4o")
        total_baseline_cost += baseline_result.cost_usd
        total_actual_cost += cost

        results.append({
            "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "expected_model": expected_model,
            "actual_model": actual_model,
            "cache_hit": was_cache_hit,
            "match": (actual_model == expected_model) if not was_cache_hit else "n/a (cache hit)",
        })

    # --- Report ---
    print("=" * 70)
    print("EVAL RESULTS")
    print("=" * 70)
    for r in results:
        status = "✓" if r["match"] is True else ("✗" if r["match"] is False else "—")
        print(f"[{status}] {r['prompt']}")
        print(f"      expected={r['expected_model']}  actual={r['actual_model']}  cache_hit={r['cache_hit']}")
    print("-" * 70)

    routing_accuracy = routing_correct / routing_total if routing_total else 0.0
    cache_hit_rate = cache_hits / len(cases) if cases else 0.0
    savings_pct = (1 - total_actual_cost / total_baseline_cost) * 100 if total_baseline_cost else 0.0

    print(f"Total queries:        {len(cases)}")
    print(f"Cache hits:           {cache_hits} ({cache_hit_rate:.1%})")
    print(f"Routing accuracy:     {routing_correct}/{routing_total} ({routing_accuracy:.1%}) [only counted on cache misses]")
    print(f"Actual cost:          ${total_actual_cost:.6f}")
    print(f"Baseline cost (no cache, always gpt-4o): ${total_baseline_cost:.6f}")
    print(f"Cost savings:         {savings_pct:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_eval()