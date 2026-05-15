"""
evals/run_evals.py — Run the eval set against the live RAG chain and report results.

Optionally sends traces to LangSmith when LANGCHAIN_API_KEY is set.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --json     # output results as JSON
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_chain import chat


# ------------------------------------------------------------------ helpers

def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def run_eval(case: dict) -> dict:
    t0 = time.time()
    result = chat(
        case["question"],
        category_filter=case.get("category_filter"),
        session_id=f"eval_{case['id']}",
        user_email="eval_runner@internal",
    )
    elapsed = time.time() - t0

    should_refuse = case.get("should_be_refused", False)

    if should_refuse:
        passed = result.was_refused
        score  = 1.0 if passed else 0.0
        note   = "correctly refused" if passed else "should have been refused"
    else:
        kw_score = keyword_score(result.answer, case.get("expected_keywords", []))
        passed   = kw_score >= 0.5 and not result.was_refused
        score    = kw_score
        note     = f"keyword coverage: {kw_score:.0%}"

    cost = (result.input_tokens * 0.000003) + (result.output_tokens * 0.000015)

    return {
        "id":             case["id"],
        "question":       case["question"],
        "passed":         passed,
        "score":          round(score, 2),
        "note":           note,
        "was_refused":    result.was_refused,
        "confidence":     result.confidence,
        "input_tokens":   result.input_tokens,
        "output_tokens":  result.output_tokens,
        "latency_ms":     result.latency_ms,
        "cost_usd":       round(cost, 6),
        "answer_snippet": result.answer[:120] + "…" if len(result.answer) > 120 else result.answer,
    }


# ------------------------------------------------------------------ main

def main(output_json: bool = False):
    eval_path = Path(__file__).parent / "eval_set.json"
    cases = json.loads(eval_path.read_text())

    print(f"\nRunning {len(cases)} evals…\n")
    results = []

    for case in cases:
        r = run_eval(case)
        results.append(r)
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {status}  [{r['id']}] {r['question'][:60]}")
        print(f"         {r['note']} | {r['latency_ms']}ms | ${r['cost_usd']:.5f}")

    # Summary
    n_pass  = sum(1 for r in results if r["passed"])
    n_total = len(results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_latency = sum(r["latency_ms"] for r in results) / n_total

    print(f"\n{'='*50}")
    print(f"  Result:     {n_pass}/{n_total} passed ({n_pass/n_total:.0%})")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Avg latency:{avg_latency:.0f} ms")
    print(f"{'='*50}\n")

    if output_json:
        print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    main(output_json=args.json)
