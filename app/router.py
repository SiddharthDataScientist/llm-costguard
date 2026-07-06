"""
Decides which model tier a prompt should be routed to:
- "gpt-4o-mini" for simple/lookup-style questions
- "gpt-4o" for anything requiring multi-step reasoning, tradeoffs, or design judgment

This is a heuristic classifier, not a trained model — intentional for a
weekend-scope v1. The heuristics are chosen to be easy to explain and defend:
each one maps to a real signal of "this question requires more reasoning."
"""

import re

MINI_MODEL = "gpt-4o-mini"
FULL_MODEL = "gpt-4o"

# Words/phrases that tend to signal multi-step reasoning, comparison, or
# open-ended design thinking rather than a single factual lookup.
COMPLEXITY_SIGNALS = [
    "design", "architecture", "tradeoff", "trade-off", "compare", "vs",
    "why might", "why would", "walk through", "strategy", "explain how you'd",
    "explain how would you", "debug", "race condition", "scalability",
    "migrate", "without downtime", "concurrency", "distributed",
]

# Short factual/lookup words that tend to signal a simple answer suffices.
SIMPLE_SIGNALS = [
    "what is", "what's the syntax", "how do i", "what does", "difference between",
]

LONG_PROMPT_WORD_THRESHOLD = 25  # prompts longer than this lean toward being complex


def classify_complexity(prompt: str) -> str:
    """
    Returns MINI_MODEL or FULL_MODEL based on heuristic signals in the prompt text.
    """
    text = prompt.lower()

    complexity_score = 0

    # Signal 1: known complexity phrases
    for phrase in COMPLEXITY_SIGNALS:
        if phrase in text:
            complexity_score += 1

    # Signal 2: known simplicity phrases (counteract false positives)
    for phrase in SIMPLE_SIGNALS:
        if phrase in text:
            complexity_score -= 1

    # Signal 3: prompt length — longer questions tend to bundle multiple sub-questions
    word_count = len(text.split())
    if word_count > LONG_PROMPT_WORD_THRESHOLD:
        complexity_score += 1

    # Signal 4: multiple question marks or "and" conjunctions often mean a compound question
    if text.count("?") > 1 or " and " in text:
        complexity_score += 1

    return FULL_MODEL if complexity_score > 0 else MINI_MODEL


def route(prompt: str) -> dict:
    """Convenience wrapper returning the decision plus the reasoning, useful for logging/debugging."""
    model = classify_complexity(prompt)
    return {
        "model": model,
        "word_count": len(prompt.split()),
        "reason": "heuristic complexity signals" if model == FULL_MODEL else "default (no complexity signals found)",
    }