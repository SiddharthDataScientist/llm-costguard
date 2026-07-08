"""
Unit tests for the complexity router. Run with: python -m pytest
"""

from app.router import classify_complexity, MINI_MODEL, FULL_MODEL


def test_simple_syntax_question_routes_to_mini():
    assert classify_complexity("What's the syntax for a Python list comprehension?") == MINI_MODEL


def test_design_question_routes_to_full():
    prompt = (
        "Design a database schema for a multi-tenant SaaS app with row-level isolation, "
        "what are the tradeoffs between separate schemas vs a tenant_id column?"
    )
    assert classify_complexity(prompt) == FULL_MODEL


def test_debugging_question_routes_to_full():
    prompt = "Why might a Python FastAPI app under load show increasing latency even though CPU usage stays low?"
    assert classify_complexity(prompt) == FULL_MODEL


def test_simple_lookup_question_routes_to_mini():
    assert classify_complexity("How do I install a package with pip?") == MINI_MODEL


def test_long_prompt_leans_toward_full():
    # A long, multi-clause prompt without explicit complexity keywords should
    # still lean toward the full model via the length + conjunction heuristics.
    prompt = (
        "I have a Python script that reads a CSV file and I want to know how "
        "to parse dates in it and also handle missing values and then write "
        "the cleaned output back to a new CSV file"
    )
    assert classify_complexity(prompt) == FULL_MODEL