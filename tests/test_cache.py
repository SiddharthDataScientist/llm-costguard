"""
Unit tests for the semantic cache (mock-mode Jaccard similarity).
Run with: python -m pytest
"""

import os

os.environ["MOCK_MODE"] = "true"

from app.cache import embed, _similarity, check_cache, add_to_cache, _cache_store


def setup_function():
    """Clear the in-memory cache before each test so tests don't interfere with each other."""
    _cache_store.clear()


def test_identical_prompts_are_highly_similar():
    e1 = embed("What is a Python list comprehension?")
    e2 = embed("What is a Python list comprehension?")
    assert _similarity(e1, e2) == 1.0


def test_paraphrased_prompts_are_similar_enough_to_cache():
    e1 = embed("What is a Python list comprehension?")
    e2 = embed("How do I write a list comprehension in Python?")
    assert _similarity(e1, e2) > 0.3  # above the mock-mode threshold


def test_unrelated_prompts_have_near_zero_similarity():
    e1 = embed("What is a Python list comprehension?")
    e2 = embed("Design a payment processing system with idempotency and refunds")
    assert _similarity(e1, e2) < 0.1


def test_cache_miss_when_empty():
    assert check_cache("Any question at all") is None


def test_cache_hit_on_paraphrase_after_adding():
    add_to_cache("What is a Python list comprehension?", "A list comprehension is...", "gpt-4o-mini")
    result = check_cache("How do I write a list comprehension in Python?")
    assert result is not None
    assert result.model_used == "gpt-4o-mini"


def test_cache_miss_on_unrelated_question_after_adding():
    add_to_cache("What is a Python list comprehension?", "A list comprehension is...", "gpt-4o-mini")
    result = check_cache("Design a payment processing system with idempotency and refunds")
    assert result is None