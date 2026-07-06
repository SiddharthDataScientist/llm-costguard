"""
Semantic cache: avoids paying for a fresh LLM call when a near-duplicate
question has already been answered recently.

Written in pure Python (no numpy) so we don't fight Python 3.14 wheel issues.
For a weekend project's cache size (dozens to low hundreds of entries), this
is plenty fast — no need for a real vector index yet.
"""

import os
import math
import time
import hashlib
from dataclasses import dataclass, field
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CACHE_SIZE = 200  # oldest entries get evicted past this, so the cache doesn't grow forever
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Real OpenAI embeddings cluster paraphrases very high (0.90+), but our mock
# word-hash embedding has a much smaller effective range (confirmed by testing:
# similar pair ~0.35, unrelated pair ~0.0). Use a lower threshold in mock mode
# so cache-hit logic is actually testable; switch to the real threshold once
# MOCK_MODE=false and real embeddings are in play.
SIMILARITY_THRESHOLD = 0.25 if MOCK_MODE else 0.90

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@dataclass
class CacheEntry:
    prompt: str
    embedding: list
    response_text: str
    model_used: str
    created_at: float = field(default_factory=time.time)


# In-memory store for now. Swaps to Firestore-backed persistence during deployment (step 6+/Sunday).
_cache_store: list = []


def _stable_hash(word: str) -> int:
    """Deterministic hash — unlike Python's built-in hash(), this gives the
    same value every run/process, so mock embeddings are reproducible."""
    return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)


def _mock_embedding(text: str) -> list:
    """
    Deterministic fake embedding for mock mode: hashes words into a small
    fixed-size vector so identical/similar text produces similar vectors,
    without calling the real embeddings API.
    """
    dims = 32
    vec = [0.0] * dims
    for word in text.lower().split():
        idx = _stable_hash(word) % dims
        vec[idx] += 1.0
    # normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(text: str) -> list:
    if MOCK_MODE:
        return _mock_embedding(text)

    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_cache(prompt: str) -> CacheEntry | None:
    """
    Returns the most similar cached entry if it's above SIMILARITY_THRESHOLD,
    otherwise None (meaning: go call the LLM for real).
    """
    if not _cache_store:
        return None

    query_embedding = embed(prompt)

    best_entry = None
    best_score = 0.0
    for entry in _cache_store:
        score = _cosine_similarity(query_embedding, entry.embedding)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        return best_entry
    return None


def add_to_cache(prompt: str, response_text: str, model_used: str):
    embedding = embed(prompt)
    entry = CacheEntry(
        prompt=prompt,
        embedding=embedding,
        response_text=response_text,
        model_used=model_used,
    )
    _cache_store.append(entry)

    # simple eviction: drop oldest entries past the cap
    if len(_cache_store) > MAX_CACHE_SIZE:
        _cache_store.pop(0)


def cache_stats() -> dict:
    return {"entries": len(_cache_store), "threshold": SIMILARITY_THRESHOLD}