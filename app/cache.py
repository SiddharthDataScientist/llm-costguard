"""
Semantic cache: avoids paying for a fresh LLM call when a near-duplicate
question has already been answered recently.

Written in pure Python (no numpy) so we don't fight Python 3.14 wheel issues.
For a weekend project's cache size (dozens to low hundreds of entries), this
is plenty fast — no need for a real vector index yet.

Note on mock mode: real OpenAI embeddings genuinely understand meaning, so
paraphrases score very high (0.90+) and unrelated topics score low. A fake
embedding can't replicate that — our first two attempts (hash-into-vector,
at 32 and then 256 dims) both turned out biased by prompt length rather than
topic, which produced false cache hits between totally unrelated questions.
Direct word-overlap (Jaccard similarity) is a more honest mock: it directly
measures "how many words do these two questions actually share," which is a
much closer proxy for "are these about the same thing" than a lossy hash.
"""

import os
import re
import math
import time
from dataclasses import dataclass, field
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CACHE_SIZE = 200  # oldest entries get evicted past this, so the cache doesn't grow forever
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Real embeddings (cosine similarity) cluster true paraphrases very high (0.90+).
# Mock mode uses Jaccard word-overlap instead, which lives on a 0-1 scale with
# different behavior — tuned against real measurements (paraphrase ~0.75,
# unrelated ~0.0), so 0.3 gives comfortable margin on both sides.
SIMILARITY_THRESHOLD = 0.3 if MOCK_MODE else 0.90

_STOPWORDS = {
    "a", "an", "the", "is", "are", "what", "what's", "how", "do", "does", "i",
    "of", "in", "on", "for", "to", "with", "and", "or", "you", "your", "would",
    "should", "it", "this", "that", "be", "can", "will",
}

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@dataclass
class CacheEntry:
    prompt: str
    embedding: object  # frozenset of tokens in mock mode, list of floats for real embeddings
    response_text: str
    model_used: str
    created_at: float = field(default_factory=time.time)


# In-memory store for now. Swaps to Firestore-backed persistence during deployment (Sunday).
_cache_store: list = []


def _mock_embedding(text: str):
    """
    Returns the set of meaningful (non-stopword) tokens in the text.
    Used with Jaccard similarity instead of cosine — see module docstring for why.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def embed(text: str):
    if MOCK_MODE:
        return _mock_embedding(text)

    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _jaccard_similarity(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _similarity(a, b) -> float:
    if MOCK_MODE:
        return _jaccard_similarity(a, b)
    return _cosine_similarity(a, b)


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
        score = _similarity(query_embedding, entry.embedding)
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