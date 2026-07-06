"""
LLM-CostGuard — FastAPI entrypoint.

The core request flow:
  1. Check semantic cache — if a near-duplicate prompt was answered before, return that (no LLM call)
  2. If cache miss, route to a model tier based on prompt complexity
  3. Call OpenAI (or mock), get the response
  4. Cache the result for future similar prompts
  5. Return the response along with metadata (cost, latency, cache hit/miss, model used)
"""

import time
from fastapi import FastAPI
from pydantic import BaseModel

from app.cache import check_cache, add_to_cache
from app.router import route
from app.openai_client import get_completion

app = FastAPI(
    title="LLM-CostGuard",
    description="A cost/quality-optimizing proxy for OpenAI chat completions.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Simple liveness check — used by Cloud Run and by you, right now, to confirm the server boots."""
    return {"status": "ok", "service": "llm-costguard"}


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    model_used: str
    cache_hit: bool
    cost_usd: float
    latency_ms: float
    routing_reason: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start = time.perf_counter()

    # Step 1: check cache first — cheapest possible path
    cached = check_cache(request.prompt)
    if cached is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            response=cached.response_text,
            model_used=cached.model_used,
            cache_hit=True,
            cost_usd=0.0,  # cache hits cost nothing — this is the entire point
            latency_ms=round(latency_ms, 1),
            routing_reason="served from semantic cache",
        )

    # Step 2: cache miss — decide which model tier to use
    routing_decision = route(request.prompt)
    model = routing_decision["model"]

    # Step 3: call the model (mocked or real, depending on MOCK_MODE)
    result = get_completion(request.prompt, model=model)

    # Step 4: cache this result so similar future prompts are free
    add_to_cache(request.prompt, result.text, model)

    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        response=result.text,
        model_used=model,
        cache_hit=False,
        cost_usd=result.cost_usd,
        latency_ms=round(latency_ms, 1),
        routing_reason=routing_decision["reason"],
    )