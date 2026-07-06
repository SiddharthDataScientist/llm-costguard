"""
LLM-CostGuard — FastAPI entrypoint.

Today (step 1): just a skeleton with a /health endpoint so we know
the server runs before we build anything on top of it.
"""

from fastapi import FastAPI

app = FastAPI(
    title="LLM-CostGuard",
    description="A cost/quality-optimizing proxy for OpenAI chat completions.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Simple liveness check — used by Cloud Run and by you, right now, to confirm the server boots."""
    return {"status": "ok", "service": "llm-costguard"}