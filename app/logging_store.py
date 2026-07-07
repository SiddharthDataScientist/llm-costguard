"""
Logs every /chat request to Firestore: model used, cost, latency, cache
hit/miss, and the prompt itself. This is what the eval harness and the
dashboard read from.

Switched from local SQLite (Saturday) to Firestore because Cloud Run
containers are stateless — a local file gets wiped on every restart/scale
event, so logs would silently vanish in production. Firestore persists
independently of the container lifecycle and stays within GCP's free tier
for this project's scale.
"""

import os
import time
from google.cloud import firestore

COLLECTION_NAME = "requests"

_db = None


def get_db():
    """Lazily create the Firestore client. Uses Application Default
    Credentials — locally via `gcloud auth application-default login`,
    automatically via the service account when running on Cloud Run."""
    global _db
    if _db is None:
        project_id = os.getenv("GCP_PROJECT_ID")
        _db = firestore.Client(project=project_id) if project_id else firestore.Client()
    return _db


def init_db():
    """No schema setup needed for Firestore (schemaless), but we keep this
    function so main.py's startup hook doesn't need to change."""
    pass


def log_request(prompt: str, model_used: str, cache_hit: bool, cost_usd: float,
                 latency_ms: float, routing_reason: str):
    db = get_db()
    db.collection(COLLECTION_NAME).add({
        "prompt": prompt,
        "model_used": model_used,
        "cache_hit": cache_hit,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "routing_reason": routing_reason,
        "created_at": time.time(),
    })


def get_summary_stats() -> dict:
    """Aggregate stats used by the dashboard and eval harness.
    Reads all documents and aggregates in Python — fine at this project's
    scale (dozens-hundreds of requests); a production version at high
    volume would use Firestore aggregation queries instead."""
    db = get_db()
    docs = list(db.collection(COLLECTION_NAME).stream())

    total = len(docs)
    cache_hits = sum(1 for d in docs if d.to_dict().get("cache_hit"))
    total_cost = sum(d.to_dict().get("cost_usd", 0.0) for d in docs)
    avg_latency = sum(d.to_dict().get("latency_ms", 0.0) for d in docs) / total if total else 0.0

    model_breakdown = {}
    for d in docs:
        model = d.to_dict().get("model_used", "unknown")
        model_breakdown[model] = model_breakdown.get(model, 0) + 1

    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 3) if total else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "model_breakdown": model_breakdown,
    }


def get_all_requests() -> list:
    """Used by the dashboard to plot cost/latency over time."""
    db = get_db()
    docs = db.collection(COLLECTION_NAME).order_by("created_at").stream()
    return [d.to_dict() for d in docs]