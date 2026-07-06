"""
Logs every /chat request to a local SQLite file: model used, cost, latency,
cache hit/miss, and the prompt itself. This is what the eval harness and the
dashboard will read from.

Local SQLite today; swaps to Firestore during deployment (Sunday) since
Cloud Run instances are stateless and a local file won't persist there.
"""

import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "costguard.db"


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                model_used TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms REAL NOT NULL,
                routing_reason TEXT,
                created_at REAL NOT NULL
            )
        """)


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_request(prompt: str, model_used: str, cache_hit: bool, cost_usd: float,
                 latency_ms: float, routing_reason: str):
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO requests (prompt, model_used, cache_hit, cost_usd, latency_ms, routing_reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (prompt, model_used, int(cache_hit), cost_usd, latency_ms, routing_reason, time.time()),
        )


def get_summary_stats() -> dict:
    """Aggregate stats used by the dashboard and eval harness."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        cache_hits = conn.execute("SELECT COUNT(*) FROM requests WHERE cache_hit = 1").fetchone()[0]
        total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM requests").fetchone()[0]
        avg_latency = conn.execute("SELECT COALESCE(AVG(latency_ms), 0) FROM requests").fetchone()[0]
        model_breakdown = conn.execute(
            "SELECT model_used, COUNT(*) FROM requests GROUP BY model_used"
        ).fetchall()

    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 3) if total else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "model_breakdown": dict(model_breakdown),
    }


def get_all_requests() -> list:
    """Used by the dashboard to plot cost/latency over time."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT prompt, model_used, cache_hit, cost_usd, latency_ms, routing_reason, created_at FROM requests ORDER BY created_at"
        ).fetchall()
    return [
        {
            "prompt": r[0], "model_used": r[1], "cache_hit": bool(r[2]),
            "cost_usd": r[3], "latency_ms": r[4], "routing_reason": r[5], "created_at": r[6],
        }
        for r in rows
    ]