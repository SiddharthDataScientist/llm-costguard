"""
LLM-CostGuard dashboard — reads request logs from Firestore and shows
cost, cache performance, and latency at a glance.

Run locally with: streamlit run dashboard/app.py
"""

import os
import sys
import datetime

import streamlit as st

st.set_page_config(page_title="LLM-CostGuard Dashboard", layout="wide")
st.title("LLM-CostGuard Dashboard")
st.caption("Live cost, cache, and routing performance for the LLM-CostGuard gateway.")

try:
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.logging_store import get_summary_stats, get_all_requests

    stats = get_summary_stats()
    requests = get_all_requests()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total requests", stats["total_requests"])
    col2.metric("Cache hit rate", f"{stats['cache_hit_rate']:.1%}")
    col3.metric("Total cost (USD)", f"${stats['total_cost_usd']:.6f}")
    col4.metric("Avg latency (ms)", f"{stats['avg_latency_ms']:.1f}")

    st.divider()

    if not requests:
        st.info("No requests logged yet. Send a few through the /chat endpoint, then refresh this page.")
    else:
        df = pd.DataFrame(requests)
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s")

        left, right = st.columns(2)

        with left:
            st.subheader("Model tier breakdown")
            model_counts = df["model_used"].value_counts()
            st.bar_chart(model_counts)

        with right:
            st.subheader("Cache hit vs miss")
            cache_counts = df["cache_hit"].map({True: "Cache hit", False: "Cache miss"}).value_counts()
            st.bar_chart(cache_counts)

        st.subheader("Cumulative cost over time")
        df_sorted = df.sort_values("created_at")
        df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
        st.line_chart(df_sorted.set_index("created_at")["cumulative_cost"])

        st.subheader("Latency over time")
        st.line_chart(df_sorted.set_index("created_at")["latency_ms"])

        st.subheader("Recent requests")
        display_df = df_sorted[["created_at", "prompt", "model_used", "cache_hit", "cost_usd", "latency_ms"]].sort_values(
            "created_at", ascending=False
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(f"Last refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("Refresh"):
        st.rerun()

except Exception as e:
    st.error("Something failed while building the dashboard:")
    st.exception(e)