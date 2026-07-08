# LLM-CostGuard

A cost- and quality-optimizing proxy for OpenAI's chat API. It reduces LLM spend through semantic caching and automatic complexity-based model routing (gpt-4o-mini vs gpt-4o), with a measured evaluation harness and a live cost/cache dashboard.

**Live backend:** https://llm-costguard-542711591694.asia-south1.run.app/docs
**Live dashboard:** https://llm-costguard-dashboard-542711591694.asia-south1.run.app/

---

## The problem

Teams calling LLM APIs in production tend to do one of two things: always use the most capable (and most expensive) model regardless of query complexity, or manually pick a cheaper model and accept worse answers on hard questions. Neither is a routing decision — it's a guess. LLM-CostGuard makes that decision automatically, and caches near-duplicate queries so repeat traffic costs nothing at all.

## Results

Measured against an 18-query, hand-labeled test set of coding/technical questions:

| Metric | Result |
|---|---|
| Routing accuracy vs. human labels | **94.4%** (17/18) |
| Cost savings vs. always-gpt-4o, no cache | **57.6%** |
| Actual cost (this eval run) | $0.008132 |
| Baseline cost (no routing, no cache) | $0.019197 |

The one routing miss is a genuine, explainable edge case (see *Key decisions* below) rather than a hidden failure — worth reading if you're evaluating the routing logic itself.

## Architecture

![Architecture diagram](docs/architecture.svg)

A request first checks the semantic cache — if a near-duplicate question has been answered before, it's returned for free. On a cache miss, a heuristic complexity classifier routes the query to either `gpt-4o-mini` (simple/lookup questions) or `gpt-4o` (design, debugging, multi-step reasoning). Every request — cache hit or miss — is logged to Firestore with cost, latency, and routing metadata, which powers the Streamlit dashboard.

## Tech stack

- **Backend:** FastAPI
- **LLM provider:** OpenAI (`gpt-4o-mini` + `gpt-4o`, two-tier routing)
- **Semantic cache:** Jaccard word-overlap similarity in mock/dev mode; cosine similarity over `text-embedding-3-small` embeddings in production
- **Storage:** GCP Firestore (serverless, survives Cloud Run's stateless restarts)
- **Secrets:** GCP Secret Manager (API key never stored as a plain env var or in code)
- **Dashboard:** Streamlit
- **Deployment:** Docker + GCP Cloud Run (two services: backend + dashboard)
- **CI/CD:** GitHub Actions — tests run on every push/PR; auto-deploy to Cloud Run on merge to `main`

## Key decisions (and what actually went wrong)

**Two OpenAI tiers, not multiple providers.** Routing between `gpt-4o-mini` and `gpt-4o` keeps the cost/quality tradeoff the actual subject of the project, rather than diluting it across provider-specific quirks.

**The mock cache broke twice, and both breaks were real bugs, not edge cases.** The first version hashed prompt words into a fixed-size vector; it turned out to be biased by prompt *length* rather than topic — a 24-word unrelated question scored artificially similar to a 6-word one purely from hash collisions. Increasing dimensionality didn't fix it. The actual fix was switching mock-mode similarity to direct word-overlap (Jaccard similarity), which correctly separates genuine paraphrases (~0.75 similarity) from unrelated topics (~0.0). Along the way, a second bug surfaced: Python's built-in `hash()` is randomized per process by default, so identical test runs produced different similarity scores across sessions — fixed with a deterministic `hashlib`-based hash instead.

**SQLite locally, Firestore in production.** Cloud Run containers are stateless — a local SQLite file gets wiped on every restart or scaling event. Firestore persists independently of the container lifecycle and stays within GCP's free tier at this project's scale.

**Secret Manager over a plain environment variable.** The OpenAI key is injected at runtime via GCP Secret Manager (`--set-secrets`), never committed, never passed as a plain `--set-env-vars` value.

**CI/CD needed three rounds of IAM debugging before it worked.** The first deploy attempt via `gcloud run deploy --source .` failed three separate times on missing permissions (Artifact Registry access, then a Cloud Build staging bucket, then that bucket's storage permissions) for the dedicated `github-actions-deployer` service account. Rather than keep granting increasingly broad roles to chase a build system I didn't fully control, I switched the pipeline to build and push the Docker image explicitly, then deploy by image reference — the same pattern already proven working for the manually-deployed dashboard service. Fewer moving parts, and permissions became exactly as broad as needed and no broader.

## Routing logic

The complexity classifier is a heuristic, not a trained model — a deliberate weekend-scope decision. It flags a query as "complex" (routes to `gpt-4o`) based on: known complexity phrases ("design," "tradeoff," "walk through," "debug," etc.), prompt length, multiple question marks, and compound "X and Y" structure; known simple-lookup phrases counteract false positives.

The one eval miss: *"What's the time complexity of Python's built-in sort, and why?"* was labeled simple but routed to `gpt-4o`, tripped by the `" and "` conjunction signal. Arguably not wrong — the question does ask for two things — but it's a real precision/recall tradeoff in the heuristic worth being upfront about rather than relabeling the test case to force a clean pass.

## What I'd improve with more time

- Replace the mock/Firestore-rebuilt cache with a real vector index (Vertex AI Vector Search or pgvector) for production-scale semantic search
- Replace the heuristic router with a small trained classifier on labeled routing data
- Expand the eval set well beyond 18 queries and add an LLM-as-judge quality check, not just routing accuracy
- Move from a service-account JSON key to Workload Identity Federation for GitHub Actions (removes the long-lived key entirely)
- Add authentication in front of the public endpoints for anything beyond a portfolio demo

## Running locally

```bash
git clone https://github.com/SiddharthDataScientist/llm-costguard.git
cd llm-costguard
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
cp .env.example .env           # then add your OpenAI key and GCP project ID
uvicorn app.main:app --reload --port 8000
```

Dashboard:
```bash
streamlit run dashboard/app.py
```

Tests:
```bash
python -m pytest tests/ -v
```

Eval harness:
```bash
python -m eval.run_eval
```