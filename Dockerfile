# Slim Python base — smaller image, faster Cloud Run cold starts
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer, so Docker caches this step
# and doesn't reinstall on every code change — only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY app/ ./app/

# Cloud Run injects the actual port to listen on via the PORT env var.
# Default to 8080 locally if not set (Cloud Run's convention, not 8000).
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}