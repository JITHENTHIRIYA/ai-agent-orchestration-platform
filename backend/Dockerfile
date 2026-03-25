# ============================================================================
# Dockerfile — AI Agent Orchestration Platform (Backend)
# ============================================================================
# Multi-stage build for a lightweight production image.
# Usage:
#   docker build --tag ai-agent-platform --file backend/Dockerfile .
#   docker run -p 8000:8000 --env-file backend/.env ai-agent-platform
# ============================================================================

# ── Stage 1: Python base image ──────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: Install dependencies ───────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching.
# Dependencies are re-installed only when requirements.txt changes.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Copy application code ──────────────────────────────────────────
COPY backend/ .

# ── Expose the API port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Run the FastAPI server with uvicorn ──────────────────────────────────────
# --host 0.0.0.0  → listen on all interfaces (required inside containers)
# --port 8000     → match the EXPOSE directive above
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
