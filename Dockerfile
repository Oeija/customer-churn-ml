# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files + package metadata (needed for editable install)
COPY pyproject.toml uv.lock README.md ./

# Copy source code (needed for editable install via -e ".[dev]")
COPY src/ ./src/

# Install dependencies into a virtual environment
RUN uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code and artifacts
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Ensure model artifacts directory exists (mount at runtime or copy during build)
RUN mkdir -p artifacts

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run FastAPI server
CMD ["uvicorn", "src.customer_churn_ml.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
