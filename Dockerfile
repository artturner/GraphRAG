# =============================================================================
# Grounded GraphRAG Tutor — Multi-stage Docker build
# =============================================================================
#
# Build:  docker build -t graphrag-tutor .
# Run:    docker run -p 8000:8000 --env-file .env graphrag-tutor
#

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# OS-level build dependencies (needed for faiss-cpu, sentence-transformers, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata first (Docker layer caching)
COPY pyproject.toml ./

# Create a virtual-env and install project dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the project in "dependency-only" mode, then again with source
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image with only what's needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="GraphRAG Team"
LABEL description="Grounded GraphRAG Tutor API"

WORKDIR /app

# Minimal runtime OS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual-env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Install the project package itself (source-only, deps already in venv)
RUN pip install --no-cache-dir --no-deps -e .

# Pre-cache the sentence-transformers model so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Create directories the app may write to
RUN mkdir -p /app/data /app/.vectorstore /app/reports

# Bundle the pre-built vector store (run ingestion locally first)
COPY .vectorstore/ /app/.vectorstore/

# Default configuration
ENV CONFIG_PATH=configs/default.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the API port
EXPOSE 8000

# Health-check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint: run the FastAPI server
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
