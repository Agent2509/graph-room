# Build the static frontend, then ship it inside the API image.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    DATA_DIR=/app/data
WORKDIR /app

COPY backend/pyproject.toml ./
COPY backend/graph_rag ./graph_rag
RUN pip install --no-cache-dir .

# Demo seed needs the corpus and the recorded LLM responses.
COPY sample_data ./sample_data
COPY fixtures ./fixtures

# Bake the embedding model so the container starts without a download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY --from=web /web/out ./static

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "graph_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
