# DocAgent API image (CPU). The retrieval index and parsed pages are built
# beforehand (python -m docagent.ingest.build) and copied in; the HuggingFace
# models are downloaded at build time, so containers start without Hub access.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles CUDA and is ~10x larger)
RUN pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# Non-root user; files are copied with the right owner (a later `chown -R` would
# duplicate every file into a new image layer).
RUN useradd --create-home --uid 10001 app && mkdir -p /app/.hf && chown app:app /app /app/.hf
USER app

# Hosted configuration. A cross-encoder re-ranker is slow on CPU-only hosts, so the
# default hosted mode is dense retrieval; pass --build-arg RETRIEVAL_MODE=dense_rerank
# (and run on a GPU) to serve the evaluated configuration.
ARG EMBED_MODEL=BAAI/bge-large-en-v1.5
ARG RERANK_MODEL=BAAI/bge-reranker-v2-m3
ARG RETRIEVAL_MODE=dense
ENV DOCAGENT_EMBED_MODEL=${EMBED_MODEL} DOCAGENT_RERANK_MODEL=${RERANK_MODEL} DOCAGENT_RETRIEVAL_MODE=${RETRIEVAL_MODE}
RUN python -c "import os; from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification; \
from sentence_transformers import SentenceTransformer; \
e, r = os.environ['DOCAGENT_EMBED_MODEL'], os.environ['DOCAGENT_RERANK_MODEL']; \
AutoTokenizer.from_pretrained(e); AutoModel.from_pretrained(e); SentenceTransformer(e); \
'rerank' in os.environ['DOCAGENT_RETRIEVAL_MODE'] and (AutoTokenizer.from_pretrained(r), AutoModelForSequenceClassification.from_pretrained(r))"

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app docagent/ docagent/
COPY --chown=app:app data/financebench/ data/financebench/
COPY --chown=app:app data/parsed/ data/parsed/
COPY --chown=app:app data/index/ data/index/
COPY --chown=app:app results/ results/

ENV PORT=8080 HF_HUB_OFFLINE=1 DOCAGENT_PUBLIC=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn docagent.api.app:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
