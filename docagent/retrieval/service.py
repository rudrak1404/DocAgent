"""Load the retrieval index in the configured setup. Defaults = the configuration
chosen on the dev split by docagent/retrieval/select.py (bge-large embedder +
bge-reranker-v2-m3 cross-encoder; see results/retrieval_selection.md). The MCP
server, the baseline and the API all use this one loader; environment variables
select a lighter setup on CPU-only hosts (e.g. DOCAGENT_RETRIEVAL_MODE=dense)."""
from __future__ import annotations

from docagent.config import (BM25_WEIGHT, EMBED_INDEX_SUBDIR, EMBED_MODEL, INDEX_DIR, RERANK_MAX_LENGTH,
                             RERANK_MODEL, RETRIEVAL_CANDIDATES, RETRIEVAL_MODE)
from docagent.retrieval.index import HybridIndex
from docagent.retrieval.models import Embedder, Reranker


def load_index(device: str | None = None) -> HybridIndex:
    reranker = (Reranker(RERANK_MODEL, device=device, max_length=RERANK_MAX_LENGTH)
                if "rerank" in RETRIEVAL_MODE else None)
    faiss_file = INDEX_DIR / EMBED_INDEX_SUBDIR / "faiss.index" if EMBED_INDEX_SUBDIR else None
    index = HybridIndex.load(INDEX_DIR, Embedder(EMBED_MODEL, device=device), reranker, faiss_file=faiss_file)
    index.mode, index.bm25_weight, index.candidates = RETRIEVAL_MODE, BM25_WEIGHT, RETRIEVAL_CANDIDATES
    return index
