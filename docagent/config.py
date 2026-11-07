"""Central configuration. Values can be overridden with environment variables
(or a .env file in the project root)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:                                   # optional: values in a .env file (never committed)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(Path.home() / ".docagent.env")
except ImportError:
    pass

DATA_DIR = Path(os.getenv("DOCAGENT_DATA_DIR", ROOT / "data"))
FB_DIR = DATA_DIR / "financebench"           # questions + document metadata (JSONL)
PDF_DIR = DATA_DIR / "pdfs"                  # downloaded filings (never committed)
PARSED_DIR = DATA_DIR / "parsed"             # per-document page extraction cache
INDEX_DIR = Path(os.getenv("DOCAGENT_INDEX_DIR", DATA_DIR / "index"))
RESULTS_DIR = ROOT / "results"

# Retrieval configuration (HuggingFace Hub ids). Defaults = the configuration
# chosen on the dev split by docagent/retrieval/select.py.
EMBED_MODEL = os.getenv("DOCAGENT_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_INDEX_SUBDIR = os.getenv("DOCAGENT_EMBED_INDEX_SUBDIR", "bge-large")   # FAISS index for EMBED_MODEL
RERANK_MODEL = os.getenv("DOCAGENT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVAL_MODE = os.getenv("DOCAGENT_RETRIEVAL_MODE", "dense_rerank")
RETRIEVAL_CANDIDATES = int(os.getenv("DOCAGENT_RETRIEVAL_CANDIDATES", 40))
RERANK_MAX_LENGTH = int(os.getenv("DOCAGENT_RERANK_MAX_LENGTH", 512))   # lower on CPU-only hosts
BM25_WEIGHT = float(os.getenv("DOCAGENT_BM25_WEIGHT", 1.0))
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking
CHUNK_TOKENS = int(os.getenv("DOCAGENT_CHUNK_TOKENS", 350))
CHUNK_OVERLAP = int(os.getenv("DOCAGENT_CHUNK_OVERLAP", 60))

# LLM (Claude via the official Anthropic SDK)
LLM_MODEL = os.getenv("DOCAGENT_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("DOCAGENT_LLM_EFFORT", "high")
JUDGE_MODEL = os.getenv("DOCAGENT_JUDGE_MODEL", "claude-opus-5-5")

# Agent limits
MAX_TOOL_CALLS = int(os.getenv("DOCAGENT_MAX_TOOL_CALLS", 14))
MAX_VERIFY_RETRIES = int(os.getenv("DOCAGENT_MAX_VERIFY_RETRIES", 2))

# Price table (USD per million tokens) used for cost accounting in evaluations.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}
