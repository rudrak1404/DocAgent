"""Shared fixtures: a tiny corpus with deterministic fake models, so tests run
offline in seconds (no model downloads, no LLM calls)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pytest

from docagent.data import DocInfo
from docagent.ingest.chunk import Chunk
from docagent.ingest.pdf import Page, ParsedDoc
from docagent.retrieval.bm25 import BM25
from docagent.retrieval.index import HybridIndex

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words vectors: same words -> similar vectors."""
    dim = DIM

    def encode(self, texts, batch_size=64, query=False):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class FakeReranker:
    def score(self, query, passages, batch_size=32):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return np.array([len(q & set(re.findall(r"[a-z0-9]+", p.lower()))) for p in passages], dtype=np.float32)


PAGES = {
    "ACME_2018_10K": [
        "ACME Corp annual report 2018\nItem 7. Management's Discussion and Analysis\nRevenue grew because of strong demand.",
        "Consolidated Statement of Cash Flows\n(Millions) | 2018 | 2017\nNet cash provided by operating activities | 6,439 | 6,240\n"
        "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)",
    ],
    "ACME_2019_10K": [
        "ACME Corp annual report 2019\nRevenue | 32,136 | 32,765\nNet income | 4,570 | 5,349",
    ],
    "GLOBEX_2018_10K": [
        "Globex Corporation 2018 annual report\nTotal current assets | 9,000 | 8,000\nTotal current liabilities | 4,500 | 4,000",
    ],
}
DOCS = {
    "ACME_2018_10K": DocInfo("ACME_2018_10K", "ACME", "Industrials", "10k", 2018),
    "ACME_2019_10K": DocInfo("ACME_2019_10K", "ACME", "Industrials", "10k", 2019),
    "GLOBEX_2018_10K": DocInfo("GLOBEX_2018_10K", "Globex", "Tech", "10k", 2018),
}


@pytest.fixture
def parsed_dir(tmp_path: Path) -> Path:
    for name, pages in PAGES.items():
        doc = ParsedDoc(name, len(pages), [Page(page=i, text=t, section="") for i, t in enumerate(pages)])
        (tmp_path / f"{name}.json").write_text(json.dumps(asdict(doc)), encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_index() -> HybridIndex:
    chunks = []
    for name, pages in PAGES.items():
        info = DOCS[name]
        for i, text in enumerate(pages):
            chunks.append(Chunk(chunk_id=f"{name}:p{i}:c0", doc_name=name, company=info.company, doc_type="10k",
                                period=str(info.doc_period), year=info.doc_period, page=i, section="",
                                kind="table" if "|" in text else "text", text=text))
    emb = FakeEmbedder()
    vecs = emb.encode([c.index_text() for c in chunks])
    fx = faiss.IndexFlatIP(DIM)
    fx.add(vecs)
    idx = HybridIndex(chunks, fx, BM25().fit([c.index_text() for c in chunks]), emb, FakeReranker())
    idx.mode = "hybrid_rerank"
    return idx


@pytest.fixture
def docs() -> dict:
    return DOCS
