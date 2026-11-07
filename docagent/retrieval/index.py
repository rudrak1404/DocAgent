"""Hybrid retrieval: FAISS dense search + BM25, fused with Reciprocal Rank
Fusion, then re-ranked by a cross-encoder. Optional metadata filters (company,
fiscal year, filing type, document) are applied *before* ranking, so a filtered
search never wastes its top-k on other companies' filings.

Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over rankers of
1 / (60 + rank of d). It needs no score calibration between BM25 and cosine
similarity, which live on unrelated scales.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from docagent.ingest.chunk import Chunk
from docagent.retrieval.bm25 import BM25

RRF_K = 60
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank", "dense_rerank")
_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "company", "co", "plc", "ltd",
             "limited", "the", "group", "holdings", "holding", "sa", "nv", "ag"}


def norm_company(name: str) -> str:
    toks = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("&", " ")).split()
    return " ".join(t for t in toks if t not in _SUFFIXES)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class HybridIndex:
    def __init__(self, chunks: list[Chunk], faiss_index, bm25: BM25, embedder=None, reranker=None):
        self.chunks = chunks
        self.faiss = faiss_index
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.mode, self.bm25_weight, self.candidates = "dense_rerank", 1.0, 40   # defaults; see service.py
        self._company = np.array([norm_company(c.company) for c in chunks])
        self._year = np.array([c.year for c in chunks])
        self._doc = np.array([c.doc_name for c in chunks])
        self._type = np.array([c.doc_type.lower() for c in chunks])
        self.by_id = {c.chunk_id: c for c in chunks}

    # ------------------------------------------------------------- build / io
    @classmethod
    def build(cls, chunks: list[Chunk], embedder, batch_size: int = 64) -> "HybridIndex":
        emb = embedder.encode([c.index_text() for c in chunks], batch_size=batch_size)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        bm25 = BM25().fit([c.index_text() for c in chunks])
        return cls(chunks, index, bm25, embedder)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(path / "faiss.index"))
        self.bm25.save(path)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path, embedder=None, reranker=None, faiss_file: Path | None = None) -> "HybridIndex":
        """faiss_file: vectors from another embedding model (same chunks, same order)."""
        chunks = [Chunk(**json.loads(line)) for line in
                  (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        fx = faiss.read_index(str(faiss_file or path / "faiss.index"))
        if fx.ntotal != len(chunks):
            raise ValueError(f"{fx.ntotal} vectors for {len(chunks)} chunks")
        return cls(chunks, fx, BM25.load(path), embedder, reranker)

    # --------------------------------------------------------------- filters
    def companies(self) -> list[str]:
        return sorted({c.company for c in self.chunks})

    def mask(self, company: str | None = None, year: int | None = None,
             doc_type: str | None = None, doc_name: str | None = None) -> np.ndarray | None:
        m = np.ones(len(self.chunks), dtype=bool)
        if company:
            q = norm_company(company)
            m &= np.array([bool(q) and (q == c or q in c or c in q) for c in self._company])
        if year:
            m &= self._year == int(year)
        if doc_type:
            m &= self._type == doc_type.lower().replace("-", "")
        if doc_name:
            m &= self._doc == doc_name
        return None if m.all() else m

    # --------------------------------------------------------------- rankers
    def dense(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], query=True)
        if mask is None:
            scores, ids = self.faiss.search(q, k)
        else:
            allowed = np.flatnonzero(mask).astype(np.int64)
            if len(allowed) == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
            scores, ids = self.faiss.search(q, min(k, len(allowed)), params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.bm25.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(k, int(np.isfinite(s).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    @staticmethod
    def rrf(*rankings: list[tuple[int, float]], weights: tuple[float, ...] | None = None) -> list[tuple[int, float]]:
        """Weighted reciprocal rank fusion: sum of w / (60 + rank) over the rankings."""
        weights = weights or (1.0,) * len(rankings)
        fused: dict[int, float] = {}
        for w, ranking in zip(weights, rankings):
            for rank, (i, _) in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0.0) + w / (RRF_K + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def rerank(self, query: str, cand: list[int]) -> list[tuple[int, float]]:
        scores = self.reranker.score(query, [self.chunks[i].index_text() for i in cand])
        return sorted(zip(cand, scores.tolist()), key=lambda kv: -kv[1])

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 8, company: str | None = None, year: int | None = None,
               doc_type: str | None = None, doc_name: str | None = None, mode: str | None = None,
               candidates: int | None = None, bm25_weight: float | None = None) -> list[Hit]:
        """mode: bm25 | dense | hybrid (weighted RRF of dense and BM25) |
        hybrid_rerank | dense_rerank (cross-encoder re-ranks the candidates).
        Unset arguments use the index's configured defaults."""
        mode = mode or self.mode
        candidates = candidates or self.candidates
        bm25_weight = self.bm25_weight if bm25_weight is None else bm25_weight
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        m = self.mask(company, year, doc_type, doc_name)
        if m is not None and not m.any():
            return []
        if mode == "bm25":
            ranked = self.sparse(query, top_k, m)
        elif mode == "dense":
            ranked = self.dense(query, top_k, m)
        elif mode == "dense_rerank":
            ranked = self.rerank(query, [i for i, _ in self.dense(query, candidates, m)])
        else:
            ranked = self.rrf(self.dense(query, candidates, m), self.sparse(query, candidates, m),
                              weights=(1.0, bm25_weight))
            if mode == "hybrid_rerank" and ranked:
                ranked = self.rerank(query, [i for i, _ in ranked[:candidates]])
        return [Hit(self.chunks[i], float(s), r) for r, (i, s) in enumerate(ranked[:top_k], 1)]
