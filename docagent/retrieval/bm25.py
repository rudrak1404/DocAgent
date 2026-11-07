"""Okapi BM25 on a sparse matrix.

The BM25 term weight of every (chunk, term) pair is precomputed into a CSR
matrix W, so scoring a query is a column slice and a row sum:
    score(d, q) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
Tokenisation keeps numbers intact (1,577.0 -> 1577.0) because financial
questions are often about specific figures and years.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the
this to was were will with which what how much did does do give their they company""".split())


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower())
    return [t.replace(",", "") for t in toks if t not in STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W: sp.csr_matrix | None = None

    def fit(self, texts: list[str]) -> "BM25":
        rows, cols, vals, lengths = [], [], [], []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for tok, c in counts.items():
                j = self.vocab.setdefault(tok, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(texts)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, len(self.vocab)))
        df = np.bincount(np.array(cols), minlength=len(self.vocab)).astype(np.float64)
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)          # Lucene / BM25+ style, always > 0
        dl = np.array(lengths, dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).tocsc()
        return self

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()

    def save(self, path: Path) -> None:
        sp.save_npz(path / "bm25_matrix.npz", self.W)
        (path / "bm25_vocab.txt").write_text("\n".join(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25":
        obj = cls()
        obj.W = sp.load_npz(path / "bm25_matrix.npz").tocsc()
        words = (path / "bm25_vocab.txt").read_text(encoding="utf-8").split("\n")
        obj.vocab = {w: i for i, w in enumerate(words)}
        return obj
