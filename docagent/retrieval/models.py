"""HuggingFace Transformers models: bi-encoder embedder and cross-encoder re-ranker.

Embedder (BAAI/bge-*-en-v1.5): the [CLS] vector of the last hidden layer,
L2-normalised, so the inner product is cosine similarity. Queries get BGE's
retrieval instruction prefix; passages do not (asymmetric retrieval).

Re-ranker (BAAI/bge-reranker-*): reads query and passage *together* through full
cross-attention and outputs one relevance logit. More accurate than comparing two
independent vectors, but too slow for the whole corpus, so it only re-orders the
first-stage retriever's top candidates.

Both models are guarded by a lock: the API and the MCP server call them from
worker threads, and one forward pass at a time keeps GPU/CPU memory bounded.
"""
from __future__ import annotations

import threading

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from docagent.config import EMBED_MODEL, EMBED_QUERY_PREFIX, RERANK_MODEL


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class Embedder:
    def __init__(self, name: str = EMBED_MODEL, device: str | None = None, max_length: int = 512):
        self.name = name
        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        self._lock = threading.Lock()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64, query: bool = False) -> np.ndarray:
        if query:
            texts = [EMBED_QUERY_PREFIX + t for t in texts]
        order = np.argsort([-len(t) for t in texts])          # length-sorted batches: less padding
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with self._lock:
            for i in range(0, len(texts), batch_size):
                idx = order[i:i + batch_size]
                enc = self.tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0].float()
                out[idx] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
        return out


class Reranker:
    def __init__(self, name: str = RERANK_MODEL, device: str | None = None, max_length: int = 512):
        self.name = name
        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval().to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length
        self._lock = threading.Lock()

    @torch.inference_mode()
    def score(self, query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
        scores = []
        with self._lock:
            for i in range(0, len(passages), batch_size):
                enc = self.tokenizer([query] * len(passages[i:i + batch_size]), passages[i:i + batch_size],
                                     padding=True, truncation="only_second", max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                scores.append(self.model(**enc).logits.view(-1).float().cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
