"""Plain-RAG baseline built with LangChain.

    question -> LangChain FAISS retriever (top-5, one shared store over all filings,
                no filters, no re-ranking) -> prompt template -> one Claude call

This is the "shared vector store" setup FinanceBench reports for its RAG
baselines. To isolate the effect of the pipeline, it uses the same chunks, the
same embedding vectors (the selected HuggingFace embedder) and the same Claude
model and settings as DocAgent; the LLM step is a RunnableLambda around the
shared Anthropic-SDK client (LLM), so both systems are billed and logged identically.
"""
from __future__ import annotations

import time

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from docagent.agent.llm import make_llm
from docagent.agent.prompts import BASELINE_SYSTEM
from docagent.config import EMBED_MODEL, EMBED_QUERY_PREFIX
from docagent.retrieval.index import HybridIndex


def _format(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['doc_name']} p.{d.metadata['page'] + 1}]\n{d.page_content}" for d in docs)


class PlainRAG:
    def __init__(self, index: HybridIndex, llm=None, k: int = 5, device: str | None = None):
        self.llm = llm or make_llm()
        emb = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, model_kwargs={"device": device or "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": EMBED_QUERY_PREFIX})
        vectors = index.faiss.reconstruct_n(0, index.faiss.ntotal)          # reuse the corpus vectors
        texts = [c.index_text() for c in index.chunks]
        metas = [{"chunk_id": c.chunk_id, "doc_name": c.doc_name, "page": c.page} for c in index.chunks]
        self.store = FAISS.from_embeddings(list(zip(texts, np.asarray(vectors).tolist())), emb, metadatas=metas,
                                           distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        self.retriever = self.store.as_retriever(search_kwargs={"k": k})
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", BASELINE_SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")])
        self.chain = (RunnableParallel(docs=self.retriever, question=RunnablePassthrough())
                      | RunnablePassthrough.assign(generation=RunnableLambda(self._generate)))

    async def _generate(self, x: dict) -> dict:
        pv = await self.prompt.ainvoke({"context": _format(x["docs"]), "question": x["question"]})
        msgs = pv.to_messages()
        system = "\n".join(m.content for m in msgs if m.type == "system")
        user = "\n".join(m.content for m in msgs if m.type == "human")
        resp, rec = await self.llm.create(system=system, messages=[{"role": "user", "content": user}], tag="baseline")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return {"text": text, "call": rec}

    async def ask(self, question: str) -> dict:
        t0 = time.time()
        try:
            out = await self.chain.ainvoke(question)
        except Exception as e:
            return {"answer": "", "status": "error", "issues": [repr(e)], "citations": [], "llm_calls": [],
                    "usage": {"cost_usd": 0.0}, "seconds": round(time.time() - t0, 2), "evidence_seen": []}
        rec = out["generation"]["call"]
        docs = out["docs"]
        return {
            "answer": out["generation"]["text"], "status": "answered",
            "citations": [{"evidence_id": d.metadata["chunk_id"], "doc_name": d.metadata["doc_name"],
                           "page": d.metadata["page"] + 1, "text": d.page_content} for d in docs],
            "llm_calls": [rec],
            "usage": {"input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"],
                      "cache_read": rec["cache_read"], "cache_write": rec["cache_write"],
                      "cost_usd": round(rec["cost_usd"], 5), "fallbacks": int(rec["fallback"])},
            "seconds": round(time.time() - t0, 2),
            "evidence_seen": [d.metadata["chunk_id"] for d in docs],
        }
