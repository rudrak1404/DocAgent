"""DocAgent web service.

    GET  /            minimal web UI
    GET  /health      liveness + loaded configuration
    GET  /documents   filings in the index
    GET  /examples    precomputed FinanceBench answers (no LLM call, no key needed)
    POST /ask         {"question": ..., "mode": "agent" | "baseline"}  -> answer, citations, trace
    /mcp              the DocAgent MCP server (Streamable HTTP) for any MCP client

/ask and /mcp require the `X-API-Key` header when DOCAGENT_API_KEY is set (every
/ask call spends LLM credit), and /ask is rate-limited per client IP.

    uvicorn docagent.api.app:create_app --factory --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from docagent import __version__

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).resolve().parents[2] / "results" / "examples.json"


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    mode: Literal["agent", "baseline"] = "agent"


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.time(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(index=None, docs=None, llm=None, api_key: str | None = None, with_baseline: bool = True) -> FastAPI:
    """Build the app. Arguments allow tests to inject a small index and a fake LLM."""
    from mcp.server.transport_security import TransportSecuritySettings

    from docagent.agent.graph import DocAgent
    from docagent.agent.llm import LLM
    from docagent.mcp_server import create_server

    if index is None:
        from docagent.retrieval.service import load_index
        index = load_index()
    if docs is None:
        from docagent.data import load_docs
        docs = load_docs()
    llm = llm or LLM()
    api_key = api_key if api_key is not None else os.getenv("DOCAGENT_API_KEY")
    limiter = RateLimiter(int(os.getenv("DOCAGENT_RATE_PER_MIN", 10)))

    mcp = create_server(index, docs)
    hosts = [h.strip() for h in os.getenv("DOCAGENT_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = (TransportSecuritySettings(allowed_hosts=hosts) if hosts else
                TransportSecuritySettings(enable_dns_rebinding_protection=False)
                if os.getenv("DOCAGENT_PUBLIC") == "1" else None)
    mcp_app = mcp.streamable_http_app(streamable_http_path="/", transport_security=security)
    agent = DocAgent(mcp, llm)
    baseline = None
    if with_baseline:
        from docagent.baseline import PlainRAG
        baseline = PlainRAG(index, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():       # a mounted app's own lifespan never runs
            yield

    app = FastAPI(title="DocAgent", version=__version__, lifespan=lifespan,
                  description="Agentic RAG analyst for SEC filings (FinanceBench).")

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        protected = request.url.path.startswith("/mcp") or request.url.path == "/ask"
        if protected and api_key:
            given = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
            if given != api_key:
                return JSONResponse({"detail": "missing or invalid X-API-Key"}, status_code=401)
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "llm": llm.model, "chunks": len(index.chunks),
                "filings": len({c.doc_name for c in index.chunks}), "retrieval_mode": index.mode,
                "auth_required": bool(api_key)}

    @app.get("/documents")
    async def documents():
        names = sorted({c.doc_name for c in index.chunks})
        return [{"doc_name": n, "company": docs[n].company, "doc_type": docs[n].doc_type,
                 "period": docs[n].doc_period} for n in names]

    @app.get("/examples")
    async def examples():
        if not EXAMPLES.exists():
            return []
        return json.loads(EXAMPLES.read_text(encoding="utf-8"))

    @app.post("/ask")
    async def ask(req: AskRequest, request: Request):
        client = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
        if not limiter.allow(client):
            raise HTTPException(429, "rate limit exceeded, try again in a minute")
        system = agent if req.mode == "agent" or baseline is None else baseline
        res = await system.ask(req.question)
        res.pop("llm_calls", None)
        for c in res.get("citations", []):
            c["text"] = c["text"][:1500]
        return {"mode": req.mode, **res}

    app.mount("/mcp", mcp_app)
    return app
