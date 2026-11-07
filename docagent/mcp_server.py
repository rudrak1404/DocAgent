"""DocAgent MCP server: the agent's tools, exposed over the Model Context Protocol.

Built on the official MCP Python SDK (v2, `MCPServer`). Tool signatures are the
input schemas and the Pydantic return types are the output schemas, so any MCP
client (this project's LangGraph agent, Claude Desktop, Claude Code, ...) sees
typed tools. Retrieval runs in a worker thread so the event loop stays free.

Run standalone:
    python -m docagent.mcp_server              # stdio (e.g. for Claude Desktop)
    python -m docagent.mcp_server --http 8001  # Streamable HTTP at http://localhost:8001/mcp
"""
from __future__ import annotations

import argparse
from typing import Literal

import anyio
from mcp.server import MCPServer
from pydantic import BaseModel, Field

from docagent.config import PARSED_DIR
from docagent.ingest.chunk import DOC_TYPE_LABEL, period_of
from docagent.ingest.pdf import load_parsed
from docagent.retrieval.index import HybridIndex, norm_company
from docagent.tools.calculator import RATIOS, CalcError, safe_eval
from docagent.tools.calculator import financial_ratio as compute_ratio

INSTRUCTIONS = """Tools for answering questions about company SEC filings (10-K annual reports,
10-Q quarterly reports, 8-K current reports, earnings releases). Typical flow: list_documents to
find the right filing, search_documents (filter by company / fiscal_year / doc_name) to find
passages, read_page to see a full page such as a financial statement, then calculate or
financial_ratio for any arithmetic. Every passage has an evidence_id to cite."""


class DocumentInfo(BaseModel):
    doc_name: str
    company: str
    doc_type: str = Field(description="10-K annual report, 10-Q quarterly report, 8-K current report or earnings release")
    period: str = Field(description="fiscal year (2018) or fiscal quarter (2023Q2) the filing covers")
    pages: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class Passage(BaseModel):
    evidence_id: str = Field(description="cite this id for any fact taken from the passage")
    doc_name: str
    company: str
    period: str
    page: int = Field(description="1-based page number in the filing PDF")
    section: str
    text: str
    relevance: float


class SearchResults(BaseModel):
    query: str
    filters: dict
    passages: list[Passage]


class PageContent(BaseModel):
    evidence_id: str = Field(description="cite this id for any fact taken from the page")
    doc_name: str
    page: int = Field(description="1-based page number")
    pages_in_document: int
    section: str
    text: str = Field(description="page text; table rows are single lines with ' | ' between columns")


class Calculation(BaseModel):
    expression: str
    value: float


class RatioResult(BaseModel):
    ratio: str
    formula: str
    inputs: dict[str, float]
    value: float
    unit: str
    value_percent: float | None = None


def create_server(index: HybridIndex, docs: dict, parsed_dir=PARSED_DIR) -> MCPServer:
    """Build the MCP server around a loaded retrieval index and document metadata."""
    mcp = MCPServer("docagent", instructions=INSTRUCTIONS)
    corpus = sorted({c.doc_name for c in index.chunks})
    n_pages = {}

    def page_count(doc_name: str) -> int:
        if doc_name not in n_pages:
            n_pages[doc_name] = load_parsed(parsed_dir, doc_name).n_pages
        return n_pages[doc_name]

    @mcp.tool()
    async def list_documents(company: str | None = None) -> DocumentList:
        """List the filings available for search, optionally only one company's
        (matching is case-insensitive and ignores suffixes like Inc. or Corporation)."""
        out = []
        for name in corpus:
            d = docs[name]
            if company:
                q, c = norm_company(company), norm_company(d.company)
                if not (q and (q == c or q in c or c in q)):
                    continue
            out.append(DocumentInfo(doc_name=name, company=d.company,
                                    doc_type=DOC_TYPE_LABEL.get(d.doc_type.lower(), d.doc_type),
                                    period=period_of(name), pages=page_count(name)))
        return DocumentList(documents=out)

    @mcp.tool()
    async def search_documents(
        query: str,
        company: str | None = None,
        fiscal_year: int | None = None,
        doc_type: Literal["10k", "10q", "8k", "earnings"] | None = None,
        doc_name: str | None = None,
        top_k: int = 8,
    ) -> SearchResults:
        """Hybrid search (BM25 + dense embeddings, re-ranked by a cross-encoder) over the
        filings. Use specific financial wording (e.g. "purchases of property, plant and
        equipment"); narrow with company, fiscal_year (the filing's fiscal year), doc_type or an
        exact doc_name from list_documents. Returns passages with evidence_ids."""
        top_k = max(1, min(int(top_k), 15))
        hits = await anyio.to_thread.run_sync(
            lambda: index.search(query, top_k=top_k, company=company, year=fiscal_year,
                                 doc_type=doc_type, doc_name=doc_name))
        return SearchResults(
            query=query,
            filters={k: v for k, v in dict(company=company, fiscal_year=fiscal_year, doc_type=doc_type,
                                           doc_name=doc_name).items() if v},
            passages=[Passage(evidence_id=h.chunk.chunk_id, doc_name=h.chunk.doc_name, company=h.chunk.company,
                              period=h.chunk.period, page=h.chunk.page + 1, section=h.chunk.section,
                              text=h.chunk.text, relevance=round(h.score, 4)) for h in hits])

    @mcp.tool()
    async def read_page(doc_name: str, page: int) -> PageContent:
        """Return the full text of one page (1-based page number) of a filing, e.g. to read a
        whole financial statement or the context around a search result."""
        if doc_name not in docs or doc_name not in corpus:
            raise ValueError(f"unknown doc_name {doc_name!r}; call list_documents")
        parsed = await anyio.to_thread.run_sync(lambda: load_parsed(parsed_dir, doc_name))
        if not 1 <= page <= parsed.n_pages:
            raise ValueError(f"{doc_name} has pages 1..{parsed.n_pages}")
        p = parsed.pages[page - 1]
        return PageContent(evidence_id=f"{doc_name}:p{page - 1}:page", doc_name=doc_name, page=page,
                           pages_in_document=parsed.n_pages, section=p.section, text=p.text)

    @mcp.tool()
    async def calculate(expression: str) -> Calculation:
        """Evaluate an arithmetic expression exactly, e.g. "(6439 - 1577) / 32765 * 100".
        Numbers, + - * / ** %, parentheses, abs/round/min/max/sqrt/log/exp. Use this for every
        computed figure instead of doing arithmetic in your head."""
        try:
            return Calculation(expression=expression, value=safe_eval(expression))
        except CalcError as e:
            raise ValueError(str(e)) from None

    ratio_doc = ("Compute a named financial ratio from its inputs using a fixed definition. Averages use "
                 "the opening and closing balance (e.g. total_assets_begin, total_assets_end). Ratios: "
                 + "; ".join(f"{k} = {v.formula}" for k, v in RATIOS.items()))

    @mcp.tool(description=ratio_doc)
    async def financial_ratio(ratio: str, inputs: dict[str, float]) -> RatioResult:
        try:
            return RatioResult(**compute_ratio(ratio, inputs))
        except CalcError as e:
            raise ValueError(str(e)) from None

    return mcp


def load_default_server() -> MCPServer:
    """Load the built index and models from data/ and return a ready server."""
    from docagent.retrieval.service import load_index
    from docagent.data import load_docs
    return create_server(load_index(), load_docs())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, help="serve Streamable HTTP on this port instead of stdio")
    a = ap.parse_args()
    server = load_default_server()
    if a.http:
        server.run("streamable-http", port=a.http)
    else:
        server.run("stdio")
