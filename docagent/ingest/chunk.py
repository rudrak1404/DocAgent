"""Page-bounded chunking of layout text.

* Windows of `CHUNK_TOKENS` tokens with `CHUNK_OVERLAP` overlap, measured with
  the embedding model's own tokenizer (offset mapping -> exact character spans,
  so a chunk is an exact substring of its page). Chunks never cross a page
  boundary, so every citation points to exactly one page.
* Column-header carry-over: if a window starts inside a table, the table's
  column-header row (e.g. "(Millions) | 2018 | 2017 | 2016") is prepended, so
  a number like "(1,373)" still says which year it belongs to.
* Every chunk carries metadata (company, filing type, period, page, section);
  its index text = a short contextual header + the chunk text. The header is
  what lets "3M capex 2018" match a cash-flow row that never says "3M".
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from docagent.data import DocInfo
from docagent.ingest.pdf import NUMERIC_CELL, ParsedDoc

DOC_TYPE_LABEL = {"10k": "10-K annual report", "10q": "10-Q quarterly report",
                  "8k": "8-K current report", "earnings": "earnings release"}
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    company: str
    doc_type: str
    period: str              # e.g. "2018" or "2023Q2"
    year: int
    page: int                # zero-indexed
    section: str
    kind: str                # "table" | "text"
    text: str

    def header(self) -> str:
        label = DOC_TYPE_LABEL.get(self.doc_type.lower(), self.doc_type)
        sec = f" | {self.section}" if self.section else ""
        return f"{self.company} {label} {self.period}{sec} | page {self.page + 1}"

    def index_text(self) -> str:
        return f"{self.header()}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def period_of(doc_name: str) -> str:
    """FinanceBench names are COMPANY_PERIOD_TYPE[_suffix]; the company itself may
    contain underscores (JOHNSON_JOHNSON) and 8-Ks carry a date suffix, so take
    the first token shaped like a period (2018 or 2023Q2)."""
    for tok in doc_name.split("_"):
        if re.fullmatch(r"\d{4}(Q[1-4])?", tok):
            return tok
    return ""


def is_column_header(line: str) -> bool:
    return "|" in line and len(YEAR.findall(line)) >= 2 and not NUMERIC_CELL.search(
        YEAR.sub("", line))


def token_spans(text: str, tokenizer, size: int, overlap: int) -> list[tuple[int, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if not offs:
        return []
    if len(offs) <= size:
        return [(0, len(text))]
    spans, step = [], size - overlap
    for start in range(0, len(offs), step):
        end = min(start + size, len(offs))
        a, b = offs[start][0], offs[end - 1][1]
        # widen to whole lines at the start (a table row is never cut in half)
        # and to whole words at the end
        a = text.rfind("\n", 0, a) + 1
        while b < len(text) and not text[b].isspace():
            b += 1
        spans.append((a, b))
        if end == len(offs):
            break
    return spans


def chunk_document(parsed: ParsedDoc, info: DocInfo, tokenizer, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    period = period_of(parsed.doc_name)
    for page in parsed.pages:
        text = page.text
        for i, (a, b) in enumerate(token_spans(text, tokenizer, size, overlap)):
            body = text[a:b].strip()
            if len(body) < 20:
                continue
            # carry the nearest column-header row above this window into it
            before = text[:a].splitlines()
            head = next((ln for ln in reversed(before[-60:]) if is_column_header(ln)), None)
            if head and head not in body:
                body = f"[columns: {head}]\n{body}"
            lines = body.splitlines()
            numeric = sum(1 for ln in lines if NUMERIC_CELL.search(ln))
            kind = "table" if numeric >= max(3, 0.4 * len(lines)) else "text"
            chunks.append(Chunk(
                chunk_id=f"{parsed.doc_name}:p{page.page}:c{i}", doc_name=parsed.doc_name,
                company=info.company, doc_type=info.doc_type, period=period, year=info.doc_period,
                page=page.page, section=page.section, kind=kind, text=body))
    return chunks
