"""Layout-aware PDF page extraction with PyMuPDF.

Financial statements are tables, often without ruling lines. Plain text
extraction can emit a statement column by column, which separates each label
from its numbers. Instead, every page's words are regrouped into visual rows
(by vertical position) and a wide horizontal gap inside a row becomes a column
break " | ". A cash-flow row then reads

    Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)

which keeps label, values and (via the header row) the years together. This is
~40x faster than PyMuPDF's table finder (≈40 ms vs ≈1.7 s per page) and works
for ruled and borderless tables alike.

Page numbers are stored zero-indexed, matching FinanceBench's evidence pages.
"""
from __future__ import annotations

import json
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*\.?\s*[-–—:|]?\s*(.{3,90})$", re.I)
STATEMENT_RE = re.compile(
    r"^\s*(consolidated\s+|condensed\s+consolidated\s+)?"
    r"(statements?\s+of\s+[a-z ,()'&]{3,80}|balance\s+sheets?)\s*$"
    r"|^\s*notes\s+to\s+(the\s+)?(condensed\s+)?consolidated\s+financial\s+statements\b", re.I)
NUMERIC_CELL = re.compile(r"\|\s*[$(]?-?\d")


@dataclass
class Page:
    page: int                 # zero-indexed
    text: str                 # layout text: one visual row per line, " | " between columns
    section: str = ""
    table_rows: int = 0       # rows with at least one numeric cell


@dataclass
class ParsedDoc:
    doc_name: str
    n_pages: int
    pages: list[Page]


def _tidy(line: str) -> str:
    line = line.replace(" ", " ").replace(" ", " ").replace("—", "-").replace("–", "-")
    line = re.sub(r"\$\s*\|\s*", "$", line)          # "$ | 5,363" -> "$5,363"
    line = re.sub(r"\s*\|\s*%", "%", line)           # "24.1 | %"  -> "24.1%"
    line = re.sub(r"\(\s*\|\s*", "(", line)          # "( | 1,577" -> "(1,577"
    line = re.sub(r"\s*\|\s*\)", ")", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def layout_text(page) -> tuple[str, int]:
    words = page.get_text("words", sort=False)
    if not words:
        return "", 0
    h = statistics.median(w[3] - w[1] for w in words) or 8.0
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, cy = [], [], 0.0
    for w in words:
        yc = (w[1] + w[3]) / 2
        if cur and abs(yc - cy) > 0.45 * h:
            rows.append(cur)
            cur = []
        cur.append(w)
        cy = yc if len(cur) == 1 else cy + (yc - cy) / len(cur)
    if cur:
        rows.append(cur)
    lines = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        parts = [r[0][4]]
        for a, b in zip(r, r[1:]):
            parts.append((" | " if b[0] - a[2] > max(10.0, 1.2 * h) else " ") + b[4])
        line = _tidy("".join(parts))
        if line:
            lines.append(line)
    table_rows = sum(1 for ln in lines if NUMERIC_CELL.search(ln))
    return "\n".join(lines), table_rows


def _section_from(text: str, current: str) -> str:
    for line in text.splitlines()[:40]:
        line = line.replace("|", " ").strip()
        if not line or len(line) > 120:
            continue
        m = ITEM_RE.match(line)
        if m and not m.group(2).strip()[:1].isdigit():
            return f"Item {m.group(1).upper()}. {m.group(2).strip().rstrip('.')}"[:100]
        if STATEMENT_RE.match(line):
            return re.sub(r"\s+", " ", line).title()[:100]
    return current


def parse_pdf(path: Path) -> ParsedDoc:
    doc = pymupdf.open(path)
    pages, section = [], ""
    for pno in range(doc.page_count):
        text, table_rows = layout_text(doc[pno])
        section = _section_from(text, section)
        pages.append(Page(page=pno, text=text, section=section, table_rows=table_rows))
    n = doc.page_count
    doc.close()
    return ParsedDoc(doc_name=path.stem, n_pages=n, pages=pages)


def _parse_to_cache(args) -> tuple[str, int, int, int]:
    pdf, out_dir = args
    out = Path(out_dir) / f"{Path(pdf).stem}.json"
    if not out.exists():
        out.write_text(json.dumps(asdict(parse_pdf(Path(pdf)))), encoding="utf-8")
    d = json.loads(out.read_text(encoding="utf-8"))
    empty = sum(1 for p in d["pages"] if len(p["text"]) < 20)
    tables = sum(1 for p in d["pages"] if p["table_rows"] >= 5)
    return d["doc_name"], d["n_pages"], empty, tables


def parse_corpus(pdf_dir: Path, out_dir: Path, workers: int = 8, log=print) -> list[tuple]:
    """Parse every PDF in parallel into out_dir/<doc>.json; return a quality report
    (doc, pages, pages without text, pages with a table of >= 5 numeric rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    report = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_parse_to_cache, [(str(p), str(out_dir)) for p in pdfs]), 1):
            report.append(row)
            log(f"[{i}/{len(pdfs)}] {row[0]}: {row[1]} pages, {row[2]} empty, {row[3]} with tables")
    return report


def load_parsed(out_dir: Path, doc_name: str) -> ParsedDoc:
    d = json.loads((out_dir / f"{doc_name}.json").read_text(encoding="utf-8"))
    return ParsedDoc(d["doc_name"], d["n_pages"], [Page(**p) for p in d["pages"]])
