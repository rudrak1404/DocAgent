"""Build the retrieval index from the downloaded filings.

    python -m docagent.ingest.build            # parse -> validate -> chunk -> embed -> index

Steps
 1. Parse every PDF (parallel, cached in data/parsed/).
 2. Validate page alignment against FinanceBench's gold evidence: for each gold
    evidence passage, find which nearby page of our extraction contains it best.
    This catches off-by-one page numbering before any metric depends on it.
 3. Chunk, embed (GPU if available) and write the FAISS + BM25 index.
"""
from __future__ import annotations

import json
import re
import time

from transformers import AutoTokenizer

from docagent.config import CHUNK_OVERLAP, CHUNK_TOKENS, EMBED_MODEL, INDEX_DIR, PARSED_DIR, PDF_DIR
from docagent.data import corpus_doc_names, load_docs, load_questions
from docagent.ingest.chunk import chunk_document
from docagent.ingest.pdf import load_parsed, parse_corpus


def _shingles(text: str, n: int = 3) -> set[str]:
    w = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def page_text(parsed, p: int) -> str:
    return parsed.pages[p].text


def validate_alignment(log=print) -> dict:
    """For each gold evidence passage, the share of its word 3-grams found on the
    gold page and on its neighbours (p-1, p+1)."""
    rows = []
    for q in load_questions():
        parsed = load_parsed(PARSED_DIR, q.doc_name)
        for ev in q.evidence:
            target = _shingles(ev.text)
            if len(target) < 5 or ev.page >= parsed.n_pages:
                rows.append({"id": q.id, "page": ev.page, "status": "skip"})
                continue
            cover = {}
            for p in (ev.page - 1, ev.page, ev.page + 1):
                if 0 <= p < parsed.n_pages:
                    cover[p] = len(target & _shingles(page_text(parsed, p))) / len(target)
            best = max(cover, key=cover.get)
            rows.append({"id": q.id, "page": ev.page, "gold_cover": cover.get(ev.page, 0.0),
                         "best_page": best, "best_cover": cover[best]})
    scored = [r for r in rows if "gold_cover" in r]
    summary = {
        "evidence_passages": len(rows),
        "checked": len(scored),
        "gold_page_is_best": sum(r["best_page"] == r["page"] for r in scored) / max(len(scored), 1),
        "gold_cover_mean": sum(r["gold_cover"] for r in scored) / max(len(scored), 1),
        "gold_cover_ge_50pct": sum(r["gold_cover"] >= 0.5 for r in scored) / max(len(scored), 1),
        "off_by_one": [r for r in scored if r["best_page"] != r["page"] and r["best_cover"] > r["gold_cover"] + 0.2],
    }
    log(f"alignment: {summary['checked']} gold passages checked; gold page is the best match for "
        f"{summary['gold_page_is_best']:.1%}; mean 3-gram coverage {summary['gold_cover_mean']:.1%}; "
        f"{len(summary['off_by_one'])} clear off-by-one cases")
    return summary


def main(workers: int = 6) -> None:
    t0 = time.time()
    names = corpus_doc_names()
    missing = [n for n in names if not (PDF_DIR / f"{n}.pdf").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} PDFs missing; run `python -m docagent.data` first")

    print("1/3 parsing PDFs ...")
    report = parse_corpus(PDF_DIR, PARSED_DIR, workers=workers)
    pages = sum(r[1] for r in report)
    empty = sum(r[2] for r in report)
    tables = sum(r[3] for r in report)
    print(f"    {len(report)} filings, {pages} pages, {empty} pages without text, {tables} pages with tables")

    print("2/3 validating page alignment against FinanceBench evidence ...")
    align = validate_alignment()

    print("3/3 chunking + embedding ...")
    from docagent.retrieval.index import HybridIndex
    from docagent.retrieval.models import Embedder
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    docs = load_docs()
    chunks = []
    for n in names:
        chunks += chunk_document(load_parsed(PARSED_DIR, n), docs[n], tok, CHUNK_TOKENS, CHUNK_OVERLAP)
    print(f"    {len(chunks)} chunks ({sum(c.kind == 'table' for c in chunks)} tables)")
    index = HybridIndex.build(chunks, Embedder())
    index.save(INDEX_DIR)
    stats = {"filings": len(report), "pages": pages, "pages_without_text": empty, "pages_with_tables": tables,
             "chunks": len(chunks), "table_chunks": sum(c.kind == "table" for c in chunks),
             "alignment": {k: v for k, v in align.items() if k != "off_by_one"},
             "off_by_one_cases": align["off_by_one"][:50],
             "build_seconds": round(time.time() - t0, 1)}
    (INDEX_DIR / "build_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"index written to {INDEX_DIR} in {stats['build_seconds']} s")


if __name__ == "__main__":
    main()
