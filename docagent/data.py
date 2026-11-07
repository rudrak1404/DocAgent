"""FinanceBench loader and PDF downloader.

FinanceBench (Islam et al., 2023, Patronus AI) open-source sample: 150 questions
over real SEC filings, each with a human gold answer and the gold evidence page
(zero-indexed). PDFs come from the public GitHub repository and are verified
against the git blob SHA-1 that GitHub reports, so a corrupted or truncated
download is detected. The PDFs are not redistributed with this project.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from docagent.config import FB_DIR, PDF_DIR

REPO = "patronus-ai/financebench"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
API = f"https://api.github.com/repos/{REPO}/contents"
UA = {"User-Agent": "docagent-research-downloader"}


@dataclass
class Evidence:
    doc_name: str
    page: int                 # zero-indexed, as in FinanceBench
    text: str


@dataclass
class Question:
    id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str
    question: str
    answer: str
    justification: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class DocInfo:
    doc_name: str
    company: str
    sector: str
    doc_type: str             # 10k, 10q, 8k, Earnings
    doc_period: int


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def download_metadata() -> None:
    FB_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
        path = FB_DIR / name
        if not path.exists():
            path.write_bytes(_fetch(f"{RAW}/data/{name}"))


def load_questions() -> list[Question]:
    out = []
    for line in (FB_DIR / "financebench_open_source.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append(Question(
            id=r["financebench_id"], company=r["company"], doc_name=r["doc_name"],
            question_type=r["question_type"], question_reasoning=r.get("question_reasoning") or "",
            question=r["question"], answer=r["answer"], justification=r.get("justification") or "",
            evidence=[Evidence(e["doc_name"], int(e["evidence_page_num"]), e["evidence_text"])
                      for e in r["evidence"]],
        ))
    return out


def load_docs() -> dict[str, DocInfo]:
    out = {}
    for line in (FB_DIR / "financebench_document_information.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["doc_name"]] = DocInfo(r["doc_name"], r["company"], r.get("gics_sector", ""),
                                     r["doc_type"], int(r["doc_period"]))
    return out


def corpus_doc_names() -> list[str]:
    """The filings referenced by the 150 open-source questions."""
    return sorted({q.doc_name for q in load_questions()})


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_pdfs(doc_names: list[str], dest: Path = PDF_DIR, log=print) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    listing = {e["name"]: e for e in json.loads(_fetch(f"{API}/pdfs"))}
    for i, name in enumerate(doc_names, 1):
        entry = listing.get(f"{name}.pdf")
        if entry is None:
            raise FileNotFoundError(f"{name}.pdf not in {REPO}/pdfs")
        path = dest / f"{name}.pdf"
        if path.exists() and git_blob_sha1(path.read_bytes()) == entry["sha"]:
            continue
        blob = _fetch(f"{RAW}/pdfs/{name}.pdf")
        if git_blob_sha1(blob) != entry["sha"]:
            raise IOError(f"checksum mismatch for {name}.pdf")
        path.write_bytes(blob)
        log(f"[{i}/{len(doc_names)}] {name}.pdf  {len(blob) / 1e6:.1f} MB")


if __name__ == "__main__":
    download_metadata()
    names = corpus_doc_names()
    print(f"{len(load_questions())} questions over {len(names)} filings")
    download_pdfs(names)
    print("all PDFs present and verified")
