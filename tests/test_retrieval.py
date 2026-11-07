import math
from collections import Counter

import numpy as np

from docagent.ingest.chunk import is_column_header, period_of, token_spans
from docagent.retrieval.bm25 import BM25, tokenize
from docagent.retrieval.index import HybridIndex, norm_company


# -------------------------------------------------------------------- BM25
def test_bm25_matches_reference_formula():
    docs = ["capital expenditure purchases of property plant and equipment",
            "revenue grew strongly in 2018", "net income and revenue 2018 2017",
            "property plant equipment net"]
    bm = BM25(k1=1.5, b=0.75).fit(docs)
    q = "property plant equipment 2018"
    toks = [tokenize(d) for d in docs]
    avgdl = np.mean([len(t) for t in toks])
    n = len(docs)
    ref = []
    for t in toks:
        tf, s = Counter(t), 0.0
        for term in tokenize(q):
            df = sum(term in tt for tt in toks)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            s += idf * tf[term] * 2.5 / (tf[term] + 1.5 * (1 - 0.75 + 0.75 * len(t) / avgdl))
        ref.append(s)
    assert np.allclose(bm.scores(q), ref, atol=1e-5)


def test_tokenizer_keeps_numbers():
    assert tokenize("Capex of $1,577.0 in FY2018") == ["capex", "1577.0", "fy", "2018"]


# ------------------------------------------------------------------ chunking
class WhitespaceTokenizer:
    """Mimics a HF fast tokenizer's offset mapping with whitespace tokens."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        offs = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


def test_token_spans_are_exact_overlapping_substrings():
    text = "\n".join(f"line {i} | {i * 10} | {i * 20}" for i in range(60))
    spans = token_spans(text, WhitespaceTokenizer(), size=40, overlap=10)
    assert len(spans) > 1
    for a, b in spans:
        assert a == 0 or text[a - 1] == "\n"           # windows start at a line boundary
    assert spans[-1][1] == len(text)                    # the whole page is covered
    assert all(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))   # overlap


def test_period_and_column_header():
    assert period_of("JOHNSON_JOHNSON_2023Q2_EARNINGS") == "2023Q2"
    assert period_of("PEPSICO_2023_8K_dated-2023-05-05") == "2023"
    assert is_column_header("(Millions) | 2018 | 2017 | 2016")
    assert not is_column_header("Revenue | 32,136 | 32,765")


# --------------------------------------------------------------------- index
def test_company_normalisation():
    assert norm_company("Johnson & Johnson Inc.") == norm_company("JOHNSON JOHNSON")
    assert norm_company("The Coca-Cola Company") == "coca cola"


def test_filters_restrict_results(tiny_index: HybridIndex):
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank", "dense_rerank"):
        hits = tiny_index.search("current assets liabilities", top_k=5, company="globex", mode=mode)
        assert hits and all(h.chunk.company == "Globex" for h in hits)
    hits = tiny_index.search("revenue net income", top_k=5, company="ACME", year=2019, mode="dense")
    assert {h.chunk.doc_name for h in hits} == {"ACME_2019_10K"}
    assert tiny_index.search("anything", company="Nobody Inc") == []


def test_cash_flow_question_finds_statement(tiny_index: HybridIndex):
    hits = tiny_index.search("purchases of property plant and equipment 2018", top_k=1, company="ACME")
    assert hits[0].chunk.chunk_id == "ACME_2018_10K:p1:c0"


def test_rrf_fusion():
    fused = dict(HybridIndex.rrf([(1, 0.9), (2, 0.5)], [(2, 9.0), (3, 1.0)]))
    assert fused[2] > fused[1] > fused[3]                # in both lists beats either single list
    assert math.isclose(fused[2], 1 / 62 + 1 / 61)
