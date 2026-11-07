"""Retrieval ablation on FinanceBench gold evidence pages (no LLM involved).

For every question, a hit is any retrieved chunk whose (filing, page) equals a
gold evidence page. Reported per retrieval mode and filter setting:
  * recall@k  = share of questions with at least one gold page in the top k
  * MRR@10    = mean reciprocal rank of the first gold page (0 if not in top 10)

Filter settings:
  * none     - one shared index over all 84 filings (the hard setting)
  * company  - restricted to the question's company (what the agent can infer
               from the question text)
  * filing   - restricted to the gold filing (an upper bound / oracle)

    python -m docagent.retrieval.evaluate
"""
from __future__ import annotations

import json
import time

import numpy as np

from docagent.config import INDEX_DIR, RESULTS_DIR
from docagent.data import load_questions
from docagent.retrieval.index import MODES, HybridIndex
from docagent.retrieval.models import Embedder, Reranker

KS = (1, 3, 5, 10)


def evaluate(index: HybridIndex, questions, modes=MODES, filters=("none", "company", "filing")) -> dict:
    out = {}
    for flt in filters:
        for mode in modes:
            ranks, t0 = [], time.time()
            for q in questions:
                gold = {(e.doc_name, e.page) for e in q.evidence}
                kw = {}
                if flt == "company":
                    kw["company"] = q.company
                elif flt == "filing":
                    kw["doc_name"] = q.doc_name
                hits = index.search(q.question, top_k=max(KS), mode=mode, **kw)
                rank = next((h.rank for h in hits if (h.chunk.doc_name, h.chunk.page) in gold), None)
                ranks.append(rank)
            r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)
            out[f"{flt}|{mode}"] = {
                "filter": flt, "mode": mode,
                **{f"recall@{k}": float(np.mean(r <= k)) for k in KS},
                "mrr@10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))),
                "ms_per_query": round((time.time() - t0) / len(questions) * 1000, 1),
                "ranks": [None if x is None else int(x) for x in ranks],
            }
            print(f"{flt:8s} {mode:14s} " + "  ".join(f"R@{k}={out[f'{flt}|{mode}'][f'recall@{k}']:.3f}" for k in KS)
                  + f"  MRR@10={out[f'{flt}|{mode}']['mrr@10']:.3f}  {out[f'{flt}|{mode}']['ms_per_query']} ms/q")
    return out


def markdown(res: dict) -> str:
    lines = ["| filter | mode | R@1 | R@3 | R@5 | R@10 | MRR@10 | ms/query |", "|---|---|---|---|---|---|---|---|"]
    for v in res.values():
        lines.append(f"| {v['filter']} | {v['mode']} | " + " | ".join(f"{v[f'recall@{k}']:.1%}" for k in KS)
                     + f" | {v['mrr@10']:.3f} | {v['ms_per_query']} |")
    return "\n".join(lines)


def main():
    index = HybridIndex.load(INDEX_DIR, Embedder(), Reranker())
    res = evaluate(index, load_questions())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (RESULTS_DIR / "retrieval_eval.md").write_text(markdown(res) + "\n", encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main()
