"""Choose the retrieval configuration without overfitting the evaluation set.

The 150 FinanceBench questions are split once (seeded, stratified by question
type) into a dev half and a test half. Candidate configurations (embedder x
mode x re-ranker x BM25 weight) are compared on the dev half only; the choice
rule is fixed in advance: highest recall@5 averaged over the "none" and
"company" filter settings (MRR@10 breaks ties). The chosen configuration and the
baselines are then reported on the untouched test half and on all questions.

    python -m docagent.retrieval.select
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from docagent.config import INDEX_DIR, RESULTS_DIR
from docagent.data import load_questions
from docagent.retrieval.evaluate import KS
from docagent.retrieval.index import HybridIndex
from docagent.retrieval.models import Embedder, Reranker

EMBEDDERS = {"bge-base": "BAAI/bge-base-en-v1.5", "bge-large": "BAAI/bge-large-en-v1.5"}
RERANKERS = {"rr-base": "BAAI/bge-reranker-base", "rr-v2-m3": "BAAI/bge-reranker-v2-m3"}
CONFIGS = [  # (embedder, mode, reranker, bm25_weight)
    ("bge-base", "bm25", None, 1.0),
    ("bge-base", "dense", None, 1.0),
    ("bge-base", "hybrid", None, 1.0),
    ("bge-base", "hybrid", None, 0.5),
    ("bge-base", "hybrid_rerank", "rr-base", 1.0),
    ("bge-base", "dense_rerank", "rr-base", 1.0),
    ("bge-base", "dense_rerank", "rr-v2-m3", 1.0),
    ("bge-base", "hybrid_rerank", "rr-v2-m3", 0.5),
    ("bge-large", "dense", None, 1.0),
    ("bge-large", "dense_rerank", "rr-v2-m3", 1.0),
    ("bge-large", "hybrid_rerank", "rr-v2-m3", 0.5),
]
FILTERS = ("none", "company", "filing")


def split(questions, seed: int = 0) -> tuple[set, set]:
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q.id)
    dev, test = set(), set()
    rng = random.Random(seed)
    for ids in by_type.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        dev |= set(ids[: len(ids) // 2])
        test |= set(ids[len(ids) // 2:])
    return dev, test


def index_for(embedder_key: str, base: HybridIndex, embedders: dict) -> HybridIndex:
    if embedder_key == "bge-base":
        return base
    path = INDEX_DIR / embedder_key
    emb = embedders[embedder_key]
    if (path / "faiss.index").exists():
        fx = faiss.read_index(str(path / "faiss.index"))
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"embedding {len(base.chunks)} chunks with {EMBEDDERS[embedder_key]} ...")
        vecs = emb.encode([c.index_text() for c in base.chunks], batch_size=48)
        fx = faiss.IndexFlatIP(vecs.shape[1])
        fx.add(vecs)
        faiss.write_index(fx, str(path / "faiss.index"))
    return HybridIndex(base.chunks, fx, base.bm25, emb, None)


def run_config(index: HybridIndex, questions, mode: str, bm25_weight: float, flt: str) -> list:
    ranks = []
    for q in questions:
        gold = {(e.doc_name, e.page) for e in q.evidence}
        kw = {"company": q.company} if flt == "company" else ({"doc_name": q.doc_name} if flt == "filing" else {})
        hits = index.search(q.question, top_k=max(KS), mode=mode, bm25_weight=bm25_weight, **kw)
        ranks.append(next((h.rank for h in hits if (h.chunk.doc_name, h.chunk.page) in gold), None))
    return ranks


def summarize(ranks: list) -> dict:
    r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)
    return {**{f"recall@{k}": float(np.mean(r <= k)) for k in KS},
            "mrr@10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0)))}


def main():
    qs = load_questions()
    dev_ids, test_ids = split(qs)
    base = HybridIndex.load(INDEX_DIR, None, None)
    embedders = {k: Embedder(v) for k, v in EMBEDDERS.items()}
    base.embedder = embedders["bge-base"]
    rerankers: dict[str, Reranker] = {}
    per_q: dict[str, list] = {}

    for emb_key, mode, rr_key, w in CONFIGS:
        idx = index_for(emb_key, base, embedders)
        if rr_key:
            if rr_key not in rerankers:
                rerankers[rr_key] = Reranker(RERANKERS[rr_key])
            idx.reranker = rerankers[rr_key]
        name = f"{emb_key}|{mode}|{rr_key or '-'}|w{w}"
        for flt in FILTERS:
            per_q[f"{name}|{flt}"] = run_config(idx, qs, mode, w, flt)
        print(name, {flt: round(summarize([r for q, r in zip(qs, per_q[f'{name}|{flt}']) if q.id in dev_ids])["recall@5"], 3)
                     for flt in FILTERS}, flush=True)

    def score(name, ids):
        return {flt: summarize([r for q, r in zip(qs, per_q[f"{name}|{flt}"]) if q.id in ids]) for flt in FILTERS}

    names = [f"{e}|{m}|{r or '-'}|w{w}" for e, m, r, w in CONFIGS]
    dev = {n: score(n, dev_ids) for n in names}
    key = lambda n: (np.mean([dev[n][f]["recall@5"] for f in ("none", "company")]),
                     np.mean([dev[n][f]["mrr@10"] for f in ("none", "company")]))
    chosen = max(names, key=key)
    test = {n: score(n, test_ids) for n in names}
    full = {n: score(n, {q.id for q in qs}) for n in names}
    out = {"split": {"dev": sorted(dev_ids), "test": sorted(test_ids)}, "selection_rule":
           "max mean recall@5 over filters none+company on dev; MRR@10 tie-break",
           "chosen": chosen, "dev": dev, "test": test, "all": full, "per_question_ranks": per_q}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    lines = [f"Chosen on dev (rule: {out['selection_rule']}): **{chosen}**\n",
             "| configuration | filter | dev R@5 | test R@1 | test R@5 | test R@10 | test MRR@10 | all R@5 | all R@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        for f in FILTERS:
            mark = " **(chosen)**" if n == chosen else ""
            lines.append(f"| {n}{mark} | {f} | {dev[n][f]['recall@5']:.1%} | {test[n][f]['recall@1']:.1%} | "
                         f"{test[n][f]['recall@5']:.1%} | {test[n][f]['recall@10']:.1%} | {test[n][f]['mrr@10']:.3f} | "
                         f"{full[n][f]['recall@5']:.1%} | {full[n][f]['recall@10']:.1%} |")
    (RESULTS_DIR / "retrieval_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
