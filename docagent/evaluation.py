"""FinanceBench evaluation: run both systems, grade with an LLM judge, report.

    python -m docagent.evaluation run --system agent --ids pilot     # 10-question pilot
    python -m docagent.evaluation run --system baseline              # all 150 (resumable)
    python -m docagent.evaluation judge --system agent
    python -m docagent.evaluation report

Grading (FinanceBench labels): correct / incorrect / refusal. The judge sees the
question, the human gold answer and its justification, and the system answer, and
follows explicit tolerance rules for numbers (rounding, units). Faithfulness is
claim-level: the answer is split into atomic claims and each is checked against
the evidence *that system actually used* (DocAgent: cited passages + calculator
results; baseline: its 5 retrieved passages).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from docagent.config import JUDGE_MODEL, RESULTS_DIR
from docagent.data import load_questions

RUNS = RESULTS_DIR / "runs"


# ------------------------------------------------------------------ systems
def build_system(name: str):
    from docagent.agent.llm import LLM
    from docagent.retrieval.service import load_index
    index = load_index()
    if name == "agent":
        from docagent.agent.graph import DocAgent
        from docagent.data import load_docs
        from docagent.mcp_server import create_server
        return DocAgent(create_server(index, load_docs()), LLM())
    from docagent.baseline import PlainRAG
    return PlainRAG(index, LLM())


def question_ids(which: str) -> list[str]:
    qs = load_questions()
    if which == "all":
        return [q.id for q in qs]
    if which == "pilot":                       # 10 questions, stratified by type, fixed seed
        rng = random.Random(7)
        out = []
        for t, n in (("metrics-generated", 4), ("domain-relevant", 3), ("novel-generated", 3)):
            ids = sorted(q.id for q in qs if q.question_type == t)
            out += rng.sample(ids, n)
        return out
    if which in ("dev", "test"):
        sel = json.loads((RESULTS_DIR / "retrieval_selection.json").read_text(encoding="utf-8"))["split"][which]
        return sel
    return which.split(",")


async def run(system: str, which: str, concurrency: int) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{system}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    qs = {q.id: q for q in load_questions()}
    todo = [i for i in question_ids(which) if i not in done]
    print(f"{system}: {len(todo)} questions to run ({len(done)} already done)", flush=True)
    if not todo:
        return
    sysobj = build_system(system)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    spent = 0.0

    async def one(qid: str):
        nonlocal spent
        async with sem:
            res = await sysobj.ask(qs[qid].question)
        row = {"id": qid, "system": system, "question": qs[qid].question, **res}
        async with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            spent += res.get("usage", {}).get("cost_usd", 0.0)
            print(f"[{system}] {qid} {res['status']:8s} {res['seconds']:6.1f}s ${res.get('usage', {}).get('cost_usd', 0):.3f} "
                  f"(total ${spent:.2f}) | {res['answer'][:90]!r}", flush=True)

    await asyncio.gather(*(one(i) for i in todo))


# ------------------------------------------------------------------- judge
class Grade(BaseModel):
    reasoning: str = Field(description="brief comparison of the response with the gold answer")
    label: Literal["correct", "incorrect", "refusal"]


class Claim(BaseModel):
    claim: str
    supported: bool


class Faithfulness(BaseModel):
    claims: list[Claim] = Field(description="every atomic factual claim in the response")


GRADE_SYSTEM = """You grade answers to questions about companies' SEC filings against a human \
expert's gold answer. Return one label:
- correct: the response gives the same answer as the gold answer. Numbers must match the gold \
value after rounding to the precision the gold answer uses, or be within 1% of it; equivalent \
units and formats are fine (e.g. $1.58 billion = $1,577 million; 0.24 = 24%). For yes/no and \
qualitative questions, the response must reach the same conclusion and include the gold answer's \
key facts without contradicting any of them. Extra correct detail is fine.
- incorrect: a wrong figure, wrong period or company, a contradiction of the gold answer, or a \
multi-part question answered wrongly in any part.
- refusal: the response does not commit to an answer (for example it says the information was \
not found or it cannot answer).
Judge only the final answer content; do not reward or penalise style."""

FAITH_SYSTEM = """You check whether an answer is faithful to the evidence a system retrieved. \
Split the response into atomic factual claims (each figure, date, comparison or factual \
statement). For each claim decide whether it is fully supported by the evidence below. A claim \
is supported if the evidence states it, or if it follows by correct arithmetic from figures in \
the evidence or from the listed calculator results. Judge support only, not correctness against \
any outside knowledge."""


def _evidence_block(row: dict) -> str:
    parts = [f"[{c['evidence_id']} | {c['doc_name']} p.{c['page']}]\n{c['text']}" for c in row.get("citations", [])]
    calcs = [f"{t['tool']}({json.dumps(t['input'])}) = {t.get('value')}" for t in row.get("trace", [])
             if t.get("tool") in ("calculate", "financial_ratio") and "value" in t]
    if calcs:
        parts.append("Calculator results:\n" + "\n".join(calcs))
    return "\n\n".join(parts) or "(no evidence)"


async def judge(system: str, concurrency: int) -> None:
    from docagent.agent.llm import LLM
    src, dst = RUNS / f"{system}.jsonl", RUNS / f"{system}.judged.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if dst.exists():
        done = {json.loads(l)["id"] for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()}
    qs = {q.id: q for q in load_questions()}
    llm = LLM(model=JUDGE_MODEL, effort="medium")
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

    async def grade(question, gold, justification, answer) -> tuple[dict, dict]:
        msg = (f"Question:\n{question}\n\nGold answer:\n{gold}\n\nGold justification:\n{justification}\n\n"
               f"Response to grade:\n{answer}")
        g, rec, _ = await llm.parse(system=GRADE_SYSTEM, messages=[{"role": "user", "content": msg}],
                                    output_format=Grade, max_tokens=8000, tag="judge")
        return g.model_dump(), rec

    async def one(row):
        q = qs[row["id"]]
        async with sem:
            out = {"id": row["id"], "system": system}
            g, rec = await grade(q.question, q.answer, q.justification, row["answer"])
            out["grade"], out["judge_cost"] = g, rec["cost_usd"]
            if row.get("unverified_answer"):           # what would the guardrail-blocked answer have scored?
                g2, rec2 = await grade(q.question, q.answer, q.justification, row["unverified_answer"])
                out["grade_unverified"], out["judge_cost"] = g2, out["judge_cost"] + rec2["cost_usd"]
            if g["label"] != "refusal" and row.get("status") == "answered":
                msg = f"Evidence:\n{_evidence_block(row)}\n\nResponse:\n{row['answer']}"
                f, rec3, _ = await llm.parse(system=FAITH_SYSTEM, messages=[{"role": "user", "content": msg}],
                                             output_format=Faithfulness, max_tokens=8000, tag="faith")
                claims = f.claims if f else []
                out["faithfulness"] = {"claims": [c.model_dump() for c in claims],
                                       "score": (sum(c.supported for c in claims) / len(claims)) if claims else None}
                out["judge_cost"] += rec3["cost_usd"]
        async with lock:
            with open(dst, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[judge {system}] {row['id']} {out['grade']['label']:9s} "
                  f"faith={out.get('faithfulness', {}).get('score')}", flush=True)

    await asyncio.gather(*(one(r) for r in rows if r["id"] not in done and r.get("status") != "error"))


# ------------------------------------------------------------------ report
def _load(system: str) -> dict:
    runs = {json.loads(l)["id"]: json.loads(l) for l in (RUNS / f"{system}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    judged = {}
    p = RUNS / f"{system}.judged.jsonl"
    if p.exists():
        judged = {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return {i: {**r, **({"judge": judged[i]} if i in judged else {})} for i, r in runs.items()}


def _stats(rows: list[dict], qs: dict) -> dict:
    n = len(rows)
    lab = [r["judge"]["grade"]["label"] if "judge" in r else ("error" if r["status"] == "error" else "ungraded") for r in rows]
    correct = sum(l == "correct" for l in lab)
    refusal = sum(l == "refusal" for l in lab)
    incorrect = sum(l == "incorrect" for l in lab)
    faith = [r["judge"]["faithfulness"]["score"] for r in rows
             if "judge" in r and r["judge"].get("faithfulness", {}).get("score") is not None]
    gold_hit = []
    for r in rows:
        gold = {(e.doc_name, e.page + 1) for e in qs[r["id"]].evidence}
        cited = {(c["doc_name"], c["page"]) for c in r.get("citations", [])}
        if cited:
            gold_hit.append(bool(gold & cited))
    return {
        "n": n, "accuracy": correct / n if n else 0, "correct": correct, "incorrect": incorrect,
        "refusal": refusal, "errors": sum(l == "error" for l in lab),
        "accuracy_when_answered": correct / (correct + incorrect) if correct + incorrect else 0,
        "hallucination_rate": incorrect / n if n else 0,
        "faithfulness": float(np.mean(faith)) if faith else None,
        "citation_hits_gold_page": float(np.mean(gold_hit)) if gold_hit else None,
        "cost_per_question": float(np.mean([r.get("usage", {}).get("cost_usd", 0) for r in rows])) if n else 0,
        "seconds_per_question": float(np.mean([r["seconds"] for r in rows])) if n else 0,
        "tool_calls_per_question": float(np.mean([r.get("tool_calls", 0) for r in rows])) if n else 0,
    }


def report() -> None:
    from scipy.stats import binomtest
    qs = {q.id: q for q in load_questions()}
    systems = {s: _load(s) for s in ("baseline", "agent") if (RUNS / f"{s}.jsonl").exists()}
    common = sorted(set.intersection(*(set(v) for v in systems.values()))) if systems else []
    split = json.loads((RESULTS_DIR / "retrieval_selection.json").read_text(encoding="utf-8"))["split"]
    out = {"questions": len(common)}
    lines = [f"# FinanceBench evaluation ({len(common)} questions)\n",
             "Evidence on gold page: DocAgent = its cited evidence; baseline = its 5 retrieved passages.\n"]
    hdr = ("| system | subset | n | accuracy | incorrect | refusal | acc. when answered | faithfulness | "
           "evidence on gold page | $/question | s/question | tool calls |")
    lines += [hdr, "|" + "---|" * 12]
    for s, rows in systems.items():
        for subset, ids in (("all", common), ("test split", [i for i in common if i in set(split["test"])]),
                            *((t, [i for i in common if qs[i].question_type == t])
                              for t in ("metrics-generated", "domain-relevant", "novel-generated"))):
            if not ids:
                continue
            st = _stats([rows[i] for i in ids], qs)
            out[f"{s}|{subset}"] = st
            f = lambda x: "-" if x is None else f"{x:.1%}"
            lines.append(f"| {s} | {subset} | {st['n']} | **{st['accuracy']:.1%}** | {st['incorrect']} | {st['refusal']} | "
                         f"{f(st['accuracy_when_answered'])} | {f(st['faithfulness'])} | {f(st['citation_hits_gold_page'])} | "
                         f"${st['cost_per_question']:.3f} | {st['seconds_per_question']:.0f} | {st['tool_calls_per_question']:.1f} |")
    if len(systems) == 2:
        a = [systems["agent"][i].get("judge", {}).get("grade", {}).get("label") == "correct" for i in common]
        b = [systems["baseline"][i].get("judge", {}).get("grade", {}).get("label") == "correct" for i in common]
        a, b = np.array(a), np.array(b)
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(a), (5000, len(a)))
        diffs = a[idx].mean(1) - b[idx].mean(1)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        out["paired"] = {"agent_only_correct": n10, "baseline_only_correct": n01, "mcnemar_p": p,
                         "accuracy_diff": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)]}
        lines += ["\n## Paired comparison (same questions)\n",
                  f"DocAgent correct where the baseline is not: {n10}; the reverse: {n01}. "
                  f"Exact McNemar test p = {p:.2g}. Accuracy difference {a.mean() - b.mean():+.1%} "
                  f"(95% bootstrap CI {lo:+.1%} to {hi:+.1%})."]
        blocked = [systems["agent"][i] for i in common if systems["agent"][i].get("unverified_answer")
                   and "judge" in systems["agent"][i] and "grade_unverified" in systems["agent"][i]["judge"]]
        if blocked:
            wrong = sum(r["judge"]["grade_unverified"]["label"] != "correct" for r in blocked)
            out["guardrail_blocked"] = {"n": len(blocked), "would_have_been_wrong_or_refusal": wrong}
            lines += ["\n## Guardrails\n",
                      f"Answers withheld by the citation/number guardrails: {len(blocked)}. Graded anyway, "
                      f"{wrong} of them would not have been correct."]
    (RESULTS_DIR / "eval_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS_DIR / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def export_examples(n_per_type: int = 3) -> None:
    """Write results/examples.json (shown in the web UI): graded DocAgent answers,
    a few per question type, including at least one refusal if there is one."""
    qs = {q.id: q for q in load_questions()}
    rows = [r for r in _load("agent").values() if "judge" in r]
    out = []
    for t in ("metrics-generated", "domain-relevant", "novel-generated"):
        pick = sorted((r for r in rows if qs[r["id"]].question_type == t),
                      key=lambda r: (r["judge"]["grade"]["label"] != "correct", r["id"]))[:n_per_type]
        out += pick
    refusals = [r for r in rows if r["status"] == "refused"]
    out += refusals[:1]
    keep = ("question", "answer", "status", "citations", "calculation", "trace", "tool_calls", "seconds", "usage")
    ex = [{**{k: r.get(k) for k in keep}, "gold_answer": qs[r["id"]].answer,
           "grade": r["judge"]["grade"]["label"], "id": r["id"]} for r in out]
    for e in ex:
        for c in e["citations"] or []:
            c["text"] = c["text"][:1500]
    (RESULTS_DIR / "examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ex)} examples")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge", "report", "examples"])
    ap.add_argument("--system", choices=["agent", "baseline"])
    ap.add_argument("--ids", default="all", help="all | pilot | dev | test | comma-separated ids")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a.system, a.ids, a.concurrency))
    elif a.cmd == "judge":
        asyncio.run(judge(a.system, a.concurrency))
    elif a.cmd == "examples":
        export_examples()
    else:
        report()


if __name__ == "__main__":
    main()
