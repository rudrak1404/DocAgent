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
JUDGE_MAX_TOKENS = 5000        # with the 8k judge context: prompts are <= ~3k tokens


# ------------------------------------------------------------------ systems
def build_system(name: str):
    from docagent.agent.llm import agent_kwargs, make_llm
    from docagent.retrieval.service import load_index
    index = load_index()
    if name == "agent":
        from docagent.agent.graph import DocAgent
        from docagent.data import load_docs
        from docagent.mcp_server import create_server
        return DocAgent(create_server(index, load_docs()), make_llm(), **agent_kwargs())
    from docagent.baseline import PlainRAG
    return PlainRAG(index, make_llm())


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


async def run(system: str, which: str, concurrency: int, tag: str = "") -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{system}{tag}.jsonl"
    done = set()
    if path.exists():                         # rows that ended in an error are run again (last row wins)
        done = {r["id"] for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
                if r.get("status") != "error"}
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
- refusal: the response does not commit to an answer to the question (for example it says the \
information was not found, is not in the provided context, or cannot be determined), even if it \
mentions related facts. A non-answer is a refusal, never "incorrect", whatever the gold answer is.
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
    from docagent.agent.llm import make_llm
    src, dst = RUNS / f"{system}.jsonl", RUNS / f"{system}.judged.jsonl"
    rows = list({r["id"]: r for r in (json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()
                                      if l.strip())}.values())          # last row per question wins
    done = set()
    if dst.exists():
        done = {json.loads(l)["id"] for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()}
    qs = {q.id: q for q in load_questions()}
    llm = make_llm("judge")
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

    class JudgeFailed(Exception):
        pass

    async def grade(question, gold, justification, answer) -> tuple[dict, dict]:
        msg = (f"Question:\n{question}\n\nGold answer:\n{gold}\n\nGold justification:\n{justification}\n\n"
               f"Response to grade:\n{answer}")
        g, rec, _ = await llm.parse(system=GRADE_SYSTEM, messages=[{"role": "user", "content": msg}],
                                    output_format=Grade, max_tokens=JUDGE_MAX_TOKENS, tag="judge")
        if g is None:
            raise JudgeFailed(f"no valid verdict after {rec.get('attempts', 1)} attempts")
        return g.model_dump(), rec

    async def one(row):
        q = qs[row["id"]]
        out = {"id": row["id"], "system": system}
        try:
            async with sem:
                attempts = []
                if row.get("status") == "refused":     # the system itself declined: a refusal by definition
                    g = {"reasoning": "system refused (status=refused)", "label": "refusal"}
                    out["grade"], out["judge_cost"] = g, 0.0
                else:
                    g, rec = await grade(q.question, q.answer, q.justification, row["answer"])
                    out["grade"], out["judge_cost"] = g, rec["cost_usd"]
                    attempts.append(rec.get("attempts", 1))
                if row.get("unverified_answer"):       # what would the guardrail-blocked answer have scored?
                    g2, rec2 = await grade(q.question, q.answer, q.justification, row["unverified_answer"])
                    out["grade_unverified"], out["judge_cost"] = g2, out["judge_cost"] + rec2["cost_usd"]
                    attempts.append(rec2.get("attempts", 1))
                if g["label"] != "refusal" and row.get("status") == "answered":
                    msg = f"Evidence:\n{_evidence_block(row)}\n\nResponse:\n{row['answer']}"
                    f, rec3, _ = await llm.parse(system=FAITH_SYSTEM, messages=[{"role": "user", "content": msg}],
                                                 output_format=Faithfulness, max_tokens=JUDGE_MAX_TOKENS, tag="faith")
                    claims = f.claims if f else []
                    out["faithfulness"] = {"claims": [c.model_dump() for c in claims],
                                           "score": (sum(c.supported for c in claims) / len(claims)) if claims else None}
                    out["judge_cost"] += rec3["cost_usd"]
                    attempts.append(rec3.get("attempts", 1))
                out["judge_attempts"] = attempts
        except Exception as e:                     # never lose the whole run to one bad call; retried next time
            print(f"[judge {system}] {row['id']} FAILED ({e!r}); will be retried on the next run", flush=True)
            return
        async with lock:
            with open(dst, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[judge {system}] {row['id']} {out['grade']['label']:9s} "
                  f"faith={out.get('faithfulness', {}).get('score')} attempts={out['judge_attempts']}", flush=True)

    await asyncio.gather(*(one(r) for r in rows if r["id"] not in done and r.get("status") != "error"))
    n_done = len({json.loads(l)["id"] for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()})
    print(f"judged {n_done}/{len(rows)} {system} rows", flush=True)


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
    graded = {s: sum("judge" in rows[i] for i in common) for s, rows in systems.items()}
    out["graded"] = graded
    lines.insert(1, "Graded: " + ", ".join(f"{s} {n}/{len(common)}" for s, n in graded.items())
                 + ("" if all(n == len(common) for n in graded.values()) else
                    "  **(incomplete: ungraded answers count as not correct)**") + "\n")
    for s, rows in systems.items():
        calls = [c for i in common for c in rows[i].get("llm_calls", []) if "output_tokens" in c]
        health = {"llm_calls": len(calls),
                  "hit_token_limit": sum(c.get("stop_reason") == "max_tokens" for c in calls),
                  "resampled": sum(c.get("attempts", 1) > 1 for c in calls),
                  "results_shortened": sum(c.get("shortened_results", 0) for c in calls)}
        out[f"{s}|generation"] = health
    for s, rows in systems.items():
        pilot = set(question_ids("pilot"))
        for subset, ids in (("all", common), ("test split", [i for i in common if i in set(split["test"])]),
                            ("excluding pilot", [i for i in common if i not in pilot]),
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
    lines += ["\n## Generation health\n"] + [
        f"{s}: {h['llm_calls']} LLM calls, {h['hit_token_limit']} stopped at the token limit, "
        f"{h['resampled']} re-sampled after a runaway generation, {h['results_shortened']} old tool results "
        f"shortened to fit the context window." for s, h in ((s, out[f"{s}|generation"]) for s in systems)]
    agr = judge_agreement()
    if agr:
        out["judge_agreement"] = agr
        lines += ["\n## Judge validation\n",
                  f"Blind manual grading of {agr['n']} judged answers (stratified by system and question type): "
                  f"the LLM judge agrees on {agr['agreement']:.0%} of labels (Cohen's kappa {agr['cohen_kappa']:.2f}) "
                  f"and on {agr['agreement_correct_vs_not']:.0%} of correct-vs-not decisions (kappa "
                  f"{agr['cohen_kappa_correct_vs_not']:.2f}). Judge said correct where the manual grade did not: "
                  f"{agr['judge_correct_manual_not']}; the reverse: {agr['manual_correct_judge_not']}. "
                  f"Accuracy on this sample: manual {agr['accuracy_manual']:.0%}, judge {agr['accuracy_judge']:.0%}."]
    (RESULTS_DIR / "eval_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS_DIR / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def audit_sample(n: int = 40, seed: int = 0) -> None:
    """Blind sample for manual grading: n/2 answers per system, stratified by question
    type, drawn from the run files without looking at the judge, and written without
    the judge's label (results/judge_audit_sample.json). Answers the system itself
    refused are excluded: their label is set by rule, not by the judge. Put manual
    labels in results/judge_audit_manual.json as {"<system>|<id>": "correct|incorrect|refusal"}."""
    qs = {q.id: q for q in load_questions()}
    rng = random.Random(seed)
    out = []
    for s in ("agent", "baseline"):
        rows = {i: r for i, r in _load(s).items() if r.get("status") not in ("refused", "error")}
        by_type: dict[str, list[str]] = {}
        for i in sorted(rows):
            by_type.setdefault(qs[i].question_type, []).append(i)
        types, per = sorted(by_type), n // 2
        for k, t in enumerate(types):
            quota = per // len(types) + (1 if k < per % len(types) else 0)
            for i in rng.sample(by_type[t], min(quota, len(by_type[t]))):
                out.append({"key": f"{s}|{i}", "question_type": t, "question": qs[i].question,
                            "gold": qs[i].answer, "justification": qs[i].justification, "answer": rows[i]["answer"]})
    rng.shuffle(out)
    (RESULTS_DIR / "judge_audit_sample.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} items (blind: no judge labels)")


def _kappa(a: list, b: list, labels) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def judge_agreement() -> dict | None:
    """Manual labels (blind sample) vs the LLM judge's labels from the judged runs."""
    manual_p = RESULTS_DIR / "judge_audit_manual.json"
    if not manual_p.exists():
        return None
    manual = json.loads(manual_p.read_text(encoding="utf-8"))
    judged = {f"{s}|{i}": r["judge"]["grade"]["label"] for s in ("agent", "baseline") if (RUNS / f"{s}.jsonl").exists()
              for i, r in _load(s).items() if "judge" in r}
    keys = sorted(k for k in manual if k in judged)
    if not keys:
        return None
    m, j = [manual[k] for k in keys], [judged[k] for k in keys]
    mb, jb = [x == "correct" for x in m], [x == "correct" for x in j]
    return {"n": len(keys),
            "agreement": sum(x == y for x, y in zip(m, j)) / len(keys),
            "cohen_kappa": _kappa(m, j, ("correct", "incorrect", "refusal")),
            "agreement_correct_vs_not": sum(x == y for x, y in zip(mb, jb)) / len(keys),
            "cohen_kappa_correct_vs_not": _kappa(mb, jb, (True, False)),
            "judge_correct_manual_not": sum(y and not x for x, y in zip(mb, jb)),
            "manual_correct_judge_not": sum(x and not y for x, y in zip(mb, jb)),
            "accuracy_manual": sum(mb) / len(keys), "accuracy_judge": sum(jb) / len(keys),
            "disagreements": [k for k, x, y in zip(keys, m, j) if x != y]}


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
    ap.add_argument("cmd", choices=["run", "judge", "report", "examples", "audit-sample"])
    ap.add_argument("--system", choices=["agent", "baseline"])
    ap.add_argument("--ids", default="all", help="all | pilot | dev | test | comma-separated ids")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the run file, e.g. .pilot-qwen3")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a.system, a.ids, a.concurrency, a.tag))
    elif a.cmd == "judge":
        asyncio.run(judge(a.system, a.concurrency))
    elif a.cmd == "examples":
        export_examples()
    elif a.cmd == "audit-sample":
        audit_sample()
    else:
        report()


if __name__ == "__main__":
    main()
