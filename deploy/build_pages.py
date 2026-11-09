"""Build the static demo site (GitHub Pages) from the evaluation results.

    python deploy/build_pages.py      # writes docs/index.html + docs/data.json

The page shows the headline metrics, the retrieval ablation, and a browser over
every FinanceBench question with the gold answer, DocAgent's answer (verdict,
citations, tool trace) and the baseline's answer, all from results/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docagent.data import load_questions  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "docs"


def _rows(system: str) -> dict:
    runs = {}
    p = RES / "runs" / f"{system}.jsonl"
    if p.exists():
        runs = {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    j = RES / "runs" / f"{system}.judged.jsonl"
    if j.exists():
        for l in j.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["id"] in runs:
                    runs[r["id"]]["judge"] = r
    return runs


def _cites(row: dict, n: int = 6) -> list:
    return [{"doc": c["doc_name"], "page": c["page"], "text": c["text"][:900]} for c in (row.get("citations") or [])[:n]]


def _retrieval() -> list[dict]:
    """Retrieval ablation on the held-out test half, no metadata filter."""
    p = RES / "retrieval_selection.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return [{"config": " / ".join(n.split("|")), "chosen": n == d["chosen"], "r1": t["none"]["recall@1"],
             "r5": t["none"]["recall@5"], "r10": t["none"]["recall@10"], "mrr": t["none"]["mrr@10"]}
            for n, t in d["test"].items()]


def main():
    qs = load_questions()
    agent, base = _rows("agent"), _rows("baseline")
    items = []
    for q in qs:
        a, b = agent.get(q.id, {}), base.get(q.id, {})
        items.append({
            "id": q.id, "company": q.company, "doc": q.doc_name, "type": q.question_type,
            "question": q.question, "gold": q.answer,
            "gold_pages": sorted({e.page + 1 for e in q.evidence}),
            "agent": {"answer": a.get("answer"), "status": a.get("status"),
                      "grade": a.get("judge", {}).get("grade", {}).get("label"),
                      "faithfulness": (a.get("judge", {}).get("faithfulness") or {}).get("score"),
                      "calculation": a.get("calculation"), "citations": _cites(a),
                      "trace": [{"tool": t["tool"], "input": t.get("input")} for t in a.get("trace", [])],
                      "seconds": a.get("seconds")} if a else None,
            "baseline": {"answer": b.get("answer"), "grade": b.get("judge", {}).get("grade", {}).get("label"),
                         "retrieved": [{"doc": c["doc_name"], "page": c["page"]} for c in b.get("citations", [])],
                         "seconds": b.get("seconds")} if b else None,
        })
    summary = json.loads((RES / "eval_summary.json").read_text(encoding="utf-8")) if (RES / "eval_summary.json").exists() else {}
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({"items": items, "summary": summary, "retrieval": _retrieval()},
                                              ensure_ascii=False), encoding="utf-8")
    (OUT / "index.html").write_text(PAGE, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} and data.json ({len(items)} questions)")


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocAgent: agentic RAG over SEC filings (FinanceBench results)</title>
<style>
  :root { --ink:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --accent:#1f5fbf; --ok:#1e7d4f; --bad:#b42318; --warn:#a15c00; --bg:#f7f8fa; }
  * { box-sizing:border-box; } body { margin:0; font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg); }
  header { background:#fff; border-bottom:1px solid var(--line); padding:22px 24px; }
  header h1 { margin:0; font-size:22px; } header p { margin:6px 0 0; color:var(--muted); max-width:900px; }
  main { max-width:1080px; margin:22px auto; padding:0 16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }
  .kpi { border:1px solid var(--line); border-radius:10px; padding:12px 14px; background:#fff; }
  .kpi .v { font-size:24px; font-weight:700; } .kpi .l { color:var(--muted); font-size:13px; }
  table { border-collapse:collapse; width:100%; font-size:13px; } th,td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
  th { background:#fafbfc; } .muted { color:var(--muted); } a { color:var(--accent); }
  .filters { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; } select,input { padding:7px 10px; border:1px solid var(--line); border-radius:8px; font:inherit; }
  .q { border:1px solid var(--line); border-radius:10px; margin:8px 0; background:#fff; }
  .q summary { cursor:pointer; padding:10px 12px; list-style:none; } .q summary::-webkit-details-marker { display:none; }
  .badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
  .correct { background:#e6f4ec; color:var(--ok); } .incorrect { background:#fde8e8; color:var(--bad); } .refusal { background:#fff3e0; color:var(--warn); } .none { background:#eef0f3; color:var(--muted); }
  .body { padding:0 14px 14px; display:grid; grid-template-columns:1fr 1fr; gap:14px; } @media (max-width:800px) { .body { grid-template-columns:1fr; } }
  .ans { white-space:pre-wrap; } .cite { border-left:3px solid var(--accent); padding:4px 10px; margin:6px 0; background:#f9fbff; }
  .cite pre { white-space:pre-wrap; margin:4px 0 0; font:12px/1.45 ui-monospace,Consolas,monospace; max-height:140px; overflow:auto; }
  .trace { font:12px/1.5 ui-monospace,Consolas,monospace; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>DocAgent: an agentic RAG analyst for SEC filings</h1>
  <p>A LangGraph agent that plans, retrieves through its own MCP server (dense search with a
  cross-encoder re-ranker, page reader, calculator, ratio library) and must pass citation and number
  guardrails, evaluated on all 150 open-source
  <a href="https://github.com/patronus-ai/financebench">FinanceBench</a> questions against a plain-RAG
  LangChain baseline. Both systems use the same free local model (Bonsai-2 9B via Ollama) on a laptop GPU.
  Code, tests and method: <a href="https://github.com/rudrak1404/DocAgent">github.com/rudrak1404/DocAgent</a>.</p>
</header>
<main>
  <div class="card"><div class="kpis" id="kpis"></div><p class="muted" id="kpinote"></p></div>
  <div class="card"><h3 style="margin-top:0">Results by question type</h3><div id="bytype"></div></div>
  <div class="card"><h3 style="margin-top:0">Retrieval ablation (no LLM)</h3>
    <p class="muted">Share of questions whose gold evidence page is retrieved. Configuration chosen on a dev half, reported on the held-out half.</p>
    <div id="retrieval"></div></div>
  <div class="card"><h3 style="margin-top:0">Every question</h3>
    <div class="filters">
      <select id="ftype"><option value="">all types</option><option>metrics-generated</option><option>domain-relevant</option><option>novel-generated</option></select>
      <select id="fgrade"><option value="">all DocAgent verdicts</option><option>correct</option><option>incorrect</option><option>refusal</option></select>
      <input id="ftext" placeholder="filter by company or text">
    </div>
    <div id="list" class="muted">Loading...</div></div>
  <p class="muted" id="judgenote">Filings and questions: FinanceBench (Patronus AI). Grades are from an LLM judge (Qwen3-8B) with explicit rules for units and rounding.</p>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = x => x == null ? "-" : (100 * x).toFixed(1) + "%";
const badge = g => `<span class="badge ${esc(g || "none")}">${esc(g || "not graded")}</span>`;
let D;
function render() {
  const t = $("ftype").value, g = $("fgrade").value, s = $("ftext").value.toLowerCase();
  const items = D.items.filter(i => (!t || i.type === t) && (!g || (i.agent && i.agent.grade === g)) &&
    (!s || (i.question + " " + i.company).toLowerCase().includes(s)));
  $("list").innerHTML = items.map(i => {
    const a = i.agent || {}, b = i.baseline || {};
    const cites = (a.citations || []).map(c => `<div class="cite"><b>${esc(c.doc)} p.${esc(c.page)}</b><pre>${esc(c.text)}</pre></div>`).join("");
    const trace = (a.trace || []).map(x => `${esc(x.tool)}(${esc(JSON.stringify(x.input))})`).join("<br>");
    return `<details class="q"><summary>${badge(a.grade)}<b>${esc(i.company)}</b> &middot; <span class="muted">${esc(i.type)}</span><br>${esc(i.question)}</summary>
      <div class="body"><div><div class="muted">Gold answer (evidence page ${esc(i.gold_pages.join(", "))} of ${esc(i.doc)})</div><div class="ans">${esc(i.gold)}</div>
        <h4>DocAgent ${badge(a.grade)}</h4><div class="ans">${esc(a.answer)}</div>
        ${a.calculation ? `<div class="muted">Calculation: ${esc(a.calculation)}</div>` : ""}
        ${trace ? `<details><summary class="muted">Tool trace</summary><div class="trace">${trace}</div></details>` : ""}</div>
      <div><h4 style="margin-top:0">Evidence cited by DocAgent</h4>${cites || '<span class="muted">none</span>'}
        <h4>Plain-RAG baseline ${badge(b.grade)}</h4><div class="ans">${esc(b.answer)}</div>
        <div class="muted">Retrieved: ${(b.retrieved || []).map(r => esc(r.doc) + " p." + esc(r.page)).join("; ")}</div></div></div></details>`;
  }).join("") || "No questions match.";
}
fetch("data.json").then(r => r.json()).then(d => {
  D = d; const S = d.summary || {};
  const A = S["agent|all"], B = S["baseline|all"];
  const k = [["DocAgent accuracy", A && pct(A.accuracy)], ["Baseline accuracy", B && pct(B.accuracy)],
             ["DocAgent accuracy when it answers", A && pct(A.accuracy_when_answered)],
             ["DocAgent faithfulness", A && pct(A.faithfulness)], ["DocAgent evidence on gold page", A && pct(A.citation_hits_gold_page)],
             ["Questions", S.questions]];
  $("kpis").innerHTML = k.map(([l, v]) => `<div class="kpi"><div class="v">${esc(v ?? "-")}</div><div class="l">${esc(l)}</div></div>`).join("");
  if (S.paired) $("kpinote").textContent = `Paired comparison: DocAgent right and baseline wrong on ${S.paired.agent_only_correct} questions, the reverse on ${S.paired.baseline_only_correct}; exact McNemar p = ${Number(S.paired.mcnemar_p).toPrecision(2)}; accuracy difference ${(100 * S.paired.accuracy_diff).toFixed(1)} points (95% bootstrap CI ${(100 * S.paired.ci95[0]).toFixed(1)} to ${(100 * S.paired.ci95[1]).toFixed(1)}).`;
  const types = ["all", "test split", "excluding pilot", "metrics-generated", "domain-relevant", "novel-generated"];
  $("bytype").innerHTML = `<table><tr><th>subset</th><th>n</th><th>DocAgent</th><th>baseline</th><th>DocAgent refusals</th><th>baseline refusals</th></tr>${types.map(t => {
    const a = S["agent|" + t], b = S["baseline|" + t]; if (!a) return "";
    return `<tr><td>${esc(t)}</td><td>${a.n}</td><td>${pct(a.accuracy)}</td><td>${b ? pct(b.accuracy) : "-"}</td><td>${a.refusal}</td><td>${b ? b.refusal : "-"}</td></tr>`; }).join("")}</table>`;
  $("retrieval").innerHTML = `<table><tr><th>configuration (test half, no filter)</th><th>recall@1</th><th>recall@5</th><th>recall@10</th><th>MRR@10</th></tr>${(d.retrieval || []).map(r =>
    `<tr${r.chosen ? ' style="font-weight:700"' : ""}><td>${esc(r.config)}${r.chosen ? " (chosen)" : ""}</td><td>${pct(r.r1)}</td><td>${pct(r.r5)}</td><td>${pct(r.r10)}</td><td>${Number(r.mrr).toFixed(3)}</td></tr>`).join("")}</table>`;
  const J = S.judge_agreement;
  if (J) $("judgenote").textContent += ` Checked against blind manual grading of ${J.n} answers: ${pct(J.agreement)} label agreement (Cohen's kappa ${J.cohen_kappa.toFixed(2)}), ${pct(J.agreement_correct_vs_not)} on correct-vs-not.`;
  ["ftype", "fgrade"].forEach(id => $(id).onchange = render); $("ftext").oninput = render; render();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
