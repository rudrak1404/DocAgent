"""Output guardrails applied to every submitted answer (deterministic, no LLM).

1. Citations required: an answer must cite at least one evidence_id.
2. Citations must be real: every cited evidence_id must have been returned by a
   tool in *this* run (no invented or remembered sources).
3. Numeric grounding: every figure in the answer and in the stated calculation
   must appear in the cited evidence or come from a calculator result (allowing
   for rounding, units such as "in millions", and sign conventions).
A failed check is sent back to the agent as an error so it can fix the answer;
after the retry budget the system refuses rather than return an unverified answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from docagent.tools.numbers import ungrounded_figures


@dataclass
class Verdict:
    ok: bool
    refusal: bool = False
    issues: list[str] = field(default_factory=list)


def verify(submission: dict, evidence: dict[str, str], calc_values: list[float], question: str) -> Verdict:
    if submission.get("insufficient_evidence"):
        return Verdict(ok=True, refusal=True)
    answer = (submission.get("answer") or "").strip()
    cites = list(dict.fromkeys(submission.get("citations") or []))
    issues = []
    if not answer:
        issues.append("The answer is empty.")
    if not cites:
        issues.append("No citations. Cite the evidence_id of every passage or page you used.")
    unknown = [c for c in cites if c not in evidence]
    if unknown:
        issues.append(f"These citations were not returned by any tool in this session: {unknown}. "
                      "Cite only evidence_ids you received.")
    texts = [evidence[c] for c in cites if c in evidence]
    bad = ungrounded_figures(answer + "\n" + (submission.get("calculation") or ""), texts, calc_values, question)
    if bad:
        issues.append(f"These figures are not in your cited evidence or any calculate/financial_ratio "
                      f"result: {bad}. Cite the passage containing each figure (read_page if needed), "
                      "or compute it with calculate.")
    return Verdict(ok=not issues, issues=issues)
