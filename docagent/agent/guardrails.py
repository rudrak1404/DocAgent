"""Output guardrails applied to every submitted answer (deterministic, no LLM).

1. Citations required: an answer must cite at least one evidence_id.
2. Citations must be real: every cited evidence_id must have been returned by a
   tool in *this* run (no invented or remembered sources).
3. Numeric grounding: every figure in the answer and in the stated calculation
   must appear in the cited evidence or come from a calculator result (allowing
   for rounding, units such as "in millions", and sign conventions).
A failed check is sent back to the agent as an error so it can fix the answer;
after the retry budget the system refuses rather than return an unverified answer.

The feedback is specific but never supplies an answer: it lists the evidence_ids
the agent actually received and, for an unsupported figure, which of those
received passages contain it, so a model that took a number from a passage but
cited the wrong id can repair the citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from docagent.tools.numbers import extract_numbers, grounded, is_checkable, ungrounded_figures

FORMULA_CONSTANTS = [100.0, 1000.0, 12.0, 365.0]


@dataclass
class Verdict:
    ok: bool
    refusal: bool = False
    issues: list[str] = field(default_factory=list)


def _where(figure_text: str, evidence: dict[str, str], limit: int = 3) -> list[str]:
    """evidence_ids whose text contains a number matching the figure."""
    nums = [n for n in extract_numbers(figure_text) if is_checkable(n)]
    if not nums:
        return []
    n = nums[0]
    hits = []
    for eid, text in evidence.items():
        pool = [m["abs"] * m["scale"] for m in extract_numbers(text)] + [m["abs"] for m in extract_numbers(text)]
        if grounded(n, pool):
            hits.append(eid)
            if len(hits) >= limit:
                break
    return hits


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
        received = list(evidence)[-30:]
        issues.append(f"These citations were not returned by any tool in this session: {unknown}. "
                      f"Cite only evidence_ids you received. Received so far: {received}")
    texts = [evidence[c] for c in cites if c in evidence]
    # unit-conversion constants (x100 for a percentage, months, days) are not filing figures:
    # allowed in the stated calculation only; the answer's own figures are checked as before
    bad = list(dict.fromkeys(
        ungrounded_figures(answer, texts, calc_values, question)
        + ungrounded_figures(submission.get("calculation") or "", texts, list(calc_values) + FORMULA_CONSTANTS, question)))
    if bad:
        hints = []
        for fig in bad[:6]:
            where = _where(fig, evidence)
            if where:
                hints.append(f"{fig} appears in {where}")
        issues.append(f"These figures are not in your cited evidence or any calculate/financial_ratio "
                      f"result: {bad}. Cite the passage containing each figure (read_page if needed), "
                      "or compute it with calculate."
                      + (" Passages you received that contain them: " + "; ".join(hints) + "." if hints else ""))
    return Verdict(ok=not issues, issues=issues)
