"""Number extraction and matching for the numeric-grounding guardrail.

A figure in an answer counts as grounded if it equals (within the precision it
is written with) a figure in the cited evidence or a calculator result, allowing
for the usual presentation differences in filings:
  * scale: "$1.58 billion" vs "1,577" in a table "(in millions)";
  * sign: "(1,577)" is how filings print negatives / outflows;
  * percent vs fraction: "24.1%" vs 0.241.
Years and small counting numbers are not checked (they are not financial
figures, and they appear in the question itself).
"""
from __future__ import annotations

import math
import re

NUM_RE = re.compile(
    r"(?P<neg>\(|-|−)?\$?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<close>\))?"
    r"\s?(?P<pct>%|percent)?\s?(?P<scale>thousand|million|billion|trillion|mn|bn|[kmb](?![a-z]))?",
    re.I)
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
         "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}


def extract_numbers(text: str) -> list[dict]:
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group("num")
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        neg = bool(m.group("neg")) and (m.group("neg") != "(" or bool(m.group("close")))
        scale = SCALE.get((m.group("scale") or "").lower(), 1.0)
        out.append({"text": m.group(0).strip(), "value": -value if neg else value,
                    "abs": value, "decimals": decimals, "scale": scale,
                    "percent": bool(m.group("pct"))})
    return out


def is_checkable(n: dict) -> bool:
    """Skip years and small integers (counts, item numbers, 'FY2018', 'Q2')."""
    v = n["abs"]
    if n["decimals"] == 0 and n["scale"] == 1.0 and not n["percent"]:
        if 1900 <= v <= 2100 or v < 10:
            return False
    return True


def variants(n: dict) -> list[tuple[float, float]]:
    """(value, rounding tolerance) pairs an answer figure may correspond to in
    the evidence. The tolerance is half a unit of the last written digit, carried
    through the same scaling, so "$1.6 billion" matches 1,577 (in millions)."""
    half_unit = 0.5 * 10 ** (-n["decimals"])
    base, tol = n["abs"] * n["scale"], half_unit * n["scale"]
    out = [(base / s, tol / s) for s in (1.0, 1e3, 1e6, 1e9)]   # tables in units/thousands/millions/billions
    if n["percent"]:
        out.append((n["abs"] / 100, half_unit / 100))           # 24.1% vs a ratio of 0.241
    else:
        out.append((n["abs"] * 100, half_unit * 100))           # 0.241 vs 24.1%
    return [(v, t) for v, t in out if v > 0 and math.isfinite(v)]


def grounded(n: dict, pool: list[float]) -> bool:
    """Is answer figure n a rounding of (or within 0.5% of) some figure in the
    pool of evidence numbers and calculator outputs (compared by magnitude)?"""
    vs = variants(n)
    for target in pool:
        t = abs(target)
        for v, tol in vs:
            if abs(v - t) <= max(tol, 0.005 * t) + 1e-9:
                return True
    return False


def ungrounded_figures(answer: str, evidence_texts: list[str], calc_values: list[float],
                       question: str = "") -> list[str]:
    """Figures in `answer` that do not match any evidence or calculator number."""
    pool = [n["abs"] * n["scale"] for t in evidence_texts for n in extract_numbers(t)]
    pool += [n["abs"] for t in evidence_texts for n in extract_numbers(t)]
    pool += [abs(v) for v in calc_values]
    pool += [abs(v) * 100 for v in calc_values]
    pool += [n["abs"] * n["scale"] for n in extract_numbers(question)]
    bad = []
    for n in extract_numbers(answer):
        if is_checkable(n) and not grounded(n, pool):
            bad.append(n["text"])
    return bad
