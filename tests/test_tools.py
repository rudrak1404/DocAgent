import math

import pytest

from docagent.tools.calculator import CalcError, financial_ratio, safe_eval
from docagent.tools.numbers import extract_numbers, grounded, ungrounded_figures


# ------------------------------------------------------------------ calculator
def test_safe_eval_arithmetic():
    assert safe_eval("(6439 - 1577) / 32765 * 100") == pytest.approx((6439 - 1577) / 32765 * 100)
    assert safe_eval("1,577.5 + 2") == 1579.5                 # thousands separators allowed
    assert safe_eval("round(2/3, 4)") == 0.6667
    assert safe_eval("sqrt(16) + abs(-2) + max(1, 3)") == 9


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "().__class__", "open('x')", "a + 1",
    "[1, 2]", "lambda: 1", "1 if True else 2", "x.y",
])
def test_safe_eval_rejects_non_arithmetic(expr):
    with pytest.raises(CalcError):
        safe_eval(expr)


def test_safe_eval_guards():
    with pytest.raises(CalcError):
        safe_eval("1/0")
    with pytest.raises(CalcError):
        safe_eval("10 ** 1000")
    with pytest.raises(CalcError):
        safe_eval("1" * 500)


def test_financial_ratios():
    r = financial_ratio("fixed_asset_turnover", {"revenue": 100, "net_ppe_begin": 40, "net_ppe_end": 60})
    assert r["value"] == 2.0 and "average" in r["formula"]
    r = financial_ratio("CAGR", {"begin_value": 100, "end_value": 121, "years": 2})
    assert r["value"] == pytest.approx(0.10) and r["value_percent"] == pytest.approx(10.0)
    assert financial_ratio("free cash flow", {"operating_cash_flow": 6439, "capital_expenditure": -1577})["value"] == 4862
    with pytest.raises(CalcError, match="unknown ratio"):
        financial_ratio("vibes", {})
    with pytest.raises(CalcError, match="missing"):
        financial_ratio("current_ratio", {"current_assets": 1})


# --------------------------------------------------------------------- numbers
def test_extract_numbers():
    ns = extract_numbers("Capex was $1,577 million (1,373 in 2017), margin 24.1%, about $1.58 billion.")
    vals = [(n["abs"], n["scale"], n["percent"]) for n in ns]
    assert (1577.0, 1e6, False) in vals and (24.1, 1.0, True) in vals and (1.58, 1e9, False) in vals
    neg = extract_numbers("(1,577)")[0]
    assert neg["value"] == -1577


def test_grounding_handles_units_rounding_and_percent():
    n = lambda s: extract_numbers(s)[0]
    assert grounded(n("$1,577 million"), [1577])          # table "in millions"
    assert grounded(n("$1.6 billion"), [1577])            # valid rounding of 1,577 million
    assert not grounded(n("$1,600 million"), [1577])      # stated precisely, but wrong
    assert grounded(n("24.1%"), [0.2412])                 # percent vs calculator fraction
    assert grounded(n("1577.00"), [1577])


def test_ungrounded_figures_ignores_years_and_question_numbers():
    ev = ["Purchases of PP&E | (1,577) | (1,373)"]
    assert ungrounded_figures("FY2018 capex was $1,577 million.", ev, [], "What is FY2018 capex?") == []
    assert ungrounded_figures("FY2018 capex was $1,999 million.", ev, [], "") == ["$1,999 million"]
    assert ungrounded_figures("Ratio is 0.74.", ev, [0.7413], "") == []
