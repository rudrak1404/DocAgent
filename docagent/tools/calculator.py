"""Deterministic finance arithmetic for the agent.

* `safe_eval`: arithmetic on numbers only. The expression is parsed to a Python
  AST and walked against a whitelist (numbers, + - * / ** %, parentheses, and a
  few pure functions). No names, attributes, calls to anything else, or
  `eval`, so text copied out of a document can never execute code.
* `financial_ratio`: named ratios with their formulas, so the model does not
  have to remember (or invent) a definition. Averages follow FinanceBench's
  convention: average of the opening and closing balance.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt,
          "log": math.log, "exp": math.exp}
MAX_LEN = 400


class CalcError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalcError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalcError("division by zero")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression. Thousands separators are allowed
    (1,577.0); currency symbols, units and words are not."""
    if len(expression) > MAX_LEN:
        raise CalcError("expression too long")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.strip())
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"not a valid arithmetic expression: {e.msg}") from None
    value = _eval(tree)
    if isinstance(value, complex) or not math.isfinite(value):
        raise CalcError("result is not a finite real number")
    return float(value)


# ------------------------------------------------------------------ ratios
def _avg(begin: float, end: float) -> float:
    return (begin + end) / 2.0


@dataclass(frozen=True)
class Ratio:
    formula: str
    inputs: tuple[str, ...]
    fn: object
    unit: str = "ratio"           # ratio | percent | days | currency | times


RATIOS: dict[str, Ratio] = {
    "gross_margin": Ratio("gross_profit / revenue", ("gross_profit", "revenue"),
                          lambda d: d["gross_profit"] / d["revenue"], "percent"),
    "operating_margin": Ratio("operating_income / revenue", ("operating_income", "revenue"),
                              lambda d: d["operating_income"] / d["revenue"], "percent"),
    "net_profit_margin": Ratio("net_income / revenue", ("net_income", "revenue"),
                               lambda d: d["net_income"] / d["revenue"], "percent"),
    "ebitda": Ratio("operating_income + depreciation_and_amortization",
                    ("operating_income", "depreciation_and_amortization"),
                    lambda d: d["operating_income"] + d["depreciation_and_amortization"], "currency"),
    "ebitda_margin": Ratio("(operating_income + depreciation_and_amortization) / revenue",
                           ("operating_income", "depreciation_and_amortization", "revenue"),
                           lambda d: (d["operating_income"] + d["depreciation_and_amortization"]) / d["revenue"], "percent"),
    "current_ratio": Ratio("current_assets / current_liabilities", ("current_assets", "current_liabilities"),
                           lambda d: d["current_assets"] / d["current_liabilities"]),
    "quick_ratio": Ratio("(cash_and_equivalents + short_term_investments + accounts_receivable) / current_liabilities",
                         ("cash_and_equivalents", "short_term_investments", "accounts_receivable", "current_liabilities"),
                         lambda d: (d["cash_and_equivalents"] + d["short_term_investments"] + d["accounts_receivable"])
                         / d["current_liabilities"]),
    "cash_ratio": Ratio("cash_and_equivalents / current_liabilities", ("cash_and_equivalents", "current_liabilities"),
                        lambda d: d["cash_and_equivalents"] / d["current_liabilities"]),
    "working_capital": Ratio("current_assets - current_liabilities", ("current_assets", "current_liabilities"),
                             lambda d: d["current_assets"] - d["current_liabilities"], "currency"),
    "debt_to_equity": Ratio("total_debt / total_equity", ("total_debt", "total_equity"),
                            lambda d: d["total_debt"] / d["total_equity"]),
    "debt_to_assets": Ratio("total_debt / total_assets", ("total_debt", "total_assets"),
                            lambda d: d["total_debt"] / d["total_assets"]),
    "interest_coverage": Ratio("operating_income / interest_expense", ("operating_income", "interest_expense"),
                               lambda d: d["operating_income"] / d["interest_expense"], "times"),
    "return_on_assets": Ratio("net_income / average(total_assets_begin, total_assets_end)",
                              ("net_income", "total_assets_begin", "total_assets_end"),
                              lambda d: d["net_income"] / _avg(d["total_assets_begin"], d["total_assets_end"]), "percent"),
    "return_on_equity": Ratio("net_income / average(total_equity_begin, total_equity_end)",
                              ("net_income", "total_equity_begin", "total_equity_end"),
                              lambda d: d["net_income"] / _avg(d["total_equity_begin"], d["total_equity_end"]), "percent"),
    "asset_turnover": Ratio("revenue / average(total_assets_begin, total_assets_end)",
                            ("revenue", "total_assets_begin", "total_assets_end"),
                            lambda d: d["revenue"] / _avg(d["total_assets_begin"], d["total_assets_end"]), "times"),
    "fixed_asset_turnover": Ratio("revenue / average(net_ppe_begin, net_ppe_end)",
                                  ("revenue", "net_ppe_begin", "net_ppe_end"),
                                  lambda d: d["revenue"] / _avg(d["net_ppe_begin"], d["net_ppe_end"]), "times"),
    "inventory_turnover": Ratio("cost_of_goods_sold / average(inventory_begin, inventory_end)",
                                ("cost_of_goods_sold", "inventory_begin", "inventory_end"),
                                lambda d: d["cost_of_goods_sold"] / _avg(d["inventory_begin"], d["inventory_end"]), "times"),
    "days_inventory_outstanding": Ratio("365 * average(inventory_begin, inventory_end) / cost_of_goods_sold",
                                        ("inventory_begin", "inventory_end", "cost_of_goods_sold"),
                                        lambda d: 365 * _avg(d["inventory_begin"], d["inventory_end"]) / d["cost_of_goods_sold"], "days"),
    "days_sales_outstanding": Ratio("365 * average(accounts_receivable_begin, accounts_receivable_end) / revenue",
                                    ("accounts_receivable_begin", "accounts_receivable_end", "revenue"),
                                    lambda d: 365 * _avg(d["accounts_receivable_begin"], d["accounts_receivable_end"]) / d["revenue"], "days"),
    "days_payable_outstanding": Ratio("365 * average(accounts_payable_begin, accounts_payable_end) / cost_of_goods_sold",
                                      ("accounts_payable_begin", "accounts_payable_end", "cost_of_goods_sold"),
                                      lambda d: 365 * _avg(d["accounts_payable_begin"], d["accounts_payable_end"]) / d["cost_of_goods_sold"], "days"),
    "free_cash_flow": Ratio("operating_cash_flow - capital_expenditure", ("operating_cash_flow", "capital_expenditure"),
                            lambda d: d["operating_cash_flow"] - abs(d["capital_expenditure"]), "currency"),
    "capex_to_revenue": Ratio("capital_expenditure / revenue", ("capital_expenditure", "revenue"),
                              lambda d: abs(d["capital_expenditure"]) / d["revenue"], "percent"),
    "dividend_payout_ratio": Ratio("dividends_paid / net_income", ("dividends_paid", "net_income"),
                                   lambda d: abs(d["dividends_paid"]) / d["net_income"], "percent"),
    "effective_tax_rate": Ratio("income_tax_expense / pretax_income", ("income_tax_expense", "pretax_income"),
                                lambda d: d["income_tax_expense"] / d["pretax_income"], "percent"),
    "operating_cash_flow_ratio": Ratio("operating_cash_flow / current_liabilities",
                                       ("operating_cash_flow", "current_liabilities"),
                                       lambda d: d["operating_cash_flow"] / d["current_liabilities"]),
    "growth_rate": Ratio("(current - prior) / prior", ("current", "prior"),
                         lambda d: (d["current"] - d["prior"]) / abs(d["prior"]), "percent"),
    "cagr": Ratio("(end_value / begin_value) ** (1 / years) - 1", ("begin_value", "end_value", "years"),
                  lambda d: (d["end_value"] / d["begin_value"]) ** (1 / d["years"]) - 1, "percent"),
}


def financial_ratio(name: str, inputs: dict[str, float]) -> dict:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in RATIOS:
        raise CalcError(f"unknown ratio '{name}'. Known: {', '.join(sorted(RATIOS))}")
    r = RATIOS[key]
    missing = [k for k in r.inputs if k not in inputs]
    if missing:
        raise CalcError(f"{key} needs inputs {list(r.inputs)}; missing {missing}")
    vals = {k: float(inputs[k]) for k in r.inputs}
    try:
        value = float(r.fn(vals))
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    out = {"ratio": key, "formula": r.formula, "inputs": vals, "value": value, "unit": r.unit}
    if r.unit == "percent":
        out["value_percent"] = value * 100
    return out
