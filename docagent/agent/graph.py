"""DocAgent: a LangGraph agent that plans, retrieves through MCP tools, calculates,
and passes deterministic guardrails before an answer leaves the system.

    START -> plan -> agent --tool calls--> tools ---------> agent
                       |  --submit_answer--> verify --fail--> agent  (feedback, max 2 retries)
                       |                       '--pass------> END
                       |  --no tool call----> nudge ------> agent
                       |  --tool budget used--> budget ---> agent
                       '--refusal / gave up--> END

* plan:   Claude produces a structured research plan (company, period, facts,
          formula, search queries) at low effort.
* agent:  Claude (official Anthropic SDK) with the MCP server's tools plus a
          strict `submit_answer` tool. Thinking blocks are passed back unchanged.
* tools:  every tool call is a real MCP `tools/call` (in-memory transport when
          the server runs in-process, Streamable HTTP when it is remote). Each
          returned passage/page is recorded in an evidence registry.
* verify: citation and numeric-grounding guardrails (agent/guardrails.py).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client
from pydantic import BaseModel, Field

from docagent.agent.guardrails import verify
from docagent.agent.llm import LLM
from docagent.agent.prompts import AGENT_SYSTEM, BUDGET_EXHAUSTED, NUDGE, PLANNER_SYSTEM, SUBMIT_TOOL
from docagent.config import MAX_TOOL_CALLS, MAX_VERIFY_RETRIES

MAX_NUDGES = 3


class Plan(BaseModel):
    company: str = Field(description="company the question is about")
    fiscal_period: str = Field(description="fiscal year or quarter, e.g. FY2018 or Q2 2023; empty if none")
    filing: str = Field(description="most likely source filing type, e.g. 10-K")
    facts_needed: list[str] = Field(description="the specific figures or statements to find")
    calculation: str = Field(description="formula to apply, or empty if none")
    search_queries: list[str] = Field(description="2-4 short search queries using the wording filings use")


class State(TypedDict, total=False):
    question: str
    messages: list
    evidence: dict          # evidence_id -> {"text", "doc_name", "page"}
    calc_values: list
    plan: dict | None
    tool_calls: int
    verify_attempts: int
    nudges: int
    pending: list           # tool_use blocks of the last assistant turn
    stop_reason: str
    submission: dict | None
    issues: list
    status: str             # answered | refused | error
    calls: list             # LLM call records (tokens, cost, latency)
    trace: list             # tool call records


def _render_plan(p: dict) -> str:
    lines = [f"- company: {p['company']}", f"- period: {p['fiscal_period'] or 'n/a'}",
             f"- likely filing: {p['filing']}", "- facts needed: " + "; ".join(p["facts_needed"])]
    if p.get("calculation"):
        lines.append(f"- calculation: {p['calculation']}")
    lines.append("- suggested searches: " + " | ".join(p["search_queries"]))
    return "\n".join(lines)


def _echo_safe(content: list[dict]) -> list[dict]:
    """After a mid-output server-side fallback the response contains a `fallback`
    block; thinking / tool_use blocks produced *before* the last such block must not
    be sent back (and those tool calls must not run). Everything after it is kept."""
    marks = [i for i, b in enumerate(content) if b.get("type") == "fallback"]
    if not marks:
        return content
    cut = marks[-1]
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(content) if i > cut or (b.get("type") not in drop and b.get("type") != "fallback")]


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    out = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        out["is_error"] = True
    return out


class DocAgent:
    def __init__(self, mcp_target, llm: LLM | None = None, max_tool_calls: int = MAX_TOOL_CALLS,
                 max_verify_retries: int = MAX_VERIFY_RETRIES, use_planner: bool = True):
        """mcp_target: an MCPServer object (in-process, in-memory transport) or a
        Streamable HTTP URL such as http://localhost:8080/mcp."""
        self.mcp_target = mcp_target
        self.llm = llm or LLM()
        self.max_tool_calls = max_tool_calls
        self.max_verify_retries = max_verify_retries
        self.use_planner = use_planner

    # ------------------------------------------------------------------ graph
    def _build(self, client: Client, tools: list):
        llm = self.llm

        async def plan(s: State) -> dict:
            q = s["question"]
            calls, plan_obj, text = [], None, ""
            if self.use_planner:
                try:
                    parsed, rec, _ = await llm.parse(
                        system=PLANNER_SYSTEM, messages=[{"role": "user", "content": f"Question: {q}"}],
                        output_format=Plan, effort="low", max_tokens=6000, tag="plan")
                    calls.append(rec)
                    if parsed is not None:
                        plan_obj = parsed.model_dump()
                        text = _render_plan(plan_obj)
                except Exception as e:          # planning is an aid, never a point of failure
                    calls.append({"tag": "plan", "error": repr(e)})
            first = f"Question: {q}"
            if text:
                first += f"\n\nResearch plan from a planner (adapt it if the filings say otherwise):\n{text}"
            return {"plan": plan_obj, "messages": [{"role": "user", "content": first}], "calls": calls}

        async def agent(s: State) -> dict:
            resp, rec = await llm.create(system=AGENT_SYSTEM, messages=s["messages"], tools=tools, tag="agent")
            content = _echo_safe([b.to_dict() for b in resp.content])
            return {"messages": s["messages"] + [{"role": "assistant", "content": content}],
                    "pending": [b for b in content if b.get("type") == "tool_use"],
                    "stop_reason": resp.stop_reason, "calls": s["calls"] + [rec]}

        def route(s: State) -> str:
            if s["stop_reason"] == "refusal":
                return "refuse"
            names = [b["name"] for b in s["pending"]]
            if "submit_answer" in names:
                return "verify"
            if names:
                return "tools" if s["tool_calls"] < self.max_tool_calls else "budget"
            return "nudge" if s["nudges"] < MAX_NUDGES else "give_up"

        async def call_one(block: dict) -> tuple[dict, dict]:
            t0 = time.time()
            res = await client.call_tool(block["name"], block.get("input") or {})
            rec = {"tool": block["name"], "input": block.get("input"), "seconds": round(time.time() - t0, 2),
                   "is_error": bool(res.is_error)}
            if res.is_error or res.structured_content is None:
                text = " ".join(getattr(c, "text", "") for c in res.content) or "tool error"
                rec["error"] = text[:500]
                return _tool_result(block["id"], text, is_error=bool(res.is_error)), rec | {"structured": None}
            sc = res.structured_content
            return _tool_result(block["id"], json.dumps(sc, ensure_ascii=False, separators=(",", ":"))), \
                rec | {"structured": sc}

        async def tools_node(s: State) -> dict:
            results = await asyncio.gather(*(call_one(b) for b in s["pending"]))
            evidence, calc = dict(s["evidence"]), list(s["calc_values"])
            trace = list(s["trace"])
            for _, rec in results:
                sc = rec.pop("structured")
                if sc:
                    if rec["tool"] == "search_documents":
                        for p in sc.get("passages", []):
                            evidence[p["evidence_id"]] = {"text": p["text"], "doc_name": p["doc_name"], "page": p["page"]}
                        rec["returned"] = [p["evidence_id"] for p in sc.get("passages", [])]
                    elif rec["tool"] == "read_page":
                        evidence[sc["evidence_id"]] = {"text": sc["text"], "doc_name": sc["doc_name"], "page": sc["page"]}
                        rec["returned"] = [sc["evidence_id"]]
                    elif rec["tool"] in ("calculate", "financial_ratio"):
                        calc.append(sc["value"])
                        if sc.get("value_percent") is not None:
                            calc.append(sc["value_percent"])
                        rec["value"] = sc["value"]
                trace.append(rec)
            msg = {"role": "user", "content": [r for r, _ in results]}
            return {"messages": s["messages"] + [msg], "evidence": evidence, "calc_values": calc,
                    "trace": trace, "tool_calls": s["tool_calls"] + len(s["pending"])}

        async def budget(s: State) -> dict:
            msg = {"role": "user", "content": [_tool_result(b["id"], BUDGET_EXHAUSTED, True) for b in s["pending"]]}
            return {"messages": s["messages"] + [msg], "nudges": s["nudges"] + 1}

        async def nudge(s: State) -> dict:
            return {"messages": s["messages"] + [{"role": "user", "content": NUDGE}], "nudges": s["nudges"] + 1}

        async def verify_node(s: State) -> dict:
            sub_block = next(b for b in s["pending"] if b["name"] == "submit_answer")
            sub = sub_block.get("input") or {}
            texts = {k: v["text"] for k, v in s["evidence"].items()}
            v = verify(sub, texts, s["calc_values"], s["question"])
            if v.ok:
                return {"submission": sub, "status": "refused" if v.refusal else "answered", "issues": []}
            if s["verify_attempts"] < self.max_verify_retries:
                feedback = "Answer rejected by verification:\n- " + "\n- ".join(v.issues) + \
                           "\nFix the problems (research more if needed) and call submit_answer again."
                results = [_tool_result(b["id"], feedback if b is sub_block else
                                        "Not executed: an answer was submitted in the same turn.", True)
                           for b in s["pending"]]
                return {"messages": s["messages"] + [{"role": "user", "content": results}],
                        "verify_attempts": s["verify_attempts"] + 1, "issues": v.issues}
            return {"submission": sub, "status": "refused", "issues": v.issues}

        def after_verify(s: State) -> str:
            return END if s.get("status") else "agent"

        async def refuse(s: State) -> dict:
            return {"status": "refused", "issues": ["model declined to answer"]}

        async def give_up(s: State) -> dict:
            return {"status": "refused", "issues": ["no answer submitted"]}

        g = StateGraph(State)
        for name, fn in [("plan", plan), ("agent", agent), ("tools", tools_node), ("budget", budget),
                         ("nudge", nudge), ("verify", verify_node), ("refuse", refuse), ("give_up", give_up)]:
            g.add_node(name, fn)
        g.add_edge(START, "plan")
        g.add_edge("plan", "agent")
        g.add_conditional_edges("agent", route, ["tools", "budget", "verify", "nudge", "refuse", "give_up"])
        for n in ("tools", "budget", "nudge"):
            g.add_edge(n, "agent")
        g.add_conditional_edges("verify", after_verify, ["agent", END])
        g.add_edge("refuse", END)
        g.add_edge("give_up", END)
        return g.compile()

    # -------------------------------------------------------------------- run
    async def ask(self, question: str) -> dict:
        t0 = time.time()
        state: State = {"question": question, "messages": [], "evidence": {}, "calc_values": [],
                        "plan": None, "tool_calls": 0, "verify_attempts": 0, "nudges": 0, "pending": [],
                        "submission": None, "issues": [], "calls": [], "trace": []}
        try:
            async with Client(self.mcp_target) as client:
                listed = await client.list_tools()
                tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                         for t in sorted(listed.tools, key=lambda t: t.name)] + [SUBMIT_TOOL]
                state = await self._build(client, tools).ainvoke(state, {"recursion_limit": 100})
        except Exception as e:
            state["status"] = "error"
            state["issues"] = [repr(e)]
        return self._result(state, time.time() - t0)

    @staticmethod
    def _result(s: State, seconds: float) -> dict:
        sub = s.get("submission") or {}
        status = s.get("status", "error")
        ev = s.get("evidence", {})
        cites = [c for c in dict.fromkeys(sub.get("citations") or []) if c in ev]
        calls = [c for c in s.get("calls", []) if "cost_usd" in c]
        if status == "answered":
            answer = sub.get("answer", "")
        elif sub.get("insufficient_evidence"):
            answer = sub.get("answer") or "The filings available do not contain enough evidence to answer."
        else:
            answer = "I could not produce a verified answer from the filings."
        return {
            "answer": answer, "status": status,
            "citations": [{"evidence_id": c, "doc_name": ev[c]["doc_name"], "page": ev[c]["page"],
                           "text": ev[c]["text"]} for c in cites],
            "calculation": sub.get("calculation", ""), "plan": s.get("plan"),
            "guardrail": {"verify_attempts": s.get("verify_attempts", 0), "issues": s.get("issues", [])},
            "unverified_answer": sub.get("answer") if status != "answered" else None,
            "tool_calls": s.get("tool_calls", 0), "trace": s.get("trace", []), "llm_calls": s.get("calls", []),
            "usage": {"input_tokens": sum(c["input_tokens"] for c in calls),
                      "output_tokens": sum(c["output_tokens"] for c in calls),
                      "cache_read": sum(c["cache_read"] for c in calls),
                      "cache_write": sum(c["cache_write"] for c in calls),
                      "cost_usd": round(sum(c["cost_usd"] for c in calls), 5),
                      "fallbacks": sum(1 for c in calls if c.get("fallback"))},
            "seconds": round(seconds, 2),
            "evidence_seen": sorted(ev),
        }
