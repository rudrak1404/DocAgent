"""MCP server over the real MCP protocol (in-memory transport) and the full
LangGraph agent loop driven by a scripted fake Claude client."""
import asyncio
from types import SimpleNamespace

from mcp import Client

from docagent.agent.graph import DocAgent
from docagent.agent.guardrails import verify
from docagent.mcp_server import create_server


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ guardrails
def test_guardrails():
    ev = {"D:p1:c0": "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)"}
    ok = lambda **kw: verify({"answer": "", "citations": [], "calculation": "", "insufficient_evidence": False, **kw},
                             ev, [], "What was FY2018 capex?")
    assert ok(answer="FY2018 capex was $1,577 million.", citations=["D:p1:c0"]).ok
    assert not ok(answer="FY2018 capex was $1,577 million.").ok                      # no citation
    assert not ok(answer="$1,577 million", citations=["made-up:p9"]).ok              # invented source
    v = ok(answer="FY2018 capex was $1,650 million.", citations=["D:p1:c0"])          # number not in evidence
    assert not v.ok and "1,650" in v.issues[0]
    assert verify({"answer": "Not found", "citations": [], "calculation": "", "insufficient_evidence": True},
                  ev, [], "").refusal


# ------------------------------------------------------------------ MCP server
def test_mcp_tools_over_protocol(tiny_index, docs, parsed_dir):
    server = create_server(tiny_index, docs, parsed_dir=parsed_dir)

    async def go():
        async with Client(server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"list_documents", "search_documents", "read_page", "calculate", "financial_ratio"}
            r = await c.call_tool("search_documents", {"query": "purchases of property plant equipment",
                                                       "company": "ACME", "fiscal_year": 2018})
            p = r.structured_content["passages"][0]
            assert p["evidence_id"] == "ACME_2018_10K:p1:c0" and p["page"] == 2      # 1-based for humans
            r = await c.call_tool("read_page", {"doc_name": "ACME_2018_10K", "page": 2})
            assert r.structured_content["evidence_id"] == "ACME_2018_10K:p1:page"
            assert "(1,577)" in r.structured_content["text"]
            r = await c.call_tool("calculate", {"expression": "6439 - 1577"})
            assert r.structured_content["value"] == 4862
            r = await c.call_tool("financial_ratio", {"ratio": "current_ratio",
                                                      "inputs": {"current_assets": 9000, "current_liabilities": 4500}})
            assert r.structured_content["value"] == 2.0
            bad = await c.call_tool("calculate", {"expression": "__import__('os')"})
            assert bad.is_error
            docs_ = (await c.call_tool("list_documents", {"company": "globex"})).structured_content["documents"]
            assert [d["doc_name"] for d in docs_] == ["GLOBEX_2018_10K"]
    run(go())


# --------------------------------------------------------------- agent loop
class Block(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


class FakeLLM:
    """Replays a script of assistant turns; records what the agent sent."""
    model = "fake"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    @staticmethod
    def _rec(tag):
        return {"tag": tag, "model": "fake", "stop_reason": "x", "input_tokens": 10, "output_tokens": 5,
                "cache_read": 0, "cache_write": 0, "fallback": False, "cost_usd": 0.001, "seconds": 0.0}

    async def parse(self, **kw):
        return None, self._rec("plan"), None              # planner unavailable -> agent continues

    async def create(self, *, system, messages, tools=None, **kw):
        self.seen.append(messages)
        content, stop = self.turns.pop(0)
        return SimpleNamespace(content=[Block(**b) for b in content], stop_reason=stop), self._rec("agent")


def tool_use(i, name, inp):
    return {"type": "tool_use", "id": i, "name": name, "input": inp}


def test_agent_retries_after_guardrail_rejection(tiny_index, docs, parsed_dir):
    server = create_server(tiny_index, docs, parsed_dir=parsed_dir)
    script = [
        ([tool_use("t1", "search_documents", {"query": "purchases of property plant equipment",
                                               "company": "ACME", "fiscal_year": 2018})], "tool_use"),
        # first submission states a figure that is not in the evidence -> rejected
        ([tool_use("t2", "submit_answer", {"answer": "FY2018 capex was $1,650 million.",
                                            "citations": ["ACME_2018_10K:p1:c0"], "calculation": "",
                                            "insufficient_evidence": False})], "tool_use"),
        ([tool_use("t3", "submit_answer", {"answer": "FY2018 capital expenditure was $1,577 million.",
                                            "citations": ["ACME_2018_10K:p1:c0"], "calculation": "",
                                            "insufficient_evidence": False})], "tool_use"),
    ]
    llm = FakeLLM(script)
    res = run(DocAgent(server, llm).ask("What is ACME's FY2018 capital expenditure in USD millions?"))
    assert res["status"] == "answered"
    assert res["answer"] == "FY2018 capital expenditure was $1,577 million."
    assert res["guardrail"]["verify_attempts"] == 1
    assert [c["evidence_id"] for c in res["citations"]] == ["ACME_2018_10K:p1:c0"]
    feedback = llm.seen[-1][-1]["content"][0]
    assert feedback["is_error"] and "1,650" in feedback["content"]
    assert res["tool_calls"] == 1 and res["trace"][0]["tool"] == "search_documents"


def test_agent_refuses_when_it_cannot_ground_the_answer(tiny_index, docs, parsed_dir):
    server = create_server(tiny_index, docs, parsed_dir=parsed_dir)
    bad = ([tool_use("x", "submit_answer", {"answer": "It was $9,999 million.", "citations": [],
                                           "calculation": "", "insufficient_evidence": False})], "tool_use")
    res = run(DocAgent(server, FakeLLM([bad, bad, bad]), max_verify_retries=2).ask("What was ACME's revenue?"))
    assert res["status"] == "refused" and res["unverified_answer"] == "It was $9,999 million."


def test_formula_constants_allowed_in_calculation_only():
    ev = {"A:p1:c0": "Capital expenditures | (155) | (131) | (116)", "A:p2:c0": "Net revenues | 7,017 | 7,500 | 6,489"}
    sub = {"answer": "Capex was 2.2% of revenue.", "citations": ["A:p1:c0", "A:p2:c0"], "insufficient_evidence": False,
           "calculation": "155 / 7017 * 100 = 2.2"}
    assert verify(sub, ev, [2.2089], "capex as % of revenue?").ok          # x100 is a conversion, not a figure
    v = verify({**sub, "answer": "Capex was 100 million."}, ev, [2.2089], "capex?")
    assert not v.ok and "100" in v.issues[0]                               # but not in the answer itself


def test_guardrail_feedback_points_to_received_evidence():
    ev = {"A:p1:c0": "Revenue | 7,017 | 6,489", "A:p2:c0": "Capital expenditures | (155) | (116)"}
    v = verify({"answer": "Capex was $155 million.", "citations": ["A:p9:c9"], "calculation": "",
                "insufficient_evidence": False}, ev, [], "What was capex?")
    assert not v.ok
    assert "Received so far" in v.issues[0] and "A:p2:c0" in v.issues[0]
    assert "appears in ['A:p2:c0']" in v.issues[1]
