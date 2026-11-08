"""Local (Ollama) backend: message conversion and response handling, against a
mocked /api/chat endpoint (no Ollama needed)."""
import asyncio
import json

import httpx
from pydantic import BaseModel

from docagent.agent.llm_local import LocalLLM, to_ollama_messages, to_ollama_tools


def test_history_conversion():
    msgs = [
        {"role": "user", "content": "Question: capex?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "x"},
            {"type": "text", "text": "Searching."},
            {"type": "tool_use", "id": "t1", "name": "search_documents", "input": {"query": "capex"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "{\"passages\": []}"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "submit_answer", "input": {"answer": "x"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": "Answer rejected", "is_error": True}]},
    ]
    out = to_ollama_messages("SYS", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert out[2]["content"] == "Searching."
    assert out[2]["tool_calls"][0]["function"] == {"name": "search_documents", "arguments": {"query": "capex"}}
    assert out[3]["tool_name"] == "search_documents"
    assert out[5]["content"].startswith("ERROR:") and out[5]["tool_name"] == "submit_answer"


def test_tool_schema_conversion():
    t = to_ollama_tools([{"name": "calculate", "description": "d", "strict": True,
                          "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}}])
    assert t[0]["type"] == "function" and t[0]["function"]["parameters"]["properties"]["expression"]


def _llm_with(responder) -> LocalLLM:
    llm = LocalLLM(model="test-model", host="127.0.0.1:11434")
    llm.client = httpx.AsyncClient(transport=httpx.MockTransport(responder), timeout=10)
    return llm


def test_create_tool_call_and_text():
    def responder(request):
        body = json.loads(request.content)
        assert body["model"] == "test-model" and body["tools"][0]["function"]["name"] == "calculate"
        # reasoning call: the model's own recommended sampling (no greedy override), fixed seed
        assert body["think"] is True and "temperature" not in body["options"] and body["options"]["seed"] == 0
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
                                                     "tool_calls": [{"function": {"name": "calculate",
                                                                                  "arguments": {"expression": "1+1"}}}]},
                                         "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 7})
    llm = _llm_with(responder)
    tools = [{"name": "calculate", "description": "", "input_schema": {"type": "object"}}]
    resp, rec = asyncio.run(llm.create(system="s", messages=[{"role": "user", "content": "hi"}], tools=tools))
    assert resp.stop_reason == "tool_use"
    b = resp.content[0].to_dict()
    assert b["type"] == "tool_use" and b["name"] == "calculate" and b["input"] == {"expression": "1+1"}
    assert rec["input_tokens"] == 50 and rec["cost_usd"] == 0.0

    llm = _llm_with(lambda r: httpx.Response(200, json={"message": {"content": "Done."}, "done_reason": "stop"}))
    resp, _ = asyncio.run(llm.create(system="s", messages=[{"role": "user", "content": "hi"}]))
    assert resp.stop_reason == "end_turn" and resp.content[0].text == "Done."


def test_parse_structured_output():
    class Grade(BaseModel):
        label: str

    def responder(request):
        assert json.loads(request.content)["format"]["properties"]["label"]
        return httpx.Response(200, json={"message": {"content": "{\"label\": \"correct\"}"}, "done_reason": "stop"})
    parsed, rec, _ = asyncio.run(_llm_with(responder).parse(system="s", messages=[{"role": "user", "content": "q"}],
                                                            output_format=Grade))
    assert parsed.label == "correct"


def test_runaway_generation_is_resampled():
    """A turn that hits the token limit while still reasoning is re-sampled with a new seed."""
    seen = []

    def responder(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json={"message": {"content": "", "thinking": "wait " * 50},
                                             "done_reason": "length", "prompt_eval_count": 40, "eval_count": 4096})
        return httpx.Response(200, json={"message": {"content": "Done."}, "done_reason": "stop",
                                         "prompt_eval_count": 40, "eval_count": 9})
    resp, rec = asyncio.run(_llm_with(responder).create(system="s", messages=[{"role": "user", "content": "hi"}]))
    assert [b["options"]["seed"] for b in seen] == [0, 1]
    assert resp.content[0].text == "Done." and rec["attempts"] == 2 and rec["output_tokens"] == 4105


def test_parse_retries_then_answers_without_reasoning():
    class Grade(BaseModel):
        label: str

    seen = []

    def responder(request):
        seen.append(json.loads(request.content))
        ok = len(seen) == 3
        return httpx.Response(200, json={"message": {"content": "{\"label\": \"correct\"}" if ok else ""},
                                         "done_reason": "stop" if ok else "length"})
    parsed, rec, _ = asyncio.run(_llm_with(responder).parse(system="s", messages=[{"role": "user", "content": "q"}],
                                                            output_format=Grade))
    assert parsed.label == "correct" and rec["attempts"] == 3
    assert [b["think"] for b in seen] == [True, True, False]
    assert seen[2]["options"]["temperature"] == 0.7          # Qwen's non-thinking recommendation


def test_context_fitting_shortens_oldest_results_first():
    from docagent.agent.graph import SHORT_NOTE, _fit
    big = json.dumps({"passages": [{"evidence_id": f"D:p{i}:c0", "doc_name": "D", "page": i, "text": "x" * 3000}
                                   for i in range(3)]})
    msgs = [{"role": "user", "content": "Question: q"}]
    for k in range(4):
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{k}", "name": "search_documents",
                                                    "input": {"query": "q"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": big}]}]
    fitted, n = _fit(msgs, budget_tokens=8000, overhead_tokens=500)
    assert n >= 1 and fitted[0] == msgs[0]
    assert SHORT_NOTE in fitted[2]["content"][0]["content"]               # oldest result shortened
    assert fitted[-1]["content"][0]["content"] == big                     # latest result untouched
    ids = [p["evidence_id"] for p in json.loads(fitted[2]["content"][0]["content"])["passages"]]
    assert ids == ["D:p0:c0", "D:p1:c0", "D:p2:c0"]                       # ids stay citable
    assert _fit(msgs[:3], budget_tokens=8000, overhead_tokens=500) == (msgs[:3], 0)   # fits: unchanged
