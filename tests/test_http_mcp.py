"""End-to-end: the FastAPI app served by uvicorn, and an MCP client talking to
the mounted server over real Streamable HTTP."""
import asyncio
import socket
import threading
import time

import uvicorn
from mcp import Client

from docagent.agent.graph import _echo_safe
from docagent.api.app import create_app
from test_agent import FakeLLM


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mcp_over_streamable_http(tiny_index, docs):
    app = create_app(index=tiny_index, docs=docs, llm=FakeLLM([]), api_key="", with_baseline=False)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        async def go():
            async with Client(f"http://127.0.0.1:{port}/mcp") as c:
                names = {t.name for t in (await c.list_tools()).tools}
                assert "search_documents" in names and "financial_ratio" in names
                r = await c.call_tool("calculate", {"expression": "(9000 / 4500)"})
                assert r.structured_content["value"] == 2.0
                r = await c.call_tool("search_documents", {"query": "current assets", "company": "Globex"})
                assert r.structured_content["passages"][0]["doc_name"] == "GLOBEX_2018_10K"
        asyncio.run(go())
    finally:
        server.should_exit = True
        th.join(timeout=10)


def test_echo_safe_drops_pre_fallback_blocks():
    content = [{"type": "thinking", "thinking": "", "signature": "a"},
               {"type": "tool_use", "id": "t0", "name": "x", "input": {}},
               {"type": "text", "text": "partial"},
               {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
               {"type": "thinking", "thinking": "", "signature": "b"},
               {"type": "tool_use", "id": "t1", "name": "y", "input": {}}]
    out = _echo_safe(content)
    assert [b.get("id") or b["type"] for b in out] == ["text", "thinking", "t1"]
    assert _echo_safe(content[:3]) == content[:3]
