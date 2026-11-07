from fastapi.testclient import TestClient

from docagent.api.app import create_app
from test_agent import FakeLLM, tool_use


def make_client(tiny_index, docs, parsed_dir, monkeypatch, turns):
    import docagent.mcp_server as ms
    monkeypatch.setattr(ms, "PARSED_DIR", parsed_dir)
    monkeypatch.setenv("DOCAGENT_RATE_PER_MIN", "2")
    app = create_app(index=tiny_index, docs=docs, llm=FakeLLM(turns), api_key="secret", with_baseline=False)
    return TestClient(app)


def test_health_documents_and_auth(tiny_index, docs, parsed_dir, monkeypatch):
    with make_client(tiny_index, docs, parsed_dir, monkeypatch, []) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["filings"] == 3 and h["auth_required"]
        assert len(c.get("/documents").json()) == 3
        assert c.post("/ask", json={"question": "What was revenue?"}).status_code == 401
        assert c.post("/mcp", json={}).status_code == 401
        assert c.get("/").status_code == 200


def test_ask_and_rate_limit(tiny_index, docs, parsed_dir, monkeypatch):
    answer = ([tool_use("t1", "search_documents", {"query": "total current assets", "company": "Globex"})], "tool_use")
    submit = ([tool_use("t2", "submit_answer", {"answer": "Total current assets were 9,000.",
                                                "citations": ["GLOBEX_2018_10K:p0:c0"], "calculation": "",
                                                "insufficient_evidence": False})], "tool_use")
    with make_client(tiny_index, docs, parsed_dir, monkeypatch, [answer, submit]) as c:
        r = c.post("/ask", json={"question": "Globex total current assets?"}, headers={"X-API-Key": "secret"})
        assert r.status_code == 200 and r.json()["status"] == "answered"
        assert r.json()["citations"][0]["doc_name"] == "GLOBEX_2018_10K"
        c.post("/ask", json={"question": "again please"}, headers={"X-API-Key": "secret"})
        assert c.post("/ask", json={"question": "third time"}, headers={"X-API-Key": "secret"}).status_code == 429
