"""Local LLM backend: an open-weight model served by Ollama on this machine.

Drop-in replacement for `LLM` (agent/llm.py): the same `create` / `parse`
interface and the same Anthropic-style message and content-block shapes, so the
LangGraph agent, the LangChain baseline and the judges run unchanged. Requests go
to Ollama's local REST API (/api/chat) with native tool calling and JSON-schema
structured outputs. Every call is logged with token counts; cost is $0.

    ollama pull qwen3:8b
    DOCAGENT_LLM_PROVIDER=ollama DOCAGENT_LOCAL_MODEL=qwen3:8b python -m docagent.evaluation ...
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx

from docagent.config import LOCAL_CONTEXT, LOCAL_MODEL, OLLAMA_HOST


class Block(SimpleNamespace):
    """Content block with the attribute / to_dict() surface of the Anthropic SDK's blocks."""

    def to_dict(self) -> dict:
        return dict(vars(self))


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    """Anthropic-style history -> Ollama chat history. tool_use blocks become
    `tool_calls`; tool_result blocks become role="tool" messages (in order)."""
    out = [{"role": "system", "content": system}] if system else []
    names: dict[str, str] = {}
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            calls = []
            for b in blocks:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b.get("input") or {}}})
            msg = {"role": "assistant", "content": _text_of(blocks)}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif isinstance(content, str):
            out.append({"role": "user", "content": content})
        else:
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    body = b["content"] if isinstance(b["content"], str) else _text_of(b["content"])
                    if b.get("is_error"):
                        body = f"ERROR: {body}"
                    out.append({"role": "tool", "content": body, "tool_name": names.get(b["tool_use_id"], "")})
                elif b.get("type") == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


# Sampling. Reasoning ("thinking") calls use the model's own recommended settings,
# which Ollama ships as Modelfile defaults (Qwen3: temperature 0.6, top-p 0.95,
# top-k 20; Bonsai-2 / Qwen3.5: temperature 1.0, top-p 0.95, top-k 20, presence
# penalty 1). Qwen-family thinking models must not be decoded greedily: in a first
# full run, temperature 0 sent 23% of baseline answers into endless reasoning loops
# that ran into the token limit. Non-thinking calls use Qwen's non-thinking
# recommendation. A fixed seed per attempt keeps runs reproducible.
NOTHINK_OPTIONS = {"temperature": 0.7, "top_p": 0.8}


class LocalLLM:
    def __init__(self, model: str = LOCAL_MODEL, host: str = OLLAMA_HOST, num_ctx: int = LOCAL_CONTEXT,
                 max_tokens: int = 6144, timeout: float = 1800.0):
        host = host if host.startswith("http") else f"http://{host}"
        self.model, self.host, self.num_ctx, self.max_tokens = model, host.rstrip("/"), num_ctx, max_tokens
        self.client = httpx.AsyncClient(timeout=timeout)
        self._n = 0

    async def _chat(self, payload: dict) -> dict:
        r = await self.client.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()

    def _payload(self, system, messages, effort, max_tokens, seed: int = 0, think: bool | None = None) -> dict:
        think = (effort or "high") != "low" if think is None else think   # reasoning on except cheap steps
        options = {"num_ctx": self.num_ctx, "num_predict": max_tokens or self.max_tokens, "seed": seed}
        if not think:
            options |= NOTHINK_OPTIONS
        return {"model": self.model, "messages": to_ollama_messages(system, messages), "stream": False,
                "think": think, "options": options}

    def _record(self, data: dict, tag: str, t0: float, stop: str, attempts: list[dict]) -> dict:
        return {"tag": tag, "model": self.model, "stop_reason": stop,
                "input_tokens": sum(a.get("prompt_eval_count", 0) or 0 for a in attempts),
                "output_tokens": sum(a.get("eval_count", 0) or 0 for a in attempts),
                "cache_read": 0, "cache_write": 0, "fallback": False, "cost_usd": 0.0,
                "seconds": round(time.time() - t0, 2), "attempts": len(attempts),
                "prompt_tokens_last": data.get("prompt_eval_count", 0) or 0}

    async def create(self, *, system: str, messages: list, tools: list | None = None,
                     effort: str | None = None, max_tokens: int | None = None, tag: str = "",
                     retries: int = 1):
        """One chat turn. A turn that ran into the token limit while still reasoning
        (no text, no tool call) is re-sampled with a new seed, up to `retries` times."""
        t0, attempts = time.time(), []
        for attempt in range(retries + 1):
            payload = self._payload(system, messages, effort, max_tokens, seed=attempt)
            if tools:
                payload["tools"] = to_ollama_tools(tools)
            data = await self._chat(payload)
            attempts.append(data)
            msg = data.get("message", {})
            if not (data.get("done_reason") == "length" and not msg.get("tool_calls")
                    and not msg.get("content", "").strip()):
                break
        msg = data.get("message", {})
        blocks = []
        if msg.get("content", "").strip():
            blocks.append(Block(type="text", text=msg["content"]))
        for call in msg.get("tool_calls") or []:
            self._n += 1
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            blocks.append(Block(type="tool_use", id=f"toolu_local_{self._n}", name=fn.get("name", ""), input=args))
        stop = ("tool_use" if any(b.type == "tool_use" for b in blocks)
                else "max_tokens" if data.get("done_reason") == "length" else "end_turn")
        resp = SimpleNamespace(content=blocks, stop_reason=stop, model=self.model)
        return resp, self._record(data, tag, t0, stop, attempts)

    async def parse(self, *, system: str, messages: list, output_format, effort: str | None = None,
                    max_tokens: int | None = None, tag: str = "", retries: int = 2):
        """Structured output (Ollama JSON-schema `format`). If the reply does not
        validate (e.g. the reasoning ran into the token limit), re-sample with a new
        seed; the last attempt answers without reasoning. Returns (None, ...) only if
        every attempt failed."""
        t0, attempts, parsed = time.time(), [], None
        for attempt in range(retries + 1):
            think = False if attempt == retries and attempt > 0 else None
            payload = self._payload(system, messages, effort, max_tokens, seed=attempt, think=think)
            payload["format"] = output_format.model_json_schema()
            data = await self._chat(payload)
            attempts.append(data)
            try:
                parsed = output_format.model_validate_json(data.get("message", {}).get("content", ""))
                break
            except Exception:
                parsed = None
        return parsed, self._record(data, tag, t0, "end_turn", attempts), data
