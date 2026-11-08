"""Claude client used by the agent, the baseline and the judges.

Calls go through the official Anthropic Python SDK (beta Messages API):
  * adaptive thinking is always on for Claude Opus 5.5; depth is set with
    `output_config.effort` (there are no sampling parameters on this API);
  * `fallbacks="default"` (beta `server-side-fallback-2026-07-01`) re-runs a
    request that a safety classifier declines on Anthropic's recommended
    fallback model instead of failing; every such event is recorded;
  * top-level `cache_control` caches the growing conversation prefix, so each
    agent turn pays full price only for the new tokens;
  * every call's usage (incl. cache reads/writes) and cost is logged.
"""
from __future__ import annotations

import time

import anthropic

from docagent.config import LLM_EFFORT, LLM_MODEL, PRICES

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def cost_usd(model: str, usage) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    g = lambda k: (getattr(usage, k, 0) or 0)
    return (g("input_tokens") * p["input"] + g("output_tokens") * p["output"]
            + g("cache_read_input_tokens") * p["cache_read"]
            + g("cache_creation_input_tokens") * p["cache_write"]) / 1e6


class LLM:
    def __init__(self, model: str = LLM_MODEL, effort: str = LLM_EFFORT, max_tokens: int = 16000,
                 client: anthropic.AsyncAnthropic | None = None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.client = client or anthropic.AsyncAnthropic(max_retries=8, timeout=600.0)

    def _common(self, effort: str | None, max_tokens: int | None) -> dict:
        return dict(model=self.model, max_tokens=max_tokens or self.max_tokens,
                    output_config={"effort": effort or self.effort},
                    cache_control={"type": "ephemeral"},
                    betas=[FALLBACK_BETA], fallbacks="default")

    @staticmethod
    def record(resp, tag: str, t0: float) -> dict:
        u = resp.usage
        iterations = getattr(u, "iterations", None) or []
        return {
            "tag": tag, "model": resp.model, "stop_reason": resp.stop_reason,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "fallback": any(getattr(it, "type", "") == "fallback_message" for it in iterations),
            "cost_usd": cost_usd(resp.model, u), "seconds": round(time.time() - t0, 2),
            "request_id": getattr(resp, "_request_id", None),
        }

    async def create(self, *, system: str, messages: list, tools: list | None = None,
                     effort: str | None = None, max_tokens: int | None = None, tag: str = ""):
        kw = self._common(effort, max_tokens)
        if tools:
            kw["tools"] = tools
        t0 = time.time()
        resp = await self.client.beta.messages.create(system=system, messages=messages, **kw)
        return resp, self.record(resp, tag, t0)

    async def parse(self, *, system: str, messages: list, output_format, effort: str | None = None,
                    max_tokens: int | None = None, tag: str = ""):
        t0 = time.time()
        resp = await self.client.beta.messages.parse(system=system, messages=messages,
                                                     output_format=output_format,
                                                     **self._common(effort, max_tokens))
        return resp.parsed_output, self.record(resp, tag, t0), resp


def make_llm(role: str = "agent"):
    """The configured LLM backend for a role ("agent" or "judge"):
    DOCAGENT_LLM_PROVIDER=ollama (default, local and free) or anthropic (Claude API)."""
    from docagent.config import JUDGE_MODEL, LLM_PROVIDER, LOCAL_JUDGE_MODEL, LOCAL_MODEL
    if LLM_PROVIDER == "anthropic":
        return LLM(model=JUDGE_MODEL if role == "judge" else LLM_MODEL,
                   effort="medium" if role == "judge" else LLM_EFFORT)
    from docagent.agent.llm_local import LocalLLM
    from docagent.config import LOCAL_JUDGE_CONTEXT
    if role == "judge":
        return LocalLLM(model=LOCAL_JUDGE_MODEL, num_ctx=LOCAL_JUDGE_CONTEXT)
    return LocalLLM(model=LOCAL_MODEL)


def agent_kwargs() -> dict:
    """DocAgent settings that depend on the backend (small local context windows)."""
    from docagent.config import LLM_PROVIDER, LOCAL_CONTEXT, LOCAL_RESULT_CHARS
    if LLM_PROVIDER == "anthropic":
        return {}
    return {"max_result_chars": LOCAL_RESULT_CHARS, "context_tokens": LOCAL_CONTEXT}
