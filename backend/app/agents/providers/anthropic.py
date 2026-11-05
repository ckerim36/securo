from __future__ import annotations

import json
import re
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import httpx

from app.agents.config import get_agent_settings

from app.agents.providers.base import (
    ChatChunk,
    ChatMessage,
    LLMAuthError,
    LLMNotSupportedError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
    ToolDefinition,
    Usage,
)


# Claude 5 generation models think on every request (Opus 5.5 and Fable 5.1
# reject a request that tries to disable it). Their responses carry signed
# `thinking` blocks that a tool loop must replay unchanged, their output cap
# has to leave room for the reasoning, and depth is steered with `effort`.
_ALWAYS_THINKING = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-5")
# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
# Claude Haiku 5.5 has none, so it is left out on purpose.
_SERVER_FALLBACK = re.compile(r"^claude-(opus-5|fable-5|sonnet-5-5)")
_DEFAULT_MAX_TOKENS = 4096
_THINKING_MAX_TOKENS = 32000
_BETA_THINKING_BINDING = "thinking-binding-controls-2026-08-01"
_BETA_SERVER_FALLBACK = "server-side-fallback-2026-07-01"
_REFUSAL_TEXT = "The model declined to answer this request. Try rephrasing it."


def _always_thinks(model: str) -> bool:
    return bool(_ALWAYS_THINKING.match(model or ""))


def _split_system(messages: list[ChatMessage]) -> tuple[Optional[str], list[ChatMessage]]:
    sys: list[str] = []
    rest: list[ChatMessage] = []
    for m in messages:
        if m.role == "system" and m.content:
            sys.append(m.content)
        else:
            rest.append(m)
    return ("\n\n".join(sys) if sys else None), rest


def _serialize_messages(messages: list[ChatMessage]) -> list[dict]:
    """Anthropic Messages API: alternating user/assistant, tool results
    are user-role messages with content blocks."""
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content or "",
            }
            # Results of parallel calls go back together in one user turn;
            # one turn per result teaches the model to stop calling tools in
            # parallel.
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and all(b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if m.role == "assistant" and m.raw_content:
            # Verbatim replay keeps signed thinking blocks valid.
            out.append({"role": "assistant", "content": m.raw_content})
            continue
        if m.role == "assistant" and m.tool_calls:
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": m.role, "content": m.content or ""})
    return out


def _serialize_tools(tools: Optional[list[ToolDefinition]]) -> Optional[list[dict]]:
    if not tools:
        return None
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]


def _raise_for_status(status: int, body: str) -> None:
    if status in (401, 403):
        raise LLMAuthError(f"Anthropic auth failed: {body}", status=status)
    if status == 429:
        raise LLMRateLimitError(f"Anthropic rate limit: {body}", status=status)
    if status >= 500:
        raise LLMUnavailableError(f"Anthropic {status}: {body}", status=status)
    if status >= 400:
        raise LLMUnavailableError(f"Anthropic {status}: {body}", status=status)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_embeddings = False

    def __init__(self, *, api_key: str, base_url: str = "https://api.anthropic.com/v1", model: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url, model=model)

    def _headers(self, betas: Optional[list[str]] = None) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def _is_first_party(self) -> bool:
        # Server-side fallbacks exist on the Claude API itself, not
        # necessarily on a proxy or gateway configured as base_url.
        return (urlparse(self.base_url).hostname or "") == "api.anthropic.com"

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        url = f"{self.base_url.rstrip('/')}/messages"
        system, rest = _split_system(messages)
        # `temperature` is intentionally not sent: current Claude models reject it
        # with "temperature is deprecated for this model" (issue #542).
        payload: dict = {
            "model": model,
            "messages": _serialize_messages(rest),
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _serialize_tools(tools)
        betas: list[str] = []
        if _always_thinks(model):
            # Thinking shares the output budget with the reply, so the old
            # 4096 default would cut answers off mid-sentence.
            payload["max_tokens"] = max_tokens or _THINKING_MAX_TOKENS
            # If a replayed thinking block ever fails the conversation check,
            # drop it instead of failing the whole request with a 400.
            payload["thinking"] = {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            betas.append(_BETA_THINKING_BINDING)
            payload["output_config"] = {"effort": get_agent_settings().anthropic_effort}
            # System prompt and tool list repeat on every loop iteration.
            payload["cache_control"] = {"type": "ephemeral"}
            if _SERVER_FALLBACK.match(model) and self._is_first_party():
                payload["fallbacks"] = "default"
                betas.append(_BETA_SERVER_FALLBACK)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream("POST", url, json=payload, headers=self._headers(betas)) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        _raise_for_status(resp.status_code, body)
                    open_block: dict = {}
                    usage = Usage()
                    # The assistant turn as received, for verbatim replay.
                    raw_open: dict[int, dict] = {}
                    raw_done: list[dict] = []
                    emitted_text = False
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str:
                            continue
                        evt = json.loads(data_str)
                        et = evt.get("type")
                        if et == "content_block_start":
                            block = evt.get("content_block") or {}
                            idx = evt.get("index", 0)
                            btype = block.get("type")
                            if btype == "tool_use":
                                tcid = block.get("id") or f"toolu_{idx}"
                                open_block[idx] = {"kind": "tool", "id": tcid, "name": block.get("name") or ""}
                                raw_open[idx] = {
                                    "type": "tool_use", "id": tcid, "name": open_block[idx]["name"], "input": {},
                                    "_args": "",
                                }
                                yield ChatChunk(type="tool_call_start", tool_call_id=tcid, tool_name=open_block[idx]["name"])
                            elif btype == "text":
                                open_block[idx] = {"kind": "text"}
                                raw_open[idx] = {"type": "text", "text": block.get("text") or ""}
                            elif btype == "thinking":
                                raw_open[idx] = {
                                    "type": "thinking",
                                    "thinking": block.get("thinking") or "",
                                    "signature": block.get("signature") or "",
                                }
                            elif btype == "redacted_thinking":
                                raw_open[idx] = dict(block)
                            elif btype == "fallback":
                                # A refused model handed over mid-response. Its
                                # reasoning and tool calls must not be echoed
                                # back; text before the boundary is kept.
                                raw_done = [b for b in raw_done if b["type"] == "text"]
                        elif et == "content_block_delta":
                            idx = evt.get("index", 0)
                            delta = evt.get("delta") or {}
                            blk = open_block.get(idx) or {}
                            raw = raw_open.get(idx)
                            dtype = delta.get("type")
                            if dtype == "text_delta" and delta.get("text"):
                                emitted_text = True
                                if raw is not None:
                                    raw["text"] += delta["text"]
                                yield ChatChunk(type="text_delta", text=delta["text"])
                            elif dtype == "input_json_delta" and blk.get("kind") == "tool":
                                if raw is not None:
                                    raw["_args"] += delta.get("partial_json") or ""
                                yield ChatChunk(
                                    type="tool_call_args_delta",
                                    tool_call_id=blk["id"],
                                    args_delta=delta.get("partial_json") or "",
                                )
                            elif dtype == "thinking_delta" and raw is not None:
                                raw["thinking"] += delta.get("thinking") or ""
                            elif dtype == "signature_delta" and raw is not None:
                                raw["signature"] += delta.get("signature") or ""
                        elif et == "content_block_stop":
                            idx = evt.get("index", 0)
                            blk = open_block.pop(idx, None)
                            raw = raw_open.pop(idx, None)
                            if raw is not None:
                                if raw["type"] == "tool_use":
                                    args = raw.pop("_args")
                                    try:
                                        raw["input"] = json.loads(args) if args else {}
                                    except json.JSONDecodeError:
                                        raw["input"] = {}
                                # The API rejects empty text blocks on replay;
                                # empty thinking blocks are valid and must stay.
                                if raw["type"] != "text" or raw["text"]:
                                    raw_done.append(raw)
                            if blk and blk.get("kind") == "tool":
                                yield ChatChunk(type="tool_call_end", tool_call_id=blk["id"])
                        elif et == "message_delta":
                            u = (evt.get("usage") or {})
                            usage = Usage(
                                input_tokens=usage.input_tokens or int(u.get("input_tokens") or 0),
                                output_tokens=int(u.get("output_tokens") or usage.output_tokens),
                            )
                            stop = (evt.get("delta") or {}).get("stop_reason")
                            if stop:
                                if stop == "refusal" and not emitted_text:
                                    yield ChatChunk(type="text_delta", text=_REFUSAL_TEXT)
                                if raw_done:
                                    yield ChatChunk(type="raw_content", raw_content=raw_done)
                                yield ChatChunk(type="usage", usage=usage)
                                yield ChatChunk(type="finish", finish_reason=stop)
                        elif et == "message_start":
                            u = ((evt.get("message") or {}).get("usage") or {})
                            usage = Usage(
                                input_tokens=int(u.get("input_tokens") or 0),
                                output_tokens=int(u.get("output_tokens") or 0),
                            )
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Anthropic unreachable: {exc}") from exc

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        raise LLMNotSupportedError("Anthropic does not provide an embeddings endpoint")
