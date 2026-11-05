"""Coverage for the streaming chat + embed paths in
app/agents/providers/{openai,ollama,anthropic}.py. These providers open
httpx.AsyncClient.stream(...) inline, so we replace AsyncClient with a
fake whose stream() yields a pre-baked SSE/JSONL response.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Iterable

import pytest

import app.agents.providers.anthropic as anthropic_module
import app.agents.providers.ollama as ollama_module
from app.agents.providers.anthropic import AnthropicProvider
from app.agents.providers.base import (
    ChatMessage,
    LLMAuthError,
    LLMNotSupportedError,
    LLMRateLimitError,
    LLMUnavailableError,
    ToolDefinition,
)
from app.agents.providers.ollama import OllamaProvider
from app.agents.providers.openai import OpenAIProvider
from app.agents.providers.openai_compatible import OpenAICompatibleProvider


# --------------------------------------------------------------------- httpx fake

class _FakeStreamResponse:
    """Stand-in for the response object yielded from `client.stream(...)`."""

    def __init__(self, *, status_code: int, lines: Iterable[str], body: bytes = b""):
        self.status_code = status_code
        self._lines = list(lines)
        self._body = body

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self) -> bytes:
        return self._body


class _FakeResponse:
    """Stand-in for a plain (non-streaming) response."""

    def __init__(self, *, status_code: int, json_body=None, text: str = ""):
        self.status_code = status_code
        self._body = json_body
        self.text = text

    def json(self):
        return self._body


class _FakeAsyncClient:
    queue_stream: list[_FakeStreamResponse] = []
    queue_post: list[_FakeResponse] = []
    posted: list[tuple[str, dict]] = []

    def __init__(self, *_, **__):  # accept (timeout=...) kw
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @asynccontextmanager
    async def stream(self, method, url, *, json=None, headers=None):  # noqa: A002
        type(self).posted.append((url, json or {}))
        yield type(self).queue_stream.pop(0)

    async def post(self, url, *, json=None, headers=None):  # noqa: A002
        type(self).posted.append((url, json or {}))
        return type(self).queue_post.pop(0)


@pytest.fixture(autouse=True)
def _fake_httpx(monkeypatch):
    _FakeAsyncClient.queue_stream = []
    _FakeAsyncClient.queue_post = []
    _FakeAsyncClient.posted = []
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(anthropic_module.httpx, "AsyncClient", _FakeAsyncClient)
    yield


# --------------------------------------------------------------------- OpenAI-compatible: chat_stream

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", [OpenAICompatibleProvider, OpenAIProvider])
@pytest.mark.parametrize("signature_timing", ["first", "late", "absent"])
async def test_openai_tool_signatures_survive_multiple_rounds(provider_class, signature_timing):
    """Keep signatures on their own calls across parallel and sequential tools."""
    def response(deltas):
        return _FakeStreamResponse(status_code=200, lines=[
            "data: " + json.dumps({"choices": [{"delta": delta}]}) for delta in deltas
        ] + ['data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}', "data: [DONE]"])

    signature_a = "opaque-A+/=="
    first = {"index": 0, "id": "a", "function": {"name": "lookup", "arguments": '{"city":'}}
    extra_a = {"google": {"thought_signature": signature_a}}
    if signature_timing == "first":
        first["extra_content"] = extra_a
    deltas = [
        {"tool_calls": [first]},
        {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
    ]
    if signature_timing == "late":
        deltas.append({"tool_calls": [{"index": 0, "extra_content": extra_a}]})
    second_call = {"index": 0, "id": "c", "function": {"name": "lookup", "arguments": "{}"}}
    if signature_timing != "absent":
        second_call["extra_content"] = {"google": {"thought_signature": "opaque-B=="}}
    _FakeAsyncClient.queue_stream.extend([
        response(deltas), response([{"tool_calls": [second_call]}]),
        response([{"content": "Done"}]),
    ])
    provider = provider_class(base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    messages = [ChatMessage(role="user", content="Look up cities")]
    for round_index in range(2):
        result = await provider.chat(messages, model="gemini-test")
        if round_index == 0:
            assert [tc.id for tc in result.tool_calls] == ["a", "b"]
            assert result.tool_calls[0].arguments == {"city": "Paris"}
        messages.append(ChatMessage(role="assistant", tool_calls=result.tool_calls))
        messages.extend(ChatMessage(role="tool", tool_call_id=tc.id, content="ok") for tc in result.tool_calls)
    await provider.chat(messages, model="gemini-test")

    assert len(_FakeAsyncClient.posted) == 3
    for url, payload in _FakeAsyncClient.posted:
        assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        for message in payload["messages"]:
            for call in message.get("tool_calls", []):
                if signature_timing == "absent" or call["id"] == "b":
                    assert set(call) == {"id", "type", "function"}
                else:
                    expected = signature_a if call["id"] == "a" else "opaque-B=="
                    assert call["extra_content"] == {"google": {"thought_signature": expected}}
    final_calls = [tc for m in _FakeAsyncClient.posted[-1][1]["messages"] for tc in m.get("tool_calls", [])]
    assert [tc["id"] for tc in final_calls] == ["a", "b", "c"]


# --------------------------------------------------------------------- Ollama: chat_stream

@pytest.mark.asyncio
async def test_ollama_chat_stream_yields_text_and_finish():
    """Two JSONL deltas + a final done frame should produce text deltas
    plus usage + finish chunks."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({"message": {"content": "Hel"}}),
        json.dumps({"message": {"content": "lo"}}),
        json.dumps({
            "message": {"content": ""},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": 7, "eval_count": 4,
        }),
        "",  # blank line — should be skipped
    ]))

    provider = OllamaProvider(base_url="http://ollama:11434")
    chunks = []
    async for chunk in provider.chat_stream(
        [ChatMessage(role="user", content="hi")], model="llama3", temperature=0.1, max_tokens=64,
    ):
        chunks.append(chunk)

    text = "".join(c.text or "" for c in chunks if c.type == "text_delta")
    assert text == "Hello"
    assert any(c.type == "usage" and c.usage and c.usage.input_tokens == 7 for c in chunks)
    assert any(c.type == "finish" and c.finish_reason == "stop" for c in chunks)

    # Verify the request payload picked up our temperature + num_predict.
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 64


@pytest.mark.asyncio
async def test_ollama_chat_stream_emits_tool_call_events():
    """A streaming tool_calls payload should produce start/args/end chunks."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_accounts", "arguments": {"limit": 5}}},
                ],
            },
        }),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "tool_calls"}),
    ]))

    provider = OllamaProvider()
    types = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="llama3"):
        types.append(c.type)
    assert "tool_call_start" in types
    assert "tool_call_args_delta" in types
    assert "tool_call_end" in types


@pytest.mark.asyncio
async def test_ollama_chat_stream_raises_on_http_error():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(
        status_code=503, lines=[], body=b"unavailable",
    ))
    provider = OllamaProvider()
    with pytest.raises(LLMUnavailableError):
        async for _ in provider.chat_stream(
            [ChatMessage(role="user", content="x")], model="llama3",
        ):
            pass


@pytest.mark.asyncio
async def test_ollama_chat_stream_includes_tools_in_payload():
    """When tools are passed, they must show up in the request body."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
    ]))
    provider = OllamaProvider()
    tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
    async for _ in provider.chat_stream(
        [ChatMessage(role="user", content="x")], model="llama3", tools=tools,
    ):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["tools"][0]["function"]["name"] == "t"


# --------------------------------------------------------------------- Ollama: embed

@pytest.mark.asyncio
async def test_ollama_embed_returns_vectors_per_input():
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=200, json_body={
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }))
    provider = OllamaProvider()
    out = await provider.embed(["a", "b"], model="nomic-embed-text")
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_ollama_embed_handles_missing_embeddings_array():
    """If the server returns an unexpected shape, we should yield [] rather than crash."""
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=200, json_body={"unexpected": True}))
    provider = OllamaProvider()
    out = await provider.embed(["a"], model="nomic-embed-text")
    assert out == []


@pytest.mark.asyncio
async def test_ollama_embed_raises_on_http_error():
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=500, text="boom"))
    provider = OllamaProvider()
    with pytest.raises(LLMUnavailableError):
        await provider.embed(["a"], model="nomic-embed-text")


# --------------------------------------------------------------------- Anthropic: chat_stream

@pytest.mark.asyncio
async def test_anthropic_chat_stream_parses_text_block_sequence():
    sse = [
        f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': 5, 'output_tokens': 0}}})}",
        f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}})}",
        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hi'}})}",
        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '!'}})}",
        f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}",
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}})}",
        "",  # blank — skipped
        "non-sse line — should be ignored",  # not 'data:' prefixed
    ]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))

    provider = AnthropicProvider(api_key="sk-test")
    chunks = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="hi")], model="claude-x"):
        chunks.append(c)

    text = "".join(c.text or "" for c in chunks if c.type == "text_delta")
    assert text == "Hi!"
    assert any(c.type == "usage" and c.usage is not None and c.usage.output_tokens == 2 for c in chunks)
    assert any(c.type == "finish" and c.finish_reason == "end_turn" for c in chunks)


@pytest.mark.asyncio
async def test_anthropic_chat_stream_parses_tool_use_block():
    # Build SSE lines via explicit json.dumps payloads — avoids backslash
    # escapes inside f-strings, which Python 3.11 doesn't allow.
    events = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_accounts"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"limit":'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "5}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
    ]
    sse = ["data: " + json.dumps(e) for e in events]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))

    provider = AnthropicProvider(api_key="sk-test")
    chunks = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-x"):
        chunks.append(c)

    starts = [c for c in chunks if c.type == "tool_call_start"]
    args = [c for c in chunks if c.type == "tool_call_args_delta"]
    ends = [c for c in chunks if c.type == "tool_call_end"]
    assert starts and starts[0].tool_name == "list_accounts"
    assert "".join(c.args_delta or "" for c in args) == '{"limit":5}'
    assert ends and ends[0].tool_call_id == "toolu_1"


@pytest.mark.asyncio
async def test_anthropic_includes_system_and_tools_in_payload():
    sse = [f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {}})}"]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))
    provider = AnthropicProvider(api_key="sk-test")
    tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
    msgs = [
        ChatMessage(role="system", content="part1"),
        ChatMessage(role="system", content="part2"),
        ChatMessage(role="user", content="hi"),
    ]
    async for _ in provider.chat_stream(msgs, model="claude-x", tools=tools, max_tokens=200):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["system"] == "part1\n\npart2"
    assert payload["max_tokens"] == 200
    assert payload["tools"][0]["name"] == "t"


@pytest.mark.asyncio
async def test_anthropic_payload_omits_temperature():
    sse = [f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {}})}"]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))
    provider = AnthropicProvider(api_key="sk-test")
    msgs = [ChatMessage(role="user", content="hi")]
    async for _ in provider.chat_stream(msgs, model="claude-x", temperature=0.2):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert "temperature" not in payload


# --------------------------------------------------------------------- Anthropic: HTTP errors → typed exceptions

@pytest.mark.asyncio
@pytest.mark.parametrize("status,exc_type", [
    (401, LLMAuthError),
    (403, LLMAuthError),
    (429, LLMRateLimitError),
    (500, LLMUnavailableError),
    (502, LLMUnavailableError),
    (400, LLMUnavailableError),
])
async def test_anthropic_http_errors_map_to_typed_exceptions(status, exc_type):
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(
        status_code=status, lines=[], body=b'{"error":{"message":"x"}}',
    ))
    provider = AnthropicProvider(api_key="sk-test")
    with pytest.raises(exc_type):
        async for _ in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-x"):
            pass


@pytest.mark.asyncio
async def test_anthropic_embed_raises_not_supported():
    provider = AnthropicProvider(api_key="sk-test")
    with pytest.raises(LLMNotSupportedError):
        await provider.embed(["a"], model="x")


# --------------------------------------------------------------------- Anthropic: Claude 5 thinking models

def _sse(events: list[dict]) -> list[str]:
    return ["data: " + json.dumps(e) for e in events]


def _thinking_tool_turn() -> list[dict]:
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_accounts"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"limit": 5}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}},
    ]


@pytest.mark.asyncio
async def test_anthropic_thinking_model_payload():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse([
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ])))
    provider = AnthropicProvider(api_key="sk-test")
    async for _ in provider.chat_stream([ChatMessage(role="user", content="hi")], model="claude-opus-5-5"):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["max_tokens"] == 32000
    assert payload["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["fallbacks"] == "default"
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_anthropic_older_model_payload_unchanged():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse([
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ])))
    provider = AnthropicProvider(api_key="sk-test")
    async for _ in provider.chat_stream([ChatMessage(role="user", content="hi")], model="claude-opus-4-8"):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["max_tokens"] == 4096
    for key in ("thinking", "output_config", "fallbacks", "cache_control"):
        assert key not in payload


@pytest.mark.asyncio
async def test_anthropic_no_fallbacks_through_a_proxy_or_for_haiku():
    for model, base_url in (("claude-opus-5-5", "https://gateway.example.com/v1"), ("claude-haiku-5-5", None)):
        _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse([
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        ])))
        kwargs = {"base_url": base_url} if base_url else {}
        provider = AnthropicProvider(api_key="sk-test", **kwargs)
        async for _ in provider.chat_stream([ChatMessage(role="user", content="hi")], model=model):
            pass
        _, payload = _FakeAsyncClient.posted[-1]
        assert "fallbacks" not in payload
        assert payload["thinking"]["type"] == "adaptive"


@pytest.mark.asyncio
async def test_anthropic_thinking_blocks_replayed_verbatim_in_tool_loop():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse(_thinking_tool_turn())))
    provider = AnthropicProvider(api_key="sk-test")
    chunks = [c async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-opus-5-5")]

    raw = [c for c in chunks if c.type == "raw_content"]
    assert len(raw) == 1
    assert raw[0].raw_content == [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "tool_use", "id": "toolu_1", "name": "list_accounts", "input": {"limit": 5}},
    ]
    # raw_content arrives before finish so the executor can attach it.
    assert [c.type for c in chunks][-3:] == ["raw_content", "usage", "finish"]

    # Second request of the loop: assistant turn goes back unchanged, and
    # the results of parallel calls share one user turn.
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse([
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ])))
    history = [
        ChatMessage(role="user", content="x"),
        ChatMessage(role="assistant", content=None, raw_content=raw[0].raw_content),
        ChatMessage(role="tool", content="r1", tool_call_id="toolu_1"),
        ChatMessage(role="tool", content="r2", tool_call_id="toolu_2"),
    ]
    async for _ in provider.chat_stream(history, model="claude-opus-5-5"):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["messages"][1] == {"role": "assistant", "content": raw[0].raw_content}
    assert payload["messages"][2]["role"] == "user"
    assert [b["tool_use_id"] for b in payload["messages"][2]["content"]] == ["toolu_1", "toolu_2"]
    assert len(payload["messages"]) == 3


@pytest.mark.asyncio
async def test_anthropic_fallback_boundary_drops_earlier_reasoning():
    events = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "refused"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "fallback"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Antwort"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse(events)))
    provider = AnthropicProvider(api_key="sk-test")
    chunks = [c async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-opus-5-5")]
    raw = next(c for c in chunks if c.type == "raw_content")
    assert raw.raw_content == [{"type": "text", "text": "Antwort"}]


@pytest.mark.asyncio
async def test_anthropic_refusal_without_text_is_explained():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=_sse([
        {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}},
    ])))
    provider = AnthropicProvider(api_key="sk-test")
    chunks = [c async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-opus-5-5")]
    assert any(c.type == "text_delta" and c.text for c in chunks)
    assert chunks[-1].type == "finish" and chunks[-1].finish_reason == "refusal"
