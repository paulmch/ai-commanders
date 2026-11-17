"""
In-process stand-in for the Anthropic Messages API, for free tests and smoke runs.

`mock_anthropic_transport()` returns an httpx2 transport (the SDK's HTTP
layer) that answers `messages.create` requests with the same minimal, valid
tool calls `mock_openrouter` scripts for every call type of a battle, in
Anthropic's response shape (a thinking block with a signature, then tool_use
blocks). It also emulates the prompt cache: a request whose tools + system
prefix (up to the system cache breakpoint, with model and effort) was already
seen reports that prefix as `cache_read_input_tokens`, a new one as a cache
write split by TTL - so a prefix that silently changes between checkpoints
shows up as writes on every call, exactly as it would on the real API.

`http_client=anthropic.DefaultHttpxClient(transport=mock_anthropic_transport())`
on an AnthropicClient runs the whole paid code path (translation, effort,
cache markers, budget reservation and settlement, audit) with no network
call and no key.
"""
import itertools
import json
import threading
from typing import Any, Callable, Dict, List, Optional

import httpx2

from .mock_openrouter import _mock_calls


def _openai_view(body: Dict[str, Any]) -> Dict[str, Any]:
    tools = [{"function": {"name": t.get("name")}} for t in body.get("tools") or []]
    messages = []
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            content = [{"type": "text", "text": b.get("text", "")} for b in content
                       if isinstance(b, dict) and b.get("type") == "text"]
        messages.append({"role": m.get("role"), "content": content})
    return {"tools": tools, "messages": messages}


def _cached_prefix(body: Dict[str, Any]) -> Optional[str]:
    """Bytes of the cacheable prefix (tools + system up to the marker), or None."""
    system = body.get("system")
    if not isinstance(system, list):
        return None
    marked = [i for i, b in enumerate(system) if isinstance(b, dict) and b.get("cache_control")]
    if not marked:
        return None
    upto = [{k: v for k, v in b.items() if k != "cache_control"} for b in system[:marked[-1] + 1]]
    return json.dumps({"model": body.get("model"), "tools": body.get("tools"),
                       "system": upto, "thinking": body.get("thinking"),
                       "output_config": body.get("output_config")}, sort_keys=True)


def mock_anthropic_transport(
    points: int = 100,
    max_ships: int = 8,
    seed: int = 0,
    *,
    record: Optional[List[Dict[str, Any]]] = None,
    respond: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> httpx2.MockTransport:
    """
    record: every request body is appended here (tests inspect them).
    respond(body) -> a full response dict (or an httpx2.Response) to return
        instead of the scripted one, or None for the default - tests use it
        for refusals, max_tokens stops and error statuses.
    """
    seen: set = set()
    lock = threading.Lock()
    ids = itertools.count(1)

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if record is not None:
            with lock:
                record.append(body)
        if respond is not None:
            custom = respond(body)
            if isinstance(custom, httpx2.Response):
                return custom
            if custom is not None:
                return httpx2.Response(200, json=custom)

        prefix = _cached_prefix(body)
        total = max(1, len(json.dumps(body)) // 4)
        read = written = 0
        if prefix is not None:
            prefix_tokens = max(1, len(prefix) // 4)
            with lock:
                if prefix in seen:
                    read = prefix_tokens
                else:
                    written = prefix_tokens
                    seen.add(prefix)
        ttl_1h = any((b.get("cache_control") or {}).get("ttl") == "1h"
                     for b in body.get("system") or [] if isinstance(b, dict))

        calls = _mock_calls(_openai_view(body), points, max_ships, seed) if body.get("tools") else []
        with lock:
            serial = next(ids)
        content: List[Dict[str, Any]] = [
            {"type": "thinking", "thinking": "", "signature": f"mock-signature-{serial}"}]
        for n, (name, args) in enumerate(calls):
            content.append({"type": "tool_use", "id": f"toolu_mock_{serial}_{n}",
                            "name": name, "input": args})
        if not calls:
            content.append({"type": "text", "text": "Acknowledged."})
        output = 40 + 30 * len(calls)
        return httpx2.Response(200, headers={"request-id": f"req_mock_{serial}"}, json={
            "id": f"msg_mock_{serial}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "stop_details": None,
            "usage": {
                "input_tokens": max(1, total - read - written),
                "output_tokens": output,
                "cache_creation_input_tokens": written,
                "cache_read_input_tokens": read,
                "cache_creation": {"ephemeral_5m_input_tokens": 0 if ttl_1h else written,
                                   "ephemeral_1h_input_tokens": written if ttl_1h else 0},
                "output_tokens_details": {"thinking_tokens": 20},
            },
        })

    return httpx2.MockTransport(handle)


def message_response(content: List[Dict[str, Any]], stop_reason: str, model: str = "claude-haiku-5-5",
                     usage: Optional[Dict[str, Any]] = None,
                     stop_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A raw Messages API response body (for `respond=` overrides in tests)."""
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage or {"input_tokens": 100, "output_tokens": 20,
                               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
