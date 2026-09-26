"""
In-process stand-in for OpenRouter chat completions, for free smoke tests.

`mock_transport()` returns an httpx transport that answers CaptainClient
requests locally with minimal, valid tool calls for every call type the
battle makes (draft selection and formation, admiral directive and per-ship
orders, captain turns) and an OpenRouter-shaped `usage` block whose `cost` is
computed from catalog pricing. Swapping it into a real CaptainClient runs the
whole paid code path (payload building, reasoning, budget reservation with
images, settlement, audit) without a network call or an API key.
"""
import json
import math
import re
from collections import Counter
from typing import Any, Dict

import httpx

from .spending import image_prompt_tokens


def _last_user_text(messages) -> str:
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        return " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict))
    return ""


def _mock_calls(payload: Dict[str, Any], points: int, max_ships: int, seed: int):
    names = {t.get("function", {}).get("name") for t in payload.get("tools") or []}
    text = _last_user_text(payload.get("messages"))
    if "select_fleet" in names:
        from .fleet_draft import auto_draft
        picks = Counter(s.ship_type for s in auto_draft("alpha", points, max_ships, seed=seed).ships)
        return [("select_fleet", {"ships": [{"ship_type": t, "count": n} for t, n in picks.items()],
                                  "rationale": "Mock draft: the automatic fleet."})]
    if "set_formation" in names:
        return [("set_formation", {"formation_name": "mock default line", "placements": []})]
    if "set_fleet_directive" in names:
        return [("set_fleet_directive", {"directive": "MOCK: close to gun range, focus the nearest "
                                                      "enemy, evade incoming torpedoes."})]
    if "issue_order" in names:
        match = re.search(r"Issue tactical order to (.+?)\. You MUST", text)
        ship = match.group(1) if match else "unknown"
        return [("issue_order", {"ship_name": ship, "priority": "NORMAL",
                                 "order_text": "Intercept the nearest enemy and fire when optimal."})]
    if "set_maneuver" in names:
        calls = [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.4}),
                 ("set_weapons_order", {"spinal_mode": "FIRE_WHEN_OPTIMAL",
                                        "turret_mode": "FIRE_WHEN_OPTIMAL"})]
        if "launch_torpedo" in names:
            calls.append(("launch_torpedo", {"count": 1}))
        return [c for c in calls if c[0] in names]
    return []


def mock_transport(pricing: Dict[str, Dict[str, Any]], points: int = 100,
                   max_ships: int = 8, seed: int = 0) -> httpx.MockTransport:
    """httpx transport answering chat completions locally (see module docstring)."""
    counter = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        counter["n"] += 1
        images = []
        text_bytes = 0
        for message in payload.get("messages") or []:
            content = message.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for part in parts:
                if part.get("type") == "image_url":
                    images.append(part["image_url"]["url"])
                else:
                    text_bytes += len(str(part.get("text", "")))
        text_bytes += len(json.dumps(payload.get("tools") or []))
        prompt_tokens = math.ceil(text_bytes / 3.8) + sum(
            image_prompt_tokens(url, payload["model"]) for url in images)
        completion_tokens = 400
        rates = pricing.get(payload["model"], {})
        cost = (prompt_tokens * float(rates.get("prompt", 0))
                + completion_tokens * float(rates.get("completion", 0)))
        calls = _mock_calls(payload, points, max_ships, seed)
        message: Dict[str, Any] = {"role": "assistant", "content": None if calls else "Acknowledged."}
        if calls:
            message["tool_calls"] = [
                {"id": f"mock-{counter['n']}-{i}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)]
        return httpx.Response(200, json={
            "id": f"gen-mock-{counter['n']}",
            "model": payload["model"],
            "choices": [{"message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        })

    return httpx.MockTransport(handle)
