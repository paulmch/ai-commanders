"""
End-of-battle cost report: calls, tokens and dollars per role and side.

Built from the recording's `model_call` events (live BattleEvent objects or
recorded dicts), so it works on old recordings too (role inferred from the
offered tools when the event has no `role`). Rows:

    side   alpha / beta
    role   draft / directive / order / captain / other (call_policy.call_role),
           plus "warmup" for cache pre-warms (cache_warmup mode), billed on
           their own and recorded on the call that waited for them
    calls, input (uncached), cache_read, cache_write, output, thinking
           tokens, usd, thinking_share (calls with any thinking tokens / calls)
"""
from typing import Any, Dict, Iterable, Optional, Tuple

from .call_policy import call_role, tool_names

FIELDS = ("calls", "input", "cache_read", "cache_write", "output", "thinking", "usd", "thinking_calls")


def _event_parts(event: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    if isinstance(event, dict):
        return event.get("event_type"), event.get("data") or {}
    return getattr(event, "event_type", None), getattr(event, "data", None) or {}


def usage_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
    """(input, cache_read, cache_write, output, thinking) from an Anthropic or OpenRouter usage block."""
    native = usage.get("anthropic") or {}
    if native or "input_tokens" in usage:
        src = native or usage
        return {"input": int(src.get("input_tokens") or 0),
                "cache_read": int(src.get("cache_read_input_tokens") or 0),
                "cache_write": int(src.get("cache_creation_input_tokens") or 0),
                "output": int(src.get("output_tokens") or 0),
                "thinking": int(src.get("thinking_tokens") or 0)}
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"input": max(0, int(usage.get("prompt_tokens") or 0) - cached - written),
            "cache_read": cached, "cache_write": written,
            "output": int(usage.get("completion_tokens") or 0), "thinking": int(reasoning)}


def cost_report(events: Iterable[Any], assets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{'rows': [...], 'by_side': {...}, 'total': {...}} from model_call events."""
    table: Dict[Tuple[str, str], Dict[str, float]] = {}

    def add(side: str, role: str, tokens: Dict[str, int], usd: float) -> None:
        row = table.setdefault((side, role), {k: 0 for k in FIELDS})
        row["calls"] += 1
        for key in ("input", "cache_read", "cache_write", "output", "thinking"):
            row[key] += tokens.get(key, 0)
        row["usd"] += usd
        row["thinking_calls"] += 1 if tokens.get("thinking", 0) > 0 else 0

    for event in events:
        kind, data = _event_parts(event)
        if kind != "model_call":
            continue
        side = data.get("faction") or "?"
        role = data.get("role")
        if not role:
            tools = (assets or {}).get(data.get("tools_ref")) if assets else None
            role = call_role(data.get("actor"), data.get("phase"), tool_names(tools))
        usage = data.get("usage") or {}
        add(side, role, usage_tokens(usage), float(usage.get("cost") or 0.0))
        warm = data.get("cache_warmup")
        if isinstance(warm, dict):
            add(side, "warmup", usage_tokens(warm.get("usage") or {}), float(warm.get("cost") or 0.0))

    def finish(row: Dict[str, float]) -> Dict[str, Any]:
        out = {k: (round(v, 6) if k == "usd" else int(v)) for k, v in row.items()}
        out["thinking_share"] = round(row["thinking_calls"] / row["calls"], 3) if row["calls"] else 0.0
        return out

    rows = [{"side": side, "role": role, **finish(row)} for (side, role), row in sorted(table.items())]
    by_side: Dict[str, Dict[str, float]] = {}
    total = {k: 0 for k in FIELDS}
    for (side, _), row in table.items():
        acc = by_side.setdefault(side, {k: 0 for k in FIELDS})
        for k in FIELDS:
            acc[k] += row[k]
            total[k] += row[k]
    return {"rows": rows, "by_side": {s: finish(r) for s, r in sorted(by_side.items())},
            "total": finish(total)}


def format_cost_report(report: Dict[str, Any]) -> str:
    """A fixed-width table for stdout."""
    head = (f"{'side':6s} {'role':10s} {'calls':>6s} {'input':>10s} {'cache rd':>10s} "
            f"{'cache wr':>10s} {'output':>9s} {'thinking':>9s} {'USD':>9s} {'think%':>7s}")
    lines = ["COST REPORT (per role and side, from model_call events)", head, "-" * len(head)]

    def line(side: str, role: str, r: Dict[str, Any]) -> str:
        return (f"{side:6s} {role:10s} {r['calls']:6d} {r['input']:10d} {r['cache_read']:10d} "
                f"{r['cache_write']:10d} {r['output']:9d} {r['thinking']:9d} {r['usd']:9.4f} "
                f"{100 * r['thinking_share']:6.0f}%")

    for row in report["rows"]:
        lines.append(line(row["side"], row["role"], row))
    lines.append("-" * len(head))
    for side, row in report["by_side"].items():
        lines.append(line(side, "ALL", row))
    lines.append(line("ALL", "ALL", report["total"]))
    return "\n".join(lines)
