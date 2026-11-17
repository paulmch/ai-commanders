"""
Per-call policy for LLM commanders: which ROLE a call plays, and (for
experiments) which effort it is sent at.

Roles. A battle's model calls play five roles, told apart from what the
recording or the request already carries (old recordings have only `actor`,
the draft `phase` and the tool list):

    draft       the pre-battle fleet purchase (select_fleet)
    directive   the admiral's Phase-1 call (set_fleet_directive offered)
    order       the admiral's per-ship Phase-2 call (only issue_order offered)
    captain     a captain's decision call
    other       anything else: personality selection, an admiral's free-text
                answer to discuss_with_admiral, notebook distillation

`call_role()` is the one classifier; the replay harness, the cost report and
the effort policy all use it.
"""
from typing import Any, Iterable, Optional

ROLES = ("draft", "directive", "order", "captain", "other")


def tool_names(tools: Optional[Iterable[Any]]) -> list:
    """Names of OpenAI-style function tools (or Anthropic-shaped tools)."""
    names = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(name)
    return names


def call_role(actor: Optional[str], phase: Optional[str] = None,
              names: Optional[Iterable[str]] = None) -> str:
    """
    The role of one model call (see the module docstring).

    actor: 'captain' / 'admiral' (model_call.actor), phase: model_call.phase
    ('draft' for calls merged from the draft recorders), names: the tool names
    offered to the call.
    """
    names = set(names or ())
    if phase == "draft" or "select_fleet" in names:
        return "draft"
    if actor == "captain":
        return "captain" if names else "other"
    if actor == "admiral":
        if "set_fleet_directive" in names:
            return "directive"
        if names == {"issue_order"}:
            return "order"
        return "other"
    return "other"
