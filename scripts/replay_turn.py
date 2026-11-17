#!/usr/bin/env python3
"""
Re-send recorded model calls to another model / effort, N times each.

Every `model_call` event in a recording keeps the call's input verbatim
(`message_refs` and `tools_ref` into the recording's `assets`). This script
selects recorded calls and sends the identical system + user messages + tools
again - to a chosen model at a chosen effort, optionally with a prompt patch -
so a model, an effort level or a proposed prompt line can be compared with
what the recorded model did on IDENTICAL battle state, for cents per turn
instead of a whole battle.

    # what would be sent, and what it would cost (no network)
    uv run python scripts/replay_turn.py data/recordings/p5_sonnet_battles_20261116/p5_sonnet5_vs_sonnet55_d \\
        --role captain --checkpoint 8 --variant claude-sonnet-5-5:high -n 3 --dry-run

    # the same offline against the mock client (how the script is tested)
    uv run python scripts/replay_turn.py <recording> --role captain --checkpoint 8 \\
        --variant claude-sonnet-5-5:medium --variant claude-sonnet-5-5:high -n 3 --mock

    # A/B a prompt line on identical state, paid, hard cap $2
    uv run python scripts/replay_turn.py <recording> --select turns.json \\
        --variant claude-sonnet-5-5:medium --variant claude-sonnet-5-5:medium:patch.json \\
        -n 5 --spend-limit 2 --out data/recordings/replays/ab_line

Variants: `--variant MODEL[:EFFORT[:PATCH_FILE]]` (repeatable). EFFORT omitted or
"rec" = the effort the recorded call was sent at. Without --variant, one variant
is built from --model / --effort / --patch.

Patch file (JSON): a list of edits applied to the message text before sending,
    [{"target": "user", "find": "coast & shoot", "replace": "coast & fire"},
     {"target": "system", "find": "^REPLY:.*$", "replace": "", "regex": true}]
target: "user" (default), "system" or "both"; regex: re.MULTILINE pattern.
An edit that matches in none of the selected calls is an error (the A/B would
silently compare two identical prompts). A system-message edit changes the
cached prefix: the dry run shows the extra cache writes it costs.

Output: PREFIX.jsonl (one row per reply: source call, model, effort,
replicate, tool calls with arguments, text, thinking summary, usage, cost,
latency) and PREFIX.md (per source call: what the recorded reply did, what each
variant did, how many replicates agree). Prompts are never printed in full.

Only bare Anthropic model ids (claude-...) are supported: the dollar cap
prices them from the client's own table.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.anthropic_client import (  # noqa: E402
    EFFORT_LEVELS, cost_usd, is_anthropic_model, model_spec, reservation_usd,
)
from src.llm.call_policy import ROLES, call_role, tool_names  # noqa: E402
from src.llm.client import LLMCallError  # noqa: E402
from src.recording_io import load_recording  # noqa: E402

ROLE_ALIASES = {"admiral-directive": "directive", "admiral_directive": "directive",
                "admiral-order": "order", "admiral_order": "order"}
ORDER_SUBJECT = re.compile(r"Issue tactical order to (.+?)\. You MUST")
# Free-text arguments: excluded from the action signature (agreement is about
# what the ship / fleet DOES, not how it is phrased).
FREE_TEXT_ARGS = {"order_text", "reason", "message", "directive", "plan", "note", "question",
                  "rationale", "text", "content", "summary", "answer", "details", "say"}


# -----------------------------------------------------------------------------
# Loading recorded calls
# -----------------------------------------------------------------------------

def recording_files(path: Path) -> List[Path]:
    """A recording file, or every battle recording + draft-call file in a folder."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("battle_*.json.gz")) + sorted(path.glob("battle_*.json"))
        files = [f for f in files if not f.name.endswith(".draft.json")]
        if not files:
            # A live-draft report folder without the battle (aborted draft):
            # the draft calls are all there is.
            files = sorted(path.glob("*_draft_calls.json"))
        return files
    raise FileNotFoundError(f"No recording at {path}")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or []
                     if isinstance(p, dict) and p.get("type") == "text")


def load_calls(paths: List[Path]) -> List[Dict[str, Any]]:
    """Every recorded model_call as a self-contained source-call dict."""
    calls: List[Dict[str, Any]] = []
    for path in paths:
        recording = load_recording(path)
        assets = recording.get("assets") or {}
        interval = float(recording.get("decision_interval_s")
                         or (recording.get("provenance") or {}).get("decision_interval_s") or 30.0)
        name = path.parent.name if path.name.startswith(("battle_", "alpha_", "beta_")) else path.stem
        for event in recording.get("events") or []:
            if event.get("event_type") != "model_call":
                continue
            data = event.get("data") or {}
            refs = data.get("message_refs") or []
            if not refs or any(r not in assets for r in refs):
                continue
            messages = [assets[r] for r in refs]
            tools = assets.get(data.get("tools_ref"), []) or []
            names = tool_names(tools)
            role = data.get("role") or call_role(data.get("actor"), data.get("phase"), names)
            user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            subject = None
            if role == "order" and user is not None:
                match = ORDER_SUBJECT.search(_content_text(user.get("content")))
                subject = match.group(1) if match else None
            request = data.get("request") or {}
            t = float(event.get("timestamp") or 0.0)
            calls.append({
                "recording": name, "path": str(path), "event_id": event.get("event_id"),
                "t": t, "checkpoint": int(round(t / interval)) if interval else None,
                "actor": data.get("actor"), "faction": data.get("faction"),
                "ship_id": event.get("ship_id"), "subject": subject, "role": role,
                "model": data.get("model"),
                "effort": (request.get("output_config") or {}).get("effort"),
                "max_tokens": request.get("max_tokens"),
                "messages": messages, "tools": tools, "tool_names": names,
                "usage": data.get("usage") or {}, "output": data.get("output") or {},
            })
    return calls


def call_key(call: Dict[str, Any]) -> str:
    return f"{call['recording']}:{call['event_id']}"


def select_calls(calls: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = calls
    if args.select:
        wanted = json.loads(Path(args.select).read_text())
        entries = wanted.get("calls", wanted) if isinstance(wanted, dict) else wanted
        keys = set()
        for entry in entries:
            if isinstance(entry, str):
                keys.add(entry)
            else:
                rec = Path(str(entry.get("recording", ""))).name.replace(".json.gz", "")
                keys.add(f"{rec}:{entry['event_id']}")
        by_key = {call_key(c): c for c in calls}
        by_id = {}
        for c in calls:
            by_id.setdefault(c["event_id"], []).append(c)
        picked = []
        for key in keys:
            if key in by_key:
                picked.append(by_key[key])
            else:
                eid = key.split(":")[-1]
                picked.extend(by_id.get(eid, []) if ":" not in key else [])
        missing = len(keys) - len(picked)
        if missing > 0:
            raise SystemExit(f"--select: {missing} of {len(keys)} calls are not in the recordings")
        out = picked
    if args.role:
        roles = {ROLE_ALIASES.get(r, r) for r in args.role}
        out = [c for c in out if c["role"] in roles]
    if args.faction:
        out = [c for c in out if c["faction"] == args.faction]
    if args.ship:
        needles = [s.lower() for s in args.ship]
        out = [c for c in out if any(
            n == str(c["ship_id"] or "").lower() or n in str(c["subject"] or "").lower()
            for n in needles)]
    if args.checkpoint:
        out = [c for c in out if c["checkpoint"] in set(args.checkpoint)]
    if args.t:
        out = [c for c in out if any(abs(c["t"] - t) < 0.5 for t in args.t)]
    if args.event:
        wanted = set(args.event)
        out = [c for c in out if c["event_id"] in wanted or call_key(c) in wanted]
    if args.model_recorded:
        out = [c for c in out if c["model"] == args.model_recorded]
    out = sorted(out, key=lambda c: (c["recording"], c["t"], str(c["faction"]), str(c["ship_id"] or c["subject"]),
                                     int(str(c["event_id"]).lstrip("e") or 0)))
    if args.sample and len(out) > args.sample:
        out = sorted(random.Random(args.seed).sample(out, args.sample),
                     key=lambda c: (c["recording"], c["t"]))
    if args.limit:
        out = out[:args.limit]
    return out


# -----------------------------------------------------------------------------
# Patches
# -----------------------------------------------------------------------------

def load_patch(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    edits = json.loads(Path(path).read_text())
    if isinstance(edits, dict):
        edits = edits.get("edits", [])
    out = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict) or "find" not in edit:
            raise SystemExit(f"{path}: edit {i} needs a 'find' (and 'replace')")
        target = edit.get("target", "user")
        if target not in ("user", "system", "both"):
            raise SystemExit(f"{path}: edit {i} target must be user/system/both")
        regex = bool(edit.get("regex", False))
        pattern = re.compile(edit["find"] if regex else re.escape(edit["find"]), re.MULTILINE)
        out.append({"target": target, "pattern": pattern, "replace": str(edit.get("replace", "")),
                    "regex": regex, "find": edit["find"], "count": int(edit.get("count", 0))})
    return out


def _sub_text(text: str, edit: Dict[str, Any]) -> Tuple[str, int]:
    repl = edit["replace"] if edit["regex"] else edit["replace"].replace("\\", "\\\\")
    return edit["pattern"].subn(repl, text, count=edit["count"])


def apply_patch(messages: List[Dict[str, Any]], edits: List[Dict[str, Any]]
                ) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Patched copy of the messages and the number of matches per edit."""
    counts = [0] * len(edits)
    if not edits:
        return messages, counts
    out = []
    for message in messages:
        role = message.get("role")
        message = json.loads(json.dumps(message))
        for i, edit in enumerate(edits):
            if edit["target"] != "both" and edit["target"] != role:
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"], n = _sub_text(content, edit)
                counts[i] += n
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"], n = _sub_text(part.get("text", ""), edit)
                        counts[i] += n
        out.append(message)
    return out, counts


# -----------------------------------------------------------------------------
# Variants, jobs, estimates
# -----------------------------------------------------------------------------

def parse_variant(spec: str) -> Dict[str, Any]:
    parts = spec.split(":", 2)
    model = parts[0]
    effort = parts[1] if len(parts) > 1 and parts[1] not in ("", "rec", "recorded") else None
    patch = parts[2] if len(parts) > 2 and parts[2] else None
    if not is_anthropic_model(model):
        raise SystemExit(f"--variant {spec}: only bare Anthropic ids (claude-...) are supported")
    model_spec(model)   # unknown ids raise before anything is sent
    if effort is not None and effort not in EFFORT_LEVELS:
        raise SystemExit(f"--variant {spec}: effort must be one of {', '.join(EFFORT_LEVELS)} or 'rec'")
    label = f"{model}@{effort or 'rec'}" + (f"+{Path(patch).stem}" if patch else "")
    return {"label": label, "model": model, "effort": effort, "patch_file": patch,
            "edits": load_patch(patch)}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_jobs(calls, variants, replicates) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One job per (source call, variant, replicate), plus patch match statistics."""
    jobs = []
    patch_stats: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        totals = [0] * len(variant["edits"])
        unmatched: List[str] = []
        for call in calls:
            messages, counts = apply_patch(call["messages"], variant["edits"])
            totals = [a + b for a, b in zip(totals, counts)]
            if variant["edits"] and not all(counts):
                unmatched.append(call_key(call))
            effort = variant["effort"] or call["effort"] or "medium"
            system = [m for m in messages if m.get("role") == "system"]
            prefix = _digest({"model": variant["model"], "effort": effort,
                              "tools": call["tools"], "system": system})
            for rep in range(replicates):
                jobs.append({"call": call, "variant": variant, "replicate": rep + 1,
                             "messages": messages, "effort": effort, "prefix": prefix,
                             "max_tokens": call["max_tokens"]})
        if variant["edits"]:
            patch_stats[variant["label"]] = {
                "matches_per_edit": totals, "calls_with_a_missing_edit": unmatched,
                "edits": [e["find"][:60] for e in variant["edits"]]}
    return jobs, patch_stats


def _tokens(usage: Dict[str, Any]) -> Tuple[int, int, int]:
    """(uncached input, cacheable prefix, output) from a recorded usage block."""
    native = usage.get("anthropic") or {}
    if native:
        uncached = int(native.get("input_tokens") or 0)
        prefix = int(native.get("cache_read_input_tokens") or 0) + int(native.get("cache_creation_input_tokens") or 0)
        output = int(native.get("output_tokens") or 0)
        return uncached, prefix, output
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    prefix = int(details.get("cached_tokens") or 0) + int(details.get("cache_write_tokens") or 0)
    return prompt - prefix, prefix, int(usage.get("completion_tokens") or 0)


def estimate(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expected dollars from the RECORDED token counts: the first job of each
    cached prefix writes it (1-hour TTL, as the battle client does), the rest
    read it; output = the recorded reply's output tokens (thinking included) -
    a different effort changes that, so treat it as an estimate. The upper
    bound is the sum of the cap reservations.
    """
    seen = set()
    expected = 0.0
    bound = 0.0
    for job in jobs:
        uncached, prefix, output = _tokens(job["call"]["usage"])
        first = job["prefix"] not in seen
        seen.add(job["prefix"])
        usage = {"input_tokens": uncached, "output_tokens": output,
                 "cache_creation_input_tokens": prefix if first else 0,
                 "cache_read_input_tokens": 0 if first else prefix,
                 "cache_creation": {"ephemeral_1h_input_tokens": prefix if first else 0}}
        expected += cost_usd(job["variant"]["model"], usage)
        bound += reservation_usd(job["variant"]["model"],
                                 {"messages": job["messages"], "tools": job["call"]["tools"]},
                                 int(job["max_tokens"] or 8192))[0]
    return {"jobs": len(jobs), "prefixes": len(seen), "expected_usd": round(expected, 4),
            "reservation_bound_usd": round(bound, 4)}


# -----------------------------------------------------------------------------
# Signatures and the comparison table
# -----------------------------------------------------------------------------

def _norm(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value if len(value) <= 40 else None
    if isinstance(value, dict):
        inner = {k: _norm(v) for k, v in sorted(value.items()) if k not in FREE_TEXT_ARGS}
        return {k: v for k, v in inner.items() if v is not None}
    if isinstance(value, list):
        return [_norm(v) for v in value[:8]]
    return None


def _call_args(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if "function" in call:
        fn = call["function"]
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {"_unparsed": True}
        return fn.get("name", "?"), args
    return call.get("name", "?"), call.get("args") or call.get("arguments") or {}


def action_signature(tool_calls: List[Dict[str, Any]]) -> str:
    """What a reply DID: tool names plus their discrete arguments, order-free."""
    parts = []
    for call in tool_calls or []:
        name, args = _call_args(call)
        norm = _norm(args) if isinstance(args, dict) else {}
        inner = ",".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in norm.items())
        parts.append(f"{name}({inner})")
    return " ; ".join(sorted(parts)) or "(no tool call)"


def comparison_table(calls, variants, rows) -> str:
    by_call: Dict[str, Dict[str, List[Dict[str, Any]]]] = OrderedDict()
    for call in calls:
        by_call[call_key(call)] = {v["label"]: [] for v in variants}
    for row in rows:
        by_call.setdefault(row["source"]["key"], {}).setdefault(row["variant"]["label"], []).append(row)
    lines = ["# Replay comparison", "",
             "Signature = tool names + discrete arguments (free text excluded). "
             "`agree` = replicates whose signature equals the variant's most common one; "
             "`=rec` = replicates matching the recorded reply.", ""]
    for call in calls:
        key = call_key(call)
        who = call["ship_id"] or call["subject"] or call["faction"]
        recorded = action_signature(call["output"].get("tool_calls") or [])
        lines.append(f"## {key} T+{call['t']:.0f} cp{call['checkpoint']} {call['role']} {who} "
                     f"(recorded {call['model']} @ {call['effort']})")
        lines.append(f"- recorded: `{recorded[:300]}`")
        for variant in variants:
            replies = by_call.get(key, {}).get(variant["label"], [])
            ok = [r for r in replies if not r.get("error")]
            sigs = Counter(r["signature"] for r in ok)
            errors = len(replies) - len(ok)
            if not sigs:
                lines.append(f"- {variant['label']}: no reply" + (f" ({errors} errors)" if errors else ""))
                continue
            top, n_top = sigs.most_common(1)[0]
            same = sum(1 for r in ok if r["signature"] == recorded)
            cost = sum(r.get("cost") or 0 for r in replies)
            think = [r.get("thinking_tokens") or 0 for r in ok]
            lines.append(f"- {variant['label']}: agree {n_top}/{len(ok)}, =rec {same}/{len(ok)}, "
                         f"{len(sigs)} distinct, thinking tokens {sorted(think)}, ${cost:.4f}"
                         + (f", {errors} errors" if errors else ""))
            for sig, n in sigs.most_common(3):
                lines.append(f"  - {n}x `{sig[:300]}`")
        lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Sending
# -----------------------------------------------------------------------------

def make_client(variants, args):
    from src.llm.providers import create_llm_client
    models = sorted({v["model"] for v in variants})
    options: Dict[str, Any] = {"max_retries": 2}
    if args.mock:
        import anthropic
        from src.llm.mock_anthropic import mock_anthropic_transport
        options.update(api_key="mock-no-network", http_client=anthropic.DefaultHttpxClient(
            transport=mock_anthropic_transport(seed=args.seed, latency_s=args.mock_latency)))
    if args.thinking_summary:
        options["thinking_display"] = "summarized"
    client = create_llm_client(models[0], models, temperature=None, max_tokens=16000,
                               reasoning_effort=None, timeout=300, anthropic_options=options)
    return client


def send(client, job: Dict[str, Any]) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    def observe(payload, data):
        captured["data"] = data

    call = job["call"]
    started = time.monotonic()
    error = None
    try:
        with client.audit_scope(observe):
            if call["tools"]:
                client.decide_with_tools(job["messages"], call["tools"], model=job["variant"]["model"],
                                         tool_choice="auto", max_tokens=job["max_tokens"],
                                         reasoning_effort=job["effort"])
            else:
                client.complete(job["messages"], model=job["variant"]["model"])
    except LLMCallError as e:
        error = str(e)[:400]
    latency = round(time.monotonic() - started, 2)
    data = captured.get("data") or {}
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    tool_calls = [{"name": n, "args": a} for n, a in
                  (_call_args(c) for c in message.get("tool_calls") or [])]
    usage = data.get("usage") or {}
    native = usage.get("anthropic") or {}
    return {
        "source": {"key": call_key(call), "recording": call["recording"], "event_id": call["event_id"],
                   "t": call["t"], "checkpoint": call["checkpoint"], "role": call["role"],
                   "faction": call["faction"], "ship_id": call["ship_id"], "subject": call["subject"],
                   "model": call["model"], "effort": call["effort"],
                   "signature": action_signature(call["output"].get("tool_calls") or [])},
        "variant": {"label": job["variant"]["label"], "model": job["variant"]["model"],
                    "effort": job["effort"], "patch": job["variant"]["patch_file"]},
        "replicate": job["replicate"],
        "tool_calls": tool_calls,
        "signature": action_signature(tool_calls) if not error else None,
        "text": (message.get("content") or "")[:4000] or None,
        "thinking_summary": data.get("thinking_summary"),
        "thinking_tokens": native.get("thinking_tokens"),
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
        "stop_reason": data.get("stop_reason"),
        "usage": native or usage,
        "cost": usage.get("cost"),
        "latency_s": latency,
        "error": error,
    }


def run_jobs(client, jobs, workers) -> List[Dict[str, Any]]:
    """
    Two waves: the first job of every distinct cached prefix, then the rest.
    Concurrent requests on one prefix would each write the cache (an entry is
    readable only once the first response has started); this way every
    prefix is written once and read by the other replicates.
    """
    first, rest, seen = [], [], set()
    for job in jobs:
        (rest if job["prefix"] in seen else first).append(job)
        seen.add(job["prefix"])
    rows: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def go(job):
        row = send(client, job)
        with lock:
            rows.append(row)
            done = len(rows)
        tag = "ERROR " + row["error"][:80] if row["error"] else row["signature"][:90]
        print(f"[{done}/{len(jobs)}] {row['source']['key']} {row['variant']['label']} "
              f"#{row['replicate']} ${row['cost'] or 0:.4f} {row['latency_s']}s  {tag}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(go, first))
        list(pool.map(go, rest))
    rows.sort(key=lambda r: (r["source"]["key"], r["variant"]["label"], r["replicate"]))
    return rows


# -----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", help="recording files or folders")
    sel = p.add_argument_group("selection")
    sel.add_argument("--role", action="append",
                     choices=list(ROLES) + sorted(ROLE_ALIASES), help="repeatable")
    sel.add_argument("--faction", choices=("alpha", "beta"))
    sel.add_argument("--ship", action="append", help="ship id, or part of the ordered ship's name (admiral orders)")
    sel.add_argument("--checkpoint", action="append", type=int, help="checkpoint number (t / decision interval)")
    sel.add_argument("--t", action="append", type=float, help="sim time of the call, seconds")
    sel.add_argument("--event", action="append", help="event id (e123) or RECORDING:e123")
    sel.add_argument("--model-recorded", help="only calls the recording sent to this model")
    sel.add_argument("--select", help="named selection file: JSON list of 'recording:event_id' or "
                                      "{recording, event_id} (or {'calls': [...]})")
    sel.add_argument("--sample", type=int, help="random sample of N selected calls (--seed)")
    sel.add_argument("--limit", type=int, help="at most N source calls")
    sel.add_argument("--save-selection", help="write the selected call keys to this JSON file")
    var = p.add_argument_group("variants")
    var.add_argument("--variant", action="append", help="MODEL[:EFFORT[:PATCH_FILE]], repeatable")
    var.add_argument("--model", help="single variant: model (default: the recorded model)")
    var.add_argument("--effort", choices=EFFORT_LEVELS, help="single variant: effort (default: recorded)")
    var.add_argument("--patch", help="single variant: patch file")
    var.add_argument("-n", "--replicates", type=int, default=1)
    var.add_argument("--thinking-summary", action="store_true",
                     help="request thinking display 'summarized' and store the summary per reply "
                          "(billed the same as the default omitted display)")
    run = p.add_argument_group("running")
    run.add_argument("--workers", type=int, default=4, help="concurrent requests (default 4)")
    run.add_argument("--spend-limit", type=float, help="hard dollar cap for this run (required when paid)")
    run.add_argument("--out", help="output prefix (default data/recordings/replays/replay_<UTC stamp>)")
    run.add_argument("--mock", action="store_true", help="offline: the mock Anthropic client answers")
    run.add_argument("--mock-latency", type=float, default=0.0, help=argparse.SUPPRESS)
    run.add_argument("--dry-run", action="store_true", help="print what would be sent and its cost; send nothing")
    run.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.replicates < 1:
        p.error("-n must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [f for r in args.recordings for f in recording_files(Path(r))]
    calls = select_calls(load_calls(paths), args)
    if not calls:
        print("No recorded calls match the selection.", file=sys.stderr)
        return 2
    if args.variant:
        variants = [parse_variant(v) for v in args.variant]
    else:
        model = args.model or calls[0]["model"]
        variants = [parse_variant(f"{model}:{args.effort or 'rec'}:{args.patch or ''}")]
    labels = [v["label"] for v in variants]
    if len(set(labels)) != len(labels):
        raise SystemExit("Two variants have the same model, effort and patch file")
    jobs, patch_stats = build_jobs(calls, variants, args.replicates)
    for label, stats in patch_stats.items():
        for find, n in zip(stats["edits"], stats["matches_per_edit"]):
            print(f"[patch] {label}: {n} matches for {find!r}")
            if n == 0:
                raise SystemExit(f"Patch edit {find!r} ({label}) matches nothing in the selected calls")
        if stats["calls_with_a_missing_edit"]:
            print(f"[patch] {label}: {len(stats['calls_with_a_missing_edit'])} calls lack at least one "
                  f"match: {stats['calls_with_a_missing_edit'][:5]}")
    roles = Counter(c["role"] for c in calls)
    print(f"Selected {len(calls)} recorded calls ({dict(roles)}) from {len(paths)} file(s); "
          f"{len(variants)} variant(s) x {args.replicates} replicate(s) = {len(jobs)} requests")
    est = estimate(jobs)
    print(f"Estimate (recorded token counts): ${est['expected_usd']:.4f} expected, "
          f"{est['prefixes']} cache prefixes written; cap reservations would total "
          f"${est['reservation_bound_usd']:.4f}")
    if args.save_selection:
        Path(args.save_selection).write_text(json.dumps([call_key(c) for c in calls], indent=1))
    if args.dry_run:
        for call in calls[:40]:
            user = sum(len(_content_text(m.get("content"))) for m in call["messages"] if m.get("role") == "user")
            system = sum(len(_content_text(m.get("content"))) for m in call["messages"] if m.get("role") == "system")
            print(f"  {call_key(call)} T+{call['t']:.0f} {call['role']:9s} "
                  f"{(call['ship_id'] or call['subject'] or call['faction'])!s:22s} "
                  f"rec {call['model']}@{call['effort']} system {system} ch, user {user} ch, "
                  f"{len(call['tool_names'])} tools, max_tokens {call['max_tokens']}")
        if len(calls) > 40:
            print(f"  ... {len(calls) - 40} more")
        for variant in variants:
            prefixes = {j["prefix"] for j in jobs if j["variant"] is variant}
            print(f"  variant {variant['label']}: {len(prefixes)} distinct cached prefixes")
        return 0

    from src.llm.spending import RequestBudget
    paid = not args.mock
    if paid and not args.spend_limit:
        raise SystemExit("A paid replay needs --spend-limit USD (a hard cap)")
    limit = args.spend_limit or max(1.0, est["reservation_bound_usd"] * 2)
    out = Path(args.out) if args.out else ROOT / "data/recordings/replays" / (
        "replay_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    client = make_client(variants, args)
    client.budget = RequestBudget(limit, {})
    rows: List[Dict[str, Any]] = []
    try:
        rows = run_jobs(client, jobs, args.workers)
    finally:
        with open(f"{out}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        Path(f"{out}.md").write_text(comparison_table(calls, variants, rows))
        spent = client.budget.reported_usd
    errors = sum(1 for r in rows if r.get("error"))
    print(f"\n{len(rows)} replies ({errors} errors), ${spent:.4f} spent of the ${limit:.2f} cap"
          + (" (EXHAUSTED)" if client.budget.exhausted else ""))
    print(f"Wrote {out}.jsonl and {out}.md")
    return 0 if not client.budget.exhausted else 3


def _token_total(rows) -> Dict[str, int]:
    total: Dict[str, int] = Counter()
    for row in rows:
        usage = row.get("usage") or {}
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
            total[key] += int(usage.get(key) or 0)
    return dict(total)


if __name__ == "__main__":
    sys.exit(main())
