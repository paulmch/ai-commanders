"""Experiment modes (all off by default), each proven offline on the mock Anthropic client."""
import json

import pytest

from src.llm.admiral import INTENT_ONLY_CAPTAIN_ORDER, INTENT_ONLY_NOTE
from src.llm.cost_report import cost_report, format_cost_report
from src.llm.mock_anthropic import message_response
from tests.test_anthropic_client import _battle

S55 = "claude-sonnet-5-5"


def _events(output):
    return json.loads(output.read_text())["events"]


def _calls(events):
    return [e["data"] for e in events if e["event_type"] == "model_call"]


def _tool_names(body):
    return {t["name"] for t in body.get("tools") or []}


# ---------------------------------------------------------------------------
# (b) no admiral, (c) LLM admiral over heuristic captains
# ---------------------------------------------------------------------------

def test_llm_captains_fight_without_an_admiral(tmp_path):
    record = []
    runner, client, output, data = _battle(tmp_path, S55, "claude-haiku-5-5", record=record,
                                           turns=3, admirals=False)
    result = runner.run_battle(data)
    calls = _calls(_events(output))
    assert result.checkpoints_used == 3
    assert calls and {d["role"] for d in calls} == {"captain"}
    assert not any("set_fleet_directive" in _tool_names(b) for b in record)
    assert all(not c.has_admiral for c in {**runner.alpha_captains, **runner.beta_captains}.values())


def test_llm_admiral_over_heuristic_captains_calls_only_the_admiral(tmp_path):
    record = []
    runner, client, output, data = _battle(tmp_path, S55, S55, record=record, turns=3,
                                           captain_models={"alpha": "heuristic", "beta": "heuristic"})
    runner.run_battle(data)
    events = _events(output)
    roles = {d["role"] for d in _calls(events)}
    assert roles == {"directive", "order"}
    assert any(e["event_type"] == "admiral_order" for e in events)
    # The rule-based captains still decide (and are recorded) every checkpoint.
    assert sum(1 for e in events if e["event_type"] == "captain_decision") == 3 * 4


# ---------------------------------------------------------------------------
# (d) intent-only admiral
# ---------------------------------------------------------------------------

def _directive_with_salvo(body):
    if "set_fleet_directive" not in _tool_names(body):
        return None
    content = [{"type": "thinking", "thinking": "", "signature": "s"},
               {"type": "tool_use", "id": "toolu_d", "name": "set_fleet_directive",
                "input": {"directive": "Hold range; corvettes launch on my mark."}},
               {"type": "tool_use", "id": "toolu_s", "name": "issue_order",
                "input": {"ship_name": "Dart", "torpedo_salvo": 2, "torpedo_target": "Sovereign",
                          "heading": {"x": 1, "y": 0, "z": 0}, "order_text": "ignored"}}]
    return message_response(content, "tool_use", model=body["model"])


def test_intent_only_admiral_issues_directive_and_salvos_but_no_per_ship_calls(tmp_path):
    from src.llm.fleet_config import AdmiralConfig
    record = []
    runner, client, output, data = _battle(tmp_path, S55, S55, record=record, turns=2,
                                           respond=_directive_with_salvo,
                                           ships=("battleship", "corvette"))
    runner.fleet_config.alpha_fleet.admiral.intent_only = True
    runner.run_battle(data)
    events = _events(output)
    alpha_roles = [d["role"] for d in _calls(events) if d["faction"] == "alpha"]
    beta_roles = [d["role"] for d in _calls(events) if d["faction"] == "beta"]
    assert "order" not in alpha_roles and alpha_roles.count("directive") == 2
    assert beta_roles.count("order") == 4                      # the other side is unchanged
    # The directive turn tells the admiral; the captains get the intent order.
    alpha_directives = [b for b in record if "set_fleet_directive" in _tool_names(b)
                        and any(INTENT_ONLY_NOTE[:40] in m["content"][0]["text"]
                                for m in b["messages"] if m["role"] == "user")]
    assert len(alpha_directives) == 2
    orders = [e["data"] for e in events if e["event_type"] == "admiral_order"
              and e["data"].get("faction", "alpha") == "alpha" and e["ship_id"].startswith("alpha")]
    assert orders and all(INTENT_ONLY_CAPTAIN_ORDER in (o.get("order_text") or "") for o in orders)
    # The salvo order survives; the heading in the same call does not.
    salvo = [o for o in orders if o.get("torpedo_salvo")]
    assert salvo and all(o["torpedo_salvo"] == 2 for o in salvo)
    assert all(not o.get("heading") for o in orders)
    assert AdmiralConfig(model="m").intent_only is False       # off by default


# ---------------------------------------------------------------------------
# (f) quiet-turn policy
# ---------------------------------------------------------------------------

def _positions(runner):
    return {sid: (round(s.position.x, 3), round(s.position.y, 3), round(s.position.z, 3))
            for sid, s in {**runner.alpha_ships, **runner.beta_ships}.items()}


def test_quiet_turns_skip_calls_record_why_and_fly_exactly_what_the_repeated_order_flies(tmp_path):
    kw = dict(turns=6, ships=("frigate", "frigate"), admirals=False)
    base, *_r1 = _battle(tmp_path / "a", S55, S55, **kw)
    base.run_battle(_r1[-1])
    quiet, client, output, data = _battle(tmp_path / "b", S55, S55, config={"quiet_turns": True}, **kw)
    quiet.run_battle(data)
    events = _events(output)
    skips = [e for e in events if e["event_type"] == "captain_turn_skipped"]
    assert skips, "no quiet turn happened - the test fleet must leave some turns quiet"
    assert all(e["data"]["reason"].startswith("quiet:") and e["data"]["renewed"] for e in skips)
    n_calls = sum(1 for d in _calls(events) if d["role"] == "captain")
    assert n_calls == 6 * 4 - len(skips)
    # The mock captain re-issues the same INTERCEPT every turn; a skipped turn
    # renews it instead. The ships fly identical trajectories.
    assert _positions(quiet) == _positions(base)


def test_quiet_turns_are_off_by_default(tmp_path):
    runner, client, output, data = _battle(tmp_path, S55, S55, turns=4, ships=("frigate", "frigate"),
                                           admirals=False)
    runner.run_battle(data)
    assert not any(e["event_type"] == "captain_turn_skipped" for e in _events(output))


# ---------------------------------------------------------------------------
# (e) cache warm-up
# ---------------------------------------------------------------------------

def test_cache_warmup_turns_concurrent_first_writes_into_reads(tmp_path):
    ships = ("battlecruiser",) * 3 + ("frigate",) * 2
    totals = {}
    for warm in (False, True):
        record = []
        runner, client, output, data = _battle(
            tmp_path / str(warm), S55, S55, record=record, turns=1, ships=ships,
            anthropic_extra={"cache_warmup": warm},
            transport_options={"latency_s": 0.1, "thinking_tokens": 300})
        runner.run_battle(data)
        report = cost_report(_events(output))
        captains = [r for r in report["rows"] if r["role"] == "captain"]
        totals[warm] = (report, sum(r["cache_write"] for r in captains), client, record)
    cold, warmed = totals[False], totals[True]
    assert cold[1] > 0 and warmed[1] == 0                     # captains only read with warm-up
    assert warmed[0]["total"]["cache_write"] < cold[0]["total"]["cache_write"] / 2
    assert warmed[0]["total"]["usd"] < cold[0]["total"]["usd"]
    prewarms = [b for b in warmed[3] if b["max_tokens"] == 0]
    assert prewarms and all(b["messages"] == [{"role": "user", "content": "warmup"}] for b in prewarms)
    assert not any(b["max_tokens"] == 0 for b in cold[3])    # off by default
    assert any(r["role"] == "warmup" for r in warmed[0]["rows"])
    # Warm-ups are billed: the client's own total equals the report's.
    assert warmed[2].stats.cost_usd == pytest.approx(warmed[0]["total"]["usd"], abs=1e-6)
    assert "warmup" in format_cost_report(warmed[0])


def test_cost_report_reads_old_recordings_without_role_fields():
    events = [{"event_type": "model_call", "data": {
        "actor": "admiral", "faction": "alpha", "tools_ref": "t",
        "usage": {"cost": 0.5, "anthropic": {"input_tokens": 10, "output_tokens": 5,
                                             "cache_read_input_tokens": 100,
                                             "cache_creation_input_tokens": 0, "thinking_tokens": 3}}}},
        {"event_type": "model_call", "data": {
            "actor": "captain", "faction": "beta", "tools_ref": "c",
            "usage": {"cost": 0.25, "prompt_tokens": 50, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10}}}}]
    assets = {"t": [{"function": {"name": "issue_order"}}], "c": [{"function": {"name": "set_maneuver"}}]}
    report = cost_report(events, assets)
    rows = {(r["side"], r["role"]): r for r in report["rows"]}
    assert rows[("alpha", "order")]["cache_read"] == 100 and rows[("alpha", "order")]["thinking_share"] == 1.0
    assert rows[("beta", "captain")]["input"] == 20 and rows[("beta", "captain")]["cache_write"] == 10
    assert report["total"]["usd"] == pytest.approx(0.75)
