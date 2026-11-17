"""Per-role / per-side effort, captain triggers and the effort wrapper (src/llm/call_policy.py)."""
import hashlib
import json

import pytest

from src.llm.call_policy import (
    EffortPolicy, PolicyClient, TurnFacts, captain_triggers, order_kind, order_signature,
    parse_effort_specs, quiet_turn_reason,
)
from tests.test_anthropic_client import _battle


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

def test_effort_specs_parse_per_side_and_role_and_reject_junk():
    assert parse_effort_specs(["directive=high", "alpha.captain=low", "beta.trigger=xhigh",
                               "alpha.admiral-order=medium"]) == {
        "alpha": {"directive": "high", "captain": "low", "order": "medium"},
        "beta": {"directive": "high", "trigger": "xhigh"}}
    assert parse_effort_specs(["captain=low", "alpha.captain=high"])["alpha"]["captain"] == "high"
    for bad in ("captain", "gamma.captain=low", "alpha.pilot=low", "captain=extreme"):
        with pytest.raises(ValueError):
            parse_effort_specs([bad])


def test_triggers_each_fire_alone_and_a_quiet_turn_has_none():
    assert captain_triggers(TurnFacts()) == []
    assert captain_triggers(TurnFacts(first_turn=True)) == ["first"]
    assert captain_triggers(TurnFacts(torpedoes_inbound=2)) == ["torpedo"]
    assert captain_triggers(TurnFacts(merge_warning=True)) == ["merge"]
    assert captain_triggers(TurnFacts(merge_warning=True, order_moves_ship=True)) == ["merge", "conflict"]
    assert captain_triggers(TurnFacts(order_moves_ship=True)) == []
    assert captain_triggers(TurnFacts(damage_taken=True)) == ["damage"]
    assert captain_triggers(TurnFacts(order_kind="PADLOCK", previous_order_kind="HEADING")) == ["new_order"]
    assert captain_triggers(TurnFacts(order_kind="HEADING", previous_order_kind="HEADING")) == []
    assert captain_triggers(TurnFacts(order_kind=None, previous_order_kind="TEXT")) == ["new_order"]


def test_quiet_turn_needs_no_trigger_no_traffic_and_the_same_applied_order():
    assert quiet_turn_reason(TurnFacts()) is not None
    assert quiet_turn_reason(TurnFacts(order_changed=True)) is None
    assert quiet_turn_reason(TurnFacts(new_messages=1)) is None
    assert quiet_turn_reason(TurnFacts(damage_taken=True)) is None
    assert quiet_turn_reason(TurnFacts(first_turn=True)) is None


class _Order:
    def __init__(self, **kw):
        self.maneuver = kw.get("maneuver")
        self.maneuver_error = kw.get("maneuver_error")
        self.heading = kw.get("heading")
        self.throttle = kw.get("throttle")
        self.suggested_target = kw.get("target")
        self.torpedo_salvo = kw.get("salvo")
        self.torpedo_target = None
        self.order_text = kw.get("text", "prose")


def test_order_kind_and_signature_ignore_prose_and_small_jitter():
    heading = _Order(heading={"x": 1.0, "y": 0.0, "z": 0.0}, throttle=1.0, text="burn now")
    jitter = _Order(heading={"x": 0.999, "y": 0.01, "z": 0.0}, throttle=1.0, text="different words")
    assert order_kind([heading]) == "HEADING"
    assert order_signature([heading]) == order_signature([jitter])
    pad = _Order(maneuver={"maneuver_type": "padlock", "target": "e1"}, salvo=3)
    assert order_kind([pad]) == "PADLOCK+SALVO"
    assert order_kind([_Order()]) == "TEXT" and order_kind([]) is None
    assert order_signature([pad]) != order_signature([heading])


class _Inner:
    reasoning_effort = "medium"

    def __init__(self):
        self.calls = []

    def decide_with_tools(self, messages, tools, model=None, **kw):
        self.calls.append(kw)
        return []

    def complete(self, messages, model=None, **kw):
        self.calls.append(kw)


def _tools(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


def test_policy_client_sends_role_effort_and_nothing_when_unset():
    inner = _Inner()
    policy = EffortPolicy({"alpha": {"directive": "high", "order": "low"}})
    admiral = PolicyClient(inner, "alpha", "admiral", policy)
    admiral.decide_with_tools([], _tools("set_fleet_directive", "issue_order"), model="m")
    admiral.decide_with_tools([], _tools("issue_order"), model="m")
    admiral.decide_with_tools([], _tools("set_fleet_directive"), model="m", reasoning_effort="xhigh")
    assert [c.get("reasoning_effort") for c in inner.calls] == ["high", "low", "xhigh"]
    assert admiral.reasoning_effort == "high"          # the retry ladder starts here
    assert admiral.call_annotation() == {"role": "directive", "effort_reason": "explicit"}
    beta = PolicyClient(inner, "beta", "admiral", policy)
    beta.decide_with_tools([], _tools("issue_order"), model="m")
    assert "reasoning_effort" not in inner.calls[-1]    # beta has no setting: unchanged request
    assert beta.reasoning_effort == "medium"
    captain = PolicyClient(inner, "alpha", "captain", EffortPolicy({"alpha": {"trigger": "high"}}))
    captain.turn_effort = captain.policy.captain_turn("alpha", ["torpedo"])
    captain.decide_with_tools([], _tools("set_maneuver"), model="m")
    assert inner.calls[-1]["reasoning_effort"] == "high"
    assert captain.call_annotation()["effort_reason"] == "trigger:torpedo"
    captain.turn_effort = captain.policy.captain_turn("alpha", [])
    captain.decide_with_tools([], _tools("set_maneuver"), model="m")
    assert "reasoning_effort" not in inner.calls[-1]


# ---------------------------------------------------------------------------
# Battles on the mock
# ---------------------------------------------------------------------------

def _hashes(record):
    return sorted(hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() for b in record)


def test_default_battle_requests_are_unchanged(tmp_path):
    """No experiment flag: the requests are what the battle sent before call_policy existed."""
    record = []
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-haiku-5-5",
                                           record=record, turns=3)
    runner.run_battle(data)
    # 2 sides x 3 checkpoints x (directive + 2 orders + 2 captains) = 30 requests.
    assert len(record) == 30
    for body in record:
        assert set(body) == {"model", "max_tokens", "messages", "thinking", "output_config",
                             "system", "tools", "tool_choice"}
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium"}
        assert body["max_tokens"] == 8192 and body["tool_choice"] == {"type": "auto"}
        assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert not runner._policy_clients            # no client was wrapped
    # An explicit policy naming the default effort for every role sends the
    # identical bytes: the wrapper itself changes nothing.
    record2 = []
    same = {s: {r: "medium" for r in ("directive", "order", "captain", "trigger")}
            for s in ("alpha", "beta")}
    runner2, *_rest = _battle(tmp_path / "b", "claude-sonnet-5-5", "claude-haiku-5-5",
                              record=record2, turns=3, config={"effort_policy": same})
    runner2.run_battle(_rest[-1])
    assert runner2._policy_clients
    assert _hashes(record2) == _hashes(record)


def test_per_role_per_side_effort_and_trigger_reach_the_requests_and_the_recording(tmp_path):
    record = []
    policy = {"alpha": {"directive": "high", "order": "low", "captain": "low", "trigger": "xhigh"}}
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5",
                                           record=record, turns=3, config={"effort_policy": policy})
    runner.run_battle(data)
    calls = [e for e in json.loads(output.read_text())["events"] if e["event_type"] == "model_call"]
    by = {}
    for c in calls:
        d = c["data"]
        by.setdefault((d["faction"], d["role"]), []).append(d)
        assert d["effort"] == d["request"]["output_config"]["effort"]
    assert {d["effort"] for d in by[("alpha", "directive")]} == {"high"}
    assert {d["effort"] for d in by[("alpha", "order")]} == {"low"}
    assert {d["effort"] for d in by[("beta", "directive")]} == {"medium"}
    assert {d["effort"] for d in by[("beta", "captain")]} == {"medium"}
    assert all(d["effort_reason"] == "default" for d in by[("beta", "captain")])
    alpha_captain = by[("alpha", "captain")]
    first = [d for d in alpha_captain if d["effort_reason"].startswith("trigger:")]
    assert first and all(d["effort"] == "xhigh" for d in first)
    assert all("first" in d["effort_reason"] for d in alpha_captain[:2])
    assert all(d["effort"] == "low" for d in alpha_captain if d["effort_reason"] == "role")
    # The requests themselves carry the effort (the recording is not lying).
    efforts = {b["output_config"]["effort"] for b in record if b["model"] == "claude-sonnet-5-5"}
    assert {"high", "low", "xhigh"} <= efforts


def test_admiral_over_cheaper_captains_on_one_side(tmp_path):
    """Opus admiral, Haiku captains on alpha: each call goes to its own model."""
    record = []
    runner, client, output, data = _battle(
        tmp_path, "claude-opus-5-5", "claude-sonnet-5", record=record, turns=2,
        captain_models={"alpha": "claude-haiku-5-5"})
    runner.run_battle(data)
    calls = [e["data"] for e in json.loads(output.read_text())["events"] if e["event_type"] == "model_call"]
    roles = {(d["faction"], d["role"]): d["model"] for d in calls}
    assert roles[("alpha", "directive")] == roles[("alpha", "order")] == "claude-opus-5-5"
    assert roles[("alpha", "captain")] == "claude-haiku-5-5"
    assert roles[("beta", "captain")] == "claude-sonnet-5"


def test_merge_and_damage_triggers_fire_in_a_real_gun_fight_and_match_the_captains_warning(tmp_path):
    """The runner's trigger data is not silently empty: a closing gun fight on the
    mock produces merge and damage triggers, and the cheap detail=False range
    picture warns exactly when the captain's own (detail=True) one does."""
    from collections import Counter
    from src.llm.range_control import analyze_range_control
    policy = {s: {"trigger": "high"} for s in ("alpha", "beta")}
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5-5", turns=12,
                                           ships=("destroyer", "destroyer"),
                                           config={"effort_policy": policy})
    original = runner._captain_turn_facts
    checked = []

    def facts(ship_id, captain, orders):
        ship = runner.simulation.get_ship(ship_id)
        cheap = any(i.warning for i in analyze_range_control(runner.simulation, [ship], detail=False))
        full = any(i.warning for i in analyze_range_control(runner.simulation, [ship], detail=True))
        checked.append((cheap, full))
        return original(ship_id, captain, orders)

    runner._captain_turn_facts = facts
    runner.run_battle(data)
    reasons = Counter()
    for e in json.loads(output.read_text())["events"]:
        if e["event_type"] == "model_call" and e["data"]["role"] == "captain":
            reason = e["data"]["effort_reason"]
            for r in reason.replace("trigger:", "").split(","):
                reasons[r] += 1
            assert e["data"]["effort"] == ("high" if reason.startswith("trigger:") else "medium")
    assert reasons["merge"] > 0 and reasons["damage"] > 0 and reasons["default"] > 0
    assert checked and all(cheap == full for cheap, full in checked)
    assert any(full for _, full in checked)


def test_fleet_json_effort_blocks_and_cli_effort_merge(tmp_path):
    from scripts import run_llm_battle
    from src.llm.fleet_config import BattleFleetConfig
    from src.recording_io import load_recording
    config = {
        "battle_name": "effort json", "time_limit_s": 90, "decision_interval_s": 30,
        "initial_distance_km": 500, "max_checkpoints": 2, "seed": 3, "personality_selection": False,
        "alpha_fleet": {"admiral": "claude-sonnet-5-5", "effort": {"directive": "high", "captain": "low"},
                        "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]},
        "beta_fleet": {"admiral": "claude-sonnet-5", "ships": [{"ship_type": "frigate", "model": "claude-sonnet-5"}]},
    }
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(config))
    from src.llm.battle_runner import BattleConfig
    assert BattleConfig.from_fleet_config(BattleFleetConfig.from_json(str(path))).effort_policy == {
        "alpha": {"directive": "high", "captain": "low"}}
    code = run_llm_battle.main(["--fleet-config", str(path), "--mock-llm", "--quiet",
                                "--recording-dir", str(tmp_path / "rec"),
                                "--effort", "alpha.order=xhigh", "--effort", "beta.captain=high"])
    assert code == 0
    recording = load_recording(next((tmp_path / "rec").glob("battle_*")))
    seen = {(e["data"]["faction"], e["data"]["role"], e["data"]["effort"])
            for e in recording["events"] if e["event_type"] == "model_call"}
    assert {("alpha", "directive", "high"), ("alpha", "captain", "low"), ("alpha", "order", "xhigh"),
            ("beta", "captain", "high"), ("beta", "directive", "medium")} <= seen
