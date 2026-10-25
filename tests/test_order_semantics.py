"""
Admiral order semantics (pass-4 follow-up, "orders" workstream).

A real 150-pt DeepSeek mirror battle issued 14 per-ship orders with
maneuver=MAINTAIN + evade=dodge to put dodging on top of the fleet heading
("PHASE 3 CANCEL-CLOSURE BURN ... burn at 100%" / MAINTAIN): MAINTAIN cuts
the engine, so those ships coasted while the fleet burned. Pinned here:

- evade ALONE rides on the fleet ("all ships") order in force, else on the
  ship's current maneuver (renewed) - never dropped, never a coast;
- a per-ship maneuver REPLACES the fleet order, and the captain is told so;
- MAINTAIN / PADLOCK never burn: no throttle is rendered for them anywhere
  (order line, MY SHIP, FRIENDLIES, ENGINE FEEDBACK), and their spec carries
  throttle 1.0 so an evade overlay on them still dodges;
- a heading given with MAINTAIN / EVADE (or PADLOCK at throttle > 0) flies
  that heading;
- order text that states a burn the structured order does not fly (or the
  reverse) is flagged to the captain;
- the issue_order schema and admiral doctrine say all of the above.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm.admiral import AdmiralOrder, _attach_heading, _fleet_heading_order
from src.llm.admiral_tools import ADMIRAL_TOOLS
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CallStats, ToolCall
from src.llm.execution_feedback import _helm_text
from src.llm.maneuver_orders import (
    admiral_maneuver_spec,
    describe_maneuver_spec,
    order_throttle_conflict,
    text_throttles,
)
from src.llm.prompts import (
    ADMIRAL_SHIP_ORDER_DOCTRINE,
    CaptainPersonality,
    _maneuver_text,
    build_admiral_ship_order_messages,
    format_admiral_orders_for_captain,
)
from src.simulation import ManeuverType

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _issue_order_param(name):
    tool = next(t for t in ADMIRAL_TOOLS if t["function"]["name"] == "issue_order")
    return tool["function"]["parameters"]["properties"][name]["description"]


# =============================================================================
# Spec parsing
# =============================================================================

class TestSpec:

    def test_evade_alone_is_an_overlay(self):
        assert admiral_maneuver_spec({"evade": "dodge"}) == {
            "maneuver_type": "OVERLAY", "evade": "dodge", "throttle": 1.0}
        assert admiral_maneuver_spec({"order_text": "x"}) is None

    @pytest.mark.parametrize("kind", ["MAINTAIN", "PADLOCK"])
    def test_engine_off_maneuvers_carry_full_overlay_throttle(self, kind):
        # "PADLOCK ..., throttle 0" meant "no burn"; at 0 the dodge it also
        # ordered could not burn at all.
        spec = admiral_maneuver_spec({"maneuver": kind, "throttle": 0, "evade": "dodge"})
        assert spec["maneuver_type"] == kind and spec["throttle"] == 1.0
        assert spec["evade"] == "dodge"

    def test_maintain_with_a_heading_flies_the_heading(self):
        hdg = {"x": -1.0, "y": 0.0, "z": 0.0}
        burn = admiral_maneuver_spec({"maneuver": "MAINTAIN", "heading": hdg, "throttle": 0.6})
        assert burn == {"maneuver_type": "HEADING", "throttle": 0.6}
        # MAINTAIN says no burn: without a throttle the heading only points the nose.
        nose = admiral_maneuver_spec({"maneuver": "MAINTAIN", "heading": hdg})
        assert nose == {"maneuver_type": "HEADING", "throttle": 0.0}

    def test_evade_with_a_heading_flies_the_heading_evading(self):
        hdg = {"x": 0.03, "y": 1.0, "z": 0.0}
        assert admiral_maneuver_spec({"maneuver": "EVADE", "heading": hdg, "throttle": 1.0}) == {
            "maneuver_type": "HEADING", "evade": "full", "throttle": 1.0}
        assert admiral_maneuver_spec({"maneuver": "EVADE", "heading": hdg,
                                      "evade": "dodge"})["evade"] == "dodge"

    def test_padlock_with_a_heading_burns_only_on_an_explicit_throttle(self):
        hdg = {"x": 1.0, "y": 0.02, "z": 0.0}
        assert admiral_maneuver_spec({"maneuver": "PADLOCK", "heading": hdg,
                                      "throttle": 0.5})["maneuver_type"] == "HEADING"
        for args in ({}, {"throttle": 0}):
            spec = admiral_maneuver_spec({"maneuver": "PADLOCK", "heading": hdg, **args})
            assert spec["maneuver_type"] == "PADLOCK"

    def test_other_maneuvers_still_win_over_a_heading(self):
        spec = admiral_maneuver_spec({"maneuver": "INTERCEPT", "heading": {"x": 1, "y": 0, "z": 0},
                                      "standoff_km": 80})
        assert spec["maneuver_type"] == "INTERCEPT"

    def test_attach(self):
        fleet = _fleet_heading_order({"ship_name": "all ships", "order_text": "",
                                      "heading": {"x": 1, "y": 0, "z": 0}, "throttle": 0.5})
        # Evade alone: the fleet heading with the overlay, in the formation.
        dodger = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="")
        _attach_heading(dodger, {"evade": "dodge"}, fleet)
        assert dodger.formation is True and dodger.heading == fleet.heading
        assert dodger.maneuver == {"maneuver_type": "HEADING", "throttle": 0.5, "evade": "dodge"}
        assert dodger.replaces_fleet_order is None
        # No fleet order: the overlay spec goes to the runner.
        alone = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="")
        _attach_heading(alone, {"evade": "dodge"}, None)
        assert alone.maneuver["maneuver_type"] == "OVERLAY" and alone.formation is False
        # A per-ship MAINTAIN replaces the fleet heading - and says which.
        coaster = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="")
        _attach_heading(coaster, {"maneuver": "MAINTAIN", "evade": "dodge"}, fleet)
        assert coaster.maneuver["maneuver_type"] == "MAINTAIN" and coaster.formation is False
        assert coaster.replaces_fleet_order == "HEADING (+1.00,+0.00,+0.00) @50%"
        # A fleet PADLOCK carries throttle 1.0 like any engine-off order.
        padlock = _fleet_heading_order({"ship_name": "all ships", "maneuver": "PADLOCK",
                                        "maneuver_target": "beta_1", "throttle": 0})
        assert padlock.throttle == 1.0 and padlock.maneuver["throttle"] == 1.0


# =============================================================================
# Rendering: engine-off maneuvers never show a throttle
# =============================================================================

class TestEngineOffRendering:

    def test_order_line(self):
        assert describe_maneuver_spec({"maneuver_type": "PADLOCK", "target": "OCS Lancer-2",
                                       "evade": "dodge", "throttle": 1.0}) == \
            "PADLOCK OCS Lancer-2 (coasts, engine off), evade dodge"
        assert describe_maneuver_spec({"maneuver_type": "MAINTAIN", "throttle": 1.0}) == \
            "MAINTAIN (coasts, engine off)"
        assert "%" not in describe_maneuver_spec({"maneuver_type": "PADLOCK", "throttle": 0.0})
        assert describe_maneuver_spec({"maneuver_type": "OVERLAY", "evade": "dodge"}) == \
            "your current maneuver kept, evade dodge"

    def test_my_ship_and_friendlies(self):
        assert _maneuver_text({"type": "PADLOCK", "throttle": 1.0}) == "PADLOCK"
        assert _maneuver_text({"type": "MAINTAIN", "throttle": 1.0}) == "MAINTAIN"
        assert _maneuver_text({"type": "INTERCEPT", "throttle": 0.4}) == "INTERCEPT@40%"

    def test_decision_history_and_lost_turns(self):
        from src.llm.maneuver_orders import maneuver_label
        assert maneuver_label("PADLOCK", 1.0) == "PADLOCK"
        assert maneuver_label(ManeuverType.MAINTAIN, 1.0) == "MAINTAIN"
        assert maneuver_label("INTERCEPT", 0.4) == "INTERCEPT@40%"
        from src.llm.captain import LLMCaptain
        captain = LLMCaptain(LLMCaptainConfig(name="c", ship_name="TIS Ward-1",
                                              ship_type="destroyer", fleet_data=FLEET),
                             client=None)
        captain.last_tool_calls = [ToolCall(id="1", name="set_maneuver", arguments={
            "maneuver_type": "PADLOCK", "throttle": 1.0})]
        assert "PADLOCK" in captain.get_last_decision_summary()
        assert "PADLOCK@" not in captain.get_last_decision_summary()

    def test_engine_feedback(self):
        text = _helm_text({"mode": "PADLOCK", "requested_throttle": 1.0, "applied_throttle": 0.0})
        assert text == "PADLOCK, coasting (engine off, nose on target)"
        assert "turning onto the burn line" not in text
        assert _helm_text({"mode": "HEADING", "requested_throttle": 1.0,
                           "applied_throttle": 0.1}).endswith("(turning onto the burn line, "
                                                             "or fuel/engine limited)")
        # A throttle-0 order is flying, not "no maneuver"; its evade overlay
        # burns at full throttle while it has the helm.
        assert _helm_text({"mode": "HEADING", "requested_throttle": 0.0,
                           "applied_throttle": 0.0}) == "HEADING, throttle 0 (attitude only, no burn)"
        assert _helm_text({"mode": "RUN", "requested_throttle": 0.0, "applied_throttle": 1.0}) == \
            "RUN (evasion has the helm), applied 100% throttle"
        assert _helm_text({"mode": "COAST", "requested_throttle": 0.0}) == "COASTING (no maneuver)"


# =============================================================================
# Text vs structured throttle
# =============================================================================

class TestThrottleConflict:

    @pytest.mark.parametrize("text,expected", [
        ("PADLOCK OCS Lancer-2 - nose-on, no burn (throttle 0%).", [0.0]),
        ("fly the fleet heading (-1.00, +0.00, +0.00) at 60% throttle", [0.6]),
        ("burn at 100% of your acceleration", [1.0]),
        ("throttle 0.4 this checkpoint", [0.4]),
        ("NO closing burn (throttle 0, attitude only)", [0.0]),
        ("Zero throttle - coast nose-on", [0.0]),
        ("Do NOT burn at 60% throttle; hold.", []),
        ("Hold the line at 450 km, 30% of the fleet is engaged", []),
        # Stated zeros in words.
        ("PADLOCK - nose-on, NO thrust (coast-and-shoot).", [0.0]),
        ("Coast, no burn, guns free.", [0.0]),
        # Conditional / future clauses are not the order for now.
        ("Coast now; if he launches, burn at 100% throttle away.", []),
        ("PADLOCK. Burn at 60% throttle only if range < 200 km.", []),
        ("Hold PADLOCK, no burn. Next checkpoint we burn at 80% throttle.", [0.0]),
        ("At ~350 km expect the fleet PADLOCK order - nose on target, no burn.", []),
        # ... but a trailing note is not a condition, nor is "until".
        ("Execute the burn on heading (-1.00,-0.03,-0.02) at 100% throttle for this "
         "checkpoint (expect ~64 s, renew next checkpoint).", [1.0]),
        ("Burn at 40% throttle until relative speed is ~0.", [0.4]),
        ("You should burn at 50% throttle.", [0.5]),
        ("Should he launch, burn at 100% throttle away.", []),
    ])
    def test_stated_throttles(self, text, expected):
        assert text_throttles(text) == expected

    def _order(self, text, spec=None, heading=None, throttle=1.0):
        return AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text=text,
                            maneuver=spec, heading=heading, throttle=throttle)

    def test_coast_now_brake_later_padlock_is_no_conflict(self):
        # Recorded (NEW mirror battle, T+150 OCS Ward-2): a correct PADLOCK
        # whose text plans a later brake burn was told "!! ... fly it yourself".
        spec = {"maneuver_type": "PADLOCK", "target": "TIS Sovereign-1", "evade": "dodge",
                "throttle": 1.0}
        text = ("MAINTAIN PADLOCK on TIS Sovereign-1 \u2014 nose-on, NO thrust (coast-and-shoot). "
                "Do not fly the +X fleet burn; you are already closing at 0.6 km/s and a "
                "tail-on burn would mask your guns and invite a merge. Hold the current "
                "velocity vector and shoot: Coilgun Battery Mk3 on TIS Sovereign-1, "
                "FIRE_WHEN_OPTIMAL (lower spinal/turret gates as fleet-directed so guns actually "
                "fire). Track the range to Sovereign-1; when it falls to ~350 km, execute a "
                "short controlled brake burn on heading (+0.99,+0.13,+0.02) at ~50% throttle to "
                "kill closure near 350 km, then immediately re-establish PADLOCK nose-on TIS "
                "Sovereign-1.")
        assert text_throttles(text) == [0.0]
        assert order_throttle_conflict(self._order(text, spec)) is None
        for text in ("PADLOCK. Burn at 60% throttle only if range < 200 km.",
                     "Coast now; if he launches, burn at 100% throttle away.",
                     "Hold PADLOCK, no burn. Next checkpoint we burn at 80% throttle."):
            assert order_throttle_conflict(self._order(text, spec)) is None

    def test_padlock_said_throttle_0_is_no_conflict(self):
        spec = {"maneuver_type": "PADLOCK", "target": "OCS Lancer-2", "throttle": 1.0}
        assert order_throttle_conflict(self._order(
            "PADLOCK OCS Lancer-2, throttle 0% (coast and shoot)", spec)) is None

    def test_maintain_under_a_burn_text_is_flagged(self):
        spec = {"maneuver_type": "MAINTAIN", "evade": "dodge", "throttle": 1.0}
        text = order_throttle_conflict(self._order(
            "PHASE 2b CONTROLLED CLOSE. Fly the fleet heading (+1.00,+0.02,+0.00) at 50% "
            "throttle, tight formation.", spec))
        assert text.startswith("the text asks for a 50% burn, but the applied MAINTAIN coasts")

    def test_burning_order_said_coast_is_flagged(self):
        text = order_throttle_conflict(self._order(
            "Coast-and-shoot: HEADING (-0.99,-0.13,-0.07), throttle 0.",
            heading={"x": -0.99, "y": -0.13, "z": -0.07}, throttle=1.0))
        assert text.startswith("the text says throttle 0 (no burn), but the applied order "
                               "burns at 100%")
        assert order_throttle_conflict(self._order(
            "Burn at 40% throttle.", heading={"x": 1, "y": 0, "z": 0}, throttle=1.0)) == \
            "the text says 40% throttle, the applied order 100%."

    def test_unclear_or_matching_text_is_left_alone(self):
        heading = {"x": 1, "y": 0, "z": 0}
        assert order_throttle_conflict(self._order(
            "Burn at 60% throttle now, then coast at throttle 0.", heading=heading)) is None
        assert order_throttle_conflict(self._order(
            "Fly the fleet heading at 60% throttle.", heading=heading, throttle=0.6)) is None
        assert order_throttle_conflict(self._order("Hold the line.", heading=heading)) is None
        unapplied = self._order("Burn at 40% throttle.", heading=heading)
        unapplied.maneuver_error = "no live ship"
        assert order_throttle_conflict(unapplied) is None


# =============================================================================
# The schema and doctrine say it
# =============================================================================

class TestSchemaAndDoctrine:

    def test_issue_order_schema(self):
        maneuver = _issue_order_param("maneuver")
        assert "REPLACES the fleet" in maneuver and "MAINTAIN = engine OFF" in maneuver
        assert "To BURN on a course give heading + throttle" in maneuver
        evade = _issue_order_param("evade")
        assert "ALONE (no maneuver, no heading) it rides on the fleet order" in evade
        assert "never add MAINTAIN just to carry it" in evade
        assert "PADLOCK and MAINTAIN never burn" in _issue_order_param("throttle")
        # ... and every "never burn" says what a heading given with them does
        # (admiral_maneuver_spec flies it).
        assert ("except that MAINTAIN + heading, or PADLOCK + heading at throttle above 0, "
                "flies that HEADING at this throttle") in _issue_order_param("throttle")
        assert "both unless you also give a heading" in maneuver
        from src.llm.prompts import ADMIRAL_DOCTRINE_CORE
        core = " ".join(ADMIRAL_DOCTRINE_CORE.split())
        assert ("(MAINTAIN and PADLOCK never burn: throttle does not apply to them - but given "
                "WITH a heading, MAINTAIN, or PADLOCK at throttle > 0, flies that HEADING") in core
        from src.llm.prompts import build_admiral_static
        directive = " ".join(build_admiral_static("A", "alpha", FLEET, None,
                                                  phase="directive").split())
        assert "unless its own order sets another heading or maneuver" in directive

    def test_per_ship_order_doctrine_and_fleet_order_notice(self):
        assert "give\n  neither heading nor maneuver to keep it flying the fleet order" in \
            ADMIRAL_SHIP_ORDER_DOCTRINE
        fleet = _fleet_heading_order({"ship_name": "all ships", "maneuver": "PADLOCK",
                                      "maneuver_target": "OCS Lancer-2", "evade": "dodge"})
        user = build_admiral_ship_order_messages(
            admiral_name="A", ship_name="TIS Ward-1", ship_type="frigate", captain_name="C",
            fleet_directive="Hold.", snapshot=None, fleet_heading=fleet)[1]["content"]
        assert ("FLEET ORDER IN FORCE (your \"all ships\" order): PADLOCK OCS Lancer-2 "
                "(coasts, engine off), evade dodge. TIS Ward-1 flies it unless") in user
        assert "To keep it and dodge: evade alone." in user
        coast = _fleet_heading_order({"ship_name": "all ships", "order_text": "",
                                      "heading": {"x": 1, "y": 0, "z": 0}, "throttle": 0})
        user = build_admiral_ship_order_messages(
            admiral_name="A", ship_name="TIS Ward-1", ship_type="frigate", captain_name="C",
            fleet_directive="Hold.", snapshot=None, fleet_heading=coast)[1]["content"]
        assert ("FLEET HEADING IN FORCE (your \"all ships\" order): (+1.00,+0.00,+0.00) "
                "attitude only (throttle 0, no burn). TIS Ward-1 flies it unless") in user
        assert "To keep it and dodge: evade alone (the dodge burns at full throttle)." in user


# =============================================================================
# Through the real checkpoint loop
# =============================================================================

def _run_battle(script, checkpoints=2):
    """
    A 2v1 fleet battle with stub LLMs. ``script(checkpoint, call)`` returns
    the admiral's (directive-call extras, {ship_id: per-ship issue_order
    args}); captains only acknowledge, so what flies is what the admiral
    ordered. Returns (runner, {checkpoint: {ship_id: maneuver at decision}},
    {checkpoint: {ship_id: captain turn text}}).
    """
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner
    from src.llm.fleet_config import BattleFleetConfig

    holder = {"checkpoint": 0}
    flown, turns = {}, {}

    class Stub:
        stats = CallStats()
        session_id = None

        def decide_with_tools(self, messages, tools, model=None, temperature=None,
                              tool_choice="auto"):
            runner = holder["runner"]
            sim = runner.simulation
            names = {t["function"]["name"] for t in tools}
            text = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
            if "set_fleet_directive" in names:
                faction = "alpha" if "ALPHA" in messages[0]["content"] else "beta"
                if faction == "alpha":
                    holder["checkpoint"] += 1
                extra, _ = script(holder["checkpoint"]) if faction == "alpha" else ([], {})
                return [ToolCall(id="1", name="set_fleet_directive",
                                 arguments={"directive": "Hold the line."})] + [
                    ToolCall(id=f"f{i}", name="issue_order", arguments=args)
                    for i, args in enumerate(extra)]
            if names == {"issue_order"}:
                ship = re.search(r"Issue tactical order to (.+?)\. You MUST", text).group(1)
                sid = next(s.ship_id for s in sim.ships.values() if s.name == ship)
                _, per_ship = script(holder["checkpoint"]) if sid.startswith("alpha") else ([], {})
                args = {"ship_name": ship, "order_text": "As ordered.", "priority": "NORMAL"}
                args.update(per_ship.get(sid, {}))
                return [ToolCall(id="1", name="issue_order", arguments=args)]
            match = re.search(r"commanding (.+?)\.", text)
            ship = next((s for s in sim.ships.values() if match and s.name == match.group(1)), None)
            if ship is not None and ship.faction == "alpha":
                cp = holder["checkpoint"]
                flown.setdefault(cp, {})[ship.ship_id] = ship.current_maneuver
                turns.setdefault(cp, {}).setdefault(ship.ship_id, text)
            return [ToolCall(id="1", name="respond_to_orders",
                             arguments={"response_type": "ACKNOWLEDGE"})]

    fleet_config = BattleFleetConfig.from_dict({
        "battle_name": "Orders", "time_limit_s": 600.0, "decision_interval_s": 30.0,
        "initial_distance_km": 500.0,
        "alpha_fleet": {"admiral": {"model": "stub/m"}, "ships": [
            {"ship_type": "destroyer", "model": "stub/m"},
            {"ship_type": "destroyer", "model": "stub/m"}]},
        "beta_fleet": {"admiral": {"model": "stub/m"}, "ships": [
            {"ship_type": "destroyer", "model": "stub/m"}]},
    })
    config = BattleConfig(initial_distance_km=500.0, time_limit_s=600.0,
                          max_checkpoints=checkpoints, verbose=False,
                          personality_selection=False, record_battle=False,
                          parallel_llm=False, seed=7)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/m",
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(config=config, alpha_config=dummy, beta_config=dummy,
                             client=Stub(), fleet_config=fleet_config)
    holder["runner"] = runner
    runner.run_fleet_battle(FLEET)
    return runner, flown, turns


FLEET_BURN = {"ship_name": "all ships", "order_text": "Formation burn.",
              "heading": {"x": 1.0, "y": 0.0, "z": 0.0}, "throttle": 0.5}


def test_evade_alone_rides_the_fleet_heading_through_the_runner():
    """The trap the DeepSeek admiral fell into, done right: evade alone keeps
    the formation burn and adds the dodge; MAINTAIN + dodge coasts - and its
    captain is told it replaced the fleet burn its text asks for."""
    def script(cp):
        return [FLEET_BURN], {
            "alpha_1": {"evade": "dodge"},
            "alpha_2": {"maneuver": "MAINTAIN", "evade": "dodge",
                        "order_text": "Fly the fleet heading (+1.00,+0.00,+0.00) at 50% throttle."},
        }

    _, flown, turns = _run_battle(script, checkpoints=1)
    a1, a2 = flown[1]["alpha_1"], flown[1]["alpha_2"]
    assert a1.maneuver_type == ManeuverType.HEADING and a1.evade == "dodge"
    assert a1.heading_direction == {"x": 1.0, "y": 0.0, "z": 0.0} and a1.throttle > 0
    assert a2.maneuver_type == ManeuverType.MAINTAIN and a2.evade == "dodge"

    t1, t2 = turns[1]["alpha_1"], turns[1]["alpha_2"]
    assert "HEADING ORDER (ALREADY APPLIED to your ship): (+1.00,+0.00,+0.00)" in t1
    assert "[FORMATION:" in t1 and "...with evade dodge" in t1
    assert ("MANEUVER ORDER (ALREADY APPLIED to your ship): MAINTAIN (coasts, engine off), "
            "evade dodge - for your ship INSTEAD of the fleet order HEADING "
            "(+1.00,+0.00,+0.00) @50%") in t2
    assert ("!! ORDER TEXT vs APPLIED: the text asks for a 50% burn, but the applied "
            "MAINTAIN coasts (engine off)") in t2


def test_evade_alone_without_a_fleet_order_overlays_the_current_maneuver():
    """No "all ships" order: the overlay goes on what the ship already flies
    (renewed for the checkpoint); a ship flying nothing coasts and dodges."""
    def script(cp):
        if cp == 1:
            return [], {"alpha_1": {"heading": {"x": 0.0, "y": 1.0, "z": 0.0}, "throttle": 0.7}}
        return [], {"alpha_1": {"evade": "dodge"}, "alpha_2": {"evade": "full"}}

    _, flown, turns = _run_battle(script, checkpoints=2)
    first, second = flown[1]["alpha_1"], flown[2]["alpha_1"]
    assert first.maneuver_type == ManeuverType.HEADING and first.evade is None
    assert second.maneuver_type == ManeuverType.HEADING and second.evade == "dodge"
    assert second.heading_direction == first.heading_direction
    assert second.throttle == pytest.approx(0.7)
    assert second.start_time > first.start_time          # renewed, not expiring
    coaster = flown[2]["alpha_2"]
    assert coaster.maneuver_type == ManeuverType.MAINTAIN and coaster.evade == "full"
    assert ("MANEUVER ORDER (ALREADY APPLIED to your ship): your current HEADING "
            "(+0.00,+1.00,+0.00) @70%, renewed, evade dodge") in turns[2]["alpha_1"]
    assert ("MANEUVER ORDER (ALREADY APPLIED to your ship): coasting (engine off), "
            "evade full") in turns[2]["alpha_2"]


def test_maintain_with_a_heading_burns_through_the_runner():
    def script(cp):
        return [], {"alpha_1": {"maneuver": "MAINTAIN", "heading": {"x": 0, "y": -1, "z": 0},
                                "throttle": 0.6}}

    _, flown, turns = _run_battle(script, checkpoints=1)
    m = flown[1]["alpha_1"]
    assert m.maneuver_type == ManeuverType.HEADING and m.throttle == pytest.approx(0.6)
    assert m.heading_direction == {"x": 0.0, "y": -1.0, "z": 0.0}
    assert "HEADING ORDER (ALREADY APPLIED to your ship): (+0.00,-1.00,+0.00) at 60% throttle" \
        in turns[1]["alpha_1"]


FLEET_COAST = {"ship_name": "all ships", "order_text": "Coast and shoot.",
               "heading": {"x": 1.0, "y": 0.0, "z": 0.0}, "throttle": 0}


def _flank_torpedo_hit(maneuver):
    """Fly ``maneuver`` (re-timed to t=0) on a destroyer while a corvette
    400 km off its +y flank launches one torpedo at it: (hit location, max
    throttle applied while evading)."""
    from dataclasses import replace
    from src.physics import Vector3D
    from src.simulation import (CombatSimulation, Maneuver, SimulationEventType,
                                create_ship_from_fleet_data)

    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=42)
    ships = {}
    for sid, stype, faction, pos, fwd in (("tgt", "destroyer", "alpha", (0, 0, 0), (1, 0, 0)),
                                          ("atk", "corvette", "beta", (0, 400e3, 0), (0, -1, 0))):
        ships[sid] = create_ship_from_fleet_data(
            sid, stype, faction, FLEET, position=Vector3D(*pos), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(*fwd))
        ships[sid].name = sid
        sim.add_ship(ships[sid])
    sim.inject_command("tgt", replace(maneuver, start_time=0.0, duration=0.0,
                                      burn_start_time=None))
    sim.inject_command("atk", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
    assert sim.inject_command("atk", {"type": "launch_torpedo", "target_id": "tgt"})
    evading = 0.0
    for _ in range(400):
        sim.step()
        if ships["tgt"].execution_mode in ("RUN", "EVADE", "PRESENT"):
            evading = max(evading, ships["tgt"].applied_throttle)
        if not sim.torpedoes:
            break
    impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
    return (impacts[0].data["hit_location"] if impacts else None), evading


def test_evade_alone_under_a_throttle_0_fleet_heading_still_dodges():
    """A coast-and-shoot fleet heading (throttle 0) + per-ship evade alone:
    the ship keeps the attitude-only heading, and the dodge it ordered burns
    at full throttle while a round is inbound (it used to burn at 0 * mod
    and take the torpedo in the flank). MAINTAIN + heading + evade is the
    same order."""
    def script(cp):
        return [FLEET_COAST], {
            "alpha_1": {"evade": "dodge"},
            "alpha_2": {"maneuver": "MAINTAIN", "heading": {"x": 0.0, "y": -1.0, "z": 0.0},
                        "evade": "full"},
        }

    _, flown, turns = _run_battle(script, checkpoints=1)
    a1, a2 = flown[1]["alpha_1"], flown[1]["alpha_2"]
    assert a1.maneuver_type == ManeuverType.HEADING and a1.throttle == 0.0
    assert a1.evade == "dodge" and a1.heading_direction == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert a2.maneuver_type == ManeuverType.HEADING and a2.throttle == 0.0
    assert a2.evade == "full" and a2.heading_direction == {"x": 0.0, "y": -1.0, "z": 0.0}
    assert ("...with evade dodge: evasion flies the ship (at full throttle: the heading "
            "itself does not burn) only while a torpedo or slug is inbound") in turns[1]["alpha_1"]
    # What the runner flies dodges: the flank shot lands on the nose armor
    # ('dodge' presents it on a token burn, 'full' RUNs at full throttle).
    for maneuver, power in ((a1, 0.0), (a2, 0.99)):
        location, evading = _flank_torpedo_hit(maneuver)
        assert location == "nose" and evading > power
