"""
Captain-turn fixes from the pass-5 command-interface audit (lane cmdcap).

Each class pins one finding of the captain / comms audits
(/home/plmch/aic-wt/p5/shots/cmd/{captain,comms}/REPORT.md, proposals C# / M#):
the test fails on the code before the fix.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.communication import normalize_report, normalize_signal
from src.llm.maneuver_orders import enum_token, parse_evade
from src.llm.tools import radiator_command_from_args
from src.physics import Vector3D
from src.simulation import CombatSimulation, ManeuverType, create_ship_from_fleet_data

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _sim(own_type="destroyer"):
    sim = CombatSimulation(seed=1, time_step=1.0, decision_interval=30.0)
    for sid, name, stype, faction, pos in (
        ("alpha_1", "TIS Falchion-1", own_type, "alpha", (0, 0, 0)),
        ("alpha_2", "TIS Ward-1", "frigate", "alpha", (5, 0, 0)),
        ("beta_1", "OCS Falchion-1", "destroyer", "beta", (200, 10, 0)),
        ("beta_2", "OCS Lancer-1", "battlecruiser", "beta", (210, -10, 0)),
    ):
        ship = create_ship_from_fleet_data(sid, stype, faction, FLEET,
                                           position=Vector3D(*(c * 1000.0 for c in pos)),
                                           velocity=Vector3D(0, 0, 0),
                                           forward=Vector3D(1 if faction == "alpha" else -1, 0, 0))
        ship.name = name
        sim.add_ship(ship)
    return sim


def _captain(ship_type="destroyer"):
    config = LLMCaptainConfig(name="Falchion-1", ship_name="TIS Falchion-1", ship_type=ship_type,
                              fleet_data=FLEET)
    captain = LLMCaptain(config, client=Mock())
    captain.setup_weapon_groups(ship_type, FLEET)
    return captain


def _call(captain, sim, name, **args):
    return captain._execute_tool(ToolCall(id="c1", name=name, arguments=args), sim, "alpha_1")


# =============================================================================
# C10: a double-encoded enum ('"dodge"') must not drop the call
# =============================================================================

class TestQuotedEnums:

    @pytest.mark.parametrize("raw,want", [('"dodge"', "dodge"), ("'full'", "full"),
                                          (' "Dodge" ', "dodge"), ("“dodge”", "dodge"),
                                          ("DODGE", "dodge")])
    def test_parse_evade_tolerates_quotes_case_and_spaces(self, raw, want):
        assert parse_evade(raw) == want

    def test_enum_token(self):
        assert enum_token(' "\'PADLOCK\'" ') == "PADLOCK"
        assert enum_token("FIRE_WHEN_OPTIMAL") == "FIRE_WHEN_OPTIMAL"

    def test_set_heading_with_a_quoted_dodge_is_flown_with_the_dodge(self):
        # Recorded: c alpha_1 T+60, b2 beta_3 / beta_4 T+150 - the whole
        # set_heading was dropped and the ship flew the admiral's order.
        sim, cap = _sim(), _captain()
        m = _call(cap, sim, "set_heading", direction={"x": 0, "y": 1, "z": 0}, throttle=1.0,
                  evade='"dodge"')
        assert m is not None and m.maneuver_type == ManeuverType.HEADING
        assert m.evade == "dodge"
        assert not cap.pending_tool_errors

    def test_an_unreadable_evade_applies_the_rest_and_names_the_field(self):
        sim, cap = _sim(), _captain()
        m = _call(cap, sim, "set_heading", direction={"x": 0, "y": 1, "z": 0}, throttle=0.6,
                  evade="sideways")
        assert m is not None and m.maneuver_type == ManeuverType.HEADING
        assert m.throttle == 0.6 and m.evade is None
        assert any("evade" in e and "WAS applied" in e for e in cap.pending_tool_errors)
        m = _call(cap, sim, "set_maneuver", maneuver_type="PADLOCK", evade="sideways")
        assert m is not None and m.maneuver_type == ManeuverType.PADLOCK and m.evade is None

    @pytest.mark.parametrize("raw", ['"PADLOCK"', "padlock", " Padlock "])
    def test_set_maneuver_type_tolerates_quotes_and_case(self, raw):
        m = _call(_captain(), _sim(), "set_maneuver", maneuver_type=raw)
        assert m is not None and m.maneuver_type == ManeuverType.PADLOCK

    def test_quoted_evade_maneuver(self):
        m = _call(_captain(), _sim(), "set_maneuver", maneuver_type='"EVADE"', evade='"dodge"')
        assert m is not None and m.maneuver_type == ManeuverType.EVASIVE

    def test_string_throttle_is_parsed_not_passed_raw(self):
        sim, cap = _sim(), _captain()
        assert _call(cap, sim, "set_heading", direction={"x": 1, "y": 0, "z": 0},
                     throttle="0.5").throttle == pytest.approx(0.5)
        assert _call(cap, sim, "set_maneuver", maneuver_type="INTERCEPT",
                     throttle='"40%"').throttle == pytest.approx(0.4)
        m = _call(cap, sim, "set_maneuver", maneuver_type="INTERCEPT", throttle="fast")
        assert m is not None and m.throttle == 1.0
        assert any("throttle" in e for e in cap.pending_tool_errors)

    def test_numeric_throttle_passes_through_unchanged(self):
        sim, cap = _sim(), _captain()
        assert _call(cap, sim, "set_maneuver", maneuver_type="INTERCEPT", throttle=1).throttle == 1

    def test_quoted_weapon_mode_is_obeyed_not_turned_into_fire(self):
        # Sibling: an unknown mode silently became FIRE_WHEN_OPTIMAL, so a
        # double-encoded '"HOLD_FIRE"' opened fire.
        sim, cap = _sim(), _captain()
        cmd = _call(cap, sim, "set_weapons_order", spinal_mode='"HOLD_FIRE"',
                    coilguns_mode="fire_immediate")
        modes = {o.weapon_slot: o.command.name for o in cmd["orders"]}
        groups = cap.weapon_groups
        assert all(modes[s] == "HOLD_FIRE" for s in groups["spinal"])
        assert all(modes[s] == "FIRE_IMMEDIATE" for s in groups["coilguns"])
        assert not cap.pending_tool_errors

    def test_unknown_weapon_mode_keeps_that_group_and_names_the_field(self):
        sim, cap = _sim(), _captain()
        cmd = _call(cap, sim, "set_weapons_order", spinal_mode="BLAZE",
                    coilguns_mode="HOLD_FIRE")
        modes = {o.weapon_slot: o.command.name for o in cmd["orders"]}
        assert set(modes) == set(cap.weapon_groups["coilguns"])
        assert any("spinal_mode" in e for e in cap.pending_tool_errors)
        # Only unreadable modes: nothing is switched to FIRE_WHEN_OPTIMAL.
        cmd = _call(cap, sim, "set_weapons_order", spinal_mode="BLAZE")
        assert cmd["orders"] == []

    def test_string_gate_is_a_number(self):
        cmd = _call(_captain(), _sim(), "set_weapons_order", spinal_mode="FIRE_WHEN_OPTIMAL",
                    spinal_min_probability="0.5")
        assert all(o.min_hit_probability == pytest.approx(0.5) for o in cmd["orders"])

    def test_report_status_and_signal_kind_tolerate_quotes(self):
        # Sibling: '"DEVIATE"' was filed as an ACK, '"TARGET_CLAIM"' as FREE.
        report, errors = normalize_report({"status": '"DEVIATE"', "reason": "merge"})
        assert report["status"] == "DEVIATE" and not errors
        signal, _ = normalize_signal({"kind": "'target_claim'", "text": "mine"})
        assert signal["kind"] == "TARGET_CLAIM"

    def test_radiator_mode_tolerates_quotes(self):
        # Sibling: mode '"auto"' fell through to a manual RETRACT.
        assert radiator_command_from_args({"mode": '"auto"'})["mode"] == "auto"
        assert radiator_command_from_args({"extend": '"true"'})["extend"] is True


# =============================================================================
# C1: the helm feedback names the real cause of a short burn
# =============================================================================

from types import SimpleNamespace  # noqa: E402

from src.llm.execution_feedback import execution_feedback, helm_window, helm_window_text  # noqa: E402


def _ev(kind, t, ship="me", target=None, **data):
    return SimpleNamespace(event_type=SimpleNamespace(name=kind), timestamp=t, ship_id=ship,
                           target_id=target, data=data)


def _state(t, mode, req, app, engine=1.0, fuel=1.0):
    return _ev("EXECUTION_STATE", t, mode=mode, requested_throttle=req, applied_throttle=app,
               engine_fraction=engine, fuel_fraction=fuel)


def _helm_sim(events):
    ships = {"me": SimpleNamespace(name="TIS Me", ship_id="me", is_destroyed=False),
             "foe": SimpleNamespace(name="OCS Foe", ship_id="foe", is_destroyed=False)}
    return SimpleNamespace(events=events, current_time=60.0, decision_interval=30.0,
                           get_ship=ships.get)


HEADING = dict(maneuver_type="HEADING", throttle=1.0, evade="full",
               heading_direction={"x": 1, "y": 0, "z": 0})


class TestHelmWindow:

    def test_a_swung_burn_counts_as_burned_and_is_named(self):
        # The evade overlay on a burning order keeps the burn and swings it
        # (execution_state swing: true): those seconds are on the order.
        swing = dict(requested_throttle=1.0, applied_throttle=1.0, engine_fraction=1.0,
                     fuel_fraction=1.0, swing=True)
        events = [_ev("MANEUVER_STARTED", 30.0, **HEADING),
                  _state(30.0, "HEADING", 1.0, 1.0),
                  _ev("EXECUTION_STATE", 40.0, mode="EVADE", **swing),
                  _ev("EXECUTION_STATE", 52.0, mode="BRACE", **swing),
                  _ev("PROJECTILE_LAUNCHED", 41.0, ship="foe", target="me")]
        w = helm_window(_helm_sim(events), "me", 30.0, 60.0)
        assert w["burned_s"] == pytest.approx(30.0) and not w["modes"]
        assert w["swing"] == {"EVADE": pytest.approx(12.0), "BRACE": pytest.approx(8.0)}
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert text == ("helm T+30-60 (HEADING @100%, evade full): burned 30 of 30 s on your "
                        "order; evasion swung the burn within 20° of your line 20 s: EVADE 12 s "
                        "dodging, BRACE 8 s armour (fire from OCS Foe [foe])")

    def test_burning_while_still_turning_onto_the_line_is_not_the_orders_burn(self):
        swing = dict(requested_throttle=1.0, applied_throttle=1.0, engine_fraction=1.0,
                     fuel_fraction=1.0, swing=True)
        events = [_ev("MANEUVER_STARTED", 30.0, **HEADING),
                  _ev("EXECUTION_STATE", 30.0, mode="TANK", off_line=True, **swing),
                  _ev("EXECUTION_STATE", 45.0, mode="EVADE", **swing),
                  _ev("PROJECTILE_LAUNCHED", 31.0, ship="foe", target="me")]
        w = helm_window(_helm_sim(events), "me", 30.0, 60.0)
        assert w["swing_turn_s"] == pytest.approx(15.0) and w["burned_s"] == pytest.approx(15.0)
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert "burned 15 of 30 s on your order" in text
        assert "turning onto the burn line at your throttle under fire 15 s" in text

    def test_beyond_90_degrees_off_the_line_the_drive_idles_and_it_says_turning(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **HEADING),
                  _ev("EXECUTION_STATE", 30.0, mode="TANK", requested_throttle=1.0,
                      applied_throttle=0.05, engine_fraction=1.0, fuel_fraction=1.0,
                      swing=True, off_line=True),
                  _ev("EXECUTION_STATE", 40.0, mode="EVADE", requested_throttle=1.0,
                      applied_throttle=1.0, engine_fraction=1.0, fuel_fraction=1.0,
                      swing=True),
                  _ev("PROJECTILE_LAUNCHED", 31.0, ship="foe", target="me")]
        w = helm_window(_helm_sim(events), "me", 30.0, 60.0)
        assert w["turning_s"] == pytest.approx(10.0) and w["swing_turn_s"] == 0.0
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert "turning onto the burn line 10 s" in text and "at your throttle" not in text

    def test_evasion_holding_the_burn_is_named_with_the_threat(self):
        # Recorded (d alpha_1 T+240): "BRACE, requested 100% throttle, applied
        # 0% (turning onto the burn line, or fuel/engine limited)". (Recorded
        # before the evade overlay kept a burning order's burn; the window
        # line still reads such recordings - and a hold without 'swing'.)
        events = [_ev("MANEUVER_STARTED", 30.0, **HEADING),
                  _state(30.0, "HEADING", 1.0, 0.05),     # 4 s turning
                  _state(34.0, "HEADING", 1.0, 1.0),      # 8 s burning
                  _state(42.0, "BRACE", 1.0, 0.0),        # 12 s braced
                  _state(54.0, "EVADE", 1.0, 1.0),        # 6 s jinking
                  _ev("PROJECTILE_LAUNCHED", 40.0, ship="foe", target="me")]
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert text.startswith("helm T+30-60 (HEADING @100%, evade full): burned 8 of 30 s")
        assert "evasion held the burn 18 s: BRACE 12 s drive off, EVADE 6 s jinking" in text
        assert "fire from OCS Foe [foe]" in text
        assert "turning onto the burn line 4 s" in text
        assert "fuel/engine limited" not in text

    def test_a_dead_drive_is_named(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **dict(HEADING, evade=None)),
                  _state(30.0, "HEADING", 1.0, 0.0, engine=0.0)]
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert "burned 0 of 30 s" in text and "DRIVE DEAD 30 s" in text

    def test_the_interlock_is_named(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **dict(HEADING, evade=None)),
                  _state(30.0, "HEADING", 1.0, 1.0),
                  _ev("DRIVE_INTERLOCK", 40.0, phase="enter"),
                  _state(40.0, "HEADING", 1.0, 0.2),
                  _ev("DRIVE_INTERLOCK", 55.0, phase="exit"),
                  _state(55.0, "HEADING", 1.0, 1.0)]
        w = helm_window(_helm_sim(events), "me", 30.0, 60.0)
        assert w["interlock_s"] == pytest.approx(15.0) and w["burned_s"] == pytest.approx(15.0)
        assert "drive interlock 15 s" in helm_window_text(_helm_sim(events), "me", 30.0, 60.0)

    def test_a_full_burn_needs_no_window_line(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **dict(HEADING, evade=None)),
                  _state(30.0, "HEADING", 1.0, 1.0)]
        assert helm_window_text(_helm_sim(events), "me", 30.0, 60.0) is None

    def test_engine_off_order_with_the_overlay_says_evasion_flew_it(self):
        events = [_ev("MANEUVER_STARTED", 30.0, maneuver_type="PADLOCK", throttle=1.0,
                      evade="dodge"),
                  _state(30.0, "PADLOCK", 1.0, 0.0), _state(50.0, "TANK", 1.0, 0.5)]
        text = helm_window_text(_helm_sim(events), "me", 30.0, 60.0)
        assert text == "helm T+30-60 (PADLOCK, evade dodge): evasion flew the ship 10 s: TANK 10 s half throttle"

    def test_new_orders_applied_now_are_not_last_windows(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **dict(HEADING, evade=None)),
                  _state(30.0, "HEADING", 1.0, 0.0, engine=0.0),
                  _ev("MANEUVER_STARTED", 60.0, maneuver_type="PADLOCK", throttle=1.0)]
        assert "HEADING @100%" in helm_window_text(_helm_sim(events), "me", 30.0, 60.0)

    def test_window_line_replaces_the_catch_all_in_engine_feedback(self):
        events = [_ev("MANEUVER_STARTED", 30.0, **HEADING), _state(30.0, "BRACE", 1.0, 0.0)]
        sim = _helm_sim(events)
        text = execution_feedback(sim, ["me"])
        assert "BRACE 30 s drive off" in text and "fuel/engine limited" not in text


# =============================================================================
# C2: a dead (or damaged) drive is stated, and a burn order to it is unflyable
# =============================================================================

from src.llm.admiral import AdmiralOrder  # noqa: E402
from src.llm.prompts import format_my_ship  # noqa: E402


def _kill_drive(ship, health=5.0):
    for module in ship._get_modules_by_type("engine"):
        module.health_percent = health


class TestDeadDrive:

    def test_my_ship_says_drive_dead_instead_of_dropping_the_g(self):
        # Recorded (c beta_5 T+390): "Frigate, 90 deg turn ~83 s" - no g at all.
        text = format_my_ship("Ward-2", "TIS Ward-2", "frigate", {},
                              {"own": {"ship_id": "beta_5", "accel_g": 0.0, "engine_fraction": 0.0,
                                       "turn_90_rcs_s": 83.0}}, FLEET)
        assert "DRIVE DEAD (0 g): cannot burn; RCS turns only (90° ~83 s)" in text.splitlines()[0]

    def test_my_ship_says_a_damaged_drive(self):
        text = format_my_ship("Ward-2", "TIS Ward-2", "frigate", {},
                              {"own": {"ship_id": "beta_5", "accel_g": 1.8,
                                       "engine_fraction": 0.6}}, FLEET)
        assert "1.8g (drive damaged: 60% thrust)" in text.splitlines()[0]

    def test_live_ship_status_carries_the_dead_drive(self):
        sim, cap = _sim(), _captain()
        _kill_drive(sim.get_ship("alpha_1"))
        status = cap._build_tactical_status(sim.get_ship("alpha_1"), sim.get_ship("beta_1"), sim)
        assert status["own"]["engine_fraction"] == 0.0 and status["own"]["accel_g"] == 0.0
        assert status["own"]["turn_90_rcs_s"] > 0

    def test_order_check_marks_a_burn_order_unflyable(self):
        sim, cap = _sim(), _captain()
        _kill_drive(sim.get_ship("alpha_1"))
        order = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="TIS Falchion-1",
                             order_text="Burn away.", heading={"x": -1, "y": 0, "z": 0},
                             throttle=1.0)
        check = cap._order_check(order, sim, sim.get_ship("alpha_1"), [])
        assert check["engine"].startswith("NOT FLYABLE: your drive is DEAD (0 g)")
        padlock = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="TIS Falchion-1",
                               order_text="Nose on.", maneuver={"maneuver_type": "PADLOCK",
                                                                "throttle": 1.0})
        assert "NOT FLYABLE" not in (cap._order_check(padlock, sim, sim.get_ship("alpha_1"), [])
                                     .get("engine") or "")


# =============================================================================
# M1: the captain sees the verdict its admiral reads; a departure is put back
# to it once, neutrally; `reason` on action calls is filed; the EVADE trap
# =============================================================================

from tests.test_fleet_comms import Stub, events, make_runner  # noqa: E402
from src.llm.tools import ACTION_REASON_TOOLS, CAPTAIN_TOOLS_BASE  # noqa: E402

HEADING_ORDER = {"order_text": "Burn on the open-range heading and evade inbound fire.",
                 "priority": "HIGH", "heading": {"x": -1.0, "y": 0.0, "z": 0.0}, "throttle": 1.0,
                 "evade": "full"}


def _loop_runner(tmp_path, captain, checkpoints=3):
    stub = Stub(captain=captain, order=lambda ship: dict(HEADING_ORDER))
    runner = make_runner(stub, tmp_path, alpha=(("destroyer", "stub/dd"),),
                         beta=(("destroyer", "stub/foe"),), checkpoints=checkpoints,
                         distance_km=400.0)
    stub.texts = {}

    def wrapped(ctx):
        stub.texts[(ctx.model, ctx.cp, ctx.turn)] = ctx.messages[-1]["content"]
        return captain(ctx)
    stub.captain = wrapped
    return stub, runner


def _reports(runner):
    return {e.timestamp: e.data for e in events(runner, "captain_report") if e.ship_id == "alpha_1"}


class TestComplianceLoop:

    def test_evade_over_a_heading_order_is_put_back_and_shown_next_turn(self, tmp_path):
        def captain(ctx):
            if ctx.model != "stub/dd":
                return []
            if ctx.cp == 1 and ctx.turn == 1:
                return [("set_maneuver", {"maneuver_type": "EVADE"})]
            return [("set_primary_target", {"target_id": "beta_1"})]
        stub, runner = _loop_runner(tmp_path, captain)
        runner.run_fleet_battle(FLEET)
        # Same checkpoint: one ORDER CHECK, neutral, naming the trap.
        check = stub.texts[("stub/dd", 1, 2)]
        assert "=== ORDER CHECK (same checkpoint, T+30) ===" in check
        assert "set_maneuver EVADE@100% REPLACES it" in check
        assert "the order's evade already rides on it" in check
        assert "Deviating can be right" in check
        assert ("stub/dd", 1, 3) not in stub.texts          # asked once
        # Next checkpoint: the verdict the admiral read, once.
        assert ("LAST CHECKPOINT (T+30) your Admiral read NOT as ordered: maneuver: ordered "
                "HEADING, flew EVADE (your set_maneuver EVADE@100% replaced it); no reason "
                "filed.") in stub.texts[("stub/dd", 2, 1)]
        # It complied at T+60: no verdict line at T+90, no follow-up at T+60.
        assert "LAST CHECKPOINT" not in stub.texts[("stub/dd", 3, 1)]
        assert ("stub/dd", 2, 2) not in stub.texts

    def test_ack_in_the_order_check_withdraws_the_departing_call(self, tmp_path):
        def captain(ctx):
            if ctx.model != "stub/dd":
                return []
            if ctx.cp == 1 and ctx.turn == 1:
                return [("set_maneuver", {"maneuver_type": "EVADE"})]
            if ctx.cp == 1:
                return [("report_to_admiral", {"status": "ACK"})]
            return []
        _, runner = _loop_runner(tmp_path, captain, checkpoints=2)
        runner.run_fleet_battle(FLEET)
        comp = _reports(runner)[30.0]["compliance"]
        assert comp["executed_maneuver"] == "HEADING" and comp["followed"]

    def test_a_reason_on_the_call_is_filed_and_skips_the_check(self, tmp_path):
        def captain(ctx):
            if ctx.model != "stub/dd":
                return []
            if ctx.cp == 1:
                return [("set_maneuver", {"maneuver_type": "EVADE",
                                          "reason": "slugs inbound; EVADE until clear"})]
            return []
        stub, runner = _loop_runner(tmp_path, captain, checkpoints=2)
        runner.run_fleet_battle(FLEET)
        assert ("stub/dd", 1, 2) not in stub.texts
        report = _reports(runner)[30.0]
        assert report["status"] == "DEVIATE"
        assert report["reason"] == "set_maneuver: slugs inbound; EVADE until clear"
        assert ("you filed DEVIATE: set_maneuver: slugs inbound"
                in stub.texts[("stub/dd", 2, 1)])

    def test_complying_turns_get_no_check_and_no_verdict(self, tmp_path):
        def captain(ctx):
            return [("set_primary_target", {"target_id": "beta_1"})] if ctx.model == "stub/dd" else []
        stub, runner = _loop_runner(tmp_path, captain)
        runner.run_fleet_battle(FLEET)
        assert all(turn == 1 for (model, _, turn) in stub.texts if model == "stub/dd")
        assert not any("LAST CHECKPOINT" in t for t in stub.texts.values())

    def test_reason_field_and_evade_wording_in_the_tools(self):
        cap = _captain()
        cap.has_admiral = True
        tools = {t["function"]["name"]: t["function"] for t in cap.get_tools_for_context()}
        for name in ACTION_REASON_TOOLS:
            if name in tools:
                assert "reason" in tools[name]["parameters"]["properties"]
        cap.has_admiral = False
        tools = {t["function"]["name"]: t["function"] for t in cap.get_tools_for_context()}
        assert "reason" not in tools["set_maneuver"]["parameters"]["properties"]
        # The base schema is not mutated.
        base = next(t for t in CAPTAIN_TOOLS_BASE if t["function"]["name"] == "set_maneuver")
        assert "reason" not in base["function"]["parameters"]["properties"]
        text = tools["set_maneuver"]["parameters"]["properties"]["maneuver_type"]["description"]
        assert "EVADE REPLACES whatever you fly" in text


# =============================================================================
# C7/C8/M7: contradictions and wrong lines
# =============================================================================

import math  # noqa: E402
from src.llm import arcs_status  # noqa: E402
from src.llm.captain import rounds_to_breach  # noqa: E402
from src.llm.prompts import (  # noqa: E402
    _comms_line, build_captain_doctrine, format_threats_to_me)
from src.llm.range_control import (  # noqa: E402
    HeadingOption, guns_bearing, merged_options)


class TestRealArcs:

    def _frame(self, sim, ship, forward, roll_deg):
        """Point ``ship`` along ``forward`` and roll it about that axis."""
        f = forward.normalized()
        up0 = Vector3D(0, 0, 1) if abs(f.z) < 0.9 else Vector3D(0, 1, 0)
        side = f.cross(up0).normalized()
        up = up0 - f * f.dot(up0)
        up = up.normalized()
        r = math.radians(roll_deg)
        up = up * math.cos(r) + side * math.sin(r)
        ship.kinematic_state.forward = f
        ship.kinematic_state.up = up
        arcs = sim._arcs_for(ship)
        arcs.roll = 0.0
        arcs.set_frame(f, up, sim.current_time)

    @pytest.mark.parametrize("angle", [0, 45, 90, 120, 150, 170, 178])
    def test_batteries_that_bear_match_the_engine_masks_at_some_roll(self, angle):
        # The old analytic cone said "guns masked" past 90 deg while the ARCS
        # lines said the turrets bore (128 contradicting vector pairs).
        sim = _sim()
        me, target = sim.get_ship("alpha_1"), sim.get_ship("beta_1")
        los = (target.position - me.position).normalized()
        perp = Vector3D(0, 0, 1)
        a = math.radians(angle)
        nose = los * math.cos(a) + perp * math.sin(a)
        claim = guns_bearing(sim, [me], angle)
        best = 0
        for roll in range(0, 360, 5):
            self._frame(sim, me, nose, roll)
            report = arcs_status.gun_report(sim, me, target)
            best = max(best, sum(1 for r in report["mounts"] if r["bears"] and r["kind"] == "gun"))
        if best:
            assert "batter" in claim, (angle, claim)
            assert f"{best} of" in claim, (angle, claim, best)
        else:
            assert "batter" not in claim, (angle, claim)

    def test_merged_options_fold_near_identical_vectors(self):
        def opt(key, label, x, y):
            return HeadingOption(key=key, label=label, heading=Vector3D(x, y, 0).normalized(),
                                 throttle=1.0, turn_s=1.0, burn_s=30.0, effect="e")
        rows = merged_options([opt("wide", "pass wide", -0.08, 0.99), opt("left", "flank left", 0.08, -1),
                               opt("right", "flank right", -0.08, 1.0)])
        assert [label for _, label in rows] == ["pass wide = flank right", "flank left"]

    def test_coast_and_shoot_states_its_range_cost_and_every_line_its_armor(self):
        sim, cap = _sim(), _captain()
        from src.llm.range_control import analyze_range_control, format_captain_range_control
        text = format_captain_range_control(analyze_range_control(sim, [sim.get_ship("alpha_1")]))
        face = next(l for l in text.splitlines() if "coast & shoot" in l)
        assert "coasting: closest approach" in face or "coasting: range not closing" in face
        for line in text.splitlines():
            if "set_heading(" in line:
                assert " cm)" in line and ("bear" in line), line
        assert "guns masked" not in text and "turrets bear" not in text


class TestDoctrineLines:

    def _doc(self, ship_type, names, admiral=True):
        return build_captain_doctrine(ship_type=ship_type, fleet_data=FLEET, tool_names=names,
                                      has_admiral=admiral)

    def test_no_turret_180_claims_and_no_selling(self):
        names = ["set_maneuver", "set_heading", "set_weapons_order", "set_primary_target",
                 "report_to_admiral", "surrender"]
        doc = self._doc("destroyer", names)
        for phrase in ("180°", "180 deg", "(recommended", "worth far more", "single biggest",
                       "BEST during", "keep this toward the threat", "spinal_mode / turret_mode"):
            assert phrase not in doc, phrase
        assert "- spinal_mode / coilguns_mode: FIRE_IMMEDIATE" in doc
        bc = self._doc("battlecruiser", names)
        assert "spinal_mode / coilguns_mode" in bc or "heavy_coilguns_mode" in bc

    def test_propose_draw_only_when_offered(self):
        doc = self._doc("corvette", ["set_maneuver", "launch_torpedo", "surrender",
                                     "report_to_admiral"])
        assert "propose_draw" not in doc and "- surrender: strike your colours" in doc
        solo = self._doc("corvette", ["set_maneuver", "surrender", "propose_draw"], admiral=False)
        assert "- surrender / propose_draw:" in solo

    def test_gunless_heat_line_has_no_guns_rung(self):
        status = {"thermal": {"heat_percent": 70.0, "heat_gj": 224, "capacity_gj": 320,
                              "net_rate_mw": 10.0, "radiators_extended": False,
                              "radiator_available_mw": 780, "radiator_rating_mw": 780}}
        corvette = format_my_ship("Dart", "TIS Dart", "corvette", status, {}, FLEET)
        destroyer = format_my_ship("F", "TIS F", "destroyer", status, {}, FLEET)
        assert "guns hold" not in corvette and "radiators forced" in corvette
        assert "guns hold" in destroyer

    def test_tot_sync_eta_is_rebased_to_an_absolute_time(self):
        line = _comms_line({"sender_name": "TIS Dart-2", "sender_id": "alpha_2",
                            "channel": "fleet_net", "sent_t": 60.0, "text": "my rounds land in 45 s",
                            "structured": {"eta_s": 45}})
        assert "[rounds land ~T+105]" in line


class TestTorpedoBlock:

    def _torp(self, need, own=1, allied=2):
        return {"distance_km": 200, "closing_kps": 5.0, "eta_seconds": 40, "source": "X",
                "est_impact_gj": 25.0, "nez_inside": True, "blinded": False,
                "pd_turrets_needed": need, "own_pd_turrets": own, "allied_pd_turrets": allied}

    def test_saturated_rounds_carry_one_verdict(self):
        # Recorded (b2 beta_1 T+60): "your PD umbrella can blind it" on every
        # round, then "SATURATION: ... PD blinds at most 4, 11 arrive".
        text = format_threats_to_me({"torpedo_threats": [self._torp(3) for _ in range(5)]})
        assert "can blind it" not in text and "CANNOT stop it" not in text
        assert "SATURATION: 5 live rounds vs 3 turrets bearing (~3 per round): PD blinds at most 1, 4 arrive" in text
        assert "EVADE now" not in text

    def test_unsaturated_pair_says_so(self):
        text = format_threats_to_me({"torpedo_threats": [self._torp(1), self._torp(1)]})
        assert "enough turrets to blind every one" in text and "SATURATION" not in text

    def test_rounds_to_breach_is_the_engine_hit(self):
        from tests.test_fixes_pass5_damage import _ship as _dmg_ship, _torpedo_flight
        sim = CombatSimulation(seed=1, time_step=1.0)
        _dmg_ship(sim, "shooter", "alpha", Vector3D(500_000, 0, 0))
        target = _dmg_ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="destroyer")
        nose = target.armor.get_section("nose")
        before = nose.thickness_cm
        # Straight onto the nose at 12 km/s.
        flight = _torpedo_flight(sim, target.forward * 1000.0,
                                 target.forward * -12_000.0, "victim")
        gj = 0.5 * flight.torpedo.specs.penetrator_mass_kg * 12_000.0 ** 2 / 1e9
        import copy
        trial = copy.deepcopy(nose)
        trial.thickness_cm = before
        sim._resolve_torpedo_hit(flight, target)
        from src.llm.captain import TORPEDO_KINETIC_FLAT_CHIPPING, TORPEDO_KINETIC_IMPACT_AREA_M2
        trial.apply_energy_damage(energy_gj=gj, flat_chipping=TORPEDO_KINETIC_FLAT_CHIPPING,
                                  impact_area_m2=TORPEDO_KINETIC_IMPACT_AREA_M2)
        assert nose.thickness_cm == pytest.approx(trial.thickness_cm, abs=1e-6)
        n = rounds_to_breach(target.armor.get_section("lateral"), gj)
        assert n is not None and n >= 1


# =============================================================================
# C4/C6: PADLOCK / MAINTAIN orders get the ENGINE ORDER CHECK
# =============================================================================

class TestEngineOffOrderCheck:

    @pytest.mark.parametrize("kind", ["PADLOCK", "MAINTAIN"])
    def test_engine_off_order_states_the_coast_closest_approach(self, kind):
        from src.llm.range_control import analyze_range_control
        sim, cap = _sim(), _captain()
        me = sim.get_ship("alpha_1")
        me.kinematic_state.velocity = Vector3D(2000.0, 0.0, 0.0)   # closing at 2 km/s
        infos = analyze_range_control(sim, [me])
        order = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="TIS Falchion-1",
                             order_text="Nose on.", maneuver={"maneuver_type": kind,
                                                              "throttle": 1.0})
        engine = cap._order_check(order, sim, me, infos)["engine"]
        near = min(infos, key=lambda i: i.cpa_km)
        assert engine.startswith(f"{kind} coasts (engine off) -> closest approach "
                                 f"{near.cpa_km:.1f} km")
        assert "MERGE" in engine            # 2 km/s straight at a ship 200 km out

    def test_engine_off_order_when_nothing_closes(self):
        from src.llm.range_control import analyze_range_control
        sim, cap = _sim(), _captain()
        me = sim.get_ship("alpha_1")
        me.kinematic_state.velocity = Vector3D(-1000.0, 0.0, 0.0)  # opening
        order = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="TIS Falchion-1",
                             order_text="Nose on.", maneuver={"maneuver_type": "PADLOCK"})
        engine = cap._order_check(order, sim, me, analyze_range_control(sim, [me]))["engine"]
        assert engine == "PADLOCK coasts (engine off) -> range to every enemy group is not closing"


# =============================================================================
# C5: a smaller turn when the admiral's maneuver flies and nothing triggers
# =============================================================================

from src.llm.arcs_status import blind_cone_doctrine, blind_cones  # noqa: E402
from src.llm.range_control import _bears_at_some_roll  # noqa: E402
from src.weapon_arcs import load_hull_mounts  # noqa: E402

PADLOCK_ORDER = {"order_text": "Nose on the nearest, fire when optimal.", "priority": "NORMAL",
                 "maneuver": "PADLOCK", "evade": "dodge"}


class TestCompactTurn:

    def _texts(self, tmp_path, order, admirals=True, distance=600.0):
        stub = Stub(captain=lambda ctx: [("set_primary_target", {"target_id": "beta_1"})],
                    order=lambda ship: dict(order))
        runner = make_runner(stub, tmp_path, alpha=(("destroyer", "stub/dd"), ("frigate", "stub/ff")),
                             beta=(("destroyer", "stub/foe"),), checkpoints=2,
                             distance_km=distance, admirals=admirals)
        texts = {}
        inner = stub.captain

        def wrapped(ctx):
            texts[(ctx.model, ctx.cp)] = ctx.messages[-1]["content"]
            return inner(ctx)
        stub.captain = wrapped
        runner.run_fleet_battle(FLEET)
        return texts

    def test_admiral_maneuver_flying_and_nothing_triggering_is_compact(self, tmp_path):
        text = self._texts(tmp_path, PADLOCK_ORDER)[("stub/dd", 2)]
        assert "=== RANGE CONTROL (compact" in text
        assert "set_heading(" not in text                     # vectors only
        friendlies = text[text.index("FRIENDLIES"):text.index("=== RANGE CONTROL")]
        assert "dir_to" not in friendlies and "rel " not in friendlies
        assert "REPLY:" not in text and "ARCS blind cones" not in text

    def test_no_admiral_gets_the_full_blocks(self, tmp_path):
        text = self._texts(tmp_path, PADLOCK_ORDER, admirals=False)[("stub/dd", 2)]
        assert "compact" not in text and "set_heading(" in text
        assert "dir_to" in text[text.index("FRIENDLIES"):]

    def test_a_trigger_brings_the_full_blocks_back(self):
        cap = _captain()
        cap.has_admiral = True
        cap.admiral_orders = [AdmiralOrder(target_ship_id="alpha_1", target_ship_name="x",
                                           order_text="x", maneuver={"maneuver_type": "PADLOCK"})]
        calm = SimpleNamespace(warning=None)
        assert cap._compact_turn([calm], "ORDERS", {}) is True
        assert cap._compact_turn([SimpleNamespace(warning="MERGE IN 30 s")], "ORDERS", {}) is False
        assert cap._compact_turn([calm], "ORDERS\n      !! Your order and RANGE CONTROL conflict", {}) is False
        assert cap._compact_turn([calm], "  LAST CHECKPOINT (T+30) your Admiral read NOT", {}) is False
        assert cap._compact_turn([calm], "ORDERS", {"torpedo_threats": [{"blinded": True}]}) is False
        cap.admiral_orders = [AdmiralOrder(target_ship_id="alpha_1", target_ship_name="x",
                                           order_text="Close in.")]            # prose only
        assert cap._compact_turn([calm], "ORDERS", {}) is False


class TestBlindConeDoctrine:

    @pytest.mark.parametrize("hull", sorted(load_hull_mounts()))
    def test_outside_the_stated_cones_some_mount_bears(self, hull):
        mounts = load_hull_mounts()[hull]
        cones = blind_cones(mounts)
        for kind in ("gun", "pd"):
            ms = [m for m in mounts.mounts if m.kind == kind]
            for deg in range(181):
                inside = deg <= cones.get(f"{kind}_nose", 0.0) or 180 - deg <= cones.get(f"{kind}_stern", 0.0)
                if ms and not inside:
                    assert any(_bears_at_some_roll(mounts, m)[deg] for m in ms), (hull, kind, deg)

    def test_destroyer_doctrine_states_its_stern_cone(self):
        cones = blind_cones(load_hull_mounts()["destroyer"])
        text = blind_cone_doctrine("destroyer")
        assert (f"No gun battery can fire at a target within {cones['gun_stern']:.0f} deg of dead "
                "astern, at any roll") in text
        assert "stern-arc" not in text
        doc = build_captain_doctrine(ship_type="destroyer", fleet_data=FLEET)
        assert text in doc


# =============================================================================
# M2 / M5 / M6: what peers and the engine know
# =============================================================================

from src.llm.fleet_comms import SIGNAL_LAUNCH, SIGNAL_TARGET, compress_datalink  # noqa: E402
from src.llm.prompts import format_friendlies  # noqa: E402


class TestPeerKnowledge:

    def test_ordered_targets_line_shows_a_split(self, tmp_path):
        def order(ship):
            return {"order_text": "Engage.", "priority": "NORMAL",
                    "suggested_target": "beta_1" if "Dart" in ship else "beta_2"}
        stub = Stub(captain=lambda ctx: [], order=order)
        runner = make_runner(stub, tmp_path, beta=(("destroyer", "stub/lancer"),
                                                    ("destroyer", "stub/lancer")), checkpoints=1)
        texts = {}

        def captain(ctx):
            texts[ctx.model] = ctx.messages[-1]["content"]
            return []
        stub.captain = captain
        runner.run_fleet_battle(FLEET)
        line = next(l for l in texts["stub/dart"].splitlines() if "ORDERED TARGETS" in l)
        assert "<- you" in line and line.count("<-") == 2, line

    def test_a_friendly_under_torpedo_attack_says_so(self):
        from tests.test_fixes_pass5_damage import _torpedo_flight
        sim, cap = _sim(), _captain()
        mate = sim.get_ship("alpha_2")
        for i in range(3):
            _torpedo_flight(sim, mate.position + Vector3D(60_000.0, 0, 0),
                            Vector3D(-8_000.0, 0, 0), "alpha_2", torpedo_id=f"t{i}")
        info = cap._friendly_info(sim.get_ship("alpha_1"), mate, sim, [])
        assert info["torps_inbound"] == 3 and info["torps_first_s"] == pytest.approx(7.5, abs=0.5)
        assert info["torps_pd"]["per_round"] >= 1
        line = format_friendlies([info])
        assert "!! 3 torpedoes inbound, first impact ~8 s" in line
        assert "its PD umbrella (" in line

    def test_datalink_is_compressed_for_the_captain(self):
        def eng(kind, text, **st):
            return {"sender_role": "engine", "kind": kind, "text": text, "structured": st,
                    "channel": "fleet_net", "sent_t": 90.0}
        items = [eng(SIGNAL_TARGET, "TIS A switched to OCS X"),
                 eng(SIGNAL_LAUNCH, "a", subject_name="TIS Dart-1", target_id="b1",
                     target_name="OCS X", rounds=2, in_flight=2, eta_s_min=68.0, eta_s_max=442.0),
                 eng(SIGNAL_LAUNCH, "b", subject_name="TIS Dart-2", target_id="b1",
                     target_name="OCS X", rounds=1, in_flight=1, eta_s_min=60.0, eta_s_max=60.0),
                 {"sender_role": "captain", "kind": "FREE", "text": "hello"}]
        out = compress_datalink(items)
        assert [i["text"] for i in out] == [
            "TIS Dart-1, TIS Dart-2 launched 3 torpedoes at OCS X, 3 in flight, first impact in ~60 s",
            "hello"]
        assert items[1]["text"] == "a"                     # the recorded items are untouched


# =============================================================================
# M3: each faction drafts its own ship names (commander drafts)
# =============================================================================

from src.llm.fleet_draft import (  # noqa: E402
    BETA_CLASS_NICKNAMES, CLASS_NICKNAMES, auto_draft, name_drafted_ships)


class TestFactionNames:

    def test_pools_share_no_name(self):
        assert not set(CLASS_NICKNAMES.values()) & set(BETA_CLASS_NICKNAMES.values())
        assert set(BETA_CLASS_NICKNAMES) == set(CLASS_NICKNAMES)

    def test_commander_drafts_get_their_own_names(self):
        roster = ["destroyer", "destroyer", "battlecruiser"]
        alpha = [s.ship_name for s in name_drafted_ships(roster, "alpha", faction_pool=True)]
        beta = [s.ship_name for s in name_drafted_ships(roster, "beta", faction_pool=True)]
        assert alpha == ["TIS Falchion-1", "TIS Falchion-2", "TIS Lancer"]
        assert beta == ["OCS Saber-1", "OCS Saber-2", "OCS Halberd"]

    def test_llm_beta_draft_uses_the_beta_pool(self):
        from tests.test_fleet_draft import DraftStubClient
        from src.llm.fleet_draft import SHIP_POINT_COSTS, run_admiral_draft
        budget = 2 * SHIP_POINT_COSTS["cruiser_torpedo"] + 2 * SHIP_POINT_COSTS["destroyer"]
        draft = run_admiral_draft(client=DraftStubClient(), model="stub-model",
                                  admiral_name="Admiral Stub", faction="beta", fleet_data=FLEET,
                                  budget=budget, max_ships=8, verbose=False)
        assert [s.ship_name for s in draft.ships][:3] == ["OCS Marlin-1", "OCS Marlin-2", "OCS Saber-1"]

    def test_seeded_auto_drafts_keep_the_shared_pool(self):
        # perf_check / tournaments / heuristic battles: names are in their
        # hashed event streams, which must not move.
        names = [s.ship_name for s in auto_draft("beta", 100, 8, seed=3).ships]
        assert all(any(n.split(" ", 1)[1].startswith(nick) for nick in CLASS_NICKNAMES.values())
                   for n in names)


# =============================================================================
# Review follow-ups: the ORDER CHECK reply never doubles a launch; quoted
# ship refs and admiral maneuver enums
# =============================================================================

class TestOrderCheckReply:

    def test_a_repeated_turn_does_not_launch_twice(self, tmp_path):
        def order(ship):
            return {"order_text": "Launch on the destroyer.", "priority": "NORMAL",
                    "suggested_target": "beta_1"}

        def captain(ctx):
            if ctx.model != "stub/dart" or ctx.cp != 1:
                return []
            # Off the ordered target, with a launch - twice (a model repeating its turn).
            return [("set_primary_target", {"target_id": "beta_2"}),
                    ("launch_torpedo", {"count": 1, "target_id": "beta_2"})]
        stub = Stub(captain=captain, order=order)
        runner = make_runner(stub, tmp_path, beta=(("destroyer", "stub/lancer"),
                                                    ("destroyer", "stub/lancer")),
                             checkpoints=1, distance_km=400.0)
        runner.run_fleet_battle(FLEET)
        assert stub.turns[("stub/dart", 1)] == 2               # the ORDER CHECK was asked
        captain = runner.alpha_captains["alpha_1"]
        executed = [tc.name for tc in captain.last_tool_calls]      # requested calls are recorded
        assert executed.count("launch_torpedo") == 1, executed

    def test_reply_round_drops_repeats_and_launches(self):
        sim, cap = _sim("corvette"), _captain("corvette")
        cap.config.has_torpedoes = True
        first = [ToolCall(id="1", name="set_primary_target", arguments={"target_id": "beta_2"}),
                 ToolCall(id="2", name="launch_torpedo", arguments={"count": 1, "target_id": "beta_2"})]
        commands, executed, state = [], [], {"maneuver_issued": False}
        cap._execute_round(first, sim, "alpha_1", commands, executed, state)
        n = len(commands)
        cap._apply_order_check_reply(list(first), ["target: ordered x, engaging y"], sim,
                                     "alpha_1", commands, executed, state)
        assert len(commands) == n                      # nothing doubled
        assert sum(1 for c in commands if c.get("type") == "launch_torpedo") == 1

    def test_quoted_ship_refs_resolve(self):
        sim, cap = _sim("corvette"), _captain("corvette")
        cap.config.has_torpedoes = True
        cmd = _call(cap, sim, "set_primary_target", target_id='"OCS Lancer-1"')
        assert cmd == {"type": "set_target", "target_id": "beta_2"}
        m = _call(cap, sim, "set_maneuver", maneuver_type="PADLOCK", target_id="'beta_2'")
        assert m.target_id == "beta_2"

    def test_admiral_maneuver_enum_tolerates_quotes(self):
        from src.llm.maneuver_orders import admiral_maneuver_spec
        spec = admiral_maneuver_spec({"maneuver": '"PADLOCK"', "evade": '"dodge"'})
        assert spec["maneuver_type"] == "PADLOCK" and spec["evade"] == "dodge"
