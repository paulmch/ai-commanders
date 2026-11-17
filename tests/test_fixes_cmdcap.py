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

    def test_evasion_holding_the_burn_is_named_with_the_threat(self):
        # Recorded (d alpha_1 T+240): "BRACE, requested 100% throttle, applied
        # 0% (turning onto the burn line, or fuel/engine limited)".
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
