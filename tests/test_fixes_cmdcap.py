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
