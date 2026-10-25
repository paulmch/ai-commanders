"""
The maneuver TOOLS (set_maneuver / set_heading for captains, issue_order for
admirals, set_maneuver for MCP clients) expose the engine's range keeping,
live headings and evasion overlay (tests/test_maneuver_semantics.py) - and
build exactly the old maneuvers when none of the new parameters are given.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.maneuver_orders import (
    coordinate_triples,
    describe_maneuver_spec,
    parse_evade,
    position_like_direction_note,
)
from src.llm.tools import CAPTAIN_TOOLS, get_captain_tools_for_ship
from src.physics import Vector3D
from src.simulation import CombatSimulation, Maneuver, ManeuverType, create_ship_from_fleet_data

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _sim():
    sim = CombatSimulation(seed=1, time_step=1.0, decision_interval=30.0)
    for sid, name, stype, faction, pos in (
        ("alpha_1", "TIS Dart-6", "corvette", "alpha", (187, 51, 22)),
        ("alpha_2", "TIS Ward-1", "frigate", "alpha", (150, 0, 0)),
        ("beta_1", "OCS Falchion-1", "destroyer", "beta", (89, 58, 11)),
        ("beta_2", "OCS Falchion-2", "destroyer", "beta", (60, -40, 0)),
    ):
        ship = create_ship_from_fleet_data(sid, stype, faction, FLEET,
                                           position=Vector3D(*(c * 1000.0 for c in pos)),
                                           velocity=Vector3D(0, 0, 0),
                                           forward=Vector3D(-1, 0, 0))
        ship.name = name
        sim.add_ship(ship)
    return sim


def _captain():
    config = LLMCaptainConfig(name="Solo", ship_name="TIS Dart-6", ship_type="corvette",
                              fleet_data=FLEET)
    return LLMCaptain(config, client=Mock())


def _call(captain, sim, name, **args):
    return captain._execute_tool(ToolCall(id="c1", name=name, arguments=args), sim, "alpha_1")


def _tool(name):
    return next(t for t in CAPTAIN_TOOLS if t["function"]["name"] == name)["function"]


# =============================================================================
# Schemas
# =============================================================================

class TestSchemas:

    def test_set_maneuver_offers_the_new_parameters(self):
        fn = _tool("set_maneuver")
        props = fn["parameters"]["properties"]
        assert "HOLD_RANGE" in props["maneuver_type"]["enum"]
        for key in ("target_id", "standoff_km", "arrival_closing_kps", "range_km", "evade"):
            assert key in props
        assert props["evade"]["enum"] == ["full", "dodge"]
        assert fn["parameters"]["required"] == ["maneuver_type"]
        text = props["maneuver_type"]["description"]
        assert "NEVER BRAKES" in text and "THROUGH the target" in text
        assert "standoff_km" in text
        # One maneuver slot per decision, said plainly.
        assert "ONE maneuver per decision" in fn["description"]

    def test_set_heading_is_a_direction_or_a_live_bearing(self):
        fn = _tool("set_heading")
        props = fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["throttle"]
        assert {"direction", "toward_point", "toward_ship", "evade"} <= set(props)
        assert "world" in props["direction"]["description"].lower()
        assert "km" in props["toward_point"]["description"]
        assert "NOT A DESTINATION" in fn["description"]

    def test_ship_specific_tools_carry_the_same_schemas(self):
        tools = get_captain_tools_for_ship("destroyer", FLEET, has_torpedoes=False)
        names = {t["function"]["name"]: t["function"] for t in tools}
        assert names["set_maneuver"] == _tool("set_maneuver")
        assert names["set_heading"] == _tool("set_heading")


# =============================================================================
# set_maneuver
# =============================================================================

class TestSetManeuver:

    def test_plain_call_builds_the_old_maneuver(self):
        sim, captain = _sim(), _captain()
        captain.primary_target_id = "beta_2"
        m = _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT", throttle=0.6)
        assert m == Maneuver(ManeuverType.INTERCEPT, start_time=0.0, duration=30.0,
                             throttle=0.6, target_id="beta_2")
        assert not m.range_keeping
        assert captain.pending_tool_errors == []

    def test_target_id_by_name_flies_relative_to_it_without_retargeting_guns(self):
        sim, captain = _sim(), _captain()
        captain.primary_target_id = "beta_2"
        m = _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT",
                  target_id="Falchion-1", standoff_km=50, arrival_closing_kps=0.3)
        assert m.target_id == "beta_1"
        assert captain.primary_target_id == "beta_2"
        assert m.standoff_m == 50_000.0 and m.arrival_closing_mps == 300.0
        assert m.range_keeping

    def test_unknown_target_is_rejected_with_the_choices(self):
        sim, captain = _sim(), _captain()
        assert _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT",
                     target_id="Nonesuch") is None
        assert "OCS Falchion-1 [beta_1]" in captain.pending_tool_errors[-1]

    def test_hold_range_defaults_to_the_current_range(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_maneuver", maneuver_type="HOLD_RANGE", target_id="beta_1")
        expected = (sim.get_ship("beta_1").position - sim.get_ship("alpha_1").position).magnitude
        assert m.maneuver_type == ManeuverType.HOLD_RANGE
        assert m.hold_range_m == pytest.approx(expected, abs=1.0)

    def test_hold_range_takes_range_km_and_reads_standoff_as_range(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_maneuver", maneuver_type="HOLD_RANGE",
                  target_id="beta_1", range_km=200)
        assert m.hold_range_m == 200_000.0
        m = _call(captain, sim, "set_maneuver", maneuver_type="HOLD_RANGE",
                  target_id="beta_1", standoff_km=150)
        assert m.hold_range_m == 150_000.0
        assert "standoff_km as range_km" in captain.pending_tool_errors[-1]
        assert captain.pending_tool_errors[-1].startswith("NOTE (executed, not rejected)")

    def test_misplaced_parameters_are_noted_not_fatal(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_maneuver", maneuver_type="BRAKE", standoff_km=50)
        assert m.maneuver_type == ManeuverType.BRAKE and m.standoff_m is None
        assert "does not apply to BRAKE" in captain.pending_tool_errors[-1]

    def test_evade_overlay_and_bad_values(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT", evade="dodge")
        assert m.evade == "dodge"
        assert _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT",
                     evade="sideways") is None
        assert "evade must be" in captain.pending_tool_errors[-1]
        assert _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT",
                     standoff_km=-5) is None
        m = _call(captain, sim, "set_maneuver", maneuver_type="INTERCEPT", standoff_km="80 km")
        assert m.standoff_m == 80_000.0

    def test_parse_evade_accepts_booleans(self):
        assert parse_evade(True) == "full"
        assert parse_evade(False) is None
        assert parse_evade("none") is None


# =============================================================================
# set_heading
# =============================================================================

class TestSetHeading:

    def test_plain_direction_builds_the_old_maneuver(self):
        sim, captain = _sim(), _captain()
        d = {"x": 0.6, "y": -0.8, "z": 0.0}
        m = _call(captain, sim, "set_heading", direction=d, throttle=0.5)
        assert m == Maneuver(ManeuverType.HEADING, start_time=0.0, duration=30.0,
                             throttle=0.5, heading_direction=d)
        assert m.heading_direction is d

    def test_toward_point_flies_the_bearing_to_the_point(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_heading", toward_point={"x": 90, "y": 55, "z": 10},
                  throttle=1.0)
        assert m.heading_point_km == {"x": 90.0, "y": 55.0, "z": 10.0}
        h = m.heading_direction
        # The audit's expected bearing from (187,51,22) to (90,55,10).
        assert (h["x"], h["y"], h["z"]) == pytest.approx((-0.99, 0.04, -0.12), abs=0.01)

    def test_toward_ship_by_short_name(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_heading", toward_ship="falchion-2", throttle=0.0,
                  evade="full")
        assert m.heading_target_id == "beta_2" and m.evade == "full"
        assert m.heading_direction is not None

    def test_one_target_form_wins_and_the_rest_is_noted(self):
        sim, captain = _sim(), _captain()
        m = _call(captain, sim, "set_heading", direction={"x": 1, "y": 0, "z": 0},
                  toward_point={"x": 0, "y": 0, "z": 0}, throttle=1.0)
        assert m.heading_point_km == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert "ignored direction" in captain.pending_tool_errors[-1]

    def test_unknown_ship_and_bad_point_are_rejected(self):
        sim, captain = _sim(), _captain()
        assert _call(captain, sim, "set_heading", toward_ship="Ghost", throttle=1) is None
        assert _call(captain, sim, "set_heading", toward_point={"x": "far"}, throttle=1) is None
        assert len(captain.pending_tool_errors) == 2


class TestPositionLikeDirectionGuard:
    """CP-03: Dart-6 pasted the order's position into set_heading."""

    ORDER = "Dart-6: flip and INTERCEPT their cluster center (~90,55,10), torpedoes on arrival."

    def _with_order(self, text):
        captain = _captain()
        captain.receive_admiral_orders([SimpleNamespace(order_text=text)], "")
        return captain

    def test_pasted_position_is_flown_as_given_but_explained(self):
        sim = _sim()
        captain = self._with_order(self.ORDER)
        m = _call(captain, sim, "set_heading", direction={"x": 90, "y": 55, "z": 10},
                  throttle=1.0)
        assert m.heading_direction == {"x": 90, "y": 55, "z": 10}   # still a direction
        note = captain.pending_tool_errors[-1]
        assert "ACCEPTED as a DIRECTION" in note
        assert "toward_point={x: 90, y: 55, z: 10}" in note
        # The audit measured 144.5 deg off the enemy centroid; to the quoted
        # point itself it is 146 deg.
        me = sim.get_ship("alpha_1").position
        off = Vector3D(90, 55, 10).normalized().angle_to((Vector3D(90e3, 55e3, 10e3) - me).normalized())
        assert f"{math.degrees(off):.0f} deg off" in note and 140 < math.degrees(off) < 150
        captain.last_tool_errors, captain.pending_tool_errors = captain.pending_tool_errors, []
        assert "ACCEPTED as a DIRECTION" in captain._format_tool_errors()

    def test_unit_vectors_and_unrelated_vectors_are_left_alone(self):
        sim = _sim()
        captain = self._with_order(self.ORDER)
        _call(captain, sim, "set_heading", direction={"x": -0.9, "y": 0.4, "z": 0.0},
              throttle=1.0)
        _call(captain, sim, "set_heading", direction={"x": -97, "y": 4, "z": -12},
              throttle=1.0)
        assert captain.pending_tool_errors == []

    def test_no_orders_no_guard(self):
        sim, captain = _sim(), _captain()
        _call(captain, sim, "set_heading", direction={"x": 90, "y": 55, "z": 10}, throttle=1.0)
        assert captain.pending_tool_errors == []

    def test_coordinate_formats(self):
        text = ("approach X=260, Y=0, Z=0 then hold; rally (100 km, -20 km, 5 km); "
                "centre (~90,55,10)")
        assert coordinate_triples([text]) == [(100.0, -20.0, 5.0), (90.0, 55.0, 10.0),
                                              (260.0, 0.0, 0.0)]
        assert position_like_direction_note({"x": 260, "y": 0, "z": 0}, None, [text])


def test_describe_maneuver_spec():
    line = describe_maneuver_spec({"maneuver_type": "INTERCEPT", "target": "Falchion-2",
                                   "standoff_km": 50, "evade": "dodge", "throttle": 0.8})
    assert line == "INTERCEPT Falchion-2, standoff 50 km, 80% throttle, evade dodge"
    assert describe_maneuver_spec(None) is None
