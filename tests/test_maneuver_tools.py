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


# =============================================================================
# MCP set_maneuver
# =============================================================================

def _mcp_call(**arguments):
    """Dispatch one MCP set_maneuver call in shared-memory mode; returns the
    response text and the queued command (or None)."""
    import asyncio
    import mcp.types as mcp_types
    from src.llm.mcp_server import create_mcp_server
    from src.llm.mcp_state import MCPSharedState, get_mcp_state
    MCPSharedState.reset()
    state = get_mcp_state()
    state.register_faction("alpha")
    server = create_mcp_server("alpha")
    handler = server.request_handlers[mcp_types.CallToolRequest]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name="set_maneuver", arguments=arguments))
    text = asyncio.run(handler(request)).root.content[0].text
    queued = state.get_pending_commands("alpha")
    return text, (queued[0] if queued else None)


def _mcp_apply(sim, **parameters):
    from src.llm.mcp_controller import apply_mcp_commands_to_simulation
    from src.llm.mcp_state import MCPCommand, MCPCommandType
    ship_id = parameters.pop("ship_id", "alpha_1")
    cmd = MCPCommand(MCPCommandType.SET_MANEUVER, ship_id=ship_id, parameters=parameters)
    return apply_mcp_commands_to_simulation([cmd], sim, "alpha")


class TestMcpSetManeuver:

    def test_schema(self):
        from tests.test_fixes_mcp import _list_tools
        schema = _list_tools()["set_maneuver"].inputSchema
        props = schema["properties"]
        assert "HOLD_RANGE" in props["maneuver_type"]["enum"]
        for key in ("standoff_km", "arrival_closing_kps", "range_km", "evade",
                    "toward_point", "toward_ship"):
            assert key in props
        assert "NEVER BRAKES" in _list_tools()["set_maneuver"].description

    def test_plain_call_queues_the_old_parameters(self):
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="INTERCEPT", target_id="beta_1")
        assert cmd.parameters == {"maneuver_type": "INTERCEPT", "throttle": 1.0,
                                  "target_id": "beta_1", "heading_direction": None}
        assert "NOTE" not in text

    def test_new_parameters_ride_along_and_are_validated(self):
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="INTERCEPT", target_id="beta_1",
                              standoff_km=50, evade="dodge")
        assert cmd.parameters["standoff_km"] == 50 and cmd.parameters["evade"] == "dodge"
        assert "standoff_km=50" in text
        # Schema validation (minimum 0) or our own check - either way nothing queues.
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="INTERCEPT", standoff_km=-1)
        assert cmd is None and ("ERROR" in text or "minimum" in text)
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="HEADING",
                              toward_point={"x": 1, "y": 2})
        assert cmd is None
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="HEADING")
        assert "HEADING needs" in text and cmd is None

    def test_position_like_heading_direction_gets_a_note(self):
        text, cmd = _mcp_call(ship_id="alpha_1", maneuver_type="HEADING",
                              heading_direction={"x": 90, "y": 55, "z": 10})
        assert cmd is not None and "use toward_point" in text

    def test_apply_standoff_by_name_and_hold_range_default(self):
        sim = _sim()
        result = _mcp_apply(sim, maneuver_type="INTERCEPT", target_id="Falchion-1",
                            standoff_km=50, throttle=1.0)
        assert result["errors"] == []
        m = sim.get_ship("alpha_1").current_maneuver
        assert m.target_id == "beta_1" and m.standoff_m == 50_000.0
        result = _mcp_apply(sim, maneuver_type="HOLD_RANGE", target_id="beta_2")
        m = sim.get_ship("alpha_1").current_maneuver
        assert m.maneuver_type == ManeuverType.HOLD_RANGE and m.hold_range_m > 100_000.0

    def test_apply_plain_command_unchanged(self):
        sim = _sim()
        _mcp_apply(sim, maneuver_type="INTERCEPT", target_id="beta_1", throttle=0.5,
                   heading_direction=None)
        m = sim.get_ship("alpha_1").current_maneuver
        assert m == Maneuver(ManeuverType.INTERCEPT, start_time=0.0, duration=30.0,
                             throttle=0.5, target_id="beta_1")

    def test_intercept_without_target_uses_the_primary_target_or_errors(self):
        sim = _sim()
        result = _mcp_apply(sim, maneuver_type="INTERCEPT")
        assert "needs target_id" in result["errors"][0]["error"]
        sim.get_ship("alpha_1").primary_target_id = "beta_2"
        result = _mcp_apply(sim, maneuver_type="INTERCEPT")
        assert result["errors"] == []
        assert sim.get_ship("alpha_1").current_maneuver.target_id == "beta_2"

    def test_apply_toward_point_and_toward_ship(self):
        sim = _sim()
        _mcp_apply(sim, maneuver_type="HEADING", toward_point={"x": 90, "y": 55, "z": 10})
        m = sim.get_ship("alpha_1").current_maneuver
        assert m.heading_point_km == {"x": 90.0, "y": 55.0, "z": 10.0}
        assert m.heading_direction["x"] < -0.98
        _mcp_apply(sim, maneuver_type="HEADING", toward_ship="beta_2", evade="full")
        m = sim.get_ship("alpha_1").current_maneuver
        assert m.heading_target_id == "beta_2" and m.evade == "full"

    def test_fleet_orders(self):
        """ship_id "all": INTERCEPT goes to every ship; HEADING toward_point
        is ONE heading from the fleet centre, flown as a formation."""
        sim = _sim()
        result = _mcp_apply(sim, ship_id="all", maneuver_type="INTERCEPT", target_id="beta_1",
                            standoff_km=40)
        assert {r["ship_id"] for r in result["applied"]} == {"alpha_1", "alpha_2"}
        assert all(sim.get_ship(s).current_maneuver.standoff_m == 40_000.0
                   for s in ("alpha_1", "alpha_2"))
        _mcp_apply(sim, ship_id="all", maneuver_type="HEADING",
                   toward_point={"x": 0, "y": 0, "z": 0}, throttle=1.0)
        a, b = (sim.get_ship(s).current_maneuver for s in ("alpha_1", "alpha_2"))
        assert a.heading_direction == b.heading_direction
        assert a.heading_point_km is None          # a fixed formation heading
        assert a.burn_start_time == b.burn_start_time
        centre = (sim.get_ship("alpha_1").position + sim.get_ship("alpha_2").position) * 0.5
        expected = (centre * -1.0).normalized()
        h = a.heading_direction
        assert (h["x"], h["y"], h["z"]) == pytest.approx(expected.to_tuple(), abs=1e-3)


# =============================================================================
# Admiral issue_order
# =============================================================================

class TestAdmiralOrders:

    def test_spec_parsing(self):
        from src.llm.maneuver_orders import admiral_maneuver_spec
        assert admiral_maneuver_spec({"ship_name": "x", "heading": {"x": 1, "y": 0, "z": 0}}) is None
        assert admiral_maneuver_spec({"evade": "full"}) is None     # nothing to fly
        spec = admiral_maneuver_spec({"maneuver": "intercept", "suggested_target": "Falchion-1",
                                      "standoff_km": "60", "arrival_closing_kps": -2,
                                      "evade": "sideways", "throttle": 0.7})
        assert spec == {"maneuver_type": "INTERCEPT", "target": "Falchion-1",
                        "standoff_km": 60.0, "throttle": 0.7}
        spec = admiral_maneuver_spec({"toward_point": {"x": 1, "y": 2, "z": 3}, "evade": "dodge"})
        assert spec["maneuver_type"] == "HEADING" and spec["evade"] == "dodge"
        assert spec["toward_point"] == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_orders_carry_the_spec_and_inherit_the_fleet_order(self):
        from src.llm.admiral import AdmiralOrder, _attach_heading, _fleet_heading_order
        fleet = _fleet_heading_order({"ship_name": "all ships", "order_text": "Close to 80.",
                                      "maneuver": "INTERCEPT", "maneuver_target": "beta_1",
                                      "standoff_km": 80})
        assert fleet.maneuver["standoff_km"] == 80.0 and fleet.heading is None
        assert _fleet_heading_order({"ship_name": "Dart-1", "maneuver": "INTERCEPT"}) is None
        plain = _fleet_heading_order({"ship_name": "all ships", "order_text": "",
                                      "heading": {"x": 1, "y": 0, "z": 0}})
        assert plain.maneuver is None and plain.heading == {"x": 1.0, "y": 0.0, "z": 0.0}
        order = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="Go.")
        _attach_heading(order, {}, fleet)
        assert order.maneuver == fleet.maneuver and order.formation is True
        own = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="Hold.")
        _attach_heading(own, {"maneuver": "HOLD_RANGE", "range_km": 300}, fleet)
        assert own.maneuver["maneuver_type"] == "HOLD_RANGE" and own.formation is False

    def _apply(self, sim, orders):
        from src.llm.battle_runner import LLMBattleRunner
        fake = SimpleNamespace(simulation=sim, config=SimpleNamespace(verbose=False))
        entries = [(sid, None, sim.get_ship(sid).faction) for sid in orders]
        LLMBattleRunner._apply_admiral_headings(fake, entries, {k: [v] for k, v in orders.items()})

    def test_runner_flies_structured_and_toward_orders(self):
        from src.llm.admiral import AdmiralOrder
        sim = _sim()
        intercept = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="A", order_text="",
                                 throttle=0.8, maneuver={"maneuver_type": "INTERCEPT",
                                                         "target": "Falchion-2",
                                                         "standoff_km": 50.0, "evade": "dodge"})
        point = AdmiralOrder(target_ship_id="alpha_2", target_ship_name="B", order_text="",
                             throttle=1.0, maneuver={"maneuver_type": "HEADING",
                                                     "toward_point": {"x": 0, "y": 0, "z": 0}})
        self._apply(sim, {"alpha_1": intercept, "alpha_2": point})
        m1 = sim.get_ship("alpha_1").current_maneuver
        assert (m1.maneuver_type, m1.target_id, m1.standoff_m, m1.evade, m1.throttle) == (
            ManeuverType.INTERCEPT, "beta_2", 50_000.0, "dodge", 0.8)
        assert intercept.applied_throttle == 0.8
        m2 = sim.get_ship("alpha_2").current_maneuver
        assert m2.heading_point_km == {"x": 0.0, "y": 0.0, "z": 0.0}
        # The captain is shown the bearing it was sent on.
        assert point.heading == m2.heading_direction and point.heading["x"] < -0.99

    def test_runner_formation_toward_point_is_one_heading(self):
        from src.llm.admiral import AdmiralOrder
        sim = _sim()
        spec = {"maneuver_type": "HEADING", "toward_point": {"x": 0, "y": 0, "z": 0},
                "evade": "full"}
        orders = {sid: AdmiralOrder(target_ship_id=sid, target_ship_name=sid, order_text="",
                                    throttle=1.0, formation=True, maneuver=dict(spec))
                  for sid in ("alpha_1", "alpha_2")}
        self._apply(sim, orders)
        a, b = (sim.get_ship(s).current_maneuver for s in ("alpha_1", "alpha_2"))
        assert a.heading_direction == b.heading_direction
        assert a.burn_start_time == b.burn_start_time and a.evade == b.evade == "full"
        assert a.heading_point_km is None

    def test_captain_sees_the_applied_maneuver_order(self):
        from src.llm.admiral import AdmiralOrder
        from src.llm.prompts import format_admiral_orders_for_captain
        order = AdmiralOrder(target_ship_id="a", target_ship_name="A", order_text="Close in.",
                             throttle=1.0, maneuver={"maneuver_type": "HOLD_RANGE",
                                                     "target": "Falchion-1", "range_km": 200.0})
        text = format_admiral_orders_for_captain([order], "")
        assert "MANEUVER ORDER (ALREADY APPLIED" in text and "hold 200 km" in text
        assert "NO set_maneuver/set_heading" in text

    def test_heuristic_captain_keeps_an_applied_maneuver_order(self):
        from src.llm.admiral import AdmiralOrder
        from src.llm.captain import LLMCaptainConfig
        from src.llm.heuristic_captain import HEURISTIC_MODEL, HeuristicCaptain
        sim = _sim()
        captain = HeuristicCaptain(LLMCaptainConfig(
            name="H", ship_name="TIS Dart-6", model=HEURISTIC_MODEL, ship_type="corvette",
            fleet_data=FLEET))
        captain.receive_admiral_orders([AdmiralOrder(
            target_ship_id="alpha_1", target_ship_name="A", order_text="Hold at 200.",
            throttle=1.0, maneuver={"maneuver_type": "HOLD_RANGE", "range_km": 200.0})])
        captain.decide("alpha_1", sim)
        names = [tc.name for tc in captain.last_tool_calls]
        assert "set_maneuver" not in names and "set_heading" not in names


def test_admiral_all_ships_standoff_intercept_end_to_end():
    """Through the real runner: the directive call orders "all ships"
    INTERCEPT with a standoff and the dodge overlay; every ship flies it at
    once and every captain is shown it as already applied."""
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner
    from src.llm.client import CallStats
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.prompts import CaptainPersonality

    seen = {"captain": []}

    class Stub:
        stats = CallStats()
        session_id = None

        def decide_with_tools(self, messages, tools, model=None, temperature=None,
                              tool_choice="auto"):
            names = {t["function"]["name"] for t in tools}
            if names == {"issue_order"}:
                return [ToolCall(id="1", name="issue_order", arguments={
                    "ship_name": "x", "order_text": "As the fleet.", "priority": "NORMAL"})]
            if "set_fleet_directive" in names:
                return [
                    ToolCall(id="1", name="set_fleet_directive",
                             arguments={"directive": "Close to 80 km and hold."}),
                    ToolCall(id="2", name="issue_order", arguments={
                        "ship_name": "all ships", "order_text": "Close to 80 km.",
                        "maneuver": "INTERCEPT", "maneuver_target": "beta_1",
                        "standoff_km": 80, "evade": "dodge", "throttle": 0.9}),
                ]
            seen["captain"].append(messages)
            return [ToolCall(id="1", name="respond_to_orders",
                             arguments={"response_type": "ACKNOWLEDGE"})]

    fleet_config = BattleFleetConfig.from_dict({
        "battle_name": "Standoff", "time_limit_s": 300.0, "decision_interval_s": 30.0,
        "initial_distance_km": 500.0,
        "alpha_fleet": {"admiral": {"model": "stub/m"}, "ships": [
            {"ship_type": "corvette", "model": "stub/m"},
            {"ship_type": "destroyer", "model": "stub/m"}]},
        "beta_fleet": {"admiral": {"model": "stub/m"}, "ships": [
            {"ship_type": "destroyer", "model": "stub/m"}]},
    })
    config = BattleConfig(initial_distance_km=500.0, time_limit_s=300.0, max_checkpoints=1,
                          verbose=False, personality_selection=False, record_battle=False,
                          parallel_llm=False, seed=7)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/m",
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(config=config, alpha_config=dummy, beta_config=dummy,
                             client=Stub(), fleet_config=fleet_config)
    runner.run_fleet_battle(FLEET)

    for sid in runner.alpha_captains:
        m = runner.simulation.get_ship(sid).current_maneuver
        assert (m.maneuver_type, m.target_id, m.standoff_m, m.evade, m.throttle) == (
            ManeuverType.INTERCEPT, "beta_1", 80_000.0, "dodge", 0.9)
    turns = [m[-1]["content"] for m in seen["captain"]]
    assert turns and all("MANEUVER ORDER (ALREADY APPLIED" in t and "standoff 80 km" in t
                         for t in turns)
