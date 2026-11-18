"""
The torpedo launch profile on every way a launch is ordered (pass 5, lane
retarget): captain launch_torpedo, admiral issue_order torpedo_profile and
set_fleet_directive salvo_plan profile (through the captain that flies the
salvo, LLM or rule-based), MCP launch_torpedo - each ends as the round's
profile in the engine. Mock clients only.
"""

import pytest

from src.llm.client import ToolCall
from src.llm.admiral import AdmiralOrder
from src.simulation import CombatSimulation, SimulationEventType

from tests.test_admiral_intel import FLEET, ScriptedClient, make_admiral, ship
from tests.test_admiral_command_interface import own_caps


def _duel():
    sim = CombatSimulation(seed=1)
    a = ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
    ship(sim, "beta_1", "battlecruiser", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    return sim, a


def _captain(order=None):
    from unittest.mock import Mock
    from src.llm.captain import LLMCaptain, LLMCaptainConfig
    cap = LLMCaptain(LLMCaptainConfig(name="C", ship_name="TIS alpha_1", ship_type="corvette",
                                      fleet_data=FLEET, has_torpedoes=True), client=Mock())
    cap.admiral_orders = [order] if order else []
    return cap


def _launch_cmds(cap, sim, args):
    out = cap._execute_tool(ToolCall("t", "launch_torpedo", args), sim, "alpha_1")
    return out if isinstance(out, list) else [out]


def _engine_profiles(sim, cmds):
    for c in cmds:
        assert sim.inject_command("alpha_1", c)
    for _ in range(3):
        sim.step()
    return [tf.torpedo.profile for tf in sim.torpedoes]


class TestCaptainLaunch:

    def test_tool_schema_offers_the_profile(self):
        from src.llm.tools import TORPEDO_TOOL
        prop = TORPEDO_TOOL["function"]["parameters"]["properties"]["profile"]
        assert prop["enum"] == ["sprint", "reserve"]

    def test_profile_reaches_the_round(self):
        sim, _ = _duel()
        cmds = _launch_cmds(_captain(), sim, {"count": 1, "target_id": "beta_1",
                                              "profile": "reserve"})
        assert cmds[0]["profile"] == "reserve"
        assert _engine_profiles(sim, cmds) == ["reserve"]
        ev = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
        assert ev[0].data["profile"] == "reserve"

    def test_no_profile_is_sprint_and_the_command_is_unchanged(self):
        sim, _ = _duel()
        cmds = _launch_cmds(_captain(), sim, {"count": 1, "target_id": "beta_1"})
        assert "profile" not in cmds[0]
        assert _engine_profiles(sim, cmds) == ["sprint"]

    def test_admiral_ordered_profile_is_the_default(self):
        sim, _ = _duel()
        order = AdmiralOrder(target_ship_id="alpha_1", target_ship_name="TIS alpha_1",
                             order_text="launch", torpedo_salvo=1, torpedo_target="beta_1",
                             torpedo_profile="reserve")
        cmds = _launch_cmds(_captain(order), sim, {"count": 1, "target_id": "beta_1"})
        assert _engine_profiles(sim, cmds) == ["reserve"]
        # an explicit call wins
        sim2, _ = _duel()
        cmds = _launch_cmds(_captain(order), sim2, {"count": 1, "target_id": "beta_1",
                                                    "profile": "sprint"})
        assert _engine_profiles(sim2, cmds) == ["sprint"]

    def test_fire_solution_compares_both_profiles(self):
        from src.llm.fire_solutions import torpedo_solution
        sim, a = _duel()
        sol = torpedo_solution(sim, a, sim.get_ship("beta_1"), FLEET)
        assert sol.vs_coasting_reserve is not None and sol.vs_coasting_reserve.hits
        assert sol.vs_coasting_reserve.tof_s > sol.vs_coasting.tof_s
        assert sol.vs_coasting_reserve.reach_km > sol.vs_coasting.reach_km
        text = sol.summary()
        assert "PROFILES sprint TOF" in text and "/ reserve TOF" in text
        assert "km abeam" in text


class TestAdmiralOrders:

    def test_salvo_plan_profile_reaches_the_order(self):
        sim = CombatSimulation(seed=1)
        for i in (1, 2):
            ship(sim, f"alpha_{i}", "corvette", "alpha", (0, 10 * i, 0))
        ship(sim, "beta_1", "battleship", "beta", (600, 0, 0), fwd=(-1, 0, 0))
        plan = [{"ship_name": "TIS alpha_1", "rounds": 2, "target": "OCS beta_1",
                 "profile": "reserve"},
                {"ship_name": "TIS alpha_2", "rounds": 1, "target": "OCS beta_1"}]
        client = ScriptedClient([([ToolCall("d", "set_fleet_directive",
                                            {"directive": "strike", "salvo_plan": plan})],
                                  "tool_calls")])
        admiral = make_admiral("alpha", client=client)
        decision = admiral.decide(simulation=sim, captains=own_caps(sim, "alpha"))
        prof = {o.target_ship_id: o.torpedo_profile for o in decision.fleet_orders}
        assert prof == {"alpha_1": "reserve", "alpha_2": None}

    def test_issue_order_schema_and_parse(self):
        from src.llm.admiral_tools import ADMIRAL_TOOLS, get_admiral_tools
        from src.llm.admiral import _profile_arg
        issue = next(t for t in ADMIRAL_TOOLS if t["function"]["name"] == "issue_order")
        assert issue["function"]["parameters"]["properties"]["torpedo_profile"]["enum"] == [
            "sprint", "reserve"]
        intent = next(t for t in get_admiral_tools(intent_only=True)
                      if t["function"]["name"] == "issue_order")
        assert "torpedo_profile" in intent["function"]["parameters"]["properties"]
        directive = next(t for t in ADMIRAL_TOOLS
                         if t["function"]["name"] == "set_fleet_directive")
        item = directive["function"]["parameters"]["properties"]["salvo_plan"]["items"]
        assert item["properties"]["profile"]["enum"] == ["sprint", "reserve"]
        assert _profile_arg("Reserve") == "reserve" and _profile_arg("x") is None

    def test_heuristic_captain_flies_the_ordered_profile(self):
        from src.llm.heuristic_captain import HeuristicCaptain
        from src.llm.captain import LLMCaptainConfig
        for profile in ("reserve", None):
            sim, _ = _duel()
            cap = HeuristicCaptain(LLMCaptainConfig(name="H", ship_name="TIS alpha_1",
                                                    ship_type="corvette", fleet_data=FLEET,
                                                    has_torpedoes=True))
            cap.admiral_orders = [AdmiralOrder(
                target_ship_id="alpha_1", target_ship_name="TIS alpha_1", order_text="launch",
                torpedo_salvo=1, torpedo_target="beta_1", torpedo_profile=profile)]
            cmds = cap.decide("alpha_1", sim)
            launches = [c for c in cmds if isinstance(c, dict)
                        and c.get("type") == "launch_torpedo"]
            assert launches
            assert launches[0].get("profile") == profile


class TestMCP:

    def test_mcp_controller_passes_the_profile(self):
        from src.llm.mcp_state import MCPCommand, MCPCommandType
        from src.llm.mcp_controller import _apply_mcp_commands
        sim, _ = _duel()
        cmd = MCPCommand(command_type=MCPCommandType.LAUNCH_TORPEDO, ship_id="alpha_1",
                         parameters={"target_id": "beta_1", "profile": "reserve"},
                         timestamp=0.0)
        res = _apply_mcp_commands([cmd], sim, "alpha")
        assert res["applied"] and not res["errors"]
        for _ in range(2):
            sim.step()
        assert [tf.torpedo.profile for tf in sim.torpedoes] == ["reserve"]

    def test_mcp_tool_schema(self):
        import inspect
        from src.llm import mcp_server
        src = inspect.getsource(mcp_server)
        assert '"profile": {' in src and 'PROFILE_FACT + " " + LINK_FACT' in src


class TestWhereRoundsWent:
    """Review defect 5: the captain's launch history and the admiral's digest
    say where re-tasked rounds went and why."""

    def test_captain_history_names_the_retask(self):
        sim, a = _duel()
        from tests.test_admiral_intel import ship
        ship(sim, "beta_2", "battlecruiser", "beta", (520, 30, 0), fwd=(-1, 0, 0))
        cap = _captain()
        cap.ship_id = "alpha_1"
        cmds = _launch_cmds(cap, sim, {"count": 1, "target_id": "beta_1", "profile": "reserve"})
        sim.inject_command("alpha_1", cmds[0])
        sim.step()
        tf = sim.torpedoes[0]
        sim._reassign_torpedo(tf, sim.get_ship("beta_2"), "datalink_surplus", "g")
        cap.decision_history = [{"time": 0.0, "tool_calls": [
            {"name": "launch_torpedo", "args": {"count": 1, "target_id": "beta_1"}}]}]
        text = cap._format_decision_history(simulation=sim)
        assert "datalink re-tasked 1 from" in text and "(surplus)" in text

    def test_admiral_digest_splits_reasons(self):
        from src.llm import fleet_intel as intel
        sim, _ = _duel()
        sim._log_event(SimulationEventType.TORPEDO_RETARGETED, "alpha_1", target_id="beta_1",
                       data={"torpedo_id": "t", "reason": "target_lost"})
        sim._log_event(SimulationEventType.TORPEDO_RETARGETED, "alpha_1", target_id="beta_1",
                       data={"torpedo_id": "u", "reason": "datalink_surplus"})
        d = intel.checkpoint_digest(sim, "alpha", -1.0, until_t=1.0)
        assert d.torps["ours"]["retasked_lost"] == 1
        assert d.torps["ours"]["retasked_surplus"] == 1
