"""
Cross-workstream seams of pass 4 (comms x admiral intel x captain intel),
fixed while merging the parallel branches.
"""
import types

from src.llm.battle_runner import LLMBattleRunner
from src.llm.battle_runner import load_fleet_data
from src.llm.communication import FleetCommunicationChannel
from src.llm.prompts import _format_captain_reports


def _channel():
    channel = FleetCommunicationChannel()
    channel.register_ship("alpha_1", "Cpt A", "TIS Dart-1", "alpha")
    channel.register_ship("alpha_2", "Cpt B", "TIS Lancer-1", "alpha")
    channel.register_ship("beta_2", "Cpt D", "OCS Lancer-1", "beta")
    return channel


class TestLabelReferences:
    """The admiral turn state labels every ship 'NAME [id]'; references written
    that way by commanders must resolve to that ship, the id winning."""

    def test_label_resolves_by_id(self):
        assert _channel().match_ship("OCS Lancer-1 [beta_2]") == ["beta_2"]

    def test_label_id_beats_a_colliding_name(self):
        # "Lancer-1" alone is ambiguous across the factions; the id is not.
        assert _channel().match_ship("Lancer-1 [alpha_2]") == ["alpha_2"]

    def test_unknown_label_id_falls_back_to_the_name(self):
        assert _channel().match_ship("TIS Dart-1 [nope]") == ["alpha_1"]

    def test_excluded_id_is_not_returned(self):
        assert _channel().match_ship("TIS Dart-1 [alpha_1]", exclude="alpha_1") == []


class TestReportsCarryTheDiscussion:
    """discuss_with_admiral Q&A rides on the asking ship's report; the admiral
    must see it next checkpoint, so such an ACK is never collapsed."""

    @staticmethod
    def _snapshot():
        ship = types.SimpleNamespace(ship_name="TIS Dart-1", ship_id="alpha_1")
        return types.SimpleNamespace(friendly_ships=[ship], checkpoint=3,
                                     labels={"alpha_1": "TIS Dart-1 [alpha_1]"})

    def test_ack_with_discussion_is_listed_with_the_exchange(self):
        report = {"ship_name": "TIS Dart-1", "ship_id": "alpha_1", "status": "ACK",
                  "checkpoint": 2, "compliance": None,
                  "discussion": [{"question": "Hold torpedoes until 200 km?",
                                  "answer": "Yes, hold until the salvo order."}]}
        text = _format_captain_reports([report], self._snapshot())
        assert "ACK, nothing to flag" not in text
        assert "asked you: \"Hold torpedoes until 200 km?\"" in text
        assert "you said: \"Yes, hold until the salvo order.\"" in text

    def test_plain_ack_still_collapses(self):
        report = {"ship_name": "TIS Dart-1", "ship_id": "alpha_1", "status": "ACK",
                  "checkpoint": 2, "compliance": None}
        text = _format_captain_reports([report], self._snapshot())
        assert "ACK, nothing to flag (1): TIS Dart-1 [alpha_1]" in text


class TestForceRetryNamesOnlyRealTools:
    @staticmethod
    def _captain(ship_type, has_torpedoes):
        cfg = types.SimpleNamespace(ship_type=ship_type, fleet_data=load_fleet_data(),
                                    has_torpedoes=has_torpedoes)
        return types.SimpleNamespace(config=cfg)

    def test_gunless_corvette_is_not_told_to_set_weapons_order(self):
        text = LLMBattleRunner._force_retry_text(self._captain("corvette", True))
        assert "set_weapons_order" not in text
        assert "launch_torpedo" in text and "set_maneuver" in text

    def test_gun_hull_keeps_set_weapons_order(self):
        text = LLMBattleRunner._force_retry_text(self._captain("destroyer", False))
        assert "set_weapons_order" in text and "launch_torpedo" not in text


class TestManeuverTargetsAcceptLabels:
    """Admirals write maneuver targets as 'NAME [id]' (the label every turn
    state uses); battle 2026-10-25 had a PADLOCK order rejected for it."""

    def test_label_form_resolves_by_id(self):
        from tests.test_maneuver_tools import _sim
        from src.llm.maneuver_orders import resolve_ship
        sim = _sim()
        assert resolve_ship(sim, "alpha_1", "OCS Falchion-2 [beta_2]").ship_id == "beta_2"
        # The id inside the brackets wins over the name in front of it.
        assert resolve_ship(sim, "alpha_1", "OCS Falchion-1 [beta_2]").ship_id == "beta_2"
        # Unknown id: fall back to the name.
        assert resolve_ship(sim, "alpha_1", "OCS Falchion-1 [nope]").ship_id == "beta_1"
        assert resolve_ship(sim, "alpha_1", "[nope]") is None


class TestAdmiralPromptScript:
    """scripts/test_admiral_prompt.py builds snapshots by hand; it had rotted
    (FriendlyShipSnapshot gained weapons_destroyed) without anything noticing."""

    def test_script_renders_the_prompt(self):
        import importlib.util
        import io
        from contextlib import redirect_stdout
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "test_admiral_prompt.py"
        spec = importlib.util.spec_from_file_location("test_admiral_prompt_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = io.StringIO()
        with redirect_stdout(out):
            module.main()
        text = out.getvalue()
        assert "=== FLEET SITUATION === T+30s" in text
        assert "TIS Haiku-3 [alpha_3] DN" in text
        assert "=== ENEMY THREAT TABLE" in text


class TestMcpOrdnanceSources:
    """get_battle_state names who fired each inbound round. The torpedo
    source was read off the Torpedo physics object (it lives on the
    TorpedoInFlight record), so every inbound torpedo came from 'Unknown'."""

    @staticmethod
    def _sim():
        from tests.test_admiral_intel import FLEET, add_torpedo, ship
        from src.simulation import CombatSimulation, ProjectileInFlight
        from src.projectile import KineticProjectile
        from src.physics import Vector3D
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "corvette", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        add_torpedo(sim, "torp_beta_1_0001", "beta_1", "alpha_1", (200, 0, 0), (-12, 0, 0))
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="p1", projectile=KineticProjectile(
                mass_kg=50.0, velocity=Vector3D(-8000.0, 0, 0), position=Vector3D(250e3, 0, 0)),
            source_ship_id="beta_1", target_ship_id="alpha_1"))
        return sim, FLEET

    def test_inbound_torpedo_and_slug_name_their_shooter(self):
        from src.llm.mcp_controller import MCPController, MCPControllerConfig
        sim, fleet = self._sim()
        controller = MCPController(MCPControllerConfig(faction="alpha"), fleet)
        torps = controller._build_torpedo_data(sim, [{"ship_id": "alpha_1"}])
        assert [(t["source_ship"], t["source_name"]) for t in torps] == [("beta_1", "OCS beta_1")]
        slugs = controller._build_projectile_data(sim)
        assert [(p["source_ship"], p["source_name"]) for p in slugs] == [("beta_1", "OCS beta_1")]


class TestDuelShipsCarryTheirNames:
    """Every prompt labels other ships by ship.name; only the fleet setup set
    it, so a 1v1 captain's ENEMIES line read 'beta Destroyer'."""

    def test_duel_enemy_is_labelled_by_its_name(self):
        from unittest.mock import Mock
        from src.llm.battle_runner import BattleConfig
        from src.llm.captain import LLMCaptainConfig
        fleet = load_fleet_data()
        config = BattleConfig(verbose=False, record_battle=False, personality_selection=False)
        alpha = LLMCaptainConfig(name="Captain A", ship_name="TIS Anvil", fleet_data=fleet)
        beta = LLMCaptainConfig(name="Captain B", ship_name="HFS Hammer", fleet_data=fleet)
        runner = LLMBattleRunner(config, alpha, beta, client=Mock())
        runner.setup_battle(fleet)
        sim = runner.simulation
        assert sim.get_ship("alpha").name == "TIS Anvil"
        assert sim.get_ship("beta").name == "HFS Hammer"
        tactical = runner.alpha_captain._build_tactical_status(
            sim.get_ship("alpha"), sim.get_ship("beta"), sim)
        assert [e["name"] for e in tactical["enemies"]] == ["HFS Hammer"]


class TestMcpCommandFeedback:
    """An MCP commander's set_maneuver is only checked statically when it is
    queued; the engine applies it at ready(). A rejection there (INTERCEPT
    with no target, a toward_ship that does not resolve) and the engine's
    notes (HOLD_RANGE latching its range) reached only the server console -
    the commander was told 'Maneuver set'. They now ride the next state."""

    def test_rejections_and_notes_reach_the_next_state(self):
        from tests.test_maneuver_tools import FLEET, _sim
        from src.llm.mcp_controller import (
            MCPController, MCPControllerConfig, apply_mcp_commands_to_simulation)
        from src.llm.mcp_state import MCPCommand, MCPCommandType
        sim = _sim()
        results = apply_mcp_commands_to_simulation([
            MCPCommand(MCPCommandType.SET_MANEUVER, ship_id="alpha_1",
                       parameters={"maneuver_type": "INTERCEPT"}),
            MCPCommand(MCPCommandType.SET_MANEUVER, ship_id="alpha_2",
                       parameters={"maneuver_type": "HOLD_RANGE", "target_id": "beta_2"}),
        ], sim, "alpha")
        controller = MCPController(MCPControllerConfig(faction="alpha"), FLEET)
        assert controller.build_state_for_mcp(sim, []).to_dict()["command_feedback"] == []
        controller.record_command_results(results, sim.current_time)
        captains = [types.SimpleNamespace(ship_id=sid, config=types.SimpleNamespace(
            ship_type=sim.get_ship(sid).ship_type, name=sid)) for sid in ("alpha_1", "alpha_2")]
        feedback = controller.build_state_for_mcp(sim, captains).to_dict()["command_feedback"]
        rejected = [f for f in feedback if f["status"] == "rejected"]
        noted = [f for f in feedback if f["status"] == "applied"]
        assert [(f["ship_id"], f["command"]) for f in rejected] == [("alpha_1", "set_maneuver")]
        assert "needs target_id" in rejected[0]["error"]
        assert [f["ship_id"] for f in noted] == ["alpha_2"] and "km now" in noted[0]["notes"][0]
        # The next checkpoint's results replace them.
        controller.record_command_results({"applied": [], "errors": []}, 30.0)
        assert controller.build_state_for_mcp(sim, captains).to_dict()["command_feedback"] == []


class TestMcpBattlePortFlag:
    """scripts/mcp_battle.py --port was always overridden by the config's
    mcp.http_port (default 8765), so a second battle could not pick a port."""

    @staticmethod
    def _run(monkeypatch, port):
        import asyncio
        import importlib.util
        from pathlib import Path
        import src.llm.mcp_http_server as http_server
        seen = {}

        async def fake_run(runner, fleet_data, host, port):
            seen["port"] = port
            outcome = types.SimpleNamespace(value="draw")
            return types.SimpleNamespace(outcome=outcome, winner=None, reason="stub",
                                         duration_s=0.0, checkpoints_used=0)

        monkeypatch.setattr(http_server, "run_battle_with_http_server", fake_run)
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("mcp_battle_script", root / "scripts" / "mcp_battle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        asyncio.run(module.run_mcp_battle(str(root / "data" / "fleet_config_mcp_vs_mcp.json"),
                                          verbose=False, http_port=port))
        return seen["port"]

    def test_explicit_port_wins_over_the_config(self, monkeypatch, capsys):
        assert self._run(monkeypatch, 8899) == 8899

    def test_config_port_is_the_default(self, monkeypatch, capsys):
        assert self._run(monkeypatch, None) == 8765
