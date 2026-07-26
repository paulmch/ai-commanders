"""
Regression tests for audited defects in the MCP integration layer.

Covers src/llm/mcp_controller.py, src/llm/mcp_server.py,
src/llm/mcp_http_server.py and scripts/mcp_battle.py.
"""

import asyncio
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp.types as mcp_types

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    create_ship_from_fleet_data,
)
from src.firecontrol import WeaponsCommand, calculate_hit_probability
from src.thermal import RadiatorState
from src.llm.mcp_chat import AdmiralChat
from src.llm.mcp_state import (
    MCPCommand,
    MCPCommandType,
    MCPSharedState,
    get_mcp_state,
)
from src.llm.mcp_controller import (
    MCPController,
    MCPControllerConfig,
    apply_mcp_commands_to_simulation,
    compute_hit_probability,
)
from src.llm.mcp_server import (
    SharedStateProvider,
    create_mcp_server,
    generate_battle_plot,
)
from src.llm.mcp_http_server import apply_mcp_fleet_control

REPO_ROOT = Path(__file__).parent.parent


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    with open(REPO_ROOT / "data" / "fleet_ships.json") as f:
        return json.load(f)


def _make_ship(fleet_data, ship_id, ship_type, faction, x_km, forward):
    return create_ship_from_fleet_data(
        ship_id=ship_id,
        ship_type=ship_type,
        faction=faction,
        fleet_data=fleet_data,
        position=Vector3D(x_km * 1000.0, 0, 0),
        velocity=Vector3D(0, 0, 0),
        forward=forward,
    )


@pytest.fixture
def sim(fleet_data):
    """
    Alpha corvette (the only hull with a torpedo launcher) plus an alpha
    destroyer (the only hull with a spinal, i.e. non-turreted, gun), facing two
    beta destroyers.
    """
    simulation = CombatSimulation()
    alpha = _make_ship(fleet_data, "alpha_1", "corvette", "alpha", 0, Vector3D(1, 0, 0))
    alpha.name = "ISS Vanguard"
    alpha2 = _make_ship(fleet_data, "alpha_2", "destroyer", "alpha", 10, Vector3D(1, 0, 0))
    alpha2.name = "ISS Resolute"
    beta = _make_ship(fleet_data, "beta_1", "destroyer", "beta", 200, Vector3D(-1, 0, 0))
    beta.name = "Black Kite"
    beta2 = _make_ship(fleet_data, "beta_2", "destroyer", "beta", 300, Vector3D(-1, 0, 0))
    beta2.name = "Red Kite"
    for s in (alpha, alpha2, beta, beta2):
        simulation.add_ship(s)
    return simulation


def _move_to_km(ship, x_km):
    """Reposition a ship (ShipCombatState.position is a read-only property)."""
    ship.kinematic_state.position = Vector3D(x_km * 1000.0, 0, 0)


def _cmd(command_type, ship_id=None, **params):
    return MCPCommand(command_type=command_type, ship_id=ship_id, parameters=params)


def _captain_stub(ship_id, ship_type, name="Cpt. Stub"):
    return SimpleNamespace(
        ship_id=ship_id,
        config=SimpleNamespace(ship_type=ship_type, name=name),
    )


@pytest.fixture
def controller(fleet_data):
    MCPSharedState.reset()
    return MCPController(
        MCPControllerConfig(faction="alpha", name="Cmdr Test"),
        fleet_data,
        chat=AdmiralChat(),
    )


# =============================================================================
# Finding 1: fleet-level control commands must not be silent no-ops
# =============================================================================

class TestFleetControlCommands:

    def test_process_control_commands_applies_message_draw_surrender(self, controller):
        applied = controller.process_control_commands(
            [
                _cmd(MCPCommandType.SEND_MESSAGE, content="stand down"),
                _cmd(MCPCommandType.PROPOSE_DRAW),
                _cmd(MCPCommandType.SURRENDER),
            ],
            current_time=42.0,
        )
        assert controller.has_proposed_draw is True
        assert controller.has_surrendered is True
        kinds = {entry["command"] for entry in applied}
        assert kinds == {"send_message", "propose_fleet_draw", "surrender_fleet"}
        # The message must actually reach the chat system.
        history = controller.chat.get_recent_history("beta")
        assert any("stand down" in json.dumps(m) for m in history)

    def test_accept_draw_is_distinct_from_propose(self, controller):
        controller.process_control_commands(
            [_cmd(MCPCommandType.PROPOSE_DRAW, accept=True)], current_time=1.0
        )
        assert controller.has_accepted_draw is True
        assert controller.has_proposed_draw is False

    def test_apply_mcp_commands_still_ignores_control_commands(self, sim):
        """Control commands are controller state; the sim applier must not claim them."""
        results = apply_mcp_commands_to_simulation(
            [
                _cmd(MCPCommandType.SURRENDER),
                _cmd(MCPCommandType.SEND_MESSAGE, content="hi"),
            ],
            sim,
            "alpha",
        )
        assert results["applied"] == []

    def test_http_loop_surrender_marks_ships(self, sim, controller):
        controller.has_surrendered = True
        runner = SimpleNamespace(
            simulation=sim,
            config=SimpleNamespace(verbose=False),
            alpha_mcp=controller,
            beta_mcp=None,
            alpha_ships={"alpha_1": sim.get_ship("alpha_1")},
            beta_ships={"beta_1": sim.get_ship("beta_1")},
        )
        stop = apply_mcp_fleet_control(runner)
        assert stop is False
        assert sim.get_ship("alpha_1").is_surrendered is True
        assert sim.get_ship("beta_1").is_surrendered is False

    def test_http_loop_mutual_draw_stops_battle(self, sim, fleet_data):
        MCPSharedState.reset()
        chat = AdmiralChat()
        alpha = MCPController(MCPControllerConfig(faction="alpha"), fleet_data, chat)
        beta = MCPController(MCPControllerConfig(faction="beta"), fleet_data, chat)
        runner = SimpleNamespace(
            simulation=sim,
            config=SimpleNamespace(verbose=False),
            alpha_mcp=alpha,
            beta_mcp=beta,
            alpha_ships={"alpha_1": sim.get_ship("alpha_1")},
            beta_ships={"beta_1": sim.get_ship("beta_1")},
        )
        assert apply_mcp_fleet_control(runner) is False

        alpha.has_proposed_draw = True
        assert apply_mcp_fleet_control(runner) is False  # not yet accepted

        beta.has_accepted_draw = True
        assert apply_mcp_fleet_control(runner) is True


# =============================================================================
# Findings 2 / 11: displayed hit probability must be the simulation's own model
# =============================================================================

class TestHitProbability:

    def test_matches_firecontrol_model(self, sim):
        # alpha_2 is the destroyer: the only hull carrying a spinal coilgun.
        shooter = sim.get_ship("alpha_2")
        target = sim.get_ship("beta_1")
        result = compute_hit_probability(shooter, target)

        weapon_state = shooter.weapons[result["weapon_slot"]]
        expected = calculate_hit_probability(
            shooter_position=shooter.position,
            shooter_velocity=shooter.velocity,
            target_position=target.position,
            target_velocity=target.velocity,
            target_geometry=target.geometry,
            target_forward=target.forward,
            muzzle_velocity_kps=weapon_state.weapon.muzzle_velocity_kps,
            target_is_evading=False,
        )
        assert expected.hit_probability > 0, "fixture geometry must allow a shot"
        assert result["hit_chance"] == pytest.approx(100.0 * expected.hit_probability)

    def test_never_selects_point_defense_slot(self, sim):
        result = compute_hit_probability(sim.get_ship("alpha_2"), sim.get_ship("beta_1"))
        assert result["weapon_slot"] is not None
        assert not result["weapon_slot"].startswith("pd_")

    def test_prefers_the_spinal_mount_over_turrets(self, sim):
        shooter = sim.get_ship("alpha_2")
        slot = compute_hit_probability(shooter, sim.get_ship("beta_1"))["weapon_slot"]
        assert shooter.weapons[slot].weapon.is_turreted is False

    def test_decreases_with_range(self, sim):
        shooter = sim.get_ship("alpha_2")
        target = sim.get_ship("beta_1")
        probs = []
        for range_km in (100, 300, 600, 1200):
            _move_to_km(target, range_km)
            probs.append(compute_hit_probability(shooter, target)["hit_chance"])
        assert probs == sorted(probs, reverse=True)
        assert probs[0] > probs[-1]

    def test_evading_target_is_harder_to_hit(self, sim):
        shooter = sim.get_ship("alpha_2")
        target = sim.get_ship("beta_1")
        calm = compute_hit_probability(shooter, target)["hit_chance"]
        target.current_maneuver = Maneuver(
            maneuver_type=ManeuverType.EVASIVE,
            start_time=0.0,
        )
        evading = compute_hit_probability(shooter, target)["hit_chance"]
        assert evading < calm

    def test_state_hit_chance_equals_model_and_beats_old_linear_formula(self, sim, controller):
        target = sim.get_ship("beta_1")
        _move_to_km(target, 800)

        captains = [
            _captain_stub("alpha_1", "corvette"),
            _captain_stub("alpha_2", "destroyer"),
        ]
        state = controller.build_state_for_mcp(sim, captains)
        enemy = next(e for e in state.enemy_ships if e["ship_id"] == "beta_1")

        shooter = sim.get_ship(enemy["hit_chance_shooter_id"])
        model = compute_hit_probability(shooter, target)["hit_chance"]
        assert enemy["hit_chance"] == pytest.approx(model)

        # The removed formula was 100 * (1 - d/1500); at 800 km it claimed ~47%.
        old_formula = 100.0 * (1.0 - 800.0 / 1500.0)
        assert enemy["hit_chance"] < old_formula


# =============================================================================
# Finding 3 / 12: launch_torpedo must actually launch (or report an error)
# =============================================================================

class TestLaunchTorpedo:

    def test_launch_creates_a_torpedo(self, sim):
        before = len(sim.torpedoes)
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.LAUNCH_TORPEDO, "alpha_1", target_id="beta_1")],
            sim,
            "alpha",
        )
        assert results["errors"] == []
        assert len(results["applied"]) == 1
        assert len(sim.torpedoes) == before + 1

    def test_failed_launch_is_reported_as_error(self, sim):
        # The destroyer has no torpedo launcher, so this must not report success.
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.LAUNCH_TORPEDO, "beta_1", target_id="alpha_1")],
            sim,
            "beta",
        )
        assert results["applied"] == []
        assert len(results["errors"]) == 1
        assert results["errors"][0]["command"] == "launch_torpedo"


# =============================================================================
# Finding 4: FIRE_AT_RANGE needs a range or it can never fire
# =============================================================================

class TestFireAtRange:

    def test_rejected_without_max_range(self, sim):
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_WEAPONS_ORDER, "alpha_2", spinal_mode="FIRE_AT_RANGE")],
            sim,
            "alpha",
        )
        assert results["errors"], "FIRE_AT_RANGE with no range must be rejected"
        assert not sim.get_ship("alpha_2").weapons_orders

    def test_max_range_is_forwarded_to_the_order(self, sim):
        apply_mcp_commands_to_simulation(
            [
                _cmd(
                    MCPCommandType.SET_WEAPONS_ORDER,
                    "alpha_2",
                    spinal_mode="FIRE_AT_RANGE",
                    max_range_km=450.0,
                )
            ],
            sim,
            "alpha",
        )
        orders = sim.get_ship("alpha_2").weapons_orders
        assert orders
        for order in orders.values():
            assert order.command == WeaponsCommand.FIRE_AT_RANGE
            assert order.max_range_km == pytest.approx(450.0)

    def test_min_hit_probability_is_forwarded(self, sim):
        apply_mcp_commands_to_simulation(
            [
                _cmd(
                    MCPCommandType.SET_WEAPONS_ORDER,
                    "alpha_2",
                    spinal_mode="FIRE_WHEN_OPTIMAL",
                    min_hit_probability=0.6,
                )
            ],
            sim,
            "alpha",
        )
        orders = sim.get_ship("alpha_2").weapons_orders
        assert orders
        for order in orders.values():
            assert order.min_hit_probability == pytest.approx(0.6)


# =============================================================================
# Finding 5: weapons orders must follow the ship's live primary target
# =============================================================================

class TestWeaponsOrderTargeting:

    def test_order_does_not_snapshot_the_target(self, sim):
        apply_mcp_commands_to_simulation(
            [
                _cmd(MCPCommandType.SET_PRIMARY_TARGET, "alpha_2", target_id="beta_1"),
                _cmd(MCPCommandType.SET_WEAPONS_ORDER, "alpha_2", spinal_mode="FIRE_IMMEDIATE"),
            ],
            sim,
            "alpha",
        )
        ship = sim.get_ship("alpha_2")
        assert ship.primary_target_id == "beta_1"
        assert ship.weapons_orders
        # A baked-in id would survive retargeting and keep shooting at beta_1.
        for order in ship.weapons_orders.values():
            assert order.target_id is None

        apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_PRIMARY_TARGET, "alpha_2", target_id="beta_2")],
            sim,
            "alpha",
        )
        assert ship.primary_target_id == "beta_2"
        # simulation resolves `order.target_id or ship.primary_target_id`
        for order in ship.weapons_orders.values():
            assert (order.target_id or ship.primary_target_id) == "beta_2"


# =============================================================================
# Findings 6 / 18: invalid input must produce errors, not silent success
# =============================================================================

class TestValidation:

    def test_unknown_ship_id_is_an_error(self, sim):
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_MANEUVER, "nope", maneuver_type="INTERCEPT")],
            sim,
            "alpha",
        )
        assert results["applied"] == []
        assert results["errors"][0]["error"] == "Ship not found"

    def test_unknown_maneuver_name_is_an_error(self, sim):
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_MANEUVER, "alpha_1", maneuver_type="WARP_SPEED")],
            sim,
            "alpha",
        )
        assert results["applied"] == []
        assert results["errors"]

    def test_invalid_primary_target_is_rejected(self, sim):
        ship = sim.get_ship("alpha_1")
        ship.primary_target_id = "beta_1"
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_PRIMARY_TARGET, "alpha_1", target_id="ghost_9")],
            sim,
            "alpha",
        )
        assert results["applied"] == []
        assert results["errors"]
        # A bad id must not silently disable the ship's guns.
        assert ship.primary_target_id == "beta_1"

    def test_friendly_ship_is_not_a_valid_target(self, sim):
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_PRIMARY_TARGET, "alpha_1", target_id="alpha_1")],
            sim,
            "alpha",
        )
        assert results["errors"]

    def test_clearing_the_target_is_allowed(self, sim):
        ship = sim.get_ship("alpha_1")
        ship.primary_target_id = "beta_1"
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_PRIMARY_TARGET, "alpha_1", target_id="NONE")],
            sim,
            "alpha",
        )
        assert results["errors"] == []
        assert ship.primary_target_id is None


# =============================================================================
# Finding 7: shared-memory ready() must fail loudly instead of hanging
# =============================================================================

class TestSharedMemoryReady:

    def test_signal_ready_without_battle_raises(self):
        MCPSharedState.reset()
        state = get_mcp_state()
        state.register_faction("alpha")
        provider = SharedStateProvider(state)
        with pytest.raises(RuntimeError, match="No battle is running"):
            provider.signal_ready("alpha")

    def test_signal_ready_works_once_the_runner_owns_the_loop(self):
        MCPSharedState.reset()
        state = get_mcp_state()
        state.register_faction("alpha")

        async def scenario():
            state.set_event_loop(asyncio.get_running_loop())
            SharedStateProvider(state).signal_ready("alpha")
            return state.is_ready("alpha")

        assert asyncio.run(scenario()) is True


# =============================================================================
# Findings 9 / 13: battle progress must be published, not hardcoded
# =============================================================================

class TestBattleProgress:

    def test_state_reflects_published_progress(self, sim, controller):
        captains = [_captain_stub("alpha_1", "corvette")]
        default_state = controller.build_state_for_mcp(sim, captains)
        assert default_state.checkpoint_number == 0

        controller.set_battle_progress(
            checkpoint_number=7,
            is_battle_active=False,
            enemy_proposed_draw=True,
        )
        state = controller.build_state_for_mcp(sim, captains)
        assert state.checkpoint_number == 7
        assert state.is_battle_active is False
        assert state.enemy_proposed_draw is True


# =============================================================================
# Finding 14: battle_plot must report true 3D ranges
# =============================================================================

class TestBattlePlotDistance:

    def _plot(self):
        state = {
            "friendly_ships": [{
                "ship_id": "alpha_1",
                "ship_name": "Vanguard",
                "position_km": {"x": 0, "y": 0, "z": 0},
                "velocity_vector": {"x": 0, "y": 0, "z": 0},
                "hull_integrity": 100,
            }],
            "enemy_ships": [{
                "ship_id": "beta_1",
                "ship_name": "Kite",
                "position_km": {"x": 30, "y": 40, "z": 1200},
                "velocity_vector": {"x": 0, "y": 0, "z": 0},
            }],
        }
        return generate_battle_plot(state, "alpha", "xy")

    def test_out_of_plane_range_is_not_understated(self):
        plot = self._plot()
        true_range = math.sqrt(30 ** 2 + 40 ** 2 + 1200 ** 2)
        assert f"{true_range:.1f} km" in plot
        # The old code printed the 50 km in-plane projection as *the* range.
        assert "[A1] → [B1]: 50.0 km" not in plot

    def test_in_plane_value_is_labelled_when_it_differs(self):
        assert "50.0 km in-plane" in self._plot()

    def test_coplanar_engagement_has_no_redundant_annotation(self):
        state = {
            "friendly_ships": [{
                "ship_id": "a", "ship_name": "A",
                "position_km": {"x": 0, "y": 0, "z": 0},
                "velocity_vector": {"x": 0, "y": 0, "z": 0},
                "hull_integrity": 100,
            }],
            "enemy_ships": [{
                "ship_id": "b", "ship_name": "B",
                "position_km": {"x": 300, "y": 400, "z": 0},
                "velocity_vector": {"x": 0, "y": 0, "z": 0},
            }],
        }
        plot = generate_battle_plot(state, "alpha", "xy")
        assert "500.0 km" in plot
        assert "in-plane" not in plot


# =============================================================================
# Finding 16: real ship names must be exposed
# =============================================================================

class TestShipNames:

    def test_friendly_and_enemy_names_are_real_names(self, sim, controller):
        state = controller.build_state_for_mcp(sim, [_captain_stub("alpha_1", "corvette")])
        friendly = state.friendly_ships[0]
        assert friendly["ship_id"] == "alpha_1"
        assert friendly["ship_name"] == "ISS Vanguard"

        enemy = next(e for e in state.enemy_ships if e["ship_id"] == "beta_1")
        assert enemy["ship_name"] == "Black Kite"

    def test_falls_back_to_id_when_unnamed(self, sim, controller):
        del sim.get_ship("alpha_1").name
        state = controller.build_state_for_mcp(sim, [_captain_stub("alpha_1", "corvette")])
        assert state.friendly_ships[0]["ship_name"] == "alpha_1"


# =============================================================================
# Finding 20: radiators_extended must be derived from the thermal system
# =============================================================================

class TestRadiators:

    def test_state_tracks_the_thermal_system(self, sim, controller):
        ship = sim.get_ship("alpha_1")
        captains = [_captain_stub("alpha_1", "corvette")]

        ship.thermal_system.radiators.retract_all()
        state = controller.build_state_for_mcp(sim, captains)
        assert state.friendly_ships[0]["radiators_extended"] is False

        ship.thermal_system.radiators.extend_all()
        state = controller.build_state_for_mcp(sim, captains)
        assert state.friendly_ships[0]["radiators_extended"] is True

    def test_command_reports_actual_state_not_the_request(self, sim):
        ship = sim.get_ship("alpha_1")
        ship.thermal_system.radiators.retract_all()
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_RADIATORS, "alpha_1", extend=True)], sim, "alpha"
        )
        entry = results["applied"][0]
        assert entry["radiators_extended"] is True
        assert entry["radiators_affected"] >= 1
        assert any(
            rad.state == RadiatorState.EXTENDED
            for rad in ship.thermal_system.radiators.radiators.values()
        )

    def test_destroyed_radiators_cannot_be_extended(self, sim):
        ship = sim.get_ship("alpha_1")
        radiators = ship.thermal_system.radiators.radiators
        for rad in radiators.values():
            rad.state = RadiatorState.DESTROYED
        results = apply_mcp_commands_to_simulation(
            [_cmd(MCPCommandType.SET_RADIATORS, "alpha_1", extend=True)], sim, "alpha"
        )
        entry = results["applied"][0]
        # The old shadow flag reported True here regardless.
        assert entry["radiators_extended"] is False


# =============================================================================
# Findings 10 / 17 / 9: tool schema contract
# =============================================================================

def _list_tools(faction="alpha"):
    MCPSharedState.reset()
    server = create_mcp_server(faction, http_url="http://localhost:8765")
    handler = server.request_handlers[mcp_types.ListToolsRequest]
    result = asyncio.run(handler(mcp_types.ListToolsRequest(method="tools/list")))
    return {tool.name: tool for tool in result.root.tools}


class TestToolSchemas:

    def test_pd_mode_is_not_advertised(self):
        """pd_mode was accepted and silently dropped; point defense is automatic."""
        schema = _list_tools()["set_weapons_order"].inputSchema
        assert "pd_mode" not in schema["properties"]

    def test_fire_at_range_has_a_range_parameter(self):
        schema = _list_tools()["set_weapons_order"].inputSchema
        assert "max_range_km" in schema["properties"]
        assert "FIRE_AT_RANGE" in schema["properties"]["spinal_mode"]["enum"]

    def test_heading_direction_is_documented_as_world_frame(self):
        schema = _list_tools()["set_maneuver"].inputSchema
        heading = schema["properties"]["heading_direction"]
        description = heading["description"].lower()
        assert "world" in description
        # The old text told the agent +x pointed at the enemy, which is only
        # true for alpha and backwards for beta.
        for axis in ("x", "y", "z"):
            assert "toward enemy" not in heading["properties"][axis]["description"].lower()

    def test_get_status_tool_exists(self):
        assert "get_status" in _list_tools()

    def test_launch_torpedo_description_mentions_failure(self):
        description = _list_tools()["launch_torpedo"].description.lower()
        assert "launcher" in description


# =============================================================================
# Finding 15: HTTP failures must be actionable
# =============================================================================

class TestHttpErrorMessages:

    def test_connect_error_points_at_the_battle_script(self):
        import httpx
        from src.llm.mcp_server import describe_http_failure

        msg = describe_http_failure(
            "http://localhost:8765", httpx.ConnectError("nope")
        )
        assert "not reachable" in msg
        assert "mcp_battle.py" in msg

    def test_timeout_says_to_retry(self):
        import httpx
        from src.llm.mcp_server import describe_http_failure

        msg = describe_http_failure(
            "http://localhost:8765", httpx.ReadTimeout("slow")
        )
        assert "retry" in msg.lower()

    def test_tool_call_returns_text_instead_of_raising(self):
        """A dead battle server must not surface a raw httpx traceback."""
        MCPSharedState.reset()
        server = create_mcp_server("alpha", http_url="http://localhost:9")
        handler = server.request_handlers[mcp_types.CallToolRequest]
        request = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(
                name="get_battle_state", arguments={}
            ),
        )
        result = asyncio.run(handler(request))
        text = result.root.content[0].text
        assert "localhost:9" in text
        assert "Traceback" not in text


# =============================================================================
# Finding 21: stdio MCP entry point must not write to stdout
# =============================================================================

class TestMcpBattleBanner:

    def test_banner_goes_to_stderr(self, capsys, monkeypatch):
        spec = importlib.util.spec_from_file_location(
            "mcp_battle_script", REPO_ROOT / "scripts" / "mcp_battle.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        import src.llm.mcp_server as mcp_server_module

        async def fake_run_mcp_server(faction, http_url=None):
            return None

        monkeypatch.setattr(mcp_server_module, "run_mcp_server", fake_run_mcp_server)
        capsys.readouterr()  # drop anything the import printed

        module.start_mcp_server("alpha", http_url="http://localhost:8765")

        captured = capsys.readouterr()
        # stdout is the JSON-RPC transport: any byte here corrupts the handshake.
        assert captured.out == ""
        assert "Starting MCP server for alpha" in captured.err
