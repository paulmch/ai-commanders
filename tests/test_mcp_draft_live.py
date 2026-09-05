"""
Tests for the MCP draft phase and the live spectator endpoints.

The draft phase lets MCP clients buy fleets and place formations over the
battle HTTP API before the simulation exists; the /live endpoints let the
web viewer follow an in-progress battle. No network, no LLM: aiohttp's
in-process test client drives the real HTTP handlers.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.llm.battle_recorder import BattleEvent, BattleRecorder
from src.llm.battle_runner import BattleConfig, LLMBattleRunner
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import (
    BattleFleetConfig,
    DraftConfig,
    FleetDefinition,
    ShipConfig,
)
from src.llm.fleet_draft import SHIP_POINT_COSTS
from src.llm.mcp_draft import DraftManager
from src.llm.mcp_http_server import MCPHttpServer, run_mcp_draft_phase
from src.llm.prompts import CaptainPersonality


FLEET_DATA = json.loads(
    (Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text()
)


def make_manager(budget=100, max_ships=8, mcp=frozenset({"alpha"})):
    return DraftManager(
        fleet_data=FLEET_DATA,
        budget=budget,
        max_ships=max_ships,
        initial_distance_km=500.0,
        mcp_factions=set(mcp),
    )


async def make_test_client(server: MCPHttpServer) -> TestClient:
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    return client


class TestDraftManager:

    def test_valid_selection_names_and_prices_ships(self):
        manager = make_manager()
        result, error = manager.select(
            "alpha",
            [{"ship_type": "destroyer", "count": 2},
             {"ship_type": "cruiser", "count": 1}],
            rationale="gunline",
        )
        assert error is None
        assert result["points_spent"] == 2 * SHIP_POINT_COSTS["destroyer"] + SHIP_POINT_COSTS["cruiser"]
        names = [s["ship_name"] for s in result["your_ships"]]
        assert names == ["TIS Falchion-1", "TIS Falchion-2", "TIS Bastion"]
        # Default line-abreast slots are already assigned
        offsets = [s["offset_km"] for s in result["your_ships"]]
        assert offsets[0]["y"] != offsets[1]["y"]

    def test_selection_rejects_over_budget_and_unknown_hulls(self):
        manager = make_manager(budget=20)
        _, error = manager.select("alpha", [{"ship_type": "battleship", "count": 1}])
        assert "Over budget" in error
        _, error = manager.select("alpha", [{"ship_type": "star_destroyer", "count": 1}])
        assert "Unknown hull" in error
        _, error = manager.select("alpha", [])
        assert "at least one ship" in error

    def test_selection_replaces_previous(self):
        manager = make_manager()
        manager.select("alpha", [{"ship_type": "destroyer", "count": 4}])
        result, error = manager.select("alpha", [{"ship_type": "frigate", "count": 1}])
        assert error is None
        assert result["points_spent"] == SHIP_POINT_COSTS["frigate"]
        assert len(result["your_ships"]) == 1

    def test_formation_requires_selection_then_clamps(self):
        manager = make_manager()
        _, error = manager.formation("alpha", [])
        assert "select_fleet" in error

        manager.select("alpha", [{"ship_type": "destroyer", "count": 2}])
        result, error = manager.formation(
            "alpha",
            [{"ship_name": "Falchion-1", "x_km": 999.0, "y_km": 0.0}],
            formation_name="vanguard",
        )
        assert error is None
        assert result["formation_name"] == "vanguard"
        assert any("clamped" in note for note in result["notes"])
        placed = {s["ship_name"]: s["offset_km"] for s in result["your_ships"]}
        assert placed["TIS Falchion-1"]["x"] == 150.0  # FORMATION_MAX_OFFSET_KM

    def test_commit_gating(self):
        manager = make_manager()
        ok, error = manager.commit("alpha")
        assert not ok and "select_fleet" in error

        manager.select("alpha", [{"ship_type": "corvette", "count": 1}])
        ok, error = manager.commit("alpha")
        assert ok and error is None
        assert manager.waiting_for() == ["beta"]

        # Committed drafts are frozen
        _, error = manager.select("alpha", [{"ship_type": "frigate", "count": 1}])
        assert "already committed" in error

        from src.llm.fleet_draft import auto_draft
        manager.set_full_draft("beta", auto_draft("beta", 100, 8, seed=1))
        assert manager.all_committed()

    def test_state_dict_and_live_summary(self):
        manager = make_manager(budget=60, max_ships=4)
        state = manager.state_dict("alpha")
        assert state["phase"] == "draft"
        assert state["budget"] == 60
        assert state["max_ships"] == 4
        assert "SHIP CATALOG" in state["catalog"]
        assert state["ship_costs"]["destroyer"] == SHIP_POINT_COSTS["destroyer"]
        assert state["committed"] is False

        manager.select("alpha", [{"ship_type": "destroyer", "count": 2}])
        manager.commit("alpha")
        summary = manager.live_summary()
        assert summary["alpha"]["ready"] is True
        assert summary["alpha"]["ships"] == 2
        assert summary["alpha"]["is_mcp"] is True
        assert summary["beta"]["ready"] is False


class TestDraftEndpoints:

    def test_draft_routes_without_manager_404(self):
        async def scenario():
            server = MCPHttpServer()
            client = await make_test_client(server)
            try:
                resp = await client.get("/draft/alpha")
                assert resp.status == 404
                assert "no draft phase" in (await resp.json())["error"].lower()
            finally:
                await client.close()
        asyncio.run(scenario())

    def test_full_draft_flow_over_http(self):
        async def scenario():
            server = MCPHttpServer()
            server.set_draft_manager(make_manager(budget=60))
            client = await make_test_client(server)
            try:
                # Ready before selecting is rejected with a pointer to select_fleet
                resp = await client.post("/ready/alpha")
                assert resp.status == 409
                assert "select_fleet" in (await resp.json())["error"]

                resp = await client.get("/draft/alpha")
                assert resp.status == 200
                assert resp.headers["Access-Control-Allow-Origin"] == "*"
                state = await resp.json()
                assert state["points_remaining"] == 60

                resp = await client.post("/draft/alpha/select", json={
                    "ships": [{"ship_type": "battleship", "count": 2}],
                })
                assert resp.status == 409
                assert "Over budget" in (await resp.json())["error"]

                resp = await client.post("/draft/alpha/select", json={
                    "ships": [{"ship_type": "destroyer", "count": 3}],
                    "rationale": "wall",
                })
                assert resp.status == 200
                body = await resp.json()
                assert body["points_spent"] == 3 * SHIP_POINT_COSTS["destroyer"]

                resp = await client.post("/draft/alpha/formation", json={
                    "placements": [
                        {"ship_name": "Falchion-2", "x_km": -20.0, "y_km": 0.0},
                    ],
                    "formation_name": "reserve wedge",
                })
                assert resp.status == 200

                resp = await client.post("/ready/alpha")
                assert resp.status == 200
                body = await resp.json()
                assert body["status"] == "draft_committed"
                assert body["waiting_for"] == ["beta"]

                # Status reports the draft phase
                resp = await client.get("/status")
                status = await resp.json()
                assert status["phase"] == "draft"
                assert status["waiting_for"] == ["beta"]
            finally:
                await client.close()
        asyncio.run(scenario())


def _frame(t, ships):
    return {"t": t, "ships": ships, "projectiles": [], "torpedoes": []}


def _ship_state(pos, vel, destroyed=False, dying=False):
    return {
        "pos": list(pos), "vel": list(vel), "fwd": [1, 0, 0], "thrust": 1.0,
        "maneuver": "INTERCEPT", "destroyed": destroyed, "dying": dying,
        "hull": 100.0, "armor": {"nose": 100.0},
    }


def make_live_runner():
    """A fake runner exposing exactly what the /live endpoints read."""
    recorder = BattleRecorder()
    recorder._is_recording = True
    recorder.recording.is_fleet_battle = True
    recorder.recording.battle_name = "Live Test"
    recorder.recording.sim_trace = [
        _frame(1.0, {
            "alpha_1": _ship_state([0, 0, 0], [100, 0, 0]),
            "beta_1": _ship_state([500000, 0, 0], [-50, 0, 0]),
        }),
        _frame(2.0, {
            "alpha_1": _ship_state([105, 0, 0], [110, 0, 0]),
            "beta_1": _ship_state([499950, 0, 0], [-50, 0, 0], destroyed=True),
        }),
    ]
    recorder.events = [
        BattleEvent(timestamp=0.5, event_type="battle_start"),
        BattleEvent(timestamp=1.5, event_type="shot_fired", ship_id="alpha_1"),
    ]
    return SimpleNamespace(
        recorder=recorder,
        simulation=SimpleNamespace(current_time=2.0),
        fleet_config=SimpleNamespace(decision_interval_s=30.0),
        config=SimpleNamespace(verbose=False, seed=1),
    )


class TestLiveEndpoints:

    def test_live_recording_full_and_incremental(self):
        async def scenario():
            server = MCPHttpServer()
            server.set_live_source(make_live_runner())
            server.set_battle_status("running", checkpoint=0)
            client = await make_test_client(server)
            try:
                resp = await client.get("/live/recording")
                assert resp.status == 200
                assert resp.headers["Access-Control-Allow-Origin"] == "*"
                body = await resp.json()

                assert body["live"]["phase"] == "battle"
                assert body["live"]["status"] == "running"
                assert body["live"]["sim_time_s"] == 2.0
                assert body["live"]["next_checkpoint_t"] == 30.0
                assert body["live"]["decision_interval_s"] == 30.0

                rec = body["recording"]
                assert rec["battle_name"] == "Live Test"
                assert rec["is_fleet_battle"] is True
                assert [f["t"] for f in rec["sim_trace"]] == [1.0, 2.0]
                assert [e["timestamp"] for e in rec["events"]] == [0.5, 1.5]
                # The saved-recording shape the viewer's loader expects
                for key in ("recording_version", "alpha_fleet", "beta_fleet",
                            "winner", "result_reason", "duration_s"):
                    assert key in rec

                resp = await client.get("/live/recording?since_t=1.0")
                body = await resp.json()
                assert [f["t"] for f in body["recording"]["sim_trace"]] == [2.0]
                assert [e["timestamp"] for e in body["recording"]["events"]] == [1.5]
                assert body["recording"]["battle_name"] == "Live Test"

                resp = await client.get("/live/recording?since_t=nonsense")
                assert resp.status == 400
            finally:
                await client.close()
        asyncio.run(scenario())

    def test_live_recording_null_before_battle(self):
        async def scenario():
            server = MCPHttpServer()  # no live source at all
            client = await make_test_client(server)
            try:
                resp = await client.get("/live/recording")
                body = await resp.json()
                assert body["recording"] is None
                assert body["live"]["phase"] == "battle"

                # With an active draft the phase flips and recording stays null
                server.set_draft_manager(make_manager())
                resp = await client.get("/live/recording")
                body = await resp.json()
                assert body["recording"] is None
                assert body["live"]["phase"] == "draft"
                assert body["live"]["draft"]["alpha"]["ready"] is False
            finally:
                await client.close()
        asyncio.run(scenario())

    def test_predictions_match_constant_acceleration(self):
        async def scenario():
            server = MCPHttpServer()
            server.set_live_source(make_live_runner())
            client = await make_test_client(server)
            try:
                resp = await client.get("/live/predictions")
                body = await resp.json()
                assert body["t"] == 2.0
                assert body["t_checkpoint"] == 30.0
                # Destroyed ships carry no prediction
                assert "beta_1" not in body["ships"]

                pred = body["ships"]["alpha_1"]
                # accel from the last two frames: (110-100)/1 = 10 m/s^2 in x
                p1, v1, a = 105.0, 110.0, 10.0
                horizon = 28.0
                assert pred["path"][0] == pytest.approx([105.0, 0.0, 0.0])
                expected_final = p1 + v1 * horizon + 0.5 * a * horizon ** 2
                assert pred["checkpoint_pos"][0] == pytest.approx(expected_final)
                assert pred["checkpoint_pos"] == pred["path"][-1]
                # Sampled every 2s plus the final point: dt=0,2,...,26 then 28
                assert len(pred["path"]) == 15
            finally:
                await client.close()
        asyncio.run(scenario())

    def test_predictions_empty_during_draft(self):
        async def scenario():
            server = MCPHttpServer()
            server.set_live_source(make_live_runner())
            server.set_draft_manager(make_manager())
            client = await make_test_client(server)
            try:
                resp = await client.get("/live/predictions")
                body = await resp.json()
                assert body == {"t": 0.0, "t_checkpoint": 0.0, "ships": {}}
            finally:
                await client.close()
        asyncio.run(scenario())


class TestDraftPhaseEndToEnd:

    def test_mcp_draft_vs_auto_draft(self):
        """Full phase: MCP alpha drafts over HTTP, beta auto-drafts."""
        async def scenario():
            cfg = BattleFleetConfig.from_dict({
                "battle_name": "Draft E2E",
                "initial_distance_km": 400.0,
                "draft": {"budget": 60, "max_ships": 4},
                "alpha_fleet": {"mcp": {"enabled": True}},
                "beta_fleet": {},
            })
            runner = SimpleNamespace(
                fleet_config=cfg,
                config=SimpleNamespace(verbose=False, seed=3),
                client=None,
            )
            server = MCPHttpServer()
            server.set_live_source(runner)
            client = await make_test_client(server)
            try:
                phase_task = asyncio.ensure_future(
                    run_mcp_draft_phase(runner, FLEET_DATA, server))
                while server._draft_manager is None:
                    await asyncio.sleep(0.01)

                resp = await client.post("/draft/alpha/select", json={
                    "ships": [{"ship_type": "destroyer", "count": 2}],
                })
                assert resp.status == 200
                resp = await client.post("/draft/alpha/formation", json={
                    "placements": [
                        {"ship_name": "Falchion-1", "x_km": 20.0, "y_km": 0.0},
                    ],
                })
                assert resp.status == 200
                resp = await client.post("/ready/alpha")
                assert (await resp.json())["status"] == "draft_committed"

                await asyncio.wait_for(phase_task, timeout=10)
            finally:
                await client.close()

            # Alpha: drafted fleet with MCP control and world positions in km
            alpha = cfg.alpha_fleet
            assert alpha.mcp is not None and alpha.mcp.enabled
            assert alpha.admiral is None
            assert [s.model for s in alpha.ships] == ["mcp", "mcp"]
            assert alpha.ships[0].ship_name == "TIS Falchion-1"
            # anchor -200 km + placed 20 km toward the enemy
            assert alpha.ships[0].position == pytest.approx(
                {"x": -180.0, "y": 0.0, "z": 0.0})

            # Beta: auto-drafted heuristic fleet, committed without a client
            beta = cfg.beta_fleet
            assert beta.ships, "auto-draft must produce ships"
            assert all(s.model == "heuristic" for s in beta.ships)
            assert all(s.position is not None for s in beta.ships)
            assert server._draft_manager.active is False
        asyncio.run(scenario())


class TestDraftConfigParsing:

    def test_draft_block_relaxes_ship_requirement(self):
        cfg = BattleFleetConfig.from_dict({
            "battle_name": "X",
            "draft": {"budget": 120, "max_ships": 6, "captain_model": "heuristic"},
            "alpha_fleet": {"mcp": {"enabled": True}},
            "beta_fleet": {},
        })
        assert cfg.draft.budget == 120
        assert cfg.draft.max_ships == 6
        assert cfg.alpha_fleet.ships == []

    def test_no_draft_still_requires_ships(self):
        with pytest.raises(ValueError):
            BattleFleetConfig.from_dict({
                "battle_name": "X",
                "alpha_fleet": {},
                "beta_fleet": {},
            })

    def test_disabled_draft_is_ignored(self):
        with pytest.raises(ValueError):
            BattleFleetConfig.from_dict({
                "battle_name": "X",
                "draft": {"enabled": False},
                "alpha_fleet": {},
                "beta_fleet": {},
            })


class TestPartialPositionDefaults:

    def test_partial_position_dict_defaults_are_km(self):
        """
        A hand-built ShipConfig with only some position keys must fall back
        to km defaults - the old metre defaults got multiplied by 1000 again
        and flew ships 1000x too far out.
        """
        def ship(faction, position):
            return ShipConfig(
                ship_id=f"{faction}_1", ship_type="destroyer",
                model="heuristic", position=position,
                captain_name="Cap", ship_name=f"{faction.upper()} Test",
            )

        cfg = BattleFleetConfig(
            battle_name="pos-test",
            alpha_fleet=FleetDefinition(
                ships=[ship("alpha", {"y": 10.0})], faction="alpha"),
            beta_fleet=FleetDefinition(
                ships=[ship("beta", {"z": -5.0})], faction="beta"),
            initial_distance_km=500.0,
        )
        dummy = LLMCaptainConfig(name="-", ship_name="-", model="heuristic",
                                 personality=CaptainPersonality.BALANCED)
        runner = LLMBattleRunner(
            config=BattleConfig(
                initial_distance_km=500.0, verbose=False,
                personality_selection=False, record_battle=False, seed=1),
            alpha_config=dummy, beta_config=dummy,
            client=object(), fleet_config=cfg,
        )
        runner.setup_fleet_battle(FLEET_DATA)

        alpha_pos = runner.alpha_ships["alpha_1"].position
        assert alpha_pos.x == pytest.approx(-250_000.0)  # -half_dist in m
        assert alpha_pos.y == pytest.approx(10_000.0)    # 10 km
        beta_pos = runner.beta_ships["beta_1"].position
        assert beta_pos.x == pytest.approx(250_000.0)
        assert beta_pos.z == pytest.approx(-5_000.0)
