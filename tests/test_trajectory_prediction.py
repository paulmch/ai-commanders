"""
Trajectory prediction (src/trajectory_prediction.py) and the sim_trace fields
it feeds: every recorded frame carries, per live ship, the predicted path to
the next checkpoint (`plan`) plus the kinematics the viewer animates with.

The load-bearing properties:
* the shadow reproduces the engine exactly when nothing random or unforeseen
  happens (steady maneuvers, deterministic evasion against known threats);
* WOBBLE (random) is predicted nominally with a sigma;
* predicting never changes the battle - no RNG draws, no shared state.
"""

import asyncio
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.physics import Vector3D
from src.projectile import KineticProjectile
from src.simulation import (
    CombatSimulation, Maneuver, ManeuverType, ProjectileInFlight,
    TorpedoInFlight, create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
from src.trajectory_prediction import (
    PLAN_SAMPLE_DT_S, TraceEnricher, next_checkpoint_time, predict_plans,
    wobble_sigma_m,
)

ROOT = Path(__file__).parent.parent
FLEET_DATA = json.loads((ROOT / "data" / "fleet_ships.json").read_text())


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_duel(a_maneuver, a_type="destroyer", b_type="cruiser", seed=5, **a_kwargs):
    """Two ships 300 km apart; beta intercepts alpha."""
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = create_ship_from_fleet_data("a", a_type, "alpha", FLEET_DATA,
                                    Vector3D(0, 0, 0), Vector3D(100, 50, 0), Vector3D(1, 0, 0))
    b = create_ship_from_fleet_data("b", b_type, "beta", FLEET_DATA,
                                    Vector3D(300e3, 40e3, 5e3), Vector3D(-200, 300, 0),
                                    Vector3D(-1, 0, 0))
    sim.add_ship(a)
    sim.add_ship(b)
    a.current_maneuver = Maneuver(a_maneuver, start_time=0, duration=1000,
                                  target_id="b", **a_kwargs)
    b.current_maneuver = Maneuver(ManeuverType.INTERCEPT, start_time=0, duration=1000,
                                  throttle=0.8, target_id="a")
    return sim


def run_to(sim, t_end):
    while sim.current_time < t_end - 1e-9:
        sim.step()


def pos(ship):
    return (ship.position.x, ship.position.y, ship.position.z)


# =============================================================================
# predict_plans
# =============================================================================

class TestPlanShape:

    def test_next_checkpoint_time(self):
        assert next_checkpoint_time(0.0, 30.0) == 30.0
        assert next_checkpoint_time(29.0, 30.0) == 30.0
        # A frame taken AT a checkpoint predicts to the following one
        assert next_checkpoint_time(30.0, 30.0) == 60.0
        assert next_checkpoint_time(59.5, 30.0) == 60.0

    def test_path_samples_now_every_dt_and_ends_exactly_at_checkpoint(self):
        sim = make_duel(ManeuverType.INTERCEPT, throttle=1.0)
        run_to(sim, 3.0)
        plans = predict_plans(sim, 30.0)
        assert set(plans) == {"a", "b"}
        plan = plans["a"]
        assert plan["t_cp"] == 30.0
        assert plan["dt"] == PLAN_SAMPLE_DT_S == 2.0
        assert plan["sigma"] == 0.0
        # t = 3, 5, ..., 29 (14 samples) + the checkpoint itself
        assert len(plan["path"]) == 15
        assert plan["path"][0] == [round(c) for c in pos(sim.ships["a"])]
        assert all(isinstance(c, int) for p in plan["path"] for c in p)

        run_to(sim, 30.0)
        assert math.dist(plan["path"][-1], pos(sim.ships["a"])) < 1.0

    def test_intermediate_samples_lie_on_the_actual_track(self):
        sim = make_duel(ManeuverType.INTERCEPT, throttle=1.0)
        run_to(sim, 4.0)
        plan = predict_plans(sim, 30.0)["a"]
        actual = {}
        while sim.current_time < 30.0 - 1e-9:
            sim.step()
            actual[round(sim.current_time, 6)] = pos(sim.ships["a"])
        for i, point in enumerate(plan["path"][1:-1], start=1):
            t = 4.0 + i * plan["dt"]
            assert math.dist(point, actual[round(t, 6)]) < 1.0

    def test_destroyed_and_dying_ships_get_no_plan(self):
        sim = make_duel(ManeuverType.INTERCEPT, throttle=1.0)
        run_to(sim, 2.0)
        sim.ships["b"].is_dying = True
        sim.ships["b"].dying_time_remaining = 20.0
        assert set(predict_plans(sim, 30.0)) == {"a"}
        sim.ships["b"].is_destroyed = True
        assert set(predict_plans(sim, 30.0)) == {"a"}


class TestSteadyManeuversPredictExactly:
    """No randomness, no combat: the shadow must BE the engine (int rounding only)."""

    @pytest.mark.parametrize("maneuver,kwargs", [
        (ManeuverType.INTERCEPT, {"throttle": 1.0}),
        (ManeuverType.BURN, {"throttle": 1.0, "direction": Vector3D(0, 1, 0)}),
        (ManeuverType.BRAKE, {"throttle": 0.8}),
        (ManeuverType.PADLOCK, {}),
        (ManeuverType.MAINTAIN, {}),
        (ManeuverType.HEADING, {"throttle": 0.7, "heading_direction": {"x": 0, "y": 0, "z": 1}}),
        (ManeuverType.ROTATE, {"direction": Vector3D(0, -1, 0)}),
    ])
    def test_checkpoint_position_matches_actual(self, maneuver, kwargs):
        sim = make_duel(maneuver, **kwargs)
        run_to(sim, 3.0)
        plans = predict_plans(sim, 30.0)
        run_to(sim, 30.0)
        for sid in ("a", "b"):
            assert math.dist(plans[sid]["path"][-1], pos(sim.ships[sid])) < 1.0, sid

    def test_beats_constant_acceleration_extrapolation_through_a_turn(self):
        """The plan follows the guidance through the turn; the old live-mode
        extrapolation (a from the last two frames) cannot."""
        sim = make_duel(ManeuverType.BURN, throttle=1.0, direction=Vector3D(0, 1, 0))
        v0 = sim.ships["a"].velocity
        sim.step()
        a = sim.ships["a"]
        acc = a.velocity - v0
        plan = predict_plans(sim, 30.0)["a"]
        h = 29.0
        guess = a.position + a.velocity * h + acc * (0.5 * h * h)
        run_to(sim, 30.0)
        assert math.dist(plan["path"][-1], pos(a)) < 1.0
        assert math.dist(guess.to_tuple(), pos(a)) > 1_000.0


class TestEvasionPrediction:

    def test_wobble_is_nominal_with_sigma(self):
        sim = make_duel(ManeuverType.EVASIVE, throttle=1.0)
        run_to(sim, 3.0)
        plans = predict_plans(sim, 30.0)
        sigma = plans["a"]["sigma"]
        assert sigma > 10.0                # a 2 g destroyer wobbling for 27 s
        assert plans["b"]["sigma"] == 0.0  # the interceptor does not wobble
        run_to(sim, 30.0)
        err = math.dist(plans["a"]["path"][-1], pos(sim.ships["a"]))
        assert err < 4.0 * sigma

    def test_wobble_sigma_scales_with_thrust_and_vanishes_without_wobble(self):
        dts = [1.0] * 20
        acc = [20.0] * 20
        alpha, omega = math.radians(0.5), math.radians(6.0)
        s1 = wobble_sigma_m(dts, acc, [True] * 20, alpha, omega)
        s2 = wobble_sigma_m(dts, [2 * a for a in acc], [True] * 20, alpha, omega)
        assert s1 > 0.0
        assert s2 == pytest.approx(2.0 * s1)
        assert wobble_sigma_m(dts, acc, [False] * 20, alpha, omega) == 0.0
        # Longer horizons disperse more
        assert wobble_sigma_m(dts[:5], acc[:5], [True] * 5, alpha, omega) < s1

    def test_evade_jinks_against_a_known_round_are_continued(self):
        """EVADE noise is keyed on (seed, ship, 8 s cycle): the shadow repeats it."""
        sim = make_duel(ManeuverType.EVASIVE, throttle=1.0)
        sim.ships["a"].point_defense = []   # no PD: the round flies unchanged
        run_to(sim, 2.0)
        a = sim.ships["a"]
        # A heavy slug from dead ahead passing 300 m off the hull in 12 s
        slug = KineticProjectile(
            mass_kg=50.0,
            position=a.position + Vector3D(120_000, 300, 0),
            velocity=a.velocity + Vector3D(-10_000, 0, 0),
        )
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="slug", projectile=slug, source_ship_id="b",
            target_ship_id="a", launch_time=sim.current_time))
        run_to(sim, 3.0)
        assert a.execution_mode == "EVADE"
        plans = predict_plans(sim, 30.0)
        modes = set()
        while sim.current_time < 30.0 - 1e-9:
            sim.step()
            modes.add(a.execution_mode)
        assert "EVADE" in modes
        # EVADE while the round is inbound, then WOBBLE: the checkpoint
        # error is the wobble part only, well inside a few sigma, and far
        # below the kilometres an EVADE jink moves the ship.
        err = math.dist(plans["a"]["path"][-1], pos(a))
        assert err < max(50.0, 4.0 * plans["a"]["sigma"])

    def test_run_from_guided_torpedo_is_continued(self):
        sim = make_duel(ManeuverType.EVASIVE, a_type="cruiser", throttle=1.0)
        sim.ships["a"].point_defense = []
        run_to(sim, 2.0)
        a = sim.ships["a"]
        torp = Torpedo(specs=TorpedoSpecs.trident(),
                       position=a.position + Vector3D(250_000, 20_000, 0),
                       velocity=a.velocity + Vector3D(-4_000, 0, 0),
                       target_id="a", guidance_mode=GuidanceMode.COLLISION)
        torp.armed = True
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="t1", torpedo=torp, source_ship_id="b",
            launch_time=sim.current_time))
        run_to(sim, 3.0)
        assert a.execution_mode == "RUN"
        plans = predict_plans(sim, 30.0)
        modes = set()
        while sim.current_time < 30.0 - 1e-9:
            sim.step()
            modes.add(a.execution_mode)
        assert modes <= {"RUN", "PRESENT"}   # deterministic reactions only
        assert plans["a"]["sigma"] == 0.0
        assert math.dist(plans["a"]["path"][-1], pos(a)) < 1.0


class TestIsolation:

    def _fingerprint(self, sim):
        ships = {}
        for sid, s in sim.ships.items():
            ks = s.kinematic_state
            ships[sid] = (
                pos(s), (ks.velocity.x, ks.velocity.y, ks.velocity.z),
                (ks.forward.x, ks.forward.y, ks.forward.z), (ks.up.x, ks.up.y, ks.up.z),
                ks.propellant_kg, s.angular_velocity_rad_s,
                None if s.rotation_state is None else (
                    s.rotation_state.phase, s.rotation_state.target_direction.to_tuple()),
                s.current_maneuver and (s.current_maneuver.maneuver_type,
                                        s.current_maneuver.start_time),
                getattr(s, "execution_mode", None), getattr(s, "applied_throttle", None),
                s.thermal_system.heatsink.current_heat_gj if s.thermal_system else None,
                {loc.value: (sec.thickness_cm, sec.chipping_fraction)
                 for loc, sec in s.armor.sections.items()} if s.armor else None,
                {slot: w.cooldown_remaining for slot, w in s.weapons.items()},
            )
        torps = [(f.torpedo_id, f.terminal_latched, f.torpedo.position.to_tuple(),
                  f.torpedo.velocity.to_tuple(), f.torpedo.remaining_delta_v_kps,
                  f.torpedo.guidance_mode, f.torpedo.estimated_target_accel.to_tuple())
                 for f in sim.torpedoes]
        projs = [(f.projectile_id, f.projectile.position.to_tuple(),
                  f.projectile.velocity.to_tuple()) for f in sim.projectiles]
        return (sim.current_time, sim.rng.getstate(), len(sim.events), ships, torps, projs,
                copy.deepcopy(sim.evidence.transitions))

    def test_prediction_mutates_nothing_and_draws_no_randomness(self):
        sim = make_duel(ManeuverType.EVASIVE, a_type="cruiser", throttle=1.0)
        run_to(sim, 2.0)
        a = sim.ships["a"]
        torp = Torpedo(specs=TorpedoSpecs.trident(),
                       position=a.position + Vector3D(60_000, 5_000, 0),
                       velocity=a.velocity + Vector3D(-6_000, 0, 0),
                       target_id="a", guidance_mode=GuidanceMode.COLLISION)
        torp.armed = True
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="t1", torpedo=torp, source_ship_id="b",
            launch_time=sim.current_time))
        slug = KineticProjectile(mass_kg=50.0,
                                 position=a.position + Vector3D(80_000, 200, 0),
                                 velocity=a.velocity + Vector3D(-10_000, 0, 0))
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="p1", projectile=slug, source_ship_id="b",
            target_ship_id="a", launch_time=sim.current_time))
        run_to(sim, 4.0)   # the torpedo is now terminal: PRESENT scoring runs
        before = self._fingerprint(sim)
        predict_plans(sim, 30.0)
        predict_plans(sim, 60.0)
        assert self._fingerprint(sim) == before

    def test_seeded_battle_identical_with_prediction_on_and_off(self, tmp_path):
        """
        End to end through the runner: a scripted gun duel (EVADE jinks,
        wobble, coilgun fire, PD) replays bit-for-bit with predictions on.
        """
        gen = _load_script("generate_test_battle")
        recordings = {}
        for on in (True, False):
            config, a_cfg, b_cfg, a_script, b_script = gen.gun_duel()
            config.verbose = False
            config.max_checkpoints = 6
            config.recording_dir = str(tmp_path / ("on" if on else "off"))
            config.record_predictions = on
            runner = gen.ScriptedBattleRunner(
                config=config, alpha_config=a_cfg, beta_config=b_cfg, client=None,
                alpha_script=a_script, beta_script=b_script)
            result = runner.run_battle(gen.load_fleet_data())
            recordings[on] = json.loads(Path(result.recording_file).read_text())

        on, off = recordings[True], recordings[False]
        assert on["decision_interval_s"] == 30.0
        planned = [s for f in on["sim_trace"] for s in f["ships"].values() if "plan" in s]
        assert planned, "predictions on must record plans"
        assert not any("plan" in s for f in off["sim_trace"] for s in f["ships"].values())
        for rec in (on, off):
            rec.pop("recorded_at")
            for f in rec["sim_trace"]:
                for s in f["ships"].values():
                    s.pop("plan", None)
        assert on["sim_trace"] == off["sim_trace"]
        strip = lambda r: [(e["timestamp"], e["event_type"], e["ship_id"],
                            json.dumps(e["data"], sort_keys=True)) for e in r["events"]]
        assert strip(on) == strip(off)
        assert (on["winner"], on["duration_s"]) == (off["winner"], off["duration_s"])


# =============================================================================
# TraceEnricher (sim_trace fields)
# =============================================================================

def _entries(sim):
    return {sid: {"thrust": getattr(s, "applied_throttle", 0.0) if not s.is_dying
                  else s.dying_thrust} for sid, s in sim.ships.items()}


class TestTraceEnricher:

    def test_fields_and_units(self):
        sim = make_duel(ManeuverType.BURN, throttle=1.0, direction=Vector3D(0, 1, 0))
        enricher = TraceEnricher(30.0)
        for _ in range(4):
            sim.step()
            entries = _entries(sim)
            enricher.enrich(sim, entries)
        a = entries["a"]
        ship = sim.ships["a"]
        ks = ship.kinematic_state
        # Thrust acceleration along the nose: F/m (full throttle, healthy engines)
        expected = ks.thrust_n / ks.mass_kg
        assert math.hypot(*a["acc"]) == pytest.approx(expected, rel=1e-3)
        assert Vector3D(*a["acc"]).normalized().dot(ship.forward) > 0.999
        # Turning 90 deg onto +y with the torch lit: thrust vectoring, about +z
        assert a["turn"] == "tv"
        assert a["omega"][2] > 0 and abs(a["omega"][0]) < 1e-9
        assert a["evade"] is None and a["tgt"] == "b"
        assert a["plan"]["t_cp"] == 30.0

    def test_omega_matches_the_rotation_between_frames(self):
        sim = make_duel(ManeuverType.ROTATE, direction=Vector3D(0, -1, 0))
        enricher = TraceEnricher(30.0, predict=False)
        sim.step()
        enricher.enrich(sim, _entries(sim))
        f0 = sim.ships["a"].forward
        sim.step()
        entries = _entries(sim)
        enricher.enrich(sim, entries)
        f1 = sim.ships["a"].forward
        a = entries["a"]
        assert math.hypot(*a["omega"]) == pytest.approx(f0.angle_to(f1), abs=2e-5)
        assert f0.angle_to(f1) > 1e-4
        assert a["omega"][2] < 0          # nose swinging from +x toward -y
        assert a["turn"] == "rcs"          # ROTATE never lights the torch
        assert "plan" not in a             # predict=False

    def test_dying_hulk_tumbles_without_plan_and_destroyed_ship_is_bare(self):
        sim = make_duel(ManeuverType.INTERCEPT, throttle=1.0)
        enricher = TraceEnricher(30.0)
        sim.step()
        b = sim.ships["b"]
        b.is_dying = True
        b.dying_time_remaining = 20.0
        b.dying_tumble_axis = Vector3D(0, 0, 1)
        b.dying_tumble_rate_rad_s = 0.1
        sim.step()
        entries = _entries(sim)
        enricher.enrich(sim, entries)
        assert "plan" not in entries["b"]
        assert entries["b"]["omega"] == [0.0, 0.0, 0.1]
        assert entries["b"]["turn"] is None and entries["b"]["tgt"] is None
        assert "plan" in entries["a"]

        b.is_destroyed = True
        entries = _entries(sim)
        enricher.enrich(sim, entries)
        assert entries["b"] == {"thrust": entries["b"]["thrust"]}

    def test_wobbling_ship_reports_mode_target_and_sigma(self):
        sim = make_duel(ManeuverType.EVASIVE, throttle=1.0)
        sim.ships["a"].current_maneuver.target_id = None
        enricher = TraceEnricher(30.0)
        for _ in range(3):
            sim.step()
            entries = _entries(sim)
            enricher.enrich(sim, entries)
        a = entries["a"]
        assert a["evade"] == "wobble"
        assert a["tgt"] == "b"            # the nearest enemy it homes on
        assert a["plan"]["sigma"] > 0.0


# =============================================================================
# Live /predictions serves the recorded plan
# =============================================================================

def _live_runner(with_plan):
    from src.llm.battle_recorder import BattleRecorder
    recorder = BattleRecorder()
    recorder._is_recording = True
    recorder.recording.is_fleet_battle = True
    ship = {"pos": [105.0, 0.0, 0.0], "vel": [110.0, 0.0, 0.0], "destroyed": False,
            "dying": False}
    if with_plan:
        ship["plan"] = {"t_cp": 30.0, "dt": 2.0, "sigma": 42.5,
                        "path": [[105, 0, 0], [400, 10, 0], [3000, 90, 0]]}
    recorder.recording.sim_trace = [
        {"t": 1.0, "ships": {"alpha_1": {"pos": [0.0, 0.0, 0.0], "vel": [100.0, 0.0, 0.0]}}},
        {"t": 2.0, "ships": {"alpha_1": ship}},
    ]
    return SimpleNamespace(
        recorder=recorder,
        simulation=SimpleNamespace(current_time=2.0),
        fleet_config=SimpleNamespace(decision_interval_s=30.0),
        config=SimpleNamespace(verbose=False, seed=1),
    )


class TestLivePredictions:

    def _get(self, runner):
        from aiohttp.test_utils import TestClient, TestServer
        from src.llm.mcp_http_server import MCPHttpServer

        async def scenario():
            server = MCPHttpServer()
            server.set_live_source(runner)
            client = TestClient(TestServer(server.build_app()))
            await client.start_server()
            try:
                resp = await client.get("/live/predictions")
                return await resp.json()
            finally:
                await client.close()
        return asyncio.run(scenario())

    def test_endpoint_serves_the_recorded_plan(self):
        body = self._get(_live_runner(with_plan=True))
        assert body["t"] == 2.0 and body["t_checkpoint"] == 30.0
        pred = body["ships"]["alpha_1"]
        assert pred["path"] == [[105, 0, 0], [400, 10, 0], [3000, 90, 0]]
        assert pred["checkpoint_pos"] == [3000, 90, 0]
        assert pred["sigma"] == 42.5

    def test_endpoint_falls_back_to_extrapolation_without_plan(self):
        body = self._get(_live_runner(with_plan=False))
        pred = body["ships"]["alpha_1"]
        assert pred["sigma"] == 0.0
        # constant acceleration 10 m/s^2 over the 28 s horizon
        assert pred["checkpoint_pos"][0] == pytest.approx(105 + 110 * 28 + 0.5 * 10 * 28 ** 2)

    def test_fake_live_server_serves_the_recorded_plan(self):
        fake = _load_script("fake_live_server")
        frames = _live_runner(with_plan=True).recorder.recording.sim_trace
        frames.append({"t": 3.0, "ships": {"alpha_1": {
            "pos": [220.0, 0.0, 0.0], "vel": [120.0, 0.0, 0.0]}}})
        server = fake.FakeLiveServer({"sim_trace": frames, "events": []}, speed=1.0,
                                     draft_seconds=0.0, decision_interval=30.0)

        server.cursor = lambda: 2.5   # frame t=2 (with plan) is the current one
        body = json.loads(asyncio.run(server.handle_predictions(None)).text)
        pred = body["ships"]["alpha_1"]
        assert pred["path"][-1] == pred["checkpoint_pos"] == [3000, 90, 0]
        assert pred["sigma"] == 42.5
        assert body["t_checkpoint"] == 30.0

        server.cursor = lambda: 1.5   # frame t=1 has no plan: extrapolation
        body = json.loads(asyncio.run(server.handle_predictions(None)).text)
        assert body["ships"]["alpha_1"]["sigma"] == 0.0
