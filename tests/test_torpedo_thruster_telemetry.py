"""
Torpedo thruster telemetry: each torpedo remembers what its motors did on
the last tick (main throttle + thrust/nose direction, lateral RCS thrust
vector) and the sim trace records it per torpedo per frame as `thr`, `fwd`
and `rcs` for the replay viewer's plume, attitude and RCS puffs.

The load-bearing property: the bookkeeping is write-only. Guidance and
physics never read it, it draws no randomness, and a seeded battle replays
bit-for-bit with it stubbed out.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from src.llm.battle_recorder import BattleRecorder
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation, Maneuver, ManeuverType, TorpedoInFlight, create_ship_from_fleet_data,
)
from src.torpedo import GuidanceCommand, GuidanceMode, Torpedo, TorpedoGuidance, TorpedoSpecs

ROOT = Path(__file__).parent.parent
FLEET_DATA = json.loads((ROOT / "data" / "fleet_ships.json").read_text())


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torpedo(**kw):
    return Torpedo(specs=TorpedoSpecs.trident(), position=Vector3D(0, 0, 0),
                   velocity=Vector3D(1000, 0, 0), target_id="t", **kw)


class TestTorpedoBookkeeping:

    def test_main_burn_records_throttle_and_unit_direction(self):
        t = _torpedo()
        t.begin_thruster_tick()
        t.apply_thrust(Vector3D(0, 3, 4), 1.0, throttle=0.4)
        assert t.last_main_throttle == pytest.approx(0.4)
        assert t.last_main_direction.to_tuple() == pytest.approx((0, 0.6, 0.8))
        assert t.last_rcs_thrust.magnitude == 0.0

    def test_rcs_burn_records_direction_times_throttle(self):
        t = _torpedo()
        t.begin_thruster_tick()
        t.apply_lateral_thrust(Vector3D(0, 0, -2), 1.0, throttle=0.25)
        assert t.last_rcs_thrust.to_tuple() == pytest.approx((0, 0, -0.25))
        assert t.last_main_throttle == 0.0 and t.last_main_direction is None

    def test_begin_tick_clears_and_update_starts_a_fresh_tick(self):
        t = _torpedo()
        t.apply_thrust(Vector3D(1, 0, 0), 1.0)
        t.apply_lateral_thrust(Vector3D(0, 1, 0), 1.0, 0.5)
        t.begin_thruster_tick()
        assert (t.last_main_throttle, t.last_main_direction, t.last_rcs_thrust.magnitude) == (0.0, None, 0.0)
        t.apply_thrust(Vector3D(1, 0, 0), 1.0)
        t.guidance_mode = GuidanceMode.COAST
        t.update(1.0, Vector3D(1e6, 0, 0), Vector3D(0, 0, 0))
        assert t.last_main_throttle == 0.0, "a coasting tick reports a cold engine"

    def test_burn_cut_short_by_exhaustion_counts_pro_rata(self):
        t = _torpedo()
        # Leave exactly a quarter tick of full-throttle propellant.
        flow = t.specs.thrust_n / (t.specs.exhaust_velocity_kps * 1000.0)
        t.current_mass_kg = t.specs.dry_mass_kg + 0.25 * flow
        t.begin_thruster_tick()
        t.apply_thrust(Vector3D(1, 0, 0), 1.0, throttle=1.0)
        assert t.fuel_exhausted
        assert t.last_main_throttle == pytest.approx(0.25)

    def test_telemetry_is_not_part_of_equality_or_repr(self):
        a, b = _torpedo(), _torpedo()
        a.last_main_throttle = 0.9
        a.last_rcs_thrust = Vector3D(0, 0.5, 0)
        assert a == b
        assert "last_main" not in repr(a)


def _duel_with_torpedo(seed=5):
    """A cruiser evading a Trident fired from 60 km (guidance: COLLISION)."""
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = create_ship_from_fleet_data("a", "cruiser", "alpha", FLEET_DATA,
                                    Vector3D(0, 0, 0), Vector3D(100, 50, 0), Vector3D(1, 0, 0))
    b = create_ship_from_fleet_data("b", "destroyer", "beta", FLEET_DATA,
                                    Vector3D(300e3, 40e3, 5e3), Vector3D(-200, 300, 0),
                                    Vector3D(-1, 0, 0))
    sim.add_ship(a)
    sim.add_ship(b)
    a.current_maneuver = Maneuver(ManeuverType.EVASIVE, start_time=0, duration=1000,
                                  throttle=1.0, target_id="b")
    b.current_maneuver = Maneuver(ManeuverType.INTERCEPT, start_time=0, duration=1000,
                                  throttle=0.8, target_id="a")
    torp = Torpedo(specs=TorpedoSpecs.trident(),
                   position=a.position + Vector3D(60_000, 5_000, 0),
                   velocity=a.velocity + Vector3D(-6_000, 1_500, 0),
                   target_id="a", guidance_mode=GuidanceMode.COLLISION)
    torp.armed = True
    sim.torpedoes.append(TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="b",
                                         launch_time=0.0))
    return sim, torp


class TestSimulationTicks:

    def test_each_tick_reports_exactly_the_impulse_it_applied(self):
        sim, torp = _duel_with_torpedo()
        commands = []
        orig = TorpedoGuidance._collision_course_guidance

        def spy(self, *args):
            cmd = orig(self, *args)
            commands.append(cmd)
            return cmd

        TorpedoGuidance._collision_course_guidance = spy
        try:
            seen = 0
            while sim.torpedoes and seen < 40:
                v0 = torp.velocity
                n = len(commands)
                sim.step()
                if not sim.torpedoes or len(commands) == n:
                    break
                cmd: GuidanceCommand = commands[-1]
                seen += 1
                dv = torp.velocity - v0
                if cmd.use_rcs:
                    assert torp.last_main_throttle == 0.0 and torp.last_main_direction is None
                    assert torp.last_rcs_thrust.magnitude == pytest.approx(cmd.throttle)
                    assert torp.last_rcs_thrust.normalized().dot(dv.normalized()) > 0.999999
                elif cmd.throttle > 0:
                    assert torp.last_main_throttle == pytest.approx(min(1.0, cmd.throttle))
                    d = torp.last_main_direction
                    assert d.magnitude == pytest.approx(1.0)
                    assert d.dot(dv.normalized()) > 0.999999
                    assert torp.last_rcs_thrust.magnitude == 0.0
                else:
                    assert torp.last_main_throttle == 0.0 and torp.last_rcs_thrust.magnitude == 0.0
            assert seen > 5
        finally:
            TorpedoGuidance._collision_course_guidance = orig

    def test_disabled_round_reports_cold_thrusters(self):
        sim, torp = _duel_with_torpedo()
        sim.step()
        assert torp.last_main_throttle > 0.0
        sim.torpedoes[0].is_disabled = True
        sim.step()
        assert torp.last_main_throttle == 0.0 and torp.last_main_direction is None


class TestRecorderFields:

    def _frame(self, **telemetry):
        rec = BattleRecorder.__new__(BattleRecorder)
        rec._is_recording = True
        rec.recording = type("R", (), {"sim_trace": []})()
        torp = {"id": "t", "position": (1.0, 2.0, 3.0), "velocity": (4.0, 5.0, 6.0),
                "source_ship_id": "a", "target_ship_id": "b", "dv_remaining_kps": 9.0,
                "heat_absorbed_j": 0.0, "is_disabled": False, **telemetry}
        rec.record_sim_frame(timestamp=1.0, ships={}, projectiles=[], torpedoes=[torp])
        return rec.recording.sim_trace[0]["torpedoes"][0]

    def test_burning(self):
        f = self._frame(main_throttle=0.73456, main_direction=(0.6, 0.8, 0.0001),
                        rcs_thrust=(0.0, 0.0, 0.0))
        assert f["thr"] == 0.735 and f["fwd"] == [0.6, 0.8, 0.0] and "rcs" not in f

    def test_rcs_trim_with_engine_cold(self):
        f = self._frame(main_throttle=0.0, main_direction=None, rcs_thrust=(0.0, -0.12345, 0.03))
        assert f["thr"] == 0.0 and "fwd" not in f and f["rcs"] == [0.0, -0.123, 0.03]

    def test_producers_without_telemetry_record_the_old_shape(self):
        f = self._frame()
        assert not {"thr", "fwd", "rcs"} & set(f)


def _scripted_torpedo_battle(tmp_path, name, checkpoints=4):
    gen = _load_script("generate_test_battle")
    config, a_cfg, b_cfg, a_script, b_script = gen.torpedo_strike()
    config.verbose = False
    config.max_checkpoints = checkpoints
    config.record_predictions = False
    config.recording_dir = str(tmp_path / name)
    runner = gen.ScriptedBattleRunner(
        config=config, alpha_config=a_cfg, beta_config=b_cfg, client=None,
        alpha_script=a_script, beta_script=b_script)
    result = runner.run_battle(gen.load_fleet_data())
    return json.loads(Path(result.recording_file).read_text())


class TestSeededBattle:

    def test_trace_carries_telemetry_and_bookkeeping_changes_nothing(self, tmp_path, monkeypatch):
        on = _scripted_torpedo_battle(tmp_path, "on")

        entries = [p for f in on["sim_trace"] for p in f["torpedoes"]]
        assert entries, "the scenario must put torpedoes in flight"
        assert all("thr" in p and 0.0 <= p["thr"] <= 1.0 for p in entries)
        burning = [p for p in entries if p["thr"] > 0]
        assert burning and all(abs(math.hypot(*p["fwd"]) - 1) < 3e-3 for p in burning)
        assert not any("fwd" in p for p in entries if p["thr"] == 0)
        assert all(0 < math.hypot(*p["rcs"]) <= 1.0 + 1e-3 for p in entries if "rcs" in p)
        assert not any(p["thr"] > 0 for p in entries if p["disabled"])

        # Stub the bookkeeping out entirely: the battle must not notice.
        monkeypatch.setattr(Torpedo, "begin_thruster_tick", lambda self: None)
        monkeypatch.setattr(Torpedo, "_log_main_burn", lambda self, d, t: None)
        monkeypatch.setattr(Torpedo, "_log_rcs_burn", lambda self, d, t: None)
        off = _scripted_torpedo_battle(tmp_path, "off")

        for rec in (on, off):
            rec.pop("recorded_at", None)
            rec.pop("provenance", None)
            for f in rec["sim_trace"]:
                for p in f["torpedoes"]:
                    for key in ("thr", "fwd", "rcs"):
                        p.pop(key, None)
        assert on["sim_trace"] == off["sim_trace"]
        strip = lambda r: [(e["timestamp"], e["event_type"], e["ship_id"],
                            json.dumps(e["data"], sort_keys=True)) for e in r["events"]]
        assert strip(on) == strip(off)
        assert (on["winner"], on["duration_s"]) == (off["winner"], off["duration_s"])
