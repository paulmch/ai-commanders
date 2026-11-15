"""
Drive plume wash (src/drive_plume.py, pass 5).

A burning torch projects a narrow cone astern; any other ship inside takes
heat, and close in loses armour and fittings on the facing it shows the jet.
No RNG; computed from one snapshot per tick so the outcome cannot depend on
ship order; shadow simulations are inert; torpedoes are affected only behind
PLUME_AFFECTS_TORPEDOES (off by default).
"""

import json
import math
from pathlib import Path

import pytest

from src import battle_damage as bd
from src import drive_plume as dp
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.thermal import REACTOR_MELTDOWN_HEAT_FRACTION
from src.torpedo import Torpedo, TorpedoSpecs

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)

FULL = dp.PLUME_REFERENCE_JET_POWER_W


@pytest.fixture(autouse=True)
def _no_drive_interlock(monkeypatch):
    """This module pins the wash MODEL (what a ship in a friendly jet takes).
    The engine's drive interlock, which keeps a friendly out of that
    situation in the first place, is switched off here and tested in
    tests/test_drive_interlock.py."""
    monkeypatch.setattr(dp, "PLUME_INTERLOCK_ENABLED", False)


def _ship(sim, ship_id, faction, pos_km, ship_type="cruiser", forward=(1, 0, 0)):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET,
        position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(0, 0, 0), forward=Vector3D(*forward))
    sim.add_ship(ship)
    return ship


def _burn(ship, throttle=1.0):
    ship.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                     duration=1e9, throttle=throttle)


def _washes(sim):
    return [e for e in sim.events if e.event_type == SimulationEventType.PLUME_WASH]


def _line_astern(seed=5, gap_km=10.0, order=("lead", "tail", "foe")):
    """A burning cruiser with a friendly destroyer dead astern and a far enemy."""
    sim = CombatSimulation(seed=seed, time_step=1.0)
    specs = {
        "lead": ("alpha", (0, 0, 0), "cruiser", (1, 0, 0)),
        "tail": ("alpha", (-gap_km, 0, 0), "destroyer", (1, 0, 0)),
        "foe": ("beta", (2000, 0, 0), "destroyer", (-1, 0, 0)),
    }
    for sid in order:
        faction, pos, hull, fwd = specs[sid]
        _ship(sim, sid, faction, pos, hull, fwd)
    _burn(sim.ships["lead"])
    return sim


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def test_reference_jet_power_is_the_documented_300_tw():
    ship = create_ship_from_fleet_data("x", "cruiser", "alpha", FLEET)
    ks = ship.kinematic_state
    assert dp.jet_power_w(ks.thrust_n, ks.exhaust_velocity_ms, 1.0) == pytest.approx(FULL)
    assert FULL == pytest.approx(300.3e12, rel=1e-3)


def test_every_hull_mounts_the_reference_torch():
    # The plume scales with actual thrust; today all hulls share one drive.
    for hull in FLEET["ships"]:
        ks = create_ship_from_fleet_data("x", hull, "alpha", FLEET).kinematic_state
        assert dp.jet_power_w(ks.thrust_n, ks.exhaust_velocity_ms, 1.0) == pytest.approx(FULL), hull


def test_heat_rate_on_axis_follows_the_clamped_inverse_square():
    assert dp.plume_heat_rate_gj_s(FULL, 10.0, 0.0) == pytest.approx(dp.PLUME_HEAT_GJ_PER_S_AT_10KM)
    near2 = dp.PLUME_NEAR_FIELD_KM ** 2
    for d in (0.0, 2.0, 5.0, 20.0, 50.0, 100.0):
        want = dp.PLUME_HEAT_GJ_PER_S_AT_10KM * (100.0 + near2) / (d * d + near2)
        assert dp.plume_heat_rate_gj_s(FULL, d, 0.0) == pytest.approx(want)
    rates = [dp.plume_heat_rate_gj_s(FULL, d, 0.0) for d in (0, 5, 10, 20, 50, 100)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] < 100.0                      # the near field is clamped
    assert dp.plume_heat_rate_gj_s(FULL, dp.PLUME_MAX_RANGE_KM + 0.1, 0.0) == 0.0


def test_heat_rate_falls_off_with_the_angle_and_stops_at_the_cone_edge():
    on_axis = dp.plume_heat_rate_gj_s(FULL, 10.0, 0.0)
    at_core = dp.plume_heat_rate_gj_s(FULL, 10.0, dp.PLUME_CORE_HALF_ANGLE_DEG)
    assert at_core == pytest.approx(on_axis / math.e)
    angles = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0]
    rates = [dp.plume_heat_rate_gj_s(FULL, 10.0, a) for a in angles]
    assert rates == sorted(rates, reverse=True) and rates[-1] > 0.0
    assert dp.plume_heat_rate_gj_s(FULL, 10.0, dp.PLUME_MAX_HALF_ANGLE_DEG + 0.01) == 0.0


def test_heat_rate_scales_with_the_thrust_actually_produced():
    full = dp.plume_heat_rate_gj_s(FULL, 10.0, 0.0)
    assert dp.plume_heat_rate_gj_s(0.5 * FULL, 10.0, 0.0) == pytest.approx(0.5 * full)
    assert dp.plume_heat_rate_gj_s(0.0, 10.0, 0.0) == 0.0


def test_ablation_only_above_the_threshold():
    assert dp.plume_armor_ablation_cm_s(dp.PLUME_ABLATION_THRESHOLD_GJ_PER_S) == 0.0
    assert dp.plume_armor_ablation_cm_s(16.0) == pytest.approx(
        (16.0 - dp.PLUME_ABLATION_THRESHOLD_GJ_PER_S) * dp.PLUME_ARMOR_ABLATION_CM_PER_GJ)


def test_cone_geometry_inside_and_outside():
    origin, nose = Vector3D(0, 0, 0), Vector3D(1, 0, 0)
    d, angle = dp.cone_geometry(origin, nose, Vector3D(-10_000, 0, 0))
    assert d == pytest.approx(10.0) and angle == pytest.approx(0.0, abs=1e-6)
    d, angle = dp.cone_geometry(origin, nose, Vector3D(-10_000, 524.1, 0))
    assert angle == pytest.approx(3.0, abs=0.01)
    assert dp.cone_geometry(origin, nose, Vector3D(10_000, 0, 0)) is None        # ahead
    assert dp.cone_geometry(origin, nose, Vector3D(0, 10_000, 0)) is None        # abeam
    assert dp.cone_geometry(origin, nose, Vector3D(-10_000, 1_400, 0)) is None   # 8 deg off
    assert dp.cone_geometry(origin, nose, Vector3D(-101_000, 0, 0)) is None      # too far
    assert dp.cone_geometry(origin, nose, origin) is None


# ---------------------------------------------------------------------------
# The engine hook
# ---------------------------------------------------------------------------

def test_a_ship_astern_of_a_burning_torch_is_washed():
    sim = _line_astern()
    tail = sim.ships["tail"]
    nose_before = tail.armor.get_section(bd.HitLocation.NOSE).thickness_cm
    flank_before = tail.armor.get_section(bd.HitLocation.LATERAL).thickness_cm
    for _ in range(12):
        sim.step()
    events = _washes(sim)
    assert [e.data["phase"] for e in events] == ["enter", "sustain", "sustain"]
    first = events[0]
    assert first.ship_id == "tail" and first.data["source_ship_id"] == "lead"
    assert first.data["friendly"] is True and first.data["hit_location"] == "nose"
    assert first.data["distance_km"] == pytest.approx(10.0, abs=0.1)
    assert first.data["heat_rate_gj_s"] == pytest.approx(16.0, rel=0.03)
    # ~14 GJ/s for 12 s on a 320 GJ sink.
    assert 0.35 < tail.thermal_system.heat_fraction < 0.60
    # It shows the jet its bow: the nose armour burns, the flank does not.
    assert tail.armor.get_section(bd.HitLocation.NOSE).thickness_cm < nose_before - 2.0
    assert tail.armor.get_section(bd.HitLocation.LATERAL).thickness_cm == flank_before
    damaged = [e for e in sim.events if e.event_type == SimulationEventType.MOUNT_DAMAGED]
    assert damaged and all(e.data["cause"] == "plume_wash" for e in damaged)
    assert all(e.data["source_ship_id"] == "lead" for e in damaged)
    assert events[-1].data["fittings_hit"]
    stats = dp.plume_stats(sim)
    assert stats["friendly_ship_seconds"] == 12.0 and stats["enemy_ship_seconds"] == 0.0
    assert stats["friendly_heat_gj"] == pytest.approx(events[-1].data["heat_gj"], abs=40.0)


def test_sustained_wash_cripples_but_never_melts_the_reactor_by_itself():
    sim = CombatSimulation(seed=5, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0))
    tail = _ship(sim, "tail", "alpha", (-6, 0, 0), "destroyer")
    _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
    _burn(lead)
    for _ in range(40):
        sim.step()
        lead.kinematic_state.position = Vector3D(0, 0, 0)      # hold the geometry
        lead.kinematic_state.velocity = Vector3D(0, 0, 0)
    assert not tail.is_destroyed and not tail.is_dying
    ceiling = dp.PLUME_HEAT_CEILING_FRACTION
    assert ceiling < REACTOR_MELTDOWN_HEAT_FRACTION
    # (The wash stops at the ceiling; with its radiators held in - they are
    # not forced out into a jet - the ship's own idle heat creeps on top.)
    assert 1.0 < tail.thermal_system.heat_fraction <= ceiling + 1e-3
    wrecked = [e for e in sim.events if e.event_type == SimulationEventType.MOUNT_DAMAGED
               and e.data["destroyed"]]
    assert wrecked, "40 s at 6 km wrecks the fittings on the washed facing"


def test_nothing_changes_for_ships_outside_every_cone():
    def run(burning):
        sim = CombatSimulation(seed=5, time_step=1.0)
        lead = _ship(sim, "lead", "alpha", (0, 0, 0))
        wing = _ship(sim, "wing", "alpha", (-10, 12, 0), "destroyer")
        _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
        if burning:
            _burn(lead)
        for _ in range(20):
            sim.step()
        return sim, wing
    sim, wing = run(True)
    quiet_sim, quiet_wing = run(False)
    assert not _washes(sim) and getattr(sim, "_plume_state", None) is None
    assert wing.thermal_system.heat_fraction == quiet_wing.thermal_system.heat_fraction
    assert {k: s.thickness_cm for k, s in wing.armor.sections.items()} == \
           {k: s.thickness_cm for k, s in quiet_wing.armor.sections.items()}


def test_throttle_and_drive_damage_scale_the_wash_and_a_dead_drive_emits_nothing():
    def rate(throttle=1.0, engine_health=100.0):
        sim = _line_astern()
        lead = sim.ships["lead"]
        _burn(lead, throttle)
        for module in lead._get_modules_by_type("engine"):
            module.health_percent = engine_health
        sim.step()
        events = _washes(sim)
        return events[0].data["heat_rate_gj_s"] if events else 0.0
    full = rate()
    assert rate(throttle=0.5) == pytest.approx(0.5 * full, rel=0.02)
    assert rate(engine_health=60.0) == pytest.approx(0.6 * full, rel=0.02)
    assert rate(engine_health=0.0) == 0.0
    sim = _line_astern()
    sim.ships["lead"].current_maneuver = None            # coasting
    sim.step()
    assert not _washes(sim)


def test_a_destroyed_ship_emits_nothing_and_a_dying_hulks_torch_does():
    sim = _line_astern()
    lead = sim.ships["lead"]
    lead.is_destroyed = True
    lead.applied_throttle = 1.0                           # stale value
    assert dp.compute_washes(sim) == []
    lead.is_destroyed = False
    lead.is_dying = True
    lead.dying_thrust = 0.2
    washes = dp.compute_washes(sim)
    assert [(w.emitter_id, w.victim_id) for w in washes] == [("lead", "tail")]
    assert washes[0].heat_rate_gj_s == pytest.approx(0.2 * 16.0, rel=0.01)
    lead.dying_thrust = 0.0
    assert dp.compute_washes(sim) == []


def test_the_wash_does_not_depend_on_ship_order():
    """Two fleets washing each other and themselves: inserting the ships in
    the opposite order gives the same heat, armour, fittings and events."""
    def run(order):
        sim = CombatSimulation(seed=11, time_step=1.0)
        specs = {
            "a1": ("alpha", (0, 0, 0), "cruiser", (1, 0, 0)),
            "a2": ("alpha", (-8, 0.2, 0), "destroyer", (1, 0, 0)),
            "b1": ("beta", (-16, 0, 0), "cruiser", (-1, 0, 0)),     # washes nobody
            "b2": ("beta", (9, 0, 0.1), "destroyer", (1, 0, 0)),    # washes a1 and a2
        }
        for sid in order:
            faction, pos, hull, fwd = specs[sid]
            _ship(sim, sid, faction, pos, hull, fwd)
        for sid in ("a1", "b1", "b2"):
            _burn(sim.ships[sid])
        for _ in range(15):
            sim.step()
        state = {
            sid: (round(s.thermal_system.heat_fraction, 9),
                  tuple(round(sec.thickness_cm, 9) for _, sec in sorted(
                      s.armor.sections.items(), key=lambda kv: kv[0].value)),
                  tuple(round(ws.mount_health, 9) for _, ws in sorted(s.weapons.items())))
            for sid, s in sim.ships.items()
        }
        events = sorted((e.timestamp, e.ship_id, json.dumps(e.data, sort_keys=True, default=str))
                        for e in _washes(sim))
        return state, events
    forward = run(("a1", "a2", "b1", "b2"))
    backward = run(("b2", "b1", "a2", "a1"))
    assert forward == backward
    assert len(forward[1]) > 3


def test_events_are_throttled_enter_sustain_exit_with_totals():
    sim = _line_astern()
    lead = sim.ships["lead"]
    for _ in range(13):
        sim.step()
    lead.current_maneuver = None                          # torch off
    sim.step()
    sim.step()
    events = _washes(sim)
    assert [e.data["phase"] for e in events] == ["enter", "sustain", "sustain", "exit"]
    assert [e.timestamp for e in events[:3]] == [0.0, 5.0, 10.0]
    last = events[-1].data
    assert last["duration_s"] == 13.0 and last["heat_rate_gj_s"] == 0.0
    assert last["heat_gj"] == pytest.approx(dp.plume_stats(sim)["friendly_heat_gj"], abs=0.2)
    assert last["heat_gj"] > events[1].data["heat_gj"] > events[0].data["heat_gj"]
    assert last["peak_heat_rate_gj_s"] == pytest.approx(16.0, rel=0.03)
    assert sim._plume_state["contacts"] == {}


def test_seeded_runs_record_identical_wash():
    def stream():
        sim = _line_astern(seed=77, gap_km=7.0)
        for _ in range(30):
            sim.step()
        return [(e.timestamp, e.event_type.name, e.ship_id, json.dumps(e.data, sort_keys=True, default=str))
                for e in sim.events
                if e.event_type in (SimulationEventType.PLUME_WASH, SimulationEventType.MOUNT_DAMAGED)]
    first = stream()
    assert first == stream() and len(first) > 6


def test_the_wash_never_draws_from_the_battle_or_damage_rng():
    sim = _line_astern(gap_km=6.0)
    sim.ships["lead"].applied_throttle = 1.0
    battle, damage = sim.rng.getstate(), sim._damage_rng.getstate()
    for _ in range(20):
        dp.resolve_plume_wash(sim, 1.0)
    assert sim.rng.getstate() == battle and sim._damage_rng.getstate() == damage
    assert _washes(sim)


# ---------------------------------------------------------------------------
# Shadows stay inert
# ---------------------------------------------------------------------------

def test_trajectory_prediction_shadow_takes_no_wash_and_records_nothing():
    from src.trajectory_prediction import predict_plans
    sim = _line_astern()
    sim.step()                                            # the torch is lit
    tail = sim.ships["tail"]
    heat, armour = tail.thermal_system.heat_fraction, tail.armor.get_section(bd.HitLocation.NOSE).thickness_cm
    events, rng = len(sim.events), sim.rng.getstate()
    assert predict_plans(sim, 30.0)
    assert len(sim.events) == events and sim.rng.getstate() == rng
    assert tail.thermal_system.heat_fraction == heat
    assert tail.armor.get_section(bd.HitLocation.NOSE).thickness_cm == armour


def test_torpedo_forecast_shadow_switches_the_wash_off(monkeypatch):
    from src.llm import fire_solutions
    sim = CombatSimulation(seed=3, time_step=1.0)
    shooter = _ship(sim, "sh", "alpha", (0, 0, 0), "corvette")
    target = _ship(sim, "tg", "beta", (300, 0, 0), "destroyer", (1, 0, 0))   # fleeing: tail to the shooter
    seen = []
    real = dp.compute_washes
    monkeypatch.setattr(dp, "compute_washes", lambda s, *a, **k: seen.append(s) or real(s, *a, **k))
    shot = fire_solutions.fly_torpedo(sim, shooter, target, FLEET, target_evades=True)
    assert shot is not None
    assert seen == [], "the forecast's private engine must not resolve plume wash"
    # ...and the flag is what does it.
    quiet = _line_astern()
    quiet.plume_wash_enabled = False
    for _ in range(5):
        quiet.step()
    assert not _washes(quiet) and quiet.ships["tail"].thermal_system.heat_fraction < 0.01


# ---------------------------------------------------------------------------
# Torpedoes (behind the switch)
# ---------------------------------------------------------------------------

def _torpedo(sim, torpedo_id, pos_km, source="foe", target="lead"):
    torp = Torpedo(specs=TorpedoSpecs(), position=Vector3D(*(c * 1000.0 for c in pos_km)),
                   velocity=Vector3D(0, 0, 0), target_id=target)
    torp.armed = True
    flight = TorpedoInFlight(torpedo_id=torpedo_id, torpedo=torp,
                             source_ship_id=source, launch_time=sim.current_time)
    sim.torpedoes.append(flight)
    return flight


def test_torpedoes_ignore_the_plume_by_default():
    assert dp.PLUME_AFFECTS_TORPEDOES is False
    sim = _line_astern()
    sim.ships["lead"].applied_throttle = 1.0
    flight = _torpedo(sim, "t1", (-8, 0, 0))
    for _ in range(10):
        dp.resolve_plume_wash(sim, 1.0)
    assert flight in sim.torpedoes and not flight.is_disabled
    assert dp.plume_stats(sim)["torpedoes_blinded"] == 0


def test_switch_on_a_torpedo_in_the_exhaust_is_blinded_then_destroyed(monkeypatch):
    monkeypatch.setattr(dp, "PLUME_AFFECTS_TORPEDOES", True)
    sim = _line_astern()
    lead = sim.ships["lead"]
    lead.applied_throttle = 1.0
    inbound = _torpedo(sim, "t_in", (-30, 0, 0))                # 2.2 GJ/s
    abeam = _torpedo(sim, "t_abeam", (0, 30, 0))
    own = _torpedo(sim, "t_own", (-8, 0, 0), source="lead", target="foe")
    dp.resolve_plume_wash(sim, 1.0)
    assert inbound.is_disabled and inbound.blast_blinded and inbound in sim.torpedoes
    blinded = [e for e in _washes(sim) if e.data.get("phase") == "torpedo_blinded"]
    assert len(blinded) == 1 and blinded[0].data["torpedo_id"] == "t_in"
    assert blinded[0].ship_id == "lead"
    for _ in range(4):
        dp.resolve_plume_wash(sim, 1.0)
    assert inbound not in sim.torpedoes
    killed = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_INTERCEPTED]
    assert len(killed) == 1 and killed[0].data["cause"] == "plume_wash"
    assert killed[0].data["plume_ship_id"] == "lead"
    # Outside the cone, and the emitter's own round: untouched.
    assert abeam in sim.torpedoes and not abeam.is_disabled
    assert own in sim.torpedoes and not own.is_disabled
    stats = dp.plume_stats(sim)
    assert (stats["torpedoes_blinded"], stats["torpedoes_destroyed"]) == (1, 1)


# ---------------------------------------------------------------------------
# Commander views
# ---------------------------------------------------------------------------

def test_status_lines_for_the_washed_ship_and_the_emitter():
    sim = _line_astern(gap_km=12.0)
    sim.step()
    lead, tail = sim.ships["lead"], sim.ships["tail"]
    lead.name, tail.name = "TIS Sovereign", "TIS Dart"
    line, dose = dp.plume_status_lines(sim, tail)
    assert dose.startswith("CREW DOSE 0.1 Sv from drive plumes (impaired at 2, incapacitated at 8)")
    assert line.startswith("PLUME WASH: in TIS Sovereign's exhaust (friendly), 12 km astern")
    rate = dp.compute_washes(sim)[0].heat_rate_gj_s
    assert line.endswith(f"+{rate:.1f} GJ/s")
    (mine,) = dp.plume_status_lines(sim, lead)
    assert mine.startswith("PLUME WASH: your exhaust is washing friendly TIS Dart, 12 km astern")
    assert dp.plume_status_lines(sim, sim.ships["foe"]) == []


def test_status_warns_of_the_risk_before_the_burn_and_names_an_enemy_torch():
    sim = _line_astern(gap_km=12.0)
    lead, tail, foe = sim.ships["lead"], sim.ships["tail"], sim.ships["foe"]
    lead.current_maneuver = None
    sim.step()                                            # coasting: nobody washed
    assert dp.plume_status_lines(sim, tail) == []
    (risk,) = dp.plume_status_lines(sim, lead)
    assert risk.startswith("PLUME RISK: friendly tail is 12 km astern inside your exhaust cone")
    # A friendly far astern (a wash under PLUME_RISK_MIN_GJ_PER_S) is not worth a line.
    tail.kinematic_state.position = Vector3D(-80_000, 0, 0)
    assert dp.plume_heat_rate_gj_s(FULL, 80.0, 0.0) < dp.PLUME_RISK_MIN_GJ_PER_S
    assert dp.plume_status_lines(sim, lead) == []
    tail.kinematic_state.position = Vector3D(-12_000, 0, 0)
    # An enemy that turns tail and burns 20 km ahead of the lead washes it.
    foe.kinematic_state.position = Vector3D(20_000, 0, 0)
    foe.kinematic_state.forward = Vector3D(1, 0, 0)
    foe.applied_throttle = 1.0
    lines = dp.plume_status_lines(sim, lead)
    assert any(l.startswith("PLUME WASH: in foe's exhaust (ENEMY), 20 km astern") for l in lines)
    assert dp.plume_status_lines(sim, foe) == []          # washing an enemy is not a warning


def test_captain_and_admiral_status_render_the_warning():
    from src.llm import prompts
    line = "PLUME WASH: in TIS Sovereign's exhaust (friendly), 12 km astern, 0 deg off its axis, +11.9 GJ/s"
    text = prompts.format_my_ship(
        "Vex", "TIS Dart", "destroyer",
        {"hull_integrity": 100, "plume_wash": [line]}, {}, FLEET)
    assert "  " + line in text.splitlines()
    assert "PLUME" not in prompts.format_my_ship(
        "Vex", "TIS Dart", "destroyer", {"hull_integrity": 100}, {}, FLEET)


def test_mcp_and_admiral_snapshots_carry_the_warning():
    import inspect
    from src.llm import admiral, mcp_controller
    assert "plume_wash" in {f.name for f in __import__("dataclasses").fields(admiral.FriendlyShipSnapshot)}
    assert '"plume_wash": _plume_status_lines(simulation, ship)' in inspect.getsource(
        mcp_controller.MCPController._build_friendly_ship_data)
    sim = _line_astern(gap_km=12.0)
    sim.step()
    assert admiral._plume_status_lines(sim, sim.ships["tail"]) == dp.plume_status_lines(sim, sim.ships["tail"])
    assert mcp_controller._plume_status_lines(sim, sim.ships["tail"])


# ---------------------------------------------------------------------------
# Formations
# ---------------------------------------------------------------------------

def test_formation_conflicts_find_a_follower_in_the_leaders_exhaust():
    line_astern = [(0.0, 0.0, 0.0), (-10.0, 0.0, 0.0), (-10.0, 12.0, 0.0)]
    assert dp.formation_plume_conflicts(line_astern) == [(0, 1)]
    assert dp.formation_plume_conflicts([(0.0, 0.0, 0.0), (0.0, 8.0, 0.0)]) == []
    assert dp.formation_plume_conflicts([(0.0, 0.0, 0.0), (-80.0, 0.0, 0.0)]) == []   # beyond the avoid range


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("budget", [100, 200, 300])
def test_auto_draft_formations_keep_every_ship_out_of_friendly_exhaust(seed, budget):
    from src.llm.fleet_draft import auto_draft
    draft = auto_draft("alpha", budget=budget, seed=seed)
    assert dp.formation_plume_conflicts([s.offset_km for s in draft.ships]) == []


# ---------------------------------------------------------------------------
# The rule-based layer keeps friendlies out of the jet
# ---------------------------------------------------------------------------

def test_plume_safe_throttle_eases_the_drive_for_a_friendly_astern():
    from src.llm import heuristic_tactics as T
    sim = _line_astern(gap_km=30.0)
    lead, tail = sim.ships["lead"], sim.ships["tail"]
    eased = T.plume_safe_throttle(sim, lead, 1.0)
    assert 0.0 < eased < 1.0
    full = dp.plume_heat_rate_gj_s(FULL, 30.0, 0.0)
    assert eased * full <= T.PLUME_SAFE_WASH_GJ_PER_S + 0.02
    # The wash it would then put on the friendly is under what radiators reject.
    lead.applied_throttle = eased
    (wash,) = dp.compute_washes(sim)
    assert wash.heat_rate_gj_s <= 0.78
    # Unchanged: a torpedo dodge, a ship that is not burning, the follower
    # (nobody astern of it), an enemy astern, a friendly abeam or far astern.
    assert T.plume_safe_throttle(sim, lead, 1.0, threat=True) == 1.0
    assert T.plume_safe_throttle(sim, lead, 0.0) == 0.0
    assert T.plume_safe_throttle(sim, tail, 1.0) == 1.0
    tail.faction = "beta"
    assert T.plume_safe_throttle(sim, lead, 1.0) == 1.0
    tail.faction = "alpha"
    tail.kinematic_state.position = Vector3D(-30_000, 12_000, 0)
    assert T.plume_safe_throttle(sim, lead, 1.0) == 1.0
    tail.kinematic_state.position = Vector3D(-(dp.PLUME_AVOID_RANGE_KM + 5) * 1000.0, 0, 0)
    assert T.plume_safe_throttle(sim, lead, 1.0) == 1.0


def test_plume_safe_throttle_coasts_when_close_and_checks_an_ordered_heading():
    from src.llm import heuristic_tactics as T
    sim = _line_astern(gap_km=5.0)
    lead, tail = sim.ships["lead"], sim.ships["tail"]
    assert T.plume_safe_throttle(sim, lead, 1.0) == 0.0          # even 5 % would cook it
    # Friendly abeam: clear on the present nose, but a HEADING order that
    # points the nose away from it would put it dead astern.
    tail.kinematic_state.position = Vector3D(0, -30_000, 0)
    assert T.plume_safe_throttle(sim, lead, 1.0) == 1.0
    assert T.plume_safe_throttle(sim, lead, 1.0, direction={"x": 0.0, "y": 1.0, "z": 0.0}) < 1.0
    assert T.plume_safe_throttle(sim, lead, 1.0, direction={"x": 0.0, "y": -1.0, "z": 0.0}) == 1.0


def _decide(doctrine, gap_km=30.0):
    from src.llm.captain import LLMCaptainConfig
    from src.llm.heuristic_captain import HEURISTIC_MODEL, HeuristicCaptain
    sim = CombatSimulation(seed=9, time_step=1.0)
    _ship(sim, "lead", "alpha", (0, 0, 0), "destroyer")
    _ship(sim, "tail", "alpha", (-gap_km, 0, 0), "destroyer")
    _ship(sim, "foe", "beta", (600, 0, 0), "destroyer", (-1, 0, 0))
    config = LLMCaptainConfig("C", "lead", model=HEURISTIC_MODEL, ship_type="destroyer",
                              fleet_data=FLEET)
    captain = HeuristicCaptain(config, doctrine=doctrine)
    found = [c for c in captain.decide("lead", sim) if isinstance(c, Maneuver)]
    assert found
    return found[-1]


def test_tactical_captain_eases_its_burn_and_legacy_does_not():
    from src.llm import heuristic_tactics as T
    from src.llm.heuristic_captain import DOCTRINE_LEGACY, DOCTRINE_TACTICAL
    tactical = _decide(DOCTRINE_TACTICAL)
    assert tactical is not None and tactical.throttle > 0.0
    full = dp.plume_heat_rate_gj_s(FULL, 30.0, 0.0)
    if tactical.maneuver_type == ManeuverType.HEADING:
        # Side-slipped (heuristic_tactics.plume_clear_heading): full burn on a
        # heading whose exhaust line clears the friendly astern.
        import math
        h = tactical.heading_direction
        n = math.sqrt(h["x"] ** 2 + h["y"] ** 2 + h["z"] ** 2)
        assert h["x"] / n >= math.cos(math.radians(46))            # still toward the enemy
        off_deg = math.degrees(math.acos(max(-1.0, min(1.0, h["x"] / n))))
        assert off_deg > dp.PLUME_AVOID_HALF_ANGLE_DEG              # jet off the friendly
    else:
        assert tactical.throttle < 1.0
        assert tactical.throttle * full <= T.PLUME_SAFE_WASH_GJ_PER_S + 0.02
    assert _decide(DOCTRINE_TACTICAL, gap_km=dp.PLUME_AVOID_RANGE_KM + 20).throttle == 1.0
    assert _decide(DOCTRINE_LEGACY).throttle == 1.0             # the pass-4 captain, bit for bit


def test_apply_formation_moves_a_follower_out_of_the_leaders_exhaust():
    from src.llm.fleet_draft import apply_formation, name_drafted_ships
    ships = name_drafted_ships(["battleship", "destroyer", "cruiser"], "alpha")
    notes = apply_formation(ships, [
        {"ship_name": ships[0].ship_name, "x_km": 0, "y_km": 0, "z_km": 0},
        {"ship_name": ships[1].ship_name, "x_km": -20, "y_km": 0, "z_km": 0},     # line astern
        {"ship_name": ships[2].ship_name, "x_km": -10, "y_km": 15, "z_km": 0},    # already clear
    ])
    offsets = [s.offset_km for s in ships]
    assert dp.formation_plume_conflicts(offsets) == []
    assert offsets[0] == (0.0, 0.0, 0.0) and offsets[2] == (-10.0, 15.0, 0.0)
    assert offsets[1][0] == -20.0 and offsets[1][1] != 0.0
    assert any(ships[1].ship_name in n and "drive plumes" in n for n in notes)


def test_plume_safe_throttle_sees_a_friendly_it_is_about_to_overtake():
    """Seed-5 free battle, t = 510 s: a destroyer burned past a friendly
    6 km ahead and washed it from 2-5 km (56 GJ in 4 s). The snapshot check
    sees nobody astern; the extrapolation over the decision interval does."""
    from src.llm import heuristic_tactics as T
    sim = CombatSimulation(seed=5, time_step=1.0)
    me = _ship(sim, "me", "alpha", (0, 0, 0), "destroyer")
    ahead = _ship(sim, "ahead", "alpha", (6, 0.3, 0), "destroyer")      # coasting, just off the nose
    _ship(sim, "foe", "beta", (900, 0, 0), "destroyer", (-1, 0, 0))
    assert dp.friendlies_in_exhaust(sim, me) == []                       # nobody astern now
    clearance = dp.exhaust_clearance_km(sim, me, None, 30.0)
    assert clearance is not None and clearance < 6.0                     # ...but 2 g for 30 s gains 8.8 km
    assert T.plume_safe_throttle(sim, me, 1.0) < 0.1
    # A friendly it cannot reach in the interval, or one burning along with
    # it, does not hold it back.
    ahead.kinematic_state.position = Vector3D(25_000, 300, 0)
    assert dp.exhaust_clearance_km(sim, me, None, 30.0) is None
    assert T.plume_safe_throttle(sim, me, 1.0) == 1.0
    ahead.kinematic_state.position = Vector3D(6_000, 300, 0)
    ahead.applied_throttle = 1.0
    assert T.plume_safe_throttle(sim, me, 1.0) == 1.0


def test_compute_washes_agrees_with_cone_geometry():
    sim = CombatSimulation(seed=1, time_step=1.0)
    spots = [(0, 0, 0), (-10, 0.5, 0), (-40, -3, 1), (-10, 12, 0), (30, 0, 0), (-99, 0, 0), (-101, 0, 0)]
    for i, pos in enumerate(spots):
        s = _ship(sim, f"s{i}", "alpha" if i % 2 else "beta", pos, "destroyer")
        s.applied_throttle = 1.0 if i in (0, 4) else 0.0
    got = {(w.emitter_id, w.victim_id): w for w in dp.compute_washes(sim)}
    want = {}
    for e in (sim.ships["s0"], sim.ships["s4"]):
        for v in sim.ships.values():
            hit = None if v is e else dp.cone_geometry(e.position, e.forward, v.position)
            if hit and dp.plume_heat_rate_gj_s(FULL, *hit) > 0.0:
                want[(e.ship_id, v.ship_id)] = hit
    assert set(got) == set(want) and len(got) >= 4
    for key, (d_km, angle) in want.items():
        assert got[key].distance_km == pytest.approx(d_km) and got[key].off_axis_deg == pytest.approx(angle)
        assert got[key].heat_rate_gj_s == pytest.approx(dp.plume_heat_rate_gj_s(FULL, d_km, angle))


# ---------------------------------------------------------------------------
# A sweeping jet (owner's reference: a fleeing ship whipping its plume across
# a pursuer) and the crew's radiation dose
# ---------------------------------------------------------------------------

def test_sweep_axes_fill_the_sector_between_two_ticks():
    assert dp.sweep_axes(None, (1.0, 0.0, 0.0)) == [(1.0, 0.0, 0.0)]
    assert dp.sweep_axes((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == [(1.0, 0.0, 0.0)]
    a = math.radians(30.0)
    axes = dp.sweep_axes((1.0, 0.0, 0.0), (math.cos(a), math.sin(a), 0.0))
    assert len(axes) == 20 and axes[-1] == pytest.approx((math.cos(a), math.sin(a), 0.0))
    angles = [math.degrees(math.atan2(y, x)) for x, y, _ in axes]
    assert angles == sorted(angles)
    gaps = [b - a for a, b in zip([0.0] + angles, angles)]
    assert max(gaps) < dp.PLUME_CORE_HALF_ANGLE_DEG       # no ship fits between two samples


def test_a_fast_sweep_cannot_tunnel_past_a_ship():
    """The emitter's nose jumps 30 deg in one tick; a ship 15 deg off the old
    exhaust axis is inside neither the old nor the new cone, but the jet
    crossed it: it takes the dwell-weighted heat and the event shows the
    peak of the crossing and the facing that glows."""
    sim = CombatSimulation(seed=5, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0), "corvette")
    a = math.radians(15.0)
    tail = _ship(sim, "tail", "beta", (-10 * math.cos(a), -10 * math.sin(a), 0), "destroyer", (1, 0, 0))
    _ship(sim, "far", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
    lead.applied_throttle = 1.0
    dp.resolve_plume_wash(sim, 1.0)                        # burning, nose +x: nobody washed
    assert not _washes(sim)
    b = math.radians(30.0)
    lead.kinematic_state.forward = Vector3D(math.cos(b), math.sin(b), 0.0)
    assert dp.compute_washes(sim) == []                    # the snapshot alone misses it
    sim.current_time += 1.0
    dp.resolve_plume_wash(sim, 1.0)
    (event,) = _washes(sim)
    assert event.ship_id == "tail" and event.data["phase"] == "enter" and event.data["swept"] is True
    assert event.data["hit_location"] == "nose" and event.data["friendly"] is False
    assert event.data["peak_heat_rate_gj_s"] == pytest.approx(16.0, rel=0.05)
    # Dwell: the 7.5 deg cone covers a quarter of the 30 deg sweep at most.
    assert 1.0 < event.data["heat_gj"] < 0.5 * event.data["peak_heat_rate_gj_s"]
    assert tail.thermal_system.heatsink.current_heat_gj == pytest.approx(event.data["heat_gj"], abs=0.06)
    assert event.data["dose_sv"] > 0.0 and tail.crew_dose_sv == pytest.approx(event.data["dose_sv"], abs=1e-3)
    sim.current_time += 1.0
    dp.resolve_plume_wash(sim, 1.0)                        # the jet has moved on
    assert [e.data["phase"] for e in _washes(sim)] == ["enter", "exit"]


def test_a_steady_burn_is_not_resampled():
    sim = _line_astern()
    for _ in range(3):
        sim.step()
    assert all(e.data["swept"] is False for e in _washes(sim))


def test_dose_is_shielded_by_the_armour_on_the_washed_facing():
    assert dp.dose_shielding(0.0) == 1.0
    assert dp.dose_shielding(dp.PLUME_DOSE_SHIELD_EFOLD_CM) == pytest.approx(1.0 / math.e)
    assert dp.plume_dose_rate_sv_s(16.0, 0.0) == pytest.approx(16.0 * dp.PLUME_DOSE_SV_PER_GJ)

    def dose_after(seconds, tail_forward, hull="destroyer"):
        sim = CombatSimulation(seed=5, time_step=1.0)
        lead = _ship(sim, "lead", "alpha", (0, 0, 0))
        tail = _ship(sim, "tail", "alpha", (-10, 0, 0), hull, tail_forward)
        _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
        lead.applied_throttle = 1.0
        for _ in range(seconds):
            dp.resolve_plume_wash(sim, 1.0)
            sim.current_time += 1.0
        return tail, sim
    bow_on, _ = dose_after(5, (1, 0, 0))
    beam_on, _ = dose_after(5, (0, 1, 0))
    battleship, _ = dose_after(5, (1, 0, 0), "battleship")
    assert beam_on.crew_dose_sv > 3.0 * bow_on.crew_dose_sv > 0.0     # 26 cm flank vs 151 cm bow
    assert battleship.crew_dose_sv < 0.5 * bow_on.crew_dose_sv        # a 262 cm belt protects its crew
    nose = bow_on.armor.get_section(bd.HitLocation.NOSE).thickness_cm
    assert bow_on.crew_dose_sv == pytest.approx(
        5 * dp.plume_dose_rate_sv_s(16.0, nose), rel=0.02)


def test_crew_dose_thresholds_impair_then_incapacitate():
    sim = CombatSimulation(seed=5, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0))
    tail = _ship(sim, "tail", "alpha", (-8, 0, 0), "destroyer", (0, 1, 0))      # beam-on
    foe = _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
    _burn(lead)
    _burn(tail)
    gun = tail.weapons["weapon_1"]
    base = gun._cooldown_after_shot(gun.weapon.salvo_size - 1)
    crossings = []
    for _ in range(60):
        sim.step()
        lead.kinematic_state.position = Vector3D(0, 0, 0)
        lead.kinematic_state.velocity = Vector3D(0, 0, 0)
        tail.kinematic_state.position = Vector3D(-8000, 0, 0)
        tail.kinematic_state.velocity = Vector3D(0, 0, 0)
        if tail.crew_incapacitated:
            break
    events = [e for e in sim.events if e.event_type == SimulationEventType.CREW_RADIATION]
    assert [e.data["state"] for e in events] == ["impaired", "incapacitated"]
    assert all(e.ship_id == "tail" and e.data["source_ship_id"] == "lead" for e in events)
    assert events[0].data["threshold_sv"] == dp.PLUME_CREW_IMPAIRED_SV <= events[0].data["dose_sv"]
    assert events[1].data["threshold_sv"] == dp.PLUME_CREW_INCAPACITATED_SV <= events[1].data["dose_sv"]
    assert events[0].timestamp < events[1].timestamp
    # Impaired: guns and tubes cycle slower (the damaged-mount mechanism).
    assert gun._cooldown_after_shot(gun.weapon.salvo_size - 1) == pytest.approx(
        base * dp.PLUME_CREW_IMPAIRED_CYCLE_FACTOR * bd.mount_cooldown_factor(gun.mount_health))
    # Incapacitated: no maneuver, no fire, unable to deal damage; still a ship.
    assert tail.crew_incapacitated and tail.current_maneuver is None and not tail.is_destroyed
    assert bd.ship_can_deal_damage(tail) is False and bd.ship_disarmed_by_damage(tail) is True
    assert bd.ship_can_deal_damage(lead) is True
    _burn(tail)
    speed = tail.velocity.magnitude
    shots = sum(ws.shots_fired for ws in tail.weapons.values())
    tail.kinematic_state.position = Vector3D(-8000, 50_000, 0)    # out of the jet
    for _ in range(5):
        sim.step()
    assert tail.current_maneuver is None and tail.velocity.magnitude == pytest.approx(speed)
    assert sum(ws.shots_fired for ws in tail.weapons.values()) == shots
    assert sim._handle_torpedo_command(tail, {"target_id": "foe"}) is False
    # The dose is permanent, in the wash events and in the status line.
    assert tail.crew_dose_sv >= dp.PLUME_CREW_INCAPACITATED_SV
    last = [e for e in _washes(sim) if e.ship_id == "tail"][-1].data
    assert last["crew_state"] == "incapacitated" and last["crew_dose_sv"] >= 8.0
    assert dp.plume_status_lines(sim, tail)[-1].startswith("CREW INCAPACITATED by plume radiation (")
    stats = dp.plume_stats(sim)
    assert (stats["crews_impaired"], stats["crews_incapacitated"]) == (1, 1)


def test_impaired_status_line_and_incapacitated_role():
    from src.llm import heuristic_tactics as T
    sim = _line_astern()
    tail = sim.ships["tail"]
    assert dp.plume_status_lines(sim, tail) == [] and dp.crew_state(tail) == "fit"
    dp._irradiate(sim, tail, 2.5, "lead")
    assert dp.crew_state(tail) == "impaired" and T.role(tail) == "gun"
    (line,) = dp.plume_status_lines(sim, tail)
    assert line == "CREW IMPAIRED by plume radiation (2.5 Sv, incapacitated at 8): weapons cycle x1.5"
    dp._irradiate(sim, tail, 6.0, "lead")
    assert dp.crew_state(tail) == "incapacitated" and T.role(tail) == "unarmed"


def test_impaired_torpedo_boat_cycles_slower_and_an_incapacitated_one_cannot_launch():
    sim = CombatSimulation(seed=5, time_step=1.0)
    boat = _ship(sim, "boat", "alpha", (0, 0, 0), "cruiser_torpedo")
    _ship(sim, "foe", "beta", (400, 0, 0), "destroyer", (-1, 0, 0))
    tubes = boat.ready_torpedo_launchers
    assert len(tubes) == 4 and all(t.current_magazine > 0 for t in tubes)
    base = [t.cooldown_seconds for t in tubes]
    assert sim._handle_torpedo_command(boat, {"target_id": "foe"}) is True     # one tube now mid-cooldown
    dp._irradiate(sim, boat, dp.PLUME_CREW_IMPAIRED_SV, "foe")
    assert boat.crew_impaired and not boat.crew_incapacitated
    assert [t.cooldown_seconds for t in tubes] == pytest.approx(
        [b * dp.PLUME_CREW_IMPAIRED_CYCLE_FACTOR for b in base])
    # Later tube damage multiplies on top of the crew factor, it does not undo it.
    fitting = next(f for f in bd._launcher_fittings(boat))
    bd.damage_fitting(sim, boat, fitting, 50.0, "surface_hit")
    assert tubes[0].cooldown_seconds == pytest.approx(
        base[0] * dp.PLUME_CREW_IMPAIRED_CYCLE_FACTOR * bd.mount_cooldown_factor(50.0))
    assert sim._handle_torpedo_command(boat, {"target_id": "foe"}) is True     # impaired crews still fight
    launched = len(sim.torpedoes)
    dp._irradiate(sim, boat, dp.PLUME_CREW_INCAPACITATED_SV, "foe")
    assert boat.crew_incapacitated and bd.ship_can_deal_damage(boat) is False
    # Immediate path and the pending (reserved salvo) path: nothing leaves.
    assert sim._handle_torpedo_command(boat, {"target_id": "foe"}) is False
    sim.pending_torpedo_launches.append(("boat", {"target_id": "foe"}, sim.current_time + 30.0))
    for _ in range(5):
        sim.step()
    assert len([t for t in sim.torpedoes if t.source_ship_id == "boat"]) <= launched
    assert sum(1 for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED
               and e.ship_id == "boat") == 2
    # Its rounds already in flight still count for the victory rule.
    assert bd.side_can_deal_damage(sim, [boat]) is bool(
        [t for t in sim.torpedoes if t.source_ship_id == "boat"])


# ---------------------------------------------------------------------------
# Seeded battle: the demo scenario (scripts/generate_test_battle.py plume_wash)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def column_battles(tmp_path_factory):
    """The plume demo battle - a destroyer 11 km astern of a burning
    battleship - flown by legacy captains (no plume avoidance) and by the
    tactical ones."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_test_battle", REPO / "scripts" / "generate_test_battle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = {}
    # interlock=False: the scenario's point is the wash itself (the default of
    # the demo generator); tests/test_drive_interlock.py flies it interlocked.
    for doctrine in ("legacy", "tactical"):
        _, path = module.plume_wash_battle(
            recording_dir=str(tmp_path_factory.mktemp(doctrine)), doctrine=doctrine,
            interlock=False)
        out[doctrine] = json.loads(Path(path).read_text())
    return out


def _plume_events(recording):
    return [e for e in recording["events"] if e["event_type"] == "plume_wash"]


def test_seeded_battle_legacy_column_cooks_its_follower(column_battles):
    rec = column_battles["legacy"]
    first = [e for e in _plume_events(rec)
             if e["ship_id"] == "alpha_2" and e["data"]["source_ship_id"] == "alpha_1"]
    phases = [e["data"]["phase"] for e in first]
    assert phases[0] == "enter" and phases[-1] == "exit" and phases.count("sustain") >= 6
    assert first[0]["timestamp"] == 30.0 and first[0]["data"]["hit_location"] == "nose"
    assert first[0]["data"]["distance_km"] == pytest.approx(11.0, abs=0.1)
    total = first[-1]["data"]
    assert first[-1]["timestamp"] == 69.0 and total["duration_s"] == 39.0
    assert total["heat_gj"] > 400.0 and total["armor_ablation_cm"] > 8.0
    assert 3.0 < total["dose_sv"] < 4.0 and total["crew_state"] == "impaired"
    # Its captain put the radiators out at the 60 s checkpoint, into the jet:
    # all four panels are gone, listed in the episode and recorded.
    assert len(total["radiators"]) == 4 and all(p["destroyed"] for p in total["radiators"])
    shredded = [e for e in rec["events"] if e["event_type"] == "radiator_damaged"]
    assert [(e["timestamp"], e["ship_id"], e["data"]["cause"]) for e in shredded] == \
        [(60.0, "alpha_2", "plume_wash")] * 4
    burned = [e for e in rec["events"] if e["event_type"] == "mount_damaged"
              and e["data"].get("cause") == "plume_wash"]
    assert len(burned) == 16 and {e["ship_id"] for e in burned} == {"alpha_2"}
    assert sum(1 for e in burned if e["data"]["destroyed"]) == 4       # spinal, turret, both PD
    crew = [e for e in rec["events"] if e["event_type"] == "crew_radiation"]
    assert [(e["timestamp"], e["ship_id"], e["data"]["state"]) for e in crew] == [(50.0, "alpha_2", "impaired")]
    # The cruiser 14 km abeam is never washed.
    assert not [e for e in _plume_events(rec) if e["ship_id"] == "alpha_3"]
    # The wash itself stops at the ceiling; the destroyer, at 149 % with no
    # radiator left and its own drive still burning, melts down a second
    # after it leaves the jet.
    dead = [(e["timestamp"], e["ship_id"], e["data"].get("cause")) for e in rec["events"]
            if e["event_type"] == "ship_destroyed"]
    assert dead == [(70.0, "alpha_2", "reactor_overheat")]


def test_seeded_battle_tactical_captains_keep_the_follower_out_of_the_jet(column_battles):
    legacy = sum(e["data"]["heat_gj"] for e in _plume_events(column_battles["legacy"])
                 if e["data"]["phase"] == "exit" and e["data"]["friendly"])
    rec = column_battles["tactical"]
    tactical = [e["data"] for e in _plume_events(rec) if e["data"]["friendly"]]
    assert legacy > 400.0
    assert max([d["heat_gj"] for d in tactical] or [0.0]) < 0.05 * legacy
    assert max([d.get("crew_dose_sv", 0.0) for d in tactical] or [0.0]) < 0.2
    assert not [e for e in rec["events"] if e["event_type"] == "crew_radiation"]
    assert not [e for e in rec["events"] if e["event_type"] == "mount_damaged"
                and e["data"].get("cause") == "plume_wash"]


# ---------------------------------------------------------------------------
# Radiators in the jet
# ---------------------------------------------------------------------------

def _washed(gap_km, extended, seconds=3, tail_forward=(1, 0, 0)):
    from src.thermal import RadiatorPosition  # noqa: F401 - documents the key type
    sim = CombatSimulation(seed=5, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0))
    tail = _ship(sim, "tail", "alpha", (-gap_km, 0, 0), "destroyer", tail_forward)
    _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
    lead.applied_throttle = 1.0
    if extended:
        sim._set_radiators(tail, True, reason="test")
    for _ in range(seconds):
        dp.resolve_plume_wash(sim, 1.0)
        sim.current_time += 1.0
    return sim, tail


def _panel_health(ship):
    return {p.value: panel.health_percent for p, panel in ship.thermal_system.radiators.radiators.items()}


def test_extended_radiators_in_a_strong_wash_are_destroyed_and_recorded():
    sim, tail = _washed(10.0, extended=True, seconds=2)
    assert all(h == 0.0 for h in _panel_health(tail).values())
    burned = [e for e in sim.events if e.event_type == SimulationEventType.RADIATOR_DAMAGED]
    assert len(burned) == len(_panel_health(tail)) and all(e.ship_id == "tail" for e in burned)
    assert all(e.data["cause"] == "plume_wash" and e.data["source_ship_id"] == "lead"
               and e.data["destroyed"] and e.data["health_pct"] == 0.0 for e in burned)
    assert burned[0].timestamp == 0.0                         # gone on the first tick at 10 km
    listed = _washes(sim)[-1].data["radiators"] if len(_washes(sim)) > 1 else _washes(sim)[0].data["radiators"]
    assert _washes(sim)[0].data["radiators"] and all(p["destroyed"] for p in _washes(sim)[0].data["radiators"])
    assert {p["radiator_position"] for p in listed} == set(_panel_health(tail))
    assert sum(p["capacity_lost_kw"] for p in listed) == pytest.approx(780_000.0, rel=0.02)
    assert dp.plume_stats(sim)["radiators_destroyed"] == len(_panel_health(tail))


def test_retracted_radiators_are_untouched_and_a_weak_wash_does_nothing():
    sim, tail = _washed(10.0, extended=False, seconds=3)
    assert all(h == 100.0 for h in _panel_health(tail).values())
    assert not [e for e in sim.events if e.event_type == SimulationEventType.RADIATOR_DAMAGED]
    assert _washes(sim)[0].data["radiators"] == []
    # 60 km astern: 0.55 GJ/s, less than the array rejects - extended panels live.
    assert dp.plume_heat_rate_gj_s(FULL, 60.0, 0.0) < dp.PLUME_RADIATOR_WASH_MIN_GJ_PER_S
    sim, tail = _washed(60.0, extended=True, seconds=30)
    assert _washes(sim) and all(h == 100.0 for h in _panel_health(tail).values())
    assert not [e for e in sim.events if e.event_type == SimulationEventType.RADIATOR_DAMAGED]
    # 20 km: a panel lasts two ticks, not one.
    sim, tail = _washed(20.0, extended=True, seconds=1)
    assert all(0.0 < h < 100.0 for h in _panel_health(tail).values())
    dp.resolve_plume_wash(sim, 1.0)
    assert all(h == 0.0 for h in _panel_health(tail).values())


def test_heat_ladder_does_not_force_radiators_out_into_a_jet():
    sim = CombatSimulation(seed=5, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0))
    tail = _ship(sim, "tail", "alpha", (-10, 0, 0), "destroyer")
    _ship(sim, "foe", "beta", (3000, 0, 0), "destroyer", (-1, 0, 0))
    _burn(lead)
    sim.step()                                                   # the wash is now known
    sink = tail.thermal_system.heatsink
    sink.current_heat_gj = 1.2 * sink.capacity_gj               # past the 100 % forced-extension line
    for _ in range(6):
        sim.step()
        lead.kinematic_state.position = Vector3D(0, 0, 0)
        lead.kinematic_state.velocity = Vector3D(0, 0, 0)
        assert tail.thermal_system.radiators_forced and dp.wash_rate_on(sim, "tail") > 10.0
        assert tail.radiators_extended is False                  # held in: ~16 GJ/s of jet outside
    assert all(h == 100.0 for h in _panel_health(tail).values())
    # The lead stops burning: one tick later the ship is clear and the ladder acts.
    lead.current_maneuver = None
    sim.step()
    sim.step()
    assert dp.wash_rate_on(sim, "tail") == 0.0 and dp.radiators_in_jet(sim, tail) is False
    sim.step()
    assert tail.thermal_system.radiators_forced
    assert tail.radiators_extended is True                       # clear of the jet: the ladder acts


def test_forced_extension_is_suppressed_from_the_tick_after_the_wash_starts():
    sim, tail = _washed(10.0, extended=False, seconds=1)
    sink = tail.thermal_system.heatsink
    sink.current_heat_gj = 1.2 * sink.capacity_gj
    assert dp.radiators_in_jet(sim, tail) is True
    sim._update_radiator_automation(tail)
    assert tail.radiators_extended is False                    # held in: it is in the jet
    sim.current_time += 5.0                                     # the wash state is stale: clear
    assert dp.radiators_in_jet(sim, tail) is False
    sim._update_radiator_automation(tail)
    assert tail.radiators_extended is True
