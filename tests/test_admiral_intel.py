"""
Admiral information picture (pass-4 audit, workstream B).

Pins what the admiral is told and that it is TRUE: side-attributed ordnance
(ADM-01), the directive-failure contract (ADM-02), captain reports (ADM-03
rendering), fleet-centre geometry (ADM-04), torpedo solutions (ADM-05),
the since-last-checkpoint digest (ADM-06), the compact ranked turn state
(ADM-07, golden text), per-ship order frames (ADM-08/R6), composed doctrine
(ADM-10), the cacheable message split (ADM-11) and the enemy armament line
(R2). Battle A geometry comes from tests/fixtures/battle_a_frames.json
(sim_trace frames of data/recordings/sonnet5_vs_dsv41flash_150pt_20260926).
"""

import json
import math
import os
import types
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.projectile import KineticProjectile
from src.simulation import (
    CombatSimulation,
    ProjectileInFlight,
    SimulationEvent,
    SimulationEventType,
    TorpedoInFlight,
    _build_torpedo_specs,
    create_ship_from_fleet_data,
)
from src.torpedo import Torpedo
from src.llm import fleet_intel as intel
from src.llm.admiral import LLMAdmiral
from src.llm.client import LLMCallError, ToolCall
from src.llm.prompts import (
    ADMIRAL_STATE_MARKER,
    build_admiral_messages,
    build_admiral_prompt,
    build_admiral_ship_order_messages,
    format_admiral_turn_state,
)

ROOT = Path(__file__).resolve().parent.parent
FLEET = json.loads((ROOT / "data" / "fleet_ships.json").read_text())
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "battle_a_frames.json").read_text())
GOLDEN = ROOT / "tests" / "fixtures" / "admiral_turn_state_t180_alpha.txt"


# =============================================================================
# Helpers
# =============================================================================

def battle_a_sim(t: int) -> CombatSimulation:
    """Battle A at sim time t, rebuilt from its recorded frame."""
    frame = FIXTURE["frames"][str(t)]
    sim = CombatSimulation(decision_interval=30.0, seed=1)
    sim.current_time = float(t)
    for sid, s in frame["ships"].items():
        if s["destroyed"]:
            continue
        meta = FIXTURE["ships"][sid]
        ship = create_ship_from_fleet_data(
            sid, meta["ship_type"], sid.split("_")[0], FLEET,
            position=Vector3D(*s["pos"]), velocity=Vector3D(*s["vel"]),
            forward=Vector3D(*s["fwd"]))
        ship.name = meta["ship_name"]
        ship.applied_throttle = float(s["thrust"] or 0.0)
        sim.add_ship(ship)
    specs = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])
    for x in frame["torpedoes"]:
        torp = Torpedo(specs=specs, position=Vector3D(*x["pos"]),
                       velocity=Vector3D(*x["vel"]), target_id=x["target"])
        torp.remaining_delta_v_kps = x["dv_remaining"]
        torp.current_mass_kg = specs.dry_mass_kg * math.exp(
            x["dv_remaining"] / specs.exhaust_velocity_kps)
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id=x["id"], torpedo=torp, source_ship_id=x["source"],
            is_disabled=x["disabled"]))
    for x in frame["projectiles"]:
        proj = KineticProjectile(mass_kg=x["mass_kg"], velocity=Vector3D(*x["vel"]),
                                 position=Vector3D(*x["pos"]))
        sim.projectiles.append(ProjectileInFlight(
            projectile_id=x["id"], projectile=proj, source_ship_id=x["source"],
            target_ship_id=x["target"]))
    return sim


def captains_for(faction, ships=None):
    ships = ships or FIXTURE["ships"]
    return [types.SimpleNamespace(
        ship_id=sid, ship_name=meta["ship_name"], name=f"Captain {sid}",
        config=types.SimpleNamespace(ship_type=meta["ship_type"], name=f"Captain {sid}"))
        for sid, meta in ships.items() if sid.startswith(faction)]


def make_admiral(faction, client=None, model="x/y"):
    cfg = types.SimpleNamespace(name=f"Admiral {faction}", model=model, personality=None,
                                vision=False, notebook_text=None, image_input=False)
    admiral = LLMAdmiral(cfg, faction, client=client, fleet_data=FLEET)
    admiral.parallel_ship_orders = False
    return admiral


def battle_a_turn(faction, t=180, reports=None):
    admiral = make_admiral(faction)
    caps = captains_for(faction)
    if str(t - 15) in FIXTURE["frames"]:
        admiral._snapshot_t_minus_15 = admiral._build_snapshot(battle_a_sim(t - 15), caps)
    sim = battle_a_sim(t)
    snap = admiral._build_snapshot(sim, caps, full=True)
    msgs = build_admiral_messages(
        admiral_name=admiral.config.name, faction=faction,
        snapshot_t_minus_15=admiral._snapshot_t_minus_15, snapshot_t_zero=snap,
        personality=None, fleet_data=FLEET, enemy_has_admiral=True, phase="directive",
        standing_plan="Phase 1: close. Phase 2: salvo.", standing_plan_time=30.0,
        captain_reports=reports, battle_classes=admiral._battle_classes)
    return admiral, sim, snap, msgs


def ship(sim, sid, stype, faction, pos_km, vel_kps=(0, 0, 0), fwd=(1, 0, 0)):
    s = create_ship_from_fleet_data(
        sid, stype, faction, FLEET, position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(*(c * 1000.0 for c in vel_kps)), forward=Vector3D(*fwd))
    s.name = f"{'TIS' if faction == 'alpha' else 'OCS'} {sid}"
    sim.add_ship(s)
    return s


def add_torpedo(sim, tid, source, target, pos_km, vel_kps, disabled=False, dv_kps=14.0):
    specs = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])
    torp = Torpedo(specs=specs, position=Vector3D(*(c * 1000.0 for c in pos_km)),
                   velocity=Vector3D(*(c * 1000.0 for c in vel_kps)), target_id=target)
    torp.remaining_delta_v_kps = dv_kps
    torp.current_mass_kg = specs.dry_mass_kg * math.exp(dv_kps / specs.exhaust_velocity_kps)
    sim.torpedoes.append(TorpedoInFlight(torpedo_id=tid, torpedo=torp, source_ship_id=source,
                                         is_disabled=disabled))


def live_turn(admiral, sim, captains, **kwargs):
    snap = admiral._build_snapshot(sim, captains, full=True)
    return snap, format_admiral_turn_state(snap, FLEET, phase="directive", **kwargs)


def caps_from_sim(sim, faction):
    return [types.SimpleNamespace(
        ship_id=s.ship_id, ship_name=s.name, name=f"Captain {s.ship_id}",
        config=types.SimpleNamespace(ship_type=s.ship_type, name=f"Captain {s.ship_id}"))
        for s in sim.ships.values() if s.faction == faction]


def section(text, heading):
    """The block of a turn state that starts with ``heading``."""
    start = text.index(heading)
    end = text.find("\n\n", start)
    return text[start:end if end != -1 else len(text)]


# =============================================================================
# ADM-01: ordnance is side-attributed; dead rounds threaten only on their line
# =============================================================================

class TestOrdnance:

    def _duel(self):
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        return sim

    def test_own_outgoing_torpedoes_are_not_inbound(self):
        sim = self._duel()
        add_torpedo(sim, "torp_alpha_1_0001", "alpha_1", "beta_1", (50, 0, 0), (6, 0, 0))
        _, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        ordnance = section(text, "=== ORDNANCE IN FLIGHT ===")
        assert "none - nothing in flight is on course for your hulls" in ordnance
        assert "-> OCS beta_1 [beta_1]: 1 guided" in ordnance
        assert "Torpedoes inbound on" not in section(text, "!! NOW")
        # The same round IS inbound for the other side.
        _, beta_text = live_turn(make_admiral("beta"), sim, caps_from_sim(sim, "beta"))
        assert "1x torpedo -> OCS beta_1 [beta_1] from TIS alpha_1 [alpha_1]" in beta_text
        assert "Torpedoes inbound on: OCS beta_1 [beta_1] (1 torpedo ETA" in beta_text

    def test_receding_blinded_wreck_is_spent_not_inbound(self):
        sim = self._duel()
        # Blinded, moving away from everyone.
        add_torpedo(sim, "torp_beta_1_0001", "beta_1", "alpha_1", (-40, 0, 0), (-9, 0, 0),
                    disabled=True)
        snap, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        wreck = next(r for r in snap.rounds if r.round_id == "torp_beta_1_0001")
        assert wreck.state == "blinded" and wreck.threat_id is None
        assert "spent: 0 of yours and 1 of theirs are dead rounds on nobody's line" in text
        assert "dead torpedo" not in section(text, "INBOUND ON YOU")

    def test_wreck_on_a_collision_course_is_a_threat(self):
        sim = self._duel()
        # Our OWN blinded round, coasting straight at our corvette: 20 s out.
        add_torpedo(sim, "torp_alpha_1_0009", "alpha_1", "beta_1", (200, 0.1, 0), (-10, 0, 0),
                    disabled=True)
        snap, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        wreck = next(r for r in snap.rounds if r.round_id == "torp_alpha_1_0009")
        assert wreck.threat_id == "alpha_1"
        assert wreck.tca_s == pytest.approx(20.0, abs=0.5)
        assert "1x dead torpedo (blinded) -> TIS alpha_1 [alpha_1]: passes 0.10 km in 20 s" in text
        assert "EVADE makes the engine dodge" in text

    def test_battle_a_t270_wrecks_are_no_threat_to_beta(self):
        """The audit case: beta's admiral was told 'Ten blinded ballistic torpedo
        wrecks are inbound on every hull' - all ten were receding."""
        _, _, snap, msgs = battle_a_turn("beta", t=270)
        dead = [r for r in snap.rounds if r.kind == "torpedo" and r.ballistic]
        assert len(dead) == 10
        assert all(r.threat_id is None for r in dead)
        live = msgs[1]["content"]
        assert "dead torpedo" not in section(live, "INBOUND ON YOU")
        assert "10 of theirs are dead rounds on nobody's line" in live
        assert "Torpedoes inbound on" not in section(live, "!! NOW")

    def test_no_eta_sentinel_anywhere(self):
        for faction in ("alpha", "beta"):
            for t in (180, 270):
                live = battle_a_turn(faction, t=t)[3][1]["content"]
                assert "999" not in live

    def test_slug_eta_uses_relative_closure(self):
        sim = self._duel()
        sim.ships["alpha_1"].kinematic_state.velocity = Vector3D(3000.0, 0, 0)
        proj = KineticProjectile(mass_kg=50, velocity=Vector3D(-5000.0, 0, 0),
                                 position=Vector3D(80_000.0, 0, 0))
        sim.projectiles.append(ProjectileInFlight(projectile_id="p1", projectile=proj,
                                                  source_ship_id="beta_1",
                                                  target_ship_id="alpha_1"))
        snap, _ = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        slug = next(r for r in snap.rounds if r.kind == "slug")
        assert slug.eta_s == pytest.approx(80.0 / 8.0, rel=1e-3)  # closing 8 km/s, not 5


# =============================================================================
# ADM-05 / R5: torpedo solutions and salvo math
# =============================================================================

def _solution(target_type, rng_km, closing_kps, salvo=8):
    sim = CombatSimulation(seed=1)
    me = ship(sim, "me", "corvette", "alpha", (0, 0, 0))
    tg = ship(sim, "tg", target_type, "beta", (rng_km, 0, 0), vel_kps=(-closing_kps, 0, 0),
              fwd=(-1, 0, 0))
    specs = intel.torpedo_specs_for(sim, FLEET, [me])
    return intel.torpedo_solution(me.position, me.velocity, tg, specs, FLEET, sim,
                                  salvo_available=salvo)


class TestTorpedoSolution:

    def test_closing_launch_is_good(self):
        sol = _solution("destroyer", 400.0, 4.0)
        assert sol.verdict == "GOOD"
        assert sol.energy_gj >= intel.GOOD_ENERGY_GJ
        assert 30 < sol.tof_s < 90

    def test_strongly_opening_target_is_no_solution(self):
        sol = _solution("corvette", 300.0, -6.0)
        assert sol.verdict == "NO"

    def test_battle_a_t270_launch_at_falchion_is_not_good(self):
        """T+270: 8 rounds at Falchion-2 opening 3-5 km/s -> 0 hits."""
        sim = battle_a_sim(270)
        alphas = [s for s in sim.ships.values() if s.faction == "alpha"]
        pos, vel = intel.centre(alphas)
        target = sim.get_ship("beta_6")
        specs = intel.torpedo_specs_for(sim, FLEET, alphas)
        sol = intel.torpedo_solution(pos, vel, target, specs, FLEET, sim, salvo_available=24)
        assert sol.closing_kps < 0
        assert sol.verdict in ("POOR", "NO")
        _, _, snap, msgs = battle_a_turn("alpha", t=270)
        row = next(e for e in snap.enemy_ships if e.ship_id == "beta_6")
        assert row.solution.verdict != "GOOD"
        assert "OCS Falchion-2 [beta_6] DD" in msgs[1]["content"]

    @pytest.mark.parametrize("closing_kps,lo,hi", [
        (0.0, 12.5, 14.5),    # 'launched with zero closure -> impacts ~13.5 km/s'
        (12.0, 14.0, 17.0),   # 'launched closing 12 km/s -> impacts ~14-17 km/s'
        (26.0, 27.0, 30.0),   # 'launched closing 26 km/s -> impacts ~28 km/s'
    ])
    def test_pursuit_model_matches_the_measured_impact_anchors(self, closing_kps, lo, hi):
        """The captain doctrine's impact speeds are measured on the engine
        (400 km launch, target not running); the pursuit model must agree."""
        specs = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])
        run = intel.solve_pursuit(400_000.0, closing_kps * 1000.0, 0.0, 0.0, specs)
        assert run["hit"]
        assert lo <= run["impact_mps"] / 1000.0 <= hi

    @staticmethod
    def _engine_impact(target_type, rng_km, closing_kps):
        """One real Trident on the shipped engine at a target that RUNS from
        launch (EVADE, nose already pointing away), PD removed so only
        guidance and kinematics count: (impact km/s, GJ), or None on a miss."""
        from src.simulation import Maneuver, ManeuverType
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=3)
        shooter = ship(sim, "sh", "corvette", "beta", (rng_km, 0, 0), fwd=(-1, 0, 0))
        target = ship(sim, "tg", target_type, "alpha", (0, 0, 0), vel_kps=(closing_kps, 0, 0),
                      fwd=(-1, 0, 0))
        target.point_defense = []
        specs = intel.torpedo_specs_for(sim, FLEET, [shooter])
        sol = intel.torpedo_solution(shooter.position, shooter.velocity, target, specs, FLEET,
                                     None, salvo_available=1)
        target.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE, start_time=0.0,
                                           duration=1e9, throttle=1.0)
        assert sim.inject_command("sh", {"type": "launch_torpedo", "target_id": "tg"})
        while sim.current_time < 600.0 and sim.torpedoes:
            for ev in sim.step():
                if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                    return sol, (ev.data["impact_speed_kps"], ev.data["total_damage_gj"])
        return sol, None

    @pytest.mark.parametrize("target_type,rng_km,closing_kps", [
        ("battleship", 30.0, 0.0),
        ("battleship", 100.0, 0.0),
        ("battleship", 100.0, 6.0),
        ("battleship", 250.0, 2.0),
        ("battleship", 400.0, 0.0),
        ("corvette", 100.0, 0.0),
        ("corvette", 250.0, 2.0),
    ])
    def test_solution_matches_the_engine_at_every_range(self, target_type, rng_km, closing_kps):
        """
        The pursuit model flies the engine's own throttle law (full thrust
        below the closure floor, then only the NEZ surplus), so a short-range
        launch is NOT credited with thrust it had no time to fire: at 100 km
        and zero closure the engine lands ~6 km/s / ~5 GJ, not the ~13 km/s
        a long flight reaches. Within a few percent and never optimistic.
        """
        sol, hit = self._engine_impact(target_type, rng_km, closing_kps)
        assert hit is not None, "reference launch must connect"
        engine_kps, engine_gj = hit
        assert sol.impact_kps is not None
        assert 0.88 * engine_kps <= sol.impact_kps <= 1.03 * engine_kps
        if engine_gj < intel.GOOD_ENERGY_GJ:
            assert sol.verdict != "GOOD", (sol.verdict, sol.energy_gj, engine_gj)
        if engine_gj < intel.NO_ENERGY_GJ:
            assert sol.verdict == "NO"

    def test_short_range_zero_closure_launch_is_not_good(self):
        """Reviewer repro: 100 km, nobody closing - the engine lands < 5 GJ."""
        sol = _solution("battleship", 100.0, 0.0)
        assert sol.verdict == "NO"
        assert sol.energy_gj < intel.NO_ENERGY_GJ
        assert "lands only" in sol.reason

    def test_salvo_sizing_scales_with_the_energy_each_round_lands(self):
        assert intel.kill_rounds_at(6, 22.0) == 6
        assert intel.kill_rounds_at(6, 30.0) == 6          # never fewer than the doctrine
        assert intel.kill_rounds_at(6, 11.0) == 12
        assert intel.kill_rounds_at(4, None) == 4
        _, _, _, msgs = battle_a_turn("alpha", t=180)
        econ = section(msgs[1]["content"], "=== TORPEDO ECONOMY ===")
        line = next(l for l in econ.splitlines() if "OCS Falchion-2 [beta_6]" in l)
        assert "(POOR)" in line and " to kill at ~7 GJ each = " in line

    def test_salvo_sizing_reference_is_the_doctrine_number(self):
        from src.llm.prompts import ADMIRAL_TORPEDO_DOCTRINE
        assert intel.KILL_ROUNDS_REF_GJ == 22.0
        assert f"(~{intel.KILL_ROUNDS_REF_GJ:.0f} GJ/round, evading targets): " \
               f"{intel.KILL_ROUNDS_LIGHT} rounds kill a" in ADMIRAL_TORPEDO_DOCTRINE
        assert f"{intel.KILL_ROUNDS_HEAVY} rounds kill anything" in ADMIRAL_TORPEDO_DOCTRINE

    def test_verdict_thresholds_sit_between_the_doctrine_anchors(self):
        ke = lambda v_kps: 0.5 * 250 * (v_kps * 1000) ** 2 / 1e9  # noqa: E731
        assert intel.GOOD_ENERGY_GJ < ke(13.5)        # a zero-closure launch is GOOD
        assert intel.NO_ENERGY_GJ < ke(8) < intel.GOOD_ENERGY_GJ  # a receding lob is POOR

    def test_in_flight_rounds_report_whether_they_still_catch_a_runner(self):
        _, _, snap, msgs = battle_a_turn("alpha", t=180)
        guided = [r for r in snap.rounds if r.side == "ours" and r.state == "guided"]
        assert guided and all(r.can_reach for r in guided)
        assert "still catch a runner" in msgs[1]["content"]

    def test_salvo_math_counts_pd_cover_and_rounds_already_inbound(self):
        _, _, snap, msgs = battle_a_turn("alpha", t=180)
        econ = section(msgs[1]["content"], "=== TORPEDO ECONOMY ===")
        assert "Yours: 64/64 rounds left; 24 launches possible" in econ
        assert "up to 3 per tube per checkpoint" in econ
        lancer = next(e for e in snap.enemy_ships if e.ship_id == "beta_2")
        assert lancer.our_rounds_inbound == 2
        line = next(l for l in econ.splitlines() if "OCS Lancer-1 [beta_2]" in l)
        need = int(line.split(" together")[0].split("= ")[-1])
        assert f"2 of yours already inbound -> {need - 2} more" in line
        assert "launch together or not at all" in econ


# =============================================================================
# ADM-04: geometry from the fleet centre, who is closing, no symmetric "fleeing"
# =============================================================================

class TestGeometry:

    def test_battle_a_t180_merge_is_first_and_correct(self):
        _, _, snap, msgs = battle_a_turn("alpha", t=180)
        live = msgs[1]["content"]
        now = section(live, "!! NOW")
        assert "GROUP A: MERGE IN 45 s at 1.4 km (OCS Lancer-3), too late to brake" in now
        geo = section(live, "=== GEOMETRY")
        assert "closing 4.03 km/s (-60 km in 15 s)" in geo
        assert "TOO LATE to brake" in geo
        ours, theirs = snap.group_motion["GROUP A"]
        assert (ours + theirs) / 1000.0 == pytest.approx(snap.range_groups[0].closing_kps, abs=0.02)

    def test_after_the_pass_nobody_is_told_the_enemy_flees(self):
        for faction in ("alpha", "beta"):
            _, _, snap, msgs = battle_a_turn(faction, t=270)
            live = msgs[1]["content"]
            assert "fleeing" not in live.lower()
            group = snap.range_groups[0]
            assert group.closing_kps < 0
            assert f"OPENING {-group.closing_kps:.2f} km/s" in live
            ours, theirs = snap.group_motion[group.label]
            assert (ours + theirs) / 1000.0 == pytest.approx(group.closing_kps, abs=0.02)

    @staticmethod
    def _one_v_one(own_vel_kps, en_vel_kps, en_throttle=0.0):
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        sim.current_time = 60.0
        ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0), vel_kps=own_vel_kps)
        foe = ship(sim, "beta_1", "battleship", "beta", (100, 0, 0), vel_kps=en_vel_kps,
                   fwd=(-1, 0, 0))
        foe.applied_throttle = en_throttle
        snap, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        line = next(l for l in section(text, "=== GEOMETRY").splitlines()
                    if l.startswith("GROUP A"))
        return snap, line

    def test_who_is_closing_follows_who_moved_not_ship_counts(self):
        """1 v 1 at 100 km closing 2 km/s, three ways. In the old centre-of-
        mass frame all three printed 'you 1.0 + they 1.0'."""
        _, line = self._one_v_one((2, 0, 0), (0, 0, 0))          # we charge a sitting enemy
        assert "= you 2.0 km/s toward them + they 0.0 toward you; they coast" in line
        _, line = self._one_v_one((0, 0, 0), (-2, 0, 0))         # he charges us
        assert "= you 0.0 km/s toward them + they 2.0 toward you" in line
        _, line = self._one_v_one((-3, 0, 0), (-5, 0, 0))        # we run, he chases faster
        assert "= you 3.0 km/s away from them + they 5.0 toward you" in line

    def test_geometry_says_which_way_they_burn_now(self):
        snap, line = self._one_v_one((0, 0, 0), (-2, 0, 0), en_throttle=1.0)
        assert snap.group_burn_g["GROUP A"] == pytest.approx(1.0, abs=0.05)   # 1 g battleship
        assert line.endswith("; they burn toward you 1.0g")
        # Nose away from us at full throttle: burning away (braking his charge).
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
        foe = ship(sim, "beta_1", "battleship", "beta", (100, 0, 0), vel_kps=(-2, 0, 0),
                   fwd=(1, 0, 0))
        foe.applied_throttle = 1.0
        _, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        assert "+ they 2.0 toward you; they burn away 1.0g" in text

    def test_enemy_rows_use_the_fleet_centre_and_name_the_nearest_hull(self):
        _, sim, snap, msgs = battle_a_turn("alpha", t=180)
        sov = next(e for e in snap.enemy_ships if e.ship_id == "beta_1")
        alphas = [s for s in sim.ships.values() if s.faction == "alpha"]
        pos, _ = intel.centre(alphas)
        assert sov.range_km == pytest.approx((sim.get_ship("beta_1").position - pos).magnitude / 1000.0)
        assert sov.range_km > sov.distance_from_closest_friendly_km + 20
        row = next(l for l in msgs[1]["content"].splitlines() if l.startswith("OCS Sovereign [beta_1]"))
        assert f"{sov.range_km:.0f} km +" in row
        assert f"nearest TIS Dart-1 [alpha_1] {sov.distance_from_closest_friendly_km:.0f} km" in row


# =============================================================================
# R2 / threat table: launchers visible, damage dealt, hull
# =============================================================================

class TestThreatTable:

    def test_enemy_torpedo_launchers_are_visible(self):
        live = battle_a_turn("beta", t=180)[3][1]["content"]
        row = next(l for l in live.splitlines() if l.startswith("TIS Dart-1 [alpha_1]"))
        assert "1 torpedo tube, ~8/8 rounds left (0 seen)" in row
        assert "their torpedoes: 8 tubes" in live

    def test_torpedo_cruiser_and_siege_dreadnought_armament(self):
        cruiser = intel.class_armament("cruiser_torpedo", FLEET)
        assert cruiser["summary"] == "4 torpedo tubes" and cruiser["magazine"] == 48
        siege = intel.class_armament("dreadnought_siege", FLEET)
        assert "siege spinal" in siege["summary"] and siege["launchers"] == 0

    def test_damage_dealt_to_us_is_attributed(self):
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "cruiser_torpedo", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        for k in range(3):
            sim.events.append(SimulationEvent(SimulationEventType.TORPEDO_LAUNCHED, 10.0 + k,
                                              "beta_1", "alpha_1", {}))
        sim.events.append(SimulationEvent(SimulationEventType.TORPEDO_IMPACT, 40.0, "beta_1",
                                          "alpha_1", {"total_damage_gj": 22.5}))
        sim.current_time = 60.0
        snap, text = live_turn(make_admiral("alpha"), sim, caps_from_sim(sim, "alpha"))
        row = next(l for l in text.splitlines() if l.startswith("OCS beta_1 [beta_1]"))
        assert "4 torpedo tubes, ~45/48 rounds left (3 seen)" in row
        assert "dealt 1 hits 22 GJ" in row
        assert "Theirs: 3 launched so far; 4 tubes in their fleet (~45 rounds left)" in text


# =============================================================================
# ADM-06: since-last-checkpoint digest
# =============================================================================

class TestDigest:

    def test_kills_losses_and_damage_sources(self):
        sim = CombatSimulation(decision_interval=30.0, seed=1)
        ship(sim, "alpha_1", "cruiser_torpedo", "alpha", (0, 0, 0))
        ship(sim, "alpha_2", "cruiser_torpedo", "alpha", (0, 30, 0))
        ship(sim, "beta_1", "battlecruiser", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        dead = ship(sim, "beta_3", "destroyer", "beta", (300, 40, 0), fwd=(-1, 0, 0))
        dead.is_destroyed = True
        E = SimulationEventType
        events = [
            (95.0, E.TORPEDO_LAUNCHED, "alpha_1", "beta_1", {}),      # before the window
            (130.0, E.TORPEDO_LAUNCHED, "alpha_1", "beta_1", {}),
            (131.0, E.TORPEDO_LAUNCHED, "alpha_2", "beta_1", {}),
            (150.0, E.TORPEDO_IMPACT, "alpha_1", "beta_1", {"total_damage_gj": 30.0}),
            (151.0, E.TORPEDO_IMPACT, "alpha_2", "beta_3", {"total_damage_gj": 25.0}),
            (152.0, E.PD_TORPEDO_DISABLED, "beta_1", None, {"source_ship": "alpha_2"}),
            (155.0, E.SHIP_DESTROYED, "beta_3", None, {"killer_id": "alpha_2",
                                                     "cause": "reactor_destroyed"}),
            (160.0, E.PROJECTILE_LAUNCHED, "beta_1", "alpha_1", {}),
            (161.0, E.PROJECTILE_IMPACT, "beta_1", "alpha_1", {"kinetic_energy_gj": 4.3}),
        ]
        for ts, et, sid, tid, data in events:
            sim.events.append(SimulationEvent(et, ts, sid, tid, data))
        digest = intel.checkpoint_digest(sim, "alpha", since_t=120.0, until_t=180.0)
        assert [k["ship_id"] for k in digest.kills] == ["beta_3"]
        assert digest.kills[0]["killer_id"] == "alpha_2"
        assert digest.losses == []
        assert digest.torps["ours"]["launched"] == 2
        assert digest.torps["ours"]["impact"] == 2 and digest.torps["ours"]["blinded"] == 1
        assert digest.taken["alpha_1"].hits[("beta_1", "slug")] == 1
        assert digest.dealt["beta_1"].energy_gj == pytest.approx(30.0)

        admiral = make_admiral("alpha")
        admiral._last_decision_t = 120.0
        sim.current_time = 180.0
        _, text = live_turn(admiral, sim, caps_from_sim(sim, "alpha"))
        since = section(text, "=== SINCE LAST CHECKPOINT (T+120 -> T+180) ===")
        assert ("Kills: OCS beta_3 [beta_3] destroyed by TIS alpha_2 [alpha_2], reactor "
                "destroyed T+155 | Losses: none") in since
        assert "TIS alpha_1 [alpha_1]: 1 slug from OCS beta_1 [beta_1]; 4 GJ" in since
        assert "Torpedoes - yours: 2 launched, 2 hit, 1 blinded by PD | theirs: none" in since
        assert "Gun rounds - yours: none | theirs: 1 fired, 1 hit" in since


# =============================================================================
# ADM-03 rendering: CAPTAIN REPORTS from admiral.captain_reports
# =============================================================================

REPORTS = [
    {"report_id": "r1", "faction": "alpha", "checkpoint": 6, "captain_name": "C1",
     "ship_name": "TIS Dart-1", "status": "ACK", "reason": "", "intent": "",
     "contact": None, "request": None,
     "compliance": {"ordered_maneuver": "INTERCEPT", "executed_maneuver": "INTERCEPT",
                    "ordered_target": "beta_2", "actual_target": "beta_2",
                    "ordered_salvo": 2, "launched": 2}},
    {"report_id": "r2", "faction": "alpha", "checkpoint": 6, "captain_name": "C2",
     "ship_name": "TIS Dart-2", "status": "DEVIATE",
     "reason": "Falchion-2 is at 325 km and separating - a launch now is a low-value lob",
     "intent": "hold fire, close first", "contact": {"ship": "OCS Falchion-2",
                                                      "assessment": "opening 3.6 km/s",
                                                      "note": "receding"},
     "request": {"kind": "retarget", "text": "Assign me Lancer-3 instead"},
     "compliance": {"ordered_maneuver": "INTERCEPT", "executed_maneuver": "EVASIVE",
                    "ordered_target": "beta_6", "actual_target": "beta_6",
                    "ordered_salvo": 2, "launched": 0}},
    {"report_id": "r3", "faction": "alpha", "checkpoint": 6, "captain_name": "C3",
     "ship_name": "TIS Dart-3", "status": "ACK", "reason": "", "intent": "",
     "contact": None, "request": None,
     "compliance": {"ordered_maneuver": "INTERCEPT", "executed_maneuver": "HEADING",
                    "ordered_target": None, "actual_target": None,
                    "ordered_salvo": None, "launched": 0}},
    {"report_id": "r4", "faction": "alpha", "checkpoint": 6, "captain_name": "C4",
     "ship_name": "TIS Dart-4", "status": "UNABLE", "reason": "magazine empty",
     "intent": None, "contact": None, "request": None, "compliance": None},
    {"report_id": "r5", "faction": "alpha", "checkpoint": 6, "captain_name": "C5",
     "ship_name": "TIS Dart-5", "status": "ACK", "reason": "", "intent": "",
     "contact": None, "request": None, "compliance": None},
]


class TestCaptainReports:

    def test_non_ack_first_and_plain_acks_collapsed(self):
        live = battle_a_turn("alpha", t=180, reports=REPORTS)[3][1]["content"]
        block = section(live, "=== CAPTAIN REPORTS (filed at checkpoint 6) ===").splitlines()
        assert block[1].startswith("UNABLE TIS Dart-4 [alpha_4]: \"magazine empty\"")
        assert block[2].startswith("DEVIATE TIS Dart-2 [alpha_2]: \"Falchion-2 is at 325 km")
        assert "| contact: OCS Falchion-2 opening 3.6 km/s - receding" in block[2]
        assert "| asks [retarget]: Assign me Lancer-3 instead" in block[2]
        assert "did: ordered INTERCEPT -> flew EVASIVE, salvo 2 ordered, 0 launched (NOT as ordered)" in block[2]
        # An ACK that flew something else is not collapsed away.
        assert block[3].startswith("ACK TIS Dart-3 [alpha_3] | did: ordered INTERCEPT -> flew HEADING")
        assert block[4] == "ACK, nothing to flag (2): TIS Dart-1 [alpha_1], TIS Dart-5 [alpha_5]"
        # Near the top: right after the NOW block.
        assert live.index("=== CAPTAIN REPORTS") < live.index("=== GEOMETRY")

    def test_malformed_report_fields_do_not_break_the_turn(self):
        odd = [{"ship_name": "TIS Dart-6", "status": None, "reason": 42,
                "compliance": {"ordered_salvo": "2", "launched": None,
                               "ordered_maneuver": None, "executed_maneuver": "EVASIVE"},
                "contact": "not a dict", "request": {"text": None}}]
        live = battle_a_turn("alpha", t=180, reports=odd)[3][1]["content"]
        assert ("ACK TIS Dart-6 [alpha_6]: \"42\" | did: flew EVASIVE, salvo 2 ordered, "
                "0 launched (NOT as ordered)") in live

    def test_missing_or_empty_reports_still_render_the_path(self):
        admiral, sim, snap, _ = battle_a_turn("alpha", t=180)
        assert admiral._captain_reports() == []           # attribute absent
        admiral.captain_reports = []
        msgs = build_admiral_messages(
            admiral_name="A", faction="alpha", snapshot_t_minus_15=None, snapshot_t_zero=snap,
            personality=None, fleet_data=FLEET, phase="directive",
            captain_reports=admiral._captain_reports())
        assert "(none yet - captains report after their first decision)" in msgs[1]["content"]
        snap.checkpoint = 4
        text = format_admiral_turn_state(snap, FLEET, captain_reports=[])
        assert "(no reports filed last checkpoint)" in text

    def test_comms_verdict_is_authoritative(self):
        """An A-shaped report (fleet_comms.compliance): the target is written as
        the turn state labels it, the comms layer judged it followed - it must
        not be flagged by comparing raw strings."""
        followed = {"ship_name": "TIS Dart-2", "ship_id": "alpha_2", "status": "ACK",
                    "compliance": {"ordered_maneuver": "INTERCEPT", "executed_maneuver": "INTERCEPT",
                                   "ordered_target": "OCS Lancer-1 [beta_2]",
                                   "actual_target": "OCS Lancer-1", "ordered_salvo": None,
                                   "launched": 0, "final": True, "followed": True,
                                   "deviations": []}}
        deviated = {"ship_name": "TIS Dart-3", "ship_id": "alpha_3", "status": "ACK",
                    "compliance": {"ordered_maneuver": "INTERCEPT/EVADE",
                                   "executed_maneuver": "HEADING",
                                   "ordered_target": "OCS Lancer-1", "actual_target": None,
                                   "ordered_salvo": 3, "launched": 1, "final": False,
                                   "followed": False,
                                   "deviations": ["maneuver: ordered INTERCEPT/EVADE, flew HEADING",
                                                  "target: ordered OCS Lancer-1, engaging nothing",
                                                  "salvo: ordered 3, committed 1"]}}
        live = battle_a_turn("alpha", t=180, reports=[followed, deviated])[3][1]["content"]
        block = section(live, "=== CAPTAIN REPORTS")
        assert "ACK, nothing to flag (1): TIS Dart-2 [alpha_2]" in block
        assert ("ACK TIS Dart-3 [alpha_3] | did: maneuver: ordered INTERCEPT/EVADE, flew HEADING; "
                "target: ordered OCS Lancer-1, engaging nothing; salvo: ordered 3, committed 1 "
                "(NOT as ordered)") in block

    def test_legacy_record_resolves_labels_and_flags_a_dropped_target(self):
        def report(ot, at, salvo=None, launched=0):
            return {"ship_name": "TIS Dart-2", "ship_id": "alpha_2", "status": "ACK",
                    "compliance": {"ordered_maneuver": "INTERCEPT",
                                   "executed_maneuver": "INTERCEPT", "ordered_target": ot,
                                   "actual_target": at, "ordered_salvo": salvo,
                                   "launched": launched}}

        def block(r):
            return section(battle_a_turn("alpha", t=180, reports=[r])[3][1]["content"],
                           "=== CAPTAIN REPORTS")

        # Label vs name vs id of the same hull: complied.
        for ot, at in (("OCS Lancer-1 [beta_2]", "OCS Lancer-1"), ("beta_2", "OCS Lancer-1"),
                       ("ocs lancer-1", "OCS Lancer-1 [beta_2]")):
            assert "nothing to flag (1)" in block(report(ot, at)), (ot, at)
        # Ordered a target, engaging nothing: a deviation (it used to need both set).
        text = block(report("OCS Lancer-1 [beta_2]", None))
        assert "target OCS Lancer-1 [beta_2] -> nothing (NOT as ordered)" in text
        text = block(report("OCS Lancer-1", "OCS Lancer-2"))
        assert "target OCS Lancer-1 [beta_2] -> OCS Lancer-2 [beta_3] (NOT as ordered)" in text
        # An explicit hold, broken.
        text = block(report(None, None, salvo=0, launched=2))
        assert "salvo hold ordered, 2 launched (NOT as ordered)" in text

    def test_order_targets_written_as_labels_reach_the_stack_as_names(self):
        """The turn state teaches 'NAME [ID]'; captains, the comms compliance
        matcher and the recorder match bare names / ids."""
        client = ScriptedClient([([DIRECTIVE], "tool_calls")])

        def order_call(messages, tools, **kw):
            return [ToolCall(id="o", name="issue_order", arguments={
                "ship_name": "TIS Dart-1 [alpha_1]", "order_text": "Kill Lancer-1.",
                "suggested_target": "OCS Lancer-1 [beta_2]", "torpedo_salvo": 2,
                "torpedo_target": "Lancer-1 [beta_2]"})]

        admiral, decision, _ = _decide(client)
        assert decision.fleet_orders
        admiral.client = types.SimpleNamespace(decide_with_tools=order_call)
        order = admiral._issue_ship_order(
            ship_info={"ship_id": "alpha_1", "ship_name": "TIS Dart-1", "ship_type": "corvette",
                       "captain_name": "C1"},
            snapshot_t0=admiral._prev_snapshot, fleet_directive="Hold.",
            simulation=battle_a_sim(180))
        assert order.suggested_target == "OCS Lancer-1"
        assert order.torpedo_target == "OCS Lancer-1"            # the id decides
        # Phase-1 style issue_order: the ship label resolves to its id.
        legacy = admiral._execute_tools([ToolCall(id="x", name="issue_order", arguments={
            "ship_name": "TIS Dart-1 [alpha_1]", "order_text": "Hold.",
            "suggested_target": "[beta_2]"})], 180.0)
        assert legacy.fleet_orders[0].target_ship_id == "alpha_1"
        assert legacy.fleet_orders[0].suggested_target == "OCS Lancer-1"

    def test_per_ship_order_shows_that_ships_report(self):
        admiral, sim, snap, _ = battle_a_turn("alpha", t=180)
        admiral.captain_reports = REPORTS
        msgs = build_admiral_ship_order_messages(
            admiral_name="A", ship_name="TIS Dart-2", ship_type="corvette", captain_name="C2",
            fleet_directive="Hold the wall.", snapshot=snap, ship_id="alpha_2",
            ship_view=admiral._ship_view(sim, "alpha_2", snap), captain_report=REPORTS[1])
        user = msgs[1]["content"]
        assert "ITS CAPTAIN'S LAST REPORT: DEVIATE TIS Dart-2 [alpha_2]" in user
        assert "(NOT as ordered)" in user


# =============================================================================
# ADM-02: directive failure - retry, then an honest empty directive
# =============================================================================

class ScriptedClient:
    """decide_with_tools answers from a script of (calls, finish_reason)."""

    def __init__(self, script, reasoning_effort="medium", max_tokens=8192):
        self.script = list(script)
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.calls = []
        self._meta = {}

    def decide_with_tools(self, messages, tools, model=None, temperature=None,
                          tool_choice="auto", max_tokens=None, reasoning_effort=None):
        names = {t["function"]["name"] for t in tools}
        self.calls.append({"messages": messages, "names": names, "max_tokens": max_tokens,
                           "reasoning_effort": reasoning_effort})
        if names == {"issue_order"}:
            self._meta = {"finish_reason": "tool_calls"}
            return [ToolCall(id="o", name="issue_order",
                             arguments={"ship_name": "x", "order_text": "Hold."})]
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        calls, finish = item
        self._meta = {"finish_reason": finish, "max_tokens": max_tokens or self.max_tokens,
                      "reasoning_tokens": 7005 if finish == "length" else None}
        return calls

    def last_call_meta(self):
        return dict(self._meta)

    def complete(self, *a, **k):  # pragma: no cover
        raise AssertionError("no captain discussion here")


DIRECTIVE = ToolCall(id="d", name="set_fleet_directive", arguments={"directive": "Hold the wall."})
PLAN = ToolCall(id="p", name="set_battle_plan", arguments={"plan": "Phase 1: wall."})


def _decide(client, t=180):
    admiral = make_admiral("alpha", client=client)
    sim = battle_a_sim(t)
    return admiral, admiral.decide(simulation=sim, captains=captains_for("alpha")), sim


class TestDirectiveFailure:

    def test_truncation_is_retried_with_less_reasoning(self):
        client = ScriptedClient([([PLAN], "length"), ([DIRECTIVE], "tool_calls")])
        admiral, decision, sim = _decide(client)
        phase1 = [c for c in client.calls if "set_fleet_directive" in c["names"]]
        assert len(phase1) == 2
        assert phase1[1]["reasoning_effort"] == "low"
        assert "RETRY: your previous answer" in phase1[1]["messages"][-1]["content"]
        # The cached system prefix is untouched by the retry.
        assert phase1[0]["messages"][0] == phase1[1]["messages"][0]
        assert decision.fleet_directive == "Hold the wall."
        assert decision.directive_failure is None
        assert decision.directive_recovered.startswith("truncated at 8192 tokens, 7005")
        assert decision.battle_plan_update == "Phase 1: wall."   # kept from the cut-off answer
        assert admiral.last_directive == "Hold the wall."
        assert admiral.last_directive_t == sim.current_time

    def test_at_low_effort_the_retry_gets_more_room_instead(self):
        client = ScriptedClient([([], "length"), ([DIRECTIVE], "tool_calls")],
                                reasoning_effort="low")
        _decide(client)
        retry = [c for c in client.calls if "set_fleet_directive" in c["names"]][1]
        assert retry["reasoning_effort"] is None
        assert retry["max_tokens"] == 12288

    def test_persistent_failure_leaves_no_stale_directive(self):
        client = ScriptedClient([([DIRECTIVE], "tool_calls")] + [([], "length")] * 2)
        admiral = make_admiral("alpha", client=client)
        admiral.decide(simulation=battle_a_sim(165), captains=captains_for("alpha"))
        assert admiral.last_directive == "Hold the wall."
        sim = battle_a_sim(180)
        decision = admiral.decide(simulation=sim, captains=captains_for("alpha"))
        assert decision.call_failed
        assert decision.directive_failure.startswith("truncated at 8192 tokens")
        assert admiral.last_directive == ""
        assert admiral.last_directive_t == 180.0
        orders = [c for c in client.calls if c["names"] == {"issue_order"}][-8:]
        for call in orders:
            text = call["messages"][-1]["content"]
            assert 'You have already set the fleet directive:\n""' not in text
            assert "NO FLEET DIRECTIVE THIS CHECKPOINT - the directive call failed (truncated" in text
            assert "Your last directive (T+165s - STALE" in text
            # Nothing here asserts what captains do or do not see of the old
            # one: the captain prompt may echo it, labelled as not current.
            assert "captains do not see" not in text and "Captains receive none" not in text
        # The next checkpoint's NOW block says so.
        client.script = [([DIRECTIVE], "tool_calls")]
        admiral.decide(simulation=battle_a_sim(270), captains=captains_for("alpha"))
        phase1 = [c for c in client.calls if "set_fleet_directive" in c["names"]][-1]
        now = phase1["messages"][-1]["content"]
        assert "Your fleet directive at T+180s FAILED (truncated" in now
        assert "captains were told there was no new directive" in now
        assert "flew without one" not in now

    def test_call_error_is_a_failed_checkpoint_without_retry(self):
        client = ScriptedClient([LLMCallError("HTTP 400: bad model")])
        admiral, decision, _ = _decide(client)
        assert decision.call_failed and decision.directive_failure.startswith("call error")
        assert admiral.last_directive == ""
        assert len(client.calls) == 1   # no retry, no per-ship orders

    def test_runner_never_hands_captains_a_stale_directive(self, tmp_path):
        """
        Checkpoint 1's directive is never handed to a checkpoint-2 captain as
        its CURRENT directive. The captain prompt may still quote it, but only
        on a line that says there is no new fleet directive (the captain
        workstream echoes the last one, labelled with its age). Judged per
        captain turn, not by counting calls: a captain may get follow-up calls.
        """
        import re
        from src.llm.battle_runner import BattleConfig, LLMBattleRunner
        from src.llm.captain import LLMCaptainConfig
        from src.llm.client import CallStats
        from src.llm.fleet_config import BattleFleetConfig
        from src.llm.prompts import CaptainPersonality

        script = [([ToolCall(id="d", name="set_fleet_directive",
                             arguments={"directive": "CHECKPOINT-ONE DIRECTIVE"})], "tool_calls"),
                  ([], "length"), ([], "length")]
        captain_turns = []          # (directive calls made so far, prompt text)
        directive_calls = [0]

        def flatten(messages):
            out = []
            for m in messages:
                content = m.get("content") if isinstance(m, dict) else None
                if isinstance(content, list):
                    out.extend(str(part.get("text", "")) for part in content
                               if isinstance(part, dict))
                elif content:
                    out.append(str(content))
            return "\n".join(out)

        class Stub(ScriptedClient):
            stats = CallStats()
            session_id = None

            def decide_with_tools(self, messages, tools, model=None, temperature=None,
                                  tool_choice="auto", max_tokens=None, reasoning_effort=None):
                names = {t["function"]["name"] for t in tools}
                if "set_fleet_directive" in names:
                    directive_calls[0] += 1
                elif names != {"issue_order"}:
                    captain_turns.append((directive_calls[0], flatten(messages)))
                    return [ToolCall(id="c", name="respond_to_orders",
                                     arguments={"response_type": "ACKNOWLEDGE"})]
                return super().decide_with_tools(messages, tools, model, temperature,
                                                 tool_choice, max_tokens, reasoning_effort)

        fleet_config = BattleFleetConfig.from_dict({
            "battle_name": "Stale", "time_limit_s": 300.0, "decision_interval_s": 30.0,
            "initial_distance_km": 600.0,
            "alpha_fleet": {"admiral": {"model": "stub/m"},
                            "ships": [{"ship_type": "destroyer", "model": "stub/m"}]},
            "beta_fleet": {"ships": [{"ship_type": "destroyer", "model": "stub/m"}]},
        })
        config = BattleConfig(initial_distance_km=600.0, time_limit_s=300.0, max_checkpoints=2,
                              verbose=False, personality_selection=False, record_battle=True,
                              recording_dir=str(tmp_path), parallel_llm=False, seed=7)
        dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/m",
                                 personality=CaptainPersonality.BALANCED)
        runner = LLMBattleRunner(config=config, alpha_config=dummy, beta_config=dummy,
                                 client=Stub(script), fleet_config=fleet_config)
        runner.run_fleet_battle(FLEET)

        current = re.compile(r"FLEET DIRECTIVE[^:\n]*:\s*CHECKPOINT-ONE DIRECTIVE")
        first = [text for n, text in captain_turns if n == 1]
        assert any(current.search(text) for text in first), "control: checkpoint 1 got it"
        second = [text for n, text in captain_turns if n >= 2]
        assert second, "the checkpoint-2 captains were never prompted"
        for text in second:
            assert not current.search(text), \
                "the checkpoint-1 directive reached a checkpoint-2 captain as current"
            for line in text.splitlines():
                if "CHECKPOINT-ONE DIRECTIVE" in line:
                    assert "no new fleet directive" in line.lower(), line
        failed = [e for e in runner.recorder.events if e.event_type == "admiral_directive_failed"]
        assert len(failed) == 1 and failed[0].data["faction"] == "alpha"
        assert failed[0].data["reason"].startswith("truncated at 8192 tokens")


def test_ticker_prints_the_failed_directive_and_cache_line():
    import io
    from src.llm.battle_ticker import BattleTicker
    from src.llm.client import CallStats

    stats = CallStats()
    stats.record_usage({"prompt_tokens": 1000,
                        "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 0}})
    stats.record_usage({"prompt_tokens": 1000,
                        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 900}})
    events = [types.SimpleNamespace(event_type="admiral_directive_failed", ship_id=None,
                                    timestamp=210.0,
                                    data={"faction": "beta", "reason": "truncated at 8192 tokens"}),
              types.SimpleNamespace(event_type="admiral_directive", ship_id=None, timestamp=210.0,
                                    data={"faction": "alpha", "directive": "Hold the wall."})]
    runner = types.SimpleNamespace(
        simulation=types.SimpleNamespace(current_time=210.0), checkpoint_count=7,
        recorder=types.SimpleNamespace(events=events), alpha_ships={}, beta_ships={},
        alpha_admiral=object(), beta_admiral=object(),
        client=types.SimpleNamespace(stats=stats))
    stream = io.StringIO()
    BattleTicker(stream=stream).checkpoint(runner)
    text = stream.getvalue()
    assert "A admiral: Hold the wall." in text
    assert ("B admiral: (DIRECTIVE FAILED: truncated at 8192 tokens - no new directive for "
            "the captains this checkpoint)") in text
    assert "Prompt cache: 1 hit / 1 write / 0 cold calls, 0.8k read + 0.9k written" in text


# =============================================================================
# R6 / ADM-08: per-ship orders in the ordered ship's frame
# =============================================================================

def test_per_ship_order_ranges_are_from_the_ordered_ship():
    sim = CombatSimulation(decision_interval=30.0, seed=1)
    ship(sim, "alpha_1", "cruiser_torpedo", "alpha", (0, 0, 0))
    ship(sim, "alpha_2", "cruiser_torpedo", "alpha", (0, 300, 0))
    foe = ship(sim, "beta_1", "battlecruiser", "beta", (400, 0, 0), fwd=(-1, 0, 0))
    admiral = make_admiral("alpha")
    caps = caps_from_sim(sim, "alpha")
    snap = admiral._build_snapshot(sim, caps, full=True)
    texts = {}
    for sid in ("alpha_1", "alpha_2"):
        msgs = build_admiral_ship_order_messages(
            admiral_name="A", ship_name=f"TIS {sid}", ship_type="cruiser_torpedo",
            captain_name="C", fleet_directive="", snapshot=snap, ship_id=sid,
            ship_view=admiral._ship_view(sim, sid, snap), directive_failure="no set_fleet_directive call")
        texts[sid] = msgs[1]["content"]
    d1 = (foe.position - sim.get_ship("alpha_1").position).magnitude / 1000.0
    d2 = (foe.position - sim.get_ship("alpha_2").position).magnitude / 1000.0
    assert f"OCS beta_1 [beta_1] BC: {d1:.0f} km" in texts["alpha_1"]
    assert f"OCS beta_1 [beta_1] BC: {d2:.0f} km" in texts["alpha_2"]
    assert round(d1) != round(d2)
    for text in texts.values():
        assert "fleet directive:\n\"\"" not in text
        assert "| torpedo GOOD" in text or "| torpedo POOR" in text
        assert "=== RANGE CONTROL (world-frame headings for issue_order heading= for this ship" in text


# =============================================================================
# ADM-10 / ADM-11: doctrine for the fleets in the battle, cacheable split
# =============================================================================

class TestDoctrineAndCaching:

    def test_torpedo_fleet_gets_no_gun_doctrine_and_gun_fleet_no_torpedo_doctrine(self):
        torp = build_admiral_messages(
            admiral_name="A", faction="alpha", snapshot_t_minus_15=None, snapshot_t_zero=None,
            personality=None, fleet_data=FLEET, phase="directive",
            battle_classes={"ours": ["corvette"], "theirs": ["battleship"]})[0]["content"]
        assert "WHEN TO COMMIT TRIDENTS" in torp and "FLEET TACTICAL OPTIONS" not in torp
        assert "ABUNDANT" not in torp and "TORPEDOES ARE SCARCE" in torp
        assert "corvette 8 rounds (1 tube x 8)" in torp
        guns = build_admiral_messages(
            admiral_name="A", faction="alpha", snapshot_t_minus_15=None, snapshot_t_zero=None,
            personality=None, fleet_data=FLEET, phase="directive",
            battle_classes={"ours": ["battleship"], "theirs": ["destroyer"]})[0]["content"]
        assert "FLEET TACTICAL OPTIONS" in guns
        assert "WHEN TO COMMIT TRIDENTS" not in guns and "HOW POINT DEFENSE" not in guns
        assert "TORPEDO verdicts" not in guns
        # A one-weapon fleet's doctrine drops the other weapon's whole block.
        from src.llm.prompts import ADMIRAL_GUN_DOCTRINE, build_admiral_static
        everything = build_admiral_static("A", "alpha", FLEET, None, phase="directive")
        torp_doctrine = torp.split("=== FORCES")[0]
        assert len(torp_doctrine) <= len(everything) - 0.9 * len(ADMIRAL_GUN_DOCTRINE)

    def test_system_prompt_is_byte_stable_across_checkpoints(self):
        admiral = make_admiral("alpha")
        caps = captains_for("alpha")
        systems = []
        for t in (165, 180, 270):
            snap = admiral._build_snapshot(battle_a_sim(t), caps, full=True)
            msgs = build_admiral_messages(
                admiral_name="A", faction="alpha", snapshot_t_minus_15=None, snapshot_t_zero=snap,
                personality="Calm.", fleet_data=FLEET, phase="directive", standing_plan=f"plan {t}",
                standing_plan_time=float(t), battle_classes=admiral._battle_classes,
                captain_reports=REPORTS if t == 270 else None)
            systems.append(msgs[0]["content"])
            assert f"plan {t}" in msgs[1]["content"]
            assert ADMIRAL_STATE_MARKER in msgs[1]["content"]
        assert systems[0] == systems[1] == systems[2]
        assert "=== FORCES" in systems[0] and "Battlecruiser (BC)" in systems[0]
        assert "T+" not in systems[0]

    def test_decide_sends_static_system_and_live_user_turn(self):
        client = ScriptedClient([([DIRECTIVE], "tool_calls")] * 2)
        admiral = make_admiral("alpha", client=client)
        admiral.decide(simulation=battle_a_sim(165), captains=captains_for("alpha"))
        admiral.decide(simulation=battle_a_sim(180), captains=captains_for("alpha"))
        phase1 = [c for c in client.calls if "set_fleet_directive" in c["names"]]
        assert phase1[0]["messages"][0] == phase1[1]["messages"][0]
        assert phase1[1]["messages"][1]["content"].startswith("ADMIRAL CHECKPOINT 2.")
        assert "=== ENEMY THREAT TABLE" in phase1[1]["messages"][1]["content"]


# =============================================================================
# ADM-07: golden text + size budget
# =============================================================================

def test_turn_state_golden_text_battle_a_t180():
    """
    The whole alpha turn state at battle A T+180 (13 ships), pinned. Regenerate
    after an intended format change with UPDATE_GOLDEN=1 and read the diff.
    """
    live = battle_a_turn("alpha", t=180, reports=REPORTS)[3][1]["content"]
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(live + "\n")
    assert live + "\n" == GOLDEN.read_text()


@pytest.mark.parametrize("faction", ["alpha", "beta"])
@pytest.mark.parametrize("t", [180, 270])
def test_live_state_fits_the_token_budget(faction, t):
    """<= 2.5k tokens of live state at 14 ships (~3.5 chars per token here)."""
    live = battle_a_turn(faction, t=t, reports=REPORTS)[3][1]["content"]
    assert len(live) / 3.5 <= 2500, len(live)
