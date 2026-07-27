"""
Regression tests for three defects found by measuring the shipped simulation.

1. Point defense booked NOTHING when it blinded a torpedo. The only counter,
   metrics.total_torpedo_intercepted, was gated on the 1 GJ structural kill,
   which the live dwell path can never reach (PD drops a torpedo from its
   target list the instant the 50 MJ seeker threshold is crossed). Blinds and
   hard kills are now counted separately - deliberately NOT summed, because a
   blinded torpedo still hits a target that does not maneuver.

2. The admiral was told every inbound kinetic round was a "Coilgun",
   regardless of what fired it. This is prompt surface an LLM reads.

3. A round's actual flight geometry had no influence on whether it hit. The
   outcome was decided purely by a probability frozen at launch, so PD ablation
   recoil and post-launch evasion were physically simulated and causally inert.

Every number asserted here was measured against the shipped code first.
"""

import io
import json
import math
from contextlib import redirect_stdout
from unittest.mock import Mock

import pytest

from src.physics import Vector3D
from src.pointdefense import (
    PDLaser,
    PD_WALLPLUG_EFFICIENCY,
    TORPEDO_ELECTRONICS_THRESHOLD_J,
    TORPEDO_WARHEAD_THRESHOLD_J,
)
from src.projectile import KineticProjectile
from src.simulation import (
    CombatSimulation,
    HIT_TOLERANCE_M,
    Maneuver,
    ManeuverType,
    PDLaserState,
    ProjectileInFlight,
    SimulationEventType,
    TERMINAL_DISPERSION_SIGMA_M,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs


@pytest.fixture(scope="module")
def fleet_data():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _quiet(fn, *args, **kwargs):
    """Run fn with the simulation's console chatter suppressed."""
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# =============================================================================
# 1. Point defense must book blinds, and must not call them interceptions
# =============================================================================

def _salvo_against_screen(fleet_data, n_escorts, salvo=6, evading=False, seed=23):
    """Torpedo cruiser salvos a battleship sitting behind n destroyer escorts."""
    sim = CombatSimulation(seed=seed, time_step=0.5)
    victim = create_ship_from_fleet_data(
        "victim", "battleship", "alpha", fleet_data,
        position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    sim.add_ship(victim)
    if evading:
        victim.current_maneuver = Maneuver(
            maneuver_type=ManeuverType.EVASIVE, start_time=0.0, throttle=1.0)
    for i in range(n_escorts):
        ang = 2 * math.pi * i / max(n_escorts, 1)
        sim.add_ship(create_ship_from_fleet_data(
            f"esc{i}", "destroyer", "alpha", fleet_data,
            position=Vector3D(30_000 * math.cos(ang), 30_000 * math.sin(ang), 0),
            forward=Vector3D(1, 0, 0)))
    sim.add_ship(create_ship_from_fleet_data(
        "att", "cruiser_torpedo", "beta", fleet_data,
        position=Vector3D(350_000, 0, 0), forward=Vector3D(-1, 0, 0)))

    launched, steps = 0, 0
    while steps < 1200:
        if launched < salvo and sim.inject_command(
                "att", {"type": "launch_torpedo", "target_id": "victim"}):
            launched += 1
        sim.step()
        steps += 1
        if launched >= salvo and not sim.torpedoes:
            break
    return sim


class TestPointDefenseBooksWhatItActuallyDid:

    def test_blinding_torpedoes_is_counted(self, fleet_data):
        """
        The whole defect: a screen that blinds every incoming round used to
        leave every PD counter reading zero.
        """
        sim = _quiet(_salvo_against_screen, fleet_data, n_escorts=4)

        disabled_events = [e for e in sim.events
                           if e.event_type == SimulationEventType.PD_TORPEDO_DISABLED]
        assert disabled_events, "scenario must actually blind torpedoes"
        assert sim.metrics.total_torpedo_seeker_killed == len(disabled_events) > 0, (
            "seeker kills must be counted, and counted once per torpedo"
        )

    def test_a_blind_is_never_reported_as_an_interception(self, fleet_data):
        """
        Blinds must not be folded into total_torpedo_intercepted. PD that only
        ever blinds has intercepted nothing - the rounds are still inbound.
        """
        sim = _quiet(_salvo_against_screen, fleet_data, n_escorts=4)

        destroyed_events = [e for e in sim.events
                            if e.event_type == SimulationEventType.PD_TORPEDO_DESTROYED]
        assert sim.metrics.total_torpedo_seeker_killed > 0
        assert sim.metrics.total_torpedo_intercepted == len(destroyed_events), (
            "the interception counter must track structural kills ONLY"
        )

    def test_blinded_torpedoes_that_still_hit_are_recorded(self, fleet_data):
        """
        Measured on the shipped sim: 6 torpedoes launched, 6 blinded by a
        4-destroyer screen, and all 6 still hit a battleship that did not
        maneuver. That has to be visible in the metrics, or PD's contribution
        reads as a save when nothing was saved.
        """
        sim = _quiet(_salvo_against_screen, fleet_data, n_escorts=4, evading=False)

        assert sim.metrics.total_torpedo_seeker_killed > 0
        assert sim.metrics.total_torpedo_hits_after_seeker_kill > 0, (
            "a blinded torpedo still hits a non-maneuvering target - say so"
        )
        assert (sim.metrics.total_torpedo_hits_after_seeker_kill
                <= sim.metrics.total_torpedo_hits)
        # Nothing was actually taken off the board here.
        assert sim.metrics.total_torpedo_neutralized == 0

    def test_evasion_is_what_converts_a_blind_into_a_save(self, fleet_data):
        """The blind only buys something if the target then moves."""
        still = _quiet(_salvo_against_screen, fleet_data, n_escorts=8, evading=False)
        moving = _quiet(_salvo_against_screen, fleet_data, n_escorts=8, evading=True)

        assert still.metrics.total_torpedo_seeker_killed > 0
        assert moving.metrics.total_torpedo_seeker_killed > 0
        assert moving.metrics.total_torpedo_hits < still.metrics.total_torpedo_hits
        assert moving.metrics.total_torpedo_neutralized > still.metrics.total_torpedo_neutralized

    def test_disable_event_is_emitted_once_per_torpedo(self, fleet_data):
        """
        Several turrets are deliberately assigned to one torpedo, so the tick
        that crosses the threshold used to emit one PD_TORPEDO_DISABLED per
        remaining turret. Measured before the fix: 6 torpedoes, 8 events.
        """
        sim = _quiet(_salvo_against_screen, fleet_data, n_escorts=8)

        ids = [e.data.get("torpedo_id") for e in sim.events
               if e.event_type == SimulationEventType.PD_TORPEDO_DISABLED]
        assert ids, "scenario must actually blind torpedoes"
        assert len(ids) == len(set(ids)), f"duplicate disable events: {ids}"

    def test_summary_line_states_blinds_and_kills_separately(self, fleet_data):
        sim = _quiet(_salvo_against_screen, fleet_data, n_escorts=4)
        summary = sim.metrics.torpedo_outcome_summary()
        assert "blinded by PD" in summary
        assert "destroyed by PD" in summary
        assert "still hit" in summary


class TestBothPointDefencePathsBookTheSameWay:
    """
    _pd_dwell_torpedo (the battle loop) and _pd_engage_torpedo (the one-call
    burst API) differ only in how much beam energy they deliver. What a
    delivered joule DOES to a torpedo must not depend on which one delivered
    it, so they share _apply_pd_torpedo_result. These tests fail if either
    path grows its own private copy of the bookkeeping again.
    """

    def _rig(self, fleet_data, n_turrets=4):
        sim = CombatSimulation(seed=11, time_step=1.0)
        ship = create_ship_from_fleet_data(
            "d", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        ship.point_defense = []
        for i in range(n_turrets):
            laser = PDLaser(power_mw=5.0, aperture_m=0.5, wavelength_nm=1000.0,
                            range_km=250.0, cooldown_s=5.0, name=f"xpd_{i}")
            ship.point_defense.append(
                PDLaserState(laser=laser, turret_name=f"xpd_{i}"))
            if ship.power_system:
                ship.power_system.add_weapon_capacitor(f"xpd_{i}", {
                    "type": "point_defense",
                    "power_draw_mw": 5.0 / PD_WALLPLUG_EFFICIENCY,
                    "cooldown_s": 5.0})
        sim.add_ship(ship)

        specs = TorpedoSpecs.from_fleet_data(
            warhead_yield_gj=0.0, penetrator_mass_kg=250.0,
            ammo_mass_kg=3600.0, exhaust_velocity_kps=8.0)
        torp = Torpedo(specs=specs, position=Vector3D(5_000, 0, 0),
                       velocity=Vector3D(-50, 0, 0), target_id="d")
        torp.armed = True
        torp.fuel_exhausted = True
        torp.guidance_mode = GuidanceMode.COAST
        tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="x")
        sim.torpedoes.append(tf)
        return sim, ship, tf

    def test_legacy_burst_path_books_the_seeker_kill(self, fleet_data):
        sim, ship, tf = self._rig(fleet_data)
        # Park the torpedo just under the seeker threshold so a single burst
        # from this path carries it over: the assertion under test is the
        # bookkeeping, not how many bursts the capacitor allows.
        tf.heat_absorbed_j = TORPEDO_ELECTRONICS_THRESHOLD_J * 0.999
        _quiet(sim._pd_engage_torpedo, ship, ship.point_defense[0], tf, 1.0)

        assert tf.is_disabled, "burst path must be able to blind a torpedo"
        assert sim.metrics.total_torpedo_seeker_killed == 1
        assert ship.pd_seeker_kills == 1
        assert sim.metrics.total_torpedo_intercepted == 0

    def test_dwell_path_books_the_seeker_kill(self, fleet_data):
        sim, ship, tf = self._rig(fleet_data)
        for _ in range(40):
            _quiet(sim.step)
            if tf.is_disabled or tf not in sim.torpedoes:
                break
        assert tf.is_disabled
        assert sim.metrics.total_torpedo_seeker_killed == 1
        assert ship.pd_seeker_kills == 1

    def test_a_structural_kill_still_books_an_interception(self, fleet_data):
        """
        The hard-kill path is real, just unreachable through normal dwell. Drive
        it directly so the interception counter keeps a live regression guard.
        """
        sim, ship, tf = self._rig(fleet_data)
        destroyed = tf.absorb_pd_heat(TORPEDO_WARHEAD_THRESHOLD_J)
        assert destroyed
        _quiet(sim._apply_pd_torpedo_result, ship, ship.point_defense[0], tf, destroyed)

        assert sim.metrics.total_torpedo_intercepted == 1
        assert ship.pd_intercepts == 1
        assert tf not in sim.torpedoes
        # A hard kill is not also a seeker kill.
        assert sim.metrics.total_torpedo_seeker_killed == 0

    def test_thresholds_are_ordered_as_the_metrics_assume(self):
        assert TORPEDO_ELECTRONICS_THRESHOLD_J < TORPEDO_WARHEAD_THRESHOLD_J


# =============================================================================
# 2. The admiral must be told which weapon actually fired
# =============================================================================

class TestAdmiralNamesTheRealWeapon:

    def _admiral(self, fleet_data):
        from src.llm.admiral import LLMAdmiral
        from src.llm.fleet_config import AdmiralConfig
        return LLMAdmiral(
            config=AdmiralConfig(model="test/model", name="T"),
            faction="alpha", client=Mock(), fleet_data=fleet_data,
        )

    def _sim_with_shot(self, fleet_data, shooter_type):
        from src.firecontrol import WeaponsCommand, WeaponsOrder
        sim = CombatSimulation(seed=3, time_step=1.0, decision_interval=1e9)
        shooter = create_ship_from_fleet_data(
            "shooter", shooter_type, "beta", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        target = create_ship_from_fleet_data(
            "target", "destroyer", "alpha", fleet_data,
            position=Vector3D(120_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(shooter)
        sim.add_ship(target)
        shooter.weapons_orders = {
            slot: WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE,
                               weapon_slot=slot, target_id="target",
                               min_hit_probability=0.0)
            for slot in shooter.weapons if not slot.startswith("pd_")
        }
        for _ in range(3):
            _quiet(sim.step)
            if sim.projectiles:
                break
        return sim

    def test_inbound_rounds_are_named_by_the_weapon_that_fired_them(self, fleet_data):
        sim = self._sim_with_shot(fleet_data, "battleship")
        assert sim.projectiles, "scenario must put a round in flight"
        snaps = self._admiral(fleet_data)._build_projectile_snapshots(sim)
        kinetic = [s for s in snaps if not s.weapon_type.startswith("Torpedo")]
        assert kinetic, "kinetic rounds must be reported"

        real_names = {p.weapon.name for p in sim.projectiles if p.weapon}
        assert real_names, "projectiles must carry the weapon that fired them"
        for snap in kinetic:
            assert snap.weapon_type in real_names, (
                f"admiral told {snap.weapon_type!r}, real weapons were {real_names}"
            )

    def test_a_spinal_round_is_not_reported_as_a_coilgun(self, fleet_data):
        """
        The hardcoded label made a 4.29 GJ spinal round indistinguishable from
        a 0.72 GJ turret round in the admiral's prompt.
        """
        sim = self._sim_with_shot(fleet_data, "battleship")
        spinal = [p for p in sim.projectiles
                  if p.weapon and "coiler" in p.weapon.weapon_type]
        if not spinal:
            pytest.skip("no spinal round in flight this tick")

        snaps = self._admiral(fleet_data)._build_projectile_snapshots(sim)
        labels = {s.weapon_type for s in snaps}
        assert labels != {"Coilgun"}, "label reverted to the hardcoded string"
        assert any("Coiler" in lbl for lbl in labels), (
            f"a spinal coiler round must be named as one; got {labels}"
        )

    def test_projectile_without_a_weapon_still_gets_a_label(self, fleet_data):
        """Tools and tests inject bare projectiles; the label must not crash."""
        from src.llm.admiral import _projectile_weapon_label
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-6000, 0, 0),
                                 position=Vector3D(10_000, 0, 0))
        pf = ProjectileInFlight(projectile_id="p", projectile=proj,
                                source_ship_id="a", target_ship_id="b")
        assert _projectile_weapon_label(pf) == "Kinetic round"

    def test_weapons_summary_uses_real_names_not_substring_guesses(self, fleet_data):
        """
        'heavy_siege_coiler_mk3' has no 'spinal' in its key, so substring
        matching reported this triple-nose spinal mount as a "Heavy Coilgun".
        """
        admiral = self._admiral(fleet_data)
        spec = {"weapons": [{"type": "heavy_siege_coiler_mk3"},
                            {"type": "light_coilgun_mk3"}]}
        summary = admiral._build_weapons_summary(spec)
        assert "Heavy Siege Coiler Mk3" in summary, summary
        assert "Light Coilgun Battery Mk3" in summary, summary
        assert "Heavy Coilgun" not in summary, (
            f"siege coiler still mislabelled: {summary}"
        )


# =============================================================================
# 3. Terminal geometry must influence the outcome
# =============================================================================

class TestMissMarginGeometry:
    """The measurement the terminal roll is built on has to be exact."""

    def _ship(self, fleet_data, forward=Vector3D(1, 0, 0)):
        sim = _quiet(CombatSimulation, seed=1)
        ship = create_ship_from_fleet_data(
            "d", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=forward)
        _quiet(sim.add_ship, ship)
        return sim, ship

    def test_a_round_through_the_hull_has_zero_margin(self, fleet_data):
        sim, ship = self._ship(fleet_data)
        assert sim._projectile_miss_margin_m(
            Vector3D(100_000, 0, 0), Vector3D(-6000, 0, 0), ship) == 0.0

    def test_margin_is_clearance_from_the_hull_surface(self, fleet_data):
        sim, ship = self._ship(fleet_data)
        offset = 300.0
        margin = sim._projectile_miss_margin_m(
            Vector3D(0, 100_000, offset), Vector3D(0, -6000, 0), ship)
        assert margin == pytest.approx(offset - ship.geometry.radius_m)

    def test_the_hull_is_a_capsule_not_a_point(self, fleet_data):
        """A round passing over the bow still hits a 125 m long ship."""
        sim, ship = self._ship(fleet_data)
        along_axis = ship.geometry.length_m / 2 - 1.0
        assert sim._projectile_miss_margin_m(
            Vector3D(along_axis, 100_000, 0), Vector3D(0, -6000, 0), ship) == 0.0

    def test_margin_is_measured_in_the_target_rest_frame(self, fleet_data):
        """
        A crossing target and a stationary one are different problems; the
        margin must follow the relative trajectory, not the absolute one.
        """
        sim, ship = self._ship(fleet_data)
        ship.kinematic_state.velocity = Vector3D(0, 2000, 0)
        # Round led perfectly: relative velocity points straight at the hull.
        margin = sim._projectile_miss_margin_m(
            Vector3D(100_000, 0, 0), Vector3D(-6000, 2000, 0), ship)
        assert margin == 0.0

    def test_target_position_argument_is_honoured(self, fleet_data):
        """
        Passing the wrong-instant target position invents miss distance. This
        is the trap that made a dead-on crossing shot look like a 1 km miss.
        """
        sim, ship = self._ship(fleet_data)
        contemporaneous = sim._projectile_miss_margin_m(
            Vector3D(100_000, 0, 0), Vector3D(-6000, 0, 0), ship,
            Vector3D(0, 0, 0))
        stale = sim._projectile_miss_margin_m(
            Vector3D(100_000, 0, 0), Vector3D(-6000, 0, 0), ship,
            Vector3D(0, 2000, 0))
        assert contemporaneous == 0.0
        assert stale > 1000.0


def _slug_engagement(fleet_data, n_escorts, seed, p_hit=0.8, escort_angle_deg=45):
    """
    A 40 kg slug at 6 km/s, dead-on at a PD-less corvette, optionally with
    escorts whose ablation recoil can push it off the line.

    Returns (deflection_m, landed).
    """
    result = {}
    sim = CombatSimulation(seed=seed, time_step=0.5)
    victim = create_ship_from_fleet_data(
        "v", "corvette", "alpha", fleet_data,
        position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    victim.point_defense = []
    sim.add_ship(victim)
    for i in range(n_escorts):
        a = math.radians(escort_angle_deg + i * 90)
        sim.add_ship(create_ship_from_fleet_data(
            f"esc{i}", "destroyer", "alpha", fleet_data,
            position=Vector3D(25_000 * math.cos(a), 25_000 * math.sin(a), 0),
            forward=Vector3D(1, 0, 0)))
    sim.add_ship(create_ship_from_fleet_data(
        "e", "corvette", "beta", fleet_data,
        position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0)))

    proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-6000, 0, 0),
                             position=Vector3D(250_000, 0, 0))
    pf = ProjectileInFlight(projectile_id="s", projectile=proj,
                            source_ship_id="e", target_ship_id="v",
                            hit_probability=p_hit)
    pf.launch_miss_margin_m = sim._projectile_miss_margin_m(
        proj.position, proj.velocity, victim)
    sim.projectiles.append(pf)

    original = CombatSimulation._resolve_projectile_hit_geometric

    def spy(self, proj_flight, target, impact_point, target_position=None):
        margin = self._projectile_miss_margin_m(
            proj_flight.projectile.position, proj_flight.projectile.velocity,
            target, target_position)
        before = self.metrics.total_hits
        original(self, proj_flight, target, impact_point, target_position)
        result["deflection"] = max(0.0, margin - proj_flight.launch_miss_margin_m)
        result["landed"] = self.metrics.total_hits > before

    sim._resolve_projectile_hit_geometric = spy.__get__(sim, CombatSimulation)
    for _ in range(300):
        sim.step()
        if pf not in sim.projectiles:
            break
    return result.get("deflection"), result.get("landed")


class TestAblationRecoilIsCausallyLive:
    """
    The defect: PD ablation recoil moved rounds 156-208 m and changed nothing,
    because the hit/miss decision never consulted where the round actually
    went. These are the tests that fail if the outcome goes back to being
    decided purely by the launch-time probability.
    """

    def test_an_escort_deflects_the_round_measurably(self, fleet_data):
        bare, _ = _quiet(_slug_engagement, fleet_data, 0, seed=0)
        escorted, _ = _quiet(_slug_engagement, fleet_data, 1, seed=0)
        assert bare == pytest.approx(0.0, abs=1.0), (
            "an unescorted dead-on round must not be deflected"
        )
        assert escorted > 100.0, (
            f"escort ablation recoil moved the round only {escorted:.0f} m"
        )
        assert escorted < HIT_TOLERANCE_M, (
            "deflection must stay inside the gate - otherwise this test is "
            "measuring the gate, not the terminal roll"
        )

    def test_deflection_lowers_the_realized_hit_rate(self, fleet_data):
        """
        Statistical, over a fixed seed sweep: the ONLY difference between the
        two arms is the escort's beam. Measured on the shipped code the two
        rates were identical (deflection was inert).
        """
        n = 60
        bare = sum(1 for s in range(n)
                   if _quiet(_slug_engagement, fleet_data, 0, seed=s)[1])
        escorted = sum(1 for s in range(n)
                       if _quiet(_slug_engagement, fleet_data, 1, seed=s)[1])
        assert bare > escorted, (
            f"escort deflection did not cost the shooter anything: "
            f"{bare}/{n} unescorted vs {escorted}/{n} escorted"
        )

    def test_a_symmetric_screen_cancels_its_own_push(self, fleet_data):
        """
        Four escorts on a ring push the round from opposite sides. The physics
        already cancelled; the point is that the outcome model agrees rather
        than inventing a penalty.
        """
        deflection, _ = _quiet(_slug_engagement, fleet_data, 4, seed=0)
        assert deflection == pytest.approx(0.0, abs=1.0)

    def test_deflection_shaves_the_odds_rather_than_flipping_them(self):
        """
        A 200 m nudge on a round fire control had dead to rights must cost
        real probability without being an automatic miss.
        """
        factor = math.exp(-(200.0 ** 2) / (2.0 * TERMINAL_DISPERSION_SIGMA_M ** 2))
        assert 0.5 < factor < 0.85, factor
        assert math.exp(0.0) == 1.0

    def test_a_clean_solution_is_completely_unaffected(self, fleet_data):
        """
        Zero deflection must leave the launch-time probability untouched, so
        unperturbed gun combat is not rebalanced by this mechanism at all.
        """
        n = 40
        landed = sum(1 for s in range(n)
                     if _quiet(_slug_engagement, fleet_data, 0, seed=s,
                               p_hit=1.0)[1])
        assert landed == n, (
            "a P(hit)=1.0 round with no deflection must always land"
        )


class TestFireControlLeadErrorIsNotChargedTwice:
    """
    The launch-time P(hit) already prices in flight time and crossing angle.
    Charging the terminal roll for the resulting geometric miss as well drove
    crossing-target hit rate from 22% to 0% when measured - a rebalance, not a
    bug fix. Only deflection acquired AFTER launch may count.
    """

    def test_a_constant_velocity_crossing_shot_is_not_penalised(self, fleet_data):
        from src.firecontrol import WeaponsCommand, WeaponsOrder

        deflections = []
        original = CombatSimulation._resolve_projectile_hit_geometric

        def spy(self, proj_flight, target, impact_point, target_position=None):
            margin = self._projectile_miss_margin_m(
                proj_flight.projectile.position,
                proj_flight.projectile.velocity, target, target_position)
            deflections.append(
                max(0.0, margin - proj_flight.launch_miss_margin_m))
            original(self, proj_flight, target, impact_point, target_position)

        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=42)
        shooter = create_ship_from_fleet_data(
            "shooter", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        target = create_ship_from_fleet_data(
            "target", "destroyer", "beta", fleet_data,
            position=Vector3D(150_000, 0, 0),
            velocity=Vector3D(0, 2000, 0),      # pure crossing, no maneuver
            forward=Vector3D(-1, 0, 0))
        shooter.point_defense = []
        target.point_defense = []
        sim.add_ship(shooter)
        sim.add_ship(target)
        shooter.weapons_orders = {
            slot: WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE,
                               weapon_slot=slot, target_id="target",
                               min_hit_probability=0.0)
            for slot in shooter.weapons if not slot.startswith("pd_")
        }
        sim._resolve_projectile_hit_geometric = spy.__get__(sim, CombatSimulation)
        for _ in range(90):
            _quiet(sim.step)

        assert deflections, "crossing engagement must resolve some rounds"
        assert max(deflections) < 1.0, (
            f"a constant-velocity crossing target produced up to "
            f"{max(deflections):.0f} m of 'deflection' - the firing solution's "
            f"own lead error is being charged a second time"
        )

    def test_launch_margin_is_recorded_on_every_round(self, fleet_data):
        from src.firecontrol import WeaponsCommand, WeaponsOrder
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=7)
        shooter = create_ship_from_fleet_data(
            "shooter", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        target = create_ship_from_fleet_data(
            "target", "destroyer", "beta", fleet_data,
            position=Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(shooter)
        sim.add_ship(target)
        shooter.weapons_orders = {
            slot: WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE,
                               weapon_slot=slot, target_id="target",
                               min_hit_probability=0.0)
            for slot in shooter.weapons if not slot.startswith("pd_")
        }
        for _ in range(3):
            _quiet(sim.step)
            if sim.projectiles:
                break
        assert sim.projectiles
        for pf in sim.projectiles:
            assert pf.launch_miss_margin_m >= 0.0
            assert math.isfinite(pf.launch_miss_margin_m)


class TestHitToleranceIsStillTheSingleGate:
    """
    The tolerance envelope must not be quietly shrunk to make deflection
    matter - that would rebalance every gun in the game at once.
    """

    def test_tolerance_is_unchanged(self):
        assert HIT_TOLERANCE_M == 500.0

    def test_dispersion_sigma_is_anchored_to_the_envelope(self):
        assert TERMINAL_DISPERSION_SIGMA_M == HIT_TOLERANCE_M / 2.0
