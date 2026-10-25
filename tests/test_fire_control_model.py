"""
Fire-control P(hit) = conversion x track (src/firecontrol.py).

The engine lands a slug in two steps: the round must pass inside the
HIT_TOLERANCE_M gate around the hull (a physical question - where did the
target go during the flight?), then it rolls against the conversion
probability times a geometry factor. P(hit) used to be the conversion number
alone, so it ignored the gate - which is where nearly every long-range round
was lost (0/600+ beyond 200 km against dodging targets while captains were
shown 12-19%). These tests pin the model's algebra, its wiring into the
simulation (one shared call for gate, forecasts and MCP) and the rule that
the dispersion roll consumes the CONVERSION part only.
"""

import json
import math
import random

import pytest

from src.firecontrol import (
    DODGE_ACCEL_FRACTION,
    EVASIVE_DODGE_ACCEL_FRACTION,
    HIT_TOLERANCE_M,
    JINK_ACCEL_FRACTION,
    PENDING_BURN_FRACTION,
    TERMINAL_DISPERSION_SIGMA_M,
    UNFORESEEN_ACCEL_FRACTION,
    TargetMotion,
    WeaponsCommand,
    WeaponsOrder,
    calculate_hit_probability,
    displacement_sigma_m,
    track_probability,
)
from src.physics import Vector3D
from src.simulation import (
    BALLISTIC_THREAT_HORIZON_S,
    EVASION_MIN_JINK_S,
    EVASION_TURN_ALLOWANCE_S,
    HEADING_BURN_CONE_DEG,
    CombatSimulation,
    Maneuver,
    ManeuverType,
    ProjectileInFlight,
    SimulationEventType,
    create_ship_from_fleet_data,
)


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------

class TestTrackProbability:

    def test_no_displacement_is_certain(self):
        assert track_probability(0.0, 515.0) == 1.0

    def test_matches_the_engines_gate_and_geometry_factor(self):
        """Closed form == Monte Carlo of exactly what the engine does with a
        round whose target ended up a 2-D Gaussian away from the aim point:
        gate on |r| < R, then multiply by exp(-r^2 / 2 sigma_t^2)."""
        rng = random.Random(7)
        R = HIT_TOLERANCE_M + 15.6
        for sigma in (50.0, 250.0, 600.0, 2000.0):
            n, acc = 60_000, 0.0
            for _ in range(n):
                x, y = rng.gauss(0, sigma), rng.gauss(0, sigma)
                r2 = x * x + y * y
                if r2 < R * R:
                    acc += math.exp(-r2 / (2 * TERMINAL_DISPERSION_SIGMA_M ** 2))
            assert track_probability(sigma, R) == pytest.approx(acc / n, abs=0.006), sigma

    def test_decreases_with_displacement(self):
        values = [track_probability(s, 515.0) for s in (0, 100, 300, 1000, 5000)]
        assert values == sorted(values, reverse=True)
        # Far outside the envelope it falls as (sigma_t / sigma)^2.
        st = TERMINAL_DISPERSION_SIGMA_M
        assert track_probability(5000.0, 515.0) == pytest.approx(
            (st / 5000.0) ** 2 * (1 - math.exp(-515.0 ** 2 / (2 * st ** 2))), rel=0.01)


class TestDisplacementSigma:

    def test_grows_as_flight_time_squared(self):
        m = TargetMotion(accel_ms2=19.62)
        assert displacement_sigma_m(30.0, m) == pytest.approx(
            UNFORESEEN_ACCEL_FRACTION * 19.62 * 30.0 ** 2 / 2)
        assert displacement_sigma_m(40.0, m) / displacement_sigma_m(20.0, m) == pytest.approx(4.0)

    def test_a_target_with_no_drive_cannot_surprise(self):
        assert displacement_sigma_m(60.0, TargetMotion(accel_ms2=0.0, dodges=True)) == 0.0

    def test_dodging_needs_the_evasions_minimum_time_to_go(self):
        calm = TargetMotion(accel_ms2=19.62)
        dodger = TargetMotion(accel_ms2=19.62, dodges=True)
        assert displacement_sigma_m(6.0, dodger) == displacement_sigma_m(6.0, calm)
        assert displacement_sigma_m(20.0, dodger) > 5 * displacement_sigma_m(20.0, calm)

    def test_dodge_term(self):
        m = TargetMotion(accel_ms2=10.0, dodges=True, dodge_throttle=0.5)
        t = 40.0
        free = UNFORESEEN_ACCEL_FRACTION * 10.0 * t * t / 2
        dodge = DODGE_ACCEL_FRACTION * 10.0 * 0.5 * t * t / 2
        assert displacement_sigma_m(t, m) == pytest.approx(math.hypot(free, dodge))
        evasive = TargetMotion(accel_ms2=10.0, dodges=True, evasive=True, dodge_throttle=0.5)
        dodge = EVASIVE_DODGE_ACCEL_FRACTION * 10.0 * 0.5 * t * t / 2
        assert displacement_sigma_m(t, evasive) == pytest.approx(math.hypot(free, dodge))

    def test_jink_term_is_the_burn_the_lead_misses(self):
        """A dodger caught mid-jink keeps being pushed by the burn its lead
        ignores, from the moment of firing (no minimum time to go: it is
        already moving). The EVASIVE maneuver's lead and dodge fraction
        cover its own jinks."""
        t = 20.0
        m = TargetMotion(accel_ms2=10.0, dodges=True, jink_accel_ms2=8.0)
        free = UNFORESEEN_ACCEL_FRACTION * 10.0 * t * t / 2
        dodge = DODGE_ACCEL_FRACTION * 10.0 * t * t / 2
        jink = JINK_ACCEL_FRACTION * 8.0 * t * t / 2
        assert displacement_sigma_m(t, m) == pytest.approx(math.sqrt(free ** 2 + dodge ** 2 + jink ** 2))
        assert displacement_sigma_m(5.0, m) == pytest.approx(math.hypot(
            UNFORESEEN_ACCEL_FRACTION * 10.0 * 12.5, JINK_ACCEL_FRACTION * 8.0 * 12.5))
        evasive = TargetMotion(accel_ms2=10.0, dodges=True, evasive=True, jink_accel_ms2=8.0)
        assert displacement_sigma_m(t, evasive) == displacement_sigma_m(
            t, TargetMotion(accel_ms2=10.0, dodges=True, evasive=True))

    def test_pending_burn_grows_from_when_it_lights(self):
        m = TargetMotion(accel_ms2=10.0, pending_burn_ms2=6.0, pending_burn_in_s=12.0)
        free = lambda t: UNFORESEEN_ACCEL_FRACTION * 10.0 * t * t / 2  # noqa: E731
        assert displacement_sigma_m(10.0, m) == pytest.approx(free(10.0))
        burn = PENDING_BURN_FRACTION * 6.0 * 18.0 ** 2 / 2
        assert displacement_sigma_m(30.0, m) == pytest.approx(math.hypot(free(30.0), burn))

    def test_dodging_starts_at_the_threat_horizon(self):
        """Evasion ignores rounds more than dodge_horizon_s out: a 90 s round
        is dodged for the last 60 s only."""
        m = TargetMotion(accel_ms2=10.0, dodges=True, dodge_horizon_s=60.0)
        free = UNFORESEEN_ACCEL_FRACTION * 10.0 * 90.0 ** 2 / 2
        dodge = DODGE_ACCEL_FRACTION * 10.0 * 60.0 ** 2 / 2
        assert displacement_sigma_m(90.0, m) == pytest.approx(math.hypot(free, dodge))


class TestCalculateHitProbability:

    def _solve(self, fleet, range_km, motion, evading=False):
        target = create_ship_from_fleet_data("t", "destroyer", "beta", fleet)
        return calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0), shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(range_km * 1000.0, 0, 0), target_velocity=Vector3D(0, 0, 0),
            target_geometry=target.geometry, target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=9.9, target_is_evading=evading, target_motion=motion)

    def test_without_motion_it_is_the_conversion_number(self, fleet):
        s = self._solve(fleet, 300, None)
        assert s.track_probability == 1.0
        assert s.hit_probability == s.conversion_probability

    def test_hit_probability_is_conversion_times_track(self, fleet):
        motion = TargetMotion(accel_ms2=19.62, dodges=True)
        s = self._solve(fleet, 300, motion)
        legacy = self._solve(fleet, 300, None)
        assert s.conversion_probability == legacy.conversion_probability
        assert s.hit_probability == pytest.approx(s.conversion_probability * s.track_probability)
        assert s.displacement_sigma_m == pytest.approx(displacement_sigma_m(s.time_of_flight_s, motion))
        assert s.track_probability == pytest.approx(
            track_probability(s.displacement_sigma_m, HIT_TOLERANCE_M + 15.625))

    def test_an_alert_dodger_is_unhittable_at_long_range(self, fleet):
        """The measured fact behind the flash-mirror 0/609: a destroyer that
        dodges is out of reach of a 300 km spinal round (30 s flight)."""
        dodger = TargetMotion(accel_ms2=19.62, dodges=True)
        calm = TargetMotion(accel_ms2=19.62)
        assert self._solve(fleet, 300, dodger).hit_probability < 0.03
        assert self._solve(fleet, 300, calm).hit_probability > 0.15
        # ... while a close round still connects: no time to dodge.
        assert self._solve(fleet, 50, dodger).hit_probability == pytest.approx(
            self._solve(fleet, 50, calm).hit_probability, rel=0.01)


# ---------------------------------------------------------------------------
# Wiring into the simulation
# ---------------------------------------------------------------------------

def _duel(fleet, range_km=300, target_type="destroyer", seed=5):
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    shooter = create_ship_from_fleet_data(
        "a", "battleship", "alpha", fleet, position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    target = create_ship_from_fleet_data(
        "b", target_type, "beta", fleet, position=Vector3D(range_km * 1000.0, 0, 0),
        forward=Vector3D(-1, 0, 0))
    sim.add_ship(shooter)
    sim.add_ship(target)
    return sim, shooter, target


class TestSimulationWiring:

    def test_target_motion_reads_the_live_ship(self, fleet):
        sim, _, target = _duel(fleet)
        m = sim.target_motion(target)
        assert m.accel_ms2 == pytest.approx(target.kinematic_state.max_acceleration_ms2())
        assert m.dodges is False and m.jink_accel_ms2 == 0.0 and m.pending_burn_ms2 == 0.0
        assert m.dodge_horizon_s == BALLISTIC_THREAT_HORIZON_S
        assert m.dodge_min_time_s == EVASION_TURN_ALLOWANCE_S + EVASION_MIN_JINK_S

    @pytest.mark.parametrize("maneuver", [
        Maneuver(ManeuverType.EVASIVE, 0.0, throttle=1.0),
        Maneuver(ManeuverType.PADLOCK, 0.0, throttle=0.8, target_id="a", evade="dodge"),
        Maneuver(ManeuverType.HOLD_RANGE, 0.0, throttle=1.0, target_id="a", evade="full"),
    ])
    def test_evasion_armed_targets_dodge(self, fleet, maneuver):
        """Armed = the evasion reacts to rounds: EVASIVE or the evade overlay
        on any maneuver (the flash-mirror fleets flew PADLOCK + dodge and
        looked like coasters to the old EVASIVE-only test)."""
        sim, _, target = _duel(fleet)
        target.current_maneuver = maneuver
        m = sim.target_motion(target)
        assert m.dodges is True
        assert m.dodge_throttle == maneuver.throttle
        assert m.evasive == (maneuver.maneuver_type == ManeuverType.EVASIVE)

    def test_a_dodger_mid_jink_is_charged_the_burn_its_lead_ignores(self, fleet):
        """PADLOCK + dodge, evasion flying it broadside-on across the line of
        fire at full throttle: the lead assumes it coasts, so the whole burn
        is missed. Burning straight at the shooter it moves along the line of
        fire only - nothing across it. Between jinks (the base maneuver
        flying) there is nothing to charge."""
        sim, shooter, target = _duel(fleet)
        target.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0.0, target_id="a", evade="dodge")
        target.applied_throttle = 1.0
        target.kinematic_state.forward = Vector3D(0, 1, 0)
        target.execution_mode = "PADLOCK"
        assert sim.target_motion(target, shooter).jink_accel_ms2 == 0.0
        target.execution_mode = "EVADE"
        a = target.kinematic_state.max_acceleration_ms2()
        assert sim.target_motion(target, shooter).jink_accel_ms2 == pytest.approx(a, rel=1e-6)
        target.kinematic_state.forward = Vector3D(-1, 0, 0)
        assert sim.target_motion(target, shooter).jink_accel_ms2 == pytest.approx(0.0, abs=1e-6)
        target.current_maneuver = Maneuver(ManeuverType.EVASIVE, 0.0, throttle=1.0)
        assert sim.target_motion(target, shooter).jink_accel_ms2 == 0.0

    def test_a_turning_heading_target_carries_its_pending_burn(self, fleet):
        """HEADING broadside to the line of fire, nose still on the shooter:
        the ordered full burn lights once the nose is inside the burn cone,
        and that is what the forecast charges from then on."""
        sim, shooter, target = _duel(fleet)
        target.current_maneuver = Maneuver(
            ManeuverType.HEADING, 0.0, throttle=1.0, heading_direction={"x": 0.0, "y": 1.0, "z": 0.0})
        target.applied_throttle = 0.05
        m = sim.target_motion(target, shooter)
        a = target.kinematic_state.max_acceleration_ms2()
        assert m.pending_burn_ms2 == pytest.approx(a, rel=0.01)
        assert 0.0 < m.pending_burn_in_s < sim._estimate_turn_time(target, math.pi / 2, engines_on=True)
        attitude = Maneuver(ManeuverType.HEADING, 0.0, throttle=0.0,
                            heading_direction={"x": 0.0, "y": 1.0, "z": 0.0})
        target.current_maneuver = attitude
        assert sim.target_motion(target, shooter).pending_burn_ms2 == 0.0

    @pytest.mark.parametrize("hull, angle_deg", [("corvette", 150), ("destroyer", 90), ("battleship", 120)])
    def test_burn_light_time_matches_the_turn(self, fleet, hull, angle_deg):
        """The forecast's 'burn lights in' is the engine's own turn: fly the
        HEADING and the drive lights within a tick or two of the estimate,
        both from rest and mid-turn (a from-rest turn to the cone edge was
        up to 2x long for a ship already swinging round)."""
        sim, _, ship = _duel(fleet, target_type=hull)
        th = math.radians(angle_deg)
        d = Vector3D(math.cos(th), math.sin(th), 0.0)
        ship.kinematic_state.forward = Vector3D(1, 0, 0)
        ship.current_maneuver = Maneuver(ManeuverType.HEADING, 0.0, duration=1e9, throttle=1.0,
                                         heading_direction={"x": d.x, "y": d.y, "z": 0.0})
        estimates = []
        while getattr(ship, "applied_throttle", 0.0) < 0.5 and sim.current_time < 200:
            estimates.append((sim.current_time, sim._heading_burn_light_s(ship, d)))
            sim.step()
        lit = sim.current_time
        for t, est in (estimates[0], estimates[len(estimates) // 2]):
            assert abs((t + est) - lit) <= 2.5, (hull, t, est, lit)
        assert ship.forward.angle_to(d) <= math.radians(HEADING_BURN_CONE_DEG) + 1e-6

    def test_one_number_for_gate_forecast_and_record(self, fleet):
        sim, shooter, target = _duel(fleet, range_km=150)
        target.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0.0, target_id="a", evade="dodge")
        slot, ws = next((k, w) for k, w in shooter.weapons.items() if not w.weapon.is_turreted)
        solution = sim.gun_firing_solution(shooter, target, ws)
        assert 0 < solution.hit_probability < solution.conversion_probability
        shooter.weapons_orders = {slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "b")}
        sim._process_weapons_orders(shooter)
        shot = next(e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED)
        assert shot.data["hit_probability"] == pytest.approx(solution.hit_probability)
        assert shot.data["conversion_probability"] == pytest.approx(solution.conversion_probability)
        assert sim.projectiles[-1].conversion_probability == pytest.approx(solution.conversion_probability)

    def test_fire_when_optimal_gates_on_the_calibrated_number(self, fleet):
        """A 0.10 gate holds fire on an alert dodger at 300 km (a wasted
        round: the 0/609 of the flash mirror) and fires on the same hull when
        it coasts."""
        results = {}
        for armed in (False, True):
            sim, shooter, target = _duel(fleet, range_km=300)
            target.current_maneuver = Maneuver(
                ManeuverType.PADLOCK, 0.0, target_id="a", evade="dodge" if armed else None)
            slot = next(k for k, w in shooter.weapons.items() if not w.weapon.is_turreted)
            shooter.weapons_orders = {slot: WeaponsOrder(
                WeaponsCommand.FIRE_WHEN_OPTIMAL, slot, "b", min_hit_probability=0.10)}
            sim._process_weapons_orders(shooter)
            results[armed] = len(sim.projectiles)
        assert results == {False: 1, True: 0}

    def test_dispersion_roll_consumes_the_conversion_part_only(self, fleet):
        """The gate has just been decided physically, so the roll must not
        charge the target's motion again: a round that reached the hull with
        conversion 1.0 lands even if the forecast was 0."""
        sim, shooter, target = _duel(fleet, range_km=10)
        ws = next(w for w in shooter.weapons.values() if not w.weapon.is_turreted)
        sim._launch_projectile(shooter, target, ws, hit_probability=0.0,
                               conversion_probability=1.0, weapon_slot="weapon_0")
        flight: ProjectileInFlight = sim.projectiles[-1]
        sim._resolve_projectile_hit_geometric(flight, target, None, target.position)
        assert not [e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_MISS]
        assert [e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_IMPACT]


class TestTheLeadReadsThePlume:
    """The lead extrapolates the drive's actual output for HEADING and BURN
    targets, not the order (a HEADING ship turns first on a token throttle;
    reading 'full burn along the heading' over-led rounds by 0.5 a T^2)."""

    def test_turning_heading_target_is_not_led_at_full_burn(self, fleet):
        sim, _, target = _duel(fleet)
        target.current_maneuver = Maneuver(
            ManeuverType.HEADING, 0.0, throttle=1.0, heading_direction={"x": 0.0, "y": 1.0, "z": 0.0})
        target.applied_throttle = 0.05          # still swinging round (token throttle)
        accel = sim._estimate_target_acceleration(target)
        assert accel.magnitude == pytest.approx(0.05 * target.kinematic_state.max_acceleration_ms2(), rel=1e-6)
        assert accel.normalized().dot(target.forward) == pytest.approx(1.0)

    def test_burning_target_is_led_with_its_burn(self, fleet):
        sim, _, target = _duel(fleet)
        target.current_maneuver = Maneuver(ManeuverType.BURN, 0.0, throttle=1.0, direction=Vector3D(0, 1, 0))
        target.kinematic_state.forward = Vector3D(0, 1, 0)
        target.applied_throttle = 1.0
        accel = sim._estimate_target_acceleration(target)
        assert accel.magnitude == pytest.approx(target.kinematic_state.max_acceleration_ms2(), rel=1e-6)
        assert accel.y > 0
