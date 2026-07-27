"""
Invariant tests for torpedo terminal guidance: augmented proportional
navigation (APN) with a no-escape-zone (NEZ) commit logic.

These tests assert PHYSICAL INVARIANTS, not magic constants copied from any
particular run:

- the target-acceleration estimator converges on a constant-g target;
- the APN command chases the target's escape direction and carries the
  (N/2) * a_target feed-forward term;
- the NEZ shrinks as target acceleration rises, as range grows, and as
  delta-v is spent; a fuel-exhausted or separating torpedo can never claim it;
- a torpedo launched inside its NEZ kills a 3 g evader regardless of when
  the evasion starts;
- a torpedo launched far outside its NEZ against an early, hard 3 g jink
  still misses (it is NOT unconditionally lethal);
- hit rate is monotonically non-increasing in evasion lead time;
- a delta-v reserve survives to impact on a clean intercept, and a
  fuel-exhausted torpedo flies ballistically (cannot correct at all).
"""

import json
import math

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    create_ship_from_fleet_data,
)
from src.torpedo import (
    APN_NAV_RATIO,
    ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2,
    GuidanceMode,
    NEZ_DV_RESERVE_KPS,
    Torpedo,
    TorpedoGuidance,
    TorpedoSpecs,
)

G = 9.81


@pytest.fixture(scope="module")
def fleet_data():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _torpedo(position=(0, 0, 0), velocity=(13000, 0, 0)) -> Torpedo:
    return Torpedo(
        specs=TorpedoSpecs.trident(),
        position=Vector3D(*position),
        velocity=Vector3D(*velocity),
        target_id="t",
        guidance_mode=GuidanceMode.COLLISION,
    )


# =============================================================================
# Target-acceleration estimator
# =============================================================================

class TestTargetAccelEstimator:

    def test_estimate_converges_on_constant_acceleration(self):
        """Feeding a constant-g velocity history must converge on that g."""
        torp = _torpedo()
        guidance = TorpedoGuidance()
        accel = Vector3D(0.0, 3.0 * G, 0.0)
        vel = Vector3D(-2000.0, 0.0, 0.0)
        dt = 0.5
        for _ in range(12):  # 6 s of observations
            vel = vel + accel * dt
            est = guidance._update_target_accel_estimate(torp, vel, dt)
        err = (est - accel).magnitude
        assert err < 0.05 * accel.magnitude, (
            f"estimator off by {err:.2f} m/s^2 after 6 s of constant-g data"
        )

    def test_first_observation_yields_no_estimate(self):
        """One sample cannot be differenced - the estimate must stay zero."""
        torp = _torpedo()
        est = TorpedoGuidance()._update_target_accel_estimate(
            torp, Vector3D(500.0, 0.0, 0.0), 0.5
        )
        assert est.magnitude == 0.0


# =============================================================================
# APN command law
# =============================================================================

class TestAPNCommand:

    def test_command_chases_the_escape_direction(self):
        """Target drifting +y relative to the torpedo -> command points +y."""
        torp = _torpedo(velocity=(13000, 0, 0))
        cmd = TorpedoGuidance().update_collision_guidance(
            torp, Vector3D(150_000, 0, 0), Vector3D(0, 800.0, 0), 0.5
        )
        assert cmd.throttle > 0.0, cmd.reason
        assert cmd.direction.y > 0.0, (
            f"command {cmd.direction} does not chase a +y escape ({cmd.reason})"
        )

    def test_feed_forward_term_matches_apn_theory(self):
        """
        With zero lateral rate and a known constant target acceleration, the
        commanded lateral acceleration must be (N/2) * a_target_perp -
        independent of range. This is the augmentation the old law lacked.
        """
        guidance = TorpedoGuidance()
        a_t = 3.0 * G
        torp = _torpedo(velocity=(13000, 0, 0))
        # Prime the estimator with a perfectly constant-g history.
        vel = Vector3D(0.0, 0.0, 0.0)
        dt = 0.5
        for _ in range(40):
            vel = vel + Vector3D(0.0, a_t, 0.0) * dt
            guidance._update_target_accel_estimate(torp, vel, dt)

        # Reconstruct the command magnitude from the law's own geometry:
        # zero lateral velocity => a_cmd = (N/2) * a_T_perp.
        est = torp.estimated_target_accel
        assert abs(est.y - a_t) < 0.05 * a_t
        expected = (APN_NAV_RATIO / 2.0) * a_t

        # Torpedo co-moving laterally with the target (v_lat = 0).
        torp.velocity = Vector3D(13000.0, vel.y, 0.0)
        torp.last_observed_target_velocity = None  # do not disturb estimate
        cmd = guidance.update_collision_guidance(
            torp, Vector3D(torp.position.x + 200_000, 0, 0), vel, 0.0
        )
        # The commanded acceleration is throttle * available accel along the
        # lateral component of the direction.
        avail = torp.specs.acceleration_at_mass(torp.current_mass_kg)
        rcs_avail = torp.specs.rcs_thrust_n / torp.current_mass_kg
        authority = rcs_avail if cmd.use_rcs else avail
        lateral_frac = abs(cmd.direction.y)
        applied = cmd.throttle * authority * lateral_frac
        assert applied == pytest.approx(expected, rel=0.15), (
            f"applied lateral accel {applied:.1f} m/s^2 vs APN feed-forward "
            f"(N/2)*a_T = {expected:.1f} m/s^2 ({cmd.reason})"
        )


# =============================================================================
# No-escape zone
# =============================================================================

class TestNEZ:

    def _nez(self, torp, distance_m=150_000.0, a_t=ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2):
        return TorpedoGuidance().nez_status(
            torp, Vector3D(torp.position.x + distance_m, torp.position.y, 0),
            Vector3D(0, 0, 0), target_max_lateral_accel_ms2=a_t,
        )

    def test_fuel_exhausted_torpedo_cannot_claim_nez(self):
        torp = _torpedo()
        torp.fuel_exhausted = True
        nez = self._nez(torp)
        assert not nez.inside
        assert "fuel" in nez.reason

    def test_burning_the_whole_budget_leaves_the_nez(self):
        """Actually spending the propellant (not just a flag) exits the NEZ."""
        torp = _torpedo()
        assert self._nez(torp).inside
        while not torp.fuel_exhausted:
            torp.apply_thrust(Vector3D(1, 0, 0), 5.0)
        assert not self._nez(torp).inside

    def test_separating_torpedo_is_outside(self):
        torp = _torpedo(velocity=(-5000, 0, 0))  # flying away
        nez = self._nez(torp)
        assert not nez.inside
        assert nez.reason == "not closing"

    def test_nez_shrinks_as_target_acceleration_rises(self):
        """
        Both margins must be monotonically non-increasing in target
        acceleration, and once the torpedo falls outside it must never
        re-enter as the target gets MORE agile.
        """
        torp = _torpedo()
        prev_dv_margin = math.inf
        prev_accel_margin = math.inf
        was_outside = False
        saw_inside = False
        for g_level in [0.5, 1, 2, 3, 4, 6, 8, 12, 20, 40]:
            nez = self._nez(torp, a_t=g_level * G)
            assert nez.dv_margin_kps <= prev_dv_margin + 1e-9
            assert nez.accel_margin <= prev_accel_margin + 1e-9
            if nez.inside:
                saw_inside = True
                assert not was_outside, (
                    f"NEZ re-entered at {g_level} g after being outside"
                )
            else:
                was_outside = True
            prev_dv_margin = nez.dv_margin_kps
            prev_accel_margin = nez.accel_margin
        assert saw_inside, "NEZ empty even vs a near-inert target - too strict"
        assert was_outside, "NEZ still claimed vs a 40 g target - too lax"

    def test_commit_range_decreases_with_target_agility(self):
        """The maximum commit range must be non-increasing in target accel."""
        def max_commit_range(a_t):
            torp = _torpedo()
            best = 0.0
            for d_km in range(50, 4001, 50):
                if self._nez(torp, distance_m=d_km * 1000.0, a_t=a_t).inside:
                    best = d_km
            return best

        ranges = [max_commit_range(g_level * G) for g_level in (1.0, 2.0, 3.0)]
        assert ranges[0] >= ranges[1] >= ranges[2], (
            f"commit range grew with target agility: {ranges} km for 1/2/3 g"
        )
        assert ranges[2] > 0, "no commit range at all vs a 3 g target"

    def test_nez_shrinks_with_range(self):
        """Farther launch (longer t_go) must never increase the dv margin."""
        torp = _torpedo()
        margins = [
            self._nez(torp, distance_m=d).dv_margin_kps
            for d in (50_000, 150_000, 400_000, 1_000_000, 2_500_000)
        ]
        for near, far in zip(margins, margins[1:]):
            assert far <= near + 1e-9, f"dv margin grew with range: {margins}"

    def test_spending_delta_v_erodes_the_nez(self):
        """The same geometry with less remaining delta-v has less margin."""
        fresh = _torpedo()
        spent = _torpedo()
        spent.apply_thrust(Vector3D(1, 0, 0), 20.0)  # burn a chunk
        spent.velocity = Vector3D(*fresh.velocity.to_tuple())  # same geometry
        m_fresh = self._nez(fresh).dv_margin_kps
        m_spent = self._nez(spent).dv_margin_kps
        assert m_spent < m_fresh

    def test_requirements_scale_with_the_derivation(self):
        """
        Zero lateral rate: required accel = margin * a_t and required dv =
        margin * a_t * t_go + reserve, straight from the documented
        derivation.
        """
        torp = _torpedo(velocity=(13000, 0, 0))
        d = 130_000.0
        nez = self._nez(torp, distance_m=d)
        t_go = d / 13000.0
        a_t = ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2
        margin = APN_NAV_RATIO / 2.0
        assert nez.time_to_go_s == pytest.approx(t_go, rel=1e-6)
        assert nez.target_escape_displacement_m == pytest.approx(
            0.5 * a_t * t_go ** 2, rel=1e-6
        )
        assert nez.required_lateral_accel_ms2 == pytest.approx(
            margin * a_t, rel=1e-6
        )
        assert nez.required_delta_v_kps == pytest.approx(
            margin * a_t * t_go / 1000.0 + NEZ_DV_RESERVE_KPS, rel=1e-6
        )


# =============================================================================
# Integration: full simulation against an evading corvette
# =============================================================================

def _engagement(fleet_data, seed, onset_s, launch_range_m,
                shooter_v=0.0, target_v=0.0, evasion="jink",
                max_steps=4000, nez_probe_time=None):
    """
    Launch one torpedo at T=0; target begins 3 g evasion at onset_s.

    evasion="jink": hard constant lateral BURN at full (3 g) throttle.
    evasion="evasive": the stock EVASIVE maneuver.

    Returns (hit, closest_approach_m, torpedo, nez_probe).
    """
    sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=seed)
    shooter = create_ship_from_fleet_data(
        "alpha", "corvette", "alpha", fleet_data,
        position=Vector3D(0, 0, 0), velocity=Vector3D(shooter_v, 0, 0),
        forward=Vector3D(1, 0, 0))
    target = create_ship_from_fleet_data(
        "beta", "corvette", "beta", fleet_data,
        position=Vector3D(launch_range_m, 0, 0),
        velocity=Vector3D(target_v, 0, 0), forward=Vector3D(-1, 0, 0))
    sim.add_ship(shooter)
    sim.add_ship(target)
    # Point defense off: this measures guidance, not PD attrition.
    shooter.point_defense = []
    target.point_defense = []
    assert sim.inject_command("alpha", {"type": "launch_torpedo",
                                        "target_id": "beta"})
    guidance = TorpedoGuidance()
    started = False
    tf = None
    nez_probe = None
    steps = 0
    while sim.torpedoes and steps < max_steps:
        if not started and sim.current_time >= onset_s:
            if evasion == "evasive":
                m = Maneuver(ManeuverType.EVASIVE, start_time=sim.current_time,
                             duration=0.0, throttle=1.0)
            else:
                m = Maneuver(ManeuverType.BURN, start_time=sim.current_time,
                             duration=0.0, throttle=1.0,
                             direction=Vector3D(0, 1, 0))
            sim.inject_command("beta", m)
            started = True
        tf = sim.torpedoes[0]
        if nez_probe_time is not None and nez_probe is None \
                and sim.current_time >= nez_probe_time:
            nez_probe = guidance.nez_status(
                tf.torpedo, target.position, target.velocity)
        sim.step()
        steps += 1
    hit = sim.metrics.total_torpedo_hits > 0
    cpa = tf.min_distance_to_target if tf else float("inf")
    return hit, cpa, (tf.torpedo if tf else None), nez_probe


class TestInsideNEZKillsEvader:

    @pytest.mark.parametrize("onset_s", [0.0, 10.0, 20.0])
    def test_launched_inside_nez_hits_a_3g_jinker(self, fleet_data, onset_s):
        """
        300 km at 12 km/s closure (~23 s flight): the launch is inside the
        NEZ, so a 3 g jink at ANY onset must die. The old law scored 0 here.
        """
        hit, cpa, _, nez = _engagement(
            fleet_data, seed=1, onset_s=onset_s, launch_range_m=300_000,
            shooter_v=6000.0, target_v=-6000.0, nez_probe_time=1.0)
        assert nez is not None and nez.inside, (
            f"engagement was supposed to start inside the NEZ: {nez}"
        )
        assert hit, f"inside-NEZ torpedo missed by {cpa/1000:.2f} km"

    def test_launched_inside_nez_hits_the_stock_evasive_pattern(self, fleet_data):
        hit, cpa, _, _ = _engagement(
            fleet_data, seed=2, onset_s=5.0, launch_range_m=300_000,
            shooter_v=6000.0, target_v=-6000.0, evasion="evasive")
        assert hit, f"missed the stock EVASIVE corvette by {cpa/1000:.2f} km"


class TestOutsideNEZCanMiss:

    def test_early_hard_jink_far_outside_nez_escapes(self, fleet_data):
        """
        2300 km cold launch: the torpedo spends nearly its whole budget on
        closure, so a target that jinks from T+0 outruns the remaining
        correction budget. The torpedo must NOT be unconditionally lethal.
        """
        hit, cpa, torp, nez = _engagement(
            fleet_data, seed=1, onset_s=0.0, launch_range_m=2_300_000,
            nez_probe_time=30.0)
        assert nez is not None and not nez.inside, (
            "engagement was supposed to be outside the NEZ at probe time"
        )
        assert not hit, "torpedo hit despite being far outside its NEZ"
        assert cpa > 500.0, f"escape should be clean, cpa={cpa:.0f} m"
        # The miss is a delta-v story: the budget ran out chasing.
        assert torp.remaining_delta_v_kps < 1.0

    def test_hit_rate_monotone_in_evasion_lead_time(self, fleet_data):
        """
        Evading earlier (more lead time before impact) must never HELP the
        torpedo. Swept across the NEZ boundary at 2300 km, hit rate must be
        monotonically non-decreasing in onset time (equivalently,
        non-increasing in lead time) - and must actually change across the
        sweep so the assertion is not vacuous.
        """
        onsets = [0.0, 120.0, 160.0]
        seeds = [1, 2]
        rates = []
        for onset in onsets:
            hits = sum(
                1 for s in seeds
                if _engagement(fleet_data, seed=s, onset_s=onset,
                               launch_range_m=2_300_000)[0]
            )
            rates.append(hits / len(seeds))
        for earlier, later in zip(rates, rates[1:]):
            assert earlier <= later, (
                f"hit rate increased with MORE evasion lead time: "
                f"{list(zip(onsets, rates))}"
            )
        assert rates[-1] > rates[0], (
            f"sweep never crossed the NEZ boundary (rates {rates}) - "
            f"monotonicity held vacuously"
        )


class TestReserveAndBallistics:

    def test_reserve_survives_a_clean_intercept(self, fleet_data):
        """
        The NEZ terminal burn must stop above the reserve: a torpedo that
        hits must not have burned bone-dry on the way in.
        """
        hit, _, torp, _ = _engagement(
            fleet_data, seed=1, onset_s=10.0, launch_range_m=300_000,
            shooter_v=6000.0, target_v=-6000.0)
        assert hit
        assert not torp.fuel_exhausted
        assert torp.remaining_delta_v_kps > 0.1, (
            f"only {torp.remaining_delta_v_kps:.2f} km/s left at impact - "
            f"the reserve did not survive"
        )

    def test_fuel_exhausted_torpedo_flies_ballistically(self):
        """
        A dead-engine torpedo on a 1 km-miss trajectory must stay on it -
        guidance can issue no correction whatsoever.
        """
        torp = _torpedo(velocity=(13000, 0, 0))
        while not torp.fuel_exhausted:
            torp.apply_thrust(Vector3D(0, 0, 1), 5.0)
        torp.velocity = Vector3D(13000.0, 0.0, 0.0)

        target_pos = Vector3D(130_000.0, 1_000.0, 0.0)  # 1 km lateral offset
        target_vel = Vector3D(0.0, 0.0, 0.0)
        dt = 0.5
        best = float("inf")
        v_before = Vector3D(*torp.velocity.to_tuple())
        for _ in range(60):
            torp.update(dt, target_pos, target_vel)
            best = min(best, torp.position.distance_to(target_pos))
        assert (torp.velocity - v_before).magnitude == 0.0, (
            "fuel-exhausted torpedo changed velocity"
        )
        assert best > 900.0, (
            f"ballistic torpedo 'corrected' to {best:.0f} m without fuel"
        )
