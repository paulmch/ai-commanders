"""
Invariant tests for the point-defense redesign.

The redesign replaces discrete fixed-value PD shots with a continuous-dwell
beam model (energy = intensity(range) x dwell x coupling), couples PD fire
into the ship's power and thermal budgets, triages torpedoes by whether they
can still be killed before impact, and strictly prioritises torpedoes over
kinetic slugs. Balance target: a lone hull generally LOSES to one 12 g
Trident; the counter is massed mutual PD support, not single-ship PD.

These tests assert INVARIANTS (monotonicity, ordering, conservation-style
budget effects), not exact tuned values.
"""

import json
import math

import pytest

from src.physics import Vector3D
from src.pointdefense import (
    PDLaser,
    PD_ABSORPTIVITY,
    PD_WALLPLUG_EFFICIENCY,
    TORPEDO_CROSS_SECTION_M2,
    TORPEDO_ELECTRONICS_THRESHOLD_J,
    delivered_power_w,
    dwell_energy_j,
    energy_before_impact_j,
    full_coupling_range_m,
)
from src.projectile import KineticProjectile
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    PDLaserState,
    ProjectileInFlight,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import Torpedo, TorpedoSpecs, GuidanceMode


@pytest.fixture(scope="module")
def fleet_data():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


TRIDENT_SPECS = TorpedoSpecs.from_fleet_data(
    warhead_yield_gj=0.0, penetrator_mass_kg=250.0, ammo_mass_kg=3600.0,
    exhaust_velocity_kps=8.0,
)


def _ballistic_trident(range_m: float, closure_ms: float, target_id: str) -> Torpedo:
    """A Trident coasting dead-on at constant closure (controlled experiments)."""
    torp = Torpedo(
        specs=TRIDENT_SPECS,
        position=Vector3D(range_m, 0, 0),
        velocity=Vector3D(-closure_ms, 0, 0),
        target_id=target_id,
    )
    torp.armed = True
    torp.fuel_exhausted = True
    torp.guidance_mode = GuidanceMode.COAST
    return torp


def _pd_ship(sim, ship_id, faction, position, n_turrets, fleet_data,
             ship_type="destroyer"):
    """Fleet ship with exactly n fleet-spec PD turrets (5 MW, 250 km, 5 s)."""
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, fleet_data,
        position=position, forward=Vector3D(1, 0, 0),
    )
    ship.point_defense = []
    for i in range(n_turrets):
        laser = PDLaser(power_mw=5.0, aperture_m=0.5, wavelength_nm=1000.0,
                        range_km=250.0, cooldown_s=5.0, name=f"xpd_{i}")
        ship.point_defense.append(PDLaserState(laser=laser, turret_name=f"xpd_{i}"))
        if ship.power_system:
            ship.power_system.add_weapon_capacitor(f"xpd_{i}", {
                "type": "point_defense",
                "power_draw_mw": 5.0 / PD_WALLPLUG_EFFICIENCY,
                "cooldown_s": 5.0,
            })
    sim.add_ship(ship)
    return ship


def _run_ballistic_intercept(fleet_data, n_turrets, closure_kps,
                             start_km=260.0, dt=0.5):
    """Returns (disabled_before_impact, heat_absorbed_j)."""
    sim = CombatSimulation(seed=7, time_step=dt)
    defender = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), n_turrets, fleet_data)
    enemy = create_ship_from_fleet_data(
        "e", "corvette", "beta", fleet_data,
        position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
    sim.add_ship(enemy)
    torp = _ballistic_trident(start_km * 1000, closure_kps * 1000, "d")
    tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="e")
    sim.torpedoes.append(tf)
    for _ in range(int(300 / dt)):
        sim.step()
        if tf not in sim.torpedoes:
            break
    return tf.is_disabled, tf.heat_absorbed_j


# =============================================================================
# 1. Beam model: lethality must increase monotonically as range closes
# =============================================================================

class TestDwellModel:

    def test_delivered_power_increases_monotonically_as_range_closes(self):
        ranges = [250, 200, 150, 120, 100, 50, 10, 1]
        powers = [delivered_power_w(5e6, r) for r in ranges]
        for closer, farther in zip(powers[1:], powers[:-1]):
            assert closer >= farther, (
                "PD lethality must not decrease as the target closes"
            )
        # Outside the full-coupling range the gain must be strict (optics).
        assert delivered_power_w(5e6, 150) > delivered_power_w(5e6, 250)
        # Inside it, the whole beam is already on target.
        assert delivered_power_w(5e6, 50) == pytest.approx(5e6 * PD_ABSORPTIVITY)

    def test_reproduces_measured_baseline_energies(self):
        """Delivered energy per 5 s of dwell matches the verified baseline
        (7.50 MJ at <=100 km ... ~1.5 MJ at 250 km)."""
        assert dwell_energy_j(5e6, 100, 5.0) == pytest.approx(7.5e6, rel=1e-6)
        assert dwell_energy_j(5e6, 150, 5.0) == pytest.approx(4.24e6, rel=0.05)
        assert dwell_energy_j(5e6, 200, 5.0) == pytest.approx(2.39e6, rel=0.05)
        assert dwell_energy_j(5e6, 250, 5.0) == pytest.approx(1.53e6, rel=0.05)

    def test_full_coupling_range_is_physical_not_tuned(self):
        """r_full = sqrt(A/pi)/theta_eff with theta_eff = hypot(lambda/2D, jitter)."""
        theta = math.hypot(1e-6 / (2 * 0.5), 5e-6)
        expected = math.sqrt(TORPEDO_CROSS_SECTION_M2 / math.pi) / theta
        assert full_coupling_range_m() == pytest.approx(expected, rel=1e-9)
        assert 100_000 < full_coupling_range_m() < 120_000  # ~110 km

    def test_closed_form_matches_numeric_integration(self):
        """The savability integral is exact, not a fit."""
        v = 12_000.0
        closed = energy_before_impact_j(5e6, 250.0, v, max_range_km=250.0)
        dt = 0.005
        r, numeric = 250_000.0, 0.0
        while r > 0:
            numeric += delivered_power_w(5e6, r / 1000.0) * dt
            r -= v * dt
        assert closed == pytest.approx(numeric, rel=0.01)

    def test_energy_before_impact_decreases_with_closure(self):
        energies = [energy_before_impact_j(5e6, 250.0, v * 1000, max_range_km=250.0)
                    for v in (4, 8, 12, 20, 26)]
        for faster, slower in zip(energies[1:], energies[:-1]):
            assert faster < slower

    def test_continuous_delivery_is_dt_invariant(self, fleet_data):
        """Average delivered power must not depend on the integrator step."""
        def heat_with_dt(dt):
            sim = CombatSimulation(seed=9, time_step=dt)
            d = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), 1, fleet_data)
            e = create_ship_from_fleet_data(
                "e", "corvette", "beta", fleet_data,
                position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
            sim.add_ship(e)
            torp = _ballistic_trident(60_000, 500, "d")
            tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="e")
            sim.torpedoes.append(tf)
            for _ in range(int(30 / dt)):
                sim.step()
            return tf.heat_absorbed_j

        h_fine, h_coarse = heat_with_dt(0.5), heat_with_dt(1.0)
        assert h_fine > 0
        assert h_coarse == pytest.approx(h_fine, rel=0.10)


# =============================================================================
# 2. Power and heat coupling: defending must cost something
# =============================================================================

class TestPowerHeatCoupling:

    def test_pd_capacitor_sized_for_electrical_energy(self, fleet_data):
        """Capacitor stores beam/eta per burst, so its heat equals the true
        waste heat and its charge rate is the turret's real bus draw."""
        ship = create_ship_from_fleet_data("s", "destroyer", "alpha", fleet_data)
        for pd in ship.point_defense:
            cap = ship.power_system.weapon_capacitors[pd.turret_name]
            beam_burst_mj = pd.laser.power_mw * pd.laser.cooldown_s
            assert cap.capacity_mj == pytest.approx(
                beam_burst_mj / PD_WALLPLUG_EFFICIENCY)
            assert cap.charge_rate_mw == pytest.approx(
                pd.laser.power_mw / PD_WALLPLUG_EFFICIENCY)

    def test_each_turret_has_its_own_capacitor(self, fleet_data):
        """Regression: both destroyer turrets were named 'PD Laser Turret' and
        silently shared one capacitor, so only one of them ever fired."""
        ship = create_ship_from_fleet_data("s", "destroyer", "alpha", fleet_data)
        names = [pd.turret_name for pd in ship.point_defense]
        assert len(names) == len(set(names)), f"duplicate turret names: {names}"
        for name in names:
            assert name in ship.power_system.weapon_capacitors

    def test_sustained_fire_raises_heat_and_draws_bus_power(self, fleet_data):
        """120 s of 2-turret dwell must show up in the thermal ledger (waste
        heat = 3x beam power at 25% wall-plug efficiency) and in the PD
        electrical-draw ledger that competes with the drive."""
        def run(firing):
            sim = CombatSimulation(seed=3, time_step=1.0)
            d = create_ship_from_fleet_data(
                "d", "destroyer", "alpha", fleet_data,
                position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
            e = create_ship_from_fleet_data(
                "e", "corvette", "beta", fleet_data,
                position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
            sim.add_ship(d)
            sim.add_ship(e)
            n = 0
            max_draw = 0.0
            for _ in range(120):
                if firing and not any(not tf.is_disabled for tf in sim.torpedoes):
                    torp = _ballistic_trident(240_000, 10, "d")
                    sim.torpedoes.append(TorpedoInFlight(
                        torpedo_id=f"t{n}", torpedo=torp, source_ship_id="e"))
                    n += 1
                sim.step()
                max_draw = max(max_draw, d.pd_power_draw_gw)
            return d.thermal_system.heatsink.current_heat_gj, max_draw

        idle_heat, idle_draw = run(False)
        fire_heat, fire_draw = run(True)

        # Heat: 2 turrets x 15 MW waste heat -> +3.6 GJ over 120 s.
        assert fire_heat > idle_heat + 1.0, (
            f"sustained PD fire added only {fire_heat - idle_heat:.2f} GJ "
            f"of heat - the thermal coupling is not working"
        )
        # Power: 2 turrets x 20 MW electrical = 0.04 GW while firing.
        assert idle_draw == 0.0
        assert fire_draw == pytest.approx(0.04, rel=0.05)

    def test_full_throttle_pd_fire_drains_the_battery(self, fleet_data):
        """At throttle 1.0 the drive claims the whole reactor, so capacitor
        recharge must come from the battery: same budget as the drive."""
        ship = create_ship_from_fleet_data("s", "destroyer", "alpha", fleet_data)
        ps = ship.power_system
        ps.set_drive_throttle(1.0)
        start_gj = ps.battery.current_charge_gj
        for _ in range(60):
            for pd in ship.point_defense:
                if ps.can_weapon_fire(pd.turret_name) and pd.can_fire():
                    pd.engage()
                    ps.fire_weapon(pd.turret_name)
            for pd in ship.point_defense:
                pd.update(1.0)
            ps.update(1.0)
        drained = start_gj - ps.battery.current_charge_gj
        # 2 turrets x 20 MW x 60 s = 2.4 GJ
        assert drained == pytest.approx(2.4, rel=0.10), (
            f"expected ~2.4 GJ battery drain from 60 s of 2-turret fire at "
            f"full throttle, measured {drained:.2f} GJ"
        )

    def test_burst_start_is_gated_on_capacitor_charge(self, fleet_data):
        """can_weapon_fire is the gate: an empty capacitor blocks the burst."""
        sim = CombatSimulation(seed=4, time_step=1.0)
        d = create_ship_from_fleet_data(
            "d", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(d)
        sim.add_ship(e)
        # Drain every PD capacitor and forbid recharge (dead battery + full
        # throttle drive claim).
        for pd in d.point_defense:
            d.power_system.weapon_capacitors[pd.turret_name].current_charge_mj = 0.0
        d.power_system.battery.current_charge_gj = 0.0
        d.power_system.set_drive_throttle(1.0)
        d.current_maneuver = Maneuver(
            maneuver_type=ManeuverType.BURN, start_time=0.0,
            throttle=1.0, direction=Vector3D(1, 0, 0))

        torp = _ballistic_trident(100_000, 1000, "d")
        tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="e")
        sim.torpedoes.append(tf)
        sim.step()
        assert tf.heat_absorbed_j == 0.0, (
            "PD fired with an empty capacitor - the power gate is bypassed"
        )


# =============================================================================
# 3. Target prioritisation: torpedoes outrank slugs; savable outrank unsavable
# =============================================================================

class TestPrioritisation:

    def _dual_threat_sim(self, fleet_data):
        sim = CombatSimulation(seed=5, time_step=1.0)
        d = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), 2, fleet_data)
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(e)
        # Slug: closer and faster (more "urgent" naively)
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-8000, 0, 0),
                                 position=Vector3D(60_000, 0, 0))
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="slug", projectile=proj,
            source_ship_id="e", target_ship_id="d"))
        # Torpedo: farther and slower
        torp = _ballistic_trident(150_000, 2000, "d")
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="torp", torpedo=torp, source_ship_id="e"))
        return sim

    def test_torpedoes_outrank_slugs(self, fleet_data):
        sim = self._dual_threat_sim(fleet_data)
        for _ in range(5):
            sim.step()
        pd_events = [ev for ev in sim.events
                     if ev.event_type == SimulationEventType.PD_ENGAGED]
        assert pd_events, "PD never engaged anything"
        assert all(ev.data["target_type"] == "torpedo" for ev in pd_events), (
            "PD burned dwell on a slug while a torpedo was in the envelope"
        )

    def test_slugs_engaged_when_no_torpedo_in_envelope(self, fleet_data):
        sim = CombatSimulation(seed=5, time_step=1.0)
        d = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), 2, fleet_data)
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(e)
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-8000, 0, 0),
                                 position=Vector3D(60_000, 0, 0))
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="slug", projectile=proj,
            source_ship_id="e", target_ship_id="d"))
        for _ in range(3):
            sim.step()
        slug_events = [ev for ev in sim.events
                       if ev.event_type == SimulationEventType.PD_ENGAGED
                       and ev.data["target_type"] == "slug"]
        assert slug_events, "with no torpedoes around, PD should ablate slugs"

    def test_savable_torpedo_preferred_over_unsavable(self, fleet_data):
        """Two inbound torpedoes: one killable before impact, one not.
        Both turrets must dwell on the killable one."""
        sim = CombatSimulation(seed=6, time_step=1.0)
        d = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), 2, fleet_data)
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(e)
        # Savable: 10 km/s from 240 km -> ~2 turrets' worth of dwell available
        savable = _ballistic_trident(240_000, 10_000, "d")
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="savable", torpedo=savable, source_ship_id="e"))
        # Unsavable: 26 km/s from 240 km -> needs ~6 turrets, we have 2
        hopeless = _ballistic_trident(240_000, 26_000, "d")
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="hopeless", torpedo=hopeless, source_ship_id="e"))
        sim.step()
        engaged = [ev.data["target_id"] for ev in sim.events
                   if ev.event_type == SimulationEventType.PD_ENGAGED]
        assert engaged, "PD never engaged"
        assert set(engaged) == {"savable"}, (
            f"dwell went to {set(engaged)} - turrets must not burn dwell on an "
            f"unkillable torpedo while a savable one is inbound"
        )


# =============================================================================
# 4. Balance: a lone hull loses; numbers win
# =============================================================================

class TestBalance:

    def test_single_turret_cannot_stop_high_closure_trident(self, fleet_data):
        disabled, heat = _run_ballistic_intercept(fleet_data, 1, 20.0)
        assert not disabled, "1 turret must not stop a 20 km/s Trident"
        assert heat < TORPEDO_ELECTRONICS_THRESHOLD_J

    def test_two_turrets_cannot_stop_12kps_trident(self, fleet_data):
        """The workhorse case: a lone destroyer (2 turrets) loses to one
        12 g Trident at its 12 km/s cruise floor."""
        disabled, heat = _run_ballistic_intercept(fleet_data, 2, 12.0)
        assert not disabled
        assert heat < TORPEDO_ELECTRONICS_THRESHOLD_J

    def test_low_closure_is_killable_by_one_turret(self, fleet_data):
        disabled, _ = _run_ballistic_intercept(fleet_data, 1, 4.0)
        assert disabled, "a slow 4 km/s approach must die to even one turret"

    def test_four_turrets_stop_12kps_but_not_26kps(self, fleet_data):
        disabled_12, _ = _run_ballistic_intercept(fleet_data, 4, 12.0)
        disabled_26, _ = _run_ballistic_intercept(fleet_data, 4, 26.0)
        assert disabled_12, "a dreadnought's 4 turrets should stop 12 km/s"
        assert not disabled_26, (
            "26 km/s must defeat even 4 turrets - torpedoes have to keep "
            "a closure regime where they reliably win"
        )

    def test_kill_probability_never_decreases_with_turret_count(self, fleet_data):
        for closure in (8.0, 20.0):
            outcomes = [
                _run_ballistic_intercept(fleet_data, n, closure)[0]
                for n in (1, 2, 4)
            ]
            for fewer, more in zip(outcomes[:-1], outcomes[1:]):
                assert (not fewer) or more, (
                    f"adding turrets made the intercept WORSE at {closure} km/s"
                )

    def test_lone_destroyer_loses_to_guided_trident(self, fleet_data):
        """End-to-end: evading fleet destroyer with its own 2 turrets vs one
        launched, APN-guided 12 g Trident. The torpedo should win."""
        sim = CombatSimulation(seed=11, time_step=0.5)
        d = create_ship_from_fleet_data(
            "d", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        a = create_ship_from_fleet_data(
            "a", "cruiser_torpedo", "beta", fleet_data,
            position=Vector3D(400_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(d)
        sim.add_ship(a)
        d.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE,
                                      start_time=0.0, throttle=1.0)
        assert sim.inject_command("a", {"type": "launch_torpedo", "target_id": "d"})
        steps = 0
        while sim.torpedoes and steps < 2000:
            sim.step()
            steps += 1
        assert sim.metrics.total_torpedo_hits == 1, (
            "a single ship out-defended a 12 g torpedo - PD is overtuned"
        )


# =============================================================================
# 5. Massed defense: leakage decreases with escort count
# =============================================================================

class TestWallOfShips:

    def _salvo_defense(self, fleet_data, n_escorts, salvo=4):
        sim = CombatSimulation(seed=23, time_step=0.5)
        victim = create_ship_from_fleet_data(
            "victim", "battleship", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        sim.add_ship(victim)
        victim.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE,
                                           start_time=0.0, throttle=1.0)
        for i in range(n_escorts):
            ang = 2 * math.pi * i / max(n_escorts, 1)
            esc = create_ship_from_fleet_data(
                f"esc{i}", "destroyer", "alpha", fleet_data,
                position=Vector3D(30_000 * math.cos(ang),
                                  30_000 * math.sin(ang), 0),
                forward=Vector3D(1, 0, 0))
            sim.add_ship(esc)
        att = create_ship_from_fleet_data(
            "att", "cruiser_torpedo", "beta", fleet_data,
            position=Vector3D(350_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(att)
        launched, steps = 0, 0
        while steps < 900:
            if launched < salvo and sim.inject_command(
                    "att", {"type": "launch_torpedo", "target_id": "victim"}):
                launched += 1
            sim.step()
            steps += 1
            if launched >= salvo and not sim.torpedoes:
                break
        return sim.metrics.total_torpedo_hits

    def test_leakage_decreases_with_escort_count(self, fleet_data):
        hits_alone = self._salvo_defense(fleet_data, 0)
        hits_screen = self._salvo_defense(fleet_data, 4)
        hits_wall = self._salvo_defense(fleet_data, 8)
        assert hits_alone > 0, (
            "torpedoes must still get through against a lone capital ship"
        )
        assert hits_screen < hits_alone, "4 escorts must reduce leakage"
        assert hits_wall <= hits_screen, "more escorts must never leak more"

    def test_multiple_hulls_concentrate_on_one_torpedo(self, fleet_data):
        """Overlapping envelopes: 3 hulls' turrets must deposit ~3x one
        hull's heat into the same torpedo."""
        def heat_rate(n_ships):
            sim = CombatSimulation(seed=8, time_step=1.0)
            for i in range(n_ships):
                _pd_ship(sim, f"s{i}", "alpha",
                         Vector3D(0, 20_000 * i, 0), 2, fleet_data)
            e = create_ship_from_fleet_data(
                "e", "corvette", "beta", fleet_data,
                position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
            sim.add_ship(e)
            torp = _ballistic_trident(80_000, 100, "s0")
            tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="e")
            sim.torpedoes.append(tf)
            # 5 s window: 3 hulls deliver ~45 MJ, still below the 50 MJ
            # disable threshold, so the rate comparison is not clipped.
            for _ in range(5):
                sim.step()
            return tf.heat_absorbed_j

        single, tripled = heat_rate(1), heat_rate(3)
        assert single > 0
        assert tripled == pytest.approx(3.0 * single, rel=0.25), (
            f"3 hulls delivered {tripled / single:.2f}x one hull's heat - "
            f"mutual PD support is not concentrating"
        )


# =============================================================================
# 6. Ablation recoil on slugs
# =============================================================================

class TestAblationRecoil:

    def _slug_run(self, fleet_data, escort_angle_deg):
        """40 kg slug at 6 km/s, dead-on at a PD-less victim. Optionally one
        escort destroyer 25 km away at the given angle off the threat axis."""
        sim = CombatSimulation(seed=5, time_step=0.5)
        victim = create_ship_from_fleet_data(
            "v", "corvette", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        victim.point_defense = []
        sim.add_ship(victim)
        if escort_angle_deg is not None:
            a = math.radians(escort_angle_deg)
            esc = create_ship_from_fleet_data(
                "esc", "destroyer", "alpha", fleet_data,
                position=Vector3D(25_000 * math.cos(a), 25_000 * math.sin(a), 0),
                forward=Vector3D(1, 0, 0))
            sim.add_ship(esc)
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(e)
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-6000, 0, 0),
                                 position=Vector3D(250_000, 0, 0))
        pf = ProjectileInFlight(projectile_id="s", projectile=proj,
                                source_ship_id="e", target_ship_id="v")
        sim.projectiles.append(pf)
        for _ in range(200):
            sim.step()
            if pf not in sim.projectiles:
                break
        return proj, pf

    def test_own_hull_dwell_decelerates_and_lightens_the_slug(self, fleet_data):
        """Head-on geometry: recoil is nearly pure deceleration, and ablation
        removes mass - impact KE must drop on both counts."""
        sim = CombatSimulation(seed=5, time_step=0.5)
        d = create_ship_from_fleet_data(
            "d", "destroyer", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(d)
        sim.add_ship(e)
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-6000, 0, 0),
                                 position=Vector3D(250_000, 0, 0))
        pf = ProjectileInFlight(projectile_id="s", projectile=proj,
                                source_ship_id="e", target_ship_id="d")
        sim.projectiles.append(pf)
        for _ in range(200):
            sim.step()
            if pf not in sim.projectiles:
                break
        assert getattr(proj, "_pd_ablation", 0.0) > 0.0, "no mass ablated"
        assert proj.velocity.magnitude < 6000.0, (
            "head-on ablation recoil must decelerate the slug"
        )
        # Guardrail: gun combat must remain viable - a lone hull's PD shaves
        # a few percent off a slug, it does not delete it.
        assert proj.velocity.magnitude > 5700.0
        assert getattr(proj, "_pd_ablation", 0.0) < 4.0

    def test_escort_dwell_deflects_the_slug_laterally(self, fleet_data):
        """Oblique geometry: an escort's beam pushes the round off the line
        to the consort - CPA to the victim must open up."""
        _, pf_bare = self._slug_run(fleet_data, None)
        _, pf_escort = self._slug_run(fleet_data, 45)
        assert pf_bare.min_distance_to_target < 10.0  # dead-on without help
        assert pf_escort.min_distance_to_target > 50.0, (
            f"escort deflection moved CPA only "
            f"{pf_escort.min_distance_to_target:.1f} m"
        )

    def test_recoil_pushes_slug_away_from_the_shooting_turret(self, fleet_data):
        """Momentum direction check: the plume leaves toward the turret, the
        slug gains velocity along turret->slug."""
        sim = CombatSimulation(seed=5, time_step=0.5)
        d = _pd_ship(sim, "d", "alpha", Vector3D(0, 0, 0), 2, fleet_data)
        e = create_ship_from_fleet_data(
            "e", "corvette", "beta", fleet_data,
            position=Vector3D(900_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.add_ship(e)
        proj = KineticProjectile(mass_kg=40.0, velocity=Vector3D(-6000, 0, 0),
                                 position=Vector3D(15_000, 0, 0))
        pf = ProjectileInFlight(projectile_id="s", projectile=proj,
                                source_ship_id="e", target_ship_id="d")
        sim.projectiles.append(pf)
        v_before = Vector3D(proj.velocity.x, proj.velocity.y, proj.velocity.z)
        pos_before = Vector3D(proj.position.x, proj.position.y, proj.position.z)
        sim.step()
        dv = proj.velocity - v_before
        beam_axis = (pos_before - d.position).normalized()
        assert dv.magnitude > 0.0, "no recoil applied"
        assert dv.dot(beam_axis) > 0.0, (
            "recoil must push the slug AWAY from the shooter along the beam"
        )
