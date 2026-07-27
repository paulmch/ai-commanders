"""
Drift tests: every quantitative claim in the LLM-facing doctrine must be tied
to the code (constant, closed form, or a small seeded run of the real engine).

The doctrine went stale once already because nothing bound prompt text to
behaviour: captains were told torpedo delta-v adds to closure (the engine
enforces a 12 km/s closure floor and a terminal burn instead), quoted PD
survival fractions that were never measured, and were never told that a
PD "kill" only blinds the torpedo. Each claim below names the prompt text it
guards. If a test here fails, fix the DOCTRINE or the model - never the test
alone.
"""

import json
import math

import pytest

from src.physics import Vector3D
from src.pointdefense import (
    PDLaser,
    PD_ABSORPTIVITY,
    PD_WALLPLUG_EFFICIENCY,
    TORPEDO_ELECTRONICS_THRESHOLD_J,
    full_coupling_range_m,
)
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import MIN_CLOSING_SPEED_KPS, Torpedo, TorpedoSpecs, GuidanceMode
from src.llm.prompts import (
    ADMIRAL_DOCTRINE,
    CAPTAIN_DOCTRINE,
    build_ship_capabilities_from_fleet,
    format_torpedo_threats,
    pd_doctrine_numbers,
)


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corvette_capabilities(fleet):
    return build_ship_capabilities_from_fleet(
        ship_name="TIS Wasp", ship_type="corvette", fleet_data=fleet,
        hull_integrity=100, heat_percent=0, delta_v_remaining=500,
        nose_armor=212, lateral_armor=36, tail_armor=42,
        heatsink_capacity=525, radiators_extended=False,
    )


def _build(sim, ship_id, stype, faction, fleet, x_km, vx_kps, fwd):
    ship = create_ship_from_fleet_data(
        ship_id, stype, faction, fleet,
        position=Vector3D(x_km * 1000.0, 0, 0),
        velocity=Vector3D(vx_kps * 1000.0, 0, 0),
        forward=Vector3D(fwd, 0, 0))
    sim.add_ship(ship)
    return ship


def _evade(ship):
    ship.current_maneuver = Maneuver(
        maneuver_type=ManeuverType.EVASIVE, start_time=0.0,
        duration=1e9, throttle=1.0)


def _ballistic_trident(range_m, closure_mps, target_id, disabled=False):
    """A Trident already up to speed, coasting dead-on at the target."""
    specs = TorpedoSpecs.from_fleet_data(
        warhead_yield_gj=0.0, penetrator_mass_kg=250.0, ammo_mass_kg=3600.0,
        exhaust_velocity_kps=8.0)
    torp = Torpedo(specs=specs, position=Vector3D(range_m, 0, 0),
                   velocity=Vector3D(-closure_mps, 0, 0), target_id=target_id)
    torp.armed = True
    torp.fuel_exhausted = True
    torp.guidance_mode = GuidanceMode.COAST
    return TorpedoInFlight(torpedo_id=f"t{range_m}", torpedo=torp,
                           source_ship_id="attacker", is_disabled=disabled)


def _run_torpedo_engagement(fleet, target_type, range_km, closure_kps,
                            evading, n_torps, seed, spacing_s=0.0,
                            max_t=400.0):
    """Real launch -> guided flight -> PD -> impact, on the shipped engine."""
    sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=seed)
    _build(sim, 'sh', 'cruiser_torpedo', 'beta', fleet,
           range_km, -closure_kps / 2, -1)
    target = _build(sim, 'tg', target_type, 'alpha', fleet,
                    0.0, closure_kps / 2, 1)
    if evading:
        _evade(target)
    launched, next_t, impacts = 0, 0.0, []
    for _ in range(int(max_t / 0.5)):
        while launched < n_torps and sim.current_time >= next_t:
            if sim.inject_command('sh', {'type': 'launch_torpedo',
                                         'target_id': 'tg'}):
                launched += 1
                next_t = sim.current_time + spacing_s
            else:
                break
        for ev in sim.step():
            if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                impacts.append(ev.data)
        if launched >= n_torps and not sim.torpedoes:
            break
        if target.is_destroyed:
            break
    return impacts, target


# =============================================================================
# Point defense: closed-form numbers quoted to both roles
# =============================================================================

class TestPDDoctrineNumbers:
    """Guards the PD block in format_weapon_groups_for_prompt and the
    '=== TORPEDO DEFENSE ===' / 'TORPEDO WARFARE' doctrine sections."""

    def test_pd_numbers_come_from_fleet_data(self, fleet):
        pd = pd_doctrine_numbers(fleet)
        spec = fleet["weapon_types"]["pd_laser"]
        assert pd["range_km"] == spec["range_km"]
        assert pd["power_mw"] == spec["power_draw_mw"]

    def test_full_coupling_range_is_about_110_km(self, fleet):
        """Doctrine: 'the whole beam lands only inside ~110 km'."""
        pd = pd_doctrine_numbers(fleet)
        assert 100.0 <= pd["full_coupling_km"] <= 120.0, (
            f"full-coupling range moved to {pd['full_coupling_km']:.0f} km - "
            "update the PD optics line in the capabilities block"
        )

    def test_one_turret_blinds_about_5_kps_of_closure(self, fleet):
        """Doctrine (captain + admiral): 'each PD turret ~5 km/s of closure'.

        Derived from the same closed form the engine's savability triage uses
        (PDLaser.energy_before_impact_j vs the 50 MJ seeker threshold).
        """
        pd = pd_doctrine_numbers(fleet)
        per = pd["blind_closure_kps_per_turret"]
        assert 4.5 <= per <= 6.0, (
            f"one turret now blinds {per:.2f} km/s of closure - the '~5 km/s "
            "per turret' rule in CAPTAIN_DOCTRINE and ADMIRAL_DOCTRINE is stale"
        )
        assert "~5 km/s" in CAPTAIN_DOCTRINE
        assert "~5 km/s of closure; N turrets ~5N km/s" in ADMIRAL_DOCTRINE

    def test_closed_form_matches_engine_kill_ranges(self, fleet):
        """The ~5 km/s rule must hold in the real engine, not just on paper:
        one turret kills a 4 km/s ballistic Trident and leaks a 12 km/s one."""
        results = {}
        for closure_kps in (4.0, 12.0):
            sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=7)
            target = _build(sim, 'tg', 'corvette', 'alpha', fleet, 0, 0, 1)  # 1 turret
            _build(sim, 'en', 'corvette', 'beta', fleet, 900, 0, -1)
            tf = _ballistic_trident(255_000, closure_kps * 1000, 'tg')
            sim.torpedoes.append(tf)
            for _ in range(300):
                sim.step()
                if tf not in sim.torpedoes:
                    break
            results[closure_kps] = tf.is_disabled
        assert results[4.0] is True, "1 turret no longer kills a 4 km/s closure"
        assert results[12.0] is False, (
            "1 turret now kills a 12 km/s closure - PD is far stronger than "
            "the doctrine says; re-measure and rewrite the tables"
        )

    def test_seeker_kill_threshold_is_50_mj(self):
        """Doctrine: 'seeker kill at 50 MJ absorbed beam'."""
        assert TORPEDO_ELECTRONICS_THRESHOLD_J == 50e6
        assert "50 MJ" in CAPTAIN_DOCTRINE

    def test_waste_heat_is_15_mw_per_firing_turret(self, fleet):
        """Doctrine (captain block + admiral): '~15 MW hull heat per turret'."""
        pd = pd_doctrine_numbers(fleet)
        assert abs(pd["waste_heat_mw"] - 15.0) < 1.0, (
            f"waste heat per turret is now {pd['waste_heat_mw']:.1f} MW"
        )
        laser = PDLaser.from_fleet_data(fleet["weapon_types"]["pd_laser"])
        assert laser.waste_heat_w() == laser.power_w * (
            1 - PD_WALLPLUG_EFFICIENCY) / PD_WALLPLUG_EFFICIENCY
        assert "~15 MW hull heat per firing turret" in ADMIRAL_DOCTRINE

    def test_guided_arrival_band_13_to_28_kps(self, fleet):
        """Doctrine: 'guided torpedoes arrive at 13-28 km/s'. Anchored by the
        band's two ends: a zero-closure launch and a 26 km/s head-on."""
        impacts_slow, _ = _run_torpedo_engagement(
            fleet, 'destroyer', 400, 0.0, True, 1, seed=101)
        impacts_fast, _ = _run_torpedo_engagement(
            fleet, 'destroyer', 400, 26.0, True, 1, seed=102, max_t=120.0)
        assert impacts_slow and impacts_fast, "reference launches must connect"
        v_slow = impacts_slow[0]['impact_speed_kps']
        v_fast = impacts_fast[0]['impact_speed_kps']
        assert 12.0 <= v_slow <= 16.0, f"slow-end anchor moved: {v_slow:.1f} km/s"
        assert 24.0 <= v_fast <= 32.0, f"fast-end anchor moved: {v_fast:.1f} km/s"
        assert "13-28 km/s" in CAPTAIN_DOCTRINE
        assert "13-28 km/s" in ADMIRAL_DOCTRINE


# =============================================================================
# Torpedo capability block (captain) and Trident doctrine (admiral)
# =============================================================================

class TestTorpedoDoctrineNumbers:
    def test_trident_stats_quoted_from_fleet_data(self, fleet, corvette_capabilities):
        wt = fleet["weapon_types"]["torpedo_launcher"]
        text = corvette_capabilities
        assert f"{wt['penetrator_mass_kg']:.0f} kg kinetic penetrator" in text
        assert f"{wt['delta_v_kps']} km/s own delta-v" in text
        assert f"Reload {wt['cooldown_s']}s" in text
        assert f"{wt['magazine']} rounds per launcher" in text

    def test_cruise_floor_quoted_matches_guidance_constant(self, corvette_capabilities):
        """Block: 'Guidance never lets closure drop below ~12 km/s'."""
        assert MIN_CLOSING_SPEED_KPS == 12.0
        assert "below ~12 km/s" in corvette_capabilities

    def test_impact_energy_anchors_match_ke_formula(self, fleet, corvette_capabilities):
        """The GJ figures are computed from 0.5*m*v^2 at the measured speeds,
        so they can only drift if the anchors or the penetrator mass move."""
        pen = fleet["weapon_types"]["torpedo_launcher"]["penetrator_mass_kg"]

        def ke(v_kps):
            return 0.5 * pen * (v_kps * 1000) ** 2 / 1e9

        assert f"~{ke(13.5):.0f} GJ" in corvette_capabilities   # zero-closure launch
        assert f"~{ke(28):.0f} GJ" in corvette_capabilities     # 26 km/s head-on
        assert 22.0 <= ke(13.5) <= 24.0
        assert 96.0 <= ke(28) <= 100.0

    def test_zero_closure_launch_lands_the_quoted_energy(self, fleet):
        """Block: 'launched with zero closure -> ~23 GJ' - real engine."""
        impacts, _ = _run_torpedo_engagement(
            fleet, 'destroyer', 400, 0.0, True, 1, seed=110)
        assert len(impacts) == 1
        dmg = impacts[0]['total_damage_gj']
        assert 18.0 <= dmg <= 30.0, (
            f"a zero-closure launch now lands {dmg:.1f} GJ - the ~23 GJ anchor "
            "in the torpedo block and admiral doctrine is stale"
        )

    def test_receding_evader_escapes(self, fleet):
        """Block: 'a receding ship that keeps running usually outlasts the
        round's delta-v'. Admiral: 'NEVER launch at a receding evading ship'."""
        impacts, target = _run_torpedo_engagement(
            fleet, 'destroyer', 400, -6.0, True, 1, seed=111, max_t=350.0)
        assert impacts == [], "receding evaders no longer escape - retune doctrine"
        assert not target.is_destroyed

    def test_closing_launch_connects_against_lone_evader(self, fleet):
        """Block: 'a CLOSING launch vs a lone ship almost always connects'."""
        impacts, _ = _run_torpedo_engagement(
            fleet, 'destroyer', 400, 12.0, True, 1, seed=112, max_t=120.0)
        assert len(impacts) == 1, (
            "a closing launch no longer connects vs a lone evading destroyer - "
            "PD or evasion got stronger; the lethality doctrine is stale"
        )

    def test_simultaneous_salvo_beats_sequential(self, fleet):
        """Block + torpedo_salvo tool: simultaneous rounds saturate PD, the
        same rounds spaced 30s are blinded one at a time and dodged."""
        sim_impacts, _ = _run_torpedo_engagement(
            fleet, 'dreadnought', 400, 12.0, True, 4, seed=700, spacing_s=0.0)
        seq_impacts, _ = _run_torpedo_engagement(
            fleet, 'dreadnought', 400, 12.0, True, 4, seed=700, spacing_s=30.0)
        assert len(sim_impacts) >= 3, (
            f"simultaneous 4-round salvo only landed {len(sim_impacts)}/4"
        )
        assert len(sim_impacts) > len(seq_impacts), (
            "sequential launches now match simultaneous ones - the salvo-timing "
            "doctrine (24/24 vs 12/24) is stale"
        )


# =============================================================================
# Blinding vs stopping - the core PD doctrine claim
# =============================================================================

class TestBlindingIsNotStopping:
    def test_blinded_torpedo_still_hits_nonmaneuvering_ship(self, fleet):
        """Doctrine: 'a blinded round coasts ballistically and STILL HITS a
        ship flying straight'."""
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=5)
        target = _build(sim, 'tg', 'destroyer', 'alpha', fleet, 0, 0, 1)
        target.point_defense = []  # nothing left to shave it
        _build(sim, 'en', 'corvette', 'beta', fleet, 900, 0, -1)
        sim.torpedoes.append(
            _ballistic_trident(30_000, 12_000, 'tg', disabled=True))
        hit = False
        for _ in range(30):
            for ev in sim.step():
                if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                    hit = True
        assert hit, (
            "a blinded dead-on torpedo no longer hits a non-maneuvering ship - "
            "the 'PD alone stops nothing' doctrine needs re-measuring"
        )

    def test_evasion_dodges_a_blinded_torpedo(self, fleet):
        """Doctrine: 'your EVADE makes blinded rounds miss'. Same wreck, but
        blinded far out and the target maneuvering."""
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=5)
        target = _build(sim, 'tg', 'destroyer', 'alpha', fleet, 0, 0, 1)
        target.point_defense = []
        _evade(target)
        _build(sim, 'en', 'corvette', 'beta', fleet, 900, 0, -1)
        sim.torpedoes.append(
            _ballistic_trident(250_000, 12_000, 'tg', disabled=True))
        hit = False
        for _ in range(120):
            for ev in sim.step():
                if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                    hit = True
        assert not hit, (
            "an evading ship was hit by a wreck blinded 250 km out - either "
            "evasion regressed or the doctrine's dodge claim is wrong"
        )


# =============================================================================
# Displays: the data contract behind the awareness sections
# =============================================================================

class TestThreatDisplayContract:
    def test_enriched_fields_are_rendered(self):
        text = format_torpedo_threats([
            {"distance_km": 180.0, "closing_kps": 22.4, "eta_seconds": 8.0,
             "source": "HFS Sonnet5", "est_impact_gj": 63.0,
             "nez_inside": True, "pd_turrets_needed": 5, "own_pd_turrets": 2},
            {"distance_km": 90.0, "closing_kps": 12.0, "eta_seconds": 8.0,
             "source": "HFS Sonnet5", "blinded": True},
        ])
        assert "est ~63 GJ" in text
        assert "NEZ: CANNOT be outrun" in text
        assert "needs ~5 turret(s), you have 2" in text
        assert "CANNOT stop it" in text
        assert "BLINDED torpedo" in text
        assert "KEEP MANEUVERING" in text

    def test_live_sim_populates_enrichment(self, fleet):
        """captain._build_tactical_status must fill the enriched keys from the
        engine's own threat evaluation, not leave the renderer starved."""
        from unittest.mock import Mock
        from src.llm.captain import LLMCaptain, LLMCaptainConfig

        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=42)
        target = _build(sim, 'tg', 'destroyer', 'alpha', fleet, 0, 6.0, 1)
        _build(sim, 'sh', 'cruiser_torpedo', 'beta', fleet, 400, -6.0, -1)
        assert sim.inject_command('sh', {'type': 'launch_torpedo',
                                         'target_id': 'tg'})
        for _ in range(20):
            sim.step()

        captain = LLMCaptain(
            LLMCaptainConfig(name="C", ship_name="tg", ship_type="destroyer",
                             fleet_data=fleet),
            client=Mock(),
        )
        tactical = captain._build_tactical_status(target, None, sim)
        threats = tactical["torpedo_threats"]
        assert threats, "live torpedo did not reach the threat display"
        t = threats[0]
        for key in ("est_impact_gj", "nez_inside", "pd_turrets_needed",
                    "own_pd_turrets", "blinded"):
            assert key in t, f"threat enrichment lost key {key!r}"
        assert t["own_pd_turrets"] == 2  # destroyer mounts 2 PD turrets
        assert t["nez_inside"] is True  # closing Trident vs 2g hull

        status = captain._build_ship_status(target)
        assert status["pd_turrets_total"] == 2
        assert status["pd_turrets_operational"] == 2

    def test_admiral_sees_torpedoes_and_magazines(self, fleet):
        from unittest.mock import Mock
        from src.llm.admiral import AdmiralConfig, LLMAdmiral
        from src.llm.captain import LLMCaptain, LLMCaptainConfig

        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=42)
        _build(sim, 'tg', 'destroyer', 'alpha', fleet, 0, 6.0, 1)
        _build(sim, 'sh', 'cruiser_torpedo', 'beta', fleet, 400, -6.0, -1)
        assert sim.inject_command('sh', {'type': 'launch_torpedo',
                                         'target_id': 'tg'})
        for _ in range(10):
            sim.step()

        admiral = LLMAdmiral(AdmiralConfig(name="A", model="m"),
                             faction="beta", client=Mock(), fleet_data=fleet)
        captain = LLMCaptain(
            LLMCaptainConfig(name="B", ship_name="sh",
                             ship_type="cruiser_torpedo", fleet_data=fleet),
            client=Mock(),
        )
        captain.ship_id = 'sh'
        snap = admiral._build_snapshot(sim, [captain])

        friendly = snap.friendly_ships[0]
        # 4 launchers x 12 rounds (per-slot magazine override) minus 1 fired
        assert friendly.torpedoes_remaining == 47
        assert friendly.pd_turrets_total == 4
        torp_snaps = [p for p in snap.projectiles
                      if "orpedo" in p.weapon_type]
        assert torp_snaps, "in-flight torpedo invisible to the admiral"
        assert "guided" in torp_snaps[0].weapon_type


# =============================================================================
# Formation doctrine (admiral)
# =============================================================================

class TestFormationDoctrine:
    def test_escort_wall_reduces_leakage(self, fleet):
        """Admiral doctrine: escorts cut salvo leakage, and the effect needs
        mass (a big wall clearly beats none). Single seed of the measured
        17/18 alone vs 6/18 with 8 escorts result."""

        def wall(n_escorts):
            sim = CombatSimulation(seed=23, time_step=0.5, decision_interval=1e9)
            victim = _build(sim, 'victim', 'battleship', 'alpha', fleet, 0, 0, 1)
            _evade(victim)
            for i in range(n_escorts):
                ang = 2 * math.pi * i / max(n_escorts, 1)
                esc = create_ship_from_fleet_data(
                    f'esc{i}', 'destroyer', 'alpha', fleet,
                    position=Vector3D(30_000 * math.cos(ang),
                                      30_000 * math.sin(ang), 0),
                    forward=Vector3D(1, 0, 0))
                sim.add_ship(esc)
            _build(sim, 'att', 'cruiser_torpedo', 'beta', fleet, 450, 0, -1)
            launched, hits = 0, 0
            for _ in range(1600):
                if launched < 6:
                    if sim.inject_command('att', {'type': 'launch_torpedo',
                                                  'target_id': 'victim'}):
                        launched += 1
                        continue
                for ev in sim.step():
                    if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                        hits += 1
                if launched >= 6 and not sim.torpedoes:
                    break
            return hits

        alone, walled = wall(0), wall(8)
        assert alone >= 5, f"lone battleship now stops salvos ({alone}/6 hit)"
        assert walled < alone, (
            f"8-escort wall no longer reduces leakage ({walled}/6 vs {alone}/6) "
            "- the formation doctrine numbers are stale"
        )
