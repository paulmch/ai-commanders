"""
Tests for the reworked evasive maneuvering system.

Threat model under test (CombatSimulation._calculate_evasion):

* GUIDED torpedoes (live seeker) -> RUN (time-extending burn away along the
  threat axis), then PRESENT (rotate thickest armor onto the threat bearing)
  in the terminal phase.
* UNGUIDED threats (kinetic rounds and seeker-killed torpedoes) -> lateral
  EVADE, exactly as before; disabled torpedoes are now ballistic masses that
  can physically hit.

All assertions are INVARIANTS (direction relationships, armor arcs, ordering
of outcomes), not magic constants copied from measurement output.
"""

import json
import math
from pathlib import Path

import pytest

from src.simulation import (
    CombatSimulation, create_ship_from_fleet_data,
    Maneuver, ManeuverType, SimulationEventType, TorpedoInFlight,
    ProjectileInFlight, PRESENT_THROTTLE_MOD,
)
from src.physics import Vector3D
from src.projectile import KineticProjectile
from src.torpedo import Torpedo, TorpedoSpecs, GuidanceMode
from src.combat import HitLocation
from src.geometry import ShipGeometry


# =============================================================================
# FIXTURES / HELPERS
# =============================================================================

@pytest.fixture
def fleet_data():
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(data_path, "r") as f:
        return json.load(f)


def make_sim(seed=42):
    return CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)


def make_ship(fleet_data, ship_id="tgt", ship_type="destroyer", faction="alpha",
              position=None, velocity=None, forward=None):
    return create_ship_from_fleet_data(
        ship_id=ship_id, ship_type=ship_type, faction=faction,
        fleet_data=fleet_data,
        position=position or Vector3D(0, 0, 0),
        velocity=velocity or Vector3D(0, 0, 0),
        forward=forward or Vector3D(1, 0, 0),
    )


def inject_torpedo(sim, position, velocity, target_id="tgt",
                   disabled=False, armed=True, source_ship_id="atk",
                   torpedo_id="torp_test"):
    """Place a torpedo directly into the simulation."""
    torp = Torpedo(
        specs=TorpedoSpecs.trident(),
        position=position,
        velocity=velocity,
        target_id=target_id,
        guidance_mode=GuidanceMode.COLLISION,
    )
    torp.armed = armed
    if disabled:
        # A seeker-killed torpedo: electronics dead, engine irrelevant
        torp.fuel_exhausted = True
        torp.remaining_delta_v_kps = 0.0
    flight = TorpedoInFlight(
        torpedo_id=torpedo_id, torpedo=torp,
        source_ship_id=source_ship_id, launch_time=sim.current_time,
    )
    flight.is_disabled = disabled
    sim.torpedoes.append(flight)
    return flight


# =============================================================================
# MODE SELECTION INVARIANTS
# =============================================================================

class TestModeSelection:

    def test_guided_torpedo_produces_time_extending_response(self, fleet_data):
        """A guided torpedo inbound must produce RUN (thrust away along the
        threat axis), not a lateral jink."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Guided torpedo 300 km up the +X axis, closing at 2 km/s
        inject_torpedo(sim, Vector3D(300_000, 0, 0), Vector3D(-2000, 0, 0))

        direction, mode, throttle = sim._calculate_evasion(ship, 1.0)

        assert mode == 'RUN'
        assert direction is not None
        away = Vector3D(-1, 0, 0)  # directly away from the torpedo
        # Single threat: run direction is the exact anti-bearing, not a jink
        assert direction.dot(away) > 0.99
        assert throttle == 1.0  # full burn - the point is to buy time

    def test_unguided_torpedo_produces_lateral_jink(self, fleet_data):
        """A seeker-killed torpedo is ballistic: the response is the lateral
        EVADE used for kinetic rounds, not RUN."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Disabled torpedo dead-on from 150 km at 3 km/s -> TCA 50 s
        inject_torpedo(sim, Vector3D(150_000, 0, 0), Vector3D(-3000, 0, 0),
                       disabled=True)

        direction, mode, throttle = sim._calculate_evasion(ship, 1.0)

        assert mode == 'EVADE'
        assert direction is not None
        # Jink must be roughly perpendicular to the threat axis: the along-
        # axis component is bounded by the +-15 degree jink noise cone.
        threat_axis = Vector3D(-1, 0, 0)
        assert abs(direction.dot(threat_axis)) < 0.6

    def test_no_threats_wobble(self, fleet_data):
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        direction, mode, throttle = sim._calculate_evasion(ship, 1.0)
        assert mode == 'WOBBLE'

    def test_imminent_slug_dodged_before_distant_torpedo(self, fleet_data):
        """Priority: a kinetic round arriving before the torpedo is dodged
        first; once it is gone the ship goes back to RUN."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Guided torpedo far out (t_go ~150 s)
        inject_torpedo(sim, Vector3D(300_000, 0, 0), Vector3D(-2000, 0, 0))
        # Slug dead-on, TCA 20 s
        slug = KineticProjectile(
            position=Vector3D(0, 160_000, 0),
            velocity=Vector3D(0, -8000, 0),
            mass_kg=50.0,
        )
        sim.projectiles.append(ProjectileInFlight(
            projectile_id="slug_test", projectile=slug,
            source_ship_id="atk", target_ship_id="tgt",
        ))

        _, mode_with_slug, _ = sim._calculate_evasion(ship, 1.0)
        assert mode_with_slug == 'EVADE'

        sim.projectiles.clear()
        _, mode_without_slug, _ = sim._calculate_evasion(ship, 1.0)
        assert mode_without_slug == 'RUN'


# =============================================================================
# TERMINAL ARMOR PRESENTATION
# =============================================================================

class TestTerminalPresentation:

    def test_presentation_puts_bearing_in_thickest_arc_nose(self, fleet_data):
        """Destroyer (nose thickest): with the torpedo 60 degrees off the
        nose and enough time to turn, PRESENT must command a facing that
        makes the impact a NOSE hit."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Torpedo 60 deg off the nose, 175 km out, closing 5 km/s -> t_go 35 s
        bearing = Vector3D(math.cos(math.radians(60)),
                           math.sin(math.radians(60)), 0)
        pos = bearing * 175_000
        vel = bearing * -5000  # straight at the ship
        inject_torpedo(sim, pos, vel)

        direction, mode, throttle = sim._calculate_evasion(ship, 1.0)

        assert mode == 'PRESENT'
        assert direction is not None
        assert throttle == PRESENT_THROTTLE_MOD
        # If the ship reaches the commanded facing, the impact vector must
        # strike the NOSE arc (thickest section for every fleet hull).
        impact_vector = vel.normalized()
        geometry = ship.geometry or ShipGeometry(
            length_m=100, radius_m=12.5,
            nose_cone_length_m=20, engine_section_length_m=20)
        location = geometry.calculate_hit_location(impact_vector, direction)
        assert location == HitLocation.NOSE

    def test_presentation_respects_thickest_section_not_assumed_nose(self, fleet_data):
        """'Thickest facing' must be derived from actual armor: gut the nose
        and beef up the lateral belt, and PRESENT must go broadside."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Rewrite armor: lateral thickest by far
        ship.armor.get_section(HitLocation.NOSE).thickness_cm = 15.0
        ship.armor.get_section(HitLocation.LATERAL).thickness_cm = 150.0
        ship.armor.get_section(HitLocation.TAIL).thickness_cm = 20.0

        # Note: at 60 deg off the nose the bearing is ALREADY in the lateral
        # band, so from far out the correct choice is to keep RUNning (zero
        # turn needed). Force the terminal phase with a short time-to-go.
        bearing = Vector3D(math.cos(math.radians(60)),
                           math.sin(math.radians(60)), 0)
        inject_torpedo(sim, bearing * 40_000, bearing * -5000)  # t_go 8 s

        direction, mode, _ = sim._calculate_evasion(ship, 1.0)

        assert mode == 'PRESENT'
        impact_vector = (bearing * -1)
        geometry = ship.geometry
        location = geometry.calculate_hit_location(impact_vector, direction)
        assert location == HitLocation.LATERAL

    def test_presentation_never_increases_expected_damage(self, fleet_data):
        """The chosen facing's expected damage must never exceed the
        do-nothing (current attitude) baseline - including with several
        torpedoes inbound from different bearings."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        # Two torpedoes from very different bearings, both with time to turn
        b1 = Vector3D(0, 1, 0)
        b2 = Vector3D(math.cos(math.radians(-40)),
                      math.sin(math.radians(-40)), 0)
        inject_torpedo(sim, b1 * 175_000, b1 * -5000, torpedo_id="t1")
        inject_torpedo(sim, b2 * 175_000, b2 * -5000, torpedo_id="t2")

        threats = sim._gather_guided_torpedo_threats(ship)
        assert len(threats) == 2
        terminal = [t for t in threats if t['terminal']]
        assert terminal, "both torpedoes should be terminal in this setup"

        chosen = sim._calculate_presentation_direction(ship, terminal)
        baseline = sim._expected_presentation_damage_gj(
            ship, ship.forward, terminal)
        chosen_damage = sim._expected_presentation_damage_gj(
            ship, chosen, terminal)
        assert chosen_damage <= baseline + 1e-9

    def test_presentation_rotation_is_gradual(self, fleet_data):
        """Rotation is not free: the commanded facing must be approached at
        the ship's finite angular rate, never teleported."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        atk = make_ship(fleet_data, ship_id="atk", ship_type="corvette",
                        faction="beta", position=Vector3D(0, 400_000, 0),
                        forward=Vector3D(0, -1, 0))
        sim.add_ship(ship)
        sim.add_ship(atk)
        sim.inject_command("tgt", Maneuver(ManeuverType.EVASIVE,
                                           start_time=0.0, throttle=1.0))
        sim.inject_command("atk", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
        assert sim.inject_command("atk", {"type": "launch_torpedo",
                                          "target_id": "tgt"})

        _, max_vel_rad_s = sim._get_rotation_params(ship, engines_on=True)
        prev_forward = Vector3D(ship.forward.x, ship.forward.y, ship.forward.z)
        for _ in range(120):
            sim.step()
            step_angle = prev_forward.angle_to(ship.forward)
            # Per-tick rotation can never exceed the max angular velocity
            assert step_angle <= max_vel_rad_s * sim.time_step * 1.05 + 1e-6
            prev_forward = Vector3D(ship.forward.x, ship.forward.y, ship.forward.z)
            if not sim.torpedoes:
                break


# =============================================================================
# DISABLED TORPEDOES ARE PHYSICAL OBJECTS
# =============================================================================

class TestDisabledTorpedoImpacts:

    def test_disabled_torpedo_hits_nonevading_target(self, fleet_data):
        """A seeker-killed torpedo on a dead-on trajectory must strike a
        target that does not maneuver."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        sim.inject_command("tgt", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
        inject_torpedo(sim, Vector3D(0, 150_000, 0), Vector3D(0, -3000, 0),
                       disabled=True)

        for _ in range(120):
            sim.step()
            if not sim.torpedoes:
                break

        impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
        assert len(impacts) == 1
        assert ship.damage_taken_gj > 0
        # Ship faces +X, torpedo arrives along -Y: a flank strike
        assert impacts[0].data['hit_location'] == 'lateral'

    def test_disabled_torpedo_misses_offset_geometry(self, fleet_data):
        """No guidance means no correction: a trajectory passing 5 km abeam
        must NOT hit."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        sim.inject_command("tgt", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
        inject_torpedo(sim, Vector3D(5_000, 150_000, 0), Vector3D(0, -3000, 0),
                       disabled=True)

        for _ in range(120):
            sim.step()

        impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
        assert len(impacts) == 0
        assert ship.damage_taken_gj == 0

    def test_evasion_defeats_disabled_torpedo(self, fleet_data):
        """Lateral evasion must turn a certain ballistic hit into a miss
        given ~50 s of warning - and the same trajectory must hit the
        non-evading control (same seed)."""
        for seed in (1, 2, 3):
            hits = {}
            for maneuver in (ManeuverType.EVASIVE, ManeuverType.MAINTAIN):
                sim = make_sim(seed)
                ship = make_ship(fleet_data)
                sim.add_ship(ship)
                sim.inject_command("tgt", Maneuver(maneuver, start_time=0.0,
                                                   throttle=1.0))
                inject_torpedo(sim, Vector3D(0, 150_000, 0),
                               Vector3D(0, -3000, 0), disabled=True)
                for _ in range(120):
                    sim.step()
                    if not sim.torpedoes:
                        break
                impacts = sim.get_events_by_type(
                    SimulationEventType.TORPEDO_IMPACT)
                hits[maneuver] = len(impacts)
            assert hits[ManeuverType.MAINTAIN] == 1, \
                f"seed {seed}: control should be hit"
            assert hits[ManeuverType.EVASIVE] == 0, \
                f"seed {seed}: evading ship should escape a ballistic torpedo"


# =============================================================================
# INTEGRATION: RUN BUYS TIME, PRESENT SHIFTS HIT LOCATION
# =============================================================================

def _run_guided_engagement(fleet_data, seed, maneuver_type,
                           attacker_pos, target_forward=Vector3D(1, 0, 0)):
    sim = make_sim(seed)
    ship = make_ship(fleet_data, forward=target_forward)
    atk = make_ship(fleet_data, ship_id="atk", ship_type="corvette",
                    faction="beta", position=attacker_pos,
                    forward=(Vector3D(0, 0, 0) - attacker_pos).normalized())
    sim.add_ship(ship)
    sim.add_ship(atk)
    sim.inject_command("tgt", Maneuver(maneuver_type, start_time=0.0,
                                       throttle=1.0))
    sim.inject_command("atk", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
    assert sim.inject_command("atk", {"type": "launch_torpedo",
                                      "target_id": "tgt"})
    resolved_t = None
    for _ in range(400):
        sim.step()
        if not sim.torpedoes:
            resolved_t = sim.current_time
            break
    impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
    return {
        'resolved_t': resolved_t,
        'hit': len(impacts) > 0,
        'impact_speed_kps': impacts[0].data['impact_speed_kps'] if impacts else None,
        'damage_gj': impacts[0].data['total_damage_gj'] if impacts else 0.0,
        'hit_location': impacts[0].data['hit_location'] if impacts else None,
    }


class TestGuidedEngagementOutcomes:

    def test_running_reduces_closing_speed_and_damage(self, fleet_data):
        """The point of RUN: against the same torpedo, an EVASIVE target must
        be struck at a LOWER relative speed (and so take less kinetic damage)
        than a passive one, and no earlier."""
        evasive = _run_guided_engagement(
            fleet_data, 42, ManeuverType.EVASIVE, Vector3D(400_000, 0, 0))
        passive = _run_guided_engagement(
            fleet_data, 42, ManeuverType.MAINTAIN, Vector3D(400_000, 0, 0))
        # Inside the NEZ the torpedo still hits - evasion is not magic
        assert evasive['hit'] and passive['hit']
        assert evasive['impact_speed_kps'] < passive['impact_speed_kps']
        assert evasive['damage_gj'] < passive['damage_gj']
        assert evasive['resolved_t'] >= passive['resolved_t']

    def test_presentation_shifts_flank_shot_to_thick_facing(self, fleet_data):
        """Torpedo from the flank: the passive ship takes a LATERAL hit; the
        evading ship must catch it on its thickest section (nose for every
        fleet hull)."""
        passive = _run_guided_engagement(
            fleet_data, 42, ManeuverType.MAINTAIN, Vector3D(0, 400_000, 0))
        evasive = _run_guided_engagement(
            fleet_data, 42, ManeuverType.EVASIVE, Vector3D(0, 400_000, 0))
        assert passive['hit'] and evasive['hit']
        assert passive['hit_location'] == 'lateral'
        assert evasive['hit_location'] == 'nose'


# =============================================================================
# STATUS REPORTING
# =============================================================================

class TestEvasionStatus:

    def test_status_counts_torpedo_threats(self, fleet_data):
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        inject_torpedo(sim, Vector3D(300_000, 0, 0), Vector3D(-2000, 0, 0))
        status = sim._get_evasion_status(ship)
        assert status['threat_count'] >= 1
        assert status['mode'] in ('RUN', 'PRESENT')

    def test_status_query_does_not_latch_commit(self, fleet_data):
        """dt=0 status queries must not mutate the terminal commit latch."""
        sim = make_sim()
        ship = make_ship(fleet_data)
        sim.add_ship(ship)
        bearing = Vector3D(0, 1, 0)
        flight = inject_torpedo(sim, bearing * 175_000, bearing * -5000)
        assert not flight.terminal_latched
        sim._get_evasion_status(ship)  # read-only path (dt=0)
        assert not flight.terminal_latched
        sim._calculate_evasion(ship, 1.0)  # live maneuver path latches
        assert flight.terminal_latched
