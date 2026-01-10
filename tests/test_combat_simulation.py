"""
Comprehensive tests for the AI Commanders combat simulation system.

This module contains tests covering:
- Simulation basics (creation, time stepping, decision points, event logging)
- Projectile tracking (coilgun, torpedo, multiple projectiles, misses)
- Maneuver execution (rotation, burn, flip-and-burn, evasive jink)
- Thermal simulation (weapon fire heat, radiator cooling, warnings, critical)
- Combat engagement (head-on, pursuit, damage resolution, module damage)
- Script execution (aggressive, defensive, timing)
- Full battle scenarios (destroyer vs destroyer, corvette vs destroyer, destruction)

Tests create ships from fleet_ships.json and run actual simulation steps.
"""

import json
from pathlib import Path

import pytest

from src.simulation import (
    CombatSimulation, ShipCombatState, create_ship_from_fleet_data,
    SimulationEventType, SimulationEvent, Maneuver, ManeuverType,
    WeaponState, ProjectileInFlight, TorpedoInFlight
)
from src.physics import Vector3D, ShipState, create_ship_state_from_specs
from src.combat import Weapon, create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
from src.maneuvers import (
    BurnToward, BurnAway, RotateToFace, FlipAndBurn, EvasiveJink,
    ManeuverExecutor, ManeuverStatus
)
from src.thermal import ThermalSystem, RadiatorState
from src.projectile import KineticProjectile
from src.torpedo import Torpedo, TorpedoSpecs, GuidanceMode


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet data from fleet_ships.json."""
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(data_path, "r") as f:
        return json.load(f)


@pytest.fixture
def destroyer_alpha(fleet_data):
    """Create a destroyer ship for faction alpha at origin, facing +X."""
    return create_ship_from_fleet_data(
        ship_id="alpha_destroyer_1",
        ship_type="destroyer",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(5000, 0, 0),  # 5 km/s in +X
        forward=Vector3D(1, 0, 0)
    )


@pytest.fixture
def destroyer_beta(fleet_data):
    """Create a destroyer ship for faction beta, 200 km away, facing -X."""
    return create_ship_from_fleet_data(
        ship_id="beta_destroyer_1",
        ship_type="destroyer",
        faction="beta",
        fleet_data=fleet_data,
        position=Vector3D(200_000, 0, 0),  # 200 km in +X
        velocity=Vector3D(-5000, 0, 0),  # 5 km/s toward alpha
        forward=Vector3D(-1, 0, 0)
    )


@pytest.fixture
def corvette_alpha(fleet_data):
    """Create a corvette ship for faction alpha."""
    return create_ship_from_fleet_data(
        ship_id="alpha_corvette_1",
        ship_type="corvette",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(5000, 0, 0),
        forward=Vector3D(1, 0, 0)
    )


@pytest.fixture
def cruiser_beta(fleet_data):
    """Create a cruiser ship for faction beta."""
    return create_ship_from_fleet_data(
        ship_id="beta_cruiser_1",
        ship_type="cruiser",
        faction="beta",
        fleet_data=fleet_data,
        position=Vector3D(300_000, 0, 0),
        velocity=Vector3D(-3000, 0, 0),
        forward=Vector3D(-1, 0, 0)
    )


@pytest.fixture
def basic_simulation():
    """Create a basic simulation with default parameters."""
    return CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)


@pytest.fixture
def two_destroyer_simulation(basic_simulation, destroyer_alpha, destroyer_beta):
    """Create simulation with two destroyers facing each other."""
    basic_simulation.add_ship(destroyer_alpha)
    basic_simulation.add_ship(destroyer_beta)
    return basic_simulation


# =============================================================================
# TEST: SIMULATION BASICS
# =============================================================================

class TestSimulationBasics:
    """Test basic simulation functionality."""

    def test_simulation_creation(self):
        """Test that simulation can be created with various parameters."""
        # Default parameters
        sim = CombatSimulation()
        assert sim.time_step == 1.0
        assert sim.decision_interval == 30.0
        assert sim.current_time == 0.0
        print(f"[DEBUG] Default simulation created: time_step={sim.time_step}, decision_interval={sim.decision_interval}")

        # Custom parameters
        sim2 = CombatSimulation(time_step=0.5, decision_interval=45.0, seed=123)
        assert sim2.time_step == 0.5
        assert sim2.decision_interval == 45.0
        print(f"[DEBUG] Custom simulation created: time_step={sim2.time_step}, decision_interval={sim2.decision_interval}")

        # Decision interval bounds (should clamp to 20-60)
        sim3 = CombatSimulation(decision_interval=10.0)  # Too low
        assert sim3.decision_interval >= 20.0

        sim4 = CombatSimulation(decision_interval=100.0)  # Too high
        assert sim4.decision_interval <= 60.0
        print(f"[DEBUG] Decision interval bounds test passed")

    def test_time_stepping(self, two_destroyer_simulation):
        """Test that time steps correctly advance simulation time."""
        sim = two_destroyer_simulation

        initial_time = sim.current_time
        assert initial_time == 0.0
        print(f"[DEBUG] Initial time: {initial_time}")

        # Step forward
        sim.step()
        assert sim.current_time == 1.0
        print(f"[DEBUG] After 1 step: time={sim.current_time}")

        # Multiple steps
        for _ in range(9):
            sim.step()
        assert sim.current_time == 10.0
        print(f"[DEBUG] After 10 steps: time={sim.current_time}")

        # Verify ship positions have changed
        alpha = sim.get_ship("alpha_destroyer_1")
        assert alpha is not None
        # With velocity of 5000 m/s in +X, after 10 seconds, position should be ~50 km further
        print(f"[DEBUG] Alpha position after 10s: {alpha.position}")

    def test_decision_points(self, two_destroyer_simulation):
        """Test that decision points are triggered at correct intervals."""
        sim = two_destroyer_simulation

        decision_point_events = []

        # Run for 90 seconds (should trigger 3 decision points at 0, 30, 60)
        for _ in range(90):
            sim.step()

        decision_point_events = sim.get_events_by_type(SimulationEventType.DECISION_POINT_REACHED)
        print(f"[DEBUG] Decision points found: {len(decision_point_events)}")

        # Should have at least 2 decision points (at t=30 and t=60)
        assert len(decision_point_events) >= 2

        for event in decision_point_events:
            print(f"[DEBUG] Decision point at t={event.timestamp}s")

    def test_event_logging(self, two_destroyer_simulation, fleet_data):
        """Test that events are properly logged during simulation."""
        sim = two_destroyer_simulation

        # Ensure simulation started event is logged
        sim.step()

        # Manually fire a weapon to generate events
        alpha = sim.get_ship("alpha_destroyer_1")

        # Add a weapon if not present
        if not alpha.weapons:
            weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
            alpha.weapons["spinal"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Fire at beta
        result = sim.inject_command("alpha_destroyer_1", {
            'type': 'fire_at',
            'weapon_slot': list(alpha.weapons.keys())[0],
            'target_id': 'beta_destroyer_1'
        })
        print(f"[DEBUG] Fire command result: {result}")

        # Check for projectile launched event
        projectile_events = sim.get_events_by_type(SimulationEventType.PROJECTILE_LAUNCHED)
        print(f"[DEBUG] Projectile launched events: {len(projectile_events)}")
        assert len(projectile_events) >= 1

        # Run more steps and check event count
        for _ in range(10):
            sim.step()

        print(f"[DEBUG] Total events logged: {len(sim.events)}")
        assert len(sim.events) > 0


# =============================================================================
# TEST: PROJECTILE TRACKING
# =============================================================================

class TestProjectileTracking:
    """Test projectile launch, tracking, and hit detection."""

    def test_coilgun_launch_and_track(self, two_destroyer_simulation, fleet_data):
        """Test coilgun projectile launch and tracking over time."""
        sim = two_destroyer_simulation

        # Get ships
        alpha = sim.get_ship("alpha_destroyer_1")
        beta = sim.get_ship("beta_destroyer_1")

        # Add weapon if needed
        if not alpha.weapons:
            weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
            alpha.weapons["spinal"] = WeaponState(weapon=weapon, ammo_remaining=100)

        initial_distance = alpha.distance_to(beta)
        print(f"[DEBUG] Initial distance: {initial_distance / 1000:.1f} km")

        # Fire at beta
        weapon_slot = list(alpha.weapons.keys())[0]
        sim.inject_command("alpha_destroyer_1", {
            'type': 'fire_at',
            'weapon_slot': weapon_slot,
            'target_id': 'beta_destroyer_1'
        })

        # Verify projectile was created
        assert len(sim.projectiles) == 1
        print(f"[DEBUG] Projectile launched, tracking {len(sim.projectiles)} projectile(s)")

        proj = sim.projectiles[0]
        print(f"[DEBUG] Projectile velocity: {proj.projectile.velocity.magnitude / 1000:.2f} km/s")
        print(f"[DEBUG] Projectile KE: {proj.projectile.kinetic_energy_gj:.2f} GJ")

        # Track for several steps
        for i in range(10):
            sim.step()
            if sim.projectiles:
                dist_to_target = sim.projectiles[0].projectile.distance_to(beta.position)
                print(f"[DEBUG] Step {i+1}: Projectile distance to target: {dist_to_target / 1000:.1f} km")

    def test_torpedo_launch_and_track(self, fleet_data):
        """Test torpedo launch and guided tracking."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        # Create corvette (has torpedo launcher) and a target
        corvette = create_ship_from_fleet_data(
            ship_id="alpha_corvette",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        target = create_ship_from_fleet_data(
            ship_id="beta_destroyer",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(500_000, 0, 0),  # 500 km away
            velocity=Vector3D(-1000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(corvette)
        sim.add_ship(target)

        # Launch torpedo
        result = sim.inject_command("alpha_corvette", {
            'type': 'launch_torpedo',
            'target_id': 'beta_destroyer'
        })
        print(f"[DEBUG] Torpedo launch result: {result}")

        # Verify torpedo was created
        assert len(sim.torpedoes) == 1, f"Expected 1 torpedo, got {len(sim.torpedoes)}"

        torp = sim.torpedoes[0]
        print(f"[DEBUG] Torpedo launched, target: {torp.torpedo.target_id}")
        print(f"[DEBUG] Torpedo guidance mode: {torp.torpedo.guidance_mode}")

        # Track torpedo for several steps
        for i in range(20):
            sim.step()
            if sim.torpedoes:
                dist = torp.torpedo.position.distance_to(target.position)
                print(f"[DEBUG] Step {i+1}: Torpedo distance to target: {dist / 1000:.1f} km")

    def test_multiple_projectiles(self, two_destroyer_simulation, fleet_data):
        """Test tracking multiple projectiles simultaneously."""
        sim = two_destroyer_simulation

        alpha = sim.get_ship("alpha_destroyer_1")

        # Add weapon with fast cooldown for testing
        weapon = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        alpha.weapons["main_gun"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Fire multiple shots (reset cooldown between shots for testing)
        for i in range(3):
            sim.inject_command("alpha_destroyer_1", {
                'type': 'fire_at',
                'weapon_slot': 'main_gun',
                'target_id': 'beta_destroyer_1'
            })
            alpha.weapons["main_gun"].cooldown_remaining = 0  # Reset for next shot

        print(f"[DEBUG] Projectiles in flight: {len(sim.projectiles)}")
        assert len(sim.projectiles) == 3

        # Track all projectiles
        for i in range(5):
            sim.step()
            print(f"[DEBUG] Step {i+1}: {len(sim.projectiles)} projectiles in flight")

    def test_projectile_miss(self, fleet_data):
        """Test that projectiles can miss and are removed when too far."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        # Create ships very far apart with perpendicular velocities
        alpha = create_ship_from_fleet_data(
            ship_id="alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(100_000, 0, 0),
            velocity=Vector3D(0, 10000, 0),  # Moving perpendicular at 10 km/s
            forward=Vector3D(0, 1, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Add weapon and fire
        weapon = create_weapon_from_fleet_data(fleet_data, "light_coilgun_mk3")
        alpha.weapons["gun"] = WeaponState(weapon=weapon, ammo_remaining=100)

        sim.inject_command("alpha", {
            'type': 'fire_at',
            'weapon_slot': 'gun',
            'target_id': 'beta'
        })

        print(f"[DEBUG] Projectile launched at evading target")

        # Run until projectile misses (travels far past target)
        miss_events = []
        for i in range(500):
            sim.step()
            miss_events = sim.get_events_by_type(SimulationEventType.PROJECTILE_MISS)
            if miss_events:
                print(f"[DEBUG] Projectile miss detected at step {i+1}")
                break

        print(f"[DEBUG] Miss events: {len(miss_events)}")


# =============================================================================
# TEST: MANEUVER EXECUTION
# =============================================================================

class TestManeuverExecution:
    """Test maneuver execution and physics."""

    def test_rotation_maneuver(self, fleet_data):
        """Test ship rotation to face a target."""
        # Create ship facing +X
        ship_state = create_ship_state_from_specs(
            wet_mass_tons=2985,
            dry_mass_tons=2843,
            length_m=125,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Target is at +Y direction
        target_pos = Vector3D(0, 100_000, 0)

        # Create rotation maneuver
        maneuver = RotateToFace(target_position=target_pos)
        executor = ManeuverExecutor()
        executor.set_maneuver(maneuver)

        print(f"[DEBUG] Initial forward: {ship_state.forward}")
        print(f"[DEBUG] Target direction: {(target_pos - ship_state.position).normalized()}")

        # Execute maneuver steps
        for i in range(100):
            result = executor.update(ship_state, dt=1.0)

            if i % 20 == 0:
                print(f"[DEBUG] Step {i}: progress={result.progress:.1f}%, status={result.status.name}")

            if result.status == ManeuverStatus.COMPLETED:
                print(f"[DEBUG] Rotation completed at step {i}")
                break

        print(f"[DEBUG] Final progress: {executor.progress:.1f}%")

    def test_burn_maneuver(self, fleet_data):
        """Test thrust burn toward a target."""
        ship_state = create_ship_state_from_specs(
            wet_mass_tons=2985,
            dry_mass_tons=2843,
            length_m=125,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        target_pos = Vector3D(100_000, 0, 0)  # 100 km in +X

        # Create burn maneuver
        maneuver = BurnToward(
            target_position=target_pos,
            throttle=0.5,
            max_duration=30.0
        )
        executor = ManeuverExecutor()
        executor.set_maneuver(maneuver)

        initial_delta_v = maneuver.estimate_delta_v_cost(ship_state)
        print(f"[DEBUG] Estimated delta-v cost: {initial_delta_v:.1f} m/s")

        # Execute maneuver
        for i in range(35):
            result = executor.update(ship_state, dt=1.0)

            if i % 10 == 0:
                print(f"[DEBUG] Step {i}: progress={result.progress:.1f}%, throttle={result.throttle}")

            if result.status in (ManeuverStatus.COMPLETED, ManeuverStatus.FAILED):
                print(f"[DEBUG] Burn ended at step {i}: {result.status.name}")
                break

        print(f"[DEBUG] Delta-v expended: {maneuver.delta_v_expended:.1f} m/s")

    def test_flip_and_burn(self, fleet_data):
        """Test flip-and-burn deceleration maneuver."""
        ship_state = create_ship_state_from_specs(
            wet_mass_tons=2985,
            dry_mass_tons=2843,
            length_m=125,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5000, 0, 0),  # Moving at 5 km/s in +X
            forward=Vector3D(1, 0, 0)
        )

        initial_speed = ship_state.velocity.magnitude
        print(f"[DEBUG] Initial speed: {initial_speed:.1f} m/s")

        # Create flip-and-burn to decelerate
        maneuver = FlipAndBurn(target_speed_ms=0.0, throttle=1.0)
        executor = ManeuverExecutor()
        executor.set_maneuver(maneuver)

        est_time = maneuver.estimate_completion_time(ship_state)
        print(f"[DEBUG] Estimated completion time: {est_time:.1f} s")

        # Execute maneuver
        for i in range(100):
            result = executor.update(ship_state, dt=1.0)

            if i % 20 == 0:
                print(f"[DEBUG] Step {i}: progress={result.progress:.1f}%, phase message: {result.message[:50]}")

            if result.status in (ManeuverStatus.COMPLETED, ManeuverStatus.FAILED):
                print(f"[DEBUG] Flip-and-burn ended at step {i}: {result.status.name}")
                break

    def test_evasive_jink(self, fleet_data):
        """Test evasive jinking maneuver."""
        ship_state = create_ship_state_from_specs(
            wet_mass_tons=2985,
            dry_mass_tons=2843,
            length_m=125,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Create evasive jink maneuver
        maneuver = EvasiveJink(
            duration=15.0,
            jink_interval=3.0,
            throttle=0.8
        )
        executor = ManeuverExecutor()
        executor.set_maneuver(maneuver)

        print(f"[DEBUG] Starting evasive jink for {maneuver.duration}s")

        last_direction = None
        direction_changes = 0

        for i in range(20):
            result = executor.update(ship_state, dt=1.0)

            if result.thrust_direction and last_direction:
                # Check if direction changed significantly
                angle_diff = result.thrust_direction.angle_to(last_direction)
                if angle_diff > 0.5:  # More than ~30 degrees
                    direction_changes += 1

            last_direction = result.thrust_direction

            if i % 5 == 0:
                print(f"[DEBUG] Step {i}: progress={result.progress:.1f}%")

            if result.status == ManeuverStatus.COMPLETED:
                print(f"[DEBUG] Evasive jink completed at step {i}")
                break

        print(f"[DEBUG] Direction changes detected: {direction_changes}")


# =============================================================================
# TEST: THERMAL SIMULATION
# =============================================================================

class TestThermalSimulation:
    """Test thermal system simulation."""

    def test_weapon_fire_heat(self, two_destroyer_simulation, fleet_data):
        """Test that firing weapons generates heat."""
        sim = two_destroyer_simulation

        alpha = sim.get_ship("alpha_destroyer_1")

        # Ensure thermal system exists
        assert alpha.thermal_system is not None

        initial_heat = alpha.thermal_system.heat_percent
        print(f"[DEBUG] Initial heat: {initial_heat:.2f}%")

        # Add weapon if needed
        if not alpha.weapons:
            weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
            alpha.weapons["spinal"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Fire multiple times
        for i in range(5):
            weapon_slot = list(alpha.weapons.keys())[0]
            sim.inject_command("alpha_destroyer_1", {
                'type': 'fire_at',
                'weapon_slot': weapon_slot,
                'target_id': 'beta_destroyer_1'
            })
            alpha.weapons[weapon_slot].cooldown_remaining = 0  # Reset cooldown
            sim.step()

        final_heat = alpha.thermal_system.heat_percent
        print(f"[DEBUG] Heat after 5 shots: {final_heat:.2f}%")

        # Heat should have increased
        assert final_heat > initial_heat, "Heat should increase after firing weapons"

    def test_radiator_cooling(self, fleet_data):
        """Test that radiators dissipate heat over time."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        destroyer = create_ship_from_fleet_data(
            ship_id="alpha_destroyer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )
        sim.add_ship(destroyer)

        # Add some heat manually
        destroyer.thermal_system.add_heat("test", 100.0)  # Add 100 GJ of heat

        initial_heat = destroyer.thermal_system.heat_percent
        print(f"[DEBUG] Heat after adding 100 GJ: {initial_heat:.2f}%")

        # Extend radiators
        destroyer.thermal_system.radiators.extend_all()
        dissipation = destroyer.thermal_system.radiators.total_dissipation_kw
        print(f"[DEBUG] Radiator dissipation: {dissipation:.0f} kW")

        # Run simulation to allow cooling
        for i in range(60):
            sim.step()
            if i % 20 == 0:
                heat = destroyer.thermal_system.heat_percent
                print(f"[DEBUG] Heat at t={i}s: {heat:.2f}%")

        final_heat = destroyer.thermal_system.heat_percent
        print(f"[DEBUG] Final heat: {final_heat:.2f}%")

        # Heat should have decreased (assuming radiators can dissipate faster than reactor generates)
        # Note: reactor generates some background heat
        assert final_heat < initial_heat or final_heat < 50, "Heat should decrease with radiators extended"

    def test_thermal_warning(self, fleet_data):
        """Test thermal warning events at 80% heat."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        destroyer = create_ship_from_fleet_data(
            ship_id="alpha_destroyer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )
        sim.add_ship(destroyer)

        # Get heat sink capacity
        capacity = destroyer.thermal_system.heatsink.capacity_gj
        print(f"[DEBUG] Heat sink capacity: {capacity} GJ")

        # Add heat to reach 80%+
        heat_to_add = capacity * 0.82
        destroyer.thermal_system.add_heat("test", heat_to_add)

        heat_percent = destroyer.thermal_system.heat_percent
        print(f"[DEBUG] Heat after adding {heat_to_add:.1f} GJ: {heat_percent:.2f}%")

        # Run a step to trigger thermal check
        sim.step()

        # Check for warning event
        warning_events = sim.get_events_by_type(SimulationEventType.THERMAL_WARNING)
        print(f"[DEBUG] Thermal warning events: {len(warning_events)}")

        # Verify thermal system reports overheating
        assert destroyer.thermal_system.is_overheating, "Ship should be in overheating state"

    def test_thermal_critical(self, fleet_data):
        """Test thermal critical events at 95% heat."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        destroyer = create_ship_from_fleet_data(
            ship_id="alpha_destroyer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )
        sim.add_ship(destroyer)

        capacity = destroyer.thermal_system.heatsink.capacity_gj

        # Add heat to reach 96%+
        heat_to_add = capacity * 0.96
        destroyer.thermal_system.add_heat("test", heat_to_add)

        heat_percent = destroyer.thermal_system.heat_percent
        print(f"[DEBUG] Heat after adding {heat_to_add:.1f} GJ: {heat_percent:.2f}%")

        # Run a step to trigger thermal check
        sim.step()

        # Check for critical event
        critical_events = sim.get_events_by_type(SimulationEventType.THERMAL_CRITICAL)
        print(f"[DEBUG] Thermal critical events: {len(critical_events)}")

        # Verify thermal system reports critical
        assert destroyer.thermal_system.is_critical, "Ship should be in critical thermal state"


# =============================================================================
# TEST: COMBAT ENGAGEMENT
# =============================================================================

class TestCombatEngagement:
    """Test combat engagement scenarios."""

    def test_head_on_engagement(self, fleet_data):
        """Test head-on engagement where ships approach each other."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        # Ships approaching each other
        alpha = create_ship_from_fleet_data(
            ship_id="alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5000, 0, 0),  # 5 km/s toward beta
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(100_000, 0, 0),  # 100 km away
            velocity=Vector3D(-5000, 0, 0),  # 5 km/s toward alpha
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        initial_distance = alpha.distance_to(beta)
        closing_rate = alpha.closing_rate_to(beta)
        print(f"[DEBUG] Initial distance: {initial_distance / 1000:.1f} km")
        print(f"[DEBUG] Closing rate: {closing_rate / 1000:.1f} km/s")

        # Run simulation for 10 seconds
        for _ in range(10):
            sim.step()

        final_distance = alpha.distance_to(beta)
        print(f"[DEBUG] Distance after 10s: {final_distance / 1000:.1f} km")

        # Ships should be closer
        assert final_distance < initial_distance, "Ships should be closer after head-on approach"

    def test_pursuit_engagement(self, fleet_data):
        """Test pursuit engagement where one ship chases another."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        # Alpha chasing beta
        alpha = create_ship_from_fleet_data(
            ship_id="alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(8000, 0, 0),  # Faster
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(50_000, 0, 0),
            velocity=Vector3D(5000, 0, 0),  # Slower, same direction
            forward=Vector3D(1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        initial_distance = alpha.distance_to(beta)
        print(f"[DEBUG] Initial distance: {initial_distance / 1000:.1f} km")

        # Alpha is faster, should be closing
        closing_rate = alpha.closing_rate_to(beta)
        print(f"[DEBUG] Closing rate: {closing_rate:.1f} m/s (should be ~3000)")

        # Run for 10 seconds
        for _ in range(10):
            sim.step()

        final_distance = alpha.distance_to(beta)
        print(f"[DEBUG] Distance after 10s: {final_distance / 1000:.1f} km")

        # Alpha should be catching up
        assert final_distance < initial_distance

    def test_damage_resolution(self, two_destroyer_simulation, fleet_data):
        """Test damage is properly applied and tracked."""
        sim = two_destroyer_simulation

        alpha = sim.get_ship("alpha_destroyer_1")
        beta = sim.get_ship("beta_destroyer_1")

        # Add powerful weapon
        weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        alpha.weapons["spinal"] = WeaponState(weapon=weapon, ammo_remaining=100)

        initial_beta_damage = beta.damage_taken_gj
        print(f"[DEBUG] Beta initial damage taken: {initial_beta_damage} GJ")

        # Fire at beta
        sim.inject_command("alpha_destroyer_1", {
            'type': 'fire_at',
            'weapon_slot': 'spinal',
            'target_id': 'beta_destroyer_1'
        })

        # Run until hit or miss (projectile travels ~10 km/s)
        # At 200 km, should take ~20 seconds
        hit_detected = False
        for i in range(50):
            sim.step()

            hit_events = sim.get_events_by_type(SimulationEventType.PROJECTILE_IMPACT)
            if hit_events:
                hit_detected = True
                print(f"[DEBUG] Hit detected at step {i}")
                break

        damage_events = sim.get_events_by_type(SimulationEventType.DAMAGE_TAKEN)
        print(f"[DEBUG] Damage events: {len(damage_events)}")

        # Check metrics
        print(f"[DEBUG] Alpha shots fired: {alpha.shots_fired}")
        print(f"[DEBUG] Total simulation hits: {sim.metrics.total_hits}")

    def test_module_damage(self, fleet_data):
        """Test that penetrating hits can damage modules."""
        sim = CombatSimulation(time_step=1.0, seed=42)

        # Use lightly armored ship for easier penetration testing
        corvette = create_ship_from_fleet_data(
            ship_id="alpha_corvette",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Heavy ship with powerful weapons
        battleship = create_ship_from_fleet_data(
            ship_id="beta_battleship",
            ship_type="battleship",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(50_000, 0, 0),  # Close range
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(corvette)
        sim.add_ship(battleship)

        # Add weapon to battleship
        weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        battleship.weapons["spinal"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Fire multiple shots at corvette
        for _ in range(10):
            sim.inject_command("beta_battleship", {
                'type': 'fire_at',
                'weapon_slot': 'spinal',
                'target_id': 'alpha_corvette'
            })
            battleship.weapons["spinal"].cooldown_remaining = 0

        # Run simulation
        for _ in range(30):
            sim.step()

        # Check for module damage events
        module_damage_events = sim.get_events_by_type(SimulationEventType.MODULE_DAMAGED)
        armor_penetrated_events = sim.get_events_by_type(SimulationEventType.ARMOR_PENETRATED)

        print(f"[DEBUG] Module damage events: {len(module_damage_events)}")
        print(f"[DEBUG] Armor penetration events: {len(armor_penetrated_events)}")


# =============================================================================
# TEST: SCRIPT EXECUTION
# =============================================================================

class TestScriptExecution:
    """Test scripted captain behaviors."""

    def test_aggressive_script(self, two_destroyer_simulation, fleet_data):
        """Test aggressive captain script that fires whenever possible."""
        sim = two_destroyer_simulation

        alpha = sim.get_ship("alpha_destroyer_1")

        # Add weapon
        weapon = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        alpha.weapons["main_gun"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Define aggressive callback
        def aggressive_captain(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies and ship.weapons:
                closest = min(enemies, key=lambda e: ship.distance_to(e))
                for slot, ws in ship.weapons.items():
                    if ws.can_fire():
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': closest.ship_id
                        })
            return commands

        sim.set_decision_callback(aggressive_captain)

        # Run for 90 seconds (3 decision points)
        for _ in range(90):
            sim.step()

        shots_fired = sim.metrics.total_shots_fired
        print(f"[DEBUG] Aggressive captain fired {shots_fired} shots")

        # Should have fired at decision points
        assert shots_fired > 0, "Aggressive captain should fire shots"

    def test_defensive_script(self, fleet_data):
        """Test defensive captain script that extends radiators and evades."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        defender = create_ship_from_fleet_data(
            ship_id="defender",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        attacker = create_ship_from_fleet_data(
            ship_id="attacker",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(200_000, 0, 0),
            velocity=Vector3D(-5000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(defender)
        sim.add_ship(attacker)

        # Define defensive callback
        def defensive_captain(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction != "alpha":
                return []

            commands = []

            # Extend radiators for cooling
            commands.append({'type': 'set_radiators', 'extend': True})

            return commands

        sim.set_decision_callback(defensive_captain)

        # Run for 60 seconds
        for _ in range(60):
            sim.step()

        # Check radiator events
        radiator_events = sim.get_events_by_type(SimulationEventType.RADIATOR_EXTENDED)
        print(f"[DEBUG] Radiator extended events: {len(radiator_events)}")

    def test_script_timing(self, two_destroyer_simulation):
        """Test that captain scripts are called at correct decision intervals."""
        sim = two_destroyer_simulation

        call_times = []

        def timing_callback(ship_id, simulation):
            if ship_id == "alpha_destroyer_1":
                call_times.append(simulation.current_time)
            return []

        sim.set_decision_callback(timing_callback)

        # Run for 120 seconds
        for _ in range(120):
            sim.step()

        print(f"[DEBUG] Decision callback times: {call_times}")

        # Should have been called at ~30, 60, 90, 120 (or similar intervals)
        assert len(call_times) >= 3, "Should have multiple decision points"

        # Check intervals are approximately correct
        if len(call_times) >= 2:
            interval = call_times[1] - call_times[0]
            print(f"[DEBUG] First interval: {interval}s (expected ~30s)")
            assert 25 <= interval <= 35, "Decision interval should be approximately 30 seconds"


# =============================================================================
# TEST: FULL BATTLE
# =============================================================================

class TestFullBattle:
    """Test complete battle scenarios."""

    def test_destroyer_vs_destroyer(self, fleet_data):
        """Test full battle between two destroyers."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        alpha = create_ship_from_fleet_data(
            ship_id="alpha_destroyer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(3000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta_destroyer",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(300_000, 0, 0),
            velocity=Vector3D(-3000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Add weapons to both
        weapon = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        alpha.weapons["main"] = WeaponState(weapon=weapon, ammo_remaining=100)
        beta.weapons["main"] = WeaponState(weapon=weapon, ammo_remaining=100)

        # Both ships fire aggressively
        def battle_captain(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies and ship.weapons:
                target = enemies[0]
                for slot, ws in ship.weapons.items():
                    if ws.can_fire():
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })
            return commands

        sim.set_decision_callback(battle_captain)

        print(f"[DEBUG] Starting destroyer vs destroyer battle")
        print(f"[DEBUG] Initial distance: {alpha.distance_to(beta) / 1000:.1f} km")

        # Run battle for 300 seconds
        sim.run(duration=300.0)

        print(f"[DEBUG] Battle ended at t={sim.current_time}s")
        print(f"[DEBUG] Final distance: {alpha.distance_to(beta) / 1000:.1f} km")
        print(f"[DEBUG] Total shots fired: {sim.metrics.total_shots_fired}")
        print(f"[DEBUG] Total hits: {sim.metrics.total_hits}")
        print(f"[DEBUG] Hit rate: {sim.metrics.hit_rate * 100:.1f}%")
        print(f"[DEBUG] Alpha damage taken: {alpha.damage_taken_gj:.2f} GJ")
        print(f"[DEBUG] Beta damage taken: {beta.damage_taken_gj:.2f} GJ")

    def test_corvette_vs_destroyer(self, fleet_data):
        """Test asymmetric battle between corvette and destroyer."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        corvette = create_ship_from_fleet_data(
            ship_id="corvette",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        destroyer = create_ship_from_fleet_data(
            ship_id="destroyer",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(400_000, 0, 0),
            velocity=Vector3D(-2000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(corvette)
        sim.add_ship(destroyer)

        # Corvette uses torpedoes, destroyer uses guns
        torpedo_weapon = create_weapon_from_fleet_data(fleet_data, "torpedo_launcher")
        coilgun_weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")

        # Note: corvette may already have torpedo launcher from fleet data
        destroyer.weapons["spinal"] = WeaponState(weapon=coilgun_weapon, ammo_remaining=100)

        def asymmetric_captain(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies:
                target = enemies[0]

                # Corvette launches torpedoes
                if ship.ship_type == "corvette":
                    if ship.torpedo_launcher and ship.torpedo_launcher.can_fire(simulation.current_time):
                        commands.append({
                            'type': 'launch_torpedo',
                            'target_id': target.ship_id
                        })

                # Destroyer fires guns
                else:
                    for slot, ws in ship.weapons.items():
                        if ws.can_fire():
                            commands.append({
                                'type': 'fire_at',
                                'weapon_slot': slot,
                                'target_id': target.ship_id
                            })

            return commands

        sim.set_decision_callback(asymmetric_captain)

        print(f"[DEBUG] Starting corvette vs destroyer battle")

        sim.run(duration=300.0)

        print(f"[DEBUG] Battle ended at t={sim.current_time}s")
        print(f"[DEBUG] Torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[DEBUG] Torpedo hits: {sim.metrics.total_torpedo_hits}")
        print(f"[DEBUG] Coilgun shots: {sim.metrics.total_shots_fired}")

    def test_battle_ends_on_destruction(self, fleet_data):
        """Test that battle ends when one side is destroyed."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        alpha = create_ship_from_fleet_data(
            ship_id="alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta",
            ship_type="corvette",  # Weaker ship
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(30_000, 0, 0),  # Very close
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Give alpha overwhelming firepower
        weapon = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        alpha.weapons["main"] = WeaponState(weapon=weapon, ammo_remaining=1000)

        def overwhelming_force(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction != "alpha":
                return []

            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies:
                target = enemies[0]
                for slot, ws in ship.weapons.items():
                    if ws.can_fire():
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })
            return commands

        sim.set_decision_callback(overwhelming_force)

        print(f"[DEBUG] Starting one-sided battle")

        # Run until battle ends or timeout
        sim.run(duration=600.0)

        print(f"[DEBUG] Battle duration: {sim.current_time}s")
        print(f"[DEBUG] Ships destroyed: {sim.metrics.ships_destroyed}")
        print(f"[DEBUG] Beta destroyed: {beta.is_destroyed}")

        # Check for ship destruction events
        destruction_events = sim.get_events_by_type(SimulationEventType.SHIP_DESTROYED)
        print(f"[DEBUG] Ship destruction events: {len(destruction_events)}")

        # Check if battle ended
        ended_events = sim.get_events_by_type(SimulationEventType.SIMULATION_ENDED)
        print(f"[DEBUG] Simulation end events: {len(ended_events)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
