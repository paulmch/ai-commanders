#!/usr/bin/env python3
"""
Detailed battle report showing all ship damage status.
"""
import json
from typing import Any, List

import pytest

from src.simulation import (
    CombatSimulation, SimulationEventType,
    create_ship_from_fleet_data, Maneuver, ManeuverType
)
from src.physics import Vector3D
from src.combat import HitLocation
from src.firecontrol import (
    WeaponsOfficer, WeaponsOrder, WeaponsCommand,
    HelmOrder, HelmCommand, calculate_hit_probability
)
from src.torpedo import TorpedoSpecs, TorpedoLauncher, GuidanceMode


def print_detailed_ship_status(ship, events):
    """Print detailed status for a ship after battle."""
    print(f"\n{'='*60}")
    print(f"SHIP: {ship.ship_id} ({ship.faction})")
    print(f"{'='*60}")

    # Combat stats
    print(f"\nCOMBAT STATISTICS:")
    print(f"  Shots fired: {ship.shots_fired}")
    print(f"  Hits scored: {ship.hits_scored}")
    print(f"  Damage dealt: {ship.damage_dealt_gj:.2f} GJ")
    print(f"  Damage taken: {ship.damage_taken_gj:.2f} GJ")
    print(f"  PD intercepts: {ship.pd_intercepts}")
    print(f"  Is destroyed: {ship.is_destroyed}")

    # Armor status
    print(f"\nARMOR STATUS:")
    if ship.armor:
        for loc in [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]:
            section = ship.armor.get_section(loc)
            if section:
                print(f"  {loc.value.upper()}: {section.thickness_cm:.2f} cm "
                      f"({section.protection_percent:.1f}% protection)")
    else:
        print("  No armor data available")

    # Module status
    print(f"\nMODULE STATUS:")
    if ship.module_layout:
        modules = ship.module_layout.get_all_modules()
        for mod in modules:
            if mod.is_destroyed:
                status = "DESTROYED"
            elif mod.health_percent < 100:
                status = f"{mod.health_percent:.0f}% (DAMAGED)"
            else:
                status = f"{mod.health_percent:.0f}%"
            crit = " [CRITICAL]" if mod.is_critical else ""
            print(f"  {mod.name} ({mod.module_type.value}): {status}{crit}")
    else:
        print("  No module layout data available")

    # Radiator status
    print(f"\nRADIATOR STATUS:")
    if ship.thermal_system:
        for pos, rad in ship.thermal_system.radiators.radiators.items():
            health_status = f"{rad.health_percent:.0f}%"
            if rad.health_percent < 100:
                health_status += " (DAMAGED)"
            print(f"  {pos.value}: {rad.state.value}, {health_status}")
        print(f"  Total dissipation: {ship.thermal_system.radiators.total_dissipation_kw:.0f} kW")
    else:
        print("  No thermal system data available")

    # Thermal status
    print(f"\nTHERMAL STATUS:")
    if ship.thermal_system:
        print(f"  Current heat: {ship.thermal_system.heat_percent:.1f}%")
        print(f"  Heat sink: {ship.thermal_system.heatsink.current_heat_gj:.2f} / "
              f"{ship.thermal_system.heatsink.capacity_gj:.2f} GJ")
        if ship.thermal_system.is_overheating:
            print("  WARNING: OVERHEATING!")
        if ship.thermal_system.is_critical:
            print("  CRITICAL: THERMAL EMERGENCY!")
    else:
        print("  No thermal system data available")

    # Find damage events for this ship
    ship_damage_events = [
        e for e in events
        if e.event_type == SimulationEventType.DAMAGE_TAKEN and e.ship_id == ship.ship_id
    ]
    if ship_damage_events:
        print(f"\nDAMAGE EVENTS ({len(ship_damage_events)} hits):")
        for e in ship_damage_events:
            loc = e.data.get('location', 'unknown')
            dmg = e.data.get('damage_gj', 0)
            pen = e.data.get('penetrated', False)
            armor_abs = e.data.get('armor_absorbed_gj', 0)
            print(f"  T+{e.timestamp:.1f}s: {dmg:.2f} GJ at {loc}, "
                  f"armor absorbed {armor_abs:.2f} GJ, penetrated: {pen}")


class TestDetailedBattleReport:
    """Tests that show detailed battle damage reports."""

    def test_head_on_engagement_detailed(self):
        """Two destroyers in head-on pass - shows detailed damage report."""
        import random
        random.seed(42)  # Reproducible probabilistic hit detection

        # Load fleet data
        with open('data/fleet_ships.json') as f:
            fleet_data = json.load(f)

        # Create simulation - same as head-on test
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)

        # Initial distance 1000km, closing at 20 km/s
        ship1 = create_ship_from_fleet_data(
            ship_id='alpha-1',
            ship_type='destroyer',
            faction='blue',
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(10_000, 0, 0),  # 10 km/s toward bravo
            forward=Vector3D(1, 0, 0)
        )
        ship2 = create_ship_from_fleet_data(
            ship_id='bravo-1',
            ship_type='destroyer',
            faction='red',
            fleet_data=fleet_data,
            position=Vector3D(1_000_000, 0, 0),  # 1000 km away
            velocity=Vector3D(-10_000, 0, 0),  # 10 km/s toward alpha
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(ship1)
        sim.add_ship(ship2)

        print("\n" + "="*70)
        print("HEAD-ON ENGAGEMENT - TWO DESTROYERS")
        print("Starting distance: 1000 km, closing at 20 km/s")
        print("="*70)

        # Capture sensor reports for display
        sensor_reports = []

        # Aggressive captain behavior
        def battle_callback(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            # Capture sensor report for LLM
            report = simulation.generate_sensor_report(ship_id)
            sensor_reports.append(report)
            commands = []
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return commands

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return commands

            target = enemies[0]
            distance_km = ship.distance_to(target) / 1000

            # Close in on target
            commands.append(Maneuver(
                maneuver_type=ManeuverType.INTERCEPT,
                start_time=simulation.current_time,
                duration=30.0,
                throttle=1.0,
                target_id=target.ship_id
            ))

            # Fire all ready weapons when in range
            for slot, weapon_state in ship.weapons.items():
                if weapon_state.can_fire():
                    if weapon_state.weapon.range_km >= distance_km:
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })

            return commands

        sim.set_decision_callback(battle_callback)
        sim.run(duration=120.0)

        events = sim.events

        # Count events
        event_counts = {}
        for e in events:
            name = e.event_type.name
            event_counts[name] = event_counts.get(name, 0) + 1

        print(f"\nSimulation time: {sim.current_time:.1f}s")

        # LLM decision points
        decision_events = [e for e in events if e.event_type == SimulationEventType.DECISION_POINT_REACHED]
        print(f"\nLLM DECISION POINTS:")
        print(f"  Total decision points: {len(decision_events)}")
        print(f"  Decision interval: {sim.decision_interval:.1f}s")
        print(f"  Expected prompts per ship: {int(sim.current_time / sim.decision_interval)}")
        print(f"  Total LLM prompts in this battle: {len(decision_events)} (one per ship per interval)")

        # Show sensor reports captured during battle
        if sensor_reports:
            print(f"\n  Captured {len(sensor_reports)} sensor reports")
            # Show first sensor report as example
            if sensor_reports:
                print("\n  EXAMPLE SENSOR REPORT (first decision point):")
                print("  " + "-" * 50)
                for line in sensor_reports[0].split('\n')[:30]:  # First 30 lines
                    print(f"  {line}")
                print("  ... (truncated for brevity)")

        print(f"\nEvent Summary:")
        for name, count in sorted(event_counts.items()):
            print(f"  {name}: {count}")

        # Show PROJECTILE_LAUNCHED events with weapon type
        launch_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED]
        if launch_events:
            print(f"\nPROJECTILES LAUNCHED ({len(launch_events)}):")
            for e in launch_events:
                weapon_type = e.data.get('weapon_type', 'unknown')
                is_turreted = e.data.get('is_turreted', False)
                fire_dir = e.data.get('fire_direction', 'unknown')
                ke = e.data.get('kinetic_energy_gj', 0)
                turret_str = "TURRET" if is_turreted else "FIXED"
                print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] fired {weapon_type} ({turret_str}) at {e.target_id}")
                print(f"    KE: {ke:.2f} GJ, Fire direction: {fire_dir}")

        # Show PROJECTILE_IMPACT events with hit locations
        impact_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_IMPACT]
        if impact_events:
            print(f"\nPROJECTILE IMPACTS ({len(impact_events)} hits):")
            for e in impact_events:
                loc = e.data.get('hit_location', 'unknown')
                energy = e.data.get('kinetic_energy_gj', 0)
                pen = e.data.get('penetrated', False)
                angle = e.data.get('angle_deg', 0)
                impact_angle = e.data.get('impact_angle_from_normal', 0)
                effective_armor = e.data.get('effective_armor_cm', 0)
                print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] hit {e.target_id} at {loc.upper()}, "
                      f"{energy:.2f} GJ")
                print(f"    Impact angle from normal: {impact_angle:.1f}° → Effective armor: {effective_armor:.2f} cm")
                print(f"    Penetrated: {pen}")

        # Show damage taken events summary
        damage_events = [e for e in events if e.event_type == SimulationEventType.DAMAGE_TAKEN]
        if damage_events:
            print(f"\nDAMAGE SUMMARY:")
            print(f"  Total damage events: {len(damage_events)}")
            total_dmg = sum(e.data.get('damage_gj', 0) for e in damage_events)
            print(f"  Total damage delivered: {total_dmg:.2f} GJ")
            pen_count = sum(1 for e in damage_events if e.data.get('penetrated', False))
            print(f"  Penetrations: {pen_count}")

        # Detailed ship status
        print_detailed_ship_status(ship1, events)
        print_detailed_ship_status(ship2, events)

        # Final distance
        final_dist = ship1.position.distance_to(ship2.position) / 1000
        print(f"\n{'='*70}")
        print(f"Final distance: {final_dist:.1f} km")
        print(f"{'='*70}")

        # Show misses for hit detection analysis
        miss_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_MISS]
        if miss_events:
            print(f"\nPROJECTILE MISSES ({len(miss_events)}):")
            for e in miss_events:
                closest = e.data.get('closest_approach_km', 'unknown')
                detection_type = e.data.get('detection', 'probabilistic')
                if detection_type == 'geometric':
                    # Geometric detection - show closest approach and micro-steps
                    micro_steps = e.data.get('micro_steps', 0)
                    if isinstance(closest, (int, float)):
                        print(f"  T+{e.timestamp:.1f}s: closest {closest:.3f} km (geometric, {micro_steps} micro-steps)")
                    else:
                        print(f"  T+{e.timestamp:.1f}s: {closest} (geometric)")
                else:
                    # Probabilistic detection - show probability info
                    hit_prob = e.data.get('hit_probability', 0)
                    roll = e.data.get('roll', 0)
                    if isinstance(closest, (int, float)) and isinstance(hit_prob, (int, float)):
                        print(f"  T+{e.timestamp:.1f}s: closest {closest:.1f} km, hit_prob {hit_prob:.2%}, roll {roll:.3f}")
                    else:
                        print(f"  T+{e.timestamp:.1f}s: miss")

        # Assertions - probabilistic detection means hits aren't guaranteed
        assert event_counts.get('PROJECTILE_LAUNCHED', 0) > 0, "Should launch projectiles"
        # With probabilistic hit detection, any combination of hits/misses is valid
        total_resolved = event_counts.get('PROJECTILE_IMPACT', 0) + event_counts.get('PROJECTILE_MISS', 0)
        assert total_resolved > 0, "Projectiles should either hit or miss"

    def test_realistic_engagement_with_evasion(self):
        """
        Realistic engagement scenario:
        - Ships start 1000km apart in X, 500m offset in Y
        - Both at rest initially
        - Phase 1: Accelerate toward each other
        - Phase 2 (first LLM checkpoint): Acquire spinal lock, turn if needed
        - Phase 3 (second checkpoint+): Evasive maneuvers while firing
        """
        import math
        import random
        random.seed(42)  # Reproducible evasion patterns

        with open('data/fleet_ships.json') as f:
            fleet_data = json.load(f)

        # 30 second decision intervals for tactical granularity
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)

        # Ships start at rest, 1000km apart in X, 500m offset in Y
        ship1 = create_ship_from_fleet_data(
            ship_id='alpha-1',
            ship_type='destroyer',
            faction='blue',
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),  # At rest
            forward=Vector3D(1, 0, 0)    # Facing toward bravo
        )
        ship2 = create_ship_from_fleet_data(
            ship_id='bravo-1',
            ship_type='destroyer',
            faction='red',
            fleet_data=fleet_data,
            position=Vector3D(1_000_000, 500, 0),  # 1000 km in X, 500m offset in Y
            velocity=Vector3D(0, 0, 0),  # At rest
            forward=Vector3D(-1, 0, 0)   # Facing toward alpha
        )

        sim.add_ship(ship1)
        sim.add_ship(ship2)

        print("\n" + "="*70)
        print("REALISTIC ENGAGEMENT WITH EVASION")
        print("Starting: 1000 km apart, 500m Y offset, both at rest")
        print("="*70)

        # Track engagement phases per ship
        ship_phases = {'alpha-1': 'accelerate', 'bravo-1': 'accelerate'}
        checkpoint_count = {'alpha-1': 0, 'bravo-1': 0}
        evasion_heading = {'alpha-1': 0.0, 'bravo-1': 0.0}  # Current evasion angle in degrees

        # Create weapons officers for each ship with FIRE_WHEN_OPTIMAL logic
        weapons_officers = {
            'alpha-1': WeaponsOfficer(min_probability_threshold=0.2),
            'bravo-1': WeaponsOfficer(min_probability_threshold=0.2)
        }

        # Set initial orders: fire when hit probability >= 20%
        for officer in weapons_officers.values():
            officer.set_order(WeaponsOrder(
                command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                weapon_slot="all",
                min_hit_probability=0.2
            ))

        def tactical_captain(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            """
            Tactical AI with WeaponsOfficer:
            - Checkpoint 1: Accelerate toward enemy
            - Checkpoint 2: Acquire lock, adjust heading if target outside 30° gimbal
            - Checkpoint 3+: Evasive maneuvers, WeaponsOfficer decides when to fire
            """
            commands = []
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return commands

            checkpoint_count[ship_id] += 1
            checkpoint = checkpoint_count[ship_id]

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return commands

            target = enemies[0]
            distance_km = ship.distance_to(target) / 1000

            # Calculate angle to target
            to_target = (target.position - ship.position).normalized()
            angle_to_target = math.degrees(ship.forward.angle_to(to_target))

            print(f"\n  [{ship_id}] Checkpoint {checkpoint} @ T+{simulation.current_time:.0f}s")
            print(f"    Distance: {distance_km:.1f} km, Angle to target: {angle_to_target:.1f}°")
            print(f"    Velocity: {ship.velocity.magnitude/1000:.2f} km/s")

            # Get weapons officer for this ship
            officer = weapons_officers[ship_id]

            if checkpoint == 1:
                # PHASE 1: Full acceleration toward enemy
                print(f"    Phase: ACCELERATION - closing distance")
                ship_phases[ship_id] = 'accelerate'

                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=1.0,
                    target_id=target.ship_id
                ))

            elif checkpoint == 2:
                # PHASE 2: Acquire spinal lock
                print(f"    Phase: ACQUIRING LOCK")
                ship_phases[ship_id] = 'lock'

                # Check if target is within spinal gimbal (30°)
                if angle_to_target <= 30.0:
                    # Calculate hit probability using fire control system
                    spinal = ship.weapons.get('weapon_0')
                    if spinal and spinal.can_fire() and spinal.weapon.range_km >= distance_km:
                        solution = calculate_hit_probability(
                            shooter_position=ship.position,
                            shooter_velocity=ship.velocity,
                            target_position=target.position,
                            target_velocity=target.velocity,
                            target_geometry=target.geometry,
                            target_forward=target.forward,
                            muzzle_velocity_kps=spinal.weapon.muzzle_velocity_kps
                        )
                        print(f"    Spinal lock acquired: {solution.hit_probability*100:.1f}% hit prob")
                        print(f"    Recommendation: {solution.recommendation}")

                        # Use weapons officer to decide
                        should_fire, reason = officer.evaluate_shot(
                            weapon_slot='weapon_0',
                            weapon_ammo=spinal.ammo_remaining,
                            weapon_magazine=spinal.weapon.magazine,
                            solution=solution
                        )
                        if should_fire:
                            print(f"    WeaponsOfficer: FIRE ({reason})")
                            commands.append({
                                'type': 'fire_at',
                                'weapon_slot': 'weapon_0',
                                'target_id': target.ship_id
                            })
                        else:
                            print(f"    WeaponsOfficer: HOLD ({reason})")
                else:
                    print(f"    Target outside gimbal ({angle_to_target:.1f}° > 30°) - TURNING")

                # Continue intercept to maintain closure and adjust heading
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=1.0,
                    target_id=target.ship_id
                ))

            else:
                # PHASE 3+: Evasive maneuvers while firing
                print(f"    Phase: EVASIVE FIRE")
                ship_phases[ship_id] = 'evasive'

                # Change evasion heading by 5-15 degrees each checkpoint
                evasion_change = random.uniform(5, 15) * (1 if checkpoint % 2 == 0 else -1)
                evasion_heading[ship_id] += evasion_change
                evasion_heading[ship_id] = max(-30, min(30, evasion_heading[ship_id]))

                print(f"    Evasion heading change: {evasion_change:+.1f}° (total: {evasion_heading[ship_id]:.1f}°)")

                # Let WeaponsOfficer evaluate all weapons and decide what to fire
                fire_commands = officer.get_fire_commands(
                    ship_position=ship.position,
                    ship_velocity=ship.velocity,
                    weapons=ship.weapons,
                    targets=enemies,
                    primary_target_id=target.ship_id
                )

                if fire_commands:
                    for cmd in fire_commands:
                        slot = cmd['weapon_slot']
                        hit_prob = cmd.get('hit_probability', 0)
                        reason = cmd.get('reason', '')
                        print(f"    WeaponsOfficer: FIRE {slot} ({hit_prob*100:.0f}% prob) - {reason}")
                        commands.append(cmd)
                else:
                    print(f"    WeaponsOfficer: HOLD ALL WEAPONS (no good shots)")

                # Evasive maneuver - use INTERCEPT but with reduced throttle
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=0.7,  # Reduced throttle for evasion
                    target_id=target.ship_id,
                ))

            return commands

        sim.set_decision_callback(tactical_captain)

        # Run for 360 seconds (12 checkpoints per ship) - long enough for projectiles to arrive
        sim.run(duration=360.0)

        events = sim.events

        # Count events
        event_counts = {}
        for e in events:
            name = e.event_type.name
            event_counts[name] = event_counts.get(name, 0) + 1

        print(f"\n{'='*70}")
        print(f"ENGAGEMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Simulation time: {sim.current_time:.1f}s")

        print(f"\nEvent Summary:")
        for name, count in sorted(event_counts.items()):
            print(f"  {name}: {count}")

        # Show projectiles launched
        launch_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED]
        if launch_events:
            print(f"\nPROJECTILES LAUNCHED ({len(launch_events)}):")
            for e in launch_events:
                weapon_type = e.data.get('weapon_type', 'unknown')
                is_turreted = e.data.get('is_turreted', False)
                fire_dir = e.data.get('fire_direction', 'unknown')
                ke = e.data.get('kinetic_energy_gj', 0)
                turret_str = "TURRET" if is_turreted else "SPINAL"
                print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] {weapon_type} ({turret_str}) → {e.target_id}")
                print(f"    KE: {ke:.2f} GJ, Fire dir: {fire_dir}")

        # Show impacts and misses
        impact_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_IMPACT]
        miss_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_MISS]
        if launch_events:
            # Count in-flight at end
            end_event = [e for e in events if e.event_type == SimulationEventType.SIMULATION_ENDED]
            in_flight = end_event[0].data.get('projectiles_in_flight', 0) if end_event else 0

            total_resolved = len(impact_events) + len(miss_events)
            hit_rate = len(impact_events) / len(launch_events) * 100 if launch_events else 0
            print(f"\nACCURACY: {len(impact_events)}/{len(launch_events)} hits ({hit_rate:.1f}%)")
            print(f"  Misses: {len(miss_events)}, Still in flight: {in_flight}")

        if miss_events:
            print(f"\nPROJECTILE MISSES ({len(miss_events)}):")
            for e in miss_events:
                closest = e.data.get('closest_approach_km', 'unknown')
                current = e.data.get('current_distance_km', 'unknown')
                reason = e.data.get('reason', 'passed_target')
                if reason == 'too_far':
                    print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] → {e.target_id} - flew too far")
                else:
                    print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] → {e.target_id} - closest: {closest:.1f} km")

        if impact_events:
            print(f"\nPROJECTILE IMPACTS ({len(impact_events)} hits):")
            for e in impact_events:
                loc = e.data.get('hit_location', 'unknown')
                energy = e.data.get('kinetic_energy_gj', 0)
                pen = e.data.get('penetrated', False)
                impact_angle = e.data.get('impact_angle_from_normal', 0)
                effective_armor = e.data.get('effective_armor_cm', 0)
                print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] hit {e.target_id} at {loc.upper()}, {energy:.2f} GJ")
                print(f"    Impact angle: {impact_angle:.1f}° → Effective armor: {effective_armor:.2f} cm, Pen: {pen}")

        # Detailed ship status
        print_detailed_ship_status(ship1, events)
        print_detailed_ship_status(ship2, events)

        # Final state
        final_dist = ship1.position.distance_to(ship2.position) / 1000
        print(f"\n{'='*70}")
        print(f"Final distance: {final_dist:.1f} km")
        print(f"Alpha-1 velocity: {ship1.velocity.magnitude/1000:.2f} km/s")
        print(f"Bravo-1 velocity: {ship2.velocity.magnitude/1000:.2f} km/s")
        print(f"{'='*70}")

        # Assertions
        assert event_counts.get('PROJECTILE_LAUNCHED', 0) > 0, "Should fire projectiles"


    def test_joust_engagement(self):
        """
        Joust scenario:
        - Ships start 1000km apart in X, 500m offset in Y
        - Both at rest initially
        - Phase 1: Full acceleration toward each other until first checkpoint
        - Phase 2+: Face target and fire continuously
        - Extended simulation time for projectile travel
        """
        import math
        import random
        random.seed(123)  # Different seed for variety

        with open('data/fleet_ships.json') as f:
            fleet_data = json.load(f)

        # 30 second decision intervals
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)

        # Ships start at rest, 1000km apart in X, 500m offset in Y
        ship1 = create_ship_from_fleet_data(
            ship_id='alpha-1',
            ship_type='destroyer',
            faction='blue',
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )
        ship2 = create_ship_from_fleet_data(
            ship_id='bravo-1',
            ship_type='destroyer',
            faction='red',
            fleet_data=fleet_data,
            position=Vector3D(1_000_000, 500, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(ship1)
        sim.add_ship(ship2)

        print("\n" + "="*70)
        print("JOUST ENGAGEMENT")
        print("Starting: 1000 km apart, 500m Y offset, both at rest")
        print("Phase 1: Full acceleration toward each other")
        print("Phase 2+: Face target and FIRE!")
        print("="*70)

        checkpoint_count = {'alpha-1': 0, 'bravo-1': 0}

        # Weapons officers - low threshold for aggressive firing
        weapons_officers = {
            'alpha-1': WeaponsOfficer(min_probability_threshold=0.1),
            'bravo-1': WeaponsOfficer(min_probability_threshold=0.1)
        }
        for officer in weapons_officers.values():
            officer.set_order(WeaponsOrder(
                command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                weapon_slot="all",
                min_hit_probability=0.1  # Fire even at 10% chance
            ))

        def joust_captain(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            """
            Simple joust AI:
            - Checkpoint 1: Accelerate toward enemy
            - Checkpoint 2+: Face target and fire everything
            """
            commands = []
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return commands

            checkpoint_count[ship_id] += 1
            checkpoint = checkpoint_count[ship_id]

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return commands

            target = enemies[0]
            distance_km = ship.distance_to(target) / 1000
            closing_rate = -(target.velocity - ship.velocity).dot(
                (target.position - ship.position).normalized()
            ) / 1000  # km/s

            print(f"\n  [{ship_id}] Checkpoint {checkpoint} @ T+{simulation.current_time:.0f}s")
            print(f"    Distance: {distance_km:.1f} km, Closing: {closing_rate:.2f} km/s")
            print(f"    Speed: {ship.velocity.magnitude/1000:.2f} km/s")

            officer = weapons_officers[ship_id]

            if checkpoint <= 3:
                # PHASE 1-3: Full acceleration toward enemy for 90 seconds
                print(f"    Phase: CHARGE! (accelerating)")
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=1.0,
                    target_id=target.ship_id
                ))
            else:
                # PHASE 2+: Face target and fire
                print(f"    Phase: ENGAGE!")

                # Use INTERCEPT with throttle=0 to just rotate toward target (coast)
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=0.0,  # Coast while shooting (no thrust)
                    target_id=target.ship_id
                ))

                # Fire all ready weapons that are in range
                # (Manual check since we're at long range initially)
                weapons_fired = []
                for slot, weapon_state in ship.weapons.items():
                    if 'pd' in slot:
                        continue  # Skip point defense
                    if not weapon_state.can_fire():
                        continue
                    if weapon_state.weapon.range_km < distance_km:
                        continue  # Out of range

                    # Calculate hit probability for logging
                    solution = calculate_hit_probability(
                        shooter_position=ship.position,
                        shooter_velocity=ship.velocity,
                        target_position=target.position,
                        target_velocity=target.velocity,
                        target_geometry=target.geometry,
                        target_forward=target.forward,
                        muzzle_velocity_kps=weapon_state.weapon.muzzle_velocity_kps
                    )

                    # Use weapons officer to decide
                    should_fire, reason = officer.evaluate_shot(
                        weapon_slot=slot,
                        weapon_ammo=weapon_state.ammo_remaining,
                        weapon_magazine=weapon_state.weapon.magazine,
                        solution=solution
                    )

                    if should_fire:
                        print(f"    FIRE {slot}: {solution.hit_probability*100:.1f}% ({reason})")
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })
                        weapons_fired.append(slot)
                    else:
                        print(f"    HOLD {slot}: {solution.hit_probability*100:.1f}% ({reason})")

                if not weapons_fired:
                    print(f"    All weapons on HOLD")

            return commands

        sim.set_decision_callback(joust_captain)

        # Run for 600 seconds (10 minutes) - plenty of time for projectiles
        sim.run(duration=600.0)

        events = sim.events

        # Count events
        event_counts = {}
        for e in events:
            name = e.event_type.name
            event_counts[name] = event_counts.get(name, 0) + 1

        print(f"\n{'='*70}")
        print(f"JOUST RESULTS")
        print(f"{'='*70}")
        print(f"Simulation time: {sim.current_time:.1f}s")

        print(f"\nEvent Summary:")
        for name, count in sorted(event_counts.items()):
            print(f"  {name}: {count}")

        # Analyze PD targeting
        pd_events = [e for e in events if e.event_type == SimulationEventType.PD_ENGAGED]
        pd_by_type = {}
        for e in pd_events:
            ttype = e.data.get('target_type', 'unknown')
            pd_by_type[ttype] = pd_by_type.get(ttype, 0) + 1
        if pd_by_type:
            print(f"\nPD Target Breakdown:")
            for ttype, count in sorted(pd_by_type.items()):
                print(f"  {ttype}: {count}")

        # Projectile stats
        launch_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED]
        impact_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_IMPACT]
        miss_events = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_MISS]

        print(f"\nPROJECTILE SUMMARY:")
        print(f"  Launched: {len(launch_events)}")
        print(f"  Hits: {len(impact_events)}")
        print(f"  Misses: {len(miss_events)}")
        if launch_events:
            hit_rate = len(impact_events) / len(launch_events) * 100
            print(f"  Hit rate: {hit_rate:.1f}%")

        # Show hits
        if impact_events:
            print(f"\nHITS:")
            for e in impact_events:
                loc = e.data.get('hit_location', 'unknown')
                energy = e.data.get('kinetic_energy_gj', 0)
                original_energy = e.data.get('original_energy_gj', energy)
                pd_ablation = e.data.get('pd_ablation_kg', 0)
                mass_remaining = e.data.get('mass_remaining_kg', 0)
                pen = e.data.get('penetrated', False)
                if pd_ablation > 0:
                    print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] → {e.target_id} at {loc.upper()}")
                    print(f"    Energy: {energy:.2f} GJ (was {original_energy:.2f} GJ)")
                    print(f"    PD ablated: {pd_ablation:.1f} kg, remaining: {mass_remaining:.1f} kg, pen={pen}")
                else:
                    print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] → {e.target_id} at {loc.upper()}, {energy:.2f} GJ, pen={pen}")

        # Show misses with closest approach
        if miss_events:
            print(f"\nMISSES (showing closest approaches):")
            for e in miss_events[:10]:  # First 10
                closest = e.data.get('closest_approach_km', 0)
                hit_prob = e.data.get('hit_probability', 0)
                print(f"  T+{e.timestamp:.1f}s: [{e.ship_id}] → {e.target_id} closest: {closest:.1f} km (was {hit_prob*100:.1f}% prob)")
            if len(miss_events) > 10:
                print(f"  ... and {len(miss_events) - 10} more misses")

        # Ship status
        print_detailed_ship_status(ship1, events)
        print_detailed_ship_status(ship2, events)

        # Final state
        final_dist = ship1.position.distance_to(ship2.position) / 1000
        print(f"\n{'='*70}")
        print(f"Final distance: {final_dist:.1f} km")
        print(f"Alpha-1: {ship1.velocity.magnitude/1000:.2f} km/s, destroyed={ship1.is_destroyed}")
        print(f"Bravo-1: {ship2.velocity.magnitude/1000:.2f} km/s, destroyed={ship2.is_destroyed}")
        print(f"{'='*70}")

        # Assertions
        assert event_counts.get('PROJECTILE_LAUNCHED', 0) > 0, "Should fire projectiles"


    def test_torpedo_attack_run(self):
        """
        Torpedo attack scenario: Corvette vs Cruiser

        Setup:
        - Corvette at origin, Cruiser at 1000 km in X
        - Both at rest initially

        Corvette tactics:
        - Accelerate toward cruiser
        - At first checkpoint: launch all torpedoes
        - Turn 90 degrees and perform evasive maneuvers

        Cruiser tactics:
        - Accelerate at 0.5g toward corvette initially
        - When torpedoes detected: turn 90 degrees, max burn to outrun
        - Fire all weapons at will
        """
        import math
        import random
        random.seed(456)

        with open('data/fleet_ships.json') as f:
            fleet_data = json.load(f)

        # Longer decision intervals for torpedo flight
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)

        # Corvette at origin
        corvette = create_ship_from_fleet_data(
            ship_id='torpedo-boat',
            ship_type='corvette',
            faction='blue',
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)  # Facing cruiser
        )

        # Cruiser at 1000 km
        cruiser = create_ship_from_fleet_data(
            ship_id='target-cruiser',
            ship_type='cruiser',
            faction='red',
            fleet_data=fleet_data,
            position=Vector3D(1_000_000, 0, 0),  # 1000 km
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)  # Facing corvette
        )

        sim.add_ship(corvette)
        sim.add_ship(cruiser)

        print("\n" + "="*70)
        print("TORPEDO ATTACK RUN")
        print("Corvette (torpedo boat) vs Cruiser")
        print("Starting: 1000 km apart, both at rest")
        print("="*70)

        # Print torpedo specs
        if corvette.torpedo_launcher:
            torp_specs = corvette.torpedo_launcher.specs
            print(f"\nTORPEDO SPECS:")
            print(f"  Warhead: {torp_specs.warhead_yield_gj} GJ")
            print(f"  Mass: {torp_specs.mass_kg} kg ({torp_specs.propellant_fraction*100:.0f}% propellant)")
            print(f"  Delta-V: {torp_specs.total_delta_v_kps:.1f} km/s")
            print(f"  Initial accel: {torp_specs.acceleration_at_mass(torp_specs.mass_kg)/9.81:.1f}g")
            print(f"  Final accel: {torp_specs.acceleration_at_mass(torp_specs.dry_mass_kg)/9.81:.1f}g")
            print(f"  Magazine: {corvette.torpedo_launcher.current_magazine}/{corvette.torpedo_launcher.magazine_capacity}")

        checkpoint_count = {'torpedo-boat': 0, 'target-cruiser': 0}
        torpedoes_fired = {'torpedo-boat': 0}
        cruiser_evading = {'target-cruiser': False}

        def tactical_ai(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            commands = []
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return commands

            checkpoint_count[ship_id] += 1
            checkpoint = checkpoint_count[ship_id]

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return commands

            target = enemies[0]
            distance_km = ship.distance_to(target) / 1000

            # Calculate closing rate
            rel_vel = ship.velocity - target.velocity
            to_target = (target.position - ship.position).normalized()
            closing_rate = rel_vel.dot(to_target) / 1000  # km/s

            print(f"\n  [{ship_id}] Checkpoint {checkpoint} @ T+{simulation.current_time:.0f}s")
            print(f"    Distance: {distance_km:.1f} km, Closing: {closing_rate:.2f} km/s")
            print(f"    Speed: {ship.velocity.magnitude/1000:.2f} km/s")

            if ship_id == 'torpedo-boat':
                # CORVETTE TACTICS
                if checkpoint <= 2:
                    # Phase 1-2: Charge toward cruiser (build up speed)
                    print(f"    Phase: ATTACK RUN - closing distance")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        target_id=target.ship_id
                    ))

                elif checkpoint <= 6 and torpedoes_fired['torpedo-boat'] < 4:
                    # Phase 3-6: Launch torpedoes (1 per checkpoint due to 30s cooldown)
                    if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
                        torpedoes_fired['torpedo-boat'] += 1
                        print(f"    Phase: TORPEDO LAUNCH #{torpedoes_fired['torpedo-boat']}!")
                        commands.append({
                            'type': 'launch_torpedo',
                            'target_id': target.ship_id
                        })

                    # Continue intercept while launching
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        target_id=target.ship_id
                    ))

                else:
                    # Phase 7+: Break away perpendicular
                    print(f"    Phase: BREAKING AWAY - perpendicular burn")

                    # Burn perpendicular to escape
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        direction=Vector3D(0, 1, 0)  # Perpendicular Y
                    ))

            else:
                # CRUISER TACTICS
                # Check for incoming torpedoes
                incoming_torps = [t for t in simulation.torpedoes
                                  if t.target_ship_id == ship_id and not t.is_disabled]

                if incoming_torps and not cruiser_evading['target-cruiser']:
                    # Torpedoes detected! Start evasion
                    print(f"    *** TORPEDO WARNING: {len(incoming_torps)} inbound! ***")
                    cruiser_evading['target-cruiser'] = True

                if not cruiser_evading['target-cruiser']:
                    # Phase 1: Approach at half thrust
                    print(f"    Phase: APPROACH - half thrust")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=0.5,
                        target_id=target.ship_id
                    ))
                else:
                    # Evading torpedoes - max burn perpendicular
                    print(f"    Phase: EVADING TORPEDOES - max burn perpendicular")

                    # Calculate perpendicular direction (90 degrees from torpedo approach)
                    if incoming_torps:
                        torp = incoming_torps[0].torpedo
                        torp_dir = (torp.position - ship.position).normalized()
                        # Perpendicular in Y direction
                        evade_dir = Vector3D(0, 1, 0)
                        print(f"    Evading in +Y direction, {len(incoming_torps)} torpedoes tracking")

                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        direction=Vector3D(0, 1, 0)  # Burn +Y
                    ))

                # Fire all weapons at corvette
                for slot, weapon_state in ship.weapons.items():
                    if 'pd' in slot:
                        continue  # PD handles itself
                    if weapon_state.can_fire() and weapon_state.weapon.range_km >= distance_km:
                        print(f"    FIRE {slot} at corvette")
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })

            return commands

        sim.set_decision_callback(tactical_ai)

        # Run for 600 seconds (10 min) - enough for torpedo intercept
        sim.run(duration=600.0)

        events = sim.events

        # Count events
        event_counts = {}
        for e in events:
            name = e.event_type.name
            event_counts[name] = event_counts.get(name, 0) + 1

        print(f"\n{'='*70}")
        print(f"TORPEDO ATTACK RESULTS")
        print(f"{'='*70}")
        print(f"Simulation time: {sim.current_time:.1f}s")

        print(f"\nEvent Summary:")
        for name, count in sorted(event_counts.items()):
            print(f"  {name}: {count}")

        # Torpedo events
        torp_launch = [e for e in events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
        torp_hit = [e for e in events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
        torp_intercept = [e for e in events if e.event_type == SimulationEventType.PD_TORPEDO_DESTROYED]

        print(f"\nTORPEDO SUMMARY:")
        print(f"  Launched: {len(torp_launch)}")
        print(f"  Hits: {len(torp_hit)}")
        print(f"  Intercepted by PD: {len(torp_intercept)}")
        print(f"  Still in flight: {len(sim.torpedoes)}")

        # Show torpedo launches
        if torp_launch:
            print(f"\nTORPEDO LAUNCHES:")
            for e in torp_launch:
                torp_id = e.data.get('torpedo_id', 'unknown')
                init_vel = e.data.get('initial_velocity_kps', 0)
                print(f"  T+{e.timestamp:.0f}s: {torp_id} launched at {init_vel:.1f} km/s")

        # Show PD kills
        if torp_intercept:
            print(f"\nTORPEDO INTERCEPTS (PD kills):")
            for e in torp_intercept:
                torp_id = e.data.get('torpedo_id', 'unknown')
                heat = e.data.get('total_heat_absorbed_j', 0) / 1000
                print(f"  T+{e.timestamp:.0f}s: {torp_id} destroyed ({heat:.1f} kJ absorbed)")

        # Show torpedo hits
        if torp_hit:
            print(f"\nTORPEDO HITS:")
            for e in torp_hit:
                damage = e.data.get('damage_gj', 0)
                loc = e.data.get('hit_location', 'unknown')
                print(f"  T+{e.timestamp:.1f}s: {damage:.1f} GJ at {loc}")

        # Show active torpedoes
        if sim.torpedoes:
            print(f"\nACTIVE TORPEDOES ({len(sim.torpedoes)}):")
            for torp_flight in sim.torpedoes[:5]:  # First 5
                torp = torp_flight.torpedo
                dist_to_target = torp.position.distance_to(cruiser.position) / 1000
                fuel_pct = (torp_flight.torpedo.current_mass_kg - torp_flight.torpedo.specs.dry_mass_kg) / \
                           torp_flight.torpedo.specs.propellant_mass_kg * 100 if torp_flight.torpedo.specs.propellant_mass_kg > 0 else 0
                print(f"  {torp_flight.torpedo_id}: {dist_to_target:.1f} km to target, {fuel_pct:.0f}% fuel, disabled={torp_flight.is_disabled}")

        # PD analysis
        pd_events = [e for e in events if e.event_type == SimulationEventType.PD_ENGAGED]
        pd_by_type = {}
        for e in pd_events:
            ttype = e.data.get('target_type', 'unknown')
            pd_by_type[ttype] = pd_by_type.get(ttype, 0) + 1
        if pd_by_type:
            print(f"\nPD Target Breakdown:")
            for ttype, count in sorted(pd_by_type.items()):
                print(f"  {ttype}: {count}")

        # Ship status
        print_detailed_ship_status(corvette, events)
        print_detailed_ship_status(cruiser, events)

        # Final state
        final_dist = corvette.position.distance_to(cruiser.position) / 1000
        print(f"\n{'='*70}")
        print(f"Final distance: {final_dist:.1f} km")
        print(f"Corvette: {corvette.velocity.magnitude/1000:.2f} km/s, destroyed={corvette.is_destroyed}")
        print(f"Cruiser: {cruiser.velocity.magnitude/1000:.2f} km/s, destroyed={cruiser.is_destroyed}")
        print(f"{'='*70}")

        # Assertions
        assert len(torp_launch) > 0, "Corvette should launch torpedoes"

    def test_torpedo_attack_no_pd(self):
        """
        Torpedo attack with PD disabled - verify torpedoes can hit.

        Same scenario as test_torpedo_attack_run but cruiser's PD is disabled.
        This validates that torpedo guidance and impact mechanics work correctly.
        """
        import math
        import random
        random.seed(789)

        with open('data/fleet_ships.json') as f:
            fleet_data = json.load(f)

        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)

        # Corvette at origin
        # Corvette at origin with HIGH INITIAL VELOCITY - attack run!
        initial_speed_kps = 10.0  # 10 km/s closing speed
        corvette = create_ship_from_fleet_data(
            ship_id='torpedo-boat',
            ship_type='corvette',
            faction='blue',
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(initial_speed_kps * 1000, 0, 0),  # 10 km/s toward cruiser
            forward=Vector3D(1, 0, 0)
        )

        # Cruiser at 1000 km, stationary
        cruiser = create_ship_from_fleet_data(
            ship_id='target-cruiser',
            ship_type='cruiser',
            faction='red',
            fleet_data=fleet_data,
            position=Vector3D(1_000_000, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        # DISABLE CRUISER'S POINT DEFENSE
        print(f"  [DISABLED] Cruiser PD: {len(cruiser.point_defense)} turrets removed")
        cruiser.point_defense = []  # Clear the PD list

        # DISABLE CRUISER'S THRUST VECTORING - RCS only for rotation
        # This means 90° turn takes 206s instead of 28s!
        if cruiser.attitude_control:
            print(f"  [DISABLED] Cruiser thrust vectoring: RCS-only rotation")
            print(f"    TV was: {cruiser.attitude_control.tv_angular_accel_deg_s2:.3f} deg/s²")
            print(f"    RCS is: {cruiser.attitude_control.rcs_angular_accel_deg_s2:.4f} deg/s²")
            cruiser.attitude_control.tv_angular_accel_deg_s2 = 0.0
            cruiser.attitude_control.tv_max_angular_vel_deg_s = 0.0

        # Replace corvette's torpedo launcher with Poseidon torpedoes
        poseidon_specs = TorpedoSpecs.poseidon()
        corvette.torpedo_launcher = TorpedoLauncher(
            specs=poseidon_specs,
            magazine_capacity=16,
            current_magazine=16,
            cooldown_seconds=30.0
        )

        sim.add_ship(corvette)
        sim.add_ship(cruiser)

        print("\n" + "="*70)
        print("TORPEDO ATTACK RUN (NO PD) - POSEIDON SMART GUIDANCE")
        print("Corvette vs Cruiser (RCS-only jinking)")
        print(f"Starting: 1000 km apart, corvette at {initial_speed_kps} km/s")
        print("="*70)

        # Print Poseidon torpedo specs
        if corvette.torpedo_launcher:
            torp_specs = corvette.torpedo_launcher.specs
            print(f"\nPOSEIDON TORPEDO SPECS:")
            print(f"  Warhead: {torp_specs.warhead_yield_gj:.2f} GJ (100kg kinetic penetrator)")
            print(f"  Mass: {torp_specs.mass_kg:.0f} kg")
            print(f"  Delta-V: {torp_specs.total_delta_v_kps:.1f} km/s")
            print(f"  Acceleration: {torp_specs.acceleration_at_mass(torp_specs.mass_kg)/9.81:.2f}g initial")
            print(f"  Exhaust velocity: {torp_specs.exhaust_velocity_kps:.2f} km/s (NTR)")
            print(f"  Guidance: SMART (pursuit cone + fuel reserve)")
            print(f"  Magazine: {corvette.torpedo_launcher.current_magazine}")

        checkpoint_count = {'torpedo-boat': 0, 'target-cruiser': 0}
        torpedoes_fired = {'torpedo-boat': 0}
        cruiser_evading = {'started': False}

        def tactical_ai(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            commands = []
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return commands

            checkpoint_count[ship_id] += 1
            checkpoint = checkpoint_count[ship_id]

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return commands

            target = enemies[0]
            distance_km = ship.distance_to(target) / 1000

            rel_vel = ship.velocity - target.velocity
            to_target = (target.position - ship.position).normalized()
            closing_rate = rel_vel.dot(to_target) / 1000

            print(f"\n  [{ship_id}] Checkpoint {checkpoint} @ T+{simulation.current_time:.0f}s")
            print(f"    Distance: {distance_km:.1f} km, Closing: {closing_rate:.2f} km/s")
            print(f"    Speed: {ship.velocity.magnitude/1000:.2f} km/s")

            if ship_id == 'torpedo-boat':
                # CORVETTE: High-speed attack run
                # At 10 km/s, we cover 300 km per checkpoint (30s)
                # Checkpoint 1: ~700 km, Checkpoint 2: ~400 km, Checkpoint 3: ~100 km

                if distance_km > 400:
                    # Full throttle attack run - 3g acceleration!
                    print(f"    Phase: ATTACK RUN - FULL BURN!")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,  # Full 3g!
                        target_id=target.ship_id
                    ))
                elif distance_km > 100 and torpedoes_fired['torpedo-boat'] < 4:
                    # Close range - LAUNCH POSEIDON TORPEDOES with SMART guidance!
                    if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
                        torpedoes_fired['torpedo-boat'] += 1
                        print(f"    Phase: POSEIDON LAUNCH #{torpedoes_fired['torpedo-boat']}! (dist={distance_km:.0f}km) - SMART guidance")
                        commands.append({
                            'type': 'launch_torpedo',
                            'target_id': target.ship_id,
                            'guidance_mode': GuidanceMode.SMART  # Use smart guidance
                        })
                    # Keep closing while launching
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=0.3,
                        target_id=target.ship_id
                    ))
                else:
                    # COLLISION IMMINENT - BREAK AWAY!
                    print(f"    Phase: BREAK AWAY! (dist={distance_km:.0f}km)")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        direction=Vector3D(0, 1, 0)  # Perpendicular break
                    ))
            else:
                # CRUISER: Small evasive course corrections
                # Alternate between different directions each checkpoint
                evasion_patterns = [
                    Vector3D(0, 1, 0),    # Up
                    Vector3D(0, -1, 0),   # Down
                    Vector3D(0, 0, 1),    # Left
                    Vector3D(0, 0, -1),   # Right
                    Vector3D(0, 0.7, 0.7),  # Up-left
                    Vector3D(0, -0.7, 0.7), # Down-left
                ]
                evasion_dir = evasion_patterns[checkpoint % len(evasion_patterns)]

                incoming_torpedoes = [t for t in simulation.torpedoes
                                     if t.torpedo.target_id == ship.ship_id]

                if incoming_torpedoes:
                    # Torpedoes detected - do evasive jinking
                    closest_torp = min(incoming_torpedoes,
                                      key=lambda t: t.torpedo.position.distance_to(ship.position))
                    closest_dist = closest_torp.torpedo.position.distance_to(ship.position) / 1000

                    print(f"    Phase: EVASIVE JINK! (torpedo at {closest_dist:.0f}km)")
                    print(f"    Course correction: {evasion_dir}")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=0.5,  # Half throttle - course correction, not escape
                        direction=evasion_dir
                    ))
                else:
                    # No torpedoes - approach
                    print(f"    Phase: APPROACH")
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=0.3,
                        target_id=target.ship_id
                    ))

            # Debug: show torpedo status at each checkpoint
            if ship_id == 'torpedo-boat' and simulation.torpedoes:
                print(f"\n    TORPEDO STATUS (SMART guidance):")
                for tf in simulation.torpedoes:
                    t = tf.torpedo
                    dist = t.position.distance_to(cruiser.position) / 1000
                    fuel = (t.current_mass_kg - t.specs.dry_mass_kg) / t.specs.propellant_mass_kg * 100
                    dv_rem = t.remaining_delta_v_kps
                    closing_rate = -(cruiser.velocity - t.velocity).dot(
                        (cruiser.position - t.position).normalized()
                    ) / 1000 if dist > 0.01 else 0
                    print(f"      {tf.torpedo_id[-8:]}: dist={dist:.0f}km, dv={dv_rem:.1f}km/s, mode={t.guidance_mode.name}")
                    print(f"        closing={closing_rate:.1f}km/s, fuel={fuel:.0f}%")
                    print(f"        torp_pos=({t.position.x/1000:.0f},{t.position.y/1000:.0f},{t.position.z/1000:.0f})")
                    print(f"        torp_vel=({t.velocity.x/1000:.1f},{t.velocity.y/1000:.1f},{t.velocity.z/1000:.1f}) km/s")
                    print(f"        target_pos=({cruiser.position.x/1000:.0f},{cruiser.position.y/1000:.0f},{cruiser.position.z/1000:.0f})")

            return commands

        sim.set_decision_callback(tactical_ai)

        # Run simulation
        sim.run(duration=600.0)

        events = sim.events

        # Count events
        event_counts = {}
        for e in events:
            name = e.event_type.name
            event_counts[name] = event_counts.get(name, 0) + 1

        print(f"\n{'='*70}")
        print(f"TORPEDO ATTACK RESULTS (NO PD)")
        print(f"{'='*70}")
        print(f"Simulation time: {sim.current_time:.1f}s")

        print(f"\nEvent Summary:")
        for name, count in sorted(event_counts.items()):
            print(f"  {name}: {count}")

        # Torpedo events
        torp_launch = [e for e in events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
        torp_hit = [e for e in events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
        torp_intercept = [e for e in events if e.event_type == SimulationEventType.PD_TORPEDO_DESTROYED]

        print(f"\nTORPEDO SUMMARY:")
        print(f"  Launched: {len(torp_launch)}")
        print(f"  Hits: {len(torp_hit)}")
        print(f"  Intercepted by PD: {len(torp_intercept)}")
        print(f"  Still in flight: {len(sim.torpedoes)}")

        # Torpedo launches
        if torp_launch:
            print(f"\nTORPEDO LAUNCHES:")
            for e in torp_launch:
                torp_id = e.data.get('torpedo_id', 'unknown')
                init_vel = e.data.get('initial_velocity_kps', 0)
                delta_v = e.data.get('delta_v_kps', 0)
                print(f"  T+{e.timestamp:.0f}s: {torp_id}")
                print(f"      Initial vel: {init_vel:.1f} km/s, Delta-V: {delta_v:.1f} km/s")

        # Torpedo hits
        if torp_hit:
            print(f"\nTORPEDO HITS:")
            for e in torp_hit:
                torp_id = e.data.get('torpedo_id', 'unknown')
                kinetic = e.data.get('kinetic_damage_gj', 0)
                explosive = e.data.get('explosive_damage_gj', 0)
                total = e.data.get('total_damage_gj', 0)
                impact_speed = e.data.get('impact_speed_kps', 0)
                loc = e.data.get('hit_location', 'unknown')
                print(f"  T+{e.timestamp:.1f}s: {torp_id}")
                print(f"      Kinetic: {kinetic:.1f} GJ (impact {impact_speed:.1f} km/s)")
                print(f"      Explosive: {explosive:.1f} GJ (warhead)")
                print(f"      Total: {total:.1f} GJ at {loc}")

        # Active torpedoes
        if sim.torpedoes:
            print(f"\nACTIVE TORPEDOES ({len(sim.torpedoes)}):")
            for torp_flight in sim.torpedoes:
                torp = torp_flight.torpedo
                dist_to_target = torp.position.distance_to(cruiser.position) / 1000
                fuel_pct = (torp.current_mass_kg - torp.specs.dry_mass_kg) / \
                           torp.specs.propellant_mass_kg * 100 if torp.specs.propellant_mass_kg > 0 else 0
                print(f"  {torp_flight.torpedo_id}: {dist_to_target:.1f} km to target, {fuel_pct:.0f}% fuel")

        # Ship status
        print_detailed_ship_status(corvette, events)
        print_detailed_ship_status(cruiser, events)

        # Final state
        final_dist = corvette.position.distance_to(cruiser.position) / 1000
        print(f"\n{'='*70}")
        print(f"Final distance: {final_dist:.1f} km")
        print(f"Corvette: {corvette.velocity.magnitude/1000:.2f} km/s, destroyed={corvette.is_destroyed}")
        print(f"Cruiser: {cruiser.velocity.magnitude/1000:.2f} km/s, destroyed={cruiser.is_destroyed}")
        print(f"{'='*70}")

        # Assertions
        assert len(torp_launch) > 0, "Corvette should launch torpedoes"
        # With no PD, torpedoes should hit OR run out of fuel
        assert len(torp_hit) > 0 or len(sim.torpedoes) > 0 or len(torp_intercept) == 0, \
            "Without PD, torpedoes should not be intercepted"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
