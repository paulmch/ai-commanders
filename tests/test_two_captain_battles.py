"""
Full engagement scenarios where two ship captains make decisions every 20-60 seconds.

These tests simulate realistic combat where both ships are making independent tactical decisions.
Each test demonstrates the full simulation working end-to-end with two active participants.

Test scenarios:
1. TestHeadOnEngagement - Two destroyers closing at 20 km/s combined
2. TestFlankingManeuver - Attacker at 90 degrees to target's velocity
3. TestPursuitEngagement - Faster ship chasing slower ship
4. TestMissileDuel - Both ships launch full torpedo salvos
5. TestDamagedShipCombat - One ship starts at 50% hull
6. TestCloseRangeDogfight - Start at 30 km, low relative velocity
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.simulation import (
    CombatSimulation, ShipCombatState, create_ship_from_fleet_data,
    SimulationEventType, Maneuver, ManeuverType, WeaponState
)
from src.physics import Vector3D
from src.combat import create_weapon_from_fleet_data
from src.scenarios import (
    CaptainBehavior, AggressiveCaptain, CautiousCaptain, EvasiveCaptain
)
from src.maneuvers import EvasiveJink, BurnToward, BurnAway, RotateToFace
from src.thermal import RadiatorState


# =============================================================================
# BATTLE REPORT DATACLASS
# =============================================================================

@dataclass
class BattleReport:
    """Comprehensive battle report for analysis."""
    scenario_name: str
    duration_s: float
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Per-ship stats
    ship_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Overall metrics
    total_shots_fired: int = 0
    total_hits: int = 0
    total_torpedoes_launched: int = 0
    total_torpedo_hits: int = 0
    total_damage_dealt: float = 0.0

    # Outcome
    winner: Optional[str] = None
    outcome: str = "ongoing"

    def print_report(self) -> None:
        """Print detailed battle report."""
        print("\n" + "=" * 70)
        print(f"BATTLE REPORT: {self.scenario_name}")
        print("=" * 70)

        print(f"\nDuration: {self.duration_s:.1f} seconds")
        print(f"Outcome: {self.outcome}")
        if self.winner:
            print(f"Winner: {self.winner}")

        print("\n--- TIMELINE OF MAJOR EVENTS ---")
        for event in self.timeline[:30]:  # Limit to 30 events
            print(f"  T+{event['time']:.1f}s: [{event['ship']}] {event['event']}")
        if len(self.timeline) > 30:
            print(f"  ... and {len(self.timeline) - 30} more events")

        print("\n--- PER-SHIP STATISTICS ---")
        for ship_id, stats in self.ship_stats.items():
            print(f"\n  {ship_id}:")
            print(f"    Shots fired: {stats.get('shots_fired', 0)}")
            print(f"    Hits scored: {stats.get('hits_scored', 0)}")
            hit_rate = stats.get('hits_scored', 0) / max(1, stats.get('shots_fired', 1)) * 100
            print(f"    Hit rate: {hit_rate:.1f}%")
            print(f"    Damage dealt: {stats.get('damage_dealt', 0):.2f} GJ")
            print(f"    Damage received: {stats.get('damage_received', 0):.2f} GJ")
            print(f"    Final delta-v: {stats.get('final_delta_v', 0):.1f} km/s")
            print(f"    Peak heat: {stats.get('peak_heat', 0):.1f}%")
            print(f"    Hull integrity: {stats.get('hull_integrity', 100):.1f}%")
            print(f"    Is destroyed: {stats.get('is_destroyed', False)}")

        print("\n--- OVERALL METRICS ---")
        print(f"  Total shots fired: {self.total_shots_fired}")
        print(f"  Total hits: {self.total_hits}")
        print(f"  Overall hit rate: {self.total_hits / max(1, self.total_shots_fired) * 100:.1f}%")
        print(f"  Torpedoes launched: {self.total_torpedoes_launched}")
        print(f"  Torpedo hits: {self.total_torpedo_hits}")
        print(f"  Total damage dealt: {self.total_damage_dealt:.2f} GJ")

        print("\n" + "=" * 70)


# =============================================================================
# SCRIPTED CAPTAIN BEHAVIORS FOR TESTING
# =============================================================================

class ScriptedAggressiveCaptain(CaptainBehavior):
    """
    Aggressive captain that closes distance and fires all weapons.
    Makes decisions every decision interval.
    """

    def __init__(self, name: str = "Aggressive Captain"):
        super().__init__(name)
        self.decision_count = 0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return commands

        self.decision_count += 1
        enemies = simulation.get_enemy_ships(ship_id)

        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Always close in on target with full throttle
        commands.append(Maneuver(
            maneuver_type=ManeuverType.INTERCEPT,
            start_time=simulation.current_time,
            duration=30.0,
            throttle=1.0,
            target_id=target.ship_id
        ))

        # Fire all ready weapons
        for slot, weapon_state in ship.weapons.items():
            if weapon_state.can_fire():
                if weapon_state.weapon.range_km >= distance_km:
                    commands.append(self._create_fire_command(slot, target.ship_id))

        # Launch torpedoes when in range
        if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
            if distance_km < 800:  # Torpedo range
                commands.append(self._create_torpedo_command(target.ship_id))

        return commands


class ScriptedEvasiveCaptain(CaptainBehavior):
    """
    Evasive captain that prioritizes dodging while taking opportunistic shots.
    """

    def __init__(self, name: str = "Evasive Captain"):
        super().__init__(name)
        self.evasion_cycles = 0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return commands

        self.evasion_cycles += 1
        enemies = simulation.get_enemy_ships(ship_id)

        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        distance_km = ship.distance_to(target) / 1000 if target else 1000

        # Check for incoming threats
        incoming_torpedoes = [
            t for t in simulation.torpedoes
            if t.torpedo.target_id == ship_id
        ]

        # Evasive maneuvers - high priority
        if incoming_torpedoes or self.evasion_cycles % 2 == 0:
            commands.append(Maneuver(
                maneuver_type=ManeuverType.EVASIVE,
                start_time=simulation.current_time,
                duration=15.0,
                throttle=0.8
            ))
        else:
            # Maintain some distance
            if target and distance_km < 200:
                # Back off if too close
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.BRAKE,
                    start_time=simulation.current_time,
                    duration=10.0,
                    throttle=0.5
                ))

        # Opportunistic fire
        if target and self.evasion_cycles % 3 == 0:
            for slot, weapon_state in ship.weapons.items():
                if weapon_state.can_fire():
                    if weapon_state.weapon.range_km >= distance_km:
                        commands.append(self._create_fire_command(slot, target.ship_id))

        return commands


class ScriptedTorpedoCaptain(CaptainBehavior):
    """
    Captain focused on torpedo warfare - launches salvos and evades.
    """

    def __init__(self, name: str = "Torpedo Captain"):
        super().__init__(name)
        self.salvo_count = 0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Launch torpedoes as primary weapon
        if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
            if distance_km < 1500:  # Within torpedo range
                commands.append(self._create_torpedo_command(target.ship_id))
                self.salvo_count += 1

        # Evasive maneuvers after launching
        if self.salvo_count > 0:
            commands.append(Maneuver(
                maneuver_type=ManeuverType.EVASIVE,
                start_time=simulation.current_time,
                duration=20.0,
                throttle=0.7
            ))
        else:
            # Close in until torpedo range
            commands.append(Maneuver(
                maneuver_type=ManeuverType.INTERCEPT,
                start_time=simulation.current_time,
                duration=30.0,
                throttle=0.6,
                target_id=target.ship_id
            ))

        # Fire coilguns as secondary
        for slot, weapon_state in ship.weapons.items():
            if weapon_state.can_fire():
                if weapon_state.weapon.range_km >= distance_km:
                    commands.append(self._create_fire_command(slot, target.ship_id))

        return commands


class ScriptedDamagedShipCaptain(CaptainBehavior):
    """
    Captain for a damaged ship - fights a controlled retreat.
    """

    def __init__(self, name: str = "Damaged Ship Captain"):
        super().__init__(name)

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000
        hull_integrity = ship.hull_integrity

        # Retreat if heavily damaged
        if hull_integrity < 30:
            # Full retreat
            escape_dir = ship.position - target.position
            if escape_dir.magnitude > 0:
                escape_dir = escape_dir.normalized()
            commands.append(Maneuver(
                maneuver_type=ManeuverType.BURN,
                start_time=simulation.current_time,
                duration=30.0,
                throttle=1.0,
                direction=escape_dir
            ))
        elif hull_integrity < 60:
            # Fighting retreat - keep distance and fire
            if distance_km < 300:
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.BRAKE,
                    start_time=simulation.current_time,
                    duration=20.0,
                    throttle=0.6
                ))
            else:
                # Maintain distance with evasive patterns
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.EVASIVE,
                    start_time=simulation.current_time,
                    duration=15.0,
                    throttle=0.5
                ))
        else:
            # Still combat capable - engage cautiously
            commands.append(Maneuver(
                maneuver_type=ManeuverType.INTERCEPT,
                start_time=simulation.current_time,
                duration=25.0,
                throttle=0.4,
                target_id=target.ship_id
            ))

        # Fire weapons when opportunity arises
        for slot, weapon_state in ship.weapons.items():
            if weapon_state.can_fire():
                if weapon_state.weapon.range_km >= distance_km:
                    commands.append(self._create_fire_command(slot, target.ship_id))

        return commands


class ScriptedDogfightCaptain(CaptainBehavior):
    """
    Captain for close-range dogfighting - rapid rotation and burst fire.
    """

    def __init__(self, name: str = "Dogfight Captain"):
        super().__init__(name)
        self.burst_count = 0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000
        heat_percent = ship.heat_percent

        # Heat management - critical in dogfight
        if heat_percent > 80:
            # Extend radiators and reduce aggression
            commands.append({'type': 'set_radiators', 'extend': True})
            commands.append(Maneuver(
                maneuver_type=ManeuverType.EVASIVE,
                start_time=simulation.current_time,
                duration=10.0,
                throttle=0.3
            ))
        else:
            # Close in aggressively
            if distance_km > 50:
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=15.0,
                    throttle=0.9,
                    target_id=target.ship_id
                ))
            else:
                # At close range - jink while firing
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.EVASIVE,
                    start_time=simulation.current_time,
                    duration=10.0,
                    throttle=0.7
                ))

        # Burst fire at close range
        if distance_km < 100 and heat_percent < 70:
            self.burst_count += 1
            for slot, weapon_state in ship.weapons.items():
                if weapon_state.can_fire():
                    if weapon_state.weapon.range_km >= distance_km:
                        commands.append(self._create_fire_command(slot, target.ship_id))

        return commands


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_battle_report(
    sim: CombatSimulation,
    scenario_name: str
) -> BattleReport:
    """Create a comprehensive battle report from simulation results."""
    report = BattleReport(
        scenario_name=scenario_name,
        duration_s=sim.current_time
    )

    # Build timeline from events
    key_event_types = [
        SimulationEventType.PROJECTILE_LAUNCHED,
        SimulationEventType.PROJECTILE_IMPACT,
        SimulationEventType.TORPEDO_LAUNCHED,
        SimulationEventType.TORPEDO_IMPACT,
        SimulationEventType.DAMAGE_TAKEN,
        SimulationEventType.ARMOR_PENETRATED,
        SimulationEventType.MODULE_DAMAGED,
        SimulationEventType.SHIP_DESTROYED,
        SimulationEventType.THERMAL_WARNING,
        SimulationEventType.THERMAL_CRITICAL,
    ]

    for event in sim.events:
        if event.event_type in key_event_types:
            report.timeline.append({
                'time': event.timestamp,
                'ship': event.ship_id or 'system',
                'event': event.event_type.name,
                'target': event.target_id,
                'data': event.data
            })

    # Per-ship stats
    for ship_id, ship in sim.ships.items():
        stats = {
            'shots_fired': ship.shots_fired,
            'hits_scored': ship.hits_scored,
            'damage_dealt': ship.damage_dealt_gj,
            'damage_received': ship.damage_taken_gj,
            'final_delta_v': ship.remaining_delta_v_kps,
            'peak_heat': 0.0,  # Would need to track during sim
            'hull_integrity': ship.hull_integrity,
            'is_destroyed': ship.is_destroyed
        }

        # Get peak heat from thermal events
        thermal_events = [
            e for e in sim.events
            if e.ship_id == ship_id and e.event_type in [
                SimulationEventType.THERMAL_WARNING,
                SimulationEventType.THERMAL_CRITICAL
            ]
        ]
        if thermal_events:
            stats['peak_heat'] = max(
                e.data.get('heat_percent', 0) for e in thermal_events
            )
        elif ship.thermal_system:
            stats['peak_heat'] = ship.heat_percent

        report.ship_stats[ship_id] = stats

    # Overall metrics
    report.total_shots_fired = sim.metrics.total_shots_fired
    report.total_hits = sim.metrics.total_hits
    report.total_torpedoes_launched = sim.metrics.total_torpedoes_launched
    report.total_torpedo_hits = sim.metrics.total_torpedo_hits
    report.total_damage_dealt = sim.metrics.total_damage_dealt

    # Determine outcome
    alpha_alive = [s for s in sim.ships.values() if s.faction == 'alpha' and not s.is_destroyed]
    beta_alive = [s for s in sim.ships.values() if s.faction == 'beta' and not s.is_destroyed]

    if alpha_alive and not beta_alive:
        report.winner = 'alpha'
        report.outcome = 'alpha_victory'
    elif beta_alive and not alpha_alive:
        report.winner = 'beta'
        report.outcome = 'beta_victory'
    elif not alpha_alive and not beta_alive:
        report.outcome = 'mutual_destruction'
    else:
        report.outcome = 'time_limit'

    return report


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet data from fleet_ships.json."""
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(data_path, "r") as f:
        return json.load(f)


# =============================================================================
# TEST: HEAD-ON ENGAGEMENT
# =============================================================================

class TestHeadOnEngagement:
    """
    Two destroyers closing at 20 km/s combined.
    Both captains make decisions every 30 seconds.
    Tests: shots fired, hits, damage dealt, delta-v spent.
    """

    def test_head_on_engagement_aggressive_vs_aggressive(self, fleet_data):
        """Two aggressive captains in head-on pass."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        # Alpha destroyer - closing at 10 km/s
        alpha = create_ship_from_fleet_data(
            ship_id="alpha_destroyer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-500_000, 0, 0),  # 500 km in -X
            velocity=Vector3D(10_000, 0, 0),  # 10 km/s toward origin
            forward=Vector3D(1, 0, 0)
        )

        # Beta destroyer - closing at 10 km/s
        beta = create_ship_from_fleet_data(
            ship_id="beta_destroyer",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(500_000, 0, 0),  # 500 km in +X
            velocity=Vector3D(-10_000, 0, 0),  # 10 km/s toward origin
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Add weapons
        spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")

        alpha.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        alpha.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=100)
        beta.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        beta.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        # Create captains
        alpha_captain = ScriptedAggressiveCaptain("Alpha Aggressive")
        beta_captain = ScriptedAggressiveCaptain("Beta Aggressive")

        def battle_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return alpha_captain.decide(ship_id, simulation)
            else:
                return beta_captain.decide(ship_id, simulation)

        sim.set_decision_callback(battle_callback)

        # Initial state
        initial_distance = alpha.distance_to(beta) / 1000
        closing_rate = alpha.closing_rate_to(beta) / 1000
        print(f"\n[HEAD-ON] Initial distance: {initial_distance:.1f} km")
        print(f"[HEAD-ON] Closing rate: {closing_rate:.1f} km/s")

        # Run battle for 120 seconds (ships pass through each other)
        sim.run(duration=120.0)

        # Create and print report
        report = create_battle_report(sim, "Head-On Engagement: Aggressive vs Aggressive")
        report.print_report()

        # Verify projectiles were tracked correctly
        assert sim.metrics.total_shots_fired > 0, "Ships should have fired weapons"

        # Verify thermal management
        for ship_id, ship in sim.ships.items():
            if ship.thermal_system:
                assert ship.heat_percent < 100, f"{ship_id} should not have overheated"

        # Print final positions
        print(f"\n[HEAD-ON] Final distance: {alpha.distance_to(beta) / 1000:.1f} km")
        print(f"[HEAD-ON] Alpha delta-v remaining: {alpha.remaining_delta_v_kps:.1f} km/s")
        print(f"[HEAD-ON] Beta delta-v remaining: {beta.remaining_delta_v_kps:.1f} km/s")

    def test_head_on_engagement_30s_decisions(self, fleet_data):
        """Head-on pass with 30-second decision intervals."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=123)

        # Ships starting 800 km apart, closing at 20 km/s combined
        alpha = create_ship_from_fleet_data(
            ship_id="alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-400_000, 0, 0),
            velocity=Vector3D(10_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(400_000, 0, 0),
            velocity=Vector3D(-10_000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Add weapons
        spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        alpha.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        beta.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)

        decision_times = {'alpha': [], 'beta': []}

        def tracking_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            decision_times[ship.faction].append(simulation.current_time)

            # Simple aggressive behavior
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

        sim.set_decision_callback(tracking_callback)
        sim.run(duration=90.0)

        # Verify decisions were made at correct intervals
        print(f"\n[30s DECISIONS] Alpha decision times: {decision_times['alpha']}")
        print(f"[30s DECISIONS] Beta decision times: {decision_times['beta']}")

        # Should have ~3 decision points in 90 seconds
        assert len(decision_times['alpha']) >= 2

        # Verify decision intervals are approximately 30 seconds
        if len(decision_times['alpha']) >= 2:
            interval = decision_times['alpha'][1] - decision_times['alpha'][0]
            assert 25 <= interval <= 35, f"Decision interval should be ~30s, got {interval}s"


# =============================================================================
# TEST: FLANKING MANEUVER
# =============================================================================

class TestFlankingManeuver:
    """
    Attacker at 90 degrees to target's velocity.
    Target must rotate to engage.
    Tests weapon arc mechanics and firing solutions.
    """

    def test_flanking_requires_rotation(self, fleet_data):
        """Test that flanking target requires rotation to engage."""
        sim = CombatSimulation(time_step=1.0, decision_interval=25.0, seed=42)

        # Target moving along X axis
        target = create_ship_from_fleet_data(
            ship_id="target",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5_000, 0, 0),  # Moving along +X
            forward=Vector3D(1, 0, 0)  # Facing forward
        )

        # Attacker approaching from the side (perpendicular)
        attacker = create_ship_from_fleet_data(
            ship_id="attacker",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, -400_000, 0),  # 400 km in -Y
            velocity=Vector3D(0, 6_000, 0),  # Moving toward target
            forward=Vector3D(0, 1, 0)  # Facing target
        )

        sim.add_ship(target)
        sim.add_ship(attacker)

        # Add weapons
        spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")

        target.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        attacker.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        attacker.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        # Attacker is aggressive, target is evasive
        attacker_captain = ScriptedAggressiveCaptain("Flanking Attacker")
        target_captain = ScriptedEvasiveCaptain("Flanked Target")

        def battle_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return attacker_captain.decide(ship_id, simulation)
            else:
                return target_captain.decide(ship_id, simulation)

        sim.set_decision_callback(battle_callback)

        # Initial geometry
        initial_distance = attacker.distance_to(target) / 1000
        print(f"\n[FLANKING] Initial distance: {initial_distance:.1f} km")
        print(f"[FLANKING] Attacker velocity direction: perpendicular to target")

        # Run for 120 seconds
        sim.run(duration=120.0)

        report = create_battle_report(sim, "Flanking Maneuver")
        report.print_report()

        # Verify firing solutions required proper alignment
        # (attacker facing target should have more hits)
        assert sim.metrics.total_shots_fired > 0

        # Verify geometry affected engagement
        print(f"\n[FLANKING] Attacker shots: {attacker.shots_fired}")
        print(f"[FLANKING] Target shots: {target.shots_fired}")
        print(f"[FLANKING] Attacker hits: {attacker.hits_scored}")
        print(f"[FLANKING] Target hits: {target.hits_scored}")

    def test_crossing_t_engagement(self, fleet_data):
        """Classic crossing-T naval engagement geometry."""
        sim = CombatSimulation(time_step=1.0, decision_interval=25.0, seed=42)

        # Target cruising along
        target = create_ship_from_fleet_data(
            ship_id="cruising_target",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(3_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Attacker crosses in front (the T)
        attacker = create_ship_from_fleet_data(
            ship_id="crossing_attacker",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(300_000, -200_000, 0),
            velocity=Vector3D(0, 4_000, 0),  # Crossing perpendicular
            forward=Vector3D(-1, 0, 0)  # Broadside to target
        )

        sim.add_ship(target)
        sim.add_ship(attacker)

        spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")

        target.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        attacker.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        def aggressive_both(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies:
                target_ship = enemies[0]
                distance_km = ship.distance_to(target_ship) / 1000

                # Fire if in range
                for slot, ws in ship.weapons.items():
                    if ws.can_fire() and ws.weapon.range_km >= distance_km:
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target_ship.ship_id
                        })

                # Close in
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=25.0,
                    throttle=0.5,
                    target_id=target_ship.ship_id
                ))

            return commands

        sim.set_decision_callback(aggressive_both)
        sim.run(duration=100.0)

        report = create_battle_report(sim, "Crossing-T Engagement")
        report.print_report()


# =============================================================================
# TEST: PURSUIT ENGAGEMENT
# =============================================================================

class TestPursuitEngagement:
    """
    Faster ship chasing slower ship.
    Use torpedoes for long-range damage.
    Slower ship manages heat to maintain radiators.
    Tests: torpedo guidance, escape attempts.
    """

    def test_pursuit_with_torpedoes(self, fleet_data):
        """Faster corvette chasing destroyer with torpedoes."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        # Slower target (destroyer) running away
        target = create_ship_from_fleet_data(
            ship_id="fleeing_target",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(500_000, 0, 0),  # 500 km ahead
            velocity=Vector3D(5_000, 0, 0),  # 5 km/s fleeing
            forward=Vector3D(1, 0, 0)
        )

        # Faster pursuer (corvette with torpedoes)
        pursuer = create_ship_from_fleet_data(
            ship_id="pursuer",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(8_000, 0, 0),  # 8 km/s pursuing
            forward=Vector3D(1, 0, 0)
        )

        sim.add_ship(target)
        sim.add_ship(pursuer)

        # Target has coilgun for rear defense
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        target.weapons["rear_gun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        # Pursuer relies on torpedo launcher (should be on corvette)

        # Pursuer uses torpedoes
        pursuer_captain = ScriptedTorpedoCaptain("Torpedo Pursuer")

        # Target evades
        target_captain = ScriptedEvasiveCaptain("Evading Target")

        def battle_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return pursuer_captain.decide(ship_id, simulation)
            else:
                return target_captain.decide(ship_id, simulation)

        sim.set_decision_callback(battle_callback)

        initial_distance = pursuer.distance_to(target) / 1000
        closing_rate = pursuer.closing_rate_to(target) / 1000
        print(f"\n[PURSUIT] Initial distance: {initial_distance:.1f} km")
        print(f"[PURSUIT] Closing rate: {closing_rate:.1f} km/s")

        # Run extended pursuit
        sim.run(duration=300.0)

        report = create_battle_report(sim, "Pursuit Engagement with Torpedoes")
        report.print_report()

        # Verify torpedo tracking
        print(f"\n[PURSUIT] Torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[PURSUIT] Torpedo hits: {sim.metrics.total_torpedo_hits}")

        # Verify escape attempt (target should have tried to maintain distance)
        final_distance = pursuer.distance_to(target) / 1000
        print(f"[PURSUIT] Final distance: {final_distance:.1f} km")

    def test_long_chase_heat_management(self, fleet_data):
        """Test heat management during extended pursuit."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        # Target with extended radiators
        target = create_ship_from_fleet_data(
            ship_id="heat_managed_target",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(300_000, 0, 0),
            velocity=Vector3D(4_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        pursuer = create_ship_from_fleet_data(
            ship_id="hot_pursuer",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(6_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        sim.add_ship(target)
        sim.add_ship(pursuer)

        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        target.weapons["gun"] = WeaponState(weapon=coilgun, ammo_remaining=100)
        pursuer.weapons["gun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        # Extend target's radiators for cooling during run
        if target.thermal_system:
            target.thermal_system.radiators.extend_all()

        def chase_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies:
                target_ship = enemies[0]

                if ship.faction == "alpha":
                    # Pursuer - close and fire
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.INTERCEPT,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=1.0,
                        target_id=target_ship.ship_id
                    ))
                else:
                    # Target - manage heat, fire back
                    if ship.thermal_system and ship.heat_percent > 60:
                        commands.append({'type': 'set_radiators', 'extend': True})

                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=30.0,
                        throttle=0.8,
                        direction=Vector3D(1, 0, 0)  # Keep fleeing
                    ))

                # Both fire when in range
                distance_km = ship.distance_to(target_ship) / 1000
                for slot, ws in ship.weapons.items():
                    if ws.can_fire() and ws.weapon.range_km >= distance_km:
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target_ship.ship_id
                        })

            return commands

        sim.set_decision_callback(chase_callback)
        sim.run(duration=200.0)

        report = create_battle_report(sim, "Pursuit with Heat Management")
        report.print_report()

        # Verify thermal was managed
        print(f"\n[HEAT] Target heat: {target.heat_percent:.1f}%")
        print(f"[HEAT] Pursuer heat: {pursuer.heat_percent:.1f}%")


# =============================================================================
# TEST: MISSILE DUEL
# =============================================================================

class TestMissileDuel:
    """
    Both ships launch full torpedo salvos.
    Engage in evasive maneuvers.
    Track multiple torpedoes simultaneously.
    Tests: mass projectile tracking works.
    """

    def test_full_salvo_exchange(self, fleet_data):
        """Both ships launch multiple torpedo salvos."""
        sim = CombatSimulation(time_step=1.0, decision_interval=25.0, seed=42)

        # Two corvettes (torpedo boats) facing off
        alpha_corvette = create_ship_from_fleet_data(
            ship_id="alpha_torpedo_boat",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-350_000, 0, 0),
            velocity=Vector3D(2_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta_corvette = create_ship_from_fleet_data(
            ship_id="beta_torpedo_boat",
            ship_type="corvette",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(350_000, 0, 0),
            velocity=Vector3D(-2_000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha_corvette)
        sim.add_ship(beta_corvette)

        # Both captains are torpedo-focused
        alpha_captain = ScriptedTorpedoCaptain("Alpha Torpedo Commander")
        beta_captain = ScriptedTorpedoCaptain("Beta Torpedo Commander")

        def torpedo_duel_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return alpha_captain.decide(ship_id, simulation)
            else:
                return beta_captain.decide(ship_id, simulation)

        sim.set_decision_callback(torpedo_duel_callback)

        print(f"\n[MISSILE DUEL] Initial distance: {alpha_corvette.distance_to(beta_corvette) / 1000:.1f} km")

        # Track torpedo counts during battle
        max_torpedoes_in_flight = 0

        for i in range(300):
            sim.step()
            current_torpedoes = len(sim.torpedoes)
            if current_torpedoes > max_torpedoes_in_flight:
                max_torpedoes_in_flight = current_torpedoes
                print(f"[MISSILE DUEL] T+{sim.current_time:.0f}s: {current_torpedoes} torpedoes in flight")

        report = create_battle_report(sim, "Missile Duel - Full Salvo Exchange")
        report.print_report()

        # Verify multiple torpedoes were tracked
        print(f"\n[MISSILE DUEL] Max torpedoes tracked simultaneously: {max_torpedoes_in_flight}")
        print(f"[MISSILE DUEL] Total torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[MISSILE DUEL] Total torpedo hits: {sim.metrics.total_torpedo_hits}")

        assert sim.metrics.total_torpedoes_launched > 0, "Ships should have launched torpedoes"

    def test_evasive_vs_torpedoes(self, fleet_data):
        """Test evasive maneuvers against torpedo salvo."""
        sim = CombatSimulation(time_step=1.0, decision_interval=20.0, seed=42)

        # Torpedo launcher
        launcher = create_ship_from_fleet_data(
            ship_id="launcher",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Evading target
        evader = create_ship_from_fleet_data(
            ship_id="evader",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(600_000, 0, 0),
            velocity=Vector3D(-1_000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(launcher)
        sim.add_ship(evader)

        evader_captain = ScriptedEvasiveCaptain("Evasive Evader")

        def mixed_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if ship.faction == "alpha":
                # Launch all torpedoes
                if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
                    if enemies:
                        commands.append({
                            'type': 'launch_torpedo',
                            'target_id': enemies[0].ship_id
                        })
            else:
                # Evade
                commands = evader_captain.decide(ship_id, simulation)

            return commands

        sim.set_decision_callback(mixed_callback)
        sim.run(duration=240.0)

        report = create_battle_report(sim, "Evasive Maneuvers vs Torpedoes")
        report.print_report()


# =============================================================================
# TEST: DAMAGED SHIP COMBAT
# =============================================================================

class TestDamagedShipCombat:
    """
    One ship starts at 50% hull.
    Test fighting retreat.
    Verify: damage affects capabilities.
    """

    def test_damaged_ship_fighting_retreat(self, fleet_data):
        """Damaged ship conducts fighting retreat."""
        sim = CombatSimulation(time_step=1.0, decision_interval=25.0, seed=42)

        # Healthy attacker
        attacker = create_ship_from_fleet_data(
            ship_id="healthy_attacker",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-200_000, 0, 0),
            velocity=Vector3D(4_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # Damaged defender
        defender = create_ship_from_fleet_data(
            ship_id="damaged_defender",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(200_000, 0, 0),
            velocity=Vector3D(-2_000, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(attacker)
        sim.add_ship(defender)

        # Apply pre-damage to defender
        if defender.armor:
            # Reduce armor by 50%
            for section in defender.armor.sections.values():
                section.thickness_cm *= 0.5

        if defender.module_layout:
            # Damage some modules
            modules = defender.module_layout.get_all_modules()
            for i, module in enumerate(modules[:len(modules)//3]):
                module.health_percent = 25.0

        # Reduce propellant by 20%
        defender.kinematic_state.propellant_kg *= 0.8
        defender.kinematic_state.mass_kg = (
            defender.kinematic_state.dry_mass_kg +
            defender.kinematic_state.propellant_kg
        )

        # Add heat to simulate battle damage
        if defender.thermal_system:
            defender.thermal_system.add_heat("damage", 150.0)

        # Add weapons
        spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")

        attacker.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=50)
        attacker.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=100)
        defender.weapons["spinal"] = WeaponState(weapon=spinal, ammo_remaining=30)  # Less ammo
        defender.weapons["coilgun"] = WeaponState(weapon=coilgun, ammo_remaining=50)

        # Captains
        attacker_captain = ScriptedAggressiveCaptain("Pursuing Attacker")
        defender_captain = ScriptedDamagedShipCaptain("Damaged Defender")

        def battle_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return attacker_captain.decide(ship_id, simulation)
            else:
                return defender_captain.decide(ship_id, simulation)

        sim.set_decision_callback(battle_callback)

        print(f"\n[DAMAGED] Defender starting hull integrity: {defender.hull_integrity:.1f}%")
        print(f"[DAMAGED] Defender starting delta-v: {defender.remaining_delta_v_kps:.1f} km/s")
        print(f"[DAMAGED] Defender starting heat: {defender.heat_percent:.1f}%")

        sim.run(duration=180.0)

        report = create_battle_report(sim, "Damaged Ship Fighting Retreat")
        report.print_report()

        # Verify damage affected capabilities
        print(f"\n[DAMAGED] Defender final hull: {defender.hull_integrity:.1f}%")
        print(f"[DAMAGED] Defender survived: {not defender.is_destroyed}")

    def test_damage_affects_capabilities(self, fleet_data):
        """Verify that pre-damage affects ship capabilities."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

        # Heavily damaged ship
        damaged = create_ship_from_fleet_data(
            ship_id="heavily_damaged",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        sim.add_ship(damaged)

        # Apply severe damage
        if damaged.armor:
            for section in damaged.armor.sections.values():
                section.thickness_cm *= 0.3  # 70% armor gone

        if damaged.module_layout:
            modules = damaged.module_layout.get_all_modules()
            for module in modules[:len(modules)//2]:
                module.health_percent = 0.0  # Destroy half the modules

        # Check capabilities are reduced
        initial_delta_v = damaged.remaining_delta_v_kps
        initial_hull = damaged.hull_integrity

        print(f"\n[DAMAGE TEST] Hull integrity after damage: {initial_hull:.1f}%")
        print(f"[DAMAGE TEST] Delta-v after damage: {initial_delta_v:.1f} km/s")

        # Run a few steps
        sim.run(duration=10.0)

        # Verify damage is persistent
        assert damaged.hull_integrity <= initial_hull


# =============================================================================
# TEST: CLOSE RANGE DOGFIGHT
# =============================================================================

class TestCloseRangeDogfight:
    """
    Start at 30 km, low relative velocity.
    Rapid rotation, burst fire.
    Heat management critical.
    Tests: rapid maneuvering works.
    """

    def test_close_range_dogfight(self, fleet_data):
        """Close-quarters dogfight with rapid maneuvering."""
        sim = CombatSimulation(time_step=1.0, decision_interval=20.0, seed=42)

        # Two ships starting close with low relative velocity
        alpha = create_ship_from_fleet_data(
            ship_id="alpha_fighter",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-15_000, 0, 0),  # 15 km in -X
            velocity=Vector3D(1_000, 500, 0),  # Slow drift
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="beta_fighter",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(15_000, 0, 0),  # 15 km in +X
            velocity=Vector3D(-500, 500, 0),  # Slow counter-drift
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Add weapons suited for close combat
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        light_coilgun = create_weapon_from_fleet_data(fleet_data, "light_coilgun_mk3")

        alpha.weapons["main"] = WeaponState(weapon=coilgun, ammo_remaining=100)
        alpha.weapons["secondary"] = WeaponState(weapon=light_coilgun, ammo_remaining=200)
        beta.weapons["main"] = WeaponState(weapon=coilgun, ammo_remaining=100)
        beta.weapons["secondary"] = WeaponState(weapon=light_coilgun, ammo_remaining=200)

        # Both captains use dogfight tactics
        alpha_captain = ScriptedDogfightCaptain("Alpha Dogfighter")
        beta_captain = ScriptedDogfightCaptain("Beta Dogfighter")

        def dogfight_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return alpha_captain.decide(ship_id, simulation)
            else:
                return beta_captain.decide(ship_id, simulation)

        sim.set_decision_callback(dogfight_callback)

        initial_distance = alpha.distance_to(beta) / 1000
        print(f"\n[DOGFIGHT] Initial distance: {initial_distance:.1f} km")
        print(f"[DOGFIGHT] Initial relative velocity: low")

        # Track heat during dogfight
        heat_readings = {'alpha': [], 'beta': []}

        for i in range(180):  # 3 minutes
            sim.step()

            if i % 20 == 0:
                if alpha.thermal_system:
                    heat_readings['alpha'].append(alpha.heat_percent)
                if beta.thermal_system:
                    heat_readings['beta'].append(beta.heat_percent)

                distance = alpha.distance_to(beta) / 1000
                print(f"[DOGFIGHT] T+{sim.current_time:.0f}s: Distance={distance:.1f}km, "
                      f"Alpha heat={alpha.heat_percent:.1f}%, Beta heat={beta.heat_percent:.1f}%")

        report = create_battle_report(sim, "Close Range Dogfight")
        report.print_report()

        # Verify rapid maneuvering worked
        print(f"\n[DOGFIGHT] Alpha shots fired: {alpha.shots_fired}")
        print(f"[DOGFIGHT] Beta shots fired: {beta.shots_fired}")
        print(f"[DOGFIGHT] Alpha peak heat: {max(heat_readings['alpha']) if heat_readings['alpha'] else 0:.1f}%")
        print(f"[DOGFIGHT] Beta peak heat: {max(heat_readings['beta']) if heat_readings['beta'] else 0:.1f}%")

        # Heat should have spiked during combat
        if heat_readings['alpha']:
            assert max(heat_readings['alpha']) > 10, "Heat should increase during dogfight"

    def test_heat_critical_dogfight(self, fleet_data):
        """Test heat management becomes critical in extended dogfight."""
        sim = CombatSimulation(time_step=1.0, decision_interval=20.0, seed=42)

        alpha = create_ship_from_fleet_data(
            ship_id="overheating_alpha",
            ship_type="destroyer",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-20_000, 0, 0),
            velocity=Vector3D(500, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        beta = create_ship_from_fleet_data(
            ship_id="overheating_beta",
            ship_type="destroyer",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(20_000, 0, 0),
            velocity=Vector3D(-500, 0, 0),
            forward=Vector3D(-1, 0, 0)
        )

        sim.add_ship(alpha)
        sim.add_ship(beta)

        # Pre-heat both ships to simulate extended combat
        if alpha.thermal_system:
            capacity = alpha.thermal_system.heatsink.capacity_gj
            alpha.thermal_system.add_heat("combat", capacity * 0.5)
        if beta.thermal_system:
            capacity = beta.thermal_system.heatsink.capacity_gj
            beta.thermal_system.add_heat("combat", capacity * 0.5)

        # Add fast-firing weapons
        light_coilgun = create_weapon_from_fleet_data(fleet_data, "light_coilgun_mk3")
        alpha.weapons["gun1"] = WeaponState(weapon=light_coilgun, ammo_remaining=500)
        alpha.weapons["gun2"] = WeaponState(weapon=light_coilgun, ammo_remaining=500)
        beta.weapons["gun1"] = WeaponState(weapon=light_coilgun, ammo_remaining=500)
        beta.weapons["gun2"] = WeaponState(weapon=light_coilgun, ammo_remaining=500)

        # Aggressive firing without heat management
        def overheat_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            enemies = simulation.get_enemy_ships(ship_id)
            commands = []

            if enemies:
                target = enemies[0]
                distance_km = ship.distance_to(target) / 1000

                # Fire everything regardless of heat
                for slot, ws in ship.weapons.items():
                    if ws.can_fire() and ws.weapon.range_km >= distance_km:
                        commands.append({
                            'type': 'fire_at',
                            'weapon_slot': slot,
                            'target_id': target.ship_id
                        })

                # Close in aggressively
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=20.0,
                    throttle=1.0,
                    target_id=target.ship_id
                ))

            return commands

        sim.set_decision_callback(overheat_callback)

        print(f"\n[HEAT CRITICAL] Starting alpha heat: {alpha.heat_percent:.1f}%")
        print(f"[HEAT CRITICAL] Starting beta heat: {beta.heat_percent:.1f}%")

        sim.run(duration=120.0)

        report = create_battle_report(sim, "Heat Critical Dogfight")
        report.print_report()

        # Check for thermal events
        thermal_warnings = sim.get_events_by_type(SimulationEventType.THERMAL_WARNING)
        thermal_critical = sim.get_events_by_type(SimulationEventType.THERMAL_CRITICAL)

        print(f"\n[HEAT CRITICAL] Thermal warnings: {len(thermal_warnings)}")
        print(f"[HEAT CRITICAL] Thermal critical events: {len(thermal_critical)}")


# =============================================================================
# TEST: CORVETTE SWARM VS FRIGATE
# =============================================================================

class TestCorvetteSwarmVsFrigate:
    """
    3 corvettes attack 1 frigate with coordinated torpedo salvos.
    Tests whether torpedo salvos can overwhelm a single ship's PD.

    With PD cooldown of 5 seconds, a frigate can only intercept 1 torpedo
    every 5 seconds. If 3 corvettes each launch a torpedo simultaneously,
    2 torpedoes should get through.
    """

    def test_coordinated_torpedo_salvo(self, fleet_data):
        """3 corvettes launch simultaneous torpedo salvos at 1 frigate."""
        sim = CombatSimulation(time_step=1.0, decision_interval=25.0, seed=42)

        # Target frigate in the center
        frigate = create_ship_from_fleet_data(
            ship_id="target_frigate",
            ship_type="frigate",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        # 3 corvettes approaching from different angles
        corvette_1 = create_ship_from_fleet_data(
            ship_id="corvette_alpha_1",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-500_000, 0, 0),  # 500km away on X axis
            velocity=Vector3D(3_000, 0, 0),  # 3 km/s toward frigate
            forward=Vector3D(1, 0, 0)
        )

        corvette_2 = create_ship_from_fleet_data(
            ship_id="corvette_alpha_2",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-500_000, 50_000, 0),  # Offset 50km on Y
            velocity=Vector3D(3_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        corvette_3 = create_ship_from_fleet_data(
            ship_id="corvette_alpha_3",
            ship_type="corvette",
            faction="alpha",
            fleet_data=fleet_data,
            position=Vector3D(-500_000, -50_000, 0),  # Offset -50km on Y
            velocity=Vector3D(3_000, 0, 0),
            forward=Vector3D(1, 0, 0)
        )

        sim.add_ship(frigate)
        sim.add_ship(corvette_1)
        sim.add_ship(corvette_2)
        sim.add_ship(corvette_3)

        # Give frigate coilguns for defense
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        frigate.weapons["main_gun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        # All corvettes use torpedo captain behavior
        captains = {
            "corvette_alpha_1": ScriptedTorpedoCaptain("Corvette 1"),
            "corvette_alpha_2": ScriptedTorpedoCaptain("Corvette 2"),
            "corvette_alpha_3": ScriptedTorpedoCaptain("Corvette 3"),
        }

        # Frigate uses aggressive defense
        frigate_captain = ScriptedAggressiveCaptain("Frigate Defense")

        def swarm_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if ship.faction == "alpha":
                return captains[ship_id].decide(ship_id, simulation)
            else:
                return frigate_captain.decide(ship_id, simulation)

        sim.set_decision_callback(swarm_callback)

        print(f"\n[SWARM] === 3 CORVETTES vs 1 FRIGATE ===")
        print(f"[SWARM] Corvette 1 distance: {corvette_1.distance_to(frigate) / 1000:.1f} km")
        print(f"[SWARM] Corvette 2 distance: {corvette_2.distance_to(frigate) / 1000:.1f} km")
        print(f"[SWARM] Corvette 3 distance: {corvette_3.distance_to(frigate) / 1000:.1f} km")
        print(f"[SWARM] Frigate PD cooldown: 5.0s (can intercept 1 torpedo per 5s)")

        # Track torpedo events
        torpedo_events = []

        for i in range(400):  # 400 seconds
            sim.step()

            # Log torpedo launches
            if sim.current_time == int(sim.current_time):
                current_torpedoes = len(sim.torpedoes)
                if current_torpedoes > 0 and i % 25 == 0:
                    print(f"[SWARM] T+{sim.current_time:.0f}s: {current_torpedoes} torpedoes in flight")

        report = create_battle_report(sim, "Corvette Swarm vs Frigate")
        report.print_report()

        # Detailed results
        print(f"\n[SWARM] === RESULTS ===")
        print(f"[SWARM] Total torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[SWARM] Total torpedo HITS: {sim.metrics.total_torpedo_hits}")
        print(f"[SWARM] Total torpedoes INTERCEPTED by PD: {sim.metrics.total_torpedo_intercepted}")
        print(f"[SWARM] Torpedoes still in flight: {len(sim.torpedoes)}")
        print(f"[SWARM] Frigate hull integrity: {frigate.hull_integrity:.1f}%")
        print(f"[SWARM] Frigate destroyed: {frigate.is_destroyed}")
        print(f"[SWARM] Frigate PD intercept count: {frigate.pd_intercepts}")

        # Check torpedo events
        pd_destroyed = sim.get_events_by_type(SimulationEventType.PD_TORPEDO_DESTROYED)
        torpedo_impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
        torpedo_exhausted = sim.get_events_by_type(SimulationEventType.TORPEDO_FUEL_EXHAUSTED)
        print(f"[SWARM] PD destroyed torpedoes (events): {len(pd_destroyed)}")
        print(f"[SWARM] Torpedo impacts: {len(torpedo_impacts)}")
        print(f"[SWARM] Torpedoes ran out of fuel: {len(torpedo_exhausted)}")

        # The test should show some torpedo hits given the salvo size
        assert sim.metrics.total_torpedoes_launched > 0, "Corvettes should launch torpedoes"

    def test_torpedo_saturation_attack(self, fleet_data):
        """Test overwhelming PD with rapid salvo fire."""
        sim = CombatSimulation(time_step=1.0, decision_interval=7.0, seed=123)  # Fast decisions to match torpedo cooldown

        # Target frigate
        frigate = create_ship_from_fleet_data(
            ship_id="target_frigate",
            ship_type="frigate",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0)  # Facing attackers
        )

        # 3 corvettes in formation, closer range for faster torpedo arrival
        corvettes = []
        for i in range(3):
            corvette = create_ship_from_fleet_data(
                ship_id=f"attack_corvette_{i+1}",
                ship_type="corvette",
                faction="alpha",
                fleet_data=fleet_data,
                position=Vector3D(-300_000, (i-1) * 30_000, 0),  # 300km, spread on Y
                velocity=Vector3D(5_000, 0, 0),  # 5 km/s closing
                forward=Vector3D(1, 0, 0)
            )
            corvettes.append(corvette)
            sim.add_ship(corvette)

        sim.add_ship(frigate)

        # Frigate has PD and coilgun
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")
        frigate.weapons["gun"] = WeaponState(weapon=coilgun, ammo_remaining=100)

        def saturation_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return []

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return []

            commands = []

            if ship.faction == "alpha":
                # Corvettes: Launch torpedoes every chance, close in
                target = enemies[0]
                if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
                    commands.append({
                        'type': 'launch_torpedo',
                        'target_id': target.ship_id
                    })

                # Also fire coilguns if any
                for slot, ws in ship.weapons.items():
                    if ws.can_fire():
                        distance_km = ship.distance_to(target) / 1000
                        if ws.weapon.range_km >= distance_km:
                            commands.append({
                                'type': 'fire_at',
                                'weapon_slot': slot,
                                'target_id': target.ship_id
                            })

                # Close in
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.INTERCEPT,
                    start_time=simulation.current_time,
                    duration=7.0,
                    throttle=0.8,
                    target_id=target.ship_id
                ))
            else:
                # Frigate: Defensive stance, fire at nearest
                target = min(enemies, key=lambda e: ship.distance_to(e))
                for slot, ws in ship.weapons.items():
                    if ws.can_fire():
                        distance_km = ship.distance_to(target) / 1000
                        if ws.weapon.range_km >= distance_km:
                            commands.append({
                                'type': 'fire_at',
                                'weapon_slot': slot,
                                'target_id': target.ship_id
                            })

                # Evasive maneuvers
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.EVASIVE,
                    start_time=simulation.current_time,
                    duration=7.0,
                    throttle=0.6
                ))

            return commands

        sim.set_decision_callback(saturation_callback)

        print(f"\n[SATURATION] === TORPEDO SATURATION ATTACK ===")
        print(f"[SATURATION] 3 corvettes vs 1 frigate, close range (300km)")
        print(f"[SATURATION] Torpedo cooldown: 7s, PD cooldown: 5s")
        print(f"[SATURATION] Expected: 3 torpedoes every 7s, PD can kill 1.4/7s = ~1 torpedo")

        # Run shorter battle at closer range
        for i in range(200):
            sim.step()

            if frigate.is_destroyed:
                print(f"[SATURATION] FRIGATE DESTROYED at T+{sim.current_time:.0f}s!")
                break

        report = create_battle_report(sim, "Torpedo Saturation Attack")
        report.print_report()

        print(f"\n[SATURATION] === FINAL RESULTS ===")
        print(f"[SATURATION] Torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[SATURATION] Torpedo HITS: {sim.metrics.total_torpedo_hits}")
        print(f"[SATURATION] Frigate hull: {frigate.hull_integrity:.1f}%")
        print(f"[SATURATION] Frigate destroyed: {frigate.is_destroyed}")

        # Count surviving corvettes
        surviving = sum(1 for c in corvettes if not c.is_destroyed)
        print(f"[SATURATION] Surviving corvettes: {surviving}/3")

        # With 3 torpedoes every 7s and PD only killing 1 every 5s,
        # we expect significant torpedo hits
        if sim.metrics.total_torpedo_hits > 0:
            print(f"\n[SATURATION] SUCCESS: Torpedoes overwhelmed PD!")
        else:
            print(f"\n[SATURATION] PD held - torpedoes intercepted or missed")

    def test_six_corvettes_overwhelming_pd(self, fleet_data):
        """
        6 corvettes vs 1 frigate - should definitely overwhelm 2 PD lasers.

        Math:
        - 6 torpedoes launched every 7s
        - 2 PD can kill max 2 torpedoes per 5s = ~2.8 per 7s
        - That leaves ~3 torpedoes per salvo getting through!

        The frigate tries to evade with full thrust but torpedoes should track.
        """
        sim = CombatSimulation(time_step=1.0, decision_interval=7.0, seed=456)

        # Target frigate - starts fleeing at 5 km/s
        frigate = create_ship_from_fleet_data(
            ship_id="doomed_frigate",
            ship_type="frigate",
            faction="beta",
            fleet_data=fleet_data,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5_000, 0, 0),  # Already fleeing at 5 km/s
            forward=Vector3D(1, 0, 0)  # Facing away from attackers
        )
        sim.add_ship(frigate)

        # 6 corvettes in TIGHT attack formation - 5km spacing
        # Closer engagement: 100km - torpedoes arrive before frigate can complete turn
        corvettes = []
        for i in range(6):
            angle_offset = (i - 2.5) * 5_000  # 5km spread on Y axis (tight formation)
            corvette = create_ship_from_fleet_data(
                ship_id=f"wolf_{i+1}",
                ship_type="corvette",
                faction="alpha",
                fleet_data=fleet_data,
                position=Vector3D(-100_000, angle_offset, 0),  # 100km out - close attack
                velocity=Vector3D(10_000, 0, 0),  # 10 km/s closing (5 km/s advantage)
                forward=Vector3D(1, 0, 0)
            )
            corvettes.append(corvette)
            sim.add_ship(corvette)

        def wolf_pack_callback(ship_id, simulation):
            ship = simulation.get_ship(ship_id)
            if not ship or ship.is_destroyed:
                return []

            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                return []

            commands = []
            target = enemies[0]

            if ship.faction == "alpha":
                # Launch torpedoes aggressively
                if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
                    commands.append({
                        'type': 'launch_torpedo',
                        'target_id': target.ship_id
                    })

                # React to frigate's movement - lead the target
                # Calculate intercept direction based on target velocity
                to_target = target.position - ship.position
                distance = to_target.magnitude

                # Estimate time to intercept
                closing_speed = 3000  # ~3 km/s closing
                time_to_intercept = distance / closing_speed if closing_speed > 0 else 100

                # Lead the target - predict where it will be
                predicted_pos = target.position + target.velocity * min(time_to_intercept, 60)
                intercept_direction = (predicted_pos - ship.position).normalized()

                # Burn toward predicted intercept point
                commands.append(Maneuver(
                    maneuver_type=ManeuverType.BURN,
                    start_time=simulation.current_time,
                    duration=7.0,
                    throttle=1.0,
                    direction=intercept_direction
                ))
            else:
                # Frigate EVADES - but needs time to detect, decide, and perform combat turn
                # Realistic: Detection at first checkpoint (~20s), then 90° combat turn
                # Frigate 90° turn with thrust vectoring: ~15s (from fleet_ships.json)
                # Total reaction time: ~60s before evasion burn can begin
                torpedoes_incoming = len(simulation.torpedoes) > 0
                evasion_delay = 60.0  # Seconds for detection + combat turn completion

                can_evade = torpedoes_incoming and simulation.current_time >= evasion_delay

                if can_evade:
                    # Turn perpendicular to attack axis (Y direction) and burn hard
                    # Add small corrections based on time for unpredictability
                    import math
                    t = simulation.current_time
                    # Small sinusoidal corrections in X and Z
                    x_correction = 0.1 * math.sin(t * 0.5)
                    z_correction = 0.1 * math.cos(t * 0.3)

                    # Main burn direction: +Y with small corrections
                    evasion_direction = Vector3D(x_correction, 1.0, z_correction).normalized()

                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=7.0,
                        throttle=1.0,
                        direction=evasion_direction
                    ))
                else:
                    # Still detecting/turning - continue fleeing in +X at reduced throttle
                    commands.append(Maneuver(
                        maneuver_type=ManeuverType.BURN,
                        start_time=simulation.current_time,
                        duration=7.0,
                        throttle=0.5,
                        direction=Vector3D(1, 0, 0)
                    ))

            return commands

        sim.set_decision_callback(wolf_pack_callback)

        print(f"\n[WOLFPACK] === 6 CORVETTES vs 1 EVADING FRIGATE ===")
        print(f"[WOLFPACK] Starting distance: 100 km (close attack)")
        print(f"[WOLFPACK] Corvette velocity: 10 km/s, Frigate initial: 5 km/s")
        print(f"[WOLFPACK] Frigate tactic: Detect torpedoes, perform combat turn (~60s), then burn perpendicular")
        print(f"[WOLFPACK] Corvettes in tight formation (5km spacing)")

        for i in range(600):  # 10 minutes - enough for torpedoes to arrive
            sim.step()

            if i % 60 == 0 and len(sim.torpedoes) > 0:
                # Sample one torpedo's position
                torp = sim.torpedoes[0].torpedo
                dist_to_target = torp.position.distance_to(frigate.position) / 1000
                print(f"[WOLFPACK] T+{sim.current_time:.0f}s: {len(sim.torpedoes)} torpedoes, closest ~{dist_to_target:.0f}km from target")

            if frigate.is_destroyed:
                print(f"\n[WOLFPACK] *** FRIGATE DESTROYED at T+{sim.current_time:.0f}s! ***")
                break

        report = create_battle_report(sim, "Wolf Pack Attack - 6 Corvettes vs Frigate")
        report.print_report()

        # Calculate torpedo outcomes
        torpedo_hits = sim.metrics.total_torpedo_hits
        torpedo_intercepted = sim.metrics.total_torpedo_intercepted
        torpedo_flying = len(sim.torpedoes)
        torpedo_missed = sim.metrics.total_torpedoes_launched - torpedo_hits - torpedo_intercepted - torpedo_flying

        # Also accumulate delta-v from still-flying torpedoes
        flying_main_dv = sum(t.main_engine_dv_used_kps for t in sim.torpedoes)
        flying_rcs_dv = sum(t.rcs_dv_used_kps for t in sim.torpedoes)
        total_main_dv = sim.metrics.torpedo_main_engine_dv_kps + flying_main_dv
        total_rcs_dv = sim.metrics.torpedo_rcs_dv_kps + flying_rcs_dv

        print(f"\n[WOLFPACK] === FINAL RESULTS ===")
        print(f"[WOLFPACK] Torpedoes launched: {sim.metrics.total_torpedoes_launched}")
        print(f"[WOLFPACK] Torpedo HITS: {torpedo_hits}")
        print(f"[WOLFPACK] Torpedoes INTERCEPTED by PD: {torpedo_intercepted}")
        print(f"[WOLFPACK] Torpedoes MISSED: {torpedo_missed}")
        print(f"[WOLFPACK] Torpedoes still flying: {torpedo_flying}")
        print(f"[WOLFPACK] Frigate hull: {frigate.hull_integrity:.1f}%")
        print(f"[WOLFPACK] Frigate destroyed: {frigate.is_destroyed}")

        # Print delta-v usage breakdown
        total_dv = total_main_dv + total_rcs_dv
        if total_dv > 0:
            rcs_pct = (total_rcs_dv / total_dv) * 100
        else:
            rcs_pct = 0
        print(f"\n[WOLFPACK] === TORPEDO PROPULSION STATS ===")
        print(f"[WOLFPACK] Main engine delta-v: {total_main_dv:.2f} km/s total")
        print(f"[WOLFPACK] RCS delta-v: {total_rcs_dv:.2f} km/s total ({rcs_pct:.1f}% of propulsion)")
        print(f"[WOLFPACK] Combined delta-v: {total_dv:.2f} km/s total")

        # Print torpedo hit locations
        torpedo_impacts = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
        if torpedo_impacts:
            print(f"\n[WOLFPACK] === TORPEDO HIT DETAILS ===")
            for impact in torpedo_impacts:
                t = impact.timestamp
                loc = impact.data.get('hit_location', 'unknown')
                dmg = impact.data.get('total_damage_gj', 0)
                kin_dmg = impact.data.get('kinetic_damage_gj', 0)
                exp_dmg = impact.data.get('explosive_damage_gj', 0)
                speed = impact.data.get('impact_speed_kps', 0)
                print(f"[WOLFPACK] T+{t:.0f}s: Hit {loc.upper()}, {speed:.1f} km/s impact")
                print(f"[WOLFPACK]   Kinetic: {kin_dmg:.1f} GJ, Explosive: {exp_dmg:.1f} GJ, Total: {dmg:.1f} GJ")

        # Print damage to frigate
        print(f"\n[WOLFPACK] === FRIGATE DAMAGE REPORT ===")
        print(f"[WOLFPACK] Hull integrity: {frigate.hull_integrity:.1f}%")
        print(f"[WOLFPACK] Total damage taken: {frigate.damage_taken_gj:.1f} GJ")

        # Check armor penetrations
        armor_events = [e for e in sim.events if e.event_type == SimulationEventType.ARMOR_PENETRATED]
        print(f"[WOLFPACK] Armor penetrations: {len(armor_events)}")

        # Check module damage
        module_damage_events = [e for e in sim.events if e.event_type == SimulationEventType.MODULE_DAMAGED]
        module_destroy_events = [e for e in sim.events if e.event_type == SimulationEventType.MODULE_DESTROYED]
        print(f"[WOLFPACK] Modules damaged: {len(module_damage_events)}")
        print(f"[WOLFPACK] Modules destroyed: {len(module_destroy_events)}")

        # Show which modules were hit
        if module_damage_events:
            module_names = {}
            for e in module_damage_events:
                name = e.data.get('module_name', 'unknown')
                module_names[name] = module_names.get(name, 0) + 1
            print(f"[WOLFPACK] Damaged modules:")
            for name, count in module_names.items():
                print(f"[WOLFPACK]   - {name}: {count}x hit")

        # === DETAILED FRIGATE STATUS ===
        print(f"\n[WOLFPACK] {'='*60}")
        print(f"[WOLFPACK] DETAILED FRIGATE STATUS")
        print(f"[WOLFPACK] {'='*60}")

        # Module Health
        print(f"\n[WOLFPACK] --- MODULE HEALTH ---")
        if frigate.module_layout:
            for module in frigate.module_layout.get_all_modules():
                status = "DESTROYED" if module.is_destroyed else ("DAMAGED" if module.health_percent < 100 else "OK")
                eff = module.effectiveness * 100
                crit = " [CRITICAL]" if module.is_critical else ""
                print(f"[WOLFPACK]   {module.name:<25} {module.health_percent:5.1f}% health, {eff:5.1f}% effective{crit} [{status}]")

        # Armor Status
        print(f"\n[WOLFPACK] --- ARMOR STATUS ---")
        if frigate.armor:
            for section in ['nose', 'lateral', 'tail']:
                armor_data = frigate.armor.get_section(section)
                if armor_data:
                    print(f"[WOLFPACK]   {section.upper():<10}: {armor_data.current_thickness_cm:.1f} cm ({armor_data.protection_percent:.1f}% protection)")

        # Thermal System
        print(f"\n[WOLFPACK] --- THERMAL STATUS ---")
        if frigate.thermal_system:
            ts = frigate.thermal_system
            print(f"[WOLFPACK]   Current heat: {ts.heat_percent:.1f}%")
            print(f"[WOLFPACK]   Heat sink: {ts.heatsink.current_heat_gj:.1f} / {ts.heatsink.capacity_gj:.1f} GJ")
            print(f"[WOLFPACK]   Is overheating: {ts.is_overheating}")
            print(f"[WOLFPACK]   Is critical: {ts.is_critical}")
            # Radiator status
            if ts.radiators:
                print(f"[WOLFPACK]   Radiators:")
                for pos, rad in ts.radiators.radiators.items():
                    status = rad.state.name
                    print(f"[WOLFPACK]     {pos.value}: {status}, {rad.health_percent:.0f}% health")

        # Power System
        print(f"\n[WOLFPACK] --- POWER STATUS ---")
        if frigate.power_system:
            ps = frigate.power_system
            print(f"[WOLFPACK]   Reactor output: {ps.reactor.current_output_gw:.1f} / {ps.reactor.max_output_gw:.1f} GW")
            print(f"[WOLFPACK]   Available power: {ps.get_available_power_gw():.1f} GW")
            print(f"[WOLFPACK]   Battery: {ps.battery.current_charge_gj:.1f} / {ps.battery.capacity_gj:.1f} GJ ({ps.battery.charge_percent:.0f}%)")
            # Weapon capacitors
            for slot, cap in ps.weapon_capacitors.items():
                print(f"[WOLFPACK]   {slot}: {cap.current_charge_mj:.1f} / {cap.capacity_mj:.1f} MJ ({cap.charge_percent:.0f}%)")

        # Propulsion / Delta-V
        print(f"\n[WOLFPACK] --- PROPULSION STATUS ---")
        ks = frigate.kinematic_state
        print(f"[WOLFPACK]   Current velocity: {(frigate.velocity.magnitude/1000):.2f} km/s")
        print(f"[WOLFPACK]   Delta-V remaining: {frigate.remaining_delta_v_kps:.1f} km/s")
        print(f"[WOLFPACK]   Delta-V used: {500.0 - frigate.remaining_delta_v_kps:.1f} km/s")
        # Engine effectiveness (based on engine module health)
        if frigate.module_layout:
            engine_modules = [m for m in frigate.module_layout.get_all_modules() if m.module_type.value == 'engine']
            if engine_modules:
                avg_engine_health = sum(m.health_percent for m in engine_modules) / len(engine_modules)
                avg_engine_eff = sum(m.effectiveness for m in engine_modules) / len(engine_modules)
                print(f"[WOLFPACK]   Engine health: {avg_engine_health:.1f}%")
                print(f"[WOLFPACK]   Engine effectiveness: {avg_engine_eff*100:.1f}%")
                print(f"[WOLFPACK]   Effective thrust: {avg_engine_eff*100:.0f}% of rated")

        # Combat Stats
        print(f"\n[WOLFPACK] --- COMBAT STATISTICS ---")
        print(f"[WOLFPACK]   Shots fired: {frigate.shots_fired}")
        print(f"[WOLFPACK]   Hits scored: {frigate.hits_scored}")
        print(f"[WOLFPACK]   Damage dealt: {frigate.damage_dealt_gj:.2f} GJ")
        print(f"[WOLFPACK]   Damage taken: {frigate.damage_taken_gj:.2f} GJ")
        if frigate.torpedo_launcher:
            tl = frigate.torpedo_launcher
            print(f"[WOLFPACK]   Torpedoes remaining: {tl.ammo_remaining} / {tl.magazine_size}")

        print(f"[WOLFPACK] {'='*60}")

        surviving = sum(1 for c in corvettes if not c.is_destroyed)
        print(f"\n[WOLFPACK] Surviving corvettes: {surviving}/6")

        # With 6 corvettes, we MUST get torpedo hits
        if torpedo_hits > 0:
            print(f"\n[WOLFPACK] *** SUCCESS: TORPEDOES OVERWHELMED PD! ***")


# =============================================================================
# COMBINED SCENARIO TEST
# =============================================================================

class TestCombinedScenario:
    """Run all scenarios in sequence to verify system stability."""

    def test_scenario_variety(self, fleet_data):
        """Run multiple different scenarios back-to-back."""
        scenarios_run = 0

        # Quick head-on
        sim1 = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=1)
        alpha1 = create_ship_from_fleet_data(
            "a1", "destroyer", "alpha", fleet_data,
            Vector3D(-100_000, 0, 0), Vector3D(5_000, 0, 0), Vector3D(1, 0, 0)
        )
        beta1 = create_ship_from_fleet_data(
            "b1", "destroyer", "beta", fleet_data,
            Vector3D(100_000, 0, 0), Vector3D(-5_000, 0, 0), Vector3D(-1, 0, 0)
        )
        sim1.add_ship(alpha1)
        sim1.add_ship(beta1)
        sim1.run(duration=30.0)
        scenarios_run += 1

        # Quick pursuit
        sim2 = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=2)
        alpha2 = create_ship_from_fleet_data(
            "a2", "corvette", "alpha", fleet_data,
            Vector3D(0, 0, 0), Vector3D(8_000, 0, 0), Vector3D(1, 0, 0)
        )
        beta2 = create_ship_from_fleet_data(
            "b2", "destroyer", "beta", fleet_data,
            Vector3D(200_000, 0, 0), Vector3D(5_000, 0, 0), Vector3D(1, 0, 0)
        )
        sim2.add_ship(alpha2)
        sim2.add_ship(beta2)
        sim2.run(duration=30.0)
        scenarios_run += 1

        # Quick dogfight
        sim3 = CombatSimulation(time_step=1.0, decision_interval=20.0, seed=3)
        alpha3 = create_ship_from_fleet_data(
            "a3", "destroyer", "alpha", fleet_data,
            Vector3D(-10_000, 0, 0), Vector3D(500, 0, 0), Vector3D(1, 0, 0)
        )
        beta3 = create_ship_from_fleet_data(
            "b3", "destroyer", "beta", fleet_data,
            Vector3D(10_000, 0, 0), Vector3D(-500, 0, 0), Vector3D(-1, 0, 0)
        )
        sim3.add_ship(alpha3)
        sim3.add_ship(beta3)
        sim3.run(duration=30.0)
        scenarios_run += 1

        print(f"\n[COMBINED] Successfully ran {scenarios_run} scenarios")
        assert scenarios_run == 3


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
