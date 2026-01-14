"""
Battle Recorder - Records all battle events for replay functionality.

Captures:
- Ship states at each checkpoint
- All shots fired with probability, energy, angle
- All hits with damage, armor ablation, penetration
- All messages/shittalk between captains
- All maneuver and weapon commands
- Complete battle metadata
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum


class EventType(str, Enum):
    """Types of battle events."""
    # Battle lifecycle
    BATTLE_START = "battle_start"
    BATTLE_END = "battle_end"
    CHECKPOINT = "checkpoint"

    # Ship actions
    MANEUVER = "maneuver"
    WEAPONS_ORDER = "weapons_order"
    RADIATOR_CHANGE = "radiator_change"

    # Combat events
    SHOT_FIRED = "shot_fired"
    HIT = "hit"
    MISS = "miss"
    ARMOR_DAMAGE = "armor_damage"
    HULL_DAMAGE = "hull_damage"
    PENETRATION = "penetration"
    MODULE_DAMAGED = "module_damaged"
    MODULE_DESTROYED = "module_destroyed"

    # Communication
    MESSAGE = "message"
    SURRENDER = "surrender"
    DRAW_PROPOSAL = "draw_proposal"

    # Admiral/Captain interactions
    ADMIRAL_DIRECTIVE = "admiral_directive"
    ADMIRAL_ORDER = "admiral_order"
    CAPTAIN_DECISION = "captain_decision"
    CAPTAIN_ADMIRAL_DISCUSSION = "captain_admiral_discussion"

    # Ship status
    SHIP_STATE = "ship_state"
    THERMAL_WARNING = "thermal_warning"
    WEAPON_OVERHEAT = "weapon_overheat"


@dataclass
class BattleEvent:
    """A single recorded battle event."""
    timestamp: float  # Simulation time in seconds
    event_type: str
    ship_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "ship_id": self.ship_id,
            "data": self.data,
        }


@dataclass
class BattleRecording:
    """Complete recording of a battle."""
    # Metadata
    recording_version: str = "2.1"  # Bumped for fleet battle support
    recorded_at: str = ""
    is_fleet_battle: bool = False
    battle_name: str = ""

    # Battle config (1v1 mode)
    alpha_model: str = ""
    beta_model: str = ""
    alpha_name: str = ""
    beta_name: str = ""
    alpha_ship: str = ""
    beta_ship: str = ""
    alpha_personality: str = ""
    beta_personality: str = ""
    initial_distance_km: float = 0.0
    time_limit_s: float = 0.0
    max_checkpoints: int = 0
    unlimited_mode: bool = False

    # Ship specs (for replay - 1v1 mode)
    alpha_specs: Dict[str, Any] = field(default_factory=dict)
    beta_specs: Dict[str, Any] = field(default_factory=dict)

    # Fleet battle config
    alpha_fleet: Dict[str, Any] = field(default_factory=dict)  # Admiral + ships
    beta_fleet: Dict[str, Any] = field(default_factory=dict)

    # Events (checkpoints, commands, hits, etc.)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Simulation trace - per-step tracking of all objects
    # Each frame: {"t": float, "ships": {...}, "projectiles": [...], "torpedoes": [...]}
    sim_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Result
    winner: Optional[str] = None
    result_reason: str = ""
    duration_s: float = 0.0
    total_checkpoints: int = 0
    alpha_ships_remaining: int = 0
    beta_ships_remaining: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BattleRecorder:
    """
    Records all battle events for replay functionality.

    Usage:
        recorder = BattleRecorder()
        recorder.start_recording(config, alpha_config, beta_config)

        # During battle:
        recorder.record_shot_fired(...)
        recorder.record_hit(...)
        recorder.record_message(...)

        # After battle:
        recorder.end_recording(result)
        recorder.save("battle_2024_01_10.json")
    """

    def __init__(self):
        self.recording = BattleRecording()
        self.events: List[BattleEvent] = []
        self._is_recording = False

    def start_recording(
        self,
        battle_config: Any,
        alpha_config: Any,
        beta_config: Any,
        alpha_ship: Any = None,
        beta_ship: Any = None,
    ) -> None:
        """Start recording a new battle."""
        self._is_recording = True
        self.events = []

        self.recording = BattleRecording(
            recorded_at=datetime.now().isoformat(),
            alpha_model=alpha_config.model,
            beta_model=beta_config.model,
            alpha_name=alpha_config.name,
            beta_name=beta_config.name,
            alpha_ship=alpha_config.ship_name,
            beta_ship=beta_config.ship_name,
            alpha_personality=alpha_config.personality.value,
            beta_personality=beta_config.personality.value,
            initial_distance_km=battle_config.initial_distance_km,
            time_limit_s=battle_config.time_limit_s,
            max_checkpoints=battle_config.max_checkpoints,
            unlimited_mode=getattr(battle_config, 'unlimited_mode', False),
        )

        # Record ship specs if available
        if alpha_ship:
            self.recording.alpha_specs = self._extract_ship_specs(alpha_ship)
        if beta_ship:
            self.recording.beta_specs = self._extract_ship_specs(beta_ship)

        self._record_event(BattleEvent(
            timestamp=0.0,
            event_type=EventType.BATTLE_START,
            data={
                "distance_km": battle_config.initial_distance_km,
                "alpha": alpha_config.name,
                "beta": beta_config.name,
            }
        ))

    def start_fleet_recording(
        self,
        fleet_config: Any,
        battle_config: Any,
        alpha_ships: Dict[str, Any],
        beta_ships: Dict[str, Any],
        alpha_admiral: Any = None,
        beta_admiral: Any = None,
    ) -> None:
        """Start recording a fleet battle."""
        self._is_recording = True
        self.events = []

        self.recording = BattleRecording(
            recorded_at=datetime.now().isoformat(),
            is_fleet_battle=True,
            battle_name=fleet_config.battle_name,
            initial_distance_km=fleet_config.initial_distance_km,
            time_limit_s=fleet_config.time_limit_s,
            max_checkpoints=getattr(fleet_config, 'max_checkpoints', battle_config.max_checkpoints),
        )

        # Record alpha fleet
        alpha_fleet_data = {
            "admiral": None,
            "ships": [],
        }
        if alpha_admiral:
            alpha_fleet_data["admiral"] = {
                "name": alpha_admiral.name,
                "model": alpha_admiral.config.model,
            }
        for ship_id, ship in alpha_ships.items():
            ship_data = self._extract_ship_specs(ship)
            ship_data["ship_name"] = getattr(ship, 'name', ship_id)
            alpha_fleet_data["ships"].append(ship_data)
        self.recording.alpha_fleet = alpha_fleet_data

        # Record beta fleet
        beta_fleet_data = {
            "admiral": None,
            "ships": [],
        }
        if beta_admiral:
            beta_fleet_data["admiral"] = {
                "name": beta_admiral.name,
                "model": beta_admiral.config.model,
            }
        for ship_id, ship in beta_ships.items():
            ship_data = self._extract_ship_specs(ship)
            ship_data["ship_name"] = getattr(ship, 'name', ship_id)
            beta_fleet_data["ships"].append(ship_data)
        self.recording.beta_fleet = beta_fleet_data

        # For backward compatibility, use first ships for alpha_model/beta_model
        if alpha_ships:
            first_alpha = list(alpha_ships.values())[0]
            self.recording.alpha_ship = getattr(first_alpha, 'name', 'Alpha Fleet')
        if beta_ships:
            first_beta = list(beta_ships.values())[0]
            self.recording.beta_ship = getattr(first_beta, 'name', 'Beta Fleet')

        # Get model names for filename generation
        if alpha_admiral:
            self.recording.alpha_model = alpha_admiral.config.model
        elif alpha_fleet_data["ships"]:
            # Use first captain's model
            self.recording.alpha_model = fleet_config.alpha_fleet.ships[0].model
        if beta_admiral:
            self.recording.beta_model = beta_admiral.config.model
        elif beta_fleet_data["ships"]:
            self.recording.beta_model = fleet_config.beta_fleet.ships[0].model

        self._record_event(BattleEvent(
            timestamp=0.0,
            event_type=EventType.BATTLE_START,
            data={
                "is_fleet_battle": True,
                "battle_name": fleet_config.battle_name,
                "distance_km": fleet_config.initial_distance_km,
                "alpha_ships": len(alpha_ships),
                "beta_ships": len(beta_ships),
                "alpha_admiral": alpha_admiral.name if alpha_admiral else None,
                "beta_admiral": beta_admiral.name if beta_admiral else None,
            }
        ))

    def _extract_ship_specs(self, ship: Any) -> Dict[str, Any]:
        """Extract ship specifications for replay."""
        specs = {
            "ship_id": ship.ship_id,
            "hull_integrity": 100.0,
        }

        # Armor
        if hasattr(ship, 'armor') and ship.armor:
            specs["armor"] = {}
            for section_name in ["nose", "lateral", "tail"]:
                section = ship.armor.get_section(section_name)
                if section:
                    specs["armor"][section_name] = {
                        "thickness_cm": section.thickness_cm,
                        "material": getattr(section, 'material', 'titanium'),
                    }

        # Weapons
        if hasattr(ship, 'weapons'):
            specs["weapons"] = {}
            for slot, weapon in ship.weapons.items():
                specs["weapons"][slot] = {
                    "type": type(weapon).__name__,
                    "damage_gj": getattr(weapon, 'damage_gj', 0),
                }

        # Propulsion
        specs["max_acceleration_g"] = getattr(ship, 'max_acceleration_g', 2.0)
        specs["delta_v_budget_kps"] = getattr(ship, 'delta_v_budget_kps', 500)

        return specs

    def _record_event(self, event: BattleEvent) -> None:
        """Record a single event."""
        if self._is_recording:
            self.events.append(event)

    def record_checkpoint(
        self,
        timestamp: float,
        checkpoint_num: int,
        alpha_state: Dict[str, Any],
        beta_state: Dict[str, Any],
        distance_km: float,
    ) -> None:
        """Record a checkpoint with full ship states."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.CHECKPOINT,
            data={
                "checkpoint": checkpoint_num,
                "distance_km": distance_km,
                "alpha": alpha_state,
                "beta": beta_state,
            }
        ))

    def record_ship_state(
        self,
        timestamp: float,
        ship_id: str,
        position: tuple,
        velocity: tuple,
        forward: tuple,
        hull_integrity: float,
        heat_percent: float,
        armor_nose_cm: float,
        armor_lateral_cm: float,
        armor_tail_cm: float,
        delta_v_remaining: float,
        radiators_extended: bool,
    ) -> None:
        """Record complete ship state."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.SHIP_STATE,
            ship_id=ship_id,
            data={
                "position": position,
                "velocity": velocity,
                "forward": forward,
                "hull_integrity": hull_integrity,
                "heat_percent": heat_percent,
                "armor": {
                    "nose_cm": armor_nose_cm,
                    "lateral_cm": armor_lateral_cm,
                    "tail_cm": armor_tail_cm,
                },
                "delta_v_remaining": delta_v_remaining,
                "radiators_extended": radiators_extended,
            }
        ))

    def record_maneuver(
        self,
        timestamp: float,
        ship_id: str,
        maneuver_type: str,
        throttle: float,
        target_id: Optional[str] = None,
    ) -> None:
        """Record a maneuver command."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.MANEUVER,
            ship_id=ship_id,
            data={
                "maneuver_type": maneuver_type,
                "throttle": throttle,
                "target_id": target_id,
            }
        ))

    def record_weapons_order(
        self,
        timestamp: float,
        ship_id: str,
        weapon_slot: str,
        firing_mode: str,
        target_id: Optional[str] = None,
        min_hit_probability: float = 0.0,
        max_range_km: float = 500.0,
    ) -> None:
        """Record a weapons order."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.WEAPONS_ORDER,
            ship_id=ship_id,
            data={
                "weapon_slot": weapon_slot,
                "firing_mode": firing_mode,
                "target_id": target_id,
                "min_hit_probability": min_hit_probability,
                "max_range_km": max_range_km,
            }
        ))

    def record_shot_fired(
        self,
        timestamp: float,
        shooter_id: str,
        target_id: str,
        weapon_slot: str,
        hit_probability: float,
        distance_km: float,
        projectile_energy_gj: float,
        muzzle_velocity_kps: float,
        weapon_name: str = "unknown",
        eta_s: float = 0.0,
    ) -> None:
        """Record a shot being fired."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.SHOT_FIRED,
            ship_id=shooter_id,
            data={
                "target_id": target_id,
                "weapon_slot": weapon_slot,
                "weapon_name": weapon_name,
                "hit_probability": hit_probability,
                "distance_km": distance_km,
                "eta_s": eta_s,
                "projectile_energy_gj": projectile_energy_gj,
                "muzzle_velocity_kps": muzzle_velocity_kps,
            }
        ))

    def record_hit(
        self,
        timestamp: float,
        shooter_id: str,
        target_id: str,
        weapon_slot: str,
        hit_location: str,
        impact_angle_deg: float,
        kinetic_energy_gj: float,
        armor_ablation_cm: float,
        armor_remaining_cm: float,
        damage_to_hull_gj: float,
        penetrated: bool,
        critical_hit: bool = False,
        flight_time_s: float = 0.0,
    ) -> None:
        """Record a hit with full damage details."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.HIT,
            ship_id=target_id,
            data={
                "shooter_id": shooter_id,
                "weapon_slot": weapon_slot,
                "hit_location": hit_location,
                "impact_angle_deg": impact_angle_deg,
                "kinetic_energy_gj": kinetic_energy_gj,
                "armor_ablation_cm": armor_ablation_cm,
                "armor_remaining_cm": armor_remaining_cm,
                "damage_to_hull_gj": damage_to_hull_gj,
                "penetrated": penetrated,
                "critical_hit": critical_hit,
                "flight_time_s": flight_time_s,
            }
        ))

    def record_miss(
        self,
        timestamp: float,
        shooter_id: str,
        target_id: str,
        weapon_slot: str,
        hit_probability: float,
        distance_km: float,
        flight_time_s: float = 0.0,
    ) -> None:
        """Record a miss."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.MISS,
            ship_id=shooter_id,
            data={
                "target_id": target_id,
                "weapon_slot": weapon_slot,
                "hit_probability": hit_probability,
                "distance_km": distance_km,
                "flight_time_s": flight_time_s,
            }
        ))

    def record_armor_damage(
        self,
        timestamp: float,
        ship_id: str,
        location: str,
        ablation_cm: float,
        remaining_cm: float,
        chipping_fraction: float,
    ) -> None:
        """Record armor ablation."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.ARMOR_DAMAGE,
            ship_id=ship_id,
            data={
                "location": location,
                "ablation_cm": ablation_cm,
                "remaining_cm": remaining_cm,
                "chipping_fraction": chipping_fraction,
            }
        ))

    def record_module_damaged(
        self,
        timestamp: float,
        ship_id: str,
        module_name: str,
        damage_gj: float,
        destroyed: bool,
    ) -> None:
        """Record damage to a ship module."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.MODULE_DAMAGED,
            ship_id=ship_id,
            data={
                "module_name": module_name,
                "damage_gj": damage_gj,
                "destroyed": destroyed,
            }
        ))

    def record_module_destroyed(
        self,
        timestamp: float,
        ship_id: str,
        module_name: str,
    ) -> None:
        """Record destruction of a ship module."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.MODULE_DESTROYED,
            ship_id=ship_id,
            data={
                "module_name": module_name,
            }
        ))

    def record_message(
        self,
        timestamp: float,
        sender_id: str,
        sender_name: str,
        ship_name: str,
        message: str,
    ) -> None:
        """Record a message between captains."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.MESSAGE,
            ship_id=sender_id,
            data={
                "sender_name": sender_name,
                "ship_name": ship_name,
                "message": message,
            }
        ))

    def record_surrender(
        self,
        timestamp: float,
        ship_id: str,
        captain_name: str,
    ) -> None:
        """Record a surrender."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.SURRENDER,
            ship_id=ship_id,
            data={"captain_name": captain_name}
        ))

    def record_draw_proposal(
        self,
        timestamp: float,
        ship_id: str,
        captain_name: str,
    ) -> None:
        """Record a draw proposal."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.DRAW_PROPOSAL,
            ship_id=ship_id,
            data={"captain_name": captain_name}
        ))

    def record_radiator_change(
        self,
        timestamp: float,
        ship_id: str,
        extended: bool,
    ) -> None:
        """Record radiator state change."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.RADIATOR_CHANGE,
            ship_id=ship_id,
            data={"extended": extended}
        ))

    def record_thermal_warning(
        self,
        timestamp: float,
        ship_id: str,
        heat_percent: float,
        is_critical: bool,
    ) -> None:
        """Record thermal warning."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.THERMAL_WARNING,
            ship_id=ship_id,
            data={
                "heat_percent": heat_percent,
                "is_critical": is_critical,
            }
        ))

    def record_admiral_directive(
        self,
        timestamp: float,
        admiral_name: str,
        faction: str,
        directive: str,
    ) -> None:
        """Record fleet-wide directive from admiral."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.ADMIRAL_DIRECTIVE,
            data={
                "admiral_name": admiral_name,
                "faction": faction,
                "directive": directive,
            }
        ))

    def record_admiral_order(
        self,
        timestamp: float,
        admiral_name: str,
        ship_id: str,
        ship_name: str,
        order_text: str,
        priority: str = "NORMAL",
        suggested_target: Optional[str] = None,
    ) -> None:
        """Record order from admiral to specific captain."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.ADMIRAL_ORDER,
            ship_id=ship_id,
            data={
                "admiral_name": admiral_name,
                "ship_name": ship_name,
                "order_text": order_text,
                "priority": priority,
                "suggested_target": suggested_target,
            }
        ))

    def record_captain_decision(
        self,
        timestamp: float,
        ship_id: str,
        captain_name: str,
        ship_name: str,
        maneuver_type: str,
        throttle: float,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        radiators_extended: bool = False,
        acknowledgment: Optional[str] = None,
    ) -> None:
        """Record captain's decision with full context."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.CAPTAIN_DECISION,
            ship_id=ship_id,
            data={
                "captain_name": captain_name,
                "ship_name": ship_name,
                "maneuver_type": maneuver_type,
                "throttle": throttle,
                "target_id": target_id,
                "target_name": target_name,
                "radiators_extended": radiators_extended,
                "acknowledgment": acknowledgment,
            }
        ))

    def record_captain_admiral_discussion(
        self,
        timestamp: float,
        ship_id: str,
        captain_name: str,
        admiral_name: str,
        captain_question: str,
        admiral_response: str,
        exchange_number: int,
    ) -> None:
        """Record discussion exchange between captain and admiral."""
        self._record_event(BattleEvent(
            timestamp=timestamp,
            event_type=EventType.CAPTAIN_ADMIRAL_DISCUSSION,
            ship_id=ship_id,
            data={
                "captain_name": captain_name,
                "admiral_name": admiral_name,
                "captain_question": captain_question,
                "admiral_response": admiral_response,
                "exchange_number": exchange_number,
            }
        ))

    def record_sim_frame(
        self,
        timestamp: float,
        ships: Dict[str, Any],
        projectiles: List[Any],
        torpedoes: List[Any],
    ) -> None:
        """
        Record a single simulation frame with all object states.

        This is called every simulation step to capture detailed trajectory data
        for analysis of hit mechanics, evasion patterns, etc.

        Args:
            timestamp: Simulation time in seconds
            ships: Dict of ship_id -> ship state dict
            projectiles: List of projectile state dicts
            torpedoes: List of torpedo state dicts
        """
        if not self._is_recording:
            return

        frame = {
            "t": round(timestamp, 2),
            "ships": {},
            "projectiles": [],
            "torpedoes": [],
        }

        # Record ship states
        for ship_id, ship in ships.items():
            frame["ships"][ship_id] = {
                "pos": [round(ship["position"][0], 1), round(ship["position"][1], 1), round(ship["position"][2], 1)],
                "vel": [round(ship["velocity"][0], 1), round(ship["velocity"][1], 1), round(ship["velocity"][2], 1)],
                "fwd": [round(ship["forward"][0], 4), round(ship["forward"][1], 4), round(ship["forward"][2], 4)],
                "thrust": round(ship.get("thrust", 0.0), 2),
                "maneuver": ship.get("maneuver", "MAINTAIN"),
                "destroyed": ship.get("is_destroyed", False),
                "hull": ship.get("hull_pct", 100.0),
            }

        # Record projectile states (coilgun slugs)
        for proj in projectiles:
            frame["projectiles"].append({
                "id": proj["id"],
                "pos": [round(proj["position"][0], 1), round(proj["position"][1], 1), round(proj["position"][2], 1)],
                "vel": [round(proj["velocity"][0], 1), round(proj["velocity"][1], 1), round(proj["velocity"][2], 1)],
                "mass_kg": round(proj["mass_kg"], 2),
                "source": proj["source_ship_id"],
                "target": proj["target_ship_id"],
                "pd_engaged": proj.get("pd_engaged", False),
                "pd_damage_kg": round(proj.get("pd_ablation_kg", 0.0), 3),
            })

        # Record torpedo states
        for torp in torpedoes:
            frame["torpedoes"].append({
                "id": torp["id"],
                "pos": [round(torp["position"][0], 1), round(torp["position"][1], 1), round(torp["position"][2], 1)],
                "vel": [round(torp["velocity"][0], 1), round(torp["velocity"][1], 1), round(torp["velocity"][2], 1)],
                "source": torp["source_ship_id"],
                "target": torp.get("target_ship_id"),
                "dv_remaining": round(torp.get("dv_remaining_kps", 0.0), 2),
                "pd_heat_j": round(torp.get("heat_absorbed_j", 0.0), 0),
                "disabled": torp.get("is_disabled", False),
            })

        self.recording.sim_trace.append(frame)

    def end_recording(
        self,
        result: Any,
        final_time: float,
    ) -> None:
        """End recording and finalize the battle."""
        self._record_event(BattleEvent(
            timestamp=final_time,
            event_type=EventType.BATTLE_END,
            data={
                "winner": result.winner,
                "reason": result.reason,
                "outcome": result.outcome.value,
            }
        ))

        self.recording.winner = result.winner
        self.recording.result_reason = result.reason
        self.recording.duration_s = result.duration_s
        self.recording.total_checkpoints = result.checkpoints_used
        self.recording.events = [e.to_dict() for e in self.events]

        self._is_recording = False

    def save(self, filepath: str) -> str:
        """Save recording to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(self.recording.to_json())

        return str(path)

    def get_recording(self) -> BattleRecording:
        """Get the current recording."""
        return self.recording


def create_battle_filename(
    alpha_model: str,
    beta_model: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Generate a filename for a battle recording."""
    if timestamp is None:
        timestamp = datetime.now()

    # Clean model names
    def clean_name(name: str) -> str:
        name = name.split("/")[-1]
        name = name.replace("-", "_").replace(".", "_")
        return name[:20]

    alpha_clean = clean_name(alpha_model)
    beta_clean = clean_name(beta_model)
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")

    return f"battle_{alpha_clean}_vs_{beta_clean}_{date_str}.json"
