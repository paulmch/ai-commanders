"""
Battle Report Generator for AI Commanders Space Battle Simulator.

This module generates detailed battle reports from simulation results, including:
- Combat participants with ship details
- Timeline of significant events
- Per-ship statistics (shots, hits, damage, delta-v, heat)
- Multiple output formats (text, detailed text, JSON, markdown)

Integrates with simulation.py events and state for comprehensive battle analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

# Import from simulation module using try/except for compatibility
# These imports are optional - the module can work standalone for report generation
try:
    from .simulation import (
        CombatSimulation, ShipCombatState, SimulationEvent, SimulationEventType
    )
    from .combat import HitLocation
    _SIMULATION_AVAILABLE = True
except ImportError:
    try:
        from simulation import (
            CombatSimulation, ShipCombatState, SimulationEvent, SimulationEventType
        )
        from combat import HitLocation
        _SIMULATION_AVAILABLE = True
    except ImportError:
        # Simulation module not available - define minimal types for standalone use
        _SIMULATION_AVAILABLE = False
        CombatSimulation = None
        ShipCombatState = None
        SimulationEvent = None

        class SimulationEventType(Enum):
            """Minimal event types for standalone operation."""
            PROJECTILE_LAUNCHED = "projectile_launched"
            PROJECTILE_IMPACT = "projectile_impact"
            PROJECTILE_MISS = "projectile_miss"
            TORPEDO_LAUNCHED = "torpedo_launched"
            TORPEDO_IMPACT = "torpedo_impact"
            TORPEDO_INTERCEPTED = "torpedo_intercepted"
            TORPEDO_FUEL_EXHAUSTED = "torpedo_fuel_exhausted"
            MANEUVER_STARTED = "maneuver_started"
            MANEUVER_COMPLETED = "maneuver_completed"
            DAMAGE_TAKEN = "damage_taken"
            MODULE_DAMAGED = "module_damaged"
            MODULE_DESTROYED = "module_destroyed"
            SHIP_DESTROYED = "ship_destroyed"
            ARMOR_PENETRATED = "armor_penetrated"
            THERMAL_WARNING = "thermal_warning"
            THERMAL_CRITICAL = "thermal_critical"
            RADIATOR_EXTENDED = "radiator_extended"
            RADIATOR_RETRACTED = "radiator_retracted"
            RADIATOR_DAMAGED = "radiator_damaged"
            DECISION_POINT_REACHED = "decision_point_reached"
            COMMAND_ISSUED = "command_issued"
            SIMULATION_STARTED = "simulation_started"
            SIMULATION_ENDED = "simulation_ended"

        class HitLocation(Enum):
            """Hit locations for armor."""
            NOSE = "nose"
            LATERAL = "lateral"
            TAIL = "tail"


# =============================================================================
# CONSTANTS
# =============================================================================

# Line widths for report formatting
REPORT_WIDTH = 65
SEPARATOR_CHAR = "="
SUB_SEPARATOR_CHAR = "-"


class BattleOutcome(Enum):
    """Possible battle outcomes."""
    ALPHA_VICTORY = "ALPHA VICTORY"
    BRAVO_VICTORY = "BRAVO VICTORY"
    DRAW = "DRAW"
    ONGOING = "ONGOING"
    MUTUAL_DESTRUCTION = "MUTUAL DESTRUCTION"


# =============================================================================
# SHIP BATTLE STATS
# =============================================================================

@dataclass
class ArmorSectionState:
    """
    State of an armor section at battle end.

    Attributes:
        location: Which section (nose, lateral, tail).
        initial_thickness_cm: Starting armor thickness.
        final_thickness_cm: Remaining armor thickness.
        protection_percent: Final protection percentage.
    """
    location: str
    initial_thickness_cm: float
    final_thickness_cm: float
    protection_percent: float

    @property
    def damage_percent(self) -> float:
        """Calculate percentage of armor lost."""
        if self.initial_thickness_cm <= 0:
            return 0.0
        return ((self.initial_thickness_cm - self.final_thickness_cm) /
                self.initial_thickness_cm) * 100


@dataclass
class ShipBattleStats:
    """
    Combat statistics for a single ship.

    Attributes:
        ship_id: Unique identifier.
        ship_name: Display name (derived from ship_id).
        ship_type: Ship class (destroyer, cruiser, etc.).
        faction: Which side the ship belongs to.
        coilgun_shots_fired: Number of coilgun rounds fired.
        torpedo_shots_fired: Number of torpedoes launched.
        coilgun_hits: Number of coilgun hits scored.
        torpedo_hits: Number of torpedo hits scored.
        damage_dealt_gj: Total damage dealt in gigajoules.
        damage_received_gj: Total damage received in gigajoules.
        delta_v_expended_kps: Delta-v consumed in km/s.
        initial_delta_v_kps: Starting delta-v budget.
        peak_heat_percent: Maximum heat level reached.
        modules_destroyed: List of destroyed module names.
        final_armor_state: Armor state per section at battle end.
        final_hull_integrity: Overall hull integrity percentage.
        is_destroyed: Whether ship was destroyed.
        kill_credit_to: Ship that destroyed this one (if destroyed).
    """
    ship_id: str
    ship_name: str
    ship_type: str
    faction: str

    # Weapons fired
    coilgun_shots_fired: int = 0
    torpedo_shots_fired: int = 0

    # Hits scored
    coilgun_hits: int = 0
    torpedo_hits: int = 0

    # Damage
    damage_dealt_gj: float = 0.0
    damage_received_gj: float = 0.0

    # Delta-v
    delta_v_expended_kps: float = 0.0
    initial_delta_v_kps: float = 0.0

    # Thermal
    peak_heat_percent: float = 0.0

    # Modules and armor
    modules_destroyed: list[str] = field(default_factory=list)
    final_armor_state: list[ArmorSectionState] = field(default_factory=list)
    final_hull_integrity: float = 100.0

    # Final status
    is_destroyed: bool = False
    kill_credit_to: Optional[str] = None

    @property
    def total_shots_fired(self) -> int:
        """Total shots fired (coilgun + torpedo)."""
        return self.coilgun_shots_fired + self.torpedo_shots_fired

    @property
    def total_hits(self) -> int:
        """Total hits scored (coilgun + torpedo)."""
        return self.coilgun_hits + self.torpedo_hits

    @property
    def hit_rate(self) -> float:
        """Overall hit rate as percentage."""
        if self.total_shots_fired == 0:
            return 0.0
        return (self.total_hits / self.total_shots_fired) * 100

    @property
    def delta_v_remaining_kps(self) -> float:
        """Remaining delta-v budget."""
        return max(0.0, self.initial_delta_v_kps - self.delta_v_expended_kps)

    @property
    def final_status(self) -> str:
        """Human-readable final status."""
        if self.is_destroyed:
            return "DESTROYED"
        elif self.final_hull_integrity < 25:
            return "CRITICAL"
        elif self.final_hull_integrity < 50:
            return "HEAVY DAMAGE"
        elif self.final_hull_integrity < 75:
            return "MODERATE DAMAGE"
        elif self.final_hull_integrity < 100:
            return "LIGHT DAMAGE"
        else:
            return "OPERATIONAL"


# =============================================================================
# EVENT TIMELINE
# =============================================================================

class TimelineEventCategory(Enum):
    """Categories for timeline events."""
    BATTLE_START = "battle_start"
    BATTLE_END = "battle_end"
    WEAPON_FIRED = "weapon_fired"
    HIT = "hit"
    MANEUVER = "maneuver"
    THERMAL = "thermal"
    CRITICAL = "critical"
    DESTRUCTION = "destruction"


@dataclass
class TimelineEvent:
    """
    A significant event in the battle timeline.

    Attributes:
        timestamp: When the event occurred (seconds).
        category: Type of event.
        ship_id: Ship that initiated/experienced the event.
        target_id: Target ship (if applicable).
        description: Human-readable description.
        data: Additional event data.
    """
    timestamp: float
    category: TimelineEventCategory
    ship_id: Optional[str]
    target_id: Optional[str]
    description: str
    data: dict = field(default_factory=dict)

    def format_timestamp(self) -> str:
        """Format timestamp as MM:SS."""
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class EventTimeline:
    """
    Chronological timeline of battle events.

    Attributes:
        events: List of timeline events in chronological order.
    """
    events: list[TimelineEvent] = field(default_factory=list)

    def add_event(
        self,
        timestamp: float,
        category: TimelineEventCategory,
        description: str,
        ship_id: Optional[str] = None,
        target_id: Optional[str] = None,
        data: Optional[dict] = None
    ) -> None:
        """Add an event to the timeline."""
        event = TimelineEvent(
            timestamp=timestamp,
            category=category,
            ship_id=ship_id,
            target_id=target_id,
            description=description,
            data=data or {}
        )
        self.events.append(event)

    def get_events_by_category(
        self, category: TimelineEventCategory
    ) -> list[TimelineEvent]:
        """Get all events of a specific category."""
        return [e for e in self.events if e.category == category]

    def get_events_for_ship(self, ship_id: str) -> list[TimelineEvent]:
        """Get all events involving a specific ship."""
        return [
            e for e in self.events
            if e.ship_id == ship_id or e.target_id == ship_id
        ]

    def get_critical_events(self) -> list[TimelineEvent]:
        """Get critical events (first blood, armor breach, destruction)."""
        critical_categories = {
            TimelineEventCategory.CRITICAL,
            TimelineEventCategory.DESTRUCTION
        }
        return [e for e in self.events if e.category in critical_categories]


# =============================================================================
# BATTLE REPORT
# =============================================================================

@dataclass
class BattleReport:
    """
    Comprehensive battle summary from simulation results.

    Attributes:
        battle_name: Name/description of the battle.
        duration_seconds: Total battle duration.
        outcome: Battle outcome (victory, draw, etc.).
        winning_faction: Faction that won (if any).
        participants: Ship stats for all participants.
        timeline: Chronological event timeline.
        metrics: Overall engagement metrics.
    """
    battle_name: str
    duration_seconds: float
    outcome: BattleOutcome
    winning_faction: Optional[str]
    participants: dict[str, ShipBattleStats]
    timeline: EventTimeline
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_formatted(self) -> str:
        """Format duration as Xm YYs."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}m {seconds:02d}s"

    def get_participants_by_faction(self, faction: str) -> list[ShipBattleStats]:
        """Get all participants belonging to a faction."""
        return [p for p in self.participants.values() if p.faction == faction]

    def get_factions(self) -> list[str]:
        """Get unique factions in the battle."""
        return list(set(p.faction for p in self.participants.values()))

    # -------------------------------------------------------------------------
    # Text Output Format
    # -------------------------------------------------------------------------

    def to_text(self) -> str:
        """
        Generate human-readable battle summary.

        Returns:
            Formatted text report.
        """
        lines = []

        # Header
        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)
        lines.append("BATTLE REPORT".center(REPORT_WIDTH))
        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)

        # Basic info
        lines.append(f"Battle: {self.battle_name}")
        lines.append(f"Duration: {self.duration_formatted}")
        lines.append(f"Outcome: {self.outcome.value}")
        lines.append("")

        # Combatants
        lines.append("COMBATANTS:")
        factions = self.get_factions()
        for faction in sorted(factions):
            faction_ships = self.get_participants_by_faction(faction)
            for ship in faction_ships:
                status = "DESTROYED" if ship.is_destroyed else f"Hull {ship.final_hull_integrity:.0f}%"
                lines.append(
                    f"  {faction.upper()}: {ship.ship_type.title()} \"{ship.ship_name}\" "
                    f"({status})"
                )
        lines.append("")

        # Key timeline events (condensed)
        lines.append("KEY EVENTS:")
        critical_events = self.timeline.get_critical_events()
        # Include first few weapon fires and hits
        weapon_events = [
            e for e in self.timeline.events
            if e.category in (TimelineEventCategory.WEAPON_FIRED, TimelineEventCategory.HIT)
        ][:5]

        all_key_events = sorted(
            critical_events + weapon_events,
            key=lambda e: e.timestamp
        )[:10]

        for event in all_key_events:
            lines.append(f"  {event.format_timestamp()} - {event.description}")
        lines.append("")

        # Statistics table
        lines.append("STATISTICS:")
        factions = sorted(self.get_factions())

        # Header row
        header = "                    "
        for faction in factions:
            header += f"{faction.upper():>12}"
        lines.append(header)

        # Calculate faction totals
        faction_stats = {}
        for faction in factions:
            ships = self.get_participants_by_faction(faction)
            faction_stats[faction] = {
                "shots": sum(s.total_shots_fired for s in ships),
                "hits": sum(s.total_hits for s in ships),
                "damage_dealt": sum(s.damage_dealt_gj for s in ships),
                "damage_taken": sum(s.damage_received_gj for s in ships),
                "delta_v": sum(s.delta_v_expended_kps for s in ships),
                "peak_heat": max((s.peak_heat_percent for s in ships), default=0),
            }

        # Data rows
        stat_rows = [
            ("Shots Fired:", "shots", ""),
            ("Hits:", "hits", ""),
            ("Damage Dealt:", "damage_dealt", " GJ"),
            ("Damage Taken:", "damage_taken", " GJ"),
            ("Delta-V Used:", "delta_v", " km/s"),
            ("Peak Heat:", "peak_heat", "%"),
        ]

        for label, key, suffix in stat_rows:
            row = f"  {label:<18}"
            for faction in factions:
                value = faction_stats[faction][key]
                if isinstance(value, float):
                    row += f"{value:>10.1f}{suffix:>2}"
                else:
                    row += f"{value:>10}{suffix:>2}"
            lines.append(row)

        lines.append("")

        # Final state
        lines.append("FINAL STATE:")
        for faction in sorted(factions):
            for ship in self.get_participants_by_faction(faction):
                if ship.is_destroyed:
                    lines.append(f"  {faction.upper()}: Hull 0%, DESTROYED")
                else:
                    lines.append(
                        f"  {faction.upper()}: Hull {ship.final_hull_integrity:.0f}%, "
                        f"{ship.final_status.lower()}"
                    )

        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Detailed Text Output
    # -------------------------------------------------------------------------

    def to_detailed_text(self) -> str:
        """
        Generate full combat log with all events.

        Returns:
            Detailed text report with complete timeline.
        """
        lines = []

        # Use the basic report as header
        lines.append(self.to_text())
        lines.append("")
        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)
        lines.append("DETAILED COMBAT LOG".center(REPORT_WIDTH))
        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)
        lines.append("")

        # Complete timeline
        lines.append("COMPLETE TIMELINE:")
        lines.append(SUB_SEPARATOR_CHAR * REPORT_WIDTH)

        for event in self.timeline.events:
            ship_tag = f"[{event.ship_id}]" if event.ship_id else ""
            target_tag = f" -> {event.target_id}" if event.target_id else ""
            lines.append(
                f"  {event.format_timestamp()} {ship_tag:<12} {event.description}{target_tag}"
            )

        lines.append("")

        # Per-ship detailed stats
        lines.append(SUB_SEPARATOR_CHAR * REPORT_WIDTH)
        lines.append("PER-SHIP STATISTICS:")
        lines.append(SUB_SEPARATOR_CHAR * REPORT_WIDTH)

        for ship_id, stats in self.participants.items():
            lines.append("")
            lines.append(f"  {stats.ship_name} ({stats.ship_type.title()}) - {stats.faction.upper()}")
            lines.append(f"  " + "-" * 40)
            lines.append(f"    Coilgun shots: {stats.coilgun_shots_fired}")
            lines.append(f"    Torpedo shots: {stats.torpedo_shots_fired}")
            lines.append(f"    Coilgun hits:  {stats.coilgun_hits}")
            lines.append(f"    Torpedo hits:  {stats.torpedo_hits}")
            lines.append(f"    Hit rate:      {stats.hit_rate:.1f}%")
            lines.append(f"    Damage dealt:  {stats.damage_dealt_gj:.2f} GJ")
            lines.append(f"    Damage taken:  {stats.damage_received_gj:.2f} GJ")
            lines.append(f"    Delta-V used:  {stats.delta_v_expended_kps:.1f} km/s")
            lines.append(f"    Delta-V left:  {stats.delta_v_remaining_kps:.1f} km/s")
            lines.append(f"    Peak heat:     {stats.peak_heat_percent:.1f}%")

            if stats.modules_destroyed:
                lines.append(f"    Modules lost:  {', '.join(stats.modules_destroyed)}")

            if stats.final_armor_state:
                lines.append("    Armor state:")
                for armor in stats.final_armor_state:
                    lines.append(
                        f"      {armor.location}: {armor.final_thickness_cm:.1f} cm "
                        f"({armor.protection_percent:.1f}% protection)"
                    )

            lines.append(f"    Final status:  {stats.final_status}")
            if stats.is_destroyed and stats.kill_credit_to:
                lines.append(f"    Destroyed by:  {stats.kill_credit_to}")

        lines.append("")
        lines.append(SEPARATOR_CHAR * REPORT_WIDTH)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # JSON Output
    # -------------------------------------------------------------------------

    def to_json(self) -> str:
        """
        Generate structured JSON report.

        Returns:
            JSON string with complete battle data.
        """
        data = {
            "battle_name": self.battle_name,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.duration_formatted,
            "outcome": self.outcome.value,
            "winning_faction": self.winning_faction,
            "factions": self.get_factions(),
            "participants": {},
            "timeline": [],
            "metrics": self.metrics
        }

        # Participants
        for ship_id, stats in self.participants.items():
            data["participants"][ship_id] = {
                "ship_id": stats.ship_id,
                "ship_name": stats.ship_name,
                "ship_type": stats.ship_type,
                "faction": stats.faction,
                "weapons": {
                    "coilgun_shots_fired": stats.coilgun_shots_fired,
                    "torpedo_shots_fired": stats.torpedo_shots_fired,
                    "coilgun_hits": stats.coilgun_hits,
                    "torpedo_hits": stats.torpedo_hits,
                    "total_shots": stats.total_shots_fired,
                    "total_hits": stats.total_hits,
                    "hit_rate_percent": stats.hit_rate
                },
                "damage": {
                    "dealt_gj": stats.damage_dealt_gj,
                    "received_gj": stats.damage_received_gj
                },
                "propulsion": {
                    "delta_v_expended_kps": stats.delta_v_expended_kps,
                    "delta_v_remaining_kps": stats.delta_v_remaining_kps,
                    "initial_delta_v_kps": stats.initial_delta_v_kps
                },
                "thermal": {
                    "peak_heat_percent": stats.peak_heat_percent
                },
                "final_state": {
                    "hull_integrity_percent": stats.final_hull_integrity,
                    "is_destroyed": stats.is_destroyed,
                    "status": stats.final_status,
                    "kill_credit_to": stats.kill_credit_to,
                    "modules_destroyed": stats.modules_destroyed,
                    "armor_sections": [
                        {
                            "location": a.location,
                            "initial_thickness_cm": a.initial_thickness_cm,
                            "final_thickness_cm": a.final_thickness_cm,
                            "protection_percent": a.protection_percent
                        }
                        for a in stats.final_armor_state
                    ]
                }
            }

        # Timeline
        for event in self.timeline.events:
            data["timeline"].append({
                "timestamp": event.timestamp,
                "timestamp_formatted": event.format_timestamp(),
                "category": event.category.value,
                "ship_id": event.ship_id,
                "target_id": event.target_id,
                "description": event.description,
                "data": event.data
            })

        return json.dumps(data, indent=2)

    # -------------------------------------------------------------------------
    # Markdown Output
    # -------------------------------------------------------------------------

    def to_markdown(self) -> str:
        """
        Generate markdown-formatted report for documentation.

        Returns:
            Markdown string suitable for documentation.
        """
        lines = []

        # Title
        lines.append(f"# Battle Report: {self.battle_name}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Duration:** {self.duration_formatted}")
        lines.append(f"- **Outcome:** {self.outcome.value}")
        if self.winning_faction:
            lines.append(f"- **Victor:** {self.winning_faction.upper()}")
        lines.append("")

        # Combatants
        lines.append("## Combatants")
        lines.append("")

        for faction in sorted(self.get_factions()):
            lines.append(f"### {faction.upper()}")
            lines.append("")
            for ship in self.get_participants_by_faction(faction):
                status = "DESTROYED" if ship.is_destroyed else ship.final_status
                lines.append(f"- **{ship.ship_name}** ({ship.ship_type.title()})")
                lines.append(f"  - Final Status: {status}")
                lines.append(f"  - Hull Integrity: {ship.final_hull_integrity:.0f}%")
            lines.append("")

        # Timeline
        lines.append("## Battle Timeline")
        lines.append("")
        lines.append("| Time | Event |")
        lines.append("|------|-------|")

        for event in self.timeline.events:
            # Escape pipe characters in description
            desc = event.description.replace("|", "\\|")
            lines.append(f"| {event.format_timestamp()} | {desc} |")

        lines.append("")

        # Statistics
        lines.append("## Statistics")
        lines.append("")

        factions = sorted(self.get_factions())

        # Build table header
        header = "| Statistic |"
        separator = "|-----------|"
        for faction in factions:
            header += f" {faction.upper()} |"
            separator += "------:|"

        lines.append(header)
        lines.append(separator)

        # Calculate faction totals
        faction_stats = {}
        for faction in factions:
            ships = self.get_participants_by_faction(faction)
            faction_stats[faction] = {
                "shots": sum(s.total_shots_fired for s in ships),
                "hits": sum(s.total_hits for s in ships),
                "damage_dealt": sum(s.damage_dealt_gj for s in ships),
                "damage_taken": sum(s.damage_received_gj for s in ships),
                "delta_v": sum(s.delta_v_expended_kps for s in ships),
                "peak_heat": max((s.peak_heat_percent for s in ships), default=0),
            }

        # Data rows
        stat_rows = [
            ("Shots Fired", "shots", ""),
            ("Hits", "hits", ""),
            ("Damage Dealt", "damage_dealt", " GJ"),
            ("Damage Taken", "damage_taken", " GJ"),
            ("Delta-V Used", "delta_v", " km/s"),
            ("Peak Heat", "peak_heat", "%"),
        ]

        for label, key, suffix in stat_rows:
            row = f"| {label} |"
            for faction in factions:
                value = faction_stats[faction][key]
                if isinstance(value, float):
                    row += f" {value:.1f}{suffix} |"
                else:
                    row += f" {value}{suffix} |"
            lines.append(row)

        lines.append("")

        # Per-ship details
        lines.append("## Ship Details")
        lines.append("")

        for ship_id, stats in self.participants.items():
            lines.append(f"### {stats.ship_name}")
            lines.append("")
            lines.append(f"- **Type:** {stats.ship_type.title()}")
            lines.append(f"- **Faction:** {stats.faction.upper()}")
            lines.append(f"- **Final Status:** {stats.final_status}")
            lines.append("")
            lines.append("#### Combat Performance")
            lines.append("")
            lines.append(f"- Shots Fired: {stats.total_shots_fired} "
                        f"(Coilgun: {stats.coilgun_shots_fired}, Torpedo: {stats.torpedo_shots_fired})")
            lines.append(f"- Hits Scored: {stats.total_hits} "
                        f"(Coilgun: {stats.coilgun_hits}, Torpedo: {stats.torpedo_hits})")
            lines.append(f"- Hit Rate: {stats.hit_rate:.1f}%")
            lines.append(f"- Damage Dealt: {stats.damage_dealt_gj:.2f} GJ")
            lines.append(f"- Damage Received: {stats.damage_received_gj:.2f} GJ")
            lines.append("")
            lines.append("#### Resource Usage")
            lines.append("")
            lines.append(f"- Delta-V Expended: {stats.delta_v_expended_kps:.1f} km/s")
            lines.append(f"- Delta-V Remaining: {stats.delta_v_remaining_kps:.1f} km/s")
            lines.append(f"- Peak Heat Level: {stats.peak_heat_percent:.1f}%")
            lines.append("")

            if stats.modules_destroyed:
                lines.append("#### Damage Sustained")
                lines.append("")
                lines.append(f"- Modules Destroyed: {', '.join(stats.modules_destroyed)}")
                lines.append("")

        lines.append("---")
        lines.append("*Report generated by AI Commanders Battle Report System*")

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_report_from_simulation(
    simulation: "CombatSimulation",
    battle_name: str = "Engagement"
) -> BattleReport:
    """
    Build a BattleReport from a completed simulation.

    This factory function extracts all relevant data from the simulation
    state and event log to create a comprehensive battle report.

    Args:
        simulation: The completed CombatSimulation instance.
        battle_name: Name/description for the battle.

    Returns:
        A fully populated BattleReport instance.

    Raises:
        RuntimeError: If simulation module is not available.
    """
    if not _SIMULATION_AVAILABLE:
        raise RuntimeError(
            "Simulation module not available. "
            "Cannot create report from simulation object."
        )

    # Determine battle outcome
    outcome, winning_faction = _determine_outcome(simulation)

    # Build ship stats
    participants = {}
    for ship_id, ship in simulation.ships.items():
        stats = _build_ship_stats(ship, simulation)
        participants[ship_id] = stats

    # Build timeline
    timeline = _build_timeline(simulation)

    # Build metrics
    metrics = {
        "total_shots_fired": simulation.metrics.total_shots_fired,
        "total_hits": simulation.metrics.total_hits,
        "total_torpedoes_launched": simulation.metrics.total_torpedoes_launched,
        "total_torpedo_hits": simulation.metrics.total_torpedo_hits,
        "total_damage_dealt": simulation.metrics.total_damage_dealt,
        "ships_destroyed": simulation.metrics.ships_destroyed,
        "hit_rate": simulation.metrics.hit_rate
    }

    return BattleReport(
        battle_name=battle_name,
        duration_seconds=simulation.current_time,
        outcome=outcome,
        winning_faction=winning_faction,
        participants=participants,
        timeline=timeline,
        metrics=metrics
    )


def _determine_outcome(simulation: CombatSimulation) -> tuple[BattleOutcome, Optional[str]]:
    """
    Determine the battle outcome from simulation state.

    Returns:
        Tuple of (outcome, winning_faction).
    """
    # Count surviving ships per faction
    faction_survivors: dict[str, int] = {}
    factions: set[str] = set()

    for ship in simulation.ships.values():
        factions.add(ship.faction)
        if not ship.is_destroyed:
            faction_survivors[ship.faction] = faction_survivors.get(ship.faction, 0) + 1

    # Determine outcome
    surviving_factions = [f for f, count in faction_survivors.items() if count > 0]

    if len(surviving_factions) == 0:
        return BattleOutcome.MUTUAL_DESTRUCTION, None
    elif len(surviving_factions) == 1:
        winning_faction = surviving_factions[0]
        # Use common faction naming convention
        if winning_faction.lower() in ["alpha", "faction_a", "a"]:
            return BattleOutcome.ALPHA_VICTORY, winning_faction
        elif winning_faction.lower() in ["bravo", "faction_b", "b"]:
            return BattleOutcome.BRAVO_VICTORY, winning_faction
        else:
            # Generic victory for named faction
            return BattleOutcome.ALPHA_VICTORY, winning_faction
    else:
        # Multiple factions surviving - check if battle is still running
        if simulation._running:
            return BattleOutcome.ONGOING, None
        else:
            return BattleOutcome.DRAW, None


def _build_ship_stats(
    ship: ShipCombatState,
    simulation: CombatSimulation
) -> ShipBattleStats:
    """
    Build statistics for a single ship from simulation data.

    Args:
        ship: The ship combat state.
        simulation: The simulation instance.

    Returns:
        Populated ShipBattleStats instance.
    """
    # Extract name from ship_id (e.g., "alpha_1" -> "Alpha 1")
    ship_name = ship.ship_id.replace("_", " ").title()

    # Count weapon usage from events
    coilgun_shots = 0
    torpedo_shots = 0
    coilgun_hits = 0
    torpedo_hits = 0
    peak_heat = 0.0
    modules_destroyed: list[str] = []

    for event in simulation.events:
        if event.ship_id == ship.ship_id:
            if event.event_type == SimulationEventType.PROJECTILE_LAUNCHED:
                coilgun_shots += 1
            elif event.event_type == SimulationEventType.TORPEDO_LAUNCHED:
                torpedo_shots += 1
            elif event.event_type == SimulationEventType.THERMAL_WARNING:
                heat = event.data.get("heat_percent", 0)
                peak_heat = max(peak_heat, heat)
            elif event.event_type == SimulationEventType.THERMAL_CRITICAL:
                heat = event.data.get("heat_percent", 0)
                peak_heat = max(peak_heat, heat)

        # Hits scored (where this ship is the source)
        if event.target_id and event.ship_id == ship.ship_id:
            if event.event_type == SimulationEventType.PROJECTILE_IMPACT:
                coilgun_hits += 1
            elif event.event_type == SimulationEventType.TORPEDO_IMPACT:
                torpedo_hits += 1

        # Modules destroyed on this ship
        if event.ship_id == ship.ship_id:
            if event.event_type == SimulationEventType.MODULE_DESTROYED:
                module_name = event.data.get("module_name", "Unknown")
                if module_name not in modules_destroyed:
                    modules_destroyed.append(module_name)

    # Get initial delta-v (if available)
    initial_delta_v = 0.0
    if ship.kinematic_state:
        # Calculate from propellant fraction if available
        initial_delta_v = ship.remaining_delta_v_kps + ship.kinematic_state.delta_v_expended_kps
        if initial_delta_v == 0:
            # Fallback estimate
            initial_delta_v = ship.remaining_delta_v_kps

    # Build armor state
    armor_state: list[ArmorSectionState] = []
    if ship.armor:
        for location in [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]:
            section = ship.armor.get_section(location)
            if section:
                armor_state.append(ArmorSectionState(
                    location=location.value,
                    initial_thickness_cm=section.thickness_cm + 0,  # Would need original
                    final_thickness_cm=section.thickness_cm,
                    protection_percent=section.protection_percent
                ))

    # Check current heat
    if ship.thermal_system:
        current_heat = ship.heat_percent
        peak_heat = max(peak_heat, current_heat)

    return ShipBattleStats(
        ship_id=ship.ship_id,
        ship_name=ship_name,
        ship_type=ship.ship_type,
        faction=ship.faction,
        coilgun_shots_fired=coilgun_shots,
        torpedo_shots_fired=torpedo_shots,
        coilgun_hits=coilgun_hits,
        torpedo_hits=torpedo_hits,
        damage_dealt_gj=ship.damage_dealt_gj,
        damage_received_gj=ship.damage_taken_gj,
        delta_v_expended_kps=getattr(
            ship.kinematic_state, 'delta_v_expended_kps', 0.0
        ) if ship.kinematic_state else 0.0,
        initial_delta_v_kps=initial_delta_v,
        peak_heat_percent=peak_heat,
        modules_destroyed=modules_destroyed,
        final_armor_state=armor_state,
        final_hull_integrity=ship.hull_integrity,
        is_destroyed=ship.is_destroyed,
        kill_credit_to=ship.kill_credit
    )


def _build_timeline(simulation: CombatSimulation) -> EventTimeline:
    """
    Build the event timeline from simulation events.

    Args:
        simulation: The simulation instance.

    Returns:
        Populated EventTimeline instance.
    """
    timeline = EventTimeline()
    first_hit_recorded = False

    for event in simulation.events:
        category = None
        description = ""

        if event.event_type == SimulationEventType.SIMULATION_STARTED:
            category = TimelineEventCategory.BATTLE_START
            description = "Battle begins"

        elif event.event_type == SimulationEventType.SIMULATION_ENDED:
            category = TimelineEventCategory.BATTLE_END
            duration = event.data.get("duration", 0)
            destroyed = event.data.get("ships_destroyed", 0)
            description = f"Battle ends after {duration:.0f}s, {destroyed} ships destroyed"

        elif event.event_type == SimulationEventType.PROJECTILE_LAUNCHED:
            category = TimelineEventCategory.WEAPON_FIRED
            ke = event.data.get("kinetic_energy_gj", 0)
            description = f"{event.ship_id} fires coilgun ({ke:.1f} GJ)"

        elif event.event_type == SimulationEventType.TORPEDO_LAUNCHED:
            category = TimelineEventCategory.WEAPON_FIRED
            description = f"{event.ship_id} launches torpedo at {event.target_id}"

        elif event.event_type == SimulationEventType.PROJECTILE_IMPACT:
            category = TimelineEventCategory.HIT
            ke = event.data.get("kinetic_energy_gj", 0)
            loc = event.data.get("hit_location", "unknown")
            description = f"Coilgun hit on {event.target_id} {loc} ({ke:.1f} GJ)"

            # Mark first blood
            if not first_hit_recorded:
                first_hit_recorded = True
                timeline.add_event(
                    timestamp=event.timestamp,
                    category=TimelineEventCategory.CRITICAL,
                    description=f"FIRST BLOOD: {event.ship_id} scores first hit",
                    ship_id=event.ship_id,
                    target_id=event.target_id
                )

        elif event.event_type == SimulationEventType.TORPEDO_IMPACT:
            category = TimelineEventCategory.HIT
            dmg = event.data.get("damage_gj", 0)
            description = f"TORPEDO IMPACT on {event.target_id} ({dmg:.1f} GJ)"

            if not first_hit_recorded:
                first_hit_recorded = True
                timeline.add_event(
                    timestamp=event.timestamp,
                    category=TimelineEventCategory.CRITICAL,
                    description=f"FIRST BLOOD: {event.ship_id} scores torpedo hit",
                    ship_id=event.ship_id,
                    target_id=event.target_id
                )

        elif event.event_type == SimulationEventType.ARMOR_PENETRATED:
            category = TimelineEventCategory.CRITICAL
            loc = event.data.get("location", "unknown")
            description = f"ARMOR BREACH on {event.ship_id} ({loc})"

        elif event.event_type == SimulationEventType.MANEUVER_STARTED:
            category = TimelineEventCategory.MANEUVER
            mtype = event.data.get("maneuver_type", "unknown")
            description = f"{event.ship_id} begins {mtype.lower()} maneuver"

        elif event.event_type == SimulationEventType.THERMAL_WARNING:
            category = TimelineEventCategory.THERMAL
            heat = event.data.get("heat_percent", 0)
            description = f"{event.ship_id} overheating warning ({heat:.0f}%)"

        elif event.event_type == SimulationEventType.THERMAL_CRITICAL:
            category = TimelineEventCategory.THERMAL
            heat = event.data.get("heat_percent", 0)
            description = f"{event.ship_id} CRITICAL HEAT ({heat:.0f}%)"

        elif event.event_type == SimulationEventType.MODULE_DESTROYED:
            category = TimelineEventCategory.CRITICAL
            module = event.data.get("module_name", "Unknown")
            description = f"{event.ship_id}: {module} DESTROYED"

        elif event.event_type == SimulationEventType.SHIP_DESTROYED:
            category = TimelineEventCategory.DESTRUCTION
            killer = event.data.get("killer_id", "unknown")
            description = f"{event.ship_id} DESTROYED by {killer}"

        # Add event if we have a category
        if category:
            timeline.add_event(
                timestamp=event.timestamp,
                category=category,
                description=description,
                ship_id=event.ship_id,
                target_id=event.target_id,
                data=event.data
            )

    return timeline


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS BATTLE REPORT SYSTEM - SELF TEST")
    print("=" * 70)

    # Create a mock report for testing output formats
    print("\n--- Creating Test Report ---")

    # Create test ship stats
    alpha_stats = ShipBattleStats(
        ship_id="alpha_1",
        ship_name="Relentless",
        ship_type="destroyer",
        faction="alpha",
        coilgun_shots_fired=12,
        torpedo_shots_fired=2,
        coilgun_hits=4,
        torpedo_hits=1,
        damage_dealt_gj=8.4,
        damage_received_gj=3.2,
        delta_v_expended_kps=45.0,
        initial_delta_v_kps=120.0,
        peak_heat_percent=68.0,
        modules_destroyed=[],
        final_armor_state=[
            ArmorSectionState("nose", 15.0, 12.5, 62.5),
            ArmorSectionState("lateral", 12.0, 10.0, 50.0),
            ArmorSectionState("tail", 10.0, 9.0, 45.0),
        ],
        final_hull_integrity=87.0,
        is_destroyed=False
    )

    bravo_stats = ShipBattleStats(
        ship_id="bravo_1",
        ship_name="Defiant",
        ship_type="destroyer",
        faction="bravo",
        coilgun_shots_fired=8,
        torpedo_shots_fired=1,
        coilgun_hits=2,
        torpedo_hits=0,
        damage_dealt_gj=3.2,
        damage_received_gj=8.4,
        delta_v_expended_kps=32.0,
        initial_delta_v_kps=120.0,
        peak_heat_percent=82.0,
        modules_destroyed=["Primary Sensor Array", "Command Bridge"],
        final_armor_state=[
            ArmorSectionState("nose", 15.0, 0.0, 0.0),
            ArmorSectionState("lateral", 12.0, 2.0, 15.0),
            ArmorSectionState("tail", 10.0, 8.0, 40.0),
        ],
        final_hull_integrity=0.0,
        is_destroyed=True,
        kill_credit_to="alpha_1"
    )

    # Create timeline
    timeline = EventTimeline()
    timeline.add_event(0.0, TimelineEventCategory.BATTLE_START, "Battle begins, range 1000 km")
    timeline.add_event(30.0, TimelineEventCategory.WEAPON_FIRED,
                       "alpha_1 fires coilgun salvo (3 rounds)", "alpha_1", "bravo_1")
    timeline.add_event(32.0, TimelineEventCategory.WEAPON_FIRED,
                       "bravo_1 fires coilgun salvo (3 rounds)", "bravo_1", "alpha_1")
    timeline.add_event(45.0, TimelineEventCategory.WEAPON_FIRED,
                       "alpha_1 launches torpedo", "alpha_1", "bravo_1")
    timeline.add_event(75.0, TimelineEventCategory.HIT,
                       "Coilgun hit on bravo_1 lateral (2.1 GJ)", "alpha_1", "bravo_1")
    timeline.add_event(75.0, TimelineEventCategory.CRITICAL,
                       "FIRST BLOOD: alpha_1 scores first hit", "alpha_1", "bravo_1")
    timeline.add_event(120.0, TimelineEventCategory.CRITICAL,
                       "ARMOR BREACH on bravo_1 (nose)", "bravo_1")
    timeline.add_event(180.0, TimelineEventCategory.DESTRUCTION,
                       "bravo_1 DESTROYED by alpha_1", "bravo_1")
    timeline.add_event(225.0, TimelineEventCategory.BATTLE_END,
                       "Battle ends after 225s, 1 ships destroyed")

    # Create report
    report = BattleReport(
        battle_name="Head-On Engagement",
        duration_seconds=225.0,
        outcome=BattleOutcome.ALPHA_VICTORY,
        winning_faction="alpha",
        participants={"alpha_1": alpha_stats, "bravo_1": bravo_stats},
        timeline=timeline,
        metrics={
            "total_shots_fired": 20,
            "total_hits": 7,
            "hit_rate": 0.35
        }
    )

    # Test text output
    print("\n" + "=" * 70)
    print("TEXT OUTPUT:")
    print("=" * 70)
    print(report.to_text())

    # Test markdown output (sample)
    print("\n" + "=" * 70)
    print("MARKDOWN OUTPUT (first 50 lines):")
    print("=" * 70)
    md_lines = report.to_markdown().split("\n")[:50]
    print("\n".join(md_lines))
    print("...")

    # Test JSON output (sample)
    print("\n" + "=" * 70)
    print("JSON OUTPUT (first 30 lines):")
    print("=" * 70)
    json_lines = report.to_json().split("\n")[:30]
    print("\n".join(json_lines))
    print("...")

    print("\n" + "=" * 70)
    print("Battle report system tests completed!")
    print("=" * 70)
