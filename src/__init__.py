"""AI Commanders space battle simulator package."""

from .combat import (
    Armor,
    CombatResolver,
    HitLocation,
    HitResult,
    ShipArmor,
    Weapon,
    create_ship_armor_from_fleet_data,
    create_weapon_from_fleet_data,
    load_fleet_data,
    simulate_combat_exchange,
)

from .command import (
    # Command types
    SetThrust,
    RotateTo,
    Engage,
    HoldFire,
    Evade,
    Command,
    # Events
    TacticalEvent,
    TacticalEventType,
    # State types
    ShipState,
    ThreatInfo,
    ProjectileInfo,
    BattleState,
    # Controllers
    TacticalController,
    StrategicController,
    LLMStrategicController,
    RuleBasedStrategicController,
    # Utilities
    validate_command,
)

from .modules import (
    # Enums
    ModuleType,
    CRITICAL_MODULE_TYPES,
    # Classes
    ModulePosition,
    Module,
    ModuleLayer,
    ModuleLayout,
    # Factory functions
    create_module_layout,
)

__all__ = [
    # Combat module
    "Armor",
    "CombatResolver",
    "HitLocation",
    "HitResult",
    "ShipArmor",
    "Weapon",
    "create_ship_armor_from_fleet_data",
    "create_weapon_from_fleet_data",
    "load_fleet_data",
    "simulate_combat_exchange",
    # Command module - Command types
    "SetThrust",
    "RotateTo",
    "Engage",
    "HoldFire",
    "Evade",
    "Command",
    # Command module - Events
    "TacticalEvent",
    "TacticalEventType",
    # Command module - State types
    "ShipState",
    "ThreatInfo",
    "ProjectileInfo",
    "BattleState",
    # Command module - Controllers
    "TacticalController",
    "StrategicController",
    "LLMStrategicController",
    "RuleBasedStrategicController",
    # Command module - Utilities
    "validate_command",
    # Modules module - Enums
    "ModuleType",
    "CRITICAL_MODULE_TYPES",
    # Modules module - Classes
    "ModulePosition",
    "Module",
    "ModuleLayer",
    "ModuleLayout",
    # Modules module - Factory functions
    "create_module_layout",
]
