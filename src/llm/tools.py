"""
Tool definitions for LLM captain decision-making.

These tools are provided to the LLM for structured command output.
Tools are dynamically filtered based on ship capabilities.
"""

from typing import List, Dict, Any


# Base captain tools - always available
CAPTAIN_TOOLS_BASE = [
    {
        "type": "function",
        "function": {
            "name": "set_maneuver",
            "description": "Set ship maneuver for the next 30 seconds of simulation time",
            "parameters": {
                "type": "object",
                "properties": {
                    "maneuver_type": {
                        "type": "string",
                        "enum": ["INTERCEPT", "EVADE", "BRAKE", "MAINTAIN"],
                        "description": (
                            "INTERCEPT: Close with enemy aggressively. "
                            "EVADE: Evasive maneuvers to dodge fire. "
                            "BRAKE: Flip and burn to slow down. "
                            "MAINTAIN: Hold current course and speed."
                        )
                    },
                    "throttle": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Throttle level 0.0-1.0 (default 1.0)"
                    }
                },
                "required": ["maneuver_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_weapons_order",
            "description": (
                "Set fire control orders for spinal and turret weapons independently. "
                "SPINAL: 9.9 km/s projectile, 4.3 GJ damage, 900km range, but ONLY fires if target within 30° of nose. "
                "TURRET: 6.0 km/s projectile, 0.7 GJ damage, 500km range, can fire at targets in full front hemisphere (180°)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spinal_mode": {
                        "type": "string",
                        "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
                        "description": (
                            "Spinal coilgun mode (high damage, requires nose pointing at target within 30°): "
                            "FIRE_IMMEDIATE: Fire when ready and target in arc. "
                            "FIRE_WHEN_OPTIMAL: Fire when target in arc AND hit probability >= threshold. "
                            "FIRE_AT_RANGE: Fire when target in arc AND within range. "
                            "HOLD_FIRE: Don't fire spinal."
                        )
                    },
                    "turret_mode": {
                        "type": "string",
                        "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
                        "description": (
                            "Turret coilgun mode (lower damage, but 180° firing arc): "
                            "FIRE_IMMEDIATE: Fire as soon as ready. "
                            "FIRE_WHEN_OPTIMAL: Fire when hit probability >= threshold. "
                            "FIRE_AT_RANGE: Fire when target within range. "
                            "HOLD_FIRE: Don't fire turrets."
                        )
                    },
                    "spinal_min_probability": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 0.9,
                        "description": "For spinal FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
                    },
                    "turret_min_probability": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 0.9,
                        "description": "For turret FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
                    },
                    "spinal_max_range_km": {
                        "type": "number",
                        "minimum": 50,
                        "maximum": 900,
                        "description": "For spinal FIRE_AT_RANGE: maximum range in km (default 500)"
                    },
                    "turret_max_range_km": {
                        "type": "number",
                        "minimum": 50,
                        "maximum": 500,
                        "description": "For turret FIRE_AT_RANGE: maximum range in km (default 300)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_radiators",
            "description": "Extend or retract heat radiators",
            "parameters": {
                "type": "object",
                "properties": {
                    "extend": {
                        "type": "boolean",
                        "description": (
                            "True: Extend radiators for faster cooling (but vulnerable to damage). "
                            "False: Retract radiators for protection (slower cooling)."
                        )
                    }
                },
                "required": ["extend"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_primary_target",
            "description": "Designate a specific enemy ship as your primary target. Weapons and intercept maneuvers will focus on this target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Name of enemy ship to target (e.g., 'ENS Aggressor')"
                    }
                },
                "required": ["target_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_heading",
            "description": "Set a course in a specific 3D direction. Use for custom positioning, flanking, or disengaging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "object",
                        "description": "Direction vector in ship-relative coordinates (will be normalized)",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Forward/backward (+forward, -backward)"
                            },
                            "y": {
                                "type": "number",
                                "description": "Left/right (+starboard, -port)"
                            },
                            "z": {
                                "type": "number",
                                "description": "Up/down (+up, -down)"
                            }
                        },
                        "required": ["x", "y", "z"]
                    },
                    "throttle": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Throttle level (0.0-1.0)"
                    }
                },
                "required": ["direction", "throttle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "OPTIONAL: Send a message to other ships. Use sparingly - only when you have something meaningful to communicate (tactical threats, surrender demands, or psychological warfare). You do NOT need to send a message every checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Brief message content (keep it short and impactful)"
                    },
                    "recipient": {
                        "type": "string",
                        "enum": ["ALL", "ALL_ENEMIES", "ALL_FRIENDLIES", "SPECIFIC"],
                        "description": "Who receives the message (default: ALL_ENEMIES)"
                    },
                    "target_ship": {
                        "type": "string",
                        "description": "Ship name if recipient is SPECIFIC"
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "surrender",
            "description": "Surrender the battle. You lose, but battle ends immediately.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_draw",
            "description": "Propose a mutual draw. The enemy captain will be notified and must also propose draw for it to take effect. Can be retracted with retract_draw.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retract_draw",
            "description": "Retract your draw proposal. Use if you previously proposed a draw but now want to continue fighting.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
]

# Torpedo tool - only available if ship has torpedoes
TORPEDO_TOOL = {
    "type": "function",
    "function": {
        "name": "launch_torpedo",
        "description": "Launch a torpedo at the enemy ship. Limited ammunition!",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    }
}

# Personality selection tool - used in pre-battle phase
PERSONALITY_SELECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_personality",
        "description": "Define your combat personality as an AI commander. Be authentic to your style!",
        "parameters": {
            "type": "object",
            "properties": {
                "personality_description": {
                    "type": "string",
                    "description": (
                        "Your combat personality (2-4 sentences). Describe: "
                        "your tactical philosophy, communication style, what drives your decisions, "
                        "and any signature approaches that feel authentically 'you' as an AI."
                    )
                }
            },
            "required": ["personality_description"]
        }
    }
}

# Tools for personality selection phase only
PERSONALITY_SELECTION_TOOLS = [PERSONALITY_SELECTION_TOOL]


def get_captain_tools(has_torpedoes: bool = False) -> List[Dict[str, Any]]:
    """
    Get the appropriate tools for a captain based on ship capabilities.

    Args:
        has_torpedoes: Whether the ship has torpedo launchers

    Returns:
        List of tool definitions for the LLM
    """
    tools = CAPTAIN_TOOLS_BASE.copy()

    if has_torpedoes:
        tools.append(TORPEDO_TOOL)

    return tools


def get_weapon_groups_for_ship(ship_type: str, fleet_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Determine weapon groups and their slots for a ship type.

    Returns dict like: {"spinal": ["weapon_0"], "heavy_coilguns": ["weapon_1", "weapon_2"], "coilguns": ["weapon_3"]}
    """
    if ship_type not in fleet_data.get("ships", {}):
        return {"coilguns": ["weapon_0", "weapon_1"]}  # Fallback

    ship_spec = fleet_data["ships"][ship_type]
    weapons = ship_spec.get("weapons", [])

    groups: Dict[str, List[str]] = {}
    for weapon in weapons:
        wtype = weapon.get("type", "")
        slot = weapon.get("slot", "")
        if not slot or slot.startswith("pd_"):
            continue  # Skip PD weapons

        if wtype == "spinal_coiler_mk3":
            groups.setdefault("spinal", []).append(slot)
        elif wtype == "heavy_coilgun_mk3":
            groups.setdefault("heavy_coilguns", []).append(slot)
        elif wtype == "coilgun_mk3":
            groups.setdefault("coilguns", []).append(slot)

    return groups


def build_weapon_tool_for_ship(ship_type: str, fleet_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the set_weapons_order tool definition based on ship's weapons."""
    groups = get_weapon_groups_for_ship(ship_type, fleet_data)
    weapon_types = fleet_data.get("weapon_types", {})

    properties: Dict[str, Any] = {}
    description_parts = ["Set fire control orders for weapons."]

    # Spinal mode (if ship has spinal)
    if "spinal" in groups:
        spinal_spec = weapon_types.get("spinal_coiler_mk3", {})
        vel = spinal_spec.get("muzzle_velocity_kps", 9.9)
        dmg = spinal_spec.get("kinetic_energy_gj", 4.32)
        rng = spinal_spec.get("range_km", 900)

        properties["spinal_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"Spinal coiler ({vel} km/s, {dmg:.1f} GJ, {rng}km range, requires nose within 30° of target)"
        }
        properties["spinal_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For spinal FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["spinal_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": f"For spinal FIRE_AT_RANGE: maximum range (default {min(500, rng)})"
        }
        description_parts.append(f"SPINAL: {vel} km/s, {dmg:.1f} GJ, fixed mount (30° arc).")

    # Heavy coilgun mode
    if "heavy_coilguns" in groups:
        heavy_spec = weapon_types.get("heavy_coilgun_mk3", {})
        vel = heavy_spec.get("muzzle_velocity_kps", 7.0)
        dmg = heavy_spec.get("kinetic_energy_gj", 1.22)
        rng = heavy_spec.get("range_km", 600)
        count = len(groups["heavy_coilguns"])

        properties["heavy_coilgun_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"Heavy coilguns x{count} ({vel} km/s, {dmg:.1f} GJ each, {rng}km range, turreted 180° arc)"
        }
        properties["heavy_coilgun_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For heavy coilgun FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["heavy_coilgun_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": f"For heavy coilgun FIRE_AT_RANGE: maximum range (default {min(400, rng)})"
        }
        description_parts.append(f"HEAVY COILGUNS x{count}: {vel} km/s, {dmg:.1f} GJ each, turreted.")

    # Standard coilgun mode
    if "coilguns" in groups:
        coil_spec = weapon_types.get("coilgun_mk3", {})
        vel = coil_spec.get("muzzle_velocity_kps", 6.0)
        dmg = coil_spec.get("kinetic_energy_gj", 0.72)
        rng = coil_spec.get("range_km", 500)
        count = len(groups["coilguns"])

        properties["coilgun_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"Coilguns x{count} ({vel} km/s, {dmg:.2f} GJ each, {rng}km range, turreted 180° arc)"
        }
        properties["coilgun_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For coilgun FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["coilgun_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": f"For coilgun FIRE_AT_RANGE: maximum range (default {min(300, rng)})"
        }
        description_parts.append(f"COILGUNS x{count}: {vel} km/s, {dmg:.2f} GJ each, turreted.")

    return {
        "type": "function",
        "function": {
            "name": "set_weapons_order",
            "description": " ".join(description_parts),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": []
            }
        }
    }


def get_captain_tools_for_ship(ship_type: str, fleet_data: Dict[str, Any], has_torpedoes: bool = False) -> List[Dict[str, Any]]:
    """Get tools appropriate for a specific ship type."""
    # Start with base tools but EXCLUDE the hardcoded set_weapons_order
    tools = [t for t in CAPTAIN_TOOLS_BASE if t["function"]["name"] != "set_weapons_order"]

    # Add dynamic weapon tool
    weapon_tool = build_weapon_tool_for_ship(ship_type, fleet_data)
    tools.insert(1, weapon_tool)  # Insert after set_maneuver

    if has_torpedoes:
        tools.append(TORPEDO_TOOL)

    return tools


# Default tools (for destroyers without torpedoes)
CAPTAIN_TOOLS = get_captain_tools(has_torpedoes=False)
