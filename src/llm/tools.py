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
            "description": "Send a message to other ships. Can target all ships, enemies only, friendlies only, or a specific ship.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message content to send"
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
            "description": "Propose a mutual draw. Requires enemy captain to also propose draw.",
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


# Default tools (for destroyers without torpedoes)
CAPTAIN_TOOLS = get_captain_tools(has_torpedoes=False)
