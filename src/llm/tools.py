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
            "description": "Set fire control orders for weapons. The tactical layer will fire automatically based on this order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "firing_mode": {
                        "type": "string",
                        "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE", "FREE_FIRE"],
                        "description": (
                            "FIRE_IMMEDIATE: Fire as soon as weapons are ready. "
                            "FIRE_WHEN_OPTIMAL: Fire only when hit probability exceeds threshold. "
                            "FIRE_AT_RANGE: Fire only when target enters specified range. "
                            "HOLD_FIRE: Don't fire, conserve ammunition. "
                            "FREE_FIRE: Fire at any valid target when ready."
                        )
                    },
                    "weapon_slot": {
                        "type": "string",
                        "enum": ["all", "spinal", "turret"],
                        "description": (
                            "all: Apply to all weapons. "
                            "spinal: Main spinal coilgun only (500-900km range, high damage). "
                            "turret: Turret coilguns only (200-500km range, can fire off-bore)."
                        )
                    },
                    "min_hit_probability": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 0.9,
                        "description": "For FIRE_WHEN_OPTIMAL: minimum hit probability to fire (default 0.3 = 30%)"
                    },
                    "max_range_km": {
                        "type": "number",
                        "minimum": 50,
                        "maximum": 1000,
                        "description": "For FIRE_AT_RANGE: maximum engagement range in km"
                    },
                    "conserve_ammo": {
                        "type": "boolean",
                        "description": "If true, be more conservative with ammunition (higher threshold)"
                    }
                },
                "required": ["firing_mode"]
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
            "name": "send_message",
            "description": "Send a message to the enemy captain (shittalk, negotiate, intimidate)",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message content to send"
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
