"""
Tool definitions for LLM Admiral decision-making.

Admirals have different tools than captains - they issue orders
rather than directly controlling ships.
"""

from typing import List, Dict, Any


# Admiral tools for fleet command
ADMIRAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "issue_order",
            "description": (
                "REQUIRED: Issue a specific order to one of your ship captains. "
                "You MUST call this tool once for EACH ship in your fleet! "
                "The captain will see your exact words and should follow them. "
                "Be specific about target, maneuver, and weapon orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ship_name": {
                        "type": "string",
                        "description": "Name of the ship to order (e.g., 'ISS Resolute')"
                    },
                    "order_text": {
                        "type": "string",
                        "description": (
                            "Your order to the captain. Be specific about maneuvers, "
                            "targets, weapon usage, and priorities. The captain will "
                            "interpret this and execute using their available tools."
                        )
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["CRITICAL", "HIGH", "NORMAL", "LOW"],
                        "description": (
                            "Priority level. CRITICAL = must execute immediately. "
                            "HIGH = important, do soon. NORMAL = standard order. "
                            "LOW = when convenient."
                        )
                    },
                    "suggested_target": {
                        "type": "string",
                        "description": "Optional: enemy ship name to focus on"
                    }
                },
                "required": ["ship_name", "order_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_fleet_directive",
            "description": (
                "Set the overall fleet strategy visible to all captains. "
                "IMPORTANT: This alone is NOT enough! You MUST ALSO call issue_order for each ship. "
                "Use set_fleet_directive ONLY to set the overall strategy, then call issue_order "
                "for each ship with specific orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": (
                            "Brief strategic directive visible to all captains. "
                            "Use SHIP NAMES not IDs (e.g., 'Focus fire on OCS Gemini-3' not 'Focus on beta_3'). "
                            "Examples: 'Focus fire on OCS Gemini-1', "
                            "'Defensive formation, conserve delta-v', "
                            "'TIS Haiku-2 draw fire while others flank'"
                        )
                    }
                },
                "required": ["directive"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "message_enemy_admiral",
            "description": (
                "Send a message to the enemy Admiral (if one exists). "
                "This is delivered immediately within the same checkpoint. "
                "Use for negotiations, demands, or psychological warfare."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message content for enemy Admiral"
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_fleet_draw",
            "description": (
                "Propose a mutual draw for the entire battle. "
                "The enemy Admiral (or captains if no Admiral) must also agree. "
                "If both sides agree, the battle ends in a draw."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "accept_fleet_draw",
            "description": (
                "Accept the enemy's draw proposal. "
                "Only use if the enemy has already proposed a draw."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reject_fleet_draw",
            "description": (
                "Explicitly reject the enemy's draw proposal. "
                "Battle will continue."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]


# Tool for captains to discuss with Admiral
DISCUSS_WITH_ADMIRAL_TOOL = {
    "type": "function",
    "function": {
        "name": "discuss_with_admiral",
        "description": (
            "Ask your Admiral a question or request clarification on orders. "
            "Limited to 2 exchanges per checkpoint. Your Admiral will respond "
            "before you finalize your decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Your question or concern for the Admiral. "
                        "Be specific about what you need clarification on."
                    )
                }
            },
            "required": ["question"]
        }
    }
}


def get_admiral_tools() -> List[Dict[str, Any]]:
    """Get tools available to Admirals."""
    return ADMIRAL_TOOLS.copy()


def get_admiral_tools_with_draw_pending() -> List[Dict[str, Any]]:
    """
    Get Admiral tools when enemy has proposed a draw.

    Adds emphasis to accept/reject options.
    """
    tools = ADMIRAL_TOOLS.copy()
    # Could modify descriptions here if needed
    return tools
