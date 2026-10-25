"""
Tool definitions for LLM Admiral decision-making.

Admirals have different tools than captains - they issue orders
rather than directly controlling ships.
"""

from typing import List, Dict, Any

from .communication import MAX_DISCUSSION_EXCHANGES


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
                    },
                    "torpedo_salvo": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": (
                            "Optional: order this ship to launch N torpedoes this "
                            "checkpoint (0 = hold). Only torpedo-armed hulls can comply - "
                            "a launcher reloads every 12s so 2 per launcher per decision "
                            "is the ceiling. Use this to COORDINATE a simultaneous "
                            "time-on-target across several ships: rounds arriving "
                            "together split enemy PD dwell, while trickled rounds are "
                            "blinded one at a time (measured vs 4 PD turrets: 4 "
                            "simultaneous = 24/24 through, same 4 spaced 30s = 12/24). "
                            "Sizing: ~4 rounds kill a destroyer, ~6 kill anything; only "
                            "commit against CLOSING targets - receding evaders escape. "
                            "Rounds whose target dies mid-flight retarget the nearest "
                            "reachable enemy on their own if their seeker is alive and "
                            "fuel allows, so overkill partially carries onto the next hull."
                        )
                    },
                    "torpedo_target": {
                        "type": "string",
                        "description": (
                            "Optional: enemy ship name or id the torpedo salvo should be "
                            "aimed at, if different from suggested_target. Concentrating a "
                            "fleet-wide salvo on one hull is how you overwhelm point defense."
                        )
                    },
                    "heading": {
                        "type": "object",
                        "description": (
                            "Optional: WORLD-frame direction {x, y, z} this ship flies this "
                            "checkpoint - applied at once as its HEADING maneuver (the "
                            "captain may still override). The drive thrusts along the NOSE, "
                            "so the ship turns onto the heading first, then burns. Copy a "
                            "vector from RANGE CONTROL (cancel closure / open range / pass "
                            "wide / flank). With ship_name \"all ships\" ONE heading goes to "
                            "the whole fleet, each hull's throttle scaled so all accelerate "
                            "like the slowest hull and the burn starting together: the "
                            "formation holds."
                        ),
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "required": ["x", "y", "z"]
                    },
                    "throttle": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Throttle for the heading or maneuver (default 1.0). 0 = attitude "
                            "only: the nose turns onto the heading and the ship coasts (e.g. "
                            "spinals back on the enemy). For an \"all ships\" heading it is a "
                            "fraction of the SLOWEST hull's acceleration."
                        )
                    },
                    "toward_point": {
                        "type": "object",
                        "description": (
                            "Optional, instead of heading: a WORLD position in km to go to "
                            "(a heading is a DIRECTION, not a destination - never put a "
                            "position in heading). The ship flies there and STOPS (burn, flip "
                            "and brake are timed for it). For \"all ships\": one heading from "
                            "the fleet centre toward it, flown as a formation - that does not "
                            "brake at the point, so re-issue it each checkpoint."
                        ),
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"}
                        },
                        "required": ["x", "y", "z"]
                    },
                    "toward_ship": {
                        "type": "string",
                        "description": (
                            "Optional, instead of heading: fly to this ship (name or id) and "
                            "fall in alongside it, matching its velocity - to rejoin a "
                            "friendly. To stand off an enemy use maneuver INTERCEPT with "
                            "standoff_km, or HOLD_RANGE."
                        )
                    },
                    "maneuver": {
                        "type": "string",
                        "enum": ["INTERCEPT", "HOLD_RANGE", "EVADE", "BRAKE", "MAINTAIN", "PADLOCK"],
                        "description": (
                            "Optional structured maneuver, applied at once like heading (give "
                            "one or the other; maneuver wins). INTERCEPT never brakes - it "
                            "passes THROUGH the target at full closing speed - unless you give "
                            "standoff_km: then it closes, flips, brakes and holds that range. "
                            "HOLD_RANGE keeps range_km (default: the range when first ordered; "
                            "re-issuing it keeps that range) to the target: the anti-merge "
                            "order. With \"all ships\" every ship flies it. "
                            "Each ship has ONE maneuver: to close in and still dodge, add evade "
                            "instead of ordering INTERCEPT and EVADE together."
                        )
                    },
                    "maneuver_target": {
                        "type": "string",
                        "description": (
                            "Enemy name or id the INTERCEPT / HOLD_RANGE / PADLOCK flies "
                            "relative to. Default: suggested_target if it names a ship, else "
                            "the ship's own target, else the nearest enemy. A maneuver_target "
                            "that names no live ship is NOT applied (the captain is told)."
                        )
                    },
                    "standoff_km": {
                        "type": "number",
                        "minimum": 0,
                        "description": "INTERCEPT: stop closing at this range and hold it."
                    },
                    "arrival_closing_kps": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "INTERCEPT: closing speed still allowed on reaching standoff_km "
                            "(default 0 = arrive matched); first arrival only, capped so the "
                            "overshoot is stopped well inside the standoff (never a "
                            "fly-through)."
                        )
                    },
                    "range_km": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "HOLD_RANGE: the range to keep (default: the range when first "
                            "ordered; re-issuing keeps it)."
                        )
                    },
                    "evade": {
                        "type": "string",
                        "enum": ["full", "dodge"],
                        "description": (
                            "Evasion on top of the heading / maneuver while a torpedo or slug "
                            "is actually inbound: 'full' = EVADE under threat (runs from guided "
                            "torpedoes, costing closure), 'dodge' = keeps course until a guided "
                            "torpedo's final seconds, then presents armor; jinks slugs."
                        )
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
            "name": "set_battle_plan",
            "description": (
                "Set or amend your STANDING BATTLE PLAN - your private memory. Unlike "
                "the fleet directive (re-issued every checkpoint, shown to captains), "
                "the standing plan is shown back to YOU, and only you, at every "
                "checkpoint until you change it - it is your only memory of your own "
                "intent between checkpoints. Use it for the shape of the whole battle: "
                "numbered phases, the conditions that trigger the next phase, torpedo "
                "reserve policy, when you would disengage. Replaces the previous plan "
                "entirely, so restate what still applies. 2-6 sentences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": (
                            "The standing plan, e.g. 'Phase 1: close to 300km in tight "
                            "formation, all PD overlapping. Phase 2: when their torpedo "
                            "cruiser has fired twice, full salvo at it. Phase 3: focus "
                            "fire remaining hulls nearest-first. Abort to draw if we "
                            "lose 2 ships before phase 2.'"
                        )
                    }
                },
                "required": ["plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "message_enemy_admiral",
            "description": (
                "Send a message to the enemy Admiral (if one exists). "
                "The enemy Admiral reads it at its next decision (the next checkpoint). "
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
            "Ask your Admiral when your orders conflict with what your sensors show "
            "and you need a decision now (status goes in report_to_admiral). Answered "
            "this checkpoint from a live picture of your ship; you then decide again "
            "with the answer, so each question costs a full extra decision. "
            f"At most {MAX_DISCUSSION_EXCHANGES} exchanges per checkpoint."
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
