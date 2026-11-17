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
                "Issue a specific order to one of your ship captains. Two kinds of field: "
                "the STRUCTURED fields are what the engine applies at once and what the "
                "compliance check compares - maneuver / heading / throttle / evade fly, "
                "suggested_target is THE target (one ship), torpedo_salvo + torpedo_target "
                "are the launch. order_text is read by the captain only and flies nothing: "
                "make it say the same as the fields. A ship has one target per checkpoint - "
                "a 'then shift to X' in the text is a second target the engine never applies "
                "(the captain may follow either)."
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
                            "Your words to the captain (it reads them; the engine does not). "
                            "Name the same maneuver and the same single target as the "
                            "structured fields; add weapon use and priorities."
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
                        "description": ("The ONE enemy ship (name or id) this ship engages: the "
                                        "captain's TARGET line and the compliance check use it.")
                    },
                    "torpedo_salvo": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 12,
                        "description": (
                            "Optional: order this ship to launch N torpedoes this "
                            "checkpoint (0 = hold). Only torpedo-armed hulls can comply - "
                            "a launcher reloads every 12s, so 3 per launcher per 30s "
                            "checkpoint is the ceiling (a 4-tube torpedo cruiser: 12; "
                            "YOUR FLEET lists each ship's launchable count). "
                            "Use this to COORDINATE a simultaneous "
                            "time-on-target across several ships: rounds arriving "
                            "together split enemy PD dwell, while trickled rounds are "
                            "blinded one at a time (measured vs 4 PD turrets: 4 "
                            "simultaneous = 24/24 through, same 4 spaced 30s = 12/24). "
                            "Sizing: ~4 rounds kill a destroyer, ~6 kill anything; a "
                            "receding evader outruns the rounds. "
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
                            "formation holds. A per-ship order with NO heading and NO maneuver "
                            "flies that fleet heading (evade alone rides on it); its own "
                            "heading or maneuver REPLACES it for that ship."
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
                            "Throttle for the heading or a burning maneuver (default 1.0). 0 = "
                            "attitude only: the nose turns onto the heading and the ship coasts "
                            "(e.g. spinals back on the enemy). For an \"all ships\" heading it "
                            "is a fraction of the SLOWEST hull's acceleration. PADLOCK and "
                            "MAINTAIN never burn - throttle does not apply to them, except "
                            "that MAINTAIN + heading, or PADLOCK + heading at throttle above 0, "
                            "flies that HEADING at this throttle (the nose follows the heading)."
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
                            "Optional structured maneuver, applied at once. It REPLACES the "
                            "fleet (\"all ships\") order for this ship: leave it out to keep "
                            "the fleet heading. MAINTAIN = engine OFF, the ship coasts (no "
                            "turn, no burn). PADLOCK = engine off, the nose tracks the target "
                            "(coast and shoot) - both unless you also give a heading (see "
                            "below). To BURN on a course give heading + throttle. "
                            "INTERCEPT never brakes - it passes THROUGH the target at full "
                            "closing speed - unless you give standoff_km: then it closes, "
                            "flips, brakes and holds that range. HOLD_RANGE keeps range_km "
                            "(default: the range when first ordered; re-issuing it keeps that "
                            "range) to the target: the anti-merge order. With \"all ships\" "
                            "every ship flies it. A heading given with MAINTAIN or EVADE (or "
                            "with PADLOCK at throttle above 0) is flown as that heading; with "
                            "any other maneuver the maneuver wins. Each ship has ONE maneuver: "
                            "to close in and still dodge, add evade instead of ordering "
                            "INTERCEPT and EVADE together."
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
                            "torpedo's final seconds, then presents armor; jinks slugs. It "
                            "burns at the order's throttle - full throttle under an order that "
                            "does not burn (throttle 0, PADLOCK, MAINTAIN). While rounds are "
                            "inbound the overlay, not your burn, flies the drive: the ship "
                            "jinks, TANKs (half throttle) or BRACEs (drive off), so an escape "
                            "burn 'with evade' under fire burns only a small part of the time "
                            "(YOUR FLEET / ENGINE show what was flown). Given "
                            "ALONE (no maneuver, no heading) it rides on the fleet order in "
                            "force - or, with none, on what the ship is already flying - so "
                            "never add MAINTAIN just to carry it: MAINTAIN switches the "
                            "engine off."
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
                "Set the overall fleet strategy visible to all captains (and, for torpedo "
                "fleets, this checkpoint's salvo_plan). Per-ship orders are requested one "
                "ship at a time after this call."
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
                    },
                    "salvo_plan": {
                        "type": "array",
                        "description": (
                            "Optional, torpedo fleets: THIS checkpoint's salvo, allocated across "
                            "your ships here, once - the per-ship order calls that follow run in "
                            "parallel and cannot see each other's torpedo_salvo (a '7-round strike' "
                            "became 3+3+3 = 9 that way). Each per-ship order is shown this "
                            "allocation and launches it unless that order sets its own "
                            "torpedo_salvo. Rounds a ship cannot launch this checkpoint (YOUR "
                            "FLEET: 'N now') are not launched."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "ship_name": {"type": "string"},
                                "rounds": {"type": "integer", "minimum": 0, "maximum": 12},
                                "target": {"type": "string"},
                            },
                            "required": ["ship_name", "rounds"],
                        },
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


# Intent-only mode (AdmiralConfig.intent_only, an experiment): no per-ship
# order phase, issue_order carries only a coordinated torpedo salvo.
INTENT_ONLY_ISSUE_ORDER = (
    "INTENT-ONLY COMMAND: there are no per-ship orders this battle - captains choose their "
    "own maneuvers and targets from your directive and battle plan. Use issue_order ONLY to "
    "order a coordinated torpedo salvo: ship_name, torpedo_salvo=N, torpedo_target=<enemy>, "
    "one call per launching ship (or set_fleet_directive salvo_plan). Every other field is "
    "ignored."
)
INTENT_ONLY_DIRECTIVE = (
    "Set the overall fleet strategy visible to all captains - in intent-only command it is "
    "their only order: they pick their own maneuvers and targets from it. For torpedo "
    "fleets, salvo_plan allocates this checkpoint's salvo."
)


def get_admiral_tools(intent_only: bool = False) -> List[Dict[str, Any]]:
    """Get tools available to Admirals (intent_only: that mode's descriptions)."""
    if not intent_only:
        return ADMIRAL_TOOLS.copy()
    import copy
    tools = copy.deepcopy(ADMIRAL_TOOLS)
    for tool in tools:
        fn = tool["function"]
        if fn["name"] == "issue_order":
            fn["description"] = INTENT_ONLY_ISSUE_ORDER
            # Only the fields this mode flies: a schema still offering heading /
            # maneuver / throttle invited orders that are silently dropped.
            params = fn["parameters"]
            params["properties"] = {k: v for k, v in params["properties"].items()
                                    if k in ("ship_name", "torpedo_salvo", "torpedo_target")}
            params["required"] = ["ship_name"]
        elif fn["name"] == "set_fleet_directive":
            fn["description"] = INTENT_ONLY_DIRECTIVE
    return tools


def get_admiral_tools_with_draw_pending() -> List[Dict[str, Any]]:
    """
    Get Admiral tools when enemy has proposed a draw.

    Adds emphasis to accept/reject options.
    """
    tools = ADMIRAL_TOOLS.copy()
    # Could modify descriptions here if needed
    return tools
