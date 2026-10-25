"""
Tool definitions for LLM captain decision-making.

These tools are provided to the LLM for structured command output.
Tools are dynamically filtered based on ship capabilities.
"""

from typing import List, Dict, Any
from copy import deepcopy
import math

try:
    from ..thermal import AUTO_RADIATOR_DEFAULT_PCT as _AUTO_PCT
    from ..thermal import HEAT_AUTO_EXTEND_FRACTION as _FORCED_FRACTION
except ImportError:  # pragma: no cover - script-style imports (src/ on sys.path)
    from thermal import AUTO_RADIATOR_DEFAULT_PCT as _AUTO_PCT
    from thermal import HEAT_AUTO_EXTEND_FRACTION as _FORCED_FRACTION

_FORCED_PCT = _FORCED_FRACTION * 100.0

from .communication import (
    CONTACT_ASSESSMENTS, CONTACT_NOTE_CHARS, MAX_ENEMY_MESSAGES_PER_DECISION,
    MAX_FLEET_SIGNALS_PER_DECISION, REPORT_INTENT_CHARS, REPORT_REASON_CHARS,
    REPORT_STATUSES, REQUEST_KINDS, REQUEST_TEXT_CHARS, SIGNAL_KINDS, SIGNAL_TEXT_CHARS,
)


# Base captain tools - always available
CAPTAIN_TOOLS_BASE = [
    {
        "type": "function",
        "function": {
            "name": "set_maneuver",
            "description": (
                "Set ship maneuver until the next decision checkpoint. ONE maneuver per "
                "decision: set_maneuver and set_heading share a single slot and only the "
                "first call counts. To close in AND dodge torpedoes, add evade to "
                "INTERCEPT/HOLD_RANGE instead of a second call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "maneuver_type": {
                        "type": "string",
                        "enum": ["INTERCEPT", "HOLD_RANGE", "EVADE", "BRAKE", "MAINTAIN", "PADLOCK"],
                        "description": (
                            "INTERCEPT: burn toward the target on a collision course. Plain "
                            "INTERCEPT NEVER BRAKES: it keeps building closure and passes "
                            "THROUGH the target at full closing speed (a merge - seconds of "
                            "fire, then flying apart). Give standoff_km and it closes, flips, "
                            "brakes and holds that range instead. "
                            "HOLD_RANGE: keep range_km (default: the range when first ordered - "
                            "renewing the order keeps it) to the target - closes or opens to it, "
                            "then cancels closure and holds it (the anti-merge maneuver; matches "
                            "his course and burn as he moves; if he out-accelerates you it "
                            "matches his burn and buys time instead of closing). "
                            "EVADE: Threat-aware defense - jinks vs gun fire; vs guided "
                            "torpedoes auto-RUNS (cuts closure, buys PD dwell) and in the "
                            "final seconds auto-PRESENTS your thickest armor. Dodges blinded "
                            "torpedoes - use it (or evade on another maneuver) under torpedo "
                            "fire. With no threat it weaves around your CURRENT course: never "
                            "chases, never brakes. "
                            "BRAKE: Flip and burn against your own velocity (world frame, not "
                            "relative to the enemy - HOLD_RANGE matches an enemy). "
                            "MAINTAIN: Coast at current velocity - no thrust, no tracking. "
                            "PADLOCK: Coast while tracking target with nose - fire spinal during "
                            "passes (no closing thrust). For a world direction, a position or "
                            "a ship's bearing use set_heading (RANGE CONTROL lists ready-made "
                            "headings)."
                        )
                    },
                    "throttle": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Throttle level 0.0-1.0 (default 1.0). Caps every burn of the "
                            "maneuver, braking included."
                        )
                    },
                    "target_id": {
                        "type": "string",
                        "description": (
                            "Ship to fly relative to (INTERCEPT, HOLD_RANGE, PADLOCK): name or "
                            "id. Default: your primary target. Does NOT change what your guns "
                            "shoot at (that is set_primary_target)."
                        )
                    },
                    "standoff_km": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "INTERCEPT only: stop closing at this range and hold it. The flip "
                            "and the braking burn are timed for you from your closing speed "
                            "and acceleration. Without it INTERCEPT never brakes."
                        )
                    },
                    "arrival_closing_kps": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "INTERCEPT only: the closing speed you may still carry when you "
                            "reach standoff_km (default 0 = arrive matched). More keeps "
                            "momentum but carries you inside the standoff before it holds. "
                            "First arrival only, and capped so the overshoot can still be "
                            "stopped well inside the standoff - you never fly through the "
                            "target. Without standoff_km there is no room to overshoot: you "
                            "arrive matched on the target."
                        )
                    },
                    "range_km": {
                        "type": "number",
                        "minimum": 0,
                        "description": (
                            "HOLD_RANGE only: the range to keep (default: the range when the "
                            "order starts; renewing the order keeps that range)."
                        )
                    },
                    "evade": {
                        "type": "string",
                        "enum": ["full", "dodge"],
                        "description": (
                            "Evasion on top of INTERCEPT/HOLD_RANGE/PADLOCK/BRAKE/MAINTAIN: "
                            "while a torpedo or slug is actually inbound, threat-aware evasion "
                            "flies the ship, and the maneuver resumes once it is clear. "
                            "'full' = exactly EVADE under threat (RUNS from guided torpedoes - "
                            "buys PD time, costs closure). 'dodge' = keeps your course against "
                            "a guided torpedo until its final seconds, then presents your "
                            "thickest armor; still jinks slugs and blinded torpedoes. Evasion "
                            "burns at your throttle - full throttle when the maneuver itself "
                            "does not burn (MAINTAIN, PADLOCK, throttle 0). Omit for no evasion."
                        )
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
                        "description": "For turret FIRE_AT_RANGE: maximum range in km (default 500)"
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
            "description": (
                "Radiator control. Manual: extend=true (cooling, panels exposed to fire) or "
                "extend=false (protected, no cooling). Or mode='auto' to hand them to the "
                f"ship: it extends at auto_extend_pct heat (default {_AUTO_PCT:.0f}) and "
                "retracts while a torpedo is inbound. Whatever you choose, radiators are "
                f"FORCED out at {_FORCED_PCT:.0f}% heat and cannot be retracted until heat "
                f"drops below {_FORCED_PCT:.0f}%."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "extend": {
                        "type": "boolean",
                        "description": (
                            "Manual control. True: extend for cooling (vulnerable to damage). "
                            "False: retract for protection (no cooling)."
                        )
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["manual", "auto"],
                        "description": (
                            "'auto' lets the ship manage radiators every tick (extend when "
                            "heat >= auto_extend_pct, retract under torpedo fire). 'manual' "
                            "(default) uses extend."
                        )
                    },
                    "auto_extend_pct": {
                        "type": "number",
                        "description": f"Heat % at which auto mode extends (default {_AUTO_PCT:.0f}).",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_primary_target",
            "description": (
                "Designate an enemy ship as your primary target: guns and INTERCEPT/PADLOCK "
                "follow it. Names can repeat across fleets - the ship id ([beta_3] in your "
                "turn) is unambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Enemy ship name or id (e.g. 'OCS Lancer-1' or 'beta_3')"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Enemy ship id (e.g. 'beta_3'); wins over target_name"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_heading",
            "description": (
                "Fly along a WORLD-frame DIRECTION, or toward a position (toward_point) or a "
                "ship (toward_ship) - give exactly one. With throttle > 0 it turns onto the "
                "heading first, then burns (the drive thrusts along the NOSE): range control - "
                "cancel closure, open range, pass wide, flank (ready-made vectors in RANGE "
                "CONTROL). With throttle 0 it is ATTITUDE ONLY: the nose turns on RCS and you "
                "coast with zero change of velocity (e.g. spinal on the enemy; PADLOCK tracks a "
                "moving target and turns faster). A DIRECTION IS NOT A DESTINATION: direction "
                "(90,55,10) flies along that vector from wherever you are and never brakes, "
                "so to go TO a position copied from an order use toward_point: the ship flies "
                "there and STOPS (burn, flip and brake timed for you, sideways drift "
                "cancelled). toward_ship flies to that ship and falls in alongside it, "
                "matching its velocity (rejoin a friendly; to stand off an enemy use "
                "set_maneuver INTERCEPT with standoff_km, or HOLD_RANGE). Throttle caps every "
                "burn; throttle 0 only turns the nose onto the current bearing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "object",
                        "description": (
                            "Direction vector in WORLD-frame coordinates (will be normalized). "
                            "NOT ship-relative: these are the same world axes used for ship "
                            "positions. Enemy bearings shown as ahead/starboard/above are "
                            "relative to your nose - use your 'Nose pointing' vector to convert. "
                            "A direction, never a position: for a position use toward_point."
                        ),
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "World X component (+x, -x)"
                            },
                            "y": {
                                "type": "number",
                                "description": "World Y component (+y, -y)"
                            },
                            "z": {
                                "type": "number",
                                "description": "World Z component (+z, -z)"
                            }
                        },
                        "required": ["x", "y", "z"]
                    },
                    "toward_point": {
                        "type": "object",
                        "description": (
                            "Instead of direction: a WORLD position in km (the frame of ship "
                            "positions, e.g. a rally point or fleet centre from your orders). "
                            "The ship flies there and stops, holding position at it."
                        ),
                        "properties": {
                            "x": {"type": "number", "description": "World X, km"},
                            "y": {"type": "number", "description": "World Y, km"},
                            "z": {"type": "number", "description": "World Z, km"}
                        },
                        "required": ["x", "y", "z"]
                    },
                    "toward_ship": {
                        "type": "string",
                        "description": (
                            "Instead of direction: a ship name or id. Flies to that ship and "
                            "falls in alongside it, matching its velocity (rejoin a friendly; "
                            "against an enemy use set_maneuver INTERCEPT standoff_km / "
                            "HOLD_RANGE)."
                        )
                    },
                    "throttle": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Throttle level (0.0-1.0). 0 = attitude only: turn the nose, "
                            "no burn, velocity unchanged."
                        )
                    },
                    "evade": {
                        "type": "string",
                        "enum": ["full", "dodge"],
                        "description": (
                            "Optional evasion on top of the heading, as in set_maneuver: "
                            "'full' = EVADE while a threat is inbound, 'dodge' = same but "
                            "never runs from a guided torpedo. Back on the heading when clear. "
                            "At throttle 0 (coast and shoot) the dodge still burns, at full "
                            "throttle."
                        )
                    }
                },
                "required": ["throttle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "OPTIONAL: transmit on the OPEN channel - the ENEMY hears it (at their next "
                "checkpoint). For taunts, surrender demands or answering an enemy hail. Never "
                "put your plans here; in a fleet, coordinate with fleet_signal and report "
                f"with report_to_admiral. Max {MAX_ENEMY_MESSAGES_PER_DECISION} per decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": f"Message text (max {SIGNAL_TEXT_CHARS} chars)."
                    },
                    "recipient": {
                        "type": "string",
                        "enum": ["ALL_ENEMIES", "SPECIFIC"],
                        "description": "ALL_ENEMIES (default) or SPECIFIC: one enemy ship."
                    },
                    "target_ship": {
                        "type": "string",
                        "description": "Enemy ship name or id when recipient is SPECIFIC."
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
            "description": (
                "Surrender your ship. Your ship will stop maneuvering, drift on its "
                "current trajectory, and become untargetable by all weapons. Enemy ships "
                "can no longer shoot at you. In fleet battles, only your ship surrenders - "
                "other friendly ships continue fighting. Use this to preserve your crew "
                "when defeat is inevitable."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "log_note",
            "description": (
                "Captain's log: write a short note to your future self. Your last few "
                "notes are shown back to you at every checkpoint - they are your ONLY "
                "memory of your own intent between decisions. Use it to keep a "
                "multi-checkpoint plan coherent: what you are trying to do, what would "
                "change your mind, what to check next checkpoint. "
                "E.g. 'Pass 1 done, missed. BRAKE now, re-attack from his tail; if his "
                "PD drops below 3 turrets, commit torpedoes.' Keep it under ~250 chars. "
                "Private: nobody else reads it - tell your Admiral with report_to_admiral, "
                "other ships with fleet_signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "The note to your future self (kept until pushed out by newer notes)."
                    }
                },
                "required": ["note"]
            }
        }
    },
]

# Torpedo tool - only available if ship has torpedoes. The physics and salvo
# doctrine live in the cached system prompt (TORPEDOES block); repeating them
# here cost ~1.7k characters per call. The tool's answer is a feasibility
# check flown on the engine: a launch that misses even if an OPENING target
# holds course is refused, other doubtful verdicts come back as warnings.
TORPEDO_TOOL = {
    "type": "function",
    "function": {
        "name": "launch_torpedo",
        "description": (
            "Launch torpedoes at an enemy before the next checkpoint. Check the target's "
            "torpedo line in ENEMIES first: launch at REACHABLE targets; UNREACHABLE, "
            "ESCAPABLE (he only has to EVADE), TOO CLOSE and OVERKILL launches waste rounds. "
            "The engine flies the round before launching: a round that misses even if an "
            "OPENING target holds course is REFUSED; other doubtful launches go ahead and "
            "the warning is shown to you next turn. Ready tubes fire together, the rest "
            "queue for reload and expire at the next checkpoint; rounds arriving together "
            "split enemy point defense - salvo, don't trickle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": (
                        "Rounds to launch before the next checkpoint (ready tubes now, the "
                        "rest on reload). Defaults to 1."
                    ),
                },
                "target_id": {
                    "type": "string",
                    "description": (
                        "Ship ID to target (e.g. 'beta_1'). Defaults to your primary target. "
                        "Use this to split a salvo onto a different ship than your guns."
                    ),
                },
                "override_check": {
                    "type": "boolean",
                    "description": (
                        "Launch even when the engine check says the round cannot reach "
                        "(default false). Only with a reason the check cannot see."
                    ),
                },
            },
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

# Captain -> Admiral report. Replaces respond_to_orders, whose ACK/DEVIATE text
# was stored on the captain and never read by anyone: the Admiral now gets
# every report at the next checkpoint together with an engine compliance line.
REPORT_TO_ADMIRAL_TOOL = {
    "type": "function",
    "function": {
        "name": "report_to_admiral",
        "description": (
            "Report to your Admiral, who reads it at the NEXT checkpoint before new orders, "
            "beside an engine compliance line (ordered vs flown maneuver, target, salvo vs "
            "rounds launched). Plain compliance needs no report: silence = ACK. DEVIATE when "
            "you deliberately do otherwise (stale intel, target separating or out of reach, "
            "torpedoes force EVADE, a better shot) - say what you see and do instead. UNABLE "
            "when you cannot comply (magazine empty, weapon lost, no delta-v, heat)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(REPORT_STATUSES),
                    "description": (
                        "ACK only to add a contact or request: an ACK with nothing else "
                        "(intent included) is filed as 'nothing to flag'."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        f"DEVIATE/UNABLE: required, max {REPORT_REASON_CHARS} chars - the fact "
                        "the Admiral lacks and what you do instead."
                    ),
                },
                "intent": {
                    "type": "string",
                    "description": (
                        f"Optional, max {REPORT_INTENT_CHARS} chars: your plan for the next "
                        "checkpoint, read beside a DEVIATE, UNABLE, contact or request."
                    ),
                },
                "contact": {
                    "type": "object",
                    "description": "Optional sensor report on one enemy ship.",
                    "properties": {
                        "ship": {"type": "string"},
                        "assessment": {"type": "string", "enum": list(CONTACT_ASSESSMENTS)},
                        "note": {"type": "string", "description": f"Max {CONTACT_NOTE_CHARS} chars."},
                    },
                    "required": ["ship", "assessment"],
                },
                "request": {
                    "type": "object",
                    "description": "Optional request for support.",
                    "properties": {
                        "kind": {"type": "string", "enum": list(REQUEST_KINDS)},
                        "text": {"type": "string", "description": f"Max {REQUEST_TEXT_CHARS} chars."},
                    },
                    "required": ["kind"],
                },
            },
            "required": ["status"],
        },
    },
}

# Captain -> captain coordination on the fleet net. send_message is the enemy
# channel; this one never leaves the fleet.
FLEET_SIGNAL_TOOL = {
    "type": "function",
    "function": {
        "name": "fleet_signal",
        "description": (
            "Signal your OWN fleet (the enemy never hears it), delivered at the NEXT "
            "checkpoint. Launches, kills, PD losses and target switches are shared "
            "automatically - signal intent: TARGET_CLAIM (who takes which hull - focus fire, "
            "no overkill), TOT_SYNC (when YOUR rounds land, eta_s, so peers launch to arrive "
            "together), PD_COVER_REQUEST (to= an escort, when your PD cannot blind what is "
            "inbound), BREAKING_OFF (you leave your post - say why). to=ADMIRAL is filed on "
            f"your report. Max {MAX_FLEET_SIGNALS_PER_DECISION} per decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(SIGNAL_KINDS)},
                "to": {
                    "type": "string",
                    "description": "ALL (default), ADMIRAL, or a friendly ship name/id.",
                },
                "target": {"type": "string", "description": "Enemy ship the signal is about."},
                "eta_s": {"type": "number", "minimum": 0,
                          "description": "TOT_SYNC: seconds until your rounds arrive."},
                "text": {"type": "string", "description": f"Max {SIGNAL_TEXT_CHARS} chars."},
            },
            "required": ["kind", "text"],
        },
    },
}

# Legacy tool: no longer offered (report_to_admiral replaces it), but a
# captain calling it - an old prompt, a recorded tool call, a test - is
# executed as an alias of report_to_admiral.
RESPOND_TO_ORDERS_TOOL = {
    "type": "function",
    "function": {
        "name": "respond_to_orders",
        "description": (
            "Legacy alias of report_to_admiral: acknowledge your Admiral's orders or "
            "explain why you are deviating. The Admiral reads it at the next checkpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response_type": {
                    "type": "string",
                    "enum": ["ACKNOWLEDGE", "DEVIATE"],
                    "description": (
                        "ACKNOWLEDGE: You will follow the Admiral's orders as given. "
                        "DEVIATE: You have tactical reasons to do something different."
                    )
                },
                "deviation_reason": {
                    "type": "string",
                    "description": (
                        "If DEVIATE: Explain WHY you are deviating and WHAT you will do instead. "
                        "Required if response_type is DEVIATE. Be specific about your reasoning."
                    )
                },
                "acknowledgment_note": {
                    "type": "string",
                    "description": (
                        "Optional: Brief note on how you will execute the orders. "
                        "E.g., 'Targeting OCS Grok-1 as ordered, intercepting at full throttle.'"
                    )
                }
            },
            "required": ["response_type"]
        }
    }
}


def radiator_command_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise set_radiators tool arguments into the engine command.

    mode='auto' -> {'type': 'set_radiators', 'mode': 'auto', 'auto_extend_pct': X}
    otherwise   -> {'type': 'set_radiators', 'extend': bool}
    Shared by LLM captains, heuristic captains and MCP so all get the same rules.
    """
    args = args or {}
    mode = str(args.get("mode") or "manual").strip().lower()
    if mode == "auto":
        pct = args.get("auto_extend_pct")
        try:
            pct = float(pct) if pct is not None else _AUTO_PCT
        except (TypeError, ValueError):
            pct = _AUTO_PCT
        return {"type": "set_radiators", "mode": "auto",
                "auto_extend_pct": max(0.0, min(100.0, pct))}
    extend = args.get("extend", False)
    if isinstance(extend, str):
        extend = extend.strip().lower() in ("true", "1", "yes", "extend", "extended")
    return {"type": "set_radiators", "extend": bool(extend)}


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

        if wtype in ("spinal_coiler_mk3", "heavy_siege_coiler_mk3"):
            # Both standard spinal and siege coiler go to "spinal" group
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

    # Property names MUST be exactly f"{group_key}_mode" etc. The executor looks
    # orders up by weapon-group key, so a singular "coilgun_mode" against a
    # "coilguns" group silently drops every turret order.
    properties: Dict[str, Any] = {}
    description_parts = ["Set fire control orders for weapons."]

    # Spinal mode (if ship has spinal - could be standard or siege coiler)
    if "spinal" in groups:
        # Determine which spinal weapon the ship has
        spinal_slot = groups["spinal"][0] if groups["spinal"] else None
        spinal_weapon_type = None
        if spinal_slot and ship_type in fleet_data.get("ships", {}):
            for w in fleet_data["ships"][ship_type].get("weapons", []):
                if w.get("slot") == spinal_slot:
                    spinal_weapon_type = w.get("type")
                    break

        # Get specs for the actual spinal weapon (siege or standard)
        if spinal_weapon_type == "heavy_siege_coiler_mk3":
            spinal_spec = weapon_types.get("heavy_siege_coiler_mk3", {})
            weapon_name = "Siege Coiler"
            gimbal = 20
        else:
            spinal_spec = weapon_types.get("spinal_coiler_mk3", {})
            weapon_name = "Spinal Coiler"
            gimbal = 30

        vel = spinal_spec.get("muzzle_velocity_kps", 9.9)
        dmg = spinal_spec.get("kinetic_energy_gj", 4.32)
        rng = spinal_spec.get("range_km", 900)

        properties["spinal_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"{weapon_name} ({vel} km/s, {dmg:.1f} GJ, {rng}km range, requires nose within {gimbal}° of target)"
        }
        properties["spinal_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For spinal FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["spinal_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": f"For spinal FIRE_AT_RANGE: maximum range (default {min(500, rng)})"
        }
        description_parts.append(f"SPINAL: {weapon_name} - {vel} km/s, {dmg:.1f} GJ, fixed mount ({gimbal}° arc).")

    # Heavy coilgun mode
    if "heavy_coilguns" in groups:
        heavy_spec = weapon_types.get("heavy_coilgun_mk3", {})
        vel = heavy_spec.get("muzzle_velocity_kps", 7.0)
        dmg = heavy_spec.get("kinetic_energy_gj", 1.22)
        rng = heavy_spec.get("range_km", 600)
        count = len(groups["heavy_coilguns"])

        properties["heavy_coilguns_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"Heavy coilguns x{count} ({vel} km/s, {dmg:.1f} GJ each, {rng}km range, turreted 180° arc)"
        }
        properties["heavy_coilguns_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For heavy coilgun FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["heavy_coilguns_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": "For heavy coilgun FIRE_AT_RANGE: maximum range (default 500)"
        }
        description_parts.append(f"HEAVY COILGUNS x{count}: {vel} km/s, {dmg:.1f} GJ each, turreted.")

    # Standard coilgun mode
    if "coilguns" in groups:
        coil_spec = weapon_types.get("coilgun_mk3", {})
        vel = coil_spec.get("muzzle_velocity_kps", 6.0)
        dmg = coil_spec.get("kinetic_energy_gj", 0.72)
        rng = coil_spec.get("range_km", 500)
        count = len(groups["coilguns"])

        properties["coilguns_mode"] = {
            "type": "string",
            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE"],
            "description": f"Coilguns x{count} ({vel} km/s, {dmg:.2f} GJ each, {rng}km range, turreted 180° arc)"
        }
        properties["coilguns_min_probability"] = {
            "type": "number", "minimum": 0.1, "maximum": 0.9,
            "description": "For coilgun FIRE_WHEN_OPTIMAL: minimum hit probability (default 0.3)"
        }
        properties["coilguns_max_range_km"] = {
            "type": "number", "minimum": 50, "maximum": rng,
            "description": "For coilgun FIRE_AT_RANGE: maximum range (default 500)"
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


def get_captain_tools_for_ship(ship_type: str, fleet_data: Dict[str, Any], has_torpedoes: bool = False,
                               decision_interval_s: float = 30.0) -> List[Dict[str, Any]]:
    """
    Tools for the weapons this hull actually mounts.

    The generic get_captain_tools() advertises set_weapons_order with
    spinal_mode/turret_mode to every captain regardless of armament. A corvette
    carries neither - only torpedoes and point defense - so its captain spent its
    weapon order on guns it did not have and never reached for launch_torpedo.
    """
    # Start with base tools but EXCLUDE the hardcoded set_weapons_order
    tools = [t for t in CAPTAIN_TOOLS_BASE if t["function"]["name"] != "set_weapons_order"]

    # Only offer fire control if the ship has gun groups to control. A hull with
    # no guns must not be handed an empty weapons tool to fill in.
    groups = get_weapon_groups_for_ship(ship_type, fleet_data)
    if groups:
        tools.insert(1, build_weapon_tool_for_ship(ship_type, fleet_data))

    if has_torpedoes:
        tool = deepcopy(TORPEDO_TOOL)
        wt = fleet_data.get("weapon_types", {}).get("torpedo_launcher", {})
        cooldown = wt.get("cooldown_s", 12)
        launchers = [w for w in fleet_data.get("ships", {}).get(ship_type, {}).get("weapons", [])
                     if w.get("type", "").startswith("torpedo")]
        capacity = sum(min(w.get("magazine", wt.get("magazine", 8)),
                           math.ceil(decision_interval_s / cooldown)) for w in launchers)
        tool["function"]["parameters"]["properties"]["count"]["maximum"] = max(1, capacity)
        tools.append(tool)

    return tools


# Default tools (for destroyers without torpedoes)
CAPTAIN_TOOLS = get_captain_tools(has_torpedoes=False)
