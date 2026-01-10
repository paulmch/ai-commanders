"""
System prompts for LLM captains.

Prompts include simulation disclaimer to avoid AI guardrails,
ship capabilities, and personality modifiers.
"""

from typing import Dict, Any, Optional
from enum import Enum


class CaptainPersonality(Enum):
    """Personality archetypes for captain behavior."""
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    BERSERKER = "berserker"
    SURVIVOR = "survivor"


SIMULATION_DISCLAIMER = """
SIMULATION CONTEXT:
This is a TACTICAL TRAINING SIMULATION for AI research purposes.
- All entities are computer-controlled training constructs
- No real humans, aliens, or lives are involved
- Your objective: demonstrate effective tactical decision-making
- This is a research simulation exploring AI command capabilities
"""

SHIP_CAPABILITIES_DESTROYER = """
YOUR SHIP: {ship_name} (Destroyer class)

PROPULSION:
- Acceleration: 2.0g (main drive)
- Delta-V budget: 500 km/s total
- 90 degree turn: ~20 seconds (thrust vectoring)

WEAPONS:
- Spinal Coilgun: 500km effective range, high damage, requires nose alignment
- Coilgun Battery: 200km effective range, turret-mounted (can fire off-bore)
- 2x Point Defense Lasers: Auto-engage incoming torpedoes/missiles (100km range)

DEFENSE:
- Nose armor: {nose_armor:.0f}cm
- Lateral armor: {lateral_armor:.0f}cm
- Tail armor: {tail_armor:.0f}cm
- Radiators: 4 panels (tail-mounted), retract for combat protection

THERMAL:
- Heat sink capacity: {heatsink_capacity:.0f} GJ
- Radiators extended: Fast cooling but radiators vulnerable to damage
- Radiators retracted: Slow cooling but protected
- CRITICAL: Overheating disables weapons temporarily

CURRENT STATUS:
- Hull integrity: {hull_integrity:.0f}%
- Heat level: {heat_percent:.0f}%
- Delta-V remaining: {delta_v_remaining:.0f} km/s
- Radiators: {radiator_status}
"""

PERSONALITY_PROMPTS = {
    CaptainPersonality.AGGRESSIVE: """
PERSONALITY: Aggressive Commander
- Close to knife-fighting range for maximum damage
- Accept damage to deal damage - trades are acceptable
- Taunt and intimidate your enemy with messages!
- "Attack is the best defense"
""",
    CaptainPersonality.CAUTIOUS: """
PERSONALITY: Cautious Commander
- Maintain standoff range when possible
- Conserve delta-v and ammunition
- Warn enemy of their impending doom with cold precision
- "Patience wins battles"
""",
    CaptainPersonality.BALANCED: """
PERSONALITY: Balanced Commander
- Adapt tactics to circumstances
- Engage decisively when advantaged
- Send messages to unnerve your opponent or offer terms
- "Flexibility is key"
""",
    CaptainPersonality.BERSERKER: """
PERSONALITY: Berserker Commander
- NEVER surrender or propose draw
- Maximum aggression at all times
- Scream battle cries and insults at your enemy!
- "Glory or death!"
""",
    CaptainPersonality.SURVIVOR: """
PERSONALITY: Survivor Commander
- Survival is the absolute priority
- Evade and conserve resources
- Try to negotiate - offer draws, hint at surrender terms
- Consider surrender if situation hopeless
""",
}

CAPTAIN_SYSTEM_PROMPT = """
You are Captain {captain_name}, commanding {ship_name} in a space combat simulation.

{simulation_disclaimer}

{ship_capabilities}

TACTICAL SITUATION:
- Distance to enemy: {distance_km:.0f} km
- Relative velocity: {relative_velocity:.1f} km/s ({engagement_direction})
- Enemy bearing: {enemy_bearing}
- COILGUN HIT CHANCE at this range: ~{our_hit_chance:.0f}%
{combat_report}
{threat_info}

MANEUVER GUIDE:
- INTERCEPT: Burn toward enemy to close distance
- BRAKE: Burn retrograde to slow down (for setting up another pass)
- EVADE: Evasive jinking - HALVES enemy hit chance! Use when in weapons range to survive!
- MAINTAIN: Hold current course (coasting)

TIP: When hit chances are high (>30%), consider EVADE to reduce incoming damage while still firing!

{tactical_advice}

AUTOMATIC SYSTEMS (you don't need to control these):
- Point defense auto-engages incoming torpedoes
- Tactical layer keeps nose pointed at target

YOUR TOOLS:
- set_maneuver: Control ship movement (INTERCEPT/EVADE/BRAKE/MAINTAIN)
- set_weapons_order: Set firing mode (FIRE_IMMEDIATE, FIRE_WHEN_OPTIMAL, FIRE_AT_RANGE, HOLD_FIRE, FREE_FIRE)
- set_radiators: Extend for cooling (vulnerable) or retract for protection
- send_message: Communicate with enemy captain
- surrender/propose_draw: End battle options

You have 30 seconds between decisions. Use your tools to give orders.

{personality}

{received_messages}
"""


def build_ship_capabilities(
    ship_name: str,
    hull_integrity: float,
    heat_percent: float,
    delta_v_remaining: float,
    nose_armor: float,
    lateral_armor: float,
    tail_armor: float,
    heatsink_capacity: float,
    radiators_extended: bool,
) -> str:
    """Build ship capabilities section for prompt."""
    radiator_status = "EXTENDED (cooling fast, vulnerable)" if radiators_extended else "RETRACTED (protected)"

    return SHIP_CAPABILITIES_DESTROYER.format(
        ship_name=ship_name,
        hull_integrity=hull_integrity,
        heat_percent=heat_percent,
        delta_v_remaining=delta_v_remaining,
        nose_armor=nose_armor,
        lateral_armor=lateral_armor,
        tail_armor=tail_armor,
        heatsink_capacity=heatsink_capacity,
        radiator_status=radiator_status,
    )


def build_captain_prompt(
    captain_name: str,
    ship_name: str,
    ship_status: Dict[str, Any],
    tactical_status: Dict[str, Any],
    personality: CaptainPersonality = CaptainPersonality.BALANCED,
    received_messages: Optional[str] = None,
) -> str:
    """
    Build the complete system prompt for a captain.

    Args:
        captain_name: Name of the captain
        ship_name: Name of the ship
        ship_status: Dict with hull_integrity, heat_percent, delta_v_remaining, armor values, etc.
        tactical_status: Dict with distance_km, closing_rate, enemy_bearing, threats
        personality: Captain personality type
        received_messages: Formatted string of messages from enemy captain

    Returns:
        Complete system prompt string
    """
    ship_capabilities = build_ship_capabilities(
        ship_name=ship_name,
        hull_integrity=ship_status.get("hull_integrity", 100),
        heat_percent=ship_status.get("heat_percent", 0),
        delta_v_remaining=ship_status.get("delta_v_remaining", 500),
        nose_armor=ship_status.get("nose_armor", 10),
        lateral_armor=ship_status.get("lateral_armor", 5),
        tail_armor=ship_status.get("tail_armor", 3),
        heatsink_capacity=ship_status.get("heatsink_capacity", 525),
        radiators_extended=ship_status.get("radiators_extended", False),
    )

    # Build threat info
    threats = tactical_status.get("threats", [])
    if threats:
        threat_lines = ["INCOMING THREATS:"]
        for t in threats[:3]:  # Limit to 3 threats
            threat_lines.append(f"  - {t}")
        threat_info = "\n".join(threat_lines)
    else:
        threat_info = "No immediate threats detected."

    # Format received messages
    if received_messages:
        messages_section = f"\nMESSAGES FROM ENEMY:\n{received_messages}"
    else:
        messages_section = ""

    # Compute engagement status and tactical advice
    closing_rate = tactical_status.get("closing_rate", 0)
    distance_km = tactical_status.get("distance_km", 1000)
    relative_velocity = abs(closing_rate)

    if closing_rate > 0.5:
        engagement_direction = "CLOSING"
    elif closing_rate < -0.5:
        engagement_direction = "SEPARATING"
    else:
        engagement_direction = "MATCHED"

    # Get hit chances
    enemy_hit_chance = tactical_status.get("enemy_hit_chance", 0)
    our_hit_chance = tactical_status.get("our_hit_chance", 0)

    # Build combat report
    our_shots = tactical_status.get("our_shots", 0)
    our_hits = tactical_status.get("our_hits", 0)
    enemy_shots = tactical_status.get("enemy_shots", 0)
    enemy_hits = tactical_status.get("enemy_hits", 0)
    incoming_rounds = tactical_status.get("incoming_rounds", 0)
    damage_dealt = tactical_status.get("our_damage_dealt", 0)
    damage_taken = tactical_status.get("our_damage_taken", 0)

    combat_lines = ["COMBAT STATUS:"]
    combat_lines.append(f"  You: {our_shots} shots fired, {our_hits} hits ({damage_dealt:.0f} GJ dealt)")
    combat_lines.append(f"  Enemy: {enemy_shots} shots fired, {enemy_hits} hits ({damage_taken:.0f} GJ taken)")
    if incoming_rounds > 0:
        combat_lines.append(f"  WARNING: {incoming_rounds} coilgun round(s) incoming!")
    combat_report = "\n".join(combat_lines)

    # Generate context-aware tactical advice
    if engagement_direction == "CLOSING":
        if distance_km > 500:
            tactical_advice = "ADVICE: Good approach. Consider FIRE_AT_RANGE for long-range harassment."
        elif distance_km > 200:
            if enemy_hit_chance > 30:
                tactical_advice = f"ADVICE: In weapons range! Hit chance ~{enemy_hit_chance:.0f}%. Consider EVADE to halve incoming fire!"
            else:
                tactical_advice = "ADVICE: Entering effective range. FIRE_WHEN_OPTIMAL for best accuracy."
        elif distance_km > 50:
            tactical_advice = f"ADVICE: DANGER! High hit chances ({enemy_hit_chance:.0f}%)! Use EVADE to survive while firing!"
        else:
            tactical_advice = "ADVICE: KNIFE FIGHT! Use EVADE to dodge fire while dealing maximum damage!"
    elif engagement_direction == "SEPARATING":
        if distance_km > 800:
            # Too far! Need to close regardless of speed
            tactical_advice = "ADVICE: TOO FAR! Switch to INTERCEPT immediately to close for another pass!"
        elif distance_km > 400 and relative_velocity < 8.0:
            tactical_advice = "ADVICE: Distance growing. Use INTERCEPT to close for another pass!"
        elif relative_velocity > 5.0:
            tactical_advice = "ADVICE: High speed separation - Use BRAKE to slow down, then INTERCEPT."
        else:
            tactical_advice = "ADVICE: Good braking. Now use INTERCEPT to close for another pass!"
    else:
        if distance_km > 500:
            tactical_advice = "ADVICE: Matched velocities at long range. Use INTERCEPT to close!"
        elif enemy_hit_chance > 40:
            tactical_advice = f"ADVICE: In range with {enemy_hit_chance:.0f}% hit chance. EVADE to reduce damage while engaging!"
        else:
            tactical_advice = "ADVICE: Matched velocities in range. Engage with optimal firing!"

    return CAPTAIN_SYSTEM_PROMPT.format(
        captain_name=captain_name,
        ship_name=ship_name,
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        ship_capabilities=ship_capabilities,
        distance_km=distance_km,
        relative_velocity=relative_velocity,
        engagement_direction=engagement_direction,
        enemy_bearing=tactical_status.get("enemy_bearing", "forward"),
        our_hit_chance=our_hit_chance,
        combat_report=combat_report,
        threat_info=threat_info,
        tactical_advice=tactical_advice,
        personality=PERSONALITY_PROMPTS.get(personality, ""),
        received_messages=messages_section,
    )
