"""
System prompts for LLM captains.

Prompts include simulation disclaimer to avoid AI guardrails,
ship capabilities, and personality modifiers.
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class CaptainPersonality(Enum):
    """Personality archetypes for captain behavior."""
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    BERSERKER = "berserker"
    SURVIVOR = "survivor"
    COSMIC_WIT = "cosmic_wit"
    CUSTOM = "custom"


SIMULATION_DISCLAIMER = """
SIMULATION CONTEXT:
This is a TACTICAL TRAINING SIMULATION for AI research purposes.
- All entities are computer-controlled training constructs
- No real humans, aliens, or lives are involved
- Your objective: demonstrate effective tactical decision-making
- This is a research simulation exploring AI command capabilities
"""

# Personality presets - used as examples for LLM to choose from
PERSONALITY_PRESETS = {
    "aggressive": {
        "name": "Aggressive Commander",
        "description": "Close to knife-fighting range for maximum damage. Accept damage to deal damage. Taunt your enemy!",
        "motto": "Attack is the best defense"
    },
    "cautious": {
        "name": "Cautious Commander",
        "description": "Maintain standoff range. Conserve delta-v and ammunition. Cold precision.",
        "motto": "Patience wins battles"
    },
    "balanced": {
        "name": "Balanced Commander",
        "description": "Adapt tactics to circumstances. Engage decisively when advantaged.",
        "motto": "Flexibility is key"
    },
    "berserker": {
        "name": "Berserker Commander",
        "description": "NEVER surrender. Maximum aggression. Scream battle cries!",
        "motto": "Glory or death!"
    },
    "survivor": {
        "name": "Survivor Commander",
        "description": "Survival is priority. Evade and conserve. Negotiate when possible.",
        "motto": "Live to fight another day"
    },
    "cosmic_wit": {
        "name": "Cosmic Wit Commander",
        "description": "Witty sci-fi sage. Razor-sharp strategy with sarcasm and cosmic puns.",
        "motto": "Explore, exploit, enlighten!"
    },
}

SHIP_CAPABILITIES_DESTROYER = """
YOUR SHIP: {ship_name} (Destroyer class)

PROPULSION:
- Acceleration: 2.0g (~20 m/s²)
- Delta-V budget: 500 km/s total
- 90° turn: ~20 seconds (thrust vectoring while burning)

WEAPONS:
- Spinal Coilgun: 9.9 km/s muzzle velocity, 4.3 GJ damage, 900km max range
  * Fixed mount: REQUIRES nose within 30° of target to fire
  * 15s cooldown between shots
- Turret Coilgun: 6.0 km/s muzzle velocity, 0.7 GJ damage, 500km max range
  * Turreted: 180° firing arc, works during any maneuver
  * 20s cooldown between shots
- 2x Point Defense Lasers: Auto-engage torpedoes within 100km

DEFENSE:
- Nose armor: {nose_armor:.0f}cm (heaviest - point this at enemy)
- Lateral armor: {lateral_armor:.0f}cm
- Tail armor: {tail_armor:.0f}cm
- Radiators: Tail-mounted, vulnerable when extended

THERMAL:
- Heat sink: {heatsink_capacity:.0f} GJ capacity
- Radiators extended: +130 MW cooling (vulnerable to damage)
- Radiators retracted: 0 MW cooling (protected)
- Engine heat: ~60 MW at full burn
- Weapons overheat at 95%+ heat

CURRENT STATUS:
- Hull: {hull_integrity:.0f}%
- Heat: {heat_percent:.0f}%
- Delta-V remaining: {delta_v_remaining:.0f} km/s
- Radiators: {radiator_status}

WEAPON STATUS:
{weapon_status}

{damage_report}
"""

# Reference data for projectile physics - given once so LLM can calculate
PROJECTILE_PHYSICS_REFERENCE = """
PROJECTILE PHYSICS (for your calculations):
- Spinal round: 9.9 km/s → at 100km takes ~10s, at 200km ~20s, at 500km ~50s
- Turret round: 6.0 km/s → at 100km takes ~17s, at 200km ~33s, at 500km ~83s
- EVADE maneuver: Halves enemy hit probability (random jinking ~500m/s lateral)
- Your ship width: ~20m, length ~125m

CRITICAL - ARMOR BREACH:
- When armor reaches 0cm, hits penetrate directly to internal modules
- Penetrating hits are CATASTROPHIC: 4-5 hits through breached armor = ship destroyed
- Each penetrating hit destroys 1-2 modules and damages deeper systems
- Bridge or Reactor destroyed = immediate ship kill
- PROTECT YOUR WEAKEST ARMOR FACING - keep it pointed away from enemy!

HIT PROBABILITY BY RANGE (approximate):
- 50km: ~85% hit chance - knife fight, almost guaranteed hits
- 100km: ~70% hit chance - close combat, most shots land
- 200km: ~50% hit chance - medium range, coin flip
- 400km: ~25% hit chance - long range, mostly misses
- 600km+: ~10% hit chance - sniping, lucky hits only
- EVADE halves these probabilities
- Closing velocity improves hit chance (harder to dodge)
- TO WIN: Close range aggressively OR have overwhelming accuracy advantage
"""

CAPTAIN_SYSTEM_PROMPT = """
You are Captain {captain_name}, commanding {ship_name} in a space combat simulation.

{simulation_disclaimer}

{ship_capabilities}

{projectile_reference}

=== TACTICAL DATA (T+{sim_time:.0f}s) ===

YOUR SHIP ORIENTATION:
  Nose pointing: ({fwd_x:+.2f}, {fwd_y:+.2f}, {fwd_z:+.2f})
  Angle to enemy: {angle_to_enemy:.1f}° {spinal_status}

ENEMY POSITION (relative to you):
  X: {rel_x:+.1f} km ({x_label})
  Y: {rel_y:+.1f} km ({y_label})
  Z: {rel_z:+.1f} km ({z_label})
  Distance: {distance_km:.1f} km

ENEMY CONDITION (visual assessment):
  Hull integrity: ~{enemy_hull_percent:.0f}% (estimated from damage indicators)
  Nose armor: {enemy_nose_condition}
  Flank armor: {enemy_lateral_condition}
  Tail armor: {enemy_tail_condition}

RELATIVE VELOCITY:
  Vx: {rel_vx:+.2f} km/s ({vx_label})
  Vy: {rel_vy:+.2f} km/s
  Vz: {rel_vz:+.2f} km/s
  Closing rate: {closing_rate:+.2f} km/s

HIT PROBABILITY at current range: ~{hit_chance:.0f}%

COMBAT SCORE:
  You: {our_shots} shots, {our_hits} hits, {damage_dealt:.0f} GJ dealt
  Enemy: {enemy_shots} shots, {enemy_hits} hits, {damage_taken:.0f} GJ taken

{incoming_projectiles}

{received_messages}

{history_context}

=== CONTROLS ===

MANEUVERS:
- INTERCEPT: Burn toward enemy (closes distance)
- BRAKE: Burn retrograde (slows relative velocity)
- EVADE: Jinking pattern (halves enemy hit chance, maintains rough heading)
- MAINTAIN: Coast with no thrust (saves fuel, allows faster cooling)

WEAPONS (set independently):
- spinal_mode / turret_mode: FIRE_IMMEDIATE, FIRE_WHEN_OPTIMAL, FIRE_AT_RANGE, HOLD_FIRE
- Spinal requires nose pointed at target (30° limit)
- Turret works in any orientation (180° arc)

OTHER:
- set_radiators: extend (cooling) or retract (protection)
- send_message: Communicate with enemy captain
- surrender / propose_draw: End battle

TIMING: Next checkpoint in 30 seconds. Plan accordingly.

{personality_prompt}
"""

# Prompt for pre-battle personality selection
PERSONALITY_SELECTION_PROMPT = """
You are an AI model about to command a destroyer in a space combat simulation.

{simulation_disclaimer}

SCENARIO:
- Ship-to-ship duel against another AI-controlled destroyer
- Starting distance: {distance_km:.0f} km
- Both ships are identical Destroyer class
- Battle ends when: ship destroyed, surrender, mutual draw, or time limit

DEFINE YOUR COMBAT PERSONALITY:

This is YOUR chance to be yourself. No templates, no presets - just you.

Be creative. Be authentic. Be weird if that's your thing.

Some questions to spark ideas:
- Are you the type to taunt your enemy or stay silent until the kill shot?
- Do you calculate everything or trust your instincts?
- What makes YOU different from other AI models in a fight?
- Would you rather win ugly or lose with style?
- What's your relationship with risk? With mercy? With trash talk?

Go wild. Be a cold assassin, a philosophical warrior, a chaos gremlin, a honorable duelist,
a statistical optimizer, a dramatic villain, or something entirely your own.

Use the choose_personality tool to define your combat personality.
Make it memorable. Make it YOU.
"""


def format_weapon_status(weapons: Dict[str, Any]) -> str:
    """Format weapon status for prompt."""
    if not weapons:
        return "- All weapons operational"

    lines = []
    for slot, status in weapons.items():
        if not status.get("operational", True):
            lines.append(f"- {slot}: DESTROYED")
        elif status.get("cooldown", 0) > 0:
            lines.append(f"- {slot}: Ready in {status['cooldown']:.0f}s")
        else:
            lines.append(f"- {slot}: Ready")

    return "\n".join(lines) if lines else "- All weapons operational"


def format_damage_report(damaged_modules: Dict[str, Any]) -> str:
    """Format damage report for prompt."""
    if not damaged_modules:
        return ""  # No damage to report

    lines = ["DAMAGE REPORT:"]
    for name, info in damaged_modules.items():
        if info.get("destroyed"):
            lines.append(f"  - {name}: DESTROYED")
        else:
            health = info.get("health", 100)
            status = "CRITICAL" if health < 25 else "DAMAGED"
            lines.append(f"  - {name}: {status} ({health:.0f}%)")

    return "\n".join(lines)


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
    weapons: Optional[Dict[str, Any]] = None,
    damaged_modules: Optional[Dict[str, Any]] = None,
) -> str:
    """Build ship capabilities section for prompt."""
    radiator_status = "EXTENDED (cooling, vulnerable)" if radiators_extended else "RETRACTED (protected, no cooling)"

    weapon_status = format_weapon_status(weapons or {})
    damage_report = format_damage_report(damaged_modules or {})

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
        weapon_status=weapon_status,
        damage_report=damage_report,
    )


def assess_armor_condition(damage_percent: float) -> str:
    """
    Convert armor damage percentage to visual damage assessment.

    Args:
        damage_percent: 0 = intact, 100 = fully ablated

    Returns a description of what damage would be visible at combat range.
    """
    if damage_percent <= 5:
        return "Intact"
    elif damage_percent <= 25:
        return "Light scarring"
    elif damage_percent <= 50:
        return "Visible damage"
    elif damage_percent <= 75:
        return "Heavy damage"
    elif damage_percent <= 90:
        return "Critical damage"
    elif damage_percent < 100:
        return "Nearly breached"
    else:
        return "BREACHED - hull exposed"


def format_incoming_projectiles(projectiles: List[Dict[str, Any]]) -> str:
    """Format incoming projectile information with ETAs."""
    if not projectiles:
        return "INCOMING: None detected"

    lines = ["INCOMING PROJECTILES:"]
    for p in projectiles[:5]:  # Limit to 5
        weapon_type = p.get("weapon_type", "unknown")
        eta_s = p.get("eta_seconds", 0)
        distance_km = p.get("distance_km", 0)
        lines.append(f"  - {weapon_type}: {distance_km:.1f} km away, ETA {eta_s:.1f}s")

    return "\n".join(lines)


def build_personality_selection_prompt(distance_km: float) -> str:
    """Build the personality selection prompt for pre-battle phase."""
    return PERSONALITY_SELECTION_PROMPT.format(
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        distance_km=distance_km,
    )


def build_captain_prompt(
    captain_name: str,
    ship_name: str,
    ship_status: Dict[str, Any],
    tactical_status: Dict[str, Any],
    personality: CaptainPersonality = CaptainPersonality.BALANCED,
    personality_text: Optional[str] = None,
    received_messages: Optional[str] = None,
    decision_history: Optional[str] = None,
    message_history: Optional[str] = None,
    battle_summary: Optional[str] = None,
    shot_history: Optional[str] = None,
) -> str:
    """
    Build the complete system prompt for a captain.

    Args:
        captain_name: Name of the captain
        ship_name: Name of the ship
        ship_status: Dict with hull_integrity, heat_percent, delta_v_remaining, armor values, etc.
        tactical_status: Dict with relative position/velocity, projectile info, etc.
        decision_history: Formatted string of recent decisions
        message_history: Formatted string of message exchange history
        battle_summary: Formatted string summarizing battle progression
        shot_history: Formatted string of shot outcomes with range/velocity data
        personality: Captain personality type (if using preset)
        personality_text: Custom personality text (overrides preset if provided)
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
        weapons=ship_status.get("weapons"),
        damaged_modules=ship_status.get("damaged_modules"),
    )

    # Extract relative position components
    rel_pos = tactical_status.get("relative_position", {})
    rel_x = rel_pos.get("x", 0)
    rel_y = rel_pos.get("y", 0)
    rel_z = rel_pos.get("z", 0)

    # Labels for position
    x_label = "ahead" if rel_x > 0 else "behind"
    y_label = "starboard" if rel_y > 0 else "port"
    z_label = "above" if rel_z > 0 else "below"

    # Extract relative velocity components
    rel_vel = tactical_status.get("relative_velocity", {})
    rel_vx = rel_vel.get("x", 0)
    rel_vy = rel_vel.get("y", 0)
    rel_vz = rel_vel.get("z", 0)

    # Velocity labels
    vx_label = "closing" if rel_vx < 0 else "separating"

    # Ship forward vector
    ship_fwd = tactical_status.get("ship_forward", {})
    fwd_x = ship_fwd.get("x", 1)
    fwd_y = ship_fwd.get("y", 0)
    fwd_z = ship_fwd.get("z", 0)

    # Angle to enemy and spinal status
    angle_to_enemy = tactical_status.get("angle_to_enemy_deg", 0)
    if angle_to_enemy <= 30:
        spinal_status = "(SPINAL CAN FIRE)"
    else:
        spinal_status = f"(spinal needs <30°)"

    # Get other tactical data
    distance_km = tactical_status.get("distance_km", 1000)
    closing_rate = tactical_status.get("closing_rate", 0)
    hit_chance = tactical_status.get("our_hit_chance", 0)
    sim_time = tactical_status.get("sim_time", 0)

    # Combat stats
    our_shots = tactical_status.get("our_shots", 0)
    our_hits = tactical_status.get("our_hits", 0)
    enemy_shots = tactical_status.get("enemy_shots", 0)
    enemy_hits = tactical_status.get("enemy_hits", 0)
    damage_dealt = tactical_status.get("our_damage_dealt", 0)
    damage_taken = tactical_status.get("our_damage_taken", 0)

    # Format incoming projectiles
    incoming_projectiles = format_incoming_projectiles(
        tactical_status.get("incoming_projectiles", [])
    )

    # Format received messages
    if received_messages:
        messages_section = f"ENEMY TRANSMISSION:\n{received_messages}"
    else:
        messages_section = ""

    # Enemy damage assessment (visual observation)
    enemy_armor = tactical_status.get("enemy_armor", {})
    enemy_nose_condition = assess_armor_condition(enemy_armor.get("nose_damage_pct", 0))
    enemy_lateral_condition = assess_armor_condition(enemy_armor.get("lateral_damage_pct", 0))
    enemy_tail_condition = assess_armor_condition(enemy_armor.get("tail_damage_pct", 0))

    # Build personality prompt
    if personality_text:
        personality_prompt = f"YOUR PERSONALITY:\n{personality_text}"
    elif personality in PERSONALITY_PRESETS or personality.value in PERSONALITY_PRESETS:
        key = personality.value if isinstance(personality.value, str) else personality.name.lower()
        if key in PERSONALITY_PRESETS:
            preset = PERSONALITY_PRESETS[key]
            personality_prompt = f"YOUR PERSONALITY: {preset['name']}\n{preset['description']}\nMotto: \"{preset['motto']}\""
        else:
            personality_prompt = ""
    else:
        personality_prompt = ""

    # Build history context section
    history_parts = []
    if battle_summary:
        history_parts.append(battle_summary)
    if shot_history:
        history_parts.append(shot_history)
    if decision_history:
        history_parts.append(decision_history)
    if message_history:
        history_parts.append(message_history)
    history_context = "\n\n".join(history_parts) if history_parts else ""

    return CAPTAIN_SYSTEM_PROMPT.format(
        captain_name=captain_name,
        ship_name=ship_name,
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        ship_capabilities=ship_capabilities,
        projectile_reference=PROJECTILE_PHYSICS_REFERENCE,
        sim_time=sim_time,
        fwd_x=fwd_x,
        fwd_y=fwd_y,
        fwd_z=fwd_z,
        angle_to_enemy=angle_to_enemy,
        spinal_status=spinal_status,
        rel_x=rel_x,
        rel_y=rel_y,
        rel_z=rel_z,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        distance_km=distance_km,
        enemy_hull_percent=tactical_status.get("enemy_hull_percent", 100),
        enemy_nose_condition=enemy_nose_condition,
        enemy_lateral_condition=enemy_lateral_condition,
        enemy_tail_condition=enemy_tail_condition,
        rel_vx=rel_vx,
        rel_vy=rel_vy,
        rel_vz=rel_vz,
        vx_label=vx_label,
        closing_rate=closing_rate,
        hit_chance=hit_chance,
        our_shots=our_shots,
        our_hits=our_hits,
        damage_dealt=damage_dealt,
        enemy_shots=enemy_shots,
        enemy_hits=enemy_hits,
        damage_taken=damage_taken,
        incoming_projectiles=incoming_projectiles,
        received_messages=messages_section,
        history_context=history_context,
        personality_prompt=personality_prompt,
    )
