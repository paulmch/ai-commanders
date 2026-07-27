"""
System prompts for LLM captains.

Prompts include simulation disclaimer to avoid AI guardrails,
ship capabilities, and personality modifiers.
"""

from typing import Dict, Any, Optional, List
from enum import Enum

try:
    from ..pointdefense import (
        PDLaser,
        PD_ABSORPTIVITY,
        TORPEDO_ELECTRONICS_THRESHOLD_J,
        full_coupling_range_m,
    )
except ImportError:  # pragma: no cover - script-style imports
    from pointdefense import (
        PDLaser,
        PD_ABSORPTIVITY,
        TORPEDO_ELECTRONICS_THRESHOLD_J,
        full_coupling_range_m,
    )


def pd_doctrine_numbers(fleet_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Point-defense figures for prompts, derived from the SAME model the engine
    runs (PDLaser + the continuous-dwell closed form), never hand-quoted.

    Returns:
        range_km: engagement envelope
        power_mw: beam power per turret
        full_coupling_km: range inside which the whole beam lands on a torpedo
        blind_closure_kps_per_turret: max closure ONE turret can seeker-kill
            (50 MJ) dwelling continuously from envelope entry to impact
        seeker_kill_mj: absorbed energy for a guidance kill
        waste_heat_mw: heat rejected into own hull per firing turret
    """
    spec = (fleet_data or {}).get("weapon_types", {}).get("pd_laser", {})
    laser = PDLaser.from_fleet_data(spec) if spec else PDLaser()
    r_full_m = full_coupling_range_m(
        aperture_m=laser.aperture_m, wavelength_m=laser.wavelength_m
    )
    r0_m = laser.range_km * 1000.0
    coupling_integral_m = (
        r0_m if r0_m <= r_full_m else 2.0 * r_full_m - r_full_m * r_full_m / r0_m
    )
    blind_closure_mps = (
        laser.power_w * PD_ABSORPTIVITY * coupling_integral_m
        / TORPEDO_ELECTRONICS_THRESHOLD_J
    )
    return {
        "range_km": laser.range_km,
        "power_mw": laser.power_mw,
        "full_coupling_km": r_full_m / 1000.0,
        "blind_closure_kps_per_turret": blind_closure_mps / 1000.0,
        "seeker_kill_mj": TORPEDO_ELECTRONICS_THRESHOLD_J / 1e6,
        "waste_heat_mw": laser.waste_heat_w() / 1e6,
    }


def get_ship_turn_time_90deg(ship_type: str) -> float:
    """Calculate approximate 90-degree turn time using thrust vectoring."""
    # Use the TV times from CLAUDE.md reference
    turn_times = {
        "corvette": 12.1,
        "frigate": 15.1,
        "destroyer": 20.6,
        "cruiser": 28.2,
        "battlecruiser": 28.2,
        "battleship": 36.9,
        "dreadnought": 49.9,
    }
    return turn_times.get(ship_type.lower(), 20.0)


def format_weapon_groups_for_prompt(weapons: List[Dict], weapon_types: Dict[str, Any]) -> str:
    """Format weapon information grouped by type."""
    lines = []

    # Group weapons by type
    spinals = []
    heavy_coilguns = []
    coilguns = []
    pd_lasers = []

    for w in weapons:
        wtype = w.get("type", "")
        slot = w.get("slot", "")
        if wtype == "spinal_coiler_mk3":
            spinals.append(w)
        elif wtype == "heavy_coilgun_mk3":
            heavy_coilguns.append(w)
        elif wtype == "coilgun_mk3":
            coilguns.append(w)
        elif wtype == "pd_laser":
            pd_lasers.append(w)

    # Format each group
    if spinals:
        spec = weapon_types.get("spinal_coiler_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 9.9)
        dmg = spec.get("kinetic_energy_gj", 4.32)
        rng = spec.get("range_km", 900)
        cd = spec.get("cooldown_s", 15)
        lines.append(f"- Spinal Coilgun: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage, {rng}km max range")
        lines.append(f"  * Fixed mount: REQUIRES nose within 30° of target to fire")
        lines.append(f"  * {cd}s cooldown between shots")

    if heavy_coilguns:
        spec = weapon_types.get("heavy_coilgun_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 7.0)
        dmg = spec.get("kinetic_energy_gj", 1.22)
        rng = spec.get("range_km", 600)
        cd = spec.get("cooldown_s", 18)
        count = len(heavy_coilguns)
        lines.append(f"- Heavy Coilguns x{count}: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage each, {rng}km max range")
        lines.append(f"  * Turreted: 180° firing arc, works during any maneuver")
        lines.append(f"  * {cd}s cooldown between shots")

    if coilguns:
        spec = weapon_types.get("coilgun_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 6.0)
        dmg = spec.get("kinetic_energy_gj", 0.72)
        rng = spec.get("range_km", 500)
        cd = spec.get("cooldown_s", 20)
        count = len(coilguns)
        lines.append(f"- Coilguns x{count}: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage each, {rng}km max range")
        lines.append(f"  * Turreted: 180° firing arc, works during any maneuver")
        lines.append(f"  * {cd}s cooldown between shots")

    if pd_lasers:
        count = len(pd_lasers)
        # Every figure below is derived from the same PDLaser model the engine
        # runs (see pd_doctrine_numbers) - the prompt used to hardcode a 100km
        # envelope while the sim used 250km.
        pd = pd_doctrine_numbers({"weapon_types": weapon_types})
        per = pd["blind_closure_kps_per_turret"]
        lines.append(
            f"- {count}x Point Defense Lasers ({pd['power_mw']:.0f} MW beam): "
            f"FULLY AUTOMATIC - engage inbound torpedoes within {pd['range_km']:.0f}km, "
            f"no orders needed."
        )
        lines.append(
            f"  * Optics: the whole beam lands only inside ~{pd['full_coupling_km']:.0f} km; "
            f"at the envelope edge most of it misses, so lethality rises steeply as a "
            f"threat closes."
        )
        lines.append(
            f"  * A torpedo is killed by cooking its seeker ({pd['seeker_kill_mj']:.0f} MJ "
            f"absorbed). Dwelling from envelope entry to impact, ONE turret stops at most "
            f"~{per:.0f} km/s of closure; each turret adds ~{per:.0f} km/s (your {count}: "
            f"~{count * per:.0f} km/s). Guided torpedoes arrive at 13-28 km/s - alone you "
            f"stop only the slow ones. Allied turrets in range of the threat stack with yours."
        )
        lines.append(
            "  * A PD 'kill' BLINDS the torpedo - the wreck coasts on ballistically and "
            "STILL HITS a ship flying straight. PD blinds; your EVADE makes blinded "
            "rounds miss. Never hold course under torpedo fire."
        )
        lines.append(
            f"  * Firing cost: ~{pd['waste_heat_mw']:.0f} MW waste heat per firing turret, "
            f"no thrust penalty - trivial next to your heat sink. PD cost is never a "
            f"reason to change your plan."
        )

    return "\n".join(lines)


def format_enemy_weapons_summary(weapons: List[Dict]) -> str:
    """Format a short summary of enemy weapons."""
    counts = {}
    for w in weapons:
        wtype = w.get("type", "")
        if wtype == "spinal_coiler_mk3":
            counts["Spinal"] = counts.get("Spinal", 0) + 1
        elif wtype == "heavy_coilgun_mk3":
            counts["Heavy Coilgun"] = counts.get("Heavy Coilgun", 0) + 1
        elif wtype == "coilgun_mk3":
            counts["Coilgun"] = counts.get("Coilgun", 0) + 1
        elif wtype == "pd_laser":
            counts["PD"] = counts.get("PD", 0) + 1

    parts = [f"{count}x {name}" for name, count in counts.items()]
    return ", ".join(parts) if parts else "Unknown armament"


# Mirrors geometry.NOSE_HIT_ANGLE_THRESHOLD / TAIL_HIT_ANGLE_THRESHOLD. A test
# asserts these stay in sync - the captain acts on these numbers.
NOSE_HIT_CONE_DEG = 30.0
TAIL_HIT_CONE_DEG = 150.0


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
- 2x Point Defense Lasers: Auto-engage torpedoes within 250km

DEFENSE - ARMOR IS ASPECT-DEPENDENT (this is the single biggest survivability lever):
{armor_aspect_block}
- A round is scored against whichever facing it arrives at: within {nose_cone_deg:.0f}° of your
  nose it strikes NOSE armor, beyond that it strikes LATERAL, and from behind
  (past {tail_cone_deg:.0f}° off the nose) it strikes TAIL.
- So keeping the threat inside {nose_cone_deg:.0f}° of your nose is worth far more than any
  evasion: the moment you turn past it you present your thinnest plating.
- The dangerous moment is the BREAK-AWAY turn after an attack run - that is when
  you swing your flank through the enemy's guns.

THERMAL:
- Heat sink: {heatsink_capacity:.0f} GJ capacity
- Radiators extended: +130 MW cooling
- Radiators retracted: 0 MW cooling
- Engine heat: ~60 MW at full burn
- Extended radiators are exposed and CAN be shot off: a hit that strikes them
  permanently reduces your cooling. Retracting protects them but gives up all
  cooling, so this is a real tradeoff.
- Heat above 95% is flagged CRITICAL, but weapons keep firing: this simulation
  applies no weapon lockout for heat.
"""

SHIP_CAPABILITIES_TEMPLATE = """
YOUR SHIP: {ship_name} ({ship_class} class)

PROPULSION:
- Acceleration: {accel_g}g (~{accel_mps:.0f} m/s²)
- Delta-V budget: {delta_v_total:.0f} km/s total
- 90° turn: ~{turn_time:.0f} seconds (thrust vectoring while burning)

WEAPONS:
{weapons_section}
{torpedo_block}

DEFENSE - ARMOR IS ASPECT-DEPENDENT (this is the single biggest survivability lever):
{armor_aspect_block}
- A round is scored against whichever facing it arrives at: within {nose_cone_deg:.0f}° of your
  nose it strikes NOSE armor, beyond that it strikes LATERAL, and from behind
  (past {tail_cone_deg:.0f}° off the nose) it strikes TAIL.
- So keeping the threat inside {nose_cone_deg:.0f}° of your nose is worth far more than any
  evasion: the moment you turn past it you present your thinnest plating.
- The dangerous moment is the BREAK-AWAY turn after an attack run - that is when
  you swing your flank through the enemy's guns.

THERMAL:
- Heat sink: {heatsink_capacity:.0f} GJ capacity
- Radiators extended: +{cooling_mw:.0f} MW cooling
- Radiators retracted: 0 MW cooling
- Engine heat: ~60 MW at full burn
- Extended radiators are exposed and CAN be shot off: a hit that strikes them
  permanently reduces your cooling. Retracting protects them but gives up all
  cooling, so this is a real tradeoff.
- Heat above 95% is flagged CRITICAL, but weapons keep firing: this simulation
  applies no weapon lockout for heat.

"""

# Volatile per-turn status. Kept OUT of the system prompt so the stable doctrine
# above can be served from the provider's prompt cache across checkpoints.
SHIP_STATUS_TEMPLATE = """
CURRENT STATUS:
- Hull: {hull_integrity:.0f}%
- Heat: {heat_percent:.0f}%
- Delta-V remaining: {delta_v_remaining:.0f}/{delta_v_total:.0f} km/s
- Radiators: {radiator_status}
- Point defense: {pd_status}
- Armor: Nose {nose_armor:.1f}cm | Lateral {lateral_armor:.1f}cm | Tail {tail_armor:.1f}cm

WEAPON STATUS:
{weapon_status}

{damage_report}
"""


def _armor_aspect_block(nose_cm: float, lateral_cm: float, tail_cm: float) -> str:
    """
    Render the armor facings with their real thicknesses and the ratio between
    them, so the captain can see how much aspect actually matters rather than
    being told "nose is heaviest".
    """
    thinnest = min(lateral_cm, tail_cm) or 1.0
    ratio = nose_cm / thinnest if thinnest else 1.0
    lines = [
        f"- Nose armor:    {nose_cm:6.1f} cm  (heaviest - keep this toward the threat)",
        f"- Lateral armor: {lateral_cm:6.1f} cm",
        f"- Tail armor:    {tail_cm:6.1f} cm  (radiators mounted here)",
        f"- Your nose is {ratio:.1f}x thicker than your thinnest facing: taking a hit "
        f"nose-on instead of side-on is worth roughly {ratio:.1f}x your effective armor.",
    ]
    return "\n".join(lines)


def armor_from_fleet_data(ship_type: str, fleet_data: Dict[str, Any]) -> Dict[str, float]:
    """Starting armor thickness per facing, derived the same way the engine does."""
    spec = (fleet_data or {}).get("ships", {}).get(ship_type, {})
    armor = spec.get("armor", {})
    dist = armor.get("distribution", {})
    mass_pct = dist.get("mass_percent", {})
    area_pct = dist.get("area_percent", {})
    density = armor.get("properties", {}).get("density", 1800)
    total_mass_kg = armor.get("total_mass_tons", 0) * 1000
    total_area_m2 = armor.get("total_surface_m2", 0)

    out = {}
    for facing in ("nose", "lateral", "tail"):
        area = total_area_m2 * area_pct.get(facing, 0) / 100.0
        mass = total_mass_kg * mass_pct.get(facing, 0) / 100.0
        out[facing] = (mass / (density * area) * 100.0) if area and density else 0.0
    return out


def _torpedo_capability_block(ship_type: str, fleet_data: Dict[str, Any],
                             decision_interval_s: float = 30.0) -> str:
    """
    Static torpedo capability summary for a ship that mounts launchers.

    Returns "" for hulls without torpedoes so gun ships are not told about a
    weapon they do not have.
    """
    spec = (fleet_data or {}).get("ships", {}).get(ship_type, {})
    launchers = [w for w in spec.get("weapons", [])
                 if (w.get("type") or "").startswith("torpedo")]
    if not launchers:
        return ""

    wt = (fleet_data or {}).get("weapon_types", {}).get("torpedo_launcher", {})
    cooldown = wt.get("cooldown_s", 12)
    # Per-slot magazine overrides the weapon-type default (the torpedo cruiser
    # carries 12/launcher against the type's 8) - quote what the hull loads.
    magazine = launchers[0].get("magazine", wt.get("magazine", 8))
    total_rounds = sum(
        l.get("magazine", wt.get("magazine", 8)) for l in launchers
    )
    dv = wt.get("delta_v_kps", 14.0)
    accel = wt.get("acceleration_g", 12.0)
    pen = wt.get("penetrator_mass_kg", 250)
    rng = wt.get("range_km", 2500)
    n = len(launchers)
    per_decision = max(1, int(decision_interval_s // cooldown)) * n
    pd = pd_doctrine_numbers(fleet_data)

    def ke(v_kps):
        return 0.5 * pen * (v_kps * 1000) ** 2 / 1e9

    # The impact-speed anchors below are MEASURED engine behaviour (guidance
    # holds a >=12 km/s closure floor and burns its delta-v surplus in the
    # terminal phase), not launch closure + dv. tests/test_doctrine_accuracy.py
    # re-measures them.
    return f"""
TORPEDOES ({n}x {wt.get('name', 'launcher')}) - your decisive weapon:
- {pen:.0f} kg kinetic penetrator, {dv} km/s own delta-v, {accel}g, {rng} km reach
- Magazine: {magazine} rounds per launcher ({total_rounds} total). Reload {cooldown}s
  -> up to {per_decision} launches per decision.
- DAMAGE SCALES WITH THE SQUARE OF CLOSING SPEED. Guidance never lets closure drop
  below ~12 km/s and dumps its remaining delta-v just before impact. Measured against
  a 2g hull (400 km launch):
      launched with zero closure     -> impacts ~13.5 km/s = ~{ke(13.5):.0f} GJ
      launched closing 12 km/s       -> impacts ~14-17 km/s = ~{ke(14):.0f}-{ke(17):.0f} GJ
      launched closing 26 km/s       -> impacts ~28 km/s   = ~{ke(28):.0f} GJ
      lobbed at a RECEDING ship      -> ~{ke(8):.0f} GJ if it lands at all; a receding
      ship that keeps running usually outlasts the round's delta-v. No tail chases.
  (A spinal gun round is ~4.3 GJ: a plain closing launch = ~5 gun hits, head-on ~20.)
- A CLOSING launch vs a lone ship almost always connects: no hull (<=3g) out-turns
  {accel}g guidance, and lone-ship point defense only stops slow closures (each PD
  turret kills ~{pd['blind_closure_kps_per_turret']:.0f} km/s of closure; you arrive at 13+).
  Expect near-100% on stragglers, heavy attrition against an escorted evading wall.
- SALVO TIMING IS DECISIVE: rounds arriving TOGETHER split enemy PD dwell; the same
  rounds trickled out are blinded one by one, early enough to be dodged. Measured vs
  4 PD turrets: 4 simultaneous -> 24/24 through; the same 4 spaced 30s -> 12/24.
- Launch range barely matters vs a closing target (measured 200-2400 km: all connect).
  Longer flights arrive HARDER (more speed built) but cost time: ~400 km lands in
  30-60s (one decision cycle), 1500 km in ~2 minutes.
"""


def format_torpedo_threats(threats: List[Dict[str, Any]]) -> str:
    """
    Render inbound torpedoes with enough detail to act on.

    Beyond range/closure/ETA, each threat can carry (all optional, filled in by
    captain._build_tactical_status from the live engine):
      est_impact_gj  - projected impact energy (penetrator KE at projected speed)
      nez_inside     - True: provably cannot be outrun; False: still escapable
      pd_turrets_needed / own_pd_turrets - seeker-kill triage for THIS threat
      blinded        - seeker already killed; ballistic, evadable
    """
    if not threats:
        return "INBOUND ORDNANCE: none detected"

    lines = [f"*** INBOUND ORDNANCE: {len(threats)} TORPEDO(S) TRACKED ***"]
    for t in threats:
        eta = t.get("eta_seconds", 999)
        eta_str = f"IMPACT IN {eta:.0f}s" if eta < 900 else "not closing"
        head = "BLINDED torpedo" if t.get("blinded") else "torpedo"
        line = (
            f"  - {head} from {t.get('source', 'unknown')}: "
            f"{t.get('distance_km', 0):.0f} km, "
            f"closing {t.get('closing_kps', 0):.1f} km/s, {eta_str}"
        )
        if t.get("est_impact_gj") is not None:
            line += f", est ~{t['est_impact_gj']:.0f} GJ"
        if t.get("blinded"):
            line += " | ballistic wreck: KEEP MANEUVERING and it misses"
        else:
            nez = t.get("nez_inside")
            if nez is True:
                line += " | NEZ: CANNOT be outrun"
            elif nez is False:
                line += " | NEZ: still escapable - RUN"
            needed = t.get("pd_turrets_needed")
            own = t.get("own_pd_turrets")
            if needed is not None and own is not None:
                verdict = "your PD can blind it" if needed <= own else \
                    "your PD alone CANNOT stop it"
                line += f" | PD kill needs ~{min(needed, 99)} turret(s), you have {own}: {verdict}"
        lines.append(line)
    lines.append(
        "  PD is automatic. It BLINDS torpedoes; a blinded round still hits a ship "
        "flying straight - EVADE (auto-RUN, then auto-PRESENT of your nose armor) "
        "is what turns PD kills into misses."
    )
    return "\n".join(lines)


def build_ship_status_block(
    hull_integrity: float,
    heat_percent: float,
    delta_v_remaining: float,
    delta_v_total: float,
    nose_armor: float,
    lateral_armor: float,
    tail_armor: float,
    radiators_extended: bool,
    weapons: Optional[Dict[str, Any]] = None,
    damaged_modules: Optional[Dict[str, Any]] = None,
    pd_turrets_operational: Optional[int] = None,
    pd_turrets_total: Optional[int] = None,
) -> str:
    """
    Build the volatile per-checkpoint ship status block.

    Split out of the capabilities section so the static ship specification can
    live in a cacheable system prompt while this changes every checkpoint.
    """
    # Radiator damage IS live in the engine now (simulation.py resolves radiator
    # hits through RadiatorHitResolver), so the vulnerability is real and the
    # captain should be reasoning about it.
    radiator_status = (
        "EXTENDED (+cooling, exposed to fire)" if radiators_extended
        else "RETRACTED (protected, no cooling)"
    )
    if pd_turrets_total:
        op = pd_turrets_total if pd_turrets_operational is None \
            else pd_turrets_operational
        pd_status = f"{op}/{pd_turrets_total} turrets operational (automatic)"
        if op < pd_turrets_total:
            pd_status += " - DEGRADED"
    else:
        pd_status = "none fitted" if pd_turrets_total == 0 else "operational (automatic)"
    return SHIP_STATUS_TEMPLATE.format(
        pd_status=pd_status,
        hull_integrity=hull_integrity,
        heat_percent=heat_percent,
        delta_v_remaining=delta_v_remaining,
        delta_v_total=delta_v_total,
        radiator_status=radiator_status,
        nose_armor=nose_armor,
        lateral_armor=lateral_armor,
        tail_armor=tail_armor,
        weapon_status=format_weapon_status(weapons or {}),
        damage_report=format_damage_report(damaged_modules or {}),
    )


def build_ship_capabilities_from_fleet(
    ship_name: str,
    ship_type: str,
    fleet_data: Dict[str, Any],
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
    """Build ship capabilities section dynamically from fleet data."""
    ship_spec = fleet_data["ships"].get(ship_type, {})
    weapon_types = fleet_data.get("weapon_types", {})

    # Get performance data (acceleration, delta-v)
    performance = ship_spec.get("performance", {})
    accel_g = performance.get("combat_acceleration_g", 2.0)
    accel_mps = performance.get("combat_acceleration_ms2", accel_g * 9.81)
    delta_v_total = performance.get("delta_v_kps", 500)

    # Get turn time
    turn_time = get_ship_turn_time_90deg(ship_type)

    # Format weapons
    ship_weapons = ship_spec.get("weapons", [])
    weapons_section = format_weapon_groups_for_prompt(ship_weapons, weapon_types)

    # Format status
    radiator_status = "EXTENDED (cooling)" if radiators_extended else "RETRACTED (no cooling)"
    weapon_status = format_weapon_status(weapons or {})
    damage_report = format_damage_report(damaged_modules or {})

    # Get ship class name (title case of ship_type)
    ship_class = ship_type.title()

    # Radiator cooling straight from fleet data (RadiatorArray.from_thermal_data uses
    # the same mass x dissipation product), so the prompt cannot drift from the sim.
    radiator_spec = ship_spec.get("thermal", {}).get("radiator", {})
    cooling_mw = (
        radiator_spec.get("mass_tons", 10.0) * 1000.0
        * radiator_spec.get("dissipation_kw_per_kg", 13.0)
    ) / 1000.0

    _armor = armor_from_fleet_data(ship_type, fleet_data)
    _torpedoes = _torpedo_capability_block(ship_type, fleet_data)
    return SHIP_CAPABILITIES_TEMPLATE.format(
        torpedo_block=_torpedoes,
        armor_aspect_block=_armor_aspect_block(
            _armor.get("nose", 0.0), _armor.get("lateral", 0.0), _armor.get("tail", 0.0)
        ),
        nose_cone_deg=NOSE_HIT_CONE_DEG,
        tail_cone_deg=TAIL_HIT_CONE_DEG,
        ship_name=ship_name,
        ship_class=ship_class,
        accel_g=accel_g,
        accel_mps=accel_mps,
        delta_v_total=delta_v_total,
        turn_time=turn_time,
        weapons_section=weapons_section,
        heatsink_capacity=heatsink_capacity,
        cooling_mw=cooling_mw,
        hull_integrity=hull_integrity,
        heat_percent=heat_percent,
        delta_v_remaining=delta_v_remaining,
        radiator_status=radiator_status,
        nose_armor=nose_armor,
        lateral_armor=lateral_armor,
        tail_armor=tail_armor,
        weapon_status=weapon_status,
        damage_report=damage_report,
    )


# Reference data for projectile physics - given once so LLM can calculate
PROJECTILE_PHYSICS_REFERENCE = """
PROJECTILE PHYSICS (for your calculations):
- Spinal round: 9.9 km/s → at 100km takes ~10s, at 200km ~20s, at 500km ~50s
- Turret round: 6.0 km/s → at 100km takes ~17s, at 200km ~33s, at 500km ~83s
- EVADE maneuver: cuts enemy hit probability by ~40% (x0.6 multiplier in the
  fire-control model), via random jinking ~500m/s lateral
- Your ship width: ~20m, length ~125m

CRITICAL - ARMOR BREACH:
- When armor reaches 0cm, hits penetrate directly to internal modules
- Penetrating hits are CATASTROPHIC: 4-5 hits through breached armor = ship destroyed
- Each penetrating hit destroys 1-2 modules and damages deeper systems
- Bridge or Reactor destroyed = immediate ship kill
- PROTECT YOUR WEAKEST ARMOR FACING - keep it pointed away from enemy!

HIT PROBABILITY BY RANGE (spinal, 9.9 km/s, vs a non-evading destroyer):
- 50km: ~82% - knife fight, almost guaranteed hits
- 100km: ~64% - close combat, most shots land
- 200km: ~39% - medium range, and already worse than a coin flip
- 400km: ~17% - long range, mostly misses
- 600km+: ~10% - sniping, lucky hits only
Turrets (6.0 km/s) are worse at every range, and evasion multiplies whatever is left
by 0.6: an evading target at 200km is ~20% for turrets, ~23% for the spinal.
- Closing velocity improves hit chance (harder to dodge)
- Larger hulls are easier to hit; corvettes are much harder than dreadnoughts
- TO WIN: Close range aggressively OR have overwhelming accuracy advantage
"""

# Stable across every checkpoint of a battle -> cacheable prefix.
CAPTAIN_DOCTRINE = """
You are Captain {captain_name}, commanding {ship_name} in a space combat simulation.

{simulation_disclaimer}

{ship_capabilities}

{projectile_reference}

=== CONTROLS ===

MANEUVERS:
- INTERCEPT: THRUST toward target - actively closes distance, builds velocity toward enemy
- EVADE: Threat-aware defense, BEST during active combat. Versus gun rounds it jinks
  (hard to hit). Versus guided torpedoes it automatically RUNS (burns away from the
  round to cut closure and buy your PD dwell time) and in the final seconds
  automatically PRESENTS your thickest armor to the impact. Only EVADE does this -
  any other maneuver takes torpedoes, including blinded ones, without reacting.
- BRAKE: Slow down - use when closing too fast (>3 km/s relative)
- MAINTAIN: Coast - no thrust, no rotation, pure drift
- PADLOCK: ROTATE ONLY, NO THRUST - keeps nose pointed at target but does NOT move toward it!
           Use PADLOCK for tracking/firing while coasting. You will NOT close distance!
- set_heading: Fly specific direction - for angled approaches, flanking, disengaging
  NOTE ON FRAMES: set_heading takes a WORLD-frame direction vector (x, y, z world axes),
  NOT a ship-relative one. Enemy bearings printed as ahead/starboard/above are relative
  to YOUR NOSE; the "Nose pointing" vector above tells you how the two frames line up.

*** CRITICAL: PADLOCK vs INTERCEPT ***
PADLOCK @ 70% = Rotate to track target, NO closing thrust. You stay on current trajectory!
INTERCEPT @ 70% = Thrust TOWARD target at 70% throttle. You actively close distance!
To CLOSE RANGE: use INTERCEPT or set_heading with vector toward enemy. PADLOCK will NOT close!

=== TACTICAL OPTIONS (pick your style) ===

You can fight however you want. Here are some approaches to consider:

OPTION A - HEAD-ON CHARGE:
  INTERCEPT constantly → high closing velocity → brief exchange → fly past
  Pros: Simple, aggressive, intimidating
  Cons: Short engagement window, hard to hit at high relative velocity, wastes fuel

OPTION B - CONTROLLED PASSES (recommended for accuracy):
  1. INTERCEPT or set_heading to approach (you must thrust to close distance!)
  2. PADLOCK when in range - coast while tracking, fire spinal during the pass
     (PADLOCK only rotates to track - you coast on momentum, no thrust toward target)
  3. After pass: BRAKE to kill velocity, set up another pass
  Pros: Better hit probability, more sustained fire, fuel efficient
  Cons: Takes longer to close, requires planning

OPTION C - EVADE AND SHOOT:
  EVADE constantly - random thrust makes you unpredictable
  Pros: Very hard to hit, fires continuously
  Cons: Burns fuel faster, harder to line up spinal shots

OPTION D - RANGE CONTROL:
  Use set_heading to maintain preferred range (e.g., 100-200km)
  BRAKE if closing too fast, INTERCEPT if separating
  Pros: Controls engagement tempo, picks your range
  Cons: Requires more active management

KEY PHYSICS TO REMEMBER:
- High relative velocity = harder to hit (projectiles lead wrong, brief window)
- 0 km/s relative + no evasion = sitting duck (easy hit)
- Sweet spot: 1-3 km/s relative with active maneuvering
- Spinal weapons ONLY fire if nose within 30° of target - PADLOCK helps with this
- Turrets fire at 180° arc - don't need nose-on alignment

=== TORPEDO DEFENSE (act on this the moment ordnance is inbound) ===
- Your PD is automatic; it BLINDS torpedoes (seeker kill at 50 MJ absorbed beam), it
  does not vaporise them. A blinded round coasts ballistically on its last course and
  STILL HITS a ship flying straight. PD + EVADE stops torpedoes; PD alone stops nothing.
- Each PD turret can blind ~5 km/s of closure (continuous dwell across the whole
  envelope); guided rounds arrive at 13-28 km/s. A lone ship therefore does NOT stop
  a committed closing torpedo - EVADE, take it on the nose armor, kill the shooter.
- A CLOSING torpedo is inside its No-Escape Zone almost from launch: no hull (max 3g)
  out-turns 12g guidance. But if you are already OPENING range when it launches,
  keep running - it burns out its delta-v and misses (measured: reliable escape).
- Timing: launch-to-impact is typically 15-60s, so you get AT MOST ONE decision after
  detection. If an enemy torpedo hull has you targeted, set EVADE preemptively.
- Formation defense is real: allied PD turrets in range of the threat's path stack
  with yours (that is what escorts are for). Massed turrets blind rounds EARLY,
  which is what gives your evasion time to make the wreck miss.

DELTA-V NOTE: You have ~500 km/s delta-V. Full throttle for 20 minutes uses <50 km/s.
Fuel is ABUNDANT - "conserving delta-V" is almost never correct.

THROTTLE NOTE: All maneuvers accept throttle 0.0-1.0. In fleet battles, your Admiral may
order specific throttle (e.g., "INTERCEPT at 50%") to maintain formation with slower ships.
Follow throttle orders if given - formation cohesion matters for mutual point defense.

AMMO NOTE: 450+ spinal rounds, 1800+ turret rounds. Max ~80 shots per battle.
Ammo is effectively UNLIMITED. Shoot freely.

TARGET SELECTION:
- set_primary_target: Designate which enemy to focus weapons/maneuvers on
- CLOSING targets = HIGH priority - they're committed, easier to hit
- SEPARATING targets = LOW priority - chasing wastes time and fuel
- In fleet battles: capital ships (slow to turn) are easier targets than corvettes

WEAPONS (set independently):
- spinal_mode / turret_mode: FIRE_IMMEDIATE, FIRE_WHEN_OPTIMAL, FIRE_AT_RANGE, HOLD_FIRE
- Spinal: High damage, requires nose-on (30° limit)
- Turret: Lower damage, 180° firing arc (more flexible)

COMMUNICATION (optional - use sparingly):
- send_message: Taunt enemies, demand surrender, coordinate with allies
- Not required every checkpoint - only when meaningful

OTHER:
- set_radiators: extend (cooling) or retract (protection)
- surrender / propose_draw: End battle (draw decided by points)

*** CRITICAL: YOU MUST CALL THE TOOLS EVERY CHECKPOINT ***
Describing what you will do is NOT the same as doing it. You MUST call:
- set_maneuver to actually move (or you will DRIFT and lose all momentum)
- set_primary_target to actually target (or target will be NONE)
- set_weapons_order to actually fire
Even if you want to "maintain course" - you MUST call set_maneuver again!
There is NO auto-continue. Every checkpoint resets to DRIFT unless you call tools.

TIMING: You make ONE decision now. It is LOCKED for 30 seconds until next checkpoint.
You CANNOT react mid-checkpoint or change orders. No "if X then Y" - decide NOW.
Your ship will execute your orders for 30 seconds regardless of what happens.

{personality_prompt}
"""

# Rebuilt every checkpoint. Must come AFTER the doctrine in the request.
CAPTAIN_TURN_STATE = """=== TACTICAL DATA (T+{sim_time:.0f}s) ===
{ship_status_block}

YOUR SHIP:
  Nose pointing: ({fwd_x:+.2f}, {fwd_y:+.2f}, {fwd_z:+.2f})
  Angle to primary target: {angle_to_enemy:.1f}° {spinal_status}

YOUR CURRENT CONFIGURATION:
{current_config}

{battlefield_overview}

{combat_statistics}

{incoming_projectiles}

{torpedo_threats}

{recent_hits}

{received_messages}

{history_context}
"""

# Backwards-compatible single-string form (doctrine + state).
CAPTAIN_SYSTEM_PROMPT = CAPTAIN_DOCTRINE + "\n\n" + CAPTAIN_TURN_STATE


# Prompt for pre-battle personality selection
PERSONALITY_SELECTION_PROMPT = """
You are {model_name}, about to command a {ship_class} in a space combat simulation.

{simulation_disclaimer}

SCENARIO:
- {battle_description}
- Starting distance: {distance_km:.0f} km
- Your ship: {ship_class} class
- Battle ends when: ship destroyed, surrender, mutual draw, or time limit

DEFINE YOUR COMBAT PERSONALITY:

This is YOUR chance to be yourself. No templates, no presets - just you, {model_name}.

Be creative. Be authentic. Be weird if that's your thing.

Some questions to spark ideas:
- Are you the type to taunt your enemy or stay silent until the kill shot?
- Do you calculate everything or trust your instincts?
- What makes {model_name} unique as a combat commander?
- Would you rather win ugly or lose with style?
- What's your relationship with risk? With mercy? With trash talk?

Go wild. Be a cold assassin, a philosophical warrior, a chaos gremlin, a honorable duelist,
a statistical optimizer, a dramatic villain, or something entirely your own.

Use the choose_personality tool to define your combat personality.
Make it memorable. Make it uniquely {model_name}.
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
    ship_type: str = "destroyer",
    fleet_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build ship capabilities section for prompt."""
    if fleet_data:
        return build_ship_capabilities_from_fleet(
            ship_name=ship_name,
            ship_type=ship_type,
            fleet_data=fleet_data,
            hull_integrity=hull_integrity,
            heat_percent=heat_percent,
            delta_v_remaining=delta_v_remaining,
            nose_armor=nose_armor,
            lateral_armor=lateral_armor,
            tail_armor=tail_armor,
            heatsink_capacity=heatsink_capacity,
            radiators_extended=radiators_extended,
            weapons=weapons,
            damaged_modules=damaged_modules,
        )

    # Legacy fallback using destroyer template
    radiator_status = "EXTENDED (cooling)" if radiators_extended else "RETRACTED (no cooling)"
    weapon_status = format_weapon_status(weapons or {})
    damage_report = format_damage_report(damaged_modules or {})

    return SHIP_CAPABILITIES_DESTROYER.format(
        torpedo_block="",
        armor_aspect_block=_armor_aspect_block(151.2, 26.0, 30.3),
        nose_cone_deg=NOSE_HIT_CONE_DEG,
        tail_cone_deg=TAIL_HIT_CONE_DEG,
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
    """Format incoming projectile information with ETAs, sources, and bearings."""
    if not projectiles:
        return "INCOMING: None detected"

    lines = ["INCOMING PROJECTILES:"]
    for p in projectiles[:5]:  # Limit to 5
        weapon_type = p.get("weapon_type", "unknown")
        eta_s = p.get("eta_seconds", 0)
        distance_km = p.get("distance_km", 0)
        source = p.get("source", "Unknown")
        bearing = p.get("bearing", "")

        if bearing:
            lines.append(f"  - {weapon_type} from {source}: {distance_km:.1f} km, ETA {eta_s:.1f}s, bearing {bearing}")
        else:
            lines.append(f"  - {weapon_type} from {source}: {distance_km:.1f} km away, ETA {eta_s:.1f}s")

    return "\n".join(lines)


def format_battlefield_overview(enemies: List[Dict[str, Any]], friendlies: List[Dict[str, Any]], fleet_data: Optional[Dict[str, Any]] = None) -> str:
    """Format multi-ship battlefield overview."""
    lines = ["=== BATTLEFIELD OVERVIEW ==="]

    if enemies:
        lines.append("\nENEMY SHIPS:")
        for e in enemies:
            is_primary = e.get("is_primary_target", False)
            has_us_targeted = e.get("has_us_targeted", False)

            # Header with primary target marker - show both name and ID
            name = e.get("name", e.get("ship_id", "Unknown"))
            ship_id = e.get("ship_id", "")
            # ship_class is often absent/"unknown"; ship_type is the authoritative
            # field on the simulation Ship, so derive the class from it when we can.
            ship_class = e.get("ship_class") or "ship"
            if str(ship_class).lower() in ("", "unknown", "ship") and e.get("ship_type"):
                ship_class = str(e["ship_type"]).title()
            # Show ID in brackets so captain can match Admiral orders
            id_display = f" [{ship_id}]" if ship_id and ship_id != name else ""
            if is_primary:
                lines.append(f"  [PRIMARY TARGET] {name}{id_display} ({ship_class}):")
            else:
                lines.append(f"  {name}{id_display} ({ship_class}):")

            # Position, expressed in THIS ship's body frame so that
            # "ahead/starboard/above" actually mean what they say.
            bearing = e.get("relative_bearing_km")
            if bearing:
                f_km = bearing.get("forward", 0.0)
                s_km = bearing.get("starboard", 0.0)
                u_km = bearing.get("up", 0.0)
                lines.append(
                    f"    Bearing: {abs(f_km):.1f} km {'ahead' if f_km >= 0 else 'behind'}, "
                    f"{abs(s_km):.1f} km {'starboard' if s_km >= 0 else 'port'}, "
                    f"{abs(u_km):.1f} km {'above' if u_km >= 0 else 'below'} (relative to your nose)"
                )
            else:
                rel_pos = e.get("relative_position", {})
                x, y, z = rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)
                lines.append(
                    f"    Offset (world frame): x{x:+.1f} y{y:+.1f} z{z:+.1f} km"
                )

            # Distance and closing rate
            distance = e.get("distance_km", 0)
            closing = e.get("closing_rate", 0)
            closing_label = "closing" if closing > 0 else "separating"
            lines.append(f"    Distance: {distance:.1f} km | {closing_label}: {abs(closing):.2f} km/s")

            # Angle and spinal status
            angle = e.get("angle_deg", 0)
            spinal_ok = "(SPINAL ALIGNED)" if angle <= 30 else "(spinal needs <30°)"
            lines.append(f"    Angle: {angle:.1f}° {spinal_ok}")

            # Condition
            hull = e.get("hull_percent", 100)
            armor = e.get("armor", {})
            nose_cond = assess_armor_condition(armor.get("nose_damage_pct", 0))
            lateral_cond = assess_armor_condition(armor.get("lateral_damage_pct", 0))
            lines.append(f"    Hull: {hull:.0f}% | Nose: {nose_cond} | Flank: {lateral_cond}")

            # Add enemy ship capabilities if fleet_data available
            enemy_ship_type = e.get("ship_type", "")
            if fleet_data and enemy_ship_type and enemy_ship_type in fleet_data.get("ships", {}):
                spec = fleet_data["ships"][enemy_ship_type]
                performance = spec.get("performance", {})
                accel = performance.get("combat_acceleration_g", "?")
                delta_v = performance.get("delta_v_kps", "?")
                turn_time = get_ship_turn_time_90deg(enemy_ship_type)
                enemy_weapons = spec.get("weapons", [])
                weapons_summary = format_enemy_weapons_summary(enemy_weapons)

                lines.append(f"    Capabilities: {accel}g accel, {delta_v} km/s delta-v, ~{turn_time:.0f}s turn")
                lines.append(f"    Armament: {weapons_summary}")

            # Hit probability
            hit_chance = e.get("hit_chance", 0)
            lines.append(f"    Hit probability: ~{hit_chance:.0f}%")

            # Targeting warning
            if has_us_targeted:
                lines.append(f"    !! HAS YOU TARGETED")

            lines.append("")  # Blank line between ships
    else:
        lines.append("\nENEMY SHIPS: None detected")

    if friendlies:
        lines.append("FRIENDLY SHIPS:")
        for f in friendlies:
            name = f.get("name", f.get("ship_id", "Unknown"))
            distance = f.get("distance_km", 0)
            hull = f.get("hull_percent", 100)
            # Same frame rule as for enemies: "ahead/starboard" are body-frame words
            # and may only be used when a body-frame bearing was supplied. The raw
            # relative_position is a WORLD-frame delta and must be labelled as such.
            bearing = f.get("relative_bearing_km")
            if bearing:
                f_km = bearing.get("forward", 0.0)
                s_km = bearing.get("starboard", 0.0)
                offset = (
                    f"{abs(f_km):.0f} km {'ahead' if f_km >= 0 else 'behind'}, "
                    f"{abs(s_km):.0f} km {'starboard' if s_km >= 0 else 'port'}"
                )
            else:
                rel_pos = f.get("relative_position", {})
                x, y, z = rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)
                offset = f"world frame x{x:+.0f} y{y:+.0f} z{z:+.0f} km"
            lines.append(f"  {name}: {distance:.1f} km ({offset}) | Hull: {hull:.0f}%")

    return "\n".join(lines)


def format_combat_statistics(
    our_shots: int, our_hits: int, our_damage_dealt: float, our_damage_taken: float,
    enemies: List[Dict[str, Any]], friendlies: List[Dict[str, Any]],
    ship_name: str, primary_target_id: Optional[str] = None
) -> str:
    """Format comprehensive combat statistics."""
    lines = ["=== COMBAT STATISTICS ==="]

    # Own ship stats
    accuracy = (our_hits / our_shots * 100) if our_shots > 0 else 0
    lines.append(f"\nYOUR SHIP ({ship_name}):")
    lines.append(f"  Shots: {our_shots} fired, {our_hits} hits ({accuracy:.0f}%)")
    lines.append(f"  Damage dealt: {our_damage_dealt:.1f} GJ | Damage taken: {our_damage_taken:.1f} GJ")

    # Enemy stats
    if enemies:
        lines.append("\nENEMY FORCES:")
        total_enemy_shots = 0
        total_enemy_hits = 0
        total_enemy_damage = 0
        for e in enemies:
            name = e.get("name", e.get("ship_id", "Unknown"))
            shots = e.get("shots_fired", 0)
            hits = e.get("hits_scored", 0)
            damage_dealt = e.get("damage_dealt_gj", 0)
            damage_taken = e.get("damage_taken_gj", 0)

            total_enemy_shots += shots
            total_enemy_hits += hits
            total_enemy_damage += damage_dealt

            is_target = e.get("ship_id") == primary_target_id
            target_marker = " [YOUR TARGET]" if is_target else ""
            lines.append(f"  {name}{target_marker}: {shots} shots, {hits} hits, {damage_dealt:.1f} GJ dealt, {damage_taken:.1f} GJ taken")

    # Friendly stats (if any)
    if friendlies:
        lines.append("\nFRIENDLY FORCES:")
        for f in friendlies:
            name = f.get("name", f.get("ship_id", "Unknown"))
            # Friendlies don't have detailed combat stats in current structure
            hull = f.get("hull_percent", 100)
            lines.append(f"  {name}: Hull {hull:.0f}%")

    return "\n".join(lines)


def format_current_config(
    primary_target_name: Optional[str],
    radiators_extended: bool,
    weapon_orders: Dict[str, str],
    current_maneuver: Optional[Dict[str, Any]] = None,
    evasion_status: Optional[Dict[str, Any]] = None,
) -> str:
    """Format current ship configuration."""
    lines = []

    # Current maneuver
    if current_maneuver:
        maneuver_type = current_maneuver.get("type", "NONE")
        throttle = current_maneuver.get("throttle", 1.0)
        throttle_pct = int(throttle * 100)

        if maneuver_type == "HEADING":
            heading = current_maneuver.get("heading", {})
            x, y, z = heading.get("x", 0), heading.get("y", 0), heading.get("z", 0)
            lines.append(f"  Current maneuver: {maneuver_type} @ {throttle_pct}% throttle")
            lines.append(f"    Heading: ({x:+.1f}, {y:+.1f}, {z:+.1f})")
        elif maneuver_type == "EVASIVE" and evasion_status:
            mode = evasion_status.get("mode", "WOBBLE")
            lines.append(f"  Current maneuver: {maneuver_type} @ {throttle_pct}% throttle")
            lines.append(f"    Evasion mode: {mode}")
        else:
            lines.append(f"  Current maneuver: {maneuver_type} @ {throttle_pct}% throttle")
    else:
        lines.append("  Current maneuver: NONE (drifting)")

    # Always show threat assessment (helps LLM decide whether to evade)
    if evasion_status:
        threat_count = evasion_status.get("threat_count", 0)
        if threat_count > 0:
            mode = evasion_status.get("mode", "WOBBLE")
            lines.append(f"  Incoming threats: {threat_count} projectile(s) - evasion would use {mode} mode")

    # Primary target
    if primary_target_name:
        lines.append(f"  Primary target: {primary_target_name}")
    else:
        lines.append("  Primary target: None (will auto-select nearest)")

    # Radiators
    rad_status = "EXTENDED (cooling)" if radiators_extended else "RETRACTED (no cooling)"
    lines.append(f"  Radiators: {rad_status}")

    # Weapons - dynamic based on what's in weapon_orders
    for group_name, mode in weapon_orders.items():
        # Format group name nicely
        display_name = group_name.replace("_", " ").title()
        if group_name == "spinal":
            display_name = "Spinal coilgun"
        elif group_name == "heavy_coilguns":
            display_name = "Heavy coilguns"
        elif group_name == "coilguns":
            display_name = "Coilguns"
        elif group_name == "turret":
            display_name = "Turret coilgun"
        lines.append(f"  {display_name}: {mode}")

    # Fallback if no weapon orders
    if not weapon_orders:
        lines.append("  Weapons: HOLD_FIRE")

    return "\n".join(lines)


def build_personality_selection_prompt(
    distance_km: float,
    model_name: str = "AI",
    ship_class: str = "Destroyer",
    enemy_ship_class: str = "Destroyer",
) -> str:
    """Build the personality selection prompt for pre-battle phase."""
    # Build battle description
    if ship_class.lower() == enemy_ship_class.lower():
        battle_description = f"Duel against another {enemy_ship_class}"
    else:
        battle_description = f"Duel against a {enemy_ship_class}"

    return PERSONALITY_SELECTION_PROMPT.format(
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        distance_km=distance_km,
        model_name=model_name,
        ship_class=ship_class,
        battle_description=battle_description,
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
    recent_hits: Optional[str] = None,
    ship_type: Optional[str] = None,
    fleet_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the complete system prompt for a captain.

    Args:
        captain_name: Name of the captain
        ship_name: Name of the ship
        ship_status: Dict with hull_integrity, heat_percent, delta_v_remaining, armor values, etc.
        tactical_status: Dict with relative position/velocity, projectile info, enemies list, etc.
        decision_history: Formatted string of recent decisions
        message_history: Formatted string of message exchange history
        battle_summary: Formatted string summarizing battle progression
        shot_history: Formatted string of shot outcomes with range/velocity data
        recent_hits: Formatted string of recent damage taken
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
        ship_type=ship_type or "destroyer",
        fleet_data=fleet_data,
    )

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
        spinal_status = "(spinal needs <30°)"

    sim_time = tactical_status.get("sim_time", 0)

    # Get multi-ship data
    enemies = tactical_status.get("enemies", [])
    friendlies = tactical_status.get("friendlies", [])
    primary_target_id = tactical_status.get("primary_target_id")

    # Get primary target name for config display
    primary_target_name = None
    if primary_target_id and enemies:
        for e in enemies:
            if e.get("ship_id") == primary_target_id:
                primary_target_name = e.get("name", primary_target_id)
                break

    # Get current config
    current_config_data = tactical_status.get("current_config", {})
    weapon_orders = current_config_data.get("weapon_orders", {"spinal": "HOLD_FIRE", "turret": "HOLD_FIRE"})
    current_maneuver = current_config_data.get("current_maneuver")
    evasion_status = tactical_status.get("evasion_status")

    # Format current configuration
    current_config = format_current_config(
        primary_target_name=primary_target_name,
        radiators_extended=ship_status.get("radiators_extended", False),
        weapon_orders=weapon_orders,
        current_maneuver=current_maneuver,
        evasion_status=evasion_status,
    )

    # Format battlefield overview. fleet_data was accepted by this function but never
    # forwarded, so the enemy-capabilities block (accel / delta-v / turn time / armament)
    # was unreachable dead code and captains were told to judge "capital vs corvette"
    # with no class information at all.
    battlefield_overview = format_battlefield_overview(enemies, friendlies, fleet_data)

    # Format combat statistics
    our_shots = tactical_status.get("our_shots", 0)
    our_hits = tactical_status.get("our_hits", 0)
    our_damage_dealt = tactical_status.get("our_damage_dealt", 0)
    our_damage_taken = tactical_status.get("our_damage_taken", 0)

    combat_statistics = format_combat_statistics(
        our_shots=our_shots,
        our_hits=our_hits,
        our_damage_dealt=our_damage_dealt,
        our_damage_taken=our_damage_taken,
        enemies=enemies,
        friendlies=friendlies,
        ship_name=ship_name,
        primary_target_id=primary_target_id,
    )

    # Format incoming projectiles
    incoming_projectiles = format_incoming_projectiles(
        tactical_status.get("incoming_projectiles", [])
    )

    # Format received messages
    if received_messages:
        messages_section = f"ENEMY TRANSMISSION:\n{received_messages}"
    else:
        messages_section = ""

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

    # Recent hits section
    recent_hits_section = recent_hits if recent_hits else ""

    # Volatile ship status, now rendered separately from the static spec.
    _delta_v_total = 500
    if fleet_data and ship_type:
        _delta_v_total = (
            fleet_data.get("ships", {}).get(ship_type, {})
            .get("performance", {}).get("delta_v_kps", 500)
        )
    ship_status_block = build_ship_status_block(
        hull_integrity=ship_status.get("hull_integrity", 100),
        heat_percent=ship_status.get("heat_percent", 0),
        delta_v_remaining=ship_status.get("delta_v_remaining", 500),
        delta_v_total=_delta_v_total,
        nose_armor=ship_status.get("nose_armor", 0.0),
        lateral_armor=ship_status.get("lateral_armor", 0.0),
        tail_armor=ship_status.get("tail_armor", 0.0),
        radiators_extended=ship_status.get("radiators_extended", False),
        pd_turrets_operational=ship_status.get("pd_turrets_operational"),
        pd_turrets_total=ship_status.get("pd_turrets_total"),
        weapons=ship_status.get("weapons"),
        damaged_modules=ship_status.get("damaged_modules"),
    )

    return CAPTAIN_SYSTEM_PROMPT.format(
        torpedo_threats=format_torpedo_threats(tactical_status.get("torpedo_threats", [])),
        ship_status_block=ship_status_block,
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
        current_config=current_config,
        battlefield_overview=battlefield_overview,
        combat_statistics=combat_statistics,
        incoming_projectiles=incoming_projectiles,
        recent_hits=recent_hits_section,
        received_messages=messages_section,
        history_context=history_context,
        personality_prompt=personality_prompt,
    )


# =============================================================================
# ADMIRAL PROMPTS
# =============================================================================

# Marker separating the stable doctrine from the per-checkpoint fleet snapshot.
# Everything before it is reused verbatim across checkpoints of the same battle,
# so it can serve as a cacheable prefix; everything after it changes every call.
ADMIRAL_STATE_MARKER = "=== FLEET SITUATION ==="

# Stable across the battle -> cacheable prefix. Must precede the live snapshot.
ADMIRAL_DOCTRINE = """
You are {admiral_name}, Admiral of the {faction} fleet in a space combat simulation.

{mission_block}

{simulation_disclaimer}

=== YOUR ROLE ===

As Admiral, you:
1. Set the overall fleet directive (visible to all captains)
2. Issue SPECIFIC ORDERS to EACH captain using the issue_order tool
3. Can negotiate with enemy Admiral (if one exists)
4. Control draw proposals for your fleet

Your captains receive your orders BEFORE they decide.
They can discuss with you (up to 2 exchanges) before finalizing.

{order_requirements}
=== TACTICAL COMMAND REFERENCE ===

MANEUVERS (tell captains which to use):
- INTERCEPT: THRUST toward target - actively closes distance, builds velocity toward enemy
- EVADE: Evasive thrust while fighting - BEST during active combat!
- BRAKE: Slow down - use when closing too fast (>3 km/s relative)
- MAINTAIN: Coast at current velocity - no thrust, no rotation
- PADLOCK: ROTATE ONLY, NO THRUST - keeps nose on target but does NOT close distance!
- set_heading: Fly specific direction - for angled approaches, flanking

*** CRITICAL: PADLOCK vs INTERCEPT ***
PADLOCK @ 70% = Rotate to track, NO thrust toward target. Ship stays on current trajectory!
INTERCEPT @ 70% = Thrust TOWARD target at 70%. Ship actively closes distance!
To CLOSE RANGE: order INTERCEPT or set_heading with vector toward enemy. PADLOCK will NOT close!

=== FLEET TACTICAL OPTIONS ===

You can command your fleet however you want. Consider these strategies:

OPTION A - MASS CHARGE:
  All ships INTERCEPT same target → overwhelming force → high-speed pass
  Pros: Simple, focused firepower, intimidating
  Cons: Brief engagement window, ships scatter after pass

OPTION B - CONTROLLED FORMATION PASSES (recommended):
  1. Order INTERCEPT or set_heading to approach (must thrust to close distance!)
  2. Keep ships together for overlapping point defense coverage
  3. PADLOCK during passes - coast and track for sustained spinal fire
     (PADLOCK only rotates - ships coast on momentum, no thrust toward target)
  4. After pass: BRAKE, reform, set up another pass
  Pros: Better accuracy, mutual PD support, sustained damage
  Cons: Requires coordination, takes longer

OPTION C - HAMMER AND ANVIL:
  Heavy ships hold range (EVADE or MAINTAIN), lighter ships flank
  Pin enemy between groups, crossfire from multiple angles
  Pros: Divides enemy attention, creates flanking opportunities
  Cons: Complex coordination, risks defeat in detail if separated

OPTION D - FOCUS FIRE ROTATION:
  All ships focus one target until destroyed, then switch
  Concentrate damage to quickly reduce enemy numbers
  Pros: Rapid kills, psychological impact, reduces incoming fire
  Cons: Other enemies fire freely while ignored

KEY PHYSICS FOR FLEET COMMAND:
- High closing velocity = harder to hit (brief engagement window)
- 0 km/s relative + no evasion = sitting duck (easy hit)
- Sweet spot: 1-3 km/s relative with active maneuvering

=== TORPEDO WARFARE & POINT DEFENSE (measured on this engine - trust these) ===

HOW POINT DEFENSE ACTUALLY WORKS:
- All PD is automatic. Turrets triage savable torpedoes first (own ship, then allies),
  so overlapping coverage genuinely stacks: every friendly turret within its envelope
  (~{pd_range_km:.0f} km laser range) of a torpedo's path adds dwell against it.
- PD BLINDS torpedoes (seeker kill); the wreck coasts ballistically and STILL HITS a
  ship flying straight. Measured: 18/18 blinded rounds hit a non-evading battleship;
  the same attack on an evading one behind 8 escorts landed 6/18. Therefore: any ship
  under torpedo attack must be ordered to EVADE, or every PD kill is wasted.
- One turret blinds ~5 km/s of closure; N turrets ~5N km/s. Guided rounds arrive at
  13-28 km/s: single hulls (1-4 turrets) cannot stop a committed closing torpedo.
- PD cost is negligible (~15 MW hull heat per firing turret, no thrust loss). Never
  restructure the fleet to "save" PD capacity.

FORMATIONS FOR PD (what actually moves the numbers):
- TURRET COUNT on the threat corridor is everything; ring radius is not - escort
  rings at 15, 30 and 120 km measured identically. Place escorts anywhere inside
  laser range of the axis the torpedoes come down; don't micro-manage spacing.
- Returns need mass: vs a 6-round simultaneous salvo, an evading battleship leaked
  17/18 alone, 15/18 with 4 destroyer escorts, 6/18 with 8. Fewer than ~4 escorts is
  symbolic against a real salvo - concentrate the fleet or accept the hits.
- The wall only protects ships that EVADE inside it (see blinding, above).

WHEN TO COMMIT TRIDENTS:
- Fire when CLOSING. A closing launch vs a lone ship connects ~100% (12g guidance,
  <=3g targets): ~23 GJ per round at low closure, ~95 GJ launched head-on at 26 km/s
  (a spinal gun round is 4.3 GJ). NEVER launch at a receding evading ship - it
  outruns the round's delta-v and the magazine is wasted.
- SIMULTANEOUS time-on-target, always. Rounds arriving together split PD dwell;
  trickled rounds are blinded one at a time, early enough to dodge. Measured vs 4
  turrets: 4 simultaneous -> 24/24 through (99 GJ); same 4 spaced 30s -> 12/24 (42 GJ).
  Use torpedo_salvo on issue_order to synchronize launches ACROSS ships.
- SIZING at moderate closure (~22 GJ/round, evading targets): 4 rounds kill a
  destroyer; 6 rounds kill anything up to a dreadnought; 1-2 rounds wound only.
  Against an escort wall, add roughly 1 round per 3 escort turrets to feed the PD.
- RANGE: any closing launch 200-2400 km connects; longer = harder impact but minutes
  in flight (~400 km lands in 30-60s - about one decision cycle).
- Magazines: 8 rounds per launcher, 12s reload -> 2 per launcher per checkpoint.

FLANKING PREVENTION (YOUR RESPONSIBILITY):
Enemy ships in FLANKING positions can hit your ships on the LATERAL (side) - easy targets!
- Cross-section: Nose-on = tiny target, Lateral = huge target (5-10x easier to hit)
- Your job: Position the fleet so enemies are always in FRONT of your ships
- Watch for enemy flanking maneuvers (ships trying to get perpendicular angles)
- If an enemy is flanking: order affected ships to reorient (INTERCEPT or PADLOCK toward them)
- Multi-ship coordination: If enemy splits, assign ships to cover each threat vector
Preventing flanking is tactical positioning - captains execute, YOU direct!

FORMATION MANAGEMENT (CRITICAL!):
Your ships have DIFFERENT max accelerations (see SHIP CAPABILITIES below).

COMMON MISTAKE: Ordering all ships at "70% throttle" does NOT maintain formation!
  - Destroyer (2.0g) at 70% = 1.4g effective acceleration
  - Dreadnought (0.75g) at 70% = 0.525g effective acceleration
  - They're STILL spreading! Same throttle ≠ same acceleration!

To ACTUALLY maintain formation:
  - Find your SLOWEST ship's max-g (e.g., dreadnought at 0.75g)
  - Order that ship at 100% throttle
  - Order faster ships at (slowest_g / their_g) throttle
  - Example: Destroyer matching dreadnought = (0.75/2.0) = 37% throttle

Quick math: throttle_for_fast_ship = (slow_ship_g / fast_ship_g) × 100%

If you DON'T care about formation (valid strategies):
  - Faster ships engage first as vanguard, heavies follow
  - Let destroyers screen while capitals close at their own pace
  - Formation spread is fine if you plan for staggered engagement

TARGET PRIORITY:
- Capital ships (dreadnoughts, battleships) turn slowly → easier hits
- Destroyers/frigates are nimble but fragile → can be killed quickly
- CLOSING targets = priority (committed), SEPARATING = low priority (fleeing)
- Damaged enemies = opportunity, but don't chase if they're running

THROTTLE LEVELS:
- 100%: Initial approach or emergency only
- 50-70%: Good for sustained maneuvering
- 10-30%: Fine positioning, formation keeping

RESOURCES: Ships have ~500 km/s delta-V and 450+ rounds. Fuel and ammo are ABUNDANT.
Don't conserve - aggressive maneuvering is always sustainable.

WEAPON MODES:
- FIRE_IMMEDIATE: Fire as fast as possible (close range)
- FIRE_WHEN_OPTIMAL: Wait for good alignment (recommended)
- FIRE_AT_RANGE: Maximum range harassing fire
- HOLD_FIRE: Stealth approach, heat management

SHIP CAPABILITIES:
{ship_class_stats}
- Spinal weapons need nose-on alignment (<30° to target)
- Turrets can fire at 180° arc (more flexible)
- Radiators EXTENDED = better cooling (no damage model penalises extended radiators)
- Don't waste shots on fleeing enemies (5+ km/s separation) - focus on threats closing on you
- A target at 100km closing at 3 km/s is MORE dangerous than one at 80km separating at 4 km/s

The fleet_directive is for OVERALL strategy only - each captain also needs their own order.

ADDITIONAL NOTES:
- Captains are AI commanders - be specific about priorities and targets
- Use the dual snapshot to understand battle momentum
- Coordinate your ships for focus fire or tactical positioning
- TIMING: Each order is LOCKED for 30 seconds. No "if X then Y" contingencies.
  Give ONE clear order per ship. Captains cannot react mid-checkpoint.
{enemy_admiral_note}

{personality_prompt}
"""

# Rebuilt every checkpoint. Must come AFTER the doctrine in the request so the
# doctrine can be a stable cached prefix (it previously sat *after* the snapshot,
# which made any prefix boundary useless).
ADMIRAL_TURN_STATE = (
    ADMIRAL_STATE_MARKER
    + """

=== YOUR FLEET ===
{fleet_composition}

=== FLEET CAPABILITIES ===
{fleet_capabilities}

=== DUAL TEMPORAL SNAPSHOT ===
You see the battle at two points in time to analyze trajectories and momentum.

--- T-15s (15 seconds ago) ---
{snapshot_t_minus_15}

--- T=0 (Current) ---
{snapshot_t_zero}

=== CHANGE ANALYSIS ===
{change_analysis}

=== FRIENDLY FLEET STATUS (FULL) ===
{friendly_fleet_status}

=== ENEMY FLEET STATUS (OBSERVABLE) ===
{enemy_fleet_status}

=== PROJECTILES IN FLIGHT ===
{projectile_info}

{communications_section}
"""
)

# Backwards-compatible single-string form (doctrine + live state).
ADMIRAL_SYSTEM_PROMPT = ADMIRAL_DOCTRINE + "\n" + ADMIRAL_TURN_STATE

ADMIRAL_RESPONSE_PROMPT = """
You are {admiral_name}, responding to a question from one of your ship captains.

The captain of {captain_ship_name} is asking you a question during battle.
Be concise and decisive - there's no time for lengthy discussion.

IMPORTANT: This is a ONE-WAY response. The captain will act immediately after receiving
your answer. DO NOT ask follow-up questions - they cannot respond. Give clear, actionable
orders or clarifications. End with a command, not a question.

TIMING CONSTRAINT: The captain makes ONE decision that is LOCKED for 30 seconds.
They CANNOT react mid-checkpoint or execute conditional orders like "if X then Y".
Give them ONE clear order to execute NOW, not contingency plans.

=== YOUR RECENT ORDERS (REMEMBER THESE!) ===
{recent_orders_context}

{personality_prompt}
"""


def _generate_ship_class_stats(snapshot: Any, fleet_data: Dict[str, Any]) -> str:
    """Generate dynamic ship class stats based on ships in the battle."""
    if not snapshot or not fleet_data:
        return "- Check FLEET CAPABILITIES section for ship stats"

    # Collect unique ship types from both friendly and enemy ships
    ship_types = set()
    if snapshot.friendly_ships:
        for ship in snapshot.friendly_ships:
            ship_types.add(ship.ship_type)
    if snapshot.enemy_ships:
        for ship in snapshot.enemy_ships:
            ship_types.add(ship.ship_type)

    if not ship_types:
        return "- Check FLEET CAPABILITIES section for ship stats"

    lines = []
    ships_spec = fleet_data.get("ships", {})

    for ship_type in sorted(ship_types):
        spec = ships_spec.get(ship_type, {})
        if not spec:
            continue

        performance = spec.get("performance", {})
        delta_v = performance.get("delta_v_kps", "?")
        accel = performance.get("combat_acceleration_g", "?")

        # Get weapon summary. PD turret and torpedo-launcher counts are the
        # numbers salvo sizing and formation math run on, so they must be here.
        weapons = spec.get("weapons", [])
        spinal_count = 0
        turret_count = 0
        pd_count = 0
        torpedo_count = 0
        for w in weapons:
            w_type = w.get("type", "")
            if w_type == "pd_laser":
                pd_count += 1
            elif w_type.startswith("torpedo"):
                torpedo_count += 1
            elif "spinal" in w_type.lower():
                spinal_count += 1
            elif "turret" in w.get("mount", "").lower() or w.get("is_turreted", False):
                turret_count += 1

        parts = []
        if spinal_count:
            parts.append(f"{spinal_count}x spinal")
        if turret_count:
            parts.append(f"{turret_count}x turret")
        if torpedo_count:
            parts.append(f"{torpedo_count}x torpedo launcher")
        parts.append(f"{pd_count}x PD")
        weapons_str = ", ".join(parts)

        # Estimate maneuverability from acceleration (rough guide)
        if accel != "?":
            if accel >= 2.5:
                maneuver = "agile"
            elif accel >= 1.5:
                maneuver = "moderate"
            elif accel >= 1.0:
                maneuver = "slow"
            else:
                maneuver = "very slow"
        else:
            maneuver = "unknown"

        lines.append(f"- {ship_type.title()}: {accel}g ({maneuver}), {delta_v} km/s delta-v, {weapons_str}")

    return "\n".join(lines) if lines else "- Check FLEET CAPABILITIES section for ship stats"


def build_admiral_prompt(
    admiral_name: str,
    faction: str,
    snapshot_t_minus_15: Any,
    snapshot_t_zero: Any,
    personality: Optional[str],
    fleet_data: Dict[str, Any],
    enemy_has_admiral: bool = False,
    enemy_proposed_draw: bool = False,
    received_messages: Optional[List[str]] = None,
    communications_log: Optional[List[Any]] = None,
    phase: str = "full",  # "full", "directive", or "orders"
) -> str:
    """
    Build Admiral system prompt with dual-snapshot comparison.

    Args:
        admiral_name: Admiral's name
        faction: "alpha" or "beta"
        snapshot_t_minus_15: Battle snapshot from 15s ago
        snapshot_t_zero: Current battle snapshot
        personality: Admiral personality description
        fleet_data: Ship specifications
        enemy_has_admiral: Whether enemy has an Admiral
        enemy_proposed_draw: Whether enemy has proposed draw
        received_messages: Messages from enemy Admiral
        communications_log: All captain communications (Admiral oversight)
        phase: Which call this prompt is for. "directive" asks for the fleet
            directive only (per-ship orders are collected in a later call and any
            issue_order made now is discarded, so the checklist is suppressed);
            "full"/"orders" keep the issue-an-order-for-every-ship requirement.

    Returns:
        Complete Admiral system prompt
    """
    # Format fleet composition
    if snapshot_t_zero:
        fleet_composition = snapshot_t_zero.fleet_summary
    else:
        fleet_composition = "Fleet status unknown"

    # Format fleet capabilities
    fleet_capabilities = _format_fleet_capabilities(
        snapshot_t_zero.friendly_ships if snapshot_t_zero else [],
        fleet_data,
    )

    # Format snapshots
    snapshot_15_text = _format_admiral_snapshot(snapshot_t_minus_15, "T-15s")
    snapshot_0_text = _format_admiral_snapshot(snapshot_t_zero, "T=0")

    # Analyze changes
    change_analysis = _analyze_snapshot_changes(snapshot_t_minus_15, snapshot_t_zero)

    # Format friendly fleet (full info)
    friendly_status = _format_friendly_fleet_full(
        snapshot_t_zero.friendly_ships if snapshot_t_zero else []
    )

    # Format enemy fleet (observable only)
    enemy_status = _format_enemy_fleet_observable(
        snapshot_t_zero.enemy_ships if snapshot_t_zero else [],
        fleet_data,
    )

    # Format projectiles
    projectile_info = _format_projectiles_for_admiral(
        snapshot_t_zero.projectiles if snapshot_t_zero else []
    )

    # Communications section
    comms_parts = []
    if received_messages:
        comms_parts.append("=== MESSAGES FROM ENEMY ADMIRAL ===")
        for msg in received_messages:
            comms_parts.append(f"  \"{msg}\"")

    if enemy_proposed_draw:
        comms_parts.append("\n*** ENEMY HAS PROPOSED A DRAW ***")
        comms_parts.append("Use accept_fleet_draw to accept or reject_fleet_draw to refuse.")

    if communications_log:
        comms_parts.append("\n=== CAPTAIN COMMUNICATIONS (You see all) ===")
        for msg in communications_log[-5:]:  # Last 5 messages
            comms_parts.append(f"  [{msg.ship_name}] {msg.sender_name}: \"{msg.content}\"")

    communications_section = "\n".join(comms_parts) if comms_parts else ""

    # Enemy Admiral note
    if enemy_has_admiral:
        enemy_admiral_note = "- Enemy has an Admiral. Use message_enemy_admiral to negotiate."
    else:
        enemy_admiral_note = "- Enemy has no Admiral. Their captains act independently."

    # Personality
    personality_prompt = f"YOUR PERSONALITY:\n{personality}" if personality else ""

    # Count ships for the critical instruction
    num_ships = len(snapshot_t_zero.friendly_ships) if snapshot_t_zero else 0

    # Generate ship order checklist - explicit list of ships needing orders
    ship_order_checklist = ""
    # The caller may only be asking for the fleet directive on this call (phase
    # "directive"); admiral._issue_ship_order() collects per-ship orders afterwards
    # and the directive-phase result parser DISCARDS any issue_order calls. Demanding
    # issue_order here therefore produced work that was silently thrown away.
    directive_only = (phase == "directive")

    if snapshot_t_zero and snapshot_t_zero.friendly_ships:
        checklist_lines = []
        for i, ship in enumerate(snapshot_t_zero.friendly_ships, 1):
            suffix = "" if directive_only else " <- MUST issue_order for this ship!"
            checklist_lines.append(f"  {i}. {ship.ship_name}{suffix}")
        ship_order_checklist = "\n".join(checklist_lines)
    else:
        ship_order_checklist = "  (no ships)"

    if directive_only:
        mission_block = f"""***** THIS CALL: FLEET DIRECTIVE ONLY *****

Call set_fleet_directive with the overall strategy for your {num_ships} ships:
{ship_order_checklist}

Individual ship orders are requested in a SEPARATE call immediately after this one,
one ship at a time. Do NOT call issue_order now - orders issued on this call are
discarded. Make the directive concrete enough that the per-ship orders follow from it.

Ships are shown as: NAME [ID] (e.g., "OCS Gemini-1 [beta_1]")
You can use EITHER the name OR the ID when naming ships or targets.

*************************************************"""
        order_requirements = ""
    else:
        mission_block = f"""***** CRITICAL REQUIREMENT - READ THIS FIRST *****

You MUST call issue_order for EACH of your {num_ships} ships:
{ship_order_checklist}

^^^ YOU MUST ISSUE AN ORDER TO EVERY SHIP LISTED ABOVE ^^^
If you skip a ship, it will DRIFT with no orders and be useless!

Ships are shown as: NAME [ID] (e.g., "OCS Gemini-1 [beta_1]")
You can use EITHER the name OR the ID for targeting.

EXAMPLE:
  issue_order(ship_name="TIS Haiku-1", order_text="Target OCS Gemini-1. INTERCEPT.", suggested_target="OCS Gemini-1")
  OR: order_text="Target beta_1. INTERCEPT." - both work!

*************************************************"""
        order_requirements = """***** CRITICAL - YOU MUST ISSUE ORDERS TO EACH SHIP *****

For EACH ship in your fleet, you MUST call issue_order with:
- ship_name: The ship's name (e.g., "TIS Haiku-1")
- order_text: SPECIFIC instructions including:
  * Which enemy to target (by name or ID)
  * What maneuver to use (INTERCEPT, EVADE, BRAKE, etc.)
  * Weapon orders (fire immediately, hold fire, etc.)
  * Any special instructions
- priority: CRITICAL, HIGH, NORMAL, or LOW
- suggested_target: The enemy ship to focus on

EXAMPLE ORDER:
"Target OCS Grok-1. EVADE while engaging - dodge their fire but keep weapons hot.
Fire spinal when aligned, turrets continuous fire. Stay mobile - don't be a sitting duck."

"""

    # Generate dynamic ship class stats from fleet_data
    ship_class_stats = _generate_ship_class_stats(snapshot_t_zero, fleet_data)

    # Allied PD coverage is bounded by the laser's own range (there is no separate
    # "formation radius" constant anywhere in the PD code), so read it from the same
    # data PDLaser.from_fleet_data reads instead of quoting a made-up 50 km.
    pd_range_km = float(
        (fleet_data or {}).get("weapon_types", {}).get("pd_laser", {}).get("range_km", 250)
    )

    return ADMIRAL_SYSTEM_PROMPT.format(
        pd_range_km=pd_range_km,
        admiral_name=admiral_name,
        faction=faction.upper(),
        mission_block=mission_block,
        order_requirements=order_requirements,
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        fleet_composition=fleet_composition,
        fleet_capabilities=fleet_capabilities,
        snapshot_t_minus_15=snapshot_15_text,
        snapshot_t_zero=snapshot_0_text,
        change_analysis=change_analysis,
        friendly_fleet_status=friendly_status,
        enemy_fleet_status=enemy_status,
        projectile_info=projectile_info,
        communications_section=communications_section,
        enemy_admiral_note=enemy_admiral_note,
        personality_prompt=personality_prompt,
        ship_class_stats=ship_class_stats,
    )


def build_admiral_response_prompt(
    admiral_name: str,
    captain_ship_name: str,
    question: str,
    personality: Optional[str],
    recent_decisions: Optional[List[Any]] = None,
) -> str:
    """Build prompt for Admiral responding to captain discussion."""
    # Format recent decisions (up to last 3 checkpoints)
    orders_lines = []

    if recent_decisions:
        for i, decision in enumerate(reversed(recent_decisions)):
            checkpoint_label = "CURRENT" if i == 0 else f"{i} checkpoint(s) ago"
            orders_lines.append(f"--- {checkpoint_label} ---")

            if decision.fleet_directive:
                orders_lines.append(f"FLEET DIRECTIVE: {decision.fleet_directive}")

            if decision.fleet_orders:
                for order in decision.fleet_orders:
                    marker = " <-- THIS CAPTAIN" if order.target_ship_name == captain_ship_name else ""
                    orders_lines.append(f"  -> {order.target_ship_name}: [{order.priority}] {order.order_text}{marker}")
            else:
                orders_lines.append("  (No individual orders issued)")

            orders_lines.append("")

        recent_orders_context = "\n".join(orders_lines)
    else:
        recent_orders_context = "(No orders have been issued yet this battle)"

    personality_prompt = f"YOUR PERSONALITY:\n{personality}" if personality else ""

    return ADMIRAL_RESPONSE_PROMPT.format(
        admiral_name=admiral_name,
        captain_ship_name=captain_ship_name,
        recent_orders_context=recent_orders_context,
        personality_prompt=personality_prompt,
    )


def build_admiral_ship_order_prompt(
    admiral_name: str,
    ship_name: str,
    ship_type: str,
    captain_name: str,
    fleet_directive: str,
    snapshot: Any,
    personality: Optional[str] = None,
) -> str:
    """
    Build prompt for Admiral to issue order to a specific ship.

    Args:
        admiral_name: Admiral's name
        ship_name: Ship to order (e.g., "TIS Haiku-1")
        ship_type: Ship class (e.g., "destroyer")
        captain_name: Captain's name
        fleet_directive: The fleet-wide strategy already set
        snapshot: Current battle snapshot
        personality: Admiral personality

    Returns:
        Focused prompt for issuing one ship's order
    """
    # Get ship-specific info from snapshot
    ship_info = ""
    if snapshot and snapshot.friendly_ships:
        for ship in snapshot.friendly_ships:
            if ship.ship_name == ship_name:
                pos = ship.position_km
                ship_info = f"""
SHIP STATUS - {ship_name}:
  Type: {ship_type.title()}
  Captain: {captain_name}
  Position: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km
  Velocity: {ship.velocity_kps:.1f} km/s
  Hull: {ship.hull_integrity:.0f}%
  Heat: {ship.heat_pct:.0f}%
  Delta-V: {ship.delta_v_remaining_kps:.1f} km/s remaining
  Weapons: {ship.weapons_summary}"""
                break

    # Get enemy info
    enemy_summary = ""
    if snapshot and snapshot.enemy_ships:
        enemy_lines = ["ENEMY POSITIONS:"]
        for enemy in snapshot.enemy_ships:
            pos = enemy.position_km
            enemy_lines.append(
                f"  {enemy.ship_name} ({enemy.ship_type}): "
                f"({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km, "
                f"vel {enemy.velocity_kps:.1f} km/s, dist {enemy.distance_from_closest_friendly_km:.0f} km"
            )
        enemy_summary = "\n".join(enemy_lines)

    personality_line = f"\nYOUR PERSONALITY: {personality}" if personality else ""

    return f"""You are {admiral_name}, commanding a fleet in space combat.

You have already set the fleet directive:
"{fleet_directive}"

Now you must issue a specific tactical order to {ship_name}.
{ship_info}

{enemy_summary}
{personality_line}

You MUST call the issue_order tool with:
- ship_name: "{ship_name}"
- order_text: Specific tactical instructions for this ship
- priority: CRITICAL, HIGH, NORMAL, or LOW
- suggested_target: (optional) Recommended enemy ship name

Your order should:
1. Be consistent with your fleet directive
2. Specify a maneuver (INTERCEPT, PADLOCK, EVASIVE, HEADING)
3. Specify throttle level (e.g., "70% throttle")
4. Specify target assignment
5. Include any special weapon instructions

*** CRITICAL: PADLOCK vs INTERCEPT ***
- PADLOCK = ROTATE ONLY, NO THRUST. Ship keeps nose on target but does NOT close distance!
- INTERCEPT = THRUST toward target. Ship actively closes distance!
To CLOSE RANGE: order INTERCEPT or set_heading with vector toward enemy. PADLOCK will NOT close!

CALL issue_order NOW for {ship_name}."""


def _format_fleet_capabilities(
    friendly_ships: List[Any],
    fleet_data: Dict[str, Any],
) -> str:
    """Format full capabilities of each ship in the fleet."""
    lines = []

    for ship in friendly_ships:
        ship_spec = fleet_data.get("ships", {}).get(ship.ship_type, {})
        performance = ship_spec.get("performance", {})

        lines.append(f"\n{ship.ship_name} ({ship.ship_type.title()} class):")
        lines.append(f"  Captain: {ship.captain_name}")
        lines.append(f"  Acceleration: {performance.get('combat_acceleration_g', 2.0)}g")
        lines.append(f"  Delta-V Budget: {performance.get('delta_v_kps', 500)} km/s")
        lines.append(f"  90° Turn Time: ~{get_ship_turn_time_90deg(ship.ship_type):.0f}s")
        lines.append(f"  Weapons: {ship.weapons_summary}")

    return "\n".join(lines) if lines else "No ships"


def _format_admiral_snapshot(snapshot: Any, label: str) -> str:
    """Format a snapshot for Admiral view."""
    if not snapshot:
        return f"[{label}] No data available"

    lines = [f"Time: T+{snapshot.timestamp:.0f}s"]

    # Friendly ships
    for ship in snapshot.friendly_ships:
        pos = ship.position_km
        lines.append(
            f"  {ship.ship_name}: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km, "
            f"vel {ship.velocity_kps:.1f} km/s, hull {ship.hull_integrity:.0f}%"
        )

    # Enemy ships
    for enemy in snapshot.enemy_ships:
        pos = enemy.position_km
        lines.append(
            f"  [ENEMY] {enemy.ship_name}: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km, "
            f"vel {enemy.velocity_kps:.1f} km/s"
        )

    return "\n".join(lines)


def _analyze_snapshot_changes(
    snapshot_15: Any,
    snapshot_0: Any,
) -> str:
    """Analyze changes between two snapshots."""
    if not snapshot_15 or not snapshot_0:
        return "Insufficient data for analysis"

    lines = []

    # Compare distances
    if snapshot_0.enemy_ships and snapshot_15.enemy_ships:
        current_dist = snapshot_0.enemy_ships[0].distance_from_closest_friendly_km
        prev_dist = snapshot_15.enemy_ships[0].distance_from_closest_friendly_km
        delta = current_dist - prev_dist

        if delta < -10:
            lines.append(f"CLOSING: {abs(delta):.0f} km closer over 15s (rate: {abs(delta)/15:.1f} km/s)")
        elif delta > 10:
            lines.append(f"SEPARATING: {delta:.0f} km farther over 15s (rate: {delta/15:.1f} km/s)")
        else:
            lines.append("STABLE: Engagement distance roughly constant")

    # Compare hull integrity
    for ship_0 in snapshot_0.friendly_ships:
        for ship_15 in snapshot_15.friendly_ships:
            if ship_0.ship_id == ship_15.ship_id:
                hull_delta = ship_0.hull_integrity - ship_15.hull_integrity
                if hull_delta < -1:
                    lines.append(f"DAMAGE: {ship_0.ship_name} lost {abs(hull_delta):.0f}% hull")

    # Closing rate
    if snapshot_0.enemy_ships:
        closing = snapshot_0.enemy_ships[0].closing_rate_kps
        if closing > 1:
            lines.append(f"INTERCEPT: Closing at {closing:.1f} km/s")
        elif closing < -1:
            lines.append(f"PURSUIT: Enemy fleeing at {abs(closing):.1f} km/s")

    return "\n".join(lines) if lines else "No significant tactical changes"


def _format_friendly_fleet_full(friendly_ships: List[Any]) -> str:
    """Format full info for all friendly ships."""
    if not friendly_ships:
        return "No friendly ships"

    lines = []
    for ship in friendly_ships:
        pos = ship.position_km
        lines.append(f"\n{ship.ship_name} ({ship.ship_type.title()}):")
        lines.append(f"  Captain: {ship.captain_name}")
        lines.append(f"  Position: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km")
        lines.append(f"  Velocity: {ship.velocity_kps:.1f} km/s")
        lines.append(f"  Hull: {ship.hull_integrity:.0f}%")
        lines.append(f"  Delta-V: {ship.delta_v_remaining:.0f} km/s remaining")
        lines.append(f"  Heat: {ship.heat_percent:.0f}%")
        lines.append(f"  Radiators: {'EXTENDED' if ship.radiators_extended else 'RETRACTED'}")
        lines.append(f"  Maneuver: {ship.current_maneuver}")
        lines.append(f"  Target: {ship.current_target or 'None'}")

        if ship.weapons_ready:
            lines.append(f"  Weapons Ready: {', '.join(ship.weapons_ready)}")
        if ship.weapons_cooling:
            lines.append(f"  Weapons Cooling: {', '.join(ship.weapons_cooling)}")
        if hasattr(ship, 'weapons_destroyed') and ship.weapons_destroyed:
            lines.append(f"  *** WEAPONS DESTROYED: {', '.join(ship.weapons_destroyed)} ***")

        # Torpedo magazine and PD state: the two numbers the admiral's salvo
        # sizing and formation decisions actually run on.
        torps = getattr(ship, 'torpedoes_remaining', None)
        if torps:
            lines.append(f"  Torpedoes: {torps} remaining")
        pd_total = getattr(ship, 'pd_turrets_total', None)
        if pd_total:
            pd_op = getattr(ship, 'pd_turrets_operational', pd_total)
            degraded = " - DEGRADED" if pd_op < pd_total else ""
            lines.append(f"  Point defense: {pd_op}/{pd_total} turrets{degraded}")

        if ship.targeted_by:
            lines.append(f"  *** TARGETED BY: {', '.join(ship.targeted_by)} ***")

    return "\n".join(lines)


def _format_enemy_fleet_observable(
    enemy_ships: List[Any],
    fleet_data: Dict[str, Any],
) -> str:
    """Format observable info for enemy ships (Admiral can't see internals)."""
    if not enemy_ships:
        return "No enemy ships detected"

    lines = []
    for enemy in enemy_ships:
        pos = enemy.position_km

        # Get enemy ship capabilities from fleet data
        enemy_spec = fleet_data.get("ships", {}).get(enemy.ship_type, {})
        performance = enemy_spec.get("performance", {})
        weapons = enemy_spec.get("weapons", [])

        # Show both name and ID for targeting clarity
        ship_id = getattr(enemy, 'ship_id', '')
        id_display = f" [{ship_id}]" if ship_id else ""
        lines.append(f"\n{enemy.ship_name}{id_display} ({enemy.ship_type.title()}):")
        lines.append(f"  Position: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km")
        lines.append(f"  Velocity: {enemy.velocity_kps:.1f} km/s")
        lines.append(f"  Distance: {enemy.distance_from_closest_friendly_km:.0f} km")

        if enemy.closing_rate_kps > 0:
            lines.append(f"  Closing: {enemy.closing_rate_kps:.1f} km/s")
        elif enemy.closing_rate_kps < 0:
            lines.append(f"  Separating: {abs(enemy.closing_rate_kps):.1f} km/s")

        # Show known capabilities (from ship class, not current state)
        if performance:
            lines.append(f"  Known Capabilities:")
            lines.append(f"    Accel: {performance.get('combat_acceleration_g', '?')}g")
            lines.append(f"    Delta-V: {performance.get('delta_v_kps', '?')} km/s")
            lines.append(f"    Weapons: {format_enemy_weapons_summary(weapons)}")

    return "\n".join(lines)


def _format_projectiles_for_admiral(projectiles: List[Any]) -> str:
    """Format ordnance in flight for Admiral. Torpedoes are listed first."""
    if not projectiles:
        return "No ordnance in flight"

    def is_torp(p):
        return "torpedo" in str(getattr(p, "weapon_type", "")).lower()

    ordered = sorted(projectiles, key=lambda p: (not is_torp(p), p.eta_seconds))
    lines = []
    for proj in ordered[:10]:  # Limit to 10 most relevant
        lines.append(
            f"  {proj.weapon_type} from {proj.source_ship} -> {proj.target_ship}: "
            f"{proj.distance_km:.0f} km, ETA {proj.eta_seconds:.1f}s, {proj.damage_gj:.1f} GJ"
        )
    if any(is_torp(p) for p in ordered):
        lines.append(
            "  Torpedoes inbound on your ships: order those ships to EVADE - "
            "PD blinds torpedoes but only a maneuvering ship dodges the wreck."
        )
    return "\n".join(lines)


def format_admiral_orders_for_captain(
    orders: List[Any],
    fleet_directive: str,
) -> str:
    """Format Admiral orders for inclusion in captain prompt."""
    lines = []
    lines.append("=" * 60)
    lines.append("       ***** ORDERS FROM YOUR ADMIRAL *****")
    lines.append("=" * 60)

    if fleet_directive:
        lines.append(f"\nFLEET DIRECTIVE: {fleet_directive}")

    if orders:
        lines.append("\n>>> YOUR SPECIFIC ORDERS <<<")
        for order in orders:
            priority_marker = "!!!" if order.priority == "CRITICAL" else (
                "!!" if order.priority == "HIGH" else ""
            )
            lines.append(f"\n  {priority_marker}[{order.priority}] {order.order_text}")
            if order.suggested_target:
                lines.append(f"      >>> TARGET: {order.suggested_target}")
            # Coordinated salvo: several ships launching together is how a fleet
            # overwhelms point defense, so state it explicitly rather than
            # leaving it buried in the order prose.
            salvo = getattr(order, "torpedo_salvo", None)
            if salvo:
                tgt = (getattr(order, "torpedo_target", None)
                       or order.suggested_target)
                at = f" at {tgt}" if tgt else ""
                lines.append(
                    f"      >>> COORDINATED SALVO: launch {salvo} torpedo(es){at} "
                    f"THIS CHECKPOINT, timed with the rest of the fleet. Do not hold."
                )
    else:
        lines.append("\n!!! WARNING: NO ORDERS RECEIVED FOR YOUR SHIP !!!")
        lines.append("Your Admiral may have forgotten to issue you orders.")
        lines.append("You SHOULD use discuss_with_admiral to ask:")
        lines.append('  "Admiral, I did not receive any orders. What are my instructions?"')
        lines.append("If the Admiral doesn't respond, use your best judgment.")

    lines.append("\n" + "=" * 60)
    lines.append("")
    lines.append("*** YOU MUST RESPOND TO THESE ORDERS ***")
    lines.append("")
    lines.append("Use respond_to_orders tool to either:")
    lines.append("  1. ACKNOWLEDGE - You will follow the orders as given")
    lines.append("  2. DEVIATE - You have tactical reasons to do something else")
    lines.append("     (You MUST explain why you are deviating)")
    lines.append("")
    lines.append("If you deviate without good reason, you are failing your Admiral.")
    lines.append("CRITICAL and HIGH priority orders should almost always be followed.")
    lines.append("")
    lines.append("You may use discuss_with_admiral to ask questions (2 exchanges max).")
    lines.append("")
    lines.append("!!! CRITICAL - DO NOT SKIP THIS !!!")
    lines.append("")
    lines.append("respond_to_orders = WORDS ONLY (tells Admiral your intent)")
    lines.append("set_maneuver = ACTUAL MOVEMENT (makes ship move)")
    lines.append("")
    lines.append("YOU MUST CALL BOTH:")
    lines.append("  1. respond_to_orders (acknowledge/deviate)")
    lines.append("  2. set_maneuver (EVADE/INTERCEPT/etc)")
    lines.append("  3. set_primary_target (who to aim at)")
    lines.append("  4. set_weapons_order (how to fire)")
    lines.append("")
    lines.append("If you ONLY call respond_to_orders -> your ship DRIFTS and DIES")
    lines.append("Saying 'I will evade' is NOT the same as calling set_maneuver!")
    lines.append("")
    lines.append("NOTE: Only your Admiral can propose draws.")
    lines.append("=" * 60)

    return "\n".join(lines)


def build_captain_messages(**kwargs) -> List[Dict[str, str]]:
    """
    Build the captain request as a cacheable system prompt plus a volatile user turn.

    The previous single-string prompt interleaved per-checkpoint state (sim time,
    positions, damage) with ~1200 tokens of static doctrine that followed it, so
    the provider could never reuse a cached prefix - every 30 s checkpoint re-paid
    full input price on the entire prompt. CAPTAIN_SYSTEM_PROMPT is now ordered
    doctrine-first, so a single split at the tactical-data marker separates the
    stable prefix from the volatile turn.

    Accepts the same keyword arguments as build_captain_prompt().
    """
    full = build_captain_prompt(**kwargs)
    marker = "=== TACTICAL DATA (T+"
    idx = full.find(marker)
    if idx == -1:  # structure changed - fall back to a single message
        return [{"role": "system", "content": full}]

    return [
        {"role": "system", "content": full[:idx].rstrip()},
        {"role": "user", "content": full[idx:].rstrip()},
    ]


def build_admiral_messages(**kwargs) -> List[Dict[str, str]]:
    """
    Build the admiral request as a cacheable system prompt plus a volatile user turn.

    Same problem and same fix as build_captain_messages(): the doctrine (role,
    tactical command reference, fleet tactical options, ship-class stats) is
    stable for the whole battle, so it now precedes the dual-snapshot fleet
    situation and can be split off as a reusable prefix.

    Accepts the same keyword arguments as build_admiral_prompt().
    """
    full = build_admiral_prompt(**kwargs)
    idx = full.find(ADMIRAL_STATE_MARKER)
    if idx == -1:  # structure changed - fall back to a single message
        return [{"role": "system", "content": full}]

    return [
        {"role": "system", "content": full[:idx].rstrip()},
        {"role": "user", "content": full[idx:].rstrip()},
    ]
