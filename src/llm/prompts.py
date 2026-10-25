"""
System prompts for LLM captains.

Prompts include simulation disclaimer to avoid AI guardrails,
ship capabilities, and personality modifiers.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple
from enum import Enum

import math

try:
    from ..pointdefense import (
        PDLaser,
        PD_ABSORPTIVITY,
        TORPEDO_ELECTRONICS_THRESHOLD_J,
        full_coupling_range_m,
    )
    from ..torpedo import MIN_CLOSING_SPEED_KPS
    from .. import thermal as _thermal
    from .. import combat as _combat
    from ..power import KINETIC_WEAPON_EFFICIENCY
except ImportError:  # pragma: no cover - script-style imports
    from pointdefense import (
        PDLaser,
        PD_ABSORPTIVITY,
        TORPEDO_ELECTRONICS_THRESHOLD_J,
        full_coupling_range_m,
    )
    from torpedo import MIN_CLOSING_SPEED_KPS
    import thermal as _thermal
    import combat as _combat
    from power import KINETIC_WEAPON_EFFICIENCY


# Closing speeds tabulated for the captain's PD block. The middle entry is the
# guidance closure floor, which is what a normally launched round actually
# arrives at; the others bracket it.
PD_DWELL_TABLE_KPS = (4.0, 8.0, MIN_CLOSING_SPEED_KPS, 20.0)


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
    # What ONE turret delivers against a round crossing the whole envelope at a
    # given closing speed, and how many turrets that implies for a seeker kill.
    # This is the same closed form, re-expressed in the units a captain can
    # actually read off their threat list (MJ and turret counts) instead of a
    # "km/s of closure" figure that invites comparison against the wrong speed.
    dwell_table = []
    for v_kps in PD_DWELL_TABLE_KPS:
        mj = laser.energy_before_impact_j(laser.range_km, v_kps * 1000.0) / 1e6
        dwell_table.append({
            "closing_kps": v_kps,
            "mj_per_turret": mj,
            "turrets_to_blind": max(
                1, math.ceil(TORPEDO_ELECTRONICS_THRESHOLD_J / 1e6 / mj)
            ) if mj > 0 else 99,
        })
    at_floor = next(
        r for r in dwell_table if r["closing_kps"] == MIN_CLOSING_SPEED_KPS
    )
    return {
        "range_km": laser.range_km,
        "power_mw": laser.power_mw,
        "full_coupling_km": r_full_m / 1000.0,
        "blind_closure_kps_per_turret": blind_closure_mps / 1000.0,
        "seeker_kill_mj": TORPEDO_ELECTRONICS_THRESHOLD_J / 1e6,
        "waste_heat_mw": laser.waste_heat_w() / 1e6,
        # Guidance never lets a fuelled round close slower than this, so it is
        # the speed PD actually has to cope with (src/torpedo.py).
        "closure_floor_kps": MIN_CLOSING_SPEED_KPS,
        "dwell_table": dwell_table,
        "mj_per_turret_at_floor": at_floor["mj_per_turret"],
        "turrets_to_blind_at_floor": at_floor["turrets_to_blind"],
        # Seconds a round at the closure floor spends inside the envelope.
        "envelope_crossing_s": laser.range_km / MIN_CLOSING_SPEED_KPS,
    }


def thermal_doctrine_numbers(
    fleet_data: Optional[Dict[str, Any]], ship_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Heat figures for prompts, derived from the same data and constants the
    engine runs (ThermalSystem.from_ship_data, power.KINETIC_WEAPON_EFFICIENCY,
    the heat ladder in src/thermal.py, combat.radiator_hit_chance), never
    hand-quoted. Bound by tests/test_doctrine_accuracy.py.
    """
    ships = (fleet_data or {}).get("ships", {})
    spec = ships.get(ship_type) if ship_type else None
    if spec is None:
        spec = ships.get("destroyer") or (next(iter(ships.values())) if ships else {})
    thermal_data = (spec or {}).get("thermal", {})
    radiators = _thermal.RadiatorArray.from_ship_data(thermal_data)
    sink = _thermal.HeatSink.from_ship_data(thermal_data)
    radiator_mw = radiators.nominal_dissipation_kw / 1000.0
    drive_mw = radiator_mw * _thermal.DRIVE_HEAT_TO_RADIATOR_RATIO

    weapon_types = (fleet_data or {}).get("weapon_types", {})
    per_shot: Dict[str, float] = {}
    for weapon in (spec or {}).get("weapons", []):
        wt = weapon_types.get(weapon.get("type"), {})
        ke = wt.get("kinetic_energy_gj", 0.0) or 0.0
        if ke > 0:
            per_shot[wt.get("name", weapon.get("type"))] = (
                ke * (1.0 / KINETIC_WEAPON_EFFICIENCY - 1.0))
    pd_spec = weapon_types.get("pd_laser")
    laser = PDLaser.from_fleet_data(pd_spec) if pd_spec else PDLaser()

    return {
        "sink_gj": sink.capacity_gj,
        "radiator_mw": radiator_mw,
        "drive_full_mw": drive_mw,
        "full_burn_fill_min": (sink.capacity_gj / (drive_mw / 1000.0) / 60.0
                               if drive_mw > 0 else float("inf")),
        "per_shot_gj": per_shot,
        "pd_heat_mw": laser.waste_heat_w() / 1e6,
        "gun_hold_pct": _thermal.HEAT_GUN_HOLD_FRACTION * 100.0,
        "forced_pct": _thermal.HEAT_AUTO_EXTEND_FRACTION * 100.0,
        "meltdown_pct": _thermal.REACTOR_MELTDOWN_HEAT_FRACTION * 100.0,
        "auto_default_pct": _thermal.AUTO_RADIATOR_DEFAULT_PCT,
        "rad_hit_extended_pct": _combat.RADIATOR_HIT_CHANCE_EXTENDED * 100.0,
        "rad_hit_retracted_pct": _combat.RADIATOR_HIT_CHANCE_RETRACTED * 100.0,
        "rad_hit_nose_extended_pct": _combat.RADIATOR_HIT_CHANCE_NOSE_EXTENDED * 100.0,
        "rad_hit_nose_retracted_pct": _combat.RADIATOR_HIT_CHANCE_NOSE_RETRACTED * 100.0,
    }


def thermal_doctrine_block(nums: Dict[str, Any]) -> str:
    """The captain's THERMAL doctrine block (static per hull, cacheable)."""
    shots = ", ".join(f"{name} {gj:.2f} GJ" for name, gj in nums["per_shot_gj"].items())
    lines = [
        "THERMAL - HEAT IS A HARD LIMIT (heat can pass 100%; this ladder is enforced):",
        f"- Heat sink: {nums['sink_gj']:.0f} GJ capacity",
        f"- Radiators extended: +{nums['radiator_mw']:.0f} MW cooling; retracted: 0 MW cooling",
        f"- Drive heat: {nums['drive_full_mw']:.0f} MW at full burn, proportional to throttle - "
        f"exactly your radiator capacity. A full burn with radiators retracted fills the sink "
        f"in ~{nums['full_burn_fill_min']:.0f} min; extended, the drive alone holds level and "
        f"only weapon heat accumulates.",
        (f"- Weapon heat per shot: {shots}. " if shots else "- ")
        + f"Point defense: ~{nums['pd_heat_mw']:.0f} MW per firing turret.",
        f"- >= {nums['gun_hold_pct']:.0f}% heat: your GUNS HOLD FIRE (point defense keeps firing).",
        f"- >= {nums['forced_pct']:.0f}% heat: radiators are FORCED OUT (exposed to fire) and "
        f"cannot be retracted until heat drops back below {nums['forced_pct']:.0f}%.",
        f"- >= {nums['meltdown_pct']:.0f}% heat: REACTOR MELTDOWN - your ship is destroyed instantly.",
        f"- Extended radiators are exposed and CAN be shot off: a hit on your flank or tail strikes "
        f"a panel {nums['rad_hit_extended_pct']:.0f}% of the time (nose "
        f"{nums['rad_hit_nose_extended_pct']:.0f}%); retracted {nums['rad_hit_retracted_pct']:.0f}% "
        f"(nose {nums['rad_hit_nose_retracted_pct']:.0f}%). Each lost panel is a quarter of your "
        f"cooling, permanently.",
        f"- set_radiators(mode=\"auto\") hands them to the ship: extend at your chosen heat % "
        f"(default {nums['auto_default_pct']:.0f}%), retract while a torpedo is inbound. To cool "
        f"down: ease the burn below full, extend, or hold fire.",
    ]
    return "\n".join(lines)


def _fmt_minutes(seconds: Optional[float]) -> str:
    if seconds is None:
        return "never"
    if seconds <= 0:
        return "NOW"
    return f"{seconds / 60.0:.1f} min"


def format_heat_status(thermal: Optional[Dict[str, Any]], heat_percent: float) -> Dict[str, str]:
    """
    Volatile heat lines for the status block from ThermalSystem.outlook():
    heat % with the net rate, time to 100% / meltdown retracted vs extended,
    and radiator mode + panel health.
    """
    if not thermal:
        return {
            "heat_line": f"{heat_percent:.0f}%",
            "radiator_line": None,
        }
    ladder = (f"guns hold {_thermal.HEAT_GUN_HOLD_FRACTION * 100:.0f}%, radiators forced "
              f"{_thermal.HEAT_AUTO_EXTEND_FRACTION * 100:.0f}%, MELTDOWN "
              f"{_thermal.REACTOR_MELTDOWN_HEAT_FRACTION * 100:.0f}%")
    heat_line = (
        f"{thermal['heat_percent']:.0f}% ({thermal['heat_gj']:.0f}/{thermal['capacity_gj']:.0f} GJ), "
        f"net {thermal['net_rate_mw']:+.0f} MW now [{ladder}]"
    )
    if thermal.get("guns_heat_limited"):
        heat_line += " - GUNS HOLDING FIRE (too hot)"
    def _path(t100: Optional[float], tmelt: Optional[float]) -> str:
        if t100 is None:
            return "never reaches 100%"
        return f"100% in {_fmt_minutes(t100)}, meltdown in {_fmt_minutes(tmelt)}"

    heat_line += (
        f"\n  Outlook at current heat load ~{thermal['load_mw']:.0f} MW: RETRACTED -> "
        f"{_path(thermal['time_to_100_retracted_s'], thermal['time_to_meltdown_retracted_s'])}"
        f" | EXTENDED ({thermal['radiator_available_mw']:.0f} MW) -> "
        f"{_path(thermal['time_to_100_extended_s'], thermal['time_to_meltdown_extended_s'])}"
    )
    state = "EXTENDED (+cooling, exposed to fire)" if thermal["radiators_extended"] \
        else "RETRACTED (protected, no cooling)"
    mode = thermal.get("radiator_mode", "manual")
    mode_txt = (f"auto: extends at {thermal.get('auto_extend_pct', 0):.0f}%"
                if mode == "auto" else "manual")
    panels = thermal.get("radiator_panels") or {}
    panel_txt = " ".join(
        f"{k.replace('tail_', '')} {v:.0f}%" for k, v in panels.items())
    radiator_line = (
        f"{state} [{mode_txt}] - panels {panel_txt} = "
        f"{thermal['radiator_available_mw']:.0f}/{thermal['radiator_rating_mw']:.0f} MW"
    )
    return {"heat_line": heat_line, "radiator_line": radiator_line}


def format_heat_brief(thermal: Optional[Dict[str, Any]], heat_percent: float) -> str:
    """One-line heat summary for the admiral's fleet table."""
    if not thermal:
        return f"{heat_percent:.0f}%"
    text = (f"{thermal['heat_percent']:.0f}% (net {thermal['net_rate_mw']:+.0f} MW; "
            f"100% in {_fmt_minutes(thermal['time_to_100_retracted_s'])} retracted / "
            f"{_fmt_minutes(thermal['time_to_100_extended_s'])} extended; meltdown "
            f"{_fmt_minutes(thermal['time_to_meltdown_retracted_s'])} / "
            f"{_fmt_minutes(thermal['time_to_meltdown_extended_s'])})")
    if thermal.get("guns_heat_limited"):
        text += " GUNS HOLDING"
    return text


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
        # Lower thrust-to-mass than the line dreadnought: turns slower.
        "dreadnought_siege": 52.3,
    }
    key = (ship_type or "").lower()
    # Variants share their base hull's turn (cruiser_torpedo is a cruiser
    # hull, dreadnought_siege a dreadnought); they used to fall through to a
    # 20 s default, quoting a torpedo cruiser as turning like a destroyer.
    return turn_times.get(key, turn_times.get(key.split("_")[0], 20.0))


# Display names for hull variants whose ship_type does not title-case into
# something a captain would recognise ("Cruiser_Torpedo").
_CLASS_LABELS = {
    "cruiser_torpedo": "Torpedo Cruiser",
    "dreadnought_siege": "Siege Dreadnought",
}


def class_label(ship_type: Optional[str]) -> str:
    """Human class name for a ship_type ('cruiser_torpedo' -> 'Torpedo Cruiser')."""
    if not ship_type:
        return "Ship"
    return _CLASS_LABELS.get(ship_type, ship_type.replace("_", " ").title())


def ship_has_guns(ship_type: Optional[str], fleet_data: Optional[Dict[str, Any]]) -> bool:
    """
    True when the hull mounts any gun (spinal, siege or turret coilgun).

    Without fleet data the legacy destroyer template applies, which has guns.
    Torpedo-only hulls (corvette, torpedo cruiser) get no gun doctrine, gun
    status or gun forecasts anywhere in their prompt.
    """
    if not fleet_data or not ship_type:
        return True
    spec = fleet_data.get("ships", {}).get(ship_type)
    if spec is None:
        return True
    weapon_types = fleet_data.get("weapon_types", {})
    for w in spec.get("weapons", []):
        wt = weapon_types.get(w.get("type"), {})
        if w.get("type") == "pd_laser" or (w.get("type") or "").startswith("torpedo"):
            continue
        if wt.get("kinetic_energy_gj", 0) > 0 or w.get("type", "").endswith("_mk3"):
            return True
    return False


def _gun_cadence_for_prompt(spec: Dict[str, Any]) -> str:
    """Describe individual shots and the wait after the last shot."""
    cooldown = spec.get("cooldown_s", 0)
    salvo = spec.get("salvo_size", 1)
    if salvo <= 1:
        return f"{cooldown:g}s cooldown between shots"
    intra = spec.get("intra_salvo_cooldown_s", 0)
    cycle = cooldown + (salvo - 1) * intra
    return (f"{salvo}-shot salvos: {intra:g}s between shots, then {cooldown:g}s "
            f"cooldown ({cycle:g}s cycle)")


def format_weapon_groups_for_prompt(weapons: List[Dict], weapon_types: Dict[str, Any]) -> str:
    """Format weapon information grouped by type."""
    lines = []

    # Group weapons by type
    spinals = []
    heavy_coilguns = []
    coilguns = []
    other_guns = {}
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
        elif weapon_types.get(wtype, {}).get("kinetic_energy_gj", 0) > 0:
            other_guns.setdefault(wtype, []).append(w)

    # Format each group
    if spinals:
        spec = weapon_types.get("spinal_coiler_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 9.9)
        dmg = spec.get("kinetic_energy_gj", 4.32)
        rng = spec.get("range_km", 900)
        lines.append(f"- Spinal Coilgun: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage, {rng}km max range")
        lines.append(f"  * Fixed mount: REQUIRES nose within 30° of target to fire")
        lines.append(f"  * {_gun_cadence_for_prompt(spec)}")

    if heavy_coilguns:
        spec = weapon_types.get("heavy_coilgun_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 7.0)
        dmg = spec.get("kinetic_energy_gj", 1.22)
        rng = spec.get("range_km", 600)
        count = len(heavy_coilguns)
        lines.append(f"- Heavy Coilguns x{count}: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage each, {rng}km max range")
        lines.append(f"  * Turreted: 180° firing arc, works during any maneuver")
        lines.append(f"  * {_gun_cadence_for_prompt(spec)}")

    if coilguns:
        spec = weapon_types.get("coilgun_mk3", {})
        vel = spec.get("muzzle_velocity_kps", 6.0)
        dmg = spec.get("kinetic_energy_gj", 0.72)
        rng = spec.get("range_km", 500)
        count = len(coilguns)
        lines.append(f"- Coilguns x{count}: {vel} km/s muzzle velocity, {dmg:.2f} GJ damage each, {rng}km max range")
        lines.append(f"  * Turreted: 180° firing arc, works during any maneuver")
        lines.append(f"  * {_gun_cadence_for_prompt(spec)}")

    for wtype, mounts in other_guns.items():
        spec = weapon_types[wtype]
        lines.append(
            f"- {spec.get('name', wtype)} x{len(mounts)}: "
            f"{spec['muzzle_velocity_kps']:g} km/s muzzle velocity, "
            f"{spec['kinetic_energy_gj']:g} GJ damage each, "
            f"{spec['range_km']:g}km max range"
        )
        pivot = spec.get("pivot_range_deg", 180)
        if spec.get("is_turreted", True):
            lines.append(f"  * Turreted: {pivot:g}° firing arc")
        else:
            lines.append(f"  * Fixed mount: REQUIRES nose within {pivot:g}° of target to fire")
        lines.append(f"  * {_gun_cadence_for_prompt(spec)}")

    if pd_lasers:
        count = len(pd_lasers)
        # Every figure below is derived from the same PDLaser model the engine
        # runs (see pd_doctrine_numbers) - the prompt used to hardcode a 100km
        # envelope while the sim used 250km.
        pd = pd_doctrine_numbers({"weapon_types": weapon_types})
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
        # The dose PD lands is set by the TORPEDO's closing speed, not by the
        # hull-to-hull range rate. Stating the rule as "one turret stops ~5 km/s
        # of closure" was true but read as a comparison against the range rate
        # on the battlefield line - so a duel whose ships closed at 1.25 km/s
        # looked comfortably defensible while all 8 rounds went through
        # un-blinded. Quote MJ against the number on the threat line instead.
        floor = pd["closure_floor_kps"]
        need = pd["turrets_to_blind_at_floor"]
        verdict = ("CAN blind one" if count >= need else
                   f"CANNOT blind one ({need} needed)")
        table = "; ".join(
            f"{r['closing_kps']:.0f}->{r['mj_per_turret']:.0f}MJ/{r['turrets_to_blind']}t"
            for r in pd["dwell_table"]
        )
        lines.append(
            f"  * A torpedo dies by cooking its seeker ({pd['seeker_kill_mj']:.0f} MJ "
            f"absorbed). The dose is set by the TORPEDO's closing speed - the "
            f"'closing X km/s' on each INBOUND ORDNANCE line, NOT your range rate to "
            f"the enemy ship. Guidance holds a fuelled round at >={floor:.0f} km/s "
            f"however slowly the ships approach, so it crosses your "
            f"{pd['range_km']:.0f} km envelope in ~{pd['envelope_crossing_s']:.0f}s."
        )
        lines.append(f"    closing km/s -> MJ one turret lands/turrets needed: {table}")
        lines.append(
            f"    Normal launches arrive at ~{floor:.0f} km/s = "
            f"~{pd['mj_per_turret_at_floor']:.0f} MJ per turret, so {need} turrets must "
            f"bear on ONE round. You have {count}: you {verdict}. Allied turrets within "
            f"{pd['range_km']:.0f} km stack - the only way a small hull gets a kill."
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


# Short display names for every weapon type in fleet_ships.json. Anything not
# listed falls back to the weapon's own "name" - nothing is silently dropped.
_WEAPON_SHORT_NAMES = {
    "spinal_coiler_mk3": "Spinal",
    "heavy_siege_coiler_mk3": "Siege coiler",
    "heavy_coilgun_mk3": "Heavy Coilgun",
    "coilgun_mk3": "Coilgun",
    "light_coilgun_mk3": "Light coilgun",
    "torpedo_launcher": "Torpedo tube",
    "pd_laser": "PD",
}


def format_enemy_weapons_summary(
    weapons: List[Dict],
    weapon_types: Optional[Dict[str, Any]] = None,
    torpedoes_remaining: Optional[int] = None,
    pd_operational: Optional[int] = None,
) -> str:
    """
    Short armament summary for a hull: EVERY weapon type it mounts.

    This used to count only spinal / heavy / coilgun / PD and silently drop
    torpedo launchers and the siege coiler, so a torpedo corvette read
    "1x PD" and a 48-round torpedo cruiser "4x PD" - captains and admirals
    concluded the enemy had "no offensive weapons" and never pre-empted the
    salvo that killed them.

    torpedoes_remaining / pd_operational: live counts when observed (the
    magazine total from the spec otherwise; PD shown as live/total).
    """
    counts: Dict[str, int] = {}
    order: List[str] = []
    magazine = 0
    for w in weapons:
        wtype = w.get("type", "")
        name = _WEAPON_SHORT_NAMES.get(wtype) or (
            (weapon_types or {}).get(wtype, {}).get("name") or wtype or "weapon")
        if name not in counts:
            order.append(name)
        counts[name] = counts.get(name, 0) + 1
        if wtype.startswith("torpedo"):
            magazine += int(w.get("magazine")
                            or (weapon_types or {}).get(wtype, {}).get("magazine")
                            or 8)

    parts = []
    for name in order:
        n = counts[name]
        if name == "Torpedo tube":
            rounds = magazine if torpedoes_remaining is None else torpedoes_remaining
            left = "rounds" if torpedoes_remaining is None else "rounds left"
            parts.append(f"{n}x Torpedo tube ({rounds} {left})")
        elif name == "PD" and pd_operational is not None:
            parts.append(f"PD {pd_operational}/{n}")
        else:
            parts.append(f"{n}x {name}")
    return ", ".join(parts) if parts else "Unknown armament"


def class_reference_block(fleet_data: Optional[Dict[str, Any]]) -> str:
    """
    Static per-class table (acceleration, turn, full armament) for the cached
    doctrine, so per-enemy lines in the volatile turn carry only live state.
    """
    ships = (fleet_data or {}).get("ships", {})
    if not ships:
        return ""
    weapon_types = (fleet_data or {}).get("weapon_types", {})
    lines = ["CLASS REFERENCE (every hull in the game; the turn names each ship's class):"]
    for stype, spec in ships.items():
        perf = spec.get("performance", {})
        lines.append(
            f"  {class_label(stype):<18} {perf.get('combat_acceleration_g', '?')}g, "
            f"90° turn ~{get_ship_turn_time_90deg(stype):.0f} s | "
            f"{format_enemy_weapons_summary(spec.get('weapons', []), weapon_types)}"
        )
    return "\n".join(lines) + "\n"


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
YOUR SHIP CLASS: Destroyer

PROPULSION:
- Acceleration: 2.0g (~20 m/s²)
- Delta-V budget: 500 km/s total
- 90° turn: ~20 seconds (thrust vectoring while burning)

WEAPONS:
- Spinal Coilgun: 9.9 km/s muzzle velocity, 4.3 GJ damage, 900km max range
  * Fixed mount: REQUIRES nose within 30° of target to fire
  * 5-shot salvos: 12s between shots, then 24s cooldown (72s cycle)
- Turret Coilgun: 6.0 km/s muzzle velocity, 0.7 GJ damage, 500km max range
  * Turreted: 180° firing arc, works during any maneuver
  * 5-shot salvos: 10s between shots, then 20s cooldown (60s cycle)
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

{thermal_block}
"""

SHIP_CAPABILITIES_TEMPLATE = """
YOUR SHIP CLASS: {ship_class}

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

{thermal_block}

"""

# Volatile per-turn status. Kept OUT of the system prompt so the stable doctrine
# above can be served from the provider's prompt cache across checkpoints.
SHIP_STATUS_TEMPLATE = """
CURRENT STATUS:
- Hull: {hull_integrity:.0f}%
- Heat: {heat_line}
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
    per_decision = sum(min(l.get("magazine", wt.get("magazine", 8)),
                           math.ceil(decision_interval_s / cooldown)) for l in launchers)
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
  -> up to {per_decision} launches per {decision_interval_s:g}s decision if all tubes start ready.
  Ready tubes fire together; additional rounds launch as each tube reloads.
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
  {accel}g guidance, and your round crosses their point defense envelope at the
  >={pd['closure_floor_kps']:.0f} km/s guidance floor whatever the two ships are doing.
  That is only ~{pd['mj_per_turret_at_floor']:.0f} MJ per defending turret against a
  {pd['seeker_kill_mj']:.0f} MJ seeker kill: they need {pd['turrets_to_blind_at_floor']} turrets
  bearing on it. Expect near-100% on stragglers and on 1-2 turret hulls, heavy attrition
  against an escorted evading wall.
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
    thermal: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the volatile per-checkpoint ship status block.

    Split out of the capabilities section so the static ship specification can
    live in a cacheable system prompt while this changes every checkpoint.
    thermal: ThermalSystem.outlook() - adds the net heat rate, time to 100% /
    meltdown retracted vs extended, and radiator mode and panel health.
    """
    # Radiator damage IS live in the engine now (simulation.py resolves radiator
    # hits through RadiatorHitResolver), so the vulnerability is real and the
    # captain should be reasoning about it.
    heat = format_heat_status(thermal, heat_percent)
    radiator_status = heat["radiator_line"] or (
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
        heat_line=heat["heat_line"],
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
    decision_interval_s: float = 30.0,
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
    ship_class = class_label(ship_type)

    # Heat doctrine straight from fleet data and engine constants (the same
    # RadiatorArray / HeatSink construction the sim uses), so it cannot drift.
    thermal_block = thermal_doctrine_block(thermal_doctrine_numbers(fleet_data, ship_type))

    _armor = armor_from_fleet_data(ship_type, fleet_data)
    _torpedoes = _torpedo_capability_block(ship_type, fleet_data, decision_interval_s)
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
        thermal_block=thermal_block,
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


# Reference data for projectile physics - given once so LLM can calculate.
# Gun hulls only: a torpedo boat carries no guns, and handing it spinal
# tables made corvette captains (and their admirals) reason about - and once
# "enable" - coilguns the hull does not mount.
GUN_PHYSICS_REFERENCE = """
PROJECTILE PHYSICS (for your calculations):
- Spinal round: 9.9 km/s → at 100km takes ~10s, at 200km ~20s, at 500km ~50s
- Turret round: 6.0 km/s → at 100km takes ~17s, at 200km ~33s, at 500km ~83s
- EVADE maneuver: cuts enemy hit probability by ~40% (x0.6 multiplier in the
  fire-control model), via random jinking ~500m/s lateral
- Your ship width: ~20m, length ~125m

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

# Every hull.
ARMOR_BREACH_REFERENCE = """
CRITICAL - ARMOR BREACH:
- When armor reaches 0cm, hits penetrate directly to internal modules
- Penetrating hits are CATASTROPHIC: 4-5 hits through breached armor = ship destroyed
- Each penetrating hit destroys 1-2 modules and damages deeper systems
- Bridge or Reactor destroyed = immediate ship kill
- PROTECT YOUR WEAKEST ARMOR FACING - keep it pointed away from enemy!
"""

# Backwards-compatible combined block (gun physics + breach).
PROJECTILE_PHYSICS_REFERENCE = GUN_PHYSICS_REFERENCE + ARMOR_BREACH_REFERENCE

# Gun hulls: how to fight with guns.
GUN_TACTICS_BLOCK = """=== TACTICAL OPTIONS (pick your style) ===

You can fight however you want. Here are some approaches to consider:

OPTION A - HEAD-ON CHARGE:
  INTERCEPT constantly → high closing velocity → brief exchange → fly past
  Pros: Simple, aggressive, intimidating
  Cons: Short engagement window, hard to hit at high relative velocity, wastes fuel

OPTION B - CONTROLLED PASSES (recommended for accuracy):
  1. Approach (INTERCEPT from far out, or set_heading for your approach angle)
  2. Before you are committed: cancel closure (or pass wide) so the pass is slow or wide
  3. Coast and shoot: PADLOCK - spinal and nose armor on him during the pass
  Pros: Better hit probability, more sustained fire, fuel efficient
  Cons: Takes longer to close, requires planning

OPTION C - EVADE AND SHOOT:
  EVADE constantly - random thrust makes you unpredictable (holds your course)
  Pros: Very hard to hit, fires continuously
  Cons: Burns fuel faster, harder to line up spinal shots, does not control range

OPTION D - HOLD RANGE / KITE:
  cancel closure at your chosen range, then alternate coast-and-shoot with short
  range-correcting burns (set_heading from RANGE CONTROL). A faster hull can hold
  range on a slower one indefinitely.
  Pros: Controls engagement tempo, picks your range
  Cons: Requires more active management

KEY PHYSICS TO REMEMBER:
- High relative velocity = harder to hit (projectiles lead wrong, brief window)
- 0 km/s relative + no evasion = sitting duck (easy hit)
- Sweet spot: 1-3 km/s relative with active maneuvering
- Spinal weapons ONLY fire if nose within 30° of target - PADLOCK helps with this
- Turrets fire at 180° arc - don't need nose-on alignment"""

# Torpedo-only hulls (corvette, torpedo cruiser): no gun content at all.
TORPEDO_TACTICS_BLOCK = """=== TORPEDO BOAT TACTICS (you mount no guns) ===
- Your launchers are your only weapon; point defense only defends. Every round counts.
- Each ENEMIES entry carries a torpedo line: the engine FLIES a round from you to him
  before you decide. Launch at REACHABLE targets. UNREACHABLE (misses even if he holds
  course), ESCAPABLE (he only has to EVADE), TOO CLOSE (guidance has no room to build
  speed) and OVERKILL (enough rounds already inbound) launches are wasted rounds.
- Range is your friend: rounds launched from a few hundred km arrive far harder than
  knife-fight launches. Launch on the approach, then open range or pass wide - a merge
  gains a torpedo boat nothing.
- Salvo, don't trickle: launch the "salvo >= N together" his point-defense cover needs
  in ONE checkpoint, timed with the fleet's launches.
- Enemy torpedo hulls that have you targeted can land a round before your next
  decision: THREATS TO ME says when. EVADE early."""

# Stable across every checkpoint of a battle -> cacheable prefix. Nothing
# ship-specific (name, id) lives here, so every hull of a class with the same
# personality shares one cached prefix; the name opens the volatile turn.
CAPTAIN_DOCTRINE = """
You are a starship captain in a space combat simulation. Your name, your ship and the
tactical picture open every turn.

{simulation_disclaimer}

{ship_capabilities}
{class_reference}
{projectile_reference}
{armor_breach_reference}
=== CONTROLS ===

MANEUVERS:
- INTERCEPT: burn straight at the target. Only for the first approach from far out or to
  finish a cripple - it builds closure you cannot take back (see RANGE CONTROL below).
- EVADE: Threat-aware defense, BEST during active combat. Versus gun rounds it jinks
  (hard to hit). Versus guided torpedoes it automatically RUNS (burns away from the
  round to cut closure and buy your PD dwell time) and in the final seconds
  automatically PRESENTS your thickest armor to the impact. Only EVADE does this -
  any other maneuver takes torpedoes, including blinded ones, without reacting.
  With no threat it weaves around your CURRENT course: it never chases, but it
  never brakes either - EVADE does not stop a merge.
- BRAKE: burn against your own (world-frame) velocity - not against the enemy's
- MAINTAIN: Coast - no thrust, no rotation, pure drift
- PADLOCK: coast with the nose tracking the target (nose armor - and a spinal - on him). NO closing thrust;
  the drive idles at 5% only while turning, for fast thrust-vectored turns.
- set_heading(direction, throttle): point the ship along a WORLD-frame direction. It
  turns onto the heading first, then burns (the drive thrusts along the NOSE).
  throttle 0 = ATTITUDE ONLY: the nose turns on RCS and you coast with zero change of
  velocity (e.g. nose on a fixed bearing) - RCS turns are slow, see your turn times.
  NOTE ON FRAMES: set_heading takes a WORLD-frame direction vector (x, y, z world axes),
  NOT a ship-relative one and NOT a position. Every ship in your turn carries
  "dir_to (x,y,z)": the world unit vector from you to it - paste it as the direction.
  A POSITION (e.g. an admiral's "(260,0,0) km") is not a direction: subtract YOUR world
  position (MY SHIP) first - your ORDERS block resolves such points for you.
  Ahead/starboard/above words are relative to YOUR NOSE (MY SHIP gives the nose vector).

=== RANGE CONTROL (Newtonian - read this) ===
- There are no brakes, only burns. Stopping a closure takes v^2/(2a) of room plus the
  flip: at 4.6 km/s closure a 3g corvette needs 156 s / 359 km (+18 s to flip), a 1g
  battleship 469 s / 1078 km (+55 s). By the time the enemy "looks close" the merge is
  locked in: a pass at a few km, seconds of fire, then flying apart.
- INTERCEPT only for the first approach or finishing a cripple. For EVERYTHING that
  involves range - approach angle, holding range, kiting, avoiding a merge, flanking -
  use set_heading with a vector from RANGE CONTROL in your tactical data:
    cancel closure    burn against the relative velocity - range holds where it stops
    open range        burn straight away from the enemy group
    pass wide         burn across the line of sight - a merge becomes a wide pass
    flank left/right  move across his line of fire
  Each line gives the burn time, the effect, and which way your NOSE points meanwhile.
  Renew the same heading each checkpoint until the stated burn time is used up.
- "!! MERGE IN ..." means act THIS checkpoint; if braking is too late, pass wide.
{rhythm_note}

{tactics_block}

=== TORPEDO DEFENSE (act on this the moment ordnance is inbound) ===
- Your PD is automatic; it BLINDS torpedoes (seeker kill at 50 MJ absorbed beam), it
  does not vaporise them. A blinded round coasts ballistically on its last course and
  STILL HITS a ship flying straight. PD + EVADE stops torpedoes; PD alone stops nothing.
- THE CLOSURE THAT SIZES YOUR PD IS THE TORPEDO'S, NOT THE ENEMY SHIP'S. Two hulls
  drifting together at 1 km/s still take rounds crossing the PD envelope at 12+ km/s
  and landing at 13-28 km/s: guidance holds a closure floor whatever the ships do. Judge PD
  by the "closing X km/s" on the INBOUND ORDNANCE line against the MJ-per-turret table
  in your PD spec - never by your range rate. Measured: 1-2 turret hulls do NOT blind a
  normal launch (<=46 of the 50 MJ needed); 3+ turrets on one round do. A lone ship
  does NOT stop a committed torpedo - EVADE, nose armor, kill the shooter.
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
{gun_notes}
TARGET SELECTION:
- set_primary_target: {primary_target_note}
- Rank targets by DAMAGE YOU CAN DELIVER x THREAT HE POSES: {damage_measure}, against
  what he can still do to your fleet (his guns and torpedo tubes, listed per enemy).
  Closeness alone is not a reason: {closeness_example}.
- CLOSING targets are easier to hit and to torpedo; OPENING targets escape torpedoes
  and waste gun time - do not chase them.
- Enemy torpedo hulls that have you or your fleet targeted are the threat to remove
  before they launch.
{comms_block}
OTHER:
- set_radiators: extend (cooling) or retract (protection)
- surrender / propose_draw: End battle (draw decided by points)
- log_note: CAPTAIN'S LOG - a short note to your future self, shown back to you
  every checkpoint. It is your only memory of your own intent between decisions:
  each checkpoint you see fresh data but not what you were trying to do. Log your
  plan and what would change it (e.g. "Pass 2: brake at 150km, torpedoes when his
  PD is under 3 turrets"), then next checkpoint execute it or log the new plan.

=== YOUR TURN ===
Every turn lists, in order: MY SHIP (your world position, velocity and nose - the frame
set_heading uses), THREATS TO ME, ENEMIES, FRIENDLIES, RANGE CONTROL, ORDERS, FLEET NET /
ENEMY TRANSMISSIONS, then your recent decisions, captain's log and ENGINE FEEDBACK (what
your ship actually did with your last orders, and any call that was rejected).

*** EXECUTE YOUR ORDERS WITH THE AVAILABLE TOOLS ***
- EVERY decision needs at least one action tool: {action_tools}.
  A reply with only words, reports or notes moves and fires NOTHING.
- Renew set_maneuver or set_heading each checkpoint to keep maneuvering.
  When a maneuver expires, thrust stops; you coast with your existing momentum.
  MAINTAIN is a deliberate coast - call it when coasting IS the plan.
  Exception: a HEADING ORDER from your admiral is already applied to your ship -
  calling set_maneuver/set_heading REPLACES it (and breaks formation). Only override
  it for a reason (e.g. EVADE an inbound torpedo).
{execute_weapon_notes}Describing an action does not execute it. Use the corresponding tool.

TIMING: You make ONE decision now. It is LOCKED for {decision_interval_s:g} seconds until next checkpoint.
You CANNOT react mid-checkpoint or change orders. No "if X then Y" - decide NOW.
The automatic fire-control, point-defense and evasion systems react between checkpoints.

{personality_prompt}
"""

# RANGE CONTROL rhythm, per armament (CAPTAIN_DOCTRINE {rhythm_note}).
GUN_RHYTHM_NOTE = """- RHYTHM: burn phases and shooting phases alternate. Burning for range points the nose
  along the heading - cancel closure = TAIL to the enemy: thin tail armor exposed,
  spinal masked, turrets only if he is inside their arc. Then coast and shoot: PADLOCK
  (or set_heading toward him at throttle 0) swings nose armor and spinal back on him."""
TORPEDO_RHYTHM_NOTE = """- RHYTHM: burn phases and coasting alternate. Burning for range points the nose along
  the heading - cancel closure = TAIL to the enemy: thin tail armor exposed. Launch
  before or between burns (tubes fire in any attitude), then swing nose armor back toward
  the threat (PADLOCK, or set_heading toward him at throttle 0) before rounds arrive."""

# Gun hulls only (CAPTAIN_DOCTRINE {gun_notes}).
GUN_NOTES = """
AMMO NOTE: 450+ spinal rounds, 1800+ turret rounds. Max ~80 shots per battle.
Ammo is effectively UNLIMITED. Shoot freely.

WEAPONS (set independently with set_weapons_order):
- spinal_mode / turret_mode: FIRE_IMMEDIATE, FIRE_WHEN_OPTIMAL, FIRE_AT_RANGE, HOLD_FIRE
- Spinal: High damage, requires nose-on (30° limit)
- Turret: Lower damage, 180° firing arc (more flexible)
"""

# Marker opening the volatile per-checkpoint turn (build_captain_messages and
# tests split on it).
CAPTAIN_TURN_MARKER = "=== TACTICAL DATA (T+"


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
    decision_interval_s: float = 30.0,
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
            decision_interval_s=decision_interval_s,
        )

    # Legacy fallback using destroyer template
    radiator_status = "EXTENDED (cooling)" if radiators_extended else "RETRACTED (no cooling)"
    weapon_status = format_weapon_status(weapons or {})
    damage_report = format_damage_report(damaged_modules or {})

    return SHIP_CAPABILITIES_DESTROYER.format(
        thermal_block=thermal_doctrine_block(thermal_doctrine_numbers(None)),
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
                weapons_summary = format_enemy_weapons_summary(
                    enemy_weapons, fleet_data.get("weapon_types", {}),
                    torpedoes_remaining=e.get("torpedoes_remaining"),
                    pd_operational=e.get("pd_operational"))

                lines.append(f"    Capabilities: {accel}g accel, {delta_v} km/s delta-v, ~{turn_time:.0f}s turn")
                lines.append(f"    Armament: {weapons_summary}")

            # Current fire-control probability and phase-aware shot opportunity.
            forecasts = e.get("weapon_hit_forecasts")
            if forecasts:
                for forecast in forecasts:
                    lines.append(
                        f"    {forecast['weapon_slot']} ({forecast['weapon_name']}): "
                        f"P(hit) {forecast['hit_probability']:.0%}; "
                        f"expected hits/checkpoint if firing now: "
                        f"{forecast['expected_hits']:.2f} "
                        f"({forecast['hit_probability']:.0%} x "
                        f"{forecast['shots']} shots/{forecast['window_s']:g}s)"
                    )
                lines.append("    Assumes current aim/evasion persist and power/heat permit; "
                             "shot counts include cooldown, salvo position and ammo.")
            else:
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
            if mode == "WOBBLE":
                mode = "WOBBLE (no threat: weaving around your current course - not braking)"
            lines.append(f"    Evasion mode: {mode}")
        else:
            lines.append(f"  Current maneuver: {maneuver_type} @ {throttle_pct}% throttle")
    else:
        lines.append("  Current maneuver: NONE (drifting)")

    # Always show threat assessment (helps LLM decide whether to evade). Only
    # threats on a collision course with THIS ship are counted (see
    # simulation._get_evasion_status), broken down by kind.
    if evasion_status:
        threat_count = evasion_status.get("threat_count", 0)
        if threat_count > 0:
            mode = evasion_status.get("mode", "WOBBLE")
            lines.append(f"  {format_threat_breakdown(evasion_status)} - evasion would use {mode} mode")

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


def format_threat_breakdown(evasion_status: Optional[Dict[str, Any]],
                            prefix: str = "Threats on a collision course") -> str:
    """'Threats on a collision course: 2 gun rounds, 1 guided torpedo, 0 blinded wrecks'."""
    es = evasion_status or {}
    gun = es.get("gun_rounds")
    if gun is None:  # legacy status dict: total only
        return f"{prefix}: {es.get('threat_count', 0)}"
    guided = es.get("guided_torpedoes", 0)
    wrecks = es.get("blinded_wrecks", 0)
    return (f"{prefix}: {gun} gun round{'s' if gun != 1 else ''}, "
            f"{guided} guided torpedo{'es' if guided != 1 else ''}, "
            f"{wrecks} blinded wreck{'s' if wrecks != 1 else ''}")


def build_personality_prompt(
    personality: CaptainPersonality = CaptainPersonality.BALANCED,
    personality_text: Optional[str] = None,
    notebook_text: Optional[str] = None,
) -> str:
    """Personality (+ commander's notebook) for the cached doctrine."""
    personality_prompt = ""
    if personality_text:
        personality_prompt = f"YOUR PERSONALITY:\n{personality_text}"
    else:
        key = personality.value if isinstance(personality.value, str) else personality.name.lower()
        preset = PERSONALITY_PRESETS.get(key)
        if preset:
            personality_prompt = (f"YOUR PERSONALITY: {preset['name']}\n{preset['description']}"
                                  f"\nMotto: \"{preset['motto']}\"")
    # Commander's notebook (cross-battle lessons) rides with the personality in
    # the stable doctrine prefix: fixed for the whole battle, so it caches.
    if notebook_text:
        personality_prompt = (f"{personality_prompt}\n\n{notebook_text}" if personality_prompt
                              else notebook_text)
    return personality_prompt


def build_captain_doctrine(
    ship_type: Optional[str] = None,
    fleet_data: Optional[Dict[str, Any]] = None,
    personality: CaptainPersonality = CaptainPersonality.BALANCED,
    personality_text: Optional[str] = None,
    notebook_text: Optional[str] = None,
    decision_interval_s: float = 30.0,
    tool_names: Optional[Iterable[str]] = None,
    has_admiral: bool = False,
    ship_status: Optional[Dict[str, Any]] = None,
    **_ignored: Any,
) -> str:
    """
    The captain's cached system prompt: identical across every checkpoint of a
    battle and free of the ship's name, so hulls of one class with the same
    personality share one cached prefix.

    Gated on armament: torpedo-only hulls get no gun physics, gun tactics,
    ammo or weapons-order text (they used to, and reasoned about spinal
    coilguns they do not mount); gun hulls get no torpedo tactics.
    tool_names: the tools this captain is offered (fleet_signal / report
    tool / launch_torpedo mentions follow them).
    """
    ship_status = ship_status or {}
    names = set(tool_names or ())
    has_guns = ship_has_guns(ship_type, fleet_data)
    has_torps = ("launch_torpedo" in names) or bool(
        fleet_data and ship_type and any(
            (w.get("type") or "").startswith("torpedo")
            for w in fleet_data.get("ships", {}).get(ship_type, {}).get("weapons", [])))

    ship_capabilities = build_ship_capabilities(
        ship_name="",
        hull_integrity=ship_status.get("hull_integrity", 100),
        heat_percent=ship_status.get("heat_percent", 0),
        delta_v_remaining=ship_status.get("delta_v_remaining", 500),
        nose_armor=ship_status.get("nose_armor", 10),
        lateral_armor=ship_status.get("lateral_armor", 5),
        tail_armor=ship_status.get("tail_armor", 3),
        heatsink_capacity=ship_status.get("heatsink_capacity", 525),
        radiators_extended=False,
        ship_type=ship_type or "destroyer",
        fleet_data=fleet_data,
        decision_interval_s=decision_interval_s,
    )

    actions = ["set_maneuver or set_heading"]
    if has_guns:
        actions.append("set_weapons_order")
    if has_torps:
        actions.append("launch_torpedo")
    actions.append("set_primary_target")
    weapon_notes = []
    if has_guns:
        weapon_notes += [
            "- set_primary_target changes the target followed by persistent gun orders.",
            "- set_weapons_order sets firing policy; it persists until changed.",
            "- Guns fire salvos one slug at a time; the hit-probability gate and firing arc",
            "  are checked for EVERY shot. Interrupted salvos resume when firing is possible.",
        ]
    if has_torps:
        weapon_notes.append("- launch_torpedo orders a salvo for this decision window; check the "
                            "torpedo line of the target first.")

    comms = ["COMMUNICATION (optional - only when it changes what someone will do):"]
    if "fleet_signal" in names:
        comms.append("- fleet_signal: a short line to your own fleet (launch intent, target split, "
                     "PD cover request); your fleet reads it in FLEET NET.")
    if "send_message" in names or not names:
        comms.append("- send_message: the OPEN channel - the ENEMY reads it (taunts, surrender "
                     "demands). Not for coordination.")
    if has_admiral:
        report = "report_to_admiral" if "report_to_admiral" in names else "respond_to_orders"
        comms.append(f"- {report}: to your Admiral, ONLY to DEVIATE or report UNABLE (say why) or "
                     "to file a request/contact. Acting on your orders with tools IS the "
                     "acknowledgment - no reply needed.")
        comms.append("- discuss_with_admiral: ask before you act when an order is unclear "
                     "(one round trip).")
        comms.append("- ORDERS: follow them unless a CHECK printed under an order shows it fails "
                     "(UNREACHABLE salvo, merge, wrong range); then DEVIATE and say why.")

    return CAPTAIN_DOCTRINE.format(
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        ship_capabilities=ship_capabilities,
        class_reference=class_reference_block(fleet_data),
        projectile_reference=GUN_PHYSICS_REFERENCE if has_guns else "",
        armor_breach_reference=ARMOR_BREACH_REFERENCE,
        tactics_block=GUN_TACTICS_BLOCK if has_guns else TORPEDO_TACTICS_BLOCK,
        gun_notes=GUN_NOTES if has_guns else "",
        rhythm_note=GUN_RHYTHM_NOTE if has_guns else TORPEDO_RHYTHM_NOTE,
        damage_measure=" or ".join(
            ([("your expected GJ per shot (P(hit) x round energy)")] if has_guns else [])
            + (["a REACHABLE torpedo line and its impact energy"] if has_torps else []))
        or "the damage you can actually land",
        closeness_example=("a small fast hull at long range soaks shots for nothing" if has_guns
                           else "a close target whose torpedo line says TOO CLOSE is a wasted round"),
        primary_target_note=(
            ("the enemy your guns and INTERCEPT/PADLOCK follow"
             if has_guns else "the enemy INTERCEPT/PADLOCK and your launches follow by default")
            + (" (launch_torpedo can take its own target_id)" if has_torps else "")),
        comms_block="\n".join(comms) + "\n",
        action_tools=", ".join(actions),
        execute_weapon_notes="".join(f"{line}\n" for line in weapon_notes),
        decision_interval_s=decision_interval_s,
        personality_prompt=build_personality_prompt(personality, personality_text, notebook_text),
    )


# ---------------------------------------------------------------------------
# Volatile captain turn
# ---------------------------------------------------------------------------

# Per-item budget for FLEET NET / ENEMY TRANSMISSIONS (~40 tokens each).
COMMS_ITEM_MAX_CHARS = 160
FLEET_NET_MAX_ITEMS = 8
ENEMY_TRANSMISSIONS_MAX_ITEMS = 3
# Channels of the comms_message schema heard as friendly traffic.
FRIENDLY_CHANNELS = frozenset({"fleet_net", "direct", "admiral_to_captain"})
# Heat outlook is shown only when 100% is this close (seconds).
HEAT_OUTLOOK_SHOW_S = 600.0


def strip_captain_title(name: Optional[str]) -> str:
    """'Captain Dart-3' -> 'Dart-3' (configs often include the title already)."""
    name = (name or "").strip()
    return name[len("Captain "):] if name.startswith("Captain ") else name


def ship_ref(name: Optional[str], ship_id: Optional[str]) -> str:
    """'OCS Lancer-1 [beta_3]': the id disambiguates names shared across fleets."""
    name = name or ship_id or "?"
    return f"{name} [{ship_id}]" if ship_id and ship_id != name else str(name)


def _vec(d: Optional[Dict[str, float]], fmt: str = "+.2f") -> str:
    d = d or {}
    return f"({d.get('x', 0.0):{fmt}},{d.get('y', 0.0):{fmt}},{d.get('z', 0.0):{fmt}})"


def _rate(closing_kps: float) -> str:
    return (f"closing {closing_kps:.2f} km/s" if closing_kps >= 0
            else f"OPENING {-closing_kps:.2f} km/s")


def _body_words(bearing: Optional[Dict[str, float]]) -> str:
    if not bearing:
        return ""
    return "-".join((
        "ahead" if bearing.get("forward", 0.0) >= 0 else "behind",
        "starboard" if bearing.get("starboard", 0.0) >= 0 else "port",
        "above" if bearing.get("up", 0.0) >= 0 else "below",
    ))


def _maneuver_text(m: Optional[Dict[str, Any]]) -> str:
    if not m:
        return "coasting"
    kind = m.get("type", "?")
    kind = "EVADE" if kind == "EVASIVE" else kind
    text = f"{kind}@{float(m.get('throttle', 1.0)) * 100:.0f}%"
    if kind == "HEADING" and m.get("heading"):
        text = f"HEADING {_vec(m['heading'])}@{float(m.get('throttle', 1.0)) * 100:.0f}%"
    return text


def _heat_lines(ship_status: Dict[str, Any]) -> List[str]:
    """Heat + radiators; the time-to-100% outlook only when it is close."""
    thermal = ship_status.get("thermal")
    heat_pct = ship_status.get("heat_percent", 0.0)
    if not thermal:
        rad = "EXTENDED (+cooling, exposed)" if ship_status.get("radiators_extended") \
            else "RETRACTED (protected, no cooling)"
        return [f"  heat {heat_pct:.0f}% | radiators {rad}"]
    line = (f"  heat {thermal['heat_percent']:.0f}% ({thermal['heat_gj']:.0f}/"
            f"{thermal['capacity_gj']:.0f} GJ), net {thermal['net_rate_mw']:+.0f} MW")
    if thermal.get("guns_heat_limited"):
        line += " - GUNS HOLDING FIRE (too hot)"
    extended = thermal.get("radiators_extended")
    t100 = thermal.get("time_to_100_extended_s" if extended else "time_to_100_retracted_s")
    if thermal["heat_percent"] >= 50 or (t100 is not None and t100 < HEAT_OUTLOOK_SHOW_S):
        line += (f" [guns hold {_thermal.HEAT_GUN_HOLD_FRACTION * 100:.0f}%, radiators forced "
                 f"{_thermal.HEAT_AUTO_EXTEND_FRACTION * 100:.0f}%, MELTDOWN "
                 f"{_thermal.REACTOR_MELTDOWN_HEAT_FRACTION * 100:.0f}%]")
    if t100 is not None and t100 < HEAT_OUTLOOK_SHOW_S:
        tmelt = thermal.get("time_to_meltdown_extended_s" if extended
                            else "time_to_meltdown_retracted_s")
        line += f" | at this load: 100% in {_fmt_minutes(t100)}, meltdown in {_fmt_minutes(tmelt)}"
        if not extended:
            other = thermal.get("time_to_100_extended_s")
            line += (" (extended: never)" if other is None
                     else f" (extended: 100% in {_fmt_minutes(other)})")
    state = "EXTENDED (+cooling, exposed to fire)" if extended else "RETRACTED (protected, no cooling)"
    mode = thermal.get("radiator_mode", "manual")
    mode_txt = (f"auto: extends at {thermal.get('auto_extend_pct', 0):.0f}%" if mode == "auto"
                else "manual")
    panels = thermal.get("radiator_panels") or {}
    damaged = [f"{k.replace('tail_', '')} {v:.0f}%" for k, v in panels.items() if v < 99.5]
    rad = (f"  radiators {state} [{mode_txt}] {thermal['radiator_available_mw']:.0f}/"
           f"{thermal['radiator_rating_mw']:.0f} MW")
    if damaged:
        rad += " - damaged panels: " + ", ".join(damaged)
    return [line, rad]


def _weapons_line(ship_status: Dict[str, Any], has_guns: bool,
                  weapon_orders: Optional[Dict[str, str]] = None) -> str:
    parts = []
    remaining = ship_status.get("torpedoes_remaining")
    if remaining is not None:
        cap = ship_status.get("torpedo_capacity")
        tubes = ship_status.get("torpedo_tubes") or []
        ready = sum(1 for t in tubes if t.get("ready"))
        txt = f"torpedoes {remaining}{'/' + str(cap) if cap else ''}"
        if tubes:
            reloading = [t["cooldown"] for t in tubes if not t.get("ready") and t.get("cooldown")]
            txt += f", tubes {ready}/{len(tubes)} ready"
            if reloading:
                txt += f" (next in {min(reloading):.0f} s)"
        capacity = ship_status.get("launch_capacity")
        if capacity is not None:
            txt += f", up to {capacity} launch{'es' if capacity != 1 else ''} before your next decision"
        parts.append(txt)
    if has_guns and not ship_status.get("gun_groups"):
        # No per-mount status (legacy callers): the standing gun orders.
        orders = weapon_orders or {"spinal": "HOLD_FIRE", "turret": "HOLD_FIRE"}
        parts.append("guns " + ", ".join(f"{g} {m}" for g, m in orders.items()))
    if has_guns:
        for g in ship_status.get("gun_groups") or []:
            state = []
            if g.get("ready"):
                state.append(f"{g['ready']} ready")
            if g.get("reloading"):
                state.append(f"{len(g['reloading'])} reloading {min(g['reloading']):.0f} s")
            if g.get("destroyed"):
                state.append(f"{g['destroyed']} DESTROYED")
            label = g.get("label", g.get("group", "guns"))
            count = g.get("count", 1)
            parts.append(f"{label}{' x' + str(count) if count > 1 else ''} {g.get('mode', 'HOLD_FIRE')}"
                         f" ({', '.join(state) or 'idle'})")
    total = ship_status.get("pd_turrets_total")
    if total:
        op = ship_status.get("pd_turrets_operational", total)
        parts.append(f"PD {op}/{total} turrets{' - DEGRADED' if op < total else ''} (automatic)")
    return "  weapons: " + " | ".join(parts) if parts else ""


def format_my_ship(
    captain_name: str,
    ship_name: str,
    ship_type: Optional[str],
    ship_status: Dict[str, Any],
    tactical_status: Dict[str, Any],
    fleet_data: Optional[Dict[str, Any]],
    decision_interval_s: float = 30.0,
    recent_hits: Optional[str] = None,
) -> str:
    """MY SHIP: identity, world-frame state, condition, weapons, what it is flying."""
    own = tactical_status.get("own") or {}
    sim_time = tactical_status.get("sim_time", 0.0)
    turn = own.get("turn_90_s") or get_ship_turn_time_90deg(ship_type or "destroyer")
    head = f"MY SHIP: {ship_ref(ship_name, own.get('ship_id'))} - {class_label(ship_type)}"
    if own.get("accel_g"):
        head += f", {own['accel_g']:.1f}g"
    head += f", 90° turn ~{turn:.0f} s"
    lines = [head]
    state = []
    if own.get("position_km"):
        state.append(f"position {_vec(own['position_km'], '+.1f')} km")
    if own.get("velocity_kps"):
        state.append(f"velocity {_vec(own['velocity_kps'])} km/s = {own.get('speed_kps', 0.0):.2f} km/s")
    fwd = tactical_status.get("ship_forward")
    if fwd:
        state.append(f"nose {_vec(fwd)}")
    if state:
        lines.append("  " + " | ".join(state))

    dv_total = 500.0
    if fleet_data and ship_type:
        dv_total = float(fleet_data.get("ships", {}).get(ship_type, {})
                         .get("performance", {}).get("delta_v_kps", 500))
    # Clamp: a fresh ship computed 502/500 from the rocket equation.
    dv = min(float(ship_status.get("delta_v_remaining", dv_total)), dv_total)
    lines.append(
        f"  hull {ship_status.get('hull_integrity', 100):.0f}% | armor nose "
        f"{ship_status.get('nose_armor', 0.0):.0f} / lateral {ship_status.get('lateral_armor', 0.0):.0f}"
        f" / tail {ship_status.get('tail_armor', 0.0):.0f} cm | delta-v {dv:.0f}/{dv_total:.0f} km/s")
    lines.extend(_heat_lines(ship_status))
    weapons = _weapons_line(ship_status, ship_has_guns(ship_type, fleet_data),
                            (tactical_status.get("current_config") or {}).get("weapon_orders"))
    if weapons:
        lines.append(weapons)

    cfg = tactical_status.get("current_config") or {}
    m = cfg.get("current_maneuver")
    evasion = tactical_status.get("evasion_status") or {}
    if m:
        fly = f"  flying: {_maneuver_text(m)}"
        if m.get("type") == "EVASIVE" and evasion.get("mode"):
            mode = evasion["mode"]
            fly += f" (mode {mode}{': no threat, weaving around your course - not braking' if mode == 'WOBBLE' else ''})"
        start, dur = m.get("start_time"), m.get("duration")
        if start is not None and dur:
            end = start + dur
            if start >= sim_time - 1e-6:
                fly += f" - applied T+{start:.0f}, runs to T+{end:.0f}"
            elif end <= sim_time + 1e-6:
                fly += f" - issued T+{start:.0f}, EXPIRES NOW: renew it or the ship coasts"
            else:
                fly += f" - issued T+{start:.0f}, runs to T+{end:.0f}"
        lines.append(fly)
    else:
        speed = own.get("speed_kps")
        lines.append("  flying: nothing - COASTING" + (f" at {speed:.2f} km/s" if speed is not None else ""))

    target = None
    pid = tactical_status.get("primary_target_id")
    for e in tactical_status.get("enemies", []):
        if e.get("ship_id") == pid:
            target = ship_ref(e.get("name"), pid)
    lines.append(f"  primary target: {target or 'none (guns and INTERCEPT use the nearest enemy)'}")

    shots = tactical_status.get("our_shots")
    if shots:
        lines.append(f"  record: {shots} shot{'s' if shots != 1 else ''}, "
                     f"{tactical_status.get('our_hits', 0)} hit, dealt "
                     f"{tactical_status.get('our_damage_dealt', 0.0):.1f} GJ, took "
                     f"{tactical_status.get('our_damage_taken', 0.0):.1f} GJ")
    damage = format_damage_report(ship_status.get("damaged_modules") or {})
    if damage:
        lines.append("  " + damage.replace("\n  - ", "\n    - "))
    if recent_hits:
        lines.append("  " + recent_hits.replace("\n", "\n  "))
    return "\n".join(lines)


def _torpedo_threat_line(t: Dict[str, Any], allied_note: bool = True) -> str:
    eta = t.get("eta_seconds", 999)
    eta_str = f"IMPACT IN {eta:.0f}s" if eta < 900 else "not closing"
    head = "BLINDED torpedo" if t.get("blinded") else "torpedo"
    source = t.get("source", "unknown")
    if t.get("blinded") and t.get("own_side"):
        source = f"{source} - YOUR OWN SIDE'S round"
    line = (f"  - {head} from {source}: {t.get('distance_km', 0):.0f} km, "
            f"closing {t.get('closing_kps', 0):.1f} km/s, {eta_str}")
    if t.get("est_impact_gj") is not None:
        line += f", est ~{t['est_impact_gj']:.0f} GJ"
    if t.get("blinded"):
        line += " | ballistic wreck: KEEP MANEUVERING and it misses"
        if t.get("miss_km") is not None:
            line += f" (passes {t['miss_km']:.1f} km from you if you hold course)"
        return line
    nez = t.get("nez_inside")
    if nez is True:
        line += " | NEZ: CANNOT be outrun"
    elif nez is False:
        line += " | NEZ: still escapable - RUN"
    needed = t.get("pd_turrets_needed")
    own = t.get("own_pd_turrets")
    if needed is not None and own is not None:
        allied = t.get("allied_pd_turrets") if allied_note else None
        if allied:
            total = own + allied
            verdict = ("your PD umbrella can blind it - EVADE makes the wreck miss"
                       if needed <= total else "your PD umbrella CANNOT stop it")
            line += (f" | PD kill needs ~{min(needed, 99)} turret(s), you have {own} + allies "
                     f"{allied} bearing: {verdict}")
        else:
            verdict = "your PD can blind it" if needed <= own else "your PD alone CANNOT stop it"
            line += f" | PD kill needs ~{min(needed, 99)} turret(s), you have {own}: {verdict}"
    return line


def _pd_saturation_line(torps: List[Dict[str, Any]]) -> str:
    """
    Several live rounds at once: the per-round PD verdicts above each assume
    every turret dwells on THAT round. Say how many the umbrella can blind.
    """
    live = [t for t in torps if not t.get("blinded") and t.get("pd_turrets_needed")]
    if len(live) < 2:
        return ""
    per_round = max(1, min(int(t["pd_turrets_needed"]) for t in live))
    turrets = max((t.get("own_pd_turrets") or 0) + (t.get("allied_pd_turrets") or 0) for t in live)
    can_blind = min(len(live), turrets // per_round)
    return (f"    SATURATION: {len(live)} live rounds vs {turrets} turrets bearing (~{per_round} per "
            f"round): PD blinds at most {can_blind}, {len(live) - can_blind} arrive with a live "
            "seeker - EVADE now, nose armor toward them.")


def format_threats_to_me(
    tactical_status: Dict[str, Any],
    merge_warnings: Optional[List[str]] = None,
    decision_interval_s: float = 30.0,
) -> str:
    """
    THREATS TO ME: what is on a collision course with this ship, who can put
    something there before the next decision, and any merge warning.
    """
    evasion = tactical_status.get("evasion_status") or {}
    torps = tactical_status.get("torpedo_threats") or []
    rounds = tactical_status.get("incoming_projectiles") or []
    hulls = tactical_status.get("torpedo_hulls_on_me") or []
    guns = tactical_status.get("guns_on_me") or []
    merges = merge_warnings or []
    if not (torps or rounds or hulls or guns or merges or evasion.get("threat_count")):
        return "THREATS TO ME: none - nothing inbound, no enemy has you targeted."

    lines = [f"THREATS TO ME ({format_threat_breakdown(evasion, 'on a collision course') if evasion else 'status unavailable'}):"]
    if torps:
        lines.append(f"  INBOUND ORDNANCE: {len(torps)} torpedo(es):")
        lines.extend("  " + _torpedo_threat_line(t) for t in torps[:6])
        if len(torps) > 6:
            lines.append(f"    ... {len(torps) - 6} more")
        saturation = _pd_saturation_line(torps)
        if saturation:
            lines.append(saturation)
    if rounds:
        first = rounds[0]
        lines.append(
            f"  gun rounds inbound: {len(rounds)} - first {first.get('weapon_type', '?')} from "
            f"{first.get('source', '?')}, {first.get('distance_km', 0):.0f} km, ETA "
            f"{first.get('eta_seconds', 0):.1f} s, {first.get('bearing', '')}".rstrip(", "))
    if hulls:
        etas = [h["launch_tof_s"] for h in hulls if h.get("launch_tof_s") is not None]
        names = ", ".join(
            f"{ship_ref(h.get('name'), h.get('ship_id'))} ({h.get('tubes', 1)} tube"
            f"{'s' if h.get('tubes', 1) != 1 else ''}, {h.get('rounds', '?')} rounds)"
            for h in hulls[:4])
        line = f"  TORPEDO HULLS TARGETING YOU: {names}"
        if etas:
            soon = min(etas)
            line += (f" - a launch now lands in ~{soon:.0f}"
                     f"{'-' + format(max(etas), '.0f') if max(etas) - soon >= 5 else ''} s")
            if soon <= decision_interval_s:
                line += (", BEFORE you decide again. Opening range before they launch is the only "
                         "escape; EVADE (auto-RUN) now.")
            elif soon <= decision_interval_s * 2:
                line += (": after a launch you get at most ONE more decision. Opening range before "
                         "they launch is the only escape; EVADE (auto-RUN) early.")
        lines.append(line)
    if guns:
        lines.append("  GUNS ON YOU: " + "; ".join(
            f"{ship_ref(g.get('name'), g.get('ship_id'))} {g.get('range_km', 0):.0f} km, "
            f"P(hit you) {g.get('p_hit', 0):.0%} x {g.get('shots', 0)} shots/{decision_interval_s:g}s"
            for g in guns[:3]))
    for w in merges:
        lines.append(f"  !! {w}")
    return "\n".join(lines)


def _enemy_threat_text(e: Dict[str, Any]) -> str:
    """His remaining gun output (torpedo rounds are on the armament line)."""
    gun = e.get("gun_gj_per_min")
    return f"guns ~{gun:.0f} GJ/min" if gun else ""


def _group_forecasts(forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identical mounts (same weapon) as one entry: shots and expected hits summed."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for f in forecasts:
        g = grouped.get(f["weapon_name"])
        if g is None:
            grouped[f["weapon_name"]] = dict(f, mounts=1)
            continue
        g["mounts"] += 1
        g["shots"] += f["shots"]
        g["expected_hits"] += f["expected_hits"]
        g["hit_probability"] = max(g["hit_probability"], f["hit_probability"])
        if f.get("expected_gj_per_shot") is not None:
            g["expected_gj_per_shot"] = max(g.get("expected_gj_per_shot") or 0.0,
                                            f["expected_gj_per_shot"])
    return list(grouped.values())


def format_enemy_contacts(
    enemies: List[Dict[str, Any]],
    fleet_data: Optional[Dict[str, Any]] = None,
    has_guns: bool = True,
    has_torpedoes: bool = False,
    decision_interval_s: float = 30.0,
) -> str:
    """
    ENEMIES: one line of live state per contact, then the fire solution that
    matters for THIS hull - a flown torpedo line (TOF, impact energy, PD
    cover, friendly rounds inbound, REACHABLE/...) on launchers, gun
    forecasts on gun ships. Static class data is in the cached CLASS
    REFERENCE.
    """
    if not enemies:
        return "ENEMIES: none detected"
    weapon_types = (fleet_data or {}).get("weapon_types", {})
    lines = [f"ENEMIES ({len(enemies)}; dir_to = world unit vector from you to him - paste it "
             "into set_heading):"]
    for e in enemies:
        sid = e.get("ship_id", "")
        stype = e.get("ship_type")
        cls = e.get("ship_class")
        if not cls or str(cls).lower() in ("unknown", "ship"):
            cls = class_label(stype) if stype else "Ship"
        head = "  [PRIMARY] " if e.get("is_primary_target") else "  "
        line = (f"{head}{ship_ref(e.get('name'), sid)} {cls} | {e.get('distance_km', 0):.0f} km, "
                f"{_rate(e.get('closing_rate', 0.0))}")
        if e.get("dir_to"):
            line += f" | dir_to {_vec(e['dir_to'])}"
        angle = e.get("angle_deg")
        if angle is not None:
            words = _body_words(e.get("relative_bearing_km"))
            line += f", {angle:.0f}° off your nose" + (f" ({words})" if words else "")
        armor = e.get("armor") or {}
        nose = assess_armor_condition(armor.get("nose_damage_pct", 0))
        flank = assess_armor_condition(armor.get("lateral_damage_pct", 0))
        cond = "armor intact" if nose == flank == "Intact" else f"nose {nose}, flank {flank}"
        line += f" | hull {e.get('hull_percent', 100):.0f}%, {cond}"
        spec = (fleet_data or {}).get("ships", {}).get(stype or "", {})
        if spec:
            line += " | " + format_enemy_weapons_summary(
                spec.get("weapons", []), weapon_types,
                torpedoes_remaining=e.get("torpedoes_remaining"),
                pd_operational=e.get("pd_operational"))
        threat = _enemy_threat_text(e)
        if threat:
            line += f" | {threat}"
        if e.get("has_us_targeted"):
            line += " | !! HAS YOU TARGETED"
        lines.append(line)
        if has_torpedoes:
            sol = e.get("torpedo_solution")
            if sol:
                lines.append(f"    torpedo: {sol}")
        if has_guns:
            forecasts = _group_forecasts(e.get("weapon_hit_forecasts") or [])
            if forecasts:
                lines.append("    guns: " + " | ".join(
                    f"{f['weapon_name']}{' x' + str(f['mounts']) if f['mounts'] > 1 else ''} "
                    f"P(hit) {f['hit_probability']:.0%}, {f['shots']} shots/{f['window_s']:g}s = "
                    f"{f['expected_hits']:.2f} hits"
                    + (f", ~{f['expected_gj_per_shot']:.2f} GJ/shot" if f.get("expected_gj_per_shot")
                       is not None else "")
                    for f in forecasts))
    if has_guns and any(e.get("weapon_hit_forecasts") for e in enemies):
        lines.append("  (guns: P(hit) is the fire-control gate per shot now; shots counted for the next "
                     f"{decision_interval_s:g} s incl. salvo cooldown, if aim and evasion persist)")
    return "\n".join(lines)


def format_friendlies(
    friendlies: List[Dict[str, Any]],
    umbrella: Optional[Dict[str, Any]] = None,
) -> str:
    """FRIENDLIES: who is where, doing what, at whom - and the PD over you."""
    if not friendlies:
        return "FRIENDLIES: none"
    head = "FRIENDLIES"
    if umbrella:
        allies = umbrella.get("allies") or []
        total = umbrella.get("own", 0) + sum(n for _, n in allies)
        head += (f" (PD umbrella over you: {total} turrets within {umbrella.get('range_km', 250):.0f} km"
                 f" - yours {umbrella.get('own', 0)} + {len(allies)} "
                 f"{'ship' if len(allies) == 1 else 'ships'})")
    lines = [head + ":"]
    for f in friendlies:
        line = f"  {ship_ref(f.get('name'), f.get('ship_id'))}"
        if f.get("ship_type"):
            line += f" {class_label(f['ship_type'])}"
        line += f" | {f.get('distance_km', 0):.0f} km"
        if f.get("dir_to"):
            line += f" dir_to {_vec(f['dir_to'])}"
        elif f.get("relative_bearing_km"):
            b = f["relative_bearing_km"]
            line += (f" ({abs(b.get('forward', 0)):.0f} km {'ahead' if b.get('forward', 0) >= 0 else 'behind'}, "
                     f"{abs(b.get('starboard', 0)):.0f} km {'starboard' if b.get('starboard', 0) >= 0 else 'port'})")
        elif f.get("relative_position"):
            line += f" (world frame {_vec(f['relative_position'], '+.0f')} km)"
        if f.get("rel_speed_kps") is not None:
            line += f" | rel {f['rel_speed_kps']:.2f} km/s"
        if "maneuver" in f:
            line += f" | {_maneuver_text(f.get('maneuver'))}"
            if f.get("target_name"):
                line += f" -> {ship_ref(f['target_name'], f.get('target_id'))}"
        if f.get("torpedoes_remaining") is not None:
            line += f" | torps {f['torpedoes_remaining']}"
            if f.get("rounds_in_flight"):
                line += f", {f['rounds_in_flight']} in flight"
                if f.get("first_impact_s") is not None:
                    line += f" (first lands ~{f['first_impact_s']:.0f} s)"
        if f.get("pd_total"):
            line += f" | PD {f.get('pd_operational', f['pd_total'])}/{f['pd_total']}"
        line += f" | hull {f.get('hull_percent', 100):.0f}%"
        lines.append(line)
    return "\n".join(lines)


def _comms_line(item: Dict[str, Any]) -> str:
    role = item.get("sender_role")
    name = item.get("sender_name") or item.get("sender_id") or "?"
    if role == "engine":
        sender = "ENGINE"
    elif role == "admiral":
        sender = f"Admiral {strip_captain_title(name)}" if not str(name).startswith("Admiral") else name
    else:
        sender = ship_ref(name, item.get("sender_id"))
    channel = item.get("channel")
    to = {"direct": " -> you", "admiral_to_captain": " -> you", "fleet_net": " -> fleet"}.get(channel, "")
    text = " ".join(str(item.get("text", "")).split())
    if len(text) > COMMS_ITEM_MAX_CHARS:
        text = text[:COMMS_ITEM_MAX_CHARS - 3] + "..."
    t = item.get("sent_t", item.get("delivered_t"))
    when = f"T+{float(t):.0f} " if isinstance(t, (int, float)) else ""
    return f"  {when}{sender}{to}: \"{text}\""


def is_friendly_comms(item: Dict[str, Any], own_faction: Optional[str] = None) -> bool:
    """FLEET NET vs ENEMY TRANSMISSIONS for one comms_message dict."""
    channel = item.get("channel")
    if item.get("sender_role") == "engine" or channel in FRIENDLY_CHANNELS:
        return True
    if channel == "enemy_broadcast":
        return False
    if own_faction and item.get("faction"):
        return item["faction"] == own_faction
    return False


def format_comms(
    items: Optional[List[Dict[str, Any]]],
    own_faction: Optional[str] = None,
) -> str:
    """
    FLEET NET (own fleet, admiral and engine signals) and ENEMY
    TRANSMISSIONS, as separate sections. Everything used to arrive under
    "ENEMY TRANSMISSION:", so a wingman's "taking Lancer-3" read as enemy
    chatter and captains stopped coordinating.
    """
    items = list(items or [])
    if not items:
        return ""
    friendly = [i for i in items if is_friendly_comms(i, own_faction)]
    enemy = [i for i in items if not is_friendly_comms(i, own_faction)]
    out = []
    if friendly:
        lines = ["FLEET NET (your fleet, your admiral and engine signals):"]
        lines += [_comms_line(i) for i in friendly[:FLEET_NET_MAX_ITEMS]]
        if len(friendly) > FLEET_NET_MAX_ITEMS:
            lines.append(f"  ... {len(friendly) - FLEET_NET_MAX_ITEMS} more")
        out.append("\n".join(lines))
    if enemy:
        lines = ["ENEMY TRANSMISSIONS (open channel - the enemy is talking):"]
        lines += [_comms_line(i) for i in enemy[:ENEMY_TRANSMISSIONS_MAX_ITEMS]]
        if len(enemy) > ENEMY_TRANSMISSIONS_MAX_ITEMS:
            lines.append(f"  ... {len(enemy) - ENEMY_TRANSMISSIONS_MAX_ITEMS} more")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def build_captain_turn(
    captain_name: str,
    ship_name: str,
    ship_status: Dict[str, Any],
    tactical_status: Dict[str, Any],
    ship_type: Optional[str] = None,
    fleet_data: Optional[Dict[str, Any]] = None,
    decision_interval_s: float = 30.0,
    received_messages: Optional[str] = None,
    decision_history: Optional[str] = None,
    message_history: Optional[str] = None,
    battle_summary: Optional[str] = None,
    shot_history: Optional[str] = None,
    recent_hits: Optional[str] = None,
    decision_number: Optional[int] = None,
    lost_turn: Optional[str] = None,
    range_control: Optional[str] = None,
    merge_warnings: Optional[List[str]] = None,
    orders: Optional[str] = None,
    comms: Optional[List[Dict[str, Any]]] = None,
    own_faction: Optional[str] = None,
    captain_log: Optional[str] = None,
    execution_feedback: Optional[str] = None,
    tool_errors: Optional[str] = None,
    tool_names: Optional[Iterable[str]] = None,
    **_ignored: Any,
) -> str:
    """
    The volatile per-checkpoint turn, in the order a captain needs it:
    who/when, MY SHIP, THREATS TO ME, ENEMIES (+ fire solutions), FRIENDLIES,
    RANGE CONTROL, ORDERS, FLEET NET / ENEMY TRANSMISSIONS, then history,
    log and engine feedback. Every section is optional so test doubles and
    1v1 duels render cleanly.
    """
    sim_time = tactical_status.get("sim_time", 0.0)
    has_guns = ship_has_guns(ship_type, fleet_data)
    names = set(tool_names or ())
    has_torps = ("launch_torpedo" in names) or (ship_status.get("torpedoes_remaining") is not None)
    sections = []
    header = [f"{CAPTAIN_TURN_MARKER}{sim_time:.0f}s) ==="]
    who = strip_captain_title(captain_name)
    header.append(
        f"You are Captain {who}, commanding {ship_name}."
        + (f" Decision {decision_number}:" if decision_number else "")
        + f" your orders lock for {decision_interval_s:g} s; next decision T+{sim_time + decision_interval_s:.0f}.")
    sections.append("\n".join(header))
    if lost_turn:
        sections.append(lost_turn)
    sections.append(format_my_ship(captain_name, ship_name, ship_type, ship_status, tactical_status,
                                   fleet_data, decision_interval_s, recent_hits))
    sections.append(format_threats_to_me(tactical_status, merge_warnings, decision_interval_s))
    sections.append(format_enemy_contacts(tactical_status.get("enemies", []), fleet_data,
                                          has_guns=has_guns, has_torpedoes=has_torps,
                                          decision_interval_s=decision_interval_s))
    sections.append(format_friendlies(tactical_status.get("friendlies", []),
                                      tactical_status.get("pd_umbrella")))
    if range_control:
        sections.append(range_control)
    if orders:
        sections.append(orders)
    comms_text = format_comms(comms, own_faction)
    if comms_text:
        sections.append(comms_text)
    if received_messages:
        # Legacy free-text form: the sender's side is unknown here, so it is
        # NOT labelled as enemy traffic.
        sections.append(f"MESSAGES RECEIVED:\n{received_messages}")
    history = [h for h in (battle_summary, shot_history if has_guns else None,
                           decision_history, captain_log, message_history) if h]
    if history:
        sections.append("\n\n".join(history))
    if execution_feedback:
        sections.append(execution_feedback)
    if tool_errors:
        sections.append(tool_errors)
    actions = ["set_maneuver/set_heading"]
    if has_guns:
        actions.append("set_weapons_order")
    if has_torps:
        actions.append("launch_torpedo")
    actions.append("set_primary_target")
    sections.append(
        f"DECISION POINT {decision_number or 1}. What are your orders, Captain? Issue them as tool "
        f"calls ({', '.join(actions)}); words alone do nothing.")
    return "\n\n".join(s for s in sections if s)


def build_captain_prompt(**kwargs: Any) -> str:
    """
    Complete captain prompt as one string: cached doctrine + volatile turn.

    Accepts the build_captain_turn / build_captain_doctrine keyword
    arguments (captain_name, ship_name, ship_status, tactical_status,
    personality, personality_text, ship_type, fleet_data, notebook_text,
    decision_interval_s, history strings, orders, comms, ...).
    """
    system, turn = _captain_parts(**kwargs)
    return system + "\n\n" + turn


def _captain_parts(**kwargs: Any) -> Tuple[str, str]:
    kwargs.setdefault("ship_status", {})
    kwargs.setdefault("tactical_status", {})
    kwargs.setdefault("captain_name", "Captain")
    kwargs.setdefault("ship_name", "Ship")
    system = build_captain_doctrine(**kwargs).strip()
    turn = build_captain_turn(**kwargs).strip()
    return system, turn


def build_captain_messages(**kwargs: Any) -> List[Dict[str, str]]:
    """
    The captain request as a cacheable system prompt plus a volatile user turn.

    The system prompt is the doctrine alone (no name, no per-checkpoint
    state), byte-identical across checkpoints so the provider serves it from
    cache; everything that changes lives in the user turn.

    Accepts the same keyword arguments as build_captain_prompt().
    """
    system, turn = _captain_parts(**kwargs)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": turn},
    ]


# =============================================================================
# ADMIRAL PROMPTS
# =============================================================================
#
# An admiral request is two messages (build_admiral_messages):
#   system - battle-STATIC text: doctrine composed for the fleets actually in
#            this battle (gun doctrine only for gun fleets, torpedo doctrine
#            only where launchers exist) plus a per-class FORCES table. Byte-
#            stable across checkpoints, so it is a cacheable prefix.
#   user   - the live turn state (format_admiral_turn_state): ranked blocks,
#            one line per item, decisions before reference material.
# Audit references: ADM-01 (ordnance), ADM-04 (geometry), ADM-06 (digest),
# ADM-07 (layout), ADM-10 (doctrine), ADM-11 (caching), ADM-13 (labels).

# Marker separating the stable doctrine from the per-checkpoint fleet snapshot.
# Everything before it is reused verbatim across checkpoints of the same battle,
# so it can serve as a cacheable prefix; everything after it changes every call.
ADMIRAL_STATE_MARKER = "=== FLEET SITUATION ==="

# Stable across the battle -> cacheable prefix. Must precede the live snapshot.
ADMIRAL_DOCTRINE_CORE = """
You are {admiral_name}, Admiral of the {faction} fleet in a space combat simulation.

{mission_block}

{simulation_disclaimer}

=== YOUR ROLE ===

As Admiral, you:
1. Set the overall fleet directive (visible to all captains)
2. Issue SPECIFIC ORDERS to EACH captain using the issue_order tool
3. Can negotiate with enemy Admiral (if one exists)
4. Control draw proposals for your fleet
5. Keep a STANDING BATTLE PLAN (set_battle_plan): your private cross-checkpoint
   memory. Everything else you see is rebuilt from scratch each checkpoint - the
   plan is the ONLY record of your own intent that survives. Set it early
   (phases + trigger conditions), then each checkpoint either execute the next
   phase or amend it. Directive = this checkpoint's order to captains; standing
   plan = the whole battle's shape, for you alone.

Your captains receive your orders BEFORE they decide. Each files a report
(ACK / DEVIATE / UNABLE, plus what it actually flew and fired) that you see at the
next checkpoint under CAPTAIN REPORTS. They can discuss with you (up to 2
exchanges) before finalizing.

{order_requirements}
=== HOW TO READ THE FLEET SITUATION ===
- "!! NOW" is what needs action THIS checkpoint. Read it first.
- Ranges and closures are measured from YOUR FLEET CENTRE unless a line says
  otherwise; + closing, - opening. Every ship is NAME [ID]; either works in tools.
- ORDNANCE "INBOUND ON YOU" lists only rounds that can hit your hulls: enemy rounds
  homing on them, and any dead (seeker-blinded or dry) torpedo whose straight line
  passes within {wreck_miss_km:.1f} km of one of your hulls in the next
  {wreck_horizon_s:.0f} s (the engine's own collision test). Every other dead round
  is spent - it is not a reason to EVADE.
{verdict_legend}
=== TACTICAL COMMAND REFERENCE ===

MANEUVERS (tell captains which to use):
- INTERCEPT: burn straight at the target - only for the first approach or finishing a cripple
- EVADE: Threat-aware defense - jinks vs guns, RUNS/PRESENTS vs torpedoes. With no threat
  it weaves around the ship's CURRENT course: it never chases and never brakes, so an
  EVADE order does not stop a merge.
- BRAKE: burn against the ship's own (world-frame) velocity
- MAINTAIN: Coast at current velocity - no thrust, no rotation
- PADLOCK: coast with the nose tracking the target (spinal on him), NO closing thrust
- HEADING (issue_order heading={{x,y,z}}, throttle): fly a WORLD-frame direction - the ship
  turns onto it, then burns (the drive thrusts along the NOSE). throttle 0 = attitude only
  (nose turns, no burn, e.g. spinals back on the enemy). Applied directly to the ship.

=== RANGE CONTROL (Newtonian - your most important lever) ===
- No brakes, only burns. Stopping a closure takes v^2/(2a) of room plus the flip: at
  4.6 km/s closure a 3g corvette needs 156 s / 359 km (+18 s flip), a 1g battleship
  469 s / 1078 km (+55 s). Fleets that "look close" are already committed to a merge -
  both sides fly through each other in seconds and nothing meaningful happens after.
- INTERCEPT only for the first approach or finishing a cripple. Use HEADING for
  everything involving range: approach angle, holding range, kiting, avoiding merges,
  flanking. HEADINGS in the fleet situation lists ready-made world-frame headings
  per enemy group (cancel closure / hold range, open range, re-engage after a pass,
  pass wide, flank left/right) with the burn time, the effect, and which way the NOSE
  points meanwhile. Renew each checkpoint until the stated burn time is used up.
  "!! MERGE IN ..." = act THIS checkpoint.
- FORMATION: one heading for the whole fleet = issue_order(ship_name="all ships",
  heading=..., throttle=...) on the directive call. Each hull's throttle is scaled to the
  SLOWEST hull's acceleration and the burn starts when the slowest has turned, so the
  formation (and its overlapping PD) holds. Per-ship orders inherit it unless they set
  their own heading. Same throttle on different hulls is NOT the same acceleration:
  a 2g destroyer at 70% still out-runs a 0.75g dreadnought at 70%.
- RHYTHM: burn phases and shooting phases alternate. Burning for range points the nose
  along the heading (cancel closure = TAILS to the enemy: thin tail armor, spinals
  masked). Then coast and shoot: PADLOCK or HEADING toward the enemy at throttle 0 puts
  nose armor and spinals back on him.
{pd_doctrine}{torpedo_doctrine}{gun_doctrine}
HEAT AND RESOURCES:
- Radiators EXTENDED = better cooling, but exposed panels can be shot off permanently.
- PD is cheap: ~{pd_heat_mw:.0f} MW hull heat per firing turret, no thrust loss.
- HEAT LADDER (enforced; heat can pass 100%): >= {heat_hold_pct:.0f}% a ship's GUNS HOLD FIRE
  (PD keeps firing); >= {heat_forced_pct:.0f}% its radiators are FORCED OUT (exposed to fire);
  >= {heat_meltdown_pct:.0f}% REACTOR MELTDOWN - the ship is destroyed. A full burn makes exactly
  the radiators' capacity in heat ({heat_drive_mw:.0f} MW): burning hard retracted fills a sink in
  ~{heat_fill_min:.0f} min and gunfire/PD heat comes on top. YOUR FLEET shows heat and time to
  100% for every ship that is warming; tell hot ships to ease the burn, extend radiators or
  hold fire (set_radiators mode="auto" lets a captain's ship manage its radiators itself).
- {resources_line}

ADDITIONAL NOTES:
- Captains are AI commanders - be specific about priorities and targets
- GEOMETRY shows each group's range change over the last 15 s, who has done the
  closing (each side's speed along the line in the battle frame, where the fleets
  started) and which way they burn now; SINCE LAST CHECKPOINT shows kills, losses
  and who hit whom
- TIMING: Each order is LOCKED for {decision_interval_s:g} seconds. No "if X then Y" contingencies.
  Give ONE clear order per ship. Captains cannot react mid-checkpoint.
{enemy_admiral_note}

{personality_prompt}
"""

# Included when torpedoes are in the battle (either side): how PD meets them.
ADMIRAL_PD_DOCTRINE = """
=== TORPEDO WARFARE & POINT DEFENSE (measured on this engine - trust these) ===

HOW POINT DEFENSE ACTUALLY WORKS:
- All PD is automatic. Turrets triage savable torpedoes first (own ship, then allies),
  so overlapping coverage genuinely stacks: every friendly turret within its envelope
  (~{pd_range_km:.0f} km laser range) of a torpedo's path adds dwell against it.
- PD BLINDS torpedoes (seeker kill); the dead round then coasts in a straight line and
  hits only a hull whose path that line crosses - one flying straight at it. Measured:
  18/18 blinded rounds hit a non-evading battleship; the same attack on an evading one
  behind 8 escorts landed 6/18. So a ship with torpedoes INBOUND (see ORDNANCE) must be
  ordered to EVADE, or every PD kill is wasted; a dead round on nobody's line needs nothing.
- DWELL IS SET BY THE TORPEDO'S CLOSING SPEED, NOT BY FLEET-TO-FLEET CLOSURE. Guidance
  holds a fuelled round at >={pd_closure_floor_kps:.0f} km/s however slowly the fleets
  approach, so it crosses the envelope in ~{pd_envelope_crossing_s:.0f}s and eats only
  ~{pd_mj_per_turret:.0f} MJ per turret against a {pd_seeker_kill_mj:.0f} MJ seeker
  kill: {pd_turrets_to_blind} turrets must bear on ONE round to blind it (impacts land
  at 13-28 km/s). A 1-2 turret hull never blinds a normally launched torpedo alone -
  massing turrets is the ONLY lever you have.
- PD cost is small (see HEAT: no thrust loss). Never restructure the fleet to "save" PD
  capacity.

FORMATIONS FOR PD (what actually moves the numbers):
- TURRET COUNT on the threat corridor is everything; ring radius is not - escort
  rings at 15, 30 and 120 km measured identically. Place escorts anywhere inside
  laser range of the axis the torpedoes come down; don't micro-manage spacing.
- Returns need mass: vs a 6-round simultaneous salvo, an evading battleship leaked
  17/18 alone, 15/18 with 4 destroyer escorts, 6/18 with 8. Fewer than ~4 escorts is
  symbolic against a real salvo - concentrate the fleet or accept the hits.
- The wall only protects ships that EVADE inside it (see blinding, above).
"""

# Included when YOUR fleet mounts launchers.
ADMIRAL_TORPEDO_DOCTRINE = """
WHEN TO COMMIT TRIDENTS:
- Fire when CLOSING. A closing launch vs a lone ship connects ~100% (12g guidance,
  <=3g targets): ~23 GJ per round at low closure, ~95 GJ launched head-on at 26 km/s
  (a spinal gun round is 4.3 GJ). NEVER launch at a receding evading ship - it
  outruns the round's delta-v and the magazine is wasted. The THREAT TABLE's
  torpedo verdict does this math for you: launch on GOOD.
- SIMULTANEOUS time-on-target, always. Rounds arriving together split PD dwell;
  trickled rounds are blinded one at a time, early enough to dodge. Measured vs 4
  turrets: 4 simultaneous -> 24/24 through (99 GJ); same 4 spaced 30s -> 12/24 (42 GJ).
  Use torpedo_salvo on issue_order to synchronize launches ACROSS ships.
- SIZING at moderate closure (~22 GJ/round, evading targets): 4 rounds kill a
  destroyer; 6 rounds kill anything up to a dreadnought; 1-2 rounds wound only.
  Against an escort wall, add the rounds the PD around the target soaks (TORPEDO
  ECONOMY works it out per target: turrets in cover / {pd_turrets_to_blind} per round).
  Do not pile more rounds onto a hull that already has enough inbound.
- RANGE: any closing launch 200-2400 km connects; longer = harder impact but minutes
  in flight (~400 km lands in 30-60s - about one decision cycle).
- MAGAZINES ARE FINITE: {magazine_line}. Reload {reload_s:.0f}s -> at most
  {launches_per_tube} launches per tube per {decision_interval_s:g}s checkpoint.
"""

# Included when YOUR fleet mounts guns.
ADMIRAL_GUN_DOCTRINE = """
=== FLEET TACTICAL OPTIONS ===

You can command your fleet however you want. Consider these strategies:

OPTION A - MASS CHARGE:
  All ships INTERCEPT same target -> overwhelming force -> high-speed pass
  Pros: Simple, focused firepower, intimidating
  Cons: Brief engagement window, ships scatter after pass

OPTION B - CONTROLLED FORMATION PASSES (recommended):
  1. Approach as a formation ("all ships" heading, or INTERCEPT from far out)
  2. Keep ships together for overlapping point defense coverage
  3. Before you are committed: cancel closure (or pass wide) as a formation
  4. Coast and shoot: PADLOCK - spinals and nose armor on target during the pass
  Pros: Better accuracy, mutual PD support, sustained damage
  Cons: Requires coordination, takes longer

OPTION C - HAMMER AND ANVIL:
  Heavy ships hold range (cancel-closure HEADING), lighter ships flank (flank HEADING)
  Pin enemy between groups, crossfire from multiple angles
  Pros: Divides enemy attention, creates flanking opportunities
  Cons: Complex coordination, risks defeat in detail if separated

OPTION D - FOCUS FIRE ROTATION:
  All ships focus one target until destroyed, then switch
  Pros: Rapid kills, reduces incoming fire
  Cons: Other enemies fire freely while ignored

GUNNERY:
- High closing velocity = harder to hit (brief engagement window); 0 km/s relative and
  no evasion = sitting duck. Sweet spot: 1-3 km/s relative with active maneuvering.
- Spinal weapons need nose-on alignment (<30 deg to target); turrets fire over a 180 deg arc.
- Cross-section: nose-on = tiny target, lateral = 5-10x easier to hit. Keep enemies in
  FRONT of your ships; if one flanks, reorient the affected ships (PADLOCK toward it).
- Weapon modes: FIRE_IMMEDIATE (close range), FIRE_WHEN_OPTIMAL (recommended),
  FIRE_AT_RANGE (harassing fire), HOLD_FIRE (heat management).
- Target priority: CLOSING targets first (committed); a SEPARATING ship (5+ km/s) is
  wasted fire. Capital ships turn slowly (easier hits); escorts die quickly.
"""

ADMIRAL_DOCTRINE = (ADMIRAL_DOCTRINE_CORE + ADMIRAL_PD_DOCTRINE
                    + ADMIRAL_TORPEDO_DOCTRINE + ADMIRAL_GUN_DOCTRINE)

# Rebuilt every checkpoint; follows the doctrine so the doctrine can be a
# stable cached prefix. Its blocks are rendered by format_admiral_turn_state.
ADMIRAL_TURN_STATE = ADMIRAL_STATE_MARKER + " {header}\n\n{body}\n"

# Backwards-compatible single-string form (doctrine + live state).
ADMIRAL_SYSTEM_PROMPT = ADMIRAL_DOCTRINE + "\n" + ADMIRAL_TURN_STATE

ADMIRAL_RESPONSE_PROMPT = """
You are {admiral_name}, responding to a question from one of your ship captains.

The captain of {captain_ship_name} is asking you a question during battle.
Be concise and decisive - there's no time for lengthy discussion.

IMPORTANT: This is a ONE-WAY response. The captain will act immediately after receiving
your answer. DO NOT ask follow-up questions - they cannot respond. Give clear, actionable
orders or clarifications. End with a command, not a question.

TIMING CONSTRAINT: The captain makes ONE decision that is LOCKED for {decision_interval_s:g} seconds.
They CANNOT react mid-checkpoint or execute conditional orders like "if X then Y".
Give them ONE clear order to execute NOW, not contingency plans.

{standing_plan_section}{situation_section}=== YOUR RECENT ORDERS (REMEMBER THESE!) ===
{recent_orders_context}

{personality_prompt}
"""

# Per-ship order calls: battle-static system part.
ADMIRAL_SHIP_ORDER_DOCTRINE = """You are {admiral_name}, Admiral of the {faction} fleet in a space combat simulation.

THIS CALL: ONE order for ONE ship (named in the request) - call issue_order for it with:
- ship_name: the ship named in the request
- order_text: specific tactical instructions - maneuver, throttle, target, weapon use
- priority: CRITICAL, HIGH, NORMAL, or LOW
- suggested_target: (optional) recommended enemy ship (name or id)
- heading + throttle: (optional) a WORLD-frame {{x,y,z}} from RANGE CONTROL, applied to the
  ship at once (throttle 0 = attitude only: nose onto the heading, no burn)
- torpedo_salvo + torpedo_target: (torpedo hulls) rounds to launch THIS checkpoint,
  timed with the rest of the fleet

Your order should:
1. Be consistent with your fleet directive and standing plan
2. Specify a maneuver (HEADING, PADLOCK, EVASIVE, INTERCEPT)
3. Specify throttle level (e.g., "70% throttle")
4. Specify target assignment
5. Include any special weapon instructions

*** RANGE IS CONTROLLED WITH HEADINGS ***
- INTERCEPT only for the first approach or finishing a cripple - it builds closure that
  ends in a merge. To hold range, kite, pass wide or flank: a HEADING from RANGE CONTROL.
- PADLOCK = nose on target, NO thrust: the coast-and-shoot phase between burns.
- EVASIVE holds the current course (dodges fire, never brakes, never chases).
- Everything in the request is measured from THIS ship (ranges, closures, torpedo
  verdicts, inbound rounds){torpedo_note}
{personality_line}"""


# -----------------------------------------------------------------------------
# Battle-static parts
# -----------------------------------------------------------------------------

def _battle_flags(battle_classes: Dict[str, List[str]], fleet_data: Dict[str, Any]) -> Dict[str, bool]:
    """Which doctrine applies: guns/launchers on each side of THIS battle."""
    try:
        from .fleet_intel import class_armament
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import class_armament
    flags = {"own_guns": False, "own_tubes": False, "enemy_guns": False, "enemy_tubes": False}
    for side, prefix in (("ours", "own"), ("theirs", "enemy")):
        for ship_type in battle_classes.get(side) or []:
            arm = class_armament(ship_type, fleet_data)
            if arm["launchers"]:
                flags[f"{prefix}_tubes"] = True
            if arm["gun_range_km"] > 0:
                flags[f"{prefix}_guns"] = True
    return flags


def _classes_from_snapshot(snapshot: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"ours": [], "theirs": []}
    if snapshot is None:
        return out
    for side, rows in (("ours", getattr(snapshot, "friendly_ships", None) or []),
                       ("theirs", getattr(snapshot, "enemy_ships", None) or [])):
        for row in rows:
            if row.ship_type and row.ship_type not in out[side]:
                out[side].append(row.ship_type)
    return out


def _class_line(ship_type: str, fleet_data: Dict[str, Any]) -> str:
    """One static capability line per ship class (the FORCES table)."""
    try:
        from .fleet_intel import class_abbr
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import class_abbr
    spec = (fleet_data or {}).get("ships", {}).get(ship_type, {})
    perf = spec.get("performance", {})
    att = spec.get("attitude_control", {})
    tv = (att.get("thrust_vectoring") or {}).get("time_to_rotate_90_deg_s") or get_ship_turn_time_90deg(ship_type)
    rcs = (att.get("rcs") or {}).get("time_to_rotate_90_deg_s")
    wtypes = (fleet_data or {}).get("weapon_types", {})
    counts: Dict[str, int] = {}
    pd = tubes = rounds = 0
    for w in spec.get("weapons", []):
        wtype = w.get("type", "")
        ws = wtypes.get(wtype, {})
        if wtype.startswith("pd_"):
            pd += 1
            continue
        if wtype.startswith("torpedo"):
            tubes += 1
            rounds += int(w.get("magazine", ws.get("magazine", 8)))
            continue
        gj = ws.get("kinetic_energy_gj")
        key = (f"{ws.get('name', wtype)} ({gj:.1f} GJ, {ws.get('range_km', 0):.0f} km)"
               if gj else str(ws.get("name", wtype)))
        counts[key] = counts.get(key, 0) + 1
    arms = [f"{n}x {k}" for k, n in counts.items()]
    if tubes:
        arms.append(f"{tubes}x torpedo tube ({rounds} rounds)")
    armor = armor_from_fleet_data(ship_type, fleet_data)
    turn = f"90 deg turn {tv:.0f} s TV" + (f" / {rcs:.0f} s RCS" if rcs else "")
    return (f"- {ship_type.replace('_', ' ').title()} ({class_abbr(ship_type)}): "
            f"{perf.get('combat_acceleration_g', '?')}g, {turn}, "
            f"{_num(perf.get('delta_v_kps', '?'))} km/s dv | {' + '.join(arms) or 'unarmed'} | {pd} PD | "
            f"armor nose/side/tail {armor['nose']:.0f}/{armor['lateral']:.0f}/{armor['tail']:.0f} cm")


def _num(value: Any) -> str:
    """500.0 -> '500', 0.75 -> '0.75', '?' -> '?'."""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _format_forces(battle_classes: Dict[str, List[str]], fleet_data: Dict[str, Any]) -> str:
    """Static per-class capabilities of both fleets (cached with the doctrine)."""
    if not (battle_classes.get("ours") or battle_classes.get("theirs")):
        return ""
    lines = ["=== FORCES (ship classes in this battle - fixed; live numbers are in the FLEET SITUATION) ==="]
    for side, title in (("ours", "YOURS"), ("theirs", "THEIRS")):
        types = battle_classes.get(side) or []
        if types:
            lines.append(f"{title}:")
            lines.extend(_class_line(t, fleet_data) for t in types)
    return "\n".join(lines)


def _magazine_line(own_classes: List[str], fleet_data: Dict[str, Any]) -> str:
    try:
        from .fleet_intel import class_armament
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import class_armament
    parts = []
    for ship_type in own_classes:
        arm = class_armament(ship_type, fleet_data)
        if arm["launchers"]:
            per = arm["magazine"] // max(1, arm["launchers"])
            parts.append(f"{ship_type.replace('_', ' ')} {arm['magazine']} rounds "
                         f"({arm['launchers']} tube{'s' if arm['launchers'] > 1 else ''} x {per})")
    return ("; ".join(parts) + " - TORPEDO ECONOMY shows what is left") if parts else ""


def torpedo_launches_per_checkpoint(fleet_data: Dict[str, Any], decision_interval_s: float) -> int:
    """Launches one ready tube can make in a checkpoint (engine: ceil(window / reload))."""
    reload_s = float(((fleet_data or {}).get("weapon_types") or {})
                     .get("torpedo_launcher", {}).get("cooldown_s", 12))
    return max(1, math.ceil(decision_interval_s / reload_s)) if reload_s > 0 else 1


def _resources_line(flags: Dict[str, bool], fleet_data: Dict[str, Any],
                    own_classes: List[str]) -> str:
    """Derived resource facts (the old 'ammo is ABUNDANT' line was wrong for torpedoes)."""
    wtypes = (fleet_data or {}).get("weapon_types", {})
    gun_mags = []
    for ship_type in own_classes:
        for w in (fleet_data or {}).get("ships", {}).get(ship_type, {}).get("weapons", []):
            wtype = w.get("type", "")
            if not wtype.startswith(("pd_", "torpedo")):
                gun_mags.append(int(w.get("magazine", wtypes.get(wtype, {}).get("magazine", 0)) or 0))
    parts = ["RESOURCES: ~500 km/s delta-v per hull - maneuver freely, fuel is not the limit"]
    if gun_mags:
        parts.append(f"gun magazines {min(gun_mags)}+ rounds per mount (deep)")
    if flags.get("own_tubes"):
        parts.append("TORPEDOES ARE SCARCE - every round must count (see TORPEDO ECONOMY)")
    return "; ".join(parts) + "."


def _verdict_legend(flags: Dict[str, bool]) -> str:
    if not flags.get("own_tubes"):
        return ""
    try:
        from .fleet_intel import GOOD_ENERGY_GJ, NO_ENERGY_GJ
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import GOOD_ENERGY_GJ, NO_ENERGY_GJ
    return (
        "- TORPEDO verdicts assume the target RUNS (full burn away for the whole flight, as\n"
        f"  EVADE does): GOOD = reaches him and lands >= {GOOD_ENERGY_GJ:.0f} GJ; POOR = reaches,\n"
        "  but he is receding or it arrives slow - long PD dwell, a fraction of the damage;\n"
        f"  NO = runs dry short, lands < {NO_ENERGY_GJ:.0f} GJ, or his PD blinds more slow rounds\n"
        "  than you can launch. Launch on GOOD; POOR only to empty a doomed magazine.\n"
        "  A round needs distance to build speed: launched close, it arrives slow whatever\n"
        "  the closure - the verdict already counts that.\n")


def build_admiral_static(
    admiral_name: str,
    faction: str,
    fleet_data: Dict[str, Any],
    personality: Optional[str],
    phase: str = "full",
    enemy_has_admiral: bool = False,
    notebook_text: Optional[str] = None,
    decision_interval_s: float = 30.0,
    battle_classes: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    The battle-static system prompt: doctrine for the fleets in this battle
    plus the FORCES table. Depends only on per-battle inputs, so it is
    byte-identical across checkpoints (cacheable).
    """
    battle_classes = battle_classes or {"ours": [], "theirs": []}
    flags = _battle_flags(battle_classes, fleet_data)
    known = bool(battle_classes.get("ours") or battle_classes.get("theirs"))
    # Without class info (hand-built calls) keep every doctrine part.
    use_guns = flags["own_guns"] or not known
    use_tubes = flags["own_tubes"] or not known
    use_pd = flags["own_tubes"] or flags["enemy_tubes"] or not known

    directive_only = (phase == "directive")
    if directive_only:
        mission_block = """***** THIS CALL: FLEET DIRECTIVE ONLY *****

Call set_fleet_directive with the overall strategy for your fleet (the ships are listed
under YOUR FLEET in the situation). Captains read it verbatim: keep it under ~150 words
and answer with tool calls, not a long analysis - a reply cut off at the token limit
leaves the whole fleet without a directive.

Individual ship orders are requested in a SEPARATE call immediately after this one,
one ship at a time. Do NOT call issue_order now for individual ships - those orders
are discarded. The ONE order this call accepts is a fleet-wide formation heading:
issue_order(ship_name="all ships", order_text=..., heading={x,y,z}, throttle=T) -
every ship flies it (throttle scaled per hull to your slowest ship, burns start
together) unless its own order sets another heading. Make the directive concrete
enough that the per-ship orders follow from it.

Ships are shown as: NAME [ID] (e.g., "OCS Gemini-1 [beta_1]")
You can use EITHER the name OR the ID when naming ships or targets.

*************************************************"""
        order_requirements = ""
    else:
        mission_block = """***** CRITICAL REQUIREMENT - READ THIS FIRST *****

You MUST call issue_order for EACH of your ships (every ship listed under YOUR FLEET
in the situation below is marked "MUST issue_order").

^^^ YOU MUST ISSUE AN ORDER TO EVERY SHIP LISTED ^^^
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

"""

    enemy_admiral_note = ("- Enemy has an Admiral. Use message_enemy_admiral to negotiate."
                          if enemy_has_admiral else
                          "- Enemy has no Admiral. Their captains act independently.")
    personality_prompt = f"YOUR PERSONALITY:\n{personality}" if personality else ""
    # Commander's notebook (cross-battle lessons): stable per battle, so it
    # belongs in the cacheable doctrine prefix next to the personality.
    if notebook_text:
        personality_prompt = (f"{personality_prompt}\n\n{notebook_text}" if personality_prompt
                              else notebook_text)

    # Allied PD coverage is bounded by the laser's own range (there is no separate
    # "formation radius" constant anywhere in the PD code), so read it from the same
    # data PDLaser.from_fleet_data reads instead of quoting a made-up 50 km.
    pd_range_km = float(
        (fleet_data or {}).get("weapon_types", {}).get("pd_laser", {}).get("range_km", 250)
    )
    pd = pd_doctrine_numbers(fleet_data)
    heat = thermal_doctrine_numbers(fleet_data)
    try:
        from .fleet_intel import WRECK_THREAT_MISS_M, ballistic_horizon_s
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import WRECK_THREAT_MISS_M, ballistic_horizon_s
    pd_values = dict(
        pd_range_km=pd_range_km,
        pd_heat_mw=heat["pd_heat_mw"],
        pd_closure_floor_kps=pd["closure_floor_kps"],
        pd_envelope_crossing_s=pd["envelope_crossing_s"],
        pd_mj_per_turret=pd["mj_per_turret_at_floor"],
        pd_seeker_kill_mj=pd["seeker_kill_mj"],
        pd_turrets_to_blind=pd["turrets_to_blind_at_floor"],
    )
    own_classes = battle_classes.get("ours") or []
    reload_s = float(((fleet_data or {}).get("weapon_types") or {})
                     .get("torpedo_launcher", {}).get("cooldown_s", 12))
    magazine_line = _magazine_line(own_classes, fleet_data) or (
        "8 rounds per corvette tube, 12 per torpedo-cruiser tube")
    pd_doctrine = ADMIRAL_PD_DOCTRINE.format(**pd_values) if use_pd else ""
    torpedo_doctrine = ADMIRAL_TORPEDO_DOCTRINE.format(
        magazine_line=magazine_line, reload_s=reload_s,
        launches_per_tube=torpedo_launches_per_checkpoint(fleet_data, decision_interval_s),
        decision_interval_s=decision_interval_s, **pd_values) if use_tubes else ""
    gun_doctrine = ADMIRAL_GUN_DOCTRINE if use_guns else ""

    doctrine = ADMIRAL_DOCTRINE_CORE.format(
        admiral_name=admiral_name,
        faction=faction.upper(),
        mission_block=mission_block,
        order_requirements=order_requirements,
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        verdict_legend=_verdict_legend({"own_tubes": use_tubes}),
        wreck_miss_km=WRECK_THREAT_MISS_M / 1000.0,
        wreck_horizon_s=ballistic_horizon_s(),
        pd_doctrine=pd_doctrine,
        torpedo_doctrine=torpedo_doctrine,
        gun_doctrine=gun_doctrine,
        heat_hold_pct=heat["gun_hold_pct"],
        heat_forced_pct=heat["forced_pct"],
        heat_meltdown_pct=heat["meltdown_pct"],
        heat_drive_mw=heat["drive_full_mw"],
        heat_fill_min=heat["full_burn_fill_min"],
        pd_heat_mw=heat["pd_heat_mw"],
        resources_line=_resources_line({"own_tubes": use_tubes}, fleet_data, own_classes),
        decision_interval_s=decision_interval_s,
        enemy_admiral_note=enemy_admiral_note,
        personality_prompt=personality_prompt,
    )
    forces = _format_forces(battle_classes, fleet_data)
    return doctrine.rstrip() + ("\n\n" + forces if forces else "") + "\n"


# -----------------------------------------------------------------------------
# Live turn state
# -----------------------------------------------------------------------------

def _lbl(snapshot: Any, ship_id: Optional[str], name: Optional[str] = None) -> str:
    """NAME [ID] for any ship the snapshot knows (ADM-13: one format everywhere)."""
    if not ship_id:
        return name or "?"
    labels = getattr(snapshot, "labels", None) or {}
    if ship_id in labels:
        return labels[ship_id]
    for row in list(getattr(snapshot, "friendly_ships", None) or []) + \
            list(getattr(snapshot, "enemy_ships", None) or []):
        if row.ship_id == ship_id:
            return f"{row.ship_name} [{ship_id}]" if row.ship_name != ship_id else f"[{ship_id}]"
    return f"{name} [{ship_id}]" if name and name != ship_id else f"[{ship_id}]"


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _span_text(values: List[float], fmt: str = "{:.0f}") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "?"
    lo, hi = min(values), max(values)
    return fmt.format(lo) if abs(hi - lo) < 0.5 else f"{fmt.format(lo)}-{fmt.format(hi)}"


def _type_abbr(snapshot: Any, ship_id: str) -> str:
    try:
        from .fleet_intel import class_abbr
    except ImportError:  # pragma: no cover
        from llm.fleet_intel import class_abbr
    types = getattr(snapshot, "ship_types", None) or {}
    stype = types.get(ship_id)
    if stype is None:
        for row in list(getattr(snapshot, "friendly_ships", None) or []) + \
                list(getattr(snapshot, "enemy_ships", None) or []):
            if row.ship_id == ship_id:
                stype = row.ship_type
    return class_abbr(stype)


def _own_ids(snapshot: Any) -> set:
    return {r.ship_id for r in getattr(snapshot, "friendly_ships", None) or []}


def _inbound_rounds(snapshot: Any) -> List[Any]:
    """Rounds that can hit OUR hulls: enemy rounds homing on them + dead rounds on their line."""
    own = _own_ids(snapshot)
    return [r for r in getattr(snapshot, "rounds", None) or []
            if r.threat_id in own and (r.side == "theirs" or r.ballistic)]


def _format_now(snapshot: Any, last_failure: Optional[Tuple[float, str]],
                heat_hold_pct: float) -> str:
    items = []
    for info in getattr(snapshot, "range_groups", None) or []:
        if info.warning:
            items.append(f"{info.label}: {info.warning}")
    # Torpedoes (guided, or dead on a hull's line) are the ordnance an
    # admiral acts on this checkpoint; gun rounds are listed under ORDNANCE.
    by_ship: Dict[str, List[Any]] = {}
    for r in _inbound_rounds(snapshot):
        if r.kind == "torpedo":
            by_ship.setdefault(r.threat_id, []).append(r)
    if by_ship:
        parts = []
        for sid, rounds in sorted(by_ship.items(),
                                  key=lambda kv: min((r.eta_s or 1e9) for r in kv[1])):
            kinds = []
            torps = [r for r in rounds if not r.ballistic]
            dead = [r for r in rounds if r.ballistic]
            if torps:
                kinds.append(f"{len(torps)} torpedo{'es' if len(torps) > 1 else ''} "
                             f"ETA {_span_text([r.eta_s for r in torps])} s")
            if dead:
                kinds.append(f"{len(dead)} dead torpedo{'es' if len(dead) > 1 else ''} on its line "
                             f"in {_span_text([r.tca_s for r in dead])} s")
            parts.append(f"{_lbl(snapshot, sid)} ({', '.join(kinds)})")
        items.append("Torpedoes inbound on: " + "; ".join(parts[:5])
                     + (f"; +{len(parts) - 5} more" if len(parts) > 5 else "")
                     + " - see ORDNANCE")
    for row in getattr(snapshot, "friendly_ships", None) or []:
        problems = []
        if row.hull_integrity < 35:
            problems.append(f"hull {row.hull_integrity:.0f}%")
        if row.heat_percent >= heat_hold_pct:
            problems.append(f"heat {row.heat_percent:.0f}% (guns hold at {heat_hold_pct:.0f}%)")
        if problems:
            items.append(f"{_lbl(snapshot, row.ship_id, row.ship_name)}: {', '.join(problems)}")
    if last_failure:
        t, why = last_failure
        items.append(f"Your fleet directive at T+{t:.0f}s FAILED ({why}): captains were told "
                     "there was no new directive. Answer with tool calls first, analysis short.")
    if not items:
        items.append("nothing urgent: no merge warning, no torpedo inbound on your hulls")
    return "!! NOW (act on these first)\n" + "\n".join(f"- {i}" for i in items)


_STATUS_ORDER = {"UNABLE": 0, "DEVIATE": 1, "ACK": 2}


def _compliance_text(compliance: Optional[Dict[str, Any]], snapshot: Any) -> Tuple[str, bool]:
    """
    ('did: ...', followed?) from a report's compliance record.

    The comms layer's own verdict (fleet_comms.compliance: ``followed`` and
    one ``deviations`` line per mismatch) is authoritative when present - it
    resolved the ordered target to a hull and knows which maneuvers the order
    allowed. Older records without it are judged here on the raw fields, with
    targets resolved through the snapshot's labels so 'OCS Lancer-1 [beta_2]',
    'OCS Lancer-1' and 'beta_2' are the same ship.
    """
    if not isinstance(compliance, dict):
        return "", True

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    em = compliance.get("executed_maneuver")
    salvo_raw = compliance.get("ordered_salvo")
    osalvo, launched = count(salvo_raw), count(compliance.get("launched"))
    verb = "committed" if compliance.get("final") is False else "launched"
    salvo_text = f"salvo {osalvo} ordered, {launched} {verb}" if (osalvo or launched) else ""

    verdict, deviations = compliance.get("followed"), compliance.get("deviations")
    if isinstance(verdict, bool) or isinstance(deviations, list):
        lines = [_clip(d, 120) for d in deviations or [] if d] if isinstance(deviations, list) else []
        followed = verdict if isinstance(verdict, bool) else not lines
        if lines:
            return "did: " + "; ".join(lines), followed
        parts = [p for p in (f"flew {em}" if em else "", salvo_text) if p]
        return ("did: " + ", ".join(parts)) if parts else "", followed

    # Legacy record: compare the fields.
    parts, followed = [], True
    om = compliance.get("ordered_maneuver")
    if om or em:
        if om and em and str(om).upper() != str(em).upper():
            parts.append(f"ordered {om} -> flew {em}")
            followed = False
        elif em:
            parts.append(f"flew {em}")
    ot, at = compliance.get("ordered_target"), compliance.get("actual_target")
    if ot:
        try:
            from .fleet_intel import resolve_ship_ref, split_label
        except ImportError:  # pragma: no cover
            from llm.fleet_intel import resolve_ship_ref, split_label
        labels = getattr(snapshot, "labels", None) or {}
        names = {sid: split_label(lbl)[0] or sid for sid, lbl in labels.items()}

        def key(ref: Any) -> str:
            return resolve_ship_ref(ref, names) or split_label(ref)[0].lower()

        if not at or key(ot) != key(at):
            shown = [_lbl(snapshot, resolve_ship_ref(r, names), split_label(r)[0]) if r
                     else "nothing" for r in (ot, at)]
            parts.append(f"target {shown[0]} -> {shown[1]}")
            followed = False
    if osalvo and launched < osalvo:
        followed = False
    elif launched and str(salvo_raw).strip() == "0":
        followed = False                    # an explicit hold, broken
        salvo_text = f"salvo hold ordered, {launched} {verb}"
    if salvo_text:
        parts.append(salvo_text)
    return ("did: " + ", ".join(parts)) if parts else "", followed


def _format_captain_reports(reports: Optional[List[Dict[str, Any]]], snapshot: Any) -> str:
    """CAPTAIN REPORTS: non-ACK first, one line each; plain ACKs collapsed into one line."""
    if reports is None:
        return ""
    name_to_id = {r.ship_name: r.ship_id for r in getattr(snapshot, "friendly_ships", None) or []}
    if not reports:
        first = (getattr(snapshot, "checkpoint", 0) or 0) <= 1
        return ("=== CAPTAIN REPORTS ===\n" + ("(none yet - captains report after their first decision)"
                                                if first else "(no reports filed last checkpoint)"))
    checkpoints = sorted({r.get("checkpoint") for r in reports if r.get("checkpoint") is not None})
    when = f" (filed at checkpoint {checkpoints[-1]})" if checkpoints else ""
    lines = [f"=== CAPTAIN REPORTS{when} ==="]
    acks = []
    for r in sorted(reports, key=lambda r: _STATUS_ORDER.get(str(r.get("status", "")).upper(), 3)):
        status = str(r.get("status") or "ACK").upper()
        sid = r.get("ship_id") or name_to_id.get(r.get("ship_name"))
        who = _lbl(snapshot, sid, r.get("ship_name")) if sid else (r.get("ship_name") or "?")
        did, followed = _compliance_text(r.get("compliance"), snapshot)
        contact, request = r.get("contact"), r.get("request")
        if status == "ACK" and followed and not contact and not request:
            acks.append(who)
            continue
        text = f"{status} {who}"
        if r.get("reason"):
            text += f": \"{_clip(r['reason'], 200)}\""
        if r.get("intent"):
            text += f" | intends: {_clip(r['intent'], 120)}"
        if isinstance(contact, dict) and (contact.get("ship") or contact.get("note")):
            text += (f" | contact: {contact.get('ship') or '?'}"
                     + (f" {contact['assessment']}" if contact.get("assessment") else "")
                     + (f" - {_clip(contact['note'], 100)}" if contact.get("note") else ""))
        if isinstance(request, dict) and request.get("text"):
            text += f" | asks [{request.get('kind') or 'request'}]: {_clip(request['text'], 120)}"
        if did:
            text += f" | {did}" + ("" if followed else " (NOT as ordered)")
        lines.append(text)
    if acks:
        lines.append(f"ACK, nothing to flag ({len(acks)}): " + ", ".join(acks))
    return "\n".join(lines)


def _format_since(snapshot: Any) -> str:
    digest = getattr(snapshot, "digest", None)
    if digest is None:
        return ""
    lines = [f"=== SINCE LAST CHECKPOINT (T+{digest.since_t:.0f} -> T+{digest.until_t:.0f}) ==="]
    before = getattr(snapshot, "hull_before", None) or {}
    now = getattr(snapshot, "hull_now", None) or {}

    def deaths(entries):
        out = []
        for e in entries:
            how = "crippled, detonating" if e.get("dying") else "destroyed"
            killer = f" by {_lbl(snapshot, e['killer_id'])}" if e.get("killer_id") else ""
            cause = e.get("cause")
            cause = f", {cause.replace('_', ' ')}" if cause and not e.get("dying") else ""
            out.append(f"{_lbl(snapshot, e['ship_id'])} {how}{killer}{cause} T+{e['t']:.0f}")
        return "; ".join(out) if out else "none"

    lines.append(f"Kills: {deaths(digest.kills)} | Losses: {deaths(digest.losses)}")

    def ledger_lines(ledgers):
        rows = []
        for sid, led in sorted(ledgers.items(), key=lambda kv: -kv[1].energy_gj):
            src: Dict[Tuple[str, str], int] = {}
            for (source, kind), n in led.hits.items():
                src[(source, kind)] = src.get((source, kind), 0) + n
            desc = ", ".join(f"{n} {kind}{'s' if n > 1 else ''} from {_lbl(snapshot, source)}"
                             for (source, kind), n in sorted(src.items(), key=lambda kv: -kv[1])[:3])
            if len(src) > 3:
                desc += f", +{len(src) - 3} more sources"
            hull = ""
            if sid in before and sid in now:
                hull = f" hull {before[sid]:.0f}->{now[sid]:.0f}%"
            rows.append(f"  {_lbl(snapshot, sid)}{hull}: {desc}; {led.energy_gj:.0f} GJ")
        return rows

    taken = ledger_lines(digest.taken)
    dealt = ledger_lines(digest.dealt)
    lines.append("Hits on you:" + ("" if taken else " none"))
    lines.extend(taken[:6])
    lines.append("Your hits:" + ("" if dealt else " none"))
    lines.extend(dealt[:6])

    def torps(c):
        if not c.get("launched") and not sum(c.values()):
            return "none"
        order = (("launched", "launched"), ("impact", "hit"), ("blinded", "blinded by PD"),
                 ("destroyed", "destroyed by PD"), ("dry", "ran dry"), ("missed", "missed"),
                 ("retargeted", "retargeted"))
        return ", ".join(f"{c[k]} {word}" for k, word in order if c.get(k))

    lines.append(f"Torpedoes - yours: {torps(digest.torps['ours'])} | theirs: {torps(digest.torps['theirs'])}")
    sl = digest.slugs

    def guns(c):
        return f"{c.get('fired', 0)} fired, {c.get('hit', 0)} hit" if c.get("fired") or c.get("hit") else "none"

    if sum(sl["ours"].values()) or sum(sl["theirs"].values()):
        lines.append(f"Gun rounds - yours: {guns(sl['ours'])} | theirs: {guns(sl['theirs'])}")
    return "\n".join(lines)


def _format_geometry(snapshot: Any) -> str:
    groups = getattr(snapshot, "range_groups", None) or []
    centre = getattr(snapshot, "own_centre", None) or {}
    if not groups and not centre:
        return ""
    lines = ["=== GEOMETRY (from your fleet centre; + closing, - opening) ==="]
    if centre:
        x, y, z = centre["pos_km"]
        lines.append(f"You: {centre['n']} ships, centre ({x:.0f},{y:.0f},{z:.0f}) km, "
                     f"{centre['speed_kps']:.1f} km/s toward {centre['direction']}, "
                     f"spread {centre['spread_km']:.0f} km")
    enemies = {e.ship_id: e for e in getattr(snapshot, "enemy_ships", None) or []}
    own_rows = getattr(snapshot, "friendly_ships", None) or []
    own_tubes = any(r.launchers for r in own_rows)
    own_gun_range = max([getattr(r, "gun_range_km", 0.0) or 0.0 for r in own_rows] or [0.0])
    changes = getattr(snapshot, "group_range_change_km", None) or {}
    motion = getattr(snapshot, "group_motion", None) or {}
    burns = getattr(snapshot, "group_burn_g", None) or {}
    opened = set(getattr(snapshot, "groups_opened", None) or [])

    def gun_reach(who: str, reach_km: float, info: Any) -> str:
        if reach_km <= 0:
            return ""
        if info.range_km <= reach_km:
            when = "IN RANGE now"
        elif info.closing_kps > 0.05:
            when = f"in range in ~{(info.range_km - reach_km) / info.closing_kps:.0f} s"
        else:
            when = "out of range, not closing"
        return f"{who} guns reach {reach_km:.0f} km ({when})"

    for info in groups[:4]:
        members = [enemies.get(sid) for sid in info.ship_ids]
        comp = ", ".join(f"{_lbl(snapshot, sid)} {_type_abbr(snapshot, sid)}" for sid in info.ship_ids[:5])
        if len(info.ship_ids) > 5:
            comp += f", +{len(info.ship_ids) - 5}"
        rate = (f"closing {info.closing_kps:.2f} km/s" if info.closing_kps >= 0
                else f"OPENING {-info.closing_kps:.2f} km/s")
        head = f"{info.label} ({comp}): {info.range_km:.0f} km, {rate}"
        if info.label in changes:
            head += f" ({changes[info.label]:+.0f} km in 15 s)"
        if info.label in motion:
            ours, theirs = motion[info.label]
            head += (f" = you {abs(ours) / 1000.0:.1f} km/s {'toward' if ours >= 0 else 'away from'} them"
                     f" + they {abs(theirs) / 1000.0:.1f} {'toward' if theirs >= 0 else 'away from'} you")
        if info.label in burns:
            g = burns[info.label]
            head += ("; they coast" if abs(g) < 0.05 else
                     f"; they burn {'toward you' if g > 0 else 'away'} {abs(g):.1f}g")
        lines.append(head)
        if info.cpa_t_s > 0.5:
            who = f" ({info.cpa_ship})" if info.cpa_ship else ""
            coast = f"if both coast: closest {info.cpa_km:.1f} km{who} in {info.cpa_t_s:.0f} s"
        else:
            coast = "range not closing (closest is now)"
        stop = ""
        if info.closing_kps > 0 and info.stop_s > 0:
            late = info.warning and ("too late" in info.warning or "out-accelerates" in info.warning)
            stop = (f" | your stop {info.stop_km:.0f} km / {info.stop_s:.0f} s at {info.accel_g:.1f}g "
                    f"+{info.flip_s:.0f} s flip" + (" - TOO LATE to brake" if late else ""))
        lines.append(f"  {coast}{stop}")
        reach = []
        their_gun = max([getattr(m, "gun_range_km", 0.0) or 0.0 for m in members if m is not None] or [0.0])
        their_tubes = sum(m.launchers for m in members if m is not None)
        for text in (gun_reach("their", their_gun, info), gun_reach("your", own_gun_range, info)):
            if text:
                reach.append(text)
        verdicts = [m.solution.verdict for m in members if m is not None and m.solution is not None]
        if own_tubes and verdicts:
            reach.append("your torpedoes: " + ", ".join(
                f"{verdicts.count(v)} {v}" for v in ("GOOD", "POOR", "NO") if verdicts.count(v)))
        if their_tubes:
            reach.append(f"their torpedoes: {their_tubes} tube{'s' if their_tubes > 1 else ''}")
        if reach:
            lines.append("  reach - " + "; ".join(reach))
        if info.label in opened:
            lines.append("  you passed through them since the last checkpoint: the range is now opening")
    if len(groups) > 4:
        lines.append(f"(+{len(groups) - 4} more groups further out)")
    return "\n".join(lines)


def _solution_text(sol: Any) -> str:
    if sol is None:
        return ""
    if sol.verdict == "GOOD":
        return f"GOOD {sol.tof_s:.0f} s ~{sol.energy_gj:.0f} GJ"
    if sol.verdict == "POOR":
        return f"POOR ({sol.reason}) ~{sol.energy_gj:.0f} GJ"
    return f"NO ({sol.reason})"


def _format_threat_table(snapshot: Any) -> str:
    rows = list(getattr(snapshot, "enemy_ships", None) or [])
    if not rows:
        return "=== ENEMY THREAT TABLE ===\nNo enemy ships detected"
    own_tubes = any(r.launchers for r in getattr(snapshot, "friendly_ships", None) or [])
    lines = ["=== ENEMY THREAT TABLE (range/closure from your fleet centre) ==="]
    rows.sort(key=lambda e: e.range_km if e.range_km is not None else e.distance_from_closest_friendly_km)
    for e in rows:
        parts = [f"{_lbl(snapshot, e.ship_id, e.ship_name)} {_type_abbr(snapshot, e.ship_id)}"
                 + (f" {e.accel_g:.1f}g" if e.accel_g else "")]
        if e.range_km is not None:
            parts.append(f"{e.range_km:.0f} km {e.closing_kps:+.1f} km/s")
        if e.nearest_own_id:
            parts.append(f"nearest {_lbl(snapshot, e.nearest_own_id)} "
                         f"{e.distance_from_closest_friendly_km:.0f} km")
        elif e.range_km is None:
            parts.append(f"nearest hull {e.distance_from_closest_friendly_km:.0f} km "
                         f"{e.closing_rate_kps:+.1f} km/s")
        if e.hull_integrity is not None:
            parts.append(f"hull {e.hull_integrity:.0f}%")
        if e.pd_total:
            parts.append(f"PD {e.pd_operational}/{e.pd_total}")
        arms = e.armament or ""
        if e.launchers:
            left = max(0, e.magazine - e.torpedoes_launched)
            arms += f", ~{left}/{e.magazine} rounds left ({e.torpedoes_launched} seen)"
        if arms:
            parts.append(arms)
        if e.targeting_id:
            parts.append(f"aims at {_lbl(snapshot, e.targeting_id)}")
        parts.append(f"dealt {e.dealt_hits} hits {e.dealt_gj:.0f} GJ" if e.dealt_hits else "dealt nothing")
        if own_tubes and e.solution is not None:
            text = f"torpedo {_solution_text(e.solution)}"
            if e.our_rounds_inbound:
                text += f", {e.our_rounds_inbound} of yours inbound"
            parts.append(text)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _sources(snapshot: Any, rs: List[Any]) -> str:
    ids = []
    for r in rs:
        if r.source_id and r.source_id not in ids:
            ids.append(r.source_id)
    text = ", ".join(_lbl(snapshot, sid) for sid in ids[:3])
    return text + (f" +{len(ids) - 3}" if len(ids) > 3 else "")


def _inbound_lines(snapshot: Any, inbound: List[Any], indent: str) -> List[str]:
    """
    Inbound rounds grouped per salvo: one line per (kind, target hull) with the
    shooters named, rather than one line per round (the old list capped at 10
    identical lines hid the rest).
    """
    groups: Dict[Tuple[str, str], List[Any]] = {}
    for r in inbound:
        kind = "dead" if (r.kind == "torpedo" and r.ballistic) else r.kind
        groups.setdefault((kind, r.threat_id), []).append(r)
    order = {"torpedo": 0, "dead": 1, "slug": 2}
    lines = []
    for (kind, target), rs in sorted(groups.items(),
                                     key=lambda kv: (order[kv[0][0]], min(r.eta_s or 1e9 for r in kv[1]))):
        n = len(rs)
        energy = _span_text([r.energy_gj for r in rs])
        if kind == "torpedo":
            inside = [r.nez_inside for r in rs if r.nez_inside is not None]
            nez = ""
            if inside:
                k = sum(1 for v in inside if v)
                nez = (" - inside NEZ: cannot be outrun, EVADE presents armor" if k == len(inside)
                       else " - can still be outrun: EVADE runs" if k == 0
                       else f" - {k}/{len(inside)} inside NEZ")
            lines.append(f"{indent}{n}x torpedo -> {_lbl(snapshot, target)} from {_sources(snapshot, rs)}: "
                         f"{_span_text([r.range_km for r in rs])} km, ETA {_span_text([r.eta_s for r in rs])} s, "
                         f"~{energy} GJ each{nez}")
        elif kind == "dead":
            states = sorted({r.state for r in rs})
            lines.append(f"{indent}{n}x dead torpedo ({'/'.join(states)}) -> {_lbl(snapshot, target)}: passes "
                         f"{_span_text([r.miss_km for r in rs], '{:.2f}')} km in {_span_text([r.tca_s for r in rs])} s, "
                         f"~{energy} GJ each")
        else:
            weapons = sorted({r.weapon for r in rs})
            lines.append(f"{indent}{n}x {'/'.join(weapons)} -> {_lbl(snapshot, target)} from "
                         f"{_sources(snapshot, rs)}: ETA {_span_text([r.eta_s for r in rs])} s, "
                         f"~{_span_text([r.energy_gj for r in rs], '{:.1f}')} GJ each")
    return lines


def _format_ordnance(snapshot: Any) -> str:
    rounds = list(getattr(snapshot, "rounds", None) or [])
    if not rounds and getattr(snapshot, "digest", None) is None:
        return ""
    lines = ["=== ORDNANCE IN FLIGHT ===", "INBOUND ON YOU (can hit your hulls):"]
    inbound = _inbound_rounds(snapshot)
    lines.extend(_inbound_lines(snapshot, inbound, "  "))
    if inbound:
        ships = sorted({r.threat_id for r in inbound}, key=lambda s: (len(s), s))
        lines.append("  EVADE makes the engine dodge (slugs, dead rounds), RUN or PRESENT armor "
                     "(torpedoes) for: " + ", ".join(_lbl(snapshot, s) for s in ships))
    else:
        lines.append("  none - nothing in flight is on course for your hulls; no EVADE needed for ordnance")

    ours = [r for r in rounds if r.side == "ours"]
    guided: Dict[str, List[Any]] = {}
    for r in ours:
        if r.kind == "torpedo" and r.state == "guided":
            guided.setdefault(r.target_id, []).append(r)
    on_course = [r for r in rounds if r.kind == "torpedo" and r.ballistic and r.threat_id
                 and r.threat_id not in _own_ids(snapshot)]
    spent_ours = sum(1 for r in ours if r.kind == "torpedo" and r.ballistic and not r.threat_id)
    spent_theirs = sum(1 for r in rounds if r.side == "theirs" and r.kind == "torpedo"
                       and r.ballistic and not r.threat_id)
    our_slugs = sum(1 for r in ours if r.kind == "slug")
    if guided or on_course or spent_ours or our_slugs:
        lines.append("YOURS:")
    for target, rs in sorted(guided.items(), key=lambda kv: min(r.eta_s or 1e9 for r in kv[1])):
        reach = [r.can_reach for r in rs if r.can_reach is not None]
        fate = ""
        if reach:
            k = sum(1 for v in reach if v)
            fate = (f", {k}/{len(reach)} still catch a runner"
                    + ("" if k == len(reach) else ", the rest run dry first"))
        etas = [r.eta_s for r in rs if r.eta_s is not None]
        when = f"ETA {_span_text(etas)} s" if etas else "not closing yet"
        lines.append(f"  -> {_lbl(snapshot, target)}: {len(rs)} guided, {when}{fate}")
    by_target: Dict[str, List[Any]] = {}
    for r in on_course:
        by_target.setdefault(r.threat_id, []).append(r)
    for target, rs in by_target.items():
        lines.append(f"  {len(rs)} dead round(s) still on course for {_lbl(snapshot, target)} "
                     f"(passes {_span_text([r.miss_km for r in rs], '{:.2f}')} km in "
                     f"{_span_text([r.tca_s for r in rs])} s)")
    if spent_ours or spent_theirs:
        lines.append(f"  spent: {spent_ours} of yours and {spent_theirs} of theirs are dead rounds "
                     "on nobody's line - ignore them")
    if our_slugs:
        lines.append(f"  your gun rounds in flight: {our_slugs}")
    return "\n".join(lines)


def _format_torpedo_economy(snapshot: Any, fleet_data: Dict[str, Any]) -> str:
    own = getattr(snapshot, "friendly_ships", None) or []
    enemies = getattr(snapshot, "enemy_ships", None) or []
    own_tubes = any(r.launchers for r in own)
    their_tubes = sum(e.launchers for e in enemies)
    seen = getattr(snapshot, "enemy_torpedoes_seen", 0) or 0
    if not own_tubes and not their_tubes and not seen:
        return ""
    lines = ["=== TORPEDO ECONOMY ==="]
    interval = getattr(snapshot, "decision_interval_s", 30.0) or 30.0
    if own_tubes:
        remaining, start = getattr(snapshot, "torpedo_magazine", (0, 0))
        per = torpedo_launches_per_checkpoint(fleet_data, interval)
        reload_s = getattr(snapshot, "reload_s", 0.0) or 12.0
        lines.append(f"Yours: {remaining}/{start or remaining} rounds left; "
                     f"{getattr(snapshot, 'launches_available', 0)} launches possible before "
                     f"T+{snapshot.timestamp + interval:.0f}s (reload {reload_s:.0f} s = up to {per} "
                     f"per tube per checkpoint)")
        try:
            from .fleet_intel import kill_rounds_at, turrets_to_blind
        except ImportError:  # pragma: no cover
            from llm.fleet_intel import kill_rounds_at, turrets_to_blind
        floor_k = turrets_to_blind(fleet_data, MIN_CLOSING_SPEED_KPS)
        kill = getattr(snapshot, "kill_rounds", None) or {}
        rank = {"GOOD": 0, "POOR": 1, "NO": 2}
        targets = [e for e in enemies if e.solution is not None and e.solution.verdict != "NO"]
        targets.sort(key=lambda e: (rank.get(e.solution.verdict, 3), e.range_km or 0.0))
        for e in targets[:3]:
            sol = e.solution
            cover = sol.pd_cover if sol.pd_cover is not None else e.pd_operational
            soak = sol.soak if sol.soak is not None else cover // max(1, floor_k)
            # Sized at the doctrine's per-round energy; slow rounds need more.
            to_kill = kill_rounds_at(kill.get(e.ship_id, 6), sol.energy_gj)
            each = (f" at ~{sol.energy_gj:.0f} GJ each"
                    if to_kill != kill.get(e.ship_id, 6) else "")
            need = soak + to_kill
            text = (f"Salvo vs {_lbl(snapshot, e.ship_id)} ({sol.verdict}): {cover} PD turrets cover it "
                    f"-> ~{soak} soaked + {to_kill} to kill{each} = {need} together")
            more = need - e.our_rounds_inbound
            if e.our_rounds_inbound:
                text += (f"; {e.our_rounds_inbound} of yours already inbound -> "
                         + ("ENOUGH, do not add (overkill)" if more <= 0 else f"{more} more"))
            available = getattr(snapshot, "launches_available", 0) or 0
            if more > available:
                text += f" (you can launch {available} now: not enough to kill through that PD)"
            lines.append(text)
        if not targets:
            lines.append("No target has a GOOD/POOR solution now: hold the magazine "
                         "(re-engage first if the range is opening).")
        lines.append(f"Trickled rounds die: 1-2 rounds into {floor_k}+ turrets are blinded "
                     "one by one - launch together or not at all.")
    if their_tubes or seen:
        left = sum(max(0, e.magazine - e.torpedoes_launched) for e in enemies if e.launchers)
        lines.append(f"Theirs: {seen} launched so far; {their_tubes} tube{'s' if their_tubes != 1 else ''} "
                     f"in their fleet (~{left} rounds left). Inbound ones are under ORDNANCE.")
    else:
        lines.append("Theirs: no torpedo launchers in their fleet - nothing to EVADE from torpedoes.")
    return "\n".join(lines)


def _heat_compact(row: Any) -> str:
    thermal = getattr(row, "thermal", None)
    pct = row.heat_percent
    text = ""
    warming = False
    if thermal:
        t100 = thermal.get("time_to_100_retracted_s")
        warming = t100 is not None and t100 > 0 and thermal.get("net_rate_mw", 0) > 0
        if pct >= 20 or warming:
            text = f" heat {pct:.0f}%"
            if warming:
                ext = thermal.get("time_to_100_extended_s")
                text += (f" (+{thermal.get('net_rate_mw', 0):.0f} MW: 100% in {_fmt_minutes(t100)} "
                         f"retracted / {_fmt_minutes(ext)} extended)")
        if thermal.get("guns_heat_limited"):
            text += " GUNS HOLDING"
    elif pct >= 20:
        text = f" heat {pct:.0f}%"
    if getattr(row, "radiators_extended", False):
        text += " radiators OUT"
    return text


def _format_fleet(snapshot: Any, phase: str) -> str:
    rows = getattr(snapshot, "friendly_ships", None) or []
    if not rows:
        return "=== YOUR FLEET ===\nFleet status unknown" if snapshot is None else "=== YOUR FLEET ===\n(no ships)"
    hulls = [r.hull_integrity for r in rows]
    heats = [r.heat_percent for r in rows]
    pd_op = sum(r.pd_turrets_operational for r in rows)
    pd_tot = sum(r.pd_turrets_total for r in rows)
    hull_range = _span_text(hulls)
    summary = (f"All: hull {hull_range}%, heat max {max(heats):.0f}%, "
               f"dv >= {min(r.delta_v_remaining for r in rows):.0f} km/s, PD {pd_op}/{pd_tot}")
    if any(r.launchers for r in rows):
        summary += (f", torpedoes {sum(r.torpedoes_remaining for r in rows)} "
                    f"({sum(r.launches_available for r in rows)} launchable now)")
    lines = [f"=== YOUR FLEET ({len(rows)} ships) ===", summary]
    must = " <- MUST issue_order for this ship!" if phase != "directive" else ""
    for r in rows:
        bits = [f"hull {r.hull_integrity:.0f}%"]
        heat = _heat_compact(r)
        if heat:
            bits.append(heat.strip())
        if r.pd_turrets_total and r.pd_turrets_operational < r.pd_turrets_total:
            bits.append(f"PD {r.pd_turrets_operational}/{r.pd_turrets_total} DEGRADED")
        if r.launchers or r.torpedoes_remaining:
            bits.append(f"torp {r.torpedoes_remaining} ({r.launches_available} now)")
        if r.weapons_destroyed:
            bits.append(f"LOST {', '.join(r.weapons_destroyed)}")
        if r.delta_v_remaining < 300:
            bits.append(f"dv {r.delta_v_remaining:.0f} km/s")
        doing = r.current_maneuver
        if r.current_target:
            doing += f" -> {_lbl(snapshot, r.current_target)}"
        line = (f"{_lbl(snapshot, r.ship_id, r.ship_name)} {_type_abbr(snapshot, r.ship_id)} | "
                f"{' '.join(bits)} | {doing}")
        if r.targeted_by:
            line += f" | aimed at by {', '.join(r.targeted_by)}"
        lines.append(line + must)
    return "\n".join(lines)


def _format_headings(snapshot: Any) -> str:
    text = _text_attr(snapshot, "headings")
    if text:
        return ("=== HEADINGS (world frame, fleet centre: issue_order heading=; the drive "
                "thrusts along the NOSE) ===\n" + text)
    # Hand-built snapshots may carry only the legacy full range-control block.
    return _text_attr(snapshot, "range_control")


def _format_engine(snapshot: Any) -> str:
    notes = getattr(snapshot, "engine_notes", None) or []
    if not notes:
        return ""
    return "=== ENGINE (what your orders became) ===\n" + "\n".join(notes)


def _standing_plan_section(standing_plan: Optional[str], standing_plan_time: float) -> str:
    # Standing battle plan: the admiral's only self-authored memory. Rendered in
    # the volatile turn state (it changes when amended) so the cached doctrine
    # prefix is untouched.
    if standing_plan:
        return (
            f"=== YOUR STANDING BATTLE PLAN (set by you at T+{standing_plan_time:.0f}s) ===\n"
            f"{standing_plan}\n"
            "This is your own plan, echoed back to you. Execute its next phase, or amend it\n"
            "with set_battle_plan if the situation has changed."
        )
    return (
        "=== YOUR STANDING BATTLE PLAN ===\n"
        "(none yet) You have no memory of your own intent between checkpoints except\n"
        "this plan. Call set_battle_plan now - phases plus the conditions that trigger\n"
        "them - so your future self fights the same battle you are planning."
    )


def format_admiral_turn_state(
    snapshot: Any,
    fleet_data: Dict[str, Any],
    phase: str = "full",
    standing_plan: Optional[str] = None,
    standing_plan_time: float = 0.0,
    captain_reports: Optional[List[Dict[str, Any]]] = None,
    last_directive_failure: Optional[Tuple[float, str]] = None,
    received_messages: Optional[List[str]] = None,
    enemy_proposed_draw: bool = False,
    communications_log: Optional[List[Any]] = None,
    decision_interval_s: float = 30.0,
) -> str:
    """The live, per-checkpoint user turn (everything after ADMIRAL_STATE_MARKER)."""
    heat = thermal_doctrine_numbers(fleet_data)
    if snapshot is not None:
        cp = getattr(snapshot, "checkpoint", 0) or 0
        interval = getattr(snapshot, "decision_interval_s", decision_interval_s) or decision_interval_s
        header = (f"T+{snapshot.timestamp:.0f}s" + (f", checkpoint {cp}" if cp else "")
                  + f" - these orders run until T+{snapshot.timestamp + interval:.0f}s")
    else:
        header = "(no live data)"
    blocks = []
    if snapshot is not None:
        blocks.append(_format_now(snapshot, last_directive_failure, heat["gun_hold_pct"]))
        blocks.append(_format_captain_reports(captain_reports, snapshot))
        blocks.append(_format_since(snapshot))
        blocks.append(_format_geometry(snapshot))
        blocks.append(_format_threat_table(snapshot))
        blocks.append(_format_ordnance(snapshot))
        blocks.append(_format_torpedo_economy(snapshot, fleet_data))
    blocks.append(_format_fleet(snapshot, phase))
    if snapshot is not None:
        blocks.append(_format_headings(snapshot))
        blocks.append(_format_engine(snapshot))
    blocks.append(_standing_plan_section(standing_plan, standing_plan_time))

    comms = []
    if received_messages:
        comms.append("=== MESSAGES FROM ENEMY ADMIRAL ===")
        comms.extend(f"  \"{msg}\"" for msg in received_messages)
    if enemy_proposed_draw:
        comms.append("*** ENEMY HAS PROPOSED A DRAW ***")
        comms.append("Use accept_fleet_draw to accept or reject_fleet_draw to refuse.")
    if communications_log:
        comms.append("=== CAPTAIN COMMUNICATIONS (You see all) ===")
        for msg in communications_log[-5:]:  # Last 5 messages
            comms.append(f"  [{msg.ship_name}] {msg.sender_name}: \"{msg.content}\"")
    blocks.append("\n".join(comms))
    body = "\n\n".join(b for b in blocks if b)
    return ADMIRAL_TURN_STATE.format(header=header, body=body)


def _text_attr(obj: Any, name: str) -> str:
    """A string attribute of a snapshot-like object, "" when absent/not text."""
    value = getattr(obj, name, "") if obj is not None else ""
    return value if isinstance(value, str) else ""


def _admiral_parts(
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
    phase: str = "full",
    standing_plan: Optional[str] = None,
    standing_plan_time: float = 0.0,
    notebook_text: Optional[str] = None,
    decision_interval_s: float = 30.0,
    captain_reports: Optional[List[Dict[str, Any]]] = None,
    last_directive_failure: Optional[Tuple[float, str]] = None,
    battle_classes: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str]:
    """(static system text, live turn state)."""
    classes = battle_classes if battle_classes and (battle_classes.get("ours")
                                                    or battle_classes.get("theirs")) \
        else _classes_from_snapshot(snapshot_t_zero)
    static = build_admiral_static(
        admiral_name=admiral_name, faction=faction, fleet_data=fleet_data,
        personality=personality, phase=phase, enemy_has_admiral=enemy_has_admiral,
        notebook_text=notebook_text, decision_interval_s=decision_interval_s,
        battle_classes=classes)
    live = format_admiral_turn_state(
        snapshot_t_zero, fleet_data, phase=phase, standing_plan=standing_plan,
        standing_plan_time=standing_plan_time, captain_reports=captain_reports,
        last_directive_failure=last_directive_failure, received_messages=received_messages,
        enemy_proposed_draw=enemy_proposed_draw, communications_log=communications_log,
        decision_interval_s=decision_interval_s)
    return static, live


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
    standing_plan: Optional[str] = None,
    standing_plan_time: float = 0.0,
    notebook_text: Optional[str] = None,
    decision_interval_s: float = 30.0,
    captain_reports: Optional[List[Dict[str, Any]]] = None,
    last_directive_failure: Optional[Tuple[float, str]] = None,
    battle_classes: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Build the whole admiral prompt as one string (doctrine + turn state).

    LLMAdmiral sends it as two messages (build_admiral_messages): the
    battle-static doctrine/FORCES as the cacheable system prompt and the live
    turn state as the user turn. snapshot_t_minus_15 is accepted for
    compatibility; its information reaches the turn state as each group's
    range change (AdmiralSnapshot.group_range_change_km).

    Args:
        phase: Which call this prompt is for. "directive" asks for the fleet
            directive only (per-ship orders are collected in a later call and any
            issue_order made now is discarded, so the checklist is suppressed);
            "full"/"orders" keep the issue-an-order-for-every-ship requirement.
        captain_reports: captain_report dicts filed at the previous checkpoint
            (rendered as CAPTAIN REPORTS; None omits the block).
        last_directive_failure: (sim time, reason) if the previous checkpoint
            ended without a directive.
        battle_classes: {"ours": [...], "theirs": [...]} ship classes seen this
            battle - picks the doctrine parts and the FORCES table.
    """
    static, live = _admiral_parts(
        admiral_name, faction, snapshot_t_minus_15, snapshot_t_zero, personality, fleet_data,
        enemy_has_admiral=enemy_has_admiral, enemy_proposed_draw=enemy_proposed_draw,
        received_messages=received_messages, communications_log=communications_log,
        phase=phase, standing_plan=standing_plan, standing_plan_time=standing_plan_time,
        notebook_text=notebook_text, decision_interval_s=decision_interval_s,
        captain_reports=captain_reports, last_directive_failure=last_directive_failure,
        battle_classes=battle_classes)
    return static + "\n" + live


def build_admiral_situation_brief(snapshot: Any) -> str:
    """Compact live picture for the admiral's replies to captains."""
    if snapshot is None:
        return ""
    blocks = [_format_geometry(snapshot), _format_threat_table(snapshot)]
    inbound = _format_ordnance(snapshot)
    if inbound:
        blocks.append(inbound.split("\nYOURS:")[0])
    return "\n\n".join(b for b in blocks if b)


def build_admiral_response_prompt(
    admiral_name: str,
    captain_ship_name: str,
    question: str,
    personality: Optional[str],
    recent_decisions: Optional[List[Any]] = None,
    standing_plan: Optional[str] = None,
    decision_interval_s: float = 30.0,
    situation: Optional[str] = None,
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
            elif getattr(decision, "directive_failure", None):
                orders_lines.append(f"FLEET DIRECTIVE: none (the call failed: {decision.directive_failure})")

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

    standing_plan_section = (
        f"=== YOUR STANDING BATTLE PLAN ===\n{standing_plan}\n\n"
        if standing_plan else ""
    )
    situation_section = (
        f"=== SITUATION AT YOUR LAST CHECKPOINT ===\n{situation}\n\n" if situation else ""
    )

    return ADMIRAL_RESPONSE_PROMPT.format(
        decision_interval_s=decision_interval_s,
        admiral_name=admiral_name,
        captain_ship_name=captain_ship_name,
        standing_plan_section=standing_plan_section,
        situation_section=situation_section,
        recent_orders_context=recent_orders_context,
        personality_prompt=personality_prompt,
    )


def _ship_row(snapshot: Any, ship_id: Optional[str], ship_name: str) -> Any:
    for row in getattr(snapshot, "friendly_ships", None) or []:
        if (ship_id and row.ship_id == ship_id) or row.ship_name == ship_name:
            return row
    return None


def build_admiral_ship_order_messages(
    admiral_name: str,
    ship_name: str,
    ship_type: str,
    captain_name: str,
    fleet_directive: str,
    snapshot: Any,
    personality: Optional[str] = None,
    standing_plan: Optional[str] = None,
    fleet_heading: Any = None,
    ship_id: Optional[str] = None,
    ship_view: Optional[Dict[str, Any]] = None,
    captain_report: Optional[Dict[str, Any]] = None,
    directive_failure: Optional[str] = None,
    previous_directive: Optional[Tuple[float, str]] = None,
    faction: str = "",
) -> List[Dict[str, str]]:
    """
    The per-ship order request as [system, user].

    system: battle-static order instructions. user: this checkpoint's plan and
    directive, then the ship in ITS OWN frame - status, its captain's last
    report, rounds inbound on it, every enemy's range/closure/torpedo verdict
    measured from it (not from whichever friendly hull is nearest, R6), and
    its own range-control headings.
    """
    view = ship_view or {}
    row = _ship_row(snapshot, ship_id, ship_name)
    ship_id = ship_id or (row.ship_id if row is not None else None)
    has_tubes = bool(row is not None and (row.launchers or row.torpedoes_remaining))
    torpedo_note = (" - launch torpedoes only on a GOOD verdict; POOR rounds arrive slow and weak,"
                    " NO rounds are wasted." if has_tubes else ".")
    system = ADMIRAL_SHIP_ORDER_DOCTRINE.format(
        admiral_name=admiral_name, faction=(faction or "your").upper(),
        torpedo_note=torpedo_note,
        personality_line=f"\nYOUR PERSONALITY: {personality}" if personality else "")

    parts = []
    if standing_plan:
        parts.append(f"YOUR STANDING BATTLE PLAN (keep this order consistent with it):\n\"{standing_plan}\"")
    if fleet_directive:
        parts.append(f"You have already set the fleet directive:\n\"{fleet_directive}\"")
    else:
        why = f" - the directive call failed ({directive_failure})" if directive_failure else ""
        text = (f"NO FLEET DIRECTIVE THIS CHECKPOINT{why}. Captains are told there is no new "
                "directive, so this order is all the new guidance this ship gets: make it complete.")
        if previous_directive:
            t, prev = previous_directive
            text += (f"\nYour last directive (T+{t:.0f}s - STALE, for your reference only; it was "
                     f"not reissued as this checkpoint's directive): \"{_clip(prev, 400)}\"")
        parts.append(text)
    heading = getattr(fleet_heading, "heading", None)
    if isinstance(heading, dict):
        thr = getattr(fleet_heading, "throttle", 1.0)
        thr = 1.0 if thr is None else thr
        parts.append(
            f"FLEET HEADING IN FORCE (your \"all ships\" order): "
            f"({heading.get('x', 0):+.2f},{heading.get('y', 0):+.2f},{heading.get('z', 0):+.2f}) "
            f"at {thr * 100:.0f}% of the slowest hull. {ship_name} flies it (throttle scaled "
            f"to hold formation) unless you give it its own heading now.")

    label = _lbl(snapshot, ship_id, ship_name) if ship_id else ship_name
    if row is not None:
        status = [f"SHIP STATUS - {label} ({ship_type.replace('_', ' ').title()}), captain {captain_name}:",
                  f"  hull {row.hull_integrity:.0f}%{_heat_compact(row)}, dv {row.delta_v_remaining:.0f} km/s, "
                  f"{row.velocity_kps:.1f} km/s, PD {row.pd_turrets_operational}/{row.pd_turrets_total}"]
        if row.launchers or row.torpedoes_remaining:
            status.append(f"  torpedoes: {row.torpedoes_remaining} left, "
                          f"{view.get('launches_available', row.launches_available)} launchable before the next checkpoint")
        status.append(f"  weapons: {row.weapons_summary}"
                      + (f" | LOST: {', '.join(row.weapons_destroyed)}" if row.weapons_destroyed else ""))
        doing = row.current_maneuver + (f" -> {_lbl(snapshot, row.current_target)}" if row.current_target else "")
        status.append(f"  now flying: {doing}"
                      + (f" | aimed at by {', '.join(row.targeted_by)}" if row.targeted_by else ""))
        parts.append("\n".join(status))
    else:
        parts.append(f"SHIP STATUS - {label} ({ship_type.title()}), captain {captain_name}")

    if captain_report:
        report = _format_captain_reports([captain_report], snapshot).split("\n", 1)
        parts.append("ITS CAPTAIN'S LAST REPORT: " + (report[1] if len(report) > 1 else "(empty)"))
    elif captain_report is None and getattr(snapshot, "checkpoint", 0) and snapshot.checkpoint > 1:
        parts.append("ITS CAPTAIN'S LAST REPORT: none filed")

    inbound = view.get("inbound") or []
    if inbound:
        parts.append("INBOUND ON THIS SHIP:\n" + "\n".join(_inbound_lines(snapshot, inbound, "  ")))
    elif view:
        parts.append("INBOUND ON THIS SHIP: nothing")

    rows = view.get("enemies") or []
    if rows:
        lines = [f"ENEMIES FROM {label} (nearest first; + closing, - opening):"]
        for e in rows[:8]:
            text = (f"  {_lbl(snapshot, e['id'])} {_type_abbr(snapshot, e['id'])}: {e['range_km']:.0f} km "
                    f"{e['closing_kps']:+.1f} km/s, bearing {e['bearing']}")
            if e.get("solution") is not None:
                text += f" | torpedo {_solution_text(e['solution'])}"
            lines.append(text)
        if len(rows) > 8:
            lines.append(f"  (+{len(rows) - 8} more further out)")
        parts.append("\n".join(lines))
    elif snapshot is not None and getattr(snapshot, "enemy_ships", None):
        # No ship-frame data: say what the numbers are, never pass them off as this ship's.
        lines = ["ENEMY POSITIONS (range to YOUR NEAREST hull, not to this ship):"]
        for enemy in snapshot.enemy_ships:
            pos = enemy.position_km
            lines.append(f"  {_lbl(snapshot, enemy.ship_id, enemy.ship_name)} ({enemy.ship_type}): "
                         f"({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) km, "
                         f"{enemy.distance_from_closest_friendly_km:.0f} km")
        parts.append("\n".join(lines))
    if view.get("range_control"):
        parts.append(view["range_control"])
    parts.append(f"CALL issue_order NOW for {ship_name}.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)}]


def build_admiral_ship_order_prompt(
    admiral_name: str,
    ship_name: str,
    ship_type: str,
    captain_name: str,
    fleet_directive: str,
    snapshot: Any,
    personality: Optional[str] = None,
    standing_plan: Optional[str] = None,
    fleet_heading: Any = None,
    **kwargs: Any,
) -> str:
    """Single-string form of build_admiral_ship_order_messages (system + user)."""
    messages = build_admiral_ship_order_messages(
        admiral_name=admiral_name, ship_name=ship_name, ship_type=ship_type,
        captain_name=captain_name, fleet_directive=fleet_directive, snapshot=snapshot,
        personality=personality, standing_plan=standing_plan, fleet_heading=fleet_heading,
        **kwargs)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]


def _age(issued_at: Optional[float], now: Optional[float]) -> str:
    """'issued T+180, this checkpoint' / 'issued T+150, 30 s old'."""
    if issued_at is None:
        return ""
    if now is None or now - issued_at < 1.0:
        return f"issued T+{issued_at:.0f}, this checkpoint"
    return f"issued T+{issued_at:.0f}, {now - issued_at:.0f} s old"


def _is_clarification(order: Any) -> bool:
    return str(getattr(order, "order_text", "")).lstrip().startswith("[CLARIFICATION]")


def format_admiral_orders_for_captain(
    orders: List[Any],
    fleet_directive: Optional[str],
    now: Optional[float] = None,
    checks: Optional[List[Dict[str, Any]]] = None,
    reply_tool: str = "respond_to_orders",
    directive_issued_at: Optional[float] = None,
    previous_directive: Optional[Tuple[str, float]] = None,
    previous_orders: Optional[Tuple[List[Any], float]] = None,
) -> str:
    """
    The ORDERS block of a captain's turn.

    Each order is rendered as fields with its age: priority + verbatim text,
    then engine-resolved lines under it (computed by the captain from live
    state, passed in ``checks`` parallel to ``orders``):
      target  - the named target with the captain's OWN sensor range/closure
      salvo   - SALVO CHECK from a flown torpedo (UNREACHABLE / OVERKILL ...)
      engine  - ENGINE ORDER CHECK of the ordered maneuver/heading
      points  - absolute positions quoted in the text, resolved to dir_to
      numbers - distances in the text that disagree with the sensors
      conflict- the order fights a RANGE CONTROL merge warning
    A CLARIFICATION (discuss_with_admiral) supersedes the orders before it;
    those are collapsed instead of left standing next to their correction.
    An empty directive means the admiral issued none THIS checkpoint; the
    last one seen is shown with its age instead.

    The old block closed with ~25 lines of "YOU MUST RESPOND / failing your
    Admiral / call set_weapons_order" boilerplate - the protocol now lives
    in the cached doctrine, and acting on the orders is the acknowledgment.
    """
    checks = list(checks or [])
    lines = ["ORDERS FROM YOUR ADMIRAL" + (f" (T+{now:.0f})" if now is not None else "") + ":"]

    if fleet_directive:
        age = _age(directive_issued_at if directive_issued_at is not None else now, now)
        lines.append(f"  FLEET DIRECTIVE{' (' + age + ')' if age else ''}: {fleet_directive}")
    elif fleet_directive is not None or previous_directive:
        line = "  FLEET DIRECTIVE: (no new fleet directive this checkpoint)"
        if previous_directive:
            text, t = previous_directive
            line += f" - last one ({_age(t, now)}): {text}"
        lines.append(line)

    clar_idx = max((i for i, o in enumerate(orders or []) if _is_clarification(o)), default=-1)
    applied: List[str] = []  # admiral motion the runner already put on the ship
    if orders:
        for i, order in enumerate(orders):
            check = checks[i] if i < len(checks) else {}
            if i < clar_idx and not _is_clarification(order):
                text = " ".join(str(order.order_text).split())
                lines.append(f"  (superseded by the clarification below) [{order.priority}] "
                             f"{text[:80]}{'...' if len(text) > 80 else ''}")
                continue
            marker = "!!!" if order.priority == "CRITICAL" else ("!!" if order.priority == "HIGH" else "")
            issued = getattr(order, "issued_at", None)
            age = _age(issued if issued is not None else now, now)
            note = ""
            if _is_clarification(order) and i > 0:
                note = " - REPLACES the orders above"
            lines.append(f"  {marker}[{order.priority}]{' (' + age + note + ')' if age or note else ''} "
                         f"{order.order_text}")
            if check.get("target"):
                lines.append(f"      >>> TARGET: {check['target']}")
            elif order.suggested_target:
                lines.append(f"      >>> TARGET: {order.suggested_target}")
            # Coordinated salvo: several ships launching together is how a fleet
            # overwhelms point defense, so state it explicitly rather than
            # leaving it buried in the order prose - and check it can connect.
            salvo = getattr(order, "torpedo_salvo", None)
            if salvo:
                tgt = (getattr(order, "torpedo_target", None) or order.suggested_target)
                at = f" at {tgt}" if tgt else ""
                lines.append(
                    f"      >>> COORDINATED SALVO: launch {salvo} torpedo(es){at} "
                    f"THIS CHECKPOINT, timed with the rest of the fleet.")
                if check.get("salvo"):
                    lines.append(f"          SALVO CHECK: {check['salvo']}")
            heading = getattr(order, "heading", None)
            if isinstance(heading, dict):
                applied.append("HEADING ORDER")
                thr = getattr(order, "applied_throttle", None)
                if thr is None:
                    thr = getattr(order, "throttle", None)
                thr = 1.0 if thr is None else thr
                formation = (" [FORMATION: throttle matched to the slowest hull, burn "
                             "starts with the fleet]" if getattr(order, "formation", False) else "")
                what = ("attitude only - nose onto it, no burn" if thr <= 0
                        else f"at {thr * 100:.0f}% throttle")
                lines.append(
                    f"      >>> HEADING ORDER (ALREADY APPLIED to your ship): "
                    f"({heading.get('x', 0):+.2f},{heading.get('y', 0):+.2f},"
                    f"{heading.get('z', 0):+.2f}) {what}{formation}. Keep it: calling "
                    f"set_maneuver/set_heading replaces it. Override only for a reason "
                    f"(e.g. EVADE an inbound torpedo).")
            # [D maneuver] structured admiral maneuver (AdmiralOrder.maneuver:
            # INTERCEPT with standoff, HOLD_RANGE, ...): ALREADY APPLIED by the
            # runner, or NOT APPLIED with the reason.
            from .maneuver_orders import admiral_maneuver_lines, admiral_order_moves_ship
            lines.extend(admiral_maneuver_lines(order))
            if (getattr(order, "maneuver", None) and not isinstance(heading, dict)
                    and admiral_order_moves_ship(order)):
                applied.append("MANEUVER ORDER")
            if check.get("engine"):
                lines.append(f"      >>> ENGINE ORDER CHECK: {check['engine']}")
            for point in check.get("points") or []:
                lines.append(f"      >>> POSITION IN ORDER: {point}")
            for num in check.get("numbers") or []:
                lines.append(f"      >>> NUMBER CHECK: {num}")
            if check.get("conflict"):
                lines.append(f"      !! {check['conflict']}")
    elif previous_orders and previous_orders[0]:
        prev, t = previous_orders
        lines.append(f"  No new orders for your ship this checkpoint. Last orders ({_age(t, now)}), "
                     "still your standing orders unless the directive changed:")
        for order in prev[-2:]:
            text = " ".join(str(order.order_text).split())
            lines.append(f"    [{order.priority}] {text[:200]}{'...' if len(text) > 200 else ''}")
    else:
        lines.append("  !!! NO ORDERS RECEIVED FOR YOUR SHIP: follow the fleet directive, or ask with "
                     "discuss_with_admiral (\"Admiral, I received no orders. What are my "
                     "instructions?\"); otherwise use your best judgment.")

    reply = (f"  REPLY: acting on these orders with your tools IS your acknowledgment. Call "
             f"{reply_tool} only to DEVIATE or report UNABLE (say why), or to send a request or "
             f"contact report. A CHECK above that says the order fails is a reason to DEVIATE.")
    if applied:
        reply += (f" Your {'/'.join(dict.fromkeys(applied))} is already flying: no maneuver call "
                  "needed unless you deviate - still set your target and weapons/launches.")
    lines.append(reply)
    return "\n".join(lines)


def build_admiral_messages(**kwargs) -> List[Dict[str, str]]:
    """
    Build the admiral request as a cacheable system prompt plus a volatile user turn.

    system: battle-static doctrine composed for the fleets in this battle plus
    the per-class FORCES table - byte-identical across checkpoints, so the
    provider serves it from cache. user: the live turn state. (The old single
    system string put live state behind the cache breakpoint: battle A's
    admiral wrote 381k cache tokens and read none - audit ADM-11.)

    Accepts the same keyword arguments as build_admiral_prompt().
    """
    static, live = _admiral_parts(**kwargs)
    return [
        {"role": "system", "content": static.rstrip()},
        {"role": "user", "content": live.rstrip()},
    ]
