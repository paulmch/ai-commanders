"""
Fleet draft mode: admirals build their own fleets from a point budget.

Flow (per side, before the battle):
  1. SELECTION - the admiral sees a costed ship catalog and a point budget
     and calls select_fleet with the hulls it wants.
  2. FORMATION - the admiral places its drafted ships in a starting
     formation (offsets in km from its fleet anchor) via set_formation.

The drafted fleet is then flown by cheap captains - the rule-based
HeuristicCaptain by default, or any (cheap) LLM model - while the admiral
keeps issuing orders every checkpoint like in any fleet battle.

Both phases run through the normal tool-calling client with a bounded
validation loop: an illegal draft (over budget, unknown hull) is fed back
as an error message and retried, and after too many failures the side falls
back to a deterministic auto-draft so a battle always starts.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .client import LLMCallError
from .torpedo_doctrine import LINK_FACT, PROFILE_FACT

try:
    from ..drive_plume import formation_plume_conflicts
except ImportError:  # pragma: no cover - script-style import path
    from drive_plume import formation_plume_conflicts

# ---------------------------------------------------------------------------
# Point costs
# ---------------------------------------------------------------------------

# Costs are COMPUTED from data/fleet_ships.json, not hand-typed, so a hull
# that gets more armour or a fatter magazine gets more expensive on its own.
#
# A hull pays for three things:
#   platform  - armour tonnage (what it takes to kill it) plus PD turrets
#   guns      - sustained energy throughput of its gun mounts (GJ/s)
#   torpedoes - the magazine's DELIVERED ENERGY: rounds x per-round yield
#
# That last term is the 2026-08-13 rebalance. Torpedo armament used to be
# almost free: cruiser_torpedo cost 30 pts against the gun cruiser's 26 for
# the same hull, same 241cm nose and DOUBLE the PD - so 48 guided rounds
# came to 4 points. But a Trident is not a dumb slug. It is a guided
# fusion-torch penetrator that steers all the way in, and at the engine's
# enforced 12 km/s closure floor (MIN_CLOSING_SPEED_KPS in src/torpedo.py)
# a 250 kg penetrator lands 18 GJ - 4.2x a spinal coiler slug, 25x a
# coilgun round - rising past 40 GJ head-on. Priced per GJ/s of sustained
# output the old table read 5.0 pts per GJ/s for the torpedo cruiser
# against 87.7 for the dreadnought: a 17.5x efficiency gap that made
# saturation doctrine the only rational draft at every budget (two frontier
# admirals independently drafted 5x cruiser_torpedo at 150 pts, and the one
# that deviated was swept 0-for-7).
#
# So the magazine is now priced on what it actually delivers.
ARMOR_TONS_PER_POINT = 55.0
PD_POINTS_PER_TURRET = 1.0
GUN_POINTS_PER_GJ_PER_S = 8.0
# Points per GJ of stowed torpedo warhead energy, i.e. per round x yield.
#
# History of the dial: 0.032 (2026-08-13), 0.040 (2026-09-26, re-checked
# 2026-09-27) - both fitted against the old heuristic captain, which launched
# about 14 of a torpedo cruiser's 48 rounds in a battle.
#
# 2026-11-15 (pass 5): 0.040 -> 0.102, i.e. 0.72 -> 1.84 pts per stowed
# round. The tactical captains mass and synchronise salvos and empty every
# magazine in the first 90-150 s, before a gun fleet starting 500 km away has
# fired a round. Measured with scripts/draft_tournament.py (seeded round
# robin, both side assignments), at the old prices:
#   - 144 pts of corvettes (8 hulls, 64 rounds) annihilated every 150-pt gun
#     fleet without loss; so did 108 pts (6 hulls), and 54 pts (3 hulls, 24
#     rounds) still destroyed 76-147 pts of each;
#   - a launched round destroyed 2.2-6.1 pts of gun hulls (2.2 with rounds
#     left over, 6.1 for three boats against a battle line, 3-4 typically), at
#     150 and at 200 pts alike, while it cost 0.72 pts - 2.25 with its share
#     of the corvette around it.
# The magazine is therefore priced so that a torpedo hull costs about what
# its rounds destroy, at the LOW end of that range (gun fleets are about to
# close better, and a torpedo cruiser is one hull to lose): 8 rounds on a
# corvette come to 27 pts all-in (3.4 per round), 48 on a torpedo cruiser to
# 119 (2.5 per round). 0.102 sits inside the band (0.0993-0.1060) in which
# neither price moves and the 150-pt torpedo doctrine stays "one torpedo
# cruiser plus a corvette".
#
# What this dial can NOT do is balance the verdicts: a corvette salvo is a
# step function (two boats' salvos are blinded and miss, three boats' kill a
# battle line) and the boats are never caught afterwards - see the
# "2026-11-15" section of docs/draft_mode.md before retuning.
#
# 2026-11-15, round 2: 0.102 -> 0.080 (cruiser_torpedo 119 -> 100, corvette
# 27 -> 24). A hard constraint from the owner's side: EVERY hull must be
# draftable at DEFAULT_POINT_BUDGET (100), and a 119-pt torpedo cruiser was
# not. 0.080 is the dearest dial that keeps it at 100 (the band is
# 0.0797-0.0808; tests/test_draft_tournament.py pins the constraint). It is
# BELOW what the magazine destroys: on the merged engine (gun ships close, a
# dry fleet surrenders or is ruled out, the verdict is scored at draft
# prices) a torpedo hull with a few frigates still beats every gun fleet.
# That needs point defence and range to bite, not a price.
TORPEDO_POINTS_PER_GJ = 0.080


def _load_ship_specs() -> Dict[str, Any]:
    """Fleet data for pricing only - a local read, so importing this module
    never pulls in the battle runner."""
    path = Path(__file__).parent.parent.parent / "data" / "fleet_ships.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def torpedo_round_energy_gj(fleet_data: Dict[str, Any]) -> float:
    """
    Energy one torpedo delivers on the guaranteed-worst-case intercept.

    The Trident is a kinetic penetrator (warhead_yield_gj is 0), so its
    "explosive factor" is KE = 1/2 m v^2 at the guidance law's enforced
    closure floor. Any explosive yield, if a future round has one, adds on
    top.
    """
    spec = fleet_data.get("weapon_types", {}).get("torpedo_launcher", {})
    mass_kg = spec.get("penetrator_mass_kg", 250.0)
    # Mirrors MIN_CLOSING_SPEED_KPS in src/torpedo.py - the floor the
    # guidance law holds, so it is the least a round can be worth.
    closure_ms = 12.0 * 1000.0
    kinetic_gj = 0.5 * mass_kg * closure_ms ** 2 / 1e9
    return kinetic_gj + spec.get("warhead_yield_gj", 0.0)


def _mount_throughput_gj_per_s(weapon_type: str, fleet_data: Dict[str, Any]) -> float:
    """Sustained energy output of one gun mount, in GJ/s."""
    spec = fleet_data.get("weapon_types", {}).get(weapon_type, {})
    salvo = spec.get("salvo_size", 1)
    cooldown = spec.get("cooldown_s", 0) + (salvo - 1) * spec.get(
        "intra_salvo_cooldown_s", 0)
    if cooldown <= 0:
        return 0.0
    return spec.get("kinetic_energy_gj", 0.0) * salvo / cooldown


def ship_point_cost(ship_type: str, fleet_data: Dict[str, Any]) -> int:
    """Draft cost of one hull, derived from its own stats."""
    spec = fleet_data.get("ships", {}).get(ship_type)
    if not spec:
        return 0
    armor_tons = spec.get("mass_breakdown", {}).get("armor", 0.0)
    points = armor_tons / ARMOR_TONS_PER_POINT

    round_gj = torpedo_round_energy_gj(fleet_data)
    for weapon in spec.get("weapons", []):
        wtype = weapon.get("type")
        if wtype == "pd_laser":
            points += PD_POINTS_PER_TURRET
        elif wtype == "torpedo_launcher":
            rounds = weapon.get("magazine", fleet_data.get("weapon_types", {})
                                .get("torpedo_launcher", {}).get("magazine", 16))
            points += rounds * round_gj * TORPEDO_POINTS_PER_GJ
        elif wtype:
            points += (_mount_throughput_gj_per_s(wtype, fleet_data)
                       * GUN_POINTS_PER_GJ_PER_S)
    return max(1, round(points))


def build_ship_point_costs(fleet_data: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, int]:
    """The costed catalog: every hull in fleet data, priced by the model."""
    data = fleet_data if fleet_data is not None else _load_ship_specs()
    return {stype: ship_point_cost(stype, data) for stype in data.get("ships", {})}


SHIP_POINT_COSTS: Dict[str, int] = build_ship_point_costs()

DEFAULT_POINT_BUDGET = 100
DEFAULT_MAX_SHIPS = 8
FORMATION_MAX_OFFSET_KM = 150.0
MIN_SEPARATION_KM = 2.0

# Ship-class nicknames used to build readable ship names ("TIS Falchion-2"
# beats "TIS Heuristic-2" in every log, plot and recording).
CLASS_NICKNAMES: Dict[str, str] = {
    "corvette": "Dart",
    "frigate": "Ward",
    "destroyer": "Falchion",
    "battlecruiser": "Lancer",
    "cruiser": "Bastion",
    "cruiser_torpedo": "Harpoon",
    "battleship": "Sovereign",
    "dreadnought": "Colossus",
    "dreadnought_siege": "Breaker",
}


@dataclass
class DraftedShip:
    """One drafted hull with its resolved identity and formation slot."""
    ship_type: str
    ship_id: str
    ship_name: str
    # Formation offset from the fleet anchor, admiral-local frame:
    # +x toward the enemy, y lateral, z vertical. Kilometers.
    offset_km: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FleetDraft:
    """Complete result of one side's draft."""
    faction: str
    budget: int
    ships: List[DraftedShip] = field(default_factory=list)
    points_spent: int = 0
    formation_name: str = "line abreast"
    selection_rationale: str = ""
    formation_rationale: str = ""
    auto: bool = False  # True if the fallback drafted (LLM failed/absent)
    # Set when the model's last invalid fleet was trimmed to validity
    # (trim_selection): {"submitted": [...], "dropped": [...], "kept": [...]}.
    trimmed: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def _weapons_summary(ship_spec: Dict[str, Any], fleet_data: Dict[str, Any]) -> str:
    weapon_types = fleet_data.get("weapon_types", {})
    counts: Dict[str, int] = {}
    launchers = 0
    for weapon in ship_spec.get("weapons", []):
        wtype = weapon.get("type", "")
        if wtype.startswith("pd_"):
            continue
        if wtype == "torpedo_launcher":
            launchers += 1
            continue
        wspec = weapon_types.get(wtype, {})
        name = wspec.get("name", wtype)
        damage = wspec.get("kinetic_energy_gj", 0)
        label = f"{name} ({damage:.1f} GJ muzzle)" if damage else name
        counts[label] = counts.get(label, 0) + 1
    parts = [f"{n}x {label}" for label, n in counts.items()]
    if launchers:
        parts.append(f"{launchers}x torpedo launcher")
    return ", ".join(parts) if parts else "unarmed"


def _torpedo_magazine(ship_spec: Dict[str, Any], fleet_data: Dict[str, Any]) -> int:
    """
    Total torpedo rounds the sim will actually load: the per-launcher
    "magazine" on each weapon entry, with the same fallback chain the
    simulation's ship factory uses. (The modules/magazines block is
    unreliable - the torpedo cruiser's is stale gun-cruiser data.)
    """
    default = (fleet_data.get("weapon_types", {})
               .get("torpedo_launcher", {}).get("magazine", 16))
    return sum(
        w.get("magazine", default)
        for w in ship_spec.get("weapons", [])
        if w.get("type") == "torpedo_launcher"
    )


def build_catalog_text(fleet_data: Dict[str, Any]) -> str:
    """Human/LLM-readable costed catalog of every draftable hull."""
    round_gj = torpedo_round_energy_gj(fleet_data)
    lines = [
        "SHIP CATALOG (cost in points | one line per hull class):",
        "",
        "PRICING: a hull costs armour tonnage + PD turrets + weapon energy. "
        f"Torpedoes are priced per round on delivered yield ({round_gj:.0f} GJ "
        "each at the 12 km/s closure floor, vs 4.3 GJ for a spinal coiler "
        "slug), so magazine depth is the single biggest line item on a "
        "torpedo hull. Saturation is buyable, but you pay for every round.",
        "",
    ]
    for ship_type, cost in SHIP_POINT_COSTS.items():
        spec = fleet_data.get("ships", {}).get(ship_type)
        if not spec:
            continue
        perf = spec.get("performance", {})
        sections = spec.get("armor", {}).get("sections", {})
        armor = "/".join(
            f"{sections.get(loc, {}).get('thickness_cm', 0):.0f}"
            for loc in ("nose", "lateral", "tail"))
        pd = sum(1 for w in spec.get("weapons", []) if w.get("type") == "pd_laser")
        mag = _torpedo_magazine(spec, fleet_data)
        # Role text is the designer's own briefing on how the hull fights -
        # truncating it once cut the torpedo cruiser's line mid-sentence,
        # hiding its defining stat (8 rounds per decision) from buyers.
        role = spec.get("role", "")
        lines.append(
            f"- {ship_type} [{cost} pts] {perf.get('combat_acceleration_g', 0)}g, "
            f"armor {armor}cm (nose/lat/tail), {pd}x PD laser"
            + (f", {mag} torpedoes" if mag else "")
            + f" | {_weapons_summary(spec, fleet_data)} | {role}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Draft tools
# ---------------------------------------------------------------------------

def select_fleet_tool(budget: Optional[int] = None, max_ships: Optional[int] = None) -> Dict[str, Any]:
    """
    The select_fleet tool. The price sum comes FIRST in the schema
    (`budget_check`), before `ships`: Sonnet 5.5 wrote `ships` first and
    added up afterwards ("58+50+36+14 = 158 is too high, so verify cost") -
    6 of its 10 submissions were invalid and one battle never started. With
    a budget and cap the limits are stated in the description and bounded in
    the schema where JSON schema can (hull names, per-line count <= cap; the
    SUM of counts cannot be expressed there).
    """
    limits = (f"Limits: total cost <= {budget} points AND at most {max_ships} hulls "
              f"(the sum of all counts). " if budget is not None and max_ships is not None else
              "Total cost must not exceed your point budget. ")
    count: Dict[str, Any] = {"type": "integer", "minimum": 1}
    if max_ships is not None:
        count["maximum"] = int(max_ships)
    hull: Dict[str, Any] = {"type": "string"}
    if SHIP_POINT_COSTS:
        hull["enum"] = list(SHIP_POINT_COSTS)
    return {
        "type": "function",
        "function": {
            "name": "select_fleet",
            "description": (
                "Buy your fleet from the catalog. " + limits
                + "Unspent points are wasted. You must buy at least one ship. Fill "
                "budget_check first, then ships with exactly the hulls you added up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_check": {
                        "type": "string",
                        "description": (
                            "Write this FIRST: every hull line with count x price, the "
                            "point total and the hull count against the limits, e.g. "
                            "'2 battleship 2x43=86 + 3 destroyer 3x18=54 = 140 of "
                            f"{budget if budget is not None else 150} points; 5 of "
                            f"{max_ships if max_ships is not None else 8} hulls'"),
                    },
                    "ships": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {"ship_type": hull, "count": count},
                            "required": ["ship_type", "count"],
                        },
                        "description": "Hull classes and how many of each to buy (as in budget_check)",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One or two sentences on the fleet concept",
                    },
                },
                "required": ["budget_check", "ships"],
            },
        },
    }


SELECT_FLEET_TOOL = {
    "type": "function",
    "function": {
        "name": "select_fleet",
        "description": (
            "Buy your fleet from the catalog. Total cost must not exceed your "
            "point budget; unspent points are wasted. You must buy at least "
            "one ship."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ship_type": {"type": "string"},
                            "count": {"type": "integer", "minimum": 1},
                        },
                        "required": ["ship_type", "count"],
                    },
                    "description": "Hull classes and how many of each to buy",
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences on the fleet concept",
                },
            },
            "required": ["ships"],
        },
    },
}

SET_FORMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "set_formation",
        "description": (
            "Place your drafted ships in their battle-start formation. "
            "Offsets are km from your fleet anchor in YOUR frame: +x points "
            "at the enemy fleet, y is lateral, z is vertical. Offsets are "
            f"limited to +/-{FORMATION_MAX_OFFSET_KM:.0f} km per axis. Ships "
            "you do not place get default line-abreast slots."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "placements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ship_name": {"type": "string"},
                            "x_km": {"type": "number"},
                            "y_km": {"type": "number"},
                            "z_km": {"type": "number"},
                        },
                        "required": ["ship_name", "x_km", "y_km"],
                    },
                },
                "formation_name": {
                    "type": "string",
                    "description": "Short name for the formation (e.g. 'PD wall')",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this shape (screening, PD overlap, flanking...)",
                },
            },
            "required": ["placements"],
        },
    },
}


# ---------------------------------------------------------------------------
# Validation / resolution
# ---------------------------------------------------------------------------

def validate_selection(
    ships_arg: List[Dict[str, Any]],
    budget: int,
    max_ships: int,
) -> Tuple[Optional[List[str]], int, Optional[str]]:
    """
    Validate a select_fleet payload.

    Returns (flat ship_type list, points_spent, error). Exactly one of
    list/error is set.
    """
    if not isinstance(ships_arg, list) or not ships_arg:
        return None, 0, "You must buy at least one ship (ships array was empty)."

    entries: List[Tuple[str, int]] = []
    spent = 0
    n_ships = 0
    for entry in ships_arg:
        if not isinstance(entry, dict):
            return None, 0, "Each ships entry must be an object with ship_type and count."
        stype = str(entry.get("ship_type", "")).strip().lower()
        count = entry.get("count", 1)
        if stype not in SHIP_POINT_COSTS:
            return None, 0, (
                f"Unknown hull '{stype}'. Valid hulls: "
                f"{', '.join(SHIP_POINT_COSTS)}.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return None, 0, f"Invalid count {count!r} for {stype}."
        # Tally only; the list is expanded after the limit checks so an
        # external count like 10**100 cannot allocate a huge list.
        entries.append((stype, count))
        spent += SHIP_POINT_COSTS[stype] * count
        n_ships += count

    lines = " + ".join(f"{count}x {stype} ({count}x{SHIP_POINT_COSTS[stype]}="
                       f"{count * SHIP_POINT_COSTS[stype]})" for stype, count in entries)
    echo = f" You submitted: {lines} = {spent} points, {n_ships} hulls."
    problems = []
    if spent > budget:
        problems.append(f"Over budget: that fleet costs {spent} points but you only have "
                        f"{budget} ({spent - budget} over). Drop or downgrade ships.")
    if n_ships > max_ships:
        problems.append(f"Too many hulls: {n_ships} ships, the cap is {max_ships} "
                        f"({n_ships - max_ships} too many).")
    if problems:
        return None, 0, " ".join(problems) + echo
    flat = [stype for stype, count in entries for _ in range(count)]
    return flat, spent, None


def trim_selection(ships_arg: Any, budget: int, max_ships: int
                   ) -> Tuple[Optional[List[str]], int, List[str]]:
    """
    The last fleet a model submitted, trimmed to validity: unknown hulls and
    malformed lines are dropped, then the CHEAPEST hulls one by one until the
    fleet fits both the point budget and the hull cap. Returns (flat ship
    types, points, dropped hull types); (None, 0, dropped) when nothing is
    left. Deterministic: ties drop the later line first.
    """
    dropped: List[str] = []
    picked: List[str] = []
    for entry in ships_arg if isinstance(ships_arg, list) else []:
        if not isinstance(entry, dict):
            continue
        stype = str(entry.get("ship_type", "")).strip().lower()
        count = entry.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            continue
        if stype not in SHIP_POINT_COSTS:
            dropped.extend([stype or "?"] * min(count, max_ships + 1))
            continue
        picked.extend([stype] * min(count, 10 * max(1, max_ships)))
    while picked and (sum(SHIP_POINT_COSTS[t] for t in picked) > budget or len(picked) > max_ships):
        i = min(range(len(picked)), key=lambda k: (SHIP_POINT_COSTS[picked[k]], -k))
        dropped.append(picked.pop(i))
    if not picked:
        return None, 0, dropped
    return picked, sum(SHIP_POINT_COSTS[t] for t in picked), dropped


# Beta's own names (lane cmdcap, comms audit M3): both fleets drew from one
# pool, so "TIS Lancer-1" faced "OCS Lancer-1" and a "the name also matches
# your own" warning rode on up to 264 of 279 captain turns of a battle. Used
# for commander (LLM admiral) drafts only: the seeded auto-draft battles
# (perf_check, tournaments, heuristic runs) keep the shared pool, so their
# event streams are unchanged; recordings, fixtures and configs keep the
# names they carry, and the collision handling stays for those.
BETA_CLASS_NICKNAMES: Dict[str, str] = {
    "corvette": "Stiletto",
    "frigate": "Picket",
    "destroyer": "Saber",
    "battlecruiser": "Halberd",
    "cruiser": "Rampart",
    "cruiser_torpedo": "Marlin",
    "battleship": "Monarch",
    "dreadnought": "Leviathan",
    "dreadnought_siege": "Sledge",
}


def name_drafted_ships(ship_types: List[str], faction: str,
                       faction_pool: bool = False) -> List[DraftedShip]:
    """Assign ids and readable names ("TIS Falchion-2") to drafted hulls;
    faction_pool: beta draws BETA_CLASS_NICKNAMES ("OCS Saber-2")."""
    prefix = "TIS" if faction == "alpha" else "OCS"
    pool = BETA_CLASS_NICKNAMES if faction_pool and faction != "alpha" else CLASS_NICKNAMES
    counters: Dict[str, int] = {}
    drafted = []
    for i, stype in enumerate(ship_types):
        nickname = pool.get(stype) or CLASS_NICKNAMES.get(stype, stype.title())
        counters[nickname] = counters.get(nickname, 0) + 1
        n = counters[nickname]
        # Only number when a class repeats, matching fleet naming elsewhere
        suffix = f"-{n}" if ship_types.count(stype) > 1 else ""
        drafted.append(DraftedShip(
            ship_type=stype,
            ship_id=f"{faction}_{i + 1}",
            ship_name=f"{prefix} {nickname}{suffix}",
        ))
    return drafted


def default_formation_offsets(n: int) -> List[Tuple[float, float, float]]:
    """Line abreast, centered, 8 km lateral spacing."""
    return [(0.0, (i - (n - 1) / 2) * 8.0, 0.0) for i in range(n)]


def apply_formation(
    drafted: List[DraftedShip],
    placements: List[Dict[str, Any]],
) -> List[str]:
    """
    Resolve set_formation placements onto drafted ships (in place).

    Offsets are clamped to the legal cube and nudged apart if two ships sit
    closer than MIN_SEPARATION_KM. Unplaced ships keep default slots.
    Returns a list of human-readable adjustment notes.
    """
    notes: List[str] = []
    by_name = {s.ship_name.lower(): s for s in drafted}
    defaults = default_formation_offsets(len(drafted))
    for ship, off in zip(drafted, defaults):
        ship.offset_km = off

    for p in placements or []:
        raw_name = str(p.get("ship_name", "")).strip()
        ship = by_name.get(raw_name.lower())
        if ship is None:
            # Tolerate partial names ("Falchion-2" for "TIS Falchion-2")
            matches = [s for key, s in by_name.items() if raw_name.lower() in key]
            if len(matches) == 1:
                ship = matches[0]
            else:
                notes.append(f"ignored placement for unknown ship '{raw_name}'")
                continue
        try:
            x = float(p.get("x_km", 0.0))
            y = float(p.get("y_km", 0.0))
            z = float(p.get("z_km", 0.0))
        except (TypeError, ValueError):
            notes.append(f"ignored non-numeric placement for {ship.ship_name}")
            continue
        clamp = FORMATION_MAX_OFFSET_KM
        cx = max(-clamp, min(clamp, x))
        cy = max(-clamp, min(clamp, y))
        cz = max(-clamp, min(clamp, z))
        if (cx, cy, cz) != (x, y, z):
            notes.append(f"clamped {ship.ship_name} offset to the "
                         f"+/-{clamp:.0f} km cube")
        ship.offset_km = (cx, cy, cz)

    # Place each hull against ALL settled predecessors. A later pairwise nudge
    # can otherwise undo an earlier separation or push a hull outside the cube.
    clamp = FORMATION_MAX_OFFSET_KM
    for i, ship in enumerate(drafted):
        x, y, z = (max(-clamp, min(clamp, c)) for c in ship.offset_km)
        step = MIN_SEPARATION_KM * 2
        candidates = [y] + [y + sign * step * n
                            for n in range(1, math.ceil(2 * clamp / step) + 1)
                            for sign in (1, -1)]
        for cy in candidates:
            if abs(cy) <= clamp and all(
                math.dist((x, cy, z), other.offset_km) >= MIN_SEPARATION_KM
                # ...and out of each other's drive plume when the fleet
                # burns straight ahead (src/drive_plume.py).
                and not formation_plume_conflicts([(x, cy, z), other.offset_km])
                for other in drafted[:i]
            ):
                if (x, cy, z) != ship.offset_km:
                    notes.append(f"nudged {ship.ship_name} to a separated slot inside the formation cube"
                                 " (clear of friendly drive plumes)")
                ship.offset_km = (x, cy, z)
                break
        else:
            raise ValueError("Too many overlapping hulls for a separated lateral formation")
    return notes


def world_positions_km(
    draft: FleetDraft,
    initial_distance_km: float,
) -> Dict[str, Dict[str, float]]:
    """
    Convert admiral-local formation offsets to world coordinates (km).

    Alpha anchors at -D/2 facing +X; beta anchors at +D/2 facing -X. Beta's
    local frame is rotated 180 deg about Z (not mirrored), so "left flank"
    means the same thing to both admirals.
    """
    half = initial_distance_km / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for ship in draft.ships:
        ox, oy, oz = ship.offset_km
        if draft.faction == "alpha":
            out[ship.ship_id] = {"x": -half + ox, "y": oy, "z": oz}
        else:
            out[ship.ship_id] = {"x": half - ox, "y": -oy, "z": oz}
    return out


# ---------------------------------------------------------------------------
# Auto-draft (fallback + offline testing)
# ---------------------------------------------------------------------------

# Auto-draft doctrines: the hull preference cycle each one buys from. The
# ORDER is part of the seed contract (auto_draft picks by index from the seed).
AUTO_DRAFT_DOCTRINES: Dict[str, List[str]] = {
    "gunline": ["destroyer", "destroyer", "cruiser", "frigate"],
    "torpedo": ["cruiser_torpedo", "corvette", "corvette", "frigate"],
    "combined": ["battleship", "destroyer", "corvette"],
    "flagship": ["dreadnought", "frigate", "corvette"],
}


def auto_draft(
    faction: str,
    budget: int = DEFAULT_POINT_BUDGET,
    max_ships: int = DEFAULT_MAX_SHIPS,
    seed: Optional[int] = None,
    doctrine: Optional[str] = None,
) -> FleetDraft:
    """
    Deterministic budget-respecting draft used when no admiral drafts.

    Picks a doctrine (gunline / torpedo swarm / combined arms / flagship) from
    the seed - or the named one, `doctrine`, a key of AUTO_DRAFT_DOCTRINES -
    and greedily fills the budget, so two auto-drafted sides still produce an
    interesting asymmetric battle.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < min(SHIP_POINT_COSTS.values()):
        raise ValueError("Draft budget must be an integer sufficient to buy at least one ship")
    if isinstance(max_ships, bool) or not isinstance(max_ships, int) or max_ships < 1:
        raise ValueError("max_ships must be a positive integer")
    rng = random.Random(seed if seed is not None else 0)
    doctrines = list(AUTO_DRAFT_DOCTRINES.values())
    # The seed draw happens even for a named doctrine, so the RNG stream (and
    # anything later drawn from it) does not depend on how it was chosen.
    preference = doctrines[rng.randrange(len(doctrines))]
    if doctrine is not None:
        if doctrine not in AUTO_DRAFT_DOCTRINES:
            raise ValueError(f"unknown auto-draft doctrine {doctrine!r} "
                             f"(one of {', '.join(AUTO_DRAFT_DOCTRINES)})")
        preference = AUTO_DRAFT_DOCTRINES[doctrine]
    preference = list(preference)
    if not any(SHIP_POINT_COSTS[s] <= budget for s in preference):
        preference = [min(SHIP_POINT_COSTS, key=SHIP_POINT_COSTS.get)]

    picked: List[str] = []
    spent = 0
    # Cycle the preference list, buying whatever still fits
    idx = 0
    while len(picked) < max_ships:
        candidates = [s for s in preference
                      if SHIP_POINT_COSTS[s] <= budget - spent]
        if not candidates:
            break
        stype = preference[idx % len(preference)]
        if SHIP_POINT_COSTS[stype] > budget - spent:
            idx += 1
            continue
        picked.append(stype)
        spent += SHIP_POINT_COSTS[stype]
        idx += 1

    # The hull cap can bind long before the budget does (the gunline fielded
    # the same 144-point fleet at budgets 150 to 300). Then upgrade: swap the
    # cheapest hull for the dearest doctrine hull the remaining points allow,
    # until no swap fits.
    if len(picked) >= max_ships:
        while True:
            i = min(range(len(picked)), key=lambda k: (SHIP_POINT_COSTS[picked[k]], k))
            cheapest = SHIP_POINT_COSTS[picked[i]]
            upgrades = [s for s in preference
                        if cheapest < SHIP_POINT_COSTS[s] <= budget - spent + cheapest]
            if not upgrades:
                break
            best = max(upgrades, key=lambda s: (SHIP_POINT_COSTS[s], -preference.index(s)))
            spent += SHIP_POINT_COSTS[best] - cheapest
            picked[i] = best

    draft = FleetDraft(
        faction=faction, budget=budget,
        points_spent=spent, auto=True,
        formation_name="staggered wall",
        selection_rationale="auto-draft",
    )
    draft.ships = name_drafted_ships(picked, faction)
    # Staggered wall: lateral spread with alternating depth and height
    n = len(draft.ships)
    for i, ship in enumerate(draft.ships):
        lateral = (i - (n - 1) / 2) * 12.0
        depth = -10.0 if i % 2 else 0.0
        height = 6.0 if i % 3 == 2 else 0.0
        ship.offset_km = (depth, lateral, height)
    return draft


# ---------------------------------------------------------------------------
# LLM draft
# ---------------------------------------------------------------------------

def _selection_prompt(admiral_name: str, faction: str, budget: int,
                      max_ships: int, catalog: str,
                      initial_distance_km: float,
                      captain_model: Optional[str] = None) -> str:
    crew = (f"Your ships will be flown by independent {captain_model} AI captains "
            "with tactical tools and execution feedback. They receive your fleet "
            "directive and individual checkpoint orders."
            if captain_model else
            "Your ships will be flown by competent-but-simple AI captains that "
            "follow your checkpoint orders - draft a fleet whose plan survives simple execution.")
    return f"""You are {admiral_name}, fleet admiral of the {faction.upper()} force.

Before battle you must BUY YOUR FLEET. You have {budget} POINTS and may field
at most {max_ships} ships. The enemy admiral has the same budget and the same
catalog; you cannot see their picks. Battle starts at {initial_distance_km:.0f} km
head-on separation. {crew}

{catalog}

DOCTRINE NOTES:
- Coilguns: spinal mounts hit hard but need the nose on target; turrets cover
  the front hemisphere. Gun accuracy rises steeply as range drops below
  ~300 km.
- Torpedoes: 12g guided rounds. Point defense needs 3+ turrets dwelling on a
  round to blind its seeker, so SATURATION (many rounds arriving together)
  beats trickled launches. {LINK_FACT} Launch profiles: {PROFILE_FACT}
  Torpedo hulls die fast if caught alone.
- Point defense also shaves incoming coilgun slugs. Ships with 3-4 PD turrets
  anchor a defensive wall.
- Fast light hulls (3g) can dictate range against slow capitals (<= 1g).

WEAPON EFFECTS (calibrated from recorded battles):
- ALL kinetic impacts scale with closing speed - gun GJ ratings are muzzle
  energy against a stationary target. Mutual closure multiplies them and
  mutual recession bleeds them off (measured: the same guns landed 0.2 to
  8.7 GJ per hit depending on encounter geometry). Coilgun rounds do NOT
  accelerate after launch - no guidance, so at long range an evading
  target simply is not where the slug arrives.
- Armor ablates at roughly 1.5 cm per GJ before anything penetrates.
  Noses carry 3-8x the armor of flanks and tails - kill ships by cracking
  a thin facing, then penetrating hits wreck modules (bridge, magazines,
  reactor).
- Torpedoes DO accelerate and steer all the way in (12g), so their closure
  scaling is one-sided in the attacker's favor: ~95 GJ against a head-on
  charge, ~23 GJ at zero closure, ~8 GJ chasing a receding ship - and a
  receding EVADING target usually escapes the round entirely.

TORPEDO FLIGHT PROFILE (what a defender exploits):
- Each round carries 14 km/s of delta-v at 12g (about two minutes of total
  burn) and flies augmented proportional navigation. It never brakes:
  braking would waste its own warhead. Outside its no-escape zone it
  spends only what steering demands - hoarding delta-v so it can reach
  the NEZ at all - and merely enforces a minimum cruise closure.
- The no-escape zone is its commit line: the round continuously checks
  whether your best full-lateral burn could still build miss distance
  faster than its remaining thrust authority and fuel can null. Once you
  cannot, it commits - dumping every spare m/s into MORE closing speed
  (impact energy scales as v^2, and a shorter run gives PD fewer dwell
  windows), holding back only a ~0.5 km/s reserve for final corrections.
- So the fight against a torpedo is won EARLY or not at all: hard evasion
  and low closure before the NEZ force it to burn its budget in the
  chase, and a dry round is a ballistic miss. PD blinding is inherently a
  terminal-window affair (laser coupling only bites in roughly the last
  110 km, by which point the round is usually committed) - PD thins
  salvos, it does not save a ship that let the geometry close.

TORPEDO COUNTERPLAY (seeded runs: a 4 battlecruiser + 2 destroyer + 2 frigate
gun wall against 5 corvettes + 3 frigates, 40 rounds):
- From 500 km a nose-on wall took 78% of the rounds (on the nose) and lost 7
  of 8 hulls; burning straight away it took 46% (on the thin tail, ~19% less
  energy each) and lost 7 of 8; charging it took 73% and lost all 8. From
  1000 km: nose-on lost 8 of 8, burning away 5 of 8, charging 2 of 8 (and
  killed 1.3 boats). Burning away leaves more rounds blinded or dry; what
  still hits lands on the tail. An emptied torpedo hull mounts no guns: it
  dies to whatever can catch it afterward.
- A fleet only kites as fast as its slowest hull. Mixing 3g screens with
  <=1g capitals means choosing between abandoning the capitals or eating
  the waves at capital speed - commit to one doctrine, not half of each.

Call select_fleet with your purchase. Spend your points well - unspent points
are wasted."""


def _formation_prompt(admiral_name: str, faction: str, draft: FleetDraft,
                      initial_distance_km: float) -> str:
    roster = "\n".join(
        f"- {s.ship_name} ({s.ship_type}, {SHIP_POINT_COSTS[s.ship_type]} pts)"
        for s in draft.ships)
    return f"""You are {admiral_name}. Your fleet is bought:

{roster}

Now PLACE YOUR FORMATION. Coordinates are km offsets from your fleet anchor,
in YOUR frame: +x points AT the enemy, y is lateral (your left/right), z is
vertical. Offsets are limited to +/-{FORMATION_MAX_OFFSET_KM:.0f} km per axis; the
enemy fleet starts {initial_distance_km:.0f} km down the +x axis.

Considerations:
- Ships within ~100 km of each other can overlap PD coverage against torpedoes.
- A ship pushed +x is your vanguard (absorbs the first exchange); -x holds it
  back as a reserve or keeps a fragile torpedo hull out of gun range.
- Lateral/vertical spread complicates enemy spinal geometry but dilutes PD.

Call set_formation with a placement for each ship."""


def _retry_turn(calls: List[Any], select: Any, error: str) -> List[Dict[str, Any]]:
    """
    The rejected attempt as a real exchange: the model's own tool call(s)
    as the assistant turn, the validator's error as the select_fleet
    tool_result (every other call answered too, as the API requires). A
    reply without any tool call gets a plain user nudge instead. The old
    retry rewrote the attempt as "(previous attempt rejected: ...)", hiding
    the fleet the model had actually submitted.
    """
    tool_calls = [c for c in calls or [] if getattr(c, "id", None)]
    if select is None or not tool_calls:
        return [{"role": "user", "content": f"INVALID: {error} Try again with select_fleet."}]
    turn: List[Dict[str, Any]] = [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": c.id, "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments or {})}}
                       for c in tool_calls]}]
    for c in tool_calls:
        turn.append({"role": "tool", "tool_call_id": c.id,
                     "content": (f"REJECTED: {error} Call select_fleet again with a fleet inside "
                                 "both limits." if c is select else "ignored: only select_fleet counts here")})
    return turn


def _refused(client: Any) -> bool:
    """True if the client's last reply on this thread was a refusal (never re-asked)."""
    finish = getattr(client, "last_finish_reason", None)
    try:
        return callable(finish) and finish() == "refusal"
    except Exception:  # noqa: BLE001 - test doubles
        return False


def run_admiral_draft(
    client: Any,
    model: str,
    admiral_name: str,
    faction: str,
    fleet_data: Dict[str, Any],
    budget: int = DEFAULT_POINT_BUDGET,
    max_ships: int = DEFAULT_MAX_SHIPS,
    initial_distance_km: float = 500.0,
    verbose: bool = True,
    seed: Optional[int] = None,
    captain_model: Optional[str] = None,
    allow_auto_fallback: bool = True,
    allow_trim: bool = True,
) -> FleetDraft:
    """
    Run the two-phase LLM draft for one side.

    An invalid selection is answered as a real tool exchange: the model's
    select_fleet call stays in the conversation and the validator's error
    (the rejected fleet with each line's cost, the total, the cap and the
    excess) comes back as its tool_result, so the next attempt can see what
    it submitted. After the last attempt the model's last fleet is trimmed
    to validity (allow_trim: cheapest hulls dropped until it fits; recorded
    as a draft_auto_trimmed event when the client carries a recorder) -
    the model's own hulls, so this runs whatever allow_auto_fallback says.
    With nothing to trim (API failure, no select_fleet call, only unknown
    hulls) the side falls back to auto_draft, or with
    allow_auto_fallback=False the battle does not start.
    """
    catalog = build_catalog_text(fleet_data)
    draft = FleetDraft(faction=faction, budget=budget)
    tool = select_fleet_tool(budget, max_ships)

    # --- Phase 1: selection, with validation feedback loop ---
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _selection_prompt(
            admiral_name, faction, budget, max_ships, catalog,
            initial_distance_km, captain_model)},
        {"role": "user", "content": "Buy your fleet now with select_fleet."},
    ]
    flat: Optional[List[str]] = None
    last_ships: Any = None
    last_rationale = ""
    # Some provider endpoints reject a forced tool call (DeepSeek's own API,
    # Claude with thinking on Bedrock). A rejected "required" is asked again
    # once with "auto"; the prompts already insist on the tool.
    tool_choice = "required"
    for attempt in range(3):
        try:
            calls = client.decide_with_tools(
                messages, [tool], model=model,
                tool_choice=tool_choice)
        except LLMCallError as e:
            if tool_choice == "required" and not e.retryable:
                print(f"[DRAFT {faction}] forced tool call rejected ({e}); retrying with tool_choice=auto")
                tool_choice = "auto"
                continue
            print(f"[DRAFT {faction}] selection call failed: {e}")
            break
        if _refused(client):
            print(f"[DRAFT {faction}] selection refused by the model (stop_reason=refusal)")
            break
        select = next((c for c in calls if c.name == "select_fleet"), None)
        if select is None:
            error = "You made no select_fleet call. You MUST call select_fleet."
        else:
            last_ships = select.arguments.get("ships")
            last_rationale = (select.arguments.get("rationale") or "").strip()
            flat, spent, error = validate_selection(last_ships, budget, max_ships)
        if flat is not None:
            draft.points_spent = spent
            draft.selection_rationale = last_rationale
            break
        if verbose:
            print(f"[DRAFT {faction}] invalid selection (attempt {attempt + 1}): {error}")
        messages.extend(_retry_turn(calls, select, error))

    if flat is None and allow_trim and last_ships is not None:
        kept, spent, dropped = trim_selection(last_ships, budget, max_ships)
        if kept is not None:
            flat = kept
            draft.points_spent = spent
            draft.selection_rationale = last_rationale
            draft.trimmed = {"submitted": last_ships, "dropped": dropped, "kept": list(kept),
                             "points": spent, "budget": budget, "max_ships": max_ships}
            print(f"[DRAFT {faction}] no valid fleet after the retries: trimmed the last one "
                  f"to {spent}/{budget} pts, {len(kept)}/{max_ships} hulls (dropped: "
                  f"{', '.join(dropped) or 'nothing'})")
            recorder = getattr(client, "recorder", None)
            if recorder is not None and callable(getattr(recorder, "record", None)):
                try:
                    recorder.record(0.0, "draft_auto_trimmed", None, faction=faction,
                                    **draft.trimmed)
                except Exception as e:  # noqa: BLE001 - the record must not stop the draft
                    print(f"[DRAFT {faction}] could not record draft_auto_trimmed: {e}")

    if flat is None:
        if not allow_auto_fallback:
            raise RuntimeError(f'{faction} admiral did not submit a valid fleet; battle not started')
        print(f"[DRAFT {faction}] falling back to auto-draft")
        auto = auto_draft(faction, budget, max_ships, seed=seed)
        return auto

    draft.ships = name_drafted_ships(flat, faction, faction_pool=True)
    if verbose:
        roster = ", ".join(s.ship_name for s in draft.ships)
        print(f"[DRAFT {faction}] {admiral_name} spends "
              f"{draft.points_spent}/{budget} pts: {roster}")
        if draft.selection_rationale:
            print(f"[DRAFT {faction}]   concept: {draft.selection_rationale}")

    # --- Phase 2: formation ---
    form_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _formation_prompt(
            admiral_name, faction, draft, initial_distance_km)},
        {"role": "user", "content": "Place your formation now with set_formation."},
    ]
    placements: Optional[List[Dict[str, Any]]] = None
    for attempt in range(2):
        try:
            calls = client.decide_with_tools(
                form_messages, [SET_FORMATION_TOOL], model=model,
                tool_choice=tool_choice)
        except LLMCallError as e:
            if tool_choice == "required" and not e.retryable:
                print(f"[DRAFT {faction}] forced tool call rejected ({e}); retrying with tool_choice=auto")
                tool_choice = "auto"
                continue
            print(f"[DRAFT {faction}] formation call failed: {e}")
            break
        if _refused(client):
            print(f"[DRAFT {faction}] formation refused by the model (stop_reason=refusal)")
            break
        form = next((c for c in calls if c.name == "set_formation"), None)
        if form is not None:
            placements = form.arguments.get("placements") or []
            draft.formation_name = (form.arguments.get("formation_name")
                                    or "custom").strip()
            draft.formation_rationale = (form.arguments.get("rationale") or "").strip()
            break
        form_messages.append({"role": "user", "content":
                              "You MUST call set_formation with placements."})

    notes = apply_formation(draft.ships, placements or [])
    if verbose:
        if placements is None:
            print(f"[DRAFT {faction}] no formation given - default line abreast")
        else:
            print(f"[DRAFT {faction}] formation '{draft.formation_name}'")
            if draft.formation_rationale:
                print(f"[DRAFT {faction}]   intent: {draft.formation_rationale}")
        for note in notes:
            print(f"[DRAFT {faction}]   note: {note}")
    return draft


# ---------------------------------------------------------------------------
# Conversion to a battle fleet definition
# ---------------------------------------------------------------------------

def draft_to_fleet_definition(
    draft: FleetDraft,
    initial_distance_km: float,
    captain_model: str = "heuristic",
    admiral_config: Optional[Any] = None,
    temperature: float = 0.7,
    mcp_config: Optional[Any] = None,
) -> Any:
    """Build a FleetDefinition (ships + positions + admiral/MCP) from a draft."""
    from .fleet_config import FleetDefinition, ShipConfig

    positions = world_positions_km(draft, initial_distance_km)
    ships = []
    for ship in draft.ships:
        ships.append(ShipConfig(
            ship_id=ship.ship_id,
            ship_type=ship.ship_type,
            model=captain_model,
            temperature=temperature,
            position=positions[ship.ship_id],
            captain_name=f"Captain {ship.ship_name.split(' ', 1)[1]}",
            ship_name=ship.ship_name,
        ))
    return FleetDefinition(
        ships=ships,
        faction=draft.faction,
        admiral=admiral_config,
        mcp=mcp_config,
    )


def draft_summary_dict(draft: FleetDraft) -> Dict[str, Any]:
    """JSON-safe draft summary for recordings / sidecar files."""
    return {
        "faction": draft.faction,
        "budget": draft.budget,
        "points_spent": draft.points_spent,
        "auto_draft": draft.auto,
        **({"auto_trimmed": draft.trimmed} if draft.trimmed else {}),
        "formation_name": draft.formation_name,
        "selection_rationale": draft.selection_rationale,
        "formation_rationale": draft.formation_rationale,
        "ships": [
            {
                "ship_id": s.ship_id,
                "ship_name": s.ship_name,
                "ship_type": s.ship_type,
                "cost": SHIP_POINT_COSTS[s.ship_type],
                "offset_km": list(s.offset_km),
            }
            for s in draft.ships
        ],
    }
