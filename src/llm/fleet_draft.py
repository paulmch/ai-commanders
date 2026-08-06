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

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .client import LLMCallError

# ---------------------------------------------------------------------------
# Point costs
# ---------------------------------------------------------------------------

# Hand-tuned, anchored on wet mass and adjusted for capability: the frigate
# is priced below the corvette despite equal mass (thin skin, light guns),
# the battlecruiser below the cruiser (same guns, 177cm vs 241cm nose), and
# the torpedo cruiser above both (a saturation salvo that can kill a
# dreadnought). Default budget 100 buys e.g. one dreadnought + escorts, or
# six destroyers, or a torpedo-cruiser wolfpack - real tradeoffs.
SHIP_POINT_COSTS: Dict[str, int] = {
    "corvette": 10,
    "frigate": 8,
    "destroyer": 16,
    "battlecruiser": 22,
    "cruiser": 26,
    "cruiser_torpedo": 30,
    "battleship": 40,
    "dreadnought": 55,
    "dreadnought_siege": 58,
}

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
    lines = [
        "SHIP CATALOG (cost in points | one line per hull class):",
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

    flat: List[str] = []
    spent = 0
    for entry in ships_arg:
        stype = str(entry.get("ship_type", "")).strip().lower()
        count = entry.get("count", 1)
        if stype not in SHIP_POINT_COSTS:
            return None, 0, (
                f"Unknown hull '{stype}'. Valid hulls: "
                f"{', '.join(SHIP_POINT_COSTS)}.")
        if not isinstance(count, int) or count < 1:
            return None, 0, f"Invalid count {count!r} for {stype}."
        flat.extend([stype] * count)
        spent += SHIP_POINT_COSTS[stype] * count

    if spent > budget:
        return None, 0, (
            f"Over budget: that fleet costs {spent} points but you only have "
            f"{budget}. Drop or downgrade ships.")
    if len(flat) > max_ships:
        return None, 0, (
            f"Too many hulls: {len(flat)} ships, the cap is {max_ships}.")
    return flat, spent, None


def name_drafted_ships(ship_types: List[str], faction: str) -> List[DraftedShip]:
    """Assign ids and readable names ("TIS Falchion-2") to drafted hulls."""
    prefix = "TIS" if faction == "alpha" else "OCS"
    counters: Dict[str, int] = {}
    drafted = []
    for i, stype in enumerate(ship_types):
        nickname = CLASS_NICKNAMES.get(stype, stype.title())
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

    # Enforce minimum separation with a simple lateral nudge
    for i, a in enumerate(drafted):
        for b in drafted[i + 1:]:
            ax, ay, az = a.offset_km
            bx, by, bz = b.offset_km
            d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
            if d2 < MIN_SEPARATION_KM ** 2:
                b.offset_km = (bx, by + MIN_SEPARATION_KM * 2, bz)
                notes.append(f"nudged {b.ship_name} {MIN_SEPARATION_KM * 2:.0f} km "
                             f"laterally (was overlapping {a.ship_name})")
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

def auto_draft(
    faction: str,
    budget: int = DEFAULT_POINT_BUDGET,
    max_ships: int = DEFAULT_MAX_SHIPS,
    seed: Optional[int] = None,
) -> FleetDraft:
    """
    Deterministic budget-respecting draft used when no admiral drafts.

    Picks a doctrine (gunline / torpedo swarm / combined arms) from the seed
    and greedily fills the budget, so two auto-drafted sides still produce an
    interesting asymmetric battle.
    """
    rng = random.Random(seed if seed is not None else 0)
    doctrines = [
        ["destroyer", "destroyer", "cruiser", "frigate"],          # gunline
        ["cruiser_torpedo", "corvette", "corvette", "frigate"],    # torpedo swarm
        ["battleship", "destroyer", "corvette"],                   # combined arms
        ["dreadnought", "frigate", "corvette"],                    # flagship
    ]
    preference = doctrines[rng.randrange(len(doctrines))]

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
                      initial_distance_km: float) -> str:
    return f"""You are {admiral_name}, fleet admiral of the {faction.upper()} force.

Before battle you must BUY YOUR FLEET. You have {budget} POINTS and may field
at most {max_ships} ships. The enemy admiral has the same budget and the same
catalog; you cannot see their picks. Battle starts at {initial_distance_km:.0f} km
head-on separation. Your ships will be flown by competent-but-simple AI
captains that follow your checkpoint orders - draft a fleet whose plan
survives simple execution.

{catalog}

DOCTRINE NOTES:
- Coilguns: spinal mounts hit hard but need the nose on target; turrets cover
  the front hemisphere. Gun accuracy rises steeply as range drops below
  ~300 km.
- Torpedoes: 12g guided rounds. Point defense needs 3+ turrets dwelling on a
  round to blind its seeker, so SATURATION (many rounds arriving together)
  beats trickled launches. Rounds whose target dies mid-flight retarget a
  reachable enemy on their own (live seeker + fuel permitting; healthy
  seekers maximize impact energy, PD-singed ones race the fastest
  intercept before blinding). Torpedo hulls die fast if caught alone.
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

TORPEDO COUNTERPLAY:
- Charging a torpedo fleet feeds it - closure IS its warhead. Burning AWAY
  cuts impact energy, stretches flight time so PD gets full dwell to blind
  seekers, and bleeds finite magazines dry. An emptied torpedo hull mounts
  no guns: it dies to whatever can catch it afterward.
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
) -> FleetDraft:
    """
    Run the two-phase LLM draft for one side.

    Any unrecoverable failure (API errors, three invalid selections) falls
    back to auto_draft so the battle always starts.
    """
    catalog = build_catalog_text(fleet_data)
    draft = FleetDraft(faction=faction, budget=budget)

    # --- Phase 1: selection, with validation feedback loop ---
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _selection_prompt(
            admiral_name, faction, budget, max_ships, catalog,
            initial_distance_km)},
        {"role": "user", "content": "Buy your fleet now with select_fleet."},
    ]
    flat: Optional[List[str]] = None
    for attempt in range(3):
        try:
            calls = client.decide_with_tools(messages, [SELECT_FLEET_TOOL], model=model)
        except LLMCallError as e:
            print(f"[DRAFT {faction}] selection call failed: {e}")
            break
        select = next((c for c in calls if c.name == "select_fleet"), None)
        if select is None:
            error = "You made no select_fleet call. You MUST call select_fleet."
        else:
            flat, spent, error = validate_selection(
                select.arguments.get("ships"), budget, max_ships)
        if flat is not None:
            draft.points_spent = spent
            draft.selection_rationale = (select.arguments.get("rationale") or "").strip()
            break
        if verbose:
            print(f"[DRAFT {faction}] invalid selection (attempt {attempt + 1}): {error}")
        messages.append({"role": "assistant", "content":
                         f"(previous attempt rejected: {error})"})
        messages.append({"role": "user", "content":
                         f"INVALID: {error} Try again with select_fleet."})

    if flat is None:
        print(f"[DRAFT {faction}] falling back to auto-draft")
        auto = auto_draft(faction, budget, max_ships, seed=seed)
        return auto

    draft.ships = name_drafted_ships(flat, faction)
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
            calls = client.decide_with_tools(form_messages, [SET_FORMATION_TOOL], model=model)
        except LLMCallError as e:
            print(f"[DRAFT {faction}] formation call failed: {e}")
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
) -> Any:
    """Build a FleetDefinition (ships + positions + admiral) from a draft."""
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
    )


def draft_summary_dict(draft: FleetDraft) -> Dict[str, Any]:
    """JSON-safe draft summary for recordings / sidecar files."""
    return {
        "faction": draft.faction,
        "budget": draft.budget,
        "points_spent": draft.points_spent,
        "auto_draft": draft.auto,
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
