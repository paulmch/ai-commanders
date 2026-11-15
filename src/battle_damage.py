"""
Battle damage consequences (pass 5): what a hit does beyond armour and hull
percentage, and what a ship's death does to everything around it.

Three automatic mechanics, all resolved by the engine (no commander tools):

1. SURFACE FITTINGS. Every hull hit - penetrating or not - can strike an
   external fitting: a gun mount or a point-defence turret. A damaged gun mount
   fires slower, a wrecked one stops firing; PD turrets are lost one at a time.
   The long armour-ablation phase of a gun duel therefore strips a ship's teeth
   and its torpedo defence before the armour ever opens.

2. REACTOR BLAST. Every ship death ends in its reactor going up
   (CombatSimulation._destroy_ship). The detonation scorches the armour facing
   it on nearby ships, dumps heat into their sinks, shreds extended radiators
   and exposed fittings, shortens the fuse of nearby dying hulks (chain
   reactions) and kills or blinds torpedoes flying through it. Friend and foe
   alike; a blast never touches hull modules, so it cannot kill a healthy ship
   on its own.

3. MAGAZINE COOK-OFF. A destroyed magazine module burns off part of the
   ship's remaining rounds (torpedoes on torpedo hulls, gun ammunition on gun
   hulls) and the secondary explosion tears into the neighbouring compartments.

Randomness comes from CombatSimulation._damage_rng, a stream derived from the
battle seed (the same pattern as the per-torpedo and per-projectile streams),
so these rolls never shift the main battle RNG that decides hit/miss.

Every consequence is recorded; see docs/pass-5-recording-additions.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional

import sys

try:
    from .combat import HitLocation
    from .modules import ModuleType
except ImportError:  # pragma: no cover - script-style import path
    from combat import HitLocation
    from modules import ModuleType

if TYPE_CHECKING:  # pragma: no cover
    from .simulation import CombatSimulation, ShipCombatState


def _event_types(sim: "CombatSimulation") -> Any:
    """SimulationEventType of the module that defines this simulation
    (avoids a circular import and works for both import styles)."""
    return sys.modules[type(sim).__module__].SimulationEventType


# =============================================================================
# SURFACE FITTINGS
# =============================================================================

# Chance that one hull hit also strikes an external fitting, by struck facing.
# The flanks and the stern carry the turrets; the bow shows a small
# cross-section with only the spinal mount and the forward PD in it.
SURFACE_FITTING_HIT_CHANCE = {
    HitLocation.NOSE: 0.12,
    HitLocation.LATERAL: 0.18,
    HitLocation.TAIL: 0.18,
}
# Fitting health lost per GJ of impact energy (percent). A ~2.7 GJ coilgun hit
# takes ~2/3 of a mount: the first hit degrades it, the second wrecks it. A
# torpedo (~25 GJ) wrecks whatever it catches.
MOUNT_DAMAGE_PCT_PER_GJ = 25.0
# The drive bells are the one fitting a TAIL hit cannot miss for long: chance
# that a hull hit on the stern strikes the engine assembly (rolled before the
# turret chance, on the same draw), and engine-module health lost per GJ. A
# ~2.7 GJ slug costs ~1/3 of the thrust, a torpedo up the tail leaves the ship
# adrift - crippled, not dead (the engine is not a critical module).
ENGINE_HIT_CHANCE_TAIL = 0.30
ENGINE_DAMAGE_PCT_PER_GJ = 12.0
# A damaged gun mount cycles slower: cooldowns are divided by its health
# fraction, capped at this factor (so a mount at 50% health or less fires at
# half rate until it is wrecked).
MOUNT_MAX_COOLDOWN_FACTOR = 2.0


def mount_cooldown_factor(mount_health_pct: float) -> float:
    """Cooldown multiplier of a gun mount at the given health (1.0 = intact)."""
    if mount_health_pct >= 100.0:
        return 1.0
    if mount_health_pct <= 0.0:
        return MOUNT_MAX_COOLDOWN_FACTOR
    return min(MOUNT_MAX_COOLDOWN_FACTOR, 100.0 / mount_health_pct)


def _is_spinal(weapon_state: Any) -> bool:
    name = (getattr(weapon_state.weapon, "name", "") or "").lower()
    return "spinal" in name or "siege" in name


def _is_gun(weapon_state: Any) -> bool:
    """True for a coilgun mount (the weapons dict may also list PD/launchers)."""
    name = (getattr(weapon_state.weapon, "name", "") or "").lower()
    return not ("torpedo" in name or "laser" in name or "point defense" in name)


def _gun_fittings(ship: "ShipCombatState", hit_location: Optional[HitLocation]) -> List[tuple]:
    """Operational gun mounts that a hit on this facing can reach.

    The spinal mount lives in the bow: only a NOSE hit (or a blast, which
    passes hit_location=None for 'any facing') can reach it.
    """
    out = []
    for slot, ws in ship.weapons.items():
        if slot.startswith("pd") or not ws.is_operational:
            continue
        if not _is_gun(ws):
            continue
        if _is_spinal(ws) and hit_location not in (None, HitLocation.NOSE):
            continue
        out.append(("gun", slot, ws))
    return out


def _pd_fittings(ship: "ShipCombatState") -> List[tuple]:
    return [("pd", i, pd) for i, pd in enumerate(ship.point_defense) if pd.is_operational]


def damage_fitting(
    sim: "CombatSimulation",
    ship: "ShipCombatState",
    fitting: tuple,
    health_loss_pct: float,
    cause: str,
    source_ship_id: Optional[str] = None,
    hit_location: Optional[HitLocation] = None,
) -> bool:
    """Apply health loss to one fitting, record it. Returns True if destroyed."""
    SimulationEventType = _event_types(sim)

    kind, key, state = fitting
    if kind == "engine":
        return _damage_engine(sim, ship, state, health_loss_pct, cause,
                              source_ship_id, hit_location)
    before = state.mount_health
    state.mount_health = max(0.0, before - health_loss_pct)
    destroyed = state.mount_health <= 0.0
    if destroyed:
        state.is_operational = False
        if kind == "pd":
            # Gun-dictionary twin of the turret, where a hull lists one.
            twin = ship.weapons.get(f"pd_{key}")
            if twin is not None:
                twin.is_operational = False
            if hasattr(state, "reset_target"):
                state.reset_target()

    if kind == "gun":
        slot = key
        index = list(ship.weapons.keys()).index(key)
        name = getattr(state.weapon, "name", key)
        factor = mount_cooldown_factor(state.mount_health)
    else:
        slot = getattr(state, "turret_name", f"pd_{key}")
        index = key
        name = "PD Laser"
        factor = 1.0

    sim._log_event(SimulationEventType.MOUNT_DAMAGED, ship.ship_id, data={
        'kind': kind,
        'slot': slot,
        'index': index,
        'weapon_name': name,
        'health_pct': round(state.mount_health, 1),
        'health_lost_pct': round(before - state.mount_health, 1),
        'destroyed': destroyed,
        'cooldown_factor': None if destroyed else round(factor, 2),
        'cause': cause,
        'hit_location': hit_location.value if hit_location is not None else None,
        'source_ship_id': source_ship_id,
    })
    if destroyed:
        print(f"  !!! [{ship.ship_id}] {name} ({slot}) KNOCKED OUT ({cause})")
    return destroyed


def _damage_engine(
    sim: "CombatSimulation",
    ship: "ShipCombatState",
    module: Any,
    health_loss_pct: float,
    cause: str,
    source_ship_id: Optional[str],
    hit_location: Optional[HitLocation],
) -> bool:
    """Surface damage to an engine module; thrust follows engine_effectiveness."""
    SimulationEventType = _event_types(sim)
    before = module.health_percent
    module.health_percent = max(0.0, before - health_loss_pct)
    destroyed = module.is_destroyed
    engines = ship._get_modules_by_type("engine")
    sim._log_event(SimulationEventType.MOUNT_DAMAGED, ship.ship_id, data={
        'kind': 'engine',
        'slot': module.name,
        'index': engines.index(module) if module in engines else 0,
        'weapon_name': module.name,
        'health_pct': round(module.health_percent, 1),
        'health_lost_pct': round(before - module.health_percent, 1),
        'destroyed': destroyed,
        'cooldown_factor': None,
        'thrust_fraction': round(ship.engine_effectiveness, 3),
        'cause': cause,
        'hit_location': hit_location.value if hit_location is not None else None,
        'source_ship_id': source_ship_id,
    })
    if destroyed:
        sim._log_event(SimulationEventType.MODULE_DESTROYED, ship.ship_id, data={
            'module_name': module.name,
            'cause': cause,
        })
        print(f"  !!! [{ship.ship_id}] {module.name} WRECKED ({cause}) - "
              f"thrust {ship.engine_effectiveness * 100:.0f}%")
    return destroyed


def resolve_surface_hit(
    sim: "CombatSimulation",
    target: "ShipCombatState",
    hit_location: HitLocation,
    energy_gj: float,
    source_ship_id: Optional[str],
) -> None:
    """Roll whether a hull hit also struck an external fitting, and apply it."""
    if target.is_destroyed or target.is_dying:
        return
    chance = SURFACE_FITTING_HIT_CHANCE.get(hit_location, 0.0)
    rng = sim._damage_rng
    # Always two draws per hit, so the stream position depends only on the
    # number of hits resolved, not on what earlier hits knocked out.
    roll = rng.random()
    pick = rng.random()
    engine_chance = 0.0
    engines: List[Any] = []
    if hit_location == HitLocation.TAIL:
        engines = [m for m in target._get_modules_by_type("engine") if not m.is_destroyed]
        if engines:
            engine_chance = ENGINE_HIT_CHANCE_TAIL
    if roll < engine_chance:
        module = engines[min(len(engines) - 1, int(pick * len(engines)))]
        damage_fitting(
            sim, target, ("engine", module.name, module),
            health_loss_pct=energy_gj * ENGINE_DAMAGE_PCT_PER_GJ,
            cause='surface_hit',
            source_ship_id=source_ship_id,
            hit_location=hit_location,
        )
        return
    if roll >= engine_chance + chance:
        return
    fittings = _gun_fittings(target, hit_location) + _pd_fittings(target)
    if not fittings:
        return
    fitting = fittings[min(len(fittings) - 1, int(pick * len(fittings)))]
    damage_fitting(
        sim, target, fitting,
        health_loss_pct=energy_gj * MOUNT_DAMAGE_PCT_PER_GJ,
        cause='surface_hit',
        source_ship_id=source_ship_id,
        hit_location=hit_location,
    )


# =============================================================================
# REACTOR BLAST
# =============================================================================

# Blast radius: REACTOR_BLAST_RADIUS_KM for a hull of
# REACTOR_BLAST_REFERENCE_MASS_TONS dry mass (a cruiser), scaling with the
# square root of dry mass (corvette/frigate 1895 t -> 10.6 km, destroyer ->
# 13.0 km, cruiser 3790 t -> 15 km, battleship -> 18.4 km, dreadnought ->
# 21.2 km). Ships in a formation sit 5-20 km apart, so a death catches a
# close neighbour but never a whole fleet.
REACTOR_BLAST_RADIUS_KM = 15.0
REACTOR_BLAST_REFERENCE_MASS_TONS = 3790.0
# Nominal yield, recorded for the viewer (flash size); effects use intensity.
REACTOR_BLAST_YIELD_GJ_PER_TON = 0.15
# Intensity at distance d is (1 - d / radius)^2: 1.0 at the hull, 0.44 at a
# third of the radius, 0.11 at two thirds, 0 at the edge.
# Armour on the facing toward the blast is ablated by intensity x this (cm).
REACTOR_BLAST_MAX_ARMOR_ABLATION_CM = 12.0
# Heat dumped into the victim's sink at intensity 1.0 (GJ; sinks hold 320 GJ).
REACTOR_BLAST_MAX_HEAT_GJ = 80.0
# Damage to each EXTENDED radiator panel at intensity 1.0 (GJ; a panel has
# ~5 GJ of health). Retracted panels are shielded by the hull.
REACTOR_BLAST_RADIATOR_DAMAGE_GJ = 6.0
# Each exposed fitting is caught with probability intensity x this, and then
# loses intensity x REACTOR_BLAST_FITTING_DAMAGE_PCT of its health.
REACTOR_BLAST_FITTING_HIT_CHANCE = 0.6
REACTOR_BLAST_FITTING_DAMAGE_PCT = 80.0
# A dying hulk inside the blast loses intensity x this many seconds of fuse.
REACTOR_BLAST_FUSE_CUT_S = 20.0
# Torpedoes: destroyed inside this fraction of the radius, seeker blinded
# inside the second (cruiser: 3.75 km / 9 km).
REACTOR_BLAST_TORPEDO_KILL_FRACTION = 0.25
REACTOR_BLAST_TORPEDO_BLIND_FRACTION = 0.60
# Effects below this intensity are not applied or recorded.
REACTOR_BLAST_MIN_INTENSITY = 0.01


def reactor_blast_radius_km(dry_mass_tons: float) -> float:
    """Blast radius of a hull of the given dry mass."""
    return REACTOR_BLAST_RADIUS_KM * math.sqrt(
        max(dry_mass_tons, 1.0) / REACTOR_BLAST_REFERENCE_MASS_TONS)


def blast_intensity(distance_km: float, radius_km: float) -> float:
    """Blast intensity 0..1 at a distance from the detonation."""
    if radius_km <= 0.0 or distance_km >= radius_km:
        return 0.0
    return (1.0 - max(0.0, distance_km) / radius_km) ** 2


def _hull_mass_tons(ship: "ShipCombatState") -> float:
    ks = getattr(ship, "kinematic_state", None)
    dry = getattr(ks, "dry_mass_kg", None)
    if dry:
        return float(dry) / 1000.0
    return REACTOR_BLAST_REFERENCE_MASS_TONS


def queue_reactor_blast(sim: "CombatSimulation", ship: "ShipCombatState", cause: str) -> None:
    """Called from _destroy_ship: remember the detonation, resolve it later.

    The death can happen deep inside hit resolution or a ship update; applying
    the blast there would mutate ships and the torpedo list mid-iteration. The
    queue is drained once per tick by resolve_pending_blasts.
    """
    mass_tons = _hull_mass_tons(ship)
    sim._pending_blasts.append({
        'ship_id': ship.ship_id,
        'ship_type': ship.ship_type,
        'position': ship.position,
        'cause': cause,
        'mass_tons': mass_tons,
        'radius_km': reactor_blast_radius_km(mass_tons),
        'yield_gj': REACTOR_BLAST_YIELD_GJ_PER_TON * mass_tons,
    })


def resolve_pending_blasts(sim: "CombatSimulation") -> None:
    """Apply every detonation queued this tick (deaths it causes wait a tick)."""
    if not sim._pending_blasts:
        return
    blasts, sim._pending_blasts = sim._pending_blasts, []
    for blast in blasts:
        _resolve_blast(sim, blast)


def _resolve_blast(sim: "CombatSimulation", blast: dict) -> None:
    SimulationEventType = _event_types(sim)

    origin = blast['position']
    radius_km = blast['radius_km']
    source_id = blast['ship_id']
    rng = sim._damage_rng

    # --- torpedoes -----------------------------------------------------------
    kill_km = radius_km * REACTOR_BLAST_TORPEDO_KILL_FRACTION
    blind_km = radius_km * REACTOR_BLAST_TORPEDO_BLIND_FRACTION
    destroyed_ids: List[str] = []
    blinded_ids: List[str] = []
    for tf in list(sim.torpedoes):
        d_km = tf.torpedo.position.distance_to(origin) / 1000.0
        if d_km <= kill_km:
            destroyed_ids.append(tf.torpedo_id)
            sim.torpedoes.remove(tf)
            sim.metrics.total_torpedo_blast_destroyed += 1
            sim._log_event(SimulationEventType.TORPEDO_INTERCEPTED,
                           tf.source_ship_id, tf.torpedo.target_id, {
                'torpedo_id': tf.torpedo_id,
                'cause': 'reactor_blast',
                'blast_ship_id': source_id,
                'distance_km': round(d_km, 2),
                'total_heat_absorbed_j': tf.heat_absorbed_j,
            })
        elif d_km <= blind_km and not tf.is_disabled:
            tf.is_disabled = True
            # PD must not claim (or re-log) a seeker the blast killed.
            tf.seeker_kill_logged = True
            blinded_ids.append(tf.torpedo_id)
            sim.metrics.total_torpedo_blast_blinded += 1

    # --- ships ---------------------------------------------------------------
    ships_hit: List[str] = []
    for ship in list(sim.ships.values()):
        if ship.ship_id == source_id or ship.is_destroyed:
            continue
        offset = ship.position - origin
        d_km = offset.magnitude / 1000.0
        intensity = blast_intensity(d_km, radius_km)
        if intensity < REACTOR_BLAST_MIN_INTENSITY:
            continue
        ships_hit.append(ship.ship_id)
        data: dict = {
            'source_ship_id': source_id,
            'distance_km': round(d_km, 2),
            'intensity': round(intensity, 3),
        }

        if ship.is_dying:
            cut = intensity * REACTOR_BLAST_FUSE_CUT_S
            ship.dying_time_remaining = max(0.0, ship.dying_time_remaining - cut)
            data['fuse_cut_s'] = round(cut, 1)
            data['detonation_in_s'] = round(ship.dying_time_remaining, 1)
            sim._log_event(SimulationEventType.BLAST_DAMAGE, ship.ship_id, data=data)
            continue

        # Armour on the facing toward the blast.
        facing = HitLocation.LATERAL
        if ship.geometry and offset.magnitude > 1e-6:
            facing = ship.geometry.calculate_hit_location(offset.normalized(), ship.forward)
        ablation = 0.0
        remaining = None
        if ship.armor:
            section = ship.armor.get_section(facing)
            if section:
                ablation = min(section.thickness_cm,
                               intensity * REACTOR_BLAST_MAX_ARMOR_ABLATION_CM)
                section.thickness_cm -= ablation
                remaining = section.thickness_cm
        data['hit_location'] = facing.value
        data['armor_ablation_cm'] = round(ablation, 2)
        if remaining is not None:
            data['armor_remaining_cm'] = round(remaining, 2)

        # Heat pulse and exposed radiators.
        heat_gj = 0.0
        panels: List[dict] = []
        thermal = ship.thermal_system
        if thermal is not None:
            heat_gj = intensity * REACTOR_BLAST_MAX_HEAT_GJ
            thermal.add_heat('reactor_blast', heat_gj)
            for position in sorted(thermal.radiators.radiators, key=lambda p: p.value):
                if not thermal.is_radiator_extended(position):
                    continue
                panel = thermal.radiators.radiators[position]
                capacity_before = _panel_capacity_kw(panel)
                _taken, destroyed, _lost = thermal.apply_radiator_damage(
                    position, intensity * REACTOR_BLAST_RADIATOR_DAMAGE_GJ)
                panels.append({
                    'radiator_position': position.value,
                    'health_pct': round(panel.health_percent, 1),
                    'destroyed': destroyed,
                    'capacity_lost_kw': round(capacity_before - _panel_capacity_kw(panel), 1),
                })
            data['heat'] = round(thermal.heat_fraction, 3)
        data['heat_gj'] = round(heat_gj, 1)
        data['radiators'] = panels

        sim._log_event(SimulationEventType.BLAST_DAMAGE, ship.ship_id, data=data)

        # Exposed fittings (each recorded as its own mount_damaged event).
        for fitting in _gun_fittings(ship, None) + _pd_fittings(ship):
            if rng.random() < intensity * REACTOR_BLAST_FITTING_HIT_CHANCE:
                damage_fitting(
                    sim, ship, fitting,
                    health_loss_pct=intensity * REACTOR_BLAST_FITTING_DAMAGE_PCT,
                    cause='reactor_blast',
                    source_ship_id=source_id,
                    hit_location=facing,
                )

    sim._log_event(SimulationEventType.REACTOR_BLAST, source_id, data={
        'ship_type': blast['ship_type'],
        'cause': blast['cause'],
        'position': [round(origin.x, 1), round(origin.y, 1), round(origin.z, 1)],
        'radius_km': round(radius_km, 2),
        'torpedo_kill_radius_km': round(kill_km, 2),
        'torpedo_blind_radius_km': round(blind_km, 2),
        'yield_gj': round(blast['yield_gj'], 1),
        'ships_hit': ships_hit,
        'torpedoes_destroyed': destroyed_ids,
        'torpedoes_blinded': blinded_ids,
    })
    if ships_hit or destroyed_ids or blinded_ids:
        print(f"  *** [{source_id}] REACTOR BLAST ({radius_km:.1f} km): "
              f"{len(ships_hit)} ships caught, {len(destroyed_ids)} torpedoes "
              f"destroyed, {len(blinded_ids)} blinded")


def _panel_capacity_kw(panel: Any) -> float:
    """Rejection a panel could give if extended now (0 once destroyed)."""
    if panel.health_percent <= 0.0:
        return 0.0
    return panel.max_dissipation_kw * panel.health_percent / 100.0


# =============================================================================
# MAGAZINE COOK-OFF
# =============================================================================

# Share of the ship's remaining rounds a destroyed magazine burns off.
COOKOFF_ROUNDS_LOST_FRACTION = 0.5
# Secondary explosion energy per torpedo lost (GJ) and its cap; a gun magazine
# (inert slugs, capacitor banks) gives a small fixed secondary instead.
COOKOFF_GJ_PER_TORPEDO = 0.75
COOKOFF_MAX_GJ = 6.0
COOKOFF_GUN_MAGAZINE_GJ = 1.5


def resolve_magazine_cookoff(
    sim: "CombatSimulation",
    ship: "ShipCombatState",
    module: Any,
    attacker_id: Optional[str],
) -> None:
    """A MAGAZINE module was just destroyed: burn rounds, blow the neighbours."""
    SimulationEventType = _event_types(sim)

    if getattr(module, "module_type", None) != ModuleType.MAGAZINE:
        return

    torpedoes_lost = 0
    for launcher in ship.ready_torpedo_launchers:
        have = int(getattr(launcher, "current_magazine", 0) or 0)
        lost = int(math.ceil(have * COOKOFF_ROUNDS_LOST_FRACTION))
        if lost > 0:
            launcher.current_magazine = have - lost
            torpedoes_lost += lost

    gun_rounds_lost = 0
    for slot, ws in ship.weapons.items():
        if slot.startswith("pd") or not _is_gun(ws):
            continue
        have = int(ws.ammo_remaining)
        lost = int(math.ceil(have * COOKOFF_ROUNDS_LOST_FRACTION))
        if lost > 0:
            ws.ammo_remaining = have - lost
            gun_rounds_lost += lost

    if torpedoes_lost > 0:
        secondary_gj = min(COOKOFF_MAX_GJ, torpedoes_lost * COOKOFF_GJ_PER_TORPEDO)
    elif gun_rounds_lost > 0:
        secondary_gj = COOKOFF_GUN_MAGAZINE_GJ
    else:
        secondary_gj = 0.0

    # The secondary tears into the compartments next to the magazine: same
    # layer first, then the layers fore and aft of it. The bridge and the
    # reactor sit in their own citadels and are never reached, so a cook-off
    # cripples a ship but is not by itself a kill.
    wrecked: List[str] = []
    damaged: List[str] = []
    remaining = secondary_gj
    layout = ship.module_layout
    if layout is not None and remaining > 0.0:
        layer_idx = module.position.layer_index
        neighbours = [
            m for layer in layout.layers
            for m in layer.modules
            if m is not module and not m.is_destroyed and not m.is_critical
            and abs(layer.layer_index - layer_idx) <= 1
        ]
        neighbours.sort(key=lambda m: (abs(m.position.layer_index - layer_idx),
                                       abs(m.position.lateral_offset
                                           - module.position.lateral_offset),
                                       m.name))
        for neighbour in neighbours:
            if remaining < 0.05:
                break
            before = remaining
            remaining = neighbour.damage(remaining)
            applied = before - remaining
            if applied <= 0.0:
                continue
            damaged.append(neighbour.name)
            sim._log_event(SimulationEventType.MODULE_DAMAGED, ship.ship_id, data={
                'module_name': neighbour.name,
                'damage_gj': applied,
                'destroyed': neighbour.is_destroyed,
                'health_pct': round(neighbour.health_percent, 1),
                'cause': 'magazine_cookoff',
            })
            if neighbour.is_destroyed:
                wrecked.append(neighbour.name)
                sim._log_event(SimulationEventType.MODULE_DESTROYED, ship.ship_id, data={
                    'module_name': neighbour.name,
                    'cause': 'magazine_cookoff',
                })
                sim._disable_weapon_for_module(ship, neighbour.name)

    sim._log_event(SimulationEventType.MAGAZINE_COOKOFF, ship.ship_id, data={
        'module_name': module.name,
        'torpedoes_lost': torpedoes_lost,
        'torpedoes_remaining': ship.torpedoes_remaining,
        'gun_rounds_lost': gun_rounds_lost,
        'secondary_gj': round(secondary_gj, 2),
        'modules_damaged': damaged,
        'modules_destroyed': wrecked,
        'source_ship_id': attacker_id,
    })
    print(f"  !!! [{ship.ship_id}] MAGAZINE COOK-OFF ({module.name}): "
          f"{torpedoes_lost} torpedoes, {gun_rounds_lost} gun rounds lost, "
          f"{secondary_gj:.1f} GJ secondary")
