"""
Drive plume wash (pass 5): a burning torch is a hazard to whatever sits
astern of it.

Every hull flies on the same torch: 58.56 MN at 10,256 km/s exhaust velocity,
i.e. 0.5 x F x v_e = 300 TW of jet power at full throttle. The engine used to
ignore the jet completely; ships flew in each other's exhaust for free. This
module makes the plume a real thing:

* EMITTER. A ship (or a dying hulk whose torch still sputters) that produced
  thrust this tick projects a narrow cone astern, along the exact opposite of
  its nose vector (the engine never deflects the nozzle: _update_ship always
  propagates with zero gimbal, so the thrust axis IS the nose axis). The jet
  power is the thrust actually produced: applied throttle (which already
  includes engine damage and dry tanks) x thrust x exhaust velocity / 2. A
  dead, throttled-back or damaged drive washes less or nothing.
* VICTIM. Any other ship - friend or foe - inside the cone takes heat into
  its sink and, where the wash is strong, loses armour on the facing it shows
  the jet and has the fittings on that facing burned (battle_damage's
  fitting machinery, cause 'plume_wash').
* TORPEDOES (PLUME_AFFECTS_TORPEDOES, off by default in pass 5): a torpedo
  flying up a burning ship's exhaust accumulates a dose; its seeker is
  blinded, then the round is destroyed.

Scaling (a game-design decision, not physics). Taken literally the jet is a
weapon of mass destruction: 300 TW in a 3 degree cone is ~200 MW/m^2 at 20 km,
about 400 GJ/s on a 2000 m^2 flank - a full 320 GJ heat sink every second.
The game keeps the shape (inverse square, narrow Gaussian cone) and scales the
coupling down by roughly 1/85 (PLUME_HEAT_GJ_PER_S_AT_10KM: 16 GJ/s where the
literal number is ~1500): the plasma is thin, fast and mostly streams round a
hull. The near field is clamped (PLUME_NEAR_FIELD_KM) so a merge at 1-2 km is
not an instant kill, and the heat stops just under the meltdown line (the
same ceiling as a reactor blast): a wash cripples - guns hold at 90 % heat,
armour and fittings burn off - and anything else the victim then does to its
own heat balance finishes it, but the wash alone never detonates a reactor.

No RNG at all. The hook (resolve_plume_wash, called once per tick from
CombatSimulation.step) computes every wash from ONE snapshot of positions and
thrust and applies them afterwards in ship-id order, so the result does not
depend on which ship or side is processed first. Shadow simulations are
inert: the trajectory-prediction shadow never calls step(), and the
torpedo-forecast shadow switches the hook off (sim.plume_wash_enabled).

Recorded as `plume_wash` events; see docs/pass-5-recording-additions.md.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    from . import battle_damage
    from .combat import HitLocation
except ImportError:  # script-style import path (src/ itself on sys.path)
    try:
        import battle_damage
        from combat import HitLocation
    except ImportError:
        # `llm` imported as a top-level package (tests/test_mcp.py): the
        # engine modules only load through the `src` package.
        from src import battle_damage
        from src.combat import HitLocation

if TYPE_CHECKING:  # pragma: no cover
    from .simulation import CombatSimulation, ShipCombatState


# =============================================================================
# CONSTANTS
# =============================================================================

# Jet power of the reference torch at full throttle: 0.5 x 58.56 MN x
# 10,256 km/s = 300.3 TW. Every hull in data/fleet_ships.json mounts this
# drive, so today all torches are equal; a different drive scales linearly.
PLUME_REFERENCE_JET_POWER_W = 0.5 * 58.56e6 * 10_256e3

# Cone: Gaussian in the angle off the exhaust axis, exp(-(angle/core)^2),
# with a hard edge. 3 deg core / 7.5 deg edge: at 10 km the core is ~0.5 km
# wide and the edge 1.3 km from the axis; a wingman 12 km abeam is far out.
PLUME_CORE_HALF_ANGLE_DEG = 3.0
PLUME_MAX_HALF_ANGLE_DEG = 7.5

# Heat into the victim's sink, on the axis, 10 km astern of the reference
# torch at full throttle (GJ/s). Sinks hold 320 GJ; radiators reject 0.78.
PLUME_HEAT_GJ_PER_S_AT_10KM = 16.0
# Inverse square with a near-field clamp: rate ~ 1 / (d^2 + near^2). Inside
# ~5 km the rate flattens (80 GJ/s at the nozzle instead of infinity).
PLUME_NEAR_FIELD_KM = 5.0
# Hard outer limit, and the smallest rate that counts as a wash at all.
# Beyond either the effect is exactly zero (bit-identical battles for ships
# that are in nobody's exhaust).
PLUME_MAX_RANGE_KM = 100.0
PLUME_MIN_HEAT_GJ_PER_S = 0.05

# The heat never takes a sink past this fraction of its rating (the reactor
# blast's ceiling, just under REACTOR_MELTDOWN_HEAT_FRACTION).
PLUME_HEAT_CEILING_FRACTION = battle_damage.REACTOR_BLAST_HEAT_CEILING_FRACTION

# Close in, the wash also burns the hull. Only the part of the heat rate ABOVE
# this threshold ablates armour and damages fittings (the reference torch at
# full throttle exceeds it inside ~21 km on the axis).
PLUME_ABLATION_THRESHOLD_GJ_PER_S = 4.0
# Armour burned off the washed facing per GJ above the threshold (cm).
PLUME_ARMOR_ABLATION_CM_PER_GJ = 0.03
# Health lost by every exposed fitting on the washed facing per GJ above the
# threshold (percent), applied in steps of PLUME_FITTING_DAMAGE_QUANTUM_PCT so
# a sustained wash does not record one mount_damaged per fitting per tick.
PLUME_FITTING_DAMAGE_PCT_PER_GJ = 0.5
PLUME_FITTING_DAMAGE_QUANTUM_PCT = 25.0

# A turning ship sweeps its jet. Within one tick the thrust axis is sampled
# every PLUME_SWEEP_STEP_DEG between the previous and the present nose
# (half the core half-angle, at most PLUME_SWEEP_MAX_SAMPLES samples) and the
# wash is the mean over the samples, so a fast sweep cannot step over a ship
# between two ticks: it deposits its dwell-weighted share and is recorded
# with the peak rate of the crossing.
PLUME_SWEEP_STEP_DEG = 1.5
PLUME_SWEEP_MAX_SAMPLES = 40

# Crew radiation. The jet is intensely radioactive; time in it gives the
# crew a dose that stays for the battle. Dose rate = wash rate (the full
# rate, not the heat the sink still accepts) x PLUME_DOSE_SV_PER_GJ x
# shielding, where the armour on the facing shown to the jet shields as
# exp(-thickness / PLUME_DOSE_SHIELD_EFOLD_CM): a battleship's 262 cm bow
# passes 7 %, a destroyer's 151 cm bow 22 %, a 26 cm flank 77 %, a frigate's
# 12 cm flank 88 %.
PLUME_DOSE_SV_PER_GJ = 0.03
PLUME_DOSE_SHIELD_EFOLD_CM = 100.0
# Impaired crew (acute radiation sickness): every gun and torpedo tube cycles
# PLUME_CREW_IMPAIRED_CYCLE_FACTOR times slower (the mechanism damaged mounts
# use; hit rolls, lead and evasion are untouched).
PLUME_CREW_IMPAIRED_SV = 2.0
PLUME_CREW_IMPAIRED_CYCLE_FACTOR = 1.5
# Incapacitated crew: the ship stops fighting and maneuvering (it coasts;
# automatic PD keeps running) and counts as unable to deal damage.
PLUME_CREW_INCAPACITATED_SV = 8.0

# A sustained wash records `plume_wash` on entry, every this many seconds
# while it lasts, and on exit (with the totals of the whole episode).
PLUME_EVENT_INTERVAL_S = 5.0

# Torpedoes in the plume. OFF in pass 5: a fleeing ship's tail chase is flown
# straight up its exhaust, so switching this on changes torpedo balance (see
# the plume lane's report for seeded measurements).
PLUME_AFFECTS_TORPEDOES = False
# Dose is the ship-equivalent heat (GJ) the round has flown through. The
# seeker dies at the first value, the round is destroyed at the second. A
# torpedo is never washed by the ship that launched it (tubes eject clear of
# the jet).
PLUME_TORPEDO_BLIND_DOSE_GJ = 2.0
PLUME_TORPEDO_KILL_DOSE_GJ = 10.0

# Rule-based captains (src/llm/heuristic_tactics.py) and formation placement
# (src/llm/fleet_draft.py) keep friendlies out of the cone with this margin:
# a friendly counts as "in my exhaust" inside PLUME_AVOID_HALF_ANGLE_DEG of
# dead astern and PLUME_AVOID_RANGE_KM.
PLUME_AVOID_HALF_ANGLE_DEG = 10.0
PLUME_AVOID_RANGE_KM = 60.0

_COS_MAX = math.cos(math.radians(PLUME_MAX_HALF_ANGLE_DEG))


# =============================================================================
# MODEL
# =============================================================================

def jet_power_w(thrust_n: float, exhaust_velocity_ms: float, throttle: float) -> float:
    """Jet power of a torch producing ``throttle`` x ``thrust_n`` (W)."""
    return 0.5 * max(0.0, throttle) * max(0.0, thrust_n) * max(0.0, exhaust_velocity_ms)


def plume_heat_rate_gj_s(jet_w: float, distance_km: float, off_axis_deg: float) -> float:
    """Heat a ship takes (GJ/s) at a distance and angle off the exhaust axis.

    Exactly 0.0 outside the cone, beyond PLUME_MAX_RANGE_KM, or when the
    rate would be under PLUME_MIN_HEAT_GJ_PER_S.
    """
    if jet_w <= 0.0 or distance_km > PLUME_MAX_RANGE_KM:
        return 0.0
    if off_axis_deg > PLUME_MAX_HALF_ANGLE_DEG or off_axis_deg < 0.0:
        return 0.0
    near2 = PLUME_NEAR_FIELD_KM ** 2
    rate = (PLUME_HEAT_GJ_PER_S_AT_10KM
            * (jet_w / PLUME_REFERENCE_JET_POWER_W)
            * (100.0 + near2) / (max(0.0, distance_km) ** 2 + near2)
            * math.exp(-(off_axis_deg / PLUME_CORE_HALF_ANGLE_DEG) ** 2))
    return rate if rate >= PLUME_MIN_HEAT_GJ_PER_S else 0.0


def plume_armor_ablation_cm_s(heat_rate_gj_s: float) -> float:
    """Armour burned off the washed facing per second at a heat rate."""
    return max(0.0, heat_rate_gj_s - PLUME_ABLATION_THRESHOLD_GJ_PER_S) * PLUME_ARMOR_ABLATION_CM_PER_GJ


def plume_ablation_range_km(jet_w: float = PLUME_REFERENCE_JET_POWER_W) -> float:
    """On-axis distance inside which a torch of this jet power burns armour
    and fittings (heat rate above PLUME_ABLATION_THRESHOLD_GJ_PER_S)."""
    near2 = PLUME_NEAR_FIELD_KM ** 2
    peak = PLUME_HEAT_GJ_PER_S_AT_10KM * (jet_w / PLUME_REFERENCE_JET_POWER_W) * (100.0 + near2)
    d2 = peak / PLUME_ABLATION_THRESHOLD_GJ_PER_S - near2
    return math.sqrt(d2) if d2 > 0.0 else 0.0


def dose_shielding(armor_cm: float) -> float:
    """Fraction of the jet's radiation that passes this much armour (0..1)."""
    return math.exp(-max(0.0, armor_cm) / PLUME_DOSE_SHIELD_EFOLD_CM)


def plume_dose_rate_sv_s(heat_rate_gj_s: float, armor_cm: float) -> float:
    """Crew dose rate (Sv/s) at a wash rate behind the given armour."""
    return max(0.0, heat_rate_gj_s) * PLUME_DOSE_SV_PER_GJ * dose_shielding(armor_cm)


def crew_state(ship: Any) -> str:
    """'fit', 'impaired' or 'incapacitated' (plume radiation)."""
    if getattr(ship, "crew_incapacitated", False) is True:
        return "incapacitated"
    if getattr(ship, "crew_impaired", False) is True:
        return "impaired"
    return "fit"


def sweep_axes(previous: Optional[Tuple[float, float, float]],
               current: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    """Thrust-axis samples for one tick: the present nose, preceded by the
    intermediate directions back toward where it pointed a tick ago when the
    ship turned more than PLUME_SWEEP_STEP_DEG while burning."""
    if previous is None:
        return [current]
    dot = previous[0] * current[0] + previous[1] * current[1] + previous[2] * current[2]
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    if angle <= PLUME_SWEEP_STEP_DEG:
        return [current]
    n = min(PLUME_SWEEP_MAX_SAMPLES, int(math.ceil(angle / PLUME_SWEEP_STEP_DEG)))
    out = []
    for k in range(1, n + 1):
        f = k / n
        x = previous[0] + (current[0] - previous[0]) * f
        y = previous[1] + (current[1] - previous[1]) * f
        z = previous[2] + (current[2] - previous[2]) * f
        mag = math.sqrt(x * x + y * y + z * z)
        out.append((x / mag, y / mag, z / mag) if mag > 1e-9 else current)
    return out


def emitter_throttle(ship: Any) -> float:
    """Throttle the ship's torch actually produced on the last tick (0..1).

    applied_throttle / dying_thrust already include engine damage and dry
    tanks; a destroyed ship emits nothing.
    """
    if getattr(ship, "is_destroyed", False):
        return 0.0
    if getattr(ship, "is_dying", False):
        value = getattr(ship, "dying_thrust", 0.0)
    else:
        value = getattr(ship, "applied_throttle", 0.0)
    return float(value) if isinstance(value, (int, float)) and value > 0.0 else 0.0


def emitter_jet_power_w(ship: Any, throttle: Optional[float] = None) -> float:
    """Jet power of a ship's torch at ``throttle`` (default: what it produced
    on the last tick). An explicit throttle is still scaled by drive damage."""
    ks = getattr(ship, "kinematic_state", None)
    if ks is None:
        return 0.0
    if throttle is None:
        throttle = emitter_throttle(ship)
    elif getattr(ship, "is_destroyed", False):
        throttle = 0.0
    else:
        fraction = getattr(ship, "get_effective_thrust_fraction", None)
        throttle = max(0.0, min(1.0, throttle)) * (fraction() if callable(fraction) else 1.0)
    return jet_power_w(ks.thrust_n, ks.exhaust_velocity_ms, throttle)


def cone_geometry(emitter_pos: Any, emitter_forward: Any, point: Any) -> Optional[Tuple[float, float]]:
    """(distance_km, off_axis_deg) of ``point`` in the exhaust cone of a ship
    at ``emitter_pos`` with nose ``emitter_forward``; None when the point is
    outside the cone or beyond PLUME_MAX_RANGE_KM. Cheap rejection first:
    range, then the half-space astern, then the cone angle."""
    rx = point.x - emitter_pos.x
    ry = point.y - emitter_pos.y
    rz = point.z - emitter_pos.z
    d2 = rx * rx + ry * ry + rz * rz
    limit_m = PLUME_MAX_RANGE_KM * 1000.0
    if d2 > limit_m * limit_m or d2 <= 0.0:
        return None
    axial = -(rx * emitter_forward.x + ry * emitter_forward.y + rz * emitter_forward.z)
    if axial <= 0.0:
        return None
    d = math.sqrt(d2)
    cos_angle = axial / d
    if cos_angle < _COS_MAX:
        return None
    return d / 1000.0, math.degrees(math.acos(min(1.0, cos_angle)))


@dataclass
class Wash:
    """One emitter washing one ship on one tick (or in a what-if query)."""
    emitter_id: str
    victim_id: str
    distance_km: float
    off_axis_deg: float
    heat_rate_gj_s: float       # mean over the tick (a sweep dwells only part of it)
    throttle: float
    friendly: bool
    peak_rate_gj_s: float = 0.0  # highest instantaneous rate within the tick
    swept: bool = False          # the jet swept across the ship this tick


def compute_washes(sim: Any, throttle_override: Optional[Dict[str, float]] = None,
                   previous_axes: Optional[Dict[str, Tuple[float, float, float]]] = None,
                   axes_out: Optional[Dict[str, Tuple[float, float, float]]] = None) -> List[Wash]:
    """Every wash in the current state, sorted by (victim, emitter).

    Pure geometry from one snapshot: nothing is mutated. ``throttle_override``
    maps ship_id -> throttle for what-if queries ("would my burn wash
    someone?"); other ships use what they actually produced last tick.
    ``previous_axes`` (ship_id -> nose unit vector a tick ago, for ships that
    were burning then) makes a turning emitter sweep its jet over the sector
    in between (sweep_axes); ``axes_out`` receives this tick's emitter axes.
    """
    ships = [s for s in sim.ships.values() if not s.is_destroyed]
    emitters = []
    for s in ships:
        override = throttle_override.get(s.ship_id) if throttle_override else None
        power = emitter_jet_power_w(s, override)
        if power > 0.0:
            throttle = override if override is not None else emitter_throttle(s)
            emitters.append((s, power, throttle))
    if not emitters:
        return []
    out: List[Wash] = []
    # The per-pair test is inlined on plain floats (same rejection order as
    # cone_geometry: range, half-space astern, cone angle): almost every
    # pair leaves at the first or second comparison.
    points = [(v, v.position) for v in ships]
    points = [(v, p.x, p.y, p.z) for v, p in points]
    limit2 = (PLUME_MAX_RANGE_KM * 1000.0) ** 2
    for e, power, throttle in emitters:
        e_pos = e.position
        e_fwd = e.forward
        ex, ey, ez = e_pos.x, e_pos.y, e_pos.z
        fx, fy, fz = e_fwd.x, e_fwd.y, e_fwd.z
        if axes_out is not None:
            axes_out[e.ship_id] = (fx, fy, fz)
        axes = None
        if previous_axes:
            axes = sweep_axes(previous_axes.get(e.ship_id), (fx, fy, fz))
            if len(axes) == 1:
                axes = None
        for v, px, py, pz in points:
            rx = px - ex
            ry = py - ey
            rz = pz - ez
            d2 = rx * rx + ry * ry + rz * rz
            if d2 > limit2 or d2 <= 0.0:
                continue
            if axes is not None:
                # Sweeping jet: mean rate over the sampled axes, geometry of
                # the strongest sample.
                d = math.sqrt(d2)
                d_km = d / 1000.0
                total = 0.0
                best_rate = 0.0
                best_angle = 0.0
                for ax_x, ax_y, ax_z in axes:
                    cos_angle = -(rx * ax_x + ry * ax_y + rz * ax_z) / d
                    if cos_angle < _COS_MAX:
                        continue
                    angle = math.degrees(math.acos(min(1.0, cos_angle)))
                    r = plume_heat_rate_gj_s(power, d_km, angle)
                    total += r
                    if r > best_rate:
                        best_rate, best_angle = r, angle
                mean = total / len(axes)
                if best_rate > 0.0 and mean > 0.0:
                    out.append(Wash(e.ship_id, v.ship_id, d_km, best_angle, mean, throttle,
                                    e.faction == v.faction, best_rate, True))
                continue
            axial = -(rx * fx + ry * fy + rz * fz)
            if axial <= 0.0:
                continue
            d = math.sqrt(d2)
            cos_angle = axial / d
            if cos_angle < _COS_MAX:
                continue
            d_km = d / 1000.0
            angle = math.degrees(math.acos(min(1.0, cos_angle)))
            rate = plume_heat_rate_gj_s(power, d_km, angle)
            if rate <= 0.0:
                continue
            out.append(Wash(e.ship_id, v.ship_id, d_km, angle, rate, throttle,
                            e.faction == v.faction, rate, False))
    out.sort(key=lambda w: (w.victim_id, w.emitter_id))
    return out


# =============================================================================
# ENGINE HOOK
# =============================================================================

def _event_types(sim: Any) -> Any:
    return sys.modules[type(sim).__module__].SimulationEventType


def _state(sim: Any) -> dict:
    state = getattr(sim, "_plume_state", None)
    if state is None:
        state = {
            'contacts': {},        # (emitter_id, victim_id) -> episode dict
            'fitting_debt': {},    # victim_id -> pending fitting health loss (%)
            'torpedo_dose': {},    # torpedo_id -> dose (GJ)
            'stats': {
                'friendly_ship_seconds': 0.0, 'enemy_ship_seconds': 0.0,
                'friendly_heat_gj': 0.0, 'enemy_heat_gj': 0.0,
                'friendly_dose_sv': 0.0, 'enemy_dose_sv': 0.0,
                'crews_impaired': 0, 'crews_incapacitated': 0,
                'armor_ablation_cm': 0.0, 'fittings_hit': 0, 'fittings_destroyed': 0,
                'torpedoes_blinded': 0, 'torpedoes_destroyed': 0,
            },
        }
        sim._plume_state = state
    return state


def plume_stats(sim: Any) -> dict:
    """Battle totals of the wash so far (a copy)."""
    return dict(_state(sim)['stats'])


def resolve_plume_wash(sim: "CombatSimulation", dt: float) -> None:
    """Apply this tick's drive-plume wash. Called once per tick from step(),
    after every ship, projectile and torpedo has moved."""
    if not getattr(sim, "plume_wash_enabled", True):
        return
    axes: Dict[str, Tuple[float, float, float]] = {}
    washes = compute_washes(sim, None, getattr(sim, "_plume_axes", None), axes)
    sim._plume_axes = axes       # this tick's burning emitters, for the next sweep
    state = getattr(sim, "_plume_state", None)
    if not washes and state is None:
        if PLUME_AFFECTS_TORPEDOES and sim.torpedoes:
            _wash_torpedoes(sim, dt)
        return
    state = _state(sim)
    contacts = state['contacts']
    stats = state['stats']
    now = sim.current_time
    SimulationEventType = _event_types(sim)

    seen = set()
    for wash in washes:                      # sorted: order-independent
        victim = sim.ships[wash.victim_id]
        key = (wash.emitter_id, wash.victim_id)
        seen.add(key)
        side = 'friendly' if wash.friendly else 'enemy'
        stats[f'{side}_ship_seconds'] += dt
        if victim.is_dying:
            # A hulk has no heat management left to upset and no captain to
            # warn; the contact is tracked (and recorded) but does nothing.
            heat_gj = 0.0
            ablation = 0.0
            facing = None
            remaining = None
            fittings: List[str] = []
            dose = 0.0
        else:
            heat_gj, ablation, facing, remaining, fittings, dose = _apply_to_ship(
                sim, victim, wash, dt)
        stats[f'{side}_heat_gj'] += heat_gj
        stats[f'{side}_dose_sv'] += dose
        stats['armor_ablation_cm'] += ablation

        episode = contacts.get(key)
        phase = None
        if episode is None:
            episode = contacts[key] = {
                'start': now, 'last_event': now, 'heat_gj': 0.0,
                'armor_ablation_cm': 0.0, 'fittings': [], 'peak': 0.0,
                'friendly': wash.friendly, 'dose_sv': 0.0, 'swept': False,
            }
            phase = 'enter'
        episode['heat_gj'] += heat_gj
        episode['armor_ablation_cm'] += ablation
        episode['fittings'].extend(f for f in fittings if f not in episode['fittings'])
        episode['peak'] = max(episode['peak'], wash.peak_rate_gj_s)
        episode['dose_sv'] += dose
        episode['swept'] = episode['swept'] or wash.swept
        episode['last'] = (wash.distance_km, wash.off_axis_deg, wash.heat_rate_gj_s,
                           wash.throttle, facing, remaining)
        episode['end'] = now + dt
        if phase is None and now - episode['last_event'] >= PLUME_EVENT_INTERVAL_S - 1e-9:
            phase = 'sustain'
        if phase is not None:
            episode['last_event'] = now
            _log(sim, SimulationEventType, key, episode, phase, victim)

    for key in sorted(k for k in contacts if k not in seen):
        episode = contacts.pop(key)
        _log(sim, SimulationEventType, key, episode, 'exit', sim.ships.get(key[1]))

    if PLUME_AFFECTS_TORPEDOES and sim.torpedoes:
        _wash_torpedoes(sim, dt)


def _apply_to_ship(sim: Any, victim: Any, wash: Wash, dt: float) -> tuple:
    """Heat, armour and fittings for one wash. Returns (heat_gj added,
    armour cm lost, facing, armour cm remaining, fittings damaged)."""
    emitter = sim.ships[wash.emitter_id]
    stats = _state(sim)['stats']

    heat_gj = 0.0
    thermal = victim.thermal_system
    if thermal is not None:
        sink = thermal.heatsink
        headroom = PLUME_HEAT_CEILING_FRACTION * sink.capacity_gj - sink.current_heat_gj
        heat_gj = max(0.0, min(wash.heat_rate_gj_s * dt, headroom))
        if heat_gj > 0.0:
            thermal.add_heat('plume_wash', heat_gj)

    # The facing the victim shows the jet (the exhaust travels emitter -> victim).
    facing = HitLocation.LATERAL
    offset = victim.position - emitter.position
    if victim.geometry and offset.magnitude > 1e-6:
        facing = victim.geometry.calculate_hit_location(offset.normalized(), victim.forward)

    excess_gj = max(0.0, wash.heat_rate_gj_s - PLUME_ABLATION_THRESHOLD_GJ_PER_S) * dt
    ablation = 0.0
    remaining = None
    section = victim.armor.get_section(facing) if victim.armor else None
    if section is not None:
        ablation = min(section.thickness_cm, excess_gj * PLUME_ARMOR_ABLATION_CM_PER_GJ)
        section.thickness_cm -= ablation
        remaining = section.thickness_cm

    damaged: List[str] = []
    if excess_gj > 0.0:
        debt = _state(sim)['fitting_debt']
        owed = debt.get(victim.ship_id, 0.0) + excess_gj * PLUME_FITTING_DAMAGE_PCT_PER_GJ
        if owed >= PLUME_FITTING_DAMAGE_QUANTUM_PCT:
            for fitting in battle_damage._exposed_fittings(victim, facing):
                kind, key, _ = fitting
                destroyed = battle_damage.damage_fitting(
                    sim, victim, fitting, health_loss_pct=owed,
                    cause='plume_wash', source_ship_id=wash.emitter_id,
                    hit_location=facing)
                slot = key if kind != 'pd' else f"pd_{key}"
                damaged.append(f"{kind}:{slot}")
                stats['fittings_hit'] += 1
                stats['fittings_destroyed'] += 1 if destroyed else 0
            owed = 0.0
        debt[victim.ship_id] = owed

    # Crew dose: the full wash rate, shielded by what armour is left on the
    # washed facing.
    dose = plume_dose_rate_sv_s(wash.heat_rate_gj_s,
                                section.thickness_cm if section is not None else 0.0) * dt
    if dose > 0.0:
        _irradiate(sim, victim, dose, wash.emitter_id)
    return heat_gj, ablation, facing, remaining, damaged, dose


def _irradiate(sim: Any, victim: Any, dose_sv: float, source_ship_id: Optional[str]) -> None:
    """Add crew dose and apply the threshold effects (each crossed once)."""
    SimulationEventType = _event_types(sim)
    stats = _state(sim)['stats']
    victim.crew_dose_sv = getattr(victim, "crew_dose_sv", 0.0) + dose_sv
    total = victim.crew_dose_sv
    if total >= PLUME_CREW_IMPAIRED_SV and not getattr(victim, "crew_impaired", False):
        victim.crew_impaired = True
        stats['crews_impaired'] += 1
        factor = PLUME_CREW_IMPAIRED_CYCLE_FACTOR
        for slot, ws in victim.weapons.items():
            if not slot.startswith("pd"):
                ws.crew_cycle_factor = factor
        for launcher in victim.ready_torpedo_launchers:
            # Scale the intact cooldown too, so later tube damage
            # (battle_damage.damage_fitting) multiplies on top of it.
            intact = getattr(launcher, "_intact_cooldown_seconds", None)
            if intact is not None:
                launcher._intact_cooldown_seconds = intact * factor
            launcher.cooldown_seconds = launcher.cooldown_seconds * factor
        sim._log_event(SimulationEventType.CREW_RADIATION, victim.ship_id, data={
            'state': 'impaired',
            'dose_sv': round(total, 2),
            'threshold_sv': PLUME_CREW_IMPAIRED_SV,
            'cooldown_factor': factor,
            'source_ship_id': source_ship_id,
        })
        sim._echo(f"  ~~~ [{victim.ship_id}] CREW IMPAIRED by plume radiation "
                  f"({total:.1f} Sv): weapons cycle x{factor:.1f}")
    if total >= PLUME_CREW_INCAPACITATED_SV and not getattr(victim, "crew_incapacitated", False):
        victim.crew_incapacitated = True
        victim.current_maneuver = None
        stats['crews_incapacitated'] += 1
        sim._log_event(SimulationEventType.CREW_RADIATION, victim.ship_id, data={
            'state': 'incapacitated',
            'dose_sv': round(total, 2),
            'threshold_sv': PLUME_CREW_INCAPACITATED_SV,
            'cooldown_factor': None,
            'source_ship_id': source_ship_id,
        })
        sim._echo(f"  ~~~ [{victim.ship_id}] CREW INCAPACITATED by plume radiation "
                  f"({total:.1f} Sv): the ship no longer fights or maneuvers")


def _log(sim: Any, SimulationEventType: Any, key: tuple, episode: dict, phase: str,
         victim: Any) -> None:
    emitter_id, victim_id = key
    d_km, angle, rate, throttle, facing, remaining = episode['last']
    data = {
        'source_ship_id': emitter_id,
        'phase': phase,
        'friendly': episode['friendly'],
        'distance_km': round(d_km, 2),
        'off_axis_deg': round(angle, 2),
        'heat_rate_gj_s': 0.0 if phase == 'exit' else round(rate, 2),
        'peak_heat_rate_gj_s': round(episode['peak'], 2),
        'emitter_throttle': round(throttle, 3),
        'duration_s': round(episode['end'] - episode['start'], 1),
        'heat_gj': round(episode['heat_gj'], 1),
        'armor_ablation_cm': round(episode['armor_ablation_cm'], 2),
        'fittings_hit': list(episode['fittings']),
        'hit_location': facing.value if facing is not None else None,
        'swept': episode['swept'],
        'dose_sv': round(episode['dose_sv'], 3),
    }
    if victim is not None and facing is not None:
        data['crew_dose_sv'] = round(getattr(victim, 'crew_dose_sv', 0.0), 3)
        data['crew_state'] = crew_state(victim)
    if remaining is not None:
        data['armor_remaining_cm'] = round(remaining, 2)
    thermal = getattr(victim, 'thermal_system', None) if victim is not None else None
    if thermal is not None and facing is not None:
        data['heat'] = round(thermal.heat_fraction, 3)
    sim._log_event(SimulationEventType.PLUME_WASH, victim_id, data=data)
    if phase == 'enter':
        sim._echo(f"  ~~~ [{victim_id}] PLUME WASH: in {emitter_id}'s exhaust, "
                  f"{d_km:.1f} km astern, +{rate:.1f} GJ/s")


def _wash_torpedoes(sim: Any, dt: float) -> None:
    """Dose every torpedo flying in a burning ship's exhaust (switch ON only)."""
    SimulationEventType = _event_types(sim)
    state = _state(sim)
    doses = state['torpedo_dose']
    stats = state['stats']
    emitters = [(s, emitter_jet_power_w(s)) for s in sim.ships.values() if not s.is_destroyed]
    emitters = sorted(((s, p) for s, p in emitters if p > 0.0), key=lambda sp: sp[0].ship_id)
    if not emitters:
        return
    for tf in list(sim.torpedoes):
        rate = 0.0
        worst = None
        for e, power in emitters:
            if e.ship_id == tf.source_ship_id:
                continue
            hit = cone_geometry(e.position, e.forward, tf.torpedo.position)
            if hit is None:
                continue
            r = plume_heat_rate_gj_s(power, hit[0], hit[1])
            if r > 0.0:
                rate += r
                if worst is None or r > worst[1]:
                    worst = (e.ship_id, r, hit[0], hit[1])
        if worst is None:
            continue
        dose = doses.get(tf.torpedo_id, 0.0) + rate * dt
        doses[tf.torpedo_id] = dose
        if dose >= PLUME_TORPEDO_KILL_DOSE_GJ:
            sim.torpedoes.remove(tf)
            stats['torpedoes_destroyed'] += 1
            sim._log_event(SimulationEventType.TORPEDO_INTERCEPTED,
                           tf.source_ship_id, tf.torpedo.target_id, {
                'torpedo_id': tf.torpedo_id,
                'cause': 'plume_wash',
                'plume_ship_id': worst[0],
                'distance_km': round(worst[2], 2),
                'off_axis_deg': round(worst[3], 2),
                'total_heat_absorbed_j': tf.heat_absorbed_j,
            })
        elif dose >= PLUME_TORPEDO_BLIND_DOSE_GJ and not tf.is_disabled:
            tf.is_disabled = True
            tf.blast_blinded = True      # not PD's seeker kill
            tf.seeker_kill_logged = True
            stats['torpedoes_blinded'] += 1
            sim._log_event(SimulationEventType.PLUME_WASH, worst[0], tf.torpedo.target_id, {
                'phase': 'torpedo_blinded',
                'torpedo_id': tf.torpedo_id,
                'torpedo_source_ship_id': tf.source_ship_id,
                'distance_km': round(worst[2], 2),
                'off_axis_deg': round(worst[3], 2),
                'dose_gj': round(dose, 2),
            })


# =============================================================================
# COMMANDER VIEWS (state, not tactics)
# =============================================================================

def _name(ship: Any) -> str:
    return getattr(ship, "name", None) or getattr(ship, "ship_id", "?")


def plume_status_lines(sim: Any, ship: Any) -> List[str]:
    """Warning lines for one ship's status display, from the live geometry:
    whose exhaust it sits in, and which friendlies its own exhaust is washing
    (or would wash at full throttle, when it is not burning right now)."""
    ships = getattr(sim, "ships", None)
    ship_id = getattr(ship, "ship_id", None)
    if not isinstance(ships, dict) or ship_id not in ships:
        return []
    if getattr(ship, "is_destroyed", False) or getattr(ship, "is_dying", False):
        return []
    lines: List[str] = []
    crew: List[str] = []         # the dose line goes after the wash warnings
    dose = getattr(ship, "crew_dose_sv", 0.0)
    if isinstance(dose, (int, float)) and dose >= 0.05:
        state = crew_state(ship)
        if state == "incapacitated":
            crew.append(f"CREW INCAPACITATED by plume radiation ({dose:.1f} Sv): the ship no "
                         f"longer fights or maneuvers")
        elif state == "impaired":
            crew.append(f"CREW IMPAIRED by plume radiation ({dose:.1f} Sv, incapacitated at "
                         f"{PLUME_CREW_INCAPACITATED_SV:.0f}): weapons cycle "
                         f"x{PLUME_CREW_IMPAIRED_CYCLE_FACTOR:.1f}")
        else:
            crew.append(f"CREW DOSE {dose:.1f} Sv from drive plumes (impaired at "
                         f"{PLUME_CREW_IMPAIRED_SV:.0f}, incapacitated at "
                         f"{PLUME_CREW_INCAPACITATED_SV:.0f})")
    try:
        actual = compute_washes(sim)
        for w in actual:
            if w.victim_id == ship_id:
                emitter = ships[w.emitter_id]
                lines.append(
                    f"PLUME WASH: in {_name(emitter)}'s exhaust "
                    f"({'friendly' if w.friendly else 'ENEMY'}), {w.distance_km:.0f} km astern, "
                    f"{w.off_axis_deg:.0f} deg off its axis, +{w.heat_rate_gj_s:.1f} GJ/s")
        washing = {w.victim_id for w in actual if w.emitter_id == ship_id}
        for w in actual:
            if w.emitter_id == ship_id and w.friendly:
                lines.append(
                    f"PLUME WASH: your exhaust is washing friendly {_name(ships[w.victim_id])}, "
                    f"{w.distance_km:.0f} km astern, +{w.heat_rate_gj_s:.1f} GJ/s on it")
        for w in compute_washes(sim, {ship_id: 1.0}):
            if w.emitter_id == ship_id and w.friendly and w.victim_id not in washing:
                lines.append(
                    f"PLUME RISK: friendly {_name(ships[w.victim_id])} is {w.distance_km:.0f} km "
                    f"astern inside your exhaust cone - a full burn on this heading puts "
                    f"+{w.heat_rate_gj_s:.1f} GJ/s on it")
    except Exception:  # noqa: BLE001 - advisory only, never fail a status build
        return lines + crew
    return lines + crew


def friendlies_in_exhaust(sim: Any, ship: Any, forward: Any = None,
                          half_angle_deg: float = PLUME_AVOID_HALF_ANGLE_DEG,
                          range_km: float = PLUME_AVOID_RANGE_KM) -> List[Tuple[Any, float, float]]:
    """Friendly ships astern of ``ship`` inside the avoidance cone when its
    nose points along ``forward`` (default: where it points now):
    [(ship, distance_km, off_axis_deg)], nearest first. For the rule-based
    captains; wider than the damaging cone on purpose."""
    fwd = forward if forward is not None else ship.forward
    mag = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z)
    if mag <= 1e-9:
        return []
    cos_limit = math.cos(math.radians(half_angle_deg))
    out = []
    pos = ship.position
    for other in sim.ships.values():
        if (other is ship or other.faction != ship.faction or other.is_destroyed
                or getattr(other, "is_dying", False)):
            continue
        rx, ry, rz = other.position.x - pos.x, other.position.y - pos.y, other.position.z - pos.z
        d = math.sqrt(rx * rx + ry * ry + rz * rz)
        if d <= 0.0 or d > range_km * 1000.0:
            continue
        cos_angle = -(rx * fwd.x + ry * fwd.y + rz * fwd.z) / (d * mag)
        if cos_angle < cos_limit:
            continue
        out.append((other, d / 1000.0, math.degrees(math.acos(min(1.0, cos_angle)))))
    out.sort(key=lambda t: (t[1], t[0].ship_id))
    return out


def exhaust_clearance_km(sim: Any, ship: Any, forward: Any = None, horizon_s: float = 30.0,
                         step_s: float = 5.0,
                         half_angle_deg: float = PLUME_AVOID_HALF_ANGLE_DEG,
                         range_km: float = PLUME_AVOID_RANGE_KM) -> Optional[float]:
    """Smallest distance (km) at which a friendly sits in this ship's
    avoidance cone astern now or at any ``step_s`` sample over the next
    ``horizon_s``, if the ship burns at full throttle along ``forward``
    (default: its present nose). None when the exhaust stays clear.

    Constant-acceleration extrapolation: this ship at full thrust along
    ``forward``, every friendly on its present velocity plus the thrust it
    produced last tick. It catches the case the snapshot misses: a friendly
    just AHEAD that a full burn overtakes, leaving it dead astern at 2-5 km.
    """
    fwd = forward if forward is not None else ship.forward
    mag = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z)
    if mag <= 1e-9:
        return None
    fx, fy, fz = fwd.x / mag, fwd.y / mag, fwd.z / mag
    ks = ship.kinematic_state
    fraction = getattr(ship, "get_effective_thrust_fraction", None)
    accel = ks.max_acceleration_ms2() * (fraction() if callable(fraction) else 1.0)
    cos_limit = math.cos(math.radians(half_angle_deg))
    limit_m = range_km * 1000.0
    pos, vel = ship.position, ship.velocity
    steps = max(0, int(round(horizon_s / step_s))) if step_s > 0 else 0
    best: Optional[float] = None
    for other in sim.ships.values():
        if (other is ship or other.faction != ship.faction or other.is_destroyed
                or getattr(other, "is_dying", False)):
            continue
        o_pos, o_vel, o_fwd = other.position, other.velocity, other.forward
        o_acc = emitter_throttle(other) * other.kinematic_state.max_acceleration_ms2()
        rx, ry, rz = o_pos.x - pos.x, o_pos.y - pos.y, o_pos.z - pos.z
        vx, vy, vz = o_vel.x - vel.x, o_vel.y - vel.y, o_vel.z - vel.z
        ax = o_acc * o_fwd.x - accel * fx
        ay = o_acc * o_fwd.y - accel * fy
        az = o_acc * o_fwd.z - accel * fz
        for i in range(steps + 1):
            t = i * step_s
            h = 0.5 * t * t
            x, y, z = rx + vx * t + ax * h, ry + vy * t + ay * h, rz + vz * t + az * h
            d = math.sqrt(x * x + y * y + z * z)
            if d <= 0.0 or d > limit_m:
                continue
            if -(x * fx + y * fy + z * fz) / d < cos_limit:
                continue
            if best is None or d / 1000.0 < best:
                best = d / 1000.0
    return best


def formation_plume_conflicts(offsets_km: List[Tuple[float, float, float]],
                              half_angle_deg: float = PLUME_AVOID_HALF_ANGLE_DEG,
                              range_km: float = PLUME_AVOID_RANGE_KM) -> List[Tuple[int, int]]:
    """Pairs (leader index, follower index) of a formation in which the
    follower sits in the leader's exhaust when the fleet burns straight
    ahead. Offsets are admiral-local km: x = depth (+ toward the enemy)."""
    cos_limit = math.cos(math.radians(half_angle_deg))
    out = []
    for i, lead in enumerate(offsets_km):
        for j, tail in enumerate(offsets_km):
            if i == j:
                continue
            astern = lead[0] - tail[0]
            if astern <= 0.0:
                continue
            d = math.dist(lead, tail)
            if d <= range_km and astern / d >= cos_limit:
                out.append((i, j))
    return out
