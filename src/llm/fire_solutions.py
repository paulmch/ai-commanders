"""
Engine-measured torpedo fire solutions for commanders.

What would a torpedo launched NOW do - how long it flies, how hard it lands,
whether it connects at all - and how much point defense covers the target,
and how many friendly rounds are already on their way to it.

Torpedo flight has no honest closed form: guidance holds a closure floor,
burns its surplus inside the no-escape zone, fights an evading target and
dumps what is left in a terminal burn, and range matters enormously (a
zero-closure launch from 15 km lands ~0.5 GJ, from 400 km ~22 GJ). So the
estimate FLIES the round: ``fly_torpedo`` runs the real engine
(CombatSimulation + TorpedoGuidance) on a private two-ship shadow - shooter
and target cloned at their current position, velocity and attitude, point
defense stripped (PD cover is reported separately, as turret counts), the
target either holding course or on EVADE. The shadow has its own seed and
never touches the battle's state or RNG, so asking never changes a battle.

Verdicts (most severe first):
  UNREACHABLE  misses even if the target holds course
  ESCAPABLE    connects only if the target holds course; EVADE escapes it
  TOO CLOSE    connects, but guidance cannot build speed in the range left
  OVERKILL     enough friendly rounds are already inbound
  REACHABLE    connects even against an evading target
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..physics import Vector3D
except ImportError:  # pragma: no cover - script-style imports
    from physics import Vector3D


# Horizon of a shadow flight. A launch at 1500 km lands in ~2 minutes; a
# round still flying after 3 minutes is not a fire solution for a 30 s
# decision cycle.
SHADOW_HORIZON_S = 180.0
# Shadow step: the battle's own time step, clamped. Finer than 0.5 s buys
# nothing measurable; coarser than 1 s starts to move impact speeds.
SHADOW_DT_MIN_S = 0.5
SHADOW_DT_MAX_S = 1.0

# Below this impact energy a torpedo is not worth its tube: about one spinal
# coilgun round (4.3 GJ). Measured on the engine: zero-closure launches from
# 15 / 40 / 100 km land ~0.5 / ~1.6 / ~5 GJ - guidance needs room to build
# speed - against ~19-22 GJ from 250-400 km.
LOW_YIELD_GJ = 5.0
# OVERKILL once this many rounds BEYOND the salvo that saturates the
# target's point defense are already inbound: that is several un-blinded
# torpedoes landing together, more than most hulls survive.
OVERKILL_MARGIN_ROUNDS = 3


@dataclass
class TorpedoShot:
    """Outcome of one shadow flight."""
    hits: bool
    tof_s: Optional[float] = None
    impact_kps: Optional[float] = None
    impact_gj: Optional[float] = None
    note: str = ""          # why it missed (fuel / horizon) when it did


@dataclass
class TorpedoSolution:
    """Everything a commander needs to decide one launch at one target."""
    target_id: str
    target_name: str
    range_km: float
    closing_kps: float              # + closing, - opening (launcher->target)
    vs_evading: Optional[TorpedoShot]
    vs_coasting: Optional[TorpedoShot]
    pd_target: int                  # target's own live PD turrets
    pd_escorts: int                 # his allies' live turrets within PD range of him
    salvo_to_saturate: int          # rounds arriving together to get one through
    friendly_inbound: int           # live friendly seekers already homing on him
    verdict: str = "REACHABLE"
    reason: str = ""

    @property
    def pd_cover(self) -> int:
        return self.pd_target + self.pd_escorts

    @property
    def best_shot(self) -> Optional[TorpedoShot]:
        """The shot to quote: vs an evading target when it connects."""
        if self.vs_evading and self.vs_evading.hits:
            return self.vs_evading
        if self.vs_coasting and self.vs_coasting.hits:
            return self.vs_coasting
        return None

    @property
    def expected_gj(self) -> float:
        """Ranking key: damage a launch can deliver (0 when it cannot)."""
        if self.verdict in ("UNREACHABLE", "OVERKILL"):
            return 0.0
        shot = self.best_shot
        gj = shot.impact_gj if shot and shot.impact_gj else 0.0
        return gj * (0.5 if self.verdict == "ESCAPABLE" else 1.0)

    def summary(self) -> str:
        """One line for a prompt: verdict first, then the numbers behind it."""
        rate = (f"closing {self.closing_kps:.1f}" if self.closing_kps >= 0
                else f"OPENING {-self.closing_kps:.1f}")
        shot = self.best_shot
        flight = (f"TOF ~{shot.tof_s:.0f} s, ~{shot.impact_kps:.1f} km/s ~{shot.impact_gj:.0f} GJ"
                  if shot and shot.tof_s is not None else "")
        cover = (f"PD over him {self.pd_target}+{self.pd_escorts} -> salvo >= "
                 f"{self.salvo_to_saturate} together")
        inbound = f"friendly rounds inbound {self.friendly_inbound}"
        if self.verdict == "UNREACHABLE":
            return f"UNREACHABLE ({rate} km/s): {self.reason}"
        if self.verdict == "ESCAPABLE":
            return f"ESCAPABLE: {self.reason} | {flight} if he holds course | {inbound}"
        if self.verdict == "TOO CLOSE":
            return f"TOO CLOSE: {self.reason} | {inbound}"
        if self.verdict == "OVERKILL":
            return f"OVERKILL: {self.reason} | {flight}"
        return f"REACHABLE | {flight} | {cover} | {inbound}"


def _shadow_dt(sim: Any) -> float:
    dt = float(getattr(sim, "time_step", 1.0) or 1.0)
    return min(SHADOW_DT_MAX_S, max(SHADOW_DT_MIN_S, dt))


def _clone(ship: Any, ship_id: str, faction: str, fleet_data: Dict[str, Any]):
    from ..simulation import create_ship_from_fleet_data
    return create_ship_from_fleet_data(
        ship_id, ship.ship_type, faction, fleet_data,
        position=Vector3D(ship.position.x, ship.position.y, ship.position.z),
        velocity=Vector3D(ship.velocity.x, ship.velocity.y, ship.velocity.z),
        forward=Vector3D(ship.forward.x, ship.forward.y, ship.forward.z),
    )


def fly_torpedo(
    sim: Any,
    shooter: Any,
    target: Any,
    fleet_data: Dict[str, Any],
    target_evades: bool,
    horizon_s: float = SHADOW_HORIZON_S,
) -> Optional[TorpedoShot]:
    """
    Fly one torpedo from ``shooter`` at ``target`` on a private engine shadow.

    Returns None when the shooter's class mounts no launcher (or the shadow
    cannot be built, e.g. a test double without a ship_type).
    """
    from ..simulation import (CombatSimulation, Maneuver, ManeuverType,
                              SimulationEventType)

    if not fleet_data or not getattr(shooter, "ship_type", None) \
            or not getattr(target, "ship_type", None):
        return None
    try:
        shadow = CombatSimulation(time_step=_shadow_dt(sim), decision_interval=1e9, seed=0)
        sh = _clone(shooter, "shooter", "alpha", fleet_data)
        tg = _clone(target, "target", "beta", fleet_data)
    except Exception:  # noqa: BLE001 - advisory only, never fail a decision
        return None
    if not sh.ready_torpedo_launchers:
        return None
    tg.point_defense = []   # PD is reported as turret cover, not flown
    sh.point_defense = []
    if target_evades:
        tg.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE,
                                       start_time=0.0, duration=1e9, throttle=1.0)
    shadow.add_ship(sh)
    shadow.add_ship(tg)
    if not shadow._handle_torpedo_command(sh, {"target_id": "target"}):
        return None
    fuel_out = False
    while shadow.current_time < horizon_s and shadow.torpedoes:
        for ev in shadow.step():
            if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                return TorpedoShot(hits=True, tof_s=shadow.current_time,
                                   impact_kps=ev.data.get("impact_speed_kps"),
                                   impact_gj=ev.data.get("total_damage_gj"))
            if ev.event_type == SimulationEventType.TORPEDO_FUEL_EXHAUSTED:
                fuel_out = True
        if tg.is_destroyed:
            break
    return TorpedoShot(hits=False, note=("burns out before intercept" if fuel_out
                                         else f"no intercept within {horizon_s:.0f} s"))


def pd_cover(sim: Any, target: Any, pd_range_km: float) -> Tuple[int, int, List[Tuple[str, int]]]:
    """
    Live point-defense turrets covering ``target``: its own, and its allies'
    within ``pd_range_km`` of it (they bear on rounds in its terminal path).
    Returns (own, escorts_total, [(escort name, turrets), ...]).
    """
    def live(ship) -> int:
        return sum(1 for pd in (getattr(ship, "point_defense", None) or [])
                   if getattr(pd, "is_operational", True))

    own = live(target)
    escorts: List[Tuple[str, int]] = []
    friends = sim.get_friendly_ships(target.ship_id) if hasattr(sim, "get_friendly_ships") else []
    for f in friends:
        n = live(f)
        if n and (f.position - target.position).magnitude <= pd_range_km * 1000.0:
            escorts.append((getattr(f, "name", f.ship_id), n))
    return own, sum(n for _, n in escorts), escorts


def rounds_inbound(sim: Any, faction: str, target_id: str) -> List[Any]:
    """Live-seeker torpedoes launched by ``faction`` homing on ``target_id``."""
    out = []
    for tf in getattr(sim, "torpedoes", None) or []:
        if getattr(tf, "is_disabled", False):
            continue
        if getattr(tf.torpedo, "target_id", None) != target_id:
            continue
        src = sim.get_ship(tf.source_ship_id) if hasattr(sim, "get_ship") else None
        if src is not None and src.faction == faction:
            out.append(tf)
    return out


def salvo_to_saturate(turrets: int, turrets_per_round: int) -> int:
    """
    Rounds that must arrive TOGETHER for one to get through un-blinded:
    every ``turrets_per_round`` turrets dwelling on one round blind it.
    """
    per = max(1, int(turrets_per_round))
    return turrets // per + 1


def torpedo_solution(
    sim: Any,
    shooter: Any,
    target: Any,
    fleet_data: Dict[str, Any],
    cache: Optional[Dict[Any, Any]] = None,
) -> Optional[TorpedoSolution]:
    """
    Fire solution for a torpedo from ``shooter`` at ``target`` right now.

    ``cache`` (any dict, e.g. per decision) memoises shadow flights by sim
    time and ship ids - the same solution feeds the target list, the salvo
    check of an order and the launch tool's feasibility answer.
    """
    from .prompts import pd_doctrine_numbers

    key = (round(float(getattr(sim, "current_time", 0.0)), 3), shooter.ship_id, target.ship_id)
    if cache is not None and key in cache:
        return cache[key]

    rel = target.position - shooter.position
    rng_m = rel.magnitude
    closing = (-(rel.normalized()).dot(target.velocity - shooter.velocity) / 1000.0
               if rng_m > 0 else 0.0)
    pd = pd_doctrine_numbers(fleet_data)
    own, escorts, _ = pd_cover(sim, target, pd["range_km"])
    inbound = len(rounds_inbound(sim, shooter.faction, target.ship_id))
    need = salvo_to_saturate(own + escorts, pd["turrets_to_blind_at_floor"])

    evading = fly_torpedo(sim, shooter, target, fleet_data, target_evades=True)
    if evading is None:
        if cache is not None:
            cache[key] = None
        return None
    coasting = None if evading.hits else fly_torpedo(sim, shooter, target, fleet_data,
                                                     target_evades=False)
    sol = TorpedoSolution(
        target_id=target.ship_id, target_name=getattr(target, "name", target.ship_id),
        range_km=rng_m / 1000.0, closing_kps=closing,
        vs_evading=evading, vs_coasting=coasting,
        pd_target=own, pd_escorts=escorts, salvo_to_saturate=need,
        friendly_inbound=inbound,
    )
    shot = sol.best_shot
    if shot is None:
        sol.verdict = "UNREACHABLE"
        sol.reason = (f"the round {coasting.note if coasting else evading.note} "
                      f"even if he holds course - expected ~0 GJ")
    elif not evading.hits:
        sol.verdict = "ESCAPABLE"
        sol.reason = f"if he EVADEs the round {evading.note}"
    elif (shot.impact_gj or 0.0) < LOW_YIELD_GJ:
        sol.verdict = "TOO CLOSE"
        sol.reason = (f"~{shot.impact_gj:.1f} GJ at {shot.impact_kps:.1f} km/s - "
                      f"guidance cannot build speed in {sol.range_km:.0f} km; open range first")
    elif inbound >= need + OVERKILL_MARGIN_ROUNDS:
        sol.verdict = "OVERKILL"
        sol.reason = (f"{inbound} friendly rounds already inbound (a salvo of {need} "
                      f"saturates his {sol.pd_cover} PD turrets) - pick another target")
    if cache is not None:
        cache[key] = sol
    return sol
