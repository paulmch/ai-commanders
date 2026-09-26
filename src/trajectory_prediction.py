"""
Trajectory prediction: where will each ship be at the next checkpoint?

Every recorded sim-trace frame carries, per live ship, a `plan` - the
best-estimate path from the ship's CURRENT actual state to the next decision
checkpoint. The viewer draws it as a track (replays and live mode alike).

How the estimate is made
------------------------
A movement-only shadow of the battle (:class:`PredictionSimulation`) is
propagated forward with the engine's own step size and its own ship-update
code (``CombatSimulation._update_ship``: maneuver guidance, bang-bang
rotation, threat-aware evasion, semi-implicit Euler ``propagate_state``).
Every ship moves, so intercept / padlock / wobble guidance has a moving
target. Nothing else happens in the shadow: no weapons fire, no damage, no
point defence, no thermal or power bookkeeping.

Evasion is continued where it is deterministic:

* EVADE jinks draw their noise from a throwaway RNG keyed on
  (seed, ship, 8 s cycle) - the shadow reproduces them exactly, so the
  predicted path shows the jinks. Rounds already in flight are propagated
  ballistically as threats; rounds fired later cannot be foreseen.
* RUN / PRESENT react to guided torpedoes; the shadow flies copies of the
  torpedoes in flight with the real guidance law (no PD, no hit roll).
* WOBBLE (no dangerous threat) jitters the aim with ``sim.rng.gauss`` every
  tick - genuinely random. The shadow flies the NOMINAL command (zero
  wobble) and estimates the 1-sigma dispersion of the checkpoint position
  from a small Monte Carlo of the one-axis rotation controller chasing the
  jittered aim (:func:`wobble_sigma_m`).
* What the shadow cannot foresee (rounds fired later, torpedo launches, PD
  kills, hits, target deaths) moves reactive evaders by kilometres. An
  empirical per-mode term scaled by time to go (:func:`reactive_sigma_m`,
  calibrated on the showcase battles) covers it. The plan's ``sigma`` is
  both terms in quadrature; its ``mode`` names the evasion mode so the
  viewer can style a low-confidence plan.

Checkpoint frames
-----------------
The frame recorded AT a checkpoint precedes that checkpoint's orders, so its
plan first flies the expiring maneuvers and carries ``"stale": true``. The
battle runner re-predicts it in place once the orders are applied
(:meth:`TraceEnricher.refresh_plans`): replays carry the post-order plan.

Isolation guarantee
-------------------
Prediction must never change the battle. The shadow owns copies of every
piece of state the ship-update path WRITES (kinematic state, rotation state,
maneuver, torpedo flights and torpedoes), strips the subsystems it would
otherwise tick (thermal, power, weapons), never touches ``sim.rng`` (it has a
nominal stand-in that returns the mean), and swallows event logging.
Shared objects that remain (modules, geometry, torpedo specs, and the armor
of non-evading ships) are only read; evading ships get private armor copies
because presentation scoring trials impacts on them.
``tests/test_trajectory_prediction.py`` pins this with a seeded battle that
must produce identical results with prediction on and off.
"""

from __future__ import annotations

import copy
import functools
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .simulation import CombatSimulation, ManeuverType, ShipCombatState
    from .physics import Vector3D, propagate_state
    from .replay_evidence import ExecutionEvidence
    from .torpedo import TorpedoGuidance, GuidanceMode, SAFE_ARMING_DISTANCE_M
except ImportError:  # legacy imports with src/ on sys.path
    from simulation import CombatSimulation, ManeuverType, ShipCombatState
    from physics import Vector3D, propagate_state
    from replay_evidence import ExecutionEvidence
    from torpedo import TorpedoGuidance, GuidanceMode, SAFE_ARMING_DISTANCE_M


# Spacing of the recorded plan samples (s of sim time). The shadow itself
# steps with the simulation's own time step; the path is resampled to this
# grid (plus the checkpoint itself as the final point) to keep recordings
# small. 2 s resolves the 8 s EVADE jink cycle comfortably.
PLAN_SAMPLE_DT_S = 2.0

# 1-sigma of the per-tick aim jitter in WOBBLE mode. Must match the
# rng.gauss(0, radians(5)) draw in CombatSimulation._update_ship.
WOBBLE_AIM_SIGMA_RAD = math.radians(5.0)

# Monte Carlo size for the wobble dispersion estimate, and the fixed seed of
# its private noise table (common random numbers: sigma is a smooth,
# reproducible function of its inputs and never touches the battle RNG).
WOBBLE_MC_PATHS = 64
# Jittered ticks run before each Monte Carlo path to start it in the steady
# wobble regime rather than from rest.
WOBBLE_BURN_IN_TICKS = 12
_WOBBLE_NOISE_SEED = 0x5EED_B0B

# Mirror of _rotate_ship_toward's completion tolerances.
_ANGLE_TOLERANCE = 0.01
_VELOCITY_TOLERANCE = 0.01

# A ghost torpedo is spent once its in-step closest approach to its target
# falls inside the terminal envelope the real simulation resolves at.
_TORPEDO_TERMINAL_ENVELOPE_M = 5_000.0


def next_checkpoint_time(t: float, interval: float) -> float:
    """
    The next decision checkpoint strictly after ``t``.

    Checkpoints fall on multiples of the decision interval. A frame taken
    exactly AT a checkpoint predicts to the following one (first with the
    expiring orders, marked stale, then again once the new orders are
    applied). Same formula as the live /predictions endpoint.
    """
    return (math.floor(t / interval + 1e-9) + 1) * interval


# ---------------------------------------------------------------------------
# Shadow objects
# ---------------------------------------------------------------------------

class _NominalRng:
    """Stand-in for sim.rng inside the shadow: every draw returns its mean."""

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return mu

    normalvariate = gauss

    def random(self) -> float:
        return 0.5

    def uniform(self, a: float, b: float) -> float:
        return 0.5 * (a + b)


class _GhostShip(ShipCombatState):
    """
    Movement-only copy of a ShipCombatState (built without __init__).

    Engine effectiveness is frozen at the source ship's value: modules take
    no damage inside the shadow, and re-walking the module list on every
    shadow tick is the dominant cost otherwise.
    """

    @property
    def engine_effectiveness(self) -> float:  # type: ignore[override]
        return self._ghost_engine_eff


_FAST_COPY_TYPES: Dict[type, type] = {}


def _fast_deepcopy_type(cls: type) -> type:
    """
    Subclass of an armor-section class whose deepcopy is a shallow copy.

    _presentation_score deep-copies the struck armor section to trial an
    impact; sections hold only scalars (the trial reassigns thickness and
    chipping), so a shallow copy is equivalent - and the deepcopy was the
    single largest cost of predicting a PRESENTing ship.
    """
    fast = _FAST_COPY_TYPES.get(cls)
    if fast is None:
        fast = type(cls.__name__, (cls,), {
            "__deepcopy__": lambda self, memo: copy.copy(self),
        })
        _FAST_COPY_TYPES[cls] = fast
    return fast


def _ghost_armor(armor: Any) -> Any:
    """Private armor copy (read by evasion scoring) with cheap section trials."""
    sections = getattr(armor, "sections", None)
    if not sections:
        return armor
    ghost = copy.copy(armor)
    ghost.sections = {}
    for loc, section in sections.items():
        fast = _fast_deepcopy_type(type(section)).__new__(_fast_deepcopy_type(type(section)))
        fast.__dict__.update(section.__dict__)
        ghost.sections[loc] = fast
    return ghost


def _ghost_ship(ship: ShipCombatState) -> _GhostShip:
    ghost = _GhostShip.__new__(_GhostShip)
    ghost.__dict__.update(ship.__dict__)
    ghost._ghost_engine_eff = ship.engine_effectiveness
    # State the ship-update path writes: private copies.
    ghost.kinematic_state = ship.kinematic_state.copy()
    ghost.rotation_state = copy.copy(ship.rotation_state)
    ghost.current_maneuver = copy.copy(ship.current_maneuver)
    if (ship.current_maneuver is not None
            and ship.current_maneuver.maneuver_type == ManeuverType.EVASIVE):
        ghost.armor = _ghost_armor(ship.armor)
    # Subsystems the shadow must not tick.
    ghost.thermal_system = None
    ghost.power_system = None
    ghost.weapons = {}
    ghost.pd_power_draw_gw = 0.0
    return ghost


class _GhostRound:
    """A kinetic round as a pure ballistic threat (duck-types ProjectileInFlight)."""

    __slots__ = ("target_ship_id", "position", "velocity", "kinetic_energy_gj",
                 "mass_kg", "_pd_ablation")

    def __init__(self, flight: Any) -> None:
        proj = flight.projectile
        self.target_ship_id = flight.target_ship_id
        self.position = proj.position
        self.velocity = proj.velocity
        self.kinetic_energy_gj = proj.kinetic_energy_gj
        # Evasion threat energy uses relative speed and post-ablation mass.
        self.mass_kg = getattr(proj, "mass_kg", 50.0)
        self._pd_ablation = getattr(proj, "_pd_ablation", 0.0)

    @property
    def projectile(self) -> "_GhostRound":
        return self


def _ghost_torpedo_flight(flight: Any) -> Any:
    ghost = type(flight).__new__(type(flight))
    ghost.__dict__.update(flight.__dict__)
    ghost.__dict__.pop("_rng", None)  # never roll a real torpedo's dice
    torp = flight.torpedo
    ghost_torp = type(torp).__new__(type(torp))
    ghost_torp.__dict__.update(torp.__dict__)
    ghost.torpedo = ghost_torp
    return ghost


class PredictionSimulation(CombatSimulation):
    """
    Movement-only shadow of a running CombatSimulation.

    Reuses the real ship-update code on ghost copies; see the module
    docstring for what is and is not simulated.
    """

    def __init__(self, source: CombatSimulation) -> None:  # noqa: D401 - no super().__init__
        self.time_step = source.time_step
        self.decision_interval = source.decision_interval
        self.current_time = source.current_time
        self.last_decision_time = source.last_decision_time
        self._initial_seed = source._initial_seed
        self.rng = _NominalRng()
        self.evidence = ExecutionEvidence()
        self.events = []
        self._event_callbacks = []
        self.pending_torpedo_launches = []
        self.ships = {sid: _ghost_ship(s) for sid, s in source.ships.items()}

        # Threats only matter to ships whose maneuver reacts to them.
        evaders = {
            sid for sid, s in self.ships.items()
            if s.current_maneuver is not None
            and s.current_maneuver.maneuver_type == ManeuverType.EVASIVE
            and not s.is_destroyed and not s.is_dying
        }
        self.projectiles = [
            _GhostRound(f) for f in source.projectiles
            if f.target_ship_id in evaders
        ] if evaders else []
        self.torpedoes = [
            _ghost_torpedo_flight(f) for f in source.torpedoes
        ] if evaders else []

    # The shadow logs nothing: events are battle history, not prediction.
    def _log_event(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def advance(self, dt: float) -> None:
        """One shadow tick: ships (in dict order, like step()), then threats."""
        for ship in self.ships.values():
            if ship.is_destroyed:
                continue
            if ship.is_dying:
                # The hulk's torch flicker is random: coast it (it only
                # matters as a target for someone else's guidance).
                ship.kinematic_state = propagate_state(ship.kinematic_state, dt, 0.0)
                continue
            self._update_ship(ship, dt)

        for rnd in self.projectiles:
            rnd.position = rnd.position + rnd.velocity * dt

        if self.torpedoes:
            self.torpedoes = [f for f in self.torpedoes
                              if not self._advance_ghost_torpedo(f, dt)]

        self.current_time += dt

    def _advance_ghost_torpedo(self, flight: Any, dt: float) -> bool:
        """Fly a ghost torpedo one tick. Returns True once it is spent."""
        torp = flight.torpedo
        target = self.ships.get(torp.target_id)
        if flight.is_disabled or target is None or target.is_destroyed:
            torp.position = torp.position + torp.velocity * dt
            return False

        prev_pos = torp.position
        # Same guidance dispatch as CombatSimulation._update_torpedoes, minus
        # retargeting, dv bookkeeping and the hit resolution.
        if torp.guidance_mode == GuidanceMode.COLLISION:
            command = TorpedoGuidance().update_collision_guidance(
                torp, target.position, target.velocity, dt)
            torp.time_since_launch += dt
            if not torp.armed and torp.distance_from_launch() >= SAFE_ARMING_DISTANCE_M:
                torp.armed = True
            if command.throttle > 0 and command.direction.magnitude > 0.01:
                if command.use_rcs:
                    torp.apply_lateral_thrust(command.direction, dt, command.throttle)
                else:
                    torp.apply_thrust(command.direction, dt, command.throttle)
            if command.rcs_direction is not None and command.rcs_throttle > 0:
                torp.apply_lateral_thrust(command.rcs_direction, dt, command.rcs_throttle)
            torp.position = torp.position + torp.velocity * dt
        elif torp.guidance_mode == GuidanceMode.SMART:
            command = TorpedoGuidance().update_smart_guidance(
                torp, target.position, target.velocity,
                target.kinematic_state.max_acceleration_g(), dt)
            torp.time_since_launch += dt
            if not torp.armed and torp.distance_from_launch() >= SAFE_ARMING_DISTANCE_M:
                torp.armed = True
            if command.throttle > 0 and command.direction.magnitude > 0.01:
                torp.apply_thrust(command.direction, dt, command.throttle)
            torp.position = torp.position + torp.velocity * dt
        else:
            torp.update(dt, target.position, target.velocity)

        # Terminal encounter this tick (hit or miss, the real round is
        # resolved here): it stops being a threat.
        target_pos_t0 = target.position - target.velocity * dt
        tca, cpa = self._calculate_time_to_closest_approach(
            prev_pos, torp.velocity, target_pos_t0, target.velocity)
        return tca <= dt and cpa < _TORPEDO_TERMINAL_ENVELOPE_M


# ---------------------------------------------------------------------------
# Wobble dispersion
# ---------------------------------------------------------------------------

def _chase(h: float, s: float, aim: float, dt: float,
           angular_accel: float, max_angular_vel: float) -> Tuple[float, float]:
    """One tick of _rotate_ship_toward in one axis: (heading, rate) -> new."""
    err = aim - h
    angle = abs(err)
    if angle < _ANGLE_TOLERANCE and s < _VELOCITY_TOLERANCE:
        return h, 0.0
    if angle <= s * s / (2.0 * angular_accel) + _ANGLE_TOLERANCE:
        s = max(0.0, s - angular_accel * dt)
    elif s < max_angular_vel:
        s = min(s + angular_accel * dt, max_angular_vel)
    rot = min(s * dt, angle)
    return (h + rot if err > 0 else h - rot), s


@functools.lru_cache(maxsize=1024)
def _wobble_lever_matrix(
    dts: Tuple[float, ...],
    wobbling: Tuple[bool, ...],
    angular_accel: float,
    max_angular_vel: float,
    paths: int,
) -> "np.ndarray":
    """
    Monte Carlo of the wobbling nose, reduced to a lever matrix M[p, k] such
    that path p's lateral offset from the noiseless path at the horizon is
    sum_k M[p, k] * a_k for per-tick thrust accelerations a_k.

    Each path starts from a nose state drawn from the steady wobble regime
    (a burn-in of WOBBLE_BURN_IN_TICKS jittered ticks - a wobbling ship is
    rarely at rest in rotation, and the controller's response to new jitter
    depends on its current rate), then flies the horizon twice from that
    state: jittered and noiseless. Only the difference is dispersion; the
    noiseless path is what the shadow predicts.

    Semi-implicit Euler (v += a*sin(h)*dt; x += v*dt) makes the final
    position linear in the accelerations: x_N = sum_k a_k*sin(h_k)*dt_k*L_k
    with L_k = sum_{j>=k} dt_j. The heading histories depend only on the
    controller and the tick pattern, so the Monte Carlo runs once per (tick
    pattern, rotation authority) and every later call is a dot product.
    """
    n = len(dts)
    rng = random.Random(_WOBBLE_NOISE_SEED)  # private: never the battle RNG
    burn_dt = max(dts) if dts else 1.0
    lever = [0.0] * n
    acc = 0.0
    for k in range(n - 1, -1, -1):
        acc += dts[k]
        lever[k] = dts[k] * acc
    out = np.zeros((paths, n))
    for p in range(paths):
        h = s = 0.0
        for _ in range(WOBBLE_BURN_IN_TICKS):
            h, s = _chase(h, s, rng.gauss(0.0, WOBBLE_AIM_SIGMA_RAD), burn_dt,
                          angular_accel, max_angular_vel)
        hn, sn = h, s          # noiseless twin
        row = out[p]
        for k in range(n):
            aim = rng.gauss(0.0, WOBBLE_AIM_SIGMA_RAD) if wobbling[k] else 0.0
            h, s = _chase(h, s, aim, dts[k], angular_accel, max_angular_vel)
            hn, sn = _chase(hn, sn, 0.0, dts[k], angular_accel, max_angular_vel)
            row[k] = (math.sin(h) - math.sin(hn)) * lever[k]
    out.setflags(write=False)
    return out


def wobble_sigma_m(
    dts: Sequence[float],
    accels: Sequence[float],
    wobbling: Sequence[bool],
    angular_accel: float,
    max_angular_vel: float,
    paths: int = WOBBLE_MC_PATHS,
) -> float:
    """
    1-sigma lateral dispersion (m) at the end of a horizon caused by WOBBLE.

    One-axis model of what the simulation does each tick while wobbling: the
    aim is the nominal direction rotated by N(0, 5 deg) about the ship's up
    axis; the nose chases it with the same bang-bang controller as
    CombatSimulation._rotate_ship_toward (scalar rate magnitude, direction
    re-evaluated every tick, brake when inside the stopping angle); the
    thrust acceleration a_k then pushes sideways by a_k*sin(heading error),
    integrated with the same semi-implicit Euler as propagate_state. On
    non-wobbling ticks the aim is nominal (the error decays back to zero).

    Args:
        dts: tick lengths (s) over the horizon.
        accels: nominal thrust acceleration per tick (m/s^2).
        wobbling: whether the ship was in WOBBLE on each tick.
        angular_accel / max_angular_vel: the ship's rotation authority
            (rad/s^2, rad/s) with engines on.
    """
    if not dts or not any(wobbling) or angular_accel <= 0.0:
        return 0.0
    lever = _wobble_lever_matrix(
        tuple(round(d, 6) for d in dts), tuple(bool(w) for w in wobbling),
        round(angular_accel, 12), round(max_angular_vel, 12), paths)
    offsets = lever @ np.asarray(accels, dtype=float)
    return float(np.sqrt(np.mean(offsets * offsets)))


# ---------------------------------------------------------------------------
# Reactive-evasion dispersion
# ---------------------------------------------------------------------------

# Evasion execution modes (CombatSimulation._calculate_evasion).
_EVASION_MODES: Tuple[str, ...] = ("WOBBLE", "EVADE", "TANK", "BRACE", "RUN", "PRESENT")

# Reactive-evasion dispersion coefficient k per mode (dimensionless): a ship
# whose evasion can react to threats the shadow cannot see (rounds fired
# after the frame, torpedoes launched, PD kills, hits, retargeting) has an
# unknown acceleration of ~k * a_max sustained to the checkpoint, so its
# checkpoint position disperses by k * a_max * tau^2 / 2 (tau = time to go).
#
# Calibrated on the two showcase battles (run_draft_battle.py --auto-draft
# --no-admirals --seed 3 --budget 100 and --seed 11 --budget 200; 13.2k
# plans) so that the total sigma (this in quadrature with the WOBBLE Monte
# Carlo term) puts 90% of actual checkpoint positions within 2 sigma for
# ships in each current mode; held-out seeds 5 and 7 (--budget 150) land
# at 82-99%. The ship's k is the largest over its current mode and every
# mode its shadow flies through before the checkpoint (a wobbling ship
# with a known torpedo inbound already carries RUN's uncertainty). WOBBLE's
# small k covers threats that appear later; TANK is floored to it (TANK
# ships nearly always pass through a higher-k mode).
REACTIVE_SIGMA_K: Dict[str, float] = {
    "PRESENT": 0.215,
    "BRACE": 0.171,
    "EVADE": 0.109,
    "RUN": 0.067,
    "TANK": 0.0092,
    "WOBBLE": 0.0092,
}


def _evasion_mode(ship: Any) -> Optional[str]:
    """The ship's evasion execution mode, or None when not EVASIVE."""
    maneuver = ship.current_maneuver
    if maneuver is None or maneuver.maneuver_type != ManeuverType.EVASIVE:
        return None
    mode = getattr(ship, "execution_mode", None)
    return mode if mode in _EVASION_MODES else None


def reactive_sigma_m(
    dts: Sequence[float],
    modes: Sequence[Optional[str]],
    mode_now: Optional[str],
    a_max: float,
) -> Tuple[float, Optional[str]]:
    """
    Threat-reactive part of the checkpoint dispersion, and the plan's mode.

    Args:
        dts: shadow tick lengths (s) up to the checkpoint.
        modes: the shadow's evasion mode on each tick (None = not evading).
        mode_now: the real ship's evasion mode at the frame (None = not
            evading, or not yet evading under freshly issued orders).
        a_max: the ship's full-thrust acceleration (m/s^2).

    Returns (sigma_m, mode): sigma = k * a_max * tau^2 / 2 with k the
    largest REACTIVE_SIGMA_K over mode_now and ``modes``; ``mode`` is
    mode_now, else the first mode the shadow enters, else None (sigma 0).
    """
    mode = mode_now or next((m for m in modes if m), None)
    if mode is None:
        return 0.0, None
    k = max(REACTIVE_SIGMA_K.get(m, 0.0) for m in {mode, *(m for m in modes if m)})
    tau = float(sum(dts))
    return k * a_max * tau * tau / 2.0, mode


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _resample(times: List[float], positions: List[Vector3D], t0: float,
              sample_dt: float, t_end: float) -> List[List[int]]:
    """Positions at t0, t0+dt, ... (< t_end) and exactly t_end, in int metres."""
    out: List[List[int]] = []
    j = 0
    k = 0
    while True:
        ts = t0 + k * sample_dt
        if ts >= t_end - 1e-6:
            break
        while j + 1 < len(times) and times[j + 1] < ts - 1e-9:
            j += 1
        if j + 1 < len(times) and times[j + 1] - times[j] > 1e-12:
            f = (ts - times[j]) / (times[j + 1] - times[j])
            f = min(1.0, max(0.0, f))
            p = positions[j] + (positions[j + 1] - positions[j]) * f
        else:
            p = positions[j]
        out.append([int(round(p.x)), int(round(p.y)), int(round(p.z))])
        k += 1
    p = positions[-1]
    out.append([int(round(p.x)), int(round(p.y)), int(round(p.z))])
    return out


def predict_plans(
    sim: CombatSimulation,
    t_cp: float,
    sample_dt: float = PLAN_SAMPLE_DT_S,
    ship_ids: Optional[Iterable[str]] = None,
    diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Predicted path to ``t_cp`` for every live (not destroyed, not dying) ship.

    Returns {ship_id: {"t_cp", "dt", "path", "sigma", "fwd", "mode"}}:
      path   int-metre positions at now, now+dt, ... and exactly t_cp (last).
      sigma  1-sigma dispersion (m) of the checkpoint position: WOBBLE
             jitter and reactive-evasion uncertainty combined in quadrature
             (see reactive_sigma_m); 0.0 for a ship that is not evading.
      fwd    the shadow's nose unit vector at t_cp (4 decimals).
      mode   the ship's evasion mode now (lowercase; under fresh EVASIVE
             orders, the first one the shadow enters), None when not evading.

    ``diagnostics``, when given, receives per-ship inputs of the sigma
    model (for calibration scripts). Never mutates ``sim`` and never draws
    from its RNG.
    """
    t0 = sim.current_time
    wanted = set(ship_ids) if ship_ids is not None else None
    shadow = PredictionSimulation(sim)
    tracked = [
        sid for sid, s in shadow.ships.items()
        if not s.is_destroyed and not s.is_dying
        and (wanted is None or sid in wanted)
    ]
    if not tracked:
        return {}

    times = [t0]
    positions = {sid: [shadow.ships[sid].position] for sid in tracked}
    dts: List[float] = []
    accels: Dict[str, List[float]] = {sid: [] for sid in tracked}
    modes: Dict[str, List[Optional[str]]] = {sid: [] for sid in tracked}
    # Mode the real ship is in right now (set by its last update tick).
    mode_now = {sid: _evasion_mode(sim.ships[sid]) for sid in tracked}

    step = sim.time_step if sim.time_step > 0 else 1.0
    while shadow.current_time < t_cp - 1e-9:
        dt = min(step, t_cp - shadow.current_time)
        shadow.advance(dt)
        dts.append(dt)
        times.append(shadow.current_time)
        for sid in tracked:
            ghost = shadow.ships[sid]
            positions[sid].append(ghost.position)
            ks = ghost.kinematic_state
            throttle = getattr(ghost, "applied_throttle", 0.0) if not ghost.is_dying else 0.0
            accels[sid].append(ks.thrust_n * throttle / ks.mass_kg if ks.mass_kg > 0 else 0.0)
            modes[sid].append(_evasion_mode(ghost))

    plans: Dict[str, Dict[str, Any]] = {}
    for sid in tracked:
        wobble = [m == "WOBBLE" for m in modes[sid]]
        sigma_w = 0.0
        if any(wobble):
            alpha, omega_max = sim._get_rotation_params(sim.ships[sid], engines_on=True)
            sigma_w = wobble_sigma_m(dts, accels[sid], wobble, alpha, omega_max)
        a_max = sim.ships[sid].kinematic_state.max_acceleration_ms2()
        sigma_r, mode = reactive_sigma_m(dts, modes[sid], mode_now[sid], a_max)
        if diagnostics is not None:
            diagnostics[sid] = {
                "dts": list(dts), "modes": list(modes[sid]), "mode_now": mode_now[sid],
                "a_max": a_max, "sigma_wobble": sigma_w, "sigma_reactive": sigma_r,
            }
        fwd = shadow.ships[sid].kinematic_state.forward
        plans[sid] = {
            "t_cp": round(t_cp, 3),
            "dt": sample_dt,
            "path": _resample(times, positions[sid], t0, sample_dt, t_cp),
            "sigma": round(math.hypot(sigma_w, sigma_r), 1),
            "fwd": [round(fwd.x, 4), round(fwd.y, 4), round(fwd.z, 4)],
            "mode": mode.lower() if mode else None,
        }
    return plans


# ---------------------------------------------------------------------------
# sim_trace enrichment
# ---------------------------------------------------------------------------

# Below this |alpha| (rad/s^2) no actuator is considered to be torquing. The
# weakest real authority (dreadnought RCS) is ~3e-5 rad/s^2.
_TORQUE_EPS = 1e-6


def _vec(v: Vector3D, nd: int) -> List[float]:
    return [round(v.x, nd), round(v.y, nd), round(v.z, nd)]


def _rotation_rate(f0: Vector3D, f1: Vector3D, dt: float) -> Vector3D:
    """World-frame angular velocity that carries unit vector f0 onto f1 in dt."""
    axis = f0.cross(f1)
    s = axis.magnitude
    if s < 1e-12 or dt <= 0.0:
        return Vector3D(0.0, 0.0, 0.0)
    angle = math.atan2(s, f0.dot(f1))
    return axis * (angle / (s * dt))


class TraceEnricher:
    """
    Adds the viewer-facing kinematics and the predicted plan to sim-trace
    ship entries. One instance per battle: angular velocity and acceleration
    are finite differences against the previous recorded frame.

    Fields written into each ship entry (all optional for readers):
      acc    [ax, ay, az] m/s^2 world - thrust acceleration applied over the
             tick that ended at this frame (forward * thrust * F / m).
      omega  [wx, wy, wz] rad/s world - rotation of the nose since the
             previous frame (dying hulks: the exact tumble vector).
      alpha  [ax, ay, az] rad/s^2 world - change of omega since the previous
             frame (the torque direction); zeros on a ship's first frame.
      turn   "tv" | "rcs" | None - actuator producing that torque: thrust
             vectoring while the torch burns, RCS with it cold; None when
             not torquing (and for dying hulks - attitude control is dead).
      evade  None | "wobble" | "evade" | "tank" | "brace" | "run" | "present"
             - active mode of an EVASIVE maneuver.
      tgt    ship id the maneuver steers relative to (INTERCEPT / PADLOCK
             target; for a wobbling EVASIVE without one, the nearest enemy
             it homes on), else None.
      plan   see predict_plans(); omitted for dying ships and when
             prediction is disabled. A frame taken AT a checkpoint is
             recorded before that checkpoint's orders exist: its plan
             carries "stale": true (it flies out the expiring orders)
             until refresh_plans() re-predicts it with the new ones.
    Destroyed ships get none of these fields.
    """

    def __init__(self, decision_interval_s: float, predict: bool = True,
                 sample_dt: float = PLAN_SAMPLE_DT_S) -> None:
        self.interval = decision_interval_s
        self.predict = predict
        self.sample_dt = sample_dt
        # ship_id -> (t, forward, omega) of the last enriched frame
        self._prev: Dict[str, Tuple[float, Vector3D, Vector3D]] = {}

    def at_checkpoint(self, t: float) -> bool:
        """Whether ``t`` is a decision checkpoint (orders are issued there)."""
        if t <= 1e-9 or self.interval <= 0:
            return False
        return abs(t / self.interval - round(t / self.interval)) < 1e-6

    def enrich(self, sim: CombatSimulation, ships: Dict[str, Dict[str, Any]]) -> None:
        t = sim.current_time
        plans: Dict[str, Dict[str, Any]] = {}
        if self.predict:
            plans = predict_plans(sim, next_checkpoint_time(t, self.interval),
                                  sample_dt=self.sample_dt)
            if self.at_checkpoint(t):
                for plan in plans.values():
                    plan["stale"] = True

        for sid, entry in ships.items():
            ship = sim.ships.get(sid)
            if ship is None or ship.is_destroyed:
                self._prev.pop(sid, None)
                continue
            dying = bool(getattr(ship, "is_dying", False))
            ks = ship.kinematic_state
            fwd = ks.forward

            thrust = float(entry.get("thrust", 0.0) or 0.0)
            acc_mag = ks.thrust_n * thrust / ks.mass_kg if ks.mass_kg > 0 else 0.0
            acc = fwd * acc_mag

            prev = self._prev.get(sid)
            if dying and ship.dying_tumble_axis is not None:
                omega = ship.dying_tumble_axis.normalized() * ship.dying_tumble_rate_rad_s
            elif prev is not None and t - prev[0] > 1e-9:
                omega = _rotation_rate(prev[1], fwd, t - prev[0])
            elif prev is not None:
                omega = prev[2]
            else:
                omega = Vector3D(0.0, 0.0, 0.0)
            if prev is not None and t - prev[0] > 1e-9:
                alpha = (omega - prev[2]) * (1.0 / (t - prev[0]))
            else:
                alpha = Vector3D(0.0, 0.0, 0.0)
            if prev is None or t - prev[0] > 1e-9:
                self._prev[sid] = (t, Vector3D(fwd.x, fwd.y, fwd.z), omega)

            turn = None
            if not dying and alpha.magnitude > _TORQUE_EPS:
                turn = "tv" if thrust > 0.0 else "rcs"

            maneuver = None if dying else ship.current_maneuver
            evade = None
            tgt = None
            if maneuver is not None:
                tgt = maneuver.target_id
                if maneuver.maneuver_type == ManeuverType.EVASIVE:
                    mode = getattr(ship, "execution_mode", None)
                    evade = mode.lower() if isinstance(mode, str) else None
                    if tgt is None and mode == "WOBBLE":
                        enemies = sim.get_enemy_ships(sid)
                        if enemies:
                            tgt = min(enemies, key=lambda e: ship.distance_to(e)).ship_id

            entry["acc"] = _vec(acc, 3)
            entry["omega"] = _vec(omega, 5)
            entry["alpha"] = _vec(alpha, 5)
            entry["turn"] = turn
            entry["evade"] = evade
            entry["tgt"] = tgt
            plan = plans.get(sid)
            if plan is not None:
                entry["plan"] = plan

    def refresh_plans(self, sim: CombatSimulation,
                      ships: Dict[str, Dict[str, Any]]) -> int:
        """
        Re-predict the plans of a frame already recorded at ``sim``'s
        current time, now that the checkpoint's orders are applied.

        ``ships`` is the recorded frame's ship dict; every entry that carries
        a plan gets a fresh one (without "stale"). Kinematic fields and the
        finite-difference history are left alone - nothing moved. Returns
        the number of plans replaced.
        """
        if not self.predict:
            return 0
        wanted = [sid for sid, entry in ships.items() if "plan" in entry]
        if not wanted:
            return 0
        plans = predict_plans(sim, next_checkpoint_time(sim.current_time, self.interval),
                              sample_dt=self.sample_dt, ship_ids=wanted)
        for sid, plan in plans.items():
            ships[sid]["plan"] = plan
        return len(plans)
