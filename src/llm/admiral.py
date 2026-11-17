"""
LLM Admiral - Fleet commander that issues orders to ship captains.

The Admiral:
- Sees a compact, ranked turn state built from the simulation each checkpoint
  (urgent items, captain reports, what changed, geometry per enemy group,
  threat table, ordnance and torpedo economy, fleet roster - prompts.py),
  with the battle-static doctrine and capabilities in a separate cacheable
  system prompt
- Has full visibility into friendly ship status and capabilities
- Issues text orders to captains
- Can negotiate with enemy Admiral (immediate)
- Controls draw proposals for the fleet
"""

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

from .client import LLMCallError
from .fleet_config import AdmiralConfig
from .admiral_tools import get_admiral_tools
from . import fleet_intel as intel

if TYPE_CHECKING:
    from .client import CaptainClient
    from .captain import LLMCaptain



def _crew_state(ship):
    """drive_plume.crew_state (works for both import styles)."""
    try:
        from ..drive_plume import crew_state
    except ImportError:
        from drive_plume import crew_state
    return crew_state(ship)


# Arcs lines the admiral does not need per ship: the class blind cones are in
# the cached FORCES block (a per-ship "guns within 15 deg of the stern" was
# read as "guns are stern-arc only" - audit A7), and the roll is automatic.
_ADMIRAL_ARCS_SKIP = ("ARCS blind cones", "ARCS roll:")


def _arcs_status_lines(simulation, ship):
    """Weapon-arc state lines (src/llm/arcs_status.py): what bears, what unmasks."""
    from .arcs_status import arcs_status_lines
    try:
        lines = arcs_status_lines(simulation, ship)
    except Exception:  # noqa: BLE001 - advisory only
        return []
    return [line for line in lines if not line.startswith(_ADMIRAL_ARCS_SKIP)]


def _plume_status_lines(simulation, ship):
    """drive_plume.plume_status_lines (works for both import styles)."""
    try:
        from ..drive_plume import plume_status_lines
    except ImportError:
        from drive_plume import plume_status_lines
    return plume_status_lines(simulation, ship)


def _damaged_fitting_lines(ship):
    """battle_damage.damaged_fitting_lines (works for both import styles)."""
    try:
        from ..battle_damage import damaged_fitting_lines
    except ImportError:  # imported as a top-level module
        from battle_damage import damaged_fitting_lines
    return damaged_fitting_lines(ship)

@dataclass
class AdmiralOrder:
    """A single order from Admiral to a specific ship."""
    target_ship_id: str  # Ship ID receiving this order
    target_ship_name: str  # Ship name for display
    order_text: str  # Free-form text instruction
    priority: str = "NORMAL"  # CRITICAL, HIGH, NORMAL, LOW
    suggested_target: Optional[str] = None  # Enemy ship name to focus on
    torpedo_salvo: Optional[int] = None  # Coordinated launch: rounds this checkpoint
    torpedo_target: Optional[str] = None  # Enemy to concentrate the salvo on
    # Structured HEADING order (world frame), applied directly as the ship's
    # maneuver by the runner. formation=True: part of a fleet-wide heading,
    # throttle is a fraction of the slowest hull's acceleration.
    heading: Optional[Dict[str, float]] = None
    throttle: Optional[float] = None
    formation: bool = False
    applied_throttle: Optional[float] = None  # per-hull throttle the runner applied
    # Structured maneuver order (issue_order maneuver / toward_point /
    # toward_ship / evade; maneuver_orders.admiral_maneuver_spec), applied by
    # the runner like a heading: {"maneuver_type", "target" | "target_hint",
    # "standoff_km", "arrival_closing_kps", "range_km", "toward_point",
    # "toward_ship", "evade", "throttle"}. None for prose-only and plain
    # heading orders.
    maneuver: Optional[Dict[str, Any]] = None
    # Set by the runner when it applies `maneuver`: what was flown (with the
    # target it resolved to), or why it could not be (the captain is told it
    # was NOT applied and must choose a maneuver).
    maneuver_applied: Optional[str] = None
    maneuver_error: Optional[str] = None
    # The fleet ("all ships") order this ship's own heading / maneuver
    # replaced, e.g. "HEADING (+1.00,+0.02,+0.00) @50%" - shown to the
    # captain so a per-ship MAINTAIN under a fleet burn reads as what it is.
    replaces_fleet_order: Optional[str] = None
    # Enemy ships the order prose names besides its structured target(s)
    # (fleet_intel.order_text_target_note), shown to the captain as such.
    text_also_names: List[str] = field(default_factory=list)
    # The directive's salvo allocation this order carries ("directive") or
    # replaced ("order"); None without one.
    salvo_source: Optional[str] = None


@dataclass
class AdmiralDecision:
    """Complete decision package from an Admiral."""
    timestamp: float
    fleet_orders: List[AdmiralOrder] = field(default_factory=list)
    fleet_directive: str = ""  # Overall strategic intent
    proposed_draw: bool = False
    accepted_draw: bool = False
    rejected_draw: bool = False
    message_to_enemy_admiral: Optional[str] = None
    reasoning: str = ""  # Admiral's tactical reasoning (for logs)
    call_failed: bool = False  # True if the LLM call errored (not a real decision)
    battle_plan_update: Optional[str] = None  # New standing plan set this checkpoint
    # Fleet-wide formation heading (issue_order to "all ships" on the
    # directive call); per-ship orders without their own heading inherit it.
    fleet_heading: Optional[AdmiralOrder] = None
    # Why this checkpoint has no fleet directive ("truncated at 8192 tokens",
    # "provider error", "no set_fleet_directive call", "call error: ...").
    # None when the directive arrived. Recorded as admiral_directive_failed.
    directive_failure: Optional[str] = None
    # The first attempt failed this way but the retry delivered a directive.
    directive_recovered: Optional[str] = None
    # set_fleet_directive salvo_plan: ship id -> (rounds, target ref). The
    # per-ship order calls run in parallel and cannot see each other, so the
    # fleet's salvo is allocated here, once, and each order call is shown it.
    salvo_plan: Dict[str, Tuple[int, Optional[str]]] = field(default_factory=dict)


def _heading_arg(value: Any) -> Optional[Dict[str, float]]:
    """A valid {x, y, z} heading from tool arguments, else None."""
    if not isinstance(value, dict):
        return None
    try:
        h = {k: float(value.get(k, 0.0)) for k in ("x", "y", "z")}
    except (TypeError, ValueError):
        return None
    if (h["x"] ** 2 + h["y"] ** 2 + h["z"] ** 2) ** 0.5 < 1e-9:
        return None
    return h


def _throttle_arg(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def _plain_heading_spec(spec: Dict[str, Any]) -> bool:
    """A HEADING spec that flies the given heading (evade only) rather than
    toward a point / ship - only then is the heading argument the course."""
    return (spec.get("maneuver_type") == "HEADING"
            and not spec.get("toward_point") and not spec.get("toward_ship"))


def _fleet_heading_order(args: Dict[str, Any]) -> Optional["AdmiralOrder"]:
    """issue_order(ship_name="all ships", heading=... | maneuver=... |
    toward_point/toward_ship=...) -> the fleet-wide order."""
    from .range_control import is_fleet_alias
    from .maneuver_orders import admiral_maneuver_spec, spec_throttle
    if not is_fleet_alias(args.get("ship_name")):
        return None
    heading = _heading_arg(args.get("heading"))
    spec = admiral_maneuver_spec(args)
    if heading is None and spec is None:
        return None
    return AdmiralOrder(
        target_ship_id="all ships",
        target_ship_name="all ships",
        order_text=args.get("order_text", ""),
        priority=args.get("priority", "NORMAL"),
        heading=heading if spec is None or _plain_heading_spec(spec) else None,
        throttle=spec_throttle(spec, args),
        formation=True,
        maneuver=spec,
    )


def _attach_heading(order: "AdmiralOrder", args: Dict[str, Any],
                    fleet_heading: Optional["AdmiralOrder"]) -> None:
    """Own heading / structured maneuver from the order's arguments, else the
    fleet order in force. An evade-only order (OVERLAY spec) rides on the
    fleet order; with none in force the runner puts the overlay on the
    ship's current maneuver. An order of the ship's own that replaces a
    fleet order in force says so (replaces_fleet_order, for the captain)."""
    from .maneuver_orders import (
        OVERLAY, admiral_maneuver_spec, fleet_order_text, spec_throttle)
    heading = _heading_arg(args.get("heading"))
    spec = admiral_maneuver_spec(args)
    fleet_flies = fleet_heading is not None and (
        fleet_heading.heading is not None or getattr(fleet_heading, "maneuver", None))
    if spec is not None and spec.get("maneuver_type") == OVERLAY and fleet_flies:
        # A per-ship evade on top of the fleet order in force: that ship
        # flies the fleet order with the evasion overlay.
        fleet_spec = dict(getattr(fleet_heading, "maneuver", None)
                          or {"maneuver_type": "HEADING",
                              "throttle": fleet_heading.throttle})
        if fleet_spec.get("maneuver_type") != "EVADE":
            fleet_spec["evade"] = spec["evade"]
            if fleet_heading.heading is not None:
                order.heading = dict(fleet_heading.heading)
            order.maneuver = fleet_spec
            order.throttle = fleet_heading.throttle
            order.formation = True
            return
        spec = None     # the fleet already EVADEs: inherit it below
    if spec is not None:
        order.maneuver = spec
        order.throttle = spec_throttle(spec, args)
        order.formation = False
        if heading is not None and _plain_heading_spec(spec):
            order.heading = heading
    elif heading is not None:
        order.heading = heading
        order.throttle = _throttle_arg(args.get("throttle", 1.0))
        order.formation = False
    elif fleet_flies:
        if fleet_heading.heading is not None:
            order.heading = dict(fleet_heading.heading)
        if getattr(fleet_heading, "maneuver", None):
            order.maneuver = dict(fleet_heading.maneuver)
        order.throttle = fleet_heading.throttle
        order.formation = True
        return
    if (fleet_flies and (order.maneuver or order.heading)
            and _motion_key(order) != _motion_key(fleet_heading)):
        order.replaces_fleet_order = fleet_order_text(fleet_heading)


def _motion_key(order: "AdmiralOrder") -> Tuple[Any, ...]:
    """What an order flies (maneuver, target, heading, point/ship) - evade
    and throttle aside: a per-ship PADLOCK on the fleet's PADLOCK target
    replaces nothing worth telling the captain."""
    spec = getattr(order, "maneuver", None) or {}
    heading = getattr(order, "heading", None)
    target = str(spec.get("target") or spec.get("target_hint") or "")
    return (spec.get("maneuver_type") or "HEADING",
            target.split("[", 1)[0].strip().lower(),    # "OCS Lancer-2 [beta_2]"
            tuple(round(float(heading.get(k, 0.0)), 2) for k in "xyz")
            if isinstance(heading, dict) else None,
            repr(spec.get("toward_point")), str(spec.get("toward_ship") or "").lower())


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True if ``fn`` takes keyword ``name`` (test stubs often do not)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


def _has_directive(calls: List[Any]) -> bool:
    return any(getattr(c, "name", None) == "set_fleet_directive"
               and str((getattr(c, "arguments", None) or {}).get("directive") or "").strip()
               for c in calls)


def _failure_text(meta: Dict[str, Any]) -> str:
    """Why a directive call came back without a directive, from its call meta."""
    finish = meta.get("finish_reason")
    if finish == "length":
        reasoning = meta.get("reasoning_tokens")
        spent = f", {reasoning} of them reasoning" if reasoning else ""
        return f"truncated at {meta.get('max_tokens') or '?'} tokens{spent}"
    if finish == "error":
        return "provider error (finish_reason=error)"
    if finish == "content_filter":
        return "blocked by the provider content filter"
    if finish == "refusal":
        category = (meta.get("stop_details") or {}).get("category")
        return "refused by the model (stop_reason=refusal" + (
            f", category {category})" if category else ")")
    return "no set_fleet_directive call"


def _with_retry_nudge(messages: List[Dict[str, Any]], failure: str) -> List[Dict[str, Any]]:
    """Same request (cached system prefix intact) with a short-answer nudge."""
    nudge = (f"\n\nRETRY: your previous answer for this checkpoint produced no fleet directive "
             f"({failure}). Call set_fleet_directive NOW, at most 120 words, no long analysis. "
             f"Add set_battle_plan / an \"all ships\" heading only if they changed.")
    out = [dict(m) for m in messages]
    last = out[-1]
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = list(content) + [{"type": "text", "text": nudge.strip()}]
    else:
        last["content"] = f"{content or ''}{nudge}"
    return out


def _range_change(prev15: Optional["AdmiralSnapshot"], own_ids: List[str],
                  enemy_ids: List[str], range_now_km: float) -> Optional[float]:
    """Change of a group's range from our fleet centre over the pre-snapshot gap."""
    if prev15 is None:
        return None
    own = [r.position_km for r in prev15.friendly_ships if r.ship_id in own_ids]
    them = [r.position_km for r in prev15.enemy_ships if r.ship_id in enemy_ids]
    if not own or not them:
        return None

    def mean(points):
        return [sum(p[k] for p in points) / len(points) for k in ("x", "y", "z")]

    a, b = mean(own), mean(them)
    then = sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
    return range_now_km - then


@dataclass
class FriendlyShipSnapshot:
    """Full status snapshot for a friendly ship."""
    ship_id: str
    ship_name: str
    ship_type: str
    captain_name: str

    # Position/velocity (in km and km/s)
    position_km: Dict[str, float]
    velocity_kps: float
    velocity_vector: Dict[str, float]

    # Combat status
    hull_integrity: float
    delta_v_remaining: float
    heat_percent: float

    # Ship capabilities (from fleet data)
    max_acceleration_g: float
    max_delta_v: float
    weapons_summary: str  # e.g., "1x Spinal (4.3 GJ), 2x Coilgun (0.7 GJ)"

    # Weapon status
    weapons_ready: List[str]
    weapons_cooling: List[str]
    weapons_destroyed: List[str]  # Destroyed weapon systems

    # Current orders/state
    current_maneuver: str
    current_target: Optional[str]
    radiators_extended: bool

    # Targeting info
    targeted_by: List[str]  # Enemy ship names targeting this ship

    # Torpedo magazine and point-defense state (defaults keep older
    # constructor call sites working). These are the inputs to the admiral's
    # salvo sizing and formation decisions.
    torpedoes_remaining: int = 0
    pd_turrets_total: int = 0
    pd_turrets_operational: int = 0
    # ThermalSystem.outlook(): net heat rate, time to 100% / meltdown
    # retracted vs extended, radiator mode and health (None if no thermal).
    thermal: Optional[Dict[str, Any]] = None
    # Torpedo tubes, their class magazine, and launches still possible
    # before the next checkpoint (simulation.torpedo_launch_capacity).
    launchers: int = 0
    magazine_capacity: int = 0
    launches_available: int = 0
    gun_range_km: float = 0.0
    # Damaged-but-working fittings, one short line each
    # (battle_damage.damaged_fitting_lines).
    fittings_damaged: List[str] = field(default_factory=list)
    # Drive plume wash warnings (drive_plume.plume_status_lines).
    plume_wash: List[str] = field(default_factory=list)
    # Crew radiation state (drive_plume.crew_state) and dose; an
    # incapacitated crew cannot execute orders.
    crew_state: str = "fit"
    crew_dose_sv: float = 0.0
    # Engine output fraction (battle damage): 0 = drive dead, cannot burn.
    drive_fraction: float = 1.0


@dataclass
class EnemyShipSnapshot:
    """Observable info about enemy ships."""
    ship_id: str
    ship_name: str
    ship_type: str

    # Observable position/velocity
    position_km: Dict[str, float]
    velocity_kps: float
    velocity_vector: Dict[str, float]

    # Range and closure to the NEAREST friendly hull (not the fleet: see
    # range_km/closing_kps). Kept for older consumers; rendered as "nearest".
    distance_from_closest_friendly_km: float
    closing_rate_kps: float  # Positive = closing

    # Fleet-level view (ADM-04/07): range and closure from our fleet centre,
    # which friendly hull is nearest, observed damage and armament, what it
    # has done to us, and whether a torpedo launched now can catch it.
    range_km: Optional[float] = None
    closing_kps: Optional[float] = None
    nearest_own_id: Optional[str] = None
    hull_integrity: Optional[float] = None
    accel_g: float = 0.0
    pd_operational: int = 0
    pd_total: int = 0
    # Observed from outside, like hull and PD: "fit", "impaired" (it took a
    # heavy plume wash) or "incapacitated" (it no longer maneuvers or fires).
    crew_state: str = "fit"
    armament: str = ""
    launchers: int = 0
    magazine: int = 0
    torpedoes_launched: int = 0
    targeting_id: Optional[str] = None
    dealt_hits: int = 0
    dealt_gj: float = 0.0
    solution: Any = None          # fleet_intel.TorpedoSolution from our launchers
    our_rounds_inbound: int = 0   # our guided rounds homing on it now
    group: str = ""
    gun_range_km: float = 0.0
    # The face it shows our fleet centre (fleet_intel.observed_facing):
    # NOSE / SIDE / TAIL, that facing's armour, burning / turning.
    facing: Optional[Dict[str, Any]] = None


@dataclass
class ProjectileSnapshot:
    """Info about projectiles in flight."""
    source_ship: str
    target_ship: str
    weapon_type: str
    distance_km: float
    eta_seconds: float
    damage_gj: float
    # Side and state (ADM-01): "ours"/"theirs"; guided | blinded | dry |
    # orphaned | ballistic. threat_ship: the hull a guided round homes on or
    # a ballistic one will actually strike (engine closest-approach test),
    # None for a spent round on nobody's line.
    side: str = ""
    state: str = ""
    threat_ship: Optional[str] = None


def _projectile_weapon_label(proj_flight: Any) -> str:
    """
    Name the weapon that actually fired an inbound slug.

    Every round in flight carries the Weapon that launched it
    (ProjectileInFlight.weapon, set by CombatSimulation._launch_projectile), so
    the admiral can be told "Heavy Siege Coiler Mk3" rather than the hardcoded
    "Coilgun" this used to report for every projectile regardless of origin.
    This is prompt surface - an LLM reads it to decide whether an inbound is
    worth maneuvering away from - so a 7.25 GJ spinal round must not be
    reported as a 0.72 GJ turret round.

    Falls back to the projectile's own weapon_type string, then to a generic
    label, for rounds injected directly by tests/tools without a Weapon.
    """
    weapon = getattr(proj_flight, "weapon", None)
    if weapon is not None:
        name = getattr(weapon, "name", None)
        if name:
            return str(name)
        wtype = getattr(weapon, "weapon_type", None)
        if wtype:
            return str(wtype)
    wtype = getattr(getattr(proj_flight, "projectile", None), "weapon_type", None)
    if wtype:
        return str(wtype)
    return "Kinetic round"


@dataclass
class AdmiralSnapshot:
    """
    Complete snapshot for Admiral decision-making.

    Contains TWO temporal views: T-15s and T=0 for trajectory analysis.
    """
    timestamp: float

    # Friendly fleet (full info)
    friendly_ships: List[FriendlyShipSnapshot]

    # Enemy fleet (observable only)
    enemy_ships: List[EnemyShipSnapshot]

    # Projectiles in flight
    projectiles: List[ProjectileSnapshot]

    # Fleet capabilities summary
    fleet_summary: str  # e.g., "2 ships: 1 Destroyer, 1 Frigate"

    # Range control (src/llm/range_control.py): fleet centre vs enemy groups,
    # precomputed headings, formation throttles. Filled for the T=0 snapshot.
    range_control: str = ""

    # Live turn-state inputs (filled by LLMAdmiral._build_snapshot with a
    # full=True build; empty defaults keep hand-built snapshots rendering).
    faction: str = ""
    checkpoint: int = 0
    decision_interval_s: float = 30.0
    labels: Dict[str, str] = field(default_factory=dict)       # id -> "NAME [id]"
    ship_types: Dict[str, str] = field(default_factory=dict)   # id -> class
    range_groups: List[Any] = field(default_factory=list)      # GroupRangeInfo, fleet centre
    group_range_change_km: Dict[str, float] = field(default_factory=dict)  # vs T-15 s
    group_motion: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # battle frame, m/s
    group_burn_g: Dict[str, float] = field(default_factory=dict)  # their thrust toward us (- away)
    groups_opened: List[str] = field(default_factory=list)     # closing -> opening since last CP
    own_centre: Dict[str, Any] = field(default_factory=dict)
    headings: str = ""                                         # option lines + FORMATION
    rounds: List[Any] = field(default_factory=list)            # fleet_intel.RoundInfo
    digest: Any = None                                         # fleet_intel.CheckpointDigest
    hull_before: Dict[str, float] = field(default_factory=dict)
    hull_now: Dict[str, float] = field(default_factory=dict)
    torpedo_magazine: Tuple[int, int] = (0, 0)                 # (remaining, battle-start)
    launches_available: int = 0
    reload_s: float = 0.0
    enemy_torpedoes_seen: int = 0
    engine_notes: List[str] = field(default_factory=list)
    kill_rounds: Dict[str, int] = field(default_factory=dict)
    # Battle clock (fleet_intel.battle_clock) and, behind AdmiralConfig.show_score,
    # the time-limit verdict line (fleet_intel.score_line).
    clock: Optional[Dict[str, Any]] = None
    score_line: str = ""
    # Draw exchange lines for this turn (our open offer, its answer).
    draw_lines: List[str] = field(default_factory=list)
    # NOW-block text per group label for a merge alarm that repeats unchanged
    # (LLMAdmiral._damp_merge_alarms); groups not in it show info.warning.
    merge_alarms: Dict[str, str] = field(default_factory=dict)
    # Per own ship under live torpedo attack: fleet_intel.torpedo_defence.
    torpedo_defence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Ships that have not flown their ordered maneuver for 2+ checkpoints
    # running, with what they flew and the engine's reason (NOW block).
    noncompliance: List[str] = field(default_factory=list)


class LLMAdmiral:
    """
    LLM-powered fleet Admiral that issues orders to captains.
    """

    def __init__(
        self,
        config: AdmiralConfig,
        faction: str,
        client: 'CaptainClient',
        fleet_data: Dict[str, Any],
    ):
        """
        Initialize Admiral.

        Args:
            config: Admiral configuration
            faction: "alpha" or "beta"
            client: LLM client for API calls
            fleet_data: Ship specifications from fleet_ships.json
        """
        self.config = config
        self.faction = faction
        self.client = client
        self.fleet_data = fleet_data
        self.tools = get_admiral_tools()

        # State
        self.decision_count = 0
        # CONTRACT (battle_runner, captains, MCP): last_directive holds THIS
        # checkpoint's directive only - "" when the directive call failed, never
        # a previous checkpoint's text dressed up as current (ADM-02). Set
        # together with last_directive_t (the sim time it was issued for).
        self.last_directive: Optional[str] = None
        self.last_directive_t: Optional[float] = None
        # The last directive that actually arrived and when; shown to the
        # admiral's own per-ship calls (labelled stale) after a failure.
        self._good_directive: Optional[Tuple[float, str]] = None
        # Why the previous checkpoint's directive call failed, for the NOW block.
        self._last_failure: Optional[Tuple[float, str]] = None
        self.order_history: List[AdmiralDecision] = []
        self.has_proposed_draw = False
        self.has_accepted_draw = False
        # Draw exchange (settle_draw_offers): when our open offer was made, and
        # what we must be told at the next checkpoint (a rejection, an accept
        # with nothing to accept).
        self.draw_offer_t: Optional[float] = None
        self.draw_notices: List[str] = []
        # This checkpoint's decision (settle_draw_offers reads both sides').
        self.last_decision: Optional[AdmiralDecision] = None

        # When the battle ends (fleet_intel.battle_clock), set by the runner;
        # None (unlimited mode, hand-built admirals) shows no clock.
        self.battle_clock: Optional[Dict[str, Any]] = None

        # Standing battle plan: the admiral's self-authored memory. Everything
        # else the admiral sees is rebuilt from the simulation each checkpoint;
        # this is the one piece of text that carries the admiral's own INTENT
        # forward, echoed back verbatim every checkpoint until amended.
        self.standing_plan: Optional[str] = None
        self.standing_plan_time: float = 0.0

        # Issue the per-ship orders of Phase 2 concurrently. Each order is an
        # independent LLM call built from the same frozen snapshot, so a fleet
        # of N ships costs one order's wall-clock instead of N. The runner may
        # disable this (BattleConfig.parallel_llm=False) for strictly ordered
        # output.
        self.parallel_ship_orders: bool = True

        # Store T-15s snapshot for comparison
        self._snapshot_t_minus_15: Optional[AdmiralSnapshot] = None

        # Pending message to enemy Admiral
        self._pending_enemy_message: Optional[str] = None

        # Received messages from enemy Admiral
        self._received_enemy_messages: List[str] = []

        # Captain reports filed at the previous checkpoint (report_to_admiral,
        # implicit ACKs, ADMIRAL-addressed fleet signals and discussion Q&A),
        # each with an engine compliance line - captain_report dicts set by
        # the runner before decide(). The admiral's only view of what its
        # captains said and actually did.
        self.captain_reports: List[Dict[str, Any]] = []

        # discuss_with_admiral exchanges, oldest first: {t, checkpoint,
        # ship_name, question, answer}. Fed back into later replies to the
        # same captain; this checkpoint's Q&A also rides on that ship's
        # captain report at the next checkpoint.
        self.discussion_log: List[Dict[str, Any]] = []

        # Ship name to ID mapping (set during setup)
        self._ship_name_to_id: Dict[str, str] = {}

        # Battle-static view: every ship class seen on each side (union, in
        # first-seen order). It feeds the cached system prompt, so it only
        # grows - a dead class keeps its line and the prefix stays byte-stable.
        self._battle_classes: Dict[str, List[str]] = {"ours": [], "theirs": []}
        # Previous checkpoint's snapshot (hull deltas, closure flips) and its time.
        self._prev_snapshot: Optional["AdmiralSnapshot"] = None
        self._last_decision_t: Optional[float] = None
        # Battle-start torpedo magazine of the fleet (economy line).
        self._magazine_start: Optional[int] = None
        # Compact situation text of the last checkpoint (respond_to_captain).
        self._last_brief: str = ""
        # Every ship's id -> name (both sides), for resolving the "NAME [id]"
        # labels the turn state teaches the admiral to write in tool calls.
        self._ship_names: Dict[str, str] = {}

    @property
    def name(self) -> str:
        """Get Admiral's name (from config or derived from model)."""
        if self.config.name:
            return self.config.name
        # Derive from model if not set
        from .fleet_config import _get_short_model_name
        return f"Admiral {_get_short_model_name(self.config.model)}"

    @property
    def wants_vision(self) -> bool:
        """True if this admiral should receive tactical-plot images."""
        from .admiral_view import is_vision_model
        if not getattr(self.config, "vision", False):
            return False
        verified = getattr(self.config, "image_input", None)
        return bool(verified) if verified is not None else is_vision_model(self.config.model)

    def _plain_ref(self, value: Any) -> Any:
        """An order's target as captains, comms and the recorder match it: the
        ship's full name instead of the "NAME [id]" label (fleet_intel.plain_ship_ref)."""
        return intel.plain_ship_ref(value, self._ship_names)

    def set_ship_mapping(self, name_to_id: Dict[str, str]) -> None:
        """Set the mapping from ship names to IDs."""
        self._ship_name_to_id = name_to_id

    def select_personality(self, num_ships: int, verbose: bool = False) -> Dict[str, Any]:
        """
        Let the Admiral define their command personality before battle.

        Args:
            num_ships: Number of ships under command
            verbose: Whether to print selection info

        Returns:
            Dict with chosen personality info
        """
        from .tools import PERSONALITY_SELECTION_TOOLS

        # Extract a clean model name for personalization
        model_path = self.config.model.replace("openrouter/", "")
        model_name = model_path.split("/")[-1]
        model_name = "-".join(part.capitalize() for part in model_name.split("-"))

        prompt = f"""You are about to take command as Admiral of a fleet in a space combat simulation.

You command {num_ships} ships. As Admiral, you issue strategic orders to your ship captains.

Before the battle begins, define your COMMAND PERSONALITY as an Admiral.
This is about your leadership and strategic style, not individual combat tactics.

Consider:
- Your strategic philosophy (aggressive offense, calculated defense, adaptive flexibility?)
- Your leadership style (direct orders, empowering captains, micromanagement?)
- How you coordinate multiple ships (focus fire, divide and conquer, defensive formations?)
- Your decision-making approach (bold risks, conservative safety, opportunistic?)

Use the choose_personality tool to define your Admiral personality (2-4 sentences).
Be authentic to how you would command a fleet as {model_name}."""

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with personality selection tool (use admiral's configured model)
        try:
            tool_calls = self.client.decide_with_tools(
                messages=messages,
                tools=PERSONALITY_SELECTION_TOOLS,
                model=self.config.model,
            )
        except LLMCallError as e:
            print(f"[ADMIRAL {self.config.name}] personality selection failed: {e}")
            tool_calls = []

        result = {"personality_description": None}

        for tc in tool_calls:
            if tc.name == "choose_personality":
                personality_desc = tc.arguments.get("personality_description", "")
                result["personality_description"] = personality_desc

                if personality_desc:
                    self.config.personality = personality_desc

                if verbose:
                    print(f"  [Admiral {self.name}] Defined command personality")
                    if personality_desc:
                        print(f"    {personality_desc[:80]}...")

        return result

    def capture_pre_snapshot(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
    ) -> None:
        """
        Capture T-15s snapshot for comparison.

        Called 15 seconds before checkpoint to capture pre-snapshot data.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
        """
        self._snapshot_t_minus_15 = self._build_snapshot(simulation, captains)

    def decide(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
        enemy_admiral: Optional['LLMAdmiral'] = None,
        battle_image_png: Optional[bytes] = None,
    ) -> AdmiralDecision:
        """
        Make fleet-level decisions using two-phase approach.

        Phase 1: Set fleet directive (overall strategy)
        Phase 2: Issue order to each ship individually

        Called at checkpoint, BEFORE captains decide.

        A directive call that comes back without set_fleet_directive (output
        truncated at max_tokens, a provider-side error, or simply no call) is
        retried once with a short nudge and a cheaper budget. If that fails
        too, the checkpoint has NO directive: last_directive becomes "" and
        the failure is recorded on the decision - captains are never handed
        the previous checkpoint's text as if it were current (ADM-02).

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
            enemy_admiral: Enemy Admiral (if exists)

        Returns:
            AdmiralDecision with orders for captains
        """
        from .prompts import build_admiral_messages, build_admiral_situation_brief

        now = simulation.current_time
        snapshot_t0 = self._build_snapshot(simulation, captains, full=True)
        decision = AdmiralDecision(timestamp=now)
        try:
            self._last_brief = build_admiral_situation_brief(snapshot_t0)
        except Exception:  # noqa: BLE001 - reply context only
            self._last_brief = ""

        # Get list of active ships that need orders
        active_ships = []
        for captain in captains:
            ship_id = getattr(captain, 'ship_id', None)
            if ship_id:
                ship = simulation.get_ship(ship_id)
                # A crew incapacitated by plume radiation cannot execute
                # anything: it stays in the fleet picture (marked), but no
                # per-ship order call is spent on it.
                if (ship and not ship.is_destroyed
                        and getattr(ship, 'crew_incapacitated', False) is not True):
                    active_ships.append({
                        'ship_id': ship_id,
                        'ship_name': captain.ship_name,
                        'captain_name': captain.name,
                        'ship_type': captain.config.ship_type,
                    })

        # PHASE 1: Get fleet directive (overall strategy). Battle-static
        # doctrine + capabilities go in the system message (a stable,
        # cacheable prefix); the live turn state is the user message.
        messages = build_admiral_messages(
            admiral_name=self.config.name,
            faction=self.faction,
            snapshot_t_minus_15=self._snapshot_t_minus_15,
            snapshot_t_zero=snapshot_t0,
            personality=self.config.personality,
            fleet_data=self.fleet_data,
            enemy_has_admiral=enemy_admiral is not None,
            enemy_proposed_draw=enemy_admiral.has_proposed_draw if enemy_admiral else False,
            received_messages=self._received_enemy_messages,
            phase="directive",  # Signal that we only want the directive
            standing_plan=self.standing_plan,
            standing_plan_time=self.standing_plan_time,
            notebook_text=getattr(self.config, "notebook_text", None),
            decision_interval_s=getattr(simulation, "decision_interval", 30.0),
            captain_reports=self._captain_reports(),
            last_directive_failure=self._last_failure,
            battle_classes=self._battle_classes,
            vision=self.wants_vision,
        )

        user_text = (f"ADMIRAL CHECKPOINT {self.decision_count + 1}. "
                     "Set your fleet directive (overall strategy). "
                     "You will issue individual ship orders next.\n\n"
                     + messages[1]["content"])

        # Attach the tactical plot as an image content part when available.
        # OpenRouter forwards multi-part user content to multimodal models;
        # gating on wants_vision keeps text-only models on plain strings.
        if battle_image_png and self.wants_vision:
            # The plot's legend is in the cached system prompt
            # (prompts.ADMIRAL_VISION_LEGEND); the turn carries only the image.
            from .admiral_view import png_to_data_url
            user_content: Any = [
                {"type": "text", "text": user_text + "\n\nTactical plot attached (legend: TACTICAL PLOT)."},
                {"type": "image_url", "image_url": {"url": png_to_data_url(battle_image_png)}},
            ]
        else:
            user_content = user_text

        messages = [
            messages[0],
            {"role": "user", "content": user_content},
        ]

        tool_calls, failure, recovered, call_error = self._directive_call(messages)
        self._last_decision_t = now
        self._prev_snapshot = snapshot_t0
        if call_error:
            # A failed call is NOT a decision to stand down - flag it so the
            # battle record shows a degraded turn rather than a deliberate one.
            print(f"[ADMIRAL {self.config.name}] directive call failed: {call_error}")
            decision.call_failed = True
            decision.directive_failure = f"call error: {call_error}"
            self._publish_directive(decision, now)
            self.last_decision = decision
            return decision

        # Extract directive and any draw/message actions
        for call in tool_calls:
            name = call.name
            args = call.arguments

            if name == "set_fleet_directive":
                decision.fleet_directive = (args.get("directive") or "").strip()
                decision.salvo_plan = self._salvo_plan(args.get("salvo_plan"))
            elif name == "set_battle_plan":
                plan = args.get("plan", "").strip()
                if plan:
                    self.standing_plan = plan
                    self.standing_plan_time = simulation.current_time
                    decision.battle_plan_update = plan
            elif name == "issue_order":
                # The only order this call accepts: ONE fleet-wide formation
                # heading (ship_name "all ships"). Per-ship orders come next.
                fleet_order = _fleet_heading_order(args)
                if fleet_order is not None:
                    decision.fleet_heading = fleet_order
            elif name == "message_enemy_admiral":
                self._pending_enemy_message = args.get("message", "")
                decision.message_to_enemy_admiral = self._pending_enemy_message
            elif name == "propose_fleet_draw":
                if not self.has_proposed_draw:
                    self.draw_offer_t = now
                self.has_proposed_draw = True
                decision.proposed_draw = True
            elif name == "accept_fleet_draw":
                self.has_accepted_draw = True
                decision.accepted_draw = True
            elif name == "reject_fleet_draw":
                decision.rejected_draw = True

        if failure:
            decision.call_failed = True
            decision.directive_failure = failure
            print(f"[ADMIRAL {self.config.name}] NO fleet directive at T+{now:.0f}s: {failure}")
        decision.directive_recovered = recovered
        self._publish_directive(decision, now)

        # PHASE 2: Issue order to each ship individually. All orders derive
        # from the same frozen snapshot and directive, so they are independent
        # calls - fan them out and collect in fleet order.
        def issue(ship_info: dict) -> Optional[AdmiralOrder]:
            return self._issue_ship_order(
                ship_info=ship_info,
                snapshot_t0=snapshot_t0,
                fleet_directive=decision.fleet_directive,
                simulation=simulation,
                fleet_heading=decision.fleet_heading,
                directive_failure=decision.directive_failure,
                salvo_plan=decision.salvo_plan,
            )

        if self.parallel_ship_orders and len(active_ships) > 1:
            with ThreadPoolExecutor(
                max_workers=min(8, len(active_ships)),
                thread_name_prefix=f"admiral-{self.faction}-orders",
            ) as pool:
                ship_orders = list(pool.map(issue, active_ships))
        else:
            ship_orders = [issue(ship_info) for ship_info in active_ships]

        decision.fleet_orders.extend(o for o in ship_orders if o)

        # Update state
        self.decision_count += 1
        self.order_history.append(decision)

        # Clear received messages after processing
        self._received_enemy_messages.clear()
        self.draw_notices.clear()   # shown in this checkpoint's turn

        self.last_decision = decision
        return decision

    def _publish_directive(self, decision: AdmiralDecision, now: float) -> None:
        """Apply the last_directive contract for this checkpoint."""
        self.last_directive = decision.fleet_directive or ""
        self.last_directive_t = now
        if decision.fleet_directive:
            self._good_directive = (now, decision.fleet_directive)
            self._last_failure = None
        else:
            self._last_failure = (now, decision.directive_failure or "no directive")

    def _captain_reports(self) -> List[Dict[str, Any]]:
        """Reports filed by our captains at the previous checkpoint (comms intake)."""
        reports = getattr(self, "captain_reports", None)
        return [r for r in (reports or []) if isinstance(r, dict)]

    # -- directive call with one cheap retry --------------------------------

    _EFFORT_LADDER = ("xhigh", "high", "medium", "low")

    def _tool_call(self, messages, tools, **overrides) -> Tuple[List[Any], Dict[str, Any], Optional[str]]:
        """(tool calls, call meta, error text) - overrides only where supported."""
        fn = self.client.decide_with_tools
        kwargs = {k: v for k, v in overrides.items()
                  if v is not None and _accepts_kwarg(fn, k)}
        try:
            calls = fn(messages, tools, model=self.config.model, **kwargs)
        except LLMCallError as e:
            return [], {}, str(e)
        meta_fn = getattr(self.client, "last_call_meta", None)
        meta = meta_fn() if callable(meta_fn) else {}
        return list(calls or []), (meta if isinstance(meta, dict) else {}), None

    def _directive_call(self, messages) -> Tuple[List[Any], Optional[str], Optional[str], Optional[str]]:
        """
        Phase-1 call. Returns (tool calls, failure, recovered, call_error):
        failure = why there is still no directive; recovered = why the first
        attempt failed when the retry saved it; call_error = LLMCallError text
        (no retry: the client already retried transient errors).
        """
        calls, meta, error = self._tool_call(messages, self.tools)
        if error:
            return [], None, None, error
        if _has_directive(calls):
            return calls, None, None, None

        first = _failure_text(meta)
        if meta.get("finish_reason") == "refusal":
            # Not re-asked: the checkpoint has no directive and says why
            # (admiral_directive_failed), and the battle goes on.
            print(f"[ADMIRAL {self.config.name}] directive call refused by the model - "
                  "no directive this checkpoint")
            return [], first, None, None
        overrides: Dict[str, Any] = {}
        effort = getattr(self.client, "reasoning_effort", None)
        if meta.get("finish_reason") == "length":
            # Reasoning ate the budget (battle A: 7,005 of 8,192 tokens):
            # one step less reasoning if there is a step left, else more room.
            if isinstance(effort, str) and effort in self._EFFORT_LADDER[:-1]:
                overrides["reasoning_effort"] = self._EFFORT_LADDER[
                    self._EFFORT_LADDER.index(effort) + 1]
            else:
                limit = meta.get("max_tokens") or getattr(self.client, "max_tokens", None)
                if isinstance(limit, int) and limit > 0:
                    overrides["max_tokens"] = int(limit * 1.5)
        print(f"[ADMIRAL {self.config.name}] directive call returned no directive "
              f"({first}) - retrying once with a short-answer nudge")
        retry_calls, retry_meta, retry_error = self._tool_call(
            _with_retry_nudge(messages, first), self.tools, **overrides)
        if retry_error:
            return calls, f"{first}; retry: call error", None, None
        if _has_directive(retry_calls):
            # Keep a battle plan / heading from the cut-off first answer if the
            # short retry did not repeat it.
            names = {c.name for c in retry_calls}
            extra = [c for c in calls if c.name not in names
                     and c.name in ("set_battle_plan", "issue_order")]
            return retry_calls + extra, None, first, None
        return calls + retry_calls, f"{first}; retry: {_failure_text(retry_meta)}", None, None

    def _issue_ship_order(
        self,
        ship_info: dict,
        snapshot_t0: 'AdmiralSnapshot',
        fleet_directive: str,
        simulation: Any,
        fleet_heading: Optional[AdmiralOrder] = None,
        directive_failure: Optional[str] = None,
        salvo_plan: Optional[Dict[str, Tuple[int, Optional[str]]]] = None,
    ) -> Optional[AdmiralOrder]:
        """
        Issue a specific order to a single ship.

        The prompt is built in THAT ship's frame (ranges, closures, torpedo
        solutions and threats measured from it - R6/ADM-08), with its
        captain's last report, and split into a battle-static system message
        plus the per-ship user turn.

        Args:
            ship_info: Dict with ship_id, ship_name, captain_name, ship_type
            snapshot_t0: Current battle snapshot
            fleet_directive: The fleet-wide strategy ("" if none this checkpoint)
            simulation: Current simulation
            directive_failure: why this checkpoint has no directive, if so

        Returns:
            AdmiralOrder for this ship
        """
        from .prompts import build_admiral_ship_order_messages

        try:
            ship_view = self._ship_view(simulation, ship_info['ship_id'], snapshot_t0)
        except Exception as e:  # noqa: BLE001 - advisory data must not kill the order
            print(f"[ADMIRAL {self.config.name}] ship view for {ship_info['ship_id']} failed: {e}")
            ship_view = {}
        report = next((r for r in self._captain_reports()
                       if r.get("ship_name") == ship_info['ship_name']
                       or r.get("ship_id") == ship_info['ship_id']), None)
        previous = None
        if directive_failure and self._good_directive:
            previous = self._good_directive
        messages = build_admiral_ship_order_messages(
            admiral_name=self.config.name,
            ship_name=ship_info['ship_name'],
            ship_type=ship_info['ship_type'],
            captain_name=ship_info['captain_name'],
            fleet_directive=fleet_directive,
            snapshot=snapshot_t0,
            personality=self.config.personality,
            standing_plan=self.standing_plan,
            fleet_heading=fleet_heading,
            ship_id=ship_info['ship_id'],
            ship_view=ship_view,
            captain_report=report,
            directive_failure=directive_failure,
            previous_directive=previous,
            faction=self.faction,
            salvo_plan=self._salvo_plan_view(salvo_plan, ship_info['ship_id']),
        )
        messages[1]["content"] = (
            f"Issue tactical order to {ship_info['ship_name']}. You MUST call issue_order "
            f"for this ship.\n\n{messages[1]['content']}")

        # Use only the issue_order tool for this call
        order_tool = [t for t in self.tools if t.get("function", {}).get("name") == "issue_order"]

        # Call LLM (use admiral's configured model)
        try:
            tool_calls = self.client.decide_with_tools(messages, order_tool, model=self.config.model)
        except LLMCallError as e:
            print(f"[ADMIRAL {self.config.name}] ship-order call failed: {e}")
            return None

        call = next((c for c in tool_calls if c.name == "issue_order"), None)
        note = self._order_text_note(call.arguments, simulation) if call is not None else None
        if note and getattr(call, "id", None):
            # The prose names a ship other than the structured target: say so
            # in the tool result and let the admiral correct it (one follow-up,
            # only on a mismatch).
            follow = list(messages) + [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": call.id, "type": "function", "function": {
                        "name": "issue_order", "arguments": json.dumps(call.arguments or {})}}]},
                {"role": "tool", "tool_call_id": call.id, "content": f"Order recorded. NOTE: {note}"},
                {"role": "user", "content": (
                    f"Issue tactical order to {ship_info['ship_name']}. You MUST call issue_order "
                    "again only to change it - a reply without a call keeps the order above.")}]
            try:
                again = self.client.decide_with_tools(follow, order_tool, model=self.config.model)
            except Exception as e:  # noqa: BLE001 - optional: keep the first order
                print(f"[ADMIRAL {self.config.name}] order follow-up failed: {e}")
                again = []
            second = next((c for c in again or [] if c.name == "issue_order"), None)
            if second is not None:
                call = second
        if call is not None:
            args = call.arguments
            order = AdmiralOrder(
                target_ship_id=ship_info['ship_id'],
                target_ship_name=ship_info['ship_name'],
                order_text=args.get("order_text", "Continue as directed."),
                priority=args.get("priority", "NORMAL"),
                suggested_target=self._plain_ref(args.get("suggested_target")),
                torpedo_salvo=args.get("torpedo_salvo"),
                torpedo_target=self._plain_ref(args.get("torpedo_target")),
            )
            self._apply_salvo_plan(order, args, salvo_plan)
            _attach_heading(order, args, fleet_heading)
            try:
                order.text_also_names = self._prose_only_targets(args, simulation)
            except Exception:  # noqa: BLE001 - advisory only
                order.text_also_names = []
            return order

        # If no order was issued, create a default order
        basis = (f"fleet directive: {fleet_directive[:100]}..." if fleet_directive
                 else "standing orders (no fleet directive this checkpoint)")
        order = AdmiralOrder(
            target_ship_id=ship_info['ship_id'],
            target_ship_name=ship_info['ship_name'],
            order_text=f"[AUTO] Continue per {basis}",
            priority="NORMAL",
            suggested_target=None,
        )
        self._apply_salvo_plan(order, {}, salvo_plan)
        _attach_heading(order, {}, fleet_heading)
        return order

    # -- salvo allocation and order consistency --------------------------------

    def _salvo_plan(self, raw: Any) -> Dict[str, Tuple[int, Optional[str]]]:
        """set_fleet_directive salvo_plan -> {own ship id: (rounds, target)}."""
        plan: Dict[str, Tuple[int, Optional[str]]] = {}
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            sid = intel.resolve_ship_ref(entry.get("ship_name"), self._ship_names) \
                or self._ship_name_to_id.get(str(entry.get("ship_name") or ""))
            if not sid or not str(sid).startswith(self.faction):
                continue
            try:
                rounds = max(0, int(entry.get("rounds", 0)))
            except (TypeError, ValueError):
                continue
            plan[sid] = (rounds, self._plain_ref(entry.get("target")) if entry.get("target") else None)
        return plan

    def _salvo_plan_view(self, plan: Optional[Dict[str, Tuple[int, Optional[str]]]],
                         ship_id: str) -> Optional[Dict[str, Any]]:
        if not plan:
            return None
        mine = plan.get(ship_id)
        targets: Dict[str, int] = {}
        for rounds, target in plan.values():
            if rounds:
                targets[target or "?"] = targets.get(target or "?", 0) + rounds
        return {"mine": mine, "total": sum(r for r, _ in plan.values()), "by_target": targets,
                "ships": len([1 for r, _ in plan.values() if r])}

    def _apply_salvo_plan(self, order: AdmiralOrder, args: Dict[str, Any],
                          plan: Optional[Dict[str, Tuple[int, Optional[str]]]]) -> None:
        """The directive's allocation flies unless this order sets its own salvo."""
        mine = (plan or {}).get(order.target_ship_id)
        if mine is None:
            return
        if args.get("torpedo_salvo") is None:
            order.torpedo_salvo, alloc_target = mine
            if order.torpedo_target is None and alloc_target:
                order.torpedo_target = alloc_target
            order.salvo_source = "directive"
        else:
            order.salvo_source = "order"

    def _enemy_and_own_ids(self, simulation: Any) -> Tuple[List[str], List[str]]:
        ships = list((getattr(simulation, "ships", None) or {}).values())
        own = [s.ship_id for s in ships if getattr(s, "faction", None) == self.faction]
        enemy = [s.ship_id for s in ships if getattr(s, "faction", None) != self.faction
                 and intel.is_live(s)]
        for s in ships:
            self._ship_names.setdefault(s.ship_id, intel.ship_name(s))
        return own, enemy

    def _order_text_note(self, args: Dict[str, Any], simulation: Any) -> Optional[str]:
        try:
            own, enemy = self._enemy_and_own_ids(simulation)
            return intel.order_text_target_note(args or {}, self._ship_names, own, enemy)
        except Exception:  # noqa: BLE001 - advisory only
            return None

    def _prose_only_targets(self, args: Dict[str, Any], simulation: Any) -> List[str]:
        own, enemy = self._enemy_and_own_ids(simulation)
        short_own = [self._ship_names.get(s, "").split(" ", 1)[1] for s in own
                     if " " in self._ship_names.get(s, "")]
        structured = {intel.resolve_ship_ref(args.get(k), self._ship_names)
                      for k in ("suggested_target", "maneuver_target", "torpedo_target") if args.get(k)}
        named = intel.ships_named_in_text(args.get("order_text") or "", self._ship_names, enemy, short_own)
        return [intel.label(self._ship_names.get(s), s) for s in named if s not in structured]

    def _ship_view(self, simulation: Any, ship_id: str, snapshot: 'AdmiralSnapshot') -> Dict[str, Any]:
        """
        One ship's own frame for its order prompt: every enemy's range,
        closure and torpedo solution measured FROM THIS SHIP, the rounds
        inbound on it, and its own range-control block.
        """
        from .range_control import analyze_range_control, format_range_control
        ship = simulation.get_ship(ship_id)
        if ship is None:
            return {}
        enemies = [e for e in simulation.get_enemy_ships(ship_id) if intel.is_live(e)]
        has_tubes = bool(getattr(ship, "torpedoes_remaining", 0))
        specs = intel.torpedo_specs_for(simulation, self.fleet_data, [ship]) if has_tubes else None
        salvo = intel.launch_capacity(simulation, ship)
        rows = []
        for enemy in enemies:
            rel = enemy.position - ship.position
            rng = rel.magnitude
            row = {
                "id": enemy.ship_id,
                "range_km": rng / 1000.0,
                "closing_kps": intel._closing(rel, enemy.velocity - ship.velocity) / 1000.0,
                "bearing": intel.direction_str(rel),
                "solution": None,
                "facing": intel.observed_facing(enemy, ship.position),
            }
            if specs is not None:
                row["solution"] = intel.torpedo_solution(
                    ship.position, ship.velocity, enemy, specs, self.fleet_data,
                    simulation, salvo_available=max(1, snapshot.launches_available or salvo))
            rows.append(row)
        rows.sort(key=lambda r: r["range_km"])
        inbound = [r for r in snapshot.rounds
                   if r.threat_id == ship_id and (r.side == "theirs" or r.ballistic)]
        try:
            infos = analyze_range_control(simulation, [ship], detail=True)
            rc_text = format_range_control(infos, who="Ship", max_detail=1)
            rc_text = intel.replace_option_rows(rc_text, simulation, infos[:1], [ship])
        except Exception:  # noqa: BLE001 - advisory only
            rc_text = ""
        return {"enemies": rows, "inbound": inbound, "range_control": rc_text,
                "launches_available": salvo}

    def respond_to_captain(
        self,
        captain_ship_name: str,
        question: str,
        simulation: Any,
        ship_id: Optional[str] = None,
    ) -> str:
        """
        Respond to a captain's discussion request.

        The reply prompt alone holds only the plan and recent orders; the
        user turn adds a compact live picture measured from the asking ship
        (its state, enemies by range and closing speed, torpedoes on it),
        this checkpoint's directive and order, and earlier exchanges with the
        same captain. The exchange is remembered in `discussion_log`.

        Args:
            captain_ship_name: Name of the captain's ship
            question: Captain's question
            simulation: Current simulation state
            ship_id: The asking ship's id (looked up from the name if omitted)

        Returns:
            Admiral's response text
        """
        from .prompts import build_admiral_response_prompt
        from .fleet_comms import discussion_context

        # Get recent decisions for context (up to last 3)
        recent_decisions = self.order_history[-3:] if self.order_history else []

        # Build context for response
        prompt = build_admiral_response_prompt(
            admiral_name=self.config.name,
            captain_ship_name=captain_ship_name,
            question=question,
            personality=self.config.personality,
            recent_decisions=recent_decisions,
            standing_plan=self.standing_plan,
            decision_interval_s=getattr(simulation, "decision_interval", 30.0),
            situation=getattr(self, "_last_brief", "") or None,
        )

        ship_id = ship_id or self._ship_name_to_id.get(captain_ship_name)
        current = self.order_history[-1] if self.order_history else None
        order_text = None
        if current is not None:
            for order in current.fleet_orders:
                if order.target_ship_name == captain_ship_name or order.target_ship_id == ship_id:
                    order_text = order.order_text
                    break
        context = discussion_context(
            simulation, ship_id, captain_ship_name,
            directive=self.last_directive,
            order_text=order_text,
            earlier=[qa for qa in self.discussion_log if qa["ship_name"] == captain_ship_name],
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{context}\n\nCaptain of {captain_ship_name} asks: {question}"},
        ]

        # Get response (no tools, just text). Pass the admiral's own model -
        # otherwise a shared client answers as whichever model built it.
        try:
            response = self.client.complete(messages, model=self.config.model)
        except LLMCallError as e:
            print(f"[ADMIRAL {self.config.name}] reply to captain failed: {e}")
            return ""
        answer = response.content or ""
        self.discussion_log.append({
            "t": getattr(simulation, "current_time", 0.0),
            "checkpoint": self.decision_count,
            "ship_name": captain_ship_name,
            "question": question,
            "answer": answer,
        })
        return answer

    def receive_enemy_admiral_message(self, message: str) -> None:
        """Receive a message from enemy Admiral."""
        self._received_enemy_messages.append(message)

    def get_pending_enemy_message(self) -> Optional[str]:
        """Get and clear pending message to enemy Admiral."""
        msg = self._pending_enemy_message
        self._pending_enemy_message = None
        return msg

    def _build_snapshot(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
        full: bool = False,
    ) -> AdmiralSnapshot:
        """
        Build a snapshot of the current battle state.

        full=True (the checkpoint snapshot) adds everything the turn state is
        built from: fleet-centre geometry per enemy group, side-attributed
        ordnance, the since-last-checkpoint digest, the threat table and the
        torpedo economy. The T-15 s pre-snapshot only needs positions.
        """
        # Get friendly ships
        friendly_snapshots = []
        own_ships = []
        for captain in captains:
            # Use ship_id to look up ship (set by battle_runner)
            ship_id = getattr(captain, 'ship_id', None)
            if not ship_id:
                continue
            ship = simulation.get_ship(ship_id)
            if ship and not ship.is_destroyed:
                own_ships.append(ship)
                friendly_snapshots.append(
                    self._build_friendly_snapshot(ship, captain, simulation)
                )

        # Get enemy ships
        enemy_snapshots = []
        enemies = []
        if captains:
            # Get first captain's ship to find enemies
            first_ship_id = getattr(captains[0], 'ship_id', None)
            if not first_ship_id:
                first_ship_id = captains[0].config.ship_name
            for enemy in simulation.get_enemy_ships(first_ship_id):
                if not enemy.is_destroyed:
                    enemies.append(enemy)
                    enemy_snapshots.append(
                        self._build_enemy_snapshot(enemy, friendly_snapshots, simulation)
                    )

        # Get projectiles
        projectile_snapshots = self._build_projectile_snapshots(simulation)

        # Build fleet summary
        ship_types = [s.ship_type for s in friendly_snapshots]
        type_counts = {}
        for t in ship_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        summary_parts = [f"{count} {stype.title()}" for stype, count in type_counts.items()]
        fleet_summary = f"{len(friendly_snapshots)} ships: {', '.join(summary_parts)}"

        snapshot = AdmiralSnapshot(
            timestamp=simulation.current_time,
            friendly_ships=friendly_snapshots,
            enemy_ships=enemy_snapshots,
            projectiles=projectile_snapshots,
            fleet_summary=fleet_summary,
            faction=self.faction,
        )
        if full:
            self._enrich_snapshot(snapshot, simulation, own_ships, enemies)
        return snapshot

    def _enrich_snapshot(
        self,
        snap: AdmiralSnapshot,
        simulation: Any,
        own_ships: List[Any],
        enemies: List[Any],
    ) -> None:
        """Fill the turn-state fields of a checkpoint snapshot (see _build_snapshot)."""
        from .range_control import (analyze_range_control, current_accel,
                                    format_heading_options, format_range_control,
                                    formation_line)
        now = simulation.current_time
        interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0)
        snap.checkpoint = self.decision_count + 1
        snap.decision_interval_s = interval
        snap.clock = getattr(self, "battle_clock", None)
        snap.draw_lines = list(getattr(self, "draw_notices", None) or [])
        if self.has_proposed_draw and self.draw_offer_t is not None:
            snap.draw_lines.append(f"Your draw proposal (T+{self.draw_offer_t:.0f}s) is open: "
                                   "no answer from the enemy admiral yet.")
        if getattr(self.config, "show_score", False):
            try:
                snap.score_line = intel.score_line(simulation, self.faction)
            except Exception:  # noqa: BLE001 - advisory only
                snap.score_line = ""
        for sid, ship in getattr(simulation, "ships", {}).items():
            snap.labels[sid] = intel.ship_label(ship)
            snap.ship_types[sid] = getattr(ship, "ship_type", "") or ""
            self._ship_names[sid] = intel.ship_name(ship)
        for kind, rows in (("ours", snap.friendly_ships), ("theirs", snap.enemy_ships)):
            for row in rows:
                if row.ship_type and row.ship_type not in self._battle_classes[kind]:
                    self._battle_classes[kind].append(row.ship_type)

        live_own = [s for s in own_ships if intel.is_live(s)]
        live_enemies = [e for e in enemies if intel.is_live(e)]

        # Geometry: fleet centre vs enemy groups (range control).
        try:
            infos = analyze_range_control(simulation, live_own, live_enemies, detail=False) \
                if live_own and live_enemies else []
        except Exception as e:  # noqa: BLE001 - advisory only
            print(f"[ADMIRAL {self.config.name}] range control failed: {e}")
            infos = []
        snap.range_groups = infos
        snap.merge_alarms = self._damp_merge_alarms(infos, now)
        formation = formation_line(live_own)
        try:
            # Uniform rows: range effect, the face you show, the face HE shows
            # you afterwards (fleet_intel.heading_option_rows).
            options = intel.admiral_headings_block(simulation, infos, live_own)
        except Exception as e:  # noqa: BLE001 - advisory only
            print(f"[ADMIRAL {self.config.name}] heading rows failed: {e}")
            options = format_heading_options(infos)
        snap.headings = "\n".join(x for x in (options, formation) if x)
        snap.range_control = "\n".join(
            x for x in (format_range_control(infos, who="Fleet"), formation) if x)
        group_of = {sid: info.label for info in infos for sid in info.ship_ids}
        if live_own:
            own_pos, own_vel = intel.centre(live_own)
            spread = max((s.position - own_pos).magnitude for s in live_own) / 1000.0
            snap.own_centre = {
                "pos_km": (own_pos.x / 1000.0, own_pos.y / 1000.0, own_pos.z / 1000.0),
                "speed_kps": own_vel.magnitude / 1000.0,
                "direction": intel.direction_str(own_vel),
                "spread_km": spread, "n": len(live_own),
            }
            members = {}
            for e in live_enemies:
                members.setdefault(group_of.get(e.ship_id, ""), []).append(e)
            prev15 = self._snapshot_t_minus_15
            for info in infos:
                group = members.get(info.label) or []
                if not group:
                    continue
                g_pos, g_vel = intel.centre(group)
                snap.group_motion[info.label] = intel.motion_split(own_pos, own_vel, g_pos, g_vel)
                snap.group_burn_g[info.label] = intel.burn_toward_g(
                    own_pos, g_pos, [current_accel(m) for m in group])
                delta = _range_change(prev15, [s.ship_id for s in live_own], info.ship_ids,
                                      info.range_km)
                if delta is not None:
                    snap.group_range_change_km[info.label] = delta
        else:
            own_pos = own_vel = None

        # Ordnance in flight, side-attributed (ADM-01).
        launcher_ships = [s for s in live_own if getattr(s, "torpedoes_remaining", 0)
                          or getattr(s, "ready_torpedo_launchers", None)]
        specs = (intel.torpedo_specs_for(simulation, self.fleet_data, launcher_ships)
                 if launcher_ships else None)
        snap.rounds = intel.gather_rounds(simulation, self.faction, specs)
        try:
            snap.torpedo_defence = intel.torpedo_defence(simulation, self.faction, self.fleet_data,
                                                         snap.rounds)
        except Exception as e:  # noqa: BLE001 - advisory only
            print(f"[ADMIRAL {self.config.name}] torpedo defence forecast failed: {e}")

        # Since the last checkpoint (ADM-06).
        since = self._last_decision_t if self._last_decision_t is not None else now - interval
        snap.digest = intel.checkpoint_digest(simulation, self.faction, since, now)
        prev = self._prev_snapshot
        if prev is not None:
            snap.hull_before = {r.ship_id: r.hull_integrity for r in prev.friendly_ships}
            snap.hull_before.update({r.ship_id: r.hull_integrity for r in prev.enemy_ships
                                     if r.hull_integrity is not None})
        for ship in list(own_ships) + list(enemies):
            snap.hull_now[ship.ship_id] = float(getattr(ship, "hull_integrity", 100.0))
        for entry in snap.digest.kills + snap.digest.losses:
            dead = simulation.get_ship(entry["ship_id"])
            if dead is not None:
                snap.hull_now.setdefault(entry["ship_id"], float(dead.hull_integrity))

        # Torpedo economy.
        remaining = sum(r.torpedoes_remaining for r in snap.friendly_ships)
        if self._magazine_start is None:
            self._magazine_start = sum(r.magazine_capacity for r in snap.friendly_ships)
        snap.torpedo_magazine = (remaining, self._magazine_start or 0)
        snap.launches_available = sum(r.launches_available for r in snap.friendly_ships)
        snap.reload_s = float(((self.fleet_data or {}).get("weapon_types") or {})
                              .get("torpedo_launcher", {}).get("cooldown_s", 12.0))
        ledger = intel.damage_ledger(simulation, self.faction)
        snap.enemy_torpedoes_seen = int(sum(r["torpedoes_launched"] for r in ledger.values()))

        # Threat table rows: fleet-centre frame, damage dealt, torpedo solution.
        prev_closing = {r.ship_id: r.closing_kps for r in (prev.enemy_ships if prev else [])}
        live_launchers = [s for s in launcher_ships if intel.is_live(s)]
        launch_pos = launch_vel = None
        if live_launchers:
            launch_pos, launch_vel = intel.centre(live_launchers)
        inbound_ours = {}
        for r in snap.rounds:
            if r.side == "ours" and r.state == "guided":
                inbound_ours[r.target_id] = inbound_ours.get(r.target_id, 0) + 1
        by_id = {e.ship_id: e for e in enemies}
        for row in snap.enemy_ships:
            enemy = by_id.get(row.ship_id)
            if enemy is None:
                continue
            row.group = group_of.get(row.ship_id, "")
            if own_pos is not None:
                rel = enemy.position - own_pos
                row.range_km = rel.magnitude / 1000.0
                row.closing_kps = intel._closing(rel, enemy.velocity - own_vel) / 1000.0
                row.facing = intel.observed_facing(enemy, own_pos)
            if live_own:
                nearest = min(live_own, key=lambda s: (s.position - enemy.position).magnitude)
                row.nearest_own_id = nearest.ship_id
            hit = ledger.get(row.ship_id) or {}
            row.dealt_hits = int(hit.get("hits", 0))
            row.dealt_gj = float(hit.get("gj", 0.0))
            row.torpedoes_launched = int(hit.get("torpedoes_launched", 0))
            row.our_rounds_inbound = inbound_ours.get(row.ship_id, 0)
            snap.kill_rounds[row.ship_id] = intel.kill_rounds(row.ship_type, row.hull_integrity)
            if specs is not None and launch_pos is not None and intel.is_live(enemy):
                row.solution = intel.torpedo_solution(
                    launch_pos, launch_vel, enemy, specs, self.fleet_data, simulation,
                    salvo_available=max(1, snap.launches_available))
        for info in infos:
            was = [prev_closing.get(sid) for sid in info.ship_ids if prev_closing.get(sid) is not None]
            if info.closing_kps < 0 and was and sum(1 for c in was if c > 0) * 2 > len(was):
                snap.groups_opened.append(info.label)

        snap.engine_notes = intel.engine_notes(
            simulation, [s.ship_id for s in own_ships], since,
            label_of=lambda sid: snap.labels.get(sid, sid))
        try:
            snap.noncompliance = self._track_noncompliance(simulation, own_ships, since, now, snap)
        except Exception as e:  # noqa: BLE001 - advisory only
            print(f"[ADMIRAL {self.config.name}] non-compliance check failed: {e}")

    @staticmethod
    def _alarm_kind(warning: str) -> str:
        if "out-accelerates" in warning:
            return "he out-accelerates you"
        if "too late" in warning:
            return "too late to brake"
        return "brake now or pass"

    def _damp_merge_alarms(self, infos: List[Any], now: float) -> Dict[str, str]:
        """
        A merge alarm repeated unchanged (same group, same kind) at
        consecutive checkpoints is shown once in full, then as one short line
        with its count - one side got the same alarm eight checkpoints running
        (audit A9). The full numbers stay in GEOMETRY.
        """
        seen = self.__dict__.setdefault("_merge_alarm_seen", {})
        interval = None
        out: Dict[str, str] = {}
        current = {}
        for info in infos:
            if not info.warning:
                continue
            kind = self._alarm_kind(info.warning)
            prev = seen.get(info.label)
            last_t = self._last_decision_t
            if prev and prev["kind"] == kind and last_t is not None and abs(prev["t"] - last_t) < 1e-6:
                entry = {"kind": kind, "t": now, "first": prev["first"], "n": prev["n"] + 1}
                out[info.label] = (f"merge alarm unchanged since T+{entry['first']:.0f}s ({kind}; "
                                   f"{entry['n']} checkpoints running): closest {info.cpa_km:.1f} km "
                                   f"in {info.cpa_t_s:.0f} s - numbers in GEOMETRY")
            else:
                entry = {"kind": kind, "t": now, "first": now, "n": 1}
            current[info.label] = entry
        seen.clear()
        seen.update(current)
        return out

    # An ordered burn whose maneuver requested thrust that the drive did not
    # produce for at least this share of the window was not flown. Judged on
    # requested vs applied throttle, so a ship coasting by design (HOLD_RANGE
    # at its range, a standoff reached, PADLOCK) is never flagged.
    NONCOMPLIANCE_HELD = 0.5
    NONCOMPLIANCE_STREAK = 2

    def _track_noncompliance(self, simulation: Any, own_ships: List[Any], since: float,
                             now: float, snap: "AdmiralSnapshot") -> List[str]:
        """
        Escalation for the NOW block (audit A10): a ship that has not flown
        its ordered maneuver for NONCOMPLIANCE_STREAK checkpoints running -
        its captain replaced the order (compliance deviation), or an ordered
        burn never burned - with what it flew instead and the engine's reason
        (drive dead, evasion held the drive). The admiral re-issued the same
        HEADING seven times to a ship that could not fly it.
        """
        streaks = self.__dict__.setdefault("_noncompliance", {})
        last = self.order_history[-1] if self.order_history else None
        if last is None or self._last_decision_t is None:
            return []
        orders = {o.target_ship_id: o for o in last.fleet_orders}
        reports = {r.get("ship_id") or self._ship_name_to_id.get(r.get("ship_name")): r
                   for r in self._captain_reports()}
        live = {s.ship_id: s for s in own_ships if intel.is_live(s)}
        helm = intel.helm_summary(simulation, list(live), since, now)
        lines = []
        for sid, ship in live.items():
            order = orders.get(sid)
            spec = getattr(order, "maneuver", None) or {}
            kind = spec.get("maneuver_type") or ("HEADING" if getattr(order, "heading", None) else None)
            if order is None or kind is None:
                streaks.pop(sid, None)
                continue
            thr = getattr(order, "applied_throttle", None)
            thr = getattr(order, "throttle", None) if thr is None else thr
            thr = 1.0 if thr is None else float(thr)
            # Burn orders only: BRAKE idles once the ship is at rest and EVADE
            # with nothing to dodge weaves on and off - both as ordered.
            wants_burn = kind in ("INTERCEPT", "HOLD_RANGE") or (kind == "HEADING" and thr > 0)
            h = helm.get(sid) or {}
            comp = ((reports.get(sid) or {}).get("compliance") or {})
            replaced = any(str(d).startswith("maneuver:") for d in comp.get("deviations") or [])
            flew = comp.get("executed_maneuver") or (ship.current_maneuver.maneuver_type.name
                                                     if getattr(ship, "current_maneuver", None) else "?")
            burned = float(h.get("burn_frac") or 0.0)
            engine = h.get("engine_fraction")
            engine = float(getattr(ship, "engine_effectiveness", 1.0)) if engine is None else float(engine)
            held = float(h.get("held_frac") or 0.0)
            if not (replaced or (wants_burn and held >= self.NONCOMPLIANCE_HELD)):
                streaks.pop(sid, None)
                continue
            if engine <= 0.0:
                why = "drive dead (0 g): it cannot burn, only turn on RCS"
            elif replaced:
                why = f"its captain replaced your order (flew {flew})"
            else:
                modes = {m: f for m, f in (h.get("modes") or {}).items()
                         if m in intel.EVASION_MODES and m != "WOBBLE"}
                if modes and sum(modes.values()) >= 0.4:
                    top = max(modes, key=modes.get)
                    why = (f"evasion held the drive ({top} {100 * sum(modes.values()):.0f}% of the "
                           f"window: {intel.EVASION_MODES[top]})")
                else:
                    why = "the drive stayed idle (turning onto the burn line, or interlock)"
            ordered = kind + (f" @{thr * 100:.0f}%" if wants_burn and kind == "HEADING" else "")
            entry = streaks.setdefault(sid, {"n": 0, "ordered": ordered})
            entry["n"] += 1
            entry["ordered"] = ordered
            if entry["n"] >= self.NONCOMPLIANCE_STREAK:
                lines.append(
                    f"{snap.labels.get(sid, sid)}: has not flown your {ordered} for {entry['n']} "
                    f"checkpoints running - flew {flew}, drive burning {100 * burned:.0f}% of the "
                    f"last window; reason: {why}")
        return lines

    def _build_friendly_snapshot(
        self,
        ship: Any,
        captain: 'LLMCaptain',
        simulation: Any,
    ) -> FriendlyShipSnapshot:
        """Build full snapshot for a friendly ship."""
        # Get ship capabilities from fleet data
        ship_spec = self.fleet_data.get("ships", {}).get(captain.config.ship_type, {})
        performance = ship_spec.get("performance", {})

        # Build weapons summary
        weapons_summary = self._build_weapons_summary(ship_spec)

        # Weapons status
        weapons_ready = []
        weapons_cooling = []
        weapons_destroyed = []
        if hasattr(ship, 'weapons'):
            for slot, weapon in ship.weapons.items():
                if slot.startswith("pd_"):
                    continue
                # Check if weapon is destroyed
                if hasattr(weapon, 'is_operational') and not weapon.is_operational:
                    weapons_destroyed.append(slot)
                elif weapon.cooldown_remaining <= 0:
                    weapons_ready.append(slot)
                else:
                    weapons_cooling.append(f"{slot} ({weapon.cooldown_remaining:.0f}s)")

        # Get targeting info
        targeted_by = []
        for enemy in simulation.get_enemy_ships(ship.ship_id):
            if hasattr(enemy, 'primary_target_id') and enemy.primary_target_id == ship.ship_id:
                targeted_by.append(intel.ship_label(enemy))

        # Current maneuver
        maneuver_str = "MAINTAIN"
        if hasattr(ship, 'current_maneuver') and ship.current_maneuver:
            maneuver_str = ship.current_maneuver.maneuver_type.name

        # Position in km
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        # Velocity
        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        # Torpedo magazine (sums every launcher) and PD turret state
        torpedoes_remaining = getattr(ship, 'torpedoes_remaining', 0) or 0
        pd_list = getattr(ship, 'point_defense', None) or []
        pd_total = len(pd_list)
        pd_operational = sum(
            1 for pd in pd_list if getattr(pd, 'is_operational', True)
        )
        armament = intel.class_armament(captain.config.ship_type, self.fleet_data)

        thermal_system = getattr(ship, 'thermal_system', None)
        return FriendlyShipSnapshot(
            thermal=thermal_system.outlook() if thermal_system is not None else None,
            torpedoes_remaining=torpedoes_remaining,
            pd_turrets_total=pd_total,
            pd_turrets_operational=pd_operational,
            launchers=armament["launchers"],
            magazine_capacity=armament["magazine"],
            launches_available=intel.launch_capacity(simulation, ship),
            gun_range_km=armament["gun_range_km"],
            ship_id=ship.ship_id,
            ship_name=getattr(ship, 'name', None) or ship.ship_id,
            ship_type=captain.config.ship_type,
            captain_name=captain.config.name,
            position_km=pos_km,
            velocity_kps=vel_kps,
            velocity_vector=vel_vector,
            hull_integrity=ship.hull_integrity,
            delta_v_remaining=ship.remaining_delta_v_kps,
            heat_percent=ship.heat_percent if hasattr(ship, 'heat_percent') else 0,
            max_acceleration_g=performance.get("combat_acceleration_g", 2.0),
            max_delta_v=performance.get("delta_v_kps", 500),
            weapons_summary=weapons_summary,
            weapons_ready=weapons_ready,
            weapons_cooling=weapons_cooling,
            weapons_destroyed=weapons_destroyed,
            fittings_damaged=_damaged_fitting_lines(ship),
            plume_wash=_plume_status_lines(simulation, ship) + _arcs_status_lines(simulation, ship),
            crew_state=_crew_state(ship),
            crew_dose_sv=float(getattr(ship, "crew_dose_sv", 0.0) or 0.0),
            drive_fraction=float(getattr(ship, "engine_effectiveness", 1.0)
                                 if isinstance(getattr(ship, "engine_effectiveness", 1.0), (int, float))
                                 else 1.0),
            current_maneuver=maneuver_str,
            current_target=ship.primary_target_id if hasattr(ship, 'primary_target_id') else None,
            radiators_extended=ship.radiators_extended if hasattr(ship, 'radiators_extended') else False,
            targeted_by=targeted_by,
        )

    def _build_enemy_snapshot(
        self,
        ship: Any,
        friendly_ships: List[FriendlyShipSnapshot],
        simulation: Any,
    ) -> EnemyShipSnapshot:
        """Build observable snapshot for an enemy ship."""
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        # Range and closing rate to the NEAREST friendly hull. The fleet-level
        # numbers (from the fleet centre) are filled in by _enrich_snapshot.
        min_dist = float('inf')
        closing_rate = 0.0
        for friendly in friendly_ships:
            # Calculate distance
            dx = pos_km["x"] - friendly.position_km["x"]
            dy = pos_km["y"] - friendly.position_km["y"]
            dz = pos_km["z"] - friendly.position_km["z"]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            if dist < min_dist:
                min_dist = dist
                # Calculate closing rate (positive = closing)
                dvx = vel_vector["x"] - friendly.velocity_vector["x"]
                dvy = vel_vector["y"] - friendly.velocity_vector["y"]
                dvz = vel_vector["z"] - friendly.velocity_vector["z"]
                # Dot product of relative velocity with direction vector
                if dist > 0:
                    closing_rate = -(dx*dvx + dy*dvy + dz*dvz) / dist

        ship_type = ship.ship_type if hasattr(ship, 'ship_type') else "unknown"
        armament = intel.class_armament(ship_type, self.fleet_data)
        pd_op, pd_total = intel.pd_turrets(ship)
        return EnemyShipSnapshot(
            ship_id=ship.ship_id,
            ship_name=getattr(ship, 'name', None) or ship.ship_id,
            ship_type=ship_type,
            position_km=pos_km,
            velocity_kps=vel_kps,
            velocity_vector=vel_vector,
            distance_from_closest_friendly_km=min_dist,
            closing_rate_kps=closing_rate,
            hull_integrity=float(getattr(ship, "hull_integrity", 100.0)),
            accel_g=intel.max_accel_ms2(ship) / 9.80665,
            pd_operational=pd_op,
            pd_total=pd_total,
            crew_state=_crew_state(ship),
            armament=armament["summary"],
            launchers=armament["launchers"],
            magazine=armament["magazine"],
            targeting_id=getattr(ship, "primary_target_id", None),
            gun_range_km=armament["gun_range_km"],
        )

    def _build_projectile_snapshots(self, simulation: Any) -> List[ProjectileSnapshot]:
        """
        Build snapshots of ordnance in flight, side-attributed and classified.

        Torpedoes were previously invisible to the admiral entirely - the one
        weapon whose defense (ordering EVADE, massing PD) is the admiral's job.
        Then they were all shown, our own included and blinded wrecks receding
        from everyone counted as "inbound" (ADM-01): a dead round now carries a
        threat_ship only when the engine's closest-approach test says it will
        actually strike that hull.
        """
        snapshots = []
        rounds = intel.gather_rounds(simulation, self.faction)
        for r in sorted(rounds, key=lambda r: (r.kind != "torpedo", r.eta_s if r.eta_s is not None else 1e9)):
            if r.kind == "torpedo":
                label = ("Torpedo (guided)" if r.state == "guided"
                         else "Torpedo (BLINDED, ballistic)" if r.state == "blinded"
                         else f"Torpedo ({r.state.upper()}, ballistic)")
            else:
                label = r.weapon
            snapshots.append(ProjectileSnapshot(
                source_ship=r.source_id,
                target_ship=r.threat_id or r.target_id,
                weapon_type=label,
                distance_km=r.range_km if r.range_km is not None else 0.0,
                eta_seconds=r.eta_s if r.eta_s is not None else float("inf"),
                damage_gj=r.energy_gj,
                side=r.side,
                state=r.state,
                threat_ship=r.threat_id,
            ))
        return snapshots

    def _build_weapons_summary(self, ship_spec: Dict[str, Any]) -> str:
        """Build a human-readable weapons summary from ship spec."""
        weapons = ship_spec.get("weapons", [])
        weapon_types = self.fleet_data.get("weapon_types", {})

        counts = {}
        for weapon in weapons:
            wtype = weapon.get("type", "")
            if wtype.startswith("pd_"):
                continue  # Skip point defense

            # Get weapon stats
            wspec = weapon_types.get(wtype, {})
            damage = wspec.get("kinetic_energy_gj", 0)

            # Use the weapon's real display name from fleet data. The previous
            # substring guessing collapsed distinct weapons onto wrong labels:
            # "heavy_siege_coiler_mk3" (a 7.25 GJ triple-nose spinal mount) has
            # no "spinal" in its key, so it fell through to "Heavy Coilgun",
            # and "light_coilgun_mk3" (0.125 GJ) was reported as plain
            # "Coilgun" alongside the 0.72 GJ turret of the same name.
            display = wspec.get("name")
            if display:
                name = f"{display} ({damage:.1f} GJ)" if damage else str(display)
            else:
                name = wtype

            counts[name] = counts.get(name, 0) + 1

        parts = [f"{count}x {name}" for name, count in counts.items()]
        return ", ".join(parts) if parts else "No weapons"

    def _execute_tools(
        self,
        tool_calls: List[Any],
        timestamp: float,
    ) -> AdmiralDecision:
        """Execute tool calls and build decision."""
        decision = AdmiralDecision(timestamp=timestamp)

        for call in tool_calls:
            name = call.name
            args = call.arguments

            if name == "issue_order":
                ship_name = args.get("ship_name", "")
                ship_id = self._ship_name_to_id.get(ship_name)
                if ship_id is None:
                    # "NAME [id]" as the turn state labels every ship.
                    ship_id = intel.resolve_ship_ref(ship_name, self._ship_names) or ship_name

                fleet_order = _fleet_heading_order(args)
                if fleet_order is not None:
                    decision.fleet_heading = fleet_order
                    continue
                order = AdmiralOrder(
                    target_ship_id=ship_id,
                    target_ship_name=ship_name,
                    order_text=args.get("order_text", ""),
                    priority=args.get("priority", "NORMAL"),
                    suggested_target=self._plain_ref(args.get("suggested_target")),
                    torpedo_salvo=args.get("torpedo_salvo"),
                    torpedo_target=self._plain_ref(args.get("torpedo_target")),
                )
                _attach_heading(order, args, None)
                decision.fleet_orders.append(order)

            elif name == "set_fleet_directive":
                decision.fleet_directive = args.get("directive", "")
                self.last_directive = decision.fleet_directive

            elif name == "set_battle_plan":
                plan = args.get("plan", "").strip()
                if plan:
                    self.standing_plan = plan
                    self.standing_plan_time = timestamp
                    decision.battle_plan_update = plan

            elif name == "message_enemy_admiral":
                self._pending_enemy_message = args.get("message", "")
                decision.message_to_enemy_admiral = self._pending_enemy_message

            elif name == "propose_fleet_draw":
                self.has_proposed_draw = True
                decision.proposed_draw = True

            elif name == "accept_fleet_draw":
                self.has_accepted_draw = True
                decision.accepted_draw = True

            elif name == "reject_fleet_draw":
                decision.rejected_draw = True

        return decision


def settle_draw_offers(alpha: Optional["LLMAdmiral"], beta: Optional["LLMAdmiral"],
                       now: float) -> List[Tuple[str, str, str]]:
    """
    Settle this checkpoint's draw traffic between two LLM admirals and say
    what to record, as (sender faction, kind, text) diplomacy messages.

    - propose: recorded once per offer (DRAW_PROPOSAL); the offer stays open.
    - accept with the other side's offer open: DRAW_ACCEPT (the runner's
      _fleet_draw_agreed then ends the battle). Accept with nothing open is
      dropped and the acceptor told - a stale accept must not turn a later
      proposal into an instant draw nobody saw.
    - reject with an offer open: DRAW_REJECT, the offer is CLEARED and the
      proposer told at its next checkpoint (it used to stay open forever and
      the proposer never heard).
    Only decisions taken at ``now`` count.
    """
    out: List[Tuple[str, str, str]] = []
    if alpha is None or beta is None:
        return out
    pairs = (("alpha", alpha, beta), ("beta", beta, alpha))
    decided = {}
    for faction, me, _ in pairs:
        d = getattr(me, "last_decision", None)
        decided[faction] = d if d is not None and abs(getattr(d, "timestamp", -1.0) - now) < 1e-6 else None
    for faction, me, other in pairs:
        d = decided[faction]
        if d is None:
            continue
        if d.proposed_draw and me.draw_offer_t is not None and abs(me.draw_offer_t - now) < 1e-6:
            out.append((faction, "DRAW_PROPOSAL", f"{me.name} proposes a fleet draw."))
        if d.accepted_draw:
            if other.has_proposed_draw:
                out.append((faction, "DRAW_ACCEPT", f"{me.name} accepts the fleet draw."))
            else:
                me.has_accepted_draw = False
                me.draw_notices.append(f"T+{now:.0f}s: you accepted a draw, but no enemy offer "
                                       "was open - nothing happened.")
        if d.rejected_draw and other.has_proposed_draw:
            offered = other.draw_offer_t
            out.append((faction, "DRAW_REJECT", f"{me.name} rejects the fleet draw."))
            other.has_proposed_draw = False
            other.draw_offer_t = None
            other.draw_notices.append(
                f"T+{now:.0f}s: the enemy admiral REJECTED your draw proposal"
                + (f" of T+{offered:.0f}s" if offered is not None else "")
                + ". The offer is closed; propose again to re-open it.")
    return out
