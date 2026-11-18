"""
Admiral vision: the tactical plot a multimodal admiral sees each checkpoint.

The admiral's text turn state carries every number; the picture carries the
geometry that text cannot: where both fleets are and how they are arranged,
which way every hull points (nose / flank / tail toward the enemy), where each
ship will be at the next checkpoint, who is inside whose gun band and PD
umbrella, which torpedo waves are flying onto whom and when they land, what
changed since the last checkpoint, and the height dimension.

One fixed 1280 x 720 PNG (about 1,230 image tokens at width x height / 750;
below every model's downscale cap, so the model sees it pixel for pixel):

- header line: time, checkpoint N of M, which colour is you, torpedo counts,
  score (only when the admiral's score flag is on);
- PLAN (top-down, world X right, world Y up, the same axes the text's
  headings use): ships as oriented dart glyphs sized by class, filled from the
  tail in proportion to hull integrity, a torch flame when the drive burns;
  a thin line to a hollow ghost where the ship will be at the next checkpoint
  if it coasts (yellow ring: that ghost is within the 150 km gun band of an
  enemy ghost); dead ships as crosses; each fleet's gun band (150 km, solid)
  and PD umbrella (dotted) as the outline of the union of per-ship circles;
  torpedo waves aggregated into one arrow per wave with count, target and
  time to first impact; the fleets' coasting closest approach; full ship
  names (exactly the names in the text and in issue_order) placed by a
  collision-avoiding layout with leader lines, never over a ship;
- ZOOM inset in a free corner when ships of both fleets are too close for the
  plan to separate (merge range);
- SIDE VIEW strip under the plan: world X (same scale and columns as the
  plan) against height Z, never vertically exaggerated.

Framing is stateless (same state, same picture): the plan width is the
smallest step of a fixed ladder that holds every live ship, ghost and
threatening torpedo, the centre snaps to an eighth of that width.

Rendering is deterministic (no wall clock, bundled DejaVu fonts, PNG written
without a version stamp) and never raises into the battle: the runner treats
a failed render as text-only.

Everything degrades silently to text-only:
- matplotlib missing (it lives behind the `viz` optional dependency group)
- the admiral's model is not vision-capable
- the admiral config has vision disabled (the default)
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vision capability
# ---------------------------------------------------------------------------

# Model-id substrings (matched against the id lowercased, provider prefix and
# all) that indicate a multimodal model. Conservative: a text-only model that
# receives an image part gets a hard API error mid-battle, so unknown families
# default to text-only.
_VISION_SUBSTRINGS = (
    "claude",        # Anthropic: all current Claude models accept images
    "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5",   # OpenAI multimodal line
    "gemini",        # Google: all Gemini models are multimodal
    "grok-2-vision", "grok-4", "grok-3",           # xAI vision-capable
    "pixtral",       # Mistral vision line
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen3.8-max",
    "kimi-k3",       # Moonshot's K3 is natively multimodal
    "llama-3.2-11b-vision", "llama-3.2-90b-vision", "llama-4",
    "-vl-",          # generic vision-language marker used by several vendors
)


def is_vision_model(model: str) -> bool:
    """True if the model id looks like a multimodal (image-accepting) model."""
    if not model:
        return False
    lowered = model.lower()
    return any(s in lowered for s in _VISION_SUBSTRINGS)


def png_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG bytes as a data URL usable in an image_url content part."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_W, CANVAS_H, DPI = 1280, 720, 100
# Regions in pixels (x0, y0, x1, y1), origin top-left.
HEADER_H = 26
MAP_RECT = (8.0, 30.0, 1272.0, 562.0)
STRIP_RECT = (8.0, 572.0, 1272.0, 712.0)

OWN_COLOR = "#2ad4ff"
ENEMY_COLOR = "#ff7a45"
BG = "#0a0d14"
PANEL_BG = "#0d121b"
GRID = "#1a2a3a"
GRID_TEXT = "#4f6478"
TEXT = "#d4dee9"
DIM = "#7f8fa0"
WARN = "#ffd84a"
FONT = "DejaVu Sans"
LABEL_PT = 7.6
SMALL_PT = 6.6
HEADER_PT = 9.0

# Effective gun band (km): "beyond ~150 km a dodger is rarely hit" and the
# measured share of hull damage inside 150 km (doctrine, prompts.py GUNNERY /
# GUN EXCHANGE). The guns' nominal range_km (900) is not a useful envelope.
GUN_BAND_KM = 150.0


def gun_band_label() -> str:
    """The plot's key for the solid envelope (hit rates bound to the GUN
    EXCHANGE fixture through prompts.measured_hit_rate)."""
    try:
        from .prompts import GUN_EXCHANGE_BANDS, measured_hit_rate
    except ImportError:  # pragma: no cover
        from llm.prompts import GUN_EXCHANGE_BANDS, measured_hit_rate
    band, edge = measured_hit_rate(GUN_BAND_KM - 1.0)
    far_band, far = GUN_EXCHANGE_BANDS[-1][0], GUN_EXCHANGE_BANDS[-1][1]
    return (f"solid: gun band {GUN_BAND_KM:.0f} km (hits {edge}% at {band} km, {far}% at "
            f"{far_band} km; not the nominal reach) | dotted: PD")
DEFAULT_PD_RANGE_KM = 250.0
# Below this applied throttle the torch is PADLOCK's token burn: coasting
# (fleet_intel.BURNING_MIN_THROTTLE).
BURNING_MIN_THROTTLE = 0.1
# Plan labels carry the height above / below the viewer's fleet median from
# this many km on (the plan alone hides it).
HEIGHT_TAG_KM = 50.0
# Coasting closest approach of the two fleets' centres is marked when it lies
# ahead within this horizon.
CPA_HORIZON_S = 600.0
# Ships of one side within this distance of each other launch as one group:
# a torpedo wave is one group's rounds on one target.
SOURCE_LINK_KM = 150.0

# Glyph length in pixels by hull class (nose to tail); width is 0.64 x length.
GLYPH_LEN_PX = {
    "corvette": 15.0, "frigate": 17.0, "destroyer": 20.0, "cruiser": 23.0,
    "cruiser_torpedo": 23.0, "battlecruiser": 25.0, "battleship": 28.0,
    "dreadnought": 31.0, "dreadnought_siege": 31.0,
}
DEFAULT_GLYPH_LEN_PX = 20.0
# Dart outline in hull units (nose at +x, length 1): nose, port tail,
# tail notch, starboard tail.
_DART = ((0.5, 0.0), (-0.5, 0.32), (-0.28, 0.0), (-0.5, -0.32))

# Plan width ladder (km). The plan's height is width x MAP aspect.
WIDTH_LADDER_KM = tuple(m * 10.0 ** k for k in range(1, 6)
                        for m in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0))
# Labels need room: content is kept this many pixels (x, y) off the edges.
FRAME_MARGIN_PX = (100.0, 30.0)
# Ships whose glyph centres are closer than this (px) cannot be told apart.
CROWD_PX = 24.0
INSET_W, INSET_H = 400.0, 225.0
INSET_MIN_ZOOM = 2.0

_FLEET_DATA: Optional[Dict[str, Any]] = None


def _fleet_data() -> Dict[str, Any]:
    global _FLEET_DATA
    if _FLEET_DATA is None:
        path = Path(__file__).resolve().parents[2] / "data" / "fleet_ships.json"
        try:
            _FLEET_DATA = json.loads(path.read_text())
        except (OSError, ValueError):
            _FLEET_DATA = {}
    return _FLEET_DATA


def class_kit(ship_type: str) -> Dict[str, Any]:
    """{'guns': bool, 'pd': bool, 'pd_km': float} for a hull class (fleet data)."""
    spec = ((_fleet_data().get("ships") or {}).get(ship_type or "") or {})
    wtypes = _fleet_data().get("weapon_types") or {}
    guns = pd = False
    pd_km = DEFAULT_PD_RANGE_KM
    for w in spec.get("weapons", []):
        wtype = str(w.get("type", ""))
        if wtype.startswith("pd_"):
            pd = True
            pd_km = float((wtypes.get(wtype) or {}).get("range_km", pd_km) or pd_km)
        elif not wtype.startswith("torpedo"):
            guns = True
    if not spec:  # unknown class: assume a gun ship with PD
        guns = pd = True
    return {"guns": guns, "pd": pd, "pd_km": pd_km}


# ---------------------------------------------------------------------------
# Frames: live simulation and recording trace share one schema
# ---------------------------------------------------------------------------

def build_live_frame(simulation: Any) -> Dict[str, Any]:
    """
    Assemble the renderer's frame dict straight from a live CombatSimulation.

    Field names and units match the recording sim_trace format exactly
    (`pos`/`vel`/`fwd` in meters, `hull` percent, torpedo `dv_remaining` in
    km/s), so the same renderer works on live state and on replays. Torpedo
    rounds are annotated with their state, the hull they will strike and the
    time to impact (annotate_torpedoes).
    """
    frame: Dict[str, Any] = {
        "t": simulation.current_time,
        "ships": {},
        "projectiles": [],
        "torpedoes": [],
    }

    for ship_id, ship in simulation.ships.items():
        if not ship:
            continue
        maneuver = "MAINTAIN"
        thrust = 0.0
        if ship.current_maneuver and not ship.is_destroyed:
            maneuver = ship.current_maneuver.maneuver_type.name
            # What the drive PRODUCES (engine damage, interlock, evasion and
            # PADLOCK's token burn included), not the order's throttle.
            thrust = float(getattr(ship, "applied_throttle", ship.current_maneuver.throttle) or 0.0)
        frame["ships"][ship_id] = {
            "pos": [ship.position.x, ship.position.y, ship.position.z],
            "vel": [ship.velocity.x, ship.velocity.y, ship.velocity.z],
            "fwd": [ship.forward.x, ship.forward.y, ship.forward.z],
            "thrust": thrust,
            "maneuver": maneuver,
            "destroyed": ship.is_destroyed,
            "dying": bool(getattr(ship, "is_dying", False)),
            "hull": getattr(ship, "hull_integrity", 100.0),
        }

    for proj_flight in simulation.projectiles:
        proj = proj_flight.projectile
        frame["projectiles"].append({
            "id": proj_flight.projectile_id,
            "pos": [proj.position.x, proj.position.y, proj.position.z],
            "source": proj_flight.source_ship_id,
        })

    masses = {}
    specs = None
    for torp_flight in simulation.torpedoes:
        torp = torp_flight.torpedo
        specs = specs or getattr(torp, "specs", None)
        masses[torp_flight.torpedo_id] = float(getattr(torp, "current_mass_kg", 0.0) or 0.0) or None
        frame["torpedoes"].append({
            "id": torp_flight.torpedo_id,
            "pos": [torp.position.x, torp.position.y, torp.position.z],
            "vel": [torp.velocity.x, torp.velocity.y, torp.velocity.z],
            "source": torp_flight.source_ship_id,
            "target": torp.target_id,
            "disabled": torp_flight.is_disabled,
            "dv_remaining": float(getattr(torp, "remaining_delta_v_kps", 0.0) or 0.0),
        })
    annotate_torpedoes(frame, specs=specs, masses=masses)
    return frame


def _torpedo_specs(specs: Any = None) -> Any:
    if specs is not None:
        return specs
    try:
        from .fleet_intel import torpedo_specs_for
        return torpedo_specs_for(None, _fleet_data())
    except Exception:  # noqa: BLE001 - advisory only
        return None


def _vec(v: Sequence[float]):
    from ..physics import Vector3D
    return Vector3D(float(v[0]), float(v[1]), float(v[2]))


def annotate_torpedoes(frame: Dict[str, Any], specs: Any = None,
                       masses: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, Any]:
    """
    Add 'state' (guided | blinded | orphaned | dry), 'threat' (the hull it will
    strike, None for a wreck on nobody's line) and 'eta' (s) to every torpedo
    of ``frame`` - the text turn state's own model (fleet_intel.gather_rounds):
    a guided round flies the guidance law's pursuit (solve_pursuit) against a
    target that holds course; a ballistic one (blinded, dry, orphaned) strikes
    the first hull its straight line passes within the engine's miss distance.
    ``masses`` (torpedo id -> kg) come from the live sim; from a trace the mass
    follows from the delta-v left (rocket equation).
    """
    try:
        from . import fleet_intel as intel
    except ImportError:  # pragma: no cover - script-style imports
        import llm.fleet_intel as intel  # type: ignore
    specs = _torpedo_specs(specs)
    ships = frame.get("ships", {})
    live = [SimpleNamespace(ship_id=sid, position=_vec(s["pos"]), velocity=_vec(s["vel"]))
            for sid, s in sorted(ships.items())
            if not s.get("destroyed") and not s.get("dying")]
    for tp in frame.get("torpedoes", []):
        target = ships.get(tp.get("target"))
        dv_kps = float(tp.get("dv_remaining", 0.0) or 0.0)
        if tp.get("disabled"):
            state = "blinded"
        elif target is None or target.get("destroyed") or target.get("dying"):
            state = "orphaned"
        elif "dv_remaining" in tp and dv_kps <= 1e-6:
            state = "dry"
        else:
            state = "guided"
        tp["state"], tp["threat"], tp["eta"] = state, None, None
        if not tp.get("vel"):
            if state == "guided":
                tp["threat"] = tp.get("target")
            continue
        pos, vel = _vec(tp["pos"]), _vec(tp["vel"])
        if state == "guided":
            rel_pos = pos - _vec(target["pos"])
            rel_vel = vel - _vec(target["vel"])
            dist = rel_pos.magnitude
            closing = -rel_pos.dot(rel_vel) / dist if dist > 0 else 0.0
            eta = dist / closing if closing > 1.0 else None
            if specs is not None:
                mass = (masses or {}).get(tp.get("id"))
                if mass is None:
                    ve = max(1.0, float(specs.exhaust_velocity_kps) * 1000.0)
                    mass = min(float(specs.mass_kg),
                               float(specs.dry_mass_kg) * math.exp(dv_kps * 1000.0 / ve))
                try:
                    fly = intel.solve_pursuit(dist, closing, 0.0, 0.0, specs, mass_kg=mass,
                                              dv_left_mps=dv_kps * 1000.0)
                    if fly.get("hit"):
                        eta = fly["tof_s"]
                except Exception:  # noqa: BLE001 - advisory only
                    pass
            tp["threat"], tp["eta"] = tp.get("target"), eta
        else:
            hit = intel.ballistic_threat(pos, vel, live)
            if hit is not None:
                tp["threat"], tp["eta"] = hit[0].ship_id, hit[1]
    return frame


def frame_from_trace(trace_frame: Dict[str, Any]) -> Dict[str, Any]:
    """A recorded sim_trace frame in the live frame's form (torpedoes annotated)."""
    frame = {
        "t": trace_frame.get("t", 0.0),
        "ships": {sid: dict(s) for sid, s in (trace_frame.get("ships") or {}).items()},
        "projectiles": list(trace_frame.get("projectiles") or []),
        "torpedoes": [dict(tp) for tp in (trace_frame.get("torpedoes") or [])],
    }
    return annotate_torpedoes(frame)


# ---------------------------------------------------------------------------
# Scene model (pure: no matplotlib)
# ---------------------------------------------------------------------------

def _faction_of(entity_id: str, ships_meta: Dict[str, Dict[str, str]]) -> str:
    meta = ships_meta.get(str(entity_id))
    if meta and meta.get("faction"):
        return meta["faction"]
    return "alpha" if str(entity_id).startswith("alpha") else "beta"


def face_tag(fwd: Sequence[float], pos: Sequence[float], toward: Sequence[float]) -> str:
    """NOSE / SIDE / TAIL that a hull at ``pos`` pointing ``fwd`` shows the
    point ``toward`` - the engine's hit cones (fleet_intel.face_of)."""
    los = [toward[i] - pos[i] for i in range(3)]
    nl = math.sqrt(sum(c * c for c in los))
    nf = math.sqrt(sum(c * c for c in fwd))
    if nl < 1e-6 or nf < 1e-9:
        return "NOSE"
    cos = max(-1.0, min(1.0, sum(fwd[i] * los[i] for i in range(3)) / (nl * nf)))
    angle = math.degrees(math.acos(cos))
    try:
        from ..geometry import NOSE_HIT_ANGLE_THRESHOLD as nose, TAIL_HIT_ANGLE_THRESHOLD as tail
    except ImportError:  # pragma: no cover - script-style imports
        nose, tail = 30.0, 150.0
    return "NOSE" if angle < nose else ("TAIL" if angle > tail else "SIDE")


@dataclass
class ShipItem:
    sid: str
    name: str
    ship_type: str
    own: bool
    alive: bool
    dead_new: bool            # destroyed / dying since the previous checkpoint
    pos: Tuple[float, float, float]          # km
    vel: Tuple[float, float, float]          # km/s
    fwd: Tuple[float, float, float]
    hull: float               # percent
    burning: bool
    ghost: Tuple[float, float, float]        # km, coasting to the next checkpoint
    face: str = "NOSE"        # face shown to the other fleet's centre
    dz: float = 0.0           # km above (+) / below the viewer's fleet median height
    ghost_in_band: bool = False
    guns: bool = True
    pd: bool = True
    pd_km: float = DEFAULT_PD_RANGE_KM

    @property
    def glyph_len(self) -> float:
        return GLYPH_LEN_PX.get(self.ship_type, DEFAULT_GLYPH_LEN_PX)

    @property
    def keep_out(self) -> float:
        """Label keep-out radius (px) around the glyph, its torch flame included."""
        return self.glyph_len * (1.0 if self.burning else 0.5) + 2.0

    def label(self) -> str:
        if not self.alive:
            return f"{self.name} DESTROYED"
        text = self.name
        if self.hull < 99.5:
            text += f" {self.hull:.0f}%"
        if self.face != "NOSE":
            text += f" {self.face}"
        if abs(self.dz) >= HEIGHT_TAG_KM:
            text += f" Z{self.dz:+.0f}"
        return text


@dataclass
class Wave:
    own: bool                 # launched by the viewer's fleet
    target: Optional[str]
    target_name: str
    n_live: int
    n_blind: int             # unguided: blinded seeker, dry, or target lost
    eta: Optional[float]      # first impact (s)
    new: bool                 # a round launched since the previous checkpoint
    rounds: List[Tuple[float, float, float, bool]]   # km xyz, blind
    centroid: Tuple[float, float, float]
    tail: Tuple[float, float, float]          # the round furthest from the target

    def label(self) -> str:
        if self.n_live:
            text = f"{self.n_live} -> {self.target_name}"
            if self.n_blind:
                text += f" +{self.n_blind} unguided"
        else:
            text = f"{self.n_blind} unguided -> {self.target_name}"
        if self.eta is not None:
            text += f" {self.eta:.0f}s"
        if self.new:
            text += " NEW"
        return text


@dataclass
class Scene:
    t: float
    faction: str
    ships: List[ShipItem]
    waves: List[Wave]
    spent_own: int            # dead rounds on nobody's line (not drawn)
    spent_enemy: int
    torps_own: Tuple[int, int]      # (live, blind) in flight
    torps_enemy: Tuple[int, int]
    cpa: Optional[Dict[str, Any]]   # own/enemy centre at closest approach
    header: str
    interval_s: float


def _km(v: Sequence[float]) -> Tuple[float, float, float]:
    return (float(v[0]) / 1000.0, float(v[1]) / 1000.0, float(v[2]) / 1000.0)


def _mean(points: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    n = max(1, len(points))
    return tuple(sum(p[i] for p in points) / n for i in range(len(points[0]))) if points else ()


def build_scene(
    frame: Dict[str, Any],
    ships_meta: Dict[str, Dict[str, str]],
    faction: Optional[str] = None,
    prev_frame: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[int] = None,
    total_checkpoints: Optional[int] = None,
    score_text: Optional[str] = None,
    interval_s: float = 30.0,
) -> Scene:
    """Everything the plot shows, computed from one frame (and the previous
    checkpoint's frame for what changed). Pure; no matplotlib."""
    faction = faction or "alpha"
    ships_meta = ships_meta or {}
    t = float(frame.get("t", 0.0))
    prev_ships = (prev_frame or {}).get("ships") or {}
    items: List[ShipItem] = []
    for sid in sorted(frame.get("ships", {})):
        s = frame["ships"][sid]
        meta = ships_meta.get(sid, {})
        dead = bool(s.get("destroyed") or s.get("dying"))
        was = prev_ships.get(sid)
        dead_new = dead and (prev_frame is None or was is None
                             or not (was.get("destroyed") or was.get("dying")))
        if prev_frame is None and dead:
            dead_new = False
        pos, vel = _km(s["pos"]), _km(s.get("vel") or (0.0, 0.0, 0.0))
        kit = class_kit(meta.get("type", ""))
        items.append(ShipItem(
            sid=sid, name=str(meta.get("name") or sid), ship_type=meta.get("type", ""),
            own=_faction_of(sid, ships_meta) == faction, alive=not dead, dead_new=dead_new,
            pos=pos, vel=vel, fwd=tuple(s.get("fwd") or (1.0, 0.0, 0.0)),
            hull=float(s.get("hull", 100.0) or 0.0),
            burning=(not dead) and float(s.get("thrust", 0.0) or 0.0) >= BURNING_MIN_THROTTLE,
            ghost=tuple(pos[i] + vel[i] * interval_s for i in range(3)),
            guns=kit["guns"], pd=kit["pd"], pd_km=kit["pd_km"]))

    own_live = [it for it in items if it.own and it.alive]
    en_live = [it for it in items if not it.own and it.alive]
    own_c = _mean([it.pos for it in own_live]) if own_live else None
    en_c = _mean([it.pos for it in en_live]) if en_live else None
    zs = sorted(it.pos[2] for it in (own_live or [it for it in items if it.own] or items))
    z_ref = zs[len(zs) // 2] if zs else 0.0
    for it in items:
        it.dz = it.pos[2] - z_ref
        if not it.alive:
            continue
        other = en_c if it.own else own_c
        if other:
            it.face = face_tag(it.fwd, it.pos, other)
        foes = en_live if it.own else own_live
        it.ghost_in_band = any(math.dist(it.ghost, f.ghost) <= GUN_BAND_KM
                               for f in foes if f.guns or it.guns)

    # Coasting closest approach of the two fleet centres.
    cpa = None
    if own_live and en_live:
        ov = _mean([it.vel for it in own_live])
        ev = _mean([it.vel for it in en_live])
        dp = [en_c[i] - own_c[i] for i in range(3)]
        dv = [ev[i] - ov[i] for i in range(3)]
        vv = sum(c * c for c in dv)
        if vv > 1e-6:
            tca = -sum(dp[i] * dv[i] for i in range(3)) / vv
            if 1.0 < tca <= CPA_HORIZON_S:
                miss = math.sqrt(sum((dp[i] + dv[i] * tca) ** 2 for i in range(3)))
                cpa = {"t": tca, "km": miss,
                       "own": tuple(own_c[i] + ov[i] * tca for i in range(3)),
                       "enemy": tuple(en_c[i] + ev[i] * tca for i in range(3))}

    # Torpedo rounds by side and the hull each will strike; dead rounds on
    # nobody's line are only counted.
    names = {it.sid: it.name for it in items}
    prev_ids = {tp.get("id") for tp in (prev_frame or {}).get("torpedoes", [])}
    groups: Dict[Tuple[bool, Optional[str]], List[Dict[str, Any]]] = {}
    counts = {True: [0, 0], False: [0, 0]}
    spent = {True: 0, False: 0}
    for tp in sorted(frame.get("torpedoes", []), key=lambda r: str(r.get("id"))):
        own = _faction_of(tp.get("source", ""), ships_meta) == faction
        blind = tp.get("state", "blinded" if tp.get("disabled") else "guided") != "guided"
        counts[own][1 if blind else 0] += 1
        threat = tp.get("threat", tp.get("target"))
        if blind and threat is None:
            spent[own] += 1
            continue
        groups.setdefault((own, threat), []).append(tp)
    # Launching groups: each side's ships (alive or not) linked within
    # SOURCE_LINK_KM; a wave is one group's rounds on one threatened hull.
    src_group: Dict[str, int] = {}
    for side_own in (True, False):
        side = [it for it in items if it.own == side_own]
        parent = list(range(len(side)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(side)):
            for j in range(i + 1, len(side)):
                if math.dist(side[i].pos, side[j].pos) <= SOURCE_LINK_KM:
                    parent[find(i)] = find(j)
        for i, it in enumerate(side):
            src_group[it.sid] = find(i)
    by_wave: Dict[Tuple[bool, int, str], List[Dict[str, Any]]] = {}
    for (own, threat), rounds in groups.items():
        for tp in rounds:
            key = (own, src_group.get(tp.get("source"), -1), str(threat))
            by_wave.setdefault(key, []).append(tp)
    by_id = {it.sid: it for it in items}
    waves: List[Wave] = []
    for (own, _grp, threat), rs in sorted(by_wave.items(), key=lambda kv: (not kv[0][0], kv[0][2], kv[0][1])):
        pts = [_km(tp["pos"]) for tp in rs]
        blind_flags = [r.get("state", "guided") != "guided" for r in rs]
        etas = [r["eta"] for r in rs if r.get("eta") is not None]
        tgt = by_id.get(threat)
        if tgt is not None:
            tail = max(pts, key=lambda p: math.dist(p, tgt.pos))
        else:
            tail = pts[0]
        waves.append(Wave(
            own=own, target=threat, target_name=names.get(threat, str(threat)),
            n_live=sum(1 for b in blind_flags if not b), n_blind=sum(blind_flags),
            eta=min(etas) if etas else None,
            new=prev_frame is not None and any(r.get("id") not in prev_ids for r in rs),
            rounds=[(*pts[k], blind_flags[k]) for k in range(len(rs))],
            centroid=_mean(pts), tail=tail))

    header = _header(t, faction, checkpoint, total_checkpoints, counts, score_text)
    return Scene(t=t, faction=faction, ships=items, waves=waves,
                 spent_own=spent[True], spent_enemy=spent[False],
                 torps_own=tuple(counts[True]), torps_enemy=tuple(counts[False]),
                 cpa=cpa, header=header, interval_s=interval_s)


def _header(t: float, faction: str, checkpoint: Optional[int], total: Optional[int],
            counts: Dict[bool, List[int]], score_text: Optional[str]) -> str:
    parts = [f"T+{t:.0f}s"]
    if checkpoint:
        parts.append(f"checkpoint {checkpoint}" + (f" of {total}" if total else ""))
    enemy = "beta" if faction == "alpha" else "alpha"
    parts.append(f"YOU {faction.upper()} = cyan, {enemy.upper()} = orange")

    def torps(c: List[int]) -> str:
        return f"{c[0]}" + (f" (+{c[1]} unguided)" if c[1] else "")
    parts.append(f"torpedoes in flight: yours {torps(counts[True])}, theirs {torps(counts[False])}")
    if score_text:
        parts.append(score_text)
    return "   |   ".join(parts)


# ---------------------------------------------------------------------------
# Framing (pure, stateless)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class View:
    """World (km) -> pixel mapping of one panel."""
    cx: float
    cy: float
    width_km: float
    rect: Tuple[float, float, float, float]

    @property
    def px_per_km(self) -> float:
        return (self.rect[2] - self.rect[0]) / self.width_km

    @property
    def height_km(self) -> float:
        return (self.rect[3] - self.rect[1]) / self.px_per_km

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        s = self.px_per_km
        mx = (self.rect[0] + self.rect[2]) / 2.0
        my = (self.rect[1] + self.rect[3]) / 2.0
        return mx + (x - self.cx) * s, my - (y - self.cy) * s

    def contains_px(self, px: float, py: float, pad: float = 0.0) -> bool:
        return (self.rect[0] + pad <= px <= self.rect[2] - pad
                and self.rect[1] + pad <= py <= self.rect[3] - pad)


def choose_view(points: Sequence[Tuple[float, float]],
                rect: Tuple[float, float, float, float] = MAP_RECT) -> View:
    """
    The framing rule: the smallest width on WIDTH_LADDER_KM whose frame, centred
    on the content's midpoint snapped to a twentieth of that width, holds every
    point FRAME_MARGIN_PX inside its edges (room for labels). Stateless: the
    same content always gives the same frame, so consecutive checkpoints share
    a frame until the content outgrows (or shrinks below) a ladder step.
    """
    if not points:
        return View(0.0, 0.0, 1000.0, rect)
    for width in WIDTH_LADDER_KM:
        view = frame_at(points, width, rect)
        if view is not None:
            return view
    width = WIDTH_LADDER_KM[-1]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    step = width / 20.0
    return View(round((min(xs) + max(xs)) / 2.0 / step) * step,
                round((min(ys) + max(ys)) / 2.0 / step) * step, width, rect)


def frame_at(points: Sequence[Tuple[float, float]], width: float,
             rect: Tuple[float, float, float, float] = MAP_RECT) -> Optional[View]:
    """The frame of ``width`` (centre snapped to width / 20) if it holds every
    point FRAME_MARGIN_PX inside its edges, else None."""
    w_px, h_px = rect[2] - rect[0], rect[3] - rect[1]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    step = width / 20.0
    cx = round((min(xs) + max(xs)) / 2.0 / step) * step
    cy = round((min(ys) + max(ys)) / 2.0 / step) * step
    kmpp = width / w_px
    half_w = (w_px / 2.0 - FRAME_MARGIN_PX[0]) * kmpp
    half_h = (h_px / 2.0 - FRAME_MARGIN_PX[1]) * kmpp
    if all(abs(x - cx) <= half_w and abs(y - cy) <= half_h for x, y in zip(xs, ys)):
        return View(cx, cy, width, rect)
    return None


def frame_points(scene: Scene) -> List[Tuple[float, float]]:
    """What the plan must show: live ships and their ghosts, ships lost since
    the last checkpoint, and every round that threatens a hull."""
    pts: List[Tuple[float, float]] = []
    for it in scene.ships:
        if it.alive:
            pts.append(it.pos[:2])
            pts.append(it.ghost[:2])
        elif it.dead_new:
            pts.append(it.pos[:2])
    for w in scene.waves:
        pts.extend((r[0], r[1]) for r in w.rounds)
    if not pts:  # nothing alive: frame the wrecks
        pts = [it.pos[:2] for it in scene.ships]
    return pts


def nice_step(span: float, target_lines: float = 8.0) -> float:
    raw = max(span / target_lines, 1e-6)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def strip_scale(scene: Scene, view: View,
                rect: Tuple[float, float, float, float] = STRIP_RECT) -> Tuple[float, float]:
    """(z centre km, vertical km per px) of the side view: the plan's own
    scale unless the height spread does not fit, then compressed by 2, 4, ...
    (never exaggerated). The centre snaps like the plan's."""
    zs = [it.pos[2] for it in scene.ships if it.alive or it.dead_new]
    zs += [r[2] for w in scene.waves for r in w.rounds]
    kmpp = 1.0 / view.px_per_km
    if not zs:
        return 0.0, kmpp
    lo, hi = min(zs), max(zs)
    usable = (rect[3] - rect[1]) - 24.0
    factor = 1
    while (hi - lo) / (kmpp * factor) > usable and factor < 4096:
        factor *= 2
    step = max(kmpp * factor * 10.0, 1e-6)
    return round((lo + hi) / 2.0 / step) * step, kmpp * factor


# ---------------------------------------------------------------------------
# Inset and label layout (pure)
# ---------------------------------------------------------------------------

Rect = Tuple[float, float, float, float]   # x0, y0, x1, y1 (px, y down)


def rects_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def rect_circle_overlap(r: Rect, cx: float, cy: float, rad: float) -> bool:
    nx = min(max(cx, r[0]), r[2])
    ny = min(max(cy, r[1]), r[3])
    return (nx - cx) ** 2 + (ny - cy) ** 2 < rad * rad


@dataclass
class Inset:
    view: View                 # the zoomed world window drawn into rect
    source: Rect               # the zoomed region on the plan (px)
    ships: List[str]           # ship ids labelled in the inset, not on the plan


def choose_inset(scene: Scene, view: View) -> Optional[Inset]:
    """
    A zoom of the merge when ships of BOTH fleets sit closer than CROWD_PX on
    the plan (their glyphs would overlap). The window covers every crowded
    ship, zoomed at least INSET_MIN_ZOOM, in the plan corner (or top / bottom middle)
    covering the fewest glyphs. Pure function of the scene and the plan's view.
    """
    live = [it for it in scene.ships if it.alive]
    px = {it.sid: view.to_px(*it.pos[:2]) for it in live}
    crowded = set()
    mixed = False
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if math.dist(px[a.sid], px[b.sid]) < CROWD_PX:
                crowded.update((a.sid, b.sid))
                mixed = mixed or a.own != b.own
    if not mixed:
        return None
    # pull in close neighbours so the zoomed window shows the local picture
    changed = True
    while changed:
        changed = False
        for it in live:
            if it.sid not in crowded and any(math.dist(px[it.sid], px[c]) < 2 * CROWD_PX for c in crowded):
                crowded.add(it.sid)
                changed = True
    members = [it for it in live if it.sid in crowded]
    xs = [it.pos[0] for it in members]
    ys = [it.pos[1] for it in members]
    span_x = max(max(xs) - min(xs), 1e-3)
    span_y = max(max(ys) - min(ys), 1e-3)
    width_km = max(span_x / 0.70, span_y / 0.62 * INSET_W / INSET_H)
    zoom = view.width_km / width_km * INSET_W / (view.rect[2] - view.rect[0])
    if zoom < INSET_MIN_ZOOM:
        width_km = view.width_km * INSET_W / (view.rect[2] - view.rect[0]) / INSET_MIN_ZOOM
    best = None
    x0, y0, x1, y1 = view.rect
    mid = (x0 + x1 - INSET_W) / 2.0
    corners = [(x0 + 6, y0 + 6), (x1 - 6 - INSET_W, y0 + 6),
               (x0 + 6, y1 - 6 - INSET_H), (x1 - 6 - INSET_W, y1 - 6 - INSET_H),
               (mid, y0 + 6), (mid, y1 - 6 - INSET_H)]
    glyphs = [(*view.to_px(*it.pos[:2]), it.glyph_len / 2.0 + 3.0)
              for it in scene.ships if it.alive or it.dead_new]
    others = [view.to_px(*it.ghost[:2]) for it in live]
    others += [view.to_px(r[0], r[1]) for w in scene.waves for r in w.rounds]
    for k, (cx0, cy0) in enumerate(corners):
        r = (cx0 - 3, cy0 - 3, cx0 + INSET_W + 3, cy0 + INSET_H + 3)
        cost = 100 * sum(1 for gx, gy, gr in glyphs if rect_circle_overlap(r, gx, gy, gr))
        cost += sum(1 for p in others if r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3])
        if best is None or cost < best[0]:
            best = (cost, k, (cx0, cy0))
    cx0, cy0 = best[2]
    rect = (cx0, cy0, cx0 + INSET_W, cy0 + INSET_H)
    mx, my = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    iv = View(mx, my, width_km, rect)
    sx0, sy0 = view.to_px(mx - width_km / 2.0, my + iv.height_km / 2.0)
    sx1, sy1 = view.to_px(mx + width_km / 2.0, my - iv.height_km / 2.0)
    return Inset(view=iv, source=(sx0, sy0, sx1, sy1), ships=sorted(crowded))


@dataclass
class LabelReq:
    key: str
    text: str
    anchor: Tuple[float, float]   # px point the label belongs to
    radius: float                 # keep-out radius around the anchor (glyph)
    color: str
    away: Tuple[float, float] = (0.0, 0.0)   # preferred direction (unit-ish)
    size_pt: float = LABEL_PT
    bold: bool = False


@dataclass
class LabelPlacement:
    key: str
    text: str
    rect: Rect
    text_xy: Tuple[float, float]   # left-top of the text (px)
    leader: Optional[Tuple[float, float, float, float]]
    color: str
    size_pt: float
    bold: bool


_TEXT_CACHE: Dict[Tuple[str, float, bool], Tuple[float, float]] = {}


def text_size_px(text: str, size_pt: float = LABEL_PT, bold: bool = False) -> Tuple[float, float]:
    """(width, height) in pixels of ``text`` in the plot font at DPI."""
    key = (text, size_pt, bold)
    hit = _TEXT_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextToPath
        prop = FontProperties(family=FONT, size=size_pt, weight="bold" if bold else "normal")
        w, _h, _d = TextToPath().get_text_width_height_descent(text, prop, ismath=False)
        out = (w * DPI / 72.0, size_pt * DPI / 72.0 * 1.18)
    except Exception:  # noqa: BLE001 - metrics fallback (no matplotlib)
        out = (len(text) * size_pt * DPI / 72.0 * 0.6, size_pt * DPI / 72.0 * 1.18)
    if len(_TEXT_CACHE) > 4096:
        _TEXT_CACHE.clear()
    _TEXT_CACHE[key] = out
    return out


_DIRS = [(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
         (0.7071, -0.7071), (-0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071)]
_RINGS = (3.0, 14.0, 28.0, 46.0, 68.0, 96.0, 130.0)
LABEL_PAD = 1.5


def place_labels(reqs: Sequence[LabelReq], bounds: Rect,
                 blocked_circles: Sequence[Tuple[float, float, float]] = (),
                 blocked_rects: Sequence[Rect] = (),
                 soft_rects: Sequence[Rect] = ()) -> List[LabelPlacement]:
    """
    Greedy label layout. Hard rules: a label stays inside ``bounds``, never
    overlaps a glyph (``blocked_circles`` plus every request's own keep-out
    circle), a blocked rectangle, or another label. Candidates sit in 8
    directions around the anchor on growing rings; beyond the first ring a
    leader line joins label and glyph. Preference: nearest ring, then the
    request's ``away`` direction, then fewest soft overlaps; failing all
    rings, the nearest legal spot anywhere in ``bounds``. Requests in crowded
    spots are placed first; ties break by key (deterministic). A request with
    no legal spot at all is dropped (never drawn over a ship).
    """
    circles = list(blocked_circles) + [(r.anchor[0], r.anchor[1], r.radius) for r in reqs]
    placed: List[LabelPlacement] = []
    taken: List[Rect] = list(blocked_rects)

    def crowd(r: LabelReq) -> int:
        return sum(1 for c in circles if math.dist(c[:2], r.anchor) < 60.0)
    order = sorted(reqs, key=lambda r: (-crowd(r), r.key))
    for req in order:
        w, h = text_size_px(req.text, req.size_pt, req.bold)
        w += 2 * LABEL_PAD
        h += 2 * LABEL_PAD
        best = None
        for ri, ring in enumerate(_RINGS):
            d = req.radius + ring
            for dx, dy in _DIRS:
                ax_, ay_ = req.anchor[0] + dx * d, req.anchor[1] + dy * d
                # anchor the rect's nearest side/corner to (ax_, ay_)
                x0 = ax_ if dx > 0.3 else (ax_ - w if dx < -0.3 else ax_ - w / 2.0)
                y0 = ay_ if dy > 0.3 else (ay_ - h if dy < -0.3 else ay_ - h / 2.0)
                rect = (x0, y0, x0 + w, y0 + h)
                if not (bounds[0] <= rect[0] and rect[2] <= bounds[2]
                        and bounds[1] <= rect[1] and rect[3] <= bounds[3]):
                    continue
                if any(rect_circle_overlap(rect, cx, cy, cr) for cx, cy, cr in circles):
                    continue
                if any(rects_overlap(rect, o) for o in taken):
                    continue
                soft = sum(1 for o in soft_rects if rects_overlap(rect, o))
                pref = -(dx * req.away[0] + dy * req.away[1])
                cost = ri * 10.0 + pref * 3.0 + soft * 8.0 + (4.0 if dx == 0 else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, rect, ri, (ax_, ay_))
            if best is not None and best[0] < (ri + 1) * 10.0 - 3.0:
                break
        if best is None:
            # last resort: the nearest legal spot anywhere in the panel, with a leader
            gx = bounds[0]
            while gx + w <= bounds[2]:
                gy = bounds[1]
                while gy + h <= bounds[3]:
                    rect = (gx, gy, gx + w, gy + h)
                    if not any(rect_circle_overlap(rect, cx, cy, cr) for cx, cy, cr in circles) \
                            and not any(rects_overlap(rect, o) for o in taken):
                        nx = min(max(req.anchor[0], rect[0]), rect[2])
                        ny = min(max(req.anchor[1], rect[1]), rect[3])
                        d = math.dist((nx, ny), req.anchor)
                        if best is None or d < best[0]:
                            best = (d, rect, len(_RINGS), (nx, ny))
                    gy += 6.0
                gx += 6.0
        if best is None:
            continue
        _, rect, ri, (ax_, ay_) = best
        leader = None
        if ri >= 1:
            ux, uy = ax_ - req.anchor[0], ay_ - req.anchor[1]
            n = math.hypot(ux, uy) or 1.0
            sx, sy = req.anchor[0] + ux / n * (req.radius + 1.0), req.anchor[1] + uy / n * (req.radius + 1.0)
            ex = min(max(sx, rect[0]), rect[2])
            ey = min(max(sy, rect[1]), rect[3])
            leader = (sx, sy, ex, ey)
        taken.append(rect)
        placed.append(LabelPlacement(key=req.key, text=req.text, rect=rect,
                                     text_xy=(rect[0] + LABEL_PAD, rect[1] + LABEL_PAD),
                                     leader=leader, color=req.color, size_pt=req.size_pt,
                                     bold=req.bold))
    return sorted(placed, key=lambda p: p.key)


# ---------------------------------------------------------------------------
# Layout: scene -> display list (pure)
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    scene: Scene
    view: View
    inset: Optional[Inset]
    strip_zc: float
    strip_kmpp: float
    labels: List[LabelPlacement] = field(default_factory=list)
    inset_labels: List[LabelPlacement] = field(default_factory=list)
    strip_labels: List[LabelPlacement] = field(default_factory=list)
    glyph_circles: List[Tuple[float, float, float]] = field(default_factory=list)
    wave_segments: List[Tuple[Wave, Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def _wave_segment(view: View, w: Wave, by_id: Dict[str, ShipItem]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    start = view.to_px(*w.tail[:2])
    tgt = by_id.get(w.target)
    if tgt is None:
        return start, start
    end = view.to_px(*tgt.pos[:2])
    ux, uy = _unit(end[0] - start[0], end[1] - start[1])
    stop = tgt.glyph_len / 2.0 + 5.0
    if math.dist(start, end) <= stop + 2.0:
        return start, start
    return start, (end[0] - ux * stop, end[1] - uy * stop)


def _segment_rects(a: Tuple[float, float], b: Tuple[float, float], n: int = 6, r: float = 3.0) -> List[Rect]:
    out = []
    for k in range(n + 1):
        x = a[0] + (b[0] - a[0]) * k / n
        y = a[1] + (b[1] - a[1]) * k / n
        out.append((x - r, y - r, x + r, y + r))
    return out


def layout_scene(scene: Scene) -> Layout:
    view = choose_view(frame_points(scene))
    inset = choose_inset(scene, view)
    zc, kmpp = strip_scale(scene, view)
    lay = Layout(scene=scene, view=view, inset=inset, strip_zc=zc, strip_kmpp=kmpp)
    by_id = {it.sid: it for it in scene.ships}
    in_inset = set()
    if inset:
        # exactly the ships inside the dashed box are labelled in the zoom,
        # every other ship on the plan
        sx0, sy0, sx1, sy1 = inset.source
        for it in scene.ships:
            if it.alive or it.dead_new:
                px, py = view.to_px(*it.pos[:2])
                if sx0 <= px <= sx1 and sy0 <= py <= sy1:
                    in_inset.add(it.sid)
        inset.ships = sorted(in_inset)
    own_live = [it for it in scene.ships if it.own and it.alive]
    en_live = [it for it in scene.ships if not it.own and it.alive]
    own_c = _mean([it.pos for it in own_live]) if own_live else None
    en_c = _mean([it.pos for it in en_live]) if en_live else None

    def away_dir(it: ShipItem, v: View) -> Tuple[float, float]:
        other = en_c if it.own else own_c
        if not other:
            return (0.0, -1.0)
        a = v.to_px(*it.pos[:2])
        b = v.to_px(*other[:2])
        ux, uy = _unit(a[0] - b[0], a[1] - b[1])
        return (ux, uy) if (ux or uy) else (0.0, -1.0)

    # glyph keep-out circles on the plan
    circles = []
    for it in scene.ships:
        if it.alive or it.dead_new:
            p = view.to_px(*it.pos[:2])
            if view.contains_px(*p):
                circles.append((p[0], p[1], it.keep_out))
    lay.glyph_circles = circles

    soft: List[Rect] = []
    for w in scene.waves:
        a, b = _wave_segment(view, w, by_id)
        lay.wave_segments.append((w, a, b))
        soft.extend(_segment_rects(a, b))
    for it in scene.ships:
        if it.alive:
            soft.extend(_segment_rects(view.to_px(*it.pos[:2]), view.to_px(*it.ghost[:2]), n=4, r=2.0))
    blocked = [inset_frame(inset)] if inset else []
    r = view.rect
    blocked += [(r[0], r[1], r[0] + 210, r[1] + 16),          # panel title
                (r[2] - 200, r[1], r[2], r[1] + 26),          # scale bar
                (r[0], r[3] - 14, r[2], r[3])]                # grid labels (X)
    if scene.cpa:
        for key in ("own", "enemy"):
            p = view.to_px(*scene.cpa[key][:2])
            blocked.append((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5))

    reqs: List[LabelReq] = []
    if scene.cpa:
        a = view.to_px(*scene.cpa["own"][:2])
        b = view.to_px(*scene.cpa["enemy"][:2])
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if view.contains_px(*mid):
            reqs.append(LabelReq(key="c:cpa", text=f"closest {scene.cpa['km']:.0f} km in "
                                 f"{scene.cpa['t']:.0f}s (coasting)", anchor=mid,
                                 radius=6.0, color=WARN, size_pt=SMALL_PT))
    for it in scene.ships:
        if not (it.alive or it.dead_new) or it.sid in in_inset:
            continue
        p = view.to_px(*it.pos[:2])
        if not view.contains_px(*p):
            continue
        reqs.append(LabelReq(key="s:" + it.sid, text=it.label(), anchor=p,
                             radius=it.keep_out,
                             color=OWN_COLOR if it.own else ENEMY_COLOR,
                             away=away_dir(it, view), bold=it.dead_new))
    for k, (w, a, b) in enumerate(lay.wave_segments):
        c = view.to_px(*w.centroid[:2])
        if not view.contains_px(*c):
            continue
        ux, uy = _unit(a[0] - b[0], a[1] - b[1])
        ux, uy = -uy, ux   # beside the stream, not along it
        reqs.append(LabelReq(key=f"w:{k:03d}", text=w.label(), anchor=c, radius=5.0,
                             color=OWN_COLOR if w.own else ENEMY_COLOR,
                             away=(ux, uy) if (ux or uy) else (0.0, -1.0), size_pt=LABEL_PT,
                             bold=w.new))
    bounds = (view.rect[0] + 2, view.rect[1] + 2, view.rect[2] - 2, view.rect[3] - 2)
    lay.labels = place_labels(reqs, bounds, blocked_circles=circles, blocked_rects=blocked,
                              soft_rects=soft)

    if inset:
        iv = inset.view
        icircles = []
        ireqs = []
        for it in scene.ships:
            if not (it.alive or it.dead_new):
                continue
            p = iv.to_px(*it.pos[:2])
            if not iv.contains_px(*p):
                continue
            icircles.append((p[0], p[1], it.keep_out))
            if it.sid in in_inset:
                ireqs.append(LabelReq(key="s:" + it.sid, text=it.label(), anchor=p,
                                      radius=it.keep_out,
                                      color=OWN_COLOR if it.own else ENEMY_COLOR,
                                      away=away_dir(it, iv), bold=it.dead_new))
        r = iv.rect
        lay.inset_labels = place_labels(ireqs, (r[0] + 2, r[1] + 16, r[2] - 2, r[3] - 2),
                                        blocked_circles=icircles)

    # side view: label ships that stand out of the viewer's fleet plane
    sreqs = []
    srect = STRIP_RECT
    own_z = sorted(it.pos[2] for it in scene.ships if it.alive and it.own) or \
        sorted(it.pos[2] for it in scene.ships if it.alive)
    if own_z:
        med = own_z[len(own_z) // 2]
        thresh = max(5.0, 14.0 * kmpp)
        for it in scene.ships:
            if it.alive and abs(it.pos[2] - med) > thresh:
                p = strip_px(lay, it.pos[0], it.pos[2])
                if srect[0] <= p[0] <= srect[2]:
                    sreqs.append(LabelReq(key="s:" + it.sid, text=it.name, anchor=p, radius=6.0,
                                          color=OWN_COLOR if it.own else ENEMY_COLOR,
                                          away=(0.0, -1.0 if it.pos[2] > med else 1.0),
                                          size_pt=SMALL_PT))
    scircles = [(*strip_px(lay, it.pos[0], it.pos[2]), 6.0) for it in scene.ships if it.alive]
    lay.strip_labels = place_labels(sreqs, (srect[0] + 2, srect[1] + 2, srect[2] - 2, srect[3] - 2),
                                    blocked_circles=scircles,
                                    blocked_rects=[(srect[0], srect[1], srect[0] + 330, srect[1] + 15),
                                                   (srect[0], srect[1], srect[0] + 44, srect[3])])
    return lay


def inset_frame(inset: Inset) -> Rect:
    r = inset.view.rect
    return (r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3)


def strip_px(lay: Layout, x_km: float, z_km: float) -> Tuple[float, float]:
    px, _ = lay.view.to_px(x_km, lay.view.cy)
    r = STRIP_RECT
    my = (r[1] + r[3]) / 2.0 + 6.0
    return px, my - (z_km - lay.strip_zc) / lay.strip_kmpp


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_matplotlib_warned = False


def _matplotlib():
    """Lazy matplotlib import (Agg, no pyplot); None if unavailable."""
    global _matplotlib_warned
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        return Figure, FigureCanvasAgg
    except ImportError:
        if not _matplotlib_warned:
            _matplotlib_warned = True
            print("[ADMIRAL VISION] matplotlib not installed "
                  "(uv sync --extra viz) - admirals fall back to text-only.")
        return None


def _dart(cx: float, cy: float, fx: float, fy: float, length: float,
          fill_frac: float = 1.0) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """(outline, filled part) of a dart glyph in px. (fx, fy) is the nose
    direction in px space (y down), its length the foreshortening (0..1)."""
    n = math.hypot(fx, fy)
    if n < 1e-9:
        fx, fy, n = 1.0, 0.0, 1.0
    ux, uy = fx / n, fy / n
    fore = max(0.38, min(1.0, n))
    L = length * fore
    W = length

    def tr(px: float, py: float) -> Tuple[float, float]:
        return (cx + ux * px * L - uy * py * W, cy + uy * px * L + ux * py * W)
    outline = [tr(*p) for p in _DART]
    # fill from the tail forward: clip the dart at x <= -0.5 + frac
    cut = -0.5 + max(0.0, min(1.0, fill_frac))
    poly = list(_DART)
    clipped: List[Tuple[float, float]] = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ina, inb = a[0] <= cut, b[0] <= cut
        if ina:
            clipped.append(a)
        if ina != inb:
            t = (cut - a[0]) / (b[0] - a[0])
            clipped.append((cut, a[1] + (b[1] - a[1]) * t))
    return outline, [tr(*p) for p in clipped]


def _flame(cx: float, cy: float, fx: float, fy: float, length: float) -> List[Tuple[float, float]]:
    n = math.hypot(fx, fy) or 1.0
    ux, uy = fx / n, fy / n
    fore = max(0.38, min(1.0, n))
    L = length * fore
    W = length
    base = -0.5 * L - 1.0
    tip = base - 0.55 * length

    def tr(px: float, py: float) -> Tuple[float, float]:
        return (cx + ux * px - uy * py * W, cy + uy * px + ux * py * W)
    return [tr(base, 0.13), tr(tip, 0.0), tr(base, -0.13)]


def _union_outline(centres: Sequence[Tuple[float, float]], radius_km: float,
                   n: int = 160) -> List[List[Tuple[float, float]]]:
    """Polylines (km) of the outline of the union of equal circles."""
    runs: List[List[Tuple[float, float]]] = []
    r2 = (radius_km * 0.9995) ** 2
    for i, (cx, cy) in enumerate(centres):
        cur: List[Tuple[float, float]] = []
        for k in range(n + 1):
            a = 2 * math.pi * k / n
            p = (cx + radius_km * math.cos(a), cy + radius_km * math.sin(a))
            inside = any((p[0] - ox) ** 2 + (p[1] - oy) ** 2 < r2
                         for j, (ox, oy) in enumerate(centres) if j != i)
            if inside:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            runs.append(cur)
    return runs


def render_layout_png(lay: Layout) -> Optional[bytes]:
    mpl = _matplotlib()
    if mpl is None:
        return None
    Figure, FigureCanvasAgg = mpl
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import Rectangle, FancyArrowPatch

    scene, view = lay.scene, lay.view
    fig = Figure(figsize=(CANVAS_W / DPI, CANVAS_H / DPI), dpi=DPI, facecolor=BG)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    from matplotlib.transforms import Bbox, TransformedBbox

    def addc(coll, rect: Rect):
        ax.add_collection(coll, autolim=False)
        return clip_to(coll, rect)

    def adda(artist, rect: Rect):
        ax.add_artist(artist)
        return clip_to(artist, rect)

    def clip_to(artist, rect: Rect):
        artist.set_clip_on(True)
        artist.set_clip_box(TransformedBbox(Bbox.from_extents(rect[0], rect[3], rect[2], rect[1]),
                                            ax.transData))
        return artist

    def text(x, y, s, color=TEXT, size=LABEL_PT, ha="left", va="top", bold=False, alpha=1.0, z=10):
        ax.text(x, y, s, color=color, fontsize=size, family=FONT, ha=ha, va=va,
                fontweight="bold" if bold else "normal", alpha=alpha, zorder=z)

    # ---- header
    text(10, 6, scene.header, color=TEXT, size=HEADER_PT)

    # ---- panels
    for rect in (MAP_RECT, STRIP_RECT):
        ax.add_artist(Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                               facecolor=PANEL_BG, edgecolor="#26384a", lw=0.8, zorder=0))

    def draw_plan(v: View, rect: Rect, labels: List[LabelPlacement], detail: bool, z0: float = 1.0):
        # grid
        step = nice_step(v.width_km)
        segs = []
        x_lo, x_hi = v.cx - v.width_km / 2, v.cx + v.width_km / 2
        y_lo, y_hi = v.cy - v.height_km / 2, v.cy + v.height_km / 2
        k = math.ceil(x_lo / step)
        while k * step <= x_hi:
            px, _ = v.to_px(k * step, 0)
            segs.append([(px, rect[1]), (px, rect[3])])
            if detail:
                text(px + 2, rect[3] - 12, f"{k * step:g}", color=GRID_TEXT, size=SMALL_PT - 0.6, z=z0 + 0.1)
            k += 1
        k = math.ceil(y_lo / step)
        while k * step <= y_hi:
            _, py = v.to_px(0, k * step)
            segs.append([(rect[0], py), (rect[2], py)])
            if detail and py > rect[1] + 26:
                text(rect[0] + 3, py - 10, f"{k * step:g}", color=GRID_TEXT, size=SMALL_PT - 0.6, z=z0 + 0.1)
            k += 1
        addc(LineCollection(segs, colors=GRID, linewidths=0.6, zorder=z0), rect)
        # scale bar (top-right)
        bar_px = step * v.px_per_km
        bx1 = rect[2] - 12
        by = rect[1] + 20
        ax.plot([bx1 - bar_px, bx1], [by, by], color=DIM, lw=1.6, zorder=z0 + 0.2, solid_capstyle="butt")
        text(bx1 - bar_px / 2, by - 3, f"{step:g} km (grid)", color=DIM, size=SMALL_PT, ha="center",
             va="bottom", z=z0 + 0.2)
        if detail:
            # What the solid envelope is: the effective band, not the nominal reach.
            text(bx1, by + 5, gun_band_label(), color=DIM, size=SMALL_PT, ha="right", va="top",
                 z=z0 + 0.2)

        # envelopes (plan only)
        if detail:
            for own in (True, False):
                color = OWN_COLOR if own else ENEMY_COLOR
                fleet = [it for it in scene.ships if it.alive and it.own == own]
                for kind, radius, style in (("guns", GUN_BAND_KM, "-"), ("pd", None, (0, (1.5, 3.0)))):
                    members = [it for it in fleet if (it.guns if kind == "guns" else it.pd)]
                    if not members:
                        continue
                    rad = radius if radius else max(it.pd_km for it in members)
                    runs = _union_outline([it.pos[:2] for it in members], rad)
                    segs = [[v.to_px(*p) for p in run] for run in runs]
                    addc(LineCollection(
                        segs, colors=color, linewidths=0.9 if kind == "guns" else 1.0,
                        linestyles=style, alpha=0.42 if kind == "guns" else 0.5, zorder=z0 + 0.5), rect)

        # closest approach of the fleet centres
        if detail and scene.cpa:
            a = v.to_px(*scene.cpa["own"][:2])
            b = v.to_px(*scene.cpa["enemy"][:2])
            cpa_segs = [[a, b]]
            for p in (a, b):
                cpa_segs += [[(p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4)],
                             [(p[0] - 4, p[1] + 4), (p[0] + 4, p[1] - 4)]]
            addc(LineCollection(cpa_segs, colors=WARN, linewidths=[0.9] + [1.4] * 4,
                                                     linestyles=[(0, (2, 2))] + ["-"] * 4,
                                                     zorder=z0 + 1), rect)

        # motion lines and ghosts
        lines, colors = [], []
        ghost_polys, ghost_cols = [], []
        for it in scene.ships:
            if not it.alive:
                continue
            a = v.to_px(*it.pos[:2])
            g = v.to_px(*it.ghost[:2])
            color = OWN_COLOR if it.own else ENEMY_COLOR
            lines.append([a, g])
            colors.append(color)
            fx, fy = it.fwd[0], -it.fwd[1]
            outline, _ = _dart(g[0], g[1], fx, fy, it.glyph_len * 0.8)
            ghost_polys.append(outline)
            ghost_cols.append(WARN if it.ghost_in_band else color)
        addc(LineCollection(lines, colors=colors, linewidths=0.9, alpha=0.75,
                                                 zorder=z0 + 2), rect)
        addc(PolyCollection(ghost_polys, facecolors="none", edgecolors=ghost_cols,
                                                 linewidths=1.1, alpha=0.95, zorder=z0 + 2.1), rect)

        # torpedo waves
        by_id = {it.sid: it for it in scene.ships}
        for w in scene.waves:
            color = OWN_COLOR if w.own else ENEMY_COLOR
            live_pts = [v.to_px(r[0], r[1]) for r in w.rounds if not r[3]]
            blind_pts = [v.to_px(r[0], r[1]) for r in w.rounds if r[3]]
            if live_pts:
                clip_to(ax.scatter([p[0] for p in live_pts], [p[1] for p in live_pts], s=7, c=color,
                                   marker="o", linewidths=0, zorder=z0 + 3), rect)
            if blind_pts:
                clip_to(ax.scatter([p[0] for p in blind_pts], [p[1] for p in blind_pts], s=8,
                                   facecolors="none", edgecolors="#9aa6b2", marker="o", linewidths=0.8,
                                   zorder=z0 + 3), rect)
            a, b = _wave_segment(v, w, by_id)
            if math.dist(a, b) > 4:
                arrow = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                        color=color if w.n_live else "#9aa6b2",
                                        lw=1.3, ls="-" if w.n_live else "--",
                                        alpha=0.9, zorder=z0 + 3.1, shrinkA=0, shrinkB=0)
                adda(arrow, rect)

        # ships
        fills, fill_cols, outlines, out_cols, flames = [], [], [], [], []
        crosses, cross_cols, cross_w = [], [], []
        for it in scene.ships:
            p = v.to_px(*it.pos[:2])
            color = OWN_COLOR if it.own else ENEMY_COLOR
            if not it.alive:
                size = 7.0 if it.dead_new else 4.5
                for sgn in (1, -1):
                    crosses.append([(p[0] - size, p[1] - sgn * size), (p[0] + size, p[1] + sgn * size)])
                    cross_cols.append(TEXT if it.dead_new else color + "8c")
                    cross_w.append(2.0 if it.dead_new else 1.1)
                continue
            fx, fy = it.fwd[0], -it.fwd[1]
            outline, filled = _dart(p[0], p[1], fx, fy, it.glyph_len, it.hull / 100.0)
            outlines.append(outline)
            out_cols.append(color)
            if len(filled) >= 3:
                fills.append(filled)
                fill_cols.append(color)
            if it.burning:
                flames.append(_flame(p[0], p[1], fx, fy, it.glyph_len))
        addc(LineCollection(crosses, colors=cross_cols, linewidths=cross_w,
                                                 zorder=z0 + 4), rect)
        addc(PolyCollection(flames, facecolors="#fff1b8", edgecolors="#ffb347",
                                                 linewidths=0.6, zorder=z0 + 4), rect)
        addc(PolyCollection(outlines, facecolors=BG, edgecolors="none",
                                                 zorder=z0 + 4.1), rect)
        addc(PolyCollection(fills, facecolors=fill_cols, edgecolors="none",
                                                 zorder=z0 + 4.2), rect)
        addc(PolyCollection(outlines, facecolors="none", edgecolors=out_cols,
                                                 linewidths=1.3, zorder=z0 + 4.3), rect)
        # nose dot: white, so the pointing end reads at a glance
        noses = []
        for it in scene.ships:
            if it.alive:
                p = v.to_px(*it.pos[:2])
                outline, _ = _dart(p[0], p[1], it.fwd[0], -it.fwd[1], it.glyph_len)
                noses.append(outline[0])
        if noses:
            clip_to(ax.scatter([q[0] for q in noses], [q[1] for q in noses], s=9, c="white",
                               linewidths=0, zorder=z0 + 4.4), rect)

        # labels
        segs = [[(l.leader[0], l.leader[1]), (l.leader[2], l.leader[3])] for l in labels if l.leader]
        if segs:
            ax.add_collection(autolim=False, collection=LineCollection(segs, colors="#8a9bb0", linewidths=0.6, alpha=0.8,
                                             zorder=z0 + 5))
        boxes = [[(l.rect[0], l.rect[1]), (l.rect[2], l.rect[1]), (l.rect[2], l.rect[3]),
                  (l.rect[0], l.rect[3])] for l in labels]
        if boxes:
            ax.add_collection(PolyCollection(boxes, facecolors=BG, edgecolors="none", alpha=0.72,
                                             zorder=z0 + 5.1), autolim=False)
        for l in labels:
            text(l.text_xy[0], l.text_xy[1] + 0.5, l.text, color=l.color, size=l.size_pt,
                 bold=l.bold, z=z0 + 5.2)

    draw_plan(view, MAP_RECT, lay.labels, detail=True)
    text(MAP_RECT[0] + 6, MAP_RECT[1] + 4, "PLAN (top-down, X right / Y up, km)", color=DIM, size=SMALL_PT)

    if lay.inset:
        ins = lay.inset
        s = ins.source
        ax.add_artist(Rectangle((s[0], s[1]), s[2] - s[0], s[3] - s[1], fill=False, edgecolor=WARN,
                               lw=0.9, ls=(0, (3, 2)), zorder=20))
        r = ins.view.rect
        ax.add_artist(Rectangle((r[0] - 3, r[1] - 3), r[2] - r[0] + 6, r[3] - r[1] + 6, facecolor=BG,
                               edgecolor=WARN, lw=1.0, zorder=20.5))
        ax.add_artist(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor=PANEL_BG,
                               edgecolor="none", zorder=20.6))
        draw_plan(ins.view, r, lay.inset_labels, detail=False, z0=21.0)
        zoom = ins.view.px_per_km / view.px_per_km
        text(r[0] + 4, r[1] + 3, f"ZOOM x{zoom:.0f} (dashed box)", color=WARN, size=SMALL_PT, z=30)

    # ---- side view strip
    r = STRIP_RECT
    vert = lay.strip_kmpp * view.px_per_km
    text(r[0] + 6, r[1] + 3, "SIDE VIEW (X as in the plan, height Z up), " +
         ("vertical 1:1" if vert < 1.01 else f"height compressed x{vert:.0f}"), color=DIM, size=SMALL_PT)
    zstep = nice_step((r[3] - r[1]) * lay.strip_kmpp, 4)
    segs = []
    z_lo = lay.strip_zc - (r[3] - r[1]) / 2 * lay.strip_kmpp
    z_hi = lay.strip_zc + (r[3] - r[1]) / 2 * lay.strip_kmpp
    k = math.ceil(z_lo / zstep)
    while k * zstep <= z_hi:
        _, py = strip_px(lay, 0.0, k * zstep)
        if r[1] + 2 < py < r[3] - 2:
            segs.append([(r[0], py), (r[2], py)])
            if r[1] + 28 < py:
                text(r[0] + 3, py - 10, f"Z {k * zstep:g}", color=GRID_TEXT, size=SMALL_PT - 0.6)
        k += 1
    step = nice_step(view.width_km)
    k = math.ceil((view.cx - view.width_km / 2) / step)
    while k * step <= view.cx + view.width_km / 2:
        px, _ = view.to_px(k * step, 0)
        segs.append([(px, r[1]), (px, r[3])])
        k += 1
    addc(LineCollection(segs, colors=GRID, linewidths=0.6, zorder=1), r)
    pts, cols = [], []
    for w in scene.waves:
        for rd in w.rounds:
            pts.append(strip_px(lay, rd[0], rd[2]))
            cols.append("#9aa6b2" if rd[3] else (OWN_COLOR if w.own else ENEMY_COLOR))
    if pts:
        sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=5, c=cols, linewidths=0, zorder=3)
        clip_to(sc, r)
    outlines, out_cols = [], []
    for it in scene.ships:
        p = strip_px(lay, it.pos[0], it.pos[2])
        color = OWN_COLOR if it.own else ENEMY_COLOR
        if not it.alive:
            if it.dead_new:
                for sgn in (1, -1):
                    ln, = ax.plot([p[0] - 4, p[0] + 4], [p[1] - sgn * 4, p[1] + sgn * 4], color=TEXT,
                                  lw=1.4, zorder=4)
                    clip_to(ln, r)
            continue
        outline, _ = _dart(p[0], p[1], it.fwd[0], -it.fwd[2], it.glyph_len * 0.75)
        outlines.append(outline)
        out_cols.append(color)
    addc(PolyCollection(outlines, facecolors=out_cols, edgecolors="white",
                                             linewidths=0.4, zorder=4), r)
    for l in lay.strip_labels:
        if l.leader:
            ax.plot([l.leader[0], l.leader[2]], [l.leader[1], l.leader[3]], color="#8a9bb0", lw=0.5,
                    zorder=5)
        text(l.text_xy[0], l.text_xy[1] + 0.5, l.text, color=l.color, size=l.size_pt, z=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, metadata={"Software": None})
    return buf.getvalue()


def render_scene_png(scene: Scene) -> Optional[bytes]:
    return render_layout_png(layout_scene(scene))


def render_frame_png(
    frame: Dict[str, Any],
    ships_meta: Dict[str, Dict[str, str]],
    faction: Optional[str] = None,
    prev_frame: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[int] = None,
    total_checkpoints: Optional[int] = None,
    score_text: Optional[str] = None,
    interval_s: float = 30.0,
) -> Optional[bytes]:
    """Render one frame (live or from a trace) to PNG bytes (None without matplotlib)."""
    if "state" not in next(iter(frame.get("torpedoes") or [{"state": 1}])):
        annotate_torpedoes(frame)
    scene = build_scene(frame, ships_meta, faction=faction, prev_frame=prev_frame,
                        checkpoint=checkpoint, total_checkpoints=total_checkpoints,
                        score_text=score_text, interval_s=interval_s)
    return render_scene_png(scene)


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------

def score_header(simulation: Any, faction: str) -> str:
    """'score you 212 / them 148' - the time-limit verdict's two scores
    (fleet_intel.time_limit_scores), for the header when the score flag is on."""
    try:
        from .fleet_intel import time_limit_scores
        ships = list((getattr(simulation, "ships", None) or {}).values())
        a, b = time_limit_scores([s for s in ships if getattr(s, "faction", None) == "alpha"],
                                 [s for s in ships if getattr(s, "faction", None) == "beta"])
    except Exception:  # noqa: BLE001 - advisory only
        return ""
    you, them = (a, b) if faction == "alpha" else (b, a)
    return f"score you {you:.0f} / them {them:.0f}"


class AdmiralViewState:
    """
    What the plot needs beyond the current state: the frame each faction saw
    at its previous checkpoint (deaths and new salvos since then). The battle
    runner owns one instance while at least one admiral has vision.
    """

    def __init__(self) -> None:
        self._last_frame: Dict[str, Dict[str, Any]] = {}

    def render(
        self,
        simulation: Any,
        ships_meta: Dict[str, Dict[str, str]],
        faction: Optional[str] = None,
        checkpoint: Optional[int] = None,
        total_checkpoints: Optional[int] = None,
        score_text: Optional[str] = None,
    ) -> Optional[bytes]:
        """Render the current battle state to PNG bytes (None if no renderer)."""
        frame = build_live_frame(simulation)
        interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0)
        png = render_frame_png(frame, ships_meta, faction=faction,
                               prev_frame=self._last_frame.get(faction or ""),
                               checkpoint=checkpoint, total_checkpoints=total_checkpoints,
                               score_text=score_text, interval_s=interval)
        self._last_frame[faction or ""] = frame
        return png
