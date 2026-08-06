"""
Admiral vision: render a live tactical plot for multimodal admirals.

Implements the plan in docs/admiral_vision.md. Text reports carry scalars
well but geometry poorly; a picture carries "two torpedo streams converging
from bearing 040 and 320 while the enemy kites away" in one glance. At each
admiral checkpoint the battle runner renders the current simulation state to
a PNG (top-down annotated plot plus a 3D perspective panel) and attaches it
to the admiral's directive prompt as a base64 image content part. OpenRouter
passes such parts through to multimodal models.

Everything degrades silently to text-only:
- matplotlib missing (it lives behind the `viz` optional dependency group)
- the admiral's model is not vision-capable
- the admiral config has vision disabled (the default)

The renderer is intentionally omniscient-but-symmetric: it shows exactly the
same position/velocity information the existing text snapshots already give
both admirals, so it adds geometry comprehension, not new intelligence.
"""

from __future__ import annotations

import base64
import io
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

FACTION_COLORS = {"alpha": "#00d4ff", "beta": "#ff6644"}
BG = "#0a0d14"
GRID = "#1e3246"
TEXT = "#c8d6e5"

# How far back torpedo trails reach, and the sampling cadence.
TRAIL_WINDOW_S = 25.0

# PD beam lines are drawn for engagements within this many seconds of "now".
PD_BEAM_WINDOW_S = 1.5

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
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
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
# Live frame assembly
# ---------------------------------------------------------------------------

def build_live_frame(simulation: Any) -> Dict[str, Any]:
    """
    Assemble the renderer's frame dict straight from a live CombatSimulation.

    Field names and units match the recording sim_trace format exactly
    (`pos`/`vel`/`fwd` in meters, `hull` percent), so the same renderer works
    on live state and on replays.
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
            thrust = ship.current_maneuver.throttle
        frame["ships"][ship_id] = {
            "pos": [ship.position.x, ship.position.y, ship.position.z],
            "vel": [ship.velocity.x, ship.velocity.y, ship.velocity.z],
            "fwd": [ship.forward.x, ship.forward.y, ship.forward.z],
            "thrust": thrust,
            "maneuver": maneuver,
            "destroyed": ship.is_destroyed,
            "hull": getattr(ship, "hull_integrity", 100.0),
        }

    for proj_flight in simulation.projectiles:
        proj = proj_flight.projectile
        frame["projectiles"].append({
            "id": proj_flight.projectile_id,
            "pos": [proj.position.x, proj.position.y, proj.position.z],
            "source": proj_flight.source_ship_id,
        })

    for torp_flight in simulation.torpedoes:
        torp = torp_flight.torpedo
        frame["torpedoes"].append({
            "id": torp_flight.torpedo_id,
            "pos": [torp.position.x, torp.position.y, torp.position.z],
            "source": torp_flight.source_ship_id,
            "target": torp.target_id,
            "disabled": torp_flight.is_disabled,
        })

    return frame


class AdmiralViewState:
    """
    Rolling battle history the renderer needs but the live sim does not keep.

    The battle runner owns one instance, calls `sample()` once per simulation
    step and `note_event()` from its event callback. Both are cheap enough to
    run every tick; they only exist while at least one admiral has vision.
    """

    def __init__(self, trail_window_s: float = TRAIL_WINDOW_S):
        self.trail_window_s = trail_window_s
        # (t, {torpedo_id: (x_m, y_m, z_m)})
        self._trail_samples: deque = deque()
        # (t, event_type_name, ship_id, data)
        self._events: deque = deque(maxlen=400)

    def sample(self, simulation: Any) -> None:
        """Record torpedo positions for trail rendering; prune old samples."""
        t = simulation.current_time
        torps = getattr(simulation, "torpedoes", None)
        if torps:
            positions = {
                tf.torpedo_id: (tf.torpedo.position.x,
                                tf.torpedo.position.y,
                                tf.torpedo.position.z)
                for tf in torps
            }
            self._trail_samples.append((t, positions))
        while self._trail_samples and self._trail_samples[0][0] < t - self.trail_window_s:
            self._trail_samples.popleft()

    def note_event(self, event: Any) -> None:
        """Buffer a simulation event for PD beams and the events ticker."""
        etype = getattr(event.event_type, "name", str(event.event_type))
        self._events.append((event.timestamp, etype, event.ship_id, event.data or {}))

    # -- renderer inputs -----------------------------------------------------

    def trails(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """torpedo_id -> [(x_m, y_m, z_m), ...] over the trailing window."""
        trails: Dict[str, List[Tuple[float, float, float]]] = {}
        for _, positions in self._trail_samples:
            for torp_id, pos in positions.items():
                trails.setdefault(torp_id, []).append(pos)
        return trails

    def active_pd_beams(self, t: float) -> List[Tuple[str, str]]:
        """(shooter_ship_id, target_id) pairs engaged within the beam window."""
        beams = set()
        for ts, etype, ship_id, data in self._events:
            if etype == "PD_ENGAGED" and abs(ts - t) <= PD_BEAM_WINDOW_S:
                beams.add((ship_id, data.get("target_id")))
        return list(beams)

    def recent_events(self, t: float, window_s: float = 45.0) -> List[Tuple[float, str]]:
        """(timestamp, label) for ticker-worthy events in the trailing window."""
        interesting = {
            "TORPEDO_LAUNCHED": "torpedo launched",
            "TORPEDO_IMPACT": "TORPEDO IMPACT",
            "TORPEDO_FUEL_EXHAUSTED": "torpedo burnout",
            "TORPEDO_RETARGETED": "torpedo retargeted",
            "PD_TORPEDO_DISABLED": "seeker blinded",
            "PD_TORPEDO_DESTROYED": "torpedo destroyed",
            "PROJECTILE_IMPACT": "gun hit",
            "ARMOR_PENETRATED": "PENETRATION",
            "MODULE_DESTROYED": "module destroyed",
            "SHIP_DESTROYED": "SHIP DESTROYED",
        }
        out = []
        for ts, etype, ship_id, _ in self._events:
            if t - window_s <= ts <= t and etype in interesting:
                out.append((ts, f"{interesting[etype]} [{ship_id}]"))
        return out[-6:]

    def render(
        self,
        simulation: Any,
        ships_meta: Dict[str, Dict[str, str]],
        faction: Optional[str] = None,
    ) -> Optional[bytes]:
        """Render the current battle state to PNG bytes (None if no renderer)."""
        frame = build_live_frame(simulation)
        return render_frame_png(
            frame=frame,
            ships_meta=ships_meta,
            trails=self.trails(),
            beams=self.active_pd_beams(frame["t"]),
            recent_events=self.recent_events(frame["t"]),
            faction=faction,
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_matplotlib_warned = False


def _get_plt():
    """Lazy matplotlib import; warn once and return None if unavailable."""
    global _matplotlib_warned
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if not _matplotlib_warned:
            _matplotlib_warned = True
            print("[ADMIRAL VISION] matplotlib not installed "
                  "(uv sync --extra viz) - admirals fall back to text-only.")
        return None


def _faction_of(entity_id: str, ships_meta: Dict[str, Dict[str, str]]) -> str:
    meta = ships_meta.get(str(entity_id))
    if meta and meta.get("faction"):
        return meta["faction"]
    return "alpha" if str(entity_id).startswith("alpha") else "beta"


def _torpedo_source(torp_id: str, source_by_id: Dict[str, str]) -> str:
    """Source ship id for a torpedo, from the live frame or the id itself."""
    if torp_id in source_by_id:
        return source_by_id[torp_id]
    # Simulation ids are "torp_{ship_id}_{hex8}"
    if torp_id.startswith("torp_"):
        return torp_id[5:].rsplit("_", 1)[0]
    return torp_id


def render_frame_png(
    frame: Dict[str, Any],
    ships_meta: Dict[str, Dict[str, str]],
    trails: Dict[str, List[Tuple[float, float, float]]],
    beams: List[Tuple[str, str]],
    recent_events: Optional[List[Tuple[float, str]]] = None,
    faction: Optional[str] = None,
) -> Optional[bytes]:
    """
    Render one battle frame to PNG bytes.

    Left panel: annotated top-down tactical plot (the workhorse - names,
    hull, maneuver, headings, velocity vectors, torpedo trails, PD beams).
    Right panel: 3D perspective of the same state, which is what makes
    out-of-plane geometry (climbing torpedoes, stacked formations) legible.
    """
    plt = _get_plt()
    if plt is None:
        return None
    from matplotlib.patches import FancyArrow

    t = frame.get("t", 0.0)
    fig = plt.figure(figsize=(13.6, 7.6), dpi=105)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax.set_facecolor(BG)

    km = 1 / 1000.0
    torp_pos = {tp["id"]: (tp["pos"][0] * km, tp["pos"][1] * km)
                for tp in frame.get("torpedoes", [])}
    ship_pos = {sid: (s["pos"][0] * km, s["pos"][1] * km)
                for sid, s in frame.get("ships", {}).items()}

    # PD beams under everything
    for shooter, target in beams:
        src = ship_pos.get(shooter)
        dst = torp_pos.get(target) or ship_pos.get(target)
        if src and dst:
            ax.plot([src[0], dst[0]], [src[1], dst[1]],
                    color="#ff3322", lw=1.2, alpha=0.85, zorder=2)

    # Torpedo trails + markers
    torp_source = {tp["id"]: tp["source"] for tp in frame.get("torpedoes", [])}
    for torp_id, pts in trails.items():
        if len(pts) > 1:
            xs = [p[0] * km for p in pts]
            ys = [p[1] * km for p in pts]
            src = _torpedo_source(torp_id, torp_source)
            ax.plot(xs, ys, color=FACTION_COLORS[_faction_of(src, ships_meta)],
                    lw=0.8, alpha=0.35, zorder=3)
    for tp in frame.get("torpedoes", []):
        x, y = tp["pos"][0] * km, tp["pos"][1] * km
        color = FACTION_COLORS[_faction_of(tp["source"], ships_meta)]
        ax.scatter([x], [y], marker="D", s=28,
                   c="#666677" if tp.get("disabled") else color,
                   edgecolors="white", linewidths=0.4, zorder=5)

    # Projectiles
    for proj in frame.get("projectiles", []):
        ax.scatter([proj["pos"][0] * km], [proj["pos"][1] * km],
                   marker=".", s=14, c="#ffee66", zorder=4)

    # Ships. Labels flip to the left of ships on the right half of the
    # battlespace so stat cards stay inside the frame.
    ship_xs = [p[0] for p in ship_pos.values()]
    mid_x = (min(ship_xs) + max(ship_xs)) / 2 if ship_xs else 0.0
    for ship_id, s in frame.get("ships", {}).items():
        if s.get("destroyed"):
            continue
        x, y = s["pos"][0] * km, s["pos"][1] * km
        color = FACTION_COLORS[_faction_of(ship_id, ships_meta)]
        heading = math.atan2(s["fwd"][1], s["fwd"][0])
        ax.scatter([x], [y], marker=(3, 0, math.degrees(heading) - 90),
                   s=260, c=color, edgecolors="white", linewidths=0.8, zorder=6)

        vx, vy = s["vel"][0] * km, s["vel"][1] * km
        speed = math.hypot(vx, vy)
        if speed > 0.05:
            ax.add_patch(FancyArrow(
                x, y, vx * 25.0, vy * 25.0,
                width=0.4, head_width=6, head_length=9,
                color=color, alpha=0.55, zorder=5))

        meta = ships_meta.get(ship_id, {})
        name = meta.get("name", ship_id)
        cls = meta.get("type", "").replace("_", " ").upper()
        label = (f"{name}\n{cls}\n"
                 f"hull {s.get('hull', 100):.0f}%  "
                 f"{'thrust' if s.get('thrust', 0) > 0.05 else 'coast'}  "
                 f"{speed:.1f} km/s\n{s.get('maneuver', '')}")
        on_right = x > mid_x
        ax.annotate(label, (x, y),
                    xytext=(-14 if on_right else 14, 14),
                    textcoords="offset points", fontsize=8.5,
                    ha="right" if on_right else "left",
                    color=TEXT, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#101826",
                              ec=color, alpha=0.85), zorder=7)

    # Range line between closest opposing pair
    alive = [(sid, ship_pos[sid]) for sid, s in frame.get("ships", {}).items()
             if not s.get("destroyed")]
    closest = None
    for i, (sid_a, pa) in enumerate(alive):
        for sid_b, pb in alive[i + 1:]:
            if _faction_of(sid_a, ships_meta) == _faction_of(sid_b, ships_meta):
                continue
            d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if closest is None or d < closest[0]:
                closest = (d, pa, pb)
    if closest:
        d, pa, pb = closest
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=GRID, lw=0.7,
                linestyle="--", alpha=0.8, zorder=1)
        ax.annotate(f"{d:.0f} km", ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2),
                    fontsize=9, color="#8899aa", family="monospace",
                    ha="center", xytext=(0, 6), textcoords="offset points")

    # Recent-events ticker
    if recent_events:
        lines = [f"T+{ts:.0f} {label}" for ts, label in recent_events]
        ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, color="#7f8fa0", family="monospace", va="bottom")

    torps_a = sum(1 for tp in frame.get("torpedoes", [])
                  if _faction_of(tp["source"], ships_meta) == "alpha")
    torps_b = len(frame.get("torpedoes", [])) - torps_a
    you = ""
    if faction:
        you = f"   YOU COMMAND: {faction.upper()} " \
              f"({'cyan' if faction == 'alpha' else 'orange'})"
    ax.set_title(
        f"TACTICAL PLOT  T+{t:.0f}s   "
        f"torpedoes: alpha {torps_a} / beta {torps_b}{you}",
        color=TEXT, fontsize=10.5, family="monospace", pad=12)

    ax.grid(color=GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5a6b7c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xlabel("X (km)", color="#5a6b7c", fontsize=8)
    ax.set_ylabel("Y (km)", color="#5a6b7c", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    _render_3d_panel(ax3d, frame, ships_meta, trails)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_3d_panel(ax3d, frame, ships_meta, trails) -> None:
    """3D perspective: same state, viewed from above-and-behind at an angle."""
    km = 1 / 1000.0
    xs_all, ys_all, zs_all = [], [], []

    torp_source = {tp["id"]: tp["source"] for tp in frame.get("torpedoes", [])}
    for torp_id, pts in trails.items():
        if len(pts) > 1:
            src = _torpedo_source(torp_id, torp_source)
            ax3d.plot([p[0] * km for p in pts],
                      [p[1] * km for p in pts],
                      [p[2] * km for p in pts],
                      color=FACTION_COLORS[_faction_of(src, ships_meta)],
                      lw=0.7, alpha=0.35)

    for tp in frame.get("torpedoes", []):
        x, y, z = (c * km for c in tp["pos"])
        xs_all.append(x); ys_all.append(y); zs_all.append(z)
        color = "#666677" if tp.get("disabled") else \
            FACTION_COLORS[_faction_of(tp["source"], ships_meta)]
        ax3d.scatter([x], [y], [z], marker="D", s=16, c=color,
                     edgecolors="white", linewidths=0.3)

    for ship_id, s in frame.get("ships", {}).items():
        if s.get("destroyed"):
            continue
        x, y, z = (c * km for c in s["pos"])
        xs_all.append(x); ys_all.append(y); zs_all.append(z)
        color = FACTION_COLORS[_faction_of(ship_id, ships_meta)]
        ax3d.scatter([x], [y], [z], marker="^", s=90, c=color,
                     edgecolors="white", linewidths=0.6)
        # Velocity vector so 3D motion is readable
        vx, vy, vz = (c * km * 25.0 for c in s["vel"])
        if math.hypot(vx, vy, vz) > 1.0:
            ax3d.plot([x, x + vx], [y, y + vy], [z, z + vz],
                      color=color, lw=1.1, alpha=0.6)
        name = ships_meta.get(ship_id, {}).get("name", ship_id)
        ax3d.text(x, y, z, f"  {name}", fontsize=7, color=TEXT,
                  family="monospace")

    # Near-honest aspect: box proportional to data extents, except Z gets a
    # minimum visual thickness. A perfectly planar battle would otherwise
    # collapse the panel to an unreadable sliver; the tick labels still carry
    # the true Z values, so out-of-plane deviations remain judgeable.
    if xs_all:
        def span(vals, min_span=0.0):
            lo, hi = min(vals), max(vals)
            width = max(hi - lo, min_span)
            center = (lo + hi) / 2
            pad = max(width * 0.15, 10.0)
            return center - width / 2 - pad, center + width / 2 + pad
        xlim = span(xs_all)
        ylim = span(ys_all)
        horiz = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        zlim = span(zs_all, min_span=0.30 * horiz)
        ax3d.set_xlim(*xlim); ax3d.set_ylim(*ylim); ax3d.set_zlim(*zlim)
        ax3d.set_box_aspect((max(xlim[1] - xlim[0], 1.0),
                             max(ylim[1] - ylim[0], 1.0),
                             max(zlim[1] - zlim[0], 1.0)))
        ax3d.locator_params(nbins=4)

    ax3d.set_facecolor(BG)
    ax3d.view_init(elev=24, azim=-55)
    ax3d.set_title("3D PERSPECTIVE", color="#7f8fa0", fontsize=9,
                   family="monospace", pad=2)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color((0.04, 0.05, 0.08, 1.0))
        axis.label.set_color("#5a6b7c")
        axis.set_tick_params(colors="#5a6b7c", labelsize=6)
    ax3d.set_xlabel("X (km)", fontsize=7)
    ax3d.set_ylabel("Y (km)", fontsize=7)
    ax3d.set_zlabel("Z (km)", fontsize=7)
