#!/usr/bin/env python3
"""
Render a tactical snapshot image from a battle recording (or live state).

Prototype for "admiral vision": a compact, high-contrast top-down plot of the
battle a multimodal LLM (or human) can read at a glance - who is where,
headings, velocity vectors, torpedo streams, and active PD engagements.
Geometry is the one thing the text status reports cannot convey; this image
is meant to complement them, not replace them.

Usage:
    uv run --extra viz python scripts/render_admiral_view.py \
        --recording data/recordings/battle_X.json --time 95 --out /tmp/view.png

The input is a recording with sim_trace, but the frame dict consumed by
render_frame() is exactly what the live simulation can produce each tick, so
the same renderer works for in-battle admiral snapshots.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

FACTION_COLORS = {"alpha": "#00d4ff", "beta": "#ff6644"}
BG = "#0a0d14"
GRID = "#1e3246"
TEXT = "#c8d6e5"


def faction_of(entity_id: str) -> str:
    return "alpha" if str(entity_id).startswith("alpha") else "beta"


def find_frame(sim_trace: list, t: float) -> dict:
    """Nearest trace frame at or before t (falls back to first frame)."""
    best = sim_trace[0]
    for frame in sim_trace:
        if frame["t"] <= t:
            best = frame
        else:
            break
    return best


def torpedo_trails(sim_trace: list, t: float, seconds: float = 25.0) -> dict:
    """torpedo_id -> [(x, y), ...] positions over the trailing window."""
    trails = {}
    for frame in sim_trace:
        if t - seconds <= frame["t"] <= t:
            for torp in frame.get("torpedoes", []):
                trails.setdefault(torp["id"], []).append(
                    (torp["pos"][0] / 1000.0, torp["pos"][1] / 1000.0))
    return trails


def active_pd_beams(events: list, t: float) -> list:
    """(shooter_id, target_id) pairs with a pd_fired tick within ~1.5s of t."""
    beams = set()
    for e in events:
        if e["event_type"] == "pd_fired" and abs(e["timestamp"] - t) <= 1.5:
            beams.add((e["ship_id"], e["data"].get("target_id")))
    return list(beams)


def render_frame(frame: dict, ships_meta: dict, trails: dict, beams: list,
                 t: float, out_path: str, recent_events: list = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Torpedo positions by id for beam endpoint lookup
    torp_pos = {torp["id"]: (torp["pos"][0] / 1000.0, torp["pos"][1] / 1000.0)
                for torp in frame.get("torpedoes", [])}
    ship_pos = {sid: (s["pos"][0] / 1000.0, s["pos"][1] / 1000.0)
                for sid, s in frame.get("ships", {}).items()}

    # PD beams first (under everything)
    for shooter, target in beams:
        src = ship_pos.get(shooter)
        dst = torp_pos.get(target) or ship_pos.get(target)
        if src and dst:
            ax.plot([src[0], dst[0]], [src[1], dst[1]],
                    color="#ff3322", lw=1.2, alpha=0.85, zorder=2)

    # Torpedo trails + markers
    for torp_id, pts in trails.items():
        if len(pts) > 1:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=FACTION_COLORS[faction_of(torp_id)],
                    lw=0.8, alpha=0.35, zorder=3)
    for torp in frame.get("torpedoes", []):
        x, y = torp["pos"][0] / 1000.0, torp["pos"][1] / 1000.0
        color = FACTION_COLORS[faction_of(torp["source"])]
        dead = torp.get("disabled")
        ax.scatter([x], [y], marker="D", s=28,
                   c="#666677" if dead else color,
                   edgecolors="white", linewidths=0.4, zorder=5)

    # Projectiles
    for proj in frame.get("projectiles", []):
        ax.scatter([proj["pos"][0] / 1000.0], [proj["pos"][1] / 1000.0],
                   marker=".", s=14, c="#ffee66", zorder=4)

    # Ships: oriented triangle + velocity arrow + label block
    for ship_id, s in frame.get("ships", {}).items():
        if s.get("destroyed"):
            continue
        x, y = s["pos"][0] / 1000.0, s["pos"][1] / 1000.0
        color = FACTION_COLORS[faction_of(ship_id)]
        heading = math.atan2(s["fwd"][1], s["fwd"][0])

        ax.scatter([x], [y], marker=(3, 0, math.degrees(heading) - 90),
                   s=260, c=color, edgecolors="white", linewidths=0.8, zorder=6)

        # Velocity arrow (km/s scaled up to be visible at battle scale)
        vx, vy = s["vel"][0] / 1000.0, s["vel"][1] / 1000.0
        speed = math.hypot(vx, vy)
        if speed > 0.05:
            arrow_scale = 25.0
            ax.add_patch(FancyArrow(
                x, y, vx * arrow_scale, vy * arrow_scale,
                width=0.4, head_width=6, head_length=9,
                color=color, alpha=0.55, zorder=5))

        meta = ships_meta.get(ship_id, {})
        name = meta.get("name", ship_id)
        cls = meta.get("type", "").replace("_", " ").upper()
        label = (f"{name}\n{cls}\n"
                 f"hull {s.get('hull', 100):.0f}%  "
                 f"{'thrust' if s.get('thrust', 0) > 0.05 else 'coast'}  "
                 f"{speed:.1f} km/s\n{s.get('maneuver', '')}")
        ax.annotate(label, (x, y), xytext=(14, 14),
                    textcoords="offset points", fontsize=8.5,
                    color=TEXT, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#101826",
                              ec=color, alpha=0.85), zorder=7)

    # Distance line between the two main combatants
    ship_ids = [sid for sid, s in frame.get("ships", {}).items()
                if not s.get("destroyed")]
    if len(ship_ids) >= 2:
        a, b = ship_pos[ship_ids[0]], ship_pos[ship_ids[1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=0.7,
                linestyle="--", alpha=0.8, zorder=1)
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        ax.annotate(f"{dist:.0f} km", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                    fontsize=9, color="#8899aa", family="monospace",
                    ha="center", xytext=(0, 6), textcoords="offset points")

    # Recent-events ticker (bottom): the "what just happened" context
    if recent_events:
        lines = []
        for e in recent_events[-4:]:
            lines.append(f"T+{e['timestamp']:.0f} {e['event_type'].replace('_', ' ')}")
        ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, color="#7f8fa0", family="monospace", va="bottom")

    torps_a = sum(1 for tp in frame.get("torpedoes", []) if faction_of(tp["source"]) == "alpha")
    torps_b = len(frame.get("torpedoes", [])) - torps_a
    ax.set_title(
        f"TACTICAL PLOT  T+{t:.0f}s   "
        f"torpedoes in flight: alpha {torps_a} / beta {torps_b}",
        color=TEXT, fontsize=11, family="monospace", pad=12)

    ax.grid(color=GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5a6b7c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xlabel("X (km)", color="#5a6b7c", fontsize=8)
    ax.set_ylabel("Y (km)", color="#5a6b7c", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.recording).read_text())
    sim_trace = data.get("sim_trace", [])
    if not sim_trace:
        print("Recording has no sim_trace (re-run battle with --trace)")
        return 1

    frame = find_frame(sim_trace, args.time)
    trails = torpedo_trails(sim_trace, args.time)
    beams = active_pd_beams(data.get("events", []), frame["t"])
    recent = [e for e in data.get("events", [])
              if frame["t"] - 30 <= e["timestamp"] <= frame["t"]
              and e["event_type"] in (
                  "torpedo_launched", "torpedo_impact", "pd_torpedo_disabled",
                  "torpedo_miss", "hit", "module_destroyed", "penetration")]

    ships_meta = {}
    for side in ("alpha", "beta"):
        ships_meta[side] = {
            "name": data.get(f"{side}_ship", side),
            "type": (data.get(f"{side}_specs") or {}).get("ship_type", ""),
        }
    for fleet_key, side in (("alpha_fleet", "alpha"), ("beta_fleet", "beta")):
        for ship in (data.get(fleet_key) or {}).get("ships", []) or []:
            ships_meta[ship["ship_id"]] = {
                "name": ship.get("ship_name", ship["ship_id"]),
                "type": ship.get("ship_type", ""),
            }

    out = render_frame(frame, ships_meta, trails, beams, frame["t"],
                       args.out, recent)
    print(f"Rendered {out} (frame T+{frame['t']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
