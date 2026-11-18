#!/usr/bin/env python3
"""
Render the admiral's tactical plot (src/llm/admiral_view.py) from a recording.

The recording's sim_trace frames have the live frame's schema, so this is the
exact image an admiral with vision would have been sent at that checkpoint
(the previous checkpoint's frame supplies "what changed").

    uv run --extra viz python scripts/render_admiral_view.py \
        data/recordings/battle_X.json.gz --checkpoint 3 --faction alpha --out /tmp/cp003_alpha.png

    # every checkpoint for both sides into a folder (cpNNN_<faction>.png)
    uv run --extra viz python scripts/render_admiral_view.py battle_X.json.gz --all --out-dir /tmp/plots

Checkpoint N is at T = N x decision_interval_s (the runner's numbering: the
saved vision frames cp003_alpha.png are T+90 at 30 s). --time T renders an
arbitrary trace time instead.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recording_io import load_recording  # noqa: E402
from src.llm.admiral_view import frame_from_trace, render_frame_png  # noqa: E402


def find_frame(sim_trace: list, t: float) -> dict:
    """Nearest trace frame at or before t (falls back to the first frame)."""
    best = sim_trace[0]
    for frame in sim_trace:
        if frame["t"] <= t + 1e-6:
            best = frame
        else:
            break
    return best


def ships_meta_of(data: dict) -> dict:
    meta = {}
    for side in ("alpha", "beta"):
        meta[side] = {
            "name": data.get(f"{side}_ship") or side,
            "type": (data.get(f"{side}_specs") or {}).get("ship_type", ""),
            "faction": side,
        }
    for fleet_key, side in (("alpha_fleet", "alpha"), ("beta_fleet", "beta")):
        for ship in (data.get(fleet_key) or {}).get("ships", []) or []:
            meta[ship["ship_id"]] = {
                "name": ship.get("ship_name", ship["ship_id"]),
                "type": ship.get("ship_type", ""),
                "faction": side,
            }
    return meta


def render_at(data: dict, t: float, faction: str, checkpoint=None, score_text=None) -> bytes:
    trace = data["sim_trace"]
    interval = float(data.get("decision_interval_s") or 30.0)
    frame = frame_from_trace(find_frame(trace, t))
    # the previous checkpoint's frame (none at the first checkpoint, as live)
    prev = frame_from_trace(find_frame(trace, t - interval)) if t - interval >= interval - 1e-6 else None
    total = data.get("max_checkpoints") or data.get("total_checkpoints")
    return render_frame_png(frame, ships_meta_of(data), faction=faction, prev_frame=prev,
                            checkpoint=checkpoint, total_checkpoints=total,
                            score_text=score_text, interval_s=interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    parser.add_argument("--checkpoint", type=int, help="checkpoint number (T = N x interval)")
    parser.add_argument("--time", type=float, help="trace time in seconds instead of a checkpoint")
    parser.add_argument("--faction", choices=("alpha", "beta"), default="alpha")
    parser.add_argument("--score", help="score text for the header (the score flag's line)")
    parser.add_argument("--out", help="output PNG (single render)")
    parser.add_argument("--all", action="store_true", help="every checkpoint, both factions")
    parser.add_argument("--out-dir", help="folder for --all")
    args = parser.parse_args()

    data = load_recording(args.recording)
    if not data.get("sim_trace"):
        print("Recording has no sim_trace (re-run the battle with --trace)")
        return 1
    interval = float(data.get("decision_interval_s") or 30.0)
    if args.all:
        out_dir = Path(args.out_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        end = data["sim_trace"][-1]["t"]
        n = 1
        while n * interval <= end + 1e-6:
            for faction in ("alpha", "beta"):
                png = render_at(data, n * interval, faction, checkpoint=n)
                (out_dir / f"cp{n:03d}_{faction}.png").write_bytes(png)
            n += 1
        print(f"Rendered {2 * (n - 1)} frames into {out_dir}")
        return 0
    if args.checkpoint is None and args.time is None:
        parser.error("give --checkpoint N, --time T or --all")
    t = args.time if args.time is not None else args.checkpoint * interval
    start = time.perf_counter()
    png = render_at(data, t, args.faction, checkpoint=args.checkpoint, score_text=args.score)
    ms = (time.perf_counter() - start) * 1000.0
    out = args.out or f"cp{(args.checkpoint or 0):03d}_{args.faction}.png"
    Path(out).write_bytes(png)
    print(f"Rendered {out} (T+{t:.0f}s, {args.faction}, {ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
