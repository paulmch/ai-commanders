#!/usr/bin/env python3
"""Fake live-battle server: replays a saved recording as if it were live.

Dev harness for the viewer's live mode. Loads a saved BattleRecording JSON
and advances an internal sim-time cursor at --speed x wall-clock, serving
the same /live/... contract the real MCP HTTP server implements:

    GET /live/recording[?since_t=<float>]  -> {"live": {...}, "recording": {...}|null}
    GET /live/predictions                  -> {"t", "t_checkpoint", "ships": {...}}

Usage:
    uv run python scripts/fake_live_server.py visualizer/public/recordings/torpedo.json --speed 4
    uv run python scripts/fake_live_server.py <recording.json> --draft-seconds 20   # draft splash first
"""

import argparse
import json
import time
from bisect import bisect_right

from aiohttp import web


class FakeLiveServer:
    def __init__(self, recording, speed, draft_seconds, decision_interval):
        self.recording = recording
        self.speed = speed
        self.draft_seconds = draft_seconds
        self.interval = decision_interval
        self.frames = sorted(recording.get("sim_trace") or [], key=lambda f: f["t"])
        self.events = sorted(recording.get("events") or [], key=lambda e: e["timestamp"])
        self.frame_ts = [f["t"] for f in self.frames]
        self.t_end = self.frame_ts[-1] if self.frame_ts else 0.0
        self.start = time.monotonic()

    def cursor(self):
        """Sim-time cursor, or None while the synthetic draft phase runs."""
        battle_elapsed = (time.monotonic() - self.start) - self.draft_seconds
        if battle_elapsed < 0:
            return None
        return battle_elapsed * self.speed

    # ------------------------------------------------------------- live block

    def draft_live_block(self):
        elapsed = time.monotonic() - self.start
        frac = elapsed / self.draft_seconds if self.draft_seconds else 1.0
        alpha_ready = frac >= 0.6
        beta_ready = frac >= 0.9
        return {
            "phase": "draft",
            "status": "drafting",
            "checkpoint": 0,
            "decision_interval_s": self.interval,
            "sim_time_s": 0.0,
            "next_checkpoint_t": 0.0,
            "waiting_for": [f for f, done in (("alpha", alpha_ready), ("beta", beta_ready)) if not done],
            "draft": {
                "alpha": {"ready": alpha_ready, "ships": min(6, 1 + int(frac * 8)),
                          "points_spent": min(196, int(frac * 320))},
                "beta": {"ready": beta_ready, "ships": min(5, 1 + int(frac * 6)),
                         "points_spent": min(188, int(frac * 210))},
            },
        }

    def battle_live_block(self, cur):
        ended = cur >= self.t_end
        cur = min(cur, self.t_end)
        checkpoint = int(cur // self.interval)
        return {
            "phase": "ended" if ended else "battle",
            "status": "ended" if ended else "running",
            "checkpoint": checkpoint,
            "decision_interval_s": self.interval,
            "sim_time_s": round(cur, 1),
            "next_checkpoint_t": (checkpoint + 1) * self.interval,
            "waiting_for": [],
            "draft": None,
        }

    # -------------------------------------------------------------- endpoints

    async def handle_recording(self, request):
        cur = self.cursor()
        if cur is None:
            return json_response({"live": self.draft_live_block(), "recording": None})

        ended = cur >= self.t_end
        cur = min(cur, self.t_end)
        try:
            since_t = float(request.query["since_t"])
        except (KeyError, ValueError):
            since_t = float("-inf")

        rec = dict(self.recording)
        rec["sim_trace"] = [f for f in self.frames if since_t < f["t"] <= cur]
        rec["events"] = [e for e in self.events if since_t < e["timestamp"] <= cur]
        if not ended:
            # A battle in progress has no verdict yet
            rec["winner"] = None
            rec["result_reason"] = None
            rec["duration_s"] = cur

        return json_response({"live": self.battle_live_block(cur), "recording": rec})

    async def handle_predictions(self, request):
        cur = self.cursor()
        if cur is None:
            return json_response({"t": 0, "t_checkpoint": 0, "ships": {}})
        if cur >= self.t_end or len(self.frames) < 1:
            cur = min(cur, self.t_end)
            return json_response({"t": round(cur, 1), "t_checkpoint": round(cur, 1), "ships": {}})

        i1 = max(0, bisect_right(self.frame_ts, cur) - 1)
        i0 = max(0, i1 - 1)
        f0, f1 = self.frames[i0], self.frames[i1]
        dt_f = f1["t"] - f0["t"]

        checkpoint = int(cur // self.interval)
        t_cp = (checkpoint + 1) * self.interval

        # Sample times every ~2s of sim time from the cursor to the
        # checkpoint, always including the checkpoint itself
        ts = []
        t = cur
        while t < t_cp:
            ts.append(t)
            t += 2.0
        ts.append(t_cp)

        ships = {}
        for ship_id, s1 in (f1.get("ships") or {}).items():
            if s1.get("destroyed") or s1.get("dying"):
                continue
            p1, v1 = s1["pos"], s1["vel"]
            s0 = (f0.get("ships") or {}).get(ship_id)
            if s0 and not s0.get("destroyed") and dt_f > 0:
                a = [(v1[k] - s0["vel"][k]) / dt_f for k in range(3)]
            else:
                a = [0.0, 0.0, 0.0]
            path = []
            for t in ts:
                d = t - f1["t"]
                path.append([p1[k] + v1[k] * d + 0.5 * a[k] * d * d for k in range(3)])
            ships[ship_id] = {"path": path, "checkpoint_pos": path[-1]}

        return json_response({"t": round(cur, 1), "t_checkpoint": t_cp, "ships": ships})


def json_response(data):
    resp = web.json_response(data)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording", help="Path to a saved battle recording JSON")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Sim seconds per wall-clock second (default 4)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--draft-seconds", type=float, default=0.0,
                        help="Serve a synthetic draft phase for N wall-clock seconds first")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Decision interval in sim seconds (default 30)")
    args = parser.parse_args()

    with open(args.recording) as f:
        recording = json.load(f)

    server = FakeLiveServer(recording, args.speed, args.draft_seconds, args.interval)
    app = web.Application()
    app.router.add_get("/live/recording", server.handle_recording)
    app.router.add_get("/live/predictions", server.handle_predictions)

    print(f"Fake live server: {args.recording} at {args.speed}x on :{args.port} "
          f"({len(server.frames)} frames, t_end={server.t_end}s, draft={args.draft_seconds}s)")
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
