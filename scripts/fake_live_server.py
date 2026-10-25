#!/usr/bin/env python3
"""Fake live-battle server: replays a saved recording as if it were live.

Dev harness for the viewer's live mode. Loads a saved BattleRecording JSON
and advances an internal sim-time cursor at --speed x wall-clock, serving
the same /live/... contract the real battle servers implement
(src/llm/mcp_http_server.py, scripts/run_live_draft_battle.py):

    GET /live/recording[?since_t=<float>&since_seq=<int>]
        -> {"live": {...}, "recording": {...}|null}
    GET /live/predictions
        -> {"t", "t_checkpoint", "ships": {...}, "stale"}   (external clients;
           the viewer draws tracks from the frames' recorded `plan`)

Like a real LLM battle, each checkpoint can hold the battle for
--pause-seconds while the "commanders deliberate": the status reads
'paused', the checkpoint's decision events trickle out by sequence, the
next model call per faction is reported in `pending_calls`, and the spend
grows towards the recording's billing total. --die-at / --exit-after-end
make the server vanish (crash, or a --no-linger exit) to exercise the
viewer's connection-lost handling.

Usage:
    uv run python scripts/fake_live_server.py visualizer/public/recordings/torpedo.json --speed 4
    uv run python scripts/fake_live_server.py <recording.json> --draft-seconds 20   # draft splash first
    uv run python scripts/fake_live_server.py <recording.json> --speed 2 --pause-seconds 25
"""

import argparse
import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from pathlib import Path

from aiohttp import web

# Recording-level keys that are not per-poll payload: trimmed like the real
# server does (assets go out in full once, then only what new events cite).
_STREAMED = ("sim_trace", "events", "assets")


class FakeLiveServer:
    def __init__(self, recording, speed, draft_seconds, decision_interval,
                 pause_seconds=0.0, battle_id=None, replay_url=None,
                 die_at=None, exit_after_end=None):
        self.recording = recording
        self.speed = speed
        self.draft_seconds = draft_seconds
        self.interval = decision_interval
        self.pause_seconds = max(0.0, pause_seconds)
        self.die_at = die_at
        self.exit_after_end = exit_after_end
        self.replay_url = replay_url
        self.frames = sorted(recording.get("sim_trace") or [], key=lambda f: f["t"])
        self.events = sorted(recording.get("events") or [],
                             key=lambda e: (e["timestamp"], e.get("sequence") or 0))
        for i, event in enumerate(self.events, 1):
            if not event.get('event_id'):
                event['event_id'] = f'legacy-live-{i}'
            if not event.get('sequence'):
                event['sequence'] = i
        self.event_ts = [e["timestamp"] for e in self.events]
        self.frame_ts = [f["t"] for f in self.frames]
        self.t_end = self.frame_ts[-1] if self.frame_ts else 0.0
        self.assets = recording.get("assets") or {}
        self.base = {k: v for k, v in recording.items() if k not in _STREAMED}
        self.start = time.monotonic()
        self.battle_id = battle_id or hashlib.sha1(
            f"{id(recording)}|{time.time()}".encode()).hexdigest()[:12]
        prov = recording.get("provenance") or {}
        self.billing = prov.get("billing") or {}
        self.draft_summary = prov.get("draft") or {}
        self.models = self.billing.get("models") or {
            f: (recording.get(f"{f}_model") or None) for f in ("alpha", "beta")}
        self.model_calls_total = sum(1 for e in self.events if e["event_type"] == "model_call")
        # Checkpoints inside the recording (the first decision follows one
        # opening interval of coasting)
        self.checkpoints = []
        k = 1
        while k * self.interval < self.t_end - 1e-9:
            self.checkpoints.append(k * self.interval)
            k += 1
        self.ended_wall = None

    # ------------------------------------------------------------- clock

    def clock(self):
        """(cursor, pause) where pause is None or (checkpoint_t, wall_s_into_pause).
        cursor is None while the synthetic draft phase runs."""
        w = (time.monotonic() - self.start) - self.draft_seconds
        if w < 0:
            return None, None
        t = 0.0
        for cp in self.checkpoints:
            run_wall = (cp - t) / self.speed
            if w < run_wall:
                return t + w * self.speed, None
            w -= run_wall
            t = cp
            if w < self.pause_seconds:
                return cp, (cp, w)
            w -= self.pause_seconds
        return min(self.t_end, t + w * self.speed), None

    def cursor(self):
        """Sim-time cursor, or None while the synthetic draft phase runs."""
        return self.clock()[0]

    def status_age(self, cur, pause):
        """Wall seconds in the current status (running segment or pause)."""
        if pause:
            return pause[1]
        prev_cp = max([0.0] + [c for c in self.checkpoints if c <= cur])
        return max(0.0, (cur - prev_cp) / self.speed)

    # ---------------------------------------------------- event release

    def released_events(self, cur, pause):
        """Events visible at this cursor: everything before it, and at a
        paused checkpoint only the first part of that instant's events (by
        sequence), released evenly across the pause."""
        hi = bisect_right(self.event_ts, cur + 1e-9)
        if not pause or self.pause_seconds <= 0:
            return self.events[:hi], []
        cp, into = pause
        lo = bisect_right(self.event_ts, cp - 1e-9)
        at_cp = self.events[lo:hi]
        # Hold back a little so the last orders land just before the resume
        frac = min(1.0, into / (self.pause_seconds * 0.9))
        n = int(math.floor(frac * len(at_cp)))
        return self.events[:lo] + at_cp[:n], at_cp[n:]

    def pending_calls(self, held, pause):
        """Next unreleased model call per faction, aged since the last release."""
        if not pause:
            return []
        per_faction = {}
        for e in held:
            d = e.get("data") or {}
            if e["event_type"] == "model_call" and d.get("faction") not in per_faction:
                per_faction[d.get("faction")] = {
                    "actor": d.get("actor"), "faction": d.get("faction"),
                    "ship_id": e.get("ship_id"), "model": d.get("model"),
                    "age_s": round(min(pause[1], 4.0 + (pause[1] % 9.0)), 1)}
        return [per_faction[f] for f in sorted(per_faction, key=str)]

    def spend(self, released):
        if not self.billing:
            return None
        done = sum(1 for e in released if e["event_type"] == "model_call")
        frac = done / self.model_calls_total if self.model_calls_total else 1.0
        return {
            "reported_usd": round(float(self.billing.get("reported_cost_usd") or 0) * frac, 4),
            "committed_usd": round(float(self.billing.get("committed_cost_usd") or 0) * frac, 4),
            "limit_usd": self.billing.get("spend_limit_usd"),
            "calls": int(round((self.billing.get("calls") or 0) * frac)),
            "failures": int(round((self.billing.get("failures") or 0) * frac)),
            "retries": int(round((self.billing.get("retries") or 0) * frac)),
            "truncated": int(round((self.billing.get("truncated") or 0) * frac)),
            "mode": "fake",
        }

    # ------------------------------------------------------------- live block

    def _common(self):
        return {"battle_id": self.battle_id, "models": self.models,
                "replay_url": self.replay_url, "decision_interval_s": self.interval}

    def draft_live_block(self):
        elapsed = time.monotonic() - self.start
        frac = elapsed / self.draft_seconds if self.draft_seconds else 1.0
        ready = {"alpha": frac >= 0.6, "beta": frac >= 0.9}
        both = all(ready.values())
        draft = {}
        for f in ("alpha", "beta"):
            summary = self.draft_summary.get(f) or {}
            fleet = (self.recording.get(f"{f}_fleet") or {}).get("ships") or []
            ships = summary.get("ships") or [
                {"ship_name": s.get("ship_name"), "ship_type": s.get("ship_type"), "cost": None,
                 "offset_km": [0.0, 0.0, 0.0]} for s in fleet]
            roster = [{"name": s.get("ship_name"), "type": s.get("ship_type"),
                       "cost": s.get("cost"), "offset_km": s.get("offset_km") or [0.0, 0.0, 0.0]}
                      for s in ships] if ready[f] else []
            spent = summary.get("points_spent") or sum(s.get("cost") or 0 for s in ships)
            draft[f] = {
                "ready": ready[f], "is_mcp": False,
                "ships": len(ships) if ready[f] else 0,
                "points_spent": spent if ready[f] else 0,
                "ship_names": [r["name"] for r in roster],
                "formation_name": (summary.get("formation_name") or "") if ready[f] else "",
                "budget": summary.get("budget") or self.billing.get("points_per_side"),
                "model": self.models.get(f), "auto": False,
                "elapsed_s": round(elapsed, 1),
                "ready_after_s": round(self.draft_seconds * (0.6 if f == "alpha" else 0.9), 1)
                if ready[f] else None,
                "roster": roster,
                "selection_rationale": summary.get("selection_rationale") if both else None,
                "formation_rationale": summary.get("formation_rationale") if both else None,
            }
        return {
            **self._common(),
            "phase": "draft",
            "status": "drafting",
            "checkpoint": 0,
            "sim_time_s": 0.0,
            "next_checkpoint_t": 0.0,
            "waiting_for": [f for f, done in ready.items() if not done],
            "draft": draft,
            "status_age_s": round(elapsed, 1),
            "pending_calls": [{"actor": "admiral", "faction": f, "ship_id": None,
                               "model": self.models.get(f), "age_s": round(elapsed, 1)}
                              for f, done in ready.items() if not done],
            "spend": None, "stage": "Drafting fleets and formations", "error": None,
            "result": None,
        }

    def battle_live_block(self, cur, pause, released, held):
        ended = cur >= self.t_end and not pause
        checkpoint = int(cur // self.interval + 1e-9)
        stopped = any(e["event_type"] == "battle_stopped" for e in released)
        if ended:
            stage = "Battle stopped" if stopped else "Battle complete"
        else:
            stage = "Admirals and captains deciding" if pause else "Simulating orders"
        pending = self.pending_calls(held, pause)
        return {
            **self._common(),
            "phase": "ended" if ended else "battle",
            "status": "ended" if ended else ("paused" if pause else "running"),
            "checkpoint": checkpoint,
            "sim_time_s": round(cur, 1),
            "next_checkpoint_t": (math.floor(cur / self.interval + 1e-9) + 1) * self.interval,
            "waiting_for": sorted({c["faction"] for c in pending if c["faction"]}),
            "draft": None,
            "status_age_s": round(self.status_age(cur, pause), 1),
            "pending_calls": pending,
            "spend": self.spend(released),
            "stage": stage,
            "error": self.recording.get("result_reason") if (ended and stopped) else None,
            "result": {"winner": self.recording.get("winner"),
                       "reason": self.recording.get("result_reason")} if ended else None,
        }

    # -------------------------------------------------------------- endpoints

    def _maybe_die(self, cur, ended):
        if self.die_at is not None and cur is not None and cur >= self.die_at:
            print(f"--die-at {self.die_at}: exiting at sim t={cur:.1f}", flush=True)
            os._exit(3)
        if ended and self.exit_after_end is not None:
            if self.ended_wall is None:
                self.ended_wall = time.monotonic()
            if time.monotonic() - self.ended_wall >= self.exit_after_end:
                print("--exit-after-end: battle over, exiting", flush=True)
                os._exit(0)

    async def handle_recording(self, request):
        cur, pause = self.clock()
        if cur is None:
            return json_response({"live": self.draft_live_block(), "recording": None})
        ended = cur >= self.t_end and not pause
        self._maybe_die(cur, ended)
        try:
            since_t = float(request.query["since_t"])
        except (KeyError, ValueError):
            since_t = float("-inf")
        has_seq = "since_seq" in request.query
        since_seq = int(request.query.get('since_seq', -1))

        released, held = self.released_events(cur, pause)
        rec = dict(self.base)
        lo = bisect_right(self.frame_ts, since_t)
        hi = bisect_right(self.frame_ts, cur + 1e-9)
        rec["sim_trace"] = self.frames[lo:hi]
        rec["events"] = [e for e in released if
                         (e['sequence'] > since_seq if has_seq else since_t < e['timestamp'])]
        if has_seq:
            refs = set()
            for e in rec["events"]:
                data = e.get("data") or {}
                refs.update(data.get("message_refs") or [])
                if data.get("tools_ref"):
                    refs.add(data["tools_ref"])
            rec["assets"] = {k: self.assets[k] for k in refs if k in self.assets}
        else:
            rec["assets"] = self.assets
        if not ended:
            # A battle in progress has no verdict yet (and no final billing)
            rec["winner"] = None
            rec["result_reason"] = None
            rec["duration_s"] = cur
            prov = dict(rec.get("provenance") or {})
            prov.pop("billing", None)
            rec["provenance"] = prov
        live = self.battle_live_block(cur, pause, released, held)
        return json_response({"live": live, "recording": rec})

    async def handle_predictions(self, request):
        cur = self.cursor()
        if cur is None:
            return json_response({"t": 0, "t_checkpoint": 0, "ships": {}, "stale": False})
        if cur >= self.t_end or len(self.frames) < 1:
            cur = min(cur, self.t_end)
            return json_response({"t": round(cur, 1), "t_checkpoint": round(cur, 1), "ships": {},
                                  "stale": False})

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
            # Recordings with trajectory prediction carry the engine's own
            # look-ahead for this frame: serve it as the live server does.
            plan = s1.get("plan")
            if plan and plan.get("path"):
                t_cp = plan.get("t_cp", t_cp)
                ships[ship_id] = {"path": plan["path"], "checkpoint_pos": plan["path"][-1],
                                  "sigma": plan.get("sigma", 0.0), "dt": plan.get("dt"),
                                  "stale": bool(plan.get("stale", False))}
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
            ships[ship_id] = {"path": path, "checkpoint_pos": path[-1], "sigma": 0.0,
                              "stale": False}

        return json_response({"t": round(cur, 1), "t_checkpoint": t_cp, "ships": ships,
                              "stale": any(s["stale"] for s in ships.values())})


def json_response(data):
    resp = web.json_response(data)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def default_replay_url(path):
    """'/recordings/<name>' when the file is served by the viewer's dev server."""
    p = Path(path).resolve()
    return f"/recordings/{p.name}" if p.parent.name == "recordings" and \
        p.parent.parent.name == "public" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording", help="Path to a saved battle recording JSON")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Sim seconds per wall-clock second (default 4)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--draft-seconds", type=float, default=0.0,
                        help="Serve a synthetic draft phase for N wall-clock seconds first")
    parser.add_argument("--pause-seconds", type=float, default=0.0,
                        help="Hold each checkpoint for N wall seconds of commander "
                             "deliberation (status 'paused', decisions trickle out)")
    parser.add_argument("--interval", type=float, default=None,
                        help="Decision interval in sim seconds (default: the "
                             "recording's decision_interval_s, else 30)")
    parser.add_argument("--battle-id", default=None,
                        help="Battle identity served to the viewer (default: random per start)")
    parser.add_argument("--replay-url", default=None,
                        help="Saved-replay URL offered when the server goes away "
                             "(default: /recordings/<file> for files under public/recordings)")
    parser.add_argument("--die-at", type=float, default=None,
                        help="Exit abruptly once the sim cursor reaches T (crash test)")
    parser.add_argument("--exit-after-end", type=float, default=None,
                        help="Exit N wall seconds after the battle ends (0 = before any "
                             "client sees the end, like a server without linger)")
    args = parser.parse_args()

    with open(args.recording) as f:
        recording = json.load(f)
    if args.interval is None:
        prov = recording.get("provenance") or {}
        args.interval = float(recording.get("decision_interval_s")
                              or prov.get("decision_interval_s") or 30.0)
    replay_url = args.replay_url if args.replay_url is not None else default_replay_url(args.recording)

    server = FakeLiveServer(recording, args.speed, args.draft_seconds, args.interval,
                            pause_seconds=args.pause_seconds, battle_id=args.battle_id,
                            replay_url=replay_url, die_at=args.die_at,
                            exit_after_end=args.exit_after_end)
    app = web.Application()
    app.router.add_get("/live/recording", server.handle_recording)
    app.router.add_get("/live/predictions", server.handle_predictions)

    print(f"Fake live server: {args.recording} at {args.speed}x on :{args.port} "
          f"({len(server.frames)} frames, t_end={server.t_end}s, draft={args.draft_seconds}s, "
          f"pause={args.pause_seconds}s, battle_id={server.battle_id})", flush=True)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
