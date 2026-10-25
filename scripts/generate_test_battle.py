#!/usr/bin/env python3
"""
Generate battle recordings with scripted (non-LLM) captains.

The replay viewer needs recordings that exercise every mechanic the sim can
produce - torpedo salvos, PD dwell, coilgun exchanges, evasion, destruction -
without paying for LLM calls or hoping two captains improvise the right
battle. Each scenario here is a fixed per-checkpoint list of the same tool
calls a real captain would emit, executed through LLMCaptain._execute_tool so
the command path (and therefore the recording) is identical to a live battle.

Usage:
    uv run python scripts/generate_test_battle.py torpedo_strike
    uv run python scripts/generate_test_battle.py gun_duel
    uv run python scripts/generate_test_battle.py all

    # Fleet fixture for the spectator HUD: heuristic captains plus a scripted
    # command net in the pass-4 comms schema (comms_message / comms_delivered /
    # captain_report) and every legacy comms event the viewer still reads
    uv run python scripts/generate_test_battle.py hud_comms \
        --out visualizer/public/recordings/p4_hud_comms_fixture.json
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.prompts import CaptainPersonality
from src.llm.battle_runner import LLMBattleRunner, BattleConfig, load_fleet_data

# A script maps checkpoint number (1-based, the value decide() sees as
# decision_count + 1) to the tool calls to issue at that checkpoint.
Script = Dict[int, List[Tuple[str, Dict[str, Any]]]]


class ScriptedCaptain(LLMCaptain):
    """An LLMCaptain whose decisions come from a fixed script, not an API."""

    def __init__(self, config: LLMCaptainConfig, script: Script):
        super().__init__(config, client=None)  # client is never used
        self.script = script

    def select_personality(self, distance_km: float, verbose: bool = False) -> Dict[str, Any]:
        return {}

    def decide(self, ship_id: str, simulation: Any) -> List[Any]:
        if self.has_surrendered:
            return []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return []

        # Same lazy weapon-group setup as the real decide(); set_weapons_order
        # silently matches nothing without it.
        if not self.weapon_groups and self.config.fleet_data and self.config.ship_type:
            self.setup_weapon_groups(self.config.ship_type, self.config.fleet_data)

        checkpoint = self.decision_count + 1
        calls = self.script.get(checkpoint, [])

        commands: List[Any] = []
        tool_calls = [
            ToolCall(id=f"scripted_{checkpoint}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ]
        for tc in tool_calls:
            cmd = self._execute_tool(tc, simulation, ship_id)
            if cmd is not None:
                if isinstance(cmd, list):
                    commands.extend(c for c in cmd if c is not None)
                else:
                    commands.append(cmd)

        self.decision_count += 1
        self.last_tool_calls = tool_calls
        self.decision_history.append({
            "checkpoint": self.decision_count,
            "time": simulation.current_time,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in tool_calls],
            "commands_count": len(commands),
        })
        return commands


class ScriptedBattleRunner(LLMBattleRunner):
    """Battle runner that swaps in scripted captains after normal setup."""

    def __init__(self, *args, alpha_script: Script, beta_script: Script, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_script = alpha_script
        self._beta_script = beta_script

    def setup_battle(self, fleet_data: Dict[str, Any]) -> None:
        super().setup_battle(fleet_data)
        self.alpha_captain = ScriptedCaptain(self.alpha_config, self._alpha_script)
        self.beta_captain = ScriptedCaptain(self.beta_config, self._beta_script)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def torpedo_strike() -> Tuple[BattleConfig, LLMCaptainConfig, LLMCaptainConfig, Script, Script]:
    """
    Two-sided torpedo exchange: a 4-PD torpedo cruiser against an agile
    corvette. The corvette's launches fly into massed PD (3+ turrets is the
    doctrine threshold for blinding a seeker) -> intercepted / disabled
    torpedoes; the cruiser's launches chase a permanently evading 3g corvette
    -> misses and fuel exhaustion, with enough saturation late to land hits.

    Doctrine note: EVADE lasts one 30s decision window, so it must be
    re-issued every checkpoint to stay in effect.
    """
    config = BattleConfig(
        initial_distance_km=500.0,
        time_limit_s=900.0,
        max_checkpoints=20,
        alpha_ship_type="cruiser_torpedo",
        beta_ship_type="corvette",
        verbose=True,
        personality_selection=False,
        record_battle=True,
        record_sim_trace=True,
        seed=42,
    )
    alpha_cfg = LLMCaptainConfig(
        name="Captain Script-A",
        ship_name="TIS Harpoon",
        model="scripted-torpedo",
        personality=CaptainPersonality.BALANCED,
    )
    beta_cfg = LLMCaptainConfig(
        name="Captain Script-B",
        ship_name="HFS Dartfish",
        model="scripted-defender",
        personality=CaptainPersonality.BALANCED,
    )

    # Alpha keeps thrusting the whole time so blinded corvette rounds - whose
    # guidance froze on a stale solution - miss and chase into fuel
    # exhaustion. The launch ramp starts as single probing rounds (the
    # corvette's evasion and PD heat get a chance to matter) and only
    # saturates late, so the finale still lands.
    alpha_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 1}),
            ("send_message", {"message": "Tube one away. Your move, Dartfish."})],
        2: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 1})],
        3: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        4: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        5: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 2})],
        6: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        7: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
            ("launch_torpedo", {"count": 2})],
        8: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 4})],
        9: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
            ("launch_torpedo", {"count": 4})],
        10: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
             ("launch_torpedo", {"count": 4})],
        11: [("launch_torpedo", {"count": 4})],
        12: [("launch_torpedo", {"count": 4})],
        13: [("launch_torpedo", {"count": 4})],
    }
    # The corvette never stops dancing, and spends its whole 8-round magazine
    # early so its torpedoes are in flight while the cruiser's PD is fresh.
    beta_script: Script = {
        cp: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})]
        for cp in range(1, 21)
    }
    beta_script[1] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2}),
                      ("set_radiators", {"extend": True})]
    beta_script[2] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2}),
                      ("send_message", {"message": "Nice fireworks. Watch mine fly."})]
    beta_script[3] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2})]
    beta_script[4] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2})]
    return config, alpha_cfg, beta_cfg, alpha_script, beta_script


def gun_duel() -> Tuple[BattleConfig, LLMCaptainConfig, LLMCaptainConfig, Script, Script]:
    """
    Two destroyers close and slug it out with coilguns. Exercises:
    shot_fired / hit / miss, pd_fired dwell on slugs, pd_slug_damaged /
    destroyed, armor_damage, penetration, module damage, destruction.
    """
    config = BattleConfig(
        initial_distance_km=350.0,
        time_limit_s=900.0,
        max_checkpoints=24,
        alpha_ship_type="destroyer",
        beta_ship_type="destroyer",
        verbose=True,
        personality_selection=False,
        record_battle=True,
        record_sim_trace=True,
        seed=7,
    )
    alpha_cfg = LLMCaptainConfig(
        name="Captain Script-A",
        ship_name="TIS Broadside",
        model="scripted-gunner",
        personality=CaptainPersonality.BALANCED,
    )
    beta_cfg = LLMCaptainConfig(
        name="Captain Script-B",
        ship_name="HFS Riposte",
        model="scripted-gunner",
        personality=CaptainPersonality.BALANCED,
    )

    guns_free = ("set_weapons_order", {"spinal_mode": "FIRE_WHEN_OPTIMAL",
                                       "turret_mode": "FIRE_WHEN_OPTIMAL",
                                       "spinal_min_probability": 0.2,
                                       "turret_min_probability": 0.2})
    alpha_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}), guns_free,
            ("send_message", {"message": "Riposte, this is Broadside. Guns free."})],
        3: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})],
        4: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.6})],
        6: [("set_maneuver", {"maneuver_type": "PADLOCK"})],
        9: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0})],
    }
    beta_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}), guns_free],
        2: [("set_radiators", {"extend": True})],
        4: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})],
        5: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0})],
        8: [("set_maneuver", {"maneuver_type": "PADLOCK"})],
    }
    return config, alpha_cfg, beta_cfg, alpha_script, beta_script


SCENARIOS = {
    "torpedo_strike": torpedo_strike,
    "gun_duel": gun_duel,
}


# ---------------------------------------------------------------------------
# Fleet fixture: a scripted command net for the spectator HUD
# ---------------------------------------------------------------------------

# Deliberately carries markup: the viewer must show it as text, never parse it.
HOSTILE_TAUNT = '<img src=x onerror="alert(1)"> <b>Strike your colours</b>, {target}. This channel is ours.'

PLANS = {
    "alpha": [
        "Phase 1: close as one formation to ~250 km, PD bubbles overlapping. Phase 2: synchronized "
        "salvo on the nearest hull to saturate its point defense. Phase 3: roll fire onto the next "
        "nearest; anyone below 40% hull breaks off and screens.",
        "Amended at CP4: their screen is thinning. Keep the pressure on whatever is closest, pair "
        "ships so every target takes rounds from two bearings, and do not chase a dying hulk.",
    ],
    "beta": [
        "Hold a tight wall and let them come to us: guns free inside 200 km, torpedoes only on a "
        "closing target, heaviest armor toward the threat axis.",
        "Amended at CP4: losses accepted - collapse the wall onto the flagship and trade fire "
        "at the shortest range we can get.",
    ],
}
DIRECTIVES = {
    "alpha": [
        "All ships: close on {target} ({range:.0f} km) in formation. Hold torpedoes for the "
        "synchronized salvo; guns free when optimal.",
        "Salvo window: every launcher fires on {target} this checkpoint, time on target together. "
        "Then evade inside the PD umbrella.",
        "Keep {target} under fire ({range:.0f} km). Damaged ships break off and screen; nobody "
        "closes alone.",
    ],
    "beta": [
        "Wall up and present armor toward {target} ({range:.0f} km). Guns free inside 200 km.",
        "{target} is leading their formation - concentrate everything on it.",
        "Hold the line. Retarget onto {target}; torpedoes only on a closing target.",
    ],
}


class HudCommsRunner(LLMBattleRunner):
    """
    Heuristic-captain fleet battle (no LLM, no admirals) that also records a
    scripted command net around every checkpoint, in the pass-4 comms
    schema the viewer's COMMAND NET, kill feed and decision panel read.

    Before the captains decide: admiral_plan / admiral_directive /
    admiral_order (the legacy admiral events), engine auto-signals and an
    admiral_to_captain message (comms_message), and comms_delivered for
    every message whose delivery time has come. After: captain_report
    (ACK / DEVIATE / UNABLE with the order-vs-execution compliance block),
    fleet_net contact reports, a direct message, a captain_to_admiral
    request, enemy broadcasts (one carrying HTML, to prove the viewer
    escapes it), admiral diplomacy, a message that is never delivered, a
    legacy 'message' event, a captain_admiral_discussion and captain logs.

    The words are synthetic; ships, ranges, launches, kills and timing are
    the real battle. Nothing here touches the battle RNG.
    """

    def setup_fleet_battle(self, fleet_data: Dict[str, Any]) -> None:
        super().setup_fleet_battle(fleet_data)
        self._net = {"msg": 0, "report": 0, "pending": [], "orders": {}, "last_t": 0.0}

    # -- helpers ------------------------------------------------------------

    def _fleet(self, faction: str) -> Dict[str, Any]:
        return self.alpha_ships if faction == "alpha" else self.beta_ships

    def _alive(self, faction: str) -> List[str]:
        return [sid for sid, ship in self._fleet(faction).items()
                if not ship.is_destroyed and not getattr(ship, "is_dying", False)]

    def _name(self, ship_id: Optional[str]) -> str:
        ship = self.simulation.get_ship(ship_id) if ship_id else None
        return getattr(ship, "name", None) or ship_id or "?"

    def _nearest(self, ship_id: str, faction: str) -> Tuple[Optional[str], float]:
        other = "beta" if faction == "alpha" else "alpha"
        me = self.simulation.get_ship(ship_id)
        best, best_d = None, float("inf")
        for sid in self._alive(other):
            d = me.position.distance_to(self.simulation.get_ship(sid).position)
            if d < best_d:
                best, best_d = sid, d
        return best, best_d / 1000.0

    def _launches(self, ship_ids, since: float, until: float) -> List[Any]:
        return [e for e in self.recorder.events
                if e.event_type == "torpedo_launched" and e.ship_id in ship_ids and since < e.timestamp <= until]

    def _msg(self, t, cp, channel, sender_id, role, sender_name, faction, spec, recipients, text,
             kind="message", delay=0.0, structured=None, blocked=False) -> str:
        self._net["msg"] += 1
        msg_id = f"m{self._net['msg']:03d}"
        self.recorder.record(
            t, "comms_message", sender_id, msg_id=msg_id, channel=channel, kind=kind,
            sender_id=sender_id, sender_role=role, sender_name=sender_name, faction=faction,
            recipient_spec=spec, recipient_ids=list(recipients), text=text,
            structured=structured or {}, checkpoint=cp, sent_t=t, deliver_at_t=t + delay)
        self._net["pending"].append({"msg_id": msg_id, "recipients": list(recipients),
                                     "deliver_at": t + delay, "blocked": blocked})
        return msg_id

    def _deliver(self, t: float, cp: int) -> None:
        keep = []
        for p in self._net["pending"]:
            if p["deliver_at"] > t + 1e-6:
                keep.append(p)
                continue
            if p["blocked"]:
                continue  # jammed: never delivered
            for rid in p["recipients"]:
                ship = self.simulation.get_ship(rid)
                if ship is None or ship.is_destroyed:
                    continue  # a lost ship never hears it
                self.recorder.record(t, "comms_delivered", rid, msg_id=p["msg_id"],
                                     recipient_id=rid, delivered_t=t, checkpoint=cp)
        self._net["pending"] = keep

    # -- checkpoint hook ------------------------------------------------------

    def _run_captain_decisions(self, captain_entries, admiral_orders, *args, **kwargs):
        record = kwargs.get("record", args[0] if args else True)
        if not (record and self.recorder):
            return super()._run_captain_decisions(captain_entries, admiral_orders, *args, **kwargs)
        t, cp = self.simulation.current_time, self.checkpoint_count
        self._before_decisions(t, cp)
        result = super()._run_captain_decisions(captain_entries, admiral_orders, *args, **kwargs)
        self._after_decisions(t, cp, captain_entries)
        self._net["last_t"] = t
        return result

    def _before_decisions(self, t: float, cp: int) -> None:
        rec = self.recorder
        for faction in ("alpha", "beta"):
            alive = self._alive(faction)
            if not alive:
                continue
            admiral = f"Admiral {faction.title()}"
            lead_target, lead_range = self._nearest(alive[0], faction)
            if lead_target is None:
                continue
            if cp in (1, 4):
                rec.record_admiral_plan(t, admiral, faction, PLANS[faction][0 if cp == 1 else 1])
            directive = DIRECTIVES[faction][(cp - 1) % 3].format(target=self._name(lead_target), range=lead_range)
            rec.record_admiral_directive(t, admiral, faction, directive)
            for i, sid in enumerate(alive):
                target, rng = self._nearest(sid, faction)
                ship = self.simulation.get_ship(sid)
                has_torps = bool(getattr(ship, "torpedo_launchers", None) or getattr(ship, "torpedo_launcher", None))
                salvo = 2 if has_torps and cp % 3 == 2 else None
                maneuver = "EVADE" if ship.hull_integrity < 50 else "INTERCEPT"
                priority = "CRITICAL" if cp == 2 and i == 0 else "HIGH" if salvo else "NORMAL"
                text = (f"{maneuver} toward {self._name(target)} ({rng:.0f} km). "
                        + (f"Launch {salvo} Tridents at {self._name(target)} with the fleet salvo. " if salvo else "")
                        + ("Guns free when optimal." if not salvo else "Hold formation."))
                rec.record_admiral_order(t, admiral, sid, self._name(sid), text, priority=priority,
                                         suggested_target=self._name(target), torpedo_salvo=salvo,
                                         torpedo_target=target if salvo else None, faction=faction)
                self._net["orders"][sid] = {"maneuver": maneuver, "target": target, "salvo": salvo}
            # Engine auto-signal: what the fleet launched since the last checkpoint
            launches = self._launches(set(self._fleet(faction)), self._net["last_t"], t)
            if launches:
                targets: Dict[str, int] = {}
                for e in launches:
                    name = self._name(e.data.get("target_id"))
                    targets[name] = targets.get(name, 0) + 1
                summary = ", ".join(f"{n} at {name}" for name, n in targets.items())
                self._msg(t, cp, "fleet_net", None, "engine", "Fleet systems", faction, "all_friendlies", alive,
                          f"{len(launches)} torpedo{'es' if len(launches) > 1 else ''} away since "
                          f"T+{self._net['last_t']:.0f}s: {summary}.",
                          kind="launch_summary", structured={"launches": len(launches), "targets": targets})
            if cp == 2:
                self._msg(t, cp, "admiral_to_captain", None, "admiral", admiral, faction, "ship", [alive[0]],
                          f"{self._name(alive[0])}: you anchor the salvo. Everyone times on your launch.", kind="order_note")
        self._deliver(t, cp)

    def _after_decisions(self, t: float, cp: int, entries) -> None:
        rec = self.recorder
        interval = self.fleet_config.decision_interval_s if self.fleet_config else 30.0
        decisions = {}
        for e in rec.events:
            if e.event_type == "captain_decision" and e.timestamp == t:
                decisions[e.ship_id] = e
        by_faction: Dict[str, List[Tuple[str, Any]]] = {"alpha": [], "beta": []}
        for sid, captain, faction in entries:
            by_faction[faction].append((sid, captain))
        for faction, members in by_faction.items():
            for i, (sid, captain) in enumerate(members):
                order = self._net["orders"].get(sid, {})
                decision = decisions.get(sid)
                executed = decision.data.get("maneuver_type") if decision else None
                actual = decision.data.get("target_id") if decision else None
                launched = len(self._launches({sid}, t - interval, t))
                self._net["report"] += 1
                status, reason, request = "ACK", "", None
                intent = f"{executed or 'Holding'} on {self._name(actual or order.get('target'))}."
                if (cp + 2 * i) % 7 == 3:
                    status = "DEVIATE"
                    buddy = members[i - 1][0] if i > 0 else (members[1][0] if len(members) > 1 else None)
                    reason = (f"Not closing on {self._name(order.get('target'))} this checkpoint: holding standoff"
                              + (f" to cover {self._name(buddy)}'s PD gap." if buddy else " until the salvo is ready."))
                elif order.get("salvo") and i % 2 == 1:
                    status = "UNABLE"
                    reason = "Launchers still cycling - salvo will be late by one checkpoint."
                    request = {"kind": "reload", "text": "Shift my salvo to the next checkpoint."}
                target, rng = self._nearest(sid, faction)
                rec.record(t, "captain_report", sid, report_id=f"r{self._net['report']:03d}", faction=faction,
                           checkpoint=cp, captain_name=captain.name, ship_name=self._name(sid), status=status,
                           reason=reason, intent=intent,
                           contact={"ship": self._name(target), "assessment": "closing" if rng < 300 else "holding",
                                    "note": f"{rng:.0f} km"} if target and i == 0 else None,
                           request=request,
                           compliance={"ordered_maneuver": order.get("maneuver"), "executed_maneuver": executed,
                                       "ordered_target": order.get("target"), "actual_target": actual,
                                       "ordered_salvo": order.get("salvo"), "launched": launched})
                if i == 0 and cp % 2 == 1:
                    rec.record_captain_log(t, sid, captain.name,
                                           f"CP{cp}: {executed or 'holding'}, hull {self.simulation.get_ship(sid).hull_integrity:.0f}%. "
                                           f"Watching {self._name(target)} at {rng:.0f} km.")
            alive = self._alive(faction)
            if not alive:
                continue
            other = "beta" if faction == "alpha" else "alpha"
            enemies = self._alive(other)
            target, rng = self._nearest(alive[0], faction)
            if len(alive) > 1 and target:
                self._msg(t, cp, "fleet_net", alive[0], "captain", self._name(alive[0]), faction, "all_friendlies",
                          alive[1:], f"Contact: {self._name(target)} at {rng:.0f} km. I have the lead; follow my salvo.",
                          kind="contact", delay=interval)
            if cp == 2 and len(alive) > 1:
                self._msg(t, cp, "direct", alive[1], "captain", self._name(alive[1]), faction, "ship", [alive[0]],
                          f"{self._name(alive[0])}, covering your six with PD. Call it if you need me closer.",
                          delay=interval)
                self._msg(t, cp, "captain_to_admiral", alive[-1], "captain", self._name(alive[-1]), faction, "admiral", [],
                          "Request permission to break formation and close on the nearest hull.",
                          kind="request", delay=interval)
            if cp == 2 and faction == "beta" and enemies:
                self._msg(t, cp, "enemy_broadcast", alive[0], "captain", self._name(alive[0]), faction, "enemy",
                          enemies, HOSTILE_TAUNT.format(target=self._name(enemies[0])), kind="taunt", delay=interval)
            if cp == 3 and faction == "alpha" and enemies:
                self._msg(t, cp, "admiral_diplomacy", None, "admiral", "Admiral Alpha", faction, "enemy_admiral", [],
                          "Admiral Beta: your screen is spent. Withdraw and we hold fire.", kind="diplomacy",
                          delay=interval)
            if cp == 3 and len(alive) > 1:
                self._msg(t, cp, "direct", alive[0], "captain", self._name(alive[0]), faction, "ship", [alive[-1]],
                          "Switching to the flank - this one is jammed and never arrives.", delay=interval, blocked=True)
        if cp == 1:
            beta = self._alive("beta")
            if beta:
                rec.record_message(t, beta[0], "Captain " + self._name(beta[0]).split()[-1], self._name(beta[0]),
                                   "Legacy channel check (pre-v4 message event): we see you coming.")
        if cp == 2:
            alpha = self._alive("alpha")
            if alpha:
                rec.record_captain_admiral_discussion(
                    t, alpha[0], "Captain " + self._name(alpha[0]).split()[-1], "Admiral Alpha",
                    "Order says hold the salvo, but the target is inside 250 km. Launch now?",
                    "Negative. Hold for the synchronized salvo; one ragged launch feeds their PD.", 1)


def hud_comms_battle(seed: int = 7, points: int = 100, max_ships: int = 6, distance_km: float = 300.0,
                     max_checkpoints: int = 12, recording_dir: Optional[str] = None,
                     verbose: bool = False) -> Tuple[Any, Optional[str]]:
    """
    Run the HUD command-net fixture battle; returns (result, recording file).
    Seed 7 at 300 km: 4 v 4, two beta ships crippled and destroyed (169 ->
    187 s and 256 -> 280 s), so the kill feed's dying countdown shows too.
    """
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import auto_draft, draft_to_fleet_definition
    from src.llm.heuristic_captain import HEURISTIC_MODEL

    drafts = {f: auto_draft(f, budget=points, max_ships=max_ships, seed=seed + (0 if f == "alpha" else 1))
              for f in ("alpha", "beta")}
    time_limit = max_checkpoints * 30.0 + 30.0
    fleet_config = BattleFleetConfig(
        battle_name=f"HUD comms fixture ({points} pts, seed {seed})",
        alpha_fleet=draft_to_fleet_definition(drafts["alpha"], distance_km, captain_model=HEURISTIC_MODEL),
        beta_fleet=draft_to_fleet_definition(drafts["beta"], distance_km, captain_model=HEURISTIC_MODEL),
        time_limit_s=time_limit, initial_distance_km=distance_km,
        record_battle=True, record_sim_trace=True, personality_selection=False)
    config = BattleConfig(
        initial_distance_km=distance_km, time_limit_s=time_limit, max_checkpoints=max_checkpoints,
        verbose=verbose, personality_selection=False, record_battle=True, record_sim_trace=True,
        record_predictions=True, recording_dir=recording_dir or "data/recordings", seed=seed)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = HudCommsRunner(config=config, alpha_config=dummy, beta_config=dummy, client=None,
                            fleet_config=fleet_config)
    result = runner.run_battle(load_fleet_data())
    return result, result.recording_file


FLEET_SCENARIOS = {
    "hud_comms": hud_comms_battle,
}


def run_scenario(name: str, recording_dir: Optional[str] = None,
                 predictions: bool = True) -> Optional[str]:
    config, alpha_cfg, beta_cfg, alpha_script, beta_script = SCENARIOS[name]()
    if recording_dir:
        config.recording_dir = recording_dir
    config.record_predictions = predictions
    fleet_data = load_fleet_data()

    runner = ScriptedBattleRunner(
        config=config,
        alpha_config=alpha_cfg,
        beta_config=beta_cfg,
        client=None,  # scripted captains never call an LLM
        alpha_script=alpha_script,
        beta_script=beta_script,
    )
    result = runner.run_battle(fleet_data)

    print(f"\n=== {name} ===")
    print(f"Winner: {result.winner or 'Draw'} - {result.reason}")
    print(f"Duration: {result.duration_s:.0f}s, checkpoints: {result.checkpoints_used}")
    print(f"Recording: {result.recording_file}")
    return result.recording_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SCENARIOS, *FLEET_SCENARIOS, "all"])
    parser.add_argument("--recording-dir", default=None,
                        help="Where to save recordings (default data/recordings)")
    parser.add_argument("--no-predictions", action="store_true",
                        help="Skip the per-frame predicted path (sim_trace 'plan')")
    parser.add_argument("--seed", type=int, default=7, help="hud_comms: draft + combat seed")
    parser.add_argument("--out", default=None,
                        help="hud_comms: also copy the recording here (e.g. "
                             "visualizer/public/recordings/p4_hud_comms_fixture.json)")
    args = parser.parse_args()

    if args.scenario in FLEET_SCENARIOS:
        result, path = FLEET_SCENARIOS[args.scenario](seed=args.seed, recording_dir=args.recording_dir)
        print(f"\n=== {args.scenario} ===")
        print(f"Winner: {result.winner or 'Draw'} - {result.reason}")
        print(f"Duration: {result.duration_s:.0f}s, checkpoints: {result.checkpoints_used}")
        print(f"Recording: {path}")
        if args.out and path:
            import shutil
            shutil.copyfile(path, args.out)
            print(f"Copied to {args.out}")
        return 0

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        run_scenario(name, args.recording_dir, predictions=not args.no_predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
