"""
Compact, human-readable battle ticker for people following a battle as text.

One block per draft side, one block per checkpoint and one result block, each
starting with a greppable header line such as

    === CP 5 | t=150s | spend $0.84/$4.50 ===

Blocks go to stdout (flushed) and are appended to a log file. The ticker only
reads battle state (simulation ships, recorded events); it never changes it.
"""
from collections import Counter
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

TYPE_ABBR = {
    "corvette": "CO", "frigate": "FF", "destroyer": "DD", "cruiser": "CA",
    "cruiser_torpedo": "CT", "battlecruiser": "BC", "battleship": "BB",
    "dreadnought": "DN", "dreadnought_siege": "DS",
}
FACTIONS = ("alpha", "beta")
DIRECTIVE_CHARS = 140
MAX_HIT_GROUPS = 5


def short_name(name: str) -> str:
    """'TIS Harpoon-1' -> 'Harpoon-1'."""
    parts = str(name or "?").split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0].isupper() else str(name)


def type_abbr(ship_type: str) -> str:
    return TYPE_ABBR.get(ship_type, str(ship_type)[:2].upper())


def clip(text: Any, limit: int = DIRECTIVE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def model_label(model: Optional[str]) -> str:
    return str(model or "heuristic").split("/")[-1]


class BattleTicker:
    """Formats and emits ticker blocks; tracks the event cursor between checkpoints."""

    def __init__(
        self,
        log_path: Any = None,
        spend: Callable[[], Tuple[float, float]] = lambda: (0.0, 0.0),
        limit_usd: Optional[float] = None,
        mode: str = "",
        models: Optional[Dict[str, str]] = None,
        stream: Any = None,
    ):
        self.log_path = log_path
        self.spend = spend
        self.limit_usd = limit_usd
        self.mode = mode
        self.models = models or {}
        self.stream = stream
        self._lock = threading.Lock()
        self._cursor = 0
        self._death_times: Dict[str, float] = {}
        self._projectile_weapons: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def emit(self, lines: Iterable[str]) -> str:
        text = "\n".join(lines)
        with self._lock:
            print(text, file=self.stream or sys.stdout, flush=True)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as log:
                    log.write(text + "\n")
        return text

    def spend_text(self) -> str:
        reported, committed = self.spend()
        if self.limit_usd is None:
            text = f"spend ${reported:.2f}"
        else:
            text = f"spend ${reported:.2f}/${self.limit_usd:.2f}"
        if committed - reported > 0.005:
            text += f" (+${committed - reported:.2f} held)"
        return text + (f" ({self.mode})" if self.mode else "")

    def header(self, *parts: str) -> str:
        return "=== " + " | ".join([*parts, self.spend_text()]) + " ==="

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def start(self, name: str, points: int, max_ships: int, turns: int,
              costs: Dict[str, int], extra: str = "") -> str:
        vs = " vs ".join(model_label(self.models.get(f)) for f in FACTIONS)
        catalog = " ".join(f"{type_abbr(t)}={t}:{c}" for t, c in costs.items())
        lines = [self.header(f"BATTLE {name}", vs),
                 f"{points} pts/side, max {max_ships} ships, {turns} turns of 30s. {extra}".rstrip(),
                 f"Costs: {catalog}"]
        return self.emit(lines)

    def draft(self, faction: str, summary: Dict[str, Any]) -> str:
        ships = " | ".join(
            f"{short_name(s.get('ship_name'))} {type_abbr(s.get('ship_type'))} {s.get('cost')}"
            for s in summary.get("ships", []))
        lines = [
            self.header(f"DRAFT {faction.upper()}", model_label(self.models.get(faction)),
                        f"{summary.get('points_spent')}/{summary.get('budget')} pts"
                        + (" (auto)" if summary.get("auto_draft") else "")),
            f"Ships ({len(summary.get('ships', []))}): {ships}",
            f"Formation: {clip(summary.get('formation_name') or 'default')}"
            + (f" - {clip(summary.get('formation_rationale'), 110)}"
               if summary.get("formation_rationale") else ""),
        ]
        if summary.get("selection_rationale"):
            lines.append(f"Concept: {clip(summary['selection_rationale'])}")
        return self.emit(lines)

    def checkpoint(self, runner: Any) -> str:
        t = runner.simulation.current_time
        events = self._new_events(runner)
        lines = [self.header(f"CP {runner.checkpoint_count}", f"t={t:.0f}s")]
        lines += self._side_lines(runner)
        lines += self._digest_lines(runner, events)
        directives = {e.data.get("faction"): e.data.get("directive")
                      for e in events if e.event_type == "admiral_directive"}
        failures = {e.data.get("faction"): e.data.get("reason")
                    for e in events if e.event_type == "admiral_directive_failed"}
        retried = {e.data.get("faction"): e.data.get("reason")
                   for e in events if e.event_type == "admiral_directive_retried"}
        for faction in FACTIONS:
            admiral = getattr(runner, f"{faction}_admiral", None)
            if admiral is None:
                text = "(no admiral - heuristic captains)"
            elif failures.get(faction):
                text = (f"(DIRECTIVE FAILED: {clip(failures[faction], 90)} - "
                        "captains get no directive this checkpoint)")
            else:
                text = clip(directives.get(faction)) or "(no new directive)"
                if retried.get(faction):
                    text += f" [after retry: {clip(retried[faction], 60)}]"
            lines.append(f"{faction[0].upper()} admiral: {text}")
        cache = self._cache_line(runner)
        if cache:
            lines.append(cache)
        return self.emit(lines)

    @staticmethod
    def _cache_line(runner: Any) -> str:
        """Running prompt-cache totals of the battle's client (CP-05 proof)."""
        stats = getattr(getattr(runner, "client", None), "stats", None)
        summary = getattr(stats, "cache_summary", None)
        if not callable(summary):
            return ""
        text = summary()
        return f"Prompt {text}" if isinstance(text, str) and "no calls" not in text else ""

    def result(self, runner: Any, winner: Optional[str], reason: Optional[str],
               recording: str, calls: int = 0) -> str:
        header = ["RESULT"]
        if runner is not None and getattr(runner, "simulation", None) is not None:
            header.append(f"t={runner.simulation.current_time:.0f}s")
            header.append(f"after CP {runner.checkpoint_count}")
        lines = [self.header(*header),
                 f"Winner: {(winner or 'none').upper()} - {clip(reason or 'no result', 200)}"]
        if runner is not None and getattr(runner, "simulation", None) is not None:
            lines += self._side_lines(runner)
            events = self._new_events(runner)
            if events:
                lines += [f"Final interval - {line}" for line in self._digest_lines(runner, events)]
        reported, committed = self.spend()
        lines.append(f"Spend: ${reported:.4f} reported, ${committed:.4f} committed"
                     + (f" of ${self.limit_usd:.2f} limit" if self.limit_usd is not None else "")
                     + f", {calls} model calls")
        lines.append(f"Recording: {recording}")
        return self.emit(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_events(self, runner: Any) -> List[Any]:
        recorder = getattr(runner, "recorder", None)
        if recorder is None:
            return []
        events = recorder.events[self._cursor:]
        self._cursor = len(recorder.events)
        for e in events:
            if e.event_type == "ship_destroyed" and e.ship_id:
                self._death_times.setdefault(e.ship_id, e.timestamp)
            elif e.event_type == "shot_fired":
                pid = e.data.get("projectile_id")
                if pid:
                    self._projectile_weapons[pid] = str(e.data.get("weapon_type", ""))
        return events

    def _ships(self, runner: Any, faction: str) -> Dict[str, Any]:
        return getattr(runner, f"{faction}_ships", None) or {}

    def _name(self, runner: Any, ship_id: Optional[str]) -> str:
        """Faction-tagged short name ('B:Ward-1'); mirrored drafts share class names."""
        for faction in FACTIONS:
            ship = self._ships(runner, faction).get(ship_id)
            if ship is not None:
                return f"{faction[0].upper()}:{short_name(getattr(ship, 'name', ship_id))}"
        return str(ship_id or "?")

    def _side_lines(self, runner: Any) -> List[str]:
        lines = []
        for faction in FACTIONS:
            ships = self._ships(runner, faction)
            alive = 0
            hull_total = 0.0
            entries = []
            for sid in sorted(ships, key=lambda s: (len(s), s)):
                ship = ships[sid]
                hull = 0.0 if ship.is_destroyed else max(0.0, float(ship.hull_integrity))
                entry = f"{short_name(getattr(ship, 'name', sid))} {type_abbr(ship.ship_type)}"
                if ship.is_destroyed:
                    died = self._death_times.get(sid)
                    entry += f" DEAD@{died:.0f}s" if died is not None else " DEAD"
                else:
                    entry += f" hull {hull:.0f}%{self._armor_heat(ship)}"
                    if getattr(ship, "is_dying", False):
                        entry += " DYING"
                    elif getattr(ship, "is_surrendered", False):
                        entry += " SURRENDERED"
                    else:
                        alive += 1
                        maneuver = getattr(ship, "current_maneuver", None)
                        mtype = getattr(getattr(maneuver, "maneuver_type", None), "name", "")
                        if mtype == "EVASIVE":
                            mode = str(getattr(ship, "execution_mode", "") or "evade").lower()
                            entry += f" [{mode}]"
                hull_total += hull
                entries.append(entry)
            pct = hull_total / len(ships) if ships else 0.0
            lines.append(f"{faction.upper()} {model_label(self.models.get(faction))}: "
                         f"{alive}/{len(ships)} alive, hull {pct:.0f}% | " + " | ".join(entries))
        return lines

    @staticmethod
    def _armor_heat(ship: Any) -> str:
        """' arm 3/36cm heat 42%': weakest facing now/originally, heat % (+' rad' if out)."""
        text = ""
        sections = getattr(getattr(ship, "armor", None), "sections", None) or {}
        if sections:
            weakest = min(sections.values(), key=lambda a: a.thickness_cm)
            original = getattr(weakest, "original_thickness_cm", 0.0) or weakest.thickness_cm
            text += f" arm {weakest.thickness_cm:.0f}/{original:.0f}cm"
        if getattr(ship, "thermal_system", None) is not None:
            text += f" heat {float(ship.heat_percent):.0f}%"
            if getattr(ship, "radiators_extended", False) is True:
                text += " rad"
        return text

    def _faction(self, runner: Any, ship_id: Optional[str]) -> Optional[str]:
        for faction in FACTIONS:
            if ship_id in self._ships(runner, faction):
                return faction
        return None

    def _digest_lines(self, runner: Any, events: List[Any]) -> List[str]:
        groups: Counter = Counter()
        kills = []
        stats = {f: Counter() for f in FACTIONS}
        for e in events:
            et, data = e.event_type, e.data
            if et == "hit":
                shooter = data.get("shooter_id") or data.get("source_ship_id")
                weapon = self._projectile_weapons.get(data.get("projectile_id"), "")
                groups[(shooter, e.ship_id, "spinal" if "spinal" in weapon else "gun")] += 1
                if self._faction(runner, shooter):
                    stats[self._faction(runner, shooter)]["hits"] += 1
            elif et == "torpedo_impact":
                groups[(e.ship_id, data.get("target_id"), "torp")] += 1
                if self._faction(runner, e.ship_id):
                    stats[self._faction(runner, e.ship_id)]["impacts"] += 1
            elif et == "shot_fired" and self._faction(runner, e.ship_id):
                stats[self._faction(runner, e.ship_id)]["shots"] += 1
            elif et == "torpedo_launched" and self._faction(runner, e.ship_id):
                stats[self._faction(runner, e.ship_id)]["torps"] += 1
            elif et == "pd_torpedo_disabled" and self._faction(runner, e.ship_id):
                stats[self._faction(runner, e.ship_id)]["pd_kills"] += 1
            elif et == "ship_destroyed":
                killer = data.get('killer_id')
                by = f" by {self._name(runner, killer)}" if killer else ""
                kills.append(f"{self._name(runner, e.ship_id)}{by} "
                             f"({data.get('cause') or 'destroyed'}) @{e.timestamp:.0f}s")
            elif et == "ship_dying":
                kills.append(f"{self._name(runner, e.ship_id)} crippled by "
                             f"{self._name(runner, data.get('killer_id'))} "
                             f"(detonates in {data.get('detonation_in_s', 0):.0f}s) @{e.timestamp:.0f}s")
        ranked = groups.most_common()
        hits = " | ".join(f"{self._name(runner, s)}>{self._name(runner, t)} {kind} x{n}"
                          for (s, t, kind), n in ranked[:MAX_HIT_GROUPS])
        if len(ranked) > MAX_HIT_GROUPS:
            hits += f" | +{len(ranked) - MAX_HIT_GROUPS} more"
        fire = " ; ".join(
            f"{f[0].upper()} {s['shots']} slugs/{s['hits']} hit, {s['torps']} torps/"
            f"{s['impacts']} impact, PD downed {s['pd_kills']} torps"
            for f, s in stats.items())
        return [f"Hits: {hits or 'none'}",
                f"Kills: {' ; '.join(kills) if kills else 'none'}",
                f"Fire: {fire}"]
