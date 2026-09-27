"""
HeuristicCaptain - the "cheap AI" ship captain for draft mode.

A rule-based captain that costs zero tokens: it obeys the structured parts of
its admiral's orders (suggested target, torpedo salvos, a few order-text
keywords) and otherwise flies sane default doctrine - evade inbound
torpedoes, close to gun range, keep a torpedo boat out of the enemy's teeth,
manage radiators.

It emits decisions as the same ToolCalls a real captain would and executes
them through LLMCaptain._execute_tool, so the command path, validation and
recording are identical to a live LLM battle (the ScriptedCaptain pattern
from scripts/generate_test_battle.py, but reactive instead of pre-planned).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .captain import LLMCaptain, LLMCaptainConfig
from .client import ToolCall

# Sentinel model id that makes the battle runner build a HeuristicCaptain
# instead of an LLM-backed captain.
HEURISTIC_MODEL = "heuristic"

# Ranges in km
GUN_APPROACH_RANGE = 400.0
GUN_KNIFE_RANGE = 120.0
TORPEDO_STANDOFF = 250.0
TORPEDO_MAX_LAUNCH_RANGE = 700.0
TORPEDO_THREAT_ETA_S = 75.0
# A torpedo hull inside TORPEDO_STANDOFF closing faster than this burns to
# hold range (range-control heading) instead of weaving on course.
HOLD_RANGE_CLOSING_KPS = 0.3
OVERSHOOT_CLOSING_KPS = 12.0

# Heat management (percent of heat-sink capacity). The engine's ladder:
# guns hold >= 90%, radiators forced out >= 100%, reactor meltdown >= 150%.
HEAT_EXTEND_PCT = 65.0        # extend whenever no torpedo is inbound
HEAT_EXTEND_SAFE_PCT = 35.0   # ...or earlier when nothing can shoot the panels
HEAT_EMERGENCY_PCT = 85.0     # extend even under torpedo fire; cut the burn
HEAT_EASE_PCT = 75.0          # start easing the burn below radiator capacity
HEAT_SAFE_PCT = 20.0          # cool enough to button up
HEAT_OVERLIMIT_PCT = 100.0    # past this even an evasion burn is capped
SAFE_ENEMY_RANGE_KM = 800.0   # beyond every gun's reach: panels are safe out


class HeuristicCaptain(LLMCaptain):
    """Rule-based captain: admiral's orders first, sound doctrine second."""

    def __init__(self, config: LLMCaptainConfig, client: Any = None):
        super().__init__(config, client=None)  # never calls an LLM
        self._last_target_id: Optional[str] = None

    def select_personality(self, distance_km: float, verbose: bool = False) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, ship_id: str, simulation: Any) -> List[Any]:
        if self.has_surrendered:
            return []
        ship = simulation.get_ship(ship_id)
        if (not ship or ship.is_destroyed or getattr(ship, "is_dying", False)
                or getattr(ship, "is_surrendered", False)):
            return []

        if not self.weapon_groups and self.config.fleet_data and self.config.ship_type:
            self.setup_weapon_groups(self.config.ship_type, self.config.fleet_data)

        enemies = [e for e in simulation.get_enemy_ships(ship_id)
                   if not e.is_destroyed and not getattr(e, "is_dying", False)
                   and not getattr(e, "is_surrendered", False)]
        checkpoint = self.decision_count + 1

        calls: List[Tuple[str, Dict[str, Any]]] = []
        if not enemies:
            calls.append(("set_maneuver", {"maneuver_type": "MAINTAIN"}))
        else:
            calls = self._plan(ship, ship_id, enemies, simulation, checkpoint)

        tool_calls = [
            ToolCall(id=f"heuristic_{checkpoint}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ]
        commands: List[Any] = []
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

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, ship, ship_id, enemies, simulation, checkpoint):
        calls: List[Tuple[str, Dict[str, Any]]] = []
        order = self.admiral_orders[0] if self.admiral_orders else None
        order_text = (order.order_text if order else "").lower()

        target = self._pick_target(ship, enemies, order)
        target_name = getattr(target, "name", target.ship_id)
        dist_km, closing_kps = self._range_and_closing(ship, target)

        # Target designation (idempotent; keeps sim-side primary target synced)
        calls.append(("set_primary_target", {"target_name": target_name}))

        # Gun posture: (re)issued when the target changes so WeaponsOrders
        # never keep firing at a stale target id.
        if self.weapon_groups and target.ship_id != self._last_target_id:
            calls.append(("set_weapons_order", {
                "spinal_mode": "FIRE_WHEN_OPTIMAL",
                "turret_mode": "FIRE_WHEN_OPTIMAL",
                "spinal_min_probability": 0.25,
                "turret_min_probability": 0.2,
            }))
        self._last_target_id = target.ship_id

        # Torpedoes: the admiral's salvo order wins; otherwise probe with
        # small salvos while closing.
        has_launcher = getattr(ship, "torpedo_launcher", None) is not None
        if has_launcher:
            max_salvo = self._max_torpedo_salvo(ship_id, simulation)
            salvo = 0
            torp_target = target
            if order and order.torpedo_salvo:
                salvo = min(int(order.torpedo_salvo), max_salvo)
                if order.torpedo_target:
                    torp_target = self._match_enemy(order.torpedo_target, enemies) or target
            elif (closing_kps > -0.1 and dist_km < TORPEDO_MAX_LAUNCH_RANGE
                  and checkpoint % 2 == 1):
                salvo = min(2, max_salvo)
            if salvo > 0:
                calls.append(("launch_torpedo", {
                    "count": salvo, "target_id": torp_target.ship_id}))

        # Maneuver priority: dodge torpedoes > admiral heading order (already
        # applied by the runner - keep it, it holds the formation) > admiral
        # keyword > role doctrine
        threat = self._torpedo_threat(ship, ship_id, simulation)
        admiral_heading = bool(order and isinstance(getattr(order, "heading", None), dict))
        maneuver = self._admiral_maneuver(order_text)
        if threat:
            maneuver = ("EVADE", 1.0)
        elif admiral_heading:
            maneuver = ("KEEP", 0.0)
        if maneuver is None:
            maneuver = self._doctrine_maneuver(dist_km, closing_kps)
        heat = getattr(ship, "heat_percent", 0.0) or 0.0
        if maneuver[0] == "HOLD_RANGE":
            call = self._range_control_call(ship, target, simulation, maneuver[1])
            if call is None:
                maneuver = ("EVADE", maneuver[1])
            else:
                args = call[1] if len(call) > 1 else None
                if isinstance(args, dict) and "throttle" in args:
                    # The heat cap applies to range-control burns too.
                    args["throttle"] = self._heat_throttle_cap(
                        ship, heat, args["throttle"], threat)
                calls.append(call)
        if maneuver[0] not in ("KEEP", "HOLD_RANGE"):
            throttle = self._heat_throttle_cap(ship, heat, maneuver[1], threat)
            calls.append(("set_maneuver", {
                "maneuver_type": maneuver[0], "throttle": throttle}))

        # Radiators: shed heat when hot or when nothing can hit the panels,
        # button up under torpedo fire - unless heat is past the emergency
        # line, where cooling beats panel risk (the ladder would force them
        # out at 100% anyway).
        want = self._radiator_wish(ship, ship_id, heat, threat, enemies, simulation)
        if want is not None and want != bool(getattr(ship, "radiators_extended", False)):
            calls.append(("set_radiators", {"extend": want}))

        return calls

    def _radiator_wish(self, ship, ship_id, heat, threat, enemies, simulation) -> Optional[bool]:
        """True = extend, False = retract, None = leave as is."""
        if getattr(ship, "thermal_system", None) is None:
            return None
        if heat >= HEAT_EMERGENCY_PCT:
            return True
        if threat:
            return False
        if heat >= HEAT_EXTEND_PCT:
            return True
        if heat >= HEAT_EXTEND_SAFE_PCT and self._panels_safe(ship, ship_id, enemies, simulation):
            return True
        if heat < HEAT_SAFE_PCT:
            return False
        return None

    def _panels_safe(self, ship, ship_id, enemies, simulation) -> bool:
        """No enemy within gun reach and no slug inbound on us."""
        nearest_km = min(
            (ship.position.distance_to(e.position) / 1000.0 for e in enemies),
            default=float("inf"))
        if nearest_km < SAFE_ENEMY_RANGE_KM:
            return False
        return not any(getattr(p, "target_ship_id", None) == ship_id
                       for p in getattr(simulation, "projectiles", []) or [])

    def _heat_throttle_cap(self, ship, heat: float, throttle: float, threat: bool) -> float:
        """
        Keep the drive under what the radiators can reject once the ship runs
        hot. Full burn makes exactly the radiators' design rejection, so a
        damaged array (or gunfire on top) means a full burn keeps heating.
        A torpedo dodge keeps full throttle until heat passes 100%.
        """
        thermal = getattr(ship, "thermal_system", None)
        if thermal is None or heat < HEAT_EASE_PCT:
            return throttle
        drive_full_kw = getattr(thermal, "drive_heat_full_kw", 0.0) or 0.0
        if drive_full_kw <= 0:
            return throttle
        ratio = thermal.radiators.extended_dissipation_kw / drive_full_kw
        if threat and heat < HEAT_OVERLIMIT_PCT:
            return throttle
        if heat >= HEAT_EMERGENCY_PCT:
            cap = max(0.15, 0.5 * ratio)
        else:
            cap = max(0.3, 0.8 * ratio)
        return round(min(throttle, cap), 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_enemy(self, name: str, enemies: List[Any]) -> Optional[Any]:
        if not name:
            return None
        lowered = name.lower()
        for enemy in enemies:
            enemy_name = getattr(enemy, "name", enemy.ship_id)
            if enemy_name.lower() == lowered or enemy.ship_id == name:
                return enemy
        for enemy in enemies:
            enemy_name = getattr(enemy, "name", enemy.ship_id)
            if lowered in enemy_name.lower():
                return enemy
        return None

    def _pick_target(self, ship, enemies, order) -> Any:
        if order:
            for wanted in (order.torpedo_target, order.suggested_target):
                match = self._match_enemy(wanted, enemies)
                if match is not None:
                    return match
        return min(enemies,
                   key=lambda e: ship.position.distance_to(e.position))

    def _range_and_closing(self, ship, enemy) -> Tuple[float, float]:
        rel_pos = enemy.position - ship.position
        dist_m = rel_pos.magnitude
        if dist_m <= 0:
            return 0.0, 0.0
        rel_vel = enemy.velocity - ship.velocity
        closing_ms = -rel_vel.dot(rel_pos.normalized())
        return dist_m / 1000.0, closing_ms / 1000.0

    def _torpedo_threat(self, ship, ship_id: str, simulation) -> bool:
        """True if a live torpedo is inbound on us and worth dodging NOW."""
        for torp_flight in getattr(simulation, "torpedoes", []) or []:
            torp = torp_flight.torpedo
            if torp.target_id != ship_id:
                continue
            rel_pos = torp.position - ship.position
            dist_m = rel_pos.magnitude
            if dist_m <= 0:
                return True
            rel_vel = torp.velocity - ship.velocity
            closing_ms = -rel_vel.dot(rel_pos.normalized())
            if closing_ms <= 0:
                continue
            if dist_m / closing_ms < TORPEDO_THREAT_ETA_S:
                return True
        return False

    def _admiral_maneuver(self, order_text: str) -> Optional[Tuple[str, float]]:
        """Map explicit maneuver language in the admiral's order to an action."""
        if not order_text:
            return None
        keyword_map = [
            (("evade", "evasive", "dodge"), ("EVADE", 1.0)),
            (("brake", "slow down", "kill velocity"), ("BRAKE", 0.8)),
            (("padlock", "hold position", "coast and track"), ("PADLOCK", 1.0)),
            (("kite", "keep range", "stand off", "standoff", "maintain distance",
              "hold range", "cancel closure"), ("HOLD_RANGE", 1.0)),
            (("intercept", "close", "attack run", "charge", "engage close"),
             ("INTERCEPT", 0.9)),
        ]
        for keywords, action in keyword_map:
            if any(k in order_text for k in keywords):
                return action
        return None

    def _doctrine_maneuver(self, dist_km: float, closing_kps: float) -> Tuple[str, float]:
        if self.weapon_groups:
            # Gunship: close to effective range, avoid a blind overshoot,
            # then coast-track through the pass.
            if dist_km < 200.0 and closing_kps > OVERSHOOT_CLOSING_KPS:
                return ("BRAKE", 0.8)
            if dist_km > GUN_APPROACH_RANGE:
                return ("INTERCEPT", 1.0)
            if dist_km > GUN_KNIFE_RANGE:
                return ("INTERCEPT", 0.7)
            return ("PADLOCK", 1.0)
        # Pure torpedo hull: stay out of gun range, keep the seeker geometry.
        # Inside the standoff range and still closing, burn against the
        # relative velocity (EVADE would hold course straight into a merge).
        if dist_km < TORPEDO_STANDOFF:
            if closing_kps > HOLD_RANGE_CLOSING_KPS:
                return ("HOLD_RANGE", 1.0)
            return ("EVADE", 0.9)
        return ("INTERCEPT", 0.6)

    def _range_control_call(self, ship, target, simulation, throttle: float):
        """set_heading call for the precomputed range-control heading vs the
        target's group: cancel closure, or pass wide when braking is too late."""
        from .range_control import analyze_range_control
        try:
            infos = analyze_range_control(simulation, [ship], detail=False)
        except Exception:
            return None
        group = next((g for g in infos if target.ship_id in g.ship_ids), None)
        if group is None:
            return None
        option = None
        if group.warning and "too late" in group.warning:
            option = group.option("wide")
        option = option or group.option("cancel")
        if option is None or option.throttle <= 0:
            return None
        h = option.heading
        return ("set_heading", {
            "direction": {"x": round(h.x, 4), "y": round(h.y, 4), "z": round(h.z, 4)},
            "throttle": round(min(1.0, option.throttle * throttle), 3)})
