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

Two doctrines:

- "tactical" (default): flies inside a shared fleet plan
  (src/llm/heuristic_tactics.py) - reach-aware focus fire, nose-first gun
  ships (closure cap, coast through a controlled pass facing the nearest
  enemy gun, re-close nose first; they still pass through each other, at
  ~3-4 km/s instead of ~6; capital ships close at <= 1 km/s), torpedo boats
  holding a standoff and firing every launchable round at big, slow,
  thinly covered targets, PD-saturating salvos first, the evade
  overlay against guided torpedoes, crippled ships opening range, a
  formation-matched approach, and the plan announced on the FLEET NET.
- "legacy": the pass-4 captain (nearest target, INTERCEPT/BRAKE/PADLOCK,
  two-round torpedo probes). Kept bit-for-bit reachable: the fire-control
  calibration fixture (tests/fixtures/fire_control_shots.json) holds rounds
  from 32 seeded battles flown with it (run_draft_battle.py --doctrine legacy).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from .captain import LLMCaptain, LLMCaptainConfig
from .client import ToolCall
from .communication import MAX_FLEET_SIGNALS_PER_DECISION
from .maneuver_orders import admiral_order_moves_ship
from . import heuristic_tactics as tactics
from . import heuristic_flank as flank

# Sentinel model id that makes the battle runner build a HeuristicCaptain
# instead of an LLM-backed captain.
HEURISTIC_MODEL = "heuristic"

DOCTRINE_TACTICAL = "tactical"
DOCTRINE_LEGACY = "legacy"
# Experimental variants (src/llm/heuristic_flank.py): the static wall and
# pincer / envelopment / two-axis torpedo strikes. Never the default.
from .heuristic_flank import FLANK_DOCTRINES  # noqa: E402
DOCTRINES = (DOCTRINE_TACTICAL, DOCTRINE_LEGACY) + tuple(FLANK_DOCTRINES)
DEFAULT_DOCTRINE = DOCTRINE_TACTICAL

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

    def __init__(self, config: LLMCaptainConfig, client: Any = None,
                 doctrine: Optional[str] = None):
        super().__init__(config, client=None)  # never calls an LLM
        self._last_target_id: Optional[str] = None
        doctrine = doctrine or DEFAULT_DOCTRINE
        if doctrine not in DOCTRINES and flank.flank_spec(doctrine) is None:
            raise ValueError(f"unknown heuristic doctrine {doctrine!r} (one of {DOCTRINES})")
        self.doctrine = doctrine
        self._decision_interval = 30.0
        self._broke_off = False
        self._withdrawn = False
        self._face_id: Optional[str] = None

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
        elif self.doctrine == DOCTRINE_LEGACY:
            calls = self._plan(ship, ship_id, enemies, simulation, checkpoint)
        else:
            calls = self._plan_tactical(ship, ship_id, enemies, simulation, checkpoint)

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
                call = {"count": salvo, "target_id": torp_target.ship_id}
                if order and getattr(order, "torpedo_profile", None):
                    call["profile"] = order.torpedo_profile
                calls.append(("launch_torpedo", call))

        # Maneuver priority: dodge torpedoes > admiral heading / structured
        # maneuver order (already applied by the runner - keep it, it holds
        # the formation) > admiral keyword > role doctrine
        threat = self._torpedo_threat(ship, ship_id, simulation)
        # Only an order the runner actually flew (one whose target did not
        # resolve was NOT applied: fly our own maneuver instead).
        admiral_heading = bool(order and admiral_order_moves_ship(order))
        # An applied order with the evasion overlay already dodges the
        # torpedo (and keeps its course otherwise): no need to override it.
        overlay = admiral_heading and bool((getattr(order, "maneuver", None) or {}).get("evade"))
        maneuver = self._admiral_maneuver(order_text)
        if threat and not overlay:
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

    # ------------------------------------------------------------------
    # Tactical doctrine (src/llm/heuristic_tactics.py holds the fleet plan)
    # ------------------------------------------------------------------

    def _plan_tactical(self, ship, ship_id, enemies, simulation, checkpoint):
        calls: List[Tuple[str, Dict[str, Any]]] = []
        order = self.admiral_orders[0] if self.admiral_orders else None
        order_text = (order.order_text if order else "").lower()
        plan = tactics.fleet_plan(simulation, ship.faction, self.config.fleet_data)
        my_role = tactics.role(ship)
        if (my_role == "unarmed" and not self.has_admiral
                and tactics.should_surrender(simulation, ship, enemies)):
            # Nothing to shoot with, no armed ship and no round of ours left
            # in flight, and an enemy that can catch us is closing: strike.
            return [("surrender", {"reason": "all weapons out, no rounds in flight, cannot get away"})]
        self._decision_interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0)
        self._sim = simulation

        target = self._tactical_target(ship, enemies, order, plan, my_role)
        calls.append(("set_primary_target", {"target_id": target.ship_id}))
        if self.weapon_groups and target.ship_id != self._last_target_id:
            calls.append(("set_weapons_order", {
                "spinal_mode": "FIRE_WHEN_OPTIMAL",
                "turret_mode": "FIRE_WHEN_OPTIMAL",
                "spinal_min_probability": 0.25,
                "turret_min_probability": 0.2,
            }))
        self._last_target_id = target.ship_id

        # Torpedoes: the admiral's salvo order wins; otherwise the fleet's
        # massed salvo (every boat launches the same checkpoint) or nothing.
        if getattr(ship, "torpedo_launcher", None) is not None:
            max_salvo = self._max_torpedo_salvo(ship_id, simulation)
            if order and order.torpedo_salvo:
                salvo = min(int(order.torpedo_salvo), max_salvo)
                match = self._match_enemy(order.torpedo_target, enemies) if order.torpedo_target else None
                if salvo > 0:
                    call = {"count": salvo, "target_id": (match or target).ship_id}
                    if getattr(order, "torpedo_profile", None):
                        call["profile"] = order.torpedo_profile
                    calls.append(("launch_torpedo", call))
            else:
                for torp_target_id, rounds in plan.launches.get(ship_id, []):
                    salvo = min(rounds, max_salvo)
                    max_salvo -= salvo
                    if salvo > 0:
                        calls.append(("launch_torpedo", {"count": salvo,
                                                         "target_id": torp_target_id}))

        if any(tactics._alive(f) for f in simulation.get_friendly_ships(ship_id)
               if f.ship_id != ship_id):
            # (The last ship of a fleet has nobody to signal to.)
            calls.extend(self._fleet_net_signals(ship, ship_id, enemies, plan))

        threat = self._torpedo_threat(ship, ship_id, simulation)
        heat = getattr(ship, "heat_percent", 0.0) or 0.0
        admiral_heading = bool(order and admiral_order_moves_ship(order))
        keyword = self._admiral_maneuver(order_text)
        overlay = admiral_heading and bool((getattr(order, "maneuver", None) or {}).get("evade"))
        if admiral_heading and threat and not overlay:
            calls.append(("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}))
        elif admiral_heading:
            pass  # the runner already flies it (and it holds the formation)
        elif keyword is not None and keyword[0] == "HOLD_RANGE":
            # "Keep range": hold the range we are at (the engine latches it).
            args = {"maneuver_type": "HOLD_RANGE", "target_id": target.ship_id,
                    "throttle": self._heat_throttle_cap(ship, heat, 1.0, threat)}
            if threat:
                args["evade"] = "full"
            calls.append(("set_maneuver", args))
        elif keyword is not None:
            maneuver = ("EVADE", 1.0) if threat else keyword
            calls.append(("set_maneuver", {
                "maneuver_type": maneuver[0],
                "throttle": self._heat_throttle_cap(ship, heat, maneuver[1], threat)}))
        elif (hulk := tactics.blast_threat(simulation, ship)) is not None:
            # A dying hulk's reactor is about to go: get out of the blast.
            away = ship.position - hulk.position
            if away.magnitude < 1.0:
                # Sitting on the hulk: burn away from the nearest enemy instead.
                near = min(enemies, key=lambda e: (ship.position.distance_to(e.position), e.ship_id))
                away = ship.position - near.position
            away = away.normalized()
            calls.append(("set_heading", {
                "direction": {"x": round(away.x, 4), "y": round(away.y, 4), "z": round(away.z, 4)},
                "throttle": self._heat_throttle_cap(ship, heat, 1.0, threat)}))
        else:
            args = self._tactical_maneuver(ship, ship_id, target, enemies, plan, my_role, threat)
            args["throttle"] = self._heat_throttle_cap(ship, heat, args["throttle"], threat)
            # ...and do not cook a friendly astern with the torch (src/drive_plume.py).
            aim = args.get("direction") if args.get("maneuver_type") == "HEADING" else None
            ordered = args["throttle"]
            args["throttle"] = tactics.plume_safe_throttle(
                simulation, ship, args["throttle"], threat, aim)
            burn = None
            if args["throttle"] < tactics.PLUME_SLIP_TRIGGER_SHARE * ordered:
                # A friendly sits in the jet: side-slip the burn instead of idling.
                if aim is not None:
                    burn = SimpleNamespace(**aim)
                elif args.get("maneuver_type") == "INTERCEPT" and args.get("target_id"):
                    ref = simulation.get_ship(args["target_id"])
                    burn = (ref.position - ship.position) if ref is not None else None
            if burn is not None:
                slipped, t = tactics.plume_clear_heading(simulation, ship, burn, ordered, threat)
                if t > args["throttle"]:
                    args = {"maneuver_type": "HEADING", "throttle": t,
                            "direction": {"x": round(slipped.x, 4), "y": round(slipped.y, 4),
                                          "z": round(slipped.z, 4)},
                            **({"evade": args["evade"]} if args.get("evade") else {})}
            if args.get("maneuver_type") == "HEADING":
                args.pop("maneuver_type")
                calls.append(("set_heading", args))
            else:
                calls.append(("set_maneuver", args))

        want = self._radiator_wish(ship, ship_id, heat, threat, enemies, simulation)
        if want is not None and want != bool(getattr(ship, "radiators_extended", False)):
            calls.append(("set_radiators", {"extend": want}))
        return calls

    def _fleet_net_signals(self, ship, ship_id, enemies, plan) -> List[Tuple[str, Dict[str, Any]]]:
        """The plan, said out loud on the FLEET NET (recorded comms_message
        events the viewer's command net shows): the flag ship claims each
        new focus target and calls every massed salvo; a crippled ship
        announces that it is breaking off."""
        names = {e.ship_id: getattr(e, "name", e.ship_id) for e in enemies}
        out: List[Tuple[str, Dict[str, Any]]] = []
        disarmed = tactics.role(ship) == "unarmed"
        crippled = tactics.hull_pct(ship) < tactics.CRIPPLED_HULL_PCT or disarmed
        if crippled and not self._broke_off:
            why = ("weapons out" if disarmed else f"hull at {tactics.hull_pct(ship):.0f}%")
            out.append(("fleet_signal", {
                "kind": "BREAKING_OFF",
                "text": f"{why[0].upper() + why[1:]} - breaking off, opening range."}))
        self._broke_off = crippled
        if self.has_admiral:
            return out      # the admiral, not the flag ship, sets the fleet's targets
        if ship_id == plan.flag_id and plan.focus_changed and plan.focus_id in names:
            out.append(("fleet_signal", {
                "kind": "TARGET_CLAIM", "target": names[plan.focus_id],
                "text": f"Focus fire: all guns on {names[plan.focus_id]}."}))
        if ship_id == plan.flag_id and plan.salvos:
            first = plan.salvos[0]
            if first.target_id in names:
                rounds = sum(first.rounds.values())
                note = (f"{first.need} saturate his point defence" if first.saturating else
                        f"short of the {first.need} that would saturate his point defence")
                out.append(("fleet_signal", {
                    "kind": "TOT_SYNC", "target": names[first.target_id],
                    "text": (f"Torpedo salvo: {rounds} rounds from {len(first.rounds)} "
                             f"boat(s) on {names[first.target_id]}, launching together "
                             f"({note}).")}))
        return out[:MAX_FLEET_SIGNALS_PER_DECISION]

    def _tactical_target(self, ship, enemies, order, plan, my_role) -> Any:
        """Admiral's target > the boats' salvo target (torpedo boats) > the
        fleet focus, weighed from this gun ship's position (gun_target)."""
        if order:
            for wanted in (order.torpedo_target, order.suggested_target):
                match = self._match_enemy(wanted, enemies)
                if match is not None:
                    return match
        nearest = min(enemies, key=lambda e: (ship.position.distance_to(e.position), e.ship_id))
        if my_role == "torpedo" and plan.torpedo_target_id:
            pick = next((e for e in enemies if e.ship_id == plan.torpedo_target_id), None)
            if pick is not None:
                return pick
        focus = next((e for e in enemies if e.ship_id == plan.focus_id), None)
        if my_role != "gun":
            return focus or nearest
        return tactics.gun_target(ship, enemies, plan.focus_id) or nearest

    def _tactical_maneuver(self, ship, ship_id, target, enemies, plan, my_role,
                           threat: bool) -> Dict[str, Any]:
        """set_maneuver arguments for range discipline by hull role."""
        def km(e) -> float:
            return ship.position.distance_to(e.position) / 1000.0

        gunned = [e for e in enemies if tactics.has_guns(e)]
        nearest_gun = min(gunned, key=lambda e: (km(e), e.ship_id)) if gunned else None
        gun_km = km(nearest_gun) if nearest_gun is not None else float("inf")
        # Run from (or present to) guided torpedoes. No dodge against guns:
        # the jink swings the nose off and shows the thin lateral armour.
        evade = "full" if threat else None

        def order(kind: str, ref, throttle: float, **extra) -> Dict[str, Any]:
            args: Dict[str, Any] = {"maneuver_type": kind, "target_id": ref.ship_id,
                                    "throttle": throttle, **extra}
            if evade:
                args["evade"] = evade
            return args

        if tactics.hull_pct(ship) < tactics.CRIPPLED_HULL_PCT or my_role == "unarmed":
            self._withdrawn = True
        if self._withdrawn and nearest_gun is not None:
            # Crippled or disarmed, for good: open range from the nearest gun
            # (guns still free) and never come back in - beyond the retreat
            # range it holds the range it has. Inside it the ship flies a
            # plain HEADING straight away from that gun at full burn: HOLD_RANGE
            # (with or without the evade overlay) caps its opening speed at
            # what it could stop at 350 km and the chaser kept closing.
            if gun_km < tactics.RETREAT_RANGE_KM:
                away = tactics.withdraw_direction(ship, nearest_gun)
                args = {"maneuver_type": "HEADING", "throttle": 1.0,
                        "direction": {"x": round(away.x, 4), "y": round(away.y, 4),
                                      "z": round(away.z, 4)}}
                if evade:
                    args["evade"] = evade
                return args
            return order("HOLD_RANGE", nearest_gun, 1.0,
                         range_km=round(max(tactics.RETREAT_RANGE_KM, gun_km), 1))
        if my_role == "torpedo" and nearest_gun is None:
            # No enemy gun left: nothing to stand off from. Close to a short
            # launch range so rounds can reach - two boat fleets holding
            # 300 km from each other never launched again (the launch check
            # refuses rounds at an opening target) and sat to the time limit
            # with loaded tubes.
            prey = min(enemies, key=lambda e: (km(e), e.ship_id))
            return order("INTERCEPT", prey, 1.0, standoff_km=tactics.BOAT_NO_GUNS_RANGE_KM)
        flank = self._flank_maneuver(ship, ship_id, target, enemies, plan, my_role,
                                     nearest_gun, evade)
        if flank is not None:
            return flank
        if my_role != "gun":
            anchor = nearest_gun or min(enemies, key=lambda e: (km(e), e.ship_id))
            return order("HOLD_RANGE", anchor, 1.0,
                         range_km=tactics.boat_station_km(self._sim, ship, anchor))
        r_km = tactics.engage_range_km(ship)
        dist_km, closing_kps = self._range_and_closing(ship, target)
        throttle = plan.formation_throttle.get(ship_id, 1.0) if dist_km > tactics.GUNS_HOT_KM else 1.0
        cap = (tactics.CAPITAL_MAX_CLOSING_KPS if tactics.is_capital(ship)
               else tactics.SPRINT_MAX_CLOSING_KPS)
        kind, extra = tactics.gun_maneuver(dist_km, closing_kps, r_km,
                                           tactics.max_accel_ms2(ship) * throttle,
                                           self._decision_interval, cap)
        if kind == "PADLOCK" and nearest_gun is not None and gun_km < dist_km:
            # Face the threat, shoot the focus: the nose (thickest armour,
            # spinal) tracks the nearest enemy gun while the turrets stay on
            # the fleet's target. Sticky (FACE_SWITCH_FACTOR): a slow-turning
            # hull swinging between threats shows its side to all of them.
            face = next((e for e in gunned if e.ship_id == self._face_id), None)
            if face is None or km(face) > tactics.FACE_SWITCH_FACTOR * gun_km:
                face = nearest_gun
            self._face_id = face.ship_id
            if km(face) < dist_km:
                return self._aspect_or_padlock(ship, face, order)
        else:
            self._face_id = None
        if kind == "PADLOCK":
            return self._aspect_or_padlock(ship, target, order)
        return order(kind, target, throttle if kind == "INTERCEPT" else 1.0, **extra)

    def _aspect_or_padlock(self, ship, face, order) -> Dict[str, Any]:
        """Coast with the nose at the hull's gun aspect to ``face`` (weapon
        arcs: heuristic_tactics.gun_aspect_deg) - an attitude-only HEADING -
        or PADLOCK it nose-on when the aspect is 0 or the range is too short
        for a fixed attitude to keep it inside the nose-armour arc."""
        aspect = tactics.gun_aspect_deg(getattr(ship, "ship_type", ""))
        rng_km = ship.position.distance_to(face.position) / 1000.0
        if aspect <= 0.0 or rng_km < tactics.ASPECT_MIN_RANGE_KM or tactics.is_capital(ship):
            # Capital hulls PADLOCK: a fixed 30 s attitude drifts off the
            # target and a slow hull (and its slow roll) never re-settles -
            # measured on the pass-5 masks, battleships fired ~5 % fewer
            # rounds at their aspect than PADLOCKed nose-on.
            return order("PADLOCK", face, 1.0)
        d = tactics.aspect_direction(ship, face, aspect)
        args = {"maneuver_type": "HEADING", "throttle": 0.0,
                "direction": {"x": round(d.x, 4), "y": round(d.y, 4), "z": round(d.z, 4)}}
        evade = order("PADLOCK", face, 1.0).get("evade")
        if evade:
            args["evade"] = evade
        return args

    def _flank_maneuver(self, ship, ship_id, target, enemies, plan, my_role, nearest_gun,
                        evade) -> Optional[Dict[str, Any]]:
        """Maneuver for the experimental flank doctrines (heuristic_flank),
        or None to fly the tactical doctrine's."""
        spec = flank.flank_spec(self.doctrine)
        if spec is None:
            return None
        st = flank.flank_state(self._sim, ship.faction, spec)
        group = st.groups.get(ship_id, "hold")

        def with_evade(args, overlay=None):
            ov = evade or overlay
            if ov:
                args["evade"] = ov
            return args

        if my_role == "torpedo":
            if spec.kind != "split" or nearest_gun is None:
                return None
            boats = sorted(sid for sid, g in st.groups.items() if g == "boat")
            key = "boat_a" if boats.index(ship_id) % 2 == 0 else "boat_b"
            station = st.stations.get(key)
            if station is None:
                return None
            return with_evade({"maneuver_type": "HEADING", "throttle": 1.0,
                               "toward_point": flank.point_km(station)})
        if my_role != "gun" or spec.kind == "split":
            return None
        gunned = [e for e in enemies if tactics.has_guns(e)] or enemies
        near = min(gunned, key=lambda e: (ship.position.distance_to(e.position), e.ship_id))
        if group == "hold":
            engage = flank.WALL_ENGAGE_KM
            if spec.kind == "pincer" and not st.attack_started:
                engage = flank.HOLD_WAIT_KM
            dist_km, closing = self._range_and_closing(ship, near)
            kind = flank.wall_maneuver(dist_km, closing, engage)
            throttle = plan.formation_throttle.get(ship_id, 1.0) if kind == "INTERCEPT" else 1.0
            return with_evade({"maneuver_type": kind, "target_id": near.ship_id, "throttle": throttle})
        station = st.stations.get(group)
        if station is None:
            return None
        if not st.attack_started:
            # Positioning: fly to the station with the evade overlay on.
            return with_evade({"maneuver_type": "HEADING", "throttle": 1.0,
                               "toward_point": flank.point_km(station)}, "dodge")
        # Committed attack run to our own engagement range.
        r_km = tactics.engage_range_km(ship)
        dist_km, _ = self._range_and_closing(ship, target)
        if dist_km > r_km * tactics.HOLD_BAND_FACTOR:
            return with_evade({"maneuver_type": "INTERCEPT", "target_id": target.ship_id,
                               "throttle": 1.0, "standoff_km": r_km,
                               "arrival_closing_kps": 0.6})
        return with_evade({"maneuver_type": "PADLOCK", "target_id": target.ship_id, "throttle": 1.0})

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
