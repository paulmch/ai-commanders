"""
The captain's per-turn picture and the decide() harness (pass 4, captain
workstream). Golden-text tests pin the turn format; the rest pin the rules
behind it:

- CP-06 turn order and content: MY SHIP (world frame), THREATS TO ME,
  ENEMIES with fire solutions, FRIENDLIES with PD cover, RANGE CONTROL,
  ORDERS with age and engine checks, FLEET NET / ENEMY TRANSMISSIONS.
- R1: a reply with no action gets a same-checkpoint follow-up ("NO COMMANDS
  ISSUED") - 31% of captain decisions used to be acknowledgments only.
- CP-09: failed / truncated calls are retried, then recorded as LOST with
  the maneuver renewed and the next prompt told so.
- CP-08 / R7: torpedo-only hulls carry no gun content; R2: enemy torpedo
  tubes are listed; CP-14: history shows launches/headings/targets/reports;
  CP-17: names shared across fleets resolve by id; COMMS-6/9: traffic is
  labelled by side and the mandatory-reply boilerplate is gone.

Regenerate the golden files after an intended format change with
    UPDATE_GOLDEN=1 uv run pytest tests/test_captain_turn.py
and review the diff.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
from src.llm.admiral import AdmiralOrder
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import CaptainClient, ToolCall
from src.llm.mock_openrouter import mock_transport
from src.llm.prompts import (
    build_captain_doctrine,
    build_captain_messages,
    class_label,
    format_admiral_orders_for_captain,
    format_comms,
    format_enemy_weapons_summary,
    get_ship_turn_time_90deg,
)

GOLDEN = Path(__file__).parent / "golden"
FLEET = json.load(open("data/fleet_ships.json"))


def _golden(name: str, text: str) -> str:
    """The stored golden text (rewritten first when UPDATE_GOLDEN is set)."""
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text + "\n")
    return path.read_text().rstrip("\n")


GOLDEN_HINT = ("captain turn format changed - if intended, regenerate the golden file with "
               "UPDATE_GOLDEN=1 and review the diff")


# =============================================================================
# Fixtures: a hand-built turn (no engine numbers, so the golden text only
# moves when the FORMAT does)
# =============================================================================

THERMAL = {
    "heat_percent": 7.0, "heat_gj": 22.0, "capacity_gj": 320.0, "net_rate_mw": 48.0,
    "load_mw": 1.0, "radiators_extended": False, "radiator_mode": "manual",
    "time_to_100_retracted_s": None, "time_to_meltdown_retracted_s": None,
    "time_to_100_extended_s": None, "time_to_meltdown_extended_s": None,
    "radiator_available_mw": 780.0, "radiator_rating_mw": 780.0,
    "radiator_panels": {"tail_port": 100.0, "tail_starboard": 100.0,
                        "tail_dorsal": 100.0, "tail_ventral": 100.0},
}


def _enemy(sid, name, stype, dist, closing, dir_to, angle, **extra):
    e = {
        "ship_id": sid, "name": name, "ship_type": stype, "ship_class": "unknown",
        "distance_km": dist, "closing_rate": closing, "dir_to": dir_to, "angle_deg": angle,
        "relative_bearing_km": {"forward": dist, "starboard": -3.0, "up": 1.0},
        "hull_percent": 100.0, "armor": {}, "is_primary_target": False,
        "has_us_targeted": False, "pd_operational": 2,
    }
    e.update(extra)
    return e


def corvette_kwargs():
    enemies = [
        _enemy("beta_5", "OCS Falchion-1", "destroyer", 110.3, 4.68,
               {"x": -0.99, "y": 0.12, "z": 0.02}, 7.4, is_primary_target=True,
               has_us_targeted=True, gun_gj_per_min=21.2,
               torpedo_solution="REACHABLE | TOF ~9 s, ~16.2 km/s ~33 GJ | PD over him 2+4 -> "
                                "salvo >= 3 together | friendly rounds inbound 1"),
        _enemy("beta_7", "OCS Lancer-3", "cruiser_torpedo", 251.0, -3.51,
               {"x": -0.80, "y": -0.60, "z": 0.0}, 41.0, has_us_targeted=True,
               torpedoes_remaining=40, pd_operational=4,
               armor={"nose_damage_pct": 30.0, "lateral_damage_pct": 0.0},
               torpedo_solution="UNREACHABLE (OPENING 3.5 km/s): the round burns out before "
                                "intercept even if he holds course - expected ~0 GJ"),
    ]
    friendlies = [{
        "ship_id": "alpha_1", "name": "TIS Dart-1", "ship_type": "corvette",
        "distance_km": 37.4, "dir_to": {"x": 0.54, "y": -0.81, "z": 0.27},
        "rel_speed_kps": 0.21, "hull_percent": 100.0,
        "maneuver": {"type": "EVASIVE", "throttle": 0.8, "heading": None},
        "target_id": "beta_5", "target_name": "OCS Falchion-1",
        "torpedoes_remaining": 6, "rounds_in_flight": 1, "first_impact_s": 25.0,
        "pd_total": 1, "pd_operational": 1,
    }]
    tactical = {
        "sim_time": 180.0,
        "ship_forward": {"x": -0.96, "y": -0.27, "z": 0.01},
        "own": {"ship_id": "alpha_3", "position_km": {"x": -25.0, "y": 9.0, "z": 3.0},
                "velocity_kps": {"x": 2.9, "y": -0.5, "z": 0.1}, "speed_kps": 2.95,
                "accel_g": 3.0, "turn_90_s": 11.8},
        "primary_target_id": "beta_5",
        "our_shots": 3, "our_hits": 1, "our_damage_dealt": 21.6, "our_damage_taken": 0.0,
        "enemies": enemies,
        "friendlies": friendlies,
        "pd_umbrella": {"own": 1, "allies": [("TIS Dart-1", 1)], "range_km": 250.0},
        "torpedo_threats": [{
            "distance_km": 180.0, "closing_kps": 12.4, "eta_seconds": 15.0,
            "source": "OCS Lancer-3", "est_impact_gj": 30.0, "nez_inside": True,
            "blinded": False, "pd_turrets_needed": 3, "own_pd_turrets": 1,
            "allied_pd_turrets": 1,
        }, {
            "distance_km": 40.0, "closing_kps": 9.0, "eta_seconds": 4.0,
            "source": "TIS Dart-2 [alpha_2]", "est_impact_gj": 10.0, "blinded": True,
            "own_side": True, "miss_km": 0.4,
        }],
        "incoming_projectiles": [],
        "evasion_status": {"mode": "RUN", "threat_count": 2, "gun_rounds": 0,
                           "guided_torpedoes": 1, "blinded_wrecks": 1},
        "torpedo_hulls_on_me": [{"ship_id": "beta_7", "name": "OCS Lancer-3", "tubes": 4,
                                 "rounds": 40, "range_km": 251.0, "launch_tof_s": 38.0}],
        "current_config": {"weapon_orders": {}, "current_maneuver": {
            "type": "EVASIVE", "throttle": 0.8, "start_time": 150.0, "duration": 30.0}},
    }
    ship_status = {
        "hull_integrity": 100.0, "heat_percent": 7.0, "delta_v_remaining": 502.3,
        "nose_armor": 211.9, "lateral_armor": 36.3, "tail_armor": 42.4,
        "radiators_extended": False, "thermal": THERMAL,
        "torpedoes_remaining": 7, "torpedo_capacity": 8,
        "torpedo_tubes": [{"ready": True, "cooldown": 0.0}], "launch_capacity": 3,
        "pd_turrets_total": 1, "pd_turrets_operational": 1,
    }
    orders = format_admiral_orders_for_captain(
        [AdmiralOrder(target_ship_id="alpha_3", target_ship_name="TIS Dart-3",
                      order_text="INTERCEPT toward OCS Falchion-1 at 80%; fleet centre (90,55,10).",
                      priority="CRITICAL", suggested_target="OCS Falchion-1",
                      torpedo_salvo=1, torpedo_target="OCS Lancer-3")],
        "", now=180.0, reply_tool="report_to_admiral",
        previous_directive=("Hit the destroyers first; torpedo cruisers last.", 150.0),
        checks=[{
            "target": "OCS Lancer-3 [beta_7] - your sensors: 251 km, OPENING 3.51 km/s",
            "salvo": "UNREACHABLE (OPENING 3.5 km/s): the round burns out before intercept "
                     "even if he holds course - expected ~0 GJ. DEVIATE is the correct "
                     "response - say why.",
            "engine": "INTERCEPT@80% held for 2 checkpoints -> closest approach 1.2 km to OCS "
                      "Falchion-1 [beta_5] in 24 s, passing at 4.7 km/s = a MERGE (seconds "
                      "of fire, then flying apart)",
            "conflict": "Your order and RANGE CONTROL conflict (merge): choose and explain - "
                        "DEVIATE is allowed.",
            "points": ["(90,55,10) km is a POSITION: from you dir_to (+0.99,+0.40,+0.06), "
                       "125 km - set_heading takes that direction, not the position"],
        }])
    comms = [
        {"msg_id": "m1", "channel": "fleet_net", "sender_role": "captain",
         "sender_name": "TIS Dart-1", "sender_id": "alpha_1", "faction": "alpha",
         "text": "Taking Falchion-1 with 2 rounds at T+180, cover me.", "sent_t": 150.0},
        {"msg_id": "m2", "channel": "fleet_net", "sender_role": "engine",
         "sender_name": "engine", "sender_id": None, "faction": "alpha",
         "text": "TIS Dart-4 launched 2 torpedoes at OCS Falchion-1 [beta_5], ETA ~T+205.",
         "sent_t": 165.0},
        {"msg_id": "m3", "channel": "enemy_broadcast", "sender_role": "captain",
         "sender_name": "OCS Lancer-3", "sender_id": "beta_7", "faction": "beta",
         "text": "Your corvettes will burn.", "sent_t": 150.0},
    ]
    return dict(
        captain_name="Captain Dart-3", ship_name="TIS Dart-3", ship_type="corvette",
        fleet_data=FLEET, decision_interval_s=30.0, decision_number=7,
        ship_status=ship_status, tactical_status=tactical,
        merge_warnings=["GROUP A: MERGE IN 24 s at 1.2 km (OCS Falchion-1), too late to brake "
                        "(need 375 km incl. flip, have 110 km); pass wide or burn laterally NOW."],
        range_control="=== RANGE CONTROL (world-frame headings for set_heading) ===\n"
                      "GROUP A: OCS Falchion-1 (1 ship) - 110 km, closing 4.68 km/s",
        orders=orders, comms=comms, own_faction="alpha",
        decision_history=("YOUR RECENT DECISIONS (newest last; launches with what they did):\n"
                          "  T+150 EVADE@80% | torp x1 -> OCS Falchion-1 [beta_5]: 1 in flight | "
                          "tgt OCS Falchion-1 [beta_5]"),
        captain_log="YOUR CAPTAIN'S LOG (notes you wrote to your future self - execute or "
                    "supersede them):\n  T+150s: Salvo on Falchion-1, then open range.",
        execution_feedback="ENGINE FEEDBACK (what your ship actually did; weapon policies "
                           "persist, maneuvers and queued launches expire at the next "
                           "checkpoint):\n  helm (T+151): RUN, requested 80% throttle, applied 80%",
        tool_names=["discuss_with_admiral", "fleet_signal", "launch_torpedo", "log_note",
                    "report_to_admiral", "send_message", "set_heading", "set_maneuver",
                    "set_primary_target", "set_radiators"],
        has_admiral=True,
    )


def destroyer_kwargs():
    kw = corvette_kwargs()
    kw.update(captain_name="Vance", ship_name="OCS Falchion-1", ship_type="destroyer",
              orders=None, comms=None, merge_warnings=None, range_control=None,
              captain_log=None, execution_feedback=None, has_admiral=False,
              tool_names=["log_note", "send_message", "set_heading", "set_maneuver",
                          "set_primary_target", "set_radiators", "set_weapons_order"])
    status = dict(kw["ship_status"])
    for key in ("torpedoes_remaining", "torpedo_capacity", "torpedo_tubes", "launch_capacity"):
        status.pop(key)
    status.update(pd_turrets_total=2, pd_turrets_operational=2, gun_groups=[
        {"group": "spinal", "label": "Spinal Coiler Mk3", "mode": "FIRE_WHEN_OPTIMAL",
         "count": 1, "ready": 1, "reloading": [], "destroyed": 0},
        {"group": "coilguns", "label": "Coilgun Battery Mk3", "mode": "FIRE_WHEN_OPTIMAL",
         "count": 1, "ready": 0, "reloading": [8.0], "destroyed": 0}])
    kw["ship_status"] = status
    tactical = dict(kw["tactical_status"])
    tactical["own"] = dict(tactical["own"], ship_id="beta_5", accel_g=2.0, turn_90_s=20.4)
    tactical["torpedo_threats"] = []
    tactical["torpedo_hulls_on_me"] = []
    tactical["evasion_status"] = {"mode": "WOBBLE", "threat_count": 0, "gun_rounds": 0,
                                  "guided_torpedoes": 0, "blinded_wrecks": 0}
    tactical["guns_on_me"] = [{"ship_id": "alpha_9", "name": "TIS Anvil-1", "range_km": 240.0,
                               "p_hit": 0.21, "shots": 4}]
    tactical["current_config"] = {"weapon_orders": {"spinal": "FIRE_WHEN_OPTIMAL"},
                                  "current_maneuver": None}
    forecast = {"weapon_slot": "weapon_1", "weapon_name": "Coilgun Battery Mk3",
                "hit_probability": 0.05, "shots": 3, "window_s": 30.0,
                "expected_hits": 0.15, "expected_gj_per_shot": 0.036}
    tactical["enemies"] = [
        dict(_enemy("alpha_3", "TIS Dart-3", "corvette", 110.3, 4.68,
                    {"x": 0.99, "y": -0.12, "z": -0.02}, 3.0, has_us_targeted=True,
                    torpedoes_remaining=7, pd_operational=1, is_primary_target=True),
             weapon_hit_forecasts=[
                 {"weapon_slot": "weapon_0", "weapon_name": "Spinal Coiler Mk3",
                  "hit_probability": 0.12, "shots": 3, "window_s": 30.0,
                  "expected_hits": 0.36, "expected_gj_per_shot": 0.52},
                 forecast, dict(forecast, weapon_slot="weapon_2")]),
    ]
    tactical["friendlies"] = []
    kw["tactical_status"] = tactical
    return kw


# =============================================================================
# Golden turns
# =============================================================================

class TestGoldenTurn:

    def test_corvette_turn(self):
        turn = build_captain_messages(**corvette_kwargs())[1]["content"]
        assert turn == _golden("captain_turn_corvette.txt", turn), GOLDEN_HINT

    def test_destroyer_turn(self):
        turn = build_captain_messages(**destroyer_kwargs())[1]["content"]
        assert turn == _golden("captain_turn_destroyer.txt", turn), GOLDEN_HINT

    def test_turn_sections_come_in_the_cp06_order(self):
        turn = build_captain_messages(**corvette_kwargs())[1]["content"]
        order = ["=== TACTICAL DATA (T+180s) ===", "MY SHIP:", "THREATS TO ME", "ENEMIES (",
                 "FRIENDLIES (", "=== RANGE CONTROL", "ORDERS FROM YOUR ADMIRAL", "FLEET NET",
                 "ENEMY TRANSMISSIONS", "YOUR RECENT DECISIONS", "YOUR CAPTAIN'S LOG",
                 "ENGINE FEEDBACK", "DECISION POINT 7"]
        positions = [turn.index(marker) for marker in order]
        assert positions == sorted(positions)

    def test_my_ship_is_in_the_world_frame_and_clamped(self):
        turn = build_captain_messages(**corvette_kwargs())[1]["content"]
        assert "position (-25.0,+9.0,+3.0) km" in turn
        assert "velocity (+2.90,-0.50,+0.10) km/s = 2.95 km/s" in turn
        assert "delta-v 500/500 km/s" in turn  # 502.3 from the rocket equation, clamped
        assert "You are Captain Dart-3," in turn  # not "Captain Captain Dart-3"
        assert "EXPIRES NOW" in turn

    def test_volatile_turn_budget(self):
        """
        The turn must not grow past the pre-pass-4 turn (~7.4k characters for
        a corvette in an 8-ship fleet, ~12.7k for a gun hull facing 8): a
        fully loaded fixture turn stays under 7k.
        """
        assert len(build_captain_messages(**corvette_kwargs())[1]["content"]) < 7000


# =============================================================================
# Doctrine: cached, nameless, gated on armament
# =============================================================================

class TestDoctrine:

    def test_system_prompt_is_shared_by_hulls_of_a_class(self):
        a = build_captain_messages(**corvette_kwargs())[0]["content"]
        other = corvette_kwargs()
        other.update(captain_name="Captain Dart-5", ship_name="TIS Dart-5")
        other["tactical_status"] = dict(other["tactical_status"], sim_time=600.0)
        assert build_captain_messages(**other)[0]["content"] == a
        assert "Dart-3" not in a and "Dart-5" not in a

    @pytest.mark.parametrize("ship_type", ["corvette", "cruiser_torpedo"])
    def test_torpedo_only_hulls_carry_no_gun_content(self, ship_type):
        kw = corvette_kwargs()
        kw["ship_type"] = ship_type
        system, turn = (m["content"] for m in build_captain_messages(**kw))
        for text in (system, turn):
            assert "SPINAL" not in text  # "SPINAL CAN FIRE", "SPINAL ALIGNED", ...
            for gun_text in ("set_weapons_order", "spinal_mode", "Spinal coilgun",
                             "Angle to primary target", "AMMO NOTE", "HIT PROBABILITY BY RANGE",
                             "PROJECTILE PHYSICS", "TACTICAL OPTIONS", "spinal masked",
                             "Hit probability", "P(hit)", "Turret coilgun"):
                assert gun_text not in text, gun_text
        assert "TORPEDO BOAT TACTICS" in system

    def test_gun_hulls_keep_gun_doctrine(self):
        system, turn = (m["content"] for m in build_captain_messages(**destroyer_kwargs()))
        assert "HIT PROBABILITY BY RANGE" in system and "set_weapons_order" in system
        assert "TACTICAL OPTIONS" in system and "TORPEDO BOAT TACTICS" not in system
        assert "Spinal Coiler Mk3 FIRE_WHEN_OPTIMAL (1 ready)" in turn
        # Identical mounts grouped, expected damage per shot shown (R10).
        assert "Coilgun Battery Mk3 x2 P(hit) 5%, 6 shots/30s = 0.30 hits" in turn
        assert "~0.52 GJ/shot" in turn

    def test_protocol_and_tool_names_follow_the_offered_tools(self):
        system = build_captain_messages(**corvette_kwargs())[0]["content"]
        assert "fleet_signal" in system and "report_to_admiral" in system
        assert "failing your Admiral" not in system
        legacy = corvette_kwargs()
        legacy["tool_names"] = ["respond_to_orders", "send_message", "set_maneuver"]
        system = build_captain_messages(**legacy)[0]["content"]
        assert "respond_to_orders" in system and "fleet_signal" not in system

    def test_class_reference_lists_every_hull_with_its_armament(self):
        system = build_captain_doctrine(ship_type="destroyer", fleet_data=FLEET)
        for stype in FLEET["ships"]:
            assert class_label(stype) in system
        assert "Torpedo Cruiser" in system and "4x Torpedo tube (48 rounds)" in system
        assert "1x Torpedo tube (8 rounds)" in system
        assert "1x Siege coiler" in system

    def test_class_turn_times_match_the_engine_estimator(self):
        """The CLASS REFERENCE turn times are bound to the engine's turn model."""
        import math
        sim = CombatSimulation(seed=1)
        for stype in FLEET["ships"]:
            ship = create_ship_from_fleet_data("s", stype, "alpha", FLEET, position=Vector3D(0, 0, 0))
            engine = sim._estimate_turn_time(ship, math.pi / 2, engines_on=True)
            assert get_ship_turn_time_90deg(stype) == pytest.approx(engine, abs=1.0), stype


# =============================================================================
# Armament summary (R2) and comms rendering (COMMS-6)
# =============================================================================

class TestArmamentAndComms:

    def test_enemy_torpedo_tubes_are_listed(self):
        assert "Torpedo tube" in format_enemy_weapons_summary(FLEET["ships"]["corvette"]["weapons"])
        text = format_enemy_weapons_summary(FLEET["ships"]["cruiser_torpedo"]["weapons"],
                                            FLEET["weapon_types"], torpedoes_remaining=31,
                                            pd_operational=3)
        assert "4x Torpedo tube (31 rounds left)" in text and "PD 3/4" in text
        siege = format_enemy_weapons_summary(FLEET["ships"]["dreadnought_siege"]["weapons"])
        assert "Siege coiler" in siege
        light = format_enemy_weapons_summary([{"type": "light_coilgun_mk3"}])
        assert "Light coilgun" in light

    def test_inbox_is_split_by_side(self):
        text = format_comms(corvette_kwargs()["comms"], own_faction="alpha")
        fleet, enemy = text.split("ENEMY TRANSMISSIONS")
        assert "TIS Dart-1 [alpha_1] -> fleet" in fleet and "ENGINE" in fleet
        assert "Your corvettes will burn" in enemy and "Your corvettes" not in fleet
        assert "ENEMY TRANSMISSION:" not in text

    def test_items_are_budgeted(self):
        item = {"channel": "fleet_net", "sender_role": "captain", "sender_name": "TIS A",
                "sender_id": "alpha_1", "text": "x" * 500, "sent_t": 10.0}
        text = format_comms([dict(item, msg_id=str(i)) for i in range(12)])
        lines = [l for l in text.splitlines() if l.startswith("  T+")]
        assert len(lines) == 8 and "... 4 more" in text
        assert all(len(l) < 220 for l in lines)

    def test_empty_inbox_renders_nothing(self):
        assert format_comms([]) == "" and format_comms(None) == ""


# =============================================================================
# Orders block (CP-04, CP-07, CP-11, COMMS-9)
# =============================================================================

def _order(text="Hold.", **kw):
    return AdmiralOrder(target_ship_id="a", target_ship_name="A", order_text=text, **kw)


class TestOrdersBlock:

    def test_no_new_directive_shows_the_last_one_with_its_age(self):
        text = format_admiral_orders_for_captain(
            [_order()], "", now=240.0, previous_directive=("Focus Dart-1.", 180.0))
        assert "(no new fleet directive this checkpoint)" in text
        assert "issued T+180, 60 s old" in text and "Focus Dart-1." in text

    def test_orders_carry_their_age(self):
        order = _order(priority="HIGH")
        order.issued_at = 150.0
        text = format_admiral_orders_for_captain([order], "Go.", now=180.0)
        assert "issued T+150, 30 s old" in text
        assert "FLEET DIRECTIVE (issued T+180, this checkpoint): Go." in text

    def test_clarification_replaces_the_order_it_answers(self):
        text = format_admiral_orders_for_captain(
            [_order("Launch ALL 4 on OCS Lancer-4 (86 km).", priority="CRITICAL"),
             _order("[CLARIFICATION] Target OCS Lancer-3 instead.")], "Go.", now=210.0)
        assert "(superseded by the clarification below) [CRITICAL] Launch ALL 4" in text
        assert "REPLACES the orders above" in text
        assert "!!![CRITICAL]" not in text

    def test_standing_orders_shown_when_none_arrived(self):
        text = format_admiral_orders_for_captain(
            [], None, now=240.0, previous_orders=([_order("Hold the wall.")], 210.0))
        assert "No new orders for your ship this checkpoint" in text
        assert "issued T+210, 30 s old" in text and "Hold the wall." in text

    def test_no_mandatory_reply_boilerplate(self):
        text = format_admiral_orders_for_captain([_order()], "Go.", now=0.0,
                                                 reply_tool="report_to_admiral")
        assert "YOU MUST" not in text and "failing your Admiral" not in text
        assert "set_weapons_order" not in text
        assert "IS your acknowledgment" in text and "report_to_admiral" in text


# =============================================================================
# Live captain: order checks, history, names, launch feasibility
# =============================================================================

def _sim_pair(range_km=400.0, closure_kps=0.0, my_type="corvette", their_type="destroyer"):
    sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=5)
    me = create_ship_from_fleet_data("alpha_1", my_type, "alpha", FLEET,
                                     position=Vector3D(0, 0, 0),
                                     velocity=Vector3D(closure_kps * 500.0, 0, 0),
                                     forward=Vector3D(1, 0, 0))
    me.name = "TIS Lancer-1"
    foe = create_ship_from_fleet_data("beta_1", their_type, "beta", FLEET,
                                      position=Vector3D(range_km * 1000.0, 0, 0),
                                      velocity=Vector3D(-closure_kps * 500.0, 0, 0),
                                      forward=Vector3D(-1, 0, 0))
    foe.name = "OCS Lancer-1"
    sim.add_ship(me)
    sim.add_ship(foe)
    return sim, me, foe


def _captain(ship_type="corvette", client=None):
    captain = LLMCaptain(LLMCaptainConfig(name="Captain Test", ship_name="TIS Lancer-1",
                                          ship_type=ship_type, fleet_data=FLEET,
                                          model="anthropic/claude-sonnet-5"),
                         client=client or Mock())
    captain.ship_id = "alpha_1"
    return captain


class TestOrderChecks:

    def test_salvo_at_an_opening_target_is_flagged_unreachable(self):
        sim, me, foe = _sim_pair(400.0, -6.0)
        foe.current_maneuver = Maneuver(ManeuverType.EVASIVE, start_time=0.0, duration=1e9)
        captain = _captain()
        check = captain._order_check(
            _order("COORDINATED SALVO on OCS Lancer-1, do not hold.", torpedo_salvo=1,
                   torpedo_target="OCS Lancer-1"), sim, me, [])
        assert "OPENING" in check["target"]
        assert ("UNREACHABLE" in check["salvo"] or "ESCAPABLE" in check["salvo"])
        assert "DEVIATE" in check["salvo"]

    def test_intercept_order_into_a_merge_is_checked(self):
        sim, me, foe = _sim_pair(150.0, 4.0, my_type="destroyer")
        check = _captain("destroyer")._order_check(
            _order("INTERCEPT OCS Lancer-1 at 80%", suggested_target="OCS Lancer-1"), sim, me,
            [Mock(warning="MERGE IN 30 s")])
        assert "INTERCEPT@80%" in check["engine"] and "MERGE" in check["engine"]
        assert "conflict" in check

    def test_positions_and_numbers_in_the_text_are_resolved(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        check = _captain()._order_check(
            _order("Form on the fleet centre (X=90, Y=55, Z=10); OCS Lancer-1 is 86 km out."),
            sim, me, [])
        (point,) = check["points"]
        assert point.startswith("(90,55,10) km is a POSITION: from you dir_to (+0.85,+0.52,+0.09)")
        (number,) = check["numbers"]
        assert number == ("order quotes 86 km next to OCS Lancer-1 [beta_1]; your sensors: "
                          "400 km from you")
        # By id works too; a number in the next clause is not a claim about the ship.
        check = _captain()._order_check(_order("beta_1 is 86 km out."), sim, me, [])
        assert check["numbers"][0].startswith("order quotes 86 km next to OCS Lancer-1")
        check = _captain()._order_check(_order("Hit OCS Lancer-1. Then hold 300 km."), sim, me, [])
        assert "numbers" not in check

    def test_name_shared_with_a_friendly_is_flagged(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        check = _captain()._order_check(_order(suggested_target="Lancer-1"), sim, me, [])
        assert check["target"].startswith("OCS Lancer-1 [beta_1]")
        assert "also matches your own TIS Lancer-1 [alpha_1]" in check["target"]


class TestOrderTextHeuristics:
    """
    The prose heuristics never print an engine "check" of something the order
    does not say: a CHECK showing an order failing is a reason to DEVIATE.
    """

    def _two_fleets(self):
        """Us: TIS Lancer-1 + TIS Lancer-2 (40 km abeam). Them: OCS Lancer-1/-2."""
        sim, me, foe = _sim_pair(400.0, 0.0)
        mate = create_ship_from_fleet_data("alpha_2", "corvette", "alpha", FLEET,
                                           position=Vector3D(0, 40_000, 0))
        mate.name = "TIS Lancer-2"
        sim.add_ship(mate)
        foe2 = create_ship_from_fleet_data("beta_2", "destroyer", "beta", FLEET,
                                           position=Vector3D(420_000, 0, 0))
        foe2.name = "OCS Lancer-2"
        sim.add_ship(foe2)
        return sim, me

    def test_a_friendly_sharing_the_bare_name_is_not_number_checked(self):
        """CP-17 collision: 'TIS Lancer-2 (40 km)' is our wingman, not OCS Lancer-2."""
        sim, me = self._two_fleets()
        check = _captain()._order_check(
            _order("Form up on TIS Lancer-2 (40 km off your beam) and screen it."), sim, me, [])
        assert "numbers" not in check
        check = _captain()._order_check(
            _order("Screen Lancer-2 (40 km off your beam)."), sim, me, [])
        assert "numbers" not in check  # bare name: ours or theirs - not claimed
        # Named in full (or by id), the enemy is still checked.
        check = _captain()._order_check(_order("OCS Lancer-2 is 40 km out."), sim, me, [])
        assert check["numbers"] == ["order quotes 40 km next to OCS Lancer-2 [beta_2]; your "
                                    "sensors: 420 km from you"]
        check = _captain()._order_check(_order("Lancer-2 [beta_2] is 40 km out."), sim, me, [])
        assert "OCS Lancer-2 [beta_2]" in check["numbers"][0]

    def test_a_foreign_fleet_prefix_hides_a_unique_bare_name(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        foe.name = "OCS Falchion-1"
        captain = _captain()
        assert "numbers" not in captain._order_check(
            _order("Escort TIS Falchion-1 (40 km)."), sim, me, [])
        assert "numbers" in captain._order_check(_order("Falchion-1 is 40 km out."), sim, me, [])

    def test_negated_intercept_and_an_unrelated_percentage_are_not_checked(self):
        sim, me, foe = _sim_pair(150.0, 4.0, my_type="destroyer")
        captain = _captain("destroyer")
        for text in ("Hold position, do NOT intercept; your hull is at 40%.",
                     "Don't INTERCEPT OCS Lancer-1, keep range.",
                     "Avoid an INTERCEPT at all costs."):
            assert "engine" not in captain._order_check(_order(text), sim, me, []), text
        # An INTERCEPT order reads its throttle only next to the word.
        check = captain._order_check(_order("INTERCEPT OCS Lancer-1; your hull is at 40%."),
                                     sim, me, [])
        assert check["engine"].startswith("INTERCEPT@100%")
        check = captain._order_check(_order("No hesitation, INTERCEPT@60% now."), sim, me, [])
        assert check["engine"].startswith("INTERCEPT@60%")
        check = captain._order_check(_order("At 30% INTERCEPT OCS Lancer-1."), sim, me, [])
        assert check["engine"].startswith("INTERCEPT@30%")

    def test_velocity_triples_are_not_positions_and_speeds_not_distances(self):
        sim, me, foe = _sim_pair(300.0, 2.0, my_type="destroyer")
        captain = _captain("destroyer")
        check = captain._order_check(
            _order("OCS Lancer-1 is moving at (3.5, -1.2, 0.4) km/s. Hold here."), sim, me, [])
        assert "points" not in check and "numbers" not in check
        check = captain._order_check(_order("Enemy velocity (3500, -1200, 400), rally at "
                                            "(100, -20, 5)."), sim, me, [])
        assert len(check["points"]) == 1 and check["points"][0].startswith("(100,-20,5) km")
        # "closing at 3 km/s" is a speed, not a 3 km range claim.
        check = captain._order_check(_order("OCS Lancer-1 closing at 3 km/s."), sim, me, [])
        assert "numbers" not in check


class TestStructuredAdmiralManeuver:
    """
    AdmiralOrder.maneuver (maneuver workstream): a structured maneuver the
    runner applies like a heading. Only getattr is used - the field may be
    absent on older order objects.
    """

    def test_no_follow_up_while_an_admiral_maneuver_is_flying(self):
        ack = _respond([("respond_to_orders", {"response_type": "ACKNOWLEDGE"})])
        client, seen = _client({0: ack})
        sim, captain = _fleet_captain(client)
        captain.admiral_orders[0].maneuver = {"maneuver_type": "INTERCEPT",
                                              "target": "OCS Lancer-1", "standoff_km": 50.0,
                                              "throttle": 1.0}
        captain.decide("alpha_1", sim)
        assert len(seen) == 1

    def test_reply_line_names_the_applied_maneuver(self):
        order = _order("Hold 200 km on Lancer-1.")
        order.maneuver = {"maneuver_type": "HOLD_RANGE", "target": "OCS Lancer-1",
                          "range_km": 200.0, "throttle": 1.0}
        text = format_admiral_orders_for_captain([order], "Go.", now=0.0)
        assert "Your MANEUVER ORDER is already flying" in text
        heading = _order("Formation heading.")
        heading.heading = {"x": 1.0, "y": 0.0, "z": 0.0}
        text = format_admiral_orders_for_captain([heading, order], "Go.", now=0.0)
        assert "Your HEADING ORDER/MANEUVER ORDER is already flying" in text

    def test_engine_check_follows_the_spec_not_the_prose(self):
        sim, me, foe = _sim_pair(150.0, 4.0, my_type="destroyer")
        captain = _captain("destroyer")
        warn = [Mock(warning="MERGE IN 30 s")]
        # Standoff INTERCEPT brakes on its own: no straight-burn MERGE claim,
        # even though the prose says INTERCEPT.
        order = _order("INTERCEPT OCS Lancer-1 at 80%, stop 50 km out.")
        order.maneuver = {"maneuver_type": "INTERCEPT", "target": "OCS Lancer-1",
                          "standoff_km": 50.0, "throttle": 0.8}
        check = captain._order_check(order, sim, me, warn)
        assert "engine" not in check and "conflict" not in check
        # A plain structured INTERCEPT is a straight burn: checked at its throttle.
        order = _order("Close the range.")
        order.maneuver = {"maneuver_type": "INTERCEPT", "target": "OCS Lancer-1",
                          "throttle": 0.6}
        check = captain._order_check(order, sim, me, warn)
        assert check["engine"].startswith("INTERCEPT@60%") and "MERGE" in check["engine"]
        assert "conflict" in check
        # HEADING toward a ship flies the bearing to it.
        order = _order("Turn onto them.")
        order.maneuver = {"maneuver_type": "HEADING", "toward_ship": "OCS Lancer-1",
                          "throttle": 1.0}
        check = captain._order_check(order, sim, me, [])
        assert check["engine"].startswith("this heading for one checkpoint")

    def test_tool_notes_ride_their_own_channel(self):
        """An executed call's note is not shown as a REJECTED call to reissue."""
        captain = _captain()
        captain._record_tool_note("set_maneuver HOLD_RANGE: read standoff_km as range_km")
        assert captain.pending_tool_errors == []
        captain.last_tool_notes, captain.pending_tool_notes = captain.pending_tool_notes, []
        text = captain._format_tool_errors()
        assert "ENGINE NOTES ON YOUR LAST ORDERS (accepted" in text
        assert "REJECTED" not in text


class TestNamesAndTargets:

    def test_set_primary_target_resolves_ids_and_refuses_ambiguity(self):
        captain = _captain()
        enemies = [Mock(ship_id="beta_3", name="OCS Lancer-1"),
                   Mock(ship_id="beta_4", name="OCS Lancer-2")]
        sim = Mock(get_enemy_ships=Mock(return_value=enemies))
        call = lambda **a: captain._execute_tool(ToolCall(id="1", name="set_primary_target",
                                                          arguments=a), sim, "alpha_1")
        assert call(target_name="Lancer-2") == {"type": "set_target", "target_id": "beta_4"}
        assert call(target_id="beta_3") == {"type": "set_target", "target_id": "beta_3"}
        assert call(target_name="Lancer") is None
        assert "ambiguous" in captain.pending_tool_errors[-1]
        assert "[beta_3]" in captain.pending_tool_errors[-1]


class TestLaunchFeasibility:

    def _launch(self, captain, sim, **args):
        return captain._execute_tool(ToolCall(id="1", name="launch_torpedo", arguments=args),
                                     sim, "alpha_1")

    def test_unreachable_opening_target_is_refused_unless_overridden(self):
        sim, me, foe = _sim_pair(800.0, -9.0)
        foe.current_maneuver = Maneuver(ManeuverType.EVASIVE, start_time=0.0, duration=1e9)
        captain = _captain()
        sol = captain._torpedo_solution(sim, me, foe)
        assert sol.verdict == "UNREACHABLE" and sol.closing_kps < 0
        assert self._launch(captain, sim, target_id="beta_1") is None
        assert "REFUSED" in captain.pending_tool_errors[-1]
        rounds = self._launch(captain, sim, target_id="beta_1", override_check=True)
        assert rounds and rounds[0]["target_id"] == "beta_1"

    def test_doubtful_launch_goes_ahead_with_a_note(self):
        sim, me, foe = _sim_pair(15.0, 0.0)
        captain = _captain()
        rounds = self._launch(captain, sim, target_id="beta_1")
        assert rounds
        assert "TOO CLOSE" in captain.pending_tool_notes[-1]
        captain.last_tool_notes = captain.pending_tool_notes
        assert "ENGINE NOTES ON YOUR LAST ORDERS" in captain._format_tool_errors()

    def test_trickle_into_point_defense_is_noted(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        for i in range(3):  # escorts: 2 + 3x3 turrets cover him
            esc = create_ship_from_fleet_data(f"beta_{i + 2}", "battleship", "beta", FLEET,
                                              position=Vector3D(400_000, (i + 1) * 30_000, 0))
            sim.add_ship(esc)
        captain = _captain()
        assert self._launch(captain, sim, target_id="beta_1", count=1)
        assert "TRICKLE" in captain.pending_tool_notes[-1]


class TestDecisionHistory:

    def test_launches_headings_targets_and_reports_are_shown_with_outcomes(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        foe.point_defense = []
        captain = _captain()
        captain.decision_history.append({"checkpoint": 1, "time": 0.0, "acted": True, "tool_calls": [
            {"name": "set_heading", "args": {"direction": {"x": 1, "y": 0, "z": 0}, "throttle": 0.7}},
            {"name": "set_primary_target", "args": {"target_name": "OCS Lancer-1"}},
            {"name": "launch_torpedo", "args": {"count": 1, "target_id": "beta_1"}},
            {"name": "report_to_admiral", "args": {"status": "DEVIATE",
                                                   "reason": "Target opening; holding fire."}},
        ]})
        assert sim.inject_command("alpha_1", {"type": "launch_torpedo", "target_id": "beta_1"})
        for _ in range(90):
            sim.step()
        text = captain._format_decision_history(simulation=sim)
        line = text.splitlines()[1]
        assert line.startswith("  T+0 HEADING (+1.00,+0.00,+0.00)@70%")
        assert "tgt OCS Lancer-1 [beta_1]" in line
        assert "report DEVIATE: Target opening; holding fire." in line
        assert "torp x1 -> OCS Lancer-1 [beta_1]: 1 hit" in line and "GJ" in line

    def test_action_less_turn_is_called_out(self):
        captain = _captain()
        captain.decision_history.append({"checkpoint": 1, "time": 30.0, "acted": False,
                                         "followups": 2, "tool_calls": [
                                             {"name": "respond_to_orders",
                                              "args": {"response_type": "ACKNOWLEDGE"}}]})
        line = captain._format_decision_history().splitlines()[1]
        assert line == ("  T+30 report ACK (after 2 follow-ups) - NO COMMANDS: the ship coasted")


# =============================================================================
# decide() harness: follow-up rounds (R1) and lost turns (CP-09)
# =============================================================================

def _respond(calls, finish="tool_calls"):
    message = {"role": "assistant", "content": None if calls else "Acknowledged."}
    if calls:
        message["tool_calls"] = [{"id": f"c{i}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(a)}}
                                 for i, (n, a) in enumerate(calls)]
    return httpx.Response(200, json={
        "id": "gen-test", "model": "anthropic/claude-sonnet-5",
        "choices": [{"message": message, "finish_reason": finish if calls else (
            finish if finish != "tool_calls" else "stop")}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20,
                  "cost": 0.0, "prompt_tokens_details": {"cached_tokens": 0}}})


def _client(script):
    """
    A real CaptainClient on a scripted transport. ``script`` maps the n-th
    captain request (0-based) to a response; anything beyond it goes to the
    stock mock_openrouter, which answers a captain turn with real commands.
    """
    inner = mock_transport({}, 100, 8, 1)
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        n = len(seen) - 1
        if n in script:
            return script[n]
        return inner.handle_request(request)

    client = CaptainClient(model="anthropic/claude-sonnet-5", api_key="mock-no-network",
                           temperature=None)
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    return client, seen


def _fleet_captain(client):
    sim, me, foe = _sim_pair(400.0, 0.0)
    captain = _captain(client=client)
    captain.has_admiral = True
    captain.receive_admiral_orders([_order("Close and launch on OCS Lancer-1.",
                                           suggested_target="OCS Lancer-1")], "Engage.")
    return sim, captain


class TestFollowUpRounds:

    def test_ack_only_reply_gets_a_follow_up_that_yields_commands(self):
        ack = _respond([("respond_to_orders", {"response_type": "ACKNOWLEDGE",
                                               "acknowledgment_note": "Engaging."})])
        client, seen = _client({0: ack})
        sim, captain = _fleet_captain(client)

        commands = captain.decide("alpha_1", sim)

        assert len(seen) == 2, "an ack-only reply must trigger exactly one follow-up"
        follow_up = seen[1]["messages"][-1]["content"]
        assert "=== FOLLOW-UP (same checkpoint, T+0) ===" in follow_up
        assert "respond_to_orders(ACKNOWLEDGE) - recorded" in follow_up
        assert "NO COMMANDS ISSUED" in follow_up
        assert "set_weapons_order" not in follow_up  # corvette: only its own tools
        # The system prefix is unchanged (still cached), the turn is extended.
        assert seen[1]["messages"][0] == seen[0]["messages"][0]
        assert any(isinstance(c, Maneuver) for c in commands)
        assert any(isinstance(c, dict) and c.get("type") == "launch_torpedo" for c in commands)
        entry = captain.decision_history[-1]
        assert entry["followups"] == 1 and entry["acted"] is True
        assert [tc.name for tc in captain.last_requested_tool_calls][0] == "respond_to_orders"
        # The report is not executed twice when the follow-up repeats it.
        assert sum(1 for tc in captain.last_tool_calls if tc.name == "respond_to_orders") == 1

    def test_follow_ups_are_capped_per_checkpoint(self):
        ack = _respond([("log_note", {"note": "Thinking."})])
        client, seen = _client({0: ack, 1: ack, 2: ack, 3: ack})
        sim, captain = _fleet_captain(client)
        assert captain.decide("alpha_1", sim) == []
        assert len(seen) == 3  # the reply + 2 follow-ups, never more
        assert captain.decision_history[-1]["acted"] is False
        # A discussion re-decide in the SAME checkpoint gets no fresh budget.
        captain.revert_last_decision()
        captain.decide("alpha_1", sim)
        assert len(seen) == 4

    def test_no_follow_up_after_an_action_or_a_question(self):
        act = _respond([("set_maneuver", {"maneuver_type": "EVADE"})])
        client, seen = _client({0: act})
        sim, captain = _fleet_captain(client)
        captain.decide("alpha_1", sim)
        assert len(seen) == 1

        ask = _respond([("discuss_with_admiral", {"question": "Which target?"})])
        client, seen = _client({0: ask})
        sim, captain = _fleet_captain(client)
        captain.decide("alpha_1", sim)
        assert len(seen) == 1  # the runner re-decides with the admiral's answer

    def test_no_follow_up_while_an_admiral_heading_is_flying(self):
        ack = _respond([("respond_to_orders", {"response_type": "ACKNOWLEDGE"})])
        client, seen = _client({0: ack})
        sim, captain = _fleet_captain(client)
        captain.admiral_orders[0].heading = {"x": 1.0, "y": 0.0, "z": 0.0}
        captain.decide("alpha_1", sim)
        assert len(seen) == 1


class TestLostTurns:

    def test_truncated_reply_is_retried_then_lost_and_the_maneuver_renewed(self):
        cut = _respond([], finish="length")
        client, seen = _client({0: cut, 1: cut})
        sim, captain = _fleet_captain(client)
        me = sim.get_ship("alpha_1")
        me.current_maneuver = Maneuver(ManeuverType.EVASIVE, start_time=-30.0, duration=30.0,
                                       throttle=0.8)

        commands = captain.decide("alpha_1", sim)

        assert len(seen) == 2 and "=== RETRY" in seen[1]["messages"][-1]["content"]
        assert captain.last_call_failed is True
        assert "finish_reason=length" in captain.last_failure_reason
        (renewed,) = commands
        assert renewed.maneuver_type == ManeuverType.EVASIVE
        assert renewed.start_time == 0.0 and renewed.duration == 30 and renewed.throttle == 0.8
        notice = captain._format_lost_turn()
        assert "YOUR LAST ORDERS WERE LOST" in notice and "renewed your previous EVASIVE@80%" in notice

        # The next turn opens with the notice, once.
        client2, seen2 = _client({})
        captain.client = client2
        sim.current_time = 30.0
        captain.decide("alpha_1", sim)
        assert "YOUR LAST ORDERS WERE LOST" in seen2[0]["messages"][-1]["content"]
        assert "LOST (reply cut off" in seen2[0]["messages"][-1]["content"]
        assert captain.lost_turn is None

    def test_lost_turn_under_a_fresh_admiral_heading_is_not_called_a_coast(self):
        """The runner applied the admiral's HEADING at T+60; the ship flies it on."""
        cut = _respond([], finish="length")
        client, seen = _client({0: cut, 1: cut})
        sim, captain = _fleet_captain(client)
        order = captain.admiral_orders[0]
        order.heading, order.throttle = {"x": 0.0, "y": 1.0, "z": 0.0}, 0.5
        sim.current_time = 60.0
        me = sim.get_ship("alpha_1")
        me.current_maneuver = Maneuver(ManeuverType.HEADING, start_time=60.0, duration=30.0,
                                       throttle=0.5, heading_direction={"x": 0.0, "y": 1.0,
                                                                        "z": 0.0})

        assert captain.decide("alpha_1", sim) == []  # nothing to renew: it is fresh

        notice = captain._format_lost_turn()
        assert "your Admiral's HEADING@50% (applied at T+60) kept flying until T+90" in notice
        assert "COASTED" not in notice
        line = captain._format_decision_history(simulation=sim).splitlines()[-1]
        assert line.endswith("- your Admiral's HEADING@50% kept flying")
        assert "coasted" not in line

        # Not an admiral's: worded without the attribution. No maneuver: a coast.
        captain.admiral_orders[0].heading = None
        captain.client, _ = _client({0: cut, 1: cut})
        sim.current_time = 90.0
        me.current_maneuver = Maneuver(ManeuverType.EVASIVE, start_time=90.0, duration=30.0)
        captain.decide("alpha_1", sim)
        assert "EVASIVE@100% (applied at T+90) kept flying" in captain._format_lost_turn()
        captain.client, _ = _client({0: cut, 1: cut})
        sim.current_time = 120.0
        me.current_maneuver = None
        captain.decide("alpha_1", sim)
        assert "no maneuver was active, so the ship COASTED T+120-150" in \
            captain._format_lost_turn()

    def test_a_retry_that_succeeds_is_a_normal_turn(self):
        client, seen = _client({0: _respond([], finish="error")})
        sim, captain = _fleet_captain(client)
        commands = captain.decide("alpha_1", sim)
        assert len(seen) == 2 and commands
        assert captain.last_call_failed is False and captain.lost_turn is None


class TestLiveTurn:
    """The whole turn built from a live engine state."""

    def test_live_corvette_turn_has_the_new_sections(self):
        sim, me, foe = _sim_pair(400.0, 2.0)
        mate = create_ship_from_fleet_data("alpha_2", "corvette", "alpha", FLEET,
                                           position=Vector3D(0, 50_000, 0))
        sim.add_ship(mate)
        client, seen = _client({})
        captain = _captain(client=client)
        captain.targeting_me = ["beta_1"]
        captain.inbox = [{"msg_id": "x1", "channel": "enemy_broadcast", "sender_role": "captain",
                          "sender_name": "OCS Lancer-1", "sender_id": "beta_1", "faction": "beta",
                          "text": "Surrender.", "sent_t": 0.0}]
        captain.decide("alpha_1", sim)
        turn = seen[0]["messages"][-1]["content"]
        assert "MY SHIP: TIS Lancer-1 [alpha_1] - Corvette" in turn
        assert "position (+0.0,+0.0,+0.0) km" in turn
        assert "OCS Lancer-1 [beta_1] Destroyer | 400 km, closing 2.00 km/s" in turn
        assert "dir_to (+1.00,+0.00,+0.00)" in turn
        assert "torpedo: REACHABLE" in turn
        assert "FRIENDLIES (PD umbrella over you: 2 turrets within 250 km" in turn
        assert "TIS" not in turn.split("ENEMY TRANSMISSIONS")[1].split("\n")[1]
        assert "Surrender." in turn.split("ENEMY TRANSMISSIONS")[1]
        assert "GUNS ON YOU" in turn  # the destroyer has us targeted and in arc


def _place(sim, fleet, sid, stype, faction, x_km, vx_kps=0.0, fwd=1.0, y_km=0.0):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, fleet, position=Vector3D(x_km * 1000.0, y_km * 1000.0, 0.0),
        velocity=Vector3D(vx_kps * 1000.0, 0.0, 0.0), forward=Vector3D(fwd, 0.0, 0.0))
    sim.add_ship(ship)
    return ship


def _place_wreck(sim, source, position, velocity, tid):
    torp = Torpedo(specs=TorpedoSpecs.trident(), position=position, velocity=velocity,
                   target_id="nobody", guidance_mode=GuidanceMode.COAST)
    flight = TorpedoInFlight(torpedo_id=tid, torpedo=torp, source_ship_id=source,
                             is_disabled=True)
    sim.torpedoes.append(flight)
    return flight


class TestThreatDisplayAgreement:
    """CP-01: the THREATS TO ME count and list come from one engine filter."""

    @pytest.fixture
    def fleet(self):
        return FLEET

    def test_captain_display_agrees_with_the_count(self, fleet):
        """The listed wrecks and the counted wrecks come from one filter."""
        from unittest.mock import Mock
        from src.llm.captain import LLMCaptain, LLMCaptainConfig

        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        me = _place(sim, fleet, "alpha_1", "corvette", "alpha", 0.0)
        _place(sim, fleet, "beta_1", "destroyer", "beta", 600.0, fwd=-1.0)
        _place_wreck(sim, "alpha_1", Vector3D(300_000, 0, 0), Vector3D(9_000, 0, 0), "away")
        _place(sim, fleet, "alpha_2", "corvette", "alpha", 0.0, y_km=60.0)
        _place_wreck(sim, "alpha_2", Vector3D(200_000, 1_000, 0), Vector3D(-10_000, 0, 0), "in")
        captain = LLMCaptain(LLMCaptainConfig(name="C", ship_name="TIS A", ship_type="corvette",
                                              fleet_data=fleet), client=Mock())
        tactical = captain._build_tactical_status(me, None, sim)
        wrecks = [t for t in tactical["torpedo_threats"] if t["blinded"]]
        assert len(wrecks) == tactical["evasion_status"]["blinded_wrecks"] == 1
        assert wrecks[0]["own_side"] is True
        assert wrecks[0]["miss_km"] == pytest.approx(1.0, abs=0.01)


class TestLegacyMessages:
    """COMMS-6: CaptainMessage deliveries are labelled by the SENDER's side."""

    def test_friendly_and_enemy_messages_are_split_and_survive_a_re_decide(self):
        from src.llm.communication import CaptainMessage
        sim, me, foe = _sim_pair(400.0, 0.0)
        mate = create_ship_from_fleet_data("alpha_2", "corvette", "alpha", FLEET,
                                           position=Vector3D(0, 40_000, 0))
        mate.name = "TIS Dart-2"
        sim.add_ship(mate)
        client, seen = _client({})
        captain = _captain(client=client)
        captain.receive_messages([
            CaptainMessage(sender_id="alpha_2", sender_name="Captain Dart-2",
                           ship_name="TIS Dart-2", content="Taking the destroyer.", timestamp=0.0),
            CaptainMessage(sender_id="beta_1", sender_name="Captain Foe",
                           ship_name="OCS Lancer-1", content="Surrender.", timestamp=0.0),
        ])
        captain.decide("alpha_1", sim)
        turn = seen[0]["messages"][-1]["content"]
        fleet_net = turn.split("FLEET NET")[1].split("ENEMY TRANSMISSIONS")[0]
        assert "TIS Dart-2 [alpha_2]" in fleet_net and "Taking the destroyer." in fleet_net
        assert "Surrender." in turn.split("ENEMY TRANSMISSIONS")[1]
        assert "ENEMY TRANSMISSION:" not in turn

        captain.revert_last_decision()  # discussion re-decide, same checkpoint
        captain.decide("alpha_1", sim)
        assert "Taking the destroyer." in seen[-1]["messages"][-1]["content"]

        # Next checkpoint: the exchange moves to the side-labelled log.
        sim.current_time = 30.0
        log = captain._format_message_history(before=30.0)
        assert "[TIS Dart-2 [alpha_2] (friendly)]" in log
        assert "[ENEMY OCS Lancer-1 [beta_1]]" in log


class TestInboxEntersTheLogOnce:
    """
    Comms contract: the runner's receive_inbox only STORES captain.inbox; the
    captain's turn renders it (FLEET NET / ENEMY TRANSMISSIONS) and is the one
    place it enters message_history - once per message, even across a
    same-checkpoint re-decide, and never in the same turn's COMMUNICATION LOG.
    """

    WINGMAN = {"msg_id": "cp1-alpha_2-1", "channel": "fleet_net", "kind": "TARGET_CLAIM",
               "sender_role": "captain", "sender_name": "TIS Dart-2", "sender_id": "alpha_2",
               "faction": "alpha", "recipient_spec": "ALL", "recipient_ids": ["alpha_1"],
               "text": "Taking the destroyer.", "sent_t": 0.0, "delivered_t": 0.0}
    DATALINK = {"msg_id": "cp1-engine-1", "channel": "fleet_net", "kind": "LAUNCH",
                "sender_role": "engine", "sender_name": "DATALINK", "sender_id": None,
                "faction": "alpha", "text": "TIS Dart-2 launched 1 torpedo at OCS Lancer-1, "
                "impact ~T+45.", "sent_t": 0.0, "delivered_t": 0.0}

    def _decide(self, captain, sim, seen):
        captain.decide("alpha_1", sim)
        return seen[-1]["messages"][-1]["content"]

    def test_one_inbox_item_is_one_history_entry(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        client, seen = _client({})
        captain = _captain(client=client)
        captain.inbox = [dict(self.WINGMAN), dict(self.DATALINK)]

        turn = self._decide(captain, sim, seen)
        fleet_net = turn.split("FLEET NET")[1]
        assert "Taking the destroyer." in fleet_net and "launched 1 torpedo" in fleet_net
        assert "COMMUNICATION LOG" not in turn  # nothing from EARLIER checkpoints yet
        assert [m["text"] for m in captain.message_history] == ["Taking the destroyer."]

        # A discussion re-decide in the same checkpoint sees the same inbox.
        captain.revert_last_decision()
        turn = self._decide(captain, sim, seen)
        assert "Taking the destroyer." in turn.split("FLEET NET")[1]
        assert len(captain.message_history) == 1

        # Next checkpoint (inbox replaced): in the log once; the datalink
        # telemetry never enters it.
        captain.inbox = []
        sim.current_time = 30.0
        turn = self._decide(captain, sim, seen)
        log = turn.split("COMMUNICATION LOG (earlier checkpoints):")[1]
        assert log.count("Taking the destroyer.") == 1
        assert "[TIS Dart-2 [alpha_2] (friendly)]" in log
        assert "launched 1 torpedo" not in turn
        assert len(captain.message_history) == 1

    def test_items_without_a_msg_id_are_deduplicated_too(self):
        sim, me, foe = _sim_pair(400.0, 0.0)
        client, seen = _client({})
        captain = _captain(client=client)
        item = {k: v for k, v in self.WINGMAN.items() if k != "msg_id"}
        captain.inbox = [item]
        self._decide(captain, sim, seen)
        captain.revert_last_decision()
        self._decide(captain, sim, seen)
        assert len(captain.message_history) == 1
