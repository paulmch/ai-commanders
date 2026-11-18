"""
Admiral command interface, pass 5 wave 5 (lane cmdadm): what the admiral is
shown and can call, each pinned against the code that used to say otherwise.

  facing      enemy facing in the threat table / per-ship ENEMIES, HEADINGS rows
  clock       "checkpoint N of M, battle ends at T+", the pure time-limit scorer
  draft       budget_check schema, echoing validator, tool-result retries, trim
  torpedoes   per-ship torpedo defence forecast (only while rounds are inbound)
  cleanup     blind cones, heat, gun-hold lines, vision legend, dead comms block
  orders      prose vs structured target, salvo allocation on the directive call
  merge       out-accelerated alarm, damped repeats
  escalation  repeated non-compliance in NOW, dead drive in YOUR FLEET
  draw        propose / accept / reject as a recorded exchange
  plot        burning = produced thrust, names in the ticker
"""

import json
import math
import types

import pytest

from src.llm import fleet_intel as intel
from src.llm.admiral import AdmiralDecision, AdmiralOrder, settle_draw_offers
from src.llm.client import LLMCallError, ToolCall
from src.llm.prompts import (
    ADMIRAL_VISION_LEGEND, build_admiral_messages, build_admiral_ship_order_messages,
    build_admiral_static, format_admiral_orders_for_captain, format_admiral_turn_state,
)
from src.physics import Vector3D
from src.simulation import CombatSimulation, SimulationEvent, SimulationEventType

from tests.test_admiral_intel import (
    FLEET, ScriptedClient, add_torpedo, battle_a_sim, captains_for, make_admiral, ship,
)


def own_caps(sim, faction):
    return [types.SimpleNamespace(
        ship_id=sid, ship_name=s.name, name=f"Captain {sid}",
        config=types.SimpleNamespace(ship_type=s.ship_type, name=f"Captain {sid}"))
        for sid, s in sim.ships.items() if s.faction == faction]


def directive_turn(admiral, sim, phase="directive", **kw):
    snap = admiral._build_snapshot(sim, own_caps(sim, admiral.faction), full=True)
    msgs = build_admiral_messages(
        admiral_name="A", faction=admiral.faction, snapshot_t_minus_15=None, snapshot_t_zero=snap,
        personality=None, fleet_data=FLEET, phase=phase, battle_classes=admiral._battle_classes, **kw)
    return snap, msgs


# =============================================================================
# 2. Enemy facing (A4) and the HEADINGS rows (A3)
# =============================================================================

class TestEnemyFacing:
    def test_observed_facing_uses_the_hit_cones_and_live_armour(self):
        sim = CombatSimulation(seed=1)
        me = ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        side = ship(sim, "beta_1", "battlecruiser", "beta", (200, 0, 0), fwd=(0, 1, 0))
        nose = ship(sim, "beta_2", "frigate", "beta", (200, 50, 0), fwd=(-1, -0.25, 0))
        tail = ship(sim, "beta_3", "destroyer", "beta", (300, 0, 0), fwd=(1, 0, 0))
        side.applied_throttle = 1.0
        nose.applied_throttle = 0.05          # PADLOCK's token burn is not a burn
        f = intel.observed_facing(side, me.position)
        assert f["face"] == "SIDE" and f["armor_cm"] == pytest.approx(30.3, abs=0.1)
        assert f["burning"] and intel.facing_text(f) == "SIDE 30cm, burning"
        assert intel.facing_text(intel.observed_facing(nose, me.position)) == "NOSE 71cm, coasting"
        assert intel.observed_facing(tail, me.position)["face"] == "TAIL"
        tail.angular_velocity_rad_s = math.radians(5.0)
        assert intel.facing_text(intel.observed_facing(tail, me.position)).endswith("turning")

    def test_threat_table_and_ship_frame_show_it(self):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "battlecruiser", "beta", (200, 0, 0), fwd=(0, 1, 0))
        admiral = make_admiral("alpha")
        snap, msgs = directive_turn(admiral, sim)
        assert "shows you SIDE 30cm, coasting" in msgs[1]["content"]
        view = admiral._ship_view(sim, "alpha_1", snap)
        order = build_admiral_ship_order_messages(
            admiral_name="A", ship_name="TIS alpha_1", ship_type="destroyer", captain_name="C",
            fleet_directive="x", snapshot=snap, ship_id="alpha_1", ship_view=view)
        assert "shows it SIDE 30cm, coasting" in order[1]["content"]

    def test_him_after_angle_is_the_geometry(self):
        """A 60 s full burn at 1 g straight across the line moves you 17.65 km
        sideways; an enemy 200 km off, nose on you and holding attitude, then
        shows you his nose at atan(17.65 / 200) = 5.0 deg."""
        from src.llm.range_control import HeadingOption
        sim = CombatSimulation(seed=1)
        me = ship(sim, "alpha_1", "battleship", "alpha", (0, 0, 0))
        him = ship(sim, "beta_1", "destroyer", "beta", (200, 0, 0), fwd=(-1, 0, 0))
        opt = HeadingOption(key="left", label="flank left", heading=Vector3D(0, 1, 0), throttle=1.0,
                            turn_s=0.0, burn_s=60.0, effect="~60 s")
        info = types.SimpleNamespace(options=[opt], accel_g=1.0, names=["OCS beta_1"],
                                     cpa_km=0.0, cpa_t_s=0.0)
        text = intel.him_after_text(sim, info, [me], [him], opt)
        lateral_km = 0.5 * 9.80665 * 60.0 ** 2 / 1000.0
        expect = math.degrees(math.atan(lateral_km / 200.0))
        assert text == f"him: NOSE 0->NOSE {expect:.0f} deg"

    def test_every_heading_row_states_both_faces(self):
        _, _, snap, msgs = __import__("tests.test_admiral_intel", fromlist=["battle_a_turn"]).battle_a_turn("beta", t=180)
        rows = [l for l in snap.headings.splitlines() if l.startswith("  ")]
        assert rows
        for row in rows:
            # range_control's own fields (the face you show, what bears) + his face after
            assert " to enemy" in row and "| him: " in row and "->" in row, row


# =============================================================================
# 3. The clock and the score function
# =============================================================================

class TestClock:
    def test_live_runner_and_base_runner_clocks(self):
        # run_live_draft_battle: 20 turns, time limit 21 x 30 s, last orders flown.
        assert intel.battle_clock(630, 20, 30, last_orders_fly=True) == {
            "checkpoints": 20, "end_s": 630.0, "interval_s": 30.0}
        # The base runner stops right after checkpoint M's decisions.
        assert intel.battle_clock(630, 20, 30)["end_s"] == 600.0
        # BattleConfig defaults: 1200 s, 40 checkpoints -> the time limit binds.
        assert intel.battle_clock(1200, 40, 30) == {"checkpoints": 39, "end_s": 1200.0,
                                                    "interval_s": 30.0}
        assert intel.battle_clock(630, 20, 30, unlimited=True) is None

    def test_runner_flag_matches_the_live_runner(self):
        from scripts.run_live_draft_battle import LiveDraftRunner
        from src.llm.battle_runner import LLMBattleRunner
        assert LiveDraftRunner.FLIES_LAST_ORDERS and not LLMBattleRunner.FLIES_LAST_ORDERS

    def test_turn_header_says_checkpoint_n_of_m(self):
        sim = battle_a_sim(180)
        admiral = make_admiral("alpha")
        admiral.battle_clock = intel.battle_clock(630, 20, 30, last_orders_fly=True)
        admiral.decision_count = 5
        _, msgs = directive_turn(admiral, sim)
        assert ("T+180s, checkpoint 6 of 20, battle ends at T+630s - these orders run until "
                "T+210s") in msgs[1]["content"]
        assert "SCORE" not in msgs[1]["content"]
        # At the base runner's last checkpoint the orders are never flown.
        admiral.battle_clock = {"checkpoints": 6, "end_s": 180.0, "interval_s": 30.0}
        _, msgs = directive_turn(admiral, sim)
        assert "the battle ends now: these orders are not flown" in msgs[1]["content"]

    def test_score_line_only_behind_the_flag(self):
        sim = battle_a_sim(180)
        admiral = make_admiral("alpha")
        admiral.config.show_score = True
        _, msgs = directive_turn(admiral, sim)
        a, b = intel.time_limit_scores([s for s in sim.ships.values() if s.faction == "alpha"],
                                       [s for s in sim.ships.values() if s.faction == "beta"])
        assert f"SCORE IF THE BATTLE ENDED NOW (time-limit verdict): you {a:.0f} / them {b:.0f}" \
            in msgs[1]["content"]

    def test_runner_verdict_is_the_pure_function(self):
        """battle_runner._evaluate_fleet_tactical_advantage now calls
        time_limit_scores; the reason string carries the same floats."""
        from src.llm.battle_runner import LLMBattleRunner
        sim = battle_a_sim(180)
        sim.get_ship("alpha_1").is_destroyed = True
        for s in sim.ships.values():
            s.shots_fired, s.hits_scored, s.damage_dealt_gj = 7, 3, 12.5
        runner = LLMBattleRunner.__new__(LLMBattleRunner)
        runner.alpha_ships = {k: v for k, v in sim.ships.items() if v.faction == "alpha"}
        runner.beta_ships = {k: v for k, v in sim.ships.items() if v.faction == "beta"}
        runner.config = types.SimpleNamespace(verbose=False)
        _, _, reason = runner._evaluate_fleet_tactical_advantage()
        a, b = intel.time_limit_scores(runner.alpha_ships.values(), runner.beta_ships.values())
        assert f"{a:.1f}" in reason and f"{b:.1f}" in reason


# =============================================================================
# 4. The draft
# =============================================================================

from src.llm import fleet_draft as fd  # noqa: E402


class DraftClient:
    """Answers select_fleet from a list of ship lists; records every request."""

    def __init__(self, fleets, recorder=None):
        self.fleets = list(fleets)
        self.requests = []
        self.recorder = recorder

    def decide_with_tools(self, messages, tools, model=None, tool_choice=None, **kw):
        self.requests.append({"messages": [dict(m) for m in messages],
                              "tools": tools})
        names = {t["function"]["name"] for t in tools}
        if "select_fleet" in names:
            ships = self.fleets.pop(0)
            return [ToolCall(id=f"call_{len(self.requests)}", name="select_fleet",
                             arguments={"budget_check": "sum", "ships": ships})]
        return [ToolCall(id="f", name="set_formation", arguments={"placements": []})]


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, t, kind, ship_id=None, **data):
        self.events.append((kind, data))


TEN = [{"ship_type": "frigate", "count": 10}]
OVER = [{"ship_type": "battleship", "count": 2}, {"ship_type": "battlecruiser", "count": 2},
        {"ship_type": "destroyer", "count": 2}]   # 86 + 50 + 36 = 172


class TestDraft:
    def test_schema_puts_the_price_sum_first_and_bounds_what_it_can(self):
        tool = fd.select_fleet_tool(150, 8)
        params = tool["function"]["parameters"]
        assert list(params["properties"])[:2] == ["budget_check", "ships"]
        assert params["required"] == ["budget_check", "ships"]
        item = params["properties"]["ships"]["items"]["properties"]
        assert item["count"]["maximum"] == 8
        assert set(item["ship_type"]["enum"]) == set(fd.SHIP_POINT_COSTS)
        assert "150 points AND at most 8 hulls" in tool["function"]["description"]

    def test_error_echoes_the_rejected_fleet(self):
        _, _, error = fd.validate_selection(OVER, 150, 8)
        assert "172 points but you only have 150 (22 over)" in error
        assert "2x battleship (2x43=86)" in error and "= 172 points, 6 hulls" in error
        _, _, error = fd.validate_selection(TEN, 150, 8)
        assert "10 ships, the cap is 8 (2 too many)" in error and "1x" not in error

    def test_retry_is_a_real_tool_exchange(self):
        client = DraftClient([OVER, [{"ship_type": "destroyer", "count": 2}]])
        draft = fd.run_admiral_draft(client, "m", "A", "alpha", FLEET, budget=150, max_ships=8,
                                     verbose=False)
        assert [s.ship_type for s in draft.ships] == ["destroyer", "destroyer"]
        second = client.requests[1]["messages"]
        assistant, result = second[-2], second[-1]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["function"]["name"] == "select_fleet"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["ships"] == OVER
        assert result["role"] == "tool" and result["tool_call_id"] == assistant["tool_calls"][0]["id"]
        assert "REJECTED: Over budget" in result["content"] and "= 172 points" in result["content"]
        assert not any("previous attempt rejected" in str(m.get("content")) for m in second)

    def test_the_exchange_translates_for_the_anthropic_api(self):
        """The retry turn is valid Messages-API input: tool_use then tool_result."""
        from src.llm.anthropic_client import translate_messages
        client = DraftClient([OVER, [{"ship_type": "frigate", "count": 1}]])
        fd.run_admiral_draft(client, "m", "A", "alpha", FLEET, budget=150, max_ships=8, verbose=False)
        _, msgs = translate_messages(client.requests[1]["messages"])
        assert msgs[-2]["role"] == "assistant" and msgs[-2]["content"][0]["type"] == "tool_use"
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"][0]["type"] == "tool_result"

    def test_three_invalid_fleets_are_trimmed_not_aborted(self):
        rec = Recorder()
        client = DraftClient([TEN, OVER, OVER], recorder=rec)
        draft = fd.run_admiral_draft(client, "m", "A", "alpha", FLEET, budget=150, max_ships=8,
                                     verbose=False, allow_auto_fallback=False)
        kept = [s.ship_type for s in draft.ships]
        assert sum(fd.SHIP_POINT_COSTS[t] for t in kept) <= 150 and len(kept) <= 8
        # the cheapest hulls went first: 172 -> drop one destroyer (18) -> 154 -> another -> 136
        assert sorted(kept) == ["battlecruiser", "battlecruiser", "battleship", "battleship"]
        assert draft.trimmed["dropped"] == ["destroyer", "destroyer"]
        assert rec.events[0][0] == "draft_auto_trimmed"
        assert rec.events[0][1]["dropped"] == ["destroyer", "destroyer"]
        assert fd.draft_summary_dict(draft)["auto_trimmed"]["kept"] == draft.trimmed["kept"]

    def test_hull_cap_trim_and_switches(self):
        kept, spent, dropped = fd.trim_selection(TEN, 150, 8)
        assert kept == ["frigate"] * 8 and dropped == ["frigate"] * 2 and spent == 56
        with pytest.raises(RuntimeError):
            fd.run_admiral_draft(DraftClient([TEN, TEN, TEN]), "m", "A", "alpha", FLEET, budget=150,
                                 max_ships=8, verbose=False, allow_auto_fallback=False, allow_trim=False)

    def test_an_api_failure_still_does_not_substitute_a_fleet(self):
        client = types.SimpleNamespace(decide_with_tools=lambda *a, **k: (_ for _ in ()).throw(
            LLMCallError("down")))
        with pytest.raises(RuntimeError):
            fd.run_admiral_draft(client, "m", "A", "alpha", FLEET, verbose=False,
                                 allow_auto_fallback=False)


# =============================================================================
# 5. Torpedo defence (A6)
# =============================================================================

def torpedo_sim():
    sim = CombatSimulation(seed=1)
    ship(sim, "alpha_1", "battlecruiser", "alpha", (0, 0, 0), fwd=(1, 0, 0))
    ship(sim, "alpha_2", "destroyer", "alpha", (0, 20, 0), fwd=(1, 0, 0))
    ship(sim, "beta_1", "corvette", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    for i in range(6):
        add_torpedo(sim, f"t{i}", "beta_1", "alpha_1", (300 + 3 * i, 0, 0), (-6.0, 0, 0))
    return sim


class TestTorpedoDefence:
    def test_forecast_per_targeted_ship(self):
        sim = torpedo_sim()
        rounds = intel.gather_rounds(sim, "alpha")
        d = intel.torpedo_defence(sim, "alpha", FLEET, rounds)
        assert list(d) == ["alpha_1"]
        x = d["alpha_1"]
        assert x["rounds"] == 6 and x["leakers"] == 6 - x["can_blind"]
        assert x["turrets_bearing"] == intel.pd_cover(sim, sim.get_ship("alpha_1"),
                                                      intel.pd_range_m(FLEET),
                                                      threat_from=Vector3D(307500.0, 0, 0))
        assert x["run_heading"].x < -0.99            # straight away from the rounds
        # The engine's own forecast: running after the turn lands no harder, no sooner.
        assert x["run"]["impact_kps"] <= x["hold"]["impact_kps"] * 1.01
        assert x["run"]["first_s"] >= x["hold"]["first_s"]
        line = intel.torpedo_defence_line("TIS alpha_1 [alpha_1]", x)
        assert "6 rounds, first impact" in line and "PD turrets bear" in line
        assert "Burning away heading(-1.00,+0.00,+0.00) @100%" in line

    def test_shown_only_while_rounds_are_inbound(self):
        admiral = make_admiral("alpha")
        _, msgs = directive_turn(admiral, torpedo_sim())
        assert "TORPEDO DEFENCE (per targeted ship" in msgs[1]["content"]
        quiet = torpedo_sim()
        quiet.torpedoes.clear()
        _, msgs = directive_turn(make_admiral("alpha"), quiet)
        assert "TORPEDO DEFENCE (per" not in msgs[1]["content"]

    @pytest.mark.parametrize("cls,dist_km,closing_kps", [
        ("corvette", 800, 3.0), ("battlecruiser", 800, 3.0), ("destroyer", 600, 4.0)])
    def test_forecast_matches_engine_flights(self, cls, dist_km, closing_kps):
        """Fly one round at a hull (no PD) in the engine, holding course and
        burning straight away: the forecast's first impact and impact speed
        when it holds, and 'runs dry before it reaches you' when it runs."""
        from src.simulation import Maneuver, ManeuverType
        out = {}
        for mode in ("hold", "run"):
            sim = CombatSimulation(seed=1, decision_interval=1e9)
            sim.echo = False
            target = ship(sim, "alpha_1", cls, "alpha", (0, 0, 0), fwd=(1, 0, 0))
            ship(sim, "beta_1", "corvette", "beta", (3000, 0, 0), fwd=(-1, 0, 0))
            target.point_defense = []
            add_torpedo(sim, "t0", "beta_1", "alpha_1", (dist_km, 0, 0), (-closing_kps, 0, 0))
            target.current_maneuver = (
                Maneuver(ManeuverType.MAINTAIN, 0.0) if mode == "hold" else
                Maneuver(ManeuverType.HEADING, 0.0, throttle=1.0,
                         heading_direction={"x": -1, "y": 0, "z": 0}))
            forecast = intel.torpedo_defence(sim, "alpha", FLEET,
                                             intel.gather_rounds(sim, "alpha"))["alpha_1"][mode]
            impact = None
            for _ in range(1500):
                sim.step()
                hits = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
                if hits:
                    impact = (sim.current_time, hits[0].data["impact_speed_kps"])
                    break
                if not sim.torpedoes:
                    break
            out[mode] = (forecast, impact)
        hold, flown = out["hold"]
        assert flown is not None and hold["hits"] == 1
        assert hold["first_s"] == pytest.approx(flown[0], abs=3.0)
        assert hold["impact_kps"] == pytest.approx(flown[1], rel=0.06)
        run, flown = out["run"]
        assert flown is None and run["hits"] == 0 and run["dry"] == 1

    def test_eta_is_the_guidance_forecast_not_range_over_closing(self):
        sim = torpedo_sim()
        r = next(r for r in intel.gather_rounds(sim, "alpha") if r.round_id == "t0")
        fly = intel.solve_pursuit(300_000.0, 6000.0, 0.0, 0.0, sim.torpedoes[0].torpedo.specs,
                                  mass_kg=sim.torpedoes[0].torpedo.current_mass_kg,
                                  dv_left_mps=14000.0)
        assert r.eta_s == pytest.approx(fly["tof_s"])
        assert r.eta_s < 300.0 / 6.0      # it keeps burning: sooner than range / closing


# =============================================================================
# 6. Turn-state cleanup (A7)
# =============================================================================

class TestCleanup:
    def test_blind_cones_live_in_forces_not_in_every_ship_line(self):
        sim = battle_a_sim(180)
        sim.weapon_arcs = True
        admiral = make_admiral("alpha")
        _, msgs = directive_turn(admiral, sim)
        assert "ARCS blind cones" not in msgs[1]["content"]
        assert "ARCS roll:" not in msgs[1]["content"]

    def test_shared_gun_holds_are_one_line(self):
        sim = CombatSimulation(seed=1)
        for i in (1, 2, 3):
            ship(sim, f"alpha_{i}", "frigate", "alpha", (0, 10 * i, 0))
            sim.events.append(SimulationEvent(SimulationEventType.WEAPON_STATUS, 1.0, f"alpha_{i}",
                                              None, {"weapon_slot": "weapon_0",
                                                     "reason": "probability_gate"}))
        notes = intel.engine_notes(sim, ["alpha_1", "alpha_2", "alpha_3"], 0.0,
                                   label_of=lambda s: s)
        assert notes == ["guns held: probability_gate (3 ships): alpha_1, alpha_2, alpha_3"]

    def test_vision_legend_moves_to_the_system_prompt(self):
        static = build_admiral_static("A", "alpha", FLEET, None, phase="directive",
                                      battle_classes={"ours": ["frigate"], "theirs": ["frigate"]},
                                      vision=True)
        assert ADMIRAL_VISION_LEGEND in static
        assert ADMIRAL_VISION_LEGEND not in build_admiral_static(
            "A", "alpha", FLEET, None, phase="directive",
            battle_classes={"ours": ["frigate"], "theirs": ["frigate"]})

        class Client(ScriptedClient):
            pass
        client = Client([([ToolCall("d", "set_fleet_directive", {"directive": "x"})], "tool_calls")])
        admiral = make_admiral("alpha", client=client)
        admiral.config.vision = True
        admiral.config.image_input = True
        admiral.decide(simulation=battle_a_sim(180), captains=captains_for("alpha"),
                       battle_image_png=b"\x89PNG fake")
        call = next(c for c in client.calls if "set_fleet_directive" in c["names"])
        text = call["messages"][1]["content"][0]["text"]
        assert "Cyan = alpha" not in text and "Tactical plot attached" in text
        assert "TACTICAL PLOT" in call["messages"][0]["content"]

    def test_directive_call_no_longer_asks_for_per_ship_orders(self):
        static = build_admiral_static("A", "alpha", FLEET, None, phase="directive")
        assert "Issue SPECIFIC ORDERS to EACH captain" not in static
        assert "Issue SPECIFIC ORDERS to EACH captain" in build_admiral_static(
            "A", "alpha", FLEET, None, phase="full")

    def test_dead_communications_block_is_gone(self):
        from src.llm.admiral import LLMAdmiral
        assert not hasattr(LLMAdmiral, "add_to_communications_log")
        text = format_admiral_turn_state(None, FLEET)
        assert "CAPTAIN COMMUNICATIONS" not in text


# =============================================================================
# 7. Order consistency
# =============================================================================

class TestOrderConsistency:
    NAMES = {"alpha_1": "TIS Lancer-1", "beta_1": "OCS Lancer-1", "beta_2": "OCS Falchion-2"}

    def test_prose_naming_another_ship_is_caught(self):
        note = intel.order_text_target_note(
            {"order_text": "Fire at OCS Falchion-2 first, PADLOCK on Lancer-1.",
             "suggested_target": "OCS Lancer-1"}, self.NAMES, ["alpha_1"], ["beta_1", "beta_2"])
        assert "OCS Falchion-2 [beta_2]" in note and "apply OCS Lancer-1 [beta_1]" in note
        # A later "then shift to X" behind the structured target is left to the
        # captain's ORDERS line (40% of recorded orders): no follow-up call.
        assert intel.order_text_target_note(
            {"order_text": "PADLOCK on OCS Lancer-1, then shift to OCS Falchion-2.",
             "suggested_target": "OCS Lancer-1"}, self.NAMES, ["alpha_1"], ["beta_1", "beta_2"]) is None
        assert "no structured target" in intel.order_text_target_note(
            {"order_text": "Engage OCS Falchion-2."}, self.NAMES, ["alpha_1"], ["beta_1", "beta_2"])
        # The own ship's short name is ambiguous ('Lancer-1' is both): not a mismatch.
        assert intel.order_text_target_note(
            {"order_text": "Stay behind Lancer-1, fire at OCS Falchion-2",
             "suggested_target": "beta_2"}, self.NAMES, ["alpha_1"], ["beta_1", "beta_2"]) is None

    def test_mismatch_gets_a_tool_result_and_the_correction_flies(self):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        ship(sim, "beta_2", "frigate", "beta", (300, 40, 0), fwd=(-1, 0, 0))
        answers = [
            [ToolCall("o1", "issue_order", {"ship_name": "TIS alpha_1", "suggested_target": "beta_1",
                                            "order_text": "Engage OCS beta_2 first, then OCS beta_1."})],
            [ToolCall("o2", "issue_order", {"ship_name": "TIS alpha_1", "suggested_target": "beta_2",
                                            "order_text": "Engage OCS beta_2."})],
        ]
        seen = []

        class Client:
            def decide_with_tools(self, messages, tools, model=None, **kw):
                seen.append(messages)
                return answers.pop(0)
        admiral = make_admiral("alpha", client=Client())
        snap = admiral._build_snapshot(sim, own_caps(sim, "alpha"), full=True)
        order = admiral._issue_ship_order(
            {"ship_id": "alpha_1", "ship_name": "TIS alpha_1", "captain_name": "C",
             "ship_type": "destroyer"}, snap, "x", sim)
        assert len(seen) == 2
        assert seen[1][-3]["role"] == "assistant" and seen[1][-2]["role"] == "tool"
        assert "names OCS beta_2 [beta_2]" in seen[1][-2]["content"]
        assert order.suggested_target == "beta_2" and order.text_also_names == []

    def test_captain_is_told_which_target_flies(self):
        order = AdmiralOrder("alpha_1", "TIS alpha_1", "Engage A then B.", suggested_target="OCS A")
        order.text_also_names = ["OCS B [beta_2]"]
        text = format_admiral_orders_for_captain([order], "x", now=30.0)
        assert "The order text also names OCS B [beta_2]: the TARGET above is the ordered one" in text

    def test_salvo_is_allocated_once_on_the_directive_call(self):
        sim = CombatSimulation(seed=1)
        for i in (1, 2, 3):
            ship(sim, f"alpha_{i}", "corvette", "alpha", (0, 10 * i, 0))
        ship(sim, "beta_1", "battleship", "beta", (600, 0, 0), fwd=(-1, 0, 0))
        plan = [{"ship_name": f"TIS alpha_{i}", "rounds": r, "target": "OCS beta_1"}
                for i, r in ((1, 3), (2, 3), (3, 1))]
        prompts = []

        class Client(ScriptedClient):
            def decide_with_tools(self, messages, tools, model=None, **kw):
                if {t["function"]["name"] for t in tools} == {"issue_order"}:
                    prompts.append(messages[1]["content"])
                return super().decide_with_tools(messages, tools, model=model, **kw)
        client = Client([([ToolCall("d", "set_fleet_directive",
                                    {"directive": "7-round strike", "salvo_plan": plan})], "tool_calls")])
        admiral = make_admiral("alpha", client=client)
        decision = admiral.decide(simulation=sim, captains=own_caps(sim, "alpha"))
        salvos = {o.target_ship_id: (o.torpedo_salvo, o.torpedo_target, o.salvo_source)
                  for o in decision.fleet_orders}
        assert salvos == {"alpha_1": (3, "OCS beta_1", "directive"),
                          "alpha_2": (3, "OCS beta_1", "directive"),
                          "alpha_3": (1, "OCS beta_1", "directive")}
        assert sum(s[0] for s in salvos.values()) == 7
        assert all("SALVO ALLOCATION (your directive call, this checkpoint): fleet 7 rounds" in p
                   for p in prompts)


# =============================================================================
# 8. Merge alarm (A9)
# =============================================================================

class TestMergeAlarm:
    def _closing_pair(self, his_g):
        from src.llm.range_control import analyze_range_control
        sim = CombatSimulation(seed=1)
        me = ship(sim, "alpha_1", "battleship", "alpha", (0, 0, 0), vel_kps=(3, 0, 0))
        him = ship(sim, "beta_1", "frigate", "beta", (400, 0.5, 0), vel_kps=(-3, 0, 0),
                   fwd=(-1, 0, 0))
        him.applied_throttle = his_g / 3.0     # a 3g hull burning toward me
        return sim, analyze_range_control(sim, [me], detail=False)[0]

    def test_out_accelerated_alarm_names_it_without_absurd_distances(self):
        _, info = self._closing_pair(his_g=0.9)
        # 0.9g against a 1g battleship is not "out-accelerating" it (review cmdrev):
        # the alarm states what his burn leaves of the braking.
        assert "out-accelerates" not in info.warning and "too late" in info.warning
        assert "his burn toward you (0.9g) leaves 0.1g of your 1.0g to brake" in info.warning
        assert "need" not in info.warning
        assert "cannot hold this range" in info.warning
        _, faster = self._closing_pair(his_g=1.2)
        assert "he out-accelerates you" in faster.warning

    def test_repeats_are_damped(self):
        sim, _ = self._closing_pair(his_g=0.9)
        admiral = make_admiral("alpha")
        caps = own_caps(sim, "alpha")
        first = admiral._build_snapshot(sim, caps, full=True)
        admiral._last_decision_t = sim.current_time
        sim.current_time += 30.0
        second = admiral._build_snapshot(sim, caps, full=True)
        assert first.merge_alarms == {}
        assert "merge alarm unchanged since T+0s (his burn eats your braking; 2 checkpoints running)" \
            in second.merge_alarms["GROUP A"]


# =============================================================================
# 9. Escalating repeated non-compliance (A10)
# =============================================================================

class TestEscalation:
    def test_a_ship_coasting_by_design_is_not_flagged(self):
        """HOLD_RANGE at its range requests no thrust: not a refusal to fly."""
        lines = [snap.noncompliance for _, snap in self._setup(mode="HOLD_RANGE", requested=0.0)]
        assert lines == [[], [], []]

    def _setup(self, mode="BRACE", engine=1.0, executed="HEADING", requested=1.0):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (600, 0, 0), fwd=(-1, 0, 0))
        admiral = make_admiral("alpha")
        caps = own_caps(sim, "alpha")
        for cp in (1, 2, 3):
            t0 = sim.current_time
            order = AdmiralOrder("alpha_1", "TIS alpha_1", "Burn away.", heading={"x": -1, "y": 0, "z": 0},
                                 throttle=1.0)
            admiral.order_history.append(AdmiralDecision(timestamp=t0, fleet_orders=[order]))
            admiral._last_decision_t = t0
            sim.events.append(SimulationEvent(SimulationEventType.EXECUTION_STATE, t0 + 1.0, "alpha_1",
                                              None, {"mode": mode, "applied_throttle": 0.0,
                                                     "requested_throttle": requested,
                                                     "engine_fraction": engine}))
            sim.current_time = t0 + 30.0
            admiral.captain_reports = [{"ship_id": "alpha_1", "ship_name": "TIS alpha_1", "status": "ACK",
                                        "compliance": {"executed_maneuver": executed, "followed": True,
                                                       "deviations": []}}]
            snap = admiral._build_snapshot(sim, caps, full=True)
            yield cp, snap

    def test_two_checkpoints_running_reach_the_now_block(self):
        lines = {cp: snap.noncompliance for cp, snap in self._setup()}
        assert lines[1] == []
        assert "has not flown your HEADING @100% for 2 checkpoints running" in lines[2][0]
        assert "evasion held the drive (BRACE" in lines[2][0]
        assert "for 3 checkpoints running" in lines[3][0]

    def test_a_burn_the_evade_overlay_swings_is_flown_not_held(self):
        """The overlay on a burning order keeps the burn (execution_state
        swing: true): nothing to escalate, and helm_summary books those
        seconds as swing, not as a hold."""
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (600, 0, 0), fwd=(-1, 0, 0))
        admiral = make_admiral("alpha")
        caps = own_caps(sim, "alpha")
        lines = []
        for cp in (1, 2, 3):
            t0 = sim.current_time
            order = AdmiralOrder("alpha_1", "TIS alpha_1", "Burn away.", heading={"x": -1, "y": 0, "z": 0},
                                 throttle=1.0)
            admiral.order_history.append(AdmiralDecision(timestamp=t0, fleet_orders=[order]))
            admiral._last_decision_t = t0
            sim.events.append(SimulationEvent(SimulationEventType.EXECUTION_STATE, t0 + 1.0, "alpha_1",
                                              None, {"mode": "BRACE", "applied_throttle": 1.0,
                                                     "requested_throttle": 1.0,
                                                     "engine_fraction": 1.0, "swing": True}))
            sim.current_time = t0 + 30.0
            admiral.captain_reports = [{"ship_id": "alpha_1", "ship_name": "TIS alpha_1", "status": "ACK",
                                        "compliance": {"executed_maneuver": "HEADING", "followed": True,
                                                       "deviations": []}}]
            lines.append(admiral._build_snapshot(sim, caps, full=True).noncompliance)
        assert lines == [[], [], []]
        from src.llm import fleet_intel
        h = fleet_intel.helm_summary(sim, ["alpha_1"], 60.0, 90.0)["alpha_1"]
        assert h["held_frac"] == 0.0 and h["swing"]["BRACE"] == pytest.approx(h["modes"]["BRACE"])

    def test_dead_drive_is_the_reason_and_marked_in_your_fleet(self):
        snaps = dict(self._setup(engine=0.0))
        assert "drive dead (0 g)" in snaps[2].noncompliance[0]
        snaps[2].friendly_ships[0].drive_fraction = 0.0
        text = format_admiral_turn_state(snaps[2], FLEET)
        assert "DRIVE DEAD (0 g): cannot burn, RCS turns only" in text
        assert "has not flown your HEADING" in text.split("!! NOW")[1].split("===")[0]


# =============================================================================
# 10. Draw as a real exchange (M4)
# =============================================================================

class TestDraw:
    def _pair(self):
        a, b = make_admiral("alpha"), make_admiral("beta")
        return a, b

    def test_reject_clears_the_offer_and_tells_the_proposer(self):
        a, b = self._pair()
        a.has_proposed_draw, a.draw_offer_t = True, 510.0
        a.last_decision = AdmiralDecision(timestamp=510.0, proposed_draw=True)
        b.last_decision = AdmiralDecision(timestamp=510.0)
        assert settle_draw_offers(a, b, 510.0) == [
            ("alpha", "DRAW_PROPOSAL", "Admiral alpha proposes a fleet draw.")]
        b.last_decision = AdmiralDecision(timestamp=540.0, rejected_draw=True)
        a.last_decision = AdmiralDecision(timestamp=540.0)
        out = settle_draw_offers(a, b, 540.0)
        assert out == [("beta", "DRAW_REJECT", "Admiral beta rejects the fleet draw.")]
        assert not a.has_proposed_draw
        assert "REJECTED your draw proposal of T+510s" in a.draw_notices[0]
        # The proposer reads it next turn; the rejecter is not shown the offer again.
        snap = a._build_snapshot(battle_a_sim(180), captains_for("alpha"), full=True)
        assert "REJECTED your draw proposal" in format_admiral_turn_state(snap, FLEET)

    def test_stale_accept_cannot_end_a_battle_later(self):
        a, b = self._pair()
        b.has_accepted_draw = True
        b.last_decision = AdmiralDecision(timestamp=30.0, accepted_draw=True)
        a.last_decision = AdmiralDecision(timestamp=30.0)
        assert settle_draw_offers(a, b, 30.0) == []
        assert not b.has_accepted_draw and "no enemy offer was open" in b.draw_notices[0]

    def test_runner_records_the_exchange_as_diplomacy(self):
        from src.llm.battle_runner import LLMBattleRunner
        runner = LLMBattleRunner.__new__(LLMBattleRunner)
        a, b = self._pair()
        a.has_proposed_draw, a.draw_offer_t = True, 60.0
        a.last_decision = AdmiralDecision(timestamp=60.0, proposed_draw=True)
        b.last_decision = AdmiralDecision(timestamp=60.0)
        runner.alpha_admiral, runner.beta_admiral = a, b
        runner.simulation = types.SimpleNamespace(current_time=60.0)
        runner.config = types.SimpleNamespace(verbose=False)
        recorded = []
        runner._record_diplomacy = lambda *args, **kw: recorded.append((args, kw))
        runner._exchange_admiral_messages()
        assert recorded and recorded[0][1]["kind"] == "DRAW_PROPOSAL"
        assert recorded[0][0][3:5] == ("alpha", "beta")


# =============================================================================
# 11. The tactical plot
# =============================================================================

class TestPlot:
    def test_burning_is_produced_thrust_and_the_plot_names_ships(self):
        from src.llm.admiral_view import build_live_frame, build_scene
        from src.simulation import Maneuver, ManeuverType
        sim = CombatSimulation(seed=1)
        s = ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        s.current_maneuver = Maneuver(ManeuverType.PADLOCK, 1.0, target_id="x")
        s.applied_throttle = 0.05
        frame = build_live_frame(sim)
        assert frame["ships"]["alpha_1"]["thrust"] == pytest.approx(0.05)
        scene = build_scene(frame, {"alpha_1": {"name": "TIS Falchion-1", "type": "destroyer",
                                                "faction": "alpha"}}, faction="alpha")
        item = scene.ships[0]
        assert not item.burning                      # PADLOCK's token burn is coasting
        assert item.label().startswith("TIS Falchion-1")


# =============================================================================
# The multi-turn paths through the real Anthropic client (mock transport)
# =============================================================================

def _anthropic(model, respond, record):
    import anthropic
    from src.llm.anthropic_client import AnthropicClient
    from src.llm.mock_anthropic import mock_anthropic_transport
    transport = mock_anthropic_transport(record=record, respond=respond)
    return AnthropicClient(model, api_key="sk-ant-test-FAKE", max_retries=0,
                           http_client=anthropic.DefaultHttpxClient(transport=transport))


def _tool_use_response(model, name, args, serial):
    from src.llm.mock_anthropic import message_response
    return message_response([{"type": "thinking", "thinking": "adding up", "signature": f"sig-{serial}"},
                             {"type": "tool_use", "id": f"toolu_{serial}", "name": name, "input": args}],
                            "tool_use", model=model)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-sonnet-5"])
class TestThroughTheAnthropicClient:
    def test_draft_retry_replays_the_turn_then_trims_and_records(self, model):
        from types import SimpleNamespace
        from src.llm.battle_recorder import BattleRecorder
        from src.llm.replay_audit import AuditedClient
        record, serial = [], iter(range(1, 100))

        def respond(body):
            if any(t["name"] == "select_fleet" for t in body.get("tools") or []):
                return _tool_use_response(model, "select_fleet",
                                          {"budget_check": "86+50+36=172", "ships": OVER}, next(serial))
            return None
        recorder = BattleRecorder()
        recorder._is_recording = True
        client = AuditedClient(_anthropic(model, respond, record), recorder,
                               SimpleNamespace(current_time=0), "admiral", "alpha")
        draft = fd.run_admiral_draft(client, model, "A", "alpha", FLEET, budget=150, max_ships=8,
                                     verbose=False, allow_auto_fallback=False)
        selects = [b for b in record if any(t["name"] == "select_fleet" for t in b.get("tools") or [])]
        assert len(selects) == 3
        second = selects[1]["messages"]
        # The model's own turn, replayed with its thinking block, then the tool result.
        assert [b["type"] for b in second[-2]["content"]] == ["thinking", "tool_use"]
        assert second[-2]["content"][1]["input"]["ships"] == OVER
        result = second[-1]["content"][0]
        assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
        assert "= 172 points, 6 hulls" in result["content"]
        assert sorted(s.ship_type for s in draft.ships) == ["battlecruiser", "battlecruiser",
                                                           "battleship", "battleship"]
        trims = [e for e in recorder.events if e.event_type == "draft_auto_trimmed"]
        assert len(trims) == 1 and trims[0].data["dropped"] == ["destroyer", "destroyer"]

    def test_order_follow_up_is_a_valid_tool_exchange(self, model):
        record, serial = [], iter(range(1, 100))
        answers = [{"ship_name": "TIS alpha_1", "suggested_target": "beta_1",
                    "order_text": "Engage OCS beta_2 first, then OCS beta_1."},
                   {"ship_name": "TIS alpha_1", "suggested_target": "beta_2",
                    "order_text": "Engage OCS beta_2."}]

        def respond(body):
            if [t["name"] for t in body.get("tools") or []] == ["issue_order"]:
                return _tool_use_response(model, "issue_order", answers.pop(0), next(serial))
            return None
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        ship(sim, "beta_2", "frigate", "beta", (300, 40, 0), fwd=(-1, 0, 0))
        admiral = make_admiral("alpha", client=_anthropic(model, respond, record), model=model)
        snap = admiral._build_snapshot(sim, own_caps(sim, "alpha"), full=True)
        order = admiral._issue_ship_order(
            {"ship_id": "alpha_1", "ship_name": "TIS alpha_1", "captain_name": "C",
             "ship_type": "destroyer"}, snap, "x", sim)
        assert len(record) == 2
        follow = record[1]["messages"]
        assert [b["type"] for b in follow[-2]["content"]] == ["thinking", "tool_use"]
        assert follow[-1]["role"] == "user"
        kinds = [b["type"] for b in follow[-1]["content"]]
        assert kinds[0] == "tool_result" and "text" in kinds   # result + the closing line, one turn
        assert "names OCS beta_2 [beta_2]" in follow[-1]["content"][0]["content"]
        assert order.suggested_target == "beta_2"


# =============================================================================
# The same draft paths through the OpenRouter client with a spend budget
# =============================================================================

def _openrouter_draft_client(fleets, record):
    import httpx
    from src.llm.client import CaptainClient
    from src.llm.spending import RequestBudget
    model = "deepseek/deepseek-v4.1-flash"
    pricing = {model: {"prompt": "0.0000003", "completion": "0.0000012",
                       "input_cache_read": "0.000000006"}}
    fleets = list(fleets)

    def handle(request):
        payload = json.loads(request.content)
        record.append(payload)
        names = [t["function"]["name"] for t in payload.get("tools") or []]
        if "select_fleet" in names:
            name, args = "select_fleet", {"budget_check": "sum", "ships": fleets.pop(0)}
        else:
            name, args = "set_formation", {"placements": []}
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{len(record)}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}
        return httpx.Response(200, json={
            "id": f"gen-{len(record)}", "model": payload["model"],
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
                      "cost": 0.0004, "prompt_tokens_details": {"cached_tokens": 0}}})
    client = CaptainClient(model=model, api_key="test", temperature=None, max_tokens=8192,
                           reasoning_effort="high")
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    client.budget = RequestBudget(1, pricing)
    return client, model


class TestDraftThroughOpenRouter:
    def test_ten_hulls_then_over_budget_then_valid(self):
        record = []
        client, model = _openrouter_draft_client([TEN, OVER, [{"ship_type": "frigate", "count": 8}]],
                                                 record)
        draft = fd.run_admiral_draft(client, model, "A", "alpha", FLEET, budget=150, max_ships=8,
                                     verbose=False, allow_auto_fallback=False)
        assert [s.ship_type for s in draft.ships] == ["frigate"] * 8 and draft.trimmed is None
        third = record[2]["messages"]
        roles = [m["role"] for m in third]
        assert roles[-4:] == ["assistant", "tool", "assistant", "tool"]
        assert "10 ships, the cap is 8 (2 too many)" in third[-3]["content"]
        assert "172 points but you only have 150 (22 over)" in third[-1]["content"]
        assert client.budget.reported_usd > 0      # every call reserved and settled

    def test_ten_hulls_three_times_is_trimmed_to_the_cap(self):
        record = []
        client, model = _openrouter_draft_client([TEN, TEN, TEN], record)
        draft = fd.run_admiral_draft(client, model, "A", "alpha", FLEET, budget=150, max_ships=8,
                                     verbose=False, allow_auto_fallback=False)
        assert [s.ship_type for s in draft.ships] == ["frigate"] * 8
        assert draft.trimmed["dropped"] == ["frigate", "frigate"]


# =============================================================================
# Intent-only command (cmdrun's experiment mode): prompts and tools agree
# =============================================================================

class TestIntentOnly:
    def test_system_prompt_and_tools_match_the_mode_only_when_on(self):
        from src.llm.admiral_tools import get_admiral_tools
        default = build_admiral_static("A", "alpha", FLEET, None, phase="directive")
        intent = build_admiral_static("A", "alpha", FLEET, None, phase="directive", intent_only=True)
        assert "requested in a SEPARATE call" in default
        assert "requested in a SEPARATE call" not in intent
        assert "There is NO per-ship order phase" in intent
        assert "No per-ship orders in this battle" in intent
        tools = {t["function"]["name"]: t["function"]["description"]
                 for t in get_admiral_tools(intent_only=True)}
        assert tools["issue_order"].startswith("INTENT-ONLY COMMAND")
        assert "only order" in tools["set_fleet_directive"]
        plain = {t["function"]["name"]: t["function"]["description"] for t in get_admiral_tools()}
        assert not plain["issue_order"].startswith("INTENT-ONLY")

    def test_intent_only_turn_uses_them_and_keeps_the_salvo_plan(self):
        sim = CombatSimulation(seed=1)
        for i in (1, 2):
            ship(sim, f"alpha_{i}", "corvette", "alpha", (0, 10 * i, 0))
        ship(sim, "beta_1", "battleship", "beta", (600, 0, 0), fwd=(-1, 0, 0))
        plan = [{"ship_name": "TIS alpha_1", "rounds": 2, "target": "OCS beta_1"}]
        client = ScriptedClient([([ToolCall("d", "set_fleet_directive",
                                            {"directive": "x", "salvo_plan": plan})], "tool_calls")])
        admiral = make_admiral("alpha", client=client)
        admiral.config.intent_only = True
        decision = admiral.decide(simulation=sim, captains=own_caps(sim, "alpha"))
        call = client.calls[0]
        assert "There is NO per-ship order phase" in call["messages"][0]["content"]
        assert {o.target_ship_id: o.torpedo_salvo for o in decision.fleet_orders} == {
            "alpha_1": 2, "alpha_2": None}
        assert admiral.last_decision is decision


# =============================================================================
# Gun reach: nominal reach always next to the measured hit rate
# =============================================================================

class TestGunReachWording:
    def test_measured_rate_comes_from_the_gun_exchange_table(self):
        from src.llm.prompts import GUN_EXCHANGE_BANDS, measured_hit_rate
        rates = {b[0]: b[1] for b in GUN_EXCHANGE_BANDS}
        assert measured_hit_rate(10) == ("0-50", rates["0-50"])
        assert measured_hit_rate(149) == ("100-150", rates["100-150"])
        assert measured_hit_rate(150) == ("150-200", rates["150-200"])
        assert measured_hit_rate(510) == ("300+", rates["300+"])

    def test_geometry_says_nominal_and_measured(self):
        from src.llm.prompts import GUN_EXCHANGE_BANDS, measured_hit_rate
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (480, 0, 0), fwd=(-1, 0, 0))
        _, msgs = directive_turn(make_admiral("alpha"), sim)
        line = next(l for l in msgs[1]["content"].splitlines() if l.startswith("  reach - "))
        band, hit = measured_hit_rate(480)
        assert "guns' nominal reach 900 km (within it now)" in line
        assert f"measured hit rate at 480 km (the {band} km band) ~{hit}%" in line
        assert f"{GUN_EXCHANGE_BANDS[0][1]}% at 0-50 km" in line
        assert "IN RANGE" not in line

    def test_forces_captain_caps_plot_label_and_legend_say_what_reach_means(self):
        from src.llm.admiral_view import GUN_BAND_KM, gun_band_label
        from src.llm.prompts import (ADMIRAL_VISION_LEGEND, GUN_EXCHANGE_BANDS,
                                     build_ship_capabilities_from_fleet, measured_hit_rate, reach_note)
        near, far = GUN_EXCHANGE_BANDS[0], GUN_EXCHANGE_BANDS[-1]
        assert reach_note() == (f"Gun reach (km) is how far a round flies, not where it hits: measured "
                                f"hit rate {near[1]}% at {near[0]} km, falling to {far[1]}% at {far[0]} km "
                                "(GUN EXCHANGE, by range at firing).")
        forces = build_admiral_static("A", "alpha", FLEET, None, phase="directive",
                                      battle_classes={"ours": ["destroyer"], "theirs": ["frigate"]})
        assert reach_note() in forces and "(4.3 GJ, reach 900 km)" in forces
        caps = build_ship_capabilities_from_fleet(
            ship_name="x", ship_type="destroyer", fleet_data=FLEET, hull_integrity=100, heat_percent=0,
            delta_v_remaining=500, nose_armor=151, lateral_armor=26, tail_armor=30,
            heatsink_capacity=320, radiators_extended=False)
        assert reach_note() in caps
        band, edge = measured_hit_rate(GUN_BAND_KM - 1)
        assert f"hits {edge}% at {band} km, {far[1]}% at {far[0]} km; not the nominal reach" in gun_band_label()
        flat = " ".join(ADMIRAL_VISION_LEGEND.split())
        assert "the effective gun band, not the guns' reach" in flat
        assert f"measured hit rate {edge}% at {band} km" in flat
