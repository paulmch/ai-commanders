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

    def test_every_heading_row_states_both_faces(self):
        _, _, snap, msgs = __import__("tests.test_admiral_intel", fromlist=["battle_a_turn"]).battle_a_turn("beta", t=180)
        rows = [l for l in snap.headings.splitlines() if l.startswith("  ")]
        assert rows
        for row in rows:
            assert "| you: " in row and "| him: " in row and "->" in row, row
        coast = next(r for r in rows if "throttle 0 or PADLOCK" in r)
        assert "no range change" in coast   # the one row that used to have no downside


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
        # The engine's guidance law: running after the turn lands slower, later.
        assert x["run"]["impact_kps"] <= x["hold"]["impact_kps"]
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
            {"order_text": "PADLOCK on Lancer-1, then shift to OCS Falchion-2.",
             "suggested_target": "OCS Lancer-1"}, self.NAMES, ["alpha_1"], ["beta_1", "beta_2"])
        assert "OCS Falchion-2 [beta_2]" in note and "apply OCS Lancer-1 [beta_1]" in note
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
                                            "order_text": "Engage OCS beta_1, then OCS beta_2."})],
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
        assert "he out-accelerates you" in info.warning and "too late" in info.warning
        assert "need" not in info.warning
        assert "cannot hold this range" in info.warning

    def test_repeats_are_damped(self):
        sim, _ = self._closing_pair(his_g=0.9)
        admiral = make_admiral("alpha")
        caps = own_caps(sim, "alpha")
        first = admiral._build_snapshot(sim, caps, full=True)
        admiral._last_decision_t = sim.current_time
        sim.current_time += 30.0
        second = admiral._build_snapshot(sim, caps, full=True)
        assert first.merge_alarms == {}
        assert "merge alarm unchanged since T+0s (he out-accelerates you; 2 checkpoints running)" \
            in second.merge_alarms["GROUP A"]


# =============================================================================
# 9. Escalating repeated non-compliance (A10)
# =============================================================================

class TestEscalation:
    def _setup(self, mode="BRACE", engine=1.0, executed="HEADING"):
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
                                                     "requested_throttle": 1.0, "engine_fraction": engine}))
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
    def test_burning_is_produced_thrust_and_the_ticker_names_ships(self):
        from src.llm.admiral_view import AdmiralViewState, build_live_frame
        from src.simulation import Maneuver, ManeuverType
        sim = CombatSimulation(seed=1)
        s = ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        s.current_maneuver = Maneuver(ManeuverType.PADLOCK, 1.0, target_id="x")
        s.applied_throttle = 0.05
        assert build_live_frame(sim)["ships"]["alpha_1"]["thrust"] == pytest.approx(0.05)
        view = AdmiralViewState()
        view.note_event(types.SimpleNamespace(event_type=types.SimpleNamespace(name="TORPEDO_IMPACT"),
                                              timestamp=10.0, ship_id="alpha_1", data={}))
        assert view.recent_events(12.0, names={"alpha_1": "TIS Falchion-1"}) == [
            (10.0, "TORPEDO IMPACT: TIS Falchion-1")]
