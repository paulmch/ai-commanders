"""
Anti-merge: EVADE without a threat holds its course, HEADING is the range
control tool (turn, then burn; throttle 0 = attitude only), and commanders get
precomputed headings whose stated effects come true in the engine.

Background: in the Sonnet-5 vs DeepSeek battle (data/recordings/
sonnet5_vs_dsv41flash_150pt_20260926) the fleets flew through each other at
~4.6 km/s after a single pass. EVADE with no threat used to INTERCEPT the
nearest enemy with a wobble ("hold the wall" chased), every maneuver closed
range, and nobody was told that a 3 g corvette needs 156 s / 359 km to kill
4.6 km/s of closure.
"""

import json
import math
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    HEADING_BURN_CONE_DEG,
    Maneuver,
    ManeuverType,
    WOBBLE_JINK_ACCEL_S,
    create_ship_from_fleet_data,
)
from src.llm.range_control import (
    MERGE_KM,
    PASS_WIDE_TARGET_KM,
    analyze_range_control,
    admiral_range_control,
    captain_range_control,
    formation_heading_maneuvers,
    formation_throttles,
    ship_accel_ms2,
)

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _ship(sim, sid, stype, faction, pos_km, vel_kps=(0.0, 0.0, 0.0), fwd=(1.0, 0.0, 0.0)):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, FLEET,
        position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(*(c * 1000.0 for c in vel_kps)),
        forward=Vector3D(*fwd))
    ship.name = sid
    sim.add_ship(ship)
    return ship


def _coast(ship):
    ship.current_maneuver = Maneuver(ManeuverType.MAINTAIN, start_time=0.0, duration=0.0)


def _range_km(a, b):
    return (a.position - b.position).magnitude / 1000.0


def _evade(sim, ship_id, throttle=1.0):
    sim.inject_command(ship_id, Maneuver(
        ManeuverType.EVASIVE, start_time=sim.current_time, duration=30.0, throttle=throttle))


def _fly_option(sim, ship, opt, extra_s=0):
    """Fly a precomputed heading for its stated turn+burn time, renewing it
    every 30 s checkpoint exactly as a commander would."""
    t0 = sim.current_time
    t_end = t0 + opt.turn_s + opt.burn_s
    h = {"x": opt.heading.x, "y": opt.heading.y, "z": opt.heading.z}
    while sim.current_time < t_end - 1e-6:
        if (sim.current_time - t0) % 30.0 < 1e-6:
            sim.inject_command(ship.ship_id, Maneuver(
                ManeuverType.HEADING, start_time=sim.current_time,
                duration=min(30.0, t_end - sim.current_time),
                throttle=opt.throttle, heading_direction=h))
        sim.step()
    for _ in range(int(extra_s)):
        sim.step()


# =============================================================================
# A. EVADE with no threat holds its course
# =============================================================================

class TestEvadeWithoutThreatHoldsCourse:

    def test_ship_at_rest_does_not_close_on_the_enemy(self):
        sim = CombatSimulation(seed=4, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        foe = _ship(sim, "foe", "battleship", "beta", (500, 0, 0), fwd=(-1, 0, 0))
        _coast(foe)
        r0 = _range_km(me, foe)
        speeds = []
        for k in range(180):
            if k % 30 == 0:
                _evade(sim, "me")
            sim.step()
            assert me.execution_mode == "WOBBLE"
            speeds.append(me.velocity.magnitude)
        # The old WOBBLE intercepted the nearest enemy: ~475 km closer after
        # 3 minutes at 3 g. Holding course, the weave stays local.
        assert abs(_range_km(me, foe) - r0) < 10.0
        jink = me.kinematic_state.max_acceleration_ms2() * WOBBLE_JINK_ACCEL_S
        assert max(speeds) < 2.0 * jink

    def test_moving_ship_keeps_its_course_and_closure(self):
        sim = CombatSimulation(seed=4, time_step=1.0)
        # Crossing the enemy's bow at 2 km/s while closing at 1 km/s.
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0), vel_kps=(1.0, 2.0, 0.0),
                   fwd=(0, 1, 0))
        foe = _ship(sim, "foe", "destroyer", "beta", (400, 0, 0), fwd=(-1, 0, 0))
        _coast(foe)
        v0 = Vector3D(me.velocity.x, me.velocity.y, me.velocity.z)
        samples = []
        for k in range(240):
            if k % 30 == 0:
                _evade(sim, "me")
            sim.step()
            samples.append(me.velocity)
        mean = Vector3D(sum(v.x for v in samples) / len(samples),
                        sum(v.y for v in samples) / len(samples),
                        sum(v.z for v in samples) / len(samples))
        # Course held: the mean velocity stays on the original vector (the
        # renewed orders keep the latched course - no random walk), and no
        # sample strays further than the weave amplitude.
        jink = me.kinematic_state.max_acceleration_ms2() * WOBBLE_JINK_ACCEL_S
        assert (mean - v0).magnitude < 0.5 * jink
        assert max((v - v0).magnitude for v in samples) < 2.0 * jink
        assert math.degrees(mean.angle_to(v0)) < 3.0

    def test_wobble_has_no_implicit_target(self):
        sim = CombatSimulation(seed=4, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        _ship(sim, "foe", "battleship", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        _evade(sim, "me")
        sim.step()
        assert me.current_maneuver.hold_velocity is not None
        assert me.current_maneuver.target_id is None


# =============================================================================
# HEADING: turn first, then burn; throttle 0 = attitude only
# =============================================================================

class TestHeadingAttitudeAndBurn:

    @pytest.mark.parametrize("stype", ["corvette", "destroyer"])
    def test_throttle_zero_turns_the_nose_without_changing_velocity(self, stype):
        sim = CombatSimulation(seed=1, time_step=1.0)
        ship = _ship(sim, "s", stype, "alpha", (0, 0, 0), vel_kps=(1.0, 0.2, 0.0))
        v0 = Vector3D(ship.velocity.x, ship.velocity.y, ship.velocity.z)
        target = Vector3D(0, 1, 0)
        sim.inject_command("s", Maneuver(
            ManeuverType.HEADING, start_time=0.0, duration=0.0, throttle=0.0,
            heading_direction={"x": 0, "y": 1, "z": 0}))
        # Documented RCS turn time (fleet data, CLAUDE.md "90 deg Turn (RCS)").
        rcs_90 = FLEET["ships"][stype]["attitude_control"]["rcs"]["time_to_rotate_90_deg_s"]
        for _ in range(int(math.ceil(rcs_90 * 1.1))):
            sim.step()
            assert ship.applied_throttle == 0.0
        assert (ship.velocity - v0).magnitude < 1e-9
        assert math.degrees(ship.forward.angle_to(target)) < 1.0

    def test_burning_heading_turns_before_it_burns(self):
        """A 180-degree flip no longer throws the ship sideways."""
        sim = CombatSimulation(seed=1, time_step=1.0)
        ship = _ship(sim, "s", "corvette", "alpha", (0, 0, 0))
        sim.inject_command("s", Maneuver(
            ManeuverType.HEADING, start_time=0.0, duration=0.0, throttle=1.0,
            heading_direction={"x": -1, "y": 0, "z": 0}))
        for _ in range(60):
            sim.step()
        v = ship.velocity
        assert v.x < -1000.0                      # it burned along the heading
        # Sideways residue: only the token turning burn (was ~300 m/s when
        # the drive burned at full throttle all the way round).
        assert math.hypot(v.y, v.z) < 40.0


# =============================================================================
# B'. Precomputed headings do what they say
# =============================================================================

def _duel(own_type, en_type, rng_km, closing_kps, lateral_km=0.0, seed=3):
    sim = CombatSimulation(seed=seed, time_step=1.0)
    me = _ship(sim, "me", own_type, "alpha", (0, 0, 0), vel_kps=(closing_kps, 0, 0))
    foe = _ship(sim, "foe", en_type, "beta", (rng_km, lateral_km, 0), fwd=(-1, 0, 0))
    _coast(foe)
    return sim, me, foe


class TestPrecomputedHeadings:

    @pytest.mark.parametrize("own,foe_type,rng,closing", [
        ("corvette", "battleship", 500.0, 1.5),
        ("destroyer", "cruiser", 400.0, 1.0),
        ("battleship", "corvette", 1500.0, 2.0),
    ])
    def test_cancel_closure_holds_range_as_stated(self, own, foe_type, rng, closing):
        sim, me, foe = _duel(own, foe_type, rng, closing)
        group = analyze_range_control(sim, [me])[0]
        opt = group.option("cancel")
        assert opt.throttle == pytest.approx(1.0)
        assert "TAIL to enemy" in opt.aspect       # the nose points away while braking
        _fly_option(sim, me, opt)
        held = _range_km(me, foe)
        assert held == pytest.approx(opt.end_range_km, rel=0.01)
        assert (me.velocity - foe.velocity).magnitude < 60.0
        _fly_option(sim, me, group.option("face"))  # coast & shoot a checkpoint
        for _ in range(30):
            sim.step()
        assert _range_km(me, foe) == pytest.approx(held, abs=3.0)

    def test_small_closure_is_cancelled_within_one_checkpoint(self):
        sim, me, foe = _duel("corvette", "destroyer", 300.0, 0.3)
        opt = analyze_range_control(sim, [me])[0].option("cancel")
        assert opt.turn_s + opt.burn_s <= 30.0 + 1e-6
        assert 0.0 < opt.throttle < 1.0            # spread over the checkpoint
        _fly_option(sim, me, opt)
        assert (me.velocity - foe.velocity).magnitude < 30.0
        assert _range_km(me, foe) == pytest.approx(opt.end_range_km, rel=0.01)

    def test_pass_wide_turns_a_merge_into_a_wide_pass(self):
        sim, me, foe = _duel("corvette", "battleship", 600.0, 3.0, lateral_km=20.0)
        group = analyze_range_control(sim, [me])[0]
        assert group.cpa_km == pytest.approx(20.0, abs=0.5)
        opt = group.option("wide")
        assert opt.cpa_km >= PASS_WIDE_TARGET_KM
        closest = float("inf")
        t0 = sim.current_time
        _fly_option(sim, me, opt)
        while sim.current_time - t0 < opt.cpa_t_s + 60.0:
            sim.step()
            closest = min(closest, _range_km(me, foe))
        assert closest >= 0.95 * opt.cpa_km
        assert closest > 5.0 * group.cpa_km

    def test_merge_warning_when_braking_is_too_late(self):
        # The recorded merge: 4.6 km/s closure, ~312 km out.
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0), vel_kps=(2.3, 0, 0))
        foe = _ship(sim, "foe", "battleship", "beta", (312, 1.1, 0), vel_kps=(-2.3, 0, 0),
                    fwd=(-1, 0, 0))
        _coast(foe)
        text = captain_range_control(sim, me)
        assert "!! MERGE IN 68 s at 1.1 km" in text
        assert "too late to brake" in text
        assert "stop: 156 s / " in text and " km at 3.0g" in text
        group = analyze_range_control(sim, [me])[0]
        assert group.stop_km == pytest.approx(359.5, abs=1.0)   # 4.6^2 / (2 * 3 g)
        assert "pass wide" in text and "set_heading(" in text
        assert "TAIL to enemy" in text and "nose armor 212 cm" in text

    def test_early_warning_while_braking_still_works(self):
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0), vel_kps=(4.6, 0, 0))
        foe = _ship(sim, "foe", "battleship", "beta", (480, 0.5, 0), fwd=(-1, 0, 0))
        _coast(foe)
        group = analyze_range_control(sim, [me])[0]
        assert group.warning and "braking must start within" in group.warning
        assert group.option("cancel").cpa_km > MERGE_KM

    def test_no_warning_without_a_merge(self):
        sim, me, foe = _duel("destroyer", "corvette", 600.0, 0.8, lateral_km=200.0)
        assert analyze_range_control(sim, [me])[0].warning is None


# =============================================================================
# Formation: one heading, matched acceleration, synchronized burn
# =============================================================================

class TestFormationHeading:

    def _fleet(self, sim):
        ships = [
            _ship(sim, "c", "corvette", "alpha", (0, 0, 0)),
            _ship(sim, "d", "destroyer", "alpha", (0, 20, 0), fwd=(0, 1, 0)),
            _ship(sim, "b", "battleship", "alpha", (0, 40, 0), fwd=(1, 1, 0)),
        ]
        foe = _ship(sim, "foe", "cruiser", "beta", (800, 20, 0), fwd=(-1, 0, 0))
        _coast(foe)
        return ships

    def test_throttles_match_the_slowest_hull(self):
        sim = CombatSimulation(seed=2, time_step=1.0)
        ships = self._fleet(sim)
        thr = formation_throttles(ships, 1.0)
        accels = {s.ship_id: ship_accel_ms2(s) * thr[s.ship_id] for s in ships}
        assert thr["b"] == pytest.approx(1.0)
        assert max(accels.values()) == pytest.approx(min(accels.values()), rel=1e-9)
        maneuvers = formation_heading_maneuvers(sim, ships, {"x": -1, "y": 0.3, "z": 0}, 1.0)
        assert len({m.burn_start_time for m in maneuvers.values()}) == 1

    def test_fleet_heading_keeps_the_formation(self):
        sim = CombatSimulation(seed=2, time_step=1.0)
        ships = self._fleet(sim)

        def spread():
            return [(a.position - b.position).magnitude for a, b in
                    ((ships[0], ships[1]), (ships[1], ships[2]), (ships[0], ships[2]))]

        before = spread()
        start = Vector3D(ships[0].position.x, ships[0].position.y, ships[0].position.z)
        for k in range(180):
            if k % 30 == 0:
                for sid, m in formation_heading_maneuvers(
                        sim, ships, {"x": -1, "y": 0.3, "z": 0}, 1.0).items():
                    sim.inject_command(sid, m)
            sim.step()
        after = spread()
        travelled_km = (ships[0].position - start).magnitude / 1000.0
        assert travelled_km > 50.0
        for b, a in zip(before, after):
            assert abs(a - b) < 2500.0, (before, after)

    def test_admiral_all_ships_order_is_expanded_by_the_runner(self):
        from src.llm.battle_runner import BattleConfig, LLMBattleRunner
        from src.llm.captain import LLMCaptainConfig
        from src.llm.client import CallStats, ToolCall
        from src.llm.fleet_config import BattleFleetConfig
        from src.llm.prompts import CaptainPersonality

        seen = {"captain": [], "phase1": [], "order": []}

        class Stub:
            stats = CallStats()
            session_id = None

            def decide_with_tools(self, messages, tools, model=None, temperature=None,
                                  tool_choice="auto"):
                names = {t["function"]["name"] for t in tools}
                if names == {"issue_order"}:
                    seen["order"].append(messages)
                    return [ToolCall(id="1", name="issue_order", arguments={
                        "ship_name": "x", "order_text": "Hold formation.", "priority": "NORMAL"})]
                if "set_fleet_directive" in names:
                    seen["phase1"].append(messages)
                    return [
                        ToolCall(id="1", name="set_fleet_directive",
                                 arguments={"directive": "Cancel closure as one."}),
                        ToolCall(id="2", name="issue_order", arguments={
                            "ship_name": "all ships", "order_text": "Formation burn.",
                            "heading": {"x": -1, "y": 0, "z": 0}, "throttle": 1.0}),
                    ]
                seen["captain"].append(messages)
                return [ToolCall(id="1", name="respond_to_orders",
                                 arguments={"response_type": "ACKNOWLEDGE"})]

        fleet_config = BattleFleetConfig.from_dict({
            "battle_name": "Formation", "time_limit_s": 300.0, "decision_interval_s": 30.0,
            "initial_distance_km": 500.0,
            "alpha_fleet": {"admiral": {"model": "stub/m"}, "ships": [
                {"ship_type": "corvette", "model": "stub/m"},
                {"ship_type": "battleship", "model": "stub/m"}]},
            "beta_fleet": {"admiral": {"model": "stub/m"}, "ships": [
                {"ship_type": "destroyer", "model": "stub/m"}]},
        })
        config = BattleConfig(initial_distance_km=500.0, time_limit_s=300.0,
                              max_checkpoints=1, verbose=False, personality_selection=False,
                              record_battle=False, parallel_llm=False, seed=7)
        dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/m",
                                 personality=CaptainPersonality.BALANCED)
        runner = LLMBattleRunner(config=config, alpha_config=dummy, beta_config=dummy,
                                 client=Stub(), fleet_config=fleet_config)
        runner.run_fleet_battle(FLEET)

        alpha = [runner.simulation.get_ship(sid) for sid in runner.alpha_captains]
        maneuvers = {s.ship_id: s.current_maneuver for s in alpha}
        assert all(m.maneuver_type == ManeuverType.HEADING for m in maneuvers.values())
        accel = {s.ship_id: ship_accel_ms2(s) * maneuvers[s.ship_id].throttle for s in alpha}
        assert max(accel.values()) == pytest.approx(min(accel.values()), rel=1e-3)
        assert len({m.burn_start_time for m in maneuvers.values()}) == 1
        # The admiral saw range control (its headings, in the live user turn);
        # the per-ship order call saw the formation heading; the captains saw
        # the applied order.
        assert any("=== HEADINGS" in m[-1]["content"] and "cancel closure" in m[-1]["content"]
                   for m in seen["phase1"])
        assert any("FLEET HEADING IN FORCE" in m[-1]["content"] for m in seen["order"])
        captain_turns = [m[-1]["content"] for m in seen["captain"]]
        assert any("HEADING ORDER (ALREADY APPLIED" in t and "FORMATION" in t
                   for t in captain_turns)
        assert any("=== RANGE CONTROL" in t for t in captain_turns)


# =============================================================================
# Prompt / MCP surfaces
# =============================================================================

class TestSurfaces:

    def test_admiral_block_has_formation_throttles(self):
        sim = CombatSimulation(seed=2, time_step=1.0)
        c = _ship(sim, "Dart-1", "corvette", "alpha", (0, 0, 0))
        b = _ship(sim, "Anvil-1", "battleship", "alpha", (0, 20, 0))
        _ship(sim, "Wall-1", "battleship", "beta", (400, 0, 0), vel_kps=(-1.0, 0, 0),
              fwd=(-1, 0, 0))
        text = admiral_range_control(sim, [c, b])
        assert "RANGE CONTROL" in text and "cancel closure" in text
        assert "Dart-1 33%" in text and "Anvil-1 100%" in text
        assert 'ship_name="all ships"' in text

    def test_heading_order_is_shown_as_applied(self):
        from src.llm.admiral import AdmiralOrder
        from src.llm.prompts import format_admiral_orders_for_captain
        order = AdmiralOrder(target_ship_id="a", target_ship_name="A", order_text="Hold.",
                             heading={"x": -1.0, "y": 0.0, "z": 0.0}, throttle=1.0,
                             formation=True, applied_throttle=0.33)
        text = format_admiral_orders_for_captain([order], "Hold the wall.")
        assert "HEADING ORDER (ALREADY APPLIED" in text
        assert "33% throttle" in text and "FORMATION" in text

    def test_mcp_fleet_heading_and_state(self):
        from src.llm.mcp_controller import apply_mcp_commands_to_simulation
        from src.llm.mcp_state import MCPCommand, MCPCommandType
        from src.llm.range_control import mcp_range_control
        sim = CombatSimulation(seed=2, time_step=1.0)
        c = _ship(sim, "c", "corvette", "alpha", (0, 0, 0))
        b = _ship(sim, "b", "battleship", "alpha", (0, 20, 0))
        _ship(sim, "foe", "destroyer", "beta", (500, 0, 0), vel_kps=(-2.0, 0, 0),
              fwd=(-1, 0, 0))
        state = mcp_range_control(sim, [c, b])
        assert state["fleet_groups"][0]["options"][0]["option"] == "cancel"
        assert state["formation_throttles"]["c"] == pytest.approx(1.0 / 3.0, rel=0.01)
        cmd = MCPCommand(MCPCommandType.SET_MANEUVER, ship_id="all", parameters={
            "maneuver_type": "HEADING", "throttle": 1.0,
            "heading_direction": {"x": -1, "y": 0, "z": 0}})
        result = apply_mcp_commands_to_simulation([cmd], sim, "alpha")
        assert {r["ship_id"] for r in result["applied"]} == {"c", "b"}
        assert c.current_maneuver.throttle == pytest.approx(1.0 / 3.0, rel=0.01)
        assert b.current_maneuver.throttle == pytest.approx(1.0)
        assert c.current_maneuver.burn_start_time == b.current_maneuver.burn_start_time


class TestHeuristicCaptainRangeControl:

    def _captain(self, stype):
        from src.llm.captain import LLMCaptainConfig
        from src.llm.heuristic_captain import HEURISTIC_MODEL, HeuristicCaptain
        from src.llm.prompts import CaptainPersonality
        return HeuristicCaptain(LLMCaptainConfig(
            name="H", ship_name="me", model=HEURISTIC_MODEL,
            personality=CaptainPersonality.BALANCED, ship_type=stype, fleet_data=FLEET))

    def _order(self, text, heading=None):
        from src.llm.admiral import AdmiralOrder
        return AdmiralOrder(target_ship_id="me", target_ship_name="me", order_text=text,
                            heading=heading, throttle=1.0 if heading else None)

    def test_keeps_an_applied_admiral_heading(self):
        sim, me, foe = _duel("destroyer", "cruiser", 400.0, 1.0)
        captain = self._captain("destroyer")
        captain.receive_admiral_orders([self._order("Formation burn.", {"x": -1, "y": 0, "z": 0})])
        captain.decide("me", sim)
        names = [tc.name for tc in captain.last_tool_calls]
        assert "set_maneuver" not in names and "set_heading" not in names

    def test_keep_range_order_flies_the_cancel_closure_heading(self):
        sim, me, foe = _duel("destroyer", "cruiser", 400.0, 1.0)
        captain = self._captain("destroyer")
        captain.receive_admiral_orders([self._order("Keep range at 400 km, do not chase.")])
        commands = captain.decide("me", sim)
        heading = [c for c in commands if isinstance(c, Maneuver)]
        assert heading and heading[0].maneuver_type == ManeuverType.HEADING
        h = heading[0].heading_direction
        assert h["x"] < -0.99                   # against the 1 km/s closure

    def test_torpedo_boat_inside_standoff_burns_to_hold_range(self):
        sim, me, foe = _duel("corvette", "battleship", 200.0, 1.5)
        captain = self._captain("corvette")
        commands = captain.decide("me", sim)
        maneuvers = [c for c in commands if isinstance(c, Maneuver)]
        assert maneuvers[0].maneuver_type == ManeuverType.HEADING


# =============================================================================
# Determinism
# =============================================================================

def test_evade_weave_and_range_control_are_deterministic():
    def run():
        sim = CombatSimulation(seed=11, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0), vel_kps=(1.0, 0.5, 0))
        foe = _ship(sim, "foe", "battleship", "beta", (450, 30, 0), fwd=(-1, 0, 0))
        _coast(foe)
        text = captain_range_control(sim, me)
        for k in range(150):
            if k % 30 == 0:
                _evade(sim, "me")
            sim.step()
        return text, me.position.to_tuple(), me.velocity.to_tuple()

    assert run() == run()


def test_heading_burn_cone_constant_is_what_the_estimates_assume():
    # range_control's burn-start model and the engine share this constant.
    assert 0.0 < HEADING_BURN_CONE_DEG <= 15.0


# =============================================================================
# Evidence: a corvette group kites a charging battleship line at ~300 km
# =============================================================================

def test_corvette_group_kites_a_battleship_line_with_the_precomputed_headings():
    """
    A commander that only ever picks one of the precomputed options - coast &
    shoot while braking now would still stop outside the band, cancel closure
    once the stated hold range falls into it, open range if pushed inside -
    flown as ONE formation heading for the group, holds ~300 km against a
    battleship line that INTERCEPTs at full burn: no merge, formation intact.
    """
    import itertools
    target_km, tol_km = 300.0, 30.0
    sim = CombatSimulation(seed=5, time_step=1.0)
    noses = [(1, 0, 0), (0, 1, 0), (-1, 1, 0), (0, -1, 0)]
    darts = [_ship(sim, f"dart{i}", "corvette", "alpha", (0, 15 * i, 0), fwd=noses[i])
             for i in range(4)]
    walls = [_ship(sim, f"wall{i}", "battleship", "beta", (700, -10 + 20 * i, 0),
                   fwd=(-1, 0, 0)) for i in range(3)]

    def spacing():
        return [(a.position - b.position).magnitude for a, b in itertools.combinations(darts, 2)]

    spacing0 = spacing()
    in_band, closest, worst_dev = [], float("inf"), 0.0
    for _ in range(22):
        for w in walls:
            prey = min(darts, key=lambda d: (d.position - w.position).magnitude)
            sim.inject_command(w.ship_id, Maneuver(
                ManeuverType.INTERCEPT, start_time=sim.current_time, duration=30.0,
                throttle=1.0, target_id=prey.ship_id))
        group = analyze_range_control(sim, darts)[0]
        cancel = group.option("cancel")
        if group.range_km < target_km - tol_km and group.closing_kps > -0.3:
            opt = group.option("open")
        elif cancel.end_range_km > target_km + tol_km and not group.warning:
            opt = group.option("face")
        else:
            opt = cancel
        for sid, m in formation_heading_maneuvers(sim, darts, opt.heading, opt.throttle).items():
            sim.inject_command(sid, m)
        for _ in range(30):
            sim.step()
            closest = min(closest, min(_range_km(d, w) for d in darts for w in walls))
        centre = lambda ships: Vector3D(sum(s.position.x for s in ships) / len(ships),
                                        sum(s.position.y for s in ships) / len(ships), 0.0)
        rng = (centre(darts) - centre(walls)).magnitude / 1000.0
        in_band.append(abs(rng - target_km) <= tol_km)
        worst_dev = max(worst_dev, max(abs(a - b) for a, b in zip(spacing(), spacing0)))

    streak = max(len(list(g)) for held, g in itertools.groupby(in_band) if held)
    assert streak >= 10, in_band
    assert closest > target_km - 2 * tol_km          # never merged, never even close
    assert worst_dev < 1000.0                         # formation held to < 1 km
    assert walls[0].velocity.magnitude > 5000.0       # the line really was charging
