"""
Fixes from the adversarial review of the weapon-arcs merge (pass 5, wave 4).

Each class pins one finding: a behaviour the merged engine got wrong, with a
scenario that failed on it.
"""

import math

import pytest

from src import weapon_arcs as wa
from src.physics import Vector3D
from src.simulation import CombatSimulation, SimulationEventType, WeaponsCommand, WeaponsOrder

from tests.test_weapon_arcs import ALL_OPEN, _fire_at, _ship, _sim, _table, _torpedo


def _draft_battle(seed, budget=200, time_limit=600.0, on_sim=None):
    """One auto-drafted heuristic battle, as `run_draft_battle.py --auto-draft
    --no-admirals --seed SEED --budget BUDGET`; ``on_sim(sim)`` runs once the
    simulation exists. Returns the runner."""
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import auto_draft, draft_to_fleet_definition
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality

    drafts = {f: auto_draft(f, budget=budget, max_ships=8, seed=seed + (0 if f == "alpha" else 1))
              for f in ("alpha", "beta")}
    fleet_config = BattleFleetConfig(
        battle_name="arcs review",
        alpha_fleet=draft_to_fleet_definition(drafts["alpha"], 500.0,
                                              captain_model=HEURISTIC_MODEL, admiral_config=None),
        beta_fleet=draft_to_fleet_definition(drafts["beta"], 500.0,
                                             captain_model=HEURISTIC_MODEL, admiral_config=None),
        time_limit_s=time_limit, initial_distance_km=500.0,
        record_battle=False, record_sim_trace=False,
        personality_selection=False, use_notebooks=False)
    config = BattleConfig(initial_distance_km=500.0, time_limit_s=time_limit, verbose=False,
                          personality_selection=False, record_battle=False,
                          record_sim_trace=False, record_predictions=False, seed=seed)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(config=config, alpha_config=dummy, beta_config=dummy,
                             client=None, fleet_config=fleet_config)
    setup = LLMBattleRunner.setup_fleet_battle

    def probed_setup(self, data):
        setup(self, data)
        self.simulation.echo = False
        if on_sim is not None:
            on_sim(self.simulation)

    LLMBattleRunner.setup_fleet_battle = probed_setup
    try:
        runner.run_battle(load_fleet_data())
    finally:
        LLMBattleRunner.setup_fleet_battle = setup
    return runner


# =============================================================================
# Finding 1: the roll goal hunted between two attitudes on alternate re-plans
# =============================================================================

class TestRollGoalDoesNotHunt:

    ABA_WINDOW_S = 30.0
    SAME_GOAL_DEG = 20.0

    @pytest.mark.parametrize("seed", [103, 104])
    def test_no_goal_returns_to_the_one_it_just_left(self, seed, monkeypatch):
        """Seeds 103/104 (budget 200): the merged planner flipped some hulls'
        goal every 5 s between two attitudes ~90 degrees apart (27 and 35
        quick reversals) while the hull, needing 70-250 s per 90 degrees,
        reached neither. No goal may now come back within 20 degrees of the
        goal it replaced inside 30 s."""
        goals = {}
        plan = CombatSimulation._arcs_plan_roll

        def logged(self, ship, arcs):
            before = arcs.roll_goal
            plan(self, ship, arcs)
            if arcs.roll_goal != before:
                goals.setdefault(ship.ship_id, []).append(
                    (self.current_time, before, arcs.roll_goal))

        monkeypatch.setattr(CombatSimulation, "_arcs_plan_roll", logged)
        _draft_battle(seed)
        assert goals, "no hull ever rolled - the scenario no longer exercises the planner"
        hunts = []
        for sid, changes in goals.items():
            for (t0, a, b), (t1, b2, c) in zip(changes, changes[1:]):
                if (t1 - t0 <= self.ABA_WINDOW_S
                        and abs(math.degrees(c - a)) <= self.SAME_GOAL_DEG):
                    hunts.append((sid, t0, math.degrees(a), math.degrees(b), t1, math.degrees(c)))
        assert not hunts, hunts

    def test_the_gun_hysteresis_margin_holds_a_standing_goal(self, monkeypatch):
        """A candidate a little better for the guns (less than
        ROLL_GUN_HYSTERESIS_FRACTION of the hull's gun energy) must not
        displace the standing goal; one clearly better does."""
        sim = _sim(_table(), roll=True)
        ship = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        sim.step()
        arcs = ship.arcs
        mount = arcs.guns["weapon_1"]
        weight = 10.0
        samples = len(wa.ROLL_MARGIN_SAMPLES_RAD)
        margin = wa.ROLL_GUN_HYSTERESIS_FRACTION * samples * weight
        monkeypatch.setattr(sim, "_arcs_roll_inputs",
                            lambda s, a: ([], [], [(mount, weight)]))
        for gain, moves in ((0.5 * margin, False), (2.0 * margin, True)):
            arcs.roll = arcs.roll_goal = 0.0
            arcs.goal_since = None
            arcs.goal_eta = 0.0
            target = math.radians(wa.ROLL_CANDIDATE_STEP_DEG)

            def score(phi, torps, pd, guns, reach_s=0.0, gain=gain, target=target):
                return (0, 5.0 + (gain if abs(phi - target) < 1e-9 else 0.0))

            monkeypatch.setattr(sim, "_arcs_roll_score", score)
            sim._arcs_plan_roll(ship, arcs)
            assert (arcs.roll_goal != 0.0) is moves, (gain, margin, arcs.roll_goal)

    def test_a_goal_it_cannot_reach_before_impact_does_not_displace_one_it_can(self):
        """A round ~15 s out on a bearing only a ~180 degree roll opens for
        the PD: a destroyer needs ~100 s for that roll, so the goal stays with
        the gun target (the merged planner rolled for the torpedo anyway)."""
        table = _table(pd0=lambda az, el: el < 50, pd1=lambda az, el: el < 50)
        sim = _sim(table, roll=True)
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(0, 150_000, 60_000), forward=Vector3D(0, -1, 0))
        _fire_at(shooter, "b")
        sim.step()
        goal = shooter.arcs.roll_goal
        pos = Vector3D(0, -60_000, -200_000)
        vel = pos.normalized() * -14_000.0        # straight in at 14 km/s: ~15 s
        _torpedo(sim, "t", pos, vel, "a", "b")
        accel, rate = sim._arcs_roll_params(shooter)
        assert wa.roll_time_s(math.pi, accel, rate) > 60.0
        for _ in range(6):
            sim.step()
        assert shooter.arcs.roll_goal == goal
