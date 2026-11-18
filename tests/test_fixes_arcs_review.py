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


def _draft_battle(seed, budget=200, time_limit=600.0, on_sim=None, doctrine=None):
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
                          record_sim_trace=False, record_predictions=False, seed=seed,
                          heuristic_doctrine={"alpha": doctrine, "beta": doctrine} if doctrine else {})
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

    # The scenario the fix was measured on: the captains of that time
    # ("classic", the default until the wall/pincer/close-and-pass switch),
    # and the current default where it does not hit the open case below.
    @pytest.mark.parametrize("seed,doctrine", [
        (103, "classic"), (104, "classic"), (104, "tactical"),
    ])
    def test_no_goal_returns_to_the_one_it_just_left(self, seed, doctrine, monkeypatch):
        """Seeds 103/104 (budget 200): the merged planner flipped some hulls'
        goal every 5 s between two attitudes ~90 degrees apart (27 and 35
        quick reversals) while the hull, needing 70-250 s per 90 degrees,
        reached neither. No goal may now leave for a different attitude (more
        than 20 degrees away) and come back within 20 degrees inside 30 s -
        unless the threat picture it left for is gone: a hull that rolled to
        meet a pair of inbound rounds and rolls back once they have struck or
        passed is reacting, not hunting (seed 104 after the launch sequence,
        src/torpedo_launch.py, shifted the torpedo waves: alpha_6 rolls -120
        for two rounds 25 km out at T+95 and back at T+110, after they hit)."""
        goals = {}
        plan = CombatSimulation._arcs_plan_roll

        def logged(self, ship, arcs):
            before = arcs.roll_goal
            plan(self, ship, arcs)
            if arcs.roll_goal != before:
                horizon_m = (wa.ROLL_PD_HORIZON_ENVELOPES * 1000.0
                             * max((pd.laser.range_km for pd in ship.point_defense),
                                   default=250.0))
                threats = frozenset(
                    tf.torpedo_id for tf in self.torpedoes
                    if not tf.is_disabled and self.get_ship(tf.source_ship_id) is not None
                    and self.get_ship(tf.source_ship_id).faction != ship.faction
                    and tf.torpedo.position.distance_to(ship.position) <= horizon_m)
                goals.setdefault(ship.ship_id, []).append(
                    (self.current_time, before, arcs.roll_goal, threats))

        monkeypatch.setattr(CombatSimulation, "_arcs_plan_roll", logged)
        _draft_battle(seed, doctrine=doctrine)
        assert goals, "no hull ever rolled - the scenario no longer exercises the planner"
        hunts = []
        for sid, changes in goals.items():
            for (t0, a, b, th0), (t1, b2, c, th1) in zip(changes, changes[1:]):
                if (t1 - t0 <= self.ABA_WINDOW_S
                        and abs(math.degrees(b - a)) > self.SAME_GOAL_DEG
                        and abs(math.degrees(c - a)) <= self.SAME_GOAL_DEG
                        and th0 <= th1):         # nothing it rolled for has gone
                    hunts.append((sid, t0, math.degrees(a), math.degrees(b), t1, math.degrees(c)))
        assert not hunts, hunts

    @pytest.mark.xfail(strict=True, reason=(
        "engine roll planner (CombatSimulation._arcs_plan_roll), PD-first branch: under a "
        "swarm of 7-10 inbound torpedoes a flanking destroyer (seed 103, beta_6) sets its "
        "goal -45 -> -24 -> 7 -> -21 deg (t = 70, 85, 100, 115 s) while its orders are "
        "constant (same gun target, same HEADING toward_point): scored on only the 7 "
        "torpedoes present at both t = 100 and 115 the best roll still moves 7 -> -21, so "
        "it is the torpedoes' geometry, not attrition; a PD gain skips the goal_eta wait, "
        "each re-plan comes before the hull (roll -23 -> -21) gets anywhere near the goal. "
        "Owner: the arcs/roll code, not the bots."))
    def test_pd_first_goal_does_not_reverse_under_a_torpedo_swarm(self, monkeypatch):
        """Smallest reproduction: seed 103 (budget 200, default doctrine) to 120 s,
        beta_6 only."""
        changes = []
        plan = CombatSimulation._arcs_plan_roll

        def logged(self, ship, arcs):
            before = arcs.roll_goal
            plan(self, ship, arcs)
            if ship.ship_id == "beta_6" and arcs.roll_goal != before:
                changes.append((self.current_time, before, arcs.roll_goal))

        monkeypatch.setattr(CombatSimulation, "_arcs_plan_roll", logged)
        _draft_battle(103, time_limit=120.0, doctrine="tactical")
        assert changes, "beta_6 never re-planned its roll - the scenario moved"
        for (t0, a, b), (t1, b2, c) in zip(changes, changes[1:]):
            assert not (t1 - t0 <= self.ABA_WINDOW_S
                        and abs(math.degrees(b - a)) > self.SAME_GOAL_DEG
                        and abs(math.degrees(c - a)) <= self.SAME_GOAL_DEG), (t0, t1)

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


# =============================================================================
# Finding 3: a fire_at slewed the mount twice in one tick and never fired
# =============================================================================

class TestFireAtRetargetsTheMountForTheTick:

    def _scene(self):
        sim = _sim(_table(**ALL_OPEN))
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))   # dead ahead
        _ship(sim, "c", "beta", Vector3D(0, 150_000, 0), forward=Vector3D(0, -1, 0))   # 90 deg to port
        a.weapons_orders = {"weapon_1": WeaponsOrder(WeaponsCommand.HOLD_FIRE, "weapon_1", "b",
                                                     min_hit_probability=0.0)}
        sim.step()                      # t=0: trained on b (the standing order)
        return sim, a, a.weapons["weapon_1"].mount

    @staticmethod
    def _aim(m):
        return (m.ax, m.ay, m.az)

    @staticmethod
    def _angle(p, q):
        return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(p, q))))))

    def test_a_mount_slews_at_most_once_per_tick(self):
        sim, a, m = self._scene()
        rate = math.degrees(m.rate)
        for _ in range(3):
            before = self._aim(m)
            sim._arcs_begin_tick()
            sim._handle_fire_command(a, {"weapon_slot": "weapon_1", "target_id": "c"})
            mid = self._aim(m)
            sim.step()
            moved = self._angle(before, mid) + self._angle(mid, self._aim(m))
            assert moved <= rate * sim.time_step + wa.AIM_TOLERANCE_DEG + 1e-6, moved

    def test_a_repeated_fire_at_swings_the_mount_over_and_fires(self):
        sim, a, m = self._scene()
        fired_at = None
        for _ in range(6):
            sim._arcs_begin_tick()
            if sim._handle_fire_command(a, {"weapon_slot": "weapon_1", "target_id": "c"}):
                fired_at = sim.current_time
                break
            sim.step()
        # 90 degrees at 30 deg/s: on in the third tick
        assert fired_at is not None and fired_at <= 4.0, fired_at


# =============================================================================
# Finding 2: no behavioural test pinned that a still-slewing PD turret holds
# its dwell (simulation._update_point_defense: dwell only when ON_TARGET)
# =============================================================================

class TestPdSlewGate:

    def test_a_pd_turret_dwells_only_once_it_has_slewed_onto_the_round(self):
        """Both turrets bear everywhere (all-open table) and are trained on a
        round to port; it vanishes and a second round appears ~170 degrees
        away, inside the envelope. 170 degrees at 120 deg/s is two ticks: on
        the first the turrets are assigned but still swinging and must put
        no energy on it; on the second they are on and dwell."""
        sim = _sim(_table(**ALL_OPEN))
        _ship(sim, "d", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "e", "beta", Vector3D(0, 900_000, 0), forward=Vector3D(0, -1, 0))
        first = _torpedo(sim, "t1", Vector3D(0, 200_000, 0), Vector3D(0, -4000.0, 0), "d", "e")
        for _ in range(3):
            sim.step()
        pd = sim.ships["d"].arcs.pd
        assert all(m.state == wa.ON_TARGET for m in pd)
        sim.torpedoes.remove(first)
        a = math.radians(-100.0)                  # ~170 degrees from the first round
        pos = Vector3D(200_000 * math.cos(a), 200_000 * math.sin(a), 0)
        second = _torpedo(sim, "t2", pos, pos.normalized() * -4000.0, "d", "e")
        sim.step()
        assert {m.state for m in pd} == {wa.SLEWING}
        assert second.heat_absorbed_j == 0.0
        sim.step()
        assert all(m.state == wa.ON_TARGET for m in pd)
        assert second.heat_absorbed_j > 0.0


# =============================================================================
# Finding 5: a battery whose target its own hull masks never engaged an
# enemy it does bear on
# =============================================================================

class TestMaskedBatteryFallsBack:
    """Destroyer fixture table: the dorsal battery bears only above the deck.
    'low' (the primary) sits below the deck, 'high' above it, both in range."""

    def _scene(self, order_target=None, command=WeaponsCommand.FIRE_IMMEDIATE):
        sim = _sim(_table())
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "low", "beta", Vector3D(100_000, 0, -60_000), forward=Vector3D(-1, 0, 0))
        _ship(sim, "high", "beta", Vector3D(100_000, 20_000, 60_000), forward=Vector3D(-1, 0, 0))
        a.primary_target_id = "low"
        a.weapons_orders = {"weapon_1": WeaponsOrder(command, "weapon_1", order_target,
                                                     min_hit_probability=0.0)}
        return sim, a

    @staticmethod
    def _shots(sim):
        from collections import Counter
        return Counter(e.target_id for e in sim.events
                       if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED
                       and e.ship_id == "a" and e.data.get("weapon_slot") == "weapon_1")

    def test_a_battery_following_the_primary_engages_what_it_bears_on(self):
        sim, a = self._scene()
        for _ in range(15):
            sim.step()
        shots = self._shots(sim)
        assert shots["high"] > 0 and shots["low"] == 0, shots
        assert a.weapons["weapon_1"].mount.fallback_id == "high"

    def test_it_returns_to_the_primary_when_the_primary_unmasks(self):
        sim, a = self._scene()
        for _ in range(15):
            sim.step()
        sim.ships["low"].kinematic_state.position = Vector3D(100_000, -20_000, 60_000)
        before = self._shots(sim)
        for _ in range(25):
            sim.step()
        after = self._shots(sim)
        assert after["low"] > before["low"], (before, after)
        assert a.weapons["weapon_1"].mount.fallback_id is None

    def test_a_named_single_target_order_stays_binding(self):
        sim, a = self._scene(order_target="low")
        for _ in range(15):
            sim.step()
        assert sum(self._shots(sim).values()) == 0
        assert a.weapons["weapon_1"].mount.state == wa.MASKED

    def test_free_fire_with_a_named_target_may_fall_back(self):
        sim, a = self._scene(order_target="low", command=WeaponsCommand.FREE_FIRE)
        for _ in range(15):
            sim.step()
        assert self._shots(sim)["high"] > 0


# =============================================================================
# Finding 6: settle was a no-op in runner battles (t = 0 names no targets),
# and every mount's first designation - at t = 30 s or later - was a free
# instant swing through any angle
# =============================================================================

class TestNoFreeSnapInBattle:

    def test_a_first_designation_after_the_start_slews(self):
        """Settled at t = 0 with no gun target; the first target named during
        the battle lies 90 degrees from the stowed battery: it slews."""
        sim = _sim(_table(**ALL_OPEN))
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        sim.step()                               # t=0: nothing named
        _fire_at(a, "b")                         # named mid-battle (not at a checkpoint)
        sim.step()
        m = a.weapons["weapon_1"].mount
        assert m.state == wa.SLEWING            # stowed dorsal (el 90) -> the nose: 3 s at 30 deg/s
        states = []
        for _ in range(3):
            sim.step()
            states.append(m.state)
        assert states[-1] == wa.ON_TARGET

    def test_runner_battles_settle_at_the_first_checkpoint_that_names_targets(self, monkeypatch):
        """In a runner battle checkpoint 1 is at T+0 (pass-5 schedule) and names
        the targets before anything has fired: that is where the engagement
        settles (rolled and trained), once."""
        calls = []
        settle = CombatSimulation.settle_arcs

        def logged(self):
            calls.append(self.current_time)
            settle(self)

        monkeypatch.setattr(CombatSimulation, "settle_arcs", logged)
        _draft_battle(104, time_limit=90.0)
        assert calls == [0.0], calls


# =============================================================================
# Finding 7: bookkeeping - a reported mask was never closed when its ship
# left the fight, and such ships reported their last state (often 2,
# 'on target') for the rest of the battle
# =============================================================================

class TestMountsStandDownWhenTheShipIsOut:

    def _masked(self):
        sim = _sim(_table())
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "low", "beta", Vector3D(100_000, 0, -60_000), forward=Vector3D(-1, 0, 0))
        _fire_at(a, "low")                       # named target: no fallback
        for _ in range(6):
            sim.step()
        m = a.weapons["weapon_1"].mount
        assert m.masked_reported
        return sim, a, m

    @staticmethod
    def _exits(sim):
        return [e for e in sim.events if e.event_type == SimulationEventType.MOUNT_MASKED
                and e.data["masked"] is False]

    @pytest.mark.parametrize("how", ["is_destroyed", "is_dying", "is_surrendered"])
    def test_a_reported_mask_is_closed_and_the_mounts_go_idle(self, how):
        sim, a, m = self._masked()
        setattr(a, how, True)
        sim.step()
        exits = self._exits(sim)
        assert len(exits) == 1 and exits[0].data["slot"] == "weapon_1"
        assert exits[0].data["cause"] == how[3:]
        assert all(mount.state == wa.IDLE for mount in a.arcs.mounts)
        for _ in range(3):
            sim.step()
        assert len(self._exits(sim)) == 1          # once

    def test_a_crew_incapacitated_ship_idles_its_guns_not_its_pd(self):
        sim = _sim(_table(**ALL_OPEN))
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        _fire_at(a, "b")
        sim.step()
        assert a.weapons["weapon_1"].mount.state == wa.ON_TARGET
        a.crew_incapacitated = True
        _torpedo(sim, "t", Vector3D(0, 200_000, 0), Vector3D(0, -4000.0, 0), "a", "b")
        for _ in range(3):
            sim.step()
        assert a.weapons["weapon_1"].mount.state == wa.IDLE
        assert any(m.state == wa.ON_TARGET for m in a.arcs.pd)


# =============================================================================
# Finding 8: arcs-on counterparts of the head-on scenarios that pin numbers
# commanders are given (the originals keep all-round fire)
# =============================================================================

def _bearing_on(sim, ship, point):
    """Operational PD turrets of ``ship`` whose own-hull mask leaves the line
    of sight to ``point`` open - computed here from the mask table, not
    through src/llm/arcs_status."""
    arcs = sim._arcs_for(ship)
    arcs.set_frame(ship.forward, ship.kinematic_state.up, sim.current_time)
    d = (point - ship.position).normalized()
    cell = wa.cell_index(*arcs.to_body(d.x, d.y, d.z))
    return sum(1 for pd in ship.point_defense
               if pd.is_operational and (pd.mount is None or not pd.mount.mask[cell]))


@pytest.fixture
def fleet():
    import json
    with open("data/fleet_ships.json") as f:
        return json.load(f)


class TestCommanderNumbersUnderArcs:

    def test_salvo_sizing_counts_the_turrets_that_bear(self, fleet):
        """Counterpart of test_captain_fire_solutions'
        test_salvo_sizing_uses_the_bound_pd_numbers (all-round fire: 2 + 3)."""
        from src.llm.fire_solutions import salvo_to_saturate, torpedo_solution
        from src.llm.prompts import pd_doctrine_numbers
        from tests.test_captain_fire_solutions import _duel, _ship as _fs_ship
        per = pd_doctrine_numbers(fleet)["turrets_to_blind_at_floor"]
        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        esc = _fs_ship(sim, fleet, "beta_2", "battleship", "beta", 0.0, y_km=100.0)
        _fs_ship(sim, fleet, "beta_3", "battleship", "beta", 0.0, y_km=900.0)
        sol = torpedo_solution(sim, sh, tg, fleet)
        own, escorts = _bearing_on(sim, tg, sh.position), _bearing_on(sim, esc, sh.position)
        assert (sol.pd_target, sol.pd_escorts) == (own, escorts)
        assert own + escorts < 2 + 3              # fewer than fitted bear on one direction
        assert sol.salvo_to_saturate == salvo_to_saturate(own + escorts, per)

    def test_threat_display_shows_the_own_turrets_that_bear(self, fleet):
        """Counterpart of test_doctrine_accuracy's
        test_live_sim_populates_enrichment (all-round fire: 2 of 2)."""
        from unittest.mock import Mock
        from src.llm.captain import LLMCaptain, LLMCaptainConfig
        from tests.test_doctrine_accuracy import _build
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=42)
        target = _build(sim, 'tg', 'destroyer', 'alpha', fleet, 0, 6.0, 1)
        _build(sim, 'sh', 'cruiser_torpedo', 'beta', fleet, 400, -6.0, -1)
        assert sim.inject_command('sh', {'type': 'launch_torpedo', 'target_id': 'tg'})
        for _ in range(20):
            sim.step()
        captain = LLMCaptain(LLMCaptainConfig(name="C", ship_name="tg", ship_type="destroyer",
                                              fleet_data=fleet), client=Mock())
        threats = captain._build_tactical_status(target, None, sim)["torpedo_threats"]
        assert threats
        want = _bearing_on(sim, target, sim.torpedoes[0].torpedo.position)
        assert threats[0]["own_pd_turrets"] == want
        assert captain._build_ship_status(target)["pd_turrets_total"] == 2   # fitted stays fitted

    def test_the_salvo_plan_sizes_against_the_turrets_that_bear(self):
        """Counterpart of test_heuristic_tactics'
        test_a_lone_corvette_still_fires_at_a_covered_group: against the
        turrets that bear, one corvette's tubes saturate the group."""
        from src.llm import heuristic_tactics as T
        from src.llm.fire_solutions import pd_cover, salvo_to_saturate
        from src.llm.prompts import pd_doctrine_numbers
        from tests.test_heuristic_tactics import FLEET, _salvo_sim
        sim = _salvo_sim(1, escorts=4, boat="corvette")
        plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
        pd = pd_doctrine_numbers(FLEET)
        boat, tgt = sim.get_ship("a0"), sim.get_ship("tgt")
        own, escorts, _ = pd_cover(sim, tgt, pd["range_km"], threat_from=boat.position)
        independent = sum(_bearing_on(sim, s, boat.position) for s in sim.ships.values()
                          if s.faction == "beta"
                          and (s.position - tgt.position).magnitude <= pd["range_km"] * 1000.0)
        assert own + escorts == independent
        need = salvo_to_saturate(own + escorts, pd["turrets_to_blind_at_floor"]) + T.SALVO_MARGIN_ROUNDS
        assert len(plan.salvos) == 1 and plan.salvos[0].need == need
        assert plan.salvos[0].saturating is (plan.salvos[0].available >= need)

    @pytest.mark.parametrize("hull", ["battleship", "dreadnought"])
    def test_three_fitted_turrets_are_not_three_that_bear(self, fleet, hull):
        """Counterpart of test_one_and_two_turret_hulls_do_not_blind_a_normal_launch
        (all-round fire: 3 turrets blind a 500 km launch). The doctrine counts
        turrets that BEAR: head-on, at most two of a battleship's three or a
        dreadnought's four bear on the round at the end, and it is not blinded."""
        from tests.test_doctrine_accuracy import _torpedo_run
        counts = []
        blinded, fitted = _torpedo_run(
            fleet, hull, 500.0, seed=11,
            sample=lambda sim, target, tf: counts.append(
                _bearing_on(sim, target, tf.torpedo.position)))
        assert fitted >= 3 and counts
        assert max(counts[-10:]) < 3
        assert blinded is False
        if hull == "battleship":
            # the figure the PD doctrine quotes
            from src.llm import prompts
            assert max(counts[-10:]) == 2 and fitted == 3
            assert "head-on, 2 of a battleship's 3" in " ".join(prompts.CAPTAIN_DOCTRINE.split())


# =============================================================================
# Finding 4: blind cones about the thrust axis that no roll can clear - the
# commander-facing ARCS lines must say so plainly
# =============================================================================

class TestBlindConesAreStatedPlainly:

    @staticmethod
    def _real_sim():
        return CombatSimulation(time_step=1.0, decision_interval=1e9, seed=5)

    def test_cones_are_real_no_mount_of_the_kind_bears_at_any_roll(self):
        from src.llm.arcs_status import blind_cones
        for hull in wa.load_hull_mounts().values():
            cones = blind_cones(hull)
            for kind in ("pd", "gun"):
                masks = [m.mask for m in hull.mounts if m.kind == kind]
                for end, sign in (("nose", 1.0), ("stern", -1.0)):
                    a = math.radians(cones[f"{kind}_{end}"])
                    if not masks or a <= 0.0:
                        continue
                    for k in range(360):           # every roll, 1 degree apart
                        p = math.radians(k)
                        cell = wa.cell_index(sign * math.cos(a), math.sin(a) * math.cos(p),
                                             math.sin(a) * math.sin(p))
                        assert all(mk[cell] for mk in masks), (hull.hull, kind, end, k)

    def test_a_target_dead_ahead_of_a_cruiser_says_turn_the_ship(self):
        from src.llm.arcs_status import blind_cones, gun_arcs_text
        sim = self._real_sim()
        if blind_cones(sim.hull_mounts["cruiser"])["gun_nose"] < 1.0:
            pytest.skip("the cruiser's batteries bear dead ahead on this mount table")
        a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0), hull="cruiser")
        _ship(sim, "b", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        _fire_at(a, "b", slots=("weapon_1", "weapon_2"))
        sim.step()
        text = gun_arcs_text(sim, a, sim.get_ship("b"))
        assert "weapon_1 masked: the target is 0 deg off the nose" in text
        assert "weapon_2 masked: the target is 0 deg off the nose" in text
        assert text.count("inside the hull's blind cone") == 2 and "turn the ship" in text

    def test_a_torpedo_dead_astern_of_a_dreadnought_says_no_roll_helps(self):
        from src.llm.arcs_status import arcs_status_lines, blind_cones
        sim = self._real_sim()
        if blind_cones(sim.hull_mounts["dreadnought"])["pd_stern"] < 1.0:
            pytest.skip("the dreadnought's PD bears dead astern on this mount table")
        d = _ship(sim, "d", "alpha", Vector3D(0, 0, 0), hull="dreadnought")
        _ship(sim, "e", "beta", Vector3D(-900_000, 0, 0), forward=Vector3D(1, 0, 0))
        _torpedo(sim, "t", Vector3D(-200_000, 0, 0), Vector3D(4000.0, 0, 0), "d", "e")
        sim.step()
        lines = arcs_status_lines(sim, d)
        pd = [l for l in lines if l.startswith("ARCS PD:")]
        assert pd and "0 of 4 turrets bear" in pd[0] and "no roll helps" in pd[0], lines
        assert any(l.startswith("ARCS blind cones") and "PD within" in l and "stern" in l
                   for l in lines), lines
