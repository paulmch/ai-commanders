"""
Captain intel from the engine: threat counting, flown torpedo fire
solutions, and execution feedback (pass 4, captain workstream).

- CP-01: "Incoming threats: N" counted every blinded torpedo in the battle,
  including the captain's OWN side's wrecks drifting away 300 km off, next
  to "INBOUND ORDNANCE: none" - captains evaded phantom rounds.
- CP-04 / CP-12 / R5 / R11: torpedo hulls never saw a fire solution (time of
  flight, impact energy, reachability, PD cover, rounds already inbound) and
  launched 19 of 49 rounds at opening targets for 0 GJ.
- R1: no commander was told when a ship had been issued nothing at all.
"""

import json
import random

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
from src.llm.fire_solutions import (
    LOW_YIELD_GJ,
    OVERKILL_MARGIN_ROUNDS,
    fly_torpedo,
    rounds_inbound,
    salvo_to_saturate,
    torpedo_solution,
)
from src.llm.prompts import pd_doctrine_numbers
from src.llm.execution_feedback import execution_feedback, idle_checkpoints


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _ship(sim, fleet, sid, stype, faction, x_km, vx_kps=0.0, fwd=1.0, y_km=0.0):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, fleet,
        position=Vector3D(x_km * 1000.0, y_km * 1000.0, 0.0),
        velocity=Vector3D(vx_kps * 1000.0, 0.0, 0.0),
        forward=Vector3D(fwd, 0.0, 0.0))
    sim.add_ship(ship)
    return ship


def _wreck(sim, source, position, velocity, tid):
    torp = Torpedo(specs=TorpedoSpecs.trident(), position=position, velocity=velocity,
                   target_id="nobody", guidance_mode=GuidanceMode.COAST)
    torp.armed = True
    flight = TorpedoInFlight(torpedo_id=tid, torpedo=torp, source_ship_id=source,
                             is_disabled=True)
    sim.torpedoes.append(flight)
    return flight


# =============================================================================
# CP-01: threat count = threats on a collision course with THIS ship
# =============================================================================

class TestThreatCount:

    def test_own_sides_receding_wreck_is_not_a_threat(self, fleet):
        """An alpha torpedo blinded 300 km from alpha_1 and moving away -> 0."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        me = _ship(sim, fleet, "alpha_1", "corvette", "alpha", 0.0)
        _ship(sim, fleet, "beta_1", "destroyer", "beta", 600.0, fwd=-1.0)
        _wreck(sim, "alpha_1", Vector3D(300_000, 0, 0), Vector3D(9_000, 0, 0), "w1")

        status = sim._get_evasion_status(me)
        assert status["threat_count"] == 0
        assert status["blinded_wrecks"] == 0
        assert sim.blinded_wreck_threats(me) == []

    def test_wreck_aimed_straight_at_the_ship_counts_whoever_launched_it(self, fleet):
        """A dead seeker does not discriminate: an own-side wreck on a
        collision course is listed, with its closest approach."""
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        me = _ship(sim, fleet, "alpha_1", "corvette", "alpha", 0.0)
        _wreck(sim, "alpha_2", Vector3D(300_000, 0, 0), Vector3D(-9_000, 0, 0), "w1")

        status = sim._get_evasion_status(me)
        assert status["threat_count"] == 1
        assert status["blinded_wrecks"] == 1
        (w,) = sim.blinded_wreck_threats(me)
        assert w["tca_s"] == pytest.approx(300_000 / 9_000, rel=1e-6)
        assert w["miss_m"] == pytest.approx(0.0, abs=1.0)

    def test_wide_miss_and_late_arrival_are_not_threats(self, fleet):
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        me = _ship(sim, fleet, "alpha_1", "corvette", "alpha", 0.0)
        # passes 40 km wide
        _wreck(sim, "beta_1", Vector3D(300_000, 40_000, 0), Vector3D(-9_000, 0, 0), "wide")
        # on course but 1000 s out
        _wreck(sim, "beta_1", Vector3D(1_000_000, 0, 0), Vector3D(-1_000, 0, 0), "late")
        assert sim._get_evasion_status(me)["blinded_wrecks"] == 0

    def test_breakdown_counts_guided_and_gun_rounds(self, fleet):
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        me = _ship(sim, fleet, "alpha_1", "destroyer", "alpha", 0.0)
        shooter = _ship(sim, fleet, "beta_1", "cruiser_torpedo", "beta", 400.0, fwd=-1.0)
        assert sim.inject_command("beta_1", {"type": "launch_torpedo", "target_id": "alpha_1"})
        status = sim._get_evasion_status(me)
        assert status["guided_torpedoes"] == 1
        assert status["gun_rounds"] == 0
        assert status["threat_count"] == 1
        # The launcher's own torpedo is not a threat to the launcher.
        assert sim._get_evasion_status(shooter)["threat_count"] == 0


# =============================================================================
# Torpedo fire solutions flown on the engine
# =============================================================================

def _duel(fleet, range_km, closure_kps, target_type="destroyer", shooter_type="corvette"):
    sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=3)
    sh = _ship(sim, fleet, "alpha_1", shooter_type, "alpha", range_km, -closure_kps / 2, -1.0)
    tg = _ship(sim, fleet, "beta_1", target_type, "beta", 0.0, closure_kps / 2, 1.0)
    return sim, sh, tg


class TestTorpedoShadow:

    def test_shadow_flight_matches_the_real_engine(self, fleet):
        """
        The shadow IS the engine: the same launch flown in a real battle sim
        (point defense stripped, target holding course) lands at the same
        time and speed the solution quotes.
        """
        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        shot = fly_torpedo(sim, sh, tg, fleet, target_evades=False)
        assert shot is not None and shot.hits

        real, rsh, rtg = _duel(fleet, 400.0, 0.0)
        rtg.point_defense = []
        assert real._handle_torpedo_command(rsh, {"target_id": "beta_1"})
        impact = None
        while real.current_time < 200 and real.torpedoes and impact is None:
            for ev in real.step():
                if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                    impact = (real.current_time, ev.data["impact_speed_kps"],
                              ev.data["total_damage_gj"])
        assert impact is not None
        assert shot.tof_s == pytest.approx(impact[0], abs=1.0)
        assert shot.impact_kps == pytest.approx(impact[1], rel=0.05)
        assert shot.impact_gj == pytest.approx(impact[2], rel=0.1)

    def test_shadow_never_touches_the_battle(self, fleet):
        """Asking for a solution must not move ships, add events or draw RNG."""
        sim, sh, tg = _duel(fleet, 400.0, -4.0)
        state = sim.rng.getstate()
        before = (sh.position, tg.position, len(sim.events), len(sim.torpedoes),
                  sh.torpedoes_remaining)
        torpedo_solution(sim, sh, tg, fleet)
        assert sim.rng.getstate() == state
        assert (sh.position, tg.position, len(sim.events), len(sim.torpedoes),
                sh.torpedoes_remaining) == before

    def test_no_launcher_no_solution(self, fleet):
        sim, sh, tg = _duel(fleet, 400.0, 0.0, shooter_type="destroyer")
        assert fly_torpedo(sim, sh, tg, fleet, target_evades=True) is None
        assert torpedo_solution(sim, sh, tg, fleet) is None


class TestTorpedoVerdicts:

    def test_closing_launch_at_range_is_reachable(self, fleet):
        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        sol = torpedo_solution(sim, sh, tg, fleet)
        assert sol.verdict == "REACHABLE"
        shot = sol.best_shot
        # The doctrine's anchor: a zero-closure launch from 400 km ~23 GJ.
        assert 18.0 <= shot.impact_gj <= 30.0
        assert "REACHABLE" in sol.summary() and "TOF ~" in sol.summary()

    def test_knife_fight_launch_is_too_close(self, fleet):
        """Measured: zero closure from 15 km lands ~0.5 GJ - guidance has no room."""
        sim, sh, tg = _duel(fleet, 15.0, 0.0)
        sol = torpedo_solution(sim, sh, tg, fleet)
        assert sol.verdict == "TOO CLOSE"
        assert sol.best_shot.impact_gj < LOW_YIELD_GJ
        assert "open range first" in sol.summary()

    def test_receding_target_escapes_if_it_runs(self, fleet):
        """Doctrine: a receding ship that keeps running outlasts the round."""
        sim, sh, tg = _duel(fleet, 400.0, -6.0)
        sol = torpedo_solution(sim, sh, tg, fleet)
        assert sol.verdict in ("ESCAPABLE", "UNREACHABLE")
        assert not sol.vs_evading.hits
        assert sol.closing_kps == pytest.approx(-6.0, abs=0.01)
        assert "OPENING" in sol.summary() or "EVADE" in sol.summary()

    def test_overkill_counts_live_friendly_rounds_only(self, fleet):
        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        need = torpedo_solution(sim, sh, tg, fleet).salvo_to_saturate
        # Same-side live seekers homing on him (a blinded one does not count).
        for i in range(need + OVERKILL_MARGIN_ROUNDS):
            torp = Torpedo(specs=TorpedoSpecs.trident(), position=Vector3D(300_000, i * 10.0, 0),
                           velocity=Vector3D(-12_000, 0, 0), target_id="beta_1")
            sim.torpedoes.append(TorpedoInFlight(torpedo_id=f"t{i}", torpedo=torp,
                                                 source_ship_id="alpha_1"))
        sim.torpedoes[0].is_disabled = True
        assert len(rounds_inbound(sim, "alpha", "beta_1")) == need + OVERKILL_MARGIN_ROUNDS - 1
        assert torpedo_solution(sim, sh, tg, fleet).verdict == "REACHABLE"
        sim.torpedoes[0].is_disabled = False
        sol = torpedo_solution(sim, sh, tg, fleet)
        assert sol.verdict == "OVERKILL"
        assert sol.friendly_inbound == need + OVERKILL_MARGIN_ROUNDS

    def test_salvo_sizing_uses_the_bound_pd_numbers(self, fleet):
        """
        'salvo >= N together': every turrets_to_blind_at_floor turrets blind one
        round, so N = cover // that + 1 - the same closed form the PD doctrine
        quotes (3 turrets per round at the guidance floor).
        """
        per = pd_doctrine_numbers(fleet)["turrets_to_blind_at_floor"]
        assert per == 3
        assert salvo_to_saturate(0, per) == 1
        assert salvo_to_saturate(2, per) == 1
        assert salvo_to_saturate(4, per) == 2
        assert salvo_to_saturate(10, per) == 4

        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        _ship(sim, fleet, "beta_2", "battleship", "beta", 0.0, y_km=100.0)   # escort, 3 PD
        _ship(sim, fleet, "beta_3", "battleship", "beta", 0.0, y_km=900.0)   # too far to cover
        sol = torpedo_solution(sim, sh, tg, fleet)
        assert (sol.pd_target, sol.pd_escorts) == (2, 3)
        assert sol.salvo_to_saturate == salvo_to_saturate(5, per) == 2

    def test_solutions_are_memoised_per_decision(self, fleet):
        sim, sh, tg = _duel(fleet, 400.0, 0.0)
        cache = {}
        first = torpedo_solution(sim, sh, tg, fleet, cache=cache)
        assert torpedo_solution(sim, sh, tg, fleet, cache=cache) is first


# =============================================================================
# Execution feedback: sentences, and NO COMMANDS ISSUED (R1)
# =============================================================================

class TestExecutionFeedback:

    def test_idle_ship_is_flagged_with_its_speed(self, fleet):
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        _ship(sim, fleet, "alpha_1", "destroyer", "alpha", 0.0, vx_kps=1.5)
        _ship(sim, fleet, "beta_1", "destroyer", "beta", 900.0, fwd=-1.0)
        # T+0: a command; T+30 and T+60: nothing.
        sim.inject_command("alpha_1", Maneuver(ManeuverType.MAINTAIN, start_time=0.0, duration=30))
        for _ in range(90):
            sim.step()
        assert idle_checkpoints(sim.events, "alpha_1", sim.current_time, 30) == [60.0, 30.0]
        text = execution_feedback(sim, ["alpha_1"])
        assert "NO COMMANDS ISSUED for 2 consecutive checkpoints (T+30, T+60)" in text
        assert "coasting at 1.50 km/s" in text

        # A command in the last window clears it.
        sim.inject_command("alpha_1", Maneuver(ManeuverType.MAINTAIN, start_time=sim.current_time,
                                               duration=30))
        for _ in range(30):
            sim.step()
        assert "NO COMMANDS" not in execution_feedback(sim, ["alpha_1"])

    def test_gun_gate_is_a_sentence_not_json(self, fleet):
        from types import SimpleNamespace
        from src.simulation import SimulationEventType as T
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        _ship(sim, fleet, "alpha_1", "destroyer", "alpha", 0.0)
        sim.inject_command("alpha_1", Maneuver(ManeuverType.MAINTAIN, start_time=0.0, duration=30))
        sim._log_event(T.WEAPON_STATUS, "alpha_1", "beta_1",
                       {"weapon_slot": "weapon_0", "reason": "probability_gate",
                        "hit_probability": 0.18, "min_hit_probability": 0.3})
        text = execution_feedback(sim, ["alpha_1"])
        assert "weapon_0 (T+0): held: P(hit) 0.18 below your 0.30 gate" in text
        assert "{" not in text

    def test_admiral_view_prefixes_ship_names(self, fleet):
        sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=1)
        a = _ship(sim, fleet, "alpha_1", "destroyer", "alpha", 0.0)
        a.name = "TIS One"
        _ship(sim, fleet, "alpha_2", "destroyer", "alpha", 0.0, y_km=50.0)
        for _ in range(31):
            sim.step()
        text = execution_feedback(sim, ["alpha_1", "alpha_2"])
        assert "TIS One [alpha_1] NO COMMANDS ISSUED" in text
        assert "alpha_2 NO COMMANDS ISSUED" in text
