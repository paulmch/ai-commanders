"""
Torpedo retargeting audit (pass 5, lane retarget): behaviours of
CombatSimulation._retarget_torpedo that had no pinned test, plus the two
defects found (xfail, see the reasons).

Already pinned elsewhere and not repeated here: retarget onto a reachable
enemy and hit, blinded seeker / dry tank / unreachable candidate orphan, the
terminal reserve, the seeker-heat imperative, surrendered candidates
(tests/test_torpedo_retargeting.py); a dying hulk keeps the lock and the
round retargets only after the detonation (tests/test_dying_ships.py); a
retarget starts a fresh engagement (tests/test_fixes_audit_2026_09_26_core.py).

Every scenario injects a round directly (PD off everywhere) so the retarget
decision is the only thing under test.
"""

import contextlib
import io
import json
import math
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    SimulationEventType,
    TorpedoInFlight,
    _build_torpedo_specs,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, MIN_CLOSING_SPEED_KPS, Torpedo

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)

SPECS = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])


def _sim(seed=3):
    return CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)


def _ship(sim, ship_id, ship_type, faction, position, velocity=None):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET, position=position,
        velocity=velocity or Vector3D(0, 0, 0),
        forward=Vector3D(-1, 0, 0) if faction == "beta" else Vector3D(1, 0, 0))
    ship.point_defense = []
    sim.add_ship(ship)
    return ship


def _round(sim, target_id, dv_kps, *, source="a", position=None, velocity=None,
           torpedo_id="t1"):
    """A cruising round (12 km/s along +x unless told otherwise) holding
    exactly dv_kps, mass from the rocket equation."""
    torp = Torpedo(specs=SPECS, position=position or Vector3D(0, 0, 0),
                   velocity=velocity or Vector3D(12_000, 0, 0),
                   target_id=target_id, guidance_mode=GuidanceMode.COLLISION)
    torp.armed = True
    torp.current_mass_kg = SPECS.dry_mass_kg * math.exp(
        dv_kps / SPECS.exhaust_velocity_kps)
    torp.remaining_delta_v_kps = dv_kps
    flight = TorpedoInFlight(torpedo_id=torpedo_id, torpedo=torp,
                             source_ship_id=source, launch_time=sim.current_time)
    sim.torpedoes.append(flight)
    return flight


def _steps(sim, n):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            sim.step()
            if not sim.torpedoes:
                break


def _events(sim, kind):
    return [e for e in sim.events if e.event_type == kind]


def _retargets(sim):
    return [(e.data["old_target_id"], e.data["new_target_id"])
            for e in _events(sim, SimulationEventType.TORPEDO_RETARGETED)]


# ---------------------------------------------------------------------------
# Repeated retargets
# ---------------------------------------------------------------------------

class TestRepeatedRetarget:

    def test_round_retargets_again_when_its_new_target_dies_too(self):
        """There is no retarget limit: every tick a live, fuelled round is
        orphaned it looks again, so a second death swings it a second time."""
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        b2 = _ship(sim, "b2", "battlecruiser", "beta", Vector3D(300e3, 5e3, 0))
        _ship(sim, "b3", "battlecruiser", "beta", Vector3D(400e3, -40e3, 0))
        flight = _round(sim, "b1", 4.0)

        b1.is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]

        b2.is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2"), ("b2", "b3")]
        assert flight.torpedo.target_id == "b3"

    def test_second_retarget_pays_for_the_first(self):
        """The second look is gated on what the tank holds THEN: a round whose
        first chase burned it below the reserve orphans for good (this is
        what happened to torp_alpha_3_0013 / _0015 in the p5w5_b3 battle:
        4.20 km/s at the first retarget, 1.27 when the new target died)."""
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        b2 = _ship(sim, "b2", "battlecruiser", "beta", Vector3D(300e3, 5e3, 0))
        _ship(sim, "b3", "battlecruiser", "beta", Vector3D(400e3, -30e3, 0))
        flight = _round(sim, "b1", 4.0)
        b1.is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]

        torp = flight.torpedo
        reserve = max(sim.RETARGET_MIN_RESERVE_KPS,
                      SPECS.total_delta_v_kps * sim.RETARGET_RESERVE_FRACTION)
        b3 = sim.get_ship("b3")
        _, _, left_full = torp.can_intercept(b3.position, b3.velocity)
        steer = torp.remaining_delta_v_kps - left_full
        # Tank trimmed so b3 is reachable but only by eating the reserve.
        dv = steer + 0.5 * reserve
        torp.current_mass_kg = SPECS.dry_mass_kg * math.exp(dv / SPECS.exhaust_velocity_kps)
        torp.remaining_delta_v_kps = dv
        ok, _, left = torp.can_intercept(b3.position, b3.velocity)
        assert ok and 0.0 < left < reserve

        sim.get_ship("b2").is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]
        assert torp.target_id == "b2"


# ---------------------------------------------------------------------------
# Who may be acquired
# ---------------------------------------------------------------------------

class TestCandidateSet:

    def test_never_retargets_onto_its_own_side(self):
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        _ship(sim, "a2", "battlecruiser", "alpha", Vector3D(300e3, 5e3, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        _ship(sim, "b9", "battlecruiser", "beta", Vector3D(-3_000e3, 3_000e3, 0))
        flight = _round(sim, "b1", 4.0)
        # the friendly is the cheapest intercept by far; the only enemy left
        # is out of reach
        a2 = sim.get_ship("a2")
        assert flight.torpedo.can_intercept(a2.position, a2.velocity)[0]
        b1.is_destroyed = True
        _steps(sim, 3)
        assert _retargets(sim) == []
        assert flight.torpedo.target_id == "b1"

    def test_crew_incapacitated_enemy_is_still_acquired(self):
        """get_enemy_ships drops destroyed / dying / surrendered hulls only: a
        radiation-incapacitated crew's ship is still an enemy hull (a launch
        at it is accepted too), so an orphan may swing onto it."""
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        b2 = _ship(sim, "b2", "battlecruiser", "beta", Vector3D(300e3, 5e3, 0))
        b2.crew_incapacitated = True
        _round(sim, "b1", 4.0)
        b1.is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]

    def _surrender_scenario(self, b2_pos=(300e3, 5e3, 0)):
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        _ship(sim, "b2", "battlecruiser", "beta", Vector3D(*b2_pos))
        flight = _round(sim, "b1", 4.0)
        b1.is_surrendered = True
        return sim, flight

    def test_rounds_on_a_ship_that_surrenders_are_reassigned_like_a_hulk(self):
        """Owner decision (pass 5): a ship that strikes its colours releases
        the rounds inbound on it to the datalink exactly as a dying hulk."""
        sim, flight = self._surrender_scenario()
        _steps(sim, 1)
        ev = _events(sim, SimulationEventType.TORPEDO_RETARGETED)
        assert [(e.data["old_target_id"], e.data["new_target_id"], e.data["reason"])
                for e in ev] == [("b1", "b2", "target_surrendered")]
        assert ev[0].data["group"].endswith(":target_surrendered")

    def test_with_nothing_else_reachable_the_round_keeps_homing_on_it(self):
        """The hulk rule: rounds nobody can re-task still come at the
        surrendered ship (a surrender shields no fleet-mate and does not
        shield the ship from rounds that had nowhere else to go)."""
        sim, flight = self._surrender_scenario(b2_pos=(-3_000e3, 3_000e3, 0))
        _steps(sim, 30)
        assert _retargets(sim) == []
        assert [e.target_id for e in _events(sim, SimulationEventType.TORPEDO_IMPACT)] == ["b1"]

    def test_link_off_keeps_the_old_rule(self):
        """AIC_TORPEDO_DATALINK=0: the pre-link engine - inbound rounds keep
        homing on a ship that surrenders."""
        sim, flight = self._surrender_scenario()
        sim.torpedo_datalink = False
        _steps(sim, 30)
        assert _retargets(sim) == []
        assert flight.torpedo.target_id == "b1" or not sim.torpedoes
        hits = _events(sim, SimulationEventType.TORPEDO_IMPACT)
        assert [e.target_id for e in hits] == ["b1"]


# ---------------------------------------------------------------------------
# After the retarget
# ---------------------------------------------------------------------------

class TestAfterRetarget:

    def test_round_blinded_after_retargeting_coasts_and_never_retargets_again(self):
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        b2 = _ship(sim, "b2", "battlecruiser", "beta", Vector3D(300e3, 5e3, 0))
        _ship(sim, "b3", "battlecruiser", "beta", Vector3D(400e3, -40e3, 0))
        flight = _round(sim, "b1", 4.0)
        b1.is_destroyed = True
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]

        flight.is_disabled = True
        b2.is_destroyed = True
        v0 = Vector3D(flight.torpedo.velocity.x, flight.torpedo.velocity.y,
                      flight.torpedo.velocity.z)
        dv0 = flight.torpedo.remaining_delta_v_kps
        _steps(sim, 3)
        assert _retargets(sim) == [("b1", "b2")]
        assert flight.torpedo.target_id == "b2"
        assert flight.torpedo.velocity.x == v0.x and flight.torpedo.velocity.y == v0.y
        assert flight.torpedo.remaining_delta_v_kps == dv0

    def test_orphan_coasting_through_a_friendly_hull_strikes_it(self):
        """'A ballistic mass does not discriminate': an orphan with nothing to
        retarget onto hits whatever hull its trajectory intersects, own side
        included (the collision test skips only destroyed ships)."""
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 50e3, 0))
        _ship(sim, "a2", "battlecruiser", "alpha", Vector3D(60e3, 0, 0))
        flight = _round(sim, "b1", 0.0)
        flight.torpedo.fuel_exhausted = True
        b1.is_destroyed = True
        _steps(sim, 10)
        hits = _events(sim, SimulationEventType.TORPEDO_IMPACT)
        assert [e.target_id for e in hits] == ["a2"]

    def test_accepted_retarget_onto_a_passive_still_target_hits(self):
        """Control for the cruise-floor xfail below: the same steer against a
        target the round already closes on at ~12 km/s lands."""
        sim, flight, cand = _cruise_floor_scenario(Vector3D(0, 0, 0))
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]
        _steps(sim, 120)
        assert [e.target_id for e in _events(sim, SimulationEventType.TORPEDO_IMPACT)] == ["b2"]

    def test_accepted_retarget_onto_a_passive_receding_target_hits(self):
        """Was the audit's DEFECT (xfail): the pre-link rule accepted the
        swing because can_intercept said the lateral steer left >= the 1.12
        km/s reserve, but the COLLISION guidance first burned the round back
        up to its 12 km/s cruise floor along the line of sight (~1.6 km/s
        here) and arrived dry - an 8.7 km miss on a target that never moved.
        A datalink reassignment sets the cruise floor to the closure the
        round has plus its spare beyond the reserve (torpedo.cruise_floor_kps),
        so it never burns the reserve to regain speed."""
        sim, flight, cand = _cruise_floor_scenario(Vector3D(1_600, 0, 0))
        torp = flight.torpedo
        reserve = max(sim.RETARGET_MIN_RESERVE_KPS,
                      SPECS.total_delta_v_kps * sim.RETARGET_RESERVE_FRACTION)
        ok, _, left = torp.can_intercept(cand.position, cand.velocity)
        rel = cand.position - torp.position
        closure = -(cand.velocity - torp.velocity).dot(rel.normalized()) / 1000.0
        # Preconditions (hold today): the gate accepts, the round closes
        # below the cruise floor, and floor deficit + steer exceed the tank.
        assert ok and left >= reserve
        assert MIN_CLOSING_SPEED_KPS - closure > left
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]
        assert flight.torpedo.cruise_floor_override_kps < MIN_CLOSING_SPEED_KPS
        _steps(sim, 120)
        assert [e.target_id for e in _events(sim, SimulationEventType.TORPEDO_IMPACT)] == ["b2"]

    def test_pre_link_rule_still_misses_the_receding_target(self):
        """The measurement switch (torpedo_datalink = False) keeps the old
        per-round rule exactly - including its cruise-floor miss."""
        sim, flight, cand = _cruise_floor_scenario(Vector3D(1_600, 0, 0))
        sim.torpedo_datalink = False
        _steps(sim, 1)
        assert _retargets(sim) == [("b1", "b2")]
        assert flight.torpedo.cruise_floor_override_kps is None
        _steps(sim, 120)
        assert not _events(sim, SimulationEventType.TORPEDO_IMPACT)


def _cruise_floor_scenario(cand_velocity):
    """A 12 km/s round with 2.3 km/s left whose target just died; one other
    enemy (passive, no orders, PD off) ~250 km downrange and slightly abeam.
    Still: closure ~12 km/s. Receding at 1.6 km/s: closure ~10.4 km/s."""
    sim = _sim()
    _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
    b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, -40e3, 0))
    y = 25e3 if cand_velocity.x else 21.7e3
    cand = _ship(sim, "b2", "frigate", "beta", Vector3D(250e3, y, 0), cand_velocity)
    flight = _round(sim, "b1", 2.3)
    b1.is_destroyed = True
    return sim, flight, cand


# ---------------------------------------------------------------------------
# Choice: symmetry, order, determinism
# ---------------------------------------------------------------------------

def _tie_sim(order, faction="alpha", mirror=False):
    """A round on the axis between two mirror-image candidates (port and
    starboard of its track): identical can_intercept answers, bit for bit."""
    sx = -1.0 if mirror else 1.0
    enemy = "beta" if faction == "alpha" else "alpha"
    sim = _sim()
    _ship(sim, "src", "corvette", faction, Vector3D(sx * -2_000e3, 0, 0))
    dead = _ship(sim, "dead", "battlecruiser", enemy, Vector3D(sx * 150e3, 0, 0))
    pos = {"port": Vector3D(sx * 250e3, 20e3, 0), "stbd": Vector3D(sx * 250e3, -20e3, 0)}
    for name in order:
        _ship(sim, name, "frigate", enemy, pos[name])
    flight = _round(sim, "dead", 3.0, source="src",
                    velocity=Vector3D(sx * 12_000, 0, 0))
    dead.is_destroyed = True
    return sim, flight


class TestChoiceSymmetry:

    def test_mirrored_sides_make_mirrored_choices(self):
        """Alpha's orphan and beta's mirror-image orphan pick mirror-image
        targets (asymmetric candidates: no tie involved)."""
        picks = []
        for faction, mirror in (("alpha", False), ("beta", True)):
            sx = -1.0 if mirror else 1.0
            enemy = "beta" if faction == "alpha" else "alpha"
            sim = _sim()
            _ship(sim, "src", "corvette", faction, Vector3D(sx * -2_000e3, 0, 0))
            dead = _ship(sim, "dead", "battlecruiser", enemy, Vector3D(sx * 150e3, 0, 0))
            _ship(sim, "near", "frigate", enemy, Vector3D(sx * 250e3, 12e3, 0))
            _ship(sim, "far", "frigate", enemy, Vector3D(sx * 320e3, -30e3, 0))
            flight = _round(sim, "dead", 3.0, source="src",
                            velocity=Vector3D(sx * 12_000, 0, 0))
            dead.is_destroyed = True
            _steps(sim, 1)
            picks.append(flight.torpedo.target_id)
        assert picks[0] == picks[1] != "dead"

    def test_exact_tie_inputs_are_bit_identical(self):
        sim, flight = _tie_sim(["port", "stbd"])
        a = flight.torpedo.can_intercept(sim.get_ship("port").position, Vector3D(0, 0, 0))
        b = flight.torpedo.can_intercept(sim.get_ship("stbd").position, Vector3D(0, 0, 0))
        assert a == b

    def test_tie_break_does_not_depend_on_insertion_order(self):
        """Was an xfail: the pre-link rule kept the FIRST of equally scored
        candidates in self.ships insertion order. The datalink breaks ties
        by ship id."""
        picks = []
        for order in (["port", "stbd"], ["stbd", "port"]):
            sim, flight = _tie_sim(order)
            _steps(sim, 1)
            picks.append(flight.torpedo.target_id)
        assert picks[0] == picks[1] == "port"   # the smaller id

    def test_pre_link_rule_tie_follows_insertion_order(self):
        picks = []
        for order in (["port", "stbd"], ["stbd", "port"]):
            sim, flight = _tie_sim(order)
            sim.torpedo_datalink = False
            _steps(sim, 1)
            picks.append(flight.torpedo.target_id)
        assert picks == ["port", "stbd"]

    def test_retarget_is_deterministic_for_a_seed(self):
        def run():
            sim = _sim(seed=11)
            _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
            b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
            _ship(sim, "b2", "battlecruiser", "beta", Vector3D(300e3, 5e3, 0))
            _ship(sim, "b3", "frigate", "beta", Vector3D(260e3, -20e3, 0))
            _round(sim, "b1", 4.0)
            b1.is_destroyed = True
            _steps(sim, 60)
            return [(e.event_type.name, e.timestamp, e.target_id,
                     json.dumps(e.data, sort_keys=True, default=str))
                    for e in sim.events
                    if e.event_type in (SimulationEventType.TORPEDO_RETARGETED,
                                        SimulationEventType.TORPEDO_IMPACT,
                                        SimulationEventType.PROJECTILE_MISS)]
        first = run()
        assert first and first == run()
