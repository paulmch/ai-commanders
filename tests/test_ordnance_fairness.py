"""
Two fairness defects in torpedo resolution, found by replaying a real battle.

A Kimi-vs-Sonnet corvette duel ended with one ship destroyed even though both
captains issued identical commands at every checkpoint and the sim trace was
mirrored to the decimal: at T+146 each side had a round 11.54 km out with
0.74 km/s of delta-v left. One landed, the other silently did not.

Two separate causes, one test class each:

1. The hit roll came from the shared simulation RNG, so two rounds reaching
   closest approach on the same tick with identical geometry drew consecutive
   values from one stream. The runner asks alpha to decide before beta, so
   alpha's rounds were always earlier in self.torpedoes and always drew first.

2. A round whose target died was deleted outright, cancelling ordnance that
   was already seconds from impact.
"""

import json

import pytest

from src.physics import Vector3D
from src.simulation import CombatSimulation, create_ship_from_fleet_data


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as fh:
        return json.load(fh)


def _mirrored_duel(fleet, seed):
    """Two identical corvettes, identical geometry, simultaneous launch."""
    # Combat resolution partly draws from the module-level `random` (see
    # LLMBattleRunner._seed_rng), so seed it too or the duel's outcome
    # depends on which tests happened to run earlier in the session.
    import random
    random.seed(seed)
    sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=seed)
    alpha = create_ship_from_fleet_data(
        "alpha", "corvette", "alpha", fleet,
        position=Vector3D(-200_000, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(1, 0, 0),
    )
    beta = create_ship_from_fleet_data(
        "beta", "corvette", "beta", fleet,
        position=Vector3D(200_000, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(-1, 0, 0),
    )
    sim.add_ship(alpha)
    sim.add_ship(beta)
    sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})
    sim.inject_command("beta", {"type": "launch_torpedo", "target_id": "alpha"})

    for _ in range(2000):
        sim.step()
        if not sim.torpedoes:
            break
    return alpha, beta


class TestMirroredEngagementsResolveAlike:
    """Identical geometry must produce identical outcomes on both sides."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_mirrored_duel_is_symmetric(self, fleet, seed):
        """
        Both rounds have the same miss distance and therefore the same
        hit_probability. Whether they land must not depend on which one the
        torpedo loop happens to reach first.
        """
        alpha, beta = _mirrored_duel(fleet, seed)

        assert alpha.damage_taken_gj == pytest.approx(beta.damage_taken_gj, abs=0.01), (
            f"seed {seed}: mirrored rounds resolved differently - "
            f"alpha took {alpha.damage_taken_gj:.2f} GJ, "
            f"beta took {beta.damage_taken_gj:.2f} GJ. The hit roll is "
            f"order-dependent again."
        )
        assert alpha.is_destroyed == beta.is_destroyed, (
            f"seed {seed}: a symmetric duel produced a single winner"
        )

    def test_roll_is_independent_of_evaluation_order(self, fleet):
        """
        The per-torpedo stream must key off the torpedo's identity, not its
        position in the list, so reversing the list cannot change any outcome.
        """
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=42)
        alpha = create_ship_from_fleet_data(
            "alpha", "corvette", "alpha", fleet,
            position=Vector3D(-200_000, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        beta = create_ship_from_fleet_data(
            "beta", "corvette", "beta", fleet,
            position=Vector3D(200_000, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        sim.add_ship(alpha)
        sim.add_ship(beta)
        sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})
        sim.inject_command("beta", {"type": "launch_torpedo", "target_id": "alpha"})
        for _ in range(40):
            sim.step()
            if len(sim.torpedoes) >= 2:
                break

        assert len(sim.torpedoes) >= 2, "expected both rounds in flight"
        first, second = sim.torpedoes[0], sim.torpedoes[1]

        def fresh_roll(sim_obj, torp):
            # The stream is cached per round; drop it so we re-derive from
            # (seed, torpedo_id) rather than continuing an advanced generator.
            if hasattr(torp, "_rng"):
                del torp._rng
            return sim_obj._torpedo_rng(torp).random()

        rolls_in_order = (fresh_roll(sim, first), fresh_roll(sim, second))
        # Same seed, drawn in the opposite order.
        reversed_rolls = (fresh_roll(sim, second), fresh_roll(sim, first))

        assert rolls_in_order[0] == reversed_rolls[1], (
            "a round's roll changed when it was drawn second - the stream is "
            "still order-dependent"
        )
        assert rolls_in_order[1] == reversed_rolls[0]

    def test_resolution_draws_from_the_per_torpedo_stream(self, fleet):
        """
        The load-bearing assertion for this fix.

        The mirrored-duel tests above are a useful regression guard but they
        do NOT prove this: in a perfectly symmetric encounter the miss
        distance is zero, hit_probability saturates at 0.98, and both rounds
        land whatever the roll says. Verified by mutation - reverting the fix
        leaves them all green. So check the mechanism directly: the hit roll
        must come from the round's own stream, not the shared one.
        """
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=9)
        alpha = create_ship_from_fleet_data(
            "alpha", "corvette", "alpha", fleet,
            position=Vector3D(-200_000, 0, 0), velocity=Vector3D(6000, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        beta = create_ship_from_fleet_data(
            "beta", "corvette", "beta", fleet,
            position=Vector3D(200_000, 0, 0), velocity=Vector3D(-6000, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        sim.add_ship(alpha)
        sim.add_ship(beta)
        sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})

        used = []
        original = sim._torpedo_rng

        def spy(torp_flight):
            used.append(torp_flight.torpedo_id)
            return original(torp_flight)

        sim._torpedo_rng = spy
        for _ in range(3000):
            sim.step()
            if not sim.torpedoes:
                break

        assert used, (
            "the torpedo hit roll never consulted the per-torpedo RNG - it is "
            "back on the shared simulation stream, so evaluation order can "
            "again decide which of two identical rounds lands"
        )

    def test_streams_are_seed_dependent(self, fleet):
        """Different simulation seeds must still give different rolls."""
        sims = [
            CombatSimulation(time_step=0.5, decision_interval=1e9, seed=s)
            for s in (1, 2)
        ]
        rolls = []
        for sim in sims:
            ship = create_ship_from_fleet_data(
                "alpha", "corvette", "alpha", fleet,
                position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(1, 0, 0),
            )
            target = create_ship_from_fleet_data(
                "beta", "corvette", "beta", fleet,
                position=Vector3D(400_000, 0, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(-1, 0, 0),
            )
            sim.add_ship(ship)
            sim.add_ship(target)
            sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})
            for _ in range(40):
                sim.step()
                if sim.torpedoes:
                    break
            rolls.append(sim._torpedo_rng(sim.torpedoes[0]).random())

        assert rolls[0] != rolls[1], "seeding no longer varies the hit roll"


class TestOrphanedOrdnanceKeepsFlying:
    """A round must not evaporate because the ship it was aimed at died."""

    def test_orphan_coasts_instead_of_vanishing(self, fleet):
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=3)
        alpha = create_ship_from_fleet_data(
            "alpha", "corvette", "alpha", fleet,
            position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        beta = create_ship_from_fleet_data(
            "beta", "corvette", "beta", fleet,
            position=Vector3D(600_000, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        sim.add_ship(alpha)
        sim.add_ship(beta)
        sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})

        for _ in range(40):
            sim.step()
            if sim.torpedoes:
                break
        assert sim.torpedoes, "no torpedo was launched"
        torp = sim.torpedoes[0]

        # Let it get up to speed before orphaning it.
        for _ in range(80):
            sim.step()
        speed_kps = torp.torpedo.velocity.magnitude / 1000
        assert speed_kps > 1.0, "torpedo never accelerated; test setup is wrong"

        beta.is_destroyed = True
        start = Vector3D(
            torp.torpedo.position.x, torp.torpedo.position.y, torp.torpedo.position.z
        )
        for _ in range(60):
            sim.step()

        still_tracked = any(t.torpedo_id == torp.torpedo_id for t in sim.torpedoes)
        coasted_km = (torp.torpedo.position - start).magnitude / 1000

        assert still_tracked, (
            "the round was deleted the moment its target died - ordnance "
            "already in flight must not be cancelled retroactively"
        )
        assert coasted_km > 10.0, (
            f"orphaned round only moved {coasted_km:.1f} km; it should coast "
            f"ballistically at roughly its {speed_kps:.1f} km/s flight speed"
        )

    def test_orphan_is_eventually_culled(self, fleet):
        """Coasting forever would leak objects; it must still be cleaned up."""
        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=5)
        alpha = create_ship_from_fleet_data(
            "alpha", "corvette", "alpha", fleet,
            position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        beta = create_ship_from_fleet_data(
            "beta", "corvette", "beta", fleet,
            position=Vector3D(600_000, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        sim.add_ship(alpha)
        sim.add_ship(beta)
        sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})
        for _ in range(40):
            sim.step()
            if sim.torpedoes:
                break

        # Let it reach cruise speed first, so the cull is exercised on a round
        # that is actually travelling rather than one still leaving the tube.
        for _ in range(80):
            sim.step()
        beta.is_destroyed = True

        # Generous cap: the range cull is 10,000 km and the timeout is 3600 s
        # of flight, so this must cover more than 3600 s at dt=0.5.
        for _ in range(9000):
            sim.step()
            if not sim.torpedoes:
                break

        assert not sim.torpedoes, (
            "orphaned round was never culled - it will be scanned by every "
            "ship's point defence on every tick for the rest of the battle"
        )
