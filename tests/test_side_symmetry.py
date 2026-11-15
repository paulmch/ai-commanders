"""
Side symmetry: a battle must not care which fleet is called alpha.

Background (pass 5): mirror battles - identical fleets, identical rule-based
captains, beta placed as alpha rotated 180 degrees about Z - were won the same
way seed after seed (one 7-ship torpedo roster: alpha 14, beta 1 in 18 seeds;
a dreadnought roster: beta 14, alpha 2 in 19). Swapping the order in which the
two fleets were added to the simulation, same seed, swapped the winner;
rotating the battlefield did not. The cause was CombatSimulation.step()
updating ships in place in dict order (alpha first), so every later ship
steered on the already-moved state and drive burn of the earlier ones: two
fleets ordered to HOLD_RANGE on each other did not fly mirrored courses (alpha
went to CLOSE at full throttle, beta saw alpha charging and sat in HOLD).
Ships now all update from one start-of-step snapshot (_update_ships).

These tests are exact, not statistical: in a mirrored engagement with no
random draw (or with every draw stubbed to the same value) the two sides must
stay mirror images of each other to float tolerance, and the result must not
depend on which fleet was inserted first.
"""

import json
import random
from pathlib import Path

import pytest

from src.firecontrol import WeaponsCommand, WeaponsOrder
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    create_ship_from_fleet_data,
)

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())

# The whole-battle guard: an explicit torpedo roster (not an auto-draft, whose
# picks move with the draft costs) whose first salvos land at ~100 s, run
# until the first kills are in.
WHOLE_BATTLE_ROSTER = ["cruiser_torpedo", "corvette", "corvette", "corvette",
                       "corvette", "frigate"]
WHOLE_BATTLE_SEED = 101
WHOLE_BATTLE_S = 150.0

# Mirrored positions are compared in metres at ranges of ~1e5 m; a mirror
# image differs from its original only by the sign of x and y, so anything
# beyond rounding noise is a real asymmetry.
TOL = 1e-6


def _mirror(v: Vector3D) -> Vector3D:
    """180 degrees about Z: how beta is placed relative to alpha."""
    return Vector3D(-v.x, -v.y, v.z)


def _other(ship_id: str) -> str:
    side, _, n = ship_id.partition("_")
    return ("beta_" if side == "alpha" else "alpha_") + n


def _build(roster, layout_km, first="alpha", seed=7):
    """
    Mirrored fleets: alpha_i of type roster[i] at layout_km[i] facing +X,
    beta_i at the mirrored point facing -X. ``first`` is the fleet added to
    the simulation first ('alpha' is what the battle runner does;
    'interleaved' alternates beta, alpha).
    """
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    sim.echo = False
    specs = []
    for i, (stype, pos_km) in enumerate(zip(roster, layout_km), start=1):
        pos = Vector3D(*(c * 1000.0 for c in pos_km))
        specs.append((f"alpha_{i}", "alpha", stype, pos, Vector3D(1, 0, 0)))
        specs.append((f"beta_{i}", "beta", stype, _mirror(pos), Vector3D(-1, 0, 0)))
    if first == "alpha":
        specs.sort(key=lambda s: s[1] != "alpha")
    elif first == "beta":
        specs.sort(key=lambda s: s[1] != "beta")
    elif first == "interleaved":
        specs = [s for pair in zip(specs[1::2], specs[0::2]) for s in pair]
    else:
        raise ValueError(first)
    for sid, faction, stype, pos, fwd in specs:
        ship = create_ship_from_fleet_data(
            sid, stype, faction, FLEET, position=pos,
            velocity=Vector3D(0, 0, 0), forward=fwd)
        ship.name = sid
        sim.add_ship(ship)
    return sim


def _order_both(sim, n, make):
    """Give alpha_n and beta_n the same order; ``make(enemy)`` builds it from
    a function mapping an enemy index to that side's enemy ship id."""
    for side, foe in (("alpha", "beta"), ("beta", "alpha")):
        command = make(lambda k, foe=foe: f"{foe}_{k}")
        assert sim.inject_command(f"{side}_{n}", command), (side, n, command)


def _close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _vec_close(a: Vector3D, b: Vector3D, tol=TOL):
    return _close(a.x, b.x, tol) and _close(a.y, b.y, tol) and _close(a.z, b.z, tol)


def _ship_scalars(ship):
    thermal = ship.thermal_system.heat_fraction if ship.thermal_system else 0.0
    row = {
        "hull": ship.hull_integrity,
        "propellant": ship.kinematic_state.propellant_kg,
        "throttle": getattr(ship, "applied_throttle", 0.0),
        "heat": thermal,
        "destroyed": float(ship.is_destroyed),
        "dying": float(ship.is_dying),
        "damage_taken": ship.damage_taken_gj,
        "damage_dealt": ship.damage_dealt_gj,
        "shots": float(ship.shots_fired),
        "hits": float(ship.hits_scored),
    }
    if ship.armor:
        for location, section in ship.armor.sections.items():
            row[f"armor_{location.value}"] = section.thickness_cm
    for slot, weapon in ship.weapons.items():
        row[f"{slot}_mount"] = weapon.mount_health
        row[f"{slot}_ammo"] = float(weapon.ammo_remaining)
        row[f"{slot}_cooldown"] = weapon.cooldown_remaining
    return row


def _mirror_breaks(sim):
    """Every way the two sides differ from being mirror images right now."""
    out = []
    for sid, a in sim.ships.items():
        if a.faction != "alpha":
            continue
        b = sim.ships[_other(sid)]
        for name in ("position", "velocity", "forward"):
            va, vb = getattr(a.kinematic_state, name), _mirror(getattr(b.kinematic_state, name))
            if not _vec_close(va, vb):
                out.append(f"{sid} {name}: {va} vs mirrored {_other(sid)} {vb}")
        sa, sb = _ship_scalars(a), _ship_scalars(b)
        for key in sa:
            if not _close(sa[key], sb[key]):
                out.append(f"{sid} {key}: {sa[key]} vs {_other(sid)} {sb[key]}")
        # (a hulk that was dying before its first update has no mode yet)
        mode_a, mode_b = getattr(a, "execution_mode", None), getattr(b, "execution_mode", None)
        if mode_a != mode_b:
            out.append(f"{sid} mode {mode_a} vs {_other(sid)} {mode_b}")

    def torpedoes(side):
        rows = [f for f in sim.torpedoes if f.source_ship_id.startswith(side)]
        return sorted(rows, key=lambda f: (f.source_ship_id.partition("_")[2],
                                           f.launch_time, f.torpedo_id[-4:]))
    ta, tb = torpedoes("alpha"), torpedoes("beta")
    if len(ta) != len(tb):
        out.append(f"torpedoes in flight: alpha {len(ta)} vs beta {len(tb)}")
    for fa, fb in zip(ta, tb):
        if not (_vec_close(fa.torpedo.position, _mirror(fb.torpedo.position))
                and _vec_close(fa.torpedo.velocity, _mirror(fb.torpedo.velocity))
                and _close(fa.heat_absorbed_j, fb.heat_absorbed_j)
                and fa.is_disabled == fb.is_disabled
                and fa.torpedo.target_id == _other(fb.torpedo.target_id)):
            out.append(f"torpedo {fa.torpedo_id} vs {fb.torpedo_id}: "
                       f"{fa.torpedo.position} heat {fa.heat_absorbed_j} disabled {fa.is_disabled} / "
                       f"{_mirror(fb.torpedo.position)} heat {fb.heat_absorbed_j} disabled {fb.is_disabled}")

    def rounds(side):
        rows = [f for f in sim.projectiles if f.source_ship_id.startswith(side)]
        return sorted(rows, key=lambda f: (f.source_ship_id.partition("_")[2], f.launch_time,
                                           f.projectile.position.z, abs(f.projectile.position.y),
                                           abs(f.projectile.position.x)))
    pa, pb = rounds("alpha"), rounds("beta")
    if len(pa) != len(pb):
        out.append(f"rounds in flight: alpha {len(pa)} vs beta {len(pb)}")
    for fa, fb in zip(pa, pb):
        if not _vec_close(fa.projectile.position, _mirror(fb.projectile.position)):
            out.append(f"round {fa.projectile_id} vs {fb.projectile_id}: "
                       f"{fa.projectile.position} / {_mirror(fb.projectile.position)}")
    return out


def _snapshot(sim):
    """Order-independent state of the battle, keyed by ship id."""
    ships = {}
    for sid, s in sim.ships.items():
        ks = s.kinematic_state
        ships[sid] = (ks.position.x, ks.position.y, ks.position.z,
                      ks.velocity.x, ks.velocity.y, ks.velocity.z,
                      ks.forward.x, ks.forward.y, ks.forward.z,
                      *_ship_scalars(s).values(), getattr(s, "execution_mode", None))
    torpedoes = sorted(
        (f.torpedo_id, f.torpedo.position.x, f.torpedo.position.y, f.torpedo.position.z,
         f.heat_absorbed_j, f.is_disabled) for f in sim.torpedoes)
    return ships, torpedoes


def _event_counts(sim, kinds):
    """(tick, kind, side, ship number) -> count, with beta relabelled as
    alpha so that a mirrored battle gives two equal tables."""
    tables = {"alpha": {}, "beta": {}}
    for e in sim.events:
        if e.event_type not in kinds or not e.ship_id:
            continue
        side, _, n = e.ship_id.partition("_")
        key = (round(e.timestamp, 6), e.event_type.name, n)
        tables[side][key] = tables[side].get(key, 0) + 1
    return tables


class _SameRolls(random.Random):
    """Every draw returns the same value, so mirrored events get identical
    rolls no matter which side the engine happens to resolve first."""

    def random(self):
        return 0.31

    def getrandbits(self, k):
        return 0

    def gauss(self, mu=0.0, sigma=1.0):
        return mu

    def normalvariate(self, mu=0.0, sigma=1.0):
        return mu


# =============================================================================
# A. The regression: mutual HOLD_RANGE
# =============================================================================

def _hold_range(sim, n, target_n, range_km, t=None):
    _order_both(sim, n, lambda enemy: Maneuver(
        ManeuverType.HOLD_RANGE, start_time=sim.current_time if t is None else t,
        duration=30.0, throttle=1.0, target_id=enemy(target_n),
        hold_range_m=range_km * 1000.0))


class TestMutualRangeKeepingIsMirrored:

    @pytest.mark.parametrize("first", ["alpha", "beta"])
    def test_both_sides_fly_the_same_phase_from_the_first_tick(self, first):
        # 500 km apart, everybody ordered to hold 300 km: the 1.5 g cruisers
        # on the enemy's 3 g frigate, the frigates on the enemy cruiser. All
        # four must CLOSE. Updated in place, the cruiser of the fleet inserted
        # first went to CLOSE at full throttle (the enemy frigate's drive was
        # still cold when it looked); the other cruiser looked after that
        # frigate had lit its drive, could not plan a stop against a 3 g
        # charge and sat in HOLD at a token throttle.
        sim = _build(["cruiser", "frigate"], [(-250, 0, 0), (-256, 12, 0)], first=first)
        _hold_range(sim, 1, 2, 300.0)
        _hold_range(sim, 2, 1, 300.0)
        sim.step()
        for n in (1, 2):
            a, b = sim.ships[f"alpha_{n}"], sim.ships[f"beta_{n}"]
            assert a.execution_mode == b.execution_mode == "CLOSE"
            assert a.applied_throttle == pytest.approx(b.applied_throttle, abs=1e-12)
            assert a.applied_throttle > 0.9
        assert not _mirror_breaks(sim)

    def test_mirrored_fleets_stay_mirrored_and_arrive_on_the_range(self):
        sim = _build(["cruiser", "corvette"], [(-250, 0, 0), (-262, 14, 6)])
        ranges = []
        for k in range(420):
            if k % 30 == 0:
                _hold_range(sim, 1, 1, 300.0)
                _hold_range(sim, 2, 1, 300.0)
            sim.step()
            assert not _mirror_breaks(sim), (sim.current_time, _mirror_breaks(sim)[:3])
            ranges.append((sim.ships["alpha_1"].position
                           - sim.ships["beta_1"].position).magnitude / 1000.0)
        # Both closing at once must still end ON the range, not in a merge.
        assert min(ranges) > 270.0
        assert ranges[-1] == pytest.approx(300.0, abs=15.0)


# =============================================================================
# B. Insertion order must not matter, and mirrored orders fly mirrored
# =============================================================================

ROSTER = ["corvette", "destroyer", "battleship", "frigate"]
LAYOUT = [(-262, 20, 0), (-250, 6, 5), (-244, -8, 0), (-256, -22, -4)]


def _fleet_orders(sim):
    """One checkpoint of mixed range-control orders, the same on both sides."""
    now = sim.current_time
    _order_both(sim, 1, lambda enemy: Maneuver(
        ManeuverType.INTERCEPT, start_time=now, duration=30.0, throttle=1.0,
        target_id=enemy(2), standoff_m=120_000.0))
    _order_both(sim, 2, lambda enemy: Maneuver(
        ManeuverType.HOLD_RANGE, start_time=now, duration=30.0, throttle=1.0,
        target_id=enemy(3), hold_range_m=200_000.0))
    _order_both(sim, 3, lambda enemy: Maneuver(
        ManeuverType.INTERCEPT, start_time=now, duration=30.0, throttle=0.6,
        target_id=enemy(3)))
    # Escort: fall in alongside the own battleship.
    for side in ("alpha", "beta"):
        assert sim.inject_command(f"{side}_4", Maneuver(
            ManeuverType.HEADING, start_time=now, duration=30.0, throttle=1.0,
            heading_target_id=f"{side}_3"))


def _fly_fleet(first, ticks=150, check_mirror=False):
    sim = _build(ROSTER, LAYOUT, first=first)
    for k in range(ticks):
        if k % 30 == 0:
            _fleet_orders(sim)
        sim.step()
        if check_mirror:
            breaks = _mirror_breaks(sim)
            assert not breaks, (sim.current_time, breaks[:3])
    return sim


class TestManeuveringDoesNotDependOnWhoWasAddedFirst:

    def test_mirrored_orders_fly_mirrored_courses(self):
        sim = _fly_fleet("alpha", check_mirror=True)
        # ... and the fleets actually manoeuvred (not a vacuous pass).
        assert sim.ships["alpha_1"].velocity.magnitude > 500.0
        assert sim.ships["alpha_4"].velocity.magnitude > 1.0

    @pytest.mark.parametrize("first", ["beta", "interleaved"])
    def test_insertion_order_does_not_change_the_battle(self, first):
        reference = _snapshot(_fly_fleet("alpha"))
        ships, torpedoes = _snapshot(_fly_fleet(first))
        assert torpedoes == reference[1]
        for sid, row in reference[0].items():
            for got, want in zip(ships[sid], row):
                if isinstance(want, str):
                    assert got == want, sid
                else:
                    assert _close(got, want, 1e-9), (sid, got, want)


# =============================================================================
# C. Torpedoes and point defence
# =============================================================================

DAMAGE_EVENTS = {
    SimulationEventType.TORPEDO_IMPACT, SimulationEventType.PROJECTILE_IMPACT,
    SimulationEventType.DAMAGE_TAKEN, SimulationEventType.SHIP_DESTROYED,
}


def _torpedo_exchange(first):
    sim = _build(["cruiser_torpedo", "corvette", "frigate", "destroyer"],
                 [(-150, 0, 0), (-156, 9, 0), (-146, -7, 3), (-144, 8, -3)], first=first)
    for n, target in ((1, 3), (2, 4)):
        for _ in range(2):
            _order_both(sim, n, lambda enemy: {
                "type": "launch_torpedo", "target_id": enemy(target),
                "queue_if_reloading": True})
    return sim


class TestTorpedoesAndPointDefenceAreMirrored:

    def test_salvo_flight_and_pd_dwell_are_mirror_images(self):
        sim = _torpedo_exchange("alpha")
        pd_kinds = {SimulationEventType.PD_ENGAGED, SimulationEventType.PD_TORPEDO_DISABLED,
                    SimulationEventType.TORPEDO_LAUNCHED}
        for _ in range(150):
            sim.step()
            if any(e.event_type in DAMAGE_EVENTS for e in sim.events):
                break       # impact damage rolls are random: stop before them
            breaks = _mirror_breaks(sim)
            assert not breaks, (sim.current_time, breaks[:3])
        tables = _event_counts(sim, pd_kinds)
        assert tables["alpha"] == tables["beta"]
        kinds = {k[1] for k in tables["alpha"]}
        # Not vacuous: rounds flew and point defence dwelled on them.
        assert {"TORPEDO_LAUNCHED", "PD_ENGAGED"} <= kinds
        assert sim.current_time > 20.0

    def test_insertion_order_does_not_change_the_exchange(self):
        runs = []
        for first in ("alpha", "beta"):
            sim = _torpedo_exchange(first)
            for _ in range(40):
                sim.step()
            runs.append(_snapshot(sim))
        (ships_a, torps_a), (ships_b, torps_b) = runs
        assert len(torps_a) == len(torps_b) > 0
        for ra, rb in zip(torps_a, torps_b):
            assert ra[0] == rb[0] and ra[5] == rb[5]
            assert all(_close(x, y, 1e-9) for x, y in zip(ra[1:5], rb[1:5])), (ra, rb)
        for sid, row in ships_a.items():
            assert all(x == y if isinstance(x, str) else _close(x, y, 1e-9)
                       for x, y in zip(ships_b[sid], row)), sid


# =============================================================================
# D. Gunfire with identical rolls
# =============================================================================

class TestGunDuelWithIdenticalRollsIsMirrored:

    def test_mirrored_gun_lines_trade_mirrored_hits(self, monkeypatch):
        # Every random draw is stubbed to one value (the battle RNG, the
        # damage stream and the per-round / per-ship streams the engine
        # derives with random.Random(...)), so the only thing that could make
        # the sides differ is the engine itself: who fires first, whose
        # rounds are resolved first, whose ship is updated first.
        monkeypatch.setattr(random, "Random", _SameRolls)
        sim = _build(["destroyer", "cruiser"], [(-45, 6, 0), (-40, -6, 2)])
        assert isinstance(sim.rng, _SameRolls) and isinstance(sim._damage_rng, _SameRolls)
        for n, target in ((1, 2), (2, 1)):
            _order_both(sim, n, lambda enemy: {
                "type": "weapons_orders", "default_target_id": enemy(target),
                "orders": [WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE,
                                        weapon_slot=slot, target_id=enemy(target))
                           for slot in sim.ships[f"alpha_{n}"].weapons
                           if not slot.startswith("pd")]})
        for k in range(90):
            if k % 30 == 0:
                _order_both(sim, 1, lambda enemy: Maneuver(
                    ManeuverType.HOLD_RANGE, start_time=sim.current_time, duration=30.0,
                    throttle=1.0, target_id=enemy(2), hold_range_m=60_000.0))
                _order_both(sim, 2, lambda enemy: Maneuver(
                    ManeuverType.PADLOCK, start_time=sim.current_time, duration=30.0,
                    target_id=enemy(1)))
            sim.step()
            breaks = _mirror_breaks(sim)
            assert not breaks, (sim.current_time, breaks[:3])
        kinds = {SimulationEventType.PROJECTILE_LAUNCHED, SimulationEventType.PROJECTILE_IMPACT,
                 SimulationEventType.PROJECTILE_MISS, SimulationEventType.DAMAGE_TAKEN,
                 SimulationEventType.MODULE_DAMAGED}
        tables = _event_counts(sim, kinds)
        assert tables["alpha"] == tables["beta"]
        seen = {k[1] for k in tables["alpha"]}
        assert {"PROJECTILE_LAUNCHED", "PROJECTILE_IMPACT"} <= seen   # hits landed


# =============================================================================
# E. A whole draft battle: runner + rule-based captains + engine
# =============================================================================

class TestWholeMirrorBattleWithIdenticalRollsIsMirrored:

    def test_heuristic_draft_battle_stays_a_mirror_image(self, monkeypatch):
        # The battle the side-bias sweeps run (identical auto-drafted fleets,
        # identical rule-based captains), with every roll stubbed to one
        # value: checkpoint order, command application, torpedo salvos, PD,
        # kills, reactor blasts and retargeting must then treat both sides
        # alike, tick for tick.
        from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
        from src.llm.captain import LLMCaptainConfig
        from src.llm.fleet_config import BattleFleetConfig
        from src.llm.fleet_draft import (
            FleetDraft, default_formation_offsets, draft_to_fleet_definition,
            name_drafted_ships)
        from src.llm.heuristic_captain import HEURISTIC_MODEL
        from src.llm.prompts import CaptainPersonality

        fleet_data = load_fleet_data()
        drafts = {}
        for side in ("alpha", "beta"):
            draft = FleetDraft(faction=side, budget=0, auto=True)
            draft.ships = name_drafted_ships(WHOLE_BATTLE_ROSTER, side)
            for ship, offset in zip(draft.ships,
                                    default_formation_offsets(len(draft.ships))):
                ship.offset_km = offset
            drafts[side] = draft
        monkeypatch.setattr(random, "Random", _SameRolls)

        fleets = {side: draft_to_fleet_definition(
            drafts[side], 500.0, captain_model=HEURISTIC_MODEL, admiral_config=None)
            for side in ("alpha", "beta")}
        runner = LLMBattleRunner(
            config=BattleConfig(
                initial_distance_km=500.0, time_limit_s=WHOLE_BATTLE_S, verbose=False,
                personality_selection=False, record_battle=False, record_sim_trace=False,
                record_predictions=False, seed=WHOLE_BATTLE_SEED),
            alpha_config=LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                                          personality=CaptainPersonality.BALANCED),
            beta_config=LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                                         personality=CaptainPersonality.BALANCED),
            client=None,
            fleet_config=BattleFleetConfig(
                battle_name="mirror", alpha_fleet=fleets["alpha"], beta_fleet=fleets["beta"],
                time_limit_s=WHOLE_BATTLE_S, initial_distance_km=500.0,
                record_battle=False, record_sim_trace=False,
                personality_selection=False, use_notebooks=False))

        broken = []
        step = CombatSimulation.step

        def checked_step(sim):
            events = step(sim)
            sim.echo = False
            if sim is runner.simulation and not broken:
                breaks = _mirror_breaks(sim)
                if breaks:
                    broken.append((sim.current_time, breaks[:4]))
            return events

        monkeypatch.setattr(CombatSimulation, "step", checked_step)
        result = runner.run_battle(fleet_data)

        assert not broken, broken
        assert result.winner is None, result.reason
        sim = runner.simulation
        tables = _event_counts(sim, {
            SimulationEventType.TORPEDO_LAUNCHED, SimulationEventType.TORPEDO_IMPACT,
            SimulationEventType.SHIP_DESTROYED, SimulationEventType.PD_TORPEDO_DISABLED})
        assert tables["alpha"] == tables["beta"]
        seen = {k[1] for k in tables["alpha"]}
        # Not vacuous: salvos landed and ships died - on both sides alike.
        assert {"TORPEDO_LAUNCHED", "TORPEDO_IMPACT", "SHIP_DESTROYED"} <= seen, (
            "scenario went stale (no salvo landed or nobody died in "
            f"{WHOLE_BATTLE_S:.0f} s): lengthen WHOLE_BATTLE_S or change the "
            f"roster - this is not a symmetry failure. Saw {sorted(seen)}")


# =============================================================================
# F. Deaths and incapacitation inside the ship loop
# =============================================================================

def _chasers(first):
    """Frigates (1) chased by the enemy's corvettes (2), plain INTERCEPT."""
    sim = _build(["frigate", "corvette"], [(-120, 0, 0), (-126, 9, 0)], first=first, seed=11)
    _order_both(sim, 2, lambda enemy: Maneuver(
        ManeuverType.INTERCEPT, start_time=0.0, duration=30.0, throttle=1.0,
        target_id=enemy(1)))
    return sim


class TestDeathsInsideTheShipLoopAreSeenByAllAtOnce:

    @pytest.mark.parametrize("first", ["alpha", "beta"])
    def test_reactor_meltdown_mid_loop(self, first):
        # Both frigates start the step past the meltdown line and blow up
        # while being updated. Their chasers must both still steer on them
        # this step: is_destroyed used to flip at once, so the chaser updated
        # after its target's meltdown coasted (target gone) while its mirror
        # image, updated before its own target's meltdown, burned.
        sim = _chasers(first)
        for sid in ("alpha_1", "beta_1"):
            sink = sim.ships[sid].thermal_system.heatsink
            sink.current_heat_gj = 1.6 * sink.capacity_gj
        sim.step()
        assert sim.ships["alpha_1"].is_destroyed and sim.ships["beta_1"].is_destroyed
        melted = [e.ship_id for e in sim.events
                  if e.event_type == SimulationEventType.REACTOR_MELTDOWN]
        assert sorted(melted) == ["alpha_1", "beta_1"]
        a, b = sim.ships["alpha_2"], sim.ships["beta_2"]
        assert a.applied_throttle == pytest.approx(b.applied_throttle, abs=1e-12)
        assert a.applied_throttle > 0.0          # both saw a live target this step
        assert not _mirror_breaks(sim)
        sim.step()                               # ... and both coast from the next
        assert a.applied_throttle == b.applied_throttle == 0.0
        assert not _mirror_breaks(sim)

    @pytest.mark.parametrize("first", ["alpha", "beta"])
    def test_dying_hulk_detonating_mid_loop(self, first, monkeypatch):
        # Same for a hulk whose fuse runs out inside the loop (the tumble and
        # torch flicker are random: every roll stubbed to one value).
        monkeypatch.setattr(random, "Random", _SameRolls)
        sim = _chasers(first)
        for sid in ("alpha_1", "beta_1"):
            sim._start_death_spiral(sim.ships[sid], None)
            sim.ships[sid].dying_time_remaining = 1.5
        for _ in range(4):
            sim.step()
            assert not _mirror_breaks(sim), (sim.current_time, _mirror_breaks(sim)[:3])
        assert sim.ships["alpha_1"].is_destroyed and sim.ships["beta_1"].is_destroyed

    @pytest.mark.parametrize("first", ["beta", "interleaved"])
    def test_incapacitated_crews_do_not_make_insertion_order_matter(self, first):
        # Plume radiation: an incapacitated crew's ship drops its maneuver in
        # _update_ship and is skipped by the weapons loop. Ships keeping range
        # on it must fly the same whoever is updated first.
        def fly(order):
            sim = _build(["frigate", "corvette"], [(-250, 0, 0), (-256, 12, 0)], first=order)
            for k in range(60):
                if k % 30 == 0:
                    _hold_range(sim, 1, 2, 300.0)
                    _hold_range(sim, 2, 1, 300.0)
                if k == 5:
                    for sid in ("alpha_1", "beta_1"):
                        sim.ships[sid].crew_incapacitated = True
                sim.step()
                assert not _mirror_breaks(sim), (sim.current_time, _mirror_breaks(sim)[:3])
            assert sim.ships["alpha_1"].current_maneuver is None
            return _snapshot(sim)
        reference, got = fly("alpha"), fly(first)
        for sid, row in reference[0].items():
            assert all(x == y if isinstance(x, str) else _close(x, y, 1e-9)
                       for x, y in zip(got[0][sid], row)), sid
