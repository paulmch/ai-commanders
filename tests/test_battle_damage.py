"""
Battle-damage consequences (src/battle_damage.py, pass 5).

- SURFACE FITTINGS: any hull hit can degrade or knock out a gun mount or a PD
  turret; a TAIL hit can hit the drive. Degraded mounts fire slower.
- REACTOR BLAST: every ship death queues a detonation that scorches nearby
  ships, shreds extended radiators, shortens the fuse of dying hulks and
  kills / blinds torpedoes - never hull modules.
- MAGAZINE COOK-OFF: a destroyed magazine burns rounds and blows the
  neighbouring (non-critical) compartments.

The rolls come from CombatSimulation._damage_rng, never from the main battle
RNG that decides hit and miss.
"""

import json
from pathlib import Path

import pytest

from src import battle_damage as bd
from src.combat import HitLocation
from src.modules import ModuleType
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.thermal import RadiatorPosition
from src.torpedo import Torpedo, TorpedoSpecs

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)


class _Rolls:
    """Scripted stand-in for the damage RNG."""

    def __init__(self, *values):
        self.values = list(values)
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.values.pop(0) if self.values else 0.999


def _ship(sim, ship_id, faction, position, ship_type="cruiser", forward=None):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET,
        position=position, forward=forward or Vector3D(1, 0, 0))
    sim.add_ship(ship)
    return ship


def _events(sim, event_type):
    return [e for e in sim.events if e.event_type == event_type]


def _torpedo(sim, torpedo_id, position, target_id="victim"):
    torp = Torpedo(specs=TorpedoSpecs(), position=position,
                   velocity=Vector3D(0, 0, 0), target_id=target_id)
    torp.armed = True
    flight = TorpedoInFlight(torpedo_id=torpedo_id, torpedo=torp,
                             source_ship_id="shooter", launch_time=sim.current_time)
    sim.torpedoes.append(flight)
    return flight


# ---------------------------------------------------------------------------
# Surface fittings
# ---------------------------------------------------------------------------

def test_mount_cooldown_factor_curve():
    assert bd.mount_cooldown_factor(100.0) == 1.0
    assert bd.mount_cooldown_factor(80.0) == pytest.approx(1.25)
    assert bd.mount_cooldown_factor(50.0) == pytest.approx(2.0)
    # capped: a mount hanging on at 10% does not take 10x as long
    assert bd.mount_cooldown_factor(10.0) == bd.MOUNT_MAX_COOLDOWN_FACTOR
    assert bd.mount_cooldown_factor(0.0) == bd.MOUNT_MAX_COOLDOWN_FACTOR


def test_degraded_gun_mount_fires_slower_and_forecasts_fewer_shots():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
    gun = ship.weapons["weapon_1"]
    intact_shots = gun.shots_in_window(60.0)
    intact_cooldown = gun._cooldown_after_shot(0)

    gun.mount_health = 50.0
    assert gun._cooldown_after_shot(0) == pytest.approx(2.0 * intact_cooldown)
    assert gun.shots_in_window(60.0) < intact_shots
    assert gun.fire()
    assert gun.cooldown_remaining == pytest.approx(2.0 * intact_cooldown)


def test_surface_hit_degrades_then_knocks_out_a_gun_mount():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    # lateral hit: candidates are the two coilgun batteries, then the two PD
    # turrets (the spinal mount is out of reach) -> pick 0.0 = weapon_1
    assert [f[1] for f in bd._gun_fittings(ship, HitLocation.LATERAL)] == ["weapon_1", "weapon_2"]

    hit_gj = 60.0 / bd.MOUNT_DAMAGE_PCT_PER_GJ   # a hit worth 60% of a mount
    sim._damage_rng = _Rolls(0.0, 0.0)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, hit_gj, "shooter")
    gun = ship.weapons["weapon_1"]
    assert gun.mount_health == pytest.approx(40.0)
    assert gun.is_operational

    sim._damage_rng = _Rolls(0.0, 0.0)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, hit_gj, "shooter")
    assert gun.mount_health == 0.0
    assert not gun.is_operational
    assert not gun.can_fire()
    # every other fitting is untouched
    assert ship.weapons["weapon_0"].is_operational
    assert ship.weapons["weapon_2"].mount_health == 100.0
    assert all(pd.is_operational for pd in ship.point_defense)
    # and the hull is not: a surface hit never touches modules
    assert ship.hull_integrity == 100.0

    events = _events(sim, SimulationEventType.MOUNT_DAMAGED)
    assert [e.data["destroyed"] for e in events] == [False, True]
    first = events[0].data
    assert first["kind"] == "gun" and first["slot"] == "weapon_1" and first["index"] == 1
    assert first["health_pct"] == pytest.approx(40.0)
    assert first["cooldown_factor"] == pytest.approx(2.0)
    assert first["cause"] == "surface_hit" and first["hit_location"] == "lateral"
    assert first["source_ship_id"] == "shooter"
    assert events[1].data["cooldown_factor"] is None


def test_pd_turrets_are_lost_one_at_a_time():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="cruiser_torpedo")
    assert len(ship.point_defense) == 4
    assert bd._gun_fittings(ship, None) == []  # launchers are not gun mounts

    # fittings: 4 tubes then 4 turrets; pick 0.8 of 8 -> turret index 2
    sim._damage_rng = _Rolls(0.0, 0.8)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL,
                           100.0 / bd.PD_TURRET_DAMAGE_PCT_PER_GJ, "shooter")
    assert [pd.is_operational for pd in ship.point_defense] == [True, True, False, True]
    assert not ship.weapons["pd_2"].is_operational
    event = _events(sim, SimulationEventType.MOUNT_DAMAGED)[0].data
    assert event["kind"] == "pd" and event["index"] == 2 and event["destroyed"]
    assert event["slot"] == ship.point_defense[2].turret_name


def test_torpedo_tube_degrades_then_is_lost_with_its_rounds():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="cruiser_torpedo")
    launchers = ship.ready_torpedo_launchers
    intact_cooldown = launchers[1].cooldown_seconds
    assert [f[1] for f in bd._launcher_fittings(ship)] == [
        "weapon_0", "weapon_1", "weapon_2", "weapon_3"]

    # pick 0.2 of 8 fittings -> tube index 1; the hit takes half its health
    half_gj = 50.0 / bd.MOUNT_DAMAGE_PCT_PER_GJ
    sim._damage_rng = _Rolls(0.0, 0.2)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, half_gj, "shooter")
    assert launchers[1].cooldown_seconds == pytest.approx(2.0 * intact_cooldown)
    assert launchers[1].current_magazine == 12
    assert launchers[0].cooldown_seconds == intact_cooldown

    sim._damage_rng = _Rolls(0.0, 0.2)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, half_gj, "shooter")
    assert launchers[1].current_magazine == 0
    assert not launchers[1].can_launch(sim.current_time + 1e6)
    assert ship.torpedoes_remaining == 36
    assert not ship.weapons["weapon_1"].is_operational
    events = _events(sim, SimulationEventType.MOUNT_DAMAGED)
    assert [e.data["kind"] for e in events] == ["launcher", "launcher"]
    assert events[0].data["cooldown_factor"] == pytest.approx(2.0)
    assert events[1].data["destroyed"] and events[1].data["torpedoes_lost"] == 12
    assert events[1].data["torpedoes_remaining"] == 36
    assert events[1].data["index"] == 1 and events[1].data["slot"] == "weapon_1"


def test_spinal_mount_is_only_reachable_through_the_bow():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    assert "weapon_0" in [f[1] for f in bd._gun_fittings(ship, HitLocation.NOSE)]
    assert "weapon_0" not in [f[1] for f in bd._gun_fittings(ship, HitLocation.LATERAL)]
    assert "weapon_0" not in [f[1] for f in bd._gun_fittings(ship, HitLocation.TAIL)]


def test_surface_roll_misses_above_the_chance_and_always_draws_twice():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    chance = bd.SURFACE_FITTING_HIT_CHANCE[HitLocation.LATERAL]

    rolls = _Rolls(chance + 0.001, 0.0)
    sim._damage_rng = rolls
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, 5.0, "shooter")
    assert rolls.draws == 2
    assert not _events(sim, SimulationEventType.MOUNT_DAMAGED)

    rolls = _Rolls(chance - 0.001, 0.0)
    sim._damage_rng = rolls
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, 5.0, "shooter")
    assert rolls.draws == 2
    assert len(_events(sim, SimulationEventType.MOUNT_DAMAGED)) == 1


def test_surface_hits_never_draw_from_the_battle_rng_and_are_seed_deterministic():
    def run(seed):
        sim = CombatSimulation(seed=seed, time_step=1.0)
        ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="battleship")
        state = sim.rng.getstate()
        for i in range(60):
            location = (HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL)[i % 3]
            bd.resolve_surface_hit(sim, ship, location, 1.0, "shooter")
        assert sim.rng.getstate() == state
        return [(e.data["kind"], e.data["slot"], e.data["health_pct"])
                for e in _events(sim, SimulationEventType.MOUNT_DAMAGED)]

    first = run(7)
    assert first, "60 hits at 15-55% must strike at least one fitting"
    assert run(7) == first
    assert run(8) != first


def test_tail_hit_can_cripple_the_drive_without_killing_the_ship():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    assert ship.engine_effectiveness == 1.0

    # a slug up the tail: roll under ENGINE_HIT_CHANCE_TAIL hits the drive
    sim._damage_rng = _Rolls(bd.ENGINE_HIT_CHANCE_TAIL - 0.001, 0.0)
    bd.resolve_surface_hit(sim, ship, HitLocation.TAIL, 2.5, "shooter")
    expected = 1.0 - 2.5 * bd.ENGINE_DAMAGE_PCT_PER_GJ / 100.0
    assert ship.engine_effectiveness == pytest.approx(expected)
    assert ship.get_effective_thrust_fraction() == pytest.approx(expected)
    event = _events(sim, SimulationEventType.MOUNT_DAMAGED)[-1].data
    assert event["kind"] == "engine" and not event["destroyed"]
    assert event["thrust_fraction"] == pytest.approx(expected, abs=1e-3)

    # a torpedo up the tail: adrift, but not dead and not dying
    sim._damage_rng = _Rolls(0.0, 0.0)
    bd.resolve_surface_hit(sim, ship, HitLocation.TAIL, 25.0, "shooter")
    assert ship.engine_effectiveness == 0.0
    sim._check_ship_destroyed(ship, "shooter")
    assert not ship.is_dying and not ship.is_destroyed
    destroyed = _events(sim, SimulationEventType.MODULE_DESTROYED)
    assert [e.data["module_name"] for e in destroyed] == ["Main Engine Assembly"]

    # the same roll on the flank cannot reach the drive (it hits a turret)
    other = _ship(sim, "other", "beta", Vector3D(1e9, 0, 0))
    sim._damage_rng = _Rolls(0.0, 0.0)
    bd.resolve_surface_hit(sim, other, HitLocation.LATERAL, 25.0, "shooter")
    assert other.engine_effectiveness == 1.0


def test_dying_hulk_has_no_fittings_left_to_lose():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    ship.is_dying = True
    sim._damage_rng = _Rolls(0.0, 0.0)
    bd.resolve_surface_hit(sim, ship, HitLocation.LATERAL, 5.0, "shooter")
    assert not _events(sim, SimulationEventType.MOUNT_DAMAGED)


# ---------------------------------------------------------------------------
# Reactor blast
# ---------------------------------------------------------------------------

def test_blast_radius_scales_with_hull_mass():
    assert bd.reactor_blast_radius_km(bd.REACTOR_BLAST_REFERENCE_MASS_TONS) == pytest.approx(
        bd.REACTOR_BLAST_RADIUS_KM)
    sim = CombatSimulation(seed=1, time_step=1.0)
    radii = {}
    for i, hull in enumerate(("corvette", "destroyer", "cruiser", "battleship", "dreadnought")):
        ship = _ship(sim, hull, "alpha", Vector3D(i * 1e9, 0, 0), ship_type=hull)
        radii[hull] = bd.reactor_blast_radius_km(bd._hull_mass_tons(ship))
    assert radii["corvette"] < radii["destroyer"] < radii["cruiser"] \
        < radii["battleship"] < radii["dreadnought"]
    assert radii["corvette"] == pytest.approx(10.6, abs=0.1)
    assert radii["cruiser"] == pytest.approx(15.0, abs=0.1)
    assert radii["dreadnought"] == pytest.approx(21.2, abs=0.1)


def test_blast_intensity_falls_to_zero_at_the_edge():
    assert bd.blast_intensity(0.0, 15.0) == 1.0
    assert bd.blast_intensity(5.0, 15.0) == pytest.approx((2 / 3) ** 2)
    assert bd.blast_intensity(15.0, 15.0) == 0.0
    assert bd.blast_intensity(40.0, 15.0) == 0.0


def _detonate(sim, ship):
    sim._destroy_ship(ship, "killer", cause="reactor_destroyed")


def test_death_queues_the_blast_and_the_tick_resolves_it():
    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    near = _ship(sim, "near", "beta", Vector3D(0, 5_000, 0))
    armor_before = near.armor.get_section(HitLocation.LATERAL).thickness_cm

    _detonate(sim, dead)
    # nothing is applied inside the death itself
    assert len(sim._pending_blasts) == 1
    assert near.armor.get_section(HitLocation.LATERAL).thickness_cm == armor_before
    assert not _events(sim, SimulationEventType.REACTOR_BLAST)
    bd.resolve_pending_blasts(sim)
    assert sim._pending_blasts == []
    assert len(_events(sim, SimulationEventType.REACTOR_BLAST)) == 1
    assert near.armor.get_section(HitLocation.LATERAL).thickness_cm < armor_before


def test_blast_scorches_the_facing_armour_and_heats_the_neighbour():
    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    # 5 km off the dead ship's beam, nose along +X: the blast comes in on the flank
    near = _ship(sim, "near", "beta", Vector3D(0, 5_000, 0))
    far = _ship(sim, "far", "beta", Vector3D(0, 40_000, 0))
    sim._damage_rng = _Rolls()  # no fitting is caught
    lateral = near.armor.get_section(HitLocation.LATERAL)
    armor_before = lateral.thickness_cm
    nose_before = near.armor.get_section(HitLocation.NOSE).thickness_cm
    heat_before = near.thermal_system.heatsink.current_heat_gj
    far_armor = far.armor.get_section(HitLocation.LATERAL).thickness_cm

    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    radius = bd.reactor_blast_radius_km(bd._hull_mass_tons(dead))
    intensity = bd.blast_intensity(5.0, radius)
    assert lateral.thickness_cm == pytest.approx(
        armor_before - intensity * bd.REACTOR_BLAST_MAX_ARMOR_ABLATION_CM)
    assert near.armor.get_section(HitLocation.NOSE).thickness_cm == nose_before
    assert near.thermal_system.heatsink.current_heat_gj == pytest.approx(
        heat_before + intensity * bd.REACTOR_BLAST_MAX_HEAT_GJ)
    # a blast never reaches inside the hull
    assert near.hull_integrity == 100.0
    assert not near.is_dying and not near.is_destroyed
    # out of range: untouched and unrecorded
    assert far.armor.get_section(HitLocation.LATERAL).thickness_cm == far_armor

    blast = _events(sim, SimulationEventType.REACTOR_BLAST)[0]
    assert blast.ship_id == "dead"
    assert blast.data["ships_hit"] == ["near"]
    assert blast.data["radius_km"] == pytest.approx(radius, abs=0.01)
    assert blast.data["position"] == [0.0, 0.0, 0.0]
    assert blast.data["cause"] == "reactor_destroyed"
    hits = _events(sim, SimulationEventType.BLAST_DAMAGE)
    assert [e.ship_id for e in hits] == ["near"]
    data = hits[0].data
    assert data["source_ship_id"] == "dead"
    assert data["distance_km"] == pytest.approx(5.0)
    assert data["intensity"] == pytest.approx(intensity, abs=1e-3)
    assert data["hit_location"] == "lateral"
    assert data["armor_ablation_cm"] == pytest.approx(
        intensity * bd.REACTOR_BLAST_MAX_ARMOR_ABLATION_CM, abs=0.01)
    assert data["heat_gj"] == pytest.approx(intensity * bd.REACTOR_BLAST_MAX_HEAT_GJ, abs=0.1)
    assert data["radiators"] == []  # panels were retracted


def test_blast_shreds_extended_radiators_and_exposed_fittings():
    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    near = _ship(sim, "near", "beta", Vector3D(0, 2_000, 0))
    near.thermal_system.radiators.extend_all()
    sim._damage_rng = _Rolls(0.0)  # the first fitting (spinal mount) is caught

    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    intensity = bd.blast_intensity(2.0, bd.reactor_blast_radius_km(bd._hull_mass_tons(dead)))
    panels = near.thermal_system.radiators.radiators
    expected_health = max(0.0, 100.0 - intensity * bd.REACTOR_BLAST_RADIATOR_DAMAGE_GJ * 20.0)
    assert all(p.health_percent == pytest.approx(expected_health) for p in panels.values())
    data = _events(sim, SimulationEventType.BLAST_DAMAGE)[0].data
    assert len(data["radiators"]) == 4
    assert {p["radiator_position"] for p in data["radiators"]} == {p.value for p in RadiatorPosition}
    assert all(p["capacity_lost_kw"] > 0 for p in data["radiators"])

    mounts = _events(sim, SimulationEventType.MOUNT_DAMAGED)
    assert len(mounts) == 1
    assert mounts[0].data["cause"] == "reactor_blast"
    assert mounts[0].data["slot"] == "weapon_0"
    assert mounts[0].data["health_lost_pct"] == pytest.approx(
        intensity * bd.REACTOR_BLAST_FITTING_DAMAGE_PCT, abs=0.1)


def test_blast_kills_close_torpedoes_and_blinds_the_next_ring():
    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    _ship(sim, "victim", "beta", Vector3D(1e9, 0, 0))
    radius = bd.reactor_blast_radius_km(bd._hull_mass_tons(dead))
    kill_km = radius * bd.REACTOR_BLAST_TORPEDO_KILL_FRACTION
    blind_km = radius * bd.REACTOR_BLAST_TORPEDO_BLIND_FRACTION
    inside = _torpedo(sim, "t_kill", Vector3D(0, (kill_km - 0.5) * 1000, 0))
    ring = _torpedo(sim, "t_blind", Vector3D(0, (blind_km - 0.5) * 1000, 0))
    outside = _torpedo(sim, "t_free", Vector3D(0, (blind_km + 0.5) * 1000, 0))

    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    assert inside not in sim.torpedoes
    assert ring in sim.torpedoes and ring.is_disabled and ring.seeker_kill_logged
    assert outside in sim.torpedoes and not outside.is_disabled
    assert sim.metrics.total_torpedo_blast_destroyed == 1
    assert sim.metrics.total_torpedo_blast_blinded == 1
    # PD gets no credit for the blast's work
    assert sim.metrics.total_torpedo_intercepted == 0
    assert sim.metrics.total_torpedo_seeker_killed == 0

    blast = _events(sim, SimulationEventType.REACTOR_BLAST)[0].data
    assert blast["torpedoes_destroyed"] == ["t_kill"]
    assert blast["torpedoes_blinded"] == ["t_blind"]
    assert blast["torpedo_kill_radius_km"] == pytest.approx(kill_km, abs=0.01)
    assert blast["torpedo_blind_radius_km"] == pytest.approx(blind_km, abs=0.01)
    gone = _events(sim, SimulationEventType.TORPEDO_INTERCEPTED)
    assert len(gone) == 1
    assert gone[0].ship_id == "shooter" and gone[0].target_id == "victim"
    assert gone[0].data["torpedo_id"] == "t_kill"
    assert gone[0].data["cause"] == "reactor_blast"
    assert gone[0].data["blast_ship_id"] == "dead"


def test_blast_shortens_the_fuse_of_a_nearby_dying_hulk():
    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    hulk = _ship(sim, "hulk", "beta", Vector3D(0, 3_000, 0))
    for module in hulk._get_modules_by_type("bridge"):
        module.health_percent = 0.0
    sim._check_ship_destroyed(hulk, "killer")
    assert hulk.is_dying
    hulk.dying_time_remaining = 25.0
    armor_before = hulk.armor.get_section(HitLocation.LATERAL).thickness_cm

    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    intensity = bd.blast_intensity(3.0, bd.reactor_blast_radius_km(bd._hull_mass_tons(dead)))
    assert hulk.dying_time_remaining == pytest.approx(
        25.0 - intensity * bd.REACTOR_BLAST_FUSE_CUT_S)
    assert hulk.armor.get_section(HitLocation.LATERAL).thickness_cm == armor_before
    data = _events(sim, SimulationEventType.BLAST_DAMAGE)[0].data
    assert data["fuse_cut_s"] == pytest.approx(intensity * bd.REACTOR_BLAST_FUSE_CUT_S, abs=0.1)


def test_chain_detonation_waits_for_the_next_tick():
    sim = CombatSimulation(seed=1, time_step=1.0)
    first = _ship(sim, "first", "alpha", Vector3D(0, 0, 0))
    hulk = _ship(sim, "hulk", "alpha", Vector3D(0, 500, 0))
    _ship(sim, "enemy", "beta", Vector3D(1e9, 0, 0))
    for module in hulk._get_modules_by_type("bridge"):
        module.health_percent = 0.0
    sim._check_ship_destroyed(hulk, "killer")
    hulk.dying_time_remaining = 10.0  # the blast (intensity ~0.9) cuts ~18 s

    _detonate(sim, first)
    sim.step()   # first's blast lands, the hulk's fuse hits zero
    assert len(_events(sim, SimulationEventType.REACTOR_BLAST)) == 1
    assert hulk.dying_time_remaining == 0.0 and not hulk.is_destroyed
    sim.step()   # the hulk detonates and its own blast resolves
    assert hulk.is_destroyed
    assert [e.ship_id for e in _events(sim, SimulationEventType.REACTOR_BLAST)] == ["first", "hulk"]


def test_last_tick_flushes_the_final_blast_into_the_event_log():
    sim = CombatSimulation(seed=1, time_step=1.0)
    last = _ship(sim, "last", "alpha", Vector3D(0, 0, 0))
    _ship(sim, "winner", "beta", Vector3D(0, 4_000, 0))
    sim._running = True
    for module in last._get_modules_by_type("bridge"):
        module.health_percent = 0.0
    sim._check_ship_destroyed(last, "winner")
    last.dying_time_remaining = 0.5

    sim.step()   # the fuse runs out, the blast resolves, the battle ends
    assert last.is_destroyed
    assert not sim._running
    assert sim._pending_blasts == []
    blast = _events(sim, SimulationEventType.REACTOR_BLAST)
    assert len(blast) == 1 and blast[0].data["ships_hit"] == ["winner"]


# ---------------------------------------------------------------------------
# Magazine cook-off
# ---------------------------------------------------------------------------

def test_magazine_cookoff_burns_torpedoes_and_spares_the_citadel():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="cruiser_torpedo")
    magazine = ship.module_layout.get_modules_by_type(ModuleType.MAGAZINE)[0]
    assert ship.torpedoes_remaining == 48
    magazine.health_percent = 0.0

    bd.resolve_magazine_cookoff(sim, ship, magazine, "shooter")

    assert ship.torpedoes_remaining == 24
    assert [l.current_magazine for l in ship.ready_torpedo_launchers] == [6, 6, 6, 6]
    event = _events(sim, SimulationEventType.MAGAZINE_COOKOFF)[0].data
    assert event["module_name"] == magazine.name
    assert event["torpedoes_lost"] == 24 and event["torpedoes_remaining"] == 24
    assert event["gun_rounds_lost"] == 0
    assert event["secondary_gj"] == bd.COOKOFF_MAX_GJ
    assert event["source_ship_id"] == "shooter"
    assert event["modules_damaged"], "the secondary must reach a neighbour"
    # bridge and reactor are never reached: a cook-off is not a kill
    assert all(m.health_percent == 100.0 for m in ship.module_layout.get_critical_modules())
    sim._check_ship_destroyed(ship, "shooter")
    assert not ship.is_dying and not ship.is_destroyed
    assert ship.hull_integrity < 100.0
    damaged = _events(sim, SimulationEventType.MODULE_DAMAGED)
    assert damaged and all(e.data["cause"] == "magazine_cookoff" for e in damaged)
    assert all("health_pct" in e.data for e in damaged)


def test_gun_magazine_cookoff_burns_ammunition():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    magazine = ship.module_layout.get_modules_by_type(ModuleType.MAGAZINE)[0]
    ammo_before = {slot: ws.ammo_remaining for slot, ws in ship.weapons.items()}
    magazine.health_percent = 0.0

    bd.resolve_magazine_cookoff(sim, ship, magazine, "shooter")

    for slot in ("weapon_0", "weapon_1", "weapon_2"):
        assert ship.weapons[slot].ammo_remaining == ammo_before[slot] // 2
    event = _events(sim, SimulationEventType.MAGAZINE_COOKOFF)[0].data
    assert event["torpedoes_lost"] == 0
    assert event["gun_rounds_lost"] == sum(
        ammo_before[s] - ship.weapons[s].ammo_remaining
        for s in ("weapon_0", "weapon_1", "weapon_2"))
    assert event["secondary_gj"] == bd.COOKOFF_GUN_MAGAZINE_GJ


def test_other_modules_do_not_cook_off():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    hull = ship.module_layout.get_modules_by_type(ModuleType.HULL)[0]
    hull.health_percent = 0.0
    bd.resolve_magazine_cookoff(sim, ship, hull, "shooter")
    assert not _events(sim, SimulationEventType.MAGAZINE_COOKOFF)


# ---------------------------------------------------------------------------
# Radiator damage reporting
# ---------------------------------------------------------------------------

def test_radiator_damaged_reports_rated_capacity_lost_even_when_retracted():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0))

    class _AlwaysHit:
        def choice(self, seq):
            return seq[0]

        def random(self):
            return 0.0

    sim.combat_resolver.radiator_resolver.rng = _AlwaysHit()
    sim._resolve_radiator_hit(ship, None, HitLocation.LATERAL, energy_gj=2.0)

    data = _events(sim, SimulationEventType.RADIATOR_DAMAGED)[0].data
    panel = ship.thermal_system.radiators.radiators[RadiatorPosition(data["radiator_position"])]
    assert data["dissipation_lost_kw"] == 0.0  # retracted: nothing was being rejected
    assert data["health_pct"] == pytest.approx(panel.health_percent, abs=0.1)
    assert data["capacity_lost_kw"] == pytest.approx(
        panel.max_dissipation_kw * (100.0 - panel.health_percent) / 100.0, abs=0.1)
    assert data["array_health_pct"] == pytest.approx(
        ship.thermal_system.radiators.health_percent, abs=0.1)


# ---------------------------------------------------------------------------
# What reaches the recording and the captains
# ---------------------------------------------------------------------------

def test_new_events_reach_the_recording_with_every_field():
    from src.llm.battle_recorder import BattleRecorder
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner
    from src.llm.captain import LLMCaptainConfig

    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0), ship_type="cruiser_torpedo")
    near = _ship(sim, "near", "beta", Vector3D(0, 2_000, 0))
    _torpedo(sim, "t_kill", Vector3D(0, 1_000, 0), target_id="near")
    config = LLMCaptainConfig(name="Test", ship_name="Test")
    recorder = BattleRecorder()
    recorder.start_recording(BattleConfig(), config, config, dead, near)
    runner = LLMBattleRunner(BattleConfig(), config, config, client=None)
    runner.recorder = recorder
    sim.add_event_callback(runner._handle_simulation_event)

    sim._damage_rng = _Rolls(0.0, 0.0, 0.0)
    bd.resolve_surface_hit(sim, near, HitLocation.LATERAL, 1.0, "dead")
    magazine = dead.module_layout.get_modules_by_type(ModuleType.MAGAZINE)[0]
    magazine.health_percent = 0.0
    bd.resolve_magazine_cookoff(sim, dead, magazine, "near")
    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    by_type = {}
    for event in recorder.events:
        by_type.setdefault(event.event_type, []).append(event)
    for name in ("mount_damaged", "magazine_cookoff", "blast_damage",
                 "reactor_blast", "torpedo_intercepted"):
        assert name in by_type, f"{name} missing from the recording"

    mount = by_type["mount_damaged"][0]
    assert mount.ship_id == "near"
    assert {"kind", "slot", "index", "weapon_name", "health_pct", "health_lost_pct",
            "destroyed", "cooldown_factor", "cause", "hit_location",
            "source_ship_id"} <= set(mount.data)
    blast = by_type["reactor_blast"][0]
    assert blast.ship_id == "dead"
    assert {"ship_type", "cause", "position", "radius_km", "torpedo_kill_radius_km",
            "torpedo_blind_radius_km", "yield_gj", "ships_hit",
            "torpedoes_destroyed", "torpedoes_blinded"} <= set(blast.data)
    assert blast.data["ships_hit"] == ["near"]
    assert blast.data["torpedoes_destroyed"] == ["t_kill"]
    gone = by_type["torpedo_intercepted"][0]
    assert gone.data["cause"] == "reactor_blast" and gone.data["blast_ship_id"] == "dead"
    assert gone.data["torpedo_id"] == "t_kill"
    cook = by_type["magazine_cookoff"][0]
    assert cook.data["torpedoes_lost"] == 24
    # the recording must stay JSON-serialisable
    json.dumps([e.data for e in recorder.events])


def test_launching_captain_sees_its_round_destroyed_by_the_blast():
    from src.llm.captain import LLMCaptain, LLMCaptainConfig

    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    _ship(sim, "shooter", "beta", Vector3D(500_000, 0, 0), ship_type="cruiser_torpedo")
    flight = _torpedo(sim, "t1", Vector3D(0, 1_000, 0), target_id="dead")
    sim._log_event(SimulationEventType.TORPEDO_LAUNCHED, "shooter", "dead",
                   {"torpedo_id": flight.torpedo_id})
    _detonate(sim, dead)
    bd.resolve_pending_blasts(sim)

    captain = LLMCaptain(LLMCaptainConfig(name="Test", ship_name="Test"), client=None)
    captain.ship_id = "shooter"
    outcomes = captain._launch_outcomes(sim)
    assert outcomes["t1"]["outcome"] == "destroyed by blast"
