"""
Regression tests for engagement symmetry and combat bookkeeping.

These lock in fixes for defects where the *order* ships happened to be stored in,
rather than the tactical situation, decided combat outcomes.
"""

import json

import pytest

from src.combat import HitLocation
from src.firecontrol import WeaponsCommand, WeaponsOrder
from src.physics import Vector3D
from src.simulation import CombatSimulation, create_ship_from_fleet_data


@pytest.fixture(scope="module")
def fleet_data():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _crossing_engagement(fleet_data, shooter_first: bool):
    """
    Shooter at the origin, target 100 km downrange crossing laterally at 2 km/s.

    The only difference between the two runs is the order the ships are inserted
    into the simulation, which must not affect the outcome.
    """
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

    shooter = create_ship_from_fleet_data(
        "shooter", "destroyer", "alpha", fleet_data,
        position=Vector3D(0.0, 0.0, 0.0),
        velocity=Vector3D(0.0, 0.0, 0.0),
        forward=Vector3D(1.0, 0.0, 0.0),
    )
    target = create_ship_from_fleet_data(
        "target", "destroyer", "beta", fleet_data,
        position=Vector3D(100_000.0, 0.0, 0.0),
        velocity=Vector3D(0.0, 2_000.0, 0.0),   # 2 km/s pure crossing
        forward=Vector3D(-1.0, 0.0, 0.0),
    )

    if shooter_first:
        sim.add_ship(shooter)
        sim.add_ship(target)
    else:
        sim.add_ship(target)
        sim.add_ship(shooter)

    shooter.weapons_orders = {
        slot: WeaponsOrder(
            command=WeaponsCommand.FIRE_IMMEDIATE,
            weapon_slot=slot,
            target_id="target",
            min_hit_probability=0.0,
        )
        for slot in shooter.weapons
        if not slot.startswith("pd_")
    }

    for _ in range(30):
        sim.step()

    return sim


def _shot_stats(sim):
    """(shots_fired, hits_scored) for the shooter."""
    shooter = sim.ships["shooter"]
    return shooter.shots_fired, shooter.hits_scored


class TestEngagementSymmetry:
    """Insertion order must not decide who hits."""

    def test_crossing_shots_are_insertion_order_independent(self, fleet_data):
        """
        Regression: _process_weapons_orders used to run inside the per-ship update
        loop, so a shooter aimed at a T1 target if the target had already been
        updated and a T0 target otherwise, while projectile propagation always
        assumed T0. The residual aim error was exactly |v_target| * dt, which for a
        2 km/s crossing target at dt=1 is 2 km - far outside the hit tolerance.
        Result: identical engagements resolved differently depending only on dict
        insertion order.
        """
        first = _crossing_engagement(fleet_data, shooter_first=True)
        second = _crossing_engagement(fleet_data, shooter_first=False)

        hits_first = _shot_stats(first)
        hits_second = _shot_stats(second)

        assert hits_first == hits_second, (
            f"Engagement outcome depends on ship insertion order: "
            f"shooter-first scored {hits_first} hits, shooter-second scored {hits_second}. "
            f"Firing solutions must be computed from a single consistent snapshot."
        )

    def test_crossing_target_is_actually_hittable(self, fleet_data):
        """
        Guard against the symmetry test passing trivially because both orders now
        score zero hits.
        """
        sim = _crossing_engagement(fleet_data, shooter_first=True)
        shots, hits = _shot_stats(sim)
        assert shots > 0, "shooter never fired; scenario is not exercising fire control"
        assert hits > 0, (
            f"Shooter scored 0 hits from {shots} shots on a 100 km crossing target; "
            f"the symmetry assertion would pass vacuously."
        )


class TestArmorReadout:
    """Armor sections must be reachable the way callers actually query them."""

    def test_get_section_accepts_strings_and_enums(self, fleet_data):
        """
        Regression: ShipArmor.sections is keyed by HitLocation, but the LLM captain
        queried it with plain strings ("nose"/"lateral"/"tail"), so every lookup
        returned None and captains were shown a hardcoded 10/5/3 cm fallback that
        never changed as armor ablated.
        """
        ship = create_ship_from_fleet_data("s1", "destroyer", "alpha", fleet_data)

        for loc in HitLocation:
            by_enum = ship.armor.get_section(loc)
            by_string = ship.armor.get_section(loc.value)
            assert by_enum is not None, f"no armor section for {loc}"
            assert by_string is by_enum, f"string lookup failed for {loc.value!r}"

        # A real destroyer is far thicker than the old hardcoded fallback.
        nose = ship.armor.get_section(HitLocation.NOSE).current_thickness_cm
        assert nose > 20.0, (
            f"nose armor reads {nose} cm - suspiciously close to the removed "
            f"hardcoded 10 cm fallback"
        )


class TestArmorSemantics:
    """Armor must reduce, never increase, what reaches the hull."""

    def test_energy_to_hull_never_increases_with_armor(self, fleet_data):
        """
        Regression: Armor.apply_energy_damage() returns the energy that passes
        THROUGH to the hull, but simulation.py subtracted it from the incoming KE
        as though it were the energy absorbed. That inverted the armor model -
        thicker armor produced more internal damage.
        """
        import copy

        ship = create_ship_from_fleet_data("s1", "destroyer", "alpha", fleet_data)
        base = ship.armor.get_section(HitLocation.NOSE)

        results = []
        for thickness in [0.0, 10.0, 50.0, 100.0, 151.2, 300.0]:
            section = copy.deepcopy(base)
            section.thickness_cm = thickness
            section.chipping_fraction = 0.0
            _, energy_to_hull, _ = section.apply_energy_damage(
                energy_gj=4.3, flat_chipping=0.35, impact_area_m2=0.01
            )
            results.append((thickness, energy_to_hull))

        for (t_prev, e_prev), (t_next, e_next) in zip(results, results[1:]):
            assert e_next <= e_prev + 1e-9, (
                f"{t_next} cm of armor lets {e_next:.4f} GJ through vs "
                f"{e_prev:.4f} GJ at {t_prev} cm - armor is making damage worse"
            )

        # And thick armor must actually help.
        assert results[-1][1] < results[0][1], "300 cm of armor stopped nothing"


class TestGeometryFallbacks:
    """Every geometry fallback must actually construct."""

    def test_all_shipgeometry_constructions_use_real_fields(self):
        """
        Regression: three fallbacks across firecontrol.py and mcp_controller.py
        passed beam_m / height_m / nose_section_length, none of which exist on
        ShipGeometry. Each would raise TypeError the moment a geometry-less ship
        reached it - a crash on exactly the degraded path meant to prevent one.
        """
        import re
        from pathlib import Path

        from src.geometry import ShipGeometry

        valid = set(ShipGeometry.__dataclass_fields__)
        offenders = []
        for path in Path("src").rglob("*.py"):
            text = path.read_text()
            for match in re.finditer(r"ShipGeometry\(([^)]*)\)", text, re.S):
                body = match.group(1)
                if '"' in body or "'" in body or "{" in body:
                    continue  # a repr/format string, not a construction
                unknown = set(re.findall(r"(\w+)\s*=", body)) - valid
                if unknown:
                    offenders.append(f"{path}: {sorted(unknown)}")

        assert not offenders, "invalid ShipGeometry constructions:\n  " + "\n  ".join(offenders)


class TestConstantsHaveOneSourceOfTruth:
    """Duplicated tuning constants silently defeat rebalancing."""

    def test_torpedo_pd_thresholds_are_not_duplicated(self):
        """
        Regression: simulation.TorpedoInFlight carried its own copy of the PD kill
        thresholds at 10 kJ / 100 kJ while pointdefense.py held the real values.
        The battle loop reads the TorpedoInFlight copy, so rebalancing point
        defense had zero effect on an actual battle - PD still one-shot-killed
        every torpedo at any range, and torpedoes never reached a target.
        """
        from src.pointdefense import (
            TORPEDO_ELECTRONICS_THRESHOLD_J,
            TORPEDO_WARHEAD_THRESHOLD_J,
        )
        from src.simulation import TorpedoInFlight

        assert TorpedoInFlight.ELECTRONICS_THRESHOLD_J == TORPEDO_ELECTRONICS_THRESHOLD_J
        assert TorpedoInFlight.WARHEAD_THRESHOLD_J == TORPEDO_WARHEAD_THRESHOLD_J

    def test_a_torpedo_survives_long_range_point_defense(self):
        """
        A single PD burst at the edge of the envelope must not kill a multi-tonne
        torpedo - otherwise torpedoes can never cross the envelope at all.
        """
        import math

        from src.pointdefense import (
            PD_ABSORPTIVITY, PD_POINTING_JITTER_RAD,
            TORPEDO_ELECTRONICS_THRESHOLD_J,
        )

        power_w, dwell_s, range_km = 5e6, 5.0, 250.0
        spot_radius = PD_POINTING_JITTER_RAD * range_km * 1000
        spot_area = math.pi * spot_radius ** 2
        effective = power_w if spot_area <= 1.0 else power_w * (1.0 / spot_area)
        delivered = effective * dwell_s * PD_ABSORPTIVITY

        assert delivered < TORPEDO_ELECTRONICS_THRESHOLD_J, (
            f"one PD burst at {range_km:.0f} km delivers {delivered/1e6:.1f} MJ vs a "
            f"{TORPEDO_ELECTRONICS_THRESHOLD_J/1e6:.0f} MJ seeker threshold - point "
            f"defense is back to one-shot-killing torpedoes"
        )


class TestMultiLauncherHulls:
    """A hull mounting several launchers must fire at its full rate."""

    def test_torpedo_cruiser_builds_all_launchers(self, fleet_data):
        """
        Regression: ship construction `break`-ed after the first torpedo weapon
        entry, so a multi-launcher hull fired at the rate of a single launcher.
        """
        from src.simulation import create_ship_from_fleet_data

        ship = create_ship_from_fleet_data("tc", "cruiser_torpedo", "alpha", fleet_data)
        assert ship.torpedo_launcher_count == 4
        assert ship.torpedoes_remaining == 48
        # Legacy singular accessor must keep working for existing call sites.
        assert ship.torpedo_launcher is ship.torpedo_launchers[0]

    def test_single_launcher_hull_unchanged(self, fleet_data):
        from src.simulation import create_ship_from_fleet_data

        corvette = create_ship_from_fleet_data("c", "corvette", "beta", fleet_data)
        assert corvette.torpedo_launcher_count == 1
        assert corvette.torpedo_launcher is not None

    def test_gun_hull_has_no_launchers(self, fleet_data):
        from src.simulation import create_ship_from_fleet_data

        destroyer = create_ship_from_fleet_data("d", "destroyer", "beta", fleet_data)
        assert destroyer.torpedo_launcher_count == 0
        assert destroyer.torpedoes_remaining == 0

    def test_salvo_ceiling_scales_with_launcher_count(self, fleet_data):
        """4 launchers over a 30s decision at 12s reload = 8 rounds."""
        from unittest.mock import Mock

        from src.llm.captain import LLMCaptain, LLMCaptainConfig
        from src.simulation import create_ship_from_fleet_data

        for ship_type, expected in (("cruiser_torpedo", 8), ("corvette", 2)):
            ship = create_ship_from_fleet_data("s", ship_type, "alpha", fleet_data)
            captain = LLMCaptain(
                LLMCaptainConfig(name="C", ship_name="S", ship_type=ship_type,
                                 fleet_data=fleet_data),
                client=Mock(),
            )

            class Sim:
                decision_interval = 30.0

                def get_ship(self, _):
                    return ship

            assert captain._max_torpedo_salvo("s", Sim()) == expected, ship_type


class TestTraceRecordingWithTorpedoes:
    """Trace recording must not crash the battle it is recording."""

    def test_sim_frame_records_a_torpedo_in_flight(self, fleet_data):
        """
        Regression: _record_sim_frame read torp.delta_v_remaining_kps, but the
        attribute is remaining_delta_v_kps. With --trace enabled the entire
        battle died with AttributeError the instant the first torpedo launched -
        and only a torpedo-armed hull could ever trigger it.
        """
        from src.physics import Vector3D
        from src.torpedo import Torpedo, TorpedoSpecs

        torp = Torpedo(
            specs=TorpedoSpecs(),
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(1000, 0, 0),
            target_id="beta",
        )
        # Exactly the fields the recorder reads.
        assert isinstance(torp.remaining_delta_v_kps, float)
        assert torp.target_id == "beta"
        assert isinstance(torp.fuel_exhausted, bool)

        assert not hasattr(torp, "delta_v_remaining_kps"), (
            "attribute was renamed - update _record_sim_frame to match"
        )


class TestTorpedoTerminalBurn:
    """
    A torpedo that reaches its target on course used to coast in hoarding
    delta-v it would never spend. Terminal burn converts that fuel into impact
    energy (which scales as v^2) and shortens the run, giving point defense
    fewer engagement windows.
    """

    def _engagement(self, fleet_data, shooter_v, target_v):
        from src.physics import Vector3D
        from src.simulation import CombatSimulation, create_ship_from_fleet_data

        sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=3)
        shooter = create_ship_from_fleet_data(
            "alpha", "corvette", "alpha", fleet_data,
            position=Vector3D(0, 0, 0), velocity=Vector3D(shooter_v, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        target = create_ship_from_fleet_data(
            "beta", "corvette", "beta", fleet_data,
            position=Vector3D(300_000, 0, 0), velocity=Vector3D(target_v, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        sim.add_ship(shooter)
        sim.add_ship(target)
        assert sim.inject_command("alpha", {"type": "launch_torpedo", "target_id": "beta"})

        peak_closure, steps, dv_left = 0.0, 0, 14.0
        while sim.torpedoes and steps < 1200:
            sim.step()
            steps += 1
            if sim.torpedoes:
                torp = sim.torpedoes[0].torpedo
                peak_closure = max(peak_closure, (torp.velocity - target.velocity).magnitude / 1000)
                dv_left = torp.remaining_delta_v_kps
        return peak_closure, 14.0 - dv_left, steps * 0.5

    def test_torpedo_spends_fuel_instead_of_hoarding_it(self, fleet_data):
        """On-course torpedoes must burn, not coast in with a full tank."""
        for shooter_v, target_v in ((0, 0), (6000, -6000)):
            _, dv_used, _ = self._engagement(fleet_data, shooter_v, target_v)
            assert dv_used > 0.5, (
                f"torpedo arrived having spent only {dv_used:.2f} km/s - it is "
                f"coasting in with fuel it will never use"
            )

    def test_terminal_burn_exceeds_the_cruise_floor(self, fleet_data):
        """
        Guidance cruises to a 12 km/s closing floor. Terminal burn must push
        impact speed past it rather than stopping there.
        """
        closure, _, _ = self._engagement(fleet_data, 6000, -6000)
        assert closure > 12.5, (
            f"impact closure {closure:.1f} km/s - terminal burn is not firing"
        )

    def test_a_reserve_is_held_back_for_late_evasion(self, fleet_data):
        """
        Committing the entire budget would leave nothing to null a late jink.
        Some delta-v must survive to impact.
        """
        _, dv_used, _ = self._engagement(fleet_data, 0, 0)
        assert dv_used < 14.0, "torpedo burned its entire budget with no reserve"
