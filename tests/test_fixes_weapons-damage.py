"""
Regression tests for the audited defects in the weapons/damage partition.

Covers src/scenarios.py, src/torpedo.py, src/modules.py, src/damage.py,
src/combat.py and src/pointdefense.py. Assertions are written as relationships
and invariants (energy conservation, frame independence, monotonicity) rather
than magic constants, so they keep their teeth if the numbers are retuned.
"""

import json
import math
from pathlib import Path

import pytest

from src.combat import (
    CombatResolver,
    HitLocation,
    create_ship_armor_from_fleet_data,
    create_weapon_from_fleet_data,
)
from src.damage import (
    DamageCone,
    DamagePropagator,
    WeaponDamageProfile,
)
from src.damage import Module as DamageModule
from src.damage import ModuleLayout as DamageModuleLayout
from src.modules import Module, ModuleLayout, ModulePosition, ModuleType
from src.physics import Vector3D
from src.pointdefense import (
    PDLaser,
    TargetMaterial,
    estimate_cross_section_m2,
)
from src.torpedo import (
    GuidanceMode,
    Torpedo,
    TorpedoGuidance,
    TorpedoSpecs,
)


FLEET_PATH = Path(__file__).resolve().parents[1] / "data" / "fleet_ships.json"


@pytest.fixture(scope="module")
def fleet_data():
    with open(FLEET_PATH) as f:
        return json.load(f)


# =============================================================================
# Finding 1: scenario ships were built by a divergent, wrong setup path
# =============================================================================

class TestScenarioShipSetup:
    """ScenarioRunner must build ships the same way the simulator does."""

    def _build(self, scenario_name):
        from src.scenarios import SCENARIO_REGISTRY, ScenarioRunner
        from src.simulation import CombatSimulation

        factory = SCENARIO_REGISTRY[scenario_name]
        config = factory() if callable(factory) else factory
        runner = ScenarioRunner()
        sim = CombatSimulation()
        runner._setup_ships(sim, config)
        return config, sim

    def test_ships_only_carry_their_own_weapons(self, fleet_data):
        """A ship must not be issued one of every weapon type in the file."""
        _, sim = self._build("head_on_pass")
        all_weapon_types = fleet_data["weapon_types"]

        for ship in sim.ships.values():
            declared = fleet_data["ships"][ship.ship_type]["weapons"]
            assert len(ship.weapons) == len(declared), (
                f"{ship.ship_type} has {len(ship.weapons)} weapon slots but "
                f"declares {len(declared)}"
            )
            assert len(ship.weapons) < len(all_weapon_types), (
                "ship was armed with the entire weapon catalogue"
            )

    def test_scenario_ships_have_full_subsystems(self):
        """Geometry, power and point defense must be wired up."""
        _, sim = self._build("head_on_pass")
        for ship in sim.ships.values():
            assert ship.geometry is not None
            assert ship.power_system is not None
            assert len(ship.point_defense) > 0

    def test_ship_mass_comes_from_fleet_data(self, fleet_data):
        """Masses must not silently fall back to a single hard-coded default."""
        _, sim = self._build("head_on_pass")
        for ship in sim.ships.values():
            perf = fleet_data["ships"][ship.ship_type]["performance"]
            expected_kg = perf["max_wet_mass_tons"] * 1000.0
            assert ship.kinematic_state.mass_kg == pytest.approx(expected_kg, rel=1e-6)

    def test_missile_exchange_actually_has_torpedoes(self):
        """The torpedo scenario must field hulls that mount launchers."""
        _, sim = self._build("missile_exchange")
        assert sim.ships, "scenario produced no ships"
        for ship in sim.ships.values():
            assert ship.torpedo_launcher is not None, (
                f"{ship.ship_id} ({ship.ship_type}) has no torpedo launcher"
            )
            assert ship.torpedo_launcher.current_magazine > 0

    def test_no_zero_velocity_direct_fire_gun_on_every_ship(self, fleet_data):
        """Ships must not be handed weapons their class does not mount."""
        _, sim = self._build("head_on_pass")
        for ship in sim.ships.values():
            declared_types = {
                w["type"] for w in fleet_data["ships"][ship.ship_type]["weapons"]
            }
            for state in ship.weapons.values():
                assert any(
                    state.weapon.name == fleet_data["weapon_types"][t]["name"]
                    for t in declared_types
                ), f"{state.weapon.name} is not part of the {ship.ship_type} loadout"


def test_ammo_multiplier_applies_to_actual_loadout():
    """ammo_multiplier must scale whatever the ship really carries."""
    from src.scenarios import ScenarioConfig, ScenarioRunner, ShipConfiguration, VictoryCondition
    from src.simulation import CombatSimulation

    def build(multiplier):
        config = ScenarioConfig(
            name="ammo_probe",
            display_name="Ammo Probe",
            description="",
            ships=[
                ShipConfiguration(
                    ship_id="a1",
                    ship_type="destroyer",
                    faction="alpha",
                    position_km=(0, 0, 0),
                    velocity_kps=(0, 0, 0),
                    ammo_multiplier=multiplier,
                ),
                ShipConfiguration(
                    ship_id="b1",
                    ship_type="destroyer",
                    faction="beta",
                    position_km=(100, 0, 0),
                    velocity_kps=(0, 0, 0),
                ),
            ],
            decision_interval=30.0,
            time_limit_s=60.0,
            victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        )
        sim = CombatSimulation()
        ScenarioRunner()._setup_ships(sim, config)
        return sim.ships["a1"]

    base = build(1.0)
    doubled = build(2.0)
    base_ammo = sum(w.ammo_remaining for w in base.weapons.values())
    doubled_ammo = sum(w.ammo_remaining for w in doubled.weapons.values())
    assert base_ammo > 0
    assert doubled_ammo == 2 * base_ammo


# =============================================================================
# Finding 4: proportional navigation cross-product was inverted
# =============================================================================

def _crossing_torpedo(mode):
    """Torpedo closing along +x, target 100 km downrange crossing at +y."""
    specs = TorpedoSpecs.trident()
    torpedo = Torpedo(
        specs=specs,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(10000, 0, 0),
        target_id="t",
        guidance_mode=mode,
    )
    return torpedo, Vector3D(100000, 0, 0), Vector3D(0, 2000, 0)


class TestProportionalNavigationSign:

    def test_pn_steers_toward_the_lead_point(self):
        """PN command must have a +y component when the target crosses to +y."""
        torpedo, target_pos, target_vel = _crossing_torpedo(
            GuidanceMode.PROPORTIONAL_NAV
        )
        direction = TorpedoGuidance()._proportional_nav_guidance(
            torpedo, target_pos, target_vel, dt=1.0
        )
        assert direction.y > 0, (
            f"PN steers away from the lead point: {direction}"
        )

    def test_pn_agrees_with_smart_guidance(self):
        """PN and SMART solve the same geometry and must agree in sign."""
        torpedo, target_pos, target_vel = _crossing_torpedo(
            GuidanceMode.PROPORTIONAL_NAV
        )
        pn_dir = TorpedoGuidance()._proportional_nav_guidance(
            torpedo, target_pos, target_vel, dt=1.0
        )
        smart_torpedo, _, _ = _crossing_torpedo(GuidanceMode.SMART)
        smart_cmd = TorpedoGuidance()._smart_guidance(
            smart_torpedo, target_pos, target_vel, 1.0, 1.0
        )
        assert math.copysign(1.0, pn_dir.y) == math.copysign(1.0, smart_cmd.direction.y)

    def test_intercept_correction_leads_target(self):
        """The intercept-mode PN correction term must lead, not trail."""
        torpedo, target_pos, target_vel = _crossing_torpedo(GuidanceMode.INTERCEPT)
        direction = TorpedoGuidance()._intercept_guidance(
            torpedo, target_pos, target_vel, dt=1.0
        )
        assert direction.y > 0


# =============================================================================
# Finding 9: guidance must be Galilean-invariant
# =============================================================================

class TestGuidanceFrameInvariance:

    @staticmethod
    def _thrust_direction(boost_mps):
        """Same engagement geometry, viewed from a frame boosted along +y."""
        boost = Vector3D(0, boost_mps, 0)
        specs = TorpedoSpecs.trident()
        torpedo = Torpedo(
            specs=specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(20000, 0, 0) + boost,
            target_id="t",
            guidance_mode=GuidanceMode.INTERCEPT,
        )
        target_pos = Vector3D(80000, 0, 0)
        target_vel = Vector3D(10000, 0, 0) + boost
        return TorpedoGuidance()._intercept_guidance(
            torpedo, target_pos, target_vel, dt=1.0
        )

    def test_braking_command_is_frame_independent(self):
        rest = self._thrust_direction(0.0)
        for boost in (10000.0, 40000.0):
            boosted = self._thrust_direction(boost)
            assert boosted.x == pytest.approx(rest.x, abs=1e-6)
            assert boosted.y == pytest.approx(rest.y, abs=1e-6)
            assert boosted.z == pytest.approx(rest.z, abs=1e-6)


# =============================================================================
# Finding 10: Torpedo.update must honour the guidance command's channel
# =============================================================================

class TestTorpedoHonoursGuidanceCommand:

    def test_rcs_trim_does_not_become_a_main_engine_burn(self):
        """An RCS-only trim must cost RCS-scale delta-v, not main-engine scale."""
        specs = TorpedoSpecs.trident()
        # Geometry that produces an RCS_TRIM command: high closing speed and a
        # small residual lateral drift the RCS has time to remove before impact.
        def make():
            return Torpedo(
                specs=specs,
                position=Vector3D(0, 0, 0),
                velocity=Vector3D(20000, 100, 0),
                target_id="t",
                guidance_mode=GuidanceMode.COLLISION,
            )

        target_pos = Vector3D(2000000, 0, 0)
        target_vel = Vector3D(0, 0, 0)

        command = TorpedoGuidance().update_guidance_command(
            make(), target_pos, target_vel, 1.0
        )
        assert command.use_rcs, f"expected an RCS trim, got {command.reason}"

        torpedo = make()
        before = torpedo.remaining_delta_v_kps
        torpedo.update(1.0, target_pos, target_vel)
        rcs_cost = before - torpedo.remaining_delta_v_kps

        # Compare against what a full main-engine burn would have cost.
        reference = make()
        ref_before = reference.remaining_delta_v_kps
        reference.apply_thrust(Vector3D(0, -1, 0), 1.0)
        main_cost = ref_before - reference.remaining_delta_v_kps

        assert rcs_cost > 0
        assert rcs_cost < main_cost / 5.0, (
            f"RCS trim cost {rcs_cost:.4f} km/s vs main burn {main_cost:.4f} km/s"
        )

    def test_rcs_trim_does_not_overshoot(self):
        """A fine trim must shave the drift, not blow through it."""
        specs = TorpedoSpecs.trident()
        torpedo = Torpedo(
            specs=specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(20000, 100, 0),
            target_id="t",
            guidance_mode=GuidanceMode.COLLISION,
        )
        torpedo.update(1.0, Vector3D(2000000, 0, 0), Vector3D(0, 0, 0))
        # Drift reduced but not reversed - a full main-engine burn here would
        # swing it past zero.
        assert 0.0 < torpedo.velocity.y < 100.0


# =============================================================================
# Finding 2: COLLISION guidance must intercept a crossing target
# =============================================================================

def _closest_approach(crossing_speed_mps, dt=0.2, max_steps=6000):
    """Fly a torpedo against a constant-velocity crossing target."""
    torpedo = Torpedo(
        specs=TorpedoSpecs.trident(),
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(8000, 0, 0),
        target_id="t",
        guidance_mode=GuidanceMode.COLLISION,
    )
    target_pos = Vector3D(300000, 0, 0)
    target_vel = Vector3D(0, crossing_speed_mps, 0)

    best = float("inf")
    for step in range(max_steps):
        rel_p = torpedo.position - target_pos
        rel_v = torpedo.velocity - target_vel
        vv = rel_v.dot(rel_v)
        if vv > 0:
            # Continuous closest approach within this step - sampling only at
            # step boundaries would report a multi-km "miss" for a clean hit.
            tau = max(0.0, min(dt, -rel_p.dot(rel_v) / vv))
            best = min(best, (rel_p + rel_v * tau).magnitude)
        torpedo.update(dt, target_pos, target_vel)
        target_pos = target_pos + target_vel * dt
        if rel_p.magnitude > 500000 and step > 200:
            break
    return best


class TestCollisionGuidanceIntercept:

    def test_hits_a_non_maneuvering_head_on_target(self):
        assert _closest_approach(0.0) < 100.0

    @pytest.mark.parametrize("crossing_mps", [1000.0, 2000.0])
    def test_hits_a_non_maneuvering_crossing_target(self, crossing_mps):
        miss = _closest_approach(crossing_mps)
        assert miss < 500.0, f"missed a crossing target by {miss / 1000:.1f} km"

    def test_miss_degrades_gracefully_with_crossing_speed(self):
        """Harder geometry may miss, but must not be better than an easy one."""
        easy = _closest_approach(1000.0)
        hard = _closest_approach(5000.0)
        assert easy <= hard


# =============================================================================
# Finding 5: Module.armor_rating must actually protect
# =============================================================================

class TestModuleArmorRating:

    @staticmethod
    def _module(armor_rating):
        return Module(
            name="probe",
            module_type=ModuleType.CARGO,
            health_percent=100.0,
            armor_rating=armor_rating,
            position=ModulePosition(0),
        )

    def test_armored_module_takes_less_damage(self):
        bare = self._module(0.0)
        armored = self._module(0.5)
        bare.damage(1.0)
        armored.damage(1.0)
        assert armored.health_percent > bare.health_percent

    def test_damage_scales_with_one_minus_armor_rating(self):
        bare = self._module(0.0)
        armored = self._module(0.5)
        bare.damage(1.0)
        armored.damage(1.0)
        bare_loss = 100.0 - bare.health_percent
        armored_loss = 100.0 - armored.health_percent
        assert armored_loss == pytest.approx(bare_loss * 0.5)

    def test_unarmored_behaviour_unchanged(self):
        bare = self._module(0.0)
        bare.damage(1.0)
        assert bare.health_percent == pytest.approx(50.0)

    def test_armor_absorbs_less_energy_so_more_passes_through(self):
        bare = self._module(0.0)
        armored = self._module(0.5)
        assert armored.damage(1.0) > bare.damage(1.0)

    def test_layout_bulkheads_are_tougher_than_cargo(self, fleet_data):
        """The armored bulkheads in the real layouts must be meaningfully harder."""
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
        bulkheads = [
            m for m in layout.get_all_modules() if m.armor_rating > 0.0
        ]
        assert bulkheads, "destroyer layout defines no internal armor"
        reference = self._module(0.0)
        reference.damage(0.5)
        for bulkhead in bulkheads:
            before = bulkhead.health_percent
            bulkhead.damage(0.5)
            assert (before - bulkhead.health_percent) < (100.0 - reference.health_percent)


# =============================================================================
# Finding 3: lateral hit geometry must be in the ship's frame
# =============================================================================

class TestLateralConeGeometry:

    def test_module_set_is_independent_of_world_frame_axis(self, fleet_data):
        """Identical hits differing only in world-frame direction must match."""
        results = []
        for direction in [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0)]:
            layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
            modules = layout.get_modules_in_cone(
                HitLocation.LATERAL, angle_deg=30.0, direction_vector=direction
            )
            results.append([m.name for m in modules])
        assert all(r == results[0] for r in results), (
            "lateral damage depends on the arbitrary world axis of the shot"
        )

    def test_bridge_is_reachable_by_lateral_fire(self, fleet_data):
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
        names = [
            m.name
            for m in layout.get_modules_in_cone(HitLocation.LATERAL, angle_deg=30.0)
        ]
        assert any("Bridge" in n for n in names)

    def test_outer_modules_shield_the_centreline(self, fleet_data):
        """Hull-side modules must be reached before centreline modules."""
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.LATERAL, angle_deg=30.0)
        offsets = [abs(m.position.lateral_offset) for m in modules]
        assert offsets == sorted(offsets, reverse=True)

    def test_impact_side_decides_which_flank_is_hit_first(self, fleet_data):
        """With the ship's orientation known, near-side modules come first."""
        ship_right = (1.0, 0.0, 0.0)

        def first_offsets(direction):
            layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
            modules = layout.get_modules_in_cone(
                HitLocation.LATERAL,
                angle_deg=30.0,
                direction_vector=direction,
                ship_right=ship_right,
            )
            return modules[0].position.lateral_offset

        # Travelling toward starboard means entering through the port flank.
        toward_starboard = first_offsets((1.0, 0.0, 0.0))
        toward_port = first_offsets((-1.0, 0.0, 0.0))
        assert toward_starboard < 0 < toward_port

    def test_entry_layer_constrains_axial_reach(self, fleet_data):
        """A hit at a known station cannot rake the whole hull."""
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
        modules = layout.get_modules_in_cone(
            HitLocation.LATERAL,
            angle_deg=30.0,
            ship_right=(1.0, 0.0, 0.0),
            direction_vector=(1.0, 0.0, 0.0),
            entry_layer_index=2,
        )
        assert modules
        assert {m.position.layer_index for m in modules} == {2}

    def test_nose_hits_still_ordered_by_depth(self, fleet_data):
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.NOSE, angle_deg=30.0)
        layers = [m.position.layer_index for m in modules]
        assert layers == sorted(layers)


# =============================================================================
# Finding 6: damage propagation must not create energy
# =============================================================================

def _straight_line_layout(count=5, spacing=10.0, health=5.0):
    modules = [
        DamageModule(
            name=f"M{i}",
            position=Vector3D(i * spacing, 0.0, 0.0),
            health=health,
            max_health=health,
            radius_m=2.0,
        )
        for i in range(count)
    ]
    return DamageModuleLayout(modules=modules, ship_length_m=100.0)


class TestDamagePropagationEnergyConservation:

    @pytest.mark.parametrize(
        "profile",
        [
            WeaponDamageProfile.KINETIC,
            WeaponDamageProfile.EXPLOSIVE,
            WeaponDamageProfile.LASER,
        ],
    )
    def test_total_damage_never_exceeds_cone_energy(self, profile):
        layout = _straight_line_layout()
        cone = DamageCone(
            entry_point=Vector3D(-1.0, 0.0, 0.0),
            direction=Vector3D(1.0, 0.0, 0.0),
            cone_angle_deg=15.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
            damage_profile=profile,
        )
        results = DamagePropagator(enable_spalling=False).propagate(cone, layout)
        total = sum(r.damage_taken_gj for r in results)
        assert total <= cone.initial_energy_gj + 1e-9, (
            f"{total:.3f} GJ dealt from a {cone.initial_energy_gj} GJ cone"
        )

    def test_reported_remaining_energy_is_monotonic(self):
        layout = _straight_line_layout()
        cone = DamageCone(
            entry_point=Vector3D(-1.0, 0.0, 0.0),
            direction=Vector3D(1.0, 0.0, 0.0),
            cone_angle_deg=15.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        results = DamagePropagator(enable_spalling=False).propagate(cone, layout)
        remaining = [r.remaining_energy_gj for r in results]
        assert remaining == sorted(remaining, reverse=True), remaining
        assert remaining[0] <= cone.initial_energy_gj

    def test_decay_accumulates_over_the_path(self):
        """Deeper modules must see strictly less energy than shallower ones."""
        layout = _straight_line_layout(count=5, spacing=20.0, health=0.05)
        cone = DamageCone(
            entry_point=Vector3D(-1.0, 0.0, 0.0),
            direction=Vector3D(1.0, 0.0, 0.0),
            cone_angle_deg=15.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
            damage_profile=WeaponDamageProfile.EXPLOSIVE,
        )
        results = DamagePropagator(enable_spalling=False).propagate(cone, layout)
        assert len(results) >= 3
        energies = [r.remaining_energy_gj for r in results]
        for earlier, later in zip(energies, energies[1:]):
            assert later < earlier


# =============================================================================
# Finding 7: resolve_hit energy accounting must balance
# =============================================================================

class TestResolveHitEnergyBalance:

    @pytest.mark.parametrize("weapon_type", ["light_coilgun_mk3", "spinal_coiler_mk3"])
    @pytest.mark.parametrize("location", [HitLocation.NOSE, HitLocation.LATERAL])
    def test_absorbed_plus_remaining_never_exceeds_kinetic_energy(
        self, fleet_data, weapon_type, location
    ):
        import random

        weapon = create_weapon_from_fleet_data(fleet_data, weapon_type)
        armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        resolver = CombatResolver(rng=random.Random(7))

        for _ in range(60):
            result = resolver.resolve_hit(weapon, armor, location=location)
            total = result.damage_absorbed + result.remaining_damage_gj
            assert total <= weapon.kinetic_energy_gj + 1e-9, (
                f"{total:.3f} GJ accounted for from a "
                f"{weapon.kinetic_energy_gj:.3f} GJ projectile"
            )

    def test_bleed_through_branch_is_live(self, fleet_data):
        """A non-penetrating hit must report its computed bleed-through."""
        import random

        weapon = create_weapon_from_fleet_data(fleet_data, "light_coilgun_mk3")
        armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        resolver = CombatResolver(rng=random.Random(3))

        result = resolver.resolve_hit(weapon, armor, location=HitLocation.NOSE)
        assert not result.penetrated
        assert result.remaining_damage_gj > 0.0
        assert result.damage_absorbed + result.remaining_damage_gj == pytest.approx(
            weapon.kinetic_energy_gj
        )


# =============================================================================
# Finding 8: PD ablation must depend on range
# =============================================================================

@pytest.fixture
def pd_laser():
    return PDLaser(
        power_mw=5.0,
        aperture_m=0.5,
        wavelength_nm=1000.0,
        range_km=250.0,
        cooldown_s=0.5,
    )


class TestPointDefenseRangeDependence:

    def test_coupling_fraction_is_bounded(self, pd_laser):
        for distance in (1.0, 50.0, 250.0):
            fraction = pd_laser.beam_coupling_fraction(distance, 0.01)
            assert 0.0 <= fraction <= 1.0

    def test_coupling_falls_off_with_range(self, pd_laser):
        near = pd_laser.beam_coupling_fraction(10.0, 0.01)
        far = pd_laser.beam_coupling_fraction(250.0, 0.01)
        assert far < near

    def test_ablation_rate_is_non_increasing_with_range(self, pd_laser):
        cross_section = estimate_cross_section_m2(50.0, TargetMaterial.STEEL)
        rates = [
            pd_laser.calculate_ablation_rate(d, TargetMaterial.STEEL, cross_section)
            for d in (1.0, 10.0, 50.0, 100.0, 250.0)
        ]
        for near, far in zip(rates, rates[1:]):
            assert far <= near
        assert rates[-1] < rates[0], "ablation rate is range-independent"

    def test_intercept_is_harder_at_long_range(self, pd_laser):
        near = pd_laser.shots_to_destroy_slug(50.0, 10.0, TargetMaterial.STEEL)
        far = pd_laser.shots_to_destroy_slug(50.0, 250.0, TargetMaterial.STEEL)
        assert far > near

    def test_cross_section_scales_with_mass_and_density(self):
        light = estimate_cross_section_m2(1.0, TargetMaterial.STEEL)
        heavy = estimate_cross_section_m2(100.0, TargetMaterial.STEEL)
        assert heavy > light
        # Denser material of the same mass is smaller.
        steel = estimate_cross_section_m2(50.0, TargetMaterial.STEEL)
        tungsten = estimate_cross_section_m2(50.0, TargetMaterial.TUNGSTEN)
        assert tungsten < steel

    def test_slug_engagement_is_slower_at_range(self, pd_laser):
        from src.pointdefense import PDEngagement, Slug

        engagement = PDEngagement(pd_laser)
        near = engagement.engage_slug(
            Slug(mass_kg=20.0, material=TargetMaterial.STEEL), distance_km=10.0
        )
        far = engagement.engage_slug(
            Slug(mass_kg=20.0, material=TargetMaterial.STEEL), distance_km=250.0
        )
        assert far.shots_fired > near.shots_fired
