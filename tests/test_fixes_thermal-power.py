"""
Regression tests for audited defects in src/thermal.py and src/power.py.

Each test here fails against the pre-fix code. They assert relationships and
invariants (energy conservation, per-tick budgets, sentinel semantics) rather
than magic constants copied from output.
"""

import dataclasses

import pytest

from src import combat
from src.power import (
    Battery,
    PowerSystem,
    Reactor,
    WeaponCapacitor,
    KINETIC_WEAPON_EFFICIENCY,
    calculate_weapon_energy_mj,
)
from src.thermal import (
    DropletRadiator,
    HeatSink,
    HeatSource,
    RadiatorArray,
    RadiatorPosition,
    RadiatorState,
    ThermalSystem,
    coerce_radiator_position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_thermal(
    capacity_gj: float,
    current_heat_gj: float,
    dissipation_kw: float,
    generation_kw: float,
) -> ThermalSystem:
    """Build a ThermalSystem with exactly known dissipation and generation."""
    radiators = RadiatorArray(radiators={
        pos: DropletRadiator(
            position=pos,
            max_dissipation_kw=dissipation_kw / 4.0,
            mass_tons=1.0,
            state=RadiatorState.EXTENDED,
        )
        for pos in RadiatorPosition
    })
    return ThermalSystem(
        heatsink=HeatSink(capacity_gj=capacity_gj, current_heat_gj=current_heat_gj),
        radiators=radiators,
        heat_sources=[HeatSource("test", generation_kw, active=True)],
    )


# ---------------------------------------------------------------------------
# Finding 1: combat.RadiatorPosition vs thermal.RadiatorPosition
# ---------------------------------------------------------------------------

class TestRadiatorPositionInterop:
    """
    combat.py used to define a second, incompatible RadiatorPosition enum
    (PORT/STARBOARD/DORSAL/VENTRAL) while thermal keyed its radiators by
    TAIL_PORT/... - so every lookup from the combat side missed and radiator
    damage silently did nothing. The two are now one enum.
    """

    def test_combat_and_thermal_share_one_enum(self):
        assert combat.RadiatorPosition is RadiatorPosition, (
            "combat.py has re-introduced its own RadiatorPosition - the mismatch "
            "that made radiator damage a no-op"
        )
        assert {p.name for p in combat.RadiatorPosition} == {
            "TAIL_PORT", "TAIL_STARBOARD", "TAIL_DORSAL", "TAIL_VENTRAL"
        }

    def test_string_positions_are_still_coerced(self):
        """The shim must keep normalising strings and bare names defensively."""
        for raw, expected in (
            ("PORT", RadiatorPosition.TAIL_PORT),
            ("tail_port", RadiatorPosition.TAIL_PORT),
            ("STARBOARD", RadiatorPosition.TAIL_STARBOARD),
            (RadiatorPosition.TAIL_DORSAL, RadiatorPosition.TAIL_DORSAL),
        ):
            assert coerce_radiator_position(raw) is expected

    def test_is_radiator_extended_tracks_state(self):
        th = make_thermal(100.0, 0.0, 1000.0, 0.0)
        th.radiators.retract_all()
        assert th.is_radiator_extended(RadiatorPosition.TAIL_PORT) is False

        th.radiators.extend_all()
        # Pre-fix this stayed False forever: the dict lookup returned None.
        assert th.is_radiator_extended(RadiatorPosition.TAIL_PORT) is True

    def test_radiator_damage_has_effect(self):
        th = make_thermal(100.0, 0.0, 4000.0, 0.0)
        th.radiators.extend_all()
        before = th.radiators.total_dissipation_kw

        damage_taken, destroyed, dissipation_lost = th.apply_radiator_damage(
            RadiatorPosition.TAIL_STARBOARD, 2.0
        )

        assert damage_taken > 0.0
        assert dissipation_lost > 0.0
        assert th.radiators.total_dissipation_kw < before
        assert destroyed is False

    def test_radiator_hit_resolver_end_to_end(self):
        """A hit resolved through combat.py must actually damage a radiator."""
        th = make_thermal(100.0, 0.0, 4000.0, 0.0)
        th.radiators.extend_all()
        health_before = sum(
            r.health_percent for r in th.radiators.radiators.values()
        )

        weapon = combat.Weapon(
            name="Test Coiler",
            weapon_type="kinetic",
            kinetic_energy_gj=5.0,
            cooldown_s=10.0,
            range_km=1000.0,
            flat_chipping=0.0,
        )

        # Force a hit: rng.random() must return <= hit chance.
        class AlwaysHit:
            def choice(self, seq):
                return seq[0]

            def random(self):
                return 0.0

        resolver = combat.RadiatorHitResolver(rng=AlwaysHit())
        result = resolver.resolve_radiator_hit(
            weapon, th, combat.HitLocation.LATERAL
        )

        assert result is not None and result.hit is True
        health_after = sum(
            r.health_percent for r in th.radiators.radiators.values()
        )
        assert health_after < health_before

    def test_unknown_position_is_loud(self):
        th = make_thermal(100.0, 0.0, 1000.0, 0.0)
        with pytest.raises(KeyError):
            th.is_radiator_extended("NOT_A_RADIATOR")


# ---------------------------------------------------------------------------
# Finding 7: heat overflow accounting in ThermalSystem.update
# ---------------------------------------------------------------------------

class TestHeatAccounting:
    """update() must report what actually happened to the stored heat."""

    def test_absorb_with_overflow_reports_excess(self):
        sink = HeatSink(capacity_gj=100.0, current_heat_gj=90.0)
        absorbed, overflow = sink.absorb_with_overflow(30.0)
        assert absorbed == pytest.approx(10.0)
        assert overflow == pytest.approx(20.0)
        assert sink.current_heat_gj == pytest.approx(100.0)
        # Legacy boolean API keeps working.
        assert sink.absorb(1.0) is False

    @pytest.mark.parametrize(
        "capacity,initial,dissipation_kw,generation_kw",
        [
            (100.0, 0.0, 0.0, 1_000_000.0),        # heating, no radiators
            (100.0, 50.0, 2_000_000.0, 1_000_000.0),  # cooling
            (10.0, 10.0, 0.0, 1_000_000.0),        # full sink, overflow
            (10.0, 10.0, 2_000_000.0, 1_000_000.0),  # full sink, spare radiators
            (100.0, 0.0, 1_000_000.0, 1_000_000.0),  # exact equilibrium
        ],
    )
    def test_net_heat_matches_actual_change(
        self, capacity, initial, dissipation_kw, generation_kw
    ):
        th = make_thermal(capacity, initial, dissipation_kw, generation_kw)
        before = th.heatsink.current_heat_gj

        result = th.update(dt_seconds=1.0)

        after = th.heatsink.current_heat_gj
        # net_heat_gj is the change in *stored* heat...
        assert result["net_heat_gj"] == pytest.approx(after - before)
        # ...and the books balance: what was made either radiated, was stored,
        # or was explicitly reported as overflow.
        assert result["heat_generated_gj"] == pytest.approx(
            result["heat_dissipated_gj"]
            + result["heat_overflow_gj"]
            + (after - before)
        )
        assert result["heat_overflow_gj"] >= 0.0

    def test_overflow_is_reported_not_hidden(self):
        th = make_thermal(10.0, 10.0, 0.0, 1_000_000.0)  # 1 GJ/s, sink full
        result = th.update(dt_seconds=1.0)
        assert result["heat_generated_gj"] == pytest.approx(1.0)
        assert result["heat_overflow_gj"] == pytest.approx(1.0)
        assert result["net_heat_gj"] == pytest.approx(0.0)

    def test_full_sink_does_not_block_radiators(self):
        """Radiators with spare capacity must pass generated heat straight out."""
        # 2 GJ/s of radiator throughput, 1 GJ/s generated, sink already full.
        th = make_thermal(10.0, 10.0, 2_000_000.0, 1_000_000.0)
        result = th.update(dt_seconds=1.0)

        assert result["heat_overflow_gj"] == pytest.approx(0.0)
        # Generation is rejected directly; the remaining 1 GJ/s drains the sink.
        assert result["heat_dissipated_gj"] == pytest.approx(2.0)
        assert th.heatsink.current_heat_gj == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Finding 8: kinetic capacitor energy budget
# ---------------------------------------------------------------------------

class TestKineticEnergyBudget:
    def test_capacitor_energy_budget_is_closed(self):
        kinetic_energy_gj = 4.32
        cap = WeaponCapacitor.from_weapon_data(
            {
                "type": "kinetic",
                "cooldown_s": 20.0,
                "kinetic_energy_gj": kinetic_energy_gj,
            },
            "spinal",
        )
        muzzle_energy_mj = kinetic_energy_gj * 1000.0

        # Stored energy = energy delivered to the slug + waste heat. Pre-fix the
        # capacitor was sized at 2x muzzle energy but only 30% of it was
        # accounted for as heat, leaving 40% unexplained.
        assert cap.calculate_heat_generated() + muzzle_energy_mj == pytest.approx(
            cap.capacity_mj
        )
        assert cap.capacity_mj * cap.efficiency == pytest.approx(muzzle_energy_mj)

    def test_helper_matches_capacitor_sizing(self):
        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 10.0,
            "kinetic_energy_gj": 1.5,
        }
        cap = WeaponCapacitor.from_weapon_data(weapon_data, "gun")
        assert calculate_weapon_energy_mj(weapon_data) == pytest.approx(
            cap.capacity_mj
        )


# ---------------------------------------------------------------------------
# Finding 11: Battery charge sentinel
# ---------------------------------------------------------------------------

class TestBatterySentinel:
    def test_unspecified_charge_starts_full(self):
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0,
        )
        assert battery.current_charge_gj == pytest.approx(100.0)

    def test_explicitly_empty_battery_stays_empty(self):
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0,
            current_charge_gj=0.0,
        )
        assert battery.current_charge_gj == 0.0
        assert battery.is_depleted is True

    def test_replace_on_drained_battery_does_not_refill(self):
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0,
        )
        battery.discharge(10.0, 10.0)  # fully drain
        assert battery.is_depleted

        copy = dataclasses.replace(battery)
        assert copy.current_charge_gj == 0.0
        assert copy.is_depleted is True


# ---------------------------------------------------------------------------
# Finding 5: battery discharge rate is a per-timestep limit
# ---------------------------------------------------------------------------

class TestBatteryDischargeBudget:
    def _starved_system(self, n_weapons: int) -> PowerSystem:
        ps = PowerSystem(
            reactor=Reactor(max_output_gw=0.0),
            battery=Battery(
                capacity_gj=1000.0,
                max_discharge_rate_gw=1.0,
                max_recharge_rate_gw=1.0,
            ),
        )
        ps.set_drive_throttle(0.0)
        for i in range(n_weapons):
            cap = ps.add_weapon_capacitor(
                f"gun_{i}",
                {"type": "kinetic", "cooldown_s": 1.0, "kinetic_energy_gj": 1.0},
            )
            cap.current_charge_mj = 0.0
        return ps

    def test_total_draw_respects_rate_limit(self):
        ps = self._starved_system(n_weapons=4)
        before = ps.battery.current_charge_gj

        ps.update(1.0)

        drained = before - ps.battery.current_charge_gj
        # 1 GW * 1 s = 1 GJ per tick, no matter how many capacitors ask.
        assert drained == pytest.approx(1.0, abs=1e-9)

    def test_draw_does_not_scale_with_weapon_count(self):
        one = self._starved_system(n_weapons=1)
        many = self._starved_system(n_weapons=8)

        one.update(1.0)
        many.update(1.0)

        drained_one = 1000.0 - one.battery.current_charge_gj
        drained_many = 1000.0 - many.battery.current_charge_gj
        assert drained_many == pytest.approx(drained_one, abs=1e-9)

    def test_budget_scales_with_timestep(self):
        ps = self._starved_system(n_weapons=4)
        ps.update(2.0)
        drained = 1000.0 - ps.battery.current_charge_gj
        assert drained == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Finding 10: reactor output fraction is validated
# ---------------------------------------------------------------------------

class TestReactorOutputFraction:
    def test_set_output_fraction_clamps(self):
        reactor = Reactor(max_output_gw=100.0)
        assert reactor.set_output_fraction(0.25) == pytest.approx(0.25)
        assert reactor.current_output_gw == pytest.approx(25.0)

        assert reactor.set_output_fraction(-1.0) == 0.0
        assert reactor.current_output_gw == 0.0

        assert reactor.set_output_fraction(5.0) == 1.0
        assert reactor.current_output_gw == pytest.approx(100.0)

    def test_waste_heat_tracks_output(self):
        reactor = Reactor(max_output_gw=1000.0, efficiency=0.99)
        full = reactor.calculate_waste_heat_gw()
        reactor.set_output_fraction(0.5)
        assert reactor.calculate_waste_heat_gw() == pytest.approx(full / 2.0)
