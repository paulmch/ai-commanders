"""
Heat model (round 2): TI per-shot weapon heat, the mass-neutral thermal refit,
drive heat tied to radiator capacity, the heat ladder (90% guns hold / 100%
radiators forced out / 150% reactor meltdown), the systems-bus power model,
the battery data path, radiator automation, telemetry and determinism.
"""

import copy
import json
import math

import pytest

from src.combat import HitLocation, radiator_hit_chance
from src.physics import Vector3D
from src.power import KINETIC_WEAPON_EFFICIENCY, PowerSystem
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.thermal import (
    AUTO_RADIATOR_HYSTERESIS_PCT,
    HEAT_AUTO_EXTEND_FRACTION,
    HEAT_GUN_HOLD_FRACTION,
    HEAT_WARNING_LEVELS,
    REACTOR_MELTDOWN_HEAT_FRACTION,
    ThermalSystem,
)


HULLS = ("corvette", "frigate", "destroyer", "cruiser", "battlecruiser",
         "battleship", "dreadnought", "dreadnought_siege", "cruiser_torpedo")

# Pre-refit masses and class accelerations (CLAUDE.md ship table): the thermal
# refit only moves tonnage between the heat sink and the radiators.
EXPECTED_MASS = {  # (total_dry, total_wet, combat g)
    "corvette": (1895.0, 1990.0, 3.0),
    "frigate": (1895.0, 1990.0, 3.0),
    "destroyer": (2843.0, 2985.0, 2.0),
    "cruiser": (3790.0, 3980.0, 1.5),
    "battlecruiser": (3790.0, 3980.0, 1.5),
    "battleship": (5685.0, 5969.0, 1.0),
    "dreadnought": (7581.0, 7959.0, 0.75),
    "cruiser_torpedo": (3790.0, 3980.0, 1.5),
}


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _ship(sim, sid, stype, faction, fleet, x_km=0.0, fwd=1.0):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, fleet,
        position=Vector3D(x_km * 1000.0, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(fwd, 0, 0))
    sim.add_ship(ship)
    return ship


def _events(sim, etype):
    return sim.get_events_by_type(etype)


# =============================================================================
# Per-shot weapon heat (decision 3)
# =============================================================================

class TestPerShotHeat:
    @pytest.mark.parametrize("weapon,expected_gj", [
        ("coilgun_mk3", 0.72 * (1 / 0.7 - 1)),            # 0.31 GJ
        ("heavy_coilgun_mk3", 1.22 * (1 / 0.7 - 1)),      # 0.52 GJ
        ("spinal_coiler_mk3", 4.29 * (1 / 0.7 - 1)),      # 1.84 GJ
        ("heavy_siege_coiler_mk3", 7.25 * (1 / 0.7 - 1)),  # 3.11 GJ
    ])
    def test_ti_mk3_efficiency(self, fleet, weapon, expected_gj):
        """Shot energy = KE / 0.7, waste heat = shot energy x 0.3."""
        assert KINETIC_WEAPON_EFFICIENCY == 0.7
        ke = fleet["weapon_types"][weapon]["kinetic_energy_gj"]
        ps = PowerSystem.from_ship_data({})
        cap = ps.add_weapon_capacitor(weapon, {
            "type": "kinetic", "kinetic_energy_gj": ke, "cooldown_s": 20.0})
        assert cap.capacity_mj == pytest.approx(ke * 1000.0 / 0.7)
        assert ps.fire_weapon(weapon) == pytest.approx(expected_gj)

    def test_a_shot_heats_the_ship_exactly_once(self, fleet):
        """No flat 0.5 GJ adder on top of the capacitor waste (the old double count)."""
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
        a = _ship(sim, "a", "destroyer", "alpha", fleet, 0, 1)
        _ship(sim, "b", "destroyer", "beta", fleet, 100, -1)
        before = a.thermal_system.heatsink.current_heat_gj
        assert sim.inject_command("a", {"type": "fire_at", "weapon_slot": "weapon_0",
                                        "target_id": "b"})
        added = a.thermal_system.heatsink.current_heat_gj - before
        assert added == pytest.approx(4.29 * (1 / 0.7 - 1))

    def test_pd_burst_heat_is_ti_turret(self, fleet):
        """TI PointDefenseLaserTurret: 50 MJ / 0.25 drawn, 150 MJ heat per 5 s cycle."""
        ship = create_ship_from_fleet_data("d", "destroyer", "alpha", fleet)
        pd = ship.point_defense[0]
        cap = ship.power_system.weapon_capacitors[pd.turret_name]
        assert cap.capacity_mj == pytest.approx(200.0)
        assert cap.calculate_heat_generated() == pytest.approx(150.0)
        assert pd.laser.waste_heat_w() == pytest.approx(30e6)
        # The beam itself (the calibrated kill model) is unchanged.
        assert pd.laser.power_mw == fleet["weapon_types"]["pd_laser"]["power_draw_mw"] == 5


# =============================================================================
# Thermal refit (decision 5)
# =============================================================================

class TestThermalRefit:
    @pytest.mark.parametrize("hull", HULLS)
    def test_sink_radiator_and_drive_numbers(self, fleet, hull):
        th = ThermalSystem.from_ship_data(fleet["ships"][hull])
        assert th.heatsink.capacity_gj == 320.0
        assert th.radiators.nominal_dissipation_kw == pytest.approx(780_000.0)
        # Drive heat at full throttle = extended radiator rejection
        assert th.drive_heat_full_kw == pytest.approx(780_000.0)
        th.set_drive_throttle(0.5)
        assert th.get_total_heat_generation_kw() == pytest.approx(390_000.0 + 1_000.0)

    @pytest.mark.parametrize("hull", HULLS)
    def test_refit_is_mass_neutral(self, fleet, hull):
        spec = fleet["ships"][hull]
        thermal = spec["thermal"]
        # 138 t of sink + radiator before (128 + 10) and after (78 + 60)
        assert thermal["heatsink"]["mass_tons"] + thermal["radiator"]["mass_tons"] == 138
        assert thermal["heatsink"]["mass_tons"] == 78
        assert thermal["radiator"]["mass_tons"] == 60
        assert (thermal["heatsink"]["mass_tons"] + thermal["radiator"]["mass_tons"]
                + thermal["battery"]["mass_tons"]) == spec["mass_breakdown"]["thermal_systems"]
        # Lithium ratio kept (TI LithiumHeatSink: 525 GJ / 128 t)
        assert thermal["heatsink"]["capacity_gj"] / 78 == pytest.approx(525 / 128, rel=0.01)
        banks = spec["modules"]["heatsinks"]
        assert sum(b["mass_tons"] for b in banks) == pytest.approx(78)
        assert sum(b["capacity_gj"] for b in banks) == 320
        if hull in EXPECTED_MASS:
            dry, wet, g = EXPECTED_MASS[hull]
            mb = spec["mass_breakdown"]
            assert (mb["total_dry"], mb["total_wet"]) == (dry, wet)
            ship = create_ship_from_fleet_data("s", hull, "alpha", fleet)
            accel_g = ship.kinematic_state.max_acceleration_ms2() / 9.81
            assert accel_g == pytest.approx(g, rel=0.02)


# =============================================================================
# Heat ladder (decision 6)
# =============================================================================

class TestHeatLadder:
    def _duel(self, fleet, seed=5):
        from src.firecontrol import WeaponsCommand, WeaponsOrder
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
        a = _ship(sim, "a", "destroyer", "alpha", fleet, 0, 1)
        b = _ship(sim, "b", "destroyer", "beta", fleet, 300, -1)
        sim.inject_command("a", {"type": "set_target", "target_id": "b"})
        sim.inject_command("a", {"type": "weapons_order", "order": WeaponsOrder(
            command=WeaponsCommand.FIRE_IMMEDIATE, weapon_slot="all", target_id="b")})
        return sim, a, b

    def test_heat_is_never_discarded(self, fleet):
        sim, a, _ = self._duel(fleet)
        cap = a.thermal_system.heatsink.capacity_gj
        a.thermal_system.add_heat("test", cap * 1.2)
        assert a.heat_fraction == pytest.approx(1.2)
        assert a.heat_percent == pytest.approx(120.0)

    def test_guns_hold_at_90_percent(self, fleet):
        sim, a, _ = self._duel(fleet)
        cap = a.thermal_system.heatsink.capacity_gj
        a.thermal_system.heatsink.current_heat_gj = cap * (HEAT_GUN_HOLD_FRACTION - 0.05)
        sim.step()
        assert a.shots_fired > 0  # below the line the guns fire
        fired = a.shots_fired
        a.thermal_system.heatsink.current_heat_gj = cap * (HEAT_GUN_HOLD_FRACTION + 0.001)
        for _ in range(25):  # past a full gun cooldown
            a.thermal_system.heatsink.current_heat_gj = max(
                a.thermal_system.heatsink.current_heat_gj, cap * (HEAT_GUN_HOLD_FRACTION + 0.001))
            sim.step()
        assert a.shots_fired == fired
        reasons = [e.data.get("reason") for e in _events(sim, SimulationEventType.WEAPON_STATUS)
                   if e.ship_id == "a"]
        assert "heat_limited" in reasons
        # fire_at commands are gated too
        assert not sim.inject_command("a", {"type": "fire_at", "weapon_slot": "weapon_1",
                                            "target_id": "b"})

    def test_point_defense_keeps_firing_above_90(self, fleet):
        from tests.test_doctrine_accuracy import _ballistic_trident
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=3)
        d = _ship(sim, "d", "destroyer", "alpha", fleet, 0, 1)
        _ship(sim, "e", "corvette", "beta", fleet, 900, -1)
        cap = d.thermal_system.heatsink.capacity_gj
        d.thermal_system.heatsink.current_heat_gj = cap * 0.95
        tf = _ballistic_trident(240_000, 4_000, "d")
        tf.source_ship_id = "e"
        sim.torpedoes.append(tf)
        for _ in range(20):
            sim.step()
        assert _events(sim, SimulationEventType.PD_ENGAGED), "PD went silent above 90%"

    def test_radiators_forced_out_at_100_and_retract_refused(self, fleet):
        sim, a, _ = self._duel(fleet)
        th = a.thermal_system
        assert not a.radiators_extended
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * HEAT_AUTO_EXTEND_FRACTION
        sim.step()
        assert a.radiators_extended
        forced = [e for e in _events(sim, SimulationEventType.RADIATOR_EXTENDED)
                  if e.ship_id == "a"]
        assert forced and forced[-1].data.get("reason") == "heat_limit"
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * 1.1
        assert not sim.inject_command("a", {"type": "set_radiators", "extend": False})
        assert a.radiators_extended
        # Below 100% the captain may retract again
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * 0.5
        assert sim.inject_command("a", {"type": "set_radiators", "extend": False})
        assert not a.radiators_extended

    def test_drive_is_never_capped(self, fleet):
        sim, a, _ = self._duel(fleet)
        a.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                      duration=1e9, throttle=1.0, direction=Vector3D(1, 0, 0))
        a.thermal_system.heatsink.current_heat_gj = a.thermal_system.heatsink.capacity_gj * 1.3
        sim.step()
        assert a.applied_throttle == pytest.approx(1.0)

    def test_meltdown_kills_through_the_reactor_path(self, fleet):
        sim, a, _ = self._duel(fleet)
        cap = a.thermal_system.heatsink.capacity_gj
        a.thermal_system.heatsink.current_heat_gj = cap * (REACTOR_MELTDOWN_HEAT_FRACTION - 0.01)
        sim.step()
        assert not a.is_destroyed
        a.thermal_system.add_heat("test", cap * 0.05)  # e.g. a salvo's worth of heat
        sim.step()
        assert a.is_destroyed and not a.is_dying
        melt = [e for e in _events(sim, SimulationEventType.REACTOR_MELTDOWN) if e.ship_id == "a"]
        assert len(melt) == 1 and melt[0].data["heat"] >= REACTOR_MELTDOWN_HEAT_FRACTION
        destroyed = [e for e in _events(sim, SimulationEventType.SHIP_DESTROYED)
                     if e.ship_id == "a"]
        assert destroyed[0].data["cause"] == "reactor_overheat"
        assert destroyed[0].data["killer_id"] is None
        # Same death as a reactor kill: the reactor modules are destroyed and
        # logged (the replay viewer detonates on that).
        reactors = a._get_modules_by_type("reactor")
        assert reactors and all(m.is_destroyed for m in reactors)
        logged = {e.data["module_name"] for e in _events(sim, SimulationEventType.MODULE_DESTROYED)
                  if e.ship_id == "a"}
        assert {m.name for m in reactors} <= logged

    def test_heat_warnings_fire_once_per_upward_crossing(self, fleet):
        sim, a, _ = self._duel(fleet)
        th = a.thermal_system
        cap = th.heatsink.capacity_gj
        for frac in (0.5, 0.91, 0.92, 1.01, 1.02, 1.26, 1.27):
            th.heatsink.current_heat_gj = cap * frac
            sim.step()
        levels = [e.data["level"] for e in _events(sim, SimulationEventType.HEAT_WARNING)
                  if e.ship_id == "a"]
        assert levels == list(HEAT_WARNING_LEVELS)
        # Cool well below 0.9, then cross again: the level re-arms once.
        th.heatsink.current_heat_gj = cap * 0.5
        sim.step()
        th.heatsink.current_heat_gj = cap * 0.95
        sim.step()
        levels = [e.data["level"] for e in _events(sim, SimulationEventType.HEAT_WARNING)
                  if e.ship_id == "a"]
        assert levels == list(HEAT_WARNING_LEVELS) + [0.9]

    def test_full_burn_retracted_reaches_100_then_holds(self, fleet):
        """Full burn = radiator capacity: retracted fills 320 GJ in ~410 s, then
        the forced-out radiators hold the drive's heat level."""
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=2)
        a = _ship(sim, "a", "frigate", "alpha", fleet, 0, 1)
        _ship(sim, "b", "frigate", "beta", fleet, 5000, -1)
        a.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                      duration=1e9, throttle=1.0, direction=Vector3D(0, 1, 0))
        t_full = None
        for i in range(700):
            sim.step()
            if t_full is None and a.heat_fraction >= 1.0:
                t_full = i + 1
        assert 380 <= t_full <= 440
        assert a.radiators_extended
        assert 1.0 <= a.heat_fraction < 1.05  # held, not melting down
        assert not a.is_destroyed


# =============================================================================
# Radiator automation and the one radiator-hit model
# =============================================================================

class TestRadiatorAutomation:
    def test_auto_mode_extends_at_threshold_and_retracts_under_torpedo_fire(self, fleet):
        from tests.test_doctrine_accuracy import _ballistic_trident
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=3)
        d = _ship(sim, "d", "destroyer", "alpha", fleet, 0, 1)
        _ship(sim, "e", "corvette", "beta", fleet, 2000, -1)
        th = d.thermal_system
        assert sim.inject_command("d", {"type": "set_radiators", "mode": "auto",
                                        "auto_extend_pct": 50})
        assert th.radiator_mode == "auto" and th.auto_extend_pct == 50
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * 0.40
        sim.step()
        assert not d.radiators_extended
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * 0.55
        sim.step()
        assert d.radiators_extended
        # A torpedo inbound (ETA < 75 s) buttons them up...
        tf = _ballistic_trident(400_000, 10_000, "d")  # ETA 40 s
        tf.source_ship_id = "e"
        sim.torpedoes.append(tf)
        sim.step()
        assert not d.radiators_extended
        # ...unless the ladder forces them out.
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * 1.01
        sim.step()
        assert d.radiators_extended
        sim.torpedoes.clear()
        # Hysteresis: retract only once heat is well below the extend line.
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * (0.50 - AUTO_RADIATOR_HYSTERESIS_PCT / 100 + 0.05)
        sim.step()
        assert d.radiators_extended
        th.heatsink.current_heat_gj = th.heatsink.capacity_gj * (0.50 - AUTO_RADIATOR_HYSTERESIS_PCT / 100 - 0.05)
        sim.step()
        assert not d.radiators_extended
        # A manual order hands control back.
        sim.inject_command("d", {"type": "set_radiators", "extend": True})
        assert th.radiator_mode == "manual" and d.radiators_extended

    def test_tool_arguments_normalise_the_same_for_every_client(self):
        from src.llm.tools import radiator_command_from_args
        assert radiator_command_from_args({"mode": "auto"}) == {
            "type": "set_radiators", "mode": "auto", "auto_extend_pct": 65.0}
        assert radiator_command_from_args({"mode": "AUTO", "auto_extend_pct": 140}) == {
            "type": "set_radiators", "mode": "auto", "auto_extend_pct": 100.0}
        assert radiator_command_from_args({"extend": True}) == {
            "type": "set_radiators", "extend": True}
        assert radiator_command_from_args({"extend": "false"}) == {
            "type": "set_radiators", "extend": False}

    def test_mcp_auto_mode_goes_through_the_engine(self, fleet):
        from src.llm.mcp_controller import apply_mcp_commands_to_simulation
        from src.llm.mcp_state import MCPCommand, MCPCommandType
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
        a = _ship(sim, "alpha_1", "frigate", "alpha", fleet, 0, 1)
        _ship(sim, "beta_1", "frigate", "beta", fleet, 900, -1)
        results = apply_mcp_commands_to_simulation([MCPCommand(
            command_type=MCPCommandType.SET_RADIATORS, ship_id="alpha_1",
            parameters={"mode": "auto", "auto_extend_pct": 30}, timestamp=0.0)], sim, "alpha")
        assert results["applied"][0]["mode"] == "auto"
        assert a.thermal_system.radiator_mode == "auto"
        assert a.thermal_system.auto_extend_pct == 30


class TestRadiatorHits:
    def test_one_model_nose_spares_retracted_panels(self):
        assert radiator_hit_chance(HitLocation.NOSE, extended=False) == 0.0
        assert radiator_hit_chance(HitLocation.NOSE, extended=True) == pytest.approx(0.10)
        assert radiator_hit_chance(HitLocation.LATERAL, extended=True) == pytest.approx(0.20)
        assert radiator_hit_chance(HitLocation.TAIL, extended=False) == pytest.approx(0.05)
        from src.thermal import RadiatorArray
        assert not hasattr(RadiatorArray, "get_hit_probability")

    def test_torpedo_impacts_damage_radiators(self, fleet):
        from tests.test_doctrine_accuracy import _ballistic_trident

        class AlwaysHit:
            def random(self):
                return 0.0

            def choice(self, seq):
                return seq[0]

        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=3)
        t = _ship(sim, "t", "frigate", "alpha", fleet, 0, 1)
        _ship(sim, "s", "corvette", "beta", fleet, 900, -1)
        t.thermal_system.radiators.extend_all()
        sim.combat_resolver.radiator_resolver.rng = AlwaysHit()
        tf = _ballistic_trident(1_000, 10_000, "t")  # ~12.5 GJ penetrator
        tf.source_ship_id = "s"
        sim._resolve_torpedo_hit(tf, t)
        hits = [e for e in _events(sim, SimulationEventType.RADIATOR_DAMAGED) if e.ship_id == "t"]
        assert hits and hits[0].data["destroyed"]  # a torpedo wrecks a panel
        assert t.thermal_system.radiators.health_percent == pytest.approx(75.0)


# =============================================================================
# Power: systems bus + battery path (decision 7)
# =============================================================================

class TestSystemsBus:
    def test_battery_read_from_thermal_block(self, fleet):
        """The fleet data keeps the battery under thermal.battery; it used to be
        read from propulsion.battery, so editing it did nothing."""
        mutated = copy.deepcopy(fleet)
        mutated["ships"]["destroyer"]["thermal"]["battery"]["capacity_gj"] = 80.0
        mutated["ships"]["destroyer"]["thermal"]["battery"]["recharge_rate_gj_s"] = 0.05
        ship = create_ship_from_fleet_data("d", "destroyer", "alpha", mutated)
        assert ship.power_system.battery.capacity_gj == 80.0
        assert ship.power_system.battery.max_recharge_rate_gw == 0.05
        # Legacy location still honoured
        legacy = {"propulsion": {"battery": {"capacity_gj": 42.0}}}
        assert PowerSystem.from_ship_data(legacy).battery.capacity_gj == 42.0

    def test_healthy_bus_never_starves_even_at_full_throttle(self, fleet):
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
        a = _ship(sim, "a", "dreadnought", "alpha", fleet, 0, 1)
        _ship(sim, "b", "destroyer", "beta", fleet, 5000, -1)
        a.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                      duration=1e9, throttle=1.0, direction=Vector3D(1, 0, 0))
        ps = a.power_system
        for cap in ps.weapon_capacitors.values():
            cap.current_charge_mj = 0.0
        for _ in range(30):
            sim.step()
        assert all(c.is_charged for c in ps.weapon_capacitors.values())
        assert ps.battery.charge_percent == pytest.approx(100.0)

    def test_reactor_damage_drains_battery_then_slows_charging(self, fleet):
        ship = create_ship_from_fleet_data("d", "destroyer", "alpha", fleet)
        ps = ship.power_system
        spinal = ps.weapon_capacitors["weapon_0"]
        ps.set_bus_scale(0.5)
        spinal.current_charge_mj = 0.0
        ps.update(1.0)
        # Battery covers the deficit: full charge rate
        assert spinal.current_charge_mj == pytest.approx(spinal.charge_rate_mw)
        assert ps.battery.current_charge_gj < ps.battery.capacity_gj
        # Battery dry: the bus alone, at half power, is all there is
        ps.battery.current_charge_gj = 0.0
        spinal.current_charge_mj = 0.0
        ps.update(1.0)
        assert spinal.current_charge_mj < spinal.charge_rate_mw

    def test_no_cooldown_multiplier(self, fleet):
        """Reactor damage acts through the bus now, not a cooldown multiplier."""
        from src.simulation import ShipCombatState
        assert not hasattr(ShipCombatState, "get_weapon_cooldown_multiplier")
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
        a = _ship(sim, "a", "destroyer", "alpha", fleet, 0, 1)
        _ship(sim, "b", "destroyer", "beta", fleet, 5000, -1)
        for r in a._get_modules_by_type("reactor"):
            r.health_percent = 30.0
        ws = a.weapons["weapon_0"]
        ws.fire()
        remaining = ws.cooldown_remaining
        sim.step()
        assert ws.cooldown_remaining == pytest.approx(max(0.0, remaining - 1.0))
        assert a.power_system.bus_scale < 1.0


# =============================================================================
# Telemetry and determinism
# =============================================================================

class TestTelemetry:
    def test_trace_fields(self, fleet):
        from src.llm.battle_runner import _thermal_trace_fields
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
        a = _ship(sim, "a", "frigate", "alpha", fleet, 0, 1)
        _ship(sim, "b", "frigate", "beta", fleet, 5000, -1)
        a.thermal_system.heatsink.current_heat_gj = 160.0
        sim.step()
        f = _thermal_trace_fields(a)
        assert f["heat"] == pytest.approx(0.5, abs=0.001) and f["rad"] == 0
        assert f["rad_out"] == 0.0
        sim.inject_command("a", {"type": "set_radiators", "extend": True})
        sim.step()
        f = _thermal_trace_fields(a)
        assert f["rad"] == 1
        assert f["rad_out"] == pytest.approx(0.780, abs=0.001)
        assert f["heat_rate"] < 0  # cooling
        for key in ("heat", "rad", "heat_rate", "rad_out"):
            assert isinstance(f[key], (int, float))

    def test_recorder_passes_heat_fields_through(self):
        from src.llm.battle_recorder import BattleRecorder, _TRACE_OPTIONAL_SHIP_KEYS
        assert {"heat", "rad", "heat_rate", "rad_out"} <= set(_TRACE_OPTIONAL_SHIP_KEYS)

    def test_outlook_times(self, fleet):
        th = ThermalSystem.from_ship_data(fleet["ships"]["destroyer"])
        th.set_drive_throttle(1.0)
        for _ in range(60):
            th.update(1.0)
        out = th.outlook()
        # 780 MW continuous load retracted: 320 GJ in ~410 s from empty
        assert out["load_mw"] == pytest.approx(781.0, rel=0.01)
        assert out["time_to_100_retracted_s"] == pytest.approx(
            (320.0 - out["heat_gj"]) / 0.781, rel=0.02)
        assert out["time_to_meltdown_retracted_s"] > out["time_to_100_retracted_s"]
        # Extended, the drive's heat is rejected: only the 1 MW housekeeping remains
        assert out["time_to_100_extended_s"] > 10 * out["time_to_100_retracted_s"]


class TestDeterminism:
    def _run(self, fleet):
        from src.llm.heuristic_captain import HeuristicCaptain  # noqa: F401 (import side effects)
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=11)
        a = _ship(sim, "a", "dreadnought", "alpha", fleet, 0, 1)
        b = _ship(sim, "b", "cruiser_torpedo", "beta", fleet, 600, -1)
        a.current_maneuver = Maneuver(maneuver_type=ManeuverType.INTERCEPT, start_time=0.0,
                                      duration=1e9, throttle=1.0, target_id="b")
        b.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE, start_time=0.0,
                                      duration=1e9, throttle=1.0)
        sim.inject_command("a", {"type": "set_radiators", "mode": "auto", "auto_extend_pct": 40})
        for i in range(240):
            if i % 20 == 0:
                sim.inject_command("b", {"type": "launch_torpedo", "target_id": "a"})
            sim.step()
        return ([(round(s.heat_fraction, 9), s.radiators_extended, s.is_destroyed)
                 for s in (a, b)],
                [(e.event_type.name, e.ship_id, e.timestamp) for e in sim.events
                 if e.event_type.name in ("HEAT_WARNING", "RADIATOR_EXTENDED",
                                          "RADIATOR_RETRACTED", "RADIATOR_DAMAGED",
                                          "REACTOR_MELTDOWN")])

    def test_identical_seeds_identical_heat(self, fleet):
        assert self._run(fleet) == self._run(fleet)

    def test_prediction_shadow_leaves_heat_and_rng_alone(self, fleet):
        from src.trajectory_prediction import PredictionSimulation
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=4)
        a = _ship(sim, "a", "destroyer", "alpha", fleet, 0, 1)
        _ship(sim, "b", "destroyer", "beta", fleet, 600, -1)
        a.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                      duration=1e9, throttle=1.0, direction=Vector3D(1, 0, 0))
        for _ in range(5):
            sim.step()
        heat = a.thermal_system.heatsink.current_heat_gj
        rng_state = sim.rng.getstate()
        shadow = PredictionSimulation(sim)
        for _ in range(30):
            shadow.advance(1.0)
        assert a.thermal_system.heatsink.current_heat_gj == heat
        assert sim.rng.getstate() == rng_state
