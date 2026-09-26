"""TI salvo timing, per-shot fire control, replay fields and captain forecasts."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.combat import Weapon
from src.firecontrol import WeaponsCommand, WeaponsOrder, calculate_hit_probability
from src.llm.battle_recorder import BattleRecorder
from src.llm.battle_runner import BattleConfig, LLMBattleRunner
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.fleet_draft import _mount_throughput_gj_per_s, ship_point_cost
from src.llm.prompts import (
    SHIP_CAPABILITIES_DESTROYER,
    format_battlefield_overview,
    format_weapon_groups_for_prompt,
)
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation, SimulationEventType, WeaponState, create_ship_from_fleet_data,
)


# Read from TIMagneticGunTemplate.json (TI installation, UTF-8 BOM).
TI_TIMING = {
    "coilgun_mk3": (5, 10, 20),
    "heavy_coilgun_mk3": (5, 10, 20),
    "light_coilgun_mk3": (5, 10, 20),
    "spinal_coiler_mk3": (5, 12, 24),
    "heavy_siege_coiler_mk3": (3, 18, 24),
}


@pytest.fixture(scope="module")
def fleet():
    return json.loads(Path("data/fleet_ships.json").read_text())


def engagement(fleet, weapon_type="coilgun_mk3", seed=3):
    data = copy.deepcopy(fleet)
    data["ships"]["frigate"]["weapons"] = [{"type": weapon_type}]
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    shooter = create_ship_from_fleet_data(
        "alpha", "frigate", "alpha", data,
        position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0),
    )
    target = create_ship_from_fleet_data(
        "beta", "dreadnought", "beta", data,
        position=Vector3D(100_000, 0, 0), forward=Vector3D(-1, 0, 0),
    )
    shooter.point_defense = []
    target.point_defense = []
    sim.add_ship(shooter)
    sim.add_ship(target)
    shooter.weapons_orders["weapon_0"] = WeaponsOrder(
        weapon_slot="weapon_0", command=WeaponsCommand.FIRE_IMMEDIATE,
        target_id="beta",
    )
    return sim, shooter, target, shooter.weapons["weapon_0"]


def shot_events(sim):
    return [e for e in sim.events
            if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED]


@pytest.mark.parametrize("weapon_type", TI_TIMING)
def test_cadence_magazine_energy_and_throughput(fleet, weapon_type):
    """Real powered guns complete two cycles, one projectile/round/charge each."""
    n, intra, cooldown = TI_TIMING[weapon_type]
    spec = fleet["weapon_types"][weapon_type]
    assert (spec["salvo_size"], spec["intra_salvo_cooldown_s"],
            spec["cooldown_s"]) == (n, intra, cooldown)
    sim, ship, _, ws = engagement(fleet, weapon_type)
    assert (ws.weapon.salvo_size, ws.weapon.intra_salvo_cooldown_s,
            ws.weapon.cooldown_s) == (n, intra, cooldown)
    capacitor = ship.power_system.weapon_capacitors["weapon_0"]
    assert capacitor.charge_rate_mw * intra == pytest.approx(capacitor.capacity_mj)
    ship.power_system.fire_weapon = Mock(wraps=ship.power_system.fire_weapon)
    ship.thermal_system.add_heat = Mock(wraps=ship.thermal_system.add_heat)
    magazine = ws.ammo_remaining
    cycle = cooldown + (n - 1) * intra
    for _ in range(2 * cycle):
        sim.step()
    shots = shot_events(sim)
    assert [e.timestamp for e in shots] == [
        start + index * intra for start in (0, cycle) for index in range(n)
    ]
    assert ws.ammo_remaining == magazine - 2 * n
    assert ship.shots_fired == ws.shots_fired == 2 * n
    assert len({e.data["projectile_id"] for e in shots}) == 2 * n
    assert ship.power_system.fire_weapon.call_count == 2 * n
    heat_calls = [c.args for c in ship.thermal_system.add_heat.call_args_list
                  if c.args[0] == "weapons"]
    assert len(heat_calls) == 2 * n
    assert sum(c[1] for c in heat_calls) == pytest.approx(
        2 * n * capacitor.capacity_mj / 1000 * (1 - capacitor.efficiency))
    # Pricing must equal observed sustained output for EVERY gun, including siege.
    measured = len(shots) * spec["kinetic_energy_gj"] / (2 * cycle)
    assert _mount_throughput_gj_per_s(weapon_type, fleet) == pytest.approx(measured)
    assert [e.data["salvo_index"] for e in shots] == list(range(n)) * 2
    assert all(e.data["salvo_size"] == n for e in shots)
    assert [e.data["barrel"] for e in shots] == (
        [i % 2 for i in range(2 * n)] if ws.weapon.is_turreted else [0] * (2 * n))


def test_single_shot_defaults_and_empty_magazine(fleet):
    spec = dict(fleet["weapon_types"]["coilgun_mk3"])
    spec.pop("salvo_size")
    spec.pop("intra_salvo_cooldown_s")
    ws = WeaponState(Weapon.from_json("legacy", spec), ammo_remaining=2)
    assert ws.fire()
    assert ws.cooldown_remaining == 20
    assert not ws.fire()
    assert ws.ammo_remaining == 1
    ws.update(20)
    assert ws.fire()
    ws.update(100)
    assert not ws.fire()
    assert ws.ammo_remaining == 0
    assert ws.salvo_index == 0


def test_magazine_can_run_out_mid_salvo(fleet):
    sim, _, _, ws = engagement(fleet)
    ws.ammo_remaining = 2
    for _ in range(70):
        sim.step()
    assert [e.timestamp for e in shot_events(sim)] == [0, 10]
    assert ws.ammo_remaining == 0
    assert ws.salvo_index == 2


@pytest.mark.parametrize("interruption", ["probability", "target", "arc", "hold", "power"])
def test_every_shot_is_gated_and_paused_salvo_resumes(fleet, monkeypatch, interruption):
    sim, ship, target, ws = engagement(fleet, "spinal_coiler_mk3")
    # Isolate fire control from flight/damage while retaining real aim and power.
    probability = [0.8]
    calls = []

    def solution(**kwargs):
        result = calculate_hit_probability(**kwargs)
        result.hit_probability = probability[0]
        calls.append(probability[0])
        return result

    monkeypatch.setattr("src.firecontrol.calculate_hit_probability", solution)
    order = ship.weapons_orders["weapon_0"]
    order.command = WeaponsCommand.FIRE_WHEN_OPTIMAL
    order.min_hit_probability = 0.5
    sim._process_weapons_orders(ship)
    assert len(shot_events(sim)) == 1
    ws.update(12)
    cap = ship.power_system.weapon_capacitors["weapon_0"]
    cap.current_charge_mj = cap.capacity_mj
    before_ammo = ws.ammo_remaining
    if interruption == "probability":
        probability[0] = 0.2
    elif interruption == "target":
        order.target_id = "lost"
    elif interruption == "arc":
        target.kinematic_state.position = Vector3D(0, 100_000, 0)
    elif interruption == "hold":
        order.command = WeaponsCommand.HOLD_FIRE
    else:
        cap.current_charge_mj = 0
    before_charge = cap.current_charge_mj
    sim.current_time = 12
    sim._process_weapons_orders(ship)
    ws.update(88)  # A long interruption does not refund or consume salvo shots.
    assert ws.salvo_index == 1
    assert ws.ammo_remaining == before_ammo
    assert cap.current_charge_mj == before_charge
    assert len(shot_events(sim)) == 1
    probability[0] = 0.8
    order.target_id = "beta"
    order.command = WeaponsCommand.FIRE_WHEN_OPTIMAL
    target.kinematic_state.position = Vector3D(100_000, 0, 0)
    cap.current_charge_mj = cap.capacity_mj
    sim.current_time = 100
    sim._process_weapons_orders(ship)
    sim._process_weapons_orders(ship)
    assert [e.timestamp for e in shot_events(sim)] == [0, 100]
    assert [e.data["salvo_index"] for e in shot_events(sim)] == [0, 1]
    assert ws.cooldown_remaining == 12
    assert ws.ammo_remaining == before_ammo - 1
    assert cap.current_charge_mj == 0
    if interruption == "probability":
        assert calls == [0.8, 0.2, 0.8]


def test_direct_fire_command_also_fires_one_shot(fleet):
    sim, ship, _, ws = engagement(fleet)
    command = {"type": "fire_at", "weapon_slot": "weapon_0", "target_id": "beta"}
    assert sim.inject_command(ship.ship_id, command)
    assert not sim.inject_command(ship.ship_id, command)
    assert len(sim.projectiles) == 1
    assert ws.salvo_index == 1
    assert ws.cooldown_remaining == 10
    assert shot_events(sim)[0].data["barrel"] == 0


def test_seeded_determinism_and_forecast_does_not_draw_rng(fleet):
    def run(forecast):
        sim, ship, target, _ = engagement(fleet, seed=11)
        captain = LLMCaptain(LLMCaptainConfig(name="Test", ship_name="Test"), client=None)
        for _ in range(150):
            if forecast:
                captain._build_enemy_info(ship, target, sim)
            sim.step()
        return [(e.timestamp, e.event_type, e.ship_id, e.target_id, e.data)
                for e in sim.events]

    assert run(False) == run(False) == run(True)


def test_recorded_shot_fields_survive_runner(fleet):
    sim, ship, target, _ = engagement(fleet)
    config = LLMCaptainConfig(name="Test", ship_name="Test")
    recorder = BattleRecorder()
    recorder.start_recording(BattleConfig(), config, config, ship, target)
    runner = LLMBattleRunner(BattleConfig(), config, config, client=None)
    runner.recorder = recorder
    sim._process_weapons_orders(ship)
    runner._handle_simulation_event(shot_events(sim)[0])
    recorded = next(e for e in recorder.events if e.event_type == "shot_fired")
    assert {k: recorded.data[k] for k in ("salvo_index", "salvo_size", "barrel")} == {
        "salvo_index": 0, "salvo_size": 5, "barrel": 0,
    }


def test_checkpoint_forecast_counts_phase_cooldown_and_ammo(fleet):
    sim, ship, target, ws = engagement(fleet)
    captain = LLMCaptain(LLMCaptainConfig(name="Test", ship_name="Test"), client=None)
    info = captain._build_enemy_info(ship, target, sim)
    forecast = info["weapon_hit_forecasts"][0]
    assert forecast["shots"] == 3  # 0, 10, 20; 30 belongs to the next checkpoint.
    assert forecast["expected_hits"] == pytest.approx(3 * forecast["hit_probability"])
    assert ws.salvo_index == ws.shots_fired == 0  # Pure forecast.
    text = format_battlefield_overview([info], [], fleet)
    assert "expected hits/checkpoint if firing now:" in text
    assert "3 shots/30s" in text
    ws.salvo_index = 4
    assert ws.shots_in_window(30) == 2  # Last shot now, then new salvo at 20.
    ws.cooldown_remaining = 15
    assert ws.shots_in_window(30) == 1
    ws.cooldown_remaining = 0
    ws.ammo_remaining = 1
    assert ws.shots_in_window(30) == 1
    ws.is_operational = False
    assert ws.shots_in_window(30) == 0


@pytest.mark.parametrize("weapon_type", TI_TIMING)
def test_doctrine_describes_each_guns_true_salvo(fleet, weapon_type):
    text = format_weapon_groups_for_prompt([{"type": weapon_type}], fleet["weapon_types"])
    n, intra, cooldown = TI_TIMING[weapon_type]
    claim = f"{n}-shot salvos: {intra}s between shots, then {cooldown}s cooldown"
    assert claim in text
    if weapon_type in ("spinal_coiler_mk3", "coilgun_mk3"):
        assert claim in SHIP_CAPABILITIES_DESTROYER


@pytest.mark.parametrize("ship_type", [
    "corvette", "frigate", "destroyer", "cruiser", "battlecruiser",
    "battleship", "dreadnought", "dreadnought_siege", "cruiser_torpedo",
])
def test_documented_costs_match_pricer(fleet, ship_type):
    text = Path("docs/draft_mode.md").read_text()
    assert f"| {ship_type} | {ship_point_cost(ship_type, fleet)} |" in text
