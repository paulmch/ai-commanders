"""
Drive interlock (src/drive_plume.py): the engine caps every ship's main-drive
throttle so its exhaust puts at most PLUME_INTERLOCK_MAX_WASH_GJ_PER_S on a
FRIENDLY ship - whoever commands it, evading or not. Enemies never limit it.
And crew state (plume radiation) as a first-class fact for every commander:
fleet net, admiral, captain weapon lines, MCP, and a runner that asks an
incapacitated crew nothing.
"""

import json
from pathlib import Path

import pytest

from src import battle_damage as bd
from src import drive_plume as dp
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation, Maneuver, ManeuverType, SimulationEventType, create_ship_from_fleet_data,
)

REPO = Path(__file__).resolve().parent.parent
FLEET = json.loads((REPO / "data" / "fleet_ships.json").read_text())


def _ship(sim, ship_id, faction, pos_km, ship_type="destroyer", forward=(1, 0, 0)):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET,
        position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(0, 0, 0), forward=Vector3D(*forward))
    sim.add_ship(ship)
    return ship


def _burn(ship, throttle=1.0):
    ship.current_maneuver = Maneuver(maneuver_type=ManeuverType.BURN, start_time=0.0,
                                     duration=1e9, throttle=throttle)


def _column(tail_faction="alpha", gap_km=10.0, seed=5):
    sim = CombatSimulation(seed=seed, time_step=1.0)
    lead = _ship(sim, "lead", "alpha", (0, 0, 0), "cruiser")
    tail = _ship(sim, "tail", tail_faction, (-gap_km, 0.2, 0))
    _ship(sim, "foe", "beta", (3000, 0, 0), forward=(-1, 0, 0))
    _burn(lead)
    return sim, lead, tail


def _events(sim, kind):
    return [e for e in sim.events if e.event_type == kind]


# ---------------------------------------------------------------------------
# The interlock
# ---------------------------------------------------------------------------

def test_throttle_is_held_for_a_friendly_astern():
    sim, lead, tail = _column()
    for _ in range(10):
        sim.step()
    assert 0.0 < lead.applied_throttle < 0.05                    # ordered 1.0
    washes = _events(sim, SimulationEventType.PLUME_WASH)
    assert washes and max(e.data["peak_heat_rate_gj_s"] for e in washes) <= \
        dp.PLUME_INTERLOCK_MAX_WASH_GJ_PER_S + 1e-6
    assert tail.thermal_system.heat_fraction < 0.03 and tail.crew_dose_sv < 0.05
    assert not _events(sim, SimulationEventType.MOUNT_DAMAGED)
    (enter,) = _events(sim, SimulationEventType.DRIVE_INTERLOCK)
    assert enter.ship_id == "lead" and enter.data["phase"] == "enter"
    assert enter.data["friendly_ship_id"] == "tail" and enter.data["ordered_throttle"] == 1.0
    assert enter.data["throttle_cap"] == pytest.approx(lead.applied_throttle, abs=0.01)
    assert enter.data["distance_km"] == pytest.approx(10.0, abs=0.1) and enter.data["evading"] is False
    lead.name, tail.name = "TIS Sovereign", "TIS Dart-2"
    line = dp.plume_status_lines(sim, lead)[0]
    assert line == ("DRIVE INTERLOCK: throttle held at 3 % of the ordered 100 %: "
                    "friendly TIS Dart-2 is 10 km astern in the jet")
    assert dp.interlock_stats(sim) == {"ship_seconds": 10.0, "evading_ship_seconds": 0.0, "episodes": 1}
    # The friendly clears the jet: full throttle again, an exit event with the total.
    tail.kinematic_state.position = Vector3D(-10_000, 30_000, 0)
    sim.step()
    assert lead.applied_throttle == 1.0 and lead.drive_interlock is None
    exit_ = _events(sim, SimulationEventType.DRIVE_INTERLOCK)[-1]
    assert exit_.data["phase"] == "exit" and exit_.data["duration_s"] == 10.0
    assert dp.plume_status_lines(sim, lead) == [] or "DRIVE INTERLOCK" not in dp.plume_status_lines(sim, lead)[0]


def test_an_enemy_astern_never_limits_the_drive():
    sim, lead, tail = _column(tail_faction="beta")
    for _ in range(10):
        sim.step()
    assert lead.applied_throttle == 1.0 and lead.drive_interlock is None
    assert not _events(sim, SimulationEventType.DRIVE_INTERLOCK)
    assert tail.thermal_system.heat_fraction > 0.3              # washed at the full rate
    assert _events(sim, SimulationEventType.PLUME_WASH)[0].data["friendly"] is False


def test_who_counts_as_a_friendly():
    sim, lead, tail = _column()
    assert "lead" in dp.interlock_caps(sim, 1.0)
    tail.is_surrendered = True                                   # still crewed: counts
    assert "lead" in dp.interlock_caps(sim, 1.0)
    tail.is_surrendered = False
    tail.is_dying = True                                         # a hulk takes nothing: does not
    assert dp.interlock_caps(sim, 1.0) == {}
    tail.is_dying = False
    tail.is_destroyed = True
    assert dp.interlock_caps(sim, 1.0) == {}
    tail.is_destroyed = False
    lead.current_maneuver = None                                 # nothing ordered: nothing to cap
    assert dp.interlock_caps(sim, 1.0) == {}


def test_the_cap_follows_the_wash_model_and_a_turning_jet():
    sim, lead, tail = _column(gap_km=30.0)
    cap, friendly, d_km, _ = dp.interlock_caps(sim, 1.0)["lead"]
    full = dp.plume_heat_rate_gj_s(dp.emitter_jet_power_w(lead, 1.0), d_km, 0.0)
    assert friendly == "tail" and cap == pytest.approx(dp.PLUME_INTERLOCK_MAX_WASH_GJ_PER_S / full, rel=0.02)
    # A friendly 20 deg off the stern is clear of a steady jet...
    import math
    a = math.radians(20.0)
    tail.kinematic_state.position = Vector3D(-10_000 * math.cos(a), -10_000 * math.sin(a), 0)
    assert dp.interlock_caps(sim, 1.0) == {}
    # ...but not of a jet that swept across it since the last tick,
    sim._plume_axes = {"lead": (math.cos(math.radians(40)), math.sin(math.radians(40)), 0.0)}
    assert "lead" in dp.interlock_caps(sim, 1.0)
    # nor of a ship still turning toward it (cone opened by rate x dt).
    sim._plume_axes = {}
    lead.angular_velocity_rad_s = math.radians(15.0)
    assert "lead" in dp.interlock_caps(sim, 1.0)


def test_always_on_also_while_evading_and_the_exemption_switch(monkeypatch):
    assert dp.PLUME_INTERLOCK_EXEMPT_TORPEDO_EVASION is False
    sim, lead, tail = _column()
    sim._drive_interlock_caps = dp.interlock_caps(sim, 1.0)
    lead.execution_mode = "RUN"
    assert dp.apply_interlock(sim, lead, 1.0, 1.0) < 0.05
    assert lead.drive_interlock["evading"] is True
    assert dp.interlock_stats(sim)["evading_ship_seconds"] == 1.0
    monkeypatch.setattr(dp, "PLUME_INTERLOCK_EXEMPT_TORPEDO_EVASION", True)
    assert dp.apply_interlock(sim, lead, 1.0, 1.0) == 1.0
    lead.execution_mode = "INTERCEPT"
    assert dp.apply_interlock(sim, lead, 1.0, 1.0) < 0.05        # the exemption is for evasion only


def test_mirrored_fleets_are_held_identically_whatever_the_insertion_order():
    """Side symmetry (style of tests/test_side_symmetry.py): two mirrored
    columns; the caps come from one start-of-tick snapshot, so swapping the
    order in which the fleets are added changes nothing."""
    def run(order):
        sim = CombatSimulation(seed=3, time_step=1.0)
        specs = {
            "alpha_1": ("alpha", (-200, 0, 0), "cruiser", (1, 0, 0)),
            "alpha_2": ("alpha", (-212, 0.3, 0), "destroyer", (1, 0, 0)),
            "beta_1": ("beta", (200, 0, 0), "cruiser", (-1, 0, 0)),
            "beta_2": ("beta", (212, -0.3, 0), "destroyer", (-1, 0, 0)),
        }
        for sid in order:
            faction, pos, hull, fwd = specs[sid]
            _ship(sim, sid, faction, pos, hull, fwd)
        for sid in specs:
            _burn(sim.ships[sid])
        track = []
        for _ in range(25):
            sim.step()
            track.append({sid: (round(s.applied_throttle, 9), round(s.position.x, 3), round(s.position.y, 3),
                                round(s.thermal_system.heat_fraction, 9))
                          for sid, s in sim.ships.items()})
        return track
    first = run(("alpha_1", "alpha_2", "beta_1", "beta_2"))
    swapped = run(("beta_2", "beta_1", "alpha_2", "alpha_1"))
    assert first == swapped
    for frame in first:
        a, b = frame["alpha_1"], frame["beta_1"]
        assert a[0] == b[0] and a[1] == pytest.approx(-b[1]) and a[3] == b[3]    # mirror images
        fa, fb = frame["alpha_2"], frame["beta_2"]
        assert fa[0] == fb[0] and fa[1] == pytest.approx(-fb[1])
    assert first[0]["alpha_1"][0] < 0.05 and first[0]["alpha_2"][0] == 1.0       # lead held, follower free


def test_prediction_shadow_predicts_the_held_throttle_and_stays_silent():
    from src.trajectory_prediction import predict_plans
    sim, lead, tail = _column()
    sim.step()
    events, rng = len(sim.events), sim.rng.getstate()
    plan = predict_plans(sim, 30.0)["lead"]
    assert len(sim.events) == events and sim.rng.getstate() == rng
    start = lead.position
    while sim.current_time < plan["t_cp"] - 1e-9:
        sim.step()
    end = plan["path"][-1]
    moved = lead.position.distance_to(start)
    assert moved < 300.0                                         # held: ~0.5 m/s^2, not 15
    assert Vector3D(*end).distance_to(lead.position) < 5.0


# ---------------------------------------------------------------------------
# Seeded battle: the column demo, interlocked
# ---------------------------------------------------------------------------

def test_seeded_column_battle_with_legacy_captains_is_no_longer_cooked(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_test_battle", REPO / "scripts" / "generate_test_battle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, path = module.drive_interlock_battle(recording_dir=str(tmp_path), doctrine="legacy")
    rec = json.loads(Path(path).read_text())
    held = [e for e in rec["events"] if e["event_type"] == "drive_interlock"]
    assert held and held[0]["ship_id"] == "alpha_1" and held[0]["data"]["phase"] == "enter"
    assert held[0]["data"]["friendly_ship_id"] == "alpha_2"
    assert any(e["data"]["phase"] == "exit" and e["ship_id"] == "alpha_1" for e in held)
    friendly = [e["data"] for e in rec["events"] if e["event_type"] == "plume_wash" and e["data"]["friendly"]]
    assert max([d["peak_heat_rate_gj_s"] for d in friendly] or [0.0]) <= \
        dp.PLUME_INTERLOCK_MAX_WASH_GJ_PER_S + 0.01
    assert max([d.get("crew_dose_sv", 0.0) for d in friendly] or [0.0]) < 0.3
    assert not [e for e in rec["events"] if e["event_type"] == "crew_radiation"]
    assert not [e for e in rec["events"] if e["event_type"] == "mount_damaged"
                and e["data"].get("cause") == "plume_wash"]


# ---------------------------------------------------------------------------
# Crew state for every commander
# ---------------------------------------------------------------------------

def test_weapon_status_lines_show_the_combined_cycle_factor():
    sim, lead, tail = _column()
    assert bd.damaged_fitting_lines(tail) == []
    dp._irradiate(sim, tail, 2.5, "lead")
    lines = bd.damaged_fitting_lines(tail)
    assert lines and all(l.endswith("crew impaired, cycle x1.5") for l in lines)
    tail.weapons["weapon_1"].mount_health = 50.0
    line = next(l for l in bd.damaged_fitting_lines(tail) if "[weapon_1]" in l)
    assert line.endswith("DAMAGED 50 %, crew impaired, cycle x3.0")
    # ...and the rate-of-fire forecast uses that real cycle.
    gun = tail.weapons["weapon_1"]
    gun.cooldown_remaining = 0.0
    slow = gun.shots_in_window(120.0)
    gun.crew_cycle_factor, gun.mount_health = 1.0, 100.0
    assert slow < gun.shots_in_window(120.0)


def test_mcp_state_and_command_rejection():
    from src.llm.mcp_controller import (
        MCPCommand, MCPCommandType, _crew_state, apply_mcp_commands_to_simulation)
    sim, lead, tail = _column()
    dp._irradiate(sim, tail, 9.0, "lead")
    assert _crew_state(tail) == "incapacitated" and _crew_state(lead) == "fit"
    results = apply_mcp_commands_to_simulation([
        MCPCommand(MCPCommandType.SET_MANEUVER, ship_id="tail",
                   parameters={"maneuver_type": "INTERCEPT", "target_id": "foe", "throttle": 1.0}),
        MCPCommand(MCPCommandType.SET_MANEUVER, ship_id="lead",
                   parameters={"maneuver_type": "INTERCEPT", "target_id": "foe", "throttle": 1.0}),
    ], sim, "alpha")
    assert len(results["applied"]) == 1 and len(results["errors"]) == 1
    assert "Crew incapacitated by plume radiation (9.0 Sv)" in results["errors"][0]["error"]
    assert "cannot execute orders" in results["errors"][0]["error"]
    assert tail.current_maneuver is None


def test_admiral_sees_crew_state_in_now_block_fleet_table_and_threat_table():
    from unittest.mock import Mock
    from src.llm.admiral import AdmiralConfig, LLMAdmiral
    from src.llm.captain import LLMCaptain, LLMCaptainConfig
    from src.llm.prompts import build_admiral_prompt
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=42)
    _ship(sim, "a1", "alpha", (0, 0, 0))
    out = _ship(sim, "a2", "alpha", (0, 20, 0))
    hurt = _ship(sim, "a3", "alpha", (0, 40, 0))
    foe = _ship(sim, "b1", "beta", (600, 0, 0), forward=(-1, 0, 0))
    foe2 = _ship(sim, "b2", "beta", (600, 30, 0), forward=(-1, 0, 0))
    dp._irradiate(sim, out, 8.4, "a1")
    dp._irradiate(sim, hurt, 2.6, "a1")
    dp._irradiate(sim, foe, 9.0, "b2")
    dp._irradiate(sim, foe2, 3.0, "b1")
    admiral = LLMAdmiral(AdmiralConfig(name="A", model="m"), faction="alpha", client=Mock(), fleet_data=FLEET)
    captains = []
    for sid in ("a1", "a2", "a3"):
        c = LLMCaptain(LLMCaptainConfig(name=f"C{sid}", ship_name=sid, ship_type="destroyer",
                                        fleet_data=FLEET), client=Mock())
        c.ship_id = sid
        captains.append(c)
    snap = admiral._build_snapshot(sim, captains)
    rows = {r.ship_id: r for r in snap.friendly_ships}
    assert (rows["a2"].crew_state, rows["a3"].crew_state, rows["a1"].crew_state) == \
        ("incapacitated", "impaired", "fit")
    text = build_admiral_prompt("A", "alpha", snap, snap, None, FLEET)
    now = text[text.index("!! NOW"):]
    assert "CREW INCAPACITATED by plume radiation (8.4 Sv) - CANNOT EXECUTE ORDERS (coasting, not firing)" in now
    assert "crew IMPAIRED by plume radiation (2.6 Sv): weapons cycle slower" in now
    assert "CANNOT EXECUTE ORDERS" in text[text.index("YOUR FLEET"):] if "YOUR FLEET" in text else True
    assert "NOT MANEUVERING, NOT FIRING (crew incapacitated by plume wash)" in text
    assert "crew impaired by plume wash (slower fire)" in text


def test_fleet_net_posts_crew_state_and_heavy_wash(tmp_path, monkeypatch):
    from tests.test_fleet_comms import Stub, make_runner
    from src.llm.fleet_comms import EngineDatalink
    monkeypatch.setattr(dp, "PLUME_INTERLOCK_ENABLED", False)    # let a friendly be washed
    runner = make_runner(Stub(), tmp_path, alpha=(("cruiser", "stub/a"), ("destroyer", "stub/b"),
                                                 ("destroyer", "stub/c")))
    runner.setup_fleet_battle(FLEET)
    sim, channel = runner.simulation, runner.fleet_communication
    link = EngineDatalink()
    link.prime(sim)
    lead, tail = sim.get_ship("alpha_1"), sim.get_ship("alpha_2")
    lead.kinematic_state.position = Vector3D(0, 0, 0)
    lead.kinematic_state.forward = Vector3D(1, 0, 0)
    tail.kinematic_state.position = Vector3D(-6000, 0, 0)
    tail.kinematic_state.forward = Vector3D(0, 1, 0)             # thin flank to the jet
    lead.applied_throttle = 1.0
    for _ in range(12):
        dp.resolve_plume_wash(sim, 1.0)
        sim.current_time += 1.0
    assert tail.crew_incapacitated
    ids = iter(range(100))
    out = link.collect(sim, channel, {"alpha", "beta"}, 2, sim.current_time, lambda s: f"m{next(ids)}")
    own = [m for m in out if m.faction == "alpha" and m.kind == "CREW_STATE"]
    seen = [m for m in out if m.faction == "beta" and m.kind == "CREW_STATE"]
    lead_name, tail_name = channel.ship_name("alpha_1"), channel.ship_name("alpha_2")
    assert [m.text.split(" (")[0] for m in own] == [
        f"{tail_name} CREW IMPAIRED by plume radiation", f"{tail_name} CREW INCAPACITATED by plume radiation"]
    assert own[1].text.endswith("it cannot fight, maneuver or execute orders")
    assert "alpha_2" not in own[0].recipient_ids and "alpha_3" in own[0].recipient_ids
    assert seen[-1].text == (f"{tail_name} has stopped maneuvering and firing after plume wash from "
                             f"{lead_name}: crew incapacitated")
    assert seen[0].text == f"{tail_name} took heavy plume wash from {lead_name}: crew impaired, slower fire"
    (wash,) = [m for m in out if m.kind == "PLUME_WASH"]
    assert wash.faction == "alpha" and wash.text.startswith(
        f"{tail_name} is taking heavy plume wash in {lead_name}'s exhaust (11 s, peak +")
    assert "crew dose 8." in wash.text
    assert wash.structured["subject_id"] == "alpha_2" and wash.structured["source_ship_id"] == "alpha_1"
    assert link.collect(sim, channel, {"alpha", "beta"}, 3, sim.current_time, lambda s: f"m{next(ids)}") == [] \
        or all(m.kind not in ("CREW_STATE", "PLUME_WASH") for m in
               link.collect(sim, channel, {"alpha", "beta"}, 4, sim.current_time, lambda s: "x"))


def test_runner_asks_an_incapacitated_crew_nothing_and_reports_the_order_not_executable(tmp_path):
    from tests.test_fleet_comms import Stub, make_runner
    stub = Stub()
    runner = make_runner(stub, tmp_path)
    runner.setup_fleet_battle(FLEET)
    sim = runner.simulation
    dp._irradiate(sim, sim.get_ship("alpha_2"), 9.0, "alpha_1")
    runner.run_battle(FLEET) if False else None
    entries = [(sid, cap, "alpha") for sid, cap in runner.alpha_captains.items()]
    decided = []
    runner._get_captain_decision_with_discussion = lambda ship_id, captain, faction: decided.append(ship_id) or []

    class Order:
        order_text = "INTERCEPT the nearest enemy."
        torpedo_salvo = None
        torpedo_target = None
        suggested_target = None
        target_ship_id = "alpha_2"
        maneuver = None
    orders = {"alpha_2": [Order()]}
    orders["alpha_1"] = [Order()]
    sim.get_ship("alpha_1").drive_interlock = {
        "throttle_cap": 0.2, "ordered_throttle": 1.0, "friendly_ship_id": "alpha_2",
        "distance_km": 14.0, "off_axis_deg": 1.0, "evading": False}
    commands = runner._run_captain_decisions(entries, orders)
    assert decided == ["alpha_1"] and commands["alpha_2"] == []
    # The ship the interlock is holding: its report says so, and it is not a deviation.
    from src.llm.prompts import _compliance_text
    held = [e["report"] for e in runner._filed_reports["alpha"] if e["report"]["ship_id"] == "alpha_1"][0]
    name = runner.fleet_communication.ship_name("alpha_2")
    want = f"DRIVE INTERLOCK: throttle held at 20 % of the ordered 100 %: friendly"
    assert held["drive_interlock"].startswith(want) and held["drive_interlock"].endswith("is 14 km astern in the jet")
    assert held["compliance"]["interlock"] == held["drive_interlock"]
    text, followed = _compliance_text(dict(held["compliance"], followed=True, deviations=[]), None)
    assert "DRIVE INTERLOCK: throttle held at 20 % of the ordered 100 %" in text and followed is True
    filed = [e["report"] for e in runner._filed_reports.get("alpha", []) if e["report"]["ship_id"] == "alpha_2"]
    assert filed and filed[0]["status"] == "UNABLE" and filed[0]["implicit"] is True
    assert filed[0]["reason"].startswith("crew incapacitated by plume radiation (9.0 Sv): the ship cannot execute orders")
    comp = filed[0]["compliance"]
    assert comp["followed"] is False and comp["executable"] is False
    assert comp["deviations"][0] == "not executable: crew incapacitated by plume radiation"


def test_captain_sees_an_enemy_crew_out_of_action_in_its_contact_line():
    from unittest.mock import Mock
    from src.llm.captain import LLMCaptain, LLMCaptainConfig
    from src.llm.prompts import format_enemy_contacts
    sim, lead, tail = _column()
    foe = sim.ships["foe"]
    captain = LLMCaptain(LLMCaptainConfig(name="C", ship_name="lead", ship_type="cruiser",
                                          fleet_data=FLEET), client=Mock())
    captain.ship_id = "lead"
    assert "crew_state" not in captain._build_enemy_info(lead, foe, sim)
    dp._irradiate(sim, foe, 2.5, "lead")
    info = captain._build_enemy_info(lead, foe, sim)
    assert info["crew_state"] == "impaired"
    assert "crew impaired by plume wash (slower fire)" in format_enemy_contacts([info], FLEET)
    dp._irradiate(sim, foe, 6.0, "lead")
    info = captain._build_enemy_info(lead, foe, sim)
    assert "NOT MANEUVERING, NOT FIRING (crew incapacitated by plume wash)" in format_enemy_contacts([info], FLEET)
