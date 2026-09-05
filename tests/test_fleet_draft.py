"""
Tests for fleet draft mode: point budget, selection validation, formation
placement, auto-draft fallback, and the heuristic captains that fly drafted
fleets. Everything runs offline (stub client or no client at all).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
from src.llm.fleet_draft import (
    FORMATION_MAX_OFFSET_KM,
    SHIP_POINT_COSTS,
    _mount_throughput_gj_per_s,
    ship_point_cost,
    torpedo_round_energy_gj,
    apply_formation,
    auto_draft,
    build_catalog_text,
    draft_to_fleet_definition,
    name_drafted_ships,
    run_admiral_draft,
    validate_selection,
    world_positions_km,
)
from src.llm.heuristic_captain import HEURISTIC_MODEL, HeuristicCaptain
from src.llm.prompts import CaptainPersonality


@pytest.fixture(scope="module")
def fleet_data():
    return load_fleet_data()


# ---------------------------------------------------------------------------
# Catalog and costs
# ---------------------------------------------------------------------------

def test_every_costed_hull_exists_in_fleet_data(fleet_data):
    for ship_type in SHIP_POINT_COSTS:
        assert ship_type in fleet_data["ships"], ship_type


def test_every_fleet_data_hull_is_costed(fleet_data):
    for ship_type in fleet_data["ships"]:
        assert ship_type in SHIP_POINT_COSTS, ship_type


def test_torpedo_cost_scales_with_magazine_depth(fleet_data):
    """Amount of torpedoes is a price driver: same hull, more rounds, more points."""
    import copy
    lean = copy.deepcopy(fleet_data)
    for weapon in lean["ships"]["cruiser_torpedo"]["weapons"]:
        if weapon.get("type") == "torpedo_launcher":
            weapon["magazine"] = 6  # half the stock 12-round tubes
    assert (ship_point_cost("cruiser_torpedo", lean)
            < ship_point_cost("cruiser_torpedo", fleet_data))


def test_torpedo_cost_scales_with_warhead_yield(fleet_data):
    """Explosive factor is a price driver: a heavier penetrator costs more."""
    import copy
    heavy = copy.deepcopy(fleet_data)
    heavy["weapon_types"]["torpedo_launcher"]["penetrator_mass_kg"] = 500.0
    assert (ship_point_cost("cruiser_torpedo", heavy)
            > ship_point_cost("cruiser_torpedo", fleet_data))
    # A pure-gun hull is untouched by torpedo yield.
    assert ship_point_cost("dreadnought", heavy) == ship_point_cost(
        "dreadnought", fleet_data)


def test_torpedo_round_energy_matches_engine_closure_floor(fleet_data):
    """18 GJ = 250kg at the 12 km/s floor the guidance law holds."""
    from src.torpedo import MIN_CLOSING_SPEED_KPS
    assert MIN_CLOSING_SPEED_KPS == 12.0
    assert torpedo_round_energy_gj(fleet_data) == pytest.approx(18.0, rel=1e-3)


def test_torpedo_armament_is_not_nearly_free(fleet_data):
    """
    The 2026-08-13 rebalance. cruiser_torpedo and cruiser are the SAME hull
    (identical mass, armour, acceleration) - the torpedo variant swaps guns
    for 4 launchers and 2 extra PD. It used to cost 4 points more for 48
    guided rounds; now the magazine is priced on what it delivers.
    """
    gap = SHIP_POINT_COSTS["cruiser_torpedo"] - SHIP_POINT_COSTS["cruiser"]
    assert gap >= 20, f"48 torpedoes should not cost {gap} points"
    # A fleet-killer should not undercut the hull it kills.
    assert SHIP_POINT_COSTS["cruiser_torpedo"] > SHIP_POINT_COSTS["dreadnought"]


def test_points_per_delivered_energy_is_within_one_order(fleet_data):
    """
    Old table: 5.0 pts per GJ/s for cruiser_torpedo vs 87.7 for the
    dreadnought - a 17.5x gap that made saturation the only rational draft.
    """
    def throughput(stype):
        spec = fleet_data["ships"][stype]
        total = 0.0
        for weapon in spec.get("weapons", []):
            wtype = weapon.get("type")
            if wtype == "torpedo_launcher":
                total += torpedo_round_energy_gj(fleet_data) / fleet_data[
                    "weapon_types"]["torpedo_launcher"]["cooldown_s"]
            elif wtype and wtype != "pd_laser":
                total += _mount_throughput_gj_per_s(wtype, fleet_data)
        return total

    torp = SHIP_POINT_COSTS["cruiser_torpedo"] / throughput("cruiser_torpedo")
    gun = SHIP_POINT_COSTS["dreadnought"] / throughput("dreadnought")
    assert gun / torp < 10.0, f"efficiency gap still {gun / torp:.1f}x"


def test_catalog_text_mentions_all_hulls(fleet_data):
    catalog = build_catalog_text(fleet_data)
    for ship_type in SHIP_POINT_COSTS:
        assert ship_type in catalog


def test_catalog_reports_sim_truth_torpedo_magazines(fleet_data):
    """Magazine counts come from the weapon entries the sim actually loads,
    and the torpedo cruiser's defining role text is not truncated away."""
    catalog = build_catalog_text(fleet_data)
    torp_line = next(l for l in catalog.splitlines() if "cruiser_torpedo" in l)
    assert "48 torpedoes" in torp_line
    assert "8 torpedoes per 30s decision" in torp_line
    corvette_line = next(l for l in catalog.splitlines() if "- corvette" in l)
    assert "8 torpedoes" in corvette_line


# ---------------------------------------------------------------------------
# Selection validation
# ---------------------------------------------------------------------------

def test_valid_selection():
    flat, spent, error = validate_selection(
        [{"ship_type": "destroyer", "count": 2},
         {"ship_type": "corvette", "count": 1}],
        budget=100, max_ships=8)
    assert error is None
    assert flat == ["destroyer", "destroyer", "corvette"]
    assert spent == 2 * SHIP_POINT_COSTS["destroyer"] + SHIP_POINT_COSTS["corvette"]


def test_selection_over_budget_rejected():
    flat, _, error = validate_selection(
        [{"ship_type": "dreadnought", "count": 2}], budget=100, max_ships=8)
    assert flat is None and "budget" in error.lower()


def test_selection_unknown_hull_rejected():
    flat, _, error = validate_selection(
        [{"ship_type": "star_destroyer", "count": 1}], budget=100, max_ships=8)
    assert flat is None and "star_destroyer" in error


def test_selection_too_many_ships_rejected():
    flat, _, error = validate_selection(
        [{"ship_type": "frigate", "count": 9}], budget=100, max_ships=8)
    assert flat is None and "cap" in error.lower()


def test_selection_empty_rejected():
    flat, _, error = validate_selection([], budget=100, max_ships=8)
    assert flat is None and error


# ---------------------------------------------------------------------------
# Naming and formation
# ---------------------------------------------------------------------------

def test_ship_naming_and_numbering():
    drafted = name_drafted_ships(["destroyer", "destroyer", "corvette"], "alpha")
    names = [s.ship_name for s in drafted]
    assert names == ["TIS Falchion-1", "TIS Falchion-2", "TIS Dart"]
    assert [s.ship_id for s in drafted] == ["alpha_1", "alpha_2", "alpha_3"]

    beta = name_drafted_ships(["cruiser"], "beta")
    assert beta[0].ship_name == "OCS Bastion"


def test_formation_placement_clamped_and_applied():
    drafted = name_drafted_ships(["destroyer", "corvette"], "alpha")
    notes = apply_formation(drafted, [
        {"ship_name": "TIS Falchion", "x_km": 20, "y_km": -30, "z_km": 5},
        {"ship_name": "TIS Dart", "x_km": 9999, "y_km": 0, "z_km": 0},
    ])
    assert drafted[0].offset_km == (20.0, -30.0, 5.0)
    assert drafted[1].offset_km[0] == FORMATION_MAX_OFFSET_KM
    assert any("clamped" in n for n in notes)


def test_formation_min_separation_nudge():
    drafted = name_drafted_ships(["frigate", "frigate"], "alpha")
    notes = apply_formation(drafted, [
        {"ship_name": "TIS Ward-1", "x_km": 0, "y_km": 0},
        {"ship_name": "TIS Ward-2", "x_km": 0, "y_km": 0.5},
    ])
    a, b = drafted[0].offset_km, drafted[1].offset_km
    d2 = sum((x - y) ** 2 for x, y in zip(a, b))
    assert d2 >= 2.0 ** 2
    assert any("nudged" in n for n in notes)


def test_unplaced_ships_get_default_slots():
    drafted = name_drafted_ships(["frigate", "frigate", "frigate"], "beta")
    apply_formation(drafted, [])
    ys = [s.offset_km[1] for s in drafted]
    assert ys == sorted(ys) and len(set(ys)) == 3


def test_world_positions_rotation():
    alpha = auto_draft("alpha", budget=30, max_ships=1, seed=0)
    beta = auto_draft("beta", budget=30, max_ships=1, seed=0)
    alpha.ships[0].offset_km = (10.0, 20.0, 3.0)
    beta.ships[0].offset_km = (10.0, 20.0, 3.0)
    pos_a = world_positions_km(alpha, 500.0)[alpha.ships[0].ship_id]
    pos_b = world_positions_km(beta, 500.0)[beta.ships[0].ship_id]
    # +x is "toward the enemy" for both sides; beta's frame is rotated 180deg
    assert pos_a == {"x": -240.0, "y": 20.0, "z": 3.0}
    assert pos_b == {"x": 240.0, "y": -20.0, "z": 3.0}


# ---------------------------------------------------------------------------
# Auto-draft
# ---------------------------------------------------------------------------

def test_auto_draft_respects_budget_and_is_deterministic():
    for seed in range(6):
        draft = auto_draft("alpha", budget=100, max_ships=8, seed=seed)
        spent = sum(SHIP_POINT_COSTS[s.ship_type] for s in draft.ships)
        assert spent == draft.points_spent <= 100
        assert 1 <= len(draft.ships) <= 8
        again = auto_draft("alpha", budget=100, max_ships=8, seed=seed)
        assert [s.ship_type for s in again.ships] == [s.ship_type for s in draft.ships]


# ---------------------------------------------------------------------------
# LLM draft with stub client
# ---------------------------------------------------------------------------

class DraftStubClient:
    """First selection is illegal (over budget), the retry is valid."""

    def __init__(self):
        self.select_calls = 0

    def decide_with_tools(self, messages, tools, model=None, temperature=None,
                          tool_choice="auto"):
        names = {t["function"]["name"] for t in tools}
        if "select_fleet" in names:
            self.select_calls += 1
            if self.select_calls == 1:
                return [ToolCall(id="s1", name="select_fleet", arguments={
                    "ships": [{"ship_type": "dreadnought", "count": 5}],
                    "rationale": "MORE DAKKA"})]
            return [ToolCall(id="s2", name="select_fleet", arguments={
                "ships": [{"ship_type": "cruiser_torpedo", "count": 2},
                          {"ship_type": "destroyer", "count": 2}],
                "rationale": "Saturation strike with gun escort."})]
        if "set_formation" in names:
            return [ToolCall(id="f1", name="set_formation", arguments={
                "placements": [
                    {"ship_name": "TIS Harpoon-1", "x_km": -40, "y_km": -25},
                    {"ship_name": "TIS Harpoon-2", "x_km": -40, "y_km": 25},
                    {"ship_name": "TIS Falchion-1", "x_km": 10, "y_km": -10},
                    {"ship_name": "TIS Falchion-2", "x_km": 10, "y_km": 10},
                ],
                "formation_name": "gun screen",
                "rationale": "Gunships screen the torpedo cruisers."})]
        return []


def test_run_admiral_draft_with_retry(fleet_data):
    stub = DraftStubClient()
    draft = run_admiral_draft(
        client=stub, model="stub-model", admiral_name="Admiral Stub",
        faction="alpha", fleet_data=fleet_data, budget=200, max_ships=8,
        verbose=False)
    assert stub.select_calls == 2  # retry loop engaged
    assert not draft.auto
    assert draft.points_spent == (2 * SHIP_POINT_COSTS["cruiser_torpedo"]
                                  + 2 * SHIP_POINT_COSTS["destroyer"])
    assert [s.ship_type for s in draft.ships] == [
        "cruiser_torpedo", "cruiser_torpedo", "destroyer", "destroyer"]
    assert draft.formation_name == "gun screen"
    # Torpedo cruisers held back (-x), gunships forward (+x)
    assert draft.ships[0].offset_km == (-40.0, -25.0, 0.0)
    assert draft.ships[2].offset_km == (10.0, -10.0, 0.0)


class UselessClient:
    def decide_with_tools(self, messages, tools, model=None, temperature=None,
                          tool_choice="auto"):
        return []


def test_draft_falls_back_to_auto(fleet_data):
    draft = run_admiral_draft(
        client=UselessClient(), model="stub", admiral_name="Admiral Mute",
        faction="beta", fleet_data=fleet_data, verbose=False, seed=3)
    assert draft.auto
    assert draft.ships


# ---------------------------------------------------------------------------
# Heuristic captains: full offline draft battle
# ---------------------------------------------------------------------------

def test_heuristic_draft_battle_end_to_end(tmp_path):
    """Auto-draft both sides, no admirals, no LLM anywhere - and the fleets
    must actually fight (ordnance in the air, decisions every checkpoint)."""
    fleet_data = load_fleet_data()
    alpha = auto_draft("alpha", budget=60, max_ships=3, seed=0)
    beta = auto_draft("beta", budget=60, max_ships=3, seed=1)

    fleet_config = BattleFleetConfig(
        battle_name="Draft Smoke Test",
        alpha_fleet=draft_to_fleet_definition(alpha, 400.0,
                                              captain_model=HEURISTIC_MODEL),
        beta_fleet=draft_to_fleet_definition(beta, 400.0,
                                             captain_model=HEURISTIC_MODEL),
        time_limit_s=240.0,
        initial_distance_km=400.0,
        personality_selection=False,
    )
    config = BattleConfig(
        initial_distance_km=400.0,
        time_limit_s=240.0,
        max_checkpoints=6,
        verbose=False,
        personality_selection=False,
        record_battle=True,
        recording_dir=str(tmp_path),
        record_sim_trace=False,
        seed=11,
    )
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(
        config=config, alpha_config=dummy, beta_config=dummy,
        client=None, fleet_config=fleet_config)

    result = runner.run_fleet_battle(fleet_data)

    # Every captain must be the heuristic, not an LLM stub
    for captain in list(runner.alpha_captains.values()) + list(runner.beta_captains.values()):
        assert isinstance(captain, HeuristicCaptain)

    assert result.checkpoints_used >= 1

    # Recording written to the temp dir, not the repo - and it must show the
    # fleets actually engaging (ordnance in the air).
    import json
    recordings = list(tmp_path.glob("*.json"))
    assert recordings, "no recording written"
    recording = json.loads(recordings[0].read_text())
    event_types = {e["event_type"] for e in recording.get("events", [])}
    assert event_types & {"shot_fired", "torpedo_launched"}, \
        f"no ordnance events in {event_types}"

    # Every heuristic captain produced decisions
    some_captain = next(iter(runner.alpha_captains.values()))
    assert some_captain.decision_count >= 1
