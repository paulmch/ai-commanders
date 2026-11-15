"""
The draft-balance instrument (scripts/draft_tournament.py) and the price
derivation it tunes (src/llm/fleet_draft.py). No battle is fought here: the
tournament's planning, caching and scoring are pure functions of rosters.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.llm import fleet_draft
from src.llm.fleet_draft import (
    AUTO_DRAFT_DOCTRINES, DEFAULT_MAX_SHIPS, SHIP_POINT_COSTS, auto_draft,
    build_ship_point_costs, ship_point_cost,
)

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def tourney():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "draft_tournament", ROOT / "scripts" / "draft_tournament.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.calib.load_engine(ROOT)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return mod


@pytest.fixture(scope="module")
def fleet_data():
    return json.loads((ROOT / "data" / "fleet_ships.json").read_text())


def _args(tourney, **kw):
    base = dict(archetypes=list(tourney.ARCHETYPES), rows=None, budget=[150],
                seeds=1, seed_start=1, max_ships=8, distance=500.0,
                time_limit=600.0, max_checkpoints=20, battle_timeout=60)
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# auto_draft: named doctrines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_named_doctrine_is_the_seeded_doctrine(seed):
    """auto_draft(doctrine=name) drafts exactly what the seed that picks that
    doctrine drafts - naming a doctrine must not change the fleet."""
    import random
    names = list(AUTO_DRAFT_DOCTRINES)
    name = names[random.Random(seed).randrange(len(names))]
    seeded = auto_draft("alpha", 150, DEFAULT_MAX_SHIPS, seed=seed)
    named = auto_draft("alpha", 150, DEFAULT_MAX_SHIPS, seed=0, doctrine=name)
    assert [s.ship_type for s in seeded.ships] == [s.ship_type for s in named.ships]
    assert seeded.points_spent == named.points_spent


def test_unknown_doctrine_is_rejected():
    with pytest.raises(ValueError, match="unknown auto-draft doctrine"):
        auto_draft("alpha", 150, doctrine="deathball")


@pytest.mark.parametrize("budget", [100, 150, 200, 250])
@pytest.mark.parametrize("doctrine", list(AUTO_DRAFT_DOCTRINES))
def test_auto_draft_doctrines_fill_the_budget(doctrine, budget):
    """Every doctrine fields a legal fleet that leaves no hull of its own
    unbought: what is left is less than its cheapest hull, or the cap binds."""
    draft = auto_draft("alpha", budget, DEFAULT_MAX_SHIPS, doctrine=doctrine)
    types = [s.ship_type for s in draft.ships]
    assert types and len(types) <= DEFAULT_MAX_SHIPS
    assert draft.points_spent == sum(SHIP_POINT_COSTS[t] for t in types) <= budget
    left = budget - draft.points_spent
    cheapest = min(SHIP_POINT_COSTS[h] for h in AUTO_DRAFT_DOCTRINES[doctrine])
    assert left < cheapest or len(types) == DEFAULT_MAX_SHIPS, (types, left)
    # ... and never less than 85% of the budget
    assert draft.points_spent >= 0.85 * budget, (types, draft.points_spent)


# ---------------------------------------------------------------------------
# Tournament archetypes, plan and cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget", [150, 200])
def test_archetypes_are_legal_and_spend_the_budget(tourney, budget):
    costs = tourney.price_table({})
    assert costs == SHIP_POINT_COSTS
    rosters = {a: tourney.draft_roster(a, budget, costs, 8) for a in tourney.ARCHETYPES}
    for name, roster in rosters.items():
        spent = sum(costs[h] for h in roster)
        assert 0 < len(roster) <= 8, name
        assert 0.85 * budget <= spent <= budget, (name, roster, spent)
    # Seven DIFFERENT fleet concepts
    assert len({tuple(sorted(r)) for r in rosters.values()}) == len(rosters)


def test_price_overrides_move_rosters_and_are_restored(tourney):
    before = dict(SHIP_POINT_COSTS)
    dear = tourney.price_table({"TORPEDO_POINTS_PER_GJ": 2 * fleet_draft.TORPEDO_POINTS_PER_GJ})
    assert dear["cruiser_torpedo"] > before["cruiser_torpedo"]
    assert dear["destroyer"] == before["destroyer"]
    assert build_ship_point_costs() == before == SHIP_POINT_COSTS
    cheap_roster = tourney.draft_roster("torpedo", 150, before, 8)
    dear_roster = tourney.draft_roster("torpedo", 150, dear, 8)
    assert sum(dear[h] for h in dear_roster) <= 150
    assert cheap_roster != dear_roster
    assert SHIP_POINT_COSTS == before          # auto_draft's table was restored


def test_every_pairing_is_fought_in_both_side_assignments(tourney):
    args = _args(tourney, seeds=2)
    cells, tasks, fleets = tourney.plan(args, "src", tourney.price_table({}))
    n = len(tourney.ARCHETYPES)
    assert len(cells) == n * (n - 1) // 2 * 2 * 2
    by_pair = {}
    for c in cells:
        by_pair.setdefault((c["a"], c["b"], c["seed"]), []).append(c)
    for (a, b, _seed), pair in by_pair.items():
        assert sorted(c["a_side"] for c in pair) == ["alpha", "beta"]
        first, second = (tasks[c["key"]] for c in sorted(pair, key=lambda c: c["a_side"]))
        assert first["alpha_roster"] == second["beta_roster"] == fleets[150][a]
        assert first["beta_roster"] == second["alpha_roster"] == fleets[150][b]


def test_cache_key_follows_rosters_not_prices(tourney):
    """A reprice that leaves two rosters alone must reuse their battles; one
    that changes a roster must not."""
    args = _args(tourney, archetypes=["gunline", "battleline", "torpedo"])
    base = tourney.price_table({})
    dear = tourney.price_table({"TORPEDO_POINTS_PER_GJ": 2 * fleet_draft.TORPEDO_POINTS_PER_GJ})
    key = {}
    for label, costs in (("base", base), ("dear", dear)):
        cells, _tasks, _fleets = tourney.plan(args, "src", costs)
        key[label] = {(c["a"], c["b"], c["a_side"]): c["key"] for c in cells}
    assert key["base"][("gunline", "battleline", "alpha")] == key["dear"][("gunline", "battleline", "alpha")]
    assert key["base"][("gunline", "torpedo", "alpha")] != key["dear"][("gunline", "torpedo", "alpha")]
    other_engine, _, _ = tourney.plan(args, "other-src", base)
    assert {c["key"] for c in other_engine}.isdisjoint(key["base"].values())


def test_source_id_ignores_the_price_module(tourney, tmp_path):
    (tmp_path / "src" / "llm").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "simulation.py").write_text("x = 1\n")
    (tmp_path / "src" / "llm" / "fleet_draft.py").write_text("PRICE = 1\n")
    (tmp_path / "data" / "fleet_ships.json").write_text("{}")
    first = tourney.source_id(tmp_path)
    (tmp_path / "src" / "llm" / "fleet_draft.py").write_text("PRICE = 2\n")
    assert tourney.source_id(tmp_path) == first
    (tmp_path / "src" / "simulation.py").write_text("x = 2\n")
    assert tourney.source_id(tmp_path) != first


def test_score_matrix_is_antisymmetric_and_side_averaged(tourney, capsys):
    args = _args(tourney, archetypes=["gunline", "torpedo"])
    costs = tourney.price_table({})
    cells, tasks, fleets = tourney.plan(args, "src", costs)
    results = {}
    for c in cells:
        # alpha always wins: a pure side bias must average out to 50%
        results[c["key"]] = dict(tasks[c["key"]], winner="alpha", reason="Alpha tactical advantage (1 vs 0)",
                                 alpha_lost=[], beta_lost=list(tasks[c["key"]]["beta_roster"][:1]),
                                 events={})
    report = tourney.summarize(cells, results, fleets, costs, args)
    capsys.readouterr()
    b = report["budgets"][150]
    assert b["score"]["gunline"]["torpedo"] == b["score"]["torpedo"]["gunline"] == 0.5
    assert b["field"] == {"gunline": 0.5, "torpedo": 0.5}
    assert b["side_wins"] == {"alpha": 2}
    assert b["margin"]["gunline"]["torpedo"] == pytest.approx(-b["margin"]["torpedo"]["gunline"])


# ---------------------------------------------------------------------------
# The 2026-11-15 torpedo reprice (docs/draft_mode.md, pass-5 tournament)
# ---------------------------------------------------------------------------

TORPEDO_HULLS = ("corvette", "cruiser_torpedo")
# Points of gun hulls one LAUNCHED round destroyed in the tournament (massed
# salvos of 3+ boats, 150 and 200 pts): 2.2 when rounds are left over, 4.6
# against a battle line. The price of a round must sit inside that range.
MEASURED_POINTS_DESTROYED_PER_ROUND = (2.2, 4.6)


def _rounds(hull, fleet_data):
    default = fleet_data["weapon_types"]["torpedo_launcher"].get("magazine", 16)
    return sum(w.get("magazine", default) for w in fleet_data["ships"][hull]["weapons"]
               if w.get("type") == "torpedo_launcher")


@pytest.mark.parametrize("hull", TORPEDO_HULLS)
def test_a_torpedo_hull_costs_about_what_its_magazine_destroys(hull, fleet_data):
    lo, hi = MEASURED_POINTS_DESTROYED_PER_ROUND
    per_round = SHIP_POINT_COSTS[hull] / _rounds(hull, fleet_data)
    assert lo <= per_round <= hi, f"{hull}: {per_round:.2f} pts per stowed round"


def test_torpedo_dial_moves_only_torpedo_hulls(tourney, fleet_data):
    old = tourney.price_table({"TORPEDO_POINTS_PER_GJ": 0.040})
    for hull, cost in SHIP_POINT_COSTS.items():
        if hull in TORPEDO_HULLS:
            assert cost > old[hull]
        else:
            assert cost == old[hull], hull
    assert old["corvette"] == 18 and old["cruiser_torpedo"] == 65   # the pre-pass-5 table


def test_150_points_no_longer_buy_the_annihilating_swarm():
    """Six corvettes (48 rounds) annihilated every 150-pt gun fleet without
    loss, and 150 pts used to buy eight. They now buy five."""
    assert 150 // SHIP_POINT_COSTS["corvette"] <= 5
    # ... while 100 pts still buy the three boats a saturating salvo needs
    assert 100 // SHIP_POINT_COSTS["corvette"] >= 3


def test_every_hull_is_draftable_at_150_points():
    """No price so high that a hull cannot be fielded with an escort."""
    for hull, cost in SHIP_POINT_COSTS.items():
        assert cost + min(SHIP_POINT_COSTS.values()) <= 150, hull
    # The torpedo doctrine still opens with its torpedo cruiser at 150 pts
    types = [s.ship_type for s in auto_draft("alpha", 150, 8, doctrine="torpedo").ships]
    assert types[0] == "cruiser_torpedo" and "corvette" in types


def test_torpedo_price_sits_inside_its_plateau(tourney):
    """A small data edit must not flip the meta: 2% either side of the dial
    the corvette keeps its price and the torpedo fleets keep their torpedo hulls."""
    dial = fleet_draft.TORPEDO_POINTS_PER_GJ
    here = tourney.price_table({})
    for factor in (0.98, 1.02):
        near = tourney.price_table({"TORPEDO_POINTS_PER_GJ": dial * factor})
        assert near["corvette"] == here["corvette"]
        assert abs(near["cruiser_torpedo"] - here["cruiser_torpedo"]) <= 2
        for budget in (100, 150, 200):
            for archetype in ("torpedo", "swarm"):
                # (the frigate top-up may differ by a hull; the torpedo hulls may not)
                boats = [[h for h in tourney.draft_roster(archetype, budget, costs, 8)
                          if h in TORPEDO_HULLS] for costs in (near, here)]
                assert boats[0] == boats[1], (factor, budget, archetype)
