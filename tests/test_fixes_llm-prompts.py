"""
Regression tests for audited defects in src/llm/prompts.py and src/llm/victory.py.

Each test pins a claim the prompt makes to the code that actually implements it,
so the two cannot drift apart again silently.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from src.llm.prompts import (
    ADMIRAL_STATE_MARKER,
    CAPTAIN_DOCTRINE,
    PROJECTILE_PHYSICS_REFERENCE,
    build_admiral_messages,
    build_admiral_prompt,
    build_captain_messages,
    build_ship_capabilities_from_fleet,
    format_battlefield_overview,
    format_weapon_groups_for_prompt,
)
from src.llm.victory import BattleOutcome, VictoryEvaluator


@pytest.fixture(scope="module")
def fleet():
    with open("data/fleet_ships.json") as f:
        return json.load(f)


def _captain_kwargs(fleet, **overrides):
    kwargs = dict(
        captain_name="Vance",
        ship_name="TIS Resolute",
        ship_type="destroyer",
        fleet_data=fleet,
        ship_status={
            "hull_integrity": 90.0,
            "heat_percent": 50.0,
            "delta_v_remaining": 400.0,
            "nose_armor": 151.2,
            "lateral_armor": 26.0,
            "tail_armor": 30.3,
            "heatsink_capacity": 525,
            "radiators_extended": True,
        },
        tactical_status={
            "sim_time": 60.0,
            "angle_to_enemy_deg": 10.0,
            "ship_forward": {"x": 1, "y": 0, "z": 0},
            "enemies": [],
            "friendlies": [],
            "our_shots": 3, "our_hits": 1,
            "our_damage_dealt": 1.0, "our_damage_taken": 0.0,
        },
    )
    kwargs.update(overrides)
    return kwargs


class TestPointDefenseRangeComesFromFleetData:
    """The prompt hardcoded 100 km while PDLaser.from_fleet_data reads 250 km."""

    def test_quoted_pd_range_matches_the_weapon_the_sim_builds(self, fleet):
        from src.pointdefense import PDLaser

        laser = PDLaser.from_fleet_data(fleet["weapon_types"]["pd_laser"])
        text = format_weapon_groups_for_prompt(
            fleet["ships"]["destroyer"]["weapons"], fleet["weapon_types"]
        )
        assert f"within {laser.range_km:.0f}km" in text, text

    def test_range_is_read_from_data_not_baked_in(self, fleet):
        """Change the datum and the prompt must follow it."""
        mutated = copy.deepcopy(fleet["weapon_types"])
        mutated["pd_laser"]["range_km"] = 777
        text = format_weapon_groups_for_prompt(
            fleet["ships"]["destroyer"]["weapons"], mutated
        )
        assert "777km" in text
        assert "250km" not in text


class TestThermalClaimsMatchTheSimulation:
    """Captains were told about mechanics the engine never runs."""

    def test_no_weapon_overheat_lockout_is_promised(self, fleet):
        """WeaponState.can_fire has no heat term - so do not claim one."""
        import inspect

        from src.simulation import WeaponState

        source = inspect.getsource(WeaponState.can_fire)
        assert "heat" not in source, (
            "a heat gate was added to can_fire - the prompt may now legitimately "
            "promise a weapon lockout, update this test and the doctrine together"
        )

        system = build_captain_messages(**_captain_kwargs(fleet))[0]["content"]
        assert "overheat at 95" not in system.lower()

    def test_radiator_vulnerability_wording_matches_the_engine(self, fleet):
        """
        The prompt's claim about extended radiators must track whether the engine
        actually damages them. This flipped once already: the wording was removed
        while RadiatorHitResolver was unreachable, then radiator hits were wired
        into CombatSimulation - which made the honest wording dishonest again.
        Tie the two together so they cannot drift apart silently.
        """
        import inspect

        import src.simulation as simulation

        engine_source = inspect.getsource(simulation.CombatSimulation)
        radiator_damage_is_live = (
            "apply_radiator_damage" in engine_source
            or "resolve_radiator_hit" in engine_source
        )

        system = build_captain_messages(**_captain_kwargs(fleet))[0]["content"].lower()
        advertises_risk = any(
            word in system for word in ("exposed", "vulnerable", "shot off")
        )

        assert advertises_risk == radiator_damage_is_live, (
            f"engine damages radiators: {radiator_damage_is_live}, but the prompt "
            f"advertises a risk: {advertises_risk} - captains are being told "
            f"something the simulation does not do (or not told something it does)"
        )

    def test_cooling_number_is_derived_from_fleet_data(self, fleet):
        """+130 MW must come from radiator mass x dissipation, not a literal."""
        mutated = copy.deepcopy(fleet)
        rad = mutated["ships"]["destroyer"]["thermal"]["radiator"]
        rad["mass_tons"] = 20
        rad["dissipation_kw_per_kg"] = 13

        text = build_ship_capabilities_from_fleet(
            ship_name="TIS Resolute", ship_type="destroyer", fleet_data=mutated,
            hull_integrity=100, heat_percent=0, delta_v_remaining=500,
            nose_armor=150, lateral_armor=26, tail_armor=30,
            heatsink_capacity=525, radiators_extended=False,
        )
        assert "+260 MW cooling" in text, text


class TestEvasionClaimMatchesFireControl:
    def test_evade_is_described_as_the_multiplier_the_model_applies(self, fleet):
        from src.firecontrol import calculate_hit_probability
        from src.physics import Vector3D
        from src.simulation import create_ship_from_fleet_data

        target = create_ship_from_fleet_data("t", "destroyer", "beta", fleet)

        def prob(evading):
            return calculate_hit_probability(
                shooter_position=Vector3D(0, 0, 0),
                shooter_velocity=Vector3D(0, 0, 0),
                target_position=Vector3D(200_000, 0, 0),
                target_velocity=Vector3D(0, 0, 0),
                target_geometry=target.geometry,
                target_forward=Vector3D(-1, 0, 0),
                muzzle_velocity_kps=9.9,
                target_is_evading=evading,
            ).hit_probability

        ratio = prob(True) / prob(False)
        assert ratio > 0.55, "evasion is now stronger than the doctrine claims"

        text = PROJECTILE_PHYSICS_REFERENCE.lower()
        assert "halves enemy hit probability" not in text
        assert "40%" in text and "0.6" in text


class TestFrameLabelsAreHonest:
    """'ahead/starboard/above' are body-frame words and must not label world deltas."""

    def test_world_frame_friendly_offset_is_labelled_world_frame(self):
        overview = format_battlefield_overview(
            enemies=[],
            friendlies=[{
                "ship_id": "alpha_2", "name": "TIS B", "distance_km": 30.0,
                "hull_percent": 100.0,
                "relative_position": {"x": -20.0, "y": 10.0, "z": 3.0},
            }],
        )
        line = [l for l in overview.splitlines() if "TIS B" in l][0]
        assert "world frame" in line
        for body_word in ("ahead", "behind", "starboard", "port"):
            assert body_word not in line, line

    def test_body_frame_friendly_offset_uses_body_frame_words(self):
        overview = format_battlefield_overview(
            enemies=[],
            friendlies=[{
                "ship_id": "alpha_2", "name": "TIS B", "distance_km": 30.0,
                "hull_percent": 100.0,
                "relative_position": {"x": -20.0, "y": 10.0, "z": 3.0},
                "relative_bearing_km": {"forward": -20.0, "starboard": 10.0, "up": 3.0},
            }],
        )
        line = [l for l in overview.splitlines() if "TIS B" in l][0]
        assert "behind" in line and "starboard" in line
        assert "world frame" not in line

    def test_doctrine_states_set_heading_is_world_frame(self):
        """The HEADING maneuver normalises the raw dict as a world vector."""
        assert "WORLD-frame direction vector" in CAPTAIN_DOCTRINE


class TestEnemyClassAndCapabilitiesReachTheCaptain:
    def test_ship_class_falls_back_to_ship_type(self):
        overview = format_battlefield_overview(
            enemies=[{
                "ship_id": "beta_1", "name": "OCS G", "ship_class": "unknown",
                "ship_type": "dreadnought", "distance_km": 200.0,
                "closing_rate": 2.0, "angle_deg": 5.0, "hull_percent": 80.0,
                "armor": {}, "hit_chance": 39.0, "is_primary_target": True,
            }],
            friendlies=[],
        )
        assert "(Dreadnought)" in overview
        assert "unknown" not in overview.lower()

    def test_fleet_data_reaches_the_overview_from_the_captain_prompt(self, fleet):
        kwargs = _captain_kwargs(fleet)
        kwargs["tactical_status"]["enemies"] = [{
            "ship_id": "beta_1", "name": "OCS G", "ship_type": "dreadnought",
            "distance_km": 200.0, "closing_rate": 2.0, "angle_deg": 5.0,
            "hull_percent": 80.0, "armor": {}, "hit_chance": 39.0,
            "is_primary_target": True,
        }]
        user_turn = build_captain_messages(**kwargs)[1]["content"]

        spec = fleet["ships"]["dreadnought"]["performance"]
        assert "Capabilities:" in user_turn
        assert f"{spec['combat_acceleration_g']}g accel" in user_turn
        assert "Armament:" in user_turn


def _admiral_snapshot():
    from src.llm.admiral import AdmiralSnapshot, FriendlyShipSnapshot

    ship = FriendlyShipSnapshot(
        ship_id="alpha_1", ship_name="TIS Haiku-1", ship_type="destroyer",
        captain_name="Vance", position_km={"x": 0, "y": 0, "z": 0},
        velocity_kps=1.0, velocity_vector={"x": 1, "y": 0, "z": 0},
        hull_integrity=100.0, delta_v_remaining=500.0, heat_percent=5.0,
        max_acceleration_g=2.0, max_delta_v=500.0, weapons_summary="spinal",
        weapons_ready=[], weapons_cooling=[], weapons_destroyed=[],
        current_maneuver="INTERCEPT", current_target=None,
        radiators_extended=False, targeted_by=[],
    )
    return AdmiralSnapshot(
        timestamp=0.0, friendly_ships=[ship], enemy_ships=[],
        projectiles=[], fleet_summary="1 destroyer",
    )


class TestAdmiralPhaseParameter:
    """
    Phase 1 asks only for the fleet directive and its parser drops issue_order
    calls, so the phase-1 prompt must not demand them.
    """

    def test_directive_phase_suppresses_the_issue_order_checklist(self, fleet):
        snapshot = _admiral_snapshot()
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=snapshot, personality=None, fleet_data=fleet,
            phase="directive",
        )
        assert "MUST issue_order for this ship" not in prompt
        assert "MUST call issue_order for EACH" not in prompt
        assert "set_fleet_directive" in prompt
        assert "Do NOT call issue_order now" in prompt
        # The ship still has to be listed - the directive is about these ships.
        assert "TIS Haiku-1" in prompt

    def test_full_phase_still_demands_an_order_per_ship(self, fleet):
        snapshot = _admiral_snapshot()
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=snapshot, personality=None, fleet_data=fleet,
            phase="full",
        )
        assert "MUST issue_order for this ship" in prompt
        assert "MUST call issue_order for EACH" in prompt

    def test_phase_actually_changes_the_prompt(self, fleet):
        snapshot = _admiral_snapshot()
        common = dict(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=snapshot, personality=None, fleet_data=fleet,
        )
        assert build_admiral_prompt(phase="directive", **common) != \
            build_admiral_prompt(phase="full", **common)

    def test_the_order_requirement_is_stated_once(self, fleet):
        """The block used to appear twice, verbatim, in the same prompt."""
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=_admiral_snapshot(), personality=None,
            fleet_data=fleet, phase="full",
        )
        assert prompt.count("MUST ISSUE ORDERS TO EVERY SHIP") <= 1


class TestAdmiralPromptSelfConsistency:
    def test_ids_are_not_forbidden_and_permitted_at_once(self, fleet):
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=_admiral_snapshot(), personality=None,
            fleet_data=fleet, phase="full",
        )
        assert "EITHER the name OR the ID" in prompt
        assert "NOT ship IDs" not in prompt

    def test_shared_point_defense_claim_matches_pd_range(self, fleet):
        """There is no 50 km sharing radius anywhere in the PD code."""
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=_admiral_snapshot(), personality=None,
            fleet_data=fleet, phase="full",
        )
        assert "within 50km share point defense" not in prompt
        pd_range = fleet["weapon_types"]["pd_laser"]["range_km"]
        assert f"{pd_range:.0f} km" in prompt

    def test_shared_pd_range_is_read_from_data_not_baked_in(self, fleet):
        mutated = copy.deepcopy(fleet)
        mutated["weapon_types"]["pd_laser"]["range_km"] = 333
        prompt = build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=_admiral_snapshot(), personality=None,
            fleet_data=mutated, phase="full",
        )
        assert "333 km laser range" in prompt


class TestAdmiralPromptIsCacheable:
    """
    The static doctrine used to sit *after* the per-checkpoint snapshot, so no
    prefix boundary could ever cover it.
    """

    @staticmethod
    def _prompt(fleet, snapshot):
        return build_admiral_prompt(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=snapshot, personality=None, fleet_data=fleet,
            phase="full",
        )

    def test_doctrine_precedes_the_live_snapshot(self, fleet):
        prompt = self._prompt(fleet, _admiral_snapshot())
        marker = prompt.index(ADMIRAL_STATE_MARKER)
        for static_heading in (
            "=== YOUR ROLE ===",
            "=== TACTICAL COMMAND REFERENCE ===",
            "=== FLEET TACTICAL OPTIONS ===",
        ):
            assert prompt.index(static_heading) < marker, static_heading
        for volatile_heading in (
            "=== DUAL TEMPORAL SNAPSHOT ===",
            "=== ENEMY FLEET STATUS (OBSERVABLE) ===",
            "=== PROJECTILES IN FLIGHT ===",
        ):
            assert prompt.index(volatile_heading) > marker, volatile_heading

    def test_system_half_is_stable_while_the_snapshot_changes(self, fleet):
        early = _admiral_snapshot()
        later = _admiral_snapshot()
        later.friendly_ships[0].hull_integrity = 41.0
        later.friendly_ships[0].heat_percent = 88.0
        later.timestamp = 120.0

        common = dict(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            personality=None, fleet_data=fleet, phase="full",
        )
        a = build_admiral_messages(snapshot_t_zero=early, **common)
        b = build_admiral_messages(snapshot_t_zero=later, **common)

        assert a[0]["content"] == b[0]["content"], "cacheable prefix drifted"
        assert a[1]["content"] != b[1]["content"], "volatile turn did not change"

    def test_split_loses_no_content(self, fleet):
        snapshot = _admiral_snapshot()
        msgs = build_admiral_messages(
            admiral_name="Adm", faction="alpha", snapshot_t_minus_15=None,
            snapshot_t_zero=snapshot, personality=None, fleet_data=fleet,
            phase="full",
        )
        rejoined = msgs[0]["content"] + msgs[1]["content"]
        full = self._prompt(fleet, snapshot)
        # Only whitespace at the seam may differ.
        assert "".join(rejoined.split()) == "".join(full.split())


class TestDamageScoreIsBounded:
    """evaluate_by_damage documented 40/60 but the ratio term was unbounded."""

    @staticmethod
    def _ship(hull, dealt, taken):
        return SimpleNamespace(
            is_destroyed=False, hull_integrity=hull,
            damage_dealt_gj=dealt, damage_taken_gj=taken,
            module_layout=None,
            weapons={"w": SimpleNamespace(is_operational=True)},
        )

    def test_hull_is_not_drowned_out_by_a_lopsided_ratio(self):
        evaluator = VictoryEvaluator()
        # Alpha traded well but is nearly dead; Beta traded worse but is pristine.
        # Under the old unbounded ratio Alpha scored 800+ and won automatically.
        alpha = self._ship(hull=20.0, dealt=10.0, taken=0.5)
        beta = self._ship(hull=100.0, dealt=5.0, taken=10.0)

        outcome, winner, _ = evaluator.evaluate(alpha, beta, at_time_limit=True)
        assert outcome == BattleOutcome.BETA_VICTORY
        assert winner == "beta"

    def test_damage_term_never_exceeds_its_advertised_40_percent_weight(self):
        evaluator = VictoryEvaluator()
        # Both hulls at 0 -> the whole score is the damage term.
        alpha = self._ship(hull=0.0, dealt=1000.0, taken=0.0)
        beta = self._ship(hull=0.0, dealt=0.0, taken=1000.0)
        _, _, reason = evaluator.evaluate(alpha, beta, at_time_limit=True)
        scores = [float(tok.strip("()")) for tok in reason.replace("vs", " ").split()
                  if tok.strip("()").replace(".", "", 1).isdigit()]
        assert max(scores) <= 40.0 + 1e-6, reason

    def test_a_better_exchange_never_lowers_your_score(self):
        evaluator = VictoryEvaluator()
        beta = self._ship(hull=50.0, dealt=10.0, taken=10.0)

        def score_of_alpha(dealt):
            alpha = self._ship(hull=50.0, dealt=dealt, taken=10.0)
            _, _, reason = evaluator.evaluate(alpha, beta, at_time_limit=True)
            # "Alpha: X vs Beta: Y" or "X vs Y" - the first number is always Alpha's
            return float(reason.split("(")[1].split()[-1].rstrip(")")) \
                if "Alpha:" not in reason else \
                float(reason.split("Alpha:")[1].split()[0])

        scores = [score_of_alpha(d) for d in (1.0, 10.0, 100.0)]
        assert scores == sorted(scores), scores

    def test_identical_performance_is_a_draw(self):
        evaluator = VictoryEvaluator()
        alpha = self._ship(hull=70.0, dealt=30.0, taken=30.0)
        beta = self._ship(hull=70.0, dealt=30.0, taken=30.0)
        outcome, winner, _ = evaluator.evaluate(alpha, beta, at_time_limit=True)
        assert outcome == BattleOutcome.DRAW and winner is None

    def test_zero_damage_traded_does_not_crash_or_bias(self):
        evaluator = VictoryEvaluator()
        alpha = self._ship(hull=100.0, dealt=0.0, taken=0.0)
        beta = self._ship(hull=100.0, dealt=0.0, taken=0.0)
        outcome, _, _ = evaluator.evaluate(alpha, beta, at_time_limit=True)
        assert outcome == BattleOutcome.DRAW
