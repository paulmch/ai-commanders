"""
Regression tests for the llm-runtime audit fixes.

Covers src/llm/captain.py, src/llm/battle_runner.py and src/llm/client.py.
No network access: the LLM client is either mocked or has its transport stubbed.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import BattleConfig, BattleResult, LLMBattleRunner
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import CaptainClient, LLMCallError, ToolCall
from src.physics import Vector3D
from src.projectile import KineticProjectile
from src.simulation import ProjectileInFlight


FLEET_DATA = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def make_runner(ship_type: str = "corvette", **config_kwargs) -> LLMBattleRunner:
    """A fully set up 1v1 runner with real ships and a mocked LLM client."""
    config = BattleConfig(
        verbose=False,
        record_battle=False,
        personality_selection=False,
        alpha_ship_type=ship_type,
        beta_ship_type=ship_type,
        **config_kwargs,
    )
    alpha = LLMCaptainConfig(
        name="Alpha Captain", ship_name="Alpha", ship_type=ship_type, fleet_data=FLEET_DATA
    )
    beta = LLMCaptainConfig(
        name="Beta Captain", ship_name="Beta", ship_type=ship_type, fleet_data=FLEET_DATA
    )
    runner = LLMBattleRunner(config, alpha, beta, client=Mock())
    runner.setup_battle(FLEET_DATA)
    return runner


def make_captain(ship_type: str = "corvette") -> LLMCaptain:
    config = LLMCaptainConfig(
        name="Solo", ship_name="Solo", ship_type=ship_type, fleet_data=FLEET_DATA
    )
    return LLMCaptain(config, client=Mock())


# ---------------------------------------------------------------------------
# Finding 1 - shot history closing/separating sign
# ---------------------------------------------------------------------------

class TestShotHistorySign:
    """rel_velocity_kps is POSITIVE when closing (repo-wide -r_hat . v_rel)."""

    def test_positive_relative_velocity_reads_as_closing(self):
        captain = make_captain()
        captain.record_shot("coilgun_mk3", 120.0, 4.0, "HIT", 0.72)
        text = captain._format_shot_history()
        assert "closing" in text
        assert "separating" not in text

    def test_negative_relative_velocity_reads_as_separating(self):
        captain = make_captain()
        captain.record_shot("coilgun_mk3", 120.0, -4.0, "MISS")
        text = captain._format_shot_history()
        assert "separating" in text
        assert "closing" not in text

    def test_sign_convention_matches_runner(self):
        """The runner's own closing computation must agree with the formatter."""
        runner = make_runner()
        alpha = runner.simulation.get_ship("alpha")
        beta = runner.simulation.get_ship("beta")
        # Drive them together: alpha chases beta.
        rel_pos = beta.position - alpha.position
        alpha.kinematic_state.velocity = rel_pos.normalized() * 4000.0

        runner._record_captain_shot(
            source_id="alpha", target_id="beta", weapon_slot="unknown",
            result="MISS", damage_gj=0.0,
        )
        shot = runner.alpha_captain.shot_history[-1]
        assert shot["rel_velocity_kps"] > 0, "closing should be positive"
        assert "closing" in runner.alpha_captain._format_shot_history()


# ---------------------------------------------------------------------------
# Finding 2 - incoming projectile ETA must account for the target's own velocity
# ---------------------------------------------------------------------------

class TestProjectileETA:

    def _incoming(self, own_velocity_x: float):
        runner = make_runner()
        ship = runner.simulation.get_ship("alpha")
        captain = runner.alpha_captain
        ship.kinematic_state.velocity = Vector3D(own_velocity_x, 0, 0)

        # Projectile 60 km "behind" the ship on -x, travelling +x at 6 km/s.
        proj = KineticProjectile(
            position=ship.position - Vector3D(60_000, 0, 0),
            velocity=Vector3D(6000, 0, 0),
            mass_kg=40.0,
            muzzle_velocity_kps=6.0,
        )
        runner.simulation.projectiles.append(ProjectileInFlight(
            projectile_id="p1", projectile=proj,
            source_ship_id="beta", target_ship_id="alpha",
        ))
        status = captain._build_tactical_status(
            ship, runner.simulation.get_ship("beta"), runner.simulation
        )
        assert status["incoming_projectiles"], "projectile aimed at us must be listed"
        return status["incoming_projectiles"][0]

    def test_fleeing_target_doubles_time_to_impact(self):
        """Ship running away at 3 km/s from a 6 km/s round: 3 km/s closure."""
        info = self._incoming(own_velocity_x=3000.0)
        assert info["eta_seconds"] == pytest.approx(20.0, rel=1e-6)
        assert info["closing_kps"] == pytest.approx(3.0, rel=1e-6)

    def test_closing_target_shortens_time_to_impact(self):
        """Ship charging the round at 3 km/s: 9 km/s closure."""
        info = self._incoming(own_velocity_x=-3000.0)
        assert info["eta_seconds"] == pytest.approx(60_000 / 9000, rel=1e-6)

    def test_eta_ordering_is_monotonic_in_own_velocity(self):
        """Whatever the numbers, fleeing must never give a shorter ETA."""
        fleeing = self._incoming(own_velocity_x=3000.0)["eta_seconds"]
        stationary = self._incoming(own_velocity_x=0.0)["eta_seconds"]
        charging = self._incoming(own_velocity_x=-3000.0)["eta_seconds"]
        assert charging < stationary < fleeing


# ---------------------------------------------------------------------------
# Finding 12 - projectile identity must not be inferred from world-frame speed
# ---------------------------------------------------------------------------

class TestProjectileIdentification:

    def test_siege_round_is_not_labelled_a_turret_round(self):
        captain = make_captain()
        spec = FLEET_DATA["weapon_types"]["heavy_siege_coiler_mk3"]
        proj = KineticProjectile(
            position=Vector3D.zero(),
            velocity=Vector3D(4700, 0, 0),
            mass_kg=spec["warhead_mass_kg"],
            muzzle_velocity_kps=spec["muzzle_velocity_kps"],
        )
        label, energy = captain._identify_projectile(proj)
        assert label == "heavy_siege_coiler_mk3"
        assert energy == pytest.approx(spec["kinetic_energy_gj"])

    def test_label_is_independent_of_shooter_velocity(self):
        """Same weapon, wildly different world speeds -> same identification."""
        captain = make_captain()
        spec = FLEET_DATA["weapon_types"]["coilgun_mk3"]
        slow = KineticProjectile(
            position=Vector3D.zero(), velocity=Vector3D(6000, 0, 0),
            mass_kg=spec["warhead_mass_kg"], muzzle_velocity_kps=spec["muzzle_velocity_kps"],
        )
        # Fired from a ship already doing 20 km/s - world speed now 26 km/s.
        fast = KineticProjectile(
            position=Vector3D.zero(), velocity=Vector3D(26000, 0, 0),
            mass_kg=spec["warhead_mass_kg"], muzzle_velocity_kps=spec["muzzle_velocity_kps"],
        )
        assert captain._identify_projectile(slow) == captain._identify_projectile(fast)
        assert captain._identify_projectile(fast)[0] == "coilgun_mk3"

    def test_unknown_round_falls_back_to_muzzle_velocity_class(self):
        captain = make_captain()
        proj = KineticProjectile(
            position=Vector3D.zero(), velocity=Vector3D(99_000, 0, 0),
            mass_kg=1.0, muzzle_velocity_kps=12.0,
        )
        label, _ = captain._identify_projectile(proj)
        assert label == "Spinal"  # classified by muzzle velocity, not world speed


# ---------------------------------------------------------------------------
# Finding 3 - torpedo threats / magazine reach the captain, launch honours target
# ---------------------------------------------------------------------------

class TestTorpedoDisclosure:

    def test_magazine_count_is_in_ship_status(self):
        runner = make_runner("corvette")
        ship = runner.simulation.get_ship("alpha")
        assert ship.torpedo_launcher is not None, "corvette should carry torpedoes"
        status = runner.alpha_captain._build_ship_status(ship)
        assert status["torpedoes_remaining"] == ship.torpedo_launcher.current_magazine
        assert status["torpedo_capacity"] == ship.torpedo_launcher.magazine_capacity

    def test_magazine_count_is_rendered(self):
        runner = make_runner("corvette")
        ship = runner.simulation.get_ship("alpha")
        captain = runner.alpha_captain
        status = captain._build_ship_status(ship)
        text = captain._format_torpedo_section(status, {"torpedo_threats": []})
        assert str(ship.torpedo_launcher.current_magazine) in text

    def test_inbound_torpedo_is_rendered_with_eta(self):
        # Threats are rendered ONCE, in the turn state, by
        # prompts.format_torpedo_threats; _format_torpedo_section used to
        # duplicate the list and now carries only the magazine line.
        from src.llm.prompts import format_torpedo_threats

        text = format_torpedo_threats([
            {"source": "Determination", "distance_km": 120.0,
             "closing_kps": 3.0, "eta_seconds": 40.0},
        ])
        assert "INBOUND ORDNANCE" in text
        assert "Determination" in text
        assert "40s" in text

        captain = make_captain()
        section = captain._format_torpedo_section({}, {"torpedo_threats": [
            {"source": "Determination", "distance_km": 120.0,
             "closing_kps": 3.0, "eta_seconds": 40.0},
        ]})
        assert "Determination" not in section, (
            "threat list is duplicated outside the turn-state renderer again"
        )

    def test_no_torpedo_section_when_nothing_to_report(self):
        captain = make_captain()
        assert captain._format_torpedo_section({}, {"torpedo_threats": []}) == ""

    def test_launch_torpedo_uses_primary_target_not_first_enemy(self):
        runner = make_runner("corvette")
        captain = runner.alpha_captain

        class FakeSim:
            def get_enemy_ships(self, ship_id):
                return [SimpleNamespace(ship_id="decoy", name="Decoy"),
                        SimpleNamespace(ship_id="prize", name="Prize")]

        captain.primary_target_id = "prize"
        cmd = captain._execute_tool(
            ToolCall(id="1", name="launch_torpedo", arguments={}), FakeSim(), "alpha"
        )
        # launch_torpedo now emits a salvo (a list of launch commands).
        rounds = cmd if isinstance(cmd, list) else [cmd]
        assert rounds, "no torpedo command emitted"
        assert all(r["target_id"] == "prize" for r in rounds)

    def test_launch_torpedo_falls_back_when_primary_target_is_gone(self):
        captain = make_captain()

        class FakeSim:
            def get_enemy_ships(self, ship_id):
                return [SimpleNamespace(ship_id="only", name="Only")]

        captain.primary_target_id = "destroyed_ship"
        cmd = captain._execute_tool(
            ToolCall(id="1", name="launch_torpedo", arguments={}), FakeSim(), "alpha"
        )
        rounds = cmd if isinstance(cmd, list) else [cmd]
        assert rounds and all(r["target_id"] == "only" for r in rounds)


# ---------------------------------------------------------------------------
# Finding 10 - set_primary_target must reach the ship object
# ---------------------------------------------------------------------------

class TestSetPrimaryTarget:

    def test_emits_set_target_command(self):
        captain = make_captain()

        class FakeSim:
            def get_enemy_ships(self, ship_id):
                return [SimpleNamespace(ship_id="beta_2", name="Vigilant")]

        cmd = captain._execute_tool(
            ToolCall(id="1", name="set_primary_target",
                     arguments={"target_name": "Vigilant"}),
            FakeSim(), "alpha",
        )
        assert cmd == {"type": "set_target", "target_id": "beta_2"}
        assert captain.primary_target_id == "beta_2"

    def test_command_actually_sets_ship_field(self):
        """End-to-end: the emitted command must be understood by the simulation."""
        runner = make_runner()
        sim = runner.simulation
        cmd = runner.alpha_captain._execute_tool(
            ToolCall(id="1", name="set_primary_target", arguments={"target_name": "beta"}),
            sim, "alpha",
        )
        assert cmd is not None
        sim.inject_command("alpha", cmd)
        assert sim.get_ship("alpha").primary_target_id == "beta"

    def test_unknown_target_reports_an_error_to_the_model(self):
        captain = make_captain()

        class FakeSim:
            def get_enemy_ships(self, ship_id):
                return [SimpleNamespace(ship_id="beta_2", name="Vigilant")]

        cmd = captain._execute_tool(
            ToolCall(id="1", name="set_primary_target",
                     arguments={"target_name": "Nonexistent"}),
            FakeSim(), "alpha",
        )
        assert cmd is None
        assert captain.pending_tool_errors, "error must be captured for the next prompt"
        assert "Vigilant" in captain.pending_tool_errors[0], "valid targets must be listed"


# ---------------------------------------------------------------------------
# Finding 7 - tool errors are fed back to the model
# ---------------------------------------------------------------------------

class TestToolErrorFeedback:

    def test_unknown_tool_is_recorded_not_just_printed(self):
        captain = make_captain()
        captain._execute_tool(ToolCall(id="1", name="warp_drive", arguments={}), Mock(), "alpha")
        assert any("warp_drive" in e for e in captain.pending_tool_errors)

    def test_errors_are_rendered_for_the_following_turn(self):
        captain = make_captain()
        captain._record_tool_error("set_maneuver: invalid maneuver_type 'WARP'")
        # Simulate the roll-over that decide() performs.
        captain.last_tool_errors = captain.pending_tool_errors
        captain.pending_tool_errors = []
        text = captain._format_tool_errors()
        assert "REJECTED" in text
        assert "WARP" in text

    def test_no_error_block_when_last_turn_was_clean(self):
        captain = make_captain()
        assert captain._format_tool_errors() == ""

    def test_discussion_limit_is_surfaced_as_an_error(self):
        captain = make_captain()
        captain.has_admiral = True
        captain.discussion_exchanges = captain.max_discussion_exchanges
        captain._execute_tool(
            ToolCall(id="1", name="discuss_with_admiral", arguments={"question": "why?"}),
            Mock(), "alpha",
        )
        assert captain.pending_tool_errors


# ---------------------------------------------------------------------------
# Finding 6 - one checkpoint must produce exactly one decision-history entry
# ---------------------------------------------------------------------------

class TestDecisionBookkeeping:

    def test_revert_removes_exactly_one_entry(self):
        captain = make_captain()
        captain.decision_count = 3
        captain.decision_history = [{"checkpoint": 1}, {"checkpoint": 2}, {"checkpoint": 3}]
        captain.revert_last_decision()
        assert captain.decision_count == 2
        assert [d["checkpoint"] for d in captain.decision_history] == [1, 2]

    def test_revert_is_safe_when_nothing_recorded(self):
        captain = make_captain()
        captain.revert_last_decision()
        assert captain.decision_count == 0
        assert captain.decision_history == []


# ---------------------------------------------------------------------------
# Finding 9 - targeting awareness / damage-taken / shot feedback are wired up
# ---------------------------------------------------------------------------

class TestCaptainFeedbackChannels:

    def test_targeting_awareness_is_refreshed(self):
        runner = make_runner()
        runner.beta_captain.primary_target_id = "alpha"
        runner.alpha_captain.primary_target_id = None
        runner._refresh_targeting_awareness()
        assert runner.alpha_captain.targeting_me == ["beta"]
        assert runner.beta_captain.targeting_me == []

    def test_targeting_awareness_clears_when_enemy_switches_away(self):
        runner = make_runner()
        runner.beta_captain.primary_target_id = "alpha"
        runner._refresh_targeting_awareness()
        assert runner.alpha_captain.targeting_me == ["beta"]
        runner.beta_captain.primary_target_id = None
        runner._refresh_targeting_awareness()
        assert runner.alpha_captain.targeting_me == []

    def test_hit_logs_are_cleared_between_checkpoints(self):
        runner = make_runner()
        runner.alpha_captain.record_hit_received(
            time=10.0, weapon="coilgun_mk3", location="nose",
            damage_cm=3.0, remaining_cm=7.0, source_ship="Beta",
        )
        assert runner.alpha_captain.recent_hits
        runner._clear_captain_hit_logs()
        assert runner.alpha_captain.recent_hits == []

    def test_shot_feedback_works_for_non_alpha_beta_ship_ids(self):
        """Fleet ships are not called 'alpha'/'beta' - feedback must still land."""
        runner = make_runner()
        sim = runner.simulation
        alpha_ship = sim.get_ship("alpha")
        beta_ship = sim.get_ship("beta")

        # Re-register the captains under fleet-style ids pointing at real ships.
        runner.alpha_captains = {alpha_ship.ship_id: runner.alpha_captain}
        runner.beta_captains = {beta_ship.ship_id: runner.beta_captain}

        runner._record_captain_shot(
            source_id=alpha_ship.ship_id, target_id=beta_ship.ship_id,
            weapon_slot="unknown", result="HIT", damage_gj=1.2,
        )
        assert len(runner.alpha_captain.shot_history) == 1
        shot = runner.alpha_captain.shot_history[0]
        expected_km = (beta_ship.position - alpha_ship.position).magnitude / 1000
        assert shot["distance_km"] == pytest.approx(expected_km)

    def test_shot_feedback_is_attributed_to_the_shooter_only(self):
        runner = make_runner()
        runner._record_captain_shot(
            source_id="beta", target_id="alpha", weapon_slot="unknown",
            result="MISS", damage_gj=0.0,
        )
        assert len(runner.beta_captain.shot_history) == 1
        assert runner.alpha_captain.shot_history == []


# ---------------------------------------------------------------------------
# Finding 4 - admiral draw agreement
# ---------------------------------------------------------------------------

class _FakeAdmiral:
    def __init__(self, name):
        self.name = name
        self.has_proposed_draw = False
        self.has_accepted_draw = False
        self._pending_enemy_message = None
        self._received_enemy_messages = []

    def get_pending_enemy_message(self):
        msg = self._pending_enemy_message
        self._pending_enemy_message = None
        return msg

    def receive_enemy_admiral_message(self, message):
        self._received_enemy_messages.append(message)


class TestAdmiralDraw:

    def _runner_with_admirals(self):
        runner = make_runner()
        runner.alpha_admiral = _FakeAdmiral("Alpha Admiral")
        runner.beta_admiral = _FakeAdmiral("Beta Admiral")
        return runner

    def test_propose_alone_is_not_a_draw(self):
        runner = self._runner_with_admirals()
        runner.alpha_admiral.has_proposed_draw = True
        assert runner._fleet_draw_agreed() is False

    def test_propose_plus_accept_is_a_draw(self):
        runner = self._runner_with_admirals()
        runner.alpha_admiral.has_proposed_draw = True
        runner.beta_admiral.has_accepted_draw = True
        assert runner._fleet_draw_agreed() is True

    def test_draw_is_symmetric(self):
        runner = self._runner_with_admirals()
        runner.beta_admiral.has_proposed_draw = True
        runner.alpha_admiral.has_accepted_draw = True
        assert runner._fleet_draw_agreed() is True

    def test_agreed_draw_ends_the_fleet_battle(self):
        runner = self._runner_with_admirals()
        runner.is_fleet_mode = True
        runner.alpha_ships = {"alpha": runner.simulation.get_ship("alpha")}
        runner.beta_ships = {"beta": runner.simulation.get_ship("beta")}
        assert runner._is_fleet_battle_over() is False
        runner.alpha_admiral.has_proposed_draw = True
        runner.beta_admiral.has_accepted_draw = True
        assert runner._is_fleet_battle_over() is True

    def test_real_admiral_exposes_the_attributes_the_runner_reads(self):
        """Guards against the original bug: runner read a name that never existed."""
        from src.llm.admiral import LLMAdmiral
        from src.llm.fleet_config import AdmiralConfig

        admiral = LLMAdmiral(
            config=AdmiralConfig(model="test/model", name="Test"),
            faction="alpha",
            client=Mock(),
            fleet_data=FLEET_DATA,
        )
        # The runner reads these directly (no getattr default), so a rename here
        # must fail loudly rather than silently disabling draw negotiation.
        assert admiral.has_proposed_draw is False
        assert admiral.has_accepted_draw is False

        runner = make_runner()
        runner.alpha_admiral = admiral
        runner.beta_admiral = LLMAdmiral(
            config=AdmiralConfig(model="test/model", name="Test2"),
            faction="beta",
            client=Mock(),
            fleet_data=FLEET_DATA,
        )
        assert runner._fleet_draw_agreed() is False
        admiral.has_proposed_draw = True
        runner.beta_admiral.has_accepted_draw = True
        assert runner._fleet_draw_agreed() is True


# ---------------------------------------------------------------------------
# Finding 5 - admiral <-> admiral messages are delivered outside the MCP bridge
# ---------------------------------------------------------------------------

class TestAdmiralMessaging:

    def test_message_is_delivered_to_the_opponent(self):
        runner = make_runner()
        runner.alpha_admiral = _FakeAdmiral("A")
        runner.beta_admiral = _FakeAdmiral("B")
        runner.alpha_admiral._pending_enemy_message = "Withdraw and live."

        runner._exchange_admiral_messages()

        assert runner.beta_admiral._received_enemy_messages == ["Withdraw and live."]
        assert runner.alpha_admiral._received_enemy_messages == []

    def test_message_is_drained_not_resent(self):
        runner = make_runner()
        runner.alpha_admiral = _FakeAdmiral("A")
        runner.beta_admiral = _FakeAdmiral("B")
        runner.alpha_admiral._pending_enemy_message = "Once only."

        runner._exchange_admiral_messages()
        runner._exchange_admiral_messages()

        assert runner.beta_admiral._received_enemy_messages == ["Once only."]

    def test_simultaneous_messages_do_not_echo_back_to_sender(self):
        runner = make_runner()
        runner.alpha_admiral = _FakeAdmiral("A")
        runner.beta_admiral = _FakeAdmiral("B")
        runner.alpha_admiral._pending_enemy_message = "From A"
        runner.beta_admiral._pending_enemy_message = "From B"

        runner._exchange_admiral_messages()

        assert runner.beta_admiral._received_enemy_messages == ["From A"]
        assert runner.alpha_admiral._received_enemy_messages == ["From B"]

    def test_real_admirals_exchange_messages(self):
        """Same wiring against the real LLMAdmiral API, not just the fake."""
        from src.llm.admiral import LLMAdmiral
        from src.llm.fleet_config import AdmiralConfig

        runner = make_runner()
        runner.alpha_admiral = LLMAdmiral(
            config=AdmiralConfig(model="test/model", name="A"),
            faction="alpha", client=Mock(), fleet_data=FLEET_DATA,
        )
        runner.beta_admiral = LLMAdmiral(
            config=AdmiralConfig(model="test/model", name="B"),
            faction="beta", client=Mock(), fleet_data=FLEET_DATA,
        )
        runner.alpha_admiral._pending_enemy_message = "Stand down."

        runner._exchange_admiral_messages()

        assert runner.beta_admiral._received_enemy_messages == ["Stand down."]
        assert runner.alpha_admiral.get_pending_enemy_message() is None

    def test_captain_message_is_collected_in_fleet_mode(self):
        """Fleet battles only ever delivered messages; nothing collected them."""
        from src.llm.communication import FleetCommunicationChannel

        runner = make_runner()
        runner.fleet_communication = FleetCommunicationChannel()
        runner.fleet_communication.register_ship(
            "alpha_1", "Alpha Captain", "Alpha", "alpha"
        )
        captain = runner.alpha_captain
        captain.pending_message = {
            "content": "Break off or be destroyed.",
            "recipient": "ALL_ENEMIES",
            "target_ship": None,
        }

        runner._collect_captain_message(captain, "alpha_1")

        queued = runner.fleet_communication.pending_messages
        assert [m.content for m in queued] == ["Break off or be destroyed."]
        # Drained, so it is not re-sent next checkpoint.
        assert captain.get_pending_message() is None

    def test_no_admirals_is_a_noop(self):
        """With only one admiral present, nothing is exchanged or consumed."""
        runner = make_runner()
        runner.alpha_admiral = None
        beta = _FakeAdmiral("B")
        beta._pending_enemy_message = "should not be delivered anywhere"
        runner.beta_admiral = beta

        runner._exchange_admiral_messages()  # must not raise

        # With no counterpart there is nobody to deliver to: beta must not have
        # received anything, and its own outgoing message must not be silently
        # swallowed either.
        assert beta._received_enemy_messages == [], "delivered with no counterpart"
        assert beta._pending_enemy_message == "should not be delivered anywhere", (
            "outgoing message was drained despite having no recipient"
        )


# ---------------------------------------------------------------------------
# Finding 8 - reproducibility: seed is configurable, recorded and effective
# ---------------------------------------------------------------------------

class TestSeeding:

    def test_seed_defaults_to_none(self):
        assert BattleConfig().seed is None

    def test_seed_makes_the_rng_deterministic(self):
        runner = make_runner(seed=1234)
        first = [random.random() for _ in range(5)]
        runner._seed_rng()
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_different_seeds_diverge(self):
        runner_a = make_runner(seed=1)
        a = [random.random() for _ in range(5)]
        runner_b = make_runner(seed=2)
        b = [random.random() for _ in range(5)]
        assert a != b

    def test_unseeded_runner_does_not_touch_global_rng(self):
        random.seed(99)
        expected = [random.random() for _ in range(3)]
        random.seed(99)
        make_runner()  # seed=None
        assert [random.random() for _ in range(3)] == expected

    def test_seed_is_reported_on_the_result(self):
        assert "seed" in BattleResult.__dataclass_fields__


# ---------------------------------------------------------------------------
# Findings 15/16 - client: per-call temperature, sane defaults, real errors
# ---------------------------------------------------------------------------

def _client(**kwargs) -> CaptainClient:
    return CaptainClient(api_key="test-key", **kwargs)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=Mock(), response=self
            )


TOOL_REPLY = {
    "choices": [{
        "message": {"tool_calls": [
            {"id": "c1", "function": {"name": "set_maneuver",
                                      "arguments": '{"maneuver_type": "INTERCEPT"}'}}
        ]},
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 10},
}


class TestClientTemperature:

    def test_per_call_temperature_overrides_the_client_default(self):
        client = _client(temperature=0.7)
        captured = {}

        def fake_post(url, headers=None, json=None):
            captured.update(json)
            return _FakeResponse(payload=TOOL_REPLY)

        client._client.post = fake_post
        client.decide_with_tools([{"role": "user", "content": "hi"}], [], temperature=0.15)
        assert captured["temperature"] == 0.15

    def test_client_default_is_used_when_no_per_call_value(self):
        client = _client(temperature=0.42)
        captured = {}
        client._client.post = lambda url, headers=None, json=None: (
            captured.update(json) or _FakeResponse(payload=TOOL_REPLY)
        )
        client.decide_with_tools([{"role": "user", "content": "hi"}], [])
        assert captured["temperature"] == 0.42

    def test_temperature_is_omitted_entirely_when_none(self):
        """Reasoning models reject an explicit temperature with a 400."""
        client = _client(temperature=None)
        captured = {}
        client._client.post = lambda url, headers=None, json=None: (
            captured.update(json) or _FakeResponse(payload=TOOL_REPLY)
        )
        client.decide_with_tools([{"role": "user", "content": "hi"}], [])
        assert "temperature" not in captured

    def test_captain_sends_its_configured_temperature(self):
        """The per-captain temperature from fleet config must reach the API."""
        captain = make_captain()
        captain.config.temperature = 0.11
        captain.client.decide_with_tools.return_value = []

        runner = make_runner()
        captain.decide("alpha", runner.simulation)

        kwargs = captain.client.decide_with_tools.call_args.kwargs
        assert kwargs["temperature"] == 0.11


class TestClientDefaults:

    def test_default_model_is_not_the_retired_slug(self):
        assert _client().model != "anthropic/claude-3.5-sonnet"

    def test_max_tokens_fits_a_multi_tool_turn(self):
        assert _client().max_tokens >= 4096

    def test_openrouter_prefix_is_stripped(self):
        assert _client(model="openrouter/anthropic/claude-sonnet-5").model == \
            "anthropic/claude-sonnet-5"


class TestClientErrorHandling:

    def test_permanent_error_raises_instead_of_looking_like_no_orders(self):
        client = _client()
        calls = []

        def fake_post(url, headers=None, json=None):
            calls.append(1)
            return _FakeResponse(status_code=400, text="bad model")

        client._client.post = fake_post
        with pytest.raises(LLMCallError):
            client.decide_with_tools([{"role": "user", "content": "hi"}], [])
        assert len(calls) == 1, "a 400 must not be retried"

    def test_transient_error_is_retried_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("src.llm.client.time.sleep", lambda *_: None)
        client = _client()
        responses = [_FakeResponse(status_code=503, text="busy"),
                     _FakeResponse(payload=TOOL_REPLY)]
        client._client.post = lambda url, headers=None, json=None: responses.pop(0)

        result = client.decide_with_tools([{"role": "user", "content": "hi"}], [])
        assert len(result) == 1
        assert client.stats.retries == 1

    def test_empty_tool_list_is_distinct_from_failure(self):
        client = _client()
        client._client.post = lambda url, headers=None, json=None: _FakeResponse(
            payload={"choices": [{"message": {"content": "thinking"},
                                  "finish_reason": "stop"}]}
        )
        assert client.decide_with_tools([{"role": "user", "content": "hi"}], []) == []
        assert client.stats.failures == 0

    def test_malformed_tool_arguments_are_discarded_not_silently_emptied(self):
        client = _client()
        client._client.post = lambda url, headers=None, json=None: _FakeResponse(
            payload={"choices": [{
                "message": {"tool_calls": [
                    {"id": "c1", "function": {"name": "set_maneuver",
                                              "arguments": '{"maneuver_type": "INTER'}}
                ]},
                "finish_reason": "length",
            }]}
        )
        assert client.decide_with_tools([{"role": "user", "content": "hi"}], []) == []
        assert client.stats.malformed_arguments == 1

    def test_captain_failure_does_not_pollute_decision_history(self):
        captain = make_captain()
        captain.client.decide_with_tools.side_effect = LLMCallError("boom")
        runner = make_runner()

        commands = captain.decide("alpha", runner.simulation)

        assert commands == []
        assert captain.last_call_failed is True
        assert captain.decision_history == []
        assert captain.decision_count == 0
