"""
Tests for admiral vision (docs/admiral_vision.md).

Everything runs offline: a stub client stands in for OpenRouter and captures
exactly what would have been sent, so we can assert that vision-enabled
admirals get an image content part and text-only admirals keep plain strings.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.admiral_view import (
    AdmiralViewState,
    build_live_frame,
    is_vision_model,
    png_to_data_url,
)
from src.llm.battle_runner import LLMBattleRunner, BattleConfig, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import BattleFleetConfig
from src.llm.prompts import CaptainPersonality

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class StubClient:
    """Captures every would-be API call; returns canned tool calls."""

    def __init__(self):
        self.captured = []

    def decide_with_tools(self, messages, tools, model=None, temperature=None):
        self.captured.append({"messages": messages, "tools": tools, "model": model})
        names = {t["function"]["name"] for t in tools}
        if "set_fleet_directive" in names:
            return [ToolCall(id="d", name="set_fleet_directive",
                             arguments={"directive": "Close and destroy."})]
        if "issue_order" in names:
            return [ToolCall(id="o", name="issue_order",
                             arguments={"order_text": "Engage nearest enemy."})]
        return []

    def complete(self, messages, model=None, temperature=None):
        raise AssertionError("complete() should not be called in these tests")


def make_runner(alpha_vision: bool, beta_admiral: bool = True,
                beta_vision: bool = False):
    fleet_config = BattleFleetConfig.from_dict({
        "battle_name": "Vision Test",
        "time_limit_s": 300.0,
        "decision_interval_s": 30.0,
        "initial_distance_km": 400.0,
        "alpha_fleet": {
            "admiral": {"model": "anthropic/claude-haiku-4.5",
                        "vision": alpha_vision},
            "ships": [
                {"ship_type": "destroyer", "model": "anthropic/claude-haiku-4.5"},
                {"ship_type": "corvette", "model": "anthropic/claude-haiku-4.5"},
            ],
        },
        "beta_fleet": {
            **({"admiral": {"model": "anthropic/claude-haiku-4.5",
                            "vision": beta_vision}} if beta_admiral else {}),
            "ships": [
                {"ship_type": "cruiser_torpedo", "model": "anthropic/claude-haiku-4.5"},
            ],
        },
    })
    config = BattleConfig(
        initial_distance_km=400.0,
        time_limit_s=300.0,
        max_checkpoints=3,
        verbose=False,
        personality_selection=False,
        record_battle=False,
        record_sim_trace=False,
        seed=99,
    )
    captain_cfg = LLMCaptainConfig(
        name="Test", ship_name="TIS Test", model="stub",
        personality=CaptainPersonality.BALANCED,
    )
    stub = StubClient()
    runner = LLMBattleRunner(
        config=config,
        alpha_config=captain_cfg,
        beta_config=captain_cfg,
        client=stub,
        fleet_config=fleet_config,
    )
    runner.setup_fleet_battle(load_fleet_data())
    return runner, stub


# ---------------------------------------------------------------------------
# Vision-capability heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("anthropic/claude-opus-5", True),
    ("openrouter/anthropic/claude-haiku-4.5", True),
    ("google/gemini-3.5-flash", True),
    ("openai/gpt-5.2", True),
    ("qwen/qwen3-vl-235b", True),
    ("deepseek/deepseek-v4", False),
    ("moonshotai/kimi-k3", True),   # natively multimodal, allowlisted 2026-08-06
    ("moonshotai/kimi-k2", False),  # older text-only K2 stays excluded
    ("", False),
])
def test_is_vision_model(model, expected):
    assert is_vision_model(model) == expected


def test_png_data_url_prefix():
    assert png_to_data_url(b"abc").startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Live frame + rendering
# ---------------------------------------------------------------------------

def test_build_live_frame_matches_trace_schema():
    runner, _ = make_runner(alpha_vision=True)
    for _ in range(5):
        runner.simulation.step()
    frame = build_live_frame(runner.simulation)
    assert frame["t"] == runner.simulation.current_time
    assert set(frame["ships"].keys()) == {"alpha_1", "alpha_2", "beta_1"}
    ship = frame["ships"]["alpha_1"]
    for key in ("pos", "vel", "fwd", "thrust", "maneuver", "destroyed", "hull"):
        assert key in ship
    assert isinstance(frame["torpedoes"], list)
    assert isinstance(frame["projectiles"], list)


def test_render_produces_png():
    matplotlib = pytest.importorskip("matplotlib")  # noqa: F841
    runner, _ = make_runner(alpha_vision=True)
    assert runner._admiral_view is not None
    for _ in range(10):
        runner.simulation.step()
        runner._admiral_view.sample(runner.simulation)
    png = runner._admiral_view.render(
        runner.simulation, runner._ships_meta(), faction="alpha")
    assert png is not None and png[:8] == PNG_MAGIC


def test_view_state_only_allocated_when_wanted():
    runner, _ = make_runner(alpha_vision=False, beta_vision=False)
    assert runner._admiral_view is None

    runner2, _ = make_runner(alpha_vision=True)
    assert runner2._admiral_view is not None


# ---------------------------------------------------------------------------
# Prompt attachment
# ---------------------------------------------------------------------------

def test_vision_admiral_gets_image_content_part(tmp_path):
    pytest.importorskip("matplotlib")
    runner, stub = make_runner(alpha_vision=True)
    runner._admiral_view_dir = tmp_path / "vision"
    for _ in range(10):
        runner.simulation.step()
        runner._admiral_view.sample(runner.simulation)

    decision = runner._get_admiral_decision(
        runner.alpha_admiral, list(runner.alpha_captains.values()), None)
    assert decision.fleet_directive == "Close and destroy."

    # First captured call is the Phase-1 directive call - it carries the image
    directive_call = stub.captured[0]
    user_msg = [m for m in directive_call["messages"] if m["role"] == "user"][0]
    assert isinstance(user_msg["content"], list)
    kinds = [part["type"] for part in user_msg["content"]]
    assert kinds == ["text", "image_url"]
    url = user_msg["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")

    # Phase-2 per-ship order calls stay text-only (image cost x N ships is
    # the admiral-vision budget rule from the design doc)
    for call in stub.captured[1:]:
        for msg in call["messages"]:
            assert isinstance(msg["content"], str)

    # The rendered frame is persisted for post-battle inspection
    saved = list((tmp_path / "vision").glob("cp*_alpha.png"))
    assert len(saved) == 1


def test_text_only_admiral_keeps_string_content():
    runner, stub = make_runner(alpha_vision=True, beta_vision=False)
    for _ in range(10):
        runner.simulation.step()
        runner._admiral_view.sample(runner.simulation)

    runner._get_admiral_decision(
        runner.beta_admiral, list(runner.beta_captains.values()), None)
    for call in stub.captured:
        for msg in call["messages"]:
            assert isinstance(msg["content"], str)


def test_pd_beam_and_event_buffers():
    view = AdmiralViewState()

    class FakeEvent:
        def __init__(self, name, t, ship_id, data):
            self.event_type = type("ET", (), {"name": name})()
            self.timestamp = t
            self.ship_id = ship_id
            self.target_id = None
            self.data = data

    view.note_event(FakeEvent("PD_ENGAGED", 100.0, "alpha_1",
                              {"target_id": "torp_beta_1_ab"}))
    view.note_event(FakeEvent("PD_ENGAGED", 90.0, "alpha_1",
                              {"target_id": "torp_beta_1_cd"}))
    view.note_event(FakeEvent("SHIP_DESTROYED", 99.0, "beta_1", {}))

    beams = view.active_pd_beams(t=100.5)
    assert ("alpha_1", "torp_beta_1_ab") in beams
    assert all(target != "torp_beta_1_cd" for _, target in beams)

    events = view.recent_events(t=100.5)
    assert any("SHIP DESTROYED" in label for _, label in events)
