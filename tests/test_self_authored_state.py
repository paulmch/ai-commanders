"""
Tests for self-authored durable state: the admiral's standing battle plan and
the captain's log.

Both exist to carry an agent's own INTENT across otherwise-stateless
checkpoints: everything else in a prompt is telemetry rebuilt by the harness,
these two are the only text the agent writes to its future self.

No network access: LLM clients are mocked.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llm.admiral import LLMAdmiral
from src.llm.admiral_tools import get_admiral_tools
from src.llm.battle_recorder import BattleRecorder, EventType
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import AdmiralConfig
from src.llm.prompts import (
    ADMIRAL_DOCTRINE,
    CAPTAIN_DOCTRINE,
    build_admiral_prompt,
    build_admiral_response_prompt,
    build_admiral_ship_order_prompt,
)
from src.llm.tools import get_captain_tools, get_captain_tools_for_ship


FLEET_DATA = json.loads(
    (Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text()
)


def make_admiral(client=None) -> LLMAdmiral:
    return LLMAdmiral(
        config=AdmiralConfig(model="test/model", name="Test Admiral"),
        faction="alpha",
        client=client or Mock(),
        fleet_data=FLEET_DATA,
    )


def make_captain(ship_type: str = "destroyer") -> LLMCaptain:
    config = LLMCaptainConfig(
        name="Solo", ship_name="Solo", ship_type=ship_type, fleet_data=FLEET_DATA
    )
    return LLMCaptain(config, client=Mock())


def empty_sim(t: float = 120.0) -> SimpleNamespace:
    """Simulation double sufficient for LLMAdmiral.decide with no captains."""
    return SimpleNamespace(
        current_time=t,
        torpedoes=[],
        projectiles=[],
        get_ship=lambda ship_id: None,
    )


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

class TestToolSurface:

    def test_admiral_has_set_battle_plan_tool(self):
        tools = get_admiral_tools()
        plan_tools = [t for t in tools if t["function"]["name"] == "set_battle_plan"]
        assert len(plan_tools) == 1
        assert plan_tools[0]["function"]["parameters"]["required"] == ["plan"]

    def test_all_captains_have_log_note(self):
        generic = get_captain_tools()
        assert any(t["function"]["name"] == "log_note" for t in generic)
        # Hull-derived surfaces must keep it too - gun ship and gunless corvette.
        for ship_type in ("destroyer", "corvette", "torpedo_cruiser"):
            tools = get_captain_tools_for_ship(ship_type, FLEET_DATA, has_torpedoes=True)
            assert any(t["function"]["name"] == "log_note" for t in tools), ship_type


# ---------------------------------------------------------------------------
# Admiral standing plan
# ---------------------------------------------------------------------------

class TestStandingPlan:

    PLAN = "Phase 1: close to 300km. Phase 2: full salvo at their cruiser."

    def test_set_battle_plan_persists_on_the_admiral(self):
        client = Mock()
        client.decide_with_tools.return_value = [
            ToolCall(id="1", name="set_fleet_directive",
                     arguments={"directive": "Close and engage"}),
            ToolCall(id="2", name="set_battle_plan", arguments={"plan": self.PLAN}),
        ]
        admiral = make_admiral(client)

        decision = admiral.decide(simulation=empty_sim(t=120.0), captains=[])

        assert admiral.standing_plan == self.PLAN
        assert admiral.standing_plan_time == 120.0
        assert decision.battle_plan_update == self.PLAN

    def test_blank_plan_is_ignored(self):
        client = Mock()
        client.decide_with_tools.return_value = [
            ToolCall(id="1", name="set_battle_plan", arguments={"plan": "   "}),
        ]
        admiral = make_admiral(client)
        decision = admiral.decide(simulation=empty_sim(), captains=[])
        assert admiral.standing_plan is None
        assert decision.battle_plan_update is None

    def test_plan_is_echoed_back_in_the_next_checkpoint_prompt(self):
        client = Mock()
        client.decide_with_tools.return_value = [
            ToolCall(id="1", name="set_battle_plan", arguments={"plan": self.PLAN}),
        ]
        admiral = make_admiral(client)
        admiral.decide(simulation=empty_sim(t=90.0), captains=[])

        client.decide_with_tools.return_value = []
        admiral.decide(simulation=empty_sim(t=120.0), captains=[])

        # First positional arg of the last phase-1 call is the messages list.
        messages = client.decide_with_tools.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "YOUR STANDING BATTLE PLAN" in system_prompt
        assert self.PLAN in system_prompt
        assert "T+90s" in system_prompt

    def test_unset_plan_renders_a_nudge(self):
        prompt = build_admiral_prompt(
            admiral_name="A", faction="alpha",
            snapshot_t_minus_15=None, snapshot_t_zero=None,
            personality=None, fleet_data=FLEET_DATA,
        )
        assert "YOUR STANDING BATTLE PLAN" in prompt
        assert "set_battle_plan" in prompt

    def test_plan_reaches_per_ship_order_prompt(self):
        prompt = build_admiral_ship_order_prompt(
            admiral_name="A", ship_name="TIS X", ship_type="destroyer",
            captain_name="C", fleet_directive="Engage", snapshot=None,
            standing_plan=self.PLAN,
        )
        assert self.PLAN in prompt

    def test_plan_reaches_captain_response_prompt(self):
        prompt = build_admiral_response_prompt(
            admiral_name="A", captain_ship_name="TIS X", question="Orders?",
            personality=None, standing_plan=self.PLAN,
        )
        assert self.PLAN in prompt

    def test_doctrine_documents_the_tool(self):
        assert "set_battle_plan" in ADMIRAL_DOCTRINE


# ---------------------------------------------------------------------------
# Captain's log
# ---------------------------------------------------------------------------

class TestCaptainsLog:

    def test_log_note_persists_and_renders(self):
        captain = make_captain()
        sim = SimpleNamespace(current_time=90.0)
        cmd = captain._execute_tool(
            ToolCall(id="1", name="log_note",
                     arguments={"note": "Brake at 150km, attack from tail."}),
            sim, "alpha_1",
        )
        assert cmd is None  # a note is not a ship command
        assert len(captain.captain_log) == 1

        rendered = captain._format_captain_log()
        assert "T+90s" in rendered
        assert "Brake at 150km" in rendered
        assert "CAPTAIN'S LOG" in rendered

    def test_empty_log_renders_nothing(self):
        assert make_captain()._format_captain_log() == ""

    def test_blank_note_is_ignored(self):
        captain = make_captain()
        captain._execute_tool(
            ToolCall(id="1", name="log_note", arguments={"note": "  "}),
            SimpleNamespace(current_time=0.0), "alpha_1",
        )
        assert captain.captain_log == []

    def test_notes_are_bounded_in_length_and_count(self):
        captain = make_captain()
        sim = SimpleNamespace(current_time=30.0)
        captain._execute_tool(
            ToolCall(id="1", name="log_note", arguments={"note": "x" * 1000}),
            sim, "alpha_1",
        )
        assert len(captain.captain_log[0]["note"]) == captain.MAX_LOG_NOTE_CHARS + 3

        for i in range(5):
            captain._execute_tool(
                ToolCall(id=str(i), name="log_note", arguments={"note": f"note {i}"}),
                SimpleNamespace(current_time=60.0 + 30 * i), "alpha_1",
            )
        assert len(captain.captain_log) == captain.MAX_LOG_ENTRIES
        assert captain.captain_log[-1]["note"] == "note 4"
        # The truncated first note fell off the front.
        assert all(e["note"].startswith("note") for e in captain.captain_log)

    def test_doctrine_documents_the_tool(self):
        assert "log_note" in CAPTAIN_DOCTRINE


# ---------------------------------------------------------------------------
# Recorder events
# ---------------------------------------------------------------------------

class TestRecorderEvents:

    def test_admiral_plan_event(self):
        recorder = BattleRecorder()
        recorder._is_recording = True
        recorder.record_admiral_plan(
            timestamp=60.0, admiral_name="A", faction="alpha", plan="Phases...",
        )
        events = [e for e in recorder.events if e.event_type == EventType.ADMIRAL_PLAN]
        assert len(events) == 1
        assert events[0].data["plan"] == "Phases..."

    def test_captain_log_event(self):
        recorder = BattleRecorder()
        recorder._is_recording = True
        recorder.record_captain_log(
            timestamp=60.0, ship_id="alpha_1", captain_name="C", note="Remember X",
        )
        events = [e for e in recorder.events if e.event_type == EventType.CAPTAIN_LOG]
        assert len(events) == 1
        assert events[0].ship_id == "alpha_1"
        assert events[0].data["note"] == "Remember X"
