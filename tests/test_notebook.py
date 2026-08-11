"""
Tests for commander notebooks: cross-battle memory for admirals and captains.

Covers storage, status gating, prompt rendering/injection at both command
levels, the battle-digest builder, lesson distillation, and the runner opt-in.

No network access: distillation uses a mock client.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.llm import notebook as nb
from src.llm.battle_runner import BattleConfig, LLMBattleRunner
from src.llm.captain import LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import BattleFleetConfig
from src.llm.prompts import CaptainPersonality, build_admiral_prompt, build_captain_prompt


FLEET_DATA = json.loads(
    (Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text()
)

MODEL = "test/some-model-5.1"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestStorage:

    def test_slug_is_filesystem_safe(self):
        assert nb.model_slug("anthropic/claude-sonnet-5") == "anthropic_claude-sonnet-5"
        assert "/" not in nb.model_slug("a/b:c d")

    def test_add_and_reload_roundtrip(self, tmp_path):
        entry = nb.add_entry(MODEL, "Hold torpedo reserves.", role="admiral",
                             source_battle="battle.json", source_outcome="lost vs x",
                             notebook_dir=tmp_path)
        loaded = nb.load_notebook(MODEL, notebook_dir=tmp_path)
        assert loaded["model"] == MODEL
        assert [e["id"] for e in loaded["entries"]] == [entry["id"]]
        assert loaded["entries"][0]["status"] == "pending"

    def test_status_transitions(self, tmp_path):
        entry = nb.add_entry(MODEL, "Lesson.", notebook_dir=tmp_path)
        nb.set_status(MODEL, entry["id"], "accepted", notebook_dir=tmp_path,
                      validation={"with_wins": 2})
        reloaded = nb.get_entry(MODEL, entry["id"], notebook_dir=tmp_path)
        assert reloaded["status"] == "accepted"
        assert reloaded["validation"]["with_wins"] == 2

        with pytest.raises(KeyError):
            nb.set_status(MODEL, "nope", "accepted", notebook_dir=tmp_path)
        with pytest.raises(ValueError):
            nb.set_status(MODEL, entry["id"], "bogus", notebook_dir=tmp_path)

    def test_entry_validation(self, tmp_path):
        with pytest.raises(ValueError):
            nb.add_entry(MODEL, "  ", notebook_dir=tmp_path)
        with pytest.raises(ValueError):
            nb.add_entry(MODEL, "x", role="emperor", notebook_dir=tmp_path)
        long = nb.add_entry(MODEL, "y" * 1000, notebook_dir=tmp_path)
        assert len(long["text"]) == nb.MAX_ENTRY_CHARS + 3


# ---------------------------------------------------------------------------
# Prompt rendering and gating
# ---------------------------------------------------------------------------

class TestPromptText:

    def test_only_accepted_entries_render(self, tmp_path):
        pending = nb.add_entry(MODEL, "Pending lesson.", role="admiral",
                               notebook_dir=tmp_path)
        nb.add_entry(MODEL, "Accepted lesson.", role="admiral",
                     status="accepted", notebook_dir=tmp_path)
        text = nb.notebook_prompt_text(MODEL, "admiral", notebook_dir=tmp_path)
        assert "Accepted lesson." in text
        assert "Pending lesson." not in text
        assert "COMMANDER'S NOTEBOOK" in text

        # The rematch gate can force-include a pending candidate.
        forced = nb.notebook_prompt_text(MODEL, "admiral", notebook_dir=tmp_path,
                                         include_ids=[pending["id"]])
        assert "Pending lesson." in forced

    def test_role_filtering(self, tmp_path):
        nb.add_entry(MODEL, "Admiral-only.", role="admiral", status="accepted",
                     notebook_dir=tmp_path)
        nb.add_entry(MODEL, "Captain-only.", role="captain", status="accepted",
                     notebook_dir=tmp_path)
        nb.add_entry(MODEL, "Everyone.", role="any", status="accepted",
                     notebook_dir=tmp_path)

        admiral_text = nb.notebook_prompt_text(MODEL, "admiral", notebook_dir=tmp_path)
        captain_text = nb.notebook_prompt_text(MODEL, "captain", notebook_dir=tmp_path)
        assert "Admiral-only." in admiral_text and "Captain-only." not in admiral_text
        assert "Captain-only." in captain_text and "Admiral-only." not in captain_text
        assert "Everyone." in admiral_text and "Everyone." in captain_text

    def test_empty_notebook_renders_none(self, tmp_path):
        assert nb.notebook_prompt_text("test/unknown", "admiral",
                                       notebook_dir=tmp_path) is None

    def test_capped_at_newest_entries(self, tmp_path):
        for i in range(10):
            nb.add_entry(MODEL, f"Lesson {i}.", status="accepted",
                         notebook_dir=tmp_path)
        text = nb.notebook_prompt_text(MODEL, "captain", notebook_dir=tmp_path)
        assert "Lesson 9." in text
        assert "Lesson 0." not in text


class TestPromptInjection:

    NOTE = "COMMANDER'S NOTEBOOK test block: hold reserves until PD is measured."

    def test_captain_prompt_carries_notebook(self):
        prompt = build_captain_prompt(
            captain_name="C", ship_name="TIS X",
            ship_status={}, tactical_status={},
            personality=CaptainPersonality.BALANCED,
            personality_text="Cold and precise.",
            notebook_text=self.NOTE,
        )
        assert self.NOTE in prompt
        # It must live in the cacheable doctrine prefix, before the volatile turn.
        assert prompt.index(self.NOTE) < prompt.index("=== TACTICAL DATA")

    def test_admiral_prompt_carries_notebook(self):
        prompt = build_admiral_prompt(
            admiral_name="A", faction="alpha",
            snapshot_t_minus_15=None, snapshot_t_zero=None,
            personality=None, fleet_data=FLEET_DATA,
            notebook_text=self.NOTE,
        )
        assert self.NOTE in prompt

    def test_runner_injects_only_when_opted_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nb, "DEFAULT_NOTEBOOK_DIR", tmp_path)
        nb.add_entry("stub/model", "Captain lesson.", role="captain",
                     status="accepted", notebook_dir=tmp_path)
        nb.add_entry("stub/model", "Admiral lesson.", role="admiral",
                     status="accepted", notebook_dir=tmp_path)

        def build_runner(use_notebooks: bool) -> LLMBattleRunner:
            fleet_config = BattleFleetConfig.from_dict({
                "battle_name": "NB Test",
                "use_notebooks": use_notebooks,
                "alpha_fleet": {
                    "admiral": {"model": "stub/model"},
                    "ships": [{"ship_type": "destroyer", "model": "stub/model"}],
                },
                "beta_fleet": {
                    "ships": [{"ship_type": "destroyer", "model": "stub/model"}],
                },
            })
            dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/model",
                                     personality=CaptainPersonality.BALANCED)
            runner = LLMBattleRunner(
                config=BattleConfig(verbose=False, record_battle=False,
                                    personality_selection=False),
                alpha_config=dummy, beta_config=dummy,
                client=Mock(), fleet_config=fleet_config,
            )
            runner.setup_fleet_battle(FLEET_DATA)
            return runner

        on = build_runner(use_notebooks=True)
        captain = next(iter(on.alpha_captains.values()))
        assert "Captain lesson." in captain.config.notebook_text
        assert "Admiral lesson." not in captain.config.notebook_text
        assert "Admiral lesson." in on.alpha_admiral.config.notebook_text

        off = build_runner(use_notebooks=False)
        assert next(iter(off.alpha_captains.values())).config.notebook_text is None
        assert off.alpha_admiral.config.notebook_text is None

    def test_fleet_config_parses_flag(self):
        data = {
            "alpha_fleet": {"ships": [{"ship_type": "destroyer"}]},
            "beta_fleet": {"ships": [{"ship_type": "destroyer"}]},
        }
        assert BattleFleetConfig.from_dict(data).use_notebooks is False
        assert BattleFleetConfig.from_dict(
            {**data, "use_notebooks": True}).use_notebooks is True


# ---------------------------------------------------------------------------
# Digest + distillation
# ---------------------------------------------------------------------------

def make_recording() -> dict:
    return {
        "winner": "beta",
        "result_reason": "Alpha fleet destroyed",
        "total_checkpoints": 3,
        "alpha_ships_remaining": 0,
        "beta_ships_remaining": 2,
        "alpha_fleet": {
            "admiral": {"name": "Admiral A", "model": "test/alpha-admiral"},
            "ships": [{"ship_id": "alpha_1", "ship_type": "corvette"},
                      {"ship_id": "alpha_2", "ship_type": "corvette"}],
        },
        "beta_fleet": {
            "admiral": {"name": "Admiral B", "model": "test/beta-admiral"},
            "ships": [{"ship_id": "beta_1", "ship_type": "destroyer"},
                      {"ship_id": "beta_2", "ship_type": "destroyer"}],
        },
        "events": [
            {"timestamp": 30.0, "event_type": "admiral_plan", "ship_id": None,
             "data": {"faction": "alpha", "plan": "Phase 1: rush. Phase 2: pray."}},
            {"timestamp": 30.0, "event_type": "admiral_directive", "ship_id": None,
             "data": {"faction": "beta", "directive": "Kite and shred them."}},
            {"timestamp": 45.0, "event_type": "torpedo_launched", "ship_id": "alpha_1",
             "data": {"source_ship_id": "alpha_1"}},
            {"timestamp": 45.0, "event_type": "torpedo_launched", "ship_id": "alpha_2",
             "data": {"source_ship_id": "alpha_2"}},
            {"timestamp": 70.0, "event_type": "torpedo_impact", "ship_id": "beta_1",
             "data": {"target_ship": "beta_1", "damage_gj": 24.0}},
            {"timestamp": 90.0, "event_type": "module_destroyed", "ship_id": "alpha_1",
             "data": {"module": "reactor"}},
        ],
    }


class TestDigest:

    def test_digest_narrates_both_sides_and_result(self):
        digest = nb.build_battle_digest(make_recording(), side="alpha")
        assert "you commanded ALPHA" in digest
        assert "[YOU plan] Phase 1: rush" in digest
        assert "[ENEMY directive] Kite and shred" in digest
        assert "[SALVO] 2 torpedo(es)" in digest
        assert "[IMPACT] torpedo hit beta_1" in digest
        assert "YOU LOST" in digest
        digest_beta = nb.build_battle_digest(make_recording(), side="beta")
        assert "YOU WON" in digest_beta

    def test_digest_is_bounded(self):
        recording = make_recording()
        recording["events"] = recording["events"] * 500
        digest = nb.build_battle_digest(recording, side="alpha", max_chars=4000)
        assert len(digest) < 6000
        assert "trimmed" in digest

    def test_side_inference(self):
        recording = make_recording()
        assert nb.infer_side_for_model(recording, "test/alpha-admiral") == "alpha"
        assert nb.infer_side_for_model(recording, "test/beta-admiral") == "beta"
        assert nb.infer_side_for_model(recording, "test/other") is None


class TestDistill:

    def test_lessons_extracted_from_tool_call(self):
        client = Mock()
        client.decide_with_tools.return_value = [
            ToolCall(id="1", name="record_lessons", arguments={"lessons": [
                {"text": "Mass torpedoes in one salvo.", "role": "admiral"},
                {"text": "", "role": "admiral"},                 # dropped: empty
                {"text": "Bad role falls back.", "role": "odd"},  # role -> admiral
                {"text": "Over the cap.", "role": "any"},        # dropped: cap 2
            ]}),
        ]
        lessons = nb.distill_lessons(client, MODEL, "admiral", "digest...")
        assert lessons == [
            {"text": "Mass torpedoes in one salvo.", "role": "admiral"},
            {"text": "Bad role falls back.", "role": "admiral"},
        ]
        # The analysis runs on the refined model itself by default.
        assert client.decide_with_tools.call_args.kwargs["model"] == MODEL

    def test_no_tool_call_means_no_lessons(self):
        client = Mock()
        client.decide_with_tools.return_value = []
        assert nb.distill_lessons(client, MODEL, "captain", "digest...") == []
