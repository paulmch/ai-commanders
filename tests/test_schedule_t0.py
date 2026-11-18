"""
Pass-5 schedule (owner: "Move to T+0"): checkpoint 1 at T+0, checkpoint k at
(k-1) x interval, every checkpoint's orders fly a full interval (the last one
too), so N checkpoints end at T+N x interval and no model call buys orders
that never fly. Covers the base runner (fleet and duel), the live draft runner,
the clock the admirals read, and replay of old recordings.
"""
import json
from pathlib import Path

import pytest

from src.llm import fleet_intel as intel
from tests.test_anthropic_client import _battle


def _decision_times(events, kind="captain_decision"):
    return sorted({e["timestamp"] for e in events if e["event_type"] == kind})


def test_battle_clock_counts_from_t0_and_flies_the_last_orders():
    assert intel.battle_clock(600, 20, 30) == {"checkpoints": 20, "end_s": 600.0, "interval_s": 30.0}
    # Time limit binding: checkpoints at 0..1170 (40), ending at the limit.
    assert intel.battle_clock(1200, 40, 30) == {"checkpoints": 40, "end_s": 1200.0, "interval_s": 30.0}
    assert intel.battle_clock(1200, 60, 30)["checkpoints"] == 40
    assert intel.battle_clock(900, 25, 30) == {"checkpoints": 25, "end_s": 750.0, "interval_s": 30.0}
    assert intel.battle_clock(600, 20, 30, unlimited=True) is None
    assert intel.clock_text(570, 20, intel.battle_clock(600, 20, 30)) == \
        ", checkpoint 20 of 20, battle ends at T+600s"


def test_live_runner_decides_at_t0_and_ends_one_interval_after_the_last_checkpoint(tmp_path):
    record = []
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-haiku-5-5",
                                           record=record, turns=3)
    runner.run_battle(data)
    recording = json.loads(output.read_text())
    events = recording["events"]
    assert _decision_times(events) == [0.0, 30.0, 60.0]
    assert _decision_times(events, "admiral_directive") == [0.0, 30.0, 60.0]
    assert recording["first_checkpoint_s"] == 0.0
    assert runner.simulation.current_time == pytest.approx(90.0)     # last orders flew 30 s
    assert recording["total_checkpoints"] == 3
    # No call after the last checkpoint: every model call is at a checkpoint time.
    calls = {e["timestamp"] for e in events if e["event_type"] == "model_call"}
    assert calls <= {0.0, 30.0, 60.0}


def test_base_fleet_runner_flies_the_last_orders_instead_of_paying_for_unflown_ones(tmp_path):
    from scripts import run_llm_battle
    from src.recording_io import load_recording
    config = tmp_path / "fleet.json"
    config.write_text(json.dumps({
        "battle_name": "t0", "time_limit_s": 1200, "decision_interval_s": 30, "initial_distance_km": 500,
        "max_checkpoints": 4, "seed": 5, "personality_selection": False,
        "alpha_fleet": {"admiral": "claude-haiku-5-5", "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]},
        "beta_fleet": {"admiral": "claude-haiku-5-5", "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]},
    }))
    assert run_llm_battle.main(["--fleet-config", str(config), "--mock-llm", "--quiet",
                                "--recording-dir", str(tmp_path / "rec")]) == 0
    recording = load_recording(next((tmp_path / "rec").glob("battle_*")))
    events = recording["events"]
    assert _decision_times(events) == [0.0, 30.0, 60.0, 90.0]
    assert recording["duration_s"] == pytest.approx(120.0)
    assert max(e["timestamp"] for e in events if e["event_type"] == "model_call") == 90.0


def test_heuristic_duel_and_fleet_follow_the_same_schedule(tmp_path):
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    config = BattleConfig(max_checkpoints=3, time_limit_s=1200, verbose=False, record_battle=False,
                          personality_selection=False, seed=3)
    runner = LLMBattleRunner(config, LLMCaptainConfig("A", "A", model="heuristic"),
                             LLMCaptainConfig("B", "B", model="heuristic"), None)
    runner.run_battle(load_fleet_data())
    assert runner.checkpoint_count == 3
    assert runner.simulation.current_time == pytest.approx(90.0)


def test_first_admiral_checkpoint_has_no_t_minus_15_snapshot_and_still_decides(tmp_path):
    record = []
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5-5",
                                           record=record, turns=2)
    runner.run_battle(data)
    first = [b for b in record if "set_fleet_directive" in {t["name"] for t in b.get("tools") or []}]
    assert len(first) == 4                                         # 2 sides x 2 checkpoints
    text = first[0]["messages"][-1]["content"][0]["text"]
    assert "ADMIRAL CHECKPOINT 1" in text and "checkpoint 1 of 2" in text
    assert "battle ends at T+60s" in text


def test_replay_numbers_checkpoints_of_new_and_old_recordings(tmp_path):
    from scripts import replay_turn as rt
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5-5", turns=2)
    runner.run_battle(data)
    calls = rt.load_calls([output])
    assert {c["checkpoint"] for c in calls if c["role"] == "captain"} == {1, 2}
    assert {c["t"] for c in calls if c["role"] == "captain" and c["checkpoint"] == 1} == {0.0}
    # An old recording (no first_checkpoint_s): checkpoint 1 one interval in.
    old = json.loads(output.read_text())
    old.pop("first_checkpoint_s")
    for e in old["events"]:
        e["timestamp"] += 30.0
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    assert {c["checkpoint"] for c in rt.load_calls([path]) if c["role"] == "captain"} == {1, 2}


# ---------------------------------------------------------------------------
# Default effort per model, score line on by default, launch safety
# ---------------------------------------------------------------------------

def test_default_effort_is_per_model_and_overrides_still_win(tmp_path):
    from src.llm.anthropic_client import BATTLE_DEFAULT_EFFORT, anthropic_effort
    assert BATTLE_DEFAULT_EFFORT == {"claude-fable-5-2": "high", "claude-opus-5-5": "high",
                                     "claude-sonnet-5-5": "high", "claude-sonnet-5": "medium",
                                     "claude-haiku-5-5": "medium"}
    assert anthropic_effort(None, "claude-sonnet-5-5") == "high"
    assert anthropic_effort(None, "claude-haiku-5-5") == "medium"
    assert anthropic_effort("low", "claude-opus-5-5") == "low"
    record = []
    # _battle passes reasoning_effort="medium" explicitly; build the default client here.
    import anthropic
    from src.llm.mock_anthropic import mock_anthropic_transport
    from src.llm.providers import create_llm_client
    client = create_llm_client("claude-sonnet-5-5", ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-5-5"],
                               anthropic_options={"api_key": "x", "http_client": anthropic.DefaultHttpxClient(
                                   transport=mock_anthropic_transport(record=record))})
    tools = [{"type": "function", "function": {"name": "act", "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "system", "content": "S " * 600}, {"role": "user", "content": "go"}]
    for model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-5-5"):
        client.decide_with_tools(msgs, tools, model=model)
    assert [r["output_config"]["effort"] for r in record] == ["high", "high", "medium", "medium"]
    client.decide_with_tools(msgs, tools, model="claude-sonnet-5-5", reasoning_effort="low")
    assert record[-1]["output_config"]["effort"] == "low"


def test_score_line_is_on_by_default_and_none_turns_it_off(tmp_path):
    from types import SimpleNamespace
    from scripts.run_llm_battle import apply_model_overrides
    from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
    assert AdmiralConfig(model="m").show_score is True
    data = {"alpha_fleet": {"admiral": "claude-haiku-5-5", "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]},
            "beta_fleet": {"admiral": {"model": "claude-haiku-5-5", "show_score": False},
                           "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]}}
    fc = BattleFleetConfig.from_dict(data)
    assert fc.alpha_fleet.admiral.show_score and not fc.beta_fleet.admiral.show_score
    blank = dict(no_admiral=None, intent_only=None, alpha_captain_model=None, beta_captain_model=None,
                 alpha_admiral_model=None, beta_admiral_model=None)
    fc = BattleFleetConfig.from_dict(data)
    apply_model_overrides(fc, SimpleNamespace(show_score="none", **blank))
    assert not fc.alpha_fleet.admiral.show_score and not fc.beta_fleet.admiral.show_score
    fc = BattleFleetConfig.from_dict(data)
    apply_model_overrides(fc, SimpleNamespace(show_score="beta", **blank))
    assert not fc.alpha_fleet.admiral.show_score and fc.beta_fleet.admiral.show_score
    # In a battle: every directive turn carries the score line with the draw verdict; no captain turn does.
    record = []
    runner, client, output, data2 = _battle(tmp_path, "claude-haiku-5-5", "claude-haiku-5-5", record=record, turns=1)
    runner.run_battle(data2)
    texts = [(set(t["name"] for t in r.get("tools") or []), r["messages"][-1]["content"][0]["text"]) for r in record]
    directive = [t for names, t in texts if "set_fleet_directive" in names]
    captain = [t for names, t in texts if "set_maneuver" in names]
    assert directive and all("SCORE IF THE BATTLE ENDED NOW" in t for t in directive)
    assert captain and not any("SCORE IF THE BATTLE ENDED NOW" in t for t in captain)


def test_no_default_models_and_require_provider(tmp_path, capsys):
    import subprocess
    import sys
    from scripts import run_llm_battle
    from src.llm.providers import check_required_provider
    # run_llm_battle: a duel without both models stops before anything.
    assert run_llm_battle.main(["--quiet"]) == 2
    assert "no default models" in capsys.readouterr().err
    with pytest.raises(ValueError, match="another provider"):
        check_required_provider("anthropic", ["claude-sonnet-5-5", "openai/gpt-5.6-luna", "heuristic"])
    check_required_provider("anthropic", ["claude-sonnet-5-5", "heuristic"])
    config = tmp_path / "f.json"
    config.write_text(json.dumps({
        "require_provider": "anthropic", "max_checkpoints": 1, "personality_selection": False,
        "alpha_fleet": {"admiral": "claude-haiku-5-5", "ships": [{"ship_type": "frigate", "model": "openai/gpt-5.6-luna"}]},
        "beta_fleet": {"admiral": "claude-haiku-5-5", "ships": [{"ship_type": "frigate", "model": "claude-haiku-5-5"}]}}))
    assert run_llm_battle.main(["--fleet-config", str(config), "--quiet"]) == 1
    assert "another provider" in capsys.readouterr().err
    # run_live_draft_battle: no models and no --mock-llm/--offline/--rosters -> refused at parse time.
    out = subprocess.run([sys.executable, "scripts/run_live_draft_battle.py", "--name", "p5_cmdrun_never",
                          "--no-linger"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and "no default models" in out.stderr
    out = subprocess.run([sys.executable, "scripts/run_live_draft_battle.py", "--name", "p5_cmdrun_never",
                          "--alpha-model", "claude-sonnet-5-5", "--beta-model", "z-ai/glm-5.3-flash",
                          "--require-provider", "anthropic", "--no-linger"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and "another provider" in out.stderr
    assert not Path("data/recordings/p5_cmdrun_never").exists()
