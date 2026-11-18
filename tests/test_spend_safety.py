"""Spend safety: the price table, the dollar cap in fixed-fleet battles, the cumulative ledger."""
import glob
import json
from pathlib import Path

import pytest

from scripts import run_llm_battle
from src.llm.anthropic_client import ANTHROPIC_MODELS, reservation_usd
from src.llm.spending import LedgerCapExceeded, RequestBudget, SpendLedger
from src.recording_io import load_recording

EXPERIMENT_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5")


def test_price_table_matches_the_claude_api_skill_for_every_experiment_model():
    """
    USD per million tokens (claude-api skill, "Current Models", cached
    2026-11-03; shared/prompt-caching.md: 5-minute writes 1.25x input,
    1-hour writes 2x, reads 0.1x except Sonnet 5.5 / Opus 5.5 at 0.05x).
    """
    expected = {
        #                    input  output cache_read  write_5m  write_1h
        "claude-opus-5-5":   (4.00, 20.00, 0.20, 5.00, 8.00),
        "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50, 4.00),
        "claude-sonnet-5":   (2.00, 10.00, 0.20, 2.50, 4.00),
        "claude-haiku-5-5":  (0.10, 0.50, 0.01, 0.125, 0.20),
    }
    for model, (inp, out, read, w5, w1) in expected.items():
        spec = ANTHROPIC_MODELS[model]
        assert (spec["input"], spec["output"], spec["cache_read"],
                spec["cache_write_5m"], spec["cache_write_1h"]) == (inp, out, read, w5, w1), model
    # Haiku 5.5 above 100K prompt tokens: $0.50 / $2.50 (stated); the cache
    # rates on that card apply the standard multipliers (an inference).
    long = ANTHROPIC_MODELS["claude-haiku-5-5"]["long"]
    assert (long["input"], long["output"]) == (0.50, 2.50)
    assert (long["cache_read"], long["cache_write_5m"], long["cache_write_1h"]) == (0.05, 0.625, 1.00)


def test_a_model_missing_from_the_table_fails_before_the_first_request():
    import anthropic
    from src.llm.mock_anthropic import mock_anthropic_transport
    from src.llm.providers import create_llm_client
    record = []
    with pytest.raises(ValueError, match="not in the price table"):
        create_llm_client("claude-sonnet-5-5", ["claude-opus-9"], anthropic_options={
            "api_key": "x", "http_client": anthropic.DefaultHttpxClient(
                transport=mock_anthropic_transport(record=record))})
    assert record == []


# ---------------------------------------------------------------------------
# The cap in fixed-fleet battles (run_llm_battle.py --spend-limit, base runner)
# ---------------------------------------------------------------------------

def _config(tmp_path, alpha_admiral, alpha_captains, beta_admiral, beta_captains, turns=10):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({
        "battle_name": "cap test", "time_limit_s": (turns + 1) * 30, "decision_interval_s": 30,
        "initial_distance_km": 500, "max_checkpoints": turns, "seed": 7, "personality_selection": False,
        "alpha_fleet": {"admiral": alpha_admiral, "ships": [
            {"ship_type": "frigate", "model": alpha_captains}, {"ship_type": "destroyer", "model": alpha_captains}]},
        "beta_fleet": {"admiral": beta_admiral, "ships": [
            {"ship_type": "frigate", "model": beta_captains}, {"ship_type": "destroyer", "model": beta_captains}]},
    }))
    return path


def _run(tmp_path, config, *extra):
    rec_dir = tmp_path / "rec"
    code = run_llm_battle.main(["--fleet-config", str(config), "--mock-llm", "--quiet",
                                "--max-tokens", "8192", "--recording-dir", str(rec_dir), *extra])
    files = glob.glob(str(rec_dir / "battle_*"))
    return code, (load_recording(files[0]) if files else None)


def _largest_reservation(recording):
    """Largest cap reservation any request of this battle made (8192 output tokens)."""
    assets = recording["assets"]
    best = 0.0
    for e in recording["events"]:
        if e["event_type"] == "model_call":
            d = e["data"]
            payload = {"messages": [assets[r] for r in d["message_refs"]], "tools": assets[d["tools_ref"]]}
            best = max(best, reservation_usd(d["model"], payload, 8192)[0])
    return best


def _probe(tmp_path, *models):
    """Uncapped 10-checkpoint run with realistic thinking output: (total $, largest reservation)."""
    code, probe = _run(tmp_path / "probe", _config(tmp_path / "probe", *models), "--spend-limit", "1000",
                       "--mock-thinking-tokens", "1500")
    assert code == 0 and probe["total_checkpoints"] == 10
    total = sum(e["data"]["usage"]["cost"] for e in probe["events"] if e["event_type"] == "model_call")
    largest = _largest_reservation(probe)
    assert total > largest     # a cap between the two must stop the battle
    return total, largest


@pytest.mark.parametrize("model", EXPERIMENT_MODELS)
def test_spend_cap_stops_a_fixed_fleet_battle_for_each_model(tmp_path, model):
    total, largest = _probe(tmp_path, model, model, model, model)
    limit = (total + largest) / 2
    config = _config(tmp_path, model, model, model, model)
    code, recording = _run(tmp_path, config, "--spend-limit", f"{limit:.6f}",
                           "--mock-thinking-tokens", "1500", "--ledger", str(tmp_path / "ledger.jsonl"))
    assert code == 0
    stopped = [e for e in recording["events"] if e["event_type"] == "battle_stopped"]
    assert stopped and "Spending ceiling" in stopped[0]["data"]["reason"]
    assert "Spending ceiling" in recording["result_reason"] and recording["winner"] is None
    assert recording["total_checkpoints"] < 10
    spent = sum(e["data"]["usage"]["cost"] for e in recording["events"] if e["event_type"] == "model_call")
    assert 0 < spent <= limit
    close = [json.loads(line) for line in open(tmp_path / "ledger.jsonl")][-1]
    assert close["event"] == "close" and close["status"] == "stopped"
    assert close["usd"] == pytest.approx(spent, abs=1e-6) and close["mode"] == "mock"


def test_spend_cap_stops_a_battle_with_mixed_models_on_one_side(tmp_path):
    """Opus admiral over Haiku captains vs Sonnet 5.5 admiral over Sonnet 5 captains: one cap."""
    models = ("claude-opus-5-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5")
    total, largest = _probe(tmp_path, *models)
    limit = (total + largest) / 2
    code, recording = _run(tmp_path, _config(tmp_path, *models), "--spend-limit", f"{limit:.6f}",
                           "--mock-thinking-tokens", "1500")
    assert code == 0 and recording["total_checkpoints"] < 10
    calls = [e["data"] for e in recording["events"] if e["event_type"] == "model_call"]
    seen = {(d["faction"], d["role"], d["model"]) for d in calls}
    assert ("alpha", "directive", "claude-opus-5-5") in seen
    assert ("alpha", "captain", "claude-haiku-5-5") in seen
    assert ("beta", "captain", "claude-sonnet-5") in seen
    assert sum(d["usage"]["cost"] for d in calls) <= limit
    assert any(e["event_type"] == "battle_stopped" for e in recording["events"])


def test_uncapped_and_generously_capped_battles_are_not_stopped(tmp_path):
    config = _config(tmp_path, "claude-haiku-5-5", "claude-haiku-5-5", "claude-haiku-5-5",
                     "claude-haiku-5-5", turns=2)
    code, recording = _run(tmp_path, config)
    assert code == 0 and recording["total_checkpoints"] == 2
    assert not any(e["event_type"] == "battle_stopped" for e in recording["events"])


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_ledger_reserves_open_runs_at_their_cap_and_refuses_past_the_ledger_cap(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    a = ledger.open_run(script="t", name="a", models=["m"], cap_usd=5.0)
    assert ledger.total_usd() == pytest.approx(5.0)              # open: counted at its cap
    ledger.close_run(a, calls=10, tokens={}, usd=1.25)
    assert ledger.total_usd() == pytest.approx(1.25)             # closed: counted at its spend
    b = ledger.open_run(script="t", name="b", models=["m"], cap_usd=3.0, ledger_cap_usd=5.0)
    assert ledger.total_usd() == pytest.approx(4.25)
    with pytest.raises(LedgerCapExceeded):
        ledger.open_run(script="t", name="c", models=["m"], cap_usd=1.0, ledger_cap_usd=5.0)
    m = ledger.open_run(script="t", name="mock", models=["m"], cap_usd=100.0, mock=True)
    assert ledger.total_usd() == pytest.approx(4.25)             # mock runs never count
    ledger.close_run(m, calls=1, tokens={}, usd=50.0)
    assert ledger.total_usd() == pytest.approx(4.25)
    ledger.close_run(b, calls=1, tokens={}, usd=0.5)
    lines = [json.loads(x) for x in Path(tmp_path / "ledger.jsonl").read_text().splitlines()]
    close = [x for x in lines if x["event"] == "close" and x["name"] == "a"][0]
    assert {"ts", "script", "name", "models", "calls", "tokens", "usd", "cap_usd"} <= set(close)


def test_ledger_cap_refuses_a_battle_before_its_first_request(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = SpendLedger(path)
    ledger.close_run(ledger.open_run(script="t", name="old", models=["m"], cap_usd=9.0), calls=1,
                     tokens={}, usd=9.0)
    lines_before = path.read_text().count("\n")
    config = _config(tmp_path, "claude-haiku-5-5", "claude-haiku-5-5", "claude-haiku-5-5",
                     "claude-haiku-5-5", turns=2)
    code, recording = _run(tmp_path, config, "--spend-limit", "2", "--ledger", str(path),
                           "--ledger-cap", "10")
    assert code == 1 and recording is None                       # refused: no battle at all
    assert path.read_text().count("\n") == lines_before          # and nothing reserved
    code, recording = _run(tmp_path, config, "--spend-limit", "0.9", "--ledger", str(path),
                           "--ledger-cap", "10")
    assert code == 0 and recording is not None
    code = run_llm_battle.main(["--fleet-config", str(config), "--mock-llm", "--quiet",
                                "--ledger-cap", "10"])
    assert code == 1                                              # --ledger-cap needs --spend-limit


def test_budget_settles_to_reported_cost():
    budget = RequestBudget(1.0, {})
    reserved = budget.reserve_usd(0.4)
    budget.settle(reserved, 0.1)
    assert budget.committed_usd == pytest.approx(0.1) and budget.reported_usd == pytest.approx(0.1)
