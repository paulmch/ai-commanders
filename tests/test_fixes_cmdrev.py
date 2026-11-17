"""Fixes from the independent review of the command-interface wave (lane cmdrev).

Each test fails on 663f8e3 (the merged pass-5 tree before the review)."""
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from src.llm.spending import RequestBudget, SpendLedger

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Money: an interrupted run sends nothing new and the ledger keeps what it may owe
# ---------------------------------------------------------------------------

def test_a_stopped_budget_refuses_every_later_reservation_even_after_settling():
    budget = RequestBudget(10.0, {})
    reserved = budget.reserve_usd(1.0)
    budget.stop("Ctrl-C")
    assert budget.exhausted and budget.stopped == "Ctrl-C"
    with pytest.raises(RuntimeError):
        budget.reserve_usd(0.01)
    budget.settle(reserved, 0.05)           # the call in flight still settles...
    with pytest.raises(RuntimeError):       # ...and freeing its reservation reopens nothing
        budget.reserve_usd(0.01)


def test_ledger_counts_a_closed_run_at_its_committed_dollars_when_larger(tmp_path):
    """A run that closed with calls whose cost never came back (timeouts, calls in
    flight at an interrupt) may still be billed for them: the ledger keeps them."""
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    run = ledger.open_run(script="t", name="a", models=["m"], cap_usd=5.0)
    ledger.close_run(run, calls=3, tokens={}, usd=1.0, committed_usd=1.75, status="interrupted")
    assert ledger.total_usd() == pytest.approx(1.75)
    done = ledger.open_run(script="t", name="b", models=["m"], cap_usd=5.0)
    ledger.close_run(done, calls=3, tokens={}, usd=0.5, committed_usd=0.5)
    assert ledger.total_usd() == pytest.approx(2.25)


def test_replay_interrupt_drops_the_queued_requests_instead_of_paying_for_all_of_them(monkeypatch):
    """Ctrl-C during a replay: pool.map had queued every job and the executor's
    shutdown ran them all (paid) before re-raising. Now the queue is dropped
    and the budget refuses anything new."""
    from scripts import replay_turn as rt
    sent = []

    def send(client, job):
        sent.append(job["n"])
        if job["n"] == 0:
            raise KeyboardInterrupt
        return {"n": job["n"], "error": None, "signature": "", "source": {"key": "k"},
                "variant": {"label": "v"}, "replicate": 0, "cost": 0.0, "latency_s": 0.0}

    monkeypatch.setattr(rt, "send", send)
    client = type("C", (), {})()
    client.budget = RequestBudget(5.0, {})
    jobs = [{"n": i, "prefix": "p"} for i in range(12)]
    with pytest.raises(KeyboardInterrupt):
        rt.run_jobs(client, jobs, workers=1)
    assert len(sent) == 1
    assert client.budget.stopped


def test_live_battle_interrupted_by_ctrl_c_stops_buying_calls_and_closes_its_ledger_line(tmp_path):
    """One SIGINT to run_live_draft_battle: the battle thread used to keep
    playing (and paying) while the ledger line never closed. Now: no call after
    the interrupt, the ledger closes 'interrupted' with the calls the recording
    holds, and the recording says why it stopped."""
    name = f"p5_cmdrev_test_int_{os.getpid()}"
    ledger = tmp_path / "ledger.jsonl"
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    proc = subprocess.Popen(
        [sys.executable, "scripts/run_live_draft_battle.py", "--mock-llm", "--alpha-model",
         "claude-haiku-5-5", "--beta-model", "claude-haiku-5-5", "--points", "100", "--turns", "30",
         "--spend-limit", "5", "--ledger", str(ledger), "--name", name, "--no-linger", "--quiet",
         "--port", str(18000 + os.getpid() % 1000), "--sim-step-delay", "0.02"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = ROOT / "visualizer/public/recordings" / f"{name}.json"
    report_dir = ROOT / "data/recordings" / name
    try:
        for line in proc.stdout:            # blocking read: wait for checkpoint 3
            if line.startswith("=== CP 3 "):
                break
        proc.send_signal(signal.SIGINT)
        rest = proc.stdout.read()
        assert proc.wait(timeout=300) != 0
        lines = [json.loads(x) for x in ledger.read_text().splitlines()]
        close = [x for x in lines if x["event"] == "close"]
        assert close and close[0]["status"] == "interrupted", rest[-2000:]
        recording = json.loads(output.read_text())
        calls = [e for e in recording["events"] if e["event_type"] == "model_call"]
        assert close[0]["calls"] == len(calls)
        assert recording["result_reason"].startswith("Interrupted")
        assert recording["duration_s"] < 30 * 30
        status = json.loads((report_dir / "status.json").read_text())
        assert status["finished"] and status["error"].startswith("Interrupted")
    finally:
        if proc.poll() is None:
            proc.kill()
        for path in (output, output.with_suffix(".draft.json")):
            path.unlink(missing_ok=True)
        if report_dir.exists():
            import shutil
            shutil.rmtree(report_dir)
