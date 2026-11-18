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


# ---------------------------------------------------------------------------
# What the captain is told about its own last decision
# ---------------------------------------------------------------------------

def _ack_loop(tmp_path, ack):
    from tests.test_fixes_cmdcap import FLEET, _loop_runner

    def captain(ctx):
        if ctx.model != "stub/dd":
            return []
        if ctx.cp == 1 and ctx.turn == 1:
            return [("set_maneuver", {"maneuver_type": "EVADE"})]
        if ctx.cp == 1:
            return [("report_to_admiral", {"status": "ACK"})] if ack else []
        return []
    stub, runner = _loop_runner(tmp_path, captain, checkpoints=2)
    runner.run_fleet_battle(FLEET)
    turn = stub.texts[("stub/dd", 2, 1)]
    i = turn.index("YOUR RECENT DECISIONS")
    return turn[i:turn.index("\n\n", i)]


def test_a_call_withdrawn_by_the_order_check_ack_is_not_shown_as_flown(tmp_path):
    """ACK in the ORDER CHECK withdrew set_maneuver EVADE and the admiral's HEADING
    flew; the next turn said 'T+30 EVADE@100% | report ACK ... - NO COMMANDS: the
    ship coasted' - two false statements."""
    history = _ack_loop(tmp_path, ack=True)
    line = [x for x in history.splitlines() if x.startswith("  T+0 ")][0]
    assert "EVADE" not in line
    assert "coasted" not in line and "your Admiral's maneuver order flew" in line


def test_without_an_ack_the_departing_call_stays_in_the_history(tmp_path):
    history = _ack_loop(tmp_path, ack=False)
    assert "T+0 EVADE@100%" in history


def test_no_command_turn_without_an_admiral_maneuver_still_reads_as_a_coast():
    from src.llm.captain import LLMCaptain
    cap = LLMCaptain.__new__(LLMCaptain)
    cap.decision_history = [{"checkpoint": 1, "time": 30.0, "tool_calls": [], "commands_count": 0,
                             "followups": 0, "acted": False}]
    cap._launch_outcomes = lambda sim: {}
    assert "NO COMMANDS: the ship coasted" in cap._format_decision_history()


# ---------------------------------------------------------------------------
# Quiet turns: a kill or a loss on the datalink is new traffic
# ---------------------------------------------------------------------------

def test_quiet_turns_count_datalink_kills_and_losses_as_new_traffic():
    """quiet_turns promises to skip only with 'no new FLEET NET traffic', but every
    datalink line was filtered out - a captain whose target had just died was
    skipped. Kills and losses now count; launches and target switches do not."""
    from types import SimpleNamespace
    from src.llm.battle_runner import LLMBattleRunner
    from src.llm.call_policy import quiet_turn_reason
    runner = SimpleNamespace(simulation=SimpleNamespace(get_ship=lambda sid: None), _turn_state={})

    def facts(inbox):
        captain = SimpleNamespace(inbox=inbox, recent_hits=None)
        LLMBattleRunner._captain_turn_facts(runner, "alpha_1", captain, None)   # first turn
        return LLMBattleRunner._captain_turn_facts(runner, "alpha_1", captain, None)
    kill = {"sender_role": "engine", "kind": "KILL", "text": "OCS Dart-1 destroyed"}
    launch = {"sender_role": "engine", "kind": "LAUNCH", "text": "launched 1 torpedo"}
    assert facts([kill]).new_messages == 1 and quiet_turn_reason(facts([kill])) is None
    assert facts([launch]).new_messages == 0 and quiet_turn_reason(facts([launch]))


# ---------------------------------------------------------------------------
# Merges: the captain is told what the admiral is told, two-sidedly
# ---------------------------------------------------------------------------

def test_captain_merge_text_is_two_sided_like_the_admirals():
    """The admiral doctrine was re-measured (the pass is short and cuts both ways);
    the captain's still said 'seconds of fire, then flying apart', 'INTERCEPT only
    for...', 'MERGE IN means act THIS checkpoint' and the alarm ended in orders
    ('Cancel closure now, or pass wide.', '... NOW.')."""
    from tests.test_captain_turn import corvette_kwargs
    from src.llm.prompts import build_captain_messages
    system = build_captain_messages(**corvette_kwargs())[0]["content"]
    for steer in ("seconds of fire", "INTERCEPT only for", "act THIS checkpoint"):
        assert steer not in system
    from src.simulation import CombatSimulation
    from tests.test_range_control_and_evade import _coast, _ship
    from src.llm.range_control import analyze_range_control
    for vel, pos in (((2.3, 0, 0), (312, 1.1, 0)), ((4.6, 0, 0), (480, 0.5, 0))):
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0), vel_kps=vel)
        foe = _ship(sim, "foe", "battleship", "beta", pos, vel_kps=(-vel[0] if vel[0] < 3 else 0, 0, 0),
                    fwd=(-1, 0, 0))
        _coast(foe)
        warning = analyze_range_control(sim, [me])[0].warning
        assert warning and warning.startswith("MERGE IN")
        assert "NOW" not in warning and "Cancel closure now" not in warning


def test_engine_order_check_does_not_call_a_merge_seconds_of_fire():
    import json as _json
    import inspect
    from src.llm import captain
    from src.llm.tools import CAPTAIN_TOOLS_BASE
    assert "seconds of fire" not in inspect.getsource(captain)
    assert "seconds of" not in _json.dumps(CAPTAIN_TOOLS_BASE)      # the INTERCEPT tool text too


# ---------------------------------------------------------------------------
# The admiral's 'him:' field on a group whose members show different faces
# ---------------------------------------------------------------------------

def test_him_field_names_a_mixed_group_instead_of_reporting_its_median_as_everyone():
    from types import SimpleNamespace
    from src.llm.fleet_intel import him_after_text
    from src.physics import Vector3D
    own = [SimpleNamespace(position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0), mass_kg=1.0,
                           is_destroyed=False, forward=Vector3D(1, 0, 0))]
    nose = [SimpleNamespace(position=Vector3D(400e3, i * 1e3, 0), velocity=Vector3D(0, 0, 0),
                            forward=Vector3D(-1, 0, 0)) for i in range(3)]
    tail = [SimpleNamespace(position=Vector3D(450e3, i * 1e3, 0), velocity=Vector3D(0, 0, 0),
                            forward=Vector3D(1, 0, 0)) for i in range(5)]
    opt = SimpleNamespace(heading=Vector3D(-1, 0, 0), throttle=0.0, burn_s=0.0, turn_s=0.0, key="cancel")
    info = SimpleNamespace(accel_g=1.5)
    sim = SimpleNamespace(_turn_time_to=lambda *a: 10.0)
    text = him_after_text(sim, info, own, nose + tail, opt)
    assert text.startswith("him (median of 8: 3 NOSE, 5 TAIL now): TAIL")
    assert him_after_text(sim, info, own, tail, opt).startswith("him: TAIL")


# ---------------------------------------------------------------------------
# Built in review: captain battle clock, --show-score, --label
# ---------------------------------------------------------------------------

def test_captain_turn_carries_the_battle_clock_and_knows_its_last_decision(tmp_path):
    from tests.test_fixes_cmdcap import FLEET, _loop_runner
    stub, runner = _loop_runner(tmp_path, lambda ctx: [], checkpoints=2)
    runner.run_fleet_battle(FLEET)
    first, last = stub.texts[("stub/dd", 1, 1)], stub.texts[("stub/dd", 2, 1)]
    # Checkpoint 1 at T+0, checkpoint 2 at T+30; the last orders fly to T+60
    # (the base runner used to stop at its last checkpoint, orders unflown).
    assert "BATTLE CLOCK: checkpoint 1 of 2, battle ends at T+60s." in first
    assert "next decision T+30." in first
    assert "BATTLE CLOCK: checkpoint 2 of 2" in last
    assert "no later decision: the battle ends at T+60." in last
    assert "these orders are not flown" not in last
    assert runner.simulation.current_time == 60.0


def test_show_score_and_label_reach_the_admiral_status_recording_and_ledger(tmp_path):
    from scripts import run_llm_battle
    from src.recording_io import load_recording
    from tests.test_spend_safety import _config
    import glob
    config = _config(tmp_path, "claude-haiku-5-5", "claude-haiku-5-5", "claude-haiku-5-5",
                     "claude-haiku-5-5", turns=2)
    ledger, status = tmp_path / "ledger.jsonl", tmp_path / "status.json"
    code = run_llm_battle.main(["--fleet-config", str(config), "--mock-llm", "--quiet",
                                "--recording-dir", str(tmp_path / "rec"), "--spend-limit", "5",
                                "--ledger", str(ledger), "--status-file", str(status),
                                "--show-score", "alpha", "--label", "exp-7 haiku smoke"])
    assert code == 0
    rec = load_recording(glob.glob(str(tmp_path / "rec" / "battle_*"))[0])
    assert rec["provenance"]["label"] == "exp-7 haiku smoke"
    assets = rec["assets"]
    turns = {}
    for e in rec["events"]:
        d = e["data"] if e["event_type"] == "model_call" else None
        if d and d.get("role") == "directive":
            turns.setdefault(d["faction"], []).append(
                "\n".join(str(assets[r].get("content")) for r in d["message_refs"]))
    assert all("SCORE IF THE BATTLE ENDED NOW" in t for t in turns["alpha"])
    assert not any("SCORE IF THE BATTLE ENDED NOW" in t for t in turns["beta"])
    assert all("exp-7" not in t for side in turns.values() for t in side)   # never shown to a model
    assert json.loads(status.read_text())["label"] == "exp-7 haiku smoke"
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [x["label"] for x in lines] == ["exp-7 haiku smoke"] * 2
    assert run_llm_battle.main(["--fleet-config", str(config), "--mock-llm", "--quiet",
                                "--no-admiral", "beta", "--show-score", "beta"]) == 1


# ---------------------------------------------------------------------------
# Money invariant: every request that leaves the client reserved from the budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [{}, {"cache_warmup": True}])
def test_every_request_sent_reserved_from_the_budget_first(tmp_path, monkeypatch, extra):
    """Pre-warms, follow-ups, retries and per-ship orders included: the count of
    requests the transport saw equals the count of successful reservations (the
    cache warm-up test checked the warm-ups were billed, not that they reserved)."""
    from tests.test_anthropic_client import _battle
    reserved = []
    real = RequestBudget.reserve_usd

    def counting(self, amount, images=0):
        out = real(self, amount, images)
        reserved.append(out)
        return out
    monkeypatch.setattr(RequestBudget, "reserve_usd", counting)
    record = []
    runner, client, output, data = _battle(
        tmp_path, "claude-opus-5-5", "claude-sonnet-5-5", record=record, limit=50.0, turns=2,
        captain_models={"alpha": "claude-haiku-5-5"}, anthropic_extra=extra,
        ships=("battlecruiser", "battlecruiser", "frigate"))
    runner.run_battle(data)
    assert record and len(record) == len(reserved)
    if extra:
        assert any(b["max_tokens"] == 0 for b in record)


# ---------------------------------------------------------------------------
# ORDER CHECK guards no existing test covered (found by removing them)
# ---------------------------------------------------------------------------

def test_a_new_launch_in_the_order_check_reply_does_not_fire(tmp_path):
    """The reply is a fresh request: a model that launches again with other
    arguments (count 2 instead of 1) must not fire more rounds - removing the
    launch filter left every existing test green (identical repeats are also
    dropped by the dedupe)."""
    from tests.test_fleet_comms import FLEET, Stub, make_runner

    def order(ship):
        return {"order_text": "Launch on the destroyer.", "priority": "NORMAL", "suggested_target": "beta_1"}

    def captain(ctx):
        if ctx.model != "stub/dart" or ctx.cp != 1:
            return []
        launch = 1 if ctx.turn == 1 else 2
        return [("set_primary_target", {"target_id": "beta_2"}),
                ("launch_torpedo", {"count": launch, "target_id": "beta_2"})]
    stub = Stub(captain=captain, order=order)
    runner = make_runner(stub, tmp_path, beta=(("destroyer", "stub/lancer"), ("destroyer", "stub/lancer")),
                         checkpoints=1, distance_km=400.0)
    runner.run_fleet_battle(FLEET)
    assert stub.turns[("stub/dart", 1)] == 2               # the ORDER CHECK was asked
    executed = [tc for tc in runner.alpha_captains["alpha_1"].last_tool_calls
                if tc.name == "launch_torpedo"]
    assert [tc.arguments["count"] for tc in executed] == [1]    # only the first reply's launch


def test_order_check_is_asked_once_per_checkpoint_across_a_discussion_redecide(tmp_path):
    """discuss_with_admiral makes the runner call decide() again in the same
    checkpoint; the ORDER CHECK (a paid call) must not be asked again there."""
    from tests.test_fixes_cmdcap import FLEET, _loop_runner

    def captain(ctx):
        if ctx.model != "stub/dd" or ctx.cp != 1:
            return []
        calls = [("set_maneuver", {"maneuver_type": "EVADE"})]
        if ctx.turn == 2:          # the ORDER CHECK reply asks the admiral
            calls.append(("discuss_with_admiral", {"question": "Hold or evade?"}))
        return calls
    stub, runner = _loop_runner(tmp_path, captain, checkpoints=1)
    runner.run_fleet_battle(FLEET)
    checks = [k for k, text in stub.texts.items()
              if k[0] == "stub/dd" and k[1] == 1 and "=== ORDER CHECK" in text]
    assert len(checks) == 1, sorted(stub.texts)


def test_intent_only_issue_order_schema_offers_only_what_that_mode_flies():
    """Intent-only drops every issue_order field but the salvo, yet the schema still
    offered heading / maneuver / throttle / evade with their 'applied at once' texts."""
    from src.llm.admiral_tools import get_admiral_tools
    tool = next(t["function"] for t in get_admiral_tools(intent_only=True)
                if t["function"]["name"] == "issue_order")
    assert set(tool["parameters"]["properties"]) == {"ship_name", "torpedo_salvo", "torpedo_target",
                                                     "torpedo_profile"}
    assert tool["parameters"]["required"] == ["ship_name"]
    full = next(t["function"] for t in get_admiral_tools() if t["function"]["name"] == "issue_order")
    assert "heading" in full["parameters"]["properties"]          # default tools untouched


def test_score_line_states_the_verdict_with_the_runners_draw_margin(tmp_path):
    """--show-score gave the two scores but not the draw margin the runner applies
    (max 5 points, 5% of the higher): 151 vs 145 reads as a win and is a draw."""
    from types import SimpleNamespace
    from src.llm import fleet_intel
    from src.llm.battle_runner import LLMBattleRunner
    for a, b in ((151.0, 151.0), (151.0, 145.0), (151.0, 140.0), (120.0, 151.0)):
        fleet_intel.time_limit_scores, real = (lambda *_: (a, b)), fleet_intel.time_limit_scores
        try:
            line = fleet_intel.score_line(SimpleNamespace(ships={}), "alpha")
            outcome = LLMBattleRunner._evaluate_fleet_tactical_advantage(
                SimpleNamespace(alpha_ships={}, beta_ships={}, config=SimpleNamespace(verbose=False)))
        finally:
            fleet_intel.time_limit_scores = real
        expected = {"DRAW": "a DRAW", "ALPHA_VICTORY": "you win", "BETA_VICTORY": "he wins"}
        assert expected[outcome[0].name] in line, (a, b, line, outcome)


# ---------------------------------------------------------------------------
# Paid battle p5w5_b3: launches exactly as the admiral allocated were filed DEVIATE
# ---------------------------------------------------------------------------

def _salvo_battle(tmp_path, launch_at, reason=True):
    """cp1: the corvette's gun target is beta_1. cp2: the admiral orders a 1-round
    salvo at beta_2 (salvo_plan allocations reach the order the same way); the
    captain launches at `launch_at` without touching its gun target."""
    from tests.test_fleet_comms import FLEET, Stub, make_runner

    def order(ship):
        cp = stub.runner.checkpoint_count
        if cp == 1:
            return {"order_text": "Hold.", "priority": "NORMAL", "suggested_target": "beta_1"}
        return {"order_text": "Launch 1 at beta_2.", "priority": "HIGH", "suggested_target": "beta_2",
                "torpedo_salvo": 1, "torpedo_target": "beta_2"}

    def captain(ctx):
        if ctx.model != "stub/dart":
            return []
        if ctx.cp == 1:
            return [("set_primary_target", {"target_id": "beta_1"})]
        args = {"count": 1, "target_id": launch_at}
        if reason:
            args["reason"] = "Admiral's salvo at beta_2."
        return [("launch_torpedo", args)] if ctx.turn == 1 else []
    stub = Stub(captain=captain, order=order)
    runner = make_runner(stub, tmp_path, beta=(("destroyer", "stub/lancer"), ("destroyer", "stub/lancer")),
                         checkpoints=2, distance_km=400.0)
    runner.run_fleet_battle(FLEET)
    report = [e.data for e in runner.recorder.events
              if e.event_type == "captain_report" and e.ship_id == "alpha_1" and e.timestamp == 30.0][0]
    return stub, report


def test_a_launch_that_matches_the_ordered_salvo_is_compliant_and_its_reason_is_intent(tmp_path):
    stub, report = _salvo_battle(tmp_path, "beta_2")
    assert report["compliance"]["followed"], report["compliance"]
    assert report["status"] == "ACK" and "Admiral's salvo" in report["intent"]
    assert not report.get("reason")


def test_a_compliant_salvo_launch_without_a_reason_gets_no_order_check(tmp_path):
    stub, report = _salvo_battle(tmp_path, "beta_2", reason=False)
    assert stub.turns[("stub/dart", 2)] == 1               # no paid ORDER CHECK follow-up
    assert report["compliance"]["followed"]


def test_a_launch_at_another_hull_than_the_ordered_salvo_is_a_deviation(tmp_path):
    stub, report = _salvo_battle(tmp_path, "beta_1")
    devs = report["compliance"]["deviations"]
    assert any(d.startswith("salvo: ordered at") for d in devs), devs
    assert report["status"] == "DEVIATE"


@pytest.mark.parametrize("mode,flagged", [("ARRIVE", False), ("HOLD", False), ("BRACE", True)])
def test_hold_range_idle_in_its_own_arrive_or_hold_phase_is_not_escalated(mode, flagged):
    """p5w5_b3: HOLD_RANGE Darts coasting in to an ordered 470 km (ARRIVE, drive idle,
    requested_throttle 1.0 as the engine logs it) read 'has not flown your HOLD_RANGE
    for 2 checkpoints running ... the drive stayed idle while flying ARRIVE'; the admiral
    then dropped HOLD_RANGE for the battle. A hold by evasion is still reported."""
    from src.llm.admiral import AdmiralDecision, AdmiralOrder
    from src.simulation import CombatSimulation, SimulationEvent, SimulationEventType
    from tests.test_admiral_command_interface import make_admiral, own_caps, ship
    sim = CombatSimulation(seed=1)
    ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
    ship(sim, "beta_1", "destroyer", "beta", (480, 0, 0), fwd=(-1, 0, 0))
    admiral = make_admiral("alpha")
    caps = own_caps(sim, "alpha")
    lines = []
    for cp in (1, 2, 3):
        t0 = sim.current_time
        order = AdmiralOrder("alpha_1", "TIS alpha_1", "Hold 470 km.",
                             maneuver={"maneuver_type": "HOLD_RANGE", "target": "beta_1", "range_km": 470.0,
                                       "evade": "dodge", "throttle": 1.0})
        admiral.order_history.append(AdmiralDecision(timestamp=t0, fleet_orders=[order]))
        admiral._last_decision_t = t0
        sim.events.append(SimulationEvent(SimulationEventType.EXECUTION_STATE, t0 + 1.0, "alpha_1", None,
                                          {"mode": mode, "applied_throttle": 0.0, "requested_throttle": 1.0,
                                           "engine_fraction": 1.0}))
        sim.current_time = t0 + 30.0
        admiral.captain_reports = [{"ship_id": "alpha_1", "ship_name": "TIS alpha_1", "status": "ACK",
                                    "compliance": {"executed_maneuver": "HOLD_RANGE", "followed": True,
                                                   "deviations": []}}]
        lines.append(admiral._build_snapshot(sim, caps, full=True).noncompliance)
    assert bool(lines[2]) is flagged, lines
