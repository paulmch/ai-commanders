"""
Tests for concurrent LLM calls within a checkpoint.

A checkpoint's independent calls - the two admirals' decisions, each admiral's
per-ship orders, and all captains' decisions - now run concurrently
(BattleConfig.parallel_llm). Commands are applied only after everyone has
decided, so concurrency must change wall-clock, never semantics.

No network access: the client is a stub that sleeps briefly and records call
intervals so the tests can assert genuine overlap.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from src.llm.battle_runner import BattleConfig, LLMBattleRunner
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CallStats, ToolCall
from src.llm.fleet_config import BattleFleetConfig
from src.llm.prompts import CaptainPersonality


FLEET_DATA = json.loads(
    (Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text()
)

CALL_SLEEP_S = 0.1


class TimingStubClient:
    """
    Stub client that answers like a decisive commander and records, per call
    kind, the wall-clock interval of every call so tests can prove overlap.
    """

    def __init__(self):
        self.stats = CallStats()
        self.session_id = None
        self.intervals = {"phase1": [], "ship_order": [], "captain": []}
        self._lock = threading.Lock()

    def _classify(self, tools):
        names = {t["function"]["name"] for t in tools}
        if names == {"issue_order"}:
            return "ship_order"
        if "set_fleet_directive" in names:
            return "phase1"
        return "captain"

    def decide_with_tools(self, messages, tools, model=None, temperature=None):
        kind = self._classify(tools)
        start = time.monotonic()
        time.sleep(CALL_SLEEP_S)
        with self._lock:
            self.intervals[kind].append((start, time.monotonic()))

        if kind == "phase1":
            return [
                ToolCall(id="1", name="set_fleet_directive",
                         arguments={"directive": "Close and focus fire."}),
                ToolCall(id="2", name="set_battle_plan",
                         arguments={"plan": "Phase 1: close. Phase 2: salvo."}),
            ]
        if kind == "ship_order":
            return [
                ToolCall(id="1", name="issue_order",
                         arguments={"ship_name": "x",
                                    "order_text": "INTERCEPT nearest, fire at will.",
                                    "priority": "NORMAL"}),
            ]
        return [
            ToolCall(id="1", name="set_maneuver",
                     arguments={"maneuver_type": "INTERCEPT", "throttle": 0.8}),
            ToolCall(id="2", name="log_note",
                     arguments={"note": "Closing per plan."}),
            ToolCall(id="3", name="respond_to_orders",
                     arguments={"response_type": "ACKNOWLEDGE"}),
        ]

    def complete(self, messages, model=None, temperature=None):  # pragma: no cover
        raise AssertionError("no discussion expected in this stub battle")


def max_concurrency(intervals):
    """Largest number of intervals sharing a moment in time."""
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    peak = current = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def make_fleet_runner(parallel: bool, ships_per_side: int = 2):
    fleet_config = BattleFleetConfig.from_dict({
        "battle_name": "Parallel Test",
        "time_limit_s": 300.0,
        "decision_interval_s": 30.0,
        "initial_distance_km": 400.0,
        "alpha_fleet": {
            "admiral": {"model": "stub/model"},
            "ships": [{"ship_type": "destroyer", "model": "stub/model"}] * ships_per_side,
        },
        "beta_fleet": {
            "admiral": {"model": "stub/model"},
            "ships": [{"ship_type": "destroyer", "model": "stub/model"}] * ships_per_side,
        },
    })
    config = BattleConfig(
        initial_distance_km=400.0,
        time_limit_s=300.0,
        max_checkpoints=1,
        verbose=False,
        personality_selection=False,
        record_battle=False,
        parallel_llm=parallel,
        seed=7,
    )
    dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/model",
                             personality=CaptainPersonality.BALANCED)
    stub = TimingStubClient()
    runner = LLMBattleRunner(
        config=config, alpha_config=dummy, beta_config=dummy,
        client=stub, fleet_config=fleet_config,
    )
    return runner, stub


class TestParallelFleetCheckpoint:

    def test_independent_calls_overlap(self):
        runner, stub = make_fleet_runner(parallel=True)
        runner.run_fleet_battle(FLEET_DATA)

        # Two admirals decide at once, and within each decision the per-ship
        # orders fan out; the four captains fan out afterwards.
        assert max_concurrency(stub.intervals["phase1"]) >= 2
        assert max_concurrency(stub.intervals["ship_order"]) >= 2
        assert max_concurrency(stub.intervals["captain"]) >= 2

        # Semantics preserved: every agent decided.
        assert len(stub.intervals["phase1"]) == 2      # one per admiral
        assert len(stub.intervals["ship_order"]) == 4  # one per ship
        assert len(stub.intervals["captain"]) == 4

    def test_new_state_flows_through_concurrent_checkpoint(self):
        """The task-1 features must survive the parallel paths end to end."""
        runner, stub = make_fleet_runner(parallel=True)
        runner.run_fleet_battle(FLEET_DATA)

        assert runner.alpha_admiral.standing_plan == "Phase 1: close. Phase 2: salvo."
        assert runner.beta_admiral.standing_plan == "Phase 1: close. Phase 2: salvo."
        for captain in list(runner.alpha_captains.values()) + list(runner.beta_captains.values()):
            assert [e["note"] for e in captain.captain_log] == ["Closing per plan."]
            # Orders reached the captain before it decided.
            assert captain.decision_history, captain.ship_name

    def test_sequential_mode_still_works(self):
        runner, stub = make_fleet_runner(parallel=False)
        runner.run_fleet_battle(FLEET_DATA)

        assert max_concurrency(stub.intervals["captain"]) == 1
        assert max_concurrency(stub.intervals["ship_order"]) == 1
        assert len(stub.intervals["phase1"]) == 2
        assert len(stub.intervals["ship_order"]) == 4
        assert len(stub.intervals["captain"]) == 4
        assert runner.alpha_admiral.parallel_ship_orders is False


class TestCallStatsThreadSafety:

    def test_concurrent_increments_are_not_lost(self):
        stats = CallStats()
        n_threads, per_thread = 16, 500

        # Force frequent thread switches so an unlocked += would actually race.
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            def hammer():
                for _ in range(per_thread):
                    stats.record_call()
                    stats.record_retry()
                    stats.record_usage({"prompt_tokens": 3,
                                        "prompt_tokens_details": {"cached_tokens": 2}})

            threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old_interval)

        total = n_threads * per_thread
        assert stats.calls == total
        assert stats.retries == total
        assert stats.prompt_tokens == 3 * total
        assert stats.cached_tokens == 2 * total
        assert stats.cache_hit_rate == pytest.approx(2 / 3)
