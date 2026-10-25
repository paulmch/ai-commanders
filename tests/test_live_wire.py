"""
Live spectator wire (pass 4, workstream E1): battle identity, deliberation
status, spend, end-of-battle ordering and the fake live server.

No network, no LLM: aiohttp's in-process test client drives the handlers,
model calls go through a stub client.
"""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.fake_live_server import FakeLiveServer
from scripts.run_live_draft_battle import LiveDraftRunner, LiveDraftState, SpectatorServer
from src.llm.battle_recorder import BattleEvent, BattleRecorder
from src.llm.battle_runner import BattleConfig, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import BattleFleetConfig
from src.llm.fleet_draft import FleetDraft, auto_draft, draft_to_fleet_definition, name_drafted_ships
from src.llm.live_status import PendingCalls, draft_live_entry
from src.llm.mcp_draft import DraftManager
from src.llm.mcp_http_server import LIVE_CLIENT_ACTIVE_S, MCPHttpServer
from src.llm.replay_audit import AuditedClient

FLEET_DATA = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


async def _client(app_or_server):
    app = app_or_server.build_app() if hasattr(app_or_server, "build_app") else app_or_server
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _runner(sim_time=2.0):
    recorder = BattleRecorder()
    recorder._is_recording = True
    recorder.recording.is_fleet_battle = True
    recorder.recording.battle_name = "Wire Test"
    recorder.recording.sim_trace = [{"t": 1.0, "ships": {}}, {"t": 2.0, "ships": {}}]
    recorder.events = [BattleEvent(timestamp=1.5, event_type="shot_fired", ship_id="alpha_1")]
    return SimpleNamespace(recorder=recorder, simulation=SimpleNamespace(current_time=sim_time),
                           fleet_config=SimpleNamespace(decision_interval_s=30.0),
                           config=SimpleNamespace(verbose=False, seed=1))


# ---------------------------------------------------------------- PendingCalls

def test_pending_calls_track_age_and_factions():
    now = [100.0]
    calls = PendingCalls(clock=lambda: now[0])
    a = calls.start("admiral", "alpha", None, "x/model")
    now[0] = 103.0
    b = calls.start("captain", "beta", "beta_2", "y/model")
    now[0] = 110.0
    snap = calls.snapshot()
    assert [c["actor"] for c in snap] == ["admiral", "captain"]   # oldest first
    assert snap[0]["age_s"] == 10.0 and snap[1]["age_s"] == 7.0
    assert snap[1]["ship_id"] == "beta_2"
    assert calls.factions() == ["alpha", "beta"]
    calls.finish(a)
    calls.finish(a)   # idempotent
    assert calls.factions() == ["beta"] and len(calls) == 1
    calls.finish(b)
    assert calls.snapshot() == []


class _StubClient:
    model = "stub/model"

    def __init__(self, fail=False):
        self.fail = fail
        self.seen_pending = None
        self._observer = None

    def audit_scope(self, observer):
        client = self

        class Scope:
            def __enter__(self):
                client._observer = observer

            def __exit__(self, *exc):
                client._observer = None
        return Scope()

    def complete(self, messages, model=None):
        self.seen_pending = self.registry.snapshot()
        if self.fail:
            raise RuntimeError("provider down")
        self._observer({"model": model or self.model, "messages": messages},
                       {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
        return "ok"


@pytest.mark.parametrize("fail", [False, True])
def test_audited_client_registers_call_while_in_flight(fail):
    recorder = BattleRecorder()
    recorder._is_recording = True
    recorder.pending_calls = PendingCalls()
    stub = _StubClient(fail=fail)
    stub.registry = recorder.pending_calls
    audited = AuditedClient(stub, recorder, SimpleNamespace(current_time=30.0), "captain", "beta", "beta_3")
    if fail:
        with pytest.raises(RuntimeError):
            audited.complete([{"role": "user", "content": "orders?"}], model="m/one")
    else:
        audited.complete([{"role": "user", "content": "orders?"}], model="m/one")
    # In flight while the provider worked, gone afterwards (also on failure)
    assert stub.seen_pending == [{"actor": "captain", "faction": "beta", "ship_id": "beta_3",
                                  "model": "m/one", "age_s": 0.0}]
    assert len(recorder.pending_calls) == 0
    kinds = [e.event_type for e in recorder.events]
    assert kinds == (["model_error"] if fail else ["model_call"])
    if not fail:
        assert recorder.events[0].data["latency_s"] >= 0


def test_audited_client_without_registry_is_unchanged():
    recorder = BattleRecorder()
    recorder._is_recording = True
    stub = _StubClient()
    stub.registry = PendingCalls()
    AuditedClient(stub, recorder, SimpleNamespace(current_time=0), "admiral", "alpha").complete([])
    assert stub.seen_pending == []
    assert recorder.events[0].event_type == "model_call"


# ------------------------------------------------------------- live block

def test_live_block_identity_status_age_pending_and_spend():
    async def scenario():
        server = MCPHttpServer()
        other = MCPHttpServer()
        assert server.battle_id != other.battle_id and len(server.battle_id) == 12
        runner = _runner(sim_time=60.0)
        runner.recorder.pending_calls = PendingCalls()
        server.set_live_source(runner)
        server.spend_provider = lambda: {"reported_usd": 0.42, "limit_usd": 2.0, "calls": 9}
        server.replay_url = "/recordings/x.json"
        server.models = {"alpha": "a/model", "beta": "b/model"}
        server.set_battle_status("running", checkpoint=1)
        client = await _client(server)
        try:
            body = await (await client.get("/live/recording")).json()
            live = body["live"]
            assert live["battle_id"] == server.battle_id
            assert live["spend"]["reported_usd"] == 0.42
            assert live["replay_url"] == "/recordings/x.json"
            assert live["models"] == {"alpha": "a/model", "beta": "b/model"}
            assert live["pending_calls"] == [] and live["waiting_for"] == []
            assert live["result"] is None
            assert live["status_age_s"] >= 0

            # Deliberating: waiting_for comes from the calls in flight
            server.set_battle_status("paused", checkpoint=2)
            runner.recorder.pending_calls.start("admiral", "beta", None, "b/model")
            live = (await (await client.get("/live/recording")).json())["live"]
            assert live["status"] == "paused"
            assert live["waiting_for"] == ["beta"]
            assert live["pending_calls"][0]["actor"] == "admiral"
            assert live["status_age_s"] < 5

            # A broken spend provider never breaks the poll
            server.spend_provider = lambda: 1 / 0
            live = (await (await client.get("/live/recording")).json())["live"]
            assert live["spend"] is None
        finally:
            await client.close()
    asyncio.run(scenario())


def test_live_block_models_from_runner_and_result_when_ended():
    async def scenario():
        server = MCPHttpServer()
        runner = _runner()
        runner.alpha_mcp = object()
        runner.beta_admiral = SimpleNamespace(model="z/admiral")
        runner.recorder.recording.winner = "beta"
        runner.recorder.recording.result_reason = "Alpha fleet eliminated"
        server.set_live_source(runner)
        server.set_battle_status("ended", checkpoint=3)
        client = await _client(server)
        try:
            live = (await (await client.get("/live/recording")).json())["live"]
            assert live["models"] == {"alpha": "MCP", "beta": "z/admiral"}
            assert live["phase"] == "ended"
            assert live["result"] == {"winner": "beta", "reason": "Alpha fleet eliminated"}
        finally:
            await client.close()
    asyncio.run(scenario())


def test_linger_returns_at_once_without_spectators_and_after_two_ended_polls():
    async def scenario():
        server = MCPHttpServer()
        server.set_live_source(_runner())
        server.set_battle_status("ended")
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await server.linger_for_spectators(timeout_s=5)
        assert loop.time() - t0 < 0.5          # nobody ever polled
        client = await _client(server)
        try:
            await client.get("/live/recording?client=v1")
            linger = asyncio.ensure_future(server.linger_for_spectators(timeout_s=5))
            await asyncio.sleep(0.2)
            assert not linger.done()            # the end, but not the final fetch yet
            await client.get("/live/recording?client=v1&since_t=2&since_seq=0")
            await asyncio.wait_for(linger, 1.0)
        finally:
            await client.close()
        t0 = loop.time()
        server2 = MCPHttpServer()
        server2._live_clients["v9"] = {"last": time.monotonic(), "ended": 0}
        await server2.linger_for_spectators(timeout_s=0.3)   # watching, never saw the end
        assert 0.25 < loop.time() - t0 < 1.5
    asyncio.run(scenario())


def test_linger_waits_for_every_spectator_not_the_first_two_ended_fetches():
    """Two viewers watching: A's ended poll and final fetch must not end the
    linger while B, polling 2 s later, has not been told the battle ended."""

    async def scenario():
        server = MCPHttpServer()
        server.set_live_source(_runner())
        server.set_battle_status("running")
        a, b = await _client(server), await _client(server)
        try:
            await a.get("/live/recording?client=A")
            await b.get("/live/recording?client=B")
            server.set_battle_status("ended")
            linger = asyncio.ensure_future(server.linger_for_spectators(timeout_s=5))
            ended = await (await a.get("/live/recording?client=A")).json()
            assert ended["live"]["status"] == "ended"
            await a.get("/live/recording?client=A&since_t=2&since_seq=1")
            await asyncio.sleep(0.3)
            assert not linger.done(), "viewer B has not seen the end yet"
            await b.get("/live/recording?client=B")
            await asyncio.sleep(0.2)
            assert not linger.done()
            await b.get("/live/recording?client=B&since_t=2&since_seq=1")
            await asyncio.wait_for(linger, 1.0)
            # A viewer that stopped polling before the end holds nobody up
            server._live_clients["gone"] = {
                "last": time.monotonic() - LIVE_CLIENT_ACTIVE_S - 1, "ended": 0}
            await asyncio.wait_for(server.linger_for_spectators(timeout_s=5), 0.5)
            # Viewers without a client id (older builds) share one entry,
            # which is the old single-spectator behaviour
            server2 = MCPHttpServer()
            server2.set_live_source(_runner())
            server2.set_battle_status("ended")
            c = await _client(server2)
            try:
                await c.get("/live/recording")
                linger2 = asyncio.ensure_future(server2.linger_for_spectators(timeout_s=5))
                await asyncio.sleep(0.2)
                assert not linger2.done()
                await c.get("/live/recording?since_t=2")
                await asyncio.wait_for(linger2, 1.0)
            finally:
                await c.close()
        finally:
            await a.close()
            await b.close()
    asyncio.run(scenario())


def test_http_battle_flips_to_ended_only_after_the_verdict(tmp_path):
    from src.llm.mcp_http_server import run_fleet_battle_with_http
    from src.llm.battle_runner import LLMBattleRunner
    fleet = load_fleet_data()
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": [{"ship_type": "destroyer", "model": "heuristic"}]},
        "beta_fleet": {"ships": [{"ship_type": "destroyer", "model": "heuristic"}]},
        "record_battle": True, "personality_selection": False, "seed": 13,
        "time_limit_s": 30, "decision_interval_s": 30,
    })
    runner = LLMBattleRunner(BattleConfig.from_fleet_config(config, verbose=False,
                                                            recording_dir=str(tmp_path)),
                             LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), Mock(), config)
    seen = []
    server = Mock()
    server.battle_id = "abc123def456"
    server.set_battle_status.side_effect = lambda status, **kw: seen.append(
        (status, runner.recorder.recording.result_reason if runner.recorder else None))
    asyncio.run(run_fleet_battle_with_http(runner, fleet, server))
    assert seen[-1][0] == "ended"
    assert seen[-1][1]                                   # verdict already recorded
    assert runner.recorder.recording.provenance["battle_id"] == "abc123def456"
    assert isinstance(runner.recorder.pending_calls, PendingCalls)


# ----------------------------------------------------------------- drafts

def test_draft_entry_keeps_legacy_keys_and_seals_picks():
    draft = auto_draft("alpha", 100, 8, seed=1)
    draft.selection_rationale = "Fast hulls"
    sealed = draft_live_entry(draft, ready=True, model="m", budget=100)
    assert sealed["ships"] == len(draft.ships) and sealed["points_spent"] == draft.points_spent
    assert sealed["roster"] == [] and sealed["ship_names"] == []
    assert sealed["selection_rationale"] is None
    shown = draft_live_entry(draft, ready=True, reveal_ships=True, reveal_rationale=True)
    assert [r["name"] for r in shown["roster"]] == [s.ship_name for s in draft.ships]
    assert all(r["cost"] and r["type"] and len(r["offset_km"]) == 3 for r in shown["roster"])
    assert shown["selection_rationale"] == "Fast hulls"
    empty = draft_live_entry(None, ready=False)
    assert empty["ships"] == 0 and empty["roster"] == []


def test_mcp_draft_summary_reveals_only_when_everyone_committed():
    manager = DraftManager(fleet_data=FLEET_DATA, budget=100, max_ships=8,
                           initial_distance_km=500.0, mcp_factions={"alpha"})
    manager.select("alpha", [{"ship_type": "destroyer", "count": 2}])
    manager.commit("alpha")
    summary = manager.live_summary()
    assert summary["alpha"]["ready"] and summary["alpha"]["ships"] == 2
    assert summary["alpha"]["roster"] == []             # beta still picking: sealed
    assert summary["alpha"]["ready_after_s"] is not None
    assert summary["beta"]["ready"] is False and summary["beta"]["ready_after_s"] is None
    manager.set_full_draft("beta", auto_draft("beta", 100, 8, seed=1))
    summary = manager.live_summary()
    assert [r["type"] for r in summary["alpha"]["roster"]] == ["destroyer", "destroyer"]
    assert summary["beta"]["roster"] and summary["alpha"]["budget"] == 100


def test_live_draft_state_shows_each_ready_fleet_and_rationale_after_both():
    state = LiveDraftState(models={"alpha": "a/m", "beta": "b/m"}, budget=100)
    alpha = auto_draft("alpha", 100, 8, seed=1)
    alpha.selection_rationale = "Alpha plan"
    state.commit("alpha", alpha)
    summary = state.live_summary()
    assert summary["alpha"]["roster"] and summary["alpha"]["model"] == "a/m"
    assert summary["alpha"]["selection_rationale"] is None
    assert summary["beta"]["ready"] is False and summary["beta"]["roster"] == []
    assert state.waiting_for() == ["beta"]
    state.commit("beta", auto_draft("beta", 100, 8, seed=2))
    summary = state.live_summary()
    assert summary["alpha"]["selection_rationale"] == "Alpha plan"


def test_live_runner_stamps_battle_id_and_pending_registry(tmp_path):
    state = LiveDraftState()
    state.drafts = {f: FleetDraft(faction=f, budget=7, points_spent=7,
                                  ships=name_drafted_ships(['frigate'], f)) for f in ('alpha', 'beta')}
    fc = BattleFleetConfig(
        battle_name='Wire', alpha_fleet=draft_to_fleet_definition(state.drafts['alpha'], 500),
        beta_fleet=draft_to_fleet_definition(state.drafts['beta'], 500), time_limit_s=60,
        max_checkpoints=1, decision_interval_s=30, seed=1, record_battle=True,
        record_sim_trace=True, personality_selection=False)
    runner = LiveDraftRunner(BattleConfig.from_fleet_config(fc, verbose=False, recording_dir=str(tmp_path)),
                             LLMCaptainConfig('A', 'A'), LLMCaptainConfig('B', 'B'), None, fc)
    server = Mock()
    server.battle_id = 'feedfacecafe'
    pending = PendingCalls()
    runner.configure_live(server, state, {}, tmp_path / 'live.json', 0, pending_calls=pending)
    runner.run_battle(load_fleet_data())
    saved = json.loads((tmp_path / 'live.json').read_text())
    assert saved['provenance']['battle_id'] == 'feedfacecafe'
    assert runner.recorder.pending_calls is pending
    # The deliberation status no longer names a placeholder commander
    paused = [c for c in server.set_battle_status.call_args_list if c.args[0] == 'paused']
    assert paused and all(not c.kwargs.get('waiting_for') for c in paused)


def test_spectator_block_carries_stage_and_registry():
    async def scenario():
        state = LiveDraftState(models={"alpha": "a", "beta": "b"}, budget=50)
        server = SpectatorServer()
        server.set_draft_manager(state)
        server.pending_calls = PendingCalls()
        server.pending_calls.start("admiral", "alpha", None, "a")
        client = await _client(server)
        try:
            live = (await (await client.get("/live/recording")).json())["live"]
            assert live["phase"] == "draft" and live["stage"].startswith("Drafting")
            assert live["pending_calls"][0]["faction"] == "alpha"
            assert live["draft"]["alpha"]["model"] == "a" and live["draft"]["alpha"]["budget"] == 50
        finally:
            await client.close()
    asyncio.run(scenario())


# --------------------------------------------------------- fake live server

def _fake_recording():
    frames = [{"t": float(t), "ships": {}} for t in range(0, 91)]
    events = []
    seq = 0

    def add(t, kind, **data):
        nonlocal seq
        seq += 1
        events.append({"event_id": f"e{seq}", "sequence": seq, "timestamp": t,
                       "event_type": kind, "ship_id": data.pop("ship_id", None), "data": data})
    add(10.0, "shot_fired", ship_id="alpha_1")
    for i in range(4):
        add(30.0, "model_call", actor="captain", faction="alpha" if i < 2 else "beta",
            model="m", message_refs=[f"asset{i}"], ship_id=f"x_{i}")
    add(30.0, "captain_decision", ship_id="alpha_1")
    add(70.0, "hit", ship_id="beta_1")
    return {"battle_name": "Fake", "is_fleet_battle": True, "sim_trace": frames, "events": events,
            "assets": {f"asset{i}": {"role": "user", "content": str(i)} for i in range(4)},
            "winner": "alpha", "result_reason": "Beta fleet eliminated",
            "provenance": {"billing": {"reported_cost_usd": 2.0, "spend_limit_usd": 4.0,
                                       "calls": 4, "models": {"alpha": "a", "beta": "b"}}}}


def test_fake_server_clock_holds_each_checkpoint_for_the_pause():
    fake = FakeLiveServer(_fake_recording(), speed=10.0, draft_seconds=0, decision_interval=30.0,
                          pause_seconds=5.0)
    assert fake.checkpoints == [30.0, 60.0]
    for wall, expect in [(1.0, (10.0, None)), (3.5, (30.0, 0.5)), (7.9, (30.0, 4.9)),
                         (8.5, (35.0, None)), (12.0, (60.0, 1.0)), (40.0, (90.0, None))]:
        fake.start = __import__("time").monotonic() - wall
        cur, pause = fake.clock()
        assert cur == pytest.approx(expect[0], abs=0.05)
        assert (pause is None) == (expect[1] is None)
        if pause:
            assert pause[1] == pytest.approx(expect[1], abs=0.05)
    # Checkpoint events trickle out over the pause, model calls reported pending
    released, held = fake.released_events(30.0, (30.0, 0.0))
    assert [e["event_type"] for e in released] == ["shot_fired"] and len(held) == 5
    pending = fake.pending_calls(held, (30.0, 1.0))
    assert [p["faction"] for p in pending] == ["alpha", "beta"]
    released, held = fake.released_events(30.0, (30.0, 4.6))
    assert held == []


def test_fake_server_trims_assets_and_reports_end():
    async def scenario():
        fake = FakeLiveServer(_fake_recording(), speed=1000.0, draft_seconds=0,
                              decision_interval=30.0, pause_seconds=0, battle_id="fakeid")
        app = web.Application()
        app.router.add_get("/live/recording", fake.handle_recording)
        client = await _client(app)
        try:
            fake.start -= 0.04   # cursor at 40 s
            body = await (await client.get("/live/recording")).json()
            assert body["live"]["battle_id"] == "fakeid"
            assert body["live"]["status"] == "running"
            assert body["recording"]["winner"] is None
            assert "billing" not in body["recording"]["provenance"]
            assert set(body["recording"]["assets"]) == {f"asset{i}" for i in range(4)}
            last_seq = max(e["sequence"] for e in body["recording"]["events"])
            fake.start -= 1.0    # far past the end
            body = await (await client.get(f"/live/recording?since_t=40&since_seq={last_seq}")).json()
            assert [e["event_type"] for e in body["recording"]["events"]] == ["hit"]
            assert body["recording"]["assets"] == {}
            assert body["live"]["phase"] == "ended"
            assert body["live"]["result"] == {"winner": "alpha", "reason": "Beta fleet eliminated"}
            assert body["recording"]["provenance"]["billing"]["reported_cost_usd"] == 2.0
            assert body["live"]["spend"]["reported_usd"] == 2.0
        finally:
            await client.close()
    asyncio.run(scenario())
