"""Record real execution paths; never call paid providers from tests."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.llm.battle_recorder import BattleRecorder
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient
from src.llm.execution_feedback import execution_feedback
from src.llm.fleet_config import BattleFleetConfig
from src.llm.replay_audit import AuditedClient
from src.llm.replay_commentary import annotate_recording
from src.llm.spending import RequestBudget
from src.simulation import Maneuver, ManeuverType, SimulationEventType
from tests.test_project_audit import setup_ships, execute, fleet_dict


@pytest.fixture(scope='module')
def fleet():
    return load_fleet_data()


def test_ordnance_keeps_origin_after_next_decision_and_damage_is_linked(fleet):
    sim, captain = setup_ships(fleet)
    sim.evidence.decisions['a'] = 'decision-1'
    execute(captain, sim, 'set_primary_target', target_name='b')
    execute(captain, sim, 'set_weapons_order', spinal_mode='FIRE_IMMEDIATE', turret_mode='FIRE_IMMEDIATE')
    sim._process_weapons_orders(sim.get_ship('a'))
    launch = next(e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED)
    assert launch.data['decision_ids'] == ['decision-1']
    flight = sim.projectiles[0]
    sim.evidence.decisions['a'] = 'decision-2'
    execute(captain, sim, 'set_primary_target', target_name='c')
    sim._resolve_projectile_hit(flight, sim.get_ship('b'))
    impact = next(e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_IMPACT)
    assert impact.data['decision_ids'] == ['decision-1']
    assert impact.data['projectile_id'] == launch.data['projectile_id']
    damage = [e for e in sim.events if e.event_type == SimulationEventType.DAMAGE_TAKEN]
    assert damage and all(e.data['decision_ids'] == ['decision-1'] for e in damage)


def test_queued_launch_keeps_its_command_and_reports_cancellation(fleet):
    sim, captain = setup_ships(fleet, 'corvette')
    sim.evidence.decisions['a'] = 'first'
    execute(captain, sim, 'launch_torpedo', target_name='b', count=3)
    queued = [e for e in sim.events if e.data.get('status') == 'queued']
    assert len(queued) == 2
    assert all(e.data['tool_call_id'] == 'test' for e in queued)
    sim.evidence.decisions['a'] = 'later'
    sim.current_time = 12
    sim._process_pending_torpedo_launches()
    launches = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
    assert len(launches) == 2
    assert launches[-1].data['decision_ids'] == ['first']
    assert launches[-1].data['command_ids'] == queued[0].data['command_ids']
    sim.get_ship('b').is_surrendered = True
    sim._process_pending_torpedo_launches()
    assert any(e.data.get('status') == 'cancelled' and e.data['command_ids'] == queued[1].data['command_ids'] for e in sim.events)


def test_weapon_wait_is_transition_only_and_visible_without_recorder(fleet):
    sim, captain = setup_ships(fleet)
    execute(captain, sim, 'set_primary_target', target_name='b')
    execute(captain, sim, 'set_weapons_order', spinal_mode='HOLD_FIRE', turret_mode='HOLD_FIRE')
    for _ in range(4):
        sim._process_weapons_orders(sim.get_ship('a'))
    states = [e for e in sim.events if e.event_type == SimulationEventType.WEAPON_STATUS]
    assert len(states) == len(sim.get_ship('a').weapons_orders)
    assert all(e.data['reason'] == 'hold_fire' for e in states)
    assert 'hold_fire' in execution_feedback(sim, ['a'])
    assert not execution_feedback(sim, ['b'])


def test_applied_thrust_reports_coast_and_fuel_limit(fleet):
    sim, _ = setup_ships(fleet)
    ship = sim.get_ship('a')
    sim.inject_command('a', Maneuver(ManeuverType.MAINTAIN, 0, 30, throttle=1))
    sim._update_ship(ship, 1)
    assert ship.applied_throttle == 0
    sim.inject_command('a', Maneuver(ManeuverType.BURN, 0, 30, throttle=1))
    ship.kinematic_state.propellant_kg = 1e-8
    sim._update_ship(ship, 1)
    assert 0 <= ship.applied_throttle < .001
    e = [e for e in sim.events if e.event_type == SimulationEventType.EXECUTION_STATE][-1]
    assert e.data['requested_throttle'] == 1
    assert e.data['fuel_fraction'] < .001


def test_recording_captures_commands_metadata_and_raw_miss_reason(fleet):
    fc = BattleFleetConfig.from_dict(fleet_dict(record_battle=True, record_sim_trace=True, seed=73))
    runner = LLMBattleRunner(BattleConfig.from_fleet_config(fc, verbose=False), LLMCaptainConfig('A', 'A'),
                            LLMCaptainConfig('B', 'B'), None, fc)
    runner.setup_fleet_battle(fleet)
    sid = 'alpha_1'
    captain = runner.alpha_captains[sid]
    commands = captain.decide(sid, runner.simulation)
    runner._record_captain_decision(sid, captain, commands)
    for cmd in commands:
        runner.simulation.inject_command(sid, cmd)
    runner.simulation._log_event(SimulationEventType.PROJECTILE_MISS, sid, 'beta_1',
                                {'projectile_id': 'test-round', 'reason': 'dispersion', 'geometry_factor': .2})
    data = [e.to_dict() for e in runner.recorder.events]
    miss = next(e for e in data if e['event_type'] == 'miss')
    assert miss['data']['projectile_id'] == 'test-round'
    assert miss['data']['reason'] == 'dispersion'
    assert miss['data']['geometry_factor'] == .2
    assert len({e['event_id'] for e in data}) == len(data)
    assert [e['sequence'] for e in data] == list(range(1, len(data) + 1))
    assert runner.recorder.recording.provenance['seed'] == 73
    assert len(runner.recorder.recording.provenance['source_sha256']) == 64


def test_parallel_model_calls_keep_actor_inputs_and_usage_separate():
    recorder = BattleRecorder()
    recorder._is_recording = True
    client = CaptainClient(api_key='test', max_tokens=64)
    def respond(request):
        p = json.loads(request.content)
        return httpx.Response(200, json={'id': p['model'], 'model': p['model'], 'usage': {'cost': .01},
            'choices': [{'message': {'content': 'Orders ready', 'reasoning': 'should not be saved'}, 'finish_reason': 'stop'}]})
    client._client = httpx.Client(transport=httpx.MockTransport(respond))
    def call(i):
        actor = AuditedClient(client, recorder, SimpleNamespace(current_time=30), 'captain', 'alpha', str(i))
        return actor.complete([{'role': 'system', 'content': 'shared doctrine'}, {'role': 'user', 'content': str(i)}], model=str(i))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(call, range(8)))
    assert len(recorder.events) == 8
    assert len(recorder.recording.assets) == 10  # one doctrine, eight states, empty tools
    for e in recorder.events:
        assert e.ship_id == e.data['model']
        assert recorder.recording.assets[e.data['message_refs'][1]]['content'] == e.ship_id
        assert 'reasoning' not in e.data['output']
    assert client.stats.cost_usd == pytest.approx(.08)


def test_budget_reserves_parallel_calls_and_retains_unknown_costs():
    budget = RequestBudget(.025, {'cheap': {'prompt': '0', 'completion': '.01'}})
    payload = {'model': 'cheap', 'max_tokens': 1, 'messages': []}
    first = budget.reserve(payload)
    second = budget.reserve(payload)
    with pytest.raises(RuntimeError, match='budget'):
        budget.reserve(payload)
    budget.settle(first, .001)
    budget.settle(second, None)
    assert budget.committed_usd == pytest.approx(.0135)
    assert budget.reported_usd == .001


def test_model_commentary_rejects_invented_evidence():
    recording = {'events': [
        {'event_id': 'd1', 'event_type': 'captain_decision', 'data': {}},
        {'event_id': 'e2', 'event_type': 'hit', 'data': {'decision_ids': ['d1']}},
    ]}
    client = Mock()
    client.complete.return_value = SimpleNamespace(content='{"text":"An impact was recorded.","event_ids":["invented"]}', model='cheap', usage={})
    assert annotate_recording(recording, client, 'cheap') == []
    client.complete.return_value.content = '{"text":"An impact was recorded.","event_ids":["e2"]}'
    assert annotate_recording(recording, client, 'cheap')[0]['event_ids'] == ['e2']


def test_live_sequence_cursor_delivers_late_same_time_decisions():
    from tests.test_mcp_draft_live import make_test_client
    from src.llm.mcp_http_server import MCPHttpServer
    async def run():
        recorder = BattleRecorder()
        recorder._is_recording = True
        recorder.recording.is_fleet_battle = True
        recorder.recording.sim_trace = [{'t': 30, 'ships': {}}]
        first = recorder.record(30, 'checkpoint')
        server = MCPHttpServer()
        server._battle_runner = SimpleNamespace(recorder=recorder, simulation=SimpleNamespace(current_time=30), fleet_config=None)
        client = await make_test_client(server)
        try:
            recorder.record(30, 'captain_decision', 'alpha_1', commands=[])
            response = await client.get(f'/live/recording?since_t=30&since_seq={first.sequence}')
            data = await response.json()
            assert not data['recording']['sim_trace']
            assert [e['event_type'] for e in data['recording']['events']] == ['captain_decision']
            bad = await client.get('/live/recording?since_seq=-1')
            assert bad.status == 400
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['duel', 'fleet', 'async'])
def test_every_runner_records_captain_decisions(mode, fleet, tmp_path):
    config = BattleConfig(time_limit_s=61, max_checkpoints=4, personality_selection=False,
                          record_battle=True, record_sim_trace=True, recording_dir=str(tmp_path), verbose=False)
    fc = None if mode == 'duel' else BattleFleetConfig.from_dict(fleet_dict(
        time_limit_s=61, max_checkpoints=4, personality_selection=False, record_battle=True))
    runner = LLMBattleRunner(config, LLMCaptainConfig('A', 'A', model='heuristic'),
                            LLMCaptainConfig('B', 'B', model='heuristic'), None, fc)
    if mode == 'async':
        asyncio.run(runner.run_fleet_battle_async(fleet))
    else:
        runner.run_battle(fleet)
    decisions = [e for e in runner.recorder.events if e.event_type == 'captain_decision']
    assert decisions
    ids = {e.event_id for e in decisions}
    commands = [e for e in runner.recorder.events if e.event_type == 'command_status']
    assert commands and all(e.data['decision_id'] in ids for e in commands)


def test_mcp_direct_orders_and_weapon_execution_share_provenance(fleet):
    from src.llm.mcp_state import MCPCommand, MCPCommandType
    from src.llm.mcp_controller import apply_mcp_commands_to_simulation
    sim, _ = setup_ships(fleet)
    sim.recorder = BattleRecorder()
    sim.recorder._is_recording = True
    result = apply_mcp_commands_to_simulation([
        MCPCommand(MCPCommandType.SET_PRIMARY_TARGET, 'a', {'target_id': 'b'}),
        MCPCommand(MCPCommandType.SET_WEAPONS_ORDER, 'a', {'spinal_mode': 'FIRE_IMMEDIATE', 'turret_mode': 'FIRE_IMMEDIATE'}),
    ], sim, 'alpha')
    assert not result['errors']
    sim._process_weapons_orders(sim.get_ship('a'))
    launches = [e for e in sim.events if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED]
    decision = sim.recorder.events[0]
    assert decision.data['model'] == 'mcp'
    assert launches and all(e.data['decision_ids'] == [decision.event_id] for e in launches)
