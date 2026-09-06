"""Live launch checks use fake transport and heuristic captains only."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from scripts.run_live_draft_battle import LiveDraftRunner, LiveDraftState, SpectatorServer
from src.llm.battle_runner import BattleConfig, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient, LLMCallError
from src.llm.fleet_config import BattleFleetConfig
from src.llm.fleet_draft import FleetDraft, name_drafted_ships, draft_to_fleet_definition, run_admiral_draft
from src.llm.spending import RequestBudget


def test_reasoning_is_sent_and_audited_with_budget_for_both_call_types():
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Ready'},
                                                     'finish_reason': 'stop'}],
                                        'usage': {'cost': 0.001}})
    client = CaptainClient(api_key='test', temperature=None, max_tokens=8192, reasoning_effort='high')
    client._client = httpx.Client(transport=httpx.MockTransport(respond))
    client.budget = RequestBudget(1, {client.model: {'prompt': '0.0000002', 'completion': '0.0000012'}})
    observed = []
    with client.audit_scope(lambda payload, response: observed.append(payload.copy())):
        client.complete([{'role': 'user', 'content': 'Ready?'}])
        client.decide_with_tools([{'role': 'user', 'content': 'Ready?'}], [])
    assert len(requests) == len(observed) == 2
    for payload in requests + observed:
        assert payload['reasoning'] == {'effort': 'high'}
        assert payload['max_tokens'] == 8192
        assert 'temperature' not in payload
        assert payload['provider']['max_price']['completion'] == 1.2
    assert client.budget.reported_usd == pytest.approx(0.002)


def test_failed_model_draft_cannot_silently_substitute_automatic_fleet():
    client = Mock()
    client.decide_with_tools.side_effect = LLMCallError('unavailable')
    with pytest.raises(RuntimeError, match='did not submit a valid fleet'):
        run_admiral_draft(client, 'test/model', 'Admiral', 'alpha', load_fleet_data(),
                          allow_auto_fallback=False, captain_model='test/model', verbose=False)
    prompt = client.decide_with_tools.call_args.args[0][0]['content']
    assert 'independent test/model AI captains' in prompt


def test_final_turn_executes_without_an_extra_decision_and_saves_live_replay(tmp_path):
    state = LiveDraftState()
    state.drafts = {f: FleetDraft(faction=f, budget=7, points_spent=7,
                                  ships=name_drafted_ships(['frigate'], f))
                    for f in ('alpha', 'beta')}
    fc = BattleFleetConfig(
        battle_name='Live final-turn check',
        alpha_fleet=draft_to_fleet_definition(state.drafts['alpha'], 500),
        beta_fleet=draft_to_fleet_definition(state.drafts['beta'], 500),
        time_limit_s=60, max_checkpoints=1, decision_interval_s=30, seed=1,
        record_battle=True, record_sim_trace=True, personality_selection=False)
    runner = LiveDraftRunner(BattleConfig.from_fleet_config(fc, verbose=False, recording_dir=str(tmp_path)),
                             LLMCaptainConfig('A', 'A'), LLMCaptainConfig('B', 'B'), None, fc)
    output = tmp_path / 'live.json'
    runner.configure_live(Mock(), state, {}, output, 0)
    result = runner.run_battle(load_fleet_data())
    recording = json.loads(output.read_text())
    assert result.checkpoints_used == 1
    assert result.duration_s == 60
    assert recording['sim_trace'][-1]['t'] == 60
    decisions = [e for e in recording['events'] if e['event_type'] == 'captain_decision']
    assert len(decisions) == 2
    assert {e['timestamp'] for e in decisions} == {30}
    assert len({e['event_id'] for e in recording['events']}) == len(recording['events'])


def test_budget_denial_stops_live_runner():
    budget = RequestBudget(0.0001, {'test': {'prompt': '0.001', 'completion': '0.001'}})
    with pytest.raises(RuntimeError, match='exhausted'):
        budget.reserve({'model': 'test', 'messages': [], 'max_tokens': 8192})
    runner = object.__new__(LiveDraftRunner)
    runner.client = SimpleNamespace(budget=budget)
    with pytest.raises(RuntimeError, match='Spending ceiling'):
        runner.check_budget()


def test_spectator_routes_are_read_only():
    app = SpectatorServer().build_app()
    assert {route.method for route in app.router.routes()} <= {'GET', 'HEAD'}
