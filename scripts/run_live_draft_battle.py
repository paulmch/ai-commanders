#!/usr/bin/env python3
"""Budgeted, model-drafted battle with a live spectator server and durable replay."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm.battle_recorder import BattleRecorder
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
from src.llm.fleet_draft import auto_draft, draft_summary_dict, draft_to_fleet_definition, run_admiral_draft
from src.llm.mcp_http_server import MCPHttpServer
from src.llm.replay_audit import AuditedClient
from src.llm.spending import RequestBudget


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(path)


class LiveDraftState:
    def __init__(self):
        self.active = True
        self.drafts = {}
        self.stage = 'Drafting fleets and formations'
        self.error = None
        self.lock = threading.Lock()

    def waiting_for(self):
        with self.lock:
            return [f for f in ('alpha', 'beta') if f not in self.drafts]

    def live_summary(self):
        with self.lock:
            return {f: {
                'ready': f in self.drafts, 'is_mcp': False,
                'ships': len(self.drafts[f].ships) if f in self.drafts else 0,
                'points_spent': self.drafts[f].points_spent if f in self.drafts else 0,
                'ship_names': [s.ship_name for s in self.drafts[f].ships] if f in self.drafts else [],
                'formation_name': self.drafts[f].formation_name if f in self.drafts else '',
            } for f in ('alpha', 'beta')}


class SpectatorServer(MCPHttpServer):
    """Expose the existing viewer protocol without commander write endpoints."""
    def _setup_routes(self):
        self._app.router.add_get('/health', self._handle_health)
        self._app.router.add_get('/status', self._handle_status)
        self._app.router.add_get('/live/recording', self._handle_live_recording)
        self._app.router.add_get('/live/predictions', self._handle_live_predictions)

    def _live_block(self):
        block = super()._live_block()
        block.update(stage=self._draft_manager.stage, error=self._draft_manager.error)
        if self._battle_runner:
            block['checkpoint'] = self._battle_runner.checkpoint_count
        return block


class LiveDraftRunner(LLMBattleRunner):
    def configure_live(self, server, state, draft_recorders, output, step_delay=0.04):
        self.live_server = server
        self.live_state = state
        self.draft_recorders = draft_recorders
        self.live_output = output
        self.step_delay = step_delay

    def setup_fleet_battle(self, fleet_data):
        super().setup_fleet_battle(fleet_data)
        for faction, recorder in self.draft_recorders.items():
            self.recorder.recording.assets.update(recorder.recording.assets)
            for event in recorder.events:
                self.recorder.record(0, event.event_type, event.ship_id, **event.data, phase='draft')
        self.recorder.recording.provenance['draft'] = {
            f: draft_summary_dict(d) for f, d in self.live_state.drafts.items()}
        self.recorder.recording.provenance['final_turn_execution_s'] = self.fleet_config.decision_interval_s
        self.live_state.active = False
        self._record_sim_frame()
        self.save_progress()

    def _advance_to_checkpoint(self, deadline, fleet_mode=False):
        self.live_state.stage = 'Simulating orders'
        self.live_server.set_battle_status('running', self.checkpoint_count)
        for step in super()._advance_to_checkpoint(deadline, fleet_mode):
            if self.step_delay:
                time.sleep(self.step_delay)
            yield step

    def _refresh_targeting_awareness(self):
        self.check_budget()
        self.live_state.stage = 'Admirals and captains deciding'
        self.live_server.set_battle_status('paused', self.checkpoint_count,
                                         waiting_for=['AI commanders'])
        super()._refresh_targeting_awareness()

    def _log_fleet_decision(self, commands):
        super()._log_fleet_decision(commands)
        self.save_progress()
        self.check_budget()

    def check_budget(self):
        if self.client and self.client.budget and self.client.budget.exhausted:
            raise RuntimeError('Spending ceiling reached; battle stopped and partial replay saved')

    def save_progress(self):
        if not self.recorder:
            return
        self.recorder.recording.duration_s = self.simulation.current_time
        self.recorder.recording.total_checkpoints = self.checkpoint_count
        recording = self.recorder.recording.to_dict()
        recording['events'] = [e.to_dict() for e in self.recorder.events]
        write_json(self.live_output, recording)

    def _evaluate_fleet_result(self):
        # The ordinary runner stops immediately after the last orders. Give
        # those orders their full execution window without buying another turn.
        if self.checkpoint_count >= self.config.max_checkpoints and not self._is_fleet_battle_over():
            deadline = self.simulation.current_time + self.fleet_config.decision_interval_s
            for _ in self._advance_to_checkpoint(deadline, fleet_mode=True):
                self._record_sim_frame()
                if self._is_fleet_battle_over():
                    break
        result = super()._evaluate_fleet_result()
        self.save_progress()
        return result


def make_client(args):
    if args.offline:
        return None, None
    response = httpx.get('https://openrouter.ai/api/v1/models', timeout=30)
    response.raise_for_status()
    catalog = {m['id']: m for m in response.json()['data']}
    pricing = {}
    for model in (args.alpha_model, args.beta_model):
        entry = catalog.get(model)
        if not entry or 'tools' not in entry.get('supported_parameters', []):
            raise ValueError(f'Model unavailable or lacks tools: {model}')
        if args.reasoning not in (entry.get('reasoning', {}).get('supported_efforts') or []):
            raise ValueError(f'Model does not advertise {args.reasoning} reasoning: {model}')
        pricing[model] = entry['pricing']
    budget = RequestBudget(args.spend_limit, pricing)
    client = CaptainClient(model=args.alpha_model, temperature=None, max_tokens=8192,
                           reasoning_effort=args.reasoning, timeout=180,
                           session_id=f'ai-commanders-{args.name}')
    client.budget = budget
    return client, budget


async def run(args):
    output = ROOT / 'visualizer/public/recordings' / f'{args.name}.json'
    report_dir = ROOT / 'data/recordings' / args.name
    if output.exists() or report_dir.exists():
        raise FileExistsError(f'Choose a new --name; {args.name} already exists')
    client, budget = make_client(args)
    state = LiveDraftState()
    server = SpectatorServer(host='localhost', port=args.port)
    server.set_draft_manager(state)
    await server.start()
    print(f'WATCH: http://localhost:5173/?live=1&decisions=1', flush=True)
    models = {'alpha': args.alpha_model, 'beta': args.beta_model}
    draft_recorders = {}
    runner = None
    result = None
    finished = False
    fleet_data = load_fleet_data()

    def report():
        return {
            'name': args.name, 'models': models, 'points_per_side': args.points,
            'max_ships': args.max_ships, 'turns': args.turns, 'reasoning': args.reasoning,
            'output_tokens_per_call': 8192, 'seed': args.seed,
            'stage': state.stage, 'error': state.error, 'finished': finished,
            'checkpoint': runner.checkpoint_count if runner else 0,
            'sim_time_s': runner.simulation.current_time if runner and runner.simulation else 0,
            'reported_cost_usd': budget.reported_usd if budget else 0,
            'committed_cost_usd': budget.committed_usd if budget else 0,
            'spend_limit_usd': args.spend_limit,
            'pricing': budget.pricing if budget else {},
            'calls': client.stats.calls if client else 0,
            'failures': client.stats.failures if client else 0,
            'retries': client.stats.retries if client else 0,
            'truncated': client.stats.truncated if client else 0,
            'malformed_arguments': client.stats.malformed_arguments if client else 0,
            'winner': result.winner if result else None,
            'result_reason': result.reason if result else None,
            'recording': str(output),
        }

    def draft(faction):
        model = models[faction]
        recorder = BattleRecorder()
        recorder._is_recording = True
        draft_recorders[faction] = recorder
        if args.offline:
            selection = auto_draft(faction, args.points, args.max_ships,
                                   seed=args.seed + (faction == 'beta'))
        else:
            audited = AuditedClient(client, recorder, SimpleNamespace(current_time=0), 'admiral', faction)
            try:
                selection = run_admiral_draft(
                    audited, model, f'Admiral {model.split("/")[-1]}', faction, fleet_data,
                    budget=args.points, max_ships=args.max_ships, initial_distance_km=500,
                    seed=args.seed, captain_model=model, allow_auto_fallback=False)
            finally:
                write_json(report_dir / f'{faction}_draft_calls.json', {
                    'assets': recorder.recording.assets, 'events': [e.to_dict() for e in recorder.events]})
        with state.lock:
            state.drafts[faction] = selection
        write_json(report_dir / f'{faction}_draft.json', draft_summary_dict(selection))
        return selection

    def battle():
        nonlocal runner, result
        # Independent sealed drafts; neither model receives the other's choices.
        with ThreadPoolExecutor(max_workers=2) as pool:
            selections = dict(zip(models, pool.map(draft, models)))
        fleets = {}
        for faction, model in models.items():
            fleets[faction] = draft_to_fleet_definition(
                selections[faction], 500, captain_model='heuristic' if args.offline else model,
                admiral_config=None if args.offline else AdmiralConfig(
                    model=model, name=f'Admiral {model.split("/")[-1]}', temperature=None),
                temperature=None)
        # First decision follows a 30s opening coast. Forty decisions then
        # each receive 30s of execution: a maximum 1,230s recorded battle.
        fc = BattleFleetConfig(
            battle_name=f'{args.points}-point draft: {args.alpha_model} vs {args.beta_model}',
            alpha_fleet=fleets['alpha'], beta_fleet=fleets['beta'], initial_distance_km=500,
            decision_interval_s=30, time_limit_s=(args.turns + 1) * 30,
            max_checkpoints=args.turns, seed=args.seed, personality_selection=False,
            record_battle=True, record_sim_trace=True, use_notebooks=False)
        write_json(report_dir / 'fleet_config.json', asdict(fc))
        runner = LiveDraftRunner(
            BattleConfig.from_fleet_config(fc, recording_dir=str(report_dir), verbose=True),
            LLMCaptainConfig('Alpha', 'Alpha'), LLMCaptainConfig('Beta', 'Beta'), client, fc)
        runner.configure_live(server, state, draft_recorders, output, args.sim_step_delay)
        server.set_live_source(runner)
        result = runner.run_battle(fleet_data)
        write_json(output.with_suffix('.draft.json'), {
            'budget': args.points, **{f: draft_summary_dict(d) for f, d in selections.items()}})

    task = asyncio.create_task(asyncio.to_thread(battle))
    try:
        while not task.done():
            write_json(report_dir / 'status.json', report())
            await asyncio.sleep(1)
        await task
        state.stage = 'Battle complete'
    except Exception as exc:
        state.error = f'{type(exc).__name__}: {exc}'
        state.stage = 'Battle stopped'
        print(state.error, flush=True)
        if runner and runner.recorder:
            runner.recorder.recording.result_reason = state.error
            runner.recorder.record(runner.simulation.current_time, 'battle_stopped', reason=state.error)
            runner.save_progress()
    finally:
        finished = True
        state.active = False
        server.set_battle_status('ended', runner.checkpoint_count if runner else 0)
        write_json(report_dir / 'status.json', report())
        if runner and runner.recorder:
            runner.recorder.recording.provenance['billing'] = report()
            runner.save_progress()
        print(json.dumps(report()), flush=True)
    if not args.no_linger:
        print('Spectator server remains available; the same link now replays the battle.', flush=True)
        await asyncio.Event().wait()
    await server.stop()
    if state.error:
        raise RuntimeError(state.error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha-model', default='z-ai/glm-5.3-flash')
    parser.add_argument('--beta-model', default='openai/gpt-5.6-luna')
    parser.add_argument('--points', type=int, default=200)
    parser.add_argument('--max-ships', type=int, default=8)
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--spend-limit', type=float, default=8)
    parser.add_argument('--reasoning', default='high')
    parser.add_argument('--seed', type=int, default=20260906)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--sim-step-delay', type=float, default=0.04)
    parser.add_argument('--name', default='live_draft_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--offline', action='store_true', help='Free automatic drafts and heuristic captains for smoke testing')
    parser.add_argument('--no-linger', action='store_true', help='Close HTTP server after the battle')
    args = parser.parse_args()
    if args.turns < 1 or args.points < 1 or args.max_ships < 1 or args.sim_step_delay < 0:
        parser.error('Points, ships and turns must be positive; delay must be nonnegative')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a filename stem')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
