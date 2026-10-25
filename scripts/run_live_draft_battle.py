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
from src.llm.battle_ticker import BattleTicker
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
from src.llm import fleet_draft
from src.llm.fleet_draft import auto_draft, draft_summary_dict, draft_to_fleet_definition, run_admiral_draft
from src.llm.live_status import PendingCalls, draft_live_entry
from src.llm.mcp_http_server import MCPHttpServer
from src.llm.mock_openrouter import mock_transport
from src.llm.replay_audit import AuditedClient
from src.llm.spending import RequestBudget


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(path)


class LiveDraftState:
    def __init__(self, models=None, budget=None):
        self.active = True
        self.drafts = {}
        self.ready_at = {}
        self.models = models or {}
        self.budget = budget
        self.started = time.monotonic()
        self.stage = 'Drafting fleets and formations'
        self.error = None
        self.lock = threading.Lock()

    def commit(self, faction, selection):
        with self.lock:
            self.drafts[faction] = selection
            self.ready_at[faction] = time.monotonic()

    def waiting_for(self):
        with self.lock:
            return [f for f in ('alpha', 'beta') if f not in self.drafts]

    def live_summary(self):
        """Draft splash data. The admirals draft sealed and never read /live,
        so a side's roster shows as soon as it is ready; the rationales wait
        for both sides."""
        with self.lock:
            both = all(f in self.drafts for f in ('alpha', 'beta'))
            now = time.monotonic()
            return {f: draft_live_entry(
                self.drafts.get(f), ready=f in self.drafts, reveal_ships=f in self.drafts,
                reveal_rationale=both, model=self.models.get(f), budget=self.budget,
                elapsed_s=now - self.started,
                ready_after_s=(self.ready_at[f] - self.started) if f in self.ready_at else None)
                for f in ('alpha', 'beta')}


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
    ticker = None

    def configure_live(self, server, state, draft_recorders, output, step_delay=0.04,
                       ticker=None, vision_dir=None, pending_calls=None):
        self.live_server = server
        self.live_pending = pending_calls
        self.live_state = state
        self.draft_recorders = draft_recorders
        self.live_output = output
        self.step_delay = step_delay
        self.ticker = ticker
        self.vision_dir = vision_dir

    def setup_fleet_battle(self, fleet_data):
        super().setup_fleet_battle(fleet_data)
        # Keep this battle's tactical-plot frames next to its other reports.
        if self._admiral_view_dir and self.vision_dir:
            self._admiral_view_dir = Path(self.vision_dir)
        for faction, recorder in self.draft_recorders.items():
            self.recorder.recording.assets.update(recorder.recording.assets)
            for event in recorder.events:
                self.recorder.record(0, event.event_type, event.ship_id, **event.data, phase='draft')
        self.recorder.recording.provenance['draft'] = {
            f: draft_summary_dict(d) for f, d in self.live_state.drafts.items()}
        self.recorder.recording.provenance['final_turn_execution_s'] = self.fleet_config.decision_interval_s
        # Spectator wire: the same id the live block serves, and in-flight
        # model calls for the deliberation card (AuditedClient registers them)
        battle_id = getattr(self.live_server, 'battle_id', None)
        if isinstance(battle_id, str):
            self.recorder.recording.provenance['battle_id'] = battle_id
        if self.live_pending is not None:
            self.recorder.pending_calls = self.live_pending
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
        # waiting_for is derived per poll from the model calls in flight
        self.live_server.set_battle_status('paused', self.checkpoint_count)
        super()._refresh_targeting_awareness()

    def _log_fleet_decision(self, commands):
        super()._log_fleet_decision(commands)
        self.save_progress()
        if self.ticker:
            try:
                self.ticker.checkpoint(self)
            except Exception as exc:  # noqa: BLE001 - reporting must never stop a paid battle
                print(f'[TICKER] checkpoint block failed: {type(exc).__name__}: {exc}', flush=True)
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


def validate_catalog(catalog, args):
    """Refuse models the free catalog does not show fit for this battle; return pricing."""
    pricing = {}
    for model in (args.alpha_model, args.beta_model):
        entry = catalog.get(model)
        if not entry or 'tools' not in entry.get('supported_parameters', []):
            raise ValueError(f'Model unavailable or lacks tools: {model}')
        if args.reasoning not in (entry.get('reasoning', {}).get('supported_efforts') or []):
            raise ValueError(f'Model does not advertise {args.reasoning} reasoning: {model}')
        modalities = entry.get('architecture', {}).get('input_modalities') or []
        if args.admiral_vision and 'image' not in modalities:
            raise ValueError(f'Model does not advertise image input (--admiral-vision): {model}')
        pricing[model] = entry['pricing']
    return pricing


def make_client(args):
    if args.offline:
        return None, None
    # Free metadata call: availability, tools, reasoning efforts, modalities, prices.
    response = httpx.get('https://openrouter.ai/api/v1/models', timeout=30)
    response.raise_for_status()
    pricing = validate_catalog({m['id']: m for m in response.json()['data']}, args)
    budget = RequestBudget(args.spend_limit, pricing)
    client = CaptainClient(model=args.alpha_model, temperature=None, max_tokens=8192,
                           reasoning_effort=args.reasoning, timeout=180,
                           session_id=f'ai-commanders-{args.name}',
                           api_key='mock-no-network' if args.mock_llm else None)
    if args.mock_llm:
        # Every completion is answered in-process; nothing leaves the machine.
        client._client = httpx.Client(transport=mock_transport(
            pricing, args.points, args.max_ships, args.seed))
    client.budget = budget
    return client, budget


async def run(args):
    output = ROOT / 'visualizer/public/recordings' / f'{args.name}.json'
    report_dir = ROOT / 'data/recordings' / args.name
    if output.exists() or report_dir.exists():
        raise FileExistsError(f'Choose a new --name; {args.name} already exists')
    client, budget = make_client(args)
    report_dir.mkdir(parents=True, exist_ok=True)
    mode = 'offline' if args.offline else 'mock' if args.mock_llm else ''
    ticker = BattleTicker(
        log_path=report_dir / 'ticker.log',
        spend=lambda: (budget.reported_usd, budget.committed_usd) if budget else (0.0, 0.0),
        limit_usd=None if args.offline else args.spend_limit, mode=mode,
        models={'alpha': 'heuristic', 'beta': 'heuristic'} if args.offline else
        {'alpha': args.alpha_model, 'beta': args.beta_model})
    vision_note = ('Admiral vision ON: tactical plot to both admirals each checkpoint.'
                   if args.admiral_vision and not args.offline else
                   'Offline: no LLM admirals, so --admiral-vision is inert (use --mock-llm).'
                   if args.admiral_vision else 'Admiral vision off.')
    ticker.start(args.name, args.points, args.max_ships, args.turns,
                 dict(fleet_draft.SHIP_POINT_COSTS),
                 f'Reasoning {args.reasoning}. {vision_note}')
    models = {'alpha': args.alpha_model, 'beta': args.beta_model}
    shown_models = {'alpha': 'heuristic', 'beta': 'heuristic'} if args.offline else models
    state = LiveDraftState(models=shown_models, budget=args.points)
    pending = PendingCalls()
    server = SpectatorServer(host='localhost', port=args.port)
    server.set_draft_manager(state)
    server.models = shown_models
    server.pending_calls = pending
    # save_progress() keeps this file current, so a viewer that loses the
    # server mid-battle can still load everything up to the last checkpoint
    server.replay_url = f'/recordings/{output.name}'

    def spend():
        return {
            'reported_usd': round(budget.reported_usd, 4) if budget else 0.0,
            'committed_usd': round(budget.committed_usd, 4) if budget else 0.0,
            'limit_usd': None if args.offline else args.spend_limit,
            'calls': client.stats.calls if client else 0,
            'failures': client.stats.failures if client else 0,
            'retries': client.stats.retries if client else 0,
            'truncated': client.stats.truncated if client else 0,
            'mode': mode or 'paid',
        }
    server.spend_provider = spend
    await server.start()
    print(f'WATCH: http://localhost:5173/?live=1&decisions=1', flush=True)
    draft_recorders = {}
    runner = None
    result = None
    finished = False
    fleet_data = load_fleet_data()

    def report():
        return {
            'name': args.name, 'models': models, 'points_per_side': args.points,
            'max_ships': args.max_ships, 'turns': args.turns, 'reasoning': args.reasoning,
            'mode': mode or 'paid', 'admiral_vision': bool(args.admiral_vision and not args.offline),
            'images_sent': budget.images_reserved if budget else 0,
            'point_costs': dict(fleet_draft.SHIP_POINT_COSTS),
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
        recorder.pending_calls = pending   # the draft call shows on the splash
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
                    seed=args.seed, captain_model=model, allow_auto_fallback=False,
                    verbose=not args.quiet)
            finally:
                write_json(report_dir / f'{faction}_draft_calls.json', {
                    'assets': recorder.recording.assets, 'events': [e.to_dict() for e in recorder.events]})
        state.commit(faction, selection)
        write_json(report_dir / f'{faction}_draft.json', draft_summary_dict(selection))
        ticker.draft(faction, draft_summary_dict(selection))
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
                    model=model, name=f'Admiral {model.split("/")[-1]}', temperature=None,
                    # Catalog-verified in make_client; overrides the family allowlist.
                    vision=args.admiral_vision, image_input=True if args.admiral_vision else None),
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
            BattleConfig.from_fleet_config(fc, recording_dir=str(report_dir), verbose=not args.quiet),
            LLMCaptainConfig('Alpha', 'Alpha'), LLMCaptainConfig('Beta', 'Beta'), client, fc)
        runner.configure_live(server, state, draft_recorders, output, args.sim_step_delay,
                              ticker=ticker, vision_dir=report_dir / 'vision',
                              pending_calls=pending)
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
        write_json(report_dir / 'status.json', report())
        if runner and runner.recorder:
            runner.recorder.recording.provenance['billing'] = report()
            runner.save_progress()
        # Only now: a viewer seeing "ended" must already find the verdict,
        # the stop reason and the billing block in the recording
        server.set_battle_status('ended', runner.checkpoint_count if runner else 0)
        print(json.dumps(report()), flush=True)
        try:
            ticker.result(runner, result.winner if result else None,
                          (result.reason if result else None) or state.error,
                          str(output), client.stats.calls if client else 0)
        except Exception as exc:  # noqa: BLE001 - the replay is already saved
            print(f'[TICKER] result block failed: {type(exc).__name__}: {exc}', flush=True)
    if not args.no_linger:
        print('Spectator server remains available; the same link now replays the battle.', flush=True)
        await asyncio.Event().wait()
    # --no-linger: still let a polling viewer collect the final "ended" block
    await server.linger_for_spectators()
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
    parser.add_argument('--mock-llm', action='store_true',
                        help='Free smoke test of the paid path: real catalog checks, budget and '
                             'admiral vision, but every completion is answered in-process')
    parser.add_argument('--admiral-vision', action='store_true',
                        help='Attach the tactical-plot image to each admiral checkpoint call; '
                             'refused unless both models advertise image input')
    parser.add_argument('--quiet', action='store_true',
                        help='Runner verbose logging off; stdout carries the ticker blocks')
    parser.add_argument('--no-linger', action='store_true', help='Close HTTP server after the battle')
    args = parser.parse_args()
    if args.offline and args.mock_llm:
        parser.error('--offline and --mock-llm are exclusive')
    if args.admiral_vision and not args.offline:
        try:
            import matplotlib  # noqa: F401 - the tactical plot renderer
        except ImportError:
            parser.error('--admiral-vision needs matplotlib: uv sync --locked --all-extras')
    if args.turns < 1 or args.points < 1 or args.max_ships < 1 or args.sim_step_delay < 0:
        parser.error('Points, ships and turns must be positive; delay must be nonnegative')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a filename stem')
    # Text followers read stdout through pipes/files: flush every line.
    sys.stdout.reconfigure(line_buffering=True)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
