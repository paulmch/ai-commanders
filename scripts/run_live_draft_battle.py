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
from src.llm.anthropic_client import ANTHROPIC_MODELS, ANTHROPIC_PRICES_AS_OF, EFFORT_LEVELS
from src.llm.mock_anthropic import mock_anthropic_transport
from src.llm.providers import create_llm_client, provider_for_model
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
from src.llm import fleet_draft
from src.llm.fleet_draft import auto_draft, draft_summary_dict, draft_to_fleet_definition, run_admiral_draft
from src.llm.live_status import PendingCalls, draft_live_entry
from src.llm.mcp_http_server import MCPHttpServer
from src.llm.mock_openrouter import mock_transport
from src.llm.replay_audit import AuditedClient
from src.llm.spending import RequestBudget, SpendLedger

DEFAULT_LEDGER = ROOT / 'data/recordings/spend_ledger.jsonl'
# After Ctrl-C: how long the battle thread gets to finish the calls in flight
# (the client's 180 s request timeout plus a margin) before the ledger closes.
STOP_WAIT_S = 240
from src.llm.call_policy import EffortPolicy, PolicyClient, parse_effort_specs
from src.llm.cost_report import cost_report, format_cost_report


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
    FLIES_LAST_ORDERS = True  # _evaluate_fleet_result flies the last orders (admiral clock)
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
        budget = self.client.budget if self.client else None
        if budget and getattr(budget, 'stopped', None):
            raise RuntimeError(f'Battle interrupted ({budget.stopped}); no further model calls')
        if budget and budget.exhausted:
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


def anthropic_pricing(model, args):
    """Price-table entry for a bare Anthropic id (served by the Anthropic API,
    not in the OpenRouter catalog); unknown ids and efforts raise."""
    provider_for_model(model)   # unknown claude-* ids raise here
    if args.reasoning not in EFFORT_LEVELS:
        raise ValueError(f'Anthropic models take --reasoning {"/".join(EFFORT_LEVELS)}: {model}')
    # Every model in the table accepts tools and image input.
    return {**{k: v for k, v in ANTHROPIC_MODELS[model].items() if k != 'long'},
            'unit': 'USD per million tokens', 'as_of': ANTHROPIC_PRICES_AS_OF, 'provider': 'anthropic'}


def ledger_tokens(client):
    stats = getattr(client, 'stats', None)
    if stats is None:
        return {}
    return {'prompt': stats.prompt_tokens, 'cache_read': stats.cached_tokens,
            'cache_write': stats.cache_write_tokens}


def fixed_rosters(args, admiral_for):
    """The fleet config of an earlier run with this run's models, modes and turn count."""
    path = Path(args.rosters)
    if path.is_dir():
        path = path / 'fleet_config.json'
    fc = BattleFleetConfig.from_json(str(path))
    for faction in ('alpha', 'beta'):
        fleet = getattr(fc, f'{faction}_fleet')
        captain = 'heuristic' if args.offline else side_models(args)[faction]['captain']
        for ship in fleet.ships:
            ship.model = captain
            ship.temperature = None
        fleet.admiral = admiral_for(faction)
    fc.battle_name = f'fixed rosters ({path.parent.name}): {args.alpha_model} vs {args.beta_model}'
    fc.max_checkpoints = args.turns
    fc.time_limit_s = (args.turns + 1) * 30
    fc.personality_selection = False
    fc.record_battle = True
    fc.record_sim_trace = True
    fc.use_notebooks = False
    return fc


def record_stop(runner, error):
    """A stopped battle (spend ceiling, failed draft) is a recorded outcome:
    result_reason, a battle_stopped event and the partial replay on disk."""
    if runner and runner.recorder:
        runner.recorder.recording.result_reason = error
        runner.recorder.record(runner.simulation.current_time, 'battle_stopped', reason=error)
        runner.save_progress()


def side_models(args):
    """{'alpha': {'admiral': id, 'captain': id}, 'beta': {...}}: the admiral (and
    draft) model and the captain model of each side (captains default to it)."""
    return {f: {'admiral': getattr(args, f'{f}_model'),
                'captain': getattr(args, f'{f}_captain_model', None) or getattr(args, f'{f}_model')}
            for f in ('alpha', 'beta')}


def all_models(args):
    """Every LLM model id this battle may call, first-seen order ('heuristic' captains excluded)."""
    from src.llm.providers import llm_models
    return llm_models(m for roles in side_models(args).values() for m in roles.values())


def sides_of(value):
    return () if not value else ('alpha', 'beta') if value == 'both' else (value,)


def validate_catalog(catalog, args):
    """Refuse models the free catalog does not show fit for this battle; return pricing."""
    pricing = {}
    for model in all_models(args):
        if provider_for_model(model) == 'anthropic':
            pricing[model] = anthropic_pricing(model, args)
            continue
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
    models = tuple(all_models(args))
    catalog = {}
    if any(provider_for_model(m) == 'openrouter' for m in models):
        # Free metadata call: availability, tools, reasoning efforts, modalities, prices.
        response = httpx.get('https://openrouter.ai/api/v1/models', timeout=30)
        response.raise_for_status()
        catalog = {m['id']: m for m in response.json()['data']}
    pricing = validate_catalog(catalog, args)
    budget = RequestBudget(args.spend_limit, pricing)
    anthropic_options = None
    if args.mock_llm:
        import anthropic
        anthropic_options = {'api_key': 'mock-no-network', 'http_client': anthropic.DefaultHttpxClient(
            transport=mock_anthropic_transport(args.points, args.max_ships, args.seed))}
    if args.thinking_summary:
        anthropic_options = {**(anthropic_options or {}), 'thinking_display': 'summarized'}
    if args.cache_warmup:
        anthropic_options = {**(anthropic_options or {}), 'cache_warmup': True}
    # A bare Anthropic id is served by the Anthropic API, a vendor/model id by
    # OpenRouter; the sides may differ. Fails here without ANTHROPIC_API_KEY.
    client = create_llm_client(
        args.alpha_model, models, temperature=None, max_tokens=8192,
        reasoning_effort=args.reasoning, timeout=180, session_id=f'ai-commanders-{args.name}',
        anthropic_options=anthropic_options,
        openrouter_options={'api_key': 'mock-no-network'} if args.mock_llm else None)
    if args.mock_llm:
        # Every completion is answered in-process; nothing leaves the machine.
        routed = getattr(client, 'clients', None)
        for sub in (routed.values() if routed is not None else [client]):
            if getattr(sub, 'provider', None) == 'openrouter':
                sub._client = httpx.Client(transport=mock_transport(
                    pricing, args.points, args.max_ships, args.seed))
    client.budget = budget
    return client, budget


async def run(args):
    output = ROOT / 'visualizer/public/recordings' / f'{args.name}.json'
    report_dir = ROOT / 'data/recordings' / args.name
    if output.exists() or report_dir.exists():
        raise FileExistsError(f'Choose a new --name; {args.name} already exists')
    client, budget = make_client(args)
    # Paid runs reserve their cap in the cumulative ledger before the first
    # request (--ledger-cap refuses the run here); mock runs only when a
    # ledger is named explicitly, and they never count.
    ledger = ledger_run = None
    if not args.offline and (not args.mock_llm or args.ledger):
        ledger = SpendLedger(args.ledger or DEFAULT_LEDGER)
        ledger.check(args.spend_limit, args.ledger_cap)     # refuse before any pre-flight work
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
    if ledger is not None:
        # Reserve the cap only now (after the port is bound): the first paid
        # request is the draft, inside the battle task below.
        ledger_run = ledger.open_run(script='run_live_draft_battle', name=args.name,
                                     models=all_models(args), cap_usd=args.spend_limit,
                                     ledger_cap_usd=args.ledger_cap, mock=args.mock_llm)
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
            'role_models': side_models(args),
            'effort_policy': parse_effort_specs(args.effort),
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
            'refusals': client.stats.refusals if client else 0,
            'refusals_by_side': dict(client.stats.refusals_by_side) if client else {},
            'computed_cost_usd': round(client.stats.cost_usd, 6) if client else 0,
            'cache': client.stats.cache_summary() if client else None,
            'prompt_tokens': client.stats.prompt_tokens if client else 0,
            'cached_tokens': client.stats.cached_tokens if client else 0,
            'cache_write_tokens': client.stats.cache_write_tokens if client else 0,
            'winner': result.winner if result else None,
            'result_reason': result.reason if result else None,
            'recording': str(output),
            'experiment': {'no_admiral': args.no_admiral, 'intent_only': args.intent_only,
                           'quiet_turns': args.quiet_turns, 'cache_warmup': args.cache_warmup,
                           'rosters': args.rosters, 'thinking_summary': args.thinking_summary},
            'cost_report': battle_cost_report(),
        }

    def battle_cost_report():
        if runner and runner.recorder:
            events = list(runner.recorder.events)
        else:
            events = [e for r in list(draft_recorders.values()) for e in list(r.events)]
        return cost_report(events)

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
            policy = EffortPolicy(parse_effort_specs(args.effort))
            drafter = PolicyClient(client, faction, 'admiral', policy) if policy else client
            audited = AuditedClient(drafter, recorder, SimpleNamespace(current_time=0), 'admiral', faction)
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

    def admiral_for(faction):
        model = models[faction]
        if args.offline or faction in sides_of(args.no_admiral):
            return None
        return AdmiralConfig(
            model=model, name=f'Admiral {model.split("/")[-1]}', temperature=None,
            # Catalog-verified in make_client; overrides the family allowlist.
            vision=args.admiral_vision, image_input=True if args.admiral_vision else None,
            intent_only=faction in sides_of(args.intent_only))

    def battle():
        nonlocal runner, result
        if args.rosters:
            fc = fixed_rosters(args, admiral_for)
            selections = {}
        else:
            # Independent sealed drafts; neither model receives the other's choices.
            with ThreadPoolExecutor(max_workers=2) as pool:
                selections = dict(zip(models, pool.map(draft, models)))
            fleets = {}
            for faction, model in models.items():
                fleets[faction] = draft_to_fleet_definition(
                    selections[faction], 500,
                    captain_model='heuristic' if args.offline else side_models(args)[faction]['captain'],
                    admiral_config=admiral_for(faction), temperature=None)
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
            BattleConfig.from_fleet_config(fc, recording_dir=str(report_dir), verbose=not args.quiet,
                                           effort_policy=parse_effort_specs(args.effort),
                                           quiet_turns=args.quiet_turns),
            LLMCaptainConfig('Alpha', 'Alpha'), LLMCaptainConfig('Beta', 'Beta'), client, fc)
        runner.configure_live(server, state, draft_recorders, output, args.sim_step_delay,
                              ticker=ticker, vision_dir=report_dir / 'vision',
                              pending_calls=pending)
        server.set_live_source(runner)
        result = runner.run_battle(fleet_data)
        if selections:
            write_json(output.with_suffix('.draft.json'), {
                'budget': args.points, **{f: draft_summary_dict(d) for f, d in selections.items()}})

    task = asyncio.create_task(asyncio.to_thread(battle))
    interrupted = False
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
        record_stop(runner, state.error)
    except BaseException:
        # Ctrl-C: asyncio cancels this coroutine, but the battle runs in a
        # worker thread that nothing cancels - it would go on buying calls
        # (and asyncio.run waits for it). Close the budget so no new request
        # is sent, let the calls in flight settle, then record the stop.
        interrupted = True
        state.error = 'Interrupted (Ctrl-C): no new model calls after the interrupt'
        state.stage = 'Battle stopped'
        print(state.error, flush=True)
        if budget is not None:
            budget.stop('Ctrl-C')
        await asyncio.wait({task}, timeout=STOP_WAIT_S)
        if task.done() and not task.cancelled():
            task.exception()    # the thread's own "Battle interrupted" stop: retrieved, expected
        raise
    finally:
        finished = True
        state.active = False
        if ledger_run is not None:
            # First, before anything below can fail: an unclosed ledger line
            # counts the whole cap, a closed one what was spent or committed.
            ledger.close_run(ledger_run, calls=client.stats.calls if client else 0,
                             tokens=ledger_tokens(client), usd=budget.reported_usd if budget else 0.0,
                             committed_usd=budget.committed_usd if budget else None,
                             status='interrupted' if interrupted else 'error' if state.error else 'done')
        write_json(report_dir / 'status.json', report())
        if task.done():
            # (A battle thread still running would race the recorder.)
            if interrupted:
                record_stop(runner, state.error)
            if runner and runner.recorder:
                runner.recorder.recording.provenance['billing'] = report()
                runner.save_progress()
        # Only now: a viewer seeing "ended" must already find the verdict,
        # the stop reason and the billing block in the recording
        server.set_battle_status('ended', runner.checkpoint_count if runner else 0)
        print(json.dumps(report()), flush=True)
        print(format_cost_report(battle_cost_report()), flush=True)
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
    parser.add_argument('--reasoning', default='high',
                        help='Battle-wide effort: every call not covered by --effort')
    parser.add_argument('--effort', action='append', default=[], metavar='[SIDE.]ROLE=LEVEL',
                        help='Effort for one role (draft, directive, order, captain, trigger) on one '
                             'side (alpha, beta; omitted = both), repeatable, later wins. "trigger" '
                             'is the captain effort on turns with torpedoes inbound, a merge warning, '
                             'damage taken, a changed order kind or the first turn; "captain" the '
                             'other turns. Example: --effort directive=high --effort alpha.trigger=high')
    parser.add_argument('--alpha-captain-model', default=None,
                        help='Captain model of alpha (default: --alpha-model, which also drafts '
                             'and commands as admiral)')
    parser.add_argument('--beta-captain-model', default=None, help='Captain model of beta')
    parser.add_argument('--ledger', default=None,
                        help=f'Cumulative spend ledger (JSONL). Paid runs append to it (default '
                             f'{DEFAULT_LEDGER.relative_to(ROOT)}); a mock run only when named here')
    parser.add_argument('--ledger-cap', type=float, default=None,
                        help='Refuse to start when the ledger total (closed runs at their spend, '
                             'open runs at their cap) plus this run\'s --spend-limit would exceed it')
    sides = ('alpha', 'beta', 'both')
    parser.add_argument('--no-admiral', choices=sides, default=None,
                        help='Experiment: that side drafts with its model but fights without an '
                             'admiral (LLM captains on their own)')
    parser.add_argument('--intent-only', choices=sides, default=None,
                        help='Experiment: that side\'s admiral sets directive and plan (and may '
                             'order torpedo salvos) but issues no per-ship orders')
    parser.add_argument('--quiet-turns', action='store_true',
                        help='Experiment: skip a captain call on a turn with no trigger, no new '
                             'FLEET NET traffic and an unchanged applied order')
    parser.add_argument('--cache-warmup', action='store_true',
                        help='Experiment (Anthropic ids): one max_tokens=0 pre-warm per cold cached '
                             'prefix before a parallel fan-out reads it')
    parser.add_argument('--rosters', default=None,
                        help='Fixed fleets: skip the draft and fly the fleet_config.json of an earlier '
                             'run (a file or its report folder) with this run\'s models; its seed, '
                             'rosters, names and positions are kept')
    parser.add_argument('--thinking-summary', action='store_true',
                        help='Anthropic ids: request thinking display "summarized" on every call and '
                             'record the summary per model_call (thinking_summary; audit only, never '
                             'fed back). Billed the same as the default omitted display')
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
    try:
        parse_effort_specs(args.effort)
    except ValueError as exc:
        parser.error(str(exc))
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
