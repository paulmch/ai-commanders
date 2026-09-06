#!/usr/bin/env python3
"""Record two inexpensive fleet trials with a shared API budget and evidence notes."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient
from src.llm.fleet_config import BattleFleetConfig
from src.llm.replay_commentary import annotate_recording
from src.llm.spending import RequestBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=float, default=2.0, help='Shared USD ceiling across both trials and commentary')
    parser.add_argument('--output', type=Path, default=ROOT / 'visualizer/public/recordings')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'recordings/replay-trials')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    response = httpx.get('https://openrouter.ai/api/v1/models', timeout=30)
    response.raise_for_status()
    catalog = {m['id']: m for m in response.json()['data']}
    models = ['qwen/qwen3.8-flash', 'openai/gpt-oss-120b', 'deepseek/deepseek-v3.2', 'openai/gpt-4.1-mini']
    for model in models:
        if model not in catalog or 'tools' not in catalog[model].get('supported_parameters', []):
            raise ValueError(f'Model unavailable or lacks tool support: {model}')
    pricing = {model: catalog[model]['pricing'] for model in models}
    budget = RequestBudget(args.budget, pricing)
    client = CaptainClient(model=models[0], temperature=0.5, max_tokens=8192,
                           session_id='intent-replay-trials-20260906')
    client.budget = budget
    report = {'budget_usd': args.budget, 'pricing': pricing, 'trials': []}
    data = load_fleet_data()
    trials = [
        ('intent_qwen_gptoss', models[0], models[1], ['destroyer', 'corvette'], 300, 400),
        ('intent_deepseek_mini', models[2], models[3], ['destroyer', 'frigate', 'corvette'], 360, 450),
    ]
    for number, (name, alpha, beta, classes, duration, distance) in enumerate(trials):
        start_cost = budget.reported_usd
        config = {'battle_name': f'Intent & consequence: {alpha.split("/")[-1]} vs {beta.split("/")[-1]}',
                  'time_limit_s': duration, 'initial_distance_km': distance, 'decision_interval_s': 30,
                  'max_checkpoints': 14, 'seed': 20260906 + number,
                  'personality_selection': False, 'record_battle': True, 'record_sim_trace': True}
        for faction, model in [('alpha', alpha), ('beta', beta)]:
            config[f'{faction}_fleet'] = {'admiral': {'model': model, 'name': f'{faction.title()} Admiral'},
                'ships': [{'ship_type': ship, 'model': model, 'ship_name': f'{faction.title()} {ship.title()}',
                           'captain_name': f'{model.split("/")[-1]} Captain {i + 1}'} for i, ship in enumerate(classes)]}
        (args.report_dir / f'{name}_config.json').write_text(json.dumps(config, indent=2))
        fc = BattleFleetConfig.from_dict(config)
        runner = LLMBattleRunner(BattleConfig.from_fleet_config(fc, verbose=False, recording_dir=str(args.report_dir)),
                                LLMCaptainConfig('Alpha', 'Alpha'), LLMCaptainConfig('Beta', 'Beta'), client, fc)
        print(f'Starting {name}; total reported spend ${budget.reported_usd:.4f}', flush=True)
        try:
            with (args.report_dir / f'{name}.log').open('w') as log, redirect_stdout(log):
                result = runner.run_battle(data)
                recording = json.loads(Path(result.recording_file).read_text())
                annotate_recording(recording, client, models[0], limit=3)
            recording.setdefault('provenance', {})['trial_cost_usd'] = budget.reported_usd - start_cost
            output = args.output / f'{name}.json'
            output.write_text(json.dumps(recording, separators=(',', ':')))
            report['trials'].append({'name': name, 'recording': str(output), 'winner': result.winner,
                                    'duration_s': result.duration_s, 'cost_usd': budget.reported_usd - start_cost,
                                    'commentary_count': len(recording.get('commentary', [])),
                                    'model_calls': sum(e['event_type'] == 'model_call' for e in recording['events'])})
        finally:
            report.update(reported_cost_usd=budget.reported_usd, committed_cost_usd=budget.committed_usd,
                          api_calls=client.stats.calls, failures=client.stats.failures,
                          truncated=client.stats.truncated, malformed_arguments=client.stats.malformed_arguments)
            (args.report_dir / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['trials'][-1]), flush=True)
    print(f'Total reported ${budget.reported_usd:.4f}; committed ${budget.committed_usd:.4f}', flush=True)


if __name__ == '__main__':
    main()
