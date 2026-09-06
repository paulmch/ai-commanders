"""Record commander inputs once, retaining references from each model call."""
from .client import CaptainClient
from src.replay_evidence import json_value


class AuditedClient:
    def __init__(self, client, recorder, simulation, actor, faction, ship_id=None):
        self.client = client
        self.recorder = recorder
        self.simulation = simulation
        self.actor = actor
        self.faction = faction
        self.ship_id = ship_id

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _observe(self, payload, response):
        # Store public output/tool calls only. Provider reasoning is not replay prose.
        choices = response.get('choices') or [{}]
        message = choices[0].get('message') or {}
        request = {k: v for k, v in payload.items() if k not in ('messages', 'tools')}
        event = self.recorder.record(
            self.simulation.current_time, 'model_call', self.ship_id,
            actor=self.actor, faction=self.faction,
            request=request, model=response.get('model', payload.get('model')),
            generation_id=response.get('id'), usage=response.get('usage', {}),
            message_refs=[self.recorder.asset(m) for m in payload.get('messages', [])],
            tools_ref=self.recorder.asset(payload.get('tools', [])),
            output={'content': message.get('content'), 'tool_calls': message.get('tool_calls', [])},
            finish_reason=choices[0].get('finish_reason'))
        self.last_call_event_id = event.event_id

    def _call(self, method, *args, **kwargs):
        try:
            with self.client.audit_scope(self._observe):
                return getattr(self.client, method)(*args, **kwargs)
        except Exception as exc:
            self.recorder.record(self.simulation.current_time, 'model_error', self.ship_id,
                                 actor=self.actor, faction=self.faction, error=str(exc),
                                 model=kwargs.get('model', self.client.model),
                                 message_refs=[self.recorder.asset(m) for m in (args[0] if args else kwargs.get('messages', []))],
                                 tools_ref=self.recorder.asset(args[1] if len(args) > 1 else kwargs.get('tools', [])))
            raise

    def decide_with_tools(self, *args, **kwargs):
        return self._call('decide_with_tools', *args, **kwargs)

    def complete(self, *args, **kwargs):
        return self._call('complete', *args, **kwargs)


def attach_audit(runner):
    if not runner.recorder:
        return
    runner.simulation.recorder = runner.recorder
    import hashlib
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Hash actual working files too: HEAD alone does not identify a dirty checkout.
    source_hash = hashlib.sha256()
    for path in sorted((root / 'src').rglob('*.py')):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    runner.recorder.recording.provenance = {
        'seed': runner.config.seed, 'decision_interval_s': runner.simulation.decision_interval,
        'engine_revision': revision, 'source_sha256': source_hash.hexdigest(),
        'fleet_data_sha256': hashlib.sha256((root / 'data/fleet_ships.json').read_bytes()).hexdigest(),
        'config': json_value(runner.config), 'attribution': 'execution_dependencies',
    }
    entries = []
    for faction in ('alpha', 'beta'):
        captains = getattr(runner, faction + '_captains', {}) or {}
        if not captains:
            captain = getattr(runner, faction + '_captain', None)
            if captain:
                captains = {faction: captain}
        entries.extend((captain, 'captain', faction, sid) for sid, captain in captains.items())
        admiral = getattr(runner, faction + '_admiral', None)
        if admiral:
            entries.append((admiral, 'admiral', faction, None))
    for agent, actor, faction, sid in entries:
        client = getattr(agent, 'client', None)
        if isinstance(client, CaptainClient):
            agent.client = AuditedClient(client, runner.recorder, runner.simulation, actor, faction, sid)
