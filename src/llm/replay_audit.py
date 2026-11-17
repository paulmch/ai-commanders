"""Record commander inputs once, retaining references from each model call."""
import time

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

    def _observe(self, payload, response, started=None):
        # Store public output/tool calls only. Provider reasoning is not replay prose.
        choices = response.get('choices') or [{}]
        message = choices[0].get('message') or {}
        request = {k: v for k, v in payload.items() if k not in ('messages', 'tools')}
        # Role and effort of this call (src/llm/call_policy.py; optional fields,
        # docs/pass-5-recording-additions.md): role from the wrapper's
        # annotation or the offered tools; effort as actually sent.
        from .call_policy import call_role, tool_names
        annotation = getattr(self.client, 'call_annotation', None)
        annotation = annotation() if callable(annotation) else None
        annotation = annotation if isinstance(annotation, dict) else {}
        role = annotation.get('role') or call_role(self.actor, None, tool_names(payload.get('tools')))
        effort = ((request.get('output_config') or {}).get('effort')
                  or (request.get('reasoning') or {}).get('effort'))
        extra = {'role': role}
        if effort:
            extra['effort'] = effort
        if annotation.get('effort_reason'):
            extra['effort_reason'] = annotation['effort_reason']
        if response.get('cache_warmup'):
            # The cache pre-warm this call waited for (cache_warmup mode).
            extra['cache_warmup'] = response['cache_warmup']
        if response.get('thinking_summary'):
            # Summarized thinking (AnthropicClient thinking_display): for audits
            # only - never read back into any prompt.
            extra['thinking_summary'] = response['thinking_summary']
        event = self.recorder.record(
            self.simulation.current_time, 'model_call', self.ship_id,
            actor=self.actor, faction=self.faction, **extra,
            request=request, model=response.get('model', payload.get('model')),
            generation_id=response.get('id'), usage=response.get('usage', {}),
            message_refs=[self.recorder.asset(m) for m in payload.get('messages', [])],
            tools_ref=self.recorder.asset(payload.get('tools', [])),
            output={'content': message.get('content'), 'tool_calls': message.get('tool_calls', [])},
            finish_reason=choices[0].get('finish_reason'),
            **({'provider': response['provider']} if response.get('provider') else {}),
            **({'stop_reason': response['stop_reason']} if response.get('stop_reason') else {}),
            # Wall seconds from the agent's call to this response (spectator
            # pacing only; never read by the simulation).
            **({'latency_s': round(time.monotonic() - started, 2)} if started is not None else {}))
        self.last_call_event_id = event.event_id
        if choices[0].get('finish_reason') == 'refusal':
            # A refused turn issues no orders; the log says so, per side.
            details = response.get('stop_details') or {}
            self.recorder.record(
                self.simulation.current_time, 'model_refusal', self.ship_id,
                actor=self.actor, faction=self.faction,
                model=response.get('model', payload.get('model')),
                category=details.get('category'), explanation=details.get('explanation'),
                call_event_id=event.event_id)
            stats = getattr(self.client, 'stats', None)
            if stats is not None and hasattr(stats, 'record_refusal_side'):
                stats.record_refusal_side(self.faction)

    def _call(self, method, *args, **kwargs):
        # Live spectators see in-flight calls (who the checkpoint waits for)
        # when the runner hangs a live_status.PendingCalls on the recorder.
        pending = getattr(self.recorder, 'pending_calls', None)
        token = pending.start(self.actor, self.faction, self.ship_id,
                              kwargs.get('model') or getattr(self.client, 'model', None)) if pending is not None else None
        started = time.monotonic()
        try:
            with self.client.audit_scope(lambda payload, response: self._observe(payload, response, started)):
                return getattr(self.client, method)(*args, **kwargs)
        except Exception as exc:
            self.recorder.record(self.simulation.current_time, 'model_error', self.ship_id,
                                 actor=self.actor, faction=self.faction, error=str(exc),
                                 model=kwargs.get('model', self.client.model),
                                 message_refs=[self.recorder.asset(m) for m in (args[0] if args else kwargs.get('messages', []))],
                                 tools_ref=self.recorder.asset(args[1] if len(args) > 1 else kwargs.get('tools', [])))
            raise
        finally:
            if token is not None:
                pending.finish(token)

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
    from .anthropic_client import AnthropicClient
    from .client import CaptainClient
    from .providers import LLMRouter, llm_provenance
    real_clients = (CaptainClient, AnthropicClient, LLMRouter)
    side_models = {}
    for agent, actor, faction, sid in entries:
        model = getattr(getattr(agent, 'config', None), 'model', None)
        side_models.setdefault(faction, []).append(model)
    if isinstance(runner.client, real_clients) and any(side_models.values()):
        # Which provider and exact model id served each side, at which effort
        # (a January bug had one model silently playing both sides).
        runner.recorder.recording.provenance['llm'] = llm_provenance(runner.client, side_models)
    for agent, actor, faction, sid in entries:
        client = getattr(agent, 'client', None)
        # An experiment wrapper (call_policy.PolicyClient) sits inside the audit.
        if isinstance(getattr(client, 'inner', client), real_clients):
            agent.client = AuditedClient(client, runner.recorder, runner.simulation, actor, faction, sid)
