"""Execution provenance. IDs describe recorded dependencies, not counterfactuals."""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import wraps


def json_value(value):
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith('_')}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items() if not str(k).startswith('_')}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ExecutionEvidence:
    def __init__(self):
        self.counter = 0
        self.decisions = {}
        self.active = {}
        self.commands = {}
        self.command_tools = {}
        self.ordnance = {}
        self.context = []
        self.object_context = None
        self.transitions = {}
        self.dying_sources = {}

    def begin(self, ship_id, command):
        self.counter += 1
        cid = f'c{self.counter}'
        source, tool_id, call_id = self.command_tools.pop(id(command), (None, None, None))
        self.commands[cid] = {'command_id': cid,
                              'decision_id': self.decisions.get(ship_id),
                              'tool_call_id': tool_id if source is command else None,
                              'model_call_event_id': call_id if source is command else None,
                              'command': json_value(command)}
        return cid

    @contextmanager
    def using(self, ids):
        previous = self.context
        self.context = ids
        try:
            yield
        finally:
            self.context = previous

    def bind(self, ship_id, key, cid):
        previous = self.active.setdefault(ship_id, {}).get(key)
        self.active[ship_id][key] = cid
        return previous

    def enrich(self, kind, ship_id, data):
        data = dict(data or {})
        oid = data.get('projectile_id') or data.get('torpedo_id') or self.object_context
        active = self.active.get(ship_id, {})
        ids = list(self.context)
        if kind in ('PROJECTILE_LAUNCHED', 'WEAPON_STATUS') and not ids:
            ids = [active.get('weapon:' + str(data.get('weapon_slot'))), active.get('target')]
        if kind in ('MANEUVER_STARTED', 'MANEUVER_COMPLETED', 'EXECUTION_STATE') and not ids:
            ids = [active.get('maneuver')]
        if oid and kind in ('PROJECTILE_LAUNCHED', 'TORPEDO_LAUNCHED'):
            self.ordnance[oid] = [i for i in ids if i]
        elif oid:
            ids = list(self.ordnance.get(oid, [])) + ids
            if kind == 'TORPEDO_RETARGETED':
                self.ordnance[oid] = list(dict.fromkeys(i for i in ids if i))
        if kind == 'SHIP_DYING':
            self.dying_sources[ship_id] = list(ids)
        elif kind == 'SHIP_DESTROYED':
            ids += self.dying_sources.get(ship_id, [])
        ids = list(dict.fromkeys(i for i in ids if i))
        if ids:
            data['command_ids'] = ids
            data['decision_ids'] = list(dict.fromkeys(
                self.commands[i]['decision_id'] for i in ids
                if i in self.commands and self.commands[i].get('decision_id')))
        if self.object_context and not (data.get('projectile_id') or data.get('torpedo_id')):
            data['ordnance_id'] = self.object_context
        return data


def impact_evidence(method):
    """Carry ordnance identity through nested armor/module/destruction events."""
    @wraps(method)
    def wrapped(self, flight, *args, **kwargs):
        previous = self.evidence.object_context
        self.evidence.object_context = getattr(flight, 'projectile_id', None) or getattr(flight, 'torpedo_id', None)
        try:
            return method(self, flight, *args, **kwargs)
        finally:
            self.evidence.object_context = previous
    return wrapped


def tool_evidence(method):
    """Associate expanded commands (including salvos) with their exact tool call."""
    @wraps(method)
    def wrapped(self, tool, simulation, ship_id):
        result = method(self, tool, simulation, ship_id)
        evidence = getattr(simulation, 'evidence', None)
        if isinstance(getattr(evidence, 'command_tools', None), dict) and result is not None:
            call_id = getattr(self.client, 'last_call_event_id', None)
            if not isinstance(call_id, str):
                call_id = None
            for command in result if isinstance(result, list) else [result]:
                if command is not None:
                    evidence.command_tools[id(command)] = (command, tool.id, call_id)
        return result
    return wrapped
