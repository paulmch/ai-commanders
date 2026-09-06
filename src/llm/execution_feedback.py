"""Compact engine feedback shared by captains and admirals, independent of recording."""
import json


def execution_feedback(simulation, ship_ids):
    ids = set(ship_ids)
    since = simulation.current_time - getattr(simulation, 'decision_interval', 30)
    latest = {}
    problems = []
    for event in getattr(simulation, 'events', []):
        if event.ship_id not in ids:
            continue
        kind = event.event_type.name
        if kind in ('WEAPON_STATUS', 'EXECUTION_STATE'):
            latest[(event.ship_id, event.data.get('weapon_slot', 'helm'))] = {
                'ship': event.ship_id, 'time': event.timestamp, 'target_id': event.target_id,
                **{k: v for k, v in event.data.items() if k in (
                    'weapon_slot', 'reason', 'mode', 'requested_throttle', 'applied_throttle', 'target_id')}}
        elif event.timestamp >= since and kind == 'COMMAND_STATUS' and event.data.get('status') in ('rejected', 'cancelled', 'expired'):
            problems.append({'ship': event.ship_id, 'status': event.data['status'],
                             'reason': event.data.get('reason'), 'command': event.data.get('command')})
    if not latest and not problems:
        return ''
    return ('ENGINE EXECUTION FEEDBACK (observed at the listed times; weapon policies persist, '
            'maneuvers and queued launches expire):\n' + json.dumps(
                {'last_transitions': list(latest.values()), 'command_problems': problems[-12:]}, separators=(',', ':')))
