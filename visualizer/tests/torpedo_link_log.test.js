// Pass 5 retarget lane: torpedo launch profile and datalink reassignments in
// the combat log. Old recordings (no profile, no reason) read as before.
import test from 'node:test';
import assert from 'node:assert/strict';
import { formatLogEvent } from '../src/HudIndex.js';

const ev = (event_type, ship_id, data) => ({ event_type, ship_id, data, timestamp: 1 });

test('torpedo_launched names a non-default profile only', () => {
  const old = formatLogEvent(ev('torpedo_launched', 'alpha_1', { target_id: 'beta_1' }));
  assert.ok(!/profile/.test(old.text));
  const res = formatLogEvent(ev('torpedo_launched', 'alpha_1', { target_id: 'beta_1', profile: 'reserve' }));
  assert.match(res.text, /\(reserve profile\)$/);
});

test('torpedo_retargeted shows the datalink reason when present', () => {
  const legacy = formatLogEvent(ev('torpedo_retargeted', 'alpha_1',
    { torpedo_id: 't1', old_target_id: 'beta_1', new_target_id: 'beta_2' }));
  assert.ok(!/datalink/.test(legacy.text));
  const lost = formatLogEvent(ev('torpedo_retargeted', 'alpha_1',
    { torpedo_id: 't1', old_target_id: 'beta_1', new_target_id: 'beta_2',
      reason: 'target_lost', group: 'alpha@102:target_lost', group_size: 6 }));
  assert.match(lost.text, /datalink: .*lost, group of 6/);
  const surplus = formatLogEvent(ev('torpedo_retargeted', 'alpha_1',
    { torpedo_id: 't1', old_target_id: 'beta_1', new_target_id: 'beta_2',
      reason: 'datalink_surplus', group_size: 1 }));
  assert.match(surplus.text, /datalink: surplus on/);
});
