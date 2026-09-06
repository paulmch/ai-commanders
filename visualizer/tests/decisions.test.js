import test from 'node:test';
import assert from 'node:assert/strict';
import { DecisionIndex } from '../src/DecisionIndex.js';
import { BattleLoader } from '../src/BattleLoader.js';

const e = (id, t, kind, data = {}, ship_id = 'a') => ({event_id: id, sequence: Number(id.slice(1)) || 0, timestamp: t, event_type: kind, ship_id, data});
const ships = {a: {faction: 'alpha', name: 'A'}, b: {faction: 'beta', name: 'B'}};

test('a delayed impact stays with the firing decision across checkpoints and seeking', () => {
  const events = [e('e1', 30, 'captain_decision', {commands: []}),
    e('e2', 31, 'shot_fired', {projectile_id: 'p1', decision_ids: ['e1'], target_id: 'b'}),
    e('e3', 60, 'captain_decision', {commands: []}),
    e('e4', 75, 'hit', {projectile_id: 'p1', decision_ids: ['e1'], target_id: 'b'}),
    e('e5', 75, 'module_destroyed', {ordnance_id: 'p1', decision_ids: ['e1'], module_name: 'Engine'}, 'b')];
  const index = new DecisionIndex(events, ships);
  assert.deepEqual(index.detail('e3', 80).outcomes, []);
  const before = index.detail('e1', 50);
  assert.equal(before.outcomes.length, 1);
  assert.ok(before.facts.some(f => f.text.includes('no recorded terminal outcome')));
  const after = index.detail('e1', 80);
  assert.equal(after.outcomes.length, 3);
  assert.ok(after.facts.some(f => f.text === '1 modules destroyed by linked impacts.'));
  assert.ok(!after.facts.some(f => f.text.includes('no recorded terminal outcome')));
  assert.deepEqual(index.detail('e1', 50), before);
  assert.ok(after.facts.every(f => f.event_ids.every(id => index.byId.has(id))));
});

test('legacy decisions show chronological context without claiming command attribution', () => {
  const index = new DecisionIndex([
    {timestamp: 20, event_type: 'admiral_directive', data: {faction: 'alpha', directive: 'Cover the flagship.'}},
    {timestamp: 30, event_type: 'captain_decision', ship_id: 'a', data: {maneuver_type: 'EVASIVE'}},
    {timestamp: 45, event_type: 'hit', ship_id: 'a', data: {target_id: 'b'}},
    {timestamp: 60, event_type: 'captain_decision', ship_id: 'a', data: {maneuver_type: 'MAINTAIN'}},
    {timestamp: 65, event_type: 'hit', ship_id: 'a', data: {target_id: 'b'}},
  ], ships);
  const d = index.detail(index.history('a')[0].event_id, 90);
  assert.equal(d.modern, false);
  assert.equal(d.context.length, 1);
  assert.equal(d.outcomes.length, 1);
  assert.ok(d.facts.some(f => f.text.includes('during this decision window')));
});

test('late same-time events merge once, retain real duplicate legacy events, and update assets', () => {
  const loader = new BattleLoader();
  loader.parse({events: [e('e1', 30, 'checkpoint')], sim_trace: [{t: 30, ships: {}, projectiles: [], torpedoes: []}]});
  const chunk = {events: [e('e2', 30, 'captain_decision', {commands: []})], assets: {prompt: 'new input'}};
  assert.equal(loader.appendLiveData(chunk), true);
  assert.equal(loader.decisions.history('a').length, 1);
  assert.equal(loader.sourceData.assets.prompt, 'new input');
  assert.equal(loader.lastEventSequence, 2);
  assert.equal(loader.appendLiveData(chunk), false);
  assert.equal(loader.appendLiveData({events: [e('e1', 30, 'checkpoint'), ...chunk.events]}), false);
  const legacy = {timestamp: 30, event_type: 'captain_log', ship_id: 'a', data: {note: 'Same note'}};
  assert.equal(loader.appendLiveData({events: [legacy, legacy]}), true);
  assert.equal(loader.events.length, 4);
  assert.equal(loader.appendLiveData({events: [legacy, legacy]}), false);
});

test('seeker disable is an intermediate event, while destruction resolves a round', () => {
  const index = new DecisionIndex([e('e1', 0, 'captain_decision', {commands: []}),
    e('e2', 1, 'torpedo_launched', {torpedo_id: 't1', decision_ids: ['e1']}),
    e('e3', 2, 'pd_torpedo_disabled', {torpedo_id: 't1', decision_ids: ['e1']}),
    e('e4', 3, 'torpedo_impact', {torpedo_id: 't1', decision_ids: ['e1']})], ships);
  assert.ok(index.detail('e1', 2).facts.some(f => f.text.includes('no recorded terminal outcome')));
  assert.ok(!index.detail('e1', 3).facts.some(f => f.text.includes('no recorded terminal outcome')));
});
