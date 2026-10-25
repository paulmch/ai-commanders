import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { BattleLoader, lowerBoundByTime } from '../src/BattleLoader.js';

// LV-15: a live poll extends the derived indexes instead of rebuilding them.
// The contract: parse(head) + appendLiveData(chunks...) leaves the loader in
// exactly the state parse(everything) would. The comparison covers EVERY
// loader field, so a derived index added later without an incremental path
// (or a full rebuild on append) shows up here.

const IGNORED = new Set(['sourceData', '_eventKeys', '_deathScan', '_pdOpen']);

function assertSameLoader(actual, expected, label) {
  const keys = new Set([...Object.keys(actual), ...Object.keys(expected)].filter(k => !IGNORED.has(k)));
  for (const key of keys) {
    if (typeof expected[key] === 'function') continue;
    assert.deepStrictEqual(actual[key], expected[key], `${label}: loader.${key} differs`);
  }
  assert.deepStrictEqual(actual.buildDeathInfo(), expected.buildDeathInfo(), `${label}: death info differs`);
}

function syntheticRecording() {
  const ships = ['alpha_1', 'alpha_2', 'beta_1', 'beta_2'];
  const frames = [];
  for (let t = 0; t <= 60; t++) {
    const frame = { t, ships: {} };
    for (const [i, id] of ships.entries()) {
      const dir = id.startsWith('alpha') ? 1 : -1;
      frame.ships[id] = {
        pos: [dir * (200000 - t * 1500), i * 20000, 0], vel: [-dir * 1500, 0, 0], fwd: [-dir, 0, 0],
        hull: id === 'beta_2' ? Math.max(0, 100 - Math.max(0, t - 30) * 4) : 100,
        armor: { nose: 200 - (id === 'beta_2' && t > 26 ? 40 : 0), lateral: 36, tail: 42 },
        dying: id === 'beta_2' && t >= 40 && t < 45, destroyed: id === 'beta_2' && t >= 45,
      };
    }
    if (t >= 9 && t <= 11) frame.projectiles = [{ id: 'p1', pos: [t * 1000, 0, 0], vel: [9000, 0, 0], source: 'alpha_1', target: 'beta_1', mass_kg: 50 }];
    if (t >= 18 && t <= 25) frame.torpedoes = [{ id: 'torp_1', pos: [t * 2000, 0, 0], vel: [12000, 0, 0], source: 'beta_1', target: 'alpha_1', dv_remaining: 5, disabled: t >= 22 }];
    frames.push(frame);
  }
  const events = [];
  const add = (timestamp, event_type, ship_id, data = {}) =>
    events.push({ event_id: `e${events.length + 1}`, sequence: events.length + 1, timestamp, event_type, ship_id, data });
  add(0, 'battle_start', null);
  add(9, 'shot_fired', 'alpha_1', { target_id: 'beta_1', projectile_id: 'p1' });
  add(12, 'hit', 'beta_1', { projectile_id: 'p1', impact_position: [12000, 0, 0], kinetic_energy_gj: 3.5, shooter_id: 'alpha_1' });
  add(12, 'hit', 'beta_1', { projectile_id: 'p1', impact_position: [12100, 0, 0], kinetic_energy_gj: 9.9, shooter_id: 'alpha_1' });
  add(15, 'radiator_change', 'alpha_2', { extended: true });
  add(18, 'torpedo_launched', 'beta_1', { torpedo_id: 'torp_1', target_id: 'alpha_1', warhead_gj: 8 });
  for (const t of [20, 21, 22, 23, 24, 29, 30, 31]) add(t, 'pd_fired', 'alpha_1', { turret_name: 'pd1', target_id: 'torp_1', target_type: 'torpedo', energy_delivered_j: 1e6 });
  add(22, 'pd_torpedo_disabled', 'alpha_1', { torpedo_id: 'torp_1' });
  add(26, 'torpedo_impact', 'alpha_1', { torpedo_id: 'torp_1', target_id: 'alpha_1', total_damage_gj: 8, hit_location: 'nose' });
  add(27, 'module_damaged', 'beta_2', { module_name: 'sensor_array', damage_gj: 2 });
  // A checkpoint: decisions arrive over several polls at the same time
  add(30, 'model_call', null, { actor: 'admiral', faction: 'alpha', latency_s: 6.5 });
  add(30, 'admiral_directive', null, { faction: 'alpha', directive: 'Close and fire' });
  for (const id of ships) add(30, 'admiral_order', id, { order_text: 'engage' });
  for (const id of ships) add(30, 'captain_decision', id, { target_id: id.startsWith('alpha') ? 'beta_2' : 'alpha_1', target_name: 'x', faction: id.split('_')[0], commands: [] });
  add(30, 'command_status', 'alpha_2', { status: 'accepted', resolved_target_id: 'beta_1' });
  add(35, 'radiator_change', 'alpha_2', { extended: false });
  add(40, 'ship_dying', 'beta_2', { killer_id: 'alpha_1' });
  add(44, 'module_destroyed', 'beta_2', { module_name: 'reactor' });
  add(45, 'ship_destroyed', 'beta_2', { killer_id: 'alpha_1', cause: 'reactor_destroyed' });
  // The recorder numbers events in the order it records them: chronological
  events.sort((a, b) => a.timestamp - b.timestamp);
  events.forEach((e, i) => { e.event_id = `e${i + 1}`; e.sequence = i + 1; });
  return {
    battle_name: 'Append test', is_fleet_battle: true, winner: null, result_reason: null,
    alpha_fleet: { ships: [{ ship_id: 'alpha_1', ship_name: 'A1', ship_type: 'destroyer' }, { ship_id: 'alpha_2', ship_name: 'A2', ship_type: 'frigate' }] },
    beta_fleet: { ships: [{ ship_id: 'beta_1', ship_name: 'B1', ship_type: 'cruiser' }, { ship_id: 'beta_2', ship_name: 'B2', ship_type: 'corvette' }] },
    sim_trace: frames, events, assets: {},
  };
}

/** Slice a recording like the live server: frames by time, events by sequence. */
function streamed(rec, cuts) {
  const chunks = [];
  let lastT = -Infinity, lastSeq = 0;
  for (const cut of cuts) {
    const frames = rec.sim_trace.filter(f => f.t > lastT && f.t <= cut.t);
    const events = rec.events.filter(e => e.sequence > lastSeq && e.timestamp <= cut.t &&
      (cut.maxSeq == null || e.sequence <= cut.maxSeq));
    chunks.push({ ...rec, sim_trace: frames, events });
    if (frames.length) lastT = frames.at(-1).t;
    lastSeq = Math.max(lastSeq, ...events.map(e => e.sequence));
  }
  return chunks;
}

test('streamed appends build exactly the state of a single parse', () => {
  const rec = syntheticRecording();
  const full = new BattleLoader().parse(structuredClone(rec));
  const seqAt30 = rec.events.filter(e => e.timestamp <= 30).map(e => e.sequence);
  // The checkpoint at t=30 arrives in three polls (orders trickle in)
  const cuts = [{ t: 8 }, { t: 16 }, { t: 22 }, { t: 30, maxSeq: seqAt30[seqAt30.length - 8] },
    { t: 30, maxSeq: seqAt30[seqAt30.length - 3] }, { t: 30 }, { t: 38 }, { t: 41 }, { t: 46 }, { t: 60 }];
  const [head, ...rest] = streamed(structuredClone(rec), cuts);
  const live = new BattleLoader().parse(head);
  for (const chunk of rest) live.appendLiveData(chunk);
  assertSameLoader(live, full, 'synthetic');
  // Replaying everything again (the final full fetch / a saved replay) is a no-op
  assert.equal(live.appendLiveData(structuredClone(rec)), false);
  assertSameLoader(live, full, 'after full re-fetch');
});

test('a late, older event falls back to a full rebuild and stays correct', () => {
  const rec = syntheticRecording();
  const full = new BattleLoader().parse(structuredClone(rec));
  const late = rec.events.find(e => e.event_type === 'module_damaged');
  const head = { ...structuredClone(rec), sim_trace: rec.sim_trace.filter(f => f.t <= 40),
    events: rec.events.filter(e => e.timestamp <= 40 && e !== late) };
  const live = new BattleLoader().parse(structuredClone(head));
  live.appendLiveData({ ...structuredClone(rec), sim_trace: rec.sim_trace.filter(f => f.t > 40),
    events: rec.events.filter(e => e.timestamp > 40 || e === late) });
  assertSameLoader(live, full, 'out of order');
});

test('range queries are binary searches over the sorted events', () => {
  const loader = new BattleLoader().parse(syntheticRecording());
  const scan = (a, b) => loader.events.filter(e => e.timestamp >= a && e.timestamp <= b);
  for (const [a, b] of [[0, 0], [29.5, 30], [30, 30], [12, 26], [-5, 100], [61, 70], [22.5, 22.6]]) {
    assert.deepStrictEqual(loader.getEventsInRange(a, b), scan(a, b), `range ${a}-${b}`);
  }
  assert.deepStrictEqual(loader.getPDEventsInRange(20, 23).length, 4);
  assert.deepStrictEqual(loader.getHitEventsInRange(0, 60).length, 2);
  assert.equal(lowerBoundByTime([], 3), 0);
  // First hit event's energy, as the old linear find returned
  const impacts = loader.getVisualImpacts(-Infinity, Infinity);
  assert.equal(impacts.length, 1);
  assert.equal(impacts[0].kinetic_energy_gj, 3.5);
});

const RECORDING = fileURLToPath(new URL('../public/recordings/sonnet5_vs_dsv41flash_150pt_20260926.json', import.meta.url));

test('a real LLM battle streamed in 8 s polls matches its saved recording', { skip: !existsSync(RECORDING) && 'recording not present' }, () => {
  const rec = JSON.parse(readFileSync(RECORDING, 'utf8'));
  const full = new BattleLoader().parse(structuredClone(rec));
  const cuts = [];
  for (let t = 8; t < rec.sim_trace.at(-1).t + 8; t += 8) cuts.push({ t });
  const [head, ...rest] = streamed(rec, cuts);
  const live = new BattleLoader().parse(structuredClone(head));
  let worst = 0;
  for (const chunk of rest) {
    const copy = structuredClone({ ...chunk, assets: {} });
    const t0 = performance.now();
    live.appendLiveData(copy);
    worst = Math.max(worst, performance.now() - t0);
  }
  assertSameLoader(live, full, 'sonnet5');
  // Informational: per-poll append cost on a ~8k-event battle
  console.log(`worst append ${worst.toFixed(1)} ms over ${rest.length} polls (${live.events.length} events)`);
});
