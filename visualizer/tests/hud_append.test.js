import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { BattleLoader } from '../src/BattleLoader.js';
import { buildHudIndex, appendHudIndex } from '../src/HudIndex.js';

// LV-15 for the spectator HUD: a live poll EXTENDS loader.hud from the
// delta instead of rebuilding it, and the result must be exactly the index
// a single parse of the same events builds - whatever the poll boundaries,
// including polls that split one checkpoint's orders and reports.

/** A fleet battle that touches every HUD fold, with same-time traps for the open-timestamp logic. */
function recording() {
  const frames = [];
  for (let t = 0; t <= 120; t += 2) {
    frames.push({ t, ships: {
      alpha_1: { hull: 100 }, alpha_2: { hull: 100 },
      // beta_1: v3 kill (ship_dying event); its trace flag lands in the same second
      beta_1: { hull: t >= 70 ? 0 : 100, dying: t >= 70 && t < 84, destroyed: t >= 84 },
      // beta_2: pre-v3 style kill - only the trace knows (synthetic log row)
      beta_2: { hull: t >= 100 ? 0 : 80, dying: t >= 100 && t < 106, destroyed: t >= 106 }
    } });
  }
  const events = [];
  const add = (timestamp, event_type, ship_id = null, data = {}, event_id = null) =>
    events.push({ event_id: event_id || `e${events.length + 1}`, sequence: events.length + 1, timestamp, event_type, ship_id, data });
  const compliance = (launched, final, deviations) => ({ ordered_maneuver: 'INTERCEPT', executed_maneuver: 'INTERCEPT',
    ordered_target: 'OCS Falchion-1', actual_target: 'OCS Falchion-1', ordered_salvo: 2, launched, final, deviations });
  add(0, 'battle_start');
  add(0, 'decision_point_reached');
  for (const cp of [1, 2, 3]) {
    const t = cp * 30;
    // Traffic recorded BEFORE the checkpoint's first decision point event:
    // its checkpoint number is only known once that event arrives
    add(t, 'comms_message', 'alpha_2', { msg_id: `m${cp}`, channel: 'fleet_net', kind: 'contact', sender_id: 'alpha_2', sender_role: 'captain',
      sender_name: 'TIS Dart-2', faction: 'alpha', recipient_spec: 'all_friendlies', recipient_ids: ['alpha_1'], text: `Contact report ${cp}.`,
      structured: {}, checkpoint: null, sent_t: t, deliver_at_t: t + 4 });
    add(t, 'captain_log', 'alpha_1', { note: `Log ${cp}.` });
    add(t, 'decision_point_reached');
    add(t, 'model_call', null, { actor: 'admiral', faction: 'alpha', usage: { cost: 0.01 * cp }, message_refs: [`img${cp}`] });
    add(t, 'admiral_plan', null, { faction: 'alpha', admiral_name: 'Admiral A', plan: `Plan ${cp}.` });
    add(t, 'admiral_directive', null, { faction: 'alpha', admiral_name: 'Admiral A', directive: `Directive ${cp}.` });
    add(t, 'admiral_order', 'alpha_1', { faction: 'alpha', order_text: 'Launch 2 at Falchion-1.', torpedo_salvo: 2, torpedo_target: 'beta_1' });
    // Reports to the admiral from the previous checkpoint, with the final compliance
    if (cp > 1) {
      add(t, 'comms_delivered', 'alpha_admiral', { msg_id: `r${cp - 1}-alpha_1`, recipient_id: 'alpha_admiral', delivered_t: t, checkpoint: cp,
        compliance: compliance(cp === 2 ? 1 : 2, true, cp === 2 ? ['salvo: ordered 2, launched 1'] : []) });
    }
    // alpha_1: legacy ack first, then the structured report that supersedes it
    add(t, 'captain_decision', 'alpha_1', { acknowledgment: 'ACK: closing', maneuver_type: 'INTERCEPT', commands: [] });
    add(t, 'captain_report', 'alpha_1', { report_id: `r${cp}-alpha_1`, faction: 'alpha', checkpoint: cp, captain_name: 'Captain One', ship_name: 'TIS Dart-1',
      status: 'ACK', reason: '', intent: 'Closing.', contact: null, request: null, compliance: compliance(2, false, []), implicit: false });
    // beta_1: report BEFORE its decision (the WHY link resolves later)
    add(t, 'captain_report', 'beta_1', { faction: 'beta', checkpoint: cp, ship_name: 'OCS Falchion-1', status: cp === 2 ? 'DEVIATE' : 'ACK', reason: 'Holding.' });
    add(t, 'captain_decision', 'beta_1', { acknowledgment: 'DEVIATE: holding', commands: [] });
    add(t, 'captain_decision', 'beta_2', { tool_calls: [{ name: 'respond_to_orders', arguments: { response_type: 'ACKNOWLEDGE', acknowledgment_note: 'Aye.' } }] });
    add(t + 4, 'comms_delivered', 'alpha_1', { msg_id: `m${cp}`, recipient_id: 'alpha_1', delivered_t: t + 4, checkpoint: cp });
  }
  for (const t of [40, 41, 50]) add(t, 'torpedo_launched', 'alpha_1', { target_id: 'beta_1', torpedo_id: `torp_alpha_1_${t}` });
  for (const t of [44, 61, 66, 69]) add(t, 'hit', 'beta_1', { shooter_id: 'alpha_2', projectile_id: `proj_alpha_2_${t}`, kinetic_energy_gj: 2 });
  add(62, 'torpedo_impact', 'alpha_1', { torpedo_id: 'torp_alpha_1_40', target_id: 'beta_1', total_damage_gj: 20 });
  add(70, 'ship_dying', 'beta_1', { killer_id: 'alpha_1', detonation_in_s: 14, ordnance_id: 'torp_alpha_1_41' });
  add(84, 'ship_destroyed', 'beta_1', { killer_id: 'alpha_1', cause: 'reactor_destroyed' });
  add(97, 'hit', 'beta_2', { shooter_id: 'alpha_1', projectile_id: 'proj_alpha_1_97' });
  // (Repeated event ids are the loader's business: a re-sent id is the same event)
  add(110, 'admiral_directive', null, { faction: 'beta', directive: 'Off-cycle directive.' });
  add(120, 'battle_end', null, { winner: 'alpha', reason: 'Beta fleet eliminated' });
  events.sort((a, b) => a.timestamp - b.timestamp || a.sequence - b.sequence);
  events.forEach((e, i) => { e.sequence = i + 1; });
  const assets = {};
  for (const cp of [1, 2, 3]) assets[`img${cp}`] = { content: [{ type: 'text', text: 'plot' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }] };
  return {
    battle_name: 'HUD append', is_fleet_battle: true, winner: null, result_reason: null, decision_interval_s: 30,
    alpha_model: 'anthropic/claude-sonnet-5', beta_model: 'deepseek/deepseek-v4.1-flash',
    provenance: { draft: { beta: { ships: [{ ship_id: 'beta_1', cost: 31 }] } } },
    alpha_fleet: { ships: [{ ship_id: 'alpha_1', ship_name: 'TIS Dart-1', ship_type: 'corvette' }, { ship_id: 'alpha_2', ship_name: 'TIS Dart-2', ship_type: 'corvette' }] },
    beta_fleet: { ships: [{ ship_id: 'beta_1', ship_name: 'OCS Falchion-1', ship_type: 'destroyer' }, { ship_id: 'beta_2', ship_name: 'OCS Lancer-1', ship_type: 'battlecruiser' }] },
    sim_trace: frames, events, assets
  };
}

/**
 * Cut a recording into live polls at event indices: each poll brings the
 * events up to the cut and every frame up to the newest event time (frames
 * arrive with the first poll that reaches their time, as on the server).
 */
function polls(rec, cuts) {
  const out = [];
  let prev = 0, lastT = -Infinity;
  for (const cut of [...cuts, rec.events.length]) {
    const events = rec.events.slice(prev, cut);
    const until = cut >= rec.events.length ? Infinity : rec.events[cut - 1]?.timestamp ?? lastT;
    const sim_trace = rec.sim_trace.filter(f => f.t > lastT && f.t <= until);
    if (sim_trace.length) lastT = sim_trace.at(-1).t;
    out.push({ ...rec, sim_trace, events, assets: {} });
    prev = cut;
  }
  return out;
}

const prefix = (rec, chunks, n) => ({ ...rec, sim_trace: chunks.slice(0, n).flatMap(c => c.sim_trace), events: chunks.slice(0, n).flatMap(c => c.events) });

/** Stream `rec` in `cuts`, checking the hud after EVERY poll against a fresh parse of the same data. */
function streamAndCompare(rec, cuts, label, everyPoll = true) {
  const chunks = polls(structuredClone(rec), cuts);
  const live = new BattleLoader().parse(structuredClone({ ...prefix(rec, chunks, 1), assets: rec.assets }));
  let appended = 0;
  for (let i = 1; i < chunks.length; i++) {
    const before = live.hud;
    if (!live.appendLiveData(structuredClone(chunks[i]))) continue;
    appended++;
    assert.notEqual(live.hud, before, `${label}: a new hud object per append`);
    if (everyPoll || i === chunks.length - 1) {
      const full = new BattleLoader().parse(structuredClone({ ...prefix(rec, chunks, i + 1), assets: rec.assets }));
      assert.deepStrictEqual(live.hud, full.hud, `${label}: hud after poll ${i} differs from a full build`);
    }
  }
  return { live, appended };
}

test('every way of splitting the stream builds the same hud as one parse', () => {
  const rec = recording();
  const n = rec.events.length;
  // One event per poll: every same-time split there is
  streamAndCompare(rec, Array.from({ length: n - 1 }, (_, i) => i + 1), 'one event per poll');
  // Seeded random poll sizes
  let seed = 7;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let round = 0; round < 12; round++) {
    const cuts = [];
    for (let at = 1 + Math.floor(rand() * 6); at < n; at += 1 + Math.floor(rand() * 9)) cuts.push(at);
    streamAndCompare(rec, cuts, `random polls #${round}`);
  }
});

test('appends extend the index in place instead of rebuilding it', () => {
  const rec = recording();
  const cut = rec.events.findIndex(e => e.timestamp > 60);
  const [head, tail] = polls(structuredClone(rec), [cut]);
  const live = new BattleLoader().parse(structuredClone({ ...head, assets: rec.assets }));
  const before = live.hud;
  live.appendLiveData(structuredClone(tail));
  // Same (grown) arrays: the fold continued; a rebuild would have made new ones
  for (const key of ['ids', 'comms', 'deliveries', 'spend']) assert.equal(live.hud[key], before[key], `${key} extended in place`);
  // The incremental state lives off the loader and off the hud (plain, comparable data)
  assert.ok(!Object.keys(live).some(k => /builder/i.test(k)));
  assert.deepEqual(Object.keys(live.hud).sort(), ['checkpoints', 'comms', 'costs', 'deliveries', 'duration', 'hitBins', 'ids', 'kills', 'log',
    'models', 'result', 'salvos', 'spend', 'visions'].sort());
  // The fixture's traps all resolved the way a full parse resolves them
  const hud = live.hud;
  assert.deepEqual(hud.checkpoints.map(c => [c.t, c.n]), [[30, 1], [60, 2], [90, 3], [110, 4]]);
  const contact = hud.comms.filter(r => r.kind === 'net').map(r => r.cp);
  assert.deepEqual(contact, [1, 2, 3], 'traffic before the decision point still files under its checkpoint');
  assert.equal(hud.comms.filter(r => r.from === 'TIS Dart-1' && r.kind === 'report').length, 3, 'the report replaced the legacy ack');
  assert.ok(hud.comms.filter(r => r.from === 'OCS Falchion-1' && r.kind === 'report').every(r => r.decisionId), 'report before its decision still links it');
  assert.deepEqual(hud.kills.map(k => [k.shipId, k.t, k.killerId, k.inferred]), [['beta_1', 70, 'alpha_1', false], ['beta_2', 100, 'alpha_1', true]]);
  assert.ok(hud.log.some(r => r.id === 'kill:beta_2') && !hud.log.some(r => r.id === 'kill:beta_1'));
  assert.equal(hud.visions.length, 3);
  assert.deepEqual(hud.result, { winner: 'alpha', reason: 'Beta fleet eliminated', stopped: false, known: true });
});

test('a frame-only poll and a ship_dying event trailing its trace flag stay exact', () => {
  const rec = recording();
  const dying = rec.events.findIndex(e => e.event_type === 'ship_dying');
  // The poll before the event already carries the frame flagged dying: a
  // transient synthetic kill row that the event then replaces
  const chunks = polls(structuredClone(rec), [dying]);
  chunks[0].sim_trace.push(...rec.sim_trace.filter(f => f.t > chunks[0].sim_trace.at(-1).t && f.t <= 72));
  chunks[1].sim_trace = chunks[1].sim_trace.filter(f => f.t > 72);
  const live = new BattleLoader().parse(structuredClone({ ...chunks[0], assets: rec.assets }));
  assert.ok(live.hud.log.some(r => r.id === 'kill:beta_1'), 'trace-only kill shows as a synthetic row meanwhile');
  live.appendLiveData({ sim_trace: [], events: [] });   // nothing new: no rebuild
  live.appendLiveData(structuredClone(chunks[1]));
  const full = new BattleLoader().parse(structuredClone(rec));
  assert.deepStrictEqual(live.hud, full.hud);
  assert.ok(!live.hud.log.some(r => r.id === 'kill:beta_1'));
});

test('a late event older than the index falls back to a full build', () => {
  const rec = recording();
  const late = rec.events.find(e => e.event_type === 'captain_log' && e.timestamp === 60);
  const head = { ...structuredClone(rec), sim_trace: rec.sim_trace.filter(f => f.t <= 80), events: rec.events.filter(e => e.timestamp <= 80 && e !== late) };
  const live = new BattleLoader().parse(structuredClone(head));
  const before = live.hud;
  live.appendLiveData({ ...structuredClone(rec), assets: {}, sim_trace: rec.sim_trace.filter(f => f.t > 80), events: rec.events.filter(e => e.timestamp > 80 || e === late) });
  assert.notEqual(live.hud.comms, before.comms, 'rebuilt, not extended');
  assert.deepStrictEqual(live.hud, new BattleLoader().parse(structuredClone(rec)).hud);
});

test('appendHudIndex without a matching builder (or delta) builds from scratch', () => {
  const rec = recording();
  const loader = new BattleLoader().parse(structuredClone(rec));
  const full = loader.hud;
  // A foreign loader-shaped object has no builder: full build
  const other = { events: loader.events, ships: loader.ships, simTrace: loader.simTrace, sourceData: loader.sourceData, metadata: loader.metadata, duration: loader.duration };
  assert.deepStrictEqual(appendHudIndex(other, { events: [], frames: [] }), full);
  assert.deepStrictEqual(appendHudIndex(loader, null), full);
  // A delta that does not end the event list (the loader was re-sorted) is caught
  assert.deepStrictEqual(appendHudIndex(loader, { events: loader.events.slice(0, 3), frames: [] }), buildHudIndex(loader));
});

const RECORDINGS = ['p4_hud_comms_fixture.json', 'p4_comms_mock.json', 'sonnet5_vs_dsv41flash_150pt_20260926.json']
  .map(name => fileURLToPath(new URL(`../public/recordings/${name}`, import.meta.url)));

for (const path of RECORDINGS) {
  const name = path.split('/').pop();
  test(`${name} streamed in live-like polls matches its parse`, { skip: !existsSync(path) && 'recording not present' }, () => {
    const rec = JSON.parse(readFileSync(path, 'utf8'));
    rec.events.sort((a, b) => a.timestamp - b.timestamp || (a.sequence || 0) - (b.sequence || 0));
    // 8 s polls, and every checkpoint's events split in three by sequence
    const cuts = new Set();
    for (let t = 8; t < rec.events.at(-1).timestamp; t += 8) cuts.add(rec.events.findIndex(e => e.timestamp > t));
    const byT = new Map();
    rec.events.forEach((e, i) => { if (!byT.has(e.timestamp)) byT.set(e.timestamp, []); byT.get(e.timestamp).push(i); });
    for (const list of byT.values()) {
      if (list.length < 6) continue;
      cuts.add(list[Math.floor(list.length / 3)]);
      cuts.add(list[Math.floor((2 * list.length) / 3)]);
    }
    const sorted = [...cuts].filter(c => c > 0).sort((a, b) => a - b);
    const t0 = performance.now();
    const { live, appended } = streamAndCompare(rec, sorted, name, false);
    assert.ok(appended > 10);
    // Informational: the HUD share of a poll (the whole append, incl. other timelines, is more)
    const loader = live;
    const s = performance.now();
    buildHudIndex({ ...loader, ships: loader.ships });
    console.log(`${name}: ${appended} polls in ${(performance.now() - t0).toFixed(0)} ms incl. checks; one full HUD build ${(performance.now() - s).toFixed(1)} ms`);
  });
}
