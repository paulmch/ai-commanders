import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildCheckpoints, checkpointAt, buildComms, buildDeliveries, deliveryAt, buildKills, killStateAt,
  describeKill, buildLog, buildSalvos, buildSpend, spendAt, buildResult, buildHudIndex, commsMatches,
  normalizeStatus, lowerBound, upperBound, shortModel, formatLogEvent, complianceAt, complianceChips
} from '../src/HudIndex.js';
import { CombatLog } from '../src/CombatLog.js';
import { BattleLoader } from '../src/BattleLoader.js';
import { FakeDocument, allElements } from './fakedom.js';

let seq = 0;
const ev = (t, type, ship_id = null, data = {}, id = null) => {
  seq++;
  return { event_id: id || `e${seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data };
};
const ships = {
  alpha_1: { faction: 'alpha', name: 'TIS Dart-1', type: 'corvette' },
  alpha_2: { faction: 'alpha', name: 'TIS Dart-2', type: 'corvette' },
  beta_1: { faction: 'beta', name: 'OCS Falchion-1', type: 'destroyer' },
  beta_2: { faction: 'beta', name: 'OCS Lancer-1', type: 'battlecruiser' }
};
const ids = events => events.map(e => e.event_id);
const HOSTILE = '<img src=x onerror="alert(1)"><b>BOLD</b>';

test('binary search bounds count rows at or before the playhead', () => {
  const rows = [{ t: 1 }, { t: 2 }, { t: 2 }, { t: 5 }];
  assert.equal(upperBound(rows, 2), 3);
  assert.equal(lowerBound(rows, 2), 1);
  assert.equal(upperBound(rows, 0), 0);
  assert.equal(upperBound(rows, 9), 4);
  assert.equal(shortModel('anthropic/claude-sonnet-5'), 'claude-sonnet-5');
});

test('checkpoints skip the t=0 decision point and honour numbered checkpoint events', () => {
  const cps = buildCheckpoints([ev(0, 'decision_point_reached'), ev(30, 'decision_point_reached'),
    ev(30, 'captain_decision', 'alpha_1'), ev(60, 'admiral_directive'), ev(90, 'captain_decision', 'alpha_1')]);
  assert.deepEqual(cps, [{ t: 30, n: 1 }, { t: 60, n: 2 }, { t: 90, n: 3 }]);
  assert.equal(checkpointAt(cps, 29.9), 0);
  assert.equal(checkpointAt(cps, 60), 2);
  assert.equal(checkpointAt(cps, 75), 2);
  const numbered = buildCheckpoints([ev(30, 'checkpoint', null, { checkpoint: 4 }), ev(60, 'checkpoint', null, { checkpoint: 5 })]);
  assert.deepEqual(numbered.map(c => c.n), [4, 5]);
});

test('legacy command traffic becomes command-net rows with acknowledgments as reports', () => {
  const events = [
    ev(30, 'admiral_plan', null, { faction: 'alpha', admiral_name: 'Admiral S', plan: 'Phase 1: close.' }),
    ev(30, 'admiral_directive', null, { faction: 'alpha', admiral_name: 'Admiral S', directive: 'All Darts: intercept.' }),
    ev(30, 'admiral_order', 'alpha_1', { admiral_name: 'Admiral S', order_text: 'Launch 4 at Falchion-1.', priority: 'critical', torpedo_salvo: 4, torpedo_target: 'beta_1' }),
    ev(30, 'captain_decision', 'alpha_1', { tool_calls: [{ name: 'respond_to_orders', arguments: { response_type: 'DEVIATE', deviation_reason: 'Target is separating.' } }] }),
    ev(30, 'captain_decision', 'alpha_2', { acknowledgment: 'Closing as ordered.' }),
    ev(30, 'captain_log', 'alpha_2', { note: 'Holding fire.' }),
    ev(45, 'captain_admiral_discussion', 'alpha_1', { captain_question: 'Which target?', admiral_response: 'Falchion-1.' }),
    ev(50, 'message', 'beta_1', { sender_name: 'Captain B', message: 'Nice try.' }),
    ev(60, 'captain_decision', 'beta_1', { call_failed: true })
  ];
  const cps = buildCheckpoints(events);
  const rows = buildComms(events, ids(events), ships, cps);
  const tags = rows.map(r => r.tag);
  assert.deepEqual(tags, ['PLAN', 'DIRECTIVE', 'ORDER', 'DEVIATE', 'ACK', 'LOG', 'QUERY', 'TRANSMISSION', 'NO ORDERS']);
  const order = rows[2];
  assert.equal(order.to, 'TIS Dart-1');
  assert.equal(order.priority, 'CRITICAL');
  assert.deepEqual(order.chips, ['SALVO 4 → OCS Falchion-1']);
  assert.equal(rows[3].text, 'Target is separating.');
  assert.equal(rows[3].decisionId, events[3].event_id);
  assert.equal(rows[4].status, 'ACK');
  assert.match(rows[6].reply, /Falchion-1/);
  assert.equal(rows[7].open, true);
  // Open-channel traffic shows under both factions' filters; orders only under their own
  assert.ok(commsMatches(rows[7], 'alpha') && commsMatches(rows[7], 'beta'));
  assert.ok(commsMatches(order, 'alpha') && !commsMatches(order, 'beta'));
  assert.ok(commsMatches(rows[3], 'all', 'issues') && !commsMatches(rows[4], 'all', 'issues'));
  assert.ok(commsMatches(order, 'all', 'command') && !commsMatches(rows[5], 'all', 'command'));
  assert.equal(normalizeStatus('ACKNOWLEDGE'), 'ACK');
  assert.equal(normalizeStatus('unable'), 'UNABLE');
});

test('structured comms: captain_report supersedes the legacy ack, channels map, delivery state is time-aware', () => {
  const events = [
    ev(30, 'decision_point_reached'),
    ev(30, 'comms_message', 'alpha_1', { msg_id: 'm1', channel: 'fleet_net', kind: 'contact', sender_id: 'alpha_1', sender_role: 'captain',
      sender_name: 'TIS Dart-1', faction: 'alpha', recipient_spec: 'all_friendlies', recipient_ids: ['alpha_2'], text: 'Falchion-1 turning.',
      structured: {}, checkpoint: 1, sent_t: 30, deliver_at_t: 34 }),
    ev(30, 'comms_message', 'beta_1', { msg_id: 'm2', channel: 'enemy_broadcast', sender_id: 'beta_1', sender_role: 'captain',
      sender_name: 'OCS Falchion-1', faction: 'beta', recipient_spec: 'enemy', recipient_ids: ['alpha_1', 'alpha_2'], text: HOSTILE,
      checkpoint: 1, sent_t: 30, deliver_at_t: 30 }),
    ev(30, 'comms_message', null, { msg_id: 'm3', channel: 'fleet_net', sender_role: 'engine', faction: 'alpha', recipient_ids: ['alpha_1'],
      text: 'Dart-2 launched 4 at Falchion-1, ETA 40 s.', checkpoint: 1, sent_t: 30, deliver_at_t: 30 }),
    ev(30, 'captain_decision', 'alpha_1', { acknowledgment: 'ACK: closing', tool_calls: [{ name: 'report_to_admiral', arguments: { status: 'ACK' } }] }),
    ev(30, 'captain_report', 'alpha_1', { report_id: 'r1', faction: 'alpha', checkpoint: 1, captain_name: 'Captain Dart-1', ship_name: 'TIS Dart-1',
      status: 'DEVIATE', reason: 'Lancer-1 is closer.', intent: 'Engaging Lancer-1.', contact: { ship: 'OCS Lancer-1', assessment: 'closing', note: '' },
      request: null, compliance: { ordered_maneuver: 'INTERCEPT', executed_maneuver: 'EVADE', ordered_target: 'beta_1', actual_target: 'beta_2', ordered_salvo: 4, launched: 2 } }),
    ev(31, 'comms_delivered', 'alpha_1', { msg_id: 'm2', recipient_id: 'alpha_1', delivered_t: 30, checkpoint: 1 }),
    ev(34, 'comms_delivered', 'alpha_2', { msg_id: 'm1', recipient_id: 'alpha_2', delivered_t: 34, checkpoint: 1 }),
    ev(34, 'comms_delivered', 'alpha_2', { msg_id: 'm1', recipient_id: 'alpha_2', delivered_t: 34, checkpoint: 1 }, 'dup-delivery')
  ];
  const cps = buildCheckpoints(events);
  const rows = buildComms(events, ids(events), ships, cps);
  assert.deepEqual(rows.map(r => r.tag), ['FLEET NET', 'OPEN CHANNEL', 'AUTO', 'DEVIATE']);
  const [net, enemy, auto, report] = rows;
  assert.equal(net.to, 'TIS Dart-2');
  assert.deepEqual(net.chips, ['CONTACT']);
  assert.equal(enemy.open, true);
  assert.equal(enemy.text, HOSTILE);   // kept verbatim: escaping is the renderer's job (textContent)
  assert.equal(auto.kind, 'engine');
  assert.equal(report.text, 'Lancer-1 is closer.');
  assert.match(report.reply, /Intent: Engaging Lancer-1/);
  assert.deepEqual(report.chips, ['CONTACT OCS Lancer-1 · closing']);
  // No engine verdict recorded (schema-only block): the fields are compared here
  assert.deepEqual(report.compliance.chips, ['ORDERED INTERCEPT · FLEW EVADE', 'TARGET OCS Falchion-1 → OCS Lancer-1', 'SALVO 2/4']);
  assert.equal(report.compliance.deviated, true);
  const deliveries = buildDeliveries(events);
  assert.equal(deliveries.get('m1').length, 1, 'duplicate delivery events collapse');
  assert.equal(deliveryAt(net, deliveries, 31).state, 'transit');
  assert.equal(deliveryAt(net, deliveries, 34).state, 'delivered');
  assert.equal(deliveryAt(enemy, deliveries, 31).state, 'partial');
  assert.equal(deliveryAt(auto, deliveries, 31, false).state, 'transit');
  assert.equal(deliveryAt(auto, deliveries, 31, true).state, 'undelivered');
  // The report is routed to the admiral, who reads it at its next turn
  assert.deepEqual([report.msgId, report.toIds, report.deliverAt], ['r1', ['alpha_admiral'], 60]);
  assert.deepEqual(deliveryAt(report, deliveries, 31), { state: 'transit', done: 0, total: 1, at: 60 });
});

test('captain reports travel to the admiral: delivery, and the compliance it read (final) replaces the filed one', () => {
  const filed = { ordered_maneuver: 'INTERCEPT', executed_maneuver: 'INTERCEPT', ordered_target: 'the cruiser', actual_target: 'OCS Lancer-1',
    ordered_salvo: 4, launched: 4, final: false, followed: true, deviations: [] };
  const events = [
    ev(30, 'decision_point_reached'),
    ev(30, 'captain_decision', 'alpha_1', { acknowledgment: 'ACK', commands: [] }, 'dec-a1'),
    ev(30, 'captain_report', 'alpha_1', { report_id: 'r1-alpha_1', faction: 'alpha', checkpoint: 1, ship_name: 'TIS Dart-1', status: 'ACK',
      reason: '', intent: '', contact: null, request: null, compliance: filed, implicit: true, decision_event_id: 'dec-a1' }),
    ev(30, 'captain_report', 'alpha_2', { report_id: 'r1-alpha_2', faction: 'alpha', checkpoint: 1, ship_name: 'TIS Dart-2', status: 'ACK',
      reason: 'On it.', compliance: { ...filed, launched: 2, deviations: ['salvo: ordered 4, committed 2'] } }),
    ev(60, 'decision_point_reached'),
    ev(60, 'comms_delivered', 'alpha_admiral', { msg_id: 'r1-alpha_1', recipient_id: 'alpha_admiral', delivered_t: 60, checkpoint: 2,
      compliance: { ...filed, launched: 1, final: true, followed: false, deviations: ['salvo: ordered 4, launched 1'] } }),
    // The last checkpoint's report: the battle ends before the admiral's next turn
    ev(90, 'captain_report', 'alpha_1', { report_id: 'r3-alpha_1', faction: 'alpha', checkpoint: 3, status: 'DEVIATE', reason: 'Breaking off.' })
  ];
  const cps = buildCheckpoints(events);
  const rows = buildComms(events, ids(events), ships, cps, { interval: 30 });
  const [implicit, partial, last] = rows;
  assert.equal(rows.length, 3, 'the reports replace the legacy ack');
  assert.deepEqual([implicit.msgId, implicit.toIds, implicit.deliverAt, implicit.decisionId], ['r1-alpha_1', ['alpha_admiral'], 60, 'dec-a1']);
  assert.match(implicit.text, /implicit ACK/);
  // The engine's verdict wins over a field comparison: an unresolved raw
  // ordered target ('the cruiser') is not a deviation the engine counted
  assert.deepEqual(implicit.compliance, { chips: [], deviated: false, final: false });
  assert.deepEqual(complianceChips({ ordered_target: 'the cruiser', actual_target: 'OCS Lancer-1' }), ['TARGET the cruiser → OCS Lancer-1'], 'fallback without a verdict');
  assert.deepEqual(partial.compliance.chips, ['SALVO: ordered 4, committed 2']);
  const deliveries = buildDeliveries(events);
  assert.equal(deliveryAt(implicit, deliveries, 45).state, 'transit');
  assert.equal(deliveryAt(implicit, deliveries, 45).at, 60);
  assert.deepEqual(deliveryAt(implicit, deliveries, 60), { state: 'delivered', done: 1, total: 1, at: 60 });
  // Filed (rounds committed) until the admiral's copy arrives, then that copy (rounds launched)
  assert.deepEqual(complianceAt(implicit, deliveries, 59), { chips: [], deviated: false, final: false, delivered: false, at: 30 });
  assert.deepEqual(complianceAt(implicit, deliveries, 60), { chips: ['SALVO: ordered 4, launched 1'], deviated: true, final: true, delivered: true, at: 60 });
  // ISSUES takes the acknowledge-and-drift report once the admiral's copy shows it
  assert.ok(!commsMatches(implicit, 'all', 'issues', { deliveries, t: 59 }));
  assert.ok(commsMatches(implicit, 'all', 'issues', { deliveries, t: 60 }));
  assert.ok(commsMatches(partial, 'all', 'issues'), 'filed deviations count without a delivery');
  // Never delivered: the alpha_2 report once its due time passes; the last
  // report once the recording ends (due after the battle)
  assert.equal(deliveryAt(partial, deliveries, 61, true).state, 'undelivered');
  assert.equal(deliveryAt(partial, deliveries, 61, false).state, 'transit', 'live: it may still come');
  assert.equal(deliveryAt(last, deliveries, 100, true, 112).state, 'transit');
  assert.equal(deliveryAt(last, deliveries, 112, true, 112).state, 'undelivered');
});

test('duplicate event ids index once', () => {
  const e = ev(30, 'admiral_directive', null, { faction: 'beta', directive: 'Hold.' }, 'same');
  const events = [e, { ...e }];
  assert.equal(buildComms(events, ids(events), ships, []).length, 1);
});

test('kills: dying and destroyed collapse to one record; the hulk counts down to its recorded detonation', () => {
  const events = [
    ev(150, 'ship_dying', 'beta_1', { killer_id: 'alpha_1', detonation_in_s: 25.8, ordnance_id: 'torp_alpha_1_0005' }),
    ev(170, 'ship_destroyed', 'beta_1', { killer_id: 'alpha_1', cause: 'reactor_destroyed', ordnance_id: 'torp_alpha_2_0006' }),
    ev(200, 'ship_dying', 'alpha_2', { killer_id: 'beta_2', detonation_in_s: 20, ordnance_id: 'proj_beta_2_0001' }),
    ev(262, 'reactor_meltdown', 'beta_2', { heat: 1.5 }),
    ev(262, 'ship_destroyed', 'beta_2', { cause: 'reactor_overheat' })
  ];
  const kills = buildKills(events, ids(events), [], ships);
  assert.equal(kills.length, 3);
  const [k1, k2, k3] = kills;
  assert.equal(k1.t, 150);
  assert.equal(k1.killerName, 'TIS Dart-1');
  assert.equal(k1.weapon, 'torpedo');
  assert.equal(killStateAt(k1, 149), null);
  assert.deepEqual(killStateAt(k1, 160), { state: 'dying', in: 10 });
  assert.equal(killStateAt(k1, 170).state, 'destroyed');
  assert.equal(k2.weapon, 'coilgun');
  assert.equal(killStateAt(k2, 210).in, 10, 'live: no destruction yet -> predicted fuse');
  assert.equal(k3.weapon, 'meltdown');
  assert.equal(k3.killerId, null);
  assert.match(describeKill(k3), /reactor meltdown/);
});

test('pre-v3 recordings: kills come from sim_trace with the killer inferred from the last hit', () => {
  const events = [
    ev(100, 'torpedo_impact', 'alpha_1', { torpedo_id: 'torp_alpha_1_0001', target_id: 'beta_1', total_damage_gj: 24 }),
    ev(103, 'hit', 'beta_1', { shooter_id: 'alpha_2', projectile_id: 'proj_alpha_2_0003' })
  ];
  const trace = [
    { t: 100, ships: { beta_1: { destroyed: false }, alpha_1: {} } },
    { t: 104, ships: { beta_1: { destroyed: false, dying: true } } },
    { t: 110, ships: { beta_1: { destroyed: true, dying: false } } }
  ];
  const kills = buildKills(events, ids(events), trace, ships);
  assert.equal(kills.length, 1);
  assert.equal(kills[0].t, 104);
  assert.equal(kills[0].destroyedT, 110);
  assert.equal(kills[0].killerId, 'alpha_2');
  assert.equal(kills[0].inferred, true);
  assert.match(describeKill(kills[0]), /probable/);
  const log = buildLog(events, ids(events), ships, kills);
  assert.ok(log.some(r => r.id === 'kill:beta_1' && r.cat === 'kills'), 'synthetic kill row for legacy deaths');
});

test('salvos group launches per faction; spend accumulates; a budget stop is not a draw', () => {
  const events = [
    ev(90, 'torpedo_launched', 'alpha_1', { target_id: 'beta_1' }), ev(91, 'torpedo_launched', 'alpha_2', { target_id: 'beta_1' }),
    ev(92, 'torpedo_launched', 'beta_1', { target_id: 'alpha_1' }), ev(97, 'torpedo_launched', 'alpha_1', { target_id: 'beta_2' }),
    ev(30, 'model_call', null, { faction: 'alpha', usage: { cost: 0.5 } }), ev(60, 'model_call', null, { faction: 'alpha', usage: { cost: 0.25 } })
  ];
  const salvos = buildSalvos(events, ships);
  assert.deepEqual(salvos.map(s => [s.t, s.faction, s.count]), [[90, 'alpha', 2], [92, 'beta', 1], [97, 'alpha', 1]]);
  assert.deepEqual(salvos[0].targets, [['OCS Falchion-1', 2]]);
  const spend = buildSpend(events);
  assert.equal(spendAt(spend, 'alpha', 45), 0.5);
  assert.equal(spendAt(spend, 'alpha', 60), 0.75);
  assert.equal(spendAt(spend, 'beta', 60), 0);
  const stopped = buildResult([ev(360, 'battle_stopped', null, { reason: 'Spending ceiling reached' })], { winner: null, resultReason: 'RuntimeError: Spending ceiling reached' });
  assert.equal(stopped.stopped, true);
  assert.equal(buildResult([], { winner: 'alpha', resultReason: 'Beta fleet eliminated' }).stopped, false);
});

test('log formatter covers kills, command traffic and system rows as plain text', () => {
  const f = e => formatLogEvent(e, ships);
  assert.match(f(ev(150, 'ship_dying', 'beta_1', { killer_id: 'alpha_1', detonation_in_s: 25.8, ordnance_id: 'torp_x' })).text, /CRIPPLED: OCS Falchion-1 by TIS Dart-1 \(torpedo\) · reactor in ~26s/);
  assert.match(f(ev(150, 'ship_destroyed', 'beta_1', { cause: 'reactor_destroyed', killer_id: 'alpha_1' })).text, /KILL: OCS Falchion-1 destroyed \(reactor destroyed\)/);
  assert.equal(f(ev(360, 'battle_stopped', null, { reason: 'ceiling' })).cat, 'kills');
  assert.match(f(ev(30, 'admiral_directive', null, { faction: 'alpha', directive: 'All corvettes close to 300 km in formation. Hold fire until ordered.' })).text, /^ORDERS ALPHA: All corvettes close to 300 km in formation\.$/);
  assert.equal(f(ev(1, 'pd_fired', 'alpha_1')), null);
});

function logFixture() {
  const doc = new FakeDocument();
  const list = doc.createElement('div');
  const events = [
    ev(10, 'shot_fired', 'alpha_1', { target_id: 'beta_1' }),
    ev(20, 'hit', 'beta_1', { shooter_id: 'alpha_1', kinetic_energy_gj: 3 }),
    ev(30, 'message', 'beta_1', { sender_name: 'Captain B', message: HOSTILE }),
    ev(30, 'ship_dying', 'beta_1', { killer_id: 'alpha_1', detonation_in_s: 10 }),
    ev(40, 'torpedo_launched', 'alpha_1', { target_id: 'beta_1' })
  ];
  const loader = new BattleLoader();
  loader.parse({ events, sim_trace: [{ t: 0, ships: {} }, { t: 40, ships: {} }], alpha_fleet: { ships: [] } });
  loader.ships = ships;
  loader.rebuildDerivedState();
  const log = new CombatLog({ list, doc });
  log.setIndex(loader.hud);
  return { doc, list, log, loader };
}

test('combat log is idempotent while parked: no duplicate rows, no DOM churn', () => {
  const { list, log } = logFixture();
  assert.equal(log.update(30), 'rebuild');
  assert.equal(list.children.length, 4);
  for (let i = 0; i < 100; i++) assert.equal(log.update(30), 'same');
  assert.equal(list.children.length, 4);
  const ids = list.children.map(c => c.dataset.eventId);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(list.firstChild.textContent, /^00:30CRIPPLED/, 'newest first');
});

test('combat log moves forward incrementally and prunes rows newer than the playhead on a backward seek', () => {
  const { list, log } = logFixture();
  log.update(10);
  assert.equal(list.children.length, 1);
  const first = list.lastChild;
  assert.equal(log.update(20), 'append');
  assert.equal(list.lastChild, first, 'existing rows are kept, not rebuilt');
  assert.equal(list.children.length, 2);
  log.update(40);
  assert.equal(list.children.length, 5);
  assert.equal(log.update(15), 'rebuild');
  assert.deepEqual(list.children.map(c => c.querySelector('.event-time').textContent), ['00:10']);
  log.setFilter('kills');
  log.update(40);
  assert.deepEqual(list.children.map(c => c.querySelector('.event-text').textContent.slice(0, 8)), ['CRIPPLED']);
});

test('late events at the parked time appear exactly once (live checkpoint pause)', () => {
  const { list, log, loader } = logFixture();
  log.update(40);
  assert.equal(list.children.length, 5);
  loader.appendLiveData({ events: [ev(40, 'admiral_directive', null, { faction: 'beta', directive: 'Regroup on the flagship.' })] });
  log.setIndex(loader.hud);
  assert.equal(log.update(40), 'append');
  for (let i = 0; i < 10; i++) log.update(40);
  assert.equal(list.children.length, 6);
  assert.match(list.firstChild.textContent, /ORDERS BETA: Regroup on the flagship\./);
});

test('recorded text never reaches innerHTML', () => {
  const { doc, list, log } = logFixture();
  log.update(40);
  assert.deepEqual(doc.innerHTMLWrites, []);
  const row = list.children.find(c => c.textContent.includes('COMMS'));
  assert.ok(row.textContent.includes(HOSTILE));
  assert.ok(!allElements(list).some(el => el.tagName === 'IMG' || el.tagName === 'B'));
});

test('the HUD index is rebuilt on live append and matches the loader events', () => {
  const { loader } = logFixture();
  const before = loader.hud;
  loader.appendLiveData({ events: [ev(41, 'captain_log', 'alpha_1', { note: 'Reloading.' })], sim_trace: [{ t: 41, ships: {} }] });
  assert.notEqual(loader.hud, before);
  assert.equal(loader.hud.comms.at(-1).text, 'Reloading.');
  assert.deepEqual(buildHudIndex(loader).log.map(r => r.id), loader.hud.log.map(r => r.id));
});
