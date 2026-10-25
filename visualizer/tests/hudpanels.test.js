import test from 'node:test';
import assert from 'node:assert/strict';
import { FakeDocument, build, allElements } from './fakedom.js';
import { buildHudIndex } from '../src/HudIndex.js';
import { computeSides, resultTitle, cleanReason, ResultBanner, Scoreboard, resultBannerDue } from '../src/Scoreboard.js';
import { feedAt, killLines, KillFeed } from '../src/KillFeed.js';
import { timelineMarks, sparkPaths, TimelineMarkers } from '../src/TimelineMarkers.js';
import { registryDensity, registryStatus } from '../src/Registry.js';
import { CommsFeed, visibleComms, standingOrders } from '../src/CommsFeed.js';

const HOSTILE = '<img src=x onerror="alert(1)"><b>Strike your colours</b>';
const ships = {
  alpha_1: { faction: 'alpha', name: 'TIS Dart-1', type: 'corvette' },
  alpha_2: { faction: 'alpha', name: 'TIS Sovereign', type: 'battleship' },
  beta_1: { faction: 'beta', name: 'OCS Falchion-1', type: 'destroyer' },
  beta_2: { faction: 'beta', name: 'OCS Lancer-1', type: 'battlecruiser' }
};
let seq = 0;
const ev = (t, type, ship_id = null, data = {}) => ({ event_id: `e${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
const msg = (t, id, data) => ev(t, 'comms_message', data.sender_id ?? null, { msg_id: id, structured: {}, checkpoint: Math.round(t / 30), sent_t: t, ...data });

function fixture() {
  const events = [
    ev(30, 'decision_point_reached'),
    ev(30, 'admiral_plan', null, { faction: 'alpha', admiral_name: 'Admiral A', plan: 'Phase 1: close.' }),
    ev(30, 'admiral_directive', null, { faction: 'alpha', admiral_name: 'Admiral A', directive: 'All ships close on Falchion-1.' }),
    ev(30, 'admiral_directive', null, { faction: 'beta', admiral_name: 'Admiral B', directive: 'Hold the wall.' }),
    ev(30, 'admiral_order', 'alpha_1', { faction: 'alpha', admiral_name: 'Admiral A', order_text: 'Launch 2 at Falchion-1.', priority: 'CRITICAL', torpedo_salvo: 2, torpedo_target: 'beta_1' }),
    ev(30, 'captain_decision', 'alpha_1', { maneuver_type: 'INTERCEPT', commands: [] }),
    ev(30, 'captain_report', 'alpha_1', { faction: 'alpha', checkpoint: 1, status: 'DEVIATE', reason: 'Lancer-1 is closer.', ship_name: 'TIS Dart-1' }),
    msg(30, 'm1', { channel: 'fleet_net', sender_id: 'alpha_1', sender_role: 'captain', sender_name: 'TIS Dart-1', faction: 'alpha', recipient_ids: ['alpha_2'], text: 'Contact: Falchion-1 at 280 km.', deliver_at_t: 60 }),
    msg(30, 'm2', { channel: 'enemy_broadcast', sender_id: 'beta_1', sender_role: 'captain', sender_name: 'OCS Falchion-1', faction: 'beta', recipient_ids: ['alpha_1', 'alpha_2'], text: HOSTILE, deliver_at_t: 60 }),
    ev(60, 'decision_point_reached'),
    ev(60, 'comms_delivered', 'alpha_2', { msg_id: 'm1', recipient_id: 'alpha_2', delivered_t: 60, checkpoint: 2 }),
    ev(60, 'comms_delivered', 'alpha_1', { msg_id: 'm2', recipient_id: 'alpha_1', delivered_t: 60, checkpoint: 2 }),
    ev(60, 'admiral_directive', null, { faction: 'alpha', admiral_name: 'Admiral A', directive: 'Salvo now.' }),
    ev(60, 'captain_log', 'alpha_2', { note: 'Reloading.' }),
    ev(75, 'torpedo_launched', 'alpha_1', { target_id: 'beta_1' }),
    ev(90, 'ship_dying', 'beta_1', { killer_id: 'alpha_1', detonation_in_s: 20, ordnance_id: 'torp_alpha_1_0001' }),
    ev(104, 'ship_destroyed', 'beta_1', { killer_id: 'alpha_1', cause: 'reactor_destroyed' })
  ];
  const sourceData = { provenance: { draft: { alpha: { ships: [{ ship_id: 'alpha_1', cost: 18 }] }, beta: { ships: [{ ship_id: 'beta_1', cost: 31 }] } } },
    alpha_model: 'anthropic/claude-sonnet-5', beta_model: 'deepseek/deepseek-v4.1-flash' };
  const hud = buildHudIndex({ events, ships, simTrace: [], metadata: { winner: 'alpha', resultReason: 'Beta fleet eliminated' }, sourceData, duration: 120 });
  return { events, hud };
}

test('scoreboard sides: fighting ships, hull, torpedoes, kills with draft points, strength share', () => {
  const { hud } = fixture();
  const states = { alpha_1: { hull: 80 }, alpha_2: { hull: 100 }, beta_1: { dying: true, hull: 40 }, beta_2: { hull: 50 } };
  const sides = computeSides(states, [{ source: 'alpha_1' }, { source: 'alpha_2' }, { source: 'beta_2' }], ships, hud, 95);
  assert.equal(sides.alpha.alive, 2);
  assert.equal(sides.beta.alive, 1);
  assert.equal(sides.beta.dying, 1);
  assert.equal(sides.alpha.hullPct, 90);
  assert.equal(sides.beta.hullPct, 25, 'a dying ship counts as 0%');
  assert.deepEqual([sides.alpha.torps, sides.beta.torps], [2, 1]);
  assert.deepEqual([sides.alpha.kills, sides.alpha.points, sides.beta.kills], [1, 31, 0]);
  assert.deepEqual(sides.beta.pips.map(p => p.state), ['dying', 'alive']);
  // corvette 10 * 0.8 + battleship 50 vs battlecruiser 40 * 0.5
  assert.ok(Math.abs(sides.share - 58 / 78) < 1e-9);
  assert.equal(computeSides(states, [], ships, hud, 80).alpha.kills, 0, 'kills are time-indexed');
});

test('result banner: winner, budget stop, draw; exception prefixes are stripped', () => {
  assert.equal(resultTitle({ winner: 'beta' }), 'BETA VICTORIOUS');
  assert.equal(resultTitle({ winner: null, stopped: true }), 'BATTLE STOPPED');
  assert.equal(resultTitle({ winner: null, stopped: false, reason: 'Too close to call' }), 'DRAW');
  assert.equal(cleanReason('RuntimeError: Spending ceiling reached'), 'Spending ceiling reached');
  const doc = new FakeDocument();
  const root = build(doc, ['div', { class: 'result-banner hidden' }, ['button', { 'data-close': '' }], ['span', { 'data-kicker': '' }],
    ['strong', { 'data-title': '' }], ['span', { 'data-reason': '' }], ['div', { 'data-tally': '' }]]);
  const banner = new ResultBanner({ root, doc });
  const { hud } = fixture();
  const sides = computeSides({}, [], ships, hud, 120);
  banner.update(false, hud.result, sides, 120);
  assert.ok(root.classList.contains('hidden'));
  banner.update(true, hud.result, sides, 120);
  assert.ok(!root.classList.contains('hidden'));
  assert.match(root.textContent, /ALPHA VICTORIOUS/);
  assert.match(root.textContent, /1 kill \(31 pts\)/);
  root.querySelector('[data-close]').click();
  banner.update(true, hud.result, sides, 120);
  assert.ok(root.classList.contains('hidden'), 'dismissal sticks until the next recording');
});

test('result banner is the replay verdict only: a live session (running or ended) leaves it to the live end card', () => {
  assert.equal(resultBannerDue(120, 120, null), true);
  assert.equal(resultBannerDue(119, 120, null), false);
  assert.equal(resultBannerDue(140, 120, null), true, 'wreck epilogue past the last frame');
  assert.equal(resultBannerDue(120, 120, { ended: false }), false);
  assert.equal(resultBannerDue(120, 120, { ended: true }), false, 'never stacks with the live end card');
  assert.equal(resultBannerDue(0, 0, null), false, 'nothing loaded');
});

test('scoreboard DOM writes the model and only changed values', () => {
  const doc = new FakeDocument();
  const side = f => ['div', { 'data-side': f }, ['span', { 'data-model': '' }], ['b', { 'data-alive': '' }], ['span', { 'data-total': '' }],
    ['span', { 'data-pips': '' }], ['span', { 'data-hull': '' }], ['span', { 'data-torps': '' }], ['span', { 'data-kills': '' }],
    ['span', { 'data-points': '' }], ['span', { 'data-spend': '' }]];
  const root = build(doc, ['div', {}, side('alpha'), ['span', {}, ['i', { 'data-balance': '' }]], side('beta')]);
  const board = new Scoreboard({ root, doc });
  const { hud } = fixture();
  board.populate(ships, hud);
  board.update({ beta_1: { destroyed: true } }, [], ships, hud, 110);
  const alpha = root.querySelector('[data-side="alpha"]');
  assert.equal(alpha.querySelector('[data-model]').textContent, 'claude-sonnet-5');
  assert.equal(alpha.querySelector('[data-kills]').textContent, '☠ 1');
  assert.equal(alpha.querySelector('[data-points]').textContent, '31 pts');
  assert.equal(root.querySelector('[data-side="beta"]').querySelector('[data-alive]').textContent, '1');
  assert.equal(root.querySelector('[data-side="beta"]').querySelectorAll('i').length, 2, 'one pip per ship');
  // torn down (live battle switch, file selector): nothing of the old battle stays in the header
  board.clear();
  for (const f of ['alpha', 'beta']) {
    const el = root.querySelector(`[data-side="${f}"]`);
    assert.equal(el.querySelector('[data-model]').textContent, '');
    assert.equal(el.querySelector('[data-kills]').textContent, '☠ 0');
    assert.equal(el.querySelector('[data-points]').textContent, '');
    assert.deepEqual([el.querySelector('[data-alive]').textContent, el.querySelector('[data-total]').textContent], ['0', '0']);
    assert.equal(el.querySelectorAll('i').length, 0, 'no pips');
  }
  // and the next battle repopulates from scratch
  board.populate(ships, hud);
  board.update({}, [], ships, hud, 10);
  assert.equal(alpha.querySelector('[data-kills]').textContent, '☠ 0');
  assert.equal(alpha.querySelectorAll('i').length, 2);
});

test('kill feed: the hulk counts down, then lingers as destroyed, then clears', () => {
  const { hud } = fixture();
  assert.deepEqual(feedAt(hud.kills, 89), []);
  const dying = feedAt(hud.kills, 95);
  assert.equal(dying[0].state, 'dying');
  assert.equal(dying[0].in, 9);
  assert.match(killLines(dying[0].kill, dying[0]).status, /REACTOR IN 9s/);
  assert.match(killLines(dying[0].kill, dying[0]).detail, /by TIS Dart-1 · torpedo/);
  assert.equal(feedAt(hud.kills, 110)[0].state, 'destroyed');
  assert.deepEqual(feedAt(hud.kills, 120), []);
  const doc = new FakeDocument();
  const root = doc.createElement('div');
  const seeks = [];
  const feed = new KillFeed({ root, doc, onSeek: k => seeks.push(k.t) });
  feed.update(hud.kills, 95);
  const first = root.firstChild;
  feed.update(hud.kills, 95.4);
  assert.equal(root.firstChild, first, 'no rebuild until the countdown second changes');
  first.click();
  assert.deepEqual(seeks, [90]);
  assert.equal(doc.innerHTMLWrites.length, 0);
});

test('timeline marks: numbered checkpoints, salvos, kills; every marker seeks', () => {
  const { hud } = fixture();
  const marks = timelineMarks(hud, 120, ships);
  assert.deepEqual(marks.filter(m => m.kind === 'checkpoint').map(m => [m.label, m.pct, m.flag]), [['1', 25, 'deviation'], ['2', 50, null]]);
  assert.match(marks[0].title, /CP1 · 00:30 · 3 orders · 0 ACK · 1 DEVIATE\/UNABLE · 2 messages/);
  const salvo = marks.find(m => m.kind === 'salvo');
  assert.match(salvo.title, /ALPHA torpedo salvo ×1 → OCS Falchion-1/);
  const kill = marks.find(m => m.kind === 'kill');
  assert.equal(kill.faction, 'beta');
  assert.equal(kill.seekLead, 5);
  assert.ok(sparkPaths({ alpha: [0, 2, 1], beta: [1, 0, 0] }).alpha.startsWith('M'));
  assert.equal(hud.hitBins.bin, 5);
  const doc = new FakeDocument();
  const events = doc.createElement('div'), cps = doc.createElement('div');
  const seeks = [];
  new TimelineMarkers({ events, checkpoints: cps, doc, onSeek: (t, id) => seeks.push([t, id]) }).build(hud, 150, ships);
  assert.equal(cps.children.length, 2);
  // 120 s of bins (+1 partial) on a 150 s track that includes the epilogue
  assert.equal(events.querySelector('svg').style.width, `${(25 * 5 / 150) * 100}%`);
  events.querySelector('.marker-kill').click();
  assert.deepEqual(seeks, [[85, 'beta_1']]);
});

test('registry density fits the sidebar; dying ships count down', () => {
  assert.equal(registryDensity(8, 900), '');
  assert.equal(registryDensity(8, 486), 'compact');
  assert.equal(registryDensity(14, 486), 'compact');
  assert.equal(registryDensity(16, 486), 'dense');
  assert.equal(registryDensity(16, 846), 'compact');
  const { hud } = fixture();
  assert.deepEqual(registryStatus({ dying: true }, hud.kills[0], 95), { text: '☢ 9s', level: 'dying' });
  assert.deepEqual(registryStatus({ destroyed: true }, null, 200), { text: 'LOST', level: 'lost' });
  assert.deepEqual(registryStatus({ hull: 20.4 }, null, 0), { text: '20%', level: 'low' });
});

function commsFixture() {
  const doc = new FakeDocument();
  const root = build(doc, ['aside', { class: 'hidden' },
    ['button', { 'data-comms-faction': 'all' }], ['button', { 'data-comms-faction': 'alpha' }], ['button', { 'data-comms-faction': 'beta' }],
    ['button', { 'data-comms-group': 'all' }], ['button', { 'data-comms-group': 'issues' }],
    ['div', { 'data-standing': '' }], ['div', { 'data-list': '' }]]);
  const { hud } = fixture();
  const seeks = [];
  const feed = new CommsFeed({ root, doc, store: null, onSeek: r => seeks.push(r.t), onWhy: () => {} });
  feed.setOpen(true);
  feed.setIndex(hud, true);
  return { doc, root, feed, hud, seeks, list: root.querySelector('[data-list]') };
}

const rowTags = list => list.querySelectorAll('.comms-row').map(r => r.querySelector('.comms-tag').textContent);

test('command net groups by checkpoint (newest on top) and is idempotent while parked', () => {
  const { feed, list, root } = commsFixture();
  assert.equal(feed.update(31), 'rebuild');
  assert.deepEqual(rowTags(list), ['PLAN', 'DIRECTIVE', 'DIRECTIVE', 'ORDER', 'DEVIATE', 'FLEET NET', 'OPEN CHANNEL']);
  for (let i = 0; i < 50; i++) assert.equal(feed.update(31), 'same');
  assert.equal(feed.update(61), 'append');
  assert.deepEqual(list.children.map(g => g.dataset.cp), ['2', '1']);
  assert.deepEqual(rowTags(list).slice(0, 2), ['DIRECTIVE', 'LOG']);
  assert.equal(new Set(list.querySelectorAll('.comms-row').map(r => r.dataset.id)).size, 9);
  assert.equal(feed.update(31), 'rebuild', 'seeking back drops the newer checkpoint');
  assert.deepEqual(list.children.map(g => g.dataset.cp), ['1']);
  const standing = root.querySelector('[data-standing]').querySelectorAll('.comms-standing-text').map(e => e.textContent);
  assert.deepEqual(standing, ['All ships close on Falchion-1.', 'Hold the wall.']);
});

test('command net: delivery state follows the playhead; filters; recorded text stays text', () => {
  const { feed, list, doc, seeks } = commsFixture();
  feed.update(31);
  const chip = () => list.querySelector('.comms-delivery').textContent;
  assert.equal(chip(), 'IN TRANSIT · due 01:00');
  feed.update(61);
  assert.equal(chip(), '✓ DELIVERED 01:00');
  const enemy = list.querySelectorAll('.comms-row').find(r => r.classList.contains('kind-enemy'));
  assert.equal(enemy.querySelector('.comms-text').textContent, HOSTILE);
  assert.equal(enemy.querySelector('.comms-text').children.length, 0, 'no markup parsed out of the text');
  assert.ok(!allElements(list).some(el => el.tagName === 'IMG'));
  assert.deepEqual(doc.innerHTMLWrites, []);
  enemy.querySelector('.comms-time').click();
  assert.deepEqual(seeks, [30]);
  feed.setFilter('beta', 'all');
  feed.update(61);
  assert.deepEqual(rowTags(list), ['DIRECTIVE', 'OPEN CHANNEL']);
  feed.setFilter('all', 'issues');
  feed.update(61);
  assert.deepEqual(rowTags(list), ['ORDER', 'DEVIATE'], 'critical orders and deviations');
});

test('command net: a report shows its delivery to the admiral and swaps to the compliance the admiral read', () => {
  const doc = new FakeDocument();
  const root = build(doc, ['aside', {}, ['button', { 'data-comms-group': 'all' }], ['button', { 'data-comms-group': 'issues' }],
    ['div', { 'data-standing': '' }], ['div', { 'data-list': '' }]]);
  const compliance = (launched, final, deviations) => ({ ordered_maneuver: 'INTERCEPT', executed_maneuver: 'INTERCEPT', ordered_target: 'OCS Falchion-1',
    actual_target: 'OCS Falchion-1', ordered_salvo: 2, launched, final, deviations });
  const events = [
    ev(30, 'decision_point_reached'),
    ev(30, 'admiral_order', 'alpha_1', { faction: 'alpha', order_text: 'Launch 2 at Falchion-1.', torpedo_salvo: 2, torpedo_target: 'beta_1' }),
    ev(30, 'captain_report', 'alpha_1', { report_id: 'r1-alpha_1', faction: 'alpha', checkpoint: 1, ship_name: 'TIS Dart-1', status: 'ACK',
      reason: 'Salvo away on the mark.', compliance: compliance(2, false, []) }),
    // Routed nowhere recorded: its delivery chip must stay hidden
    msg(30, 'm9', { channel: 'captain_to_admiral', sender_id: 'alpha_1', sender_role: 'captain', sender_name: 'TIS Dart-1', faction: 'alpha',
      recipient_ids: [], text: 'Request permission to close.', deliver_at_t: 60 }),
    ev(60, 'decision_point_reached'),
    ev(60, 'comms_delivered', 'alpha_admiral', { msg_id: 'r1-alpha_1', recipient_id: 'alpha_admiral', delivered_t: 60, checkpoint: 2,
      compliance: compliance(1, true, ['salvo: ordered 2, launched 1']) })
  ];
  const hud = buildHudIndex({ events, ships, simTrace: [], metadata: {}, sourceData: { decision_interval_s: 30 }, duration: 90 });
  const feed = new CommsFeed({ root, doc, store: null });
  feed.setOpen(true);
  feed.setIndex(hud, true);
  const list = root.querySelector('[data-list]');
  const report = () => list.querySelectorAll('.comms-row').find(r => r.classList.contains('kind-report'));
  const chips = () => report().querySelectorAll('.comms-deviation').map(c => c.textContent);
  feed.update(45);
  assert.equal(report().querySelector('.comms-delivery').textContent, 'IN TRANSIT · due 01:00');
  assert.deepEqual(chips(), [], 'as filed: the salvo was committed');
  assert.ok(!report().classList.contains('off-orders'));
  feed.update(60);
  assert.equal(report().querySelector('.comms-delivery').textContent, '✓ DELIVERED 01:00');
  assert.deepEqual(chips(), ['SALVO: ordered 2, launched 1']);
  assert.ok(report().classList.contains('off-orders'), 'the ACK row carries the deviation edge');
  assert.ok(report().querySelector('.comms-deviation').classList.contains('final'));
  const delivery = list.querySelectorAll('.comms-delivery');
  assert.equal(delivery.length, 2);
  assert.ok(delivery.every(c => c.hidden === !c.textContent) && delivery.some(c => c.hidden), 'an empty delivery chip is hidden, not an empty box');
  assert.match(report().querySelector('.comms-deviation').title, /as the admiral read it at 01:00/);
  feed.update(45);
  assert.deepEqual(chips(), [], 'scrubbing back restores the filed verdict');
  assert.ok(!report().classList.contains('off-orders'));
  // ISSUES: the ACK joins once the admiral's copy shows it off orders
  feed.setFilter('all', 'issues');
  feed.update(45);
  assert.equal(list.querySelectorAll('.comms-row').length, 0);
  feed.update(60);
  assert.deepEqual(list.querySelectorAll('.comms-row').map(r => r.querySelector('.comms-tag').textContent), ['ACK']);
  assert.deepEqual(doc.innerHTMLWrites, []);
  // The timeline flags the checkpoint whose ACK drifted
  const cp1 = timelineMarks(hud, 90, ships).find(m => m.kind === 'checkpoint' && m.t === 30);
  assert.equal(cp1.flag, 'deviation');
  assert.match(cp1.title, /1 ACK \(1 off orders\)/);
});

test('visible rows and standing orders are pure functions of time', () => {
  const { hud } = fixture();
  assert.equal(visibleComms(hud.comms, 29).length, 0);
  assert.equal(visibleComms(hud.comms, 61, 'all', 'all', 3).length, 3, 'capped to the newest rows');
  assert.equal(standingOrders(hud.comms, 61).alpha.directive.text, 'Salvo now.');
  assert.equal(standingOrders(hud.comms, 61).alpha.plan.text, 'Phase 1: close.');
});

test('command net with no traffic says so instead of rendering an empty column', () => {
  const doc = new FakeDocument();
  const root = build(doc, ['aside', { class: 'hidden' },
    ['button', { 'data-comms-faction': 'all' }], ['button', { 'data-comms-group': 'all' }],
    ['div', { 'data-standing': '' }], ['div', { 'data-list': '' }]]);
  const ships = { alpha_1: { faction: 'alpha', name: 'TIS Dart-1', type: 'corvette' } };
  const hud = buildHudIndex({ events: [], ships, simTrace: [], metadata: {}, sourceData: {}, duration: 120 });
  assert.equal(hud.comms.length, 0);
  const feed = new CommsFeed({ root, doc, store: null });
  feed.setOpen(true);
  feed.setIndex(hud, true);
  feed.update(30);
  const list = root.querySelector('[data-list]');
  assert.equal(list.querySelector('.comms-empty')?.textContent, 'No command traffic recorded yet.');
  assert.equal(feed.update(30), 'same', 'parked: nothing re-rendered');
  assert.equal(list.querySelectorAll('.comms-empty').length, 1);
  // traffic recorded but none yet at the playhead (a live battle before CP1)
  const busy = commsFixture();
  busy.feed.update(10);
  assert.equal(busy.list.querySelector('.comms-empty')?.textContent, 'No command traffic yet.');
  busy.feed.update(40);
  assert.equal(busy.list.querySelector('.comms-empty'), null, 'the first rows replace the line');
  busy.feed.setFilter('all', 'issues');
  busy.feed.update(10);
  assert.equal(busy.list.querySelector('.comms-empty')?.textContent, 'No traffic on this filter yet.');
});
