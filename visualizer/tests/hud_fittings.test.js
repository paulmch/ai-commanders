import test from 'node:test';
import assert from 'node:assert/strict';
import { FakeDocument, build, allElements } from './fakedom.js';
import { buildHudIndex, appendHudIndex, commsMatches } from '../src/HudIndex.js';
import { fittingLayouts, slotsForModule, fittingsAt, fittingsText, fittingsFlag, blastLine } from '../src/FittingsIndex.js';
import { pipsOf, pipStrip, FittingsCard } from '../src/FittingsHud.js';
import { computeSides, endCause, armsText, Scoreboard, ResultBanner } from '../src/Scoreboard.js';
import { noticesAt, KillFeed } from '../src/KillFeed.js';
import { timelineMarks } from '../src/TimelineMarkers.js';
import { Registry, registryStatus } from '../src/Registry.js';

// Pass-5 battle damage in the spectator HUD (docs/pass-5-recording-additions.md):
// fitting state at time t, blast victims, fighting-power counts, the notices a
// spectator must not miss - all pure functions of the recording, identical
// whether it was parsed whole or streamed, and absent on older recordings.

const ships = {
  alpha_1: { faction: 'alpha', name: 'TIS Harpoon-1', type: 'cruiser_torpedo' },
  alpha_2: { faction: 'alpha', name: 'TIS Dart-1', type: 'corvette' },
  alpha_3: { faction: 'alpha', name: 'TIS Ward-1', type: 'frigate' },
  beta_1: { faction: 'beta', name: 'OCS Sovereign-1', type: 'battleship' },
  beta_2: { faction: 'beta', name: 'OCS Falchion-1', type: 'destroyer' }
};
const w = (guns, tubes, pd, gunType = 'coilgun_mk3') => {
  const out = {};
  let n = 0;
  for (let i = 0; i < guns; i++) out[`weapon_${n++}`] = { type: gunType };
  for (let i = 0; i < tubes; i++) out[`weapon_${n++}`] = { type: 'torpedo_launcher' };
  for (let i = 0; i < pd; i++) out[`pd_${i}`] = { type: 'pd_laser' };
  return out;
};
const sourceData = () => ({
  alpha_fleet: { ships: [
    { ship_id: 'alpha_1', ship_type: 'cruiser_torpedo', ship_name: 'TIS Harpoon-1', weapons: w(0, 4, 4) },
    { ship_id: 'alpha_2', ship_type: 'corvette', ship_name: 'TIS Dart-1', weapons: w(0, 1, 1) },
    { ship_id: 'alpha_3', ship_type: 'frigate', ship_name: 'TIS Ward-1', weapons: w(1, 0, 2) }
  ] },
  beta_fleet: { ships: [
    { ship_id: 'beta_1', ship_type: 'battleship', ship_name: 'OCS Sovereign-1', weapons: w(5, 0, 3) },
    { ship_id: 'beta_2', ship_type: 'destroyer', ship_name: 'OCS Falchion-1', weapons: w(2, 0, 2) }
  ] }
});

function events() {
  let seq = 0;
  const out = [];
  const ev = (t, type, ship_id = null, data = {}) => out.push({ event_id: `e${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  const mount = (t, ship, kind, slot, health, extra = {}) => ev(t, 'mount_damaged', ship, { kind, slot, index: Number(slot.split('_')[1]) || 0,
    health_pct: health, destroyed: health === 0, cause: 'surface_hit', source_ship_id: ship.startsWith('alpha') ? 'beta_1' : 'alpha_1', ...extra });
  ev(0, 'battle_start');
  for (let i = 0; i < 8; i++) ev(40 + i, 'torpedo_launched', 'alpha_2', { target_id: 'beta_1' });   // the corvette empties its magazine
  ev(50, 'torpedo_launched', 'alpha_1', { target_id: 'beta_1' });
  mount(60, 'beta_1', 'gun', 'weapon_4', 40);                 // damaged ...
  mount(64, 'beta_1', 'gun', 'weapon_4', 0);                  // ... then destroyed: the latest event wins
  mount(66, 'beta_1', 'pd', 'pd_1', 0);
  ev(70, 'module_destroyed', 'beta_2', { module_name: 'Spinal Coiler Mount' });      // older path: weapon_0, no mount_damaged
  mount(80, 'alpha_1', 'launcher', 'weapon_1', 39, { torpedoes_remaining: 40 });     // re-syncs the round count
  mount(90, 'alpha_1', 'engine', 'Main Engine Assembly', 53.1, { thrust_fraction: 0.531 });
  ev(95, 'module_destroyed', 'beta_2', { module_name: 'Dorsal Turret Mount' });      // weapon_1: Falchion-1 is disarmed
  ev(100, 'magazine_cookoff', 'beta_1', { module_name: 'Main Magazine', torpedoes_lost: 0, torpedoes_remaining: 0, gun_rounds_lost: 3825,
    modules_damaged: ['Forward Hull Section'], modules_destroyed: ['Forward Hull Section'], source_ship_id: 'alpha_1' });
  mount(104, 'alpha_3', 'gun', 'weapon_0', 0);                // Ward-1 disarmed ...
  ev(104, 'ship_dying', 'alpha_3', { killer_id: 'beta_1', detonation_in_s: 6 });     // ... and knocked out the same second
  ev(110, 'ship_destroyed', 'alpha_3', { cause: 'reactor_detonation', killer_id: 'beta_1' });
  ev(110, 'blast_damage', 'beta_2', { source_ship_id: 'alpha_3', distance_km: 7.3, intensity: 0.2, hit_location: 'nose', armor_ablation_cm: 1.4,
    armor_remaining_cm: 200, heat_gj: 20, heat: 0.3, radiators: [] });
  ev(110, 'mount_damaged', 'beta_2', { kind: 'pd', slot: 'pd_1', index: 1, health_pct: 0, destroyed: true, cause: 'reactor_blast', source_ship_id: 'alpha_3' });
  ev(110, 'torpedo_intercepted', 'alpha_1', { torpedo_id: 'torp_alpha_1_0001', cause: 'reactor_blast', blast_ship_id: 'alpha_3' });
  ev(110, 'reactor_blast', 'alpha_3', { ship_type: 'frigate', cause: 'reactor_detonation', radius_km: 10.6, ships_hit: ['beta_2'],
    torpedoes_destroyed: ['torp_alpha_1_0001'], torpedoes_blinded: ['torp_alpha_2_0002', 'torp_alpha_2_0003'] });
  mount(120, 'alpha_1', 'engine', 'Main Engine Assembly', 0, { thrust_fraction: 0 });
  ev(120, 'module_destroyed', 'alpha_1', { module_name: 'Main Engine Assembly', cause: 'surface_hit' });
  ev(150, 'battle_end', null, { winner: 'alpha', reason: 'Beta fleet combat-ineffective (no weapon left that can deal damage)' });
  return out;
}

const loaderOf = (evs, src = sourceData()) => ({ events: evs, ships, simTrace: [], metadata: {}, sourceData: src, duration: 150 });
const hudOf = (evs = events(), src) => buildHudIndex(loaderOf(evs, src));

test('fitting layouts come from the fleet specs; a destroyed module maps to its slot the way the engine does', () => {
  const lay = fittingLayouts(sourceData(), ships);
  assert.deepEqual(lay.beta_1.guns, ['weapon_0', 'weapon_1', 'weapon_2', 'weapon_3', 'weapon_4']);
  assert.deepEqual([lay.alpha_1.tubes.length, lay.alpha_1.pd.length, lay.alpha_1.guns.length], [4, 4, 0]);
  assert.equal(lay.alpha_1.faction, 'alpha');
  // 1v1 recordings: alpha_specs / beta_specs
  const duel = fittingLayouts({ alpha_specs: { ship_type: 'destroyer', weapons: w(2, 0, 2) } }, {});
  assert.deepEqual(duel.alpha.guns, ['weapon_0', 'weapon_1']);
  // Explicit destroyer map, then Nth gun module -> Nth non-PD slot, PD array -> every turret
  assert.deepEqual(slotsForModule('destroyer', lay.beta_2, 'Dorsal Turret Mount'), { slots: ['weapon_1'], engine: false });
  assert.deepEqual(slotsForModule('destroyer', lay.beta_2, 'PD Laser Ventral').slots, ['pd_1']);
  assert.deepEqual(slotsForModule('battleship', lay.beta_1, 'Heavy Coilgun Battery B').slots, ['weapon_2']);
  assert.deepEqual(slotsForModule('battleship', lay.beta_1, 'Point Defense Array').slots, ['pd_0', 'pd_1', 'pd_2']);
  assert.equal(slotsForModule('battleship', lay.beta_1, 'Main Engine Assembly').engine, true);
  assert.deepEqual(slotsForModule('battleship', lay.beta_1, 'Crew Quarters'), { slots: [], engine: false });
  // A hull the module table does not know (torpedo cruiser) maps nothing but its drive
  assert.deepEqual(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Torpedo Bay A').slots, []);
  assert.equal(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Main Engine Assembly').engine, true);
});

test('fitting state at time t: the latest event at or before t per fitting, forwards and backwards', () => {
  const { fittings } = hudOf();
  assert.equal(fittings.active, true);
  const at = (id, t) => fittingsAt(fittings, id, t);
  assert.deepEqual(at('beta_1', 0).guns, ['ok', 'ok', 'ok', 'ok', 'ok']);
  assert.deepEqual(at('beta_1', 60).guns, ['ok', 'ok', 'ok', 'ok', 'damaged']);
  assert.deepEqual(at('beta_1', 65).guns, ['ok', 'ok', 'ok', 'ok', 'destroyed']);
  assert.deepEqual(at('beta_1', 70).pd, ['ok', 'destroyed', 'ok']);
  assert.deepEqual([at('beta_1', 70).gunsUp, at('beta_1', 70).pdUp, at('beta_1', 70).armed], [4, 2, true]);
  // Scrubbing back gives the earlier state again, and the same object until the next change
  assert.deepEqual(at('beta_1', 62).guns, ['ok', 'ok', 'ok', 'ok', 'damaged']);
  assert.equal(at('beta_1', 66), at('beta_1', 99), 'identity is stable between changes');
  // module_destroyed (older path) silences the matching slots
  assert.deepEqual(at('beta_2', 69).guns, ['ok', 'ok']);
  assert.deepEqual(at('beta_2', 70).guns, ['destroyed', 'ok']);
  assert.equal(at('beta_2', 94).disarmed, false);
  assert.equal(at('beta_2', 95).disarmed, true);
  assert.equal(fittingsFlag(at('beta_2', 95)), 'DISARMED');
  // Drive: thrust fraction, then adrift
  assert.equal(at('alpha_1', 89).thrust, 1);
  assert.equal(at('alpha_1', 90).thrust, 0.531);
  assert.equal(at('alpha_1', 119).adrift, false);
  assert.equal(at('alpha_1', 120).adrift, true);
  assert.equal(fittingsFlag(at('alpha_1', 120)), 'ADRIFT');
  assert.match(fittingsText(at('alpha_1', 100)), /Tubes 4\/4 \(1 damaged\), 40 torpedoes · PD 4\/4 · Drive 53%/);
  assert.equal(fittingsAt(fittings, 'nobody', 50), null);
});

test('torpedo rounds: launches count down from the hull magazine, a recorded count re-syncs, an empty ship is flagged', () => {
  const { fittings } = hudOf();
  const at = (id, t) => fittingsAt(fittings, id, t);
  assert.equal(at('alpha_2', 0).rounds, 8);
  assert.equal(at('alpha_2', 43).rounds, 4);
  assert.equal(at('alpha_2', 46).armed, true);
  assert.deepEqual([at('alpha_2', 47).rounds, at('alpha_2', 47).dry, at('alpha_2', 47).armed], [0, true, false]);
  assert.equal(fittingsFlag(at('alpha_2', 47)), 'EMPTY');
  assert.deepEqual(pipsOf(at('alpha_2', 47)).map(p => `${p.kind}:${p.s}`), ['tube:empty', 'pd:ok', 'drive:ok']);
  assert.equal(at('alpha_1', 50).rounds, 47);
  assert.equal(at('alpha_1', 80).rounds, 40, 'mount_damaged.torpedoes_remaining wins over the running count');
  // A hull with no magazine entry has no count until an event states one
  const src = sourceData();
  src.alpha_fleet.ships[1].ship_type = 'mystery_boat';
  const unknown = hudOf(events(), src).fittings;
  assert.equal(fittingsAt(unknown, 'alpha_2', 47).rounds, null);
  assert.equal(fittingsAt(unknown, 'alpha_2', 47).armed, true);
});

test('blast victims, cook-offs, disarmed and adrift ships become notices; a ship already knocked out gets none', () => {
  const hud = hudOf();
  const blast = hud.blasts.alpha_3;
  assert.deepEqual([blast.t, blast.radiusKm, blast.torpsKilled, blast.torpsBlinded], [110, 10.6, 1, 2]);
  assert.deepEqual(blast.victims.map(v => [v.shipId, v.name, v.distanceKm, v.armorCm, v.mounts]), [['beta_2', 'OCS Falchion-1', 7.3, 1.4, [{ kind: 'pd', destroyed: true }]]]);
  assert.equal(blastLine(blast), 'BLAST 11 km · caught OCS Falchion-1 · 1 torpedo killed, 2 blinded');
  assert.equal(blastLine({ radiusKm: 15, victims: [], torpsKilled: 0, torpsBlinded: 0 }), '', 'a blast that touched nothing adds no line');
  const rows = hud.notices.map(n => [n.t, n.kind, n.shipId, n.status]);
  assert.deepEqual(rows, [
    [90, 'drive', 'alpha_1', 'DRIVE HIT · 53% THRUST'],
    [95, 'disarmed', 'beta_2', 'DISARMED'],
    [100, 'cookoff', 'beta_1', 'MAGAZINE COOK-OFF'],
    [110, 'blast', 'beta_2', 'CAUGHT IN BLAST'],
    [120, 'adrift', 'alpha_1', 'ADRIFT']
  ]);
  assert.ok(!hud.notices.some(n => n.shipId === 'alpha_3'), 'Ward-1 lost its gun the second it was knocked out: the kill tells that story');
  const caught = hud.notices.find(n => n.kind === 'blast');
  assert.equal(caught.detail, "7.3 km from TIS Ward-1's reactor: −1.4 cm nose armour, 1 PD lost");
  assert.equal(hud.notices.find(n => n.kind === 'cookoff').detail, '3825 rounds burned · 1 compartment wrecked');
  assert.equal(hud.notices.find(n => n.kind === 'adrift').detail, 'drive destroyed by OCS Sovereign-1');
  assert.equal(new Set(hud.notices.map(n => n.id)).size, hud.notices.length, 'notice ids are unique (the DOM dedupes on them)');
  // Feed window: the last 8 s, newest first, capped
  assert.deepEqual(noticesAt(hud.notices, 101).map(n => n.kind), ['cookoff', 'disarmed']);
  assert.deepEqual(noticesAt(hud.notices, 101, 8, 1).map(n => n.kind), ['cookoff']);
  assert.deepEqual(noticesAt(hud.notices, 89), []);
  assert.deepEqual(noticesAt(hud.notices, 140), []);
});

test('scoreboard counts fighting power: armed ships, working guns and tubes on ships still fighting', () => {
  const hud = hudOf();
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { hull: 100 }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  let sides = computeSides(states, [], ships, hud, 30);
  assert.deepEqual([sides.alpha.fit, sides.alpha.armed, sides.alpha.guns, sides.alpha.gunsMax, sides.alpha.tubes, sides.alpha.tubesMax], [true, 3, 1, 1, 5, 5]);
  assert.deepEqual([sides.beta.armed, sides.beta.guns, sides.beta.gunsMax, sides.beta.tubesMax], [2, 7, 7, 0]);
  sides = computeSides(states, [], ships, hud, 96);
  assert.deepEqual([sides.alpha.alive, sides.alpha.armed], [3, 2], 'the empty corvette is afloat but not armed');
  assert.deepEqual([sides.beta.alive, sides.beta.armed, sides.beta.guns], [2, 1, 4]);
  assert.deepEqual(sides.alpha.pips.map(p => p.state), ['alive', 'unarmed', 'alive']);
  assert.deepEqual(sides.beta.pips.map(p => p.state), ['alive', 'unarmed']);
  // A dying or lost ship's guns no longer count
  sides = computeSides({ ...states, beta_1: { hull: 0, dying: true } }, [], ships, hud, 96);
  assert.deepEqual([sides.beta.armed, sides.beta.guns], [0, 0]);
  assert.equal(armsText('GUNS', 4, 7), 'GUNS 4/7');
  assert.equal(armsText('TUBES', 0, 0), '');

  const doc = new FakeDocument();
  const side = f => ['div', { class: `score-side ${f}`, 'data-side': f }, ['span', { 'data-model': '' }], ['b', { 'data-alive': '' }], ['span', { 'data-total': '' }],
    ['span', { 'data-pips': '' }], ['span', { 'data-armed': '' }], ['span', { 'data-guns': '' }], ['span', { 'data-tubes': '' }], ['span', { 'data-hull': '' }],
    ['span', { 'data-torps': '' }], ['span', { 'data-kills': '' }], ['span', { 'data-points': '' }], ['span', { 'data-spend': '' }]];
  const root = build(doc, ['div', {}, side('alpha'), ['div', {}, ['i', { 'data-balance': '' }]], side('beta')]);
  const board = new Scoreboard({ root, doc });
  board.populate(ships, hud);
  board.update(states, [], ships, hud, 96);
  const text = (f, key) => root.querySelector(`[data-side="${f}"]`).querySelector(`[data-${key}]`).textContent;
  assert.deepEqual([text('alpha', 'armed'), text('alpha', 'guns'), text('alpha', 'tubes')], ['2 ARMED', 'GUNS 1/1', 'TUBES 5/5']);
  assert.deepEqual([text('beta', 'armed'), text('beta', 'guns'), text('beta', 'tubes')], ['1 ARMED', 'GUNS 4/7', '']);
  assert.equal(root.querySelector('[data-side="beta"]').querySelector('[data-guns]').dataset.level, 'hurt');
  assert.equal(root.querySelector('[data-side="beta"]').querySelector('[data-pips]').children[1].dataset.state, 'unarmed');
  board.update(states, [], ships, hud, 30);
  assert.equal(text('alpha', 'armed'), '', 'no "armed" readout while every ship afloat can fight');
});

test('older recordings: no fitting data, nothing new on screen', () => {
  // The same battle as a pass-4 engine recorded it: module_destroyed only, with fleet weapon specs
  const old = events().filter(e => !['mount_damaged', 'reactor_blast', 'blast_damage', 'magazine_cookoff'].includes(e.event_type));
  const hud = hudOf(old);
  assert.equal(hud.fittings.active, false, 'module_destroyed alone never switches the fitting HUD on');
  assert.deepEqual([hud.notices, hud.blasts, hud.fittings.states], [[], {}, {}]);
  assert.equal(fittingsAt(hud.fittings, 'beta_2', 100), null);
  // ... and a pass-5 recording without fleet specs has nothing to lay pips out on
  assert.equal(hudOf(events(), {}).fittings.active, false);
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { hull: 100 }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  const sides = computeSides(states, [], ships, hud, 96);
  assert.deepEqual([sides.alpha.fit, sides.alpha.armed, sides.alpha.alive, sides.beta.gunsMax], [false, 3, 3, 0]);
  assert.ok(sides.beta.pips.every(p => p.state === 'alive'));
  assert.deepEqual(registryStatus({ hull: 80 }, null, 96, null), { text: '80%', level: 'ok' });
  assert.ok(!timelineMarks(hud, 150, ships).some(m => m.kind === 'notice' || m.flag === 'blast'));

  const doc = new FakeDocument();
  const root = doc.createElement('div'), alpha = doc.createElement('ul'), beta = doc.createElement('ul');
  const registry = new Registry({ root, alpha, beta, doc });
  registry.populate(ships);
  registry.update(states, 96, hud.kills, hud.fittings);
  assert.equal(allElements(alpha).concat(allElements(beta)).filter(el => el.classList.contains('fit-strip')).length, 0, 'no pip strips');
  const feed = new KillFeed({ root: doc.createElement('div'), doc });
  feed.update(hud.kills, 111, hud);
  assert.ok(!allElements(feed.root).some(el => el.classList.contains('notice') || el.classList.contains('kill-blast')));
});

test('registry rows: pips appear with the first fitting data, the status flags a crippled ship', () => {
  const hud = hudOf();
  const doc = new FakeDocument();
  const root = doc.createElement('div'), alpha = doc.createElement('ul'), beta = doc.createElement('ul');
  const registry = new Registry({ root, alpha, beta, doc });
  registry.populate(ships);
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { hull: 100 }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  registry.update(states, 96, hud.kills, hud.fittings);
  const row = id => [...alpha.children, ...beta.children].find(li => li.dataset.shipId === id);
  const pips = id => allElements(row(id)).filter(el => el.classList.contains('fit')).map(el => `${el.className.replace('fit fit-', '')}:${el.dataset.s}`);
  assert.deepEqual(pips('beta_1'), ['gun:ok', 'gun:ok', 'gun:ok', 'gun:ok', 'gun:destroyed', 'pd:ok', 'pd:destroyed', 'pd:ok', 'drive:ok']);
  assert.deepEqual(pips('beta_2'), ['gun:destroyed', 'gun:destroyed', 'pd:ok', 'pd:ok', 'drive:ok']);
  assert.deepEqual(pips('alpha_1'), ['tube:ok', 'tube:damaged', 'tube:ok', 'tube:ok', 'pd:ok', 'pd:ok', 'pd:ok', 'pd:ok', 'drive:damaged']);
  const status = id => row(id).querySelector('.registry-status').textContent;
  assert.deepEqual([status('beta_1'), status('beta_2'), status('alpha_2'), status('alpha_1')], ['80%', 'DISARMED', 'EMPTY', '90%']);
  assert.ok(row('beta_2').classList.contains('crippled') && row('alpha_2').classList.contains('spent') && !row('beta_1').classList.contains('crippled'));
  assert.match(row('beta_2').querySelector('.fit-strip').title, /Guns 0\/2 · PD 2\/2 · Drive 100%/);
  // Scrub back: same nodes, earlier state; forward to the drive kill
  const strip = row('beta_2').querySelector('.fit-strip');
  registry.update(states, 30, hud.kills, hud.fittings);
  assert.equal(row('beta_2').querySelector('.fit-strip'), strip);
  assert.deepEqual(pips('beta_2'), ['gun:ok', 'gun:ok', 'pd:ok', 'pd:ok', 'drive:ok']);
  assert.equal(status('beta_2'), '70%');
  registry.update(states, 121, hud.kills, hud.fittings);
  assert.equal(status('alpha_1'), 'ADRIFT');
  assert.equal(pips('alpha_1').at(-1), 'drive:destroyed');
  // A dying ship counts down instead
  assert.equal(registryStatus({ dying: true }, null, 121, fittingsAt(hud.fittings, 'alpha_1', 121)).level, 'dying');
});

test('telemetry FITTINGS card: rows per group, thrust, rounds and the flag', () => {
  const hud = hudOf();
  const doc = new FakeDocument();
  const root = build(doc, ['div', { class: 'fit-card hidden' }]);
  const card = new FittingsCard({ root, doc });
  card.update(null);
  assert.ok(root.classList.contains('hidden'));
  card.update(fittingsAt(hud.fittings, 'alpha_1', 121));
  assert.ok(!root.classList.contains('hidden'));
  const rows = allElements(root).filter(el => el.classList.contains('fit-row'));
  const read = r => [r.querySelector('.fit-label').textContent, r.hidden, r.querySelector('.fit-value').textContent];
  assert.deepEqual(rows.map(read), [['GUNS', true, ''], ['TUBES', false, '4/4 · 40 RDS'], ['PD', false, '4/4'], ['DRIVE', false, 'NO THRUST']]);
  assert.equal(root.querySelector('.fit-flag').textContent, 'ADRIFT');
  assert.equal(rows[1].querySelector('.fit-strip').children.length, 4, 'the tube row shows only tubes');
  card.update(fittingsAt(hud.fittings, 'beta_1', 121));
  assert.deepEqual(rows.map(read), [['GUNS', false, '4/5'], ['TUBES', true, '4/4 · 40 RDS'], ['PD', false, '2/3'], ['DRIVE', false, '100% THRUST']]);
  assert.equal(root.querySelector('.fit-flag').hidden, true);
  // Strip helper on its own: null hides it
  const strip = pipStrip(doc);
  strip.set(null);
  assert.equal(strip.el.hidden, true);
});

test('kill feed: the blast line on a destroyed ship, amber notice toasts, deterministic under scrubbing', () => {
  const hud = hudOf();
  const doc = new FakeDocument();
  const seeks = [];
  const feed = new KillFeed({ root: doc.createElement('div'), doc, onSeek: x => seeks.push([x.t, x.shipId]) });
  const texts = () => feed.root.children.map(el => [el.className.trim(), el.textContent]);
  feed.update(hud.kills, 101, hud);
  assert.deepEqual(texts().map(([cls]) => cls), ['kill-toast notice notice-cookoff beta', 'kill-toast notice notice-disarmed beta']);
  assert.match(texts()[0][1], /OCS Sovereign-1.*MAGAZINE COOK-OFF.*3825 rounds burned/);
  feed.root.children[1].click();
  assert.deepEqual(seeks, [[95, 'beta_2']]);
  feed.update(hud.kills, 106, hud);
  assert.match(texts()[0][1], /TIS Ward-1.*CRIPPLED/);
  assert.ok(!texts()[0][1].includes('BLAST'), 'no blast line while the hulk is still counting down');
  feed.update(hud.kills, 111, hud);
  assert.match(texts()[0][1], /DESTROYED.*BLAST 11 km · caught OCS Falchion-1 · 1 torpedo killed, 2 blinded/);
  assert.match(texts()[1][1], /OCS Falchion-1.*CAUGHT IN BLAST.*1 PD lost/);
  const at111 = JSON.stringify(texts());
  feed.update(hud.kills, 20, hud);
  assert.equal(feed.root.hidden, true);
  feed.update(hud.kills, 111, hud);
  assert.equal(JSON.stringify(texts()), at111);
});

test('timeline: disarmed / adrift / cook-off marks, a ring and blast summary on the kill, a flag for a failed directive', () => {
  const hud = hudOf();
  const marks = timelineMarks(hud, 150, ships);
  const notices = marks.filter(m => m.kind === 'notice');
  assert.deepEqual(notices.map(m => [m.t, m.flag, m.shipId]), [[95, 'disarmed', 'beta_2'], [100, 'cookoff', 'beta_1'], [120, 'adrift', 'alpha_1']]);
  assert.match(notices[0].title, /01:35 OCS Falchion-1 DISARMED — last gun knocked out/);
  const kill = marks.find(m => m.kind === 'kill');
  assert.equal(kill.flag, 'blast');
  assert.match(kill.title, /TIS Ward-1 knocked out.*BLAST 11 km · caught OCS Falchion-1/);
});

test('command net: a failed admiral directive is an issue row, a log line and a flagged checkpoint', () => {
  let seq = 0;
  const ev = (t, type, ship_id, data = {}) => ({ event_id: `d${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  const evs = [
    ev(30, 'decision_point_reached', null),
    ev(30, 'admiral_directive_failed', null, { admiral_name: 'Admiral A', faction: 'alpha', reason: 'response truncated at max_tokens' }),
    ev(30, 'admiral_directive', null, { admiral_name: 'Admiral B', faction: 'beta', directive: 'Hold the wall.' }),
    ev(60, 'decision_point_reached', null),
    ev(60, 'admiral_directive_retried', null, { admiral_name: 'Admiral A', faction: 'alpha', reason: 'second attempt succeeded' })
  ];
  const hud = buildHudIndex({ events: evs, ships, simTrace: [], metadata: {}, sourceData: {}, duration: 90 });
  const failed = hud.comms.find(r => r.tag === 'NO DIRECTIVE');
  assert.deepEqual([failed.kind, failed.faction, failed.from, failed.cp, failed.to], ['error', 'alpha', 'Admiral A', 1, 'ALL SHIPS']);
  assert.match(failed.text, /No new fleet directive this checkpoint.*response truncated at max_tokens.*previous directive/);
  assert.ok(commsMatches(failed, 'alpha', 'issues') && !commsMatches(failed, 'beta', 'issues'));
  assert.equal(hud.comms.find(r => r.tag === 'DIRECTIVE RETRIED').kind, 'system');
  assert.ok(hud.log.some(r => r.cat === 'command' && /^NO DIRECTIVE ALPHA: admiral's directive call failed \(response truncated/.test(r.text)));
  const cps = timelineMarks(hud, 90, ships).filter(m => m.kind === 'checkpoint');
  assert.deepEqual(cps.map(m => m.flag), ['deviation', null]);
  assert.match(cps[0].title, /admiral directive FAILED/);
});

test('battle end: a side that can no longer deal damage is named, with the fighting-power tally', () => {
  assert.deepEqual(endCause({ reason: 'Beta fleet combat-ineffective (no weapon left that can deal damage)' }),
    { side: 'beta', headline: 'BETA FLEET CAN NO LONGER FIGHT · no gun or torpedo left that can deal damage' });
  assert.equal(endCause({ reason: 'Both fleets combat-ineffective - on points: Alpha tactical advantage (60.0 vs 40.0)' }).side, 'both');
  assert.equal(endCause({ reason: 'Beta fleet eliminated' }), null);
  assert.equal(endCause(null), null);

  const hud = hudOf();
  assert.equal(hud.result.reason, 'Beta fleet combat-ineffective (no weapon left that can deal damage)');
  const doc = new FakeDocument();
  const root = build(doc, ['div', { class: 'result-banner hidden' }, ['button', { 'data-close': '' }], ['span', { 'data-kicker': '' }], ['strong', { 'data-title': '' }],
    ['span', { 'data-cause': '', hidden: '' }], ['span', { 'data-reason': '' }], ['div', { 'data-tally': '' }]]);
  const banner = new ResultBanner({ root, doc });
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { destroyed: true }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  banner.update(true, hud.result, computeSides(states, [], ships, hud, 150), 150);
  assert.ok(!root.classList.contains('hidden'));
  const cause = root.querySelector('[data-cause]');
  assert.deepEqual([cause.textContent, cause.hidden, cause.dataset.side], ['BETA FLEET CAN NO LONGER FIGHT · no gun or torpedo left that can deal damage', false, 'beta']);
  assert.match(root.querySelector('[data-tally]').textContent, /ALPHA 2\/3 ships .* 1 armed · guns 0\/1 · tubes 5\/5.*BETA 2\/2 ships .* 1 armed · guns 4\/7/);
  // Any other ending keeps the banner as it was
  banner.update(true, { winner: 'alpha', reason: 'Beta fleet eliminated', known: true }, computeSides(states, [], ships, { ...hud, fittings: { active: false } }, 150), 150);
  assert.equal(cause.hidden, true);
  assert.ok(!root.querySelector('[data-tally]').textContent.includes('armed'));
});

test('live append: fittings, notices and blasts match a full build at every poll boundary', () => {
  const all = events();
  const check = cuts => {
    const loader = loaderOf([]);
    let hud = buildHudIndex(loader);
    let prev = 0;
    for (const cut of [...cuts, all.length]) {
      const delta = all.slice(prev, cut);
      loader.events.push(...delta);
      hud = appendHudIndex(loader, { events: delta, frames: [] });
      const full = hudOf(all.slice(0, cut));
      assert.deepStrictEqual(hud.fittings, full.fittings, `fittings after ${cut} events`);
      assert.deepStrictEqual(hud.notices, full.notices, `notices after ${cut} events`);
      assert.deepStrictEqual(hud.blasts, full.blasts, `blasts after ${cut} events`);
      prev = cut;
    }
    return hud;
  };
  // One event per poll: splits every tick, including the blast's victims from its summary
  const hud = check(Array.from({ length: all.length - 1 }, (_, i) => i + 1));
  assert.equal(hud.fittings.active, true);
  check([5, 6, 17, 23, 24, 26]);
  // Before the first damage event the live HUD has no fitting data; it switches on when one arrives
  const early = buildHudIndex(loaderOf(all.slice(0, 10)));
  assert.equal(early.fittings.active, false);
});
