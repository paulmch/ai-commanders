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
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { TUBE_MAGAZINE, FittingFold } from '../src/FittingsIndex.js';

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
  // The torpedo cruiser is built on the cruiser's module layout: its gun-module slots are tubes there
  assert.deepEqual(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Spinal Coiler Mount').slots, ['weapon_0']);
  assert.deepEqual(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Coilgun Battery A').slots, ['weapon_1']);
  assert.deepEqual(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Coilgun Battery B').slots, ['weapon_2']);
  assert.deepEqual(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Torpedo Bay A').slots, []);
  assert.equal(slotsForModule('cruiser_torpedo', lay.alpha_1, 'Main Engine Assembly').engine, true);
  // A hull the module table does not know maps nothing but its drive
  assert.deepEqual(slotsForModule('mystery_boat', lay.alpha_1, 'Coilgun Battery A'), { slots: [], engine: false });
  assert.equal(slotsForModule('mystery_boat', lay.alpha_1, 'Main Engine Assembly').engine, true);
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
  assert.equal(fittingsFlag(at('alpha_2', 47)), 'NO AMMO');
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
  assert.deepEqual([status('beta_1'), status('beta_2'), status('alpha_2'), status('alpha_1')], ['80%', 'DISARMED', 'NO AMMO', '90%']);
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
    { side: 'beta', headline: 'BETA FLEET CAN NO LONGER FIGHT', detail: 'no gun or torpedo left that can deal damage' });
  assert.deepEqual(endCause({ reason: 'Both fleets combat-ineffective - on points: Alpha tactical advantage (60.0 vs 40.0)' }),
    { side: 'both', headline: 'NEITHER FLEET CAN STILL FIGHT', detail: 'on points: Alpha tactical advantage (60.0 vs 40.0)' });
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
  assert.deepEqual([cause.textContent, cause.hidden, cause.dataset.side], ['BETA FLEET CAN NO LONGER FIGHT', false, 'beta']);
  assert.match(root.querySelector('[data-tally]').textContent, /ALPHA 2\/3 ships .* 1 armed · guns 0\/1 · tubes 5\/5.*BETA 2\/2 ships .* 1 armed · guns 4\/7/);
  // The recorded reason is silent, but the beaten fleet's last ships are afloat and unarmed (fitting data)
  const stranded = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { hull: 100 }, beta_1: { destroyed: true }, beta_2: { hull: 70 } };
  const eliminated = { winner: 'alpha', reason: 'Beta fleet eliminated', known: true };
  assert.deepEqual(endCause(eliminated, computeSides(stranded, [], ships, hud, 150)),
    { side: 'beta', inferred: true, headline: 'BETA FLEET CAN NO LONGER FIGHT', detail: '1 ship afloat, none armed' });
  banner.update(true, eliminated, computeSides(stranded, [], ships, hud, 150), 150);
  assert.equal(cause.textContent, 'BETA FLEET CAN NO LONGER FIGHT · 1 ship afloat, none armed');
  assert.equal(endCause(eliminated, computeSides({ ...stranded, beta_2: { destroyed: true } }, [], ships, hud, 150)), null, 'a fleet with nothing afloat was eliminated');
  // ... and a time-limit verdict is never relabelled, whatever the fitting data says
  assert.equal(endCause({ winner: 'alpha', reason: 'Alpha tactical advantage (60.0 vs 40.0)' }, computeSides(stranded, [], ships, hud, 150)), null);
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

test('torpedo rounds per tube are the fleet data\'s magazines (data/fleet_ships.json)', { skip: !existsSync(fileURLToPath(new URL('../../data/fleet_ships.json', import.meta.url))) }, () => {
  const data = JSON.parse(readFileSync(fileURLToPath(new URL('../../data/fleet_ships.json', import.meta.url)), 'utf8'));
  const fallback = data.weapon_types.torpedo_launcher.magazine;
  const fromData = {};
  for (const [type, spec] of Object.entries(data.ships)) {
    const tubes = (spec.weapons || []).filter(x => x.type === 'torpedo_launcher');
    if (!tubes.length) continue;
    const mags = new Set(tubes.map(x => x.magazine ?? fallback));
    assert.equal(mags.size, 1, `${type}: every tube holds the same number of rounds`);
    fromData[type] = [...mags][0];
  }
  assert.deepEqual(TUBE_MAGAZINE, fromData, 'FittingsIndex.TUBE_MAGAZINE drifted from the fleet data');
});

test('fleet specs that arrive after the first events refold the fittings from the start', () => {
  const all = events();
  const loader = loaderOf([], {});
  buildHudIndex(loader);
  loader.events.push(...all.slice(0, 12));
  let hud = appendHudIndex(loader, { events: all.slice(0, 12), frames: [] });
  assert.equal(hud.fittings.active, false, 'no specs yet');
  loader.sourceData = sourceData();
  loader.events.push(...all.slice(12));
  hud = appendHudIndex(loader, { events: all.slice(12), frames: [] });
  const full = hudOf(all);
  assert.deepStrictEqual(hud.fittings, full.fittings);
  assert.deepStrictEqual(hud.notices, full.notices);
  assert.ok(new FittingFold().hasLayouts === false);
});

test('surrenders recorded only as a captain tool call reach the command net, the log, the tally and the end cause', () => {
  const evs = events().filter(e => e.event_type !== 'battle_end');
  let seq = 1000;
  const ev = (t, type, ship_id, data = {}) => evs.push({ event_id: `s${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  const quit = { tool_calls: [{ id: 'h1', name: 'surrender', arguments: { reason: 'all weapons out, no armed ship left to protect' } }] };
  ev(140, 'captain_decision', 'beta_2', quit);
  ev(145, 'ship_destroyed', 'beta_1', { cause: 'reactor_destroyed', killer_id: 'alpha_1' });
  ev(145, 'reactor_blast', 'beta_1', { radius_km: 18.4, ships_hit: [], torpedoes_destroyed: [], torpedoes_blinded: [] });
  ev(150, 'captain_decision', 'beta_2', quit);   // repeated: the first counts
  ev(150, 'battle_end', null, { winner: 'alpha', reason: 'Beta fleet eliminated' });
  const hud = hudOf(evs);
  assert.deepEqual(hud.surrenders, [{ t: 140, shipId: 'beta_2', name: 'OCS Falchion-1', faction: 'beta', reason: 'all weapons out, no armed ship left to protect' }]);
  const rows = hud.comms.filter(r => r.tag === 'SURRENDER');
  assert.deepEqual(rows.map(r => [r.t, r.kind, r.faction, r.shipId]), [[140, 'system', 'beta', 'beta_2'], [150, 'system', 'beta', 'beta_2']]);
  assert.match(rows[0].text, /OCS Falchion-1 surrenders: all weapons out/);
  assert.ok(hud.log.some(r => /^SURRENDER: OCS Falchion-1 strikes its colours \(all weapons out/.test(r.text)));
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { destroyed: true }, beta_1: { destroyed: true }, beta_2: { hull: 70 } };
  assert.equal(computeSides(states, [], ships, hud, 139).beta.surrendered, 0);
  const sides = computeSides(states, [], ships, hud, 150);
  assert.deepEqual([sides.beta.alive, sides.beta.surrendered, sides.beta.armed], [1, 1, 0]);
  assert.deepEqual(endCause(hud.result, sides), { side: 'beta', inferred: true, headline: 'BETA FLEET SURRENDERED',
    detail: '1 ship struck its colours, none able to deal damage' });
  // A recorded `surrender` event wins over the tool call at the same time (one row)
  const both = [...evs];
  both.splice(both.findIndex(e => e.timestamp === 140 && e.event_type === 'captain_decision'), 0,
    { event_id: 'sx', sequence: 999, timestamp: 140, event_type: 'surrender', ship_id: 'beta_2', data: { captain_name: 'Captain B' } });
  assert.equal(hudOf(both).comms.filter(r => r.tag === 'SURRENDER' && r.t === 140).length, 1);
  assert.equal(hudOf(both).surrenders.length, 1);
  // Live: the same list whatever the poll boundaries
  const loader = loaderOf([]);
  buildHudIndex(loader);
  let live;
  for (const e of evs) { loader.events.push(e); live = appendHudIndex(loader, { events: [e], frames: [] }); }
  assert.deepStrictEqual(live.surrenders, hud.surrenders);
  assert.deepStrictEqual(live.comms, hud.comms);
});

test('a blast that only grazes a ship stays on the kill toast: no victim notice', () => {
  const evs = events();
  const graze = evs.find(e => e.event_type === 'blast_damage');
  graze.data.armor_ablation_cm = 0.2;
  const hud = hudOf(evs.filter(e => !(e.event_type === 'mount_damaged' && e.data.cause === 'reactor_blast')));
  assert.ok(!hud.notices.some(n => n.kind === 'blast'));
  assert.equal(blastLine(hud.blasts.alpha_3), 'BLAST 11 km · caught OCS Falchion-1 · 1 torpedo killed, 2 blinded');
});

test('torpedo cruiser: a destroyed module silences its tube; engine module damage short of destruction cuts thrust', () => {
  let seq = 5000;
  const evs = [];
  const ev = (t, type, ship_id, data = {}) => evs.push({ event_id: `m${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  ev(10, 'mount_damaged', 'beta_1', { kind: 'pd', slot: 'pd_0', index: 0, health_pct: 80, destroyed: false, cause: 'surface_hit' });   // a pass-5 recording
  ev(20, 'module_destroyed', 'alpha_1', { module_name: 'Coilgun Battery A' });
  ev(30, 'module_damaged', 'alpha_1', { module_name: 'Main Engine Assembly', damage_gj: 2, destroyed: false, health_pct: 60 });
  ev(40, 'module_damaged', 'alpha_1', { module_name: 'Main Engine Assembly', damage_gj: 2, destroyed: false, health_pct: 20 });
  ev(50, 'module_damaged', 'beta_1', { module_name: 'Crew Quarters', damage_gj: 2, destroyed: false, health_pct: 10 });
  const hud = hudOf(evs);
  const at = (id, t) => fittingsAt(hud.fittings, id, t);
  assert.deepEqual(at('alpha_1', 25).tubes, ['ok', 'destroyed', 'ok', 'ok']);
  assert.deepEqual([at('alpha_1', 29).thrust, at('alpha_1', 30).thrust, at('alpha_1', 40).thrust, at('alpha_1', 40).adrift], [1, 0.6, 0, true],
    'health / 100 above 25 %, no thrust at or below it (engine_effectiveness)');
  assert.equal(at('beta_1', 60).thrust, 1, 'other modules do not touch the drive');
  assert.deepEqual(hud.notices.map(n => [n.t, n.kind, n.status]), [[30, 'drive', 'DRIVE HIT · 60% THRUST'], [40, 'adrift', 'ADRIFT']]);
  // Two engine modules (dreadnought): the mean, as long as one still works
  const src = sourceData();
  src.beta_fleet.ships[0].ship_type = 'dreadnought';
  const dn = [...evs];
  const push = (t, type, ship_id, data) => dn.push({ event_id: `m${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  push(60, 'module_destroyed', 'beta_1', { module_name: 'Main Engine Assembly' });
  const two = hudOf(dn, src);
  assert.equal(fittingsAt(two.fittings, 'beta_1', 60).thrust, 0.5);
  push(70, 'module_destroyed', 'beta_1', { module_name: 'Maneuvering Thrusters' });
  assert.equal(fittingsAt(hudOf(dn, src).fittings, 'beta_1', 70).adrift, true);
});

// Plume lane (docs/pass-5-recording-additions.md): the doc's JSON examples, on this fixture's ships
function plumeEvents() {
  let seq = 7000;
  const out = [];
  const ev = (t, type, ship_id, data) => out.push({ event_id: `p${++seq}`, sequence: seq, timestamp: t, event_type: type, ship_id, data });
  const wash = (t, victim, source, data) => ev(t, 'plume_wash', victim, { source_ship_id: source, friendly: victim.split('_')[0] === source.split('_')[0],
    emitter_throttle: 1.0, swept: false, target_id: null, ...data });
  wash(30, 'beta_2', 'beta_1', { phase: 'enter', distance_km: 10.99, off_axis_deg: 0.77, heat_rate_gj_s: 12.83, peak_heat_rate_gj_s: 12.83, duration_s: 1.0, heat_gj: 12.8,
    armor_ablation_cm: 0.26, fittings_hit: [], hit_location: 'nose', dose_sv: 0.085, crew_dose_sv: 0.085, crew_state: 'fit', armor_remaining_cm: 150.94, heat: 0.043 });
  ev(35, 'mount_damaged', 'beta_2', { kind: 'gun', slot: 'weapon_0', index: 0, weapon_name: 'Spinal Coiler Mk3', health_pct: 72.1, health_lost_pct: 27.9, destroyed: false,
    cooldown_factor: 1.39, cause: 'plume_wash', hit_location: 'nose', source_ship_id: 'beta_1' });
  ev(50, 'crew_radiation', 'beta_2', { state: 'impaired', dose_sv: 2.08, threshold_sv: 2.0, cooldown_factor: 1.5, source_ship_id: 'beta_1' });
  wash(55, 'beta_2', 'beta_1', { phase: 'sustain', distance_km: 7.57, off_axis_deg: 1.94, heat_rate_gj_s: 16.01, peak_heat_rate_gj_s: 16.42, duration_s: 26.0, heat_gj: 385.4,
    armor_ablation_cm: 8.44, fittings_hit: ['gun:weapon_0', 'gun:weapon_1', 'pd:pd_0', 'pd:pd_1'], hit_location: 'nose', dose_sv: 2.662, crew_dose_sv: 2.662, crew_state: 'impaired',
    armor_remaining_cm: 142.76, heat: 1.256 });
  wash(69, 'beta_2', 'beta_1', { phase: 'exit', distance_km: 3.69, off_axis_deg: 6.72, heat_rate_gj_s: 0.0, peak_heat_rate_gj_s: 16.42, emitter_throttle: 0.7, duration_s: 39.0,
    heat_gj: 463.7, armor_ablation_cm: 9.99, fittings_hit: ['gun:weapon_0', 'gun:weapon_1', 'pd:pd_0', 'pd:pd_1'], hit_location: 'nose', dose_sv: 3.326, crew_dose_sv: 3.326,
    crew_state: 'impaired', armor_remaining_cm: 141.21, heat: 1.488 });
  // A weak, swept episode (peak 0.6 GJ/s): logged, but no toast and no mark
  wash(97, 'alpha_1', 'alpha_2', { phase: 'enter', distance_km: 40, off_axis_deg: 4.0, heat_rate_gj_s: 0.5, peak_heat_rate_gj_s: 0.5, duration_s: 1.0, heat_gj: 0.5,
    armor_ablation_cm: 0, fittings_hit: [], hit_location: 'nose', dose_sv: 0.001, crew_dose_sv: 0.001, crew_state: 'fit', armor_remaining_cm: 252.84, heat: 0.3 });
  wash(154, 'alpha_1', 'alpha_2', { phase: 'exit', distance_km: 34.34, off_axis_deg: 4.35, heat_rate_gj_s: 0.0, peak_heat_rate_gj_s: 0.6, emitter_throttle: 0.4, duration_s: 57.0,
    heat_gj: 25.5, armor_ablation_cm: 0.0, fittings_hit: [], hit_location: 'nose', swept: true, dose_sv: 0.061, crew_dose_sv: 0.061, crew_state: 'fit', armor_remaining_cm: 252.84, heat: 0.384 });
  // A dying hulk in the jet is tracked but takes nothing: no notice
  wash(160, 'alpha_3', 'beta_1', { phase: 'enter', distance_km: 5, off_axis_deg: 1, heat_rate_gj_s: 30, peak_heat_rate_gj_s: 30, duration_s: 1.0, heat_gj: 0,
    armor_ablation_cm: 0, fittings_hit: [], hit_location: null, dose_sv: 0 });
  ev(170, 'crew_radiation', 'beta_2', { state: 'incapacitated', dose_sv: 8.03, threshold_sv: 8.0, cooldown_factor: null, source_ship_id: 'alpha_1' });
  return out;
}

test('plume wash and crew radiation: notices, crew state, flags, log lines and timeline marks', () => {
  const hud = hudOf(plumeEvents());
  assert.equal(hud.fittings.active, true, 'plume events alone switch the fitting HUD on');
  assert.deepEqual(hud.notices.map(n => [n.t, n.kind, n.shipId, n.status]), [
    [30, 'plume', 'beta_2', 'IN EXHAUST'],
    [50, 'crew', 'beta_2', 'CREW IMPAIRED'],
    [55, 'plume-update', 'beta_2', 'IN EXHAUST · 26 s'],
    [69, 'plume-update', 'beta_2', 'OUT OF EXHAUST · 39 s'],
    [170, 'crew', 'beta_2', 'CREW INCAPACITATED']
  ]);
  const detail = t => hud.notices.find(n => n.t === t).detail;
  assert.equal(detail(30), 'of OCS Sovereign-1 (friendly) · 11.0 km astern · +13 GJ, −0.3 cm nose armour, 0.1 Sv');
  assert.equal(detail(50), "2.1 Sv in OCS Sovereign-1's exhaust · guns and tubes cycle 1.5x slower");
  assert.equal(detail(55), 'of OCS Sovereign-1 (friendly) · +385 GJ, −8.4 cm nose armour, 4 fittings burned, 2.7 Sv');
  assert.equal(detail(69), 'of OCS Sovereign-1 (friendly) · +464 GJ, −10.0 cm nose armour, 4 fittings burned, 3.3 Sv');
  assert.equal(detail(170), "8.0 Sv in TIS Harpoon-1's exhaust · the ship stops fighting and coasts");
  // Crew state and dose at time t, forwards and backwards
  const at = t => fittingsAt(hud.fittings, 'beta_2', t);
  assert.deepEqual([at(29).crew, at(29).dose], ['fit', 0]);
  assert.deepEqual([at(31).crew, at(31).dose], ['fit', 0.085]);
  assert.deepEqual(at(35).guns, ['damaged', 'ok'], 'a fitting burned by the wash is an ordinary mount_damaged');
  assert.deepEqual([at(50).crew, at(50).dose, fittingsFlag(at(50)), at(50).armed], ['impaired', 2.08, 'IMPAIRED', true]);
  assert.equal(at(69).dose, 3.326);
  assert.deepEqual([at(170).crew, fittingsFlag(at(170)), at(170).armed], ['incapacitated', 'CREW OUT', false]);
  assert.match(fittingsText(at(60)), /Crew impaired \(2\.7 Sv\)/);
  assert.deepEqual(registryStatus({ hull: 90 }, null, 60, at(60)), { text: 'IMPAIRED', level: 'impaired' });
  assert.deepEqual(registryStatus({ hull: 90 }, null, 170, at(170)), { text: 'CREW OUT', level: 'crippled' });
  // An incapacitated crew takes the ship out of the armed count
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { hull: 100 }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  assert.deepEqual([computeSides(states, [], ships, hud, 169).beta.armed, computeSides(states, [], ships, hud, 170).beta.armed], [2, 1]);
  // Feed: within the window the running update replaces the entry toast (one toast per ship's episode)
  assert.deepEqual(noticesAt(hud.notices, 31).map(n => n.status), ['IN EXHAUST']);
  assert.deepEqual(noticesAt(hud.notices, 56).map(n => n.status), ['IN EXHAUST · 26 s', 'CREW IMPAIRED']);
  // Log
  const log = hud.log.map(r => r.text);
  assert.ok(log.includes("EXHAUST: OCS Falchion-1 enters OCS Sovereign-1's drive plume (friendly) at 11.0 km (12.8 GJ/s)"));
  assert.ok(log.includes("EXHAUST: OCS Falchion-1 clears OCS Sovereign-1's drive plume (friendly) after 39 s (+463.7 GJ, -10.0 cm nose armour, 4 fittings burned)"));
  assert.ok(log.includes("RADIATION: OCS Falchion-1 crew impaired - weapons cycle slower (2.1 Sv, OCS Sovereign-1's exhaust)"));
  assert.ok(log.some(t => /^RADIATION: OCS Falchion-1 crew incapacitated - the ship stops fighting/.test(t)));
  assert.equal(log.filter(t => t.startsWith('EXHAUST:')).length, 4, 'enter and exit of each episode on a living ship; sustain and hulks stay out');
  // Timeline: one mark per strong episode (its entry) and one per crew threshold
  const marks = timelineMarks(hud, 180, ships).filter(m => m.kind === 'notice');
  assert.deepEqual(marks.map(m => [m.t, m.flag]), [[30, 'plume'], [50, 'crew'], [170, 'crew']]);
  assert.match(marks[0].title, /00:30 OCS Falchion-1 IN EXHAUST — of OCS Sovereign-1 \(friendly\)/);

  // Card: the crew line appears with the first dose
  const doc = new FakeDocument();
  const root = build(doc, ['div', { class: 'fit-card hidden' }]);
  const card = new FittingsCard({ root, doc });
  card.update(at(10));
  assert.equal(root.querySelector('.fit-crew').hidden, true);
  card.update(at(60));
  assert.deepEqual([root.querySelector('.fit-crew').hidden, root.querySelector('.fit-crew').querySelector('.fit-value').textContent], [false, 'IMPAIRED · 2.7 Sv']);
  assert.equal(root.querySelector('.fit-flag').textContent, 'IMPAIRED');
  // Feed DOM
  const feed = new KillFeed({ root: doc.createElement('div'), doc });
  feed.update(hud.kills, 56, hud);
  assert.deepEqual(feed.root.children.map(el => el.className.trim()), ['kill-toast notice notice-plume-update beta', 'kill-toast notice notice-crew beta']);
});

test('plume events stream like everything else, and recordings without them are untouched', () => {
  const all = [...events(), ...plumeEvents()].sort((a, b) => a.timestamp - b.timestamp);
  const loader = loaderOf([]);
  buildHudIndex(loader);
  let hud;
  for (let i = 0; i < all.length; i++) {
    loader.events.push(all[i]);
    hud = appendHudIndex(loader, { events: [all[i]], frames: [] });
    if (i % 7 === 0 || i === all.length - 1) {
      const full = hudOf(all.slice(0, i + 1));
      assert.deepStrictEqual(hud.fittings, full.fittings, `fittings after ${i + 1} events`);
      assert.deepStrictEqual(hud.notices, full.notices, `notices after ${i + 1} events`);
    }
  }
  const plain = hudOf();
  assert.ok(!plain.notices.some(n => n.kind.startsWith('plume') || n.kind === 'crew'));
  assert.ok(Object.values(plain.fittings.states).every(list => list.every(st => st.crew === 'fit' && st.dose === 0)));
});

test('recorded fleet surrenders and the truthful end reasons (pass-5 engine) are used as recorded', () => {
  // The doc's example: three corvettes surrender at 150 s, each with its event and its tool call
  const evs = events().filter(e => e.event_type !== 'battle_end');
  let seq = 9500;
  const reason = 'all weapons out, no armed ship left to protect';
  for (const id of ['beta_1', 'beta_2']) {
    evs.push({ event_id: `q${++seq}`, sequence: seq, timestamp: 150, event_type: 'captain_decision', ship_id: id, data: { tool_calls: [{ name: 'surrender', arguments: { reason } }] } });
    evs.push({ event_id: `q${++seq}`, sequence: seq, timestamp: 150, event_type: 'surrender', ship_id: id,
      data: { captain_name: 'Captain X', faction: 'beta', reason, hull_pct: 100.0, could_deal_damage: false } });
  }
  evs.push({ event_id: 'qend', sequence: 9999, timestamp: 150, event_type: 'battle_end', ship_id: null, data: { winner: 'alpha', reason: 'Beta fleet surrendered (2 ships struck their colours)' } });
  const hud = hudOf(evs);
  assert.deepEqual(hud.surrenders.map(s => [s.t, s.shipId, s.reason]), [[150, 'beta_1', reason], [150, 'beta_2', reason]]);
  const rows = hud.comms.filter(r => r.tag === 'SURRENDER');
  assert.deepEqual(rows.map(r => r.shipId), ['beta_1', 'beta_2'], 'one row per ship: the event, not event + tool call');
  assert.match(rows[0].text, /OCS Sovereign-1 surrenders: all weapons out/);
  assert.equal(hud.log.filter(r => r.text.startsWith('SURRENDER:')).length, 2, 'one log line per ship');
  assert.ok(hud.log.every(r => !('surrender' in r)));
  const states = { alpha_1: { hull: 90 }, alpha_2: { hull: 100 }, alpha_3: { destroyed: true }, beta_1: { hull: 80 }, beta_2: { hull: 70 } };
  const sides = computeSides(states, [], ships, hud, 150);
  assert.deepEqual([sides.beta.alive, sides.beta.surrendered, sides.beta.armed], [2, 2, 0]);
  assert.deepEqual(endCause(hud.result, sides), { side: 'beta', headline: 'BETA FLEET SURRENDERED', detail: '2 ships struck their colours', note: 'none able to deal damage' });
  assert.deepEqual(endCause({ winner: 'alpha', reason: 'Beta fleet destroyed or surrendered (5 destroyed, 3 surrendered)' }),
    { side: 'beta', headline: 'BETA FLEET SURRENDERED', detail: '3 ships struck their colours, 5 destroyed', note: '' });
  assert.equal(endCause({ winner: 'beta', reason: 'Alpha fleet surrendered (1 ship struck its colours)' }).detail, '1 ship struck its colours');
  assert.equal(endCause({ winner: null, reason: 'Mutual destruction/surrender: Alpha fleet eliminated; Beta fleet eliminated' }), null);
  const doc = new FakeDocument();
  const root = build(doc, ['div', { class: 'result-banner hidden' }, ['button', { 'data-close': '' }], ['span', { 'data-kicker': '' }], ['strong', { 'data-title': '' }],
    ['span', { 'data-cause': '', hidden: '' }], ['span', { 'data-reason': '' }], ['div', { 'data-tally': '' }]]);
  new ResultBanner({ root, doc }).update(true, hud.result, sides, 150);
  assert.equal(root.querySelector('[data-cause]').textContent, 'BETA FLEET SURRENDERED · none able to deal damage');
  assert.equal(root.querySelector('[data-reason]').textContent, 'Beta fleet surrendered (2 ships struck their colours)');
  // Streamed: same rows and log
  const loader = loaderOf([]);
  buildHudIndex(loader);
  let live;
  for (const e of evs) { loader.events.push(e); live = appendHudIndex(loader, { events: [e], frames: [] }); }
  assert.deepStrictEqual(live.log, hud.log);
  assert.deepStrictEqual(live.surrenders, hud.surrenders);
});

test('a recorded ship_surrendered event counts like a surrender', () => {
  const evs = events().filter(e => e.event_type !== 'battle_end');
  evs.push({ event_id: 'ss1', sequence: 9001, timestamp: 140, event_type: 'ship_surrendered', ship_id: 'beta_2', data: { reason: 'all weapons out' } });
  evs.push({ event_id: 'ss2', sequence: 9002, timestamp: 140, event_type: 'captain_decision', ship_id: 'beta_2', data: { tool_calls: [{ name: 'surrender', arguments: { reason: 'all weapons out' } }] } });
  const hud = hudOf(evs);
  assert.deepEqual(hud.surrenders.map(s => [s.t, s.shipId, s.reason]), [[140, 'beta_2', 'all weapons out']]);
  assert.equal(hud.comms.filter(r => r.tag === 'SURRENDER').length, 1, 'the event row wins over the tool call');
});
