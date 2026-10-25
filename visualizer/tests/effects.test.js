import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  buildDamageTimeline, impactPreset, impactScale, buildWrecks, wrecksAt, armorStateAt,
  detonationTime, reactorPulse, reactorPulseRate, lastHitAt, GRAZE_DEG
} from '../src/DamageTimeline.js';
import { strikeSpallDir, DamageFX } from '../src/render/DamageFX.js';
import { shipStateAt, localBasis } from '../src/DamageTimeline.js';
import { blastKinds } from '../src/render/Explosion.js';
import { WeaponTimeline, DUD_KM } from '../src/render/WeaponTimeline.js';
import { BattleLoader } from '../src/BattleLoader.js';
import { selectEngagements, ENGAGE_HOT_S, SHOT_MEMORY_S } from '../src/render/TargetingLines.js';
import { factionSeed } from '../src/render/WeaponBatches.js';
import { threatLabel, detonationLabel, hitFlash, THREAT_IMMINENT_S } from '../src/ContactOverlay.js';
import { torchCough } from '../src/render/Torch.js';
import { wearSectionWeights } from '../src/render/HullWear.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

// --------------------------------------------------------------------------
// Strike presets: kinetic torpedoes are not warheads
// --------------------------------------------------------------------------
test('the warhead fireball is chosen only for a recorded explosive yield', () => {
  assert.equal(impactPreset({ kind: 'torpedo', explosive: 0, penetrated: false }), 'kineticStrike');
  assert.equal(impactPreset({ kind: 'torpedo', explosive: 0, penetrated: true }), 'breach');
  assert.equal(impactPreset({ kind: 'torpedo', explosive: 12, penetrated: false }), 'warhead');
  assert.equal(impactPreset({ kind: 'torpedo', explosive: 12, penetrated: true }), 'warhead');
  assert.equal(impactPreset({ kind: 'slug', penetrated: true, angle: 80 }), 'pierce');
  assert.equal(impactPreset({ kind: 'slug', penetrated: false, angle: GRAZE_DEG + 1 }), 'graze');
  assert.equal(impactPreset({ kind: 'slug', penetrated: false, angle: 10 }), 'impact');
  assert.equal(impactPreset({ kind: 'pop' }), null);
  // every preset exists and fits the respawn window DamageFX replays
  const kinds = blastKinds();
  for (const k of ['kineticStrike', 'breach', 'pierce', 'graze', 'impact', 'warhead']) {
    assert.ok(kinds[k], k);
    assert.ok(kinds[k].duration <= 6, `${k} outlives the strike window`);
  }
  // a stopped penetrator has no fireball and no glass-bubble shell
  assert.equal(kinds.kineticStrike.fireR, 0);
  assert.equal(kinds.kineticStrike.shellR, 0);
});

test('kinetic strikes are capped to the struck hull; warheads keep their size', () => {
  const strike = { kind: 'torpedo', explosive: 0, penetrated: false, energy: 24 };
  const corvetteW = 1.04, dreadW = 5.6;
  assert.ok(impactScale(strike, corvetteW) <= 0.35 * corvetteW + 1e-9);
  assert.ok(impactScale(strike, dreadW) > impactScale(strike, corvetteW));
  const legacy = { kind: 'torpedo', explosive: 24, penetrated: false, energy: 24 };
  assert.ok(impactScale(legacy, corvetteW) >= 1.4);
  const slug = { kind: 'slug', penetrated: false, angle: 5, energy: 3 };
  assert.ok(impactScale(slug, corvetteW) <= 0.3 * corvetteW + 1e-9);
});

// --------------------------------------------------------------------------
// Damage timeline: torpedo breach detection, ordnance lookup, legacy warheads
// --------------------------------------------------------------------------
function fleetRecording() {
  const trace = [];
  for (let t = 0; t <= 40; t++) {
    trace.push({
      t,
      ships: {
        alpha_1: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], hull: 100, armor: { nose: 200, lateral: 40, tail: 40 } },
        beta_1: {
          pos: [100000, 0, 0], vel: [-500, 0, 0], fwd: [-1, 0, 0], hull: t < 20 ? 100 : 60,
          armor: { nose: 200, lateral: t < 10 ? 40 : 10, tail: 40 },
          dying: t >= 25 && t < 33, destroyed: t >= 33
        }
      },
      torpedoes: t <= 5 ? [{ id: 'torp_a', pos: [90000 - t * 1000, 0, 0], target: 'beta_1', source: 'alpha_1' }] : []
    });
  }
  const events = [
    { timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha_1', data: { torpedo_id: 'torp_a', target_id: 'beta_1', warhead_gj: 0 } },
    { timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha_1', data: { torpedo_id: 'torp_w', target_id: 'beta_1', warhead_gj: 20 } },
    // v3: the breach is named by the round's id
    { timestamp: 6, event_type: 'torpedo_impact', ship_id: 'alpha_1', data: { torpedo_id: 'torp_a', target_id: 'beta_1', total_damage_gj: 24, kinetic_damage_gj: 24, explosive_damage_gj: 0, hit_location: 'nose' } },
    { timestamp: 6, event_type: 'armor_damage', ship_id: 'beta_1', data: { torpedo_id: 'torp_a', damage_gj: 24, penetrated: false, source: 'kinetic' } },
    { timestamp: 6, event_type: 'torpedo_impact', ship_id: 'alpha_1', data: { torpedo_id: 'torp_b', target_id: 'beta_1', total_damage_gj: 30, kinetic_damage_gj: 30, explosive_damage_gj: 0, hit_location: 'nose' } },
    { timestamp: 6, event_type: 'damage_taken', ship_id: 'beta_1', data: { torpedo_id: 'torp_b', damage_gj: 30 } },
    { timestamp: 7, event_type: 'torpedo_impact', ship_id: 'alpha_1', data: { torpedo_id: 'torp_c', target_id: 'beta_1', total_damage_gj: 30, hit_location: 'lateral' } },
    { timestamp: 7, event_type: 'penetration', ship_id: 'beta_1', data: { ordnance_id: 'torp_c', location: 'lateral' } },
    // legacy: no explosive split, launched with a warhead
    { timestamp: 8, event_type: 'torpedo_impact', ship_id: 'alpha_1', data: { torpedo_id: 'torp_w', target_id: 'beta_1', total_damage_gj: 20, hit_location: 'nose' } },
    { timestamp: 9, event_type: 'hit', ship_id: 'beta_1', data: { source_ship_id: 'alpha_1', projectile_id: 'p1', hit_location: 'lateral', kinetic_energy_gj: 3, impact_angle_deg: 70, penetrated: false } },
    { timestamp: 25, event_type: 'ship_dying', ship_id: 'beta_1', data: { detonation_in_s: 12 } }
  ];
  return { events, sim_trace: trace };
}

test('torpedo marks read their breach from the round\'s own events and carry the strike preset', () => {
  const tl = buildDamageTimeline(fleetRecording());
  const a = tl.byOrdnance.get('torp_a'), b = tl.byOrdnance.get('torp_b');
  const c = tl.byOrdnance.get('torp_c'), w = tl.byOrdnance.get('torp_w');
  assert.equal(a.penetrated, false);
  assert.equal(a.strike, 'kineticStrike');
  assert.equal(b.penetrated, true, 'hull damage straight from the round: armour already gone');
  assert.equal(b.strike, 'breach');
  assert.equal(c.penetrated, true, 'penetration event naming the round');
  assert.equal(w.explosive, 20, 'legacy warhead round');
  assert.equal(w.strike, 'warhead');
  const slug = tl.byOrdnance.get('p1');
  assert.equal(slug.kind, 'slug');
  assert.equal(slug.strike, 'graze');
  // v3 hits name the shooter as source_ship_id: the aim comes from its bearing
  assert.ok(Math.abs(slug.approach[0]) + Math.abs(slug.approach[1]) + Math.abs(slug.approach[2]) > 0);
});

test('armour fractions per section, countdown and last hit are pure functions of time', () => {
  const tl = buildDamageTimeline(fleetRecording());
  const a = armorStateAt(tl, 'beta_1', 5);
  assert.deepEqual([a.nose, a.lateral, a.tail], [1, 1, 1]);
  const b = armorStateAt(tl, 'beta_1', 9.5);
  near(b.lateral, (40 + (10 - 40) * 0.5) / 40);
  near(armorStateAt(tl, 'beta_1', 30).lateral, 0.25);
  // a ship the trace never gives armour stays pristine
  assert.equal(armorStateAt(tl, 'nobody', 30).lateral, 1);

  const sd = tl.ships.get('beta_1');
  assert.equal(detonationTime(sd), 33, 'replay: the recorded death');
  assert.equal(detonationTime({ ...sd, deathTime: null }), 37, 'live: the ship_dying fuse');
  assert.equal(detonationTime(tl.ships.get('alpha_1')), null);

  assert.equal(lastHitAt(sd, 5.9), -Infinity);
  assert.equal(lastHitAt(sd, 6.5), 6);
  assert.ok(lastHitAt(sd, 12) >= 9);
});

test('the reactor pulse quickens toward detonation and is scrub-exact', () => {
  assert.ok(reactorPulseRate(0) > reactorPulseRate(5));
  assert.ok(reactorPulseRate(5) > reactorPulseRate(25));
  assert.equal(reactorPulse(-1), 0);
  assert.equal(reactorPulse(Infinity), 0);
  for (const t of [0, 0.3, 4, 19.7]) {
    const v = reactorPulse(t);
    assert.ok(v >= 0 && v <= 1, `${v}`);
    assert.equal(reactorPulse(t), v);
  }
  // brighter on average near the end
  const mean = (a, b) => { let s = 0; for (let i = 0; i < 400; i++) s += reactorPulse(a + (b - a) * i / 400); return s / 400; };
  assert.ok(mean(0, 2) > mean(20, 22));
});

// DamageFX without a renderer: just enough state to resolve marks on a hull
// and collect the blasts it would spawn
function boxShip(L, W) {
  const g = new THREE.Group();
  const geom = new THREE.BoxGeometry(W, W, L).toNonIndexed();
  g.userData = { size: { length: L, width: W }, parts: [{ geom, center: new THREE.Vector3(), mat: 'hull' }] };
  return g;
}
function stubFx(rec, groups) {
  const blasts = [];
  const scene = new THREE.Scene();
  const sm = { ships: groups, scene, SCALE: 0.001, _spawnBlast: (o) => { blasts.push(o); return o; } };
  const fx = Object.create(DamageFX.prototype);
  Object.assign(fx, {
    sm, scene, SCALE: 0.001, views: new Map(), shipIds: [], blastSpawned: new Set(),
    planeGeom: new THREE.PlaneGeometry(1, 1), coneGeom: new THREE.CylinderGeometry(1, 0.16, 1, 8, 1, true),
    craterMat: new THREE.MeshBasicMaterial(), ventMat: new THREE.MeshBasicMaterial(),
    raycaster: new THREE.Raycaster(), probe: new THREE.Mesh(undefined, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })),
    _cands: [], _hits: [], _st: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false },
    timeline: buildDamageTimeline(rec)
  });
  fx.probe.updateMatrixWorld();
  return { fx, blasts };
}

test('strike blasts sit on the drawn hull at the crater and ride with the struck ship', () => {
  const L = 8, W = 2;
  const rec = fleetRecording();
  const { fx, blasts } = stubFx(rec, new Map([['beta_1', boxShip(L, W)], ['alpha_1', boxShip(L, W)]]));
  assert.equal(fx.ownsImpacts(), true);
  fx._updateBlasts(9.5);
  const strikes = blasts.filter(b => ['kineticStrike', 'breach', 'warhead', 'graze'].includes(b.kind));
  assert.deepEqual(strikes.map(b => b.kind).sort(), ['breach', 'breach', 'graze', 'kineticStrike', 'warhead']);
  const local = new THREE.Vector3();
  for (const b of strikes) {
    const st = shipStateAt(rec.sim_trace, 'beta_1', b.spawnTime);
    const B = localBasis(st.fwd);
    local.copy(b.position).sub(new THREE.Vector3(...st.pos).multiplyScalar(0.001));
    const lx = local.x * B.x[0] + local.y * B.x[1] + local.z * B.x[2];
    const ly = local.x * B.y[0] + local.y * B.y[1] + local.z * B.y[2];
    const lz = local.x * B.z[0] + local.y * B.z[1] + local.z * B.z[2];
    // distance from the box skin (the displayed hull), not the recorded impact point
    const skin = Math.max(Math.abs(lx) - W / 2, Math.abs(ly) - W / 2, Math.abs(lz) - L / 2);
    assert.ok(Math.abs(skin) < 0.2 * W, `${b.kind} ${skin} off the hull`);
    near(b.baseVel.x, -0.5);                       // carried with the ship (km/s)
    near(b.dir.length(), 1);
    assert.ok(b.scale <= 0.35 * W + 1e-9 || b.kind === 'warhead');
  }
  // a second pass inside the window spawns nothing new
  const n = blasts.length;
  fx._updateBlasts(9.6);
  assert.equal(blasts.length, n);
  // the instant flash finds the same crater
  const P = new THREE.Vector3(), N = new THREE.Vector3();
  const m = fx.surfaceFor('torp_a', P, N);
  assert.equal(m.ordnanceId, 'torp_a');
  near(N.length(), 1);
  assert.equal(fx.surfaceFor('nope', P, N), null);
});

// --------------------------------------------------------------------------
// Wrecked mounts
// --------------------------------------------------------------------------
test('a destroyed battery maps onto the gun that never fires again', () => {
  const shots = new Map([['s', new Map([['weapon_1', [10, 20, 30]], ['weapon_2', [12, 22, 50]], ['weapon_3', [5]]])]]);
  const events = [
    { timestamp: 31, event_type: 'module_destroyed', ship_id: 's', data: { module_name: 'Coilgun Battery' } },
    { timestamp: 40, event_type: 'module_destroyed', ship_id: 's', data: { module_name: 'Coilgun Battery' } },
    { timestamp: 41, event_type: 'module_destroyed', ship_id: 's', data: { module_name: 'Spinal Coiler Mount' } },
    { timestamp: 42, event_type: 'radiator_damaged', ship_id: 's', data: { radiator_position: 'tail_port', destroyed: true } },
    { timestamp: 43, event_type: 'module_destroyed', ship_id: 's', data: { module_name: 'Crew Quarters' } }
  ];
  const w = buildWrecks(events, shots).get('s');
  assert.deepEqual(w.map(x => [x.kind, x.slot ?? x.side ?? null]), [
    ['gun', 'weapon_1'],        // fired until 30, silent after
    ['gun', 'weapon_3'],        // weapon_2 still fires at 50
    ['spinal', null],
    ['radiator', 'port']
  ]);
  assert.equal(w[3].destroyed, true);
  assert.equal(wrecksAt({ wrecks: w }, 40.5).length, 2);
  assert.equal(wrecksAt({ wrecks: w }, 0).length, 0);
});

// --------------------------------------------------------------------------
// Spall direction
// --------------------------------------------------------------------------
test('spall leaves a breach along the normal and skids along the plating on a graze', () => {
  const N = new THREE.Vector3(0, 1, 0);
  // attacker straight ahead of a ship facing +z, round arriving 60 deg off the normal
  const base = { shipFwd: [0, 0, 1], approach: [0, Math.cos(Math.PI / 3), Math.sin(Math.PI / 3)] };
  const out = new THREE.Vector3();
  strikeSpallDir({ ...base, strike: 'breach' }, N, out);
  near(out.y, 1);
  strikeSpallDir({ ...base, strike: 'graze' }, N, out);
  assert.ok(out.z < -0.8 && out.y > 0, `graze ${out.toArray()}`);   // along the travel (-z), off the skin
  strikeSpallDir({ ...base, strike: 'impact' }, N, out);
  near(out.length(), 1);
  assert.ok(out.y > 0.5 && out.z < 0, `impact cone between normal and mirror ${out.toArray()}`);
});

// --------------------------------------------------------------------------
// Weapon timeline: threats, duds, fire index
// --------------------------------------------------------------------------
function threatLoader() {
  const loader = new BattleLoader();
  const ship = (x, vx) => ({ pos: [x, 0, 0], vel: [vx, 0, 0], fwd: [1, 0, 0] });
  loader.simTrace = [];
  for (let t = 0; t <= 6; t++) {
    loader.simTrace.push({
      t,
      ships: { alpha: ship(0, 0), beta: ship(100000, 0) },
      torpedoes: [
        { id: 't1', pos: [10000 * t, 0, 0], vel: [10000, 0, 0], source: 'alpha', target: 'beta', dv_remaining: 5 },
        { id: 't2', pos: [5000 * t, 0, 0], vel: [5000, 0, 0], source: 'alpha', target: 'beta', dv_remaining: 5, disabled: t >= 3 }
      ]
    });
  }
  loader.events = [
    { timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha', data: { torpedo_id: 't1', target_id: 'beta' } },
    { timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha', data: { torpedo_id: 't2', target_id: 'beta' } },
    { timestamp: 1, event_type: 'shot_fired', ship_id: 'beta', data: { target_id: 'alpha', weapon_slot: 'weapon_0' } },
    { timestamp: 3, event_type: 'pd_torpedo_disabled', ship_id: 'beta', data: { torpedo_id: 't2' } },
    { timestamp: 6, event_type: 'torpedo_miss', ship_id: 'alpha', data: { torpedo_id: 't1', target_id: 'beta', closest_approach_km: DUD_KM / 10, hit_probability: 0.98 } }
  ];
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates(); loader.buildTorpedoTimeline();
  return loader;
}

test('incoming torpedoes: count, soonest closing ETA and seeker-blinded rounds', () => {
  const tl = new WeaponTimeline(threatLoader());
  const at2 = tl.threatsAt(2).get('beta');
  assert.equal(at2.count, 2);
  near(at2.eta, (100000 - 20000) / 10000, 1e-6);
  near(at2.rangeKm, 80);
  const at4 = tl.threatsAt(4).get('beta');
  assert.equal(at4.count, 1);
  assert.equal(at4.blind, 1);
  assert.equal(tl.threatsAt(2).get('alpha'), undefined);
  // reuse of the output map does not leak entries
  const out = new Map([['stale', {}]]);
  tl.threatsAt(2, out);
  assert.ok(!out.has('stale'));
  assert.equal(threatLabel(at2), '2 TORP · 8s');
  assert.equal(threatLabel({ count: 0, blind: 2 }), '');
  assert.ok(at2.eta < THREAT_IMMINENT_S);
});

test('a hit-roll failure inside the hull is a dud moment; shots and launches are indexed', () => {
  const tl = new WeaponTimeline(threatLoader());
  const dud = tl.moments.filter(m => m.kind === 'dud');
  assert.equal(dud.length, 1);
  assert.equal(dud[0].target, 'beta');
  assert.equal(dud[0].probability, 0.98);
  assert.equal(tl.lastFire('beta', 0.5), null);
  assert.equal(tl.lastFire('beta', 1).target, 'alpha');
  assert.equal(tl.lastFire('alpha', 10).torpedo, true);
});

// --------------------------------------------------------------------------
// Engagement lines
// --------------------------------------------------------------------------
test('engagement lines: fire-control target first, then recent fire, then orders', () => {
  const ships = [
    { id: 'a1', faction: 'alpha', alive: true, tgt: 'b1' },
    { id: 'a2', faction: 'alpha', alive: true, tgt: null },
    { id: 'a3', faction: 'alpha', alive: true, tgt: null },
    { id: 'a4', faction: 'alpha', alive: true, tgt: 'a1' },      // own side: dropped
    { id: 'a5', faction: 'alpha', alive: false, tgt: 'b1' },     // dying shooter: dropped
    { id: 'b1', faction: 'beta', alive: true, tgt: 'a9' },       // dead target: dropped
    { id: 'b2', faction: 'beta', alive: true, tgt: null }
  ];
  const fire = {
    a1: { t: 99.5, target: 'b1', torpedo: false },
    a2: { t: 100 - SHOT_MEMORY_S - 1, target: 'b1', torpedo: false },   // too old
    a3: { t: 50, target: 'b2', torpedo: true }                          // a launch is remembered longer
  };
  const pairs = selectEngagements(ships, id => fire[id] || null, 100, id => (id === 'a2' || id === 'b2' ? (id === 'a2' ? 'b2' : 'a1') : null));
  const byFrom = Object.fromEntries(pairs.map(p => [p.from, p]));
  assert.deepEqual(Object.keys(byFrom).sort(), ['a1', 'a2', 'a3', 'b2']);
  assert.equal(byFrom.a1.to, 'b1');
  near(byFrom.a1.heat, (1 - 0.5 / ENGAGE_HOT_S) ** 2);
  assert.equal(byFrom.a2.to, 'b2', 'orders when the last shot is stale');
  assert.equal(byFrom.a2.heat, 0);
  assert.equal(byFrom.a3.to, 'b2');
  assert.equal(byFrom.b2.to, 'a1');
});

test('faction folds into a ribbon seed without losing the seed', () => {
  for (const [f, code] of [['alpha', 1], ['beta', 2], [undefined, 0]]) {
    const w = factionSeed(0.375, f);
    assert.equal(Math.floor(w * 0.5 + 0.001), code);
    near(w - 2 * code, 0.375);
  }
  near(factionSeed(1.25, 'beta') - 4, 0.25);
});

// --------------------------------------------------------------------------
// HUD helpers, torch cough, wear sections
// --------------------------------------------------------------------------
test('countdown, hit flash and torch cough are deterministic in battle time', () => {
  assert.equal(detonationLabel(6.2), 'DETONATION 7s');
  assert.equal(detonationLabel(-0.5), 'DETONATION 0s');
  assert.equal(detonationLabel(NaN), 'REACTOR UNSTABLE');
  assert.equal(hitFlash(-0.1), 0);
  assert.equal(hitFlash(0), 1);
  assert.ok(hitFlash(0.5) > hitFlash(1.0));
  assert.equal(hitFlash(2), 0);
  for (const t of [0, 0.11, 12.345, 300]) {
    const c = torchCough(t);
    assert.ok(c >= 0 && c <= 1);
    assert.equal(torchCough(t), c);
  }
  // it actually coughs: some dropouts over a few seconds
  let peak = 0;
  for (let t = 0; t < 5; t += 0.05) peak = Math.max(peak, torchCough(t));
  assert.ok(peak > 0.5);
});

test('wear sections follow the sim: bow-facing plating is nose, flanks lateral, aft tail', () => {
  const [n1, l1, t1] = wearSectionWeights(1, 0.9);
  assert.ok(n1 > 0.95 && t1 === 0);
  const [n2, l2, t2] = wearSectionWeights(0, 0);
  assert.ok(l2 > 0.99 && n2 === 0 && t2 === 0);
  const [, , t3] = wearSectionWeights(-1, -0.9);
  assert.ok(t3 > 0.95);
  for (const w of [wearSectionWeights(0.5, 0.6), wearSectionWeights(-0.4, -0.2)]) near(w[0] + w[1] + w[2], 1);
  assert.ok(l1 >= 0);
});
