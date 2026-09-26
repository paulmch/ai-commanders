import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { BattleLoader } from '../src/BattleLoader.js';
import { Projectiles } from '../src/render/Projectiles.js';
import {
  WeaponTimeline, makeSample, sampleTrack, tracerLength, missOpacity,
  beamPulse, seedFor, torpedoPower, MISS_LIFE
} from '../src/render/WeaponTimeline.js';

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-6, `${a} != ${b}`);
const ship = { pos: [50000, 0, 0], vel: [100, 0, 0], fwd: [-1, 0, 0] };
const slug = (id, x) => ({ id, pos: [x, 0, 0], vel: [10000, 0, 0], source: 'alpha', target: 'beta' });
const torp = (x, dv) => ({ ...slug('torp', x), dv_remaining: dv, disabled: false });
function recording() {
  const loader = new BattleLoader();
  loader.simTrace = [
    { t: 0, ships: { alpha: { ...ship, pos: [0, 0, 0], fwd: [1, 0, 0] }, beta: ship } },
    { t: 1, ships: { alpha: ship, beta: ship }, projectiles: [slug('hit', 10000), slug('miss', 10000), slug('kill', 10000)], torpedoes: [torp(10000, 12)] },
    { t: 2, ships: { alpha: ship, beta: ship }, projectiles: [slug('hit', 20000), slug('miss', 20000), slug('kill', 20000)], torpedoes: [torp(20000, 11)] },
    { t: 3, ships: { alpha: ship, beta: ship }, torpedoes: [torp(30000, 10)] },
    { t: 4, ships: { alpha: ship, beta: ship } },
    { t: 10, ships: { alpha: ship, beta: ship } }
  ];
  loader.events = [
    { timestamp: 0, event_type: 'shot_fired', ship_id: 'alpha', data: { weapon_name: 'Spinal Coiler', weapon_slot: 'weapon_0' } },
    { timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha', data: { torpedo_id: 'torp', delta_v_kps: 14 } },
    { timestamp: 1, event_type: 'pd_fired', ship_id: 'beta', data: { target_id: 'torp', target_type: 'torpedo', turret_name: 'pd_0' } },
    { timestamp: 2, event_type: 'pd_fired', ship_id: 'beta', data: { target_id: 'torp', target_type: 'torpedo', turret_name: 'pd_0' } },
    { timestamp: 2.4, event_type: 'pd_torpedo_disabled', ship_id: 'beta', data: { torpedo_id: 'torp' } },
    { timestamp: 2.5, event_type: 'pd_slug_destroyed', ship_id: 'beta', data: { projectile_id: 'kill' } },
    { timestamp: 3, event_type: 'hit', ship_id: 'beta', data: { projectile_id: 'hit', impact_position: [30000, 0, 0], kinetic_energy_gj: 5 } },
    { timestamp: 4, event_type: 'torpedo_impact', ship_id: 'alpha', data: { torpedo_id: 'torp', target_id: 'beta', total_damage_gj: 24 } }
  ];
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  return loader;
}

test('virtual shutter follows speed, resolves distant streaks, and never predates launch', () => {
  near(tracerLength(10, 0.01), 0.85);
  near(tracerLength(20, 0.01), 1.7);
  near(tracerLength(10, 1), 7);
  near(tracerLength(10, 1, 0.03), 0.3);
  assert.equal(tracerLength(0, 2), 0);
  assert.equal(missOpacity(-1), 1);
  assert.equal(missOpacity(MISS_LIFE), 0);
  assert.ok(missOpacity(1) > missOpacity(2));
});

test('sampler writes reusable storage and is independent of seek order', () => {
  const timeline = new WeaponTimeline(recording()), out = makeSample();
  const tr = timeline.slugs.get('hit'), pos = out.pos;
  timeline.sample(tr, 2.8, out); near(out.pos[0], 28000);
  timeline.sample(tr, 0.2, out); near(out.pos[0], 2000);
  timeline.sample(tr, 1.5, out); near(out.pos[0], 15000);
  timeline.sample(tr, 2.8, out); near(out.pos[0], 28000);
  assert.equal(out.pos, pos);
  sampleTrack(tr, 3.5, out); near(out.pos[0], 35000);
});

test('misses fly onward, hits/intercepts stop, and the live edge stays active', () => {
  const loader = recording(), timeline = new WeaponTimeline(loader);
  assert.equal(timeline.slugs.get('hit').end, 3);
  assert.equal(timeline.slugs.get('miss').end, 2 + MISS_LIFE);
  assert.equal(timeline.slugs.get('kill').end, 2.5);
  assert.equal(timeline.slugs.get('kill').missAt, Infinity);
  const out = makeSample();
  timeline.sample(timeline.slugs.get('miss'), 4, out); near(out.pos[0], 40000);
  loader.simTrace = loader.simTrace.slice(0, 3);
  const live = new WeaponTimeline(loader);
  assert.equal(live.slugs.get('miss').end, Infinity);
});

test('torpedoes preserve launch/disable/outcome times and bridge to the impact', () => {
  const timeline = new WeaponTimeline(recording()), tr = timeline.torpedoes.get('torp');
  assert.equal(tr.start, 0); assert.equal(tr.disabledAt, 2.4); assert.equal(tr.end, 4);
  const out = makeSample();
  timeline.sample(tr, 3.5, out); near(out.pos[0], 40000);
  timeline.sample(tr, 4, out); near(out.pos[0], 50000);
  assert.equal(timeline.moments.filter(m => m.kind === 'seeker').length, 1);
});

test('power uses fuel consumption and geometry, with coast/disable taking precedence', () => {
  const a = torp(0, 12), b = torp(0, 11.9);
  assert.equal(torpedoPower(a, b, 1, 300000, 12000), 0.42);
  assert.equal(torpedoPower(a, b, 1, 60000, 12000), 1);
  assert.equal(torpedoPower(a, b, 1, 180000, 12500), 1);
  assert.equal(torpedoPower({ ...a, phase: 'TERMINAL' }, b, 1, 300000, 12000), 1);
  assert.equal(torpedoPower({ ...a, phase: 'CRUISE' }, b, 1, 30000, 12000), 0.42);
  assert.equal(torpedoPower({ ...a, phase: 'COAST' }, b, 1, 30000, 12000), 0);
  assert.equal(torpedoPower(a, a, 1, 30000, 12000), 0);
  assert.equal(torpedoPower(a, b, 1, 30000, 12000, true), 0);
});

test('PD flicker has reproducible phases and respects dwell boundaries', () => {
  const seed = seedFor('turret');
  assert.equal(seed, seedFor('turret')); assert.notEqual(seed, seedFor('other'));
  assert.equal(beamPulse(0.9, 1, 2, seed), 0);
  assert.equal(beamPulse(3, 1, 2, seed), 0);
  assert.ok(beamPulse(1.5, 1, 2, seed) > 0);
  assert.notEqual(beamPulse(1.5, 1, 2, seed), beamPulse(1.55, 1, 2, seed));
});

function renderer() {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  camera.position.set(20, 30, 50);
  const ships = new Map();
  for (const id of ['alpha', 'beta']) {
    const obj = new THREE.Group(); obj.position.set(id === 'alpha' ? 0 : 50, 0, 0);
    obj.userData = { size: { length: 8, width: 2 }, shipType: 'destroyer' };
    ships.set(id, obj); scene.add(obj);
  }
  return new Projectiles({ scene, camera, ships });
}

function snapshot(fx) {
  const result = [];
  for (const batch of [fx.ribbons, fx.flashes]) {
    result.push(batch.count);
    for (const key of Object.keys(batch.fields)) {
      const a = batch.geometry.attributes[key];
      result.push(Array.from(a.array.subarray(0, batch.count * a.itemSize)));
    }
  }
  result.push(fx.bodies.mesh.count, Array.from(fx.bodies.mesh.instanceMatrix.array.subarray(0, fx.bodies.mesh.count * 16)));
  return result;
}

test('all weapon buffers match on direct seek, playback, backward seek and pause', () => {
  const loader = recording(), fx = renderer(), direct = renderer();
  const buffers = fx.ribbons.geometry.attributes.aFrom.array;
  try {
    for (const t of [0.03, 1.53, 2.43, 2.58, 3.12, 4.16, 7.5]) {
      direct.update(loader, t);
      const expected = snapshot(direct);
      for (let s = 0; s < t; s += 0.17) fx.update(loader, s);
      fx.update(loader, t);
      assert.deepEqual(snapshot(fx), expected, `playback at ${t}`);
      fx.update(loader, 9); fx.update(loader, t);
      assert.deepEqual(snapshot(fx), expected, `backward seek at ${t}`);
      fx.update(loader, t);
      assert.deepEqual(snapshot(fx), expected, `pause at ${t}`);
      for (const part of expected.flat(2)) assert.ok(Number.isFinite(part));
    }
    assert.equal(fx.ribbons.geometry.attributes.aFrom.array, buffers);
    assert.equal(fx.manager.scene.children.filter(c => c.isLight).length, 0);
    fx.update(loader, 2.43);
    assert.equal(fx.position('torpedo', 'torp')?.x, 24.3);
    fx.clear();
    assert.equal(fx.ribbons.mesh.visible, false);
    assert.equal(fx.position('torpedo', 'torp'), null);
  } finally { fx.dispose(); direct.dispose(); }
});

test('a PD destruction pops without retaining a live torpedo body', () => {
  const loader = recording();
  loader.events = loader.events.filter(e => e.event_type !== 'torpedo_impact');
  loader.events.find(e => e.event_type === 'pd_torpedo_disabled').event_type = 'pd_torpedo_destroyed';
  loader.buildTorpedoTimeline();
  const fx = renderer();
  try {
    fx.update(loader, 2.43);
    assert.equal(fx.position('torpedo', 'torp'), null);
    assert.equal(fx.bodies.mesh.count, 0);
    assert.ok(fx.flashes.count > 0);
    assert.equal(fx.timeline.torpedoes.get('torp').end, 2.4);
    assert.equal(fx.timeline.moments.filter(m => m.kind === 'pop' && m.track.id === 'torp').length, 1);
  } finally { fx.dispose(); }
});

test('firing turret muzzle is independent of previous wall-clock slew', () => {
  const loader = recording();
  loader.events[0] = { timestamp: 1.5, event_type: 'shot_fired', ship_id: 'alpha',
    data: { target_id: 'beta', weapon_slot: 'weapon_0', weapon_name: 'Coilgun Turret' } };
  loader.events.sort((a, b) => a.timestamp - b.timestamp);
  const fx = renderer(), ship = fx.manager.ships.get('alpha');
  const mount = new THREE.Group(), yaw = new THREE.Group(), pitch = new THREE.Group();
  ship.add(mount); mount.add(yaw); yaw.add(pitch);
  ship.userData.turrets = [{ mount, yaw, pitch, s: 0.5, yawAngle: 0, pitchAngle: 0 }];
  try {
    fx.update(loader, 1.55);
    const expected = snapshot(fx);
    yaw.rotation.y = 2.9; pitch.rotation.x = -0.8;
    fx.update(loader, 1.55);
    assert.deepEqual(snapshot(fx), expected);
  } finally { fx.dispose(); }
});

test('fresh torpedoes are pushed out of the oversize launcher hull, monotonically', () => {
  const loader = recording(), fx = renderer();
  try {
    fx.update(loader, 1);
    const tr = fx.timeline.torpedoes.get('torp');
    assert.equal(tr.source, 'alpha');
    // alpha sits at x=0 at t=0 with fwd +x; the drawn hull is 8 km long.
    const at = x => fx.lift(tr, 0, new THREE.Vector3(x, 0, 0)).x;
    near(at(1e-9), 4.15);
    let prev = -Infinity;
    for (let x = 0.001; x < 9; x += 0.25) { const v = at(x); assert.ok(v > prev, `stall at ${x}`); prev = v; }
    near(at(8.5), 8.5);
    assert.equal(fx.lift(tr, 60, new THREE.Vector3(0.5, 0, 0)).x, 0.5);
  } finally { fx.dispose(); }
});

test('main.js defers seeker/intercept/burnout flares here and keeps detonations', () => {
  const fx = renderer();
  try {
    for (const k of ['seeker_kill', 'intercepted', 'destroyed', 'burnout']) assert.ok(fx.ownsMoment(k), k);
    for (const k of ['impact', 'miss']) assert.equal(fx.ownsMoment(k), false, k);
    const kinds = new WeaponTimeline(recording()).moments.map(m => m.kind);
    assert.ok(kinds.includes('ignition'));
  } finally { fx.dispose(); }
});

test('a PD burn spot sits on the torpedo surface facing the beam', () => {
  const loader = recording(), fx = renderer();
  try {
    fx.update(loader, 1.5);
    const tr = fx.timeline.torpedoes.get('torp'), out = new THREE.Vector3();
    fx.dir.set(-1, 0, 0); // beam from ahead: strikes the seeker nose (+x)
    fx.surfaceOffset(tr, true, 1.5, out);
    near(out.x, 0.11 + 0.92); near(out.y, 0);
    fx.dir.set(0, 1, 0); // beam from the side: strikes the flank
    fx.surfaceOffset(tr, true, 1.5, out);
    near(out.y, -0.17); near(out.x, 0.11);
  } finally { fx.dispose(); }
});
