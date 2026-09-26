import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { BattleLoader } from '../src/BattleLoader.js';
import { Projectiles } from '../src/render/Projectiles.js';
import { WeaponTimeline, torpedoPower } from '../src/render/WeaponTimeline.js';
import {
  RCS_LAYOUT, PULSE, SLEW, LEGACY, JET_COUNT, jetLevels, slewProfile, buildTorpedoThrusters,
  attitudeAt, pulsesAt, makePulses, legacyTrim, tickAt, throttleAt
} from '../src/render/TorpedoRcs.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const V = (x, y, z) => ({ x, y, z });
const EX = V(1, 0, 0), EY = V(0, 1, 0), ZERO = V(0, 0, 0);
const phi = k => RCS_LAYOUT.phase + (k & 3) * Math.PI / 2;
const nozzle = k => V(Math.cos(phi(k)), Math.sin(phi(k)), 0);   // body = world here
const ringZ = k => RCS_LAYOUT.rings[k >> 2];

/** Net force and torque (about the origin) of jet levels on a +Z-nosed body. */
function wrench(levels) {
  const F = [0, 0, 0], T = [0, 0, 0];
  for (let k = 0; k < JET_COUNT; k++) {
    const n = nozzle(k), l = levels[k];
    const f = [-n.x * l, -n.y * l, 0];                 // thrust opposes the exhaust
    const r = [n.x * RCS_LAYOUT.radius, n.y * RCS_LAYOUT.radius, ringZ(k)];
    F[0] += f[0]; F[1] += f[1];
    T[0] += r[1] * f[2] - r[2] * f[1];
    T[1] += r[2] * f[0] - r[0] * f[2];
    T[2] += r[0] * f[1] - r[1] * f[0];
  }
  return { F, T };
}

test('a translation trim fires the quads facing away from the thrust, on both rings', () => {
  const out = new Float32Array(JET_COUNT);
  for (const [tx, ty] of [[1, 0], [0, -0.6], [0.3, 0.4], [-0.5, 0.2]]) {
    jetLevels(EX, EY, V(tx, ty, 0.8), ZERO, out);    // the along-nose part is ignored
    const { F } = wrench(out);
    near(F[0] / 2, tx); near(F[1] / 2, ty);           // each ring delivers exactly T_lat
    for (let k = 0; k < JET_COUNT; k++) {
      const n = nozzle(k);
      if (n.x * tx + n.y * ty >= 0) assert.equal(out[k], 0, `jet ${k} exhausts along the thrust`);
      assert.equal(out[k], out[(k + 4) % 8], 'rings fire alike: no torque couple');
    }
  }
  jetLevels(EX, EY, V(0, 0, 1), ZERO, out);
  assert.deepEqual(Array.from(out), Array(8).fill(0), 'no radial quad can push along the nose');
});

test('an attitude couple turns the nose toward the push with no net force', () => {
  const out = new Float32Array(JET_COUNT);
  jetLevels(EX, EY, ZERO, V(1, 0, 0), out);
  const { F, T } = wrench(out);
  near(F[0], 0); near(F[1], 0);
  assert.ok(T[1] > 0, 'torque about +Y swings +Z toward +X');
  near(T[0], 0);
  // forward quads exhaust away from the push, aft quads toward it
  for (let k = 0; k < JET_COUNT; k++) {
    if (!(out[k] > 0)) continue;
    assert.ok(k < 4 ? nozzle(k).x > 0 : nozzle(k).x < 0, `jet ${k}`);
  }
  // body axes are honoured: a rolled body picks different quads, same wrench
  const c = Math.cos(0.7), s = Math.sin(0.7);
  jetLevels(V(c, s, 0), V(-s, c, 0), ZERO, V(1, 0, 0), out);
  assert.ok(out.some(l => l > 0));
});

test('slew profile: accelerate, coast, brake; continuous and monotone', () => {
  for (const a of [0.12, 0.25, 0.5]) {
    assert.equal(slewProfile(0, a), 0); assert.equal(slewProfile(1, a), 1);
    near(slewProfile(0.5, a), 0.5);
    let prev = 0;
    for (let f = 0.001; f < 1; f += 0.001) {
      const v = slewProfile(f, a);
      assert.ok(v >= prev - 1e-12 && v - prev < 0.004, `a=${a} f=${f}`);
      prev = v;
    }
    near(slewProfile(a, a), slewProfile(a + 1e-9, a), 1e-6);
  }
});

/** Telemetry samples: ticks ending at t = 1..n, 1 s apart. */
function tele(rows) {
  return rows.map((r, i) => ({ t: i + 1, los: r.los, state: {
    pos: [i * 1000, 0, 0], vel: [1000, 0, 0], dv_remaining: 10 - i * 0.1, disabled: !!r.disabled,
    thr: r.thr ?? 0, ...(r.fwd ? { fwd: r.fwd } : {}), ...(r.rcs ? { rcs: r.rcs } : {})
  } }));
}

test('the nose is on each burn axis by its tick start, tracks the seeker when cold, holds when blind', () => {
  const samples = tele([
    { thr: 1, fwd: [1, 0, 0] },                  // tick 1: t 0..1
    { thr: 1, fwd: [1, 0, 0], los: [0, 1, 0] },  // tick 2
    { thr: 0 },                                  // tick 3 cold: seeker LOS at its start (+Y)
    { thr: 0.5, fwd: [0, 0, 1] },                // tick 4 lit along +Z
    { thr: 0, disabled: true, los: [0, -1, 0] }, // tick 5: seeker dead -> hold
    { thr: 0, disabled: true }
  ]);
  const th = buildTorpedoThrusters(samples);
  assert.ok(th.tele);
  const n = t => { const o = attitudeAt(th, t, V(0, 0, 0)); return [o.x, o.y, o.z].map(v => Math.round(v * 1e6) / 1e6); };
  assert.deepEqual(n(0), [1, 0, 0]);
  assert.deepEqual(n(1), [1, 0, 0]);   // start of tick 2
  assert.deepEqual(n(2), [0, 1, 0]);   // start of tick 3: looking down the LOS
  assert.deepEqual(n(3), [0, 0, 1]);   // start of tick 4: on the burn axis
  assert.deepEqual(n(4.5), [0, 0, 1]); // blind: holds
  assert.deepEqual(n(9), [0, 0, 1]);
  const mid = attitudeAt(th, 1.5, V(0, 0, 0));
  near(Math.hypot(mid.x, mid.y, mid.z), 1);
  near(mid.x, Math.SQRT1_2); near(mid.y, Math.SQRT1_2);   // symmetric profile: halfway at mid-tick
  assert.equal(tickAt(th, 3.2), 3); assert.equal(throttleAt(th, 3.2), 0.5);   // samples[3] closes 3..4
  assert.equal(tickAt(th, -1), 0); assert.equal(tickAt(th, 6), -1);
});

test('attitude couples: spin-up then brake, full RCS when cold, gimbal-assisted when lit', () => {
  const th = buildTorpedoThrusters(tele([
    { thr: 0, los: [1, 0, 0] }, { thr: 0, los: [0, 1, 0] }, { thr: 0 }, { thr: 1, fwd: [0, 1, 0] }, { thr: 1, fwd: [0, 0, 1] }, { thr: 1, fwd: [0, 0, 1] }
  ]));
  // Ticks 2-3 are cold: the seeker's line of sight swings +X -> +Y during
  // tick 2 (1..2 s) and the RCS alone turns the body after it.
  const list = makePulses(), gim = { x: 0, y: 0, z: 0, angle: 0 };
  pulsesAt(th, 1.05, 0, list, gim);
  const spin = Array.from({ length: list.count }, (_, k) => list[k]).find(p => p.push.y > 0.99);
  assert.ok(spin, 'spin-up pushes the nose toward the new attitude');
  assert.equal(spin.level, 1); assert.equal(spin.start, 1);
  assert.equal(gim.angle, 0, 'no gimbal with the engine cold');
  pulsesAt(th, 1.98, 0, list, gim);
  const brake = Array.from({ length: list.count }, (_, k) => list[k]).find(p => p.push.x > 0.99);
  assert.ok(brake, 'brake pushes back toward the old attitude');
  // Tick 4 (lit, +Y -> +Z): smaller RCS share, gimbal deflects toward the slew.
  pulsesAt(th, 3.05, 0, list, gim);
  const lit = Array.from({ length: list.count }, (_, k) => list[k]).find(p => p.push.z > 0.99);
  assert.ok(lit && lit.level > 0 && lit.level <= SLEW.litLevel + 1e-6);
  assert.ok(gim.angle > 0 && gim.angle <= SLEW.gimbalMax && gim.z > 0.99);
  // A steady attitude fires nothing.
  pulsesAt(th, 4.6, 0, list, gim);
  assert.equal(list.count, 0);
});

test('translation trims pulse at the recorded throttle and puffs outlive the tick briefly', () => {
  const th = buildTorpedoThrusters(tele([
    { thr: 1, fwd: [1, 0, 0] }, { thr: 0, rcs: [0, 0.5, 0] }, { thr: 0 }
  ]));
  const list = makePulses();
  const at = t => { pulsesAt(th, t, 0.05, list, null); return Array.from({ length: list.count }, (_, k) => list[k]); };
  assert.equal(at(0.9).length, 0, 'nothing before the trim tick');
  const firing = at(1.1);
  assert.equal(firing.length, 1);
  const p = firing[0];
  near(p.start, 1.05); near(p.end, 1.05 + 0.5 * PULSE.period);
  near(p.thrust.y, 1); near(p.level, 0.55 + 0.45 * 0.5);
  assert.ok(at(1.2).length >= 1);
  assert.ok(at(2 + PULSE.puffLife * 0.5).length >= 1, 'the last puff is still drifting');
  assert.equal(at(2 + PULSE.puffLife + 0.01).length, 0);
  // Duty cycle: on for tau of each period across the tick
  let on = 0;
  for (let t = 1.0005; t < 2; t += 0.001) if (at(t).some(q => t < q.end && t >= q.start)) on += 0.001;
  near(on, 0.5, 0.02);   // five 0.1 s openings, one per 0.2 s period
});

test('old recordings: a trim only where velocity turned with the main engine cold', () => {
  const out = new Float32Array(3);
  const a = { vel: [1000, 0, 0], dv_remaining: 8 };
  // 3 m/s^2 sideways plus 2 m/s^2 along track, 0.01 km/s of fuel: an RCS trim
  assert.ok(legacyTrim(a, { vel: [1002, 3, 0], dv_remaining: 7.99 }, 1, out, 0));
  near(out[1], 3 / LEGACY.refAccel, 0.01); near(out[0], 0, 0.02); near(out[2], 0);
  // the same turn while burning hard is the gimbaled main engine
  assert.equal(legacyTrim(a, { vel: [1100, 3, 0], dv_remaining: 7.6 }, 1, out, 0), false);
  // rounding noise and dead seekers fire nothing
  assert.equal(legacyTrim(a, { vel: [1000, 0.2, 0.1], dv_remaining: 8 }, 1, out, 0), false);
  assert.equal(legacyTrim({ ...a, disabled: true }, { vel: [1000, 3, 0], dv_remaining: 8 }, 1, out, 0), false);
  // a legacy track has no attitude keys: its nose stays on the velocity
  const th = buildTorpedoThrusters([{ t: 1, state: a }, { t: 2, state: { vel: [1000, 5, 0], dv_remaining: 8 } }]);
  assert.equal(th.tele, false);
  near(th.rcs[4], 5 / LEGACY.refAccel, 1e-4);
});

test('telemetry decides the plume: RCS fuel use alone never lights the main engine', () => {
  const a = { dv_remaining: 5.32 }, b = { dv_remaining: 5.31 };
  assert.equal(torpedoPower(a, { ...b, thr: 0, rcs: [0, 0.1, 0] }, 1, 300000, 12000), 0);
  assert.equal(torpedoPower(a, { ...a, thr: 0.4 }, 1, 300000, 12000), 0.42);
  assert.equal(torpedoPower(a, { ...b, thr: 1 }, 1, 30000, 12000), 1);
});

// ---------------------------------------------------------------------------
// Renderer: instanced, deterministic on battle time
// ---------------------------------------------------------------------------
const torp = (t, extra) => ({ id: 'torp', pos: [t * 5000, 0, 0], vel: [5000, 0, 0], source: 'alpha', target: 'beta',
  dv_remaining: 10 - t * 0.2, disabled: false, ...extra });
function recording() {
  const loader = new BattleLoader();
  const ship = { pos: [60000, 0, 0], vel: [0, 0, 0], fwd: [-1, 0, 0] };
  const alpha = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0] };
  const rows = [
    { thr: 1, fwd: [1, 0, 0] }, { thr: 1, fwd: [0.8, 0.6, 0] }, { thr: 0, rcs: [0, 0, 0.7] },
    { thr: 0 }, { thr: 0.3, fwd: [1, 0, 0] }, { thr: 1, fwd: [1, 0, 0] }
  ];
  loader.simTrace = [{ t: 0, ships: { alpha, beta: ship } }, ...rows.map((r, i) => ({
    t: i + 1, ships: { alpha, beta: ship }, torpedoes: [torp(i + 1, r)]
  })), { t: 8, ships: { alpha, beta: ship } }];
  loader.events = [{ timestamp: 0, event_type: 'torpedo_launched', ship_id: 'alpha', data: { torpedo_id: 'torp' } }];
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  return loader;
}
function renderer(camX) {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  camera.position.set(camX, 0, 4);
  const ships = new Map();
  for (const id of ['alpha', 'beta']) {
    const obj = new THREE.Group(); obj.position.set(id === 'alpha' ? 0 : 60, 0, 0);
    obj.userData = { size: { length: 8, width: 2 }, shipType: 'destroyer' };
    ships.set(id, obj); scene.add(obj);
  }
  return new Projectiles({ scene, camera, ships });
}
function kinds(batch, kind) {
  const style = batch.geometry.attributes.aStyle.array, out = [];
  for (let i = 0; i < batch.count; i++) if (Math.round(style[4 * i + 2]) === kind) out.push(i);
  return out;
}
function snapshot(fx) {
  const r = [];
  for (const batch of [fx.ribbons, fx.flashes]) {
    r.push(batch.count);
    for (const key of Object.keys(batch.fields)) {
      const a = batch.geometry.attributes[key];
      r.push(Array.from(a.array.subarray(0, batch.count * a.itemSize)));
    }
  }
  for (const m of [fx.bodies.mesh, fx.bodies.bell]) r.push(m.count, Array.from(m.instanceMatrix.array.subarray(0, m.count * 16)));
  return r;
}

test('RCS jets point away from the recorded trim and appear only while it fires', () => {
  const loader = recording(), fx = renderer(15);
  try {
    fx.update(loader, 2.3);   // tick 3: engine cold, rcs +Z at 0.7
    const tr = fx.timeline.torpedoes.get('torp');
    assert.ok(tr.thrusters.tele);
    const jets = kinds(fx.ribbons, 7);
    assert.ok(jets.length > 0, 'trim jets drawn');
    assert.equal(kinds(fx.ribbons, 3).length, 0, 'main engine cold: no plume');
    const from = fx.ribbons.geometry.attributes.aFrom.array, to = fx.ribbons.geometry.attributes.aTo.array;
    for (const i of jets) assert.ok(to[3 * i + 2] - from[3 * i + 2] < 0, 'exhaust leaves opposite the +Z thrust');
    fx.update(loader, 3.5);   // tick 4: cold, no trim, steady attitude
    assert.equal(kinds(fx.ribbons, 7).length, 0);
    fx.update(loader, 4.5);   // tick 5: lit at 30%
    assert.ok(kinds(fx.ribbons, 3).length === 1);
    // gone at fleet range
    const far = renderer(15); far.manager.camera.position.set(15, 0, 4000);
    far.update(loader, 2.3);
    assert.equal(kinds(far.ribbons, 7).length, 0);
    assert.equal(kinds(far.flashes, 5).length, 0);
    far.dispose();
  } finally { fx.dispose(); }
});

test('torpedo thruster buffers match on direct seek, playback, backward seek and pause', () => {
  const loader = recording(), fx = renderer(15), direct = renderer(15);
  try {
    for (const t of [0.4, 1.02, 1.5, 2.13, 2.37, 3.08, 4.9, 6.2]) {
      direct.update(loader, t);
      const expected = snapshot(direct);
      for (let s = 0; s < t; s += 0.07) fx.update(loader, s);
      fx.update(loader, t);
      assert.deepEqual(snapshot(fx), expected, `playback at ${t}`);
      fx.update(loader, 7.5); fx.update(loader, t);
      assert.deepEqual(snapshot(fx), expected, `backward seek at ${t}`);
      for (const part of expected.flat(2)) assert.ok(Number.isFinite(part));
    }
    assert.equal(fx.manager.scene.children.filter(c => c.isLight).length, 0);
  } finally { fx.dispose(); direct.dispose(); }
});

test('the body follows the recorded thrust axis, not the velocity', () => {
  const loader = recording(), fx = renderer(15);
  try {
    fx.update(loader, 1.0);   // start of tick 2: nose on (0.8, 0.6, 0) while flying +X
    const m = new THREE.Matrix4().fromArray(fx.bodies.mesh.instanceMatrix.array, 0);
    const nose = new THREE.Vector3(0, 0, 1).transformDirection(m);
    near(nose.x, 0.8, 1e-5); near(nose.y, 0.6, 1e-5);
    assert.ok(new WeaponTimeline(loader).torpedoes.get('torp').thrusters.tele);
  } finally { fx.dispose(); }
});
