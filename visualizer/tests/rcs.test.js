import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  ATTITUDE, MODE_NONE, MODE_TV, MODE_RCS, RCS_TIMING,
  jetTorqueAxes, jetDemands, defaultRcsMounts, planJet, pulseEnvelope,
  buildShipKinematics, sampleKinematics, newSample, bodyDemand, lookAtBasis,
  RcsPlanner, smoothingWindow, demandLevel
} from '../src/render/Rcs.js';

const DEG = Math.PI / 180;
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

// Mount order from defaultRcsMounts: per station (nose, tail) the nozzles
// point +y, -y, +x, -x
const NOSE_UP = 0, NOSE_DOWN = 1, NOSE_RIGHT = 2, NOSE_LEFT = 3;
const TAIL_UP = 4, TAIL_DOWN = 5, TAIL_RIGHT = 6, TAIL_LEFT = 7;

test('pitching the nose up fires the nose-down / tail-up couple and nothing else', () => {
  const mounts = defaultRcsMounts(8, 1.6);
  const axes = jetTorqueAxes(mounts);
  // nose toward +y: omega = z x y = -x, so the torque is about -x
  const d = jetDemands(axes, -1, 0, new Float32Array(8));
  const firing = [...d].map((v, i) => (v > 0.5 ? i : -1)).filter(i => i >= 0);
  assert.deepEqual(firing, [NOSE_DOWN, TAIL_UP]);
  close(d[NOSE_DOWN], 1, 1e-6);
  close(d[TAIL_UP], 1, 1e-6);
  // the reaction force of that couple nets to zero: a pure torque
  const f = [0, 0, 0];
  for (const k of firing) for (let c = 0; c < 3; c++) f[c] -= mounts[k].dir[c];
  assert.deepEqual(f.map(v => Math.abs(v)), [0, 0, 0]);
});

test('yaw fires the lateral couple; a diagonal demand shares across both couples', () => {
  const axes = jetTorqueAxes(defaultRcsMounts(8, 1.6));
  // nose toward +x: omega = z x x = +y
  const yaw = jetDemands(axes, 0, 1, new Float32Array(8));
  assert.deepEqual([...yaw].map((v, i) => (v > 0.5 ? i : -1)).filter(i => i >= 0), [NOSE_LEFT, TAIL_RIGHT]);
  const diag = jetDemands(axes, Math.SQRT1_2, Math.SQRT1_2, new Float32Array(8));
  const on = [...diag].map((v, i) => (v > 0.5 ? i : -1)).filter(i => i >= 0).sort();
  assert.deepEqual(on, [NOSE_UP, NOSE_LEFT, TAIL_RIGHT, TAIL_DOWN].sort());
  for (const k of on) close(diag[k], Math.SQRT1_2, 1e-6);
});

test('axial (translation) nozzles and hull-supplied Vector3 mounts', () => {
  const axes = jetTorqueAxes([
    { pos: new THREE.Vector3(0.5, 0, 3), dir: new THREE.Vector3(0, 0, 1) },   // fore-facing: poor leverage
    { pos: new THREE.Vector3(0, 0.6, -3), dir: new THREE.Vector3(0, 1, 0) }   // tail top
  ]);
  assert.deepEqual([...axes.slice(0, 3)], [0, 0, 0]);
  // tail +y nozzle: force -y at z=-3 -> full-leverage torque about -x
  close(axes[3], -1);
});

test('lookAtBasis reproduces Object3D.lookAt', () => {
  for (const f of [[1, 0, 0], [0.3, 0.8, -0.52], [0, 0, -1], [0, 1, 0]]) {
    const fwd = new THREE.Vector3(...f).normalize();
    const o = new THREE.Object3D();
    o.lookAt(fwd);
    const x = [0, 0, 0], y = [0, 0, 0];
    lookAtBasis(fwd.toArray(), x, y);
    const ex = new THREE.Vector3(1, 0, 0).applyQuaternion(o.quaternion);
    const ey = new THREE.Vector3(0, 1, 0).applyQuaternion(o.quaternion);
    close(ex.x, x[0], 1e-4); close(ex.y, x[1], 1e-4); close(ex.z, x[2], 1e-4);
    close(ey.x, y[0], 1e-4); close(ey.y, y[1], 1e-4); close(ey.z, y[2], 1e-4);
  }
});

/**
 * Bang-bang pitch turn sampled at 1 Hz like the recorder: 4-decimal
 * headings, and with `fields` the recorder's omega (nose change since the
 * previous frame), alpha (change of that omega) and turn (actuator over the
 * interval ending at the frame, null without torque).
 */
function bangBangTrace({ shipType, angleDeg, thrust, tStart = 5, T = 80, fields = false }) {
  const att = ATTITUDE[shipType];
  const tv = thrust > 0;
  const a = (tv ? att.tv + att.rcs : att.rcs) * DEG;
  const th = angleDeg * DEG;
  const half = Math.sqrt(th / a);
  const angleAt = (t) => {
    const u = Math.min(Math.max(t - tStart, 0), 2 * half);
    return u < half ? 0.5 * a * u * u : th - 0.5 * a * (2 * half - u) ** 2;
  };
  const frames = [];
  let prevW = 0;
  for (let t = 0; t <= T; t += 1) {
    const ang = angleAt(t);
    // heading +x, pitching up toward +y: rotation about +z
    const s = {
      fwd: [+Math.cos(ang).toFixed(4), +Math.sin(ang).toFixed(4), 0],
      pos: [0, 0, 0], vel: [0, 0, 0], thrust, destroyed: false, dying: false, hull: 100
    };
    if (fields) {
      const w = t > 0 ? ang - angleAt(t - 1) : 0;
      s.omega = [0, 0, w];
      s.alpha = [0, 0, t > 0 ? w - prevW : 0];
      s.turn = t > tStart && t - 1 < tStart + 2 * half ? (tv ? 'tv' : 'rcs') : null;
      prevW = w;
    }
    frames.push({ t, ships: { ship: s } });
  }
  return { frames, half };
}

for (const fields of [false, true]) {
  test(`an RCS turn reads as full demand, reversing at the midpoint (${fields ? 'recorded alpha' : 'heading only'})`, () => {
    const { frames, half } = bangBangTrace({ shipType: 'destroyer', angleDeg: 20, thrust: 0, fields });
    const k = buildShipKinematics(frames, 'ship', 'destroyer');
    const s = newSample(), d = [0, 0];
    const at = (t) => { sampleKinematics(k, t, s); bodyDemand(s, d); return [s.mode, d[0], d[1]]; };
    const [m1, x1] = at(5 + half * 0.5);
    const [m2, x2] = at(5 + half * 1.5);
    assert.equal(m1, MODE_RCS);
    assert.equal(m2, MODE_RCS);
    // nose to +y: accelerate about -x, then brake about +x
    assert.ok(x1 < -0.6 && x1 > -1.3, `accel demand ${x1}`);
    assert.ok(x2 > 0.6 && x2 < 1.3, `brake demand ${x2}`);
    const [, qx, qy] = at(5 + 2 * half + 8);
    assert.equal(Math.hypot(qx, qy), 0, 'no demand once the turn is done');
  });
}

test('a torch-vectored turn is TV mode, normalised by the TV authority', () => {
  const { frames, half } = bangBangTrace({ shipType: 'corvette', angleDeg: 60, thrust: 0.6 });
  const k = buildShipKinematics(frames, 'ship', 'corvette');
  const s = newSample(), d = [0, 0];
  sampleKinematics(k, 5 + half * 0.5, s);
  bodyDemand(s, d);
  assert.equal(s.mode, MODE_TV);
  assert.ok(d[0] < -0.6 && d[0] > -1.3, `tv demand ${d[0]}`);
});

test('a dreadnought RCS turn stands out of the heading rounding noise', () => {
  assert.equal(smoothingWindow((ATTITUDE.corvette.tv + ATTITUDE.corvette.rcs) * DEG), 1);
  assert.ok(smoothingWindow(ATTITUDE.dreadnought.rcs * DEG) >= 3);
  const { frames, half } = bangBangTrace({ shipType: 'dreadnought', angleDeg: 4, thrust: 0, T: 120 });
  const k = buildShipKinematics(frames, 'ship', 'dreadnought');
  const s = newSample(), d = [0, 0];
  sampleKinematics(k, 5 + half * 0.5, s);
  bodyDemand(s, d);
  assert.ok(d[0] < -0.4, `dreadnought demand ${d[0]}`);
  // and a ship holding its heading reads (near) zero
  sampleKinematics(k, 118, s);
  bodyDemand(s, d);
  assert.ok(Math.hypot(d[0], d[1]) < 0.05);
});

test('recorded omega/alpha line up with the heading the viewer slerps (no 1 s lag)', () => {
  const opts = { shipType: 'frigate', angleDeg: 25, thrust: 0 };
  const rec = buildShipKinematics(bangBangTrace({ ...opts, fields: true }).frames, 'ship', 'frigate');
  const fd = buildShipKinematics(bangBangTrace(opts).frames, 'ship', 'frigate');
  const s = newSample(), d = [0, 0];
  const at = (k, t) => { sampleKinematics(k, t, s); bodyDemand(s, d); return d[0]; };
  let best = null;
  for (const lag of [-1, 0, 1]) {
    let num = 0, a2 = 0, b2 = 0;
    for (let t = 1; t < 70; t += 0.25) {
      const x = at(rec, t), y = at(fd, t + lag);
      num += x * y; a2 += x * x; b2 += y * y;
    }
    const r = num / Math.sqrt(a2 * b2);
    if (!best || r > best.r) best = { lag, r };
  }
  assert.equal(best.lag, 0);
  assert.ok(best.r > 0.95, `correlation ${best.r}`);
});

test('an unsigned-rate spin reversal every frame never strobes jets at full power', () => {
  // wobbling corvette spinning at the TV max rate, reversing each frame -
  // the recorder's alpha spikes to ~0.5 rad/s^2 (11x authority)
  assert.equal(demandLevel(0.9), 0.9);
  assert.equal(demandLevel(11), 0.8);
  const w = 14.8 * DEG;
  const frames = [];
  let ang = 0, prevW = 0;
  for (let t = 0; t <= 60; t++) {
    const dir = t < 10 ? 0 : (t % 2 ? 1 : -1);
    const om = dir * w;
    ang += om;
    frames.push({ t, ships: { ship: {
      fwd: [+Math.cos(ang).toFixed(4), 0, +Math.sin(ang).toFixed(4)], thrust: 0.9,
      omega: [0, -om, 0], alpha: [0, prevW - om, 0], turn: dir ? 'tv' : null, evade: 'wobble',
      destroyed: false, dying: false
    } } });
    prevW = om;
  }
  const p = new RcsPlanner({ shipId: 'ship', shipType: 'corvette', mounts: defaultRcsMounts(4, 1), trace: frames });
  let fullCells = 0, cells = 0;
  for (let c = p.cellIndex(12); c <= p.cellIndex(58); c++) {
    const pl = p.plan(c);
    cells++;
    if ([...pl.level].some((l, k) => l >= 0.99 && pl.on[k] > 0)) fullCells++;
  }
  assert.ok(fullCells <= 2, `${fullCells}/${cells} full-power cells`);
});

test('recorded turn=null means no torque; dying and destroyed ships never fire', () => {
  const { frames } = bangBangTrace({ shipType: 'destroyer', angleDeg: 20, thrust: 0, fields: true });
  for (const f of frames) f.ships.ship.turn = null;
  const k = buildShipKinematics(frames, 'ship', 'destroyer');
  const s = newSample(), d = [0, 0];
  sampleKinematics(k, 12, s);
  bodyDemand(s, d);
  assert.equal(s.mode, MODE_NONE);
  assert.deepEqual(d, [0, 0]);

  const dying = bangBangTrace({ shipType: 'destroyer', angleDeg: 20, thrust: 0 }).frames;
  for (const f of dying) f.ships.ship.dying = f.t >= 3;
  const p = new RcsPlanner({ shipId: 'ship', shipType: 'destroyer', mounts: defaultRcsMounts(8, 1.6), trace: dying });
  for (let t = 3; t < 60; t += 0.05) for (let j = 0; j < 8; j++) assert.equal(p.intensity(t, j), 0);
});

test('pulse plans: RCS turns burn hard, vectored turns are faint except one kick', () => {
  const P = RCS_TIMING.cell;
  const j = (now, prev, fut, prevRise = 0) => ({ now, prev, fut, prevRise });
  const rcs = planJet(j(1, 1, 1), MODE_RCS);
  assert.equal(rcs.level, 1);
  assert.equal(rcs.on, P);                                 // continuous at full demand
  const light = planJet(j(0.3, 0.3, 0.3), MODE_RCS);
  assert.ok(light.on >= RCS_TIMING.minPulse && light.on < P);
  assert.equal(planJet(j(0.05, 0, 0), MODE_RCS).on, 0);
  assert.equal(planJet(j(1, 1, 1), MODE_NONE).on, 0);

  const steady = planJet(j(1, 1, 1), MODE_TV);
  assert.ok(steady.level < 0.5 && steady.on > 0);
  const kick = planJet(j(1, 0, 1), MODE_TV);               // a vectored turn just started
  assert.equal(kick.level, 1);
  assert.ok(kick.on >= 0.1 && kick.on <= 0.6);
  // the cell after the kick is back to the faint assist
  assert.ok(planJet(j(1, 0.2, 1, 1), MODE_TV).level < 0.5);
  // steering jitter that reverses within the second never fires
  assert.equal(planJet(j(1, 0, 0), MODE_TV).on, 0);

  // envelope: attack, sustain, tail-off
  assert.equal(pulseEnvelope(-0.01, 0.2), 0);
  assert.equal(pulseEnvelope(0.1, 0.2), 1);
  assert.ok(pulseEnvelope(0.35, 0.2) < 0.1);
});

test('jet intensity is a pure function of battle time (seek-safe)', () => {
  const { frames } = bangBangTrace({ shipType: 'frigate', angleDeg: 25, thrust: 0 });
  const mk = () => new RcsPlanner({ shipId: 'ship', shipType: 'frigate', mounts: defaultRcsMounts(6, 1.2), trace: frames });
  const a = mk(), b = mk();
  const times = [];
  for (let t = 0; t < 70; t += 0.037) times.push(t);
  const fwd = times.map(t => [...Array(8).keys()].map(j => a.intensity(t, j)));
  const back = times.slice().reverse().map(t => [...Array(8).keys()].map(j => b.intensity(t, j))).reverse();
  assert.deepEqual(fwd, back);
  // the turn fires the pitch-up couple first, the braking couple later
  const burn = (j, t0, t1) => times.reduce((acc, t, i) => acc + (t >= t0 && t < t1 ? fwd[i][j] : 0), 0);
  assert.ok(burn(NOSE_DOWN, 6, 20) > 10 * burn(NOSE_UP, 6, 20));
  assert.ok(burn(NOSE_UP, 30, 44) > 10 * burn(NOSE_DOWN, 30, 44));
});
