import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { hullBasis, hullQuat, interpAttitude, lerpUp, validUp } from '../src/HullFrame.js';
import { Interpolator } from '../src/Interpolator.js';
import { localBasis, shipStateAt, worldToLocalDir } from '../src/DamageTimeline.js';
import { lookAtBasis } from '../src/render/Rcs.js';
import { makeSample, sampleTrack } from '../src/render/WeaponTimeline.js';

const near = (a, b, eps = 1e-6) => a.every((v, i) => Math.abs(v - b[i]) < eps);
const norm = (v) => { const l = Math.hypot(...v); return v.map(c => c / l); };

test('without `up` the hull frame is the viewer\'s old lookAt roll (world-up +Y)', () => {
  for (const f of [[1, 0, 0], [0.6, 0.3, -0.74], [-0.2, 0.9, 0.38], [0, 0, 1]].map(norm)) {
    const o = new THREE.Object3D();
    o.lookAt(new THREE.Vector3(...f));
    const x = [0, 0, 0], y = [0, 0, 0];
    assert.equal(hullBasis(f, null, x, y), false);
    assert.ok(near(x, new THREE.Vector3(1, 0, 0).applyQuaternion(o.quaternion).toArray()), `port for ${f}`);
    assert.ok(near(y, new THREE.Vector3(0, 1, 0).applyQuaternion(o.quaternion).toArray()), `dorsal for ${f}`);
    const q = hullQuat(f, null);
    assert.ok(Math.abs(Math.abs(new THREE.Quaternion(...q).dot(o.quaternion)) - 1) < 1e-6);
    // every consumer's own basis helper agrees
    const B = localBasis(f);
    assert.ok(near(B.x, x) && near(B.y, y));
    const rx = [0, 0, 0], ry = [0, 0, 0];
    lookAtBasis(f, rx, ry);
    assert.ok(near(rx, x) && near(ry, y));
  }
});

test('with `up` the hull takes the ship\'s real roll: dorsal = up, port = up x fwd', () => {
  // engine default attitude: nose +X, up +Z -> port is the engine's +Y (left)
  const x = [0, 0, 0], y = [0, 0, 0];
  assert.equal(hullBasis([1, 0, 0], [0, 0, 1], x, y), true);
  assert.ok(near(y, [0, 0, 1]) && near(x, [0, 1, 0]));
  // the old implicit roll would have put dorsal on world +Y instead
  hullBasis([1, 0, 0], null, x, y);
  assert.ok(near(y, [0, 1, 0]));
  // a rolled ship, up slightly off-perpendicular (4-decimal rounding)
  const f = norm([0.8, 0.6, 0]), up = norm([0.3, -0.4, 0.866]);
  hullBasis(f, [up[0] + 1e-4, up[1], up[2]], x, y);
  const q = new THREE.Quaternion(...hullQuat(f, up));
  assert.ok(near(new THREE.Vector3(0, 0, 1).applyQuaternion(q).toArray(), f, 1e-5), 'nose');
  assert.ok(near(new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray(), up, 1e-3), 'dorsal');
  assert.ok(Math.abs(x[0] * f[0] + x[1] * f[1] + x[2] * f[2]) < 1e-9 && Math.abs(Math.hypot(...x) - 1) < 1e-9);
  // right-handed: port x dorsal = nose
  const c = new THREE.Vector3(...x).cross(new THREE.Vector3(...y)).toArray();
  assert.ok(near(c, f, 1e-6));
  // the other consumers take the same frame
  const B = localBasis(f, null, up);
  const rx = [0, 0, 0], ry = [0, 0, 0];
  lookAtBasis(f, rx, ry, up);
  assert.ok(near(B.y, ry) && near(B.x, rx) && near(B.y, up, 1e-3));
  // a world direction along the dorsal axis is hull-local +Y
  assert.ok(near(worldToLocalDir(f, up, up), [0, 1, 0], 1e-3));
  // degenerate up (parallel to the nose, zero, NaN) falls back to the old roll
  assert.equal(hullBasis(f, f, x, y), false);
  assert.equal(validUp([0, 0, 0]), null);
  assert.equal(validUp([NaN, 0, 1]), null);
  assert.equal(validUp(undefined), null);
});

test('attitude between frames is a rotation slerp: a 90-degree roll passes through 45', () => {
  const f = [0, 0, 0], u = [0, 0, 0];
  interpAttitude([1, 0, 0], [0, 0, 1], [1, 0, 0], [0, -1, 0], 0.5, f, u);
  assert.ok(near(f, [1, 0, 0]));
  assert.ok(near(u, [0, -Math.SQRT1_2, Math.SQRT1_2]));
  interpAttitude([1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], 0, f, u);
  assert.ok(near(f, [1, 0, 0]) && near(u, [0, 0, 1]));
  interpAttitude([1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], 1, f, u);
  assert.ok(near(f, [0, 1, 0]) && near(u, [0, 0, 1]));
  // a yaw and a roll together stay orthonormal all the way
  for (let t = 0; t <= 1; t += 0.125) {
    interpAttitude([1, 0, 0], [0, 0, 1], norm([0.5, 0.5, 0.7071]), norm([-0.5, -0.5, 0.7071]), t, f, u);
    assert.ok(Math.abs(Math.hypot(...f) - 1) < 1e-9 && Math.abs(Math.hypot(...u) - 1) < 1e-9);
    assert.ok(Math.abs(f[0] * u[0] + f[1] * u[1] + f[2] * u[2]) < 1e-9);
  }
});

test('Interpolator: ships get `up` only when the recording has it', () => {
  const ship = (fwd, up) => ({ pos: [0, 0, 0], vel: [0, 0, 0], fwd, ...(up ? { up } : {}), thrust: 0, hull: 100 });
  const withUp = Interpolator.interpolateShips(
    { t: 0, ships: { a: ship([1, 0, 0], [0, 0, 1]) } }, { t: 1, ships: { a: ship([1, 0, 0], [0, -1, 0]) } }, 0.5);
  assert.ok(near(withUp.a.forward, [1, 0, 0]));
  assert.ok(near(withUp.a.up, [0, -Math.SQRT1_2, Math.SQRT1_2]));
  const old = Interpolator.interpolateShips(
    { t: 0, ships: { a: ship([1, 0, 0]) } }, { t: 1, ships: { a: ship([0, 1, 0]) } }, 0.5);
  assert.equal(old.a.up, null);
  assert.ok(near(old.a.forward, [Math.SQRT1_2, Math.SQRT1_2, 0]));
  // same time, same answer (scrubbing)
  const again = Interpolator.interpolateShips(
    { t: 0, ships: { a: ship([1, 0, 0], [0, 0, 1]) } }, { t: 1, ships: { a: ship([1, 0, 0], [0, -1, 0]) } }, 0.5);
  assert.deepEqual(again.a.up, withUp.a.up);
});

test('trace samplers carry the recorded roll and stay null without it', () => {
  const trace = [
    { t: 0, ships: { a: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], up: [0, 0, 1] }, b: { pos: [0, 0, 0], fwd: [1, 0, 0] } } },
    { t: 2, ships: { a: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], up: [0, -1, 0] }, b: { pos: [0, 0, 0], fwd: [1, 0, 0] } } }
  ];
  const st = shipStateAt(trace, 'a', 1);
  assert.ok(near(st.up, [0, -Math.SQRT1_2, Math.SQRT1_2]));
  assert.equal(shipStateAt(trace, 'b', 1).up, null);
  // a reused output object does not keep a stale up
  const out = shipStateAt(trace, 'a', 1);
  assert.equal(shipStateAt(trace, 'b', 1, out).up, null);
  const track = { samples: trace.map(f => ({ t: f.t, state: f.ships.a })) };
  const s = makeSample();
  assert.ok(sampleTrack(track, 1, s));
  assert.ok(near(s.up, [0, -Math.SQRT1_2, Math.SQRT1_2]));
  assert.ok(sampleTrack({ samples: trace.map(f => ({ t: f.t, state: f.ships.b })) }, 1, s));
  assert.equal(s.up, null);
  assert.equal(lerpUp(null, undefined, 0.3, [0, 0, 0]), null);
});
