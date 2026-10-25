import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { assembleHull, HULL_CLASSES, RCS_S, fitScale } from '../src/render/Hulls.js';
import { SceneManager } from '../src/SceneManager.js';
import {
  jetDims, jetFrames, jetInstanceMatrix, jetClearance, jetRoles, jetTorqueAxes, jetTrimTable,
  jetDemands, jetTrims, JET_ROOT, ROLE_RADIAL, ROLE_SIDE, ROLE_BLOCKED
} from '../src/render/Rcs.js';

// Displayed hull sizes (SceneManager.createShip baseSizes x 8)
const BASE = {
  corvette: [0.5, 0.13], frigate: [0.72, 0.15], destroyer: [1.0, 0.22], cruiser: [1.3, 0.3],
  cruiser_torpedo: [1.3, 0.3], battlecruiser: [1.65, 0.3], battleship: [2.0, 0.55],
  dreadnought: [2.6, 0.7], dreadnought_siege: [2.6, 0.7]
};
const close = (a, b, eps, msg = '') => assert.ok(Math.abs(a - b) <= eps, `${msg} ${a} vs ${b}`);

const _hulls = new Map();
function hullOf(type) {
  if (!_hulls.has(type)) {
    const size = { length: BASE[type][0] * 8, width: BASE[type][1] * 8 };
    const kit = assembleHull(type, size, SceneManager.prototype.getEngineConfig.call(null, type, size), 1234);
    const dims = jetDims(size.length);
    const free = jetClearance(kit.rcsMounts, kit.parts, dims.len);
    _hulls.set(type, { kit, size, dims, free, roles: jetRoles(kit.rcsMounts, free, dims.len) });
  }
  return _hulls.get(type);
}

test('every hull class is covered', () => {
  assert.deepEqual(Object.keys(BASE).sort(), [...HULL_CLASSES].sort());
});

for (const type of Object.keys(BASE)) {
  test(`${type}: every plume leaves its nozzle mouth along the mount's axis`, () => {
    const { kit, dims } = hullOf(type);
    const mounts = kit.rcsMounts;
    assert.ok(mounts.length >= 40, `${mounts.length} nozzles`);
    const frames = jetFrames(mounts);
    const M = new THREE.Matrix4();
    const o = new THREE.Vector3(), z = new THREE.Vector3(), x = new THREE.Vector3();
    mounts.forEach((m, k) => {
      for (const layer of [0, 1]) {
        for (const Ie of [0.3, 1]) {
          jetInstanceMatrix(frames[k], dims, layer, Ie, M);
          // plume origin (the cone root) sits on the nozzle exit
          o.set(0, 0, 0).applyMatrix4(M);
          assert.ok(o.distanceTo(m.pos) < 1e-6, `${type} jet ${k} origin ${o.toArray()} vs ${m.pos.toArray()}`);
          // the cone's +Z axis is the mount's outward nozzle axis
          z.setFromMatrixColumn(M, 2);
          const len = z.length();
          z.normalize();
          assert.ok(z.dot(m.dir) > 1 - 1e-9, `${type} jet ${k} axis ${z.toArray()} vs ${m.dir.toArray()}`);
          assert.ok(len > 0 && len <= dims.len + 1e-9);
          // round cross-section of the layer's radius
          x.setFromMatrixColumn(M, 0);
          close(x.length(), layer ? dims.sheath : dims.core, 1e-9, 'radius');
        }
      }
    });
  });

  test(`${type}: plumes are sized to the constant thruster block`, () => {
    const { kit, size, dims } = hullOf(type);
    const block = RCS_S * fitScale(size.length);
    close(dims.block, block, 1e-12);
    close(kit.rcsS, block, 1e-12, 'jets and hull agree on the block size');
    // the sheath leaves the nozzle at about the bell mouth (0.13 block)
    const root = JET_ROOT * dims.sheath, mouth = 0.13 * block;
    assert.ok(root >= 0.8 * mouth && root <= 1.6 * mouth, `root ${root} vs mouth ${mouth}`);
    assert.ok(JET_ROOT * dims.core < root, 'core inside the sheath');
    // a few blocks long: neither dwarfing the block nor vanishing next to it
    assert.ok(dims.len >= 3 * block && dims.len <= 6 * block, `length ${dims.len / block} blocks`);
    assert.ok(dims.flash <= 3 * block, 'nozzle flash');
    // and the same few blocks on a corvette as on a dreadnought
    close(dims.len / block, jetDims(4).len / jetDims(4).block, 1e-9);
  });

  test(`${type}: every nozzle that fires has a clear run for its whole plume`, () => {
    const { kit, dims, free, roles } = hullOf(type);
    const mounts = kit.rcsMounts;
    const firing = (i) => roles[i] === ROLE_RADIAL || roles[i] === ROLE_SIDE;
    mounts.forEach((m, i) => {
      if (firing(i)) assert.ok(free[i] >= dims.len, `${type} nozzle ${i} hits the hull at ${free[i]}`);
      if (roles[i] === ROLE_BLOCKED) assert.ok(free[i] < dims.len);
    });
    // the attitude couples are never blocked: four radial nozzles per station
    const radial = mounts.filter((m, i) => roles[i] === ROLE_RADIAL);
    const stations = new Set(mounts.map(m => Math.round(m.pos.z * 2)));
    assert.equal(radial.length, 4 * Math.round(mounts.length / 20));
    assert.ok(stations.size >= 2);
    // both ends of the hull can turn it and trim it
    const axes = jetTorqueAxes(mounts, roles);
    const table = jetTrimTable(mounts, roles);
    for (const end of [1, -1]) {
      const at = mounts.map((m, i) => i).filter(i => Math.sign(mounts[i].pos.z) === end);
      assert.ok(at.some(i => Math.hypot(axes[3 * i], axes[3 * i + 1]) > 0.45), `${type} attitude jets at ${end}`);
      assert.ok(at.filter(i => Math.hypot(table[3 * i], table[3 * i + 1]) > 0).length >= 4, `${type} trim jets at ${end}`);
    }
  });

  test(`${type}: a pitch is a balanced nose/tail couple; a trim is a pure push`, () => {
    const { kit, roles } = hullOf(type);
    const mounts = kit.rcsMounts;
    const n = mounts.length;
    // nose up: torque about -x
    const d = jetDemands(jetTorqueAxes(mounts, roles), -1, 0, new Float32Array(n));
    const F = new THREE.Vector3(), T = new THREE.Vector3(), r = new THREE.Vector3();
    let nose = 0, tail = 0;
    for (let i = 0; i < n; i++) {
      if (d[i] <= 0) continue;
      assert.equal(roles[i], ROLE_RADIAL);
      const f = mounts[i].dir.clone().multiplyScalar(-d[i]);
      F.add(f);
      T.add(r.copy(mounts[i].pos).cross(f));
      if (mounts[i].pos.z > 0) nose += d[i]; else tail += d[i];
    }
    assert.ok(nose > 0 && tail > 0, 'both stations fire');
    // nose pushed up, tail pushed down; the forces cancel to a pure torque
    // (the dreadnought splits its nose share over two stations)
    assert.ok(Math.abs(F.y) < 0.05 * (nose + tail) && Math.abs(F.x) < 0.05 * (nose + tail), `net force ${F.toArray()}`);
    assert.ok(T.x < 0, `torque ${T.toArray()}`);

    // push the hull toward +x: side (or radial) nozzles facing -x, nose and
    // tail moments balanced
    const table = jetTrimTable(mounts, roles);
    const tr = jetTrims(table, 1, 0, new Float32Array(n));
    const P = new THREE.Vector3(), Q = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      if (tr[i] <= 0) continue;
      const f = mounts[i].dir.clone().multiplyScalar(-tr[i]);
      P.add(f);
      Q.add(r.copy(mounts[i].pos).cross(f));
    }
    assert.ok(P.x > 0.5 && Math.abs(P.y) < 0.05 * P.x, `push ${P.toArray()}`);
    // no pitch/yaw from the push (about the hull origin)
    assert.ok(Math.abs(Q.y) < 0.05 * P.x * Math.abs(mounts[0].pos.z) + 1e-6, `yaw torque ${Q.toArray()}`);
  });
}
