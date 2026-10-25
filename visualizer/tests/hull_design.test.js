import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import {
  assembleHull, HULL_CLASSES, HULL_MATERIAL_KEYS, WEAPON_SIZE, RAD_AREA, WINDOW_S, fitScale, HullKit
} from '../src/render/Hulls.js';
import { SceneManager } from '../src/SceneManager.js';
import { Projectiles } from '../src/render/Projectiles.js';
import { magazineRoundState, updateMagazine, assignLaunches } from '../src/render/HullRigs.js';
import { floodLevel } from '../src/render/HullMarkings.js';
import { HeatFX } from '../src/render/Heat.js';

// Displayed hull sizes (SceneManager.createShip baseSizes x 8)
const BASE = {
  corvette: [0.5, 0.13], frigate: [0.72, 0.15], destroyer: [1.0, 0.22], cruiser: [1.3, 0.3],
  cruiser_torpedo: [1.3, 0.3], battlecruiser: [1.65, 0.3], battleship: [2.0, 0.55],
  dreadnought: [2.6, 0.7], dreadnought_siege: [2.6, 0.7]
};
const FLEET = JSON.parse(readFileSync(new URL('../../data/fleet_ships.json', import.meta.url), 'utf8'));

function sizeOf(type) {
  return { length: BASE[type][0] * 8, width: BASE[type][1] * 8 };
}
const _kits = new Map();
function kitFor(type, faction = 'alpha') {
  const key = `${type}|${faction}`;
  if (!_kits.has(key)) {
    const size = sizeOf(type);
    const cfg = SceneManager.prototype.getEngineConfig.call(null, type, size);
    _kits.set(key, assembleHull(type, size, cfg, 1234, faction));
  }
  return _kits.get(key);
}
const _rayMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
/**
 * Everything solid on a hull, in hull space, for ray tests: the baked
 * parts, the radiator wings (deployed, their widest reach) and the
 * magazine (CT hatches closed, corvette rail rounds). Turret and PD rigs
 * are built in build() with materials, so they are not in here.
 */
function solids(k) {
  if (k._solids) return k._solids;
  const meshes = k.parts.map(p => new THREE.Mesh(p.geom, _rayMat));
  for (const r of k.radiators) {
    r.mesh.updateMatrixWorld(true);
    for (const p of r.kit.parts) {
      const m = new THREE.Mesh(p.geom, _rayMat);
      m.matrixAutoUpdate = false;
      m.matrixWorld.copy(r.mesh.matrixWorld);
      meshes.push(m);
    }
  }
  for (const e of k.magazine) {
    const g = e.kind === 'rack'
      ? new THREE.BoxGeometry(e.s, e.s, e.s * 6.3).translate(0, 0, e.s * 0.25)
      : new THREE.BoxGeometry(e.s * 0.9, e.s * 0.1, e.s * 0.9);
    const m = new THREE.Mesh(g, _rayMat);
    m.position.copy(e.pos);
    m.quaternion.copy(e.quat);
    meshes.push(m);
  }
  for (const m of meshes) if (m.matrixAutoUpdate) m.updateMatrixWorld(true);
  return (k._solids = meshes);
}
const FACE_N = { '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0] };

/** Outward axis of a mount rotated `angle` about Z (0 = dorsal). */
function mountSide(angle) {
  const x = -Math.sin(angle), y = Math.cos(angle);
  if (Math.abs(y) >= Math.abs(x)) return y > 0 ? 'dorsal' : 'ventral';
  return x > 0 ? 'port' : 'starboard';
}

test('every class assembles headless with known materials and finite geometry', () => {
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    assert.ok(k.parts.length > 100, `${type}: ${k.parts.length} parts`);
    for (const p of k.parts) {
      assert.ok(HULL_MATERIAL_KEYS.includes(p.mat), `${type}: unknown material ${p.mat}`);
      for (const v of p.geom.attributes.position.array) assert.ok(Number.isFinite(v), `${type}: NaN vertex`);
    }
  }
});

test('mounts match data/fleet_ships.json: kinds, counts, slots, sides and stations', () => {
  const kindOf = { coilgun_mk3: 'coilgun', heavy_coilgun_mk3: 'heavy' };
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    const L = k.L;
    const weapons = FLEET.ships[type].weapons;
    const spinal = /spinal|siege/.test(weapons[0].type);
    assert.equal(k.hasSpinal, spinal, `${type}: spinal flag`);
    const guns = weapons.filter(w => kindOf[w.type]);
    assert.deepEqual(k.turretRigs.map(r => r.kind), guns.map(w => kindOf[w.type]), `${type}: battery kinds in slot order`);
    const pds = weapons.filter(w => w.type === 'pd_laser');
    assert.equal(k.pdRigs.length, pds.length, `${type}: PD count`);
    const launchers = weapons.filter(w => w.type === 'torpedo_launcher');
    const rounds = launchers.reduce((n, w) => n + (w.magazine ?? FLEET.weapon_types.torpedo_launcher.magazine), 0);
    assert.equal(k.magazine.length, rounds, `${type}: magazine rounds drawn`);
    // sides and fore/aft station (x: 0 nose .. 1 tail) where the data gives them
    const check = (rig, w, what) => {
      const mp = w.mount_position;
      if (!mp) return;
      if (mp.side !== 'centerline') assert.equal(mountSide(rig.angle), mp.side, `${type} ${what} side`);
      const zData = L * (0.5 - mp.x);
      assert.ok(Math.abs(rig.z - zData) <= L * 0.2, `${type} ${what}: z ${(rig.z / L).toFixed(2)}L vs data ${(zData / L).toFixed(2)}L`);
    };
    k.turretRigs.forEach((r, i) => check(r, guns[i], `weapon ${i}`));
    k.pdRigs.forEach((r, i) => check(r, pds[i], `pd_${i}`));
  }
});

test('a weapon is drawn at one physical size on every hull', () => {
  const sizes = { coilgun: [], heavy: [], pd: [] };
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    for (const r of k.turretRigs) sizes[r.kind].push(r.s);
    for (const r of k.pdRigs) sizes.pd.push(r.s);
  }
  for (const [kind, s] of Object.entries(sizes)) {
    assert.ok(s.length, kind);
    // only the mild fitScale allowance separates a corvette's from a dreadnought's
    assert.ok(Math.max(...s) / Math.min(...s) <= 1.25 / 0.9 + 1e-9, `${kind}: ${Math.min(...s)}..${Math.max(...s)}`);
    for (const v of s) assert.ok(v >= WEAPON_SIZE[kind] * 0.9 - 1e-9 && v <= WEAPON_SIZE[kind] * 1.25 + 1e-9);
  }
  // on one hull a heavy battery is visibly bigger than the standard one
  for (const type of ['battleship', 'dreadnought']) {
    const k = kitFor(type);
    const heavy = k.turretRigs.find(r => r.kind === 'heavy'), std = k.turretRigs.find(r => r.kind === 'coilgun');
    assert.ok(heavy.s > std.s * 1.3, type);
  }
});

test('turret rigs keep the muzzle geometry Projectiles.localMuzzle assumes', () => {
  const mats = Object.fromEntries(HULL_MATERIAL_KEYS.map(k => [k, new THREE.MeshBasicMaterial()]));
  for (const type of ['frigate', 'battleship']) {
    const k = kitFor(type);
    for (const r of k.turretRigs) {
      const rig = k._buildTurretRig(r, mats);
      const s = rig.s;
      assert.ok(Math.abs(rig.yaw.position.y - 0.2 * s) < 1e-9);
      assert.ok(Math.abs(rig.pitch.position.y - 0.32 * s) < 1e-9 && Math.abs(rig.pitch.position.z - 0.15 * s) < 1e-9);
      assert.equal(rig.barrels.length, 2);
      for (const b of rig.barrels) {
        const box = new THREE.Box3();
        for (const m of b.children) { m.geometry.computeBoundingBox(); box.union(m.geometry.boundingBox); }
        assert.ok(Math.abs(box.max.z - 1.4 * s) < 0.02 * s, `${type} ${r.kind}: barrel tip ${box.max.z / s} s`);
      }
    }
  }
});

test('spinal muzzles sit on real barrels where Projectiles fires them from', () => {
  const pick = { s3: new THREE.Vector3() };
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    if (!k.hasSpinal) { assert.equal(k.spinalMuzzle, null, type); continue; }
    assert.ok(k.spinalMuzzle, `${type}: spinal muzzle exported`);
    // the userData a built hull carries (Hulls.build exports spinalMuzzle;
    // Projectiles may take it directly or fall back to its per-class table)
    const ship = { userData: { size: { length: k.L, width: k.W }, shipType: type, spinalMuzzle: k.spinalMuzzle } };
    const out = Projectiles.prototype.localMuzzle.call(pick, ship, { spinal: true }, new THREE.Vector3(0, 0, 1),
      { pos: new THREE.Vector3(), dir: new THREE.Vector3() });
    assert.ok(out.pos.distanceTo(k.spinalMuzzle) < 1e-6, `${type}: Projectiles muzzle ${out.pos.toArray()} vs hull ${k.spinalMuzzle.toArray()}`);
    // a barrel part reaches the muzzle (within a tenth of a hull width)
    const hit = k.parts.some(p => (p.mat === 'barrel' || p.mat === 'dark')
      && p.geom.boundingBox.distanceToPoint(k.spinalMuzzle) < k.W * 0.02
      && p.geom.boundingBox.max.z >= k.spinalMuzzle.z - k.L * 0.03);
    assert.ok(hit, `${type}: no barrel at the spinal muzzle`);
    // and the bore is clear: no plating in front of the muzzle
    const rc = new THREE.Raycaster(k.spinalMuzzle.clone().add(new THREE.Vector3(0, 0, k.L * 0.002)),
      new THREE.Vector3(0, 0, 1), 0, k.L * 0.3);
    const block = rc.intersectObjects(solids(k), false)[0];
    assert.ok(!block, `${type}: spinal bore blocked ${block?.distance.toFixed(2)} ahead of the muzzle`);
  }
  // the dreadnought's spinal is a real gun now, not an accent bar
  const dn = kitFor('dreadnought');
  assert.ok(dn.parts.some(p => p.mat === 'barrel' && p.geom.boundingBox.min.z > dn.L * 0.35
    && Math.abs(p.center.y - dn.spinalMuzzle.y) < dn.W * 0.01), 'dreadnought keel barrel');
});

test('no painted crayon prow: livery stays off the bow tip and reaches the aft half', () => {
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    const tip = Math.max(...k.parts.map(p => p.geom.boundingBox.max.z));
    const paint = k.parts.filter(p => p.mat === 'paint');
    const paintTip = Math.max(...paint.map(p => p.geom.boundingBox.max.z));
    assert.ok(paintTip < tip - 0.05 * k.L, `${type}: paint reaches the bow tip`);
    // faction livery where the chase camera looks: drive band, cassettes
    assert.ok(paint.some(p => p.center.z < -0.3 * k.L), `${type}: no livery on the aft half`);
  }
});

test('faction livery differs in pattern, not only colour', () => {
  const count = (type, f) => kitFor(type, f).parts.filter(p => p.mat === 'paint').length;
  for (const type of ['corvette', 'destroyer', 'cruiser', 'dreadnought']) {
    assert.ok(count(type, 'alpha') > count(type, 'beta'), `${type}: alpha pinstripes vs beta band`);
  }
});

test('radiators share the common array area instead of scaling with the beam', () => {
  const totals = HULL_CLASSES.map(type => {
    const k = kitFor(type);
    const total = k.radiators.reduce((a, r) => a + r.span * r.len, 0);
    const target = RAD_AREA * fitScale(k.L) ** 2;
    assert.ok(total <= target * 1.001 && total >= target * 0.6, `${type}: ${total.toFixed(2)} u^2 vs ${target.toFixed(2)}`);
    for (const r of k.radiators) assert.equal(r.mesh.userData.stow, 'telescope');
    return total;
  });
  // was ~110x between corvette and dreadnought
  assert.ok(Math.max(...totals) / Math.min(...totals) < 3, totals.join(' '));
});

test('windows are a fixed size: a dreadnought shows more of them, not bigger ones', () => {
  const winDims = (type) => {
    const k = kitFor(type);
    const w = k.parts.filter(p => p.mat === 'window');
    const size = w.map(p => { const s = new THREE.Vector3(); p.geom.boundingBox.getSize(s); return Math.max(s.x, s.y, s.z); });
    return { n: w.length, max: Math.max(...size) };
  };
  const ff = winDims('frigate'), dn = winDims('dreadnought');
  assert.ok(dn.max / ff.max < 1.45, `window size ratio ${dn.max / ff.max}`);
  assert.ok(dn.max < WINDOW_S * 1.25 * 1.6, dn.max);
  assert.ok(dn.n > ff.n * 3, `${dn.n} vs ${ff.n} windows`);
});

test('no nav lamp sits on a hull code', () => {
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    for (const m of k.markings) {
      const ax = m.face[1];
      for (const l of k.lamps) {
        const off = ax === 'x' ? Math.abs(l.pos.x - m.x) : Math.abs(l.pos.y - m.y);
        const across = ax === 'x' ? Math.abs(l.pos.y - m.y) : Math.abs(l.pos.x - m.x);
        const along = Math.abs(l.pos.z - m.z);
        // the lamp's glow counts: a lamp beside the code still blots it
        const inside = off < k.W * 0.06 && across < m.h * 0.5 + l.size && along < m.w * 0.5 + l.size * 0.5;
        assert.ok(!inside, `${type}: lamp at ${l.pos.toArray().map(v => v.toFixed(2))} on code at z ${m.z.toFixed(2)}`);
      }
    }
  }
});

test('no radiator cassette or wing covers a hull code', () => {
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    for (const m of k.markings) {
      for (const r of k.radiators) {
        const p = r.mesh.position;
        const alongZ = Math.abs(p.z - m.z) < r.len / 2 + m.w / 2;
        const near = Math.hypot(p.x - m.x, p.y - m.y) < k.W * 0.12;
        assert.ok(!(alongZ && near), `${type}: radiator at z ${p.z.toFixed(2)} over the code at z ${m.z.toFixed(2)}`);
      }
    }
  }
});

test('every hull is one connected structure (no floating masts or parts)', () => {
  for (const type of HULL_CLASSES) {
    const k = kitFor(type);
    const n = k.parts.length;
    const eps = k.L * 0.004;
    const boxes = k.parts.map(p => p.geom.boundingBox.clone().expandByScalar(eps));
    const parent = Array.from({ length: n }, (_, i) => i);
    const find = (i) => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) if (boxes[i].intersectsBox(boxes[j])) parent[find(i)] = find(j);
    }
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(i);
    }
    const sorted = [...groups.values()].sort((a, b) => b.length - a.length);
    const strays = sorted.slice(1).flat().map(i => `${k.parts[i].mat}@${k.parts[i].center.toArray().map(v => (v / k.L).toFixed(2))}`);
    assert.equal(groups.size, 1, `${type}: floating parts ${strays.slice(0, 6).join(' ')}`);
  }
});

test('torpedo carriers carry their whole magazine, one bank per launcher', () => {
  const ct = kitFor('cruiser_torpedo');
  const banks = new Map();
  for (const e of ct.magazine) banks.set(e.bank, (banks.get(e.bank) || 0) + 1);
  assert.deepEqual([...banks.values()], [12, 12, 12, 12]);
  assert.ok(ct.magazine.every(e => e.kind === 'cell' && e.slide));
  const k = kitFor('corvette');
  assert.equal(k.magazine.length, 8);
  assert.ok(k.magazine.every(e => e.kind === 'rack'));
  // the rails ride both chines, where the focus and chase cameras see them
  assert.equal(k.magazine.filter(e => e.pos.x > 0).length, 4);
});

test('magazine rounds empty on battle time and come back on a backward seek', () => {
  assert.deepEqual({ ...magazineRoundState(undefined, 100) }, { open: 0, lamp: 0, gone: false });
  assert.deepEqual({ ...magazineRoundState(30, 20) }, { open: 0, lamp: 0, gone: false });
  const opening = magazineRoundState(30, 28);
  assert.ok(opening.open > 0.4 && opening.open < 0.6 && opening.lamp === 1 && !opening.gone);
  assert.deepEqual({ ...magazineRoundState(30, 30.2) }, { open: 1, lamp: 1, gone: true });
  assert.deepEqual({ ...magazineRoundState(30, 300) }, { open: 1, lamp: 2, gone: true });

  // a two-round rack: the first launch empties round 0, the second round 1
  const mk = () => {
    const body = { m: [new THREE.Matrix4(), new THREE.Matrix4()], setMatrixAt(i, m) { this.m[i].copy(m); }, instanceMatrix: {} };
    return {
      kind: 'rack', entries: [{ bank: 0 }, { bank: 0 }], order: [0, 1], banks: 1,
      base: [new THREE.Matrix4().makeTranslation(1, 0, 0), new THREE.Matrix4().makeTranslation(-1, 0, 0)],
      body, lamp: null, state: new Float32Array(2).fill(-1)
    };
  };
  const mag = mk();
  const times = assignLaunches([{ t: 10 }, { t: 20 }], 1);
  const shown = (i) => mag.body.m[i].determinant() !== 0;
  updateMagazine(mag, times, 5);
  assert.ok(shown(0) && shown(1));
  updateMagazine(mag, times, 15);
  assert.ok(!shown(0) && shown(1));
  updateMagazine(mag, times, 25);
  assert.ok(!shown(0) && !shown(1));
  assert.equal(updateMagazine(mag, times, 26), false, 'unchanged rounds are not rewritten');
  updateMagazine(mag, times, 12);   // seek back
  assert.ok(!shown(0) && shown(1));
  updateMagazine(mag, times, 0);
  assert.ok(shown(0) && shown(1));
});

test('floodlights burn on a sound hull and brown out with it', () => {
  assert.equal(floodLevel(10, 1, false), 1);
  assert.ok(floodLevel(10, 0.7, false) < 1 && floodLevel(10, 0.7, false) > 0.7);
  let dim = 1;
  for (let t = 0; t < 20; t += 0.05) dim = Math.min(dim, floodLevel(t, 0.5, true, 0.3));
  assert.ok(dim < 0.1);
  assert.equal(floodLevel(3.3, 0.4, false, 0.2), floodLevel(3.3, 0.4, false, 0.2));
});

test('stowed radiators telescope into their cassette instead of standing up as sails', () => {
  const k = new HullKit({ length: 8, width: 1.76 }, 3);
  const wing = k.radiator(1.5, 2, 0.6, 0, -2, -Math.PI / 2);
  const ship = new THREE.Group();
  ship.userData = { shipId: 'x', radiators: [wing], radiatorMaterials: [] };
  const fx = new HeatFX({ currentTime: 0, loader: null });
  fx.updateShip(ship, { radiatorsExtended: false });
  assert.equal(wing.rotation.z, wing.userData.deployedAngle);
  assert.ok(Math.abs(wing.scale.y - wing.userData.stowScale) < 1e-9);
  fx.updateShip(ship, { radiatorsExtended: true });
  assert.equal(wing.rotation.z, wing.userData.deployedAngle);
  assert.equal(wing.scale.y, 1);
  // DamageFX's buckle (rotation.x) is left alone
  wing.rotation.x = 0.3;
  fx.updateShip(ship, { radiatorsExtended: false });
  assert.equal(wing.rotation.x, 0.3);
});
