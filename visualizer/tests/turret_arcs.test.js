import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  arcsFor, azElToLocal, cellOf, isBlocked, clampToArc, slewDir, slewProgress, mountAngles, mountDir,
  turAim, assignPdSegments, hullExit, hasTur
} from '../src/render/TurretArcs.js';
import { cellIndex, TRAVERSE_DEG_S, AZ_CELLS, EL_CELLS, packMask } from '../src/render/MountLimits.js';
import { fallbackGunAim, fallbackPdAim, updateHullDetails } from '../src/render/HullRigs.js';
import { Interpolator } from '../src/Interpolator.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const deg = (r) => r * 180 / Math.PI;
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

test('body az/el <-> hull-local: nose +Z, port +X, dorsal +Y', () => {
  const v = V(0, 0, 0);
  assert.ok(azElToLocal(0, 0, v).distanceTo(V(0, 0, 1)) < 1e-9);       // nose
  assert.ok(azElToLocal(90, 0, v).distanceTo(V(1, 0, 0)) < 1e-9);      // port
  assert.ok(azElToLocal(0, 90, v).distanceTo(V(0, 1, 0)) < 1e-9);      // dorsal
  assert.ok(azElToLocal(-90, 0, v).distanceTo(V(-1, 0, 0)) < 1e-9);    // starboard
  assert.equal(cellOf(azElToLocal(37, -22, v)), cellIndex(37, -22));
  assert.equal(cellOf(V(0, 0, 5)), cellIndex(0, 0));                    // not unit length
});

test('arcs come from data/hull_mounts.json in its mount order, with the engine\'s traverse rates', () => {
  const dn = arcsFor('dreadnought');
  assert.deepEqual(dn.mounts.map(m => m.slot),
    ['weapon_0', 'weapon_1', 'weapon_2', 'weapon_3', 'weapon_4', 'weapon_5', 'weapon_6', 'pd_0', 'pd_1', 'pd_2', 'pd_3']);
  assert.ok(near(deg(dn.bySlot.get('pd_0').rate), 120) && near(deg(dn.bySlot.get('weapon_1').rate), 15));
  assert.ok(near(deg(arcsFor('cruiser').bySlot.get('weapon_1').rate), 30));
  assert.deepEqual(TRAVERSE_DEG_S, { pd_laser: 120, light_coilgun_mk3: 45, coilgun_mk3: 30, heavy_coilgun_mk3: 15 });
  assert.equal(arcsFor('no_such_hull'), null);
  // the cruiser's port battery: fires to port, never to starboard through the hull
  const port = arcsFor('cruiser').bySlot.get('weapon_1');
  assert.ok(port.axis.distanceTo(V(1, 0, 0)) < 1e-6);
  assert.equal(isBlocked(port, V(1, 0, 0)), false);
  assert.equal(isBlocked(port, V(-1, 0, 0)), true);
});

test('a blocked aim is clamped to the edge of the arc, an open one is left alone', () => {
  const port = arcsFor('cruiser').bySlot.get('weapon_1');
  const out = V(0, 0, 0);
  const open = V(1, 0.3, 0.2).normalize();
  assert.equal(clampToArc(port, open, out), false);
  assert.ok(out.distanceTo(open) < 1e-12);
  // a target on the starboard bow: the mount stops at the edge of its arc
  const want = V(-0.5, 0, 0.866);
  assert.equal(clampToArc(port, want, out), true);
  assert.equal(isBlocked(port, out), false, 'clamped aim is open');
  assert.ok(deg(out.angleTo(want)) < 40, `edge is the nearest open direction (${deg(out.angleTo(want)).toFixed(1)} deg away)`);
  assert.ok(out.z > 0.7, 'trained forward along the hull, not swung away');
  // every blocked direction of every mount of every hull clamps to an open cell
  for (const type of ['corvette', 'frigate', 'destroyer', 'battleship', 'dreadnought', 'cruiser_torpedo']) {
    for (const m of arcsFor(type).mounts) {
      if (m.kind === 'launcher') continue;
      for (let az = -175; az <= 180; az += 35) for (let el = -80; el <= 80; el += 20) {
        clampToArc(m, azElToLocal(az, el, V(0, 0, 0)), out);
        assert.equal(isBlocked(m, out), false, `${type} ${m.slot} az ${az} el ${el}`);
      }
    }
  }
  // same input, same answer (the nearest-cell cache does not change results)
  const again = V(0, 0, 0);
  clampToArc(port, want, again);
  assert.ok(again.distanceTo(out) < 1e-12 || clampToArc(port, want, out) && again.distanceTo(out) < 1e-12);
});

test('slew: great circle at the class rate, arriving as the shot fires', () => {
  const a = V(0, 0, 1), b = V(1, 0, 0), out = V(0, 0, 0);
  assert.ok(slewDir(a, b, 0.5, out).distanceTo(V(Math.SQRT1_2, 0, Math.SQRT1_2)) < 1e-9);
  assert.ok(slewDir(a, b, 0, out).distanceTo(a) < 1e-12 && slewDir(a, b, 1, out).distanceTo(b) < 1e-12);
  assert.ok(near(slewDir(a, V(0, 0, -1), 0.5, out).length(), 1), 'antipodal slew stays a unit vector');
  // 90 degrees at 30 deg/s takes 3 s and ends at the shot
  const rate = 30 * Math.PI / 180;
  assert.equal(slewProgress(Math.PI / 2, rate, 10, 6.9), 0);
  assert.ok(near(slewProgress(Math.PI / 2, rate, 10, 8.5), 0.5));
  assert.equal(slewProgress(Math.PI / 2, rate, 10, 10), 1);
  assert.equal(slewProgress(Math.PI / 2, rate, 10, 12), 1);
});

test('mount yaw/pitch round-trips for dorsal, ventral, flank and shoulder mounts', () => {
  const out = new THREE.Vector2(), v = V(0, 0, 0);
  for (const rz of [0, Math.PI, -Math.PI / 2, Math.PI / 2, -Math.PI / 3, Math.PI * 0.75]) {
    for (const [yaw, pitch] of [[0, 0.12], [1.1, 0.4], [-2.6, 1.2], [3.0, -0.03]]) {
      mountAngles(rz, mountDir(rz, yaw, pitch, v), out);
      assert.ok(near(out.x, yaw, 1e-9) && near(out.y, pitch, 1e-9), `rz ${rz} yaw ${yaw}`);
    }
  }
  // a dorsal mount aimed at the nose has yaw 0; a port-flank mount aimed to port looks straight up its axis
  assert.ok(near(mountAngles(0, V(0, 0, 1), out).x, 0));
  assert.ok(near(mountAngles(-Math.PI / 2, V(1, 0, 0), out).y, Math.PI / 2));
});

test('recorded `tur` aims: slerped between frames, state switches at its frame', () => {
  const out = V(0, 0, 0);
  const tur = { a: [[0, 0, 2], [90, 0, 1]], b: [[0, 0, 2], [0, 0, 3]], f: 0.5 };
  assert.equal(turAim(tur, 0, out), 2);
  assert.ok(out.distanceTo(V(0, 0, 1)) < 1e-9);
  assert.equal(turAim(tur, 1, out), 1);                       // still frame a's state at f < 1
  assert.ok(out.distanceTo(V(Math.SQRT1_2, 0, Math.SQRT1_2)) < 1e-9, 'halfway from port to the nose');
  assert.equal(turAim({ ...tur, f: 1 }, 1, out), 3);
  assert.ok(out.distanceTo(V(0, 0, 1)) < 1e-9);
  assert.equal(turAim(tur, 5, out), -1);                      // no entry: caller falls back
  assert.equal(turAim(null, 0, out), -1);
  // the Interpolator hands both frames' lists through
  const ship = (tur) => ({ pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], ...(tur ? { tur } : {}) });
  const st = Interpolator.interpolateShips({ t: 0, ships: { a: ship([[10, 0, 1]]) } }, { t: 1, ships: { a: ship([[30, 0, 2]]) } }, 0.25);
  assert.deepEqual(st.a.tur, { a: [[10, 0, 1]], b: [[30, 0, 2]], f: 0.25 });
  assert.equal(turAim(st.a.tur, 0, out), 1);
  assert.ok(near(deg(Math.atan2(out.x, out.z)), 15, 1e-6));
  assert.equal(Interpolator.interpolateShips({ t: 0, ships: { a: ship() } }, { t: 1, ships: { a: ship() } }, 0.5).a.tur, null);
  assert.equal(hasTur({ simTrace: [{ t: 0, ships: { a: ship([[0, 0, 0]]) } }] }), true);
  assert.equal(hasTur({ simTrace: [{ t: 0, ships: { a: ship() } }] }), false);
});

// A minimal rig set posed by updateHullDetails
function fakeShip(type, tur) {
  const mk = (rz, x, y) => { const g = new THREE.Group(); g.position.set(x, y, 0); g.rotation.z = rz; return g; };
  const gun = (slot, rz, x, y) => ({ mount: mk(rz, x, y), yaw: new THREE.Group(), pitch: new THREE.Group(), s: 0.6, slot, pitchZ: 0, yawAngle: 0, pitchAngle: 0 });
  const pd = (name, rz, x, y) => ({ name, mount: mk(rz, x, y), ball: new THREE.Object3D(), s: 0.4, phase: 0.3, yawAngle: 0, pitchAngle: 0 });
  const ship = new THREE.Group();
  ship.userData = {
    shipId: 'alpha_1', faction: 'alpha', shipType: type,
    gunTurrets: [gun('weapon_1', -Math.PI / 2, 1, 0), gun('weapon_2', Math.PI / 2, -1, 0)],
    pdTurrets: [pd('pd_0', 0, 0, 1), pd('pd_1', Math.PI, 0, -1)]
  };
  ship.updateMatrixWorld(true);
  return ship;
}

test('turrets are posed from `tur` when the recording has it, deterministically under scrubbing', () => {
  const ship = fakeShip('cruiser');
  const scene = { currentTime: 12, ships: new Map([['alpha_1', ship]]), loader: null };
  // cruiser mount order: weapon_0 (spinal), weapon_1 (port), weapon_2 (starboard), pd_0, pd_1
  const tur = { a: [[0, 0, 2], [90, 0, 2], [-90, 10, 1], [0, 90, 0], [180, -30, 2]], b: [[0, 0, 2], [60, 0, 2], [-90, 40, 2], [0, 90, 0], [180, -30, 3]], f: 0.5 };
  updateHullDetails(ship, { hull: 100, tur }, scene);
  const [port, stbd] = ship.userData.gunTurrets, [pd0, pd1] = ship.userData.pdTurrets;
  // port battery: aim 75 degrees to port = 15 degrees forward of its own axis
  const v = mountDir(port.mount.rotation.z, port.yawAngle, port.pitchAngle, V(0, 0, 0));
  assert.ok(v.distanceTo(azElToLocal(75, 0, V(0, 0, 0))) < 1e-6);
  assert.equal(port.arcState, 2);
  assert.equal(stbd.arcState, 1);
  assert.ok(mountDir(stbd.mount.rotation.z, stbd.yawAngle, stbd.pitchAngle, V(0, 0, 0)).distanceTo(azElToLocal(-90, 25, V(0, 0, 0))) < 1e-6);
  // PD: dorsal ball straight up its axis, ventral ball astern-low, masked at the next frame
  assert.ok(near(pd0.pitchAngle, Math.PI / 2, 1e-6));
  assert.ok(mountDir(Math.PI, pd1.yawAngle, pd1.pitchAngle, V(0, 0, 0)).distanceTo(azElToLocal(180, -30, V(0, 0, 0))) < 1e-6);
  assert.equal(pd1.arcState, 2);
  const pose = [port.yawAngle, port.pitchAngle, stbd.yawAngle, stbd.pitchAngle, pd0.yawAngle, pd1.yawAngle, pd1.pitchAngle];
  // scrub away and back: same pose (no slew state anywhere)
  updateHullDetails(ship, { hull: 100, tur: { ...tur, f: 1 } }, { ...scene, currentTime: 13 });
  assert.equal(pd1.arcState, 3);
  updateHullDetails(ship, { hull: 100, tur }, scene);
  assert.deepEqual([port.yawAngle, port.pitchAngle, stbd.yawAngle, stbd.pitchAngle, pd0.yawAngle, pd1.yawAngle, pd1.pitchAngle], pose);
});

test('without `tur` a gun never trains through its own hull and slews at its class rate', () => {
  const ship = fakeShip('cruiser');
  const enemy = new THREE.Group();
  enemy.userData = { faction: 'beta' };
  enemy.position.set(-400, 0, 300);            // starboard bow
  const other = new THREE.Group();
  other.userData = { faction: 'beta' };
  other.position.set(300, 0, 400);             // port bow
  const [port] = ship.userData.gunTurrets;
  const arc = arcsFor('cruiser').bySlot.get('weapon_1');
  const out = new THREE.Vector2(), v = V(0, 0, 0);
  // updateHullDetails computes the inverse ship matrix the aim helpers read
  const scene = { currentTime: 0, ships: new Map([['alpha_1', ship], ['b1', enemy]]), loader: null };
  updateHullDetails(ship, { hull: 100 }, scene);
  // the only enemy is to starboard: the port battery is masked and rests at its arc's edge
  assert.equal(port.arcState, 3);
  mountDir(port.mount.rotation.z, port.yawAngle, port.pitchAngle, v);
  assert.equal(isBlocked(arc, v), false);
  assert.ok(v.x > -0.2, 'not aimed across the hull to starboard');
  // slewing from the masked edge to a target on the port bow that fires at t = 10
  const dirs = [];
  for (const t of [0, 7, 8, 9, 10, 11]) {
    fallbackGunAim(port, arc, enemy, other, 10, t, out);
    dirs.push(mountDir(port.mount.rotation.z, out.x, out.y, V(0, 0, 0)));
  }
  const want = other.position.clone().sub(port.mount.position).normalize();
  assert.ok(dirs[4].distanceTo(want) < 1e-6, 'on the new target when it fires');
  assert.ok(dirs[5].distanceTo(want) < 1e-6);
  // one second before the shot it is one second of traverse (30 degrees) short
  assert.ok(Math.abs(deg(dirs[3].angleTo(want)) - 30) < 6, `${deg(dirs[3].angleTo(want)).toFixed(1)} deg short at t-1`);
  // pure in t
  fallbackGunAim(port, arc, enemy, other, 10, 8, out);
  assert.ok(mountDir(port.mount.rotation.z, out.x, out.y, V(0, 0, 0)).distanceTo(dirs[2]) < 1e-12);
});

test('without `tur` a PD ball arrives on its round as the dwell starts, tracks it, and never aims into the hull', () => {
  const ship = fakeShip('cruiser');
  const [pd0] = ship.userData.pdTurrets;
  const arc = arcsFor('cruiser').bySlot.get('pd_0');
  // a round high on the port bow from t = 20 to 24, another below the keel at t = 40
  const where = { r1: V(200, 300, 400), r2: V(0, -500, 100) };
  const scene = {
    currentTime: 0, ships: new Map([['alpha_1', ship]]), loader: null,
    weapons: { positionAt: (type, id, t, out) => out.copy(where[id]) }
  };
  updateHullDetails(ship, { hull: 100 }, scene);
  const segs = [{ start: 20, end: 24, targetType: 'torpedo', targetId: 'r1' }, { start: 40, end: 42, targetType: 'torpedo', targetId: 'r2' }];
  const out = new THREE.Vector2();
  const aimAt = (t) => { const m = fallbackPdAim(scene, pd0, arc, segs, t, out); return { d: mountDir(0, out.x, out.y, V(0, 0, 0)), m }; };
  const want = where.r1.clone().sub(pd0.mount.position).normalize();
  assert.ok(aimAt(20).d.distanceTo(want) < 1e-6, 'on the round as the beam lights');
  assert.ok(aimAt(22).d.distanceTo(want) < 1e-6 && aimAt(24.9).d.distanceTo(want) < 1e-6, 'holds it while the beam is lit');
  const before = aimAt(19.9).d.angleTo(want);
  assert.ok(before > 0 || aimAt(19).d.angleTo(want) > 0, 'was slewing onto it');
  assert.ok(deg(aimAt(19.9).d.angleTo(want)) <= 12 + 1e-6, 'within 0.1 s of traverse (120 deg/s) just before');
  // the second round is under the keel: the dorsal ball cannot bear, it rests at the edge of its arc
  const low = aimAt(41);
  assert.equal(low.m, true);
  assert.equal(isBlocked(arc, low.d), false);
  assert.ok(low.d.y > -0.2, 'not aimed down through the hull');
  // pure in t: any order of evaluation gives the same pose
  const a = aimAt(21.3).d.clone(); aimAt(5); aimAt(41);
  assert.ok(aimAt(21.3).d.distanceTo(a) < 1e-12);
});

test('old recordings: a dwell whose line is blocked moves to a PD mount with a clear line', () => {
  const pds = arcsFor('cruiser').mounts.filter(m => m.kind === 'pd');
  const up = V(0.1, 0.9, 0.2).normalize(), down = V(0, -1, 0);
  const segs = [
    { turret: 'pd_0', start: 0, end: 5, d: up },       // dorsal turret, round above: stays
    { turret: 'pd_0', start: 1, end: 6, d: down },     // dorsal turret, round below: moves to the ventral one
    { turret: 'pd_1', start: 2, end: 3, d: up },       // ventral turret, round above: moves to the dorsal one
    { turret: 'pd_1', start: 9, end: 9, d: null }      // direction unknown: stays
  ];
  const res = assignPdSegments(segs, pds, s => s.d);
  assert.deepEqual(res.map(r => r.turret), ['pd_0', 'pd_1', 'pd_0', 'pd_1']);
  assert.ok(res.every(r => !r.masked));
  // order of the input list does not matter (processed by start time)
  const rev = assignPdSegments(segs.slice().reverse(), pds, s => s.d).reverse();
  assert.deepEqual(rev, res);
  // no mount can bear (all masks blocked there): flagged, drawn from outside the hull
  const blind = pds.map(m => ({ ...m, mask: new Uint8Array(AZ_CELLS * EL_CELLS).fill(1) }));
  assert.deepEqual(assignPdSegments([segs[0]], blind, s => s.d), [{ turret: 'pd_0', masked: true }]);
  // the corvette's single dorsal turret cannot reach a round under the keel
  const cv = arcsFor('corvette').mounts.filter(m => m.kind === 'pd');
  assert.deepEqual(assignPdSegments([{ turret: 'pd_0', start: 0, end: 1 }], cv, () => down), [{ turret: 'pd_0', masked: true }]);
  // such a beam starts where its line leaves the hull's bounding ellipsoid
  const t = hullExit(V(0, 0.5, 0), V(0, -1, 0), 2, 8);
  assert.ok(near(t, 0.5 + 1.6), `exit at ${t}`);
  assert.equal(hullExit(V(0, 5, 0), V(0, 1, 0), 2, 8), 0);
  assert.equal(packMask(new Uint8Array(AZ_CELLS * EL_CELLS).fill(1)).length, 648);
});

test('pdPlan stamps every dwell from the recording alone and honours the recorded roll', async () => {
  const { pdPlan } = await import('../src/render/TurretArcs.js');
  const ship = new THREE.Group();
  ship.userData = { shipType: 'cruiser' };
  const mk = (up) => {
    const st = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], ...(up ? { up } : {}) };
    const segs = [
      { shipId: 'a', turret: 'pd_0', targetType: 'torpedo', targetId: 'above', start: 1, end: 3 },
      { shipId: 'a', turret: 'pd_0', targetType: 'torpedo', targetId: 'below', start: 2, end: 4 }
    ];
    return {
      SCALE: 0.001, ships: new Map([['a', ship]]),
      loader: { simTrace: [{ t: 0, ships: { a: st } }, { t: 10, ships: { a: st } }], pdBeamSegments: segs },
      // world +Z is "above" for a ship with up = +Z; world +Y for the old implicit roll
      weapons: { timeline: {}, positionAt: (type, id, t, out) => out.set(0, 0, id === 'above' ? 50 : -50) }
    };
  };
  // engine attitude recorded: up = +Z, so the round at world +Z is over the dorsal turret
  const rolled = mk([0, 0, 1]);
  const plan = pdPlan(rolled);
  assert.deepEqual(rolled.loader.pdBeamSegments.map(s => s.arcTurret), ['pd_0', 'pd_1']);
  assert.ok(rolled.loader.pdBeamSegments.every(s => s.arcMasked === false));
  assert.deepEqual([...plan.get('a').keys()].sort(), ['pd_0', 'pd_1']);
  assert.equal(pdPlan(rolled), plan, 'cached while the segment list is unchanged');
  // old recording (no up): dorsal is world +Y, both rounds sit on the hull's beam, pd_0 keeps both
  const old = mk(null);
  pdPlan(old);
  assert.deepEqual(old.loader.pdBeamSegments.map(s => s.arcTurret), ['pd_0', 'pd_0']);
  // a new recording on the same loader object is re-planned (keyed on the list, not its length)
  const again = mk([0, 0, 1]);
  rolled.loader.pdBeamSegments = again.loader.pdBeamSegments;
  assert.notEqual(pdPlan(rolled), plan);
  // recordings with `tur` keep the engine's turret untouched
  const eng = mk([0, 0, 1]);
  eng.loader.simTrace[0].ships.a = { ...eng.loader.simTrace[0].ships.a, tur: [[0, 0, 0]] };
  eng.loader.pdBeamSegments[1].turret = 'pd_0';
  pdPlan(eng);
  assert.deepEqual(eng.loader.pdBeamSegments.map(s => s.arcTurret), ['pd_0', 'pd_0']);
});

test('a dwell whose round moves behind the hull mid-dwell is shown from a mount that can bear', async () => {
  const { shownPdMount } = await import('../src/render/TurretArcs.js');
  const pds = arcsFor('cruiser').pds;
  assert.deepEqual(pds.map(m => m.slot), ['pd_0', 'pd_1']);
  const up = V(0, 1, 0), down = V(0, -1, 0);
  assert.equal(shownPdMount('pd_0', pds, up), 'pd_0');          // clear: stays
  assert.equal(shownPdMount('pd_0', pds, down), 'pd_1');        // under the keel: the ventral ball takes it
  assert.equal(shownPdMount('pd_1', pds, up), 'pd_0');
  // the corvette has nowhere to hand it: null (beam starts outside the hull)
  assert.equal(shownPdMount('pd_0', arcsFor('corvette').pds, down), null);
  // the stand-in ball trains on that round
  const ship = fakeShip('cruiser');
  const [pd0, pd1] = ship.userData.pdTurrets;
  const round = V(50, -400, 120);
  const scene = { currentTime: 5, ships: new Map([['alpha_1', ship]]), loader: null, weapons: { positionAt: (ty, id, t, out) => out.copy(round) } };
  updateHullDetails(ship, { hull: 100 }, scene);
  const out = new THREE.Vector2();
  fallbackPdAim(scene, pd1, arcsFor('cruiser').bySlot.get('pd_1'), [], 5, out, { targetType: 'torpedo', targetId: 'x', start: 3, end: 6 });
  const d = mountDir(Math.PI, out.x, out.y, V(0, 0, 0));
  assert.ok(d.distanceTo(round.clone().sub(pd1.mount.position).normalize()) < 1e-6);
  assert.ok(pd0);
});
