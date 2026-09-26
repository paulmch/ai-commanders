import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { HullKit } from '../src/render/Hulls.js';
import { lampLevel } from '../src/render/HullLights.js';
import { hullCode } from '../src/render/HullMarkings.js';
import {
  shipEventIndex, lastIndexBefore, recoil, doorOpen, assignLaunches
} from '../src/render/HullRigs.js';

test('strobes double-flash on battle time and die with the ship', () => {
  const strobe = { kind: 'strobe', period: 2, phase: 0, level: 3 };
  assert.equal(lampLevel(strobe, 0.02), 3);          // first flash
  assert.equal(lampLevel(strobe, 0.12), 0);          // gap
  assert.equal(lampLevel(strobe, 0.25), 3);          // second flash
  assert.equal(lampLevel(strobe, 1.0), 0);           // dark rest of the period
  assert.equal(lampLevel(strobe, 2.02), 3);          // periodic on battle time
  assert.equal(lampLevel(strobe, 0.02, { dying: true }), 0);
  // deterministic: same time, same answer
  assert.equal(lampLevel(strobe, 7.3), lampLevel(strobe, 7.3));
});

test('running lights burn steady and brown out on a failing hull', () => {
  const run = { kind: 'steady', level: 1, phase: 0.3 };
  assert.equal(lampLevel(run, 5, { hull: 1 }), 1);
  let dimmest = 1;
  for (let t = 0; t < 20; t += 0.05) dimmest = Math.min(dimmest, lampLevel(run, t, { dying: true }));
  assert.ok(dimmest < 0.1, 'a dying ship loses its running lights');
});

test('beacons pulse between a low floor and full level', () => {
  const b = { kind: 'beacon', period: 2, phase: 0, level: 2 };
  const vals = [];
  for (let t = 0; t < 2; t += 0.01) vals.push(lampLevel(b, t));
  assert.ok(Math.max(...vals) > 1.9);
  assert.ok(Math.min(...vals) < 0.2);
});

test('hull codes combine class letters with the ship number', () => {
  assert.equal(hullCode('destroyer', 'alpha_4'), 'DD-04');
  assert.equal(hullCode('cruiser_torpedo', 'beta_12'), 'CT-12');
  assert.equal(hullCode('corvette', 'alpha_1'), 'K-01');
  const hashed = hullCode('battleship', 'show_alpha_battleship');
  assert.match(hashed, /^BB-\d\d$/);
  assert.equal(hashed, hullCode('battleship', 'show_alpha_battleship'));
});

test('event index keeps per-ship shot and launch timelines sorted', () => {
  const idx = shipEventIndex([
    { event_type: 'shot_fired', ship_id: 'a', timestamp: 40, data: { target_id: 'b', weapon_slot: 'weapon_1' } },
    { event_type: 'shot_fired', ship_id: 'a', timestamp: 10, data: { target_id: 'c', weapon_slot: 'weapon_0' } },
    { event_type: 'torpedo_launched', ship_id: 'a', timestamp: 30, data: { target_id: 'b' } },
    { event_type: 'pd_fired', ship_id: 'a', timestamp: 31, data: {} }
  ]);
  assert.deepEqual(idx.shots.get('a').map(s => s.t), [10, 40]);
  assert.equal(idx.launches.get('a').length, 1);
  const shots = idx.shots.get('a');
  assert.equal(lastIndexBefore(shots, 5), -1);
  assert.equal(lastIndexBefore(shots, 10), 0);
  assert.equal(lastIndexBefore(shots, 39), 0);
  assert.equal(lastIndexBefore(shots, 100), 1);
});

test('recoil kicks at the shot and settles within a second and a half', () => {
  assert.equal(recoil(-0.1), 0);
  assert.ok(recoil(0.05) > 0.99);
  assert.ok(recoil(0.5) < 0.2);
  assert.equal(recoil(2), 0);
});

test('launch doors open before a launch and close after it', () => {
  assert.equal(doorOpen([30], 20), 0);
  assert.ok(doorOpen([30], 28) > 0 && doorOpen([30], 28) < 1);
  assert.equal(doorOpen([30], 29), 1);
  assert.equal(doorOpen([30], 32), 1);
  assert.equal(doorOpen([30], 34), 0);
  const banks = assignLaunches([{ t: 1 }, { t: 2 }, { t: 3 }, { t: 4 }, { t: 5 }], 4);
  assert.deepEqual(banks, [[1, 5], [2], [3], [4]]);
});

test('RCS stations expose outward unit nozzles on the hull skin', () => {
  const kit = new HullKit({ length: 10, width: 2 }, 7);
  kit.rcsStation(3, 0.5, 0.4, 0.12);
  const mounts = kit.rcsMounts;
  assert.ok(mounts.length >= 16, `quad of blocks with several nozzles, got ${mounts.length}`);
  for (const m of mounts) {
    assert.ok(m.pos instanceof THREE.Vector3 && m.dir instanceof THREE.Vector3);
    assert.ok(Math.abs(m.dir.length() - 1) < 1e-6);
    assert.ok(Math.abs(m.pos.z - 3) < 0.3, 'nozzles sit at the station');
    // no nozzle fires into the hull
    const radial = new THREE.Vector2(m.pos.x, m.pos.y);
    const d2 = new THREE.Vector2(m.dir.x, m.dir.y);
    assert.ok(radial.dot(d2) >= -1e-6);
  }
  // radial nozzles on all four blocks, so the quad can push every way
  const radial = mounts.filter(m => Math.abs(m.dir.z) < 1e-6 &&
    Math.abs(new THREE.Vector2(m.pos.x, m.pos.y).normalize().dot(new THREE.Vector2(m.dir.x, m.dir.y)) - 1) < 1e-6);
  assert.equal(radial.length, 4);
  for (const p of kit.parts) p.geom.dispose();
});
