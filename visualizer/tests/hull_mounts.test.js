import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import {
  buildHullMounts, serialize, MOUNTS_PATH, viewerToBody, bodyToViewer, coverage
} from '../scripts/export_hull_mounts.mjs';
import { cellIndex, packMask, unpackMask, AZ_CELLS, EL_CELLS } from '../src/render/MountLimits.js';

const FLEET = JSON.parse(readFileSync(new URL('../../data/fleet_ships.json', import.meta.url), 'utf8'));
const COMMITTED = readFileSync(MOUNTS_PATH, 'utf8');
const DATA = JSON.parse(COMMITTED);
const fresh = buildHullMounts(FLEET);
const mount = (hull, slot) => DATA.hulls[hull].mounts.find(m => m.slot === slot);
const blockedAt = (m, az, el) => unpackMask(m.blocked)[cellIndex(az, el)] === 1;

test('data/hull_mounts.json is what the hull kit exports today (regenerate after moving a mount)', () => {
  assert.equal(serialize(fresh), COMMITTED,
    'stale: run `node visualizer/scripts/export_hull_mounts.mjs` and commit data/hull_mounts.json');
});

test('the export does not depend on ship seed or faction', () => {
  assert.equal(serialize(buildHullMounts(FLEET, { seed: -77123, faction: 'beta' })), COMMITTED);
});

test('frame mapping: body (x, y, z) = hull-kit (Z nose, X port, Y dorsal)', () => {
  assert.deepEqual(viewerToBody(new THREE.Vector3(1, 2, 3)), [3, 1, 2]);
  assert.deepEqual(bodyToViewer([3, 1, 2]).toArray(), [1, 2, 3]);
  // the data file calls the cruiser's weapon_1 the port turret: +y, firing to port
  const port = mount('cruiser', 'weapon_1'), stbd = mount('cruiser', 'weapon_2');
  assert.ok(port.pos[1] > 5 && stbd.pos[1] < -5);
  assert.deepEqual(port.axis, [0, 1, 0]);
  // dorsal and ventral PD, aft of midships; spinal muzzle at the nose
  assert.ok(mount('cruiser', 'pd_0').pos[2] > 5 && mount('cruiser', 'pd_1').pos[2] < -5);
  assert.ok(mount('cruiser', 'pd_0').pos[0] < 0);
  const sp = mount('cruiser', 'weapon_0');
  assert.equal(sp.kind, 'spinal');
  assert.ok(sp.pos[0] > DATA.hulls.cruiser.length_m * 0.45);
  // metres: no pivot lies outside a box a little bigger than the hull
  for (const [type, h] of Object.entries(DATA.hulls)) {
    for (const m of h.mounts) assert.ok(Math.abs(m.pos[0]) <= h.length_m * 0.85, `${type} ${m.slot}`);
  }
});

test('mask packing: cell k is bit 7 - k % 8 of byte k >> 3, and round-trips', () => {
  const m = new Uint8Array(AZ_CELLS * EL_CELLS);
  m[0] = 1; m[9] = 1; m[2591] = 1;
  const hex = packMask(m);
  assert.equal(hex.length, 648);
  assert.equal(hex.slice(0, 4), '8040');
  assert.equal(hex.slice(-2), '01');
  assert.deepEqual(unpackMask(hex), m);
  assert.equal(cellIndex(-180, -90), 0);
  assert.equal(cellIndex(180, 0), 18 * 72);          // +180 wraps onto the -180 cell
  assert.equal(cellIndex(0.1, 90), 35 * 72 + 36);
  assert.equal(cellIndex(-0.1, 2), 18 * 72 + 35);
});

test('mounts cannot fire through their own ship', () => {
  // a dorsal turret sees up, never straight down through the hull
  for (const [type, h] of Object.entries(DATA.hulls)) {
    for (const m of h.mounts) {
      if (m.kind === 'launcher') { assert.equal(m.open_fraction, 1, `${type} ${m.slot}`); continue; }
      if (m.kind === 'spinal') {
        assert.ok(!blockedAt(m, 0.1, 0.1), `${type} spinal fires ahead`);
        assert.ok(blockedAt(m, 90, 0) && blockedAt(m, 180, 0) && blockedAt(m, 0, 60), `${type} spinal gimbal`);
        continue;
      }
      // straight out along the mount axis is open, straight back through the hull is not
      const az = Math.atan2(m.axis[1], m.axis[0]) * 180 / Math.PI, el = Math.asin(m.axis[2]) * 180 / Math.PI;
      assert.ok(!blockedAt(m, az, el - 0.01), `${type} ${m.slot}: open along its axis`);
      assert.ok(blockedAt(m, az + 180, -el + 0.01), `${type} ${m.slot}: blocked through the hull`);
      assert.ok(m.open_fraction > 0.3 && m.open_fraction <= 0.52, `${type} ${m.slot}: open ${m.open_fraction}`);
    }
  }
});

test('coverage numbers: one hemisphere mount covers half, two opposed cover all', () => {
  const up = new Uint8Array(AZ_CELLS * EL_CELLS), down = new Uint8Array(AZ_CELLS * EL_CELLS);
  for (let j = 0; j < EL_CELLS; j++) for (let i = 0; i < AZ_CELLS; i++) (j < 18 ? up : down)[j * AZ_CELLS + i] = 1;
  const one = coverage([up]);
  assert.ok(Math.abs(one.ge1 - 0.5) < 1e-9 && one.ge2 === 0);
  assert.ok(one.sectors.dorsal.ge1 === 1 && one.sectors.ventral.ge1 === 0);
  const two = coverage([up, down]);
  assert.ok(Math.abs(two.ge1 - 1) < 1e-9);
  assert.equal(two.open.length, 0);
});
