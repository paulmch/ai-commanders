import test from 'node:test';
import assert from 'node:assert/strict';
import { BattleLoader } from '../src/BattleLoader.js';
import { Interpolator } from '../src/Interpolator.js';
import { HullKit } from '../src/render/Hulls.js';

const state = (extra = {}) => ({
  pos: [10, 20, 30], vel: [2, 0, 0], fwd: [0, 0, 1],
  hull: 100, destroyed: false, dying: false, ...extra
});

test('a ship survives until the recorded destruction timestamp', () => {
  const before = { ships: { ship: state() } };
  const after = { ships: { ship: state({ destroyed: true, hull: 0 }) } };
  assert.equal(Interpolator.interpolateShips(before, after, 0.99).ship.destroyed, false);
  const destroyed = Interpolator.interpolateShips(before, after, 1).ship;
  assert.equal(destroyed.destroyed, true);
  assert.deepEqual(destroyed.position, after.ships.ship.pos);
  assert.deepEqual(destroyed.velocity, after.ships.ship.vel);
  assert.deepEqual(destroyed.forward, after.ships.ship.fwd);
});

test('zero hull stays zero and dying starts at its recorded frame', () => {
  const before = { ships: { ship: state({ hull: 0 }) } };
  const after = { ships: { ship: state({ hull: 0, dying: true }) } };
  const mid = Interpolator.interpolateShips(before, after, 0.5).ship;
  assert.equal(mid.hull, 0);
  assert.equal(mid.dying, false);
  assert.equal(Interpolator.interpolateShips(before, after, 1).ship.dying, true);
});

test('death metadata carries the recorded transform and previous death spiral', () => {
  const loader = new BattleLoader();
  loader.simTrace = [
    { t: 4, ships: { ship: state({ dying: true }) } },
    { t: 5, ships: { ship: state({ pos: [12, 20, 30], destroyed: true }) } }
  ];
  const death = loader.buildDeathInfo().ship;
  assert.equal(death.time, 5);
  assert.equal(death.wasDying, true);
  assert.deepEqual(death.position, [12, 20, 30]);
  assert.deepEqual(death.forward, [0, 0, 1]);
  assert.deepEqual(death.velocity, [2, 0, 0]);
  // Incremental live rebuilds must keep the original death anchor.
  loader.simTrace.push({ t: 6, ships: { ship: state({ destroyed: true }) } });
  assert.deepEqual(loader.buildDeathInfo().ship, death);
});

test('visual impacts expose travel-adjusted time and do not repeat when paused', () => {
  const loader = new BattleLoader();
  loader.projectileHits.set('slug', { impact_position: [10, 0, 0], target_id: 'ship' });
  loader.lastProjectileState.set('slug', { pos: [0, 0, 0], vel: [5, 0, 0], time: 10 });
  const impacts = loader.getVisualImpacts(0, 20);
  assert.equal(impacts.length, 1);
  assert.equal(impacts[0].timestamp, 12);
  assert.equal(loader.getVisualImpacts(12, 12).length, 0);
  assert.equal(loader.getVisualImpacts(12, 20).length, 0);
});

test('wedge prows join a full-width hull and point along the thrust axis', () => {
  const kit = new HullKit({ length: 10, width: 2 }, 42);
  const prow = kit.prow('hull', 1, 0.8, 3, 4);
  const { min, max } = prow.boundingBox;
  const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-6, `${a} != ${b}`);
  near(min.x, -1); near(max.x, 1);
  near(min.y, -0.8); near(max.y, 0.8);
  near(min.z, 2.5); near(max.z, 5.5);
  prow.dispose();
});
