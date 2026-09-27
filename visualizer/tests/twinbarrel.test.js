import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { BattleLoader } from '../src/BattleLoader.js';
import { Projectiles } from '../src/render/Projectiles.js';
import { shotBarrel, assignBarrels, TWIN_BARREL_X } from '../src/render/WeaponTimeline.js';
import { shipEventIndex, updateHullDetails } from '../src/render/HullRigs.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const shot = (t, ship, slot, extra = {}) => ({
  timestamp: t, event_type: 'shot_fired', ship_id: ship,
  data: { weapon_slot: slot, weapon_name: 'Heavy Coilgun Battery', target_id: 'beta', ...extra }
});

test('shotBarrel: recorded barrel wins, else alternate by shot number, spinals have one bore', () => {
  assert.equal(shotBarrel({ barrel: 1 }, 0), 1);
  assert.equal(shotBarrel({ barrel: 0 }, 1), 0);
  assert.equal(shotBarrel({}, 0), 0);
  assert.equal(shotBarrel({}, 1), 1);
  assert.equal(shotBarrel({}, 6), 0);
  assert.equal(shotBarrel({ weapon_name: 'Spinal Coiler Mk3', barrel: 1 }, 3), 0);
});

test('fallback alternation counts shots per ship and weapon slot, in event order', () => {
  const ev = [
    shot(1, 'a', 'weapon_1'), shot(1, 'a', 'weapon_2'), shot(2, 'a', 'weapon_1'),
    shot(2, 'b', 'weapon_1'), shot(3, 'a', 'weapon_1'), shot(3, 'a', 'weapon_2'),
    { timestamp: 3, event_type: 'pd_fired', ship_id: 'a', data: {} }
  ];
  const b = assignBarrels(ev);
  assert.deepEqual(ev.slice(0, 6).map(e => b.get(e)), [0, 0, 1, 0, 0, 1]);
  assert.equal(b.has(ev[6]), false);
  // a salvo carrying the contract fields keeps its recorded barrels
  const salvo = [0, 1, 2, 3, 4].map(i => shot(10 + i * 2, 'a', 'weapon_3', { salvo_index: i, salvo_size: 5, barrel: (i + 1) % 2 }));
  const bs = assignBarrels(salvo);
  assert.deepEqual(salvo.map(e => bs.get(e)), [1, 0, 1, 0, 1]);
  // the hull-rig index carries the same barrels
  const idx = shipEventIndex([...ev, ...salvo]);
  assert.deepEqual(idx.shots.get('a').filter(s => s.slot === 'weapon_1').map(s => s.barrel), [0, 1, 0]);
});

function twinRig() {
  const mount = new THREE.Group(), yaw = new THREE.Group(), pitch = new THREE.Group();
  mount.position.set(0, 1, 2);
  mount.add(yaw); yaw.add(pitch);
  const barrels = TWIN_BARREL_X.map(bx => { const g = new THREE.Group(); g.position.x = bx * 0.5; pitch.add(g); return g; });
  return { mount, yaw, pitch, barrels, s: 0.5, slot: 'weapon_1', yawAngle: 0, pitchAngle: 0, pitchZ: 0 };
}

function recording(events) {
  const loader = new BattleLoader();
  const st = (pos, fwd) => ({ pos, vel: [0, 0, 0], fwd });
  const slug = (id, x) => ({ id, pos: [x, 0, 0], vel: [6000, 0, 0], source: 'alpha', target: 'beta' });
  const frame = (t, proj) => ({ t, ships: { alpha: st([0, 0, 0], [1, 0, 0]), beta: st([400000, 0, 0], [-1, 0, 0]) }, projectiles: proj });
  loader.simTrace = [frame(0, []), frame(1, [slug('s1', 6000)]), frame(2, [slug('s1', 12000), slug('s2', 6000)]), frame(40, [])];
  loader.events = events;
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  return loader;
}

function renderer() {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 50, 50);
  const ships = new Map();
  for (const id of ['alpha', 'beta']) {
    const g = new THREE.Group();
    g.userData = { shipId: id, faction: id, size: { length: 8, width: 2 }, shipType: 'destroyer' };
    ships.set(id, g); scene.add(g);
  }
  const alpha = ships.get('alpha');
  alpha.lookAt(1, 0, 0);
  alpha.userData.gunTurrets = [twinRig()];
  ships.get('beta').position.set(400, 0, 0);
  return new Projectiles({ scene, camera, ships });
}

test('successive rounds of a twin turret leave alternate barrels, flashes at the same muzzles', () => {
  const loader = recording([
    shot(0, 'alpha', 'weapon_1', { projectile_id: 's1' }),
    shot(1, 'alpha', 'weapon_1', { projectile_id: 's2' })
  ]);
  const fx = renderer();
  try {
    fx.update(loader, 0.001);
    const alpha = fx.manager.ships.get('alpha');
    const s1 = fx.timeline.slugs.get('s1'), s2 = fx.timeline.slugs.get('s2');
    assert.equal(s1.shot.barrel, 0);
    assert.equal(s2.shot.barrel, 1);
    const p1 = alpha.worldToLocal(fx.positionAt('slug', 's1', 0, new THREE.Vector3()).clone());
    const p2 = alpha.worldToLocal(fx.positionAt('slug', 's2', 1, new THREE.Vector3()).clone());
    // the gun is laid dead ahead: the two muzzles sit side by side, 0.4 s apart
    near(p1.distanceTo(p2), 0.4 * 0.5, 1e-3);
    near(p1.y, p2.y, 1e-6);
    near(p1.z, p2.z, 1e-6);
    const [m1, m2] = fx.timeline.moments.filter(m => m.kind === 'muzzle');
    const f1 = fx.shotMuzzle(m1, alpha).pos, f2 = fx.shotMuzzle(m2, alpha).pos;
    near(f1.x, p1.x, 1e-6);
    near(f2.x, p2.x, 1e-6);
    assert.ok(Math.abs(f1.x - f2.x) > 0.15);
  } finally { fx.dispose(); }
});

test('only the barrel that fired recoils, as a pure function of battle time', () => {
  const ships = new Map();
  const a = new THREE.Group(); a.userData = { shipId: 'alpha', faction: 'alpha' };
  const b = new THREE.Group(); b.userData = { shipId: 'beta', faction: 'beta' }; b.position.set(50, 0, 0);
  ships.set('alpha', a); ships.set('beta', b);
  const rig = twinRig();
  a.userData.gunTurrets = [rig];
  const events = [shot(10, 'alpha', 'weapon_1'), shot(11, 'alpha', 'weapon_1')];
  const scene = { ships, destroyedShips: new Set(), currentTime: 0, loader: { events }, getEntityPosition: () => null };
  const pose = (t) => {
    scene.currentTime = t;
    updateHullDetails(a, { hull: 100 }, scene);
    return [rig.barrels[0].position.z, rig.barrels[1].position.z, rig.pitch.position.z];
  };
  const [b0, b1] = pose(10.06);
  assert.ok(b0 < -0.1 && b1 === 0, 'barrel 0 runs back, barrel 1 stays in battery');
  const [c0, c1] = pose(11.06);
  assert.ok(c1 < -0.1 && c1 < c0, 'the second shot kicks barrel 1');
  const again = pose(11.06);
  pose(30); pose(2);
  assert.deepEqual(pose(11.06), again);
  assert.deepEqual(pose(20), [0, 0, 0]);
});
