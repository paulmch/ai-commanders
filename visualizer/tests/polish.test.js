import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { BattleLoader } from '../src/BattleLoader.js';
import { Projectiles } from '../src/render/Projectiles.js';
import { trackEmphasis } from '../src/render/Tracks.js';
import { latestWreckEnd, WRECK_DRIFT_S } from '../src/render/Explosion.js';
import { updateHullDetails } from '../src/render/HullRigs.js';
import { GraphicsSettings } from '../src/render/Quality.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('tracks: the focused ship keeps its weight while the others dim in a close-up', () => {
  const far = trackEmphasis(false, 5000), farFocus = trackEmphasis(true, 5000);
  assert.equal(far.intensity, 1);
  assert.equal(far.width, 1);
  assert.ok(farFocus.intensity >= 1 && farFocus.width > 1);
  const close = trackEmphasis(false, 10), closeFocus = trackEmphasis(true, 10);
  assert.ok(close.intensity <= 0.3, 'other ships fade to faint guides up close');
  assert.ok(closeFocus.width > farFocus.width);
  // monotonic in range, no allocation when an out object is given
  const out = {};
  let prev = -1;
  for (let d = 0; d < 400; d += 10) {
    assert.equal(trackEmphasis(false, d, out), out);
    assert.ok(out.intensity >= prev);
    prev = out.intensity;
  }
});

test('the replay epilogue covers the latest wreck, drift included', () => {
  assert.equal(latestWreckEnd({}), -Infinity);
  const end = latestWreckEnd({
    a: { time: 100, reactorCause: true },
    b: { time: 700 },                        // coasts dark, then breaks up
    c: { time: 705, wasDying: true }         // sim death spiral: no extra drift
  });
  assert.equal(end, Math.max(700 + WRECK_DRIFT_S, 705) + 40);
});

// --- weapon origins ---------------------------------------------------------
const shipState = (pos, fwd = [1, 0, 0]) => ({ pos, vel: [0, 0, 0], fwd });
function gunRecording() {
  const loader = new BattleLoader();
  const slug = (x) => ({ id: 's1', pos: [x, 0, 0], vel: [6000, 0, 0], source: 'alpha', target: 'beta' });
  loader.simTrace = [
    { t: 0, ships: { alpha: shipState([0, 0, 0]), beta: shipState([400000, 0, 0], [-1, 0, 0]) } },
    { t: 1, ships: { alpha: shipState([0, 0, 0]), beta: shipState([400000, 0, 0], [-1, 0, 0]) }, projectiles: [slug(6000)] },
    { t: 20, ships: { alpha: shipState([0, 0, 0]), beta: shipState([400000, 0, 0], [-1, 0, 0]) }, projectiles: [slug(120000)] },
    { t: 40, ships: { alpha: shipState([0, 0, 0]), beta: shipState([400000, 0, 0], [-1, 0, 0]) } }
  ];
  loader.events = [
    { timestamp: 0, event_type: 'shot_fired', ship_id: 'alpha',
      data: { projectile_id: 's1', weapon_slot: 'weapon_1', weapon_name: 'Coilgun Battery', target_id: 'beta' } },
    { timestamp: 2, event_type: 'pd_fired', ship_id: 'beta', data: { target_id: 's1', target_type: 'slug', turret_name: 'pd_1' } }
  ];
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  return loader;
}

function gunRenderer() {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 50, 50);
  const ships = new Map();
  for (const id of ['alpha', 'beta']) {
    const g = new THREE.Group();
    g.userData = { shipId: id, faction: id, size: { length: 8, width: 2 }, shipType: 'destroyer' };
    ships.set(id, g); scene.add(g);
  }
  // alpha: hull +Z along world +X (fwd [1,0,0]), one dorsal turret on weapon_1
  const alpha = ships.get('alpha');
  alpha.lookAt(1, 0, 0);
  const mount = new THREE.Group(); mount.position.set(0, 1, 2);
  alpha.userData.gunTurrets = [{ mount, yaw: new THREE.Group(), pitch: new THREE.Group(), s: 0.5, slot: 'weapon_1', yawAngle: 0, pitchAngle: 0 }];
  // beta: two PD balls
  ships.get('beta').position.set(400, 0, 0);
  ships.get('beta').userData.pdMounts = [{ name: 'pd_0', pos: new THREE.Vector3(0, 1, 0) }, { name: 'pd_1', pos: new THREE.Vector3(0, -1, 0) }];
  ships.get('beta').userData.pdTurrets = [{ name: 'pd_0', s: 0.4 }, { name: 'pd_1', s: 0.4 }];
  return new Projectiles({ scene, camera, ships });
}

test('a slug leaves its turret muzzle and blends onto its recorded track', () => {
  const loader = gunRecording(), fx = gunRenderer();
  try {
    fx.update(loader, 0.001);
    const tr = fx.timeline.slugs.get('s1');
    assert.deepEqual(tr.shot, { slot: 'weapon_1', index: 1, spinal: false });
    // at launch: at the barrel tip above the dorsal mount, not the ship centre
    const p0 = fx.positionAt('slug', 's1', 0, new THREE.Vector3());
    const alpha = fx.manager.ships.get('alpha');
    const local = alpha.worldToLocal(p0.clone());
    near(local.y, 1 + 0.52 * 0.5, 1e-3);            // mount height + turret pivot
    assert.ok(local.z > 2, 'muzzle is ahead of the mount along the barrel');
    // far from the shooter the recorded track is used as is
    const pFar = fx.positionAt('slug', 's1', 10, new THREE.Vector3());
    near(pFar.x, 60, 1e-9); near(pFar.y, 0, 1e-9); near(pFar.z, 0, 1e-9);
    // monotonic: the round never stalls or reverses while the offset fades
    let prev = -Infinity;
    for (let t = 0; t < 6; t += 0.1) {
      const x = fx.positionAt('slug', 's1', t, new THREE.Vector3()).x;
      assert.ok(x > prev, `stall at ${t}`);
      prev = x;
    }
    // the muzzle flash is drawn at the same muzzle
    const m = fx.timeline.moments.find(k => k.kind === 'muzzle');
    const mz = fx.shotMuzzle(m, alpha);
    near(mz.pos.y, local.y, 1e-6);
  } finally { fx.dispose(); }
});

test('PD beams start at the named PD mount, falling back by index, else null', () => {
  const fx = gunRenderer();
  try {
    const beta = fx.manager.ships.get('beta');
    assert.equal(fx.pdMount({ turret: 'pd_1' }, beta).pos.y, -1);
    assert.equal(fx.pdMount({ turret: 'pd_3' }, beta).pos.y, -1);     // 3 % 2
    near(fx.pdMount({ turret: 'pd_0' }, beta).reach, 0.72 * 0.4);
    assert.equal(fx.pdMount({ turret: 'pd_0' }, fx.manager.ships.get('alpha')), null);
  } finally { fx.dispose(); }
});

test('impact flashes sit on the face met in the target frame, not the drift direction', () => {
  const loader = new BattleLoader();
  // both ships drift at 50 km/s along +Z; the round closes head-on along +X
  const drift = [0, 0, 50000];
  const st = (x, t) => ({ pos: [x, 0, drift[2] * t], vel: drift, fwd: [1, 0, 0] });
  const slug = (x, t) => ({ id: 'h', pos: [x, 0, drift[2] * t], vel: [6000, 0, drift[2]], source: 'alpha', target: 'beta' });
  loader.simTrace = [0, 1, 2, 3].map(t => ({ t, ships: { alpha: st(0, t), beta: st(24000, t) }, projectiles: t < 3 ? [slug(6000 * (t + 1), t)] : [] }));
  loader.events = [{ timestamp: 3, event_type: 'hit', ship_id: 'beta', data: { projectile_id: 'h', impact_position: [24000, 0, 150000], kinetic_energy_gj: 5, target_id: 'beta' } }];
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  const fx = gunRenderer();
  try {
    fx.update(loader, 3.0);
    const hit = fx.timeline.moments.find(m => m.kind === 'hit');
    assert.ok(hit);
    fx.moment(hit, hit.time);
    // placed on the target's -X face (approach side), not pushed along -Z
    const target = [24, 0, 150];
    assert.ok(fx.a.x < target[0] - 0.5, `flash x ${fx.a.x}`);
    near(fx.a.z, target[2], 1e-6);
  } finally { fx.dispose(); }
});

// --- hull rigs ----------------------------------------------------------------
function rigScene() {
  const ships = new Map();
  const mk = (id, faction, x) => {
    const g = new THREE.Group();
    g.position.set(x, 0, 0);
    g.userData = { shipId: id, faction };
    ships.set(id, g);
    return g;
  };
  const a = mk('a', 'alpha', 0), b = mk('b', 'beta', 50);
  b.position.y = 20;
  const mount = new THREE.Group(), yaw = new THREE.Group(), pitch = new THREE.Group();
  mount.add(yaw); yaw.add(pitch);
  a.userData.gunTurrets = [{ mount, yaw, pitch, s: 0.5, slot: 'weapon_0', yawAngle: 0, pitchAngle: 0.12, pitchZ: 0 }];
  a.userData.pdTurrets = [{ name: 'pd_0', mount: new THREE.Group(), ball: new THREE.Object3D(), s: 0.4, yawAngle: 0, pitchAngle: 0.4, phase: 0.3 }];
  return { ships, destroyedShips: new Set(), currentTime: 0, delta: 0.016, getEntityPosition: () => null };
}

test('turret poses are a pure function of battle time (pause, replay, seek)', () => {
  const scene = rigScene();
  const a = scene.ships.get('a');
  const rig = a.userData.gunTurrets[0], pd = a.userData.pdTurrets[0];
  const pose = (t) => {
    scene.currentTime = t;
    updateHullDetails(a, { hull: 100 }, scene);
    return [rig.yawAngle, rig.pitchAngle, pd.yawAngle, pd.pitchAngle];
  };
  const first = pose(12);
  for (let i = 0; i < 20; i++) assert.deepEqual(pose(12), first, 'repeated paused frames');
  pose(30); pose(3);
  assert.deepEqual(pose(12), first, 'after seeks');
  // laid on the enemy: +X in the mount frame is yaw pi/2, elevated
  near(rig.yawAngle, Math.PI / 2, 1e-9);
  assert.ok(rig.pitchAngle > 0.3);
  // dying: a fixed slump, the same however the frame is reached
  scene.currentTime = 12;
  updateHullDetails(a, { hull: 0, dying: true }, scene);
  const slump = [rig.yawAngle, rig.pitchAngle];
  pose(5);
  updateHullDetails(a, { hull: 0, dying: true }, scene);
  assert.deepEqual([rig.yawAngle, rig.pitchAngle], slump);
  assert.equal(rig.pitchAngle, -0.04);
});

// --- graphics controls ------------------------------------------------------
test('the GFX button and G key are installed once however often the viewer reloads', () => {
  const listeners = [];
  const elements = new Map();
  const el = (id) => ({
    id, children: [], listeners: 0,
    appendChild(c) { this.children.push(c); elements.set(c.id, c); },
    addEventListener() { this.listeners++; },
    classList: { toggle() {} }, setAttribute() {}, set innerHTML(v) { this.html = v; }
  });
  const host = el('viewControls');
  elements.set('viewControls', host);
  const saved = globalThis.document;
  globalThis.document = {
    getElementById: (id) => elements.get(id) || null,
    createElement: () => el(''),
    addEventListener: (type, fn) => listeners.push(fn)
  };
  const fakeRenderer = () => ({
    getContext: () => ({ getExtension: () => null, getParameter: () => 'test' }),
    getPixelRatio: () => 1, setPixelRatio() {}
  });
  try {
    const before = listeners.length;
    const first = new GraphicsSettings(fakeRenderer());
    first.installControls();
    const second = new GraphicsSettings(fakeRenderer());
    second.installControls();
    second.installControls();
    assert.equal(host.children.length, 1, 'one GFX button');
    assert.ok(listeners.length - before <= 1, 'at most one G key handler');
    // the shared controls drive the newest settings
    const level = second.level;
    listeners.at(-1)?.({ code: 'KeyG', target: null });
    if (listeners.length > before) assert.notEqual(second.level, level);
    first.dispose(); second.dispose();
  } finally {
    globalThis.document = saved;
  }
});
