import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { updateTrailHistory } from '../src/render/Trails.js';
import { CameraController } from '../src/CameraController.js';
import { CONTACT_PANEL_IDS, threatLabel } from '../src/ContactOverlay.js';
import { placeChip, layoutTactical, waveLabel } from '../src/TacticalLayout.js';

const options = { maxAge: 2.5, minStep: 0.5, max: 64 };

test('small frame steps retain a full trail instead of overwriting its history', () => {
  for (const fps of [30, 60, 144]) {
    const history = [];
    for (let i = 0; i <= 5 * fps; i++) {
      updateTrailHistory(history, new THREE.Vector3(i / fps * 10, 0, 0), i / fps, options);
    }
    assert.ok(history.length > 25 && history.length <= 64);
    assert.ok(history[0].x - history.at(-1).x > 23, `trail too short at ${fps} fps`);
    assert.equal(history[0].x, 50);
  }
});

test('paused trails stay stable, jumps break trails, and unpowered exhaust expires', () => {
  const history = [];
  for (let t = 0; t <= 2; t += 0.1) updateTrailHistory(history, new THREE.Vector3(t, 0, 0), t, options);
  const head = history[0];
  const length = history.length;
  for (let i = 0; i < 100; i++) updateTrailHistory(history, head, head.t, options);
  assert.equal(history.length, length);
  updateTrailHistory(history, new THREE.Vector3(100, 0, 0), 10, options);
  assert.equal(history.length, 1);
  updateTrailHistory(history, new THREE.Vector3(200, 0, 0), 11, options, false);
  assert.equal(history[0].x, 100);
  updateTrailHistory(history, new THREE.Vector3(300, 0, 0), 13, options, false);
  assert.equal(history.length, 0);
  updateTrailHistory(history, new THREE.Vector3(), 1, options);
  updateTrailHistory(history, new THREE.Vector3(), 0, options);
  assert.equal(history.length, 1);
  assert.equal(history[0].t, 0);
});

function cameraFixture() {
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.2, 100000);
  const controls = { target: new THREE.Vector3(), enabled: true, update() { camera.lookAt(this.target); } };
  const pos = new THREE.Vector3();
  const scene = {
    ships: new Map([['ship', { userData: { size: { length: 10 } } }]]),
    destructionEffects: [], currentTime: 0, delta: 1 / 60,
    getShipPosition: () => pos.clone()
  };
  return { camera, controls, scene, pos, controller: new CameraController(camera, controls, scene) };
}

test('orbit preserves camera distance while the ship moves and after a seek', () => {
  const { camera, controls, scene, pos, controller } = cameraFixture();
  camera.position.set(20, 10, 20);
  controller.setMode('orbit', 'ship');
  const offset = camera.position.clone();
  pos.set(80, 20, -40);
  scene.currentTime = 30;
  controller.update({ ship: {} });
  assert.deepEqual(camera.position.clone().sub(pos).toArray(), offset.toArray());
  assert.deepEqual(controls.target.toArray(), pos.toArray());
});

test('follow stays behind the ship, handles vertical flight and tracks a wreck', () => {
  const { camera, controls, scene, pos, controller } = cameraFixture();
  controller.setMode('follow', 'ship');
  for (const forward of [[0, 0, 1], [0, 1, 0]]) {
    controller.updateFollowMode(pos, { forward }, true);
    assert.ok(camera.position.clone().sub(pos).dot(new THREE.Vector3(...forward)) < 0);
    assert.ok(camera.position.toArray().every(Number.isFinite));
    controls.update();
    const facing = camera.getWorldDirection(new THREE.Vector3());
    assert.ok(facing.dot(pos.clone().sub(camera.position).normalize()) > 0.999);
  }
  scene.destructionEffects.push({ shipId: 'ship', position: pos });
  pos.x = 30;
  controller.update({ ship: { destroyed: true, forward: [0, 0, 1] } });
  assert.equal(controls.target.x, 30);
});

test('contact labels avoid existing labels, panels, and the transport controls', () => {
  const view = { width: 1280, height: 720 };
  const occupied = [{ x: 0, y: 90, w: 248, h: 500 }];
  const a = placeChip(330, 330, 10, 126, 34, -1, occupied, view);
  assert.ok(a);
  assert.ok(a.x >= 248);
  occupied.push(a);
  const b = placeChip(330, 335, 10, 126, 34, -1, occupied, view);
  assert.ok(b);
  assert.ok(b.y + b.h <= a.y || b.y >= a.y + a.h || b.x + b.w <= a.x || b.x >= a.x + a.w);
  assert.equal(placeChip(300, 700, 10, 126, 34, -1, [{ x: 0, y: 0, w: 1280, h: 720 }], view), null);
});

test('fleet framing fits both fleets on a narrow viewport and ignores old wreck positions', () => {
  const { camera, scene, controller } = cameraFixture();
  camera.aspect = 600 / 800;
  camera.updateProjectionMatrix();
  const positions = {
    alpha: new THREE.Vector3(-250, 20, -5),
    beta: new THREE.Vector3(250, -20, 5),
    wreck: new THREE.Vector3(10000, 0, 0)
  };
  scene.getShipPosition = id => positions[id].clone();
  controller.focusFleet({ alpha: {}, beta: {}, wreck: { destroyed: true } });
  camera.updateMatrixWorld();
  for (const id of ['alpha', 'beta']) {
    const p = positions[id].clone().project(camera);
    assert.ok(Math.abs(p.x) < 0.65 && Math.abs(p.y) < 0.8 && p.z < 1);
  }
  assert.equal(controller.fleetFramed, true);
  controller.setMode('orbit', 'alpha');
  assert.equal(controller.fleetFramed, false);
});

const boxesOverlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

test('a torpedo-wave chip never covers a ship, its target included', () => {
  assert.equal(threatLabel({ count: 3, eta: 41 }), '3 TORP · 41s');
  assert.equal(waveLabel({ count: 15, eta: 6.4 }, 'Halberd-1'), '15 TORP ▸ Halberd-1 · 6s');
  // the wave's head has nearly reached its target, next to a second ship
  const ships = [
    { id: 'beta_1', faction: 'beta', x: 500, y: 300, r: 6, name: 'OCS Halberd-1', type: 'battlecruiser' },
    { id: 'beta_2', faction: 'beta', x: 530, y: 304, r: 6, name: 'OCS Halberd-2', type: 'battlecruiser' },
    { id: 'alpha_1', faction: 'alpha', x: 200, y: 300, r: 6, name: 'TIS Dart-1', type: 'corvette' }
  ];
  const waves = [{ key: 'w', target: 'beta_1', count: 15, live: 15, eta: 1, x: 496, y: 300, tx: 500, ty: 300,
    text: waveLabel({ count: 15, eta: 1 }, 'Halberd-1') }];
  const out = layoutTactical({ view: { width: 1280, height: 720 }, ships, waves });
  const box = out.waves[0].box;
  assert.ok(box);
  for (const s of ships) {
    assert.ok(!boxesOverlap(box, { x: s.x - 10, y: s.y - 10, w: 20, h: 20 }), s.id);
  }
  // contacts also keep clear of the live overlays and the AUTO CAM chip
  for (const id of ['commsFeed', 'killFeed', 'directorChip', 'liveDeliberation']) assert.ok(CONTACT_PANEL_IDS.includes(id), id);
});
