// Pass-4 integration seam between the live wire (auto director, fleet view,
// off-screen chevrons) and the spectator HUD (the command net, open by
// default on wide screens): every layer must agree on the view the HUD
// leaves free.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { CameraController, frameGroups } from '../src/CameraController.js';
import { insetRectFor, projectToScreen, BLOCKER_IDS } from '../src/OffscreenIndicators.js';
import { RIGHT_PANEL_IDS, freeViewFrame, viewInsets } from '../src/ViewInsets.js';

const W = 1280, H = 720;

/** Minimal document: panels by id with a client rect (null rect = not rendered). */
function panelDoc(rects, { hidden = [] } = {}) {
  const els = {};
  for (const [id, r] of Object.entries(rects)) {
    const cls = new Set(hidden.includes(id) ? ['hidden'] : []);
    els[id] = {
      hidden: false,
      classList: { contains: c => cls.has(c) },
      getClientRects: () => (r && !cls.has('hidden') ? [r] : []),
      getBoundingClientRect: () => ({ ...r, width: r.right - r.left, height: r.bottom - r.top })
    };
  }
  return { getElementById: id => els[id] || null };
}

// The 1280x720 layout measured in the merged viewer: registry on the left,
// command net open on the right (top 90..457, x 964..1264)
const REGISTRY = { left: 16, right: 248, top: 90, bottom: 576 };
const NET = { left: 964, right: 1264, top: 90, bottom: 457 };

test('view insets count the command net with the right-hand columns', () => {
  assert.ok(RIGHT_PANEL_IDS.includes('commsFeed'));
  assert.deepEqual(viewInsets(panelDoc({ leftSidebar: REGISTRY, commsFeed: NET }), W), { left: 248, right: 316 });
  // closed (.hidden -> display:none) it frees the column again
  assert.deepEqual(viewInsets(panelDoc({ leftSidebar: REGISTRY, commsFeed: NET }, { hidden: ['commsFeed'] }), W),
    { left: 248, right: 0 });
  // telemetry + net (wide screen: the net docks left of the telemetry): the widest wins
  const wide = viewInsets(panelDoc({ leftSidebar: REGISTRY, shipTelemetry: { left: 1304, right: 1584, top: 90, bottom: 700 },
    commsFeed: { left: 992, right: 1292, top: 90, bottom: 600 } }), 1600);
  assert.deepEqual(wide, { left: 248, right: 608 });
});

test('free-view frame centres the action between the registry and the command net', () => {
  const alone = freeViewFrame(W, { left: 248, right: 0 });
  const net = freeViewFrame(W, { left: 248, right: 316 });
  assert.ok(alone.shift > 0.18, 'registry alone: shifted right');
  assert.ok(net.shift < 0 && net.shift > -0.1, 'with the net the centre moves back left of centre');
  assert.ok(net.widthShare < alone.widthShare);
  // the framed span stays inside the free strip [248, 964]
  const centre = (net.shift + 1) * W / 2, half = net.widthShare * W / 2;
  assert.ok(centre - half > 248 && centre + half < 964, `${centre - half}..${centre + half}`);
});

function fleets() {
  const alpha = [new THREE.Vector3(-260, 0, -20), new THREE.Vector3(-240, 10, 15), new THREE.Vector3(-250, -5, 0)];
  const beta = [new THREE.Vector3(250, 5, -10), new THREE.Vector3(270, -8, 20), new THREE.Vector3(260, 0, 0)];
  return { alpha, beta, all: [...alpha, ...beta] };
}

const screenX = (p, camera) => (p.clone().project(camera).x + 1) * W / 2;

test('auto director framing with the net open keeps both fleets out from under it', () => {
  const { alpha, beta, all } = fleets();
  const camera = new THREE.PerspectiveCamera(55, W / H, 0.2, 1e6);
  const centroid = pts => pts.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(pts.length);
  const frame = freeViewFrame(W, { left: 248, right: 316 });
  const view = frameGroups(all, { fov: 55, aspect: W / H, groupA: centroid(alpha), groupB: centroid(beta), ...frame });
  camera.position.copy(view.position);
  camera.lookAt(view.target);
  camera.updateMatrixWorld();
  for (const p of all) {
    const x = screenX(p, camera);
    assert.ok(x > 248 && x < 964, `ship at x=${x.toFixed(0)} hidden by a side column`);
  }
});

test('fleet view (F) honours the free-view frame; without one it keeps the old full-width framing', () => {
  const camera = new THREE.PerspectiveCamera(55, W / H, 0.2, 1e6);
  const controls = { target: new THREE.Vector3(), enabled: true, update() { camera.lookAt(this.target); } };
  const { all } = fleets();
  const states = Object.fromEntries(all.map((_, i) => [`s${i}`, {}]));
  const scene = { ships: new Map(), destructionEffects: [], getShipPosition: id => all[Number(id.slice(1))].clone() };
  const controller = new CameraController(camera, controls, scene);
  controller.focusFleet(states, freeViewFrame(W, { left: 248, right: 316 }));
  camera.updateMatrixWorld();
  for (const p of all) {
    const x = screenX(p, camera);
    assert.ok(x > 248 && x < 964, `F view put a ship under a column (x=${x.toFixed(0)})`);
  }
  assert.equal(controller.fleetFramed, true);
  const centre = new THREE.Box3().setFromPoints(all).getCenter(new THREE.Vector3());
  const frame = freeViewFrame(W, { left: 248, right: 316 });
  assert.ok(Math.abs(screenX(centre, camera) - (frame.shift + 1) * W / 2) < 1, 'fleet centre at the free-view centre');
  controller.focusFleet(states);
  camera.updateMatrixWorld();
  assert.ok(Math.abs(screenX(centre, camera) - W / 2) < 1, 'no frame: centred on screen');
});

test('off-screen chevrons treat ships under the command net as off screen', () => {
  const rect = insetRectFor(W, H, { left: 248, right: 316 });
  assert.equal(rect.left, 258);
  assert.equal(rect.right, 954);
  assert.deepEqual(insetRectFor(W, H, {}), { left: 16, top: 96, right: W - 16, bottom: H - 104 });
  const camera = new THREE.PerspectiveCamera(55, W / H, 0.2, 1e6);
  camera.updateMatrixWorld();   // at the origin, looking down -Z
  // a ship 100 units out that projects to screen x = 1100 (under the net)
  const ndc = 1100 / (W / 2) - 1;
  const underNet = new THREE.Vector3(ndc * 100 * Math.tan(THREE.MathUtils.degToRad(27.5)) * (W / H), 0, -100);
  const proj = projectToScreen(underNet, camera, W, H, rect);
  assert.ok(Math.abs(proj.x - 1100) < 1, `projected at ${proj.x}`);
  assert.equal(proj.onScreen, false, 'under the net: a chevron points at it');
  assert.equal(projectToScreen(underNet, camera, W, H, insetRectFor(W, H, { left: 248 })).onScreen, true,
    'net closed: on screen');
  assert.ok(BLOCKER_IDS.includes('killFeed') && BLOCKER_IDS.includes('directorChip'));
});
