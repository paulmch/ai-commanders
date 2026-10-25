import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  CATCHUP_MAX_RATE, LIVE_CUSHION_S, LOST_AFTER_FAILURES, backoffDelay, checkpointNumber, classifyVerdict,
  defaultMaxLag, deliberationProgress, liveDelay, liveStartTime, newBattleReason, nextCheckpointAfter,
  paceLiveHead
} from '../src/LiveSession.js';
import { TimeController } from '../src/TimeController.js';
import { CameraController, activeCut, buildDirectorCuts, frameGroups, DIRECTOR_USER_HOLD_S } from '../src/CameraController.js';
import { avoidBlockers, edgePoint } from '../src/OffscreenIndicators.js';

// ------------------------------------------------------------------ pacing

test('live head plays buffered footage at the chosen speed and waits at the edge', () => {
  let step = paceLiveHead({ current: 10, available: 40, dt: 0.5, speed: 1, status: 'running' });
  assert.equal(step.time, 10.5);
  assert.equal(step.state, 'playing');
  step = paceLiveHead({ current: 10, available: 40, dt: 0.5, speed: 2, status: 'running' });
  assert.equal(step.time, 11, 'speed buttons are honoured');
  step = paceLiveHead({ current: 37.9, available: 40, dt: 0.5, status: 'running' });
  assert.equal(step.time, 40 - LIVE_CUSHION_S, 'running: holds a cushion behind the newest frame');
  assert.equal(step.state, 'at-edge');
  step = paceLiveHead({ current: 38, available: 40, dt: 0.5, status: 'paused' });
  assert.equal(step.time, 38.5, 'paused checkpoint: no cushion, the orders frame is reached');
  step = paceLiveHead({ current: 39.5, available: 40, dt: 1, status: 'paused' });
  assert.equal(step.time, 40);
  step = paceLiveHead({ current: 40, available: 40, dt: 1, status: 'running' });
  assert.equal(step.time, 40, 'never steps backwards when the cushion returns');
});

test('live head only catches up beyond maxLag, capped unless the viewer chose faster', () => {
  const maxLag = defaultMaxLag(30);
  assert.equal(maxLag, 45);
  let step = paceLiveHead({ current: 0, available: 40, dt: 1, status: 'paused', maxLag });
  assert.equal(step.rate, 1);
  step = paceLiveHead({ current: 0, available: 60, dt: 1, status: 'paused', maxLag });
  assert.equal(step.rate, 2, '15 s over the limit doubles the rate');
  assert.equal(step.state, 'catching-up');
  step = paceLiveHead({ current: 0, available: 600, dt: 1, status: 'paused', maxLag });
  assert.equal(step.rate, CATCHUP_MAX_RATE);
  step = paceLiveHead({ current: 0, available: 600, dt: 1, speed: 8, status: 'paused', maxLag });
  assert.equal(step.rate, 8);
  step = paceLiveHead({ current: 0, available: 600, dt: 1, status: 'paused', pace: 'edge' });
  assert.equal(step.rate, 16, 'edge pace hugs the newest frame');
});

test('bursty stream: realtime pace spends deliberations watching, edge pace freezes', () => {
  // The server delivers 30 s of footage in 2 wall seconds, then deliberates
  // 58 s; the head should play smoothly and wait at the edge only briefly.
  const run = pace => {
    let current = 0, available = 2, waited = 0, maxJump = 0;
    const dt = 0.1;
    for (let wall = 0; wall < 600; wall += dt) {
      const cycle = wall % 60;
      const cp = Math.floor(wall / 60) * 30;
      available = Math.max(available, cycle < 2 ? cp + 30 * (cycle / 2) : cp + 30);
      const status = cycle < 2 ? 'running' : 'paused';
      const step = paceLiveHead({ current, available, dt, status, pace });
      maxJump = Math.max(maxJump, (step.time - current) / dt);
      if (step.state === 'at-edge') waited += dt;
      current = step.time;
    }
    return { waited, maxJump };
  };
  const realtime = run('realtime');
  const edge = run('edge');
  assert.ok(realtime.maxJump <= 1.0001, `realtime never fast-forwards (rate ${realtime.maxJump})`);
  assert.ok(realtime.waited < 0.55 * 600, `realtime watches through most deliberations (${realtime.waited.toFixed(0)} s idle)`);
  assert.ok(edge.waited > 0.85 * 600, 'edge pace sits frozen at the newest frame');
  assert.ok(edge.maxJump > 4);
});

test('joining mid-battle starts one checkpoint before the edge', () => {
  assert.equal(liveStartTime(20, 30, 'running'), 0);
  assert.equal(liveStartTime(150, 30, 'paused'), 120);
  assert.equal(liveStartTime(95, 30, 'running'), 60);
  assert.ok(liveStartTime(95, 30, 'running') <= 95 - 5);
  assert.equal(liveDelay(150, 120), 30);
  assert.equal(liveDelay(100, 120), 0);
});

test('checkpoint arithmetic and backoff', () => {
  assert.equal(nextCheckpointAfter(0, 30), 30);
  assert.equal(nextCheckpointAfter(29.99, 30), 30);
  assert.equal(nextCheckpointAfter(30, 30), 60);
  assert.equal(nextCheckpointAfter(41, 20.5), 61.5);
  assert.equal(checkpointNumber(90, 30), 3);
  assert.deepEqual([1, 2, 3, 4, 5, 9].map(backoffDelay), [2, 4, 8, 15, 15, 15]);
  assert.ok([1, 2, 3, 4, 5].slice(0, LOST_AFTER_FAILURES).map(backoffDelay).reduce((a, b) => a + b) < 60,
    'connection declared lost within a minute');
});

test('a different battle on the wire is detected', () => {
  const known = { battleId: 'aaa', started: true, lastFrameT: 300, interval: 30 };
  assert.equal(newBattleReason(known, { battle_id: 'aaa', sim_time_s: 330 }), null);
  assert.equal(newBattleReason(known, { battle_id: 'bbb', sim_time_s: 330 }), 'battle_id changed');
  // Older servers without an id: the sim clock jumping back, or a new draft
  const legacy = { battleId: null, started: true, lastFrameT: 300, interval: 30 };
  assert.equal(newBattleReason(legacy, { sim_time_s: 290, phase: 'battle' }), null);
  assert.equal(newBattleReason(legacy, { sim_time_s: 7, phase: 'battle' }), 'sim clock went back');
  assert.equal(newBattleReason(legacy, { sim_time_s: 0, phase: 'draft' }), 'draft restarted');
  assert.equal(newBattleReason({ ...legacy, started: false }, { sim_time_s: 0, phase: 'draft' }), null);
});

// ---------------------------------------------------------------- verdicts

test('verdicts: a budget stop is not a draw, a points tie is', () => {
  const stopped = classifyVerdict({ winner: null, reason: 'RuntimeError: Spending ceiling reached; battle stopped and partial replay saved',
    events: [{ event_type: 'battle_stopped', data: { reason: 'x' } }],
    billing: { reported_cost_usd: 4.674, spend_limit_usd: 6 } });
  assert.equal(stopped.kind, 'halted');
  assert.equal(stopped.title, 'BATTLE HALTED');
  assert.equal(stopped.detail, 'spending ceiling reached ($4.67 / $6.00)');
  const tie = classifyVerdict({ reason: 'Too close to call (109.4 vs 109.4)' });
  assert.equal(tie.kind, 'draw');
  assert.equal(tie.title, 'DRAW ON POINTS');
  assert.equal(tie.detail, '109.4 – 109.4');
  const win = classifyVerdict({ winner: 'beta', reason: 'Alpha fleet eliminated' });
  assert.deepEqual([win.kind, win.faction, win.title, win.detail], ['victory', 'beta', 'BETA VICTORIOUS', 'Alpha fleet eliminated']);
  assert.equal(classifyVerdict({ lost: true }).kind, 'lost');
  assert.equal(classifyVerdict({ reason: 'Mutual destruction/surrender' }).title, 'DRAW');
});

test('deliberation progress counts this checkpoint only, per side', () => {
  const ships = { alpha_1: { faction: 'alpha' }, alpha_2: { faction: 'alpha' }, beta_1: { faction: 'beta' } };
  const events = [
    { timestamp: 30, event_type: 'admiral_directive', data: { faction: 'alpha' } },
    { timestamp: 60, event_type: 'admiral_directive', data: { faction: 'alpha' } },
    { timestamp: 60, event_type: 'model_call', data: { actor: 'admiral', faction: 'alpha', latency_s: 7.25 } },
    { timestamp: 60, event_type: 'admiral_order', ship_id: 'alpha_1', data: {} },
    { timestamp: 60, event_type: 'admiral_order', ship_id: 'alpha_2', data: {} },
    { timestamp: 60, event_type: 'captain_decision', ship_id: 'alpha_1', data: {} },
    { timestamp: 60, event_type: 'model_error', ship_id: 'beta_1', data: { faction: 'beta' } },
    { timestamp: 0, event_type: 'model_call', data: { actor: 'admiral', faction: 'beta', phase: 'draft' } },
  ];
  const p = deliberationProgress({ events, ships, checkpointT: 60, alive: { alpha: 2, beta: 1 },
    pending: [{ actor: 'admiral', faction: 'beta', age_s: 41 }] });
  assert.equal(p.alpha.admiral, true);
  assert.equal(p.alpha.admiralLatency, 7.25);
  assert.equal(p.alpha.orders, 2);
  assert.equal(p.alpha.captains, 1);
  assert.equal(p.alpha.of, 2);
  assert.equal(p.beta.admiral, false);
  assert.equal(p.beta.errors, 1);
  assert.equal(p.beta.pending[0].age_s, 41);
});

// ---------------------------------------------------------- time controller

test('timeline spans past the footage; the playhead never runs past what arrived', () => {
  const tc = new TimeController(125);
  tc.setAvailable(90);
  tc.seek(110);
  assert.equal(tc.currentTime, 90, 'seek clamps to the footage');
  tc.seek(80);
  tc.play();
  tc.update(15);
  assert.equal(tc.currentTime, 90);
  assert.equal(tc.isPlaying, true, 'buffering holds, it does not pause');
  assert.equal(tc.buffering, true);
  tc.setAvailable(120);
  tc.update(5);
  assert.equal(tc.currentTime, 95);
  tc.setAvailable(Infinity);
  tc.update(100);
  assert.equal(tc.currentTime, 125);
  assert.equal(tc.isPlaying, false, 'the real end still pauses');
  const replay = new TimeController(60);
  replay.seek(70);
  assert.equal(replay.currentTime, 60, 'replays are unchanged');
});

test('a speed picked before the battle loads carries into the new timeline', () => {
  assert.equal(new TimeController(60).playbackSpeed, 1);
  const tc = new TimeController(60, { speed: 2 });
  assert.equal(tc.playbackSpeed, 2);
  tc.play();
  tc.update(5);
  assert.equal(tc.currentTime, 10);
  assert.equal(new TimeController(60, { speed: 64 }).playbackSpeed, 8, 'clamped like setSpeed');
  assert.equal(new TimeController(60, { speed: NaN }).playbackSpeed, 1);
  // main.js keeps the chosen speed on the app and hands it to every new
  // TimeController, so the active speed button and the live head agree
  // after the draft, a battle switch or another file
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(main, /new TimeController\([^;]*\{ speed: this\.playbackSpeed \}\)/);
});

// ----------------------------------------------------------------- director

test('director cuts: kills with killers, late detonations, legacy deaths and salvos', () => {
  const events = [
    { timestamp: 50, event_type: 'torpedo_launched', ship_id: 'alpha_1', data: { target_id: 'beta_2' } },
    { timestamp: 51, event_type: 'torpedo_launched', ship_id: 'alpha_2', data: { target_id: 'beta_2' } },
    { timestamp: 52.5, event_type: 'torpedo_launched', ship_id: 'alpha_3', data: { target_id: 'beta_1' } },
    { timestamp: 80, event_type: 'torpedo_launched', ship_id: 'beta_1', data: { target_id: 'alpha_1' } },
    { timestamp: 150, event_type: 'ship_dying', ship_id: 'beta_5', data: { killer_id: 'alpha_7' } },
    { timestamp: 150, event_type: 'ship_destroyed', ship_id: 'beta_5', data: { killer_id: 'alpha_7' } },
  ];
  const deaths = { beta_5: { time: 170 }, alpha_9: { time: 200 } };
  const cuts = buildDirectorCuts(events, deaths);
  assert.deepEqual(cuts.map(c => [c.kind, c.t, c.subject]), [
    ['salvo', 50, 'beta_2'], ['kill', 150, 'beta_5'], ['detonation', 170, 'beta_5'], ['kill', 200, 'alpha_9']]);
  assert.deepEqual(cuts[0].others, ['alpha_1', 'alpha_2', 'alpha_3']);
  assert.deepEqual(cuts[1].others, ['alpha_7']);
  assert.equal(activeCut(cuts, 49), null);
  assert.equal(activeCut(cuts, 53).kind, 'salvo');
  assert.equal(activeCut(cuts, 58), cuts[0]);
  assert.equal(activeCut(cuts, 58.6), null, 'salvo shot ends 6 s after the last launch');
  assert.equal(activeCut(cuts, 155).subject, 'beta_5');
  assert.equal(activeCut(cuts, 165), null);
  assert.equal(activeCut(cuts, 172).kind, 'detonation');
  const overlap = [{ kind: 'kill', t: 10, until: 18, subject: 'a' }, { kind: 'salvo', t: 12, until: 20, subject: 'b' }];
  assert.equal(activeCut(overlap, 13).kind, 'kill', 'a kill outranks a salvo');
});

test('fleet framing looks across the axis between the fleets and fits them', () => {
  const alpha = [new THREE.Vector3(-250, 0, 0), new THREE.Vector3(-240, 0, 30)];
  const beta = [new THREE.Vector3(250, 0, 0), new THREE.Vector3(260, 10, -20)];
  const groupA = alpha.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(2);
  const groupB = beta.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(2);
  const frame = frameGroups([...alpha, ...beta], { fov: 55, aspect: 16 / 9, groupA, groupB,
    fallbackDir: new THREE.Vector3(0, 0.6, 1).normalize() });
  const view = frame.position.clone().sub(frame.target);
  const axis = groupB.clone().sub(groupA).setY(0).normalize();
  assert.ok(Math.abs(view.clone().normalize().dot(axis)) < 1e-6, 'camera perpendicular to the fleet axis');
  assert.ok(view.z > 0, 'stays on the side it was already on');
  const elevation = Math.asin(view.clone().normalize().y);
  assert.ok(Math.abs(THREE.MathUtils.radToDeg(elevation) - 25) < 1e-6);
  // Every ship inside the horizontal field of view
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 1e6);
  camera.position.copy(frame.position);
  camera.lookAt(frame.target);
  camera.updateMatrixWorld();
  for (const p of [...alpha, ...beta]) {
    const ndc = p.clone().project(camera);
    assert.ok(Math.abs(ndc.x) < 0.8 && Math.abs(ndc.y) < 0.9, `ship at ${p.toArray()} framed (${ndc.x.toFixed(2)})`);
  }
  // With a side panel: the action centre lands mid free area (NDC x = shift)
  const centre = groupA.clone().add(groupB).multiplyScalar(0.5);
  const shifted = frameGroups([...alpha, ...beta], { fov: 55, aspect: 16 / 9, groupA, groupB, shift: 0.155 });
  camera.position.copy(shifted.position);
  camera.lookAt(shifted.target);
  camera.updateMatrixWorld();
  const box = new THREE.Box3().setFromPoints([...alpha, ...beta]).getCenter(new THREE.Vector3());
  assert.ok(Math.abs(box.clone().project(camera).x - 0.155) < 1e-6, 'centred in the free area');
  assert.ok(centre.distanceTo(box) < 20);
});

function directorFixture(makeControls = null) {
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.2, 100000);
  const listeners = {};
  const controls = makeControls ? makeControls(camera) : {
    target: new THREE.Vector3(), enabled: true, update() { camera.lookAt(this.target); },
    addEventListener(type, fn) { listeners[type] = fn; }
  };
  const positions = { alpha_1: new THREE.Vector3(-200, 0, 0), beta_1: new THREE.Vector3(200, 0, 0) };
  const scene = {
    SCALE: 1 / 1000, currentTime: 0, delta: 1 / 60, destructionEffects: [],
    ships: new Map(Object.keys(positions).map(id => [id, { userData: { size: { length: 10 }, faction: id.split('_')[0] } }])),
    getShipPosition: id => positions[id]?.clone() ?? null
  };
  camera.position.set(0, 300, 400);
  const controller = new CameraController(camera, controls, scene);
  return { camera, controls, scene, controller, listeners, positions };
}

test('director frames the fleets, cuts to kills, and yields to the viewer', () => {
  const { camera, controls, controller, listeners } = directorFixture();
  const states = { alpha_1: {}, beta_1: {} };
  assert.equal(controller.director.enabled, false, 'off until armed');
  controller.setDirector(true);
  controller.update(states, { time: 0, dt: 1 / 60 });
  assert.ok(Math.abs(controls.target.x) < 1e-9, 'target at the centre of both fleets');
  const view = camera.position.clone().sub(controls.target);
  assert.ok(Math.abs(view.x) < 1e-6, 'looking across the fleet axis');
  // Kill: the camera closes on the dying hull
  controller.setDirectorCuts([{ kind: 'kill', t: 10, until: 18, subject: 'beta_1', others: [] }]);
  for (let i = 0; i < 300; i++) controller.update(states, { time: 10 + i / 60, dt: 1 / 60 });
  assert.ok(controls.target.distanceTo(new THREE.Vector3(200, 0, 0)) < 1, 'kill shot centred on the victim');
  assert.ok(camera.position.distanceTo(controls.target) < 80);
  assert.equal(controller.directorState().shot, 'kill');
  // Viewer drags: director steps aside for DIRECTOR_USER_HOLD_S
  listeners.start();
  const before = camera.position.clone();
  controller.update(states, { time: 30, dt: 1 / 60 });
  assert.deepEqual(camera.position.toArray(), before.toArray());
  assert.equal(controller.directorState().hold, 'user');
  assert.ok(controller.directorState().secondsLeft <= DIRECTOR_USER_HOLD_S);
  // Something else places the camera (a script): the director steps aside
  controller.setDirector(true);
  controller.update(states, { time: 30, dt: 1 / 60 });
  camera.position.set(5, 5, 5);
  controller.update(states, { time: 30, dt: 1 / 60 });
  assert.equal(controller.directorState().hold, 'external');
  assert.deepEqual(camera.position.toArray(), [5, 5, 5]);
  // Ship focus holds it; back to the free camera releases
  controller.setDirector(true);
  controller.focusOnShip('alpha_1', states);
  assert.equal(controller.directorState().hold, 'ship');
  controller.setMode('free');
  assert.equal(controller.directorState().hold, null);
  controller.setDirector(false);
  camera.position.set(1, 2, 3);
  controller.update(states, { time: 40, dt: 1 / 60 });
  assert.deepEqual(camera.position.toArray(), [1, 2, 3], 'disarmed: the camera is the viewer\'s');
});

/** A canvas stand-in for a real OrbitControls: pointer listeners only. */
function fakeCanvas() {
  const listeners = {};
  return {
    style: {}, clientWidth: 960, clientHeight: 540,
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    removeEventListener(type, fn) { listeners[type] = (listeners[type] || []).filter(f => f !== fn); },
    setPointerCapture() {}, releasePointerCapture() {},
    fire(type, e = {}) {
      for (const fn of [...(listeners[type] || [])]) {
        fn({ pointerId: 1, pointerType: 'mouse', button: 0, preventDefault() {}, ...e });
      }
    }
  };
}

test('director takes over a damped OrbitControls without reading drag inertia as an outside move', () => {
  // OrbitControls registers a keydown listener on the global document
  const hadDocument = 'document' in globalThis;
  if (!hadDocument) globalThis.document = { addEventListener() {}, removeEventListener() {} };
  const canvas = fakeCanvas();
  let controls = null;
  try {
    const { camera, controller } = directorFixture(cam => {
      controls = new OrbitControls(cam, canvas);
      controls.enableDamping = true;   // SceneManager's settings
      controls.dampingFactor = 0.06;
      return controls;
    });
    const states = { alpha_1: {}, beta_1: {} };
    // One rendered frame at a software-GL rate: the director writes the
    // camera, then scene.render() runs controls.update()
    const frame = t => { controller.update(states, { time: t, dt: 0.25 }); controls.update(); };
    const drag = () => {
      canvas.fire('pointerdown', { clientX: 400, clientY: 300 });
      canvas.fire('pointermove', { clientX: 520, clientY: 340 });
      canvas.fire('pointerup');
    };
    controller.setDirector(true);
    for (let i = 0; i < 4; i++) frame(i);
    assert.equal(controller.directorState().active, true);
    drag();
    assert.equal(controller.directorState().hold, 'user');
    // A pressed right after the drag, with the rotation still coasting
    controller.setDirector(true);
    const before = camera.position.clone();
    for (let i = 0; i < 12; i++) frame(10 + i);
    assert.equal(controller.directorState().hold, null, 're-armed director keeps flying');
    assert.ok(camera.position.distanceTo(before) > 1, 'and it does fly');
    // The drag hold runs out with inertia left (few frames rendered meanwhile)
    drag();
    controller.director.holdUntil = performance.now() - 1;
    for (let i = 0; i < 12; i++) frame(30 + i);
    assert.equal(controller.directorState().hold, null, 'hold expiry hands back cleanly');
    // Settling the inertia never moves the camera itself
    drag();
    const pose = camera.position.clone();
    controller.setDirector(true);
    controller._settleControls();
    assert.deepEqual(camera.position.toArray(), pose.toArray());
    controls.update();
    assert.ok(camera.position.distanceTo(pose) < 1e-9, 'no inertia left to coast on');
    // A genuine outside placement still steps the director aside, and
    // that pause runs out like a drag's
    for (let i = 0; i < 2; i++) frame(50 + i);
    camera.position.set(5, 5, 5);
    frame(52);
    const held = controller.directorState();
    assert.equal(held.hold, 'external');
    assert.ok(held.secondsLeft > 0 && held.secondsLeft <= DIRECTOR_USER_HOLD_S);
    controller.director.holdUntil = performance.now() - 1;
    frame(53);
    assert.equal(controller.directorState().hold, null);
  } finally {
    controls?.dispose();
    if (!hadDocument) delete globalThis.document;
  }
});

test('off-screen chevrons sit where the ray to the fleet leaves the view', () => {
  const rect = { left: 16, top: 96, right: 1584, bottom: 796 };
  const cx = 800, cy = 446;
  let p = edgePoint(cx, cy, 5000, cy, rect);
  assert.deepEqual([p.x, p.y], [1584, cy]);
  p = edgePoint(cx, cy, cx, -9000, rect);
  assert.deepEqual([p.x, p.y], [cx, 96]);
  p = edgePoint(cx, cy, -1000, cy + 900, rect);
  assert.ok(p.x >= rect.left - 1e-9 && p.y <= rect.bottom + 1e-9);
  assert.ok(Math.abs(p.x - rect.left) < 1e-6 || Math.abs(p.y - rect.bottom) < 1e-6);
});

test('a chevron slides along its edge out from under a HUD panel', () => {
  const trackHud = { left: 1330, top: 640, right: 1592, bottom: 810 };
  const onRight = avoidBlockers({ x: 1584, y: 700, angle: 0.1 }, [trackHud]);
  assert.equal(onRight.x, 1584);
  assert.ok(onRight.y < trackHud.top, 'moved up along the right edge');
  const onBottom = avoidBlockers({ x: 1400, y: 796, angle: Math.PI / 2 - 0.1 }, [trackHud]);
  assert.equal(onBottom.y, 796);
  assert.ok(onBottom.x < trackHud.left, 'moved left along the bottom edge');
  const clear = { x: 800, y: 96, angle: -Math.PI / 2 };
  assert.equal(avoidBlockers(clear, [trackHud]), clear);
});
