import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import {
  buildHighlights, buildShotList, shotAt, highlightList, stepHighlight, buildShakeSources, shakeAt,
  handheldAt, impactPulse, closestApproach, MIN_SHOT_S, DETONATION_LOCK_S, PULLBACK_PRE_S, SHAKE_LIFE_S, SHOT_TYPES
} from '../src/Director.js';
import { shotPose, clearObstacles, obstaclesOf, fitDistance } from '../src/DirectorCamera.js';
import { CameraController, DIRECTOR_BLEND_S } from '../src/CameraController.js';
import { shipDetonationTime, blastFrameTime, wreckDriftS, WRECK_DRIFT_S, buildDamageTimeline } from '../src/DamageTimeline.js';
import { blastExtent, blastExtentOver, blastKinds, latestWreckEnd } from '../src/render/Explosion.js';
import { readFileSync } from 'node:fs';

// ------------------------------------------------------------- fixtures

const ev = (timestamp, event_type, ship_id, data = {}) => ({ timestamp, event_type, ship_id, data });
const blast = (t, id, position, radius_km = 15, killer = null) => [
  ev(t, 'ship_destroyed', id, { killer_id: killer }),
  ev(t, 'reactor_blast', id, { position, radius_km, yield_gj: 400 })
];

/** A small battle with every kind of highlight. */
function battleEvents() {
  return [
    ...[0, 0.5, 1, 1.5].map((dt, i) => ev(30 + dt, 'torpedo_launched', `alpha_${1 + (i % 2)}`, { target_id: 'beta_1' })),
    ev(60, 'shot_fired', 'beta_2', { target_id: 'alpha_1' }),
    ...[0, 0.5, 1].map(dt => ev(80 + dt, 'torpedo_impact', 'alpha_1', { target_id: 'beta_3', torpedo_id: `t${dt}`, total_damage_gj: 20 })),
    ev(99, 'torpedo_impact', 'alpha_1', { target_id: 'beta_1', torpedo_id: 'tk', total_damage_gj: 24 }),
    ev(100, 'magazine_cookoff', 'beta_1', { secondary_gj: 1.5, source_ship_id: 'alpha_1' }),
    ...blast(100, 'beta_1', [200000, 0, 0], 18.4, 'alpha_1'),
    ev(130, 'ship_dying', 'beta_2', { killer_id: 'alpha_2' }),
    ...blast(142, 'beta_2', [210000, 20000, 0], 13, 'alpha_2'),
    // Two ships on opposite fleets, same tick, far apart
    ...blast(170, 'beta_3', [220000, -20000, 0], 10.6, 'alpha_1'),
    ...blast(170, 'alpha_2', [-159000, 20000, 0], 15, 'beta_3'),
    // A chain: two neighbours a second apart
    ...blast(200, 'beta_4', [230000, 0, 0], 13, 'alpha_1'),
    ...blast(201, 'beta_5', [236000, 4000, 0], 13, 'alpha_1'),
  ].sort((a, b) => a.timestamp - b.timestamp);
}

/** Seeded dense battle: salvos, strikes and kills crowding each other. */
function denseEvents(seed = 7, n = 60) {
  let s = seed;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; };
  const events = [];
  const dead = new Set();
  for (let i = 0; i < n; i++) {
    const t = Math.round(rnd() * 400 * 2) / 2;
    const roll = rnd();
    const a = `alpha_${1 + Math.floor(rnd() * 6)}`;
    const b = `beta_${1 + Math.floor(rnd() * 6)}`;
    if (roll < 0.3) {
      for (let k = 0; k < 4; k++) events.push(ev(t + k * 0.5, 'torpedo_launched', a, { target_id: b }));
    } else if (roll < 0.6) {
      for (let k = 0; k < 3; k++) events.push(ev(t + k * 0.5, 'torpedo_impact', a, { target_id: b, torpedo_id: `x${i}_${k}`, total_damage_gj: 20 }));
    } else if (roll < 0.7) {
      events.push(ev(t, 'magazine_cookoff', b, { secondary_gj: 2 }));
    } else if (!dead.has(b)) {
      dead.add(b);
      if (rnd() < 0.4) events.push(ev(t - 8, 'ship_dying', b, { killer_id: a }));
      events.push(...blast(t, b, [200000 + 9000 * rnd(), 30000 * rnd(), 0], 10 + 8 * rnd(), a));
    }
  }
  return events.filter(e => e.timestamp >= 0).sort((x, y) => x.timestamp - y.timestamp);
}

function assertWellFormed(shots, label = '') {
  assert.ok(shots.length > 0);
  assert.equal(shots[0].t0, 0, `${label} starts at 0`);
  assert.equal(shots.at(-1).t1, Infinity, `${label} covers the rest of the battle`);
  shots.forEach((s, i) => {
    assert.ok(SHOT_TYPES.includes(s.type), `${label} known shot type ${s.type}`);
    assert.ok(s.t1 - s.t0 >= MIN_SHOT_S - 1e-9, `${label} shot ${i} (${s.type} ${s.t0}-${s.t1}) holds at least ${MIN_SHOT_S}s`);
    if (i) assert.equal(shots[i - 1].t1, s.t0, `${label} no gap or overlap before shot ${i}`);
    assert.equal(s.id, i);
  });
  // No cut inside a detonation lock
  for (const s of shots) {
    if (s.lockTo === undefined) continue;
    for (const o of shots) {
      const inside = o !== s && o.t0 > Math.max(s.t0, s.lockFrom) + 1e-9 && o.t0 < s.lockTo - 1e-9;
      assert.ok(!inside, `${label} cut at ${o.t0} inside the detonation lock ${s.lockFrom}-${s.lockTo}`);
    }
    assert.ok(s.t1 >= s.lockTo - 1e-9, `${label} detonation shot held to the end of its lock`);
  }
}

function stage(positions, { deaths = {}, lengths = {} } = {}) {
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.2, 800000);
  const listeners = {};
  const controls = {
    target: new THREE.Vector3(), enabled: true, update() { camera.lookAt(this.target); },
    addEventListener(type, fn) { listeners[type] = fn; }
  };
  const scene = {
    SCALE: 1 / 1000, currentTime: 0, delta: 1 / 60, destructionEffects: [], deathInfo: deaths,
    ships: new Map(Object.keys(positions).map(id => [id, { userData: { size: { length: lengths[id] || 10 }, faction: id.split('_')[0] } }])),
    getShipPosition: () => null
  };
  camera.position.set(0, 300, 400);
  const controller = new CameraController(camera, controls, scene);
  // Ships drift at constant velocity: state(t) is a pure function of t
  const states = t => Object.fromEntries(Object.entries(positions).map(([id, p]) => [id, {
    position: [p[0] + (p[3] || 0) * t, p[1] + (p[4] || 0) * t, p[2]], forward: [id.startsWith('alpha') ? 1 : -1, 0, 0],
    velocity: [p[3] || 0, p[4] || 0, 0], thrust: 1, destroyed: deaths[id]?.time <= t
  }]));
  return { camera, controls, scene, controller, listeners, states };
}

const POSITIONS = {
  alpha_1: [-200000, 0, 0, 300, 0], alpha_2: [-210000, 20000, 0, 300, 0],
  beta_1: [200000, 0, 0, 0, 0], beta_2: [210000, 20000, 0, 0, 0], beta_3: [220000, -20000, 0, 0, 0],
  beta_4: [230000, 0, 0, 0, 0], beta_5: [236000, 4000, 0, 0, 0]
};

// ----------------------------------------------------------- shot list

test('shot list: deterministic, covers the battle, every shot holds its minimum', () => {
  const events = battleEvents();
  const a = buildShotList(events, {});
  const b = buildShotList(structuredClone(events), {});
  assert.deepEqual(a, b, 'a pure function of the recording');
  assertWellFormed(a, 'replay');
  for (const seed of [1, 7, 42, 99]) {
    const dense = denseEvents(seed);
    assertWellFormed(buildShotList(dense, {}), `dense replay seed ${seed}`);
    assertWellFormed(buildShotList(dense, {}, { lookahead: false }), `dense live seed ${seed}`);
    assert.deepEqual(buildShotList(dense, {}), buildShotList(dense, {}));
  }
  // The shot at t is a function of t alone
  assert.equal(shotAt(a, 100.2), shotAt(a, 100.2));
  assert.equal(shotAt(a, -5), a[0]);
  assert.equal(shotAt(a, 1e9), a.at(-1));
  for (const s of a) assert.equal(shotAt(a, s.t0), s);
});

test('replay cuts early: run-in before a strike kill, hulk orbit before it blows, pull-back holds the blast', () => {
  const shots = buildShotList(battleEvents(), {});
  const at = t => shotAt(shots, t);
  // Salvo: over the launching ship's shoulder, starting before the first launch
  const salvo = at(30);
  assert.equal(salvo.type, 'shoulder');
  assert.equal(salvo.kind, 'salvo');
  assert.ok(salvo.t0 < 30 && salvo.subject.startsWith('alpha_') && salvo.target === 'beta_1');
  // First gun exchange
  assert.equal(at(60).kind, 'first_fire');
  assert.equal(at(60).subject, 'beta_2');
  // A strike without a kill: the run-in arrives before the first impact
  const strike = at(80);
  assert.deepEqual([strike.type, strike.kind, strike.subject], ['runin', 'strike', 'beta_3']);
  assert.ok(strike.t0 <= 80 - MIN_SHOT_S);
  // Kill at 100 by a torpedo strike: run-in, then the pull-back from just before the blast
  assert.deepEqual([at(97).type, at(97).kind, at(97).subject], ['runin', 'kill', 'beta_1']);
  const pull = at(100);
  assert.equal(pull.type, 'pullback');
  assert.equal(pull.t0, 100 - PULLBACK_PRE_S);
  assert.ok(pull.t1 >= 100 + DETONATION_LOCK_S);
  assert.equal(pull.kills[0].radiusKm, 18.4, 'blast radius from the reactor_blast event');
  // The cook-off on the same tick did not cut into the detonation
  assert.equal(at(100.5), pull);
  // A dying hulk is orbited until just before it detonates
  assert.deepEqual([at(135).type, at(135).kind, at(135).subject], ['orbit', 'dying', 'beta_2']);
  assert.equal(at(141.5).type, 'pullback');
  // Same tick, opposite fleets: one detonation shot, chosen by score then id - never both
  const both = shots.filter(s => s.type === 'pullback' && s.t === 170);
  assert.equal(both.length, 1);
  assert.equal(both[0].subject, 'alpha_2', 'the larger blast wins');
  // A chain of neighbours is one widening shot that holds for the last blast
  const chain = at(200.5);
  assert.deepEqual(chain.kills.map(k => k.id), ['beta_4', 'beta_5']);
  assert.ok(chain.t1 >= 201 + DETONATION_LOCK_S);
});

test('a kill credited to a dead ship is not filmed over its shoulder', () => {
  const events = [
    ...blast(50, 'alpha_1', [-200000, 0, 0], 15, 'beta_1'),
    ...blast(80, 'beta_2', [200000, 0, 0], 13, 'alpha_1'),   // alpha_1 died at 50
    ...blast(120, 'beta_3', [220000, 0, 0], 13, 'alpha_2'),  // alpha_2 lives
  ];
  const shots = buildShotList(events, {});
  const before = t => shotAt(shots, t - PULLBACK_PRE_S - 1);
  assert.deepEqual([before(80).type, before(80).subject], ['runin', 'beta_2'], 'incoming fire at the victim');
  assert.deepEqual([before(120).type, before(120).subject, before(120).target], ['shoulder', 'alpha_2', 'beta_3']);
});

test('old recording without the pass-5 events: frame-derived deaths still get a detonation shot', () => {
  const deaths = { beta_1: { time: 90, position: [200000, 0, 0] }, alpha_3: { time: 150 } };
  const legacy = [ev(40, 'torpedo_launched', 'alpha_1', {}), ev(40.5, 'torpedo_launched', 'alpha_1', {}), ev(41, 'torpedo_launched', 'alpha_2', {})];
  const shots = buildShotList(legacy, deaths);
  assertWellFormed(shots, 'legacy');
  // No reactor cause and no sim death spiral: the viewer coasts the wreck
  // dark for WRECK_DRIFT_S before it blows, and the shot waits with it
  const kill = shotAt(shots, 90 + WRECK_DRIFT_S);
  assert.equal(kill.type, 'pullback');
  assert.equal(kill.t, 90 + WRECK_DRIFT_S);
  assert.equal(kill.kills[0].radiusKm, 15, 'default blast radius');
  assert.equal(shotAt(shots, 150 + WRECK_DRIFT_S).subject, 'alpha_3');
  assert.equal(shotAt(shots, 40).kind, 'salvo');
  // Nothing at all: one wide shot
  const empty = buildShotList([], {});
  assert.deepEqual(empty.map(s => [s.type, s.t0, s.t1]), [['wide', 0, Infinity]]);
  assert.equal(buildShotList(undefined, undefined).length, 1);
  // Shake still has its sources
  assert.deepEqual(buildShakeSources(legacy, deaths).map(s => [s.t, s.ship]), [[95, 'beta_1'], [155, 'alpha_3']]);
  // ... and the camera frames it without the events' positions
  const { controller, states, camera } = stage(POSITIONS, { deaths });
  controller.setDirectorRecording({ events: legacy, deathInfo: deaths });
  controller.setDirector(true);
  controller.update(states(96), { time: 96 });
  assert.equal(controller.directorState().type, 'pullback');
  assert.ok(Number.isFinite(camera.position.x));
  // A shot whose subject never spawned falls back to the wide view
  controller.update(states(156), { time: 156 });
  assert.equal(controller.directorState().type, 'wide');
});

test('live mode has no lookahead: shots start at or after their event and the past never changes', () => {
  for (const events of [battleEvents(), denseEvents(3), denseEvents(11)]) {
    const full = buildShotList(events, {}, { lookahead: false });
    assertWellFormed(full, 'live');
    for (const s of full) {
      if (s.kind !== 'fleet') assert.ok(s.t0 >= s.t - 1e-9, `${s.type}/${s.kind} at ${s.t0} reacts to its event at ${s.t}`);
      assert.notEqual(s.kind, 'beauty', 'no foreshadowing shots without a future');
    }
    // Feed the battle in as it happens: what was on screen stays as it was
    const times = [...new Set(events.map(e => e.timestamp))];
    for (const now of times) {
      const partial = buildShotList(events.filter(e => e.timestamp <= now), {}, { lookahead: false });
      for (let t = 0; t <= now; t += 0.5) {
        const a = shotAt(partial, t);
        const b = shotAt(full, t);
        assert.deepEqual([a.type, a.kind, a.subject, a.t0], [b.type, b.kind, b.subject, b.t0],
          `shot at ${t} unchanged by events after ${now}`);
      }
    }
  }
  // Replay looks ahead where live cannot
  const replay = buildShotList(battleEvents(), {});
  const live = buildShotList(battleEvents(), {}, { lookahead: false });
  assert.equal(shotAt(replay, 97).type, 'runin');
  assert.ok(shotAt(live, 97).t0 >= 80, 'live is still on the last thing that happened');
  assert.equal(shotAt(live, 130.5).type, 'orbit', 'live reacts to the ship_dying event');
  assert.equal(shotAt(live, 142.5).type, 'pullback');
});

test('highlights: scored moments, next / previous stepping', () => {
  const events = battleEvents();
  const hs = buildHighlights(events, {});
  assert.deepEqual(hs, buildHighlights(events, {}));
  const kinds = new Set(hs.map(h => h.kind));
  for (const k of ['kill', 'dying', 'strike', 'salvo', 'cookoff', 'first_fire']) assert.ok(kinds.has(k), k);
  const kill = hs.find(h => h.kind === 'kill' && h.subject === 'beta_1');
  const salvo = hs.find(h => h.kind === 'salvo');
  assert.ok(kill.score > salvo.score, 'a kill outranks a salvo');
  const list = highlightList(events, {});
  assert.ok(list.every((h, i) => !i || h.t - list[i - 1].t >= 2), 'one entry per moment');
  assert.equal(list.find(h => h.t === 200).label, '2 kills');
  assert.equal(stepHighlight(list, 0, 1), 27, 'next: three seconds before the salvo');
  assert.equal(stepHighlight(list, 27, 1), 57);
  assert.equal(stepHighlight(list, 100, -1), 96, 'previous: back to the start of the moment on screen');
  assert.equal(stepHighlight(list, 96.5, -1), 77);
  assert.equal(stepHighlight(list, 1e6, 1), null);
  assert.equal(stepHighlight(list, 0, -1), null);
  assert.equal(stepHighlight([], 10, 1), null);
  // Disarmed: every gun/tube the recording saw hit is dead
  const disarm = [
    ev(10, 'mount_damaged', 'beta_1', { kind: 'gun', slot: 'weapon_0', destroyed: true }),
    ev(12, 'mount_damaged', 'beta_1', { kind: 'pd', slot: 'pd_0', destroyed: true }),
    ev(20, 'mount_damaged', 'beta_1', { kind: 'gun', slot: 'weapon_1', destroyed: false }),
    ev(40, 'mount_damaged', 'beta_1', { kind: 'gun', slot: 'weapon_1', destroyed: true }),
  ];
  assert.deepEqual(buildHighlights(disarm, {}).map(h => [h.kind, h.t]), [['disarmed', 40]]);
  // Closest approach from the trace
  const trace = [0, 10, 20].map(t => ({ t, ships: {
    alpha_1: { pos: [-100000 + t * 4000, 0, 0] }, beta_1: { pos: [100000 - t * 4000, 0, 0] } } }));
  assert.deepEqual(closestApproach(trace), { t: 20, a: 'alpha_1', b: 'beta_1', distanceKm: 40 });
  assert.equal(closestApproach(trace, 30), null);
  assert.equal(closestApproach(null), null);
});

// ------------------------------------------------------------- framing

test('poses are a function of battle time: scrubbing away and back gives the same camera', () => {
  const events = battleEvents();
  const run = times => {
    const { controller, states, camera, controls } = stage(POSITIONS);
    controller.setDirectorRecording({ events, deathInfo: {} });
    controller.setDirector(true);
    for (const t of times) controller.update(states(t), { time: t, dt: 0.1 });
    return [...camera.position.toArray(), ...controls.target.toArray()];
  };
  for (const t of [5, 31, 60.5, 82, 97, 100.2, 101, 135, 141.6, 171, 201.5, 300]) {
    assert.deepEqual(run([t]), run([0, 250, 100.3, 12, t]), `pose at ${t}`);
    assert.deepEqual(run([t]), run([t - 0.2, t - 0.1, t]), `played into ${t}`);
  }
});

test('every shot frames its subject from outside every hull and plume, with no jump inside a shot', () => {
  const events = battleEvents();
  const { controller, states, camera, controls } = stage(POSITIONS, { lengths: { beta_1: 16, alpha_1: 10.4 } });
  controller.setDirectorRecording({ events, deathInfo: {} });
  controller.setDirector(true);
  let prev = null;
  const seen = new Set();
  for (let t = 0; t <= 215; t += 0.25) {
    const ships = states(t);
    controller.update(ships, { time: t, dt: 0.25 });
    const shot = controller.director.shot;
    seen.add(shot.type);
    const world = controller._world(ships);
    world.deathTime = id => (events.find(e => e.event_type === 'ship_destroyed' && e.ship_id === id)?.timestamp ?? Infinity);
    const cam = camera.position.clone();
    if (controller.shakeOffset) cam.sub(controller.shakeOffset.position);
    camera.updateMatrixWorld();
    if (shot.subject && !shot.kills) {
      const ndc = world.pos(shot.subject).project(camera);
      assert.ok(Math.abs(ndc.x) < 0.9 && Math.abs(ndc.y) < 0.9 && ndc.z < 1, `${shot.type} at ${t}: subject in frame (${ndc.x.toFixed(2)}, ${ndc.y.toFixed(2)})`);
    }
    if (shot.type !== 'wide') {
      for (const o of obstaclesOf(world, t)) {
        const ab = o.b.clone().sub(o.a);
        const u = ab.lengthSq() > 1e-12 ? THREE.MathUtils.clamp(cam.clone().sub(o.a).dot(ab) / ab.lengthSq(), 0, 1) : 0;
        assert.ok(cam.distanceTo(o.a.clone().addScaledVector(ab, u)) >= o.r * 0.999, `${shot.type} at ${t}: camera inside a hull or plume`);
      }
    }
    assert.ok(cam.z > 0, 'camera stays on the +Z side of the battle plane');
    const dist = cam.distanceTo(controls.target);
    if (prev && prev.id === shot.id) {
      assert.ok(cam.distanceTo(prev.cam) < 0.25 * Math.max(dist, prev.dist), `${shot.type} at ${t}: eased, no teleport inside a shot`);
    }
    prev = { id: shot.id, cam, dist };
  }
  for (const type of ['wide', 'shoulder', 'runin', 'orbit', 'pullback']) assert.ok(seen.has(type), `${type} used`);
});

test('the pull-back ends far enough to fit the blast radius, and widens for a chain', () => {
  const events = battleEvents();
  const { controller, states, camera, controls } = stage(POSITIONS, { lengths: { beta_1: 16 } });
  controller.setDirectorRecording({ events, deathInfo: {} });
  controller.setDirector(true);
  const distAt = t => {
    controller.update(states(t), { time: t });
    const cam = camera.position.clone();
    if (controller.shakeOffset) cam.sub(controller.shakeOffset.position);
    return cam.distanceTo(new THREE.Vector3(200, 0, 0));
  };
  const near = distAt(99.2);
  const far = distAt(104);
  assert.ok(near < far * 0.8, 'starts close, pulls back');
  assert.ok(near >= 1.3 * 18.4 - 1e-6, 'never inside the blast shell');
  assert.ok(far >= fitDistance(18.4, 55, 16 / 9, 0.62) - 1.5, 'the whole shell fits the view');
  // The shell's rim projects inside the frame
  controller.update(states(104), { time: 104 });
  camera.updateMatrixWorld();
  for (const dir of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]) {
    const rim = new THREE.Vector3(200, 0, 0).addScaledVector(new THREE.Vector3(...dir), 18.4).project(camera);
    assert.ok(Math.abs(rim.x) < 1 && Math.abs(rim.y) < 1, `blast rim ${dir} in frame`);
  }
  // Chain at 200/201: by 204 both blasts are framed
  controller.update(states(204), { time: 204 });
  camera.updateMatrixWorld();
  for (const p of [[230, 0, 0], [236, 4, 0]]) {
    const ndc = new THREE.Vector3(...p).project(camera);
    assert.ok(Math.abs(ndc.x) < 0.6 && Math.abs(ndc.y) < 0.6);
  }
  assert.ok(camera.position.distanceTo(controls.target) > fitDistance(13, 55, 16 / 9, 0.62));
});

test('clearObstacles pushes the camera out of a hull sphere and a plume capsule', () => {
  const hull = { a: new THREE.Vector3(0, 0, 0), b: new THREE.Vector3(0, 0, 0), r: 5 };
  const plume = { a: new THREE.Vector3(-5, 0, 0), b: new THREE.Vector3(-30, 0, 0), r: 4 };
  const inside = clearObstacles(new THREE.Vector3(1, 1, 1), [hull, plume]);
  assert.ok(inside.length() >= 5);
  const inPlume = clearObstacles(new THREE.Vector3(-20, 0.5, 1), [hull, plume]);
  assert.ok(Math.hypot(inPlume.y, inPlume.z) >= 4, 'radially out of the plume');
  assert.equal(inPlume.x, -20);
  const onAxis = clearObstacles(new THREE.Vector3(-20, 0, 0), [hull, plume]);
  assert.ok(onAxis.z >= 4, 'dead centre leaves toward +Z');
  const clear = new THREE.Vector3(40, 40, 40);
  assert.deepEqual(clearObstacles(clear.clone(), [hull, plume]).toArray(), clear.toArray());
  assert.equal(shotPose(null, 0, {}), null);
});

// ---------------------------------------------------------- controller

test('manual input takes over at once; A hands the camera back with a wall-clock ease', () => {
  const { controller, states, camera, controls, listeners } = stage(POSITIONS);
  controller.setDirectorRecording({ events: battleEvents(), deathInfo: {} });
  controller.setDirector(true);
  controller.update(states(31), { time: 31 });
  assert.equal(controller.directorState().active, true);
  assert.equal(controller.directorState().shot, 'salvo', 'chip keeps its vocabulary');
  assert.equal(controller.directorState().type, 'shoulder');
  const directed = camera.position.clone();
  // The viewer drags: the director lets go on that very frame
  listeners.start();
  camera.position.set(10, 20, 30);
  controls.target.set(1, 2, 3);
  controller.update(states(31.1), { time: 31.1 });
  assert.deepEqual(camera.position.toArray(), [10, 20, 30]);
  assert.equal(controller.directorState().hold, 'user');
  assert.equal(controller.directorState().active, false);
  // A (setDirector): back to the director, easing from where the viewer left it
  controller.setDirector(true);
  controller.update(states(31.1), { time: 31.1 });
  assert.equal(controller.directorState().active, true);
  assert.ok(camera.position.distanceTo(new THREE.Vector3(10, 20, 30)) < 0.2 * directed.distanceTo(new THREE.Vector3(10, 20, 30)),
    'no teleport from the manual pose');
  controller.director.blend.start -= DIRECTOR_BLEND_S * 1000 + 1;
  controller.update(states(31.1), { time: 31.1 });
  const fresh = stage(POSITIONS);
  fresh.controller.setDirectorRecording({ events: battleEvents(), deathInfo: {} });
  fresh.controller.setDirector(true);
  fresh.controller.update(fresh.states(31.1), { time: 31.1 });
  assert.deepEqual(camera.position.toArray(), fresh.camera.position.toArray(), 'eased all the way onto the director pose');
  // Disarmed: the camera is the viewer's
  controller.setDirector(false);
  camera.position.set(1, 2, 3);
  controller.update(states(40), { time: 40 });
  assert.deepEqual(camera.position.toArray(), [1, 2, 3]);
  assert.equal(controller.directorState().shot, null);
});

// --------------------------------------------------------------- shake

test('shake decays on battle time and is a pure function of it', () => {
  const sources = buildShakeSources(battleEvents(), {});
  assert.ok(sources.every((s, i) => !i || s.t >= sources[i - 1].t));
  const one = [{ t: 100, position: [0, 0, 0], ship: 'x', energyGj: 500, reachKm: 20 }];
  const amp = (t, d = 10, src = one) => shakeAt(src, t, () => d).amp;
  assert.equal(amp(99.99), 0, 'nothing before the blast');
  assert.ok(amp(100) > 0.01);
  let last = Infinity;
  for (let t = 100; t <= 100 + SHAKE_LIFE_S; t += 0.1) {
    const a = amp(t);
    assert.ok(a <= last, `monotonic decay at ${t}`);
    last = a;
  }
  assert.equal(amp(100 + SHAKE_LIFE_S), 0, 'gone after its life');
  assert.ok(amp(100.2, 10) > amp(100.2, 200), 'weaker far away');
  assert.equal(amp(100.2, 10), amp(100.2, 19), 'full strength inside the reach');
  assert.ok(amp(100.2, 10, [{ ...one[0], energyGj: 2000 }]) > amp(100.2), 'scales with energy');
  assert.ok(shakeAt([one[0], one[0], one[0], one[0], one[0], one[0]], 100, () => 1).amp <= 0.02 + 1e-12, 'capped');
  assert.deepEqual(shakeAt(one, 100.37, () => 30), shakeAt(one, 100.37, () => 30));
  assert.equal(shakeAt([], 5, () => 1).amp, 0);
  // Handheld drift and the impact push are functions of battle time too
  assert.deepEqual(handheldAt(12.5), handheldAt(12.5));
  assert.ok(Math.abs(handheldAt(12.5).x) < 0.006);
  assert.equal(impactPulse([100], 99), 0);
  assert.ok(impactPulse([100], 100) === 1 && impactPulse([100], 100.5) < 0.4 && impactPulse([100], 103) === 0);
});

test('controller shake: on the camera for the draw only, never accumulates, identical after seeking away and back', () => {
  const events = battleEvents();
  const { controller, states, camera, controls } = stage(POSITIONS);
  controller.setDirectorRecording({ events, deathInfo: {} });
  const HOME = new THREE.Vector3(200, 10, 40);
  const place = () => { camera.position.copy(HOME); controls.target.set(200, 0, 0); camera.lookAt(controls.target); };
  // One rendered frame, as main.js does it: update, shake on, draw, shake off
  let drawn = null;
  const frame = t => {
    controller.update(states(t), { time: t });
    controller.applyShake(states(t), t);
    drawn = { position: camera.position.clone(), quaternion: camera.quaternion.clone(), shaking: !!controller.shakeOffset };
    controller.removeShake();
  };
  // Manual free camera near the blast at (200, 0, 0): shake works without the director
  place();
  const rest = camera.quaternion.clone();
  frame(95);
  assert.equal(drawn.shaking, false, 'calm before the blast');
  frame(100.3);
  assert.ok(drawn.shaking, 'shaking 0.3 s after the detonation');
  const shaken = drawn.position.clone();
  assert.ok(shaken.distanceTo(HOME) > 1e-4);
  // Between frames the camera is exactly where it was put: no residual
  assert.equal(controller.shakeOffset, null);
  assert.deepEqual(camera.position.toArray(), HOME.toArray());
  assert.deepEqual(controls.target.toArray(), [200, 0, 0]);
  assert.ok(camera.quaternion.angleTo(rest) < 1e-12, 'orientation restored too');
  // Paused on the same battle time: the same drawn pose every frame
  for (let i = 0; i < 30; i++) frame(100.3);
  assert.ok(drawn.position.distanceTo(shaken) < 1e-12, 'no accumulation');
  assert.deepEqual(camera.position.toArray(), HOME.toArray());
  // Something places the camera absolutely between two shaken frames (a
  // ship focus, the fleet view, a capture harness): it stays exactly there
  controller.focusOnShip('beta_1', states(100.4));
  const focused = camera.position.clone();
  frame(100.4);
  assert.ok(drawn.shaking);
  assert.deepEqual(camera.position.toArray(), focused.toArray(), 'no stale offset taken off an absolute placement');
  controller.setMode('free');
  // A draw that never finished (shake on, no removeShake) is cleaned up by the next update
  place();
  controller.applyShake(states(100.3), 100.3);
  controller.update(states(100.3), { time: 100.3 });
  assert.deepEqual(camera.position.toArray(), HOME.toArray());
  // Away (decayed) and back
  frame(150);
  assert.equal(drawn.shaking, false);
  frame(100.3);
  assert.ok(drawn.position.distanceTo(shaken) < 1e-12, 'same frame after seeking away and back');
  frame(101.5);
  assert.ok(drawn.position.distanceTo(HOME) < shaken.distanceTo(HOME), 'weaker later in battle time');
  // With the director flying: same guarantee
  controller.setDirector(true);
  frame(100.3);
  const a = drawn.position.clone();
  frame(60);
  frame(100.3);
  assert.ok(drawn.position.distanceTo(a) < 1e-12);
  // The scene no longer has a wall-clock shake of its own, and main.js wraps the draw
  const scene = readFileSync(new URL('../src/SceneManager.js', import.meta.url), 'utf8');
  assert.doesNotMatch(scene, /this\.shake\b/);
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(main, /applyShake\([^;]*;\s*this\.scene\.render\(\);\s*if \(shaking\) this\.cameraController\.removeShake\(\)/);
  assert.doesNotMatch(main, /scene\.shake/);
});

test('controller in live mode reacts to events; when the battle ends it may look ahead', () => {
  const events = battleEvents();
  const { controller, states } = stage(POSITIONS);
  const typeAt = t => { controller.update(states(t), { time: t }); return controller.directorState().type; };
  controller.setDirectorRecording({ events: events.filter(e => e.timestamp <= 97), deathInfo: {}, live: true });
  controller.setDirector(true);
  assert.notEqual(typeAt(97), 'runin', 'no run-in to a kill that has not happened');
  assert.ok(controller.highlights.every(h => h.t <= 97));
  controller.setDirectorRecording({ events: events.filter(e => e.timestamp <= 101), deathInfo: {}, live: true });
  assert.equal(typeAt(102.6), 'pullback', 'the blast is on screen once the shot before it has had its minimum');
  controller.setDirectorRecording({ events, deathInfo: {}, live: false });
  assert.equal(typeAt(97), 'runin');
  assert.equal(typeAt(100.5), 'pullback');
  // The legacy cut list still drives a controller that was never given a recording
  const legacy = stage(POSITIONS);
  legacy.scene.getShipPosition = id => new THREE.Vector3(...POSITIONS[id].slice(0, 3)).multiplyScalar(1 / 1000);
  legacy.controller.setDirector(true);
  legacy.controller.setDirectorCuts([{ kind: 'kill', t: 10, until: 18, subject: 'beta_1', others: [] }]);
  legacy.controller.update(legacy.states(12), { time: 12, dt: 1 / 60 });
  assert.equal(legacy.controller.directorState().shot, 'kill');
  assert.equal(legacy.controller.directorState().type, null);
});

test('kills are keyed on the time the viewer detonates the ship, the rule the effects use', () => {
  // The trace flags the ship destroyed one frame after its ship_destroyed event
  assert.equal(blastFrameTime(102, 101), 102);
  assert.equal(blastFrameTime(null, 101), 101, 'no frame yet (live): the event tick');
  assert.equal(blastFrameTime(110, 101), 101, 'an unrelated later frame is not this blast');
  assert.equal(shipDetonationTime({ time: 102, reactorCause: true }, 101), 102);
  assert.equal(shipDetonationTime({ time: 102, wasDying: true }, 101), 102);
  assert.equal(shipDetonationTime(null, 101), 101);
  assert.equal(shipDetonationTime({ time: 90 }), 90 + WRECK_DRIFT_S, 'old recording: dark coast first');
  assert.equal(shipDetonationTime(null), null);
  assert.equal(wreckDriftS({ reactorCause: true }), 0);
  assert.equal(latestWreckEnd({ a: { time: 700 } }), 700 + WRECK_DRIFT_S + 40, 'Explosion.js uses the same drift rule');
  // The same time the blast shell is dated at
  const trace = [100, 101, 102, 103].map(t => ({ t, ships: {
    beta_1: { pos: [200000 - 300 * (t - 100), 0, 0], vel: [-300, 0, 0], fwd: [-1, 0, 0], hull: 50, destroyed: t >= 102 },
    alpha_1: { pos: [-200000, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], hull: 100, destroyed: false } } }));
  const events = blast(101, 'beta_1', [199700, 0, 0], 18.4, 'alpha_1');
  const deaths = { beta_1: { time: 102, position: [199400, 0, 0], velocity: [-300, 0, 0], reactorCause: true } };
  const timeline = buildDamageTimeline({ events, simTrace: trace, ships: { beta_1: {}, alpha_1: {} } });
  const shellT = timeline.blasts[0].t;
  assert.equal(shellT, 102);
  const kill = buildHighlights(events, deaths).find(h => h.kind === 'kill');
  assert.equal(kill.t, shellT, 'director and blast shell agree');
  assert.equal(kill.eventT, 101);
  const shots = buildShotList(events, deaths);
  const pull = shots.find(s => s.type === 'pullback');
  assert.equal(pull.t0, shellT - PULLBACK_PRE_S, 'pull-back starts a second before the flash, not before the event');
  assert.ok(pull.lockTo >= shellT + DETONATION_LOCK_S);
  assert.equal(buildShakeSources(events, deaths)[0].t, shellT, 'and so does the shake');
  // Live: the shot may start at the event tick (all it knows), never before
  const live = buildShotList(events, {}, { lookahead: false }).find(s => s.type === 'pullback');
  assert.equal(live.t0, 101);
  // The camera is still close at the event tick and pulls back with the flash,
  // centred on the wreck as it coasts on
  const { controller, camera, controls } = stage({ alpha_1: [-200000, 0, 0], beta_1: [200000, 0, 0] }, { deaths });
  controller.setDirectorRecording({ events, deathInfo: deaths });
  controller.setDirector(true);
  const stateAt = t => ({ alpha_1: { position: [-200000, 0, 0], forward: [1, 0, 0], thrust: 0 },
    beta_1: { position: [200000 - 300 * (t - 100), 0, 0], forward: [-1, 0, 0], thrust: 0, destroyed: t >= 102 } });
  const distAt = t => { controller.update(stateAt(t), { time: t }); return camera.position.distanceTo(controls.target); };
  const close = distAt(101.2);
  assert.ok(Math.abs(distAt(101.35) - close) < 0.01 * close, 'still close after the event tick: the flash is a frame later');
  const atFlash = distAt(102.0);
  const wide = distAt(103.3);
  assert.ok(atFlash > close && atFlash < wide, 'moving out as it blows');
  assert.ok(wide > 1.3 * close, 'out with the blast');
  controller.update(stateAt(105), { time: 105 });
  assert.ok(Math.abs(controls.target.x - (199.4 - 0.3 * 3)) < 0.5, 'centre coasts with the wreck');
});

test('the pull-back fits what the death blast still draws, asked of the effect', () => {
  const K = blastKinds().reactor;
  assert.equal(blastExtent('reactor', 12, -1), 0);
  assert.equal(blastExtent('nope', 12, 1), 0);
  assert.equal(blastExtent('reactor', 12, K.duration + 1), 0);
  // Early: the shock front, far beyond the fireball; later only the fireball and its lobes
  const early = blastExtent('reactor', 12, 0.4);
  const late = blastExtent('reactor', 12, 3);
  assert.ok(early > 100, `shock front ${early.toFixed(0)}`);
  assert.ok(late > 12 * K.fireR && late < 0.5 * early, `fireball ${late.toFixed(1)}`);
  assert.equal(blastExtent('reactor', 6, 3), late / 2, 'linear in the blast scale');
  assert.ok(blastExtentOver('reactor', 12, 1.5, 5) >= late);
  // the bright ejecta ring at +1.2 s is wider than the fireball that outlasts it
  assert.ok(blastExtentOver('reactor', 12, 1.0, 1.6) > 1.5 * blastExtentOver('reactor', 12, 2.0, 5));
  assert.ok(blastExtentOver('meltdown', 12, 1.5, 5) > blastExtentOver('reactor', 12, 1.5, 5));
  // The camera uses it: a larger visual extent pulls back further
  const events = battleEvents();
  const farAt = extent => {
    const { controller, states, camera } = stage(POSITIONS, { lengths: { beta_1: 16 } });
    controller.sceneManager.deathBlastExtent = () => extent;
    controller.setDirectorRecording({ events, deathInfo: {} });
    controller.setDirector(true);
    controller.update(states(104.5), { time: 104.5 });
    return camera.position.distanceTo(new THREE.Vector3(200, 0, 0));
  };
  assert.ok(farAt(0) >= fitDistance(18.4, 55, 16 / 9, 0.62, 1.15) - 1e-6, 'never less than the recorded radius');
  assert.ok(farAt(40) >= fitDistance(40, 55, 16 / 9, 0.62, 1.15) - 1e-6);
  assert.ok(farAt(40) > 1.5 * farAt(0));
});

test('plume wash with a real cost and surrenders are highlights with their own shots', () => {
  const wash = (t, phase, heat, fittings = []) => ev(t, 'plume_wash', 'alpha_2',
    { source_ship_id: 'alpha_1', phase, heat_gj: heat, fittings_hit: fittings, armor_ablation_cm: 0.2 });
  const events = [
    // a brush with no cost: not a highlight
    ev(10, 'plume_wash', 'beta_2', { source_ship_id: 'beta_1', phase: 'enter', heat_gj: 3, fittings_hit: [] }),
    ev(12, 'plume_wash', 'beta_2', { source_ship_id: 'beta_1', phase: 'exit', heat_gj: 6, fittings_hit: [] }),
    wash(30, 'enter', 12.8), wash(35, 'sustain', 80, ['gun:weapon_0']), wash(55, 'sustain', 385, ['gun:weapon_0', 'pd:pd_0']),
    wash(69, 'exit', 463),
    ev(40, 'plume_wash', 'torp_1', { source_ship_id: 'alpha_1', phase: 'torpedo_blinded' }),
    ev(150, 'surrender', 'beta_3', { hull_pct: 40 }),
    ev(150, 'surrender', 'beta_4', { hull_pct: 80 }),
    ev(300, 'surrender', 'alpha_2', { hull_pct: 10 }),
  ];
  const hs = buildHighlights(events, {});
  assert.deepEqual(hs.map(h => [h.kind, h.t, h.subject]), [
    ['plume_wash', 30, 'alpha_2'], ['surrender', 150, 'beta_3'], ['surrender', 300, 'alpha_2']]);
  assert.equal(hs[0].target, 'alpha_1');
  assert.deepEqual(hs[1].ships, ['beta_3', 'beta_4']);
  const shots = buildShotList(events, {});
  assertWellFormed(shots, 'wash');
  const w = shotAt(shots, 31);
  assert.deepEqual([w.type, w.kind, w.cast], ['wide', 'plume_wash', ['alpha_1', 'alpha_2']]);
  assert.ok(w.t0 < 30, 'replay: on the pair before the ship enters the jet');
  assert.deepEqual([shotAt(shots, 151).type, shotAt(shots, 151).cast], ['wide', ['beta_3', 'beta_4']]);
  assert.deepEqual([shotAt(shots, 301).type, shotAt(shots, 301).subject], ['orbit', 'alpha_2']);
  assert.ok(highlightList(events, {}).some(h => h.label === 'plume wash'));
  assert.ok(highlightList(events, {}).some(h => h.label === '2 surrender'));
  // Live: a highlight only once the cost is on record, and the past never changes
  const live = buildShotList(events, {}, { lookahead: false });
  assertWellFormed(live, 'wash live');
  assert.equal(shotAt(live, 31).kind, 'fleet');
  assert.deepEqual([shotAt(live, 36).kind, shotAt(live, 36).t0], ['plume_wash', 35]);
  for (const now of [30, 35, 55, 69, 150]) {
    const partial = buildShotList(events.filter(e => e.timestamp <= now), {}, { lookahead: false });
    for (let t = 0; t <= now; t += 0.5) {
      assert.deepEqual([shotAt(partial, t).kind, shotAt(partial, t).t0], [shotAt(live, t).kind, shotAt(live, t).t0], `live prefix at ${t} / ${now}`);
    }
  }
  // The two-shot frames the emitter and the washed ship together
  const { controller, states, camera } = stage({ alpha_1: [0, 0, 0], alpha_2: [-9000, 500, 0], beta_3: [300000, 0, 0], beta_4: [305000, 0, 0] });
  controller.setDirectorRecording({ events, deathInfo: {} });
  controller.setDirector(true);
  controller.update(states(32), { time: 32 });
  camera.updateMatrixWorld();
  for (const p of [[0, 0, 0], [-9, 0.5, 0]]) {
    const ndc = new THREE.Vector3(...p).project(camera);
    assert.ok(Math.abs(ndc.x) < 0.7 && Math.abs(ndc.y) < 0.7, 'emitter and washed ship in frame');
  }
  assert.ok(camera.position.distanceTo(new THREE.Vector3(-4.5, 0.25, 0)) < 200, 'a close two-shot, not the fleet view');
});
