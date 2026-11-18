import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';

import {
  SCALE_MODES, DEFAULT_SCALE_MODE, FLOOR_SHARE, TRUE_AT, FREE_TRUE_K, FREE_REF_M,
  trueLengthM, kitLength, trueK, normalizeScaleMode, nextScaleMode, floorShare, screenShare, shownShare,
  magnification, magnificationAt, trueRegimeDistance, shipScale, freeScale, framingLength, bakeScale, BAKE_STEP
} from '../src/render/Scale.js';
import { REAL_LENGTH_M } from '../src/TrackDisplay.js';
import { HULL_CLASSES, assembleHull, hullSize, buildHull, buildShipMaterials } from '../src/render/Hulls.js';
import { SceneManager } from '../src/SceneManager.js';
import { CameraController } from '../src/CameraController.js';
import { BattleLoader } from '../src/BattleLoader.js';
import { buildDamageTimeline, shipStateAt, localBasis, blastFront } from '../src/DamageTimeline.js';
import { DamageFX } from '../src/render/DamageFX.js';
import { Blast, ShipDestruction, blastExtentOver } from '../src/render/Explosion.js';
import {
  WeaponTimeline, makeSample, sampleTrack, closestPass, passClearance, missClearance,
  TERMINAL_BEND_S, HULL_RADIUS_PER_LENGTH, MISS_CLEAR_RADII, MISS_CLEAR_M, MISS_CLEAR_MAX_M, passPush
} from '../src/render/WeaponTimeline.js';
import { buildShotList, PULLBACK_PRE_S, PULLBACK_POST_S } from '../src/Director.js';
import { shotPose, fitDistance } from '../src/DirectorCamera.js';
import { closeUpK, setCloseRangeUnit } from '../src/render/CloseRange.js';
import { quantiseShadowRadius, fitSunShadow, shadowBasis, SHADOW_MIN_RADIUS } from '../src/render/sunShadow.js';

const FLEET = JSON.parse(readFileSync(new URL('../../data/fleet_ships.json', import.meta.url), 'utf8'));
const MOUNTS = JSON.parse(readFileSync(new URL('../../data/hull_mounts.json', import.meta.url), 'utf8'));
const FIX = JSON.parse(readFileSync(new URL('fixtures/scale_hits.json', import.meta.url), 'utf8'));
const FOV = 55;
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// Effect objects bake small canvas textures (glow, puffs, plating): a
// canvas that accepts every 2D call is enough to build them without a DOM
const ctx2d = new Proxy({}, {
  get: (t, prop) => (prop === 'createImageData' || prop === 'getImageData'
    ? (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h })
    : prop === 'measureText' ? () => ({ width: 10 }) : prop === 'canvas' ? fakeCanvas() : () => ctx2d),
  set: () => true
});
function fakeCanvas() { return { width: 4, height: 4, style: {}, getContext: () => ctx2d }; }
globalThis.document ??= { createElement: () => fakeCanvas() };

// ------------------------------------------------------------------- rule

test('real hull lengths are the engine\'s, and true scale draws exactly them', () => {
  for (const type of Object.keys(FLEET.ships)) {
    assert.equal(REAL_LENGTH_M[type], FLEET.ships[type].hull.length_m, type);
    assert.equal(trueLengthM(type), MOUNTS.hulls[type].length_m, type);
    // k x kit length = the real length in scene km
    near(trueK(type) * kitLength(type) * 1000, FLEET.ships[type].hull.length_m, 1e-9);
    assert.equal(kitLength(type), hullSize(type).length);
    // the mounts export's own scale (metres per kit unit), to its 4 decimals
    near(trueK(type) * 1000, MOUNTS.hulls[type].metres_per_scene_unit, 6e-5);
  }
  assert.ok(HULL_CLASSES.every(t => REAL_LENGTH_M[t] > 0));
  // the nominal free-effect scale is the classes' mean, to a few percent
  const mean = HULL_CLASSES.reduce((s, t) => s + trueK(t), 0) / HULL_CLASSES.length;
  assert.ok(Math.abs(FREE_TRUE_K / mean - 1) < 0.1, `${FREE_TRUE_K} vs ${mean}`);
});

test('the three modes: cinematic is 1, true is the real hull at any range, auto is real up close', () => {
  assert.deepEqual(SCALE_MODES, ['auto', 'true', 'cinematic']);
  assert.equal(DEFAULT_SCALE_MODE, 'auto');
  assert.equal(normalizeScaleMode('TRUE'), 'true');
  assert.equal(normalizeScaleMode('real'), 'true');
  assert.equal(normalizeScaleMode('nonsense'), 'auto');
  assert.equal(normalizeScaleMode(null), 'auto');
  assert.deepEqual([nextScaleMode('auto'), nextScaleMode('true'), nextScaleMode('cinematic')], ['true', 'cinematic', 'auto']);
  for (const type of HULL_CLASSES) {
    for (const d of [0.05, 1, 30, 300, 3000]) {
      assert.equal(shipScale('cinematic', type, d, FOV), 1);
      assert.equal(shipScale('true', type, d, FOV), trueK(type));
      assert.ok(shipScale('auto', type, d, FOV) >= trueK(type));
    }
    const dTrue = trueRegimeDistance(REAL_LENGTH_M[type], FOV);
    // inside the near regime: EXACTLY the real size, not approximately
    for (const u of [0.01, 0.3, 0.9, 1]) assert.equal(shipScale('auto', type, dTrue * u, FOV), trueK(type), `${type} at ${u} of the true regime`);
    assert.ok(shipScale('auto', type, dTrue * 1.01, FOV) > trueK(type));
    // every close-up the viewer frames (a few hull lengths out) is inside it
    assert.ok(dTrue > 10 * REAL_LENGTH_M[type] / 1000, `${type}: true regime ${dTrue} km`);
    assert.equal(framingLength('auto', type), REAL_LENGTH_M[type] / 1000);
    assert.equal(framingLength('true', type), REAL_LENGTH_M[type] / 1000);
    assert.equal(framingLength('cinematic', type), kitLength(type));
  }
  assert.equal(freeScale('cinematic', 100, FOV), 1);
  assert.equal(freeScale('true', 100, FOV), FREE_TRUE_K);
  assert.equal(freeScale('auto', 0.2, FOV), FREE_TRUE_K);
  assert.ok(freeScale('auto', 300, FOV) > FREE_TRUE_K);
  // unknown hull types (older recordings) fall back to a destroyer, never NaN
  assert.equal(shipScale('true', 'mystery', 10, FOV), trueK('destroyer'));
});

test('auto rule: continuous with no kink, monotonic, a floor at distance, never smaller than real', () => {
  for (const L of [65, 100, 175, 275]) {
    const F = floorShare(L), S1 = TRUE_AT * F;
    // value and slope match where the regimes meet
    near(shownShare(S1, F), S1, 1e-15);
    const h = S1 * 1e-5;
    near((shownShare(S1, F) - shownShare(S1 - h, F)) / h, 1, 1e-3);
    near((shownShare(S1 + h, F) - shownShare(S1, F)) / h, 1, 1e-9);
    // monotonic in the real size on screen, i.e. a hull never GROWS on
    // screen as the camera backs off; and it tends to the floor
    let prev = Infinity;
    for (let d = 0.02; d < 5e4; d *= 1.07) {
      const s = screenShare(L / 1000, d, FOV), shown = shownShare(s, F);
      assert.ok(shown <= prev + 1e-15, `shown share grew at ${d} km`);
      assert.ok(shown >= s - 1e-15 && shown >= F - 1e-15 || s >= S1, 'never under the real size or the floor');
      assert.ok(magnification(s, F) >= 1);
      // magnified hulls stay under S1 of the viewport: two of them overlap
      // only when their centres are that close on screen
      if (s < S1) assert.ok(shown < S1);
      prev = shown;
    }
    near(shownShare(1e-9, F), F, 1e-9);
    assert.equal(shownShare(0, F), F);
    assert.equal(magnification(Infinity, F), 1);
  }
  // the floor: a corvette's is FLOOR_SHARE of the viewport height, bigger hulls get bigger dots
  near(floorShare(65), FLOOR_SHARE);
  assert.ok(floorShare(275) > floorShare(175) && floorShare(175) > floorShare(65));
  assert.ok(floorShare(275) < 2.2 * floorShare(65));
  // at fleet range the drawn hull covers its floor: 300 km out a corvette is ~14 px of 720
  const k300 = shipScale('auto', 'corvette', 300, FOV);
  const px = screenShare(kitLength('corvette') * k300, 300, FOV) * 720;
  assert.ok(px > 13.9 && px < 14.3, `${px} px`);
  // the rule is in shares of the viewport height: nothing to pass a resolution to
  assert.equal(shipScale.length, 4);
  // a narrower lens (the hero shot's 20 degrees) keeps the hull real from further away
  assert.ok(trueRegimeDistance(65, 20) > 2.5 * trueRegimeDistance(65, FOV));
  near(magnificationAt(FREE_REF_M, trueRegimeDistance(FREE_REF_M, FOV), FOV), 1, 1e-12);
});

test('two ships a few km apart: real size, and their drawn hulls cannot touch', () => {
  // camera 1 km from a battleship, a destroyer 4 km beyond it: both real size
  for (const [type, d] of [['battleship', 1], ['destroyer', 2.0], ['corvette', 1.5]]) {
    assert.equal(shipScale('auto', type, d, FOV), trueK(type), `${type} at ${d} km`);
  }
  // from 40 km both are magnified, yet the pair (4 km apart) still does not overlap
  const a = kitLength('battleship') * shipScale('auto', 'battleship', 40, FOV);
  const b = kitLength('destroyer') * shipScale('auto', 'destroyer', 40, FOV);
  assert.ok(a > 0.2 && b > 0.125, 'magnified');
  assert.ok((a + b) / 2 < 4, `drawn half-lengths ${a / 2} + ${b / 2} km < 4 km apart`);
});

// ------------------------------------------------------- hits and misses

/** Real hull kits of the fixture's ships, as DamageFX sees them (no renderer). */
function kitShip(type) {
  const size = hullSize(type);
  const kit = assembleHull(type, size, SceneManager.prototype.getEngineConfig.call(null, type, size), 1234);
  const g = new THREE.Group();
  g.userData = { size, shipType: type, parts: kit.parts };
  return g;
}
function fixtureLoader() {
  const loader = new BattleLoader();
  loader.simTrace = FIX.sim_trace;
  loader.events = FIX.events;
  loader.ships = Object.fromEntries(Object.entries(FIX.ship_types).map(([id, type]) => [id, { type, faction: id.split('_')[0] }]));
  loader.buildProjectileHitMap(); loader.buildLastProjectileStates();
  loader.buildTorpedoTimeline(); loader.buildPDBeamSegments();
  return loader;
}
function stubFx(loader, kOf) {
  const groups = new Map(Object.entries(FIX.ship_types).map(([id, type]) => [id, kitShip(type)]));
  const scene = new THREE.Scene();
  const sm = { ships: groups, scene, SCALE: 0.001, kOf, _spawnBlast: (o) => o };
  const fx = Object.create(DamageFX.prototype);
  const visual = new Map(loader.getVisualImpacts(-Infinity, Infinity).map(h => [h.projectile_id, h.timestamp]));
  Object.assign(fx, {
    sm, scene, SCALE: 0.001, views: new Map(), shipIds: [], blastSpawned: new Set(),
    planeGeom: new THREE.PlaneGeometry(1, 1), coneGeom: new THREE.CylinderGeometry(1, 0.16, 1, 8, 1, true),
    craterMat: new THREE.MeshBasicMaterial(), ventMat: new THREE.MeshBasicMaterial(),
    raycaster: new THREE.Raycaster(), probe: new THREE.Mesh(undefined, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })),
    _cands: [], _hits: [], _st: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false },
    _keptWear: new Map(), spall: { hide() {}, flush() {} }, spallOwner: [],
    timeline: buildDamageTimeline({ events: loader.events, sim_trace: loader.simTrace },
      { impactTime: (e) => visual.get(e.data?.projectile_id) ?? e.timestamp })
  });
  fx.probe.updateMatrixWorld();
  return { fx, groups };
}
/** Point (m) in a ship's hull frame at time t: [starboard-port x, dorsal y, nose z]. */
function hullFrame(trace, shipId, t, p) {
  const st = shipStateAt(trace, shipId, t);
  const B = localBasis(st.fwd, null, st.up);
  const r = [p[0] - st.pos[0], p[1] - st.pos[1], p[2] - st.pos[2]];
  const dot = (a) => r[0] * a[0] + r[1] * a[1] + r[2] * a[2];
  return [dot(B.x), dot(B.y), dot(B.z)];
}
const axisDistance = (l, L) => Math.hypot(l[0], l[1], Math.max(0, Math.abs(l[2]) - L / 2));

test('recorded hit points are NOT on the hull (the engine gates at 500 m); the fixture says by how much', () => {
  // the measurement over the whole recording the fixture was cut from
  const s = FIX.summary;
  assert.ok(s.hits > 200 && s.misses > 200);
  assert.ok(s.hit_axis_distance_m.median > 25 && s.hit_axis_distance_m.p90 > 400, JSON.stringify(s.hit_axis_distance_m));
  assert.ok(s.hits_inside_hull_radius < 0.5 * s.hits, 'most recorded impact points are off the hull');
  assert.ok(s.misses_through_hull > 0.15 * s.misses, 'and a good share of misses run through it');
  // and on the fixture's own rounds, recomputed here from the recording data
  const loader = fixtureLoader();
  const hits = FIX.rounds.filter(r => r.kind === 'hit');
  assert.equal(hits.length, 8);
  const visual = new Map(loader.getVisualImpacts(-Infinity, Infinity).map(h => [h.projectile_id, h]));
  let worst = 0;
  for (const r of hits) {
    const h = visual.get(r.id);
    const L = REAL_LENGTH_M[FIX.ship_types[r.target]];
    const d = axisDistance(hullFrame(loader.simTrace, r.target, h.timestamp, h.impact_position), L);
    assert.ok(Math.abs(d - r.recorded_axis_distance_m) < 1.5, `${r.id}: ${d} vs ${r.recorded_axis_distance_m}`);
    worst = Math.max(worst, d);
  }
  assert.ok(worst > 1000, `the fixture includes a hit recorded ${worst} m from its target's axis`);
});

test('a hit is DRAWN on the true-size hull: within the hull box, and the round ends exactly there', () => {
  const loader = fixtureLoader();
  const types = FIX.ship_types;
  const { fx } = stubFx(loader, (id) => trueK(types[id]));
  const timeline = new WeaponTimeline(loader);
  const P = new THREE.Vector3(), N = new THREE.Vector3(), out = makeSample(), raw = makeSample();
  let maxBendDeg = 0, maxShift = 0, maxSkin = 0;
  for (const r of FIX.rounds.filter(x => x.kind === 'hit')) {
    const tr = timeline.slugs.get(r.id);
    assert.ok(tr?.impact, r.id);
    const mark = fx.surfaceFor(r.id, P, N);
    assert.ok(mark, `${r.id} has a crater`);
    assert.equal(mark.shipId, r.target);
    const type = types[r.target], L = REAL_LENGTH_M[type], size = hullSize(type);
    const halfW = size.width / 2 * trueK(type) * 1000;      // the drawn half-beam (m)
    // in the target's hull frame at the impact time: inside the real hull's
    // box (fittings stand off the plating: one hull radius of margin) ...
    const l = hullFrame(loader.simTrace, r.target, mark.t, [P.x * 1000, P.y * 1000, P.z * 1000]);
    assert.ok(Math.abs(l[2]) <= L / 2 * 1.08, `${r.id}: ${l[2]} m along a ${L} m hull`);
    assert.ok(Math.hypot(l[0], l[1]) <= halfW * 2, `${r.id}: ${Math.hypot(l[0], l[1])} m off the axis of a ${2 * halfW} m beam`);
    // ... and within TOLERANCE of the engine's own hull surface (capsule,
    // radius L / 8): the drawn hull and the simulated one are the same size
    const skin = Math.abs(axisDistance(l, L) - L * HULL_RADIUS_PER_LENGTH);
    maxSkin = Math.max(maxSkin, skin);
    assert.ok(skin <= 0.35 * L, `${r.id}: drawn impact ${skin} m from the engine hull surface`);
    // the recorded point, for the same round, can be a kilometre away
    assert.ok(axisDistance(l, L) <= Math.max(r.recorded_axis_distance_m, halfW * 2) + 1e-6);
    // Projectiles.surfaceImpact hands the crater to the track; the drawn
    // round then ends exactly on it ...
    const shift = Math.hypot(P.x * 1000 - tr.impact[0], P.y * 1000 - tr.impact[1], P.z * 1000 - tr.impact[2]);
    maxShift = Math.max(maxShift, shift);
    tr.impact = [P.x * 1000, P.y * 1000, P.z * 1000];
    timeline.sample(tr, tr.end, out);
    for (let k = 0; k < 3; k++) near(out.pos[k], tr.impact[k], 1e-6);
    // ... leaves the recorded track untouched before the bend ...
    const t0 = tr.end - TERMINAL_BEND_S;
    if (t0 > tr.samples[0].t) {
      timeline.sample(tr, t0, out); sampleTrack(tr, t0, raw);
      for (let k = 0; k < 3; k++) near(out.pos[k], raw.pos[k], 1e-9);
    }
    // ... and bends by a degree or two at most on the way in
    for (let t = Math.max(tr.samples[0].t, t0); t <= tr.end; t += 0.05) {
      timeline.sample(tr, t, out); sampleTrack(tr, t, raw);
      const dot = out.vel[0] * raw.vel[0] + out.vel[1] * raw.vel[1] + out.vel[2] * raw.vel[2];
      const bend = Math.acos(Math.min(1, dot / (Math.hypot(...out.vel) * Math.hypot(...raw.vel)))) * 180 / Math.PI;
      maxBendDeg = Math.max(maxBendDeg, bend);
    }
  }
  if (process.env.SCALE_TEST_REPORT) console.log(`# drawn hits: end moved up to ${maxShift.toFixed(0)} m, bend up to ${maxBendDeg.toFixed(2)} deg, at most ${maxSkin.toFixed(1)} m from the engine hull surface`);
  assert.ok(maxShift > 500, `the drawn end is up to ${maxShift} m from the recorded one`);
  assert.ok(maxBendDeg < 4, `largest bend ${maxBendDeg} deg`);
  assert.ok(maxSkin < 60, `largest distance to the engine hull surface ${maxSkin} m`);
});

test('on the cinematic scale the same craters sit on the oversize hull (k = 1)', () => {
  const loader = fixtureLoader();
  const types = FIX.ship_types;
  const real = stubFx(loader, (id) => trueK(types[id])).fx, cine = stubFx(loader, () => 1).fx;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), n = new THREE.Vector3();
  for (const r of FIX.rounds.filter(x => x.kind === 'hit')) {
    const m = real.surfaceFor(r.id, a, n);
    cine.surfaceFor(r.id, b, n);
    const st = shipStateAt(loader.simTrace, r.target, m.t);
    const c = new THREE.Vector3(...st.pos).multiplyScalar(0.001);
    // the same hull point, 57-80 times further from the centre
    const ratio = b.clone().sub(c).length() / a.clone().sub(c).length();
    near(ratio, 1 / trueK(types[r.target]), 1e-6 / trueK(types[r.target]));
  }
});

test('a miss whose recorded line crosses the hull is drawn clear of it; wide misses are untouched', () => {
  const loader = fixtureLoader();
  const timeline = new WeaponTimeline(loader);
  const out = makeSample(), raw = makeSample();
  let pushed = 0, untouched = 0;
  for (const r of FIX.rounds.filter(x => x.kind === 'miss')) {
    const tr = timeline.slugs.get(r.id);
    assert.ok(tr && !tr.impact, r.id);
    const L = REAL_LENGTH_M[FIX.ship_types[r.target]];
    const need = MISS_CLEAR_RADII * L * HULL_RADIUS_PER_LENGTH + MISS_CLEAR_M;
    const ship = timeline.ships.get(r.target);
    const pass = closestPass(tr, ship, tr.samples.at(-1).t - 2, tr.end);
    const recorded = passClearance(pass, L).separation;
    assert.ok(Math.abs(recorded - r.recorded_separation_m) < 1.5, `${r.id}: ${recorded} vs ${r.recorded_separation_m}`);
    if (recorded >= need) {
      // already clear of the real hull: no push (a pass inside MISS_CLEAR_MAX_M
      // is kept, with a zero push, for the auto scale's magnified hulls)
      if (recorded >= MISS_CLEAR_MAX_M) assert.equal(tr.pass, null, `${r.id} is nowhere near the hull`);
      else assert.deepEqual(tr.pass.d, [0, 0, 0]);
      timeline.sample(tr, pass.t, out); sampleTrack(tr, pass.t, raw);
      assert.deepEqual(out.pos, raw.pos);
      if (tr.pass) {
        // drawn three times life size (auto, at distance) the same round clears three times as far, capped
        tr.pass.m = 3;
        timeline.sample(tr, pass.t, out);
        const d3 = axisDistance(hullFrame(loader.simTrace, r.target, pass.t, out.pos), L);
        assert.ok(d3 >= Math.min(MISS_CLEAR_MAX_M, 3 * need) - 0.5 || d3 >= recorded - 0.5, `${r.id}: ${d3} m at 3x`);
        assert.ok(d3 >= recorded - 1e-6, 'a push never brings the round closer');
        tr.pass.m = 1;
      }
      untouched++;
      continue;
    }
    pushed++;
    assert.ok(tr.pass, `${r.id} passes ${recorded} m from the axis of a ${L} m hull`);
    near(tr.pass.t1, pass.t, 1e-9);
    // drawn: at least `need` from the hull axis all the way past the ship
    for (let t = pass.t - 0.05; t <= pass.t + 0.05; t += 0.002) {
      timeline.sample(tr, t, out);
      const d = axisDistance(hullFrame(loader.simTrace, r.target, t, out.pos), L);
      assert.ok(d >= need - 0.5, `${r.id} at ${t}: ${d} m from the axis, need ${need}`);
    }
    // the push is a few tens of metres, eased in over the last seconds: no visible bend
    assert.ok(Math.hypot(...tr.pass.d) <= need + 1e-9);
    timeline.sample(tr, tr.pass.t0, out); sampleTrack(tr, tr.pass.t0, raw);
    assert.deepEqual(out.pos, raw.pos);
    // held after the pass: the round flies on parallel to its recorded line
    timeline.sample(tr, pass.t + 2, out); sampleTrack(tr, pass.t + 2, raw);
    for (let k = 0; k < 3; k++) near(out.pos[k] - raw.pos[k], tr.pass.d[k], 1e-6);
    // a hull drawn 4x (auto scale, at distance) is cleared by 4x as much; 80x (cinematic) hits the cap
    assert.equal(passPush(tr.pass, 1), Math.hypot(...tr.pass.d));
    near(passPush(tr.pass, 4), Math.min(MISS_CLEAR_MAX_M, 4 * need) - recorded, 1e-9);
    near(passPush(tr.pass, 80), MISS_CLEAR_MAX_M - recorded, 1e-9);
    tr.pass.m = 4;
    timeline.sample(tr, pass.t, out);
    assert.ok(axisDistance(hullFrame(loader.simTrace, r.target, pass.t, out.pos), L) >= Math.min(MISS_CLEAR_MAX_M, 4 * need) - 0.5);
    tr.pass.m = 1;
  }
  assert.ok(pushed >= 3 && untouched >= 2, `${pushed} pushed, ${untouched} untouched`);
  // a line dead through the axis still gets a definite, seeded side
  const dead = { rel: [0, 0, 0], dir: [1, 0, 0], axis: [0, 0, 1] };
  const c0 = passClearance(dead, 100, 0.1), c1 = passClearance(dead, 100, 0.9);
  assert.equal(c0.separation, 0);
  near(Math.hypot(...c0.normal), 1, 1e-12);
  assert.deepEqual(c0.normal.map(v => -v), c1.normal);
  assert.equal(missClearance({ samples: [] }, { samples: [] }, 100, 0, 1), null);
});

// ------------------------------------------------- effects follow the hull

test('a blast on a hull is placed and sized at the hull\'s live draw scale; the reactor blast is real', () => {
  let k = 0.016;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.001, 1e6);
  camera.position.set(0, 0, 500);
  const ctx = { scene, camera, tryLight: () => null, freeLight() {}, now: () => 10, kOf: () => k, kAt: () => 0.5 };
  const anchor = new THREE.Vector3(100, 0, 0);
  // authored in kit units: 2 units off the hull centre, 1 unit across
  const onHull = new Blast(ctx, { position: new THREE.Vector3(102, 0, 0), anchor, shipId: 'a', scale: 1, spawnTime: 10, kind: 'impact',
    baseVel: new THREE.Vector3(1, 0, 0) });
  onHull.update(10, 0);
  near(onHull.group.position.x, 100 + 2 * 0.016, 1e-12);
  near(onHull.group.scale.x, 0.016);
  // the camera backs off (auto mode): the SAME blast follows the hull's new scale, nothing is baked
  k = 0.4;
  onHull.update(11, 0);
  near(onHull.group.position.x, 100 + 2 * 0.4 + 1, 1e-12);   // + the hull's real velocity x 1 s
  near(onHull.group.scale.x, 0.4);
  // the reactor blast: real size whatever the hull's draw scale, centred on the drawn reactor
  const reactor = new Blast(ctx, { position: new THREE.Vector3(103, 0, 0), anchor, shipId: 'a', real: true, scale: 3, spawnTime: 10, kind: 'reactor' });
  reactor.update(10.5, 0);
  near(reactor.group.scale.x, 1);
  near(reactor.group.position.x, 100 + 3 * 0.4, 1e-12);
  // a free effect (no hull): the nominal scale at its own position
  const free = new Blast(ctx, { position: new THREE.Vector3(5, 5, 5), scale: 1, spawnTime: 10, kind: 'pop' });
  free.update(10.2, 0);
  near(free.group.scale.x, 0.5);
  near(free.group.position.x, 5);
  // without a scale provider (cinematic callers, old tests) everything is 1:1
  const plain = new Blast({ scene, camera, tryLight: () => null, freeLight() {}, now: () => 10 },
    { position: new THREE.Vector3(1, 2, 3), scale: 2, spawnTime: 10, kind: 'impact' });
  plain.update(10.1, 0);
  near(plain.group.scale.x, 1);
  near(plain.group.position.y, 2);
  for (const b of [onHull, reactor, free, plain]) b.dispose();
});

test('a wreck rides its real centre at the hull\'s draw scale; its reactor blast keeps the engine radius', () => {
  const g = new THREE.Group();
  const size = hullSize('destroyer');
  buildHull('destroyer', g, size, buildShipMaterials('alpha'), SceneManager.prototype.getEngineConfig.call(null, 'destroyer', size), 7);
  Object.assign(g.userData, { size, shipType: 'destroyer', k: trueK('destroyer') });
  if (!g.userData.reactorPos) g.userData.reactorPos = new THREE.Vector3(0, 0, -2);
  const blasts = [];
  const scene = new THREE.Scene();
  const ctx = { scene, camera: new THREE.PerspectiveCamera(), tryLight: () => null, freeLight() {}, now: () => 100,
    kOf: () => g.userData.k, spawnBlast: (o) => { blasts.push(o); return o; } };
  const eff = new ShipDestruction(ctx, {
    shipGroup: g, shipId: 'd', position: new THREE.Vector3(50, 0, 0), driftVel: new THREE.Vector3(2, 0, 0),
    driftDuration: 0, scale: 6, blastScale: 2.5, spawnTime: 100
  });
  eff.update(100, 1 / 60, 0);
  const reactor = blasts.find(b => b.kind === 'reactor');
  assert.equal(reactor.scale, 2.5);
  assert.equal(reactor.real, true);
  assert.equal(reactor.shipId, 'd');
  // chunks are children of one group on the wreck's real centre, scaled like the hull
  eff.update(103, 1 / 60, 0);
  near(eff.chunkGroup.position.x, 50 + 2 * 3, 1e-9);
  near(eff.chunkGroup.scale.x, trueK('destroyer'));
  assert.ok(eff.chunks.length > 0);
  for (const c of eff.chunks) assert.ok(c.mesh.position.length() < 40, 'chunk offsets are kit units, not world km');
  // the auto mode magnifies the distant wreck: same group, new scale
  g.userData.k = 0.3;
  eff.update(104, 1 / 60, 0);
  near(eff.chunkGroup.scale.x, 0.3);
  near(eff.position.x, 50 + 2 * 4 + 0, 0.3 * 2.5);   // the drawn reactor, within the magnified hull
  eff.dispose();
});

test('point lights fall off by the inverse square down to a metre (three\'s 100 m clamp is patched)', () => {
  // SceneManager lowers the clamp at import; if three renames the chunk text this fails loudly
  const chunk = THREE.ShaderChunk.lights_pars_begin;
  assert.ok(!chunk.includes('max( pow( lightDistance, decayExponent ), 0.01 )'), 'three\'s 0.1-unit falloff clamp is still in place');
  assert.ok(chunk.includes('max( pow( lightDistance, decayExponent ), 0.000001 )'));
});

test('baked buffers follow the draw scale in 2% steps: pure in k, and still in the true regime', () => {
  assert.equal(bakeScale(1), 1);                       // cinematic: exactly 1
  for (const type of HULL_CLASSES) {
    const k = trueK(type);
    assert.equal(bakeScale(k), bakeScale(k));           // a function of k alone
    assert.ok(Math.abs(bakeScale(k) / k - 1) <= (BAKE_STEP - 1) / 2 + 1e-12);
    // inside the true regime k does not move, so nothing is ever re-baked
    assert.equal(bakeScale(shipScale('auto', type, 0.3, FOV)), bakeScale(shipScale('auto', type, 1.2, FOV)));
  }
  // a slow dolly in the magnified regime re-bakes only when a 2% step is crossed
  let steps = 0, prev = null;
  for (let d = 40; d <= 44; d += 0.01) {
    const b = bakeScale(shipScale('auto', 'frigate', d, FOV));
    if (b !== prev) steps++;
    prev = b;
  }
  assert.ok(steps <= 6, `${steps} re-bakes over a 10% dolly in 400 frames`);
});

test('constants tuned on the kit-size hulls follow the scale unit', () => {
  // close-up ramp: 40..220 km cinematic, 0.64..3.5 km with real-size hulls
  setCloseRangeUnit(1);
  assert.equal(closeUpK(30), 1); assert.equal(closeUpK(250), 0);
  setCloseRangeUnit(FREE_TRUE_K);
  assert.equal(closeUpK(0.5), 1); assert.equal(closeUpK(4), 0);
  assert.ok(closeUpK(1.5) > 0 && closeUpK(1.5) < 1);
  setCloseRangeUnit(1);
  // the shadow box can close in on a real corvette (radius 39 m): at least
  // one texel per 10 cm at 2048, where the kit-scale floor gave 4 m texels
  assert.equal(quantiseShadowRadius(0.039, 1), SHADOW_MIN_RADIUS);
  const R = quantiseShadowRadius(0.039 * 1.15, FREE_TRUE_K);
  assert.ok(R >= SHADOW_MIN_RADIUS * FREE_TRUE_K && R < 0.1, `${R}`);
  const basis = shadowBasis(new THREE.Vector3(0.4, 0.5, 0.76));
  const target = new THREE.Vector3(), pos = new THREE.Vector3();
  const fit = fitSunShadow(new THREE.Vector3(120, 3, -40), basis, R, 2048, target, pos, FREE_TRUE_K);
  assert.ok(fit.texel < 1e-4);
  assert.ok(fit.near < 0.01 && fit.far > 2 * R, 'near plane off the hull, the whole box in range');
  assert.ok(fit.normalBias > 0 && fit.bias < 0);
  // the cinematic fit is what it was
  const old = fitSunShadow(new THREE.Vector3(120, 3, -40), basis, 8, 2048, target, pos);
  near(old.near, 0.5); near(old.far, 2 * (8 + 60));
});

// ------------------------------------------------------ director at true scale

function stage(positions, types, { deaths = {}, mode = 'auto' } = {}) {
  const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.0005, 800000);
  const controls = { target: new THREE.Vector3(), enabled: true, update() { camera.lookAt(this.target); }, addEventListener() {} };
  const kinds = {};
  const scene = {
    SCALE: 1 / 1000, currentTime: 0, delta: 1 / 60, destructionEffects: [], deathInfo: deaths, scaleMode: mode,
    ships: new Map(Object.keys(positions).map(id => [id, { userData: { shipType: types[id], size: hullSize(types[id]), faction: id.split('_')[0] } }])),
    getShipPosition: (id) => new THREE.Vector3(positions[id][0], positions[id][1], positions[id][2]).multiplyScalar(0.001),
    framingLength: (id) => framingLength(mode, types[id]),
    // SceneManager.deathBlastExtent with real-size hulls: the fireball fills 0.55 of the blast radius
    deathBlastExtent: (id, a, b) => blastExtentOver('reactor', 0.55 * (kinds[id] || 15) / blastExtentOver('reactor', 1, 2, 6), a, b)
  };
  camera.position.set(0, 300, 400);
  const controller = new CameraController(camera, controls, scene);
  const states = t => Object.fromEntries(Object.entries(positions).map(([id, p]) => [id, {
    position: [p[0] + (p[3] || 0) * t, p[1], p[2]], forward: [id.startsWith('alpha') ? 1 : -1, 0, 0],
    velocity: [p[3] || 0, 0, 0], thrust: 0.5, destroyed: deaths[id]?.time <= t
  }]));
  return { camera, controls, scene, controller, states, kinds };
}

test('director at real scale: close shots are a few real hull lengths out, inside the true regime', () => {
  const types = { alpha_1: 'cruiser', beta_1: 'battleship', beta_2: 'corvette' };
  const s = stage({ alpha_1: [-90000, 0, 0, 800], beta_1: [90000, 0, 0, -800], beta_2: [95000, 6000, 0, -800] }, types);
  const world = s.controller._world(s.states(50));
  assert.equal(world.realScale, true);
  assert.equal(world.unit, FREE_TRUE_K);
  near(world.length('beta_1'), 0.2); near(world.length('beta_2'), 0.065);
  for (const type of ['shoulder', 'chase', 'flyby', 'orbit', 'runin']) {
    for (const subject of ['beta_1', 'beta_2']) {
      const shot = { type, kind: 'salvo', subject, t0: 48, t1: 56, t: 54, impacts: [], target: 'alpha_1', attacker: 'alpha_1' };
      const pose = shotPose(shot, 50, world);
      assert.ok(pose, `${type} ${subject}`);
      const L = world.length(subject);
      const d = pose.position.distanceTo(world.pos(subject));
      assert.ok(d > 0.7 * L && d < 6 * L, `${type} on ${subject}: camera ${d / L} hull lengths out`);
      assert.ok(d < trueRegimeDistance(L * 1000, world.fov), `${type} on ${subject}: the hull is at real size`);
      // and the subject is at real size from there
      assert.equal(shipScale('auto', types[subject], d, world.fov), trueK(types[subject]));
    }
  }
  // the same shot on the cinematic scale is the same number of (kit) hull lengths out
  const c = stage({ alpha_1: [-90000, 0, 0, 800], beta_1: [90000, 0, 0, -800], beta_2: [95000, 6000, 0, -800] }, types, { mode: 'cinematic' });
  const cw = c.controller._world(c.states(50));
  assert.equal(cw.realScale, false); assert.equal(cw.unit, 1);
  const shot = { type: 'chase', kind: 'salvo', subject: 'beta_1', t0: 48, t1: 56, t: 54 };
  const a = shotPose(shot, 50, world), b = shotPose(shot, 50, cw);
  near(a.position.distanceTo(world.pos('beta_1')) / world.length('beta_1'),
    b.position.distanceTo(cw.pos('beta_1')) / cw.length('beta_1'), 1e-9);
});

test('director at real scale: a kill shot starts on the hull and pulls back to contain the real blast radius', () => {
  const types = { alpha_1: 'cruiser', beta_1: 'battleship' };
  const R = 18.37;   // km: the battleship's recorded reactor_blast radius
  const events = [
    { timestamp: 100, event_type: 'ship_destroyed', ship_id: 'beta_1', data: { killer_id: 'alpha_1' } },
    { timestamp: 100, event_type: 'reactor_blast', ship_id: 'beta_1', data: { position: [90000, 0, 0], radius_km: R, yield_gj: 850 } }
  ];
  const s = stage({ alpha_1: [-9000, 0, 0, 0], beta_1: [90000, 0, 0, 0] }, types);
  s.kinds.beta_1 = R;
  const shots = buildShotList(events, {});
  const pull = shots.find(x => x.type === 'pullback');
  assert.ok(pull, 'a pull-back on the kill');
  const centre = new THREE.Vector3(90, 0, 0);
  const world = () => s.controller._world(s.states(0));
  const L = 0.2;
  const dist = (t) => shotPose(pull, t, world()).position.distanceTo(centre);
  // before it blows: on the hull, at real size
  const d0 = dist(100 - PULLBACK_PRE_S);
  assert.ok(d0 < 3 * L, `starts ${d0 / L} hull lengths out`);
  assert.equal(shipScale('auto', 'battleship', d0, 55), trueK('battleship'));
  // moving out monotonically while the ring is bright (it eases back in on
  // the fireball afterwards); never inside the expanding front
  let prev = 0, maxD = 0;
  for (let t = 100 - PULLBACK_PRE_S; t <= 100 + 2.0; t += 0.05) {
    const d = dist(t);
    if (t <= 100 + 1.3) assert.ok(d >= prev - 1e-9, `camera moved in at ${t}`);
    if (t >= 100) assert.ok(d > blastFront(t - 100, R), `inside the blast front at ${t}: ${d} km`);
    prev = d; maxD = Math.max(maxD, d);
  }
  // once out, the WHOLE lethal radius is in frame for the rest of the shot (it eases in a little, never past it)
  for (let t = 100 + 1.3; t <= 100 + PULLBACK_POST_S; t += 0.1) {
    assert.ok(dist(t) >= fitDistance(R, 55, 16 / 9, 0.62, 1), `blast radius cut off at ${t}: ${dist(t)} km`);
  }
  assert.ok(maxD < 400, `and not absurdly far: ${maxD} km`);
  // the fireball the scene draws is inside that radius
  assert.ok(s.scene.deathBlastExtent('beta_1', 2, 6) <= 0.56 * R);
  // the cinematic pull-back never came closer than the shell: unchanged
  const c = stage({ alpha_1: [-9000, 0, 0, 0], beta_1: [90000, 0, 0, 0] }, types, { mode: 'cinematic' });
  const cw = c.controller._world(c.states(0));
  const cd = shotPose(pull, 100 - PULLBACK_PRE_S, cw).position.distanceTo(centre);
  assert.ok(cd >= 1.3 * R && cd >= 2.4 * kitLength('battleship') - 1e-6, `cinematic start ${cd} km`);
});

test('fleet framing floors follow the scale: a two-ship duel 4 km apart is framed, not lost in a 25 km box', () => {
  const types = { alpha_1: 'cruiser', beta_1: 'battleship' };
  const s = stage({ alpha_1: [0, 0, 0, 0], beta_1: [4000, 0, 0, 0] }, types);
  const frame = s.controller.frameFleets(s.states(0), []);
  assert.ok(frame.radius >= 2 && frame.radius < 3, `radius ${frame.radius} km`);
  const c = stage({ alpha_1: [0, 0, 0, 0], beta_1: [4000, 0, 0, 0] }, types, { mode: 'cinematic' });
  assert.equal(c.controller.frameFleets(c.states(0), []).radius, 25);
});

// ------------------------------------------------------------ old recordings

test('old recordings load: no roll, no turret poses, no reactor_blast, unknown hull types', () => {
  const rec = JSON.parse(readFileSync(new URL('fixtures/mini_recording.json', import.meta.url), 'utf8'));
  const loader = new BattleLoader();
  loader.parse(rec);
  const timeline = new WeaponTimeline(loader);
  assert.ok(timeline.ships.size >= 2);
  for (const tr of [...timeline.slugs.values(), ...timeline.torpedoes.values()]) {
    assert.ok(tr.pass === undefined || tr.pass === null || Number.isFinite(tr.pass.t1));
    const out = makeSample();
    timeline.sample(tr, Math.min(tr.end, tr.samples.at(-1).t + 0.5), out);
    assert.ok(out.pos.every(Number.isFinite) && out.vel.every(Number.isFinite), tr.id);
  }
  // a pre-pass-5 trace: positions only
  const bare = new BattleLoader();
  bare.simTrace = [
    { t: 0, ships: { a_1: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0] }, b_1: { pos: [9000, 0, 0], vel: [0, 0, 0], fwd: [-1, 0, 0] } },
      projectiles: [{ id: 'p', pos: [1000, 0, 0], vel: [8000, 0, 0], source: 'a_1', target: 'b_1' }] },
    { t: 1, ships: { a_1: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0] }, b_1: { pos: [9000, 0, 0], vel: [0, 0, 0], fwd: [-1, 0, 0] } } },
    { t: 9, ships: { a_1: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0] }, b_1: { pos: [9000, 0, 0], vel: [0, 0, 0], fwd: [-1, 0, 0] } } }
  ];
  bare.events = [];
  bare.ships = { a_1: { type: 'gunboat' }, b_1: {} };   // types this viewer never heard of
  bare.buildProjectileHitMap(); bare.buildLastProjectileStates(); bare.buildTorpedoTimeline(); bare.buildPDBeamSegments();
  const tl = new WeaponTimeline(bare);
  const p = tl.slugs.get('p');
  // the round runs dead down the target's axis and was not recorded as a hit: drawn past it, clear
  assert.ok(p.pass && Math.hypot(...p.pass.d) > 20, 'pushed clear of a destroyer-size hull');
  const out = makeSample();
  tl.sample(p, 1.0, out);
  assert.ok(Math.hypot(out.pos[1], out.pos[2]) > 20);
  // blast radius without a reactor_blast event: the engine's rule per class
  const sm = { damageFX: { timeline: { blasts: [] } }, scaleMode: 'auto', deathInfo: {} };
  sm.blastRadiusKm = SceneManager.prototype.blastRadiusKm;
  sm.destructionScale = SceneManager.prototype.destructionScale;
  assert.equal(SceneManager.prototype.blastRadiusKm.call(sm, 'x', 'cruiser'), 15);
  assert.ok(SceneManager.prototype.blastRadiusKm.call(sm, 'x', 'mystery') > 0);
  const real = SceneManager.prototype.reactorBlastScale.call(sm, 'x', 'cruiser');
  near(blastExtentOver('reactor', real, 2, 6), 0.55 * 15, 1e-9);
  sm.scaleMode = 'cinematic';
  assert.equal(SceneManager.prototype.reactorBlastScale.call(sm, 'x', 'cruiser'), sm.destructionScale('cruiser'));
});

// ------------------------------------------------------------------ drive jet

test('the far drive jet has the engine\'s geometry and fades like its heat rate', async () => {
  const { DriveJets, jetBrightness, JET_CORE_DEG, JET_MAX_DEG, JET_REACH_KM } = await import('../src/render/DriveJet.js');
  // src/drive_plume.py is the source of the angles and the reach
  const py = readFileSync(new URL('../../src/drive_plume.py', import.meta.url), 'utf8');
  const constant = (name) => Number(new RegExp(`^${name} = ([0-9.]+)`, 'm').exec(py)[1]);
  assert.equal(JET_CORE_DEG, constant('PLUME_CORE_HALF_ANGLE_DEG'));
  assert.equal(JET_MAX_DEG, constant('PLUME_MAX_HALF_ANGLE_DEG'));
  assert.ok(JET_REACH_KM <= constant('PLUME_MAX_RANGE_KM'));
  // CLAUDE.md: on axis at full burn 40 GJ/s at 5 km, 16 at 10 km, 4.7 at 20 km, 0.8 at 50 km.
  // The drawn brightness is the square root of that, relative to the 5 km near field.
  for (const [km, rate] of [[10, 16], [20, 4.7], [50, 0.8]]) {
    const want = Math.sqrt(rate / 40);
    assert.ok(Math.abs(jetBrightness(km) - want) < 0.06, `${km} km: ${jetBrightness(km)} vs ${want}`);
  }
  assert.ok(jetBrightness(5) > 0.8 && jetBrightness(0) === 1);
  for (let z = 0; z < 100; z += 1) assert.ok(jetBrightness(z + 1) < jetBrightness(z), 'monotonic');
  // one instance per burning, living hull; none on the cinematic scale
  const scene = new THREE.Scene();
  const jets = new DriveJets(scene);
  const ship = (thrust, extra = {}) => {
    const g = new THREE.Group();
    g.position.set(100, 5, -3);
    g.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.7);
    g.userData = { smoothThrust: thrust, k: 0.0125, plumeLength: 72, drawn: { length: 0.2, width: 0.055 }, ...extra };
    return g;
  };
  const ships = new Map([['a', ship(1)], ['b', ship(0)], ['c', ship(0.6, { dyingNow: true })], ['d', ship(0.5)]]);
  ships.get('d').visible = false;
  jets.update(ships, true);
  assert.equal(jets.mesh.count, 1);
  assert.equal(jets.mesh.visible, true);
  // it leaves the stern of the hull as drawn, pointing aft, at real size (no scale on the instance)
  const m = new THREE.Matrix4(); jets.mesh.getMatrixAt(0, m);
  const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  m.decompose(p, q, s);
  const a = ships.get('a');
  const stern = a.position.clone().addScaledVector(new THREE.Vector3(0, 0, 1).applyQuaternion(a.quaternion), -0.1);
  assert.ok(p.distanceTo(stern) < 1e-4, `${p.distanceTo(stern)} km from the stern`);   // (float32 instance matrix)
  near(s.x, 1, 1e-6); near(s.z, 1, 1e-6);
  assert.equal(jets.jetAttr.array[2], JET_REACH_KM);
  jets.update(ships, false);
  assert.equal(jets.mesh.count, 0);
  assert.equal(jets.mesh.visible, false);
  jets.dispose();
});
