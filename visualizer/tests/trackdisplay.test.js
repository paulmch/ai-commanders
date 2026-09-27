import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TrajectoryPredictor, deriveCheckpointSchedule, isCheckpointTime
} from '../src/TrajectoryPredictor.js';
import { Interpolator } from '../src/Interpolator.js';
import {
  hullExaggeration, magnifyRadius, magnifyOffset, formatSigma, formatCheckpointLabel,
  isReactiveMode, nextTrackScope, loadTrackScope, saveTrackScope, inTrackScope, TRACK_SCOPE_KEY
} from '../src/TrackDisplay.js';
import { TrackHud, formatExaggeration } from '../src/TrackHud.js';
import { TrackModeControl } from '../src/TrackModeControl.js';
import { trailLifeCap, focusFade, blastCloseDim, closeUpK } from '../src/render/CloseRange.js';
import { puffScale } from '../src/render/Rcs.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b}`);
const closeVec = (a, b, tol, msg) => { for (let k = 0; k < 3; k++) close(a[k], b[k], tol, `${msg || ''}[${k}]`); };

function cruise(p, v, t) {
  return { pos: [p[0] + v[0] * t, p[1] + v[1] * t, p[2] + v[2] * t], vel: [...v], fwd: [1, 0, 0], thrust: 0 };
}

function linePlan(pos, v, t, tCp, dt = 2, sigma = 0) {
  const path = [];
  for (let k = 0; t + k * dt < tCp - 1e-9; k++) path.push(pos.map((x, i) => x + v[i] * k * dt));
  path.push(pos.map((x, i) => x + v[i] * (tCp - t)));
  return { t_cp: tCp, dt, path, sigma };
}

function interpolated(loader, t) {
  const tr = loader.simTrace;
  let i = 0;
  while (i < tr.length - 1 && tr[i + 1].t <= t) i++;
  const f0 = tr[i], f1 = tr[Math.min(i + 1, tr.length - 1)];
  return Interpolator.interpolateShips(f0, f1, f1 === f0 ? 0 : (t - f0.t) / (f1.t - f0.t));
}

test('a frame exactly on a checkpoint is recognised from the header or the events', () => {
  const hdr = deriveCheckpointSchedule({ decision_interval_s: 30 }, []);
  assert.ok(isCheckpointTime(hdr, 30));
  assert.ok(isCheckpointTime(hdr, 120.0004));
  assert.ok(!isCheckpointTime(hdr, 121));
  assert.ok(!isCheckpointTime(hdr, 0), 'battle start is not a re-plan');
  const ev = deriveCheckpointSchedule({}, [45, 90].map(t => ({ event_type: 'checkpoint', timestamp: t })));
  assert.ok(isCheckpointTime(ev, 45) && isCheckpointTime(ev, 90));
  assert.ok(isCheckpointTime(ev, 135), 'extrapolated past the last recorded checkpoint');
  assert.ok(!isCheckpointTime(ev, 60));
});

test('on a checkpoint frame the next frame\'s plan wins (w0 = 0): no slide across the re-plan', () => {
  const v = [100, 0, 0];
  const f = t => ({ t, ships: { a: cruise([0, 0, 0], v, t) } });
  const f119 = f(119), f120 = f(120), f121 = f(121);
  f119.ships.a.plan = linePlan(f119.ships.a.pos, v, 119, 120, 1);
  // The checkpoint frame still flies the old orders to t=150 ...
  f120.ships.a.plan = linePlan(f120.ships.a.pos, v, 120, 150);
  // ... the frame after it already has the new ones: a 5 km sidestep
  f121.ships.a.plan = linePlan(f121.ships.a.pos, v, 121, 150);
  f121.ships.a.plan.path.at(-1)[1] += 5000;
  const loader = { simTrace: [f119, f120, f121], events: [], sourceData: { decision_interval_s: 30 } };
  const pred = new TrajectoryPredictor(loader, { samples: 16 });
  const at = t => [...pred.predict(t, interpolated(loader, t)).get('a').cp];
  closeVec(at(120), [15000, 5000, 0], 1e-6, 'already at t = 120');
  closeVec(at(120.5), [15000, 5000, 0], 1e-6, 'halfway through the second: still the new plan');
  // Before the checkpoint the old interval is untouched
  closeVec(at(119.5), [12000, 0, 0], 1e-6);
  // The live head (no f1) keeps the checkpoint frame's own plan
  const head = { simTrace: [f119, f120], events: [], sourceData: { decision_interval_s: 30 } };
  const hp = new TrajectoryPredictor(head, { samples: 8 });
  closeVec(hp.predict(120, interpolated(head, 120)).get('a').cp, [15000, 0, 0], 1e-6);
});

test('stale, mode and predicted nose come from the plans; old recordings fall back', () => {
  const v = [10, 0, 0];
  const f = t => ({ t, ships: { a: cruise([0, 0, 0], v, t) } });
  const f0 = f(10), f1 = f(11);
  f0.ships.a.plan = { ...linePlan(f0.ships.a.pos, v, 10, 30), stale: true, mode: 'run', fwd: [0, 1, 0] };
  f1.ships.a.plan = { ...linePlan(f1.ships.a.pos, v, 11, 30), mode: null, fwd: [0, 0, 1] };
  const loader = { simTrace: [f0, f1], events: [] };
  const pred = new TrajectoryPredictor(loader, { samples: 8 });
  let tr = pred.predict(10.25, interpolated(loader, 10.25)).get('a');
  assert.equal(tr.stale, true, 'a contributing stale plan marks the track provisional');
  assert.equal(tr.mode, 'run', 'mode of the dominant plan');
  assert.equal(tr.hasFwd, true);
  close(Math.hypot(...tr.cpFwd), 1, 1e-9, 'blended nose is a unit vector');
  assert.ok(tr.cpFwd[1] > tr.cpFwd[2]);
  tr = pred.predict(11, interpolated(loader, 11)).get('a');
  assert.equal(tr.stale, false);
  assert.equal(tr.mode, null, 'explicit null mode is not overridden');
  closeVec(tr.cpFwd, [0, 0, 1], 1e-9);

  // no plan.mode key: the frame's own evade mode; no plan.fwd: none
  const g0 = f(10), g1 = f(11);
  g0.ships.a.plan = linePlan(g0.ships.a.pos, v, 10, 30);
  g1.ships.a.plan = linePlan(g1.ships.a.pos, v, 11, 30);
  g0.ships.a.evade = g1.ships.a.evade = 'present';
  const old = new TrajectoryPredictor({ simTrace: [g0, g1], events: [] }, { samples: 8 });
  tr = old.predict(10.5, interpolated({ simTrace: [g0, g1] }, 10.5)).get('a');
  assert.equal(tr.mode, 'present');
  assert.equal(tr.hasFwd, false);
  assert.equal(tr.stale, false);
});

test('the drift locus lists this interval\'s earlier predictions and restarts at a checkpoint', () => {
  const v = [100, 0, 0];
  const trace = [];
  for (let t = 29; t <= 36; t++) {
    const fr = { t, ships: { a: cruise([0, 0, 0], v, t) } };
    fr.ships.a.plan = linePlan(fr.ships.a.pos, v, t, t < 30 ? 30 : 60);
    if (t > 30) fr.ships.a.plan.path.at(-1)[1] += 10 * (t - 30);   // the prediction wanders
    trace.push(fr);
  }
  const loader = { simTrace: trace, events: [], sourceData: { decision_interval_s: 30 } };
  const pred = new TrajectoryPredictor(loader, { samples: 8 });
  let tr = pred.predict(34.5, interpolated(loader, 34.5)).get('a');
  const L = tr.locus;
  assert.equal(L.n, 5, 'current + frames 34, 33, 32, 31 (the checkpoint frame 30 is left out)');
  closeVec(L.pts.subarray(0, 3), tr.cp, 1e-9, 'index 0 is the current prediction');
  assert.deepEqual([...L.times.subarray(1, 5)], [34, 33, 32, 31]);
  closeVec(L.pts.subarray(3, 6), [6000, 40, 0], 1e-9);
  closeVec(L.pts.subarray(12, 15), [6000, 10, 0], 1e-9);
  tr = pred.predict(30, interpolated(loader, 30)).get('a');
  assert.equal(tr.locus.n, 1, 'right on the checkpoint nothing earlier belongs to the new interval');

  // plan-less recordings: the frames propagated at constant acceleration
  const bare = [];
  for (let t = 31; t <= 34; t++) bare.push({ t, ships: { b: { pos: [t * 50, 0, 0], vel: [50, 0, 0], acc: [0, 1, 0], fwd: [1, 0, 0] } } });
  const kin = new TrajectoryPredictor({ simTrace: bare, events: [], sourceData: { decision_interval_s: 30 } }, { samples: 8 });
  const kt = kin.predict(33.5, interpolated({ simTrace: bare }, 33.5)).get('b');
  assert.equal(kt.source, 'kinematic');
  assert.equal(kt.locus.n, 4);
  closeVec(kt.locus.pts.subarray(3, 6), [33 * 50 + 50 * 27, 0.5 * 27 * 27, 0], 1e-9);
});

test('rings and drift are magnified like the hulls, soft-capped, never below true size', () => {
  close(hullExaggeration('corvette', 4.0), 4000 / 65, 1e-9);
  close(hullExaggeration('dreadnought', 20.8), 20800 / 275, 1e-9);
  assert.equal(hullExaggeration('mystery', 5), 60);
  const k = 60, rMax = 16000;
  close(magnifyRadius(10, k, rMax), 600, 1, 'linear while small');
  assert.equal(magnifyRadius(0, k, rMax), 0);
  let prev = 0;
  for (let r = 1; r < 5e5; r *= 1.7) {
    const m = magnifyRadius(r, k, rMax);
    assert.ok(m >= prev, 'monotonic');
    assert.ok(m >= r, 'never smaller than the true offset');
    assert.ok(m <= Math.max(r, rMax) + 1e-6, 'capped at rMax unless already larger');
    prev = m;
  }
  const out = new Float64Array(6);
  magnifyOffset([3, 4, 0], k, rMax, out, 0, 3);
  close(Math.hypot(out[3], out[4], out[5]), magnifyRadius(5, k, rMax), 1e-9);
  close(out[3] / out[4], 0.75, 1e-12, 'direction kept');
  magnifyOffset([0, 0, 0], k, rMax, out);
  assert.deepEqual([...out.subarray(0, 3)], [0, 0, 0]);
});

test('captions state the true spread; scope helpers persist', () => {
  assert.equal(formatSigma(0), '');
  assert.equal(formatSigma(7.4), '±7 m');
  assert.equal(formatSigma(47), '±47 m');
  assert.equal(formatSigma(183), '±180 m');
  assert.equal(formatSigma(1234), '±1.2 km');
  assert.equal(formatSigma(12345), '±12 km');
  assert.equal(formatCheckpointLabel({ togo: 24.2, sigma: 183 }), 'T+25s ±180 m');
  assert.equal(formatCheckpointLabel({ togo: 24.2, sigma: 0 }), 'T+25s');
  assert.equal(formatCheckpointLabel({ togo: 3, sigma: 1234, mode: 'run' }), 'T+3s ±1.2 km · RUN');
  assert.equal(formatCheckpointLabel({ togo: 3, sigma: 50, mode: 'wobble' }), 'T+3s ±50 m', 'wobble is modelled');
  assert.equal(formatCheckpointLabel({ togo: 3, stale: true, mode: 'run' }), 'T+3s · ORDERS PENDING');
  assert.equal(formatCheckpointLabel({ togo: -1 }), 'T+0s');
  assert.equal(formatCheckpointLabel({ togo: 24.2, sigma: 183, maneuver: 'HEADING', throttle: 0.6 }), 'T+25s ±180 m · HEADING 60%');
  assert.equal(formatCheckpointLabel({ togo: 5, maneuver: 'HEADING' }), 'T+5s · HEADING');
  assert.equal(formatCheckpointLabel({ togo: 5, maneuver: 'HEADING', throttle: 1, mode: 'run' }), 'T+5s · RUN', 'evasion wins');
  assert.equal(formatCheckpointLabel({ togo: 5, maneuver: 'INTERCEPT', throttle: 1 }), 'T+5s');
  assert.ok(isReactiveMode('PRESENT') && !isReactiveMode('wobble') && !isReactiveMode(null));
  assert.equal(formatExaggeration([61.5, 61.5]), '×62');
  assert.equal(formatExaggeration([57.6, 80]), '×58–80');

  assert.equal(nextTrackScope('all'), 'focus');
  assert.equal(nextTrackScope('focus'), 'all');
  assert.ok(inTrackScope('all', 'a', null));
  assert.ok(inTrackScope('focus', 'a', 'a'));
  assert.ok(!inTrackScope('focus', 'b', 'a'));
  assert.ok(!inTrackScope('focus', 'a', null), 'nothing selected: nothing drawn');
  const store = new Map();
  const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  assert.equal(loadTrackScope(storage), 'all');
  saveTrackScope(storage, 'focus');
  assert.equal(store.get(TRACK_SCOPE_KEY), 'focus');
  assert.equal(loadTrackScope({ getItem() { throw new Error('denied'); } }), 'all');
});

test('Shift+T switches the track scope, plain T still cycles the mode', () => {
  const store = new Map();
  const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  const keys = new EventTarget();
  const scopes = [];
  const hud = new TrackHud({ storage, keyTarget: keys, onScope: s => scopes.push(s) });
  const ctl = new TrackModeControl({ storage, keyTarget: keys });
  const press = (shiftKey) => { const e = new Event('keydown'); e.code = 'KeyT'; e.shiftKey = shiftKey; keys.dispatchEvent(e); };
  const mode0 = ctl.mode;
  press(true);
  assert.equal(hud.scope, 'focus');
  assert.equal(ctl.mode, mode0, 'Shift+T leaves the mode alone');
  press(false);
  assert.equal(hud.scope, 'focus', 'plain T leaves the scope alone');
  assert.notEqual(ctl.mode, mode0);
  press(true);
  assert.deepEqual(scopes, ['focus', 'all']);
  assert.equal(new TrackHud({ storage, keyTarget: new EventTarget() }).scope, 'all', 'persisted');
});

test('close-range helpers leave fleet range alone and tame close-ups', () => {
  assert.equal(closeUpK(500), 0);
  assert.equal(closeUpK(20), 1);
  // A 1 s, 20 km/s slug wake: 46 px at fleet range is untouched ...
  const pw = 0.00145;
  assert.equal(trailLifeCap(1, 20, 300 * pw, 140), 1);
  // ... in a 15-unit close-up it is capped to 140 px of screen
  const life = trailLifeCap(1, 20, 15 * pw, 140);
  close((20 * life) / (15 * pw), 140, 1e-6);
  assert.equal(trailLifeCap(1, 0, 1, 140), 1);
  // distance from the focus only matters up close
  assert.equal(focusFade(5000, 400), 1);
  assert.equal(focusFade(5, 15), 1, 'wakes at the focused ship stay');
  assert.ok(focusFade(200, 15) < 0.15, 'distant wakes fade in a close-up');
  // blast sparks: unchanged far away, dimmed a few blast radii out
  assert.equal(blastCloseDim(400, 15), 1);
  assert.equal(blastCloseDim(300, 3), 1);
  assert.ok(blastCloseDim(10, 3) < 0.7);
  assert.ok(blastCloseDim(5, 3) >= 0.4);
});

test('RCS puffs scale sub-linearly with the jet', () => {
  const corvette = 0.36 * 1.04 + 0.06 * 4.0, dread = 0.36 * 5.6 + 0.06 * 20.8;
  close(puffScale(0.6), 0.6, 1e-12);
  const ratio = puffScale(dread) / puffScale(corvette);
  assert.ok(ratio > 1.5 && ratio < dread / corvette * 0.6, `sub-linear (${ratio})`);
});
