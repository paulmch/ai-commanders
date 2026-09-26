import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TrajectoryPredictor, deriveCheckpointSchedule, nextCheckpointTime, hermiteInto, HULL_ACCEL_G
} from '../src/TrajectoryPredictor.js';
import { Interpolator } from '../src/Interpolator.js';
import { nextTrackMode, loadTrackMode, saveTrackMode, TrackModeControl, TRACK_MODE_KEY } from '../src/TrackModeControl.js';

const G = 9.80665;
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b}`);
const closeVec = (a, b, tol, msg) => { for (let k = 0; k < 3; k++) close(a[k], b[k], tol, `${msg || ''}[${k}]`); };

/** Ship moving at constant velocity v from p (1 Hz frames). */
function cruise(p, v, t) {
  return { pos: [p[0] + v[0] * t, p[1] + v[1] * t, p[2] + v[2] * t], vel: [...v], fwd: [1, 0, 0], thrust: 0 };
}

/** Straight-line plan from pos with velocity v, sampled every dt to t_cp. */
function linePlan(pos, v, t, tCp, dt = 2, sigma = 0) {
  const path = [];
  for (let k = 0; t + k * dt < tCp - 1e-9; k++) path.push(pos.map((x, i) => x + v[i] * k * dt));
  path.push(pos.map((x, i) => x + v[i] * (tCp - t)));
  return { t_cp: tCp, dt, path, sigma };
}

function interpolated(loader, t) {
  const fd = { frame0: null, frame1: null, alpha: 0 };
  const tr = loader.simTrace;
  let i = 0;
  while (i < tr.length - 1 && tr[i + 1].t <= t) i++;
  fd.frame0 = tr[i];
  fd.frame1 = tr[Math.min(i + 1, tr.length - 1)];
  fd.alpha = fd.frame1 === fd.frame0 ? 0 : (t - fd.frame0.t) / (fd.frame1.t - fd.frame0.t);
  return Interpolator.interpolateShips(fd.frame0, fd.frame1, fd.alpha);
}

test('checkpoint times come from the header, else checkpoint events, else 30 s', () => {
  const header = deriveCheckpointSchedule({ decision_interval_s: 20 }, []);
  assert.equal(nextCheckpointTime(header, 0), 20);
  assert.equal(nextCheckpointTime(header, 19.5), 20);
  assert.equal(nextCheckpointTime(header, 20), 40, 'a checkpoint at t has just passed');

  const events = [45, 45, 90, 135].map(t => ({ event_type: 'captain_decision', timestamp: t }))
    .concat([{ event_type: 'hit', timestamp: 50 }]);
  const ev = deriveCheckpointSchedule({}, events);
  assert.equal(ev.source, 'events');
  assert.equal(nextCheckpointTime(ev, 10), 45);
  assert.equal(nextCheckpointTime(ev, 50), 90);
  assert.equal(nextCheckpointTime(ev, 135), 180, 'extrapolated with the median spacing');
  assert.equal(nextCheckpointTime(ev, 200), 225);

  const oneV1 = deriveCheckpointSchedule({}, [{ event_type: 'checkpoint', timestamp: 30 }]);
  assert.equal(nextCheckpointTime(oneV1, 31), 60);

  const none = deriveCheckpointSchedule({}, [{ event_type: 'hit', timestamp: 12 }]);
  assert.equal(none.source, 'default');
  assert.equal(nextCheckpointTime(none, 31), 60);
});

test('recorded plans of the bracketing frames blend smoothly and anchor on the hull', () => {
  const v = [100, 0, 0];
  const f = t => ({ t, ships: { a: { ...cruise([0, 0, 0], v, t) } } });
  const f10 = f(10), f11 = f(11);
  // Frame 10 plans a straight cruise; frame 11 plans a 300 m sidestep at the checkpoint.
  f10.ships.a.plan = linePlan(f10.ships.a.pos, v, 10, 30, 2, 40);
  f11.ships.a.plan = linePlan(f11.ships.a.pos, v, 11, 30, 2, 20);
  f11.ships.a.plan.path.at(-1)[1] += 300;
  const loader = { simTrace: [f10, f11], events: [] };
  const pred = new TrajectoryPredictor(loader, { samples: 16 });

  let tr = pred.predict(10, interpolated(loader, 10)).get('a');
  assert.equal(tr.source, 'plan');
  assert.equal(tr.tCp, 30);
  closeVec(tr.cp, [3000, 0, 0], 1e-6, 'alpha 0 = frame 10 plan');
  close(tr.sigma, 40, 1e-9);

  const ships = interpolated(loader, 10.5);
  tr = pred.predict(10.5, ships).get('a');
  closeVec(tr.cp, [3000, 150, 0], 1e-6, 'halfway blend of the checkpoints');
  closeVec(tr.path.subarray(0, 3), ships.a.position, 1e-6, 'track starts on the rendered hull');
  assert.ok(tr.sigma < 30 && tr.sigma > 29, `sigma blends (${tr.sigma})`);
  assert.equal(tr.times[0], 10.5);
  assert.equal(tr.times[tr.n - 1], 30);

  // A displaced rendered hull pulls the start, not the planned checkpoint
  ships.a.position = [ships.a.position[0], ships.a.position[1] + 50, 0];
  tr = pred.predict(10.5, ships).get('a');
  close(tr.path[1], ships.a.position[1], 1e-6);
  closeVec(tr.cp, [3000, 150, 0], 1e-6);

  // Continuity: the marker barely moves across a tiny time step
  const a = [...pred.predict(10.49, interpolated(loader, 10.49)).get('a').cp];
  const b = [...pred.predict(10.51, interpolated(loader, 10.51)).get('a').cp];
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 10);
});

test('across a checkpoint the track runs to the nearer checkpoint, then jumps to the next', () => {
  const v = [50, 0, 0];
  const f29 = { t: 29, ships: { a: cruise([0, 0, 0], v, 29) } };
  const f30 = { t: 30, ships: { a: cruise([0, 0, 0], v, 30) } };
  f29.ships.a.plan = linePlan(f29.ships.a.pos, v, 29, 30, 1);
  f30.ships.a.plan = linePlan(f30.ships.a.pos, [0, 50, 0], 30, 60, 2);
  const loader = { simTrace: [f29, f30], events: [] };
  const pred = new TrajectoryPredictor(loader, { samples: 8 });
  let tr = pred.predict(29.5, interpolated(loader, 29.5)).get('a');
  assert.equal(tr.tCp, 30);
  closeVec(tr.cp, [1500, 0, 0], 1e-6);
  tr = pred.predict(30, interpolated(loader, 30)).get('a');
  assert.equal(tr.tCp, 60);
  closeVec(tr.cp, [1500, 1500, 0], 1e-6);
});

test('without plans the state is propagated with its acceleration to the next checkpoint', () => {
  const acc = [0, 2, 0];
  const frame = (t) => ({
    t, ships: {
      a: { pos: [10 * t, t * t, 0], vel: [10, 2 * t, 0], acc, fwd: [0, 1, 0], thrust: 1 }
    }
  });
  const loader = { simTrace: [frame(4), frame(5), frame(6)], events: [], sourceData: { decision_interval_s: 30 } };
  const pred = new TrajectoryPredictor(loader, { samples: 12 });
  const ships = interpolated(loader, 5.25);
  const tr = pred.predict(5.25, ships).get('a');
  assert.equal(tr.source, 'kinematic');
  assert.equal(tr.tCp, 30);
  // The Hermite hull position is exact for this parabola
  closeVec(ships.a.position, [52.5, 5.25 * 5.25, 0], 1e-9);
  closeVec(tr.cp, [300, 900, 0], 1e-6, 'x = x0 + vH + aH^2/2');
  closeVec(tr.cpVel, [10, 60, 0], 1e-6);
});

test('fallback acceleration: throttle x hull max along the nose, else finite differences', () => {
  const a = 0.5 * 1.5 * G;
  const frame = (t, extra = {}) => ({
    t, ships: { c: { pos: [0.5 * a * t * t, 0, 0], vel: [a * t, 0, 0], fwd: [1, 0, 0], thrust: 0.5, ...extra } }
  });
  const trace = [frame(0), frame(1), frame(2)];
  const loader = { simTrace: trace, events: [], ships: { c: { type: 'cruiser', faction: 'beta' } } };
  const pred = new TrajectoryPredictor(loader, { samples: 8 });
  const tr = pred.predict(1, interpolated(loader, 1)).get('c');
  assert.equal(tr.faction, 'beta');
  close(tr.cp[0], 0.5 * a * 30 * 30, 1e-6, 'uniform acceleration from rest');
  close(pred.hullAccel('c'), HULL_ACCEL_G.cruiser * G, 1e-9, 'class table when too few samples');

  // Calibrated from the trace when there is enough thrusting history
  const long = [];
  for (let t = 0; t <= 10; t++) long.push(frame(t));
  const cal = new TrajectoryPredictor({ simTrace: long, events: [], ships: { c: { type: 'corvette' } } });
  close(cal.hullAccel('c'), 1.5 * G, 1e-6);

  // No nose / throttle: velocity finite difference
  const bare = t => ({ t, ships: { c: { pos: [0.5 * a * t * t, 0, 0], vel: [a * t, 0, 0] } } });
  const fd = new TrajectoryPredictor({ simTrace: [bare(0), bare(1), bare(2)], events: [] }, { samples: 8 });
  const ships = { c: { position: [0.5 * a, 0, 0], velocity: [a, 0, 0] } };
  close(fd.predict(1, ships).get('c').cp[0], 0.5 * a * 30 * 30, 1e-6);
});

test('fallback sigma grows with recent acceleration jitter', () => {
  const trace = [];
  let vy = 0, y = 0;
  for (let t = 0; t <= 20; t++) {
    const ay = t % 2 ? 6 : -6;   // jinking side to side
    trace.push({ t, ships: { j: { pos: [100 * t, y, 0], vel: [100, vy, 0], fwd: [1, 0, 0] }, s: cruise([0, 0, 0], [100, 0, 0], t) } });
    vy += ay; y += vy;
  }
  const loader = { simTrace: trace, events: [] };
  const pred = new TrajectoryPredictor(loader, { samples: 8 });
  const out = pred.predict(15, interpolated(loader, 15));
  assert.equal(out.get('s').sigma, 0, 'steady cruise: no spread');
  assert.ok(out.get('j').sigma > 100, `jinking ship has spread (${out.get('j').sigma})`);
});

test('destroyed and dying ships get no track', () => {
  const f = t => ({
    t, ships: {
      live: cruise([0, 0, 0], [10, 0, 0], t),
      dying: { ...cruise([0, 0, 0], [10, 0, 0], t), dying: true },
      dead: { ...cruise([0, 0, 0], [10, 0, 0], t), destroyed: true }
    }
  });
  const loader = { simTrace: [f(0), f(1)], events: [] };
  const pred = new TrajectoryPredictor(loader);
  const out = pred.predict(0.5, interpolated(loader, 0.5));
  assert.equal(out.get('live').active, true);
  assert.ok(!out.get('dying')?.active);
  assert.ok(!out.get('dead')?.active);
  // a ship that was predicted and then dies is deactivated
  const f2 = { t: 2, ships: { ...f(2).ships, live: { ...cruise([0, 0, 0], [10, 0, 0], 2), destroyed: true } } };
  loader.simTrace.push(f2);
  assert.equal(pred.predict(2, interpolated(loader, 2)).get('live').active, false);
});

test('actual track follows the recorded future and stops at death', () => {
  const trace = [];
  for (let t = 0; t <= 40; t++) {
    const s = { pos: [t * t, 0, 0], vel: [2 * t, 0, 0], fwd: [1, 0, 0] };
    trace.push({ t, ships: { a: s, b: { ...cruise([0, 0, 0], [5, 0, 0], t), destroyed: t >= 20 } } });
  }
  const loader = { simTrace: trace, events: [], sourceData: { decision_interval_s: 30 } };
  const pred = new TrajectoryPredictor(loader, { samples: 21 });
  const ships = interpolated(loader, 10.5);
  const out = pred.predict(10.5, ships, true);
  const a = out.get('a');
  assert.equal(a.actual.n, 21);
  assert.equal(a.actual.hasCp, true);
  closeVec(a.actual.cp, [900, 0, 0], 1e-6);
  closeVec(a.actual.path.subarray(0, 3), ships.a.position, 1e-9, 'actual starts on the rendered hull');
  const b = out.get('b');
  assert.equal(b.actual.hasCp, false, 'killed before the checkpoint');
  assert.equal(b.actual.endT, 19);
  // without the flag nothing is computed
  assert.equal(pred.predict(10.5, ships, false).get('a').actual.n, 0);
});

test('hermite helpers agree and hit the samples', () => {
  const p0 = [0, 0, 0], v0 = [10, 0, 0], p1 = [10, 5, 0], v1 = [10, 10, 0];
  const out = new Float64Array(3);
  hermiteInto(out, 0, p0, v0, p1, v1, 1, 0);
  closeVec(out, p0, 1e-12);
  hermiteInto(out, 0, p0, v0, p1, v1, 1, 1);
  closeVec(out, p1, 1e-12);
  hermiteInto(out, 0, p0, v0, p1, v1, 1, 0.3);
  closeVec(out, Interpolator.hermiteVec3(p0, v0, p1, v1, 1, 0.3), 1e-12);
  // Frames without timestamps keep the old linear interpolation
  const ships = Interpolator.interpolateShips({ ships: { s: { pos: p0, vel: v0, fwd: [1, 0, 0] } } },
    { ships: { s: { pos: p1, vel: v1, fwd: [1, 0, 0] } } }, 0.5);
  assert.deepEqual(ships.s.position, [5, 2.5, 0]);
});

test('track mode cycles off -> predicted -> actual, skipping actual while live, and persists', () => {
  assert.equal(nextTrackMode('off'), 'predicted');
  assert.equal(nextTrackMode('predicted'), 'actual');
  assert.equal(nextTrackMode('actual'), 'off');
  assert.equal(nextTrackMode('predicted', { allowActual: false }), 'off');

  const store = new Map();
  const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  assert.equal(loadTrackMode(storage), 'predicted');
  saveTrackMode(storage, 'actual');
  assert.equal(store.get(TRACK_MODE_KEY), 'actual');
  assert.equal(loadTrackMode(storage), 'actual');
  store.set(TRACK_MODE_KEY, 'bogus');
  assert.equal(loadTrackMode(storage), 'predicted');

  const broken = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
  assert.equal(loadTrackMode(broken), 'predicted');
  assert.doesNotThrow(() => saveTrackMode(broken, 'off'));

  let live = true;
  const changes = [];
  const keys = new EventTarget();
  const ctl = new TrackModeControl({ storage, isLive: () => live, onChange: m => changes.push(m), keyTarget: keys });
  ctl.set('actual');
  assert.equal(ctl.effectiveMode(), 'predicted', 'no recorded future while live');
  ctl.cycle();
  assert.equal(ctl.mode, 'off');
  live = false;
  const press = () => { const e = new Event('keydown'); e.code = 'KeyT'; keys.dispatchEvent(e); };
  press();
  press();
  assert.equal(ctl.mode, 'actual');
  assert.deepEqual(changes, ['actual', 'off', 'predicted', 'actual']);
});
