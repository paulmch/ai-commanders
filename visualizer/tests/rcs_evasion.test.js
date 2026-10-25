import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { assembleHull } from '../src/render/Hulls.js';
import { SceneManager } from '../src/SceneManager.js';
import {
  ATTITUDE, JINK, JINK_NONE, JINK_DODGE, JINK_FACE, ROLE_RADIAL, ROLE_SIDE, MODE_TV, RCS_TIMING, planJet,
  jinkOf, buildShipKinematics, sampleKinematics, newSample, bodySwing, lookAtBasis,
  jetDims, jetClearance, jetRoles, RcsPlanner, defaultRcsMounts
} from '../src/render/Rcs.js';

const DEG = Math.PI / 180;

test('evasion state: recorder field, then execution mode, then the EVASIVE order', () => {
  assert.equal(jinkOf({ evade: 'evade' }), JINK_DODGE);
  assert.equal(jinkOf({ evade: 'wobble' }), JINK_DODGE);
  assert.equal(jinkOf({ evade: 'run' }), JINK_DODGE);
  assert.equal(jinkOf({ evade: 'present' }), JINK_FACE);
  assert.equal(jinkOf({ evade: 'tank' }), JINK_FACE);
  assert.equal(jinkOf({ evade: 'brace' }), JINK_NONE);
  // the recorder writes evade: null outside evasion; it wins over the mode
  assert.equal(jinkOf({ evade: null, execution_mode: 'PADLOCK', maneuver: 'EVASIVE' }), JINK_NONE);
  // recordings with execution modes but no evade field
  assert.equal(jinkOf({ execution_mode: 'EVADE' }), JINK_DODGE);
  assert.equal(jinkOf({ execution_mode: 'WOBBLE' }), JINK_DODGE);
  assert.equal(jinkOf({ execution_mode: 'PRESENT' }), JINK_FACE);
  assert.equal(jinkOf({ execution_mode: 'INTERCEPT' }), JINK_NONE);
  // and the oldest: only the maneuver order
  assert.equal(jinkOf({ maneuver: 'EVASIVE' }), JINK_DODGE);
  assert.equal(jinkOf({ maneuver: 'INTERCEPT' }), JINK_NONE);
  assert.equal(jinkOf({}), JINK_NONE);
});

/**
 * A torch-lit ship jinking the way simulation.py flies EVASIVE: spinning at
 * its turn-rate cap, the spin swung onto a new aim every `period` s (the
 * unsigned rate means the axis jumps at full rate), coasting in between.
 * Sampled at 1 Hz like the recorder: 4-decimal headings and, with
 * `fields`, omega (nose change since the previous frame), alpha and turn.
 * `state(t)` returns the per-frame evasion fields to merge in.
 */
function jinkTrace({ shipType = 'battleship', T = 70, t0 = 10, t1 = 56, period = 5, fields = true, state }) {
  const w = ATTITUDE[shipType].wtv * DEG;
  const aims = [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1], [0.6, 0.8, 0]];
  const fwd = new THREE.Vector3(1, 0, 0);
  const axis = new THREE.Vector3();
  const frames = [];
  const flips = [];
  let prevW = [0, 0, 0];
  for (let t = 0; t <= T; t++) {
    let om = [0, 0, 0];
    if (t > t0 && t <= t1) {
      const k = Math.floor((t - 1 - t0) / period);
      if ((t - 1 - t0) % period === 0) {
        // new aim: spin axis = fwd x aim, as _rotate_ship_toward picks it
        axis.copy(fwd).cross(new THREE.Vector3(...aims[k % aims.length]));
        if (axis.lengthSq() < 1e-9) axis.set(0, 1, 0);
        axis.normalize();
        flips.push(t - 1);
      }
      fwd.applyAxisAngle(axis, w).normalize();
      om = axis.toArray().map(v => +(v * w).toFixed(5));
    }
    const s = {
      pos: [0, 0, 0], vel: [0, 0, 0], fwd: fwd.toArray().map(v => +v.toFixed(4)),
      thrust: 1, maneuver: 'EVASIVE', destroyed: false, dying: false, hull: 100
    };
    if (fields) {
      const al = om.map((v, c) => v - prevW[c]);
      s.omega = om;
      s.alpha = al;
      s.turn = Math.hypot(...al) > 1e-6 ? 'tv' : null;
    }
    prevW = om;
    Object.assign(s, state ? state(t) : {});
    frames.push({ t, ships: { ship: s } });
  }
  return { frames, flips };
}

const _bb = {};
function hull(type = 'battleship') {
  if (!_bb[type]) {
    const L = { battleship: 16, corvette: 4 }[type], W = { battleship: 4.4, corvette: 1.04 }[type];
    const size = { length: L, width: W };
    const kit = assembleHull(type, size, SceneManager.prototype.getEngineConfig.call(null, type, size), 1234);
    const mounts = kit.rcsMounts.map(m => ({ pos: m.pos.toArray(), dir: m.dir.toArray() }));
    const dims = jetDims(L);
    _bb[type] = { mounts, roles: jetRoles(mounts, jetClearance(kit.rcsMounts, kit.parts, dims.len), dims.len) };
  }
  return _bb[type];
}

function planner(frames, type = 'battleship') {
  const { mounts, roles } = hull(type);
  return new RcsPlanner({ shipId: 'ship', shipType: type, mounts, roles, trace: frames });
}

/** Every pulse cell of the planner over [a, b): {t, kick, trims, jets}. */
function cells(p, a, b) {
  const out = [];
  for (let c = p.cellIndex(a); p.cellStart(c) < b; c++) {
    const pl = p.plan(c);
    let kick = false;
    const trims = [];
    for (let k = 0; k < p.jetCount; k++) {
      if (pl.on[k] <= 0) continue;
      if (p.roles[k] === ROLE_RADIAL && pl.level[k] >= JINK.kickLevel && pl.on[k] >= JINK.kick) kick = true;
      if (p.roles[k] === ROLE_SIDE) trims.push(k);
    }
    out.push({ t: p.cellStart(c), kick, trims, plan: pl });
  }
  return out;
}

const evading = (t) => ({ evade: 'evade', execution_mode: 'EVADE' });
const padlock = (t) => ({ evade: null, execution_mode: 'PADLOCK' });

test('the sim flickers EVADE/PADLOCK every second or two: the jink state holds across it', () => {
  const { frames } = jinkTrace({ state: t => (t >= 20 && t <= 40 ? (t % 3 ? evading(t) : padlock(t)) : padlock(t)) });
  const k = buildShipKinematics(frames, 'ship', 'battleship');
  for (let i = 21; i <= 39; i++) assert.equal(k.evade[i], JINK_DODGE, `frame ${i}`);
  assert.equal(k.evade[10], JINK_NONE);
  assert.equal(k.evade[50], JINK_NONE);
  // the swing is the nose velocity the viewer's slerp shows
  const s = newSample();
  sampleKinematics(k, 30.5, s);
  assert.equal(s.jink, JINK_DODGE);
  assert.ok(s.evade);
  const rate = Math.hypot(...s.swing);
  assert.ok(Math.abs(rate - ATTITUDE.battleship.wtv * DEG) < 0.002, `swing ${rate}`);
});

test('a spin swung onto a new jink kicks even though the torque does not persist', () => {
  const j = (now, prev, fut, prevRise = 0, jink = JINK_NONE) => ({ now, prev, fut, prevRise, jink });
  // outside evasion a one-frame reversal stays dark (steering jitter)
  assert.equal(planJet(j(1, 0, 0), MODE_TV).on, 0);
  // evading, it is the jink: one firm kick
  const kick = planJet(j(1, 0, 0, 0, JINK_DODGE), MODE_TV);
  assert.equal(kick.level, JINK.kickLevel);
  assert.equal(kick.on, JINK.kick);
  assert.equal(planJet(j(1, 0, 0, 0, JINK_FACE), MODE_TV).level, JINK.kickLevel);
  // ...but not again while the previous one is still recent, and a faint
  // assist follows the weave instead
  const after = planJet(j(1, 0, 0, 1, JINK_DODGE), MODE_TV);
  assert.equal(after.level, JINK.assist);
  assert.ok(after.level < JINK.kickLevel && after.on > 0 && after.on < RCS_TIMING.cell);
  assert.equal(planJet(j(0.1, 0, 0, 0, JINK_DODGE), MODE_TV).on, 0);
});

test('each new jink kicks the attitude couple exactly once', () => {
  const { frames, flips } = jinkTrace({ state: evading });
  const kicks = cells(planner(frames), 5, 66).filter(c => c.kick);
  // one kick per new aim, in the second before the slerp swings onto it,
  // wherever the ship's pulse cells happen to fall
  for (const tf of flips) {
    const near = kicks.filter(c => c.t > tf - 1.6 && c.t < tf + 0.6);
    assert.equal(near.length, 1, `flip at ${tf}: kicks at ${near.map(c => c.t.toFixed(2))}`);
  }
  assert.equal(kicks.length, flips.length);
  // outside evasion only the flips that happen to look sustained from the
  // cell grid kick (never twice)
  const calm = cells(planner(jinkTrace({ state: padlock }).frames), 5, 66).filter(c => c.kick);
  assert.ok(calm.length < flips.length, `${calm.length} calm kicks`);
  for (const tf of flips) assert.ok(calm.filter(c => c.t > tf - 1.6 && c.t < tf + 0.6).length <= 1);
});

test('a dodging ship trims toward where its nose sweeps: side nozzles, a pure push', () => {
  const { frames } = jinkTrace({ state: evading });
  const p = planner(frames);
  const { mounts } = hull();
  const s = newSample(), sw = [0, 0], X = [0, 0, 0], Y = [0, 0, 0];
  const cs = cells(p, 12, 56);
  const trimCells = cs.filter(c => c.trims.length);
  assert.ok(trimCells.length >= 8, `${trimCells.length} trim cells`);
  // trims come with each kick, otherwise sparse short pulses
  assert.ok(trimCells.length < 0.6 * cs.length, `${trimCells.length}/${cs.length}`);
  for (const c of cs.filter(cc => cc.kick)) assert.ok(c.trims.length >= 4, `kick at ${c.t} without trims`);
  for (const c of trimCells) {
    // net reaction force and moment of the firing side nozzles, body frame
    let fx = 0, fy = 0, my = 0, mx = 0, zs = 0;
    for (const k of c.trims) {
      const lv = c.plan.level[k];
      const [dx, dy] = mounts[k].dir;
      const z = mounts[k].pos[2];
      fx -= lv * dx; fy -= lv * dy;
      mx += lv * z * dy; my -= lv * z * dx;
      zs += lv * Math.abs(z);
    }
    const f = Math.hypot(fx, fy);
    assert.ok(f > 0);
    // no pitch/yaw from the push (moments about the hull origin cancel)
    assert.ok(Math.hypot(mx, my) < 0.08 * zs, `moment ${mx}, ${my} at ${c.t}`);
    // pushing the way the nose is sweeping
    sampleKinematics(p.kin, c.t + 0.4, s);
    lookAtBasis(s.fwd, X, Y);
    bodySwing(s, sw);
    const sl = Math.hypot(sw[0], sw[1]);
    assert.ok((fx * sw[0] + fy * sw[1]) / (f * sl) > 0.85, `trim ${fx},${fy} vs swing ${sw} at ${c.t}`);
  }

  // armour pointing (PRESENT/TANK) still kicks but never trims, and a ship
  // that isn't evading doesn't either
  for (const state of [t => ({ evade: 'present', execution_mode: 'PRESENT' }), padlock]) {
    const other = cells(planner(jinkTrace({ state }).frames), 5, 66);
    assert.equal(other.filter(c => c.trims.length).length, 0);
  }
  const present = cells(planner(jinkTrace({ state: t => ({ evade: 'present' }) }).frames), 5, 66);
  assert.ok(present.filter(c => c.kick).length >= 5);
});

test('recordings without evade/omega fields: heading track and the EVASIVE order', () => {
  // the oldest recordings carry headings and the maneuver order only
  const old = jinkTrace({ fields: false, state: t => ({ maneuver: 'EVASIVE' }) });
  const cs = cells(planner(old.frames), 5, 66);
  assert.ok(cs.filter(c => c.kick).length >= old.flips.length / 2, 'kicks from the heading track');
  assert.ok(cs.filter(c => c.trims.length).length >= 5, 'trims');
  const calm = jinkTrace({ fields: false, state: t => ({ maneuver: 'INTERCEPT' }) });
  assert.equal(cells(planner(calm.frames), 5, 66).filter(c => c.trims.length).length, 0);
});

test('hulls without side nozzles trim with the radial quads; corvette nose trims radially', () => {
  // raycast quads (no side nozzles at all)
  const { frames } = jinkTrace({ shipType: 'corvette', state: evading });
  const quads = new RcsPlanner({ shipId: 'ship', shipType: 'corvette', mounts: defaultRcsMounts(4, 1), trace: frames });
  let trimCells = 0;
  for (let c = quads.cellIndex(12); quads.cellStart(c) < 56; c++) {
    const pl = quads.plan(c);
    if ([...pl.on].filter(v => v > 0).length >= 2) trimCells++;
  }
  assert.ok(trimCells >= 5);
  // the corvette's bow quad sits among the torpedo rails: its side nozzles
  // would fire into them, so the bow trims with its radial nozzles
  const { mounts, roles } = hull('corvette');
  const bowSide = mounts.filter((m, i) => m.pos[2] > 0 && roles[i] === ROLE_SIDE);
  const bowRadial = mounts.filter((m, i) => m.pos[2] > 0 && roles[i] === ROLE_RADIAL);
  assert.equal(bowSide.length, 0);
  assert.equal(bowRadial.length, 4);
});

test('jink cues are a pure function of battle time (seek-safe)', () => {
  const { frames } = jinkTrace({ state: t => (t % 4 ? evading(t) : padlock(t)) });
  const a = planner(frames), b = planner(frames);
  const times = [];
  for (let t = 8; t < 60; t += 0.043) times.push(t);
  const n = a.jetCount;
  const fwd = times.map(t => Array.from({ length: n }, (_, k) => a.intensity(t, k)));
  const back = times.slice().reverse().map(t => Array.from({ length: n }, (_, k) => b.intensity(t, k))).reverse();
  assert.deepEqual(fwd, back);
  assert.ok(fwd.some(row => row.some(v => v > 0.5)));
});
