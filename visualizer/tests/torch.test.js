import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TORCH_LEVELS, clusterLevels, plumeShape, torchCough, plumeSweep, SWEEP_AGE_S, SWEEP_MAX } from '../src/render/Torch.js';
import { Interpolator } from '../src/Interpolator.js';

test('plume shape follows the throttle: idle is short and dim, full burn long and hot', () => {
  let prev = plumeShape(0);
  assert.ok(prev.length > 0.05 && prev.length < 0.25);
  assert.ok(prev.core < 0.3 && prev.wide < 0.3, 'an idle torch is a dim flicker');
  for (let th = 0.05; th <= 1.0001; th += 0.05) {
    const s = plumeShape(th);
    for (const k of ['length', 'width', 'core', 'wide']) assert.ok(s[k] > prev[k], `${k} grows with throttle`);
    prev = s;
  }
  const full = plumeShape(1);
  for (const k of ['length', 'width', 'core', 'wide']) assert.ok(Math.abs(full[k] - 1) < 1e-9, `${k} is 1 at full burn`);
  // clamped outside 0..1, same value every call (pure)
  assert.deepEqual(plumeShape(3), plumeShape(1));
  assert.deepEqual(plumeShape(-1), plumeShape(0));
});

test('a drive cluster keeps one plume worth of light', () => {
  assert.deepEqual(clusterLevels(1), { core: 1, wide: 1 });
  for (const n of [2, 3, 4, 6, 8]) {
    const k = clusterLevels(n);
    // all the wide layers of the cluster overlap: their sum stays near one
    // plume, and never reaches the bloom threshold (~1.35) on its own
    assert.ok(n * k.wide * TORCH_LEVELS.sheath < 1.35, `sheath of ${n} bells stays under bloom`);
    assert.ok(n * k.wide >= 1, 'more bells never get dimmer as a whole');
    assert.ok(k.core < 1 && k.core >= 1 / n);
  }
  // only the core (throat, diamonds) is meant to cross the bloom threshold
  assert.ok(TORCH_LEVELS.core > 1.35 && TORCH_LEVELS.sheath < 1 && TORCH_LEVELS.halo < 0.3);
});

test('torch shader keeps the renderer contracts', () => {
  const src = readFileSync(new URL('../src/render/Torch.js', import.meta.url), 'utf8');
  for (const chunk of ['<common>', '<logdepthbuf_pars_vertex>', '<logdepthbuf_vertex>', '<logdepthbuf_pars_fragment>', '<logdepthbuf_fragment>']) {
    assert.ok(src.includes(`#include ${chunk}`), chunk);
  }
  // battle time never reaches the shader: uTime is the wall-clock turbulence
  // scroll only, and the cough is a pure function of the time passed in
  assert.equal(torchCough(12.34), torchCough(12.34));
  assert.ok(!/uBattle|currentTime/.test(src));
});

// ---------------------------------------------------------------------------
// Gun classes: tracer / muzzle looks and the light pop
// ---------------------------------------------------------------------------
import { shotClass, GUN_CLASSES } from '../src/render/WeaponTimeline.js';
import { SLUG_LOOK, MUZZLE_LOOK, muzzlePop, POP_LIGHTS } from '../src/render/GunLooks.js';

test('gun class comes from the weapon name and defaults to battery', () => {
  assert.equal(shotClass({ weapon_name: 'Coilgun Battery Mk3' }), 0);
  assert.equal(shotClass({ weapon_name: 'Light Coilgun Battery Mk3' }), 0);
  assert.equal(shotClass({ weapon_name: 'Heavy Coilgun Battery Mk3' }), 1);
  assert.equal(shotClass({ weapon_name: 'Spinal Coiler Mk3' }), 2);
  assert.equal(shotClass({ weapon_name: 'Heavy Siege Coiler Mk3' }), 3);
  assert.equal(shotClass({}), 0);
  assert.equal(shotClass(undefined), 0);
  assert.equal(GUN_CLASSES.length, 4);
});

test('bigger guns throw bigger tracers and flashes; the pop is a pure envelope', () => {
  assert.equal(SLUG_LOOK.length, 4);
  assert.equal(MUZZLE_LOOK.length, 4);
  for (let c = 1; c < 4; c++) {
    for (const k of ['width', 'length', 'gain', 'head', 'headPx']) assert.ok(SLUG_LOOK[c][k] > SLUG_LOOK[c - 1][k], `slug ${k}`);
    for (const k of ['size', 'jet', 'sparks', 'spray', 'light', 'reach']) assert.ok(MUZZLE_LOOK[c][k] > MUZZLE_LOOK[c - 1][k], `muzzle ${k}`);
    assert.ok(MUZZLE_LOOK[c].rings >= MUZZLE_LOOK[c - 1].rings);
  }
  // battery fire keeps the pass-4 tracer width and level
  assert.equal(SLUG_LOOK[0].width, 0.045);
  assert.equal(SLUG_LOOK[0].gain, 1);
  assert.ok(POP_LIGHTS <= 2, 'pops leave half of the 4-light reserve to detonations');
  assert.equal(muzzlePop(0, 0.3), 1);
  assert.equal(muzzlePop(0.3, 0.3), 0);
  assert.equal(muzzlePop(-1, 0.3), 0);
  let prev = 1;
  for (let a = 0.02; a < 0.3; a += 0.02) { const v = muzzlePop(a, 0.3); assert.ok(v < prev && v >= 0); prev = v; }
  assert.equal(muzzlePop(0.1, 0.3), muzzlePop(0.1, 0.3));
});

test('a turning ship drags its plume: sweep follows the recorded turn rate', () => {
  assert.equal(Math.hypot(plumeSweep(0, 0).x, plumeSweep(0, 0).y), 0);
  // yaw about +Y (bow swings to +X, stern to -X): the old exhaust lags on +X
  const yaw = plumeSweep(0, 0.05);
  assert.ok(yaw.x > 0 && yaw.y === 0);
  assert.ok(Math.abs(yaw.x - 0.05 * SWEEP_AGE_S) < 1e-12);
  // pitch about +X (bow swings to -Y): the tail lags on -Y
  assert.ok(plumeSweep(0.05, 0).y < 0);
  // a corvette's snap turn saturates instead of folding the plume over
  const snap = plumeSweep(0.2, -0.3);
  assert.ok(Math.abs(Math.hypot(snap.x, snap.y) - SWEEP_MAX) < 1e-12);
  // reversing the turn mirrors the sweep (pure, no history)
  const back = plumeSweep(-0.2, 0.3);
  assert.ok(Math.abs(back.x + snap.x) < 1e-12 && Math.abs(back.y + snap.y) < 1e-12);
});

test('interpolated ship state carries the recorded turn rate, or null on old recordings', () => {
  const ship = (extra) => ({ pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], thrust: 1, ...extra });
  const traced = Interpolator.interpolateShips(
    { t: 0, ships: { a: ship({ omega: [0, 0.01, 0] }) } },
    { t: 1, ships: { a: ship({ omega: [0, 0.04, 0] }) } }, 0.5);
  assert.ok(Math.abs(traced.a.omega[1] - 0.025) < 1e-12 && traced.a.omega[0] === 0, 'continuous across frames');
  const old = Interpolator.interpolateShips({ t: 0, ships: { a: ship() } }, { t: 1, ships: { a: ship() } }, 0.5);
  assert.equal(old.a.omega, null);
});

test('plume axes are parallel to the thrust axis: no cant on any bell', () => {
  const src = readFileSync(new URL('../src/SceneManager.js', import.meta.url), 'utf8');
  const i = src.indexOf('createTorch({ radius: plumeR');
  assert.ok(i > 0);
  const block = src.slice(i, src.indexOf('torches.push(torch)', i));
  assert.ok(block.includes('torch.group.position.set('), 'bells are placed');
  assert.ok(!/torch\.group\.(rotation|quaternion|lookAt)/.test(block), 'and never rotated off the drive axis');
});
