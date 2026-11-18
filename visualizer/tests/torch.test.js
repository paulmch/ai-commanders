import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TORCH_LEVELS, clusterLevels, plumeShape, torchCough, flowPhase, FLOW_RATE } from '../src/render/Torch.js';
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

test('the plume is a rigid lance: no sweep, no breathing, streaks fast enough that none can be followed', () => {
  const src = readFileSync(new URL('../src/render/Torch.js', import.meta.url), 'utf8');
  // a 10,000 km/s exhaust shows no lag when the hull turns: no bend of the centreline
  assert.ok(!/uBend|plumeSweep/.test(src), 'no swept centreline');
  // nothing in the jet moves at "particle" speed: no pressure pulses, no length breathing
  assert.ok(!/pulse|breathe|uTime/.test(src), 'no slow pulses or breathing');
  // the streaks scroll several texture tiles per second (a fbm feature is
  // ~1/8 tile: over half its own length per 60 Hz frame), at incommensurate
  // rates; chemical vents (wrecks) keep slow gas
  const [a, b] = FLOW_RATE.fusion;
  assert.ok(a / 60 > 0.5 / 8 && b / 60 > 0.5 / 8, 'fusion streaks outrun the eye');
  assert.ok(Math.abs(b / a - Math.round(b / a)) > 0.2, 'incommensurate octaves');
  assert.ok(FLOW_RATE.chemical[0] < 1 && FLOW_RATE.chemical[1] < 1);
  // phases wrap on the CPU: a long session never loses float precision in the shader
  const ph = flowPhase(86400.123, FLOW_RATE.fusion);
  assert.ok(ph.every(x => x >= 0 && x < 1));
  assert.deepEqual(flowPhase(-3.3, [1, 1]).map(x => Math.round(x * 1e6) / 1e6), [0.7, 0.7]);
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
