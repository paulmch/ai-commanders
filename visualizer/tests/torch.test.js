import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TORCH_LEVELS, clusterLevels, plumeShape, torchCough } from '../src/render/Torch.js';

test('plume shape follows the throttle: idle is short and dim, full burn long and hot', () => {
  let prev = plumeShape(0);
  assert.ok(prev.length > 0.05 && prev.length < 0.25);
  assert.ok(prev.core < 0.1, 'an idle torch has no white core');
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
