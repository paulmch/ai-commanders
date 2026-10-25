import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  resolveQuality, isSoftwareRenderer, nextQuality, QUALITY_PRESETS, GraphicsSettings
} from '../src/render/Quality.js';
import { kelvinToRGB, makeStarCatalogue, makeNoiseVolume } from '../src/render/SkyTextures.js';
import { sunScreenInfo } from '../src/render/sunScreen.js';
import { dustLayerScales } from '../src/render/DustMotes.js';

const SWIFTSHADER = 'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)';

test('software rasterisers are detected, real GPUs are not', () => {
  assert.equal(isSoftwareRenderer(SWIFTSHADER), true);
  assert.equal(isSoftwareRenderer('llvmpipe (LLVM 15.0.7, 256 bits)'), true);
  assert.equal(isSoftwareRenderer('ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11)'), true);
  assert.equal(isSoftwareRenderer('ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0)'), false);
  assert.equal(isSoftwareRenderer('Apple M2'), false);
  assert.equal(isSoftwareRenderer(''), false);
});

test('quality resolves param > saved choice > auto-detect', () => {
  assert.equal(resolveQuality({ rendererName: SWIFTSHADER }), 'low');
  assert.equal(resolveQuality({ rendererName: 'NVIDIA GeForce' }), 'high');
  assert.equal(resolveQuality({ stored: 'high', rendererName: SWIFTSHADER }), 'high');
  assert.equal(resolveQuality({ param: 'low', stored: 'high', rendererName: 'NVIDIA' }), 'low');
  // garbage values fall through
  assert.equal(resolveQuality({ param: 'ultra', stored: 'bogus', rendererName: SWIFTSHADER }), 'low');
  assert.equal(nextQuality('low'), 'high');
  assert.equal(nextQuality('high'), 'low');
});

test('low preset drops the costly features, high keeps them', () => {
  const lo = QUALITY_PRESETS.low, hi = QUALITY_PRESETS.high;
  assert.equal(lo.msaa, 0);
  assert.equal(lo.godRays, false);
  assert.equal(lo.dust, false);
  assert.equal(lo.fxaa, true);
  assert.ok(hi.msaa >= 4 && hi.godRays && hi.dust);
  assert.ok(lo.skySize < hi.skySize && lo.starCount < hi.starCount);
});

test('GraphicsSettings notifies subscribers and adjusts the pixel ratio', () => {
  const renderer = {
    pr: 2,
    getContext: () => ({ getExtension: () => null, getParameter: () => 'NVIDIA GeForce', RENDERER: 0x1F01 }),
    getPixelRatio() { return this.pr; },
    setPixelRatio(v) { this.pr = v; }
  };
  const prevDpr = globalThis.devicePixelRatio;
  globalThis.devicePixelRatio = 2;
  try {
    const g = new GraphicsSettings(renderer);
    assert.equal(g.level, 'high');
    assert.equal(renderer.pr, QUALITY_PRESETS.high.pixelRatioCap);
    const seen = [];
    g.onChange((lvl, preset) => seen.push([lvl, preset.msaa]));
    g.set('low', { persist: false });
    g.set('low', { persist: false });            // no-op when unchanged
    assert.deepEqual(seen, [['low', 0]]);
    assert.equal(renderer.pr, 1);
    g.toggle();
    assert.equal(g.level, 'high');
  } finally {
    globalThis.devicePixelRatio = prevDpr;
  }
});

test('black-body colours run red to blue and are normalised', () => {
  const cool = kelvinToRGB(3000), sun = kelvinToRGB(5800), hot = kelvinToRGB(15000);
  for (const c of [cool, sun, hot]) assert.ok(Math.abs(Math.max(...c) - 1) < 1e-9);
  assert.ok(cool[0] > cool[2] * 2, 'M star is red');
  assert.ok(hot[2] >= hot[0], 'B star is blue');
  assert.ok(sun[0] >= sun[1] && sun[1] >= sun[2], 'G star is warm white');
});

test('star catalogue is deterministic, brightest-first, few spiked stars', () => {
  const a = makeStarCatalogue(3000, { seed: 5, bandNormal: [0, 1, 0] });
  const b = makeStarCatalogue(3000, { seed: 5, bandNormal: [0, 1, 0] });
  assert.deepEqual(a.color, b.color);
  const lum = (i) => a.color[i * 3] * 0.2126 + a.color[i * 3 + 1] * 0.7152 + a.color[i * 3 + 2] * 0.0722;
  // first entries far brighter than the tail (low quality keeps a prefix)
  assert.ok(lum(0) > 10 * lum(2999));
  let spiked = 0, firstPlain = -1;
  for (let i = 0; i < 3000; i++) {
    const d = Math.hypot(a.dir[i * 3], a.dir[i * 3 + 1], a.dir[i * 3 + 2]);
    assert.ok(Math.abs(d - 1) < 1e-5);
    if (a.spike[i] > 0) { spiked++; assert.equal(firstPlain, -1, 'spikes only on the brightest prefix'); }
    else if (firstPlain < 0) firstPlain = i;
  }
  assert.ok(spiked > 0 && spiked <= 28);
});

test('noise volume tiles seamlessly and is centred', () => {
  const size = 16;
  const v = makeNoiseVolume(size, 4, 3);
  assert.equal(v.length, size ** 3 * 4);
  const at = (x, y, z, c) => v[((z * size + y) * size + x) * 4 + c];
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i];
  assert.ok(Math.abs(sum / v.length - 127.5) < 12);
  // the step across the wrap is no larger than typical neighbour steps
  let wrap = 0, inner = 0;
  for (let y = 0; y < size; y++) for (let z = 0; z < size; z++) {
    wrap += Math.abs(at(size - 1, y, z, 0) - at(0, y, z, 0));
    inner += Math.abs(at(size / 2, y, z, 0) - at(size / 2 - 1, y, z, 0));
  }
  assert.ok(wrap < inner * 1.8, `wrap ${wrap} vs inner ${inner}`);
});

test('sun screen position: centred when looked at, hidden when behind', () => {
  const cam = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 1000);
  const dir = new THREE.Vector3(0.3, 0.4, -1).normalize();
  cam.lookAt(dir);
  cam.updateMatrixWorld();
  const out = { uv: new THREE.Vector2(), onScreen: 0, radiusUv: new THREE.Vector2() };
  sunScreenInfo(cam, dir, 0.01, out);
  assert.ok(Math.abs(out.uv.x - 0.5) < 1e-6 && Math.abs(out.uv.y - 0.5) < 1e-6);
  assert.equal(out.onScreen, 1);
  assert.ok(out.radiusUv.y > 0 && Math.abs(out.radiusUv.x * cam.aspect - out.radiusUv.y) < 1e-9);
  sunScreenInfo(cam, dir.clone().negate(), 0.01, out);
  assert.equal(out.onScreen, 0);
  // just off the right edge: fading, not popping
  const off = new THREE.Vector3(1.05 * Math.tan(THREE.MathUtils.degToRad(27.5)) * cam.aspect, 0, -1)
    .applyQuaternion(cam.quaternion).normalize();
  sunScreenInfo(cam, off, 0.01, out);
  assert.ok(out.uv.x > 1 && out.onScreen > 0 && out.onScreen < 1);
});

test('dust octaves cross-fade continuously across zoom', () => {
  for (const d of [0.5, 3, 17, 250, 4000]) {
    const s = dustLayerScales(d);
    assert.equal(s.s1, 2 * s.s0);
    assert.ok(Math.abs(s.w0 + s.w1 - 1) < 1e-12);
  }
  // at a power-of-two boundary the fading-out octave has zero weight
  const below = dustLayerScales(8 / 0.9 * (1 - 1e-9)), above = dustLayerScales(8 / 0.9 * (1 + 1e-9));
  assert.ok(below.w0 < 1e-6 && Math.abs(below.s1 - above.s0) < 1e-9 && above.w1 < 1e-6);
});

test('a torn-down scene leaves the shared GL context with default unpack flags', async () => {
  // the next SceneManager reuses the canvas's context; three allocates its
  // empty 3D textures with texImage3D before setting FLIP_Y itself
  const { resetUnpackState } = await import('../src/SceneManager.js');
  const calls = [];
  const gl = { UNPACK_FLIP_Y_WEBGL: 0x9240, UNPACK_PREMULTIPLY_ALPHA_WEBGL: 0x9241, pixelStorei: (k, v) => calls.push([k, v]) };
  resetUnpackState(gl);
  assert.deepEqual(calls, [[0x9240, false], [0x9241, false]]);
  resetUnpackState(null);   // no context (already lost): nothing to do
  resetUnpackState({});
  assert.equal(calls.length, 2);
});
