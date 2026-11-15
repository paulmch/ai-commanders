import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { QUALITY_PRESETS } from '../src/render/Quality.js';
import { makeRingProfile } from '../src/render/SkyTextures.js';
import {
  shadowBoxRadius, shadowBasis, fitSunShadow,
  SHADOW_MIN_RADIUS, SHADOW_MAX_RADIUS, SHADOW_DEPTH_MARGIN
} from '../src/render/sunShadow.js';

const SUN = new THREE.Vector3(0.62, 0.42, 0.28).normalize();

test('expensive look features are high-tier only', () => {
  const { low, high } = QUALITY_PRESETS;
  assert.equal(low.shadows, false);
  assert.equal(low.streak, false);
  assert.equal(low.shadowMapSize, 0);
  assert.equal(high.shadows, true);
  assert.equal(high.streak, true);
  assert.ok(high.shadowMapSize >= 2048);
});

test('shadow box covers the view at focus depth and is quantised', () => {
  for (const [dist, fov, aspect] of [[96, 20, 16 / 9], [30, 55, 16 / 9], [500, 55, 2.2], [12, 55, 0.6]]) {
    const R = shadowBoxRadius(dist, fov, aspect);
    const halfW = dist * Math.tan(THREE.MathUtils.degToRad(fov) / 2) * Math.max(aspect, 1);
    assert.ok(R >= halfW, `box ${R} covers half-width ${halfW}`);
    assert.ok(R <= Math.max(SHADOW_MIN_RADIUS, halfW * 1.08 * 2 ** 0.25) + 1e-6, 'no more than a quarter octave of slack');
    const steps = Math.log2(R) * 4;
    assert.ok(Math.abs(steps - Math.round(steps)) < 1e-6, 'quarter-octave sizes');
  }
  // a small dolly inside one step does not change the box (no resolution crawl)
  assert.equal(shadowBoxRadius(100, 55, 16 / 9), shadowBoxRadius(101, 55, 16 / 9));
  assert.equal(shadowBoxRadius(0.01, 55, 1), SHADOW_MIN_RADIUS);
  assert.equal(shadowBoxRadius(1e9, 55, 1), SHADOW_MAX_RADIUS);
  // about one texel per screen pixel: a 4096 map over a 900 px tall 16:9 view
  const R = shadowBoxRadius(96, 20, 16 / 9);
  const texel = 2 * R / 4096, pixel = 2 * 96 * Math.tan(THREE.MathUtils.degToRad(10)) / 900;
  assert.ok(texel < pixel, `texel ${texel} vs pixel ${pixel}`);
});

test('shadow basis matches a lookAt from the sun and is orthonormal', () => {
  const b = shadowBasis(SUN);
  assert.ok(Math.abs(b.x.dot(b.y)) < 1e-9 && Math.abs(b.x.dot(b.z)) < 1e-9 && Math.abs(b.y.dot(b.z)) < 1e-9);
  const cam = new THREE.OrthographicCamera();
  cam.position.copy(SUN).multiplyScalar(50);
  cam.lookAt(0, 0, 0);
  cam.updateMatrixWorld();
  const e = cam.matrixWorld.elements;
  assert.ok(new THREE.Vector3(e[0], e[1], e[2]).distanceTo(b.x) < 1e-6);
  assert.ok(new THREE.Vector3(e[4], e[5], e[6]).distanceTo(b.y) < 1e-6);
  assert.ok(new THREE.Vector3(e[8], e[9], e[10]).distanceTo(b.z) < 1e-6);
});

test('shadow box centre snaps to texels and keeps the sun direction', () => {
  const b = shadowBasis(SUN);
  const R = 32, size = 2048, texel = 2 * R / size;
  const target = new THREE.Vector3(), pos = new THREE.Vector3();
  const focus = new THREE.Vector3(123.4567, -45.678, 9.1011);
  const fit = fitSunShadow(focus, b, R, size, target, pos);
  assert.equal(fit.texel, texel);
  for (const axis of [b.x, b.y]) {
    const t = target.dot(axis) / texel;
    assert.ok(Math.abs(t - Math.round(t)) < 1e-6, 'centre on the texel grid');
    assert.ok(Math.abs(target.dot(axis) - focus.dot(axis)) <= texel / 2 + 1e-9, 'within half a texel of the focus');
  }
  assert.ok(Math.abs(target.dot(b.z) - focus.dot(b.z)) < 1e-9);
  // the light still shines along the sun direction, from outside the box
  const dir = pos.clone().sub(target);
  assert.ok(dir.clone().normalize().distanceTo(SUN) < 1e-9);
  assert.ok(Math.abs(dir.length() - (R + SHADOW_DEPTH_MARGIN)) < 1e-9);
  assert.ok(fit.far >= dir.length() + R, 'far plane behind the box');
  // sub-texel camera motion leaves the map where it is
  const t2 = new THREE.Vector3(), p2 = new THREE.Vector3();
  fitSunShadow(focus.clone().addScaledVector(b.x, texel * 0.2), b, R, size, t2, p2);
  assert.ok(Math.abs(t2.dot(b.x) - target.dot(b.x)) < 1e-9);
  // biases scale with the texel
  const coarse = fitSunShadow(focus, b, R * 8, size, t2, p2);
  assert.ok(Math.abs(coarse.normalBias / fit.normalBias - 8) < 1e-9);
  assert.ok(fit.bias < 0 && fit.bias > -1e-3);
});

test('ring profile: dense main ring, dark division, fine ringlets, clean edges', () => {
  const w = 2048;
  const p = makeRingProfile(w);
  assert.deepEqual(Array.from(makeRingProfile(w).slice(0, 400)), Array.from(p.slice(0, 400)), 'deterministic');
  const alpha = (u0, u1) => {
    let s = 0, n = 0;
    for (let x = Math.floor(u0 * w); x < Math.floor(u1 * w); x++) { s += p[x * 4 + 3] / 255; n++; }
    return s / n;
  };
  const main = alpha(0.26, 0.54), inner = alpha(0.05, 0.2), division = alpha(0.595, 0.645), outer = alpha(0.69, 0.85);
  assert.ok(main > 0.6, `main ring is dense (${main})`);
  assert.ok(division < 0.12, `division is nearly clear (${division})`);
  assert.ok(inner < main * 0.5 && inner > 0.05, `inner ring is thin (${inner})`);
  assert.ok(outer > division * 3 && outer < main, `outer ring between the two (${outer})`);
  assert.equal(p[3], 0);
  assert.ok(p[(w - 1) * 4 + 3] < 8);
  // ringlets: plenty of texel-to-texel variation inside the main ring
  let rough = 0;
  for (let x = Math.floor(0.26 * w); x < Math.floor(0.54 * w); x++) rough += Math.abs(p[x * 4 + 3] - p[(x + 1) * 4 + 3]);
  assert.ok(rough / (0.28 * w) > 1.5, `fine structure (${rough / (0.28 * w)})`);
});
