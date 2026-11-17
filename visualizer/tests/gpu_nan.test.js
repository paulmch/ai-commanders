import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { buildDamageTimeline, damageStateAt } from '../src/DamageTimeline.js';
import { DamageFX } from '../src/render/DamageFX.js';

// Owner report (pass 5): "when damage appears on the ship sometimes black
// squares appear" - on a real GPU only. GLSL leaves pow(x, y) undefined for
// x < 0: desktop GPUs return NaN, SwiftShader (the headless harness)
// returns |x|^y. The smoke billboards hid an expired puff through its
// alpha, smoothstep(..) * pow(1 - f, 1.4) with f > 1: NaN on a GPU, so the
// quad was drawn with a NaN colour, and bloom grew it into black squares.
//
// To see GPU behaviour in a headless frame, rewrite every ShaderMaterial's
// pow( to gpuPow( (then material.needsUpdate = true) with
//   float gpuPow(float x, float y) { if (x >= 0.0) return pow(x, y);
//     if (y == floor(y)) { float r = pow(-x, y); return mod(y, 2.0) == 0.0 ? r : -r; }
//     return sqrt(-1.0 - x * x); }
// (NaN for a negative base with a non-integer exponent, as ANGLE/D3D does;
// drop the integer branch to model drivers that do not expand pow(x, 2.0)).

// Beta's hull falls to 30% at t = 20 (smoking), it dies at 30, is gone at 36.
function recording() {
  const trace = [];
  for (let t = 0; t <= 60; t++) {
    const hull = t < 12 ? 100 : t < 20 ? 60 : 30;
    trace.push({ t, ships: {
      alpha: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], hull: 100 },
      beta: { pos: [100000, 0, 0], vel: [-500, 0, 0], fwd: [-1, 0, 0], hull, dying: t >= 30 && t < 36, destroyed: t >= 36 }
    } });
  }
  const events = [
    { timestamp: 11, event_type: 'hit', ship_id: 'beta',
      data: { shooter_id: 'alpha', hit_location: 'lateral', kinetic_energy_gj: 8, penetrated: true, critical_hit: true } },
    { timestamp: 21, event_type: 'hit', ship_id: 'beta',
      data: { shooter_id: 'alpha', hit_location: 'nose', kinetic_energy_gj: 6, penetrated: true } }
  ];
  return { events, sim_trace: trace };
}

function boxShip(L, W) {
  const g = new THREE.Group();
  const geom = new THREE.BoxGeometry(W, W, L).toNonIndexed();
  g.userData = { size: { length: L, width: W }, parts: [{ geom, center: new THREE.Vector3(), mat: 'hull' }] };
  return g;
}

// DamageFX without a renderer (as effects.test.js): enough to run a ship's smoke ring
function smokeView() {
  const fx = Object.create(DamageFX.prototype);
  const groups = new Map([['alpha', boxShip(8, 2)], ['beta', boxShip(8, 2)]]);
  Object.assign(fx, {
    sm: { ships: groups }, scene: new THREE.Scene(), SCALE: 0.001, views: new Map(), shipIds: [],
    planeGeom: new THREE.PlaneGeometry(1, 1), coneGeom: new THREE.CylinderGeometry(1, 0.16, 1, 8, 1, true),
    craterMat: new THREE.MeshBasicMaterial(), ventMat: new THREE.MeshBasicMaterial(), smokeMat: new THREE.MeshBasicMaterial(),
    raycaster: new THREE.Raycaster(), probe: new THREE.Mesh(undefined, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })),
    _cands: [], _hits: [], _ds: {}, _st: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false },
    _keptWear: new Map(), timeline: buildDamageTimeline(recording())
  });
  fx.probe.updateMatrixWorld();
  const v = fx.view('beta');
  return (t) => {
    v.resolveUpTo(t + 0.25);
    damageStateAt(fx.timeline, 'beta', t, v.state);
    v._updateSmoke(t);
    if (!v.smoke) return null;
    const a = v.smoke.geometry.attributes;
    return { S0: Array.from(a.aS0.array), S1: Array.from(a.aS1.array), S2: Array.from(a.aS2.array) };
  };
}

test('the smoke ring keeps no expired puff, and its buffers do not depend on the seek path', () => {
  const play = smokeView();
  const LIFE = 6.5;
  let smoked = 0;
  for (let i = 0; i <= 1200; i++) {
    const t = 18 + i * 0.025;
    const got = play(t);
    if (!got) continue;
    for (let s = 0; s < got.S0.length / 4; s++) {
      const t0 = got.S0[s * 4 + 3];
      if (t0 > 1e8) continue;                 // empty slot
      smoked++;
      const age = t - t0;
      assert.ok(age >= -1e-4 && age <= LIFE + 1e-4,   // t0 is a float32
        `slot ${s} holds a puff aged ${age} at ${t}`);
    }
    // playback, a direct seek and a backward seek land on the same buffers
    if (i % 200 === 100) {
      const direct = smokeView()(t);
      assert.deepEqual(got, direct, `direct seek at ${t}`);
      const back = smokeView();
      back(t + 9); back(t + 4);
      assert.deepEqual(back(t), got, `backward seek to ${t}`);
    }
  }
  assert.ok(smoked > 500, 'the ship smoked');
});

// --------------------------------------------------------------------------
// Every GLSL pow() in the hull damage effects has a base that cannot go
// negative. A new pow() must be reviewed and listed here with its reason.
// --------------------------------------------------------------------------
const NON_NEGATIVE = new Map([
  ['1.0 - t', 'vent: t = cone position.y, 0..1'],
  ['vEdge', 'vent: abs(dot(...))'],
  ['1.0 - fl', 'smoke: fl = clamp(f, 0.0, 1.0)'],
  ['facing', 'plume wash: facing < 0.03 is discarded first'],
  ['loss', 'hull wear: lossOf() clamps to 0..1']
]);

function powBases(src) {
  const out = [];
  const re = /(^|[^A-Za-z0-9_.])pow\(/g;
  let m;
  while ((m = re.exec(src))) {
    let i = m.index + m[0].length, depth = 0, base = '';
    for (; i < src.length; i++) {
      const c = src[i];
      if (c === '(') depth++;
      else if (c === ')') depth--;
      else if (c === ',' && depth === 0) break;
      base += c;
    }
    out.push(base.trim());
  }
  return out;
}

test('damage shaders never take pow() of a base that can go negative (NaN on GPUs)', () => {
  for (const file of ['../src/render/DamageFX.js', '../src/render/HullWear.js']) {
    const src = readFileSync(new URL(file, import.meta.url), 'utf8').replace(/Math\.pow\(/g, 'Math_pow(').replace(/\/\/[^\n]*/g, '');
    const bases = powBases(src);
    assert.ok(bases.length > 0 || file.includes('HullWear'), file);
    for (const b of bases) {
      const guarded = /^(max|clamp|abs)\(/.test(b) || NON_NEGATIVE.has(b);
      assert.ok(guarded, `${file}: pow(${b}, ...) - can the base go negative? Use dmgSq() for squares, clamp the base, or list it in NON_NEGATIVE with the reason`);
    }
  }
  assert.deepEqual(powBases('float x = pow(1.0 - f, 1.4) + pow(max(a, 0.0), 2.0);'), ['1.0 - f', 'max(a, 0.0)']);
});
