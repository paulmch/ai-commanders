import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
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
// GLSL lint over every viewer source that carries shader code (template
// literals under src/ that read as GLSL). Undefined on a negative argument:
// pow() base, sqrt / inversesqrt / log / log2; normalize() of a zero
// vector is NaN. An argument passes when a guard spans all of it
// (max/clamp/abs/exp/...), when it is 0.5 + 0.5 * sin|cos(...), dot(v, v),
// or a literal; anything else must be listed below with the reason it
// cannot go negative. normalize() cannot be decided from the text: every
// site is listed with its category, so a new one gets reviewed.
// --------------------------------------------------------------------------
const REVIEWED = new Map(Object.entries({
  // pow() bases
  'render/BlastShell.js|pow|ndv': 'abs(dot(...))',
  'render/DamageFX.js|pow|vEdge': 'vent: abs(dot(...))',
  'render/DamageFX.js|pow|1.0 - fl': 'smoke: fl = clamp(f, 0.0, 1.0)',
  'render/DamageFX.js|pow|facing': 'plume wash: facing < 0.03 is discarded first',
  'render/Environment.js|pow|h.y': 'hash33() is fract(): 0..1',
  'render/Environment.js|pow|clouds': 'fbm() clamps to 0..1',
  'render/Environment.js|pow|thick * 0.5': 'thick = 0.07 + 0.12 * exp(...)',
  'render/Environment.js|pow|fa': 'max(0.0, dot(...))',
  'render/Environment.js|pow|fb': 'max(0.0, dot(...))',
  'render/Environment.js|pow|ci': 'fbm() clamps to 0..1',
  'render/Environment.js|pow|mu': 'sqrt(max(...)) / max(dot(...), 0.0)',
  'render/HullWear.js|pow|loss': 'lossOf() clamps to 0..1',
  'render/Particles.js|pow|1.0 - f': 'f = clamp(age / aLife, 0.0, 1.0)',
  'render/PostFX.js|pow|abs(sin(ang * 3.0 + 0.35)) * w': 'w = r / 0.0028, r = length(...)',
  'render/PostFX.js|pow|abs(sin(ang * 3.0 + 0.35 + 0.5236)) * w * 1.4': 'w = r / 0.0028, r = length(...)',
  'render/Rcs.js|pow|ndv': 'abs(dot(...))',
  'render/Torch.js|pow|t': 't = clamp(s / uLen, 0.0, 1.0)',
  // sqrt / inversesqrt / log2 arguments
  'render/DamageFX.js|sqrt|s': 'vent strength: product of non-negatives, zeroed below 0.035',
  'render/DamageFX.js|inversesqrt|c2': 'c2 = dot(c, c) > 1e-30 (smaller is discarded)',
  'render/PostFX.js|sqrt|(fi + 0.5) / 16.0': 'loop index fi >= 0',
  'render/PostFX.js|sqrt|(fi + 0.5) / float(TAPS)': 'loop index fi >= 0',
  'render/PostFX.js|sqrt|vv': 'vv = dot(v, v)',
  'render/Torch.js|sqrt|dd': 'dd = max(dot(rd.xy, rd.xy), 1e-6)',
  'render/Tracks.js|log2|uDashPx * fw / uDashBase': 'fw = max(fwidth(...), 1e-6); only under uDashBase > 0',
  'render/WeaponBatches.js|sqrt|u': 'u = mix(u0, u1, uv.x), clip parameters in 0..1'
}));

// normalize(): zero only in the listed degenerate case (flagged, not fixed)
const NORMALIZE = new Map(Object.entries({
  'render/BlastShell.js|normalMatrix * normal': 'normal attribute',
  'render/BlastShell.js|position': 'sphere vertex, never the centre',
  'render/BlastShell.js|vNormal': 'interpolated normal (zero only between opposite vertex normals)',
  'render/BlastShell.js|vView': 'view vector (zero only at the camera)',
  'render/DamageFX.js|-mv.xyz': 'view vector (zero only at the camera)',
  'render/DamageFX.js|aNorm * 1.15 + rnd': 'spark direction: zero only if the random kick exactly cancels the normal',
  'render/DamageFX.js|mat3(modelViewMatrix) * mat3(instanceMatrix) * normal': 'normal attribute (vent cone)',
  'render/Environment.js|cameraPosition - vWorld': 'view vector (zero only at the camera)',
  'render/Environment.js|closest - uCenter': 'planet: ray closest point vs centre (zero only through the exact centre)',
  'render/Environment.js|d + w * 0.07': 'unit direction plus a small warp',
  'render/Environment.js|mat3(modelMatrix) * normal': 'normal attribute',
  'render/Environment.js|vDir': 'interpolated sky direction',
  'render/Environment.js|vNormal': 'interpolated normal (sphere)',
  'render/Environment.js|vWorld - cameraPosition': 'view vector (zero only at the camera)',
  'render/Explosion.js|normalMatrix * normal': 'normal attribute',
  'render/Explosion.js|position': 'sphere vertex, never the centre',
  'render/Explosion.js|vLocal': 'interpolated sphere position',
  'render/Explosion.js|vNormal': 'interpolated normal (sphere)',
  'render/Explosion.js|vView': 'view vector (zero only at the camera)',
  'render/Heat.js|normalMatrix * normal': 'normal attribute',
  'render/Heat.js|vN': 'interpolated normal (zero only between opposite vertex normals)',
  'render/Heat.js|vV': 'view vector (zero only at the camera)',
  'render/HullWear.js|vWearNor': 'interpolated normal (flat-shaded hull parts)',
  'render/PostFX.js|p + 1e-5': 'flare: screen offset from the optical centre (zero on one exact texel)',
  'render/PostFX.js|s + 1e-5': 'flare: sun offset from the optical centre (zero on one exact texel)',
  'render/Rcs.js|mat3(viewMatrix) * (mat3(m) * vec3(0.0, 0.0, 1.0))': 'rotated unit axis',
  'render/Rcs.js|mat3(viewMatrix) * (nm * normal)': 'normal attribute',
  'render/Rcs.js|mv.xyz': 'view vector (zero only at the camera)',
  'render/Rcs.js|vNormal': 'interpolated normal (zero only between opposite vertex normals)',
  'render/Rcs.js|vViewDir': 'view vector (zero only at the camera)',
  'render/Torch.js|vPos - vCam': 'view ray (zero only at the camera)',
  'render/Tracks.js|aTangent': 'zero for coincident samples: NaN side vector falls back (length test)',
  'render/Tracks.js|cameraPosition - wp': 'view vector (zero only at the camera)',
  'render/Tracks.js|normalMatrix * normal': 'normal attribute',
  'render/Tracks.js|vN': 'interpolated normal (zero only between opposite vertex normals)',
  'render/Tracks.js|vV': 'view vector (zero only at the camera)',
  'render/Trails.js|aTangent': 'zero for coincident samples: NaN side vector falls back (length test)',
  'render/Trails.js|cameraPosition - wp': 'view vector (zero only at the camera)',
  'render/WeaponBatches.js|aAxis': 'flash axis: a normalised firing direction',
  'render/WeaponBatches.js|d': 'ribbon: only when length(d) > 1e-7 (ternary)',
  'render/WeaponBatches.js|ax.xy': 'torpedo marker: only when length(ax.xy) > 1e-6 (ternary)',
  'render/WeaponBatches.js|cross(n, abs(n.y) < 0.9 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0))': 'unit n crossed with an axis it is not parallel to'
}));

const SRC = new URL('../src/', import.meta.url);
function sourceFiles(dir = SRC, rel = '') {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) out.push(...sourceFiles(new URL(`${e.name}/`, dir), `${rel}${e.name}/`));
    else if (e.name.endsWith('.js')) out.push({ url: new URL(e.name, dir), name: `${rel}${e.name}` });
  }
  return out;
}

/** Template-literal bodies (${...} blanked) with their offsets; JS comments and strings skipped. */
function templates(src) {
  const out = [];
  for (let i = 0; i < src.length;) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { i = src.indexOf('*/', i + 2) + 2; continue; }
    if (c === "'" || c === '"') { i++; while (i < src.length && src[i] !== c) { if (src[i] === '\\') i++; i++; } i++; continue; }
    if (c !== '`') { i++; continue; }
    let j = i + 1, body = '';
    const start = j;
    while (j < src.length && src[j] !== '`') {
      if (src[j] === '\\') { body += '  '; j += 2; continue; }
      if (src[j] === '$' && src[j + 1] === '{') {
        let d = 1, k = j + 2;
        while (d && k < src.length) { if (src[k] === '{') d++; else if (src[k] === '}') d--; k++; }
        body += ' '.repeat(k - j); j = k; continue;
      }
      body += src[j++];
    }
    out.push({ body, start });
    i = j + 1;
  }
  return out;
}

/** Every pow/sqrt/inversesqrt/log/log2/normalize call in GLSL: {fn, arg (first), line}. */
export function glslCalls(src) {
  const calls = [];
  for (const t of templates(src)) {
    if (!/\b(gl_|uniform |varying |vec[234]\b)/.test(t.body)) continue;
    const body = t.body.replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length));
    const re = /(^|[^A-Za-z0-9_.])(pow|sqrt|inversesqrt|log|log2|normalize)\(/g;
    let m;
    while ((m = re.exec(body))) {
      let k = m.index + m[0].length, d = 0, arg = '';
      for (; k < body.length; k++) {
        const ch = body[k];
        if (ch === '(') d++;
        else if (ch === ')') { if (!d) break; d--; } else if (ch === ',' && !d && m[2] === 'pow') break;
        arg += ch;
      }
      const line = src.slice(0, t.start + m.index).split('\n').length;
      calls.push({ fn: m[2], arg: arg.replace(/\s+/g, ' ').trim(), line });
    }
  }
  return calls;
}

/** True when a guard function spans the whole expression: max(...), abs(...) etc. */
function wrapped(arg, names) {
  const m = /^([A-Za-z_][A-Za-z0-9_]*)\(/.exec(arg);
  if (!m || !names.includes(m[1])) return false;
  let d = 0;
  for (let i = m[0].length - 1; i < arg.length; i++) {
    if (arg[i] === '(') d++;
    else if (arg[i] === ')' && --d === 0) return i === arg.length - 1;
  }
  return false;
}
const NONNEG = ['max', 'clamp', 'abs', 'exp', 'exp2', 'sqrt', 'length', 'fract', 'smoothstep', 'step', 'dot2'];
export function provablyNonNegative(arg) {
  if (/^\d+(\.\d*)?$/.test(arg)) return true;                                   // literal
  if (wrapped(arg, NONNEG)) return true;
  if (/^0\.5\s*\+\s*0\.5\s*\*\s*(sin|cos)\(/.test(arg) && wrapped(arg.replace(/^0\.5\s*\+\s*0\.5\s*\*\s*/, ''), ['sin', 'cos'])) return true;
  const dv = /^dot\((.+), (.+)\)$/.exec(arg);                                  // dot(v, v)
  return !!(dv && dv[1] === dv[2]);
}

test('viewer GLSL: no pow/sqrt/log of a value that can go negative, every normalize() reviewed', () => {
  const problems = [], seen = new Set(), seenN = new Set();
  let files = 0;
  for (const f of sourceFiles()) {
    const calls = glslCalls(readFileSync(f.url, 'utf8'));
    if (calls.length) files++;
    for (const c of calls) {
      if (c.fn === 'normalize') {
        const key = `${f.name}|${c.arg}`;
        seenN.add(key);
        if (!NORMALIZE.has(key)) problems.push(`${f.name}:${c.line} normalize(${c.arg}): can it be zero? Guard it or list it in NORMALIZE`);
        continue;
      }
      const key = `${f.name}|${c.fn}|${c.arg}`;
      if (provablyNonNegative(c.arg)) continue;
      seen.add(key);
      if (!REVIEWED.has(key)) problems.push(`${f.name}:${c.line} ${c.fn}(${c.arg}${c.fn === 'pow' ? ', ...' : ''}): can it go negative? Clamp it, write squares as x * x, or list it in REVIEWED`);
    }
  }
  assert.deepEqual(problems, []);
  assert.ok(files >= 12, `scanned ${files} files with GLSL math`);
  // stale entries would hide a later site with the same text
  for (const k of REVIEWED.keys()) assert.ok(seen.has(k), `REVIEWED entry no longer in the code: ${k}`);
  for (const k of NORMALIZE.keys()) assert.ok(seenN.has(k), `NORMALIZE entry no longer in the code: ${k}`);
});

test('the GLSL lint parses calls and guards the way it claims', () => {
  const src = 'const s = /* glsl */`\n  uniform float u;\n  void main() { float x = pow(1.0 - f, 1.4) + pow(max(a, 0.0), 2.0) + sqrt(dot(v, v)) + log2(abs(q) - 1.0); vec3 n = normalize(cross(a, b)); }`;';
  const calls = glslCalls(src);
  assert.deepEqual(calls.map(c => [c.fn, c.arg]), [['pow', '1.0 - f'], ['pow', 'max(a, 0.0)'], ['sqrt', 'dot(v, v)'], ['log2', 'abs(q) - 1.0'], ['normalize', 'cross(a, b)']]);
  assert.deepEqual(calls.map(c => provablyNonNegative(c.arg)), [false, true, true, false, false]);
  assert.equal(provablyNonNegative('0.5 + 0.5 * sin(x * 3.0)'), true);
  assert.equal(provablyNonNegative('0.5 + 0.5 * sin(x) - 1.0'), false);
  assert.equal(provablyNonNegative('max(a, 0.0) - 1.0'), false);
  // JS outside template literals is not GLSL
  assert.deepEqual(glslCalls('const n = normalize(v); const k = `<div>${normalize(x)}</div>`;'), []);
});

test('NaN/Inf guard where the scene colour enters the bloom chain and the flare meter', () => {
  const guard = 'all(lessThan(abs(c), vec3(1.0e30))) ? c : vec3(0.0)';
  const bloom = readFileSync(new URL('../src/render/Bloom.js', import.meta.url), 'utf8');
  const tap = /vec3 tap\(vec2 o\) \{[^}]*\}/.exec(bloom.slice(bloom.indexOf('DOWN_FRAG')))?.[0] || '';
  assert.ok(tap.includes('texture2D(tSrc') && tap.includes(guard), 'every downsample tap (prefilter = scene input) is guarded');
  const post = readFileSync(new URL('../src/render/PostFX.js', import.meta.url), 'utf8');
  const vis = post.slice(post.indexOf('const VIS_FRAG'), post.indexOf('const GOD_FRAG'));
  assert.ok(vis.includes(`c = ${guard}`), 'sun visibility taps of the scene are guarded');
});
