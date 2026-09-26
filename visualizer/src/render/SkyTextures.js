import * as THREE from 'three';

/**
 * Procedural data for the space backdrop: star catalogue, black-body star
 * colours, the gas giant's banded albedo and ring density, and the GLSL
 * noise used to bake the Milky Way into a cube map. Self-contained (does
 * not share Textures.js) so the sky can evolve independently of the hulls.
 */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp01 = (x) => Math.max(0, Math.min(1, x));

/**
 * Approximate black-body colour for a temperature in kelvin (Tanner
 * Helland's fit), returned as LINEAR rgb normalised so max channel = 1.
 */
export function kelvinToRGB(kelvin) {
  const t = Math.max(1000, Math.min(40000, kelvin)) / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    b = 255;
  }
  // sRGB-ish 0..255 -> linear 0..1
  const lin = (c) => {
    const s = clamp01(c / 255);
    return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  const out = [lin(r), lin(g), lin(b)];
  const m = Math.max(out[0], out[1], out[2]) || 1;
  return out.map((c) => c / m);
}

/**
 * Deterministic star catalogue, BRIGHTEST FIRST (so a draw range keeps the
 * prominent stars on low quality). Directions are biased toward the
 * galactic band; brightness follows a steep power law (many faint, few
 * bright); temperatures follow a rough main-sequence mix with hot blue and
 * cool orange giants over-represented among the bright stars.
 *
 * Returns { dir: Float32Array(n*3), color: Float32Array(n*3),
 *           size: Float32Array(n), spike: Float32Array(n) }.
 */
export function makeStarCatalogue(count, { seed = 2024, bandNormal = [0, 1, 0] } = {}) {
  const rnd = mulberry32(seed);
  const bn = new THREE.Vector3(...bandNormal).normalize();
  const stars = [];
  const v = new THREE.Vector3();
  while (stars.length < count) {
    const z = 2 * rnd() - 1;
    const th = rnd() * Math.PI * 2;
    const s = Math.sqrt(1 - z * z);
    v.set(s * Math.cos(th), s * Math.sin(th), z);
    const lat = v.dot(bn);
    // ~40% isotropic, the rest concentrated toward the band
    if (rnd() > 0.4 + 0.6 * Math.exp(-lat * lat * 14)) continue;
    const m = Math.pow(rnd(), 16);                 // 0 faint .. 1 brightest (heavy tail)
    const hot = rnd();
    let T;
    if (m > 0.2 && hot < 0.35) T = 3200 + rnd() * 1500;        // bright K/M giants
    else if (m > 0.2 && hot < 0.75) T = 9000 + rnd() * 16000;  // bright B/A
    else if (hot < 0.3) T = 3000 + rnd() * 1800;               // faint M/K dwarfs
    else if (hot < 0.62) T = 4800 + rnd() * 1600;              // G/K
    else if (hot < 0.88) T = 6200 + rnd() * 2500;              // F/A
    else T = 9000 + rnd() * 14000;                             // B
    stars.push({ x: v.x, y: v.y, z: v.z, m, T, spin: rnd() });
  }
  stars.sort((a, b) => b.m - a.m);
  const dir = new Float32Array(count * 3);
  const color = new Float32Array(count * 3);
  const size = new Float32Array(count);
  const spike = new Float32Array(count);
  const SPIKED = Math.min(28, Math.round(count * 0.003));
  stars.forEach((st, i) => {
    dir[i * 3] = st.x; dir[i * 3 + 1] = st.y; dir[i * 3 + 2] = st.z;
    const c = kelvinToRGB(st.T);
    // most stars barely visible, a few hundred clear, a couple dozen bright
    const top = i < SPIKED ? 1.6 : 1;
    const bright = (0.03 + 2.2 * st.m) * top;
    // keep colour visible but not cartoonish: 70% black-body, 30% white
    for (let k = 0; k < 3; k++) color[i * 3 + k] = (c[k] * 0.7 + 0.3) * bright;
    size[i] = 2.4 + 3.0 * Math.sqrt(st.m);
    spike[i] = i < SPIKED ? 1 - 0.6 * (i / SPIKED) : 0;
  });
  return { dir, color, size, spike };
}

// ---------------------------------------------------------------------------
// Gas giant
// ---------------------------------------------------------------------------
function valueNoise1D(rnd, cells) {
  const lat = Array.from({ length: cells + 1 }, () => rnd());
  lat[cells] = lat[0];
  return (u) => {
    const f = (((u % 1) + 1) % 1) * cells;
    const i = Math.floor(f);
    const t = f - i;
    const s = t * t * (3 - 2 * t);
    return lat[i] + (lat[i + 1] - lat[i]) * s;
  };
}

function periodicNoise2D(rnd, cx, cy) {
  const lat = new Float32Array(cx * cy);
  for (let i = 0; i < lat.length; i++) lat[i] = rnd();
  return (u, v) => {
    const fx = (((u % 1) + 1) % 1) * cx, fy = Math.max(0, Math.min(0.9999, v)) * (cy - 1);
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const tx = fx - x0, ty = fy - y0;
    const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    const x1 = (x0 + 1) % cx, y1 = Math.min(cy - 1, y0 + 1);
    const a = lat[y0 * cx + x0], b = lat[y0 * cx + x1];
    const c = lat[y1 * cx + x0], d = lat[y1 * cx + x1];
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
  };
}

let _giant = null;
/** Banded gas giant albedo, equirectangular, sRGB. Warm ochre/cream belts. */
export function getGasGiantTexture() {
  if (_giant) return _giant;
  const w = 1024, h = 512;
  const rnd = mulberry32(90210);
  const bandA = valueNoise1D(rnd, 23);
  const bandB = valueNoise1D(rnd, 57);
  const octs = [
    periodicNoise2D(rnd, 12, 7), periodicNoise2D(rnd, 24, 13),
    periodicNoise2D(rnd, 48, 25), periodicNoise2D(rnd, 96, 49)
  ];
  const fbm = (u, v) => octs[0](u, v) * 0.5 + octs[1](u, v) * 0.27 + octs[2](u, v) * 0.15 + octs[3](u, v) * 0.08;
  const palette = [
    [0.00, [120, 96, 78]], [0.18, [196, 168, 130]], [0.30, [226, 210, 180]],
    [0.42, [170, 126, 92]], [0.55, [214, 190, 150]], [0.66, [150, 110, 82]],
    [0.78, [232, 218, 192]], [0.90, [184, 146, 108]], [1.00, [120, 96, 78]]
  ];
  const ramp = (t) => {
    t = clamp01(t);
    for (let i = 1; i < palette.length; i++) {
      if (t <= palette[i][0]) {
        const a = palette[i - 1], b = palette[i];
        const f = (t - a[0]) / (b[0] - a[0]);
        return [0, 1, 2].map((k) => a[1][k] + (b[1][k] - a[1][k]) * f);
      }
    }
    return palette[palette.length - 1][1];
  };
  // Great storm: an oval in the southern belt
  const storm = { u: 0.31, v: 0.62, ru: 0.05, rv: 0.035 };
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    const lat = (v - 0.5) * Math.PI;
    for (let x = 0; x < w; x++) {
      const u = x / w;
      // Zonal shear: turbulence stretched along longitude
      const n = fbm(u * 1.0, v);
      const warp = (n - 0.5) * 0.05 + (octs[2](u * 2, v) - 0.5) * 0.02 * Math.cos(lat * 9);
      let vv = v + warp;
      // storm swirl
      const du = (u - storm.u) / storm.ru, dv = (v - storm.v) / storm.rv;
      const sr = Math.sqrt(du * du + dv * dv);
      let stormMix = 0;
      if (sr < 1.6) {
        const sw = Math.exp(-sr * sr * 1.6);
        const ang = Math.atan2(dv, du) + sw * 3.0;
        vv = storm.v + Math.sin(ang) * sr * storm.rv + (vv - v) * (1 - sw);
        stormMix = Math.exp(-sr * sr * 2.2);
      }
      const band = bandA(vv * 1.0) * 0.65 + bandB(vv * 1.0) * 0.35;
      let c = ramp(band * 0.9 + (n - 0.5) * 0.18 + 0.05);
      // polar darkening + slight blue-grey hood
      const pole = Math.pow(Math.abs(Math.sin(lat)), 6);
      c = [c[0] * (1 - 0.45 * pole), c[1] * (1 - 0.4 * pole), c[2] * (1 - 0.3 * pole)];
      if (stormMix > 0) {
        const sc = [200, 118, 84];
        c = c.map((k, i) => k + (sc[i] - k) * stormMix * 0.85);
      }
      const fine = 0.92 + 0.16 * octs[3](u * 3, v);
      const i = (y * w + x) * 4;
      img.data[i] = Math.min(255, c[0] * fine);
      img.data[i + 1] = Math.min(255, c[1] * fine);
      img.data[i + 2] = Math.min(255, c[2] * fine);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  _giant = new THREE.CanvasTexture(canvas);
  _giant.colorSpace = THREE.SRGBColorSpace;
  _giant.wrapS = THREE.RepeatWrapping;
  _giant.wrapT = THREE.ClampToEdgeWrapping;
  _giant.anisotropy = 4;
  return _giant;
}

/** Radial ring density (u = 0 inner edge .. 1 outer edge), sRGB colour + alpha. */
export function makeRingTexture() {
  const w = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = 4;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, 4);
  const rnd = mulberry32(77);
  const bands = [];
  for (let i = 0; i < 70; i++) bands.push({ c: rnd(), w: 0.002 + rnd() * 0.02, a: (rnd() - 0.45) * 0.7 });
  const gapC = 0.62; // a Cassini-like division
  for (let x = 0; x < w; x++) {
    const u = x / w;
    let a = 0.35 + 0.4 * Math.sin(u * Math.PI * 0.9 + 0.2);
    for (const b of bands) a += b.a * Math.exp(-Math.pow((u - b.c) / b.w, 2));
    a *= 1 - 0.92 * Math.exp(-Math.pow((u - gapC) / 0.018, 2));
    a *= smooth(0, 0.04, u) * smooth(1, 0.93, u);
    a = clamp01(a);
    const tone = 0.82 + 0.18 * Math.sin(u * 31 + 1.0);
    const warm = 1 - u * 0.25;
    for (let y = 0; y < 4; y++) {
      const i = (y * w + x) * 4;
      img.data[i] = 212 * tone; img.data[i + 1] = 192 * tone * (0.94 + 0.06 * warm); img.data[i + 2] = 166 * tone * warm;
      img.data[i + 3] = a * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

function smooth(a, b, x) {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// Tileable 3D gradient noise volume for the sky bake. Sampling a texture
// instead of evaluating simplex noise in-shader keeps the bake shader tiny:
// SwiftShader spent ~8 s just compiling ~100 inlined snoise() calls.
// ---------------------------------------------------------------------------
/**
 * Periodic Perlin gradient noise: `cells` lattice cells per axis, sampled
 * at `size` voxels per axis; four independent channels (RGBA), each in
 * 0..1 with 0.5 = zero. Pure function (no DOM), returns a Uint8Array.
 */
export function makeNoiseVolume(size = 64, cells = 8, seed = 7) {
  const rnd = mulberry32(seed);
  const n = cells;
  const grads = [];
  for (let ch = 0; ch < 4; ch++) {
    const g = new Float32Array(n * n * n * 3);
    for (let i = 0; i < n * n * n; i++) {
      // uniform direction
      const z = 2 * rnd() - 1, t = rnd() * Math.PI * 2, r = Math.sqrt(1 - z * z);
      g[i * 3] = r * Math.cos(t); g[i * 3 + 1] = r * Math.sin(t); g[i * 3 + 2] = z;
    }
    grads.push(g);
  }
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const out = new Uint8Array(size * size * size * 4);
  const scale = n / size;
  for (let z = 0; z < size; z++) {
    const fz = z * scale, z0 = Math.floor(fz), tz = fz - z0, wz = fade(tz);
    for (let y = 0; y < size; y++) {
      const fy = y * scale, y0 = Math.floor(fy), ty = fy - y0, wy = fade(ty);
      for (let x = 0; x < size; x++) {
        const fx = x * scale, x0 = Math.floor(fx), tx = fx - x0, wx = fade(tx);
        for (let ch = 0; ch < 4; ch++) {
          const g = grads[ch];
          let acc = 0;
          for (let c = 0; c < 8; c++) {
            const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
            const gi = (((z0 + dz) % n) * n * n + ((y0 + dy) % n) * n + ((x0 + dx) % n)) * 3;
            const dot = g[gi] * (tx - dx) + g[gi + 1] * (ty - dy) + g[gi + 2] * (tz - dz);
            const w = (dx ? wx : 1 - wx) * (dy ? wy : 1 - wy) * (dz ? wz : 1 - wz);
            acc += dot * w;
          }
          // gradient noise lies in about [-0.87, 0.87]
          out[((z * size + y) * size + x) * 4 + ch] = Math.max(0, Math.min(255, Math.round((acc * 0.62 + 0.5) * 255)));
        }
      }
    }
  }
  return out;
}

export function makeNoiseVolumeTexture(size = 64, cells = 8) {
  const tex = new THREE.Data3DTexture(makeNoiseVolume(size, cells), size, size, size);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// GLSL helpers for the bake. One unit of p = one lattice cell; the volume
// repeats every 8 units, and every octave is rotated + offset so the
// period never lines up visibly.
export const GLSL_NOISE = /* glsl */`
  uniform highp sampler3D uNoiseVol;
  vec4 nz4(vec3 p) { return texture(uNoiseVol, p * 0.125) * 2.0 - 1.0; }
  float nz(vec3 p) { return nz4(p).x; }
  const mat3 OCT_ROT = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
  // fbm in 0..1 (channel picks an independent field)
  float fbm(vec3 p, int oct) {
    float a = 0.5, s = 0.0, n = 0.0;
    for (int i = 0; i < 7; i++) {
      if (i >= oct) break;
      s += a * nz(p);
      n += a;
      p = OCT_ROT * p * 2.03 + vec3(1.7, -3.1, 2.3);
      a *= 0.5;
    }
    return clamp(0.5 + 0.9 * s / n, 0.0, 1.0);
  }
  // ridged fbm in 0..1 - thin filaments (dust lanes)
  float ridged(vec3 p, int oct) {
    float a = 0.5, s = 0.0, n = 0.0;
    for (int i = 0; i < 7; i++) {
      if (i >= oct) break;
      float r = 1.0 - abs(nz4(p).y) * 1.4;
      r = max(r, 0.0);
      s += a * r * r;
      n += a;
      p = OCT_ROT * p * 2.07 + vec3(-2.3, 1.9, 0.7);
      a *= 0.5;
    }
    return s / n;
  }
  vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
  }
`;
