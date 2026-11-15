import * as THREE from 'three';

/**
 * Procedural textures - everything the renderer needs is generated on a
 * 2D canvas at startup, so the viewer stays a single static bundle with no
 * asset downloads. Hull plating (albedo / normal / roughness-metalness),
 * tileable RGBA noise for the shaders, and the sprite alphas used by
 * particles, plumes and blasts.
 */

// ---------------------------------------------------------------------------
// Small deterministic PRNG so every ship of a faction shares the same plating
// ---------------------------------------------------------------------------
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

/** Periodic value noise on a lattice - tiles perfectly at `size`. */
function makeValueNoise(size, cells, rnd) {
  const lattice = new Float32Array(cells * cells);
  for (let i = 0; i < lattice.length; i++) lattice[i] = rnd();
  const out = new Float32Array(size * size);
  const scale = cells / size;
  const fade = (t) => t * t * (3 - 2 * t);
  for (let y = 0; y < size; y++) {
    const fy = y * scale;
    const y0 = Math.floor(fy);
    const ty = fade(fy - y0);
    const y1 = (y0 + 1) % cells;
    for (let x = 0; x < size; x++) {
      const fx = x * scale;
      const x0 = Math.floor(fx);
      const tx = fade(fx - x0);
      const x1 = (x0 + 1) % cells;
      const a = lattice[y0 * cells + x0];
      const b = lattice[y0 * cells + x1];
      const c = lattice[y1 * cells + x0];
      const d = lattice[y1 * cells + x1];
      out[y * size + x] = (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
    }
  }
  return out;
}

/** Tileable fbm: sum of periodic value noise octaves, normalised to 0..1. */
function makeFbm(size, baseCells, octaves, rnd, gain = 0.5) {
  const out = new Float32Array(size * size);
  let amp = 1, total = 0, cells = baseCells;
  for (let o = 0; o < octaves; o++) {
    const n = makeValueNoise(size, cells, rnd);
    for (let i = 0; i < out.length; i++) out[i] += n[i] * amp;
    total += amp;
    amp *= gain;
    cells *= 2;
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

function canvasOf(size, h = size) {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = h;
  return c;
}

function texOf(canvas, { srgb = false, repeat = true, anisotropy = 8 } = {}) {
  const t = new THREE.CanvasTexture(canvas);
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = anisotropy;
  t.needsUpdate = true;
  return t;
}

// ---------------------------------------------------------------------------
// Shader noise: RGBA, four independent tileable fbm fields
// ---------------------------------------------------------------------------
let _noiseTex = null;
export function getNoiseTexture() {
  if (_noiseTex) return _noiseTex;
  const size = 256;
  const rnd = mulberry32(1337);
  const fields = [
    makeFbm(size, 4, 5, rnd),
    makeFbm(size, 6, 5, rnd),
    makeFbm(size, 3, 6, rnd, 0.55),
    makeFbm(size, 8, 4, rnd)
  ];
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    img.data[i * 4] = fields[0][i] * 255;
    img.data[i * 4 + 1] = fields[1][i] * 255;
    img.data[i * 4 + 2] = fields[2][i] * 255;
    img.data[i * 4 + 3] = fields[3][i] * 255;
  }
  ctx.putImageData(img, 0, 0);
  _noiseTex = texOf(canvas, { anisotropy: 1 });
  _noiseTex.minFilter = THREE.LinearMipmapLinearFilter;
  _noiseTex.magFilter = THREE.LinearFilter;
  return _noiseTex;
}

// ---------------------------------------------------------------------------
// Sprite alphas
// ---------------------------------------------------------------------------
let _glow = null;
/** Soft radial glow - general purpose. */
export function getGlowTexture() {
  if (_glow) return _glow;
  const size = 128;
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.12, 'rgba(255,255,255,0.7)');
  grad.addColorStop(0.3, 'rgba(255,255,255,0.2)');
  grad.addColorStop(0.5, 'rgba(255,255,255,0.04)');
  grad.addColorStop(0.68, 'rgba(255,255,255,0)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  _glow = texOf(canvas, { repeat: false });
  return _glow;
}

let _star = null;
/** Tight star / spark dot with a faint cross flare. */
export function getStarTexture() {
  if (_star) return _star;
  const size = 64;
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.12, 'rgba(255,255,255,0.9)');
  grad.addColorStop(0.3, 'rgba(255,255,255,0.25)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  ctx.globalCompositeOperation = 'lighter';
  const flare = ctx.createLinearGradient(0, size / 2, size, size / 2);
  flare.addColorStop(0, 'rgba(255,255,255,0)');
  flare.addColorStop(0.5, 'rgba(255,255,255,0.35)');
  flare.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = flare;
  ctx.fillRect(0, size / 2 - 1, size, 2);
  const flare2 = ctx.createLinearGradient(size / 2, 0, size / 2, size);
  flare2.addColorStop(0, 'rgba(255,255,255,0)');
  flare2.addColorStop(0.5, 'rgba(255,255,255,0.35)');
  flare2.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = flare2;
  ctx.fillRect(size / 2 - 1, 0, 2, size);
  _star = texOf(canvas, { repeat: false });
  return _star;
}

let _streak = null;
/** Horizontal streak: bright head on the right, tapering tail to the left. */
export function getStreakTexture() {
  if (_streak) return _streak;
  const w = 128, h = 32;
  const canvas = canvasOf(w, h);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const dy = (y + 0.5) / h * 2 - 1;
    const across = Math.max(0, 1 - dy * dy);
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      const along = Math.pow(u, 2.2);            // tail fades toward u=0
      const head = Math.exp(-Math.pow((u - 0.92) / 0.06, 2)); // hot head
      const a = Math.min(1, along * across * across + head * across);
      const i = (y * w + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
      img.data[i + 3] = a * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  _streak = texOf(canvas, { repeat: false });
  return _streak;
}

const _puffs = [];
/** Wispy gas-cloud puff with fbm structure; index selects a variant. */
export function getPuffTexture(index = 0) {
  if (_puffs[index]) return _puffs[index];
  const size = 256;
  const rnd = mulberry32(900 + index * 17);
  const fbm = makeFbm(size, 3, 5, rnd, 0.55);
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x + 0.5) / size * 2 - 1;
      const dy = (y + 0.5) / size * 2 - 1;
      const r = Math.sqrt(dx * dx + dy * dy);
      const n = fbm[y * size + x];
      // radial falloff shaped by the noise so the edge is ragged, not round
      const edge = 0.55 + 0.45 * n;
      const a = Math.max(0, 1 - Math.pow(r / edge, 2.2)) * (0.35 + 0.65 * n);
      const i = (y * size + x) * 4;
      const v = 200 + 55 * n;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = Math.min(255, a * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  _puffs[index] = texOf(canvas, { repeat: false });
  return _puffs[index];
}

let _ring = null;
/** Thin luminous ring with soft inner and outer edges. */
export function getRingTexture() {
  if (_ring) return _ring;
  const size = 256;
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x + 0.5) / size * 2 - 1;
      const dy = (y + 0.5) / size * 2 - 1;
      const r = Math.sqrt(dx * dx + dy * dy);
      const a = Math.exp(-Math.pow((r - 0.82) / 0.07, 2)) + 0.25 * Math.exp(-Math.pow((r - 0.7) / 0.18, 2));
      const i = (y * size + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
      img.data[i + 3] = Math.min(255, a * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  _ring = texOf(canvas, { repeat: false });
  return _ring;
}

// ---------------------------------------------------------------------------
// Hull plating - albedo, normal, roughness(G)/metalness(B)
// ---------------------------------------------------------------------------

/**
 * Recursive panel subdivision of the tile. Returns rects {x,y,w,h}.
 * Panels are cut so the tile edges land on seams and the plating tiles.
 */
function subdividePanels(size, rnd, minSize, maxSize) {
  const out = [];
  const stack = [{ x: 0, y: 0, w: size, h: size }];
  while (stack.length) {
    const r = stack.pop();
    const big = Math.max(r.w, r.h);
    const stop = big <= maxSize && (big <= minSize * 1.6 || rnd() < 0.35);
    if (stop) {
      out.push(r);
      continue;
    }
    const vertical = r.w > r.h ? rnd() < 0.72 : rnd() < 0.28;
    const frac = 0.3 + rnd() * 0.4;
    if (vertical) {
      const cut = Math.round(r.w * frac / 8) * 8;
      if (cut < minSize || r.w - cut < minSize) { out.push(r); continue; }
      stack.push({ x: r.x, y: r.y, w: cut, h: r.h });
      stack.push({ x: r.x + cut, y: r.y, w: r.w - cut, h: r.h });
    } else {
      const cut = Math.round(r.h * frac / 8) * 8;
      if (cut < minSize || r.h - cut < minSize) { out.push(r); continue; }
      stack.push({ x: r.x, y: r.y, w: r.w, h: cut });
      stack.push({ x: r.x, y: r.y + cut, w: r.w, h: r.h - cut });
    }
  }
  return out;
}

/** Sobel a height field (0..1) into a tangent-space normal map canvas. */
function heightToNormalCanvas(height, size, strength) {
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const at = (x, y) => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1))
        - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const dy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1))
        - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      let nx = -dx * strength, ny = -dy * strength, nz = 1;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len; ny /= len; nz /= len;
      const i = (y * size + x) * 4;
      img.data[i] = (nx * 0.5 + 0.5) * 255;
      img.data[i + 1] = (ny * 0.5 + 0.5) * 255;
      img.data[i + 2] = (nz * 0.5 + 0.5) * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** One soft, irregular paint flake (anti-aliased canvas path). */
function chipFlake(ctx, x, y, r, rnd, col, alpha) {
  const n = 7;
  ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha})`;
  ctx.beginPath();
  for (let k = 0; k < n; k++) {
    const a = (k / n) * Math.PI * 2 + rnd() * 0.5;
    const rr = r * (0.55 + 0.6 * rnd());
    if (k === 0) ctx.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr * 0.75);
    else ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr * 0.75);
  }
  ctx.closePath();
  ctx.fill();
}

const _hullSets = new Map();

/** Faction livery colours (sRGB 0-255), shared with the hull markings. */
export const FACTION_PAINT = {
  alpha: [52, 118, 168],
  beta: [192, 90, 44]
};

/**
 * Faction hull plating set. `variant` is 'hull' (painted armour plating),
 * 'trim' (dark mechanical structure: dense small panels, grilles, pipes)
 * or 'paint' (faction livery plates: big panels, chipped at the edges).
 * Returns { map, normalMap, ormMap } textures, all tileable.
 */
/** Hull plating base colours (sRGB 0-255): cool naval grey vs warm sand. */
export const FACTION_PLATE = {
  alpha: [106, 115, 130],
  beta: [126, 110, 94]
};

/**
 * Armour plate set ('hull' painted armour, 'paint' faction livery), built
 * as a hierarchy rather than one grid of tiles:
 *   1. four big armour slabs per tile. Their joints are RELIEF - a wide
 *      bevelled channel in the normal map - with only a thin, low-contrast
 *      line in the albedo, so a hull reads as a few slabs under the key
 *      light and as clean plate at fleet distance
 *   2. panel lines inside each slab at clearly lower contrast
 *   3. fine detail that only resolves close up: bolt rows, small hatches,
 *      vents, stencil labels, worn edges, chips, faint streaks
 * The kit projects these UVs in hull space (HullKit._projectUVs) at one
 * scale per ship, larger on larger hulls. Painted plate is a dielectric
 * (metalness ~0.08); wear is part metal (fully metallic texels mirror the
 * black sky and render dark).
 */
function plateTextures(faction, variant) {
  const isPaint = variant === 'paint';
  const size = isPaint ? 512 : 1024;
  const S = size / 1024;
  const px = (v) => Math.max(1, Math.round(v * S));
  const rnd = mulberry32((faction === 'alpha' ? 4242 : 7777) + (isPaint ? 31 : 0));
  const livery = FACTION_PAINT[faction] || FACTION_PAINT.alpha;
  const base = isPaint ? livery.map(c => c * 0.86) : (FACTION_PLATE[faction] || FACTION_PLATE.alpha);
  const accent = faction === 'alpha' ? [40, 200, 255] : [255, 130, 50];
  const primer = faction === 'alpha' ? [52, 58, 66] : [66, 58, 54];
  const bareMetal = [176, 178, 184];
  const rgb = (c, k = 1) => `rgb(${Math.min(255, c[0] * k) | 0},${Math.min(255, c[1] * k) | 0},${Math.min(255, c[2] * k) | 0})`;
  const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  const grime = makeFbm(size, 3, 4, rnd, 0.55);
  const micro = makeFbm(size, 48, 2, rnd, 0.5);
  const albedo = canvasOf(size);
  const actx = albedo.getContext('2d');
  actx.fillStyle = rgb(base, 0.78);               // slab joints: a soft line
  actx.fillRect(0, 0, size, size);

  // Relief is kept shallow: under a glancing sun a deep joint channel or
  // surface grain reads as brickwork (the pass-5 battleship bow)
  const height = new Float32Array(size * size).fill(0.32);
  const rough = new Float32Array(size * size).fill(0.7);
  const metal = new Float32Array(size * size).fill(0.25);
  const fillRect = (field, x, y, w, h, v) => {
    x = Math.round(x); y = Math.round(y); w = Math.round(w); h = Math.round(h);
    for (let yy = y; yy < y + h; yy++) {
      const row = ((yy % size) + size) % size;
      for (let xx = x; xx < x + w; xx++) field[row * size + (((xx % size) + size) % size)] = v;
    }
  };

  // 1. slabs: two unequal courses, each cut once (u runs along the hull)
  const snap = (v) => Math.round(v / 8) * 8;
  const course = snap(size * (0.42 + 0.16 * rnd()));
  const slabs = [];
  for (const [y, h] of [[0, course], [course, size - course]]) {
    const cut = snap(size * (0.34 + 0.32 * rnd()));
    slabs.push({ x: 0, y, w: cut, h }, { x: cut, y, w: size - cut, h });
  }
  const gap = px(1.5), bev = px(7);
  for (const p of slabs) {
    const tone = 1 + (rnd() - 0.5) * 0.07;
    const col = base.map(c => c * tone);
    const pRough = (isPaint ? 0.46 : 0.5) + (rnd() - 0.5) * 0.1;
    const pHeight = 0.56 + rnd() * 0.05;
    const x0 = p.x + gap, y0 = p.y + gap, w = p.w - gap * 2, h = p.h - gap * 2;
    actx.fillStyle = rgb(col);
    actx.fillRect(x0, y0, w, h);
    for (let i = 0; i <= bev; i++) {
      fillRect(height, x0 + i, y0 + i, w - i * 2, h - i * 2, 0.32 + (pHeight - 0.32) * ((i + 1) / (bev + 1)));
    }
    fillRect(rough, x0, y0, w, h, pRough);
    fillRect(metal, x0, y0, w, h, 0.08);

    // 2. panels inside the slab: quiet lines, a little tone per panel
    const m = gap + bev + px(3);
    const ix = p.x + m, iy = p.y + m, iw = p.w - m * 2, ih = p.h - m * 2;
    const xs = [ix], ys = [iy];
    for (let c = 1, n = 2 + Math.floor(rnd() * 2); c < n; c++) xs.push(ix + Math.round(iw * (c + (rnd() - 0.5) * 0.4) / n));
    for (let c = 1, n = 1 + Math.floor(rnd() * 2); c < n; c++) ys.push(iy + Math.round(ih * (c + (rnd() - 0.5) * 0.3) / n));
    xs.push(ix + iw); ys.push(iy + ih);
    const lw = px(1.3);
    for (let a = 0; a < xs.length - 1; a++) {
      for (let b = 0; b < ys.length - 1; b++) {
        const cx0 = xs[a], cy0 = ys[b], cw = xs[a + 1] - xs[a], ch = ys[b + 1] - ys[b];
        const kind = rnd();
        let ccol = col, cRough = pRough;
        if (!isPaint && kind < 0.06) { ccol = col.map((c, i) => c * 0.95 + bareMetal[i] * 0.06); cRough = pRough - 0.03; fillRect(metal, cx0, cy0, cw, ch, 0.15); }
        else if (!isPaint && kind < 0.11) ccol = accent.map((c, i) => c * 0.035 + col[i] * 0.965);
        else if (kind < 0.24) { ccol = col.map(c => c * 0.96); cRough = pRough + 0.02; }
        else if (kind < 0.36) ccol = col.map(c => c * 1.03);
        if (ccol !== col) {
          actx.fillStyle = rgb(ccol);
          actx.fillRect(cx0, cy0, cw, ch);
          fillRect(rough, cx0, cy0, cw, ch, cRough);
        }
        // 3. one small feature on some panels
        const feat = rnd();
        const fx = cx0 + cw * (0.3 + 0.4 * rnd()), fy = cy0 + ch * (0.3 + 0.4 * rnd());
        if (cw < px(70) || ch < px(70)) continue;
        if (feat < 0.1) {                 // vent slats
          const n = 3 + Math.floor(rnd() * 3), pitch = px(6), sw = Math.min(cw * 0.3, px(34));
          for (let i = 0; i < n; i++) {
            actx.fillStyle = 'rgba(0,0,0,0.38)';
            actx.fillRect(fx - sw / 2, fy + (i - n / 2) * pitch, sw, px(2.5));
            fillRect(height, fx - sw / 2, fy + (i - n / 2) * pitch, sw, px(2.5), pHeight - 0.25);
          }
        } else if (feat < 0.2) {          // small round hatch
          const r = Math.min(cw, ch, px(90)) * 0.2;
          actx.strokeStyle = 'rgba(0,0,0,0.36)';
          actx.lineWidth = px(1.6);
          actx.beginPath(); actx.arc(fx, fy, r, 0, Math.PI * 2); actx.stroke();
          for (let yy = Math.floor(fy - r); yy <= fy + r; yy++) {
            for (let xx = Math.floor(fx - r); xx <= fx + r; xx++) {
              const d = Math.hypot(xx - fx, yy - fy);
              if (d < r - px(2)) fillRect(height, xx, yy, 1, 1, pHeight + 0.07);
              else if (d < r) fillRect(height, xx, yy, 1, 1, pHeight - 0.2);
            }
          }
          actx.fillStyle = 'rgba(0,0,0,0.3)';
          for (let b2 = 0; b2 < 6; b2++) {
            const an = b2 / 6 * Math.PI * 2;
            actx.beginPath(); actx.arc(fx + Math.cos(an) * r * 0.74, fy + Math.sin(an) * r * 0.74, px(1.3), 0, Math.PI * 2); actx.fill();
          }
        } else if (feat < 0.3) {          // bolted access panel
          const aw = Math.min(cw * 0.34, px(64)), ah = Math.min(ch * 0.34, px(44));
          actx.strokeStyle = 'rgba(0,0,0,0.3)';
          actx.lineWidth = px(1.4);
          actx.strokeRect(fx - aw / 2, fy - ah / 2, aw, ah);
          fillRect(height, fx - aw / 2, fy - ah / 2, aw, ah, pHeight - 0.16);
          fillRect(height, fx - aw / 2 + px(2), fy - ah / 2 + px(2), aw - px(4), ah - px(4), pHeight + 0.05);
        } else if (!isPaint && feat < 0.33) {   // hazard chevrons (rare)
          const sw = Math.min(cw * 0.36, px(60)), sh = px(8);
          actx.save();
          actx.beginPath(); actx.rect(fx - sw / 2, fy - sh / 2, sw, sh); actx.clip();
          actx.fillStyle = 'rgba(206,170,48,0.8)';
          actx.fillRect(fx - sw / 2, fy - sh / 2, sw, sh);
          actx.fillStyle = 'rgba(20,20,20,0.85)';
          for (let x = fx - sw / 2 - sh; x < fx + sw / 2 + sh; x += sh * 1.4) {
            actx.beginPath();
            actx.moveTo(x, fy + sh / 2); actx.lineTo(x + sh * 0.7, fy + sh / 2);
            actx.lineTo(x + sh * 1.4, fy - sh / 2); actx.lineTo(x + sh * 0.7, fy - sh / 2);
            actx.closePath(); actx.fill();
          }
          actx.restore();
        } else if (feat < 0.5) {          // stencil label
          actx.fillStyle = isPaint ? 'rgba(235,238,242,0.5)' : 'rgba(225,230,236,0.4)';
          for (let l = 0, lines = 1 + Math.floor(rnd() * 2); l < lines; l++) {
            let xx = fx - px(18);
            for (let wd = 0, words = 2 + Math.floor(rnd() * 2); wd < words; wd++) {
              const ww = px(4 + rnd() * 11);
              actx.fillRect(xx, fy + l * px(5), ww, px(2.4));
              xx += ww + px(3);
            }
          }
        }
      }
    }
    // (panel lines are albedo only: relief at panel scale turns a face lit
    // at a glancing angle into brickwork)
    actx.fillStyle = 'rgba(0,0,0,0.11)';
    for (const cx of xs.slice(1, -1)) { actx.fillRect(cx, iy, lw, ih); fillRect(height, cx, iy, lw, ih, pHeight - 0.01); }
    for (const cy of ys.slice(1, -1)) { actx.fillRect(ix, cy, iw, lw); fillRect(height, ix, cy, iw, lw, pHeight - 0.01); }

    // bolt rows along the slab edges (fine)
    const bi = gap + bev + px(4), step = px(28), rr = Math.max(0.9, 1.3 * S);
    actx.fillStyle = 'rgba(0,0,0,0.26)';
    const dot = (bx, by) => {
      actx.beginPath(); actx.arc(bx, by, rr, 0, Math.PI * 2); actx.fill();
      fillRect(height, bx - 1, by - 1, 2, 2, pHeight + 0.07);
    };
    for (let x = p.x + bi; x <= p.x + p.w - bi; x += step) { dot(x, p.y + bi); dot(x, p.y + p.h - bi); }
    for (let y = p.y + bi + step; y <= p.y + p.h - bi - step; y += step) { dot(p.x + bi, y); dot(p.x + p.w - bi, y); }

    // worn slab edges: broken thin runs of brighter metal on the bevel
    const edges = [[x0, y0 + px(1), w, 0], [x0, y0 + h - px(3), w, 0], [x0 + px(1), y0, h, 1], [x0 + w - px(3), y0, h, 1]];
    for (const [ex, ey, len, vert] of edges) {
      let pos = 0;
      while (pos < len) {
        const run = px(8 + rnd() * 60);
        if (rnd() < (isPaint ? 0.42 : 0.3)) {
          const r0 = Math.min(run, len - pos), t = px(1.5 + rnd() * 1.5);
          const rx = vert ? ex : ex + pos, ry = vert ? ey + pos : ey;
          const rw = vert ? t : r0, rh = vert ? r0 : t;
          if (isPaint) { actx.fillStyle = rgba(primer, 0.7); actx.fillRect(rx - 1, ry - 1, rw + 2, rh + 2); }
          actx.fillStyle = rgba(bareMetal, isPaint ? 0.7 : 0.5);
          actx.fillRect(rx, ry, rw, rh);
          fillRect(metal, rx, ry, rw, rh, 0.5);
          fillRect(rough, rx, ry, rw, rh, 0.32);
        }
        pos += run + px(10 + rnd() * 70);
      }
    }
    for (let c = 0, clusters = rnd() < (isPaint ? 0.8 : 0.4) ? 1 + Math.floor(rnd() * 2) : 0; c < clusters; c++) {
      const ex = rnd() < 0.5 ? x0 + px(8) : x0 + w - px(8);
      const ey = rnd() < 0.5 ? y0 + px(8) : y0 + h - px(8);
      for (let i = 0, flakes = 3 + Math.floor(rnd() * 4); i < flakes; i++) {
        const cx = ex + (rnd() - 0.5) * px(30), cy = ey + (rnd() - 0.5) * px(18);
        const r = (1.4 + rnd() * 3) * Math.max(0.7, S);
        chipFlake(actx, cx, cy, r, rnd, primer, 0.7);
        chipFlake(actx, cx, cy, r * 0.55, rnd, bareMetal, 0.7);
        fillRect(metal, cx - r * 0.5, cy - r * 0.5, r, r, 0.5);
        fillRect(rough, cx - r * 0.5, cy - r * 0.5, r, r, 0.34);
      }
    }
  }

  // faint streaks drawn out from the slab joints
  for (let i = 0, n = isPaint ? 8 : 20; i < n; i++) {
    const p = slabs[Math.floor(rnd() * slabs.length)];
    const sx = p.x + rnd() * p.w, sy = p.y + gap, sw = px(3 + rnd() * 9), len = px(60 + rnd() * 200);
    const pale = rnd() < 0.25;
    for (const oy of [0, -size]) {
      const g = actx.createLinearGradient(0, sy + oy, 0, sy + oy + len);
      g.addColorStop(0, pale ? 'rgba(220,220,215,0.07)' : 'rgba(8,7,6,0.13)');
      g.addColorStop(1, pale ? 'rgba(220,220,215,0)' : 'rgba(8,7,6,0)');
      actx.fillStyle = g;
      actx.fillRect(sx, sy + oy, sw, len);
    }
  }

  const img = actx.getImageData(0, 0, size, size);
  for (let i = 0; i < size * size; i++) {
    const g = grime[i], m = micro[i];
    const mod = (0.94 + 0.12 * g) * (0.975 + 0.05 * m);
    img.data[i * 4] = Math.min(255, img.data[i * 4] * mod);
    img.data[i * 4 + 1] = Math.min(255, img.data[i * 4 + 1] * mod);
    img.data[i * 4 + 2] = Math.min(255, img.data[i * 4 + 2] * mod);
    rough[i] = Math.min(1, Math.max(0.2, rough[i] + (g - 0.5) * 0.26 + (m - 0.5) * 0.05));
    height[i] += (m - 0.5) * 0.006;
  }
  actx.putImageData(img, 0, 0);

  const orm = canvasOf(size);
  const octx = orm.getContext('2d');
  const oimg = octx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    oimg.data[i * 4] = 255;
    oimg.data[i * 4 + 1] = Math.max(0, Math.min(255, rough[i] * 255));
    oimg.data[i * 4 + 2] = Math.max(0, Math.min(255, metal[i] * 255));
    oimg.data[i * 4 + 3] = 255;
  }
  octx.putImageData(oimg, 0, 0);
  return {
    map: texOf(albedo, { srgb: true }),
    normalMap: texOf(heightToNormalCanvas(height, size, isPaint ? 2.4 : 2.6)),
    ormMap: texOf(orm)
  };
}

export function getHullTextures(faction, variant = 'hull') {
  const key = `${faction}:${variant}`;
  if (_hullSets.has(key)) return _hullSets.get(key);
  if (variant !== 'trim') {
    const set = plateTextures(faction, variant);
    _hullSets.set(key, set);
    return set;
  }

  const size = variant === 'hull' ? 1024 : 512;
  const rnd = mulberry32(faction === 'alpha' ? 4242 : 7777 + (variant === 'trim' ? 99 : 0));
  const isTrim = variant === 'trim';
  const isPaint = variant === 'paint';

  // Base palette. Plating is a touch lighter than bare steel so the lit
  // side reads as painted armour, not a black silhouette.
  const paint = FACTION_PAINT[faction] || FACTION_PAINT.alpha;
  const base = isPaint ? paint.map(c => c * 0.82)
    : faction === 'alpha'
      ? (isTrim ? [58, 64, 74] : [98, 107, 120])
      : (isTrim ? [66, 60, 55] : [114, 104, 97]);
  const accent = faction === 'alpha' ? [40, 200, 255] : [255, 130, 50];
  // primer showing through chipped livery: dark, desaturated
  const primer = faction === 'alpha' ? [52, 58, 66] : [66, 58, 54];

  const panels = isPaint
    ? subdividePanels(size, rnd, 64, 300)
    : subdividePanels(size, rnd, isTrim ? 56 : 56, isTrim ? 200 : 260);
  const grime = makeFbm(size, 4, 5, rnd, 0.6);
  const micro = makeFbm(size, 32, 3, rnd, 0.5);

  // --- albedo ------------------------------------------------------------
  const albedo = canvasOf(size);
  const actx = albedo.getContext('2d');
  actx.fillStyle = `rgb(${base[0]},${base[1]},${base[2]})`;
  actx.fillRect(0, 0, size, size);

  // --- height + roughness/metalness fields -------------------------------
  const height = new Float32Array(size * size).fill(0.5);
  const rough = new Float32Array(size * size).fill(isTrim ? 0.72 : 0.55);
  const metal = new Float32Array(size * size).fill(isTrim ? 0.85 : 0.35);

  const fillRect = (field, x, y, w, h, v) => {
    for (let yy = y; yy < y + h; yy++) {
      const row = ((yy % size) + size) % size;
      for (let xx = x; xx < x + w; xx++) field[row * size + (((xx % size) + size) % size)] = v;
    }
  };

  for (const p of panels) {
    // Gentle per-plate tone steps: strong ones read as a checkerboard at
    // mid range. Plates differ more in roughness (sheen) than in colour.
    const tone = 1 + (rnd() - 0.5) * (isTrim ? 0.16 : isPaint ? 0.1 : 0.1);
    const kind = isPaint ? 1 : rnd();
    let col = base.map(c => c * tone);
    let pRough = (isTrim ? 0.68 : isPaint ? 0.5 : 0.52) + (rnd() - 0.5) * (isTrim ? 0.2 : 0.14);
    let pMetal = isTrim ? 0.85 : isPaint ? 0.12 : 0.3;
    let pHeight = 0.45 + rnd() * 0.25;

    if (!isTrim && kind < 0.035) {
      // bare metal plate
      col = [140 * tone, 142 * tone, 148 * tone];
      pRough = 0.38 + rnd() * 0.1;
      pMetal = 0.8;
    } else if (!isTrim && kind < 0.09) {
      // faction-painted panel: a muted livery tone. Saturated accent
      // plates tile into orange/cyan confetti on the big hulls at mid range.
      col = accent.map((c, i) => c * 0.3 + base[i] * 0.7);
      pRough = 0.42;
      pMetal = 0.15;
    } else if (kind < (isTrim ? 0.4 : 0.16)) {
      // darker structural / composite panel
      col = col.map(c => c * (isTrim ? 0.8 : 0.84));
      pRough = 0.62 + rnd() * 0.15;
      pHeight = 0.36;
    }

    // panel body inset from the seam by 2px
    const inset = 2;
    actx.fillStyle = `rgb(${col[0] | 0},${col[1] | 0},${col[2] | 0})`;
    actx.fillRect(p.x + inset, p.y + inset, p.w - inset * 2, p.h - inset * 2);
    fillRect(height, p.x + inset, p.y + inset, p.w - inset * 2, p.h - inset * 2, pHeight);
    fillRect(rough, p.x + inset, p.y + inset, p.w - inset * 2, p.h - inset * 2, pRough);
    fillRect(metal, p.x + inset, p.y + inset, p.w - inset * 2, p.h - inset * 2, pMetal);
    // seam channel (recessed)
    fillRect(height, p.x, p.y, p.w, inset, 0.2);
    fillRect(height, p.x, p.y, inset, p.h, 0.2);

    // Panel features
    const feat = rnd();
    const cx = p.x + p.w / 2, cy = p.y + p.h / 2;
    if (p.w > 40 && p.h > 40) {
      if (feat < 0.14) {
        // vent slats
        const n = Math.max(2, Math.floor(Math.min(p.h, 60) / 7));
        const sw = Math.min(p.w * 0.5, 36), sh = n * 7;
        actx.fillStyle = 'rgba(0,0,0,0.55)';
        for (let i = 0; i < n; i++) {
          actx.fillRect(cx - sw / 2, cy - sh / 2 + i * 7, sw, 3);
          fillRect(height, Math.round(cx - sw / 2), Math.round(cy - sh / 2 + i * 7), Math.round(sw), 3, 0.15);
        }
      } else if (feat < 0.22) {
        // circular hatch
        const r = Math.min(p.w, p.h) * 0.22;
        actx.strokeStyle = 'rgba(0,0,0,0.5)';
        actx.lineWidth = 2;
        actx.beginPath();
        actx.arc(cx, cy, r, 0, Math.PI * 2);
        actx.stroke();
        actx.fillStyle = `rgba(${col[0] * 0.85 | 0},${col[1] * 0.85 | 0},${col[2] * 0.85 | 0},1)`;
        actx.beginPath();
        actx.arc(cx, cy, r - 2, 0, Math.PI * 2);
        actx.fill();
        for (let yy = Math.floor(cy - r); yy <= cy + r; yy++) {
          for (let xx = Math.floor(cx - r); xx <= cx + r; xx++) {
            const d = Math.hypot(xx - cx, yy - cy);
            if (d < r - 2) fillRect(height, xx, yy, 1, 1, pHeight + 0.12);
            else if (d < r) fillRect(height, xx, yy, 1, 1, 0.25);
          }
        }
      } else if (!isTrim && !isPaint && feat < 0.27) {
        // hazard chevrons
        actx.save();
        actx.beginPath();
        actx.rect(p.x + 6, cy - 6, p.w - 12, 12);
        actx.clip();
        actx.fillStyle = 'rgba(230,190,40,0.85)';
        actx.fillRect(p.x + 6, cy - 6, p.w - 12, 12);
        actx.fillStyle = 'rgba(20,20,20,0.9)';
        for (let x = p.x - 12; x < p.x + p.w + 12; x += 16) {
          actx.beginPath();
          actx.moveTo(x, cy + 6);
          actx.lineTo(x + 8, cy + 6);
          actx.lineTo(x + 16, cy - 6);
          actx.lineTo(x + 8, cy - 6);
          actx.closePath();
          actx.fill();
        }
        actx.restore();
        fillRect(metal, p.x + 6, Math.round(cy - 6), p.w - 12, 12, 0.1);
      } else if (!isTrim && !isPaint && feat < 0.295) {
        // faction stripe along one edge (rare and muted: a scatter of
        // saturated bars reads as confetti on the big hulls)
        const horiz = p.w > p.h;
        actx.fillStyle = `rgba(${accent[0] * 0.6 | 0},${accent[1] * 0.6 | 0},${accent[2] * 0.6 | 0},0.55)`;
        if (horiz) actx.fillRect(p.x + 4, p.y + 5, p.w - 8, 5);
        else actx.fillRect(p.x + 5, p.y + 4, 5, p.h - 8);
      }
    }

    // Plate bevel: a lit upper/left lip and a shadowed lower/right lip
    // give each plate a raised read before the normal map does
    if (!isTrim && p.w > 16 && p.h > 16) {
      actx.fillStyle = 'rgba(255,255,255,0.07)';
      actx.fillRect(p.x + inset, p.y + inset, p.w - inset * 2, 2);
      actx.fillRect(p.x + inset, p.y + inset, 2, p.h - inset * 2);
      actx.fillStyle = 'rgba(0,0,0,0.12)';
      actx.fillRect(p.x + inset, p.y + p.h - inset - 2, p.w - inset * 2, 2);
      actx.fillRect(p.x + p.w - inset - 2, p.y + inset, 2, p.h - inset * 2);
    }
    // Chipped livery: a few worn clusters at plate corners and edges, each
    // a handful of soft irregular flakes (dark primer, bright metal core).
    // Isolated single-texel specks read as pixel noise up close and as grey
    // speckle once minified, so there are none.
    if (isPaint) {
      const clusters = rnd() < 0.75 ? 1 + Math.floor(rnd() * 3) : 0;
      for (let c = 0; c < clusters; c++) {
        const corner = rnd() < 0.6;
        const ex = corner ? (rnd() < 0.5 ? p.x + 5 : p.x + p.w - 5) : p.x + 8 + rnd() * Math.max(0, p.w - 16);
        const ey = rnd() < 0.5 ? p.y + 5 : p.y + p.h - 5;
        const flakes = 3 + Math.floor(rnd() * 5);
        for (let i = 0; i < flakes; i++) {
          const fx = ex + (rnd() - 0.5) * 20, fy = ey + (rnd() - 0.5) * 12;
          const r = 1.6 + rnd() * 3.4;
          chipFlake(actx, fx, fy, r, rnd, primer, 0.8);
          chipFlake(actx, fx, fy, r * 0.55, rnd, [128, 130, 135], 0.75);
          const core = r * 0.55;
          for (let yy = Math.floor(fy - core); yy <= fy + core; yy++) {
            for (let xx = Math.floor(fx - core); xx <= fx + core; xx++) {
              if (Math.hypot(xx - fx, yy - fy) > core) continue;
              fillRect(metal, xx, yy, 1, 1, 0.85);
              fillRect(rough, xx, yy, 1, 1, 0.38);
            }
          }
          for (let yy = Math.floor(fy - r); yy <= fy + r; yy++) {
            for (let xx = Math.floor(fx - r); xx <= fx + r; xx++) {
              if (Math.hypot(xx - fx, yy - fy) <= r) fillRect(height, xx, yy, 1, 1, pHeight - 0.04);
            }
          }
        }
      }
    }

    // rivets along the seams
    if (p.w > 24 && p.h > 24 && rnd() < (isTrim ? 0.9 : isPaint ? 0.4 : 0.7)) {
      const step = isTrim ? 10 : 18;
      actx.fillStyle = 'rgba(0,0,0,0.35)';
      const rr = isTrim ? 1 : 1.5;
      for (let x = p.x + 8; x < p.x + p.w - 6; x += step) {
        for (const yy of [p.y + 6, p.y + p.h - 6]) {
          actx.beginPath(); actx.arc(x, yy, rr, 0, Math.PI * 2); actx.fill();
          fillRect(height, Math.round(x - 1), Math.round(yy - 1), 2, 2, pHeight + 0.2);
        }
      }
      for (let y = p.y + 8; y < p.y + p.h - 6; y += step) {
        for (const xx of [p.x + 6, p.x + p.w - 6]) {
          actx.beginPath(); actx.arc(xx, y, rr, 0, Math.PI * 2); actx.fill();
          fillRect(height, Math.round(xx - 1), Math.round(y - 1), 2, 2, pHeight + 0.2);
        }
      }
    }
  }

  // Trim: pipe runs and grille bands for a mechanical read
  if (isTrim) {
    for (let i = 0; i < 6; i++) {
      const y = Math.floor(rnd() * size);
      const t = 6 + Math.floor(rnd() * 8);
      const g = actx.createLinearGradient(0, y, 0, y + t);
      g.addColorStop(0, 'rgba(0,0,0,0.6)');
      g.addColorStop(0.4, 'rgba(255,255,255,0.18)');
      g.addColorStop(1, 'rgba(0,0,0,0.7)');
      actx.fillStyle = g;
      actx.fillRect(0, y, size, t);
      for (let yy = 0; yy < t; yy++) {
        const s = Math.sin((yy + 0.5) / t * Math.PI);
        fillRect(height, 0, y + yy, size, 1, 0.45 + 0.4 * s);
        fillRect(metal, 0, y + yy, size, 1, 0.95);
        fillRect(rough, 0, y + yy, size, 1, 0.4);
      }
    }
  }

  // Seam lines on top (dark), then grime + micro-noise modulation
  actx.strokeStyle = isTrim ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.6)';
  actx.lineWidth = 2;
  for (const p of panels) actx.strokeRect(p.x + 1, p.y + 1, p.w - 2, p.h - 2);

  const img = actx.getImageData(0, 0, size, size);
  for (let i = 0; i < size * size; i++) {
    const g = grime[i];
    const m = micro[i];
    // grime darkens, streaked; micro noise adds surface grain
    const mod = isPaint ? (0.8 + 0.34 * g) * (0.94 + 0.12 * m) : (0.86 + 0.28 * g) * (0.94 + 0.12 * m);
    img.data[i * 4] = Math.min(255, img.data[i * 4] * mod);
    img.data[i * 4 + 1] = Math.min(255, img.data[i * 4 + 1] * mod);
    img.data[i * 4 + 2] = Math.min(255, img.data[i * 4 + 2] * mod);
    rough[i] = Math.min(1, rough[i] + (g - 0.5) * 0.3 + (m - 0.5) * 0.08);
    height[i] += (m - 0.5) * 0.06;
  }
  actx.putImageData(img, 0, 0);

  // --- pack ORM ----------------------------------------------------------
  const orm = canvasOf(size);
  const octx = orm.getContext('2d');
  const oimg = octx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    oimg.data[i * 4] = 255;
    oimg.data[i * 4 + 1] = Math.max(0, Math.min(255, rough[i] * 255));
    oimg.data[i * 4 + 2] = Math.max(0, Math.min(255, metal[i] * 255));
    oimg.data[i * 4 + 3] = 255;
  }
  octx.putImageData(oimg, 0, 0);

  const set = {
    map: texOf(albedo, { srgb: true }),
    normalMap: texOf(heightToNormalCanvas(height, size, isTrim ? 2.2 : 2.8)),
    ormMap: texOf(orm)
  };
  _hullSets.set(key, set);
  return set;
}

let _bellHeatTex = null;
/**
 * Engine bell liner glow (u around the bell, v = 0 throat .. 1 lip). The
 * heat falls off fast with distance from the throat - white-hot injector,
 * orange for the first fifth, then a dark wall cooling to a black lip - so
 * a bell reads as a deep funnel, not a lit ball. Dark cooling-channel
 * ridges ring the wall and 20 petal seams run down it. 80 x 128 sRGB.
 */
export function getBellHeatTexture() {
  if (_bellHeatTex) return _bellHeatTex;
  const w = 80, n = 128;
  const data = new Uint8Array(w * n * 4);
  // (steep enough that the outer half stays dark at full thrust too, when
  // SceneManager drives the intensity ten times higher than at idle)
  const stops = [[0, [255, 238, 205]], [0.05, [255, 184, 96]], [0.15, [200, 72, 22]], [0.32, [84, 18, 6]],
    [0.55, [20, 5, 3]], [1, [3, 1, 1]]];
  for (let j = 0; j < n; j++) {
    const v = j / (n - 1);
    let k = 0;
    while (k < stops.length - 2 && v > stops[k + 1][0]) k++;
    const [v0, c0] = stops[k], [v1, c1] = stops[k + 1];
    const u = Math.min(1, Math.max(0, (v - v0) / (v1 - v0)));
    const ridge = j > 8 && j % 13 < 2 ? 0.45 : j % 13 === 2 ? 1.12 : 1;
    for (let i = 0; i < w; i++) {
      const seam = j > 6 && i % 4 === 0 ? 0.6 : 1;
      const o = (j * w + i) * 4;
      for (let c = 0; c < 3; c++) data[o + c] = Math.min(255, (c0[c] + (c1[c] - c0[c]) * u) * ridge * seam);
      data[o + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, w, n, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  _bellHeatTex = tex;
  return tex;
}

let _accentTex = null;
/**
 * Accent light strips: soft lamp cells in a glowing housing, used as the
 * accent material's map + emissiveMap. Up close an accent box reads as a
 * lit strip of cells instead of a flat, unshaded orange slab; at range
 * the cells average out to the same glow. Every row and column keeps at
 * least the housing level (0.5), so a sliver-thin accent band never lands
 * on a dark gap. Greyscale (tinted by the material colour), tileable,
 * mean ~0.68. Returns { map, emissiveMap }.
 */
export function getAccentTexture() {
  if (_accentTex) return _accentTex;
  const size = 128, cells = 4;
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const cell = (x) => {
    const d = Math.abs(((x * cells) % 1) - 0.5);          // 0 centre .. 0.5 edge
    const k = Math.min(1, Math.max(0, (0.46 - d) / 0.26));
    return k * k * (3 - 2 * k);
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = 0.5 + 0.5 * cell((x + 0.5) / size) * cell((y + 0.5) / size);
      const i = (y * size + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // albedo in sRGB, emission linear
  _accentTex = { map: texOf(canvas, { srgb: true, anisotropy: 4 }), emissiveMap: texOf(canvas, { anisotropy: 4 }) };
  return _accentTex;
}

let _hazardTex = null;
/**
 * Hazard livery for launcher rims and magazine hatches: diagonal yellow /
 * black stripes on worn paint. Tileable (stripe pitch 1/4 of the tile),
 * used with UVs fitted to physical size like the plating. Returns { map }.
 */
export function getHazardTexture() {
  if (_hazardTex) return _hazardTex;
  const size = 128;
  const canvas = canvasOf(size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(28,28,26)';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = 'rgb(214,170,42)';
  const pitch = size / 4;
  for (let x = -size; x < size * 2; x += pitch) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + pitch / 2, 0);
    ctx.lineTo(x + pitch / 2 + size, size);
    ctx.lineTo(x + size, size);
    ctx.closePath();
    ctx.fill();
  }
  // wear: soft dark scuffs so the stripes don't read as a flat decal
  const rnd = mulberry32(2024);
  for (let i = 0; i < 60; i++) {
    ctx.fillStyle = `rgba(20,18,16,${0.15 + rnd() * 0.3})`;
    ctx.beginPath();
    ctx.arc(rnd() * size, rnd() * size, 1 + rnd() * 3, 0, Math.PI * 2);
    ctx.fill();
  }
  _hazardTex = { map: texOf(canvas, { srgb: true, anisotropy: 4 }) };
  return _hazardTex;
}

let _floodTex = null;
/**
 * Floodlight pool: the soft fan of light a hull floodlight throws on the
 * plating beside it - bright at the lamp end (v = 0), widening and fading
 * along the face. Greyscale alpha for an additive quad.
 */
export function getFloodTexture() {
  if (_floodTex) return _floodTex;
  const w = 128, h = 128;
  const canvas = canvasOf(w, h);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w * 2 - 1;        // across the beam
      const v = (y + 0.5) / h;                // away from the lamp
      const spread = 0.3 + 0.7 * v;           // the beam widens away from the lamp
      const across = Math.exp(-Math.pow(u / spread, 2) * 2.4);
      const along = Math.pow(1 - v, 1.3) * Math.min(1, v * 10);
      const a = Math.max(0, Math.min(1, across * along));
      const i = (y * w + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
      img.data[i + 3] = a * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  _floodTex = texOf(canvas, { repeat: false, anisotropy: 4 });
  return _floodTex;
}

let _radiatorTex = null;
/**
 * Radiator panel: coolant channels running along the panel with the
 * emissive gradient hottest at the root (v=0) fading toward the tip.
 * Returns { map, normalMap, emissiveMap }. Clamped UVs keep the thermal
 * gradient on a cassette from wrapping back to a hot edge at its tip.
 */
export function getRadiatorTextures() {
  if (_radiatorTex) return _radiatorTex;
  const w = 256, h = 256;
  const height = new Float32Array(w * h);
  const map = canvasOf(w, h);
  const ctx = map.getContext('2d');
  ctx.fillStyle = 'rgb(25,27,29)';
  ctx.fillRect(0, 0, w, h);
  for (let x = 0; x < w; x += 8) {
    ctx.fillStyle = 'rgb(49,50,52)';
    ctx.fillRect(x + 2, 0, 4, h);
    ctx.fillStyle = 'rgb(13,15,17)';
    ctx.fillRect(x, 0, 1, h);
    for (let y = 0; y < h; y++) {
      for (let dx = 0; dx < 8; dx++) {
        height[y * w + x + dx] = 0.35 + 0.3 * Math.sin(dx / 8 * Math.PI);
      }
    }
  }
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  for (let y = 0; y < h; y += 16) ctx.fillRect(0, y, w, 1);

  const emis = canvasOf(w, h);
  const ectx = emis.getContext('2d');
  // Neutral (hot white root -> cooler, redder tip): the hue comes from the
  // material's emissive colour, set per frame from rad_out (render/Heat.js)
  const grad = ectx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgb(255,255,255)');
  grad.addColorStop(0.22, 'rgb(236,214,196)');
  grad.addColorStop(0.65, 'rgb(128,82,60)');
  grad.addColorStop(1, 'rgb(34,12,6)');
  ectx.fillStyle = grad;
  ectx.fillRect(0, 0, w, h);
  ectx.fillStyle = 'rgba(0,0,0,0.75)';
  for (let x = 0; x < w; x += 8) ectx.fillRect(x, 0, 2, h);
  for (let y = 0; y < h; y += 16) ectx.fillRect(0, y, w, 1);

  _radiatorTex = {
    map: texOf(map, { srgb: true, repeat: false }),
    normalMap: texOf(heightToNormalCanvas(height, w, 2), { repeat: false }),
    emissiveMap: texOf(emis, { srgb: true, repeat: false })
  };
  return _radiatorTex;
}

let _planetTex = null;
/** Banded gas-giant albedo (equirect). */
export function getPlanetTexture() {
  if (_planetTex) return _planetTex;
  const w = 1024, h = 512;
  const rnd = mulberry32(31337);
  const fbm = makeFbm(512, 3, 6, rnd, 0.55);
  const turb = makeFbm(512, 6, 5, rnd, 0.5);
  const canvas = canvasOf(w, h);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const stops = [
    [0.00, [178, 150, 118]], [0.12, [214, 190, 150]], [0.22, [160, 120, 90]],
    [0.34, [226, 205, 170]], [0.45, [190, 140, 100]], [0.52, [236, 220, 190]],
    [0.63, [170, 130, 96]], [0.74, [210, 180, 140]], [0.86, [150, 112, 84]],
    [1.00, [186, 160, 126]]
  ];
  const ramp = (t) => {
    t = ((t % 1) + 1) % 1;
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const a = stops[i - 1], b = stops[i];
        const f = (t - a[0]) / (b[0] - a[0]);
        return [0, 1, 2].map(k => a[1][k] + (b[1][k] - a[1][k]) * f);
      }
    }
    return stops[stops.length - 1][1];
  };
  for (let y = 0; y < h; y++) {
    const v = y / h;
    for (let x = 0; x < w; x++) {
      const u = x / w;
      const n = fbm[((y >> 0) % 512) * 512 + ((x >> 1) % 512)];
      const t = turb[((y >> 0) % 512) * 512 + ((x >> 1) % 512)];
      // bands warp with the turbulence field to get storms and eddies
      const band = v * 3.2 + (n - 0.5) * 0.35 + (t - 0.5) * 0.12 * Math.sin(u * 12.6);
      const c = ramp(band);
      const shade = 0.9 + 0.2 * t;
      const i = (y * w + x) * 4;
      img.data[i] = Math.min(255, c[0] * shade);
      img.data[i + 1] = Math.min(255, c[1] * shade);
      img.data[i + 2] = Math.min(255, c[2] * shade);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  _planetTex = texOf(canvas, { srgb: true });
  return _planetTex;
}
