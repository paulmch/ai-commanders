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

const _hullSets = new Map();

/**
 * Faction hull plating set. `variant` is 'hull' (painted armour plating)
 * or 'trim' (dark mechanical structure: dense small panels, grilles, pipes).
 * Returns { map, normalMap, ormMap } textures, all tileable.
 */
export function getHullTextures(faction, variant = 'hull') {
  const key = `${faction}:${variant}`;
  if (_hullSets.has(key)) return _hullSets.get(key);

  const size = variant === 'hull' ? 1024 : 512;
  const rnd = mulberry32(faction === 'alpha' ? 4242 : 7777 + (variant === 'trim' ? 99 : 0));
  const isTrim = variant === 'trim';

  // Base palette
  const base = faction === 'alpha'
    ? (isTrim ? [40, 45, 52] : [88, 98, 112])
    : (isTrim ? [46, 42, 40] : [104, 94, 88]);
  const accent = faction === 'alpha' ? [40, 200, 255] : [255, 130, 50];

  const panels = subdividePanels(size, rnd, isTrim ? 20 : 56, isTrim ? 72 : 260);
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
    const tone = 1 + (rnd() - 0.5) * (isTrim ? 0.26 : 0.26);
    const kind = rnd();
    let col = base.map(c => c * tone);
    let pRough = (isTrim ? 0.68 : 0.5) + (rnd() - 0.5) * 0.2;
    let pMetal = isTrim ? 0.85 : 0.3;
    let pHeight = 0.45 + rnd() * 0.25;

    if (!isTrim && kind < 0.06) {
      // bare metal plate
      col = [150 * tone, 152 * tone, 158 * tone];
      pRough = 0.32 + rnd() * 0.1;
      pMetal = 0.95;
    } else if (!isTrim && kind < 0.09) {
      // faction-painted panel
      col = accent.map((c, i) => c * 0.55 + base[i] * 0.45);
      pRough = 0.42;
      pMetal = 0.15;
    } else if (kind < (isTrim ? 0.4 : 0.16)) {
      // darker structural / composite panel
      col = col.map(c => c * (isTrim ? 0.8 : 0.72));
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
      } else if (!isTrim && feat < 0.27) {
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
      } else if (!isTrim && feat < 0.33) {
        // faction stripe along one edge
        const horiz = p.w > p.h;
        actx.fillStyle = `rgba(${accent[0]},${accent[1]},${accent[2]},0.7)`;
        if (horiz) actx.fillRect(p.x + 4, p.y + 5, p.w - 8, 5);
        else actx.fillRect(p.x + 5, p.y + 4, 5, p.h - 8);
      }
    }

    // rivets along the seams
    if (p.w > 24 && p.h > 24 && rnd() < (isTrim ? 0.9 : 0.7)) {
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
  actx.strokeStyle = isTrim ? 'rgba(0,0,0,0.75)' : 'rgba(0,0,0,0.6)';
  actx.lineWidth = 2;
  for (const p of panels) actx.strokeRect(p.x + 1, p.y + 1, p.w - 2, p.h - 2);

  const img = actx.getImageData(0, 0, size, size);
  for (let i = 0; i < size * size; i++) {
    const g = grime[i];
    const m = micro[i];
    // grime darkens, streaked; micro noise adds surface grain
    const mod = (0.86 + 0.28 * g) * (0.94 + 0.12 * m);
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

let _radiatorTex = null;
/**
 * Radiator panel: coolant channels running along the panel with the
 * emissive gradient hottest at the root (v=0) fading toward the tip.
 * Returns { map, emissiveMap }.
 */
export function getRadiatorTextures() {
  if (_radiatorTex) return _radiatorTex;
  const w = 256, h = 512;
  const map = canvasOf(w, h);
  const ctx = map.getContext('2d');
  ctx.fillStyle = 'rgb(36,30,30)';
  ctx.fillRect(0, 0, w, h);
  for (let x = 0; x < w; x += 16) {
    ctx.fillStyle = 'rgb(58,50,48)';
    ctx.fillRect(x + 3, 0, 10, h);
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(x, 0, 2, h);
  }
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  for (let y = 0; y < h; y += 64) ctx.fillRect(0, y, w, 3);

  const emis = canvasOf(w, h);
  const ectx = emis.getContext('2d');
  const grad = ectx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgb(255,120,60)');
  grad.addColorStop(0.35, 'rgb(255,70,30)');
  grad.addColorStop(0.75, 'rgb(150,20,10)');
  grad.addColorStop(1, 'rgb(40,4,2)');
  ectx.fillStyle = grad;
  ectx.fillRect(0, 0, w, h);
  ectx.fillStyle = 'rgba(0,0,0,0.75)';
  for (let x = 0; x < w; x += 16) ectx.fillRect(x, 0, 3, h);
  for (let y = 0; y < h; y += 64) ectx.fillRect(0, y, w, 3);

  _radiatorTex = {
    map: texOf(map, { srgb: true }),
    emissiveMap: texOf(emis, { srgb: true })
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
