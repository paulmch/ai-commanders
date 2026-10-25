import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { FACTION_PAINT, getFloodTexture } from './Textures.js';

/**
 * Faction livery: painted hull codes ("DD-04") on the flanks, and the
 * hull floodlights that keep them readable on the shadowed side.
 *
 * Each ship gets one small canvas texture with its hull code; every code
 * placement on the hull is a quad baked into ONE decal mesh (one draw
 * call). Quads float a hair off the plating instead of relying on
 * polygonOffset, which the logarithmic depth buffer ignores.
 *
 * Floodlights (Expanse-style hull lighting, no dynamic lights): each is an
 * additive pool quad lying on its face, fanning out from a small lamp
 * (HullLights), all in ONE mesh per ship. The code decal is faintly
 * self-lit as if washed by them. HullRigs dims both with the hull's power
 * (floodLevel) on battle time.
 */

/** Pool opacity at full power. */
export const FLOOD_OPACITY = 0.42;
/** Hull-code self-illumination at full power. */
export const CODE_GLOW = 0.32;
/** Livery self-illumination at full power (the floodlights washing the paint). */
export const LIVERY_FLOOD = 0.16;
/** Sensor glass glow at full power. */
export const GLASS_GLOW = 0.6;

/**
 * Floodlight power 0..1 at battle time t: full on a sound hull, browning
 * out below 55 % (like the windows), mostly dark while dying. Deterministic
 * in (t, phase) so seeks and paused frames agree.
 */
export function floodLevel(t, hull = 1, dying = false, phase = 0) {
  let v = 0.4 + 0.6 * Math.max(0, Math.min(1, hull));
  if (dying || hull < 0.55) {
    const f = Math.sin(t * 6.3 + phase * 9.1) * Math.sin(t * 2.1 + phase * 27.3);
    v *= f > (dying ? 0.2 : 0.55) ? 0.12 : (dying ? 0.35 : 0.85);
  }
  return v;
}

export const CLASS_CODES = {
  corvette: 'K', frigate: 'FF', destroyer: 'DD', cruiser: 'CA', cruiser_torpedo: 'CT',
  battlecruiser: 'CB', battleship: 'BB', dreadnought: 'DN', dreadnought_siege: 'DS'
};

/** Hull code for a ship: class letters + two-digit number from its id. */
export function hullCode(shipType, shipId = '') {
  const prefix = CLASS_CODES[shipType] || 'XX';
  const digits = String(shipId).match(/(\d+)(?!.*\d)/);
  let n;
  if (digits) n = Number(digits[1]) % 100;
  else {
    let h = 0;
    for (const c of String(shipId)) h = (h * 31 + c.charCodeAt(0)) | 0;
    n = 10 + (Math.abs(h) % 90);
  }
  return `${prefix}-${String(n).padStart(2, '0')}`;
}

/** Canvas with the code in pale paint, a faction bar underneath. */
function codeTexture(text, faction) {
  const w = 512, h = 128;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  ctx.font = 'bold 92px "DejaVu Sans Condensed", "Arial Narrow", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // Stencil-style dark keyline so the code reads on lit and shadowed plating
  ctx.lineWidth = 10;
  ctx.strokeStyle = 'rgba(10,12,14,0.85)';
  ctx.strokeText(text, w / 2, h * 0.44, w * 0.94);
  ctx.fillStyle = 'rgb(232,230,220)';
  ctx.fillText(text, w / 2, h * 0.44, w * 0.94);
  const bar = FACTION_PAINT[faction] || FACTION_PAINT.alpha;
  ctx.fillStyle = `rgb(${bar[0]},${bar[1]},${bar[2]})`;
  ctx.fillRect(w * 0.08, h * 0.86, w * 0.84, h * 0.1);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/**
 * Build the decal mesh for `placements`: [{x, y, z, w, h, face}] in
 * hull-local space, face '+x' | '-x' | '+y' | '-y'. Returns the mesh.
 */
export function buildMarkings(group, placements, text, faction) {
  if (!placements.length || typeof document === 'undefined') return null;
  const geoms = [];
  for (const p of placements) {
    const g = new THREE.PlaneGeometry(p.w, p.h);
    // PlaneGeometry faces +Z with the text running along +X. Turn it so
    // it reads left-to-right from whichever side it is painted on.
    if (p.face === '+x') g.rotateY(Math.PI / 2);
    else if (p.face === '-x') g.rotateY(-Math.PI / 2);
    else if (p.face === '+y') { g.rotateX(-Math.PI / 2); g.rotateY(Math.PI / 2); }
    else if (p.face === '-y') { g.rotateX(Math.PI / 2); g.rotateY(-Math.PI / 2); }
    g.translate(p.x, p.y, p.z);
    geoms.push(g);
  }
  const merged = geoms.length === 1 ? geoms[0] : BufferGeometryUtils.mergeGeometries(geoms, false);
  if (geoms.length > 1) for (const g of geoms) g.dispose();
  const map = codeTexture(text, faction);
  const mat = new THREE.MeshStandardMaterial({
    map, alphaTest: 0.4, roughness: 0.55, metalness: 0.15,
    emissive: 0xffffff, emissiveMap: map, emissiveIntensity: CODE_GLOW
  });
  const mesh = new THREE.Mesh(merged, mat);
  mesh.name = 'hull_markings';
  mesh.renderOrder = 2;
  group.add(mesh);
  return mesh;
}

const _FACE = {
  '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0]
};

/**
 * Build the floodlight pools for `floods`: [{x, y, z, face, len, width,
 * dir}] in hull-local space - a lamp at (x, y, z) on `face` throwing light
 * `len` along the hull toward dir (+1 nose, -1 drive) and `width` across.
 * One additive mesh; returns it (null when there are none or no canvas).
 */
export function buildFloodlights(group, floods) {
  if (!floods.length || typeof document === 'undefined') return null;
  const geoms = [];
  const n = new THREE.Vector3(), along = new THREE.Vector3(), across = new THREE.Vector3();
  const m = new THREE.Matrix4();
  for (const f of floods) {
    n.fromArray(_FACE[f.face] || _FACE['+y']);
    // plane +Y (texture v = 1, the bright end) points back at the lamp
    along.set(0, 0, -(f.dir || 1));
    across.crossVectors(along, n);
    const g = new THREE.PlaneGeometry(f.width, f.len);
    m.makeBasis(across, along, n);
    m.setPosition(f.x + n.x * 0.006, f.y + n.y * 0.006, f.z + (f.dir || 1) * f.len * 0.5);
    g.applyMatrix4(m);
    geoms.push(g);
  }
  const merged = geoms.length === 1 ? geoms[0] : BufferGeometryUtils.mergeGeometries(geoms, false);
  if (geoms.length > 1) for (const g of geoms) g.dispose();
  const mat = new THREE.MeshBasicMaterial({
    map: getFloodTexture(), color: 0xffe2bc, transparent: true, opacity: FLOOD_OPACITY,
    blending: THREE.AdditiveBlending, depthWrite: false
  });
  const mesh = new THREE.Mesh(merged, mat);
  mesh.name = 'hull_floodlights';
  mesh.renderOrder = 3;
  // not plating: DamageFX's hull raycasts skip it
  mesh.userData.damageFx = true;
  group.add(mesh);
  return mesh;
}
