import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { FACTION_PAINT } from './Textures.js';

/**
 * Faction livery: painted hull codes ("DD-04") on the flanks.
 *
 * Each ship gets one small canvas texture with its hull code; every code
 * placement on the hull is a quad baked into ONE decal mesh (one draw
 * call). Quads float a hair off the plating instead of relying on
 * polygonOffset, which the logarithmic depth buffer ignores.
 */

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
  const mat = new THREE.MeshStandardMaterial({
    map: codeTexture(text, faction), alphaTest: 0.4,
    roughness: 0.55, metalness: 0.15
  });
  const mesh = new THREE.Mesh(merged, mat);
  mesh.name = 'hull_markings';
  mesh.renderOrder = 2;
  group.add(mesh);
  return mesh;
}
