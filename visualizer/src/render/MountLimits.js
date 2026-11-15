/**
 * Weapon-mount limits and the firing-arc grid shared by the hull kit
 * (Hulls.js), the turret poser (HullRigs.js / TurretArcs.js) and the arc
 * export (scripts/export_hull_mounts.mjs -> data/hull_mounts.json).
 * No three.js here: plain numbers, usable from node and the browser.
 */

/** Direction grid of data/hull_mounts.json (body frame az/el, degrees). */
export const CELL_DEG = 5;
export const AZ_CELLS = 72;
export const EL_CELLS = 36;

/** A coilgun turret depresses this far below its mount plane, a PD ball this far. */
export const GUN_DEPRESSION_DEG = 2;
export const PD_DEPRESSION_DEG = 5;

/** Pivot height above the mount origin, in mount sizes `s`. */
export const GUN_PIVOT_UP = 0.52;   // yaw group 0.2 s + trunnion 0.32 s
export const PD_PIVOT_UP = 0.3;     // ball centre

/** Yaw-swept housing other mounts must not fire through (in mount sizes). */
export const GUN_HOUSING = { r: 0.75, h: 0.85 };
export const PD_HOUSING = { r: 0.35 };

/** Grid cell k of a body-frame direction (azimuth, elevation in degrees). */
export function cellIndex(azDeg, elDeg) {
  let i = Math.floor((azDeg + 180) / CELL_DEG) % AZ_CELLS;
  if (i < 0) i += AZ_CELLS;
  const j = Math.max(0, Math.min(EL_CELLS - 1, Math.floor((elDeg + 90) / CELL_DEG)));
  return j * AZ_CELLS + i;
}

/** Uint8Array (1 per cell) -> hex string, cell k = bit 7 - k % 8 of byte k >> 3. */
export function packMask(mask) {
  let hex = '';
  for (let b = 0; b < mask.length; b += 8) {
    let v = 0;
    for (let i = 0; i < 8; i++) v = (v << 1) | (mask[b + i] ? 1 : 0);
    hex += v.toString(16).padStart(2, '0');
  }
  return hex;
}

export function unpackMask(hex) {
  const mask = new Uint8Array(AZ_CELLS * EL_CELLS);
  for (let k = 0; k < mask.length; k++) {
    mask[k] = (parseInt(hex.substr((k >> 3) * 2, 2), 16) >> (7 - (k & 7))) & 1;
  }
  return mask;
}
