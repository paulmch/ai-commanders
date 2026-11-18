/**
 * Scale - how big a hull is drawn (pure: no THREE, no clock, no state).
 *
 * 1 scene unit is 1 km and recording positions are real metres. The hull
 * kit builds every class in KIT UNITS (render/Hulls.js `hullSize`: a
 * corvette is 4 kit units long, a dreadnought 20.8), and every effect that
 * belongs to a hull is authored in the same units. A ship is then drawn by
 * one uniform factor k (scene km per kit unit) on its group:
 *
 *   cinematic  k = 1                      the pass-4/5 look: hulls 57-80x
 *                                         oversize so a fleet reads at the
 *                                         hundreds of km the sim fights at
 *   true       k = length_m / 1000 / kit  the real 65-275 m hull, always
 *   auto       k = true x magnification   real size up close, a readable
 *                                         floor at distance (default)
 *
 * AUTO RULE. Let s be the share of the viewport HEIGHT the hull's real
 * length covers when seen side on from distance d with vertical field of
 * view fov:  s = L / (2 d tan(fov / 2)).  The hull is drawn covering
 *
 *   shown(s) = s                                   for s >= S1 = TRUE_AT * F
 *   shown(s) = F + (S1 - F) (s / S1)^(S1/(S1 - F)) for s <  S1
 *
 * with F the class floor (FLOOR_SHARE of the viewport height for a 65 m
 * hull, x sqrt(L / 65 m): a dreadnought's dot is twice a corvette's). The
 * two branches meet at S1 with value AND slope equal (no pop, no kink),
 * shown() is monotonic in s (a hull never grows on screen as the camera
 * backs off) and tends to F at infinity. Magnification = shown / s >= 1,
 * exactly 1 inside the true regime.
 *
 * Because the rule is in shares of the viewport height, a 1280x720 frame
 * and a 4K frame of the same camera show the same picture, and the only
 * inputs are the camera (position, fov) and the ship's position: scrubbing
 * the timeline or re-running a director shot gives the same hull sizes.
 *
 * In the magnified regime a hull covers at most S1 of the viewport height,
 * so two ships can only overlap when their centres are within S1 of the
 * screen height of each other - where two icons of any size would. In the
 * true regime they are their real size and overlap only if they collide.
 */
import { REAL_LENGTH_M } from '../TrackDisplay.js';
import { HULL_BASE_SIZES, HULL_SIZE_SCALE } from './Hulls.js';

export const SCALE_MODES = ['auto', 'true', 'cinematic'];
export const DEFAULT_SCALE_MODE = 'auto';

/** Floor of a 65 m hull, as a share of the viewport height (14 px at 720p, 21 px at 1080p). */
export const FLOOR_SHARE = 0.0195;
/** The floor grows with the square root of the hull length from this reference. */
export const FLOOR_REF_M = 65;
export const FLOOR_EXP = 0.5;
/** A hull is at exactly its real size once that covers TRUE_AT floors of the viewport. */
export const TRUE_AT = 2;
/**
 * Free-flying things authored in kit units (slug heads, torpedo bodies,
 * flares with no ship to belong to) are drawn as fittings of a nominal
 * hull of this length at their own position.
 */
export const FREE_REF_M = 100;
/** Kit units per real metre of that nominal hull: the mean of the hull classes (1 kit unit = 16 m). */
export const FREE_TRUE_K = 0.016;

/**
 * Real hull length (m) of a class: TrackDisplay.REAL_LENGTH_M, which
 * tests/scale.test.js pins to data/fleet_ships.json and hull_mounts.json.
 */
export function trueLengthM(shipType) {
  return REAL_LENGTH_M[shipType] ?? REAL_LENGTH_M.destroyer;
}

/** Kit length (kit units) of a class: Hulls.hullSize().length. */
export function kitLength(shipType) {
  return (HULL_BASE_SIZES[shipType] || HULL_BASE_SIZES.destroyer).length * HULL_SIZE_SCALE;
}

/** Scene km per kit unit that draws a class at its real size. */
export function trueK(shipType) {
  return trueLengthM(shipType) / 1000 / kitLength(shipType);
}

export function normalizeScaleMode(mode) {
  const m = String(mode || '').toLowerCase();
  if (m === 'real' || m === 'physical') return 'true';
  return SCALE_MODES.includes(m) ? m : DEFAULT_SCALE_MODE;
}

export function nextScaleMode(mode) {
  return SCALE_MODES[(SCALE_MODES.indexOf(normalizeScaleMode(mode)) + 1) % SCALE_MODES.length];
}

/** The class floor as a share of the viewport height. */
export function floorShare(lengthM) {
  return FLOOR_SHARE * Math.pow(Math.max(1e-6, lengthM) / FLOOR_REF_M, FLOOR_EXP);
}

/** Share of the viewport height a length (km) covers side on at distance (km). */
export function screenShare(lengthKm, dist, fovDeg) {
  if (!(dist > 0)) return Infinity;
  return lengthKm / (2 * dist * Math.tan(fovDeg * Math.PI / 360));
}

/** Share of the viewport height a hull is DRAWN at, for its real share `s` and floor `F`. */
export function shownShare(s, F) {
  const S1 = TRUE_AT * F;
  if (!(s < S1)) return s;
  if (!(s > 0)) return F;
  return F + (S1 - F) * Math.pow(s / S1, S1 / (S1 - F));
}

/** Auto-mode magnification (>= 1; exactly 1 in the true regime) for real share `s`. */
export function magnification(s, F) {
  if (!(s < TRUE_AT * F)) return 1;
  if (!(s > 0)) return Infinity;
  return shownShare(s, F) / s;
}

/** Auto-mode magnification of a real length (m) at camera distance `dist` (km). */
export function magnificationAt(lengthM, dist, fovDeg) {
  return magnification(screenShare(lengthM / 1000, dist, fovDeg), floorShare(lengthM));
}

/**
 * Camera distance (km) inside which a hull of this real length is drawn at
 * exactly its real size in auto mode.
 */
export function trueRegimeDistance(lengthM, fovDeg) {
  return lengthM / 1000 / (2 * Math.tan(fovDeg * Math.PI / 360) * TRUE_AT * floorShare(lengthM));
}

/**
 * Scene km per kit unit for a ship of class `shipType` whose centre is
 * `dist` km from a camera with vertical field of view `fovDeg`.
 */
export function shipScale(mode, shipType, dist, fovDeg) {
  if (mode === 'cinematic') return 1;
  const k = trueK(shipType);
  if (mode === 'true') return k;
  const m = magnificationAt(trueLengthM(shipType), dist, fovDeg);
  return Number.isFinite(m) ? k * m : k;
}

/** The same for a free-flying effect authored in kit units (FREE_REF_M). */
export function freeScale(mode, dist, fovDeg) {
  if (mode === 'cinematic') return 1;
  if (mode === 'true') return FREE_TRUE_K;
  const m = magnificationAt(FREE_REF_M, dist, fovDeg);
  return Number.isFinite(m) ? FREE_TRUE_K * m : FREE_TRUE_K;
}

/**
 * The hull length camera framing is measured in (scene km): the real
 * length in true and auto (a close-up at a few hull lengths is inside the
 * true regime, so the framing does not depend on where the camera was),
 * the kit length in cinematic.
 */
export function framingLength(mode, shipType) {
  return mode === 'cinematic' ? kitLength(shipType) : trueLengthM(shipType) / 1000;
}

/**
 * The draw scale a world-space buffer is BAKED at (spall, the smoke ring):
 * k snapped to steps of BAKE_STEP (2%). Such buffers are rewritten when the
 * baked scale changes, so snapping keeps a moving camera in the magnified
 * regime from rewriting them every frame - and, being a function of k
 * alone, keeps them a pure function of battle time and camera. In the true
 * regime k is constant and nothing is rewritten at all. The baked offsets
 * differ from the hull's own scale by under 1%, on hulls that are a few
 * dozen pixels long.
 */
export const BAKE_STEP = 1.02;
export function bakeScale(k) {
  if (!(k > 0) || k === 1) return k > 0 ? k : 1;
  const n = Math.round(Math.log(k) / Math.log(BAKE_STEP));
  return Math.pow(BAKE_STEP, n);
}
