/**
 * Coilgun looks by gun class (WeaponTimeline.shotClass: 0 battery, 1 heavy
 * battery, 2 spinal coiler, 3 siege coiler). Plain data + pure functions,
 * shared by render/Projectiles.js and the tests.
 *
 * Tracer: `width` is the streak half-width in scene units (the ribbon
 * shader floors it at ~1 px and caps it at 3 + 1.5 * class px), `length`
 * scales the virtual-shutter streak, `gain` its HDR level (the head of a
 * battery round peaks near 3.9 in scene units, a siege round near 8),
 * `head` / `headPx` size the glow bead riding the slug.
 */
export const SLUG_LOOK = [
  { width: 0.045, length: 1.0, gain: 1.0, head: 0.085, headPx: 1.5 },
  { width: 0.06, length: 1.3, gain: 1.15, head: 0.11, headPx: 1.9 },
  { width: 0.085, length: 1.8, gain: 1.35, head: 0.16, headPx: 2.5 },
  { width: 0.11, length: 2.3, gain: 1.55, head: 0.21, headPx: 3.1 }
];

/**
 * Muzzle discharge: `size` scales the flash and its plasma ring (times the
 * hull factor), `jet` the forward jet, `sparks`/`spray` the forward spray,
 * `rings` how many discharge rings travel down the line of fire, `light`
 * the PointLight pop's peak intensity and `reach` its range (scene units).
 */
export const MUZZLE_LOOK = [
  { size: 0.9, jet: 1.0, sparks: 6, spray: 0.5, rings: 1, light: 22, reach: 14 },
  { size: 1.2, jet: 1.25, sparks: 9, spray: 0.6, rings: 1, light: 34, reach: 18 },
  { size: 3.0, jet: 1.7, sparks: 16, spray: 0.65, rings: 2, light: 60, reach: 28 },
  { size: 4.0, jet: 2.0, sparks: 22, spray: 0.75, rings: 3, light: 90, reach: 36 }
];

/** Pool lights the muzzle flashes of one frame may borrow. */
export const POP_LIGHTS = 2;

/**
 * Light-pop envelope 0..1 of a muzzle flash `age` seconds old: full at the
 * shot, gone well before the flash sprite (a pop, not a lamp). Pure in
 * battle time.
 */
export function muzzlePop(age, life) {
  if (!(age >= 0) || age >= life) return 0;
  const f = 1 - age / life;
  return f * f * f * f;
}
