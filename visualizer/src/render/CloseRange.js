/**
 * CloseRange - pure helpers that tame effects in a close-up without
 * touching the fleet-range look (no THREE; tests/trackdisplay.test.js).
 *
 * Distances are scene units (km). `viewDist` is the camera's orbit
 * distance: fleet framing sits at hundreds of units, a ship close-up at
 * tens, the same ramp trackEmphasis uses for track dimming.
 */

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** 0 at fleet range (>= 220), 1 in a close-up (<= 40). */
export function closeUpK(viewDist) {
  return 1 - smooth(40, 220, viewDist ?? 1e9);
}

/**
 * Seconds of trail history that fit in `capPx` pixels on screen. A trail
 * spans `life` seconds of world motion: at close range a 1 s slug wake is
 * 10-40 km, a laser-like line straight across the frame.
 * @param {number} life - nominal trail length (s)
 * @param {number} speed - scene units per second
 * @param {number} worldPerPx - scene units per pixel at the nearest point
 * @param {number} capPx - longest on-screen trail (px)
 */
export function trailLifeCap(life, speed, worldPerPx, capPx) {
  if (!(speed > 0) || !(worldPerPx > 0)) return life;
  return Math.min(life, (capPx * worldPerPx) / speed);
}

/**
 * Close-up fade of a trail whose head is `d` from the focus (the orbit
 * target): full strength near the focused ship, down to 10% beyond ~5
 * orbit distances. 1 everywhere at fleet range.
 */
export function focusFade(d, viewDist) {
  const k = closeUpK(viewDist);
  if (k <= 0) return 1;
  return 1 - 0.9 * k * smooth(1.5 * viewDist, 5 * viewDist, d);
}

/**
 * Spark / streak brightness for a blast of `scale` seen from `camDist`:
 * the additive burst piles up into a white blowout when the camera is a
 * few blast radii away, so it dims to 40% there; unchanged beyond ~16
 * scale units (every blast at fleet range).
 */
export function blastCloseDim(camDist, scale) {
  const r = camDist / Math.max(0.1, scale || 1);
  return 0.4 + 0.6 * smooth(3, 16, r);
}
