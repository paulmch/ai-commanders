/**
 * LaunchSequence - how a pass-5 torpedo launch is DRAWN (pure: no THREE, no
 * clock). The engine (src/torpedo_launch.py) records the whole sequence:
 *
 *   eject     1 s: the round leaves its own rack round / VLS cell
 *             (torpedo_launched mount, round, eject_pos_m, eject_dir, nose)
 *             at 10 m/s with the ship's velocity, attitude and spin
 *   separate  until ignition: lateral RCS (trace `rcs`) pushes it clear and
 *             it turns its nose (trace `nose`) onto its first heading
 *   boost     from torpedo_ignition: the torch lights (trace `thr` > 0)
 *
 * WeaponTimeline prepends the launch state (launchState) to the round's
 * trace, TorpedoRcs keys its attitude on the recorded noses; this module
 * holds the two drawing rules that make the sequence read at every hull
 * scale (render/Scale.js):
 *
 * 1. Where. At true scale the round is exactly where the engine puts it.
 *    When the hull is drawn magnified (auto at range, cinematic) its rack
 *    is m times further out than the real one, so the round's offset from
 *    its ship is stretched by the same rule near the hull - exactly m at
 *    the rack, so it leaves the drawn round's place - and by less and less
 *    as it flies out (drawnOffset: monotonic in the real distance, so it
 *    never seems to stall or back up), the stretch fading out between 4
 *    and 20 s after ignition. Continuous in m, so nothing jumps when the
 *    camera crosses between regimes.
 * 2. How big. The flying unit is drawn the size of the rack round it
 *    replaces (UNIT_LEN_KM at true scale, the corvette rack art's 13.8 m,
 *    magnified with its hull), so it does not pop when it leaves; over the
 *    same 4-20 s after ignition it grows into the free-flight torpedo look
 *    (render/Scale.js freeScale) that keeps a round readable at range.
 */

/** Length of the drawn unit at true scale (km): the corvette rack round (Hulls.js rackRoundGeometry, 6.3 x 0.135 kit). */
export const UNIT_LEN_KM = 6.3 * 0.135 * (0.065 / 4);
/** Length of the flying torpedo model (WeaponBatches torpedoGeometry, kit units, bell to seeker). */
export const BODY_LEN_KIT = 1.94;
/** Real distance from the launcher (km) over which the drawn stretch eases off. */
export const NEAR_KM = 0.08;
/** The stretch and the launch size hand over to free flight between these seconds after ignition. */
export const HANDOVER_S = [4, 20];
/** A round whose ignition is not recorded is treated as lit this long after launch (the engine's hard cap). */
export const HARD_TIMEOUT_S = 15;

const smooth01 = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/** Weight 1..0 of the launch look at battle time t (1 through the sequence, 0 in free flight). */
export function launchWeight(t, launchT, ignitionT) {
  const lit = Number.isFinite(ignitionT) ? ignitionT : launchT + HARD_TIMEOUT_S;
  return 1 - smooth01((t - lit - HANDOVER_S[0]) / (HANDOVER_S[1] - HANDOVER_S[0]));
}

/**
 * Drawn distance from the launcher for a real distance r (km): r + (m - 1)
 * w g(r), with g(rack) = rack (so the round starts on the drawn rack) and g
 * rising ever more slowly (monotonic in r: D' = 1 + (m - 1) w g' >= 1).
 */
export function drawnOffset(r, rackKm, m, w = 1) {
  if (!(m > 1) || !(w > 0) || !(r > 0)) return r;
  const rk = Math.max(1e-6, rackKm);
  const g = rk * (1 - Math.exp(-r / NEAR_KM)) / (1 - Math.exp(-rk / NEAR_KM));
  return r + (m - 1) * w * g;
}

/** Draw scale (scene km per kit unit) of the unit model while it is in its sequence, for hull magnification m. */
export function unitScale(m) {
  return UNIT_LEN_KM / BODY_LEN_KIT * Math.max(1, m || 1);
}

/** Cubic Hermite of a ship track at t (metres; the same curve SceneManager draws the hull on). */
export function shipHermite(track, t, out) {
  const s = track?.samples;
  if (!s?.length) return false;
  let lo = 0, hi = s.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (s[mid].t <= t) lo = mid; else hi = mid - 1; }
  const a = s[lo], b = s[Math.min(lo + 1, s.length - 1)];
  const h = b.t - a.t;
  if (!(h > 0) || t <= a.t || !a.state.vel || !b.state.vel) {
    const v = a.state.vel || [0, 0, 0], dt = t - a.t;
    for (let k = 0; k < 3; k++) out[k] = a.state.pos[k] + (t > b.t || h <= 0 ? v[k] * dt : 0);
    return true;
  }
  const f = Math.min(1, (t - a.t) / h), f2 = f * f, f3 = f2 * f;
  const ca = 2 * f3 - 3 * f2 + 1, cb = (f3 - 2 * f2 + f) * h, cc = 3 * f2 - 2 * f3, cd = (f3 - f2) * h;
  for (let k = 0; k < 3; k++) out[k] = ca * a.state.pos[k] + cb * a.state.vel[k] + cc * b.state.pos[k] + cd * b.state.vel[k];
  return true;
}

/**
 * The drawn point of a sequence round: `p` (its real point, any units)
 * moved off its ship's real point `ship` by drawnOffset. Writes p.
 */
export function stretchFromShip(p, ship, rackKm, m, w, unitsPerKm = 1) {
  const dx = p[0] - ship[0], dy = p[1] - ship[1], dz = p[2] - ship[2];
  const r = Math.hypot(dx, dy, dz) / unitsPerKm;
  if (!(r > 1e-9)) return p;
  const s = drawnOffset(r, rackKm, m, w) / r;
  p[0] = ship[0] + dx * s; p[1] = ship[1] + dy * s; p[2] = ship[2] + dz * s;
  return p;
}
