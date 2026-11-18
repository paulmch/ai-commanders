/**
 * PlumeProfile - the drive plume as ONE light field (pure: no THREE, no clock).
 *
 * A fusion torch's exhaust leaves at 10,256 km/s: it crosses 100 km in ten
 * milliseconds, so to the eye it is a rigid, steady lance along the thrust
 * axis. The viewer draws it as two layers that both start AT THE NOZZLE and
 * both only fade from there, so their sum - the plume - is one continuous,
 * non-increasing function of the distance aft:
 *
 *   torch  (render/Torch.js, one draw per bell) - the blinding near field:
 *          throat, standing shock diamonds, streaming sheath. Authored in
 *          kit units on the hull's group, so it is drawn at the hull's
 *          scale k (render/Scale.js) and ends a few DRAWN hull lengths aft.
 *   jet    (render/DriveJet.js, one instanced draw for the fleet) - the
 *          spear: a Gaussian beam of the engine's 3 degree core
 *          (src/drive_plume.py), in scene km. Its radius is the drive's
 *          drawn radius at the nozzle plus z tan 3 deg, so near the ship it
 *          is as wide as the drawn drive and far astern it converges to the
 *          real cone whatever k is; its side-on brightness falls as
 *          (1 + z / JET_NEAR_KM)^-JET_FALL from the nozzle (no fade-in).
 *
 * Neither layer has a fade-in or a second maximum, so there is no hole
 * between them for any hull, scale mode or magnification. The spear's
 * side-on brightness falls as 1/z (its density with the square of the
 * distance, the 3 degree cone; see JET_FALL). An optically thin beam seen from near its
 * axis gains 1/sin(angle) along each line of sight, and toward the axis's
 * vanishing point that angle falls as 1/z too: a 1/z beam would pile up
 * there into a detached disc of light (the old jet's "blob"). So the
 * line-of-sight gain is held at 1/JET_GAIN_SIN - the art call that makes
 * the screen brightness along the projected axis fall with the side-on
 * brightness from every camera: brightest at the bells, fading into the
 * distance, never a second bright region. Straight down the axis the jet
 * then adds a faint ring and the flame's own end-on disc is the picture.
 * tests/plume.test.js checks the column integrator below from the cameras
 * that showed the old blob, and the side-on profile of torch + jet in all
 * three scale modes.
 */

/** Half-angle of the engine's Gaussian core and of its wash edge (src/drive_plume.py). */
export const JET_CORE_DEG = 3.0;
export const JET_MAX_DEG = 7.5;
/** Drawn reach (km); the engine washes out to 100 km, under 0.4 GJ/s past 60. Fades over its last 40%. */
export const JET_REACH_KM = 60;
/** Side-on centreline radiance of the jet at the nozzle, full throttle (scene HDR, additive). */
export const JET_LEVEL = 0.55;
/** Near-field length (km, true scale): the side-on brightness halves over the first one. */
export const JET_NEAR_KM = 0.5;
/**
 * Far-field falloff of the side-on brightness: about 1/z (the density of a
 * 3 degree cone falls with the square of the distance), a little steeper so
 * a far spear seen as a sub-pixel hairline still fades along its length
 * (its light per unit length goes as z^(1 - JET_FALL)).
 */
export const JET_FALL = 1.15;
/** Soft ceiling of the jet's integrated radiance looking down its axis (scene HDR). */
export const JET_CAP = 2.0;
/**
 * Largest line-of-sight gain of the beam: a ray crossing it at angle a sees
 * 1/sin(a) times its side-on brightness; below JET_GAIN_SIN the gain is
 * held at 1/JET_GAIN_SIN (a crossing at ~27 degrees) - see above.
 */
export const JET_GAIN_SIN = 0.45;
/** Pixel floor of the beam radius (pixels); the floor keeps the energy, so a far spear is a faint hairline. */
export const JET_PIXEL_FLOOR = 0.75;
/** Sub-chords of the analytic column per pixel, by graphics tier. */
export const JET_SEGMENTS = { low: 4, high: 8 };

export const TAN_CORE = Math.tan(JET_CORE_DEG * Math.PI / 180);
/** Bounding cone of the draw: the wash edge plus 2 degrees, from 2.5 drive radii at the nozzle. */
export const BOUND_TAN = Math.tan((JET_MAX_DEG + 2) * Math.PI / 180);
export const BOUND_R0 = 2.5;

/** Beam radius (1/e, km) z km aft of a nozzle of drawn radius w0 km. */
export function jetWidth(z, w0) {
  return w0 + Math.max(0, z) * TAN_CORE;
}

/** Reach fade 0..1: 1 out to 60% of the reach, smoothly 0 at the reach. */
export function jetReachFade(z, reach = JET_REACH_KM) {
  const x = Math.max(0, Math.min(1, (z - 0.6 * reach) / (0.4 * reach)));
  return 1 - x * x * (3 - 2 * x);
}

/**
 * Side-on centreline radiance of the jet z km aft of the nozzle (throttle
 * 0..1). Mirrored in DriveJet.js's fragment shader (`axial`).
 */
export function jetAxial(z, throttle = 1, reach = JET_REACH_KM) {
  if (!(z >= 0) || z >= reach) return z < 0 ? JET_LEVEL * throttle : 0;
  return JET_LEVEL * throttle * Math.pow(1 + z / JET_NEAR_KM, -JET_FALL) * jetReachFade(z, reach);
}

// erf: series near 0 (the difference of two small arguments stays exact),
// Abramowitz-Stegun 7.1.27 beyond. Mirrored in the shader.
function erfa(x) {
  const a = Math.abs(x);
  let d = 1 + a * (0.278393 + a * (0.230389 + a * (0.000972 + a * 0.078108)));
  d *= d; d *= d;
  return Math.sign(x) * (1 - 1 / d);
}
export function erfS(x) {
  if (Math.abs(x) < 0.5) {
    const x2 = x * x;
    return 1.1283792 * x * (1 - x2 * (0.33333333 - x2 * (0.1 - x2 * 0.0238095)));
  }
  return erfa(x);
}

/** Chord [tIn, tOut] of the ray ro + t rd (t >= 0) through the bounding cone, or null. Jet frame: axis -Z. */
export function jetChord(ro, rd, R0, TB, reach) {
  let tIn = 0, tOut = 1e9;
  const [ox, oy, oz] = ro, [dx, dy, dz] = rd;
  if (Math.abs(dz) > 1e-9) {
    const ta = -oz / dz, tb = (-reach - oz) / dz;
    tIn = Math.max(tIn, Math.min(ta, tb));
    tOut = Math.min(tOut, Math.max(ta, tb));
  } else if (oz > 0 || oz < -reach) return null;
  const A0 = R0 - TB * oz, kz = -TB * dz;
  const qa = dx * dx + dy * dy - kz * kz;
  const qb = ox * dx + oy * dy - A0 * kz;
  const qc = ox * ox + oy * oy - A0 * A0;
  const disc = qb * qb - qa * qc;
  if (Math.abs(qa) < 1e-9) {
    if (Math.abs(qb) > 1e-12) { const t = -qc / (2 * qb); if (qb > 0) tOut = Math.min(tOut, t); else tIn = Math.max(tIn, t); }
    else if (qc > 0) return null;
  } else if (disc <= 0) {
    if (qa > 0) return null;
  } else {
    const sq = Math.sqrt(disc);
    const r1 = (-qb - sq) / qa, r2 = (-qb + sq) / qa;
    if (qa > 0) { tIn = Math.max(tIn, r1); tOut = Math.min(tOut, r2); }
    else {
      const lo = Math.min(r1, r2), hi = Math.max(r1, r2);
      if (Math.min(tOut, lo) > tIn) tOut = Math.min(tOut, lo); else tIn = Math.max(tIn, hi);
    }
  }
  return tOut > tIn ? [tIn, tOut] : null;
}

/**
 * Integrated jet radiance along one view ray, before the soft ceiling -
 * the JS twin of DriveJet.js's fragment shader (same chord, sub-chords,
 * closed-form Gaussian column per sub-chord, pixel floor).
 *
 * @param {number[]} ro  camera in the jet frame (km; nozzle at the origin, exhaust down -Z)
 * @param {number[]} rd  unit view direction in the jet frame
 * @param {object} jet   { throttle, w0 (drawn drive radius, km), reach (km) }
 * @param {object} opt   { segments, pixelWorld (km per km of range per pixel; 0 = no floor) }
 */
export function jetColumn(ro, rd, jet, opt = {}) {
  const { throttle = 1, w0, reach = JET_REACH_KM } = jet;
  const N = opt.segments ?? JET_SEGMENTS.high;
  const px = (opt.pixelWorld ?? 0) * JET_PIXEL_FLOOR;
  const dist0 = Math.hypot(ro[0], ro[1], ro[2]);
  const R0 = BOUND_R0 * w0 + 2 * px * dist0, TB = BOUND_TAN + 2 * px;
  const ch = jetChord(ro, rd, R0, TB, reach);
  if (!ch) return 0;
  const tIn = Math.max(ch[0], 0), tOut = ch[1];
  if (!(tOut > tIn)) return 0;
  const dd = rd[0] * rd[0] + rd[1] * rd[1];
  const sinT = Math.sqrt(dd);
  const par = sinT <= 1e-4;
  const gain = sinT / Math.max(sinT, JET_GAIN_SIN);
  const tc = par ? 0 : -(ro[0] * rd[0] + ro[1] * rd[1]) / dd;
  const lx = ro[0] + rd[0] * tc, ly = ro[1] + rd[1] * tc;
  const d2line = lx * lx + ly * ly;
  const zA = Math.max(-(ro[2] + rd[2] * tIn), 0), zB = Math.max(-(ro[2] + rd[2] * tOut), 0);
  const c = Math.min(JET_NEAR_KM, w0 / TAN_CORE) + 1e-6;
  const lA = Math.log(zA + c), lB = Math.log(zB + c), dzAB = zB - zA;
  // sub-chord ends spaced evenly in log(z + c): fine at the nozzle, long far aft
  const even = Math.abs(dzAB) <= 1e-4 * (Math.min(zA, zB) + c);
  const frac = (u) => (even ? u : Math.min(1, Math.max(0, (Math.exp(lA + (lB - lA) * u) - c - zA) / dzAB)));
  let col = 0;
  for (let i = 0; i < N; i++) {
    const a = tIn + (tOut - tIn) * frac(i / N), b = tIn + (tOut - tIn) * frac((i + 1) / N);
    const tm = tIn + (tOut - tIn) * frac((i + 0.5) / N);
    // width and brightness of the sub-chord: halfway between its point
    // nearest the axis and its (log) middle - continuous in the pixel, and
    // within ~12% of a brute-force integral except straight down the axis,
    // where the ceiling takes over anyway
    const te = par ? tm : 0.5 * (Math.min(Math.max(tc, a), b) + tm);
    const z = Math.max(-(ro[2] + rd[2] * te), 0);
    const w = jetWidth(z, w0);
    const wf = Math.max(w, px * te);
    const S = jetAxial(z, throttle, reach) * (w / wf);
    let d2, G;
    if (par) {
      const qx = ro[0] + rd[0] * te, qy = ro[1] + rd[1] * te;
      d2 = qx * qx + qy * qy;
      G = (b - a) / (1.7724539 * wf);
    } else {
      const k = sinT / wf;
      d2 = d2line;
      G = 0.5 * (erfS(k * (b - tc)) - erfS(k * (a - tc))) / sinT;
    }
    col += S * Math.exp(-d2 / (wf * wf)) * G * gain;
  }
  return col;
}

/** The soft ceiling the shader applies to a column. */
export function jetCap(col) {
  return JET_CAP * (1 - Math.exp(-col / JET_CAP));
}

/**
 * Drawn drive radius (kit units) of a hull's bell cluster: the farthest
 * bell centre plus its nozzle radius. Times the hull's draw scale k it is
 * the jet's radius at the nozzle.
 */
export function driveRadius(positions, nozzleR) {
  let r = 0;
  for (const p of positions) r = Math.max(r, Math.hypot(p.x, p.y));
  return r + nozzleR;
}
