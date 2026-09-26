// Torpedo thrusters for the replay viewer: body attitude, RCS jet selection
// and pulse timing. Recording-only math (metres / seconds / world frame),
// no clock, random generator or frame history: everything is a pure function
// of the recorded samples and battle time, so seeks and scrubs are exact.
//
// What the sim does (src/torpedo.py): the main engine is gimbaled and thrusts
// in whatever direction guidance commands; small precision trims can instead
// go through lateral RCS at 5% of main thrust with the engine cold. Recordings
// with thruster telemetry carry, per torpedo per frame, for the tick that
// ENDED at that frame: `thr` (main throttle), `fwd` (thrust = nose direction,
// only while the engine burns) and `rcs` (lateral RCS thrust, unit direction
// x throttle, only while it fires).
//
// Viewer model:
// - Attitude: the nose points along the tick's thrust direction by the time
//   that tick's burn starts. With the engine cold the seeker holds the
//   target: the nose points down the line of sight (which is also what
//   makes the sim's LOS-perpendicular RCS trims purely lateral), or holds
//   its last attitude once the seeker is dead, the tanks are dry or the
//   target is gone. Between keys it slews with a trapezoidal rate profile,
//   i.e. attitude jets fire a spin-up couple, the body coasts, a brake
//   couple stops it. With the engine lit the gimbal supplies most of that
//   torque (the bell visibly deflects), so only big slews add RCS puffs.
// - Translation trims: recorded `rcs` fires the quads that face away from the
//   thrust on both rings (a pure force, no torque), pulse-width modulated at
//   the recorded throttle.
// - Old recordings (no telemetry): the body follows velocity as before, and
//   a trim is inferred only where velocity turned while the main engine was
//   cold (fuel use below a lit engine's): finite-differenced acceleration
//   minus its along-axis part, above the rounding noise.

const DEG = Math.PI / 180;

/** Quad layout in the torpedo body frame (+Z nose, display units). */
export const RCS_LAYOUT = {
  rings: [-0.53, 0.47],   // z of the aft and forward quad rings
  radius: 0.175,          // nozzle exit distance from the axis
  phase: Math.PI / 4      // first quad's angle from body +X (the conduit)
};
export const JET_COUNT = 8;  // aft ring 0..3, forward ring 4..7

/** Gimbal pivot / bell length along the body axis (display units). */
export const ENGINE = { pivot: -0.675, bell: 0.27 };

export const PULSE = {
  period: 0.2,     // PWM period of a translation trim
  minOn: 0.04,     // shortest valve opening
  puffLife: 0.35   // an exhaust puff outlives its pulse by this long
};

export const SLEW = {
  bangBang: 30 * DEG,               // slews this big accelerate for half the tick
  minAccel: 0.12,                   // shortest spin-up/brake pulse (fraction of tick)
  coldMin: 0.3 * DEG, coldFull: 1.5 * DEG,  // engine cold: RCS does all the turning
  litMin: 4 * DEG, litFull: 12 * DEG,       // engine lit: gimbal first, RCS for big slews
  litLevel: 0.7,
  gimbalMax: 6 * DEG
};

export const LEGACY = {
  noise: 0.5,       // m/s^2: finite differences of 0.1 m/s-rounded velocities
  refAccel: 15,     // m/s^2 shown as a full-throttle trim (Trident RCS: 6 full, ~18 near-dry)
  mainBurn: 0.03    // km/s of delta-v per second above which the main engine was lit
};

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Recording carries thruster telemetry for this torpedo. */
export function hasTelemetry(samples) {
  for (const s of samples) if (s.state && s.state.thr !== undefined) return true;
  return false;
}

/**
 * Fraction of a slew completed at fraction f of its interval, for a
 * trapezoidal angular-rate profile that accelerates for fraction `a`,
 * coasts, and brakes for the last `a` (a = 0.5: bang-bang).
 */
export function slewProfile(f, a) {
  if (f <= 0) return 0;
  if (f >= 1) return 1;
  a = Math.min(0.5, Math.max(1e-3, a));
  const v = 1 / (1 - a);
  if (f < a) return 0.5 * v * f * f / a;
  if (f > 1 - a) return 1 - 0.5 * v * (1 - f) * (1 - f) / a;
  return v * (f - 0.5 * a);
}

/**
 * Per-jet firing levels (0..1) into out[8] for a body whose lateral axes
 * are ex/ey (world unit vectors, perpendicular to the nose; quad k's nozzle
 * points along cos(phi_k) ex + sin(phi_k) ey). `thrust` is the wanted lateral force
 * (world, |v| = level): quads facing away from it fire on both rings.
 * `push` is the wanted nose motion for an attitude couple (world, |v| =
 * level): forward quads facing away from it and aft quads facing along it
 * fire. Components along the nose are ignored (the quads are radial).
 */
export function jetLevels(ex, ey, thrust, push, out) {
  // Body-frame lateral components (the along-nose parts drop out)
  const tx = thrust.x * ex.x + thrust.y * ex.y + thrust.z * ex.z;
  const ty = thrust.x * ey.x + thrust.y * ey.y + thrust.z * ey.z;
  const px = push.x * ex.x + push.y * ex.y + push.z * ex.z;
  const py = push.x * ey.x + push.y * ey.y + push.z * ey.z;
  for (let k = 0; k < 4; k++) {
    const phi = RCS_LAYOUT.phase + k * Math.PI / 2;
    const c = Math.cos(phi), s = Math.sin(phi);
    const tr = Math.max(0, -(c * tx + s * ty));   // thrust on the body is -nozzle
    const along = c * px + s * py;
    out[k] = Math.min(1, tr + Math.max(0, along));       // aft: shove the tail back
    out[k + 4] = Math.min(1, tr + Math.max(0, -along));  // forward: shove the nose on
  }
  return out;
}

function unit3(arr, i, x, y, z) {
  const m = Math.hypot(x, y, z);
  if (m < 1e-12) return false;
  arr[3 * i] = x / m; arr[3 * i + 1] = y / m; arr[3 * i + 2] = z / m;
  return true;
}

/** Unit tangent at unit a pointing toward unit b (any perpendicular if b = -a). */
function tangentInto(arr, i, ax, ay, az, bx, by, bz) {
  const d = ax * bx + ay * by + az * bz;
  if (unit3(arr, i, bx - d * ax, by - d * ay, bz - d * az)) return;
  // parallel/antiparallel: any perpendicular of a
  if (Math.abs(ax) < 0.9) unit3(arr, i, 0, -az, ay); else unit3(arr, i, az, 0, -ax);
}

/**
 * Precompute one torpedo's thruster timeline from its recorded samples
 * ([{t, state, los?}], ascending t; `los` = vector to the live target in
 * that frame). Tick i is the sim step that ENDED at samples[i].t; it
 * started at `start[i]`.
 */
export function buildTorpedoThrusters(samples) {
  const n = samples.length;
  const tele = hasTelemetry(samples);
  const start = new Float64Array(n), end = new Float64Array(n);
  const thr = new Float32Array(n);
  const rcs = new Float32Array(3 * n);
  const nose = new Float32Array(3 * n);
  const acc = new Float32Array(n), lvl = new Float32Array(n), gim = new Float32Array(n);
  const spin = new Float32Array(3 * n), brake = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const s = samples[i].state;
    end[i] = samples[i].t;
    const dt = i > 0 ? samples[i].t - samples[i - 1].t
      : (n > 1 ? samples[1].t - samples[0].t : 1);
    start[i] = samples[i].t - (dt > 0 ? dt : 1);
    if (tele) {
      thr[i] = s.thr > 0 ? s.thr : 0;
      const r = s.rcs;
      if (r) { rcs[3 * i] = r[0]; rcs[3 * i + 1] = r[1]; rcs[3 * i + 2] = r[2]; }
      // Nose: the tick's thrust axis; coasting, the seeker's line of sight
      // at the tick's start; else hold; else the first velocity.
      const f = s.fwd;
      const prev = i > 0 ? samples[i - 1] : samples[i];
      const los = !s.disabled && !prev.state.disabled && (s.dv_remaining ?? 1) > 0 ? prev.los || samples[i].los : null;
      if (!(thr[i] > 0 && f && unit3(nose, i, f[0], f[1], f[2]))
        && !(los && unit3(nose, i, los[0], los[1], los[2]))) {
        if (i > 0) {
          nose[3 * i] = nose[3 * i - 3]; nose[3 * i + 1] = nose[3 * i - 2]; nose[3 * i + 2] = nose[3 * i - 1];
        } else if (!unit3(nose, i, s.vel?.[0] || 0, s.vel?.[1] || 0, s.vel?.[2] || 0)) {
          nose[2] = 1;
        }
      }
    } else if (i > 0) {
      legacyTrim(samples[i - 1].state, s, samples[i].t - samples[i - 1].t, rcs, i);
    }
  }
  if (tele) {
    // Slew i runs across tick i, from key i (its start) to key i+1.
    for (let i = 0; i + 1 < n; i++) {
      const ax = nose[3 * i], ay = nose[3 * i + 1], az = nose[3 * i + 2];
      const bx = nose[3 * i + 3], by = nose[3 * i + 4], bz = nose[3 * i + 5];
      const theta = Math.acos(Math.max(-1, Math.min(1, ax * bx + ay * by + az * bz)));
      if (!(theta > 1e-4)) continue;
      acc[i] = Math.min(0.5, Math.max(SLEW.minAccel, theta / SLEW.bangBang));
      const lit = thr[i] > 0;
      lvl[i] = lit ? SLEW.litLevel * smooth(SLEW.litMin, SLEW.litFull, theta)
        : smooth(SLEW.coldMin, SLEW.coldFull, theta);
      gim[i] = lit ? SLEW.gimbalMax * smooth(0, SLEW.litFull, theta) : 0;
      tangentInto(spin, i, ax, ay, az, bx, by, bz);
      tangentInto(brake, i, bx, by, bz, ax, ay, az);
    }
  }
  return { tele, n, t: samples.map(s => s.t), start, end, thr, rcs, nose, acc, lvl, gim, spin, brake };
}

/**
 * Old recordings: the RCS trim implied by a velocity change between two
 * samples, as a throttle-scaled world vector into rcs[3i..]. Zero unless the
 * main engine was cold (fuel use below a lit engine's) and the lateral part
 * of the acceleration (minus its component along the velocity, i.e. the
 * body axis) clears the rounding noise.
 */
export function legacyTrim(a, b, dt, out, i) {
  out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = 0;
  if (!(dt > 0) || a.disabled || b.disabled) return false;
  if (((a.dv_remaining ?? 0) - (b.dv_remaining ?? 0)) / dt > LEGACY.mainBurn) return false;
  const va = a.vel, vb = b.vel;
  if (!va || !vb) return false;
  let ux = va[0] + vb[0], uy = va[1] + vb[1], uz = va[2] + vb[2];
  const um = Math.hypot(ux, uy, uz);
  if (um < 1e-6) return false;
  ux /= um; uy /= um; uz /= um;
  const gx = (vb[0] - va[0]) / dt, gy = (vb[1] - va[1]) / dt, gz = (vb[2] - va[2]) / dt;
  const along = gx * ux + gy * uy + gz * uz;
  const lx = gx - along * ux, ly = gy - along * uy, lz = gz - along * uz;
  const lat = Math.hypot(lx, ly, lz);
  if (lat < LEGACY.noise) return false;
  const k = Math.min(1, lat / LEGACY.refAccel) / lat;
  out[3 * i] = lx * k; out[3 * i + 1] = ly * k; out[3 * i + 2] = lz * k;
  return true;
}

/** Index of the last entry of sorted `arr` <= x, or -1. */
function floorIndex(arr, x) {
  let lo = 0, hi = arr.length - 1;
  if (hi < 0 || x < arr[0]) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (arr[mid] <= x) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Tick in progress at battle time t (its telemetry drives thrusters), or -1. */
export function tickAt(th, t) {
  if (!th.n || t >= th.end[th.n - 1]) return -1;
  const i = floorIndex(th.t, t);   // last sample at or before t
  return i < 0 ? 0 : i + 1;
}

/**
 * Nose direction at battle time t (recordings with telemetry): keys at each
 * tick's start, trapezoidal slew between them. Writes out.x/y/z.
 */
export function attitudeAt(th, t, out) {
  const n = th.n;
  let i = floorIndex(th.start, t);
  if (i < 0) i = 0;
  const N = th.nose;
  if (i >= n - 1 || !(th.acc[i] > 0)) {
    out.x = N[3 * i]; out.y = N[3 * i + 1]; out.z = N[3 * i + 2];
    return out;
  }
  const span = th.start[i + 1] - th.start[i];
  const s = slewProfile(span > 0 ? (t - th.start[i]) / span : 1, th.acc[i]);
  // slerp between unit keys
  const ax = N[3 * i], ay = N[3 * i + 1], az = N[3 * i + 2];
  const bx = N[3 * i + 3], by = N[3 * i + 4], bz = N[3 * i + 5];
  const c = Math.max(-1, Math.min(1, ax * bx + ay * by + az * bz));
  const th0 = Math.acos(c), sn = Math.sin(th0);
  let wa, wb;
  if (sn < 1e-6) { wa = 1 - s; wb = s; } else { wa = Math.sin((1 - s) * th0) / sn; wb = Math.sin(s * th0) / sn; }
  let x = ax * wa + bx * wb, y = ay * wa + by * wb, z = az * wa + bz * wb;
  const m = Math.hypot(x, y, z) || 1;
  out.x = x / m; out.y = y / m; out.z = z / m;
  return out;
}

/** Empty pulse list for pulsesAt (caller-owned, reused every frame). */
export function makePulses(capacity = 16) {
  const list = [];
  for (let k = 0; k < capacity; k++) {
    list.push({ start: 0, end: 0, level: 0, thrust: { x: 0, y: 0, z: 0 }, push: { x: 0, y: 0, z: 0 } });
  }
  list.count = 0;
  return list;
}

function addPulse(list, start, end, level, tx, ty, tz, px, py, pz) {
  if (list.count >= list.length) return;
  const p = list[list.count++];
  p.start = start; p.end = end; p.level = level;
  p.thrust.x = tx; p.thrust.y = ty; p.thrust.z = tz;
  p.push.x = px; p.push.y = py; p.push.z = pz;
}

/**
 * Every RCS pulse that is firing at battle time t or whose exhaust puff is
 * still visible (ended less than PULSE.puffLife ago), into `list`:
 * translation trims (unit thrust direction, PWM at the recorded throttle)
 * and attitude couples (unit nose-push direction). `phase` staggers the
 * PWM grid per torpedo. Returns the gimbal deflection at t into `gimbal`
 * (world unit direction the bell exit swings toward, x/y/z, and `angle`).
 */
export function pulsesAt(th, t, phase, list, gimbal) {
  list.count = 0;
  if (gimbal) { gimbal.angle = 0; gimbal.x = gimbal.y = gimbal.z = 0; }
  if (!th.n) return list;
  const P = PULSE.period, life = PULSE.puffLife;
  const R = th.rcs;
  // Translation trims: the in-progress tick and the one before it.
  const cur = tickAt(th, t);
  const last = cur < 0 ? th.n - 1 : cur;
  for (let i = Math.max(0, last - 1); i <= last; i++) {
    const rx = R[3 * i], ry = R[3 * i + 1], rz = R[3 * i + 2];
    const tau = Math.hypot(rx, ry, rz);
    if (tau < 0.005) continue;
    const t0 = th.start[i], t1 = th.end[i];
    if (t < t0 || t > t1 + life) continue;
    const on = Math.max(PULSE.minOn, Math.min(1, tau) * P);
    const level = 0.55 + 0.45 * Math.min(1, tau);
    const m0 = Math.max(0, Math.floor((t - life - on - t0 - phase) / P));
    const m1 = Math.floor((Math.min(t, t1) - t0 - phase) / P);
    for (let m = m0; m <= m1; m++) {
      const ps = t0 + phase + m * P;
      if (ps < t0 || ps >= t1) continue;
      const pe = Math.min(ps + on, t1);
      if (ps > t || pe + life < t) continue;
      addPulse(list, ps, pe, level, rx / tau, ry / tau, rz / tau, 0, 0, 0);
    }
  }
  // Attitude couples: spin-up at the start of a slew, brake at its end.
  if (th.tele) {
    const j = floorIndex(th.start, t);
    for (let i = Math.max(0, j - 1); i <= j && i < th.n - 1; i++) {
      if (!(th.acc[i] > 0)) continue;
      const span = th.start[i + 1] - th.start[i];
      const w = th.acc[i] * span;
      const a0 = th.start[i], b1 = th.start[i + 1];
      const L = th.lvl[i];
      const S = th.spin, B = th.brake;
      if (L > 0.02) {
        if (a0 <= t && a0 + w + life >= t) addPulse(list, a0, a0 + w, L, 0, 0, 0, S[3 * i], S[3 * i + 1], S[3 * i + 2]);
        if (b1 - w <= t && b1 + life >= t) addPulse(list, b1 - w, b1, L, 0, 0, 0, B[3 * i], B[3 * i + 1], B[3 * i + 2]);
      }
      if (gimbal && th.gim[i] > 0 && i === j && w > 0) {
        // Gimbal torque follows the same bang-bang: deflect toward the slew
        // while spinning up, away from it while braking (short edge ramps).
        const D = t <= a0 + w ? S : t >= b1 - w ? B : null;
        const u = D === S ? a0 : b1 - w;
        const g = D ? Math.min(1, 4 * Math.min(t - u, u + w - t) / w) : 0;
        if (g > 0) {
          gimbal.angle = th.gim[i] * g;
          gimbal.x = D[3 * i]; gimbal.y = D[3 * i + 1]; gimbal.z = D[3 * i + 2];
        }
      }
    }
  }
  return list;
}

/** Main-engine throttle of the tick in progress at t (telemetry), else null. */
export function throttleAt(th, t) {
  if (!th.tele) return null;
  const i = tickAt(th, t);
  return i < 0 ? th.thr[th.n - 1] : th.thr[i];
}
