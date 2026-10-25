import * as THREE from 'three';
import { getPuffTexture, getGlowTexture, getNoiseTexture } from './Textures.js';
import { RCS_S, fitScale } from './Hulls.js';

/**
 * RCS attitude jets.
 *
 * The sim turns ships with bang-bang attitude control (simulation.py
 * _rotate_ship_toward): full torque until the midpoint, then full reverse
 * torque to stop on the new heading. With the torch lit the main nozzle's
 * thrust vectoring supplies ~95% of that torque; with the torch dark the
 * RCS quads do all of it, ~20x more slowly (CLAUDE.md turn-time table).
 *
 * This module reads the ship's angular acceleration around the playback
 * time out of the recording (`alpha`/`omega`/`turn` when the trace carries
 * them, otherwise a Savitzky-Golay second derivative of the recorded
 * heading), picks the nose/tail jet couples whose torque lines up with it
 * and fires them as short pulses: hard and continuous during an RCS-only
 * turn, faint during a torch-vectored one apart from a full-strength kick
 * when a turn starts and when it reverses into braking, and the odd
 * attitude-hold blip otherwise.
 *
 * Everything visible at battle time t is a pure function of the recording
 * around t (pulse timing is hashed from battle-time cells, puffs are
 * re-derived from their emission slot), so scrubbing and seeking are
 * deterministic and nothing replays from a stale spawn time.
 *
 * Body frame (matches Object3D.lookAt with world +Y up): +Z nose, +X right,
 * +Y up. Mount contract (hulls stream): ship.userData.rcsMounts =
 * [{pos: Vector3 hull-local, dir: Vector3 unit outward nozzle axis}, ...],
 * each the exit of a nozzle on a thruster block of size RCS_S * fitScale(L).
 */

const DEG = Math.PI / 180;

/** Attitude authority per class, deg/s^2 (data/fleet_ships.json attitude_control). */
export const ATTITUDE = {
  corvette: { tv: 2.445, rcs: 0.1227 },
  frigate: { tv: 1.589, rcs: 0.0518 },
  destroyer: { tv: 0.847, rcs: 0.0221 },
  cruiser: { tv: 0.454, rcs: 0.0085 },
  cruiser_torpedo: { tv: 0.454, rcs: 0.0085 },
  battlecruiser: { tv: 0.454, rcs: 0.0085 },
  battleship: { tv: 0.265, rcs: 0.0043 },
  dreadnought: { tv: 0.144, rcs: 0.0017 },
  dreadnought_siege: { tv: 0.13, rcs: 0.00154 }
};

export const MODE_NONE = 0;
export const MODE_TV = 1;
export const MODE_RCS = 2;

/** Pulse timing, battle seconds. */
export const RCS_TIMING = {
  cell: 0.5,          // one pulse decision per cell
  minPulse: 0.1,
  attack: 0.025,
  decay: 0.05,
  kickLookback: 0.75, // demand rise over this window = a turn starting/reversing
  kick: 0.4,          // full-strength pulse that opens a vectored turn / its braking
  tvGain: 0.3,        // steady RCS level while the torch does the turning
  puffSlot: 1 / 30,   // puff emission grid
  puffLife: 0.6
};

/**
 * Exhaust-puff scale for a jet of length jetLen: sub-linear in hull size,
 * so a dreadnought's jets still vent small directional puffs instead of
 * the cotton-wool clouds a linear scale gave (~0.6 = a corvette jet).
 */
export const PUFF_BASE = 0.6;
export function puffScale(jetLen) {
  return jetLen > 0 ? PUFF_BASE * Math.sqrt(jetLen / PUFF_BASE) : PUFF_BASE;
}

/**
 * Jet plume size in thruster-block units (block = RCS_S * fitScale(L), the
 * same physical size on every hull): plume length at full intensity, the
 * radius parameter of the sheath and core cones (the cone root is JET_ROOT
 * of it, so the sheath leaves the nozzle at about the bell mouth, 0.13
 * block) and the nozzle flash sprite.
 */
export const JET_BLOCKS = { len: 4.5, sheath: 0.5, core: 0.24, flash: 2.2 };
/** Plume radius at the nozzle, as a share of the cone's radius parameter. */
export const JET_ROOT = 0.3;

/** Jet dimensions (scene units) for a hull of length L. */
export function jetDims(L) {
  const s = RCS_S * fitScale(L);
  const len = JET_BLOCKS.len * s;
  return {
    block: s,
    len,
    sheath: JET_BLOCKS.sheath * s,
    core: JET_BLOCKS.core * s,
    flash: JET_BLOCKS.flash * s,
    puff: puffScale(len)
  };
}

// Recorded headings are rounded to 4 decimals: sigma of one component
const FWD_SIGMA = 1e-4 / Math.sqrt(12);

// ---------------------------------------------------------------------------
// Deterministic hashing
// ---------------------------------------------------------------------------
export function hash01(a, b = 0, c = 0) {
  let h = Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul((b | 0) + 0x632be5ab, 0x165667b1) ^ Math.imul((c | 0) + 0x5bd1e995, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export function seedOf(str) {
  let s = 0;
  for (const ch of String(str)) s = (s * 31 + ch.charCodeAt(0)) | 0;
  return s;
}

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------------------
// Kinematics from the recording
// ---------------------------------------------------------------------------

/** Quadratic Savitzky-Golay weights over 2m+1 samples (unit spacing). */
function sgWeights(m) {
  const N = 2 * m + 1;
  let S2 = 0;
  for (let j = -m; j <= m; j++) S2 += j * j;
  const raw = [];
  let W2 = 0;
  for (let j = -m; j <= m; j++) {
    const w = j * j - S2 / N;
    raw.push(w);
    W2 += w * w;
  }
  const d2 = raw.map(w => 2 * w / W2);
  const d1 = [];
  for (let j = -m; j <= m; j++) d1.push(j / S2);
  return { d1, d2, noise2: Math.sqrt(d2.reduce((a, c) => a + c * c, 0)) };
}
const SG = [null, sgWeights(1), sgWeights(2), sgWeights(3), sgWeights(4)];

/**
 * Smoothing half-width for finite-differenced headings: the narrowest
 * window (1 = plain second difference, which is exactly the impulse the
 * frame-to-frame slerp shows) whose rounding noise stays well under the
 * active actuator's authority - so a dreadnought's 0.0017 deg/s^2 RCS turn
 * still stands out of the 4-decimal heading noise, while torch-vectored
 * turns keep frame-sharp timing.
 */
export function smoothingWindow(aref) {
  for (let m = 1; m <= 4; m++) if (FWD_SIGMA * SG[m].noise2 < 0.1 * aref) return m;
  return 4;
}

/**
 * Per-frame attitude kinematics for one ship.
 * @returns {Object} typed arrays over trace frames: t, fwd (xyz), dem
 *   (world-frame angular acceleration past the noise deadband, divided by
 *   what the active actuator delivers: |dem| = 1 is a full-authority
 *   bang-bang turn), mode (MODE_*), alive, evade
 */
export function buildShipKinematics(trace, shipId, shipType) {
  const n = trace ? trace.length : 0;
  const k = {
    n,
    t: new Float64Array(n),
    fwd: new Float32Array(3 * n),
    dem: new Float32Array(3 * n),
    mode: new Int8Array(n),
    alive: new Uint8Array(n),
    evade: new Uint8Array(n)
  };
  const att = ATTITUDE[shipType] || ATTITUDE.destroyer;
  const aTv = (att.tv + att.rcs) * DEG;
  const aRcs = att.rcs * DEG;
  const present = new Uint8Array(n);
  const recA = new Uint8Array(n);
  const recW = new Uint8Array(n);
  const recAlpha = new Float32Array(3 * n);
  const recOmega = new Float32Array(3 * n);
  const turn = new Int8Array(n).fill(-1);
  const thrust = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const f = trace[i];
    k.t[i] = f.t;
    const s = f.ships ? f.ships[shipId] : null;
    if (!s) { k.fwd[3 * i + 2] = 1; continue; }
    present[i] = 1;
    k.alive[i] = s.destroyed || s.dying ? 0 : 1;
    const fw = s.fwd || [0, 0, 1];
    const l = Math.hypot(fw[0], fw[1], fw[2]) || 1;
    k.fwd[3 * i] = fw[0] / l; k.fwd[3 * i + 1] = fw[1] / l; k.fwd[3 * i + 2] = fw[2] / l;
    thrust[i] = s.thrust || 0;
    if (Array.isArray(s.alpha)) {
      recA[i] = 1;
      recAlpha[3 * i] = s.alpha[0]; recAlpha[3 * i + 1] = s.alpha[1]; recAlpha[3 * i + 2] = s.alpha[2];
    }
    if (Array.isArray(s.omega)) {
      recW[i] = 1;
      recOmega[3 * i] = s.omega[0]; recOmega[3 * i + 1] = s.omega[1]; recOmega[3 * i + 2] = s.omega[2];
    }
    if ('turn' in s) turn[i] = s.turn === 'rcs' ? MODE_RCS : s.turn === 'tv' ? MODE_TV : MODE_NONE;
    if (s.evade) k.evade[i] = 1;
  }

  const ok = (j) => j >= 0 && j < n && present[j] && k.alive[j];
  for (let i = 0; i < n; i++) {
    if (!ok(i)) continue;
    let mode;
    if (turn[i] >= 0) {
      // The recorder's turn/omega describe the interval ending at their
      // frame, while the torque impulse at frame i sits between intervals i
      // and i+1: take the actuator that starts a turn (i+1), else the one
      // that ends it (i), else a neighbour the smoothing bleeds into
      const next = i < n - 1 ? turn[i + 1] : -1;
      const prev = i > 0 ? turn[i - 1] : -1;
      mode = next > 0 ? next : turn[i] > 0 ? turn[i] : prev > 0 ? prev : MODE_NONE;
    } else {
      mode = thrust[i] > 0.02 ? MODE_TV : MODE_RCS;
    }
    k.mode[i] = mode;
  }

  const ax = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!ok(i)) continue;
    const mode = k.mode[i];
    const aref = mode === MODE_TV ? aTv : aRcs;
    ax[0] = ax[1] = ax[2] = 0;
    let dead = 1e-6;
    if (ok(i + 1) && recW[i] && recW[i + 1]) {
      // Recorded omega is the rate over the interval ending at its frame
      // (nose change since the previous frame), so the change the viewer's
      // slerp shows at frame i is omega[i+1] - omega[i]; noise-free
      const h = k.t[i + 1] - k.t[i] || 1;
      for (let c = 0; c < 3; c++) ax[c] = (recOmega[3 * (i + 1) + c] - recOmega[3 * i + c]) / h;
    } else if (ok(i + 1) && recA[i + 1]) {
      // recorded alpha = change of that omega: one frame late for frame i
      for (let c = 0; c < 3; c++) ax[c] = recAlpha[3 * (i + 1) + c];
    } else {
      // alpha_perp = f x f'' (f' = w x f with w perpendicular to f)
      let m = smoothingWindow(aref);
      while (m > 0) {
        let fits = true;
        for (let j = -m; j <= m; j++) {
          // A vectored turn's torque must not leak into a torch-dark frame's
          // wide window, where it would read as 20-50x the RCS authority
          if (!ok(i + j) || (mode === MODE_RCS && k.mode[i + j] === MODE_TV)) { fits = false; break; }
        }
        if (fits) break;
        m--;
      }
      if (m > 0) {
        const h = (k.t[i + m] - k.t[i - m]) / (2 * m);
        const w = SG[m].d2;
        let f2x = 0, f2y = 0, f2z = 0;
        for (let j = -m; j <= m; j++) {
          const q = 3 * (i + j);
          f2x += w[j + m] * k.fwd[q]; f2y += w[j + m] * k.fwd[q + 1]; f2z += w[j + m] * k.fwd[q + 2];
        }
        const h2 = h * h || 1;
        f2x /= h2; f2y /= h2; f2z /= h2;
        const fx = k.fwd[3 * i], fy = k.fwd[3 * i + 1], fz = k.fwd[3 * i + 2];
        ax[0] = fy * f2z - fz * f2y;
        ax[1] = fz * f2x - fx * f2z;
        ax[2] = fx * f2y - fy * f2x;
        dead = 3 * FWD_SIGMA * SG[m].noise2 / h2;
      }
    }
    // Normalise per frame, so interpolating across a torch-on/off edge
    // never divides a vectored turn's torque by the RCS authority
    const mag = Math.hypot(ax[0], ax[1], ax[2]);
    const raw = mode !== MODE_NONE && mag > dead ? (mag - dead) / aref : 0;
    const sc = raw > 0 ? demandLevel(raw) / mag : 0;
    k.dem[3 * i] = ax[0] * sc; k.dem[3 * i + 1] = ax[1] * sc; k.dem[3 * i + 2] = ax[2] * sc;
  }
  return k;
}

/**
 * Demand magnitude kept for a frame whose angular acceleration is `raw`
 * times the actuator's authority. Up to ~1.2 it is the real bang-bang
 * command. Far above it is not physical: the sim stores the turn rate
 * unsigned, so a wobble aim that crosses sides reverses the spin at full
 * rate within one frame (~0.5 rad/s^2 on a corvette, ~11x authority).
 * Those frames read as one firm transient instead of a max-level strobe.
 */
export function demandLevel(raw) {
  if (raw <= 1.2) return raw;
  if (raw <= 2.5) return 1.2;
  return 0.8;
}

/** Index i with t[i] <= t < t[i+1] (clamped) and the fraction into it. */
function locate(k, t) {
  const T = k.t;
  if (t <= T[0]) return [0, 0];
  if (t >= T[k.n - 1]) return [k.n - 1, 0];
  let lo = 0, hi = k.n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (T[mid] <= t) lo = mid; else hi = mid;
  }
  const span = T[hi] - T[lo];
  return [lo, span > 0 ? (t - T[lo]) / span : 0];
}

/**
 * Kinematic sample at time t. Writes into `out`: alive, mode, evade,
 * fwd [x,y,z] (unit), demand [x,y,z] (world, normalised - see dem above).
 * The mode comes from whichever bracketing frame contributes more demand.
 */
export function sampleKinematics(k, t, out) {
  out.alive = false;
  if (!k || k.n === 0) return out;
  const [i, f] = locate(k, t);
  const j = f > 0 ? i + 1 : i;
  out.alive = !!k.alive[i];     // the interpolator holds frame0 state until the next frame lands
  out.evade = !!(k.evade[i] || k.evade[j]);
  const D = k.dem;
  for (let c = 0; c < 3; c++) out.demand[c] = D[3 * i + c] + (D[3 * j + c] - D[3 * i + c]) * f;
  const wi = (1 - f) * Math.hypot(D[3 * i], D[3 * i + 1], D[3 * i + 2]);
  const wj = f * Math.hypot(D[3 * j], D[3 * j + 1], D[3 * j + 2]);
  out.mode = wi > wj || (wi === wj && f < 0.5) ? k.mode[i] : k.mode[j];
  const fx = k.fwd[3 * i] + (k.fwd[3 * j] - k.fwd[3 * i]) * f;
  const fy = k.fwd[3 * i + 1] + (k.fwd[3 * j + 1] - k.fwd[3 * i + 1]) * f;
  const fz = k.fwd[3 * i + 2] + (k.fwd[3 * j + 2] - k.fwd[3 * i + 2]) * f;
  const l = Math.hypot(fx, fy, fz) || 1;
  out.fwd[0] = fx / l; out.fwd[1] = fy / l; out.fwd[2] = fz / l;
  return out;
}

export function newSample() {
  return { alive: false, mode: MODE_NONE, evade: false, fwd: [0, 0, 1], demand: [0, 0, 0] };
}

/**
 * Body axes for a heading, exactly as Object3D.lookAt builds them for a
 * non-camera object with world +Y up: z = fwd, x = up x z, y = z x x.
 */
export function lookAtBasis(fwd, x, y) {
  let zx = fwd[0], zy = fwd[1], zz = fwd[2];
  let xx = zz, xy = 0, xz = -zx;               // (0,1,0) x z
  if (xx * xx + xz * xz === 0) {
    zz += 0.0001;
    const l = Math.hypot(zx, zy, zz);
    zx /= l; zy /= l; zz /= l;
    xx = zz; xz = -zx;
  }
  const xl = Math.hypot(xx, xy, xz);
  xx /= xl; xy /= xl; xz /= xl;
  x[0] = xx; x[1] = xy; x[2] = xz;
  y[0] = zy * xz - zz * xy;
  y[1] = zz * xx - zx * xz;
  y[2] = zx * xy - zy * xx;
}

/**
 * Torque demand in body pitch/yaw axes (1 = the bang-bang turn at the
 * active actuator's full authority), capped at 1.5.
 * @returns {number[]} out [dx, dy]
 */
export function bodyDemand(sample, out) {
  out[0] = out[1] = 0;
  if (!sample.alive || sample.mode === MODE_NONE) return out;
  const X = _bx, Y = _by;
  lookAtBasis(sample.fwd, X, Y);
  const a = sample.demand;
  const tx = a[0] * X[0] + a[1] * X[1] + a[2] * X[2];
  const ty = a[0] * Y[0] + a[1] * Y[1] + a[2] * Y[2];
  const mag = Math.hypot(tx, ty);
  if (mag === 0) return out;
  const sc = Math.min(1.5, mag) / mag;
  out[0] = tx * sc;
  out[1] = ty * sc;
  return out;
}
const _bx = [0, 0, 0], _by = [0, 0, 0];

// ---------------------------------------------------------------------------
// Jets
// ---------------------------------------------------------------------------
const vx = (v, i) => (Array.isArray(v) ? v[i] : i === 0 ? v.x : i === 1 ? v.y : v.z);

export const ROLE_RADIAL = 0;
export const ROLE_SIDE = 1;
export const ROLE_AXIAL = 2;
export const ROLE_BLOCKED = 3;

/**
 * Free run of each nozzle's plume (scene units, Infinity = clear for at
 * least `len`): the first hit of its axis on the hull's own plating
 * (userData.parts, hull-local; bounding boxes are culled first, so this is
 * a handful of small-part raycasts per nozzle, once per hull).
 * @returns {Float32Array} per mount
 */
export function jetClearance(mounts, parts, len) {
  const out = new Float32Array(mounts.length).fill(Infinity);
  if (!parts || !parts.length) return out;
  const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const meshes = parts.map(p => {
    if (!p.geom.boundingBox) p.geom.computeBoundingBox();
    return new THREE.Mesh(p.geom, mat);
  });
  const ray = new THREE.Raycaster();
  const o = new THREE.Vector3(), d = new THREE.Vector3(), hb = new THREE.Vector3();
  const hits = [];
  mounts.forEach((m, i) => {
    d.set(vx(m.dir, 0), vx(m.dir, 1), vx(m.dir, 2)).normalize();
    // start just off the bell mouth, clear of the nozzle's own block
    o.set(vx(m.pos, 0), vx(m.pos, 1), vx(m.pos, 2)).addScaledVector(d, len * 0.005);
    ray.set(o, d);
    ray.far = len;
    let best = Infinity;
    for (const mesh of meshes) {
      if (!ray.ray.intersectBox(mesh.geometry.boundingBox, hb) || hb.distanceTo(o) > len) continue;
      hits.length = 0;
      mesh.raycast(ray, hits);
      for (const h of hits) best = Math.min(best, h.distance);
    }
    out[i] = best;
  });
  mat.dispose();
  return out;
}

/**
 * What each nozzle of a thruster block is for, from its geometry: RADIAL
 * (pointing straight out from the hull axis - the attitude couples),
 * SIDE (tangential, along the skin - lateral trims) or AXIAL (fore/aft).
 * With `free` (jetClearance) a nozzle whose plume would run into the hull
 * within `len` - a side nozzle facing a torpedo rail or a launcher bank -
 * is BLOCKED and never fires.
 * @returns {Int8Array} ROLE_* per mount
 */
export function jetRoles(mounts, free = null, len = 0) {
  const out = new Int8Array(mounts.length);
  mounts.forEach((m, i) => {
    if (free && free[i] < len) { out[i] = ROLE_BLOCKED; return; }
    const dx = vx(m.dir, 0), dy = vx(m.dir, 1), dz = vx(m.dir, 2);
    const dl = Math.hypot(dx, dy, dz) || 1;
    if (Math.abs(dz) / dl > 0.7) { out[i] = ROLE_AXIAL; return; }
    const rx = vx(m.pos, 0), ry = vx(m.pos, 1);
    const rl = Math.hypot(rx, ry), ql = Math.hypot(dx, dy);
    out[i] = rl === 0 || (rx * dx + ry * dy) / (rl * ql) > 0.8 ? ROLE_RADIAL : ROLE_SIDE;
  });
  return out;
}

/**
 * Unit torque axis of each attitude jet in the body frame: the reaction
 * force is opposite the nozzle axis, torque = r x F about the hull origin.
 * Only radial nozzles turn the ship (a block's side nozzles point the same
 * ways from a few cm along the skin, and are kept for trims), and of those
 * only the ones with at least a third of the best leverage - fore/aft
 * nozzles near the axis get a zero axis and never fire for rotation. Axes
 * are unit length rather than leverage-scaled, so the nose and tail
 * stations fire a balanced couple (a pure torque) wherever they sit.
 * The couple's slight roll on hulls whose blocks sit off the diagonal is
 * left in (paired blocks cancel it).
 * @returns {Float32Array} 3 per jet
 */
export function jetTorqueAxes(mounts, roles = jetRoles(mounts)) {
  const out = new Float32Array(mounts.length * 3);
  let best = 0;
  mounts.forEach((m, i) => {
    if (roles[i] !== ROLE_RADIAL) return;
    const rx = vx(m.pos, 0), ry = vx(m.pos, 1), rz = vx(m.pos, 2);
    const dl = Math.hypot(vx(m.dir, 0), vx(m.dir, 1), vx(m.dir, 2)) || 1;
    const Fx = -vx(m.dir, 0) / dl, Fy = -vx(m.dir, 1) / dl, Fz = -vx(m.dir, 2) / dl;
    out[3 * i] = ry * Fz - rz * Fy;
    out[3 * i + 1] = rz * Fx - rx * Fz;
    out[3 * i + 2] = rx * Fy - ry * Fx;
    best = Math.max(best, Math.hypot(out[3 * i], out[3 * i + 1], out[3 * i + 2]));
  });
  // An end of the hull with several stations (the dreadnought's two bow
  // quads) shares its half of the couple between them
  const stations = [new Set(), new Set()];
  const end = (m) => (vx(m.pos, 2) >= 0 ? 0 : 1);
  const lever = (i) => Math.hypot(out[3 * i], out[3 * i + 1], out[3 * i + 2]);
  mounts.forEach((m, i) => {
    if (best > 0 && lever(i) >= best / 3) stations[end(m)].add(Math.round(vx(m.pos, 2) * 1000));
  });
  const share = stations.map(st => Math.min(stations[0].size || 1, stations[1].size || 1) / (st.size || 1));
  for (let i = 0; i < mounts.length; i++) {
    const l = lever(i);
    const sc = best > 0 && l >= best / 3 ? share[end(mounts[i])] / l : 0;
    for (let c = 0; c < 3; c++) out[3 * i + c] *= sc;
  }
  return out;
}

/** Per-jet share of a body pitch/yaw demand: projection onto its torque axis. */
export function jetDemands(axes, dx, dy, out) {
  const n = axes.length / 3;
  for (let i = 0; i < n; i++) out[i] = Math.max(0, axes[3 * i] * dx + axes[3 * i + 1] * dy);
  return out;
}

/**
 * Default mounts: an RCS quad (up/down/left/right nozzles) at a nose and a
 * tail station. `radius(z, axis)` returns the hull's surface distance from
 * the axis along a body axis at station z (from raycasts), or null.
 */
export function defaultRcsMounts(length, width, zNose = 0.4 * length, zTail = -0.4 * length, radius = null) {
  const mounts = [];
  const dirs = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  for (const z of [zNose, zTail]) {
    for (const d of dirs) {
      const r = (radius && radius(z, d)) || width * 0.3;
      mounts.push({ pos: [d[0] * r, d[1] * r, z], dir: d.slice() });
    }
  }
  return mounts;
}

/** How sharply a jet's demand rose over the kick lookback (0..1). */
export function riseOf(now, prev) {
  return Math.min(1, Math.max(0, (now - prev) * 2.2)) * smooth(0.12, 0.35, now);
}

/**
 * One pulse cell's plan for one jet, from its torque demand sampled around
 * the cell start: `now`, `prev` (kickLookback earlier), `fut` (the smaller
 * of +0.5 s and +1 s: whether the demand persists) and `prevRise` (the
 * strongest rise of ANY jet over the previous two cells, so a turn gets one
 * kick, and spin reversals every frame can't strobe couples at full power).
 * @returns {{on:number, level:number}} on-time (s, 0 = silent) and level 0..1
 */
export function planJet(j, mode, out = { on: 0, level: 0 }) {
  const P = RCS_TIMING.cell;
  out.on = 0; out.level = 0;
  if (mode === MODE_NONE || j.now <= 0) return out;
  if (mode === MODE_RCS) {
    // Torch dark: the quads do all the turning, duty follows demand; a
    // demand that doesn't persist (a one-frame reversal) fires softer
    const duty = Math.min(1, j.now / 0.7);
    if (duty < 0.15) return out;
    out.level = Math.min(j.now, j.fut) >= 0.25 ? 1 : 0.6;
    out.on = duty > 0.85 ? P : Math.max(RCS_TIMING.minPulse, duty * P);
    return out;
  }
  // Torch vectoring does the turning and the quads only assist. Steering
  // micro-corrections that reverse every frame stay dark; a sustained turn
  // gets one full-strength kick as it starts (or flips into braking) and a
  // faint steady assist after that
  const sustained = Math.min(j.now, j.fut);
  if (riseOf(j.now, j.prev) > 0.5 && j.prevRise <= 0.5 && sustained >= 0.25) {
    out.level = 1;
    out.on = RCS_TIMING.kick;
    return out;
  }
  const duty = 0.45 * sustained;
  if (duty < 0.15) return out;
  out.level = RCS_TIMING.tvGain;
  out.on = Math.max(RCS_TIMING.minPulse, duty * P);
  return out;
}

/** Pulse envelope: fast attack, sustain, quick exponential tail-off. */
export function pulseEnvelope(tau, on) {
  if (on <= 0 || tau < 0) return 0;
  if (tau < on) return Math.min(1, tau / RCS_TIMING.attack);
  return Math.exp(-(tau - on) / RCS_TIMING.decay);
}

// ---------------------------------------------------------------------------
// Per-ship planner: kinematics source + jets + pulse cells
// ---------------------------------------------------------------------------
export class RcsPlanner {
  /**
   * @param {Object} o shipId, shipType, mounts [{pos, dir}], roles
   *   (optional jetRoles, e.g. with the hull's clearance), source
   *   ({sample(t, out)}) or trace
   */
  constructor(o) {
    this.shipId = o.shipId;
    this.shipType = o.shipType;
    this.seed = seedOf(o.shipId);
    this.phase = hash01(this.seed, 11) * RCS_TIMING.cell;
    this.setMounts(o.mounts, o.roles);
    this.source = o.source || null;
    this.kin = null;
    this.traceRef = null;
    this.traceLen = -1;
    this.cells = new Map();
    this._s = newSample();
    this._d = [0, 0];
    this._pj = { on: 0, level: 0 };
    this._j = { now: 0, prev: 0, fut: 0, prevRise: 0 };
    if (o.trace) this.setTrace(o.trace);
  }

  setMounts(mounts, roles = null) {
    this.mounts = mounts;
    this.jetCount = mounts.length;
    this.roles = roles || jetRoles(mounts);
    this.axes = jetTorqueAxes(mounts, this.roles);
    this._dem = [0, 1, 2, 3, 4, 5, 6, 7].map(() => new Float32Array(this.jetCount));
    this.cells = new Map();
  }

  /** Per-jet demand at time t into `out`; returns the sample's mode/evade. */
  _jetDemandAt(t, out) {
    const s = this.sample(t, this._s);
    bodyDemand(s, this._d);
    jetDemands(this.axes, this._d[0], this._d[1], out);
    return s;
  }

  /** (Re)derive kinematics when the trace object or its length changes. */
  setTrace(trace) {
    if (trace === this.traceRef && trace.length === this.traceLen) return;
    this.traceRef = trace;
    this.traceLen = trace.length;
    this.kin = buildShipKinematics(trace, this.shipId, this.shipType);
    this.cells.clear();
  }

  sample(t, out) {
    if (this.source) return this.source.sample(t, out);
    return sampleKinematics(this.kin, t, out);
  }

  cellIndex(t) { return Math.floor((t - this.phase) / RCS_TIMING.cell); }
  cellStart(c) { return this.phase + c * RCS_TIMING.cell; }

  /** Pulse plan for cell c: shared start offset, per-jet on-time and level. */
  plan(c) {
    let p = this.cells.get(c);
    if (p) return p;
    const P = RCS_TIMING.cell;
    const n = this.jetCount;
    p = { start: 0, on: new Float32Array(n), level: new Float32Array(n) };
    const tc = this.cellStart(c);
    const s = this.sample(tc, this._s);
    if (s.alive) {
      const mode = s.mode;
      const evade = s.evade;
      const lb = RCS_TIMING.kickLookback;
      const [dn, dp, df1, df2, dq, dqp, dr, drp] = this._dem;
      this._jetDemandAt(tc, dn);
      this._jetDemandAt(tc - lb, dp);
      this._jetDemandAt(tc + 0.5, df1);
      this._jetDemandAt(tc + 1.0, df2);
      this._jetDemandAt(tc - P, dq);
      this._jetDemandAt(tc - P - lb, dqp);
      this._jetDemandAt(tc - 2 * P, dr);
      this._jetDemandAt(tc - 2 * P - lb, drp);
      let recentRise = 0;
      for (let k = 0; k < n; k++) {
        recentRise = Math.max(recentRise, riseOf(dq[k], dqp[k]), riseOf(dr[k], drp[k]));
      }
      const j = this._j;
      j.prevRise = recentRise;
      let maxOn = 0;
      for (let k = 0; k < n; k++) {
        j.now = dn[k]; j.prev = dp[k]; j.fut = Math.min(df1[k], df2[k]);
        planJet(j, mode, this._pj);
        p.on[k] = this._pj.on;
        p.level[k] = this._pj.level;
        maxOn = Math.max(maxOn, p.on[k]);
      }
      if (maxOn === 0) {
        // Attitude hold: an occasional blip on a random couple, more often
        // while the ship is jinking
        const rate = evade ? 0.3 : 0.05;
        if (hash01(this.seed, c, 7) < rate) {
          const th = hash01(this.seed, c, 8) * Math.PI * 2;
          const dh = this._dem[0];
          jetDemands(this.axes, Math.cos(th), Math.sin(th), dh);
          for (let k = 0; k < n; k++) {
            if (dh[k] > 0.6) { p.on[k] = RCS_TIMING.minPulse; p.level[k] = 0.5; }
          }
          maxOn = RCS_TIMING.minPulse;
        }
      }
      p.start = maxOn < P ? (P - maxOn) * hash01(this.seed, c, 3) : 0;
    }
    this.cells.set(c, p);
    if (this.cells.size > 48) {
      for (const key of this.cells.keys()) {
        if (key < c - 8 || key > c + 8) this.cells.delete(key);
      }
    }
    return p;
  }

  /** Jet k intensity 0..1 at battle time t (no flicker). */
  intensity(t, k) {
    const c = this.cellIndex(t);
    let best = 0;
    for (let cc = c; cc >= c - 1; cc--) {
      const p = this.plan(cc);
      const on = p.on[k];
      if (on <= 0) continue;
      const e = p.level[k] * pulseEnvelope(t - this.cellStart(cc) - p.start, on);
      if (e > best) best = e;
    }
    return best;
  }

  flicker(t, k) {
    const s = this.seed * 0.001 + k * 1.7;
    return 0.8 + 0.12 * Math.sin(t * 71.3 + s) + 0.08 * Math.sin(t * 157.1 + s * 2.3);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
const ADDITIVE = {
  transparent: true,
  depthWrite: false,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor,
  blendEquation: THREE.AddEquation
};

/** Jet plume profile along +Z: narrow at the nozzle, flaring, soft tip. */
function jetGeometry() {
  const pts = [];
  const along = 14;
  for (let i = 0; i <= along; i++) {
    const t = i / along;
    const flare = JET_ROOT + (1 - JET_ROOT) * smooth(0, 0.45, t);
    const taper = Math.pow(1 - smooth(0.5, 1.0, t), 0.6);
    pts.push(new THREE.Vector2(Math.max(0.01, flare * taper), t));
  }
  const g = new THREE.LatheGeometry(pts, 12);
  g.rotateX(Math.PI / 2);   // nozzle at z=0, tip at z=+1
  return g;
}

const jetVertex = /* glsl */`
  attribute vec4 aJet;          // intensity, layer (0 core / 1 sheath), seed
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vAxis;
  varying vec3 vJet;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vUv = uv;
    mat4 m = modelMatrix * instanceMatrix;
    vec4 wp = m * vec4(position, 1.0);
    vec4 mv = viewMatrix * wp;
    mat3 nm = transpose(inverse(mat3(m)));
    vNormal = normalize(mat3(viewMatrix) * (nm * normal));
    vViewDir = normalize(mv.xyz);
    vAxis = normalize(mat3(viewMatrix) * (mat3(m) * vec3(0.0, 0.0, 1.0)));
    vJet = aJet.xyz;
    gl_Position = projectionMatrix * mv;
    #include <logdepthbuf_vertex>
  }
`;

const jetFragment = /* glsl */`
  uniform sampler2D uNoise;
  uniform float uTime;
  uniform vec3 uCore;
  uniform vec3 uSheath;
  uniform vec3 uEdge;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vAxis;
  varying vec3 vJet;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float t = vUv.y;
    bool sheath = vJet.y > 0.5;
    float n1 = texture2D(uNoise, vec2(vUv.x * 2.0 + vJet.z, t * 1.4 - uTime * 0.8)).r;
    float turb = 0.7 + 0.6 * n1;
    float axial = exp(-t * (sheath ? 2.2 : 3.4)) * (1.0 - smoothstep(0.5, 1.0, t)) * smoothstep(0.0, 0.04, t);
    float ndv = abs(dot(normalize(vNormal), normalize(vViewDir)));
    float thick = pow(ndv, sheath ? 1.2 : 2.2);
    float endOn = abs(dot(normalize(vViewDir), vAxis));
    thick *= mix(0.1, 1.0, sqrt(max(0.0, 1.0 - endOn * endOn)));
    float a = axial * turb * thick * vJet.x;
    vec3 col = sheath ? mix(uEdge, uSheath, clamp(axial * 1.3, 0.0, 1.0))
                      : mix(uSheath, uCore, clamp(axial * 1.6, 0.0, 1.0));
    gl_FragColor = vec4(col * a * (sheath ? 0.8 : 3.4), 1.0);
  }
`;

const puffVertex = /* glsl */`
  attribute vec4 aPos;          // xyz offset from the mesh origin, w size (<0: nozzle glow)
  attribute vec4 aCol;          // rgb premultiplied, w rotation
  varying vec2 vUv;
  varying vec3 vCol;
  varying float vKind;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec4 mv = modelViewMatrix * vec4(aPos.xyz, 1.0);
    float s = abs(aPos.w);
    float c = cos(aCol.w), sn = sin(aCol.w);
    mv.xy += vec2(c * position.x - sn * position.y, sn * position.x + c * position.y) * s;
    gl_Position = projectionMatrix * mv;
    vUv = uv;
    vCol = aCol.rgb;
    vKind = aPos.w < 0.0 ? 1.0 : 0.0;
    #include <logdepthbuf_vertex>
  }
`;

const puffFragment = /* glsl */`
  uniform sampler2D uPuff;
  uniform sampler2D uGlow;
  varying vec2 vUv;
  varying vec3 vCol;
  varying float vKind;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float a = vKind > 0.5 ? texture2D(uGlow, vUv).a : texture2D(uPuff, vUv).a;
    gl_FragColor = vec4(vCol * a, 1.0);
  }
`;

const MAX_JETS = 512;      // cone instances (2 per firing jet)
const MAX_PUFFS = 3072;    // puffs + nozzle glows
const _zAxis = new THREE.Vector3(0, 0, 1);

/**
 * Per-jet frames: hull-local nozzle exit `pos`, outward axis `dir`, the
 * rotation `q` taking the plume's +Z onto it and a lateral basis u, v.
 * A plume instance is compose(pos, q, (r, r, len)) in hull space, so it
 * starts on the nozzle mouth and streams out along the mount's axis.
 */
export function jetFrames(mounts) {
  return mounts.map(m => {
    const pos = new THREE.Vector3(vx(m.pos, 0), vx(m.pos, 1), vx(m.pos, 2));
    const dir = new THREE.Vector3(vx(m.dir, 0), vx(m.dir, 1), vx(m.dir, 2)).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(_zAxis, dir);
    const u = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    const v = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    return { pos, dir, q, u, v };
  });
}

/**
 * Hull-space plume instance of one jet (layer 0 core, 1 sheath) at
 * effective intensity Ie: length and radius from the block-sized dims.
 */
export function jetInstanceMatrix(frame, dims, layer, Ie, out, _s = new THREE.Vector3()) {
  const len = dims.len * (layer ? 1.0 : 0.75) * (0.55 + 0.45 * Ie);
  const rad = layer ? dims.sheath : dims.core;
  return out.compose(frame.pos, frame.q, _s.set(rad, rad, len));
}

/**
 * Scene-side RCS renderer. One instanced cone mesh for every jet in the
 * battle and one instanced billboard mesh for puffs and nozzle flashes,
 * both positioned relative to the camera so float32 stays precise at
 * battle distances. No lights.
 */
export class RcsSystem {
  constructor(ctx) {
    this.ctx = ctx;
    this.planners = new Map();
    this.sources = new Map();
    this.enabled = true;

    const jetMat = new THREE.ShaderMaterial({
      uniforms: {
        uNoise: { value: getNoiseTexture() },
        uTime: { value: 0 },
        uCore: { value: new THREE.Color(1.0, 0.98, 1.0) },
        uSheath: { value: new THREE.Color(0.55, 0.75, 1.0) },
        uEdge: { value: new THREE.Color(0.25, 0.4, 0.95) }
      },
      vertexShader: jetVertex,
      fragmentShader: jetFragment,
      side: THREE.DoubleSide,
      ...ADDITIVE
    });
    const jetGeom = jetGeometry();
    this.jetAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_JETS * 4), 4);
    this.jetAttr.setUsage(THREE.DynamicDrawUsage);
    jetGeom.setAttribute('aJet', this.jetAttr);
    this.jets = new THREE.InstancedMesh(jetGeom, jetMat, MAX_JETS);
    this.jets.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.jets.frustumCulled = false;
    this.jets.renderOrder = 22;
    this.jets.count = 0;
    this.jets.name = 'rcsJets';

    const base = new THREE.PlaneGeometry(1, 1);
    const pg = new THREE.InstancedBufferGeometry();
    pg.index = base.index;
    pg.setAttribute('position', base.attributes.position);
    pg.setAttribute('uv', base.attributes.uv);
    this.puffPos = new THREE.InstancedBufferAttribute(new Float32Array(MAX_PUFFS * 4), 4);
    this.puffCol = new THREE.InstancedBufferAttribute(new Float32Array(MAX_PUFFS * 4), 4);
    this.puffPos.setUsage(THREE.DynamicDrawUsage);
    this.puffCol.setUsage(THREE.DynamicDrawUsage);
    pg.setAttribute('aPos', this.puffPos);
    pg.setAttribute('aCol', this.puffCol);
    pg.instanceCount = 0;
    const puffMat = new THREE.ShaderMaterial({
      uniforms: {
        uPuff: { value: getPuffTexture(1) },
        uGlow: { value: getGlowTexture() }
      },
      vertexShader: puffVertex,
      fragmentShader: puffFragment,
      side: THREE.DoubleSide,
      ...ADDITIVE
    });
    this.puffs = new THREE.Mesh(pg, puffMat);
    this.puffs.frustumCulled = false;
    this.puffs.renderOrder = 26;
    this.puffs.name = 'rcsPuffs';

    // Visible at construction so SceneManager.warmUp compiles both programs
    ctx.scene.add(this.jets);
    ctx.scene.add(this.puffs);

    this._origin = new THREE.Vector3();
    this._m = new THREE.Matrix4();
    this._m2 = new THREE.Matrix4();
    this._shipM = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._qc = new THREE.Quaternion();
    this._qe = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._w = new THREE.Vector3();
    this._p = new THREE.Vector3();
    this._s = newSample();
    this._bx = [0, 0, 0];
    this._by = [0, 0, 0];
    this._basis = new THREE.Matrix4();
    this._X = new THREE.Vector3();
    this._Y = new THREE.Vector3();
    this._Z = new THREE.Vector3();
    this._one = new THREE.Vector3(1, 1, 1);
    this._nj = 0;   // _updateShip's running instance counts
    this._np = 0;
  }

  /**
   * Drive a ship's jets from a scripted source instead of the recording
   * (showroom hulls). source.sample(t, out) fills a newSample()-shaped
   * object; pass null to go back to the trace.
   */
  setSource(shipId, source) {
    if (source) this.sources.set(shipId, source); else this.sources.delete(shipId);
    const p = this.planners.get(shipId);
    if (p) { p.source = source || null; p.cells.clear(); }
  }

  /** Hull-local jets for a ship: the hull's own mounts, else raycast quads. */
  _mountsFor(ship) {
    const ud = ship.userData;
    if (Array.isArray(ud.rcsMounts) && ud.rcsMounts.length) {
      return ud.rcsMounts.map(m => ({
        pos: [vx(m.pos, 0), vx(m.pos, 1), vx(m.pos, 2)],
        dir: new THREE.Vector3(vx(m.dir, 0), vx(m.dir, 1), vx(m.dir, 2)).normalize().toArray()
      }));
    }
    const L = ud.size?.length || 8;
    const W = ud.size?.width || 1.6;
    let meshes = ship.children.filter(c => c.isMesh && c.name.startsWith('hull_'));
    if (!meshes.length) meshes = ship.children.filter(c => c.isMesh);
    const box = new THREE.Box3();
    for (const m of meshes) {
      if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
      box.union(m.geometry.boundingBox);
    }
    let zN = 0.4 * L, zT = -0.4 * L;
    if (!box.isEmpty()) {
      const len = box.max.z - box.min.z;
      zN = box.max.z - 0.12 * len;
      zT = box.min.z + 0.15 * len;
    }
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const proxies = meshes.map(m => new THREE.Mesh(m.geometry, mat));
    const ray = new THREE.Raycaster();
    const far = W * 4 + 1;
    const radius = (z, d) => {
      if (!proxies.length) return null;
      for (const zz of [z, z * 0.9, z * 0.8]) {
        ray.set(new THREE.Vector3(d[0] * far, d[1] * far, zz), new THREE.Vector3(-d[0], -d[1], 0));
        ray.far = far;
        const hit = ray.intersectObjects(proxies, false)[0];
        if (hit) return (far - hit.distance) + W * 0.015;
      }
      return null;
    };
    const mounts = defaultRcsMounts(L, W, zN, zT, radius);
    mat.dispose();
    return mounts;
  }

  _plannerFor(shipId, ship) {
    let p = this.planners.get(shipId);
    if (p && p.group === ship) return p;
    const mounts = this._mountsFor(ship);
    // Plumes sized from the thruster block, which is one physical size on
    // every hull: a dreadnought's jets match a corvette's. Nozzles whose
    // plume would run into the hull's own fittings never fire
    const dims = jetDims(ship.userData.size?.length || 8);
    const roles = jetRoles(mounts, jetClearance(mounts, ship.userData.parts, dims.len), dims.len);
    p = new RcsPlanner({ shipId, shipType: ship.userData.shipType, mounts, roles, source: this.sources.get(shipId) });
    p.group = ship;
    p.dims = dims;
    p.jetFrames = jetFrames(mounts);
    this.planners.set(shipId, p);
    return p;
  }

  /**
   * @param {Map<string, THREE.Group>} ships  SceneManager.ships
   * @param {Array} trace                     recording sim_trace
   * @param {number} time                     battle time
   */
  update(ships, trace, time) {
    let nj = 0, np = 0;
    const cam = this.ctx.camera;
    this._origin.copy(cam.position);
    this.jets.position.copy(this._origin);
    this.puffs.position.copy(this._origin);
    this.jets.material.uniforms.uTime.value = performance.now() * 0.001;

    if (this.enabled && ships) {
      for (const [shipId, ship] of ships) {
        if (!ship.visible) continue;
        const planner = this._plannerFor(shipId, ship);
        if (!planner.source) {
          if (!trace || !trace.length) continue;
          planner.setTrace(trace);
        }
        this._updateShip(planner, ship, time, nj, np);
        nj = this._nj; np = this._np;
      }
    }

    this.jets.count = nj;
    this.jets.visible = nj > 0;
    if (nj > 0) {
      this.jets.instanceMatrix.needsUpdate = true;
      this.jetAttr.needsUpdate = true;
    }
    this.puffs.geometry.instanceCount = np;
    this.puffs.visible = np > 0;
    if (np > 0) {
      this.puffPos.needsUpdate = true;
      this.puffCol.needsUpdate = true;
    }
  }

  /** Recorded body orientation at time t (lookAt basis of the traced heading). */
  _traceQuat(planner, t, out) {
    const s = planner.sample(t, this._s);
    lookAtBasis(s.fwd, this._bx, this._by);
    this._X.fromArray(this._bx);
    this._Y.fromArray(this._by);
    this._Z.fromArray(s.fwd);
    this._basis.makeBasis(this._X, this._Y, this._Z);
    return out.setFromRotationMatrix(this._basis);
  }

  /** Write a ship's plumes from instance nj and puffs from np; the new counts land in _nj/_np. */
  _updateShip(planner, ship, time, nj, np) {
    const n = planner.jetCount;
    const frames = planner.jetFrames;
    const dims = planner.dims;
    const jl = dims.len;
    const pj = dims.puff;
    this._shipM.compose(ship.position, ship.quaternion, this._one);
    this._shipM.setPosition(
      ship.position.x - this._origin.x,
      ship.position.y - this._origin.y,
      ship.position.z - this._origin.z);

    // --- live jets: cones + nozzle flashes ---------------------------------
    const jm = this.jets.instanceMatrix.array;
    const ja = this.jetAttr.array;
    const pp = this.puffPos.array;
    const pc = this.puffCol.array;
    for (let k = 0; k < n; k++) {
      const I = planner.intensity(time, k);
      if (I < 0.01) continue;
      const fl = planner.flicker(time, k);
      const Ie = I * fl;
      const fr = frames[k];
      for (let layer = 0; layer < 2 && nj < MAX_JETS; layer++) {
        jetInstanceMatrix(fr, dims, layer, Ie, this._m, this._v);
        this._m2.multiplyMatrices(this._shipM, this._m);
        this._m2.toArray(jm, nj * 16);
        ja[nj * 4] = Ie;
        ja[nj * 4 + 1] = layer;
        ja[nj * 4 + 2] = (k * 0.37 + planner.seed * 0.001) % 1;
        ja[nj * 4 + 3] = 0;
        nj++;
      }
      if (np < MAX_PUFFS) {
        // nozzle flash (negative size = glow texture)
        this._p.copy(fr.dir).multiplyScalar(dims.block * 0.15).add(fr.pos).applyMatrix4(this._shipM);
        pp[np * 4] = this._p.x; pp[np * 4 + 1] = this._p.y; pp[np * 4 + 2] = this._p.z;
        pp[np * 4 + 3] = -dims.flash * (0.5 + 0.5 * Ie);
        const g = 1.4 * Ie;
        pc[np * 4] = 0.75 * g; pc[np * 4 + 1] = 0.85 * g; pc[np * 4 + 2] = 1.0 * g;
        pc[np * 4 + 3] = 0;
        np++;
      }
    }

    // --- puffs: re-derived from every emission slot still alive ------------
    const dt = RCS_TIMING.puffSlot;
    const life = RCS_TIMING.puffLife;
    const j0 = Math.ceil((time - life) / dt);
    const j1 = Math.floor(time / dt);
    this._nj = nj;
    this._np = np;
    if (j1 < j0) return;
    // correction from the traced to the rendered orientation (they differ
    // only if the interpolator smooths differently); applied to past poses
    this._traceQuat(planner, time, this._qc);
    this._qc.invert().premultiply(ship.quaternion);
    for (let j = j1; j >= j0 && np < MAX_PUFFS; j--) {
      const te = j * dt;
      const age = time - te;
      let qReady = false;
      for (let k = 0; k < n && np < MAX_PUFFS; k++) {
        const I = planner.intensity(te, k);
        if (I < 0.06) continue;
        const h1 = hash01(planner.seed, j, k * 5 + 1);
        const lifeK = life * (0.55 + 0.45 * h1);
        if (age > lifeK) continue;
        if (!qReady) {
          this._traceQuat(planner, te, this._qe);
          this._qe.premultiply(this._qc);
          qReady = true;
        }
        const h2 = hash01(planner.seed, j, k * 5 + 2);
        const h3 = hash01(planner.seed, j, k * 5 + 3);
        const h4 = hash01(planner.seed, j, k * 5 + 4);
        const fr = frames[k];
        // drag-slowed exhaust streaming down the nozzle axis, little scatter
        const kd = 3.0;
        const v0 = pj * (4.2 + 1.8 * h2);
        const s = v0 * (1 - Math.exp(-kd * age)) / kd;
        const ang = h3 * Math.PI * 2;
        const spread = 0.12 * h4;
        this._w.copy(fr.dir)
          .addScaledVector(fr.u, Math.cos(ang) * spread)
          .addScaledVector(fr.v, Math.sin(ang) * spread);
        this._p.copy(fr.pos).addScaledVector(this._w, s + jl * 0.1).applyQuaternion(this._qe);
        const x = this._p.x + ship.position.x - this._origin.x;
        const y = this._p.y + ship.position.y - this._origin.y;
        const z = this._p.z + ship.position.z - this._origin.z;
        const f = age / lifeK;
        const size = pj * (0.16 + 0.6 * (1 - Math.exp(-3.2 * age))) * (0.8 + 0.4 * h2);
        const env = Math.min(1, age / 0.04) * Math.pow(1 - f, 1.8);
        const a = I * env * 0.22;
        pp[np * 4] = x; pp[np * 4 + 1] = y; pp[np * 4 + 2] = z; pp[np * 4 + 3] = size;
        pc[np * 4] = a * (0.95 - 0.4 * f);
        pc[np * 4 + 1] = a * (0.97 - 0.33 * f);
        pc[np * 4 + 2] = a * (1.0 - 0.2 * f);
        pc[np * 4 + 3] = h1 * 6.283 + age * (h3 - 0.5) * 1.5;
        np++;
      }
    }
    this._np = np;
  }

  dispose() {
    this.ctx.scene.remove(this.jets);
    this.ctx.scene.remove(this.puffs);
    this.jets.geometry.dispose();
    this.jets.material.dispose();
    this.puffs.geometry.dispose();
    this.puffs.material.dispose();
  }
}
