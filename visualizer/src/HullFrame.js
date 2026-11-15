/**
 * The hull's orientation frame from the recorded attitude.
 *
 * Hull-local axes (render/Hulls.js): +Z nose, +Y dorsal, +X port. The
 * engine's body frame is x forward, z up (dorsal), y = z cross x (port),
 * so viewer (X, Y, Z) = body (y, z, x).
 *
 * Recordings from the weapon-arcs engine carry the ship's `up` vector next
 * to `fwd` in every trace frame: the hull is then oriented by both (its
 * real roll). Older recordings have only `fwd`; they keep the roll the
 * viewer always invented, three.js Object3D.lookAt with world-up +Y
 * (x = normalize(+Y cross fwd), y = fwd cross x).
 *
 * Plain arrays, no three.js: every consumer that rebuilds the hull frame
 * from a trace sample (DamageTimeline, Rcs, Projectiles, SceneManager)
 * goes through hullBasis so they cannot disagree about the roll.
 */

/** A usable recorded up vector (finite, non-zero), else null. */
export function validUp(up) {
  if (!up || up.length < 3) return null;
  const l = up[0] * up[0] + up[1] * up[1] + up[2] * up[2];
  return Number.isFinite(l) && l > 1e-12 ? up : null;
}

/**
 * Port (x) and dorsal (y) world axes of a hull whose nose points along the
 * unit vector `fwd`, honouring `up` when the recording has it.
 * Returns true when the recorded roll was used.
 */
export function hullBasis(fwd, up, x, y) {
  let zx = fwd[0], zy = fwd[1], zz = fwd[2];
  if (validUp(up)) {
    // dorsal = up made perpendicular to the nose
    const d = up[0] * zx + up[1] * zy + up[2] * zz;
    let yx = up[0] - d * zx, yy = up[1] - d * zy, yz = up[2] - d * zz;
    const yl = Math.hypot(yx, yy, yz);
    if (yl > 1e-6) {
      yx /= yl; yy /= yl; yz /= yl;
      y[0] = yx; y[1] = yy; y[2] = yz;
      x[0] = yy * zz - yz * zy;                // port = dorsal x nose
      x[1] = yz * zx - yx * zz;
      x[2] = yx * zy - yy * zx;
      return true;
    }
  }
  let xx = zz, xz = -zx;                       // (0,1,0) x z
  if (xx * xx + xz * xz === 0) {
    zz += 0.0001;
    const l = Math.hypot(zx, zy, zz);
    zx /= l; zy /= l; zz /= l;
    xx = zz; xz = -zx;
  }
  const xl = Math.hypot(xx, xz);
  xx /= xl; xz /= xl;
  x[0] = xx; x[1] = 0; x[2] = xz;
  y[0] = zy * xz;
  y[1] = zz * xx - zx * xz;
  y[2] = -zy * xx;
  return false;
}

const _x = [0, 0, 0], _y = [0, 0, 0];
/**
 * Quaternion [x, y, z, w] of the hull frame (columns port, dorsal, nose).
 * `fwd` must be unit length.
 */
export function hullQuat(fwd, up, out = [0, 0, 0, 1]) {
  hullBasis(fwd, up, _x, _y);
  // rotation matrix columns: X = _x, Y = _y, Z = fwd
  const m00 = _x[0], m10 = _x[1], m20 = _x[2];
  const m01 = _y[0], m11 = _y[1], m21 = _y[2];
  const m02 = fwd[0], m12 = fwd[1], m22 = fwd[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    out[3] = 0.25 / s; out[0] = (m21 - m12) * s; out[1] = (m02 - m20) * s; out[2] = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    out[3] = (m21 - m12) / s; out[0] = 0.25 * s; out[1] = (m01 + m10) / s; out[2] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    out[3] = (m02 - m20) / s; out[0] = (m01 + m10) / s; out[1] = 0.25 * s; out[2] = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    out[3] = (m10 - m01) / s; out[0] = (m02 + m20) / s; out[1] = (m12 + m21) / s; out[2] = 0.25 * s;
  }
  return out;
}

/** Shortest-arc slerp of two unit quaternions [x, y, z, w]. */
export function slerpQuat(a, b, t, out = [0, 0, 0, 1]) {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let c = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (c < 0) { c = -c; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let ka = 1 - t, kb = t;
  if (c < 0.9995) {
    const th = Math.acos(c), s = Math.sin(th);
    ka = Math.sin((1 - t) * th) / s; kb = Math.sin(t * th) / s;
  }
  const x = a[0] * ka + bx * kb, y = a[1] * ka + by * kb, z = a[2] * ka + bz * kb, w = a[3] * ka + bw * kb;
  const l = Math.hypot(x, y, z, w) || 1;
  out[0] = x / l; out[1] = y / l; out[2] = z / l; out[3] = w / l;
  return out;
}

const _q0 = [0, 0, 0, 1], _q1 = [0, 0, 0, 1], _q = [0, 0, 0, 1];
const _n0 = [0, 0, 1], _n1 = [0, 0, 1];
function unit(v, out) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  out[0] = v[0] / l; out[1] = v[1] / l; out[2] = v[2] / l;
  return out;
}

/**
 * Attitude between two trace frames that both carry `up`: the two hull
 * frames are slerped as rotations (nose and roll turn together), and the
 * blended nose and dorsal axes are written to outFwd / outUp.
 */
export function interpAttitude(fwd0, up0, fwd1, up1, t, outFwd, outUp) {
  hullQuat(unit(fwd0, _n0), up0, _q0);
  hullQuat(unit(fwd1, _n1), up1, _q1);
  const q = slerpQuat(_q0, _q1, t, _q);
  const x = q[0], y = q[1], z = q[2], w = q[3];
  // rotate (0,0,1) and (0,1,0)
  outFwd[0] = 2 * (x * z + w * y); outFwd[1] = 2 * (y * z - w * x); outFwd[2] = 1 - 2 * (x * x + y * y);
  outUp[0] = 2 * (x * y - w * z); outUp[1] = 1 - 2 * (x * x + z * z); outUp[2] = 2 * (y * z + w * x);
  return q;
}

/**
 * `up` between two trace states for consumers that lerp `fwd` themselves:
 * lerped and normalised (hullBasis re-squares it against the nose); null
 * when the recording has none.
 */
export function lerpUp(up0, up1, t, out) {
  const a = validUp(up0), b = validUp(up1) || a;
  if (!a && !b) return null;
  const p = a || b;
  out[0] = p[0] + (b[0] - p[0]) * t; out[1] = p[1] + (b[1] - p[1]) * t; out[2] = p[2] + (b[2] - p[2]) * t;
  const l = Math.hypot(out[0], out[1], out[2]);
  if (l < 1e-9) return null;
  out[0] /= l; out[1] /= l; out[2] /= l;
  return out;
}
