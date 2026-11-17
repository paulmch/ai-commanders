import * as THREE from 'three';
import HULL_MOUNTS from './hullMountsData.js';
import { AZ_CELLS, EL_CELLS, CELL_DEG, cellIndex, unpackMask, TRAVERSE_DEG_S } from './MountLimits.js';
import { shipStateAt } from '../DamageTimeline.js';
import { hullBasis } from '../HullFrame.js';

/**
 * Weapon arcs and turret traverse in the viewer.
 *
 * data/hull_mounts.json (here as render/hullMountsData.js) gives every
 * mount of every hull class a blocked-direction mask in the BODY frame
 * (x forward, y port, z dorsal). The hull kit's local frame is +Z nose,
 * +X port, +Y dorsal, so
 *
 *     hull-local (x, y, z) = body (y, z, x)
 *     azimuth = atan2(local.x, local.z), elevation = asin(local.y)
 *
 * Two sources drive a mount's pose (HullRigs.js):
 *   - recordings with the per-frame `tur` field: the engine's own aim
 *     [az, el, state] per mount, slerped between frames (turAim);
 *   - older recordings: the viewer's aim from the event log, slewed at the
 *     engine's class rates along the great circle and clamped to the mask
 *     (clampToArc), all as pure functions of battle time.
 * Everything here is stateless or cached from static data, so a paused
 * frame, a replayed frame and a direct seek agree.
 */

const D2R = Math.PI / 180;
const N = AZ_CELLS * EL_CELLS;

/** Hull-local unit vector of a body-frame (azimuth, elevation) in degrees. */
export function azElToLocal(az, el, out) {
  const ce = Math.cos(el * D2R);
  return out.set(ce * Math.sin(az * D2R), Math.sin(el * D2R), ce * Math.cos(az * D2R));
}

/** Grid cell of a hull-local direction (need not be unit length). */
export function cellOf(v) {
  const l = v.length() || 1;
  return cellIndex(Math.atan2(v.x, v.z) / D2R, Math.asin(Math.max(-1, Math.min(1, v.y / l))) / D2R);
}

let _cellDirs = null;
/** Hull-local centre direction of every grid cell (Float32Array 3N). */
function cellDirs() {
  if (_cellDirs) return _cellDirs;
  _cellDirs = new Float32Array(3 * N);
  const v = new THREE.Vector3();
  for (let j = 0; j < EL_CELLS; j++) {
    for (let i = 0; i < AZ_CELLS; i++) {
      azElToLocal(-180 + (i + 0.5) * CELL_DEG, -90 + (j + 0.5) * CELL_DEG, v).toArray(_cellDirs, 3 * (j * AZ_CELLS + i));
    }
  }
  return _cellDirs;
}

const _arcs = new Map();
/**
 * Arcs of a hull class: {mounts: [{index, slot, kind, type, rate (rad/s),
 * mask (Uint8Array, 1 = blocked), axis (hull-local Vector3)}], bySlot}, in
 * data/hull_mounts.json order (the order of the trace's `tur` list).
 * null for a class the file does not know.
 */
export function arcsFor(shipType, data = HULL_MOUNTS) {
  const key = data === HULL_MOUNTS ? shipType : null;
  if (key && _arcs.has(key)) return _arcs.get(key);
  const h = data.hulls?.[shipType];
  let arcs = null;
  if (h) {
    const mounts = h.mounts.map((m, index) => ({
      index, slot: m.slot, kind: m.kind, type: m.type,
      rate: (TRAVERSE_DEG_S[m.type] || 0) * D2R,
      mask: unpackMask(m.blocked),
      axis: new THREE.Vector3(m.axis[1], m.axis[2], m.axis[0]),
      near: null
    }));
    arcs = { mounts, bySlot: new Map(mounts.map(m => [m.slot, m])), pds: mounts.filter(m => m.kind === 'pd') };
  }
  if (key) _arcs.set(key, arcs);
  return arcs;
}

/** Is the hull-local direction v blocked for this mount? */
export function isBlocked(mount, v) {
  return mount.mask[cellOf(v)] === 1;
}

/**
 * v if the mount can fire along it, else the centre of the nearest open
 * cell (the edge of its arc). Returns true when it had to clamp.
 */
export function clampToArc(mount, v, out) {
  const k = cellOf(v);
  if (!mount.mask[k]) { out.copy(v); return false; }
  if (!mount.near) mount.near = new Int16Array(N).fill(-1);
  let best = mount.near[k];
  const C = cellDirs();
  if (best < 0) {
    // nearest open cell to this cell's centre (cached per cell)
    const cx = C[3 * k], cy = C[3 * k + 1], cz = C[3 * k + 2];
    let bd = -2;
    for (let c = 0; c < N; c++) {
      if (mount.mask[c]) continue;
      const d = cx * C[3 * c] + cy * C[3 * c + 1] + cz * C[3 * c + 2];
      if (d > bd) { bd = d; best = c; }
    }
    if (best < 0) { out.copy(v); return true; }   // nothing open at all
    mount.near[k] = best;
  }
  out.set(C[3 * best], C[3 * best + 1], C[3 * best + 2]);
  return true;
}

/** Unit vector `w` of the way from a to b along the great circle. */
export function slewDir(a, b, w, out) {
  if (!(w > 0)) return out.copy(a);
  if (w >= 1) return out.copy(b);
  const th = a.angleTo(b);
  if (th < 1e-5) return out.copy(b);
  const s = Math.sin(th);
  if (s < 1e-5) {
    // antipodal: swing through the mount's own "up" side, any fixed normal
    _n.set(a.z, a.x, a.y).cross(a).normalize();
    return out.copy(a).applyAxisAngle(_n, th * w);
  }
  const ka = Math.sin((1 - w) * th) / s, kb = Math.sin(w * th) / s;
  return out.set(a.x * ka + b.x * kb, a.y * ka + b.y * kb, a.z * ka + b.z * kb).normalize();
}
const _n = new THREE.Vector3();

/**
 * Slew progress 0..1 toward a new aim that must be reached at `tArrive`:
 * the mount leaves `angle / rate` seconds earlier and turns at its class
 * rate, so it is on the new target exactly when it fires.
 */
export function slewProgress(angle, rate, tArrive, t) {
  if (!(rate > 0) || angle < 1e-5) return t >= tArrive ? 1 : 0;
  return THREE.MathUtils.clamp(1 - (tArrive - t) / (angle / rate), 0, 1);
}

/** Yaw / pitch (out.x / out.y) of a mount rolled `rz` about Z that aims along hull-local v. */
export function mountAngles(rz, v, out) {
  const c = Math.cos(rz), s = Math.sin(rz);
  const x = v.x * c + v.y * s, y = -v.x * s + v.y * c;
  return out.set(Math.atan2(x, v.z), Math.atan2(y, Math.hypot(x, v.z)));
}

/** Hull-local aim of a mount rolled `rz` about Z at (yaw, pitch): inverse of mountAngles. */
export function mountDir(rz, yaw, pitch, out) {
  const cp = Math.cos(pitch);
  const x = Math.sin(yaw) * cp, y = Math.sin(pitch), z = Math.cos(yaw) * cp;
  const c = Math.cos(rz), s = Math.sin(rz);
  return out.set(x * c - y * s, x * s + y * c, z);
}

const _da = new THREE.Vector3(), _db = new THREE.Vector3();
/**
 * Recorded aim of mount `index` between two trace frames: tur = {a, b, f}
 * (frame0's list, frame1's list, blend 0..1). Writes the hull-local aim to
 * `out` and returns the mount state (0 idle, 1 slewing, 2 on target and
 * clear, 3 masked by own hull), or -1 when the recording has no entry.
 * The engine slews along the great circle, so the slerp between two
 * frames is the path the mount took.
 */
export function turAim(tur, index, out) {
  const ea = tur?.a?.[index], eb = tur?.b?.[index];
  const e0 = ea || eb, e1 = eb || ea;
  if (!e0 || !(e0.length >= 2)) return -1;
  azElToLocal(e0[0], e0[1], _da);
  azElToLocal(e1[0], e1[1], _db);
  slewDir(_da, _db, tur.f || 0, out);
  // discrete state switches at its frame, like the Interpolator's flags
  const cur = (tur.f || 0) < 1 ? e0 : e1;
  return cur[2] | 0;
}

/** Does this recording carry the engine's turret aims? */
export function hasTur(loader) {
  if (!loader) return false;
  const trace = loader.simTrace || [];
  if (loader._turTrace === trace && loader._turN === trace.length) return loader._hasTur;
  let has = false;
  for (let i = 0; i < trace.length && !has; i++) {
    const ships = trace[i].ships;
    if (!ships) continue;
    for (const id in ships) { if (Array.isArray(ships[id].tur)) { has = true; break; } }
    if (Object.keys(ships).length) break;      // the first frame with ships decides
  }
  loader._turTrace = trace;
  loader._turN = trace.length;
  loader._hasTur = has;
  return has;
}

/**
 * Which PD turret shows each dwell of one ship, for recordings made before
 * the engine knew arcs (it fired through its own hull): a dwell keeps its
 * recorded turret when that mount has a clear line at the dwell's start,
 * else moves to a PD mount that has (one not already burning another round
 * if possible, else the one facing the round best); with no clear mount it
 * stays put and is flagged `masked` (the beam is then drawn from outside
 * the hull). Pure: depends only on the segments and directions given.
 *
 * @param {Array} segs    this ship's dwell segments {turret, start, end}
 * @param {Array} pds     the hull's PD arcs (arcsFor(...).mounts, kind 'pd')
 * @param {Function} dirOf  seg -> hull-local unit Vector3 toward the round
 *                          at seg.start, or null when unknown
 * @returns {Array} [{turret, masked}] aligned with segs
 */
export function assignPdSegments(segs, pds, dirOf) {
  const order = segs.map((s, i) => i).sort((a, b) => segs[a].start - segs[b].start || a - b);
  const out = new Array(segs.length);
  const busy = new Map();     // turret -> [[start, end], ...]
  const free = (name, s) => !(busy.get(name) || []).some(([a, b]) => s.start <= b && s.end >= a);
  for (const i of order) {
    const seg = segs[i];
    const d = dirOf(seg);
    const own = pds.find(m => m.slot === seg.turret);
    let pick = seg.turret, masked = false;
    if (d && own && isBlocked(own, d)) {
      const clear = pds.filter(m => !isBlocked(m, d));
      if (clear.length) {
        const facing = (m) => m.axis.dot(d);
        const idle = clear.filter(m => free(m.slot, seg));
        pick = (idle.length ? idle : clear).reduce((b, m) => (facing(m) > facing(b) ? m : b)).slot;
      } else {
        masked = true;
      }
    }
    if (!busy.has(pick)) busy.set(pick, []);
    busy.get(pick).push([seg.start, seg.end]);
    out[i] = { turret: pick, masked };
  }
  return out;
}

/**
 * The PD turret a dwell is shown from at one moment, given the hull-local
 * direction `d` to its round right now: the planned turret (`planned`)
 * while it has a clear line; when the round has moved behind the hull for
 * it, the PD mount facing the round best that does have one; null when no
 * mount can bear (the beam then starts outside the hull). Pure.
 * Recordings with `tur` never need it: the engine only fires when clear.
 */
export function shownPdMount(planned, pds, d) {
  const own = pds.find(m => m.slot === planned);
  if (!own || !isBlocked(own, d)) return planned;
  let best = null, bd = -2;
  for (const m of pds) {
    if (isBlocked(m, d)) continue;
    const f = m.axis.dot(d);
    if (f > bd) { bd = f; best = m; }
  }
  return best ? best.slot : null;
}

const _st = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false };
const _bx = [0, 0, 0], _by = [0, 0, 0];
const _p = new THREE.Vector3();

/**
 * Stamp every PD dwell segment of the loaded recording with the turret
 * that shows it (seg.arcTurret, seg.arcMasked) and index them per ship and
 * turret (returned: Map shipId -> Map turret -> segments sorted by start).
 * Recordings with `tur` keep the engine's turret. Recomputed only when the
 * segment list grows (live mode); the result depends on the recording
 * alone, never on what was rendered before.
 */
export function pdPlan(manager) {
  const loader = manager?.loader;
  const segs = loader?.pdBeamSegments;
  if (!segs) return null;
  const weapons = manager.weapons;
  const ready = !!weapons?.positionAt && !!weapons.timeline;
  const cached = loader._pdPlan;
  if (cached && cached.segs === segs && cached.n === segs.length && cached.ready === ready
    && cached.ships === manager.ships.size) return cached.byShip;
  const engineArcs = hasTur(loader);
  const SCALE = manager.SCALE || 1;
  const trace = loader.simTrace || [];
  const perShip = new Map();
  for (const s of segs) {
    if (!perShip.has(s.shipId)) perShip.set(s.shipId, []);
    perShip.get(s.shipId).push(s);
  }
  const byShip = new Map();
  for (const [shipId, list] of perShip) {
    const ship = manager.ships.get(shipId);
    const arcs = ship && !engineArcs && ready ? arcsFor(ship.userData.shipType) : null;
    const pds = arcs ? arcs.mounts.filter(m => m.kind === 'pd') : null;
    let res = null;
    if (pds && pds.length) {
      res = assignPdSegments(list, pds, (seg) => {
        const st = shipStateAt(trace, shipId, seg.start, _st);
        const at = st && weapons.positionAt(seg.targetType, seg.targetId, seg.start, _p);
        if (!st || !at) return null;
        // world offset to the round (scene units), into the hull frame
        const dx = at.x - st.pos[0] * SCALE, dy = at.y - st.pos[1] * SCALE, dz = at.z - st.pos[2] * SCALE;
        hullBasis(st.fwd, st.up, _bx, _by);
        const v = new THREE.Vector3(
          dx * _bx[0] + dy * _bx[1] + dz * _bx[2],
          dx * _by[0] + dy * _by[1] + dz * _by[2],
          dx * st.fwd[0] + dy * st.fwd[1] + dz * st.fwd[2]);
        return v.lengthSq() > 0 ? v.normalize() : null;
      });
    }
    const byTurret = new Map();
    list.forEach((seg, i) => {
      const name = res ? res[i].turret : seg.turret;
      if (seg.arcTurret !== name) { seg.arcTurret = name; seg.mountShip = null; }   // drop Projectiles' mount cache
      seg.arcMasked = res ? res[i].masked : false;
      if (!byTurret.has(name)) byTurret.set(name, []);
      byTurret.get(name).push(seg);
    });
    for (const l of byTurret.values()) l.sort((a, b) => a.start - b.start);
    byShip.set(shipId, byTurret);
  }
  loader._pdPlan = { segs, n: segs.length, ready, ships: manager.ships.size, byShip };
  return byShip;
}

/**
 * Distance along a hull-local ray (origin o, unit d) to where it leaves
 * the hull's bounding ellipsoid (semi-axes 0.8 W, 0.8 W, 0.6 L): where a
 * beam that has no clear mount is allowed to appear. 0 when already out.
 */
export function hullExit(o, d, W, L) {
  const ax = 0.8 * W, az = 0.6 * L;
  const ox = o.x / ax, oy = o.y / ax, oz = o.z / az;
  const dx = d.x / ax, dy = d.y / ax, dz = d.z / az;
  const a = dx * dx + dy * dy + dz * dz, b = 2 * (ox * dx + oy * dy + oz * dz), c = ox * ox + oy * oy + oz * oz - 1;
  const disc = b * b - 4 * a * c;
  if (disc <= 0 || a <= 0) return 0;
  return Math.max(0, (-b + Math.sqrt(disc)) / (2 * a));
}
