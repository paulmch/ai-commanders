import * as THREE from 'three';
import { updateHullLights } from './HullLights.js';

/**
 * Per-frame hull animation driven by the recording, all on BATTLE time so
 * paused frames and timeline seeks are deterministic:
 *
 *   - coilgun turrets train on the ship's actual target (last shot fired,
 *     else the captain's ordered target, else the nearest enemy) and
 *     recoil when their weapon slot fires
 *   - PD ball turrets slew onto the torpedo / slug their dwell segment is
 *     burning (pd_fired turret_name -> pd rig), else sweep a search pattern
 *   - torpedo launch doors open around each torpedo_launched event
 *   - navigation lamps blink (HullLights)
 *
 * The scene may carry `scene.loader` (BattleLoader); without it (showroom,
 * tests) everything falls back to nearest-enemy aim and idle motion.
 */

const _m = new THREE.Matrix4();
const _v = new THREE.Vector3();
const _Z = new THREE.Vector3(0, 0, 1);
const _indexCache = new WeakMap();

/** Per-ship shot and torpedo-launch timelines from the event log. */
export function shipEventIndex(events) {
  const shots = new Map();
  const launches = new Map();
  const push = (m, k, v) => { if (!m.has(k)) m.set(k, []); m.get(k).push(v); };
  for (const e of events || []) {
    if (e.event_type === 'shot_fired') {
      push(shots, e.ship_id, { t: e.timestamp, target: e.data?.target_id, slot: e.data?.weapon_slot });
    } else if (e.event_type === 'torpedo_launched') {
      push(launches, e.ship_id, { t: e.timestamp, target: e.data?.target_id });
    }
  }
  for (const m of [shots, launches]) for (const a of m.values()) a.sort((x, y) => x.t - y.t);
  return { shots, launches, count: (events || []).length };
}

function indexFor(loader) {
  if (!loader || !loader.events) return null;
  let idx = _indexCache.get(loader);
  if (!idx || idx.count !== loader.events.length) {
    idx = shipEventIndex(loader.events);
    _indexCache.set(loader, idx);
  }
  return idx;
}

/** Index of the last entry with t <= time, or -1. */
export function lastIndexBefore(list, time) {
  if (!list || !list.length || list[0].t > time) return -1;
  let lo = 0, hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (list[mid].t <= time) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Barrel recoil 0..1 for a gun that fired `age` battle seconds ago. */
export function recoil(age) {
  if (!(age >= 0) || age > 1.4) return 0;
  if (age < 0.05) return age / 0.05;
  return Math.exp(-(age - 0.05) * 4.0);
}

/**
 * Launch-door openness 0..1: doors crack 2.5 s before the launch, are
 * fully open 1.5 s before it and close again 2.5-3.5 s after.
 */
export function doorOpen(times, t) {
  let o = 0;
  for (const tl of times) {
    const a = t - tl;
    if (a < -2.5 || a > 3.5) continue;
    const v = a < -1.5 ? a + 2.5 : a <= 2.5 ? 1 : 1 - (a - 2.5);
    if (v > o) o = v;
  }
  return Math.max(0, Math.min(1, o));
}

/** Split a ship's launches across its door banks, round robin. */
export function assignLaunches(launches, banks) {
  const out = Array.from({ length: Math.max(1, banks) }, () => []);
  (launches || []).forEach((l, i) => out[i % out.length].push(l.t));
  return out;
}

/** Ship-local -> mount-local (mount rotated about Z by rotation.z). */
function toMount(inv, mount, world, out) {
  out.copy(world).applyMatrix4(inv).sub(mount.position).applyAxisAngle(_Z, -mount.rotation.z);
  return out;
}

function enemyOk(scene, ud, tid) {
  if (!tid) return null;
  const o = scene.ships.get(tid);
  if (!o || !o.visible || o.userData.faction === ud.faction) return null;
  if (scene.destroyedShips && scene.destroyedShips.has(tid)) return null;
  return o;
}

function gunTarget(ship, scene, idx, t) {
  const ud = ship.userData;
  const id = ud.shipId;
  const shots = idx?.shots.get(id);
  const i = lastIndexBefore(shots, t);
  let o = i >= 0 && t - shots[i].t < 20 ? enemyOk(scene, ud, shots[i].target) : null;
  if (!o && scene.loader?.getShipTargetAt) {
    o = enemyOk(scene, ud, scene.loader.getShipTargetAt(id, t)?.target_id);
  }
  if (!o) {
    let bd = Infinity;
    for (const s of scene.ships.values()) {
      if (s === ship || !s.visible || s.userData.faction === ud.faction) continue;
      const d = s.position.distanceToSquared(ship.position);
      if (d < bd) { bd = d; o = s; }
    }
  }
  return o;
}

function activePD(scene, t) {
  if (scene._hullPdT !== t) {
    scene._hullPdT = t;
    scene._hullPd = scene.loader?.getActivePDBeams ? scene.loader.getActivePDBeams(t) : [];
  }
  return scene._hullPd;
}

/**
 * Hook called at the end of SceneManager.updateShip for every live ship.
 */
export function updateHullDetails(ship, state, scene) {
  const ud = ship.userData;
  const t = scene.currentTime || 0;
  const dt = Math.min(0.1, scene.delta || 0.016);
  const hull = Math.max(0, Math.min(100, state.hull ?? 100)) / 100;
  const dying = !!state.dying;

  if (ud.hullLights) {
    ud._isDead ||= () => !!scene.destroyedShips?.has(ud.shipId);
    updateHullLights(ud.hullLights, t, { dying, hull }, ud._isDead);
  }
  if (dying) return;   // a dying hull's mounts freeze where they were

  const idx = indexFor(scene.loader);
  _m.compose(ship.position, ship.quaternion, ship.scale).invert();

  // --- coilgun turrets ------------------------------------------------------
  const guns = ud.gunTurrets;
  if (guns && guns.length) {
    const target = gunTarget(ship, scene, idx, t);
    const slew = 1 - Math.exp(-dt * 2.0);
    const shots = idx?.shots.get(ud.shipId);
    const last = lastIndexBefore(shots, t);
    for (const rig of guns) {
      let ty = 0, tp = 0.12;
      if (target) {
        const v = toMount(_m, rig.mount, target.position, _v);
        ty = Math.atan2(v.x, v.z);
        tp = THREE.MathUtils.clamp(Math.atan2(v.y, Math.hypot(v.x, v.z)), -0.04, 1.45);
      }
      let dy = ty - rig.yawAngle;
      dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      rig.yawAngle += dy * slew;
      rig.pitchAngle += (tp - rig.pitchAngle) * slew;
      rig.yaw.rotation.y = rig.yawAngle;
      rig.pitch.rotation.x = -rig.pitchAngle;
      // Recoil from this mount's own weapon slot
      let k = 0;
      for (let i = last; i >= 0 && t - shots[i].t <= 1.4; i--) {
        if (shots[i].slot === rig.slot) { k = recoil(t - shots[i].t); break; }
      }
      rig.pitch.position.z = rig.pitchZ - k * rig.s * 0.3;
    }
  }

  // --- PD ball turrets -------------------------------------------------------
  const pds = ud.pdTurrets;
  if (pds && pds.length) {
    const segs = activePD(scene, t);
    const fast = 1 - Math.exp(-dt * 7.0);
    const lazy = 1 - Math.exp(-dt * 1.5);
    for (const rig of pds) {
      let seg = null;
      for (const s of segs) {
        if (s.shipId === ud.shipId && s.turret === rig.name && t >= s.start && t <= s.end + 0.6) { seg = s; break; }
      }
      const pos = seg ? scene.getEntityPosition(seg.targetType, seg.targetId) : null;
      let ty, tp, k;
      if (pos) {
        const v = toMount(_m, rig.mount, pos, _v);
        ty = Math.atan2(v.x, v.z);
        tp = Math.atan2(v.y, Math.hypot(v.x, v.z));
        k = fast;
      } else {
        // Search sweep, on battle time
        ty = rig.phase * 2 + 1.1 * Math.sin(t * 0.31 + rig.phase * 5);
        tp = 0.45 + 0.3 * Math.sin(t * 0.23 + rig.phase * 3);
        k = lazy;
      }
      tp = THREE.MathUtils.clamp(tp, -0.12, 1.5);
      let dy = ty - rig.yawAngle;
      dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      rig.yawAngle += dy * k;
      rig.pitchAngle += (tp - rig.pitchAngle) * k;
      rig.ball.rotation.set(-rig.pitchAngle, rig.yawAngle, 0, 'YXZ');
    }
  }

  // --- torpedo launch doors ---------------------------------------------------
  const doors = ud.launchDoors;
  if (doors && doors.length) {
    if (!ud._doorTimes || ud._doorIdx !== idx) {
      const banks = doors.reduce((m, d) => Math.max(m, d.bank + 1), 1);
      ud._doorTimes = assignLaunches(idx?.launches.get(ud.shipId), banks);
      ud._doorIdx = idx;
    }
    for (const d of doors) {
      const o = doorOpen(ud._doorTimes[d.bank] || [], t);
      d.mesh.position.copy(d.closed).addScaledVector(d.slide, o);
      d.mesh.visible = true;
    }
  }
}
