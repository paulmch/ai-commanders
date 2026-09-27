import * as THREE from 'three';
import { updateHullLights } from './HullLights.js';
import { assignBarrels } from './WeaponTimeline.js';

/**
 * Per-frame hull animation driven by the recording, all on BATTLE time so
 * paused frames and timeline seeks are deterministic:
 *
 *   - coilgun turrets train on the ship's actual target (last shot fired,
 *     else the captain's ordered target, else the nearest enemy), re-lay
 *     onto the next shot's target in the second before it fires, and
 *     recoil when their weapon slot fires (twin mounts: the barrel that
 *     fired, from the shot's `barrel` or alternating shot by shot)
 *   - PD ball turrets swing onto the torpedo / slug their dwell segment is
 *     burning (pd_fired turret_name -> pd rig), else sweep a search pattern
 *
 * Poses are pure functions of battle time (no per-frame slew state), so a
 * paused frame, a replayed frame and a direct seek give the same pose; a
 * dying ship's mounts fall into a fixed, seeded slump.
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
  const barrels = assignBarrels(events);
  for (const e of events || []) {
    if (e.event_type === 'shot_fired') {
      push(shots, e.ship_id, { t: e.timestamp, target: e.data?.target_id, slot: e.data?.weapon_slot, barrel: barrels.get(e) ?? 0 });
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

const RELAY_S = 1.2;
const _aim = new THREE.Vector2();
const _p = new THREE.Vector3();

/** Stable 0..1 hash of (ship id, mount index) for the dead-mount slump. */
function hashRig(id, i) {
  let h = 2166136261 ^ i;
  for (let k = 0; k < id.length; k++) h = Math.imul(h ^ id.charCodeAt(k), 16777619);
  return (h >>> 0) / 4294967296;
}

/** Yaw/pitch (x/y of out) that lay a gun rig on `target` (rest pose if none). */
function aimGun(rig, target, out) {
  if (!target) return out.set(0, 0.12);
  const v = toMount(_m, rig.mount, target.position, _v);
  return out.set(Math.atan2(v.x, v.z),
    THREE.MathUtils.clamp(Math.atan2(v.y, Math.hypot(v.x, v.z)), -0.04, 1.45));
}

/** Where the dwell's round is at battle time t (recording-derived when possible). */
function pdTargetPosition(scene, seg, t) {
  const at = scene.weapons?.positionAt?.(seg.targetType, seg.targetId, t, _p);
  if (at !== undefined) return at;
  return scene.getEntityPosition ? scene.getEntityPosition(seg.targetType, seg.targetId) : null;
}

/**
 * Hook called at the end of SceneManager.updateShip for every live ship.
 */
export function updateHullDetails(ship, state, scene) {
  const ud = ship.userData;
  const t = scene.currentTime || 0;
  const hull = Math.max(0, Math.min(100, state.hull ?? 100)) / 100;
  const dying = !!state.dying;

  if (ud.hullLights) {
    ud._isDead ||= () => !!scene.destroyedShips?.has(ud.shipId);
    updateHullLights(ud.hullLights, t, { dying, hull }, ud._isDead);
  }
  const idx = indexFor(scene.loader);
  _m.compose(ship.position, ship.quaternion, ship.scale).invert();

  // --- coilgun turrets ------------------------------------------------------
  const guns = ud.gunTurrets;
  if (guns && guns.length) {
    const shots = idx?.shots.get(ud.shipId);
    const last = lastIndexBefore(shots, t);
    const target = dying ? null : gunTarget(ship, scene, idx, t);
    // Re-lay onto the next shot's target during the second before it fires
    let next = null, w = 0;
    const nx = shots?.[last + 1];
    if (!dying && nx && nx.t - t < RELAY_S) {
      next = enemyOk(scene, ud, nx.target);
      if (next === target) next = null;
      else w = THREE.MathUtils.smoothstep(1 - (nx.t - t) / RELAY_S, 0, 1);
    }
    for (let r = 0; r < guns.length; r++) {
      const rig = guns[r];
      if (dying) {
        // fixed, seeded slump: power gone, barrels sag to the stops
        rig.yawAngle = (hashRig(ud.shipId, r) - 0.5) * 1.6;
        rig.pitchAngle = -0.04;
      } else {
        aimGun(rig, target, _aim);
        if (next) {
          const y0 = _aim.x, p0 = _aim.y;
          aimGun(rig, next, _aim);
          _aim.x = y0 + Math.atan2(Math.sin(_aim.x - y0), Math.cos(_aim.x - y0)) * w;
          _aim.y = p0 + (_aim.y - p0) * w;
        }
        rig.yawAngle = _aim.x;
        rig.pitchAngle = _aim.y;
      }
      rig.yaw.rotation.y = rig.yawAngle;
      rig.pitch.rotation.x = -rig.pitchAngle;
      // Recoil from this mount's own weapon slot; on a twin mount the
      // barrel that fired runs back and the cradle only kicks a little
      let k = -1, k0 = -1, k1 = -1;
      for (let i = last; !dying && i >= 0 && t - shots[i].t <= 1.4; i--) {
        if (shots[i].slot !== rig.slot) continue;
        const r = recoil(t - shots[i].t);
        if (k < 0) k = r;
        if (shots[i].barrel === 1) { if (k1 < 0) k1 = r; } else if (k0 < 0) k0 = r;
        if (!rig.barrels || (k0 >= 0 && k1 >= 0)) break;
      }
      k = Math.max(0, k);
      if (rig.barrels?.length === 2) {
        rig.pitch.position.z = rig.pitchZ - k * rig.s * 0.06;
        rig.barrels[0].position.z = k0 > 0 ? -k0 * rig.s * 0.3 : 0;
        rig.barrels[1].position.z = k1 > 0 ? -k1 * rig.s * 0.3 : 0;
      } else {
        rig.pitch.position.z = rig.pitchZ - k * rig.s * 0.3;
      }
    }
  }

  // --- PD ball turrets -------------------------------------------------------
  const pds = ud.pdTurrets;
  if (pds && pds.length) {
    const segs = dying ? [] : activePD(scene, t);
    for (let r = 0; r < pds.length; r++) {
      const rig = pds[r];
      let ty, tp;
      if (dying) {
        ty = rig.phase * 2;
        tp = -0.12;
      } else {
        // Search sweep, on battle time
        ty = rig.phase * 2 + 1.1 * Math.sin(t * 0.31 + rig.phase * 5);
        tp = 0.45 + 0.3 * Math.sin(t * 0.23 + rig.phase * 3);
        let seg = null;
        for (const s of segs) {
          if (s.shipId === ud.shipId && s.turret === rig.name && t >= s.start && t <= s.end + 0.6) { seg = s; break; }
        }
        const pos = seg ? pdTargetPosition(scene, seg, t) : null;
        if (pos) {
          // swing from the sweep onto the round over the first 0.3 s of
          // dwell, and back to the sweep in the 0.6 s after it ends
          const v = toMount(_m, rig.mount, pos, _v);
          const k = THREE.MathUtils.smoothstep(t - seg.start, 0, 0.3) * (1 - THREE.MathUtils.smoothstep(t - seg.end, 0.1, 0.6));
          const dy = Math.atan2(v.x, v.z) - ty;
          ty += Math.atan2(Math.sin(dy), Math.cos(dy)) * k;
          tp += (Math.atan2(v.y, Math.hypot(v.x, v.z)) - tp) * k;
        }
      }
      rig.yawAngle = ty;
      rig.pitchAngle = THREE.MathUtils.clamp(tp, -0.12, 1.5);
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
