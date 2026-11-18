import * as THREE from 'three';
import { updateHullLights } from './HullLights.js';
import { floodLevel, FLOOD_OPACITY, CODE_GLOW, LIVERY_FLOOD, GLASS_GLOW } from './HullMarkings.js';
import { assignBarrels } from './WeaponTimeline.js';
import { arcsFor, clampToArc, slewDir, slewProgress, mountAngles, mountDir, turAim, pdPlan, shownPdMount } from './TurretArcs.js';
import { GUN_DEPRESSION_DEG, PD_DEPRESSION_DEG } from './MountLimits.js';
import HULL_MOUNTS from './hullMountsData.js';

/**
 * Per-frame hull animation driven by the recording, all on BATTLE time so
 * paused frames and timeline seeks are deterministic:
 *
 *   - recordings from the weapon-arcs engine carry every mount's aim per
 *     trace frame (`tur`: [az, el, state] in the body frame): turrets and
 *     PD balls are posed from it, slerped between frames, so they slew at
 *     the engine's rates and stop at the edge of their arc when masked
 *   - without `tur`: coilgun turrets train on the ship's actual target
 *     (last shot fired, else the captain's ordered target, else the nearest
 *     enemy), slew onto the next shot's target at their class traverse rate
 *     so they arrive as it fires, and never aim into a direction their arc
 *     mask (data/hull_mounts.json) blocks: a masked mount rests at the edge
 *   - PD ball turrets slew onto the torpedo / slug their dwell segment is
 *     burning (TurretArcs.pdPlan: the recorded turret, or on old recordings
 *     a mount with a clear line), arriving as the dwell starts, else sweep
 *     a search pattern
 *   - turrets recoil when their weapon slot fires (twin mounts: the barrel
 *     that fired, from the shot's `barrel` or alternating shot by shot)
 *
 * Poses are pure functions of battle time (no per-frame slew state), so a
 * paused frame, a replayed frame and a direct seek give the same pose; a
 * dying ship's mounts fall into a fixed, seeded slump.
 *   - torpedo launch doors open around each torpedo_launched event
 *   - torpedo magazines empty round by round: a rack round leaves its rail
 *     at its launch, a VLS hatch swings up in the 2.5 s before and stays
 *     open over the empty tube, its status lamp going from the faction
 *     colour to dark
 *   - navigation lamps blink (HullLights); floodlights, the self-lit hull
 *     codes, livery and sensor glass brown out with the hull
 *     (HullMarkings.floodLevel)
 *
 * The scene may carry `scene.loader` (BattleLoader); without it (showroom,
 * tests) everything falls back to nearest-enemy aim and idle motion.
 */

const _m = new THREE.Matrix4();
const _indexCache = new WeakMap();

/** Per-ship shot and torpedo-launch timelines from the event log. */
export function shipEventIndex(events) {
  const shots = new Map();
  const launches = new Map();
  const losses = new Map();
  const push = (m, k, v) => { if (!m.has(k)) m.set(k, []); m.get(k).push(v); };
  const barrels = assignBarrels(events);
  for (const e of events || []) {
    if (e.event_type === 'shot_fired') {
      push(shots, e.ship_id, { t: e.timestamp, target: e.data?.target_id, slot: e.data?.weapon_slot, barrel: barrels.get(e) ?? 0 });
    } else if (e.event_type === 'torpedo_launched') {
      push(launches, e.ship_id, { t: e.timestamp, target: e.data?.target_id,
        mount: e.data?.mount ?? null, round: Number.isInteger(e.data?.round) ? e.data.round : null });
    } else if (e.event_type === 'mount_damaged' && e.data?.kind === 'launcher' && e.data.destroyed && e.data.torpedoes_lost > 0) {
      // a wrecked tube loses the rounds still on it (the HUD count drops with it)
      push(losses, e.ship_id, { t: e.timestamp, mount: e.data.slot ?? null, count: e.data.torpedoes_lost, all: true });
    } else if (e.event_type === 'magazine_cookoff' && e.data?.torpedoes_lost > 0) {
      push(losses, e.ship_id, { t: e.timestamp, mount: null, count: e.data.torpedoes_lost, all: false });
    }
  }
  for (const m of [shots, launches, losses]) for (const a of m.values()) a.sort((x, y) => x.t - y.t);
  return { shots, launches, losses, count: (events || []).length };
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

/**
 * Magazine bank of a launcher slot: its index among the hull class's
 * launcher mounts in data/hull_mounts.json order (the engine's `mount`;
 * Hulls.js builds one bank per launcher in the same order), or -1.
 */
export function launcherBank(shipType, slot) {
  if (!slot) return -1;
  const mounts = HULL_MOUNTS.hulls?.[shipType]?.mounts || [];
  let b = 0;
  for (const m of mounts) {
    if (m.kind !== 'launcher') continue;
    if (m.slot === slot) return b;
    b++;
  }
  return -1;
}

/**
 * Launch time of every magazine round: out[bank][round] = battle time it
 * left (undefined: still racked), and out.lost[bank][round] = the time a
 * round still racked was lost without being fired. A launch that names
 * its `mount` + `round` (pass-5 launch sequence) empties exactly that
 * round; one that does not (older recordings) is dealt as before - banks
 * round robin by launch count, the lowest round of the bank not yet gone.
 * `losses` (shipEventIndex): a wrecked tube (mount_damaged, launcher
 * destroyed) loses every round left on its bank; a magazine cook-off loses
 * `count` rounds, the highest-numbered on the fullest banks first - so the
 * rack always shows the rounds the HUD counts. `bankOf(mount)` maps a
 * launcher slot to its bank (-1: unknown, dealt like an old launch).
 */
export function assignRounds(launches, banks, bankOf = () => -1, losses = null) {
  const n = Math.max(1, banks);
  const out = Array.from({ length: n }, () => []);
  out.lost = Array.from({ length: n }, () => []);
  const used = (b, r) => out[b][r] !== undefined || out.lost[b][r] !== undefined;
  const items = [...(launches || []).map(l => ({ l })), ...(losses || []).map(x => ({ x }))]
    .map((it, i) => ({ ...it, t: (it.l || it.x).t, i, loss: !!it.x }))
    // (same-tick: launches before the losses they leave behind, then input order)
    .sort((a, b) => a.t - b.t || a.loss - b.loss || a.i - b.i);
  // rounds per bank seen so far (a bank's size is unknown here: the highest index any launch named, else 12)
  let legacy = 0;
  const size = (b) => Math.max(12, out[b].length, out.lost[b].length);
  for (const it of items) {
    if (it.l) {
      const l = it.l;
      const b = l.mount != null && Number.isInteger(l.round) && l.round >= 0 ? bankOf(l.mount) : -1;
      if (b >= 0 && b < n) {
        if (!(out[b][l.round] <= l.t)) out[b][l.round] = l.t;
        continue;
      }
      // round robin by count; a bank with nothing left passes to the next
      let bank = legacy++ % n;
      for (let k = 0; k < n; k++) {
        const b = (bank + k) % n;
        let free = false;
        for (let r = 0; r < size(b) && !free; r++) free = !used(b, r);
        if (free) { bank = b; break; }
      }
      let r = 0;
      while (used(bank, r)) r++;
      out[bank][r] = l.t;
      continue;
    }
    const x = it.x;
    if (x.all) {
      const b = bankOf(x.mount);
      if (b < 0 || b >= n) continue;
      for (let r = 0; r < size(b); r++) if (!used(b, r)) out.lost[b][r] = x.t;
      continue;
    }
    for (let k = 0; k < x.count; k++) {
      let best = -1, bestLeft = 0;
      for (let b = n - 1; b >= 0; b--) {
        let left = 0;
        for (let r = 0; r < size(b); r++) if (!used(b, r)) left++;
        if (left > bestLeft) { best = b; bestLeft = left; }
      }
      if (best < 0) break;
      let r = size(best) - 1;
      while (r >= 0 && used(best, r)) r--;
      if (r >= 0) out.lost[best][r] = x.t;
    }
  }
  return out;
}

/** Per-bank sorted launch times (launch doors), by the same dealing as assignRounds. */
export function bankTimes(rounds) {
  return rounds.map(b => b.filter(t => t !== undefined).sort((x, y) => x - y));
}

/**
 * One magazine round at battle time t, given the launch time `tl` dealt to
 * it (undefined: never fired). open: hatch slide 0..1 (opens over the
 * second from 2.5 s before launch, stays open); lamp: 0 loaded, 1 firing,
 * 2 expended; gone: the round has left (rack rounds vanish).
 */
export function magazineRoundState(tl, t, out = { open: 0, lamp: 0, gone: false }) {
  if (tl === undefined || !(t >= tl - 2.5)) {
    out.open = 0; out.lamp = 0; out.gone = false;
    return out;
  }
  out.open = Math.min(1, (t - (tl - 2.5)) / 1.0);
  out.lamp = t < tl + 0.6 ? 1 : 2;
  out.gone = t >= tl;
  return out;
}

/** Open VLS hatch: swung up past upright (rad), leaning outboard. */
export const HATCH_OPEN = 1.65;

const _rs = { open: 0, lamp: 0, gone: false };
const _mm = new THREE.Matrix4();
const _hm = new THREE.Matrix4();
const _hp = new THREE.Vector3();
const _zero = new THREE.Matrix4().makeScale(0, 0, 0);
const _firing = new THREE.Color(2.4, 2.2, 1.9);
const _spent = new THREE.Color(0.07, 0.015, 0.01);

/**
 * Hatch-local swing of a VLS hatch (cell size s, 0.9 s square, 0.1 s
 * thick) opened `open` 0..1: a turn about its outboard bottom edge (local
 * +X, along Z) by up to HATCH_OPEN, so it stands up over the coaming
 * instead of sliding off the bank.
 */
export function hatchHinge(s, open, out = new THREE.Matrix4()) {
  _hp.set(s * 0.45, -s * 0.05, 0);
  out.makeRotationZ(-HATCH_OPEN * open);
  // T(edge) R T(-edge)
  const e = out.elements;
  const px = _hp.x, py = _hp.y;
  e[12] = px - (e[0] * px + e[4] * py);
  e[13] = py - (e[1] * px + e[5] * py);
  return out;
}

/**
 * Pose a hull's magazine (Hulls.js _buildMagazine) at battle time t.
 * times[bank][round] = the battle time that round left (assignRounds).
 * Only rounds whose state changed are rewritten.
 */
export function updateMagazine(mag, times, t) {
  let dirty = false;
  for (let i = 0; i < mag.entries.length; i++) {
    const e = mag.entries[i];
    const st = magazineRoundState(times[e.bank]?.[mag.order[i]], t, _rs);
    // lost without firing (wrecked tube, cook-off): the round is gone, a cell stays shut and dark
    const lost = times.lost?.[e.bank]?.[mag.order[i]];
    if (lost !== undefined && t >= lost && !st.gone) { st.open = 0; st.lamp = 2; st.gone = true; }
    const key = Math.round(st.open * 1000) + st.lamp * 10000 + (st.gone ? 100000 : 0);
    if (mag.state[i] === key) continue;
    mag.state[i] = key;
    dirty = true;
    if (mag.kind === 'rack') {
      mag.body.setMatrixAt(i, st.gone ? _zero : mag.base[i]);
    } else {
      _mm.copy(mag.base[i]);
      if (e.hinge && st.open > 0) _mm.multiply(hatchHinge(e.s, st.open, _hm));
      mag.body.setMatrixAt(i, _mm);
      mag.lamp?.setMatrixAt(i, _mm);   // the status lamp rides on the hatch
      if (mag.lamp) mag.lamp.setColorAt(i, st.lamp === 0 ? mag.loaded : st.lamp === 1 ? _firing : _spent);
    }
  }
  if (dirty) {
    mag.body.instanceMatrix.needsUpdate = true;
    if (mag.lamp) mag.lamp.instanceMatrix.needsUpdate = true;
    if (mag.lamp?.instanceColor) mag.lamp.instanceColor.needsUpdate = true;
  }
  return dirty;
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

const _aim = new THREE.Vector2();
const _p = new THREE.Vector3();
const _dA = new THREE.Vector3(), _dB = new THREE.Vector3(), _dC = new THREE.Vector3();
const GUN_PITCH_MIN = -GUN_DEPRESSION_DEG * Math.PI / 180;
const PD_PITCH_MIN = -PD_DEPRESSION_DEG * Math.PI / 180;
/** Gun rest pose (no target): trained along the hull, barrels slightly up. */
const GUN_REST_PITCH = 0.12;
/** A PD ball stays on its round while the beam is lit: 1 s past the last recorded tick (beamPulse). */
const PD_HOLD_S = 1.0;

/** Hull-local unit direction from a mount to a world point (`inv` = inverse ship matrix). */
function localDirTo(inv, mount, world, out) {
  out.copy(world).applyMatrix4(inv).sub(mount.position);
  return out.lengthSq() > 0 ? out.normalize() : out.set(0, 0, 1);
}

/**
 * Last entry of a start-sorted segment list with start <= t, or -1.
 */
function lastStarted(list, t) {
  if (!list || !list.length || list[0].start > t) return -1;
  let lo = 0, hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (list[mid].start <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Stable 0..1 hash of (ship id, mount index) for the dead-mount slump. */
function hashRig(id, i) {
  let h = 2166136261 ^ i;
  for (let k = 0; k < id.length; k++) h = Math.imul(h ^ id.charCodeAt(k), 16777619);
  return (h >>> 0) / 4294967296;
}

/** Hull-local aim that lays a gun rig on `target` (rest pose if none). */
function gunDir(rig, target, out) {
  if (!target) return mountDir(rig.mount.rotation.z, 0, GUN_REST_PITCH, out);
  return localDirTo(_m, rig.mount, target.position, out);
}

/**
 * Where the dwell's round is at battle time t, straight from the recording
 * (never from what was last drawn); null when it is not in flight then.
 */
function pdTargetPosition(scene, seg, t) {
  const at = scene.weapons?.positionAt?.(seg.targetType, seg.targetId, t, _p);
  if (at !== undefined) return at;
  return scene.getEntityPosition ? scene.getEntityPosition(seg.targetType, seg.targetId) : null;
}

/**
 * Gun turret aim without the engine's `tur`: on the ship's target, slewing
 * at the mount's class rate onto the next shot's target so it arrives as
 * that shot fires, clamped to the mount's arc. Pure in (t, recording).
 * Writes yaw/pitch to `out`; returns true when the mount is masked.
 */
export function fallbackGunAim(rig, arc, target, next, tNext, t, out) {
  gunDir(rig, target, _dA);
  if (next) {
    gunDir(rig, next, _dB);
    const w = slewProgress(_dA.angleTo(_dB), arc?.rate || 0.5, tNext, t);
    slewDir(_dA, _dB, w, _dA);
  }
  const masked = arc ? clampToArc(arc, _dA, _dA) : false;
  mountAngles(rig.mount.rotation.z, _dA, out);
  out.y = Math.max(GUN_PITCH_MIN, out.y);
  return masked;
}

/**
 * PD ball aim without the engine's `tur`: a search sweep, slewing at the
 * PD traverse rate onto the round of this turret's next dwell so it is on
 * it when the beam lights, tracking the round through the dwell, then
 * slewing back; clamped to the mount's arc. `segs` are this turret's
 * dwells sorted by start (TurretArcs.pdPlan). Pure in (t, recording).
 * Writes yaw/pitch to `out`; returns true when the mount is masked.
 */
export function fallbackPdAim(scene, rig, arc, segs, t, out, standIn = null) {
  const rz = rig.mount.rotation.z;
  const rate = arc?.rate || 2;
  if (standIn) {
    // this ball shows another turret's dwell right now (its round went
    // behind the hull for that turret): stay on the round
    const pos = pdTargetPosition(scene, standIn, t);
    if (pos) {
      localDirTo(_m, rig.mount, pos, _dA);
      const m = arc ? clampToArc(arc, _dA, _dA) : false;
      mountAngles(rz, _dA, out);
      return m;
    }
  }
  // search sweep, on battle time
  const D = mountDir(rz, rig.phase * 2 + 1.1 * Math.sin(t * 0.31 + rig.phase * 5),
    0.45 + 0.3 * Math.sin(t * 0.23 + rig.phase * 3), _dA);
  const i = lastStarted(segs, t);
  const cur = i >= 0 ? segs[i] : null;
  if (cur) {
    // the round at t while it burns, else where it was when the dwell ended
    const hold = cur.end + PD_HOLD_S;
    const pos = pdTargetPosition(scene, cur, Math.min(t, hold)) || pdTargetPosition(scene, cur, cur.end)
      || pdTargetPosition(scene, cur, cur.start);
    if (pos) {
      localDirTo(_m, rig.mount, pos, _dB);
      if (t <= hold) D.copy(_dB);
      else slewDir(_dB, D, slewProgress(_dB.angleTo(D), rate, hold + _dB.angleTo(D) / rate, t), D);
    }
  }
  const nxt = segs && i + 1 < segs.length ? segs[i + 1] : null;
  if (nxt) {
    const pos = pdTargetPosition(scene, nxt, t) || pdTargetPosition(scene, nxt, nxt.start);
    if (pos) {
      // leave in time to be on the next round as its beam lights, but not
      // before this one's last recorded tick
      localDirTo(_m, rig.mount, pos, _dC);
      const T = D.angleTo(_dC) / rate;
      const depart = Math.max(nxt.start - T, cur ? cur.end : -Infinity);
      slewDir(D, _dC, slewProgress(T * rate, rate, depart + T, t), D);
    }
  }
  const masked = arc ? clampToArc(arc, D, D) : false;
  mountAngles(rz, D, out);
  return masked;
}

/**
 * Everything self-lit on the plating follows the hull's power `lv`
 * (floodLevel): floodlight pools, the hull codes, the livery paint and
 * the sensor glass (per-ship copies, Hulls.buildShipMaterials).
 */
export function brownOut(ud, lv) {
  if (ud.floodlights) ud.floodlights.material.opacity = FLOOD_OPACITY * lv;
  if (ud.hullMarkings?.material) ud.hullMarkings.material.emissiveIntensity = CODE_GLOW * lv;
  const mats = ud.materials;
  if (mats?.paint) mats.paint.emissiveIntensity = LIVERY_FLOOD * lv;
  if (mats?.glass) mats.glass.emissiveIntensity = GLASS_GLOW * lv;
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
  brownOut(ud, floodLevel(t, hull, dying, (ud.navPhase || 0) % 1));
  const idx = indexFor(scene.loader);
  _m.compose(ship.position, ship.quaternion, ship.scale).invert();

  const arcs = arcsFor(ud.shipType);
  const tur = state.tur || null;   // {a, b, f}: the engine's recorded mount aims

  // --- coilgun turrets ------------------------------------------------------
  const guns = ud.gunTurrets;
  if (guns && guns.length) {
    const shots = idx?.shots.get(ud.shipId);
    const last = lastIndexBefore(shots, t);
    const target = dying || tur ? null : gunTarget(ship, scene, idx, t);
    // The next shot's target: the mount slews onto it in time to fire
    const nx = shots?.[last + 1];
    let next = !dying && !tur && nx ? enemyOk(scene, ud, nx.target) : null;
    if (next === target) next = null;
    for (let r = 0; r < guns.length; r++) {
      const rig = guns[r];
      const arc = arcs?.bySlot.get(rig.slot) || null;
      let st = -1;
      if (dying) {
        // fixed, seeded slump: power gone, barrels sag to the stops
        rig.yawAngle = (hashRig(ud.shipId, r) - 0.5) * 1.6;
        rig.pitchAngle = -0.04;
      } else {
        if (tur && arc) st = turAim(tur, arc.index, _dA);
        if (st >= 0) {
          // the engine's aim, as recorded (body frame -> this mount's yaw/pitch)
          mountAngles(rig.mount.rotation.z, _dA, _aim);
        } else {
          st = fallbackGunAim(rig, arc, target, next, nx ? nx.t : 0, t, _aim) ? 3 : -1;
        }
        rig.yawAngle = _aim.x;
        rig.pitchAngle = _aim.y;
      }
      rig.arcState = st;
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
    const plan = dying || tur ? null : pdPlan(scene)?.get(ud.shipId);
    // Old recordings: a lit dwell whose round is masked for its planned
    // turret right now is shown from a mount with a clear line
    // (Projectiles.beam uses the same rule): that ball stands in for it
    let standIns = null;
    if (plan && arcs?.pds.length > 1) {
      for (const [name, list] of plan) {
        const from = pds.find(p => p.name === name);
        if (!from) continue;
        for (let i = lastStarted(list, t); i >= 0 && list[i].start > t - 30; i--) {
          const seg = list[i];
          if (t > seg.end + PD_HOLD_S) continue;
          const pos = pdTargetPosition(scene, seg, t);
          if (!pos) continue;
          const shown = shownPdMount(name, arcs.pds, localDirTo(_m, from.mount, pos, _dA));
          if (shown && shown !== name) (standIns ||= new Map()).set(shown, seg);
        }
      }
    }
    for (let r = 0; r < pds.length; r++) {
      const rig = pds[r];
      const arc = arcs?.bySlot.get(rig.name) || null;
      const rz = rig.mount.rotation.z;
      let st = -1;
      if (dying) {
        _aim.set(rig.phase * 2, -0.12);
      } else if (tur && arc && (st = turAim(tur, arc.index, _dA)) >= 0) {
        mountAngles(rz, _dA, _aim);
      } else {
        st = fallbackPdAim(scene, rig, arc, plan?.get(rig.name), t, _aim, standIns?.get(rig.name)) ? 3 : -1;
      }
      rig.arcState = st;
      rig.yawAngle = _aim.x;
      rig.pitchAngle = THREE.MathUtils.clamp(_aim.y, dying ? -0.12 : PD_PITCH_MIN, Math.PI / 2);
      rig.ball.rotation.set(-rig.pitchAngle, rig.yawAngle, 0, 'YXZ');
    }
  }
  // --- torpedo launch doors ---------------------------------------------------
  const doors = ud.launchDoors;
  if (doors && doors.length) {
    if (!ud._doorTimes || ud._doorIdx !== idx) {
      const banks = doors.reduce((m, d) => Math.max(m, d.bank + 1), 1);
      ud._doorTimes = bankTimes(assignRounds(idx?.launches.get(ud.shipId), banks, s => launcherBank(ud.shipType, s), idx?.losses.get(ud.shipId)));
      ud._doorIdx = idx;
    }
    for (const d of doors) {
      const o = doorOpen(ud._doorTimes[d.bank] || [], t);
      d.mesh.position.copy(d.closed).addScaledVector(d.slide, o);
      d.mesh.visible = true;
    }
  }
  // --- torpedo magazine -------------------------------------------------------
  const mag = ud.magazine;
  if (mag) {
    if (!mag.times || mag.idx !== idx) {
      // the exact round the engine launched (mount + round), else by count
      mag.times = assignRounds(idx?.launches.get(ud.shipId), mag.banks, s => launcherBank(ud.shipType, s), idx?.losses.get(ud.shipId));
      mag.idx = idx;
    }
    updateMagazine(mag, mag.times, t);
  }
}
