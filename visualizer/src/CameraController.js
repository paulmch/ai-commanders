import * as THREE from 'three';
import { buildShotList, shotAt, buildShakeSources, shakeAt, handheldAt, highlightList } from './Director.js';
import { frameGroups, slideToFreeArea, shotPose } from './DirectorCamera.js';
import { FREE_TRUE_K } from './render/Scale.js';

export { frameGroups, slideToFreeArea };

// ---------------------------------------------------------------- director
// The auto 'director' (LV-09 / FLEET-1): while the camera is free and the
// viewer is not steering, frame both fleets and cut to the moments that
// matter - kills and torpedo salvos. Shots are chosen by BATTLE time
// (scrubbing replays the same cuts); the camera eases in wall time.

/** Wall seconds a manual drag/zoom (or an outside camera move) keeps the director away. */
export const DIRECTOR_USER_HOLD_S = 20;
const KILL_SHOT_S = 8;
const DETONATION_SHOT_S = 6;
const SALVO_SHOT_S = 6;
const SALVO_MIN = 3;
const SALVO_WINDOW_S = 3;
/** Wall seconds the camera takes to ease back from a manual pose to the director's. */
export const DIRECTOR_BLEND_S = 0.9;
const KILL_ORBIT_DEG_PER_S = 5;
/** Smallest fleet framing radius, in kit units (25 km on the cinematic scale). */
const FLEET_MIN_RADIUS = 25;

/**
 * Director cuts from a recording, sorted by start time.
 * Kills come from ship_dying / ship_destroyed events (killer attached), or
 * from frame-derived deaths in older recordings; a reactor that cooks off
 * well after the killing blow gets its own detonation shot. A salvo is
 * SALVO_MIN+ torpedo launches by one faction within SALVO_WINDOW_S.
 * @param {Array} events - loader events (sorted by time)
 * @param {Object} deathInfo - BattleLoader.buildDeathInfo()
 * @returns {Array<{kind, t, until, subject, others}>}
 */
export function buildDirectorCuts(events = [], deathInfo = {}) {
  const cuts = [];
  const dying = new Map();
  for (const e of events) {
    if ((e.event_type === 'ship_dying' || e.event_type === 'ship_destroyed') && e.ship_id && !dying.has(e.ship_id)) {
      dying.set(e.ship_id, e.timestamp);
      cuts.push({ kind: 'kill', t: e.timestamp, until: e.timestamp + KILL_SHOT_S, subject: e.ship_id,
        others: e.data?.killer_id ? [e.data.killer_id] : [] });
    }
  }
  for (const [id, death] of Object.entries(deathInfo || {})) {
    if (!Number.isFinite(death?.time)) continue;
    const start = dying.get(id);
    if (start === undefined) {
      cuts.push({ kind: 'kill', t: death.time, until: death.time + KILL_SHOT_S, subject: id, others: [] });
    } else if (death.time - start > KILL_SHOT_S) {
      cuts.push({ kind: 'detonation', t: death.time, until: death.time + DETONATION_SHOT_S, subject: id, others: [] });
    }
  }
  const byFaction = new Map();
  for (const e of events) {
    if (e.event_type !== 'torpedo_launched' || !e.ship_id) continue;
    const faction = String(e.ship_id).split('_')[0];
    if (!byFaction.has(faction)) byFaction.set(faction, []);
    byFaction.get(faction).push(e);
  }
  for (const launches of byFaction.values()) {
    let i = 0;
    while (i < launches.length) {
      let j = i;
      while (j + 1 < launches.length && launches[j + 1].timestamp - launches[i].timestamp <= SALVO_WINDOW_S) j++;
      if (j - i + 1 >= SALVO_MIN) {
        const group = launches.slice(i, j + 1);
        const targets = new Map();
        for (const e of group) if (e.data?.target_id) targets.set(e.data.target_id, (targets.get(e.data.target_id) || 0) + 1);
        const target = [...targets.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null;
        cuts.push({ kind: 'salvo', t: group[0].timestamp, until: group.at(-1).timestamp + SALVO_SHOT_S,
          subject: target, others: [...new Set(group.map(e => e.ship_id))] });
        i = j + 1;
      } else {
        i++;
      }
    }
  }
  return cuts.sort((a, b) => a.t - b.t);
}

/**
 * The cut on screen at battle time t, or null for the fleet view. A kill
 * outranks a salvo; otherwise the most recent start wins.
 */
export function activeCut(cuts, t) {
  let best = null;
  for (const cut of cuts || []) {
    if (cut.t > t) break;
    if (t >= cut.until) continue;
    const rank = cut.kind === 'salvo' ? 0 : 1;
    const bestRank = best ? (best.kind === 'salvo' ? 0 : 1) : -1;
    if (rank > bestRank || (rank === bestRank && cut.t >= best.t)) best = cut;
  }
  return best;
}

/**
 * CameraController - Manages camera modes (free, follow, orbit) and the
 * auto director that drives the free camera until the viewer takes over.
 */
export class CameraController {
  constructor(camera, orbitControls, sceneManager) {
    this.camera = camera;
    this.orbitControls = orbitControls;
    this.sceneManager = sceneManager;

    this.mode = 'free'; // 'free', 'follow', 'orbit'
    this.targetShipId = null;
    this.lastPosition = null;
    this.lastTime = null;
    this.fleetFramed = true;
    // Director: off until main.js arms it. hold: null | 'user' (drag/zoom)
    // | 'external' (someone else placed the camera - a script; a harness
    // that frames its own shots disarms the director instead), both until
    // holdUntil | 'ship' (a ship is selected; until deselected)
    this.director = { enabled: false, hold: null, holdUntil: 0, cuts: [], shot: null, written: null,
      shots: null, blend: null };
    // Blast shake (all camera modes): sources from the recording, the
    // offset applied for the current frame (removed before the next)
    this.shakeSources = [];
    this.shakeOffset = null;
    this.highlights = [];
    this.onDirectorChange = null;
    this.orbitControls.addEventListener?.('start', () => {
      this.fleetFramed = false;
      this.holdDirector('user');
    });
  }

  // ------------------------------------------------------------ director

  /** Arm or disarm the director (A key / AUTO CAM chip). Arming clears any hold. */
  setDirector(on) {
    const d = this.director;
    d.enabled = !!on;
    d.hold = null;
    d.written = null;
    d.shot = null;
    if (on && this.mode !== 'free') this.setMode('free');
    this.onDirectorChange?.(this.directorState());
  }

  /** Pause the director: 'ship' until released, else for DIRECTOR_USER_HOLD_S. */
  holdDirector(reason) {
    const d = this.director;
    if (!d.enabled) return;
    d.hold = reason;
    d.holdUntil = reason === 'ship' ? Infinity : performance.now() + DIRECTOR_USER_HOLD_S * 1000;
    d.written = null;
    this.onDirectorChange?.(this.directorState());
  }

  /** Lift a 'ship' hold (deselect) - the director picks up again. */
  releaseDirector(reason) {
    const d = this.director;
    if (d.hold && (!reason || d.hold === reason)) {
      d.hold = null;
      d.written = null;
      this.onDirectorChange?.(this.directorState());
    }
  }

  setDirectorCuts(cuts) {
    this.director.cuts = cuts || [];
  }

  /**
   * Give the director a recording: builds the shot list (Director.js), the
   * highlight list and the shake sources. `live` = the battle is still
   * running: no lookahead, shots react to events as they arrive. Without
   * this call the director falls back to the legacy cut list.
   * @param {Object} rec - {events, deathInfo, simTrace, live}
   */
  setDirectorRecording({ events = [], deathInfo = {}, simTrace = null, live = false } = {}) {
    const d = this.director;
    d.shots = buildShotList(events, deathInfo, { lookahead: !live, simTrace });
    this.shakeSources = buildShakeSources(events, deathInfo);
    this.highlights = highlightList(events, deathInfo, { simTrace: live ? null : simTrace });
  }

  /** {enabled, active, hold, secondsLeft, shot} for the HUD chip. */
  directorState() {
    const d = this.director;
    const left = d.hold && Number.isFinite(d.holdUntil)
      ? Math.max(0, (d.holdUntil - performance.now()) / 1000) : null;
    // `shot` keeps the names the HUD chip knows (kill / salvo / fleet);
    // `type` and `kind` carry the director's own vocabulary
    const kind = d.shot?.kind || null;
    const legacy = !d.shot ? null : d.shot.type === undefined ? kind
      : (kind === 'kill' || kind === 'dying') ? 'kill' : (kind === 'salvo' || kind === 'strike') ? 'salvo' : 'fleet';
    return { enabled: d.enabled, active: d.enabled && !d.hold && this.mode === 'free', hold: d.hold,
      secondsLeft: left, shot: legacy || (d.enabled ? 'fleet' : null),
      type: d.shot?.type || null, kind, subject: d.shot?.subject || null };
  }

  /**
   * One director frame: choose the shot for this battle time and ease the
   * free camera toward it.
   * @param {Object} shipStates - interpolated states
   * @param {Object} ctx - {time (battle s), dt (wall s), torpedoes}
   */
  updateDirector(shipStates, { time = 0, dt = 1 / 60, torpedoes = [], shift = 0, widthShare = 0.62 } = {}) {
    const d = this.director;
    if (!d.enabled || this.mode !== 'free') return false;
    if (d.hold && performance.now() >= d.holdUntil) {
      d.hold = null;
      this.onDirectorChange?.(this.directorState());
    }
    if (d.hold) return false;
    const target = this.orbitControls.target;
    // Taking the camera over (armed, re-armed, a hold lifted): first stop
    // whatever a drag left coasting in the controls, and ease in from the
    // pose the viewer left (wall time; the very first frame snaps)
    if (!d.written) {
      this._settleControls();
      d.blend = d.everWritten
        ? { position: this.camera.position.clone(), target: target.clone(), start: performance.now() } : null;
    }
    // Someone else moved the camera since our last write (a script): step
    // aside for a while.
    if (d.written) {
      const dist = Math.max(1, this.camera.position.distanceTo(target));
      if (this.camera.position.distanceTo(d.written.position) > dist * 1e-3 ||
          target.distanceTo(d.written.target) > dist * 1e-3) {
        this.holdDirector('external');
        return false;
      }
    }
    if (d.shots) return this._directShot(shipStates, time, shift, widthShare);
    const cut = activeCut(d.cuts, time);
    this.director.shift = shift;
    this.director.widthShare = widthShare;
    const desired = (cut && this.frameCut(cut, shipStates, time, torpedoes)) || this.frameFleets(shipStates, torpedoes);
    const changed = (d.shot?.kind || null) !== (cut?.kind || null) || d.shot?.subject !== cut?.subject;
    d.shot = cut;
    if (changed) this.onDirectorChange?.(this.directorState());
    if (!desired) return false;
    // Cuts move briskly, the fleet view drifts; only the very first frame
    // snaps (from the scene's default camera) - re-arming eases back
    const rate = cut ? 3.2 : 1.5;
    const blend = d.everWritten ? 1 - Math.exp(-rate * Math.min(0.1, Math.max(0, dt))) : 1;
    d.everWritten = true;
    this.camera.position.lerp(desired.position, blend);
    target.lerp(desired.target, blend);
    this.camera.lookAt(target);
    d.written = { position: this.camera.position.clone(), target: target.clone() };
    this.fleetFramed = !cut;
    return true;
  }

  /**
   * The shot-list director: the pose is a pure function of battle time
   * (DirectorCamera.shotPose) plus a handheld drift on battle time; the
   * only wall-clock term is the ease back from a manual pose.
   */
  _directShot(shipStates, time, shift, widthShare) {
    const d = this.director;
    const target = this.orbitControls.target;
    const world = this._world(shipStates, shift, widthShare);
    let shot = shotAt(d.shots, time);
    let pose = shot ? shotPose(shot, time, world) : null;
    if (!pose) {
      // Subject not on stage (old recording, ship never spawned): wide
      shot = { type: 'wide', kind: 'fleet', t0: shot?.t0 ?? 0, t1: shot?.t1 ?? Infinity, id: shot?.id ?? -1, subject: null };
      pose = shotPose(shot, time, world);
    }
    const changed = d.shot?.id !== shot.id || d.shot?.type !== shot.type;
    d.shot = shot;
    if (changed) this.onDirectorChange?.(this.directorState());
    if (!pose) return false;
    // Handheld drift: a slow wander of the aim, on battle time
    const view = pose.target.clone().sub(pose.position);
    const dist = view.length();
    const right = new THREE.Vector3().crossVectors(view, this.camera.up);
    if (right.lengthSq() > 1e-12) {
      right.normalize();
      const up = new THREE.Vector3().crossVectors(right, view).normalize();
      const hand = handheldAt(time, shot.type === 'wide' ? 0.5 : 1);
      pose.target.addScaledVector(right, hand.x * dist).addScaledVector(up, hand.y * dist);
    }
    if (d.blend) {
      const u = (performance.now() - d.blend.start) / (DIRECTOR_BLEND_S * 1000);
      if (u >= 1) d.blend = null;
      else {
        const k = u <= 0 ? 0 : u * u * (3 - 2 * u);
        pose.position.lerpVectors(d.blend.position, pose.position, k);
        pose.target.lerpVectors(d.blend.target, pose.target, k);
      }
    }
    d.everWritten = true;
    this.camera.position.copy(pose.position);
    target.copy(pose.target);
    this.camera.lookAt(target);
    d.written = { position: this.camera.position.clone(), target: target.clone() };
    this.fleetFramed = shot.type === 'wide' && !shot.cast;
    return true;
  }

  /** Where everything is right now, for DirectorCamera (scene units). */
  _world(shipStates, shift = 0, widthShare = 0.62) {
    const scene = this.sceneManager;
    const scale = scene.SCALE ?? 1 / 1000;
    const deaths = scene.deathInfo || {};
    const vec = (v, k = 1) => (Array.isArray(v) && Number.isFinite(v[0]) ? new THREE.Vector3(v[0] * k, v[1] * k, v[2] * k) : null);
    return {
      ids: Object.keys(shipStates || {}),
      // The recorded position first: a hidden hull's scene position
      // depends on which effects happen to be alive (playback history)
      pos: id => vec(shipStates?.[id]?.position, scale) || scene.getShipPosition(id),
      fwd: id => { const f = vec(shipStates?.[id]?.forward); return f && f.lengthSq() > 1e-9 ? f.normalize() : null; },
      thrust: id => shipStates?.[id]?.thrust || 0,
      length: id => this.getShipLength(id),
      // scene km per kit unit that distances tuned for the kit-size hulls
      // (framing floors) are converted with: 1 in the cinematic scale mode
      unit: this.framingUnit(),
      realScale: this.framingUnit() !== 1,
      deathTime: id => (Number.isFinite(deaths[id]?.time) ? deaths[id].time : Infinity),
      // What the death blast visibly draws between two ages after it goes off
      blastExtent: (id, fromAge, toAge) => scene.deathBlastExtent?.(id, fromAge, toAge) ?? 0,
      faction: id => scene.ships.get(id)?.userData?.faction || String(id).split('_')[0],
      fov: this.camera.fov, aspect: this.camera.aspect, shift, widthShare, scale
    };
  }

  // --------------------------------------------------------------- shake

  /** Take the shake offset back off the camera (right after the draw). */
  removeShake() {
    const off = this.shakeOffset;
    if (!off) return;
    this.camera.position.sub(off.position);
    this.orbitControls.target.sub(off.target);
    this.camera.lookAt(this.orbitControls.target);
    this.shakeOffset = null;
  }

  /**
   * Blast shake for one draw: energy and distance set the amplitude and it
   * decays on battle time (Director.shakeAt). main.js calls this just
   * before scene.render() and removeShake() right after, so the offset
   * never outlives the frame: nothing accumulates, code that places the
   * camera between frames never sees it, and a frame at time t looks the
   * same however the playhead got there.
   */
  applyShake(shipStates, time) {
    this.removeShake();
    if (!this.shakeSources.length || !Number.isFinite(time)) return;
    const scale = this.sceneManager.SCALE ?? 1 / 1000;
    const cam = this.camera.position;
    const sh = shakeAt(this.shakeSources, time, s => {
      const p = s.position || shipStates?.[s.ship]?.position;
      if (!Array.isArray(p)) return null;
      return Math.hypot(p[0] * scale - cam.x, p[1] * scale - cam.y, p[2] * scale - cam.z);
    });
    if (sh.amp < 1e-5) return;
    const target = this.orbitControls.target;
    const view = target.clone().sub(cam);
    const dist = view.length();
    const right = new THREE.Vector3().crossVectors(view, this.camera.up);
    if (right.lengthSq() < 1e-12) return;
    right.normalize();
    const up = new THREE.Vector3().crossVectors(right, view).normalize();
    const offset = right.multiplyScalar(sh.x * dist).addScaledVector(up, sh.y * dist);
    this.shakeOffset = { target: offset, position: offset.clone().multiplyScalar(0.3) };
    cam.add(this.shakeOffset.position);
    target.add(this.shakeOffset.target);
    this.camera.lookAt(target);
  }

  /**
   * Drop OrbitControls' damping inertia without moving the camera. A drag
   * leaves decaying rotate/pan deltas that every controls.update() (one per
   * render) keeps applying; left in place when the director takes over,
   * that drift past its write reads as an outside camera move and parks it
   * on 'external'. three r160 keeps the deltas private: one undamped update
   * applies and zeroes them, then the pose goes back.
   */
  _settleControls() {
    const controls = this.orbitControls;
    if (!controls?.enableDamping || typeof controls.update !== 'function') return;
    const position = this.camera.position.clone();
    const quaternion = this.camera.quaternion.clone();
    const target = controls.target.clone();
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = true;
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    controls.target.copy(target);
  }

  /** Points (scene units) of the live ships, plus in-flight torpedoes. */
  _fleetPoints(shipStates, torpedoes = [], filter = null) {
    const points = [];
    const centroids = { alpha: new THREE.Vector3(), beta: new THREE.Vector3() };
    const counts = { alpha: 0, beta: 0 };
    for (const [id, state] of Object.entries(shipStates || {})) {
      if (state.destroyed || (filter && !filter(id))) continue;
      const p = this.sceneManager.getShipPosition(id);
      if (!p) continue;
      points.push(p);
      const faction = this.sceneManager.ships.get(id)?.userData?.faction || id.split('_')[0];
      if (centroids[faction]) { centroids[faction].add(p); counts[faction]++; }
    }
    for (const torp of torpedoes || []) {
      if (!torp?.position || (filter && !filter(torp.source))) continue;
      const p = new THREE.Vector3().fromArray(torp.position).multiplyScalar(this.sceneManager.SCALE ?? 1 / 1000);
      if (Number.isFinite(p.x)) points.push(p);
    }
    for (const f of ['alpha', 'beta']) if (counts[f]) centroids[f].divideScalar(counts[f]);
    return { points, groupA: counts.alpha ? centroids.alpha : null, groupB: counts.beta ? centroids.beta : null };
  }

  _currentDir() {
    const dir = this.camera.position.clone().sub(this.orbitControls.target);
    return dir.lengthSq() > 1e-9 ? dir.normalize() : new THREE.Vector3(0, 0.6, 1).normalize();
  }

  frameFleets(shipStates, torpedoes) {
    let { points, groupA, groupB } = this._fleetPoints(shipStates, torpedoes);
    if (!points.length) {
      for (const effect of this.sceneManager.destructionEffects || []) points.push(effect.position.clone());
    }
    return frameGroups(points, { fov: this.camera.fov, aspect: this.camera.aspect, groupA, groupB,
      fallbackDir: this._currentDir(), shift: this.director.shift || 0,
      widthShare: this.director.widthShare || 0.62, minRadius: FLEET_MIN_RADIUS * this.framingUnit() });
  }

  frameCut(cut, shipStates, time, torpedoes) {
    if (cut.kind === 'salvo') {
      const ids = new Set([...(cut.others || []), cut.subject].filter(Boolean));
      const { points } = this._fleetPoints(shipStates, torpedoes, id => ids.has(id));
      if (points.length < 2) return null;
      const shooters = this._fleetPoints(shipStates, [], id => (cut.others || []).includes(id));
      const subject = cut.subject ? this.sceneManager.getShipPosition(cut.subject) : null;
      const groupA = shooters.points.length ? shooters.points.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(shooters.points.length) : null;
      return frameGroups(points, { fov: this.camera.fov, aspect: this.camera.aspect, groupA, groupB: subject,
        fallbackDir: this._currentDir(), widthShare: 0.9 * (this.director.widthShare || 0.62),
        shift: this.director.shift || 0, minRadius: FLEET_MIN_RADIUS * this.framingUnit() });
    }
    // Kill / detonation: a slow orbit round the dying hull
    const pos = this.sceneManager.getShipPosition(cut.subject);
    if (!pos) return null;
    const L = this.getShipLength(cut.subject);
    const radius = Math.max(L * 5, 40 * this.framingUnit());
    const base = Math.atan2(-1, 0.6);
    const az = base + THREE.MathUtils.degToRad(KILL_ORBIT_DEG_PER_S) * (time - cut.t);
    const el = THREE.MathUtils.degToRad(18);
    const offset = new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
    // Start the orbit on the side the camera already watches from
    const view = this._currentDir();
    const turn = Math.atan2(view.x, view.z) - Math.atan2(offset.x, offset.z);
    if (cut._turn === undefined) cut._turn = turn;
    offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), cut._turn);
    return slideToFreeArea({ target: pos, position: pos.clone().addScaledVector(offset, radius), radius },
      this.director.shift || 0, this.camera.fov, this.camera.aspect);
  }

  /**
   * Camera distances scale with the ship so a corvette close-up and a
   * dreadnought close-up frame the hull the same way
   */
  getShipLength(shipId) {
    // The framing length is static per scale mode (the real hull length in
    // auto and true, the kit length in cinematic: render/Scale.js), never
    // the size the hull happens to be drawn at from the current camera
    const framed = this.sceneManager.framingLength?.(shipId);
    if (framed > 0) return framed;
    const ship = this.sceneManager.ships.get(shipId);
    return ship?.userData?.size?.length || 8;
  }

  /** Scene km per kit unit for framing floors tuned on the kit-size hulls. */
  framingUnit() {
    return (this.sceneManager.scaleMode || 'cinematic') === 'cinematic' ? 1 : FREE_TRUE_K;
  }

  /**
   * Set camera mode
   * @param {string} mode - 'free', 'follow', or 'orbit'
   * @param {string} shipId - Ship to target (for follow/orbit modes)
   */
  setMode(mode, shipId = null) {
    this.fleetFramed = false;
    // A chase/orbit camera on a ship holds the director; back to free lifts it
    if (mode !== 'free') this.holdDirector('ship');
    else this.releaseDirector('ship');
    this.mode = mode;
    this.targetShipId = shipId;
    this.lastPosition = shipId ? this.sceneManager.getShipPosition(shipId) : null;
    this.lastTime = null;

    if (mode === 'free') {
      this.orbitControls.enabled = true;
    } else if (mode === 'orbit') {
      this.orbitControls.enabled = true;
    } else {
      this.orbitControls.enabled = false;
    }
  }

  /**
   * Update camera based on current mode
   * @param {Object} shipStates - Current ship states from interpolator
   */
  update(shipStates, directorContext = null) {
    this.removeShake();   // (a draw that was never finished)
    if (this.mode === 'free') {
      if (directorContext) this.updateDirector(shipStates, directorContext);
      return;
    }
    if (!this.targetShipId) return;

    const state = shipStates[this.targetShipId];
    if (!state) return;
    if (state.destroyed && !this.sceneManager.destructionEffects.some(e => e.shipId === this.targetShipId)) return;

    const shipPos = this.sceneManager.getShipPosition(this.targetShipId);
    if (!shipPos) return;
    // Translate with the tracked ship before applying any camera easing.
    // This keeps the viewing distance stable through motion and timeline seeks.
    if (this.lastPosition) this.camera.position.add(shipPos.clone().sub(this.lastPosition));
    this.lastPosition = shipPos.clone();
    const time = this.sceneManager.currentTime;
    const snap = this.lastTime != null && (time < this.lastTime || time - this.lastTime > 0.75);
    this.lastTime = time;

    if (this.mode === 'follow') {
      this.updateFollowMode(shipPos, state, snap);
    } else if (this.mode === 'orbit') {
      this.updateOrbitMode(shipPos);
    }
  }

  /**
   * Follow mode - camera behind and above ship
   */
  updateFollowMode(shipPos, state, snap = false) {
    // Calculate offset based on ship's forward direction
    const L = this.getShipLength(this.targetShipId);
    const offset = new THREE.Vector3(L * 0.65, L * 0.8, -L * 2.3);

    if (state.forward) {
      const forward = new THREE.Vector3(state.forward[0], state.forward[1], state.forward[2]).normalize();
      const up = Math.abs(forward.y) > 0.98 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      const right = new THREE.Vector3().crossVectors(up, forward).normalize();
      const realUp = new THREE.Vector3().crossVectors(forward, right).normalize();

      // Transform offset to ship's local space
      const matrix = new THREE.Matrix4().makeBasis(right, realUp, forward);
      offset.applyMatrix4(matrix);
    }

    const targetCamPos = shipPos.clone().add(offset);

    // Smooth camera movement
    const blend = snap ? 1 : 1 - Math.exp(-6 * (this.sceneManager.delta || 1 / 60));
    this.camera.position.lerp(targetCamPos, blend);
    this.orbitControls.target.copy(shipPos);
    this.camera.lookAt(shipPos);
  }

  /**
   * Orbit mode - orbit around ship
   */
  updateOrbitMode(shipPos) {
    this.orbitControls.target.copy(shipPos);
  }

  /**
   * Focus camera on a ship
   */
  focusOnShip(shipId, shipStates) {
    this.fleetFramed = false;
    this.holdDirector('ship');
    const state = shipStates[shipId];
    if (!state) return;

    const shipPos = this.sceneManager.getShipPosition(shipId);
    if (!shipPos) return;

    // Move camera to a close viewing position scaled to the hull
    const L = this.getShipLength(shipId);
    const offset = new THREE.Vector3(L * 1.35, L * 0.7, L * 1.35);
    this.camera.position.copy(shipPos).add(offset);
    this.orbitControls.target.copy(shipPos);
    this.orbitControls.update();
    this.lastPosition = shipPos.clone();
    if (this.mode !== 'free') this.setMode(this.mode, shipId);
  }

  /**
   * Frame the active fleets with room for the side panels (F re-arms a held
   * director). `frame` {shift, widthShare} (ViewInsets.freeViewFrame) centres
   * them in the view the HUD columns leave free; without it the fleets fill
   * 0.65 of the full width.
   */
  focusFleet(shipStates, frame = null) {
    if (this.director.enabled) {
      this.releaseDirector();
      this.director.written = null;
    }
    const bounds = new THREE.Box3();
    for (const [id, state] of Object.entries(shipStates || {})) {
      if (state.destroyed) continue;
      const p = this.sceneManager.getShipPosition(id);
      if (p) bounds.expandByPoint(p);
    }
    if (bounds.isEmpty()) {
      for (const effect of this.sceneManager.destructionEffects) bounds.expandByPoint(effect.position);
    }
    if (bounds.isEmpty()) return;
    this.setMode('free');
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(FLEET_MIN_RADIUS * this.framingUnit(), bounds.getSize(new THREE.Vector3()).length() / 2);
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const share = Number.isFinite(frame?.widthShare) && frame.widthShare > 0 ? frame.widthShare : 0.65;
    const horizontal = Math.atan(Math.tan(halfFov) * this.camera.aspect * share);
    const distance = radius / Math.sin(Math.min(halfFov, horizontal)) * 1.1;
    const view = slideToFreeArea({
      target: center,
      position: center.clone().addScaledVector(new THREE.Vector3(0, 0.6, 1).normalize(), distance),
      radius
    }, Number.isFinite(frame?.shift) ? frame.shift : 0, this.camera.fov, this.camera.aspect);
    this.camera.position.copy(view.position);
    this.orbitControls.target.copy(view.target);
    this.orbitControls.update();
    this.fleetFramed = true;
  }
}
