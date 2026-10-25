import * as THREE from 'three';

// ---------------------------------------------------------------- director
// The auto 'director' (LV-09 / FLEET-1): while the camera is free and the
// viewer is not steering, frame both fleets and cut to the moments that
// matter - kills and torpedo salvos. Shots are chosen by BATTLE time
// (scrubbing replays the same cuts); the camera eases in wall time.

/** Wall seconds a manual drag/zoom keeps the director away. */
export const DIRECTOR_USER_HOLD_S = 20;
const KILL_SHOT_S = 8;
const DETONATION_SHOT_S = 6;
const SALVO_SHOT_S = 6;
const SALVO_MIN = 3;
const SALVO_WINDOW_S = 3;
const FLEET_ELEVATION = THREE.MathUtils.degToRad(25);
const KILL_ORBIT_DEG_PER_S = 5;

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
 * Camera framing for a set of points (scene units): orbit target at their
 * centre, viewed from 25 degrees up and perpendicular to the axis between
 * the two groups (both fleets side by side), far enough that the bounds
 * fill widthShare of the screen width.
 * @param {THREE.Vector3[]} points
 * @param {Object} opts - {fov, aspect, groupA, groupB (centroids or null), fallbackDir, widthShare, minRadius}
 * @returns {{target: THREE.Vector3, position: THREE.Vector3, radius: number}|null}
 */
export function frameGroups(points, { fov = 55, aspect = 16 / 9, groupA = null, groupB = null,
  fallbackDir = null, widthShare = 0.62, minRadius = 25, shift = 0 } = {}) {
  if (!points.length) return null;
  const bounds = new THREE.Box3();
  for (const p of points) bounds.expandByPoint(p);
  const target = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(minRadius, bounds.getSize(new THREE.Vector3()).length() / 2);
  const halfFov = THREE.MathUtils.degToRad(fov) / 2;
  const horizontal = Math.atan(Math.tan(halfFov) * aspect * widthShare);
  const distance = radius / Math.sin(Math.min(halfFov, horizontal)) * 1.05;
  let side = null;
  if (groupA && groupB) {
    const axis = groupB.clone().sub(groupA);
    axis.y = 0;
    if (axis.lengthSq() > 1e-6) side = new THREE.Vector3(-axis.z, 0, axis.x).normalize();
  }
  if (!side) {
    side = (fallbackDir ? fallbackDir.clone() : new THREE.Vector3(0, 0, 1));
    side.y = 0;
    if (side.lengthSq() < 1e-9) side.set(0, 0, 1);
    side.normalize();
  }
  // Keep the side the camera is already on (no flips across the battle)
  if (fallbackDir && side.dot(fallbackDir) < 0) side.negate();
  const dir = side.multiplyScalar(Math.cos(FLEET_ELEVATION)).add(new THREE.Vector3(0, Math.sin(FLEET_ELEVATION), 0));
  const position = target.clone().addScaledVector(dir, distance);
  return slideToFreeArea({ target, position, radius }, shift, fov, aspect);
}

/**
 * Centre a framing in the area the side panels leave free: slide camera
 * and target sideways so the target appears at NDC x = shift.
 */
export function slideToFreeArea(frame, shift, fov = 55, aspect = 16 / 9) {
  if (!shift) return frame;
  const view = frame.target.clone().sub(frame.position);
  const distance = view.length();
  const right = new THREE.Vector3().crossVectors(view, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-12) return frame;
  const halfWidth = distance * Math.tan(THREE.MathUtils.degToRad(fov) / 2) * aspect;
  const slide = right.normalize().multiplyScalar(-shift * halfWidth);
  frame.target.add(slide);
  frame.position.add(slide);
  return frame;
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
    // Director: off until main.js arms it. hold: null | 'user' (drag/zoom,
    // until holdUntil) | 'ship' (a ship is selected) | 'external' (someone
    // else placed the camera - a harness or script; until re-armed)
    this.director = { enabled: false, hold: null, holdUntil: 0, cuts: [], shot: null, written: null };
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

  /** Pause the director: 'user' for DIRECTOR_USER_HOLD_S, else until released. */
  holdDirector(reason) {
    const d = this.director;
    if (!d.enabled) return;
    d.hold = reason;
    d.holdUntil = reason === 'user' ? performance.now() + DIRECTOR_USER_HOLD_S * 1000 : Infinity;
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

  /** {enabled, active, hold, secondsLeft, shot} for the HUD chip. */
  directorState() {
    const d = this.director;
    const left = d.hold === 'user' ? Math.max(0, (d.holdUntil - performance.now()) / 1000) : null;
    return { enabled: d.enabled, active: d.enabled && !d.hold && this.mode === 'free', hold: d.hold,
      secondsLeft: left, shot: d.shot?.kind || (d.enabled ? 'fleet' : null) };
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
    if (d.hold === 'user' && performance.now() >= d.holdUntil) {
      d.hold = null;
      this.onDirectorChange?.(this.directorState());
    }
    if (d.hold) return false;
    const target = this.orbitControls.target;
    // Someone else moved the camera since our last write (a snapshot
    // harness, a script): step aside until re-armed.
    if (d.written) {
      const dist = Math.max(1, this.camera.position.distanceTo(target));
      if (this.camera.position.distanceTo(d.written.position) > dist * 1e-3 ||
          target.distanceTo(d.written.target) > dist * 1e-3) {
        this.holdDirector('external');
        return false;
      }
    }
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
      widthShare: this.director.widthShare || 0.62 });
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
        shift: this.director.shift || 0 });
    }
    // Kill / detonation: a slow orbit round the dying hull
    const pos = this.sceneManager.getShipPosition(cut.subject);
    if (!pos) return null;
    const L = this.getShipLength(cut.subject);
    const radius = Math.max(L * 5, 40);
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
    const ship = this.sceneManager.ships.get(shipId);
    return ship?.userData?.size?.length || 8;
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

  /** Frame the active fleets with room for the side panels (F re-arms a held director). */
  focusFleet(shipStates) {
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
    const radius = Math.max(25, bounds.getSize(new THREE.Vector3()).length() / 2);
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(halfFov) * this.camera.aspect * 0.65);
    const distance = radius / Math.sin(Math.min(halfFov, horizontal)) * 1.1;
    this.camera.position.copy(center).addScaledVector(new THREE.Vector3(0, 0.6, 1).normalize(), distance);
    this.orbitControls.target.copy(center);
    this.orbitControls.update();
    this.fleetFramed = true;
  }
}
