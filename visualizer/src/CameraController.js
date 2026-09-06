import * as THREE from 'three';

/**
 * CameraController - Manages camera modes (free, follow, orbit)
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
    this.orbitControls.addEventListener?.('start', () => { this.fleetFramed = false; });
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
  update(shipStates) {
    if (this.mode === 'free') return;
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

  /** Frame the active fleets with room for the side panels. */
  focusFleet(shipStates) {
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
