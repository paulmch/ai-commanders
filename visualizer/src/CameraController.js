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
    this.followOffset = new THREE.Vector3(0, 50, -150);
    this.smoothing = 0.1;
  }

  /**
   * Set camera mode
   * @param {string} mode - 'free', 'follow', or 'orbit'
   * @param {string} shipId - Ship to target (for follow/orbit modes)
   */
  setMode(mode, shipId = null) {
    this.mode = mode;
    this.targetShipId = shipId;

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
    if (!state || state.destroyed) return;

    const shipPos = this.sceneManager.getShipPosition(this.targetShipId);
    if (!shipPos) return;

    if (this.mode === 'follow') {
      this.updateFollowMode(shipPos, state);
    } else if (this.mode === 'orbit') {
      this.updateOrbitMode(shipPos);
    }
  }

  /**
   * Follow mode - camera behind and above ship
   */
  updateFollowMode(shipPos, state) {
    // Calculate offset based on ship's forward direction
    let offset = this.followOffset.clone();

    if (state.forward) {
      const forward = new THREE.Vector3(state.forward[0], state.forward[1], state.forward[2]).normalize();
      const up = new THREE.Vector3(0, 1, 0);
      const right = new THREE.Vector3().crossVectors(forward, up).normalize();
      const realUp = new THREE.Vector3().crossVectors(right, forward).normalize();

      // Transform offset to ship's local space
      const matrix = new THREE.Matrix4().makeBasis(right, realUp, forward.negate());
      offset.applyMatrix4(matrix);
    }

    const targetCamPos = shipPos.clone().add(offset);

    // Smooth camera movement
    this.camera.position.lerp(targetCamPos, this.smoothing);
    this.camera.lookAt(shipPos);
  }

  /**
   * Orbit mode - orbit around ship
   */
  updateOrbitMode(shipPos) {
    this.orbitControls.target.copy(shipPos);
    this.orbitControls.update();
  }

  /**
   * Focus camera on a ship
   */
  focusOnShip(shipId, shipStates) {
    const state = shipStates[shipId];
    if (!state) return;

    const shipPos = this.sceneManager.getShipPosition(shipId);
    if (!shipPos) return;

    // Move camera to good viewing position
    const offset = new THREE.Vector3(100, 50, 100);
    this.camera.position.copy(shipPos).add(offset);
    this.orbitControls.target.copy(shipPos);
    this.orbitControls.update();
  }
}
