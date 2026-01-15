/**
 * Interpolator - Smoothly interpolates between 1Hz simulation frames
 */
export class Interpolator {
  /**
   * Linear interpolation between two values
   */
  static lerp(a, b, t) {
    return a + (b - a) * t;
  }

  /**
   * Linear interpolation for 3D vectors (arrays)
   */
  static lerpVec3(a, b, t) {
    return [
      this.lerp(a[0], b[0], t),
      this.lerp(a[1], b[1], t),
      this.lerp(a[2], b[2], t)
    ];
  }

  /**
   * Spherical linear interpolation for direction vectors
   * Used for orientation to avoid flipping
   */
  static slerpVec3(a, b, t) {
    // Normalize inputs
    const aMag = Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    const bMag = Math.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);

    if (aMag === 0 || bMag === 0) return a;

    const aNorm = [a[0]/aMag, a[1]/aMag, a[2]/aMag];
    const bNorm = [b[0]/bMag, b[1]/bMag, b[2]/bMag];

    // Dot product
    let dot = aNorm[0]*bNorm[0] + aNorm[1]*bNorm[1] + aNorm[2]*bNorm[2];
    dot = Math.max(-1, Math.min(1, dot)); // Clamp

    // If very close, just lerp
    if (dot > 0.9995) {
      return this.lerpVec3(aNorm, bNorm, t);
    }

    const theta = Math.acos(dot);
    const sinTheta = Math.sin(theta);

    const s0 = Math.sin((1 - t) * theta) / sinTheta;
    const s1 = Math.sin(t * theta) / sinTheta;

    return [
      s0 * aNorm[0] + s1 * bNorm[0],
      s0 * aNorm[1] + s1 * bNorm[1],
      s0 * aNorm[2] + s1 * bNorm[2]
    ];
  }

  /**
   * Interpolate ship states between two frames
   * @param {Object} frame0 - Earlier frame
   * @param {Object} frame1 - Later frame
   * @param {number} alpha - Interpolation factor (0-1)
   * @returns {Object} Interpolated ship states
   */
  static interpolateShips(frame0, frame1, alpha) {
    const result = {};

    if (!frame0.ships || !frame1.ships) return result;

    for (const [shipId, state0] of Object.entries(frame0.ships)) {
      const state1 = frame1.ships[shipId];
      if (!state1) continue;

      // If destroyed in frame1, show destroyed state
      if (state1.destroyed) {
        result[shipId] = { ...state1, interpolated: true };
        continue;
      }

      result[shipId] = {
        position: this.lerpVec3(state0.pos, state1.pos, alpha),
        velocity: this.lerpVec3(state0.vel, state1.vel, alpha),
        forward: this.slerpVec3(state0.fwd, state1.fwd, alpha),
        thrust: this.lerp(state0.thrust || 0, state1.thrust || 0, alpha),
        hull: this.lerp(state0.hull || 100, state1.hull || 100, alpha),
        maneuver: state0.maneuver || 'MAINTAIN',
        destroyed: state1.destroyed,
        name: state1.name || shipId,
        interpolated: true
      };
    }

    return result;
  }

  /**
   * Interpolate projectile states between frames
   * Handles spawning and despawning
   * @param {Array} proj0 - Projectiles in earlier frame
   * @param {Array} proj1 - Projectiles in later frame
   * @param {number} alpha - Interpolation factor
   * @returns {Array} Interpolated projectiles
   */
  static interpolateProjectiles(proj0, proj1, alpha) {
    const result = [];
    const proj0Map = new Map((proj0 || []).map(p => [p.id, p]));
    const proj1Map = new Map((proj1 || []).map(p => [p.id, p]));

    // All projectiles that exist in frame1
    for (const [id, p1] of proj1Map) {
      const p0 = proj0Map.get(id);

      if (p0) {
        // Projectile exists in both frames - interpolate
        result.push({
          id: id,
          position: this.lerpVec3(p0.pos, p1.pos, alpha),
          velocity: p1.vel,
          mass: p1.mass_kg,
          source: p1.source,
          target: p1.target,
          pdEngaged: p1.pd_engaged,
          pdDamage: p1.pd_damage_kg,
          isNew: false
        });
      } else {
        // New projectile - spawn at current position
        result.push({
          id: id,
          position: p1.pos,
          velocity: p1.vel,
          mass: p1.mass_kg,
          source: p1.source,
          target: p1.target,
          pdEngaged: p1.pd_engaged,
          pdDamage: p1.pd_damage_kg,
          isNew: true
        });
      }
    }

    return result;
  }

  /**
   * Get fully interpolated state at a given time
   * @param {Object} frameData - Result from BattleLoader.getFrameAt()
   * @param {Object} loader - BattleLoader instance for extrapolation (optional)
   * @returns {Object} Interpolated state
   */
  static getInterpolatedState(frameData, loader = null) {
    if (!frameData) return { ships: {}, projectiles: [] };

    const { frame0, frame1, alpha } = frameData;
    const time = this.lerp(frame0.t, frame1.t, alpha);

    // Get interpolated projectiles from sim_trace
    let projectiles = this.interpolateProjectiles(frame0.projectiles, frame1.projectiles, alpha);

    // Add extrapolated projectiles (those heading to impact but no longer in trace)
    if (loader) {
      const extrapolated = loader.getExtrapolatedProjectiles(time);
      for (const ep of extrapolated) {
        // Only add if not already in the projectiles list
        if (!projectiles.find(p => p.id === ep.id)) {
          projectiles.push({
            id: ep.id,
            position: ep.pos,
            velocity: ep.vel,
            mass: ep.mass_kg,
            source: ep.source,
            target: ep.target,
            pdEngaged: false,
            pdDamage: 0,
            isNew: false,
            extrapolated: true
          });
        }
      }
    }

    return {
      ships: this.interpolateShips(frame0, frame1, alpha),
      projectiles: projectiles,
      time: time
    };
  }
}
