import * as THREE from 'three';

/**
 * Fitting the sun's shadow map to the view (pure math, no renderer).
 *
 * The battle is hundreds of km across and a hull a few units long, so one
 * shadow map cannot cover it. The orthographic shadow box is instead refit
 * every frame to what the camera looks at: a square around the orbit focus,
 * sized to the view at focus depth - about one shadow texel per screen
 * pixel at any zoom. Its half-size moves in quarter-octave steps and its
 * centre snaps to whole texels, so shadow edges do not crawl while the
 * camera orbits, pans or eases in.
 */

export const SHADOW_MIN_RADIUS = 4;      // km: closest zoom on a corvette
export const SHADOW_MAX_RADIUS = 4096;
export const SHADOW_DEPTH_MARGIN = 60;   // casters up-sun of the box (> 2 hull lengths)

/** Half-extent (km) of the shadow box for a camera `viewDist` from its focus. */
export function shadowBoxRadius(viewDist, fovDeg, aspect) {
  const fovT = Math.tan(THREE.MathUtils.degToRad(fovDeg) * 0.5);
  const raw = Math.max(viewDist, 1e-3) * fovT * Math.max(aspect, 1) * 1.08;
  const r = THREE.MathUtils.clamp(raw, SHADOW_MIN_RADIUS, SHADOW_MAX_RADIUS);
  return Math.pow(2, Math.ceil(Math.log2(r) * 4 - 1e-9) / 4);
}

/**
 * Light-space axes exactly as three's LightShadow builds them (lookAt from
 * the light toward its target with world up): z points at the sun.
 */
export function shadowBasis(sunDir) {
  const z = sunDir.clone().normalize();
  const x = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), z).normalize();
  const y = new THREE.Vector3().crossVectors(z, x).normalize();
  return { x, y, z };
}

/**
 * Fit the box: writes the texel-snapped box centre into `outTarget`, the
 * light position into `outPosition`, and returns the numbers the shadow
 * camera and bias need. Biases scale with the world size of a texel, so
 * the same settings hold from a corvette close-up to the fleet view.
 */
export function fitSunShadow(focus, basis, radius, mapSize, outTarget, outPosition) {
  const texel = (2 * radius) / mapSize;
  const sx = Math.round(focus.dot(basis.x) / texel) * texel;
  const sy = Math.round(focus.dot(basis.y) / texel) * texel;
  const sz = focus.dot(basis.z);
  const depth = radius + SHADOW_DEPTH_MARGIN;
  outTarget.set(0, 0, 0)
    .addScaledVector(basis.x, sx).addScaledVector(basis.y, sy).addScaledVector(basis.z, sz);
  outPosition.copy(outTarget).addScaledVector(basis.z, depth);
  const near = 0.5, far = 2 * depth;
  return { texel, near, far, normalBias: texel * 1.2, bias: -texel * 0.6 / (far - near) };
}
