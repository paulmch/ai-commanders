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

// The three lengths below were tuned on the kit-size hulls (4-21 km long).
// Every function takes `unit`, the scene km per kit unit of the scale mode
// (1 cinematic, 0.016 with real-size hulls: render/Scale.js), so the box
// can close in on a 65 m corvette and its near plane stays off the hull.
export const SHADOW_MIN_RADIUS = 4;      // kit units: closest zoom on a corvette
export const SHADOW_MAX_RADIUS = 4096;   // km
export const SHADOW_DEPTH_MARGIN = 60;   // kit units: casters up-sun of the box (> 2 hull lengths)
export const SHADOW_NEAR = 0.5;          // kit units

/** Half-extent (km) of the shadow box for a camera `viewDist` from its focus. */
export function shadowBoxRadius(viewDist, fovDeg, aspect, unit = 1) {
  const fovT = Math.tan(THREE.MathUtils.degToRad(fovDeg) * 0.5);
  const raw = Math.max(viewDist, 1e-3 * unit) * fovT * Math.max(aspect, 1) * 1.08;
  return quantiseShadowRadius(raw, unit);
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
export function fitSunShadow(focus, basis, radius, mapSize, outTarget, outPosition, unit = 1) {
  const texel = (2 * radius) / mapSize;
  const sx = Math.round(focus.dot(basis.x) / texel) * texel;
  const sy = Math.round(focus.dot(basis.y) / texel) * texel;
  const sz = focus.dot(basis.z);
  // With real-size hulls the auto scale can draw a distant hull about as
  // long as the box is wide: the margin then grows with the box, so its
  // casters stay inside. (The cinematic fit, unit 1, is unchanged.)
  const margin = unit === 1 ? SHADOW_DEPTH_MARGIN : Math.max(SHADOW_DEPTH_MARGIN * unit, 2 * radius);
  const depth = radius + margin;
  outTarget.set(0, 0, 0)
    .addScaledVector(basis.x, sx).addScaledVector(basis.y, sy).addScaledVector(basis.z, sz);
  outPosition.copy(outTarget).addScaledVector(basis.z, depth);
  const near = SHADOW_NEAR * unit, far = 2 * depth;
  return { texel, near, far, normalBias: texel * 1.2, bias: -texel * 0.6 / (far - near) };
}

/**
 * What the shadow box should cover this frame.
 *
 * The orbit focus alone is not enough: harness and director shots aim the
 * camera with a look-at point that can sit far behind the ships, and a wide
 * view of one hull wastes most of the map on empty space. So the box is
 * fitted to the hulls that matter on screen: the one with the largest
 * angular size, plus every other in-frustum hull at least 30% as large that
 * fits without growing the box past 3.5x the first hull. Zoomed inside the
 * main hull (the view box is smaller than the hull) the view fit wins, as
 * before; with no hull in view it is the only fit.
 *
 * `subjects` is an array of { pos: Vector3, r: km }; `n` how many are live.
 * Writes the centre into `outCenter` and returns the (unquantised) radius.
 */
export function chooseShadowFocus(camera, target, viewDist, subjects, n, outCenter, unit = 1) {
  const viewR = Math.max(viewDist, 1e-3 * unit) * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5)
    * Math.max(camera.aspect, 1) * 1.08;
  outCenter.copy(target);
  if (!n) return viewR;
  const e = camera.matrixWorld.elements;
  const tanY = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5), tanX = tanY * camera.aspect;
  let best = -1, bestA = 0;
  for (let i = 0; i < n; i++) {
    const s = subjects[i];
    const dx = s.pos.x - e[12], dy = s.pos.y - e[13], dz = s.pos.z - e[14];
    const depth = -(dx * e[8] + dy * e[9] + dz * e[10]);
    const sx = dx * e[0] + dy * e[1] + dz * e[2], sy = dx * e[4] + dy * e[5] + dz * e[6];
    s.a = 0;
    if (depth <= 0.01 * unit) continue;
    if (Math.abs(sx) > depth * tanX + s.r || Math.abs(sy) > depth * tanY + s.r) continue;
    s.a = s.r / depth;
    if (s.a > bestA) { bestA = s.a; best = i; }
  }
  if (best < 0) return viewR;
  const first = subjects[best];
  const r0 = first.r * 1.15;
  if (viewR <= r0 && target.distanceTo(first.pos) <= first.r * 1.2) return viewR;
  outCenter.copy(first.pos);
  let R = r0;
  for (let i = 0; i < n; i++) {
    const s = subjects[i];
    if (i === best || s.a < 0.3 * bestA) continue;
    // smallest sphere around the current one and this hull
    const d = outCenter.distanceTo(s.pos), rs = s.r * 1.15;
    if (d + rs <= R) continue;
    const merged = Math.max((R + d + rs) * 0.5, rs);
    if (merged > 3.5 * r0) continue;
    if (d > 1e-6) outCenter.lerp(s.pos, (merged - R) / d);
    R = merged;
  }
  return R;
}

/** Quarter-octave quantisation of a box radius (see shadowBoxRadius). */
export function quantiseShadowRadius(r, unit = 1) {
  const c = THREE.MathUtils.clamp(r, SHADOW_MIN_RADIUS * unit, SHADOW_MAX_RADIUS);
  return Math.pow(2, Math.ceil(Math.log2(c) * 4 - 1e-9) / 4);
}
