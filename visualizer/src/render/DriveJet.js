import * as THREE from 'three';

/**
 * DriveJet - the far exhaust jet of every burning torch, at its REAL size.
 *
 * The engine's drive plume (src/drive_plume.py) is not the bright flame at
 * the bells: it is a jet that washes anything within 7.5 degrees of dead
 * astern out to 100 km (a Gaussian 3 degree core; on axis at full burn
 * 40 GJ/s at 5 km, 16 at 10 km, 4.7 at 20 km, 0.8 at 50 km). On the
 * cinematic scale the oversize torch plume (10-110 km long) stood in for
 * it. With hulls at real size the torch is 0.2-1.5 km of flame, and a
 * destroyer burning 6 km astern of a battleship would glow for no visible
 * reason - so the jet itself is drawn: a faint cone of light with the
 * engine's geometry, as bright as (the square root of) the engine's
 * on-axis heat rate, scaled by the throttle the trace records.
 *
 * One instanced draw for the whole fleet. Each instance is a bounding cone
 * in scene km (never scaled with the hull: the jet is real); the fragment
 * shader integrates a Gaussian beam along its view ray through that cone
 * (24 steps, a fixed per-pixel offset, no noise and no clock: the same
 * frame at the same battle time, always), so it is right from the side,
 * looking down the jet and from inside it. Drawn in the auto and true
 * scale modes only.
 */

/** Half-angle of the jet's Gaussian core and its edge (src/drive_plume.py). */
export const JET_CORE_DEG = 3.0;
export const JET_MAX_DEG = 7.5;
/** Drawn length (km): the engine's jet reaches 100 km; past ~60 it heats by under 0.5 GJ/s. */
export const JET_REACH_KM = 60;
/** Peak radiance of the jet at full throttle (scene-linear, additive): a haze, far under the torch. */
export const JET_LEVEL = 0.18;

/**
 * Brightness of the jet `zKm` down its axis, 0..1: the square root of the
 * engine's on-axis heat rate relative to its 5 km near field (40 GJ/s),
 * i.e. about 0.63 at 10 km, 0.34 at 20 km, 0.14 at 50 km. Mirrored in the
 * fragment shader.
 */
export function jetBrightness(zKm) {
  const u = Math.max(0, zKm) / 8;
  return Math.pow(1 + u * u, -0.55);
}

const D2R = Math.PI / 180;
const TAN_CORE = Math.tan(JET_CORE_DEG * D2R);
/** The bounding cone is a little wider than the jet's edge so the Gaussian tail is not cut. */
const BOUND_TAN = Math.tan((JET_MAX_DEG + 2) * D2R);
/** Softening of the beam width (km) in the emission: no needle of light at the nozzle. */
const W_SOFT = 0.25;
/** The jet fades in over this length (km) behind the stern: the torch's flame is the picture there. */
const JET_FADE_IN_KM = 0.6;
/** Brightness looking down the jet saturates at this many times the across-the-beam value. */
const SAT = 1.3;

const vertexShader = /* glsl */`
  attribute vec4 aJet;      // throttle, nozzle radius (km), reach (km), unused
  varying vec3 vPos;
  varying vec3 vCam;
  varying vec4 vJet;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    // unit cone (apex at the origin, opening down -Z) shaped to this jet
    float z = -position.z * aJet.z;
    vec3 lp = vec3(position.xy * (aJet.y * 2.5 + z * ${BOUND_TAN.toFixed(5)}), -z);
    vPos = lp;
    vCam = (inverse(instanceMatrix) * vec4(cameraPosition, 1.0)).xyz;
    vJet = aJet;
    gl_Position = projectionMatrix * viewMatrix * instanceMatrix * vec4(lp, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform vec3 uColor;
  uniform float uLevel;
  varying vec3 vPos;
  varying vec3 vCam;
  varying vec4 vJet;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  #define STEPS 24
  // Emission per km of path at a point of the jet (z km down the axis, r km
  // off it): a Gaussian beam of the engine's 3 degree core. Dividing by its
  // width makes a line of sight ACROSS the beam equally bright wherever it
  // crosses (the gas thins as the beam widens) - times the axial falloff;
  // W_SOFT keeps the first few hundred metres, where the torch's own flame
  // is drawn, from being a needle of light.
  float emission(vec3 p) {
    float z = -p.z;
    if (z <= 0.0 || z >= vJet.z) return 0.0;
    float w = vJet.y + z * ${TAN_CORE.toFixed(6)};
    float r2 = dot(p.xy, p.xy);
    float u = z / 8.0;
    float axial = pow(1.0 + u * u, -0.55);                       // jetBrightness(z)
    // in over the first km (the torch's own flame is drawn there), out to nothing at the drawn reach
    float fade = smoothstep(0.0, ${JET_FADE_IN_KM.toFixed(1)}, z) * (1.0 - smoothstep(0.6 * vJet.z, vJet.z, z));
    return axial * fade * exp(-r2 / (w * w)) / (w + ${W_SOFT.toFixed(3)});
  }
  void main() {
    #include <logdepthbuf_fragment>
    vec3 ro = vCam;
    vec3 rd = normalize(vPos - vCam);
    // The chord of the view ray inside the bounding cone |p.xy| <= R0 + TB z,
    // 0 <= z <= reach (z = -p.z), found analytically: the mesh only says
    // which pixels to shade. (Taking the mesh's own back face as the exit
    // left dark seams along its edges, where the two disagree by rounding.)
    float R0 = vJet.y * 2.5, TB = ${BOUND_TAN.toFixed(5)};
    float tIn = 0.0, tOut = 1e9;
    // slab 0 <= z <= reach
    if (abs(rd.z) > 1e-9) {
      float ta = -ro.z / rd.z, tb = (-vJet.z - ro.z) / rd.z;
      tIn = max(tIn, min(ta, tb));
      tOut = min(tOut, max(ta, tb));
    } else if (ro.z > 0.0 || ro.z < -vJet.z) discard;
    // cone: qa t^2 + 2 qb t + qc <= 0
    float A0 = R0 - TB * ro.z, kz = -TB * rd.z;
    float qa = dot(rd.xy, rd.xy) - kz * kz;
    float qb = dot(ro.xy, rd.xy) - A0 * kz;
    float qc = dot(ro.xy, ro.xy) - A0 * A0;
    float disc = qb * qb - qa * qc;
    if (abs(qa) < 1e-9) {
      // parallel to the cone's side: one crossing
      if (abs(qb) > 1e-12) { float t = -qc / (2.0 * qb); if (qb > 0.0) tOut = min(tOut, t); else tIn = max(tIn, t); }
      else if (qc > 0.0) discard;
    } else if (disc <= 0.0) {
      if (qa > 0.0) discard;          // misses the cone (qa < 0: inside it all the way)
    } else {
      float sq = sqrt(disc);
      float r1 = (-qb - sq) / qa, r2 = (-qb + sq) / qa;
      if (qa > 0.0) {                 // inside between the roots
        tIn = max(tIn, r1); tOut = min(tOut, r2);
      } else {                        // steeper than the cone: inside before the first or after the second (r2 < r1 here)
        float lo = min(r1, r2), hi = max(r1, r2);
        // the slab keeps one nappe: at most one of the two pieces survives
        if (min(tOut, lo) > tIn) tOut = min(tOut, lo); else tIn = max(tIn, hi);
      }
    }
    if (!(tOut > tIn)) discard;
    // March the chord (a fixed, per-pixel offset hides the steps; no time
    // in it). The light is concentrated at the upstream end of the jet, so
    // a ray that runs along the axis is stepped in z with the samples
    // packed toward that end (z = z0 + dz u^2); a ray across it, evenly.
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float column = 0.0;
    if (abs(rd.z) > 0.15) {
      float zA = -(ro.z + rd.z * tIn), zB = -(ro.z + rd.z * tOut);
      float z0 = min(zA, zB), dz = abs(zB - zA);
      for (int i = 0; i < STEPS; i++) {
        float u = (float(i) + jitter) / float(STEPS);
        float z = z0 + dz * u * u;
        column += emission(ro + rd * ((-z - ro.z) / rd.z)) * u;
      }
      column *= 2.0 * dz / (float(STEPS) * abs(rd.z));
    } else {
      float dt = (tOut - tIn) / float(STEPS);
      for (int i = 0; i < STEPS; i++) column += emission(ro + rd * (tIn + (float(i) + jitter) * dt));
      column *= dt;
    }
    // sqrt(pi) is the column straight across the core: there a = uLevel x
    // throttle x axial. Looking down the jet the column is far longer; it
    // saturates at SAT times that (the glare of looking into a torch).
    float a = uLevel * vJet.x * ${SAT.toFixed(1)} * (1.0 - exp(-column / (1.7724539 * ${SAT.toFixed(1)})));
    if (a < 0.0005) discard;
    gl_FragColor = vec4(uColor * a, 1.0);
  }
`;

const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _f = new THREE.Vector3();
const _one = new THREE.Vector3(1, 1, 1);
const _m = new THREE.Matrix4();

export class DriveJets {
  constructor(scene, capacity = 32) {
    this.scene = scene;
    this.capacity = capacity;
    // Closed unit cylinder along Z (z 0 .. -1, radius 1); the vertex shader
    // shapes it into the jet's bounding cone. Closed and drawn by its back
    // faces: every view ray through the (convex) volume leaves through
    // exactly one of them, also from inside the jet or looking down it.
    const cone = new THREE.CylinderGeometry(1, 1, 1, 20, 1, false);
    cone.rotateX(Math.PI / 2);
    cone.translate(0, 0, -0.5);
    const g = new THREE.InstancedBufferGeometry();
    g.index = cone.index;
    g.setAttribute('position', cone.attributes.position);
    this.jetAttr = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
    this.jetAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aJet', this.jetAttr);
    this.material = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(0.42, 0.56, 1.0) }, uLevel: { value: JET_LEVEL } },
      vertexShader, fragmentShader,
      transparent: true, depthWrite: false, depthTest: true, side: THREE.BackSide,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
    });
    this.mesh = new THREE.InstancedMesh(g, this.material, capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 17;
    this.mesh.count = 0;
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  /**
   * One jet per burning ship, from the frame's own state: the stern of the
   * hull as drawn, the hull's axis, the recorded throttle.
   * @param {Map<string, THREE.Group>} ships - SceneManager.ships
   * @param {boolean} on - false on the cinematic scale (the torch plume is the jet there)
   */
  update(ships, on) {
    let n = 0;
    if (on && ships) {
      const A = this.jetAttr.array;
      for (const ship of ships.values()) {
        if (n >= this.capacity) break;
        const ud = ship.userData;
        const thrust = ud.smoothThrust || 0;
        if (!ship.visible || thrust < 0.02 || ud.dyingNow) continue;
        const L = ud.drawn?.length ?? 0.1, W = ud.drawn?.width ?? 0.02;   // the hull as drawn (km)
        _f.set(0, 0, 1).applyQuaternion(ship.quaternion);
        _p.copy(ship.position).addScaledVector(_f, -0.5 * L);
        // (hull frame: +Z is the nose, so the jet runs down the instance's -Z)
        _m.compose(_p, _q.copy(ship.quaternion), _one);
        this.mesh.setMatrixAt(n, _m);
        A[n * 4] = thrust; A[n * 4 + 1] = 0.5 * W; A[n * 4 + 2] = JET_REACH_KM; A[n * 4 + 3] = 0;
        n++;
      }
    }
    this.mesh.count = n;
    this.mesh.visible = n > 0;
    if (n > 0) {
      this.mesh.instanceMatrix.needsUpdate = true;
      this.jetAttr.needsUpdate = true;
    }
  }

  dispose() {
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.mesh.dispose?.();
  }
}
