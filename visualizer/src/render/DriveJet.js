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
 * shader finds, for its view ray, the point of closest approach to the jet
 * axis and shades a Gaussian beam there - analytic, no ray march, no noise
 * (the same frame at the same battle time, always). Drawn in the auto and
 * true scale modes only.
 */

/** Half-angle of the jet's Gaussian core and its edge (src/drive_plume.py). */
export const JET_CORE_DEG = 3.0;
export const JET_MAX_DEG = 7.5;
/** Drawn length (km): the engine's jet reaches 100 km; past ~60 it heats by under 0.5 GJ/s. */
export const JET_REACH_KM = 60;
/** Peak radiance of the jet at full throttle (scene-linear, additive): a haze, far under the torch. */
export const JET_LEVEL = 0.15;

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

const vertexShader = /* glsl */`
  attribute vec4 aJet;      // throttle, nozzle radius (km), reach (km), fade-in length (km)
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
  void main() {
    #include <logdepthbuf_fragment>
    vec3 rd = normalize(vPos - vCam);
    // closest approach of the view ray (vCam + t rd) to the jet axis (0, 0, -s)
    float b = -rd.z;                 // rd . axis
    float d = dot(rd, vCam);
    float e = -vCam.z;               // axis . vCam
    float denom = max(1.0 - b * b, 1e-4);
    float s = clamp((e - b * d) / denom, 0.0, vJet.z);
    float t = max(b * s - d, 0.0);   // the ray point nearest that axis point, in front of the lens
    vec3 p = vCam + rd * t;
    s = clamp(-p.z, 0.0, vJet.z);
    float r = length(p.xy);
    float w = vJet.y + s * ${TAN_CORE.toFixed(6)};          // Gaussian radius: the engine's 3 degree core
    float beam = exp(-(r * r) / (w * w));
    float u = s / 8.0;
    float axial = pow(1.0 + u * u, -0.55);                   // jetBrightness(s)
    // out of the torch's own flame, and out to nothing at the drawn reach
    float ends = smoothstep(0.0, vJet.w, s) * (1.0 - smoothstep(0.6 * vJet.z, vJet.z, s));
    // a ray along the jet looks down more of it than one across (capped: end-on is not a searchlight)
    float along = min(inversesqrt(denom), 3.0);
    float a = uLevel * vJet.x * beam * axial * ends * along;
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
        const L = ud.drawn?.length ?? 0.1, W = ud.drawn?.width ?? 0.02;
        _f.set(0, 0, 1).applyQuaternion(ship.quaternion);
        _p.copy(ship.position).addScaledVector(_f, -0.5 * L);
        // (hull frame: +Z is the nose, so the jet runs down the instance's -Z)
        _m.compose(_p, _q.copy(ship.quaternion), _one);
        this.mesh.setMatrixAt(n, _m);
        // fade in over the torch's own plume (as drawn), so the haze starts where the flame thins
        const flame = Math.max(0.05, (ud.plumeLength || 4) * (ud.k ?? 1) * (0.14 + 0.86 * thrust));
        A[n * 4] = thrust; A[n * 4 + 1] = 0.5 * W; A[n * 4 + 2] = JET_REACH_KM; A[n * 4 + 3] = flame;
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
