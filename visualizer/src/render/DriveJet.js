import * as THREE from 'three';
import {
  JET_REACH_KM, JET_LEVEL, JET_NEAR_KM, JET_FALL, JET_CAP, JET_PIXEL_FLOOR, JET_SEGMENTS, JET_GAIN_SIN, JET_LENS_CLEAR,
  TAN_CORE, BOUND_TAN, BOUND_R0
} from './PlumeProfile.js';
import { getGraphicsSettings } from './Quality.js';
import { Ribbon } from './Trails.js';

export { JET_CORE_DEG, JET_MAX_DEG, JET_REACH_KM, jetAxial } from './PlumeProfile.js';

/**
 * DriveJet - the spear of every burning torch: the far half of the one
 * plume render/PlumeProfile.js describes (the torch flame, render/Torch.js,
 * is the near half; both start at the nozzle and only fade from there).
 *
 * The engine's drive plume (src/drive_plume.py) washes anything within 7.5
 * degrees of dead astern out to 100 km (a Gaussian 3 degree core). The jet
 * draws that core: a Gaussian beam whose radius is the drive's DRAWN root radius (PlumeProfile.driveRadius)
 * at the nozzle plus z tan 3 deg, so near the ship it is as wide as the
 * drawn bells (magnified with the hull in the auto scale mode) and far
 * astern it is the real cone. Its side-on brightness is jetAxial(z): bright
 * at the bells, falling steadily from there - no fade-in, so no hole after
 * the flame, and no second maximum, so no cloud in the distance.
 *
 * One instanced draw for the whole fleet. Each instance is a bounding cone
 * in scene km; the fragment shader integrates the beam along its view ray
 * in closed form (erf over a few sub-chords spaced in log z: the metres-wide
 * root and the kilometres-wide far field are both resolved), with a pixel
 * floor that keeps the energy (a far spear is a faint hairline, not a fat
 * line). No noise and no clock: the same frame at the same battle time,
 * always - a steady lance. JS twin: PlumeProfile.jetColumn (tests).
 * ?jets=0 hides it (SceneManager).
 */

const f = (x, n = 6) => x.toFixed(n);

const vertexShader = /* glsl */`
  attribute vec4 aJet;      // throttle, drive radius at the nozzle (km), reach (km), unused
  uniform float uPixelWorld;
  varying vec3 vPos;
  varying vec3 vCam;
  varying vec4 vJet;
  varying vec2 vBound;      // bounding cone radius at the nozzle, opening slope (padded for the pixel floor)
  varying float vInside;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vCam = (inverse(instanceMatrix) * vec4(cameraPosition, 1.0)).xyz;
    // pad the bound for the pixel floor: 2 px at every point (triangle inequality)
    float px = 2.0 * ${f(JET_PIXEL_FLOOR, 3)} * uPixelWorld;
    float R0 = ${f(BOUND_R0, 2)} * aJet.y + px * length(vCam);
    float TB = ${f(BOUND_TAN)} + px;
    // unit cylinder (z 0 .. -1, radius 1) shaped into the cone
    float z = -position.z * aJet.z;
    vec3 lp = vec3(position.xy * (R0 + z * TB), -z);
    vPos = lp;
    vJet = aJet;
    vBound = vec2(R0, TB);
    // camera inside the cone: its front faces are behind the lens, the back faces carry the jet
    float zc = -vCam.z;
    float margin = 0.002 + aJet.y;
    vInside = (zc > -margin && zc < aJet.z + margin
      && length(vCam.xy) < R0 + TB * clamp(zc, 0.0, aJet.z) + margin) ? 1.0 : 0.0;
    gl_Position = projectionMatrix * viewMatrix * instanceMatrix * vec4(lp, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform vec3 uColor;
  uniform vec3 uHot;
  uniform float uPixelWorld;
  uniform int uSeg;
  varying vec3 vPos;
  varying vec3 vCam;
  varying vec4 vJet;
  varying vec2 vBound;
  varying float vInside;
  #include <common>
  #include <logdepthbuf_pars_fragment>

  float erfa(float x) {
    float a = abs(x);
    float d = 1.0 + a * (0.278393 + a * (0.230389 + a * (0.000972 + a * 0.078108)));
    d *= d; d *= d;
    return sign(x) * (1.0 - 1.0 / d);
  }
  // series near 0: the difference of two small arguments stays exact
  float erfS(float x) {
    if (abs(x) < 0.5) {
      float x2 = x * x;
      return 1.1283792 * x * (1.0 - x2 * (0.33333333 - x2 * (0.1 - x2 * 0.0238095)));
    }
    return erfa(x);
  }
  // side-on centreline radiance z km aft (PlumeProfile.jetAxial)
  float axial(float z) {
    float fall = pow(max(1.0 + z / ${f(JET_NEAR_KM, 4)}, 1.0), -${f(JET_FALL, 4)});
    float x = clamp((z - 0.6 * vJet.z) / (0.4 * vJet.z), 0.0, 1.0);
    return ${f(JET_LEVEL, 4)} * vJet.x * fall * (1.0 - x * x * (3.0 - 2.0 * x));
  }

  void main() {
    // convex volume: exactly one face per pixel - front faces from outside,
    // back faces once the camera is inside the cone
    if (gl_FrontFacing == (vInside > 0.5)) discard;
    #include <logdepthbuf_fragment>
    #if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
      // inside the jet the gas starts at the lens: nothing can occlude it
      if (vInside > 0.5) gl_FragDepthEXT = 0.0;
    #endif
    vec3 ro = vCam;
    vec3 toFrag = vPos - vCam;
    vec3 rd = toFrag / max(length(toFrag), 1e-9);   // (never normalize(0): NaN on real GPUs)
    // chord of the view ray inside the bounding cone |p.xy| <= R0 + TB z,
    // 0 <= z <= reach (z = -p.z), found analytically (PlumeProfile.jetChord)
    float R0 = vBound.x, TB = vBound.y, reach = vJet.z;
    float tIn = 0.0, tOut = 1e9;
    if (abs(rd.z) > 1e-9) {
      float ta = -ro.z / rd.z, tb = (-reach - ro.z) / rd.z;
      tIn = max(tIn, min(ta, tb));
      tOut = min(tOut, max(ta, tb));
    } else if (ro.z > 0.0 || ro.z < -reach) discard;
    float A0 = R0 - TB * ro.z, kz = -TB * rd.z;
    float qa = dot(rd.xy, rd.xy) - kz * kz;
    float qb = dot(ro.xy, rd.xy) - A0 * kz;
    float qc = dot(ro.xy, ro.xy) - A0 * A0;
    float disc = qb * qb - qa * qc;
    if (abs(qa) < 1e-9) {
      if (abs(qb) > 1e-12) { float t = -qc / (2.0 * qb); if (qb > 0.0) tOut = min(tOut, t); else tIn = max(tIn, t); }
      else if (qc > 0.0) discard;
    } else if (disc <= 0.0) {
      if (qa > 0.0) discard;
    } else {
      float sq = sqrt(max(disc, 0.0));
      float r1 = (-qb - sq) / qa, r2 = (-qb + sq) / qa;
      if (qa > 0.0) {
        tIn = max(tIn, r1); tOut = min(tOut, r2);
      } else {
        float lo = min(r1, r2), hi = max(r1, r2);
        if (min(tOut, lo) > tIn) tOut = min(tOut, lo); else tIn = max(tIn, hi);
      }
    }
    tIn = max(tIn, 0.0);
    if (!(tOut > tIn)) discard;

    // closest approach of the view line to the axis
    float dd = dot(rd.xy, rd.xy);
    float sinT = sqrt(dot(rd.xy, rd.xy));
    bool par = sinT <= 1e-4;
    // line-of-sight gain held at 1 / JET_GAIN_SIN (PlumeProfile.js): no pile-up toward the vanishing point
    float gain = sinT / max(sinT, ${f(JET_GAIN_SIN, 3)});
    float tc = par ? 0.0 : -dot(ro.xy, rd.xy) / max(dd, 1e-12);
    vec2 lq = ro.xy + rd.xy * tc;
    float d2line = dot(lq, lq);
    float w0 = vJet.y;
    float zA = max(-(ro.z + rd.z * tIn), 0.0), zB = max(-(ro.z + rd.z * tOut), 0.0);
    float c = min(${f(JET_NEAR_KM, 4)}, w0 / ${f(TAN_CORE)}) + 1e-6;
    float lA = log(max(zA + c, 1e-6)), lB = log(max(zB + c, 1e-6)), dzAB = zB - zA;
    // a chord nearly across the axis spans almost no z: plain even sub-chords
    // (also keeps fp32 cancellation in exp(log(z + c)) - c - zA out)
    bool even = abs(dzAB) <= 1e-4 * (min(zA, zB) + c);
    float px = ${f(JET_PIXEL_FLOOR, 3)} * uPixelWorld;
    float col = 0.0, hot = 0.0;
    float fa = 0.0;
    for (int i = 0; i < 8; i++) {
      if (i >= uSeg) break;
      float u1 = float(i + 1) / float(uSeg), um = (float(i) + 0.5) / float(uSeg);
      float fb = even ? u1 : clamp((exp(mix(lA, lB, u1)) - c - zA) / dzAB, 0.0, 1.0);
      float fm = even ? um : clamp((exp(mix(lA, lB, um)) - c - zA) / dzAB, 0.0, 1.0);
      float a = mix(tIn, tOut, fa), b = mix(tIn, tOut, fb), tm = mix(tIn, tOut, fm);
      fa = fb;
      // width and brightness halfway between the sub-chord's point nearest
      // the axis and its (log) middle: continuous in the pixel
      float te = par ? tm : 0.5 * (clamp(tc, a, b) + tm);
      float z = max(-(ro.z + rd.z * te), 0.0);
      float w = w0 + z * ${f(TAN_CORE)};
      float wf = max(w, px * te);
      float S = axial(z) * (w / wf);
      float d2, G;
      if (par) {
        vec2 q = ro.xy + rd.xy * te;
        d2 = dot(q, q);
        G = (b - a) / (1.7724539 * wf);
      } else {
        float k = sinT / wf;
        d2 = d2line;
        G = 0.5 * (erfS(k * (b - tc)) - erfS(k * (a - tc))) / sinT;
      }
      // no veil when the lens sits in the beam: gas within a few radii of it is not drawn
      // (at the sub-chord's point nearest the axis, where its Gaussian mass is)
      float tl = par ? tm : clamp(tc, a, b);
      float rc = ${f(JET_LENS_CLEAR, 3)} * wf / max(sinT, ${f(JET_GAIN_SIN, 3)});   // the beam's length along a grazing ray
      float lens = 1.0 - exp(-(tl * tl) / (rc * rc));
      float dc = S * exp(-d2 / (wf * wf)) * G * gain * lens;
      col += dc;
      hot += dc * exp(-z / ${f(0.6 * JET_NEAR_KM, 4)});
    }
    // looking down the axis the column is the whole jet deep: a soft
    // ceiling keeps that a bright disc at the ship, never a white-out
    float a = ${f(JET_CAP, 3)} * (1.0 - exp(-col / ${f(JET_CAP, 3)}));
    if (a < 0.0004) discard;
    vec3 tint = mix(uColor, uHot, clamp(hot / max(col, 1e-9), 0.0, 1.0));
    gl_FragColor = vec4(tint * a, 1.0);
  }
`;

const _q = new THREE.Quaternion();
const _f = new THREE.Vector3();
const _one = new THREE.Vector3(1, 1, 1);
const _m = new THREE.Matrix4();

/** Nozzle plane of a hull, in its own length aft of the centre (SceneManager places the bells there). */
export const NOZZLE_AFT = 0.53;

/**
 * The jet of one ship this frame, a pure function of the ship's drawn state:
 * origin at the nozzle plane of the hull as drawn, axis straight down the
 * hull's -Z (its thrust axis: the whole spear swings with the hull, no lag),
 * radius = the drawn drive radius. Writes into `out` {pos, quat, w0} or
 * returns null when there is no jet.
 */
export function jetPose(ship, out) {
  const ud = ship.userData;
  const thr = ud.plumeThrottle ?? ud.smoothThrust ?? 0;
  if (!ship.visible || !(thr >= 0.02) || ud.dyingNow) return null;
  const L = ud.drawn?.length ?? 0.1;
  const k = ud.k ?? 1;
  _f.set(0, 0, 1).applyQuaternion(ship.quaternion);
  out.pos.copy(ship.position).addScaledVector(_f, -NOZZLE_AFT * L);
  out.quat.copy(ship.quaternion);
  out.w0 = (ud.driveRadius ?? 0.25 * (ud.size?.width ?? 0.4 * L / k)) * k;
  out.throttle = Math.min(1, thr);
  return out;
}
const _pose = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), w0: 0, throttle: 0 };

const SEG = JET_SEGMENTS;

export class DriveJets {
  constructor(scene, capacity = 32) {
    this.scene = scene;
    this.capacity = capacity;
    // Closed unit cylinder along Z (z 0 .. -1, radius 1); the vertex shader
    // shapes it into the jet's bounding cone. Drawn front faces from
    // outside, back faces from inside (convex: one face per pixel).
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
      uniforms: {
        uColor: { value: new THREE.Color(0.32, 0.5, 1.0) },
        uHot: { value: new THREE.Color(0.5, 0.68, 1.0) },
        uPixelWorld: Ribbon.pixelWorld,
        uSeg: { value: SEG.high }
      },
      vertexShader, fragmentShader,
      transparent: true, depthWrite: false, depthTest: true, side: THREE.DoubleSide,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
    });
    this.mesh = new THREE.InstancedMesh(g, this.material, capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 17;
    this.mesh.count = 0;
    this.mesh.visible = false;
    // sub-chords follow the graphics tier (read-only use of Quality.js)
    const u = this.material.uniforms;
    this.mesh.onBeforeRender = (renderer) => {
      const s = SEG[getGraphicsSettings(renderer).level] || SEG.high;
      if (u.uSeg.value !== s) u.uSeg.value = s;
    };
    scene.add(this.mesh);
  }

  /**
   * One jet per burning ship, from the frame's own state (jetPose).
   * @param {Map<string, THREE.Group>} ships - SceneManager.ships
   * @param {boolean} on - false hides every jet (?jets=0)
   */
  update(ships, on) {
    let n = 0;
    if (on && ships) {
      const A = this.jetAttr.array;
      for (const ship of ships.values()) {
        if (n >= this.capacity) break;
        if (!jetPose(ship, _pose)) continue;
        _m.compose(_pose.pos, _q.copy(_pose.quat), _one);
        this.mesh.setMatrixAt(n, _m);
        A[n * 4] = _pose.throttle; A[n * 4 + 1] = _pose.w0; A[n * 4 + 2] = JET_REACH_KM; A[n * 4 + 3] = 0;
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
