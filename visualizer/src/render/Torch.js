import * as THREE from 'three';
import { getNoiseTexture } from './Textures.js';
import { getGraphicsSettings } from './Quality.js';
import { Ribbon } from './Trails.js';

/**
 * Fusion torch exhaust as a real volume. Each plume is ONE draw: a convex
 * bounding cone whose fragments integrate an analytic emission field along
 * the view ray, in the torch's own frame (nozzle at the origin, exhaust
 * down -Z).
 *
 * The field is three nested Gaussian tubes - a white-hot core that pinches
 * into shock diamonds, a blue sheath that fans out as the exhaust expands
 * into vacuum, a wide violet halo - plus a hot ball at the throat. A ray
 * through a Gaussian tube has a closed-form line integral (erf), so there
 * is no ray-march and no banding: the ray's chord through the plume is cut
 * into a few sub-chords only so the axial profile (decay, diamonds,
 * turbulence) is sampled along it. Side-on the sub-chord holding the ray's
 * closest approach to the axis carries the whole integral, sampled exactly
 * there; end-on every sub-chord contributes and the plume becomes the
 * bright disc it should be. The column is normalised so a layer's
 * intensity is its side-on centreline radiance in scene HDR units.
 *
 * Wall-clock time only scrolls the turbulence; length, width, colour and
 * the dying-drive cough are functions of the throttle and of battle time.
 */

const vertexShader = /* glsl */`
  uniform float uLen;
  uniform float uR0;
  uniform float uW;
  uniform vec2 uBend;
  uniform float uPixelWorld;
  varying vec3 vPos;
  varying vec3 vCam;
  varying float vInside;
  #include <common>
  #include <logdepthbuf_pars_vertex>

  // Linear in t, so the bound stays a convex cone; the last term makes room
  // for the swept centreline (uBend * s^2, at most its tip offset)
  float boundR(float t) { return uW * uR0 * mix(3.8, 15.0, t) + t * length(uBend) * uLen * uLen; }

  void main() {
    // Unit cone (z 0 .. -1, radius 1) shaped here: a short collar forward of
    // the nozzle plane for the throat glow, then out to the plume tip
    float t = clamp(-position.z, 0.0, 1.0);
    float fwd = uR0 * 2.6;
    vCam = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
    // Room for the pixel floor of the core and sheath at fleet range
    float z = mix(fwd, -uLen, t);
    float pad = 3.2 * uPixelWorld * length(vCam - vec3(0.0, 0.0, z));
    vec3 lp = vec3(position.xy * (boundR(t) + pad), z);
    vPos = lp;
    // Camera inside the cone: the front faces are behind it, so the back
    // faces carry the plume instead (see the fragment shader)
    float tc = clamp((fwd - vCam.z) / (fwd + uLen), 0.0, 1.0);
    float margin = 0.6 + 0.3 * uR0;
    vInside = (vCam.z < fwd + margin && vCam.z > -uLen - margin
      && length(vCam.xy) < boundR(tc) * 1.02 + margin) ? 1.0 : 0.0;
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(lp, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform sampler2D uNoise;
  uniform float uTime;
  uniform int uSeg;
  uniform float uLen;
  uniform float uR0;
  uniform float uW;
  uniform float uThrust;
  uniform float uDiamonds;
  uniform float uFlare;
  uniform float uSeed;
  uniform vec2 uBend;    // centreline sweep: lateral offset = uBend * s^2
  uniform float uPixelWorld;
  uniform vec3 uI;       // core, sheath, halo side-on centreline radiance
  uniform vec3 uCore;
  uniform vec3 uSheath;
  uniform vec3 uEdge;
  varying vec3 vPos;
  varying vec3 vCam;
  varying float vInside;
  #include <common>
  #include <logdepthbuf_pars_fragment>

  // Abramowitz-Stegun 7.1.27, |error| < 5e-4, no exp
  float erfa(float x) {
    float a = abs(x);
    float d = 1.0 + a * (0.278393 + a * (0.230389 + a * (0.000972 + a * 0.078108)));
    d *= d; d *= d;
    return sign(x) * (1.0 - 1.0 / d);
  }

  // Normalised column through a Gaussian tube of width sig over the ray
  // span [ga, gb] (measured from the closest approach): 1 on the centreline
  // of a side-on view, path/(sig*sqrt(pi)) looking straight down the axis
  float column(float sig, float d2, float sinT, float ga, float gb) {
    float k = sinT / sig;
    // Exact column, then eased toward the side-on value: an oblique view of
    // a long plume is deeper, but must not whiten the whole body
    return exp(-d2 / (sig * sig)) * 0.5 * (erfa(gb * k) - erfa(ga * k)) / sinT * sqrt(max(sinT, 0.2));
  }

  void main() {
    // Convex cone: exactly one face per pixel - front faces from outside,
    // back faces once the camera is inside the plume
    if (gl_FrontFacing == (vInside > 0.5)) discard;
    #include <logdepthbuf_fragment>
    #if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
      // Inside the plume the gas starts at the lens: nothing can occlude it
      if (vInside > 0.5) gl_FragDepthEXT = 0.0;
    #endif

    vec3 ro = vCam;
    vec3 rd = normalize(vPos - vCam);
    vec3 col = vec3(0.0);

    // Throat: a white-hot ball sitting in the nozzle exit. Its skirt falls
    // on the bells around it, which is the heat glow on the engine block.
    float sf = uR0 * (0.55 + 0.25 * uThrust);
    vec3 fc = ro - vec3(0.0, 0.0, -0.25 * uR0);
    float fd2 = dot(fc, fc) - pow(dot(fc, rd), 2.0);
    col += mix(uCore, uSheath, 0.3) * uFlare * (exp(-fd2 / (sf * sf)) + 0.16 * exp(-fd2 / (sf * sf * 9.0)));

    // Chord of the ray through the slab 0 <= s <= uLen (s = distance aft)
    float sA = -ro.z, sD = -rd.z;
    float t0 = 0.0, t1 = -1.0;
    if (abs(sD) > 1e-5) {
      float ta = -sA / sD, tb = (uLen - sA) / sD;
      t0 = max(min(ta, tb), 0.0);
      t1 = max(ta, tb);
    } else if (sA > 0.0 && sA < uLen) {
      t1 = 40.0 * uLen;
    }
    if (t1 > t0) {
      float dd = max(dot(rd.xy, rd.xy), 1e-6);
      float sinT = sqrt(dd);
      float tc = -dot(ro.xy, rd.xy) / dd;          // closest approach to the axis
      vec2 qc = ro.xy + rd.xy * tc;
      float lam = uR0 * (4.0 + 3.0 * uThrust);     // shock cell length
      // Down the axis (a chase camera sitting in the exhaust) the wide layers
      // are the whole plume deep: keep them a haze, not a blue screen
      float wide = mix(0.22, 1.0, smoothstep(0.05, 0.6, sinT));
      float dt = (t1 - t0) / float(uSeg);
      for (int i = 0; i < 4; i++) {
        if (i >= uSeg) break;
        float a = t0 + dt * float(i), b = a + dt;
        float te = clamp(tc, a, b);                // nearest the axis in this sub-chord
        float s = sA + sD * te;
        float t = clamp(s / uLen, 0.0, 1.0);
        // Swept centreline: the part of the offset across the ray moves the
        // closest approach sideways, the part along it slides it down the ray
        vec2 bend = uBend * s * s;
        float along = dot(bend, rd.xy) / dd;
        vec2 qs = qc - (bend - rd.xy * along);
        float d2 = dot(qs, qs);
        float tcs = tc + along;
        vec2 q = ro.xy + rd.xy * te - bend;

        // Turbulence: filaments streaming aft, laminar at the throat
        // (long in the flow direction, narrow across it), growing downstream
        float v = s / (uR0 * 44.0);
        float n1 = textureLod(uNoise, vec2(q.x / (uR0 * 7.0) + uSeed, v - uTime * 0.4), 0.0).r;
        float n2 = textureLod(uNoise, vec2(q.y / (uR0 * 5.0) + uSeed * 1.7 + 0.37, v * 2.1 - uTime * 0.7), 0.0).g;
        float fil = clamp(n1 * n2 * 4.0, 0.0, 1.8);
        float rough = smoothstep(0.0, 0.22, t);
        float coreTurb = mix(1.0, 0.7 + 0.3 * fil, rough);
        float sheathTurb = mix(1.0, 0.22 + 0.78 * fil, rough);
        float haloTurb = mix(1.0, 0.4 + 1.2 * n1, rough);

        // Shock diamonds: the core necks down and flares at each cell
        float ph = 6.28318 * s / lam - 2.0;
        // (each cell weaker and blurrier than the last: three or four read,
        // then turbulence has eaten the pattern)
        float cell = exp(-s / (lam * 1.7)) * uDiamonds;
        float pinch = 1.0 - 0.4 * cos(ph) * cell;
        float knot = 1.0 + 3.6 * pow(0.5 + 0.5 * cos(ph), mix(2.0, 5.0, cell)) * cell;
        // Pressure pulses running down the sheath (slow: no strobing)
        float pulse = 1.0 + 0.14 * sin(6.28318 * (s / (uR0 * 24.0) - uTime * 0.8 + uSeed));

        float rise = smoothstep(0.0, uR0 * 0.7, s);
        float tip = 1.0 - smoothstep(0.35, 1.0, t);
        float env = rise * tip;
        // Pixel floor: at fleet range the core is thinner than a pixel and
        // would shimmer or vanish; keep it (and the sheath) a readable
        // streak that still tells who is burning and which way
        float px = uPixelWorld * max(te, 0.01);
        float sc = max(uW * uR0 * (0.34 + 1.0 * t) * pinch, 0.8 * px);
        float ss = max(uW * uR0 * (0.85 + 5.2 * pow(t, 0.6)), 1.5 * px);
        float sh = uW * uR0 * (1.7 + 8.6 * pow(t, 0.8));
        float ga = a - tcs, gb = b - tcs;

        // White-hot only at the throat and in the knots; the core cools to
        // the sheath's blue within the first tenth of the plume
        float heat = clamp(exp(-t * 9.0) * (0.4 + 0.3 * knot), 0.0, 1.0);
        vec3 cc = mix(uSheath * 1.25, uCore, heat);
        vec3 cs = mix(uSheath, uEdge, smoothstep(0.0, 0.55, t));
        col += cc * (uI.x * exp(-t * 5.5) * knot * coreTurb * env * column(sc, d2, sinT, ga, gb));
        col += cs * (uI.y * wide * exp(-t * 3.4) * pulse * sheathTurb * env * column(ss, d2, sinT, ga, gb));
        col += uEdge * (uI.z * wide * exp(-t * 2.4) * haloTurb * env * column(sh, d2, sinT, ga, gb));
      }
    }

    // Looking down the axis the column is the whole plume deep: roll the
    // radiance off smoothly so an end-on torch is a hot disc, not a wash
    float lum = max(col.r, max(col.g, col.b));
    const float CAP = 5.0;
    col *= CAP * (1.0 - exp(-lum / CAP)) / max(lum, 1e-4);
    gl_FragColor = vec4(col, 1.0);
  }
`;

let _coneGeom = null;
/** Unit bounding cone, z 0 .. -1, capped; the vertex shader shapes it. */
function coneGeometry() {
  if (_coneGeom) return _coneGeom;
  const g = new THREE.CylinderGeometry(1, 1, 1, 20, 6, false);
  g.rotateX(Math.PI / 2);
  g.translate(0, 0, -0.5);
  _coneGeom = g;
  return g;
}

// Shared by every torch: turbulence clock, noise, sub-chord count (tier)
const shared = {
  uNoise: { value: null },
  uTime: { value: 0 },
  uSeg: { value: 4 }
};
const SEGMENTS = { low: 2, high: 4 };

const coughHash = (k) => { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

/**
 * Failing-drive cough 0..1 at battle time t: seeded dropouts on a 0.22 s
 * grid, eased between grid points (no strobing), so a sputtering torch is a
 * pure function of the timeline. 1 = the plume all but cuts out.
 */
export function torchCough(t) {
  const u = t / 0.22, k = Math.floor(u), f = u - k;
  const level = (i) => { const h = coughHash(i); return h > 0.55 ? (h - 0.55) / 0.45 : 0.15 * h; };
  const s = f * f * (3 - 2 * f);
  return level(k) + (level(k + 1) - level(k)) * s;
}

/**
 * Side-on centreline radiance of the three layers and the throat ball at
 * full throttle for ONE nozzle (scene HDR units; bloom threshold ~1.35).
 * Only the throat, the diamonds and the first stretch of core cross the
 * bloom threshold - the body of the plume stays a visible blue gradient.
 */
export const TORCH_LEVELS = { core: 1.5, sheath: 0.68, halo: 0.11, flare: 1.2 };

/**
 * Per-nozzle level scale for a drive of `cluster` bells: cores overlap by
 * about sqrt(n) along a typical view ray, the wide sheath and halo of every
 * bell overlap everywhere, so the cluster as a whole keeps the look of one
 * plume instead of summing to white.
 */
export function clusterLevels(cluster = 1) {
  const n = Math.max(1, cluster);
  return { core: 1 / Math.sqrt(n), wide: 1 / Math.pow(n, 0.85) };
}

/**
 * Plume shape at a (smoothed) throttle 0..1: an idle torch is a short dim
 * violet flicker, full burn a long white-cored spike.
 */
export function plumeShape(thrust) {
  const th = Math.pow(Math.max(0, Math.min(1, thrust)), 0.85);
  return {
    length: 0.14 + 0.86 * th,
    width: 0.72 + 0.28 * th,
    core: 0.22 + 0.78 * th,
    wide: 0.28 + 0.72 * th
  };
}

/** Extra core/sheath level of a dying drive at full sputter (x 1 + this). */
export const DYING_GASP = 1.6;

/** Seconds the exhaust at the plume tip has been in flight, for the sweep. */
export const SWEEP_AGE_S = 2.2;
/** Largest tip offset of a swept plume, as a fraction of its length. */
export const SWEEP_MAX = 0.32;
const _sweep = { x: 0, y: 0 };

/**
 * Tip offset of the plume (fraction of its length, hull frame x/y) for a
 * turn rate (wx, wy) rad/s in the hull frame (+Z = bow). Exhaust already
 * emitted keeps flying straight while the stern swings away, so the far
 * plume lags toward w x forward: a turning ship drags a curved tail.
 * Pure: the recorded turn rate in, the offset out.
 */
export function plumeSweep(wx, wy, out = { x: 0, y: 0 }) {
  let x = wy * SWEEP_AGE_S, y = -wx * SWEEP_AGE_S;
  const m = Math.hypot(x, y);
  if (m > SWEEP_MAX) { x *= SWEEP_MAX / m; y *= SWEEP_MAX / m; }
  out.x = x; out.y = y;
  return out;
}

/**
 * @param {Object} opts
 *   radius    - nozzle radius (scene units)
 *   length    - nominal plume length at 100% thrust
 *   light     - optional THREE.PointLight (budgeted by the caller)
 *   tint      - 'fusion' (white/blue) | 'chemical' (white/orange, vents)
 *   intensity - overall level scale
 *   cluster   - number of bells in the drive this nozzle belongs to
 */
export function createTorch({ radius, length, light = null, tint = 'fusion', intensity = 1, cluster = 1 }) {
  const group = new THREE.Group();
  const fusion = tint === 'fusion';
  const I = intensity;
  const k = clusterLevels(cluster);
  // The bright stretch is the first third of this; the rest is the halo
  // thinning out. Shorter than the pass-4 neon tubes so the falloff and the
  // tip land inside a hull-framing shot.
  const reach = length * (fusion ? 0.75 : 1.0);
  const levels = {
    core: TORCH_LEVELS.core * I * k.core,
    sheath: TORCH_LEVELS.sheath * I * k.wide,
    halo: TORCH_LEVELS.halo * I * k.wide,
    flare: TORCH_LEVELS.flare * Math.min(1, I) * k.core
  };

  shared.uNoise.value = getNoiseTexture();
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uNoise: shared.uNoise,
      uTime: shared.uTime,
      uSeg: shared.uSeg,
      uLen: { value: reach },
      uR0: { value: radius },
      uW: { value: 1 },
      uThrust: { value: 0 },
      uDiamonds: { value: fusion ? 1.0 : 0.35 },
      uFlare: { value: 0 },
      uSeed: { value: Math.random() },
      uBend: { value: new THREE.Vector2() },
      uPixelWorld: Ribbon.pixelWorld,
      uI: { value: new THREE.Vector3() },
      uCore: { value: new THREE.Color(...(fusion ? [1.0, 0.96, 1.0] : [1.0, 0.94, 0.82])) },
      uSheath: { value: new THREE.Color(...(fusion ? [0.15, 0.41, 1.0] : [1.0, 0.52, 0.18])) },
      uEdge: { value: new THREE.Color(...(fusion ? [0.36, 0.18, 0.95] : [0.8, 0.24, 0.1])) }
    },
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendEquation: THREE.AddEquation
  });
  const plume = new THREE.Mesh(coneGeometry(), material);
  plume.frustumCulled = false;
  plume.renderOrder = 20;
  // Sub-chord count follows the graphics tier (read-only use of Quality.js)
  plume.onBeforeRender = (renderer) => {
    const seg = SEGMENTS[getGraphicsSettings(renderer).level] || SEGMENTS.high;
    if (shared.uSeg.value !== seg) shared.uSeg.value = seg;
  };
  group.add(plume);

  const _wp = new THREE.Vector3();
  const _wd = new THREE.Vector3();
  const u = material.uniforms;
  const phase = Math.random() * Math.PI * 2;
  const state = { thrust: 0 };

  return {
    group,
    light,
    levels,
    /**
     * @param {number} thrust       0..1 (already smoothed by the caller)
     * @param {number} time         wall-clock seconds for turbulence
     * @param {number} sputter      0..1 extra flicker (dying drives)
     * @param {number} sputterTime  battle seconds driving the sputter, so a
     *                              paused or scrubbed frame always shows the
     *                              same cough (falls back to `time`)
     * @param {THREE.Vector3} turn  optional recorded turn rate in the hull
     *                              frame (rad/s): the plume sweeps behind a
     *                              turning stern
     */
    update(thrust, time, sputter = 0, sputterTime = undefined, turn = null) {
      state.thrust = thrust;
      if (thrust < 0.015) {
        group.visible = false;
        if (light) light.intensity = 0;
        return;
      }
      group.visible = true;
      // Slow breathing only - anything above ~1.5 Hz aliases into strobing
      let breathe = 0.965 + 0.025 * Math.sin(time * 4.3 + phase) + 0.012 * Math.sin(time * 7.1 + phase * 1.7);
      // A failing drive burns rough and over-rich between its dropouts, so a
      // hulk limping at a few percent throttle still shows its torch coughing
      let gasp = 1;
      if (sputter > 0) {
        breathe *= 1 - sputter * 0.75 * torchCough(sputterTime ?? time);
        gasp = 1 + DYING_GASP * sputter;
      }
      const shape = plumeShape(thrust);
      shared.uTime.value = time;
      u.uThrust.value = thrust;
      const len = reach * shape.length * breathe;
      u.uLen.value = len;
      if (turn) {
        plumeSweep(turn.x, turn.y, _sweep);
        u.uBend.value.set(_sweep.x / len, _sweep.y / len);
      } else {
        u.uBend.value.set(0, 0);
      }
      u.uW.value = shape.width;
      u.uI.value.set(levels.core * shape.core * breathe * gasp, levels.sheath * shape.wide * breathe * gasp, levels.halo * shape.wide * breathe);
      u.uFlare.value = levels.flare * Math.min(1, thrust * 1.3 * gasp * gasp) * breathe;
      if (light) {
        group.getWorldPosition(_wp);
        _wd.set(0, 0, -1).transformDirection(group.matrixWorld);
        light.position.copy(_wp).addScaledVector(_wd, radius * 3);
        light.intensity = (0.15 + 0.85 * shape.wide) * 8 * I * breathe;
      }
    },
    dispose() {
      material.dispose();   // the cone geometry and the noise are shared
    }
  };
}
