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
 * The exhaust leaves at 10,256 km/s, so to the eye the plume is a rigid,
 * steady lance down the thrust axis: it swings with the hull (no sweep, no
 * lag), its length and brightness follow the throttle the engine produced
 * this instant (ignition and cutoff are instant along its whole length),
 * and nothing in it drifts. What reads as speed is texture: fine streaks
 * long along the axis scrolling several plume lengths per second on the
 * wall clock (FLOW_RATE: a feature moves about its own length per frame, so
 * none can be followed - a shimmer, not puffs), over standing shock
 * diamonds that never move. Beyond the flame the plume continues as the
 * spear (render/DriveJet.js); render/PlumeProfile.js states how the two
 * add up to one field. Chemical vents (wrecks) keep slow, puffy gas.
 *
 * Wall-clock time only scrolls the streaks; length, width, colour, the
 * ignition flash and the dying-drive cough are functions of the throttle
 * and of battle time.
 */

/** Shock cells (aft of the nozzle) after which the standing knots are gone: past it the plume only fades. */
export const TORCH_CORE_CELLS = 3.5;

const vertexShader = /* glsl */`
  uniform float uLen;
  uniform float uR0;
  uniform float uW;
  uniform float uPixelWorld;
  varying vec3 vPos;
  varying vec3 vCam;
  varying float vInside;
  #include <common>
  #include <logdepthbuf_pars_vertex>

  // Linear in t, so the bound stays a convex cone
  float boundR(float t) { return uW * uR0 * mix(3.8, 11.0, t); }

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
  uniform vec2 uFlow;    // streak scroll phases (wall clock, wrapped on the CPU)
  uniform vec4 uStreak;  // streak scale across (x, y) and along (A, B), in nozzle radii
  uniform vec3 uRough;   // turbulence floor of core, sheath, halo (1 = smooth)
  uniform int uSeg;
  uniform float uLen;
  uniform float uR0;
  uniform float uW;
  uniform float uThrust;
  uniform float uDiamonds;
  uniform float uFlare;
  uniform float uSeed;
  uniform float uPhase;  // per-bell shock cell phase 0..1
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
    float sf = uR0 * (0.4 + 0.16 * uThrust);
    vec3 fc = ro - vec3(0.0, 0.0, -0.25 * uR0);
    float fcd = dot(fc, rd);                     // a product: pow(<0, 2.0) is NaN on some GPUs
    float fd2 = dot(fc, fc) - fcd * fcd;
    col += mix(uCore, uSheath, 0.3) * uFlare * (exp(-fd2 / (sf * sf)) + 0.1 * exp(-fd2 / (sf * sf * 7.0)));

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
      // shock cell length, a little different on every bell (uPhase) so a
      // cluster's cells never line up in rows across the plumes
      float lam = uR0 * (4.0 + 3.0 * uThrust) * (0.86 + 0.28 * uPhase);
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
        float d2 = dot(qc, qc);
        float tcs = tc;
        vec2 q = ro.xy + rd.xy * te;

        // Streaming: fine streaks, long in the flow direction and narrow
        // across it, scrolling aft fast enough that none can be followed
        // (two octaves at incommensurate rates); laminar at the throat
        float n1 = textureLod(uNoise, vec2(q.x / (uR0 * uStreak.x) + uSeed, s / (uR0 * uStreak.z) - uFlow.x), 0.0).r;
        float n2 = textureLod(uNoise, vec2(q.y / (uR0 * uStreak.y) + uSeed * 1.7 + 0.37, s / (uR0 * uStreak.w) - uFlow.y), 0.0).g;
        float fil = clamp(n1 * n2 * 4.0, 0.0, 1.8);
        float rough = smoothstep(0.0, 0.22, t);
        float coreTurb = mix(1.0, uRough.x + (1.0 - uRough.x) * fil, rough);
        float sheathTurb = mix(1.0, uRough.y + (1.0 - uRough.y) * fil, rough);
        float haloTurb = mix(1.0, uRough.z + 2.0 * (1.0 - uRough.z) * n1, rough);

        // Shock diamonds: the core necks down and flares at each cell
        float ph = 6.28318 * s / lam - 2.0 - 6.28318 * uPhase;
        // (each cell weaker and blurrier than the last: three read, and the
        // pattern is gone by TORCH_CORE_CELLS - past it the plume only fades)
        float cell = exp(-s / (lam * 1.25)) * uDiamonds * (1.0 - smoothstep(2.0 * lam, ${TORCH_CORE_CELLS.toFixed(2)} * lam, s));
        float pinch = 1.0 - 0.4 * cos(ph) * cell;
        float knot = 1.0 + 2.4 * pow(0.5 + 0.5 * cos(ph), mix(1.3, 3.0, cell)) * cell;

        float rise = smoothstep(0.0, uR0 * 0.7, s);
        float tip = 1.0 - smoothstep(0.35, 1.0, t);
        float env = rise * tip;
        // Pixel floor: at fleet range the core is thinner than a pixel and
        // would shimmer or vanish; keep it (and the sheath) a readable
        // streak that still tells who is burning and which way
        float px = uPixelWorld * max(te, 0.01);
        float sc = max(uW * uR0 * (0.34 + 0.7 * t) * pinch, 0.8 * px);
        float ss = max(uW * uR0 * (0.85 + 2.9 * pow(t, 0.7)), 1.5 * px);
        float sh = uW * uR0 * (1.7 + 6.2 * pow(t, 0.8));
        float ga = a - tcs, gb = b - tcs;

        // White-hot only at the throat and in the knots; the core cools to
        // the sheath's blue within the first tenth of the plume
        float heat = clamp(exp(-t * 9.0) * (0.4 + 0.3 * knot), 0.0, 1.0);
        vec3 cc = mix(uSheath * 1.25, uCore, heat);
        vec3 cs = mix(uSheath, uEdge, smoothstep(0.0, 0.55, t));
        col += cc * (uI.x * exp(-t * 5.5) * knot * coreTurb * env * column(sc, d2, sinT, ga, gb));
        col += cs * (uI.y * wide * exp(-t * 3.4) * sheathTurb * env * column(ss, d2, sinT, ga, gb));
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

// Shared by every torch: noise, sub-chord count (tier)
const shared = {
  uNoise: { value: null },
  uSeg: { value: 4 }
};
const SEGMENTS = { low: 2, high: 4 };

const coughHash = (k) => { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

/** Grid (battle s) of a failing drive's dropouts, and the share of each step spent switching. */
export const COUGH_STEP_S = 0.09;
export const COUGH_EDGE = 0.3;

/**
 * Failing-drive cough 0..1 at battle time t: seeded dropouts on a fast
 * grid (COUGH_STEP_S), each switched in the first COUGH_EDGE of its step -
 * a torch that stutters, since a 10,000 km/s jet cannot fade slowly - so a
 * sputtering torch is a pure function of the timeline. 1 = the plume all
 * but cuts out.
 */
export function torchCough(t) {
  const u = t / COUGH_STEP_S, k = Math.floor(u), f = Math.min(1, (u - k) / COUGH_EDGE);
  const level = (i) => { const h = coughHash(i); return h > 0.55 ? (h - 0.55) / 0.45 : 0.15 * h; };
  const s = f * f * (3 - 2 * f);
  return level(k - 1) + (level(k) - level(k - 1)) * s;
}

const smooth01 = (x) => { const c = Math.max(0, Math.min(1, x)); return c * c * (3 - 2 * c); };
/** The shader's radiance ceiling per bell (max channel, scene HDR). */
export const TORCH_CAP = 5.0;

/**
 * Side-on centreline radiance (max channel, scene HDR, after the ceiling)
 * of ONE bell `s` kit units aft of its nozzle plane, without the streaks:
 * the JS twin of the fragment shader's axial profile (throat ball, core
 * with its standing shock knots, sheath, halo). `p` is a torch's current
 * profile (torch.profile()). Used by tests/plume.test.js to check that
 * torch + spear is one continuous, non-increasing field.
 */
export function torchCentreline(s, p) {
  const { len, r0, I, flare, thrust, diamonds, phase } = p;
  if (!(len > 0)) return 0;
  const t = Math.max(0, Math.min(1, s / len));
  const lam = r0 * (4 + 3 * thrust) * (0.86 + 0.28 * phase);
  const ph = 2 * Math.PI * s / lam - 2 - 2 * Math.PI * phase;
  const cell = Math.exp(-Math.max(0, s) / (lam * 1.25)) * diamonds * (1 - smooth01((s - 2 * lam) / ((TORCH_CORE_CELLS - 2) * lam)));
  const knot = 1 + 2.4 * Math.pow(0.5 + 0.5 * Math.cos(ph), 1.3 + 1.7 * cell) * cell;
  const env = (s > 0 ? smooth01(s / (r0 * 0.7)) : 0) * (1 - smooth01((t - 0.35) / 0.65));
  const sf = r0 * (0.4 + 0.16 * thrust), fd = s + 0.25 * r0, fd2 = fd * fd;
  let lum = flare * (Math.exp(-fd2 / (sf * sf)) + 0.1 * Math.exp(-fd2 / (sf * sf * 7)));
  if (s >= 0 && s <= len) {
    lum += I[0] * Math.exp(-t * 5.5) * knot * env + I[1] * Math.exp(-t * 3.4) * env + I[2] * Math.exp(-t * 2.4) * env;
  }
  return TORCH_CAP * (1 - Math.exp(-lum / TORCH_CAP));
}


/** End of the shock-diamond zone (kit units aft of the nozzle) of a torch profile: past it the plume only fades. */
export function torchCoreEnd(p) {
  return TORCH_CORE_CELLS * p.r0 * (4 + 3 * p.thrust) * (0.86 + 0.28 * p.phase);
}

/**
 * Side-on centreline radiance of the three layers and the throat ball at
 * full throttle for ONE nozzle (scene HDR units; bloom threshold ~1.35).
 * Only the throat, the diamonds and the first stretch of core cross the
 * bloom threshold - the body of the plume stays a visible blue gradient.
 */
export const TORCH_LEVELS = { core: 2.1, sheath: 0.68, halo: 0.11, flare: 1.7 };

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

/** Ignition flash: a throttle step this large (0..1) saturates it; it dies away in IGNITION_S battle seconds. */
export const IGNITION_STEP = 0.5;
export const IGNITION_S = 0.22;
/** Ignition: throat flare, core and shock knots flare up by 1 + these at the step. */
export const IGNITION_FLARE = 1.8;
export const IGNITION_CORE = 0.55;
export const IGNITION_KNOTS = 1.2;

/**
 * Light-off flash 0..1 `age` battle seconds after the throttle stepped from
 * `from` to `to` (Interpolator: the recorded throttle is piecewise
 * constant, the engine's own 1 s tick). The whole plume is at its new
 * length on the step itself; this only flares the throat and the shock
 * knots for a blink. A cutoff is instant and dark: no transient. Pure.
 */
export function plumeIgnition(from, to, age) {
  if (!(age >= 0) || !(to > from)) return 0;
  return Math.min(1, (to - from) / IGNITION_STEP) * Math.exp(-age / IGNITION_S);
}

/** Extra core/sheath level of a dying drive at full sputter (x 1 + this). */
export const DYING_GASP = 1.6;

/**
 * One bell of a damaged drive (recorded `mount_damaged`, kind engine):
 * some bells are dark, the rest burn harder and rough to make up the thrust
 * the ship still has. Which bells die is seeded per ship, and a bell that is
 * dark at one damage level stays dark at every higher one. Pure.
 *
 * @param {number} health  engine health 0-100 (100 = intact)
 * @param {number} i       bell index, @param {number} n bells in the drive
 * @param {number} seed    per-ship seed
 * @returns out {level, sputter}: thrust multiplier (0 = dark) and cough 0..1
 */
export function damagedNozzle(health, i, n, seed, out = { level: 1, sputter: 0 }) {
  const dmg = Math.max(0, Math.min(1, 1 - health / 100));
  out.level = 1; out.sputter = 0;
  if (dmg <= 0 || n <= 0) return out;
  const dead = health <= 0 ? n : Math.min(n - 1, Math.round(dmg * n * 0.85));
  // rank of this bell in the seeded death order
  const h = (k) => coughHash(seed * 0.001 + k * 7.31);
  let rank = 0;
  for (let k = 0; k < n; k++) if (h(k) < h(i) || (h(k) === h(i) && k < i)) rank++;
  if (rank < dead) { out.level = 0; return out; }
  out.level = Math.min(1.35, n / (n - dead));
  out.sputter = Math.min(0.85, 0.25 + 0.7 * dmg + 0.2 * h(i + 31));
  return out;
}

/** Streak scroll (texture tiles per wall-clock second) and scale (nozzle radii) by tint. */
export const FLOW_RATE = { fusion: [4.3, 7.1], chemical: [0.4, 0.7] };
const STREAK = { fusion: [2.2, 1.6, 26, 17], chemical: [7, 5, 44, 44 / 2.1] };
/** Turbulence floor of core, sheath, halo (1 = smooth): a fusion jet shimmers, vented gas billows. */
const ROUGH = { fusion: [0.86, 0.5, 0.72], chemical: [0.7, 0.22, 0.4] };

/** Streak phases at wall-clock time `time` (wrapped, so float precision never degrades). */
export function flowPhase(time, rate, out = [0, 0]) {
  const a = (time * rate[0]) % 1, b = (time * rate[1]) % 1;
  out[0] = a < 0 ? a + 1 : a; out[1] = b < 0 ? b + 1 : b;
  return out;
}
const _flow = [0, 0];

/**
 * @param {Object} opts
 *   radius    - nozzle radius (scene units)
 *   length    - nominal plume length at 100% thrust
 *   light     - optional THREE.PointLight (budgeted by the caller)
 *   tint      - 'fusion' (white/blue) | 'chemical' (white/orange, vents)
 *   intensity - overall level scale
 *   cluster   - number of bells in the drive this nozzle belongs to
 */
export function createTorch({ radius, length, light = null, tint = 'fusion', intensity = 1, cluster = 1, phase: cellPhase = null }) {
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
    // per bell, and inside the liner: a big cluster must not weld its
    // throats into one blob that hides the bells
    flare: TORCH_LEVELS.flare * Math.min(1, I) * Math.min(k.core, 1.6 * k.wide)
  };

  shared.uNoise.value = getNoiseTexture();
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uNoise: shared.uNoise,
      uFlow: { value: new THREE.Vector2() },
      uStreak: { value: new THREE.Vector4(...STREAK[fusion ? 'fusion' : 'chemical']) },
      uRough: { value: new THREE.Vector3(...ROUGH[fusion ? 'fusion' : 'chemical']) },
      uSeg: shared.uSeg,
      uLen: { value: reach },
      uR0: { value: radius },
      uW: { value: 1 },
      uThrust: { value: 0 },
      uDiamonds: { value: fusion ? 1.0 : 0.35 },
      uFlare: { value: 0 },
      uSeed: { value: Math.random() },
      uPhase: { value: cellPhase ?? Math.random() },
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
  const diamonds = material.uniforms.uDiamonds.value;
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
  const lightReach = light ? light.distance : 0;   // kit units, as borrowed
  const u = material.uniforms;
  const flowRate = FLOW_RATE[fusion ? 'fusion' : 'chemical'];
  const state = { thrust: 0 };

  return {
    group,
    light,
    levels,
    /** This frame's axial profile, in the torch's kit units (torchCentreline). */
    profile() {
      if (!group.visible) return { len: 0, r0: radius, I: [0, 0, 0], flare: 0, thrust: 0, diamonds: 0, phase: u.uPhase.value };
      return { len: u.uLen.value, r0: radius, I: [u.uI.value.x, u.uI.value.y, u.uI.value.z], flare: u.uFlare.value,
        thrust: u.uThrust.value, diamonds: u.uDiamonds.value, phase: u.uPhase.value };
    },
    /**
     * @param {number} thrust       0..1: the throttle the engine produces at
     *                              this battle time (a step, not eased)
     * @param {number} time         wall-clock seconds: streak scroll only
     * @param {number} sputter      0..1 extra flicker (dying drives)
     * @param {number} sputterTime  battle seconds driving the sputter, so a
     *                              paused or scrubbed frame always shows the
     *                              same cough (falls back to `time`)
     * @param {number} surge        0..1 ignition flash (plumeIgnition)
     */
    update(thrust, time, sputter = 0, sputterTime = undefined, surge = 0) {
      state.thrust = thrust;
      if (thrust < 0.015) {
        group.visible = false;
        if (light) light.intensity = 0;
        return;
      }
      group.visible = true;
      // Steady: a healthy torch has no breathing. A failing drive burns
      // rough and over-rich between its dropouts, so a hulk limping at a
      // few percent throttle still shows its torch stuttering
      let level = 1, gasp = 1;
      if (sputter > 0) {
        level = 1 - sputter * 0.75 * torchCough(sputterTime ?? time);
        gasp = 1 + DYING_GASP * sputter;
      }
      const sg = Math.max(0, Math.min(1, surge || 0));
      const shape = plumeShape(thrust);
      flowPhase(time, flowRate, _flow);
      u.uFlow.value.set(_flow[0], _flow[1]);
      u.uThrust.value = thrust;
      u.uLen.value = reach * shape.length * (0.85 + 0.15 * level);
      u.uW.value = shape.width;
      u.uDiamonds.value = diamonds * (1 + IGNITION_KNOTS * sg);
      const coreK = 1 + IGNITION_CORE * sg;
      u.uI.value.set(levels.core * shape.core * level * gasp * coreK, levels.sheath * shape.wide * level * gasp,
        levels.halo * shape.wide * level);
      u.uFlare.value = levels.flare * Math.min(1, thrust * 1.3 * gasp * gasp) * level * (1 + IGNITION_FLARE * sg);
      if (light) {
        // A pool light lives in world space: carry the hull's draw scale
        // (render/Scale.js) into its offset, reach and - inverse-square -
        // its intensity, so the stern is lit the same at any hull size
        group.updateWorldMatrix(true, false);
        const e = group.matrixWorld.elements;
        const k = Math.hypot(e[8], e[9], e[10]);
        group.getWorldPosition(_wp);
        _wd.set(0, 0, -1).transformDirection(group.matrixWorld);
        light.position.copy(_wp).addScaledVector(_wd, radius * 3 * k);
        if (lightReach > 0) light.distance = lightReach * k;
        light.intensity = (0.15 + 0.85 * shape.wide) * 8 * I * level * (1 + 0.5 * sg) * k * k;
      }
    },
    dispose() {
      material.dispose();   // the cone geometry and the noise are shared
    }
  };
}
