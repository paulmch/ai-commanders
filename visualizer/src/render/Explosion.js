import * as THREE from 'three';
import { getNoiseTexture, getGlowTexture, mulberry32 } from './Textures.js';
import { SparkBurst, StreakBurst, PuffCloud } from './Particles.js';
import { blastCloseDim } from './CloseRange.js';
import { buildDebrisChunks } from './Hulls.js';
import { createTorch } from './Torch.js';
import { breakHull } from './DamageFX.js';
import { sectionCrackGlow } from './HullWear.js';

/**
 * Blasts and ship destruction.
 *
 * Everything is driven by battle time (age = now - spawnTime) so scrubbing
 * the timeline replays the same detonation, including its surface turbulence.
 */

const ADDITIVE = {
  transparent: true,
  depthWrite: false,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor,
  blendEquation: THREE.AddEquation
};

const TRIPLANAR_GLSL = /* glsl */`
  vec4 tri(sampler2D t, vec3 p, vec3 n) {
    vec3 w = abs(n);
    w = w * w;
    w /= (w.x + w.y + w.z + 1e-5);
    return texture2D(t, p.yz) * w.x + texture2D(t, p.xz) * w.y + texture2D(t, p.xy) * w.z;
  }
`;

// ---------------------------------------------------------------------------
// Fireball: noise-displaced sphere with a temperature ramp
// ---------------------------------------------------------------------------
function fireballMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uTime: { value: 0 },
      uHeat: { value: 1 },
      uTurb: { value: 0.3 },
      uIntensity: { value: 1 },
      uAlpha: { value: 1 },
      uSeed: { value: 0 }
    },
    vertexShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uTime;
      uniform float uTurb;
      uniform float uSeed;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      ${TRIPLANAR_GLSL}
      #include <common>
  #include <logdepthbuf_pars_vertex>
      void main() {
        vec3 n = normalize(position);
        vec4 d1 = tri(uNoise, n * 0.45 + uSeed + uTime * 0.05, n);
        vec4 d2 = tri(uNoise, n * 1.2 - uSeed * 1.7 - uTime * 0.09, n);
        float disp = (d1.r - 0.5) * 0.6 + (d2.g - 0.5) * 0.08;
        vec3 p = position * (1.0 + disp * uTurb);
        vLocal = n;
        vNormal = normalize(normalMatrix * normal);
        vec4 wp = modelMatrix * vec4(p, 1.0);
        vView = -(viewMatrix * wp).xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uTime;
      uniform float uHeat;
      uniform float uIntensity;
      uniform float uAlpha;
      uniform float uSeed;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      ${TRIPLANAR_GLSL}
      #include <logdepthbuf_pars_fragment>
      vec3 ramp(float h) {
        vec3 c0 = vec3(0.30, 0.02, 0.0);
        vec3 c1 = vec3(1.0, 0.22, 0.02);
        vec3 c2 = vec3(1.0, 0.72, 0.18);
        vec3 c3 = vec3(1.25, 1.2, 1.1);
        vec3 c4 = vec3(1.2, 1.3, 1.5);
        if (h < 0.3) return mix(c0, c1, h / 0.3);
        if (h < 0.6) return mix(c1, c2, (h - 0.3) / 0.3);
        if (h < 0.85) return mix(c2, c3, (h - 0.6) / 0.25);
        return mix(c3, c4, (h - 0.85) / 0.15);
      }
      void main() {
        #include <logdepthbuf_fragment>
        vec3 n = normalize(vLocal);
        vec3 wn = normalize(vNormal);
        float ndv = abs(dot(wn, normalize(vView)));
        vec4 s1 = tri(uNoise, n * 1.6 + uSeed + uTime * 0.08, n);
        vec4 s2 = tri(uNoise, n * 4.2 + uSeed * 2.0 - uTime * 0.13, n);
        float noise = smoothstep(0.22, 0.8, s1.r * 0.6 + s2.b * 0.4);
        // hotter in the middle of the disc, mottled by the noise
        float heat = uHeat * (0.15 + 1.0 * noise) * (0.4 + 0.6 * ndv);
        heat = clamp(heat, 0.0, 1.0);
        float edge = 1.0 - pow(1.0 - ndv, 3.5);
        float a = smoothstep(0.03, 0.35, heat) * edge * uAlpha * (0.3 + 0.7 * noise);
        vec3 col = ramp(heat) * (0.35 + 0.9 * heat);
        gl_FragColor = vec4(col * a * uIntensity, 1.0);
      }`,
    side: THREE.FrontSide,
    ...ADDITIVE
  });
}

// Shock front: a ragged plasma shell. The surface is gently noise-displaced
// so the silhouette is never a clean circle, and a noise threshold tears
// the rim into filaments; Blast.update keeps it dark for its first tens of
// milliseconds (born at the fireball's skin) - a perfect fresnel sphere at
// full strength from age 0 read as a glass bubble around the blast.
function shellMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uColor: { value: new THREE.Color(0.55, 0.75, 1.0) },
      uIntensity: { value: 1 },
      uSeed: { value: 0 },
      uRimPow: { value: 2.6 },
      uTear: { value: 0.5 }
    },
    vertexShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uSeed;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      ${TRIPLANAR_GLSL}
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vLocal = normalize(position);
        float bump = tri(uNoise, vLocal * 0.9 + uSeed * 0.37, vLocal).r;
        vec3 p = position * (0.95 + 0.1 * bump);
        vNormal = normalize(normalMatrix * normal);
        vec4 wp = modelMatrix * vec4(p, 1.0);
        vView = -(viewMatrix * wp).xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform vec3 uColor;
      uniform float uIntensity;
      uniform float uSeed;
      uniform float uRimPow;
      uniform float uTear;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      ${TRIPLANAR_GLSL}
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        float ndv = abs(dot(normalize(vNormal), normalize(vView)));
        float rim = pow(1.0 - ndv, uRimPow);
        float n = tri(uNoise, vLocal * 2.5 + uSeed, vLocal).g;
        float n2 = tri(uNoise, vLocal * 6.5 - uSeed * 1.3, vLocal).b;
        // torn into filaments: patches of the front simply are not there
        float tear = smoothstep(uTear - 0.18, uTear + 0.18, n * 0.75 + n2 * 0.45 + rim * 0.2);
        float ragged = (0.45 + 0.9 * n) * mix(1.0, tear, 0.85);
        gl_FragColor = vec4(uColor * rim * ragged * uIntensity, 1.0);
      }`,
    side: THREE.DoubleSide,
    ...ADDITIVE
  });
}

// A thin equatorial sheet of reactor ejecta, with broken, turbulent arcs.
function ejectaMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uAge: { value: 0 }, uIntensity: { value: 1 }, uSeed: { value: 0 }
    },
    vertexShader: /* glsl */`
      varying vec2 vUv;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uAge, uIntensity, uSeed;
      varying vec2 vUv;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        vec2 p = vUv * 2.0 - 1.0;
        float r = length(p);
        float n = texture2D(uNoise, p * 1.8 + uSeed).g;
        float front = 0.77 + (n - 0.5) * 0.04;
        float rim = exp(-pow((r - front) / 0.018, 2.0));
        float wake = exp(-pow((r - front + 0.08) / 0.085, 2.0)) * 0.22;
        float arcs = smoothstep(0.18, 0.68, n);
        vec3 color = mix(vec3(0.65, 1.25, 2.4), vec3(1.4, 0.35, 0.08), uAge);
        gl_FragColor = vec4(color * (rim + wake) * arcs * uIntensity, 1.0);
      }`,
    side: THREE.DoubleSide, ...ADDITIVE
  });
}

let _fireGeom = null, _shellGeom = null, _smallFireGeom = null, _ejectaGeom = null;
// PolyhedronGeometry subdivides linearly: detail d gives 20*(d+1)^2 faces
function fireGeom() { return _fireGeom || (_fireGeom = new THREE.IcosahedronGeometry(1, 28)); }
function smallFireGeom() { return _smallFireGeom || (_smallFireGeom = new THREE.IcosahedronGeometry(1, 14)); }
function shellGeom() { return _shellGeom || (_shellGeom = new THREE.IcosahedronGeometry(1, 24)); }

function easeOut(t, k = 3) { return 1 - Math.exp(-t * k); }

const _dq = new THREE.Quaternion();
const _dv = new THREE.Vector3();
const WRECK_LIFE = 40;       // big hull sections and the debris cloud linger
/** Dark coast before the reactor lets go (non-reactor kill, no sim death spiral). */
export const WRECK_DRIFT_S = 5.0;

/**
 * Battle time at which the latest destruction sequence in `deathInfo`
 * (BattleLoader.buildDeathInfo) has fully played out: drift + wreck life.
 */
export function latestWreckEnd(deathInfo) {
  let end = -Infinity;
  for (const d of Object.values(deathInfo || {})) {
    const drift = d.reactorCause || d.wasDying ? 0 : WRECK_DRIFT_S;
    end = Math.max(end, (d.time ?? 0) + drift + WRECK_LIFE);
  }
  return end;
}
const SHARD_FADE = [17, 22]; // small chunks fade out
// Blasts born on a hull's skin: spall puffs are thrown along the surface normal
const SURFACE_KINDS = new Set(['impact', 'pop', 'graze', 'pierce', 'kineticStrike', 'breach']);

/** Blast presets (exported for tests: kinds and their durations). */
export function blastKinds() { return KINDS; }

// ---------------------------------------------------------------------------
// Blast
// ---------------------------------------------------------------------------
const KINDS = {
  // Fusion reactor breach - the ship-killer
  reactor: {
    duration: 14, flash: 3.2, flashFade: 0.22, screenFlash: 0.65,
    fireR: 1.35, fireGrow: 1.2, fireLife: 6.5, fireTurb: 1.0,
    shellR: 18, shellLife: 2.4, shellW: 0.32,
    lobes: 6, ringR: 12, ringLife: 3.2,
    sparks: 2600, sparkSpeed: [2.5, 16], sparkLife: [2, 7], sparkSize: 0.075,
    streaks: 520, streakSpeed: [3, 13], streakLife: [1.2, 5], streakLen: 0.9, streakW: 0.022,
    puffs: 10, puffSize: [0.6, 1.5], puffSpeed: [0.3, 1.4], puffLife: 12,
    light: 25, lightLife: 2.2
  },
  // Overheat meltdown: containment lost to heat rather than a hit. Hotter
  // and brighter than a reactor kill: a white core flash, a whiter
  // fireball and a slow violet-white plasma shell that cools to orange.
  meltdown: {
    duration: 16, flash: 3.8, flashFade: 0.26, screenFlash: 0.85,
    fireR: 1.45, fireGrow: 1.4, fireLife: 7, fireTurb: 1.15, fireHeat: 1.7,
    shellR: 22, shellLife: 2.8, shellW: 0.34, shellColor: [0.7, 0.82, 1.25],
    core: { size: 1.7, life: 0.55, level: 7 },
    plasma: { r: 4.0, life: 4.5, rimPow: 3.0, level: 0.85, hot: [0.95, 0.88, 1.6], cool: [1.4, 0.5, 0.12] },
    lobes: 8, ringR: 15, ringLife: 3.6,
    sparks: 3000, sparkSpeed: [3, 19], sparkLife: [2, 8], sparkSize: 0.08,
    streaks: 620, streakSpeed: [3.5, 15], streakLife: [1.2, 5.5], streakLen: 1.0, streakW: 0.024,
    puffs: 10, puffSize: [0.7, 1.7], puffSpeed: [0.4, 1.6], puffLife: 12,
    light: 34, lightLife: 2.8
  },
  // Torpedo warhead: penetrator flash, fast plasma front, brief fireball
  warhead: {
    duration: 5.5, flash: 2.6, flashFade: 0.18, screenFlash: 0.25,
    fireR: 1.2, fireGrow: 1.2, fireLife: 3.2, fireTurb: 1.1,
    shellR: 10, shellLife: 1.3, shellW: 0.3,
    lobes: 3, ringR: 7, ringLife: 1.7,
    sparks: 900, sparkSpeed: [3, 18], sparkLife: [0.8, 3.5], sparkSize: 0.06,
    streaks: 260, streakSpeed: [4, 14], streakLife: [0.6, 2.5], streakLen: 0.8, streakW: 0.024,
    puffs: 9, puffSize: [0.9, 1.8], puffSpeed: [0.4, 1.6], puffLife: 4.5,
    light: 14, lightLife: 0.9
  },
  // Coilgun slug on armour: spall cone, molten spray, glowing crater
  impact: {
    duration: 3.0, flash: 2.6, flashFade: 0.12, screenFlash: 0.0,
    fireR: 0, shellR: 0,
    sparks: 380, sparkSpeed: [2, 12], sparkLife: [0.5, 2.6], sparkSize: 0.05,
    streaks: 120, streakSpeed: [3, 10], streakLife: [0.4, 1.8], streakLen: 0.6, streakW: 0.03,
    puffs: 3, puffSize: [0.5, 1.0], puffSpeed: [0.5, 1.5], puffLife: 2.2,
    light: 5, lightLife: 0.5, crater: true, cone: 0.75
  },
  // Slug skidding off the plating far from the normal: a fan of sparks
  // along the surface and a shallow gouge, hardly any vapour
  graze: {
    duration: 2.0, flash: 2.2, flashFade: 0.08, screenFlash: 0.0,
    fireR: 0, shellR: 0,
    sparks: 260, sparkSpeed: [3, 14], sparkLife: [0.3, 1.4], sparkSize: 0.04,
    streaks: 130, streakSpeed: [4, 14], streakLife: [0.25, 1.0], streakLen: 0.7, streakW: 0.025,
    puffs: 1, puffSize: [0.3, 0.6], puffSpeed: [0.3, 0.8], puffLife: 1.2,
    light: 3, lightLife: 0.3, crater: true, craterK: 0.55, cone: 0.92
  },
  // Slug through the armour: entry flash, a spall jet blown back out of
  // the hole and a brief fire inside it
  pierce: {
    duration: 3.0, flash: 3.0, flashFade: 0.12, screenFlash: 0.0, flashR: 1.6,
    fireR: 0.45, fireGrow: 2.0, fireLife: 1.0, fireTurb: 0.5,
    shellR: 0,
    sparks: 420, sparkSpeed: [2, 13], sparkLife: [0.5, 2.4], sparkSize: 0.05,
    streaks: 150, streakSpeed: [3, 11], streakLife: [0.4, 1.6], streakLen: 0.6, streakW: 0.03,
    puffs: 3, puffSize: [0.4, 0.8], puffSpeed: [0.5, 1.4], puffLife: 2.2,
    light: 5, lightLife: 0.5, crater: true, cone: 0.88
  },
  // Kinetic torpedo stopped by the armour. 250 kg at ~14 km/s dumps its
  // energy as a white flash, a plume of vaporised armour blown off the
  // plating and a spall cone - no fireball: there is no explosive.
  kineticStrike: {
    duration: 3.2, flash: 3.4, flashFade: 0.12, screenFlash: 0.05, flashR: 1.9,
    fireR: 0, shellR: 0,
    sparks: 520, sparkSpeed: [1.5, 10], sparkLife: [0.4, 2.4], sparkSize: 0.05,
    streaks: 190, streakSpeed: [2, 10], streakLife: [0.3, 1.6], streakLen: 0.6, streakW: 0.032,
    puffs: 6, puffSize: [0.45, 1.0], puffSpeed: [0.6, 1.8], puffLife: 2.0,
    light: 8, lightLife: 0.35, crater: true, cone: 0.78
  },
  // Kinetic torpedo through the armour: entry flash, a jet of spall and
  // vapour blown back out of the hole and a small fire inside it (the
  // venting that follows is DamageFX's)
  breach: {
    duration: 4.0, flash: 3.4, flashFade: 0.13, screenFlash: 0.08, flashR: 2.2,
    fireR: 0.6, fireGrow: 1.8, fireLife: 1.5, fireTurb: 0.6,
    shellR: 0,
    sparks: 520, sparkSpeed: [2, 12], sparkLife: [0.5, 2.6], sparkSize: 0.05,
    streaks: 190, streakSpeed: [3, 12], streakLife: [0.4, 1.8], streakLen: 0.6, streakW: 0.035,
    puffs: 4, puffSize: [0.4, 0.9], puffSpeed: [0.5, 1.4], puffLife: 2.4,
    light: 9, lightLife: 0.55, crater: true, cone: 0.88
  },
  // Cook-off: munitions or a fuel cell going up on a chunk (death-spiral
  // secondaries, wrecked mounts). No shock shell: at this size the front
  // was a hard-edged glass bubble around the fireball, never a shock wave;
  // thin streaks like the reactor's, not white bars
  secondary: {
    duration: 4.0, flash: 2.5, flashFade: 0.15, screenFlash: 0.0,
    fireR: 0.9, fireGrow: 1.0, fireLife: 2.2, fireTurb: 0.5,
    shellR: 0,
    sparks: 320, sparkSpeed: [2, 9], sparkLife: [0.6, 3], sparkSize: 0.05,
    streaks: 80, streakSpeed: [2, 8], streakLife: [0.5, 2], streakLen: 0.5, streakW: 0.02,
    puffs: 4, puffSize: [0.35, 0.7], puffSpeed: [0.3, 1.0], puffLife: 3.5,
    light: 6, lightLife: 0.7
  },
  // Hull fire pop on a dying ship
  pop: {
    duration: 1.6, flash: 1.6, flashFade: 0.1, screenFlash: 0.0,
    fireR: 0, shellR: 0,
    sparks: 140, sparkSpeed: [1.5, 7], sparkLife: [0.4, 1.4], sparkSize: 0.04,
    streaks: 40, streakSpeed: [2, 6], streakLife: [0.3, 1.0], streakLen: 0.4, streakW: 0.02,
    puffs: 1, puffSize: [0.35, 0.6], puffSpeed: [0.3, 0.9], puffLife: 0.9,
    light: 2, lightLife: 0.4, cone: 0.7
  },
  // Laser ablation while a PD beam dwells: a spit of sparks, no gas
  ablate: {
    duration: 0.8, flash: 1.2, flashFade: 0.08, screenFlash: 0.0,
    fireR: 0, shellR: 0,
    sparks: 70, sparkSpeed: [1.5, 6], sparkLife: [0.3, 0.8], sparkSize: 0.035,
    streaks: 20, streakSpeed: [2, 5], streakLife: [0.2, 0.6], streakLen: 0.3, streakW: 0.015,
    puffs: 0, puffSize: [0.3, 0.4], puffSpeed: [0.3, 0.6], puffLife: 0.5,
    light: 0, lightLife: 0.3, cone: 0.7
  }
};

export class Blast {
  /**
   * @param {Object} ctx  { scene, tryLight, freeLight, postfx, camera }
   * @param {Object} o    { position (Vector3), scale, spawnTime, kind,
   *                        dir (Vector3, optional spall direction),
   *                        baseVel (Vector3, optional inherited velocity) }
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.kind = o.kind;
    const K = KINDS[o.kind] || KINDS.secondary;
    this.K = K;
    this.scale = o.scale;
    this.spawnTime = o.spawnTime;
    this.duration = K.duration;
    const rnd = mulberry32(Math.floor((o.spawnTime * 977 + o.position.x * 13) % 100000));
    this.rnd = rnd;
    this.seed = rnd() * 10;
    const s = o.scale;

    this.group = new THREE.Group();
    this.group.position.copy(o.position);
    this.baseVel = o.baseVel ? o.baseVel.clone() : new THREE.Vector3();
    this.origin = o.position.clone();
    ctx.scene.add(this.group);

    const glow = getGlowTexture();

    // Flash core
    this.flash = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glow, color: new THREE.Color(K.flash, K.flash * 0.95, K.flash * 0.85),
      transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    this.flashScale = s * (K.flashR ?? (K.fireR > 0 ? 5 : 1.1));
    this.flash.scale.setScalar(this.flashScale);
    this.flash.renderOrder = 40;
    this.group.add(this.flash);

    // Fireball
    if (K.fireR > 0) {
      const big = o.kind === 'reactor' || o.kind === 'meltdown';
      this.fire = new THREE.Mesh(big ? fireGeom() : smallFireGeom(), fireballMaterial());
      this.fire.material.uniforms.uSeed.value = this.seed;
      this.fire.material.uniforms.uTurb.value = K.fireTurb;
      this.fire.renderOrder = 35;
      this.fire.frustumCulled = false;
      this.group.add(this.fire);
    }

    // Shock shell
    if (K.shellR > 0) {
      this.shell = new THREE.Mesh(shellGeom(), shellMaterial());
      this.shell.material.uniforms.uSeed.value = this.seed;
      if (K.shellColor) this.shell.material.uniforms.uColor.value.setRGB(...K.shellColor);
      this.shell.renderOrder = 36;
      this.shell.frustumCulled = false;
      this.group.add(this.shell);
    }

    // Meltdown: white-hot core flash and a thick, slow plasma shell
    if (K.core) {
      this.core = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glow, color: new THREE.Color(K.core.level, K.core.level, K.core.level),
        transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      this.core.renderOrder = 41;
      this.group.add(this.core);
    }
    if (K.plasma) {
      this.plasma = new THREE.Mesh(shellGeom(), shellMaterial());
      const u = this.plasma.material.uniforms;
      u.uSeed.value = this.seed + 3.1;
      u.uRimPow.value = K.plasma.rimPow;
      this.plasma.renderOrder = 36;
      this.plasma.frustumCulled = false;
      this.group.add(this.plasma);
    }

    this.lobes = [];
    for (let i = 0; i < (K.lobes || 0); i++) {
      const mesh = new THREE.Mesh(smallFireGeom(), fireballMaterial());
      mesh.material.uniforms.uSeed.value = this.seed + i * 2.71;
      mesh.renderOrder = 35;
      mesh.frustumCulled = false;
      const direction = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
      this.lobes.push({ mesh, direction, delay: 0.08 + rnd() * 0.38, size: 0.45 + rnd() * 0.35 });
      this.group.add(mesh);
    }
    if (K.ringR) {
      _ejectaGeom ||= new THREE.PlaneGeometry(2, 2);
      this.ejecta = new THREE.Mesh(_ejectaGeom, ejectaMaterial());
      this.ejecta.material.uniforms.uSeed.value = this.seed;
      const axis = o.axis?.clone().normalize() || new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, 1).normalize();
      this.ejecta.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      this.ejecta.renderOrder = 36;
      this.ejecta.frustumCulled = false;
      this.group.add(this.ejecta);
    }

    // Particles
    const dir = o.dir ? o.dir.clone().normalize() : null;
    this.sparks = new SparkBurst({
      count: K.sparks, speed: K.sparkSpeed.map(v => v * s), life: K.sparkLife,
      size: K.sparkSize * s, drag: 0.35, spread: s * 0.25, rnd,
      dir, cone: K.cone ?? 0, intensity: 1.3
    });
    this.group.add(this.sparks.object);
    this.streaks = new StreakBurst({
      count: K.streaks, speed: K.streakSpeed.map(v => v * s), life: K.streakLife,
      length: K.streakLen * s, width: K.streakW * s, drag: 0.3, spread: s * 0.2, rnd,
      dir, cone: K.cone ?? 0, intensity: 1.4
    });
    this.group.add(this.streaks.object);
    const surface = SURFACE_KINDS.has(o.kind);
    this.puffs = new PuffCloud({
      count: K.puffs, size: K.puffSize.map(v => v * s), speed: K.puffSpeed.map(v => v * s),
      duration: K.puffLife, spread: s * 0.3, rnd, stagger: 0.25, grow: 1.2,
      intensity: surface ? (o.kind === 'kineticStrike' || o.kind === 'breach' ? 0.55 : 0.45)
        : (o.kind === 'reactor' || o.kind === 'meltdown' ? 0.2 : 0.3)
    });
    if (dir && surface) {
      // spall goes out along the surface normal
      for (const p of this.puffs.puffs) { p.vel.lerp(dir.clone().multiplyScalar(p.vel.length()), 0.7); }
    }
    this.group.add(this.puffs.object);

    // Crater glow: a hot spot that cools over the blast duration
    if (K.crater) {
      this.crater = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glow, color: new THREE.Color(2.4, 0.9, 0.3),
        transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      this.crater.scale.setScalar(s * (K.craterK ?? 0.9));
      this.crater.renderOrder = 34;
      this.group.add(this.crater);
    }

    // Pool light: positioned in world space each update, never reparented
    this.light = K.light > 0 ? ctx.tryLight(o.kind === 'meltdown' ? 0xffeef8 : 0xffd2a0, 0, s * 60,
      o.kind === 'reactor' || o.kind === 'meltdown' ? 4 : 0) : null;

    // Screen flash scaled by how close the camera is - only for a blast
    // happening now, not one re-created by a timeline seek
    const spawnAge = ctx.now ? ctx.now() - o.spawnTime : 0;
    this.fresh = spawnAge < 0.5;
    if (K.screenFlash > 0 && ctx.postfx && this.fresh) {
      const d = ctx.camera.position.distanceTo(o.position);
      const near = s * 40;
      const amt = K.screenFlash * Math.min(1, (near / Math.max(d, 1)) ** 1.2);
      ctx.postfx.triggerFlash(amt);
    }
    this.update(ctx.now ? ctx.now() : o.spawnTime, 0);
  }

  /** @returns {boolean} still alive */
  update(now, wallTime) {
    const age = now - this.spawnTime;
    const K = this.K, s = this.scale;
    if (age < 0) { this.group.visible = false; return true; }
    if (age > this.duration) return false;
    this.group.visible = true;
    this.group.position.copy(this.origin).addScaledVector(this.baseVel, age);

    // Flash: instant, then fast exponential decay. Up close the sprite
    // would fill the frame with one flat HDR disc, so its footprint is
    // capped at 0.4 of the camera distance (about a fifth of the view) and
    // the fireball, shell and sparks carry the rest of the blast.
    const f = Math.exp(-age / K.flashFade);
    this.flash.material.opacity = Math.min(1, f * 1.2);
    const camPos = this.ctx.camera?.position;
    const cap = camPos ? camPos.distanceTo(this.group.position) * 0.4 : Infinity;
    this.flash.scale.setScalar(Math.min(cap, this.flashScale * (1 + age * 1.5)));

    let fireRadius = 0;
    if (this.fire) {
      const life = K.fireLife;
      const p = Math.min(1, age / life);
      const r = s * K.fireR * (0.25 + easeOut(age * K.fireGrow, 1.6) * 1.0) * (1 + p * 0.6);
      fireRadius = age < life ? r : 0;
      this.fire.scale.setScalar(r);
      const u = this.fire.material.uniforms;
      u.uTime.value = age;
      u.uHeat.value = (K.fireHeat || 1.25) * Math.pow(1 - p, 0.85);
      u.uAlpha.value = Math.min(1, age / 0.12) * Math.pow(1 - p, 0.6);
      u.uIntensity.value = 1.4 * (0.5 + 0.5 * Math.exp(-age / (life * 0.35)));
      u.uTurb.value = K.fireTurb * (0.7 + p * 0.9);
      this.fire.visible = age < life;
    }
    for (const lobe of this.lobes) {
      const t = age - lobe.delay;
      const p = THREE.MathUtils.clamp(t / (K.fireLife * 0.8), 0, 1);
      lobe.mesh.visible = t >= 0 && p < 1;
      const reach = s * K.fireR * easeOut(Math.max(0, t), 1.2);
      lobe.mesh.position.copy(lobe.direction).multiplyScalar(reach * (0.65 + p * 0.8));
      lobe.mesh.scale.setScalar(Math.max(0.001, reach * lobe.size * (1 + p * 0.4)));
      const u = lobe.mesh.material.uniforms;
      u.uTime.value = Math.max(0, t) * 1.3;
      u.uHeat.value = 1.05 * Math.pow(1 - p, 0.9);
      u.uAlpha.value = Math.min(1, Math.max(0, t) * 5) * Math.pow(1 - p, 0.8);
      u.uIntensity.value = 0.85;
      u.uTurb.value = 0.85 + p * 0.5;
    }
    if (this.ejecta) {
      const p = Math.min(1, age / K.ringLife);
      this.ejecta.visible = p < 1;
      this.ejecta.scale.setScalar(s * K.ringR * (0.025 + easeOut(p, 2.1)));
      this.ejecta.material.uniforms.uAge.value = p;
      this.ejecta.material.uniforms.uIntensity.value = 2.5 * Math.pow(1 - p, 2);
    }
    if (this.shell) {
      // The front is born at the fireball's skin, not as a tiny bright
      // bead in its middle, and fades in over its first ~0.1 s
      const p = Math.min(1, age / K.shellLife);
      const r = Math.max(fireRadius * 1.04, s * K.shellR * (0.05 + easeOut(p * 3.2, 1.8)));
      this.shell.scale.setScalar(r);
      const rise = THREE.MathUtils.smoothstep(age, Math.min(0.05, K.shellLife * 0.1), Math.min(0.22, K.shellLife * 0.3));
      this.shell.material.uniforms.uIntensity.value = K.shellW * 1.8 * Math.pow(1 - p, 3.5) * rise;
      // the front tears open as it thins out
      this.shell.material.uniforms.uTear.value = 0.42 + 0.3 * p;
      this.shell.visible = p < 1;
    }
    if (this.core) {
      const c = K.core, f = Math.exp(-age / c.life);
      this.core.visible = f > 0.01;
      this.core.material.opacity = Math.min(1, f * 1.4);
      this.core.scale.setScalar(Math.min(cap * 0.8, s * c.size * (0.6 + age * 2.2)));
    }
    if (this.plasma) {
      const P = K.plasma, p = Math.min(1, age / P.life);
      this.plasma.visible = p < 1;
      this.plasma.scale.setScalar(s * P.r * (0.03 + easeOut(p * 2.4, 1.3)));
      // violet-white while it is bright, cooling to orange as it fades
      const u = this.plasma.material.uniforms, k = THREE.MathUtils.smoothstep(p, 0.35, 0.9);
      u.uTear.value = 0.3 + 0.35 * p;
      u.uColor.value.setRGB(
        P.hot[0] + (P.cool[0] - P.hot[0]) * k, P.hot[1] + (P.cool[1] - P.hot[1]) * k, P.hot[2] + (P.cool[2] - P.hot[2]) * k);
      u.uIntensity.value = P.level * Math.min(1, age / 0.2) * Math.pow(1 - p, 1.8);
    }
    this.sparks.setAge(age);
    this.streaks.setAge(age);
    this.puffs.setAge(age);
    // The additive spark/streak burst piles into a white blowout when the
    // camera sits a few blast radii away; fleet range is unchanged
    const dim = camPos ? blastCloseDim(camPos.distanceTo(this.group.position), s) : 1;
    this.sparks.material.uniforms.uIntensity.value = 1.3 * dim;
    this.streaks.material.uniforms.uIntensity.value = 1.4 * dim;
    if (this.crater) {
      const p = Math.min(1, age / this.duration);
      this.crater.material.opacity = Math.pow(1 - p, 1.8);
      this.crater.material.color.setRGB(2.4 * (1 - p * 0.6), 0.9 * (1 - p * 0.8), 0.3 * (1 - p));
    }
    if (this.light) {
      this.light.position.copy(this.group.position);
      this.light.intensity = K.light * s * s * Math.exp(-age / K.lightLife);
    }
    return true;
  }

  dispose() {
    this.ctx.scene.remove(this.group);
    this.flash.material.dispose();
    if (this.fire) this.fire.material.dispose();
    if (this.shell) this.shell.material.dispose();
    if (this.core) this.core.material.dispose();
    if (this.plasma) this.plasma.material.dispose();
    for (const lobe of this.lobes) lobe.mesh.material.dispose();
    if (this.ejecta) this.ejecta.material.dispose();
    this.sparks.dispose();
    this.streaks.dispose();
    this.puffs.dispose();
    if (this.crater) this.crater.material.dispose();
    this.ctx.freeLight(this.light);
  }
}

// ---------------------------------------------------------------------------
// Ship destruction: drift (dying hulk) -> reactor breach -> hull breakup
// ---------------------------------------------------------------------------
export class ShipDestruction {
  /**
   * @param {Object} ctx  { scene, tryLight, freeLight, postfx, camera, spawnBlast(o) }
   * @param {Object} o    { shipGroup, shipId, position (Vector3 scene units),
   *                        driftVel (Vector3 scene units/s), driftDuration,
   *                        scale, spawnTime }
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.shipId = o.shipId;
    this.shipGroup = o.shipGroup;
    this.basePos = o.position.clone();
    this.driftVel = o.driftVel.clone();
    this.driftDuration = o.driftDuration;
    this.scale = o.scale;
    this.spawnTime = o.spawnTime;
    this.meltdown = !!o.meltdown;   // overheat: detonates as the 'meltdown' blast
    this.totalDuration = this.driftDuration + WRECK_LIFE;
    this.rnd = mulberry32(Math.floor(o.spawnTime * 131 + 7));
    const rnd = this.rnd;

    this.tumbleAxis = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    this.tumbleRate = 0.08 + rnd() * 0.14;
    this.initialOrientation = this.shipGroup?.quaternion.clone() || new THREE.Quaternion();
    this.tumble = new THREE.Quaternion();
    this.ignited = false;
    this.chunks = [];
    this.chunkGroup = null;
    this.secondaries = [];   // {t, chunk, fired, offset}
    this.sections = [];      // big torn hull sections (glowing edges)
    this.cloud = null;       // instanced debris cloud
    this.embers = null;      // long-lived ember sparks
    this.vents = [];
    this.pops = [];

    // Drift-phase schedule: hull fires and venting jets in ship-local space
    const g = this.shipGroup;
    const L = g?.userData.size?.length || o.scale;
    const W = g?.userData.size?.width || o.scale * 0.25;
    if (this.driftDuration > 0 && g) {
      const popCount = 6 + Math.floor(rnd() * 5);
      for (let i = 0; i < popCount; i++) {
        this.pops.push({
          t: 0.3 + rnd() * (this.driftDuration - 1.0),
          local: new THREE.Vector3((rnd() - 0.5) * W * 0.9, (rnd() - 0.5) * W * 0.9, (rnd() - 0.5) * L * 0.85),
          fired: false
        });
      }
      this.pops.sort((a, b) => a.t - b.t);
      // secondaries walk from the wound toward the reactor, a few of them
      // cooking off as proper blasts
      const rz = g.userData.reactorPos ? g.userData.reactorPos.z : -L * 0.3;
      this.pops.forEach((pop, i) => {
        const u = popCount > 1 ? i / (popCount - 1) : 0;
        pop.local.z = L * 0.38 + (rz - L * 0.38) * u + (rnd() - 0.5) * L * 0.08;
        pop.kind = rnd() < 0.3 ? 'secondary' : 'pop';
      });
      // Venting plasma jets from breached tanks
      const ventCount = 2 + Math.floor(rnd() * 2);
      for (let i = 0; i < ventCount; i++) {
        const side = new THREE.Vector3((rnd() - 0.5), (rnd() - 0.5), (rnd() - 0.5) * 0.3).normalize();
        const local = new THREE.Vector3(side.x * W * 0.45, side.y * W * 0.45, (rnd() - 0.5) * L * 0.7);
        const torch = createTorch({ radius: W * 0.09, length: L * 0.5, tint: 'chemical', intensity: 0.35 });
        torch.group.position.copy(local);
        torch.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), side);
        g.add(torch.group);
        this.vents.push({ torch, start: 0.2 + rnd() * this.driftDuration * 0.5, phase: rnd() * 6 });
      }
    }
    this.hulkPos = this.basePos.clone();
  }

  _ignite() {
    this.ignited = true;
    // Anchor to the recorded detonation time, not the frame that noticed
    // it, so a timeline jump past a death lands mid-sequence
    const now = this.spawnTime + this.driftDuration;
    const g = this.shipGroup;
    const rnd = this.rnd;
    // Detonation point: the reactor, on the drifted, tumbled hulk
    const origin = this.basePos.clone().addScaledVector(this.driftVel, this.driftDuration);
    let reactorWorld = origin.clone();
    if (g) {
      this.tumble.setFromAxisAngle(this.tumbleAxis, this.tumbleRate * this.driftDuration);
      g.quaternion.copy(this.initialOrientation).multiply(this.tumble);
      const reactor = g.userData.reactorPos || new THREE.Vector3();
      reactorWorld = reactor.clone().applyQuaternion(g.quaternion).add(origin);
    }
    this.blastPos = reactorWorld;
    this.blastOrigin = reactorWorld.clone();
    for (const v of this.vents) { v.torch.group.visible = false; }

    this.ctx.spawnBlast({
      position: reactorWorld, scale: this.scale, spawnTime: now, kind: this.meltdown ? 'meltdown' : 'reactor',
      baseVel: this.driftVel,
      axis: new THREE.Vector3(0, 0, 1).applyQuaternion(g?.quaternion || new THREE.Quaternion())
    });

    // Hull breakup: the reactor end shreds into chunks, the rest tears into
    // one or two big sections with glowing edges that tumble apart slowly
    if (g) {
      this.chunkGroup = new THREE.Group();
      this.ctx.scene.add(this.chunkGroup);
      const reactorLocal = g.userData.reactorPos || new THREE.Vector3();
      const L = g.userData.size?.length || this.scale;
      const fx = this.ctx.damage || this.ctx.scene.userData.damageFX || null;
      // wear just before the reactor let go (scrub-exact, from the timeline)
      const wear = fx?.wearAt?.(this.shipId, now - 0.05) || null;
      const broken = fx ? breakHull(g, rnd, fx, wear) : null;
      const chunks = broken ? broken.shards : buildDebrisChunks(g, 9 + Math.floor(rnd() * 4), rnd);
      if (broken) {
        const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(g.quaternion);
        for (const sec of broken.sections) {
          const worldCenter = sec.center.clone().applyQuaternion(g.quaternion).add(origin);
          const away = sec.center.z >= reactorLocal.z ? 1 : -1;
          const kick = this.scale * (0.04 + 0.05 * rnd());
          const lateral = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize().multiplyScalar(kick * 0.7);
          const vel = this.driftVel.clone().addScaledVector(axis, away * kick * (sec.cutHi > 1e5 ? 1.4 : 0.6)).add(lateral);
          const spinAxis = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
          sec.mesh.position.copy(worldCenter);
          sec.mesh.quaternion.copy(g.quaternion);
          this.chunkGroup.add(sec.mesh);
          const entry = {
            mesh: sec.mesh, overlay: sec.overlay, wear: sec.wear, p0: worldCenter, vel, q0: g.quaternion.clone(),
            spinAxis, spinRate: 0.05 + 0.2 * rnd(), heat: 0.9 + 0.4 * rnd()
          };
          this.sections.push(entry);
          // fires along the torn edge
          const tear = new THREE.Vector3(0, 0, sec.cutLo - sec.center.z);
          for (let k = 0; k < 2; k++) {
            this.secondaries.push({
              t: 0.8 + rnd() * 6.0, chunk: entry,
              offset: tear.clone().add(new THREE.Vector3((rnd() - 0.5) * L * 0.08, (rnd() - 0.5) * L * 0.08, 0)),
              scale: this.scale * (0.18 + rnd() * 0.15), kind: rnd() < 0.5 ? 'secondary' : 'pop', fired: false
            });
          }
        }
        this._buildCloud(fx, g, origin, reactorLocal, now, rnd);
      }
      for (const c of chunks) {
        const worldCenter = c.center.clone().applyQuaternion(g.quaternion).add(origin);
        const radial = c.center.clone().sub(reactorLocal);
        const dist = Math.max(radial.length(), L * 0.05);
        radial.normalize().applyQuaternion(g.quaternion);
        // closer to the reactor = harder kick; smaller pieces fly faster
        const kick = this.scale * (0.12 + 0.5 / (1 + dist / (L * 0.25))) * (0.7 + rnd() * 0.6) / Math.max(0.6, c.size / (L * 0.08));
        const vel = this.driftVel.clone()
          .addScaledVector(radial, kick)
          .add(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(kick * 0.35));
        const spinAxis = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
        const spinRate = (0.4 + rnd() * 1.6) * (L * 0.15 / Math.max(c.size, L * 0.03));
        c.mesh.position.copy(worldCenter);
        c.mesh.quaternion.copy(g.quaternion);
        this.chunkGroup.add(c.mesh);
        this.chunks.push({
          mesh: c.mesh, p0: worldCenter, vel, q0: g.quaternion.clone(), spinAxis, spinRate: Math.min(spinRate, 3.0),
          heat: 0.8 + rnd() * 0.8, size: c.size
        });
      }
      // Cook-offs on a few chunks over the next seconds
      const n = 3 + Math.floor(rnd() * 4);
      for (let i = 0; i < n && this.chunks.length; i++) {
        this.secondaries.push({
          t: 0.5 + rnd() * 5.0,
          chunk: this.chunks[Math.floor(rnd() * this.chunks.length)],
          scale: this.scale * (0.3 + rnd() * 0.3),
          fired: false
        });
      }
      g.visible = false;
    }
  }

  /**
   * Instanced debris cloud (hundreds of glowing fragments thrown from the
   * reactor end) plus long-lived embers drifting with the wreck.
   */
  _buildCloud(fx, g, origin, reactorLocal, t0, rnd) {
    const L = g.userData.size?.length || this.scale;
    const W = g.userData.size?.width || L * 0.22;
    const n = Math.round(160 + 40 * (this.scale / 6));
    const cloud = fx.createDebrisSet(n);
    const p = new THREE.Vector3(), v = new THREE.Vector3(), axis = new THREE.Vector3(), dir = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      // born inside the hull, clustered around the reactor
      const local = new THREE.Vector3((rnd() - 0.5) * W * 0.8, (rnd() - 0.5) * W * 0.8,
        reactorLocal.z + (rnd() - 0.35) * L * (0.25 + 0.6 * rnd()));
      p.copy(local).applyQuaternion(g.quaternion).add(origin);
      dir.copy(local).sub(reactorLocal);
      if (dir.lengthSq() < 1e-6) dir.set(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
      dir.normalize().add(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(0.8)).normalize()
        .applyQuaternion(g.quaternion);
      const speed = this.scale * (0.05 + 0.6 * Math.pow(rnd(), 2.2));
      v.copy(this.driftVel).addScaledVector(dir, speed);
      axis.set(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
      const size = L * (0.005 + 0.022 * Math.pow(rnd(), 2.5));
      const flat = rnd() < 0.45;
      cloud.set(i, p, t0, v, 18 + 20 * rnd(), axis, 0.5 + 4 * rnd(),
        size * (flat ? 1.6 : 1), size * (flat ? 0.25 : 0.85), size * (flat ? 1.2 : 1), 0.9 + 1.1 * rnd());
    }
    cloud.flush();
    this.cloud = cloud;

    this.embers = new SparkBurst({
      count: 700, speed: [0.05 * this.scale, 0.55 * this.scale], life: [3, 16],
      size: 0.05 * this.scale, drag: 0.12, spread: L * 0.12, rnd,
      hot: [2.4, 1.6, 0.8], warm: [1.8, 0.6, 0.12], cool: [0.35, 0.05, 0.0], intensity: 1.0
    });
    this.emberGroup = new THREE.Group();
    this.emberGroup.add(this.embers.object);
    this.emberOrigin = origin.clone().add(reactorLocal.clone().applyQuaternion(g.quaternion));
    this.emberGroup.position.copy(this.emberOrigin);
    this.ctx.scene.add(this.emberGroup);
  }

  /**
   * @param {number} now       battle time
   * @param {number} delta     wall delta (for the hulk tumble)
   * @param {number} wallTime  wall clock (torch turbulence)
   * @returns {boolean} alive
   */
  update(now, delta, wallTime) {
    const age = now - this.spawnTime;
    const blastAge = age - this.driftDuration;
    const g = this.shipGroup;
    if (age < 0) return true;
    if (blastAge > WRECK_LIFE) return false;

    if (blastAge < 0) {
      // DRIFT: coasting dark, tumbling, venting, popping
      this.hulkPos.copy(this.basePos).addScaledVector(this.driftVel, age);
      if (g) {
        g.visible = true;
        g.position.copy(this.hulkPos);
        this.tumble.setFromAxisAngle(this.tumbleAxis, this.tumbleRate * age);
        g.quaternion.copy(this.initialOrientation).multiply(this.tumble);
        for (const v of this.vents) {
          // pressure pulses on battle time (a paused frame keeps its look);
          // the wall clock only scrolls the plume's turbulence
          const a = age - v.start;
          const on = a > 0 ? Math.min(1, a / 0.6) * (0.55 + 0.45 * Math.sin(now * 1.7 + v.phase)) : 0;
          v.torch.update(on, wallTime, 0.85, now);
        }
      }
      for (const pop of this.pops) {
        if (pop.fired || pop.t > age) continue;
        pop.fired = true;
        if (age - pop.t < KINDS[pop.kind || 'pop'].duration && g) {
          const orientation = this.initialOrientation.clone().multiply(
            this.tumble.setFromAxisAngle(this.tumbleAxis, this.tumbleRate * pop.t));
          const world = pop.local.clone().applyQuaternion(orientation).add(this.basePos).addScaledVector(this.driftVel, pop.t);
          const dir = pop.local.clone().setZ(0).normalize().applyQuaternion(orientation);
          const kind = pop.kind || 'pop';
          this.ctx.spawnBlast({ position: world, scale: this.scale * (kind === 'secondary' ? 0.22 : 0.35), spawnTime: this.spawnTime + pop.t, kind, dir, baseVel: this.driftVel });
        }
      }
      return true;
    }

    if (!this.ignited) this._ignite();
    this.blastPos.copy(this.blastOrigin).addScaledVector(this.driftVel, blastAge);

    // Chunks: ballistic, tumbling, cooling from orange-hot to dark
    const fade = blastAge > SHARD_FADE[0] ? Math.max(0, 1 - (blastAge - SHARD_FADE[0]) / (SHARD_FADE[1] - SHARD_FADE[0])) : 1;
    const q = _dq;
    for (const c of this.chunks) {
      c.mesh.visible = fade > 0;
      c.mesh.position.copy(c.p0).addScaledVector(c.vel, blastAge);
      q.setFromAxisAngle(c.spinAxis, c.spinRate * blastAge);
      c.mesh.quaternion.copy(c.q0).multiply(q);
      // white-hot edges cool to a dull ember glow over ~10 s
      const heat = c.heat * (0.75 * Math.exp(-blastAge / 1.6) + 0.25 * Math.exp(-blastAge / 6));
      const mats = Array.isArray(c.mesh.material) ? c.mesh.material : [c.mesh.material];
      for (const m of mats) {
        m.emissiveIntensity = heat * 0.22;
        m.opacity = Math.max(0, fade);
      }
    }
    // Big sections: slow tumble, torn edges glow and cool, long linger
    const secFade = blastAge > WRECK_LIFE - 6 ? Math.max(0, (WRECK_LIFE - blastAge) / 6) : 1;
    for (const sec of this.sections) {
      sec.mesh.position.copy(sec.p0).addScaledVector(sec.vel, blastAge);
      q.setFromAxisAngle(sec.spinAxis, sec.spinRate * blastAge);
      sec.mesh.quaternion.copy(sec.q0).multiply(q);
      const heat = sec.heat * (1.5 * Math.exp(-blastAge / 2.2) + 0.75 * Math.exp(-blastAge / 11));
      const mats = Array.isArray(sec.mesh.material) ? sec.mesh.material : [sec.mesh.material];
      for (const m of mats) {
        m.emissiveIntensity = heat * 0.03;
        m.opacity = secFade;
      }
      if (sec.overlay) {
        sec.overlay.material.uniforms.uHeat.value = heat;
        sec.overlay.material.uniforms.uFade.value = secFade;
      }
      if (sec.wear) {
        // the hulk's wear rides on the section; its cracks cool with the edges
        sec.wear.material.opacity = secFade;
        sec.wear.userData.uniforms.uGlow.value = sectionCrackGlow(heat);
      }
    }
    if (this.embers) {
      this.emberGroup.position.copy(this.emberOrigin).addScaledVector(this.driftVel, blastAge);
      this.embers.setAge(blastAge);
    }
    for (const s of this.secondaries) {
      if (s.fired || s.t > blastAge) continue;
      s.fired = true;
      const kind = s.kind || 'secondary';
      if (blastAge - s.t < KINDS[kind].duration) {
        const pos = s.chunk.p0.clone().addScaledVector(s.chunk.vel, s.t);
        if (s.offset) {
          q.setFromAxisAngle(s.chunk.spinAxis, s.chunk.spinRate * s.t);
          _dv.copy(s.offset).applyQuaternion(q).applyQuaternion(s.chunk.q0);
          pos.add(_dv);
        }
        this.ctx.spawnBlast({ position: pos, scale: s.scale, spawnTime: this.spawnTime + this.driftDuration + s.t, kind, baseVel: s.chunk.vel });
      }
    }
    return true;
  }

  /** World position of the wreck centre (camera framing, beams). */
  get position() {
    return this.ignited && this.blastPos ? this.blastPos : this.hulkPos;
  }

  dispose() {
    for (const v of this.vents) {
      if (this.shipGroup) this.shipGroup.remove(v.torch.group);
      v.torch.dispose();
    }
    if (this.chunkGroup) {
      this.ctx.scene.remove(this.chunkGroup);
      for (const c of [...this.chunks, ...this.sections]) {
        c.mesh.geometry.dispose();
        const mats = Array.isArray(c.mesh.material) ? c.mesh.material : [c.mesh.material];
        for (const m of mats) m.dispose();
        if (c.overlay) { c.overlay.geometry.dispose(); c.overlay.material.dispose(); }
        if (c.wear) { c.wear.geometry.dispose(); c.wear.material.dispose(); }
      }
    }
    if (this.cloud) this.cloud.dispose();
    if (this.embers) {
      this.ctx.scene.remove(this.emberGroup);
      this.embers.dispose();
    }
    if (this.shipGroup) this.shipGroup.visible = false;
  }
}
