import * as THREE from 'three';
import { getNoiseTexture, getGlowTexture, mulberry32 } from './Textures.js';
import { SparkBurst, StreakBurst, PuffCloud } from './Particles.js';
import { buildDebrisChunks } from './Hulls.js';
import { createTorch } from './Torch.js';

/**
 * Blasts and ship destruction.
 *
 * Everything is driven by battle time (age = now - spawnTime) so scrubbing
 * the timeline replays a detonation exactly; only surface turbulence uses
 * the wall clock so a paused frame still looks alive.
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
        vec4 d1 = tri(uNoise, n * 0.9 + uSeed + uTime * 0.05, n);
        vec4 d2 = tri(uNoise, n * 2.3 - uSeed * 1.7 - uTime * 0.09, n);
        float disp = (d1.r - 0.5) * 0.6 + (d2.g - 0.5) * 0.2;
        vec3 p = position * (1.0 + disp * uTurb);
        vLocal = n;
        vNormal = normalize(normalMatrix * normal);
        vec4 wp = modelMatrix * vec4(p, 1.0);
        vView = normalize(cameraPosition - wp.xyz);
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
        float noise = s1.r * 0.6 + s2.b * 0.4;
        // hotter in the middle of the disc, mottled by the noise
        float heat = uHeat * (0.15 + 1.0 * noise) * (0.4 + 0.6 * ndv);
        heat = clamp(heat, 0.0, 1.0);
        float edge = 1.0 - pow(1.0 - ndv, 3.5);
        float a = smoothstep(0.03, 0.35, heat) * edge * uAlpha;
        vec3 col = ramp(heat) * (0.35 + 0.9 * heat);
        gl_FragColor = vec4(col * a * uIntensity, 1.0);
      }`,
    side: THREE.DoubleSide,
    ...ADDITIVE
  });
}

// Shock front: a thin, ragged fresnel shell
function shellMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uColor: { value: new THREE.Color(0.55, 0.75, 1.0) },
      uIntensity: { value: 1 },
      uSeed: { value: 0 }
    },
    vertexShader: /* glsl */`
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      #include <common>
  #include <logdepthbuf_pars_vertex>
      void main() {
        vLocal = normalize(position);
        vNormal = normalize(normalMatrix * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vView = normalize(cameraPosition - wp.xyz);
        gl_Position = projectionMatrix * viewMatrix * wp;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform vec3 uColor;
      uniform float uIntensity;
      uniform float uSeed;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vLocal;
      ${TRIPLANAR_GLSL}
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        float ndv = abs(dot(normalize(vNormal), normalize(vView)));
        float rim = pow(1.0 - ndv, 3.4);
        float n = tri(uNoise, vLocal * 2.5 + uSeed, vLocal).g;
        float ragged = 0.55 + 0.9 * n;
        gl_FragColor = vec4(uColor * rim * ragged * uIntensity, 1.0);
      }`,
    side: THREE.DoubleSide,
    ...ADDITIVE
  });
}

let _fireGeom = null, _shellGeom = null, _smallFireGeom = null;
// PolyhedronGeometry subdivides linearly: detail d gives 20*(d+1)^2 faces
function fireGeom() { return _fireGeom || (_fireGeom = new THREE.IcosahedronGeometry(1, 28)); }
function smallFireGeom() { return _smallFireGeom || (_smallFireGeom = new THREE.IcosahedronGeometry(1, 14)); }
function shellGeom() { return _shellGeom || (_shellGeom = new THREE.IcosahedronGeometry(1, 12)); }

function easeOut(t, k = 3) { return 1 - Math.exp(-t * k); }

// ---------------------------------------------------------------------------
// Blast
// ---------------------------------------------------------------------------
const KINDS = {
  // Fusion reactor breach - the ship-killer
  reactor: {
    duration: 14, flash: 3.2, flashFade: 0.22, screenFlash: 0.65,
    fireR: 1.15, fireGrow: 1.4, fireLife: 5.5, fireTurb: 0.3,
    shellR: 18, shellLife: 2.4, shellW: 0.32,
    sparks: 2600, sparkSpeed: [2.5, 16], sparkLife: [2, 7], sparkSize: 0.075,
    streaks: 700, streakSpeed: [3, 13], streakLife: [1.2, 5], streakLen: 0.9, streakW: 0.05,
    puffs: 10, puffSize: [0.6, 1.5], puffSpeed: [0.3, 1.4], puffLife: 12,
    light: 25, lightLife: 2.2
  },
  // Torpedo warhead: penetrator flash, fast plasma front, brief fireball
  warhead: {
    duration: 5.5, flash: 2.6, flashFade: 0.18, screenFlash: 0.25,
    fireR: 1.1, fireGrow: 1.2, fireLife: 2.8, fireTurb: 0.5,
    shellR: 10, shellLife: 1.3, shellW: 0.3,
    sparks: 900, sparkSpeed: [3, 18], sparkLife: [0.8, 3.5], sparkSize: 0.06,
    streaks: 260, streakSpeed: [4, 14], streakLife: [0.6, 2.5], streakLen: 0.8, streakW: 0.04,
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
  // Cook-off: munitions or a fuel cell going up on a chunk
  secondary: {
    duration: 4.0, flash: 2.5, flashFade: 0.15, screenFlash: 0.0,
    fireR: 0.9, fireGrow: 1.0, fireLife: 2.2, fireTurb: 0.5,
    shellR: 1.3, shellLife: 0.7, shellW: 0.3,
    sparks: 320, sparkSpeed: [2, 9], sparkLife: [0.6, 3], sparkSize: 0.05,
    streaks: 90, streakSpeed: [2, 8], streakLife: [0.5, 2], streakLen: 0.5, streakW: 0.03,
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
    this.flashScale = s * (K.fireR > 0 ? 5 : 1.1);
    this.flash.scale.setScalar(this.flashScale);
    this.flash.renderOrder = 40;
    this.group.add(this.flash);

    // Fireball
    if (K.fireR > 0) {
      this.fire = new THREE.Mesh(o.kind === 'reactor' ? fireGeom() : smallFireGeom(), fireballMaterial());
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
      this.shell.renderOrder = 36;
      this.shell.frustumCulled = false;
      this.group.add(this.shell);
    }

    // Particles
    const dir = o.dir ? o.dir.clone().normalize() : null;
    this.sparks = new SparkBurst({
      count: K.sparks, speed: K.sparkSpeed.map(v => v * s), life: K.sparkLife,
      size: K.sparkSize * s, drag: 0.35, spread: s * 0.25, rnd,
      dir, cone: K.cone ?? 0, baseVel: this.baseVel, intensity: 1.3
    });
    this.group.add(this.sparks.object);
    this.streaks = new StreakBurst({
      count: K.streaks, speed: K.streakSpeed.map(v => v * s), life: K.streakLife,
      length: K.streakLen * s, width: K.streakW * s, drag: 0.3, spread: s * 0.2, rnd,
      dir, cone: K.cone ?? 0, baseVel: this.baseVel, intensity: 1.4
    });
    this.group.add(this.streaks.object);
    this.puffs = new PuffCloud({
      count: K.puffs, size: K.puffSize.map(v => v * s), speed: K.puffSpeed.map(v => v * s),
      duration: K.puffLife, spread: s * 0.3, rnd, stagger: 0.25, grow: 1.2,
      intensity: o.kind === 'impact' || o.kind === 'pop' ? 0.45 : (o.kind === 'reactor' ? 0.2 : 0.3)
    });
    if (dir && (o.kind === 'impact' || o.kind === 'pop')) {
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
      this.crater.scale.setScalar(s * 0.9);
      this.crater.renderOrder = 34;
      this.group.add(this.crater);
    }

    // Pool light: positioned in world space each update, never reparented
    this.light = K.light > 0 ? ctx.tryLight(0xffd2a0, 0, s * 60, o.kind === 'reactor' ? 4 : 0) : null;

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
    this.update(o.spawnTime, 0);
  }

  /** @returns {boolean} still alive */
  update(now, wallTime) {
    const age = now - this.spawnTime;
    const K = this.K, s = this.scale;
    if (age < 0) { this.group.visible = false; return true; }
    if (age > this.duration) return false;
    this.group.visible = true;
    this.group.position.copy(this.origin).addScaledVector(this.baseVel, age);

    // Flash: instant, then fast exponential decay
    const f = Math.exp(-age / K.flashFade);
    this.flash.material.opacity = Math.min(1, f * 1.2);
    this.flash.scale.setScalar(this.flashScale * (1 + age * 1.5));

    if (this.fire) {
      const life = K.fireLife;
      const p = Math.min(1, age / life);
      const r = s * K.fireR * (0.25 + easeOut(age * K.fireGrow, 1.6) * 1.0) * (1 + p * 0.6);
      this.fire.scale.setScalar(r);
      const u = this.fire.material.uniforms;
      u.uTime.value = wallTime;
      u.uHeat.value = 0.95 * Math.pow(1 - p, 1.3);
      u.uAlpha.value = Math.min(1, age / 0.12) * Math.pow(1 - p, 0.6);
      u.uIntensity.value = 0.85 * (0.5 + 0.5 * Math.exp(-age / (life * 0.35)));
      u.uTurb.value = K.fireTurb * (0.7 + p * 0.9);
      this.fire.visible = age < life;
    }
    if (this.shell) {
      const p = Math.min(1, age / K.shellLife);
      const r = s * K.shellR * (0.05 + easeOut(p * 3.2, 1.8));
      this.shell.scale.setScalar(r);
      this.shell.material.uniforms.uIntensity.value = K.shellW * 1.8 * Math.pow(1 - p, 3.5);
      this.shell.visible = p < 1;
    }
    this.sparks.setAge(age);
    this.streaks.setAge(age);
    this.puffs.setAge(age);
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
    this.totalDuration = this.driftDuration + 22;
    this.rnd = mulberry32(Math.floor(o.spawnTime * 131 + 7));
    const rnd = this.rnd;

    this.tumbleAxis = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    this.tumbleRate = 0.08 + rnd() * 0.14;
    this.ignited = false;
    this.chunks = [];
    this.chunkGroup = null;
    this.secondaries = [];   // {t, chunk, fired}
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
      const reactor = g.userData.reactorPos || new THREE.Vector3();
      reactorWorld = reactor.clone().applyQuaternion(g.quaternion).add(origin);
    }
    this.blastPos = reactorWorld;
    for (const v of this.vents) { v.torch.group.visible = false; }

    this.ctx.spawnBlast({
      position: reactorWorld, scale: this.scale, spawnTime: now, kind: 'reactor',
      baseVel: this.driftVel
    });

    // Hull breakup: chunks inherit the drift velocity plus a radial kick
    if (g) {
      this.chunkGroup = new THREE.Group();
      this.ctx.scene.add(this.chunkGroup);
      const reactorLocal = g.userData.reactorPos || new THREE.Vector3();
      const chunks = buildDebrisChunks(g, 9 + Math.floor(rnd() * 4), rnd);
      const L = g.userData.size?.length || this.scale;
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
          fired: false
        });
      }
      g.visible = false;
    }
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
    if (blastAge > 22) return false;

    if (blastAge < 0) {
      // DRIFT: coasting dark, tumbling, venting, popping
      this.hulkPos.copy(this.basePos).addScaledVector(this.driftVel, age);
      if (g) {
        g.visible = true;
        g.position.copy(this.hulkPos);
        g.rotateOnAxis(this.tumbleAxis, this.tumbleRate * delta);
        for (const v of this.vents) {
          const a = age - v.start;
          const on = a > 0 ? Math.min(1, a / 0.6) * (0.55 + 0.45 * Math.sin(wallTime * 1.7 + v.phase)) : 0;
          v.torch.update(on, wallTime, 0.85);
        }
      }
      for (const pop of this.pops) {
        if (pop.fired || pop.t > age) continue;
        pop.fired = true;
        if (age - pop.t < 0.5 && g) {
          const world = pop.local.clone().applyQuaternion(g.quaternion).add(this.hulkPos);
          const dir = pop.local.clone().setZ(0).normalize().applyQuaternion(g.quaternion);
          this.ctx.spawnBlast({ position: world, scale: this.scale * 0.35, spawnTime: this.spawnTime + pop.t, kind: 'pop', dir, baseVel: this.driftVel });
        }
      }
      return true;
    }

    if (!this.ignited) this._ignite();

    // Chunks: ballistic, tumbling, cooling from orange-hot to dark
    const fadeStart = 17;
    const fade = blastAge > fadeStart ? 1 - (blastAge - fadeStart) / 5 : 1;
    const q = new THREE.Quaternion();
    for (const c of this.chunks) {
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
    for (const s of this.secondaries) {
      if (s.fired || s.t > blastAge) continue;
      s.fired = true;
      if (blastAge - s.t < 0.5) {
        const pos = s.chunk.p0.clone().addScaledVector(s.chunk.vel, blastAge);
        this.ctx.spawnBlast({ position: pos, scale: this.scale * (0.3 + this.rnd() * 0.3), spawnTime: this.spawnTime + this.driftDuration + s.t, kind: 'secondary', baseVel: s.chunk.vel });
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
      for (const c of this.chunks) {
        c.mesh.geometry.dispose();
        const mats = Array.isArray(c.mesh.material) ? c.mesh.material : [c.mesh.material];
        for (const m of mats) m.dispose();
      }
    }
    if (this.shipGroup) this.shipGroup.visible = false;
  }
}
