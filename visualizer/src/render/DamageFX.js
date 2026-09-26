import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import {
  buildDamageTimeline, shipStateAt, marksUpTo, damageStateAt, ventTau, localBasis,
  hash01, SPALL_LIFE
} from '../DamageTimeline.js';
import { getNoiseTexture, getPuffTexture, getGlowTexture, getStarTexture, mulberry32 } from './Textures.js';
import { buildDebrisChunks } from './Hulls.js';

/**
 * Persistent battle damage: everything that stays on or around a hull
 * after a hit lands (the instant impact flash belongs to the projectile
 * renderer).
 *
 *  - glowing craters that cool from white-orange to black scorch
 *  - instanced spall chunks thrown off the impact, cooling as they tumble
 *  - venting jets from breaches (atmosphere / fuel / coolant) that bleed down
 *  - electrical sparks and arcs, failing lights on a badly hurt hull
 *  - a smoke-and-ember trail behind heavily damaged ships
 *  - scorched, buckled radiator wings
 *  - death-spiral secondaries walking along the hull toward the reactor
 *  - hull breakup helpers for ShipDestruction: 2-3 big sections with
 *    glowing torn edges, an instanced debris cloud
 *
 * Every visual is a pure function of battle time: the DamageTimeline says
 * what exists at t, shaders hide what hasn't happened yet, so scrubbing
 * backward removes damage and seeking forward lands mid-sequence.
 */

const PREMULT = {
  transparent: true, depthWrite: false, blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendEquation: THREE.AddEquation
};
const ADDITIVE = {
  transparent: true, depthWrite: false, blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
};

const HEAT_GLSL = /* glsl */`
  vec3 heatRamp(float h) {
    vec3 c1 = vec3(0.55, 0.05, 0.01);
    vec3 c2 = vec3(1.7, 0.45, 0.07);
    vec3 c3 = vec3(2.5, 1.5, 0.6);
    vec3 c4 = vec3(3.2, 2.9, 2.5);
    if (h < 0.2) return c1 * (h / 0.2);
    if (h < 0.55) return mix(c1, c2, (h - 0.2) / 0.35);
    if (h < 1.0) return mix(c2, c3, (h - 0.55) / 0.45);
    return mix(c3, c4, clamp((h - 1.0) / 0.5, 0.0, 1.0));
  }
  float craterHeatF(float age, float sev) {
    float k = 0.75 + 0.35 * min(sev, 1.6);
    float h = 1.15 * exp(-age / (1.4 * k)) + 0.45 * exp(-age / (8.0 * k));
    h *= min(1.0, age / 0.05 + 0.2);
    return (age < 0.0 || age > 24.0 * k) ? 0.0 : h;
  }
  float dmgHash(float n) { return fract(sin(n * 12.9898 + 4.1414) * 43758.5453); }
`;

const VENT_TYPE = { atmo: 0, fuel: 1, coolant: 2, fire: 3 };
const CRATER_CAP = 192;
const VENT_CAP = 40;
const SPARK_EMITTERS = 8;
const SPARKS_PER = 14;
const ARCS = 4;
const ARC_SEGS = 7;
const SMOKE_DT = 0.1;
const SMOKE_LIFE = 6.5;
const SMOKE_CAP = 72;
const SPALL_BLOCK = 32;
const SPALL_BLOCKS = 128;

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _m = new THREE.Matrix4();
const _s = new THREE.Vector3();
const _Z = new THREE.Vector3(0, 0, 1);
const _Y = new THREE.Vector3(0, 1, 0);
const _basis = { x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0] };

// ---------------------------------------------------------------------------
// Materials (shared across ships: one program each, compiled at warm-up)
// ---------------------------------------------------------------------------
function craterMaterial(uNow) {
  return new THREE.ShaderMaterial({
    uniforms: { uNow, uNoise: { value: getNoiseTexture() } },
    vertexShader: /* glsl */`
      attribute vec4 aCrater;   // t0, severity, seed, breached
      uniform float uNow;
      varying vec2 vUv;
      varying vec4 vInfo;
      varying float vAge;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vUv = uv;
        vInfo = aCrater;
        vAge = uNow - aCrater.x;
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        if (vAge < 0.0) gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      varying vec2 vUv;
      varying vec4 vInfo;
      varying float vAge;
      #include <logdepthbuf_pars_fragment>
      ${HEAT_GLSL}
      void main() {
        #include <logdepthbuf_fragment>
        vec2 p = vUv * 2.0 - 1.0;
        float seed = vInfo.z;
        vec4 n = texture2D(uNoise, vUv * 0.55 + vec2(seed * 0.137, seed * 0.291));
        vec4 m = texture2D(uNoise, vUv * 1.7 + vec2(seed * 0.071, -seed * 0.113));
        float r = length(p);
        float ang = atan(p.y, p.x);
        // splash lobes: radius perturbed per direction (tileable noise wraps at +-pi)
        float lobes = texture2D(uNoise, vec2(ang * 0.15915 + seed * 0.131, seed * 0.293)).r;
        float lobes2 = texture2D(uNoise, vec2(ang * 0.31831 - seed * 0.07, 0.5 + seed * 0.11)).g;
        float rr = r / (0.45 + 0.35 * n.r + 0.45 * lobes * lobes2);
        if (rr > 1.0) discard;
        float breached = step(0.5, vInfo.w);
        // soot flung out in radial streaks, ragged at the edge
        float streak = 0.55 + 0.45 * sin(ang * (5.0 + floor(seed * 5.0)) + m.g * 7.0);
        float soot = (1.0 - smoothstep(0.05, 1.0, rr)) * (0.6 + 0.4 * m.b) * mix(1.0, streak, smoothstep(0.3, 0.85, rr));
        float bowl = 1.0 - smoothstep(0.2, 0.42, rr);
        float hole = breached * (1.0 - smoothstep(0.14, 0.22, rr + (n.g - 0.5) * 0.12));
        float a = clamp(soot * 0.92 + bowl * 0.25 + hole, 0.0, 0.98);
        // molten splash that shrinks back into the bowl as it cools, a
        // glowing lip, and a white-hot rim round a breach
        float heat = craterHeatF(vAge, vInfo.y);
        float spread = 0.3 + 0.45 * clamp(heat, 0.0, 1.2);
        float core = 1.0 - smoothstep(0.0, spread, rr);
        float lip = exp(-pow((rr - 0.38) / 0.1, 2.0)) * smoothstep(0.25, 0.7, m.r + 0.25 * m.g);
        float rim = breached * exp(-pow((rr - 0.22) / 0.06, 2.0));
        float mask = max(core * (0.55 + 0.7 * m.r) * (1.0 - hole * 0.85), max(lip * (0.5 + 0.9 * m.r), rim * 1.3));
        float h = heat * mask;
        vec3 glow = heatRamp(h) * smoothstep(0.015, 0.1, h);
        // gouged bare metal round the bowl catches a little light
        float gouge = exp(-pow((rr - 0.5) / 0.12, 2.0)) * (0.4 + 0.6 * m.g) * (1.0 - hole);
        glow += vec3(0.05, 0.05, 0.055) * gouge;
        gl_FragColor = vec4(glow, a * (1.0 - 0.5 * gouge));
      }`,
    side: THREE.DoubleSide,
    ...PREMULT
  });
}

function ventMaterial(uNow, uTime) {
  return new THREE.ShaderMaterial({
    uniforms: { uNow, uTime, uNoise: { value: getNoiseTexture() }, uIntensity: { value: 1.1 } },
    vertexShader: /* glsl */`
      attribute vec4 aVent;    // t0, size, tau, type
      attribute vec4 aVent2;   // seed
      uniform float uNow;
      uniform float uTime;
      varying float vAlong;
      varying float vStrength;
      varying float vType;
      varying float vSeed;
      varying float vEdge;
      varying vec2 vUv;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        float age = uNow - aVent.x;
        float s = aVent.y * min(1.0, max(age, 0.0) / 0.35) * exp(-max(age, 0.0) / aVent.z);
        if (age < 0.0 || s < 0.035) s = 0.0;
        float seed = aVent2.x;
        // pressure sputter: turbulence on wall-clock time
        float sp = 0.8 + 0.2 * sin(uTime * (6.0 + seed * 5.0) + seed * 40.0) * sin(uTime * 2.7 + seed * 13.0);
        vStrength = s * sp;
        vec3 p = position;
        float ss = sqrt(s);
        p.y *= 0.3 + 0.7 * ss;
        p.xz *= 0.5 + 0.5 * ss;
        p.x += sin(uTime * 2.1 + seed * 9.0 + position.y * 2.5) * 0.07 * position.y;
        p.z += cos(uTime * 1.7 + seed * 5.0 + position.y * 2.0) * 0.05 * position.y;
        vAlong = position.y;
        vType = aVent.w;
        vSeed = seed;
        vUv = uv;
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
        vec3 nv = normalize(mat3(modelViewMatrix) * mat3(instanceMatrix) * normal);
        vEdge = abs(dot(nv, normalize(-mv.xyz)));
        gl_Position = projectionMatrix * mv;
        if (s <= 0.0) gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uTime;
      uniform float uIntensity;
      varying float vAlong;
      varying float vStrength;
      varying float vType;
      varying float vSeed;
      varying float vEdge;
      varying vec2 vUv;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        float t = vAlong;
        vec4 n = texture2D(uNoise, vec2(vUv.x * 2.0 + vSeed, t * 0.8 - uTime * 1.1 + vSeed * 3.0));
        vec4 n2 = texture2D(uNoise, vec2(vUv.x * 3.0 - vSeed, t * 1.6 - uTime * 1.9));
        float body = pow(1.0 - t, 2.2) * smoothstep(0.0, 0.04, t);
        float edge = pow(vEdge, 2.6);
        // filaments: the jet tears into streaks as it expands
        float wisps = smoothstep(0.25, 0.75, n.r * 0.65 + n2.g * 0.6 - t * 0.25) * 1.6 + 0.1;
        vec3 col = vType < 0.5 ? vec3(0.78, 0.86, 1.0)
                 : vType < 1.5 ? vec3(0.6, 0.92, 1.0)
                 : vType < 2.5 ? vec3(1.0, 0.86, 0.6)
                 : vec3(1.6, 0.62, 0.18);
        col += vec3(0.8, 0.8, 0.85) * exp(-t * 14.0);    // bright throat
        float a = body * edge * wisps * vStrength;
        gl_FragColor = vec4(col * a * uIntensity, 1.0);
      }`,
    side: THREE.DoubleSide,
    ...ADDITIVE
  });
}

function sparkMaterial(uNow) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNow, uLevel: { value: 0 }, uSize: { value: 0.1 }, uSpeed: { value: 1 },
      uMap: { value: getStarTexture() }
    },
    vertexShader: /* glsl */`
      attribute vec3 aNorm;
      attribute vec4 aP;       // seed, emitter, period, weight
      uniform float uNow;
      uniform float uLevel;
      uniform float uSize;
      uniform float uSpeed;
      varying vec3 vColor;
      varying float vAlpha;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      ${HEAT_GLSL}
      void main() {
        float P = aP.z;
        float tt = uNow + aP.y * 0.618 * P;
        float cyc = floor(tt / P);
        float a = tt - cyc * P;
        float gate = step(dmgHash(aP.y * 17.0 + cyc * 3.7), uLevel * aP.w * 0.9);
        float life = 0.3 + 0.4 * dmgHash(aP.x * 91.0 + cyc);
        float la = a - dmgHash(aP.x * 13.0 + cyc * 1.3) * 0.12;
        vec3 rnd = vec3(dmgHash(aP.x * 7.1 + cyc * 1.7), dmgHash(aP.x * 3.3 + cyc * 2.9), dmgHash(aP.x * 5.9 + cyc * 4.3)) * 2.0 - 1.0;
        vec3 dir = normalize(aNorm * 1.15 + rnd);
        float speed = uSpeed * (0.25 + dmgHash(aP.x * 2.3 + cyc * 5.1));
        float fl = max(la, 0.0);
        vec3 p = position + dir * speed * (1.0 - exp(-3.0 * fl)) / 3.0;
        float f = clamp(la / life, 0.0, 1.0);
        vAlpha = gate * step(0.0, la) * step(la, life) * (1.0 - f);
        vColor = mix(vec3(2.0, 2.4, 3.4), vec3(2.2, 0.95, 0.3), smoothstep(0.05, 0.5, f));
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = max(1.5, uSize * (1.0 - 0.5 * f) * (400.0 / max(1.0, -mv.z)));
        if (vAlpha <= 0.0) gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uMap;
      varying vec3 vColor;
      varying float vAlpha;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        float a = texture2D(uMap, gl_PointCoord).a * vAlpha;
        gl_FragColor = vec4(vColor * a, 1.0);
      }`,
    ...ADDITIVE
  });
}

function smokeMaterial(uNow) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNow, uLife: { value: SMOKE_LIFE },
      uPuff: { value: getPuffTexture(1) }, uGlow: { value: getGlowTexture() }
    },
    vertexShader: /* glsl */`
      attribute vec4 aS0;   // p0.xyz, t0
      attribute vec4 aS1;   // vel.xyz, size
      attribute vec4 aS2;   // seed, ember, level, rot
      uniform float uNow;
      uniform float uLife;
      varying vec2 vUv;
      varying float vA;
      varying float vKind;
      varying float vSeed;
      varying float vAge;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        float age = uNow - aS0.w;
        float f = age / uLife;
        float ember = aS2.y;
        vec3 c = aS0.xyz + aS1.xyz * age;
        float size = aS1.w * (ember > 0.5 ? (1.0 - 0.7 * f) : (0.4 + 1.7 * sqrt(max(f, 0.0))));
        vec4 mv = modelViewMatrix * vec4(c, 1.0);
        float rot = aS2.w + age * (aS2.x - 0.5) * 0.5;
        vec2 q = position.xy;
        q = vec2(q.x * cos(rot) - q.y * sin(rot), q.x * sin(rot) + q.y * cos(rot));
        mv.xy += q * size;
        gl_Position = projectionMatrix * mv;
        vUv = uv;
        vKind = ember;
        vSeed = aS2.x;
        vAge = age;
        float live = (age < 0.0 || f > 1.0) ? 0.0 : 1.0;
        vA = live * aS2.z * (ember > 0.5 ? exp(-age / 0.9) : smoothstep(0.0, 0.06, f) * pow(1.0 - f, 1.4));
        if (vA <= 0.002) gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uPuff;
      uniform sampler2D uGlow;
      varying vec2 vUv;
      varying float vA;
      varying float vKind;
      varying float vSeed;
      varying float vAge;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        if (vKind > 0.5) {
          float g = texture2D(uGlow, vUv).a * vA;
          gl_FragColor = vec4(vec3(2.4, 0.95, 0.25) * g, 0.0);
          return;
        }
        float a = texture2D(uPuff, vUv).a * vA;
        vec3 col = vec3(0.3, 0.29, 0.28) * (0.7 + 0.5 * vSeed);
        col += vec3(1.1, 0.4, 0.08) * exp(-vAge * 2.0);   // fire-lit at the source
        gl_FragColor = vec4(col * a, a * 0.75);
      }`,
    ...PREMULT
  });
}

function debrisMaterial(uNow) {
  const mat = new THREE.MeshStandardMaterial({
    color: 0x7d8189, roughness: 0.5, metalness: 0.8, flatShading: true, envMapIntensity: 0.9
  });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uNow = uNow;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec4 aD0;   // p0.xyz (world), t0
        attribute vec4 aD1;   // vel.xyz (world units/s), life
        attribute vec4 aD2;   // spin axis.xyz, rate
        attribute vec4 aD3;   // shard scale.xyz, heat
        uniform float uNow;
        varying float vDebrisHeat;
        vec3 dmgRot(vec3 v, vec3 k, float a) {
          float c = cos(a), s = sin(a);
          return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
        }`)
      .replace('#include <begin_vertex>', `
        float dAge = uNow - aD0.w;
        float dOn = step(0.0, dAge) * step(dAge, aD1.w);
        float dShrink = 1.0 - smoothstep(0.7, 1.0, dAge / max(aD1.w, 0.01));
        vec3 transformed = dmgRot(position * aD3.xyz, aD2.xyz, aD2.w * dAge) * (dOn * dShrink)
          + aD0.xyz + aD1.xyz * dAge;
        vDebrisHeat = dOn * aD3.w * exp(-max(dAge, 0.0) / (0.4 + 1.1 * aD3.w));`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        varying float vDebrisHeat;`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance += vec3(2.4, 0.85, 0.2) * vDebrisHeat * vDebrisHeat + vec3(0.7, 0.1, 0.02) * vDebrisHeat;`);
  };
  mat.customProgramCacheKey = () => 'damage-debris-v1';
  return mat;
}

/** Additive glow on torn hull edges: same geometry, drawn over the section. */
function edgeGlowMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() }, uCutLo: { value: -1e6 }, uCutHi: { value: 1e6 },
      uEdgeW: { value: 0.3 }, uHeat: { value: 0 }, uSeed: { value: 0 }, uFade: { value: 1 }
    },
    vertexShader: /* glsl */`
      varying vec3 vLocal;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vLocal = position;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uCutLo;
      uniform float uCutHi;
      uniform float uEdgeW;
      uniform float uHeat;
      uniform float uSeed;
      uniform float uFade;
      varying vec3 vLocal;
      #include <logdepthbuf_pars_fragment>
      ${HEAT_GLSL}
      void main() {
        #include <logdepthbuf_fragment>
        float d = min(abs(vLocal.z - uCutLo), abs(vLocal.z - uCutHi));
        float n = texture2D(uNoise, vLocal.xy * 0.45 + vec2(vLocal.z * 0.23, uSeed)).r;
        float n2 = texture2D(uNoise, vLocal.zx * 0.8 + vec2(uSeed * 1.7, 0.3)).g;
        float g = exp(-d / (uEdgeW * (0.45 + 1.1 * n)));
        float h = uHeat * g * (0.35 + 1.3 * n2);
        if (h < 0.02) discard;
        gl_FragColor = vec4(heatRamp(h) * smoothstep(0.02, 0.12, h) * uFade, 1.0);
      }`,
    depthFunc: THREE.LessEqualDepth,
    ...ADDITIVE
  });
}

/** Irregular low-poly shard (indexed so vertex jitter keeps faces closed). */
function chunkGeometry() {
  let g = new THREE.IcosahedronGeometry(1, 0);
  g.deleteAttribute('normal');
  g.deleteAttribute('uv');
  g = BufferGeometryUtils.mergeVertices(g);
  const rnd = mulberry32(4242);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const k = 0.6 + 0.7 * rnd();
    p.setXYZ(i, p.getX(i) * k, p.getY(i) * (0.5 + 0.6 * rnd()), p.getZ(i) * k);
  }
  g.computeVertexNormals();
  return g;
}

function billboardGeometry(count, attrs) {
  const base = new THREE.PlaneGeometry(1, 1);
  const g = new THREE.InstancedBufferGeometry();
  g.index = base.index;
  g.setAttribute('position', base.attributes.position);
  g.setAttribute('uv', base.attributes.uv);
  for (const name of attrs) g.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4));
  g.instanceCount = count;
  return g;
}

// ---------------------------------------------------------------------------
// Instanced debris: one InstancedMesh, motion evaluated on the GPU
// ---------------------------------------------------------------------------
export class DebrisSet {
  constructor(fx, capacity) {
    this.capacity = capacity;
    const base = fx.chunkGeom;
    const g = new THREE.BufferGeometry();
    g.index = base.index;
    g.setAttribute('position', base.attributes.position);
    g.setAttribute('normal', base.attributes.normal);
    this.attrs = {};
    for (const name of ['aD0', 'aD1', 'aD2', 'aD3']) {
      const arr = new Float32Array(capacity * 4);
      const a = new THREE.InstancedBufferAttribute(arr, 4);
      a.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute(name, a);
      this.attrs[name] = a;
    }
    // life 0 everywhere = hidden until written
    this.mesh = new THREE.InstancedMesh(g, fx.debrisMat, capacity);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 5;
    this.dirtyLo = Infinity;
    this.dirtyHi = -1;
  }

  /** Write instance i: world p0 at spawn time t0, velocity, spin, shard size, heat. */
  set(i, p0, t0, vel, life, axis, rate, sx, sy, sz, heat) {
    const A = this.attrs;
    let o = i * 4;
    A.aD0.array[o] = p0.x; A.aD0.array[o + 1] = p0.y; A.aD0.array[o + 2] = p0.z; A.aD0.array[o + 3] = t0;
    A.aD1.array[o] = vel.x; A.aD1.array[o + 1] = vel.y; A.aD1.array[o + 2] = vel.z; A.aD1.array[o + 3] = life;
    A.aD2.array[o] = axis.x; A.aD2.array[o + 1] = axis.y; A.aD2.array[o + 2] = axis.z; A.aD2.array[o + 3] = rate;
    A.aD3.array[o] = sx; A.aD3.array[o + 1] = sy; A.aD3.array[o + 2] = sz; A.aD3.array[o + 3] = heat;
    if (i < this.dirtyLo) this.dirtyLo = i;
    if (i > this.dirtyHi) this.dirtyHi = i;
  }

  hide(i) {
    this.attrs.aD1.array[i * 4 + 3] = 0;
    if (i < this.dirtyLo) this.dirtyLo = i;
    if (i > this.dirtyHi) this.dirtyHi = i;
  }

  flush() {
    if (this.dirtyHi < 0) return;
    for (const a of Object.values(this.attrs)) {
      if (a.addUpdateRange) a.addUpdateRange(this.dirtyLo * 4, (this.dirtyHi - this.dirtyLo + 1) * 4);
      a.needsUpdate = true;
    }
    this.dirtyLo = Infinity;
    this.dirtyHi = -1;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.dispose?.();
  }
}

// ---------------------------------------------------------------------------
// Per-ship view: craters, vents, sparks, arcs, smoke, flicker, radiators
// ---------------------------------------------------------------------------
class ShipView {
  constructor(fx, sd, group) {
    this.fx = fx;
    this.sd = sd;
    this.id = sd.id;
    this.group = group;
    const ud = group.userData;
    this.L = ud.size?.length || 8;
    this.W = ud.size?.width || 1.8;
    this.seed = sd.seed;
    this.resolved = 0;              // marks resolved so far (in time order)
    this.state = {};
    this.ventMarks = sd.marks.filter(m => m.vent);
    this._buildPartIndex();

    // Craters: instanced decals in ship-local space
    const cap = Math.max(1, Math.min(CRATER_CAP, sd.marks.length));
    this.craterCap = cap;
    const cg = new THREE.InstancedBufferGeometry();
    const plane = fx.planeGeom;
    cg.index = plane.index;
    cg.setAttribute('position', plane.attributes.position);
    cg.setAttribute('uv', plane.attributes.uv);
    const ca = new Float32Array(cap * 4);
    for (let i = 0; i < cap; i++) ca[i * 4] = 1e9;
    this.craterAttr = new THREE.InstancedBufferAttribute(ca, 4);
    cg.setAttribute('aCrater', this.craterAttr);
    this.craters = new THREE.InstancedMesh(cg, fx.craterMat, cap);
    this.craters.frustumCulled = false;
    this.craters.renderOrder = 6;
    this.craters.count = 0;
    this.craters.userData.damageFx = true;
    group.add(this.craters);

    // Vents: instanced cones for every breach that vents
    if (this.ventMarks.length) {
      const vcap = Math.min(VENT_CAP, this.ventMarks.length);
      this.ventCap = vcap;
      const vg = new THREE.InstancedBufferGeometry();
      const cone = fx.coneGeom;
      vg.index = cone.index;
      vg.setAttribute('position', cone.attributes.position);
      vg.setAttribute('normal', cone.attributes.normal);
      vg.setAttribute('uv', cone.attributes.uv);
      const va = new Float32Array(vcap * 4);
      for (let i = 0; i < vcap; i++) { va[i * 4] = 1e9; va[i * 4 + 2] = 1; }
      this.ventAttr = new THREE.InstancedBufferAttribute(va, 4);
      this.ventAttr2 = new THREE.InstancedBufferAttribute(new Float32Array(vcap * 4), 4);
      vg.setAttribute('aVent', this.ventAttr);
      vg.setAttribute('aVent2', this.ventAttr2);
      this.vents = new THREE.InstancedMesh(vg, fx.ventMat, vcap);
      this.vents.frustumCulled = false;
      this.vents.renderOrder = 22;
      this.vents.count = 0;
      this.vents.userData.damageFx = true;
      group.add(this.vents);
    }
    this.sparks = null;
    this.arcs = null;
    this.smoke = null;
    this.emitterCount = -1;
    this.arcCycle = -1;
    this.lastNow = -Infinity;
  }

  _buildPartIndex() {
    const parts = this.group.userData.parts || [];
    this.parts = [];
    this.bbox = new THREE.Box3();
    for (const p of parts) {
      const pos = p.geom?.attributes?.position;
      if (!pos) continue;
      if (!p.geom.boundingBox) p.geom.computeBoundingBox();
      this.parts.push(p);
      this.bbox.union(p.geom.boundingBox);
    }
    if (!this.parts.length) {
      this.bbox.min.set(-this.W / 2, -this.W / 2, -this.L / 2);
      this.bbox.max.set(this.W / 2, this.W / 2, this.L / 2);
    }
  }

  /** Nearest surface hit along a ship-local ray, or null. */
  _cast(o, d) {
    const fx = this.fx;
    const ray = fx.raycaster.ray;
    ray.origin.copy(o);
    ray.direction.copy(d);
    if (!this.parts.length) return this._castMeshes(o, d);
    const cands = fx._cands;
    cands.length = 0;
    for (let i = 0; i < this.parts.length; i++) {
      const hit = ray.intersectBox(this.parts[i].geom.boundingBox, _v2);
      if (hit) cands.push(i, _v2.distanceTo(o));
    }
    // insertion sort by distance (pairs)
    for (let i = 2; i < cands.length; i += 2) {
      const pi = cands[i], pd = cands[i + 1];
      let j = i - 2;
      while (j >= 0 && cands[j + 1] > pd) { cands[j + 2] = cands[j]; cands[j + 3] = cands[j + 1]; j -= 2; }
      cands[j + 2] = pi; cands[j + 3] = pd;
    }
    let best = null;
    const hits = fx._hits;
    for (let i = 0; i < cands.length; i += 2) {
      if (best && cands[i + 1] > best.distance) break;
      hits.length = 0;
      fx.probe.geometry = this.parts[cands[i]].geom;
      fx.probe.raycast(fx.raycaster, hits);
      for (const h of hits) if (!best || h.distance < best.distance) best = h;
    }
    if (!best) return null;
    const n = best.face ? best.face.normal.clone() : d.clone().negate();
    if (n.dot(d) > 0) n.negate();
    return { p: best.point.clone(), n };
  }

  _castMeshes(o, d) {
    const g = this.group;
    g.updateWorldMatrix(true, true);
    const meshes = [];
    g.traverse(c => { if (c.isMesh && !c.isInstancedMesh && !c.userData.damageFx && c.parent === g) meshes.push(c); });
    const rc = this.fx.raycaster;
    rc.ray.origin.copy(o).applyMatrix4(g.matrixWorld);
    rc.ray.direction.copy(d).transformDirection(g.matrixWorld);
    const hits = rc.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const h = hits[0];
    const p = g.worldToLocal(h.point.clone());
    const n = h.face ? h.face.normal.clone() : d.clone().negate();
    if (n.dot(d) > 0) n.negate();
    return { p, n };
  }

  /** Surface point + normal (ship-local) for a mark; cached on the mark. */
  resolve(m) {
    if (m._res && m._resGroup === this.group) return m._res;
    const bb = this.bbox;
    const ca = Math.cos(m.aim.azimuth), sa = Math.sin(m.aim.azimuth);
    const hx = Math.max(Math.abs(bb.min.x), bb.max.x), hy = Math.max(Math.abs(bb.min.y), bb.max.y);
    const o = new THREE.Vector3(), d = new THREE.Vector3();
    let hit = null;
    if (m.aim.end !== 0) {
      for (const shrink of [1, 0.55, 0.2]) {
        const r = m.aim.radial * shrink;
        o.set(ca * r * hx, sa * r * hy, m.aim.end > 0 ? bb.max.z + 1 : bb.min.z - 1);
        d.set(0, 0, -m.aim.end);
        hit = this._cast(o, d);
        if (hit) break;
      }
    } else {
      const len = bb.max.z - bb.min.z;
      const mid = (bb.max.z + bb.min.z) / 2;
      for (const shrink of [1, 0.7, 0.4, 0]) {
        const z = mid + m.aim.axial * len * shrink;
        o.set(ca * (hx + 1), sa * (hy + 1), z);
        d.set(-ca, -sa, 0);
        hit = this._cast(o, d);
        if (hit) break;
      }
    }
    if (!hit) {
      // fall back to the bounding box skin
      hit = { p: new THREE.Vector3(ca * hx * 0.9, sa * hy * 0.9, (bb.max.z + bb.min.z) / 2 + m.aim.axial * (bb.max.z - bb.min.z)),
        n: new THREE.Vector3(ca, sa, 0) };
    }
    m._res = hit;
    m._resGroup = this.group;
    return hit;
  }

  /** Resolve (and write instance data for) every mark up to time t. */
  resolveUpTo(t) {
    const marks = this.sd.marks;
    let wroteCrater = false, wroteVent = false;
    while (this.resolved < marks.length && marks[this.resolved].t <= t) {
      const m = marks[this.resolved++];
      const res = this.resolve(m);
      this._writeCrater(m, res);
      wroteCrater = true;
      if (m.vent && this.vents) wroteVent = this._writeVent(m, res) || wroteVent;
    }
    if (wroteCrater) {
      this.craters.instanceMatrix.needsUpdate = true;
      this.craterAttr.needsUpdate = true;
    }
    if (wroteVent) {
      this.vents.instanceMatrix.needsUpdate = true;
      this.ventAttr.needsUpdate = true;
      this.ventAttr2.needsUpdate = true;
    }
  }

  _writeCrater(m, res) {
    const slot = m.index % this.craterCap;
    const W = this.W;
    const radius = W * (m.kind === 'pop' ? 0.1 : m.kind === 'torpedo' ? 0.09 : 0.06) * (0.7 + 0.9 * m.severity);
    _q.setFromUnitVectors(_Z, res.n);
    _q2.setFromAxisAngle(_Z, hash01(m.seed, 11) * Math.PI * 2);
    _q.multiply(_q2);
    _v.copy(res.p).addScaledVector(res.n, W * 0.012 + radius * 0.06);
    _s.set(radius * 2, radius * 2, 1);
    _m.compose(_v, _q, _s);
    this.craters.setMatrixAt(slot, _m);
    const a = this.craterAttr.array;
    a[slot * 4] = m.t;
    a[slot * 4 + 1] = m.severity;
    a[slot * 4 + 2] = hash01(m.seed, 12) * 97.0;
    a[slot * 4 + 3] = m.penetrated ? 1 : 0;
    this.craters.count = Math.max(this.craters.count, slot + 1);
  }

  _writeVent(m, res) {
    const vi = this.ventMarks.indexOf(m);
    if (vi < 0) return false;
    const slot = vi % this.ventCap;
    const W = this.W;
    // jet axis: surface normal, kicked sideways a little
    _v.set(hash01(m.seed, 21) - 0.5, hash01(m.seed, 22) - 0.5, hash01(m.seed, 23) - 0.5).multiplyScalar(0.7);
    _v2.copy(res.n).add(_v).normalize();
    _q.setFromUnitVectors(_Y, _v2);
    const len = W * (0.5 + 0.8 * m.ventSize) * (m.vent === 'fire' ? 0.6 : 1);
    const wid = len * (m.vent === 'fire' ? 0.22 : 0.17);
    _v.copy(res.p).addScaledVector(res.n, W * 0.01);
    _s.set(wid, len, wid);
    _m.compose(_v, _q, _s);
    this.vents.setMatrixAt(slot, _m);
    const a = this.ventAttr.array;
    a[slot * 4] = m.t;
    a[slot * 4 + 1] = m.ventSize;
    a[slot * 4 + 2] = ventTau(m);
    a[slot * 4 + 3] = VENT_TYPE[m.vent] ?? 0;
    this.ventAttr2.array[slot * 4] = hash01(m.seed, 24);
    this.vents.count = Math.max(this.vents.count, slot + 1);
    return true;
  }

  // -- sparks & arcs ----------------------------------------------------------
  _ensureSparks() {
    if (this.sparks) return;
    const n = SPARK_EMITTERS * SPARKS_PER;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aNorm', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aP', new THREE.BufferAttribute(new Float32Array(n * 4), 4));
    const mat = this.fx.sparkMat.clone();
    mat.uniforms.uNow = this.fx.uNow;
    mat.uniforms.uMap.value = getStarTexture();
    mat.uniforms.uSize.value = this.W * 0.05;
    mat.uniforms.uSpeed.value = this.W * 0.55;
    this.sparks = new THREE.Points(g, mat);
    this.sparks.frustumCulled = false;
    this.sparks.renderOrder = 26;
    this.sparks.userData.damageFx = true;
    this.group.add(this.sparks);

    const ag = new THREE.BufferGeometry();
    ag.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARCS * ARC_SEGS * 2 * 3), 3));
    this.arcMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(2.2, 2.7, 4.2), transparent: true, opacity: 1,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    this.arcs = new THREE.LineSegments(ag, this.arcMat);
    this.arcs.frustumCulled = false;
    this.arcs.renderOrder = 27;
    this.arcs.userData.damageFx = true;
    this.group.add(this.arcs);
    this.emitters = [];
  }

  /** Spark emitters: the most recent breaches, else the heaviest wounds. */
  _updateEmitters(count) {
    if (count === this.emitterCount) return;
    this.emitterCount = count;
    const marks = this.sd.marks;
    const pick = [];
    for (let i = count - 1; i >= 0 && pick.length < SPARK_EMITTERS; i--) if (marks[i].penetrated) pick.push(marks[i]);
    if (pick.length < SPARK_EMITTERS) {
      const rest = [];
      for (let i = 0; i < count; i++) if (!marks[i].penetrated) rest.push(marks[i]);
      rest.sort((a, b) => b.severity - a.severity || a.index - b.index);
      for (const m of rest) { if (pick.length >= SPARK_EMITTERS) break; pick.push(m); }
    }
    this.emitters = pick.map(m => this.resolve(m));
    // with no wound yet, sparks come from seeded spots on the flanks
    if (!this.emitters.length) {
      for (let e = 0; e < 3; e++) {
        const fake = {
          aim: { end: 0, azimuth: hash01(this.seed, 300, e) * Math.PI * 2, radial: 1, axial: (hash01(this.seed, 301, e) - 0.5) * 0.7 },
          _res: null
        };
        this.emitters.push(this.resolve(fake));
      }
    }
    const g = this.sparks.geometry;
    const pos = g.attributes.position.array, nor = g.attributes.aNorm.array, P = g.attributes.aP.array;
    for (let e = 0; e < SPARK_EMITTERS; e++) {
      const em = this.emitters[e % this.emitters.length];
      const active = e < this.emitters.length ? 1 : 0;
      const period = 0.9 + 2.2 * hash01(this.seed, 400, e);
      for (let k = 0; k < SPARKS_PER; k++) {
        const i = e * SPARKS_PER + k;
        pos[i * 3] = em.p.x; pos[i * 3 + 1] = em.p.y; pos[i * 3 + 2] = em.p.z;
        nor[i * 3] = em.n.x; nor[i * 3 + 1] = em.n.y; nor[i * 3 + 2] = em.n.z;
        P[i * 4] = hash01(this.seed, 500 + e, k) * 50;
        P[i * 4 + 1] = e + 1;
        P[i * 4 + 2] = period;
        P[i * 4 + 3] = active;
      }
    }
    g.attributes.position.needsUpdate = true;
    g.attributes.aNorm.needsUpdate = true;
    g.attributes.aP.needsUpdate = true;
    this.arcCycle = -1;
  }

  _updateArcs(now, level) {
    const cycle = Math.floor(now * 11);
    if (cycle === this.arcCycle && this.arcLevel === level) return;
    this.arcCycle = cycle;
    this.arcLevel = level;
    const arr = this.arcs.geometry.attributes.position.array;
    const W = this.W;
    let any = false;
    for (let a = 0; a < ARCS; a++) {
      const base = a * ARC_SEGS * 6;
      const em = this.emitters.length ? this.emitters[a % this.emitters.length] : null;
      const cyc = Math.floor(now * 11 + a * 0.37);
      const on = em && hash01(this.seed, 600 + a, cyc) < level * 0.42;
      if (!on) {
        for (let i = 0; i < ARC_SEGS * 6; i++) arr[base + i] = 0;
        continue;
      }
      any = true;
      // tangent frame at the emitter
      _v.set(1, 0, 0);
      if (Math.abs(em.n.x) > 0.8) _v.set(0, 1, 0);
      _v2.crossVectors(em.n, _v).normalize();             // t1
      _v.crossVectors(em.n, _v2).normalize();             // t2
      const th = hash01(this.seed, 700 + a, cyc) * Math.PI * 2;
      const len = W * (0.1 + 0.2 * hash01(this.seed, 800 + a, cyc));
      const dx = Math.cos(th) * len, dy = Math.sin(th) * len;
      let px = em.p.x + em.n.x * W * 0.01, py = em.p.y + em.n.y * W * 0.01, pz = em.p.z + em.n.z * W * 0.01;
      for (let s = 0; s < ARC_SEGS; s++) {
        const u = (s + 1) / ARC_SEGS;
        const bow = Math.sin(Math.PI * u) * len * 0.3;
        const jit = s === ARC_SEGS - 1 ? 0 : (hash01(this.seed, 900 + a * 16 + s, cyc) - 0.5) * len * 0.35;
        const qx = em.p.x + _v2.x * dx * u + _v.x * (dy * u + jit) + em.n.x * (W * 0.01 + bow);
        const qy = em.p.y + _v2.y * dx * u + _v.y * (dy * u + jit) + em.n.y * (W * 0.01 + bow);
        const qz = em.p.z + _v2.z * dx * u + _v.z * (dy * u + jit) + em.n.z * (W * 0.01 + bow);
        const o = base + s * 6;
        arr[o] = px; arr[o + 1] = py; arr[o + 2] = pz;
        arr[o + 3] = qx; arr[o + 4] = qy; arr[o + 5] = qz;
        px = qx; py = qy; pz = qz;
      }
    }
    this.arcs.visible = any;
    this.arcs.geometry.attributes.position.needsUpdate = true;
  }

  // -- smoke trail ------------------------------------------------------------
  _ensureSmoke() {
    if (this.smoke) return;
    const g = billboardGeometry(SMOKE_CAP, ['aS0', 'aS1', 'aS2']);
    for (let i = 0; i < SMOKE_CAP; i++) g.attributes.aS0.array[i * 4 + 3] = 1e9;
    this.smoke = new THREE.Mesh(g, this.fx.smokeMat);
    this.smoke.frustumCulled = false;
    this.smoke.renderOrder = 18;
    this.fx.scene.add(this.smoke);
    this.smokeK = new Float64Array(SMOKE_CAP).fill(NaN);
  }

  _updateSmoke(now) {
    const fx = this.fx;
    const kHi = Math.floor(now / SMOKE_DT);
    const kLo = Math.max(0, Math.ceil((now - SMOKE_LIFE) / SMOKE_DT));
    if (!this.smoke) {
      // only build the trail once this ship actually smokes
      if (this.state.smoke <= 0) return;
      this._ensureSmoke();
    }
    const g = this.smoke.geometry;
    const S0 = g.attributes.aS0.array, S1 = g.attributes.aS1.array, S2 = g.attributes.aS2.array;
    let dirty = false;
    for (let k = kLo; k <= kHi; k++) {
      const slot = k % SMOKE_CAP;
      if (this.smokeK[slot] === k) continue;
      this.smokeK[slot] = k;
      dirty = true;
      const o = slot * 4;
      S0[o + 3] = 1e9;               // hidden unless emitted below
      const tk = k * SMOKE_DT;
      const ds = damageStateAt(fx.timeline, this.id, tk, fx._ds);
      if (ds.destroyed || ds.smoke <= 0) continue;
      const u = hash01(this.seed, k, 31);
      const ember = hash01(this.seed, k, 32) < 0.28;
      if (!ember && u > 0.35 + 0.65 * ds.smoke) continue;
      const st = shipStateAt(fx.timeline.trace, this.id, tk, fx._st);
      if (!st) continue;
      // source: a recent breach if any, else a seeded flank spot
      const n = ds.count;
      let res = null;
      if (n > 0) {
        let pickIdx = -1;
        const h = hash01(this.seed, k, 33);
        for (let i = n - 1, seen = 0; i >= 0 && seen < 6; i--) {
          if (this.sd.marks[i].penetrated) { pickIdx = i; if (h < 0.55) break; seen++; }
        }
        if (pickIdx < 0) pickIdx = Math.floor(h * n);
        res = this.resolve(this.sd.marks[pickIdx]);
      } else {
        res = this.emitters?.[0] || null;
      }
      const B = localBasis(st.fwd, _basis);
      const lp = res ? res.p : _v.set(0, 0, 0);
      const ln = res ? res.n : _v2.set(1, 0, 0);
      const SC = fx.SCALE;
      const wx = st.pos[0] * SC + B.x[0] * lp.x + B.y[0] * lp.y + B.z[0] * lp.z;
      const wy = st.pos[1] * SC + B.x[1] * lp.x + B.y[1] * lp.y + B.z[1] * lp.z;
      const wz = st.pos[2] * SC + B.x[2] * lp.x + B.y[2] * lp.y + B.z[2] * lp.z;
      const nx = B.x[0] * ln.x + B.y[0] * ln.y + B.z[0] * ln.z;
      const ny = B.x[1] * ln.x + B.y[1] * ln.y + B.z[1] * ln.z;
      const nz = B.x[2] * ln.x + B.y[2] * ln.y + B.z[2] * ln.z;
      // left behind by the ship (a third of its velocity), puffed out of the wound
      const lag = 0.7;
      const out = this.W * (0.12 + 0.3 * hash01(this.seed, k, 34));
      const jx = (hash01(this.seed, k, 35) - 0.5) * this.W * 0.2;
      const jy = (hash01(this.seed, k, 36) - 0.5) * this.W * 0.2;
      const jz = (hash01(this.seed, k, 37) - 0.5) * this.W * 0.2;
      S0[o] = wx; S0[o + 1] = wy; S0[o + 2] = wz; S0[o + 3] = tk;
      S1[o] = st.vel[0] * SC * lag + nx * out + jx;
      S1[o + 1] = st.vel[1] * SC * lag + ny * out + jy;
      S1[o + 2] = st.vel[2] * SC * lag + nz * out + jz;
      S1[o + 3] = ember ? this.W * (0.05 + 0.05 * u) : this.W * (0.2 + 0.25 * u) * (0.6 + 0.4 * ds.smoke);
      S2[o] = hash01(this.seed, k, 38);
      S2[o + 1] = ember ? 1 : 0;
      S2[o + 2] = ember ? 1 : 0.35 + 0.5 * ds.smoke;
      S2[o + 3] = hash01(this.seed, k, 39) * 6.283;
    }
    if (dirty) {
      g.attributes.aS0.needsUpdate = true;
      g.attributes.aS1.needsUpdate = true;
      g.attributes.aS2.needsUpdate = true;
    }
  }

  // -- flicker / radiators ----------------------------------------------------
  _applyFlicker(now, level) {
    const ud = this.group.userData;
    if (level <= 0) return;
    const blink = hash01(this.seed, Math.floor(now * 7), 41);
    let mul = 1 - 0.5 * level;
    if (blink < 0.3 * level) mul *= 0.12;
    else if (blink > 1 - 0.1 * level) mul *= 1.5;
    if (ud.materials?.window) ud.materials.window.emissiveIntensity *= mul;
    const blink2 = hash01(this.seed, Math.floor(now * 5), 42);
    for (const nl of ud.navLights || []) {
      if (blink2 < 0.45 * level) nl.sprite.material.opacity *= 0.1;
    }
  }

  _applyRadiators(level) {
    const ud = this.group.userData;
    const panels = ud.radiators || [];
    if (!panels.length) return;
    if (!this.radDamage) {
      if (level <= 0) return;
      const orig = ud.radiatorMaterials?.[0];
      if (!orig) return;
      const clone = orig.clone();
      this.radDamage = { orig, clone, meshes: [], level: -1, bent: new Array(panels.length).fill(false) };
      panels.forEach((panel, i) => {
        panel.traverse(c => { if (c.isMesh && c.material === orig) this.radDamage.meshes.push({ mesh: c, panel: i }); });
      });
    }
    const R = this.radDamage;
    const k = Math.min(1, level);
    if (Math.abs(level - R.level) > 1e-3) {
      // which wings are hit and how far they buckle: a pure function of level
      R.level = level;
      for (let i = 0; i < panels.length; i++) {
        R.bent[i] = hash01(this.seed, 1000 + i) < level * 0.9;
        const h = hash01(this.seed, 1100 + i);
        panels[i].rotation.x = R.bent[i] ? (h < 0.5 ? -1 : 1) * (0.12 + 0.3 * h) * level : 0;
      }
      for (const e of R.meshes) e.mesh.material = R.bent[e.panel] ? R.clone : R.orig;
      // severed coolant loop: the wing goes cold and scorched while the
      // rest of the array keeps glowing
      R.clone.color.setRGB(1 - 0.6 * k, 1 - 0.66 * k, 1 - 0.7 * k);
      R.clone.emissive.setRGB(1, 0.4, 0.2);
    }
    R.clone.emissiveIntensity = R.orig.emissiveIntensity * (1 - 0.85 * k);
  }

  update(now) {
    const fx = this.fx;
    const marks = this.sd.marks;
    if (now < this.lastNow && marks.length > this.craterCap) {
      // slots wrap: a backward seek must restore the craters they held
      this.resolved = Math.max(0, marksUpTo(marks, now + 0.25) - this.craterCap);
    }
    this.lastNow = now;
    this.resolveUpTo(now + 0.25);
    const st = damageStateAt(fx.timeline, this.id, now, this.state);
    const visible = this.group.visible;
    this._updateSmoke(now);
    if (!visible) return;
    this.craters.visible = this.craters.count > 0;

    // sparks and arcs once the hull is failing
    if (st.sparks > 0.01) {
      this._ensureSparks();
      this._updateEmitters(st.count);
      this.sparks.visible = true;
      this.sparks.material.uniforms.uLevel.value = st.sparks;
      this._updateArcs(now, st.sparks);
    } else if (this.sparks) {
      this.sparks.visible = false;
      this.arcs.visible = false;
    }
    this._applyFlicker(now, st.flicker);
    this._applyRadiators(st.radiator);
  }

  dispose() {
    this.group.remove(this.craters);
    this.craters.geometry.dispose();
    if (this.vents) { this.group.remove(this.vents); this.vents.geometry.dispose(); }
    if (this.sparks) {
      this.group.remove(this.sparks);
      this.sparks.geometry.dispose();
      this.sparks.material.dispose();
      this.group.remove(this.arcs);
      this.arcs.geometry.dispose();
      this.arcMat.dispose();
    }
    if (this.smoke) { this.fx.scene.remove(this.smoke); this.smoke.geometry.dispose(); }
    if (this.radDamage) {
      for (const e of this.radDamage.meshes) e.mesh.material = this.radDamage.orig;
      for (const p of this.group.userData.radiators || []) p.rotation.x = 0;
      this.radDamage.clone.dispose();
    }
  }
}

// ---------------------------------------------------------------------------
// DamageFX: owner
// ---------------------------------------------------------------------------
export class DamageFX {
  /** @param {SceneManager} sm */
  constructor(sm) {
    this.sm = sm;
    this.scene = sm.scene;
    this.SCALE = sm.SCALE;
    this.timeline = null;
    this.views = new Map();
    this.shipIds = [];
    this.uNow = { value: 0 };
    this.uTime = { value: 0 };
    this.lastNow = -Infinity;

    this.planeGeom = new THREE.PlaneGeometry(1, 1);
    this.coneGeom = new THREE.CylinderGeometry(1, 0.16, 1, 14, 8, true).translate(0, 0.5, 0);
    this.chunkGeom = chunkGeometry();
    this.craterMat = craterMaterial(this.uNow);
    this.ventMat = ventMaterial(this.uNow, this.uTime);
    this.sparkMat = sparkMaterial(this.uNow);
    this.smokeMat = smokeMaterial(this.uNow);
    this.debrisMat = debrisMaterial(this.uNow);
    this.edgeMat = edgeGlowMaterial();

    this.raycaster = new THREE.Raycaster();
    this.probe = new THREE.Mesh(undefined, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    this.probe.updateMatrixWorld();
    this._cands = [];
    this._hits = [];
    this._ds = {};
    this._st = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false };

    // Spall: one world-space instanced set, block-allocated per mark
    this.spall = new DebrisSet(this, SPALL_BLOCK * SPALL_BLOCKS);
    this.scene.add(this.spall.mesh);
    this.spallOwner = new Array(SPALL_BLOCKS).fill(null);

    this.blastSpawned = new Set();
    this._addWarmObjects();
    // ShipDestruction finds the breakup helpers through its fx context
    this.scene.userData.damageFX = this;
  }

  /** Visible dummies so SceneManager.warmUp() compiles every program once. */
  _addWarmObjects() {
    const far = new THREE.Vector3(0, -40000, 0);
    const warm = new THREE.Group();
    warm.position.copy(far);
    const cg = new THREE.InstancedBufferGeometry();
    cg.index = this.planeGeom.index;
    cg.setAttribute('position', this.planeGeom.attributes.position);
    cg.setAttribute('uv', this.planeGeom.attributes.uv);
    cg.setAttribute('aCrater', new THREE.InstancedBufferAttribute(new Float32Array(4), 4));
    warm.add(new THREE.InstancedMesh(cg, this.craterMat, 1));
    const vg = new THREE.InstancedBufferGeometry();
    vg.index = this.coneGeom.index;
    for (const k of ['position', 'normal', 'uv']) vg.setAttribute(k, this.coneGeom.attributes[k]);
    vg.setAttribute('aVent', new THREE.InstancedBufferAttribute(new Float32Array(4), 4));
    vg.setAttribute('aVent2', new THREE.InstancedBufferAttribute(new Float32Array(4), 4));
    warm.add(new THREE.InstancedMesh(vg, this.ventMat, 1));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    sg.setAttribute('aNorm', new THREE.BufferAttribute(new Float32Array(3), 3));
    sg.setAttribute('aP', new THREE.BufferAttribute(new Float32Array(4), 4));
    warm.add(new THREE.Points(sg, this.sparkMat));
    warm.add(new THREE.Mesh(billboardGeometry(1, ['aS0', 'aS1', 'aS2']), this.smokeMat));
    warm.add(new THREE.Mesh(this.planeGeom, this.edgeMat));
    const ag = new THREE.BufferGeometry();
    ag.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    warm.add(new THREE.LineSegments(ag, new THREE.LineBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })));
    warm.traverse(o => { o.frustumCulled = false; });
    this.scene.add(warm);
    this._warm = warm;
  }

  /**
   * (Re)build the damage timeline from a loaded recording. Called after
   * load and after live-mode appends; resolved surface points survive
   * rebuilds via a cache keyed by mark identity.
   */
  setRecording(loader) {
    const cache = new Map();
    if (this.timeline) {
      for (const m of this.timeline.marks) if (m._res) cache.set(this._markKey(m), { res: m._res, group: m._resGroup });
    }
    const impactTime = (e) => visualImpactTime(loader, e);
    const rec = { events: loader.events || [], sim_trace: loader.simTrace || [] };
    this.timeline = buildDamageTimeline(rec, { impactTime });
    for (const m of this.timeline.marks) {
      const c = cache.get(this._markKey(m));
      if (c) { m._res = c.res; m._resGroup = c.group; }
    }
    for (const v of this.views.values()) v.dispose();
    this.views.clear();
    this.shipIds = [...this.timeline.ships.keys()];
    for (let b = 0; b < SPALL_BLOCKS; b++) this._freeSpall(b);
    this.spall.flush();
    this.sm._clearBlasts?.('damage');
    this.blastSpawned.clear();
    this.lastNow = -Infinity;
  }

  _markKey(m) {
    return `${m.shipId}|${m.kind}|${m.t.toFixed(3)}|${m.section}|${m.index}`;
  }

  view(shipId) {
    const g = this.sm.ships.get(shipId);
    if (!g || !this.timeline) return null;
    let v = this.views.get(shipId);
    if (v && v.group === g) return v;
    if (v) v.dispose();
    const sd = this.timeline.ships.get(shipId);
    if (!sd) return null;
    v = new ShipView(this, sd, g);
    this.views.set(shipId, v);
    return v;
  }

  /** World position + normal of a mark at its impact time (scene units). */
  markWorld(m, res, outP, outN) {
    const SC = this.SCALE;
    const B = localBasis(m.shipFwd, _basis);
    const p = res.p, n = res.n;
    const sp = m.shipPos || [0, 0, 0];
    outP.set(
      sp[0] * SC + B.x[0] * p.x + B.y[0] * p.y + B.z[0] * p.z,
      sp[1] * SC + B.x[1] * p.x + B.y[1] * p.y + B.z[1] * p.z,
      sp[2] * SC + B.x[2] * p.x + B.y[2] * p.y + B.z[2] * p.z);
    outN.set(
      B.x[0] * n.x + B.y[0] * n.y + B.z[0] * n.z,
      B.x[1] * n.x + B.y[1] * n.y + B.z[1] * n.z,
      B.x[2] * n.x + B.y[2] * n.y + B.z[2] * n.z).normalize();
  }

  _freeSpall(b) {
    const m = this.spallOwner[b];
    if (m) m._spallBlock = -1;
    this.spallOwner[b] = null;
    for (let i = 0; i < SPALL_BLOCK; i++) this.spall.hide(b * SPALL_BLOCK + i);
  }

  _allocSpall(m, view) {
    let b = this.spallOwner.indexOf(null);
    if (b < 0) return;
    this.spallOwner[b] = m;
    m._spallBlock = b;
    const res = view.resolve(m);
    const P = new THREE.Vector3(), N = new THREE.Vector3();
    this.markWorld(m, res, P, N);
    const SC = this.SCALE;
    const baseVel = new THREE.Vector3(m.shipVel[0] * SC, m.shipVel[1] * SC, m.shipVel[2] * SC);
    const W = view.W;
    const tangent = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const vel = new THREE.Vector3();
    const axis = new THREE.Vector3();
    const count = Math.min(SPALL_BLOCK, m.debris);
    const spread = m.kind === 'torpedo' ? 0.9 : 0.7;
    const speedK = W * (m.kind === 'torpedo' ? 1.3 : m.kind === 'secondary' ? 1.1 : 0.8) * (0.7 + 0.35 * m.severity);
    for (let i = 0; i < SPALL_BLOCK; i++) {
      const idx = b * SPALL_BLOCK + i;
      if (i >= count) { this.spall.hide(idx); continue; }
      const r = (k) => hash01(m.seed, 2000 + i, k);
      tangent.set(r(1) - 0.5, r(2) - 0.5, r(3) - 0.5).multiplyScalar(2 * spread);
      dir.copy(N).add(tangent).normalize();
      const sp = speedK * (0.25 + 1.1 * Math.pow(r(4), 1.6));
      vel.copy(baseVel).addScaledVector(dir, sp);
      axis.set(r(5) - 0.5, r(6) - 0.5, r(7) - 0.5).normalize();
      const size = W * (0.005 + 0.018 * Math.pow(r(8), 2.4)) * (0.8 + 0.4 * m.severity);
      const flat = r(9) < 0.5;
      _v.copy(P).addScaledVector(N, W * 0.02);
      const life = SPALL_LIFE * (0.55 + 0.45 * r(10));
      const heat = (m.kind === 'slug' ? 0.45 : 0.8) * (0.4 + 0.8 * r(11));
      this.spall.set(idx, _v, m.t, vel, life, axis, 1 + 5 * r(12),
        size * (flat ? 1.5 : 1), size * (flat ? 0.3 : 0.9), size * (flat ? 1.1 : 1), heat);
    }
  }

  _updateSpall(now) {
    const marks = this.timeline.marks;
    // free blocks whose mark is outside its flight window
    for (let b = 0; b < SPALL_BLOCKS; b++) {
      const m = this.spallOwner[b];
      if (m && (m.t > now || now - m.t > SPALL_LIFE)) this._freeSpall(b);
    }
    // allocate for marks in flight
    let lo = marksUpTo(marks, now - SPALL_LIFE);
    const hi = marksUpTo(marks, now);
    for (let i = lo; i < hi; i++) {
      const m = marks[i];
      if (m._spallBlock >= 0) continue;
      const v = this.view(m.shipId);
      if (!v) continue;
      this._allocSpall(m, v);
    }
    this.spall.flush();
  }

  /** Death-spiral secondaries: Blasts anchored to their scheduled times. */
  _updateBlasts(now) {
    const marks = this.timeline.marks;
    const lo = marksUpTo(marks, now - 4.0);
    const hi = marksUpTo(marks, now);
    for (let i = lo; i < hi; i++) {
      const m = marks[i];
      if (!m.blast || this.blastSpawned.has(m.uid)) continue;
      const v = this.view(m.shipId);
      if (!v) continue;
      this.blastSpawned.add(m.uid);
      const res = v.resolve(m);
      const P = new THREE.Vector3(), N = new THREE.Vector3();
      this.markWorld(m, res, P, N);
      P.addScaledVector(N, v.W * 0.05);
      const SC = this.SCALE;
      this.sm._spawnBlast({
        position: P, dir: N, kind: m.blast, category: 'damage', spawnTime: m.t,
        scale: v.W * (m.blast === 'secondary' ? 0.45 : 0.7),
        baseVel: new THREE.Vector3(m.shipVel[0] * SC, m.shipVel[1] * SC, m.shipVel[2] * SC)
      });
    }
  }

  /** Per-frame: called by SceneManager with the playback time. */
  update(now) {
    if (this._warm) {
      this.scene.remove(this._warm);
      this._warm = null;
    }
    this.uNow.value = now;
    this.uTime.value = this.sm.elapsed || 0;
    if (!this.timeline) return;
    if (now < this.lastNow - 1e-4) {
      // scrubbed backward: blasts respawn from their recorded times
      this.sm._clearBlasts('damage');
      this.blastSpawned.clear();
    }
    for (let i = 0; i < this.shipIds.length; i++) {
      const v = this.view(this.shipIds[i]);
      if (v) v.update(now);
    }
    this._updateSpall(now);
    this._updateBlasts(now);
    this.sm._updateBlasts('damage', now);
    this.lastNow = now;
  }

  // -------------------------------------------------------------------------
  // Destruction helpers (used by ShipDestruction)
  // -------------------------------------------------------------------------

  /** A standalone instanced debris cloud (caller owns and disposes it). */
  createDebrisSet(capacity) {
    const set = new DebrisSet(this, capacity);
    this.scene.add(set.mesh);
    const self = this;
    const dispose = set.dispose.bind(set);
    set.dispose = () => { self.scene.remove(set.mesh); dispose(); };
    return set;
  }

  /** Edge-glow overlay material for a hull section (program shared). */
  createEdgeMaterial() {
    const m = this.edgeMat.clone();
    m.uniforms.uNoise.value = getNoiseTexture();
    return m;
  }
}

/**
 * Split a built hull into big sections along the long axis with ragged
 * cuts: the reactor end is shredded into small chunks (buildDebrisChunks),
 * the rest breaks into one or two large sections that keep their shape.
 * @returns {null | {sections: [{mesh, center, cutLo, cutHi, overlay}], shards: [{mesh, center, size}]}}
 */
export function breakHull(shipGroup, rnd, fx) {
  const ud = shipGroup.userData;
  const parts = ud.parts || [];
  const mats = ud.materials;
  if (!parts.length || !mats) return null;
  const L = ud.size?.length || 8, W = ud.size?.width || 1.8;
  const rz = ud.reactorPos ? ud.reactorPos.z : -0.3 * L;
  const cutA = rz + L * (0.12 + 0.06 * rnd());
  const cutB = cutA + L * (0.3 + 0.14 * rnd());
  const three = rnd() < 0.65;
  const phase = rnd() * 10;
  const ragged = (p) => (Math.sin(p.center.x * 9.1 / W + phase) + Math.cos(p.center.y * 7.3 / W + phase * 1.3)) * 0.03 * L;
  const aft = [], mid = [], bow = [];
  for (const p of parts) {
    const z = p.center.z + ragged(p);
    // parts right at a tear are vaporised
    const dA = Math.abs(z - cutA), dB = three ? Math.abs(z - cutB) : Infinity;
    if (Math.min(dA, dB) < L * 0.025 && rnd() < 0.55) continue;
    if (z < cutA) aft.push(p);
    else if (three && z < cutB) mid.push(p);
    else bow.push(p);
  }
  const sections = [];
  const make = (list, lo, hi) => {
    if (!list.length) return;
    const centroid = new THREE.Vector3();
    for (const p of list) centroid.add(p.center);
    centroid.divideScalar(list.length);
    const byMat = new Map();
    const all = [];
    for (const p of list) {
      const g = p.geom.clone();
      g.translate(-centroid.x, -centroid.y, -centroid.z);
      if (!byMat.has(p.mat)) byMat.set(p.mat, []);
      byMat.get(p.mat).push(g);
      const bare = new THREE.BufferGeometry();
      bare.setAttribute('position', (g.index ? g.toNonIndexed() : g).attributes.position);
      all.push(bare);
    }
    const geoms = [], materials = [];
    for (const [k, gs] of byMat) {
      const g = BufferGeometryUtils.mergeGeometries(gs, false);
      if (!g || !mats[k]) continue;
      geoms.push(g);
      const m = mats[k].clone();
      m.transparent = true;
      m.emissive = new THREE.Color(0xff7a30);
      m.emissiveIntensity = 0;
      if (m.emissiveMap) m.emissiveMap = null;
      materials.push(m);
    }
    const merged = geoms.length ? BufferGeometryUtils.mergeGeometries(geoms, true) : null;
    if (!merged) return;
    const mesh = new THREE.Mesh(merged, materials);
    let overlay = null;
    const og = BufferGeometryUtils.mergeGeometries(all, false);
    if (og && fx) {
      const em = fx.createEdgeMaterial();
      em.uniforms.uCutLo.value = lo - centroid.z;
      em.uniforms.uCutHi.value = hi - centroid.z;
      em.uniforms.uEdgeW.value = L * 0.035;
      em.uniforms.uSeed.value = rnd() * 10;
      overlay = new THREE.Mesh(og, em);
      overlay.renderOrder = 8;
      mesh.renderOrder = 7;
      mesh.add(overlay);
    }
    sections.push({ mesh, center: centroid, cutLo: lo, cutHi: hi, overlay, length: list.length });
  };
  if (three) {
    make(mid, cutA, cutB);
    make(bow, cutB, 1e6);
  } else {
    make(bow, cutA, 1e6);
  }
  const shards = aft.length
    ? buildDebrisChunks({ userData: { parts: aft, materials: mats, size: ud.size } }, 6, rnd)
    : [];
  return { sections, shards, cutA, cutB: three ? cutB : null };
}

/** Travel-adjusted time a coilgun hit lands (as BattleLoader.getVisualImpacts). */
export function visualImpactTime(loader, e) {
  const pid = e.data?.projectile_id;
  if (!pid || !loader.projectileHits || !loader.lastProjectileState) return e.timestamp;
  const hit = loader.projectileHits.get(pid);
  const last = loader.lastProjectileState.get(pid);
  if (!hit?.impact_position || !last?.vel || !last.pos) return e.timestamp;
  const dx = hit.impact_position[0] - last.pos[0];
  const dy = hit.impact_position[1] - last.pos[1];
  const dz = hit.impact_position[2] - last.pos[2];
  const speed = Math.hypot(last.vel[0], last.vel[1], last.vel[2]);
  const travel = speed > 0 ? Math.hypot(dx, dy, dz) / speed : 1.0;
  return last.time + travel;
}
