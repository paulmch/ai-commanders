import * as THREE from 'three';
import { getStarTexture, getStreakTexture, getPuffTexture } from './Textures.js';

/**
 * GPU particle primitives. All motion is evaluated in the vertex shader
 * from per-particle initial state and a single age uniform, so a burst of
 * thousands of particles costs one uniform write per frame and scrubs
 * freely in either direction on the timeline.
 */

const ADDITIVE = {
  transparent: true,
  depthWrite: false,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor,
  blendEquation: THREE.AddEquation
};

// Drag-integrated displacement: v(t) = v0 e^{-kt}, x(t) = v0 (1 - e^{-kt}) / k
const MOTION_GLSL = /* glsl */`
  vec3 dragPos(vec3 v0, float k, float age) {
    return k > 1e-4 ? v0 * (1.0 - exp(-k * age)) / k : v0 * age;
  }
  vec3 dragVel(vec3 v0, float k, float age) {
    return v0 * exp(-k * age);
  }
`;

function randomDirection(rnd, out) {
  const theta = rnd() * Math.PI * 2;
  const phi = Math.acos(2 * rnd() - 1);
  out.set(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi));
  return out;
}

/**
 * Point sparks: hot pinpoints that fly out, slow under drag and cool from
 * white through orange to dark red.
 *
 * @param {Object} o
 *   count, speed [min,max], life [min,max], size (world units at 1 unit
 *   distance scale), drag, spread (initial volume radius), dir (optional
 *   bias direction), cone (0..1 how tightly to follow dir), hot, cool colors
 */
export class SparkBurst {
  constructor(o) {
    const count = o.count;
    const rnd = o.rnd || Math.random;
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const seed = new Float32Array(count);
    const life = new Float32Array(count);
    const d = new THREE.Vector3();
    const bias = o.dir ? o.dir.clone().normalize() : null;
    for (let i = 0; i < count; i++) {
      randomDirection(rnd, d);
      if (bias) d.lerp(bias, o.cone ?? 0.6).normalize();
      const spread = (o.spread || 0) * Math.cbrt(rnd());
      pos[i * 3] = d.x * spread; pos[i * 3 + 1] = d.y * spread; pos[i * 3 + 2] = d.z * spread;
      const s = o.speed[0] + (o.speed[1] - o.speed[0]) * Math.pow(rnd(), 1.5);
      vel[i * 3] = d.x * s + (o.baseVel?.x || 0);
      vel[i * 3 + 1] = d.y * s + (o.baseVel?.y || 0);
      vel[i * 3 + 2] = d.z * s + (o.baseVel?.z || 0);
      seed[i] = rnd();
      life[i] = o.life[0] + (o.life[1] - o.life[0]) * rnd();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('aVel', new THREE.BufferAttribute(vel, 3));
    geom.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    geom.setAttribute('aLife', new THREE.BufferAttribute(life, 1));

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uAge: { value: -1 },
        uSize: { value: o.size },
        uDrag: { value: o.drag ?? 0.6 },
        uHot: { value: new THREE.Color(...(o.hot || [1.6, 1.5, 1.4])) },
        uWarm: { value: new THREE.Color(...(o.warm || [1.4, 0.7, 0.25])) },
        uCool: { value: new THREE.Color(...(o.cool || [0.5, 0.08, 0.02])) },
        uMap: { value: getStarTexture() },
        uIntensity: { value: o.intensity ?? 1 },
        uPixelRatio: { value: 1 }
      },
      vertexShader: /* glsl */`
        attribute vec3 aVel;
        attribute float aSeed;
        attribute float aLife;
        uniform float uAge;
        uniform float uSize;
        uniform float uDrag;
        uniform float uPixelRatio;
        uniform vec3 uHot;
        uniform vec3 uWarm;
        uniform vec3 uCool;
        varying vec3 vColor;
        varying float vAlpha;
        ${MOTION_GLSL}
        #include <common>
  #include <logdepthbuf_pars_vertex>
        void main() {
          float age = max(uAge, 0.0);
          float f = clamp(age / aLife, 0.0, 1.0);
          vec3 p = position + dragPos(aVel, uDrag, age);
          // tiny tumble so the cloud isn't a perfect radial burst
          float w = age * (0.8 + aSeed) + aSeed * 6.2831;
          p += vec3(sin(w), cos(w * 1.3), sin(w * 0.7)) * aSeed * 0.03 * uSize;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float fade = uAge < 0.0 ? 0.0 : (1.0 - f) * (1.0 - f);
          vAlpha = fade * (0.6 + 0.4 * aSeed);
          // colour cools with normalised age; a few stay hot longer
          float cool = smoothstep(0.0, 0.55 + aSeed * 0.4, f);
          vColor = mix(mix(uHot, uWarm, smoothstep(0.0, 0.5, cool)), uCool, smoothstep(0.45, 1.0, cool));
          gl_PointSize = max(1.0, uSize * (0.55 + aSeed * 0.9) * (1.0 - 0.45 * f) * uPixelRatio * (400.0 / max(1.0, -mv.z)));
          #include <logdepthbuf_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        uniform float uIntensity;
        varying vec3 vColor;
        varying float vAlpha;
        #include <logdepthbuf_pars_fragment>
        void main() {
          #include <logdepthbuf_fragment>
          float a = texture2D(uMap, gl_PointCoord).a * vAlpha;
          gl_FragColor = vec4(vColor * a * uIntensity, 1.0);
        }`,
      ...ADDITIVE
    });
    this.points = new THREE.Points(geom, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 30;
    this.object = this.points;
    this.maxLife = o.life[1];
  }

  setAge(age) {
    this.material.uniforms.uAge.value = age;
    this.points.visible = age >= 0 && age <= this.maxLife;
  }

  dispose() {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * Velocity-stretched streak billboards: glowing fragments that draw a
 * short trail behind themselves in screen space. Instanced quads oriented
 * along the projected velocity each frame in the vertex shader.
 */
export class StreakBurst {
  constructor(o) {
    const count = o.count;
    const rnd = o.rnd || Math.random;
    const base = new THREE.PlaneGeometry(1, 1);
    const geom = new THREE.InstancedBufferGeometry();
    geom.index = base.index;
    geom.setAttribute('position', base.attributes.position);
    geom.setAttribute('uv', base.attributes.uv);
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const seed = new Float32Array(count);
    const life = new Float32Array(count);
    const d = new THREE.Vector3();
    const bias = o.dir ? o.dir.clone().normalize() : null;
    for (let i = 0; i < count; i++) {
      randomDirection(rnd, d);
      if (bias) d.lerp(bias, o.cone ?? 0.6).normalize();
      const spread = (o.spread || 0) * Math.cbrt(rnd());
      pos[i * 3] = d.x * spread; pos[i * 3 + 1] = d.y * spread; pos[i * 3 + 2] = d.z * spread;
      const s = o.speed[0] + (o.speed[1] - o.speed[0]) * Math.pow(rnd(), 1.3);
      vel[i * 3] = d.x * s + (o.baseVel?.x || 0);
      vel[i * 3 + 1] = d.y * s + (o.baseVel?.y || 0);
      vel[i * 3 + 2] = d.z * s + (o.baseVel?.z || 0);
      seed[i] = rnd();
      life[i] = o.life[0] + (o.life[1] - o.life[0]) * rnd();
    }
    geom.setAttribute('aPos0', new THREE.InstancedBufferAttribute(pos, 3));
    geom.setAttribute('aVel', new THREE.InstancedBufferAttribute(vel, 3));
    geom.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seed, 1));
    geom.setAttribute('aLife', new THREE.InstancedBufferAttribute(life, 1));
    geom.instanceCount = count;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uAge: { value: -1 },
        uLen: { value: o.length },
        uWidth: { value: o.width },
        uDrag: { value: o.drag ?? 0.5 },
        uHot: { value: new THREE.Color(...(o.hot || [1.8, 1.7, 1.5])) },
        uWarm: { value: new THREE.Color(...(o.warm || [1.5, 0.75, 0.3])) },
        uCool: { value: new THREE.Color(...(o.cool || [0.6, 0.1, 0.03])) },
        uMap: { value: getStreakTexture() },
        uIntensity: { value: o.intensity ?? 1 }
      },
      vertexShader: /* glsl */`
        attribute vec3 aPos0;
        attribute vec3 aVel;
        attribute float aSeed;
        attribute float aLife;
        uniform float uAge;
        uniform float uLen;
        uniform float uWidth;
        uniform float uDrag;
        uniform vec3 uHot;
        uniform vec3 uWarm;
        uniform vec3 uCool;
        varying vec2 vUv;
        varying vec3 vColor;
        varying float vAlpha;
        ${MOTION_GLSL}
        #include <common>
  #include <logdepthbuf_pars_vertex>
        void main() {
          float age = max(uAge, 0.0);
          float f = clamp(age / aLife, 0.0, 1.0);
          vec3 p = aPos0 + dragPos(aVel, uDrag, age);
          vec3 v = dragVel(aVel, uDrag, age);
          vec4 pv = modelViewMatrix * vec4(p, 1.0);
          vec3 vv = normalMatrix * v;
          vec2 d = vv.xy;
          float dl = length(d);
          d = dl > 1e-5 ? d / dl : vec2(1.0, 0.0);
          vec2 perp = vec2(-d.y, d.x);
          // streak length follows current speed so it shortens as it slows
          float speedK = exp(-uDrag * age);
          float len = uLen * (0.45 + aSeed * 0.9) * (0.25 + 0.75 * speedK);
          float wid = uWidth * (0.6 + aSeed * 0.7);
          pv.xy += d * position.x * len + perp * position.y * wid;
          gl_Position = projectionMatrix * pv;
          vUv = uv;
          float fade = uAge < 0.0 ? 0.0 : pow(1.0 - f, 1.4);
          vAlpha = fade;
          float cool = smoothstep(0.0, 0.5 + aSeed * 0.45, f);
          vColor = mix(mix(uHot, uWarm, smoothstep(0.0, 0.5, cool)), uCool, smoothstep(0.5, 1.0, cool));
          #include <logdepthbuf_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        uniform float uIntensity;
        varying vec2 vUv;
        varying vec3 vColor;
        varying float vAlpha;
        #include <logdepthbuf_pars_fragment>
        void main() {
          #include <logdepthbuf_fragment>
          float a = texture2D(uMap, vUv).a * vAlpha;
          gl_FragColor = vec4(vColor * a * uIntensity, 1.0);
        }`,
      side: THREE.DoubleSide,
      ...ADDITIVE
    });
    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 31;
    this.object = this.mesh;
    this.maxLife = o.life[1];
  }

  setAge(age) {
    this.material.uniforms.uAge.value = age;
    this.mesh.visible = age >= 0 && age <= this.maxLife;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * Billboard gas cloud: a handful of large noise puffs that drift outward,
 * rotate slowly and cool from yellow-white to dim red before fading.
 * Cheap CPU animation - never more than a few dozen sprites per blast.
 */
export class PuffCloud {
  constructor(o) {
    const rnd = o.rnd || Math.random;
    this.group = new THREE.Group();
    this.object = this.group;
    this.puffs = [];
    this.duration = o.duration;
    this.hot = new THREE.Color(...(o.hot || [1.2, 1.0, 0.72]));
    this.warm = new THREE.Color(...(o.warm || [1.3, 0.45, 0.12]));
    this.cool = new THREE.Color(...(o.cool || [0.35, 0.05, 0.02]));
    this.intensity = o.intensity ?? 1;
    const d = new THREE.Vector3();
    for (let i = 0; i < o.count; i++) {
      randomDirection(rnd, d);
      const speed = o.speed[0] + (o.speed[1] - o.speed[0]) * rnd();
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: getPuffTexture(i % 3),
        color: this.hot.clone(),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        rotation: rnd() * Math.PI * 2
      }));
      s.renderOrder = 25;
      const startR = (o.spread || 0) * Math.cbrt(rnd());
      this.puffs.push({
        sprite: s,
        origin: d.clone().multiplyScalar(startR),
        vel: d.clone().multiplyScalar(speed),
        spin: (rnd() - 0.5) * 0.6,
        size0: o.size[0] + (o.size[1] - o.size[0]) * rnd(),
        grow: o.grow ?? 1.8,
        seed: rnd(),
        delay: (o.stagger || 0) * rnd()
      });
      this.group.add(s);
    }
    this.tmp = new THREE.Color();
  }

  setAge(age) {
    const vis = age >= 0 && age <= this.duration;
    this.group.visible = vis;
    if (!vis) return;
    for (const p of this.puffs) {
      const a = age - p.delay;
      if (a < 0) { p.sprite.material.opacity = 0; continue; }
      const f = Math.min(1, a / (this.duration - p.delay));
      const drag = 1 / (1 + a * 0.55);
      p.sprite.position.copy(p.origin).addScaledVector(p.vel, a * drag);
      const size = p.size0 * (1 + p.grow * (1 - Math.exp(-a * 0.7)));
      p.sprite.scale.set(size, size, 1);
      p.sprite.material.rotation += p.spin * 0.016;
      // Envelope: quick bloom-in, long tail
      const env = Math.min(1, a / 0.25) * Math.pow(1 - f, 1.6);
      p.sprite.material.opacity = Math.min(1, env * (0.55 + 0.45 * p.seed));
      const cool = Math.min(1, f * (1.1 + p.seed * 0.4));
      if (cool < 0.5) this.tmp.copy(this.hot).lerp(this.warm, cool * 2);
      else this.tmp.copy(this.warm).lerp(this.cool, (cool - 0.5) * 2);
      p.sprite.material.color.copy(this.tmp).multiplyScalar(this.intensity);
    }
  }

  dispose() {
    for (const p of this.puffs) p.sprite.material.dispose();
  }
}
