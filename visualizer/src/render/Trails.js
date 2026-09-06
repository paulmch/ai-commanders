import * as THREE from 'three';
import { getNoiseTexture } from './Textures.js';

/**
 * Camera-facing ribbon: a strip of quads through a point history, widened
 * perpendicular to the view direction in the vertex shader. Used for
 * torpedo exhaust trails, coilgun tracers and PD laser beams (a two-point
 * ribbon with a bright core / soft halo profile).
 */

const vertexShader = /* glsl */`
  attribute float aSide;
  attribute float aT;
  attribute vec3 aTangent;
  uniform float uWidth;
  uniform float uTaper;
  uniform float uPixelWorld;
  uniform float uMaxWidthPx;
  varying float vT;
  varying float vSide;
  varying float vCover;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 wp = (modelMatrix * vec4(position, 1.0)).xyz;
    vec3 view = normalize(cameraPosition - wp);
    vec3 tangent = normalize(aTangent);
    vec3 side = cross(tangent, view);
    float sl = length(side);
    side = sl > 1e-4 ? side / sl : vec3(0.0, 1.0, 0.0);
    float w0 = uWidth * mix(1.0, uTaper, aT);
    // Never thinner than ~1.6 px: a sub-pixel strip rasterises as dots.
    // Dim by the widening so the energy on screen stays the same.
    float pixel = uPixelWorld * max(0.01, -(viewMatrix * vec4(wp, 1.0)).z);
    float limited = uMaxWidthPx > 0.0 ? min(w0, pixel * uMaxWidthPx) : w0;
    float w = max(limited, pixel * 0.8);
    vCover = min(1.0, w0 / w);
    wp += side * aSide * w;
    vT = aT;
    vSide = aSide;
    gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform vec3 uHead;
  uniform vec3 uTail;
  uniform float uIntensity;
  uniform float uFadePow;
  uniform float uCore;     // sharpness of the bright core (0 = flat profile)
  uniform float uHalo;     // halo amount
  uniform float uTime;
  uniform float uShimmer;
  uniform sampler2D uNoise;
  varying float vT;
  varying float vSide;
  varying float vCover;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float x = abs(vSide);
    float prof = uCore > 0.0
      ? exp(-x * x * uCore) + uHalo * exp(-x * x * 2.2) * (1.0 - x)
      : (1.0 - x * x);
    float along = uFadePow > 0.0 ? pow(clamp(1.0 - vT, 0.0, 1.0), uFadePow) : 1.0;
    float shimmer = 1.0;
    if (uShimmer > 0.0) {
      float n = texture2D(uNoise, vec2(vT * 6.0 - uTime * 0.4, 0.3)).r;
      shimmer = 1.0 - uShimmer + uShimmer * (0.6 + 0.8 * n);
    }
    vec3 col = mix(uTail, uHead, along);
    gl_FragColor = vec4(col * prof * along * shimmer * uIntensity * vCover, 1.0);
  }
`;

/** Keep a moving head and fixed samples; tiny frames must not erase the trail. */
export function updateTrailHistory(history, p, t, { maxAge, minStep, max }, emit = true) {
  if (history.length && (t < history[0].t || t - history[0].t > maxAge)) history.length = 0;
  if (emit) {
    const head = history[0];
    const anchor = history[1];
    if (head && (t === head.t || (anchor && anchor.distanceToSquared(p) < minStep * minStep))) {
      head.copy(p);
      head.t = t;
    } else {
      // Commit a fixed sample at the threshold, not at the previous frame's
      // head. Otherwise low frame rates consume the budget too quickly.
      if (head && anchor && minStep > 0) {
        head.copy(p);
        head.t = t;
      }
      const q = p.clone();
      q.t = t;
      history.unshift(q);
    }
  }
  const cutoff = t - maxAge;
  while (history.length && history[history.length - 1].t < cutoff) history.pop();
  if (history.length > max) history.length = max;
}

export class Ribbon {
  /** Shared: world units per pixel at unit distance (set by the scene on resize). */
  static pixelWorld = { value: 0.002 };

  /**
   * @param {Object} o
   *   maxPoints - history capacity
   *   width     - half-width at the head (scene units)
   *   taper     - width multiplier at the tail (0..1)
   *   head/tail - HDR colors [r,g,b]
   *   intensity, fadePow, core, halo, shimmer
   */
  constructor(o) {
    this.max = o.maxPoints;
    const n = this.max;
    const geom = new THREE.BufferGeometry();
    this.pos = new Float32Array(n * 2 * 3);
    this.tan = new Float32Array(n * 2 * 3);
    const side = new Float32Array(n * 2);
    this.t = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      side[i * 2] = -1;
      side[i * 2 + 1] = 1;
    }
    const index = [];
    for (let i = 0; i < n - 1; i++) {
      const a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3;
      index.push(a, b, c, b, d, c);
    }
    geom.setIndex(index);
    geom.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geom.setAttribute('aTangent', new THREE.BufferAttribute(this.tan, 3));
    geom.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    geom.setAttribute('aT', new THREE.BufferAttribute(this.t, 1));
    geom.setDrawRange(0, 0);
    this.geom = geom;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uWidth: { value: o.width },
        uTaper: { value: o.taper ?? 0.2 },
        uHead: { value: new THREE.Color(...o.head) },
        uTail: { value: new THREE.Color(...(o.tail || o.head)) },
        uIntensity: { value: o.intensity ?? 1 },
        uFadePow: { value: o.fadePow ?? 1.5 },
        uCore: { value: o.core ?? 0 },
        uHalo: { value: o.halo ?? 0 },
        uTime: { value: 0 },
        uPixelWorld: Ribbon.pixelWorld,
        uMaxWidthPx: { value: o.maxWidthPx ?? 0 },
        uShimmer: { value: o.shimmer ?? 0 },
        uNoise: { value: getNoiseTexture() }
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
    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 22;
    this.history = [];   // THREE.Vector3 (with .t when time-based), head first
    this.minStep = o.minStep ?? 0;
    // Time-based trails keep `maxAge` battle-seconds of history so the
    // trail length is independent of playback speed and frame rate
    this.maxAge = o.maxAge ?? null;
  }

  /**
   * Push the head position (scene units). Skips tiny steps. With a
   * time-based ribbon pass the battle time `t`: points older than maxAge
   * are dropped and a backward seek clears the history.
   */
  push(p, t = null, emit = true) {
    const h = this.history;
    if (this.maxAge != null && t != null) {
      updateTrailHistory(h, p, t, this, emit);
    } else if (h.length && this.minStep > 0 && h[0].distanceToSquared(p) < this.minStep * this.minStep) {
      h[0].copy(p);
    } else {
      h.unshift(p.clone());
      if (h.length > this.max) h.pop();
    }
    this.rebuild(t);
  }

  /** Replace the whole history (e.g. an explicit two-point beam). */
  setPoints(points) {
    this.history = points.map(p => p.clone()).slice(0, this.max);
    this.rebuild();
  }

  clear() {
    this.history.length = 0;
    this.geom.setDrawRange(0, 0);
  }

  rebuild(time = null) {
    const h = this.history;
    const n = h.length;
    if (n < 2) {
      this.geom.setDrawRange(0, 0);
      return;
    }
    const tmp = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      const p = h[i];
      // tangent from neighbours
      const a = h[Math.max(0, i - 1)], b = h[Math.min(n - 1, i + 1)];
      tmp.subVectors(a, b);
      if (tmp.lengthSq() < 1e-10) tmp.set(0, 0, 1);
      tmp.normalize();
      for (let k = 0; k < 2; k++) {
        const v = (i * 2 + k) * 3;
        this.pos[v] = p.x; this.pos[v + 1] = p.y; this.pos[v + 2] = p.z;
        this.tan[v] = tmp.x; this.tan[v + 1] = tmp.y; this.tan[v + 2] = tmp.z;
        this.t[i * 2 + k] = time != null && this.maxAge != null && p.t != null
          ? THREE.MathUtils.clamp((time - p.t) / this.maxAge, 0, 1)
          : i / (n - 1);
      }
    }
    this.geom.attributes.position.needsUpdate = true;
    this.geom.attributes.aTangent.needsUpdate = true;
    this.geom.attributes.aT.needsUpdate = true;
    this.geom.setDrawRange(0, (n - 1) * 6);
    this.geom.computeBoundingSphere();
  }

  dispose() {
    this.geom.dispose();
    this.material.dispose();
  }
}
