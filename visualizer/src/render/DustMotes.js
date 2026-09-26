import * as THREE from 'three';
import { mulberry32 } from './SkyTextures.js';

/**
 * Near-field dust motes: a faint, world-anchored particle field around the
 * camera that parallaxes and streaks with camera motion - a sense of speed
 * in follow/orbit shots. High quality only.
 *
 * The field is a periodic box re-centred on the camera in the vertex
 * shader (fract wrap), so motes stay put in the world while the camera
 * flies through them. Box size follows the zoom in powers of two with a
 * cross-fade between two octaves, so zooming never makes motes pop. Streak
 * length is the camera's wall-clock velocity times a short shutter, so a
 * paused frame (or a headless screenshot) shows still, round motes. Fades
 * out in the tactical view where it would only add clutter.
 */
const COUNT = 420;

export function dustLayerScales(viewDist) {
  const x = Math.log2(Math.max(viewDist, 1e-3) * 0.9);
  const k = Math.floor(x);
  const f = x - k;
  return { s0: Math.pow(2, k), s1: Math.pow(2, k + 1), w0: 1 - f, w1: f };
}

function makeLayer(seed) {
  const rnd = mulberry32(seed);
  const seeds = new Float32Array(COUNT * 3);
  const bright = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i++) {
    seeds[i * 3] = rnd(); seeds[i * 3 + 1] = rnd(); seeds[i * 3 + 2] = rnd();
    bright[i] = 0.35 + 0.65 * Math.pow(rnd(), 2);
  }
  // Points: one vertex per mote. Lines: head + tail vertices per mote.
  const pts = new THREE.BufferGeometry();
  pts.setAttribute('position', new THREE.BufferAttribute(new Float32Array(COUNT * 3), 3));
  pts.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 3));
  pts.setAttribute('aBright', new THREE.BufferAttribute(bright, 1));
  pts.setAttribute('aTail', new THREE.BufferAttribute(new Float32Array(COUNT), 1));
  const lseeds = new Float32Array(COUNT * 6);
  const lbright = new Float32Array(COUNT * 2);
  const ltail = new Float32Array(COUNT * 2);
  for (let i = 0; i < COUNT; i++) {
    for (let e = 0; e < 2; e++) {
      lseeds.set(seeds.subarray(i * 3, i * 3 + 3), (i * 2 + e) * 3);
      lbright[i * 2 + e] = bright[i];
      ltail[i * 2 + e] = e;
    }
  }
  const lines = new THREE.BufferGeometry();
  lines.setAttribute('position', new THREE.BufferAttribute(new Float32Array(COUNT * 6), 3));
  lines.setAttribute('aSeed', new THREE.BufferAttribute(lseeds, 3));
  lines.setAttribute('aBright', new THREE.BufferAttribute(lbright, 1));
  lines.setAttribute('aTail', new THREE.BufferAttribute(ltail, 1));
  return { pts, lines };
}

const VERT = /* glsl */`
  #include <common>
  #include <logdepthbuf_pars_vertex>
  attribute vec3 aSeed;
  attribute float aBright;
  attribute float aTail;
  uniform vec3 uCam;
  uniform vec3 uStreak;      // world-space streak vector (-camera velocity * shutter)
  uniform float uScale;      // box size
  uniform float uWeight;
  uniform float uPointPx;
  varying float vA;
  void main() {
    vec3 rel = (fract(aSeed - uCam / uScale + 0.5) - 0.5) * uScale;
    vec3 wp = uCam + rel + uStreak * aTail;
    vec4 mv = viewMatrix * vec4(wp, 1.0);
    gl_Position = projectionMatrix * mv;
    float d = length(rel) / uScale;
    // fade in past the near zone, out toward the box edge
    vA = aBright * uWeight * smoothstep(0.03, 0.09, d) * (1.0 - smoothstep(0.3, 0.48, d));
    gl_PointSize = uPointPx * (0.8 + 0.6 * aBright);
    #include <logdepthbuf_vertex>
  }`;

const FRAG = /* glsl */`
  #include <logdepthbuf_pars_fragment>
  uniform vec3 uColor;
  uniform float uIsPoint;
  varying float vA;
  void main() {
    #include <logdepthbuf_fragment>
    float a = vA;
    if (uIsPoint > 0.5) {
      vec2 p = gl_PointCoord * 2.0 - 1.0;
      a *= exp(-dot(p, p) * 3.0);
    }
    gl_FragColor = vec4(uColor * a, 1.0);
  }`;

export class DustMotes {
  constructor(parent) {
    this.group = new THREE.Group();
    this.group.name = 'dustMotes';
    parent.add(this.group);
    this.enabled = true;
    this.velocity = new THREE.Vector3();
    this.lastPos = null;
    this.layers = [];
    for (let i = 0; i < 2; i++) {
      const { pts, lines } = makeLayer(4242 + i * 17);
      const mk = (isPoint) => new THREE.ShaderMaterial({
        uniforms: {
          uCam: { value: new THREE.Vector3() },
          uStreak: { value: new THREE.Vector3() },
          uScale: { value: 1 },
          uWeight: { value: 0 },
          uPointPx: { value: 2.2 },
          uColor: { value: new THREE.Color(0.55, 0.55, 0.6) },
          uIsPoint: { value: isPoint ? 1 : 0 }
        },
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });
      const points = new THREE.Points(pts, mk(true));
      const streaks = new THREE.LineSegments(lines, mk(false));
      for (const o of [points, streaks]) {
        o.frustumCulled = false;
        o.renderOrder = 5;
        this.group.add(o);
      }
      this.layers.push({ points, streaks });
    }
  }

  setEnabled(on) {
    this.enabled = !!on;
    this.group.visible = this.enabled;
  }

  update(camera, delta, viewDist = null) {
    if (!this.enabled) { this.lastPos = null; return; }
    const cp = camera.position;
    // Distance to what the camera looks at: OrbitControls target when
    // SceneManager exposes it, else a nominal close-up scale
    const dist = viewDist ?? (camera.userData.viewDist || 20);
    const { s0, s1, w0, w1 } = dustLayerScales(dist);

    // Wall-clock camera velocity; a jump bigger than the box is a cut
    if (this.lastPos && delta > 0) {
      _d.subVectors(cp, this.lastPos);
      if (_d.length() > s0 * 0.5) this.velocity.set(0, 0, 0);
      else {
        _d.divideScalar(delta);
        this.velocity.lerp(_d, 1 - Math.exp(-delta * 12));
      }
    }
    this.lastPos = (this.lastPos || new THREE.Vector3()).copy(cp);

    // Tactical zoom: dust would only be noise over the plot
    const zoomFade = 1 - THREE.MathUtils.smoothstep(dist, 120, 300);
    const intensity = 0.16 * zoomFade;
    this.group.visible = intensity > 0.002;
    const scales = [s0, s1];
    const weights = [w0, w1];
    for (let i = 0; i < 2; i++) {
      const S = scales[i];
      const streak = _s.copy(this.velocity).multiplyScalar(-0.06);
      const maxLen = S * 0.25;
      if (streak.length() > maxLen) streak.setLength(maxLen);
      const len = streak.length();
      const { points, streaks } = this.layers[i];
      for (const o of [points, streaks]) {
        const u = o.material.uniforms;
        u.uCam.value.copy(cp);
        u.uScale.value = S;
        u.uStreak.value.copy(streak);
      }
      // energy conservation: a long streak is dimmer per pixel
      const pxLen = len / Math.max(1e-6, S) * 60;
      points.material.uniforms.uWeight.value = weights[i] * intensity * (1 / (1 + pxLen * 0.5));
      streaks.material.uniforms.uWeight.value = weights[i] * intensity * Math.min(1, pxLen) * 1.4 / (1 + pxLen * 0.08);
      streaks.visible = pxLen > 0.05;
    }
  }
}

const _d = new THREE.Vector3();
const _s = new THREE.Vector3();
