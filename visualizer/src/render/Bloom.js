import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/**
 * Physically-flavoured bloom (Jimenez, "Next Generation Post Processing in
 * Call of Duty: Advanced Warfare", 2014):
 *
 *   HDR scene -> soft-knee threshold + 13-tap downsample (Karis average on
 *   the first mip, so a single bright pixel cannot flicker) -> N mips ->
 *   9-tap tent upsample, each level blended into the next finer one.
 *
 * Unlike UnrealBloomPass (plain bilinear mips + separable blur) the tent
 * reconstruction keeps tiny emitters round instead of turning them into
 * soft squares that pop as they cross texels.
 */
const VERT = /* glsl */`
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }`;

const DOWN_FRAG = /* glsl */`
  uniform sampler2D tSrc;
  uniform vec2 uTexel;        // 1 / source size
  uniform float uThreshold;
  uniform float uKnee;
  uniform float uPrefilter;   // 1 on the first pass
  uniform float uKaris;       // 0 = plain average, 1 = full Karis weighting
  varying vec2 vUv;

  vec3 tap(vec2 o) { return texture2D(tSrc, vUv + o * uTexel).rgb; }
  float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
  vec3 soft(vec3 c) {
    float br = max(c.r, max(c.g, c.b));
    float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
    rq = rq * rq / (4.0 * uKnee + 1e-4);
    return c * max(rq, br - uThreshold) / max(br, 1e-4);
  }
  void main() {
    vec3 a = tap(vec2(-2.0, -2.0)), b = tap(vec2(0.0, -2.0)), c = tap(vec2(2.0, -2.0));
    vec3 d = tap(vec2(-1.0, -1.0)), e = tap(vec2(1.0, -1.0));
    vec3 f = tap(vec2(-2.0, 0.0)), g = tap(vec2(0.0, 0.0)), h = tap(vec2(2.0, 0.0));
    vec3 i = tap(vec2(-1.0, 1.0)), j = tap(vec2(1.0, 1.0));
    vec3 k = tap(vec2(-2.0, 2.0)), l = tap(vec2(0.0, 2.0)), m = tap(vec2(2.0, 2.0));
    vec3 col;
    if (uPrefilter > 0.5) {
      // Karis average: weight each 2x2 group by 1/(1+luma) so a single
      // bright pixel cannot dominate (no fireflies / texel popping)
      vec3 s0 = (d + e + i + j) * 0.25, s1 = (a + b + f + g) * 0.25, s2 = (b + c + g + h) * 0.25;
      vec3 s3 = (f + g + k + l) * 0.25, s4 = (g + h + l + m) * 0.25;
      float w0 = 0.5 / (1.0 + uKaris * lum(s0)), w1 = 0.125 / (1.0 + uKaris * lum(s1));
      float w2 = 0.125 / (1.0 + uKaris * lum(s2)), w3 = 0.125 / (1.0 + uKaris * lum(s3));
      float w4 = 0.125 / (1.0 + uKaris * lum(s4));
      col = (s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3 + s4 * w4) / (w0 + w1 + w2 + w3 + w4);
      col = soft(col);
    } else {
      col = (d + e + i + j) * 0.125
          + (a + c + k + m) * 0.03125
          + (b + f + h + l) * 0.0625
          + g * 0.125;
    }
    gl_FragColor = vec4(max(col, 0.0), 1.0);
  }`;

const UP_FRAG = /* glsl */`
  uniform sampler2D tLow;     // coarser, already-accumulated level
  uniform sampler2D tHigh;    // this level's downsample
  uniform vec2 uTexel;        // 1 / coarse size
  uniform float uScatter;
  varying vec2 vUv;
  vec3 tap(vec2 o) { return texture2D(tLow, vUv + o * uTexel).rgb; }
  void main() {
    vec3 up = tap(vec2(-1.0, -1.0)) + tap(vec2(1.0, -1.0)) + tap(vec2(-1.0, 1.0)) + tap(vec2(1.0, 1.0));
    up += 2.0 * (tap(vec2(0.0, -1.0)) + tap(vec2(-1.0, 0.0)) + tap(vec2(1.0, 0.0)) + tap(vec2(0.0, 1.0)));
    up += 4.0 * tap(vec2(0.0));
    up *= 1.0 / 16.0;
    vec3 hi = texture2D(tHigh, vUv).rgb;
    gl_FragColor = vec4(hi + up * uScatter, 1.0);
  }`;

export class BloomChain {
  constructor({ mips = 6, threshold = 1.3, knee = 0.5, scatter = 0.7, karis = 0.15 } = {}) {
    this.threshold = threshold;
    this.karis = karis;
    this.knee = knee;
    this.scatter = scatter;
    this.mipCount = mips;
    this.down = [];
    this.up = [];
    this.downMat = new THREE.ShaderMaterial({
      uniforms: {
        tSrc: { value: null }, uTexel: { value: new THREE.Vector2() },
        uThreshold: { value: threshold }, uKnee: { value: knee }, uPrefilter: { value: 1 },
        uKaris: { value: karis }
      },
      vertexShader: VERT, fragmentShader: DOWN_FRAG, depthTest: false, depthWrite: false
    });
    this.upMat = new THREE.ShaderMaterial({
      uniforms: {
        tLow: { value: null }, tHigh: { value: null },
        uTexel: { value: new THREE.Vector2() }, uScatter: { value: scatter }
      },
      vertexShader: VERT, fragmentShader: UP_FRAG, depthTest: false, depthWrite: false
    });
    this.quad = new FullScreenQuad(this.downMat);
    this.width = 0;
    this.height = 0;
  }

  _makeTarget(w, h) {
    return new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
      type: THREE.HalfFloatType, depthBuffer: false,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping
    });
  }

  setMipCount(n) {
    if (n === this.mipCount) return;
    this.mipCount = n;
    const w = this.width, h = this.height;
    this.width = this.height = 0;
    this.setSize(w, h);
  }

  setSize(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.dispose(false);
    let w = Math.round(width / 2), h = Math.round(height / 2);
    for (let i = 0; i < this.mipCount; i++) {
      this.down.push(this._makeTarget(w, h));
      this.up.push(i < this.mipCount - 1 ? this._makeTarget(w, h) : null);
      w = Math.max(1, Math.round(w / 2));
      h = Math.max(1, Math.round(h / 2));
    }
  }

  /** Runs the chain on `input` (HDR texture); returns the half-res bloom. */
  render(renderer, input, inputWidth, inputHeight) {
    const du = this.downMat.uniforms;
    du.uThreshold.value = this.threshold;
    du.uKnee.value = this.knee;
    du.uKaris.value = this.karis;
    this.quad.material = this.downMat;
    let src = input, sw = inputWidth, sh = inputHeight;
    for (let i = 0; i < this.down.length; i++) {
      du.tSrc.value = src;
      du.uTexel.value.set(1 / sw, 1 / sh);
      du.uPrefilter.value = i === 0 ? 1 : 0;
      renderer.setRenderTarget(this.down[i]);
      this.quad.render(renderer);
      src = this.down[i].texture;
      sw = this.down[i].width;
      sh = this.down[i].height;
    }
    const uu = this.upMat.uniforms;
    uu.uScatter.value = this.scatter;
    this.quad.material = this.upMat;
    let low = this.down[this.down.length - 1];
    for (let i = this.down.length - 2; i >= 0; i--) {
      uu.tLow.value = low.texture;
      uu.tHigh.value = this.down[i].texture;
      uu.uTexel.value.set(1 / low.width, 1 / low.height);
      renderer.setRenderTarget(this.up[i]);
      this.quad.render(renderer);
      low = this.up[i];
    }
    return low.texture;
  }

  dispose(all = true) {
    for (const t of this.down) t.dispose();
    for (const t of this.up) t?.dispose();
    this.down = [];
    this.up = [];
    if (all) {
      this.downMat.dispose();
      this.upMat.dispose();
      this.quad.dispose();
    }
  }
}
