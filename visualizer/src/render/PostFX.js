import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { BloomChain } from './Bloom.js';
import { getGraphicsSettings } from './Quality.js';
import { sunScreenInfo } from './sunScreen.js';

/**
 * HDR post chain (hand-rolled, no EffectComposer):
 *
 *   scene -> half-float target (4x MSAA on high)
 *         -> bloom mip chain (13-tap down / tent up)
 *         -> sun visibility: 16 taps of the HDR frame around the sun, into
 *            a 1x1 target - lens flare occlusion with no CPU readback, so a
 *            hull sliding over the sun kills the flare on the same frame
 *         -> god rays (high): half-res radial march toward the sun over
 *            an occlusion-masked glow (depth texture: open sky = source)
 *         -> composite to screen: [FXAA on low | chromatic aberration on
 *            high] + bloom + rays + lens flare (starburst, ghosts, halo),
 *            ACES tone map + sRGB (renderer settings), detonation flash,
 *            vignette, grain.
 *
 * One composite pass replaces OutputPass + grade pass: every full-screen
 * pass into a half-float target costs ~100-170 ms/frame in SwiftShader.
 */

const VERT = /* glsl */`
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }`;

const VIS_FRAG = /* glsl */`
  uniform sampler2D tScene;
  uniform vec2 uSunUv;
  uniform vec2 uRadius;     // sun disc radius in uv units (x, y)
  uniform float uRef;       // luminance that counts as "sun fully visible"
  varying vec2 vUv;
  void main() {
    float acc = 0.0;
    for (int i = 0; i < 16; i++) {
      float fi = float(i);
      float a = fi * 2.39996323;
      float r = sqrt((fi + 0.5) / 16.0) * 0.85;
      vec2 q = uSunUv + vec2(cos(a), sin(a)) * r * uRadius;
      float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
      vec3 c = texture2D(tScene, q).rgb;
      acc += inside * clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)) / uRef, 0.0, 1.0);
    }
    gl_FragColor = vec4(acc / 16.0, 0.0, 0.0, 1.0);
  }`;

const GOD_FRAG = /* glsl */`
  uniform sampler2D tDepth;
  uniform vec2 uSunUv;
  uniform float uAspect;
  uniform float uSourceR;   // glow radius of the ray source (screen heights)
  varying vec2 vUv;
  #define STEPS 48
  // Occlusion-masked source: an analytic glow around the sun, visible only
  // where nothing wrote depth (open sky). Hulls and the planet cut it, and
  // the radial march turns those cuts into shafts.
  float source(vec2 q) {
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    float sky = step(0.99999, texture2D(tDepth, q).x);
    float d = length((q - uSunUv) * vec2(uAspect, 1.0));
    return inside * sky * (exp(-d / uSourceR) + 0.08 * exp(-d / (uSourceR * 2.5)));
  }
  void main() {
    vec2 delta = (uSunUv - vUv) / float(STEPS);
    // interleaved-gradient jitter of the start turns step banding into
    // fine noise that the grain pass hides
    float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    vec2 q = vUv + delta * j;
    float acc = source(q);
    for (int i = 0; i < STEPS; i++) {
      q += delta;
      acc += source(q);
    }
    float d = length((vUv - uSunUv) * vec2(uAspect, 1.0));
    float v = acc / float(STEPS + 1) * exp(-d * 3.2);
    gl_FragColor = vec4(v, v, v, 1.0);
  }`;

const COMPOSITE_FRAG = /* glsl */`
  uniform sampler2D tScene;
  uniform sampler2D tBloom;
  uniform sampler2D tGod;
  uniform sampler2D tVis;
  uniform vec2 uTexel;
  uniform float uBloomStrength;
  uniform float uGodStrength;
  uniform float uFlash;
  uniform vec3 uFlashColor;
  uniform float uAberration;
  uniform float uVignette;
  uniform float uGrain;
  uniform float uTime;
  uniform float uAspect;
  uniform vec2 uSunUv;
  uniform float uSunOnScreen;
  uniform vec3 uSunColor;
  uniform float uFlareStrength;
  varying vec2 vUv;

  float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
  // Reversible range compression so AA filtering works on HDR input
  vec3 compress(vec3 c) { return c / (1.0 + max(c.r, max(c.g, c.b))); }
  vec3 expand(vec3 c) { return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-3); }

  #ifdef USE_FXAA
  // FXAA (Lottes, "console" variant) on range-compressed HDR taps
  vec3 fxaa(vec2 uv) {
    vec3 nw = compress(texture2D(tScene, uv + vec2(-1.0, -1.0) * uTexel).rgb);
    vec3 ne = compress(texture2D(tScene, uv + vec2(1.0, -1.0) * uTexel).rgb);
    vec3 sw = compress(texture2D(tScene, uv + vec2(-1.0, 1.0) * uTexel).rgb);
    vec3 se = compress(texture2D(tScene, uv + vec2(1.0, 1.0) * uTexel).rgb);
    vec3 m = compress(texture2D(tScene, uv).rgb);
    float lNW = luma(nw), lNE = luma(ne), lSW = luma(sw), lSE = luma(se), lM = luma(m);
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    if (lMax - lMin < max(0.02, lMax * 0.125)) return expand(m);
    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
    float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, vec2(-8.0), vec2(8.0)) * uTexel;
    vec3 a = 0.5 * (compress(texture2D(tScene, uv + dir * (1.0 / 3.0 - 0.5)).rgb) +
                    compress(texture2D(tScene, uv + dir * (2.0 / 3.0 - 0.5)).rgb));
    vec3 b = a * 0.5 + 0.25 * (compress(texture2D(tScene, uv - dir * 0.5).rgb) +
                               compress(texture2D(tScene, uv + dir * 0.5).rgb));
    float lB = luma(b);
    return expand((lB < lMin || lB > lMax) ? a : b);
  }
  #endif

  // One lens ghost: soft disc brighter toward its rim, at t along sun->centre
  vec3 ghost(vec2 p, vec2 s, float t, float rad, vec3 col) {
    float gd = length(p - s * t) / rad;
    return col * smoothstep(1.0, 0.82, gd) * (0.25 + 0.75 * gd * gd) * 0.05;
  }

  // Lens flare in HDR, driven by the measured sun visibility
  vec3 flare(vec2 uv, float vis) {
    vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
    vec2 s = (uSunUv - 0.5) * vec2(uAspect, 1.0);
    vec2 d = p - s;
    float r = length(d);
    float ang = atan(d.y, d.x);
    vec3 acc = vec3(0.0);
    // glare + 6-blade aperture starburst (constant ~2 px spike width)
    acc += uSunColor * (exp(-r * 45.0) * 1.4 + exp(-r * 9.0) * 0.1);
    float w = r / 0.0028;
    float spikes = exp(-pow(abs(sin(ang * 3.0 + 0.35)) * w, 2.0)) * exp(-r * 15.0);
    spikes += 0.45 * exp(-pow(abs(sin(ang * 3.0 + 0.35 + 0.5236)) * w * 1.4, 2.0)) * exp(-r * 20.0);
    acc += uSunColor * spikes * 1.3;
    // faint anamorphic streak
    acc += vec3(0.45, 0.62, 1.0) * exp(-abs(d.y) * 900.0) * exp(-abs(d.x) * 6.0) * 0.35;
    // ghosts along the sun -> centre axis (t < 0: mirrored past the centre)
    acc += ghost(p, s, 0.62, 0.018, vec3(1.0, 0.75, 0.45));
    acc += ghost(p, s, 0.34, 0.045, vec3(0.45, 0.8, 1.0));
    acc += ghost(p, s, -0.22, 0.03, vec3(0.6, 1.0, 0.65));
    acc += ghost(p, s, -0.46, 0.075, vec3(0.85, 0.55, 1.0));
    acc += ghost(p, s, -0.75, 0.022, vec3(1.0, 0.6, 0.4));
    acc += ghost(p, s, -1.18, 0.11, vec3(0.4, 0.65, 1.0));
    // chromatic halo ring around the optical centre, strongest toward the sun
    float hr = length(p);
    float facing = pow(max(0.0, dot(normalize(p + 1e-5), normalize(s + 1e-5))), 3.0);
    vec3 halo = vec3(exp(-pow((hr - 0.47) / 0.012, 2.0)),
                     exp(-pow((hr - 0.455) / 0.012, 2.0)),
                     exp(-pow((hr - 0.44) / 0.012, 2.0)));
    acc += halo * facing * 0.028;
    return acc * vis * uFlareStrength;
  }

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7)) + uTime * 13.7) * 43758.5453);
  }

  void main() {
    vec2 uv = vUv;
    vec2 c = uv - 0.5;
    float r2 = dot(c, c);
    vec3 col;
    #ifdef USE_FXAA
      col = fxaa(uv);
    #elif defined(USE_CA)
      // Radial chromatic aberration, stronger toward the corners and
      // during a detonation flash
      vec2 dir = c * uAberration * (1.0 + uFlash * 6.0) * (0.35 + r2 * 3.0);
      col = vec3(texture2D(tScene, uv + dir).r, texture2D(tScene, uv).g, texture2D(tScene, uv - dir).b);
    #else
      col = texture2D(tScene, uv).rgb;
    #endif

    col += texture2D(tBloom, uv).rgb * uBloomStrength;
    #ifdef USE_GODRAYS
      col += texture2D(tGod, uv).r * uGodStrength * uSunColor;
    #endif
    if (uSunOnScreen > 0.001) {
      float vis = texture2D(tVis, vec2(0.5)).r * uSunOnScreen;
      if (vis > 0.002) col += flare(uv, vis);
    }

    // ACES + sRGB exactly as the old OutputPass did (renderer settings)
    #ifdef TONE_MAPPING
      col = toneMapping(col);
    #endif
    col = linearToOutputTexel(vec4(col, 1.0)).rgb;

    // Detonation flash: a global bleach toward warm white
    col = mix(col, uFlashColor, clamp(uFlash, 0.0, 1.0));
    // Vignette
    col *= 1.0 - uVignette * smoothstep(0.15, 0.75, r2 * 1.6);
    // Fine grain, luminance-weighted so black stays clean
    float g = (hash(gl_FragCoord.xy) - 0.5) * uGrain;
    col += g * (0.25 + 0.75 * dot(col, vec3(0.333)));
    gl_FragColor = vec4(col, 1.0);
  }`;

export class PostFX {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.gfx = getGraphicsSettings(renderer);
    const preset = this.gfx.preset;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.width = size.x;
    this.height = size.y;

    this.samples = renderer.capabilities.isWebGL2 ? preset.msaa : 0;
    this.sceneRT = this._makeSceneTarget();

    // Only true emitters bloom - torches, blasts, beams, the sun
    this.bloom = new BloomChain({ mips: preset.bloomMips, threshold: 1.35, knee: 0.45, scatter: 0.62, karis: 0.15 });
    this.bloom.setSize(this.width, this.height);
    this.BLOOM_BASE = 0.22;

    const lin = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.visRT = new THREE.WebGLRenderTarget(1, 1, lin);
    this.godRT = new THREE.WebGLRenderTarget(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1), lin);

    this.visMat = new THREE.ShaderMaterial({
      uniforms: {
        tScene: { value: null }, uSunUv: { value: new THREE.Vector2() },
        uRadius: { value: new THREE.Vector2(0.01, 0.01) }, uRef: { value: 18 }
      },
      vertexShader: VERT, fragmentShader: VIS_FRAG, depthTest: false, depthWrite: false
    });
    this.godMat = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: null }, uSunUv: { value: new THREE.Vector2() },
        uAspect: { value: 1 }, uSourceR: { value: 0.035 }
      },
      vertexShader: VERT, fragmentShader: GOD_FRAG, depthTest: false, depthWrite: false
    });
    this.compositeMat = new THREE.ShaderMaterial({
      uniforms: {
        tScene: { value: null },
        tBloom: { value: null },
        tGod: { value: this.godRT.texture },
        tVis: { value: this.visRT.texture },
        uTexel: { value: new THREE.Vector2(1 / this.width, 1 / this.height) },
        uBloomStrength: { value: this.BLOOM_BASE },
        uGodStrength: { value: 0.45 },
        uFlash: { value: 0 },
        uFlashColor: { value: new THREE.Color(1, 0.96, 0.9) },
        uAberration: { value: 0.0018 },
        uVignette: { value: 0.3 },
        uGrain: { value: 0.03 },
        uTime: { value: 0 },
        uAspect: { value: this.width / this.height },
        uSunUv: { value: new THREE.Vector2(-1, -1) },
        uSunOnScreen: { value: 0 },
        uSunColor: { value: new THREE.Color(1.0, 0.92, 0.8) },
        uFlareStrength: { value: 1.0 }
      },
      vertexShader: VERT,
      fragmentShader: COMPOSITE_FRAG,
      depthTest: false,
      depthWrite: false
    });
    this._applyDefines();
    this.quad = new FullScreenQuad(this.compositeMat);

    this.flash = 0;
    this.time = 0;
    this.sun = { uv: new THREE.Vector2(-1, -1), onScreen: 0, radiusUv: new THREE.Vector2(), visible: false };
    this._offGfx = this.gfx.onChange(() => this.applyQuality());
  }

  /** Release every render target, material and the bloom chain (reload). */
  dispose() {
    this._offGfx?.();
    this.sceneRT.dispose();
    this.visRT.dispose();
    this.godRT.dispose();
    this.bloom.dispose();
    this.visMat.dispose();
    this.godMat.dispose();
    this.compositeMat.dispose();
    this.quad.dispose();
  }

  _makeSceneTarget() {
    // God rays need the depth buffer (sky = never written) as occlusion mask
    this.hasDepthTex = !!this.gfx.preset.godRays;
    return new THREE.WebGLRenderTarget(this.width, this.height, {
      type: THREE.HalfFloatType,
      samples: this.samples,
      depthTexture: this.hasDepthTex ? new THREE.DepthTexture(this.width, this.height) : undefined
    });
  }

  _applyDefines() {
    const p = this.gfx.preset;
    const defs = {};
    if (p.fxaa) defs.USE_FXAA = '';
    else if (p.aberration) defs.USE_CA = '';
    if (p.godRays) defs.USE_GODRAYS = '';
    this.compositeMat.defines = defs;
    this.compositeMat.needsUpdate = true;
  }

  applyQuality() {
    const p = this.gfx.preset;
    const samples = this.renderer.capabilities.isWebGL2 ? p.msaa : 0;
    if (samples !== this.samples || !!p.godRays !== this.hasDepthTex) {
      this.samples = samples;
      this.sceneRT.dispose();
      this.sceneRT = this._makeSceneTarget();
    }
    this.bloom.setMipCount(p.bloomMips);
    this._applyDefines();
  }

  /** Request a screen flash (0..1). Stacks with any current flash. */
  triggerFlash(amount) {
    this.flash = Math.min(1.2, this.flash + amount);
  }

  update(delta) {
    this.time += delta;
    // Fast exponential decay: a reactor breach whites the frame for a few
    // hundred ms, never seconds
    this.flash *= Math.exp(-delta * 6.0);
    if (this.flash < 0.002) this.flash = 0;
    const u = this.compositeMat.uniforms;
    u.uFlash.value = this.flash;
    u.uTime.value = this.time;
    u.uBloomStrength.value = this.BLOOM_BASE + this.flash * 0.4;

    const sun = this.scene.userData.sun;
    if (sun) {
      this.camera.updateMatrixWorld();
      sunScreenInfo(this.camera, sun.dir, sun.angularRadius, this.sun);
      if (sun.color) u.uSunColor.value.copy(sun.color);
    } else {
      this.sun.onScreen = 0;
    }
    u.uSunUv.value.copy(this.sun.uv);
    u.uSunOnScreen.value = this.sun.onScreen;
  }

  setSize() {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.width = size.x;
    this.height = size.y;
    // recreate rather than resize: keeps an attached depth texture in step
    this.sceneRT.dispose();
    this.sceneRT = this._makeSceneTarget();
    this.bloom.setSize(this.width, this.height);
    this.godRT.setSize(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1));
    const u = this.compositeMat.uniforms;
    u.uTexel.value.set(1 / this.width, 1 / this.height);
    u.uAspect.value = this.width / this.height;
  }

  render() {
    const r = this.renderer;
    // Stay in sync with any renderer resize / pixel-ratio change
    const size = r.getDrawingBufferSize(_size);
    if (size.x !== this.width || size.y !== this.height) this.setSize();

    r.setRenderTarget(this.sceneRT);
    r.clear();
    r.render(this.scene, this.camera);
    const sceneTex = this.sceneRT.texture;

    const bloomTex = this.bloom.render(r, sceneTex, this.width, this.height);
    const u = this.compositeMat.uniforms;

    if (this.sun.onScreen > 0.001) {
      const vu = this.visMat.uniforms;
      vu.tScene.value = sceneTex;
      vu.uSunUv.value.copy(this.sun.uv);
      vu.uRadius.value.copy(this.sun.radiusUv);
      this.quad.material = this.visMat;
      r.setRenderTarget(this.visRT);
      this.quad.render(r);
      if (this.gfx.preset.godRays) {
        const gu = this.godMat.uniforms;
        gu.tDepth.value = this.sceneRT.depthTexture;
        gu.uSunUv.value.copy(this.sun.uv);
        gu.uAspect.value = this.width / this.height;
        this.quad.material = this.godMat;
        r.setRenderTarget(this.godRT);
        this.quad.render(r);
        u.uGodStrength.value = 0.45;
      }
    } else {
      u.uGodStrength.value = 0;
    }

    u.tScene.value = sceneTex;
    u.tBloom.value = bloomTex;
    this.quad.material = this.compositeMat;
    r.setRenderTarget(null);
    this.quad.render(r);
  }
}

const _size = new THREE.Vector2();
