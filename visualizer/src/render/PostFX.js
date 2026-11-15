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

// Composite vertex shader: also meters the frame for exposure adaptation.
// The coarsest bloom level is the thresholded HDR frame shrunk to a few
// dozen texels, i.e. "how much of the picture is far over white". Five
// centre-weighted taps of it (per VERTEX - four fetches of a tiny texture)
// pull the exposure down while a fireball fills the view, so the blast keeps
// a hot core and coloured falloff instead of clipping flat, and the frame
// recovers as it fades. Stateless: a function of the frame alone, so
// scrubbing the timeline is deterministic.
const COMPOSITE_VERT = /* glsl */`
  uniform sampler2D tHot;
  uniform float uAdapt;
  uniform float uAdaptFloor;
  uniform float uAdaptMin;
  varying vec2 vUv;
  varying float vExposure;
  float hotAt(vec2 q) { return dot(texture2D(tHot, q).rgb, vec3(0.2126, 0.7152, 0.0722)); }
  void main() {
    vUv = uv;
    float hot = 0.32 * hotAt(vec2(0.5)) + 0.17 * (hotAt(vec2(0.27, 0.3)) + hotAt(vec2(0.73, 0.3))
              + hotAt(vec2(0.27, 0.7)) + hotAt(vec2(0.73, 0.7)));
    vExposure = max(uAdaptMin, 1.0 / (1.0 + uAdapt * max(hot - uAdaptFloor, 0.0)));
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

// Screen-space ambient occlusion (high tier). Reads the scene's depth
// texture, which holds three's LOGARITHMIC depth: d = log2(1 + w) /
// log2(far + 1), so view distance w = exp2(d * uLogBase) - 1 and the view
// position follows from the projection scale. The radius is constant on
// SCREEN (capped in world units), so plating reads at any zoom and distant
// ships - smaller than the radius - sample only sky and get none. One
// half-res pass: 10 spiral taps rotated per pixel, normals rebuilt from the
// nearer depth neighbour on each axis; the composite blurs it with 5 taps
// and keeps it off emitters.
const AO_FRAG = /* glsl */`
  uniform sampler2D tDepth;
  uniform vec2 uTexel;      // 1 / full-res size
  uniform float uLogBase;   // log2(far + 1)
  uniform vec2 uProj;       // 1 / projection[0][0], 1 / projection[1][1]
  uniform float uRadius;    // screen heights
  uniform float uMaxWorld;  // km
  uniform float uAspect;
  varying vec2 vUv;
  #define TAPS 10
  float viewW(vec2 uv) { return exp2(texture2D(tDepth, uv).x * uLogBase) - 1.0; }
  vec3 viewPos(vec2 uv, float w) { return vec3((uv * 2.0 - 1.0) * uProj * w, -w); }
  void main() {
    float d0 = texture2D(tDepth, vUv).x;
    float w0 = exp2(d0 * uLogBase) - 1.0;
    // sky, and the planet / rings (tens of thousands of km: no contact detail)
    if (d0 > 0.99999 || w0 > 20000.0) { gl_FragColor = vec4(1.0); return; }
    vec3 P = viewPos(vUv, w0);
    vec2 dx = vec2(uTexel.x, 0.0), dy = vec2(0.0, uTexel.y);
    vec3 Pl = viewPos(vUv - dx, viewW(vUv - dx)), Pr = viewPos(vUv + dx, viewW(vUv + dx));
    vec3 Pd = viewPos(vUv - dy, viewW(vUv - dy)), Pu = viewPos(vUv + dy, viewW(vUv + dy));
    vec3 ex = abs(Pr.z - P.z) < abs(P.z - Pl.z) ? Pr - P : P - Pl;
    vec3 ey = abs(Pu.z - P.z) < abs(P.z - Pd.z) ? Pu - P : P - Pd;
    vec3 n = normalize(cross(ex, ey));
    float rW = uRadius * 2.0 * uProj.y * w0;          // world size of the screen radius here
    float shrink = min(1.0, uMaxWorld / max(rW, 1e-6));
    rW *= shrink;
    vec2 rUv = uRadius * shrink * vec2(1.0 / uAspect, 1.0);
    float ang = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float occ = 0.0;
    for (int i = 0; i < TAPS; i++) {
      float fi = float(i);
      float a = ang + fi * 2.39996323;
      vec2 q = vUv + vec2(cos(a), sin(a)) * sqrt((fi + 0.5) / float(TAPS)) * rUv;
      vec3 v = viewPos(q, viewW(q)) - P;
      float vv = dot(v, v);
      float fall = max(1.0 - vv / (rW * rW), 0.0);    // only geometry within the radius occludes
      occ += fall * max(dot(v, n) / (sqrt(vv) + 1e-5) - 0.12, 0.0);
    }
    float ao = clamp(1.0 - 3.2 * occ / float(TAPS), 0.0, 1.0);
    gl_FragColor = vec4(ao, ao, ao, 1.0);
  }`;

// Anamorphic streak: a wide horizontal blur of the quarter-res bloom level
// (already thresholded - only true emitters streak). One cheap pass into a
// quarter-res target; the composite just adds it, tinted.
const STREAK_FRAG = /* glsl */`
  uniform sampler2D tSrc;
  uniform float uTexel;     // 1 / source width
  varying vec2 vUv;
  #define TAPS 28
  void main() {
    vec3 acc = texture2D(tSrc, vUv).rgb;
    float wsum = 1.0;
    for (int i = 1; i <= TAPS; i++) {
      float fi = float(i);
      float w = exp(-fi * 0.115);
      vec2 o = vec2(fi * 2.0 * uTexel, 0.0);
      acc += (texture2D(tSrc, vUv + o).rgb + texture2D(tSrc, vUv - o).rgb) * w;
      wsum += 2.0 * w;
    }
    gl_FragColor = vec4(acc / wsum, 1.0);
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
  uniform sampler2D tStreak;
  uniform sampler2D tAO;
  uniform vec2 uAoTexel;
  uniform float uAoStrength;
  uniform vec3 uStreakTint;
  uniform float uGrade;
  varying vec2 vUv;
  varying float vExposure;

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

    #ifdef USE_AO
      // Ambient occlusion: 5-tap blur of the half-res term. Emitters and
      // hot highlights are not ambient light, so it fades out with
      // luminance (bell liners, windows, plumes over a hull stay bright).
      float ao = 0.28 * texture2D(tAO, uv).r
               + 0.18 * (texture2D(tAO, uv + vec2(1.5, 1.5) * uAoTexel).r + texture2D(tAO, uv + vec2(-1.5, 1.5) * uAoTexel).r
                       + texture2D(tAO, uv + vec2(1.5, -1.5) * uAoTexel).r + texture2D(tAO, uv + vec2(-1.5, -1.5) * uAoTexel).r);
      float aoKeep = smoothstep(0.35, 1.1, luma(col));
      col *= mix(1.0 - uAoStrength * (1.0 - ao), 1.0, aoKeep);
      #ifdef AO_DEBUG
        col = vec3(ao) * 0.35;
      #endif
    #endif

    col += texture2D(tBloom, uv).rgb * uBloomStrength;
    #ifdef USE_STREAK
      col += texture2D(tStreak, uv).rgb * uStreakTint;
    #endif
    #ifdef USE_GODRAYS
      col += texture2D(tGod, uv).r * uGodStrength * uSunColor;
    #endif
    if (uSunOnScreen > 0.001) {
      float vis = texture2D(tVis, vec2(0.5)).r * uSunOnScreen;
      if (vis > 0.002) col += flare(uv, vis);
    }

    col *= vExposure;   // exposure adaptation (COMPOSITE_VERT)

    // ACES + sRGB exactly as the old OutputPass did (renderer settings)
    #ifdef TONE_MAPPING
      col = toneMapping(col);
    #endif
    col = linearToOutputTexel(vec4(col, 1.0)).rgb;

    // Print-style grade (display space): a gentle S-curve for contrast and
    // a split tone - cool shadows, warm highlights
    // (the curve is held off near-white values: hot cores keep their colour
    // gradient instead of being pushed into the clip)
    vec3 curved = col * col * (3.0 - 2.0 * col);
    float hi = smoothstep(0.55, 0.95, max(col.r, max(col.g, col.b)));
    col = mix(col, curved, uGrade * (1.0 - hi));
    float gl = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col *= mix(vec3(1.0), mix(vec3(0.93, 0.985, 1.07), vec3(1.035, 1.0, 0.95), smoothstep(0.02, 0.65, gl)), uGrade * 3.0);

    // Detonation flash: a global bleach toward warm white
    col = mix(col, uFlashColor, clamp(uFlash, 0.0, 1.0));
    // Vignette
    col *= 1.0 - uVignette * smoothstep(0.15, 0.75, r2 * 1.6);
    // Fine grain, luminance-weighted so black stays clean
    float g = (hash(gl_FragCoord.xy) - 0.5) * uGrain;
    col += g * (0.25 + 0.75 * dot(col, vec3(0.333)));
    gl_FragColor = vec4(col, 1.0);
  }`;

// Render-global look constants (lane `light` owns these; see the header of
// SceneManager's renderer setup for tone mapping / exposure).
export const BLOOM_THRESHOLD = 1.35;
export const BLOOM_STRENGTH = 0.22;
export const STREAK_STRENGTH = 0.1;
export const GRADE = 0.2;
// Exposure adaptation: exposure = 1 / (1 + ADAPT_STRENGTH * max(hot - ADAPT_FLOOR, 0)),
// never below ADAPT_MIN; `hot` is the centre-weighted mean of the coarsest
// bloom level. The floor keeps drive plumes and tracer fire from moving it.
export const ADAPT_STRENGTH = 0.55;
export const ADAPT_FLOOR = 0.3;
export const ADAPT_MIN = 0.5;
export const AO_STRENGTH = 0.85;   // how far full occlusion darkens (high tier)
export const AO_RADIUS = 0.032;    // screen heights
export const AO_MAX_WORLD = 2.5;   // km cap on the radius (extreme close-ups stay local)

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
    this.bloom = new BloomChain({ mips: preset.bloomMips, threshold: BLOOM_THRESHOLD, knee: 0.45, scatter: 0.62, karis: 0.15 });
    this.bloom.setSize(this.width, this.height);
    this.BLOOM_BASE = BLOOM_STRENGTH;

    const lin = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.visRT = new THREE.WebGLRenderTarget(1, 1, lin);
    this.godRT = new THREE.WebGLRenderTarget(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1), lin);
    this.streakRT = new THREE.WebGLRenderTarget(Math.max(1, this.width >> 2), Math.max(1, this.height >> 2), lin);
    this.aoRT = new THREE.WebGLRenderTarget(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1), {
      depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter
    });
    this.aoMat = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: null }, uTexel: { value: new THREE.Vector2() }, uLogBase: { value: 1 },
        uProj: { value: new THREE.Vector2(1, 1) }, uRadius: { value: AO_RADIUS }, uMaxWorld: { value: AO_MAX_WORLD },
        uAspect: { value: 1 }
      },
      vertexShader: VERT, fragmentShader: AO_FRAG, depthTest: false, depthWrite: false
    });
    this.streakMat = new THREE.ShaderMaterial({
      uniforms: { tSrc: { value: null }, uTexel: { value: 1 / Math.max(1, this.width >> 2) } },
      vertexShader: VERT, fragmentShader: STREAK_FRAG, depthTest: false, depthWrite: false
    });

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
        uFlareStrength: { value: 1.0 },
        tStreak: { value: this.streakRT.texture },
        tAO: { value: this.aoRT.texture },
        uAoTexel: { value: new THREE.Vector2(2 / this.width, 2 / this.height) },
        uAoStrength: { value: AO_STRENGTH },
        uStreakTint: { value: new THREE.Color(0.42, 0.62, 1.0).multiplyScalar(STREAK_STRENGTH) },
        uGrade: { value: GRADE },
        tHot: { value: null },
        uAdapt: { value: ADAPT_STRENGTH },
        uAdaptFloor: { value: ADAPT_FLOOR },
        uAdaptMin: { value: ADAPT_MIN }
      },
      vertexShader: COMPOSITE_VERT,
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
    this.streakRT.dispose();
    this.streakMat.dispose();
    this.aoRT.dispose();
    this.aoMat.dispose();
    this.bloom.dispose();
    this.visMat.dispose();
    this.godMat.dispose();
    this.compositeMat.dispose();
    this.quad.dispose();
  }

  _makeSceneTarget() {
    // God rays and AO need the depth buffer (sky = never written)
    this.hasDepthTex = !!(this.gfx.preset.godRays || this.gfx.preset.ao);
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
    if (p.streak) defs.USE_STREAK = '';
    if (p.ao) defs.USE_AO = '';
    this.compositeMat.defines = defs;
    this.compositeMat.needsUpdate = true;
  }

  applyQuality() {
    const p = this.gfx.preset;
    const samples = this.renderer.capabilities.isWebGL2 ? p.msaa : 0;
    if (samples !== this.samples || !!(p.godRays || p.ao) !== this.hasDepthTex) {
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
    this.streakRT.setSize(Math.max(1, this.width >> 2), Math.max(1, this.height >> 2));
    this.aoRT.setSize(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1));
    this.compositeMat.uniforms.uAoTexel.value.set(2 / this.width, 2 / this.height);
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

    if (this.gfx.preset.ao && this.sceneRT.depthTexture) {
      const au = this.aoMat.uniforms;
      const pe = this.camera.projectionMatrix.elements;
      au.tDepth.value = this.sceneRT.depthTexture;
      au.uTexel.value.set(1 / this.width, 1 / this.height);
      au.uLogBase.value = Math.log2(this.camera.far + 1);
      au.uProj.value.set(1 / pe[0], 1 / pe[5]);
      au.uAspect.value = this.width / this.height;
      this.quad.material = this.aoMat;
      r.setRenderTarget(this.aoRT);
      this.quad.render(r);
    }

    if (this.gfx.preset.streak) {
      const src = this.bloom.level(1) || this.bloom.level(0);
      if (src) {
        this.streakMat.uniforms.tSrc.value = src.texture;
        this.streakMat.uniforms.uTexel.value = 1 / src.width;
        this.quad.material = this.streakMat;
        r.setRenderTarget(this.streakRT);
        this.quad.render(r);
      }
    }

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
    u.tHot.value = this.bloom.coarsest();
    this.quad.material = this.compositeMat;
    r.setRenderTarget(null);
    this.quad.render(r);
  }
}

const _size = new THREE.Vector2();
