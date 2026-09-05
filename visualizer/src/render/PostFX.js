import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/**
 * HDR post chain:
 *   scene (linear, half-float, MSAA) -> bloom (high threshold: only true
 *   emitters bloom - torches, blasts, beams, the sun) -> ACES + sRGB ->
 *   final grade (detonation flash, chromatic aberration, vignette, grain).
 *
 * Tone mapping lives in the OutputPass, so materials render linear HDR and
 * additive effects can stack far above 1.0 without clipping to flat white.
 */
export class PostFX {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());

    const target = new THREE.WebGLRenderTarget(size.x, size.y, {
      type: THREE.HalfFloatType,
      samples: renderer.capabilities.isWebGL2 ? 4 : 0
    });
    this.composer = new EffectComposer(renderer, target);
    this.composer.setPixelRatio(renderer.getPixelRatio());

    this.renderPass = new RenderPass(scene, camera);
    this.composer.addPass(this.renderPass);

    // Low radius: the coarse bloom mips (one texel ~20 px) turn any tiny
    // bright source into a soft square that flickers as it crosses texels
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.5, 0.12, 1.45);
    this.composer.addPass(this.bloom);
    this.BLOOM_BASE = 0.5;

    this.output = new OutputPass();
    this.composer.addPass(this.output);

    this.finalPass = new ShaderPass({
      uniforms: {
        tDiffuse: { value: null },
        uFlash: { value: 0 },
        uFlashColor: { value: new THREE.Color(1, 0.96, 0.9) },
        uAberration: { value: 0.0018 },
        uVignette: { value: 0.32 },
        uGrain: { value: 0.035 },
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(size.x, size.y) }
      },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tDiffuse;
        uniform float uFlash;
        uniform vec3 uFlashColor;
        uniform float uAberration;
        uniform float uVignette;
        uniform float uGrain;
        uniform float uTime;
        uniform vec2 uResolution;
        varying vec2 vUv;

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7)) + uTime * 13.7) * 43758.5453);
        }

        void main() {
          vec2 uv = vUv;
          vec2 c = uv - 0.5;
          float r2 = dot(c, c);
          // Radial chromatic aberration, stronger toward the corners and
          // during a detonation flash
          float ab = uAberration * (1.0 + uFlash * 6.0) * (0.35 + r2 * 3.0);
          vec2 dir = c * ab;
          float cr = texture2D(tDiffuse, uv + dir).r;
          float cg = texture2D(tDiffuse, uv).g;
          float cb = texture2D(tDiffuse, uv - dir).b;
          vec3 col = vec3(cr, cg, cb);

          // Detonation flash: a global bleach toward warm white
          col = mix(col, uFlashColor, clamp(uFlash, 0.0, 1.0));

          // Vignette
          float vig = 1.0 - uVignette * smoothstep(0.15, 0.75, r2 * 1.6);
          col *= vig;

          // Fine grain, luminance-weighted so black stays clean
          float g = (hash(gl_FragCoord.xy) - 0.5) * uGrain;
          col += g * (0.25 + 0.75 * dot(col, vec3(0.333)));

          gl_FragColor = vec4(col, 1.0);
        }`
    });
    this.composer.addPass(this.finalPass);

    this.flash = 0;          // current frame flash amount, decays in update()
    this.flashTarget = 0;
    this.time = 0;
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
    this.finalPass.uniforms.uFlash.value = this.flash;
    this.finalPass.uniforms.uTime.value = this.time;
    this.bloom.strength = this.BLOOM_BASE + this.flash * 0.9;
  }

  setSize(w, h) {
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
    this.finalPass.uniforms.uResolution.value.set(w, h);
  }

  render() {
    this.composer.render();
  }
}
