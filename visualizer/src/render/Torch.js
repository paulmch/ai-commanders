import * as THREE from 'three';
import { getNoiseTexture, getGlowTexture } from './Textures.js';

/**
 * Fusion torch exhaust. A tapered tube rendered double-sided additive with a
 * view-dependent thickness term, so it reads as a volume of plasma rather
 * than a painted cone: scrolling turbulence, shock diamonds near the
 * nozzle, a white-hot core fading through blue to violet at the edge.
 *
 * Local frame: nozzle at the origin, exhaust streaming down -Z. Callers
 * scale the group's Z to the plume length each frame.
 */

const vertexShader = /* glsl */`
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vAxis;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    // normalMatrix handles the non-uniform Z scale of the plume
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize((viewMatrix * wp).xyz);
    vAxis = normalize((viewMatrix * modelMatrix * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    gl_Position = projectionMatrix * viewMatrix * wp;
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform sampler2D uNoise;
  uniform float uTime;
  uniform float uThrust;
  uniform float uIntensity;
  uniform float uDiamonds;
  uniform vec3 uCore;
  uniform vec3 uEdge;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vAxis;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float t = vUv.y;                       // 0 nozzle .. 1 tip
    // Turbulence scrolls toward the tip; two octaves at different speeds
    // Scroll slowly: faster than ~1 tile/s aliases into strobing at 60 Hz
    float n1 = texture2D(uNoise, vec2(vUv.x * 2.0, t * 2.5 - uTime * 0.55)).r;
    float n2 = texture2D(uNoise, vec2(vUv.x * 5.0 + 0.37, t * 6.0 - uTime * 0.9)).g;
    float turb = 0.72 + 0.56 * (n1 * 0.65 + n2 * 0.35);

    // Axial profile: hottest just outside the nozzle, exponential decay,
    // pinched to nothing at the tip
    float axial = exp(-t * 3.0) * (1.0 - smoothstep(0.75, 1.0, t)) * smoothstep(0.0, 0.05, t);

    // Shock diamonds: standing wave pattern that fades down the plume
    float wave = abs(sin(t * 3.14159 * (7.0 + 3.0 * uThrust) + 0.6));
    float diamonds = 1.0 + uDiamonds * pow(wave, 10.0) * exp(-t * 5.0);

    // A tube seen edge-on is thin: weight by |n.v| so the silhouette fades
    // Soft silhouette: fade hard toward grazing angles so the plume reads
    // as a glowing volume, not a lit tube with edges
    float ndv = abs(dot(normalize(vNormal), normalize(vViewDir)));
    float thick = pow(ndv, 1.8);
    // Looking down the axis a ray crosses the whole tube: damp the
    // per-fragment contribution so an end-on plume is a bright disc, not
    // a full-screen wash
    float endOn = abs(dot(normalize(vViewDir), vAxis));
    float sinView = sqrt(max(0.0, 1.0 - endOn * endOn));
    thick *= mix(0.04, 1.0, pow(sinView, 1.5));

    float a = axial * turb * thick * diamonds;
    vec3 col = mix(uEdge, uCore, clamp(a * 1.8, 0.0, 1.0) * (0.45 + 0.55 * uThrust));
    gl_FragColor = vec4(col * a * uIntensity, 1.0);
  }
`;

/**
 * Flame profile: the exhaust leaves the nozzle at rNozzle, flares to rMax
 * just behind the bell and tapers to a point. A straight tube projects as
 * a hard-edged rectangle (a bright square when the ship is distant);
 * the lathe profile gives a soft cigar silhouette instead.
 */
function tubeGeometry(rNozzle, rMax, radial = 28, along = 28) {
  const pts = [];
  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  for (let i = 0; i <= along; i++) {
    const t = i / along;
    const flare = rNozzle + (rMax - rNozzle) * smooth(0, 0.3, t);
    const taper = Math.pow(1 - smooth(0.3, 1.0, t), 0.75);
    const r = Math.max(rMax * 0.01, flare * taper);
    pts.push(new THREE.Vector2(r, t));
  }
  const g = new THREE.LatheGeometry(pts, radial);
  g.rotateX(-Math.PI / 2);   // nozzle at z=0, tip at z=-1
  return g;
}

function torchMaterial(core, edge, intensity, diamonds) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uTime: { value: 0 },
      uThrust: { value: 0 },
      uIntensity: { value: intensity },
      uDiamonds: { value: diamonds },
      uCore: { value: new THREE.Color(...core) },
      uEdge: { value: new THREE.Color(...edge) }
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
}

/**
 * @param {Object} opts
 *   radius   - nozzle radius (scene units)
 *   length   - full plume length at 100% thrust
 *   light    - optional THREE.PointLight (budgeted by the caller)
 *   tint     - 'fusion' (white/blue) | 'chemical' (white/orange, torpedoes)
 */
export function createTorch({ radius, length, light = null, tint = 'fusion', intensity = 1 }) {
  const group = new THREE.Group();
  const fusion = tint === 'fusion';
  const I = intensity;
  const coreCol = fusion ? [1.0, 0.97, 1.0] : [1.0, 0.95, 0.85];
  const sheathCol = fusion ? [0.35, 0.55, 1.0] : [1.0, 0.55, 0.2];
  const edgeCol = fusion ? [0.25, 0.2, 0.9] : [0.8, 0.25, 0.1];

  // Slim saturated core, blue sheath kept below white, faint violet halo:
  // a gradient across the plume, never a flat-topped white bar
  const core = new THREE.Mesh(tubeGeometry(radius * 0.38, radius * 0.6),
    torchMaterial(coreCol, sheathCol, 3.0 * I, fusion ? 1.2 : 0.3));
  const sheath = new THREE.Mesh(tubeGeometry(radius * 0.85, radius * 1.5),
    torchMaterial(sheathCol, edgeCol, 0.55, 0.4));
  const outer = new THREE.Mesh(tubeGeometry(radius * 1.3, radius * 2.3, 20, 16),
    torchMaterial(edgeCol, [0, 0, 0], 0.08, 0.0));
  for (const m of [core, sheath, outer]) {
    m.frustumCulled = false;
    m.renderOrder = 20;
    group.add(m);
  }

  // Nozzle glow: an HDR disc at the throat
  const nozzle = new THREE.Sprite(new THREE.SpriteMaterial({
    map: getGlowTexture(),
    color: new THREE.Color(fusion ? 0.8 : 1.0, fusion ? 0.9 : 0.75, 1.0),
    transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
  }));
  nozzle.scale.setScalar(radius * 2.4);
  nozzle.renderOrder = 21;
  group.add(nozzle);

  const _wp = new THREE.Vector3();
  const _wd = new THREE.Vector3();

  const phase = Math.random() * Math.PI * 2;
  const state = { thrust: 0 };

  return {
    group,
    light,
    /**
     * @param {number} thrust    0..1 (already smoothed by the caller)
     * @param {number} time      wall-clock seconds for turbulence
     * @param {number} sputter   0..1 extra flicker (dying drives)
     */
    update(thrust, time, sputter = 0) {
      state.thrust = thrust;
      if (thrust < 0.015) {
        group.visible = false;
        if (light) light.intensity = 0;
        return;
      }
      group.visible = true;
      // Slow breathing only - anything above ~1.5 Hz aliases into strobing
      let breathe = 0.965 + 0.025 * Math.sin(time * 4.3 + phase) + 0.012 * Math.sin(time * 7.1 + phase * 1.7);
      if (sputter > 0) {
        breathe *= 1 - sputter * (0.5 + 0.5 * Math.sin(time * 5.9 + phase * 2.3)) * 0.7;
      }
      // Throttle response: an idle torch is a short violet flicker, full
      // burn a long white spike
      const th = Math.pow(thrust, 0.85);
      const len = length * (0.1 + 0.9 * th) * breathe;
      const width = 0.55 + 0.5 * th;
      group.scale.set(width, width, len);
      const inten = (0.2 + 0.8 * th) * breathe;
      if (light) {
        group.getWorldPosition(_wp);
        _wd.set(0, 0, -1).transformDirection(group.matrixWorld);
        light.position.copy(_wp).addScaledVector(_wd, radius * 3);
      }
      for (const m of [core, sheath, outer]) {
        m.material.uniforms.uTime.value = time;
        m.material.uniforms.uThrust.value = thrust;
      }
      core.material.uniforms.uIntensity.value = 3.0 * I * inten;
      sheath.material.uniforms.uIntensity.value = 0.55 * I * inten;
      outer.material.uniforms.uIntensity.value = 0.08 * I * inten;
      nozzle.material.opacity = Math.min(1, thrust * 0.9) * breathe * 0.5 * Math.min(1, I);
      if (light) light.intensity = (0.15 + 0.85 * th) * 8 * I * breathe;
    },
    dispose() {
      for (const m of [core, sheath, outer]) {
        m.geometry.dispose();
        m.material.dispose();
      }
      nozzle.material.dispose();
    }
  };
}
