import * as THREE from 'three';
import { Ribbon } from './Trails.js';

/**
 * Hull navigation lights: every lamp on a ship drawn as ONE Points object
 * (one draw call per ship, no per-light sprites or PointLights).
 *
 * Kinds
 *   steady  - running lights (red port / green starboard), faction lamps
 *   strobe  - white anti-collision double flash at the hull extremities
 *   beacon  - slow smooth pulse (mast tops)
 *
 * Every blink runs on BATTLE time, so scrubbing and paused frames are
 * deterministic. Dying ships lose their strobes and their running lights
 * brown out; a destroyed hulk (updateShip no longer called) goes dark.
 */

const VERT = /* glsl */`
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aLevel;
  uniform float uPixelWorld;
  uniform float uMinPx;
  uniform float uMaxPx;
  varying vec3 vColor;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float px = aSize / (max(-mv.z, 1e-4) * uPixelWorld);
    float size = clamp(px, uMinPx, uMaxPx);
    // Below the minimum footprint keep the lamp as a faint twinkle
    // instead of a full-strength dot: energy falls with apparent size.
    float fade = clamp(px / uMinPx, 0.22, 1.0);
    vColor = aColor * aLevel * fade;
    gl_PointSize = size;
    if (aLevel <= 0.001) gl_PointSize = 0.0;
    #include <logdepthbuf_vertex>
  }
`;

const FRAG = /* glsl */`
  varying vec3 vColor;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float core = exp(-r2 * 22.0);
    float halo = exp(-r2 * 5.0) * 0.28;
    gl_FragColor = vec4(vColor * (core * 2.4 + halo), 1.0);
  }
`;

let _material = null;
function lightMaterial() {
  if (_material) return _material;
  _material = new THREE.ShaderMaterial({
    uniforms: {
      uPixelWorld: Ribbon.pixelWorld,
      uMinPx: { value: 2.2 },
      uMaxPx: { value: 42.0 }
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });
  return _material;
}

/**
 * Brightness of one lamp at battle time t.
 * @param {{kind:string, period?:number, phase?:number, level?:number}} l
 * @param {number} t      battle time (s)
 * @param {{dying?:boolean, hull?:number}} s  hull is 0..1
 */
export function lampLevel(l, t, s = {}) {
  const base = l.level ?? 1;
  const hull = s.hull ?? 1;
  const period = l.period || 1.6;
  const ph = (((t / period + (l.phase || 0)) % 1) + 1) % 1;
  if (l.kind === 'strobe') {
    if (s.dying) return 0;
    // Double flash: two short pulses a tenth of a period apart
    return (ph < 0.035 || (ph > 0.11 && ph < 0.145)) ? base : 0;
  }
  let v = base;
  if (l.kind === 'beacon') {
    const w = Math.max(0, Math.sin(ph * Math.PI * 2));
    v = base * (0.06 + 0.94 * w * w * w * w);
  }
  // Failing power: brown-out flicker below 35 % hull, near-dark while dying
  if (s.dying || hull < 0.35) {
    const seed = (l.phase || 0) * 37.1;
    const f = Math.sin(t * 7.3 + seed) * Math.sin(t * 2.9 + seed * 1.7);
    const drop = s.dying ? 0.2 : 0.55;
    v *= f > drop ? 0.05 : (s.dying ? 0.25 : 0.8);
  }
  return v;
}

/**
 * Build the lamp cloud for a ship. `lamps`: [{pos:Vector3, color:hex|Color,
 * kind, size, period, phase, level}]. Returns the Points object (added to
 * `group`) or null when there are no lamps.
 */
export function buildHullLights(group, lamps) {
  if (!lamps.length) return null;
  const n = lamps.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const size = new Float32Array(n);
  const level = new Float32Array(n);
  const c = new THREE.Color();
  lamps.forEach((l, i) => {
    pos[i * 3] = l.pos.x; pos[i * 3 + 1] = l.pos.y; pos[i * 3 + 2] = l.pos.z;
    c.set(l.color);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    size[i] = l.size;
  });
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geom.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  const levelAttr = new THREE.BufferAttribute(level, 1);
  levelAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('aLevel', levelAttr);
  geom.computeBoundingSphere();
  // Lamps sit on the hull; pad the sphere so glow halos never cull early
  geom.boundingSphere.radius *= 1.2;
  const points = new THREE.Points(geom, lightMaterial());
  points.name = 'hull_lights';
  points.renderOrder = 15;
  points.frustumCulled = true;
  points.userData.lamps = lamps;
  // A destroyed ship is no longer updated (updateShip returns early), so
  // the drifting hulk would keep its last lamp state: go dark instead.
  points.onBeforeRender = () => {
    const ud = points.userData;
    if (!ud.dark && ud.isDead && ud.isDead()) {
      levelAttr.array.fill(0);
      levelAttr.needsUpdate = true;
      ud.dark = true;
    }
  };
  group.add(points);
  return points;
}

/**
 * Per-frame update: battle time t, ship state s ({dying, hull 0..1}).
 * isDead() is polled at draw time so a hulk the update loop no longer
 * visits still goes dark.
 */
export function updateHullLights(points, t, s, isDead = null) {
  if (!points) return;
  const ud = points.userData;
  const attr = points.geometry.attributes.aLevel;
  const arr = attr.array;
  const lamps = ud.lamps;
  for (let i = 0; i < lamps.length; i++) arr[i] = lampLevel(lamps[i], t, s);
  attr.needsUpdate = true;
  if (isDead) ud.isDead = isDead;
  ud.dark = false;
}
