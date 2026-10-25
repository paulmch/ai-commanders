import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { getNoiseTexture } from './Textures.js';

/**
 * Cumulative hull wear, drawn as an overlay skin on the ship group (the
 * hull kit and its shared materials stay untouched):
 *
 *  - armour ablation per recorded section (nose / lateral / tail, as a
 *    fraction of the as-built thickness): paint and plating strip away in
 *    noise-shaped patches that grow with the loss, exposing pitted, dull
 *    bare structure ringed with soot
 *  - soot and scorching that spreads with lost hull integrity, so a beaten
 *    ship reads darker even at fleet range
 *  - a dying hulk's reactor glow: cracks along the plating that pulse
 *    faster as the detonation nears (DamageTimeline.reactorPulse)
 *
 * The overlay is the hull's own geometry (ship-local `userData.parts`),
 * inflated a hair along the normal and depth-tested LessEqual, lit by the
 * same lights. Built lazily the first time a ship shows any wear; one
 * shared program (customProgramCacheKey) with per-ship uniforms. Every
 * input is a pure function of battle time, so scrubbing is exact.
 */

// Hull kit material keys the wear skin covers (not windows, accents, bells)
const WEAR_MATS = new Set(['hull', 'trim', 'paint', 'dark']);
/** Below this the overlay stays hidden (and is never built). */
export const WEAR_EPS = 0.015;
// sanitised uniform inputs: a missing or non-finite value means "no wear"
const unit = (x) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);
const lossOf = (remaining) => unit(1 - (Number.isFinite(remaining) ? remaining : 1));

/**
 * Section weights for a ship-local surface point: nose / lateral / tail.
 * The sim splits armour by impact bearing, so forward-facing plating is
 * nose, aft-facing is tail and the flanks are lateral; plating near either
 * end leans to that end. Mirrors the overlay shader (tests pin it).
 * @param {number} nz   local normal z (+1 faces the bow)
 * @param {number} zf   local z / half length (-1 stern .. +1 bow)
 */
export function wearSectionWeights(nz, zf, out = [0, 0, 0]) {
  const s = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const nose = s(0.35, 0.8, nz + 0.3 * s(0.55, 0.95, zf));
  const tail = s(0.35, 0.8, -nz + 0.3 * s(0.55, 0.95, -zf));
  const lat = Math.max(0, 1 - nose - tail);
  const sum = Math.max(1e-3, nose + lat + tail);
  out[0] = nose / sum; out[1] = lat / sum; out[2] = tail / sum;
  return out;
}

/** Merge the wear-covered hull parts into one position+normal geometry. */
export function wearGeometry(parts) {
  const list = [];
  for (const p of parts || []) {
    if (p.mat && !WEAR_MATS.has(p.mat)) continue;
    const g = p.geom;
    const pos = g?.attributes?.position, nor = g?.attributes?.normal;
    if (!pos || !nor) continue;
    let b = new THREE.BufferGeometry();
    b.setAttribute('position', pos);
    b.setAttribute('normal', nor);
    if (g.index) { b.setIndex(g.index); b = b.toNonIndexed(); }
    list.push(b);
  }
  if (!list.length) return null;
  const merged = BufferGeometryUtils.mergeGeometries(list, false);
  return merged || null;
}

const WEAR_VERTEX_PARS = /* glsl */`
  uniform float uInflate;
  uniform vec3 uOrigin;
  varying vec3 vWearPos;
  varying vec3 vWearNor;
`;

const WEAR_FRAGMENT_PARS = /* glsl */`
  uniform sampler2D uWearNoise;
  uniform vec3 uLoss;
  uniform float uSoot;
  uniform float uGlow;
  uniform float uSeam;
  uniform float uReactorZ;
  uniform float uSpread;
  uniform float uHalfL;
  uniform float uFreq;
  uniform float uSeed;
  varying vec3 vWearPos;
  varying vec3 vWearNor;
  vec4 wearTri(vec3 p, vec3 n) {
    vec3 w = abs(n);
    w = w * w;
    w /= (w.x + w.y + w.z + 1e-5);
    return texture2D(uWearNoise, p.yz) * w.x + texture2D(uWearNoise, p.xz) * w.y + texture2D(uWearNoise, p.xy) * w.z;
  }
`;

// Runs after <color_fragment>: decides coverage and albedo of the skin
const WEAR_FRAGMENT_MAIN = /* glsl */`
  vec3 wn = normalize(vWearNor);
  float zf = vWearPos.z / uHalfL;
  float wNose = smoothstep(0.35, 0.8, wn.z + 0.3 * smoothstep(0.55, 0.95, zf));
  float wTail = smoothstep(0.35, 0.8, -wn.z + 0.3 * smoothstep(0.55, 0.95, -zf));
  float wLat = max(0.0, 1.0 - wNose - wTail);
  float loss = (uLoss.x * wNose + uLoss.y * wLat + uLoss.z * wTail) / max(1e-3, wNose + wLat + wTail);
  vec3 wp = vWearPos * uFreq;
  vec4 n1 = wearTri(wp + uSeed, wn);                 // big patches (hull-width blobs)
  vec4 n2 = wearTri(wp * 4.0 - uSeed * 1.3, wn);     // ragged edges, pitting
  vec4 n3 = wearTri(wp * 1.7 + uSeed * 0.7, wn);     // soot blotches, cracks
  // patch field, remapped so the stripped area tracks the section's loss;
  // the plating strips where the field falls under it
  float field = smoothstep(0.28, 0.72, n1.b * 0.6 + n1.r * 0.4 + (n2.g - 0.5) * 0.22);
  // Thickness lost is not area lost: even a section ablated to nothing
  // keeps islands of plating and paint
  float target = pow(loss, 1.3) * 0.86;
  float strip = smoothstep(field - 0.04, field + 0.02, target);
  float rim = smoothstep(field - 0.16, field - 0.03, target) * (1.0 - strip);
  float soot = uSoot * smoothstep(0.42, 0.7, n3.b * 0.8 + n2.r * 0.35);
  // ablated armour: heat-darkened, pitted steel; the patch edges carry
  // bronze-to-blue temper colours and soot
  vec3 bare = mix(vec3(0.05, 0.048, 0.046), vec3(0.15, 0.14, 0.13), smoothstep(0.4, 0.8, n2.b)) * (0.85 + 0.3 * n1.g);
  // burn blotches and bright gouged metal break up a fully stripped flank
  bare = mix(bare, vec3(0.022, 0.019, 0.017), smoothstep(0.55, 0.8, n3.b) * 0.75);
  float gouge = (1.0 - smoothstep(0.0, 0.025, abs(n2.a - 0.5))) * smoothstep(0.4, 0.6, n1.r);
  bare += vec3(0.16, 0.155, 0.15) * gouge;
  vec3 temper = mix(vec3(0.2, 0.11, 0.045), vec3(0.06, 0.07, 0.15), smoothstep(0.3, 0.7, n2.r));
  vec3 sootC = vec3(0.025, 0.022, 0.02);
  // dying reactor: cracks glowing through the plating, spreading out from
  // the reactor along the hull as the fuse runs down
  float reach = uSpread * uHalfL;
  float nearReactor = 1.0 - smoothstep(reach, reach + 0.35 * uHalfL, abs(vWearPos.z - uReactorZ));
  float crack = (1.0 - smoothstep(0.0, 0.035, abs(n3.g - 0.5))) * smoothstep(0.3, 0.7, n1.b + 0.25) * nearReactor;
  float glowA = crack * uGlow;
  wearStrip = strip;
  wearGlow = glowA;
  float a = clamp(strip * 0.94 + (rim * 0.7 + soot * 0.78) * (1.0 - strip), 0.0, 0.96);
  // cooled cracks stay as scorched seams (torn sections after the blast)
  a = max(a, max(glowA, crack * uSeam));
  if (a < 0.004) discard;
  vec3 skin = mix(sootC, temper, rim / max(1e-3, rim + soot));
  diffuseColor = vec4(mix(skin, bare, strip), a * diffuseColor.a);   // .a = material opacity
`;

/**
 * The overlay skin of one ship. `update` sets the wear and returns false
 * while there is nothing to show (and nothing has been built).
 */
export class HullWear {
  /**
   * @param {THREE.Group} group  ship group (hull kit with userData.parts)
   * @param {number} seed        stable per-ship seed
   */
  constructor(group, seed = 0) {
    this.group = group;
    const ud = group.userData;
    this.L = ud.size?.length || 8;
    this.W = ud.size?.width || 1.8;
    this.seed = seed;
    this.mesh = null;
    this.failed = false;
    this.uniforms = {
      uWearNoise: { value: null },     // shared noise, fetched on first build
      uLoss: { value: new THREE.Vector3() },
      uSoot: { value: 0 },
      uGlow: { value: 0 },
      uSeam: { value: 0 },             // dark crack seams (torn sections only)
      uReactorZ: { value: ud.reactorPos?.z ?? -0.3 * this.L },
      uSpread: { value: 0.3 },
      uHalfL: { value: this.L * 0.5 },
      uFreq: { value: 0.45 / this.W },
      uSeed: { value: (seed % 997) * 0.0137 },
      uInflate: { value: this.W * 0.0035 },
      uOrigin: { value: new THREE.Vector3() }   // geometry offset from ship-local (torn sections)
    };
  }

  _build() {
    const geom = wearGeometry(this.group.userData.parts);
    if (!geom) { this.failed = true; return; }
    this.uniforms.uWearNoise.value ||= getNoiseTexture();
    this.material = createWearMaterial(this.uniforms);
    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.name = 'hull_wear';
    this.mesh.renderOrder = 5;          // under crater decals (6) and vents
    this.mesh.frustumCulled = false;
    this.mesh.userData.damageFx = true; // not a hull part for DamageFX raycasts
    this.group.add(this.mesh);
  }

  /**
   * @param {{nose, lateral, tail}} armor  remaining fraction per section (0..1)
   * @param {number} soot  0..1
   * @param {number} glow  0..1 dying-reactor crack glow
   * @param {number} spread  how far along the hull the cracks reach from
   *                         the reactor (fraction of the half length,
   *                         DamageTimeline.crackSpread: 0.25 .. 1.85)
   * @returns {boolean} whether the overlay is drawn this frame
   */
  update(armor, soot, glow, spread = 0.3) {
    // Every uniform stays finite whatever the caller passes: GLSL ES leaves
    // NaN undefined, and a driver that propagates it would paint the skin
    // black or NaN (which bloom then smears across the frame)
    const u = this.uniforms;
    u.uLoss.value.set(lossOf(armor?.nose), lossOf(armor?.lateral), lossOf(armor?.tail));
    u.uSoot.value = unit(soot);
    u.uGlow.value = unit(glow);
    u.uSpread.value = Number.isFinite(spread) ? Math.max(0, spread) : 0.3;
    const L = u.uLoss.value;
    const any = Math.max(L.x, L.y, L.z, u.uSoot.value, u.uGlow.value) > WEAR_EPS;
    if (!any) { if (this.mesh) this.mesh.visible = false; return false; }
    if (!this.mesh && !this.failed) this._build();
    if (!this.mesh) return false;
    this.mesh.visible = true;
    return true;
  }

  /**
   * A frozen copy of this skin for a torn hull section (ShipDestruction):
   * the hulk's last armour loss, soot and crack reach on the section's own
   * hull `parts`, whose geometry is shifted by -origin like the section
   * mesh. The noise and section weights still run in ship-local
   * coordinates (uOrigin), so stripped patches and cracks stay where the
   * hulk had them. The reactor has gone: the cracks no longer pulse but
   * stay as dark seams, and the caller drives uGlow from the section's
   * cooling heat (sectionCrackGlow) and fades it with `material.opacity`.
   * Null when there is no wear to carry over (or no geometry); the caller
   * owns the mesh. Its uniforms are `mesh.userData.uniforms`.
   * @param {Array} parts   hull kit parts of the section (ship-local)
   * @param {THREE.Vector3} origin  the section's ship-local centroid
   * @param {{armor, soot, spread}} wear  wear at the detonation (hullWearAt)
   */
  sectionSkin(parts, origin, wear) {
    const a = wear?.armor;
    const loss = [lossOf(a?.nose), lossOf(a?.lateral), lossOf(a?.tail)];
    const soot = unit(wear?.soot);
    if (Math.max(...loss, soot) <= WEAR_EPS) return null;
    const geom = wearGeometry(parts);
    if (!geom) return null;
    geom.translate(-origin.x, -origin.y, -origin.z);
    const u = {};
    // own vectors, shared noise texture (a cloned Texture would re-upload)
    for (const [k, v] of Object.entries(this.uniforms)) u[k] = { value: v.value?.isVector3 ? v.value.clone() : v.value };
    u.uLoss.value.set(...loss);
    u.uSoot.value = soot;
    u.uGlow.value = 0;
    u.uSeam.value = 0.6;
    u.uSpread.value = Number.isFinite(wear?.spread) ? Math.max(0, wear.spread) : 0.3;
    u.uOrigin.value.copy(origin);
    const mesh = new THREE.Mesh(geom, createWearMaterial(u));
    mesh.name = 'hull_wear_section';
    mesh.frustumCulled = false;
    mesh.userData.uniforms = u;
    return mesh;
  }

  dispose() {
    if (!this.mesh) return;
    this.group.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.mesh = null;
  }
}

/**
 * Crack glow 0..1 of a torn section from its cooling heat
 * (ShipDestruction: ~2-3 at the blast, ~0.3 after 10 s): white-hot cracks
 * that fade to the dark seams over the first seconds.
 */
export function sectionCrackGlow(heat) {
  return Number.isFinite(heat) ? Math.min(1, Math.max(0, heat * 0.45)) : 0;
}

/** The shared-program wear material bound to `uniforms` (one per ship). */
export function createWearMaterial(uniforms) {
  if (!uniforms.uWearNoise.value) uniforms.uWearNoise.value = getNoiseTexture();
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: 0.9, metalness: 0.15, envMapIntensity: 0.6,
    emissive: 0x000000, transparent: true, depthWrite: false, depthFunc: THREE.LessEqualDepth
  });
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\n${WEAR_VERTEX_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vWearPos = position + uOrigin;
        vWearNor = normal;
        transformed += normal * uInflate;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\n${WEAR_FRAGMENT_PARS}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        float wearStrip = 0.0, wearGlow = 0.0;
        {
          ${WEAR_FRAGMENT_MAIN}
        }`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix(0.95, 0.58, wearStrip);`)
      .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
        metalnessFactor = mix(0.12, 0.7, wearStrip);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance += vec3(2.6, 0.7, 0.16) * wearGlow * 1.6;`);
  };
  mat.customProgramCacheKey = () => 'hull-wear-v2';
  return mat;
}
