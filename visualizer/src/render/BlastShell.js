import * as THREE from 'three';
import { blastsAt, blastFront, BLAST_EXPAND_S } from '../DamageTimeline.js';

/**
 * The reach of a reactor detonation, drawn from the recorded `reactor_blast`
 * event (timeline.blasts): a plasma front that runs out to exactly
 * `radius_km` and hangs there as a cooling bubble, so a spectator sees which
 * ships and torpedoes were inside it, plus a short, hotter inner sphere at
 * the torpedo kill radius. This is the true-scale lethal volume; the big
 * ragged shock shell of the death blast (Explosion.js) stays the spectacle.
 *
 * Everything is a pure function of battle time: the pool is refilled from
 * the timeline every frame, so a seek shows the same shells.
 */

/** Seconds the front takes to reach the blast radius. */
export const SHELL_EXPAND_S = BLAST_EXPAND_S;
/** Seconds the bubble stays visible in all. */
export const SHELL_LIFE_S = 5.0;
/** The torpedo-kill sphere: fast out, gone quickly. */
export const KILL_EXPAND_S = 0.4;
export const KILL_LIFE_S = 1.6;
/** Peak rim radiance (scene HDR units) of the outer front / the kill sphere. */
export const SHELL_LEVEL = 1.5;
export const KILL_LEVEL = 2.2;

/**
 * Outer shell `age` seconds after the detonation: radius (same unit as
 * `radius`), level 0..1 and how far it has cooled 0..1. The radius is
 * monotonic and equals `radius` from SHELL_EXPAND_S on.
 */
export function blastShellAt(age, radius, out = {}, expand = SHELL_EXPAND_S, life = SHELL_LIFE_S) {
  out.r = 0; out.level = 0; out.cool = 0;
  if (!(age >= 0) || age >= life || !(radius > 0)) return out;
  const p = Math.min(1, age / expand);
  out.r = blastFront(age, radius, expand);
  const rise = Math.min(1, age / 0.06);
  const hold = age <= expand ? 1 : Math.exp(-(age - expand) / (0.3 * (life - expand)));
  const tail = 1 - Math.min(1, Math.max(0, (age - (life - 1)) / 1));
  out.level = rise * (age <= expand ? 1 - 0.4 * p : 0.6 * hold) * tail;
  out.cool = Math.min(1, age / (expand + 0.35 * (life - expand)));
  return out;
}

const vertexShader = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vView;
  varying vec3 vLocal;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vLocal = normalize(position);
    vNormal = normalize(normalMatrix * normal);
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vView = -(viewMatrix * wp).xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
    #include <logdepthbuf_vertex>
  }
`;

const fragmentShader = /* glsl */`
  uniform vec3 uHot;
  uniform vec3 uCool;
  uniform float uLevel;
  uniform float uCoolK;
  uniform float uSeed;
  varying vec3 vNormal;
  varying vec3 vView;
  varying vec3 vLocal;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float ndv = abs(dot(normalize(vNormal), normalize(vView)));
    // limb-brightened: the bubble's outline is the blast radius
    float rim = pow(1.0 - ndv, 4.5);
    float limb = pow(1.0 - ndv, 12.0);
    // slow ripples over the skin so it reads as a surface, not a HUD ring
    float a = vLocal.y * 17.0 + sin(vLocal.x * 7.0 + uSeed * 9.0) * 2.2 + sin(vLocal.z * 5.0 - uSeed * 4.0) * 1.7;
    float rip = 0.7 + 0.3 * sin(a);
    vec3 col = mix(uHot, uCool, uCoolK) * (rim * rip * 0.45 + limb * 1.0 + 0.012);
    gl_FragColor = vec4(col * uLevel, 1.0);
  }
`;

let _geom = null;
function shellGeometry() { return _geom || (_geom = new THREE.IcosahedronGeometry(1, 16)); }

function shellMaterial(hot, cool) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uHot: { value: new THREE.Color(...hot) },
      uCool: { value: new THREE.Color(...cool) },
      uLevel: { value: 0 },
      uCoolK: { value: 0 },
      uSeed: { value: 0 }
    },
    vertexShader, fragmentShader,
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
  });
}

/** Blasts drawn at once (newest first); each uses an outer and a kill mesh. */
export const SHELL_SLOTS = 4;

export class BlastShells {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.userData.damageFx = true;
    this.slots = [];
    for (let i = 0; i < SHELL_SLOTS; i++) {
      const outer = new THREE.Mesh(shellGeometry(), shellMaterial([1.45, 1.25, 1.0], [1.3, 0.38, 0.08]));
      const kill = new THREE.Mesh(shellGeometry(), shellMaterial([1.5, 1.45, 1.5], [1.5, 0.28, 0.12]));
      for (const m of [outer, kill]) {
        m.visible = false; m.frustumCulled = false; m.renderOrder = 35;
        this.group.add(m);
      }
      this.slots.push({ outer, kill });
    }
    scene.add(this.group);
    this._range = { lo: 0, hi: 0 };
    this._s = {};
    this.active = 0;
  }

  /** A mesh carrying the shell program, for the load-time shader warm-up. */
  warmMesh() {
    const m = new THREE.Mesh(shellGeometry(), this.slots[0].outer.material);
    m.frustumCulled = false;
    return m;
  }

  /**
   * @param {Array} blasts  timeline.blasts (sorted by time; pos in metres)
   * @param {number} now    battle time
   * @param {number} scale  metres -> scene units (radii are km = scene units)
   */
  update(blasts, now, scale) {
    const r = blastsAt(blasts, now, SHELL_LIFE_S, this._range);
    let slot = 0;
    for (let i = r.hi - 1; i >= r.lo && slot < SHELL_SLOTS; i--, slot++) {
      const b = blasts[i], S = this.slots[slot], age = now - b.t;
      const o = blastShellAt(age, b.radius, this._s);
      this._place(S.outer, b, o, scale, SHELL_LEVEL);
      const k = blastShellAt(age, b.kill, this._s, KILL_EXPAND_S, KILL_LIFE_S);
      this._place(S.kill, b, k, scale, KILL_LEVEL);
    }
    this.active = slot;
    for (; slot < SHELL_SLOTS; slot++) { this.slots[slot].outer.visible = false; this.slots[slot].kill.visible = false; }
  }

  _place(mesh, b, s, scale, level) {
    mesh.visible = s.level > 0.004;
    if (!mesh.visible) return;
    mesh.position.set(b.pos[0] * scale, b.pos[1] * scale, b.pos[2] * scale);
    mesh.scale.setScalar(s.r);
    const u = mesh.material.uniforms;
    u.uLevel.value = s.level * level;
    u.uCoolK.value = s.cool;
    u.uSeed.value = b.seed % 97 / 97;
  }

  dispose() {
    this.scene.remove(this.group);
    for (const S of this.slots) { S.outer.material.dispose(); S.kill.material.dispose(); }
    this.slots = [];
  }
}
