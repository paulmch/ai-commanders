import * as THREE from 'three';
import { getNoiseTexture } from './Textures.js';

/**
 * Heat visuals: radiator deploy/retract, radiator glow, hull overheating.
 *
 * Data (sim_trace ship fields, all optional):
 *   heat     sink fill fraction (may exceed 1)
 *   rad      1 extended / 0 retracted - flips on the frame the command or
 *            the auto-extend takes effect
 *   rad_out  GW the radiators actually rejected that frame
 * Old recordings carry none of these: the deploy state then comes from the
 * radiator_change events and an extended array glows at a nominal output.
 *
 * Everything is a pure function of battle time and the recorded history:
 * deploy progress at t depends only on t - t_flip and the progress the
 * previous flips left (folded once at load), so seeks and scrubs land on
 * the same pose and glow as continuous playback.
 */

/** Panels unfold/fold over this many battle seconds. */
export const RADIATOR_DEPLOY_S = 3.5;
/**
 * Legacy fold: wings without a stow mode fold this far (rad) about their
 * root hinge. Kit radiators (stow 'telescope') instead retract along their
 * span into a cassette on the hull, keeping userData.stowScale of it.
 */
export const RADIATOR_FOLD_RAD = 1.42;
/** Extended LithiumSpray rejection at the operating temperature (60 t x 13 kW/kg). */
export const RAD_FULL_GW = 0.78;
/** TI LithiumSpray radiator operatingTemp_K. */
export const RADIATOR_TEMP_K = 1500;
/** Nominal output of an extended array in recordings without rad_out. */
export const LEGACY_RAD_FRACTION = 0.4;
/** Heat fraction at which reactor containment fails (REACTOR_MELTDOWN_HEAT_FRACTION). */
export const MELTDOWN_HEAT = 1.5;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------------------
// Timelines (built once per recording, rebuilt on live append)
// ---------------------------------------------------------------------------

/**
 * Per-ship heat/radiator timeline from the trace, falling back to
 * radiator_change events for the deploy state of old recordings.
 * @returns {Map<string, {t, heat, radOut, hasHeat, hasRad, hasRadOut, maxHeat, initial, flips}>}
 *   flips: [{t, ext, p0, radOut}] - p0 = deploy progress at the flip,
 *   radOut = output (GW) on the last frame before it (the retract fade).
 */
export function buildHeatTimelines(trace = [], events = []) {
  const raw = new Map();
  for (const f of trace) {
    for (const id in f.ships) {
      let r = raw.get(id);
      if (!r) raw.set(id, r = { t: [], heat: [], radOut: [], rad: [], hasHeat: false, hasRad: false, hasRadOut: false });
      const s = f.ships[id];
      r.t.push(f.t);
      const h = typeof s.heat === 'number' ? s.heat : NaN;
      const ro = typeof s.rad_out === 'number' ? s.rad_out : NaN;
      r.heat.push(h);
      r.radOut.push(ro);
      r.rad.push(s.rad === 1 || s.rad === true ? 1 : s.rad === 0 || s.rad === false ? 0 : -1);
      if (h === h) r.hasHeat = true;
      if (ro === ro) r.hasRadOut = true;
      if (s.rad !== undefined && s.rad !== null) r.hasRad = true;
    }
  }
  const legacy = new Map();
  for (const e of events || []) {
    if (e.event_type !== 'radiator_change' || !e.ship_id) continue;
    if (!legacy.has(e.ship_id)) legacy.set(e.ship_id, []);
    legacy.get(e.ship_id).push({ t: e.timestamp, ext: !!e.data?.extended });
  }
  const out = new Map();
  const ids = new Set([...raw.keys(), ...legacy.keys()]);
  for (const id of ids) {
    const r = raw.get(id) || { t: [], heat: [], radOut: [], rad: [], hasHeat: false, hasRad: false, hasRadOut: false };
    const n = r.t.length;
    const tl = {
      t: Float64Array.from(r.t),
      heat: Float32Array.from(r.heat),
      radOut: Float32Array.from(r.radOut),
      hasHeat: r.hasHeat, hasRad: r.hasRad, hasRadOut: r.hasRadOut,
      maxHeat: 0, initial: 0, flips: []
    };
    for (let i = 0; i < n; i++) if (tl.heat[i] > tl.maxHeat) tl.maxHeat = tl.heat[i];
    let changes;
    if (r.hasRad) {
      let i0 = r.rad.findIndex(v => v >= 0);
      tl.initial = i0 >= 0 ? r.rad[i0] : 0;
      changes = [];
      let cur = tl.initial;
      const orders = legacy.get(id) || [];
      for (let i = i0 + 1; i0 >= 0 && i < n; i++) {
        const v = r.rad[i];
        if (v < 0 || v === cur) continue;
        cur = v;
        const prev = r.radOut[i - 1];
        // The trace only shows the flip on the first frame after it: a
        // matching radiator_change order since the previous frame dates it
        // exactly (the checkpoint, in step with the decision log)
        const ord = orders.find(o => o.ext === (v === 1) && o.t >= r.t[i - 1] && o.t <= r.t[i]);
        changes.push({ t: ord ? ord.t : r.t[i], ext: v === 1, radOut: prev === prev ? prev : 0 });
      }
    } else {
      changes = (legacy.get(id) || []).slice().sort((a, b) => a.t - b.t)
        .map(c => ({ t: c.t, ext: c.ext, radOut: NaN }));
    }
    // Fold the history: progress each flip starts from
    let p = tl.initial ? 1 : 0, ext = !!tl.initial, tPrev = null;
    for (const c of changes) {
      if (c.ext === ext) continue;          // repeated order: no change
      if (tPrev !== null) p = advance(p, ext, c.t - tPrev);
      tl.flips.push({ t: c.t, ext: c.ext, p0: p, radOut: c.radOut });
      ext = c.ext;
      tPrev = c.t;
    }
    out.set(id, tl);
  }
  return out;
}

function advance(p, ext, dt) {
  return ext ? Math.min(1, p + dt / RADIATOR_DEPLOY_S) : Math.max(0, p - dt / RADIATOR_DEPLOY_S);
}

/** Index of the last element of sorted `arr` (length n) with arr[i] <= t, or -1. */
function lastAtOrBefore(arr, n, t) {
  if (!n || arr[0] > t) return -1;
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (arr[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Index of the last flip at or before t (-1 before the first). */
export function flipIndexAt(tl, t) {
  const f = tl.flips, n = f.length;
  if (!n || f[0].t > t) return -1;
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (f[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/**
 * Radiator deploy progress 0 (stowed) .. 1 (extended) at battle time t:
 * a pure function of t - t_flip and the progress at that flip.
 */
export function deployProgress(tl, t) {
  if (!tl) return 0;
  const i = flipIndexAt(tl, t);
  if (i < 0) return tl.initial ? 1 : 0;
  const f = tl.flips[i];
  return advance(f.p0, f.ext, t - f.t);
}

/** Commanded state (true = extended) at battle time t. */
export function radiatorsExtendedAt(tl, t) {
  if (!tl) return false;
  const i = flipIndexAt(tl, t);
  return i < 0 ? !!tl.initial : tl.flips[i].ext;
}

/** Linear sample of a per-frame series; NaN before the first frame / without data. */
export function sampleSeries(tl, series, t) {
  const n = tl.t.length;
  const i = lastAtOrBefore(tl.t, n, t);
  if (i < 0) return n ? series[0] : NaN;
  if (i >= n - 1) return series[n - 1];
  const a = series[i], b = series[i + 1];
  if (a !== a) return b;
  if (b !== b) return a;
  const f = (t - tl.t[i]) / (tl.t[i + 1] - tl.t[i]);
  return a + (b - a) * f;
}

/** Heat fraction at battle time t (NaN when the recording has none). */
export function heatAt(tl, t) {
  return tl && tl.hasHeat ? sampleSeries(tl, tl.heat, t) : NaN;
}

/**
 * Radiator output as a fraction of full rejection (RAD_FULL_GW) that the
 * glow should show at t, before the deploy scaling: the recorded rad_out
 * while extended, the last output before a retract while the panels fold,
 * a nominal level for recordings without rad_out.
 */
export function radiatorFraction(tl, t) {
  if (!tl) return 0;
  const i = flipIndexAt(tl, t);
  const ext = i < 0 ? !!tl.initial : tl.flips[i].ext;
  if (!tl.hasRadOut) return LEGACY_RAD_FRACTION;
  if (ext) {
    const v = sampleSeries(tl, tl.radOut, t);
    return v === v ? Math.max(0, v) / RAD_FULL_GW : 0;
  }
  const last = i < 0 ? 0 : tl.flips[i].radOut;
  return last === last ? Math.max(0, last) / RAD_FULL_GW : 0;
}

/** Hinge ease for the panels: slow break-away, slow latch. */
export function easeDeploy(p) {
  const x = clamp(p, 0, 1);
  return x * x * (3 - 2 * x);
}

// ---------------------------------------------------------------------------
// Glow mapping
// ---------------------------------------------------------------------------

/**
 * Panel temperature for an output fraction: a fixed-area radiator rejects
 * P = e*sigma*A*T^4, so T = T_op * f^(1/4) (1500 K at full rejection).
 */
export function radiatorTemperature(fraction) {
  return RADIATOR_TEMP_K * Math.pow(Math.max(0, fraction), 0.25);
}

// Linear-RGB emission ramp by panel temperature (K): Draper-point cherry
// red to orange-yellow at the operating point. Warmer than a strict
// Planck ramp at the top so full rejection reads orange-yellow after ACES.
const RAMP = [
  [700, 1.0, 0.03, 0.0],
  [850, 1.0, 0.05, 0.002],
  [1000, 1.0, 0.1, 0.006],
  [1200, 1.0, 0.2, 0.015],
  [1350, 1.0, 0.33, 0.035],
  [1500, 1.0, 0.52, 0.07],
  [1700, 1.0, 0.64, 0.14]
];

/** Blackbody-style emission colour (linear RGB, red channel 1) into out {r,g,b}. */
export function glowColor(tempK, out) {
  let i = 0;
  while (i < RAMP.length - 2 && tempK > RAMP[i + 1][0]) i++;
  const a = RAMP[i], b = RAMP[i + 1];
  const f = clamp((tempK - a[0]) / (b[0] - a[0]), 0, 1);
  out.r = a[1] + (b[1] - a[1]) * f;
  out.g = a[2] + (b[2] - a[2]) * f;
  out.b = a[3] + (b[3] - a[3]) * f;
  return out;
}

/**
 * Radiator emissive for an output fraction (rad_out / RAD_FULL_GW) into
 * out {r, g, b, intensity}: dark at zero, dull cherry red at a trickle,
 * bright orange-yellow near full 780 MW (the root of each fin then passes
 * the bloom threshold). Any output at all stays visible.
 */
export function radiatorGlow(fraction, out = { r: 0, g: 0, b: 0, intensity: 0 }) {
  const f = clamp(fraction || 0, 0, 1.25);
  if (f < 0.002) {
    out.r = out.g = out.b = 0;
    out.intensity = 0;
    return out;
  }
  glowColor(radiatorTemperature(f), out);
  out.intensity = 0.55 + 2.6 * Math.pow(f, 0.8);
  return out;
}

/**
 * Hull heat look at heat fraction h and battle time t into out:
 *   level  overlay brightness (0 below ~88%)
 *   warn   0..1 from 100% to the meltdown threshold (colour shifts red)
 *   reach  how far forward of the reactor the glow spreads (hull lengths)
 * 88-100%: a faint, patchy shimmer on the drive section. Above 100% a red-orange
 * warning pulse (1.25 Hz on battle time) that brightens and spreads
 * toward the bow as heat nears meltdown.
 */
export function hullHeatLook(h, t, melt = MELTDOWN_HEAT, out = { level: 0, warn: 0, reach: 0, pulse: 0 }) {
  if (!(h > 0)) {
    out.level = out.warn = out.pulse = 0;
    out.reach = 0.18;
    return out;
  }
  const shimmer = smoothstep(0.88, 1.0, h) * 0.09;
  const warn = h > 1 ? clamp((h - 1) / Math.max(0.01, melt - 1), 0, 1) : 0;
  const pulse = 0.5 - 0.5 * Math.cos(2 * Math.PI * 1.25 * t);
  out.warn = warn;
  out.pulse = pulse;
  out.level = shimmer + (warn > 0 ? (0.25 + 1.4 * warn) * (0.3 + 0.7 * pulse) * Math.min(1, warn * 8) : 0);
  out.reach = 0.18 + 0.5 * warn;
  return out;
}

// ---------------------------------------------------------------------------
// Hull overheating overlay: the plating re-drawn additively, hot near the
// reactor/drive, noise-shimmered on battle time
// ---------------------------------------------------------------------------
const OVERLAY_SOURCES = new Set(['hull_hull', 'hull_paint', 'hull_trim']);

function heatOverlayMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uNoise: { value: getNoiseTexture() },
      uTime: { value: 0 },
      uLevel: { value: 0 },
      uWarn: { value: 0 },
      uZHot: { value: 0 },
      uReach: { value: 0.2 },
      uL: { value: 8 },
      uPush: { value: 0.01 }
    },
    vertexShader: /* glsl */`
      uniform float uPush;
      varying vec3 vLocal;
      varying vec3 vN;
      varying vec3 vV;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vLocal = position;
        vec4 mv = modelViewMatrix * vec4(position + normal * uPush, 1.0);
        vN = normalize(normalMatrix * normal);
        vV = -mv.xyz;
        gl_Position = projectionMatrix * mv;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uNoise;
      uniform float uTime, uLevel, uWarn, uZHot, uReach, uL;
      varying vec3 vLocal;
      varying vec3 vN;
      varying vec3 vV;
      #include <common>
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        // hot aft of the reactor, falling off toward the bow
        float fwd = max(0.0, vLocal.z - uZHot) / (uL * uReach);
        float w = exp(-fwd * fwd * 2.0);
        // shimmer: two noise layers drifting on battle time
        vec2 q = vec2(vLocal.x * 0.7 + vLocal.y, vLocal.z) * (5.0 / uL);
        float n1 = texture2D(uNoise, q + vec2(0.13, -uTime * 0.22)).r;
        float n2 = texture2D(uNoise, q * 2.7 + vec2(uTime * 0.09, -uTime * 0.41)).g;
        float shimmer = 0.15 + 1.7 * smoothstep(0.35, 0.8, n1 * 0.65 + n2 * 0.35);
        float ndv = abs(dot(normalize(vN), normalize(vV)));
        float rim = pow(1.0 - ndv, 2.0);
        vec3 col = mix(vec3(1.0, 0.36, 0.09), vec3(1.0, 0.14, 0.03), uWarn);
        float a = uLevel * w * (0.35 + 1.1 * rim) * shimmer;
        gl_FragColor = vec4(col * a, 1.0);
      }`,
    transparent: true,
    depthWrite: false,
    depthFunc: THREE.LessEqualDepth,
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendEquation: THREE.AddEquation
  });
}

/**
 * Drives radiators and hull heat for every ship; owned by SceneManager
 * (`sm.heat`), fed by `sm.loader` (BattleLoader) when present.
 */
export class HeatFX {
  constructor(sm) {
    this.sm = sm;
    this.timelines = null;
    this._trace = null; this._traceLen = -1; this._events = null; this._eventLen = -1;
    this.overlays = new Map();   // ship group -> {material, meshes}
    this.melt = MELTDOWN_HEAT;
    this._glow = { r: 0, g: 0, b: 0, intensity: 0 };
    this._look = { level: 0, warn: 0, reach: 0, pulse: 0 };
  }

  /** (Re)build the timelines when the recording changed or grew. */
  prepare() {
    const loader = this.sm.loader;
    const trace = loader?.simTrace, events = loader?.events;
    if (!trace) { this.timelines = null; return null; }
    if (trace === this._trace && trace.length === this._traceLen
      && events === this._events && (events?.length ?? 0) === this._eventLen) return this.timelines;
    this._trace = trace; this._traceLen = trace.length;
    this._events = events; this._eventLen = events?.length ?? 0;
    this.timelines = buildHeatTimelines(trace, events || []);
    return this.timelines;
  }

  timeline(id) {
    return this.prepare()?.get(id) || null;
  }

  /** Heat fraction of a ship at t, NaN when the recording has none. */
  heatAt(id, t) {
    return heatAt(this.timeline(id), t);
  }

  /** Whether this recording carries heat telemetry for the ship. */
  hasHeat(id) {
    return !!this.timeline(id)?.hasHeat;
  }

  /**
   * Called from SceneManager.updateShip for every live ship. Ships outside
   * the recording (showroom) take `state.radiatorsExtended`, and optionally
   * `state.radDeploy` (0..1), `state.radOut` (GW) and `state.heat`.
   */
  updateShip(ship, state) {
    const ud = ship.userData;
    const t = this.sm.currentTime || 0;
    const tl = this.timeline(ud.shipId);
    const recorded = tl && (tl.hasRad || tl.flips.length || tl.hasRadOut);
    let p, frac;
    if (recorded) {
      p = deployProgress(tl, t);
      frac = radiatorFraction(tl, t);
    } else {
      p = typeof state.radDeploy === 'number' ? state.radDeploy : (state.radiatorsExtended ? 1 : 0);
      frac = typeof state.radOut === 'number' ? state.radOut / RAD_FULL_GW : LEGACY_RAD_FRACTION;
    }
    const e = easeDeploy(p);
    const panels = ud.radiators;
    if (panels) {
      // rotation.z / scale.y only: DamageFX owns rotation.x (buckled wings)
      for (let i = 0; i < panels.length; i++) {
        const u = panels[i].userData;
        if (u.stow === 'telescope') {
          panels[i].rotation.z = u.deployedAngle;
          panels[i].scale.y = u.stowScale + (1 - u.stowScale) * e;
        } else {
          panels[i].rotation.z = u.deployedAngle + u.foldDirection * RADIATOR_FOLD_RAD * (1 - e);
        }
      }
    }
    const glow = radiatorGlow(frac, this._glow);
    const mats = ud.radiatorMaterials;
    if (mats) {
      for (let i = 0; i < mats.length; i++) {
        mats[i].emissive.setRGB(glow.r, glow.g, glow.b);
        mats[i].emissiveIntensity = glow.intensity * e;
      }
    }

    // Hull: shimmer above ~90 %, warning pulse above 100 %
    const h = tl?.hasHeat ? heatAt(tl, t) : (typeof state.heat === 'number' ? state.heat : NaN);
    const look = hullHeatLook(h, t, this.melt, this._look);
    let ov = this.overlays.get(ship);
    if (look.level > 0.004 && !ov) ov = this._overlay(ship);
    if (ov) {
      const on = look.level > 0.004;
      const u = ov.material.uniforms;
      u.uLevel.value = look.level;
      u.uWarn.value = look.warn;
      u.uReach.value = look.reach;
      u.uTime.value = t;
      for (let i = 0; i < ov.meshes.length; i++) ov.meshes[i].visible = on;
    }
  }

  /** A destroyed or hidden ship drops its overheating overlay. */
  hide(ship) {
    const ov = this.overlays.get(ship);
    if (ov) for (const m of ov.meshes) m.visible = false;
  }

  _overlay(ship) {
    const ud = ship.userData;
    const material = heatOverlayMaterial();
    const L = ud.size?.length || 8, W = ud.size?.width || 1.8;
    material.uniforms.uL.value = L;
    material.uniforms.uZHot.value = ud.reactorPos ? ud.reactorPos.z : -0.3 * L;
    material.uniforms.uPush.value = W * 0.004;
    const sources = ship.children.filter(c => c.isMesh && OVERLAY_SOURCES.has(c.name));
    const meshes = sources.map(src => {
      const m = new THREE.Mesh(src.geometry, material);   // shares the hull geometry
      m.name = 'heat_overlay';
      m.renderOrder = 30;
      m.visible = false;
      ship.add(m);
      return m;
    });
    const ov = { material, meshes };
    this.overlays.set(ship, ov);
    return ov;
  }

  /** A throwaway mesh carrying the overlay shader, for SceneManager.warmUp. */
  warmUpProbe(position) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), heatOverlayMaterial());
    m.position.copy(position);
    return m;
  }

  dispose() {
    for (const [ship, ov] of this.overlays) {
      for (const m of ov.meshes) ship.remove(m);
      ov.material.dispose();
    }
    this.overlays.clear();
    this.timelines = null;
  }
}
