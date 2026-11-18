import * as THREE from 'three';
import { closeUpK } from './CloseRange.js';
import { sampleTrack, makeSample } from './WeaponTimeline.js';
import { factionSeed } from './WeaponBatches.js';
import { Ribbon } from './Trails.js';
import { clusterPoints, CLUSTER_PX } from '../TacticalLayout.js';

/**
 * Engagement lines: at fleet range, a thin dashed faction-coloured line
 * from every fighting ship to the ship it is engaging, the dashes marching
 * toward the target and the line flaring for a second after each shot or
 * torpedo launch. Up close they fade out (the weapons themselves read).
 *
 * Drawn into the Projectiles ribbon batch (kind 8), so they cost no extra
 * draw; every input is recording-derived at the requested battle time.
 *
 * Weight follows the label density (TacticalLayout, L key):
 *   auto     ships of one side that sit together on screen (CLUSTER_PX, the
 *            label layer's grouping) and engage the same target share ONE
 *            line, thicker with every gun ship in it; faint unless it
 *            touches the selected / hovered ship or just fired, fainter
 *            still for a ship holding an ordered target it has not shot at
 *   minimal  only the selected / hovered ship's lines and the flare of a shot
 *   full     a line per ship at the pass-4 weight
 */

/** Seconds a shot or launch keeps a ship's line flared. */
export const ENGAGE_HOT_S = 1.0;
/** How long a shot / launch still names the target when the trace has no 'tgt'. */
export const SHOT_MEMORY_S = 30;
export const LAUNCH_MEMORY_S = 60;
// Screen gap (px) left at each end so a line never runs into the brackets
const END_GAP_PX = 14;
const DASH_PX = 11;
/** Line strength (before the shot flare) by role, auto density. */
export const LINE_WEIGHT = { focus: 0.6, active: 0.14, idle: 0.06, full: 0.32 };
/** Half-width (px) of a one-ship line and the most a shared line grows to. */
const LINE_PX = 0.75, LINE_PX_MAX = 2.2;

/**
 * Who each live ship engages at battle time `t` (pure).
 *
 * Target, in order: the recorded fire-control target (trace 'tgt'), the
 * target of its latest shot (<= SHOT_MEMORY_S ago) or torpedo launch
 * (<= LAUNCH_MEMORY_S), else the captain's ordered target. Dead, dying,
 * missing and same-side targets are dropped, and so are dead or dying
 * shooters.
 *
 * @param {Array<{id, faction, alive, tgt}>} ships   alive: drawn, not dying/destroyed
 * @param {(id) => ({t, target, torpedo}|null)} lastFire   latest shot/launch at or before t
 * @param {number} t
 * @param {(id) => (string|null)} [ordered]   captain's ordered target
 * @returns {Array<{from, to, faction, heat}>}  heat 1 right after a shot, 0 after ENGAGE_HOT_S
 */
export function selectEngagements(ships, lastFire, t, ordered = null) {
  const byId = new Map(ships.map(s => [s.id, s]));
  const out = [];
  for (const s of ships) {
    if (!s.alive) continue;
    const f = lastFire(s.id);
    let to = s.tgt || null;
    if (!to && f) {
      const memory = f.torpedo ? LAUNCH_MEMORY_S : SHOT_MEMORY_S;
      if (t - f.t <= memory) to = f.target;
    }
    if (!to && ordered) to = ordered(s.id) || null;
    const target = to ? byId.get(to) : null;
    if (!target || !target.alive || target.faction === s.faction) continue;
    const age = f && f.target === to ? t - f.t : Infinity;
    const heat = age >= 0 && age < ENGAGE_HOT_S ? (1 - age / ENGAGE_HOT_S) ** 2 : 0;
    out.push({ from: s.id, to, faction: s.faction, heat });
  }
  return out;
}

/**
 * One line per (side, target, on-screen cluster of shooters) (pure).
 * @param {Array<{from, to, faction, heat, active?}>} pairs - selectEngagements output (+ active: fired lately)
 * @param {(id) => ({x, y}|null)} screenOf - shooter's screen position (px), null off screen
 * @param {number} [radius] - px: shooters closer than this share the line
 * @returns {Array<{from: string[], to, faction, heat, active, n}>} heat / active: the strongest member's
 */
export function aggregateEngagements(pairs, screenOf, radius = CLUSTER_PX) {
  const byKey = new Map();
  for (const e of pairs) {
    const key = `${e.faction}>${e.to}`;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push(e);
  }
  const out = [];
  for (const list of byKey.values()) {
    const pts = [], loose = [];
    for (const e of list) {
      const p = screenOf(e.from);
      if (p) pts.push({ id: e.from, faction: e.faction, x: p.x, y: p.y, e });
      else loose.push([{ e }]);
    }
    for (const c of [...clusterPoints(pts, radius), ...loose]) {
      out.push({ from: c.map(p => p.e.from), to: c[0].e.to, faction: c[0].e.faction, n: c.length,
        heat: Math.max(...c.map(p => p.e.heat)), active: c.some(p => p.e.active !== false) });
    }
  }
  return out;
}

/**
 * Strength (before the close-up fade) and half-width (px) of an aggregated
 * line under a label density; strength 0 = not drawn.
 */
export function engagementStyle(line, mode, focusId = null) {
  const focus = !!focusId && (line.to === focusId || line.from.includes(focusId));
  const width = Math.min(LINE_PX_MAX, LINE_PX + 0.35 * (line.n - 1));
  if (mode === 'full') return { strength: LINE_WEIGHT.full + 1.1 * line.heat, width: LINE_PX };
  if (mode === 'minimal') return { strength: (focus ? LINE_WEIGHT.focus : 0) + 1.1 * line.heat, width };
  const base = focus ? LINE_WEIGHT.focus : line.active ? LINE_WEIGHT.active : LINE_WEIGHT.idle;
  return { strength: base + 1.1 * line.heat, width };
}

export class TargetingLines {
  /** @param {import('./Projectiles.js').Projectiles} projectiles */
  constructor(projectiles) {
    this.p = projectiles;
    this.enabled = true;     // ContactOverlay mirrors its C toggle here
    this.mode = 'auto';      // label density (main.js setLabelMode)
    this.focusId = null;     // selected / hovered ship (main.js, each frame)
    this.sample = makeSample();
    this.a = new THREE.Vector3();
    this.b = new THREE.Vector3();
    this.d = new THREE.Vector3();
    this.m = new THREE.Vector3();
    this.v = new THREE.Vector3();
    this.list = [];
  }

  /** Add this frame's lines to the ribbon batch (between begin/end). */
  draw(time) {
    const p = this.p, tl = p.timeline, mgr = p.manager;
    const k = 1 - closeUpK(p.viewDist);
    if (!this.enabled || !tl || k <= 0.01) return;
    const list = this.list;
    list.length = 0;
    for (const [id, group] of mgr.ships) {
      const track = tl.ships.get(id);
      const ok = group.visible && track && sampleTrack(track, time, this.sample);
      const st = ok ? this.sample.state : null;
      list.push({
        id, faction: group.userData.faction || id.split('_')[0],
        alive: !!st && !st.destroyed && !st.dying && !mgr.destroyedShips?.has(id),
        tgt: st?.tgt || null
      });
    }
    const loader = mgr.loader;
    const pairs = selectEngagements(list, id => tl.lastFire(id, time), time,
      loader?.getShipTargetAt ? (id) => loader.getShipTargetAt(id, time)?.target_id : null);
    for (const e of pairs) {
      const f = tl.lastFire(e.from, time);
      e.active = !!f && time - f.t <= (f.torpedo ? LAUNCH_MEMORY_S : SHOT_MEMORY_S);
    }
    // shooters' screen positions (px, the contact layer's scale) for the grouping
    const cam = mgr.camera;
    cam.updateMatrixWorld();
    const halfH = (globalThis.innerHeight || 720) / 2, halfW = halfH * (cam.aspect || 1);
    const lines = this.mode === 'full'
      ? pairs.map(e => ({ from: [e.from], to: e.to, faction: e.faction, n: 1, heat: e.heat, active: e.active }))
      : aggregateEngagements(pairs, (id) => {
        const g = mgr.ships.get(id);
        if (!g) return null;
        this.v.copy(g.position).project(cam);
        if (this.v.z > 1) return null;
        return { x: this.v.x * halfW, y: -this.v.y * halfH };
      });
    for (const line of lines) {
      const { strength, width } = engagementStyle(line, this.mode, this.focusId);
      if (strength <= 0.01) continue;
      const to = mgr.ships.get(line.to);
      if (!to) continue;
      // a shared line starts at its shooters' centroid
      this.m.set(0, 0, 0);
      let n = 0;
      for (const id of line.from) {
        const g = mgr.ships.get(id);
        if (g) { this.m.add(g.position); n++; }
      }
      if (!n) continue;
      const from = this.m.divideScalar(n);
      this.d.subVectors(to.position, from);
      const len = this.d.length();
      if (len < 1e-3) continue;
      this.d.divideScalar(len);
      const gapA = END_GAP_PX * p.pixelAt(from), gapB = END_GAP_PX * p.pixelAt(to.position);
      if (len <= gapA + gapB) continue;
      this.a.copy(from).addScaledVector(this.d, gapA);
      this.b.copy(to.position).addScaledVector(this.d, -gapB);
      // dash count from the on-screen length (measured at the midpoint)
      this.m.addVectors(this.a, this.b).multiplyScalar(0.5);
      const pw = p.pixelAt(this.m);
      const px = this.a.distanceTo(this.b) / Math.max(1e-6, pw);
      const dashes = Math.max(2, Math.round(px / DASH_PX));
      p.ribbons.add(this.a, this.b, width > LINE_PX ? width * pw : 0.0001, k * strength, 8,
        factionSeed(0.5, line.faction), dashes, dashes);
    }
  }
}
