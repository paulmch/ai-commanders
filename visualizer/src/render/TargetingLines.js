import * as THREE from 'three';
import { closeUpK } from './CloseRange.js';
import { sampleTrack, makeSample } from './WeaponTimeline.js';
import { factionSeed } from './WeaponBatches.js';
import { Ribbon } from './Trails.js';

/**
 * Engagement lines: at fleet range, a thin dashed faction-coloured line
 * from every fighting ship to the ship it is engaging, the dashes marching
 * toward the target and the line flaring for a second after each shot or
 * torpedo launch. Up close they fade out (the weapons themselves read).
 *
 * Drawn into the Projectiles ribbon batch (kind 8), so they cost no extra
 * draw; every input is recording-derived at the requested battle time.
 */

/** Seconds a shot or launch keeps a ship's line flared. */
export const ENGAGE_HOT_S = 1.0;
/** How long a shot / launch still names the target when the trace has no 'tgt'. */
export const SHOT_MEMORY_S = 30;
export const LAUNCH_MEMORY_S = 60;
// Screen gap (px) left at each end so a line never runs into the brackets
const END_GAP_PX = 14;
const DASH_PX = 11;

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

export class TargetingLines {
  /** @param {import('./Projectiles.js').Projectiles} projectiles */
  constructor(projectiles) {
    this.p = projectiles;
    this.enabled = true;     // ContactOverlay mirrors its C toggle here
    this.sample = makeSample();
    this.a = new THREE.Vector3();
    this.b = new THREE.Vector3();
    this.d = new THREE.Vector3();
    this.m = new THREE.Vector3();
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
      const from = mgr.ships.get(e.from), to = mgr.ships.get(e.to);
      if (!from || !to) continue;
      this.d.subVectors(to.position, from.position);
      const len = this.d.length();
      if (len < 1e-3) continue;
      this.d.divideScalar(len);
      const gapA = END_GAP_PX * p.pixelAt(from.position), gapB = END_GAP_PX * p.pixelAt(to.position);
      if (len <= gapA + gapB) continue;
      this.a.copy(from.position).addScaledVector(this.d, gapA);
      this.b.copy(to.position).addScaledVector(this.d, -gapB);
      // dash count from the on-screen length (measured at the midpoint)
      this.m.addVectors(this.a, this.b).multiplyScalar(0.5);
      const px = this.a.distanceTo(this.b) / Math.max(1e-6, mgr.camera.position.distanceTo(this.m) * Ribbon.pixelWorld.value);
      const dashes = Math.max(2, Math.round(px / DASH_PX));
      const intensity = k * (0.32 + 1.1 * e.heat);
      p.ribbons.add(this.a, this.b, 0.0001, intensity, 8, factionSeed(0.5, e.faction), dashes, dashes);
    }
  }
}
