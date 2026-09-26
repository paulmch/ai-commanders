// Recording-only weapon math. Units here are metres / seconds; the renderer
// converts to kilometres. No clock, random generator, or frame history.
export const ION_LIFE = 1;
export const MISS_LIFE = 6;
export const IMPACT_LIFE = 0.46;
/** Loader torpedo moments this module renders itself (main.js skips them). */
export const OWNED_MOMENTS = new Set(['seeker_kill', 'intercepted', 'destroyed', 'burnout']);
export const SHUTTER = 0.085;

export function seedFor(id) {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
}

export function tracerLength(speed, pixelWorld, age = Infinity) {
  // A virtual shutter, bounded by flight age and a small screen-space floor.
  if (!(speed > 0)) return 0;
  return Math.min(speed * Math.max(0, age), Math.max(speed * SHUTTER, pixelWorld * 7));
}

export function missOpacity(age) {
  const f = Math.max(0, Math.min(1, age / MISS_LIFE));
  return (1 - f) * (1 - f);
}

export function beamPulse(time, start, end, seed) {
  if (time < start || time > end + 1) return 0;
  const envelope = Math.min(1, (time - start) / 0.08 + 0.25, (end + 1 - time) / 0.18);
  const pulse = 0.72 + 0.28 * Math.pow(0.5 + 0.5 * Math.sin(time * 43 + seed * 71), 3);
  return envelope * pulse * (0.94 + 0.06 * Math.sin(time * 113 + seed * 19));
}

export function torpedoPower(a, b, dt, range, closure, disabled = false) {
  if (disabled || a.disabled || (a.dv_remaining ?? 0) <= 0) return 0;
  const burn = dt > 0 ? ((a.dv_remaining ?? 0) - (b.dv_remaining ?? 0)) / dt : 0;
  const phase = String(a.phase || a.guidance_phase || a.guidance_mode || '').toLowerCase();
  if (phase.includes('coast') || a.thrusting === false) return 0;
  if (burn <= 0.002 && a.thrusting !== true) return 0;
  if (phase.includes('cruise') || phase.includes('intercept')) return 0.42;
  // Old recordings have only fuel consumption. Infer the terminal phase
  // from current geometry, never from a future hit or from render history.
  const terminal = phase.includes('terminal') || phase.includes('proportional')
    || (closure > 0 && (range < 50000 || range / closure < 6
      || (closure >= 12000 && range / closure < 20)));
  return terminal ? 1 : 0.42;
}

export function sampleIndex(samples, time) {
  let lo = 0, hi = samples.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (samples[mid].t <= time) lo = mid; else hi = mid - 1;
  }
  return lo;
}

export function makeSample() {
  return { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], state: null, next: null, dt: 0 };
}

/** Writes into caller-owned storage; backward seeks follow the same path. */
export function sampleTrack(track, time, out) {
  const samples = track.samples;
  if (!samples.length) return false;
  const i = sampleIndex(samples, time);
  const a = samples[i], b = samples[Math.min(i + 1, samples.length - 1)];
  const dt = b.t - a.t;
  const f = dt > 0 ? Math.max(0, Math.min(1, (time - a.t) / dt)) : 0;
  const extrapolate = time < a.t || time > b.t;
  for (let k = 0; k < 3; k++) {
    const v = a.state.vel?.[k] || 0;
    out.pos[k] = extrapolate ? a.state.pos[k] + v * (time - a.t)
      : a.state.pos[k] + (b.state.pos[k] - a.state.pos[k]) * f;
    out.vel[k] = extrapolate ? v : v + ((b.state.vel?.[k] || 0) - v) * f;
    out.fwd[k] = (a.state.fwd?.[k] ?? (k === 2 ? 1 : 0)) * (1 - f)
      + (b.state.fwd?.[k] ?? (k === 2 ? 1 : 0)) * f;
  }
  out.state = a.state;
  out.next = b.state;
  out.dt = dt;
  return true;
}

export class WeaponTimeline {
  constructor(loader) {
    this.ships = new Map();
    this.slugs = new Map();
    this.torpedoes = new Map();
    this.moments = [];
    const trace = loader.simTrace || [];
    const events = loader.events || [];
    this.end = trace.at(-1)?.t ?? 0;
    const add = (map, id, t, state) => {
      let tr = map.get(id);
      if (!tr) {
        tr = { id, seed: seedFor(id), samples: [], start: t, end: Infinity, missAt: Infinity, impact: null, disabledAt: Infinity };
        map.set(id, tr);
      }
      tr.samples.push({ t, state });
    };
    for (const f of trace) {
      for (const id in f.ships) add(this.ships, id, f.t, f.ships[id]);
      for (const p of f.projectiles || []) add(this.slugs, p.id, f.t, p);
      for (const p of f.torpedoes || []) add(this.torpedoes, p.id, f.t, p);
    }
    const shots = new Map();
    for (const e of events) {
      const d = e.data || {};
      if (e.event_type === 'shot_fired') {
        if (!shots.has(e.ship_id)) shots.set(e.ship_id, []);
        shots.get(e.ship_id).push(e);
        this.moments.push({ time: e.timestamp, kind: 'muzzle', ship: e.ship_id, data: d,
          slot: Number(d.weapon_slot?.split('_').at(-1)) || 0,
          spinal: d.weapon_name?.toLowerCase().includes('spinal'),
          seed: seedFor(`${e.ship_id}|${e.timestamp}|${d.weapon_slot}`) });
      }
      const tr = this.torpedoes.get(d.torpedo_id) || this.slugs.get(d.projectile_id);
      if (!tr) continue;
      if (e.event_type === 'torpedo_launched') {
        tr.start = e.timestamp;
        this.moments.push({ time: e.timestamp, kind: 'ignition', track: tr, seed: tr.seed });
      }
      if (e.event_type === 'torpedo_fuel_exhausted') {
        this.moments.push({ time: e.timestamp, kind: 'burnout', track: tr, seed: tr.seed });
      }
      if (e.event_type === 'torpedo_retargeted') {
        this.moments.push({ time: e.timestamp, kind: 'retarget', track: tr, target: d.new_target_id, seed: tr.seed });
      }
      if (e.event_type === 'pd_torpedo_disabled') {
        tr.disabledAt = e.timestamp;
        this.moments.push({ time: e.timestamp, kind: 'seeker', track: tr, seed: tr.seed });
      }
      if (e.event_type === 'pd_torpedo_destroyed' || e.event_type === 'torpedo_intercepted' || e.event_type === 'pd_slug_destroyed') {
        tr.end = e.timestamp;
        this.moments.push({ time: tr.end, kind: 'pop', track: tr, seed: tr.seed });
      }
    }
    // Keep exactly the loader's travel-adjusted impact timing.
    for (const hit of loader.getVisualImpacts?.(-Infinity, Infinity) || []) {
      const tr = this.slugs.get(hit.projectile_id);
      if (!tr) continue;
      tr.end = hit.timestamp;
      tr.impact = hit.impact_position;
      this.moments.push({ time: tr.end, kind: 'hit', track: tr, target: hit.target_id, position: tr.impact, energy: hit.kinetic_energy_gj, seed: tr.seed });
    }
    for (const [id, rec] of loader.torpedoes || []) {
      const tr = this.torpedoes.get(id);
      if (!tr || !rec.outcome) continue;
      const o = rec.outcome;
      tr.end = o.timestamp;
      if (o.type === 'impact') {
        tr.impact = loader.getShipPositionAt(o.targetId, o.timestamp);
        this.moments.push({ time: tr.end, kind: 'hit', track: tr, target: o.targetId, position: tr.impact, energy: o.damageGj, seed: tr.seed });
      }
    }
    for (const tr of this.slugs.values()) {
      const first = tr.samples[0], last = tr.samples.at(-1);
      const candidates = shots.get(first.state.source);
      let launch = first.t - 1;
      if (candidates) {
        for (const e of candidates) {
          if (e.timestamp > first.t) break;
          if (first.t - e.timestamp <= 2) launch = e.timestamp;
        }
      }
      tr.start = Math.max(0, launch);
      // A disappearing round without a hit/intercept continues ballistically.
      // Rounds still present at the live edge must not be labelled as misses.
      if (tr.end === Infinity && last.t < this.end) {
        tr.missAt = last.t;
        tr.end = last.t + MISS_LIFE;
      }
    }
    for (const tr of this.torpedoes.values()) {
      if (tr.end === Infinity && tr.samples.at(-1).t < this.end) tr.end = tr.samples.at(-1).t + 1;
    }
    this.moments.sort((a, b) => a.time - b.time);
  }

  sample(tr, time, out) {
    sampleTrack(tr, time, out);
    const last = tr.samples.at(-1);
    if (tr.impact && time > last.t && tr.end > last.t) {
      const f = Math.min(1, (time - last.t) / (tr.end - last.t));
      for (let k = 0; k < 3; k++) {
        out.pos[k] = last.state.pos[k] + (tr.impact[k] - last.state.pos[k]) * f;
        out.vel[k] = (tr.impact[k] - last.state.pos[k]) / (tr.end - last.t);
      }
    }
    return out;
  }
}
