// Recording-only weapon math. Units here are metres / seconds; the renderer
// converts to kilometres. No clock, random generator, or frame history.
import { buildTorpedoThrusters } from './TorpedoRcs.js';

export const ION_LIFE = 1;
export const MISS_LIFE = 6;
export const IMPACT_LIFE = 0.46;
/** Loader torpedo moments this module renders itself (main.js skips them). */
export const OWNED_MOMENTS = new Set(['seeker_kill', 'intercepted', 'destroyed', 'burnout']);
export const SHUTTER = 0.085;
/** A torpedo_miss closer than this (km) passed through the hull and failed its hit roll: a dud. */
export const DUD_KM = 0.05;

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
  // Thruster telemetry: `b` (the frame that closes this interval) records
  // the main throttle of exactly this tick. An RCS trim also burns fuel, so
  // fuel use alone would light the plume for it.
  if (b && b.thr !== undefined && !(b.thr > 0)) return 0;
  const lit = b && b.thr > 0;
  const burn = dt > 0 ? ((a.dv_remaining ?? 0) - (b.dv_remaining ?? 0)) / dt : 0;
  const phase = String(a.phase || a.guidance_phase || a.guidance_mode || '').toLowerCase();
  if (phase.includes('coast') || a.thrusting === false) return 0;
  if (!lit && burn <= 0.002 && a.thrusting !== true) return 0;
  if (phase.includes('cruise') || phase.includes('intercept')) return 0.42;
  // Old recordings have only fuel consumption. Infer the terminal phase
  // from current geometry, never from a future hit or from render history.
  const terminal = phase.includes('terminal') || phase.includes('proportional')
    || (closure > 0 && (range < 50000 || range / closure < 6
      || (closure >= 12000 && range / closure < 20)));
  return terminal ? 1 : 0.42;
}

/** Numeric weapon slot ("weapon_2" -> 2) and spinal flag of a shot_fired payload. */
export function shotSlot(d) { return Number(d.weapon_slot?.split('_').at(-1)) || 0; }
export function shotSpinal(d) { return !!d.weapon_name?.toLowerCase().includes('spinal'); }

/** Spinal muzzle per hull: [fraction of hull width below the axis, of length forward]. */
export const SPINAL_MUZZLE = { cruiser: [-0.42, 0.5], battlecruiser: [-0.245, 0.56], battleship: [-0.335, 0.5], dreadnought_siege: [-0.2, 0.8] };
export const SPINAL_MUZZLE_DEFAULT = [-0.31, 0.56];

/** Twin-barrel turret: barrel centres across the gun (x, in units of the turret size s). */
export const TWIN_BARREL_X = [-0.2, 0.2];

/**
 * Which barrel (0/1) of a twin turret a shot leaves from: the recorded
 * `barrel` when present, else alternating by the shot's sequence number
 * `seq` on that weapon slot. Spinal mounts have one bore (always 0).
 */
export function shotBarrel(d, seq = 0) {
  if (shotSpinal(d || {})) return 0;
  const b = d?.barrel;
  if (b === 0 || b === 1) return b;
  return Math.abs(Math.floor(seq || 0)) % 2;
}

/**
 * Barrel of every shot_fired event, in event order (events sorted by time):
 * a per ship + weapon-slot shot counter feeds the shotBarrel fallback.
 * @returns {Map<Object, number>} event -> barrel
 */
export function assignBarrels(events) {
  const seq = new Map(), out = new Map();
  for (const e of events || []) {
    if (e.event_type !== 'shot_fired') continue;
    const d = e.data || {};
    const key = `${e.ship_id}|${d.weapon_slot}`;
    const n = seq.get(key) || 0;
    seq.set(key, n + 1);
    out.set(e, shotBarrel(d, n));
  }
  return out;
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
    // Per ship, time sorted: every shot and torpedo launch with its target
    // (engagement lines brighten on them, and fall back to them for ships
    // whose trace carries no fire-control target)
    this.fire = new Map();
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
      for (const p of f.torpedoes || []) {
        add(this.torpedoes, p.id, f.t, p);
        // Seeker line of sight to a live target in the same frame (the
        // thruster model points a coasting round's nose along it).
        const tg = f.ships?.[p.target];
        if (tg?.pos && p.pos && !tg.destroyed) {
          this.torpedoes.get(p.id).samples.at(-1).los = [tg.pos[0] - p.pos[0], tg.pos[1] - p.pos[1], tg.pos[2] - p.pos[2]];
        }
      }
    }
    const shots = new Map();
    const shotById = new Map();
    const barrels = assignBarrels(events);
    for (const e of events) {
      const d = e.data || {};
      if ((e.event_type === 'shot_fired' || e.event_type === 'torpedo_launched') && e.ship_id && d.target_id) {
        if (!this.fire.has(e.ship_id)) this.fire.set(e.ship_id, []);
        this.fire.get(e.ship_id).push({ t: e.timestamp, target: d.target_id, torpedo: e.event_type === 'torpedo_launched' });
      }
      if (e.event_type === 'shot_fired') {
        if (!shots.has(e.ship_id)) shots.set(e.ship_id, []);
        shots.get(e.ship_id).push(e);
        if (d.projectile_id) shotById.set(d.projectile_id, e);
        this.moments.push({ time: e.timestamp, kind: 'muzzle', ship: e.ship_id, data: d,
          slot: shotSlot(d), spinal: shotSpinal(d), barrel: barrels.get(e) ?? 0,
          track: this.slugs.get(d.projectile_id) || null,
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
      // A round that brushed the hull yet failed its hit roll skips off the
      // plating instead of vanishing inside the (oversize) hull
      if (e.event_type === 'torpedo_miss' && (d.closest_approach_km ?? Infinity) < DUD_KM) {
        this.moments.push({ time: e.timestamp, kind: 'dud', track: tr, target: d.target_id,
          probability: d.hit_probability ?? null, seed: tr.seed });
      }
    }
    for (const list of this.fire.values()) list.sort((a, b) => a.t - b.t);
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
      // The firing event (exact by projectile id, else the source's latest
      // shot within 2 s) dates the launch and names the gun it left from.
      let shot = shotById.get(tr.id) || null;
      if (!shot) {
        for (const e of shots.get(first.state.source) || []) {
          if (e.timestamp > first.t) break;
          if (first.t - e.timestamp <= 2) shot = e;
        }
      }
      const launch = shot ? shot.timestamp : first.t - 1;
      tr.source = first.state.source || shot?.ship_id || '';
      tr.shot = shot ? { slot: shot.data?.weapon_slot, index: shotSlot(shot.data || {}), spinal: shotSpinal(shot.data || {}),
        barrel: barrels.get(shot) ?? 0 } : null;
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
      tr.thrusters = buildTorpedoThrusters(tr.samples);
    }
    this.moments.sort((a, b) => a.time - b.time);
  }

  /**
   * Torpedoes inbound on each ship at battle time `time`, from the
   * recording alone (range and closure of each round in flight against its
   * own target), so it works the same in replay and live:
   *   Map target id -> {count, live, blind, eta, rangeKm}
   * `live` rounds with an intact seeker, `blind` seeker-killed rounds still
   * closing on it (they coast on and can still strike); `count` = both.
   * `eta` is the soonest closing time in seconds (Infinity when none is
   * closing), `rangeKm` the nearest inbound round.
   */
  threatsAt(time, out = new Map()) {
    out.clear();
    const s = this._ts || (this._ts = makeSample()), g = this._tg || (this._tg = makeSample());
    for (const tr of this.torpedoes.values()) {
      if (time < tr.start || time >= tr.end || !tr.samples.length) continue;
      this.sample(tr, time, s);
      const target = s.state?.target;
      const ship = target ? this.ships.get(target) : null;
      if (!ship || !sampleTrack(ship, time, g) || g.state?.destroyed) continue;
      const dx = g.pos[0] - s.pos[0], dy = g.pos[1] - s.pos[1], dz = g.pos[2] - s.pos[2];
      const range = Math.hypot(dx, dy, dz);
      const closure = range > 0
        ? ((s.vel[0] - g.vel[0]) * dx + (s.vel[1] - g.vel[1]) * dy + (s.vel[2] - g.vel[2]) * dz) / range : 0;
      const blind = time >= tr.disabledAt || !!s.state.disabled;
      if (blind && !(closure > 1)) continue;   // coasting away: no longer a threat
      let e = out.get(target);
      if (!e) out.set(target, e = { count: 0, live: 0, blind: 0, eta: Infinity, rangeKm: Infinity });
      e.count++;
      if (blind) e.blind++; else e.live++;
      e.rangeKm = Math.min(e.rangeKm, range / 1000);
      if (closure > 1) e.eta = Math.min(e.eta, range / closure);
    }
    return out;
  }

  /** Latest shot / launch of a ship at or before `time` ({t, target, torpedo} or null). */
  lastFire(shipId, time) {
    const list = this.fire.get(shipId);
    if (!list || !list.length || list[0].t > time) return null;
    let lo = 0, hi = list.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (list[mid].t <= time) lo = mid; else hi = mid - 1;
    }
    return list[lo];
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
