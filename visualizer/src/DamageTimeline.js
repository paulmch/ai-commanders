/**
 * DamageTimeline - pure (no three.js) derivation of every ship's damage
 * history from a recording, so the renderer can ask "what does this ship
 * look like at battle time t?" and get the same answer however the
 * timeline was reached (playback, seek forward, scrub back).
 *
 * A "mark" is one wound on a hull: a coilgun crater, a torpedo breach, or
 * a scheduled secondary explosion while the ship is in its death spiral.
 * Marks carry everything the renderer needs to place them without any
 * randomness of its own: a seeded aim (section, azimuth, radial/axial
 * fraction), severity, whether it breached the hull (venting), what vented
 * (atmosphere / fuel / reactor coolant, from same-tick module events) and
 * the ship's recorded kinematics at the moment of impact (for spall that
 * flies off in world space).
 *
 * Coordinates: world metres as recorded; ship-local frame matches the
 * viewer's Object3D.lookAt(fwd) with world-up +Y: +Z nose, +Y dorsal.
 */

// ---------------------------------------------------------------------------
// Deterministic hashing
// ---------------------------------------------------------------------------
export function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/** Integer mix of up to four ints -> uniform float in [0, 1). */
export function hash01(a, b = 0, c = 0, d = 0) {
  let h = (Math.imul(a | 0, 0x9e3779b1) ^ Math.imul(b | 0, 0x85ebca77) ^
    Math.imul(c | 0, 0xc2b2ae3d) ^ Math.imul(d | 0, 0x27d4eb2f)) >>> 0;
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h ^= h >>> 12; h = Math.imul(h, 0x297a2d39) >>> 0;
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

// ---------------------------------------------------------------------------
// Trace kinematics
// ---------------------------------------------------------------------------

/** Index pair bracketing t and the blend between them. */
export function frameBracket(trace, t) {
  const n = trace.length;
  if (!n) return null;
  if (t <= trace[0].t) return { i0: 0, i1: 0, a: 0 };
  if (t >= trace[n - 1].t) return { i0: n - 1, i1: n - 1, a: 0 };
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (trace[mid].t <= t) lo = mid; else hi = mid;
  }
  const span = trace[hi].t - trace[lo].t;
  return { i0: lo, i1: hi, a: span > 0 ? (t - trace[lo].t) / span : 0 };
}

/**
 * Ship pose at battle time t from the recorded trace (lerped). Returns
 * null when the ship is absent. Writes into `out` when given.
 * {pos, vel, fwd (unit), hull (0..1), dying, destroyed}
 */
export function shipStateAt(trace, shipId, t, out = null) {
  const b = frameBracket(trace, t);
  if (!b) return null;
  const s0 = trace[b.i0].ships?.[shipId];
  const s1 = trace[b.i1].ships?.[shipId] || s0;
  if (!s0 || !s0.pos) return null;
  const a = b.a;
  const o = out || { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false };
  for (let k = 0; k < 3; k++) {
    o.pos[k] = s0.pos[k] + ((s1.pos?.[k] ?? s0.pos[k]) - s0.pos[k]) * a;
    const v0 = s0.vel?.[k] ?? 0, v1 = s1.vel?.[k] ?? v0;
    o.vel[k] = v0 + (v1 - v0) * a;
    const f0 = s0.fwd?.[k] ?? (k === 2 ? 1 : 0), f1 = s1.fwd?.[k] ?? f0;
    o.fwd[k] = f0 + (f1 - f0) * a;
  }
  const fl = Math.hypot(o.fwd[0], o.fwd[1], o.fwd[2]) || 1;
  o.fwd[0] /= fl; o.fwd[1] /= fl; o.fwd[2] /= fl;
  const h0 = s0.hull ?? 100, h1 = s1.hull ?? h0;
  o.hull = Math.max(0, Math.min(1, (h0 + (h1 - h0) * a) / 100));
  // discrete flags switch at their frame, like the Interpolator
  const cur = a < 1 ? s0 : s1;
  o.dying = !!cur.dying;
  o.destroyed = !!cur.destroyed;
  return o;
}

/**
 * Ship-local basis for a forward vector, identical to three.js
 * Object3D.lookAt for a non-camera object with up = +Y:
 * z = fwd, x = normalize(up x z), y = z x x.
 */
export function localBasis(fwd, out = null) {
  const o = out || { x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0] };
  const z = o.z, x = o.x, y = o.y;
  z[0] = fwd[0]; z[1] = fwd[1]; z[2] = fwd[2];
  x[0] = z[2]; x[1] = 0; x[2] = -z[0];      // (0,1,0) x z
  let xl = Math.hypot(x[0], x[2]);
  if (xl < 1e-6) { x[0] = 1; x[2] = 0; xl = 1; }  // looking straight up/down
  x[0] /= xl; x[2] /= xl;
  y[0] = z[1] * x[2] - z[2] * x[1];
  y[1] = z[2] * x[0] - z[0] * x[2];
  y[2] = z[0] * x[1] - z[1] * x[0];
  return o;
}

export function worldToLocalDir(fwd, d) {
  const B = localBasis(fwd);
  const dot = (u) => u[0] * d[0] + u[1] * d[1] + u[2] * d[2];
  return [dot(B.x), dot(B.y), dot(B.z)];
}

export function localToWorldDir(fwd, l) {
  const B = localBasis(fwd);
  return [
    B.x[0] * l[0] + B.y[0] * l[1] + B.z[0] * l[2],
    B.x[1] * l[0] + B.y[1] * l[1] + B.z[1] * l[2],
    B.x[2] * l[0] + B.y[2] * l[1] + B.z[2] * l[2]
  ];
}

// ---------------------------------------------------------------------------
// Visual curves (shared by renderer and tests)
// ---------------------------------------------------------------------------

/**
 * Crater temperature: white-hot on impact, orange within a few seconds,
 * a dull red ember by ~10 s, black scorch by ~18 s. 0 before impact.
 * Bigger wounds hold their heat longer.
 */
export function craterHeat(age, severity = 1) {
  if (age < 0) return 0;
  const k = 0.75 + 0.35 * Math.min(severity, 1.6);
  const h = 1.15 * Math.exp(-age / (1.4 * k)) + 0.45 * Math.exp(-age / (8.0 * k));
  return age > 24 * k ? 0 : h * Math.min(1, age / 0.05 + 0.2);
}

const VENT_TAU = { atmo: 16, fuel: 40, coolant: 26, fire: 9 };

/**
 * Venting jet strength (0..1): snaps open, then bleeds down as the
 * compartment empties. Zero before the breach and once it has run dry.
 */
export function ventTau(mark) {
  return (VENT_TAU[mark.vent] || 16) * (0.7 + 0.6 * mark.ventSize);
}

export function ventStrength(mark, age) {
  if (!mark.vent || age < 0) return 0;
  const tau = ventTau(mark);
  const s = mark.ventSize * Math.min(1, age / 0.35) * Math.exp(-age / tau);
  return s < 0.035 ? 0 : s;
}

/** Seconds a vent keeps running (for scheduling / culling). */
export function ventLife(mark) {
  if (!mark.vent) return 0;
  return ventTau(mark) * Math.log(Math.max(1.0001, mark.ventSize / 0.035));
}

/**
 * Dying-reactor pulse rate (Hz) `ttd` seconds before detonation: ~0.6 Hz
 * with a long fuse, climbing to 4 Hz at the end (f = 0.5 + 3.5 / (ttd + 1)).
 */
export function reactorPulseRate(ttd) {
  return 0.5 + 3.5 / (Math.max(0, ttd) + 1);
}

/**
 * Dying-reactor glow 0..1 at `ttd` seconds before detonation: pulses at
 * reactorPulseRate over a floor that brightens as the fuse runs down. The
 * phase is the closed-form integral of the rate, so the glow is a pure
 * function of battle time (scrub-exact). 0 once detonated or unknown.
 */
export function reactorPulse(ttd) {
  if (!(ttd >= 0) || !Number.isFinite(ttd)) return 0;
  const phase = -2 * Math.PI * (0.5 * ttd + 3.5 * Math.log(ttd + 1));
  const near = 1 - ttd / (ttd + 6);
  const beat = 0.5 + 0.5 * Math.sin(phase);
  return Math.min(1, (0.3 + 0.7 * near) * (0.35 + 0.65 * beat * beat));
}

/** Seconds spall debris from a mark stays in flight. */
export const SPALL_LIFE = 14;

/** A coilgun hit this far off the armour normal skids along the plating. */
export const GRAZE_DEG = 60;

/** Fleet-range bracket flash after a weapon hit (ContactOverlay). */
export const HIT_FLASH_S = 1.5;

/**
 * Persistent blast preset (render/Explosion.js KINDS) for a hit mark.
 * Torpedoes are 250 kg kinetic penetrators: only a recorded explosive
 * yield (legacy warhead rounds) gets the warhead fireball; a round the
 * armour stops is a kinetic strike, one that punches through a breach.
 * Coilgun slugs pierce when they penetrate, skid in a spark fan along the
 * plating when grazing, and otherwise crater the armour.
 */
export function impactPreset(m) {
  if (m.kind === 'torpedo') {
    if (m.explosive > 0) return 'warhead';
    return m.penetrated ? 'breach' : 'kineticStrike';
  }
  if (m.kind === 'slug') {
    if (m.penetrated) return 'pierce';
    return (m.angle || 0) > GRAZE_DEG ? 'graze' : 'impact';
  }
  return null;
}

/**
 * Blast scale (scene km) of a hit mark's preset on a hull drawn `W` wide.
 * Kinetic strikes are capped to a fraction of the displayed beam so a
 * stopped penetrator never swallows the ship it bounced off; a warhead
 * keeps its old energy-driven size.
 */
export function impactScale(m, W) {
  const E = Math.max(0.1, m.energy || 1);
  const preset = m.strike || impactPreset(m);
  if (preset === 'warhead') return Math.max(1.4, Math.min(3.2, 0.9 * Math.pow(Math.max(E, 8), 0.3)));
  if (preset === 'kineticStrike' || preset === 'breach') return Math.min(0.35 * W, 0.22 * Math.pow(E, 0.4));
  return Math.min(0.3 * W, Math.max(0.12, 0.2 * Math.pow(E, 0.35)));
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function ventKind(modules, penetrated) {
  if (!penetrated) return null;
  const names = modules.map(m => m.toLowerCase());
  if (names.some(n => n.includes('reactor'))) return 'coolant';
  if (names.some(n => n.includes('fuel'))) return 'fuel';
  if (names.some(n => n.includes('engine'))) return 'fire';
  return 'atmo';
}

/** Section a hit lands on, normalised. */
function normSection(s) {
  s = (s || '').toLowerCase();
  if (s.startsWith('nose') || s === 'front' || s === 'bow') return 'nose';
  if (s.startsWith('tail') || s === 'aft' || s === 'rear' || s === 'stern') return 'tail';
  return 'lateral';
}

/**
 * Build the per-ship damage timeline.
 *
 * @param {Object} rec   recording-shaped: {events, sim_trace}
 * @param {Object} opts  impactTime(event) -> visual time of a coilgun hit
 *                       (defaults to the event timestamp)
 * @returns {{trace, ships: Map<string, ShipDamage>, marks: Mark[]}}
 *   marks is every mark of every ship sorted by time.
 */
export function buildDamageTimeline(rec, opts = {}) {
  const events = rec.events || [];
  const trace = rec.sim_trace || rec.simTrace || [];
  const impactTime = opts.impactTime || (e => e.timestamp);
  const ships = new Map();
  const ensure = (id) => {
    if (!ships.has(id)) {
      ships.set(id, {
        id, seed: hashString(id), marks: [], dyingStart: null, deathTime: null,
        modules: [], firstT: null, armor0: null, dyingEvent: null, wrecks: []
      });
    }
    return ships.get(id);
  };

  // Death spiral windows and as-built armour from the trace
  for (const frame of trace) {
    for (const [id, s] of Object.entries(frame.ships || {})) {
      const sd = ensure(id);
      if (sd.firstT === null) sd.firstT = frame.t;
      if (!sd.armor0 && s.armor) sd.armor0 = { ...s.armor };
      if (s.dying && sd.dyingStart === null && sd.deathTime === null) sd.dyingStart = frame.t;
      if (s.destroyed && sd.deathTime === null) sd.deathTime = frame.t;
    }
  }

  // Torpedo approach: last recorded position of each round
  const torpLast = new Map();
  for (const frame of trace) {
    for (const tp of frame.torpedoes || []) torpLast.set(tp.id, tp.pos);
  }

  // Same-tick bookkeeping: kinetic armour breaches and module hits. A
  // torpedo's breach is read from events naming the round (v3: armour
  // damage / hull damage / penetration carry torpedo_id or ordnance_id),
  // else from the same-tick kinetic armour event of matching energy.
  const kineticPen = [];        // {ship, t, dmg, penetrated}
  const torpPen = new Map();    // torpedo id -> penetrated
  const penTicks = [];          // legacy penetration events: {ship, t}
  const warheads = new Map();   // torpedo id -> launch warhead_gj
  const wreckEvents = [];
  const gunShots = new Map();   // ship -> Map(slot -> [t]) (non-spinal guns)
  for (const e of events) {
    const d = e.data || {};
    const type = e.event_type;
    if (type === 'armor_damage' && d.torpedo_id) {
      torpPen.set(d.torpedo_id, !!torpPen.get(d.torpedo_id) || !!d.penetrated);
    } else if (type === 'armor_damage' && d.source === 'kinetic') {
      kineticPen.push({ ship: e.ship_id, t: e.timestamp, dmg: d.damage_gj, penetrated: !!d.penetrated });
    } else if (type === 'damage_taken' && (d.torpedo_id || d.ordnance_id)) {
      torpPen.set(d.torpedo_id || d.ordnance_id, true);   // armour already gone
    } else if (type === 'penetration') {
      if (d.ordnance_id) torpPen.set(d.ordnance_id, true);
      else penTicks.push({ ship: e.ship_id, t: e.timestamp });
    } else if (type === 'torpedo_launched') {
      warheads.set(d.torpedo_id, d.warhead_gj || 0);
    } else if (type === 'ship_dying') {
      ensure(e.ship_id).dyingEvent = { t: e.timestamp, fuse: d.detonation_in_s ?? null };
    } else if (type === 'shot_fired' && d.weapon_slot && !(d.weapon_name || '').toLowerCase().includes('spinal')) {
      if (!gunShots.has(e.ship_id)) gunShots.set(e.ship_id, new Map());
      const slots = gunShots.get(e.ship_id);
      if (!slots.has(d.weapon_slot)) slots.set(d.weapon_slot, []);
      slots.get(d.weapon_slot).push(e.timestamp);
    }
    if (type === 'module_damaged' || type === 'module_destroyed') {
      ensure(e.ship_id).modules.push({ t: e.timestamp, name: d.module_name || '' });
    }
    if (type === 'module_destroyed' || type === 'radiator_damaged') wreckEvents.push(e);
  }
  for (const [id, list] of buildWrecks(wreckEvents, gunShots)) ensure(id).wrecks = list;

  const tmp = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false };
  const addMark = (shipId, m, fromWorld) => {
    const sd = ensure(shipId);
    const st = shipStateAt(trace, shipId, m.t, tmp);
    let approach = [0, 0, 1];
    if (st && fromWorld) {
      const d = [fromWorld[0] - st.pos[0], fromWorld[1] - st.pos[1], fromWorld[2] - st.pos[2]];
      const l = Math.hypot(d[0], d[1], d[2]);
      if (l > 1e-3) approach = worldToLocalDir(st.fwd, [d[0] / l, d[1] / l, d[2] / l]);
    } else if (m.section === 'tail') approach = [0, 0, -1];
    m.approach = approach;
    m.shipPos = st ? st.pos.slice() : null;
    m.shipVel = st ? st.vel.slice() : [0, 0, 0];
    m.shipFwd = st ? st.fwd.slice() : [0, 0, 1];
    sd.marks.push(m);
    return m;
  };

  for (const e of events) {
    const d = e.data || {};
    if (e.event_type === 'hit') {
      const shooterId = d.shooter_id || d.source_ship_id;
      const shooter = shooterId ? shipStateAt(trace, shooterId, e.timestamp) : null;
      addMark(e.ship_id, {
        t: impactTime(e), eventT: e.timestamp, kind: 'slug',
        section: normSection(d.hit_location),
        energy: d.kinetic_energy_gj || 1,
        penetrated: !!d.penetrated, critical: !!d.critical_hit,
        angle: d.impact_angle_deg || 0, ordnanceId: d.projectile_id || null
      }, shooter ? shooter.pos : null);
    } else if (e.event_type === 'torpedo_impact') {
      const target = d.target_id || e.ship_id;
      const dmg = d.total_damage_gj || 8;
      let pen = torpPen.get(d.torpedo_id);
      if (pen === undefined) {
        const k = kineticPen.find(k => k.ship === target && Math.abs(k.t - e.timestamp) < 1e-6 &&
          Math.abs((k.dmg || 0) - dmg) < 1e-3);
        pen = k ? k.penetrated : penTicks.some(p => p.ship === target && Math.abs(p.t - e.timestamp) < 1e-6);
      }
      // Older recordings carry no explosive split: a round launched with a
      // warhead is an explosive one, everything else a kinetic penetrator
      const explosive = d.explosive_damage_gj ?? ((warheads.get(d.torpedo_id) || 0) > 0 ? dmg : 0);
      addMark(target, {
        t: e.timestamp, eventT: e.timestamp, kind: 'torpedo',
        section: normSection(d.hit_location),
        energy: dmg, penetrated: !!pen, critical: false, explosive,
        speed: d.impact_speed_kps || 0, ordnanceId: d.torpedo_id || null
      }, torpLast.get(d.torpedo_id) || null);
    }
  }

  // Death spiral: breaches open and secondaries walk along the hull
  // toward the reactor until the ship lets go (sim 'dying' phase)
  for (const sd of ships.values()) {
    if (sd.dyingStart === null) continue;
    const t0 = sd.dyingStart;
    const fuse = sd.dyingEvent?.fuse;
    const t1 = sd.deathTime ?? (fuse != null ? t0 + fuse : t0 + 30);
    const span = Math.max(0.5, t1 - t0);
    for (let b = 0; b < 2; b++) {
      const u = hash01(sd.seed, 7001, b);
      addMark(sd.id, {
        t: t0 + 0.15 + b * 0.9 * hash01(sd.seed, 7002, b), eventT: t0, kind: 'breach',
        section: 'lateral', energy: 12, penetrated: true, critical: false,
        forceVent: u < 0.5 ? 'fuel' : 'atmo'
      }, null);
    }
    let t = t0 + 0.4 + 0.5 * hash01(sd.seed, 7100);
    for (let k = 0; t < t1 - 0.1 && k < 60; k++) {
      const big = hash01(sd.seed, 7200, k) < 0.28;
      addMark(sd.id, {
        t, eventT: t, kind: big ? 'secondary' : 'pop', section: 'lateral',
        energy: big ? 9 : 3, penetrated: big, critical: false,
        walk: 0.38 - 0.66 * ((t - t0) / span)
      }, null);
      t += 0.35 + 0.85 * hash01(sd.seed, 7300, k);
    }
  }

  const all = [];
  const byOrdnance = new Map();
  for (const sd of ships.values()) {
    sd.marks.sort((a, b) => a.t - b.t || a.eventT - b.eventT);
    sd.marks.forEach((m, i) => {
      m.shipId = sd.id;
      m.index = i;
      m.uid = `${sd.id}#${i}`;
      m.seed = (Math.imul(sd.seed ^ 0x5bd1e995, 31) + i * 7919) >>> 0;
      finishMark(m, sd);
      if (m.ordnanceId) byOrdnance.set(m.ordnanceId, m);
      all.push(m);
    });
  }
  all.sort((a, b) => a.t - b.t);
  return { trace, ships, marks: all, byOrdnance };
}

// ---------------------------------------------------------------------------
// Wrecked mounts
// ---------------------------------------------------------------------------

/** Hull side a radiator_position names ('tail_dorsal' -> dorsal). */
function radiatorSide(pos) {
  const p = (pos || '').toLowerCase();
  for (const side of ['dorsal', 'ventral', 'port', 'starboard']) if (p.includes(side)) return side;
  return 'dorsal';
}

/**
 * Mounts knocked out over the battle, per ship, in time order:
 *   {t, kind: 'gun'|'spinal'|'radiator', slot, side, destroyed, name}
 * A destroyed 'Coilgun Battery' / turret names no weapon slot, so the slot
 * is inferred from the guns that never fire again after it: the one that
 * fired last is the one that was knocked out (null when none fired - the
 * renderer then picks a seeded mount). Radiator damage names the wing's
 * side ('tail_port'); a destroyed wing is torn off rather than buckled.
 *
 * @param {Array} events     module_destroyed / radiator_damaged events, time sorted
 * @param {Map}   gunShots   ship -> Map(weapon slot -> [shot times]), non-spinal guns
 * @returns {Map<string, Array>}
 */
export function buildWrecks(events, gunShots = new Map()) {
  const out = new Map();
  const push = (id, w) => { if (!out.has(id)) out.set(id, []); out.get(id).push(w); };
  const taken = new Map();   // ship -> Set(slot)
  for (const e of events || []) {
    const d = e.data || {};
    const ship = e.ship_id;
    if (e.event_type === 'radiator_damaged') {
      push(ship, { t: e.timestamp, kind: 'radiator', side: radiatorSide(d.radiator_position),
        end: (d.radiator_position || '').toLowerCase().startsWith('nose') ? 'nose' : 'tail',
        destroyed: !!d.destroyed, name: d.radiator_position || 'radiator' });
      continue;
    }
    if (e.event_type !== 'module_destroyed') continue;
    const name = (d.module_name || '').toLowerCase();
    if (name.includes('spinal')) {
      push(ship, { t: e.timestamp, kind: 'spinal', slot: null, name: d.module_name });
    } else if (name.includes('coilgun') || name.includes('turret') || name.includes('battery')) {
      if (!taken.has(ship)) taken.set(ship, new Set());
      const used = taken.get(ship);
      let slot = null, latest = -Infinity;
      for (const [s, times] of gunShots.get(ship) || []) {
        if (used.has(s) || times.some(x => x > e.timestamp + 0.5)) continue;
        const last = times.filter(x => x <= e.timestamp + 0.5).at(-1);
        if (last !== undefined && last > latest) { latest = last; slot = s; }
      }
      if (slot) used.add(slot);
      push(ship, { t: e.timestamp, kind: 'gun', slot, name: d.module_name });
    }
  }
  return out;
}

/** Wrecks of one ship that have happened by `now` (pure filter). */
export function wrecksAt(sd, now) {
  const list = sd?.wrecks;
  if (!list || !list.length) return [];
  return list.filter(w => w.t <= now);
}

// ---------------------------------------------------------------------------
// Armour, countdown, recent hits
// ---------------------------------------------------------------------------

/**
 * Remaining armour per section as a fraction of the as-built thickness
 * (first recorded frame), lerped between frames like shipStateAt.
 * Sections a recording never lists stay at 1. Writes into `out`.
 * {nose, lateral, tail} in 0..1
 */
export function armorStateAt(timeline, shipId, t, out = null) {
  const o = out || { nose: 1, lateral: 1, tail: 1 };
  o.nose = 1; o.lateral = 1; o.tail = 1;
  const sd = timeline?.ships?.get(shipId);
  const a0 = sd?.armor0;
  const b = a0 ? frameBracket(timeline.trace, t) : null;
  if (!b) return o;
  const s0 = timeline.trace[b.i0].ships?.[shipId];
  const s1 = timeline.trace[b.i1].ships?.[shipId] || s0;
  if (!s0?.armor) return o;
  for (const k of ['nose', 'lateral', 'tail']) {
    const init = a0[k];
    if (!(init > 0)) continue;
    const v0 = s0.armor[k] ?? init, v1 = s1?.armor?.[k] ?? v0;
    o[k] = clamp01((v0 + (v1 - v0) * b.a) / init);
  }
  return o;
}

/**
 * Battle time a dying hulk's reactor lets go, or null when the ship never
 * enters the death spiral: the recorded death when the trace has it
 * (replay), else the ship_dying fuse (live, before the end is streamed).
 */
export function detonationTime(sd) {
  if (!sd || (sd.dyingStart === null && !sd.dyingEvent)) return null;
  if (sd.deathTime !== null && sd.deathTime !== undefined) return sd.deathTime;
  const ev = sd.dyingEvent;
  return ev && ev.fuse != null ? ev.t + ev.fuse : null;
}

/** Time of the latest slug or torpedo hit on a ship at or before `now` (-Infinity if none). */
export function lastHitAt(sd, now) {
  const marks = sd?.marks;
  if (!marks) return -Infinity;
  for (let i = marksUpTo(marks, now) - 1; i >= 0; i--) {
    if (marks[i].kind === 'slug' || marks[i].kind === 'torpedo') return marks[i].t;
  }
  return -Infinity;
}

/** Severity, aim and vent parameters (all seeded, no Math.random). */
function finishMark(m, sd) {
  const r = (k) => hash01(m.seed, k);
  const E = Math.max(0.05, m.energy);
  const base = m.kind === 'torpedo' ? 1.05 : m.kind === 'secondary' ? 0.95 : m.kind === 'breach' ? 0.9 :
    m.kind === 'pop' ? 0.45 : 0.3;
  m.severity = Math.min(1.8, base + 0.2 * Math.log2(1 + E) * (m.kind === 'slug' ? 1 : 0.5) + (m.penetrated ? 0.25 : 0));

  // Aim: azimuth around the long axis from the attacker's bearing.
  // end: +1 nose cap, -1 tail, 0 flank; radial is the fraction of the
  // half-width off-axis (end caps); axial the fraction of the hull length
  // from the centre (flanks, -0.5 stern .. +0.5 bow).
  const [ax, ay, az] = m.approach;
  const lat = Math.hypot(ax, ay);
  let azimuth = lat > 0.02 ? Math.atan2(ay, ax) : r(1) * Math.PI * 2;
  const jitter = m.kind === 'slug' || m.kind === 'torpedo' ? 0.7 : Math.PI;
  azimuth += (r(2) - 0.5) * 2 * jitter;
  m.aim = { end: 0, azimuth, radial: 1, axial: 0 };
  if (m.section === 'nose') {
    m.aim.end = 1;
    m.aim.radial = 0.5 * Math.sqrt(r(3));
    m.aim.axial = 0.5;
  } else if (m.section === 'tail') {
    m.aim.end = -1;
    m.aim.radial = 0.35 + 0.6 * r(3);
    m.aim.axial = -0.5;
  } else {
    // flank hits lean toward the end the shot came from; death-spiral
    // secondaries follow their scheduled walk toward the reactor
    const lean = Math.max(-0.3, Math.min(0.3, az * 0.35));
    const axial = m.walk !== undefined ? m.walk + (r(3) - 0.5) * 0.12 : lean + (r(3) - 0.5) * 0.7;
    m.aim.axial = Math.max(-0.4, Math.min(0.42, axial));
  }

  // What comes out of the hole: modules hit at the same tick decide
  if (m.forceVent) m.vent = m.forceVent;
  else {
    const near = sd.modules.filter(x => Math.abs(x.t - m.eventT) < 0.51).map(x => x.name);
    m.vent = ventKind(near, m.penetrated);
    m.modules = near;
  }
  m.ventSize = m.vent ? Math.min(1, 0.45 + 0.08 * E + (m.kind === 'breach' ? 0.35 : 0)) : 0;
  m.debris = m.kind === 'torpedo' ? 22 : m.kind === 'secondary' ? 12 : m.kind === 'breach' ? 10 :
    m.kind === 'pop' ? 4 : Math.min(10, 3 + Math.round(E * 0.8));
  if (m.penetrated) m.debris = Math.min(32, m.debris + 5);
  m.blast = m.kind === 'pop' || m.kind === 'secondary' ? m.kind : null;
  m.strike = impactPreset(m);
}

/** Marks of one ship with t <= now (binary search on the sorted list). */
export function marksUpTo(marks, now) {
  let lo = 0, hi = marks.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (marks[mid].t <= now) lo = mid + 1; else hi = mid;
  }
  return lo;   // count
}

/**
 * Aggregate damage look of a ship at battle time `now` - a pure function
 * of the timeline and the time, so a backward seek simply yields less.
 * {hull, dying, destroyed, count, breaches, smoke, sparks, flicker, radiator, lateralHits}
 */
const _st = { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], hull: 1, dying: false, destroyed: false };

export function damageStateAt(timeline, shipId, now, out = null) {
  const sd = timeline.ships.get(shipId);
  const st = shipStateAt(timeline.trace, shipId, now, _st);
  const hull = st ? st.hull : 1;
  const dying = !!st?.dying;
  out = out || {};
  out.hull = hull; out.dying = dying; out.destroyed = !!st?.destroyed;
  out.count = 0; out.breaches = 0; out.lateralHits = 0;
  out.smoke = 0; out.sparks = 0; out.flicker = 0; out.radiator = 0;
  if (!sd) return out;
  const n = marksUpTo(sd.marks, now);
  out.count = n;
  for (let i = 0; i < n; i++) {
    const m = sd.marks[i];
    if (m.penetrated) out.breaches++;
    if (m.section === 'lateral' && m.kind !== 'pop') out.lateralHits++;
  }
  const loss = 1 - hull;
  out.smoke = dying ? 1 : clamp01((0.55 - hull) / 0.35 + 0.06 * out.breaches);
  out.sparks = dying ? 1 : clamp01((0.75 - hull) / 0.45 + 0.12 * out.breaches);
  out.flicker = dying ? 1 : clamp01((0.6 - hull) / 0.4 + 0.05 * out.breaches);
  out.radiator = clamp01(loss * 1.3 + 0.12 * out.lateralHits - 0.15);
  return out;
}

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
