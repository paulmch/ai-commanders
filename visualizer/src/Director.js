// --------------------------------------------------------------- director
// Shot selection for the auto director: recording events -> scored
// highlights -> a non-overlapping shot list covering the whole battle.
// Everything here is a pure function of the recording (no THREE, no wall
// clock, no scene): the shot at battle time t is shotAt(shots, t), so
// scrubbing, frame-stepped clips and replays all cut at the same moments.
//
// Replay mode knows the future and cuts to an event a few seconds EARLY
// (the run-in before a strike, the hulk before it blows). Live mode
// (lookahead: false) is causal: a shot never starts before the event that
// triggers it, and shots already on screen never change when more of the
// battle arrives.

import { shipDetonationTime } from './DamageTimeline.js';

/** No shot is shorter than this (battle seconds). */
export const MIN_SHOT_S = 3.0;
/** A reactor blast holds the camera this long (no cut away mid-detonation). */
export const DETONATION_LOCK_S = 3.5;
/** The pull-back starts this long before the detonation (replay only). */
export const PULLBACK_PRE_S = 1.0;
/** ... and stays for the blast shell's life (render/BlastShell.js: 5 s). */
export const PULLBACK_POST_S = 5.0;
const APPROACH_S = 4.5;          // run-in / over-the-shoulder before a kill
const SALVO_MIN = 3;
const SALVO_WINDOW_S = 3;
const STRIKE_WINDOW_S = 2.5;
const CHAIN_WINDOW_S = 4.0;      // kills this close in time ...
const CHAIN_RANGE_KM = 60;       // ... and space share one widening shot
const DEFAULT_BLAST_KM = 15;
const WASH_COST_GJ = 50;         // plume-wash heat that counts as a real cost
const WIDE_FILL_S = 9;
const BEAUTY_FILL_S = 7;

/** Shot types (the camera vocabulary; CameraController frames each one). */
export const SHOT_TYPES = ['wide', 'chase', 'flyby', 'shoulder', 'runin', 'orbit', 'pullback'];

const factionOf = id => String(id || '').split('_')[0];
const num = (v, fallback = 0) => (Number.isFinite(v) ? v : fallback);

function distKm(a, b) {
  if (!a || !b) return Infinity;
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / 1000;
}

/** A salvo shot holds this long after its last round's torch lights (launch sequence). */
export const SALVO_AFTER_LIGHTOFF_S = 2.5;

/**
 * Scored highlights of a recording, sorted by time (ties: score, then id).
 * Kinds: kill (one or a chain of reactor blasts), dying (a hulk waiting to
 * blow), strike (torpedoes arriving on one ship), salvo (3+ launches by one
 * fleet), cookoff, disarmed (a ship's last recorded gun/tube knocked out),
 * first_fire (the first gun shot), plume_wash (a ship burned in another's
 * drive exhaust, once it costs something), surrender, and closest (closest
 * approach of the fleets, from the trace). Old recordings without the pass-5 events fall
 * back to frame-derived deaths (deathInfo) with a default blast radius.
 * @param {Array} events - loader events (sorted by time)
 * @param {Object} deathInfo - BattleLoader.buildDeathInfo()
 * @param {Object} opts - {simTrace} for closest approach (optional);
 *   causal: live mode - a highlight is described only by what was known
 *   when it began (a salvo is its first three launches, at the third)
 */
export function buildHighlights(events = [], deathInfo = {}, { simTrace = null, causal = false } = {}) {
  const out = [];
  const deaths = new Map();      // id -> {t, killer, dyingAt, radiusKm, position, yieldGj}
  const death = id => {
    if (!deaths.has(id)) deaths.set(id, { id, t: null, killer: null, dyingAt: null, fuse: null, radiusKm: null, position: null, yieldGj: null });
    return deaths.get(id);
  };
  const launches = new Map();    // faction -> launches
  const ignitions = new Map();   // torpedo id -> torch light-off (pass-5 launch sequence)
  const impacts = new Map();     // target -> impacts
  const mounts = new Map();      // ship -> Map(slot -> destroyed)
  const washes = new Map();      // "ship|emitter" -> open episode
  const washList = [];
  const surrenders = [];
  let firstFire = null;
  for (const e of events || []) {
    const t = e?.timestamp;
    if (!Number.isFinite(t)) continue;
    const d = e.data || {};
    switch (e.event_type) {
      case 'ship_dying': {
        const k = death(e.ship_id);
        if (k.dyingAt === null) { k.dyingAt = t; k.fuse = num(d.detonation_in_s, 10); }
        if (!k.killer && d.killer_id) k.killer = d.killer_id;
        break;
      }
      case 'ship_destroyed': {
        const k = death(e.ship_id);
        if (k.t === null) k.t = t;
        if (!k.killer && d.killer_id) k.killer = d.killer_id;
        break;
      }
      case 'reactor_blast': {
        const k = death(e.ship_id);
        if (k.t === null) k.t = t;
        k.radiusKm = num(d.radius_km, null);
        k.yieldGj = num(d.yield_gj, null);
        if (Array.isArray(d.position)) k.position = d.position;
        break;
      }
      case 'torpedo_ignition':
        if (d.torpedo_id && !ignitions.has(d.torpedo_id)) ignitions.set(d.torpedo_id, t);
        break;
      case 'torpedo_launched': {
        const f = factionOf(e.ship_id);
        if (!launches.has(f)) launches.set(f, []);
        launches.get(f).push(e);
        break;
      }
      case 'torpedo_impact': {
        const target = d.target_id;
        if (!target) break;
        if (!impacts.has(target)) impacts.set(target, []);
        impacts.get(target).push(e);
        break;
      }
      case 'magazine_cookoff':
        out.push({ kind: 'cookoff', t, score: 60 + Math.min(20, num(d.secondary_gj)), subject: e.ship_id,
          others: d.source_ship_id ? [d.source_ship_id] : [] });
        break;
      case 'mount_damaged': {
        if (d.kind !== 'gun' && d.kind !== 'launcher') break;
        if (!mounts.has(e.ship_id)) mounts.set(e.ship_id, { slots: new Map(), done: false });
        const m = mounts.get(e.ship_id);
        m.slots.set(d.slot ?? `${d.kind}_${d.index}`, !!d.destroyed);
        // Disarmed: every gun/tube this ship has had damaged is now dead
        // and there are at least two of them (the recording only names
        // fittings that were hit, so this is "last known weapon lost")
        if (!m.done && m.slots.size >= 2 && [...m.slots.values()].every(Boolean)) {
          m.done = true;
          out.push({ kind: 'disarmed', t, score: 48, subject: e.ship_id,
            others: d.source_ship_id ? [d.source_ship_id] : [] });
        }
        break;
      }
      case 'shot_fired':
        if (!firstFire) firstFire = e;
        break;
      case 'plume_wash': {
        // One episode per (washed ship, emitter): enter .. exit. It is a
        // highlight once it has a real cost (a fitting burned, or heat /
        // armour that matters), from the tick that cost is on record.
        const src = d.source_ship_id;
        if (!src || !e.ship_id || d.phase === 'torpedo_blinded') break;
        const key = `${e.ship_id}|${src}`;
        let w = washes.get(key);
        if (!w || w.closed) { w = { t0: t, t1: t, costAt: null, heat: 0, fittings: 0, closed: false }; washes.set(key, w); washList.push({ ship: e.ship_id, source: src, w }); }
        w.t1 = t;
        w.heat = Math.max(w.heat, num(d.heat_gj));
        w.fittings = Math.max(w.fittings, (d.fittings_hit || []).length);
        if (w.costAt === null && (w.fittings > 0 || w.heat >= WASH_COST_GJ || num(d.armor_ablation_cm) >= 1)) w.costAt = t;
        if (d.phase === 'exit') w.closed = true;
        break;
      }
      case 'surrender':
        if (e.ship_id) surrenders.push({ t, id: e.ship_id });
        break;
      default:
    }
  }
  // Frame-derived deaths (old recordings, or a death the events missed)
  for (const [id, info] of Object.entries(deathInfo || {})) {
    if (Number.isFinite(info?.time)) death(id);
  }
  // A kill is keyed on the time the viewer DETONATES the ship (one trace
  // frame after the ship_destroyed event, plus a dark coast for old
  // recordings: DamageTimeline.shipDetonationTime, the rule the effects
  // use), so the pull-back happens with the flash. eventT stays the tick
  // the engine recorded: live mode may not start a shot before it knows.
  for (const k of deaths.values()) {
    const info = (deathInfo || {})[k.id] || null;
    k.eventT = k.t;
    k.t = shipDetonationTime(info, k.eventT);
    if (k.t === null) continue;
    if (k.eventT === null) k.eventT = k.t;
    k.posT = k.eventT;
    if (Array.isArray(info?.position) && Number.isFinite(info.time)) {
      // where the trace has the wreck when it blows, coasting on with it
      k.position = info.position;
      k.posT = info.time;
      k.velocity = Array.isArray(info.velocity) ? info.velocity : null;
    }
  }

  // Kills, chained when close in time and space
  const kills = [...deaths.values()].filter(k => k.t !== null)
    .sort((a, b) => a.t - b.t || (a.id < b.id ? -1 : 1));
  const chains = [];
  for (const k of kills) {
    const chain = chains.find(c => k.t - c.at(-1).t <= CHAIN_WINDOW_S && k.t - c[0].t <= 2 * CHAIN_WINDOW_S &&
      c.some(o => distKm(o.position, k.position) <= CHAIN_RANGE_KM));
    if (chain) chain.push(k);
    else chains.push([k]);
  }
  for (const chain of chains) {
    const first = chain[0];
    const radius = Math.max(...chain.map(k => k.radiusKm ?? DEFAULT_BLAST_KM));
    out.push({
      kind: 'kill', t: first.t, tLast: chain.at(-1).t, eventT: Math.min(first.eventT, first.t),
      // (causal: scored by the first blast alone - the chain is the future)
      score: causal ? 100 + (first.radiusKm ?? DEFAULT_BLAST_KM) : 100 + radius + 25 * (chain.length - 1),
      subject: first.id, others: [...new Set(chain.map(k => k.killer).filter(Boolean))],
      kills: chain.map(k => ({ id: k.id, t: k.t, killer: k.killer, radiusKm: k.radiusKm ?? DEFAULT_BLAST_KM,
        yieldGj: k.yieldGj, position: k.position, posT: k.posT, velocity: k.velocity || null })),
      dyingAt: first.dyingAt !== null && first.dyingAt < first.t ? first.dyingAt : null
    });
  }
  // Hulks: dying now, detonating later (the fuse is the engine's own
  // forecast until the blast is on record)
  for (const k of deaths.values()) {
    if (k.dyingAt === null || (k.t !== null && k.t <= k.dyingAt)) continue;
    out.push({ kind: 'dying', t: k.dyingAt, until: causal ? k.dyingAt + k.fuse : k.t ?? k.dyingAt + k.fuse,
      score: 70, subject: k.id,
      others: k.killer ? [k.killer] : [] });
  }

  // Torpedo strikes: impacts on one ship within STRIKE_WINDOW_S
  for (const [target, list] of impacts) {
    let i = 0;
    while (i < list.length) {
      let j = i;
      while (j + 1 < list.length && list[j + 1].timestamp - list[j].timestamp <= STRIKE_WINDOW_S) j++;
      const group = list.slice(i, j + 1);
      const counted = causal ? group.slice(0, 1) : group;
      const damage = counted.reduce((s, e) => s + num(e.data?.total_damage_gj), 0);
      out.push({
        kind: 'strike', t: group[0].timestamp, tLast: group.at(-1).timestamp,
        score: 55 + 3 * counted.length + Math.min(20, damage / 10), subject: target,
        others: [...new Set(group.map(e => e.ship_id).filter(Boolean))],
        impacts: group.map(e => e.timestamp), torpedoes: group.map(e => e.data?.torpedo_id).filter(Boolean)
      });
      i = j + 1;
    }
  }

  // Salvos: SALVO_MIN+ launches by one fleet within SALVO_WINDOW_S
  for (const list of launches.values()) {
    let i = 0;
    let nth = 0;
    while (i < list.length) {
      let j = i;
      while (j + 1 < list.length && list[j + 1].timestamp - list[i].timestamp <= SALVO_WINDOW_S) j++;
      if (j - i + 1 >= SALVO_MIN) {
        const group = causal ? list.slice(i, i + SALVO_MIN) : list.slice(i, j + 1);
        const targets = new Map();
        const shooters = new Map();
        for (const e of group) {
          if (e.data?.target_id) targets.set(e.data.target_id, (targets.get(e.data.target_id) || 0) + 1);
          shooters.set(e.ship_id, (shooters.get(e.ship_id) || 0) + 1);
        }
        const top = m => [...m.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
        // A fleet's salvos rotate through its launchers (a new hero ship
        // each time) and score a little less each time they repeat
        const ranked = top(shooters).map(s => s[0]);
        const hero = ranked[nth % ranked.length];
        out.push({
          kind: 'salvo', t: causal ? group.at(-1).timestamp : group[0].timestamp, tLast: group.at(-1).timestamp,
          score: 40 + 2 * group.length - 6 * Math.min(3, nth),
          subject: hero, target: top(targets)[0]?.[0] || null,
          others: ranked.filter(id => id !== hero), count: group.length,
          // the last round's torch light-off: the launch sequence takes 3-8 s
          // (recorded torpedo_ignition; only with lookahead - a live shot is
          // never stretched by events that arrive after it started)
          lit: causal ? null : group.reduce((m, e) => Math.max(m, ignitions.get(e.data?.torpedo_id) ?? -Infinity), -Infinity)
        });
        nth++;
        i = j + 1;
      } else {
        i++;
      }
    }
  }

  // Drive-plume wash with a real cost: the emitter's jet and the ship in it
  for (const { ship, source, w } of washList) {
    if (w.costAt === null) continue;
    out.push({ kind: 'plume_wash', t: causal ? w.costAt : w.t0, until: causal ? w.costAt + 6 : w.t1,
      score: 50 + (causal ? 0 : Math.min(25, w.heat / 20) + 3 * Math.min(4, w.fittings)),
      subject: ship, target: source, others: [source] });
  }
  // Surrenders: ships of one fleet striking at the same checkpoint are one moment
  for (let i = 0; i < surrenders.length;) {
    let j = i;
    while (j + 1 < surrenders.length && surrenders[j + 1].t - surrenders[i].t <= 1 &&
      factionOf(surrenders[j + 1].id) === factionOf(surrenders[i].id)) j++;
    const ids = causal ? [surrenders[i].id] : surrenders.slice(i, j + 1).map(s => s.id);
    out.push({ kind: 'surrender', t: surrenders[i].t, score: 75, subject: ids[0], others: ids.slice(1), ships: ids });
    i = j + 1;
  }

  if (firstFire) {
    out.push({ kind: 'first_fire', t: firstFire.timestamp, score: 45, subject: firstFire.ship_id,
      target: firstFire.data?.target_id || null, others: [] });
  }
  const closest = closestApproach(simTrace);
  if (closest) out.push({ kind: 'closest', t: closest.t, score: 35, subject: closest.a, target: closest.b,
    others: [closest.b], distanceKm: closest.distanceKm });

  return out.sort((a, b) => a.t - b.t || b.score - a.score || (String(a.subject) < String(b.subject) ? -1 : 1));
}

/**
 * Closest approach between live ships of opposite fleets over the trace:
 * {t, a, b, distanceKm}, or null (no trace, one fleet, or the fleets never
 * come inside 150 km - nothing a two-ship shot could show).
 */
export function closestApproach(simTrace, maxKm = 150) {
  let best = null;
  for (const frame of simTrace || []) {
    const ships = Object.entries(frame?.ships || {}).filter(([, s]) => !s.destroyed && Array.isArray(s.pos));
    for (const [a, sa] of ships) {
      if (factionOf(a) !== 'alpha') continue;
      for (const [b, sb] of ships) {
        if (factionOf(b) === 'alpha') continue;
        const d = distKm(sa.pos, sb.pos);
        if (!best || d < best.distanceKm) best = { t: frame.t, a, b, distanceKm: d };
      }
    }
  }
  return best && best.distanceKm <= maxKm ? best : null;
}

/** Candidate shots for one highlight (desired times, before scheduling). */
function candidatesFor(h, lookahead, deathTime) {
  const alive = (id, t) => id && !(deathTime(id) <= t);
  const base = { kind: h.kind, t: h.t, subject: h.subject, others: h.others || [], score: h.score };
  switch (h.kind) {
    case 'kill': {
      const last = h.tLast ?? h.t;
      const pull = {
        ...base, type: 'pullback', kills: h.kills,
        // Live: from the tick the kill is on record (the blast follows
        // within a frame); the shot on screen keeps its minimum length, so
        // the cut may arrive with the blast under way (its shell lasts 5 s)
        start: lookahead ? h.t - PULLBACK_PRE_S : h.eventT, end: last + PULLBACK_POST_S,
        lockFrom: lookahead ? h.t - PULLBACK_PRE_S : h.eventT, lockTo: last + DETONATION_LOCK_S,
        startBy: lookahead ? h.t + 1.5 : h.eventT + 2.5
      };
      if (!lookahead) return [pull];
      // The approach: over the killer's shoulder while it still lives (a
      // dead ship is credited with kills its torpedoes score later),
      // otherwise the incoming fire seen at the victim
      const killer = h.kills[0].killer;
      const approach = {
        ...base, score: h.score - 1, start: h.t - PULLBACK_PRE_S - APPROACH_S, end: h.t - PULLBACK_PRE_S,
        type: 'runin', attacker: killer || null, partOf: 'kill'
      };
      if (h.dyingAt !== null && h.t - h.dyingAt >= MIN_SHOT_S + PULLBACK_PRE_S) {
        // A hulk waiting to blow: orbit it until just before it does
        Object.assign(approach, { type: 'orbit', kind: 'dying', start: Math.max(h.dyingAt - 0.5, approach.end - 9) });
      } else if (h.strike) Object.assign(approach, { impacts: h.strike.impacts, torpedoes: h.strike.torpedoes });
      else if (alive(killer, h.t)) Object.assign(approach, { type: 'shoulder', subject: killer, target: h.subject });
      return [approach, pull];
    }
    case 'dying':
      // Replay: only a fuse long enough to hold a shot (a chain's first
      // hulk is its kill's approach). Live: always - how long it burns
      // is not known yet
      if (lookahead && h.until - h.t < MIN_SHOT_S + PULLBACK_PRE_S) return [];
      if (!lookahead) return [{ ...base, type: 'orbit', start: h.t, end: Math.max(h.t + MIN_SHOT_S, h.until + 2) }];
      return [{ ...base, type: 'orbit', start: h.t - 0.5, end: Math.max(h.t + MIN_SHOT_S, h.until - PULLBACK_PRE_S) }];
    case 'strike': {
      const last = h.tLast ?? h.t;
      return [{ ...base, type: 'runin', attacker: h.others[0] || null, impacts: h.impacts, torpedoes: h.torpedoes,
        start: lookahead ? h.t - APPROACH_S : h.t, end: last + 2.5 }];
    }
    case 'salvo':
      return [{ ...base, type: 'shoulder', target: h.target, count: h.count,
        start: lookahead ? h.t - 1.5 : h.t,
        end: Math.max((h.tLast ?? h.t) + 5, Number.isFinite(h.lit) ? h.lit + SALVO_AFTER_LIGHTOFF_S : -Infinity) }];
    case 'cookoff':
      return [{ ...base, type: 'orbit', start: lookahead ? h.t - 1.5 : h.t, end: h.t + 3.5, impacts: [h.t] }];
    case 'disarmed':
      return [{ ...base, type: 'orbit', start: lookahead ? h.t - 1.5 : h.t, end: h.t + 3.5, impacts: [h.t] }];
    case 'plume_wash': {
      // Two-shot from the side: the emitter's jet and the ship inside it
      const start = lookahead ? h.t - 1.5 : h.t;
      return [{ ...base, type: 'wide', cast: [h.target, h.subject], target: h.target,
        start, end: Math.max(start + MIN_SHOT_S, Math.min(h.until + 1, start + 9)) }];
    }
    case 'surrender':
      // One ship: a slow orbit of it. Several: the group that struck together
      return [h.ships.length > 1
        ? { ...base, type: 'wide', cast: h.ships, start: lookahead ? h.t - 1.5 : h.t, end: h.t + 5 }
        : { ...base, type: 'orbit', start: lookahead ? h.t - 1.5 : h.t, end: h.t + 5 }];
    case 'first_fire':
      return [{ ...base, type: 'shoulder', target: h.target, start: lookahead ? h.t - 2 : h.t, end: h.t + 4 }];
    case 'closest':
      return [{ ...base, type: 'wide', cast: [h.subject, h.target].filter(Boolean),
        start: lookahead ? h.t - 3 : h.t, end: h.t + 3 }];
    default:
      return [];
  }
}

/** Free intervals of [a, b] not covered by placed shots (sorted by t0). */
function freeIntervals(placed, a, b) {
  const gaps = [];
  let cursor = a;
  for (const s of placed) {
    if (s.t1 <= cursor) continue;
    if (s.t0 >= b) break;
    if (s.t0 > cursor) gaps.push([cursor, s.t0]);
    cursor = Math.max(cursor, s.t1);
  }
  if (cursor < b) gaps.push([cursor, b]);
  return gaps;
}

function finishShot(c, t0, t1) {
  const { start, end, startBy, ...shot } = c;
  return { ...shot, t0, t1 };
}

/**
 * The battle's shot list: non-overlapping shots covering [0, Infinity),
 * each at least MIN_SHOT_S long, with no cut inside a detonation lock.
 * Replay (lookahead true): highlights are placed by score, so a kill keeps
 * its run-in and pull-back and lesser events fit round it. Live (lookahead
 * false): placed in time order as they happen; a shot on screen is only
 * ever cut by a later event, never rewritten.
 * @param {Array} events
 * @param {Object} deathInfo
 * @param {Object} opts - {lookahead = true, simTrace}
 * @returns {Array<{id, type, kind, t0, t1, t, subject, others, score, ...}>}
 */
export function buildShotList(events = [], deathInfo = {}, { lookahead = true, simTrace = null } = {}) {
  const highlights = buildHighlights(events, deathInfo, { simTrace: lookahead ? simTrace : null, causal: !lookahead });
  const deaths = new Map();
  for (const h of highlights) if (h.kind === 'kill') for (const k of h.kills) deaths.set(k.id, k.t);
  const deathTime = id => (deaths.has(id) ? deaths.get(id) : Infinity);
  // A strike that lands as (or just before) its target dies is that
  // kill's approach, not a shot of its own
  for (const h of highlights) {
    if (h.kind !== 'kill') continue;
    const strike = highlights.find(s => s.kind === 'strike' && s.subject === h.subject &&
      s.t <= h.t + 0.5 && (s.tLast ?? s.t) >= h.t - APPROACH_S);
    if (strike) { h.strike = strike; strike.absorbed = true; }
    const hulk = highlights.find(d => d.kind === 'dying' && d.subject === h.subject);
    if (hulk) hulk.absorbed = true;
  }
  const candidates = [];
  for (const h of highlights) {
    if (h.absorbed && lookahead) continue;
    candidates.push(...candidatesFor(h, lookahead, deathTime));
  }

  const placed = [];
  const insert = shot => {
    let i = placed.findIndex(s => s.t0 > shot.t0);
    if (i < 0) i = placed.length;
    placed.splice(i, 0, shot);
  };
  if (lookahead) {
    const byScore = candidates.map((c, i) => ({ c, i }))
      .sort((a, b) => b.c.score - a.c.score || a.c.start - b.c.start ||
        (String(a.c.subject) < String(b.c.subject) ? -1 : String(a.c.subject) > String(b.c.subject) ? 1 : a.i - b.i));
    for (const { c } of byScore) {
      const start = Math.max(0, c.start);
      // The largest free stretch of the wanted window (ties: the earliest)
      let best = null;
      for (const [a, b] of freeIntervals(placed, start, c.end)) {
        if (b - a >= MIN_SHOT_S && (!best || b - a > best[1] - best[0] + 1e-9)) best = [a, b];
      }
      if (!best) continue;
      const [a, b] = best;
      if (c.startBy !== undefined && a > c.startBy) continue;
      // Cut short by a better shot before its own detonation is over: drop
      if (c.lockTo !== undefined && b < c.lockTo && b < c.end) continue;
      if (c.lockFrom !== undefined && a > c.lockFrom + 1e-9 && a > start) {
        if (a > (c.startBy ?? c.lockFrom)) continue;
      }
      insert(finishShot(c, a, b));
    }
  } else {
    const inOrder = candidates.map((c, i) => ({ c, i }))
      .sort((a, b) => a.c.start - b.c.start || b.c.score - a.c.score ||
        (String(a.c.subject) < String(b.c.subject) ? -1 : String(a.c.subject) > String(b.c.subject) ? 1 : a.i - b.i));
    let cur = null;
    for (const { c } of inOrder) {
      let start = Math.max(0, c.start);
      const natural = cur ? cur.t1 : 0;    // where cur ends if nothing cuts it
      if (cur && start < natural) {
        start = Math.max(start, cur.t0 + MIN_SHOT_S);
        if (cur.lockTo !== undefined && start >= (cur.lockFrom ?? cur.t0) && start < cur.lockTo) start = cur.lockTo;
        if (start < natural && c.score < cur.score) start = natural;
      }
      // The wide view that follows a shot is a shot too: it keeps its
      // minimum length (it was already on screen when this event arrived)
      if (start > natural && start - natural < MIN_SHOT_S) start = natural + MIN_SHOT_S;
      if (start > (c.startBy ?? c.start + 2)) continue;
      if (cur && cur.t1 > start) cur.t1 = start;
      // (never dropped for being short: how long the event runs is the future)
      cur = finishShot(c, start, Math.max(c.end, start + MIN_SHOT_S));
      placed.push(cur);
    }
  }

  // Fill: hold a shot across a gap too short for its own shot, otherwise
  // the wide establishing view; in replay, long quiet stretches alternate
  // it with a chase / flyby of the next ship to matter
  const shots = [];
  let cursor = 0;
  let beauty = 0;
  const fill = (a, b, next) => {
    if (b - a < MIN_SHOT_S && shots.length) { shots.at(-1).t1 = b; return; }
    const hero = lookahead && next ? (next.type === 'shoulder' ? next.subject : next.attacker || next.subject) : null;
    let t = a;
    let wide = true;
    while (t < b) {
      const want = wide ? WIDE_FILL_S : BEAUTY_FILL_S;
      // Never leave a tail shorter than a shot
      const end = b - (t + want) < MIN_SHOT_S ? b : t + want;
      if (wide || !hero || !(deathTime(hero) > end)) {
        if (shots.length && shots.at(-1).type === 'wide' && shots.at(-1).kind === 'fleet') shots.at(-1).t1 = end;
        else shots.push({ type: 'wide', kind: 'fleet', t: t, t0: t, t1: end, subject: null, others: [], score: 0 });
      } else {
        shots.push({ type: beauty++ % 2 ? 'flyby' : 'chase', kind: 'beauty', t: t, t0: t, t1: end, subject: hero,
          target: next.type === 'shoulder' ? next.target : next.subject, others: [], score: 1 });
      }
      wide = !wide;
      t = end;
      if (!Number.isFinite(b) && !wide) { shots.at(-1).t1 = Infinity; break; }
    }
  };
  for (let i = 0; i < placed.length; i++) {
    const s = placed[i];
    if (!shots.length && s.t0 - cursor < MIN_SHOT_S) s.t0 = cursor;   // no sliver before the first shot
    if (s.t0 > cursor) fill(cursor, s.t0, s);
    shots.push(s);
    cursor = s.t1;
  }
  fill(cursor, Infinity, null);
  shots.forEach((s, i) => { s.id = i; });
  return shots;
}

/** The shot on screen at battle time t (shots from buildShotList). */
export function shotAt(shots, t) {
  if (!shots?.length) return null;
  let lo = 0;
  let hi = shots.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (shots[mid].t0 <= t) lo = mid;
    else hi = mid - 1;
  }
  return shots[lo];
}

/**
 * Highlights worth jumping to (the [ and ] keys), best moments only:
 * {t, kind, subject, score, label}. Built from the same scoring as the
 * shots, so "next highlight" lands where the director cuts.
 */
export function highlightList(events = [], deathInfo = {}, { simTrace = null, minScore = 45 } = {}) {
  const label = h => ({
    kill: h.kills?.length > 1 ? `${h.kills.length} kills` : 'kill', dying: 'dying hulk', strike: 'torpedo strike',
    salvo: `salvo x${h.count}`, cookoff: 'magazine cook-off', disarmed: 'disarmed', first_fire: 'first gun exchange',
    closest: 'closest approach', plume_wash: 'plume wash', surrender: h.ships?.length > 1 ? `${h.ships.length} surrender` : 'surrender'
  })[h.kind] || h.kind;
  const list = [];
  for (const h of buildHighlights(events, deathInfo, { simTrace })) {
    if (h.score < minScore && h.kind !== 'salvo' && h.kind !== 'closest') continue;
    // One entry per moment: a strike / cook-off folded into the kill it caused
    const prev = list.at(-1);
    if (prev && h.t - prev.t < 2) {
      if (h.score > prev.score) Object.assign(prev, { kind: h.kind, subject: h.subject, score: h.score, label: label(h) });
      continue;
    }
    list.push({ t: h.t, kind: h.kind, subject: h.subject, score: h.score, label: label(h) });
  }
  return list;
}

/** Time of the next (dir > 0) or previous highlight from t, or null. */
export function stepHighlight(list, t, dir, preroll = 3) {
  if (!list?.length) return null;
  if (dir > 0) {
    const next = list.find(h => h.t - preroll > t + 0.25);
    return next ? Math.max(0, next.t - preroll) : null;
  }
  // Previous: the last highlight whose start is clearly behind the playhead
  const before = list.filter(h => h.t - preroll < t - 1.5);
  return before.length ? Math.max(0, before.at(-1).t - preroll) : null;
}

/**
 * Shake sources of a recording: reactor blasts (yield and radius from the
 * reactor_blast event, defaults for old recordings) and torpedo impacts.
 * Sorted by time. Positions are sim metres, or `ship` to read at time t.
 */
export function buildShakeSources(events = [], deathInfo = {}) {
  const sources = [];
  const blasted = new Set();
  for (const e of events || []) {
    if (!Number.isFinite(e?.timestamp)) continue;
    if (e.event_type === 'reactor_blast') {
      blasted.add(e.ship_id);
      sources.push({ t: shipDetonationTime((deathInfo || {})[e.ship_id] || null, e.timestamp),
        position: Array.isArray(e.data?.position) ? e.data.position : null, ship: e.ship_id,
        energyGj: num(e.data?.yield_gj, 400), reachKm: 1.5 * num(e.data?.radius_km, DEFAULT_BLAST_KM) });
    } else if (e.event_type === 'torpedo_impact' && e.data?.target_id) {
      sources.push({ t: e.timestamp, position: null, ship: e.data.target_id,
        energyGj: num(e.data?.total_damage_gj, 20), reachKm: 6 });
    } else if (e.event_type === 'magazine_cookoff') {
      sources.push({ t: e.timestamp, position: null, ship: e.ship_id, energyGj: 40 + 10 * num(e.data?.secondary_gj), reachKm: 8 });
    }
  }
  for (const [id, info] of Object.entries(deathInfo || {})) {
    if (blasted.has(id) || !Number.isFinite(info?.time)) continue;
    sources.push({ t: shipDetonationTime(info), position: Array.isArray(info.position) ? info.position : null, ship: id,
      energyGj: 400, reachKm: 1.5 * DEFAULT_BLAST_KM });
  }
  return sources.sort((a, b) => a.t - b.t);
}

export const SHAKE_DECAY_PER_S = 2.6;
export const SHAKE_LIFE_S = 3.0;
const SHAKE_MAX_RAD = 0.02;

/**
 * Camera shake at battle time t: an angular amplitude (radians, capped)
 * and a unit-ish 2D direction. A pure function of t, so a frame is the
 * same however the playhead got there. Each source contributes
 * sqrt(energy) x (reach / distance)^1.5 (1 inside its reach), decaying
 * exponentially on BATTLE time from its event.
 * @param {Array} sources - buildShakeSources()
 * @param {number} t - battle seconds
 * @param {Function} distanceKm - source -> camera distance in km (or null)
 * @returns {{amp: number, x: number, y: number}}
 */
export function shakeAt(sources, t, distanceKm) {
  let amp = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < (sources?.length || 0); i++) {
    const s = sources[i];
    const age = t - s.t;
    if (age < 0) break;
    if (age >= SHAKE_LIFE_S) continue;
    const d = distanceKm(s);
    if (!(d >= 0)) continue;
    const near = Math.min(1, (s.reachKm / Math.max(d, 1e-6)) ** 1.5);
    const a = 0.012 * Math.sqrt(s.energyGj / 500) * near * Math.exp(-SHAKE_DECAY_PER_S * age) *
      (1 - age / SHAKE_LIFE_S);
    amp += a;
    x += a * Math.sin(t * 37.1 + i * 1.7);
    y += a * Math.sin(t * 43.7 + 1.3 + i * 2.3);
  }
  if (amp > SHAKE_MAX_RAD) { const k = SHAKE_MAX_RAD / amp; amp *= k; x *= k; y *= k; }
  return { amp, x, y };
}

/**
 * Handheld drift: a slow two-axis wander (radians) on battle time, far
 * below the shake, so a held shot never looks nailed to a tripod.
 */
export function handheldAt(t, scale = 1) {
  return {
    x: scale * 0.0032 * (Math.sin(t * 0.37 + 0.6) + 0.45 * Math.sin(t * 0.91 + 2.1)),
    y: scale * 0.0024 * (Math.sin(t * 0.29 + 1.9) + 0.45 * Math.sin(t * 0.77 + 0.3))
  };
}

/**
 * Push-in on impacts: 0..1 pulse that snaps in at each impact time and
 * relaxes over about a second (battle time).
 */
export function impactPulse(impacts, t) {
  let p = 0;
  for (const ti of impacts || []) {
    const age = t - ti;
    if (age < -0.15 || age > 1.6) continue;
    p = Math.max(p, age < 0 ? 1 + age / 0.15 : Math.exp(-age * 2.4));
  }
  return p;
}
