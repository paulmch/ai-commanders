/**
 * Cut a small fixture of real gun hits and misses out of a recording, for
 * tests/scale.test.js (hits land on the true-size hull, misses clear it),
 * and measure - over the WHOLE recording - how far the engine's recorded
 * impact points and miss lines are from the real hull.
 *
 *   node visualizer/scripts/make_scale_fixture.mjs <recording.json> [out.json]
 *
 * Default recording: visualizer/public/recordings/p5_sonnet55_vs_sonnet5_c.json
 * (an LLM gun fight: 524 rounds). The fixture keeps, per chosen round, the
 * trace frames of its last seconds (target, shooter and the round itself)
 * and its shot / hit / miss events - nothing else of the battle.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { REAL_LENGTH_M } from '../src/TrackDisplay.js';
import {
  makeSample, sampleTrack, closestPass, passClearance, HULL_RADIUS_PER_LENGTH
} from '../src/render/WeaponTimeline.js';

const ROOT = new URL('../', import.meta.url);
const src = process.argv[2] || fileURLToPath(new URL('public/recordings/p5_sonnet55_vs_sonnet5_c.json', ROOT));
const out = process.argv[3] || fileURLToPath(new URL('tests/fixtures/scale_hits.json', ROOT));
const rec = JSON.parse(readFileSync(src, 'utf8'));
const trace = rec.sim_trace, events = rec.events;

const types = {};
for (const side of ['alpha_fleet', 'beta_fleet']) for (const s of rec[side]?.ships || []) types[s.ship_id] = s.ship_type;
const track = (get) => {
  const m = new Map();
  for (const f of trace) for (const [id, st] of get(f)) {
    if (!m.has(id)) m.set(id, { id, seed: 0, start: f.t, samples: [] });
    m.get(id).samples.push({ t: f.t, state: st });
  }
  return m;
};
const ships = track(f => Object.entries(f.ships || {}));
const slugs = track(f => (f.projectiles || []).map(p => [p.id, p]));

/** Distance (m) from point p to the engine's hull capsule axis segment of a ship at time t. */
export function axisDistance(p, ship, lengthM, t) {
  const s = makeSample();
  sampleTrack(ship, t, s);
  const fl = Math.hypot(...s.fwd) || 1, a = s.fwd.map(v => v / fl);
  const r = [p[0] - s.pos[0], p[1] - s.pos[1], p[2] - s.pos[2]];
  const along = Math.max(-lengthM / 2, Math.min(lengthM / 2, r[0] * a[0] + r[1] * a[1] + r[2] * a[2]));
  return Math.hypot(r[0] - a[0] * along, r[1] - a[1] * along, r[2] - a[2] * along);
}

const pct = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1) + 0.5))]; };
const hits = [], misses = [];
for (const e of events) {
  const d = e.data || {};
  const tr = slugs.get(d.projectile_id);
  if (!tr) continue;
  const last = tr.samples.at(-1);
  if (e.event_type === 'hit' && Array.isArray(d.impact_position)) {
    const target = e.ship_id, L = REAL_LENGTH_M[types[target]];
    const speed = Math.hypot(...last.state.vel);
    const tHit = last.t + Math.hypot(...d.impact_position.map((v, k) => v - last.state.pos[k])) / speed;
    hits.push({ e, id: d.projectile_id, target, L, tHit, axis: axisDistance(d.impact_position, ships.get(target), L, tHit) });
  } else if (e.event_type === 'miss') {
    const target = d.target_id || last.state.target, L = REAL_LENGTH_M[types[target]];
    const ship = ships.get(target);
    if (!ship) continue;
    const pass = closestPass(tr, ship, Math.max(tr.start, last.t - 2), last.t + 6);
    if (!pass) continue;
    misses.push({ e, id: d.projectile_id, target, L, tPass: pass.t, sep: passClearance(pass, L).separation });
  }
}
const hitAxis = hits.map(h => h.axis);
const summary = {
  recording: src.split('/').pop(),
  hits: hits.length,
  hit_axis_distance_m: {
    median: +pct(hitAxis, 0.5).toFixed(1), mean: +(hitAxis.reduce((a, b) => a + b, 0) / hitAxis.length).toFixed(1),
    p90: +pct(hitAxis, 0.9).toFixed(1), max: +Math.max(...hitAxis).toFixed(1)
  },
  hits_inside_hull_radius: hits.filter(h => h.axis <= h.L * HULL_RADIUS_PER_LENGTH).length,
  misses: misses.length,
  misses_through_hull: misses.filter(m => m.sep <= m.L * HULL_RADIUS_PER_LENGTH).length,
  note: 'axis distance = recorded impact_position to the hull axis segment (engine capsule, radius = length / 8) at the impact instant'
};

// A spread of hits (on the hull .. far outside it) and of misses (through the hull .. wide)
const byAxis = [...hits].sort((a, b) => a.axis - b.axis);
const pickHits = [0, 0.15, 0.35, 0.5, 0.65, 0.8, 0.92, 1].map(q => byAxis[Math.round(q * (byAxis.length - 1))]);
const bySep = [...misses].sort((a, b) => a.sep - b.sep);
const through = bySep.filter(m => m.sep <= m.L * HULL_RADIUS_PER_LENGTH);
const pickMiss = [through[0], through[Math.floor(through.length / 2)], through.at(-1),
  bySep.find(m => m.sep > m.L * HULL_RADIUS_PER_LENGTH * 1.3), bySep.find(m => m.sep > 300), bySep.at(-1)].filter(Boolean);

const r1 = (v) => Math.round(v * 10) / 10, r4 = (v) => Math.round(v * 1e4) / 1e4;
const keepShip = (s) => ({ pos: s.pos.map(r1), vel: s.vel.map(r1), fwd: s.fwd.map(r4), ...(s.up ? { up: s.up.map(r4) } : {}),
  destroyed: !!s.destroyed, dying: !!s.dying, hull: s.hull });
const frames = new Map();
const want = (t, kind, id) => {
  if (!frames.has(t)) frames.set(t, { ships: new Set(), projectiles: new Set() });
  frames.get(t)[kind].add(id);
};
const picked = [...pickHits.map(h => ({ ...h, kind: 'hit', tEnd: h.tHit })), ...pickMiss.map(m => ({ ...m, kind: 'miss', tEnd: m.tPass }))];
const keepEvents = [];
for (const p of picked) {
  const tr = slugs.get(p.id), last = tr.samples.at(-1);
  const shooter = tr.samples[0].state.source;
  for (const f of trace) {
    if (f.t < last.t - 5 || f.t > last.t + 8) continue;
    want(f.t, 'ships', p.target);
    if (shooter) want(f.t, 'ships', shooter);
    if (f.t <= last.t) want(f.t, 'projectiles', p.id);
  }
  for (const e of events) {
    if (e.data?.projectile_id !== p.id || !['shot_fired', 'hit', 'miss'].includes(e.event_type)) continue;
    const d = e.data;
    keepEvents.push({ timestamp: e.timestamp, event_type: e.event_type, ship_id: e.ship_id, data: {
      projectile_id: d.projectile_id, target_id: d.target_id, weapon_slot: d.weapon_slot, weapon_name: d.weapon_name,
      shooter_id: d.shooter_id, source_ship_id: d.source_ship_id, kinetic_energy_gj: d.kinetic_energy_gj,
      hit_location: d.hit_location, impact_angle_deg: d.impact_angle_deg, penetrated: d.penetrated,
      impact_position: d.impact_position, closest_approach_km: d.closest_approach_km, detection: d.detection,
      miss_margin_m: d.miss_margin_m
    } });
  }
}
const sim_trace = [...frames.keys()].sort((a, b) => a - b).map(t => {
  const f = trace.find(x => x.t === t), w = frames.get(t);
  return {
    t,
    ships: Object.fromEntries([...w.ships].filter(id => f.ships[id]).map(id => [id, keepShip(f.ships[id])])),
    projectiles: (f.projectiles || []).filter(p => w.projectiles.has(p.id)).map(p => ({
      id: p.id, pos: p.pos.map(r1), vel: p.vel.map(r1), source: p.source, target: p.target }))
  };
});
const fixture = {
  about: 'Real gun hits and misses cut from an LLM battle by visualizer/scripts/make_scale_fixture.mjs (do not edit)',
  summary,
  ship_types: Object.fromEntries(Object.entries(types).filter(([id]) => sim_trace.some(f => f.ships[id]))),
  rounds: picked.map(p => ({ id: p.id, kind: p.kind, target: p.target,
    ...(p.kind === 'hit' ? { recorded_axis_distance_m: +p.axis.toFixed(1) } : { recorded_separation_m: +p.sep.toFixed(1) }) })),
  events: keepEvents.sort((a, b) => a.timestamp - b.timestamp),
  sim_trace
};
writeFileSync(out, JSON.stringify(fixture));
console.log(JSON.stringify(summary, null, 1));
console.log(fixture.rounds.map(r => JSON.stringify(r)).join('\n'));
console.log('frames', sim_trace.length, 'bytes', JSON.stringify(fixture).length, '->', out);
