import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDamageTimeline, damageStateAt, marksUpTo, craterHeat, ventStrength, ventLife,
  shipStateAt, worldToLocalDir, localToWorldDir, localBasis
} from '../src/DamageTimeline.js';
import { visualImpactTime } from '../src/render/DamageFX.js';

// Two ships on the x axis: alpha at the origin facing +x, beta 100 km out
// facing back at it. Beta's hull fails from t=20, it dies spiralling at 30.
function recording() {
  const trace = [];
  for (let t = 0; t <= 40; t++) {
    const bHull = t < 12 ? 100 : t < 20 ? 60 : t < 30 ? 30 : 20;
    trace.push({
      t,
      ships: {
        alpha: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], hull: 100 },
        beta: {
          pos: [100000, 0, 0], vel: [-500, 0, 0], fwd: [-1, 0, 0], hull: bHull,
          dying: t >= 30 && t < 36, destroyed: t >= 36
        }
      },
      torpedoes: t === 9 ? [{ id: 'torp_1', pos: [90000, 0, 20000] }] : []
    });
  }
  const events = [
    { timestamp: 5, event_type: 'hit', ship_id: 'beta',
      data: { shooter_id: 'alpha', hit_location: 'nose', kinetic_energy_gj: 3, penetrated: false, projectile_id: 'p1' } },
    { timestamp: 11, event_type: 'hit', ship_id: 'beta',
      data: { shooter_id: 'alpha', hit_location: 'lateral', kinetic_energy_gj: 8, penetrated: true, critical_hit: true } },
    { timestamp: 11, event_type: 'module_damaged', ship_id: 'beta', data: { module_name: 'Crew Quarters', damage_gj: 2 } },
    { timestamp: 10, event_type: 'torpedo_impact', ship_id: 'alpha',
      data: { torpedo_id: 'torp_1', target_id: 'beta', total_damage_gj: 24.5, hit_location: 'lateral' } },
    { timestamp: 10, event_type: 'armor_damage', ship_id: 'beta',
      data: { location: 'lateral', damage_gj: 24.5, penetrated: true, source: 'kinetic' } },
    { timestamp: 10, event_type: 'module_destroyed', ship_id: 'beta', data: { module_name: 'Main Fuel Tank' } },
    // a hit on the other ship, never penetrating
    { timestamp: 15, event_type: 'hit', ship_id: 'alpha',
      data: { shooter_id: 'beta', hit_location: 'nose', kinetic_energy_gj: 1, penetrated: false } }
  ];
  return { events, sim_trace: trace };
}

test('hits and torpedo strikes become marks on the ship that was hit', () => {
  const tl = buildDamageTimeline(recording());
  const beta = tl.ships.get('beta');
  const combat = beta.marks.filter(m => m.kind === 'slug' || m.kind === 'torpedo');
  assert.deepEqual(combat.map(m => [m.kind, m.t, m.section]), [
    ['slug', 5, 'nose'], ['torpedo', 10, 'lateral'], ['slug', 11, 'lateral']
  ]);
  assert.equal(tl.ships.get('alpha').marks.length, 1);
  // torpedo breach is read from the same-tick kinetic armour event
  const torp = combat[1];
  assert.equal(torp.penetrated, true);
  assert.equal(torp.vent, 'fuel');
  const slug = combat[2];
  assert.equal(slug.vent, 'atmo');       // crew quarters breached
  assert.equal(combat[0].vent, null);    // armour held: scorch, no venting
  // marks are globally time sorted and uniquely identified
  const ts = tl.marks.map(m => m.t);
  assert.deepEqual(ts, [...ts].sort((a, b) => a - b));
  assert.equal(new Set(tl.marks.map(m => m.uid)).size, tl.marks.length);
});

test('aim follows the attacker bearing in the ship frame', () => {
  const tl = buildDamageTimeline(recording());
  const [nose] = tl.ships.get('beta').marks;
  // alpha is dead ahead of beta (beta faces -x toward it)
  assert.ok(nose.approach[2] > 0.99, `approach ${nose.approach}`);
  assert.equal(nose.aim.end, 1);
  // the torpedo came in from +z world = beta's local +x flank
  const torp = tl.ships.get('beta').marks.find(m => m.kind === 'torpedo');
  assert.equal(torp.aim.end, 0);
  assert.ok(torp.approach[0] > 0.5, `approach ${torp.approach}`);
  // local basis matches three.js lookAt(fwd) with +Y up
  const B = localBasis([-1, 0, 0]);
  assert.deepEqual(B.z, [-1, 0, 0]);
  assert.deepEqual(B.x.map(v => Math.round(v)), [0, 0, 1]);
  const w = localToWorldDir([-1, 0, 0], worldToLocalDir([-1, 0, 0], [0.6, 0.8, 0]));
  assert.ok(Math.abs(w[0] - 0.6) < 1e-9 && Math.abs(w[1] - 0.8) < 1e-9);
});

test('the timeline is deterministic: rebuilding yields identical marks', () => {
  const strip = (tl) => tl.marks.map(m => ({
    uid: m.uid, t: m.t, kind: m.kind, seed: m.seed, aim: m.aim, severity: m.severity,
    vent: m.vent, ventSize: m.ventSize, debris: m.debris
  }));
  assert.deepEqual(strip(buildDamageTimeline(recording())), strip(buildDamageTimeline(recording())));
});

test('damage state is a pure function of time, so scrubbing back removes damage', () => {
  const tl = buildDamageTimeline(recording());
  const late = damageStateAt(tl, 'beta', 33);
  const early = damageStateAt(tl, 'beta', 4);
  const again = damageStateAt(tl, 'beta', 33);
  assert.deepEqual(late, again);
  assert.equal(early.count, 0);
  assert.equal(early.breaches, 0);
  assert.equal(early.smoke, 0);
  assert.equal(early.sparks, 0);
  assert.ok(late.count > 3);
  assert.ok(late.breaches >= 2);
  assert.equal(late.dying, true);
  assert.equal(late.smoke, 1);
  assert.equal(late.sparks, 1);
  // hull 60%: sparks/flicker begin, smoke not yet heavy; hull 30%: smoking
  const mid = damageStateAt(tl, 'beta', 15);
  const worse = damageStateAt(tl, 'beta', 25);
  assert.ok(mid.sparks > 0 && mid.sparks < 1);
  assert.ok(worse.smoke > mid.smoke);
  assert.equal(damageStateAt(tl, 'alpha', 33).smoke, 0);
  assert.equal(marksUpTo(tl.ships.get('beta').marks, 10.5), 2);
});

test('death spiral schedules breaches and secondaries walking toward the reactor', () => {
  const tl = buildDamageTimeline(recording());
  const spiral = tl.ships.get('beta').marks.filter(m => ['breach', 'pop', 'secondary'].includes(m.kind));
  assert.ok(spiral.length >= 6);
  for (const m of spiral) assert.ok(m.t >= 30 && m.t < 36, `t ${m.t}`);
  assert.equal(spiral.filter(m => m.kind === 'breach').length, 2);
  const pops = spiral.filter(m => m.blast);
  assert.ok(pops.length >= 4);
  assert.ok(pops[0].aim.axial > pops[pops.length - 1].aim.axial, 'secondaries walk aft');
  // nothing is scheduled for a ship that never entered a death spiral
  assert.ok(tl.ships.get('alpha').marks.every(m => !m.blast));
});

test('craters cool to black and vents bleed down', () => {
  assert.equal(craterHeat(-1), 0);
  assert.ok(craterHeat(0.2) > 1.0, 'white hot on impact');
  assert.ok(craterHeat(3) > 0.3 && craterHeat(3) < 0.9, 'orange after a few seconds');
  assert.ok(craterHeat(12) > 0 && craterHeat(12) < 0.15, 'dull ember at 12 s');
  assert.equal(craterHeat(30), 0, 'cold scorch');
  for (let a = 0.5; a < 20; a += 0.5) assert.ok(craterHeat(a + 0.5) <= craterHeat(a));
  const tl = buildDamageTimeline(recording());
  const breach = tl.ships.get('beta').marks.find(m => m.kind === 'torpedo');
  assert.equal(ventStrength(breach, -0.1), 0);
  assert.ok(ventStrength(breach, 1) > 0.5);
  assert.ok(ventStrength(breach, 10) < ventStrength(breach, 1));
  assert.equal(ventStrength(breach, ventLife(breach) + 1), 0);
  const scorch = tl.ships.get('beta').marks[0];
  assert.equal(ventStrength(scorch, 1), 0);
});

test('ship pose lerps between trace frames and flags switch at their frame', () => {
  const rec = recording();
  const st = shipStateAt(rec.sim_trace, 'beta', 29.5);
  assert.equal(st.pos[0], 100000);
  assert.equal(st.dying, false);
  assert.equal(shipStateAt(rec.sim_trace, 'beta', 30).dying, true);
  assert.ok(Math.abs(shipStateAt(rec.sim_trace, 'beta', 11.5).hull - 0.8) < 1e-9);
});

test('coilgun marks use the travel-adjusted impact time when known', () => {
  const loader = {
    projectileHits: new Map([['p1', { impact_position: [1000, 0, 0] }]]),
    lastProjectileState: new Map([['p1', { time: 4, pos: [0, 0, 0], vel: [2000, 0, 0] }]])
  };
  const e = recording().events[0];
  assert.equal(visualImpactTime(loader, e), 4.5);
  const tl = buildDamageTimeline(recording(), { impactTime: (ev) => visualImpactTime(loader, ev) });
  assert.equal(tl.ships.get('beta').marks[0].t, 4.5);
  assert.equal(visualImpactTime(loader, recording().events[1]), 11);
});

test('a backward seek replays vent slots even when craters never wrap', async () => {
  const { rewindStart } = await import('../src/render/DamageFX.js');
  // 60 marks, every other one vents: 30 vents > a 20-slot vent buffer,
  // while 60 marks fit the crater buffer (no crater wrap)
  const marks = Array.from({ length: 60 }, (_, i) => ({ t: i, index: i, vent: i % 2 ? 'atmo' : null }));
  const vents = marks.filter(m => m.vent);
  // resolved through t=59; seek back to t=30.5: vents 1,3,...,29 (15) all
  // must be rewritten, the oldest being mark 1
  assert.equal(rewindStart(marks, vents, 30.5, 60, 192, 20), 1);
  // seek back to t=50.5: 25 vents so far, slots hold the last 20 -> from vent #5 (mark 11)
  assert.equal(rewindStart(marks, vents, 50.5, 60, 192, 20), 11);
  // nothing wraps: keep going from where we are
  assert.equal(rewindStart(marks, vents, 50.5, 60, 192, 40), 60);
  // crater wrap alone: last 16 marks up to t=40.5 -> from mark 25
  assert.equal(rewindStart(marks, [], 40.5, 60, 16, 40), 25);
});
