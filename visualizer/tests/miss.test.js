import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';

const {
  WeaponTimeline, makeSample, torpedoMissOpacity, missMarkOpacity, missMarkText, missOpacity,
  TORPEDO_MISS_LIFE, MISS_MARK_S, MISS_MARK_MAX_KM, MISS_LIFE
} = await import('../src/render/WeaponTimeline.js');
const { BattleLoader } = await import('../src/BattleLoader.js');
const { layoutTactical } = await import('../src/TacticalLayout.js');

const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} ${a} != ${b}`);

/**
 * A small recording: beta_1 sits still, alpha's torpedo t1 runs past it at
 * 3 km/s and misses at t = 10 (tank dry at 6), torpedo t2 is blinded at 5
 * and culled leaving the battle at 20 (closest pass ~0.4 km at t = 8), t3 is
 * culled 400 km out, slug s1 misses at 10 and leaves the trace.
 */
function recording() {
  const trace = [];
  for (let t = 0; t <= 30; t++) {
    const f = { t, ships: {
      alpha_1: { pos: [-90000, 0, 0], vel: [0, 0, 0], fwd: [1, 0, 0], thrust: 0 },
      beta_1: { pos: [0, 0, 0], vel: [0, 0, 0], fwd: [-1, 0, 0], thrust: 0 }
    }, projectiles: [], torpedoes: [] };
    if (t >= 1 && t <= 10) f.torpedoes.push({ id: 't1', pos: [-30000 + 3000 * t, 2400, 0], vel: [3000, 0, 0], source: 'alpha_1', target: 'beta_1', dv_remaining: t < 6 ? 1 : 0, thr: 0 });
    if (t >= 1 && t <= 20) f.torpedoes.push({ id: 't2', pos: [-16000 + 2000 * t, 400, 0], vel: [2000, 0, 0], source: 'alpha_1', target: 'beta_1', dv_remaining: 1, thr: 0, disabled: t >= 5 });
    if (t >= 1 && t <= 12) f.torpedoes.push({ id: 't3', pos: [-400000, 400000 + 1000 * t, 0], vel: [0, 1000, 0], source: 'alpha_1', target: 'beta_1', dv_remaining: 1, thr: 0 });
    if (t >= 1 && t <= 10) f.projectiles.push({ id: 's1', pos: [-50000 + 5000 * t, 30, 0], vel: [5000, 0, 0], source: 'alpha_1', target: 'beta_1' });
    trace.push(f);
  }
  const ev = (timestamp, event_type, data) => ({ timestamp, event_type, ship_id: 'alpha_1', data });
  const events = [
    ev(0, 'torpedo_launched', { torpedo_id: 't1', target_id: 'beta_1' }),
    ev(0, 'torpedo_launched', { torpedo_id: 't2', target_id: 'beta_1' }),
    ev(0, 'torpedo_launched', { torpedo_id: 't3', target_id: 'beta_1' }),
    ev(0, 'shot_fired', { projectile_id: 's1', target_id: 'beta_1', weapon_slot: 'weapon_0', weapon_name: 'Coilgun Battery Mk3' }),
    ev(5, 'pd_torpedo_disabled', { torpedo_id: 't2' }),
    ev(6, 'torpedo_fuel_exhausted', { torpedo_id: 't1' }),
    ev(10, 'torpedo_miss', { torpedo_id: 't1', target_id: 'beta_1', closest_approach_km: 2.4 }),
    ev(10, 'miss', { projectile_id: 's1', target_id: 'beta_1', closest_approach_km: 0.03 }),
    ev(20, 'torpedo_miss', { torpedo_id: 't2', target_id: 'beta_1', closest_approach_km: 0.4, reason: 'left_battle' }),
    ev(12, 'torpedo_miss', { torpedo_id: 't3', target_id: 'beta_1', closest_approach_km: 565, reason: 'left_battle' })
  ];
  return { events, sim_trace: trace };
}
function timeline() {
  const l = new BattleLoader();
  l.parse(recording());
  return new WeaponTimeline(l);
}

test('a missed torpedo flies on along its last velocity past its miss, then fades out', () => {
  const tl = timeline();
  const tr = tl.torpedoes.get('t1');
  assert.equal(tr.missAt, 10);
  near(tr.end, 10 + TORPEDO_MISS_LIFE, 1e-12, 'drawn on after the miss');
  const s = makeSample();
  for (const dt of [0.5, 1, 2, 3.5]) {
    tl.sample(tr, 10 + dt, s);
    near(s.pos[0], -30000 + 3000 * 10 + 3000 * dt, 1e-6, `x at +${dt}`);
    near(s.pos[1], 2400, 1e-6);
  }
  // the fade: 1 up to the miss, monotonic, 0 at the end
  assert.equal(torpedoMissOpacity(-1), 1);
  assert.equal(torpedoMissOpacity(0), 1);
  let prev = 1;
  for (let a = 0.05; a <= TORPEDO_MISS_LIFE + 1; a += 0.05) {
    const o = torpedoMissOpacity(a);
    assert.ok(o <= prev && o >= 0);
    prev = o;
  }
  assert.equal(torpedoMissOpacity(TORPEDO_MISS_LIFE), 0);
  assert.ok(TORPEDO_MISS_LIFE >= 2 && TORPEDO_MISS_LIFE <= 6, 'long enough to follow, short enough not to clutter');
  // it is no longer an inbound round for the wave chips once it has missed
  assert.ok(tl.roundsAt(9.5).some(r => r.id === 't1'));
  assert.ok(!tl.roundsAt(10.5).some(r => r.id === 't1'));
  // a round with no outcome that leaves the trace coasts and fades too (never a blink)
  const t3 = tl.torpedoes.get('t3');
  assert.ok(t3.end > 12);
});

test('miss markers: at the closest point, with the cause, a few seconds, none for a far cull', () => {
  const tl = timeline();
  const marks = tl.missMarks;
  const byId = new Map(marks.map(m => [m.track.id, m]));
  assert.ok(!byId.has('t3'), 'a round culled 565 km out gets no marker');
  const m1 = byId.get('t1');
  assert.equal(m1.text, 'MISS 2.4 km · TANK DRY');
  assert.ok(Math.abs(m1.t - 10) <= 1);
  near(Math.hypot(...m1.offset) / 1000, 2.4, 0.15, 'at the closest point');
  const m2 = byId.get('t2');
  assert.equal(m2.text, 'MISS 400 m · BLINDED');
  near(m2.t, 8, 0.6, 'a cull: the closest point is found on the trace, not at the cull');
  // lifetime and fade
  assert.equal(missMarkOpacity(-0.1), 0);
  assert.equal(missMarkOpacity(MISS_MARK_S), 0);
  assert.ok(missMarkOpacity(1) === 1 && missMarkOpacity(MISS_MARK_S - 0.5) < 1);
  assert.ok(MISS_MARK_S >= 2 && MISS_MARK_S <= 6 && MISS_MARK_MAX_KM >= 5);
  assert.equal(tl.missMarksAt(m1.t - 0.01).filter(x => x.mark === m1).length, 0);
  assert.equal(tl.missMarksAt(m1.t + 1).filter(x => x.mark === m1).length, 1);
  assert.equal(tl.missMarksAt(m1.t + MISS_MARK_S + 0.01).filter(x => x.mark === m1).length, 0);
  // texts
  assert.equal(missMarkText(0.00002, 'left'), 'MISS < 1 m · LEFT BATTLE');
  assert.equal(missMarkText(0.95, null), 'MISS 950 m');
  assert.equal(missMarkText(3.21, 'lost'), 'MISS 3.2 km · TARGET LOST');
  // seek away and back: the same markers at the same places
  const a = JSON.stringify(tl.missMarksAt(11).map(x => [x.mark.text, x.alpha, x.pos]));
  tl.missMarksAt(25); tl.missMarksAt(0);
  assert.equal(JSON.stringify(tl.missMarksAt(11).map(x => [x.mark.text, x.alpha, x.pos])), a);
});

test('miss markers obey the label density key: shown in auto and full, never in minimal', () => {
  const view = { width: 1280, height: 720 };
  const misses = [{ key: 't1', x: 600, y: 300, tx: 640, ty: 360, text: 'MISS 2.4 km · TANK DRY', alpha: 1 }];
  for (const mode of ['auto', 'full']) {
    const out = layoutTactical({ view, misses, mode });
    assert.equal(out.misses.length, 1);
    assert.ok(out.misses[0].box, mode);
  }
  assert.equal(layoutTactical({ view, misses, mode: 'minimal' }).misses.length, 0);
});

test('a missed slug streaks on past its target along its last velocity and fades (no marker)', () => {
  const tl = timeline();
  const s1 = tl.slugs.get('s1');
  assert.equal(s1.missAt, 10);
  near(s1.end, 10 + MISS_LIFE, 1e-12);
  const s = makeSample();
  tl.sample(s1, 11, s);
  near(s.pos[0], 5000 * 11 - 50000, 1e-6, 'ballistic past the miss');
  let prev = 1;
  for (let a = 0; a <= MISS_LIFE; a += 0.25) { const o = missOpacity(a); assert.ok(o <= prev); prev = o; }
  assert.equal(missOpacity(MISS_LIFE), 0);
  assert.ok(!tl.missMarks.some(m => m.track === s1));
});

test('old and real recordings: miss handling stays finite and a round never blinks out short of its pass', () => {
  const path = new URL('../public/recordings/p5w5_b8_opus_v_sonnet55_admirals_haiku_captains.json', import.meta.url);
  if (!existsSync(path)) return;
  const l = new BattleLoader();
  l.parse(JSON.parse(readFileSync(path, 'utf8')));
  const tl = new WeaponTimeline(l);
  const s = makeSample();
  let checked = 0;
  for (const tr of tl.torpedoes.values()) {
    if (!(tr.missAt < Infinity)) continue;
    assert.ok(tr.end > tr.missAt, tr.id);
    for (let t = tr.missAt; t < tr.end; t += 0.5) {
      tl.sample(tr, t, s);
      assert.ok(s.pos.every(Number.isFinite), tr.id);
    }
    checked++;
  }
  assert.ok(checked >= 40);
  // the seven stern-chase misses on alpha_3 are marked "tank dry", 2.1-2.8 km
  const dry = tl.missMarks.filter(m => m.target === 'alpha_3');
  assert.ok(dry.length >= 7 && dry.every(m => m.cause === 'dry' && m.cpaKm > 2 && m.cpaKm < 3), dry.map(m => m.text).join(', '));
  // retargeted rounds keep one continuous track through the switch
  for (const m of tl.moments.filter(x => x.kind === 'retarget')) {
    const tr = m.track;
    let prev = null;
    for (let t = m.time - 2; t < m.time + 2; t += 0.1) {
      if (t < tr.start || t >= tr.end) continue;
      tl.sample(tr, t, s);
      if (prev) assert.ok(Math.hypot(s.pos[0] - prev[0], s.pos[1] - prev[1], s.pos[2] - prev[2]) < 5000, `${tr.id} jumps at ${t}`);
      prev = s.pos.slice();
    }
  }
});
