import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDamageTimeline, engineStateAt, blastsAt, blastScorchCount, damageStateAt, marksUpTo, wrecksAt, BLAST_WASH_S,
  BLAST_EXPAND_S, blastFront, blastArrival
} from '../src/DamageTimeline.js';
import { blastShellAt, SHELL_EXPAND_S, SHELL_LIFE_S, KILL_EXPAND_S, KILL_LIFE_S } from '../src/render/BlastShell.js';
import { damagedNozzle } from '../src/render/Torch.js';
import { mountVentTime, MOUNT_VENT_S } from '../src/render/DamageFX.js';
import { blastKinds } from '../src/render/Explosion.js';
import { WeaponTimeline } from '../src/render/WeaponTimeline.js';
import { BattleLoader } from '../src/BattleLoader.js';
import { formatLogEvent } from '../src/HudIndex.js';

// Pass-5 damage events (docs/pass-5-recording-additions.md), field for field
const ship = (pos, extra = {}) => ({ pos, vel: [0, 0, 0], fwd: [1, 0, 0], hull: 100, thrust: 1, ...extra });
function recording(withEvents = true) {
  const sim_trace = [];
  for (let t = 0; t <= 12; t++) {
    sim_trace.push({ t, ships: {
      a: ship([0, 0, 0]), b: ship([4000, 3000, 0]), dead: ship([4000, 8000, 0], { destroyed: t >= 5 })
    }, torpedoes: t < 9 ? [{ id: 'torp_1', pos: [1000 * t, 9000, 0], vel: [1000, 0, 0], source: 'a', target: 'b', dv_remaining: 5 }] : [] });
  }
  const ev = (timestamp, event_type, ship_id, data) => ({ timestamp, event_type, ship_id, data: { target_id: null, ...data } });
  const events = !withEvents ? [] : [
    ev(2, 'mount_damaged', 'a', { kind: 'gun', slot: 'weapon_1', index: 1, weapon_name: 'Coilgun Battery Mk3', health_pct: 40, destroyed: false, cooldown_factor: 2, cause: 'surface_hit' }),
    ev(3, 'mount_damaged', 'a', { kind: 'gun', slot: 'weapon_1', index: 1, weapon_name: 'Coilgun Battery Mk3', health_pct: 0, destroyed: true, cooldown_factor: null, cause: 'surface_hit' }),
    ev(3, 'mount_damaged', 'a', { kind: 'pd', slot: 'pd_1', index: 1, weapon_name: 'PD Laser', health_pct: 0, destroyed: true, cause: 'surface_hit' }),
    ev(3, 'mount_damaged', 'a', { kind: 'launcher', slot: 'weapon_0', index: 0, weapon_name: 'Torpedo Launcher', health_pct: 62, destroyed: false, cause: 'surface_hit' }),
    ev(4, 'mount_damaged', 'a', { kind: 'gun', slot: 'weapon_0', index: 0, weapon_name: 'Spinal Coiler Mk3', health_pct: 0, destroyed: true, cause: 'surface_hit' }),
    ev(4, 'mount_damaged', 'a', { kind: 'engine', slot: 'Main Engine Assembly', index: 0, health_pct: 61.8, destroyed: false, thrust_fraction: 0.618, cause: 'surface_hit' }),
    ev(6, 'mount_damaged', 'a', { kind: 'engine', slot: 'Main Engine Assembly', index: 0, health_pct: 0, destroyed: true, thrust_fraction: 0, cause: 'surface_hit' }),
    ev(4, 'magazine_cookoff', 'b', { module_name: 'Main Magazine', torpedoes_lost: 0, gun_rounds_lost: 1125, secondary_gj: 1.5, modules_damaged: [], modules_destroyed: [] }),
    ev(5, 'blast_damage', 'b', { source_ship_id: 'dead', distance_km: 5, intensity: 0.5, hit_location: 'lateral', armor_ablation_cm: 7.5, armor_remaining_cm: 20, heat_gj: 60, heat: 0.4,
      radiators: [{ radiator_position: 'tail_port', health_pct: 42.5, destroyed: false, capacity_lost_kw: 1 }] }),
    ev(5, 'reactor_blast', 'dead', { ship_type: 'destroyer', cause: 'reactor_destroyed', position: [4000, 8000, 0], radius_km: 13, torpedo_kill_radius_km: 3.25,
      torpedo_blind_radius_km: 7.8, yield_gj: 426, ships_hit: ['b'], torpedoes_destroyed: [], torpedoes_blinded: ['torp_1'] })
  ];
  return { events, sim_trace };
}

test('recordings without the pass-5 events build the same empty state', () => {
  const tl = buildDamageTimeline(recording(false));
  assert.deepEqual(tl.blasts, []);
  for (const sd of tl.ships.values()) {
    assert.equal(engineStateAt(sd, 100), null);
    assert.deepEqual(sd.wrecks, []);
    assert.deepEqual(sd.blastHits, []);
    assert.equal(sd.marks.length, 0);
  }
  assert.equal(engineStateAt(undefined, 1), null);
  assert.deepEqual(blastsAt(undefined, 5, 5), { lo: 0, hi: 0 });
});

test('mount_damaged: the named fitting is hit, then dead for the rest of the battle', () => {
  const tl = buildDamageTimeline(recording());
  const a = tl.ships.get('a');
  assert.deepEqual(a.wrecks.map(w => [w.t, w.kind, w.slot, w.hit, w.destroyed]), [
    [2, 'gun', 'weapon_1', true, false],
    [3, 'gun', 'weapon_1', false, true],
    [3, 'pd', 'pd_1', false, true],
    [3, 'launcher', 'weapon_0', true, false],
    [4, 'spinal', 'weapon_0', false, true]
  ]);
  assert.ok(a.wrecks.every(w => w.mount));
  // state at t is a filter on time: a backward seek simply yields less
  assert.equal(wrecksAt(a, 1.9).length, 0);
  assert.equal(wrecksAt(a, 3).length, 4);
  assert.equal(wrecksAt(a, 2.5).length, 1);
  // the drive is not a wreck: it has its own state
  assert.equal(engineStateAt(a, 3.9), null);
  assert.deepEqual(engineStateAt(a, 4), { t: 4, health: 61.8, thrust: 0.618, destroyed: false });
  assert.equal(engineStateAt(a, 5.99).health, 61.8);
  assert.equal(engineStateAt(a, 6).destroyed, true);
});

test('reactor_blast: reach, victims and the wash on the facing side', () => {
  const tl = buildDamageTimeline(recording());
  assert.equal(tl.blasts.length, 1);
  const b = tl.blasts[0];
  assert.deepEqual([b.t, b.shipId, b.radius, b.kill, b.blind], [5, 'dead', 13, 3.25, 7.8]);
  assert.deepEqual(b.pos, [4000, 8000, 0]);
  assert.deepEqual(b.blinded, ['torp_1']);
  assert.deepEqual(blastsAt(tl.blasts, 4.9, SHELL_LIFE_S), { lo: 0, hi: 0 });
  assert.deepEqual(blastsAt(tl.blasts, 5, SHELL_LIFE_S), { lo: 0, hi: 1 });
  assert.deepEqual(blastsAt(tl.blasts, 5 + SHELL_LIFE_S, SHELL_LIFE_S), { lo: 1, hi: 1 });

  const sd = tl.ships.get('b');
  assert.equal(sd.blastHits.length, 1);
  assert.equal(sd.blastHits[0].intensity, 0.5);
  const scorch = sd.marks.filter(m => m.kind === 'scorch');
  assert.equal(scorch.length, blastScorchCount(0.5));
  assert.ok(blastScorchCount(1) > blastScorchCount(0.05));
  for (const m of scorch) {
    assert.ok(m.t > 5 && m.t < 5 + BLAST_EXPAND_S + BLAST_WASH_S);
    assert.equal(m.section, 'lateral');
    assert.equal(m.blast, null);
    assert.equal(m.strike, null);
    assert.equal(m.debris, 0, 'a wash throws no spall');
    assert.equal(m.vent, null);
    // the blast is on b's +Y side (hull frame: fwd = +X world): the marks
    // sit on the half of the hull that faces it
    assert.ok(m.approach[0] ** 2 + m.approach[1] ** 2 > 0.9, 'approach is square to the long axis');
  }
  // scorch marks are heat, not hits: they do not buckle the radiators...
  const before = damageStateAt(tl, 'b', 4.9, {}), after = damageStateAt(tl, 'b', 6, {});
  assert.equal(after.lateralHits, before.lateralHits);
  // ...but panels the event lists as caught extended are wrecked
  assert.deepEqual(sd.wrecks.map(w => [w.kind, w.side, w.destroyed]), [['radiator', 'port', false]]);
  // ...when the front, which takes BLAST_EXPAND_S to reach the radius, gets to them
  assert.ok(sd.wrecks[0].t > 5 && sd.wrecks[0].t < 5 + BLAST_EXPAND_S);
  assert.ok(Math.abs(blastFront(blastArrival(5, 13), 13) - 5) < 1e-9, 'arrival inverts the front');
  assert.equal(sd.blastHits[0].t, sd.wrecks[0].t);
  // before the blast nothing is there
  assert.equal(marksUpTo(sd.marks.filter(m => m.kind === 'scorch'), 4.99), 0);
});

test('magazine_cookoff: a secondary explosion venting fire from the flank', () => {
  const tl = buildDamageTimeline(recording());
  const sd = tl.ships.get('b');
  assert.deepEqual(sd.cookoffs.map(c => [c.t, c.gj]), [[4, 1.5]]);
  const m = sd.marks.find(x => x.kind === 'cookoff');
  assert.ok(m && m.t > 4 && m.t < 4.5);
  assert.equal(m.blast, 'cookoff');
  assert.equal(m.vent, 'fire');
  assert.ok(m.penetrated && m.debris >= 26, 'burning debris');
  const K = blastKinds().cookoff;
  assert.ok(K && K.fireR > 0 && K.shellR === 0 && K.cone > 0 && K.duration <= 6);
  assert.ok(K.streakW <= 0.024, 'streaks, not bars');
});

test('blast shell reaches exactly the recorded radius and is pure in time', () => {
  const R = 12.99;
  assert.equal(blastShellAt(-0.1, R).level, 0);
  assert.equal(blastShellAt(SHELL_LIFE_S, R).level, 0);
  assert.equal(blastShellAt(1, 0).level, 0);
  let prev = 0;
  for (let age = 0; age < SHELL_LIFE_S; age += 0.05) {
    const s = blastShellAt(age, R);
    assert.ok(s.r >= prev - 1e-12 && s.r <= R + 1e-9, 'monotonic, never past the radius');
    assert.ok(s.level >= 0 && s.level <= 1 && s.cool >= 0 && s.cool <= 1);
    prev = s.r;
  }
  assert.ok(Math.abs(blastShellAt(SHELL_EXPAND_S, R).r - R) < 1e-9);
  assert.ok(Math.abs(blastShellAt(SHELL_LIFE_S - 0.5, R).r - R) < 1e-9, 'the bubble hangs at the radius');
  assert.ok(blastShellAt(0.5, R).level > blastShellAt(3, R).level);
  assert.deepEqual(blastShellAt(0.7, R), blastShellAt(0.7, R));
  // the torpedo-kill sphere is out and gone well inside the outer shell's life
  assert.ok(Math.abs(blastShellAt(KILL_EXPAND_S, 3.25, {}, KILL_EXPAND_S, KILL_LIFE_S).r - 3.25) < 1e-9);
  assert.equal(blastShellAt(KILL_LIFE_S, 3.25, {}, KILL_EXPAND_S, KILL_LIFE_S).level, 0);
});

test('a damaged drive loses bells in a fixed order and the rest burn rough', () => {
  for (const n of [2, 3, 4, 6, 8]) {
    let prevDead = new Set();
    for (const health of [100, 80, 61.8, 53, 30, 10, 0]) {
      const dead = new Set();
      for (let i = 0; i < n; i++) {
        const nz = damagedNozzle(health, i, n, 4242);
        assert.deepEqual(nz, damagedNozzle(health, i, n, 4242), 'pure');
        if (nz.level === 0) dead.add(i);
        else if (health < 100) assert.ok(nz.level >= 1 && nz.level <= 1.35 && nz.sputter > 0.2 && nz.sputter <= 0.85);
        else assert.deepEqual(nz, { level: 1, sputter: 0 });
      }
      for (const i of prevDead) assert.ok(dead.has(i), 'a dark bell stays dark');
      if (health > 0) assert.ok(dead.size < n, 'a drive that still thrusts keeps a lit bell');
      else assert.equal(dead.size, n, 'a destroyed drive is dark');
      prevDead = dead;
    }
  }
  // the demo's two hits on a 4-bell torpedo cruiser: one bell out, then two
  const out = (h) => [0, 1, 2, 3].filter(i => damagedNozzle(h, i, 4, 99).level === 0).length;
  assert.deepEqual([out(61.8), out(53)], [1, 2]);
});

test('a dead mount spits sparks on battle time, thinning out', () => {
  let early = 0, late = 0, last = -1;
  for (let k = 0; k < 40; k++) {
    const t = mountVentTime(1234, k);
    assert.equal(t, mountVentTime(1234, k));
    if (t < 0) continue;
    assert.ok(t > last && t < MOUNT_VENT_S + 1);
    last = t;
    if (t < MOUNT_VENT_S / 2) early++; else late++;
  }
  assert.ok(early > late && early >= 6);
  assert.equal(mountVentTime(1234, -1), -1);
  assert.equal(mountVentTime(1234, 1000), -1);
});

test('a torpedo blinded by a reactor blast loses its seeker at the blast', () => {
  const rec = recording();
  const loader = new BattleLoader();
  loader.simTrace = rec.sim_trace;
  loader.events = rec.events;
  const tl = new WeaponTimeline(loader);
  const tr = tl.torpedoes.get('torp_1');
  assert.equal(tr.disabledAt, 5.3);
  assert.equal(tl.moments.filter(m => m.kind === 'seeker' && m.track === tr && m.time === 5.3).length, 1);
  // without the event the round keeps its seeker
  loader.events = [];
  assert.equal(new WeaponTimeline(loader).torpedoes.get('torp_1').disabledAt, Infinity);
});

test('combat log names the new damage: mounts, drive, cook-off, blast victims', () => {
  const ships = {};
  const row = (event_type, data, ship_id = 'alpha_1') => formatLogEvent({ event_type, ship_id, data }, ships);
  const gun = row('mount_damaged', { kind: 'gun', slot: 'weapon_0', weapon_name: 'Coilgun Battery Mk3', health_pct: 4.6, destroyed: false, cause: 'surface_hit' });
  assert.equal(gun.cls, 'hit');
  assert.match(gun.text, /^MOUNT HIT: .* Coilgun Battery Mk3 at 5%$/);
  const pd = row('mount_damaged', { kind: 'pd', slot: 'pd_1', health_pct: 0, destroyed: true, cause: 'reactor_blast' });
  assert.equal(pd.cls, 'destroyed');
  assert.match(pd.text, /^MOUNT DESTROYED: .* lost PD turret pd_1 · reactor blast$/);
  assert.match(row('mount_damaged', { kind: 'launcher', destroyed: true }).text, /lost torpedo tube$/);
  assert.match(row('mount_damaged', { kind: 'engine', destroyed: false, thrust_fraction: 0.618 }).text, /^DRIVE HIT: .* down to 62% thrust$/);
  assert.match(row('mount_damaged', { kind: 'engine', destroyed: true, thrust_fraction: 0 }).text, /^DRIVE OUT: .* is adrift$/);
  assert.match(row('magazine_cookoff', { module_name: 'Main Magazine', torpedoes_lost: 0, gun_rounds_lost: 1125 }).text, /^COOK-OFF: .* Main Magazine burns \(1125 rounds lost\)$/);
  assert.match(row('magazine_cookoff', { module_name: 'Main Magazine', torpedoes_lost: 24 }).text, /24 torpedoes lost/);
  const blast = row('blast_damage', { source_ship_id: 'beta_5', distance_km: 3.56, armor_ablation_cm: 7.9, hit_location: 'lateral' });
  assert.equal(blast.cat, 'kills');
  assert.match(blast.text, /^BLAST: .* caught 3\.6 km from .* reactor \(-7\.9 cm lateral armour\)$/);
  assert.match(row('blast_damage', { source_ship_id: 'beta_5', distance_km: 3, fuse_cut_s: 11.8 }).text, /fuse -11\.8 s/);
});

// ---------------------------------------------------------------------------
// Round 3: impacts on fittings, exit jets, remnant, drive transients
// ---------------------------------------------------------------------------
import { blastRemnantAt, REMNANT_LIFE_S } from '../src/render/BlastShell.js';
import { plumeTransient, TRANSIENT_RATE } from '../src/render/Torch.js';
import { Interpolator } from '../src/Interpolator.js';
import { EXIT_MAX_BEAMS } from '../src/render/DamageFX.js';

test('a round that strikes a fitting is marked for that mount; the blast coasts with the wreck', () => {
  const rec = recording();
  // the gun-mount hit at t=2 was made by a slug that also has a hit event
  rec.events.unshift({ timestamp: 2, event_type: 'hit', ship_id: 'a', data: { projectile_id: 'slug_9', hit_location: 'lateral', kinetic_energy_gj: 2.7, penetrated: false, shooter_id: 'b' } });
  rec.events.find(e => e.event_type === 'mount_damaged' && e.timestamp === 2).data.ordnance_id = 'slug_9';
  const tl = buildDamageTimeline(rec);
  const m = tl.byOrdnance.get('slug_9');
  assert.equal(m.fitting, 'gun');
  assert.equal(m.strike, 'impact');
  const w = tl.ships.get('a').wrecks.find(x => x.ordnanceId === 'slug_9');
  assert.equal(w.t, m.t, 'the mount sparks when the round is seen to land');
  // an ordinary hit carries no fitting
  rec.events.push({ timestamp: 7, event_type: 'hit', ship_id: 'a', data: { projectile_id: 'slug_10', hit_location: 'lateral', kinetic_energy_gj: 2.7, penetrated: true } });
  const tl2 = buildDamageTimeline(rec);
  assert.equal(tl2.byOrdnance.get('slug_10').fitting, undefined);
  assert.equal(tl2.byOrdnance.get('slug_10').strike, 'pierce');
  // the blast is centred on the wreck's trace position and carries its velocity
  const moving = recording();
  for (const f of moving.sim_trace) { f.ships.dead.vel = [0, 2000, 0]; f.ships.dead.pos = [4000, 8000 + 2000 * f.t, 0]; }
  const b = buildDamageTimeline(moving).blasts[0];
  assert.deepEqual(b.pos, [4000, 8000 + 2000 * b.t, 0]);
  assert.deepEqual(b.vel, [0, 2000, 0]);
});

test('impact presets: stopped, penetrating and torpedo strikes are built differently', () => {
  const K = blastKinds();
  // armour-stopped slug: flash + plasma ring on the plating, no fire
  assert.ok(K.impact.fireR === 0 && K.impact.ringR > 0 && K.impact.core && K.impact.crater);
  // penetrating slug: a tighter, faster jet and a fire inside the hole
  assert.ok(K.pierce.fireR > 0 && K.pierce.cone > K.impact.cone && K.pierce.streakSpeed[1] > K.impact.streakSpeed[1]);
  // torpedo strikes are heavier than slug hits
  assert.ok(K.kineticStrike.flash > K.impact.flash && K.kineticStrike.ringR > K.impact.ringR && K.kineticStrike.fireR === 0);
  assert.ok(K.breach.fireR > 0 && K.breach.flash >= K.kineticStrike.flash);
  // the exit jet is the tightest, fastest cone of all and has no shell
  assert.ok(K.exitJet.cone >= 0.95 && K.exitJet.shellR === 0 && K.exitJet.streakSpeed[1] > K.breach.streakSpeed[1]);
  for (const k of ['impact', 'pierce', 'kineticStrike', 'breach', 'exitJet']) {
    assert.ok(K[k].duration <= 6 && K[k].streakW <= 0.024 && K[k].flashFade <= 0.13, `${k}: a hard, short flash`);
  }
  // the reactor kill opens with a hard white core that is gone in a blink
  assert.ok(K.reactor.core.level >= 8 && K.reactor.core.life < 0.25);
  assert.ok(EXIT_MAX_BEAMS > 1 && EXIT_MAX_BEAMS < 3);
});

test('remnant glow and drive transients are pure functions of battle time / the recording', () => {
  assert.equal(blastRemnantAt(-1, 13).level, 0);
  assert.equal(blastRemnantAt(REMNANT_LIFE_S, 13).level, 0);
  assert.equal(blastRemnantAt(0.5, 13).level, 0, 'nothing while the fireball is young');
  const mid = blastRemnantAt(8, 13), late = blastRemnantAt(26, 13);
  assert.ok(mid.level > late.level && late.level > 0 && mid.r < 13 * 0.5 && late.cool > mid.cool);
  assert.deepEqual(blastRemnantAt(8, 13), mid);

  assert.deepEqual(plumeTransient(0), { surge: 0, fade: 0 });
  assert.deepEqual(plumeTransient(undefined), { surge: 0, fade: 0 });
  assert.deepEqual(plumeTransient(TRANSIENT_RATE * 2), { surge: 1, fade: 0 });
  assert.deepEqual(plumeTransient(-TRANSIENT_RATE / 2), { surge: 0, fade: 0.5 });
  // the slope comes from two neighbouring trace frames, the same at any playhead in between
  const f = (thrust) => ({ pos: [0, 0, 0], vel: [0, 0, 0], fwd: [0, 0, 1], thrust });
  const at = (a) => Interpolator.interpolateShips({ t: 10, ships: { s: f(0.2) } }, { t: 11, ships: { s: f(1.0) } }, a).s;
  assert.ok(Math.abs(at(0.1).thrustRate - 0.8) < 1e-12 && Math.abs(at(0.9).thrustRate - 0.8) < 1e-12);
  assert.ok(Math.abs(at(0.5).thrust - 0.6) < 1e-12);
});

// ---------------------------------------------------------------------------
// Drive plume wash (plume lane)
// ---------------------------------------------------------------------------
import { washAt, washHeat, WASH_COOL_S } from '../src/DamageTimeline.js';

test('plume wash: the facing heats on entry, follows the recorded rate, cools after the exit', () => {
  const rec = recording(false);
  const wash = (timestamp, phase, rate, extra = {}) => ({ timestamp, event_type: 'plume_wash', ship_id: 'a',
    data: { source_ship_id: 'b', phase, heat_rate_gj_s: rate, peak_heat_rate_gj_s: extra.peak ?? rate, hit_location: 'nose', swept: !!extra.swept, target_id: null } });
  rec.events = [wash(2, 'enter', 12.8), wash(7, 'sustain', 16), wash(9, 'exit', 0, { peak: 16 }),
    wash(10.5, 'enter', 0.08, { swept: false }), wash(11.5, 'exit', 0, { peak: 0.19, swept: true })];
  const tl = buildDamageTimeline(rec);
  const sd = tl.ships.get('a');
  assert.deepEqual(sd.washes.map(w => [w.source, w.t0, w.t1, w.section, w.swept, w.peak]), [['b', 2, 9, 'nose', false, 16], ['b', 10.5, 11.5, 'nose', true, 0.19]]);
  // nothing before the jet arrives, and nothing on a ship that is never washed
  assert.equal(washAt(sd, 1.9).heat, 0);
  assert.equal(washAt(tl.ships.get('b'), 5).heat, 0);
  assert.equal(washAt(undefined, 5).heat, 0);
  // builds over the first second or two, then tracks the rate between samples
  const h0 = washAt(sd, 2.05).heat, h1 = washAt(sd, 3.5).heat, h2 = washAt(sd, 7).heat;
  assert.ok(h0 > 0 && h0 < h1 && h1 < h2, `${h0} ${h1} ${h2}`);
  assert.ok(Math.abs(h2 - washHeat(16)) < 0.01 && washAt(sd, 7).rate === 16 && washAt(sd, 7).source === 'b' && washAt(sd, 7).jet === 'b' && washAt(sd, 7).section === 'nose');
  assert.ok(Math.abs(washAt(sd, 4.5).rate - 14.4) < 1e-9, 'rate eased between the 5 s samples');
  // after the exit: no wash rate (no impingement), the plating cools over seconds
  const c1 = washAt(sd, 9.5), c2 = washAt(sd, 10.4);
  assert.equal(c1.rate, 0);
  assert.equal(c1.jet, null);
  assert.ok(c1.heat < h2 && c2.heat < c1.heat && c2.heat > 0);
  assert.ok(Math.abs(c1.heat - washHeat(16) * Math.exp(-0.5 / WASH_COOL_S)) < 0.02);
  assert.equal(c1.t, 9, 'the glow stays where the jet last was');
  // a short swept crossing flares at its peak rate, then cools away
  const sweep = washAt(sd, 11.2);
  assert.ok(sweep.rate >= 0.19 && sweep.jet === 'b', 'the jet is on the hull again');
  assert.ok(sweep.heat > 0);
  assert.equal(washAt(sd, 40).heat, 0);
  // pure: same answer whenever asked, in any order
  assert.deepEqual(washAt(sd, 7), washAt(sd, 7));
  // heat scale: a faint wash is dull red, the demo's 16 GJ/s yellow-white, point blank white
  assert.ok(washHeat(0) === 0 && washHeat(0.2) < 0.4 && washHeat(16) === 1 && washHeat(69) > 1.4 && washHeat(1e6) <= 1.6);
  // fittings burned by the wash use the ordinary dead-mount path
  rec.events.push({ timestamp: 5, event_type: 'mount_damaged', ship_id: 'a', data: { kind: 'pd', slot: 'pd_0', index: 0, health_pct: 0, destroyed: true, cause: 'plume_wash', hit_location: 'nose', source_ship_id: 'b' } });
  const w = buildDamageTimeline(rec).ships.get('a').wrecks;
  assert.deepEqual(w.map(x => [x.t, x.kind, x.slot, x.destroyed, x.cause]), [[5, 'pd', 'pd_0', true, 'plume_wash']]);
});
