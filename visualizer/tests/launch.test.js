import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import * as THREE from 'three';

// Hull materials bake small canvas textures: a canvas that accepts every 2D call is enough
const ctx2d = new Proxy({}, {
  get: (t, prop) => (prop === 'createImageData' || prop === 'getImageData'
    ? (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h })
    : prop === 'measureText' ? () => ({ width: 10 }) : prop === 'canvas' ? fakeCanvas() : () => ctx2d),
  set: () => true
});
function fakeCanvas() { return { width: 4, height: 4, style: {}, getContext: () => ctx2d }; }
globalThis.document ??= { createElement: () => fakeCanvas() };

const { assignRounds, assignLaunches, bankTimes, launcherBank, shipEventIndex, updateMagazine } = await import('../src/render/HullRigs.js');
const { WeaponTimeline, makeSample, sampleTrack, bodyToWorld, torpedoPower } = await import('../src/render/WeaponTimeline.js');
const { attitudeAt, pulsesAt, makePulses } = await import('../src/render/TorpedoRcs.js');
const { drawnOffset, launchWeight, unitScale, shipHermite, stretchFromShip, UNIT_LEN_KM, BODY_LEN_KIT, HANDOVER_S } = await import('../src/render/LaunchSequence.js');
const { BattleLoader } = await import('../src/BattleLoader.js');
const { FittingFold } = await import('../src/FittingsIndex.js');
const { buildHull, hullSize, buildShipMaterials } = await import('../src/render/Hulls.js');
const { SceneManager } = await import('../src/SceneManager.js');
const { trueK } = await import('../src/render/Scale.js');
const { buildHighlights, buildShotList, SALVO_AFTER_LIGHTOFF_S } = await import('../src/Director.js');
const HULL_MOUNTS = (await import('../src/render/hullMountsData.js')).default;

const DEMO = new URL('../../data/recordings/p5_launch_sequence_demo.json.gz', import.meta.url);
const demo = () => JSON.parse(gunzipSync(readFileSync(DEMO)).toString('utf8'));
function demoLoader() {
  const l = new BattleLoader();
  l.parse(demo());
  return l;
}
const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} ${a} != ${b}`);
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// ------------------------------------------------------------------ the rack

test('launcher banks follow data/hull_mounts.json, and every magazine round sits where the engine ejects it from', () => {
  assert.equal(launcherBank('corvette', 'weapon_0'), 0);
  for (let i = 0; i < 4; i++) assert.equal(launcherBank('cruiser_torpedo', `weapon_${i}`), i);
  assert.equal(launcherBank('corvette', 'weapon_7'), -1);
  assert.equal(launcherBank('mystery', 'weapon_0'), -1);
  assert.equal(launcherBank('corvette', null), -1);
  // the drawn rack round (bank, call order) is the engine's rounds[round].pos:
  // kit Z/X/Y = body x/y/z, at true scale
  for (const type of ['corvette', 'cruiser_torpedo']) {
    const g = new THREE.Group(), size = hullSize(type);
    buildHull(type, g, size, buildShipMaterials('alpha'), SceneManager.prototype.getEngineConfig.call(null, type, size), 3);
    const mag = g.userData.magazine;
    const launchers = HULL_MOUNTS.hulls[type].mounts.filter(m => m.kind === 'launcher');
    assert.equal(mag.banks, launchers.length, type);
    const k = trueK(type) * 1000;
    mag.entries.forEach((e, i) => {
      const r = launchers[e.bank].rounds[mag.order[i]];
      assert.ok(r, `${type} bank ${e.bank} round ${mag.order[i]}`);
      const body = [e.pos.z * k, e.pos.x * k, e.pos.y * k];
      assert.ok(dist3(body, r.pos) < 0.6, `${type} bank ${e.bank} round ${mag.order[i]}: ${body} vs ${r.pos}`);
    });
  }
});

test('the rack empties the exact round the engine launched, and old recordings keep the count dealing', () => {
  // old recordings: identical to the pass-4 round robin
  const old = [{ t: 1 }, { t: 2 }, { t: 3 }, { t: 4 }, { t: 5 }];
  assert.deepEqual(bankTimes(assignRounds(old, 2)), assignLaunches(old, 2));
  assert.deepEqual(assignRounds(old, 2).map(b => [...b]), assignLaunches(old, 2));
  // mount + round: exactly that round, whatever the order
  const bank = (s) => ({ weapon_0: 0, weapon_1: 1 })[s] ?? -1;
  const exact = [{ t: 3, mount: 'weapon_1', round: 4 }, { t: 7, mount: 'weapon_0', round: 2 }, { t: 7, mount: 'weapon_0', round: 0 }];
  const r = assignRounds(exact, 2, bank);
  assert.equal(r[1][4], 3);
  assert.equal(r[0][2], 7);
  assert.equal(r[0][0], 7);
  assert.equal(r[0][1], undefined, 'untouched rounds stay racked');
  // mixed (a launch without the fields): dealt to the lowest round still racked
  const mixed = assignRounds([...exact, { t: 9 }], 2, bank);
  assert.equal(mixed[0][1], 9);
  // a wrecked tube loses the rounds still on it; a cook-off the highest-numbered on the fullest banks
  const lost = assignRounds([{ t: 1, mount: 'weapon_0', round: 0 }, { t: 5 }], 2, bank,
    [{ t: 3, mount: 'weapon_0', all: true, count: 11 }, { t: 4, mount: null, all: false, count: 2 }]);
  assert.equal(lost[0][0], 1);
  assert.ok([...Array(12).keys()].slice(1).every(r => lost.lost[0][r] === 3), 'the rest of bank 0 lost with its tube');
  assert.equal(lost.lost[1][11], 4);
  assert.equal(lost.lost[1][10], 4);
  assert.equal(lost[1][0], 5, 'a later launch takes a round still racked');
  // an unknown mount is dealt like an old launch
  assert.equal(assignRounds([{ t: 1, mount: 'weapon_9', round: 3 }], 2, bank)[0][0], 1);
});

test('demo recording: rack rounds leave exactly as launched and always agree with the HUD magazine count', () => {
  const rec = demo();
  const loader = demoLoader();
  const idx = shipEventIndex(loader.events);
  const fi = new FittingFold();
  fi.setLayouts(loader.sourceData, loader.ships);
  fi.add(loader.events, loader.events.map((e, i) => e.event_id || `e${i}`), 0, loader.ships);
  let checks = 0;
  for (const [ship, type] of [['alpha', 'cruiser_torpedo'], ['beta', 'corvette']]) {
    const launches = idx.launches.get(ship);
    assert.ok(launches.length >= 4 && launches.every(l => l.mount && Number.isInteger(l.round)), ship);
    const tubes = HULL_MOUNTS.hulls[type].mounts.filter(m => m.kind === 'launcher');
    const rounds = assignRounds(launches, tubes.length, s => launcherBank(type, s), idx.losses.get(ship));
    const total = tubes.reduce((n, m) => n + m.rounds.length, 0);
    for (const l of launches) assert.equal(rounds[launcherBank(type, l.mount)][l.round], l.t, `${ship} ${l.mount}#${l.round}`);
    const states = fi.states[ship] || [];
    assert.ok(states.length > 1, `${ship}: the HUD follows the magazine`);
    for (let t = 0; t <= rec.duration_s; t += 0.5) {
      let gone = 0;
      tubes.forEach((m, b) => {
        for (let r = 0; r < m.rounds.length; r++) if (rounds[b][r] <= t || rounds.lost[b][r] <= t) gone++;
      });
      let hud = null;
      for (const s of states) if (s.t <= t) hud = s.rounds;
      checks++;
      if (hud !== null) assert.equal(total - gone, hud, `${ship} at ${t}: rack ${total - gone} vs HUD ${hud}`);
    }
  }
  assert.ok(checks > 100);
});

// ------------------------------------------------------------ the unit's pose

test('a sequence round starts on its rack round with the ship\'s motion and spin, then follows the trace', () => {
  const loader = demoLoader();
  const tl = new WeaponTimeline(loader);
  let checked = 0;
  for (const tr of tl.torpedoes.values()) {
    if (!tr.launch) continue;
    const s0 = tr.samples[0];
    assert.ok(s0.synthetic && s0.t === tr.start && s0.state.ph === 'eject' && s0.state.thr === 0);
    const ship = makeSample();
    sampleTrack(tl.ships.get(tr.launch.ship), tr.start, ship);
    const rack = bodyToWorld(ship.fwd, ship.up, tr.launch.ejectPos);
    assert.ok(dist3(s0.state.pos, [ship.pos[0] + rack[0], ship.pos[1] + rack[1], ship.pos[2] + rack[2]]) < 1e-6, 'on the rack');
    // one tick of straight drift lands on the trace's first frame (the engine's
    // eject; a launch before the trace's first frame has no ship state to start from)
    const f1 = tr.samples[1];
    if (tr.start < loader.simTrace[0].t) { checked++; continue; }
    const dt = f1.t - s0.t;
    const drift = s0.state.pos.map((p, k) => p + s0.state.vel[k] * dt);
    assert.ok(dist3(drift, f1.state.pos) < 1.0, `${tr.id}: drift lands ${dist3(drift, f1.state.pos).toFixed(2)} m off the first frame`);
    // its nose at launch is the round's nose at rest turned with the hull
    const n0 = bodyToWorld(ship.fwd, ship.up, tr.launch.nose);
    const a = attitudeAt(tr.thrusters, tr.start + 1e-6, { x: 0, y: 0, z: 0 });
    assert.ok(Math.hypot(a.x - n0[0], a.y - n0[1], a.z - n0[2]) < 0.05, `${tr.id}: nose at launch`);
    // at each recorded sequence frame the nose is the recorded one (keys at tick starts)
    for (let i = 1; i < tr.samples.length; i++) {
      const st = tr.samples[i].state;
      if (!st.ph || !tr.samples[i + 1]?.state?.ph) break;   // (the ignition tick keys on its thrust axis)
      const b = attitudeAt(tr.thrusters, tr.samples[i].t, { x: 0, y: 0, z: 0 });
      assert.ok(Math.hypot(b.x - st.nose[0], b.y - st.nose[1], b.z - st.nose[2]) < 0.02, `${tr.id} nose at ${tr.samples[i].t}`);
    }
    // the torch is cold until the recorded ignition and lit right after it
    assert.ok(Number.isFinite(tr.ignitionAt), tr.id);
    const powerAt = (t) => {
      const out = makeSample();
      tl.sample(tr, t, out);
      return torpedoPower(out.state, out.next, out.dt, Infinity, 0, false);
    };
    for (let t = tr.start; t < tr.ignitionAt; t += 0.25) assert.equal(powerAt(t), 0, `${tr.id} cold at ${t}`);
    assert.ok(powerAt(tr.ignitionAt + 0.1) > 0, `${tr.id} lit after ignition`);
    // no attitude couple fires during the eject second (it only drifts)
    const list = makePulses();
    for (let t = tr.start + 0.05; t < tr.start + 0.95; t += 0.1) {
      pulsesAt(tr.thrusters, t, 0, list, null);
      for (let p = 0; p < list.count; p++) assert.ok(Math.hypot(list[p].push.x, list[p].push.y, list[p].push.z) === 0, `${tr.id} couple at ${t}`);
    }
    checked++;
  }
  assert.ok(checked >= 10, `${checked} sequence rounds`);
});

test('the separation RCS puffs come from the recorded rcs vector', () => {
  const tl = new WeaponTimeline(demoLoader());
  const tr = tl.torpedoes.get('torp_alpha_0001');
  const i = tr.samples.findIndex(s => s.state.rcs);
  assert.ok(i > 0);
  const r = tr.samples[i].state.rcs, t = tr.samples[i].t - 0.5;   // inside the tick that ended on that frame
  const list = makePulses();
  pulsesAt(tr.thrusters, t, 0, list, null);
  const trims = [...Array(list.count).keys()].map(k => list[k]).filter(p => Math.hypot(p.thrust.x, p.thrust.y, p.thrust.z) > 0.5);
  assert.ok(trims.length > 0);
  const m = Math.hypot(...r);
  for (const p of trims) near(p.thrust.x * r[0] + p.thrust.y * r[1] + p.thrust.z * r[2], m, 1e-6, 'along the recorded push');
});

test('drawn launch: exact at true scale, leaves the drawn rack when magnified, monotonic and continuous in the scale', () => {
  const rack = 0.0145;
  for (const r of [0.001, 0.0145, 0.05, 0.3, 5]) assert.equal(drawnOffset(r, rack, 1), r);
  for (const m of [1.5, 6, 60, 94]) {
    near(drawnOffset(rack, rack, m), m * rack, 1e-12, 'on the drawn rack');
    let prev = 0;
    for (let r = 0.0001; r < 50; r *= 1.07) {
      const d = drawnOffset(r, rack, m);
      assert.ok(d > prev && d >= r, `monotonic at ${r} (m ${m})`);
      prev = d;
    }
  }
  // continuous in the magnification (a camera crossing regimes moves nothing by a jump)
  for (const r of [0.01, 0.1, 1]) assert.ok(Math.abs(drawnOffset(r, rack, 1 + 1e-6) - drawnOffset(r, rack, 1)) < 1e-6);
  // the stretch and the launch size hold through the sequence and hand over after ignition
  assert.equal(launchWeight(3, 0, 4), 1);
  assert.equal(launchWeight(4 + HANDOVER_S[0], 0, 4), 1);
  assert.equal(launchWeight(4 + HANDOVER_S[1], 0, 4), 0);
  assert.equal(launchWeight(36, 0, undefined), 0, 'no ignition recorded: the engine hard cap');
  let prev = 1;
  for (let t = 0; t < 30; t += 0.1) { const w = launchWeight(t, 0, 4); assert.ok(w <= prev + 1e-12); prev = w; }
  // the unit is the rack round's size at true scale (no pop when it leaves), magnified with its hull
  near(unitScale(1) * BODY_LEN_KIT, UNIT_LEN_KM, 1e-12);
  near(unitScale(8), 8 * unitScale(1), 1e-12);
  assert.ok(UNIT_LEN_KM > 0.01 && UNIT_LEN_KM < 0.02, 'the 13.8 m rack round');
  // stretch about the real ship point: m = 1 leaves the point alone
  const p = [3, 4, 0], s = [0, 0, 0];
  assert.deepEqual(stretchFromShip([...p], s, rack, 1, 1), p);
  const q = stretchFromShip([0.01, 0, 0], s, 0.01, 10, 1);
  near(q[0], 0.1, 1e-12);
});

test('a round\'s drawn track is a pure function of battle time (seek into mid-sequence = play into it)', () => {
  const tl = new WeaponTimeline(demoLoader());
  const tr = tl.torpedoes.get('torp_beta_0003');   // the 8 s timeout, launched from a turning ship
  const ship = tl.ships.get(tr.launch.ship);
  const at = (t) => {
    const out = makeSample();
    tl.sample(tr, t, out);
    const sp = [0, 0, 0];
    shipHermite(ship, t, sp);
    const p = stretchFromShip(out.pos.map(x => x / 1000), sp.map(x => x / 1000), 0.0145, 12, launchWeight(t, tr.start, tr.ignitionAt));
    const n = attitudeAt(tr.thrusters, t, { x: 0, y: 0, z: 0 });
    return JSON.stringify([p, n]);
  };
  const probes = [12.4, 13.3, 15.75, 19.9, 20.1, 24];
  const played = new Map();
  for (let t = 12; t <= 25; t = Math.round((t + 1 / 60) * 1e6) / 1e6) {
    const v = at(t);
    for (const s of probes) if (Math.abs(t - s) < 1 / 120 && !played.has(s)) played.set(s, [t, v]);
  }
  for (const s of [...probes].reverse()) {
    at(3); at(40);
    const [t, v] = played.get(s);
    assert.equal(at(t), v, `t = ${t}`);
  }
  // continuous through the phases: no jump at the first trace frame or at ignition
  let prev = null;
  for (let t = tr.start; t < tr.ignitionAt + 2; t += 0.05) {
    const out = makeSample();
    tl.sample(tr, t, out);
    if (prev) assert.ok(dist3(out.pos, prev) < 0.05 * 1000, `step at ${t.toFixed(2)}`);
    prev = out.pos.slice();
  }
});

test('old recordings (no sequence fields) draw as before: no launch state, no extra samples', () => {
  // A battle recorded before the launch sequence existed. Read from the committed recording
  // (the staged copy under public/recordings is git-ignored and absent in CI).
  const path = new URL('../../data/recordings/p5w5_b4_opus_haiku_mirror/'
    + 'battle_claude_opus_5_5_vs_claude_opus_5_5_20261118_024342.json.gz', import.meta.url);
  const rec = JSON.parse(gunzipSync(readFileSync(path)).toString('utf8'));
  const l = new BattleLoader();
  l.parse(rec);
  const tl = new WeaponTimeline(l);
  assert.ok(tl.torpedoes.size > 0);
  for (const tr of tl.torpedoes.values()) {
    assert.equal(tr.launch, undefined);
    assert.equal(tr.ignitionAt, undefined);
    assert.ok(!tr.samples.some(s => s.synthetic));
  }
  assert.ok(!tl.moments.some(m => m.kind === 'lightoff'));
  // rack dealing by count, exactly as before
  const idx = shipEventIndex(l.events);
  for (const launches of idx.launches.values()) {
    assert.ok(launches.every(x => x.mount === null));
    assert.deepEqual(bankTimes(assignRounds(launches, 4, s => launcherBank('cruiser_torpedo', s))), assignLaunches(launches, 4));
  }
});

// ----------------------------------------------------------------- director

test('the salvo shot holds until the last round of the salvo has lit its torch (lookahead only)', () => {
  const rec = demo();
  const events = rec.events;
  const hs = buildHighlights(events, {}, {});
  const salvos = hs.filter(h => h.kind === 'salvo');
  for (const h of salvos) assert.ok(Number.isFinite(h.lit) && h.lit > h.tLast, `salvo at ${h.t} lit ${h.lit}`);
  const causal = buildHighlights(events, {}, { causal: true }).filter(h => h.kind === 'salvo');
  for (const h of causal) assert.equal(h.lit, null);
  // a salvo whose last round lights 8 s after launch keeps the shot open past +5 s
  const fake = [0, 0, 0].map((_, i) => ({ timestamp: 10, event_type: 'torpedo_launched', ship_id: 'alpha_1', data: { torpedo_id: `t${i}`, target_id: 'beta_1' } }));
  fake.push({ timestamp: 18, event_type: 'torpedo_ignition', ship_id: 'alpha_1', data: { torpedo_id: 't2' } });
  const h = buildHighlights(fake, {}, {}).find(x => x.kind === 'salvo');
  assert.equal(h.lit, 18);
  assert.ok(SALVO_AFTER_LIGHTOFF_S > 1);
});
