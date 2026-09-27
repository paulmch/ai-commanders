import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildHeatTimelines, deployProgress, radiatorsExtendedAt, radiatorFraction, heatAt,
  radiatorGlow, radiatorTemperature, hullHeatLook, easeDeploy,
  RADIATOR_DEPLOY_S, RAD_FULL_GW, LEGACY_RAD_FRACTION, MELTDOWN_HEAT
} from '../src/render/Heat.js';

/** 1 Hz trace of one ship with per-frame fields from fn(t). */
function trace(n, fn, id = 'beta_1') {
  return Array.from({ length: n }, (_, t) => ({ t, ships: { [id]: { pos: [0, 0, 0], ...fn(t) } } }));
}

test('deploy progress runs over RADIATOR_DEPLOY_S from the recorded flip frame', () => {
  const D = RADIATOR_DEPLOY_S;
  assert.ok(D >= 3 && D <= 4);
  const tl = buildHeatTimelines(trace(60, t => ({ rad: t >= 10 && t < 40 ? 1 : 0 }))).get('beta_1');
  assert.equal(tl.flips.length, 2);
  assert.equal(deployProgress(tl, 9.99), 0);
  assert.equal(deployProgress(tl, 10), 0);
  assert.ok(Math.abs(deployProgress(tl, 10 + D / 2) - 0.5) < 1e-9);
  assert.equal(deployProgress(tl, 10 + D), 1);
  assert.equal(deployProgress(tl, 39), 1);
  assert.ok(Math.abs(deployProgress(tl, 40 + D / 4) - 0.75) < 1e-9);
  assert.equal(deployProgress(tl, 40 + D + 0.1), 0);
  assert.equal(radiatorsExtendedAt(tl, 12), true);
  assert.equal(radiatorsExtendedAt(tl, 41), false);
});

test('deploy progress is a pure function of battle time (seek order does not matter)', () => {
  const tl = buildHeatTimelines(trace(60, t => ({ rad: t >= 10 && t < 12 ? 1 : 0 }))).get('beta_1');
  const times = [0, 11, 13.3, 10.5, 14, 11.7, 12.2, 55, 12.9];
  const forward = times.map(t => deployProgress(tl, t));
  const again = times.slice().reverse().map(t => deployProgress(tl, t)).reverse();
  assert.deepEqual(forward, again);
  // a retract 2 s into the deploy folds back from the reached progress
  const p12 = 2 / RADIATOR_DEPLOY_S;
  assert.ok(Math.abs(deployProgress(tl, 12) - p12) < 1e-9);
  assert.ok(Math.abs(deployProgress(tl, 13) - (p12 - 1 / RADIATOR_DEPLOY_S)) < 1e-9);
  assert.equal(deployProgress(tl, 12 + 2 + 1e-6), 0);
});

test('ships that start extended begin fully deployed', () => {
  const tl = buildHeatTimelines(trace(20, () => ({ rad: 1, rad_out: 0.3 }))).get('beta_1');
  assert.equal(deployProgress(tl, 0), 1);
  assert.equal(deployProgress(tl, 15), 1);
});

test('old recordings deploy from radiator_change events and glow at a nominal level', () => {
  const events = [
    { event_type: 'radiator_change', ship_id: 'beta_1', timestamp: 30, data: { extended: true } },
    { event_type: 'radiator_change', ship_id: 'beta_1', timestamp: 60, data: { extended: true } },   // repeat
    { event_type: 'radiator_change', ship_id: 'beta_1', timestamp: 90, data: { extended: false } }
  ];
  const tl = buildHeatTimelines(trace(120, () => ({})), events).get('beta_1');
  assert.equal(tl.flips.length, 2, 'a repeated extend order is not a flip');
  assert.equal(deployProgress(tl, 29), 0);
  assert.equal(deployProgress(tl, 30 + RADIATOR_DEPLOY_S), 1);
  assert.equal(deployProgress(tl, 75), 1);
  assert.equal(radiatorFraction(tl, 75), LEGACY_RAD_FRACTION);
  assert.ok(Number.isNaN(heatAt(tl, 75)));
});

test('a trace flip is dated by the radiator_change order since the previous frame', () => {
  // the command lands at the checkpoint (30.0); the 1 Hz trace first shows rad=1 at 31
  const events = [
    { event_type: 'radiator_change', ship_id: 'beta_1', timestamp: 30, data: { extended: true } },
    { event_type: 'radiator_change', ship_id: 'beta_1', timestamp: 10, data: { extended: true } }   // stale
  ];
  const tl = buildHeatTimelines(trace(60, t => ({ rad: t >= 31 && t < 45 ? 1 : 0 })), events).get('beta_1');
  assert.equal(tl.flips.length, 2);
  assert.equal(tl.flips[0].t, 30);
  assert.equal(tl.flips[1].t, 45, 'no order at the retract: the trace frame stands');
  assert.ok(Math.abs(deployProgress(tl, 31) - 1 / RADIATOR_DEPLOY_S) < 1e-9);
});

test('radiator fraction follows rad_out while extended and fades from the last output on retract', () => {
  const tl = buildHeatTimelines(trace(40, t => ({
    rad: t >= 5 && t < 20 ? 1 : 0,
    rad_out: t >= 5 && t < 20 ? (t < 10 ? 0.078 : 0.78) : 0
  }))).get('beta_1');
  assert.ok(Math.abs(radiatorFraction(tl, 7) - 0.1) < 1e-6);
  assert.ok(Math.abs(radiatorFraction(tl, 15) - 1) < 1e-6);
  // retract at 20: rad_out is 0 from that frame, the folding panels keep the last output
  assert.ok(Math.abs(radiatorFraction(tl, 21) - 1) < 1e-6);
  assert.equal(radiatorFraction(tl, 30) * deployProgress(tl, 30), 0);
});

test('glow is dark at zero output, visible at a trickle, bright orange-yellow at full 780 MW', () => {
  assert.equal(radiatorGlow(0).intensity, 0);
  const low = radiatorGlow(0.03 / RAD_FULL_GW);
  assert.ok(low.intensity > 0.5, 'any output stays visible');
  assert.ok(low.g / low.r < 0.07, 'low output is dull cherry red');
  const full = radiatorGlow(1);
  assert.ok(full.intensity > 2.8, 'full output passes the bloom threshold at the fin root');
  assert.ok(full.g / full.r > 0.4 && full.b < full.g, 'full output is orange-yellow');
  // monotonic brightness and hue with output
  let prev = radiatorGlow(0.01);
  for (let f = 0.05; f <= 1.0001; f += 0.05) {
    const g = { ...radiatorGlow(f) };
    assert.ok(g.intensity > prev.intensity);
    assert.ok(g.g / g.r >= prev.g / prev.r - 1e-9);
    prev = g;
  }
  assert.equal(Math.round(radiatorTemperature(1)), 1500);
  assert.ok(Math.abs(radiatorTemperature(0.1) - 843.5) < 1);
});

test('hull heat look: nothing cool, faint shimmer from ~90 %, pulsing warning above 100 % growing to meltdown', () => {
  assert.equal(hullHeatLook(0.5, 3).level, 0);
  assert.equal(hullHeatLook(NaN, 3).level, 0);
  const shimmer = hullHeatLook(0.95, 3).level;
  assert.ok(shimmer > 0 && shimmer < 0.25);
  assert.equal(hullHeatLook(0.95, 3).warn, 0);
  // warning pulses on battle time (1.25 Hz): peak and trough of one period
  const peak = hullHeatLook(1.25, 0.4), trough = hullHeatLook(1.25, 0);
  assert.ok(peak.level > trough.level * 1.5);
  assert.ok(Math.abs(peak.warn - 0.5) < 1e-9);
  // brighter and wider toward the meltdown threshold
  const near = hullHeatLook(MELTDOWN_HEAT - 0.01, 0.4);
  assert.ok(near.level > peak.level && near.reach > peak.reach);
  assert.equal(hullHeatLook(2, 0.4).warn, 1);
  // deterministic
  assert.deepEqual(hullHeatLook(1.3, 7.1), hullHeatLook(1.3, 7.1));
});

test('heat samples interpolate between 1 Hz frames and missing fields stay NaN', () => {
  const tl = buildHeatTimelines(trace(10, t => ({ heat: t * 0.1 }))).get('beta_1');
  assert.ok(Math.abs(heatAt(tl, 4.5) - 0.45) < 1e-6);
  assert.ok(Math.abs(tl.maxHeat - 0.9) < 1e-6);
  const bare = buildHeatTimelines(trace(10, () => ({}))).get('beta_1');
  assert.ok(Number.isNaN(heatAt(bare, 4)));
  assert.equal(bare.hasHeat, false);
});

test('panel ease is monotonic from stowed to extended', () => {
  assert.equal(easeDeploy(0), 0);
  assert.equal(easeDeploy(1), 1);
  let prev = 0;
  for (let p = 0.05; p <= 1; p += 0.05) { assert.ok(easeDeploy(p) > prev); prev = easeDeploy(p); }
});

test('registry heat bar: 0..meltdown scale, amber from 90 %, critical from 100 %', async () => {
  const { heatBar } = await import('../src/HeatHud.js');
  assert.deepEqual({ ...heatBar(NaN) }, { width: 0, level: 'none' });
  assert.equal(heatBar(0.3).level, 'cool');
  assert.equal(heatBar(0.92).level, 'hot');
  assert.equal(heatBar(1.0).level, 'crit');
  assert.equal(heatBar(0.75).width, 50);
  assert.equal(heatBar(2).width, 100);
  const out = { width: 0, level: '' };
  assert.equal(heatBar(1.2, MELTDOWN_HEAT, out), out);
});

test('battle title shrinks to fit one line, down to a floor, then ellipsizes', async () => {
  const { fitFontPx } = await import('../src/TitleFit.js');
  assert.equal(fitFontPx(300, 500, 18, 11), 18);          // fits: unchanged
  assert.equal(fitFontPx(600, 500, 18, 11), 15);          // scaled to fit
  assert.equal(fitFontPx(2000, 500, 18, 11), 11);         // floor; CSS ellipsis takes the rest
  assert.equal(fitFontPx(0, 500, 18, 11), 18);
  assert.ok(fitFontPx(911, 640, 18, 11) * 911 / 18 <= 640);
});

test('an overheat meltdown detonates at once, flagged for the meltdown blast', async () => {
  const { BattleLoader } = await import('../src/BattleLoader.js');
  const loader = new BattleLoader();
  const s = (destroyed) => ({ pos: [0, 0, 0], vel: [100, 0, 0], fwd: [1, 0, 0], destroyed });
  loader.simTrace = [
    { t: 0, ships: { a: s(false), b: s(false), c: s(false) } },
    { t: 1, ships: { a: s(true), b: s(true), c: s(false) } }
  ];
  loader.events = [
    { timestamp: 1, event_type: 'reactor_meltdown', ship_id: 'a', data: { heat: 1.503 } },
    { timestamp: 1, event_type: 'ship_destroyed', ship_id: 'a', data: { cause: 'reactor_overheat' } },
    { timestamp: 1, event_type: 'ship_destroyed', ship_id: 'b', data: { cause: 'hull_breach' } }
  ];
  const info = loader.buildDeathInfo();
  assert.equal(info.a.meltdown, true);
  assert.equal(info.a.reactorCause, true, 'no drift phase: containment fails at once');
  assert.ok(!info.b.meltdown && !info.b.reactorCause);
  // either marker alone is enough
  loader.events = [loader.events[1]];
  assert.equal(loader.buildDeathInfo().a.meltdown, true);
});
