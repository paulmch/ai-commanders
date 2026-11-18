import test from 'node:test';
import assert from 'node:assert/strict';
import {
  layoutTactical, clusterPoints, groupWaves, placeChip, shortName, nameStem, groupLabel, waveLabel,
  initialLabelMode, saveLabelMode, nextLabelMode, LABEL_MODES, LABEL_STORAGE_KEY, CLUSTER_PX, boxesOverlap, shipBox
} from '../src/TacticalLayout.js';
import { aggregateEngagements, engagementStyle, LINE_WEIGHT } from '../src/render/TargetingLines.js';

// Seeded PRNG (mulberry32): every random fleet is reproducible
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VIEW = { width: 1600, height: 900, top: 8, bottom: 8 };
const PANELS = [
  { x: 0, y: 0, w: 1600, h: 80 },      // header + scoreboard
  { x: 10, y: 90, w: 240, h: 670 },    // registry
  { x: 160, y: 815, w: 1280, h: 85 }   // transport
];
const TYPES = ['corvette', 'frigate', 'destroyer', 'battlecruiser'];

/** Two fleets of n ships each in loose formations, plus a wave or two. */
function randomBattle(seed, n = 8) {
  const r = rng(seed);
  const ships = [];
  for (const [f, cx, prefix] of [['alpha', 300 + 500 * r(), 'TIS'], ['beta', 900 + 500 * r(), 'OCS']]) {
    const cy = 200 + 500 * r(), spread = 20 + 220 * r();
    for (let i = 1; i <= n; i++) {
      ships.push({ id: `${f}_${i}`, faction: f, x: cx + (r() - 0.5) * spread, y: cy + (r() - 0.5) * spread * 1.4,
        r: 2 + 12 * r(), name: `${prefix} ${['Dart', 'Ward', 'Halberd'][i % 3]}-${i}`, type: TYPES[i % 4],
        dying: r() < 0.05 });
    }
  }
  const waves = [];
  for (let k = 0; k < 2; k++) {
    const t = ships[Math.floor(r() * ships.length)];
    const x = t.x + (r() - 0.5) * 300, y = t.y + (r() - 0.5) * 100;
    const w = { key: `${t.id}@${k}`, target: t.id, count: 1 + Math.floor(r() * 15), live: 1, eta: 60 * r(), x, y, tx: t.x, ty: t.y };
    w.text = waveLabel(w, shortName(t.name));
    waves.push(w);
  }
  return { ships, waves, selectedId: r() < 0.5 ? ships[Math.floor(r() * ships.length)].id : null };
}

function placedBoxes(out) {
  const boxes = [];
  for (const [id, p] of out.ships) if (p.box) boxes.push({ id, box: p.box });
  for (const g of out.groups) if (g.box) boxes.push({ id: g.key, box: g.box });
  for (const w of out.waves) if (w.box) boxes.push({ id: w.key, box: w.box });
  return boxes;
}

test('no two chips overlap, and no chip covers a hull (its own included) or a panel, for random fleets', () => {
  for (const mode of LABEL_MODES) {
    for (let seed = 1; seed <= 60; seed++) {
      const { ships, waves, selectedId } = randomBattle(seed);
      const out = layoutTactical({ view: VIEW, panels: PANELS, ships, waves, mode, selectedId });
      const boxes = placedBoxes(out);
      for (let i = 0; i < boxes.length; i++) {
        const a = boxes[i].box;
        assert.ok(a.x >= 0 && a.y >= 0 && a.x + a.w <= VIEW.width && a.y + a.h <= VIEW.height, `${mode}/${seed} ${boxes[i].id} off view`);
        for (let j = i + 1; j < boxes.length; j++) {
          assert.ok(!boxesOverlap(a, boxes[j].box), `${mode}/${seed}: ${boxes[i].id} overlaps ${boxes[j].id}`);
        }
        for (const s of ships) assert.ok(!boxesOverlap(a, shipBox(s)), `${mode}/${seed}: ${boxes[i].id} covers ${s.id}`);
        for (const p of PANELS) assert.ok(!boxesOverlap(a, p), `${mode}/${seed}: ${boxes[i].id} on a panel`);
      }
    }
  }
});

test('density modes: auto groups crowded ships, minimal shows brackets only, full a card per ship', () => {
  const ships = [
    { id: 'alpha_1', faction: 'alpha', x: 600, y: 400, r: 4, name: 'TIS Dart-1', type: 'corvette' },
    { id: 'alpha_2', faction: 'alpha', x: 612, y: 408, r: 4, name: 'TIS Dart-2', type: 'corvette' },
    { id: 'alpha_3', faction: 'alpha', x: 620, y: 395, r: 4, name: 'TIS Dart-3', type: 'corvette' },
    { id: 'alpha_4', faction: 'alpha', x: 600, y: 600, r: 4, name: 'TIS Ward-1', type: 'frigate' },
    { id: 'beta_1', faction: 'beta', x: 1100, y: 400, r: 6, name: 'OCS Halberd-1', type: 'battlecruiser', dying: true },
    { id: 'beta_2', faction: 'beta', x: 1100, y: 550, r: 6, name: 'OCS Halberd-2', type: 'battlecruiser' }
  ];
  const auto = layoutTactical({ view: VIEW, ships, selectedId: 'alpha_4' });
  assert.equal(auto.groups.length, 1);
  assert.deepEqual(auto.groups[0].ids, ['alpha_1', 'alpha_2', 'alpha_3']);
  assert.equal(auto.groups[0].text, 'Dart ×3');
  assert.equal(auto.ships.get('alpha_1').kind, 'group');
  assert.equal(auto.ships.get('alpha_4').kind, 'card');   // selected
  assert.equal(auto.ships.get('beta_1').kind, 'card');    // dying: the detonation countdown
  assert.equal(auto.ships.get('beta_2').kind, 'chip');
  // chips go on the outside flank, away from the other side
  assert.ok(auto.ships.get('beta_2').box.x > 1100);
  assert.ok(auto.groups[0].box.x + auto.groups[0].box.w < 600);
  const minimal = layoutTactical({ view: VIEW, ships, mode: 'minimal' });
  assert.equal(minimal.groups.length, 0);
  assert.deepEqual([...minimal.ships].filter(([, p]) => p.box).map(([id]) => id), ['beta_1']);
  const full = layoutTactical({ view: VIEW, ships, mode: 'full' });
  assert.equal(full.groups.length, 0);
  assert.ok([...full.ships.values()].every(p => p.kind === 'card'));
});

test('clustering is order-independent, unchanged by a pan, and a zoom-in only ever splits clusters', () => {
  const partition = cs => cs.map(c => c.map(p => p.id).join('+')).sort().join(' | ');
  for (let seed = 1; seed <= 40; seed++) {
    const { ships } = randomBattle(seed);
    const base = partition(clusterPoints(ships));
    const r = rng(seed * 7);
    const shuffled = [...ships].sort(() => r() - 0.5);
    assert.equal(partition(clusterPoints(shuffled)), base, `order ${seed}`);
    const panned = ships.map(s => ({ ...s, x: s.x + 137.25, y: s.y - 61.5 }));
    assert.equal(partition(clusterPoints(panned)), base, `pan ${seed}`);
    // zoom in about the screen centre in small steps: every partition refines the one before
    let prev = clusterPoints(ships).map(c => new Set(c.map(p => p.id)));
    for (let z = 1.02; z < 3; z *= 1.02) {
      const zoomed = ships.map(s => ({ ...s, x: 800 + (s.x - 800) * z, y: 450 + (s.y - 450) * z }));
      const cur = clusterPoints(zoomed).map(c => new Set(c.map(p => p.id)));
      for (const c of cur) assert.ok(prev.some(p => [...c].every(id => p.has(id))), `zoom ${seed} x${z.toFixed(2)} merged`);
      prev = cur;
    }
  }
  // a 1 px nudge changes nothing unless a pair sits within 1 px of the threshold
  const pts = [{ id: 'a', faction: 'alpha', x: 0, y: 0 }, { id: 'b', faction: 'alpha', x: CLUSTER_PX - 5, y: 0 }];
  assert.equal(clusterPoints(pts).length, 1);
  assert.equal(clusterPoints(pts.map(p => ({ ...p, x: p.x + 1, y: p.y + 1 }))).length, 1);
  // sides never share a cluster
  assert.equal(clusterPoints([pts[0], { id: 'c', faction: 'beta', x: 1, y: 1 }]).length, 2);
});

test('layout is a pure function of its inputs: input order and repeated calls give the same result', () => {
  for (let seed = 1; seed <= 20; seed++) {
    const b = randomBattle(seed);
    const a1 = layoutTactical({ view: VIEW, panels: PANELS, ...b });
    const r = rng(seed);
    const a2 = layoutTactical({ view: VIEW, panels: PANELS, ...b, ships: [...b.ships].sort(() => r() - 0.5) });
    const norm = o => ({ ships: [...o.ships].sort(([x], [y]) => (x < y ? -1 : 1)), groups: o.groups, waves: o.waves });
    assert.deepEqual(norm(a2), norm(a1), `seed ${seed}`);
    assert.deepEqual(norm(layoutTactical({ view: VIEW, panels: PANELS, ...b })), norm(a1));
  }
});

test('chips hold their slot while fleets drift smoothly', () => {
  // two formations drifting toward each other 1 px a frame: a chip may hop
  // to another slot only when something new blocks it, not every frame
  const { ships } = randomBattle(5);
  let prev = null, hops = 0, frames = 0;
  for (let i = 0; i < 200; i++) {
    const moved = ships.map(s => ({ ...s, x: s.x + (s.faction === 'alpha' ? i : -i) * 0.5, y: s.y + i * 0.2 }));
    const out = layoutTactical({ view: VIEW, panels: PANELS, ships: moved });
    const rel = new Map();
    for (const s of moved) {
      const p = out.ships.get(s.id);
      if (p.box) rel.set(s.id, `${Math.round(p.box.x - s.x)},${Math.round(p.box.y - s.y)}`);
    }
    if (prev) {
      for (const [id, v] of rel) { frames++; if (prev.has(id) && prev.get(id) !== v) hops++; }
    }
    prev = rel;
  }
  assert.ok(hops <= frames * 0.05, `${hops} slot hops in ${frames} chip-frames`);
});

test('placeChip prefers the given side, then the other, and never leaves the view', () => {
  const right = placeChip(500, 300, 10, 60, 16, 1, [], VIEW);
  assert.ok(right.x > 500);
  const left = placeChip(500, 300, 10, 60, 16, -1, [], VIEW);
  assert.ok(left.x + left.w < 500);
  const blocked = placeChip(500, 300, 10, 60, 16, 1, [{ x: 505, y: 0, w: 400, h: 900 }], VIEW);
  assert.ok(blocked.x + blocked.w <= 505);
  const edge = placeChip(1595, 300, 10, 60, 16, 1, [], VIEW);
  assert.ok(edge.x + edge.w <= VIEW.width);
});

test('torpedo waves: one per target and launch burst, identity fixed while the rounds fly', () => {
  const rounds = (t) => [
    { id: 't1', target: 'beta_1', launch: 30, eta: 80 - t },
    { id: 't2', target: 'beta_1', launch: 31.5, eta: 79 - t },
    { id: 't3', target: 'beta_1', launch: 36, eta: 75 - t, blind: true },
    { id: 't4', target: 'beta_1', launch: 70, eta: 120 - t },   // a second salvo: its own wave
    { id: 't5', target: 'beta_2', launch: 31, eta: 90 - t }
  ];
  const w0 = groupWaves(rounds(0));
  assert.equal(w0.length, 3);
  assert.deepEqual(w0.map(w => [w.target, w.count, w.live]), [['beta_1', 3, 2], ['beta_2', 1, 1], ['beta_1', 1, 1]]);
  assert.equal(w0[0].head, 't3');
  assert.equal(w0[0].eta, 75);
  // forty seconds on: same waves (keys), shorter ETAs
  const w40 = groupWaves(rounds(40));
  assert.deepEqual(w40.map(w => w.key), w0.map(w => w.key));
  assert.equal(w40[0].eta, 35);
  assert.equal(waveLabel(w40[0], 'Halberd-1'), '3 TORP ▸ Halberd-1 · 35s');
});

test('names: short names, stems and group chip text', () => {
  assert.equal(shortName('TIS Dart-5'), 'Dart-5');
  assert.equal(shortName('OCS Halberd-1'), 'Halberd-1');
  assert.equal(shortName('Relentless'), 'Relentless');
  assert.equal(shortName('alpha_3'), 'alpha_3');
  assert.equal(nameStem('Dart-5'), 'Dart');
  assert.equal(groupLabel([{ name: 'TIS Dart-1' }, { name: 'TIS Dart-4' }]), 'Dart ×2');
  assert.equal(groupLabel([{ name: 'TIS Dart-1', type: 'corvette' }, { name: 'TIS Ward-1', type: 'corvette' }]), '2 corvettes');
  assert.equal(groupLabel([{ name: 'A', type: 'corvette' }, { name: 'B', type: 'frigate' }, { name: 'C', type: 'frigate' }]), '3 ships');
});

test('label density: ?labels= wins for the page load, then the saved choice, then auto; L cycles', () => {
  const store = new Map();
  const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  assert.equal(initialLabelMode('', storage), 'auto');
  saveLabelMode(storage, 'minimal');
  assert.equal(store.get(LABEL_STORAGE_KEY), 'minimal');
  assert.equal(initialLabelMode('', storage), 'minimal');
  assert.equal(initialLabelMode('?labels=full', storage), 'full');
  assert.equal(initialLabelMode('?labels=bogus', storage), 'minimal');
  assert.equal(initialLabelMode('', { getItem() { throw new Error('denied'); } }), 'auto');
  assert.deepEqual([nextLabelMode('auto'), nextLabelMode('minimal'), nextLabelMode('full')], ['minimal', 'full', 'auto']);
});

test('engagement lines: shooters together on screen share one weighted line per target', () => {
  const pairs = [
    { from: 'alpha_1', to: 'beta_1', faction: 'alpha', heat: 0, active: true },
    { from: 'alpha_2', to: 'beta_1', faction: 'alpha', heat: 0.5, active: true },
    { from: 'alpha_3', to: 'beta_1', faction: 'alpha', heat: 0, active: false },
    { from: 'alpha_4', to: 'beta_2', faction: 'alpha', heat: 0, active: false },
    { from: 'beta_1', to: 'alpha_1', faction: 'beta', heat: 0, active: true }
  ];
  const at = { alpha_1: { x: 0, y: 0 }, alpha_2: { x: 10, y: 5 }, alpha_3: { x: 300, y: 0 }, alpha_4: { x: 5, y: 5 }, beta_1: { x: 900, y: 0 } };
  const lines = aggregateEngagements(pairs, id => at[id] || null);
  const key = l => `${l.from.join('+')}>${l.to}`;
  assert.deepEqual(lines.map(key).sort(), ['alpha_1+alpha_2>beta_1', 'alpha_3>beta_1', 'alpha_4>beta_2', 'beta_1>alpha_1']);
  const shared = lines.find(l => l.n === 2);
  assert.equal(shared.heat, 0.5);
  const single = lines.find(l => key(l) === 'alpha_3>beta_1');
  assert.ok(engagementStyle(shared, 'auto').width > engagementStyle(single, 'auto').width);
  // auto: faint, the idle (ordered-only) line fainter, the focused ship's line strong
  assert.ok(engagementStyle(single, 'auto').strength < engagementStyle(lines.find(l => l.to === 'alpha_1'), 'auto').strength);
  assert.equal(engagementStyle(single, 'auto', 'alpha_3').strength, LINE_WEIGHT.focus);
  assert.equal(engagementStyle(single, 'auto', 'beta_1').strength, LINE_WEIGHT.focus);   // lines onto the focus too
  // minimal: only the focus and a shot's flare
  assert.equal(engagementStyle(single, 'minimal').strength, 0);
  assert.ok(engagementStyle(shared, 'minimal').strength > 0);
  // full: the pass-4 weight, one-ship width
  assert.equal(engagementStyle(single, 'full').strength, LINE_WEIGHT.full);
});

test('wave chips are weighted by urgency: forecast, inbound, imminent, blind', async () => {
  const { waveUrgency, THREAT_IMMINENT_S, WAVE_DISTANT_S } = await import('../src/ContactOverlay.js');
  assert.equal(waveUrgency({ live: 3, eta: WAVE_DISTANT_S + 1 }), 'distant');
  assert.equal(waveUrgency({ live: 3, eta: 30 }), 'inbound');
  assert.equal(waveUrgency({ live: 1, eta: THREAT_IMMINENT_S - 1 }), 'imminent');
  assert.equal(waveUrgency({ live: 0, eta: 2 }), 'blind');
});

test('scrubbing: the layout at t is the same played through, after a seek away and back', () => {
  // ships on straight tracks; layout(t) only sees the positions at t
  const { ships: base, waves } = randomBattle(11);
  const at = t => layoutTactical({ view: VIEW, panels: PANELS, waves: waves.map(w => ({ ...w, eta: w.eta - t })),
    ships: base.map((s, i) => ({ ...s, x: s.x + Math.sin(i) * t * 3, y: s.y + Math.cos(i) * t * 2 })) });
  const played = [];
  for (let t = 0; t <= 10; t += 0.5) played.push(at(t));
  at(80);                                  // seek far away
  const back = at(4);                      // and back
  const norm = o => JSON.stringify({ ships: [...o.ships].sort(([a], [b]) => (a < b ? -1 : 1)), groups: o.groups, waves: o.waves });
  assert.equal(norm(back), norm(played[8]));
});

test('one wave chip per target: the soonest wave, later rounds counted on it', async () => {
  const { wavesByTarget } = await import('../src/TacticalLayout.js');
  const waves = groupWaves([
    { id: 'a', target: 'beta_1', launch: 10, eta: 5 }, { id: 'b', target: 'beta_1', launch: 40, eta: 30 },
    { id: 'c', target: 'beta_1', launch: 41, eta: 31 }, { id: 'd', target: 'beta_2', launch: 10, eta: 60 }
  ]);
  const chips = wavesByTarget(waves);
  assert.deepEqual(chips.map(w => [w.target, w.count, w.later]), [['beta_1', 1, 2], ['beta_2', 1, 0]]);
  assert.equal(waveLabel(chips[0], 'Halberd-1'), '1 TORP ▸ Halberd-1 · 5s +2');
});

// --- wave-chip ETA (render/WeaponTimeline.torpedoEta) ----------------------------------------
import { torpedoEta, ETA_CRUISE_MPS, ETA_BOOST_MPS2 } from '../src/render/WeaponTimeline.js';

test('wave-chip ETA: a round still on its launch sequence or in its boost gets a flight estimate, not range / closure', () => {
  // The owner's screenshot: four rounds just off the rack, ~500 km out, closing at ~18 m/s,
  // read "27664s". Launch to impact from 500 km is about 75 s.
  const cold = torpedoEta(500e3, 18, true, 3);
  assert.ok(cold > 60 && cold < 100, `cold round ETA ${cold}`);
  assert.ok(500e3 / 18 > 20000);                       // what the chip used to print
  // In its boost: closing at 1.2 km/s after 10 s; the naive figure was 408 s.
  const boost = torpedoEta(490e3, 1200, true);
  assert.ok(boost > 55 && boost < 95, `boosting round ETA ${boost}`);
  // At cruise speed, coasting or blinded: plain range / closure, as before.
  assert.equal(torpedoEta(250e3, ETA_CRUISE_MPS, true), 250e3 / ETA_CRUISE_MPS);
  assert.equal(torpedoEta(250e3, 9000, false), 250e3 / 9000);
  assert.equal(torpedoEta(250e3, 0, false), Infinity);  // not closing, no engine: no ETA
  // A short shot ends inside the boost: s = v t + a t^2 / 2.
  const t = torpedoEta(20e3, 0, true);
  assert.ok(Math.abs(0.5 * ETA_BOOST_MPS2 * t * t - 20e3) < 1);
  // Monotonic: it never grows as the round gets closer or faster.
  let last = Infinity;
  for (let r = 500e3, v = 0; r > 0; r -= 25e3, v = Math.min(v + 1500, ETA_CRUISE_MPS)) {
    const e = torpedoEta(r, v, v < ETA_CRUISE_MPS);
    assert.ok(e <= last + 1e-9, `ETA rose at range ${r}`); last = e;
  }
});
