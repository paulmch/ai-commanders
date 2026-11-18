/**
 * TacticalLayout - the fleet-range contact layer's screen layout, as pure
 * functions (no DOM, no THREE): which ships get a chip, a full card or only
 * their bracket, which crowded ships share one group chip, where every chip
 * and torpedo-wave chip goes so that none overlaps another, a hull or a HUD
 * panel, and the label density mode (L key, ?labels=, localStorage).
 *
 * Everything here is a function of the screen positions it is given (the
 * camera and battle time upstream): no easing, no memory of the last frame,
 * so a seek away and back lays the scene out identically. ContactOverlay
 * projects the ships and applies the result to the DOM.
 *
 * Density modes:
 *   auto     a compact chip per ship (short name; a hull bar once damaged),
 *            crowded ships of one side collapse into a group chip; the full
 *            card only for the selected or dying ship (hover expands any)
 *   minimal  brackets only; cards for the selected and dying ships
 *   full     every ship's full card (the pre-pass-5 look), no grouping
 * Torpedo-wave chips (count, target, time to impact) show in every mode.
 */
import { getShipClassName } from './shipSilhouettes.js';

export const LABEL_MODES = ['auto', 'minimal', 'full'];
export const DEFAULT_LABEL_MODE = 'auto';
export const LABEL_STORAGE_KEY = 'aic.viewer.labels';

/** Ships of one side closer than this on screen (px) share one group chip. */
export const CLUSTER_PX = 30;
/** Rounds at one target launched within this many seconds of each other are one wave. */
export const WAVE_GAP_S = 8;
/** Half-size (px) of the contact bracket: the smallest box a ship reserves. */
export const BRACKET_PX = 10;
/** Gap (px) between a hull's reserved box and the chip next to it. */
const GAP_PX = 6;
/** Advance of the chips' 10 px mono face (Share Tech Mono), px per character. */
export const CHAR_PX = 5.6;

export const CARD = { w: 126, h: 34 };
const CHIP_H = 16, WAVE_H = 15;

export const normalizeLabelMode = m => (LABEL_MODES.includes(m) ? m : DEFAULT_LABEL_MODE);
export const nextLabelMode = m => LABEL_MODES[(LABEL_MODES.indexOf(normalizeLabelMode(m)) + 1) % LABEL_MODES.length];

/** ?labels= wins for this page load, then the saved choice, then auto. */
export function initialLabelMode(search = '', storage = null) {
  const q = new URLSearchParams(search || '').get('labels');
  if (q && LABEL_MODES.includes(q)) return q;
  try { return normalizeLabelMode(storage?.getItem(LABEL_STORAGE_KEY)); } catch { return DEFAULT_LABEL_MODE; }
}

export function saveLabelMode(storage, mode) {
  try { storage?.setItem(LABEL_STORAGE_KEY, normalizeLabelMode(mode)); } catch { /* not persisted */ }
}

/** "TIS Dart-5" -> "Dart-5": the registry prefix is the same for a whole side. */
export function shortName(name) {
  const s = String(name ?? '').trim();
  const m = s.match(/^[A-Z]{2,4}\.?\s+(\S.*)$/);
  return m ? m[1] : s;
}

/** "Dart-5" -> "Dart" (the hull number dropped), '' when nothing is left. */
export function nameStem(short) {
  return String(short ?? '').replace(/[\s#-]*\d+[a-z]?$/i, '').trim();
}

/** Group chip text: "Dart ×5" when the names share a stem, else "3 corvettes" / "4 ships". */
export function groupLabel(members) {
  const n = members.length;
  const stems = new Set(members.map(m => nameStem(m.short ?? shortName(m.name))));
  if (stems.size === 1) {
    const [stem] = stems;
    if (stem) return `${stem} ×${n}`;
  }
  const types = new Set(members.map(m => m.type));
  if (types.size === 1) {
    const cls = getShipClassName([...types][0]).toLowerCase();
    if (cls !== 'unknown') return `${n} ${cls.endsWith('s') ? cls : `${cls}s`}`;
  }
  return `${n} ships`;
}

/** Torpedo-wave chip text: "15 TORP ▸ Halberd-1 · 6s", "+10" for later waves at the same target. */
export function waveLabel(wave, targetShort = '') {
  const eta = Number.isFinite(wave.eta) ? `${Math.max(0, Math.round(wave.eta))}s` : '--';
  return `${wave.count} TORP${targetShort ? ` ▸ ${targetShort}` : ''} · ${eta}${wave.later > 0 ? ` +${wave.later}` : ''}`;
}

/**
 * One chip per target: its soonest wave (groupWaves order), carrying the
 * rounds of the later waves at the same target as `later`.
 */
export function wavesByTarget(waves) {
  const out = new Map();
  for (const w of waves) {
    const first = out.get(w.target);
    if (first) first.later += w.count;
    else out.set(w.target, { ...w, later: 0 });
  }
  return [...out.values()];
}

export const textWidth = (text, pad = 12) => Math.ceil(pad + CHAR_PX * String(text ?? '').length);

const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
export const boxesOverlap = overlaps;

/** The square a ship reserves on screen: its bracket, or its drawn hull when that is bigger. */
export function shipBox(s) {
  const r = Math.max(BRACKET_PX, s.r || 0);
  return { x: s.x - r, y: s.y - r, w: 2 * r, h: 2 * r };
}

/**
 * Single-linkage clusters of same-side points closer than `radius` px.
 * Order-independent (ids sorted inside a cluster, clusters by first id);
 * a uniform zoom-in only ever splits clusters, a pan leaves them as they are.
 * @param {Array<{id, faction, x, y}>} points
 * @returns {Array<Array<point>>}
 */
export function clusterPoints(points, radius = CLUSTER_PX) {
  const pts = [...points].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const parent = pts.map((_, i) => i);
  const find = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  const r2 = radius * radius;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      if (pts[i].faction !== pts[j].faction) continue;
      const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
      if (dx * dx + dy * dy < r2) {
        const a = find(i), b = find(j);
        if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
      }
    }
  }
  const byRoot = new Map();
  pts.forEach((p, i) => {
    const r = find(i);
    if (!byRoot.has(r)) byRoot.set(r, []);
    byRoot.get(r).push(p);
  });
  return [...byRoot.values()];
}

/**
 * Torpedo waves: rounds at one target whose launch times chain within
 * `gap` s. Grouped by launch (fixed for a round's life), not by the
 * current ETA, so a wave keeps its identity as it flies.
 * @param {Array<{id, target, launch, eta, blind?}>} rounds - threatening rounds
 * @returns {Array<{key, target, count, live, eta, head}>} by soonest ETA; head = the round with it
 */
export function groupWaves(rounds, gap = WAVE_GAP_S) {
  const byTarget = new Map();
  for (const r of rounds) {
    if (!r.target) continue;
    if (!byTarget.has(r.target)) byTarget.set(r.target, []);
    byTarget.get(r.target).push(r);
  }
  const waves = [];
  for (const [target, list] of byTarget) {
    list.sort((a, b) => a.launch - b.launch || (a.id < b.id ? -1 : 1));
    let cur = null;
    for (const r of list) {
      if (!cur || r.launch - cur.last > gap) {
        cur = { key: `${target}@${r.launch.toFixed(1)}`, target, count: 0, live: 0, eta: Infinity, head: null, last: r.launch };
        waves.push(cur);
      }
      cur.last = r.launch;
      cur.count++;
      if (!r.blind) cur.live++;
      const eta = Number.isFinite(r.eta) ? r.eta : Infinity;
      if (cur.head === null || eta < cur.eta) { cur.eta = eta; cur.head = r.id; }
    }
  }
  for (const w of waves) delete w.last;
  return waves.sort((a, b) => a.eta - b.eta || (a.key < b.key ? -1 : 1));
}

/**
 * A clear spot for a w x h chip by the anchor (x, y) whose reserved box
 * has half-size r: beside it on `side` (+1 right, -1 left) first, then
 * the other side, above / below, then a wider ring (with a leader line).
 * Never on another box in `occupied`, never off the view. Null when full.
 */
export function placeChip(x, y, r, w, h, side, occupied, view, verticalFirst = false) {
  const { width, height, top = 8, bottom = 8, margin = 8 } = view;
  const fits = b => b.x >= margin && b.y >= top && b.x + b.w <= width - margin && b.y + b.h <= height - bottom &&
    !occupied.some(o => overlaps(b, o));
  const sides = [side || 1, -(side || 1)];
  const step = h + 3;
  if (verticalFirst) {
    // above / below, the near end at the anchor: off a stream of rounds flying sideways
    for (const dy of [-1, 1]) {
      const b = { x: side < 0 ? x - w + 4 : x - 4, y: dy < 0 ? y - r - GAP_PX - h : y + r + GAP_PX, w, h };
      if (fits(b)) return b;
    }
  }
  for (const reach of [r + GAP_PX, r + GAP_PX + 26]) {
    for (const dir of sides) {
      for (const k of [0, -1, 1, -2, 2]) {
        const b = { x: dir > 0 ? x + reach : x - reach - w, y: y - h / 2 + k * step, w, h };
        if (fits(b)) return b;
      }
    }
    for (const dy of [-1, 1]) {
      const b = { x: x - w / 2, y: dy < 0 ? y - reach - h : y + reach, w, h };
      if (fits(b)) return b;
    }
  }
  for (const dir of sides) {
    for (const k of [-3, 3, -4, 4]) {
      const b = { x: dir > 0 ? x + r + GAP_PX : x - r - GAP_PX - w, y: y - h / 2 + k * step, w, h };
      if (fits(b)) return b;
    }
  }
  return null;
}

/** Leader from the anchor to the near edge of `box`: {x, y, len, angle} in px / rad, or null when it touches. */
export function leaderTo(x, y, r, box) {
  const nx = Math.max(box.x, Math.min(x, box.x + box.w));
  const ny = Math.max(box.y, Math.min(y, box.y + box.h));
  const dx = nx - x, dy = ny - y, len = Math.hypot(dx, dy);
  return len > r + GAP_PX + 4 ? { x, y, len, angle: Math.atan2(dy, dx) } : null;
}

/**
 * Lay out the contact layer for one frame.
 * @param {Object} o
 * @param {{width, height, top?, bottom?}} o.view
 * @param {Array<{x, y, w, h}>} o.panels - HUD panels to keep clear of
 * @param {Array<{id, faction, x, y, r, name, type, dying?}>} o.ships - on screen, alive or dying
 * @param {Array<{key, target, count, eta, x, y, text}>} [o.waves] - head on screen
 * @param {string} [o.mode] - auto | minimal | full
 * @param {string|null} [o.selectedId]
 * @returns {{ships: Map<string, {kind, box, leader, group}>, groups: Array, waves: Array}}
 *   kind: 'card' | 'chip' | 'group' (inside a group chip) | 'none' (bracket only)
 */
export function layoutTactical({ view, panels = [], ships = [], waves = [], mode = DEFAULT_LABEL_MODE, selectedId = null }) {
  mode = normalizeLabelMode(mode);
  const sorted = [...ships].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const occupied = [...panels, ...sorted.map(shipBox)];
  // Chips sit on the outside flank, away from the other side's ships:
  // the space between the fleets (where the shots and torpedoes fly) stays clear
  const cx = {};
  for (const f of new Set(sorted.map(s => s.faction))) {
    const xs = sorted.filter(s => s.faction !== f);
    cx[f] = xs.length ? xs.reduce((a, s) => a + s.x, 0) / xs.length : null;
  }
  const sideOf = (s) => {
    const c = cx[s.faction];
    if (c === null || c === undefined || Math.abs(s.x - c) < 1) return s.faction === 'alpha' ? -1 : 1;
    return s.x < c ? -1 : 1;
  };

  const out = new Map();
  const cardFor = s => s.id === selectedId || !!s.dying || mode === 'full';
  const cards = sorted.filter(cardFor);
  const rest = sorted.filter(s => !cardFor(s));
  const chips = [], groups = [];
  if (mode === 'auto') {
    for (const c of clusterPoints(rest, CLUSTER_PX)) {
      if (c.length === 1) { chips.push(c[0]); continue; }
      const gx = c.reduce((a, s) => a + s.x, 0) / c.length, gy = c.reduce((a, s) => a + s.y, 0) / c.length;
      const r = Math.max(...c.map(s => Math.hypot(s.x - gx, s.y - gy) + Math.max(BRACKET_PX, s.r || 0)));
      const text = groupLabel(c);
      groups.push({ key: c.map(s => s.id).join('+'), faction: c[0].faction, ids: c.map(s => s.id),
        x: gx, y: gy, r, text, side: sideOf(c[0]), box: null, leader: null });
    }
  } else {
    for (const s of rest) out.set(s.id, { kind: 'none', box: null, leader: null, group: null });
  }

  // Placement priority: selected, dying, waves (soonest first), groups, single chips
  cards.sort((a, b) => Number(b.id === selectedId) - Number(a.id === selectedId) || Number(!!b.dying) - Number(!!a.dying));
  const place = (x, y, r, w, h, side, vertical = false) => {
    const box = placeChip(x, y, r, w, h, side, occupied, view, vertical);
    if (box) occupied.push(box);
    return box;
  };
  for (const s of cards) {
    const r = Math.max(BRACKET_PX, s.r || 0);
    const box = place(s.x, s.y, r, CARD.w, CARD.h, sideOf(s));
    out.set(s.id, { kind: box ? 'card' : 'none', box, leader: box ? leaderTo(s.x, s.y, r, box) : null, group: null });
  }
  const wavesOut = [];
  for (const w of waves) {
    const side = Math.abs(w.x - (w.tx ?? w.x)) < 1 ? 1 : (w.x < w.tx ? -1 : 1);   // trailing the head, away from its target
    const wb = { w: textWidth(w.text, 10), h: WAVE_H };
    const box = place(w.x, w.y, 5, wb.w, wb.h, side, true);
    wavesOut.push({ ...w, box, leader: box ? leaderTo(w.x, w.y, 5, box) : null });
  }
  for (const g of groups) {
    g.box = place(g.x, g.y, g.r, textWidth(g.text, 18), CHIP_H, g.side);
    g.leader = g.box ? leaderTo(g.x, g.y, g.r, g.box) : null;
    for (const id of g.ids) out.set(id, { kind: 'group', box: null, leader: null, group: g.key });
  }
  for (const s of chips) {
    const r = Math.max(BRACKET_PX, s.r || 0);
    const box = place(s.x, s.y, r, textWidth(s.short ?? shortName(s.name)), CHIP_H, sideOf(s));
    out.set(s.id, { kind: box ? 'chip' : 'none', box, leader: box ? leaderTo(s.x, s.y, r, box) : null, group: null });
  }
  return { ships: out, groups, waves: wavesOut };
}
