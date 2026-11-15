/**
 * Registry - the left-sidebar ship list for both fleets.
 *
 * Fleets of more than 8 hulls switch to compact single-line rows (and
 * denser ones past 16) so both fleets fit at 1280x720; lost ships sink to
 * the bottom of their fleet (CSS order, no DOM moves). Each row shows hull,
 * the HeatHud heat bar, a pip per fitting (guns, tubes, PD, drive: pass-5
 * recordings only, created on first use) and a status that counts a dying
 * hulk down to its reactor detonation or flags a ship DISARMED / ADRIFT /
 * NO AMMO. Per-frame updates write only values that changed.
 */
import { killStateAt } from './HudIndex.js';
import { getShipClassName, SHIP_SILHOUETTES } from './shipSilhouettes.js';
import { node, setText, defaultDoc } from './hudDom.js';
import { fittingsAt, fittingsFlag } from './FittingsIndex.js';
import { pipStrip } from './FittingsHud.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

// Row heights (px) of the three registry densities, plus the panel chrome
// (title bar, two fleet headers, padding) they share.
const ROW_PX = [['', 53], ['compact', 24], ['dense', 19]];
const CHROME_PX = 120;

/**
 * The roomiest density whose rows all fit in `available` px of sidebar:
 * '' (two-line rows), 'compact' (one line) or 'dense'. Past that the list
 * scrolls, with a fade hint.
 */
export function registryDensity(count, available = Infinity) {
  for (const [cls, row] of ROW_PX) if (count * row + CHROME_PX <= available) return cls;
  return 'dense';
}

/**
 * Registry status text for a ship at t. A living ship that can no longer
 * fight or move says so instead of its hull % (`fit` = its fitting state,
 * FittingsIndex.fittingsAt; null on recordings without fitting data): the
 * hull bar still shows the hull.
 */
export function registryStatus(state, kill, t, fit = null) {
  if (state?.destroyed) return { text: 'LOST', level: 'lost' };
  if (state?.dying) {
    const s = kill ? killStateAt(kill, t) : null;
    return { text: s?.in != null ? `☢ ${Math.ceil(s.in)}s` : 'CRIT', level: 'dying' };
  }
  const flag = fittingsFlag(fit);
  if (flag) return { text: flag, level: flag === 'NO AMMO' ? 'spent' : flag === 'IMPAIRED' ? 'impaired' : 'crippled' };
  const hull = Math.round(Math.max(0, Math.min(100, state?.hull ?? 100)));
  return { text: `${hull}%`, level: hull < 35 ? 'low' : 'ok' };
}

export class Registry {
  /**
   * @param {Object} o
   * @param {Element} o.root - #shipRegistry
   * @param {Element} o.alpha - #alphaShips
   * @param {Element} o.beta - #betaShips
   * @param {Function} o.onSelect - (shipId) => void
   */
  constructor({ root, alpha, beta, onSelect, doc = defaultDoc() } = {}) {
    this.root = root;
    this.lists = { alpha, beta };
    this.onSelect = onSelect;
    this.doc = doc;
    this.rows = new Map();   // shipId -> {li, bar, status, hull, text, cls}
    if (typeof window !== 'undefined') window.addEventListener('resize', () => this.relayout());
  }

  /** Pick the density that fits the sidebar (the combat log keeps its collapsed strip). */
  relayout() {
    const sidebar = this.root?.parentElement;
    const height = sidebar?.clientHeight || 0;
    const density = registryDensity(this.rows.size, height ? height - 48 : Infinity);
    this.root?.classList.toggle('compact', density !== '');
    this.root?.classList.toggle('dense', density === 'dense');
  }

  populate(ships) {
    const d = this.doc;
    this.rows.clear();
    this.lists.alpha?.replaceChildren();
    this.lists.beta?.replaceChildren();
    for (const [shipId, info] of Object.entries(ships || {})) {
      const li = node(d, 'li');
      const svg = d.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('class', 'registry-silhouette');
      svg.setAttribute('viewBox', '0 0 120 80');
      svg.setAttribute('aria-hidden', 'true');
      const path = d.createElementNS(SVG_NS, 'path');
      path.setAttribute('d', (SHIP_SILHOUETTES[info.type] || SHIP_SILHOUETTES.destroyer).body);
      svg.appendChild(path);
      const identity = node(d, 'span', null, 'registry-identity');
      const health = node(d, 'span', null, 'registry-health');
      const bar = node(d, 'i');
      health.append(bar);
      identity.append(node(d, 'span', info.name || shipId, 'registry-name'), node(d, 'span', getShipClassName(info.type), 'registry-class'), health);
      const status = node(d, 'span', null, 'registry-status');
      li.append(svg, identity, status);
      li.tabIndex = 0;
      li.setAttribute('role', 'button');
      li.setAttribute('aria-label', `Inspect ${info.name || shipId}`);
      li.title = `${info.name || shipId} · ${getShipClassName(info.type)}`;
      li.dataset.shipId = shipId;
      li.addEventListener('click', () => this.onSelect?.(shipId));
      li.addEventListener('keydown', e => {
        if (e.code === 'Enter' || e.code === 'Space') { e.preventDefault(); e.stopPropagation(); this.onSelect?.(shipId); }
      });
      (info.faction === 'alpha' ? this.lists.alpha : this.lists.beta)?.appendChild(li);
      this.rows.set(shipId, { li, bar, status, hull: -1, text: '', cls: '', fit: undefined, strip: null });
    }
    this.relayout();
  }

  /**
   * @param {Object} states - interpolated ship states
   * @param {number} t
   * @param {Object[]} kills - loader.hud.kills (dying countdowns)
   * @param {Object} [fittings] - loader.hud.fittings (pips and flags; inactive on older recordings)
   */
  update(states, t, kills = [], fittings = null) {
    for (const [id, row] of this.rows) {
      const state = states?.[id];
      const kill = state?.dying ? kills.find(k => k.shipId === id) : null;
      const fit = fittingsAt(fittings, id, t);
      if (row.fit !== fit) {
        row.fit = fit;
        if (fit && !row.strip) {
          // First fitting data for this ship (a live battle's first damage event): add the strip once
          row.strip = pipStrip(this.doc, 'fit-strip registry-fit');
          row.li.insertBefore(row.strip.el, row.status);
        }
        row.strip?.set(fit);
      }
      const { text, level } = registryStatus(state, kill, t, fit);
      const hull = state?.destroyed ? 0 : Math.round(Math.max(0, Math.min(100, state?.hull ?? 100)));
      if (row.hull !== hull) { row.hull = hull; row.bar.style.width = `${hull}%`; }
      if (row.text !== text) { row.text = text; setText(row.status, text); }
      if (row.cls !== level) {
        row.cls = level;
        row.li.classList.toggle('destroyed', level === 'lost');
        row.li.classList.toggle('dying', level === 'dying');
        row.li.classList.toggle('low', level === 'low');
        row.li.classList.toggle('crippled', level === 'crippled');
        row.li.classList.toggle('spent', level === 'spent');
        row.li.classList.toggle('impaired', level === 'impaired');
      }
    }
  }

  select(shipId) {
    for (const [id, row] of this.rows) row.li.classList.toggle('selected', id === shipId);
  }

  clear() {
    this.rows.clear();
    this.lists.alpha?.replaceChildren();
    this.lists.beta?.replaceChildren();
  }
}
