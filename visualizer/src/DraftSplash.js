/**
 * Draft splash (LV-12): the pre-battle draft as a broadcast graphic - both
 * fleets side by side with commander model, pick status and timer, points
 * spent, ship cards, the formation as a small plot, and both admirals'
 * reasoning once the sealed picks are revealed. Fed by the live block's
 * `draft` summary while the draft runs, and by recording.provenance.draft
 * once the battle is set up (the FLEETS chip reopens it from there).
 */
import { SHIP_SILHOUETTES, getShipClassName } from './shipSilhouettes.js';
import { formatClock } from './LiveSession.js';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const shortModel = m => (m ? String(m).split('/').pop() : '');
const baseType = t => (SHIP_SILHOUETTES[t] ? t : String(t || '').split('_')[0]);

/**
 * Normalise either source into the splash's per-faction shape.
 * @param {Object} entry - live draft entry ({ready, roster, ...}) or a
 *   provenance draft summary ({ships: [{ship_name, ship_type, cost, offset_km}], ...})
 */
export function normaliseDraftEntry(entry, faction, model) {
  if (!entry) return { faction, ready: false, roster: [], model };
  const fromSummary = Array.isArray(entry.ships);
  const roster = fromSummary
    ? entry.ships.map(s => ({ name: s.ship_name, type: s.ship_type, cost: s.cost, offset_km: s.offset_km }))
    : (entry.roster || []);
  return {
    faction,
    ready: fromSummary ? true : !!entry.ready,
    isMcp: !!entry.is_mcp,
    auto: !!(entry.auto || entry.auto_draft),
    model: entry.model || model,
    budget: entry.budget ?? null,
    pointsSpent: entry.points_spent ?? null,
    count: fromSummary ? roster.length : (entry.ships ?? roster.length),
    roster,
    formation: entry.formation_name || '',
    selectionRationale: entry.selection_rationale || null,
    formationRationale: entry.formation_rationale || null,
    elapsed: entry.elapsed_s ?? null,
    readyAfter: entry.ready_after_s ?? null,
  };
}

/** Top-down formation plot: +x (toward the enemy) points to the centre line. */
function formationPlot(roster, faction) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '-60 -60 120 120');
  svg.setAttribute('class', 'ds-plot');
  svg.setAttribute('aria-hidden', 'true');
  const pts = roster.map(s => s.offset_km || [0, 0, 0]);
  const extent = Math.max(40, ...pts.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]));
  const k = 48 / extent;
  // Alpha sits left of the centre line facing right, beta mirrored
  const dir = faction === 'alpha' ? 1 : -1;
  const axis = document.createElementNS(SVG_NS, 'path');
  axis.setAttribute('d', `M ${-52 * dir} 0 L ${52 * dir} 0 M ${46 * dir} -5 L ${52 * dir} 0 L ${46 * dir} 5`);
  axis.setAttribute('class', 'ds-axis');
  svg.appendChild(axis);
  roster.forEach((s, i) => {
    const p = pts[i];
    const dot = document.createElementNS(SVG_NS, 'circle');
    dot.setAttribute('cx', String(p[0] * k * dir));
    dot.setAttribute('cy', String(-p[1] * k));
    const size = { corvette: 2.6, frigate: 3, destroyer: 3.6, cruiser: 4.2, battlecruiser: 4.6, battleship: 5.2, dreadnought: 6 }[baseType(s.type)] || 3.6;
    dot.setAttribute('r', String(size));
    dot.setAttribute('class', 'ds-dot');
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = `${s.name} (${Math.round(p[0])}, ${Math.round(p[1])}, ${Math.round(p[2])}) km`;
    dot.appendChild(title);
    svg.appendChild(dot);
  });
  return svg;
}

function shipCard(s) {
  const li = el('li', 'ds-ship');
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 120 80');
  svg.setAttribute('class', 'ds-silhouette');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', (SHIP_SILHOUETTES[baseType(s.type)] || SHIP_SILHOUETTES.destroyer).body);
  svg.appendChild(path);
  li.append(svg, el('span', 'ds-ship-name', s.name || '—'), el('span', 'ds-ship-class', getShipClassName(s.type)),
    el('span', 'ds-ship-cost', s.cost != null ? `${s.cost}` : ''));
  return li;
}

export class DraftSplash {
  constructor({ onDismiss } = {}) {
    this.root = el('div', 'draft-splash');
    this.root.id = 'draftSplash';
    this.root.hidden = true;
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', 'Fleet draft');
    (document.getElementById('app') || document.body).appendChild(this.root);
    this.onDismiss = onDismiss;
    this.key = '';
    this.mode = null;
  }

  get visible() {
    return !this.root.hidden;
  }

  hide() {
    this.root.hidden = true;
    this.root.classList.remove('leaving');
    this.key = '';
  }

  /**
   * @param {Object} v
   * @param {'draft'|'starting'|'recap'} v.mode - live draft / battle just set up / reopened recap
   * @param {Object} v.alpha, v.beta - normaliseDraftEntry() output
   * @param {number} [v.elapsed] - draft wall seconds so far
   * @param {Object} [v.pending] - {alpha: ageS, beta: ageS} admiral calls in flight
   */
  show(v) {
    this.root.hidden = false;
    const tick = v.mode === 'draft' ? Math.floor(v.elapsed || 0) : 0;
    const key = JSON.stringify([v.mode, tick, v.alpha, v.beta, v.pending]);
    if (key === this.key) return;
    this.key = key;
    this.mode = v.mode;
    const budget = v.alpha.budget || v.beta.budget;
    const head = el('div', 'ds-head');
    head.append(el('span', 'ds-kicker', v.mode === 'draft' ? 'FLEET DRAFT' : v.mode === 'starting' ? 'FLEETS COMMITTED' : 'THE FLEETS'),
      el('span', 'ds-budget', `${budget ? `${budget} POINTS · ` : ''}SEALED PICKS`),
      el('span', 'ds-clock', v.mode === 'draft' ? formatClock(v.elapsed) : v.mode === 'starting' ? 'BATTLE STARTING' : ''));
    const cols = el('div', 'ds-cols');
    cols.append(this.side(v.alpha, v), el('div', 'ds-vs', 'VS'), this.side(v.beta, v));
    const foot = el('div', 'ds-foot');
    const both = v.alpha.ready && v.beta.ready;
    foot.append(el('span', null, both
      ? 'Both fleets committed. Each admiral picked blind; neither has seen the other\'s fleet.'
      : 'Each admiral drafts blind from the same budget. Fleets show as they commit; reasoning when both are in.'));
    if (v.mode !== 'draft') {
      const go = el('button', 'ds-dismiss', v.mode === 'starting' ? 'WATCH THE BATTLE' : 'CLOSE');
      go.addEventListener('click', () => { this.hide(); this.onDismiss?.(); });
      foot.append(go);
    }
    this.root.replaceChildren(head, cols, foot);
  }

  side(d, v) {
    const side = el('section', `ds-side ${d.faction}`);
    const header = el('header');
    header.append(el('span', 'ds-faction', d.faction.toUpperCase()), el('span', 'ds-model', shortModel(d.model) || 'commander'));
    let pill;
    if (d.ready) {
      pill = el('span', 'ds-pill ready', `READY ✓${d.readyAfter != null ? ` ${formatClock(d.readyAfter)}` : ''}${d.auto ? ' · AUTO' : ''}`);
    } else if (d.isMcp) {
      pill = el('span', 'ds-pill mcp', 'MCP PICKING');
    } else {
      const age = v.pending?.[d.faction];
      pill = el('span', 'ds-pill picking', `PICKING ⟳ ${formatClock(age ?? v.elapsed ?? 0)}`);
    }
    header.append(pill);
    side.append(header);

    const points = el('div', 'ds-points');
    const bar = el('div', 'ds-bar');
    const fill = el('i');
    const spent = d.pointsSpent || 0;
    fill.style.width = d.budget ? `${Math.min(100, 100 * spent / d.budget)}%` : '0%';
    bar.append(fill);
    points.append(bar, el('span', null, d.ready || spent ? `${spent}${d.budget ? ` / ${d.budget}` : ''} PTS · ${d.count} HULLS` : 'reviewing the catalog…'));
    side.append(points);

    if (d.roster.length) {
      const list = el('ul', 'ds-ships');
      for (const s of d.roster) list.append(shipCard(s));
      side.append(list);
      const formation = el('div', 'ds-formation');
      formation.append(formationPlot(d.roster, d.faction));
      const label = el('div', 'ds-formation-label');
      label.append(el('span', 'ds-label', 'FORMATION'), el('span', 'ds-formation-name', d.formation || '—'));
      formation.append(label);
      side.append(formation);
    } else {
      side.append(el('div', 'ds-sealed', d.ready ? 'FLEET SEALED' : 'Awaiting picks'));
    }
    if (d.selectionRationale || d.formationRationale) {
      const why = el('div', 'ds-rationale');
      if (d.selectionRationale) why.append(el('h4', null, 'WHY THESE SHIPS'), el('p', null, d.selectionRationale));
      if (d.formationRationale) why.append(el('h4', null, 'WHY THIS FORMATION'), el('p', null, d.formationRationale));
      side.append(why);
    }
    return side;
  }
}
