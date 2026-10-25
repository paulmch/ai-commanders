/**
 * CommsFeed - the COMMAND NET panel: everything the commanders said, to
 * whom, and when, up to the playhead.
 *
 * Rows come from loader.hud.comms (HudIndex.buildComms): admiral plans,
 * directives and per-ship orders (priority, salvo chips), captain reports
 * (ACK / DEVIATE / UNABLE with reason and the engine's compliance check),
 * captain logs and queries, fleet-net signals, direct messages, engine
 * auto-signals, enemy transmissions and system rows. Legacy recordings
 * (message / acknowledgment-only) render through the same rows.
 *
 * Grouped by checkpoint, newest checkpoint on top, chronological inside a
 * checkpoint; the latest directive of each side stays pinned above the list.
 * Rendering is a pure function of (playhead, filter, index): an unchanged
 * count costs one comparison per frame, forward play appends only the new
 * rows, and anything else (seek back, filter, re-sorted live data) rebuilds
 * once. All recorded text is written with textContent.
 */
import { upperBound, clockText, commsMatches, deliveryAt } from './HudIndex.js';
import { node, setText, defaultDoc } from './hudDom.js';

const STORE_KEY = 'aic.viewer.commsFilter';
const GROUPS = ['all', 'command', 'reports', 'chatter', 'issues'];
const FACTIONS = ['all', 'alpha', 'beta'];

function storage() {
  try { return typeof window !== 'undefined' ? window.localStorage : null; } catch { return null; }
}

/** Visible rows at t for a filter: the last `max` matching rows with row.t <= t (oldest first). */
export function visibleComms(rows, t, faction = 'all', group = 'all', max = 150) {
  const n = upperBound(rows || [], t);
  const out = [];
  for (let i = n - 1; i >= 0 && out.length < max; i--) {
    if (commsMatches(rows[i], faction, group)) out.push(rows[i]);
  }
  return out.reverse();
}

/** Latest directive (and plan) per faction at t: {alpha: {directive, plan}, beta: {...}}. */
export function standingOrders(rows, t) {
  const out = { alpha: { directive: null, plan: null }, beta: { directive: null, plan: null } };
  const n = upperBound(rows || [], t);
  for (let i = n - 1; i >= 0; i--) {
    const r = rows[i];
    const slot = out[r.faction];
    if (!slot) continue;
    if (r.kind === 'directive' && !slot.directive) slot.directive = r;
    else if (r.kind === 'plan' && !slot.plan) slot.plan = r;
    if (out.alpha.directive && out.alpha.plan && out.beta.directive && out.beta.plan) break;
  }
  return out;
}

const DELIVERY_TEXT = {
  transit: d => d.at != null ? `IN TRANSIT · due ${clockText(d.at)}` : 'IN TRANSIT',
  partial: d => `DELIVERED ${d.done}/${d.total}`,
  delivered: d => `✓ DELIVERED ${clockText(d.at)}`,
  undelivered: () => '✕ NEVER DELIVERED'
};

export class CommsFeed {
  /**
   * @param {Object} o
   * @param {Element} o.root - #commsFeed
   * @param {Function} [o.onSeek] - (row) => void: seek to the row and frame its ship
   * @param {Function} [o.onWhy] - (row) => void: open the decision behind the row
   * @param {Function} [o.onOpenChange] - (open) => void
   * @param {Function} [o.resolveImage] - (vision) => data URL | null
   * @param {Function} [o.onImage] - (src, caption) => void (lightbox)
   */
  constructor({ root, onSeek = null, onWhy = null, onOpenChange = null, resolveImage = null, onImage = null,
    doc = defaultDoc(), store = storage() } = {}) {
    this.root = root;
    this.doc = doc;
    this.store = store;
    this.onSeek = onSeek;
    this.onWhy = onWhy;
    this.onOpenChange = onOpenChange;
    this.resolveImage = resolveImage;
    this.onImage = onImage;
    this.list = root?.querySelector('[data-list]') || null;
    this.standing = root?.querySelector('[data-standing]') || null;
    this.faction = 'all';
    this.group = 'all';
    try {
      const saved = JSON.parse(store?.getItem(STORE_KEY) || 'null');
      if (FACTIONS.includes(saved?.faction)) this.faction = saved.faction;
      if (GROUPS.includes(saved?.group)) this.group = saved.group;
    } catch { /* defaults */ }
    this.open = false;
    this.hud = null;
    this.complete = true;
    this.shown = [];               // row ids in the DOM, oldest first
    this.rowEls = new Map();       // id -> element
    this.groupEls = new Map();     // cp -> {section, count}
    this.deliveryRows = new Map(); // id -> {row, el}
    this.key = '';
    this.deliveryKey = '';
    this.standingKey = '';
    root?.querySelectorAll('[data-comms-faction]').forEach(b => b.addEventListener('click', () => this.setFilter(b.dataset.commsFaction, this.group)));
    root?.querySelectorAll('[data-comms-group]').forEach(b => b.addEventListener('click', () => this.setFilter(this.faction, b.dataset.commsGroup)));
    root?.querySelector('[data-close]')?.addEventListener('click', () => this.setOpen(false, true));
    this._renderFilters();
  }

  /** @param {boolean} user - the viewer toggled it (persisted), vs. the per-recording default */
  setOpen(open, user = false) {
    this.open = !!open;
    this.root?.classList.toggle('hidden', !this.open);
    this.key = '';
    this.onOpenChange?.(this.open, user);
  }

  /** New or rebuilt index. complete = no more data will arrive (replay or ended live battle). */
  setIndex(hud, complete = true) {
    if (hud === this.hud && complete === this.complete) return;
    this.hud = hud;
    this.complete = complete;
    this.key = '';
    this.deliveryKey = '';
    this.standingKey = '';
    this.deliveryTimes = null;
  }

  setFilter(faction, group) {
    this.faction = FACTIONS.includes(faction) ? faction : 'all';
    this.group = GROUPS.includes(group) ? group : 'all';
    try { this.store?.setItem(STORE_KEY, JSON.stringify({ faction: this.faction, group: this.group })); } catch { /* not persisted */ }
    this.key = '';
    this.standingKey = '';
    this._renderFilters();
  }

  /** Bring the panel to playhead t. Returns 'same' | 'append' | 'rebuild' | 'closed'. */
  update(t) {
    if (!this.open || !this.list || !this.hud) return 'closed';
    const rows = this.hud.comms || [];
    const n = upperBound(rows, t);
    const key = `${this.faction}:${this.group}:${n}:${rows.length}`;
    let result = 'same';
    if (key !== this.key) {
      this.key = key;
      result = this._sync(visibleComms(rows, t, this.faction, this.group));
      this._renderStanding(t);
    }
    this._updateDeliveries(t);
    return result;
  }

  _sync(desired) {
    const shown = this.shown;
    const same = shown.length === desired.length && shown.every((id, i) => id === desired[i].id);
    if (same) return 'same';
    const last = shown.length ? shown[shown.length - 1] : null;
    const q = last == null ? -1 : desired.findIndex(r => r.id === last);
    if (q >= 0) {
      const drop = shown.length - (q + 1);
      let prefix = drop >= 0;
      for (let i = 0; prefix && i <= q; i++) prefix = shown[drop + i] === desired[i].id;
      // New rows must land in the newest group (or open a newer one)
      const topCp = this._topCp();
      for (let i = q + 1; prefix && i < desired.length; i++) prefix = topCp == null || desired[i].cp >= topCp;
      if (prefix) {
        for (const id of shown.splice(0, drop)) this._removeRow(id);
        for (let i = q + 1; i < desired.length; i++) this._appendRow(desired[i], true);
        return 'append';
      }
    }
    this.list.replaceChildren();
    this.rowEls.clear();
    this.groupEls.clear();
    this.deliveryRows.clear();
    shown.length = 0;
    for (const r of desired) this._appendRow(r, false);
    if (!desired.length) {
      this.list.append(node(this.doc, 'p', this.hud?.comms?.length ? 'No traffic on this filter yet.' : 'No command traffic recorded yet.', 'comms-empty'));
    }
    return 'rebuild';
  }

  _topCp() {
    let top = null;
    for (const cp of this.groupEls.keys()) if (top == null || cp > top) top = cp;
    return top;
  }

  _group(cp, t) {
    let g = this.groupEls.get(cp);
    if (g) return g;
    const d = this.doc;
    const section = node(d, 'section', null, 'comms-group');
    section.dataset.cp = String(cp);
    section.append(node(d, 'h4', cp > 0 ? `CP${cp} · T+${clockText(t)}` : `PRE-BATTLE · T+${clockText(t)}`, 'comms-group-head'));
    // Newest checkpoint on top: insert before the first older group
    const empty = this.list.querySelector('.comms-empty');
    if (empty) empty.remove();
    let before = null;
    for (const [other, og] of this.groupEls) {
      if (other < cp && (!before || other > Number(before.dataset.cp))) before = og.section;
    }
    this.list.insertBefore(section, before);
    g = { section, count: 0 };
    this.groupEls.set(cp, g);
    return g;
  }

  _appendRow(r, fresh) {
    const g = this._group(r.cp, r.t);
    const el = this._row(r, fresh);
    g.section.append(el);
    g.count++;
    this.rowEls.set(r.id, el);
    this.shown.push(r.id);
  }

  _removeRow(id) {
    const el = this.rowEls.get(id);
    this.rowEls.delete(id);
    this.deliveryRows.delete(id);
    if (!el) return;
    const section = el.parentNode;
    el.remove();
    const cp = Number(section?.dataset?.cp);
    const g = this.groupEls.get(cp);
    if (g && --g.count <= 0) { g.section.remove(); this.groupEls.delete(cp); }
  }

  _row(r, fresh) {
    const d = this.doc;
    const cls = ['comms-row', `kind-${r.kind}`, r.faction || 'neutral'];
    if (r.status && r.status !== 'ACK') cls.push(`status-${r.status.toLowerCase()}`);
    if (r.priority === 'CRITICAL') cls.push('prio-critical');
    if (fresh) cls.push('fresh');
    const el = node(d, 'article', null, cls.join(' '));
    el.dataset.id = r.id;
    const meta = node(d, 'div', null, 'comms-meta');
    const time = node(d, 'button', clockText(r.t), 'comms-time');
    time.type = 'button';
    time.title = 'Jump to this moment';
    time.addEventListener('click', (e) => { e.stopPropagation?.(); this.onSeek?.(r); });
    const tag = node(d, 'span', r.tag, 'comms-tag');
    const route = node(d, 'span', null, 'comms-route');
    route.append(node(d, 'b', r.from || '—'));
    if (r.to) route.append(node(d, 'span', ` → ${r.to}`));
    meta.append(time, tag, route);
    if (r.priority && r.priority !== 'NORMAL') meta.append(node(d, 'span', r.priority, 'comms-priority'));
    el.append(meta);
    if (r.text) el.append(node(d, 'div', r.text, 'comms-text'));
    if (r.reply) el.append(node(d, 'div', r.reply, 'comms-reply'));
    if (r.chips?.length || r.msgId || r.decisionId) {
      const chips = node(d, 'div', null, 'comms-chips');
      for (const c of r.chips || []) chips.append(node(d, 'span', c, 'comms-chip'));
      if (r.msgId) {
        const dl = node(d, 'span', '', 'comms-chip comms-delivery');
        chips.append(dl);
        this.deliveryRows.set(r.id, { row: r, el: dl, text: '' });
      }
      if (r.decisionId && this.onWhy) {
        const why = node(d, 'button', 'WHY ›', 'comms-why');
        why.type = 'button';
        why.title = 'Open the captain decision behind this message (D panel)';
        why.addEventListener('click', (e) => { e.stopPropagation?.(); this.onWhy(r); });
        chips.append(why);
      }
      el.append(chips);
    }
    if (r.kind === 'directive') this._vision(el, r);
    el.title = r.text && r.text.length > 180 ? 'Click to expand' : '';
    el.addEventListener('click', () => el.classList.toggle('expanded'));
    return el;
  }

  _vision(el, r) {
    const v = (this.hud?.visions || []).find(x => x.faction === r.faction && Math.abs(x.t - r.t) < 1e-6);
    const src = v && this.resolveImage?.(v);
    if (!src) return;
    const d = this.doc;
    const fig = node(d, 'button', null, 'comms-vision');
    fig.type = 'button';
    const img = d.createElement('img');
    img.alt = `Tactical plot the ${r.faction} admiral saw at CP${r.cp}`;
    img.loading = 'lazy';
    img.src = src;
    fig.append(img, node(d, 'span', `ADMIRAL VIEW · CP${r.cp}`));
    fig.addEventListener('click', (e) => { e.stopPropagation?.(); this.onImage?.(src, img.alt); });
    el.append(fig);
  }

  _updateDeliveries(t) {
    if (!this.deliveryRows.size) return;
    if (!this.deliveryTimes) {
      const times = [];
      for (const list of this.hud.deliveries?.values?.() || []) for (const x of list) times.push(x.t);
      for (const r of this.hud.comms || []) if (r.deliverAt != null) times.push(r.deliverAt);
      this.deliveryTimes = times.sort((a, b) => a - b);
    }
    const key = `${upperBound(this.deliveryTimes, t, x => x)}:${this.deliveryRows.size}:${this.complete}:${this.key}`;
    if (key === this.deliveryKey) return;
    this.deliveryKey = key;
    for (const entry of this.deliveryRows.values()) {
      const state = deliveryAt(entry.row, this.hud.deliveries, t, this.complete);
      const text = state ? DELIVERY_TEXT[state.state](state) : '';
      if (text !== entry.text) {
        entry.text = text;
        setText(entry.el, text);
        entry.el.dataset.state = state?.state || '';
        entry.el.hidden = !text;
      }
    }
  }

  _renderStanding(t) {
    if (!this.standing) return;
    const orders = standingOrders(this.hud.comms, t);
    const sides = this.faction === 'all' ? ['alpha', 'beta'] : [this.faction];
    const key = sides.map(f => `${orders[f].directive?.id}:${orders[f].plan?.id}`).join('|');
    if (key === this.standingKey) return;
    this.standingKey = key;
    const d = this.doc;
    const lines = [];
    for (const f of sides) {
      const { directive, plan } = orders[f];
      if (!directive && !plan) continue;
      const r = directive || plan;
      const line = node(d, 'button', null, `comms-standing-line ${f}`);
      line.type = 'button';
      line.title = plan && directive ? `STANDING PLAN (CP${plan.cp}): ${plan.text}` : r.text;
      line.append(node(d, 'span', `${f.toUpperCase()} · CP${r.cp}`, 'comms-standing-tag'), node(d, 'span', r.text, 'comms-standing-text'));
      line.addEventListener('click', () => line.classList.toggle('expanded'));
      lines.push(line);
    }
    this.standing.replaceChildren(...lines);
    this.standing.hidden = lines.length === 0;
  }

  _renderFilters() {
    this.root?.querySelectorAll('[data-comms-faction]').forEach(b => {
      const on = b.dataset.commsFaction === this.faction;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
    this.root?.querySelectorAll('[data-comms-group]').forEach(b => {
      const on = b.dataset.commsGroup === this.group;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
  }

  clear() {
    this.hud = null;
    this.key = '';
    this.standingKey = '';
    this.deliveryKey = '';
    this.shown.length = 0;
    this.rowEls.clear();
    this.groupEls.clear();
    this.deliveryRows.clear();
    this.list?.replaceChildren();
    this.standing?.replaceChildren();
  }
}
