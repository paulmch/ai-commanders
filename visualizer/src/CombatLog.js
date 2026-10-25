/**
 * CombatLog - the left-sidebar event log as a pure function of the playhead.
 *
 * Rows come pre-formatted from HudIndex (loader.hud.log, time-sorted, keyed
 * by event id). Each update binary-searches how many rows lie at or before
 * the playhead and re-renders only when that count, the filter or the index
 * changes; a parked playhead (paused replay, live head parked on a
 * checkpoint, the deciding pause) costs one comparison per frame and can
 * never insert a row twice. Seeking backwards drops rows newer than the
 * playhead; events that arrive late at the parked time (live checkpoint
 * orders) still appear, because the index - not a time window - decides.
 */
import { upperBound, clockText, LOG_FILTERS } from './HudIndex.js';
import { node, defaultDoc, syncNewestFirst } from './hudDom.js';

export class CombatLog {
  /**
   * @param {Object} o
   * @param {Element} o.list - #eventList
   * @param {Element} [o.chips] - container of [data-log-filter] buttons
   * @param {Function} [o.onSeek] - (row) => void when a row is clicked
   * @param {number} [o.max] - rows kept in the DOM
   */
  constructor({ list, chips = null, onSeek = null, max = 50, doc = defaultDoc() } = {}) {
    this.list = list;
    this.doc = doc;
    this.max = max;
    this.onSeek = onSeek;
    this.filter = 'all';
    this.hud = null;
    this.filtered = new Map();   // filter -> rows (per hud index)
    this.shown = [];
    this.key = '';
    this.chips = chips;
    chips?.querySelectorAll?.('[data-log-filter]').forEach(b => b.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setFilter(b.dataset.logFilter);
    }));
    this._renderChips();
  }

  /** New or rebuilt index (load, live append). */
  setIndex(hud) {
    if (hud === this.hud) return;
    this.hud = hud;
    this.filtered.clear();
    this.key = '';
  }

  setFilter(filter) {
    this.filter = LOG_FILTERS.includes(filter) ? filter : 'all';
    this.key = '';
    this._renderChips();
  }

  rows() {
    const all = this.hud?.log || [];
    if (this.filter === 'all') return all;
    let rows = this.filtered.get(this.filter);
    if (!rows) {
      rows = all.filter(r => r.cat === this.filter);
      this.filtered.set(this.filter, rows);
    }
    return rows;
  }

  /** Bring the list to playhead t. Returns 'same' | 'append' | 'rebuild'. */
  update(t) {
    if (!this.list || !this.hud) return 'same';
    const rows = this.rows();
    const n = upperBound(rows, t);
    const key = `${this.filter}:${n}:${rows.length}`;
    if (key === this.key) return 'same';
    this.key = key;
    const desired = rows.slice(Math.max(0, n - this.max), n);
    return syncNewestFirst(this.list, this.shown, desired, (row, fresh) => this.row(row, fresh));
  }

  row(r, fresh) {
    const item = node(this.doc, 'div', null, `event-item ${r.cls}${fresh ? '' : ' quiet'}`);
    item.dataset.eventId = r.id;
    item.append(node(this.doc, 'span', clockText(r.t), 'event-time'), node(this.doc, 'span', r.text, 'event-text'));
    if (this.onSeek) item.addEventListener('click', () => this.onSeek(r));
    return item;
  }

  clear() {
    this.hud = null;
    this.filtered.clear();
    this.shown.length = 0;
    this.key = '';
    this.list?.replaceChildren();
  }

  _renderChips() {
    this.chips?.querySelectorAll?.('[data-log-filter]').forEach(b => {
      const on = b.dataset.logFilter === this.filter;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
  }
}
