/**
 * TrackHud - the one place the trajectory-track countdown lives, plus the
 * glyph legend and the track scope (every ship / the focused ship only).
 *
 * Per-marker "T+Ns" captions on every ship smeared together at fleet range
 * (they are all the same number): the checkpoint countdown is shown once
 * here, and only the focused ship's marker keeps a caption (with its true
 * spread, see Tracks.js). Shift+T or the ALL/SEL button switches the scope;
 * both the scope and the legend's open state persist in localStorage.
 */
import { loadTrackScope, saveTrackScope, nextTrackScope, secondsToGo } from './TrackDisplay.js';

const LEGEND_KEY = 'aic.viewer.trackLegend';

function defaultStorage() {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** "×61" or "×58–80" for the hulls' display exaggeration range. */
export function formatExaggeration([lo, hi] = [0, 0]) {
  if (!(hi > 0)) return '×60';
  const a = Math.round(lo), b = Math.round(hi);
  return a === b || !(lo > 0) ? `×${b}` : `×${a}–${b}`;
}

const svg = (body, cls = '') => `<svg class="track-glyph ${cls}" viewBox="0 0 22 14" aria-hidden="true">${body}</svg>`;
const DIAMOND = 'M11 2.2 L15.8 7 L11 11.8 L6.2 7 Z';

const LEGEND = [
  [svg(`<path d="${DIAMOND}"/><circle class="fill" cx="11" cy="7" r="1.1"/>`), 'predicted position at checkpoint'],
  [svg('<path d="M1 7 H21" stroke-dasharray="3.2 2.4"/><path d="M11 3.2 V10.8"/>'), 'predicted track, tick = 5 s'],
  [svg('<path class="dots" d="M1.5 7 H13"/><circle cx="17.5" cy="7" r="2.8"/>', 'pale'), 'actual track / position (PRED+ACT)'],
  [svg('<circle cx="11" cy="7" r="3.1"/><circle cx="11" cy="7" r="6" stroke-dasharray="1.3 1.3" opacity=".6"/>'),
    '1σ / 2σ spread <b data-exag>×60</b> (hull scale)'],
  [svg('<path class="dots" d="M2 11.5 C5 5, 7 12, 10 7 S15 3.5, 17 6"/><path d="M19.6 3.8 L21.6 5.8 L19.6 7.8 L17.6 5.8 Z"/>', 'pale'),
    'drift of earlier predictions <b data-exag>×60</b>'],
  [svg(`<path d="${DIAMOND}" stroke-dasharray="2.6 1.7"/>`, 'stale'), 'orders pending (provisional)'],
  [svg('<circle cx="11" cy="7" r="5.4" stroke-dasharray="3.2 2.2"/><path d="M11 5.4 L12.6 7 L11 8.6 L9.4 7 Z"/>'),
    'reactive evasion: low confidence']
];

export class TrackHud {
  /**
   * @param {Object} o
   * @param {HTMLElement} [o.root] - HUD container (#trackHud)
   * @param {Storage} [o.storage] - defaults to window.localStorage
   * @param {EventTarget} [o.keyTarget] - where to listen for Shift+T (document)
   * @param {Function} [o.onScope] - (scope) => void
   */
  constructor({ root = null, storage = defaultStorage(), keyTarget = null, onScope = null } = {}) {
    this.root = root;
    this.storage = storage;
    this.onScope = onScope;
    this.scope = loadTrackScope(storage);
    let open = true;
    try { open = storage?.getItem(LEGEND_KEY) !== '0'; } catch { /* default open */ }
    this.legendOpen = open;
    this.shown = null;
    this._togo = null;
    this._hint = null;
    this._exag = null;
    if (root) this._build();
    const target = keyTarget || (typeof document !== 'undefined' ? document : null);
    target?.addEventListener('keydown', (e) => {
      if (e.code !== 'KeyT' || !e.shiftKey || e.repeat || e.altKey || e.ctrlKey || e.metaKey) return;
      const el = e.target;
      if (el && (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) || el.isContentEditable)) return;
      this.toggleScope();
    });
  }

  _build() {
    const r = this.root;
    r.title = 'Trajectory tracks to the next decision checkpoint. Spread rings and the drift of the '
      + 'prediction are magnified like the hulls (softly capped at four hull lengths); the focused '
      + 'ship’s caption gives the true spread.';
    r.innerHTML = `
      <div class="track-hud-head">
        ${svg(`<path d="${DIAMOND}"/>`, 'head')}
        <span class="track-hud-title">NEXT CHECKPOINT</span>
        <span class="track-hud-togo" data-togo>--</span>
        <button type="button" class="track-hud-btn" data-scope title="Tracks for every ship or the selected ship only (Shift+T)">ALL</button>
        <button type="button" class="track-hud-btn" data-legend-btn aria-expanded="true" title="Track legend">KEY</button>
      </div>
      <div class="track-hud-hint" data-hint hidden></div>
      <ul class="track-legend" data-legend>
        ${LEGEND.map(([glyph, text]) => `<li>${glyph}<span>${text}</span></li>`).join('')}
      </ul>`;
    this.els = {
      togo: r.querySelector('[data-togo]'),
      scope: r.querySelector('[data-scope]'),
      legendBtn: r.querySelector('[data-legend-btn]'),
      legend: r.querySelector('[data-legend]'),
      hint: r.querySelector('[data-hint]'),
      exag: [...r.querySelectorAll('[data-exag]')]
    };
    this.els.scope.addEventListener('click', () => this.toggleScope());
    this.els.legendBtn.addEventListener('click', () => this.setLegend(!this.legendOpen));
    this.setLegend(this.legendOpen, false);
    this._renderScope();
  }

  toggleScope() {
    this.setScope(nextTrackScope(this.scope));
  }

  setScope(scope) {
    this.scope = scope === 'focus' ? 'focus' : 'all';
    saveTrackScope(this.storage, this.scope);
    this._renderScope();
    this.onScope?.(this.scope);
  }

  setLegend(open, persist = true) {
    this.legendOpen = !!open;
    if (persist) {
      try { this.storage?.setItem(LEGEND_KEY, open ? '1' : '0'); } catch { /* not persisted */ }
    }
    if (!this.els) return;
    this.els.legend.hidden = !open;
    this.els.legendBtn.setAttribute('aria-expanded', String(!!open));
    this.els.legendBtn.classList.toggle('active', !!open);
  }

  _renderScope() {
    const b = this.els?.scope;
    if (!b) return;
    const sel = this.scope === 'focus';
    b.textContent = sel ? 'SEL' : 'ALL';
    b.classList.toggle('active', sel);
    b.setAttribute('aria-pressed', String(sel));
  }

  /**
   * @param {Object} h - {visible, togo, stale, scope, focusId, exaggeration: [lo, hi]}
   */
  update(h) {
    if (!this.root) return;
    this._show(h.visible !== false);
    const togo = `${secondsToGo(h.togo)}s`;
    if (togo !== this._togo) { this._togo = togo; this.els.togo.textContent = togo; }
    const hint = this.scope === 'focus' && !h.focusId ? 'SELECT A SHIP · SHIFT+T SHOWS ALL'
      : h.stale ? 'ORDERS PENDING · PROVISIONAL' : '';
    if (hint !== this._hint) {
      this._hint = hint;
      this.els.hint.textContent = hint;
      this.els.hint.hidden = !hint;
    }
    const exag = formatExaggeration(h.exaggeration);
    if (exag !== this._exag) {
      this._exag = exag;
      for (const el of this.els.exag) el.textContent = exag;
    }
  }

  hide() {
    this._show(false);
  }

  _show(on) {
    if (!this.root || this.shown === on) return;
    this.shown = on;
    this.root.classList.toggle('hidden', !on);
  }
}
