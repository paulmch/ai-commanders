/**
 * Trajectory-track display mode: off -> predicted -> predicted + actual.
 * Hotkey T cycles, the playback-bar button shows and cycles the state, and
 * the choice persists in localStorage. The "actual" (recorded future) track
 * only exists in replay: live mode skips it.
 */

export const TRACK_MODES = ['off', 'predicted', 'actual'];
export const TRACK_MODE_KEY = 'aic.viewer.trackMode';
export const DEFAULT_TRACK_MODE = 'predicted';

const LABELS = { off: 'OFF', predicted: 'PRED', actual: 'PRED+ACT' };
const TITLES = {
  off: 'Trajectory tracks off (T)',
  predicted: 'Predicted track to the next checkpoint (T)',
  actual: 'Predicted vs actual track to the next checkpoint (T)'
};

export function nextTrackMode(mode, { allowActual = true } = {}) {
  const i = TRACK_MODES.indexOf(mode);
  let next = TRACK_MODES[(i + 1) % TRACK_MODES.length];
  if (next === 'actual' && !allowActual) next = 'off';
  return next;
}

export function loadTrackMode(storage, fallback = DEFAULT_TRACK_MODE) {
  try {
    const v = storage?.getItem(TRACK_MODE_KEY);
    return TRACK_MODES.includes(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

export function saveTrackMode(storage, mode) {
  try {
    storage?.setItem(TRACK_MODE_KEY, mode);
  } catch {
    // private window / blocked storage: the mode just won't persist
  }
}

function defaultStorage() {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null;
  } catch {
    return null;
  }
}

export class TrackModeControl {
  /**
   * @param {Object} o
   * @param {HTMLElement} [o.button] - playback-bar toggle
   * @param {Storage} [o.storage] - defaults to window.localStorage
   * @param {Function} [o.isLive] - () => true while a live battle runs
   * @param {Function} [o.onChange] - (mode) => void
   * @param {EventTarget} [o.keyTarget] - where to listen for T (document)
   */
  constructor({ button = null, storage = defaultStorage(), isLive = () => false, onChange = null, keyTarget = null } = {}) {
    this.button = button;
    this.storage = storage;
    this.isLive = isLive;
    this.onChange = onChange;
    this.mode = loadTrackMode(storage);
    button?.addEventListener('click', () => this.cycle());
    const target = keyTarget || (typeof document !== 'undefined' ? document : null);
    target?.addEventListener('keydown', (e) => {
      if (e.code !== 'KeyT' || e.repeat || e.altKey || e.ctrlKey || e.metaKey) return;
      const el = e.target;
      if (el && (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) || el.isContentEditable)) return;
      this.cycle();
    });
    this.render();
  }

  /** The mode to draw this frame: actual degrades to predicted while live. */
  effectiveMode() {
    const mode = this.mode === 'actual' && this.isLive() ? 'predicted' : this.mode;
    if (mode !== this._shown) this.render(mode);   // e.g. a live battle just ended
    return mode;
  }

  cycle() {
    const current = this.effectiveMode();
    this.set(nextTrackMode(current, { allowActual: !this.isLive() }));
  }

  set(mode) {
    if (!TRACK_MODES.includes(mode)) return;
    this.mode = mode;
    saveTrackMode(this.storage, mode);
    this.render();
    this.onChange?.(mode);
  }

  render(mode = null) {
    mode = mode || (this.mode === 'actual' && this.isLive() ? 'predicted' : this.mode);
    this._shown = mode;
    const b = this.button;
    if (!b) return;
    b.dataset.mode = mode;
    b.classList.toggle('active', mode !== 'off');
    b.setAttribute('aria-pressed', String(mode !== 'off'));
    b.title = TITLES[mode];
    const state = b.querySelector('[data-track-state]');
    if (state) state.textContent = LABELS[mode];
  }
}
