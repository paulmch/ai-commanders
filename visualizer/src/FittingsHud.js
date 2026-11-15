/**
 * FittingsHud - DOM for a ship's fitting state (FittingsIndex.js): the pip
 * strip of the registry rows and the FITTINGS block of the selected-ship
 * telemetry card. One pip per gun mount (bar), torpedo tube (wedge), PD
 * turret (dot) and the drive (chevron): intact / damaged / destroyed.
 *
 * Both redraw only when the state object changes (fittingsAt returns the
 * same object until the next recorded change), so a frame costs one
 * identity compare per ship.
 */
import { fittingsText, fittingsFlag } from './FittingsIndex.js';
import { node, setText, defaultDoc } from './hudDom.js';

/** Drive pip state for a thrust fraction. */
export const driveState = st => (st.adrift ? 'destroyed' : st.thrust < 0.995 ? 'damaged' : 'ok');

/**
 * Pip descriptors of a fitting state, in display order: [{kind: 'gun'|'tube'|'pd'|'drive', s}]
 * with s = 'ok' | 'damaged' | 'destroyed' (| 'empty': an intact tube with no rounds left).
 * `only` ('guns'|'tubes'|'pd'|'drive') keeps one group.
 */
export function pipsOf(st, only = null) {
  if (!st) return [];
  const want = k => !only || only === k;
  return [
    ...(want('guns') ? st.guns.map(s => ({ kind: 'gun', s })) : []),
    ...(want('tubes') ? st.tubes.map(s => ({ kind: 'tube', s: s !== 'destroyed' && st.rounds === 0 ? 'empty' : s })) : []),
    ...(want('pd') ? st.pd.map(s => ({ kind: 'pd', s })) : []),
    ...(want('drive') ? [{ kind: 'drive', s: driveState(st) }] : [])
  ];
}

/**
 * A pip strip that keeps itself in sync with a fitting state.
 * @returns {{el: Element, set: (st) => void}}
 */
export function pipStrip(doc, cls = 'fit-strip', only = null) {
  const el = node(doc, 'span', null, cls);
  let pips = [];
  let shown;   // undefined: nothing drawn yet (null is a state: hidden)
  return {
    el,
    set(st) {
      if (st === shown) return;
      shown = st;
      const want = pipsOf(st, only);
      if (pips.length !== want.length || pips.some((p, i) => p.kind !== want[i].kind)) {
        pips = want.map(w => ({ kind: w.kind, s: '', el: node(doc, 'i', null, `fit fit-${w.kind}`) }));
        el.replaceChildren(...pips.map(p => p.el));
      }
      want.forEach((w, i) => {
        if (pips[i].s !== w.s) { pips[i].s = w.s; pips[i].el.dataset.s = w.s; }
      });
      el.title = fittingsText(st);
      el.hidden = !st;
    }
  };
}

/**
 * FITTINGS block of the ship telemetry card: one labelled row per fitting
 * group with its pips and count, the drive's thrust, torpedo rounds and the
 * DISARMED / ADRIFT / NO AMMO flag. Hidden for recordings without fitting data.
 */
export class FittingsCard {
  /** @param {Object} o - {root: #fitCard (hidden by default)} */
  constructor({ root, doc = defaultDoc() } = {}) {
    this.root = root;
    this.doc = doc;
    this.shown = undefined;
    this.rows = null;
  }

  _build() {
    const d = this.doc;
    const row = (label, only) => {
      const el = node(d, 'div', null, 'fit-row');
      const strip = pipStrip(d, 'fit-strip fit-strip-lg', only);
      const value = node(d, 'span', null, 'fit-value');
      el.append(node(d, 'span', label, 'fit-label'), strip.el, value);
      return { el, strip, value };
    };
    this.flag = node(d, 'span', null, 'fit-flag');
    const head = node(d, 'div', null, 'fit-head');
    head.append(node(d, 'span', 'FITTINGS', 'section-label'), this.flag);
    this.rows = { guns: row('GUNS', 'guns'), tubes: row('TUBES', 'tubes'), pd: row('PD', 'pd'), drive: row('DRIVE', 'drive') };
    this.root.replaceChildren(head, ...Object.values(this.rows).map(r => r.el));
  }

  /**
   * @param {Object|null} st - fittingsAt(...) for the selected ship (null hides the block)
   * @param {boolean} down - the ship is dying or destroyed
   */
  update(st, down = false) {
    if (!this.root) return;
    const key = st ? st : null;
    if (key === this.shown && down === this.down) return;
    this.shown = key;
    this.down = down;
    this.root.classList.toggle('hidden', !st);
    if (!st) return;
    if (!this.rows) this._build();
    const { guns, tubes, pd, drive } = this.rows;
    const group = (row, states, up, extra = '') => {
      row.el.hidden = states.length === 0;
      if (!states.length) return;
      row.strip.set(st);
      setText(row.value, `${up}/${states.length}${extra}`);
      row.value.dataset.level = up === 0 ? 'out' : up < states.length || states.includes('damaged') ? 'hurt' : 'ok';
    };
    group(guns, st.guns, st.gunsUp);
    group(tubes, st.tubes, st.tubesUp, st.rounds != null ? ` · ${st.rounds} RDS` : '');
    group(pd, st.pd, st.pdUp);
    drive.strip.set(st);
    setText(drive.value, st.adrift ? 'NO THRUST' : `${Math.round(st.thrust * 100)}% THRUST`);
    drive.value.dataset.level = st.adrift ? 'out' : st.thrust < 0.995 ? 'hurt' : 'ok';
    const flag = down ? '' : fittingsFlag(st);
    setText(this.flag, flag);
    this.flag.hidden = !flag;
    this.flag.dataset.flag = flag.toLowerCase().replace(' ', '-');
  }
}
