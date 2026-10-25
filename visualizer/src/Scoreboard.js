/**
 * Scoreboard - the header strip that says who is winning: per side the
 * commanding model, ships fighting / total with one hull pip per ship,
 * fleet hull %, torpedoes in flight, kills and model spend, plus a balance
 * bar of class-weighted remaining strength (the hull term of the engine's
 * time-limit verdict). Also owns the end-of-battle result banner.
 *
 * computeSides() is pure; the DOM classes only write when a value changed.
 */
import { shipValue, spendAt, killStateAt, shortModel, clockText } from './HudIndex.js';
import { node, setText, defaultDoc } from './hudDom.js';

const SIDES = ['alpha', 'beta'];

/**
 * @param {Object} states - interpolated ship states {id: {hull, dying, destroyed}}
 * @param {Object[]} torpedoes - in-flight torpedoes [{source}]
 * @param {Object} ships - loader.ships
 * @param {Object} hud - loader.hud (kills, spend)
 * @param {number} t
 */
export function computeSides(states, torpedoes, ships, hud, t) {
  const sides = {};
  for (const f of SIDES) sides[f] = { alive: 0, total: 0, dying: 0, lost: 0, hull: 0, hullPct: 0, strength: 0, maxStrength: 0, torps: 0, kills: 0, points: 0, spend: 0, pips: [] };
  for (const [id, info] of Object.entries(ships || {})) {
    const s = sides[info.faction];
    if (!s) continue;
    const st = states?.[id];
    const hull = Math.max(0, Math.min(100, st?.hull ?? 100));
    const state = st?.destroyed ? 'lost' : st?.dying ? 'dying' : 'alive';
    const value = shipValue(info.type);
    s.total++;
    s.maxStrength += value;
    if (state === 'alive') { s.alive++; s.hull += hull; s.strength += value * hull / 100; }
    else if (state === 'dying') s.dying++;
    else s.lost++;
    s.pips.push({ id, hull: state === 'alive' ? hull : 0, state });
  }
  for (const torp of torpedoes || []) {
    const f = ships?.[torp.source]?.faction || (String(torp.source || '').startsWith('alpha') ? 'alpha' : 'beta');
    if (sides[f]) sides[f].torps++;
  }
  for (const k of hud?.kills || []) {
    if (!killStateAt(k, t)) continue;
    const killer = k.faction === 'alpha' ? 'beta' : k.faction === 'beta' ? 'alpha' : null;
    if (killer) {
      sides[killer].kills++;
      sides[killer].points += hud.costs?.[k.shipId] || 0;
    }
  }
  for (const f of SIDES) {
    const s = sides[f];
    s.hullPct = s.total ? Math.round(s.hull / s.total) : 0;
    s.spend = spendAt(hud?.spend, f, t);
  }
  const a = sides.alpha.strength, b = sides.beta.strength;
  sides.share = a + b > 0 ? a / (a + b) : 0.5;
  return sides;
}

export function resultTitle(result) {
  if (!result) return '';
  if (result.winner) return `${String(result.winner).toUpperCase()} VICTORIOUS`;
  if (result.stopped) return 'BATTLE STOPPED';
  return 'DRAW';
}

/** Result reason without the exception class prefix ('RuntimeError: ...'). */
export const cleanReason = reason => String(reason || '').replace(/^[A-Z]\w*(Error|Exception):\s*/, '');

export class Scoreboard {
  /**
   * @param {Object} o
   * @param {Element} o.root - #scoreboard (contains [data-side=alpha|beta] and [data-balance])
   */
  constructor({ root, doc = defaultDoc() } = {}) {
    this.root = root;
    this.doc = doc;
    this.cache = {};
    this.sideEls = {};
    for (const f of SIDES) {
      const el = root?.querySelector(`[data-side="${f}"]`);
      if (!el) continue;
      this.sideEls[f] = {
        model: el.querySelector('[data-model]'), alive: el.querySelector('[data-alive]'), total: el.querySelector('[data-total]'),
        pips: el.querySelector('[data-pips]'), hull: el.querySelector('[data-hull]'), torps: el.querySelector('[data-torps]'),
        kills: el.querySelector('[data-kills]'), points: el.querySelector('[data-points]'), spend: el.querySelector('[data-spend]'), pipEls: []
      };
    }
    this.balance = root?.querySelector('[data-balance]');
  }

  /** Per-recording setup: model names and one pip per ship. */
  populate(ships, hud) {
    this.cache = {};
    for (const f of SIDES) {
      const els = this.sideEls[f];
      if (!els) continue;
      const model = hud?.models?.[f] || '';
      setText(els.model, shortModel(model));
      if (els.model) els.model.title = model ? `${f.toUpperCase()} commanded by ${model}` : '';
      els.pipEls = [];
      els.pips?.replaceChildren();
      for (const [id, info] of Object.entries(ships || {})) {
        if (info.faction !== f) continue;
        const pip = node(this.doc, 'i', null, 'score-pip');
        pip.title = info.name || id;
        const fill = node(this.doc, 'b');
        pip.append(fill);
        els.pips?.append(pip);
        els.pipEls.push({ pip, fill, h: -1, state: '' });
      }
    }
  }

  update(states, torpedoes, ships, hud, t) {
    const sides = computeSides(states, torpedoes, ships, hud, t);
    for (const f of SIDES) {
      const s = sides[f], els = this.sideEls[f];
      if (!els) continue;
      setText(els.alive, s.alive);
      setText(els.total, s.total);
      setText(els.hull, `${s.hullPct}%`);
      setText(els.torps, s.torps ? `▸${s.torps}` : '');
      setText(els.kills, `☠ ${s.kills}`);
      setText(els.points, s.points > 0 ? `${s.points} pts` : '');
      setText(els.spend, s.spend > 0 ? `$${s.spend.toFixed(2)}` : '');
      s.pips.forEach((p, i) => {
        const pe = els.pipEls[i];
        if (!pe) return;
        const h = Math.round(p.hull);
        if (pe.h !== h) { pe.h = h; pe.fill.style.height = `${h}%`; }
        if (pe.state !== p.state) { pe.state = p.state; pe.pip.dataset.state = p.state; }
      });
    }
    const share = Math.round(sides.share * 1000) / 10;
    if (this.balance && this.cache.share !== share) {
      this.cache.share = share;
      this.balance.style.width = `${share}%`;
      const parent = this.balance.parentNode;
      if (parent) parent.title = `Remaining fleet strength (class value × hull): ALPHA ${share.toFixed(0)}% · BETA ${(100 - share).toFixed(0)}%`;
    }
    return sides;
  }
}

/**
 * Is the replay result banner due? Only when replaying a recording (no live
 * session, ended or not) with the playhead at its end. A live battle's
 * verdict belongs to the live wire's end card (LiveCards.EndCard, shown
 * once the stream ends), so the two never stack.
 * @param {number} t - playhead
 * @param {number} duration - recording duration
 * @param {Object|null} live - the viewer's live-mode state (null in replay)
 */
export function resultBannerDue(t, duration, live) {
  return !live && duration > 0 && t >= duration - 1e-3;
}

/** End-of-battle card for a replayed recording: winner / draw / stopped with the reason and final tallies. */
export class ResultBanner {
  constructor({ root, doc = defaultDoc() } = {}) {
    this.root = root;
    this.doc = doc;
    this.dismissed = false;
    this.shown = false;
    this.key = '';
    root?.querySelector('[data-close]')?.addEventListener('click', () => { this.dismissed = true; this._show(false); });
  }

  reset() {
    this.dismissed = false;
    this.key = '';
    this._show(false);
  }

  /**
   * @param {boolean} over - the playhead has reached the end of a finished battle
   * @param {Object} result - loader.hud.result
   * @param {Object} sides - computeSides() output at the end
   * @param {number} duration
   */
  update(over, result, sides, duration) {
    const show = over && !this.dismissed && !!result?.known;
    if (show) {
      const key = `${resultTitle(result)}|${result.reason}|${sides?.alpha.alive}|${sides?.beta.alive}|${sides?.alpha.kills}|${sides?.beta.kills}`;
      if (key !== this.key) {
        this.key = key;
        this._render(result, sides, duration);
      }
    }
    this._show(show);
  }

  _render(result, sides, duration) {
    const r = this.root;
    if (!r) return;
    r.dataset.winner = result.winner || (result.stopped ? 'stopped' : 'draw');
    setText(r.querySelector('[data-kicker]'), `${result.stopped ? 'BATTLE HALTED' : 'BATTLE COMPLETE'} · T+${clockText(duration)}`);
    setText(r.querySelector('[data-title]'), resultTitle(result));
    setText(r.querySelector('[data-reason]'), cleanReason(result.reason));
    const tally = r.querySelector('[data-tally]');
    if (tally && sides) {
      const part = f => {
        const s = sides[f];
        const el = node(this.doc, 'span', null, `result-side ${f}`);
        el.append(node(this.doc, 'b', f.toUpperCase()), node(this.doc, 'span',
          ` ${s.alive}/${s.total} ships · ${s.kills} kill${s.kills === 1 ? '' : 's'}${s.points ? ` (${s.points} pts)` : ''} · hull ${s.hullPct}%`));
        return el;
      };
      tally.replaceChildren(part('alpha'), part('beta'));
    }
  }

  _show(on) {
    if (!this.root || this.shown === on) return;
    this.shown = on;
    this.root.classList.toggle('hidden', !on);
  }
}
