/**
 * TimelineMarkers - the story of the battle along the scrub bar.
 *
 * Checkpoint lane above the track: one tick per decision checkpoint (CP n),
 * labelled when there is room. Event lane below it: a mirrored hit-density
 * sparkline (alpha up, beta down, 5 s bins), torpedo salvos as faction
 * chevrons with their round count, and kills as raised faction-coloured
 * crosses. Every marker is a button: a click seeks a few seconds before the
 * moment and frames the ship involved.
 *
 * Rebuilt from loader.hud only on load and on each live append; nothing
 * here runs per frame.
 */
import { clockText } from './HudIndex.js';
import { node, defaultDoc } from './hudDom.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Marker descriptors (pure): {lane, kind, t, pct, label, title, faction, shipId, seekLead}. */
export function timelineMarks(hud, duration, ships = {}) {
  if (!hud || !(duration > 0)) return [];
  const pct = t => Math.max(0, Math.min(100, (t / duration) * 100));
  const out = [];
  const cps = hud.checkpoints || [];
  const labelEvery = cps.length > 24 ? 5 : cps.length > 12 ? 2 : 1;
  for (const cp of cps) {
    out.push({ lane: 'cp', kind: 'checkpoint', t: cp.t, pct: pct(cp.t), label: cp.n % labelEvery === 0 || cp.n === 1 ? String(cp.n) : '',
      title: `CP${cp.n} · ${clockText(cp.t)} · orders issued`, faction: null, shipId: null, seekLead: 0 });
  }
  for (const s of hud.salvos || []) {
    const targets = s.targets.map(([name, n]) => `${name}${n > 1 ? ' ×' + n : ''}`).join(', ');
    out.push({ lane: 'event', kind: 'salvo', t: s.t, pct: pct(s.t), label: s.count > 1 ? String(s.count) : '',
      title: `${clockText(s.t)} ${String(s.faction).toUpperCase()} torpedo salvo ×${s.count} → ${targets}`,
      faction: s.faction, shipId: s.shipIds[0] || null, seekLead: 2 });
  }
  for (const k of hud.kills || []) {
    const by = k.weapon === 'meltdown' ? 'reactor meltdown' : k.killerName ? `${k.inferred ? 'probably by ' : 'by '}${k.killerName}${k.weapon ? ' (' + k.weapon + ')' : ''}` : 'destroyed';
    out.push({ lane: 'event', kind: 'kill', t: k.t, pct: pct(k.t), label: '',
      title: `${clockText(k.t)} ${k.name} knocked out — ${by}`, faction: k.faction, shipId: k.shipId, seekLead: 5 });
  }
  return out;
}

/** SVG path for a mirrored density chart: alpha bins rise from the midline, beta bins fall. */
export function sparkPaths(bins, width = 1000, height = 16) {
  const mid = height / 2;
  const n = Math.max(bins?.alpha?.length || 0, bins?.beta?.length || 0);
  if (!n) return { alpha: '', beta: '' };
  const max = Math.max(1, ...bins.alpha, ...bins.beta);
  const w = width / n;
  const path = (arr, dir) => arr.map((v, i) => v > 0
    ? `M${(i * w).toFixed(1)} ${mid}h${Math.max(0.6, w - 0.6).toFixed(1)}v${(dir * Math.max(1, (v / max) * (mid - 1))).toFixed(1)}h${(-Math.max(0.6, w - 0.6)).toFixed(1)}Z`
    : '').join('');
  return { alpha: path(bins.alpha, -1), beta: path(bins.beta, 1) };
}

export class TimelineMarkers {
  /**
   * @param {Object} o
   * @param {Element} o.events - lane below the track (#timelineMarkers)
   * @param {Element} [o.checkpoints] - lane above the track (#timelineCheckpoints)
   * @param {Function} o.onSeek - (t, shipId) => void
   */
  constructor({ events, checkpoints = null, onSeek, doc = defaultDoc() } = {}) {
    this.events = events;
    this.checkpoints = checkpoints;
    this.onSeek = onSeek;
    this.doc = doc;
  }

  build(hud, duration, ships) {
    if (!this.events) return;
    const d = this.doc;
    const marks = timelineMarks(hud, duration, ships);
    const eventNodes = [];
    const cpNodes = [];
    // Density sparkline first so markers draw over it
    if (d.createElementNS && hud?.hitBins) {
      const svg = d.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('class', 'timeline-spark');
      svg.setAttribute('viewBox', '0 0 1000 16');
      svg.setAttribute('preserveAspectRatio', 'none');
      svg.setAttribute('aria-hidden', 'true');
      const paths = sparkPaths(hud.hitBins);
      for (const f of ['alpha', 'beta']) {
        if (!paths[f]) continue;
        const p = d.createElementNS(SVG_NS, 'path');
        p.setAttribute('class', `spark-${f}`);
        p.setAttribute('d', paths[f]);
        svg.appendChild(p);
      }
      eventNodes.push(svg);
    }
    for (const m of marks) {
      const el = node(d, 'button', m.label, `timeline-marker marker-${m.kind}${m.faction ? ' ' + m.faction : ''}`);
      el.type = 'button';
      el.style.left = `${m.pct}%`;
      el.title = m.title;
      el.setAttribute('aria-label', m.title);
      el.dataset.t = String(m.t);
      el.addEventListener('click', (e) => {
        e.stopPropagation?.();
        this.onSeek?.(Math.max(0, m.t - m.seekLead), m.shipId);
      });
      (m.lane === 'cp' && this.checkpoints ? cpNodes : eventNodes).push(el);
    }
    this.events.replaceChildren(...eventNodes);
    this.checkpoints?.replaceChildren(...cpNodes);
  }

  clear() {
    this.events?.replaceChildren();
    this.checkpoints?.replaceChildren();
  }
}
