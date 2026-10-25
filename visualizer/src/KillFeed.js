/**
 * KillFeed - toasts under the header for ships knocked out around the
 * playhead: who killed whom, with what, and the dying hulk's countdown to
 * its reactor detonation. Driven by loader.hud.kills, so scrubbing replays
 * the feed exactly (nothing is spawned on a wall clock). A click seeks to a
 * few seconds before the kill and frames the victim.
 */
import { killStateAt, clockText } from './HudIndex.js';
import { node, defaultDoc } from './hudDom.js';
import { getShipClassName } from './shipSilhouettes.js';

/** Kills to show at t: dying hulks, and kills within `linger` s of their destruction (newest first, at most `max`). */
export function feedAt(kills, t, linger = 10, max = 4) {
  const out = [];
  for (let i = (kills?.length || 0) - 1; i >= 0 && out.length < max; i--) {
    const k = kills[i];
    const s = killStateAt(k, t);
    if (!s) continue;
    if (s.state === 'dying' || t - (k.destroyedT ?? k.t) <= linger) out.push({ kill: k, ...s });
  }
  return out;
}

export function killLines(k, s) {
  const cls = k.type ? ` · ${getShipClassName(k.type)}` : '';
  const head = `${k.name}${cls}`;
  let detail;
  if (k.weapon === 'meltdown') detail = 'reactor meltdown (overheat)';
  else {
    const by = k.killerName ? `${k.inferred ? 'probably by' : 'by'} ${k.killerName}` : 'cause unknown';
    detail = `${by}${k.weapon ? ' · ' + k.weapon : ''}`;
  }
  const status = s.state === 'dying'
    ? (s.in != null ? `CRIPPLED · REACTOR IN ${Math.ceil(s.in)}s` : 'CRIPPLED')
    : 'DESTROYED';
  return { head, detail, status };
}

export class KillFeed {
  /**
   * @param {Object} o
   * @param {Element} o.root - #killFeed
   * @param {Function} [o.onSeek] - (kill) => void
   */
  constructor({ root, onSeek = null, doc = defaultDoc() } = {}) {
    this.root = root;
    this.onSeek = onSeek;
    this.doc = doc;
    this.key = '';
  }

  update(kills, t) {
    if (!this.root) return;
    const items = feedAt(kills, t);
    const key = items.map(i => `${i.kill.shipId}:${i.state}:${i.in != null ? Math.ceil(i.in) : ''}`).join('|');
    if (key === this.key) return;
    this.key = key;
    this.root.replaceChildren(...items.map(i => this.toast(i)));
    this.root.hidden = items.length === 0;
  }

  toast({ kill, ...s }) {
    const d = this.doc;
    const { head, detail, status } = killLines(kill, s);
    const el = node(d, 'button', null, `kill-toast ${kill.faction || ''} ${s.state}`);
    el.type = 'button';
    el.dataset.shipId = kill.shipId;
    el.title = `${clockText(kill.t)} ${head} — ${detail}. Click to replay the kill.`;
    const top = node(d, 'span', null, 'kill-top');
    top.append(node(d, 'span', '☠', 'kill-icon'), node(d, 'span', clockText(kill.t), 'kill-time'), node(d, 'span', head, 'kill-victim'));
    const bottom = node(d, 'span', null, 'kill-bottom');
    bottom.append(node(d, 'span', status, 'kill-status'), node(d, 'span', detail, `kill-by ${kill.killerFaction || ''}`));
    el.append(top, bottom);
    if (this.onSeek) el.addEventListener('click', () => this.onSeek(kill));
    return el;
  }

  clear() {
    this.key = '';
    this.root?.replaceChildren();
    if (this.root) this.root.hidden = true;
  }
}
