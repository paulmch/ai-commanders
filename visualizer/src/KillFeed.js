/**
 * KillFeed - toasts under the header for ships knocked out around the
 * playhead: who killed whom, with what, and the dying hulk's countdown to
 * its reactor detonation. Driven by loader.hud.kills, so scrubbing replays
 * the feed exactly (nothing is spawned on a wall clock). A click seeks to a
 * few seconds before the kill and frames the victim.
 *
 * Pass-5 recordings add what the blast did to a destroyed ship's toast
 * (ships caught, torpedoes killed or blinded: hud.blasts) and amber notice
 * toasts for the moments between kills (hud.notices: a ship disarmed or
 * adrift, a drive hit, a magazine cook-off, a ship caught in a blast).
 */
import { killStateAt, clockText, upperBound } from './HudIndex.js';
import { blastLine } from './FittingsIndex.js';
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

/**
 * Notices to show at t: those of the last `linger` s, newest first, the
 * latest per ship and kind, at most `max`.
 */
export function noticesAt(notices, t, linger = 8, max = 3) {
  const out = [];
  if (!notices?.length || max <= 0) return out;
  const seen = new Set();
  for (let i = upperBound(notices, t + 1e-6) - 1; i >= 0 && out.length < max; i--) {
    const n = notices[i];
    if (t - n.t > linger) break;
    const key = `${n.shipId}:${n.group || n.kind}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(n);
  }
  return out;
}

const NOTICE_ICON = { cookoff: '✹', blast: '◎', disarmed: '⚠', adrift: '⚠', drive: '⚠', plume: '≋', 'plume-update': '≋', crew: '☢' };

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

  /**
   * @param {Object[]} kills - hud.kills
   * @param {number} t
   * @param {Object} [hud] - loader.hud: notices and blasts (absent / empty on older recordings)
   */
  update(kills, t, hud = null) {
    if (!this.root) return;
    const items = feedAt(kills, t);
    // Kills first; notices fill what is left of a five-toast stack
    const notices = noticesAt(hud?.notices, t, 8, Math.min(3, 5 - items.length));
    const blasts = hud?.blasts || {};
    const key = items.map(i => `${i.kill.shipId}:${i.state}:${i.in != null ? Math.ceil(i.in) : ''}`).join('|') +
      '#' + notices.map(n => n.id).join('|');
    if (key === this.key) return;
    this.key = key;
    this.root.replaceChildren(...items.map(i => this.toast(i, blasts[i.kill.shipId])), ...notices.map(n => this.notice(n)));
    this.root.hidden = items.length + notices.length === 0;
  }

  /** Amber toast for a notice (FittingsIndex): who, what happened, what it cost. */
  notice(n) {
    const d = this.doc;
    const el = node(d, 'button', null, `kill-toast notice notice-${n.kind} ${n.faction || ''}`);
    el.type = 'button';
    el.dataset.shipId = n.shipId;
    el.title = `${clockText(n.t)} ${n.name}: ${n.status} - ${n.detail}. Click to replay.`;
    const top = node(d, 'span', null, 'kill-top');
    top.append(node(d, 'span', NOTICE_ICON[n.kind] || '⚠', 'kill-icon'), node(d, 'span', clockText(n.t), 'kill-time'),
      node(d, 'span', n.name, 'kill-victim'), node(d, 'span', n.status, 'kill-status'));
    el.append(top, node(d, 'span', n.detail, 'kill-bottom'));
    if (this.onSeek) el.addEventListener('click', () => this.onSeek(n));
    return el;
  }

  toast({ kill, ...s }, blast = null) {
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
    // What the reactor blast did, once it has gone off
    const blasted = s.state === 'destroyed' ? blastLine(blast) : '';
    if (blasted) el.append(node(d, 'span', blasted, 'kill-blast'));
    if (this.onSeek) el.addEventListener('click', () => this.onSeek(kill));
    return el;
  }

  clear() {
    this.key = '';
    this.root?.replaceChildren();
    if (this.root) this.root.hidden = true;
  }
}
