/**
 * Live-mode overlays: the checkpoint deliberation card, the connection-lost
 * card, the end-of-battle card and the auto-camera chip. Each owns a DOM
 * node it creates under #app (no index.html markup) and re-renders only
 * when its text changes. Data comes from main.js; the logic lives in
 * LiveSession.js.
 */
import { formatClock } from './LiveSession.js';
import { getShipClassName } from './shipSilhouettes.js';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

const shortModel = m => (m ? String(m).split('/').pop() : '');
const money = v => `$${Number(v || 0).toFixed(2)}`;

function mount(id, cls) {
  const node = el('div', cls);
  node.id = id;
  node.hidden = true;
  (document.getElementById('app') || document.body).appendChild(node);
  return node;
}

/**
 * COMMAND DELIBERATION: what the battle is waiting for at a checkpoint -
 * per side admiral / ship orders / captains, the calls in flight with
 * their age, elapsed time and spend.
 */
export class DeliberationCard {
  constructor() {
    this.root = mount('liveDeliberation', 'live-delib');
    this.root.setAttribute('role', 'status');
    this.root.setAttribute('aria-live', 'polite');
    this.key = '';
  }

  hide() {
    this.root.hidden = true;
  }

  /**
   * @param {Object} v - {checkpoint, checkpointT, elapsed, progress, models,
   *   spend, stage, error, shipName(id)}
   */
  show(v) {
    const rows = ['alpha', 'beta'].map(f => this.row(f, v));
    const spend = v.spend;
    const spendText = spend && (spend.limit_usd || spend.reported_usd)
      ? `SPEND ${money(spend.reported_usd)}${spend.limit_usd ? ` / ${money(spend.limit_usd)}` : ''}` : '';
    const callsText = spend ? `${spend.calls ?? 0} calls · ${spend.failures ?? 0} failed` : '';
    const key = JSON.stringify([v.checkpoint, Math.floor(v.elapsed), rows, spendText, callsText, v.stage, v.error]);
    this.root.hidden = false;
    if (key === this.key) return;
    this.key = key;
    const head = el('div', 'ld-head');
    head.append(el('span', 'ld-kicker', 'COMMAND DELIBERATION'),
      el('span', 'ld-cp', `CHECKPOINT ${v.checkpoint} · T+${Math.round(v.checkpointT)}s`),
      el('span', 'ld-clock', formatClock(v.elapsed)));
    const body = el('div', 'ld-rows');
    for (const r of rows) {
      const row = el('div', `ld-row ${r.faction}`);
      row.append(el('span', 'ld-faction', r.faction.toUpperCase()), el('span', 'ld-model', r.model));
      const steps = el('span', 'ld-steps');
      for (const s of r.steps) steps.append(el('span', `ld-step ${s.state}`, s.text));
      row.append(steps);
      if (r.pending) row.append(el('span', 'ld-pending', r.pending));
      body.append(row);
    }
    const children = [head, body];
    if (spendText || callsText) {
      const foot = el('div', 'ld-spend');
      foot.append(el('span', null, spendText));
      if (spend?.limit_usd) {
        const bar = el('div', 'ld-bar');
        const fill = el('i');
        fill.style.width = `${Math.min(100, 100 * (spend.reported_usd || 0) / spend.limit_usd)}%`;
        bar.append(fill);
        foot.append(bar);
      }
      foot.append(el('span', 'ld-calls', callsText));
      children.push(foot);
    }
    if (v.error) children.push(el('div', 'ld-error', v.error));
    this.root.replaceChildren(...children);
  }

  row(faction, v) {
    const p = v.progress?.[faction] || {};
    const of = p.of != null ? `/${p.of}` : '';
    const pendingAdmiral = (p.pending || []).find(c => c.actor === 'admiral');
    const steps = [];
    if (pendingAdmiral) {
      steps.push({ state: 'busy', text: `admiral ⟳ ${formatClock(pendingAdmiral.age_s)}` });
    } else if (p.admiral || p.admiralCalls) {
      // One call: its latency. Several (directive + per-ship orders, partly
      // parallel): the count - summed latencies would overstate the wait
      const detail = p.admiralCalls > 1 ? ` ×${p.admiralCalls}` : p.admiralLatency ? ` ${p.admiralLatency.toFixed(1)}s` : '';
      steps.push({ state: 'done', text: `admiral ✓${detail}` });
    } else {
      steps.push({ state: 'wait', text: 'admiral …' });
    }
    steps.push({ state: p.of && p.orders >= p.of ? 'done' : p.orders ? 'busy' : 'wait', text: `orders ${p.orders || 0}${of}` });
    steps.push({ state: p.of && p.captains >= p.of ? 'done' : p.captains ? 'busy' : 'wait', text: `captains ${p.captains || 0}${of}` });
    if (p.errors) steps.push({ state: 'error', text: `${p.errors} failed` });
    const captain = (p.pending || []).find(c => c.actor !== 'admiral');
    const pending = captain
      ? `⟳ ${captain.actor} ${v.shipName?.(captain.ship_id) || captain.ship_id || ''} ${formatClock(captain.age_s)}`.replace(/\s+/g, ' ')
      : '';
    return { faction, model: shortModel(v.models?.[faction]), steps, pending };
  }
}

/** CONNECTION LOST: the server stopped answering; offer the saved replay. */
export class ConnectionCard {
  constructor({ onReplay, onRetry, onStop }) {
    this.root = mount('liveConnection', 'live-conn');
    this.root.setAttribute('role', 'alert');
    this.handlers = { onReplay, onRetry, onStop };
    this.key = '';
  }

  hide() {
    this.root.hidden = true;
    this.key = '';
  }

  /** @param {Object} v - {failures, nextRetryS, lastGoodS, replayUrl, loading} */
  show(v) {
    this.root.hidden = false;
    const key = JSON.stringify([v.failures, Math.ceil(v.nextRetryS ?? -1), Math.floor(v.lastGoodS), !!v.replayUrl, v.loading, v.message]);
    if (key === this.key) return;
    this.key = key;
    const title = el('div', 'lc-title', 'CONNECTION LOST');
    const detail = el('p', 'lc-detail', v.message ||
      `The battle server stopped answering ${formatClock(v.lastGoodS)} ago (${v.failures} failed polls). ` +
      'Playback continues with everything received so far.');
    const actions = el('div', 'lc-actions');
    if (v.replayUrl) {
      const replay = el('button', 'primary', v.loading ? 'LOADING SAVED REPLAY…' : 'LOAD SAVED REPLAY');
      replay.disabled = !!v.loading;
      replay.title = v.replayUrl;
      replay.addEventListener('click', () => this.handlers.onReplay?.());
      actions.append(replay);
    }
    const retry = el('button', null, v.nextRetryS != null ? `RETRY NOW · auto in ${Math.ceil(v.nextRetryS)}s` : 'RETRY');
    retry.addEventListener('click', () => this.handlers.onRetry?.());
    const stop = el('button', null, 'END HERE');
    stop.title = 'Stop polling and treat what was received as the whole battle';
    stop.addEventListener('click', () => this.handlers.onStop?.());
    actions.append(retry, stop);
    this.root.replaceChildren(title, detail, actions);
  }
}

/**
 * End of battle: verdict, kills (click to watch), survivors, spend, and
 * the next things to do with the replay.
 */
export class EndCard {
  constructor({ onSeek, onReplay, onDecisions }) {
    this.root = mount('liveEndCard', 'live-end');
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', 'Battle result');
    this.handlers = { onSeek, onReplay, onDecisions };
    this.open = false;
  }

  hide() {
    this.root.hidden = true;
    this.open = false;
  }

  /**
   * @param {Object} v - {verdict, duration, checkpoints, kills: [{t, name, type, faction, killer}],
   *   survivors: {alpha: [{name, type, hull}], beta: [...]}, billing, models}
   */
  show(v) {
    this.open = true;
    this.root.hidden = false;
    const close = el('button', 'le-close', '×');
    close.setAttribute('aria-label', 'Close result');
    close.addEventListener('click', () => this.hide());
    const verdict = v.verdict;
    const kicker = el('div', 'le-kicker',
      `${verdict.kind === 'halted' ? 'BATTLE STOPPED' : verdict.kind === 'lost' ? 'STREAM ENDED' : 'BATTLE COMPLETE'} · T+${formatClock(v.duration)}` +
      (v.checkpoints ? ` · ${v.checkpoints} CHECKPOINTS` : ''));
    const title = el('h2', `le-title ${verdict.kind} ${verdict.faction || ''}`, verdict.title);
    const detail = el('p', 'le-detail', verdict.detail || '');

    const kills = el('section', 'le-kills');
    kills.append(el('h3', null, `KILLS · ${v.kills.length}`));
    if (v.kills.length) {
      const list = el('ol');
      for (const k of v.kills) {
        const item = el('li');
        const btn = el('button', `le-kill ${k.faction}`);
        btn.append(el('span', 'le-time', formatClock(k.t)), el('span', 'le-name', k.name),
          el('span', 'le-class', getShipClassName(k.type)));
        if (k.killer) btn.append(el('span', 'le-killer', `by ${k.killer}`));
        btn.title = `Watch the kill at ${formatClock(k.t)}`;
        btn.addEventListener('click', () => { this.hide(); this.handlers.onSeek?.(Math.max(0, k.t - 6)); });
        item.append(btn);
        list.append(item);
      }
      kills.append(list);
    } else {
      kills.append(el('p', 'le-empty', 'No ships destroyed.'));
    }

    const surv = el('section', 'le-survivors');
    surv.append(el('h3', null, 'SURVIVORS'));
    for (const f of ['alpha', 'beta']) {
      const list = v.survivors[f] || [];
      const group = el('div', `le-side ${f}`);
      group.append(el('div', 'le-side-head', `${f.toUpperCase()} ${list.length}/${v.fleetSize?.[f] ?? list.length}` +
        (v.models?.[f] ? ` · ${shortModel(v.models[f])}` : '')));
      for (const s of list) {
        const row = el('div', 'le-ship');
        const bar = el('span', 'le-hull');
        const fill = el('i');
        fill.style.width = `${Math.max(0, Math.min(100, s.hull))}%`;
        bar.append(fill);
        row.append(el('span', 'le-name', s.name), el('span', 'le-class', getShipClassName(s.type)), bar,
          el('span', 'le-pct', `${Math.round(s.hull)}%`));
        group.append(row);
      }
      surv.append(group);
    }

    const children = [close, kicker, title, detail];
    const cols = el('div', 'le-cols');
    cols.append(kills, surv);
    children.push(cols);
    const b = v.billing;
    if (b && (b.reported_cost_usd != null || b.calls != null)) {
      const parts = [];
      if (b.reported_cost_usd != null) parts.push(`SPEND ${money(b.reported_cost_usd)}${b.spend_limit_usd ? ` / ${money(b.spend_limit_usd)}` : ''}`);
      if (b.calls != null) parts.push(`${b.calls} model calls`);
      if (b.failures) parts.push(`${b.failures} failed`);
      if (b.mode && b.mode !== 'paid') parts.push(String(b.mode).toUpperCase());
      children.push(el('div', 'le-spend', parts.join(' · ')));
    }
    const actions = el('div', 'le-actions');
    const replay = el('button', 'primary', 'REPLAY FROM START');
    replay.addEventListener('click', () => { this.hide(); this.handlers.onReplay?.(); });
    actions.append(replay);
    if (v.kills.length) {
      const first = el('button', null, 'WATCH FIRST KILL');
      first.addEventListener('click', () => { this.hide(); this.handlers.onSeek?.(Math.max(0, v.kills[0].t - 6)); });
      actions.append(first);
    }
    const decisions = el('button', null, 'DECISIONS');
    decisions.addEventListener('click', () => { this.hide(); this.handlers.onDecisions?.(); });
    actions.append(decisions);
    children.push(actions);
    this.root.replaceChildren(...children);
  }
}

/** AUTO CAM chip: director on / shot / paused by the viewer; click or A toggles. */
export class DirectorChip {
  constructor({ onToggle }) {
    this.root = mount('directorChip', 'director-chip');
    this.root.hidden = false;
    this.root.setAttribute('role', 'button');
    this.root.tabIndex = 0;
    this.root.title = 'Auto camera: frames the fleets and cuts to kills and salvos. Drag or zoom to take over; A toggles.';
    this.root.addEventListener('click', () => onToggle?.());
    this.root.addEventListener('keydown', e => {
      if (e.code === 'Enter' || e.code === 'Space') { e.preventDefault(); e.stopPropagation(); onToggle?.(); }
    });
    this.text = '';
  }

  setVisible(on) {
    this.root.hidden = !on;
  }

  /** @param {Object} s - CameraController.directorState() */
  update(s) {
    let text;
    let cls;
    if (!s?.enabled) { text = 'AUTO CAM OFF'; cls = 'off'; }
    else if (s.hold === 'ship') { text = 'AUTO CAM · SHIP VIEW'; cls = 'held'; }
    else if (s.hold) {
      text = s.secondsLeft != null ? `AUTO CAM PAUSED ${Math.ceil(s.secondsLeft)}s` : 'AUTO CAM PAUSED';
      cls = 'held';
    }
    else { text = `AUTO CAM · ${({ kill: 'KILL', detonation: 'KILL', salvo: 'SALVO' })[s.shot] || 'FLEETS'}`; cls = 'on'; }
    const full = `${text} <kbd>A</kbd>`;
    if (full === this.text) return;
    this.text = full;
    this.root.className = `director-chip ${cls}`;
    this.root.innerHTML = `<span class="dc-dot"></span>${text}<kbd>A</kbd>`;
    this.root.setAttribute('aria-pressed', String(!!s?.enabled));
  }
}
