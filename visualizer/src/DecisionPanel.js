import { describeEvent, readable } from './DecisionIndex.js';
import { showLightbox } from './hudDom.js';
import { checkpointAt } from './HudIndex.js';

const node = (tag, text, cls) => {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  if (cls) el.className = cls;
  return el;
};
const clock = t => `${Math.floor(t / 60).toString().padStart(2, '0')}:${Math.floor(t % 60).toString().padStart(2, '0')}`;

export class DecisionPanel {
  constructor(app) {
    this.app = app;
    this.open = false;
    this.shipId = null;
    this.decisionId = null;
    // Follow the latest decision at the playhead until the viewer picks one
    // (selector, arrows, an evidence jump); the header chip re-arms it.
    this.followLatest = true;
    this.lastKey = '';
    this.root = document.getElementById('decisionPanel');
    this.toggle = document.getElementById('decisionsBtn');
    this.toggle.addEventListener('click', () => this.setOpen(!this.open));
    this.root.querySelector('[data-close]').addEventListener('click', () => this.setOpen(false));
    this.shipSelect = this.root.querySelector('[data-ship]');
    this.shipSelect.addEventListener('change', () => {
      this.app.selectShip(this.shipSelect.value);
      this.selectShip(this.shipSelect.value);
    });
    this.decisionSelect = this.root.querySelector('[data-decision]');
    this.decisionSelect.addEventListener('change', () => this.choose(this.decisionSelect.value));
    this.root.querySelector('[data-prev]').addEventListener('click', () => this.step(-1));
    this.root.querySelector('[data-next]').addEventListener('click', () => this.step(1));
    this.body = this.root.querySelector('[data-body]');
    this.followBtn = this.root.querySelector('[data-follow]');
    this.followBtn?.addEventListener('click', () => this.setFollow(!this.followLatest));
  }

  /** Show this decision and stop following the latest (deep links). */
  pin(id) {
    this.decisionId = id;
    this.followLatest = false;
    this.lastKey = '';
  }

  setFollow(on) {
    this.followLatest = !!on;
    if (on) this.selectedEventId = null;
    this.lastKey = '';
    this.update(this.app.timeController?.currentTime || 0);
  }

  /** Open on a ship's decision (command-net WHY button); pins it. */
  openAt(shipId, decisionId) {
    if (!this.open) this.setOpen(true);
    if (shipId && this.app.loader?.ships?.[shipId]) {
      this.app.selectShip?.(shipId);
      this.selectShip(shipId);
    }
    if (decisionId && this.app.loader?.decisions?.byId.has(decisionId)) this.choose(decisionId);
  }

  setOpen(open) {
    this.open = open;
    this.root.classList.toggle('hidden', !open);
    this.toggle.classList.toggle('active', open);
    this.toggle.setAttribute('aria-expanded', String(open));
    document.body.classList.toggle('decisions-open', open);
    if (open) {
      this.selectShip(this.app.selectedShipId || Object.keys(this.app.loader?.ships || {})[0]);
      this.root.querySelector('[data-close]').focus();
    } else this.toggle.focus();
  }

  reset() {
    this.shipId = null;
    this.decisionId = null;
    this.followLatest = true;
    this.selectedEventId = null;
    this.lastKey = '';
    this.shipSelect.replaceChildren();
    for (const [id, ship] of Object.entries(this.app.loader.ships)) {
      const option = node('option', `${ship.faction.toUpperCase()} · ${ship.name}`);
      option.value = id;
      this.shipSelect.append(option);
    }
    this.selectShip(this.app.selectedShipId || Object.keys(this.app.loader.ships)[0]);
  }

  selectShip(id) {
    if (this.shipId !== id) { this.decisionId = null; this.selectedEventId = null; this.followLatest = true; }
    this.shipId = id;
    this.shipSelect.value = id || '';
    this.lastKey = '';
    this.update(this.app.timeController?.currentTime || 0);
  }

  choose(id) {
    const e = this.app.loader.decisions.byId.get(id);
    if (!e) return;
    this.decisionId = id;
    this.followLatest = false;
    this.selectedEventId = null;
    this.jump(e, false);
  }

  step(direction) {
    const history = this.app.loader.decisions.history(this.shipId);
    const pos = history.findIndex(e => e.event_id === this.decisionId);
    const next = history[Math.max(0, Math.min(history.length - 1, pos + direction))];
    if (next) this.choose(next.event_id);
  }

  jump(event, focus = true) {
    this.followLatest = false;
    this.app.setLiveFollow(false);
    this.app.timeController.pause();
    this.app.timeController.seek(event.timestamp);
    this.app.updatePlayPauseButton();
    // Select the involved ship without switching away from the originating order.
    const id = event.data.target_id || event.ship_id;
    if (focus) {
      this.selectedEventId = event.event_id;
      if (id && this.app.loader.ships[id]) this.app.pendingEvidenceFocus = id;
    }
    this.lastKey = '';
    this.update(event.timestamp);
    if (focus) this.body.scrollTop = 0;
  }

  eventButton(event, label) {
    const button = node('button', `${clock(event.timestamp)} · ${label || describeEvent(event, this.app.loader.ships)}`, 'decision-event');
    button.type = 'button';
    button.dataset.eventId = event.event_id;
    button.addEventListener('click', () => this.jump(event));
    return button;
  }

  section(title) {
    const section = node('section', null, 'decision-section');
    section.append(node('h3', title));
    this.body.append(section);
    return section;
  }

  rawDetails(label, data) {
    const details = node('details', null, 'decision-raw');
    details.append(node('summary', label), node('pre', JSON.stringify(data, null, 2)));
    return details;
  }

  /**
   * A model call's input: image parts (the admiral's tactical plot) render
   * as thumbnails; the JSON keeps a size placeholder instead of the base64.
   */
  modelInput(call, assets, index) {
    const images = [];
    const messages = (call.data.message_refs || []).map(ref => {
      const m = assets[ref];
      if (!m) return {unavailable: ref};
      if (!Array.isArray(m.content)) return m;
      return {...m, content: m.content.map(part => {
        const url = part?.type === 'image_url' ? part.image_url?.url : null;
        if (!url) return part;
        images.push(url);
        return {type: 'image_url', image_url: `[image ${images.length}: ${Math.round(url.length * 0.75 / 1024)} KB, shown above]`};
      })};
    });
    const admiral = call.data.actor === 'admiral';
    const wrap = node('div', null, 'decision-model-input');
    if (images.length) {
      const fig = node('figure', null, 'decision-vision');
      const n = checkpointAt(this.app.loader.hud?.checkpoints || [], call.timestamp);
      const cp = n > 0 ? `CP${n} · ${clock(call.timestamp)}` : clock(call.timestamp);
      const caption = `What ${admiral ? 'Admiral' : 'Captain'} ${call.data.model || ''} saw · ${cp}`;
      for (const src of images) {
        const thumb = node('button', null, 'decision-vision-thumb');
        thumb.type = 'button';
        thumb.title = 'Enlarge';
        const img = document.createElement('img');
        img.src = src;
        img.alt = caption;
        img.loading = 'lazy';
        thumb.append(img);
        thumb.addEventListener('click', () => showLightbox(document, src, caption));
        fig.append(thumb);
      }
      fig.append(node('figcaption', caption));
      wrap.append(fig);
    }
    wrap.append(this.rawDetails(`${admiral ? 'Admiral' : 'Captain'} model input · ${call.event_id}`, {
      model: call.data.model, settings: call.data.request, messages,
      tools: assets[call.data.tools_ref], finish_reason: call.data.finish_reason, usage: call.data.usage,
    }));
    return wrap;
  }

  renderFollow(index) {
    const b = this.followBtn;
    if (!b) return;
    const pinned = !this.followLatest && index.byId.get(this.decisionId);
    b.textContent = pinned ? `PINNED ${clock(pinned.timestamp)}` : '● LATEST';
    b.classList.toggle('active', !pinned);
    b.setAttribute('aria-pressed', String(!pinned));
    b.title = pinned ? 'Showing a chosen decision. Click to follow the latest orders again.'
      : 'Following the latest decision at the playhead. Pick a checkpoint to pin one.';
  }

  update(time) {
    if (!this.open || !this.app.loader?.decisions) return;
    const index = this.app.loader.decisions;
    const history = index.history(this.shipId);
    // Live follow keeps the panel on the newest orders as they arrive
    const live = this.app.live;
    if (live?.follow && live.started && !live.ended) this.followLatest = true;
    // Resolve the decision BEFORE keying, or a followed panel never advances
    if (this.followLatest || !this.decisionId || !index.byId.has(this.decisionId)) {
      this.decisionId = index.latest(this.shipId, time)?.event_id;
    }
    const key = `${Math.floor(time)}:${index.events.length}:${this.shipId}:${this.decisionId}:${this.followLatest}`;
    if (key === this.lastKey) return;
    this.lastKey = key;
    const detail = index.detail(this.decisionId, time);
    this.renderFollow(index);
    const oldOptions = this.decisionSelect.options.length;
    if (oldOptions !== history.length || this.decisionSelect.dataset.ship !== this.shipId) {
      this.decisionSelect.replaceChildren();
      for (const e of history) {
        const option = node('option', `${clock(e.timestamp)} · ${e.data.maneuver_type || 'Orders'}`);
        option.value = e.event_id;
        this.decisionSelect.append(option);
      }
      this.decisionSelect.dataset.ship = this.shipId;
    }
    this.decisionSelect.value = this.decisionId || '';
    const pos = history.findIndex(e => e.event_id === this.decisionId);
    this.root.querySelector('[data-prev]').disabled = pos <= 0;
    this.root.querySelector('[data-next]').disabled = pos >= history.length - 1;
    // Keep expanded evidence and scroll position when playhead advances.
    const expanded = [...this.body.querySelectorAll('details[open]')].map(d => d.querySelector('summary').textContent);
    const scroll = this.body.scrollTop;
    this.body.replaceChildren();
    if (!detail || detail.decision.timestamp > time) {
      const awaiting = live && !live.ended;
      const text = awaiting && !history.some(e => e.timestamp <= time)
        ? `Awaiting first orders: commanders deciding (CP${Math.max(1, (live.checkpoint || 0) + (live.status === 'paused' ? 0 : 1))}).`
        : history.length ? 'No decision at this playhead. Use the checkpoint selector to jump to an order.'
          : 'This recording has no captain decisions for this ship.';
      this.body.append(node('p', text, 'decision-empty'));
      return;
    }
    const {decision, context, calls, notes, execution, outcomes, facts, modern, report, sent = [], received = []} = detail;
    this.body.append(node('p', `${modern ? 'LINKED EXECUTION' : 'LEGACY · OBSERVED SEQUENCE'} · evidence through ${clock(time)}`, 'decision-badge'));
    const selectedEvent = index.byId.get(this.selectedEventId);
    if (selectedEvent && selectedEvent.timestamp <= time) {
      const evidence = node('div', null, 'decision-selected-event');
      evidence.append(node('span', `SELECTED EVIDENCE · ${selectedEvent.event_id} · ${clock(selectedEvent.timestamp)}`, 'decision-label'),
        node('p', describeEvent(selectedEvent, index.ships)), this.rawDetails('Inspect event data', selectedEvent));
      this.body.append(evidence);
    }
    if (!modern) this.body.append(node('p', 'This recording has partial orders and no command attribution. Events below occurred during this decision window; they are not proven consequences.', 'decision-caveat'));
    const intent = this.section('01 / Commander intent');
    if (!context.length) intent.append(node('p', 'No admiral order recorded for this checkpoint.', 'decision-muted'));
    for (const e of context) {
      intent.append(node('span', readable(e.event_type), 'decision-label'), this.eventButton(e));
    }
    if (received.length) {
      intent.append(node('span', `fleet net received · ${received.length}`, 'decision-label'));
      for (const e of received) intent.append(this.eventButton(e));
    }
    const orders = this.section('02 / Captain orders');
    orders.append(node('p', `${decision.data.captain_name || 'Commander'} · ${decision.data.model || 'model not recorded'}`, 'decision-muted'));
    if (decision.data.call_failed) orders.append(node('p', 'Model call failed; this is not a deliberate choice to issue no orders.', 'decision-rejected'));
    if (report) {
      const status = String(report.data.status || 'ACK').toUpperCase();
      orders.append(node('span', `report to admiral · ${status}`, `decision-label decision-report-${status.toLowerCase()}`));
      const lines = [report.data.reason, report.data.intent && report.data.intent !== report.data.reason ? `Intent: ${report.data.intent}` : '',
        report.data.request?.text ? `Request: ${report.data.request.text}` : ''].filter(Boolean);
      orders.append(node('blockquote', lines.join('\n') || status));
    } else if (decision.data.acknowledgment) orders.append(node('blockquote', decision.data.acknowledgment));
    for (const e of sent) orders.append(this.eventButton(e));
    for (const e of notes) orders.append(this.eventButton(e));
    const tools = decision.data.tool_calls || [];
    if (tools.length) {
      for (const tool of tools) {
        const row = node('div', null, 'decision-tool');
        const args = tool.arguments || {};
        const parts = Object.entries(args).map(([key, value]) => {
          if (key === 'throttle') return `${Math.round(value * 100)}% thrust`;
          if (key === 'target_id') return `target: ${index.ships[value]?.name || value}`;
          if (key === 'extend') return value ? 'Extend radiators' : 'Retract radiators';
          return `${readable(key)}: ${typeof value === 'object' ? JSON.stringify(value) : value}`;
        });
        row.append(node('strong', readable(tool.name)), node('p', parts.join(' · ')));
        orders.append(row);
      }
    } else {
      orders.append(node('p', `${decision.data.maneuver_type || 'Orders'}${decision.data.target_id ? ' → ' + (index.ships[decision.data.target_id]?.name || decision.data.target_id) : ''}`));
    }
    for (const error of decision.data.tool_errors || []) orders.append(node('p', String(error), 'decision-rejected'));
    if (modern) orders.append(this.rawDetails('Resolved commands', decision.data.commands));
    const assets = this.app.loader.sourceData.assets || {};
    for (const call of calls) orders.append(this.modelInput(call, assets, index));
    const executionSection = this.section('03 / Ship execution');
    if (!execution.length) executionSection.append(node('p', 'No execution feedback recorded by this time.', 'decision-muted'));
    const latestExecution = execution.filter(e => e.event_type === 'execution_state').at(-1);
    if (latestExecution) executionSection.append(this.eventButton(latestExecution));
    const waits = execution.filter(e => e.event_type === 'weapon_status' && !['cooldown', 'fired'].includes(e.data.reason));
    for (const e of waits.slice(-3)) executionSection.append(this.eventButton(e));
    if (execution.length) {
      const ledger = node('details');
      ledger.append(node('summary', `Command and execution ledger · ${execution.length} events`));
      for (const e of execution) ledger.append(this.eventButton(e));
      executionSection.append(ledger);
    }
    const summary = this.section('04 / Observed consequences');
    summary.append(node('p', 'Summaries use recorded events. They do not establish what a different order would have changed.', 'decision-muted'));
    if (!facts.length) summary.append(node('p', 'Advance playback to see the result of these orders.', 'decision-muted'));
    for (const fact of facts) {
      const row = node('div', null, 'decision-fact');
      row.append(node('p', fact.text));
      const evidence = node('details');
      evidence.append(node('summary', `${fact.event_ids.length} supporting events`));
      for (const id of fact.event_ids) evidence.append(this.eventButton(index.byId.get(id)));
      row.append(evidence);
      summary.append(row);
    }
    const recordedCommentary = this.app.loader.sourceData.commentary?.filter(c => c.decision_id === decision.event_id) || [];
    for (const c of recordedCommentary) {
      const refs = c.event_ids.map(id => index.byId.get(id));
      if (refs.length && refs.every(e => e && e.timestamp <= time)) {
        const block = node('div', null, 'decision-fact');
        block.append(node('span', 'MODEL INTERPRETATION · ' + c.model, 'decision-label'), node('p', c.text));
        for (const e of refs) block.append(this.eventButton(e));
        summary.append(block);
      }
    }
    const outcomeDetails = node('details');
    outcomeDetails.append(node('summary', `Outcome ledger · ${outcomes.length} events`));
    for (const e of outcomes) outcomeDetails.append(this.eventButton(e));
    summary.append(outcomeDetails);
    for (const d of this.body.querySelectorAll('details')) if (expanded.includes(d.querySelector('summary').textContent)) d.open = true;
    this.body.scrollTop = scroll;
  }
}
