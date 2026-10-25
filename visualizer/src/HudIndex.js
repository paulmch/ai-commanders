/**
 * HudIndex - the spectator HUD's DOM-free view of a recording.
 *
 * Built once per parse (BattleLoader.rebuildDerivedState) from the sorted
 * event list, and EXTENDED on each live append from the delta - the events
 * and frames that arrived, all sorting after everything already indexed
 * (appendHudIndex; LV-15: a poll costs what it brings, not the whole
 * battle). Either way the result is the same object a single parse of the
 * whole recording builds, field for field. Every HUD surface (combat log,
 * command net, kill feed, scoreboard, timeline markers) is then a pure
 * function of the playhead: binary-search the precomputed, time-sorted rows
 * instead of re-querying events every frame. Rows carry the source event
 * id, which is the identity the DOM layers dedupe on - a parked playhead or
 * a late same-time arrival can never insert a row twice.
 *
 * Understands both the structured comms schema (comms_message /
 * comms_delivered / captain_report) and every legacy shape (message,
 * admiral_directive / admiral_plan / admiral_order, captain_log,
 * captain_admiral_discussion, captain_decision acknowledgments). Recordings
 * without ship_dying / ship_destroyed events (pre-v3) get their kills from
 * the sim_trace destroyed flags, with the killer inferred from the last hit.
 *
 * All text here is plain data; renderers must write it with textContent.
 */
import { eventId } from './DecisionIndex.js';

/** Class weights of the engine's time-limit verdict (battle_runner._evaluate_fleet_tactical_advantage). */
export const SHIP_VALUES = {
  corvette: 10, frigate: 15, destroyer: 20, cruiser: 30,
  battlecruiser: 40, battleship: 50, dreadnought: 70
};
export const shipValue = type => SHIP_VALUES[String(type || '').toLowerCase()] ?? 20;

/** First index whose key is >= t (rows sorted by key). */
export function lowerBound(rows, t, key = r => r.t) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(rows[mid]) < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** First index whose key is > t: the number of rows at or before t. */
export function upperBound(rows, t, key = r => r.t) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(rows[mid]) <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

export const clockText = t => {
  const s = Math.max(0, Math.floor(t || 0));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

/** 'anthropic/claude-sonnet-5' -> 'claude-sonnet-5'. */
export function shortModel(model) {
  if (!model) return '';
  return String(model).split('/').pop().replace(/:(free|beta|online)$/, '');
}

const readable = v => String(v ?? '').replaceAll('_', ' ').toLowerCase();
const firstSentence = (text, max = 140) => {
  const s = String(text ?? '').replace(/\s+/g, ' ').trim();
  const cut = s.search(/[.!?](\s|$)/);
  const one = cut > 20 ? s.slice(0, cut + 1) : s;
  return one.length > max ? one.slice(0, max - 1) + '…' : one;
};
const clip = (text, max) => {
  const s = String(text ?? '').replace(/\s+/g, ' ').trim();
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
};

export function factionOf(id, ships = {}) {
  if (!id) return null;
  const f = ships[id]?.faction;
  if (f) return f;
  const s = String(id);
  return s.startsWith('alpha') ? 'alpha' : s.startsWith('beta') ? 'beta' : null;
}

const nameOf = (id, ships) => (id && (ships[id]?.name || id)) || '';

/** ACKNOWLEDGE / ack / Deviate -> ACK / DEVIATE / UNABLE / QUERY. */
export function normalizeStatus(status) {
  const s = String(status || '').toUpperCase();
  if (s.startsWith('ACK')) return 'ACK';
  if (s.startsWith('DEV')) return 'DEVIATE';
  if (s.startsWith('UNAB') || s === 'CANNOT' || s === 'FAILED') return 'UNABLE';
  if (s.startsWith('QUER') || s.startsWith('QUEST') || s === 'REQUEST') return 'QUERY';
  return s || 'ACK';
}

const REPORT_TOOLS = new Set(['report_to_admiral', 'respond_to_orders']);

/** {status, text} from a captain_decision's report tool call or acknowledgment, else null. */
export function legacyReport(decision) {
  const d = decision.data || {};
  const call = (d.tool_calls || []).find(c => REPORT_TOOLS.has(c?.name));
  if (call) {
    const a = call.arguments || {};
    const status = normalizeStatus(a.status || a.response_type);
    const text = a.reason || a.deviation_reason || a.intent || a.acknowledgment_note || a.note || a.text || '';
    return { status, text: String(text), intent: a.intent || a.acknowledgment_note || '' };
  }
  if (typeof d.acknowledgment === 'string' && d.acknowledgment.trim()) {
    const m = /^\s*(ACK\w*|DEVIATE\w*|UNABLE|QUERY)\s*[:\-–—]?\s*/i.exec(d.acknowledgment);
    return { status: normalizeStatus(m ? m[1] : 'ACK'), text: m ? d.acknowledgment.slice(m[0].length) : d.acknowledgment };
  }
  return null;
}

/* ------------------------------------------------------------------------ */
/* Checkpoints                                                               */
/* ------------------------------------------------------------------------ */

const CHECKPOINT_SOURCES = new Set(['decision_point_reached', 'captain_decision', 'admiral_directive', 'admiral_order']);

/** Fold events[from..] into checkpoint times: Map t -> recorded number | null. */
function addCheckpointTimes(times, events, from = 0) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    const t = e.timestamp;
    if (!(t > 0)) continue;
    if (e.event_type === 'checkpoint') {
      const n = Number(e.data?.checkpoint);
      times.set(t, Number.isFinite(n) ? n : times.get(t) ?? null);
    } else if (CHECKPOINT_SOURCES.has(e.event_type) && !times.has(t)) {
      times.set(t, null);
    }
  }
}

const checkpointList = times => [...times.entries()].sort((a, b) => a[0] - b[0])
  .map(([t, n], i) => ({ t, n: n ?? i + 1 }));

/**
 * [{t, n}] sorted. Numbered `checkpoint` events win; otherwise any decision
 * point (t > 0: the t=0 decision point precedes the first orders) counts in
 * order, CP1 first - the live HUD's numbering.
 */
export function buildCheckpoints(events) {
  const times = new Map();
  addCheckpointTimes(times, events);
  return checkpointList(times);
}

/** Checkpoint number in force at time t (0 before the first). */
export function checkpointAt(checkpoints, t) {
  const i = upperBound(checkpoints, t + 1e-6) - 1;
  return i >= 0 ? checkpoints[i].n : 0;
}

/* ------------------------------------------------------------------------ */
/* Command net                                                               */
/* ------------------------------------------------------------------------ */

const CHANNELS = {
  fleet_net: { kind: 'net', tag: 'FLEET NET' },
  direct: { kind: 'net', tag: 'DIRECT' },
  captain_to_admiral: { kind: 'report', tag: 'TO ADMIRAL' },
  admiral_to_captain: { kind: 'order', tag: 'FROM ADMIRAL' },
  enemy_broadcast: { kind: 'enemy', tag: 'OPEN CHANNEL' },
  admiral_diplomacy: { kind: 'enemy', tag: 'DIPLOMACY' }
};

/** Comms kinds -> filter groups of the COMMAND NET panel. */
export const COMMS_GROUPS = {
  command: new Set(['plan', 'directive', 'order']),
  reports: new Set(['report', 'log', 'discussion']),
  chatter: new Set(['net', 'enemy', 'message', 'engine']),
  issues: null // status DEVIATE / UNABLE, compliance deviations, errors and system rows (see commsMatches)
};

/**
 * Does a row pass the panel filter? `ctx` {deliveries, t} makes the ISSUES
 * group time-aware: an ACK whose compliance the admiral later reads as off
 * orders (comms_delivered.compliance) joins it once that copy is delivered.
 */
export function commsMatches(entry, faction = 'all', group = 'all', ctx = null) {
  if (faction !== 'all' && entry.faction !== faction && !entry.open && entry.faction) return false;
  if (group === 'all') return true;
  if (group === 'issues') {
    if (entry.kind === 'error' || entry.kind === 'system' ||
      entry.status === 'DEVIATE' || entry.status === 'UNABLE' || entry.priority === 'CRITICAL') return true;
    return !!(ctx ? complianceAt(entry, ctx.deliveries, ctx.t) : entry.compliance)?.deviated;
  }
  return COMMS_GROUPS[group]?.has(entry.kind) ?? true;
}

function recipientsText(ids, spec, ships) {
  if (ids?.length) {
    const names = ids.map(id => nameOf(id, ships));
    return names.length > 3 ? `${names.slice(0, 2).join(', ')} +${names.length - 2}` : names.join(', ');
  }
  if (spec) return String(spec).replaceAll('_', ' ').toUpperCase();
  return '';
}

function salvoChips(d, ships) {
  const chips = [];
  if (d.torpedo_salvo) {
    const target = d.torpedo_target ? ` → ${nameOf(d.torpedo_target, ships) || d.torpedo_target}` : '';
    chips.push(`SALVO ${d.torpedo_salvo}${target}`);
  } else if (d.torpedo_target) chips.push(`TORPEDOES → ${d.torpedo_target}`);
  if (d.suggested_target) chips.push(`TARGET ${d.suggested_target}`);
  return chips;
}

/** 'maneuver: ordered INTERCEPT, flew EVADE' -> 'MANEUVER: ordered INTERCEPT, flew EVADE'. */
const deviationChip = line => String(line).replace(/^\s*([a-z][\w ]*):/i, (_, label) => `${label.toUpperCase()}:`);

/**
 * Chips for a compliance block. The engine's verdict (`deviations`, one
 * line per mismatch, [] when the ship complied) is used whenever it was
 * recorded; only schema-only blocks without it fall back to comparing the
 * ordered/executed fields here.
 */
export function complianceChips(c, ships = {}) {
  if (!c) return [];
  if (Array.isArray(c.deviations)) return c.deviations.filter(Boolean).map(deviationChip);
  const chips = [];
  const same = (a, b) => a == null || b == null || String(a).toUpperCase() === String(b).toUpperCase();
  if (!same(c.ordered_maneuver, c.executed_maneuver)) chips.push(`ORDERED ${c.ordered_maneuver} · FLEW ${c.executed_maneuver}`);
  if (!same(c.ordered_target, c.actual_target)) {
    chips.push(`TARGET ${nameOf(c.ordered_target, ships) || c.ordered_target} → ${nameOf(c.actual_target, ships) || c.actual_target}`);
  }
  if (c.ordered_salvo != null && c.launched != null && Number(c.launched) !== Number(c.ordered_salvo)) {
    chips.push(`SALVO ${c.launched}/${c.ordered_salvo}`);
  }
  return chips;
}

/** {chips, deviated, final} summary of a compliance block (null when none was recorded). */
export function summarizeCompliance(c, ships = {}) {
  if (!c || typeof c !== 'object') return null;
  const chips = complianceChips(c, ships);
  return { chips, deviated: chips.length > 0, final: c.final === true };
}

/**
 * The compliance a report row shows at playhead t: the copy delivered to the
 * admiral (final: rounds that actually left the tubes over the interval)
 * once that delivery has happened, else the one filed with the report
 * (rounds committed). {chips, deviated, final, delivered, at} or null.
 */
export function complianceAt(row, deliveries, t = Infinity) {
  if (row.msgId && deliveries) {
    const list = deliveries.get(row.msgId) || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const x = list[i];
      if (x.compliance && x.t <= t + 1e-6) return { ...x.compliance, delivered: true, at: x.t };
    }
  }
  return row.compliance ? { ...row.compliance, delivered: false, at: row.t } : null;
}

const commsRow = (e, id) => ({
  eventId: id, t: e.timestamp, cp: 0, faction: null, open: false, from: '', fromId: null,
  to: '', toIds: [], text: '', reply: '', status: null, priority: null, chips: [], msgId: null,
  deliverAt: null, compliance: null, shipId: null, decisionId: null
});

/**
 * Normalized command-net rows for events[from, to), time-sorted. Each row:
 * {id, eventId, t, cp, faction, open, kind, tag, from, fromId, to, toIds,
 *  text, reply, status, priority, chips[], msgId, deliverAt, compliance,
 *  shipId, decisionId}
 *
 * Every cross-event link is local to one timestamp - a captain_report
 * replaces its captain's legacy acknowledgment row recorded at the same
 * time, a row's WHY opens the decision its ship recorded at that time - so
 * the rows of a timestamp depend only on the events at that timestamp (plus
 * the checkpoints before it). That is what lets a live append re-derive
 * just the latest timestamp (HudBuilder) and still match a full build.
 * @param {Object} o - {seen: ids already indexed (dedupe), interval: decision interval s}
 */
export function buildComms(events, ids, ships, checkpoints, { from = 0, to = events.length, seen = new Set(), interval = 30 } = {}) {
  const rows = [];
  const cpOf = e => Number.isFinite(e.data?.checkpoint) ? e.data.checkpoint : checkpointAt(checkpoints, e.timestamp);
  const at = e => `${e.ship_id}:${e.timestamp}`;
  // A structured captain_report supersedes the same captain's legacy
  // acknowledgment (new recordings carry both, at the same time), and the
  // latest decision per ship and time lets a row open its decision.
  const reported = new Set();
  const decisionAt = new Map();
  for (let i = from; i < to; i++) {
    const e = events[i];
    if (e.event_type === 'captain_report' && e.ship_id) reported.add(at(e));
    else if (e.event_type === 'captain_decision') decisionAt.set(at(e), ids[i]);
  }

  const push = (e, i, row) => {
    const id = row.id || ids[i];
    if (seen.has(id)) return;
    seen.add(id);
    rows.push({ ...commsRow(e, ids[i]), cp: cpOf(e), decisionId: decisionAt.get(at(e)) || null, ...row, id });
  };

  for (let i = from; i < to; i++) {
    const e = events[i];
    const d = e.data || {};
    const ship = e.ship_id;
    switch (e.event_type) {
      case 'admiral_plan':
        if (d.plan) push(e, i, { kind: 'plan', tag: 'PLAN', faction: d.faction || null, from: d.admiral_name || 'Admiral', to: 'STANDING PLAN', text: d.plan });
        break;
      case 'admiral_directive':
        if (d.directive) push(e, i, { kind: 'directive', tag: 'DIRECTIVE', faction: d.faction || null, from: d.admiral_name || 'Admiral', to: 'ALL SHIPS', text: d.directive });
        break;
      case 'admiral_order':
        push(e, i, {
          kind: 'order', tag: 'ORDER', faction: d.faction || factionOf(ship, ships), from: d.admiral_name || 'Admiral',
          to: nameOf(ship, ships) || d.ship_name || '', toIds: ship ? [ship] : [], shipId: ship || null,
          text: d.order_text || '', priority: d.priority ? String(d.priority).toUpperCase() : null, chips: salvoChips(d, ships)
        });
        break;
      case 'captain_report': {
        const status = normalizeStatus(d.status);
        const faction = d.faction || factionOf(ship, ships);
        const text = d.reason || d.intent || d.request?.text || (d.implicit && status === 'ACK' ? 'No report filed: implicit ACK, flying its orders.' : '');
        const extra = [];
        if (d.reason && d.intent && d.intent !== d.reason) extra.push(`Intent: ${d.intent}`);
        if (d.contact?.note) extra.push(`Contact: ${d.contact.note}`);
        if (d.request?.text && d.request.text !== text) extra.push(`Request: ${d.request.text}`);
        const chips = [];
        if (d.contact?.ship) chips.push(`CONTACT ${d.contact.ship}${d.contact.assessment ? ' · ' + d.contact.assessment : ''}`);
        if (d.request?.kind) chips.push(`REQUEST ${String(d.request.kind).toUpperCase()}`);
        // The report travels to the admiral, who reads it (with the engine's
        // compliance refreshed) at its next turn: comms_delivered msg_id =
        // report_id, recipient '<faction>_admiral'.
        const routed = !!(d.report_id && faction);
        push(e, i, {
          kind: 'report', tag: status, status, faction,
          from: d.ship_name || nameOf(ship, ships) || d.captain_name || '', fromId: ship || null, shipId: ship || null,
          to: 'ADMIRAL', toIds: routed ? [`${faction}_admiral`] : [], msgId: routed ? d.report_id : null,
          deliverAt: routed ? e.timestamp + interval : null, compliance: summarizeCompliance(d.compliance, ships),
          text, reply: extra.join(' · '), chips, decisionId: d.decision_event_id || decisionAt.get(at(e)) || null
        });
        break;
      }
      case 'captain_decision': {
        const faction = d.faction || factionOf(ship, ships);
        if (d.call_failed) {
          push(e, i, { id: `${ids[i]}:failed`, kind: 'error', tag: 'NO ORDERS', faction, from: d.ship_name || nameOf(ship, ships), fromId: ship, shipId: ship,
            text: 'Captain model call failed; the ship keeps its previous orders.', decisionId: ids[i] });
          break;
        }
        if (reported.has(at(e))) break;
        const report = legacyReport(e);
        if (report) {
          push(e, i, { id: `${ids[i]}:report`, kind: 'report', tag: report.status, status: report.status, faction,
            from: d.ship_name || nameOf(ship, ships) || d.captain_name || '', fromId: ship, shipId: ship, to: 'ADMIRAL',
            text: report.text, decisionId: ids[i] });
        }
        break;
      }
      case 'captain_log':
        if (d.note) push(e, i, { kind: 'log', tag: 'LOG', faction: d.faction || factionOf(ship, ships), from: nameOf(ship, ships) || d.captain_name || '', fromId: ship, shipId: ship, to: 'LOG', text: d.note });
        break;
      case 'captain_admiral_discussion':
        push(e, i, { kind: 'discussion', tag: 'QUERY', faction: d.faction || factionOf(ship, ships), from: nameOf(ship, ships) || d.captain_name || '', fromId: ship, shipId: ship,
          to: d.admiral_name || 'ADMIRAL', text: d.captain_question || '', reply: d.admiral_response ? `${d.admiral_name || 'Admiral'}: ${d.admiral_response}` : '' });
        break;
      case 'message': {
        // Pre-v4 send_message: no routing recorded. Most reached the enemy.
        const faction = factionOf(ship, ships);
        push(e, i, { kind: 'message', tag: 'TRANSMISSION', faction, open: true, from: d.sender_name || nameOf(ship, ships), fromId: ship, shipId: ship,
          to: d.recipient_name || d.recipient || 'OPEN CHANNEL', text: d.message || '' });
        break;
      }
      case 'comms_message': {
        const channel = CHANNELS[d.channel] || { kind: 'net', tag: readable(d.channel || 'comms').toUpperCase() };
        const engine = d.sender_role === 'engine';
        const faction = d.faction || factionOf(d.sender_id || ship, ships);
        const recipients = Array.isArray(d.recipient_ids) ? d.recipient_ids : [];
        const chips = [];
        if (d.kind && !['message', 'text', d.channel].includes(d.kind)) chips.push(readable(d.kind).toUpperCase());
        push(e, i, {
          kind: engine ? 'engine' : channel.kind, tag: engine ? 'AUTO' : channel.tag, faction, open: channel.kind === 'enemy',
          from: d.sender_name || nameOf(d.sender_id || ship, ships) || (engine ? 'Fleet systems' : ''),
          fromId: d.sender_id || ship || null, shipId: d.sender_id || ship || recipients[0] || null,
          to: recipientsText(recipients, d.recipient_spec, ships) || (channel.kind === 'enemy' ? 'ALL' : ''),
          toIds: recipients, text: d.text || '', chips, msgId: d.msg_id || null,
          deliverAt: Number.isFinite(d.deliver_at_t) ? d.deliver_at_t : null
        });
        break;
      }
      case 'model_error':
        push(e, i, { kind: 'error', tag: 'MODEL ERROR', faction: d.faction || factionOf(ship, ships), from: ship ? nameOf(ship, ships) : `${d.faction || ''} admiral`.trim(),
          fromId: ship || null, shipId: ship || null, text: clip(d.error, 400) });
        break;
      case 'battle_stopped':
        push(e, i, { kind: 'system', tag: 'STOPPED', from: 'Battle control', text: d.reason || 'Battle stopped' });
        break;
      case 'surrender':
        push(e, i, { kind: 'system', tag: 'SURRENDER', faction: factionOf(ship, ships), from: d.captain_name || nameOf(ship, ships), fromId: ship, shipId: ship, text: `${nameOf(ship, ships)} surrenders.` });
        break;
      case 'draw_proposal':
        push(e, i, { kind: 'system', tag: 'DRAW OFFER', faction: factionOf(ship, ships), from: d.captain_name || nameOf(ship, ships), fromId: ship, shipId: ship, text: d.message || 'Proposes a draw.' });
        break;
    }
  }
  return rows;
}

/** Fold comms_delivered events[from..] into msg_id -> [{t, recipientId, compliance}] (re-sorting touched lists). */
function addDeliveries(out, events, from = 0, ships = {}) {
  const touched = new Set();
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    if (e.event_type !== 'comms_delivered') continue;
    const d = e.data || {};
    if (!d.msg_id) continue;
    if (!out.has(d.msg_id)) out.set(d.msg_id, []);
    const list = out.get(d.msg_id);
    const recipientId = d.recipient_id || e.ship_id;
    if (!list.some(x => x.recipientId === recipientId)) {
      // A captain report's delivery carries the compliance as the admiral read it
      list.push({ t: Number.isFinite(d.delivered_t) ? d.delivered_t : e.timestamp, recipientId,
        compliance: summarizeCompliance(d.compliance, ships) });
      touched.add(list);
    }
  }
  for (const list of touched) list.sort((a, b) => a.t - b.t);
  return out;
}

/** msg_id -> [{t, recipientId, compliance}] sorted. */
export function buildDeliveries(events, ships = {}) {
  return addDeliveries(new Map(), events, 0, ships);
}

/**
 * Delivery state of a comms row at time t: null (no routing recorded),
 * {state: 'transit'|'partial'|'delivered'|'undelivered', done, total, at}.
 * 'undelivered' only once the recording is complete (no live data to come)
 * and the due time - or the end of the recording, for traffic due after the
 * battle ended - has passed.
 */
export function deliveryAt(row, deliveries, t, complete = false, end = Infinity) {
  if (!row.msgId) return null;
  const list = deliveries.get(row.msgId) || [];
  const total = Math.max(row.toIds.length, list.length);
  const done = list.filter(x => x.t <= t + 1e-6);
  if (total === 0) return null;
  if (done.length >= total) return { state: 'delivered', done: done.length, total, at: done.at(-1).t };
  if (done.length > 0) return { state: 'partial', done: done.length, total, at: done.at(-1).t };
  if (complete && list.length === 0 && (row.deliverAt == null || t >= Math.min(row.deliverAt, end) - 1e-6)) {
    return { state: 'undelivered', done: 0, total, at: null };
  }
  return { state: 'transit', done: 0, total, at: row.deliverAt };
}

/* ------------------------------------------------------------------------ */
/* Kills                                                                     */
/* ------------------------------------------------------------------------ */

const weaponOf = ordnance => {
  const s = String(ordnance || '');
  return s.startsWith('torp') ? 'torpedo' : s.startsWith('proj') || s.startsWith('slug') ? 'coilgun' : null;
};

/**
 * Kill bookkeeping as three append-only folds - death events, first dying /
 * destroyed flags per ship in the trace, hits per target - combined into
 * the kill list on demand (a handful of lost ships).
 */
class KillFold {
  constructor() {
    this.byShip = new Map();
    this.traceDying = new Map();
    this.traceDead = new Map();
    this.hitsOn = new Map();
  }

  events(events, ids, from = 0) {
    const rec = id => {
      if (!this.byShip.has(id)) this.byShip.set(id, { shipId: id, dyingT: null, destroyedT: null, detonateAt: null, killerId: null,
        ordnanceId: null, cause: null, inferred: false, eventId: null });
      return this.byShip.get(id);
    };
    for (let i = from; i < events.length; i++) {
      const e = events[i];
      const d = e.data || {};
      // Every landed hit, for the killers the engine did not attribute
      let target = null, shooter = null, ordnance = null;
      if (e.event_type === 'torpedo_impact') { target = d.target_id; shooter = e.ship_id; ordnance = d.torpedo_id; }
      else if (e.event_type === 'hit') { target = e.ship_id; shooter = d.shooter_id || d.source_ship_id; ordnance = d.projectile_id || 'proj'; }
      if (target) {
        if (!this.hitsOn.has(target)) this.hitsOn.set(target, []);
        this.hitsOn.get(target).push({ t: e.timestamp, shooter, ordnance });
      }
      if (!e.ship_id) continue;
      if (e.event_type === 'ship_dying') {
        const r = rec(e.ship_id);
        if (r.dyingT == null) {
          r.dyingT = e.timestamp;
          if (Number.isFinite(d.detonation_in_s)) r.detonateAt = e.timestamp + d.detonation_in_s;
          r.killerId ??= d.killer_id || null;
          r.ordnanceId ??= d.ordnance_id || null;
          r.eventId ??= ids[i];
        }
      } else if (e.event_type === 'ship_destroyed') {
        const r = rec(e.ship_id);
        if (r.destroyedT == null) {
          r.destroyedT = e.timestamp;
          r.cause = d.cause || r.cause;
          r.killerId ??= d.killer_id || null;
          r.ordnanceId ??= d.ordnance_id || null;
          r.eventId ??= ids[i];
        }
      } else if (e.event_type === 'reactor_meltdown') {
        const r = rec(e.ship_id);
        r.cause = 'reactor_overheat';
        r.destroyedT ??= e.timestamp;
        r.eventId ??= ids[i];
      }
    }
  }

  /** Pre-v3 recordings: deaths only exist as sim_trace flags. */
  frames(frames) {
    for (const frame of frames || []) {
      for (const [id, s] of Object.entries(frame.ships || {})) {
        if (s.dying && !this.traceDying.has(id)) this.traceDying.set(id, frame.t);
        if (s.destroyed && !this.traceDead.has(id)) this.traceDead.set(id, frame.t);
      }
    }
  }

  kills(ships) {
    const byShip = new Map([...this.byShip].map(([id, r]) => [id, { ...r }]));
    const rec = id => {
      if (!byShip.has(id)) byShip.set(id, { shipId: id, dyingT: null, destroyedT: null, detonateAt: null, killerId: null,
        ordnanceId: null, cause: null, inferred: false, eventId: null });
      return byShip.get(id);
    };
    for (const [id, t] of this.traceDead) {
      const r = rec(id);
      r.destroyedT ??= t;
      if (r.dyingT == null && this.traceDying.has(id) && this.traceDying.get(id) < t) r.dyingT = this.traceDying.get(id);
    }
    for (const [id, t] of this.traceDying) {
      const r = rec(id);
      r.dyingT ??= t;
    }
    const out = [];
    for (const r of byShip.values()) {
      const t = r.dyingT ?? r.destroyedT;
      if (t == null) continue;
      // Killer for kills the engine did not attribute: the last hit that landed
      // on the ship within a minute before it was knocked out (marked inferred).
      if (!r.killerId && r.cause !== 'reactor_overheat') {
        const last = (this.hitsOn.get(r.shipId) || []).filter(h => h.t <= t + 0.5 && h.t >= t - 60).at(-1);
        if (last?.shooter) { r.killerId = last.shooter; r.ordnanceId ??= last.ordnance; r.inferred = true; }
      }
      const info = ships[r.shipId] || {};
      out.push({
        ...r, t, name: info.name || r.shipId, type: info.type || '', faction: factionOf(r.shipId, ships),
        killerName: r.killerId ? nameOf(r.killerId, ships) : '', killerFaction: factionOf(r.killerId, ships),
        weapon: r.cause === 'reactor_overheat' ? 'meltdown' : weaponOf(r.ordnanceId)
      });
    }
    return out.sort((a, b) => a.t - b.t);
  }
}

/**
 * One record per lost ship, sorted by the moment it was knocked out:
 * {shipId, name, type, faction, t, dyingT, destroyedT, detonateAt, killerId,
 *  killerName, killerFaction, weapon, ordnanceId, cause, inferred, eventId}
 */
export function buildKills(events, ids, trace, ships) {
  const fold = new KillFold();
  fold.events(events, ids);
  fold.frames(trace);
  return fold.kills(ships);
}

/**
 * State of a kill at playhead t: null (not yet), {state: 'dying', in: s}
 * or {state: 'destroyed'}. The recorded destruction time wins over the
 * predicted fuse (hits on a hulk shorten it).
 */
export function killStateAt(kill, t) {
  if (t < kill.t - 1e-6) return null;
  if (kill.destroyedT != null && t >= kill.destroyedT - 1e-6) return { state: 'destroyed', in: 0 };
  const end = kill.destroyedT ?? kill.detonateAt;
  return { state: 'dying', in: end != null ? Math.max(0, end - t) : null };
}

export function describeKill(kill) {
  const by = kill.killerName ? ` by ${kill.killerName}` : '';
  const how = kill.weapon === 'meltdown' ? ' (reactor meltdown)'
    : kill.weapon ? ` (${kill.weapon}${kill.cause && kill.cause !== 'reactor_overheat' ? ', ' + readable(kill.cause) : ''})`
      : kill.cause ? ` (${readable(kill.cause)})` : '';
  return `${kill.name}${kill.weapon === 'meltdown' ? '' : by}${how}${kill.inferred ? ' · probable' : ''}`;
}

/* ------------------------------------------------------------------------ */
/* Combat log                                                                */
/* ------------------------------------------------------------------------ */

/** Log categories for the filter chips. */
export const LOG_FILTERS = ['all', 'kills', 'fire', 'pd', 'command'];

/** {cls, cat, text} for a loggable event, else null. Plain text only. */
export function formatLogEvent(e, ships = {}) {
  const d = e.data || {};
  const n = id => nameOf(id, ships) || '?';
  const gj = v => (Number(v) || 0).toFixed(1);
  switch (e.event_type) {
    case 'hit': return { cls: 'hit', cat: 'fire', text: `HIT: ${n(d.shooter_id || d.source_ship_id)} strikes ${n(e.ship_id)} ${d.hit_location || ''} (${gj(d.kinetic_energy_gj)} GJ)` };
    case 'miss': return { cls: 'shot', cat: 'fire', text: `MISS: ${n(e.ship_id)} shot goes wide` };
    case 'module_destroyed': return { cls: 'destroyed', cat: 'fire', text: `DESTROYED: ${n(e.ship_id)} lost ${d.module_name}` };
    case 'module_damaged': return { cls: 'hit', cat: 'fire', text: `DAMAGE: ${n(e.ship_id)} ${d.module_name} (${gj(d.damage_gj)} GJ)` };
    case 'shot_fired': return { cls: 'shot', cat: 'fire', text: `FIRE: ${n(e.ship_id)} → ${n(d.target_id)}` };
    case 'torpedo_launched': return { cls: 'torpedo', cat: 'fire', text: `TORPEDO AWAY: ${n(e.ship_id)} → ${n(d.target_id)}` };
    case 'torpedo_impact': return { cls: 'destroyed', cat: 'fire', text: `TORPEDO IMPACT: ${n(d.target_id)} takes ${(Number(d.total_damage_gj) || 0).toFixed(0)} GJ (${d.hit_location || '?'})` };
    case 'torpedo_miss': return { cls: 'torpedo', cat: 'fire', text: `TORPEDO MISS: closest ${(Number(d.closest_approach_km) || 0).toFixed(1)} km` };
    case 'torpedo_fuel_exhausted': return { cls: 'torpedo', cat: 'fire', text: `BURNOUT: torpedo from ${n(e.ship_id)} ballistic` };
    case 'torpedo_retargeted': return { cls: 'torpedo', cat: 'fire', text: `RETARGET: torpedo from ${n(e.ship_id)} swings onto ${n(d.new_target_id)}` };
    case 'pd_torpedo_disabled': return { cls: 'pd', cat: 'pd', text: `SEEKER KILL: ${n(e.ship_id)} PD blinds inbound torpedo` };
    case 'pd_slug_destroyed': return { cls: 'pd', cat: 'pd', text: `PD KILL: ${n(e.ship_id)} vaporizes incoming slug` };
    case 'penetration': return { cls: 'destroyed', cat: 'fire', text: `ARMOR BREACH: ${n(e.ship_id)} ${d.location || ''}` };
    case 'radiator_change': return { cls: 'info', cat: 'info', text: `${n(e.ship_id)} ${d.extended ? 'extends' : 'retracts'} radiators` };
    case 'heat_warning': return { cls: 'hit', cat: 'info', text: `HEAT ${Math.round((d.level || 0) * 100)}%: ${n(e.ship_id)} ${d.level >= 1.25 ? 'nearing meltdown' : d.level >= 1 ? 'radiators auto-extend' : 'guns hold fire'}` };
    case 'reactor_meltdown': return { cls: 'kill', cat: 'kills', text: `MELTDOWN: ${n(e.ship_id)} reactor containment fails (heat ${Math.round((d.heat || 0) * 100)}%)` };
    case 'ship_dying': {
      const weapon = weaponOf(d.ordnance_id);
      const by = d.killer_id ? ` by ${n(d.killer_id)}${weapon ? ` (${weapon})` : ''}` : '';
      const fuse = Number.isFinite(d.detonation_in_s) ? ` · reactor in ~${Math.round(d.detonation_in_s)}s` : '';
      return { cls: 'kill', cat: 'kills', text: `CRIPPLED: ${n(e.ship_id)}${by}${fuse}` };
    }
    case 'ship_destroyed': return { cls: 'kill', cat: 'kills', text: `KILL: ${n(e.ship_id)} destroyed${d.cause ? ` (${readable(d.cause)})` : ''}${d.killer_id ? ` · ${n(d.killer_id)}` : ''}` };
    case 'maneuver': return { cls: 'info', cat: 'command', text: `${n(e.ship_id)}: ${d.maneuver_type}${d.throttle != null ? ` @ ${Math.round(d.throttle * 100)}%` : ''}` };
    case 'admiral_directive': return { cls: 'orders', cat: 'command', text: `ORDERS ${String(d.faction || '').toUpperCase()}: ${firstSentence(d.directive)}` };
    case 'captain_report': return { cls: d.status && normalizeStatus(d.status) !== 'ACK' ? 'deviate' : 'comms', cat: 'command', text: `${normalizeStatus(d.status)} ${d.ship_name || n(e.ship_id)}: ${clip(d.reason || d.intent || '', 140)}` };
    case 'captain_log': return { cls: 'comms', cat: 'command', text: `LOG ${n(e.ship_id)}: "${clip(d.note, 160)}"` };
    case 'message': return { cls: 'comms', cat: 'command', text: `COMMS ${d.sender_name || n(e.ship_id)}: "${clip(d.message, 200)}"` };
    case 'comms_message': return { cls: 'comms', cat: 'command', text: `COMMS ${d.sender_name || n(d.sender_id || e.ship_id)}${d.channel ? ` [${readable(d.channel)}]` : ''}: "${clip(d.text, 200)}"` };
    case 'model_error': return { cls: 'system', cat: 'command', text: `MODEL ERROR (${d.actor || 'model'}${e.ship_id ? ' ' + n(e.ship_id) : ''}): ${clip(d.error, 160)}` };
    case 'battle_stopped': return { cls: 'system', cat: 'kills', text: `BATTLE STOPPED: ${clip(d.reason, 200)}` };
    case 'battle_end': return { cls: 'system', cat: 'kills', text: `BATTLE OVER: ${d.winner ? String(d.winner).toUpperCase() + ' WINS' : 'DRAW'}${d.reason ? ' · ' + clip(d.reason, 160) : ''}` };
    default: return null;
  }
}

const DEATH_EVENTS = new Set(['ship_dying', 'ship_destroyed', 'reactor_meltdown']);

/** Fold events[from..] into log rows (append-only) and the set of ships whose death an event reports. */
function addLogRows(rows, attributed, events, ids, from, ships) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    if (DEATH_EVENTS.has(e.event_type)) attributed.add(e.ship_id);
    const f = formatLogEvent(e, ships);
    if (f) rows.push({ id: ids[i], t: e.timestamp, ...f, shipId: e.ship_id || null });
  }
}

/**
 * The log: event rows plus a synthetic row for every kill no event reports
 * (pre-v3 recordings), merged in time order - a kill row after the event
 * rows of its second. Returns the event-row array itself when there is
 * nothing to merge (every v3+ recording), so a live append stays O(delta).
 */
function mergeKillRows(rows, attributed, kills) {
  const synthetic = kills.filter(k => !attributed.has(k.shipId))
    .map(k => ({ id: `kill:${k.shipId}`, t: k.t, cls: 'kill', cat: 'kills', text: `KILL: ${describeKill(k)}`, shipId: k.shipId }));
  if (!synthetic.length) return rows;
  const out = [];
  let j = 0;
  for (const r of rows) {
    while (j < synthetic.length && synthetic[j].t < r.t) out.push(synthetic[j++]);
    out.push(r);
  }
  while (j < synthetic.length) out.push(synthetic[j++]);
  return out;
}

/** Time-sorted log rows {id, t, cls, cat, text, shipId}, plus inferred-kill rows for pre-v3 recordings. */
export function buildLog(events, ids, ships, kills) {
  const rows = [];
  const attributed = new Set();
  addLogRows(rows, attributed, events, ids, 0, ships);
  return mergeKillRows(rows, attributed, kills);
}

/* ------------------------------------------------------------------------ */
/* Timeline marks, spend, vision                                             */
/* ------------------------------------------------------------------------ */

/** Torpedo launches grouped per faction within `window` seconds (fold: open groups stay extendable). */
class SalvoFold {
  constructor(window = 3) {
    this.window = window;
    this.open = new Map();
    this.groups = [];
  }

  add(events, from, ships) {
    for (let i = from; i < events.length; i++) {
      const e = events[i];
      if (e.event_type !== 'torpedo_launched') continue;
      const faction = factionOf(e.ship_id, ships) || 'unknown';
      let s = this.open.get(faction);
      if (!s || e.timestamp - s.t > this.window) {
        s = { t: e.timestamp, faction, count: 0, targets: new Map(), shipIds: new Set() };
        this.open.set(faction, s);
        this.groups.push(s);
      }
      s.count++;
      s.shipIds.add(e.ship_id);
      const target = nameOf(e.data?.target_id, ships) || 'no lock';
      s.targets.set(target, (s.targets.get(target) || 0) + 1);
    }
  }

  salvos() {
    return this.groups.map(s => ({ ...s, targets: [...s.targets.entries()].sort((a, b) => b[1] - a[1]), shipIds: [...s.shipIds] }));
  }
}

/** Torpedo launches grouped per faction within `window` seconds: {t, faction, count, targets: [[name, n]], shipIds}. */
export function buildSalvos(events, ships, window = 3) {
  const fold = new SalvoFold(window);
  fold.add(events, 0, ships);
  return fold.salvos();
}

/** Fold hits in events[from..] into raw per-faction counts per `bin`-second bin (faction = shooter). */
function addHits(raw, events, from, ships, bin) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    let shooter = null;
    if (e.event_type === 'hit') shooter = e.data?.shooter_id || e.data?.source_ship_id;
    else if (e.event_type === 'torpedo_impact') shooter = e.ship_id;
    else continue;
    const f = factionOf(shooter, ships);
    if (!raw[f]) continue;
    const k = Math.max(0, Math.floor(e.timestamp / bin));
    raw[f][k] = (raw[f][k] || 0) + 1;
  }
}

/** Raw bin counts laid over the recording: ceil(duration / bin) + 1 bins, later hits in the last. */
function hitBinsFor(raw, duration, bin) {
  const n = Math.max(1, Math.ceil((duration || 0) / bin) + 1);
  const bins = { alpha: new Array(n).fill(0), beta: new Array(n).fill(0) };
  for (const f of ['alpha', 'beta']) {
    const counts = raw[f];
    for (let k = 0; k < counts.length; k++) if (counts[k]) bins[f][Math.min(n - 1, k)] += counts[k];
  }
  bins.bin = bin;   // seconds per bin: the chart spans bins * bin, not the timeline's epilogue
  return bins;
}

/** Per-faction hit counts in `bin`-second bins: {alpha: Int[], beta: Int[]} (faction = shooter). */
export function buildHitBins(events, ships, duration, bin = 5) {
  const raw = { alpha: [], beta: [] };
  addHits(raw, events, 0, ships, bin);
  return hitBinsFor(raw, duration, bin);
}

/** Fold model_call costs in events[from..] into cumulative per-faction spend (arrays extended in place). */
function addSpend(out, totals, events, from = 0) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    if (e.event_type !== 'model_call') continue;
    const f = e.data?.faction;
    const cost = Number(e.data?.usage?.cost);
    if (!(f in totals) || !(cost > 0)) continue;
    totals[f] += cost;
    out[f].push({ t: e.timestamp, total: totals[f] });
  }
}

/** Cumulative reported model spend per faction: {alpha: [{t, total}], beta: [...]}. */
export function buildSpend(events) {
  const out = { alpha: [], beta: [] };
  addSpend(out, { alpha: 0, beta: 0 }, events);
  return out;
}

export function spendAt(spend, faction, t) {
  const list = spend?.[faction] || [];
  const i = upperBound(list, t) - 1;
  return i >= 0 ? list[i].total : 0;
}

/** Admiral model calls with message refs in events[from..]: [{t, faction, callId, refs}]. */
function addAdmiralCalls(out, events, ids, from = 0) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    if (e.event_type !== 'model_call' || e.data?.actor !== 'admiral' || !e.data.message_refs?.length) continue;
    out.push({ t: e.timestamp, faction: e.data.faction || null, callId: ids[i], refs: e.data.message_refs });
  }
}

/** Images among the calls' messages, resolved against the assets held now (they can arrive after the call). */
function visionsOf(calls, assets = {}) {
  const out = [];
  for (const call of calls) {
    for (const ref of call.refs) {
      const content = assets?.[ref]?.content;
      if (!Array.isArray(content)) continue;
      content.forEach((p, part) => {
        if (p?.type === 'image_url' && p.image_url?.url) out.push({ t: call.t, faction: call.faction, callId: call.callId, ref, part });
      });
    }
  }
  return out;
}

/** Admiral tactical-plot images: [{t, faction, callId, ref, part}] (assets keep the data URL). */
export function buildVisions(events, ids, assets = {}) {
  const calls = [];
  addAdmiralCalls(calls, events, ids);
  return visionsOf(calls, assets);
}

/** Last battle_stopped / battle_end in events[from..], folded into `end` {stopped, ended}. */
function addEnd(end, events, from = 0) {
  for (let i = from; i < events.length; i++) {
    const e = events[i];
    if (e.event_type === 'battle_stopped') end.stopped = { t: e.timestamp, reason: e.data?.reason || '' };
    else if (e.event_type === 'battle_end') end.ended = { t: e.timestamp, winner: e.data?.winner ?? null, reason: e.data?.reason || '' };
  }
}

function resultOf({ stopped, ended }, meta = {}) {
  const reason = meta.resultReason || ended?.reason || stopped?.reason || '';
  const winner = meta.winner ?? ended?.winner ?? null;
  const halted = !!stopped || (!winner && /stopped|ceiling|budget|error|exception/i.test(reason));
  return { winner, reason, stopped: halted, known: !!(winner || reason || stopped || ended) };
}

/** The recorded result: {winner, reason, stopped} from metadata and battle_end / battle_stopped events. */
export function buildResult(events, meta = {}) {
  const end = { stopped: null, ended: null };
  addEnd(end, events);
  return resultOf(end, meta);
}

/* ------------------------------------------------------------------------ */
/* The index: full build and live append                                     */
/* ------------------------------------------------------------------------ */

/**
 * Incremental state behind one loader's hud. Every field is a fold over the
 * events (and frames) in order, so indexing them in one pass or in any
 * number of in-order appends ends in the same state. The command net is the
 * exception that proves it: rows at the latest timestamp can still change
 * (that checkpoint's orders and reports trickle in over several polls), so
 * they stay "open" and are re-derived with each append; rows of earlier
 * timestamps are final.
 */
class HudBuilder {
  constructor(loader) {
    this.shipsRef = loader.ships;
    this.ships = loader.ships || {};
    this.eventsRef = loader.events;
    this.count = 0;           // events indexed
    this.ids = [];
    this.cpTimes = new Map();
    this.checkpoints = [];
    this.comms = [];
    this.commsSeen = new Set(); // ids of final (frozen) rows
    this.openFrom = 0;        // first event of the open (latest) timestamp
    this.frozenRows = 0;      // comms rows before the open timestamp
    this.deliveries = new Map();
    this.killFold = new KillFold();
    this.logRows = [];
    this.attributed = new Set();
    this.salvoFold = new SalvoFold();
    this.hitRaw = { alpha: [], beta: [] };
    this.spend = { alpha: [], beta: [] };
    this.spendTotals = { alpha: 0, beta: 0 };
    this.admiralCalls = [];
    this.end = { stopped: null, ended: null };
  }

  /** Is `delta` (the last `n` events of the loader) a clean append to what this builder indexed? */
  canAppend(loader, n) {
    const events = loader.events || [];
    return loader.ships === this.shipsRef && events === this.eventsRef &&
      events.length - n === this.count && (this.count === 0 || events[this.count - 1] === this.last);
  }

  /** Index events[count..] and `frames` (all sorting after what is indexed). */
  add(loader, frames) {
    const events = loader.events || [];
    const ships = this.ships;
    const from = this.count;
    for (let i = from; i < events.length; i++) this.ids.push(eventId(events[i], i));
    const ids = this.ids;
    addCheckpointTimes(this.cpTimes, events, from);
    this.checkpoints = checkpointList(this.cpTimes);
    addDeliveries(this.deliveries, events, from, ships);
    this.killFold.events(events, ids, from);
    this.killFold.frames(frames);
    addLogRows(this.logRows, this.attributed, events, ids, from, ships);
    this.salvoFold.add(events, from, ships);
    addHits(this.hitRaw, events, from, ships, HIT_BIN_S);
    addSpend(this.spend, this.spendTotals, events, from);
    addAdmiralCalls(this.admiralCalls, events, ids, from);
    addEnd(this.end, events, from);

    // Command net: drop the open rows, re-derive them together with the new events
    this.comms.length = this.frozenRows;
    const local = new Set();
    const seen = { has: id => this.commsSeen.has(id) || local.has(id), add: id => local.add(id) };
    const interval = Number(loader.sourceData?.decision_interval_s) || 30;
    for (const row of buildComms(events, ids, ships, this.checkpoints, { from: this.openFrom, seen, interval })) this.comms.push(row);
    this.count = events.length;
    this.last = events[events.length - 1];
    if (this.count) {
      // Everything within a microsecond of the newest event stays open
      // (checkpointAt resolves a row's checkpoint with that tolerance)
      const cut = this.last.timestamp - 1e-6;
      this.openFrom = lowerBound(events, cut, e => e.timestamp);
      const frozen = lowerBound(this.comms, cut);
      for (let i = this.frozenRows; i < frozen; i++) this.commsSeen.add(this.comms[i].id);
      this.frozenRows = frozen;
    }
  }

  /** The hud object: a new one per build / append (panels key on its identity), sharing the append-only arrays. */
  snapshot(loader) {
    const src = loader.sourceData || {};
    const models = {
      alpha: src.alpha_model || src.alpha_fleet?.admiral?.model || '',
      beta: src.beta_model || src.beta_fleet?.admiral?.model || ''
    };
    // Draft point cost per ship (provenance.draft), when the battle was drafted
    const costs = {};
    for (const f of ['alpha', 'beta']) {
      for (const s of src.provenance?.draft?.[f]?.ships || []) if (s?.ship_id && Number.isFinite(s.cost)) costs[s.ship_id] = s.cost;
    }
    const kills = this.killFold.kills(this.ships);
    return {
      ids: this.ids,
      costs,
      checkpoints: this.checkpoints,
      comms: this.comms,
      deliveries: this.deliveries,
      kills,
      log: mergeKillRows(this.logRows, this.attributed, kills),
      salvos: this.salvoFold.salvos(),
      hitBins: hitBinsFor(this.hitRaw, loader.duration, HIT_BIN_S),
      spend: this.spend,
      visions: visionsOf(this.admiralCalls, src.assets),
      result: resultOf(this.end, loader.metadata || {}),
      models,
      duration: loader.duration || 0
    };
  }
}

const HIT_BIN_S = 5;

/** loader -> its HudBuilder. Kept off the loader and off loader.hud: both are plain, comparable data. */
const builders = new WeakMap();

/**
 * Everything the HUD needs, built from scratch (parse, or a live poll that
 * brought events older than ones already held).
 * @param {Object} loader - {events, ships, simTrace, metadata, sourceData, duration}
 */
export function buildHudIndex(loader) {
  const builder = new HudBuilder(loader);
  builder.add(loader, loader.simTrace || []);
  if (loader && typeof loader === 'object') builders.set(loader, builder);
  return builder.snapshot(loader);
}

/**
 * Extend the loader's index with a live append. `delta` = {events, frames}:
 * what arrived, already pushed onto loader.events / loader.simTrace, all of
 * it sorting after everything indexed (BattleLoader.appendLiveData passes
 * null otherwise). Costs O(delta + the open timestamp), not O(battle); any
 * mismatch with what the builder indexed falls back to a full build.
 */
export function appendHudIndex(loader, delta) {
  const builder = builders.get(loader);
  if (!delta || !builder || !builder.canAppend(loader, delta.events?.length || 0)) return buildHudIndex(loader);
  builder.add(loader, delta.frames || []);
  return builder.snapshot(loader);
}
