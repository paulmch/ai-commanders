/**
 * HudIndex - the spectator HUD's DOM-free view of a recording.
 *
 * Built once per parse / live append (BattleLoader.rebuildDerivedState) from
 * the sorted event list, so every HUD surface (combat log, command net, kill
 * feed, scoreboard, timeline markers) is a pure function of the playhead:
 * binary-search the precomputed, time-sorted rows instead of re-querying
 * events every frame. Rows carry the source event id, which is the identity
 * the DOM layers dedupe on - a parked playhead or a late same-time arrival
 * can never insert a row twice.
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

/**
 * [{t, n}] sorted. Numbered `checkpoint` events win; otherwise any decision
 * point (t > 0: the t=0 decision point precedes the first orders) counts in
 * order, CP1 first - the live HUD's numbering.
 */
export function buildCheckpoints(events) {
  const times = new Map();
  for (const e of events) {
    const t = e.timestamp;
    if (!(t > 0)) continue;
    if (e.event_type === 'checkpoint') {
      const n = Number(e.data?.checkpoint);
      times.set(t, Number.isFinite(n) ? n : times.get(t) ?? null);
    } else if (CHECKPOINT_SOURCES.has(e.event_type) && !times.has(t)) {
      times.set(t, null);
    }
  }
  return [...times.entries()].sort((a, b) => a[0] - b[0])
    .map(([t, n], i) => ({ t, n: n ?? i + 1 }));
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
  issues: null // status DEVIATE / UNABLE, errors and system rows (see commsMatches)
};

export function commsMatches(entry, faction = 'all', group = 'all') {
  if (faction !== 'all' && entry.faction !== faction && !entry.open && entry.faction) return false;
  if (group === 'all') return true;
  if (group === 'issues') return entry.kind === 'error' || entry.kind === 'system' ||
    entry.status === 'DEVIATE' || entry.status === 'UNABLE' || entry.priority === 'CRITICAL';
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

function complianceChips(c, ships) {
  if (!c) return [];
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

/**
 * Normalized command-net rows, time-sorted. Each row:
 * {id, eventId, t, cp, faction, open, kind, tag, from, fromId, to, toIds,
 *  text, reply, status, priority, chips[], msgId, deliverAt, shipId, decisionId}
 */
export function buildComms(events, ids, ships, checkpoints) {
  const rows = [];
  const seen = new Set();
  const cpOf = e => Number.isFinite(e.data?.checkpoint) ? e.data.checkpoint : checkpointAt(checkpoints, e.timestamp);
  // A structured captain_report supersedes the same captain's legacy
  // acknowledgment for that checkpoint (new recordings carry both).
  const reported = new Set();
  for (const e of events) {
    if (e.event_type === 'captain_report' && e.ship_id) reported.add(`${e.ship_id}:${cpOf(e)}`);
  }
  // Latest decision per ship and time, so a report row can open its decision
  const decisionAt = new Map();
  events.forEach((e, i) => {
    if (e.event_type === 'captain_decision') decisionAt.set(`${e.ship_id}:${e.timestamp}`, ids[i]);
  });

  const push = (e, i, row) => {
    const id = row.id || ids[i];
    if (seen.has(id)) return;
    seen.add(id);
    rows.push({
      eventId: ids[i], t: e.timestamp, cp: cpOf(e), faction: null, open: false, from: '', fromId: null,
      to: '', toIds: [], text: '', reply: '', status: null, priority: null, chips: [], msgId: null,
      deliverAt: null, shipId: null, decisionId: decisionAt.get(`${e.ship_id}:${e.timestamp}`) || null,
      ...row, id
    });
  };

  events.forEach((e, i) => {
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
        const text = d.reason || d.intent || d.request?.text || '';
        const extra = [];
        if (d.reason && d.intent && d.intent !== d.reason) extra.push(`Intent: ${d.intent}`);
        if (d.contact?.note) extra.push(`Contact: ${d.contact.note}`);
        if (d.request?.text && d.request.text !== text) extra.push(`Request: ${d.request.text}`);
        const chips = complianceChips(d.compliance, ships);
        if (d.contact?.ship) chips.push(`CONTACT ${d.contact.ship}${d.contact.assessment ? ' · ' + d.contact.assessment : ''}`);
        if (d.request?.kind) chips.push(`REQUEST ${String(d.request.kind).toUpperCase()}`);
        push(e, i, {
          kind: 'report', tag: status, status, faction: d.faction || factionOf(ship, ships),
          from: d.ship_name || nameOf(ship, ships) || d.captain_name || '', fromId: ship || null, shipId: ship || null,
          to: 'ADMIRAL', text, reply: extra.join(' · '), chips
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
        if (reported.has(`${ship}:${cpOf(e)}`)) break;
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
  });
  return rows;
}

/** msg_id -> [{t, recipientId}] sorted. */
export function buildDeliveries(events) {
  const out = new Map();
  for (const e of events) {
    if (e.event_type !== 'comms_delivered') continue;
    const d = e.data || {};
    if (!d.msg_id) continue;
    if (!out.has(d.msg_id)) out.set(d.msg_id, []);
    const list = out.get(d.msg_id);
    const recipientId = d.recipient_id || e.ship_id;
    if (!list.some(x => x.recipientId === recipientId)) {
      list.push({ t: Number.isFinite(d.delivered_t) ? d.delivered_t : e.timestamp, recipientId });
    }
  }
  for (const list of out.values()) list.sort((a, b) => a.t - b.t);
  return out;
}

/**
 * Delivery state of a comms row at time t: null (no routing recorded),
 * {state: 'transit'|'partial'|'delivered'|'undelivered', done, total, at}.
 * 'undelivered' only once the recording is complete (no live data to come).
 */
export function deliveryAt(row, deliveries, t, complete = false) {
  if (!row.msgId) return null;
  const list = deliveries.get(row.msgId) || [];
  const total = Math.max(row.toIds.length, list.length);
  const done = list.filter(x => x.t <= t + 1e-6);
  if (total === 0) return null;
  if (done.length >= total) return { state: 'delivered', done: done.length, total, at: done.at(-1).t };
  if (done.length > 0) return { state: 'partial', done: done.length, total, at: done.at(-1).t };
  if (complete && list.length === 0 && (row.deliverAt == null || t >= row.deliverAt)) return { state: 'undelivered', done: 0, total, at: null };
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
 * One record per lost ship, sorted by the moment it was knocked out:
 * {shipId, name, type, faction, t, dyingT, destroyedT, detonateAt, killerId,
 *  killerName, killerFaction, weapon, ordnanceId, cause, inferred, eventId}
 */
export function buildKills(events, ids, trace, ships) {
  const byShip = new Map();
  const rec = id => {
    if (!byShip.has(id)) byShip.set(id, { shipId: id, dyingT: null, destroyedT: null, detonateAt: null, killerId: null,
      ordnanceId: null, cause: null, inferred: false, eventId: null });
    return byShip.get(id);
  };
  events.forEach((e, i) => {
    const d = e.data || {};
    if (!e.ship_id) return;
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
  });
  // Pre-v3 recordings: deaths only exist as sim_trace flags
  const traceDying = new Map(), traceDead = new Map();
  for (const frame of trace || []) {
    for (const [id, s] of Object.entries(frame.ships || {})) {
      if (s.dying && !traceDying.has(id)) traceDying.set(id, frame.t);
      if (s.destroyed && !traceDead.has(id)) traceDead.set(id, frame.t);
    }
  }
  for (const [id, t] of traceDead) {
    const r = rec(id);
    r.destroyedT ??= t;
    if (r.dyingT == null && traceDying.has(id) && traceDying.get(id) < t) r.dyingT = traceDying.get(id);
  }
  for (const [id, t] of traceDying) {
    const r = rec(id);
    r.dyingT ??= t;
  }
  // Killer for kills the engine did not attribute: the last hit that landed
  // on the ship within a minute before it was knocked out (marked inferred).
  const hitsOn = new Map();
  for (const e of events) {
    const d = e.data || {};
    let target = null, shooter = null, ordnance = null;
    if (e.event_type === 'torpedo_impact') { target = d.target_id; shooter = e.ship_id; ordnance = d.torpedo_id; }
    else if (e.event_type === 'hit') { target = e.ship_id; shooter = d.shooter_id || d.source_ship_id; ordnance = d.projectile_id || 'proj'; }
    if (!target || !byShip.has(target)) continue;
    if (!hitsOn.has(target)) hitsOn.set(target, []);
    hitsOn.get(target).push({ t: e.timestamp, shooter, ordnance });
  }
  const out = [];
  for (const r of byShip.values()) {
    const t = r.dyingT ?? r.destroyedT;
    if (t == null) continue;
    if (!r.killerId && r.cause !== 'reactor_overheat') {
      const last = (hitsOn.get(r.shipId) || []).filter(h => h.t <= t + 0.5 && h.t >= t - 60).at(-1);
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

/** Time-sorted log rows {id, t, cls, cat, text, shipId}, plus inferred-kill rows for pre-v3 recordings. */
export function buildLog(events, ids, ships, kills) {
  const rows = [];
  const attributed = new Set();
  events.forEach((e, i) => {
    if (e.event_type === 'ship_dying' || e.event_type === 'ship_destroyed' || e.event_type === 'reactor_meltdown') attributed.add(e.ship_id);
    const f = formatLogEvent(e, ships);
    if (f) rows.push({ id: ids[i], t: e.timestamp, ...f, shipId: e.ship_id || null });
  });
  let synthetic = false;
  for (const k of kills) {
    if (attributed.has(k.shipId)) continue;
    synthetic = true;
    rows.push({ id: `kill:${k.shipId}`, t: k.t, cls: 'kill', cat: 'kills', text: `KILL: ${describeKill(k)}`, shipId: k.shipId });
  }
  if (synthetic) rows.sort((a, b) => a.t - b.t);
  return rows;
}

/* ------------------------------------------------------------------------ */
/* Timeline marks, spend, vision                                             */
/* ------------------------------------------------------------------------ */

/** Torpedo launches grouped per faction within `window` seconds: {t, faction, count, targets: [[name, n]], shipIds}. */
export function buildSalvos(events, ships, window = 3) {
  const open = new Map();
  const out = [];
  for (const e of events) {
    if (e.event_type !== 'torpedo_launched') continue;
    const faction = factionOf(e.ship_id, ships) || 'unknown';
    let s = open.get(faction);
    if (!s || e.timestamp - s.t > window) {
      s = { t: e.timestamp, faction, count: 0, targets: new Map(), shipIds: new Set() };
      open.set(faction, s);
      out.push(s);
    }
    s.count++;
    s.shipIds.add(e.ship_id);
    const target = nameOf(e.data?.target_id, ships) || 'no lock';
    s.targets.set(target, (s.targets.get(target) || 0) + 1);
  }
  return out.map(s => ({ ...s, targets: [...s.targets.entries()].sort((a, b) => b[1] - a[1]), shipIds: [...s.shipIds] }));
}

/** Per-faction hit counts in `bin`-second bins: {alpha: Int[], beta: Int[]} (faction = shooter). */
export function buildHitBins(events, ships, duration, bin = 5) {
  const n = Math.max(1, Math.ceil((duration || 0) / bin) + 1);
  const bins = { alpha: new Array(n).fill(0), beta: new Array(n).fill(0) };
  for (const e of events) {
    let shooter = null;
    if (e.event_type === 'hit') shooter = e.data?.shooter_id || e.data?.source_ship_id;
    else if (e.event_type === 'torpedo_impact') shooter = e.ship_id;
    else continue;
    const f = factionOf(shooter, ships);
    const k = Math.min(n - 1, Math.floor(e.timestamp / bin));
    if (bins[f]) bins[f][k]++;
  }
  return bins;
}

/** Cumulative reported model spend per faction: {alpha: [{t, total}], beta: [...]}. */
export function buildSpend(events) {
  const out = { alpha: [], beta: [] };
  const totals = { alpha: 0, beta: 0 };
  for (const e of events) {
    if (e.event_type !== 'model_call') continue;
    const f = e.data?.faction;
    const cost = Number(e.data?.usage?.cost);
    if (!(f in totals) || !(cost > 0)) continue;
    totals[f] += cost;
    out[f].push({ t: e.timestamp, total: totals[f] });
  }
  return out;
}

export function spendAt(spend, faction, t) {
  const list = spend?.[faction] || [];
  const i = upperBound(list, t) - 1;
  return i >= 0 ? list[i].total : 0;
}

/** Admiral tactical-plot images: [{t, faction, callId, ref, part}] (assets keep the data URL). */
export function buildVisions(events, ids, assets = {}) {
  const out = [];
  events.forEach((e, i) => {
    if (e.event_type !== 'model_call' || e.data?.actor !== 'admiral') return;
    for (const ref of e.data.message_refs || []) {
      const content = assets?.[ref]?.content;
      if (!Array.isArray(content)) continue;
      content.forEach((p, part) => {
        if (p?.type === 'image_url' && p.image_url?.url) out.push({ t: e.timestamp, faction: e.data.faction || null, callId: ids[i], ref, part });
      });
    }
  });
  return out;
}

/** The recorded result: {winner, reason, stopped} from metadata and battle_end / battle_stopped events. */
export function buildResult(events, meta = {}) {
  let stopped = null, ended = null;
  for (const e of events) {
    if (e.event_type === 'battle_stopped') stopped = { t: e.timestamp, reason: e.data?.reason || '' };
    else if (e.event_type === 'battle_end') ended = { t: e.timestamp, winner: e.data?.winner ?? null, reason: e.data?.reason || '' };
  }
  const reason = meta.resultReason || ended?.reason || stopped?.reason || '';
  const winner = meta.winner ?? ended?.winner ?? null;
  const halted = !!stopped || (!winner && /stopped|ceiling|budget|error|exception/i.test(reason));
  return { winner, reason, stopped: halted, known: !!(winner || reason || stopped || ended) };
}

/**
 * Everything the HUD needs, rebuilt from scratch per parse / live append.
 * @param {Object} loader - {events, ships, simTrace, metadata, sourceData, duration}
 */
export function buildHudIndex(loader) {
  const events = loader.events || [];
  const ships = loader.ships || {};
  const ids = events.map((e, i) => eventId(e, i));
  const checkpoints = buildCheckpoints(events);
  const kills = buildKills(events, ids, loader.simTrace, ships);
  const src = loader.sourceData || {};
  const models = {
    alpha: src.alpha_model || src.alpha_fleet?.admiral?.model || '',
    beta: src.beta_model || src.beta_fleet?.admiral?.model || ''
  };
  return {
    ids,
    checkpoints,
    comms: buildComms(events, ids, ships, checkpoints),
    deliveries: buildDeliveries(events),
    kills,
    log: buildLog(events, ids, ships, kills),
    salvos: buildSalvos(events, ships),
    hitBins: buildHitBins(events, ships, loader.duration),
    spend: buildSpend(events),
    visions: buildVisions(events, ids, src.assets),
    result: buildResult(events, loader.metadata || {}),
    models
  };
}
