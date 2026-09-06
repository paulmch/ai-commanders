/** Seekable intent/execution index. Legacy chronology never becomes attribution. */
const OUTCOMES = new Set(['hit', 'miss', 'torpedo_impact', 'torpedo_miss', 'torpedo_intercepted',
  'pd_slug_destroyed', 'pd_torpedo_destroyed', 'module_damaged', 'module_destroyed',
  'ship_dying', 'ship_destroyed', 'armor_damage', 'penetration']);
const LAUNCHES = new Set(['shot_fired', 'torpedo_launched']);
const TERMINAL = new Set(['hit', 'miss', 'torpedo_impact', 'torpedo_miss', 'torpedo_intercepted',
  'pd_slug_destroyed', 'pd_torpedo_destroyed', 'projectile_expired']);
export const eventId = (e, i = 0) => e.event_id || `legacy-${i}`;
const objectId = e => e.data.projectile_id || e.data.torpedo_id || e.data.ordnance_id;

export class DecisionIndex {
  constructor(events, ships = {}) {
    this.ships = ships;
    this.events = events.map((e, i) => ({...e, event_id: eventId(e, i), data: e.data || {}}));
    this.byId = new Map(this.events.map(e => [e.event_id, e]));
    this.decisions = this.events.filter(e => e.event_type === 'captain_decision');
    this.byShip = new Map();
    this.linked = new Map();
    for (const d of this.decisions) {
      if (!this.byShip.has(d.ship_id)) this.byShip.set(d.ship_id, []);
      this.byShip.get(d.ship_id).push(d);
      this.linked.set(d.event_id, []);
    }
    for (const e of this.events) {
      for (const id of new Set([...(e.data.decision_ids || []), e.data.decision_id].filter(Boolean))) {
        this.linked.get(id)?.push(e);
      }
    }
  }

  history(shipId) { return this.byShip.get(shipId) || []; }
  latest(shipId, time) { return this.history(shipId).filter(e => e.timestamp <= time).at(-1); }

  detail(id, time) {
    const decision = this.byId.get(id);
    if (!decision) return null;
    const modern = Array.isArray(decision.data.commands);
    const history = this.history(decision.ship_id);
    const next = history[history.indexOf(decision) + 1]?.timestamp ?? Infinity;
    const faction = this.ships[decision.ship_id]?.faction || decision.data.faction;
    const parents = (decision.data.parent_event_ids || []).map(id => this.byId.get(id)).filter(Boolean);
    for (const p of [...parents]) {
      for (const id of p.data.parent_event_ids || []) {
        const event = this.byId.get(id);
        if (event && !parents.includes(event)) parents.push(event);
      }
    }
    const before = this.events.filter(e => e.timestamp <= decision.timestamp);
    // Standing plan persists between checkpoints. Context is labelled separately
    // from direct command dependencies, including in pre-v3 recordings.
    const context = before.filter(e =>
      (e.event_type === 'admiral_order' && e.ship_id === decision.ship_id && e.timestamp === decision.timestamp) ||
      (e.event_type === 'captain_admiral_discussion' && e.ship_id === decision.ship_id && e.timestamp === decision.timestamp));
    for (const kind of ['admiral_plan', 'admiral_directive']) {
      const last = before.filter(e => e.event_type === kind && e.data.faction === faction).at(-1);
      if (last) context.unshift(last);
    }
    // Early v3 captures have exact admiral calls but no explicit parent edge.
    // Expose them as checkpoint context without inventing a particular order's source.
    for (const call of before.filter(e => e.event_type === 'model_call' && e.data.actor === 'admiral' &&
      e.data.faction === faction && e.timestamp === decision.timestamp)) {
      if (!parents.includes(call)) parents.push(call);
    }
    const notes = this.events.filter(e => e.ship_id === decision.ship_id &&
      e.timestamp === decision.timestamp && e.event_type === 'captain_log');
    const allLinked = modern ? this.linked.get(id) || [] : this.events.filter(e =>
      e.timestamp >= decision.timestamp && e.timestamp < next &&
      (e.ship_id === decision.ship_id || e.data.target_id === decision.ship_id));
    const visible = allLinked.filter(e => e.timestamp <= time);
    const execution = visible.filter(e => ['command_status', 'weapon_status', 'execution_state', 'maneuver'].includes(e.event_type));
    const outcomes = visible.filter(e => OUTCOMES.has(e.event_type) || LAUNCHES.has(e.event_type));
    const launches = visible.filter(e => LAUNCHES.has(e.event_type));
    const terminals = new Set(this.events.filter(e => e.timestamp <= time && TERMINAL.has(e.event_type)).map(objectId).filter(Boolean));
    const inFlight = launches.filter(e => objectId(e) && !terminals.has(objectId(e)));
    const facts = [];
    const fact = (text, evidence) => { if (evidence.length) facts.push({text, event_ids: evidence.map(e => e.event_id)}); };
    const issued = visible.filter(e => e.event_type === 'command_status' && e.data.status === 'issued');
    const rejected = visible.filter(e => e.event_type === 'command_status' && e.data.status === 'rejected');
    fact(`${issued.length} commands issued; ${rejected.length} rejected.`, [...issued, ...rejected]);
    fact(`${launches.filter(e => e.event_type === 'shot_fired').length} gunshots and ${launches.filter(e => e.event_type === 'torpedo_launched').length} torpedoes launched.`, launches);
    const hits = visible.filter(e => ['hit', 'torpedo_impact'].includes(e.event_type));
    fact(`${hits.length} impacts recorded${modern ? ' from these orders' : ' during this decision window'}.`, hits);
    const failures = visible.filter(e => ['miss', 'torpedo_miss', 'torpedo_intercepted', 'pd_slug_destroyed', 'pd_torpedo_destroyed'].includes(e.event_type));
    fact(`${failures.length} misses or interceptions recorded.`, failures);
    fact(`${inFlight.length} launched rounds have no recorded terminal outcome yet.`, inFlight);
    const modules = visible.filter(e => e.event_type === 'module_destroyed');
    fact(`${modules.length} modules destroyed${modern ? ' by linked impacts' : ' during this window'}.`, modules);
    const waits = execution.filter(e => e.event_type === 'weapon_status' && e.data.reason !== 'fired');
    const reasons = [...new Set(waits.map(e => readable(e.data.reason)))];
    fact(`Weapons reported: ${reasons.join(', ')}.`, waits);
    const received = this.events.filter(e => ['hit', 'torpedo_impact'].includes(e.event_type) &&
      e.data.target_id === decision.ship_id && e.timestamp >= decision.timestamp && e.timestamp < next && e.timestamp <= time);
    fact(`${received.length} incoming impacts during this decision window (context, not attribution).`, received);
    return {decision, modern, context: [...new Map([...context, ...parents.filter(e => e.event_type !== 'model_call')].map(e => [e.event_id, e])).values()],
      calls: parents.filter(e => e.event_type === 'model_call'), notes, execution, outcomes, facts};
  }
}

export function readable(value) { return String(value ?? '').replaceAll('_', ' ').toLowerCase(); }
export function describeEvent(e, ships = {}) {
  const d = e.data || {};
  const target = ships[d.target_id]?.name || d.target_id || '';
  switch (e.event_type) {
    case 'admiral_plan': return d.plan;
    case 'admiral_directive': return d.directive;
    case 'admiral_order': return d.order_text;
    case 'captain_log': return d.note;
    case 'captain_admiral_discussion': return `Captain: ${d.captain_question}\nAdmiral: ${d.admiral_response}`;
    case 'command_status': return `${readable(d.status)} · ${readable(d.command?.type || d.command?.maneuver_type || d.command?.command_type || 'command')}${d.reason ? ' · ' + d.reason : ''}`;
    case 'weapon_status': return `${readable(d.weapon_slot)} · ${readable(d.reason)}${target ? ' → ' + target : ''}`;
    case 'execution_state': return `${d.mode} · thrust ${Math.round((d.applied_throttle || 0) * 100)}% (ordered ${Math.round((d.requested_throttle || 0) * 100)}%)`;
    default: return `${readable(e.event_type)}${target ? ' → ' + target : ''}${d.module_name ? ' · ' + d.module_name : ''}${objectId(e) ? ' · ' + objectId(e) : ''}`;
  }
}
