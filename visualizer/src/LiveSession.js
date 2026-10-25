/**
 * Live-mode logic with no DOM or three.js: pacing of the live playhead,
 * checkpoint arithmetic, battle identity, reconnect backoff, verdicts and
 * the per-side progress of a checkpoint deliberation. main.js wires these
 * to the HUD; tests/live.test.js pins them.
 *
 * Pacing model ("broadcast delay"). An LLM battle streams in bursts: one
 * decision interval of footage arrives in a few wall seconds, then nothing
 * while the commanders deliberate for 30-120 s. Instead of fast-forwarding
 * each burst and then freezing, the live head plays at the chosen speed a
 * little behind the newest frame, so the deliberation pause is spent
 * watching the last interval in real time. It only speeds up (to at most
 * CATCHUP_MAX_RATE) when it falls more than maxLag behind - a fast server,
 * or a viewer that joined late. 'edge' pace is the old behaviour: hug the
 * newest frame.
 */

/** Seconds kept behind the newest frame while the sim runs (frames arrive per poll). */
export const LIVE_CUSHION_S = 2;
/** Catch-up never plays faster than this unless the viewer picked a faster speed. */
export const CATCHUP_MAX_RATE = 4;
/** Backoff between failed polls, seconds; the last value repeats. */
export const RECONNECT_BACKOFF_S = [2, 4, 8, 15];
/** Consecutive failed polls before the HUD declares the connection lost (~29 s). */
export const LOST_AFTER_FAILURES = 5;

/** Default lag allowed before catching up: one decision interval plus 15 s. */
export function defaultMaxLag(interval = 30) {
  return (interval || 30) + 15;
}

/** Sim time of the first checkpoint strictly after t (checkpoint k is at k * interval). */
export function nextCheckpointAfter(t, interval = 30) {
  const step = interval > 0 ? interval : 30;
  return (Math.floor(t / step + 1e-9) + 1) * step;
}

/** Checkpoint number of a checkpoint time (CP1 follows the opening interval). */
export function checkpointNumber(tCheckpoint, interval = 30) {
  return Math.round(tCheckpoint / (interval > 0 ? interval : 30));
}

/**
 * Where the live head should wait: the newest frame less the cushion. At a
 * paused checkpoint (or once the battle ended) there is no cushion, so the
 * checkpoint frame and the orders arriving at it are shown exactly.
 */
export function liveEdgeTarget(available, status) {
  const cushion = status === 'running' ? LIVE_CUSHION_S : 0;
  return Math.max(0, available - cushion);
}

/**
 * Advance the live playhead by one animation frame.
 * @param {Object} p
 * @param {number} p.current - playhead (sim s)
 * @param {number} p.available - newest received frame (sim s)
 * @param {number} p.dt - wall seconds since the last frame
 * @param {number} [p.speed=1] - chosen playback speed
 * @param {string} [p.status] - server status ('running' | 'paused' | ...)
 * @param {string} [p.pace='realtime'] - 'realtime' (broadcast delay) or 'edge'
 * @param {number} [p.maxLag] - lag (s) tolerated before catching up
 * @returns {{time: number, rate: number, state: 'playing'|'catching-up'|'at-edge'}}
 */
export function paceLiveHead({ current, available, dt, speed = 1, status = 'running', pace = 'realtime', maxLag = defaultMaxLag() }) {
  const target = liveEdgeTarget(available, status);
  const lag = target - current;
  if (lag <= 1e-6) {
    // Never step backwards when the cushion grows (paused -> running)
    return { time: current, rate: 0, state: 'at-edge' };
  }
  let rate;
  let state = 'playing';
  if (pace === 'edge') {
    rate = Math.max(speed, Math.min(1 + lag * 0.75, 16));
    if (rate > speed) state = 'catching-up';
  } else {
    rate = speed;
    const excess = lag - maxLag;
    if (excess > 0) {
      // Proportional: 15 s over the limit doubles the rate, 45 s quadruples it;
      // converges on maxLag without oscillating
      const boost = Math.min(CATCHUP_MAX_RATE, 1 + excess / 15);
      if (boost > speed) { rate = boost; state = 'catching-up'; }
    }
  }
  const time = Math.min(target, current + Math.max(0, dt) * rate);
  return { time, rate, state: time >= target - 1e-6 ? 'at-edge' : state };
}

/**
 * Where to start watching when the first frames arrive. Joining at the
 * start plays from 0; joining mid-battle starts one checkpoint before the
 * live edge (the last interval plays out while the commanders decide).
 */
export function liveStartTime(available, interval = 30, status = 'running') {
  const edge = liveEdgeTarget(available, status);
  if (edge <= interval + 15) return 0;
  const step = interval > 0 ? interval : 30;
  const lastCp = Math.floor((edge - 1e-6) / step) * step;
  // Too little footage after that checkpoint: start one interval earlier
  return Math.max(0, edge - lastCp >= 10 ? lastCp : lastCp - step);
}

/** Seconds the playhead runs behind the server's sim clock. */
export function liveDelay(simTimeS, current) {
  return Math.max(0, (simTimeS || 0) - (current || 0));
}

/** Wall seconds to wait before the next poll after `failures` failed polls. */
export function backoffDelay(failures) {
  if (failures <= 0) return RECONNECT_BACKOFF_S[0];
  return RECONNECT_BACKOFF_S[Math.min(failures - 1, RECONNECT_BACKOFF_S.length - 1)];
}

/**
 * A fresh id naming this viewer's live session to the server, whose
 * end-of-battle linger waits for every spectator it knows by id.
 */
export function newLiveClientId() {
  const uuid = globalThis.crypto?.randomUUID?.();
  return (uuid ? uuid.replace(/-/g, '') : Math.random().toString(36).slice(2) + Date.now().toString(36)).slice(0, 12);
}

/**
 * Query string of a /live/recording poll: the incremental cursor once the
 * battle is on screen (before that the whole recording-so-far), plus the
 * viewer's client id.
 * @param {Object} q - {clientId, started, lastFrameT, lastSeq}
 */
export function recordingQuery({ clientId = null, started = false, lastFrameT = 0, lastSeq = 0 } = {}) {
  const params = [];
  if (started) params.push(`since_t=${lastFrameT}`, `since_seq=${lastSeq || 0}`);
  if (clientId) params.push(`client=${encodeURIComponent(clientId)}`);
  return params.length ? `?${params.join('&')}` : '';
}

/**
 * Is this payload from a different battle than the one on screen?
 * Servers send battle_id; for older servers a sim clock that jumped back
 * by more than one interval (or a draft after the battle started) means
 * a restart.
 * @param {Object} known - {battleId, started, lastFrameT, interval}
 * @param {Object} info - the payload's live block
 * @returns {string|null} reason, or null when it is the same battle
 */
export function newBattleReason(known, info) {
  if (!known || !info) return null;
  if (info.battle_id && known.battleId && info.battle_id !== known.battleId) return 'battle_id changed';
  if (info.battle_id && known.battleId) return null;
  if (!known.started) return null;
  if (info.phase === 'draft') return 'draft restarted';
  const interval = info.decision_interval_s || known.interval || 30;
  if (typeof info.sim_time_s === 'number' && info.phase !== 'ended' &&
      info.sim_time_s + interval < (known.lastFrameT || 0)) return 'sim clock went back';
  return null;
}

const DRAW_ON_POINTS = /too close to call\s*\(([\d.]+)\s*vs\s*([\d.]+)\)/i;
const HALTED = /ceiling|budget|error|stopped|exception/i;

/**
 * Verdict for the end of a live battle.
 * @param {Object} p - {winner, reason, events, billing, lost}
 * @returns {{kind: 'victory'|'draw'|'halted'|'lost', faction: string|null, title: string, detail: string}}
 */
export function classifyVerdict({ winner = null, reason = null, events = [], billing = null, lost = false } = {}) {
  const stopped = events.findLast?.(e => e.event_type === 'battle_stopped')
    ?? [...events].reverse().find(e => e.event_type === 'battle_stopped');
  const why = reason || stopped?.data?.reason || billing?.error || '';
  if (winner) {
    return { kind: 'victory', faction: String(winner), title: `${String(winner).toUpperCase()} VICTORIOUS`,
      detail: why };
  }
  if (stopped || HALTED.test(why)) {
    let detail = String(why).replace(/^RuntimeError:\s*/, '').replace(/;.*$/, '');
    if (/ceiling|budget/i.test(why) && billing?.spend_limit_usd) {
      detail = `spending ceiling reached ($${Number(billing.reported_cost_usd || 0).toFixed(2)} / $${Number(billing.spend_limit_usd).toFixed(2)})`;
    }
    return { kind: 'halted', faction: null, title: 'BATTLE HALTED', detail: detail || 'stopped before a verdict' };
  }
  if (lost) {
    return { kind: 'lost', faction: null, title: 'STREAM ENDED',
      detail: 'the battle server stopped before the battle finished' };
  }
  const points = DRAW_ON_POINTS.exec(why);
  if (points) {
    return { kind: 'draw', faction: null, title: 'DRAW ON POINTS', detail: `${points[1]} – ${points[2]}` };
  }
  return { kind: 'draw', faction: null, title: 'DRAW', detail: why };
}

/** 'm:ss' for wall or sim seconds. */
export function formatClock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

const factionOf = (shipId, ships) => ships?.[shipId]?.faction || (shipId ? String(shipId).split('_')[0] : null);

/**
 * Per-faction progress of one checkpoint's deliberation, read from the
 * events streamed so far (they arrive during the pause) plus the server's
 * in-flight calls.
 * @param {Object} p
 * @param {Array} p.events - loader events (any order)
 * @param {Object} p.ships - loader.ships {id: {faction}}
 * @param {number} p.checkpointT - checkpoint sim time
 * @param {Object} [p.alive] - {alpha: n, beta: n} ships that will be asked
 * @param {Array} [p.pending] - live block pending_calls
 * @returns {Object} {alpha: {...}, beta: {...}}
 */
export function deliberationProgress({ events = [], ships = {}, checkpointT, alive = {}, pending = [] }) {
  const out = {};
  for (const f of ['alpha', 'beta']) {
    out[f] = { admiral: false, admiralLatency: null, admiralCalls: 0, orders: 0, captains: 0,
      of: alive[f] ?? null, calls: 0, errors: 0, pending: [] };
  }
  const at = e => Math.abs(e.timestamp - checkpointT) < 1e-6;
  for (const e of events) {
    if (!at(e)) continue;
    const d = e.data || {};
    switch (e.event_type) {
      case 'admiral_directive':
        if (out[d.faction]) out[d.faction].admiral = true;
        break;
      case 'admiral_order': {
        const f = factionOf(e.ship_id, ships);
        if (out[f]) out[f].orders++;
        break;
      }
      case 'captain_decision': {
        const f = d.faction || factionOf(e.ship_id, ships);
        if (out[f]) out[f].captains++;
        break;
      }
      case 'model_call': {
        const p = out[d.faction];
        if (!p || d.phase === 'draft') break;
        p.calls++;
        if (d.actor === 'admiral') {
          p.admiralCalls++;
          if (typeof d.latency_s === 'number') p.admiralLatency = (p.admiralLatency || 0) + d.latency_s;
        }
        break;
      }
      case 'model_error':
        if (out[d.faction]) out[d.faction].errors++;
        break;
    }
  }
  for (const call of pending || []) {
    if (out[call.faction]) out[call.faction].pending.push(call);
  }
  return out;
}
