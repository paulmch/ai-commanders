/**
 * TrajectoryPredictor - where will each ship be at the next checkpoint?
 *
 * Pure math, no THREE: positions in metres, times in battle seconds, the
 * same frame as sim_trace `pos`. For a playback time t it returns, per live
 * ship, a path from the ship's current (rendered) position to the next
 * checkpoint, the predicted checkpoint position and a 1-sigma dispersion.
 *
 * Sources, best first:
 *  1. The recorded nominal `plan` of the two frames bracketing t, blended by
 *     the interpolation factor so the track evolves smoothly between 1 Hz
 *     frames, and re-anchored onto the rendered hull.
 *  2. Kinematic propagation for recordings without plans: constant
 *     acceleration from the interpolated state (recorded `acc`, else
 *     throttle x hull max acceleration along the nose, else the finite
 *     difference of velocity). Sigma comes from recent acceleration jitter.
 *
 * Both recompute every frame from the actual state, so evasion wiggles
 * make the predicted checkpoint marker drift - that is the point.
 *
 * Optionally also returns the "actual" track: the recorded future path
 * from t to the same checkpoint (replay only), truncated at death or at
 * the end of the recording.
 */

export const TRACK_SAMPLES = 64;
export const DEFAULT_INTERVAL_S = 30;
const G = 9.80665;
const EPS = 1e-6;
const JITTER_WINDOW_S = 8;   // recent history used for the fallback spread
const BURN_SPREAD = 0.12;    // fallback spread per unit of constant-burn displacement

/** Max main-drive acceleration per hull class (CLAUDE.md ship table). */
export const HULL_ACCEL_G = {
  corvette: 3.0, frigate: 3.0, destroyer: 2.0, cruiser: 1.5, cruiser_torpedo: 1.5,
  battlecruiser: 1.5, battleship: 1.0, dreadnought: 0.75, dreadnought_siege: 0.75
};

/** Event types whose timestamps mark decision checkpoints. */
const CHECKPOINT_EVENTS = new Set(['checkpoint', 'captain_decision', 'admiral_order', 'admiral_directive']);

/**
 * Cubic Hermite between two samples (position + velocity), written into
 * out[o..o+2]. s in [0,1], h = sample spacing (s). Matches the ship
 * interpolation in Interpolator so the "actual" track starts on the hull.
 */
export function hermiteInto(out, o, p0, v0, p1, v1, h, s) {
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  for (let k = 0; k < 3; k++) {
    out[o + k] = h00 * p0[k] + h10 * h * v0[k] + h01 * p1[k] + h11 * h * v1[k];
  }
  return out;
}

/**
 * Checkpoint schedule of a recording: the header's decision_interval_s,
 * else the distinct timestamps of checkpoint-like events (with their
 * median spacing to extrapolate past the last one), else 30 s.
 * @param {Object} header - recording header (decision_interval_s)
 * @param {Array} events - recording events
 */
export function deriveCheckpointSchedule(header, events) {
  const interval = Number(header?.decision_interval_s);
  if (Number.isFinite(interval) && interval > 0) {
    return { interval, times: [], spacing: interval, source: 'header' };
  }
  const set = new Set();
  for (const e of events || []) {
    if (CHECKPOINT_EVENTS.has(e.event_type) && Number.isFinite(e.timestamp) && e.timestamp > 0) {
      set.add(Math.round(e.timestamp * 1000) / 1000);
    }
  }
  const times = [...set].sort((a, b) => a - b);
  if (times.length === 0) {
    return { interval: DEFAULT_INTERVAL_S, times: [], spacing: DEFAULT_INTERVAL_S, source: 'default' };
  }
  let spacing = times[0];
  if (times.length > 1) {
    const diffs = [];
    for (let i = 1; i < times.length; i++) diffs.push(times[i] - times[i - 1]);
    diffs.sort((a, b) => a - b);
    spacing = diffs[Math.floor(diffs.length / 2)];
  }
  return { interval: null, times, spacing: spacing > 0 ? spacing : DEFAULT_INTERVAL_S, source: 'events' };
}

/** First checkpoint strictly after t (a checkpoint AT t is the one just passed). */
export function nextCheckpointTime(schedule, t) {
  if (schedule.interval) {
    return (Math.floor((t + EPS) / schedule.interval) + 1) * schedule.interval;
  }
  const times = schedule.times;
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] > t + EPS) hi = mid; else lo = mid + 1;
  }
  if (lo < times.length) return times[lo];
  const last = times[times.length - 1];
  const sp = schedule.spacing;
  return last + (Math.floor((t + EPS - last) / sp) + 1) * sp;
}

function validPlan(plan) {
  return !!plan && Array.isArray(plan.path) && plan.path.length >= 2 && Number.isFinite(plan.t_cp);
}

/** Time of plan sample k: t_f + k*dt, the last one pinned to t_cp. */
function planTime(plan, tf, dt, k) {
  return k >= plan.path.length - 1 ? plan.t_cp : Math.min(tf + k * dt, plan.t_cp);
}

/** Plan sigma for a (possibly shorter) remaining horizon H: dispersion grows ~H^1.5. */
function planSigma(plan, tf, H) {
  const s = Number(plan.sigma) || 0;
  const span = plan.t_cp - tf;
  return span > 0 ? s * Math.pow(Math.min(1, H / span), 1.5) : s;
}

function makeTrack(shipId, n) {
  return {
    shipId, faction: shipId.startsWith('beta') ? 'beta' : 'alpha', active: false, source: 'plan',
    t: 0, tCp: 0, n: 0, sigma: 0,
    times: new Float64Array(n), path: new Float64Array(n * 3),
    cp: new Float64Array(3), cpVel: new Float64Array(3),
    actual: {
      n: 0, hasCp: false, endT: 0,
      times: new Float64Array(n), path: new Float64Array(n * 3), cp: new Float64Array(3)
    }
  };
}

export class TrajectoryPredictor {
  /**
   * @param {Object} source - BattleLoader (or any {simTrace, events,
   *   sourceData?, ships?}); read live on every call, so a live-mode
   *   loader that keeps appending frames just works.
   * @param {Object} [opts]
   * @param {number} [opts.samples] - points per track
   */
  constructor(source, { samples = TRACK_SAMPLES } = {}) {
    this.source = source;
    this.samples = Math.max(4, samples | 0);
    this.tracks = new Map();
    this._schedule = null;
    this._scheduleKey = null;
    this._accel = new Map();
    this._i0 = 0;
    this._i1 = 0;
    this._alpha = 0;
    // scratch (no per-frame allocation)
    this._a = new Float64Array(3);
    this._b = new Float64Array(3);
    this._off = new Float64Array(3);
    this._acc = new Float64Array(3);
    this._vel = new Float64Array(3);
  }

  get trace() {
    return this.source.simTrace || this.source.sim_trace || [];
  }

  get header() {
    return this.source.sourceData || this.source;
  }

  schedule() {
    const events = this.source.events || [];
    const key = `${this.header?.decision_interval_s}|${events.length}`;
    if (this._scheduleKey !== key) {
      this._schedule = deriveCheckpointSchedule(this.header, events);
      this._scheduleKey = key;
    }
    return this._schedule;
  }

  checkpointAfter(t) {
    return nextCheckpointTime(this.schedule(), t);
  }

  /** Hull max acceleration (m/s^2): calibrated from the trace, else class table, else specs. */
  hullAccel(shipId) {
    const trace = this.trace;
    const cached = this._accel.get(shipId);
    if (cached && (cached.samples >= 40 || cached.len === trace.length)) return cached.value;

    const ratios = [];
    for (let k = 1; k < trace.length && ratios.length < 240; k++) {
      const a = trace[k - 1].ships?.[shipId], b = trace[k].ships?.[shipId];
      if (!a || !b || a.destroyed || b.destroyed || a.dying || b.dying || !a.vel || !b.vel) continue;
      const th = b.thrust || 0;
      if (th < 0.1 || Math.abs(th - (a.thrust || 0)) > 0.02) continue;
      const h = trace[k].t - trace[k - 1].t;
      if (!(h > 0)) continue;
      const dv = Math.hypot(b.vel[0] - a.vel[0], b.vel[1] - a.vel[1], b.vel[2] - a.vel[2]);
      ratios.push(dv / (h * th));
    }
    let value;
    if (ratios.length >= 3) {
      ratios.sort((x, y) => x - y);
      value = ratios[Math.floor(ratios.length / 2)];
    } else {
      const type = this.source.ships?.[shipId]?.type;
      if (HULL_ACCEL_G[type]) value = HULL_ACCEL_G[type] * G;
      else value = (this._specAccelG(shipId) || 2.0) * G;
    }
    this._accel.set(shipId, { value, samples: ratios.length, len: trace.length });
    return value;
  }

  _specAccelG(shipId) {
    const h = this.header || {};
    for (const fleet of [h.alpha_fleet, h.beta_fleet]) {
      for (const s of fleet?.ships || []) {
        if (s.ship_id === shipId && s.max_acceleration_g) return s.max_acceleration_g;
      }
    }
    const specs = h[`${shipId}_specs`];
    return specs?.max_acceleration_g || null;
  }

  /** Locate the frames bracketing t (sets _i0, _i1, _alpha). */
  _bracket(t) {
    const trace = this.trace;
    const last = trace.length - 1;
    if (t <= trace[0].t) { this._i0 = this._i1 = 0; this._alpha = 0; return; }
    if (t >= trace[last].t) { this._i0 = this._i1 = last; this._alpha = 0; return; }
    let lo = 0, hi = last;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (trace[mid].t <= t) lo = mid; else hi = mid;
    }
    this._i0 = lo;
    this._i1 = hi;
    this._alpha = (t - trace[lo].t) / (trace[hi].t - trace[lo].t);
  }

  /**
   * Predict every live ship's track to its next checkpoint.
   * @param {number} t - playback time
   * @param {Object} ships - interpolated ship states (Interpolator output:
   *   position, velocity, forward, thrust, destroyed, dying) - the track is
   *   anchored on `position`, i.e. exactly where the hull is drawn
   * @param {boolean} [withActual] - also fill track.actual from the
   *   recorded future (replay)
   * @returns {Map<string, Object>} shipId -> track (reused objects; only
   *   entries with `active` are valid this frame)
   */
  predict(t, ships, withActual = false) {
    for (const tr of this.tracks.values()) tr.active = false;
    const trace = this.trace;
    if (!trace.length || !ships) return this.tracks;
    this._bracket(t);
    const f0 = trace[this._i0], f1 = trace[this._i1];

    for (const id in ships) {
      const st = ships[id];
      if (!st || st.destroyed || st.dying || !st.position) continue;
      const s0 = f0.ships?.[id], s1 = f1.ships?.[id];
      if (!s0 && !s1) continue;
      let tr = this.tracks.get(id);
      if (!tr) {
        tr = makeTrack(id, this.samples);
        const faction = this.source.ships?.[id]?.faction;
        if (faction) tr.faction = faction;
        this.tracks.set(id, tr);
      }
      const ok = this._fromPlans(tr, t, st, f0, s0, f1, s1)
        || this._kinematic(tr, id, t, st, f0, s0, f1, s1);
      if (!ok) continue;
      tr.active = true;
      if (withActual) this._actual(tr, id);
      else tr.actual.n = 0;
    }
    return this.tracks;
  }

  /** Position of a recorded plan at time tau (Catmull-Rom through its samples). */
  _planPos(plan, tf, vel, tau, out) {
    const P = plan.path;
    const n = P.length;
    const dt = plan.dt > 0 ? plan.dt : (plan.t_cp - tf) / (n - 1);
    if (tau <= tf || !(dt > 0)) {
      // before the plan starts (bracket blend): back along the frame velocity
      const d = tau - tf;
      const v = vel || this._segVel(plan, tf, dt, 0, this._vel);
      for (let k = 0; k < 3; k++) out[k] = P[0][k] + v[k] * d;
      return out;
    }
    if (tau >= plan.t_cp) {
      const v = this._segVel(plan, tf, dt, n - 2, this._vel);
      const d = tau - plan.t_cp;
      for (let k = 0; k < 3; k++) out[k] = P[n - 1][k] + v[k] * d;
      return out;
    }
    let i = Math.min(n - 2, Math.floor((tau - tf) / dt));
    while (i > 0 && planTime(plan, tf, dt, i) > tau) i--;
    while (i < n - 2 && planTime(plan, tf, dt, i + 1) <= tau) i++;
    const ta = planTime(plan, tf, dt, i), tb = planTime(plan, tf, dt, i + 1);
    const h = tb - ta;
    if (!(h > 0)) {
      for (let k = 0; k < 3; k++) out[k] = P[i + 1][k];
      return out;
    }
    const s = (tau - ta) / h;
    // non-uniform Catmull-Rom tangents; the first uses the frame velocity
    const m0 = this._a, m1 = this._b;
    if (i === 0 && vel) { m0[0] = vel[0]; m0[1] = vel[1]; m0[2] = vel[2]; } else this._tangent(plan, tf, dt, i, m0);
    this._tangent(plan, tf, dt, i + 1, m1);
    return hermiteInto(out, 0, P[i], m0, P[i + 1], m1, h, s);
  }

  _segVel(plan, tf, dt, i, out) {
    const P = plan.path;
    const h = planTime(plan, tf, dt, i + 1) - planTime(plan, tf, dt, i);
    for (let k = 0; k < 3; k++) out[k] = h > 0 ? (P[i + 1][k] - P[i][k]) / h : 0;
    return out;
  }

  _tangent(plan, tf, dt, i, out) {
    const P = plan.path;
    const n = P.length;
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    const h = planTime(plan, tf, dt, b) - planTime(plan, tf, dt, a);
    for (let k = 0; k < 3; k++) out[k] = h > 0 ? (P[b][k] - P[a][k]) / h : 0;
    return out;
  }

  _fromPlans(tr, t, st, f0, s0, f1, s1) {
    const p0 = validPlan(s0?.plan) ? s0.plan : null;
    const p1 = f1 !== f0 && validPlan(s1?.plan) ? s1.plan : null;
    if (!p0 && !p1) return false;
    // Target: the earliest planned checkpoint still ahead. A plan whose
    // checkpoint comes before that cannot describe the whole horizon.
    let tCp = Infinity;
    if (p0 && p0.t_cp > t + EPS) tCp = Math.min(tCp, p0.t_cp);
    if (p1 && p1.t_cp > t + EPS) tCp = Math.min(tCp, p1.t_cp);
    if (!Number.isFinite(tCp)) return false;
    const alpha = this._alpha;
    const c0 = p0 && p0.t_cp >= tCp - EPS, c1 = p1 && p1.t_cp >= tCp - EPS;
    let w0 = c0 ? (p1 ? 1 - alpha : 1) : 0;
    let w1 = c1 ? (p0 ? alpha : 1) : 0;
    if (w0 + w1 <= 0) { w0 = c0 ? 1 : 0; w1 = c1 ? 1 : 0; }
    const ws = w0 + w1;
    w0 /= ws; w1 /= ws;

    const n = this.samples;
    const H = tCp - t;
    const out = tr.path, times = tr.times, a = this._off, b = this._acc;
    for (let i = 0; i < n; i++) {
      const tau = i === n - 1 ? tCp : t + (H * i) / (n - 1);
      times[i] = tau;
      out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = 0;
      if (w0 > 0) {
        this._planPos(p0, f0.t, s0.vel, tau, a);
        for (let k = 0; k < 3; k++) out[i * 3 + k] += w0 * a[k];
      }
      if (w1 > 0) {
        this._planPos(p1, f1.t, s1.vel, tau, b);
        for (let k = 0; k < 3; k++) out[i * 3 + k] += w1 * b[k];
      }
    }
    // Re-anchor on the rendered hull, easing the correction out so the
    // planned checkpoint itself is untouched.
    const pos = st.position;
    const dx = pos[0] - out[0], dy = pos[1] - out[1], dz = pos[2] - out[2];
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1);
      const w = 1 - u * u * (3 - 2 * u);
      out[i * 3] += dx * w;
      out[i * 3 + 1] += dy * w;
      out[i * 3 + 2] += dz * w;
    }
    tr.sigma = (w0 > 0 ? w0 * planSigma(p0, f0.t, H) : 0) + (w1 > 0 ? w1 * planSigma(p1, f1.t, H) : 0);
    tr.source = 'plan';
    this._finish(tr, t, tCp, st);
    return true;
  }

  _kinematic(tr, id, t, st, f0, s0, f1, s1) {
    const tCp = this.checkpointAfter(t);
    const H = tCp - t;
    if (!(H > 0)) return false;
    const alpha = this._alpha;
    const vel = st.velocity || s0?.vel || s1?.vel;
    if (!vel) return false;
    const acc = this._acc;
    const h = f1.t - f0.t;
    if (s0?.acc || s1?.acc) {
      const a0 = s0?.acc || s1.acc, a1 = s1?.acc || s0.acc;
      for (let k = 0; k < 3; k++) acc[k] = a0[k] + (a1[k] - a0[k]) * alpha;
    } else if (st.forward && Number.isFinite(st.thrust)) {
      const f = st.forward;
      const fl = Math.hypot(f[0], f[1], f[2]) || 1;
      const am = st.thrust * this.hullAccel(id);
      for (let k = 0; k < 3; k++) acc[k] = (f[k] / fl) * am;
    } else if (s0?.vel && s1?.vel && h > 0) {
      for (let k = 0; k < 3; k++) acc[k] = (s1.vel[k] - s0.vel[k]) / h;
    } else {
      acc[0] = acc[1] = acc[2] = 0;
    }
    const n = this.samples;
    const pos = st.position;
    for (let i = 0; i < n; i++) {
      const tau = i === n - 1 ? tCp : t + (H * i) / (n - 1);
      const d = tau - t;
      tr.times[i] = tau;
      for (let k = 0; k < 3; k++) tr.path[i * 3 + k] = pos[k] + vel[k] * d + 0.5 * acc[k] * d * d;
    }
    // Spread: recent jitter held over the horizon, plus a share of the
    // current burn (throttle/heading changes). Fitted on heuristic and LLM
    // recordings: the actual checkpoint lands inside ~1.3 sigma half the time.
    const q = 0.5 * H * H;
    tr.sigma = Math.hypot(this._jitter(id, t) * q, BURN_SPREAD * Math.hypot(acc[0], acc[1], acc[2]) * q);
    tr.source = 'kinematic';
    this._finish(tr, t, tCp, st);
    for (let k = 0; k < 3; k++) tr.cpVel[k] = vel[k] + acc[k] * H;
    return true;
  }

  /** RMS deviation (m/s^2) of the finite-difference acceleration over the recent window. */
  _jitter(id, t) {
    const trace = this.trace;
    let n = 0, mx = 0, my = 0, mz = 0, m2 = 0;
    for (let j = this._i0; j > 0 && trace[j].t >= t - JITTER_WINDOW_S; j--) {
      const a = trace[j - 1].ships?.[id], b = trace[j].ships?.[id];
      if (!a?.vel || !b?.vel || a.destroyed || b.destroyed) break;
      const h = trace[j].t - trace[j - 1].t;
      if (!(h > 0)) continue;
      const ax = (b.vel[0] - a.vel[0]) / h, ay = (b.vel[1] - a.vel[1]) / h, az = (b.vel[2] - a.vel[2]) / h;
      mx += ax; my += ay; mz += az; m2 += ax * ax + ay * ay + az * az; n++;
    }
    if (n < 2) return 0;
    mx /= n; my /= n; mz /= n;
    return Math.sqrt(Math.max(0, m2 / n - (mx * mx + my * my + mz * mz)));
  }

  _finish(tr, t, tCp, st) {
    const n = this.samples;
    tr.t = t;
    tr.tCp = tCp;
    tr.n = n;
    const P = tr.path;
    for (let k = 0; k < 3; k++) tr.cp[k] = P[(n - 1) * 3 + k];
    const h = tr.times[n - 1] - tr.times[n - 2];
    for (let k = 0; k < 3; k++) {
      tr.cpVel[k] = h > 1e-6 ? (P[(n - 1) * 3 + k] - P[(n - 2) * 3 + k]) / h : (st.velocity?.[k] || 0);
    }
  }

  /** Recorded future path over the same times as the prediction. */
  _actual(tr, id) {
    const trace = this.trace;
    const act = tr.actual;
    act.n = 0;
    act.hasCp = false;
    const last = trace.length - 1;
    const n = tr.n;
    let j = this._i0;
    let k = 0;
    for (let i = 0; i < n; i++) {
      const tau = tr.times[i];
      if (tau > trace[last].t + EPS) break;
      while (j < last && trace[j + 1].t < tau) j++;
      const fa = trace[j], fb = trace[Math.min(j + 1, last)];
      const sa = fa.ships?.[id], sb = fb.ships?.[id];
      if (!sa?.pos || sa.destroyed || sa.dying) break;
      if (tau > fa.t + EPS && (!sb?.pos || sb.destroyed || sb.dying)) {
        // killed inside this interval: end the track on its last live frame
        if (k === 0 || act.times[k - 1] < fa.t - EPS) {
          act.times[k] = fa.t;
          for (let c = 0; c < 3; c++) act.path[k * 3 + c] = sa.pos[c];
          k++;
        }
        break;
      }
      const h = fb.t - fa.t;
      act.times[k] = tau;
      if (h > 0 && sa.vel && sb.vel) {
        hermiteInto(act.path, k * 3, sa.pos, sa.vel, sb.pos, sb.vel, h, Math.min(1, Math.max(0, (tau - fa.t) / h)));
      } else {
        for (let c = 0; c < 3; c++) act.path[k * 3 + c] = sa.pos[c];
      }
      k++;
    }
    act.n = k;
    act.endT = k ? act.times[k - 1] : tr.t;
    if (k === n && Math.abs(act.times[k - 1] - tr.tCp) < 1e-3) {
      act.hasCp = true;
      for (let c = 0; c < 3; c++) act.cp[c] = act.path[(k - 1) * 3 + c];
    }
  }
}
