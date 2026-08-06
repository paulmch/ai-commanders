import { getModulesForShipType } from './shipModules.js';

/**
 * BattleLoader - Loads and parses battle recording JSON files
 */
export class BattleLoader {
  constructor() {
    this.metadata = null;
    this.simTrace = [];
    this.events = [];
    this.ships = {};
    this.duration = 0;
    // Map projectile_id -> hit event with impact_position for extrapolation
    this.projectileHits = new Map();
    // Map projectile_id -> last known state for extrapolation
    this.lastProjectileState = new Map();
    // Ship damage tracking: shipId -> { armor: {nose, lateral, tail}, modules: {name: {damaged, destroyed}} }
    this.shipDamageState = new Map();
    // Initial armor values per ship (from fleet data)
    this.initialArmor = new Map();
    // Ship targeting: shipId -> [{timestamp, target_id, target_name}]
    this.shipTargets = new Map();
    // Torpedo lifecycle: torpedoId -> {launch, outcome, lastState}
    this.torpedoes = new Map();
    // Continuous PD dwell segments built from per-tick pd_fired events
    this.pdBeamSegments = [];
    // Radiator state changes: shipId -> [{timestamp, extended}]
    this.radiatorTimeline = new Map();
  }

  /**
   * Load a battle recording from URL
   * @param {string} url - URL to the JSON recording file
   */
  async load(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load recording: ${response.statusText}`);
    }
    const data = await response.json();
    return this.parse(data);
  }

  /**
   * Parse battle recording data
   * @param {Object} data - Raw JSON data
   */
  parse(data) {
    // Extract metadata
    this.metadata = {
      recordingVersion: data.recording_version,
      recordedAt: data.recorded_at,
      isFleetBattle: data.is_fleet_battle,
      battleName: data.battle_name || 'Battle',
      winner: data.winner,
      resultReason: data.result_reason,
      initialDistanceKm: data.initial_distance_km,
      timeLimitS: data.time_limit_s
    };

    // Parse sim trace
    this.simTrace = data.sim_trace || [];
    this.duration = this.simTrace.length > 0
      ? this.simTrace[this.simTrace.length - 1].t
      : data.duration_s || 0;

    // Parse events
    this.events = (data.events || []).sort((a, b) => a.timestamp - b.timestamp);

    // Extract ship information
    this.initializeShips(data);

    // Build projectile hit map for extrapolation
    this.buildProjectileHitMap();

    // Track last known projectile states from sim_trace
    this.buildLastProjectileStates();

    // Build ship damage timeline from events
    this.buildShipDamageTimeline(data);

    // Build ship target timeline from captain decisions
    this.buildShipTargetTimeline();

    // Build torpedo lifecycle map (launch -> flight -> outcome)
    this.buildTorpedoTimeline();

    // Collapse per-tick pd_fired events into continuous dwell segments
    this.buildPDBeamSegments();

    // Radiator extended/retracted changes per ship
    this.buildRadiatorTimeline();

    return this;
  }

  /**
   * Initialize ship data from recording
   */
  initializeShips(data) {
    this.ships = {};

    // Get ships from fleet data if available
    if (data.alpha_fleet && data.alpha_fleet.ships) {
      for (const ship of data.alpha_fleet.ships) {
        this.ships[ship.ship_id] = {
          id: ship.ship_id,
          name: ship.ship_name,
          faction: 'alpha',
          type: ship.ship_type || 'destroyer'
        };
      }
    }

    if (data.beta_fleet && data.beta_fleet.ships) {
      for (const ship of data.beta_fleet.ships) {
        this.ships[ship.ship_id] = {
          id: ship.ship_id,
          name: ship.ship_name,
          faction: 'beta',
          type: ship.ship_type || 'destroyer'
        };
      }
    }

    // 1v1 recordings carry ship identity in the top-level metadata, not in
    // fleet blocks - without this the registry showed raw ids and every hull
    // rendered as the fallback class
    if (Object.keys(this.ships).length === 0) {
      const sides = [
        ['alpha', data.alpha_specs, data.alpha_ship],
        ['beta', data.beta_specs, data.beta_ship]
      ];
      for (const [id, specs, name] of sides) {
        if (specs || name) {
          this.ships[id] = {
            id: id,
            name: name || id,
            faction: id,
            type: specs?.ship_type || 'destroyer'
          };
        }
      }
    }

    // Fallback: extract from sim_trace if no fleet data
    if (Object.keys(this.ships).length === 0 && this.simTrace.length > 0) {
      const firstFrame = this.simTrace[0];
      if (firstFrame.ships) {
        for (const [shipId, state] of Object.entries(firstFrame.ships)) {
          this.ships[shipId] = {
            id: shipId,
            name: state.name || shipId,
            faction: shipId.startsWith('alpha') ? 'alpha' : 'beta',
            type: 'unknown'
          };
        }
      }
    }
  }

  /**
   * Get ship type (class) for a ship
   * @param {string} shipId - Ship ID
   * @returns {string} Ship type (corvette, frigate, destroyer, cruiser, battlecruiser, battleship, dreadnought)
   */
  getShipType(shipId) {
    const ship = this.ships[shipId];
    return ship?.type || 'destroyer';
  }

  /**
   * Get frame at specific time using binary search
   * @param {number} time - Time in seconds
   * @returns {Object} Frame data at that time
   */
  getFrameAt(time) {
    if (this.simTrace.length === 0) return null;

    // Clamp time to valid range
    time = Math.max(0, Math.min(time, this.duration));

    // Binary search for bracketing frames
    let low = 0;
    let high = this.simTrace.length - 1;

    while (low < high - 1) {
      const mid = Math.floor((low + high) / 2);
      if (this.simTrace[mid].t <= time) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return {
      frame0: this.simTrace[low],
      frame1: this.simTrace[high],
      alpha: high > low ? (time - this.simTrace[low].t) / (this.simTrace[high].t - this.simTrace[low].t) : 0
    };
  }

  /**
   * Per-ship death analysis for the two-stage destruction sequence.
   *
   * Returns { shipId: { time, velocity (m/s), reactorCause } }. A death is
   * a "reactor cause" when a module_destroyed for a reactor lands within
   * the death tick - that ship detonates immediately. Everything else
   * drifts dark for a few seconds before the reactor cooks off. Velocity
   * comes from the last live trace frame so the hulk coasts on its final
   * vector.
   */
  buildDeathInfo() {
    const info = {};
    const trace = this.simTrace || [];
    for (let i = 0; i < trace.length; i++) {
      for (const [shipId, s] of Object.entries(trace[i].ships || {})) {
        if (s.destroyed && !(shipId in info)) {
          const prev = i > 0 ? trace[i - 1].ships?.[shipId] : null;
          info[shipId] = {
            time: trace[i].t,
            velocity: (prev && !prev.destroyed && prev.vel) ? prev.vel : [0, 0, 0],
            reactorCause: false
          };
        }
      }
    }
    for (const e of this.events) {
      if (e.event_type !== 'module_destroyed') continue;
      const name = (e.data?.module_name || '').toLowerCase();
      if (!name.includes('reactor')) continue;
      const death = info[e.ship_id];
      if (death && e.timestamp >= death.time - 2.0 && e.timestamp <= death.time + 0.5) {
        death.reactorCause = true;
      }
    }
    return info;
  }

  /**
   * Get events within a time range
   * @param {number} startTime - Start time in seconds
   * @param {number} endTime - End time in seconds
   * @returns {Array} Events in range
   */
  getEventsInRange(startTime, endTime) {
    return this.events.filter(e => e.timestamp >= startTime && e.timestamp <= endTime);
  }

  /**
   * Get PD fired events within a time range
   * @param {number} startTime - Start time in seconds
   * @param {number} endTime - End time in seconds
   * @returns {Array} PD fired events
   */
  getPDEventsInRange(startTime, endTime) {
    return this.events.filter(e =>
      e.event_type === 'pd_fired' &&
      e.timestamp >= startTime &&
      e.timestamp <= endTime
    );
  }

  /**
   * Get hit events within a time range
   * @param {number} startTime - Start time in seconds
   * @param {number} endTime - End time in seconds
   * @returns {Array} Hit events with impact data
   */
  getHitEventsInRange(startTime, endTime) {
    return this.events.filter(e =>
      e.event_type === 'hit' &&
      e.timestamp >= startTime &&
      e.timestamp <= endTime
    );
  }

  /**
   * Get ship count by faction
   */
  getFleetCounts() {
    const alpha = { total: 0, alive: 0 };
    const beta = { total: 0, alive: 0 };

    for (const ship of Object.values(this.ships)) {
      if (ship.faction === 'alpha') {
        alpha.total++;
      } else {
        beta.total++;
      }
    }

    return { alpha, beta };
  }

  /**
   * Build map of projectile_id -> hit event for extrapolation
   */
  buildProjectileHitMap() {
    this.projectileHits.clear();
    for (const event of this.events) {
      if (event.event_type === 'hit' && event.data?.projectile_id) {
        this.projectileHits.set(event.data.projectile_id, {
          timestamp: event.timestamp,
          impact_position: event.data.impact_position,
          target_id: event.ship_id
        });
      }
    }
  }

  /**
   * Track last known state of each projectile from sim_trace
   * Used for extrapolating projectiles after they leave the trace
   */
  buildLastProjectileStates() {
    this.lastProjectileState.clear();
    for (const frame of this.simTrace) {
      if (!frame.projectiles) continue;
      for (const proj of frame.projectiles) {
        this.lastProjectileState.set(proj.id, {
          time: frame.t,
          pos: proj.pos,
          vel: proj.vel,
          source: proj.source,
          target: proj.target,
          mass_kg: proj.mass_kg
        });
      }
    }
  }

  /**
   * Get projectiles that need extrapolation at a given time
   * Returns projectiles that have left the sim_trace but haven't hit yet
   * @param {number} time - Current time
   * @returns {Array} Extrapolated projectile states
   */
  getExtrapolatedProjectiles(time) {
    const extrapolated = [];

    for (const [projId, hitInfo] of this.projectileHits) {
      const lastState = this.lastProjectileState.get(projId);
      if (!lastState) continue;

      // Calculate travel time based on distance and velocity
      let travelTime = 1.0; // default fallback
      if (hitInfo.impact_position && lastState.vel) {
        const dx = hitInfo.impact_position[0] - lastState.pos[0];
        const dy = hitInfo.impact_position[1] - lastState.pos[1];
        const dz = hitInfo.impact_position[2] - lastState.pos[2];
        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

        const vx = lastState.vel[0];
        const vy = lastState.vel[1];
        const vz = lastState.vel[2];
        const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

        if (speed > 0) {
          travelTime = distance / speed;
        }
      }

      // Effective hit time: last recorded time + time to travel to impact
      const effectiveHitTime = lastState.time + travelTime;

      // Only extrapolate if:
      // 1. Current time is at or after projectile's last recorded time
      // 2. Current time is before or at the effective hit time
      if (time >= lastState.time && time <= effectiveHitTime) {
        const dt = time - lastState.time;

        let pos;
        if (hitInfo.impact_position && travelTime > 0) {
          // Interpolate towards impact position at constant velocity
          const t = Math.min(1.0, dt / travelTime);
          pos = [
            lastState.pos[0] + (hitInfo.impact_position[0] - lastState.pos[0]) * t,
            lastState.pos[1] + (hitInfo.impact_position[1] - lastState.pos[1]) * t,
            lastState.pos[2] + (hitInfo.impact_position[2] - lastState.pos[2]) * t
          ];
        } else {
          // Fallback: linear extrapolation using velocity
          pos = [
            lastState.pos[0] + lastState.vel[0] * dt,
            lastState.pos[1] + lastState.vel[1] * dt,
            lastState.pos[2] + lastState.vel[2] * dt
          ];
        }

        extrapolated.push({
          id: projId,
          pos: pos,
          vel: lastState.vel,
          source: lastState.source,
          target: lastState.target,
          mass_kg: lastState.mass_kg,
          extrapolated: true,
          effectiveHitTime: effectiveHitTime
        });
      }
    }

    return extrapolated;
  }

  /**
   * Build ship damage timeline from events
   * Extracts initial armor from sim_trace (t=0 or first frame) and tracks module events
   */
  buildShipDamageTimeline(data) {
    this.shipDamageState.clear();
    this.initialArmor.clear();

    // Default armor thickness (cm) per section based on destroyer from fleet_ships.json
    const DEFAULT_ARMOR = {
      nose: 212.0,
      lateral: 36.4,
      tail: 42.3
    };

    // Extract initial armor from first sim_trace frame (has actual armor values)
    const firstFrame = this.simTrace.length > 0 ? this.simTrace[0] : null;

    // Weapons per ship come from fleet blocks (fleet battles) or the
    // alpha/beta specs (1v1)
    const weaponsById = {};
    for (const fleet of [data.alpha_fleet, data.beta_fleet]) {
      for (const ship of fleet?.ships || []) {
        weaponsById[ship.ship_id] = ship.weapons || {};
      }
    }
    if (data.alpha_specs) weaponsById.alpha = data.alpha_specs.weapons || {};
    if (data.beta_specs) weaponsById.beta = data.beta_specs.weapons || {};

    // Build damage state for EVERY known ship. This used to iterate only the
    // fleet blocks, which 1v1 recordings don't have - so getShipDamageAt()
    // returned null and the telemetry panel silently kept its static
    // placeholder armor values no matter how ablated the hull actually was.
    for (const shipId of Object.keys(this.ships)) {
      let initialArmor = { ...DEFAULT_ARMOR };
      if (firstFrame?.ships?.[shipId]?.armor) {
        const frameArmor = firstFrame.ships[shipId].armor;
        if (frameArmor.nose !== undefined) initialArmor.nose = frameArmor.nose;
        if (frameArmor.lateral !== undefined) initialArmor.lateral = frameArmor.lateral;
        if (frameArmor.tail !== undefined) initialArmor.tail = frameArmor.tail;
      }

      this.shipDamageState.set(shipId, {
        moduleEvents: [], // [{timestamp, module_name, event_type}]
        initialArmor: initialArmor,
        weapons: weaponsById[shipId] || {}
      });
    }

    // Process module damage and destruction events
    for (const event of this.events) {
      if (event.event_type === 'module_destroyed' && event.data) {
        const shipId = event.ship_id;
        const state = this.shipDamageState.get(shipId);
        if (state) {
          state.moduleEvents.push({
            timestamp: event.timestamp,
            module_name: event.data.module_name,
            event_type: 'destroyed'
          });
        }
      } else if (event.event_type === 'module_damaged' && event.data) {
        const shipId = event.ship_id;
        const state = this.shipDamageState.get(shipId);
        if (state) {
          state.moduleEvents.push({
            timestamp: event.timestamp,
            module_name: event.data.module_name,
            event_type: 'damaged',
            damage_gj: event.data.damage_gj || 0
          });
        }
      }
    }
  }

  /**
   * Build ship target timeline from captain_decision events
   */
  buildShipTargetTimeline() {
    this.shipTargets.clear();

    // Process captain_decision events to track targets
    for (const event of this.events) {
      if (event.event_type === 'captain_decision' && event.data) {
        const shipId = event.ship_id;
        const targetId = event.data.target_id;
        const targetName = event.data.target_name;

        if (shipId && targetId) {
          if (!this.shipTargets.has(shipId)) {
            this.shipTargets.set(shipId, []);
          }
          this.shipTargets.get(shipId).push({
            timestamp: event.timestamp,
            target_id: targetId,
            target_name: targetName
          });
        }
      }
    }
  }

  /**
   * Get ship's current target at a specific time
   * @param {string} shipId - Ship ID
   * @param {number} time - Time in seconds
   * @returns {Object|null} Target info {target_id, target_name} or null
   */
  getShipTargetAt(shipId, time) {
    const targets = this.shipTargets.get(shipId);
    if (!targets || targets.length === 0) return null;

    // Find the most recent target decision at or before the given time
    let currentTarget = null;
    for (const t of targets) {
      if (t.timestamp <= time) {
        currentTarget = t;
      } else {
        break;
      }
    }

    return currentTarget;
  }

  /**
   * Get ship damage state at a specific time
   * @param {string} shipId - Ship ID
   * @param {number} time - Time in seconds
   * @returns {Object} Damage state {armor: {nose, lateral, tail}, modules: {name: status}}
   */
  getShipDamageAt(shipId, time) {
    const state = this.shipDamageState.get(shipId);
    if (!state) return null;

    const initialArmor = state.initialArmor;

    // Get armor from sim_trace at current time
    const frameData = this.getFrameAt(time);
    let armorRemaining = {
      nose: initialArmor.nose,
      lateral: initialArmor.lateral,
      tail: initialArmor.tail
    };

    // Try to get armor from the frame (interpolate between frame0 and frame1)
    if (frameData) {
      const { frame0, frame1, alpha } = frameData;
      const ship0 = frame0.ships?.[shipId];
      const ship1 = frame1.ships?.[shipId];

      if (ship0?.armor && ship1?.armor) {
        // Interpolate armor values
        for (const section of ['nose', 'lateral', 'tail']) {
          const a0 = ship0.armor[section];
          const a1 = ship1.armor[section];
          if (a0 !== undefined && a1 !== undefined) {
            armorRemaining[section] = a0 + (a1 - a0) * alpha;
          } else if (a1 !== undefined) {
            armorRemaining[section] = a1;
          } else if (a0 !== undefined) {
            armorRemaining[section] = a0;
          }
        }
      } else if (ship1?.armor) {
        // Use frame1 armor if available
        for (const section of ['nose', 'lateral', 'tail']) {
          if (ship1.armor[section] !== undefined) {
            armorRemaining[section] = ship1.armor[section];
          }
        }
      } else if (ship0?.armor) {
        // Use frame0 armor if available
        for (const section of ['nose', 'lateral', 'tail']) {
          if (ship0.armor[section] !== undefined) {
            armorRemaining[section] = ship0.armor[section];
          }
        }
      }
    }

    // Calculate module states
    const modules = {};

    // Try to get modules from frame data (new recordings have module health)
    let hasFrameModules = false;
    if (frameData) {
      const { frame0, frame1 } = frameData;
      const ship0 = frame0.ships?.[shipId];
      const ship1 = frame1.ships?.[shipId];

      // Use frame1 modules if available (most recent state)
      const frameModules = ship1?.modules || ship0?.modules;
      if (frameModules && Object.keys(frameModules).length > 0) {
        hasFrameModules = true;
        for (const [name, moduleData] of Object.entries(frameModules)) {
          const health = moduleData.health ?? 100;
          let status = 'operational';
          if (health <= 0) {
            status = 'destroyed';
          } else if (health < 100) {
            status = 'damaged';
          }
          modules[name] = {
            type: moduleData.type || 'module',
            status: status,
            health: health
          };
        }
      }
    }

    // If no modules from sim_trace, load from ship type definition + apply events
    if (!hasFrameModules) {
      // Get ship type and load module definitions
      const shipInfo = this.ships[shipId];
      const shipType = shipInfo?.type || 'destroyer';
      const moduleDefinitions = getModulesForShipType(shipType);

      // Initialize all modules as operational
      for (const mod of moduleDefinitions) {
        modules[mod.name] = {
          type: mod.type,
          status: 'operational',
          health: 100
        };
      }

      // Build set of destroyed module names up to current time
      const destroyedModules = new Set();
      const damagedModules = new Set();
      for (const event of state.moduleEvents) {
        if (event.timestamp <= time) {
          if (event.event_type === 'destroyed') {
            destroyedModules.add(event.module_name);
          } else if (event.event_type === 'damaged') {
            damagedModules.add(event.module_name);
          }
        }
      }

      // Apply destruction/damage status
      for (const name of Object.keys(modules)) {
        if (destroyedModules.has(name)) {
          modules[name].status = 'destroyed';
          modules[name].health = 0;
        } else if (damagedModules.has(name)) {
          modules[name].status = 'damaged';
        }
      }
    }

    return {
      armor: armorRemaining,
      initialArmor: initialArmor,
      modules: modules
    };
  }

  /**
   * Build torpedo lifecycle map from events and sim_trace.
   * Each torpedo gets: launch event, terminal outcome event, and its last
   * known kinematic state from the trace (for extrapolating the final
   * sub-second of flight, which happens between 1Hz trace frames).
   */
  buildTorpedoTimeline() {
    this.torpedoes.clear();

    const ensure = (id) => {
      if (!this.torpedoes.has(id)) {
        this.torpedoes.set(id, { launch: null, outcome: null, disabledAt: null, lastState: null });
      }
      return this.torpedoes.get(id);
    };

    for (const event of this.events) {
      const d = event.data || {};
      switch (event.event_type) {
        case 'torpedo_launched':
          ensure(d.torpedo_id).launch = {
            timestamp: event.timestamp,
            shooterId: event.ship_id,
            targetId: d.target_id,
            warheadGj: d.warhead_gj || 0,
            deltaVKps: d.delta_v_kps || 0,
            launchDistanceKm: d.launch_distance_km || 0
          };
          break;
        case 'torpedo_impact':
          ensure(d.torpedo_id).outcome = {
            type: 'impact',
            timestamp: event.timestamp,
            targetId: d.target_id,
            damageGj: d.total_damage_gj || 0,
            impactSpeedKps: d.impact_speed_kps || 0,
            hitLocation: d.hit_location || 'unknown'
          };
          break;
        case 'torpedo_miss':
          ensure(d.torpedo_id).outcome = {
            type: 'miss',
            timestamp: event.timestamp,
            targetId: d.target_id || event.ship_id,
            closestApproachKm: d.closest_approach_km || 0
          };
          break;
        // Never emitted by the current sim (PD blinds rather than destroys),
        // but handled so older/future recordings render sensibly.
        case 'torpedo_intercepted':
          ensure(d.torpedo_id).outcome = {
            type: 'intercepted',
            timestamp: event.timestamp,
            targetId: d.target_id
          };
          break;
        case 'torpedo_fuel_exhausted':
          // Not terminal: the round coasts on. Record as annotation.
          ensure(d.torpedo_id).fuelExhaustedAt = event.timestamp;
          break;
        case 'pd_torpedo_disabled':
          // Seeker killed; guidance frozen. Also not terminal.
          ensure(d.torpedo_id).disabledAt = event.timestamp;
          break;
        case 'pd_torpedo_destroyed':
          ensure(d.torpedo_id).outcome = {
            type: 'destroyed',
            timestamp: event.timestamp,
            targetId: d.source_ship_id
          };
          break;
      }
    }

    // Last known state of each torpedo from the trace
    for (const frame of this.simTrace) {
      if (!frame.torpedoes) continue;
      for (const torp of frame.torpedoes) {
        const rec = ensure(torp.id);
        rec.lastState = {
          time: frame.t,
          pos: torp.pos,
          vel: torp.vel,
          source: torp.source,
          target: torp.target,
          dvRemaining: torp.dv_remaining || 0,
          disabled: !!torp.disabled
        };
      }
    }
  }

  /**
   * Torpedoes flying their final sub-second: present in events as an
   * impact/miss after their last trace frame. Extrapolate so the round
   * visually reaches its target instead of vanishing a frame early.
   * @param {number} time - Current playback time
   * @returns {Array} Torpedo states shaped like Interpolator output
   */
  getExtrapolatedTorpedoes(time) {
    const result = [];

    for (const [id, rec] of this.torpedoes) {
      const { lastState, outcome } = rec;
      if (!lastState || !outcome) continue;
      if (time <= lastState.time || time > outcome.timestamp) continue;

      const span = outcome.timestamp - lastState.time;
      if (span <= 0 || span > 3) continue; // sanity: only bridge short gaps

      let pos;
      if (outcome.type === 'impact') {
        // Steer visually into the target's position at impact time
        const targetPos = this.getShipPositionAt(outcome.targetId, outcome.timestamp);
        const t = (time - lastState.time) / span;
        if (targetPos) {
          pos = [
            lastState.pos[0] + (targetPos[0] - lastState.pos[0]) * t,
            lastState.pos[1] + (targetPos[1] - lastState.pos[1]) * t,
            lastState.pos[2] + (targetPos[2] - lastState.pos[2]) * t
          ];
        }
      }
      if (!pos) {
        const dt = time - lastState.time;
        pos = [
          lastState.pos[0] + lastState.vel[0] * dt,
          lastState.pos[1] + lastState.vel[1] * dt,
          lastState.pos[2] + lastState.vel[2] * dt
        ];
      }

      result.push({
        id: id,
        position: pos,
        velocity: lastState.vel,
        source: lastState.source,
        target: lastState.target,
        dvRemaining: lastState.dvRemaining,
        pdHeat: 0,
        disabled: lastState.disabled,
        thrusting: false,
        extrapolated: true
      });
    }

    return result;
  }

  /**
   * Get a ship's interpolated position at a given time
   * @returns {Array|null} [x, y, z] in meters
   */
  getShipPositionAt(shipId, time) {
    const frameData = this.getFrameAt(time);
    if (!frameData) return null;
    const { frame0, frame1, alpha } = frameData;
    const s0 = frame0.ships?.[shipId];
    const s1 = frame1.ships?.[shipId];
    if (s0?.pos && s1?.pos) {
      return [
        s0.pos[0] + (s1.pos[0] - s0.pos[0]) * alpha,
        s0.pos[1] + (s1.pos[1] - s0.pos[1]) * alpha,
        s0.pos[2] + (s1.pos[2] - s0.pos[2]) * alpha
      ];
    }
    return (s1?.pos || s0?.pos) ?? null;
  }

  /**
   * Torpedo lifecycle moments crossing a time window, with world positions,
   * for spawning visual effects (detonations, fizzles, seeker kills).
   * @returns {Array} [{kind, torpedoId, position, timestamp, data}]
   */
  getTorpedoMomentsInRange(startTime, endTime) {
    const moments = [];

    const posAt = (rec, t) => {
      const ls = rec.lastState;
      if (!ls) return null;
      const dt = t - ls.time;
      if (dt < 0) {
        // Moment happens while torpedo is still in the trace - find its
        // position from the frames around t
        const frameData = this.getFrameAt(t);
        if (frameData) {
          const { frame0, frame1, alpha } = frameData;
          const t0 = (frame0.torpedoes || []).find(x => x.id === rec.id);
          const t1 = (frame1.torpedoes || []).find(x => x.id === rec.id);
          if (t0 && t1) {
            return [
              t0.pos[0] + (t1.pos[0] - t0.pos[0]) * alpha,
              t0.pos[1] + (t1.pos[1] - t0.pos[1]) * alpha,
              t0.pos[2] + (t1.pos[2] - t0.pos[2]) * alpha
            ];
          }
          if (t1 || t0) return (t1 || t0).pos;
        }
        return ls.pos;
      }
      return [
        ls.pos[0] + ls.vel[0] * dt,
        ls.pos[1] + ls.vel[1] * dt,
        ls.pos[2] + ls.vel[2] * dt
      ];
    };

    for (const [id, rec] of this.torpedoes) {
      rec.id = id; // convenience for posAt

      const outcome = rec.outcome;
      if (outcome && outcome.timestamp > startTime && outcome.timestamp <= endTime) {
        let position = null;
        if (outcome.type === 'impact') {
          position = this.getShipPositionAt(outcome.targetId, outcome.timestamp) || posAt(rec, outcome.timestamp);
        } else {
          position = posAt(rec, outcome.timestamp);
        }
        if (position) {
          moments.push({
            kind: outcome.type,
            torpedoId: id,
            position: position,
            timestamp: outcome.timestamp,
            data: outcome
          });
        }
      }

      if (rec.disabledAt !== null && rec.disabledAt > startTime && rec.disabledAt <= endTime) {
        const position = posAt(rec, rec.disabledAt);
        if (position) {
          moments.push({
            kind: 'seeker_kill',
            torpedoId: id,
            position: position,
            timestamp: rec.disabledAt,
            data: {}
          });
        }
      }

      if (rec.fuelExhaustedAt && rec.fuelExhaustedAt > startTime && rec.fuelExhaustedAt <= endTime) {
        const position = posAt(rec, rec.fuelExhaustedAt);
        if (position) {
          moments.push({
            kind: 'burnout',
            torpedoId: id,
            position: position,
            timestamp: rec.fuelExhaustedAt,
            data: {}
          });
        }
      }
    }

    return moments;
  }

  /**
   * Collapse per-tick pd_fired events into continuous dwell segments.
   * The sim emits one pd_fired per turret per tick while the beam dwells on
   * a target; rendering each as a 0.3s flash misrepresented a continuous
   * beam weapon. Events <= 2s apart on the same (ship, turret, target) are
   * one dwell.
   */
  buildPDBeamSegments() {
    this.pdBeamSegments = [];
    const open = new Map();

    const pdEvents = this.events
      .filter(e => e.event_type === 'pd_fired')
      .sort((a, b) => a.timestamp - b.timestamp);

    for (const e of pdEvents) {
      const d = e.data || {};
      const key = `${e.ship_id}|${d.turret_name}|${d.target_id}`;
      const cur = open.get(key);

      if (cur && e.timestamp - cur.end <= 2.0) {
        cur.end = e.timestamp;
        cur.ticks++;
        cur.energyJ += d.energy_delivered_j || d.heat_delivered_j || 0;
      } else {
        if (cur) this.pdBeamSegments.push(cur);
        open.set(key, {
          key: `${key}|${e.timestamp}`,
          shipId: e.ship_id,
          turret: d.turret_name || 'pd',
          targetId: d.target_id,
          targetType: d.target_type || 'unknown',
          start: e.timestamp,
          end: e.timestamp,
          ticks: 1,
          energyJ: d.energy_delivered_j || d.heat_delivered_j || 0
        });
      }
    }
    for (const seg of open.values()) {
      this.pdBeamSegments.push(seg);
    }
  }

  /**
   * PD beams that should be visible at a given time.
   * Each pd_fired tick covers ~1s of dwell, so a segment stays lit until
   * end + 1.0 (plus a short fade handled by the renderer).
   */
  getActivePDBeams(time) {
    return this.pdBeamSegments.filter(s => time >= s.start && time <= s.end + 1.0);
  }

  /**
   * Build radiator state timeline from radiator_change events
   */
  buildRadiatorTimeline() {
    this.radiatorTimeline.clear();
    for (const event of this.events) {
      if (event.event_type !== 'radiator_change') continue;
      if (!this.radiatorTimeline.has(event.ship_id)) {
        this.radiatorTimeline.set(event.ship_id, []);
      }
      this.radiatorTimeline.get(event.ship_id).push({
        timestamp: event.timestamp,
        extended: !!event.data?.extended
      });
    }
  }

  /**
   * Whether a ship's radiators are extended at a given time.
   * Ships start with radiators retracted (combat stations).
   */
  getRadiatorStateAt(shipId, time) {
    const changes = this.radiatorTimeline.get(shipId);
    if (!changes) return false;
    let extended = false;
    for (const c of changes) {
      if (c.timestamp <= time) extended = c.extended;
      else break;
    }
    return extended;
  }

  /**
   * Get projectiles that visually impacted between two times
   * Returns impact info for projectiles that crossed their effectiveHitTime
   * @param {number} startTime - Start of time window
   * @param {number} endTime - End of time window
   * @returns {Array} Impact info [{projectile_id, impact_position, kinetic_energy_gj, target_id}]
   */
  getVisualImpacts(startTime, endTime) {
    const impacts = [];

    for (const [projId, hitInfo] of this.projectileHits) {
      const lastState = this.lastProjectileState.get(projId);
      if (!lastState) continue;
      if (!hitInfo.impact_position) continue;

      // Calculate effective hit time (same logic as getExtrapolatedProjectiles)
      const dx = hitInfo.impact_position[0] - lastState.pos[0];
      const dy = hitInfo.impact_position[1] - lastState.pos[1];
      const dz = hitInfo.impact_position[2] - lastState.pos[2];
      const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

      const vx = lastState.vel[0];
      const vy = lastState.vel[1];
      const vz = lastState.vel[2];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

      let travelTime = 1.0;
      if (speed > 0) {
        travelTime = distance / speed;
      }

      const effectiveHitTime = lastState.time + travelTime;

      // Check if impact crossed within this time window
      if (effectiveHitTime > startTime && effectiveHitTime <= endTime) {
        // Get hit event data for energy info
        const hitEvent = this.events.find(e =>
          e.event_type === 'hit' && e.data?.projectile_id === projId
        );

        impacts.push({
          projectile_id: projId,
          impact_position: hitInfo.impact_position,
          kinetic_energy_gj: hitEvent?.data?.kinetic_energy_gj || 1,
          target_id: hitInfo.target_id
        });
      }
    }

    return impacts;
  }
}
