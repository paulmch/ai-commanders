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

    // Extract weapons from fleet data
    const fleets = [
      { fleet: data.alpha_fleet, faction: 'alpha' },
      { fleet: data.beta_fleet, faction: 'beta' }
    ];

    for (const { fleet } of fleets) {
      if (!fleet?.ships) continue;
      for (const ship of fleet.ships) {
        const shipId = ship.ship_id;

        // Get initial armor from first frame if available
        let initialArmor = { ...DEFAULT_ARMOR };
        if (firstFrame?.ships?.[shipId]?.armor) {
          const frameArmor = firstFrame.ships[shipId].armor;
          if (frameArmor.nose !== undefined) initialArmor.nose = frameArmor.nose;
          if (frameArmor.lateral !== undefined) initialArmor.lateral = frameArmor.lateral;
          if (frameArmor.tail !== undefined) initialArmor.tail = frameArmor.tail;
        }

        // Initialize damage state with per-section armor
        this.shipDamageState.set(shipId, {
          moduleEvents: [], // [{timestamp, module_name, event_type}]
          initialArmor: initialArmor,
          weapons: ship.weapons || {}
        });
      }
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
