import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Environment, disposeTree } from './render/Environment.js';
import { getGraphicsSettings } from './render/Quality.js';
import { PostFX } from './render/PostFX.js';
import { buildHull, buildShipMaterials, ACCENT_GLOW } from './render/Hulls.js';
import { updateHullDetails } from './render/Hulls.js';
import { createTorch } from './render/Torch.js';
import { Ribbon } from './render/Trails.js';
import { Blast, ShipDestruction, WRECK_DRIFT_S } from './render/Explosion.js';
import { DamageFX } from './render/DamageFX.js';
import { getGlowTexture } from './render/Textures.js';
import { RcsSystem } from './render/Rcs.js';
import { Projectiles } from './render/Projectiles.js';
import { HeatFX } from './render/Heat.js';

/**
 * SceneManager - owns the Three.js scene, camera, renderer and every
 * battle visual. main.js drives it with interpolated state each frame.
 *
 * Scale: 1 scene unit = 1 km; recording positions are metres. Hulls are
 * exaggerated (a corvette is ~4 units long) so a fleet reads at the
 * hundreds-of-km ranges the sim fights at.
 */
export class SceneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.ships = new Map();
    this.blasts = [];               // Blast instances, tagged with a category
    this.flares = [];               // lightweight sprite pings
    this.destructionEffects = [];   // ShipDestruction instances
    this.destroyedShips = new Set();
    this.dyingPopTimes = new Map();
    this.deathInfo = {};

    this.SCALE = 1 / 1000;
    this.currentTime = 0;
    this.delta = 0;
    this.elapsed = 0;
    this.shake = 0;
    this.frameIndex = 0;

    this.init();
  }

  // ===========================================================================
  // Setup
  // ===========================================================================
  init() {
    this.scene = new THREE.Scene();
    this.clock = new THREE.Clock();

    this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.2, 800000);
    this.camera.position.set(0, 300, 400);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,               // MSAA happens on the HDR composer target
      logarithmicDepthBuffer: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    // HDR + 4x MSAA per pixel: cap the DPR so 4K laptops stay fluid
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 60000;

    // Fixed pool of point lights. three.js bakes the light COUNT into every
    // lit shader, so adding or removing a light recompiles all hull
    // materials (a visible freeze). The pool keeps the count constant:
    // effects borrow a light, position it in world space, and hand it back
    // at intensity 0. Callers must tolerate null when the pool is dry.
    this.lightRoot = new THREE.Group();
    this.lightRoot.name = 'lightPool';
    this.scene.add(this.lightRoot);
    this.lightPool = [];
    this.LIGHT_POOL = 14;
    this.LIGHT_BASE = 10;
    for (let i = 0; i < this.LIGHT_POOL; i++) {
      const l = new THREE.PointLight(0xffffff, 0, 1, 2);
      l.position.set(0, -1e6, 0);
      this.lightRoot.add(l);
      this.lightPool.push(l);
    }

    this.environment = new Environment(this.scene, this.renderer);
    this.postfx = new PostFX(this.renderer, this.scene, this.camera);

    this.createRangeRings();
    this.damageFX = new DamageFX(this);   // persistent hull damage (render/DamageFX.js)

    // Shared context for effect objects
    this.fxCtx = {
      scene: this.scene,
      camera: this.camera,
      postfx: this.postfx,
      tryLight: (c, i, d, r) => this.tryPointLight(c, i, d, r),
      freeLight: (l) => this.freePointLight(l),
      now: () => this.currentTime,
      spawnBlast: (o) => this._spawnBlast({ ...o, category: 'destruction' })
    };
    this.rcs = new RcsSystem(this.fxCtx);   // RCS attitude jets (render/Rcs.js)
    this.heat = new HeatFX(this);           // radiators + hull heat (render/Heat.js)

    this.weapons = new Projectiles(this);
    this._updatePixelWorld();
    this.warmUp();
    this._onResize = () => this.onResize();
    window.addEventListener('resize', this._onResize);
  }

  /**
   * Tear the whole viewer down before a new recording is loaded: every
   * subsystem, every GPU resource still in the graph, the listeners and
   * the renderer itself (the next SceneManager reuses the canvas).
   */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    window.removeEventListener('resize', this._onResize);
    this.controls.dispose();
    this.clearHitEffects();
    this.clearSmallEffects();
    this.clearDestructionEffects();
    this._clearBlasts('damage');
    this.tracks?.dispose();
    this.weapons.dispose();
    this.rcs.dispose();
    this.heat.dispose();
    this.damageFX.dispose();
    this.environment.dispose();
    this.postfx.dispose();
    for (const ship of this.ships.values()) {
      for (const t of ship.userData.torches || []) t.dispose();
    }
    disposeTree(this.scene);   // hulls, lights, rings: whatever is left
    this.scene.clear();
    this.ships.clear();
    getGraphicsSettings(this.renderer).dispose();
    this.renderer.renderLists.dispose();
    this.renderer.dispose();
  }

  /** Faint concentric range rings every 100 km - a tactical scale cue. */
  createRangeRings() {
    this.tacticalOverlay = new THREE.Group();
    const mat = new THREE.LineBasicMaterial({ color: 0x3a6a8a, transparent: true, opacity: 0.12, depthWrite: false });
    this.rangeRingMaterial = mat;
    for (let r = 100; r <= 600; r += 100) {
      const pts = [];
      for (let i = 0; i <= 160; i++) {
        const a = (i / 160) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
      }
      this.tacticalOverlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
    }
    this.scene.add(this.tacticalOverlay);
  }

  tryPointLight(color, intensity, distance, reserve = 0) {
    const inUse = this.LIGHT_POOL - this.lightPool.length;
    if (inUse >= this.LIGHT_BASE + reserve || !this.lightPool.length) return null;
    const l = this.lightPool.pop();
    l.color.set(color);
    l.intensity = intensity;
    l.distance = distance;
    return l;
  }

  freePointLight(light) {
    if (!light) return;
    light.intensity = 0;
    light.position.set(0, -1e6, 0);
    if (light.parent !== this.lightRoot) this.lightRoot.add(light);
    this.lightPool.push(light);
  }

  /**
   * Compile every effect shader once at load, off-screen, so the first
   * detonation in a battle doesn't stall on shader compilation.
   */
  warmUp() {
    const far = new THREE.Vector3(0, -40000, 0);
    const blast = new Blast(this.fxCtx, { position: far, scale: 1, spawnTime: -100, kind: 'reactor' });
    blast.update(-100, 0);
    const crater = new Blast(this.fxCtx, { position: far, scale: 1, spawnTime: -100, kind: 'impact' });
    crater.update(-100, 0);
    const torch = createTorch({ radius: 0.3, length: 3, tint: 'chemical' });
    torch.group.position.copy(far);
    this.scene.add(torch.group);
    torch.update(1, 0);
    const ribbon = new Ribbon({ maxPoints: 4, width: 0.1, head: [1, 1, 1], core: 8, halo: 0.2, shimmer: 0.2 });
    ribbon.setPoints([far, far.clone().add(new THREE.Vector3(1, 0, 0)), far.clone().add(new THREE.Vector3(2, 0, 0))]);
    this.scene.add(ribbon.mesh);
    const heatProbe = this.heat.warmUpProbe(far);
    this.scene.add(heatProbe);
    this.renderer.compile(this.scene, this.camera);
    this.scene.remove(heatProbe);
    heatProbe.geometry.dispose();
    heatProbe.material.dispose();
    blast.dispose();
    crater.dispose();
    this.scene.remove(torch.group);
    torch.dispose();
    this.scene.remove(ribbon.mesh);
    ribbon.dispose();
  }

  _updatePixelWorld() {
    const h = this.renderer.getDrawingBufferSize(new THREE.Vector2()).y || 1;
    Ribbon.pixelWorld.value = 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) / h;
  }

  // ===========================================================================
  // Ships
  // ===========================================================================
  createShip(shipId, faction, shipType = 'destroyer') {
    const group = new THREE.Group();
    group.name = shipId;

    const baseSizes = {
      corvette: { length: 0.5, width: 0.13 },
      frigate: { length: 0.72, width: 0.15 },
      destroyer: { length: 1.0, width: 0.22 },
      cruiser: { length: 1.3, width: 0.3 },
      cruiser_torpedo: { length: 1.3, width: 0.3 },
      battlecruiser: { length: 1.65, width: 0.3 },
      battleship: { length: 2.0, width: 0.55 },
      dreadnought: { length: 2.6, width: 0.7 },
      dreadnought_siege: { length: 2.6, width: 0.7 }
    };
    const base = baseSizes[shipType] || baseSizes.destroyer;
    const size = { length: base.length * 8, width: base.width * 8 };

    const mats = buildShipMaterials(faction);
    const engineConfig = this.getEngineConfig(shipType, size);
    let seed = 0;
    for (const c of shipId) seed = (seed * 31 + c.charCodeAt(0)) | 0;
    buildHull(shipType, group, size, mats, engineConfig, seed);

    // Fusion torches, one per bell; one budgeted light per ship
    const torches = [];
    const plumeLength = size.length * engineConfig.plumeLength * 0.45;
    const plumeR = size.width * (engineConfig.count <= 2 ? 0.2 : engineConfig.count <= 4 ? 0.155 : 0.11);
    for (let e = 0; e < engineConfig.count; e++) {
      const p = engineConfig.positions[e];
      const light = e === 0 ? this.tryPointLight(0x7fb0ff, 0, size.length * 5) : null;
      const torch = createTorch({ radius: plumeR, length: plumeLength, light, tint: 'fusion' });
      torch.group.position.set(p.x, p.y, -size.length * 0.53);
      group.add(torch.group);
      torches.push(torch);
    }

    Object.assign(group.userData, {
      shipId, faction, shipType, size, engineConfig, torches, plumeLength,
      smoothThrust: 0, wasDying: false, dyingVents: null,
      navPhase: Math.random() * 10
    });
    this.scene.add(group);
    this.ships.set(shipId, group);
    return group;
  }

  getEngineConfig(shipType, size) {
    const W = size.width;
    const ring = (n, r, offset = 0) => Array.from({ length: n }, (_, i) => {
      const a = offset + (i / n) * Math.PI * 2;
      return { x: Math.cos(a) * r, y: Math.sin(a) * r };
    });
    const configs = {
      corvette: { count: 2, positions: [{ x: -W * 0.16, y: 0 }, { x: W * 0.16, y: 0 }], plumeLength: 6 },
      frigate: { count: 2, positions: [{ x: -W * 0.17, y: 0 }, { x: W * 0.17, y: 0 }], plumeLength: 7 },
      destroyer: { count: 3, positions: [{ x: -W * 0.24, y: 0 }, { x: 0, y: 0 }, { x: W * 0.24, y: 0 }], plumeLength: 8 },
      cruiser: { count: 4, positions: [{ x: -W * 0.18, y: W * 0.18 }, { x: W * 0.18, y: W * 0.18 }, { x: -W * 0.18, y: -W * 0.18 }, { x: W * 0.18, y: -W * 0.18 }], plumeLength: 8 },
      cruiser_torpedo: { count: 4, positions: [{ x: -W * 0.18, y: W * 0.18 }, { x: W * 0.18, y: W * 0.18 }, { x: -W * 0.18, y: -W * 0.18 }, { x: W * 0.18, y: -W * 0.18 }], plumeLength: 8 },
      battlecruiser: { count: 4, positions: [{ x: -W * 0.3, y: 0 }, { x: -W * 0.1, y: 0 }, { x: W * 0.1, y: 0 }, { x: W * 0.3, y: 0 }], plumeLength: 10 },
      battleship: { count: 6, positions: [{ x: -W * 0.28, y: W * 0.13 }, { x: 0, y: W * 0.13 }, { x: W * 0.28, y: W * 0.13 }, { x: -W * 0.28, y: -W * 0.13 }, { x: 0, y: -W * 0.13 }, { x: W * 0.28, y: -W * 0.13 }], plumeLength: 10 },
      dreadnought: { count: 8, positions: ring(8, W * 0.3, Math.PI / 8), plumeLength: 12 },
      dreadnought_siege: { count: 8, positions: ring(8, W * 0.3, Math.PI / 8), plumeLength: 12 }
    };
    return configs[shipType] || configs.destroyer;
  }

  _extinguish(ship) {
    for (const t of ship.userData.torches || []) t.update(0, this.elapsed);
  }

  _setDyingVents(ship, on) {
    const ud = ship.userData;
    if (on && !ud.dyingVents) {
      const L = ud.size.length, W = ud.size.width;
      ud.dyingVents = [];
      for (let i = 0; i < 2; i++) {
        const side = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, (Math.random() - 0.5) * 0.3).normalize();
        const torch = createTorch({ radius: W * 0.08, length: L * 0.45, tint: 'chemical', intensity: 0.35 });
        torch.group.position.set(side.x * W * 0.45, side.y * W * 0.45, (Math.random() - 0.5) * L * 0.7);
        torch.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), side);
        ship.add(torch.group);
        ud.dyingVents.push({ torch, phase: Math.random() * 6 });
      }
    } else if (!on && ud.dyingVents) {
      for (const v of ud.dyingVents) { ship.remove(v.torch.group); v.torch.dispose(); }
      ud.dyingVents = null;
    }
  }

  updateShip(shipId, state) {
    let ship = this.ships.get(shipId);
    if (!ship) {
      ship = this.createShip(shipId, shipId.startsWith('alpha') ? 'alpha' : 'beta');
    }
    const ud = ship.userData;

    if (state.destroyed) {
      if (!this.destroyedShips.has(shipId)) {
        this.destroyedShips.add(shipId);
        this.createDestructionEffect(ship.position.clone(), ud.shipType, this.currentTime, shipId);
      }
      const eff = this.destructionEffects.find(e => e.shipId === shipId);
      if (eff && !eff.ignited && this.currentTime - eff.spawnTime < eff.driftDuration) {
        this._extinguish(ship);   // the destruction effect owns the hulk now
      } else {
        ship.visible = false;
      }
      this._setDyingVents(ship, false);
      this.heat.hide(ship);
      return;
    }

    ship.visible = true;
    ship.position.set(state.position[0] * this.SCALE, state.position[1] * this.SCALE, state.position[2] * this.SCALE);
    if (state.forward) {
      const f = new THREE.Vector3(state.forward[0], state.forward[1], state.forward[2]);
      if (f.lengthSq() > 0) ship.lookAt(ship.position.clone().add(f.normalize()));
    }

    const dt = this.delta || 0.016;
    const hull = Math.max(0, Math.min(100, state.hull ?? 100)) / 100;

    // Sim-side death spiral: the trace carries the tumble and the
    // sputtering torch; layer venting, fires and failing lights on top.
    if (state.dying) {
      ud.wasDying = true;
      // breaches, venting and walking secondaries: DamageFX, on battle time
    } else {
      this._setDyingVents(ship, false);
      if (this.dyingPopTimes.has(shipId)) this.dyingPopTimes.delete(shipId);
    }

    // Radiator deploy/glow and hull heat, on battle time (render/Heat.js)
    this.heat.updateShip(ship, state);

    // Windows and accents: dim and flicker as the hull fails
    const mats = ud.materials;
    if (mats) {
      let win = 2.6 * (0.35 + 0.65 * hull);
      if (hull < 0.55 || state.dying) {
        const flick = Math.sin(this.elapsed * 6.3 + ud.navPhase) * Math.sin(this.elapsed * 2.1 + ud.navPhase * 3);
        win *= 0.55 + 0.45 * (flick > (state.dying ? 0.2 : 0.55) ? 0.2 : 1.0);
      }
      mats.window.emissiveIntensity = win;
      mats.accent.emissiveIntensity = (state.dying ? 0.19 : 0.5 + 0.5 * hull) * ACCENT_GLOW;
    }

    // Torches follow a smoothed throttle
    const targetThrust = state.thrust || 0;
    ud.smoothThrust += (targetThrust - ud.smoothThrust) * (1 - Math.exp(-dt * 3.0));
    const thrust = ud.smoothThrust;
    for (const t of ud.torches) t.update(thrust, this.elapsed, state.dying ? 0.7 : 0);
    if (mats) mats.bellHeat.emissiveIntensity = 0.3 + 2.8 * thrust;
    updateHullDetails(ship, state, this);   // hull rigs, doors, nav lamps (render/HullRigs.js)
  }

  // ===========================================================================
  // Predicted paths (live mode)
  // ===========================================================================
  /**
   * Predicted paths now render through render/Tracks.js (TrajectoryPredictor
   * drives live and replay alike, see main.js); null clears them.
   */
  updatePredictedPaths(predictions) {
    if (!predictions) this.tracks?.clear();
  }

  // ===========================================================================
  // Entity lookup
  // ===========================================================================
  getEntityPosition(targetType, targetId) {
    if (targetType === 'torpedo' || targetType === 'slug' || targetType === 'projectile') {
      return this.weapons.position(targetType, targetId);
    }
    const s = this.ships.get(targetId);
    return s && s.visible ? s.position : null;
  }

  getShipPosition(shipId) {
    const ship = this.ships.get(shipId);
    if (ship && ship.visible) return ship.position.clone();
    const eff = this.destructionEffects.find(e => e.shipId === shipId);
    if (eff) return eff.position.clone();
    return ship ? ship.position.clone() : null;
  }

  // ===========================================================================
  // Blasts: impacts, detonations, pops
  // ===========================================================================
  _spawnBlast(o) {
    const blast = new Blast(this.fxCtx, o);
    blast.category = o.category || 'hit';
    this.blasts.push(blast);
    if (blast.fresh && (o.kind === 'reactor' || o.kind === 'warhead' || o.kind === 'meltdown')) {
      const d = this.camera.position.distanceTo(o.position);
      const big = o.kind !== 'warhead';
      const reach = o.scale * (big ? (o.kind === 'meltdown' ? 55 : 45) : 18);
      this.shake = Math.min(1.2, this.shake + Math.min(1, (reach / Math.max(d, 1)) ** 1.5) * (big ? 1 : 0.4));
    }
    return blast;
  }

  _updateBlasts(category, now) {
    for (let i = this.blasts.length - 1; i >= 0; i--) {
      const b = this.blasts[i];
      if (b.category !== category) continue;
      if (!b.update(now, this.elapsed)) {
        b.dispose();
        this.blasts.splice(i, 1);
      }
    }
  }

  _clearBlasts(category) {
    for (let i = this.blasts.length - 1; i >= 0; i--) {
      if (this.blasts[i].category !== category) continue;
      this.blasts[i].dispose();
      this.blasts.splice(i, 1);
    }
  }

  /**
   * Coilgun impact on a hull. Spall is thrown back along the shot's
   * approach, approximated by the direction from the nearest ship centre.
   */
  spawnHitEffect(position, energyGj, currentTime, opts = {}) {
    const pos = new THREE.Vector3(position[0] * this.SCALE, position[1] * this.SCALE, position[2] * this.SCALE);
    let dir = null, baseVel = null;
    let best = Infinity;
    for (const ship of this.ships.values()) {
      if (!ship.visible) continue;
      const d = ship.position.distanceToSquared(pos);
      if (d < best) {
        best = d;
        dir = pos.clone().sub(ship.position);
        baseVel = null;
      }
    }
    if (dir && dir.lengthSq() > 1e-6) dir.normalize(); else dir = null;
    const scale = Math.max(0.6, Math.min(3.0, 0.8 * Math.pow(Math.max(0.5, energyGj), 0.35)));
    this._spawnBlast({ position: pos, scale, spawnTime: currentTime, kind: opts.kind || 'impact', dir, baseVel, category: 'hit' });
  }

  updateHitEffects(currentTime) {
    this._updateBlasts('hit', currentTime);
  }

  clearHitEffects() {
    this._clearBlasts('hit');
  }

  spawnTorpedoDetonation(position, damageGj, currentTime) {
    const pos = new THREE.Vector3(position[0] * this.SCALE, position[1] * this.SCALE, position[2] * this.SCALE);
    // A salvo landing on one hull in the same tick records identical impact
    // points: two identical additive blasts would just double into a white
    // blowout, so they merge into one detonation sized for the sum.
    for (let i = this.blasts.length - 1; i >= 0; i--) {
      const b = this.blasts[i];
      if (b.kind !== 'warhead' || Math.abs(b.spawnTime - currentTime) > 0.05 || b.origin.distanceToSquared(pos) > 0.25) continue;
      damageGj += b.damageGj || 0;
      b.dispose();
      this.blasts.splice(i, 1);
    }
    const scale = Math.max(1.4, Math.min(3.2, 0.9 * Math.pow(Math.max(damageGj, 8), 0.3)));
    const blast = this._spawnBlast({ position: pos, scale, spawnTime: currentTime, kind: 'warhead', category: 'hit' });
    blast.damageGj = damageGj;
    return blast;
  }

  /**
   * Small transient moments: fizzles, seeker kills, intercepts, muzzle
   * flashes, hull fires, retarget pings.
   */
  spawnSmallEffect(position, currentTime, kind) {
    const pos = new THREE.Vector3(position[0] * this.SCALE, position[1] * this.SCALE, position[2] * this.SCALE);
    switch (kind) {
      case 'intercept':
      case 'destroyed':
        this._spawnBlast({ position: pos, scale: 1.4, spawnTime: currentTime, kind: 'secondary', category: 'small' });
        break;
      case 'seeker_kill':
        this._spawnFlare(pos, [1.2, 3.0, 2.8], 4.5, 1.0, currentTime);
        this._spawnBlast({ position: pos, scale: 0.8, spawnTime: currentTime, kind: 'pop', category: 'small' });
        break;
      case 'hullpop':
        this._spawnBlast({ position: pos, scale: 1.2, spawnTime: currentTime, kind: 'pop', category: 'small' });
        break;
      case 'muzzle': {
        // place the flash at the nearest ship's spinal muzzle
        let best = null, bd = Infinity;
        for (const s of this.ships.values()) {
          if (!s.visible) continue;
          const d = s.position.distanceToSquared(pos);
          if (d < bd) { bd = d; best = s; }
        }
        let at = pos;
        if (best && bd < 400) {
          const L = best.userData.size.length, W = best.userData.size.width;
          at = new THREE.Vector3(0, -W * 0.31, L * 0.55).applyQuaternion(best.quaternion).add(best.position);
        }
        this._spawnFlare(at, [1.5, 1.3, 0.95], 0.6, 0.16, currentTime);
        break;
      }
      case 'burnout':
        this._spawnFlare(pos, [2.2, 1.0, 0.4], 2.6, 1.6, currentTime);
        break;
      case 'retarget':
        this._spawnFlare(pos, [0.6, 2.4, 2.2], 4.0, 1.2, currentTime);
        break;
      case 'miss':
      default:
        this._spawnFlare(pos, [1.2, 1.4, 1.8], 3.0, 1.2, currentTime);
        break;
    }
  }

  _spawnFlare(pos, color, scale, duration, spawnTime) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: new THREE.Color(...color), transparent: true, opacity: 1,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    s.position.copy(pos);
    s.scale.setScalar(scale);
    s.renderOrder = 40;
    this.scene.add(s);
    this.flares.push({ sprite: s, spawnTime, duration, scale });
  }

  updateSmallEffects(currentTime) {
    this._updateBlasts('small', currentTime);
    for (let i = this.flares.length - 1; i >= 0; i--) {
      const f = this.flares[i];
      const age = currentTime - f.spawnTime;
      if (age > f.duration) {
        this.scene.remove(f.sprite);
        f.sprite.material.dispose();
        this.flares.splice(i, 1);
        continue;
      }
      const p = Math.max(0, age / f.duration);
      f.sprite.visible = age >= 0;
      f.sprite.material.opacity = Math.pow(1 - p, 1.8);
      f.sprite.scale.setScalar(f.scale * (0.6 + p * 1.6));
    }
  }

  clearSmallEffects() {
    this._clearBlasts('small');
    for (const f of this.flares) {
      this.scene.remove(f.sprite);
      f.sprite.material.dispose();
    }
    this.flares = [];
  }

  // ===========================================================================
  // Ship destruction
  // ===========================================================================
  createDestructionEffect(position, shipType, currentTime, shipId = null) {
    const scaleFactors = {
      corvette: 0.6, frigate: 0.8, destroyer: 1.0, cruiser: 1.3, cruiser_torpedo: 1.3,
      battlecruiser: 1.6, battleship: 2.0, dreadnought: 2.5, dreadnought_siege: 2.5
    };
    const scale = (scaleFactors[shipType] || 1.0) * 6;
    const death = (this.deathInfo || {})[shipId] || {};
    const shipGroup = shipId ? this.ships.get(shipId) : null;
    // The sim's own death spiral (dying flag) already drifted the hulk; an
    // immediate reactor kill also goes up at once. Otherwise coast dark
    // for a few seconds before the reactor lets go.
    const simDrifted = death.wasDying || shipGroup?.userData.wasDying;
    const spawnTime = death.time ?? currentTime;
    const driftDuration = (death.reactorCause || simDrifted) ? 0 : WRECK_DRIFT_S;
    const vel = death.velocity || [0, 0, 0];
    const driftVel = new THREE.Vector3(vel[0], vel[1], vel[2]).multiplyScalar(this.SCALE);
    if (death.position) position = new THREE.Vector3(...death.position).multiplyScalar(this.SCALE);
    if (shipGroup && death.forward) {
      shipGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...death.forward).normalize());
    }

    const eff = new ShipDestruction(this.fxCtx, {
      shipGroup, shipId, position, driftVel, driftDuration, scale, spawnTime,
      meltdown: !!death.meltdown   // reactor_overheat: the brighter detonation variant
    });
    this.destructionEffects.push(eff);
    return eff;
  }

  updateDestructionEffects(currentTime) {
    this.damageFX.update(currentTime);
    for (let i = this.destructionEffects.length - 1; i >= 0; i--) {
      const eff = this.destructionEffects[i];
      if (!eff.update(currentTime, this.delta || 0.016, this.elapsed)) {
        eff.dispose();
        this.destructionEffects.splice(i, 1);
      }
    }
    this._updateBlasts('destruction', currentTime);
  }

  clearDestructionEffects() {
    for (const eff of this.destructionEffects) eff.dispose();
    this.destructionEffects = [];
    this._clearBlasts('destruction');
    this.destroyedShips.clear();
    for (const ship of this.ships.values()) ship.userData.wasDying = false;
  }

  // ===========================================================================
  // Frame
  // ===========================================================================
  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.postfx.setSize(window.innerWidth, window.innerHeight);
    this._updatePixelWorld();
  }

  render() {
    this.delta = Math.min(0.1, this.clock.getDelta());
    this.elapsed = this.clock.elapsedTime;
    this.controls.update();
    const viewingDistance = this.camera.position.distanceTo(this.controls.target);
    this.camera.userData.viewDist = viewingDistance; // Environment dust-mote scale
    this.rangeRingMaterial.opacity = 0.12 * THREE.MathUtils.smoothstep(viewingDistance, 45, 250);
    this.tacticalOverlay.visible = this.rangeRingMaterial.opacity > 0.002;
    this.environment.update(this.camera, this.delta);
    this.postfx.update(this.delta);

    // Camera shake from nearby detonations, applied only for the draw
    const saved = this.camera.position.clone();
    if (this.shake > 0.001) {
      const dist = this.camera.position.distanceTo(this.controls.target);
      const amp = this.shake * dist * 0.006;
      const t = this.elapsed;
      this.camera.position.x += Math.sin(t * 37.1) * amp;
      this.camera.position.y += Math.sin(t * 43.7 + 1.3) * amp * 0.8;
      this.camera.position.z += Math.sin(t * 29.3 + 2.1) * amp * 0.6;
      this.shake *= Math.exp(-this.delta * 3.2);
    }
    this.postfx.render();
    this.camera.position.copy(saved);
    this.frameIndex++;
  }
}
