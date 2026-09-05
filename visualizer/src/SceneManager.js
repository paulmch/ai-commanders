import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Environment } from './render/Environment.js';
import { PostFX } from './render/PostFX.js';
import { buildHull, buildShipMaterials } from './render/Hulls.js';
import { createTorch } from './render/Torch.js';
import { Ribbon } from './render/Trails.js';
import { Blast, ShipDestruction } from './render/Explosion.js';
import { getGlowTexture } from './render/Textures.js';

/**
 * SceneManager - owns the Three.js scene, camera, renderer and every
 * battle visual. main.js drives it with interpolated state each frame.
 *
 * Scale: 1 scene unit = 1 km; recording positions are metres. Hulls are
 * exaggerated (a corvette is ~4 units long) so a fleet reads at the
 * hundreds-of-km ranges the sim fights at.
 */
const _Z = new THREE.Vector3(0, 0, 1);

export class SceneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.ships = new Map();
    this.projectiles = new Map();
    this.torpedoes = new Map();
    this.beams = new Map();
    this.blasts = [];               // Blast instances, tagged with a category
    this.flares = [];               // lightweight sprite pings
    this.destructionEffects = [];   // ShipDestruction instances
    this.destroyedShips = new Set();
    this.dyingPopTimes = new Map();
    this.predictedPaths = new Map();
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

    this._updatePixelWorld();
    this.warmUp();
    window.addEventListener('resize', () => this.onResize());
  }

  /** Faint concentric range rings every 100 km - a tactical scale cue. */
  createRangeRings() {
    this.tacticalOverlay = new THREE.Group();
    const mat = new THREE.LineBasicMaterial({ color: 0x3a6a8a, transparent: true, opacity: 0.16 });
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
    this.renderer.compile(this.scene, this.camera);
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
    for (const nl of ship.userData.navLights || []) nl.sprite.material.opacity = 0;
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
      this._setDyingVents(ship, true);
      for (const v of ud.dyingVents) {
        v.torch.update(0.5 + 0.5 * Math.sin(this.elapsed * 1.9 + v.phase), this.elapsed, 0.85);
      }
      let next = this.dyingPopTimes.get(shipId);
      if (next === undefined || Math.abs(next - this.currentTime) > 3.0) next = this.currentTime + 0.3 + Math.random();
      if (this.currentTime >= next) {
        const L = ud.size.length, W = ud.size.width;
        const local = new THREE.Vector3((Math.random() - 0.5) * W * 0.9, (Math.random() - 0.5) * W * 0.9, (Math.random() - 0.5) * L * 0.85);
        const world = local.clone().applyQuaternion(ship.quaternion).add(ship.position);
        const dir = local.clone().setZ(0).normalize().applyQuaternion(ship.quaternion);
        this._spawnBlast({ position: world, scale: L * 0.35, spawnTime: this.currentTime, kind: 'pop', dir, category: 'small' });
        next = this.currentTime + 0.4 + Math.random() * 1.4;
      }
      this.dyingPopTimes.set(shipId, next);
    } else {
      this._setDyingVents(ship, false);
      if (this.dyingPopTimes.has(shipId)) this.dyingPopTimes.delete(shipId);
    }

    // Radiators unfold and glow; retracted they fold flat and go dark
    const radMats = ud.radiatorMaterials || [];
    const ext = state.radiatorsExtended ? 1 : 0.16;
    const k = Math.min(1, dt * 2.5);
    const glowTarget = state.radiatorsExtended ? 1.9 : 0.0;
    for (const m of radMats) m.emissiveIntensity += (glowTarget - m.emissiveIntensity) * k;
    for (const panel of ud.radiators || []) panel.scale.y += (ext - panel.scale.y) * k;

    // Windows and accents: dim and flicker as the hull fails
    const mats = ud.materials;
    if (mats) {
      let win = 2.6 * (0.35 + 0.65 * hull);
      if (hull < 0.55 || state.dying) {
        const flick = Math.sin(this.elapsed * 6.3 + ud.navPhase) * Math.sin(this.elapsed * 2.1 + ud.navPhase * 3);
        win *= 0.55 + 0.45 * (flick > (state.dying ? 0.2 : 0.55) ? 0.2 : 1.0);
      }
      mats.window.emissiveIntensity = win;
      mats.accent.emissiveIntensity = state.dying ? 0.3 : 1.6 * (0.5 + 0.5 * hull);
    }

    // Nav lights: steady running lights, strobes on the masts
    for (const nl of ud.navLights || []) {
      if (nl.strobe) {
        const period = nl.period * 1.8;
        const ph = ((this.elapsed + nl.phase + ud.navPhase) % period) / period;
        nl.sprite.material.opacity = state.dying ? 0 : (ph < 0.02 ? 0.35 : ph < 0.035 ? 0.15 : 0.03);
      } else {
        nl.sprite.material.opacity = state.dying ? 0.15 : 0.85;
      }
    }

    // Turrets track the nearest enemy hull, slewing at a finite rate
    const rigs = ud.turrets;
    if (rigs && rigs.length) {
      let best = null, bd = Infinity;
      for (const o of this.ships.values()) {
        if (o === ship || !o.visible || o.userData.faction === ud.faction) continue;
        const d = o.position.distanceToSquared(ship.position);
        if (d < bd) { bd = d; best = o; }
      }
      const tl = best ? ship.worldToLocal(best.position.clone()) : null;
      const slew = 1 - Math.exp(-dt * 2.0);
      for (const rig of rigs) {
        let ty = 0, tp = 0.12;
        if (tl) {
          const v = tl.clone().sub(rig.mount.position).applyAxisAngle(_Z, -rig.mount.rotation.z);
          ty = Math.atan2(v.x, v.z);
          tp = THREE.MathUtils.clamp(Math.atan2(v.y, Math.hypot(v.x, v.z)), -0.04, 1.45);
        }
        let dy = ty - rig.yawAngle;
        dy = Math.atan2(Math.sin(dy), Math.cos(dy));
        rig.yawAngle += dy * slew;
        rig.pitchAngle += (tp - rig.pitchAngle) * slew;
        rig.yaw.rotation.y = rig.yawAngle;
        rig.pitch.rotation.x = -rig.pitchAngle;
      }
    }

    // Torches follow a smoothed throttle
    const targetThrust = state.thrust || 0;
    ud.smoothThrust += (targetThrust - ud.smoothThrust) * (1 - Math.exp(-dt * 3.0));
    const thrust = ud.smoothThrust;
    for (const t of ud.torches) t.update(thrust, this.elapsed, state.dying ? 0.7 : 0);
    if (mats) mats.bellHeat.emissiveIntensity = 0.3 + 2.8 * thrust;
  }

  // ===========================================================================
  // Projectiles (coilgun slugs)
  // ===========================================================================
  updateProjectile(proj) {
    let p = this.projectiles.get(proj.id);
    if (!p) {
      // A coilgun slug is a tracer: a short hot streak along its velocity
      // with a faint tail - deliberately unlike a torpedo's motor and trail
      const group = new THREE.Group();
      const core = new THREE.Sprite(new THREE.SpriteMaterial({
        map: getGlowTexture(), color: new THREE.Color(2.2, 2.0, 1.4),
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      core.scale.setScalar(0.3);
      core.renderOrder = 24;
      group.add(core);
      const tracer = new Ribbon({
        maxPoints: 2, width: 0.055, taper: 0.5, head: [3.2, 2.8, 1.6], tail: [1.6, 1.0, 0.3],
        intensity: 1.0, fadePow: 0.7, core: 8, halo: 0.2
      });
      const trail = new Ribbon({
        maxPoints: 24, width: 0.04, taper: 0.1, head: [1.4, 1.0, 0.4], tail: [0.5, 0.15, 0.02],
        intensity: 0.6, fadePow: 1.8, minStep: 0.05, maxAge: 0.5
      });
      this.scene.add(tracer.mesh);
      this.scene.add(trail.mesh);
      this.scene.add(group);
      p = { group, core, tracer, trail, dir: new THREE.Vector3(0, 0, 1) };
      this.projectiles.set(proj.id, p);
    }
    const pos = new THREE.Vector3(proj.position[0] * this.SCALE, proj.position[1] * this.SCALE, proj.position[2] * this.SCALE);
    p.group.position.copy(pos);
    if (proj.velocity) {
      const v = new THREE.Vector3(proj.velocity[0], proj.velocity[1], proj.velocity[2]);
      if (v.lengthSq() > 1e-6) p.dir.copy(v.normalize());
    }
    p.tracer.setPoints([pos, pos.clone().addScaledVector(p.dir, -1.3)]);
    if (proj.velocity) {
      const speed = Math.hypot(proj.velocity[0], proj.velocity[1], proj.velocity[2]) * this.SCALE;
      p.trail.minStep = Math.max(0.05, speed * p.trail.maxAge / (p.trail.max - 4));
    }
    p.trail.push(pos, this.currentTime);
    if (proj.pdEngaged) {
      // under laser: ablating, hotter and redder
      p.core.material.color.setRGB(3.0, 1.2, 0.5);
      p.core.scale.setScalar(0.45 + 0.1 * Math.sin(this.elapsed * 6));
    } else {
      p.core.material.color.setRGB(2.2, 2.0, 1.4);
      p.core.scale.setScalar(0.3);
    }
  }

  cleanupProjectiles(activeIds) {
    for (const [id, p] of this.projectiles) {
      if (activeIds.has(id)) continue;
      this.scene.remove(p.group);
      this.scene.remove(p.trail.mesh);
      this.scene.remove(p.tracer.mesh);
      p.core.material.dispose();
      p.trail.dispose();
      p.tracer.dispose();
      this.projectiles.delete(id);
    }
  }

  // ===========================================================================
  // Torpedoes
  // ===========================================================================
  createTorpedoMesh(torp) {
    const group = new THREE.Group();
    const isAlpha = (torp.source || '').startsWith('alpha');
    const accent = isAlpha ? 0x2ad2ff : 0xff8a3c;
    const body = new THREE.MeshStandardMaterial({ color: 0x8b9099, metalness: 0.85, roughness: 0.38 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x23262c, metalness: 0.8, roughness: 0.5 });
    const stripe = new THREE.MeshStandardMaterial({ color: accent, emissive: accent, emissiveIntensity: 1.4, roughness: 0.4 });

    const hull = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.24, 1.7, 12), body);
    hull.rotation.x = Math.PI / 2;
    group.add(hull);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.7, 12), body);
    nose.rotation.x = Math.PI / 2;
    nose.position.z = 1.2;
    group.add(nose);
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.215, 0.215, 0.12, 12), stripe);
    band.rotation.x = Math.PI / 2;
    band.position.z = 0.55;
    group.add(band);
    const tail = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.16, 0.35, 12), dark);
    tail.rotation.x = Math.PI / 2;
    tail.position.z = -1.0;
    group.add(tail);
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.34, 0.5), dark);
      fin.position.set(0, 0.3, -0.75);
      const pivot = new THREE.Group();
      pivot.add(fin);
      pivot.rotation.z = (i / 4) * Math.PI * 2;
      group.add(pivot);
    }
    const light = this.tryPointLight(0xffb070, 0, 40);
    const torch = createTorch({ radius: 0.28, length: 7, light, tint: 'chemical' });
    torch.group.position.z = -1.2;
    group.add(torch.group);

    // Faint faction diamond at constant screen size so a torpedo stays
    // identifiable at range, where its body is sub-pixel
    const marker = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.getDiamondTexture(), color: accent, transparent: true, opacity: 0.32,
      sizeAttenuation: false, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    marker.scale.set(0.022, 0.022, 1);
    marker.renderOrder = 50;
    group.add(marker);

    const c = new THREE.Color(accent);
    const trail = new Ribbon({
      maxPoints: 64, width: 0.32, taper: 0.15, head: [c.r * 1.8, c.g * 1.8, c.b * 1.8],
      tail: [c.r * 0.5, c.g * 0.4, c.b * 0.4], intensity: 0.9, fadePow: 1.3, minStep: 0.08, maxAge: 2.5
    });
    group.userData = {
      torch, light, trail, stripe, body, accent, marker, phase: Math.random() * Math.PI * 2, tumble: new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize()
    };
    return group;
  }

  updateTorpedo(torp) {
    let t = this.torpedoes.get(torp.id);
    if (!t) {
      t = this.createTorpedoMesh(torp);
      this.scene.add(t);
      this.scene.add(t.userData.trail.mesh);
      this.torpedoes.set(torp.id, t);
    }
    const pos = new THREE.Vector3(torp.position[0] * this.SCALE, torp.position[1] * this.SCALE, torp.position[2] * this.SCALE);
    t.position.copy(pos);
    const ud = t.userData;

    // Retargeting ping + lock line (data-driven from the trace)
    if (ud.currentTargetId === undefined) {
      ud.currentTargetId = torp.target;
    } else if (torp.target && torp.target !== ud.currentTargetId) {
      ud.currentTargetId = torp.target;
      ud.retargetAt = this.currentTime;
      this.spawnSmallEffect([pos.x / this.SCALE, pos.y / this.SCALE, pos.z / this.SCALE], this.currentTime, 'retarget');
    }
    this._updateRetargetLine(t, torp, pos);

    if (!torp.disabled && torp.velocity) {
      const v = new THREE.Vector3(torp.velocity[0], torp.velocity[1], torp.velocity[2]);
      if (v.lengthSq() > 0) t.lookAt(pos.clone().add(v.normalize()));
    } else {
      t.rotateOnAxis(ud.tumble, (this.delta || 0.016) * 1.1);
    }

    if (torp.disabled) {
      ud.stripe.emissiveIntensity = 0.15;
      ud.marker.material.opacity = 0.1;
      ud.torch.update(0, this.elapsed);
    } else {
      ud.marker.material.opacity = 0.32;
      ud.stripe.emissiveIntensity = 1.4;
      ud.torch.update(torp.thrusting ? 0.9 : 0, this.elapsed);
    }
    if (!torp.disabled) {
      // spacing follows speed so the 2.5 s trail always fits the point budget
      const speed = torp.velocity ? Math.hypot(torp.velocity[0], torp.velocity[1], torp.velocity[2]) * this.SCALE : 10;
      ud.trail.minStep = Math.max(0.05, speed * ud.trail.maxAge / (ud.trail.max - 4));
      ud.trail.push(pos, this.currentTime);
    }
  }

  _updateRetargetLine(torpedo, torp, pos) {
    const RETARGET_LINE_S = 4.0;
    const age = this.currentTime - (torpedo.userData.retargetAt ?? -Infinity);
    const targetPos = (age >= 0 && age <= RETARGET_LINE_S) ? this.getEntityPosition('ship', torp.target) : null;
    let line = torpedo.userData.retargetLine;
    if (!targetPos) {
      if (line) line.visible = false;
      return;
    }
    if (!line) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      line = new THREE.Line(geom, new THREE.LineBasicMaterial({
        color: 0x66ffee, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      torpedo.userData.retargetLine = line;
      this.scene.add(line);
    }
    const arr = line.geometry.attributes.position.array;
    arr[0] = pos.x; arr[1] = pos.y; arr[2] = pos.z;
    arr[3] = targetPos.x; arr[4] = targetPos.y; arr[5] = targetPos.z;
    line.geometry.attributes.position.needsUpdate = true;
    line.material.opacity = 0.55 * (1 - age / RETARGET_LINE_S);
    line.visible = true;
  }

  cleanupTorpedoes(activeIds) {
    for (const [id, t] of this.torpedoes) {
      if (activeIds.has(id)) continue;
      this.scene.remove(t);
      this.scene.remove(t.userData.trail.mesh);
      t.userData.trail.dispose();
      t.userData.torch.dispose();
      t.userData.marker.material.dispose();
      if (t.userData.retargetLine) {
        this.scene.remove(t.userData.retargetLine);
        t.userData.retargetLine.geometry.dispose();
        t.userData.retargetLine.material.dispose();
      }
      this.freePointLight(t.userData.light);
      this.torpedoes.delete(id);
    }
  }

  // ===========================================================================
  // Predicted paths (live mode)
  // ===========================================================================
  getDiamondTexture() {
    if (this._diamondTexture) return this._diamondTexture;
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.translate(size / 2, size / 2);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = 'rgba(255,255,255,1)';
    ctx.lineWidth = 5;
    ctx.strokeRect(-15, -15, 30, 30);
    this._diamondTexture = new THREE.CanvasTexture(canvas);
    return this._diamondTexture;
  }

  updatePredictedPaths(predictions) {
    const active = new Set();
    for (const [shipId, pred] of Object.entries(predictions?.ships || {})) {
      const path = pred.path || [];
      if (path.length < 2 || !pred.checkpoint_pos) continue;
      active.add(shipId);
      this._removePredictedPath(shipId);
      const baseColor = new THREE.Color((shipId || '').startsWith('alpha') ? 0x2ad2ff : 0xff8a3c);
      const n = path.length;
      const positions = new Float32Array(n * 3);
      const colors = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        positions[i * 3] = path[i][0] * this.SCALE;
        positions[i * 3 + 1] = path[i][1] * this.SCALE;
        positions[i * 3 + 2] = path[i][2] * this.SCALE;
        const fade = 1 - 0.85 * (i / (n - 1));
        colors[i * 3] = baseColor.r * fade;
        colors[i * 3 + 1] = baseColor.g * fade;
        colors[i * 3 + 2] = baseColor.b * fade;
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      this.scene.add(line);
      const marker = new THREE.Sprite(new THREE.SpriteMaterial({
        map: this.getDiamondTexture(), color: baseColor.clone().multiplyScalar(1.4),
        transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      marker.position.set(pred.checkpoint_pos[0] * this.SCALE, pred.checkpoint_pos[1] * this.SCALE, pred.checkpoint_pos[2] * this.SCALE);
      marker.scale.set(4, 4, 1);
      this.scene.add(marker);
      this.predictedPaths.set(shipId, { line, marker });
    }
    for (const shipId of [...this.predictedPaths.keys()]) {
      if (!active.has(shipId)) this._removePredictedPath(shipId);
    }
  }

  _removePredictedPath(shipId) {
    const entry = this.predictedPaths.get(shipId);
    if (!entry) return;
    this.scene.remove(entry.line);
    entry.line.geometry.dispose();
    entry.line.material.dispose();
    this.scene.remove(entry.marker);
    entry.marker.material.dispose();
    this.predictedPaths.delete(shipId);
  }

  // ===========================================================================
  // Entity lookup
  // ===========================================================================
  getEntityPosition(targetType, targetId) {
    if (targetType === 'torpedo') {
      const t = this.torpedoes.get(targetId);
      return t ? t.position : null;
    }
    if (targetType === 'slug' || targetType === 'projectile') {
      const p = this.projectiles.get(targetId);
      return p ? p.group.position : null;
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
  // PD beams
  // ===========================================================================
  updateContinuousBeams(activeSegments, time) {
    this.currentTime = time;
    const activeKeys = new Set();
    for (const seg of activeSegments) {
      const sourceShip = this.ships.get(seg.shipId);
      const sourcePos = sourceShip && sourceShip.visible ? sourceShip.position : null;
      const targetPos = this.getEntityPosition(seg.targetType, seg.targetId);
      if (!sourcePos || !targetPos) continue;
      activeKeys.add(seg.key);
      let beam = this.beams.get(seg.key);
      if (!beam) {
        beam = this.createBeamMesh(seg, sourceShip);
        this.beams.set(seg.key, beam);
      }
      const from = beam.muzzleOffset.clone().applyQuaternion(sourceShip.quaternion).add(sourcePos);
      const dir = targetPos.clone().sub(from);
      const length = dir.length();
      if (length < 0.5) continue;
      dir.normalize();

      const fadeIn = Math.min(1, (time - seg.start) / 0.25 + 0.15);
      const fadeOut = Math.min(1, Math.max(0, (seg.end + 1.0 - time) / 0.5));
      const flicker = 0.94 + 0.045 * Math.sin(this.elapsed * 6.7 + beam.phase) + 0.015 * Math.sin(this.elapsed * 9.3 + beam.phase * 2);
      const intensity = Math.min(fadeIn, fadeOut) * flicker;

      beam.ribbon.setPoints([from, targetPos]);
      beam.ribbon.material.uniforms.uIntensity.value = 0.85 * intensity;
      beam.ribbon.material.uniforms.uTime.value = this.elapsed;
      beam.muzzle.position.copy(from);
      beam.muzzle.material.opacity = 0.6 * intensity;
      beam.impact.position.copy(targetPos);
      beam.impact.material.opacity = 0.6 * intensity;
      beam.impact.scale.setScalar(1.6 * (1 + 0.12 * Math.sin(this.elapsed * 7.7 + beam.phase)));
      beam.ribbon.mesh.visible = true;
      beam.muzzle.visible = true;
      beam.impact.visible = true;

      // Ablation sparks at the impact point while the beam dwells
      if (time - beam.lastSpark > 0.45 && intensity > 0.5) {
        beam.lastSpark = time;
        this._spawnBlast({
          position: targetPos.clone(), scale: 0.6, spawnTime: time, kind: 'ablate',
          dir: dir.clone().negate(), category: 'small'
        });
      }
    }
    for (const [key, beam] of this.beams) {
      if (!activeKeys.has(key)) {
        this.disposeBeam(beam);
        this.beams.delete(key);
      }
    }
  }

  createBeamMesh(seg, sourceShip) {
    const ribbon = new Ribbon({
      maxPoints: 2, width: 0.075, taper: 1.0, head: [1.0, 0.72, 0.5], tail: [1.0, 0.55, 0.35],
      intensity: 0.85, fadePow: 0.15, core: 10, halo: 0.25, shimmer: 0.3
    });
    ribbon.mesh.renderOrder = 23;
    this.scene.add(ribbon.mesh);
    const muzzle = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: new THREE.Color(1.4, 1.0, 0.7), transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    muzzle.scale.setScalar(0.5);
    const impact = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: new THREE.Color(1.6, 1.2, 0.85), transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    impact.scale.setScalar(0.9);
    this.scene.add(muzzle);
    this.scene.add(impact);
    // Deterministic per-turret muzzle offset in ship-local space
    let h = 0;
    for (const c of seg.turret || '') h = (h * 31 + c.charCodeAt(0)) | 0;
    const W = sourceShip?.userData.size?.width || 2;
    const L = sourceShip?.userData.size?.length || 8;
    const muzzleOffset = new THREE.Vector3(
      ((h & 0xff) / 255 - 0.5) * W * 0.9,
      (((h >> 8) & 0xff) / 255 - 0.5) * W * 0.9,
      (((h >> 16) & 0xff) / 255 - 0.5) * L * 0.6
    );
    return { ribbon, muzzle, impact, muzzleOffset, phase: Math.random() * Math.PI * 2, lastSpark: -10 };
  }

  disposeBeam(beam) {
    this.scene.remove(beam.ribbon.mesh);
    this.scene.remove(beam.muzzle);
    this.scene.remove(beam.impact);
    beam.ribbon.dispose();
    beam.muzzle.material.dispose();
    beam.impact.material.dispose();
  }

  clearPDBeams() {
    for (const beam of this.beams.values()) this.disposeBeam(beam);
    this.beams.clear();
  }

  // ===========================================================================
  // Blasts: impacts, detonations, pops
  // ===========================================================================
  _spawnBlast(o) {
    const blast = new Blast(this.fxCtx, o);
    blast.category = o.category || 'hit';
    this.blasts.push(blast);
    if (blast.fresh && (o.kind === 'reactor' || o.kind === 'warhead')) {
      const d = this.camera.position.distanceTo(o.position);
      const reach = o.scale * (o.kind === 'reactor' ? 45 : 18);
      this.shake = Math.min(1.2, this.shake + Math.min(1, (reach / Math.max(d, 1)) ** 1.5) * (o.kind === 'reactor' ? 1 : 0.4));
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
    const scale = Math.max(1.4, Math.min(3.2, 0.9 * Math.pow(Math.max(damageGj, 8), 0.3)));
    this._spawnBlast({ position: pos, scale, spawnTime: currentTime, kind: 'warhead', category: 'hit' });
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
    const simDrifted = shipGroup?.userData.wasDying;
    const driftDuration = (death.reactorCause || simDrifted) ? 0 : 4.5 + Math.random() * 1.5;
    const vel = death.velocity || [0, 0, 0];
    const driftVel = new THREE.Vector3(vel[0], vel[1], vel[2]).multiplyScalar(this.SCALE);

    const eff = new ShipDestruction(this.fxCtx, {
      shipGroup, shipId, position, driftVel, driftDuration, scale, spawnTime: currentTime
    });
    this.destructionEffects.push(eff);
    return eff;
  }

  updateDestructionEffects(currentTime) {
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
