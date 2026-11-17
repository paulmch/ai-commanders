import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Environment, disposeTree } from './render/Environment.js';
import { getGraphicsSettings } from './render/Quality.js';
import { PostFX } from './render/PostFX.js';
import { buildHull, buildShipMaterials, hullSize, ACCENT_GLOW } from './render/Hulls.js';
import { updateHullDetails } from './render/Hulls.js';
import { createTorch, damagedNozzle } from './render/Torch.js';

const _nozzle = { level: 1, sputter: 0 };

// three clamps a point light's inverse-square falloff at 0.1 scene units
// (1 / max(d^2, 0.01)): harmless while hulls were kilometres long, but at
// real size - 1 unit = 1 km, a 65 m corvette - every lamp within 100 m of
// its plating would light it flat. Lower the clamp to 1 m; a pool light's
// intensity is scaled by the borrower with the hull's draw scale squared.
{
  const chunk = THREE.ShaderChunk.lights_pars_begin;
  const clamp = 'max( pow( lightDistance, decayExponent ), 0.01 )';
  if (chunk.includes(clamp)) {
    THREE.ShaderChunk.lights_pars_begin = chunk.replace(clamp, 'max( pow( lightDistance, decayExponent ), 0.000001 )');
  }
}

const _plumeTurn = new THREE.Vector3();
const _plumeQ = new THREE.Quaternion();
import { Ribbon } from './render/Trails.js';
import { Blast, ShipDestruction, blastExtentOver } from './render/Explosion.js';
import { wreckDriftS } from './DamageTimeline.js';
import { DamageFX } from './render/DamageFX.js';
import { getGlowTexture } from './render/Textures.js';
import { RcsSystem } from './render/Rcs.js';
import { Projectiles } from './render/Projectiles.js';
import { HeatFX } from './render/Heat.js';
import { hullQuat, validUp } from './HullFrame.js';
import {
  shipScale, freeScale, framingLength, normalizeScaleMode, nextScaleMode, trueLengthM, DEFAULT_SCALE_MODE, FREE_TRUE_K
} from './render/Scale.js';
import { setCloseRangeUnit } from './render/CloseRange.js';
const _hfFwd = [0, 0, 1], _hfQuat = [0, 0, 0, 1];

/**
 * Put the canvas's WebGL pixel-unpack state back to its defaults when a
 * renderer is torn down. The context outlives the renderer (the next
 * SceneManager - a live battle switch, Load New Recording - gets the same
 * one from canvas.getContext), and three's new WebGLState allocates its
 * empty 3D / 2D-array textures with texImage3D before it touches the
 * unpack flags: left at UNPACK_FLIP_Y = true by the last image upload, both
 * fail with INVALID_OPERATION (two console warnings per new scene).
 */
export function resetUnpackState(gl) {
  if (!gl?.pixelStorei) return;
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
}

/**
 * SceneManager - owns the Three.js scene, camera, renderer and every
 * battle visual. main.js drives it with interpolated state each frame.
 *
 * Scale: 1 scene unit = 1 km; recording positions are metres. Hulls are
 * BUILT in kit units (a corvette is 4 long, render/Hulls.js) and DRAWN by
 * one uniform factor per ship, `userData.k` (scene km per kit unit), set
 * every frame by updateScale() from render/Scale.js:
 *   scale=auto       real size up close, a readable floor at distance (default)
 *   scale=true       always the real 65-275 m hull
 *   scale=cinematic  k = 1, the old fixed ~60x exaggeration
 * `userData.size` stays the kit size (hull-local code: fittings, plumes,
 * decals); `userData.drawn` is the size on the scene's km grid right now.
 * Anything in world space that belongs to a hull multiplies by its k
 * (kOf); free-flying effects authored in kit units use kAt(point).
 */
const SCALE_STORAGE_KEY = 'aic.scaleMode';
/** Share of the engine's blast radius the reactor fireball fills with real-size hulls. */
const REACTOR_FIREBALL_SHARE = 0.55;
/**
 * Blast radius (km) per class for recordings without reactor_blast events:
 * the engine's 15 km x sqrt(dry mass / cruiser dry mass) (src/battle_damage.py).
 */
const NOMINAL_BLAST_KM = {
  corvette: 10.6, frigate: 10.6, destroyer: 13, cruiser: 15, cruiser_torpedo: 15, battlecruiser: 15,
  battleship: 18.4, dreadnought: 21.2, dreadnought_siege: 22.4
};
/** Camera limits per regime: [near plane, closest orbit] in km. */
const CAMERA_LIMITS = { cinematic: [0.2, 1.5], real: [0.0005, 0.012] };

function initialScaleMode() {
  const q = new URLSearchParams(globalThis.location?.search || '').get('scale');
  if (q) return normalizeScaleMode(q);
  try { return normalizeScaleMode(globalThis.localStorage?.getItem(SCALE_STORAGE_KEY) || DEFAULT_SCALE_MODE); }
  catch { return DEFAULT_SCALE_MODE; }
}

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
    this.scaleMode = initialScaleMode();
    this.scaleListeners = new Set();
    this.currentTime = 0;
    this.delta = 0;
    this.elapsed = 0;
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
    // Sun shadows (high tier): always "enabled" on the renderer - whether a
    // shadow map is actually rendered is the sun light's castShadow, which
    // Environment drives from the quality preset (off = zero cost).
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
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
    this._applyCameraLimits();

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
      spawnBlast: (o) => this._spawnBlast({ ...o, category: 'destruction' }),
      // draw scale (render/Scale.js): of a hull, and of a free point
      kOf: (shipId) => this.kOf(shipId),
      kAt: (point) => this.kAt(point)
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
    this._disposeWarm();
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
    resetUnpackState(this.renderer.getContext?.());
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

  // ===========================================================================
  // Scale (render/Scale.js)
  // ===========================================================================
  /** Near plane and closest orbit for the current scale mode. */
  _applyCameraLimits() {
    const [near, minDistance] = CAMERA_LIMITS[this.scaleMode === 'cinematic' ? 'cinematic' : 'real'];
    if (this.camera.near !== near) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
    this.controls.minDistance = minDistance;
    // Lengths tuned on the kit-size hulls (shadow box floor, AO reach,
    // close-up thresholds) follow the mode: scene km per kit unit
    const unit = this.scaleUnit();
    if (this.environment) this.environment.scaleUnit = unit;
    if (this.postfx) this.postfx.worldUnit = unit;
    setCloseRangeUnit(unit);
  }

  /** Scene km per kit unit for constants tuned on the kit-size hulls (1 in cinematic). */
  scaleUnit() {
    return this.scaleMode === 'cinematic' ? 1 : FREE_TRUE_K;
  }

  /**
   * Switch between auto / true / cinematic. The camera keeps looking at the
   * same point from the same direction; its distance is rescaled so a hull
   * framed in one mode is framed the same way in the next.
   */
  setScaleMode(mode, { persist = true, reframe = true } = {}) {
    const next = normalizeScaleMode(mode);
    if (next === this.scaleMode) return next;
    const before = this.scaleMode;
    this.scaleMode = next;
    this._applyCameraLimits();
    const flip = (before === 'cinematic') !== (next === 'cinematic');
    if (reframe && flip) this._rescaleOrbit(before, next);
    // a reactor blast is sized per mode: let the wrecks respawn from their recorded times
    if (flip) this.clearDestructionEffects();
    if (persist) { try { globalThis.localStorage?.setItem(SCALE_STORAGE_KEY, next); } catch { /* private mode */ } }
    this.updateScale();
    for (const fn of this.scaleListeners) fn(next, before);
    return next;
  }

  cycleScaleMode() { return this.setScaleMode(nextScaleMode(this.scaleMode)); }

  /** A close-up on a hull stays a close-up across a cinematic <-> real switch. */
  _rescaleOrbit(before, next) {
    const target = this.controls.target;
    let near = null, best = Infinity;
    for (const ship of this.ships.values()) {
      if (!ship.visible) continue;
      const d = ship.position.distanceTo(target);
      if (d < best) { best = d; near = ship; }
    }
    if (!near) return;
    const type = near.userData.shipType;
    const from = framingLength(before, type), to = framingLength(next, type);
    const view = this.camera.position.clone().sub(target);
    // only when the camera is on that hull (within 12 of its lengths)
    if (best > from * 1.5 || view.length() > from * 12) return;
    target.copy(near.position);
    this.camera.position.copy(target).addScaledVector(view, to / from);
  }

  /** Scene km per kit unit a ship is drawn at this frame (1 in cinematic). */
  kOf(shipId) {
    return this.ships.get(shipId)?.userData.k ?? this.kAt(null);
  }

  /** The same for a free-flying effect authored in kit units at a world point. */
  kAt(point) {
    if (this.scaleMode === 'cinematic') return 1;
    const d = point ? this.camera.position.distanceTo(point) : 0;
    return freeScale(this.scaleMode, d, this.camera.fov);
  }

  /**
   * The hull length camera framing is measured in (scene km): static per
   * mode, so a framing never depends on where the camera was before.
   */
  framingLength(shipId) {
    return framingLength(this.scaleMode, this.ships.get(shipId)?.userData.shipType);
  }

  /**
   * Settle the orbit controls and the hull draw scales for this frame.
   * main.js calls it once the camera has been placed (director, follow,
   * user input) and BEFORE the frame's effects are evaluated, so a hull,
   * its mounts and everything attached to it use one scale per frame -
   * also on the frame of a director cut.
   */
  syncCamera() {
    this.controls.update();
    this._cameraSynced = true;
    this.updateScale();
  }

  /**
   * Draw scale of every hull for this frame: a pure function of the camera
   * (position, fov) and each ship's position. Call once per frame after the
   * camera is final and before anything reads a hull's world matrix.
   */
  updateScale() {
    const cam = this.camera.position, fov = this.camera.fov, mode = this.scaleMode;
    for (const [shipId, ship] of this.ships) {
      const ud = ship.userData;
      let at = ship.position;
      if (!ship.visible) {
        const eff = this.destructionEffects.find(e => e.shipId === shipId);
        if (eff) at = eff.position;
      }
      const k = shipScale(mode, ud.shipType, cam.distanceTo(at), fov);
      if (k !== ud.k) {
        ud.k = k;
        ship.scale.setScalar(k);
        ud.drawn.length = ud.size.length * k;
        ud.drawn.width = ud.size.width * k;
      }
    }
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
    // Compile the variants that are actually drawn: the scene renders into
    // PostFX's HDR target (linear output, no tone mapping), and compiling
    // for the canvas built programs nothing ever used.
    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.postfx.sceneRT || null);
    this.renderer.compile(this.scene, this.camera);
    this.renderer.setRenderTarget(prevTarget);
    // compile() only queues the work (parallel shader compile): the driver
    // finishes it at the program's first query. Query them now, at load,
    // instead of at the first kill.
    for (const program of this.renderer.info.programs || []) program.getUniforms?.();
    // Keep the warm-up copies, hidden, for the scene's lifetime: three
    // deletes a program with the last material using it, so disposing them
    // here threw the compiled programs away and the first kill of a battle
    // compiled every blast shader again mid-fight (a multi-second stall on
    // software GL). They hold no pool light and are never updated.
    for (const b of [blast, crater]) {
      this.freePointLight(b.light);
      b.light = null;
      b.group.visible = false;
    }
    torch.group.visible = false;
    ribbon.mesh.visible = false;
    heatProbe.visible = false;
    this._warm = { blast, crater, torch, ribbon, heatProbe };
  }

  /** Release the warm-up copies (SceneManager.dispose). */
  _disposeWarm() {
    const w = this._warm;
    if (!w) return;
    this._warm = null;
    w.blast.dispose();
    w.crater.dispose();
    this.scene.remove(w.torch.group);
    w.torch.dispose();
    this.scene.remove(w.ribbon.mesh);
    w.ribbon.dispose();
    this.scene.remove(w.heatProbe);
    w.heatProbe.geometry.dispose();
    w.heatProbe.material.dispose();
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

    const size = hullSize(shipType);

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
      const torch = createTorch({ radius: plumeR, length: plumeLength, light, tint: 'fusion', cluster: engineConfig.count,
        phase: (e * 0.6180339887) % 1 });
      torch.group.position.set(p.x, p.y, -size.length * 0.53);
      // Every plume runs exactly down its bell's axis (the ship's thrust
      // axis): a cluster is a bundle of parallel, collimated jets. The only
      // deflection is the recorded turn sweep (render/Torch.js).
      group.add(torch.group);
      torches.push(torch);
    }

    const k = shipScale(this.scaleMode, shipType, this.camera.position.distanceTo(group.position), this.camera.fov);
    group.scale.setScalar(k);
    Object.assign(group.userData, {
      shipId, faction, shipType, size, engineConfig, torches, plumeLength,
      k, drawn: { length: size.length * k, width: size.width * k }, trueLengthKm: trueLengthM(shipType) / 1000,
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
      if (f.lengthSq() > 0) {
        f.normalize();
        if (validUp(state.up)) {
          // the ship's recorded roll: nose along fwd, dorsal along up
          const q = hullQuat(f.toArray(_hfFwd), state.up, _hfQuat);
          ship.quaternion.set(q[0], q[1], q[2], q[3]);
        } else {
          // older recordings carry no roll: nose along fwd, world-up +Y
          ship.lookAt(ship.position.clone().add(f));
        }
      }
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

    // Torches follow the recorded throttle (already eased between trace
    // frames), never a wall-clock filter: a scrub shows the same plume
    const thrust = state.thrust || 0;
    ud.smoothThrust = thrust;
    const rate = state.dying ? 0 : (state.thrustRate || 0);
    // Plume sweep: the recorded turn rate in the hull frame (render/Torch.js)
    let turn = null;
    if (Array.isArray(state.omega) && !state.dying) {
      turn = _plumeTurn.fromArray(state.omega).applyQuaternion(_plumeQ.copy(ship.quaternion).invert());
    }
    // Drive damage (recorded mount_damaged, kind engine): dark bells, the
    // rest burning rough; the trace's thrust already carries the loss
    const eng = this.damageFX?.engineAt?.(shipId, this.currentTime) || null;
    for (let i = 0; i < ud.torches.length; i++) {
      const dying = state.dying ? 0.7 : 0;
      if (!eng) { ud.torches[i].update(thrust, this.elapsed, dying, this.currentTime, turn, rate); continue; }
      if (ud.driveSeed === undefined) {
        let h = 0;
        for (const c of shipId) h = (h * 31 + c.charCodeAt(0)) | 0;
        ud.driveSeed = Math.abs(h) % 100000;   // from the id: the same bells die on every load
      }
      const nz = damagedNozzle(eng.health, i, ud.torches.length, ud.driveSeed, _nozzle);
      ud.torches[i].update(Math.min(1, thrust * nz.level), this.elapsed, Math.max(dying, nz.sputter), this.currentTime, turn, rate);
    }
    if (mats) mats.bellHeat.emissiveIntensity = 0.3 + 2.8 * thrust;
    updateHullDetails(ship, state, this);   // hull rigs, doors, nav lamps (render/HullRigs.js)
  }

  /**
   * Hull self-shadowing: every lit (MeshStandard) hull mesh receives the
   * sun shadow, the opaque ones cast it. Flagged by material type, not by
   * name, and re-swept every few frames so parts other modules add later
   * (rigs, radiators, breach geometry) join in without knowing about it.
   * Plumes, sprites, beams and lamp instances are other material types and
   * never cast. Flags are inert while the sun light casts no shadow (low).
   */
  _flagShadowMeshes() {
    for (const ship of this.ships.values()) {
      if (!ship.visible) continue;
      ship.traverse(_flagShadow);
    }
    // Wreck sections and chunks live outside the ship groups and fade out
    // through a transparent material: they receive, and cast while solid.
    for (const eff of this.destructionEffects) {
      eff.chunkGroup?.traverse(_flagWreckShadow);
    }
  }

  /** Visible hulls as { pos, r } for the shadow fit (reused, no per-frame garbage). */
  _collectShadowSubjects() {
    const list = this._shadowSubjects || (this._shadowSubjects = []);
    let n = 0;
    for (const ship of this.ships.values()) {
      if (!ship.visible) continue;
      const s = list[n] || (list[n] = { pos: null, r: 0, a: 0 });
      s.pos = ship.position;
      s.r = (ship.userData.drawn?.length || 8) * 0.6;
      n++;
    }
    return n;
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
   * Coilgun impact on a hull. With a recording loaded the persistent hit
   * blasts (strike, breach, graze, spall) are spawned by DamageFX from the
   * damage timeline, placed on the displayed hull surface and carried with
   * the struck ship - this legacy path only serves callers without one.
   * Spall is thrown back along the shot's approach, approximated by the
   * direction from the nearest ship centre.
   */
  spawnHitEffect(position, energyGj, currentTime, opts = {}) {
    if (this.damageFX?.ownsImpacts()) return null;
    const pos = new THREE.Vector3(position[0] * this.SCALE, position[1] * this.SCALE, position[2] * this.SCALE);
    let dir = null, baseVel = null, struck = null;
    let best = Infinity;
    for (const ship of this.ships.values()) {
      if (!ship.visible) continue;
      const d = ship.position.distanceToSquared(pos);
      if (d < best) {
        best = d;
        struck = ship;
        dir = pos.clone().sub(ship.position);
        baseVel = null;
      }
    }
    if (dir && dir.lengthSq() > 1e-6) dir.normalize(); else dir = null;
    const scale = Math.max(0.6, Math.min(3.0, 0.8 * Math.pow(Math.max(0.5, energyGj), 0.35)));
    this._spawnBlast({ position: pos, scale, spawnTime: currentTime, kind: opts.kind || 'impact', dir, baseVel, category: 'hit',
      shipId: struck?.userData.shipId });
  }

  updateHitEffects(currentTime) {
    this._updateBlasts('hit', currentTime);
  }

  clearHitEffects() {
    this._clearBlasts('hit');
  }

  spawnTorpedoDetonation(position, damageGj, currentTime) {
    // Torpedo strikes come from DamageFX too (kinetic strike / breach, the
    // warhead fireball only for rounds with a recorded explosive yield)
    if (this.damageFX?.ownsImpacts()) return null;
    const pos = new THREE.Vector3(position[0] * this.SCALE, position[1] * this.SCALE, position[2] * this.SCALE);
    // A salvo landing on one hull in the same tick records identical impact
    // points: two identical additive blasts would just double into a white
    // blowout, so they merge into one detonation sized for the sum.
    for (let i = this.blasts.length - 1; i >= 0; i--) {
      const b = this.blasts[i];
      if (b.kind !== 'warhead' || Math.abs(b.spawnTime - currentTime) > 0.05 || b.origin.distanceToSquared(pos) > 0.25 * this.kAt(pos) ** 2) continue;
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
        if (best && bd < 400 * best.userData.k ** 2) {
          const L = best.userData.drawn.length, W = best.userData.drawn.width;
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
      f.sprite.scale.setScalar(f.scale * (0.6 + p * 1.6) * this.kAt(f.sprite.position));
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
  /** Blast scale of a hull type's death (Explosion.js units). */
  destructionScale(shipType) {
    const scaleFactors = {
      corvette: 0.6, frigate: 0.8, destroyer: 1.0, cruiser: 1.3, cruiser_torpedo: 1.3,
      battlecruiser: 1.6, battleship: 2.0, dreadnought: 2.5, dreadnought_siege: 2.5
    };
    return (scaleFactors[shipType] || 1.0) * 6;
  }

  /**
   * Reach (km) of a ship's reactor blast: the recorded reactor_blast
   * radius, else (older recordings) the engine's rule for the class.
   */
  blastRadiusKm(shipId, shipType) {
    const b = this.damageFX?.timeline?.blasts?.find(x => x.shipId === shipId);
    return b?.radius > 0 ? b.radius : NOMINAL_BLAST_KM[shipType] || NOMINAL_BLAST_KM.destroyer;
  }

  /**
   * Blast scale (Explosion.js units) of the reactor detonation itself.
   * Cinematic: the class table, as ever. With real-size hulls the blast is
   * not a property of the (tiny) hull but of the engine's blast radius:
   * its lasting fireball fills REACTOR_FIREBALL_SHARE of that radius - the
   * shell that marks the full reach is BlastShell's - whatever the hull's
   * draw scale. The blast is drawn at real size (Blast `real`).
   */
  reactorBlastScale(shipId, shipType) {
    if (this.scaleMode === 'cinematic') return this.destructionScale(shipType);
    const kind = (this.deathInfo || {})[shipId]?.meltdown ? 'meltdown' : 'reactor';
    return REACTOR_FIREBALL_SHARE * this.blastRadiusKm(shipId, shipType) / blastExtentOver(kind, 1, 2, 6);
  }

  /**
   * Largest radius (scene units) this ship's death blast visibly draws
   * between two ages after its detonation - what a camera has to fit.
   */
  deathBlastExtent(shipId, fromAge = 0, toAge = 6) {
    const type = this.ships.get(shipId)?.userData?.shipType;
    const kind = (this.deathInfo || {})[shipId]?.meltdown ? 'meltdown' : 'reactor';
    return blastExtentOver(kind, this.reactorBlastScale(shipId, type), fromAge, toAge);
  }

  createDestructionEffect(position, shipType, currentTime, shipId = null) {
    const scale = this.destructionScale(shipType);
    const death = (this.deathInfo || {})[shipId] || {};
    const shipGroup = shipId ? this.ships.get(shipId) : null;
    // The sim's own death spiral (dying flag) already drifted the hulk; an
    // immediate reactor kill also goes up at once. Otherwise coast dark
    // for a few seconds before the reactor lets go.
    const simDrifted = death.wasDying || shipGroup?.userData.wasDying;
    const spawnTime = death.time ?? currentTime;
    const driftDuration = simDrifted ? 0 : wreckDriftS(death);
    const vel = death.velocity || [0, 0, 0];
    const driftVel = new THREE.Vector3(vel[0], vel[1], vel[2]).multiplyScalar(this.SCALE);
    if (death.position) position = new THREE.Vector3(...death.position).multiplyScalar(this.SCALE);
    if (shipGroup && death.forward) {
      const df = new THREE.Vector3(...death.forward).normalize();
      if (validUp(death.up)) {
        const q = hullQuat(df.toArray(_hfFwd), death.up, _hfQuat);
        shipGroup.quaternion.set(q[0], q[1], q[2], q[3]);
      } else {
        shipGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), df);
      }
    }

    const eff = new ShipDestruction(this.fxCtx, {
      shipGroup, shipId, position, driftVel, driftDuration, scale, spawnTime,
      blastScale: this.reactorBlastScale(shipId, shipType),
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
    // main.js settles the camera (syncCamera) before the frame's effects
    // are placed; a frame it did not drive is settled here
    if (!this._cameraSynced) { this.controls.update(); this.updateScale(); }
    this._cameraSynced = false;
    const viewingDistance = this.camera.position.distanceTo(this.controls.target);
    this.camera.userData.viewDist = viewingDistance; // Environment dust-mote scale
    this.rangeRingMaterial.opacity = 0.12 * THREE.MathUtils.smoothstep(viewingDistance, 45, 250);
    this.tacticalOverlay.visible = this.rangeRingMaterial.opacity > 0.002;
    if (this.frameIndex % 20 === 0) this._flagShadowMeshes();
    const subjectCount = this._collectShadowSubjects();
    this.environment.update(this.camera, this.delta, this.controls.target, viewingDistance,
      this._shadowSubjects, subjectCount);
    this.postfx.update(this.delta);

    // (Blast shake is CameraController.applyShake / removeShake, wrapped
    // round this call by main.js: battle time, deterministic under scrubbing)
    this.postfx.render();
    this.frameIndex++;
  }
}

function _flagShadow(o) {
  if (!o.isMesh || o.userData.shadowFlagged) return;
  o.userData.shadowFlagged = true;
  const m = Array.isArray(o.material) ? o.material[0] : o.material;
  if (!m || !m.isMeshStandardMaterial) return;
  o.receiveShadow = true;
  o.castShadow = !m.transparent && m.blending === THREE.NormalBlending && m.depthWrite !== false;
}

function _flagWreckShadow(o) {
  if (!o.isMesh) return;
  const m = Array.isArray(o.material) ? o.material[0] : o.material;
  if (!m || !m.isMeshStandardMaterial) return;
  o.receiveShadow = true;
  o.castShadow = m.depthWrite !== false && m.opacity > 0.5;
}
