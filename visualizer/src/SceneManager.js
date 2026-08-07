import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

/**
 * SceneManager - Manages the Three.js scene, camera, and renderer
 */
export class SceneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.ships = new Map();
    this.projectiles = new Map();
    this.torpedoes = new Map(); // torpedoId -> THREE.Group
    this.beams = new Map(); // beamKey -> continuous PD beam objects
    this.hitEffects = []; // Active hit effects [{group, spawnTime, duration}]
    this.smallEffects = []; // Fizzles, seeker kills, muzzle flashes
    this.destructionEffects = []; // Active destruction effects
    this.destroyedShips = new Set(); // Ships that have already had destruction animation triggered
    this.dyingPopTimes = new Map(); // shipId -> next hull-fire time during sim-side death spiral

    // Scale: 1 unit = 1 km, positions in recording are meters
    this.SCALE = 1 / 1000;
    // Ships are exaggerated for visibility (actual ships are ~100m, we make them ~5km visible)
    this.SHIP_SCALE = 50;

    this.currentTime = 0; // Current playback time

    // Shared radial-gradient sprite texture for glows/plumes
    this.glowTexture = this.createGlowTexture();

    this.init();
  }

  /**
   * Soft radial gradient texture used by plumes, beam glows and flashes
   */
  createGlowTexture() {
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.6)');
    grad.addColorStop(0.6, 'rgba(255,255,255,0.15)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }

  init() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x04050c);

    // Clock for time-based animation
    this.clock = new THREE.Clock();
    this.delta = 0;
    this.elapsed = 0;

    // Camera - positioned to see battle area (ships at ±150km on X axis)
    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      500000
    );
    // Start high and back to see both fleets
    this.camera.position.set(0, 300, 400);

    // Renderer with logarithmic depth buffer for large scale differences
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      logarithmicDepthBuffer: true
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Filmic tone mapping keeps additive effects (plumes, beams, blasts)
    // from clipping to flat white and gives the scene a cinematic rolloff
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    // Orbit controls
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 50000;

    // Post-processing with bloom for ethereal glow
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.9,  // bloom strength
      0.5,  // radius
      0.25  // threshold (lower = more things bloom)
    );
    this.composer.addPass(bloomPass);
    this.bloomPass = bloomPass;
    this.BLOOM_BASE = 0.9;

    // Lighting - cool space ambience with a warm distant key light
    const hemiLight = new THREE.HemisphereLight(0x2c3e5c, 0x0a0a14, 0.7);
    this.scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff0dd, 1.4);
    keyLight.position.set(4000, 2500, 1500);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x334f6e, 0.5);
    fillLight.position.set(-3000, -1000, -2000);
    this.scene.add(fillLight);

    // Dynamic point-light budget. Every PointLight's uniforms are compiled
    // into EVERY lit material's shader, and llvmpipe / low-end GPUs cap
    // fragment uniforms (MAX_FRAGMENT_UNIFORM_VECTORS = 1024): a 6v6 fleet
    // battle with per-bell plume lights and per-torpedo lights exceeded the
    // cap and broke all rendering. Plumes, torpedoes and effects request
    // lights via tryPointLight() and must tolerate null (their additive
    // sprites still carry the look without a light).
    this.dynamicLightCount = 0;
    this.dynamicLightBudget = 10;

    // Single subtle tactical grid + range rings (the old triple grid boxed
    // the battle in visual noise)
    const gridXZ = new THREE.GridHelper(4000, 40, 0x1c3a52, 0x0c1826);
    gridXZ.material.transparent = true;
    gridXZ.material.opacity = 0.5;
    this.scene.add(gridXZ);
    this.createRangeRings();

    // Starfield + nebula + distant sun
    this.createStarfield();
    this.createNebula();
    this.createSun(keyLight.position.clone().normalize());

    // Handle resize
    window.addEventListener('resize', () => this.onResize());
  }

  /**
   * Faint concentric range rings every 100 km around the origin
   */
  createRangeRings() {
    const ringMaterial = new THREE.LineBasicMaterial({
      color: 0x1e4a66,
      transparent: true,
      opacity: 0.35
    });
    for (let r = 100; r <= 600; r += 100) {
      const points = [];
      const segments = 128;
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
      }
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      this.scene.add(new THREE.Line(geom, ringMaterial));
    }
  }

  createStarfield() {
    // Three depth layers with size and color variation
    const layers = [
      { count: 2600, size: 1.2, spread: [16000, 30000], saturation: 0.15 },
      { count: 900, size: 2.2, spread: [14000, 26000], saturation: 0.35 },
      { count: 140, size: 3.5, spread: [12000, 22000], saturation: 0.55 }
    ];

    for (const layer of layers) {
      const geom = new THREE.BufferGeometry();
      const positions = new Float32Array(layer.count * 3);
      const colors = new Float32Array(layer.count * 3);
      const color = new THREE.Color();

      for (let i = 0; i < layer.count; i++) {
        const radius = layer.spread[0] + Math.random() * (layer.spread[1] - layer.spread[0]);
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);

        positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = radius * Math.cos(phi);

        // Star temperature: mostly white, a few blue and warm outliers
        const t = Math.random();
        if (t < 0.12) color.setHSL(0.06, layer.saturation, 0.85);      // warm
        else if (t < 0.3) color.setHSL(0.6, layer.saturation, 0.85);   // blue
        else color.setHSL(0.62, layer.saturation * 0.3, 0.9);          // white
        const dim = 0.5 + Math.random() * 0.5;
        colors[i * 3] = color.r * dim;
        colors[i * 3 + 1] = color.g * dim;
        colors[i * 3 + 2] = color.b * dim;
      }

      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const mat = new THREE.PointsMaterial({
        size: layer.size,
        vertexColors: true,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.9,
        depthWrite: false
      });
      this.scene.add(new THREE.Points(geom, mat));
    }
  }

  /**
   * A few very large, very faint additive glow sprites for depth
   */
  createNebula() {
    const patches = [
      { color: 0x1a2e5e, pos: [-9000, 3000, -14000], scale: 16000, opacity: 0.10 },
      { color: 0x36245e, pos: [12000, -2000, -10000], scale: 14000, opacity: 0.08 },
      { color: 0x123a4a, pos: [3000, 6000, 13000], scale: 18000, opacity: 0.09 },
      { color: 0x40203a, pos: [-13000, -5000, 8000], scale: 12000, opacity: 0.06 }
    ];
    for (const p of patches) {
      const mat = new THREE.SpriteMaterial({
        map: this.glowTexture,
        color: p.color,
        transparent: true,
        opacity: p.opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(p.pos[0], p.pos[1], p.pos[2]);
      sprite.scale.setScalar(p.scale);
      this.scene.add(sprite);
    }
  }

  /** Kill all torch visuals on a dead/dying hull. */
  _extinguishTorches(ship) {
    for (const torch of ship.userData.torches || []) {
      torch.core.visible = false;
      torch.sheath.visible = false;
      torch.nozzleGlow.material.opacity = 0;
      for (const bead of torch.beads) bead.material.opacity = 0;
      if (torch.light) torch.light.intensity = 0;
    }
  }

  /**
   * Request a PointLight against the dynamic-light budget.
   * Returns null when the budget is spent - callers must skip the light.
   * `reserve` lets high-priority effects (ship destruction) borrow past
   * the base budget.
   */
  tryPointLight(color, intensity, distance, reserve = 0) {
    if (this.dynamicLightCount >= this.dynamicLightBudget + reserve) return null;
    this.dynamicLightCount++;
    return new THREE.PointLight(color, intensity, distance);
  }

  /** Return a budgeted light (null-safe) and dispose it. */
  freePointLight(light) {
    if (!light) return;
    this.dynamicLightCount--;
    light.dispose();
  }

  /**
   * Distant sun sprite along the key light direction
   */
  createSun(direction) {
    const core = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: 0xfff5e0,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    core.position.copy(direction.clone().multiplyScalar(28000));
    core.scale.setScalar(2600);
    this.scene.add(core);

    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: 0xffd9a0,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    halo.position.copy(core.position);
    halo.scale.setScalar(9000);
    this.scene.add(halo);
  }

  /**
   * Create a ship mesh - Expanse-style design
   * Capital ships (battleship+): Donnager-style tall angular tower
   * Smaller ships: Tachi/Rocinante-style sleek corvette
   * @param {string} shipId - Ship identifier
   * @param {string} faction - 'alpha' or 'beta'
   * @param {string} shipType - Ship class
   */
  createShip(shipId, faction, shipType = 'destroyer') {
    const group = new THREE.Group();

    // Ship size based on type
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
    const baseSize = baseSizes[shipType] || baseSizes.destroyer;

    // Scale ships to be visible at km distances
    const scale = 8;
    const size = {
      length: baseSize.length * scale,
      width: baseSize.width * scale
    };

    // Colors
    const primaryColor = faction === 'alpha' ? 0x2a3a4a : 0x3a2a2a;
    const accentColor = faction === 'alpha' ? 0x00d4ff : 0xff6644;
    const emissiveColor = faction === 'alpha' ? 0x002233 : 0x331111;

    // Materials: faction hull + accent, shared dark trim, lit windows
    const mats = {
      hull: new THREE.MeshStandardMaterial({
        color: faction === 'alpha' ? 0x32465c : 0x523634,
        metalness: 0.75,
        roughness: 0.45,
        emissive: emissiveColor,
        emissiveIntensity: 0.35
      }),
      accent: new THREE.MeshStandardMaterial({
        color: accentColor,
        metalness: 0.6,
        roughness: 0.3,
        emissive: accentColor,
        emissiveIntensity: 0.5
      }),
      trim: new THREE.MeshStandardMaterial({
        color: 0x161c24,
        metalness: 0.7,
        roughness: 0.55,
        emissive: 0x05070a,
        emissiveIntensity: 0.3
      }),
      window: new THREE.MeshBasicMaterial({ color: 0xffe9c4 })
    };

    // Engine layout is decided first so the hull builder can shape the
    // drive section around the actual bell arrangement.
    const engineConfig = this.getEngineConfig(shipType, size);
    this.buildHull(shipType, group, size, mats, engineConfig);

    // Fusion torch per engine bell: a needle of plasma - bright white core
    // cone inside a wider blue sheath cone - plus a nozzle glow and a few
    // fading glow beads along the axis for volume. All transform/opacity
    // animation, no per-particle CPU work (the old 800-particle system per
    // bell burned milliseconds every frame and read as smoke, not a torch).
    const torches = [];
    const plumeLength = size.length * engineConfig.plumeLength * 0.45;

    // Unit cone: base at origin, apex stretching down -Z, scaled per frame.
    // Vertex colors fade to black toward the apex - under additive blending
    // black is transparent, so the cone reads as a glowing plasma spike
    // instead of solid geometry when seen up close.
    const coneGeom = (radius) => {
      const g = new THREE.ConeGeometry(radius, 1, 12, 6, true);
      g.translate(0, -0.5, 0);   // base at origin, apex at -Y
      const pos = g.attributes.position;
      const colors = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        // y runs 0 (base) .. -1 (apex)
        const brightness = Math.pow(1 + pos.getY(i), 1.6);
        colors[i * 3] = brightness;
        colors[i * 3 + 1] = brightness;
        colors[i * 3 + 2] = brightness;
      }
      g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      g.rotateX(Math.PI / 2);    // apex now at -Z
      return g;
    };

    // Plume base radius follows the physical bell size for the class -
    // many small bells on a capital, two fat ones on a corvette.
    const plumeR = size.width * (engineConfig.count <= 2 ? 0.2
      : engineConfig.count <= 4 ? 0.155 : 0.11);

    for (let e = 0; e < engineConfig.count; e++) {
      const offset = engineConfig.positions[e];
      const bellZ = -size.length * 0.52;
      const bellR = plumeR;

      // One budgeted light per SHIP (first bell), not per bell - a 4-bell
      // cruiser in a 12-ship battle would otherwise exhaust the budget alone.
      const light = e === 0 ? this.tryPointLight(0x66aaff, 0, size.length * 4) : null;
      if (light) {
        light.position.set(offset.x, offset.y, bellZ);
        group.add(light);
      }

      const core = new THREE.Mesh(coneGeom(bellR * 0.55), new THREE.MeshBasicMaterial({
        color: 0xf2f8ff,
        vertexColors: true,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide
      }));
      core.position.set(offset.x, offset.y, bellZ);
      group.add(core);

      const sheath = new THREE.Mesh(coneGeom(bellR * 1.35), new THREE.MeshBasicMaterial({
        color: 0x4a7bff,
        vertexColors: true,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide
      }));
      sheath.position.set(offset.x, offset.y, bellZ);
      group.add(sheath);

      const nozzleGlow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: this.glowTexture,
        color: 0xcfe4ff,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      }));
      nozzleGlow.position.set(offset.x, offset.y, bellZ);
      nozzleGlow.scale.setScalar(size.width * 0.9);
      group.add(nozzleGlow);

      const beads = [];
      for (let b = 0; b < 4; b++) {
        const bead = new THREE.Sprite(new THREE.SpriteMaterial({
          map: this.glowTexture,
          color: b < 2 ? 0x9cc4ff : 0x5a7fdd,
          transparent: true,
          opacity: 0,
          blending: THREE.AdditiveBlending,
          depthWrite: false
        }));
        bead.position.set(offset.x, offset.y, bellZ);
        group.add(bead);
        beads.push(bead);
      }

      torches.push({
        core, sheath, nozzleGlow, beads, light,
        offset, bellZ,
        phase: Math.random() * Math.PI * 2
      });
    }

    // Store references and plume config (merge - hull builders already
    // stashed radiator references in userData)
    Object.assign(group.userData, {
      shipId,
      faction,
      shipType,
      torches,
      size,
      engineConfig,
      plumeLength
    });

    this.scene.add(group);
    this.ships.set(shipId, group);

    return group;
  }

  /**
   * Get engine configuration based on ship type
   * Returns number of engine bells and their positions
   */
  getEngineConfig(shipType, size) {
    const W = size.width;

    // Bell counts scale with hull mass: a corvette flies on a light twin
    // torch, a dreadnought rides a ring of eight. Positions must sit inside
    // each class's drive-section footprint (the hull builder draws the
    // physical bells at these same coordinates).
    const ring = (n, r, offset = 0) => Array.from({ length: n }, (_, i) => {
      const a = offset + (i / n) * Math.PI * 2;
      return { x: Math.cos(a) * r, y: Math.sin(a) * r };
    });

    const configs = {
      corvette: {
        count: 2,
        positions: [{ x: -W * 0.16, y: 0 }, { x: W * 0.16, y: 0 }],
        plumeLength: 6
      },
      frigate: {
        count: 2,
        positions: [{ x: -W * 0.17, y: 0 }, { x: W * 0.17, y: 0 }],
        plumeLength: 7
      },
      destroyer: {
        count: 3,
        positions: [
          { x: -W * 0.24, y: 0 }, { x: 0, y: 0 }, { x: W * 0.24, y: 0 }
        ],
        plumeLength: 8
      },
      cruiser: {
        count: 4,
        positions: [
          { x: -W * 0.18, y: W * 0.18 }, { x: W * 0.18, y: W * 0.18 },
          { x: -W * 0.18, y: -W * 0.18 }, { x: W * 0.18, y: -W * 0.18 }
        ],
        plumeLength: 8
      },
      cruiser_torpedo: {
        count: 4,
        positions: [
          { x: -W * 0.18, y: W * 0.18 }, { x: W * 0.18, y: W * 0.18 },
          { x: -W * 0.18, y: -W * 0.18 }, { x: W * 0.18, y: -W * 0.18 }
        ],
        plumeLength: 8
      },
      battlecruiser: {
        count: 4,
        positions: [
          { x: -W * 0.3, y: 0 }, { x: -W * 0.1, y: 0 },
          { x: W * 0.1, y: 0 }, { x: W * 0.3, y: 0 }
        ],
        plumeLength: 10
      },
      battleship: {
        count: 6,
        positions: [
          { x: -W * 0.28, y: W * 0.13 }, { x: 0, y: W * 0.13 }, { x: W * 0.28, y: W * 0.13 },
          { x: -W * 0.28, y: -W * 0.13 }, { x: 0, y: -W * 0.13 }, { x: W * 0.28, y: -W * 0.13 }
        ],
        plumeLength: 10
      },
      dreadnought: {
        count: 8,
        positions: ring(8, W * 0.3, Math.PI / 8),
        plumeLength: 12
      },
      dreadnought_siege: {
        count: 8,
        positions: ring(8, W * 0.3, Math.PI / 8),
        plumeLength: 12
      }
    };

    return configs[shipType] || configs.destroyer;
  }

  // ===========================================================================
  // EXPANSE-STYLE HULLS
  //
  // These ships are buildings that fly on their torch: stacked deck sections,
  // open truss segments, drum hulls, slab armor, and a silhouette per class
  // you can read at a glance. Local frame: nose +Z, drive -Z, dorsal +Y.
  // Every builder sets userData.radiators / radiatorMaterial / reactorPos.
  // ===========================================================================

  buildHull(shipType, group, size, mats, engineConfig) {
    const builders = {
      corvette: this.hullCorvette,
      frigate: this.hullFrigate,
      destroyer: this.hullDestroyer,
      cruiser: this.hullCruiser,
      cruiser_torpedo: this.hullTorpedoCruiser,
      battlecruiser: this.hullBattlecruiser,
      battleship: this.hullBattleship,
      dreadnought: this.hullDreadnought,
      dreadnought_siege: this.hullDreadnought
    };
    const builder = builders[shipType] || this.hullDestroyer;
    builder.call(this, group, size, mats, { siege: shipType === 'dreadnought_siege' });
    this._addDriveSection(group, size, mats, engineConfig);
  }

  // -- shared construction vocabulary ----------------------------------------

  _box(group, mat, w, h, d, x, y, z, rot = null) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    if (rot) m.rotation.set(rot[0], rot[1], rot[2]);
    group.add(m);
    return m;
  }

  /** Octagonal hull drum with its axis along Z. */
  _drum(group, mat, r, len, z, sides = 8) {
    const g = new THREE.CylinderGeometry(r, r, len, sides);
    g.rotateZ(Math.PI / sides);
    g.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(g, mat);
    m.position.set(0, 0, z);
    group.add(m);
    return m;
  }

  /** Four-sided wedge prow, tip pointing +Z. w/h shape the cross-section. */
  _prow(group, mat, w, h, len, z) {
    const g = new THREE.ConeGeometry(0.5, len, 4);
    g.rotateY(Math.PI / 4);
    g.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(g, mat);
    m.scale.set(w * 1.4, h * 1.4, 1);
    m.position.set(0, 0, z);
    group.add(m);
    return m;
  }

  /** Open truss segment: four corner longerons + two cross frames. */
  _truss(group, mat, halfW, len, z) {
    const t = halfW * 0.16;
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        this._box(group, mat, t, t, len, sx * halfW, sy * halfW, z);
      }
    }
    for (const fz of [z - len * 0.3, z + len * 0.3]) {
      this._box(group, mat, halfW * 2 + t, t, t, 0, halfW, fz);
      this._box(group, mat, halfW * 2 + t, t, t, 0, -halfW, fz);
      this._box(group, mat, t, halfW * 2 + t, t, halfW, 0, fz);
      this._box(group, mat, t, halfW * 2 + t, t, -halfW, 0, fz);
    }
  }

  /** Twin-barrel gun turret. sign=+1 dorsal, -1 ventral, or a rotation. */
  _turret(group, mats, s, x, y, z, angle = 0) {
    const turret = new THREE.Group();
    const base = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.55, s * 0.65, s * 0.25, 8), mats.trim);
    turret.add(base);
    const housing = new THREE.Mesh(new THREE.BoxGeometry(s * 0.8, s * 0.4, s * 0.9), mats.hull);
    housing.position.y = s * 0.3;
    turret.add(housing);
    for (const bx of [-0.18, 0.18]) {
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.05, s * 0.05, s * 1.1, 6), mats.trim);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(bx * s, s * 0.32, s * 0.8);
      turret.add(barrel);
    }
    turret.position.set(x, y, z);
    turret.rotation.z = angle;
    group.add(turret);
    return turret;
  }

  /** Small PD blister dome with a stub barrel. */
  _pdBlister(group, mats, s, x, y, z, angle = 0) {
    const pd = new THREE.Group();
    const dome = new THREE.Mesh(new THREE.SphereGeometry(s * 0.32, 8, 5, 0, Math.PI * 2, 0, Math.PI / 2), mats.trim);
    pd.add(dome);
    const stub = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.05, s * 0.05, s * 0.45, 5), mats.accent);
    stub.rotation.x = Math.PI / 3;
    stub.position.set(0, s * 0.22, s * 0.16);
    pd.add(stub);
    pd.position.set(x, y, z);
    pd.rotation.z = angle;
    group.add(pd);
    return pd;
  }

  /** Comms/sensor mast with a lit tip. */
  _mast(group, mats, h, x, y, z, lean = 0) {
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.03, h * 0.045, h, 4), mats.trim);
    mast.position.set(x, y + h / 2, z);
    mast.rotation.z = lean;
    group.add(mast);
    const tip = new THREE.Mesh(new THREE.SphereGeometry(h * 0.07, 6, 4), mats.accent);
    tip.position.set(x - Math.sin(lean) * h, y + h * Math.cos(lean), z);
    group.add(tip);
    return mast;
  }

  /**
   * Stepped tower: stacked deck blocks shrinking as they rise - the
   * "building on a torch" silhouette. Returns the top deck for masts.
   */
  _tower(group, mats, baseW, baseD, deckH, decks, x, yBase, z) {
    let w = baseW, d = baseD, y = yBase;
    let top = null;
    for (let i = 0; i < decks; i++) {
      top = this._box(group, mats.hull, w, deckH, d, x, y + deckH / 2, z + (i % 2 ? d * 0.06 : 0));
      y += deckH;
      w *= 0.72;
      d *= 0.78;
    }
    return { top, yTop: y };
  }

  /** Row of lit windows along Z on a face. */
  _windowRow(group, mats, n, x, y, z, spacing, s) {
    for (let i = 0; i < n; i++) {
      this._box(group, mats.window, s, s, s * 0.4, x, y, z + i * spacing);
    }
  }

  /**
   * Radiator panel whose local +Y points outward, geometry anchored at its
   * inner edge - scale.y folds it flat against the hull when retracted
   * (updateShip animates this with the radiator state).
   */
  _radiatorPanel(group, mat, thickness, span, len, x, y, z, angle) {
    const g = new THREE.BoxGeometry(thickness, span, len);
    g.translate(0, span / 2, 0);
    const m = new THREE.Mesh(g, mat);
    m.position.set(x, y, z);
    m.rotation.z = angle;   // 0 = +Y (dorsal), PI/2 = -X, -PI/2 = +X, PI = ventral
    group.add(m);
    return m;
  }

  _makeRadiatorMaterial() {
    return new THREE.MeshStandardMaterial({
      color: 0x3a2d28,
      metalness: 0.4,
      roughness: 0.6,
      emissive: 0xff5a22,
      emissiveIntensity: 0.0,
      side: THREE.DoubleSide
    });
  }

  /** Drive skirt + physical engine bells matching the plume positions. */
  _addDriveSection(group, size, mats, engineConfig) {
    const L = size.length;
    const W = size.width;

    const skirt = new THREE.Mesh(
      new THREE.CylinderGeometry(W * 0.4, W * 0.5, L * 0.1, 8),
      mats.hull
    );
    skirt.geometry.rotateZ(Math.PI / 8);
    skirt.rotation.x = Math.PI / 2;
    skirt.position.set(0, 0, -L * 0.44);
    group.add(skirt);

    const bellR = W * (engineConfig.count <= 2 ? 0.15 : engineConfig.count <= 4 ? 0.12 : 0.085);
    for (const p of engineConfig.positions) {
      const bell = new THREE.Mesh(
        new THREE.CylinderGeometry(bellR * 0.55, bellR, L * 0.06, 8, 1, true),
        mats.trim
      );
      bell.rotation.x = Math.PI / 2;
      bell.position.set(p.x, p.y, -L * 0.5);
      group.add(bell);
      const throat = new THREE.Mesh(
        new THREE.CylinderGeometry(bellR * 0.5, bellR * 0.55, L * 0.02, 8),
        mats.accent
      );
      throat.rotation.x = Math.PI / 2;
      throat.position.set(p.x, p.y, -L * 0.47);
      group.add(throat);
    }
  }

  // -- class hulls -----------------------------------------------------------

  /** Corvette: compact stacked gunboat (Rocinante energy). Twin torch. */
  hullCorvette(group, size, mats) {
    const L = size.length, W = size.width;

    this._box(group, mats.hull, W * 0.56, W * 0.62, L * 0.55, 0, 0, -L * 0.05);
    this._box(group, mats.hull, W * 0.46, W * 0.16, L * 0.42, 0, W * 0.38, -L * 0.02);
    this._box(group, mats.hull, W * 0.34, W * 0.12, L * 0.3, 0, W * 0.5, 0);
    this._prow(group, mats.hull, W * 0.28, W * 0.31, L * 0.3, L * 0.36);

    // Ventral torpedo tube doors + dorsal PD
    this._box(group, mats.accent, W * 0.24, W * 0.03, L * 0.3, 0, -W * 0.33, L * 0.08);
    this._pdBlister(group, mats, W * 0.28, 0, W * 0.56, -L * 0.14);

    this._windowRow(group, mats, 5, W * 0.29, W * 0.1, -L * 0.14, L * 0.06, W * 0.035);
    this._windowRow(group, mats, 5, -W * 0.29, W * 0.1, -L * 0.14, L * 0.06, W * 0.035);
    this._mast(group, mats, W * 0.4, 0, W * 0.56, -L * 0.22, 0.15);

    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.42, L * 0.2, -W * 0.28, 0, -L * 0.24, -Math.PI / 2),
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.42, L * 0.2, W * 0.28, 0, -L * 0.24, Math.PI / 2)
    ];
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.28);
  }

  /** Frigate: sensor head on a truss neck, drum body, tall dorsal fin. */
  hullFrigate(group, size, mats) {
    const L = size.length, W = size.width;

    this._box(group, mats.hull, W * 0.44, W * 0.44, L * 0.2, 0, 0, L * 0.36);
    this._prow(group, mats.trim, W * 0.22, W * 0.22, L * 0.14, L * 0.5);
    this._truss(group, mats.trim, W * 0.16, L * 0.14, L * 0.2);
    this._drum(group, mats.hull, W * 0.34, L * 0.52, -L * 0.08);

    // Tall dorsal fin with lit leading edge - the frigate's signature
    this._box(group, mats.hull, W * 0.05, W * 0.52, L * 0.24, 0, W * 0.5, -L * 0.06);
    this._box(group, mats.accent, W * 0.02, W * 0.5, W * 0.04, 0, W * 0.5, L * 0.05);

    this._turret(group, mats, W * 0.2, 0, W * 0.42, L * 0.32);
    this._pdBlister(group, mats, W * 0.24, 0, -W * 0.36, 0, Math.PI);
    this._windowRow(group, mats, 4, 0, W * 0.24, L * 0.28, L * 0.045, W * 0.035);
    this._mast(group, mats, W * 0.36, W * 0.12, W * 0.42, L * 0.42, -0.2);

    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.5, L * 0.22, -W * 0.3, W * 0.12, -L * 0.26, -Math.PI / 2 + 0.35),
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.5, L * 0.22, W * 0.3, W * 0.12, -L * 0.26, Math.PI / 2 - 0.35)
    ];
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.28);
  }

  /** Destroyer: armored prow, ventral spinal rail, decks stepping aft. */
  hullDestroyer(group, size, mats) {
    const L = size.length, W = size.width;

    this._prow(group, mats.hull, W * 0.36, W * 0.31, L * 0.26, L * 0.42);
    this._box(group, mats.hull, W * 0.6, W * 0.54, L * 0.36, 0, 0, L * 0.12);
    this._box(group, mats.hull, W * 0.7, W * 0.62, L * 0.4, 0, 0, -L * 0.2);

    // Spinal coilgun: glowing rail down the ventral centerline to the muzzle
    this._box(group, mats.accent, W * 0.09, W * 0.06, L * 0.8, 0, -W * 0.3, L * 0.05);
    this._box(group, mats.trim, W * 0.16, W * 0.1, L * 0.06, 0, -W * 0.3, L * 0.46);

    // Amidships tower + mast
    const t = this._tower(group, mats, W * 0.34, L * 0.16, W * 0.13, 3, 0, W * 0.27, -L * 0.06);
    this._mast(group, mats, W * 0.4, 0, t.yTop, -L * 0.06, 0.1);

    this._turret(group, mats, W * 0.24, 0, W * 0.29, L * 0.2);
    this._turret(group, mats, W * 0.24, 0, -W * 0.33, -L * 0.05, Math.PI);
    this._pdBlister(group, mats, W * 0.24, W * 0.33, W * 0.12, L * 0.05, -Math.PI / 2);
    this._pdBlister(group, mats, W * 0.24, -W * 0.33, W * 0.12, L * 0.05, Math.PI / 2);

    this._windowRow(group, mats, 6, W * 0.31, W * 0.14, -L * 0.3, L * 0.05, W * 0.03);
    this._windowRow(group, mats, 6, -W * 0.31, W * 0.14, -L * 0.3, L * 0.05, W * 0.03);

    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.55, L * 0.24, -W * 0.36, 0, -L * 0.28, -Math.PI / 2),
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.55, L * 0.24, W * 0.36, 0, -L * 0.28, Math.PI / 2)
    ];
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.3);
  }

  /** Cruiser: armored collar + heavy drum body + visible reactor drum. */
  hullCruiser(group, size, mats, opts = {}) {
    const L = size.length, W = size.width;

    this._prow(group, mats.hull, W * 0.3, W * 0.3, L * 0.2, L * 0.44);
    this._drum(group, mats.trim, W * 0.5, L * 0.16, L * 0.28);
    this._drum(group, mats.hull, W * 0.44, L * 0.5, -L * 0.02);

    // Reactor drum with warning ring - the ship's heart, visibly aft
    this._drum(group, mats.hull, W * 0.36, L * 0.14, -L * 0.32);
    this._drum(group, mats.accent, W * 0.375, L * 0.02, -L * 0.32);

    if (!opts.noSpinal) {
      this._box(group, mats.accent, W * 0.08, W * 0.06, L * 0.7, 0, -W * 0.42, L * 0.05);
      this._box(group, mats.trim, W * 0.15, W * 0.1, L * 0.06, 0, -W * 0.42, L * 0.4);
    }

    const t = this._tower(group, mats, W * 0.36, L * 0.14, W * 0.11, 3, 0, W * 0.42, L * 0.06);
    this._mast(group, mats, W * 0.42, 0, t.yTop, L * 0.06, -0.12);

    this._turret(group, mats, W * 0.26, W * 0.4, 0, L * 0.14, -Math.PI / 2);
    this._turret(group, mats, W * 0.26, -W * 0.4, 0, L * 0.14, Math.PI / 2);

    this._windowRow(group, mats, 7, W * 0.4, W * 0.16, -L * 0.2, L * 0.045, W * 0.028);
    this._windowRow(group, mats, 7, -W * 0.4, W * 0.16, -L * 0.2, L * 0.045, W * 0.028);

    // Four radiator wings in an X around the aft drum
    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [0.785, 2.356, -2.356, -0.785].map(a =>
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.6, L * 0.22,
        Math.sin(a) * -W * 0.34, Math.cos(a) * W * 0.34, -L * 0.16, a));
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.32);
  }

  /** Torpedo cruiser: cruiser drum with four VLS banks instead of guns. */
  hullTorpedoCruiser(group, size, mats) {
    const L = size.length, W = size.width;
    this.hullCruiser(group, size, mats, { noSpinal: true });

    // VLS banks: raised plates with hatch grids on four faces
    const hatch = (bank, angle) => {
      const plate = new THREE.Group();
      const base = new THREE.Mesh(new THREE.BoxGeometry(W * 0.34, W * 0.06, L * 0.28), mats.trim);
      plate.add(base);
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 3; c++) {
          const door = new THREE.Mesh(new THREE.BoxGeometry(W * 0.11, W * 0.02, L * 0.062), mats.accent);
          door.position.set((r - 0.5) * W * 0.15, W * 0.035, (c - 1) * L * 0.085);
          plate.add(door);
        }
      }
      plate.position.set(Math.sin(angle) * -W * 0.45, Math.cos(angle) * W * 0.45, bank);
      plate.rotation.z = angle;
      group.add(plate);
    };
    hatch(L * 0.12, 0);
    hatch(L * 0.12, Math.PI);
    hatch(-L * 0.06, Math.PI / 2);
    hatch(-L * 0.06, -Math.PI / 2);

    // Extra PD ring - this hull's only defense once the tubes run dry
    this._pdBlister(group, mats, W * 0.22, W * 0.34, W * 0.34, L * 0.3, -Math.PI / 4);
    this._pdBlister(group, mats, W * 0.22, -W * 0.34, W * 0.34, L * 0.3, Math.PI / 4);
    this._pdBlister(group, mats, W * 0.22, W * 0.34, -W * 0.34, L * 0.3, -Math.PI * 0.75);
    this._pdBlister(group, mats, W * 0.22, -W * 0.34, -W * 0.34, L * 0.3, Math.PI * 0.75);
  }

  /** Battlecruiser: stretched racer - split hull with an open waist truss. */
  hullBattlecruiser(group, size, mats) {
    const L = size.length, W = size.width;

    this._prow(group, mats.hull, W * 0.26, W * 0.24, L * 0.16, L * 0.46);
    this._box(group, mats.hull, W * 0.5, W * 0.44, L * 0.42, 0, 0, L * 0.19);
    this._truss(group, mats.trim, W * 0.17, L * 0.16, -L * 0.08);
    this._box(group, mats.hull, W * 0.62, W * 0.54, L * 0.28, 0, 0, -L * 0.3);

    // Forward bridge fin - tall, lean, unmistakable
    const t = this._tower(group, mats, W * 0.24, L * 0.12, W * 0.15, 4, 0, W * 0.22, L * 0.26);
    this._mast(group, mats, W * 0.5, 0, t.yTop, L * 0.26, 0);

    // Full-length dorsal spine rail + ventral spinal
    this._box(group, mats.trim, W * 0.07, W * 0.07, L * 0.7, 0, W * 0.26, -L * 0.02);
    this._box(group, mats.accent, W * 0.07, W * 0.05, L * 0.75, 0, -W * 0.25, L * 0.05);
    this._box(group, mats.trim, W * 0.13, W * 0.09, L * 0.05, 0, -W * 0.25, L * 0.44);

    this._turret(group, mats, W * 0.22, 0, W * 0.24, -L * 0.24);
    this._pdBlister(group, mats, W * 0.22, W * 0.28, 0, L * 0.1, -Math.PI / 2);
    this._pdBlister(group, mats, W * 0.22, -W * 0.28, 0, L * 0.1, Math.PI / 2);

    this._windowRow(group, mats, 8, W * 0.26, W * 0.1, L * 0.02, L * 0.045, W * 0.028);
    this._windowRow(group, mats, 8, -W * 0.26, W * 0.1, L * 0.02, L * 0.045, W * 0.028);

    // Radiators live on the exposed waist - dorsal and ventral wings
    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.66, L * 0.14, 0, W * 0.18, -L * 0.08, 0),
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.66, L * 0.14, 0, -W * 0.18, -L * 0.08, Math.PI)
    ];
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.32);
  }

  /** Battleship: broad slab of layered armor with three heavy turrets. */
  hullBattleship(group, size, mats) {
    const L = size.length, W = size.width;

    this._box(group, mats.hull, W * 0.95, W * 0.48, L * 0.58, 0, 0, -L * 0.03);
    this._prow(group, mats.hull, W * 0.5, W * 0.26, L * 0.24, L * 0.38);

    // Layered applique armor slabs, offset like plate stacks
    this._box(group, mats.trim, W * 0.8, W * 0.07, L * 0.46, 0, W * 0.27, -L * 0.01);
    this._box(group, mats.hull, W * 0.62, W * 0.07, L * 0.34, 0, W * 0.33, L * 0.03);
    this._box(group, mats.trim, W * 0.8, W * 0.07, L * 0.46, 0, -W * 0.27, -L * 0.05);

    // Heavy spinal rail
    this._box(group, mats.accent, W * 0.07, W * 0.05, L * 0.66, 0, -W * 0.33, L * 0.08);
    this._box(group, mats.trim, W * 0.13, W * 0.1, L * 0.05, 0, -W * 0.33, L * 0.43);

    // Citadel: massive stepped conning tower amidships
    const t = this._tower(group, mats, W * 0.4, L * 0.18, W * 0.12, 4, 0, W * 0.3, -L * 0.1);
    this._mast(group, mats, W * 0.45, 0, t.yTop, -L * 0.1, 0.08);
    this._mast(group, mats, W * 0.3, W * 0.2, W * 0.3, -L * 0.22, 0.5);

    this._turret(group, mats, W * 0.3, 0, W * 0.34, L * 0.18);
    this._turret(group, mats, W * 0.3, W * 0.36, W * 0.3, -L * 0.24, -0.5);
    this._turret(group, mats, W * 0.3, 0, -W * 0.33, L * 0.1, Math.PI);
    this._pdBlister(group, mats, W * 0.24, W * 0.5, 0, L * 0.15, -Math.PI / 2);
    this._pdBlister(group, mats, W * 0.24, -W * 0.5, 0, L * 0.15, Math.PI / 2);
    this._pdBlister(group, mats, W * 0.24, 0, W * 0.34, -L * 0.32);

    this._windowRow(group, mats, 8, W * 0.49, W * 0.1, -L * 0.24, L * 0.05, W * 0.025);
    this._windowRow(group, mats, 8, -W * 0.49, W * 0.1, -L * 0.24, L * 0.05, W * 0.025);

    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [
      this._radiatorPanel(group, radMat, W * 0.015, W * 0.5, L * 0.2, -W * 0.5, W * 0.1, -L * 0.3, -Math.PI / 2),
      this._radiatorPanel(group, radMat, W * 0.015, W * 0.5, L * 0.2, W * 0.5, W * 0.1, -L * 0.3, Math.PI / 2),
      this._radiatorPanel(group, radMat, W * 0.015, W * 0.5, L * 0.2, -W * 0.5, -W * 0.1, -L * 0.3, -Math.PI / 2),
      this._radiatorPanel(group, radMat, W * 0.015, W * 0.5, L * 0.2, W * 0.5, -W * 0.1, -L * 0.3, Math.PI / 2)
    ];
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.28);
  }

  /** Dreadnought: Donnager-class quad-lobe fortress on a drive skirt. */
  hullDreadnought(group, size, mats, opts = {}) {
    const L = size.length, W = size.width;

    // Central keel + four sponson pods at the diagonals
    this._drum(group, mats.hull, W * 0.3, L * 0.78, -L * 0.03);
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        this._box(group, mats.hull, W * 0.32, W * 0.32, L * 0.52,
          sx * W * 0.36, sy * W * 0.36, -L * 0.08);
        this._box(group, mats.trim, W * 0.34, W * 0.1, L * 0.3,
          sx * W * 0.36, sy * W * 0.36 + (sy > 0 ? W * 0.16 : -W * 0.16), -L * 0.04);
      }
    }

    // Bow: stacked forward citadel narrowing to the prow
    this._box(group, mats.hull, W * 0.5, W * 0.5, L * 0.16, 0, 0, L * 0.42);
    this._box(group, mats.hull, W * 0.64, W * 0.64, L * 0.14, 0, 0, L * 0.3);
    this._prow(group, mats.trim, W * 0.25, W * 0.25, L * 0.12, L * 0.54);
    this._box(group, mats.accent, W * 0.36, W * 0.05, W * 0.05, 0, 0, L * 0.5);

    // Three dorsal towers along the keel line
    for (const [tz, decks] of [[L * 0.18, 3], [-L * 0.02, 4], [-L * 0.22, 3]]) {
      const t = this._tower(group, mats, W * 0.2, L * 0.09, W * 0.09, decks, 0, W * 0.36, tz);
      if (decks === 4) this._mast(group, mats, W * 0.4, 0, t.yTop, tz, 0);
    }

    // Sponson-face turrets
    this._turret(group, mats, W * 0.26, W * 0.36, W * 0.56, L * 0.12);
    this._turret(group, mats, W * 0.26, -W * 0.36, W * 0.56, L * 0.12);
    this._turret(group, mats, W * 0.26, W * 0.36, -W * 0.56, L * 0.12, Math.PI);
    this._turret(group, mats, W * 0.26, -W * 0.36, -W * 0.56, L * 0.12, Math.PI);
    this._pdBlister(group, mats, W * 0.2, W * 0.6, 0, -L * 0.05, -Math.PI / 2);
    this._pdBlister(group, mats, W * 0.2, -W * 0.6, 0, -L * 0.05, Math.PI / 2);
    this._pdBlister(group, mats, W * 0.2, 0, W * 0.6, -L * 0.15);
    this._pdBlister(group, mats, W * 0.2, 0, -W * 0.6, -L * 0.15, Math.PI);

    this._windowRow(group, mats, 9, W * 0.31, 0, -L * 0.28, L * 0.05, W * 0.022);
    this._windowRow(group, mats, 9, -W * 0.31, 0, -L * 0.28, L * 0.05, W * 0.022);

    if (opts.siege) {
      // Siege spinal: an absurdly long barrel running out past the bow
      const barrel = new THREE.Mesh(
        new THREE.CylinderGeometry(W * 0.07, W * 0.09, L * 0.6, 8), mats.trim);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, -W * 0.2, L * 0.5);
      group.add(barrel);
      for (const bz of [L * 0.34, L * 0.52, L * 0.7]) {
        const ring = new THREE.Mesh(
          new THREE.CylinderGeometry(W * 0.11, W * 0.11, W * 0.05, 8), mats.accent);
        ring.rotation.x = Math.PI / 2;
        ring.position.set(0, -W * 0.2, bz);
        group.add(ring);
      }
      this._mast(group, mats, W * 0.5, W * 0.14, W * 0.5, L * 0.3, -0.3);
    }

    // Radiator wings on the cardinal gaps between sponsons
    const radMat = this._makeRadiatorMaterial();
    group.userData.radiators = [0, Math.PI / 2, Math.PI, -Math.PI / 2].map(a =>
      this._radiatorPanel(group, radMat, W * 0.02, W * 0.55, L * 0.3,
        Math.sin(a) * -W * 0.32, Math.cos(a) * W * 0.32, -L * 0.24, a));
    group.userData.radiatorMaterial = radMat;
    group.userData.reactorPos = new THREE.Vector3(0, 0, -L * 0.3);
  }

  /**
   * Update ship state
   * @param {string} shipId - Ship identifier
   * @param {Object} state - Ship state from interpolator
   *   state.radiatorsExtended (optional) drives the radiator glow
   */
  updateShip(shipId, state) {
    let ship = this.ships.get(shipId);

    if (!ship) {
      // Create ship if it doesn't exist
      const faction = shipId.startsWith('alpha') ? 'alpha' : 'beta';
      ship = this.createShip(shipId, faction);
    }

    if (state.destroyed) {
      // Trigger destruction animation if not already done
      if (!this.destroyedShips.has(shipId)) {
        this.destroyedShips.add(shipId);
        const shipType = ship.userData.shipType || 'destroyer';
        this.createDestructionEffect(ship.position.clone(), shipType, this.currentTime, shipId);
      }
      // Two-stage death: during the drift phase the destruction effect owns
      // the hulk - it stays visible, coasting dark on its final vector with
      // hull fires popping, until the reactor cooks off.
      const eff = this.destructionEffects.find(e => e.shipId === shipId);
      if (eff && eff.driftDuration > 0 &&
          this.currentTime - eff.spawnTime < eff.driftDuration) {
        this._extinguishTorches(ship);
      } else {
        ship.visible = false;
      }
      return;
    }

    ship.visible = true;

    // Position (convert from meters to km)
    ship.position.set(
      state.position[0] * this.SCALE,
      state.position[1] * this.SCALE,
      state.position[2] * this.SCALE
    );

    // Orientation from forward vector
    if (state.forward) {
      const forward = new THREE.Vector3(state.forward[0], state.forward[1], state.forward[2]);
      if (forward.lengthSq() > 0) {
        forward.normalize();
        const target = ship.position.clone().add(forward);
        ship.lookAt(target);
      }
    }

    // Sim-side death spiral: the trace itself already carries the tumble
    // and the sputtering torch (forward + thrust per frame) - layer
    // occasional hull fires on top so the dying hulk reads as burning,
    // not merely adrift. The reactor detonation arrives later as the
    // destroyed flag, with reactorCause set, so no client-side drift is
    // added on top of the sim's.
    if (state.dying) {
      let next = this.dyingPopTimes.get(shipId);
      // (Re)arm after load or a timeline seek in either direction
      if (next === undefined || Math.abs(next - this.currentTime) > 3.0) {
        next = this.currentTime + 0.3 + Math.random() * 1.0;
      }
      if (this.currentTime >= next) {
        const L = ship.userData.size?.length || 1.0;
        const W = ship.userData.size?.width || 0.25;
        const world = new THREE.Vector3(
          (Math.random() - 0.5) * W * 0.9,
          (Math.random() - 0.5) * W * 0.9,
          (Math.random() - 0.5) * L * 0.85
        ).applyQuaternion(ship.quaternion).add(ship.position);
        this.spawnSmallEffect(
          [world.x / this.SCALE, world.y / this.SCALE, world.z / this.SCALE],
          this.currentTime, 'hullpop');
        next = this.currentTime + 0.4 + Math.random() * 1.4;
      }
      this.dyingPopTimes.set(shipId, next);
    } else if (this.dyingPopTimes.has(shipId)) {
      this.dyingPopTimes.delete(shipId);
    }

    // Radiators: extended panels unfold outward and glow heat-orange;
    // retracted they fold nearly flat against the hull and go dark.
    const radMat = ship.userData.radiatorMaterial;
    if (radMat) {
      const k = Math.min(1, (this.delta || 0.016) * 3);
      const target = state.radiatorsExtended ? 0.55 : 0.0;
      radMat.emissiveIntensity += (target - radMat.emissiveIntensity) * k;
      const ext = state.radiatorsExtended ? 1 : 0.18;
      for (const panel of ship.userData.radiators || []) {
        panel.scale.y += (ext - panel.scale.y) * k;
      }
    }

    // Fusion torch animation: length/brightness follow a SMOOTHED throttle
    // with slow, low-amplitude breathing. The old modulation ran at 53/131
    // rad/s - far above the frame Nyquist rate - so it aliased into random
    // frame-to-frame strobing ("flashes in and out"). A steady-state fusion
    // torch is CONTINUOUS: keep every modulation below ~1.5 Hz and small.
    const { torches, plumeLength } = ship.userData;
    const targetThrust = state.thrust || 0;
    const ramp = 1 - Math.exp(-(this.delta || 0.016) * 3.0);
    if (ship.userData.smoothThrust === undefined) ship.userData.smoothThrust = targetThrust;
    ship.userData.smoothThrust += (targetThrust - ship.userData.smoothThrust) * ramp;
    const thrust = ship.userData.smoothThrust;

    if (torches) {
      for (const torch of torches) {
        if (thrust > 0.02) {
          const breathe = 0.965
            + 0.025 * Math.sin(this.elapsed * 4.7 + torch.phase)
            + 0.014 * Math.sin(this.elapsed * 7.9 + torch.phase * 1.7);
          const len = plumeLength * (0.5 + 0.8 * thrust) * breathe;
          const width = 0.7 + 0.5 * thrust;

          torch.core.visible = true;
          torch.sheath.visible = true;
          torch.core.scale.set(width, width, len);
          torch.core.material.opacity = Math.min(1, 0.95 * thrust + 0.25) * breathe;
          torch.sheath.scale.set(width * 1.15, width * 1.15, len * 1.18);
          torch.sheath.material.opacity = 0.28 * thrust * breathe;

          torch.nozzleGlow.material.opacity = Math.min(1, thrust * 1.1) * breathe;

          for (let b = 0; b < torch.beads.length; b++) {
            const bead = torch.beads[b];
            const f = (b + 1) / (torch.beads.length + 1);
            bead.position.z = torch.bellZ - len * f;
            bead.scale.setScalar(ship.userData.size.width * (1.1 - f * 0.55) * breathe);
            bead.material.opacity = 0.4 * (1 - f) * thrust * breathe;
          }

          if (torch.light) torch.light.intensity = thrust * 7 * breathe;
        } else {
          torch.core.visible = false;
          torch.sheath.visible = false;
          torch.nozzleGlow.material.opacity = 0;
          for (const bead of torch.beads) bead.material.opacity = 0;
          if (torch.light) torch.light.intensity = 0;
        }
      }
    }
  }

  /**
   * Create or update a projectile
   * @param {Object} proj - Projectile data from interpolator
   */
  updateProjectile(proj) {
    let projectile = this.projectiles.get(proj.id);

    if (!projectile) {
      // Create new projectile
      const geom = new THREE.SphereGeometry(0.55, 8, 8);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffff00
      });
      projectile = new THREE.Mesh(geom, material);

      // Trail
      const trailGeom = new THREE.BufferGeometry();
      const trailPositions = new Float32Array(30 * 3); // 30 points
      trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
      const trailMaterial = new THREE.LineBasicMaterial({
        color: 0xffaa00,
        transparent: true,
        opacity: 0.6
      });
      const trail = new THREE.Line(trailGeom, trailMaterial);

      projectile.userData = {
        trail,
        trailHistory: []
      };

      this.scene.add(projectile);
      this.scene.add(trail);
      this.projectiles.set(proj.id, projectile);
    }

    // Update position
    const pos = new THREE.Vector3(
      proj.position[0] * this.SCALE,
      proj.position[1] * this.SCALE,
      proj.position[2] * this.SCALE
    );
    projectile.position.copy(pos);

    // Update trail
    const { trail, trailHistory } = projectile.userData;
    trailHistory.unshift(pos.clone());
    if (trailHistory.length > 30) trailHistory.pop();

    const positions = trail.geometry.attributes.position.array;
    for (let i = 0; i < trailHistory.length; i++) {
      positions[i * 3] = trailHistory[i].x;
      positions[i * 3 + 1] = trailHistory[i].y;
      positions[i * 3 + 2] = trailHistory[i].z;
    }
    trail.geometry.attributes.position.needsUpdate = true;
    trail.geometry.setDrawRange(0, trailHistory.length);

    // PD engagement - change projectile color (beams handled separately via events)
    if (proj.pdEngaged) {
      projectile.material.color.setHex(0xff0000);
    } else {
      projectile.material.color.setHex(0xffff00);
    }
  }

  /**
   * Remove projectiles that no longer exist
   * @param {Set} activeIds - Set of active projectile IDs
   */
  cleanupProjectiles(activeIds) {
    for (const [id, projectile] of this.projectiles) {
      if (!activeIds.has(id)) {
        this.scene.remove(projectile);
        if (projectile.userData.trail) {
          this.scene.remove(projectile.userData.trail);
        }
        this.projectiles.delete(id);
      }
    }
  }

  /**
   * Create or update a torpedo: elongated body, exhaust plume while burning,
   * faction-colored fading trail. Disabled (seeker-killed) rounds go dark
   * and tumble.
   * @param {Object} torp - Torpedo data from interpolator
   */
  updateTorpedo(torp) {
    let torpedo = this.torpedoes.get(torp.id);

    if (!torpedo) {
      torpedo = this.createTorpedoMesh(torp);
      this.scene.add(torpedo);
      this.scene.add(torpedo.userData.trail);
      this.torpedoes.set(torp.id, torpedo);
    }

    const pos = new THREE.Vector3(
      torp.position[0] * this.SCALE,
      torp.position[1] * this.SCALE,
      torp.position[2] * this.SCALE
    );
    torpedo.position.copy(pos);

    const { bodyMaterial, plume, plumeLight, trail, trailHistory } = torpedo.userData;

    // RETARGETING: the trace carries each round's current target, so a
    // target switch (the round's own decision after its victim died) is
    // detected here data-driven - works live and under scrubbing. Show a
    // cyan ping on the round and a fading lock line to its new victim.
    if (torpedo.userData.currentTargetId === undefined) {
      torpedo.userData.currentTargetId = torp.target;
    } else if (torp.target && torp.target !== torpedo.userData.currentTargetId) {
      torpedo.userData.currentTargetId = torp.target;
      torpedo.userData.retargetAt = this.currentTime;
      this.spawnSmallEffect(
        [pos.x / this.SCALE, pos.y / this.SCALE, pos.z / this.SCALE],
        this.currentTime, 'retarget');
    }
    this._updateRetargetLine(torpedo, torp, pos);

    // Orient along velocity
    if (!torp.disabled && torp.velocity) {
      const vel = new THREE.Vector3(torp.velocity[0], torp.velocity[1], torp.velocity[2]);
      if (vel.lengthSq() > 0) {
        torpedo.lookAt(pos.clone().add(vel.normalize()));
      }
    } else {
      // Dead seeker: slow tumble
      torpedo.rotation.x += (this.delta || 0.016) * 0.9;
      torpedo.rotation.y += (this.delta || 0.016) * 0.6;
    }

    // Body glow: hot while thrusting, dark red when blinded
    if (torp.disabled) {
      bodyMaterial.emissive.setHex(0x551111);
      bodyMaterial.emissiveIntensity = 0.4;
    } else {
      bodyMaterial.emissive.setHex(torpedo.userData.accentColor);
      bodyMaterial.emissiveIntensity = 0.9;
    }

    // Exhaust plume while burning fuel
    // Slow breathing only - 37 rad/s aliased into strobing at frame rates
    const flicker = 0.93 + 0.07 * Math.sin(this.elapsed * 6.1 + torpedo.userData.phase);
    if (torp.thrusting && !torp.disabled) {
      plume.material.opacity = 0.85 * flicker;
      plume.scale.set(2.2, 2.2, 4.5);
      if (plumeLight) plumeLight.intensity = 3.5 * flicker;
    } else {
      plume.material.opacity = 0;
      if (plumeLight) plumeLight.intensity = 0;
    }

    // Trail - faction colored, fading toward the tail
    trailHistory.unshift(pos.clone());
    if (trailHistory.length > 60) trailHistory.pop();

    const positions = trail.geometry.attributes.position.array;
    const colors = trail.geometry.attributes.color.array;
    const baseColor = torpedo.userData.trailColor;
    for (let i = 0; i < trailHistory.length; i++) {
      positions[i * 3] = trailHistory[i].x;
      positions[i * 3 + 1] = trailHistory[i].y;
      positions[i * 3 + 2] = trailHistory[i].z;
      const fade = 1 - i / 60;
      colors[i * 3] = baseColor.r * fade;
      colors[i * 3 + 1] = baseColor.g * fade;
      colors[i * 3 + 2] = baseColor.b * fade;
    }
    trail.geometry.attributes.position.needsUpdate = true;
    trail.geometry.attributes.color.needsUpdate = true;
    trail.geometry.setDrawRange(0, trailHistory.length);
  }

  createTorpedoMesh(torp) {
    const group = new THREE.Group();
    const isAlpha = torp.source === 'alpha' || (torp.source || '').startsWith('alpha');
    const accentColor = isAlpha ? 0x00d4ff : 0xff6644;

    // Body: slim cone-nosed cylinder pointing +Z
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0x9aa0ad,
      metalness: 0.7,
      roughness: 0.35,
      emissive: accentColor,
      emissiveIntensity: 0.9
    });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 1.6, 8), bodyMaterial);
    body.rotation.x = Math.PI / 2;
    group.add(body);

    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.55, 8), bodyMaterial);
    nose.rotation.x = Math.PI / 2;
    nose.position.z = 1.05;
    group.add(nose);

    // Exhaust plume: additive glow sprite behind the body
    const plume = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: 0xffc880,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    plume.position.z = -1.6;
    plume.scale.set(2.2, 2.2, 1);
    group.add(plume);

    const plumeLight = this.tryPointLight(0xffaa55, 0, 40);
    if (plumeLight) {
      plumeLight.position.z = -1.8;
      group.add(plumeLight);
    }

    // Trail line with per-vertex fading colors
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(60 * 3), 3));
    trailGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(60 * 3), 3));
    const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));

    group.userData = {
      bodyMaterial,
      plume,
      plumeLight,
      trail,
      trailHistory: [],
      accentColor,
      trailColor: new THREE.Color(accentColor),
      phase: Math.random() * Math.PI * 2
    };

    return group;
  }

  /**
   * Remove torpedoes that no longer exist
   * @param {Set} activeIds - Set of active torpedo IDs
   */
  /**
   * Fading lock line from a freshly-retargeted round to its new victim,
   * shown for a few seconds so the decision is readable in the replay.
   */
  _updateRetargetLine(torpedo, torp, pos) {
    const RETARGET_LINE_S = 4.0;
    const age = this.currentTime - (torpedo.userData.retargetAt ?? -Infinity);
    const targetPos = (age >= 0 && age <= RETARGET_LINE_S)
      ? this.getEntityPosition('ship', torp.target)
      : null;

    let line = torpedo.userData.retargetLine;
    if (!targetPos) {
      if (line) line.visible = false;
      return;
    }
    if (!line) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      line = new THREE.Line(geom, new THREE.LineBasicMaterial({
        color: 0x66ffee,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false
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
    for (const [id, torpedo] of this.torpedoes) {
      if (!activeIds.has(id)) {
        this.scene.remove(torpedo);
        if (torpedo.userData.trail) {
          this.scene.remove(torpedo.userData.trail);
        }
        if (torpedo.userData.retargetLine) {
          this.scene.remove(torpedo.userData.retargetLine);
          torpedo.userData.retargetLine.geometry.dispose();
          torpedo.userData.retargetLine.material.dispose();
        }
        this.freePointLight(torpedo.userData.plumeLight);
        this.torpedoes.delete(id);
      }
    }
  }

  /**
   * Position of a live tracked entity (for beam endpoints)
   */
  getEntityPosition(targetType, targetId) {
    if (targetType === 'torpedo') {
      const t = this.torpedoes.get(targetId);
      return t ? t.position : null;
    }
    if (targetType === 'slug' || targetType === 'projectile') {
      const p = this.projectiles.get(targetId);
      return p ? p.position : null;
    }
    const s = this.ships.get(targetId);
    return s && s.visible ? s.position : null;
  }

  /**
   * Continuous PD beam rendering.
   *
   * PD lasers are dwell weapons: the sim emits one pd_fired per turret per
   * tick while the beam stays on a target. The loader collapses those into
   * dwell segments; here we keep one beam mesh per active segment and
   * re-anchor both endpoints every frame so the beam tracks shooter and
   * target through interpolated motion.
   *
   * @param {Array} activeSegments - From loader.getActivePDBeams(time)
   * @param {number} time - Current playback time
   */
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
        beam = this.createBeamMesh(seg);
        this.beams.set(seg.key, beam);
      }

      // Turret offset: spread multiple turrets' beams so they don't overlap
      const from = sourcePos.clone().add(beam.muzzleOffset);
      const dir = targetPos.clone().sub(from);
      const length = dir.length();
      if (length < 0.5) continue;
      dir.normalize();

      // Fade envelope at segment edges + gentle shimmer. The old 43/127
      // rad/s flicker aliased at frame rates - the beam edges strobed like
      // a low-refresh TV. A continuous-dwell laser should look STEADY.
      const fadeIn = Math.min(1, (time - seg.start) / 0.25 + 0.15);
      const fadeOut = Math.min(1, Math.max(0, (seg.end + 1.0 - time) / 0.5));
      const flicker = 0.94 + 0.045 * Math.sin(this.elapsed * 6.7 + beam.phase)
        + 0.015 * Math.sin(this.elapsed * 9.3 + beam.phase * 2);
      const intensity = Math.min(fadeIn, fadeOut) * flicker;

      // Anchor cylinder between endpoints
      const mid = from.clone().add(targetPos).multiplyScalar(0.5);
      beam.group.position.copy(mid);
      beam.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      beam.core.scale.set(1, length, 1);
      beam.glow.scale.set(1, length, 1);
      beam.core.material.opacity = 0.85 * intensity;
      beam.glow.material.opacity = 0.15 * intensity;

      // Endpoint glows
      beam.muzzle.position.copy(from);
      beam.muzzle.material.opacity = 0.9 * intensity;
      beam.impact.position.copy(targetPos);
      beam.impact.material.opacity = 0.85 * intensity;
      const pulse = 1 + 0.1 * Math.sin(this.elapsed * 7.7 + beam.phase);
      beam.impact.scale.setScalar(1.7 * pulse);

      beam.group.visible = true;
      beam.muzzle.visible = true;
      beam.impact.visible = true;
    }

    // Remove beams whose segments ended (or whose endpoints vanished)
    for (const [key, beam] of this.beams) {
      if (!activeKeys.has(key)) {
        this.disposeBeam(beam);
        this.beams.delete(key);
      }
    }
  }

  createBeamMesh(seg) {
    const group = new THREE.Group();

    // Unit-length cylinders along +Y, scaled to span each frame
    const coreGeom = new THREE.CylinderGeometry(0.09, 0.09, 1, 6, 1, true);
    const core = new THREE.Mesh(coreGeom, new THREE.MeshBasicMaterial({
      color: 0xffe0c8,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    group.add(core);

    const glowGeom = new THREE.CylinderGeometry(0.22, 0.22, 1, 6, 1, true);
    const glow = new THREE.Mesh(glowGeom, new THREE.MeshBasicMaterial({
      color: 0xff5533,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide
    }));
    group.add(glow);

    const muzzle = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: 0xffaa66,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    muzzle.scale.setScalar(1.8);

    const impact = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: 0xffddaa,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    impact.scale.setScalar(2.6);

    this.scene.add(group);
    this.scene.add(muzzle);
    this.scene.add(impact);

    // Deterministic per-turret muzzle offset so parallel turret beams
    // separate visually (hash the turret name)
    let h = 0;
    for (const c of seg.turret || '') h = (h * 31 + c.charCodeAt(0)) | 0;
    const muzzleOffset = new THREE.Vector3(
      ((h & 0xff) / 255 - 0.5) * 2.4,
      (((h >> 8) & 0xff) / 255 - 0.5) * 2.4,
      (((h >> 16) & 0xff) / 255 - 0.5) * 2.4
    );

    return {
      group,
      core,
      glow,
      muzzle,
      impact,
      muzzleOffset,
      phase: Math.random() * Math.PI * 2
    };
  }

  disposeBeam(beam) {
    this.scene.remove(beam.group);
    this.scene.remove(beam.muzzle);
    this.scene.remove(beam.impact);
    beam.core.geometry.dispose();
    beam.core.material.dispose();
    beam.glow.geometry.dispose();
    beam.glow.material.dispose();
    beam.muzzle.material.dispose();
    beam.impact.material.dispose();
  }

  /**
   * Clear all PD beams (used when loading new recording or scrubbing)
   */
  clearPDBeams() {
    for (const beam of this.beams.values()) {
      this.disposeBeam(beam);
    }
    this.beams.clear();
  }

  /**
   * Spawn a hit effect at an impact position
   * Enhanced with dual shockwave rings, point light, and dense particle burst
   * @param {Array} position - [x, y, z] in meters
   * @param {number} energyGj - Impact energy for scaling effect size
   * @param {number} currentTime - Current playback time
   * @param {Object} opts - {flashColor, ringColor, innerRingColor} tint overrides
   *   (torpedo detonations pass a blue-white flash)
   */
  spawnHitEffect(position, energyGj, currentTime, opts = {}) {
    const group = new THREE.Group();

    // Position in scene units (km)
    const pos = new THREE.Vector3(
      position[0] * this.SCALE,
      position[1] * this.SCALE,
      position[2] * this.SCALE
    );
    group.position.copy(pos);

    // Scale effect based on energy (1-10 GJ typical range, but allow larger)
    const scale = Math.max(1, Math.min(8, Math.pow(energyGj, 0.4)));

    // Central flash - bright sphere with higher resolution
    const flashGeom = new THREE.SphereGeometry(0.9 * scale, 24, 24);
    const flashMat = new THREE.MeshBasicMaterial({
      color: opts.flashColor ?? 0xffeeaa,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const flash = new THREE.Mesh(flashGeom, flashMat);
    group.add(flash);

    // Flash light - enhances bloom and illuminates nearby
    const flashLight = this.tryPointLight(opts.flashColor ?? 0xffaa00, 0, scale * 10);
    if (flashLight) group.add(flashLight);

    // Primary expanding ring - thin orange shockwave (a wide annulus read
    // as a giant flat disc at close range)
    const ringGeom = new THREE.RingGeometry(1.15 * scale, 1.32 * scale, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color: opts.ringColor ?? 0xff6600,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.lookAt(this.camera.position);
    group.add(ring);

    // Secondary ring - inner blue plasma ring for contrast
    const innerRingGeom = new THREE.RingGeometry(0.72 * scale, 0.85 * scale, 48);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: opts.innerRingColor ?? 0x00aaff,
      transparent: true,
      opacity: 0.4,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const innerRing = new THREE.Mesh(innerRingGeom, innerRingMat);
    innerRing.lookAt(this.camera.position);
    group.add(innerRing);

    // Particle burst - more particles with color variation
    const particleCount = 100;
    const particleGeom = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);
    const particleVelocities = [];

    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = 0;
      particlePositions[i * 3 + 1] = 0;
      particlePositions[i * 3 + 2] = 0;

      // Random color: yellow to orange to red
      const heat = Math.random();
      particleColors[i * 3] = 1.0;
      particleColors[i * 3 + 1] = 0.4 + heat * 0.6;
      particleColors[i * 3 + 2] = heat * 0.2;

      // Random velocity with more outward force
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const speed = (3 + Math.random() * 5) * scale;
      particleVelocities.push({
        x: speed * Math.sin(phi) * Math.cos(theta),
        y: speed * Math.sin(phi) * Math.sin(theta),
        z: speed * Math.cos(phi)
      });
    }

    particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    particleGeom.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

    const particleMat = new THREE.PointsMaterial({
      vertexColors: true,
      size: scale * 0.45,
      map: this.glowTexture, // soft round sparks, not squares
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true
    });
    const particles = new THREE.Points(particleGeom, particleMat);
    group.add(particles);

    this.scene.add(group);

    this.hitEffects.push({
      group: group,
      flash: flash,
      flashLight: flashLight,
      ring: ring,
      innerRing: innerRing,
      particles: particles,
      particleVelocities: particleVelocities,
      spawnTime: currentTime,
      duration: opts.duration ?? 1.2,
      scale: scale
    });
  }

  /**
   * Update hit effects - animate and remove expired ones
   * @param {number} currentTime - Current playback time
   */
  updateHitEffects(currentTime) {
    const effectsToRemove = [];

    for (let i = 0; i < this.hitEffects.length; i++) {
      const effect = this.hitEffects[i];
      const age = currentTime - effect.spawnTime;

      if (age > effect.duration) {
        // Effect expired
        effectsToRemove.push(i);
        this.scene.remove(effect.group);
        effect.flash.geometry.dispose();
        effect.flash.material.dispose();
        this.freePointLight(effect.flashLight);
        effect.ring.geometry.dispose();
        effect.ring.material.dispose();
        effect.innerRing.geometry.dispose();
        effect.innerRing.material.dispose();
        effect.particles.geometry.dispose();
        effect.particles.material.dispose();
      } else if (age >= 0) {
        const progress = age / effect.duration;
        const easedProgress = Math.pow(progress, 0.7); // Ease for more initial intensity

        // Fade flash and expand
        effect.flash.material.opacity = Math.pow(1 - progress, 2);
        effect.flash.scale.setScalar(1.0 + easedProgress * 2.2);
        if (effect.flashLight) effect.flashLight.intensity = Math.pow(1 - progress, 3) * 20;

        // Expand and fade primary ring
        effect.ring.scale.setScalar(1.0 + easedProgress * 3.2);
        effect.ring.material.opacity = 0.55 * (1 - easedProgress);
        effect.ring.lookAt(this.camera.position);

        // Expand and fade inner ring - faster expansion
        effect.innerRing.scale.setScalar(1.0 + easedProgress * 4.5);
        effect.innerRing.material.opacity = Math.max(0, 0.4 * (1 - easedProgress * 1.2));
        effect.innerRing.lookAt(this.camera.position);

        // Update particles - expand outward with deceleration and fade
        const positions = effect.particles.geometry.attributes.position.array;
        const drag = 1.0 / (1.0 + age * 2); // Quick deceleration
        for (let j = 0; j < effect.particleVelocities.length; j++) {
          const vel = effect.particleVelocities[j];
          positions[j * 3] += vel.x * this.delta * drag;
          positions[j * 3 + 1] += vel.y * this.delta * drag;
          positions[j * 3 + 2] += vel.z * this.delta * drag;
        }
        effect.particles.geometry.attributes.position.needsUpdate = true;
        effect.particles.material.opacity = Math.pow(1 - progress, 1.5);

        effect.group.visible = true;
      } else {
        effect.group.visible = false;
      }
    }

    // Remove expired effects (reverse order)
    for (let i = effectsToRemove.length - 1; i >= 0; i--) {
      this.hitEffects.splice(effectsToRemove[i], 1);
    }
  }

  /**
   * Spawn a torpedo warhead detonation - larger, blue-white cored blast
   */
  spawnTorpedoDetonation(position, damageGj, currentTime) {
    this.spawnHitEffect(position, Math.max(damageGj, 8), currentTime, {
      flashColor: 0xddeeff,
      ringColor: 0xff8833,
      innerRingColor: 0x66ccff,
      duration: 2.2
    });
  }

  /**
   * Small transient effects: torpedo fizzles, seeker kills, muzzle flashes.
   * One parametric implementation - a glow sprite + light that pops and fades.
   * @param {Array} position - [x, y, z] in meters
   * @param {number} currentTime - Playback time
   * @param {string} kind - 'miss' | 'burnout' | 'seeker_kill' | 'intercept' | 'muzzle'
   */
  spawnSmallEffect(position, currentTime, kind) {
    const presets = {
      miss: { color: 0x8899bb, scale: 4, duration: 1.4, intensity: 2 },
      burnout: { color: 0xff8844, scale: 3, duration: 1.8, intensity: 3 },
      seeker_kill: { color: 0x66ffee, scale: 5, duration: 1.0, intensity: 6 },
      intercept: { color: 0xaaddff, scale: 6, duration: 1.0, intensity: 8 },
      destroyed: { color: 0xffcc88, scale: 6, duration: 1.0, intensity: 8 },
      muzzle: { color: 0xffe9b0, scale: 3.5, duration: 0.45, intensity: 6 },
      hullpop: { color: 0xffa860, scale: 3.4, duration: 0.9, intensity: 4 },
      retarget: { color: 0x66ffee, scale: 4.5, duration: 1.2, intensity: 5 }
    };
    const p = presets[kind] || presets.miss;

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: p.color,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    sprite.position.set(
      position[0] * this.SCALE,
      position[1] * this.SCALE,
      position[2] * this.SCALE
    );
    sprite.scale.setScalar(p.scale);

    const light = this.tryPointLight(p.color, 0, p.scale * 12);
    if (light) light.position.copy(sprite.position);

    this.scene.add(sprite);
    if (light) this.scene.add(light);

    this.smallEffects.push({
      sprite,
      light,
      spawnTime: currentTime,
      duration: p.duration,
      baseScale: p.scale,
      intensity: p.intensity
    });
  }

  /**
   * Animate and expire small effects
   */
  updateSmallEffects(currentTime) {
    const toRemove = [];
    for (let i = 0; i < this.smallEffects.length; i++) {
      const fx = this.smallEffects[i];
      const age = currentTime - fx.spawnTime;
      if (age > fx.duration || age < 0) {
        if (age > fx.duration) {
          toRemove.push(i);
        } else {
          fx.sprite.visible = false;
          if (fx.light) fx.light.intensity = 0;
        }
        continue;
      }
      const progress = age / fx.duration;
      fx.sprite.visible = true;
      fx.sprite.material.opacity = Math.pow(1 - progress, 1.6);
      fx.sprite.scale.setScalar(fx.baseScale * (1 + progress * 2));
      if (fx.light) fx.light.intensity = fx.intensity * Math.pow(1 - progress, 2);
    }
    for (let i = toRemove.length - 1; i >= 0; i--) {
      const fx = this.smallEffects[toRemove[i]];
      this.scene.remove(fx.sprite);
      if (fx.light) this.scene.remove(fx.light);
      this.freePointLight(fx.light);
      fx.sprite.material.dispose();
      this.smallEffects.splice(toRemove[i], 1);
    }
  }

  clearSmallEffects() {
    for (const fx of this.smallEffects) {
      this.scene.remove(fx.sprite);
      if (fx.light) this.scene.remove(fx.light);
      this.freePointLight(fx.light);
      fx.sprite.material.dispose();
    }
    this.smallEffects = [];
  }

  /**
   * Clear all hit effects
   */
  clearHitEffects() {
    for (const effect of this.hitEffects) {
      this.scene.remove(effect.group);
      effect.flash.geometry.dispose();
      effect.flash.material.dispose();
      this.freePointLight(effect.flashLight);
      effect.ring.geometry.dispose();
      effect.ring.material.dispose();
      effect.innerRing.geometry.dispose();
      effect.innerRing.material.dispose();
      effect.particles.geometry.dispose();
      effect.particles.material.dispose();
    }
    this.hitEffects = [];
  }

  /**
   * Create ship destruction effect - terrawatt fusion reactor explosion
   * Multi-phase: hull breaches -> secondary explosions -> reactor detonation -> plasma cloud
   * @param {THREE.Vector3} position - Ship position
   * @param {string} shipType - Ship type for scaling
   * @param {number} currentTime - Current playback time
   */
  createDestructionEffect(position, shipType, currentTime, shipId = null) {
    // Scale based on ship type
    const scaleFactors = {
      corvette: 0.6, frigate: 0.8, destroyer: 1.0,
      cruiser: 1.3, cruiser_torpedo: 1.3, battlecruiser: 1.6, battleship: 2.0,
      dreadnought: 2.5, dreadnought_siege: 2.5
    };
    const baseScale = (scaleFactors[shipType] || 1.0) * 8;
    const scale = baseScale * 1.5; // Increase overall scale for more spectacle

    // Two-stage death: unless the reactor itself was the killing blow (then
    // it goes up IMMEDIATELY), the ship first drifts dark on its last
    // vector with small hull explosions popping - and only then does the
    // reactor breach into the full detonation.
    const death = (this.deathInfo || {})[shipId] || {};
    const driftDuration = death.reactorCause ? 0 : 4.5 + Math.random() * 1.5;
    const vel = death.velocity || [0, 0, 0];
    const driftVel = new THREE.Vector3(vel[0], vel[1], vel[2]).multiplyScalar(this.SCALE);
    const shipGroup = shipId ? this.ships.get(shipId) : null;

    // Hull-fire schedule for the drift phase, offsets in the ship's local
    // frame so pops track the tumbling hulk.
    const hullPops = [];
    if (driftDuration > 0 && shipGroup) {
      const L = shipGroup.userData.size?.length || scale;
      const W = shipGroup.userData.size?.width || scale * 0.25;
      const popCount = 8 + Math.floor(Math.random() * 5);
      for (let i = 0; i < popCount; i++) {
        hullPops.push({
          t: 0.35 + Math.random() * (driftDuration - 1.2),
          local: new THREE.Vector3(
            (Math.random() - 0.5) * W * 0.9,
            (Math.random() - 0.5) * W * 0.9,
            (Math.random() - 0.5) * L * 0.85
          ),
          fired: false
        });
      }
      hullPops.sort((a, b) => a.t - b.t);
    }

    const group = new THREE.Group();
    group.position.copy(position);

    // Phase 1: Multiple staggered explosions (initial hull breaches)
    const explosionCount = 16;
    const explosions = [];
    for (let i = 0; i < explosionCount; i++) {
      // Random offset from center, larger spread
      const offset = new THREE.Vector3(
        (Math.random() - 0.5) * scale * 2.0,
        (Math.random() - 0.5) * scale * 2.0,
        (Math.random() - 0.5) * scale * 2.0
      );

      // Explosion flash - larger and brighter
      const flashGeom = new THREE.SphereGeometry(scale * (0.8 + Math.random() * 0.7), 24, 24);
      const flashMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(0.08 + Math.random() * 0.05, 1.0, 0.6),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const flash = new THREE.Mesh(flashGeom, flashMat);
      flash.position.copy(offset);
      group.add(flash);

      // No per-explosion PointLight: 40 dynamic lights made every lit
      // material in the scene crawl. The additive flashes + bloom carry
      // the look on their own.
      explosions.push({
        flash,
        offset,
        delay: i * 0.15 + Math.random() * 0.1,
        duration: 1.0 + Math.random() * 0.5
      });
    }

    // Phase 1.5: Secondary explosions (munitions/fuel detonations)
    const secondaryExplosionCount = 24;
    const secondaryExplosions = [];
    for (let i = 0; i < secondaryExplosionCount; i++) {
      // Larger spread for secondary blasts
      const offset = new THREE.Vector3(
        (Math.random() - 0.5) * scale * 3.0,
        (Math.random() - 0.5) * scale * 3.0,
        (Math.random() - 0.5) * scale * 3.0
      );

      // Smaller flashes for secondary explosions
      const flashGeom = new THREE.SphereGeometry(scale * (0.4 + Math.random() * 0.5), 16, 16);
      const flashMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(0.05 + Math.random() * 0.1, 1.0, 0.7),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const flash = new THREE.Mesh(flashGeom, flashMat);
      flash.position.copy(offset);
      group.add(flash);

      secondaryExplosions.push({
        flash,
        offset,
        delay: 2.0 + i * 0.1 + Math.random() * 0.2,
        duration: 0.8 + Math.random() * 0.4
      });
    }

    // Phase 2: Main fusion reactor detonation - massive expanding plasma sphere
    const plasmaGeom = new THREE.SphereGeometry(scale * 0.5, 32, 32);
    const plasmaMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const plasmaSphere = new THREE.Mesh(plasmaGeom, plasmaMat);
    group.add(plasmaSphere);

    // Central detonation light
    // Destruction is the showpiece: let it borrow past the base budget so a
    // kill still flashes even in a light-saturated fleet battle.
    const coreLight = this.tryPointLight(0xffffff, 0, scale * 20, 4);
    if (coreLight) group.add(coreLight);

    // BLINDING FLASH - massive white sphere that whites out everything
    const flashGeom = new THREE.SphereGeometry(scale * 50, 32, 32);
    const blindingFlashMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.BackSide // Render inside so camera sees it from within
    });
    const blindingFlash = new THREE.Mesh(flashGeom, blindingFlashMat);
    group.add(blindingFlash);

    // Secondary flash light - extremely intense
    const flashLight = this.tryPointLight(0xffffff, 0, scale * 100, 4);
    if (flashLight) group.add(flashLight);

    // Shockwave ring - expanding outward
    const shockGeom = new THREE.RingGeometry(scale * 0.5, scale * 1.0, 64);
    const shockMat = new THREE.MeshBasicMaterial({
      color: 0xff8800,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const shockwave = new THREE.Mesh(shockGeom, shockMat);
    shockwave.lookAt(this.camera.position);
    group.add(shockwave);

    // Phase 3: Debris and plasma cloud.
    // Motion lives entirely in the vertex shader (velocity + drag + tumble
    // from per-particle attributes and a single uAge uniform) - the old CPU
    // path iterated 50k particles AND re-uploaded the position buffer every
    // frame, which was most of the lag during a destruction sequence.
    const debrisCount = 50000;
    const debrisGeom = new THREE.BufferGeometry();
    const debrisPositions = new Float32Array(debrisCount * 3);
    const debrisVelocitiesAttr = new Float32Array(debrisCount * 3);
    const debrisColors = new Float32Array(debrisCount * 3);
    const debrisRnd = new Float32Array(debrisCount);

    for (let i = 0; i < debrisCount; i++) {
      // Start at random position within ship volume
      debrisPositions[i * 3] = (Math.random() - 0.5) * scale;
      debrisPositions[i * 3 + 1] = (Math.random() - 0.5) * scale;
      debrisPositions[i * 3 + 2] = (Math.random() - 0.5) * scale;

      // Random velocity - higher speeds, more variation
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const speed = (10 + Math.random() * 40) * scale * 0.4;
      debrisVelocitiesAttr[i * 3] = speed * Math.sin(phi) * Math.cos(theta) + (Math.random() - 0.5) * 10;
      debrisVelocitiesAttr[i * 3 + 1] = speed * Math.sin(phi) * Math.sin(theta) + (Math.random() - 0.5) * 10;
      debrisVelocitiesAttr[i * 3 + 2] = speed * Math.cos(phi) + (Math.random() - 0.5) * 10;
      debrisRnd[i] = Math.random();

      // Color: more varied plasma-like colors
      const heat = Math.random();
      if (heat > 0.95) {
        // Blue-white plasma
        debrisColors[i * 3] = 0.8;
        debrisColors[i * 3 + 1] = 0.9;
        debrisColors[i * 3 + 2] = 1.0;
      } else if (heat > 0.7) {
        // White-hot
        debrisColors[i * 3] = 1.0;
        debrisColors[i * 3 + 1] = 0.95;
        debrisColors[i * 3 + 2] = 0.8;
      } else if (heat > 0.4) {
        // Yellow-orange
        debrisColors[i * 3] = 1.0;
        debrisColors[i * 3 + 1] = 0.6 + Math.random() * 0.4;
        debrisColors[i * 3 + 2] = 0.2;
      } else {
        // Red-hot cooling debris
        debrisColors[i * 3] = 0.8 + Math.random() * 0.2;
        debrisColors[i * 3 + 1] = 0.2 + Math.random() * 0.2;
        debrisColors[i * 3 + 2] = 0.05;
      }
    }

    debrisGeom.setAttribute('position', new THREE.BufferAttribute(debrisPositions, 3));
    debrisGeom.setAttribute('aVelocity', new THREE.BufferAttribute(debrisVelocitiesAttr, 3));
    debrisGeom.setAttribute('aColor', new THREE.BufferAttribute(debrisColors, 3));
    debrisGeom.setAttribute('aRnd', new THREE.BufferAttribute(debrisRnd, 1));

    const debrisDuration = 17.0; // totalDuration - debrisDelay
    const debrisMat = new THREE.ShaderMaterial({
      uniforms: {
        uAge: { value: -1.0 },       // seconds since debris phase start, <0 = hidden
        uDuration: { value: debrisDuration },
        uSize: { value: scale * 0.1 },
        uMap: { value: this.glowTexture }
      },
      vertexShader: `
        attribute vec3 aVelocity;
        attribute vec3 aColor;
        attribute float aRnd;
        uniform float uAge;
        uniform float uDuration;
        uniform float uSize;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float age = max(uAge, 0.0);
          float drag = 1.0 / (1.0 + age * 0.05);
          vec3 p = position + aVelocity * age * drag;
          // slow pseudo-tumble wobble, matches the old CPU look
          float w = age + aRnd * 6.2831;
          p += vec3(sin(w), cos(w), sin(w * 0.5)) * aRnd * 0.6;

          float progress = clamp(age / uDuration, 0.0, 1.0);
          float fadeIn = clamp(age / 0.85, 0.0, 1.0);
          vAlpha = uAge < 0.0 ? 0.0 : fadeIn * (1.0 - progress);
          vColor = aColor;

          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = uSize * (1.0 - progress * 0.8) * (450.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform sampler2D uMap;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          vec4 tex = texture2D(uMap, gl_PointCoord);
          gl_FragColor = vec4(vColor, 1.0) * tex * vAlpha;
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    const debris = new THREE.Points(debrisGeom, debrisMat);
    debris.frustumCulled = false; // particles fly far beyond the base bounds
    group.add(debris);

    // Lingering plasma cloud - semi-transparent sphere that fades slowly
    const cloudGeom = new THREE.SphereGeometry(scale * 2, 32, 32);
    const cloudMat = new THREE.MeshBasicMaterial({
      color: 0xff5500,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const plasmaCloud = new THREE.Mesh(cloudGeom, cloudMat);
    group.add(plasmaCloud);

    this.scene.add(group);

    this.destructionEffects.push({
      group,
      explosions,
      secondaryExplosions,
      plasmaSphere,
      coreLight,
      blindingFlash,
      flashLight,
      shockwave,
      debris,
      plasmaCloud,
      spawnTime: currentTime,
      explosionPhase: 4.0,
      secondaryPhase: 6.0,
      flashDelay: 2.5,      // Blinding flash at reactor breach
      flashDuration: 0.8,   // Very quick but intense
      plasmaDelay: 2.0,
      debrisDelay: 3.0,
      cloudDelay: 4.0,
      totalDuration: 20.0,
      scale,
      // Two-stage death state
      shipId,
      shipGroup,
      basePos: position.clone(),
      driftDuration,
      driftVel,
      hullPops,
      tumbleAxis: new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5,
        Math.random() - 0.5).normalize(),
      tumbleRate: 0.08 + Math.random() * 0.12,
      ignited: false
    });
  }

  /**
   * Update destruction effects - multi-phase fusion reactor explosion
   * @param {number} currentTime - Current playback time
   */
  updateDestructionEffects(currentTime) {
    const effectsToRemove = [];

    for (let i = 0; i < this.destructionEffects.length; i++) {
      const effect = this.destructionEffects[i];
      const age = currentTime - effect.spawnTime;
      // Everything after the drift phase runs on blastAge: the original
      // single-stage sequence, just time-shifted by the drift.
      const blastAge = age - effect.driftDuration;

      if (blastAge > effect.totalDuration) {
        // Effect expired - clean up all components
        effectsToRemove.push(i);
        this.scene.remove(effect.group);
        for (const exp of effect.explosions) {
          exp.flash.geometry.dispose();
          exp.flash.material.dispose();
        }
        for (const sec of effect.secondaryExplosions) {
          sec.flash.geometry.dispose();
          sec.flash.material.dispose();
        }
        effect.plasmaSphere.geometry.dispose();
        effect.plasmaSphere.material.dispose();
        this.freePointLight(effect.coreLight);
        effect.blindingFlash.geometry.dispose();
        effect.blindingFlash.material.dispose();
        this.freePointLight(effect.flashLight);
        effect.shockwave.geometry.dispose();
        effect.shockwave.material.dispose();
        effect.debris.geometry.dispose();
        effect.debris.material.dispose();
        effect.plasmaCloud.geometry.dispose();
        effect.plasmaCloud.material.dispose();
        // Reset bloom to default
        if (this.bloomPass) {
          this.bloomPass.strength = this.BLOOM_BASE;
        }
      } else if (age >= 0 && blastAge < 0) {
        // DRIFT PHASE: the hulk coasts dark on its last vector, tumbling
        // slowly, hull fires popping - the reactor hasn't gone yet.
        effect.group.visible = false;
        effect.group.position.copy(effect.basePos)
          .addScaledVector(effect.driftVel, age);

        const hulk = effect.shipGroup;
        if (hulk) {
          hulk.visible = true;
          hulk.position.copy(effect.group.position);
          hulk.rotateOnAxis(effect.tumbleAxis, effect.tumbleRate * (this.delta || 0.016));
        }

        for (const pop of effect.hullPops) {
          if (pop.fired || pop.t > age) continue;
          pop.fired = true;
          if (age - pop.t < 0.4 && hulk) {  // skip pops jumped over by a seek
            const world = pop.local.clone().applyQuaternion(hulk.quaternion)
              .add(effect.group.position);
            this.spawnSmallEffect(
              [world.x / this.SCALE, world.y / this.SCALE, world.z / this.SCALE],
              currentTime, 'hullpop');
          }
        }
      } else if (blastAge >= 0) {
        // REACTOR BREACH: hide the hulk, center the blast on the drifted
        // reactor position, then run the original detonation sequence.
        if (!effect.ignited) {
          effect.ignited = true;
          effect.group.position.copy(effect.basePos)
            .addScaledVector(effect.driftVel, effect.driftDuration);
          const hulk = effect.shipGroup;
          if (hulk) {
            const reactor = hulk.userData.reactorPos;
            if (reactor) {
              effect.group.position.add(
                reactor.clone().applyQuaternion(hulk.quaternion));
            }
            hulk.visible = false;
          }
        }
        effect.group.visible = true;

        // Phase 1: Update primary explosions
        for (const exp of effect.explosions) {
          const expAge = blastAge - exp.delay;
          if (expAge >= 0 && expAge < exp.duration) {
            const progress = expAge / exp.duration;
            exp.flash.material.opacity = Math.sin(progress * Math.PI) * 1.0;
            exp.flash.scale.setScalar(1.0 + progress * 5);
          } else if (expAge >= exp.duration) {
            exp.flash.material.opacity = 0;
          }
        }

        // Phase 1.5: Update secondary explosions
        for (const sec of effect.secondaryExplosions) {
          const secAge = blastAge - sec.delay;
          if (secAge >= 0 && secAge < sec.duration) {
            const progress = secAge / sec.duration;
            sec.flash.material.opacity = Math.sin(progress * Math.PI) * 0.8;
            sec.flash.scale.setScalar(1.0 + progress * 4);
          } else if (secAge >= sec.duration) {
            sec.flash.material.opacity = 0;
          }
        }

        // Phase 2: Main plasma sphere detonation
        const plasmaAge = blastAge - effect.plasmaDelay;
        if (plasmaAge >= 0) {
          const plasmaDuration = 5.0;
          const plasmaProgress = Math.min(1, plasmaAge / plasmaDuration);
          effect.plasmaSphere.material.opacity = Math.pow(1 - plasmaProgress, 0.5) * 0.8;
          effect.plasmaSphere.scale.setScalar(1.0 + plasmaProgress * 10);
          if (effect.coreLight) effect.coreLight.intensity = Math.pow(1 - plasmaProgress, 2) * 50;

          // Shockwave
          const shockProgress = plasmaAge / 8.0;
          if (shockProgress < 1) {
            effect.shockwave.material.opacity = (1 - shockProgress) * 0.6;
            effect.shockwave.scale.setScalar(1.0 + shockProgress * 20);
            effect.shockwave.lookAt(this.camera.position);
          } else {
            effect.shockwave.material.opacity = 0;
          }
        }

        // BLINDING FLASH - whites out everything at reactor breach
        const flashAge = blastAge - effect.flashDelay;
        if (flashAge >= 0 && flashAge < effect.flashDuration) {
          // Very fast attack, slower decay
          const flashProgress = flashAge / effect.flashDuration;
          let flashIntensity;
          if (flashProgress < 0.1) {
            // Instant rise to full white
            flashIntensity = flashProgress / 0.1;
          } else {
            // Exponential decay
            flashIntensity = Math.pow(1 - (flashProgress - 0.1) / 0.9, 2);
          }
          // Make it BLINDING - full opacity white
          effect.blindingFlash.material.opacity = flashIntensity * 1.0;
          effect.blindingFlash.scale.setScalar(1.0 + flashProgress * 2);
          // Extremely intense light
          if (effect.flashLight) effect.flashLight.intensity = flashIntensity * 200;
          // Also boost bloom temporarily if available
          if (this.bloomPass) {
            this.bloomPass.strength = this.BLOOM_BASE + flashIntensity * 3.0;
          }
        } else if (flashAge >= effect.flashDuration) {
          effect.blindingFlash.material.opacity = 0;
          if (effect.flashLight) effect.flashLight.intensity = 0;
          // Reset bloom
          if (this.bloomPass) {
            this.bloomPass.strength = this.BLOOM_BASE;
          }
        }

        // Phase 3: Debris ejection - all motion, fade, and shrink happen in
        // the vertex shader; one uniform write per frame is the entire cost
        effect.debris.material.uniforms.uAge.value =
          blastAge > effect.debrisDelay ? blastAge - effect.debrisDelay : -1.0;

        // Phase 4: Lingering plasma cloud
        const cloudAge = blastAge - effect.cloudDelay;
        if (cloudAge >= 0) {
          const cloudDuration = 12.0;
          const cloudProgress = Math.min(1, cloudAge / cloudDuration);
          effect.plasmaCloud.material.opacity = Math.pow(1 - cloudProgress, 1.5) * 0.22;
          effect.plasmaCloud.scale.setScalar(1.0 + cloudProgress * 5);
        }
      } else {
        effect.group.visible = false;
      }
    }

    // Remove expired effects
    for (let i = effectsToRemove.length - 1; i >= 0; i--) {
      this.destructionEffects.splice(effectsToRemove[i], 1);
    }
  }

  /**
   * Clear destruction effects and reset destroyed ships tracking
   */
  clearDestructionEffects() {
    for (const effect of this.destructionEffects) {
      this.scene.remove(effect.group);
      // Clean up primary explosions
      for (const exp of effect.explosions) {
        exp.flash.geometry.dispose();
        exp.flash.material.dispose();
      }
      // Clean up secondary explosions
      for (const sec of effect.secondaryExplosions) {
        sec.flash.geometry.dispose();
        sec.flash.material.dispose();
      }
      // Clean up plasma sphere and core light
      effect.plasmaSphere.geometry.dispose();
      effect.plasmaSphere.material.dispose();
      this.freePointLight(effect.coreLight);
      // Clean up blinding flash
      effect.blindingFlash.geometry.dispose();
      effect.blindingFlash.material.dispose();
      this.freePointLight(effect.flashLight);
      // Clean up shockwave
      effect.shockwave.geometry.dispose();
      effect.shockwave.material.dispose();
      // Clean up debris
      effect.debris.geometry.dispose();
      effect.debris.material.dispose();
      // Clean up plasma cloud
      effect.plasmaCloud.geometry.dispose();
      effect.plasmaCloud.material.dispose();
    }
    this.destructionEffects = [];
    this.destroyedShips.clear();
    // Reset bloom to default
    if (this.bloomPass) {
      this.bloomPass.strength = this.BLOOM_BASE;
    }
  }

  /**
   * Get ship position for camera targeting
   */
  getShipPosition(shipId) {
    const ship = this.ships.get(shipId);
    return ship ? ship.position.clone() : null;
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.composer.setSize(window.innerWidth, window.innerHeight);
  }

  render() {
    this.delta = this.clock.getDelta();
    this.elapsed = this.clock.elapsedTime;
    this.controls.update();
    // Use composer for bloom post-processing
    this.composer.render();
  }
}
