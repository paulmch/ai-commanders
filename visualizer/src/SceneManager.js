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
    this.pdcBeams = []; // Active PD laser beams [{line, spawnTime, duration}]
    this.hitEffects = []; // Active hit effects [{group, spawnTime, duration}]
    this.destructionEffects = []; // Active destruction effects
    this.destroyedShips = new Set(); // Ships that have already had destruction animation triggered

    // Scale: 1 unit = 1 km, positions in recording are meters
    this.SCALE = 1 / 1000;
    // Ships are exaggerated for visibility (actual ships are ~100m, we make them ~5km visible)
    this.SHIP_SCALE = 50;

    // PD beam settings
    this.PD_BEAM_DURATION = 0.3; // seconds before beam fades completely
    this.currentTime = 0; // Current playback time

    this.init();
  }

  init() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000011);

    // Clock for time-based animation
    this.clock = new THREE.Clock();
    this.delta = 0;

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

    // Orbit controls
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 50;
    this.controls.maxDistance = 50000;

    // Post-processing with bloom for ethereal glow
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.8,  // bloom strength
      0.4,  // radius
      0.2   // threshold (lower = more things bloom)
    );
    this.composer.addPass(bloomPass);
    this.bloomPass = bloomPass;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(100, 100, 100);
    this.scene.add(directionalLight);

    // Reference grid - XZ plane (floor)
    const gridXZ = new THREE.GridHelper(3000, 30, 0x444444, 0x222222);
    this.scene.add(gridXZ);

    // Reference grid - XY plane (vertical, for Z-axis reference)
    const gridXY = new THREE.GridHelper(3000, 30, 0x333355, 0x1a1a33);
    gridXY.rotation.x = Math.PI / 2; // Rotate to vertical
    this.scene.add(gridXY);

    // Reference grid - YZ plane (side vertical)
    const gridYZ = new THREE.GridHelper(3000, 30, 0x553333, 0x331a1a);
    gridYZ.rotation.z = Math.PI / 2; // Rotate to side
    this.scene.add(gridYZ);

    // Starfield
    this.createStarfield();

    // Handle resize
    window.addEventListener('resize', () => this.onResize());
  }

  createStarfield() {
    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const radius = 5000 + Math.random() * 5000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
    }

    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 2,
      sizeAttenuation: false
    });

    const stars = new THREE.Points(starsGeometry, starsMaterial);
    this.scene.add(stars);
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
      corvette: { length: 0.5, width: 0.12, isCapital: false },
      frigate: { length: 0.7, width: 0.16, isCapital: false },
      destroyer: { length: 1.0, width: 0.22, isCapital: false },
      cruiser: { length: 1.3, width: 0.28, isCapital: false },
      battlecruiser: { length: 1.6, width: 0.4, isCapital: true },
      battleship: { length: 2.0, width: 0.5, isCapital: true },
      dreadnought: { length: 2.5, width: 0.65, isCapital: true }
    };
    // Handle ship variants (e.g., dreadnought_siege uses dreadnought size)
    const variantMap = { dreadnought_siege: 'dreadnought' };
    const baseType = variantMap[shipType] || shipType;
    const baseSize = baseSizes[baseType] || baseSizes.destroyer;

    // Scale ships to be visible at km distances
    const scale = 8;
    const size = {
      length: baseSize.length * scale,
      width: baseSize.width * scale,
      isCapital: baseSize.isCapital
    };

    // Colors
    const primaryColor = faction === 'alpha' ? 0x2a3a4a : 0x3a2a2a;
    const accentColor = faction === 'alpha' ? 0x00d4ff : 0xff6644;
    const emissiveColor = faction === 'alpha' ? 0x002233 : 0x331111;

    // Materials
    const hullMaterial = new THREE.MeshStandardMaterial({
      color: primaryColor,
      metalness: 0.8,
      roughness: 0.4,
      emissive: emissiveColor,
      emissiveIntensity: 0.2
    });

    const accentMaterial = new THREE.MeshStandardMaterial({
      color: accentColor,
      metalness: 0.6,
      roughness: 0.3,
      emissive: accentColor,
      emissiveIntensity: 0.5
    });

    if (size.isCapital) {
      // DONNAGER-STYLE CAPITAL SHIP
      this.createCapitalShipHull(group, size, hullMaterial, accentMaterial);
    } else {
      // TACHI-STYLE CORVETTE/FRIGATE
      this.createCorvetteHull(group, size, hullMaterial, accentMaterial);
    }

    // Engine configuration based on ship type
    const engineConfig = this.getEngineConfig(shipType, size);

    // Create multiple engine plumes based on ship class
    const enginePlumes = [];
    const engineLights = [];
    const allPlumeRandoms = [];

    for (let e = 0; e < engineConfig.count; e++) {
      const offset = engineConfig.positions[e];

      // Engine glow light for this bell
      const engineLight = new THREE.PointLight(0x4488ff, 0, size.length * 3);
      engineLight.position.set(offset.x, offset.y, -size.length * 0.5);
      group.add(engineLight);
      engineLights.push(engineLight);

      // Engine plume particle system for this bell - more plumes = more particles total
      const plumeParticleCount = 800; // Per-plume particle count
      const plumeGeometry = new THREE.BufferGeometry();
      const plumePositions = new Float32Array(plumeParticleCount * 3);
      const plumeColors = new Float32Array(plumeParticleCount * 3);

      // Precompute per-particle randoms
      const particleRandoms = [];
      for (let i = 0; i < plumeParticleCount; i++) {
        particleRandoms.push({
          angle: Math.random() * Math.PI * 2,
          r: Math.random(),
          life: Math.random(),
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          vz: (Math.random() - 0.5) * 0.1,
          driftAngle: Math.random() * Math.PI * 2
        });
        plumePositions[i * 3] = offset.x;
        plumePositions[i * 3 + 1] = offset.y;
        plumePositions[i * 3 + 2] = 0;
        plumeColors[i * 3] = 1;
        plumeColors[i * 3 + 1] = 1;
        plumeColors[i * 3 + 2] = 1;
      }
      plumeGeometry.setAttribute('position', new THREE.BufferAttribute(plumePositions, 3));
      plumeGeometry.setAttribute('color', new THREE.BufferAttribute(plumeColors, 3));

      const plumeMaterial = new THREE.PointsMaterial({
        size: size.width * engineConfig.particleSize,
        vertexColors: true,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true
      });

      const enginePlume = new THREE.Points(plumeGeometry, plumeMaterial);
      enginePlume.userData.offset = offset;
      group.add(enginePlume);
      enginePlumes.push(enginePlume);
      allPlumeRandoms.push(particleRandoms);
    }

    // Store references and plume config
    group.userData = {
      shipId,
      faction,
      shipType,
      engineLights,
      enginePlumes,
      size,
      engineConfig,
      allPlumeRandoms,
      plumeMaxLife: 1.5,
      plumeLength: size.length * engineConfig.plumeLength,
      plumeSpread: size.width * engineConfig.plumeSpread
    };

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

    // Engine bell counts and arrangement per ship class
    const configs = {
      corvette: {
        count: 1,
        positions: [{ x: 0, y: 0 }],
        particleSize: 0.08,
        plumeLength: 6,
        plumeSpread: 4
      },
      frigate: {
        count: 2,
        positions: [
          { x: -W * 0.15, y: 0 },
          { x: W * 0.15, y: 0 }
        ],
        particleSize: 0.07,
        plumeLength: 7,
        plumeSpread: 4
      },
      destroyer: {
        count: 2,
        positions: [
          { x: -W * 0.2, y: 0 },
          { x: W * 0.2, y: 0 }
        ],
        particleSize: 0.06,
        plumeLength: 8,
        plumeSpread: 5
      },
      cruiser: {
        count: 3,
        positions: [
          { x: 0, y: W * 0.15 },
          { x: -W * 0.2, y: -W * 0.1 },
          { x: W * 0.2, y: -W * 0.1 }
        ],
        particleSize: 0.055,
        plumeLength: 8,
        plumeSpread: 5
      },
      battlecruiser: {
        count: 4,
        positions: [
          { x: -W * 0.15, y: W * 0.12 },
          { x: W * 0.15, y: W * 0.12 },
          { x: -W * 0.15, y: -W * 0.12 },
          { x: W * 0.15, y: -W * 0.12 }
        ],
        particleSize: 0.05,
        plumeLength: 9,
        plumeSpread: 6
      },
      battleship: {
        count: 5,
        positions: [
          { x: 0, y: W * 0.18 },
          { x: -W * 0.18, y: W * 0.05 },
          { x: W * 0.18, y: W * 0.05 },
          { x: -W * 0.12, y: -W * 0.15 },
          { x: W * 0.12, y: -W * 0.15 }
        ],
        particleSize: 0.045,
        plumeLength: 10,
        plumeSpread: 6
      },
      dreadnought: {
        count: 5,
        positions: [
          { x: 0, y: W * 0.2 },
          { x: -W * 0.19, y: W * 0.06 },
          { x: W * 0.19, y: W * 0.06 },
          { x: -W * 0.12, y: -W * 0.16 },
          { x: W * 0.12, y: -W * 0.16 }
        ],
        particleSize: 0.04,
        plumeLength: 12,
        plumeSpread: 7
      }
    };

    // Handle ship variants (e.g., dreadnought_siege uses dreadnought config)
    const variantMap = {
      dreadnought_siege: 'dreadnought'
    };
    const baseType = variantMap[shipType] || shipType;

    return configs[baseType] || configs.destroyer;
  }

  /**
   * Create Donnager-style capital ship hull
   * Tall angular tower with drive cone base, extensive superstructure
   */
  createCapitalShipHull(group, size, hullMaterial, accentMaterial) {
    const L = size.length;
    const W = size.width;

    // Main hull - tall angular box
    const mainHull = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.7, W * 0.8, L * 0.5),
      hullMaterial
    );
    mainHull.position.set(0, 0, 0);
    group.add(mainHull);

    // Forward section - tapered nose at front (+Z), tip pointing outward
    const noseGeom = new THREE.ConeGeometry(W * 0.5, L * 0.35, 4);
    noseGeom.rotateY(Math.PI / 4);
    const nose = new THREE.Mesh(noseGeom, hullMaterial);
    nose.rotation.x = Math.PI / 2; // FIXED: tip points +Z (outward)
    nose.position.set(0, 0, L * 0.4);
    group.add(nose);

    // Drive cone (engine section) at back (-Z), base facing outward
    const driveCone = new THREE.Mesh(
      new THREE.ConeGeometry(W * 0.6, L * 0.4, 4),
      hullMaterial
    );
    driveCone.rotation.x = Math.PI / 2; // base toward -Z (back)
    driveCone.rotation.y = Math.PI / 4;
    driveCone.position.set(0, 0, -L * 0.35);
    group.add(driveCone);

    // Bridge superstructure - tiered blocks on top (toward front +Z)
    const bridge1 = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.45, W * 0.3, L * 0.3),
      hullMaterial
    );
    bridge1.position.set(0, W * 0.5, L * 0.15);
    group.add(bridge1);

    const bridge2 = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.3, W * 0.25, L * 0.2),
      hullMaterial
    );
    bridge2.position.set(0, W * 0.7, L * 0.2);
    group.add(bridge2);

    // Command tower - tallest point (toward front)
    const tower = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.15, W * 0.35, L * 0.12),
      hullMaterial
    );
    tower.position.set(0, W * 0.95, L * 0.25);
    group.add(tower);

    // Sensor dome on top of tower
    const sensorDome = new THREE.Mesh(
      new THREE.SphereGeometry(W * 0.08, 8, 6),
      accentMaterial
    );
    sensorDome.position.set(0, W * 1.15, L * 0.25);
    group.add(sensorDome);

    // Antenna arrays - multiple masts (toward front)
    for (let i = -1; i <= 1; i += 2) {
      const antenna = new THREE.Mesh(
        new THREE.CylinderGeometry(W * 0.015, W * 0.015, W * 0.4, 4),
        accentMaterial
      );
      antenna.position.set(i * W * 0.2, W * 0.9, L * 0.35);
      antenna.rotation.z = i * Math.PI / 8;
      group.add(antenna);
    }

    // Side sponsons (weapon platforms)
    const sponsonGeom = new THREE.BoxGeometry(W * 0.2, W * 0.15, L * 0.25);
    const sponsonL = new THREE.Mesh(sponsonGeom, hullMaterial);
    sponsonL.position.set(-W * 0.45, 0, L * 0.1);
    group.add(sponsonL);

    const sponsonR = new THREE.Mesh(sponsonGeom, hullMaterial);
    sponsonR.position.set(W * 0.45, 0, L * 0.1);
    group.add(sponsonR);

    // Accent stripes - glowing panels
    const stripeL = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.04, W * 0.5, L * 0.4),
      accentMaterial
    );
    stripeL.position.set(-W * 0.36, 0, 0);
    group.add(stripeL);

    const stripeR = stripeL.clone();
    stripeR.position.set(W * 0.36, 0, 0);
    group.add(stripeR);

    // Engine bells (multiple in a cluster) at back (-Z)
    const bellPositions = [
      [0, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]
    ];
    for (const [i, j] of bellPositions) {
      const bell = new THREE.Mesh(
        new THREE.CylinderGeometry(W * 0.1, W * 0.14, L * 0.08, 8),
        accentMaterial
      );
      bell.rotation.x = Math.PI / 2;
      bell.position.set(i * W * 0.18, j * W * 0.18, -L * 0.52);
      group.add(bell);
    }

    // Radiator panels (toward engine section at back)
    const radiatorGeom = new THREE.BoxGeometry(W * 0.03, W * 0.5, L * 0.2);
    const radL = new THREE.Mesh(radiatorGeom, hullMaterial);
    radL.position.set(-W * 0.38, W * 0.25, -L * 0.15);
    group.add(radL);

    const radR = new THREE.Mesh(radiatorGeom, hullMaterial);
    radR.position.set(W * 0.38, W * 0.25, -L * 0.15);
    group.add(radR);
  }

  /**
   * Create smaller warship hull - tower style like Donnager but smaller
   * No cockpit, angular military design
   */
  createCorvetteHull(group, size, hullMaterial, accentMaterial) {
    const L = size.length;
    const W = size.width;

    // Main hull - angular box
    const mainHull = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.5, W * 0.6, L * 0.5),
      hullMaterial
    );
    mainHull.position.set(0, 0, 0);
    group.add(mainHull);

    // Nose section at front (+Z), tip pointing outward
    const noseGeom = new THREE.ConeGeometry(W * 0.35, L * 0.4, 4);
    noseGeom.rotateY(Math.PI / 4);
    const nose = new THREE.Mesh(noseGeom, hullMaterial);
    nose.rotation.x = Math.PI / 2; // FIXED: tip points +Z (outward)
    nose.position.set(0, 0, L * 0.4);
    group.add(nose);

    // Drive cone at back (-Z), base facing outward
    const driveCone = new THREE.Mesh(
      new THREE.ConeGeometry(W * 0.45, L * 0.35, 4),
      hullMaterial
    );
    driveCone.rotation.x = Math.PI / 2; // base toward -Z (back)
    driveCone.rotation.y = Math.PI / 4;
    driveCone.position.set(0, 0, -L * 0.35);
    group.add(driveCone);

    // Upper superstructure (toward front +Z)
    const superstructure = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.25, W * 0.25, L * 0.2),
      hullMaterial
    );
    superstructure.position.set(0, W * 0.35, L * 0.05);
    group.add(superstructure);

    // Sensor mast on top (toward front)
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(W * 0.03, W * 0.03, W * 0.3, 4),
      accentMaterial
    );
    mast.position.set(0, W * 0.55, L * 0.1);
    group.add(mast);

    // Engine bell at back (-Z)
    const engineBell = new THREE.Mesh(
      new THREE.CylinderGeometry(W * 0.25, W * 0.35, L * 0.1, 8),
      accentMaterial
    );
    engineBell.rotation.x = Math.PI / 2;
    engineBell.position.set(0, 0, -L * 0.5);
    group.add(engineBell);

    // Accent stripes on sides
    const stripe = new THREE.Mesh(
      new THREE.BoxGeometry(W * 0.05, W * 0.4, L * 0.4),
      accentMaterial
    );
    stripe.position.set(W * 0.27, 0, 0);
    group.add(stripe);

    const stripe2 = stripe.clone();
    stripe2.position.set(-W * 0.27, 0, 0);
    group.add(stripe2);

    // Side weapon pods
    const podGeom = new THREE.BoxGeometry(W * 0.15, W * 0.12, L * 0.2);
    const podL = new THREE.Mesh(podGeom, hullMaterial);
    podL.position.set(-W * 0.35, 0, L * 0.1);
    group.add(podL);

    const podR = new THREE.Mesh(podGeom, hullMaterial);
    podR.position.set(W * 0.35, 0, L * 0.1);
    group.add(podR);

    // Radiator fins (toward engine at back)
    const finGeom = new THREE.BoxGeometry(W * 0.02, W * 0.3, L * 0.15);
    const finL = new THREE.Mesh(finGeom, hullMaterial);
    finL.position.set(-W * 0.28, W * 0.15, -L * 0.2);
    group.add(finL);

    const finR = new THREE.Mesh(finGeom, hullMaterial);
    finR.position.set(W * 0.28, W * 0.15, -L * 0.2);
    group.add(finR);
  }

  /**
   * Update ship state
   * @param {string} shipId - Ship identifier
   * @param {Object} state - Ship state from interpolator
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
        this.createDestructionEffect(ship.position.clone(), shipType, this.currentTime);
      }
      ship.visible = false;
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

    // Engine thrust visualization - multiple plumes
    const { engineLights, enginePlumes, size, engineConfig, allPlumeRandoms } = ship.userData;
    const thrust = state.thrust || 0;

    // Scale all engine lights based on thrust
    if (engineLights) {
      for (const light of engineLights) {
        light.intensity = thrust * 6;
      }
    }

    // Animate all engine plumes
    if (enginePlumes && allPlumeRandoms && this.delta) {
      const dt = this.delta;
      const maxLife = ship.userData.plumeMaxLife;
      const lifeSpeed = (1 / maxLife) * (1 + thrust);
      const plumeLength = ship.userData.plumeLength * (1 + thrust * 0.5);
      const maxSpread = ship.userData.plumeSpread * (1 + thrust * 0.5);

      for (let e = 0; e < enginePlumes.length; e++) {
        const enginePlume = enginePlumes[e];
        const particleRandoms = allPlumeRandoms[e];
        const offset = enginePlume.userData.offset;

        if (thrust > 0) {
          enginePlume.material.opacity = Math.min(thrust * 1.2, 0.9);
          enginePlume.material.size = size.width * engineConfig.particleSize * (0.5 + thrust * 0.8);

          const positions = enginePlume.geometry.attributes.position.array;
          const colors = enginePlume.geometry.attributes.color.array;
          const particleCount = positions.length / 3;

          for (let i = 0; i < particleCount; i++) {
            const random = particleRandoms[i];
            random.life += dt * lifeSpeed;

            if (random.life > 1) {
              random.life %= 1;
              random.angle = Math.random() * Math.PI * 2;
              random.r = Math.random();
              random.vx = (Math.random() - 0.5) * 0.3;
              random.vy = (Math.random() - 0.5) * 0.3;
              random.driftAngle = Math.random() * Math.PI * 2;
            }

            random.driftAngle += (Math.random() - 0.5) * 0.1;
            random.vx += Math.cos(random.driftAngle) * dt * 0.5;
            random.vy += Math.sin(random.driftAngle) * dt * 0.5;

            const t = random.life;
            const spread = Math.pow(t, 1.5) * maxSpread;

            // Base position from engine bell offset + expansion cone
            let px = offset.x + Math.cos(random.angle) * spread * random.r;
            let py = offset.y + Math.sin(random.angle) * spread * random.r;
            let pz = -size.length * 0.5 - t * plumeLength;

            // Add wispy Brownian noise
            const wispScale = t * size.width * 0.5;
            px += random.vx * wispScale;
            py += random.vy * wispScale;
            pz += random.vz * wispScale * 0.3;

            positions[i * 3] = px;
            positions[i * 3 + 1] = py;
            positions[i * 3 + 2] = pz;

            // Color gradient: hot blueish-white at base, fading to invisible
            const fadeExp = Math.exp(-t * 3.5);
            const intensity = fadeExp * (1 - t * t * 0.3);

            colors[i * 3] = intensity * (0.8 + t * 0.4);
            colors[i * 3 + 1] = intensity * (0.7 - t * 0.4);
            colors[i * 3 + 2] = intensity * (1.0 - t * 0.6);
          }

          enginePlume.geometry.attributes.position.needsUpdate = true;
          enginePlume.geometry.attributes.color.needsUpdate = true;
        } else {
          enginePlume.material.opacity = 0;
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
      const geom = new THREE.SphereGeometry(1, 8, 8);
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
   * Spawn a PD laser beam from a pd_fired event
   * @param {Object} event - The pd_fired event
   * @param {Object} interpolatedState - Current interpolated state with ship/projectile positions
   */
  spawnPDBeam(event, interpolatedState) {
    const shipId = event.ship_id;
    const targetId = event.data.target_id;
    const targetType = event.data.target_type;

    // Get shooter ship position
    const ship = this.ships.get(shipId);
    if (!ship) return;

    // Get target position
    let targetPos = null;
    if (targetType === 'slug') {
      // Target is a projectile - find it in the projectiles map
      const projectile = this.projectiles.get(targetId);
      if (projectile) {
        targetPos = projectile.position.clone();
      }
    } else if (targetType === 'ship' || targetType === 'torpedo') {
      // Target is another ship
      const targetShip = this.ships.get(targetId);
      if (targetShip) {
        targetPos = targetShip.position.clone();
      }
    }

    if (!targetPos) return;

    // Create beam geometry
    const beamGeom = new THREE.BufferGeometry();
    const positions = new Float32Array(2 * 3);
    positions[0] = ship.position.x;
    positions[1] = ship.position.y;
    positions[2] = ship.position.z;
    positions[3] = targetPos.x;
    positions[4] = targetPos.y;
    positions[5] = targetPos.z;
    beamGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Create glowing beam material - red/orange for PD lasers
    const beamMaterial = new THREE.LineBasicMaterial({
      color: 0xff4422,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending
    });

    const beam = new THREE.Line(beamGeom, beamMaterial);

    this.scene.add(beam);
    this.pdcBeams.push({
      line: beam,
      spawnTime: event.timestamp,
      duration: this.PD_BEAM_DURATION
    });
  }

  /**
   * Update PD beams - fade out old beams and remove expired ones
   * @param {number} currentTime - Current playback time in seconds
   */
  updatePDBeams(currentTime) {
    this.currentTime = currentTime;

    // Update and remove expired beams
    const beamsToRemove = [];

    for (let i = 0; i < this.pdcBeams.length; i++) {
      const beam = this.pdcBeams[i];
      const age = currentTime - beam.spawnTime;

      if (age > beam.duration) {
        // Beam expired - mark for removal
        beamsToRemove.push(i);
        this.scene.remove(beam.line);
        beam.line.geometry.dispose();
        beam.line.material.dispose();
      } else if (age >= 0) {
        // Fade out beam based on age
        const fadeProgress = age / beam.duration;
        beam.line.material.opacity = 1.0 - fadeProgress;
        beam.line.visible = true;
      } else {
        // Beam is in the future (can happen when scrubbing backwards)
        beam.line.visible = false;
      }
    }

    // Remove expired beams (in reverse order to maintain indices)
    for (let i = beamsToRemove.length - 1; i >= 0; i--) {
      this.pdcBeams.splice(beamsToRemove[i], 1);
    }
  }

  /**
   * Clear all PD beams (used when loading new recording or scrubbing)
   */
  clearPDBeams() {
    for (const beam of this.pdcBeams) {
      this.scene.remove(beam.line);
      beam.line.geometry.dispose();
      beam.line.material.dispose();
    }
    this.pdcBeams = [];
  }

  /**
   * Spawn a hit effect at an impact position
   * Enhanced with dual shockwave rings, point light, and dense particle burst
   * @param {Array} position - [x, y, z] in meters
   * @param {number} energyGj - Impact energy for scaling effect size
   * @param {number} currentTime - Current playback time
   */
  spawnHitEffect(position, energyGj, currentTime) {
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
    const flashGeom = new THREE.SphereGeometry(2 * scale, 24, 24);
    const flashMat = new THREE.MeshBasicMaterial({
      color: 0xffeeaa,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending
    });
    const flash = new THREE.Mesh(flashGeom, flashMat);
    group.add(flash);

    // Flash light - enhances bloom and illuminates nearby
    const flashLight = new THREE.PointLight(0xffaa00, 0, scale * 10);
    group.add(flashLight);

    // Primary expanding ring - orange shockwave
    const ringGeom = new THREE.RingGeometry(1 * scale, 2 * scale, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff6600,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.lookAt(this.camera.position);
    group.add(ring);

    // Secondary ring - inner blue plasma ring for contrast
    const innerRingGeom = new THREE.RingGeometry(0.5 * scale, 1.5 * scale, 48);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: 0x00aaff,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
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
      size: scale * 1.5,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
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
      duration: 1.2, // Slightly longer for more spectacle
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
        effect.flashLight.dispose();
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
        effect.flash.scale.setScalar(1.0 + easedProgress * 3);
        effect.flashLight.intensity = Math.pow(1 - progress, 3) * 20;

        // Expand and fade primary ring
        effect.ring.scale.setScalar(1.0 + easedProgress * 6);
        effect.ring.material.opacity = 0.8 * (1 - easedProgress);
        effect.ring.lookAt(this.camera.position);

        // Expand and fade inner ring - faster expansion
        effect.innerRing.scale.setScalar(1.0 + easedProgress * 8);
        effect.innerRing.material.opacity = Math.max(0, 0.6 * (1 - easedProgress * 1.2));
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
   * Clear all hit effects
   */
  clearHitEffects() {
    for (const effect of this.hitEffects) {
      this.scene.remove(effect.group);
      effect.flash.geometry.dispose();
      effect.flash.material.dispose();
      effect.flashLight.dispose();
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
  createDestructionEffect(position, shipType, currentTime) {
    // Scale based on ship type
    const scaleFactors = {
      corvette: 0.6, frigate: 0.8, destroyer: 1.0,
      cruiser: 1.3, battlecruiser: 1.6, battleship: 2.0,
      dreadnought: 2.5, dreadnought_siege: 2.5
    };
    const baseScale = (scaleFactors[shipType] || 1.0) * 8;
    const scale = baseScale * 1.5; // Increase overall scale for more spectacle

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
        blending: THREE.AdditiveBlending
      });
      const flash = new THREE.Mesh(flashGeom, flashMat);
      flash.position.copy(offset);
      group.add(flash);

      // Add point light for each explosion to enhance bloom
      const expLight = new THREE.PointLight(0xffaa00, 0, scale * 5);
      expLight.position.copy(offset);
      group.add(expLight);

      explosions.push({
        flash,
        light: expLight,
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
        blending: THREE.AdditiveBlending
      });
      const flash = new THREE.Mesh(flashGeom, flashMat);
      flash.position.copy(offset);
      group.add(flash);

      // Dimmer lights for secondary
      const expLight = new THREE.PointLight(0xff6600, 0, scale * 3);
      expLight.position.copy(offset);
      group.add(expLight);

      secondaryExplosions.push({
        flash,
        light: expLight,
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
      blending: THREE.AdditiveBlending
    });
    const plasmaSphere = new THREE.Mesh(plasmaGeom, plasmaMat);
    group.add(plasmaSphere);

    // Central detonation light
    const coreLight = new THREE.PointLight(0xffffff, 0, scale * 20);
    group.add(coreLight);

    // BLINDING FLASH - massive white sphere that whites out everything
    const flashGeom = new THREE.SphereGeometry(scale * 50, 32, 32);
    const blindingFlashMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide // Render inside so camera sees it from within
    });
    const blindingFlash = new THREE.Mesh(flashGeom, blindingFlashMat);
    group.add(blindingFlash);

    // Secondary flash light - extremely intense
    const flashLight = new THREE.PointLight(0xffffff, 0, scale * 100);
    group.add(flashLight);

    // Shockwave ring - expanding outward
    const shockGeom = new THREE.RingGeometry(scale * 0.5, scale * 1.0, 64);
    const shockMat = new THREE.MeshBasicMaterial({
      color: 0xff8800,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    const shockwave = new THREE.Mesh(shockGeom, shockMat);
    shockwave.lookAt(this.camera.position);
    group.add(shockwave);

    // Phase 3: Debris and plasma cloud - 100000 particles for massive cloud
    const debrisCount = 100000;
    const debrisGeom = new THREE.BufferGeometry();
    const debrisPositions = new Float32Array(debrisCount * 3);
    const debrisColors = new Float32Array(debrisCount * 3);
    const debrisVelocities = [];

    for (let i = 0; i < debrisCount; i++) {
      // Start at random position within ship volume
      debrisPositions[i * 3] = (Math.random() - 0.5) * scale;
      debrisPositions[i * 3 + 1] = (Math.random() - 0.5) * scale;
      debrisPositions[i * 3 + 2] = (Math.random() - 0.5) * scale;

      // Random velocity - higher speeds, more variation
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const speed = (10 + Math.random() * 40) * scale * 0.4;
      debrisVelocities.push({
        x: speed * Math.sin(phi) * Math.cos(theta) + (Math.random() - 0.5) * 10,
        y: speed * Math.sin(phi) * Math.sin(theta) + (Math.random() - 0.5) * 10,
        z: speed * Math.cos(phi) + (Math.random() - 0.5) * 10,
        rotX: (Math.random() - 0.5) * 0.1,
        rotY: (Math.random() - 0.5) * 0.1,
        rotZ: (Math.random() - 0.5) * 0.1
      });

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
    debrisGeom.setAttribute('color', new THREE.BufferAttribute(debrisColors, 3));

    const debrisMat = new THREE.PointsMaterial({
      size: scale * 0.1,
      transparent: true,
      opacity: 0,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    });

    const debris = new THREE.Points(debrisGeom, debrisMat);
    group.add(debris);

    // Lingering plasma cloud - semi-transparent sphere that fades slowly
    const cloudGeom = new THREE.SphereGeometry(scale * 2, 32, 32);
    const cloudMat = new THREE.MeshBasicMaterial({
      color: 0xff5500,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending
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
      debrisVelocities,
      spawnTime: currentTime,
      explosionPhase: 4.0,
      secondaryPhase: 6.0,
      flashDelay: 2.5,      // Blinding flash at reactor breach
      flashDuration: 0.8,   // Very quick but intense
      plasmaDelay: 2.0,
      debrisDelay: 3.0,
      cloudDelay: 4.0,
      totalDuration: 20.0,
      scale
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

      if (age > effect.totalDuration) {
        // Effect expired - clean up all components
        effectsToRemove.push(i);
        this.scene.remove(effect.group);
        for (const exp of effect.explosions) {
          exp.flash.geometry.dispose();
          exp.flash.material.dispose();
          exp.light.dispose();
        }
        for (const sec of effect.secondaryExplosions) {
          sec.flash.geometry.dispose();
          sec.flash.material.dispose();
          sec.light.dispose();
        }
        effect.plasmaSphere.geometry.dispose();
        effect.plasmaSphere.material.dispose();
        effect.coreLight.dispose();
        effect.blindingFlash.geometry.dispose();
        effect.blindingFlash.material.dispose();
        effect.flashLight.dispose();
        effect.shockwave.geometry.dispose();
        effect.shockwave.material.dispose();
        effect.debris.geometry.dispose();
        effect.debris.material.dispose();
        effect.plasmaCloud.geometry.dispose();
        effect.plasmaCloud.material.dispose();
        // Reset bloom to default
        if (this.bloomPass) {
          this.bloomPass.strength = 0.8;
        }
      } else if (age >= 0) {
        effect.group.visible = true;

        // Phase 1: Update primary explosions
        for (const exp of effect.explosions) {
          const expAge = age - exp.delay;
          if (expAge >= 0 && expAge < exp.duration) {
            const progress = expAge / exp.duration;
            exp.flash.material.opacity = Math.sin(progress * Math.PI) * 1.0;
            exp.flash.scale.setScalar(1.0 + progress * 5);
            exp.light.intensity = Math.sin(progress * Math.PI) * 20;
          } else if (expAge >= exp.duration) {
            exp.flash.material.opacity = 0;
            exp.light.intensity = 0;
          }
        }

        // Phase 1.5: Update secondary explosions
        for (const sec of effect.secondaryExplosions) {
          const secAge = age - sec.delay;
          if (secAge >= 0 && secAge < sec.duration) {
            const progress = secAge / sec.duration;
            sec.flash.material.opacity = Math.sin(progress * Math.PI) * 0.8;
            sec.flash.scale.setScalar(1.0 + progress * 4);
            sec.light.intensity = Math.sin(progress * Math.PI) * 15;
          } else if (secAge >= sec.duration) {
            sec.flash.material.opacity = 0;
            sec.light.intensity = 0;
          }
        }

        // Phase 2: Main plasma sphere detonation
        const plasmaAge = age - effect.plasmaDelay;
        if (plasmaAge >= 0) {
          const plasmaDuration = 5.0;
          const plasmaProgress = Math.min(1, plasmaAge / plasmaDuration);
          effect.plasmaSphere.material.opacity = Math.pow(1 - plasmaProgress, 0.5) * 0.8;
          effect.plasmaSphere.scale.setScalar(1.0 + plasmaProgress * 10);
          effect.coreLight.intensity = Math.pow(1 - plasmaProgress, 2) * 50;

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
        const flashAge = age - effect.flashDelay;
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
          effect.flashLight.intensity = flashIntensity * 200;
          // Also boost bloom temporarily if available
          if (this.bloomPass) {
            this.bloomPass.strength = 0.8 + flashIntensity * 3.0;
          }
        } else if (flashAge >= effect.flashDuration) {
          effect.blindingFlash.material.opacity = 0;
          effect.flashLight.intensity = 0;
          // Reset bloom
          if (this.bloomPass) {
            this.bloomPass.strength = 0.8;
          }
        }

        // Phase 3: Debris ejection
        if (age > effect.debrisDelay) {
          const debrisAge = age - effect.debrisDelay;
          const debrisDuration = effect.totalDuration - effect.debrisDelay;
          const debrisProgress = debrisAge / debrisDuration;

          // Fade in/out
          if (debrisProgress < 0.05) {
            effect.debris.material.opacity = debrisProgress * 20;
          } else {
            effect.debris.material.opacity = Math.max(0, 1.0 - (debrisProgress - 0.05) / 0.95);
          }

          // Update positions with drag and tumbling
          const positions = effect.debris.geometry.attributes.position.array;
          const dt = this.delta;
          for (let j = 0; j < effect.debrisVelocities.length; j++) {
            const vel = effect.debrisVelocities[j];
            const drag = 1.0 / (1.0 + debrisAge * 0.05);
            positions[j * 3] += vel.x * dt * drag;
            positions[j * 3 + 1] += vel.y * dt * drag;
            positions[j * 3 + 2] += vel.z * dt * drag;

            // Simulate tumbling
            positions[j * 3] += Math.sin(debrisAge + j) * vel.rotX * dt;
            positions[j * 3 + 1] += Math.cos(debrisAge + j) * vel.rotY * dt;
            positions[j * 3 + 2] += Math.sin(debrisAge + j * 0.5) * vel.rotZ * dt;
          }
          effect.debris.geometry.attributes.position.needsUpdate = true;

          // Gradually shrink and cool particles
          effect.debris.material.size = effect.scale * 0.1 * (1.0 - debrisProgress * 0.8);
        }

        // Phase 4: Lingering plasma cloud
        const cloudAge = age - effect.cloudDelay;
        if (cloudAge >= 0) {
          const cloudDuration = 12.0;
          const cloudProgress = Math.min(1, cloudAge / cloudDuration);
          effect.plasmaCloud.material.opacity = Math.pow(1 - cloudProgress, 1.5) * 0.4;
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
        exp.light.dispose();
      }
      // Clean up secondary explosions
      for (const sec of effect.secondaryExplosions) {
        sec.flash.geometry.dispose();
        sec.flash.material.dispose();
        sec.light.dispose();
      }
      // Clean up plasma sphere and core light
      effect.plasmaSphere.geometry.dispose();
      effect.plasmaSphere.material.dispose();
      effect.coreLight.dispose();
      // Clean up blinding flash
      effect.blindingFlash.geometry.dispose();
      effect.blindingFlash.material.dispose();
      effect.flashLight.dispose();
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
      this.bloomPass.strength = 0.8;
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
    this.controls.update();
    // Use composer for bloom post-processing
    this.composer.render();
  }
}
