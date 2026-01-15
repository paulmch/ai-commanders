/**
 * AI Commanders - Tactical Battle Visualizer
 * Main entry point - orchestrates all components
 */

import { BattleLoader } from './BattleLoader.js';
import { Interpolator } from './Interpolator.js';
import { SceneManager } from './SceneManager.js';
import { TimeController } from './TimeController.js';
import { CameraController } from './CameraController.js';
import { generateShipSVG, getShipClassName } from './shipSilhouettes.js';

class BattleVisualizer {
  constructor() {
    // Core components
    this.loader = null;
    this.scene = null;
    this.timeController = null;
    this.cameraController = null;

    // State
    this.isInitialized = false;
    this.selectedShipId = null;
    this.lastTime = 0;
    this.lastPDEventTime = 0; // Track PD events separately
    this.lastHitEventTime = 0; // Track hit events for visual effects
    this.lastDestructionTime = 0; // Track destruction events for scrubbing
    this.currentState = null;

    // DOM elements
    this.elements = {};

    // Animation
    this.animationId = null;
  }

  /**
   * Initialize the visualizer
   */
  async init() {
    this.cacheElements();
    this.setupEventListeners();
    this.hideLoading();
    this.showFileSelector();
  }

  /**
   * Cache DOM elements for performance
   */
  cacheElements() {
    this.elements = {
      // Canvas
      canvas: document.getElementById('canvas'),

      // Overlays
      loading: document.getElementById('loading'),
      loadingStatus: document.getElementById('loadingStatus'),
      loadingProgress: document.getElementById('loadingProgress'),
      fileSelector: document.getElementById('fileSelector'),

      // Header
      battleName: document.getElementById('battleName'),
      missionTime: document.getElementById('missionTime'),
      alphaCount: document.getElementById('alphaCount'),
      betaCount: document.getElementById('betaCount'),
      alphaBar: document.getElementById('alphaBar'),
      betaBar: document.getElementById('betaBar'),

      // Ship registry
      alphaShips: document.getElementById('alphaShips'),
      betaShips: document.getElementById('betaShips'),

      // Ship telemetry
      shipTelemetry: document.getElementById('shipTelemetry'),
      shipName: document.getElementById('shipName'),
      shipClass: document.getElementById('shipClass'),
      shipDesignation: document.getElementById('shipDesignation'),
      hullBar: document.getElementById('hullBar'),
      hullValue: document.getElementById('hullValue'),
      thrustBar: document.getElementById('thrustBar'),
      thrustValue: document.getElementById('thrustValue'),
      targetValue: document.getElementById('targetValue'),
      maneuverValue: document.getElementById('maneuverValue'),
      positionValue: document.getElementById('positionValue'),
      velocityValue: document.getElementById('velocityValue'),
      closeTelemetry: document.getElementById('closeTelemetry'),

      // Armor elements
      armorNose: document.getElementById('armorNose'),
      armorLateral: document.getElementById('armorLateral'),
      armorTail: document.getElementById('armorTail'),
      armorNoseBar: document.getElementById('armorNoseBar'),
      armorLateralBar: document.getElementById('armorLateralBar'),
      armorTailBar: document.getElementById('armorTailBar'),
      armorNoseValue: document.getElementById('armorNoseValue'),
      armorLateralValue: document.getElementById('armorLateralValue'),
      armorTailValue: document.getElementById('armorTailValue'),

      // Module grid
      moduleGrid: document.getElementById('moduleGrid'),

      // Camera buttons
      followBtn: document.getElementById('followBtn'),
      orbitBtn: document.getElementById('orbitBtn'),
      freeBtn: document.getElementById('freeBtn'),

      // Playback
      playPauseBtn: document.getElementById('playPauseBtn'),
      timeline: document.getElementById('timeline'),
      timelineProgress: document.getElementById('timelineProgress'),
      timelineHandle: document.getElementById('timelineHandle'),
      timeInput: document.getElementById('timeInput'),
      totalTime: document.getElementById('totalTime'),

      // Event log
      eventLog: document.getElementById('eventLog'),
      eventList: document.getElementById('eventList'),
      collapseEventLog: document.getElementById('collapseEventLog'),

      // File selector
      selectFileBtn: document.getElementById('selectFileBtn'),
      fileInput: document.getElementById('fileInput'),
      urlInput: document.getElementById('urlInput'),
      loadUrlBtn: document.getElementById('loadUrlBtn'),
      loadNewBtn: document.getElementById('loadNewBtn'),
    };
  }

  /**
   * Setup all event listeners
   */
  setupEventListeners() {
    // File selection
    this.elements.selectFileBtn.addEventListener('click', () => {
      this.elements.fileInput.click();
    });

    this.elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.loadFromFile(e.target.files[0]);
      }
    });

    this.elements.loadUrlBtn.addEventListener('click', () => {
      const url = this.elements.urlInput.value.trim();
      if (url) {
        this.loadFromUrl(url);
      }
    });

    this.elements.urlInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.elements.loadUrlBtn.click();
      }
    });

    // Load new recording button
    this.elements.loadNewBtn?.addEventListener('click', () => {
      this.resetAndShowFileSelector();
    });

    // Playback controls
    this.elements.playPauseBtn.addEventListener('click', () => {
      this.togglePlayback();
    });

    this.elements.timeline.addEventListener('input', (e) => {
      if (this.timeController) {
        const percent = e.target.value / 1000;
        this.timeController.seekPercent(percent);
      }
    });

    // Speed buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const speed = parseFloat(e.target.dataset.speed);
        this.setPlaybackSpeed(speed);
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
      });
    });

    // Camera mode buttons
    this.elements.followBtn.addEventListener('click', () => this.setCameraMode('follow'));
    this.elements.orbitBtn.addEventListener('click', () => this.setCameraMode('orbit'));
    this.elements.freeBtn.addEventListener('click', () => this.setCameraMode('free'));

    // Close telemetry panel
    this.elements.closeTelemetry.addEventListener('click', () => {
      this.deselectShip();
    });

    // Collapse event log
    this.elements.collapseEventLog?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.elements.eventLog.classList.toggle('collapsed');
      this.elements.collapseEventLog.textContent =
        this.elements.eventLog.classList.contains('collapsed') ? '+' : '−';
    });

    // Time input - seek to entered timestamp
    this.elements.timeInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.seekToInputTime();
        this.elements.timeInput.blur();
      } else if (e.key === 'Escape') {
        // Reset to current time on Escape
        this.elements.timeInput.value = this.formatTime(this.timeController?.currentTime || 0);
        this.elements.timeInput.blur();
      }
    });

    this.elements.timeInput.addEventListener('blur', () => {
      this.seekToInputTime();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Don't handle shortcuts when typing in an input
      if (e.target.tagName === 'INPUT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlayback();
      } else if (e.code === 'ArrowLeft') {
        this.timeController?.seek(this.timeController.currentTime - 5);
      } else if (e.code === 'ArrowRight') {
        this.timeController?.seek(this.timeController.currentTime + 5);
      }
    });

    // Window resize
    window.addEventListener('resize', () => {
      if (this.scene) {
        this.scene.onResize();
      }
    });
  }

  /**
   * Load battle from file
   */
  async loadFromFile(file) {
    this.hideFileSelector();
    this.showLoading();
    this.setLoadingStatus('Reading file...');

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      await this.initializeBattle(data, file.name);
    } catch (error) {
      console.error('Failed to load file:', error);
      this.setLoadingStatus(`Error: ${error.message}`);
      setTimeout(() => {
        this.hideLoading();
        this.showFileSelector();
      }, 2000);
    }
  }

  /**
   * Load battle from URL
   */
  async loadFromUrl(url) {
    this.hideFileSelector();
    this.showLoading();
    this.setLoadingStatus('Fetching recording...');

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      this.setLoadingStatus('Parsing data...');
      const data = await response.json();
      await this.initializeBattle(data, url.split('/').pop());
    } catch (error) {
      console.error('Failed to load URL:', error);
      this.setLoadingStatus(`Error: ${error.message}`);
      setTimeout(() => {
        this.hideLoading();
        this.showFileSelector();
      }, 2000);
    }
  }

  /**
   * Initialize battle with loaded data
   */
  async initializeBattle(data, filename) {
    this.setLoadingStatus('Initializing...');
    this.setLoadingProgress(20);

    // Parse battle data
    this.loader = new BattleLoader();
    this.loader.parse(data);
    this.setLoadingProgress(40);

    // Initialize scene
    this.setLoadingStatus('Creating tactical display...');
    this.scene = new SceneManager(this.elements.canvas);
    this.setLoadingProgress(60);

    // Initialize time controller
    this.timeController = new TimeController(this.loader.duration);
    this.timeController.onTimeChange((time, duration) => {
      this.updateTimeDisplay(time, duration);
    });
    this.setLoadingProgress(80);

    // Initialize camera controller
    this.cameraController = new CameraController(
      this.scene.camera,
      this.scene.controls,
      this.scene
    );

    // Create ships
    this.setLoadingStatus('Spawning fleet assets...');
    for (const [shipId, shipInfo] of Object.entries(this.loader.ships)) {
      this.scene.createShip(shipId, shipInfo.faction, shipInfo.type);
    }
    this.setLoadingProgress(100);

    // Update UI
    this.updateBattleInfo();
    this.populateShipLists();
    this.updateFleetStatus();

    // Hide loading and start
    this.isInitialized = true;
    await this.delay(300);
    this.hideLoading();

    // Start animation loop
    this.startAnimationLoop();

    // Auto-play
    this.timeController.play();
    this.updatePlayPauseButton();
  }

  /**
   * Main animation loop
   */
  startAnimationLoop() {
    let lastTimestamp = 0;

    const animate = (timestamp) => {
      const deltaTime = (timestamp - lastTimestamp) / 1000;
      lastTimestamp = timestamp;

      if (this.isInitialized && deltaTime < 1) {
        // Update time
        this.timeController.update(deltaTime);

        // Get interpolated state (pass loader for projectile extrapolation to impact)
        const frameData = this.loader.getFrameAt(this.timeController.currentTime);
        this.currentState = Interpolator.getInterpolatedState(frameData, this.loader);

        // Update ships
        for (const [shipId, state] of Object.entries(this.currentState.ships)) {
          this.scene.updateShip(shipId, state);
        }

        // Update projectiles
        const activeProjectileIds = new Set();
        for (const proj of this.currentState.projectiles) {
          this.scene.updateProjectile(proj);
          activeProjectileIds.add(proj.id);
        }
        this.scene.cleanupProjectiles(activeProjectileIds);

        // Handle PD beam visualization
        const currentPlaybackTime = this.timeController.currentTime;

        // If scrubbing backwards, clear existing beams and hit effects
        if (currentPlaybackTime < this.lastPDEventTime) {
          this.scene.clearPDBeams();
          this.lastPDEventTime = currentPlaybackTime;
        }
        if (currentPlaybackTime < this.lastHitEventTime) {
          this.scene.clearHitEffects();
          this.lastHitEventTime = currentPlaybackTime;
        }
        if (currentPlaybackTime < this.lastDestructionTime) {
          this.scene.clearDestructionEffects();
          this.lastDestructionTime = currentPlaybackTime;
        }

        // Spawn new PD beams for events since last frame
        const pdEvents = this.loader.getPDEventsInRange(this.lastPDEventTime, currentPlaybackTime);
        for (const event of pdEvents) {
          this.scene.spawnPDBeam(event, this.currentState);
        }
        this.lastPDEventTime = currentPlaybackTime;

        // Update PD beam fading
        this.scene.updatePDBeams(currentPlaybackTime);

        // Spawn hit effects when projectiles visually reach their targets
        // (based on extrapolated travel time, not recording event timestamps)
        const visualImpacts = this.loader.getVisualImpacts(this.lastHitEventTime, currentPlaybackTime);
        for (const impact of visualImpacts) {
          this.scene.spawnHitEffect(
            impact.impact_position,
            impact.kinetic_energy_gj,
            currentPlaybackTime
          );
        }
        this.lastHitEventTime = currentPlaybackTime;

        // Update hit effect animations
        this.scene.updateHitEffects(currentPlaybackTime);

        // Update destruction effect animations
        this.scene.updateDestructionEffects(currentPlaybackTime);
        if (this.scene.destructionEffects.length > 0 || this.scene.destroyedShips.size > 0) {
          this.lastDestructionTime = currentPlaybackTime;
        }

        // Update camera
        this.cameraController.update(this.currentState.ships);

        // Update selected ship telemetry
        if (this.selectedShipId && this.currentState.ships[this.selectedShipId]) {
          this.updateShipTelemetry(this.currentState.ships[this.selectedShipId]);
        }

        // Update fleet status
        this.updateFleetStatus();

        // Process events
        this.processEvents();
      }

      // Render
      this.scene.render();

      this.animationId = requestAnimationFrame(animate);
    };

    this.animationId = requestAnimationFrame(animate);
  }

  /**
   * Update battle info in header
   */
  updateBattleInfo() {
    this.elements.battleName.textContent = this.loader.metadata.battleName || 'FLEET ENGAGEMENT';
    this.elements.totalTime.textContent = this.formatTime(this.loader.duration);
  }

  /**
   * Populate ship lists in registry panel
   */
  populateShipLists() {
    this.elements.alphaShips.innerHTML = '';
    this.elements.betaShips.innerHTML = '';

    for (const [shipId, shipInfo] of Object.entries(this.loader.ships)) {
      const li = document.createElement('li');
      li.textContent = shipInfo.name || shipId;
      li.dataset.shipId = shipId;
      li.addEventListener('click', () => this.selectShip(shipId));

      if (shipInfo.faction === 'alpha') {
        this.elements.alphaShips.appendChild(li);
      } else {
        this.elements.betaShips.appendChild(li);
      }
    }
  }

  /**
   * Update fleet status (alive counts)
   */
  updateFleetStatus() {
    if (!this.currentState) return;

    let alphaAlive = 0, alphaTotal = 0;
    let betaAlive = 0, betaTotal = 0;

    for (const [shipId, state] of Object.entries(this.currentState.ships)) {
      const faction = shipId.startsWith('alpha') ? 'alpha' : 'beta';
      if (faction === 'alpha') {
        alphaTotal++;
        if (!state.destroyed) alphaAlive++;
      } else {
        betaTotal++;
        if (!state.destroyed) betaAlive++;
      }
    }

    this.elements.alphaCount.textContent = `${alphaAlive}/${alphaTotal}`;
    this.elements.betaCount.textContent = `${betaAlive}/${betaTotal}`;
    this.elements.alphaBar.style.width = `${(alphaAlive / alphaTotal) * 100}%`;
    this.elements.betaBar.style.width = `${(betaAlive / betaTotal) * 100}%`;

    // Update ship list destroyed states
    document.querySelectorAll('.ship-list li').forEach(li => {
      const shipId = li.dataset.shipId;
      const state = this.currentState.ships[shipId];
      if (state?.destroyed) {
        li.classList.add('destroyed');
      } else {
        li.classList.remove('destroyed');
      }
    });
  }

  /**
   * Select a ship
   */
  selectShip(shipId) {
    // Update selection state
    this.selectedShipId = shipId;

    // Update ship list UI
    document.querySelectorAll('.ship-list li').forEach(li => {
      li.classList.toggle('selected', li.dataset.shipId === shipId);
    });

    // Show telemetry panel
    this.elements.shipTelemetry.classList.remove('hidden');

    // Update telemetry header
    const shipInfo = this.loader.ships[shipId];
    this.elements.shipName.textContent = shipInfo?.name || shipId;
    this.elements.shipClass.textContent = getShipClassName(shipInfo?.type || 'destroyer');
    this.elements.shipDesignation.textContent = shipId.toUpperCase().replace('_', '-');

    // Update ship silhouette based on ship type
    this.updateShipSilhouette(shipInfo?.type || 'destroyer');

    // Set faction color
    const faction = shipId.startsWith('alpha') ? 'alpha' : 'beta';
    this.elements.shipName.style.color = faction === 'alpha'
      ? 'var(--alpha-primary)'
      : 'var(--beta-primary)';

    // Focus camera on ship
    if (this.currentState?.ships[shipId]) {
      this.cameraController.focusOnShip(shipId, this.currentState.ships);
    }
  }

  /**
   * Deselect current ship
   */
  deselectShip() {
    this.selectedShipId = null;
    this.elements.shipTelemetry.classList.add('hidden');

    document.querySelectorAll('.ship-list li').forEach(li => {
      li.classList.remove('selected');
    });

    this.setCameraMode('free');
  }

  /**
   * Update the ship silhouette SVG based on ship type
   */
  updateShipSilhouette(shipType) {
    const silhouetteSvg = document.querySelector('.ship-outline');
    if (silhouetteSvg) {
      silhouetteSvg.innerHTML = generateShipSVG(shipType);

      // Re-cache the armor face elements after SVG update
      this.elements.armorNose = document.getElementById('armorNose');
      this.elements.armorLateral = document.getElementById('armorLateral');
      this.elements.armorTail = document.getElementById('armorTail');
    }
  }

  /**
   * Update ship telemetry panel
   */
  updateShipTelemetry(state) {
    if (!state) {
      // Ship destroyed or state unavailable
      this.elements.hullBar.style.width = '0%';
      this.elements.hullValue.textContent = 'DESTROYED';
      this.elements.thrustBar.style.width = '0%';
      this.elements.thrustValue.textContent = '-';
      this.elements.maneuverValue.textContent = '-';
      this.elements.positionValue.textContent = '-';
      this.elements.velocityValue.textContent = '-';
      if (this.elements.targetValue) {
        this.elements.targetValue.textContent = '-';
      }
      return;
    }

    const hull = state.hull || 100;
    const thrust = (state.thrust || 0) * 100;

    this.elements.hullBar.style.width = `${hull}%`;
    this.elements.hullValue.textContent = state.destroyed ? 'DESTROYED' : `${hull.toFixed(0)}%`;

    this.elements.thrustBar.style.width = `${thrust}%`;
    this.elements.thrustValue.textContent = `${thrust.toFixed(0)}%`;

    this.elements.maneuverValue.textContent = state.maneuver || 'MAINTAIN';

    // Position in km
    const pos = state.position;
    if (pos && pos.length >= 3) {
      this.elements.positionValue.textContent = `(${(pos[0]/1000).toFixed(0)}, ${(pos[1]/1000).toFixed(0)}, ${(pos[2]/1000).toFixed(0)}) km`;
    } else {
      this.elements.positionValue.textContent = '-';
    }

    // Velocity magnitude in km/s
    const vel = state.velocity;
    if (vel && vel.length >= 3) {
      const speed = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]) / 1000;
      this.elements.velocityValue.textContent = `${speed.toFixed(1)} km/s`;
    } else {
      this.elements.velocityValue.textContent = '-';
    }

    // Target info
    if (this.elements.targetValue && this.loader) {
      const targetInfo = this.loader.getShipTargetAt(this.selectedShipId, this.timeController?.currentTime || 0);
      if (targetInfo) {
        this.elements.targetValue.textContent = targetInfo.target_name || targetInfo.target_id || '-';
      } else {
        this.elements.targetValue.textContent = '-';
      }
    }

    // Update armor and module status
    if (this.selectedShipId && this.loader) {
      this.updateArmorDisplay(this.selectedShipId);
      this.updateModuleDisplay(this.selectedShipId);
    }
  }

  /**
   * Update armor schematic display
   */
  updateArmorDisplay(shipId) {
    const currentTime = this.timeController?.currentTime || 0;
    const damageState = this.loader.getShipDamageAt(shipId, currentTime);

    if (!damageState) return;

    const { armor, initialArmor } = damageState;

    // Update each armor section
    const sections = ['nose', 'lateral', 'tail'];
    const elements = {
      nose: { bar: this.elements.armorNoseBar, value: this.elements.armorNoseValue, face: this.elements.armorNose },
      lateral: { bar: this.elements.armorLateralBar, value: this.elements.armorLateralValue, face: this.elements.armorLateral },
      tail: { bar: this.elements.armorTailBar, value: this.elements.armorTailValue, face: this.elements.armorTail }
    };

    for (const section of sections) {
      const remaining = armor[section];
      const initial = initialArmor[section] || 100;
      const percent = (remaining / initial) * 100;
      const el = elements[section];

      if (!el.bar || !el.value) continue;

      // Update bar width
      el.bar.style.width = `${percent}%`;

      // Update value text
      el.value.textContent = `${remaining.toFixed(0)}cm`;

      // Get the parent armor-reading element for label color
      const readingEl = el.bar.closest('.armor-reading');

      // Update visual state classes - all green/yellow/red based on damage
      el.bar.classList.remove('damaged', 'critical');
      if (readingEl) readingEl.classList.remove('damaged', 'critical');
      if (el.face) {
        el.face.classList.remove('highlight', 'damaged', 'critical');
        el.face.classList.add('highlight');
      }

      if (percent < 25) {
        el.bar.classList.add('critical');
        if (readingEl) readingEl.classList.add('critical');
        if (el.face) el.face.classList.add('critical');
      } else if (percent < 60) {
        el.bar.classList.add('damaged');
        if (readingEl) readingEl.classList.add('damaged');
        if (el.face) el.face.classList.add('damaged');
      }
    }
  }

  /**
   * Update module status display
   */
  updateModuleDisplay(shipId) {
    const currentTime = this.timeController?.currentTime || 0;
    const damageState = this.loader.getShipDamageAt(shipId, currentTime);

    if (!damageState || !this.elements.moduleGrid) return;

    const { modules } = damageState;

    // Clear existing modules
    this.elements.moduleGrid.innerHTML = '';

    // Module icons based on name
    const getModuleIcon = (name) => {
      const lowerName = name.toLowerCase();
      if (lowerName.includes('weapon') || lowerName.includes('coiler') || lowerName.includes('cannon') || lowerName.includes('spinal')) return '⚔';
      if (lowerName.includes('pd') || lowerName.includes('point defense') || lowerName.includes('laser')) return '◎';
      if (lowerName.includes('engine') || lowerName.includes('drive')) return '◈';
      if (lowerName.includes('reactor')) return '⬡';
      if (lowerName.includes('bridge')) return '◉';
      if (lowerName.includes('sensor')) return '◐';
      if (lowerName.includes('hull')) return '▣';
      if (lowerName.includes('fuel') || lowerName.includes('tank')) return '◯';
      if (lowerName.includes('magazine') || lowerName.includes('ammo')) return '▪';
      if (lowerName.includes('armor')) return '▦';
      return '◇';
    };

    // Sort modules: destroyed first, then damaged, then operational
    const sortedModules = Object.entries(modules).sort((a, b) => {
      const statusOrder = { destroyed: 0, damaged: 1, operational: 2 };
      return (statusOrder[a[1].status] || 2) - (statusOrder[b[1].status] || 2);
    });

    // Add module items
    for (const [key, module] of sortedModules) {
      const icon = getModuleIcon(key);
      const displayName = key.replace(/_/g, ' ').toUpperCase();
      const status = module.status || 'operational';
      const health = module.health ?? 100;

      const item = document.createElement('div');
      item.className = `module-item ${status}`;

      // Show health percentage
      let statusText;
      if (status === 'destroyed') {
        statusText = 'DESTROYED';
      } else if (health < 100) {
        statusText = `${health.toFixed(0)}%`;
      } else {
        statusText = 'OK';
      }

      item.innerHTML = `
        <div class="module-icon">${icon}</div>
        <div class="module-info">
          <span class="module-name">${displayName}</span>
          <span class="module-status">${statusText}</span>
        </div>
      `;

      this.elements.moduleGrid.appendChild(item);
    }

    // If no modules found, show placeholder
    if (this.elements.moduleGrid.children.length === 0) {
      this.elements.moduleGrid.innerHTML = `
        <div class="module-item operational">
          <div class="module-icon">◇</div>
          <div class="module-info">
            <span class="module-name">SYSTEMS</span>
            <span class="module-status">NOMINAL</span>
          </div>
        </div>
      `;
    }
  }

  /**
   * Set camera mode
   */
  setCameraMode(mode) {
    // Update buttons
    this.elements.followBtn.classList.toggle('active', mode === 'follow');
    this.elements.orbitBtn.classList.toggle('active', mode === 'orbit');
    this.elements.freeBtn.classList.toggle('active', mode === 'free');

    // Set camera mode
    this.cameraController.setMode(mode, this.selectedShipId);
  }

  /**
   * Toggle playback
   */
  togglePlayback() {
    if (this.timeController) {
      this.timeController.toggle();
      this.updatePlayPauseButton();
    }
  }

  /**
   * Update play/pause button state
   */
  updatePlayPauseButton() {
    const playIcon = this.elements.playPauseBtn.querySelector('.play');
    const pauseIcon = this.elements.playPauseBtn.querySelector('.pause');

    if (this.timeController?.isPlaying) {
      playIcon.classList.add('hidden');
      pauseIcon.classList.remove('hidden');
    } else {
      playIcon.classList.remove('hidden');
      pauseIcon.classList.add('hidden');
    }
  }

  /**
   * Set playback speed
   */
  setPlaybackSpeed(speed) {
    if (this.timeController) {
      this.timeController.setSpeed(speed);
    }
  }

  /**
   * Update time display
   */
  updateTimeDisplay(time, duration) {
    // Only update input if it's not focused (user might be typing)
    if (document.activeElement !== this.elements.timeInput) {
      this.elements.timeInput.value = this.formatTime(time);
    }
    this.elements.missionTime.textContent = this.formatTimeFull(time);

    const progress = (time / duration) * 100;
    this.elements.timelineProgress.style.width = `${progress}%`;
    this.elements.timelineHandle.style.left = `${progress}%`;
    this.elements.timeline.value = progress * 10;
  }

  /**
   * Process and display events
   */
  processEvents() {
    const currentTime = this.timeController.currentTime;
    const events = this.loader.getEventsInRange(this.lastTime, currentTime);

    for (const event of events) {
      this.addEventToLog(event);
    }

    this.lastTime = currentTime;
  }

  /**
   * Add event to combat log
   */
  addEventToLog(event) {
    const eventTypes = {
      'hit': { class: 'hit', format: (e) => `HIT: ${e.ship_id} hit by ${e.data?.shooter_id}` },
      'module_destroyed': { class: 'destroyed', format: (e) => `DESTROYED: ${e.ship_id} lost ${e.data?.module_name}` },
      'shot_fired': { class: 'shot', format: (e) => `FIRE: ${e.ship_id} targeting ${e.data?.target_id}` },
    };

    const config = eventTypes[event.event_type];
    if (!config) return;

    const item = document.createElement('div');
    item.className = `event-item ${config.class}`;
    item.innerHTML = `<span class="event-time">${this.formatTime(event.timestamp)}</span>${config.format(event)}`;

    this.elements.eventList.insertBefore(item, this.elements.eventList.firstChild);

    // Limit log size
    while (this.elements.eventList.children.length > 50) {
      this.elements.eventList.removeChild(this.elements.eventList.lastChild);
    }
  }

  /**
   * Format time as MM:SS
   */
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Format time as HH:MM:SS
   */
  formatTimeFull(seconds) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Parse time string to seconds
   * Accepts: "MM:SS", "HH:MM:SS", or raw seconds (e.g., "125")
   */
  parseTime(timeStr) {
    if (!timeStr || typeof timeStr !== 'string') return null;

    const trimmed = timeStr.trim();

    // Try parsing as raw number (seconds)
    if (/^\d+(\.\d+)?$/.test(trimmed)) {
      return parseFloat(trimmed);
    }

    // Try parsing as MM:SS or HH:MM:SS
    const parts = trimmed.split(':');
    if (parts.length === 2) {
      // MM:SS
      const mins = parseInt(parts[0], 10);
      const secs = parseFloat(parts[1]);
      if (!isNaN(mins) && !isNaN(secs)) {
        return mins * 60 + secs;
      }
    } else if (parts.length === 3) {
      // HH:MM:SS
      const hours = parseInt(parts[0], 10);
      const mins = parseInt(parts[1], 10);
      const secs = parseFloat(parts[2]);
      if (!isNaN(hours) && !isNaN(mins) && !isNaN(secs)) {
        return hours * 3600 + mins * 60 + secs;
      }
    }

    return null;
  }

  /**
   * Seek to the time entered in the time input field
   */
  seekToInputTime() {
    if (!this.timeController) return;

    const inputValue = this.elements.timeInput.value;
    const targetTime = this.parseTime(inputValue);

    if (targetTime !== null) {
      // Clamp to valid range
      const clampedTime = Math.max(0, Math.min(targetTime, this.timeController.duration));
      this.timeController.seek(clampedTime);
    }

    // Update display to show actual current time (in case of invalid input or clamping)
    this.elements.timeInput.value = this.formatTime(this.timeController.currentTime);
  }

  // Loading helpers
  showLoading() {
    this.elements.loading.classList.remove('hidden');
  }

  hideLoading() {
    this.elements.loading.classList.add('hidden');
  }

  setLoadingStatus(status) {
    this.elements.loadingStatus.textContent = status;
  }

  setLoadingProgress(percent) {
    this.elements.loadingProgress.style.width = `${percent}%`;
  }

  showFileSelector() {
    this.elements.fileSelector.classList.remove('hidden');
  }

  hideFileSelector() {
    this.elements.fileSelector.classList.add('hidden');
  }

  /**
   * Reset state and show file selector to load a new recording
   */
  resetAndShowFileSelector() {
    // Stop animation loop
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Stop playback
    if (this.timeController) {
      this.timeController.pause();
    }

    // Clear scene objects
    if (this.scene) {
      // Remove all ships
      for (const [id, ship] of this.scene.ships) {
        this.scene.scene.remove(ship);
      }
      this.scene.ships.clear();

      // Remove all projectiles
      for (const [id, proj] of this.scene.projectiles) {
        this.scene.scene.remove(proj);
        if (proj.userData.trail) {
          this.scene.scene.remove(proj.userData.trail);
        }
      }
      this.scene.projectiles.clear();

      // Clear PD beams, hit effects, and destruction effects
      this.scene.clearPDBeams();
      this.scene.clearHitEffects();
      this.scene.clearDestructionEffects();
    }

    // Reset state
    this.isInitialized = false;
    this.selectedShipId = null;
    this.lastTime = 0;
    this.lastPDEventTime = 0;
    this.lastHitEventTime = 0;
    this.lastDestructionTime = 0;
    this.currentState = null;

    // Clear UI
    this.elements.alphaShips.innerHTML = '';
    this.elements.betaShips.innerHTML = '';
    this.elements.eventList.innerHTML = '';
    this.elements.shipTelemetry.classList.add('hidden');

    // Show file selector
    this.showFileSelector();
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize on load
const visualizer = new BattleVisualizer();
visualizer.init();
