import { DecisionPanel } from './DecisionPanel.js';
/**
 * AI Commanders - Tactical Battle Visualizer
 * Main entry point - orchestrates all components
 */

import { BattleLoader } from './BattleLoader.js';
import { Interpolator } from './Interpolator.js';
import { SceneManager } from './SceneManager.js';
import { TimeController } from './TimeController.js';
import { CameraController, buildDirectorCuts } from './CameraController.js';
import { ContactOverlay } from './ContactOverlay.js';
import {
  LOST_AFTER_FAILURES, backoffDelay, checkpointNumber, classifyVerdict, defaultMaxLag,
  newLiveClientId, recordingQuery,
  deliberationProgress, formatClock, liveDelay, liveStartTime, newBattleReason, nextCheckpointAfter,
  paceLiveHead
} from './LiveSession.js';
import { ConnectionCard, DeliberationCard, DirectorChip, EndCard } from './LiveCards.js';
import { DraftSplash, normaliseDraftEntry } from './DraftSplash.js';
import { OffscreenIndicators } from './OffscreenIndicators.js';
import { generateShipSVG, getShipClassName, SHIP_SILHOUETTES } from './shipSilhouettes.js';
import { TrajectoryPredictor } from './TrajectoryPredictor.js';
import { TrackModeControl } from './TrackModeControl.js';
import { Tracks } from './render/Tracks.js';
import { TrackHud } from './TrackHud.js';
import { latestWreckEnd } from './render/Explosion.js';
import { fitBattleTitle } from './TitleFit.js';
import { updateRegistryHeat } from './HeatHud.js';
import { CombatLog } from './CombatLog.js';
import { CommsFeed } from './CommsFeed.js';
import { KillFeed } from './KillFeed.js';
import { Registry } from './Registry.js';
import { Scoreboard, ResultBanner, resultBannerDue } from './Scoreboard.js';
import { TimelineMarkers } from './TimelineMarkers.js';
import { showLightbox } from './hudDom.js';
import { freeViewFrame, viewInsets } from './ViewInsets.js';

class BattleVisualizer {
  constructor() {
    // Core components
    this.loader = null;
    this.scene = null;
    this.timeController = null;
    this.cameraController = null;

    // State
    this.isInitialized = false;
    // The speed the viewer picked. Lives here, not on the TimeController:
    // a speed chosen before the battle loads (live draft, connecting, a
    // battle switch, the file selector) must carry into the next one.
    this.playbackSpeed = 1.0;
    this.live = null; // Live-mode state; null when replaying a saved recording
    this.selectedShipId = null;
    this.lastTime = 0;
    this.lastHitEventTime = 0; // Track hit events for visual effects
    this.lastDestructionTime = 0; // Track destruction events for scrubbing
    this.lastMomentTime = 0; // Track torpedo/muzzle visual moments
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
    this.decisions = new DecisionPanel(this);
    this.initHud();   // scoreboard, kill feed, command net, combat log, registry, timeline markers
    this.contacts = new ContactOverlay(document.getElementById('shipContacts'), id => this.selectShip(id));
    this.setupEventListeners();
    // Trajectory tracks: T / playback-bar button; live PATHS mirrors it
    this.trackControl = new TrackModeControl({
      button: document.getElementById('tracksBtn'),
      isLive: () => !!this.live && !this.live.ended,
      onChange: mode => { if (this.live) { this.live.showPaths = mode !== 'off'; this.updateLiveHud(); } }
    });
    this.trackHud = new TrackHud({ root: document.getElementById('trackHud') });   // countdown, legend, Shift+T scope
    this.initLiveUi();
    this.hideLoading();

    // ?live=1 connects to a running battle server; ?recording=<url>
    // auto-loads a recording, skipping the file selector (used by the
    // launch scripts and the Playwright harness)
    const params = new URLSearchParams(window.location.search);
    const autoUrl = params.get('recording');
    if (params.get('live')) {
      this.startLiveMode(params.get('live_base') || '/live');
    } else if (autoUrl) {
      this.loadFromUrl(autoUrl);
    } else {
      this.showFileSelector();
    }
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
      liveBtn: document.getElementById('liveBtn'),

      // Live mode
      liveHud: document.getElementById('liveHud'),
      liveDot: document.getElementById('liveDot'),
      liveStatusText: document.getElementById('liveStatusText'),
      liveFollowBtn: document.getElementById('liveFollowBtn'),
      livePathsBtn: document.getElementById('livePathsBtn'),
      checkpointMarker: document.getElementById('checkpointMarker'),
      checkpointLabel: document.getElementById('checkpointLabel'),
    };
  }

  /**
   * Setup all event listeners
   */
  setupEventListeners() {
    document.getElementById('fleetViewBtn').addEventListener('click', () => this.showFleetView());
    document.getElementById('contactsBtn').addEventListener('click', () => this.toggleContacts());
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

    // Live mode entry + HUD toggles
    this.elements.liveBtn?.addEventListener('click', () => {
      window.location.href = '?live=1';
    });
    this.elements.liveFollowBtn?.addEventListener('click', () => {
      if (this.live) this.setLiveFollow(!this.live.follow);
    });
    this.elements.livePathsBtn?.addEventListener('click', () => {
      if (this.live) this.setLivePaths(!this.live.showPaths);
    });

    // Playback controls
    this.elements.playPauseBtn.addEventListener('click', () => {
      this.togglePlayback();
    });

    this.elements.timeline.addEventListener('input', (e) => {
      if (this.timeController) {
        this.setLiveFollow(false); // manual scrub releases the live head
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
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) || e.target.isContentEditable) return;
      if (e.altKey || e.ctrlKey || e.metaKey || (e.code === 'Space' && e.target.tagName === 'BUTTON')) return;

      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlayback();
      } else if (e.code === 'ArrowLeft') {
        this.setLiveFollow(false);
        this.timeController?.seek(this.timeController.currentTime - 5);
      } else if (e.code === 'ArrowRight') {
        this.setLiveFollow(false);
        this.timeController?.seek(this.timeController.currentTime + 5);
      } else if (e.code === 'KeyF') {
        this.showFleetView();
      } else if (e.code === 'KeyD' && this.isInitialized) {
        this.decisions.setOpen(!this.decisions.open);
      } else if (e.code === 'Escape' && this.decisions.open) {
        this.decisions.setOpen(false);
      } else if (e.code === 'KeyC') {
        this.toggleContacts();
      } else if (e.code === 'KeyA' && !e.shiftKey && this.isInitialized) {
        this.toggleDirector();
      } else if (e.code === 'KeyN' && this.isInitialized) {
        this.toggleComms();
      }
    });

    // Window resize
    window.addEventListener('resize', () => {
      if (this.scene) {
        this.scene.onResize();
        const cam = this.cameraController;
        if (cam?.fleetFramed && !cam.director.enabled) cam.focusFleet(this.currentState?.ships, this.directorFrame(true));
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
    this.scene?.dispose();   // no-op after a reset; guards any other reload path
    this.scene = new SceneManager(this.elements.canvas);
    this.tracks = new Tracks(this.scene, new TrajectoryPredictor(this.loader));
    this.tracks.hud = this.trackHud;
    this.scene.loader = this.loader;   // hull rigs read targets, shots, launches, PD dwell
    this.setLoadingProgress(60);

    // Initialize time controller. The extra seconds are an epilogue: a ship
    // destroyed near the end still gets its full destruction sequence
    // instead of freezing mid-fireball when the recording ends. A live
    // battle has no epilogue until it actually ends: its timeline spans the
    // footage received so far up to the next checkpoint, and the playhead
    // may only run to the newest frame (available).
    const liveRunning = this.live && !this.live.ended;
    this.timeController = new TimeController(liveRunning ? this.liveTimelineSpan() : this.replayEnd(),
      { speed: this.playbackSpeed });
    if (liveRunning) this.timeController.setAvailable(this.loader.duration);
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
    this.cameraController.onDirectorChange = state => this.directorChip?.update(state);
    // Auto director on unless ?director=0 (a harness placing the camera itself)
    const directorOn = new URLSearchParams(window.location.search).get('director') !== '0';
    this.needsFleetFraming = !directorOn && window.innerWidth < 900;

    // Create ships
    this.setLoadingStatus('Spawning fleet assets...');
    for (const [shipId, shipInfo] of Object.entries(this.loader.ships)) {
      this.scene.createShip(shipId, shipInfo.faction, shipInfo.type);
    }
    // Death causes + final vectors drive the two-stage destruction sequence
    this.scene.deathInfo = this.loader.buildDeathInfo();
    this.scene.damageFX?.setRecording(this.loader);
    this.cameraController.setDirectorCuts(buildDirectorCuts(this.loader.events, this.scene.deathInfo));
    this.cameraController.setDirector(directorOn);
    this.setLoadingProgress(100);

    // Update UI
    this.updateBattleInfo();
    this.populateShipLists();
    this.decisions.reset();
    this.resetHud();
    this.contacts.populate(this.loader.ships);
    document.getElementById('viewControls').classList.remove('hidden');
    this.populateTimelineMarkers();
    this.updateFleetStatus();

    // Hide loading and start
    this.isInitialized = true;
    this.directorChip?.setVisible(true);
    await this.delay(300);
    this.hideLoading();

    // Start animation loop
    this.startAnimationLoop();

    // Live: join one checkpoint before the live edge (or at the start); a
    // battle that already ended is a replay, watched from the beginning
    if (liveRunning && this.live.phase !== 'ended') {
      this.timeController.seek(liveStartTime(this.loader.duration, this.live.interval, this.live.status));
    }

    // Deep links can open a specific decision at an observed outcome.
    const replayParams = new URLSearchParams(window.location.search);
    const startTime = Number(replayParams.get('t'));
    if (replayParams.has('t') && Number.isFinite(startTime)) this.timeController.seek(startTime);
    const initialShip = replayParams.get('ship');
    if (initialShip && this.loader.ships[initialShip]) this.selectShip(initialShip);
    if (replayParams.get('decisions') === '1') {
      this.decisions.setOpen(true);
      const decisionId = replayParams.get('decision');
      if (decisionId && this.loader.decisions.byId.has(decisionId)) this.decisions.pin(decisionId);
      this.decisions.update(this.timeController.currentTime);
      if (this.live) this.timeController.play();
      else this.timeController.pause();
    } else this.timeController.play();
    this.updatePlayPauseButton();
  }

  /**
   * Replay timeline end: the recording plus a 20 s epilogue, extended until
   * the latest wreck (drift + 40 s breakup) has fully played out.
   */
  replayEnd() {
    return Math.max(this.loader.duration + 20, latestWreckEnd(this.loader.buildDeathInfo()));
  }

  /**
   * Main animation loop
   */
  startAnimationLoop() {
    let lastTimestamp = 0;

    const animate = (timestamp) => {
      const deltaTime = (timestamp - lastTimestamp) / 1000;
      lastTimestamp = timestamp;

      // Headless harnesses (scripts/viewer_snapshot.py) render frames far
      // slower than realtime and opt out of the stalled-tab guard.
      if (this.isInitialized && (deltaTime < 1 || this.slowFrameOk)) {
        // Update time. Live follow-head tracks the newest frame instead of
        // free-running (which would auto-pause at the still-growing end).
        if (this.live?.started && this.live.follow && !this.live.ended && this.timeController.isPlaying) {
          this.updateFollowHead(deltaTime);
        } else {
          this.timeController.update(deltaTime);
        }

        // Get interpolated state (pass loader for projectile extrapolation to impact)
        const frameData = this.loader.getFrameAt(this.timeController.currentTime);
        this.currentState = Interpolator.getInterpolatedState(frameData, this.loader);

        const currentPlaybackTime = this.timeController.currentTime;
        this.decisions.update(currentPlaybackTime);
        this.scene.currentTime = currentPlaybackTime;

        // Clear before updating ships so a backwards seek can recreate a
        // wreck at its recorded age in this same frame.
        if (currentPlaybackTime < this.lastHitEventTime) {
          this.scene.clearHitEffects();
          this.lastHitEventTime = currentPlaybackTime - 6;
        }
        if (currentPlaybackTime < this.lastDestructionTime) {
          this.scene.clearDestructionEffects();
          this.lastDestructionTime = currentPlaybackTime;
        }
        if (currentPlaybackTime < this.lastMomentTime) {
          this.scene.clearSmallEffects();
          this.lastMomentTime = currentPlaybackTime - 6;
        }

        // Update ships (with radiator state from the event timeline)
        for (const [shipId, state] of Object.entries(this.currentState.ships)) {
          state.radiatorsExtended = this.loader.getRadiatorStateAt(shipId, currentPlaybackTime);
          this.scene.updateShip(shipId, state);
        }
        // RCS attitude jets, fired from the recorded rotation around this time
        this.scene.rcs.update(this.scene.ships, this.loader.simTrace, currentPlaybackTime);

        // Slugs, torpedoes, PD beams, muzzle discharge and the instant
        // impact flash: stateless, evaluated from the recording each frame
        this.scene.weapons.update(this.loader, currentPlaybackTime);

        // Torpedo lifecycle moments crossing this frame: detonations,
        // misses, seeker kills, burnouts - plus gun muzzle flashes
        const momentStart = Math.max(this.lastMomentTime, currentPlaybackTime - 6);
        const moments = this.loader.getTorpedoMomentsInRange(momentStart, currentPlaybackTime);
        for (const m of moments) {
          if (m.kind === 'impact') {
            this.scene.spawnTorpedoDetonation(m.position, m.data.damageGj || 8, m.timestamp);
          } else if (!this.scene.weapons.ownsMoment(m.kind)) {
            this.scene.spawnSmallEffect(m.position, m.timestamp, m.kind);
          }
        }
        this.lastMomentTime = currentPlaybackTime;

        // Animate small effects
        this.scene.updateSmallEffects(currentPlaybackTime);

        // Spawn hit effects when projectiles visually reach their targets
        // (based on extrapolated travel time, not recording event timestamps)
        const visualImpacts = this.loader.getVisualImpacts(Math.max(this.lastHitEventTime, currentPlaybackTime - 6), currentPlaybackTime);
        for (const impact of visualImpacts) {
          this.scene.spawnHitEffect(
            impact.impact_position,
            impact.kinetic_energy_gj,
            impact.timestamp
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

        // Predicted (and in replay, actual) tracks to the next checkpoint.
        // Past the last frame (epilogue, finished live battle) there is no
        // next checkpoint: no tracks, no stale countdown.
        const trackMode = this.trackControl.effectiveMode();
        this.tracks.frame(currentPlaybackTime, this.currentState.ships,
          currentPlaybackTime > this.loader.duration + 0.5 ? 'off' : trackMode,
          this.selectedShipId || this.cameraController.targetShipId);

        // Update camera
        if (this.needsFleetFraming) {
          this.cameraController.focusFleet(this.currentState.ships, this.directorFrame(true));
          this.needsFleetFraming = false;
        }
        if (this.pendingEvidenceFocus) {
          this.cameraController.focusOnShip(this.pendingEvidenceFocus, this.currentState.ships);
          this.cameraController.setMode('orbit', this.pendingEvidenceFocus);
          this.pendingEvidenceFocus = null;
        }
        this.cameraController.update(this.currentState.ships,
          { time: currentPlaybackTime, dt: deltaTime, torpedoes: this.currentState.torpedoes, ...this.directorFrame() });

        // Update selected ship telemetry
        if (this.selectedShipId && this.currentState.ships[this.selectedShipId]) {
          this.updateShipTelemetry(this.currentState.ships[this.selectedShipId]);
        }

        // Update fleet status
        this.updateFleetStatus();
        updateRegistryHeat(this.scene.heat, currentPlaybackTime);   // registry heat bars (HeatHud.js)

        // Process events
        this.processEvents();
      }

      // Render
      this.scene.render();
      if (this.isInitialized) {
        this.contacts.update(this.scene, this.currentState?.ships, this.selectedShipId);
        this.offscreen?.update(this.scene, this.currentState?.ships, this.loader.ships, this.selectedShipId);
        this.directorChip?.update(this.cameraController.directorState());
      }

      this.animationId = requestAnimationFrame(animate);
    };

    this.animationId = requestAnimationFrame(animate);
  }

  // ==================== Live mode ====================
  //
  // A live battle streams in bursts: one decision interval of footage in a
  // few seconds, then nothing while the commanders deliberate. The viewer
  // plays it like a broadcast with a short delay (LiveSession.paceLiveHead),
  // shows who the battle is waiting for at each checkpoint, survives server
  // restarts (battle_id) and crashes (backoff, connection-lost card, saved
  // replay), and closes with a verdict card once the footage has played out.

  /** Live overlays: draft splash, deliberation / connection / end cards, auto-cam chip, off-screen chevrons. */
  initLiveUi() {
    this.draftSplash = new DraftSplash({ onDismiss: () => this.dismissDraftSplash() });
    this.deliberationCard = new DeliberationCard();
    this.connectionCard = new ConnectionCard({
      onReplay: () => this.loadSavedReplay(),
      onRetry: () => this.retryLiveNow(),
      onStop: () => this.endLiveHere(),
    });
    this.endCard = new EndCard({
      onSeek: t => this.watchFrom(t),
      onReplay: () => this.watchFrom(0),
      onDecisions: () => this.decisions.setOpen(true),
    });
    this.directorChip = new DirectorChip({ onToggle: () => this.toggleDirector() });
    this.directorChip.setVisible(false);
    this.offscreen = new OffscreenIndicators({
      onFleet: faction => this.frameFaction(faction),
      onShip: id => this.selectShip(id),
    });
    document.getElementById('livePaceBtn')?.addEventListener('click', () => {
      if (this.live) this.setLivePace(this.live.pace === 'edge' ? 'realtime' : 'edge');
    });
    document.getElementById('liveFleetsBtn')?.addEventListener('click', () => this.showDraftRecap());
    this.elements.liveStatusText?.addEventListener('click', () => {
      if (this.live?.ended && this.live.endShown) this.showEndCard();
    });
  }

  /**
   * Where the auto director and the fleet view put the action: centred (NDC
   * x `shift`) in the view left free by the registry and the right-hand
   * columns (telemetry, decisions, command net), filling a matching share
   * of the width (ViewInsets.js). Re-measured at most twice a second;
   * `fresh` re-measures now (a panel was just toggled).
   */
  directorFrame(fresh = false) {
    const now = performance.now();
    if (!fresh && this._frameAt && now - this._frameAt < 500) return this._frame;
    this._frameAt = now;
    const width = window.innerWidth || 1;
    this._frame = freeViewFrame(width, viewInsets(document, width));
    return this._frame;
  }

  /** A: arm the auto director (or switch it off when it is already flying). */
  toggleDirector() {
    const cam = this.cameraController;
    if (!cam) return;
    const on = !(cam.director.enabled && !cam.director.hold);
    if (on && this.selectedShipId) this.deselectShip();   // the director flies the free camera
    cam.setDirector(on);
  }

  /** Off-screen chevron click: frame that fleet (re-arms a held director). */
  frameFaction(faction) {
    const cam = this.cameraController;
    if (!cam || !this.currentState) return;
    if (cam.director.enabled) {
      cam.focusFleet(this.currentState.ships, this.directorFrame(true));
      return;
    }
    const ships = Object.fromEntries(Object.entries(this.currentState.ships)
      .filter(([id]) => (this.loader.ships[id]?.faction || id.split('_')[0]) === faction));
    cam.focusFleet(ships, this.directorFrame(true));
  }

  /** End card / kill list: play from t (leaves the live head). */
  watchFrom(t) {
    if (!this.timeController) return;
    this.setLiveFollow(false);
    this.timeController.seek(t);
    this.timeController.play();
    this.updatePlayPauseButton();
  }

  /**
   * Enter live mode: poll the battle server, stream frames into the
   * loader, and keep playback tracking the live head.
   * @param {string} base - Base URL of the live API (default '/live')
   * @param {Object} [opts] - {banner}: HUD notice, e.g. after a battle switch
   */
  startLiveMode(base, { banner = null } = {}) {
    const now = performance.now();
    this.live = {
      base: base,
      clientId: newLiveClientId(),   // names this viewer to the server's end-of-battle linger
      battleId: null,
      phase: null,
      status: null,
      prevStatus: null,
      statusSince: now,
      checkpoint: 0,
      simTimeS: 0,
      nextCheckpointT: 0,
      interval: 30,
      waitingFor: [],
      draft: null,
      stage: null,
      error: null,
      pendingCalls: [],
      statusAge: null,
      receivedAt: now,
      spend: null,
      models: null,
      replayUrl: null,
      result: null,
      provenance: null,
      follow: true,
      pace: 'realtime',   // 'realtime' (broadcast delay) | 'edge' (hug the newest frame)
      paceState: 'playing',
      showPaths: this.trackControl.mode !== 'off',
      lastFrameT: 0,
      started: false, // battle data received and scene initialized
      ended: false,
      endShown: false,
      verdict: null,
      connected: false,
      everConnected: false,
      failures: 0,
      lost: false,
      stopped: false,
      lastOkAt: now,
      nextPollAt: now,
      timer: null,
      uiTimer: null,
      polling: false,
      sawDraft: false,
      splashUntil: 0,
      recapOpen: false,
      banner,
      bannerUntil: banner ? now + 8000 : 0,
    };
    this.hideFileSelector();
    this.elements.liveHud.classList.remove('hidden');
    // HUD clocks (deliberation timer, delay, retry countdown) tick between polls
    this.live.uiTimer = setInterval(() => this.updateLiveHud(), 500);
    this.updateLiveHud();
    this.pollLive();
  }

  /** A different battle is on the wire: tear this one down and follow the new one. */
  restartLiveMode(banner) {
    const base = this.live?.base || '/live';
    this.teardownBattle();
    this.startLiveMode(base, { banner });
  }

  /**
   * One poll cycle. Reschedules itself: every 2 s while the server answers,
   * backing off 2 -> 4 -> 8 -> 15 s while it does not. After
   * LOST_AFTER_FAILURES misses the connection-lost card offers the saved
   * replay; retries continue in the background.
   */
  async pollLive() {
    const live = this.live;
    if (!live || live.ended || live.polling || live.stopped) return;
    live.polling = true;
    if (live.timer) { clearTimeout(live.timer); live.timer = null; }
    let ok = false;
    try {
      const query = recordingQuery({ clientId: live.clientId, started: live.started,
        lastFrameT: live.lastFrameT, lastSeq: this.loader?.lastEventSequence });
      const response = await fetch(`${live.base}/recording${query}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (this.live !== live) return;   // torn down or switched battles meanwhile
      ok = true;
      live.connected = live.everConnected = true;
      live.failures = 0;
      live.lost = false;
      live.lastOkAt = performance.now();
      await this.handleLivePayload(payload);
    } catch (error) {
      if (this.live !== live) return;
      console.warn('Live poll failed:', error);
      live.connected = false;
      live.failures++;
      if (live.everConnected && live.failures >= LOST_AFTER_FAILURES) live.lost = true;
    } finally {
      if (this.live === live) live.polling = false;
    }
    if (this.live !== live || live.ended || live.stopped) return;
    // Before the first answer the server is probably still starting: keep it brisk
    const delay = ok ? 2 : backoffDelay(live.everConnected ? live.failures : Math.min(live.failures, 2));
    live.nextPollAt = performance.now() + delay * 1000;
    live.timer = setTimeout(() => this.pollLive(), delay * 1000);
    this.updateLiveHud();
  }

  retryLiveNow() {
    const live = this.live;
    if (!live || live.ended) return;
    live.stopped = false;
    this.pollLive();
  }

  async handleLivePayload(payload) {
    const live = this.live;
    const info = payload.live || {};
    const reason = newBattleReason(
      { battleId: live.battleId, started: live.started, lastFrameT: live.lastFrameT, interval: live.interval }, info);
    if (reason) {
      console.info(`Live: a new battle is on the wire (${reason}); switching`);
      this.restartLiveMode('NEW BATTLE DETECTED');
      return;
    }
    live.battleId = info.battle_id || live.battleId;
    live.phase = info.phase;
    live.status = info.status;
    if (live.status !== live.prevStatus) {
      live.prevStatus = live.status;
      live.statusSince = performance.now();
    }
    live.checkpoint = info.checkpoint ?? 0;
    live.simTimeS = info.sim_time_s ?? 0;
    live.nextCheckpointT = info.next_checkpoint_t ?? 0;
    live.interval = info.decision_interval_s || live.interval || 30;
    live.waitingFor = info.waiting_for || [];
    live.draft = info.draft || live.draft;
    live.stage = info.stage ?? null;
    live.error = info.error ?? null;
    live.pendingCalls = info.pending_calls || [];
    live.statusAge = typeof info.status_age_s === 'number' ? info.status_age_s : null;
    live.receivedAt = performance.now();
    live.spend = info.spend ?? live.spend;
    live.models = info.models || live.models;
    live.replayUrl = info.replay_url ?? live.replayUrl;
    live.result = info.result ?? live.result;
    if (info.phase === 'draft') live.sawDraft = true;

    const recording = payload.recording;
    if (recording?.provenance) live.provenance = recording.provenance;
    if (!live.started) {
      // recording stays null through the draft phase; wait for frames so
      // the interpolator always has something to bracket
      if (recording && (recording.sim_trace || []).length > 0) {
        await this.initializeBattle(recording, 'LIVE');
        if (this.live !== live) return;
        live.started = true;
        live.lastFrameT = this.loader.duration;
        // Coming from the draft: keep the committed fleets (and both
        // admirals' reasoning, now unsealed) up while the fleets close
        if (live.sawDraft) live.splashUntil = performance.now() + 14000;
      }
    } else if (recording) {
      this.applyLiveChunk(recording);
    }

    if (info.phase === 'ended' && !live.ended) {
      await this.finishLiveBattle();
    }

    this.updateLiveHud();
  }

  /**
   * Merge one poll's recording chunk. Deliberation-only polls (orders,
   * model calls) skip the death / damage / director rebuilds.
   */
  applyLiveChunk(recording) {
    const loader = this.loader;
    const frames = loader.simTrace.length;
    if (!loader.appendLiveData(recording)) {
      this.updateLiveTimeline();
      return false;
    }
    this.live.lastFrameT = loader.duration;
    const commandLayer = /^(model_|admiral_|captain_|comms_|command_status|decision_point|message$)/;
    const combat = loader.simTrace.length > frames ||
      (recording.events || []).some(e => !commandLayer.test(e.event_type || ''));
    if (combat) {
      this.scene.deathInfo = loader.buildDeathInfo();
      this.scene.damageFX?.setRecording(loader);
      this.cameraController.setDirectorCuts(buildDirectorCuts(loader.events, this.scene.deathInfo));
    }
    this.updateLiveTimeline();
    this.updateBattleInfo();
    this.populateTimelineMarkers();
    return true;
  }

  /**
   * Live timeline: the bar spans the footage received plus the run up to the
   * next checkpoint (so its marker sits at its true place); the playhead may
   * only reach the newest frame.
   */
  liveTimelineSpan() {
    const live = this.live;
    const available = this.loader?.duration || 0;
    return Math.max(available, nextCheckpointAfter(Math.max(live?.simTimeS || 0, available), live?.interval || 30)) + 5;
  }

  updateLiveTimeline() {
    const tc = this.timeController;
    if (!tc || !this.live || this.live.ended) return;
    tc.setAvailable(this.loader.duration);
    tc.setDuration(this.liveTimelineSpan());
    this.updateBufferMask();
  }

  /** Hatched span of the timeline whose footage has not arrived yet. */
  updateBufferMask() {
    let mask = document.getElementById('liveBufferMask');
    const track = document.querySelector('#playbackControls .timeline-track');
    if (!mask && track) {
      mask = document.createElement('div');
      mask.id = 'liveBufferMask';
      mask.className = 'live-buffer-mask';
      mask.title = 'Not received yet';
      track.appendChild(mask);
    }
    if (!mask) return;
    const tc = this.timeController;
    const show = !!(this.live?.started && !this.live.ended && tc && tc.available < tc.duration);
    mask.hidden = !show;
    if (show) mask.style.left = `${(tc.available / tc.duration) * 100}%`;
  }

  /**
   * Battle over: one more incremental fetch as a safety net (the server
   * lingers for it), then hand back to normal replay. The 'ended' poll
   * itself already carried the verdict, so a failed fetch loses nothing.
   */
  async finishLiveBattle({ fetchFinal = true } = {}) {
    const live = this.live;
    if (live.timer) {
      clearTimeout(live.timer);
      live.timer = null;
    }
    if (fetchFinal && live.started && !live.lost) {
      try {
        const query = recordingQuery({ clientId: live.clientId, started: true,
          lastFrameT: live.lastFrameT, lastSeq: this.loader.lastEventSequence });
        const response = await fetch(`${live.base}/recording${query}`, { cache: 'no-store' });
        if (response.ok && this.live === live) {
          const payload = await response.json();
          if (payload.recording?.provenance) live.provenance = payload.recording.provenance;
          if (payload.recording) this.applyLiveChunk(payload.recording);
          live.result = payload.live?.result ?? live.result;
        }
      } catch (error) {
        console.warn('Final live fetch failed (keeping what was received):', error);
      }
    }
    if (this.live !== live) return;
    live.ended = true;
    live.lost = false;
    live.recapOpen = false;
    this.connectionCard.hide();
    this.deliberationCard.hide();
    if (live.started) {
      live.lastFrameT = this.loader.duration;
      // Restore the destruction epilogue for the completed replay
      this.timeController.setAvailable(Infinity);
      this.timeController.setDuration(this.replayEnd());
      this.updateBufferMask();
      this.updateBattleInfo();
      this.populateTimelineMarkers();
      this.scene.updatePredictedPaths?.(null);
      if (live.follow) {
        this.timeController.play();   // play out the delayed footage at the chosen speed
        this.updatePlayPauseButton();
      }
    }
    live.verdict = this.computeVerdict();
    this.updateLiveHud();
  }

  computeVerdict() {
    const live = this.live;
    const meta = this.loader?.metadata || {};
    return classifyVerdict({
      winner: (live.started ? meta.winner : null) ?? live.result?.winner ?? null,
      reason: (live.started ? meta.resultReason : null) ?? live.result?.reason ?? live.error ?? null,
      events: live.started ? this.loader.events : [],
      billing: live.provenance?.billing || null,
      lost: !!live.endedLost,
    });
  }

  /** Server gone: load its saved replay (written every checkpoint) and finish from it. */
  async loadSavedReplay() {
    const live = this.live;
    if (!live?.replayUrl || live.replayLoading) return;
    live.replayLoading = true;
    this.updateLiveHud();
    try {
      const response = await fetch(live.replayUrl, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (this.live !== live) return;
      const savedId = data.provenance?.battle_id;
      if (savedId && live.battleId && savedId !== live.battleId) {
        console.warn(`Saved replay ${live.replayUrl} is battle ${savedId}, not ${live.battleId}`);
      }
      live.stopped = true;
      if (live.timer) { clearTimeout(live.timer); live.timer = null; }
      if (data.provenance) live.provenance = data.provenance;
      if (!live.started) {
        if ((data.sim_trace || []).length) {
          await this.initializeBattle(data, 'LIVE');
          if (this.live !== live) return;
          live.started = true;
        }
      } else {
        this.applyLiveChunk(data);
      }
      live.result = { winner: data.winner ?? null, reason: data.result_reason ?? null };
      live.endedLost = !data.winner && !data.result_reason;
      live.lost = true;   // no final fetch from a dead server
      await this.finishLiveBattle({ fetchFinal: false });
    } catch (error) {
      console.warn('Saved replay failed:', error);
      live.replayError = `Saved replay unavailable (${error.message || error})`;
    } finally {
      if (this.live === live) {
        live.replayLoading = false;
        this.updateLiveHud();
      }
    }
  }

  /** Stop polling and treat what was received as the whole battle. */
  async endLiveHere() {
    const live = this.live;
    if (!live || live.ended) return;
    live.stopped = true;
    live.endedLost = true;
    live.lost = true;
    await this.finishLiveBattle({ fetchFinal: false });
  }

  /**
   * Advance the live head (broadcast delay, see LiveSession.paceLiveHead).
   * Speed buttons set the rate; it only speeds up past maxLag behind.
   */
  updateFollowHead(deltaTime) {
    const tc = this.timeController;
    const live = this.live;
    const step = paceLiveHead({
      current: tc.currentTime, available: this.loader.duration, dt: deltaTime,
      speed: tc.playbackSpeed, status: live.status, pace: live.pace, maxLag: defaultMaxLag(live.interval),
    });
    live.paceState = step.state;
    if (step.time !== tc.currentTime) {
      tc.currentTime = step.time;
      tc.notifyCallbacks();
    }
  }

  setLiveFollow(on) {
    if (!this.live || this.live.follow === on) return;
    this.live.follow = on;
    if (on && this.timeController && !this.live.ended && this.loader) {
      const tc = this.timeController;
      // Rejoin the broadcast; from far back, start at the last interval
      if (this.loader.duration - tc.currentTime > defaultMaxLag(this.live.interval)) {
        tc.seek(Math.max(0, this.loader.duration - this.live.interval));
      }
      tc.play();
      this.updatePlayPauseButton();
    }
    this.updateLiveHud();
  }

  setLivePace(pace) {
    if (!this.live) return;
    this.live.pace = pace === 'edge' ? 'edge' : 'realtime';
    if (!this.live.follow) this.setLiveFollow(true);
    this.updateLiveHud();
  }

  setLivePaths(on) {
    if (!this.live) return;
    // Same tracks as replay (TrajectoryPredictor over the streamed frames)
    this.trackControl.set(on ? 'predicted' : 'off');
  }

  /** Wall seconds in the server's current status (deliberation timer). */
  liveStatusElapsed() {
    const live = this.live;
    const now = performance.now();
    if (live.statusAge != null) return live.statusAge + (now - live.receivedAt) / 1000;
    return (now - live.statusSince) / 1000;
  }

  /**
   * The live HUD and overlays. Runs on every poll and twice a second
   * (clocks); every element writes the DOM only when its text changes.
   */
  updateLiveHud() {
    const live = this.live;
    if (!live) return;
    const now = performance.now();
    const tc = this.timeController;
    const current = live.started && tc ? tc.currentTime : 0;
    const available = live.started ? this.loader.duration : 0;
    const atEdge = live.started && !live.ended && available - current < 2.5;
    const deciding = live.connected && live.status === 'paused' && live.started && !live.ended;
    const elapsed = this.liveStatusElapsed();

    let text;
    let dot;
    if (live.ended) {
      const footageLeft = live.started ? this.loader.duration - current : 0;
      if (live.endShown || !live.started || footageLeft <= 0.5) {
        const v = live.verdict || this.computeVerdict();
        text = `${v.title}${v.detail ? ` — ${v.detail}` : ''}`;
      } else {
        text = `BATTLE OVER · ${formatClock(footageLeft)} OF FOOTAGE TO PLAY`;
      }
      dot = 'ended';
    } else if (live.lost) {
      text = 'CONNECTION LOST · battle server not responding';
      dot = 'lost';
    } else if (!live.connected) {
      const wait = Math.max(0, Math.ceil((live.nextPollAt - now) / 1000));
      text = live.everConnected
        ? `RECONNECTING · ${live.failures} missed · retry in ${wait}s`
        : `CONNECTING…${live.failures ? ` retry in ${wait}s` : ''}`;
      dot = 'wait';
    } else if (!live.started) {
      text = live.phase === 'draft' ? this.formatDraftStatus(live.draft)
        : `WAITING FOR THE BATTLE${live.stage ? ` · ${live.stage}` : ''}`;
      dot = 'deciding';
    } else if (deciding && atEdge) {
      text = `CP${live.checkpoint} · COMMANDERS DECIDING ${formatClock(elapsed)}`;
      dot = 'deciding';
    } else {
      text = `LIVE · T+${Math.round(current)}s`;
      if (live.status === 'paused') text += ` · CP${live.checkpoint} DECIDING ${formatClock(elapsed)}`;
      if (!live.follow) text += ' · REVIEWING';
      dot = 'live';
    }
    if (live.bannerUntil > now) text = `${live.banner} · ${text}`;
    if (this.elements.liveStatusText.textContent !== text) this.elements.liveStatusText.textContent = text;
    this.elements.liveStatusText.classList.toggle('clickable', !!(live.ended && live.endShown));
    const dotClass = `live-dot ${dot}${dot === 'live' ? ' active' : ''}`;
    if (this.elements.liveDot.className !== dotClass) this.elements.liveDot.className = dotClass;

    // Delay chip: how far the playhead runs behind the battle server
    const delayEl = document.getElementById('liveDelay');
    if (delayEl) {
      const show = live.started && !live.ended;
      delayEl.classList.toggle('hidden', !show);
      if (show) {
        const delay = liveDelay(Math.max(live.simTimeS, available), current);
        const label = !live.follow ? `−${Math.round(delay)}s · PAUSED HEAD`
          : live.paceState === 'catching-up' ? `⏩ −${Math.round(delay)}s`
            : delay < 2.5 ? 'AT LIVE EDGE' : `DELAY −${Math.round(delay)}s`;
        if (delayEl.textContent !== label) delayEl.textContent = label;
        delayEl.classList.toggle('catching', live.paceState === 'catching-up');
      }
    }
    const paceBtn = document.getElementById('livePaceBtn');
    if (paceBtn) {
      const label = live.pace === 'edge' ? 'LIVE EDGE' : 'REALTIME';
      if (paceBtn.textContent !== label) paceBtn.textContent = label;
      paceBtn.classList.toggle('active', live.follow);
      paceBtn.style.display = live.started && !live.ended ? '' : 'none';
    }
    const fleetsBtn = document.getElementById('liveFleetsBtn');
    if (fleetsBtn) fleetsBtn.style.display = live.started && live.provenance?.draft ? '' : 'none';

    this.elements.liveFollowBtn.classList.toggle('active', live.follow);
    this.elements.livePathsBtn.classList.toggle('active', live.showPaths);
    const togglesHidden = live.ended || !live.started;
    this.elements.liveFollowBtn.style.display = togglesHidden ? 'none' : '';
    this.elements.livePathsBtn.style.display = togglesHidden ? 'none' : '';

    // Deliberation card at the live edge; a compact HUD suffix otherwise
    if (deciding && atEdge && live.connected) this.deliberationCard.show(this.deliberationView(elapsed));
    else this.deliberationCard.hide();

    if (live.lost && !live.ended) {
      this.connectionCard.show({
        failures: live.failures,
        nextRetryS: live.stopped ? null : Math.max(0, (live.nextPollAt - now) / 1000),
        lastGoodS: (now - live.lastOkAt) / 1000,
        replayUrl: live.replayUrl,
        loading: live.replayLoading,
        message: live.replayError || null,
      });
    } else {
      this.connectionCard.hide();
    }

    this.updateDraftSplash(now);

    // Verdict card once the delayed footage has played out (no spoilers)
    if (live.ended && !live.endShown && (!live.started || current >= this.loader.duration + 2)) {
      live.endShown = true;
      this.showEndCard();
    }
    this.updateCheckpointMarker();
  }

  /** Data for the deliberation card at checkpoint live.simTimeS. */
  deliberationView(elapsed) {
    const live = this.live;
    const cpT = live.simTimeS || this.loader.duration;
    const alive = { alpha: 0, beta: 0 };
    for (const [id, s] of Object.entries(this.loader.simTrace.at(-1)?.ships || {})) {
      if (s.destroyed || s.dying || s.surrendered) continue;
      const f = this.loader.ships[id]?.faction || id.split('_')[0];
      if (f in alive) alive[f]++;
    }
    const progress = deliberationProgress({
      events: this.loader.getEventsInRange(cpT - 1e-3, cpT + 1e-3), ships: this.loader.ships,
      checkpointT: cpT, alive, pending: live.pendingCalls,
    });
    const waiting = !live.pendingCalls.length && live.waitingFor.length
      ? `Waiting for ${live.waitingFor.join(', ')}` : null;
    return {
      checkpoint: live.checkpoint, checkpointT: cpT, elapsed, progress, models: live.models,
      spend: live.spend, stage: live.stage, error: live.error || waiting,
      shipName: id => this.loader.ships[id]?.name || id,
    };
  }

  formatDraftStatus(draft) {
    if (!draft) return 'FLEET DRAFT';
    const parts = [];
    for (const [faction, d] of Object.entries(draft)) {
      if (!d) continue;
      parts.push(`${faction.toUpperCase()} ${d.ready ? 'READY' : d.is_mcp ? 'MCP PICKING' : 'PICKING'}`);
    }
    return `FLEET DRAFT · ${parts.join(' · ')}`;
  }

  /** Draft splash while drafting, briefly at battle start, and when reopened (FLEETS). */
  updateDraftSplash(now) {
    const live = this.live;
    let mode = null;
    if (!live.started && live.phase === 'draft') mode = 'draft';
    else if (live.recapOpen) mode = 'recap';
    else if (live.started && live.splashUntil > now && !live.ended) mode = 'starting';
    if (!mode) {
      if (this.draftSplash.visible) this.draftSplash.hide();
      return;
    }
    const summary = live.provenance?.draft;
    const pick = f => (mode !== 'draft' && summary?.[f])
      ? normaliseDraftEntry(summary[f], f, live.models?.[f])
      : normaliseDraftEntry(live.draft?.[f], f, live.models?.[f]);
    const pending = {};
    for (const call of live.pendingCalls || []) {
      if (call.faction && pending[call.faction] == null) pending[call.faction] = call.age_s + (now - live.receivedAt) / 1000;
    }
    const elapsed = live.draft?.alpha?.elapsed_s ?? live.draft?.beta?.elapsed_s;
    this.draftSplash.show({
      mode, alpha: pick('alpha'), beta: pick('beta'), pending,
      elapsed: elapsed != null ? elapsed + (now - live.receivedAt) / 1000 : this.liveStatusElapsed(),
    });
  }

  dismissDraftSplash() {
    if (!this.live) return;
    this.live.splashUntil = 0;
    this.live.recapOpen = false;
  }

  showDraftRecap() {
    if (!this.live?.provenance?.draft) return;
    this.live.recapOpen = true;
    this.updateLiveHud();
  }

  showEndCard() {
    if (!this.live) return;
    this.endCard.show(this.buildEndSummary());
  }

  /** Verdict, kills, survivors and spend for the end card. */
  buildEndSummary() {
    const live = this.live;
    const verdict = live.verdict || this.computeVerdict();
    const spend = live.spend;
    const billing = live.provenance?.billing || (spend ? {
      reported_cost_usd: spend.reported_usd, spend_limit_usd: spend.limit_usd, calls: spend.calls,
      failures: spend.failures, mode: spend.mode } : null);
    const summary = { verdict, duration: live.simTimeS, checkpoints: live.checkpoint, kills: [],
      survivors: { alpha: [], beta: [] }, fleetSize: { alpha: 0, beta: 0 }, billing, models: live.models };
    if (!live.started || !this.loader) return summary;
    const loader = this.loader;
    const name = id => loader.ships[id]?.name || id;
    const deaths = this.scene?.deathInfo || loader.buildDeathInfo();
    const first = new Map();
    for (const e of loader.events) {
      if ((e.event_type === 'ship_dying' || e.event_type === 'ship_destroyed') && e.ship_id && !first.has(e.ship_id)) {
        first.set(e.ship_id, { t: e.timestamp, killer: e.data?.killer_id });
      }
    }
    for (const id of new Set([...first.keys(), ...Object.keys(deaths)])) {
      const info = loader.ships[id] || {};
      const k = first.get(id);
      summary.kills.push({ t: k?.t ?? deaths[id].time, name: name(id), type: info.type,
        faction: info.faction || id.split('_')[0], killer: k?.killer ? name(k.killer) : null });
    }
    summary.kills.sort((a, b) => a.t - b.t);
    const last = loader.simTrace.at(-1)?.ships || {};
    for (const [id, info] of Object.entries(loader.ships)) {
      const f = info.faction;
      if (!(f in summary.fleetSize)) continue;
      summary.fleetSize[f]++;
      const s = last[id];
      if (s && !s.destroyed && !s.dying) summary.survivors[f].push({ name: name(id), type: info.type, hull: s.hull ?? 100 });
    }
    summary.duration = loader.duration;
    summary.checkpoints = live.checkpoint || loader.sourceData?.total_checkpoints || 0;
    return summary;
  }

  /**
   * Gold tick on the timeline at the playhead's next checkpoint (or the one
   * being decided), labelled with the time to go / deliberation clock.
   */
  updateCheckpointMarker() {
    const el = this.elements.checkpointMarker;
    if (!el) return;
    const live = this.live;
    const tc = this.timeController;
    if (!live?.started || live.ended || !tc?.duration) {
      el.classList.add('hidden');
      return;
    }
    const t = tc.currentTime;
    const deciding = live.status === 'paused' && this.loader.duration - t < 2.5;
    const cpT = deciding ? live.simTimeS : nextCheckpointAfter(t, live.interval);
    const n = deciding ? live.checkpoint : checkpointNumber(cpT, live.interval);
    const label = deciding ? `CP${n} · DECIDING ${formatClock(this.liveStatusElapsed())}` : `CP${n} · ${formatClock(cpT - t)}`;
    el.style.left = `${Math.min(100, (cpT / tc.duration) * 100)}%`;
    el.title = `Checkpoint ${n} at T+${Math.round(cpT)}s`;
    if (this.elements.checkpointLabel.textContent !== label) this.elements.checkpointLabel.textContent = label;
    el.classList.toggle('deciding', deciding);
    el.classList.remove('hidden');
  }

  hideLiveOverlays() {
    this.draftSplash?.hide();
    this.deliberationCard?.hide();
    this.connectionCard?.hide();
    this.endCard?.hide();
    this.directorChip?.setVisible(false);
    this.offscreen?.clear();
    const mask = document.getElementById('liveBufferMask');
    if (mask) mask.hidden = true;
  }

  /**
   * Update battle info in header
   */
  updateBattleInfo() {
    this.elements.battleName.textContent = this.loader.metadata.battleName || 'FLEET ENGAGEMENT';
    fitBattleTitle(this.elements.battleName, this.elements.battleName.textContent);   // one line + tooltip
    this.elements.totalTime.textContent = this.formatTime(this.loader.duration);
  }

  /**
   * Populate ship lists in registry panel
   */
  populateShipLists() {
    this.registry.populate(this.loader.ships);
  }

  /**
   * Drop colored markers on the timeline for key battle moments
   */
  populateTimelineMarkers() {
    if (!this.loader?.duration || !this.timeController) return;
    this.timelineMarkers.build(this.loader.hud, this.timeController.duration, this.loader.ships);
  }

  /**
   * Update fleet status (alive counts)
   */
  updateFleetStatus() {
    if (!this.currentState || !this.loader?.hud) return;
    const t = this.timeController.currentTime;
    const hud = this.loader.hud;
    const sides = this.scoreboard.update(this.currentState.ships, this.currentState.torpedoes, this.loader.ships, hud, t);
    this.registry.update(this.currentState.ships, t, hud.kills);
    this.killFeed.update(hud.kills, t);
    // A dying ship is a confirmed kill counting down to its reactor
    // detonation; the banner waits for the end of a replayed recording (a
    // live battle's verdict is the live end card's, never both).
    const over = resultBannerDue(t, this.loader.duration, this.live);
    this.resultBanner.update(over, hud.result, sides, this.loader.duration);
  }

  /**
   * Select a ship
   */
  selectShip(shipId) {
    // Update selection state
    this.selectedShipId = shipId;
    this.decisions.selectShip(shipId);

    // Update ship list UI
    this.registry.select(shipId);

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
    const faction = shipInfo?.faction || (shipId.startsWith('alpha') ? 'alpha' : 'beta');
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

    this.registry.select(null);

    this.setCameraMode('free');
  }

  showFleetView() {
    if (!this.isInitialized) return;
    this.deselectShip();
    this.cameraController.focusFleet(this.currentState?.ships, this.directorFrame(true));
  }

  toggleContacts() {
    this.contacts.enabled = !this.contacts.enabled;
    const button = document.getElementById('contactsBtn');
    button.classList.toggle('active', this.contacts.enabled);
    button.setAttribute('aria-pressed', String(this.contacts.enabled));
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

    const hull = Math.max(0, Math.min(100, state.hull ?? 100));
    const thrust = (state.thrust || 0) * 100;

    this.elements.hullBar.style.width = `${hull}%`;
    this.elements.hullValue.textContent = state.destroyed ? 'DESTROYED'
      : state.dying ? 'REACTOR CRITICAL' : `${hull.toFixed(0)}%`;

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

      // Module keys come from the recording: build with textContent
      const info = document.createElement('div');
      info.className = 'module-info';
      for (const [cls, text] of [['module-name', displayName], ['module-status', statusText]]) {
        const span = document.createElement('span');
        span.className = cls;
        span.textContent = text;
        info.appendChild(span);
      }
      const iconEl = document.createElement('div');
      iconEl.className = 'module-icon';
      iconEl.textContent = icon;
      item.append(iconEl, info);

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
      if (!this.timeController.isPlaying) {
        this.setLiveFollow(false); // pausing releases the live head
      }
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
    if (!Number.isFinite(speed)) return;
    this.playbackSpeed = speed;
    this.timeController?.setSpeed(speed);
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
   * Bring the time-indexed HUD panels (combat log, command net) to the
   * playhead. Both are pure functions of (time, index): a parked playhead
   * re-renders nothing, a backward seek prunes, late live events append once.
   */
  processEvents() {
    const currentTime = this.timeController.currentTime;
    const complete = !this.live || !!this.live.ended;
    this.combatLog.setIndex(this.loader.hud);
    this.combatLog.update(currentTime);
    this.commsFeed.setIndex(this.loader.hud, complete);
    this.commsFeed.update(currentTime);
    this.lastTime = currentTime;
  }

  // ==================== Spectator HUD ====================

  initHud() {
    const $ = id => document.getElementById(id);
    this.registry = new Registry({ root: $('shipRegistry'), alpha: this.elements.alphaShips, beta: this.elements.betaShips,
      onSelect: id => this.selectShip(id) });
    this.combatLog = new CombatLog({ list: this.elements.eventList, chips: $('eventLogFilters'),
      onSeek: row => this.hudSeek(row.t, row.shipId) });
    this.scoreboard = new Scoreboard({ root: $('scoreboard') });
    this.resultBanner = new ResultBanner({ root: $('resultBanner') });
    this.killFeed = new KillFeed({ root: $('killFeed'), onSeek: kill => this.hudSeek(kill.t - 5, kill.shipId) });
    this.timelineMarkers = new TimelineMarkers({ events: $('timelineMarkers'), checkpoints: $('timelineCheckpoints'),
      onSeek: (t, shipId) => this.hudSeek(t, shipId) });
    this.commsFeed = new CommsFeed({
      root: $('commsFeed'),
      onSeek: row => this.hudSeek(row.t, row.shipId),
      onWhy: row => this.decisions.openAt(row.shipId, row.decisionId),
      onOpenChange: (open, user) => {
        const button = $('commsBtn');
        button?.classList.toggle('active', open);
        button?.setAttribute('aria-pressed', String(open));
        document.body.classList.toggle('comms-open', open);
        // Only the viewer's own toggle is remembered, never the width-based default
        if (user) try { window.localStorage.setItem('aic.viewer.commsOpen', open ? '1' : '0'); } catch { /* not persisted */ }
      },
      resolveImage: v => this.loader?.sourceData?.assets?.[v.ref]?.content?.[v.part]?.image_url?.url || null,
      onImage: (src, caption) => showLightbox(document, src, caption)
    });
    $('commsBtn')?.addEventListener('click', () => this.toggleComms());
  }

  /** Per-recording HUD setup (initializeBattle). */
  resetHud() {
    this.scoreboard.populate(this.loader.ships, this.loader.hud);
    this.resultBanner.reset();
    this.killFeed.clear();
    this.combatLog.clear();
    this.commsFeed.clear();
    // The command net opens by default where there is room for it, unless
    // the viewer closed it last time - or the replay carries no command
    // traffic at all (heuristic captains, pre-pass-4 recordings): an empty
    // column would only hide a third of the battle and the AUTO CAM chip.
    let saved = null;
    try { saved = window.localStorage.getItem('aic.viewer.commsOpen'); } catch { /* default */ }
    const traffic = !!this.live || (this.loader.hud?.comms?.length || 0) > 0;
    this.commsFeed.setOpen(saved != null ? saved === '1' : window.innerWidth >= 1100 && traffic);
  }

  toggleComms() {
    this.commsFeed.setOpen(!this.commsFeed.open, true);
    if (this.commsFeed.open && this.decisions.open) this.decisions.setOpen(false);
  }

  /** HUD click-through (log row, comms row, kill toast, timeline marker): pause at t and frame the ship. */
  hudSeek(t, shipId = null) {
    if (!this.timeController) return;
    this.setLiveFollow(false);
    this.timeController.pause();
    this.timeController.seek(Math.max(0, Math.min(t, this.timeController.duration)));
    this.updatePlayPauseButton();
    if (shipId && this.loader?.ships[shipId]) this.pendingEvidenceFocus = shipId;
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
      // A real jump (not the display refresh on blur) releases the live head
      if (Math.abs(clampedTime - this.timeController.currentTime) > 0.5) {
        this.setLiveFollow(false);
      }
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
    this.teardownBattle();
    this.showFileSelector();
  }

  /**
   * Stop the battle on screen: polling, playback, the scene and its UI.
   * Shared by the file selector reset and a live battle switch.
   */
  teardownBattle() {
    this.decisions.setOpen(false);
    this.pendingEvidenceFocus = null;
    // Leaving live mode: stop polling, hide the HUD and checkpoint marker
    if (this.live) {
      if (this.live.timer) clearTimeout(this.live.timer);
      if (this.live.uiTimer) clearInterval(this.live.uiTimer);
      this.live = null;
      this.elements.liveHud.classList.add('hidden');
      this.elements.checkpointMarker?.classList.add('hidden');
    }
    this.hideLiveOverlays();

    // Stop animation loop
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Stop playback
    if (this.timeController) {
      this.timeController.pause();
    }

    // Tear down the scene: ships, effects, tracks, render targets,
    // listeners and the renderer (the next load builds a fresh one)
    this.scene?.dispose();

    // Reset state
    this.isInitialized = false;
    this.contacts.clear();
    document.getElementById('viewControls').classList.add('hidden');
    this.selectedShipId = null;
    this.lastTime = 0;
    this.lastHitEventTime = 0;
    this.lastDestructionTime = 0;
    this.lastMomentTime = 0;
    this.currentState = null;

    // Clear UI
    this.registry.clear();
    this.combatLog.clear();
    this.commsFeed.clear();
    this.killFeed.clear();
    this.resultBanner.reset();
    this.timelineMarkers.clear();
    this.elements.shipTelemetry.classList.add('hidden');
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize on load
const visualizer = new BattleVisualizer();
visualizer.init();

// Exposed for automation (Playwright verification, headless snapshot capture)
window.visualizer = visualizer;
