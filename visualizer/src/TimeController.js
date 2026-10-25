/**
 * TimeController - Manages playback timing
 *
 * `duration` is the length of the timeline (what the scrub bar spans and
 * timeline markers are placed against). `available` is how much of it has
 * footage: Infinity for a saved recording, the newest received frame in a
 * live battle, where the bar already spans up to the next checkpoint. The
 * playhead never runs or seeks past `available`; reaching it while more
 * footage is due holds playback ("buffering") instead of pausing.
 */
export class TimeController {
  /**
   * @param {number} duration - timeline span in seconds
   * @param {Object} [opts] - {speed}: initial playback speed, so a speed the
   *   viewer picked before the battle loaded (draft, battle switch) carries over
   */
  constructor(duration, { speed = 1.0 } = {}) {
    this.duration = duration;
    this.available = Infinity;
    this.currentTime = 0;
    this.playbackSpeed = 1.0;
    this.setSpeed(speed);
    this.isPlaying = false;
    this.callbacks = [];
  }

  /** Latest time the playhead may reach (end of footage). */
  get limit() {
    return Math.min(this.duration, this.available);
  }

  /** True when playback is held at the end of the footage received so far. */
  get buffering() {
    return this.isPlaying && this.available < this.duration && this.currentTime >= this.available - 1e-6;
  }

  /**
   * Start playback
   */
  play() {
    this.isPlaying = true;
  }

  /**
   * Pause playback
   */
  pause() {
    this.isPlaying = false;
  }

  /**
   * Toggle play/pause
   */
  toggle() {
    this.isPlaying = !this.isPlaying;
  }

  /**
   * Set playback speed
   * @param {number} speed - Playback speed multiplier
   */
  setSpeed(speed) {
    if (!Number.isFinite(speed)) return;
    this.playbackSpeed = Math.max(0.25, Math.min(8, speed));
  }

  /**
   * Grow (or correct) the total duration - live mode extends it as frames
   * stream in from the battle server
   * @param {number} duration - New duration in seconds
   */
  setDuration(duration) {
    this.duration = duration;
    if (this.currentTime > this.limit) {
      this.currentTime = this.limit;
    }
    this.notifyCallbacks();
  }

  /**
   * Live mode: footage received so far (Infinity = all of it). The
   * timeline itself (duration) may already extend further.
   * @param {number} available - Newest frame time in seconds
   */
  setAvailable(available) {
    this.available = available;
    if (this.currentTime > this.limit) {
      this.currentTime = this.limit;
    }
    this.notifyCallbacks();
  }

  /**
   * Seek to specific time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    this.currentTime = Math.max(0, Math.min(this.limit, time));
    this.notifyCallbacks();
  }

  /**
   * Seek by percentage (0-1)
   * @param {number} percent - Percentage of duration
   */
  seekPercent(percent) {
    this.seek(percent * this.duration);
  }

  /**
   * Update time based on delta time
   * @param {number} deltaTime - Time since last update in seconds
   */
  update(deltaTime) {
    if (!this.isPlaying) return;

    const before = this.currentTime;
    this.currentTime += deltaTime * this.playbackSpeed;

    if (this.currentTime >= this.duration) {
      this.currentTime = this.duration;
      this.isPlaying = false;
    } else if (this.currentTime > this.available) {
      // More footage is coming (live): hold at its end, keep playing
      this.currentTime = Math.max(before, this.available);
    }

    this.notifyCallbacks();
  }

  /**
   * Register a callback for time changes
   * @param {Function} callback - Function to call on time change
   */
  onTimeChange(callback) {
    this.callbacks.push(callback);
  }

  /**
   * Notify all registered callbacks
   */
  notifyCallbacks() {
    for (const callback of this.callbacks) {
      callback(this.currentTime, this.duration);
    }
  }

  /**
   * Get current time as formatted string
   */
  getTimeString() {
    return `${this.formatTime(this.currentTime)} / ${this.formatTime(this.duration)}`;
  }

  /**
   * Format seconds as MM:SS
   */
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Get progress as percentage (0-100)
   */
  getProgress() {
    return this.duration > 0 ? (this.currentTime / this.duration) * 100 : 0;
  }
}
