/**
 * TimeController - Manages playback timing
 */
export class TimeController {
  constructor(duration) {
    this.duration = duration;
    this.currentTime = 0;
    this.playbackSpeed = 1.0;
    this.isPlaying = false;
    this.callbacks = [];
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
    this.playbackSpeed = Math.max(0.25, Math.min(8, speed));
  }

  /**
   * Seek to specific time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    this.currentTime = Math.max(0, Math.min(this.duration, time));
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

    this.currentTime += deltaTime * this.playbackSpeed;

    if (this.currentTime >= this.duration) {
      this.currentTime = this.duration;
      this.isPlaying = false;
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
