/**
 * Graphics quality: 'low' | 'high'.
 *
 * Resolution order: `?gfx=low|high` URL parameter, then the choice saved in
 * localStorage by the GFX button / G hotkey, then auto-detection - software
 * rasterisers (SwiftShader, llvmpipe, ...) start on 'low' so the viewer
 * stays usable without a GPU; everything else starts on 'high'.
 *
 * The render modules (Environment, PostFX, DustMotes) subscribe with
 * onChange() and rebuild what the preset changes; nothing else in the
 * viewer needs to know the level.
 */

export const QUALITY_LEVELS = ['low', 'high'];

export const QUALITY_PRESETS = {
  low: {
    msaa: 0,             // 4x MSAA on the half-float target is ~650 ms/frame in SwiftShader
    fxaa: true,          // folded into the composite pass, no extra target
    aberration: false,
    godRays: false,
    dust: false,
    skySize: 512,        // baked sky cube face size
    starCount: 3200,     // brightest-first, so low keeps the prominent stars
    bloomMips: 5,
    pixelRatioCap: 1
  },
  high: {
    msaa: 4,
    fxaa: false,
    aberration: true,
    godRays: true,
    dust: true,
    skySize: 1024,
    starCount: 9000,
    bloomMips: 6,
    pixelRatioCap: 1.5
  }
};

const STORAGE_KEY = 'aic.gfxQuality';
const SOFTWARE_RE = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render|mesa offscreen|gdi generic/i;

export function isValidQuality(q) {
  return QUALITY_LEVELS.includes(q);
}

/** True when the unmasked WebGL renderer string names a CPU rasteriser. */
export function isSoftwareRenderer(rendererName) {
  return SOFTWARE_RE.test(String(rendererName || ''));
}

/** Pick the level: explicit param > saved choice > auto-detect. */
export function resolveQuality({ param = null, stored = null, rendererName = '' } = {}) {
  if (isValidQuality(param)) return param;
  if (isValidQuality(stored)) return stored;
  return isSoftwareRenderer(rendererName) ? 'low' : 'high';
}

export function nextQuality(level) {
  const i = QUALITY_LEVELS.indexOf(level);
  return QUALITY_LEVELS[(i + 1) % QUALITY_LEVELS.length];
}

/** Unmasked GL renderer name, or '' if the extension is unavailable. */
export function detectRendererName(renderer) {
  try {
    const gl = renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (ext) return String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '');
    return String(gl.getParameter(gl.RENDERER) || '');
  } catch {
    return '';
  }
}

function readStored() {
  try { return globalThis.localStorage?.getItem(STORAGE_KEY) ?? null; } catch { return null; }
}

function writeStored(level) {
  try { globalThis.localStorage?.setItem(STORAGE_KEY, level); } catch { /* private mode */ }
}

function readParam() {
  try {
    const q = new URLSearchParams(globalThis.location?.search || '').get('gfx');
    if (q === 'auto') { globalThis.localStorage?.removeItem(STORAGE_KEY); return null; }
    return q;
  } catch {
    return null;
  }
}

// The page-wide GFX button / G key drive whichever settings installed them last.
let _active = null;
let _keyInstalled = false;

export class GraphicsSettings {
  constructor(renderer) {
    this.renderer = renderer;
    this.rendererName = detectRendererName(renderer);
    this.software = isSoftwareRenderer(this.rendererName);
    this.level = resolveQuality({ param: readParam(), stored: readStored(), rendererName: this.rendererName });
    this.listeners = new Set();
    this.button = null;
    this._applyPixelRatio();
  }

  get preset() {
    return QUALITY_PRESETS[this.level];
  }

  onChange(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  set(level, { persist = true } = {}) {
    if (!isValidQuality(level) || level === this.level) return;
    this.level = level;
    if (persist) writeStored(level);
    this._applyPixelRatio();
    for (const fn of this.listeners) fn(level, this.preset);
    this._syncButton();
    // SceneManager.onResize resizes the renderer + post chain for the new DPR
    try { globalThis.dispatchEvent?.(new Event('resize')); } catch { /* node */ }
  }

  toggle() {
    this.set(nextQuality(this.level));
  }

  _applyPixelRatio() {
    const dpr = globalThis.devicePixelRatio || 1;
    const want = Math.min(dpr, this.preset.pixelRatioCap);
    if (Math.abs(this.renderer.getPixelRatio() - want) > 1e-3) this.renderer.setPixelRatio(want);
  }

  /**
   * GFX button in the view controls + the G hotkey. Idempotent across
   * reloads: every renderer gets its own GraphicsSettings, but the page
   * keeps ONE button and ONE key handler, which drive the newest settings.
   */
  installControls() {
    if (typeof document === 'undefined') return;
    _active = this;
    let b = document.getElementById('gfxBtn');
    if (!b) {
      const host = document.getElementById('viewControls');
      if (host) {
        b = document.createElement('button');
        b.id = 'gfxBtn';
        b.type = 'button';
        b.addEventListener('click', () => _active?.toggle());
        host.appendChild(b);
      }
    }
    this.button = b || null;
    this._syncButton();
    if (!_keyInstalled) {
      _keyInstalled = true;
      document.addEventListener('keydown', (e) => {
        if (e.code !== 'KeyG' || e.altKey || e.ctrlKey || e.metaKey) return;
        const t = e.target;
        if (t && (['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || t.isContentEditable)) return;
        _active?.toggle();
      });
    }
  }

  /** Drop subscribers and hand the shared controls back (renderer disposal). */
  dispose() {
    this.listeners.clear();
    if (_active === this) _active = null;
    this.button = null;
    _settings.delete(this.renderer);
  }

  _syncButton() {
    if (!this.button) return;
    const high = this.level === 'high';
    this.button.innerHTML = `GFX ${high ? 'HIGH' : 'LOW'} <kbd>G</kbd>`;
    this.button.classList.toggle('active', high);
    this.button.setAttribute('aria-pressed', String(high));
    this.button.title = `Graphics quality (G)${this.software ? ' - software renderer detected' : ''}`;
  }
}

const _settings = new WeakMap();
/** One GraphicsSettings per renderer, shared by every render module. */
export function getGraphicsSettings(renderer) {
  let s = _settings.get(renderer);
  if (!s) {
    s = new GraphicsSettings(renderer);
    _settings.set(renderer, s);
    if (typeof window !== 'undefined') window.__gfx = s;
  }
  return s;
}
