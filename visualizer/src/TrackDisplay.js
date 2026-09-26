/**
 * TrackDisplay - pure helpers for how trajectory tracks are drawn and
 * captioned (no THREE, no DOM; unit-tested in tests/trackdisplay.test.js).
 *
 * The viewer draws hulls roughly 60-80x oversize so a fleet reads at the
 * hundreds-of-km ranges the sim fights at. The evasion spread of a
 * predicted checkpoint (sigma, tens to hundreds of metres) and the way the
 * prediction wanders between checkpoints are therefore far below a hull
 * length on screen. To keep them visible AND comparable with the ships,
 * the spread rings and the drift locus are magnified by the same factor as
 * the hull they belong to (hullExaggeration), with a soft cap so a large
 * re-plan does not throw a glyph across the whole battle (magnifyRadius).
 * The caption always states the true number.
 */

/** Real hull lengths (m), data/fleet_ships.json. */
export const REAL_LENGTH_M = {
  corvette: 65, frigate: 100, destroyer: 125, cruiser: 175, cruiser_torpedo: 175,
  battlecruiser: 175, battleship: 200, dreadnought: 275, dreadnought_siege: 275
};
export const DEFAULT_EXAGGERATION = 60;
/** Soft cap of a magnified glyph, in displayed hull lengths. */
export const GLYPH_RMAX_HULLS = 4;

/**
 * Evasion modes that react to threats nobody can foresee (new rounds,
 * torpedo terminal phases): a prediction made in one is low confidence.
 * WOBBLE is modelled statistically (sigma), so it is not listed.
 */
export const REACTIVE_MODES = new Set(['evade', 'tank', 'brace', 'run', 'present']);

export function isReactiveMode(mode) {
  return typeof mode === 'string' && REACTIVE_MODES.has(mode.toLowerCase());
}

/**
 * Visual exaggeration of a hull: drawn length / real length.
 * @param {string} shipType
 * @param {number} displayLength - drawn hull length in scene units (km)
 */
export function hullExaggeration(shipType, displayLength) {
  const real = REAL_LENGTH_M[shipType];
  if (!real || !(displayLength > 0)) return DEFAULT_EXAGGERATION;
  return (displayLength * 1000) / real;
}

/**
 * Displayed radius of a true offset r (m) magnified by k: linear (r*k) while
 * small, eased into rMax (tanh), and never drawn smaller than it really is.
 * Monotonic in r, so the ordering of offsets survives the compression.
 */
export function magnifyRadius(r, k, rMax) {
  if (!(r > 0)) return 0;
  const lin = r * (k > 0 ? k : 1);
  if (!(rMax > 0)) return lin;
  return Math.max(r, rMax * Math.tanh(lin / rMax));
}

/**
 * Magnify the offset d = p - centre (metres, xyz at d[o..o+2]) in place of
 * `out[oo..oo+2]` with magnifyRadius, keeping its direction.
 */
export function magnifyOffset(d, k, rMax, out, o = 0, oo = 0) {
  const x = d[o], y = d[o + 1], z = d[o + 2];
  const r = Math.hypot(x, y, z);
  if (!(r > 0)) { out[oo] = out[oo + 1] = out[oo + 2] = 0; return out; }
  const f = magnifyRadius(r, k, rMax) / r;
  out[oo] = x * f; out[oo + 1] = y * f; out[oo + 2] = z * f;
  return out;
}

/** Two significant figures, e.g. 183 -> 180, 47 -> 47, 7.4 -> 7. */
function sig2(v) {
  if (v < 10) return Math.round(v);
  const p = Math.pow(10, Math.floor(Math.log10(v)) - 1);
  return Math.round(v / p) * p;
}

/**
 * "±180 m", "±1.2 km", "±12 km"; '' when there is no modelled spread.
 * @param {number} sigma - 1-sigma dispersion in metres
 */
export function formatSigma(sigma) {
  if (!(sigma >= 0.5)) return '';
  if (sigma < 995) return `±${sig2(sigma)} m`;
  const km = sigma / 1000;
  return km < 9.95 ? `±${km.toFixed(1)} km` : `±${Math.round(km)} km`;
}

/** Whole seconds to go, never negative ("T+0s" at arrival). */
export function secondsToGo(togo) {
  return Math.max(0, Math.ceil((Number(togo) || 0) - 1e-6));
}

/**
 * Caption of the focused ship's checkpoint marker:
 * "T+25s ±180 m", "T+25s ±1.2 km · RUN", "T+3s · ORDERS PENDING".
 */
export function formatCheckpointLabel({ togo, sigma = 0, stale = false, mode = null } = {}) {
  let s = `T+${secondsToGo(togo)}s`;
  const spread = formatSigma(sigma);
  if (spread) s += ` ${spread}`;
  if (stale) s += ' · ORDERS PENDING';
  else if (isReactiveMode(mode)) s += ` · ${mode.toUpperCase()}`;
  return s;
}

// ---------------------------------------------------------------------------
// Track scope: every ship, or only the focused (selected / followed) one
// ---------------------------------------------------------------------------
export const TRACK_SCOPES = ['all', 'focus'];
export const TRACK_SCOPE_KEY = 'aic.viewer.trackScope';

export function nextTrackScope(scope) {
  return scope === 'focus' ? 'all' : 'focus';
}

export function loadTrackScope(storage, fallback = 'all') {
  try {
    const v = storage?.getItem(TRACK_SCOPE_KEY);
    return TRACK_SCOPES.includes(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

export function saveTrackScope(storage, scope) {
  try {
    storage?.setItem(TRACK_SCOPE_KEY, scope);
  } catch {
    // blocked storage: the scope just won't persist
  }
}

/** Is ship `id` drawn under `scope` with focus `focusId`? */
export function inTrackScope(scope, id, focusId) {
  return scope !== 'focus' || (!!focusId && id === focusId);
}
