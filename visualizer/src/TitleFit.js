/**
 * Battle title fitting: one line, always. A long draft name ("150-point
 * draft: anthropic/claude-sonnet-5 vs deepseek/deepseek-v4.1-flash")
 * wrapped to three lines and covered the fleet scoreboard. The title now
 * shrinks toward a floor size to fit its slot, then ellipsizes (CSS), with
 * the full name as a tooltip.
 */

/**
 * Font size (px) that fits text measuring `naturalPx` at `maxPx` into
 * `availablePx`, never below `minPx` (the ellipsis takes the rest).
 * Letter-spacing is in em, so width scales linearly with the size.
 */
export function fitFontPx(naturalPx, availablePx, maxPx = 18, minPx = 11) {
  if (!(naturalPx > 0) || !(availablePx > 0) || naturalPx <= availablePx) return maxPx;
  const px = Math.floor((maxPx * availablePx / naturalPx) * 10) / 10;
  return Math.max(minPx, Math.min(maxPx, px));
}

const _installed = new WeakSet();

/**
 * Set `text` on the title element, fit it to one line and keep it fitted
 * when its slot resizes. The CSS size (media queries included) is the
 * maximum; `minPx` is the floor before the ellipsis.
 */
export function fitBattleTitle(el, text, { minPx = 10 } = {}) {
  if (!el) return;
  if (text !== undefined) {
    el.textContent = text;
    el.title = text;
  }
  const fit = () => {
    el.style.fontSize = '';
    const maxPx = parseFloat(getComputedStyle(el).fontSize) || 18;
    const px = fitFontPx(el.scrollWidth, el.clientWidth, maxPx, Math.min(minPx, maxPx));
    if (px < maxPx) el.style.fontSize = `${px}px`;
  };
  fit();
  if (!_installed.has(el) && typeof ResizeObserver !== 'undefined') {
    _installed.add(el);
    // the header's box only changes with the viewport, never with the fit
    new ResizeObserver(fit).observe(el.closest('header') || el.parentElement || el);
  }
}
