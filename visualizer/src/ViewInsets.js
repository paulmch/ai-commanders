/**
 * The battle view left free by the HUD's side columns. The auto director,
 * the fleet view (F) and the off-screen chevrons must agree on it: a column
 * one of them forgets hides ships under it (the pass-4 command net, open by
 * default on wide screens, sat on the director's whole right-hand fleet and
 * swallowed the right-edge chevrons).
 */

/** Columns docked on the left / right edge (ids in index.html). */
export const LEFT_PANEL_IDS = ['leftSidebar'];
export const RIGHT_PANEL_IDS = ['shipTelemetry', 'decisionPanel', 'commsFeed'];

/** Rendered and not hidden (offsetParent is null for position:fixed panels). */
export const panelShown = el => !!el && !el.hidden && !el.classList?.contains('hidden') &&
  el.getClientRects().length > 0;

/**
 * {left, right}: pixels the shown side columns take from each edge of a
 * `width`-wide view. A column counts only on the side its box sits on.
 * @param {Document} doc
 * @param {number} width - viewport width (px)
 */
export function viewInsets(doc, width) {
  const inset = { left: 0, right: 0 };
  const box = id => {
    const el = doc?.getElementById?.(id);
    if (!panelShown(el)) return null;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 ? r : null;
  };
  for (const id of LEFT_PANEL_IDS) {
    const r = box(id);
    if (r && r.right < width / 2) inset.left = Math.max(inset.left, r.right);
  }
  for (const id of RIGHT_PANEL_IDS) {
    const r = box(id);
    if (r && r.left > width / 2) inset.right = Math.max(inset.right, width - r.left);
  }
  return inset;
}

/**
 * Framing for the free view between the insets: the NDC x the action is
 * centred on (`shift`) and the share of the screen width it fills
 * (CameraController.frameGroups / slideToFreeArea). 0.62 of the width with
 * the registry alone (free ~0.85), scaled down as side columns narrow it.
 */
export function freeViewFrame(width, { left = 0, right = 0 } = {}) {
  const w = Math.max(1, width);
  const free = Math.max(0.3, (w - left - right) / w);
  return { shift: (left - right) / w, widthShare: 0.73 * free };
}
