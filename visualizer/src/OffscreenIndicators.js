import * as THREE from 'three';

/**
 * Edge chevrons for what is outside the view (LV-09): one per fleet with
 * ships off screen ("◀ BETA ×6 · 412 km") and one for the selected ship.
 * Clicking a fleet chevron frames it. Own container; ContactOverlay keeps
 * the on-screen brackets.
 */

/**
 * Where a ray from the view centre towards (x, y) leaves the inset rect.
 * @returns {{x, y, angle}} screen position on the rect edge and the ray angle
 */
export function edgePoint(cx, cy, x, y, rect) {
  let dx = x - cx;
  let dy = y - cy;
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) dy = 1;
  const tx = dx > 0 ? (rect.right - cx) / dx : dx < 0 ? (rect.left - cx) / dx : Infinity;
  const ty = dy > 0 ? (rect.bottom - cy) / dy : dy < 0 ? (rect.top - cy) / dy : Infinity;
  const t = Math.max(0, Math.min(tx, ty));
  return { x: cx + dx * t, y: cy + dy * t, angle: Math.atan2(dy, dx) };
}

/**
 * Screen projection of a scene point, flipped when behind the camera so the
 * chevron still points the right way round.
 * @returns {{x, y, onScreen}}
 */
export function projectToScreen(point, camera, width, height, rect, scratch = new THREE.Vector3()) {
  const view = scratch.copy(point).applyMatrix4(camera.matrixWorldInverse);
  const behind = view.z > -camera.near;
  scratch.copy(point).project(camera);
  let { x, y } = scratch;
  if (behind) { x = -x; y = -y; }
  const sx = (x + 1) * width / 2;
  const sy = (1 - y) * height / 2;
  const onScreen = !behind && sx >= rect.left && sx <= rect.right && sy >= rect.top && sy <= rect.bottom;
  if (behind) {
    // Push far out along the flipped direction so it lands on the edge
    return { x: width / 2 + (sx - width / 2) * 1e3, y: height / 2 + (sy - height / 2) * 1e3, onScreen: false };
  }
  return { x: sx, y: sy, onScreen };
}

/**
 * Slide an edge point along its edge until it is clear of the blockers
 * (a chevron under the track HUD reads as part of it).
 */
export function avoidBlockers(edge, blockers, step = 18, tries = 24) {
  const inside = (x, y) => blockers.some(b => x >= b.left && x <= b.right && y >= b.top && y <= b.bottom);
  if (!inside(edge.x, edge.y)) return edge;
  // Along a vertical edge (left/right) move in y, along a horizontal one in x
  const vertical = Math.abs(Math.cos(edge.angle)) > Math.abs(Math.sin(edge.angle));
  for (let i = 1; i <= tries; i++) {
    for (const sign of [-1, 1]) {
      const x = vertical ? edge.x : edge.x + sign * i * step;
      const y = vertical ? edge.y + sign * i * step : edge.y;
      if (!inside(x, y)) return { ...edge, x, y };
    }
  }
  return edge;
}

/** Rendered and not hidden (offsetParent is null for position:fixed panels). */
const shown = el => !!el && !el.hidden && !el.classList.contains('hidden') && el.getClientRects().length > 0;

export class OffscreenIndicators {
  constructor({ onFleet, onShip } = {}) {
    this.root = document.createElement('div');
    this.root.id = 'offscreenIndicators';
    (document.getElementById('app') || document.body).appendChild(this.root);
    this.onFleet = onFleet;
    this.onShip = onShip;
    this.enabled = true;
    this.chevrons = new Map();
    this.p = new THREE.Vector3();
    this.scratch = new THREE.Vector3();
  }

  chevron(key, cls) {
    let c = this.chevrons.get(key);
    if (!c) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `offscreen-chevron ${cls}`;
      button.innerHTML = '<span class="oc-arrow">▶</span><span class="oc-label"></span>';
      button.addEventListener('click', () => (key.startsWith('fleet:') ? this.onFleet?.(key.slice(6)) : this.onShip?.(key.slice(5))));
      this.root.appendChild(button);
      c = { button, arrow: button.querySelector('.oc-arrow'), label: button.querySelector('.oc-label'), text: '' };
      this.chevrons.set(key, c);
    }
    return c;
  }

  /** Screen boxes a chevron must not sit under (corner HUD panels). */
  blockers() {
    const boxes = [];
    for (const id of ['trackHud', 'viewControls', 'directorChip', 'liveHud', 'liveConnection']) {
      const el = document.getElementById(id);
      if (!shown(el)) continue;
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) boxes.push({ left: r.left - 8, top: r.top - 8, right: r.right + 8, bottom: r.bottom + 8 });
    }
    return boxes;
  }

  /** Viewport area left for the battle (inside the HUD panels). */
  insetRect(width, height) {
    const rect = { left: 16, top: 96, right: width - 16, bottom: height - 104 };
    const left = document.getElementById('leftSidebar');
    if (shown(left)) {
      const r = left.getBoundingClientRect();
      if (r.width > 0 && r.right < width / 2) rect.left = Math.max(rect.left, r.right + 10);
    }
    for (const id of ['shipTelemetry', 'decisionPanel']) {
      const panel = document.getElementById(id);
      if (!shown(panel)) continue;
      const r = panel.getBoundingClientRect();
      if (r.width > 0 && r.left > width / 2) rect.right = Math.min(rect.right, r.left - 10);
    }
    return rect;
  }

  /**
   * @param {Object} scene - SceneManager (camera, ships, controls)
   * @param {Object} states - interpolated ship states
   * @param {Object} ships - loader.ships {id: {name, faction}}
   * @param {string|null} selectedId
   */
  update(scene, states, ships, selectedId) {
    this.root.hidden = !this.enabled;
    if (!this.enabled || !scene?.camera) return;
    const width = window.innerWidth, height = window.innerHeight;
    const camera = scene.camera;
    camera.updateMatrixWorld();
    const rect = this.insetRect(width, height);
    const cx = (rect.left + rect.right) / 2, cy = (rect.top + rect.bottom) / 2;
    const target = scene.controls?.target || new THREE.Vector3();
    const groups = { alpha: [], beta: [] };
    let selected = null;
    for (const [id, state] of Object.entries(states || {})) {
      if (!state || state.destroyed) continue;
      const ship = scene.ships.get(id);
      if (!ship) continue;
      const faction = ships?.[id]?.faction || ship.userData?.faction || id.split('_')[0];
      const proj = projectToScreen(ship.position, camera, width, height, rect, this.scratch);
      if (!proj.onScreen && groups[faction]) groups[faction].push(ship.position);
      if (id === selectedId && !proj.onScreen) selected = { id, pos: ship.position };
    }
    const seen = new Set();
    this._blockers = this.blockers();
    for (const faction of ['alpha', 'beta']) {
      const off = groups[faction];
      if (!off.length) continue;
      const centroid = this.p.set(0, 0, 0);
      for (const p of off) centroid.add(p);
      centroid.divideScalar(off.length);
      const km = centroid.distanceTo(target);
      const proj = projectToScreen(centroid, camera, width, height, rect, this.scratch);
      const edge = edgePoint(cx, cy, proj.x, proj.y, rect);
      this.place(`fleet:${faction}`, `fleet ${faction}`, edge,
        `${faction.toUpperCase()} ×${off.length} · ${km >= 10 ? Math.round(km) : km.toFixed(1)} km`,
        `${off.length} ${faction} ship${off.length > 1 ? 's' : ''} off screen - click to frame`);
      seen.add(`fleet:${faction}`);
    }
    if (selected) {
      const proj = projectToScreen(selected.pos, camera, width, height, rect, this.scratch);
      const edge = edgePoint(cx, cy, proj.x, proj.y, rect);
      const name = ships?.[selected.id]?.name || selected.id;
      this.place(`ship:${selected.id}`, `ship ${ships?.[selected.id]?.faction || ''}`, edge, name, `${name} is off screen - click to focus`);
      seen.add(`ship:${selected.id}`);
    }
    for (const [key, c] of this.chevrons) c.button.hidden = !seen.has(key);
  }

  place(key, cls, edge, text, title) {
    edge = avoidBlockers(edge, this._blockers || []);
    const c = this.chevron(key, cls);
    c.button.hidden = false;
    if (text !== c.text) {
      c.text = text;
      c.label.textContent = text;
      c.button.title = title;
      c.button.setAttribute('aria-label', title);
    }
    // Label sits inward from the arrow so it never runs off screen
    const right = Math.cos(edge.angle) > 0;
    c.button.classList.toggle('flip', right);
    c.button.style.transform = `translate(${Math.round(edge.x)}px, ${Math.round(edge.y)}px)`;
    c.arrow.style.transform = `rotate(${edge.angle}rad)`;
  }

  clear() {
    this.root.replaceChildren();
    this.chevrons.clear();
  }
}
