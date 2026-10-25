// Capture helpers for the README screenshots (docs/images/readme/README.md).
// Passed to scripts/viewer_snapshot.py as --eval "$(cat docs/images/readme/capture-helpers.js) ...call...";
// they only drive the viewer's own API (window.visualizer) - no pixels are edited.

/** Hull lineup: every class, alpha column facing the beta column, biggest hull on top,
 *  flank-on to a long lens. Ships are spawned with the viewer's own createShip/updateShip. */
function readmeLineup({ az = 20, el = 6, gap = 1.2, mid = 4, margin = 1.1, thrust = 0.3, fov = 12 } = {}) {
  const s = window.visualizer.scene, T = s.camera.position.constructor;
  window.visualizer.trackControl?.set('off');
  const CLASSES = ['dreadnought_siege', 'dreadnought', 'battleship', 'battlecruiser', 'cruiser_torpedo',
                   'cruiser', 'destroyer', 'frigate', 'corvette'];
  // local bounds of each hull (y = height, z = length) from its rendered meshes
  const ext = {};
  for (const t of CLASSES) {
    const id = `__lineup_probe_${t}`;
    const g = s.ships.get(id) || s.createShip(id, 'alpha', t);
    s.updateShip(id, { position: [5e6, 0, 0], forward: [0, 0, 1], thrust: 0, hull: 100, radiatorsExtended: false });
    g.updateMatrixWorld(true);
    const inv = g.matrixWorld.clone().invert();
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    g.traverseVisible(o => {
      if (!o.isMesh || !o.geometry) return;
      if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
      const bb = o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld).applyMatrix4(inv);
      ['x', 'y', 'z'].forEach((k, i) => { mn[i] = Math.min(mn[i], bb.min[k]); mx[i] = Math.max(mx[i], bb.max[k]); });
    });
    ext[t] = { h: mx[1] - mn[1], ymax: mx[1], zmax: mx[2], L: mx[2] - mn[2] };
  }
  // camera-aligned frame: c toward the camera, r screen right, u screen up
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  const c = new T(Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)).normalize();
  const r = new T().crossVectors(new T(0, 1, 0), c).normalize();
  const u = new T().crossVectors(c, r).normalize();
  const C = new T(9000, 4000, -6000);                       // far from any battle
  const put = (id, faction, type, x, y, fwd) => {
    const ship = s.ships.get(id) || s.createShip(id, faction, type);
    const p = C.clone().addScaledVector(r, x).addScaledVector(u, y);
    ship.up.copy(u);
    ship.userData.smoothThrust = thrust;
    s.updateShip(id, { position: [p.x * 1000, p.y * 1000, p.z * 1000], forward: [fwd.x, fwd.y, fwd.z],
      thrust, hull: 100, radiatorsExtended: false });
    ship.userData.smoothThrust = thrust;
  };
  const H = CLASSES.reduce((sum, t) => sum + ext[t].h, 0) + gap * (CLASSES.length - 1);
  const W = 2 * Math.max(...CLASSES.map(t => ext[t].L)) + mid;
  let y = H / 2;
  for (const t of CLASSES) {
    const yc = y - ext[t].ymax;
    put(`lineup_alpha_${t}`, 'alpha', t, -mid / 2 - ext[t].zmax, yc, r);            // noses meet in the middle
    put(`lineup_beta_${t}`, 'beta', t, mid / 2 + ext[t].zmax, yc, r.clone().negate());
    y -= ext[t].h + gap;
  }
  const halfV = Math.max(H / 2, W / 2 / s.camera.aspect) * margin;
  s.camera.fov = fov;
  s.camera.updateProjectionMatrix();
  s.camera.position.copy(C.clone().addScaledVector(c, halfV / Math.tan(fov * Math.PI / 360)));
  s.controls.target.copy(C);
  s.controls.update();
  return { W, H };
}

/** Command net: apply a filter, render it now, expand the DEVIATE reports and scroll the
 *  report whose text contains `match` to the top of the list (as a viewer scrolling would). */
function readmeCommsTo(match, faction = 'all', group = 'reports', pad = 26) {
  const v = window.visualizer;
  v.commsFeed.setFilter(faction, group);
  v.commsFeed.update(v.timeController.currentTime);
  const rows = [...document.querySelectorAll('#commsFeed .comms-row.status-deviate')];
  rows.forEach(el => el.classList.add('expanded'));
  const row = rows.find(el => el.textContent.includes(match)) || rows[0];
  if (!row) return 'no DEVIATE rows';
  let box = row.parentElement;
  while (box && !(box.scrollHeight > box.clientHeight + 2 && /(auto|scroll)/.test(getComputedStyle(box).overflowY))) box = box.parentElement;
  if (box) box.scrollTop += row.getBoundingClientRect().top - box.getBoundingClientRect().top - pad;
  return rows.length;
}

/** Decisions panel: scroll the section whose title starts with `title` (e.g. '03') to the top. */
function readmeDecisionTo(title, pad = 6) {
  const body = window.visualizer.decisions.body;
  const h = [...body.querySelectorAll('h3')].find(el => el.textContent.trim().startsWith(title));
  if (h) body.scrollTop += h.getBoundingClientRect().top - body.getBoundingClientRect().top - pad;
  return !!h;
}
