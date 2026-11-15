/**
 * Export every hull class's weapon mounts and blocked firing directions to
 * data/hull_mounts.json (contract: BRIEF "Wave 4 addendum").
 *
 *   node visualizer/scripts/export_hull_mounts.mjs            write the file
 *   node visualizer/scripts/export_hull_mounts.mjs --check    exit 1 if stale
 *   node visualizer/scripts/export_hull_mounts.mjs --report   coverage tables
 *
 * FRAME MAPPING. The hull kit builds a ship nose +Z, dorsal +Y, port +X
 * (scene units, origin at the hull centre). The engine's body frame is
 * x = forward, z = dorsal, y = z cross x = port. So
 *
 *     body (x, y, z) = viewer (Z, X, Y) * (length_m / scene length)
 *
 * a cyclic permutation, no sign flips (both right-handed).
 * azimuth = atan2(y, x) (0 nose, + port), elevation = asin(z) (+ dorsal).
 *
 * A direction is BLOCKED for a mount when a ray from the mount's pivot hits
 * the ship's own static geometry (every baked hull part except the mount's
 * own base, the closed launch doors, and the yaw-swept housings of the
 * OTHER mounts), sampled at the cell centre and its four corners, then
 * grown by one cell; or when it lies outside the mount's mechanical limits
 * (MountLimits.js; not grown, so a spinal cone keeps the engine's gimbal).
 * Radiators, other mounts' barrels, magazine rounds and the seed-scattered
 * deck clutter (parts tagged `seeded`: it differs ship to ship, the file is
 * keyed by class) are not occluders.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { assembleHull, HULL_CLASSES, hullSize } from '../src/render/Hulls.js';
import { SceneManager } from '../src/SceneManager.js';
import {
  AZ_CELLS, EL_CELLS, CELL_DEG, GUN_DEPRESSION_DEG, PD_DEPRESSION_DEG,
  GUN_HOUSING, PD_HOUSING, GUN_PIVOT_UP, PD_PIVOT_UP, cellIndex, packMask, unpackMask
} from '../src/render/MountLimits.js';

const ROOT = new URL('../../', import.meta.url);
export const MOUNTS_PATH = fileURLToPath(new URL('data/hull_mounts.json', ROOT));
const FLEET_PATH = fileURLToPath(new URL('data/fleet_ships.json', ROOT));
/** Geometry seed of the export (clutter is excluded, so it does not matter). */
export const EXPORT_SEED = 1234;
const D2R = Math.PI / 180;

/** viewer vector (X port, Y dorsal, Z nose) -> body [x fwd, y port, z dorsal] */
export function viewerToBody(v, k = 1) { return [v.z * k, v.x * k, v.y * k]; }
/** body direction [x, y, z] -> viewer Vector3 */
export function bodyToViewer(b, out = new THREE.Vector3()) { return out.set(b[1], b[2], b[0]); }

/** Unit body direction of (azimuth, elevation) in degrees. */
function bodyDir(az, el) {
  const ce = Math.cos(el * D2R);
  return [ce * Math.cos(az * D2R), ce * Math.sin(az * D2R), Math.sin(el * D2R)];
}

// -- occluders ---------------------------------------------------------------
function occluder(geom, owner = null) {
  const g = geom.index ? geom.toNonIndexed() : geom;
  g.computeBoundingSphere();
  return { pos: g.attributes.position.array, c: g.boundingSphere.center, r: g.boundingSphere.radius, owner };
}

/** Yaw-swept housing of a mount: a drum along the mount axis, or a ball. */
function housing(rig, kind) {
  const a = rig.angle, s = rig.s;
  const up = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
  const base = new THREE.Vector3(rig.x, rig.y, rig.z);
  if (kind === 'pd') {
    const g = new THREE.SphereGeometry(s * PD_HOUSING.r, 12, 8);
    g.translate(...base.addScaledVector(up, s * PD_PIVOT_UP).toArray());
    return g;
  }
  const h = s * GUN_HOUSING.h;
  const g = new THREE.CylinderGeometry(s * GUN_HOUSING.r, s * GUN_HOUSING.r, h, 16);
  g.translate(0, h / 2, 0);
  g.applyMatrix4(new THREE.Matrix4().makeRotationZ(a).setPosition(base));
  return g;
}

export function buildOccluders(kit) {
  const occ = [];
  const ownerOf = new Map();
  kit.turretRigs.forEach((r, i) => { for (let j = r.base[0]; j < r.base[1]; j++) ownerOf.set(j, `gun${i}`); });
  kit.pdRigs.forEach((r, i) => { for (let j = r.base[0]; j < r.base[1]; j++) ownerOf.set(j, `pd${i}`); });
  kit.parts.forEach((p, j) => { if (!p.seeded) occ.push(occluder(p.geom, ownerOf.get(j) || null)); });
  for (const d of kit.doors) for (const p of d.kit.parts) occ.push(occluder(p.geom));
  kit.turretRigs.forEach((r, i) => occ.push(occluder(housing(r, 'gun'), `gun${i}`)));
  kit.pdRigs.forEach((r, i) => occ.push(occluder(housing(r, 'pd'), `pd${i}`)));
  return occ;
}

const EPS = 1e-7;
/** Does the ray (o, unit d) hit any occluder not owned by `owner`? */
export function rayBlocked(occ, ox, oy, oz, dx, dy, dz, owner) {
  for (const b of occ) {
    if (owner && b.owner === owner) continue;
    // bounding sphere reject
    const cx = b.c.x - ox, cy = b.c.y - oy, cz = b.c.z - oz;
    const tca = cx * dx + cy * dy + cz * dz;
    const d2 = cx * cx + cy * cy + cz * cz - tca * tca;
    const r2 = b.r * b.r;
    if (d2 > r2 || (tca < 0 && cx * cx + cy * cy + cz * cz > r2)) continue;
    const p = b.pos;
    for (let i = 0; i < p.length; i += 9) {
      const e1x = p[i + 3] - p[i], e1y = p[i + 4] - p[i + 1], e1z = p[i + 5] - p[i + 2];
      const e2x = p[i + 6] - p[i], e2y = p[i + 7] - p[i + 1], e2z = p[i + 8] - p[i + 2];
      const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (det > -EPS && det < EPS) continue;
      const inv = 1 / det;
      const tx = ox - p[i], ty = oy - p[i + 1], tz = oz - p[i + 2];
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      if ((e2x * qx + e2y * qy + e2z * qz) * inv > 1e-4) return true;
    }
  }
  return false;
}

/**
 * Blocked mask (Uint8Array, row-major by elevation then azimuth) of a mount.
 * @param occ occluders   @param pivot viewer Vector3   @param owner tag
 * @param mech (bodyDir) => true when outside the mechanical limits
 * Returns {mask, selfBlocked}: selfBlocked = share of the mechanically
 * reachable sphere lost to the ship's own geometry (with the margin).
 */
export function mountMask(occ, pivot, owner, mech) {
  const hit = (az, el) => {
    const b = bodyDir(az, el);
    return rayBlocked(occ, pivot.x, pivot.y, pivot.z, b[1], b[2], b[0], owner);
  };
  // corner lattice (shared between cells) + centres
  const corner = new Uint8Array((AZ_CELLS + 1) * (EL_CELLS + 1));
  for (let j = 0; j <= EL_CELLS; j++) {
    for (let i = 0; i < AZ_CELLS; i++) corner[j * (AZ_CELLS + 1) + i] = hit(-180 + i * CELL_DEG, -90 + j * CELL_DEG) ? 1 : 0;
    corner[j * (AZ_CELLS + 1) + AZ_CELLS] = corner[j * (AZ_CELLS + 1)];
  }
  const geo = new Uint8Array(AZ_CELLS * EL_CELLS);
  for (let j = 0; j < EL_CELLS; j++) {
    for (let i = 0; i < AZ_CELLS; i++) {
      const c0 = j * (AZ_CELLS + 1) + i, c1 = c0 + AZ_CELLS + 1;
      geo[j * AZ_CELLS + i] = (corner[c0] || corner[c0 + 1] || corner[c1] || corner[c1 + 1]
        || hit(-180 + (i + 0.5) * CELL_DEG, -90 + (j + 0.5) * CELL_DEG)) ? 1 : 0;
    }
  }
  // one-cell safety margin on geometry hits (azimuth wraps)
  const mask = new Uint8Array(AZ_CELLS * EL_CELLS);
  let reach = 0, lost = 0;
  for (let j = 0; j < EL_CELLS; j++) {
    const w = Math.cos((-90 + (j + 0.5) * CELL_DEG) * D2R);
    for (let i = 0; i < AZ_CELLS; i++) {
      let g = 0;
      for (let dj = -1; dj <= 1 && !g; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= EL_CELLS) continue;
        for (let di = -1; di <= 1; di++) if (geo[jj * AZ_CELLS + (i + di + AZ_CELLS) % AZ_CELLS]) { g = 1; break; }
      }
      const m = mech(bodyDir(-180 + (i + 0.5) * CELL_DEG, -90 + (j + 0.5) * CELL_DEG));
      if (!m) { reach += w; if (g) lost += w; }
      mask[j * AZ_CELLS + i] = (g || m) ? 1 : 0;
    }
  }
  return { mask, selfBlocked: reach ? lost / reach : 0 };
}

/** Solid-angle weighted share of the sphere where pred(cell index) holds. */
export function sphereFraction(pred) {
  let a = 0, tot = 0;
  for (let j = 0; j < EL_CELLS; j++) {
    const w = Math.cos((-90 + (j + 0.5) * CELL_DEG) * D2R);
    for (let i = 0; i < AZ_CELLS; i++) { tot += w; if (pred(j * AZ_CELLS + i, i, j)) a += w; }
  }
  return a / tot;
}

const r2 = (v) => Math.round(v * 100) / 100;
const r4 = (v) => Math.round(v * 1e4) / 1e4;

/** Mounts of one hull class, in data/fleet_ships.json weapons order. */
export function hullMounts(type, fleet, { seed = EXPORT_SEED, faction = 'alpha' } = {}) {
  const ship = fleet.ships[type];
  const size = hullSize(type);
  const kit = assembleHull(type, size, SceneManager.prototype.getEngineConfig.call(null, type, size), seed, faction);
  const k = ship.hull.length_m / size.length;
  const occ = buildOccluders(kit);
  const weapons = ship.weapons;
  const spinal = /spinal|siege/.test(weapons[0].type);
  const guns = weapons.filter(w => /coilgun/.test(w.type));
  const pds = weapons.filter(w => w.type === 'pd_laser');
  const tubes = weapons.filter(w => w.type === 'torpedo_launcher');
  if (kit.hasSpinal !== spinal) throw new Error(`${type}: spinal mismatch between hull kit and fleet_ships.json`);
  if (kit.turretRigs.length !== guns.length) throw new Error(`${type}: ${kit.turretRigs.length} turrets drawn, ${guns.length} in data`);
  if (kit.pdRigs.length !== pds.length) throw new Error(`${type}: ${kit.pdRigs.length} PD drawn, ${pds.length} in data`);
  const banks = kit.magazine.reduce((n, e) => Math.max(n, e.bank + 1), 0);
  let gi = 0, pi = 0, ti = 0;
  const mounts = [];
  for (const w of weapons) {
    const m = { slot: w.slot, kind: null, type: w.type };
    let res;
    if (w.type === 'pd_laser') {
      if (w.slot !== `pd_${pi}`) throw new Error(`${type}: PD slot ${w.slot} is not pd_${pi}`);
      const rig = kit.pdRigs[pi];
      const up = new THREE.Vector3(-Math.sin(rig.angle), Math.cos(rig.angle), 0);
      const pivot = new THREE.Vector3(rig.x, rig.y, rig.z).addScaledVector(up, rig.s * PD_PIVOT_UP);
      const axis = viewerToBody(up);
      const lim = Math.sin(-PD_DEPRESSION_DEG * D2R);
      res = mountMask(occ, pivot, `pd${pi}`, (b) => b[0] * axis[0] + b[1] * axis[1] + b[2] * axis[2] < lim);
      Object.assign(m, { kind: 'pd', pos: viewerToBody(pivot, k).map(r2), axis: axis.map(r4), min_elevation_deg: -PD_DEPRESSION_DEG });
      pi++;
    } else if (w.type === 'torpedo_launcher') {
      const rounds = kit.magazine.filter(e => banks <= 1 || e.bank === ti % banks);
      const c = new THREE.Vector3();
      for (const e of rounds) c.add(e.pos);
      if (rounds.length) c.divideScalar(rounds.length);
      const radial = new THREE.Vector3(c.x, c.y, 0);
      const axis = banks > 1 && radial.length() > 1e-6 ? viewerToBody(radial.normalize()) : [1, 0, 0];
      res = { mask: new Uint8Array(AZ_CELLS * EL_CELLS), selfBlocked: 0 };
      Object.assign(m, { kind: 'launcher', pos: viewerToBody(c, k).map(r2), axis: axis.map(r4) });
      ti++;
    } else if (spinal && w === weapons[0]) {
      const cone = Math.cos(fleet.weapon_types[w.type].pivot_range_deg * D2R);
      // a hair ahead of the muzzle so the rail's own lip is behind the ray
      const pivot = kit.spinalMuzzle.clone().add(new THREE.Vector3(0, 0, kit.L * 0.005));
      res = mountMask(occ, pivot, null, (b) => b[0] < cone - 1e-9);
      Object.assign(m, { kind: 'spinal', pos: viewerToBody(kit.spinalMuzzle, k).map(r2), axis: [1, 0, 0], gimbal_deg: fleet.weapon_types[w.type].pivot_range_deg });
    } else {
      const rig = kit.turretRigs[gi];
      if (w.slot !== `weapon_${gi + (spinal ? 1 : 0)}`) throw new Error(`${type}: gun slot ${w.slot} out of order`);
      const up = new THREE.Vector3(-Math.sin(rig.angle), Math.cos(rig.angle), 0);
      const pivot = new THREE.Vector3(rig.x, rig.y, rig.z).addScaledVector(up, rig.s * GUN_PIVOT_UP);
      const axis = viewerToBody(up);
      const lim = Math.sin(-GUN_DEPRESSION_DEG * D2R);
      res = mountMask(occ, pivot, `gun${gi}`, (b) => b[0] * axis[0] + b[1] * axis[1] + b[2] * axis[2] < lim);
      Object.assign(m, { kind: 'gun', pos: viewerToBody(pivot, k).map(r2), axis: axis.map(r4), min_elevation_deg: -GUN_DEPRESSION_DEG });
      gi++;
    }
    m.blocked = packMask(res.mask);
    m.open_fraction = r4(sphereFraction((c) => !res.mask[c]));
    m.self_blocked_fraction = r4(res.selfBlocked);
    mounts.push(m);
  }
  return { length_m: ship.hull.length_m, metres_per_scene_unit: r4(k), mounts };
}

export function buildHullMounts(fleet = JSON.parse(readFileSync(FLEET_PATH, 'utf8')), opts = {}) {
  const hulls = {};
  for (const type of Object.keys(fleet.ships)) {
    if (!HULL_CLASSES.includes(type)) throw new Error(`no hull kit builder for ${type}`);
    hulls[type] = hullMounts(type, fleet, opts);
  }
  return {
    schema: 1,
    generated_by: 'visualizer/scripts/export_hull_mounts.mjs (do not edit; regenerate after any hull-kit or fleet_ships.json change)',
    frame: 'x forward, y port, z dorsal; metres from the hull centre; angles in degrees',
    viewer_frame: 'body (x, y, z) = hull-kit (Z nose, X port, Y dorsal) * metres_per_scene_unit; azimuth = atan2(y, x), elevation = asin(z)',
    grid: {
      az_step_deg: CELL_DEG, el_step_deg: CELL_DEG, az_cells: AZ_CELLS, el_cells: EL_CELLS,
      order: 'row-major by elevation cell from -90 up, then azimuth cell from -180 up',
      cell: 'az_cell = floor((az + 180) / 5) mod 72, el_cell = clamp(floor((el + 90) / 5), 0, 35); k = el_cell * 72 + az_cell',
      bits: 'cell k is bit (7 - k % 8) of byte k // 8 of bytes.fromhex(blocked) (MSB first), 648 hex digits; 1 = blocked'
    },
    notes: {
      mounts: 'listed in data/fleet_ships.json weapons order; the per-frame `tur` trace field uses this order',
      pos: 'pivot: gun = yaw axis at trunnion height, pd = ball centre, spinal = muzzle, launcher = magazine bank centre',
      axis: 'mount normal (the yaw axis; elevation of a turret is measured from the plane normal to it); spinal: boresight',
      blocked: 'own static geometry sampled at cell centre + corners and grown by one cell, OR outside the mechanical limits (turret depression below min_elevation_deg under the mount plane; spinal outside gimbal_deg of the nose); launchers are all-open',
      open_fraction: 'solid-angle weighted share of the sphere left open',
      self_blocked_fraction: 'share of the mechanically reachable directions lost to the ship\'s own geometry (incl. margin)'
    },
    hulls
  };
}

export function serialize(data) { return JSON.stringify(data, null, 1) + '\n'; }

/** The same data as an ES module for the viewer (render/TurretArcs.js). */
export const VIEWER_MODULE_PATH = fileURLToPath(new URL('../src/render/hullMountsData.js', import.meta.url));
export function serializeModule(data) {
  const hulls = {};
  for (const [type, h] of Object.entries(data.hulls)) {
    hulls[type] = { mounts: h.mounts.map(m => ({ slot: m.slot, kind: m.kind, type: m.type, pos: m.pos, axis: m.axis, blocked: m.blocked })) };
  }
  return '// GENERATED by visualizer/scripts/export_hull_mounts.mjs from data/hull_mounts.json - do not edit.\n'
    + '// Body frame: x forward, y port, z dorsal; grid and bit packing as in MountLimits.js.\n'
    + `export default ${JSON.stringify({ schema: data.schema, hulls }, null, 1)};\n`;
}

// -- coverage report ---------------------------------------------------------
const SECTORS = { nose: [1, 0, 0], stern: [-1, 0, 0], port: [0, 1, 0], starboard: [0, -1, 0], dorsal: [0, 0, 1], ventral: [0, 0, -1] };
const CONE_DEG = 30;
function regionName(b) {
  const parts = [];
  if (Math.abs(b[0]) > 0.38) parts.push(b[0] > 0 ? 'ahead' : 'astern');
  if (Math.abs(b[1]) > 0.38) parts.push(b[1] > 0 ? 'port' : 'starboard');
  if (Math.abs(b[2]) > 0.38) parts.push(b[2] > 0 ? 'high' : 'low');
  return parts.join(' ') || 'abeam';
}

/** Coverage numbers of a set of masks (Uint8Array[]) over the sphere. */
export function coverage(masks) {
  const count = (c) => masks.reduce((n, m) => n + (m[c] ? 0 : 1), 0);
  const out = { ge1: sphereFraction(c => count(c) >= 1), ge2: sphereFraction(c => count(c) >= 2), sectors: {}, open: [] };
  for (const [name, ax] of Object.entries(SECTORS)) {
    const inCone = (i, j) => {
      const b = bodyDir(-180 + (i + 0.5) * CELL_DEG, -90 + (j + 0.5) * CELL_DEG);
      return b[0] * ax[0] + b[1] * ax[1] + b[2] * ax[2] >= Math.cos(CONE_DEG * D2R);
    };
    const tot = sphereFraction((c, i, j) => inCone(i, j));
    out.sectors[name] = {
      ge1: sphereFraction((c, i, j) => inCone(i, j) && count(c) >= 1) / tot,
      ge2: sphereFraction((c, i, j) => inCone(i, j) && count(c) >= 2) / tot
    };
  }
  // name the uncovered directions: 26 regions, listed when > 20% uncovered
  const reg = new Map();
  for (let j = 0; j < EL_CELLS; j++) {
    const w = Math.cos((-90 + (j + 0.5) * CELL_DEG) * D2R);
    for (let i = 0; i < AZ_CELLS; i++) {
      const n = regionName(bodyDir(-180 + (i + 0.5) * CELL_DEG, -90 + (j + 0.5) * CELL_DEG));
      const r = reg.get(n) || { tot: 0, un: 0 };
      r.tot += w; if (count(j * AZ_CELLS + i) === 0) r.un += w;
      reg.set(n, r);
    }
  }
  for (const [n, r] of reg) if (r.un / r.tot > 0.2) out.open.push(`${n} ${Math.round(100 * r.un / r.tot)}%`);
  return out;
}

export function report(data) {
  const pc = (v) => `${Math.round(v * 100)}%`.padStart(4);
  const lines = [];
  for (const [type, h] of Object.entries(data.hulls)) {
    lines.push(`\n## ${type} (${h.length_m} m)`);
    lines.push(h.mounts.map(m => `${m.slot}[${m.kind}] open ${pc(m.open_fraction)} self-blocked ${pc(m.self_blocked_fraction)}`).join('\n'));
    for (const [label, kinds] of [['PD', ['pd']], ['guns', ['gun', 'spinal']]]) {
      const masks = h.mounts.filter(m => kinds.includes(m.kind)).map(m => unpackMask(m.blocked));
      if (!masks.length) continue;
      const c = coverage(masks);
      lines.push(`${label} x${masks.length}: >=1 ${pc(c.ge1)}  >=2 ${pc(c.ge2)} | ` +
        Object.entries(c.sectors).map(([n, s]) => `${n} ${pc(s.ge1)}/${pc(s.ge2)}`).join('  '));
      lines.push(`  uncovered: ${c.open.length ? c.open.join(', ') : 'none'}`);
    }
  }
  return lines.join('\n');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const data = buildHullMounts();
  const text = serialize(data);
  if (process.argv.includes('--report')) {
    console.log(report(data));
  } else if (process.argv.includes('--check')) {
    if (readFileSync(MOUNTS_PATH, 'utf8') !== text || readFileSync(VIEWER_MODULE_PATH, 'utf8') !== serializeModule(data)) { console.error('data/hull_mounts.json is stale: run node visualizer/scripts/export_hull_mounts.mjs'); process.exit(1); }
    console.log('data/hull_mounts.json is up to date');
  } else {
    writeFileSync(MOUNTS_PATH, text);
    writeFileSync(VIEWER_MODULE_PATH, serializeModule(data));
    console.log(`wrote ${MOUNTS_PATH} and ${VIEWER_MODULE_PATH}`);
  }
}
