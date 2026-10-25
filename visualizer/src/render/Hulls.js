import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import {
  getHullTextures, getRadiatorTextures, getAccentTexture, getHazardTexture, mulberry32
} from './Textures.js';
import { buildHullLights } from './HullLights.js';
import { buildMarkings, buildFloodlights, hullCode, LIVERY_FLOOD, GLASS_GLOW } from './HullMarkings.js';
import { TWIN_BARREL_X } from './WeaponTimeline.js';
export { updateHullDetails } from './HullRigs.js';

/**
 * Ship hull kit.
 *
 * Every hull is assembled from parts (bevelled plating, drums, trusses,
 * pipes, greebles, turrets...) that are baked into ship-local space with
 * UVs fitted to their physical size, then merged into one mesh per
 * material. A dreadnought is ~600 parts but ~9 draw calls. The part list
 * is kept on the group so the destruction sequence can split the hull
 * back into chunks along real part boundaries.
 *
 * Local frame: nose +Z, drive -Z, dorsal +Y, port +X (SceneManager points
 * +Z down the recorded forward vector with world-up +Y). Sizes in scene km.
 *
 * Besides the merged plating a hull carries a few live parts, each one
 * draw call: coilgun turret rigs, PD ball turrets, torpedo magazines
 * (instanced rack rounds / VLS hatches + status lamps), the nav-lamp cloud
 * (HullLights), the hull-code decals and floodlight pools (HullMarkings).
 *
 * Contract with the rest of the viewer (keep when changing hulls):
 *   userData.rcsMounts    every RCS nozzle {pos, dir} (render/Rcs.js)
 *   userData.gunTurrets   rigs {mount, yaw, pitch, barrels, s, slot}; the
 *                         muzzle sits 1.4 s down the barrels from the pitch
 *                         pivot at (0, 0.52 s, 0.15 s) above the mount
 *                         (Projectiles.localMuzzle) for every mount kind
 *   userData.pdMounts     PD ball centres by turret name pd_N (call order
 *                         matches the recording's pd_fired turret_name)
 *   userData.spinalMuzzle hull-local muzzle of the spinal mount, matching
 *                         Projectiles' SPINAL table
 *   userData.radiators    wing groups: Heat.js owns rotation.z / scale.y,
 *                         DamageFX owns rotation.x (buckled wings)
 */

const TEXEL = 1.35;   // plating tiles per scene unit

// ---------------------------------------------------------------------------
// Absolute fitting sizes
//
// Hull lengths are exaggerated per class, but equipment is not: the same
// coilgun battery, PD turret, window, RCS quad or radiator array is drawn
// at one physical size on every hull, so a dreadnought reads big by
// carrying MANY same-size fittings rather than looking like a zoomed-in
// frigate. fitScale() is a mild (L/8)^0.25 allowance so small-hull
// fittings stay legible and big-hull ones don't turn into greebles.
// ---------------------------------------------------------------------------

/** Mount size `s` (scene units) per weapon kind, before fitScale. */
export const WEAPON_SIZE = { coilgun: 0.62, heavy: 0.86, pd: 0.42 };
/** Deck window height; spacing and count follow from it. */
export const WINDOW_S = 0.05;
/** RCS thruster block size. */
export const RCS_S = 0.15;
/** Sensor mast height. */
export const MAST_H = 0.8;
/** Nav lamp glow size. */
export const LAMP_S = 0.075;
/**
 * Radiator panel area (scene units^2) carried by every hull: fleet_ships
 * gives all classes the same 60 t LithiumSpray array, 780 MW at 1500 K,
 * which needs ~1500 m^2 of two-sided panel - ~6.4 u^2 at the ~65x scale
 * the viewer draws hulls at. Split across the class's wings.
 */
export const RAD_AREA = 6.4;
/** Share of the extended span left showing when a wing is stowed. */
export const RADIATOR_STOW = 0.12;

/** Mild size allowance for fittings on a hull of length L. */
export function fitScale(L) {
  return Math.min(1.25, Math.max(0.9, Math.pow(L / 8, 0.25)));
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
const _sharedMats = new Map();

/** Accent emissive intensity at full hull (SceneManager dims it with damage). */
export const ACCENT_GLOW = 2.35;

// SceneManager drives the window emissiveIntensity (2.6 at full hull); the
// colour carries the level, so windows glow like a warship's portholes
// rather than a liner's promenade deck.
const WINDOW_EMISSIVE = new THREE.Color(0xffe4bd).multiplyScalar(0.5);

export { LIVERY_FLOOD };

function factionAccent(faction) {
  return faction === 'alpha' ? 0x2ad2ff : 0xff8a3c;
}

export function buildShipMaterials(faction) {
  const key = faction;
  if (!_sharedMats.has(key)) {
    const hullT = getHullTextures(faction, 'hull');
    const trimT = getHullTextures(faction, 'trim');
    const paintT = getHullTextures(faction, 'paint');
    _sharedMats.set(key, {
      hull: new THREE.MeshStandardMaterial({
        map: hullT.map, normalMap: hullT.normalMap, normalScale: new THREE.Vector2(0.9, 0.9),
        roughnessMap: hullT.ormMap, metalnessMap: hullT.ormMap,
        roughness: 1, metalness: 1, envMapIntensity: 1.0
      }),
      trim: new THREE.MeshStandardMaterial({
        map: trimT.map, normalMap: trimT.normalMap, normalScale: new THREE.Vector2(1.0, 1.0),
        roughnessMap: trimT.ormMap, metalnessMap: trimT.ormMap,
        roughness: 1, metalness: 1, envMapIntensity: 0.9
      }),
      // Faction livery: nose rims, drive bands, radiator cassettes, stripes.
      // Floodlit (a faint emissive copy of the albedo) so the livery still
      // names the side on a shadowed or backlit hull.
      paint: new THREE.MeshStandardMaterial({
        map: paintT.map, normalMap: paintT.normalMap, normalScale: new THREE.Vector2(0.8, 0.8),
        roughnessMap: paintT.ormMap, metalnessMap: paintT.ormMap,
        roughness: 1, metalness: 1, envMapIntensity: 0.85,
        emissive: 0xffffff, emissiveMap: paintT.map, emissiveIntensity: LIVERY_FLOOD
      }),
      dark: new THREE.MeshStandardMaterial({
        color: 0x1a1d22, roughness: 0.62, metalness: 0.85, envMapIntensity: 0.8
      }),
      barrel: new THREE.MeshStandardMaterial({
        color: 0x2a2d33, roughness: 0.45, metalness: 0.95, envMapIntensity: 1.0
      }),
      // Launcher rims and magazine clamps
      hazard: new THREE.MeshStandardMaterial({
        map: getHazardTexture().map, roughness: 0.62, metalness: 0.25, envMapIntensity: 0.7
      }),
      // Sensor windows and domes
      glass: new THREE.MeshStandardMaterial({
        color: 0x0c1c27, roughness: 0.12, metalness: 0.6, envMapIntensity: 1.6,
        emissive: 0x0d3550, emissiveIntensity: GLASS_GLOW
      })
    });
  }
  const shared = _sharedMats.get(key);
  const accent = factionAccent(faction);
  // Per-ship emissive materials (damage flicker, dying dimming). Livery
  // and sensor glass are self-lit too, so each ship gets its own copy that
  // HullRigs browns out with the hull (floodLevel), like the codes.
  const mats = {
    ...shared,
    paint: shared.paint.clone(),
    glass: shared.glass.clone(),
    // Accent light strips: lamp cells (getAccentTexture, mean ~0.68 of the
    // peak), glow raised to keep the old read at range
    accent: new THREE.MeshStandardMaterial({
      color: accent, emissive: accent, emissiveIntensity: ACCENT_GLOW, roughness: 0.35, metalness: 0.1,
      map: getAccentTexture().map, emissiveMap: getAccentTexture().emissiveMap
    }),
    window: new THREE.MeshStandardMaterial({
      color: 0x0a0a0c, emissive: WINDOW_EMISSIVE, emissiveIntensity: 2.6, roughness: 0.2, metalness: 0.2
    }),
    bellHeat: new THREE.MeshStandardMaterial({
      color: 0x1a1010, emissive: 0xff7a30, emissiveIntensity: 0.0, roughness: 0.8, metalness: 0.2,
      side: THREE.DoubleSide
    })
  };
  // Non-enumerable so code iterating the material set only sees materials
  Object.defineProperty(mats, 'faction', { value: faction, enumerable: false });
  return mats;
}

/** Material keys a hull part may use (buildShipMaterials + the radiator). */
export const HULL_MATERIAL_KEYS = ['hull', 'trim', 'paint', 'dark', 'barrel', 'hazard', 'glass', 'accent', 'window', 'bellHeat'];

// ---------------------------------------------------------------------------
// UV fitting: scale unit UVs to physical size so the plating tiles evenly
// ---------------------------------------------------------------------------
function offsetUVs(g, ox, oy, start = 0, count = -1) {
  const uv = g.attributes.uv;
  const end = count < 0 ? uv.count : start + count;
  for (let i = start; i < end; i++) uv.setXY(i, uv.getX(i) + ox, uv.getY(i) + oy);
}

function scaleUVRange(g, start, count, sx, sy) {
  const uv = g.attributes.uv;
  for (let i = start; i < start + count; i++) uv.setXY(i, uv.getX(i) * sx, uv.getY(i) * sy);
}

/** Box faces: px nx (d x h), py ny (w x d), pz nz (w x h). */
function fitBoxUVs(g, w, h, d, rnd, texel = TEXEL) {
  const sizes = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  // group vertex ranges: vertices are laid out face by face in BoxGeometry
  const perFace = g.attributes.uv.count / 6;
  for (let f = 0; f < 6; f++) {
    scaleUVRange(g, f * perFace, perFace, sizes[f][0] * texel, sizes[f][1] * texel);
    offsetUVs(g, rnd(), rnd(), f * perFace, perFace);
  }
}

function fitCylinderUVs(g, r, len, rnd, texel = TEXEL) {
  const groups = g.groups.length ? g.groups : [{ start: 0, count: g.index.count }];
  // Torso vertices come first; caps after. Use the index groups to find
  // vertex ranges - torso is group 0.
  const uv = g.attributes.uv;
  const torsoVerts = groups.length > 1 ? Math.min(uv.count, indexRangeMaxVertex(g, groups[0]) + 1) : uv.count;
  scaleUVRange(g, 0, torsoVerts, 2 * Math.PI * r * texel, len * texel);
  if (torsoVerts < uv.count) scaleUVRange(g, torsoVerts, uv.count - torsoVerts, 2 * r * texel, 2 * r * texel);
  offsetUVs(g, rnd(), rnd());
}

function indexRangeMaxVertex(g, group) {
  const idx = g.index.array;
  let m = 0;
  for (let i = group.start; i < group.start + group.count; i++) if (idx[i] > m) m = idx[i];
  return m;
}

/**
 * Frustum between two centred rectangles: w0 x h0 at z = 0 and w1 x h1 at
 * z = len. Non-indexed with flat normals; each face's UVs are fitted to
 * its size. The armour layers of rectangular nose caps.
 */
export function taperBoxGeometry(w0, h0, w1, h1, len, rnd = Math.random, texel = TEXEL) {
  const a = [[-w0 / 2, -h0 / 2, 0], [w0 / 2, -h0 / 2, 0], [w0 / 2, h0 / 2, 0], [-w0 / 2, h0 / 2, 0]];
  const b = [[-w1 / 2, -h1 / 2, len], [w1 / 2, -h1 / 2, len], [w1 / 2, h1 / 2, len], [-w1 / 2, h1 / 2, len]];
  const pos = [], uv = [];
  const quad = (p0, p1, p2, p3, uw, uh) => {
    const ox = rnd(), oy = rnd();
    const u = [[0, 0], [uw, 0], [uw, uh], [0, uh]].map(([x, y]) => [x * texel + ox, y * texel + oy]);
    for (const [i, p] of [[0, p0], [1, p1], [2, p2], [0, p0], [2, p2], [3, p3]]) {
      pos.push(...p);
      uv.push(...u[i]);
    }
  };
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    quad(a[i], a[j], b[j], b[i], dist(a[i], a[j]), dist(a[i], b[i]));
  }
  quad(b[0], b[1], b[2], b[3], w1, h1);          // front glacis (+Z)
  quad(a[0], a[3], a[2], a[1], h0, w0);          // back (-Z)
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.computeVertexNormals();
  return g;
}

// ---------------------------------------------------------------------------
// Kit
// ---------------------------------------------------------------------------
export class HullKit {
  constructor(size, seed) {
    this.L = size.length;
    this.W = size.width;
    // Plating density: a fixed tile size reads as a fine mosaic on the big
    // hulls, so tiles grow with hull length above a destroyer (8 units) -
    // a dreadnought's plates come out ~1.8x the size of a frigate's.
    this.texel = TEXEL * Math.min(1, Math.pow(8 / this.L, 0.6));
    this.k = fitScale(this.L);
    this.rnd = mulberry32(seed);
    this.faction = 'alpha';   // livery pattern (buildHull sets it)
    this.parts = [];          // {geom, mat, center}
    this.radiators = [];      // {mesh} separate animated meshes
    this.lamps = [];          // nav lamps {pos, color, kind, size, period, phase, level}
    this.bellHeat = [];       // meshes whose emissive follows thrust
    this.turretRigs = [];     // {s,kind,x,y,z,angle} moving turret parts built in build()
    this.pdRigs = [];         // {s,x,y,z,angle} PD ball turrets built in build()
    this.rcsMounts = [];      // {pos, dir} RCS nozzle exits, hull-local
    this.markings = [];       // hull-code decal placements
    this.floods = [];         // floodlight pools {x,y,z,face,len,width,dir}
    this.magazine = [];       // torpedo rounds {kind,bank,pos,quat,s,hinge}
    this.doors = [];          // {kit, bank, slide} torpedo launch doors
    this.drive = null;        // {shape:'torch'|'box'|'drum', ...} set by the class builder
    this.hasSpinal = false;   // weapon_0 is a spinal mount (turrets start at weapon_1)
    this.spinalMuzzle = null; // hull-local muzzle point of the spinal mount
    this.reactorPos = new THREE.Vector3(0, 0, -this.L * 0.3);
    this._m = new THREE.Matrix4();
  }

  // -- absolute sizes -------------------------------------------------------
  /** Mount size of a weapon kind ('coilgun' | 'heavy' | 'pd') on this hull. */
  ws(kind) { return WEAPON_SIZE[kind] * this.k; }
  get winS() { return WINDOW_S * this.k; }
  get rcsS() { return RCS_S * this.k; }
  get mastH() { return MAST_H * this.k; }
  get lampS() { return LAMP_S * this.k; }

  /**
   * Wing size for `n` radiator wings sharing RAD_AREA (x fitScale^2), as
   * {span, len}: len/span ~ `aspect`, each clamped to the space the class
   * has for them.
   */
  radiatorSize(n, { maxSpan = Infinity, maxLen = Infinity, aspect = 1.3 } = {}) {
    const area = RAD_AREA * this.k * this.k / n;
    let len = Math.min(maxLen, Math.sqrt(area * aspect));
    let span = Math.min(maxSpan, area / len);
    if (span * len < area * 0.999) len = Math.min(maxLen, area / span);
    return { span, len };
  }

  // -- core add -------------------------------------------------------------
  /** Add a geometry under a transform. `mat` is a material key. */
  add(mat, geom, matrix) {
    // RoundedBoxGeometry is non-indexed; mergeGeometries needs all parts
    // alike, so everything goes non-indexed
    if (geom.index) geom = geom.toNonIndexed();
    geom.applyMatrix4(matrix);
    geom.computeBoundingBox();
    const c = new THREE.Vector3();
    geom.boundingBox.getCenter(c);
    this.parts.push({ geom, mat, center: c });
    return geom;
  }

  /** Compose translate + euler rotation. */
  xf(x, y, z, rx = 0, ry = 0, rz = 0, parent = null) {
    const m = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz));
    m.setPosition(x, y, z);
    if (parent) m.premultiply(parent);
    return m;
  }

  // -- primitives -----------------------------------------------------------
  /** Bevelled plating box. */
  box(mat, w, h, d, x, y, z, { rot = null, bevel = 0.35, seg = 2, parent = null } = {}) {
    const r = Math.min(w, h, d) * 0.5 * bevel;
    const g = bevel > 0
      ? new RoundedBoxGeometry(w, h, d, seg, r)
      : new THREE.BoxGeometry(w, h, d);
    fitBoxUVs(g, w, h, d, this.rnd, this.texel);
    const m = this.xf(x, y, z, ...(rot || [0, 0, 0]), parent);
    return this.add(mat, g, m);
  }

  /** Drum with its axis along Z. sides=8 gives the octagonal hull look. */
  drum(mat, r, len, z, { x = 0, y = 0, sides = 8, rTop = null, open = false, parent = null, rot = null, sx = 1, sy = 1 } = {}) {
    const g = new THREE.CylinderGeometry(rTop ?? r, r, len, sides, 1, open);
    if (sides <= 8) g.rotateY(Math.PI / sides);
    fitCylinderUVs(g, r, len, this.rnd, this.texel);
    g.rotateX(Math.PI / 2);
    if (sx !== 1 || sy !== 1) g.scale(sx, sy, 1);
    const m = this.xf(x, y, z, ...(rot || [0, 0, 0]), parent);
    return this.add(mat, g, m);
  }

  /** Thin ring rib around a drum (a slightly larger, very short drum). */
  rib(mat, r, thick, z, { sides = 8, x = 0, y = 0, sx = 1, sy = 1 } = {}) {
    return this.drum(mat, r, thick, z, { sides, x, y, sx, sy });
  }

  /**
   * Flat ring facing aft (-Z) at z: a lit rim on a thrust plate. Phase
   * matches drum() so the ring lines up with an octagonal skirt.
   */
  annulus(mat, rIn, rOut, z, { sides = 8, x = 0, y = 0 } = {}) {
    // drum() puts vertices at -PI/2 (+ PI/sides when sides <= 8) + k 2PI/sides;
    // turning the ring to face -Z maps angle a to PI - a
    const start = sides <= 8 ? -Math.PI / 2 - Math.PI / sides : -Math.PI / 2;
    const g = new THREE.RingGeometry(rIn, rOut, sides, 1, start, Math.PI * 2);
    g.rotateY(Math.PI);
    return this.add(mat, g, this.xf(x, y, z));
  }

  /** Rectangular armour frustum, base (w0 x h0) at z0, front (w1 x h1) at z0+len. */
  taperBox(mat, w0, h0, w1, h1, len, x, y, z0, { parent = null } = {}) {
    const g = taperBoxGeometry(w0, h0, w1, h1, len, this.rnd, this.texel);
    return this.add(mat, g, this.xf(x, y, z0, 0, 0, 0, parent));
  }

  /** Four-sided wedge prow, half-width w / half-height h, apex at +Z. */
  prow(mat, w, h, len, z, { parent = null } = {}) {
    const g = new THREE.ConeGeometry(0.5, len, 4);
    g.rotateY(Math.PI / 4);
    g.rotateX(Math.PI / 2);
    g.scale(w * 2 * Math.SQRT2, h * 2 * Math.SQRT2, 1);
    fitCylinderUVs(g, (w + h) * 0.5, len, this.rnd, this.texel);
    return this.add(mat, g, this.xf(0, 0, z, 0, 0, 0, parent));
  }

  /**
   * Stepped ablative nose cap, from z0 forward over `len`: `layers` armour
   * frusta, each starting a step inside the last one's front face, so the
   * bow reads as stacked plate (Terra Invicta puts half a hull's armour
   * mass in the nose) instead of a painted cone. A thin faction-painted
   * rim rides the first layer - the livery a head-on enemy sees.
   *
   *   round:       r (base radius), sides, sx/sy aspect, taper (tip/base)
   *   rectangular: w, h (base size), tw/th (tip/base per axis)
   *   tip:         'lens' (sensor window), 'dome', or null (bare glacis)
   * Returns the z of the front face.
   */
  noseCap({ z0, len, r = null, w = null, h = null, sides = 8, sx = 1, sy = 1, taper = 0.5, tw = 0.5, th = 0.5,
    layers = 3, step = 0.9, x = 0, y = 0, tip = 'lens', rim = true }) {
    const seg = len / layers;
    const round = r !== null;
    const lerp = (a, b, u) => a + (b - a) * u;
    for (let i = 0; i < layers; i++) {
      const u0 = i / layers, u1 = (i + 1) / layers;
      const zA = z0 + i * seg;
      if (round) {
        const rb = r * lerp(1, taper, u0) * (i ? step : 1);
        const rt = r * lerp(1, taper, u1);
        this.drum('hull', rb, seg, zA + seg / 2, { rTop: rt, sides, x, y, sx, sy });
        if (i === 0 && rim) this.drum('paint', rb * 1.025, seg * 0.32, zA + seg * 0.16, { rTop: lerp(rb, rt, 0.32) * 1.025, sides, x, y, sx, sy });
      } else {
        const k0 = i ? step : 1;
        const bw = w * lerp(1, tw, u0) * k0, bh = h * lerp(1, th, u0) * k0;
        const fw = w * lerp(1, tw, u1), fh = h * lerp(1, th, u1);
        this.taperBox('hull', bw, bh, fw, fh, seg, x, y, zA);
        if (i === 0 && rim) {
          const e = 1.03;
          this.taperBox('paint', bw * e, bh * e, lerp(bw, fw, 0.32) * e, lerp(bh, fh, 0.32) * e, seg * 0.32, x, y, zA);
        }
      }
    }
    const zf = z0 + len;
    const tr = round ? r * taper * Math.min(sx, sy) : Math.min(w * tw, h * th) * 0.5;
    if (tip === 'lens') {
      // sensor window set into the glacis, ringed by a dark bezel
      this.drum('dark', tr * 0.5, len * 0.05, zf + len * 0.01, { x, y, sides: 12 });
      this.drum('glass', tr * 0.36, len * 0.05, zf + len * 0.025, { x, y, sides: 12 });
    } else if (tip === 'dome') {
      const g = new THREE.SphereGeometry(tr * 0.85, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
      g.rotateX(Math.PI / 2);
      this.add('glass', g, this.xf(x, y, zf - tr * 0.1));
      this.drum('dark', tr * 0.92, len * 0.06, zf - len * 0.01, { x, y, sides: 16 });
    }
    return zf;
  }

  /** Solid cone (radar dish, nose cap), axis along +Z. */
  cone(mat, r, len, x, y, z, { rot = null, sides = 12, parent = null } = {}) {
    const g = new THREE.ConeGeometry(r, len, sides);
    fitCylinderUVs(g, r, len, this.rnd, this.texel);
    g.rotateX(Math.PI / 2);
    return this.add(mat, g, this.xf(x, y, z, ...(rot || [0, 0, 0]), parent));
  }

  sphere(mat, r, x, y, z, { parent = null, ws = 12, hs = 8 } = {}) {
    const g = new THREE.SphereGeometry(r, ws, hs);
    scaleUVRange(g, 0, g.attributes.uv.count, 2 * Math.PI * r * this.texel, Math.PI * r * this.texel);
    return this.add(mat, g, this.xf(x, y, z, 0, 0, 0, parent));
  }

  /** Open truss: four longerons, cross frames, diagonal braces. */
  truss(mat, halfW, len, z, { x = 0, y = 0 } = {}) {
    const t = Math.max(halfW * 0.14, 0.012);
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        this.box(mat, t, t, len, x + sx * halfW, y + sy * halfW, z, { bevel: 0 });
      }
    }
    const frames = Math.max(2, Math.round(len / (halfW * 1.2)));
    for (let i = 0; i < frames; i++) {
      const fz = z - len / 2 + (i + 0.5) * (len / frames);
      this.box(mat, halfW * 2 + t, t, t, x, y + halfW, fz, { bevel: 0 });
      this.box(mat, halfW * 2 + t, t, t, x, y - halfW, fz, { bevel: 0 });
      this.box(mat, t, halfW * 2 + t, t, x + halfW, y, fz, { bevel: 0 });
      this.box(mat, t, halfW * 2 + t, t, x - halfW, y, fz, { bevel: 0 });
      // diagonals on the two side faces
      const dl = Math.hypot(halfW * 2, len / frames) * 0.98;
      const ang = Math.atan2(halfW * 2, len / frames);
      this.box(mat, t * 0.8, t * 0.8, dl, x + halfW, y, fz, { bevel: 0, rot: [ang * (i % 2 ? 1 : -1), 0, 0] });
      this.box(mat, t * 0.8, t * 0.8, dl, x - halfW, y, fz, { bevel: 0, rot: [ang * (i % 2 ? -1 : 1), 0, 0] });
    }
  }

  /** Pipe run along Z on a flank, with clamps. */
  pipe(mat, r, len, x, y, z, { clamps = 3 } = {}) {
    this.drum(mat, r, len, z, { x, y, sides: 10 });
    for (let i = 0; i < clamps; i++) {
      const cz = z - len / 2 + (i + 0.5) * (len / clamps);
      this.drum('dark', r * 1.3, r * 1.2, cz, { x, y, sides: 10 });
    }
  }

  /**
   * Applique armour belt: a raised bevelled slab with a thinner, wider
   * backing plate and chamfered edge strips. `face` is the outward axis.
   */
  armorBelt(w, h, d, x, y, z, { face = '+y', mat = 'hull' } = {}) {
    this.box('trim', w * 1.06, h * 0.45, d * 1.04, x, y, z, { bevel: 0.2 });
    this.box(mat, w, h, d, x, y + (face === '+y' ? h * 0.3 : face === '-y' ? -h * 0.3 : 0), z, { bevel: 0.45 });
  }

  /**
   * Greeble field: scatter small machinery boxes/tanks on a plane.
   *   axis: '+y' | '-y' | '+x' | '-x'; level: coordinate of the plane;
   *   uRange/zRange: extents in the in-plane axis and along Z
   */
  greebles(axis, level, uRange, zRange, count, sMin, sMax, { mats = ['trim', 'dark', 'hull'] } = {}) {
    const rnd = this.rnd;
    for (let i = 0; i < count; i++) {
      const u = uRange[0] + rnd() * (uRange[1] - uRange[0]);
      const zc = zRange[0] + rnd() * (zRange[1] - zRange[0]);
      const a = sMin + rnd() * (sMax - sMin);
      const b = sMin + rnd() * (sMax - sMin) * 1.6;
      const hgt = a * (0.35 + rnd() * 0.8);
      const mat = mats[Math.floor(rnd() * mats.length)];
      const tank = rnd() < 0.22;
      const sign = axis[0] === '+' ? 1 : -1;
      const vertical = axis[1] === 'y';
      const x = vertical ? u : level + sign * hgt * 0.4;
      const y = vertical ? level + sign * hgt * 0.4 : u;
      if (tank) {
        // cylindrical tank lying along Z with end caps
        this.drum(mat, a * 0.5, b, zc, { x, y, sides: 10 });
        this.sphere(mat, a * 0.5, x, y, zc + b / 2, { ws: 10, hs: 6 });
        this.sphere(mat, a * 0.5, x, y, zc - b / 2, { ws: 10, hs: 6 });
      } else if (vertical) {
        this.box(mat, a, hgt, b, x, y, zc, { bevel: 0.25, seg: 1 });
      } else {
        this.box(mat, hgt, a, b, x, y, zc, { bevel: 0.25, seg: 1 });
      }
    }
  }

  /**
   * Coilgun battery: 'coilgun' (Coilgun Battery Mk3) or 'heavy' (Heavy
   * Coilgun Battery Mk3), drawn at the kind's absolute size. Only the
   * static ring (and a heavy battery's armoured barbette) is baked into
   * the hull; the housing (yaw) and barrels (elevation) are built as a rig
   * in build() so the guns can track a target. angle rotates the mount
   * about Z (0 = dorsal, PI = ventral, -PI/2 = port +X, PI/2 = starboard).
   * Rigs take weapon slots in call order (after a spinal mount).
   */
  turret(kind, x, y, z, angle = 0) {
    const s = this.ws(kind);
    const heavy = kind === 'heavy';
    const P = this.xf(x, y, z, 0, 0, angle);
    // A heavy battery stands on an armoured barbette that lifts the whole
    // mount (the rig origin moves with it, so muzzles stay consistent)
    const hb = heavy ? s * 0.26 : 0;
    if (heavy) {
      const bar = s * 0.62;
      this.drum('hull', bar, hb + s * 0.12, 0, { sides: 16, parent: this.xf(0, (hb - s * 0.12) / 2, 0, Math.PI / 2, 0, 0, P) });
      this.drum('paint', bar * 1.02, hb * 0.32, 0, { sides: 16, parent: this.xf(0, hb * 0.55, 0, Math.PI / 2, 0, 0, P) });
      this.drum('trim', bar * 1.06, s * 0.05, 0, { sides: 16, parent: this.xf(0, hb, 0, Math.PI / 2, 0, 0, P) });
    }
    const Q = this.xf(0, hb, 0, 0, 0, 0, P);
    this.drum('dark', s * 0.62, s * 0.14, 0, { sides: 14, parent: this.xf(0, s * 0.07, 0, Math.PI / 2, 0, 0, Q) });
    this.drum('trim', s * 0.5, s * 0.1, 0, { sides: 14, parent: this.xf(0, s * 0.17, 0, Math.PI / 2, 0, 0, Q) });
    this.turretRigs.push({ s, kind, x: x - Math.sin(angle) * hb, y: y + Math.cos(angle) * hb, z, angle });
  }

  /**
   * PD laser turret: a static barbette baked into the hull and a ball
   * mount (built in build()) that slews onto the torpedo or slug its
   * dwell segment is burning. Rigs are named pd_0, pd_1... in call order,
   * matching the recording's pd_fired turret_name.
   */
  pd(x, y, z, angle = 0) {
    const s = this.ws('pd');
    const P = this.xf(x, y, z, 0, 0, angle);
    this.drum('trim', s * 0.42, s * 0.1, 0, { sides: 12, parent: this.xf(0, s * 0.05, 0, Math.PI / 2, 0, 0, P) });
    this.drum('dark', s * 0.34, s * 0.08, 0, { sides: 12, parent: this.xf(0, s * 0.13, 0, Math.PI / 2, 0, 0, P) });
    this.pdRigs.push({ s, x, y, z, angle });
  }

  /** Sensor mast with a lit tip and a couple of dishes. */
  mast(h, x, y, z, lean = 0) {
    const P = this.xf(x, y, z, 0, 0, lean);
    this.drum('dark', h * 0.035, h, 0, { sides: 6, parent: this.xf(0, h / 2, 0, Math.PI / 2, 0, 0, P) });
    this.box('trim', h * 0.16, h * 0.06, h * 0.16, 0, h * 0.55, 0, { bevel: 0.2, parent: P });
    this.cone('trim', h * 0.11, h * 0.05, 0, h * 0.78, h * 0.06, { rot: [-0.4, 0, 0], sides: 10, parent: P });
    this.box('dark', h * 0.05, h * 0.05, h * 0.05, 0, h * 1.02, 0, { bevel: 0, parent: P });
    // Red anti-collision beacon on the mast top
    this.lamp(x - Math.sin(lean) * h * 1.06, y + Math.cos(lean) * h * 1.06, z,
      { color: 0xff2a1a, kind: 'beacon', size: this.lampS * 1.3, period: 2.3, level: 2.2 });
  }

  /**
   * Comms / sensor boom: an open lattice tower standing off the hull with
   * two phased-array panels, a dish and a beacon - the one thing allowed
   * to break a torch ship's skin (see the class notes below).
   */
  boom(h, x, y, z, { lean = 0 } = {}) {
    const P = this.xf(x, y, z, 0, 0, lean);
    const t = h * 0.035, hw = h * 0.07;
    for (const sz of [-1, 1]) {
      for (const sxx of [-1, 1]) this.box('trim', t, h, t, sxx * hw, h / 2, sz * hw, { bevel: 0, parent: P });
    }
    const n = Math.max(3, Math.round(h / (hw * 3)));
    for (let i = 1; i <= n; i++) {
      const yy = (i / (n + 0.4)) * h;
      this.box('trim', hw * 2 + t, t * 0.8, t * 0.8, 0, yy, hw, { bevel: 0, parent: P });
      this.box('trim', hw * 2 + t, t * 0.8, t * 0.8, 0, yy, -hw, { bevel: 0, parent: P });
      this.box('trim', t * 0.8, t * 0.8, hw * 2 + t, hw, yy, 0, { bevel: 0, parent: P });
      this.box('trim', t * 0.8, t * 0.8, hw * 2 + t, -hw, yy, 0, { bevel: 0, parent: P });
    }
    // phased arrays facing fore and aft, a dish on top, a whip
    this.box('dark', h * 0.34, h * 0.22, h * 0.03, 0, h * 0.62, hw + h * 0.03, { bevel: 0.2, parent: P });
    this.box('dark', h * 0.3, h * 0.2, h * 0.03, 0, h * 0.38, -hw - h * 0.03, { bevel: 0.2, parent: P });
    this.box('trim', h * 0.36, h * 0.025, h * 0.05, 0, h * 0.74, hw + h * 0.03, { bevel: 0, parent: P });
    this.cone('trim', h * 0.16, h * 0.06, 0, h * 0.95, h * 0.04, { rot: [-0.5, 0, 0], sides: 14, parent: P });
    this.drum('barrel', h * 0.01, h * 0.4, 0, { sides: 5, parent: this.xf(hw, h * 1.15, 0, Math.PI / 2, 0, 0, P) });
    this.lamp(x - Math.sin(lean) * h * 1.36, y + Math.cos(lean) * h * 1.36, z,
      { color: 0xff2a1a, kind: 'beacon', size: this.lampS * 1.3, period: 2.3, level: 2.2 });
  }

  /**
   * Flush sensor / comms cluster. A torch ship is a tower along its thrust
   * axis - it has no naval bridge superstructure - so command spaces sit
   * inside the hull and only antennas, dishes and arrays break the skin.
   */
  sensorCluster(s, x, y, z) {
    const P = this.xf(x, y, z);
    this.box('trim', s * 1.3, s * 0.1, s * 1.1, 0, s * 0.05, 0, { bevel: 0.2, parent: P });
    this.box('hull', s * 0.7, s * 0.16, s * 0.6, -s * 0.2, s * 0.15, s * 0.15, { bevel: 0.3, parent: P });
    // main dish on a pedestal, looking forward-up
    this.drum('dark', s * 0.05, s * 0.3, 0, { sides: 6, parent: this.xf(s * 0.3, s * 0.25, -s * 0.3, Math.PI / 2, 0, 0, P) });
    this.cone('trim', s * 0.32, s * 0.12, s * 0.3, s * 0.42, -s * 0.3, { rot: [-1.1, 0, 0], sides: 16, parent: P });
    // phased-array panel, canted
    this.box('dark', s * 0.55, s * 0.05, s * 0.4, -s * 0.35, s * 0.3, -s * 0.3, { bevel: 0.2, rot: [0.5, 0, 0.3], parent: P });
    // whip antennas
    this.drum('barrel', s * 0.012, s * 0.9, 0, { sides: 5, parent: this.xf(s * 0.45, s * 0.55, s * 0.35, Math.PI / 2, 0, 0.2, P) });
    this.drum('barrel', s * 0.012, s * 0.7, 0, { sides: 5, parent: this.xf(-s * 0.5, s * 0.45, s * 0.4, Math.PI / 2, 0, -0.3, P) });
    this.box('accent', s * 0.08, s * 0.03, s * 0.08, -s * 0.2, s * 0.24, s * 0.15, { bevel: 0, parent: P });
  }

  /** One deck window, or (30 %, seeded) an unlit one: crews sleep. */
  _window(w, h, d, x, y, z, rot = null) {
    this.box(this.rnd() < 0.3 ? 'dark' : 'window', w, h, d, x, y, z, { bevel: 0, rot });
  }

  /**
   * Deck windows ringing a drum of `sides` at z: decks stack along the
   * thrust axis, so their windows run around the hull, not along it.
   * Windows are a fixed physical size (WINDOW_S); a wider face carries
   * more of them. Faces are numbered from the first one at or after +X,
   * counter-clockwise (octagon: 0 +X, 2 dorsal, 4 -X, 6 ventral).
   */
  deckRing(r, z, { sides = 8, skipFaces = [], s = this.winS, sx = 1, sy = 1 } = {}) {
    const apothem = r * Math.cos(Math.PI / sides);
    const faceW = 2 * r * Math.sin(Math.PI / sides);
    // face-centre angles of drum(): rotated drums (sides <= 8) put a
    // vertex half a face from the axes, unrotated ones put one on -Y
    const phase = sides <= 8 ? ((sides % 4 === 0) ? 0 : Math.PI / sides) : Math.PI / sides;
    const perFace = Math.max(1, Math.floor(faceW * 0.8 / (s * 3.2)));
    for (let k = 0; k < sides; k++) {
      if (skipFaces.includes(k)) continue;
      const a = phase + k * (2 * Math.PI / sides);
      const dx = Math.cos(a), dy = Math.sin(a);
      const tx = -Math.sin(a), ty = Math.cos(a);
      for (let j = 0; j < perFace; j++) {
        const off = (j - (perFace - 1) / 2) * (faceW * 0.8 / perFace);
        const x = (dx * apothem + tx * off) * sx;
        const y = (dy * apothem + ty * off) * sy;
        this._window(s * 1.5, s * 0.35, s, x, y, z, [0, 0, a - Math.PI / 2]);
      }
    }
  }

  /** Deck windows around a rectangular section (w wide, h tall) at z. */
  deckBand(w, h, z, { top = true, bottom = true, sides = true, s = this.winS } = {}) {
    const pitch = s * 3.4;
    const nx = Math.max(1, Math.floor(w * 0.8 / pitch));
    const ny = Math.max(1, Math.floor(h * 0.75 / pitch));
    for (let i = 0; i < nx; i++) {
      const x = (i - (nx - 1) / 2) * pitch;
      if (top) this._window(s * 1.4, s * 0.35, s, x, h / 2, z);
      if (bottom) this._window(s * 1.4, s * 0.35, s, x, -h / 2, z);
    }
    if (!sides) return;
    for (let j = 0; j < ny; j++) {
      const y = (j - (ny - 1) / 2) * pitch;
      this._window(s * 0.35, s * 1.4, s, w / 2, y, z);
      this._window(s * 0.35, s * 1.4, s, -w / 2, y, z);
    }
  }

  /** Thin band hugging a rectangular section (w x h) at z, in `mat`. */
  frameBand(mat, w, h, z, t, { d = t } = {}) {
    this.box(mat, w + t * 0.6, t, d, 0, h / 2, z, { bevel: 0 });
    this.box(mat, w + t * 0.6, t, d, 0, -h / 2, z, { bevel: 0 });
    this.box(mat, t, h + t * 0.6, d, w / 2, 0, z, { bevel: 0 });
    this.box(mat, t, h + t * 0.6, d, -w / 2, 0, z, { bevel: 0 });
  }

  /** Thin lit accent band hugging a rectangular section (w x h) at z. */
  opsBand(w, h, z, t) {
    this.frameBand('accent', w, h, z, t);
  }

  /** Lit ops-deck band: a thin accent ring with a window ring beside it. */
  opsDeck(r, z, { sides = 8 } = {}) {
    this.rib('accent', r * 1.005, this.winS * 0.5, z + this.winS * 1.2, { sides });
    this.deckRing(r, z, { sides });
  }

  /** Row of lit windows. axis 'x' spaces along Z on an X-facing wall. */
  windows(n, x, y, z, spacing, s, axis = 'x') {
    for (let i = 0; i < n; i++) {
      if (axis === 'x') this.box('window', s * 0.3, s, s * 0.6, x, y, z + i * spacing, { bevel: 0 });
      else this.box('window', s * 0.6, s * 0.3, s, x, y, z + i * spacing, { bevel: 0 });
    }
  }

  /** A nav lamp, drawn by HullLights. color 'faction' = the livery accent. */
  lamp(x, y, z, { color = 0xffffff, kind = 'steady', size = null, period = 1.6, phase = null, level = 1 } = {}) {
    this.lamps.push({
      pos: new THREE.Vector3(x, y, z), color, kind, size: size ?? this.lampS,
      period, phase: phase ?? this.rnd(), level
    });
  }

  /** Port (red, +X) / starboard (green, -X) running lights at a station. */
  runningLights(x, y, z) {
    this.lamp(x, y, z, { color: 0xff2020, size: this.lampS * 1.1, level: 1.5, phase: 0.1 });
    this.lamp(-x, y, z, { color: 0x20ff50, size: this.lampS * 1.1, level: 1.5, phase: 0.6 });
  }

  /** White anti-collision strobe (double flash). */
  strobe(x, y, z, phase = 0) {
    this.lamp(x, y, z, { kind: 'strobe', size: this.lampS * 1.8, period: 1.7, phase, level: 4.0 });
  }

  /** Faction-coloured formation lamp. */
  formationLamp(x, y, z) {
    this.lamp(x, y, z, { color: 'faction', size: this.lampS, level: 1.2 });
  }

  /**
   * Painted livery band around a rectangular section (w x h) at z: four
   * plates standing just proud of the faces, stopping short of the
   * bevelled corners.
   */
  paintBand(w, h, z, t, { top = true, bottom = true, sides = true } = {}) {
    const e = Math.min(w, h) * 0.012 + 0.004;
    if (top) this.box('paint', w * 0.84, e * 2, t, 0, h / 2, z, { bevel: 0.3, seg: 1 });
    if (bottom) this.box('paint', w * 0.84, e * 2, t, 0, -h / 2, z, { bevel: 0.3, seg: 1 });
    if (sides) {
      this.box('paint', e * 2, h * 0.8, t, w / 2, 0, z, { bevel: 0.3, seg: 1 });
      this.box('paint', e * 2, h * 0.8, t, -w / 2, 0, z, { bevel: 0.3, seg: 1 });
    }
  }

  /**
   * Faction livery pattern, readable without colour: alpha wears twin
   * pinstripes, beta one wide band. draw(zCentre, width) paints a band.
   */
  livery(z, t, draw) {
    if (this.faction === 'beta') draw(z, t);
    else {
      draw(z + t * 0.32, t * 0.22);
      draw(z - t * 0.32, t * 0.22);
    }
  }

  /**
   * Painted dorsal stripe along the spine (top-down and chase-camera
   * livery), on a deck at height y from z0 to z1. Follows the faction
   * pattern across the beam: alpha two pinstripes, beta one wide stripe.
   */
  spineStripe(y, z0, z1, width, { x = 0 } = {}) {
    const e = 0.012 * this.k;
    const len = z1 - z0, zc = (z0 + z1) / 2;
    if (this.faction === 'beta') this.box('paint', width, e, len, x, y + e / 2, zc, { bevel: 0.3, seg: 1 });
    else {
      for (const sx of [-1, 1]) this.box('paint', width * 0.24, e, len, x + sx * width * 0.32, y + e / 2, zc, { bevel: 0.3, seg: 1 });
    }
  }

  /**
   * Hull code decal (white on the plating) on a flat face, washed by a
   * hull floodlight just aft of it so it reads on the shadowed side too.
   * flood 'fore' puts the lamp just ahead of the code, shining aft (for a
   * code in a recess whose aft end is walled in), false drops it.
   * hull_design.test.js ray-tests every placement: nothing may stand
   * between the outside and the decal, and plating must be right behind it.
   */
  marking(x, y, z, h, face, { flood = true } = {}) {
    const off = this.W * 0.004;
    const ox = face === '+x' ? off : face === '-x' ? -off : 0;
    const oy = face === '+y' ? off : face === '-y' ? -off : 0;
    const w = h * 4;
    this.markings.push({ x: x + ox, y: y + oy, z, w, h, face });
    if (flood === 'fore') this.flood(x, y, z + w * 0.62, face, w * 1.45, h * 2.4, -1);
    else if (flood) this.flood(x, y, z - w * 0.62, face, w * 1.45, h * 2.4, 1);
  }

  /** Both flanks at once. */
  markings2(x, y, z, h, opts) {
    this.marking(x, y, z, h, '+x', opts);
    this.marking(-x, y, z, h, '-x', opts);
  }

  /**
   * Hull floodlight: a small warm lamp on the face `face` at (x, y, z)
   * throwing a pool of light `len` along the face toward `dir` (+1 = the
   * nose) and `width` across. Expanse ships light their own flanks; here
   * it keeps codes and livery readable in shadow. Drawn by HullMarkings.
   */
  flood(x, y, z, face, len, width, dir = 1) {
    const n = { '+x': [1, 0], '-x': [-1, 0], '+y': [0, 1], '-y': [0, -1] }[face] || [0, 1];
    const lift = this.lampS * 0.6;
    this.floods.push({ x, y, z, face, len, width, dir });
    this.lamp(x + n[0] * lift, y + n[1] * lift, z, { color: 0xfff0d8, size: this.lampS * 0.8, level: 1.1 });
  }

  /**
   * RCS quad at a station: four thruster blocks around the hull at z,
   * each with a radial, two tangential and two axial nozzles. rx/ry are
   * the skin offsets of the blocks; diag puts them on the corners
   * (X pattern) instead of the faces (+ pattern). Every nozzle exit is
   * recorded in rcsMounts {pos, dir} for the RCS jet effect.
   */
  rcsStation(z, rx, ry, s, { diag = false } = {}) {
    const dirs = diag
      ? [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => [a * Math.SQRT1_2, b * Math.SQRT1_2])
      : [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const up = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion();
    for (const [nx, ny] of dirs) {
      const n = new THREE.Vector3(nx, ny, 0);
      const tan = new THREE.Vector3(-ny, nx, 0);
      const base = diag
        ? new THREE.Vector3(Math.sign(nx) * rx, Math.sign(ny) * ry, z)
        : new THREE.Vector3(nx * rx, ny * ry, z);
      // Block: local +Y outward, +Z along the hull
      const P = new THREE.Matrix4().makeBasis(new THREE.Vector3(ny, -nx, 0), n, new THREE.Vector3(0, 0, 1)).setPosition(base);
      this.box('hull', s, s * 0.5, s * 1.25, 0, s * 0.25, 0, { bevel: 0.3, seg: 1, parent: P });
      this.box('trim', s * 1.15, s * 0.12, s * 1.4, 0, s * 0.04, 0, { bevel: 0.2, seg: 1, parent: P });
      const nozzles = [
        [n, base.clone().addScaledVector(n, s * 0.5)],
        [tan, base.clone().addScaledVector(n, s * 0.27).addScaledVector(tan, s * 0.5)],
        [tan.clone().negate(), base.clone().addScaledVector(n, s * 0.27).addScaledVector(tan, -s * 0.5)],
        [new THREE.Vector3(0, 0, 1), base.clone().addScaledVector(n, s * 0.27).setZ(z + s * 0.625)],
        [new THREE.Vector3(0, 0, -1), base.clone().addScaledVector(n, s * 0.27).setZ(z - s * 0.625)]
      ];
      for (const [dir, exit] of nozzles) {
        // Bell: throat on the block face, mouth at the exit, opening outward
        const len = s * 0.2;
        const g = new THREE.CylinderGeometry(s * 0.13, s * 0.055, len, 8, 1, true);
        g.translate(0, len / 2, 0);
        q.setFromUnitVectors(up, dir);
        const m = new THREE.Matrix4().compose(exit.clone().addScaledVector(dir, -len), q, new THREE.Vector3(1, 1, 1));
        this.add('barrel', g, m);
        this.rcsMounts.push({ pos: exit.clone(), dir: dir.clone() });
      }
    }
  }

  /**
   * Torpedo launch door: the geometry added through the returned kit is
   * built as its own mesh that slides by `slide` (hull-local) when open.
   * Launches are dealt to door banks round robin.
   */
  launchDoor(bank, slide) {
    const kit = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    this.doors.push({ kit, bank, slide });
    return kit;
  }

  /**
   * One torpedo round of the magazine, visible on the hull until it is
   * fired (HullRigs.updateMagazine, on battle time). Launches are dealt to
   * banks round robin (one bank per launcher, as the doors were) and
   * empty a bank's rounds in call order.
   *   kind 'rack': an external round, body along +Z (corvette rails)
   *   kind 'cell': a VLS hatch at pos, its local +Y the bank's `up`; with
   *                `hinge` it swings up about its local +X edge when its
   *                round fires (HullRigs.HATCH_OPEN: just past upright,
   *                leaning outboard over the coaming) and stays open
   * `matrix` places the round/hatch (hull-local), `s` is the cell size.
   */
  magazineRound(kind, bank, matrix, s, { hinge = false } = {}) {
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
    matrix.decompose(pos, quat, scl);
    this.magazine.push({ kind, bank, pos, quat, s, hinge });
  }

  /**
   * Hinged heat exchanger: graphite fin banks, headers and raised heat
   * pipes, on a painted cassette. Stowed, the wing telescopes back into
   * the cassette (Heat.js scales it along its span) instead of folding up
   * into a sail.
   */
  radiator(span, len, x, y, z, angle, thickness = null) {
    const t = thickness ?? 0.03 * this.k;
    // Local +Y is the deployed span. A mirrored mount must point outboard.
    if (-Math.sin(angle) * x + Math.cos(angle) * y < 0) angle += Math.PI;
    const wing = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    const bays = Math.max(3, Math.min(7, Math.round(len / span * 3)));
    const bayLen = len / bays;
    const rail = t * 1.8;
    // Separate thin fin cassettes leave real shadow gaps between banks.
    for (let b = 0; b < bays; b++) {
      const bz = -len / 2 + (b + 0.5) * bayLen;
      const g = new THREE.BoxGeometry(t, span * 0.9, bayLen * 0.92);
      const uv = g.attributes.uv;
      for (let i = 0; i < uv.count; i++) {
        if (i < 8) uv.setXY(i, uv.getX(i), 1 - uv.getY(i));
        else uv.setXY(i, 0, 0);
      }
      wing.add('radiator', g, wing.xf(0, span * 0.51, bz));
      // Raised coolant tubes on both faces, running from hot root to cool tip.
      for (const side of [-1, 1]) {
        for (const dz of [-0.3, 0, 0.3]) {
          wing.box('barrel', t * 0.65, span * 0.91, t * 0.65,
            side * t * 0.65, span * 0.51, bz + dz * bayLen, { bevel: 0 });
        }
      }
      // (all structure in 'barrel': two draws per wing, panel + frame)
      for (let j = 1; j < 12; j++) {
        wing.box('barrel', t * 1.5, span * 0.008, bayLen * 0.9,
          0, span * (0.06 + j * 0.075), bz, { bevel: 0 });
      }
    }
    for (let b = 0; b <= bays; b++) {
      wing.box('barrel', rail, span, rail, 0, span / 2, -len / 2 + b * bayLen, { bevel: 0 });
    }
    for (const sy of [0.035, 0.98]) {
      wing.drum('barrel', rail, len * 1.02, 0, { y: span * sy, sides: 8 });
    }
    const root = new THREE.Group();
    root.name = 'radiator_wing';
    root.position.set(x, y, z);
    root.rotation.z = angle;
    root.userData.deployedAngle = angle;
    root.userData.foldDirection = -Math.sign(Math.sin(angle) || 1) * (y < 0 ? -1 : 1);
    root.userData.stow = 'telescope';
    root.userData.stowScale = RADIATOR_STOW;
    // Painted cassette the wing retracts into, hinge blocks and the supply
    // manifold joining the exchanger to the hull: faction livery on the
    // aft half where the chase camera sees it.
    const P = this.xf(x, y, z, 0, 0, angle);
    const ch = span * (RADIATOR_STOW + 0.04);
    this.box('paint', t * 6, ch, len * 1.06, 0, ch / 2 - t, 0, { bevel: 0.3, parent: P });
    this.box('trim', t * 7, t * 1.6, len * 1.1, 0, ch, 0, { bevel: 0.2, parent: P });
    this.drum('barrel', t * 2.4, len * 0.96, 0, { sides: 10, parent: this.xf(0, -t * 0.6, 0, 0, 0, 0, P) });
    this.radiators.push({ mesh: root, kit: wing, span, len });
    return root;
  }

  /**
   * Drive section around the engine bells, shaped per class (this.drive):
   *   torch - small hulls: open truss thrust frame, shadow shield and a
   *           bare converter core with magnetic-nozzle rings - a torch
   *           with a ship attached, which the 65 m corvette really is
   *   box   - box-section hulls: armoured engine block flush with the
   *           aft section (w x h)
   *   drum  - drum hulls: octagonal skirt flush with the drum radius r
   * Bells sit where SceneManager.getEngineConfig puts the plumes. Every
   * shape carries the faction livery band and an accent rim on the thrust
   * frame, the identity the chase camera sees.
   */
  driveSection(engineConfig) {
    const L = this.L, W = this.W;
    const d = this.drive || { shape: 'drum', r: W * 0.5 };
    const n = engineConfig.count;
    const bellR = W * (n <= 2 ? 0.16 : n <= 4 ? 0.125 : 0.09);
    let ex = 0, ey = 0;
    for (const p of engineConfig.positions) {
      ex = Math.max(ex, Math.abs(p.x) + bellR);
      ey = Math.max(ey, Math.abs(p.y) + bellR);
    }
    const rcsS = this.rcsS;
    if (d.shape === 'torch') {
      const rS = Math.max(Math.hypot(ex, ey) * 1.08, d.r ?? 0);
      const zS = d.z ?? -L * 0.36;
      // Shadow shield between crew and converter, livery on its rim
      this.drum('trim', rS, L * 0.022, zS, { sides: 12 });
      this.livery(zS, L * 0.022, (zc, t) => this.rib('paint', rS * 1.02, t, zc, { sides: 12 }));
      this.drum('hull', rS * 0.86, L * 0.02, zS - L * 0.018, { sides: 12, rTop: rS * 0.95 });
      // Converter core on its truss, coolant loops, warning ring
      const zC = (zS - L * 0.49) / 2;
      const coreLen = zS - L * 0.03 - (-L * 0.475);
      this.drum('hull', rS * 0.36, coreLen, zC - L * 0.01, { sides: 10 });
      this.drum('accent', rS * 0.375, L * 0.008, zC, { sides: 10 });
      for (const sx of [-1, 1]) this.pipe('barrel', rS * 0.05, coreLen * 0.8, sx * rS * 0.48, 0, zC, { clamps: 2 });
      this.truss('trim', rS * 0.62, zS - (-L * 0.488), (zS - L * 0.488) / 2);
      // Thrust frame plate
      this.drum('trim', rS * 0.95, L * 0.018, -L * 0.488, { sides: 12 });
      this.annulus('accent', rS * 0.86, rS * 0.95, -L * 0.4975, { sides: 12 });
      this.rcsStation(zS - L * 0.045, rS * 0.6, rS * 0.6, rcsS, { diag: true });
      this.strobe(0, rS * 1.02, zS, 0.5);
      this.strobe(0, -rS * 1.02, zS, 0.55);
    } else if (d.shape === 'box') {
      const w = Math.max(d.w, ex * 2.1), h = Math.max(d.h, ey * 2.1);
      this.drum('trim', Math.min(w, h) * 0.32, L * 0.16, -L * 0.355, { sides: 8 });
      this.box('hull', w, h, L * 0.12, 0, 0, -L * 0.425, { bevel: 0.16 });
      this.frameBand('trim', w * 1.01, h * 1.01, -L * 0.372, L * 0.012);
      this.frameBand('trim', w * 1.01, h * 1.01, -L * 0.478, L * 0.012);
      this.livery(-L * 0.425, L * 0.06, (zc, t) => this.paintBand(w, h, zc, t));
      this.box('trim', w * 0.95, h * 0.95, L * 0.02, 0, 0, -L * 0.49, { bevel: 0.2 });
      this.frameBand('accent', w * 0.955, h * 0.955, -L * 0.4985, Math.min(w, h) * 0.02, { d: L * 0.004 });
      this.rcsStation(-L * 0.425, w * 0.5, h * 0.5, rcsS, { diag: true });
      this.strobe(0, h * 0.53, -L * 0.47, 0.5);
      this.strobe(0, -h * 0.53, -L * 0.47, 0.55);
    } else {
      const r = Math.max(d.r, Math.hypot(ex, ey) * 1.05);
      const sides = d.sides ?? 8;
      // Continuous pressure trunk overlaps both the aft hull and the skirt.
      this.drum('trim', r * 0.5, L * 0.16, -L * 0.355, { sides: 8 });
      this.drum('hull', r, L * 0.1, -L * 0.44, { rTop: r * 0.98, sides });
      this.rib('trim', r * 1.03, L * 0.012, -L * 0.396, { sides });
      this.rib('trim', r * 1.03, L * 0.012, -L * 0.484, { sides });
      this.livery(-L * 0.44, L * 0.05, (zc, t) => this.rib('paint', r * 1.015, t, zc, { sides }));
      this.drum('trim', r * 0.95, L * 0.02, -L * 0.49, { sides });
      this.annulus('accent', r * 0.86, r * 0.95, -L * 0.5005, { sides });
      const rr = r * Math.cos(Math.PI / sides) * Math.SQRT1_2;
      this.rcsStation(-L * 0.44, rr, rr, rcsS, { diag: true });
      this.strobe(0, r * 1.04, -L * 0.47, 0.5);
      this.strobe(0, -r * 1.04, -L * 0.47, 0.55);
    }
    for (const p of engineConfig.positions) {
      // bell: open cone, dark, with a stiffening ring and throat collar
      const bell = new THREE.CylinderGeometry(bellR * 0.5, bellR, L * 0.065, 18, 1, true);
      fitCylinderUVs(bell, bellR, L * 0.065, this.rnd, this.texel);
      bell.rotateX(Math.PI / 2);
      this.add('dark', bell, this.xf(p.x, p.y, -L * 0.515));
      this.drum('barrel', bellR * 1.02, L * 0.008, -L * 0.545, { x: p.x, y: p.y, sides: 18, open: true });
      this.drum('trim', bellR * 0.58, L * 0.02, -L * 0.485, { x: p.x, y: p.y, sides: 12 });
      if (d.shape === 'torch') {
        // magnetic nozzle: coil rings standing off the bell
        this.drum('trim', bellR * 1.2, L * 0.006, -L * 0.53, { x: p.x, y: p.y, sides: 18, open: true });
        this.drum('accent', bellR * 1.12, L * 0.003, -L * 0.536, { x: p.x, y: p.y, sides: 18, open: true });
      }
      // inner heat liner: separate animated mesh
      const liner = new THREE.Mesh(
        new THREE.CylinderGeometry(bellR * 0.46, bellR * 0.94, L * 0.06, 18, 1, true),
        null);
      liner.geometry.rotateX(Math.PI / 2);
      liner.position.set(p.x, p.y, -L * 0.515);
      this.bellHeat.push(liner);
      // feed pipes from the skirt to the bell
      this.drum('barrel', bellR * 0.12, L * 0.05, -L * 0.465, { x: p.x * 0.9, y: p.y * 0.9, sides: 6 });
    }
  }

  // -- assembly -------------------------------------------------------------
  /** Merge the parts into one mesh per material under `group`. */
  _mergeInto(group, mats) {
    const byMat = new Map();
    for (const p of this.parts) {
      if (!byMat.has(p.mat)) byMat.set(p.mat, []);
      byMat.get(p.mat).push(p.geom);
    }
    for (const [key, geoms] of byMat) {
      if (!mats[key]) { console.error('hull material missing:', key); continue; }
      const merged = BufferGeometryUtils.mergeGeometries(geoms, false);
      if (!merged) { console.error('hull merge failed for', key); continue; }
      const mesh = new THREE.Mesh(merged, mats[key]);
      mesh.name = `hull_${key}`;
      group.add(mesh);
    }
  }

  /**
   * Movable turret: mount (static, on the hull) > yaw > pitch (barrels).
   * Both kinds share the muzzle geometry Projectiles assumes (yaw 0.2 s,
   * pitch pivot (0, 0.32 s, 0.15 s), barrel tips 1.4 s out); a heavy
   * battery is bigger (s), wears an armoured glacis and coil sleeves.
   */
  _buildTurretRig(r, mats) {
    const s = r.s;
    const heavy = r.kind === 'heavy';
    const mount = new THREE.Group();
    mount.position.set(r.x, r.y, r.z);
    mount.rotation.z = r.angle;
    const yaw = new THREE.Group();
    yaw.position.y = s * 0.2;
    mount.add(yaw);
    const yk = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    yk.box('trim', s * 0.9, s * 0.1, s * 0.62, 0, s * 0.05, -s * 0.1, { bevel: 0.2 });
    if (heavy) {
      // armoured gunhouse: sloped glacis, raised rangefinder ears
      yk.box('hull', s * 0.96, s * 0.4, s * 0.86, 0, s * 0.26, -s * 0.1, { bevel: 0.3 });
      yk.taperBox('hull', s * 0.96, s * 0.4, s * 0.8, s * 0.22, s * 0.2, 0, s * 0.24, s * 0.33);
      yk.box('paint', s * 0.5, s * 0.03, s * 0.5, 0, s * 0.47, -s * 0.18, { bevel: 0.2, seg: 1 });
      for (const ex of [-1, 1]) yk.box('trim', s * 0.14, s * 0.12, s * 0.22, ex * s * 0.52, s * 0.42, -s * 0.2, { bevel: 0.3 });
    } else {
      yk.box('hull', s * 0.82, s * 0.42, s * 0.9, 0, s * 0.28, -s * 0.05, { bevel: 0.35 });
      yk.box('paint', s * 0.4, s * 0.03, s * 0.4, 0, s * 0.5, -s * 0.15, { bevel: 0.2, seg: 1 });
      yk.box('trim', s * 0.3, s * 0.16, s * 0.3, s * 0.3, s * 0.55, -s * 0.2, { bevel: 0.3 });
    }
    yk._mergeInto(yaw, mats);
    const pitch = new THREE.Group();
    pitch.position.set(0, s * 0.32, s * 0.15);
    yaw.add(pitch);
    const pk = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    pk.box('dark', s * (heavy ? 0.7 : 0.6), s * (heavy ? 0.34 : 0.3), s * 0.28, 0, 0, s * 0.12, { bevel: 0.25 });   // mantlet
    pk._mergeInto(pitch, mats);
    // Twin barrels, each its own group so the one that fired recoils
    // (HullRigs) and the round leaves its muzzle (Projectiles)
    const barrels = TWIN_BARREL_X.map((bx) => {
      const g = new THREE.Group();
      g.name = 'turret_barrel';
      g.position.x = bx * s;
      const bk = new HullKit({ length: this.L, width: this.W }, pk.rnd() * 1e6);
      bk.drum('barrel', s * (heavy ? 0.07 : 0.06), s * 1.25, s * 0.78, { sides: 8 });
      bk.drum('barrel', s * (heavy ? 0.095 : 0.085), s * 0.14, s * 1.33, { sides: 8 });   // muzzle brake
      bk.drum('barrel', s * 0.09, s * 0.12, s * 0.45, { sides: 8 });                     // recoil collar
      if (heavy) {
        // coil sleeves down the barrel (same material: one draw per barrel)
        for (const cz of [0.62, 0.86, 1.1]) bk.drum('barrel', s * 0.1, s * 0.07, s * cz, { sides: 10 });
      }
      bk._mergeInto(g, mats);
      pitch.add(g);
      return g;
    });
    return { mount, yaw, pitch, barrels, s, kind: r.kind, yawAngle: 0, pitchAngle: 0.12, pitchZ: pitch.position.z, slot: r.slot };
  }

  /** PD ball mount: one mesh (sphere, emitter housing, lens barrel). */
  _buildPdRig(r, mats, name) {
    const s = r.s;
    const mount = new THREE.Group();
    mount.position.set(r.x, r.y, r.z);
    mount.rotation.z = r.angle;
    const bk = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    bk.sphere('hull', s * 0.3, 0, 0, 0, { ws: 14, hs: 10 });
    bk.box('hull', s * 0.2, s * 0.2, s * 0.32, 0, 0, s * 0.24, { bevel: 0.35, seg: 1 });
    bk.drum('hull', s * 0.05, s * 0.36, s * 0.5, { sides: 8 });
    bk.drum('hull', s * 0.08, s * 0.05, s * 0.68, { sides: 10 });
    bk.box('hull', s * 0.14, s * 0.06, s * 0.14, s * 0.2, s * 0.2, -s * 0.05, { bevel: 0.3, seg: 1 });
    const geoms = bk.parts.map(p => p.geom);
    const ball = new THREE.Mesh(BufferGeometryUtils.mergeGeometries(geoms, false), mats.hull);
    for (const g of geoms) g.dispose();
    ball.name = `hull_${name}`;
    ball.position.y = s * 0.3;
    ball.rotation.order = 'YXZ';
    mount.add(ball);
    return { name, mount, ball, s, yawAngle: 0, pitchAngle: 0.4, phase: this.rnd() };
  }

  /**
   * Torpedo magazine meshes: rack rounds and VLS hatches are instanced
   * (one draw call each), status lamps one more. HullRigs empties them on
   * battle time from the torpedo_launched events.
   */
  _buildMagazine(group, mats, faction) {
    if (!this.magazine.length) return null;
    const entries = this.magazine;
    const s = entries[0].s;
    const kind = entries[0].kind;
    let body, lamp = null;
    const n = entries.length;
    if (kind === 'rack') {
      body = new THREE.InstancedMesh(rackRoundGeometry(s), rackRoundMaterial(), n);
    } else {
      const g = new THREE.BoxGeometry(s * 0.9, s * 0.1, s * 0.9);
      body = new THREE.InstancedMesh(g, mats.hull, n);
      const lg = new THREE.BoxGeometry(s * 0.22, s * 0.04, s * 0.22);
      lg.translate(s * 0.28, s * 0.07, -s * 0.28);
      lamp = new THREE.InstancedMesh(lg, new THREE.MeshBasicMaterial({ color: 0xffffff }), n);
      lamp.name = 'hull_magazine_lamp';
    }
    body.name = kind === 'rack' ? 'hull_magazine_rack' : 'hull_magazine_hatch';
    const base = [];
    const m = new THREE.Matrix4();
    const one = new THREE.Vector3(1, 1, 1);
    const loaded = new THREE.Color(factionAccent(faction));
    entries.forEach((e, i) => {
      m.compose(e.pos, e.quat, one);
      base.push(m.clone());
      body.setMatrixAt(i, m);
      if (lamp) { lamp.setMatrixAt(i, m); lamp.setColorAt(i, loaded); }
    });
    for (const mesh of [body, lamp]) {
      if (!mesh) continue;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      // decorations for DamageFX's hull raycasts are skipped (instanced)
      group.add(mesh);
    }
    if (lamp) lamp.instanceColor.setUsage(THREE.DynamicDrawUsage);
    const banks = entries.reduce((b, e) => Math.max(b, e.bank + 1), 1);
    // order of each round within its bank (call order)
    const seen = new Array(banks).fill(0);
    const order = entries.map(e => seen[e.bank]++);
    return { kind, entries, order, banks, base, body, lamp, loaded, state: new Float32Array(n).fill(-1) };
  }

  build(group, mats) {
    const faction = mats.faction || this.faction || 'alpha';
    this._mergeInto(group, mats);
    if (this.radiators.length) {
      const tex = getRadiatorTextures();
      const radiatorMaterial = new THREE.MeshStandardMaterial({
        map: tex.map, emissiveMap: tex.emissiveMap, emissive: 0xffffff, emissiveIntensity: 0,
        normalMap: tex.normalMap, normalScale: new THREE.Vector2(0.65, 0.65),
        roughness: 0.82, metalness: 0.25
      });
      for (const r of this.radiators) {
        r.material = radiatorMaterial;
        r.kit._mergeInto(r.mesh, { ...mats, radiator: radiatorMaterial });
        for (const p of r.kit.parts) p.geom.dispose();
        group.add(r.mesh);
      }
    }
    for (const liner of this.bellHeat) {
      liner.material = mats.bellHeat;
      group.add(liner);
    }
    // Coilgun turrets map onto the weapon slots after a spinal mount
    const turrets = [];
    this.turretRigs.forEach((r, i) => {
      r.slot = `weapon_${i + (this.hasSpinal ? 1 : 0)}`;
      const rig = this._buildTurretRig(r, mats);
      group.add(rig.mount);
      turrets.push(rig);
    });
    const pds = this.pdRigs.map((r, i) => {
      const rig = this._buildPdRig(r, mats, `pd_${i}`);
      group.add(rig.mount);
      return rig;
    });
    const doors = this.doors.map(d => {
      const geoms = d.kit.parts.map(p => p.geom);
      const mesh = new THREE.Mesh(BufferGeometryUtils.mergeGeometries(geoms, false), mats.hull);
      for (const g of geoms) g.dispose();
      mesh.name = 'hull_launch_door';
      group.add(mesh);
      return { mesh, bank: d.bank, slide: d.slide, closed: new THREE.Vector3() };
    });
    const magazine = this._buildMagazine(group, mats, faction);
    // Nav lamps: one Points draw call; 'faction' lamps take the livery accent
    const accent = factionAccent(faction);
    for (const l of this.lamps) if (l.color === 'faction') l.color = accent;
    const lights = buildHullLights(group, this.lamps);
    const markings = buildMarkings(group, this.markings, hullCode(group.userData.shipType || this.shipType, group.name), faction);
    const floods = buildFloodlights(group, this.floods);

    group.userData.parts = this.parts;
    group.userData.radiators = this.radiators.map(r => r.mesh);
    group.userData.radiatorMaterials = [...new Set(this.radiators.map(r => r.material))];
    group.userData.bellHeat = this.bellHeat;
    // Lamps are HullLights now; an empty list keeps legacy sprite loops inert
    group.userData.navLights = [];
    group.userData.hullLights = lights;
    group.userData.hullMarkings = markings;
    group.userData.floodlights = floods;
    // `turrets` stays unset so the legacy nearest-enemy loop in
    // SceneManager.updateShip skips; HullRigs drives gunTurrets instead
    group.userData.gunTurrets = turrets;
    group.userData.pdTurrets = pds;
    group.userData.pdMounts = pds.map(r => ({
      name: r.name, pos: r.mount.position.clone().add(new THREE.Vector3(0, r.s * 0.3, 0).applyAxisAngle(new THREE.Vector3(0, 0, 1), r.mount.rotation.z))
    }));
    group.userData.launchDoors = doors;
    group.userData.magazine = magazine;
    group.userData.rcsMounts = this.rcsMounts;
    group.userData.reactorPos = this.reactorPos;
    group.userData.spinalMuzzle = this.spinalMuzzle;
    group.userData.materials = mats;
  }
}

// ---------------------------------------------------------------------------
// Magazine round geometry (instanced): a stowed Trident on its rail
// ---------------------------------------------------------------------------
let _rackMaterial = null;
function rackRoundMaterial() {
  return _rackMaterial ||= new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.7, roughness: 0.38 });
}

/** Torpedo body for a rack: tank, penetrator, ogive nose, bell; length ~s*6. */
function rackRoundGeometry(s) {
  const R = s * 0.5;
  const parts = [];
  const add = (g, color, z) => {
    if (g.index) g = g.toNonIndexed();
    g.rotateX(Math.PI / 2);
    g.translate(0, 0, z);
    const c = new THREE.Color(color);
    const n = g.attributes.position.count;
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.deleteAttribute('uv');
    parts.push(g);
  };
  add(new THREE.CylinderGeometry(R, R, s * 3.6, 14, 1, false), 0xa7afb8, -s * 0.6);
  add(new THREE.CylinderGeometry(R, R, s * 1.2, 14, 1, true), 0x6a737d, s * 1.8);
  add(new THREE.ConeGeometry(R, s * 1.0, 14, 1, true), 0x5e6772, s * 2.9);
  add(new THREE.CylinderGeometry(R * 0.55, R * 0.95, s * 0.5, 12, 1, true), 0x262b31, -s * 2.65);
  add(new THREE.CylinderGeometry(R * 1.04, R * 1.04, s * 0.12, 14, 1, true), 0xc2a978, s * 1.2);
  return BufferGeometryUtils.mergeGeometries(parts, false);
}

// ---------------------------------------------------------------------------
// Class hulls
//
// Expanse rules: the ship is a tower stacked along its thrust axis. Decks
// are perpendicular to the drive, so windows ring the hull; there is no
// naval bridge superstructure - command spaces are inside, marked by a
// lit ops-deck band near the nose, and only sensors break the skin.
//
// Each class has its own frontal outline (the aspect an enemy sees):
// corvette hex needle, frigate sensor dome on a gun head, destroyer chisel,
// cruiser blunt octagon, torpedo cruiser a short shield cap ringed by its
// launcher banks, battlecruiser a vertical knife, battleship a wide flat
// chisel, dreadnought a stepped citadel over its keel spinal. Weapons sit
// where data/fleet_ships.json mounts them (dorsal/ventral/port/starboard,
// fore/aft) and are drawn at their absolute size.
// ---------------------------------------------------------------------------

/** Corvette: torpedo boat - hex hull on a bare torch, 8 rounds on rails. */
function hullCorvette(k) {
  const L = k.L, W = k.W;
  const R = W * 0.36;                          // hex: chines at +-R, flat decks
  const deck = R * Math.cos(Math.PI / 6);      // at +-0.866 R
  k.drum('hull', R, L * 0.64, -L * 0.02, { sides: 6 });
  for (const z of [-L * 0.25, -L * 0.07, L * 0.11]) k.rib('trim', R * 1.03, L * 0.012, z, { sides: 6 });
  k.noseCap({ z0: L * 0.3, len: L * 0.15, r: R, sides: 6, taper: 0.36, layers: 3 });
  // Raised dorsal spine (machinery, PD, sensors) with the livery stripe
  const spineY = deck + W * 0.035;
  k.box('hull', R * 0.92, W * 0.07, L * 0.42, 0, deck, -L * 0.07, { bevel: 0.4 });
  // (the stripe breaks for the hull code, painted on the bare deck)
  const codeZ = -L * 0.045, codeH = W * 0.09;
  k.spineStripe(spineY, -L * 0.27, codeZ - codeH * 2 - L * 0.008, R * 0.62);
  k.spineStripe(spineY, codeZ + codeH * 2 + L * 0.008, L * 0.12, R * 0.62);
  k.box('hull', R * 0.8, W * 0.05, L * 0.3, 0, -deck, -L * 0.08, { bevel: 0.4 });
  // ops deck forward, crew decks aft
  k.rib('accent', R * 1.012, L * 0.006, L * 0.26, { sides: 6 });
  for (const z of [L * 0.2, L * 0.02, -L * 0.16]) k.deckRing(R, z, { sides: 6 });
  // Torpedo rails: 2 x 2 rounds on each chine (the 8-round magazine of the
  // nose launcher), noses forward in hazard-striped cradles. Each round
  // leaves its rail when it is fired (HullRigs), sides alternating.
  const rs = 0.15 * k.k;                       // round diameter (~6.3 rs long)
  const zr = L * 0.17;
  for (const sx of [1, -1]) {
    k.box('trim', rs * 0.5, rs * 2.6, L * 0.2, sx * (R + rs * 0.1), 0, zr, { bevel: 0.3 });
    for (const dz of [-0.07, 0.07]) {
      k.box('hazard', rs * 2.7, rs * 2.5, L * 0.014, sx * (R + rs * 1.25), 0, zr + dz * L, { bevel: 0.2, seg: 1 });
    }
  }
  for (const col of [1.95, 0.8]) {
    for (const row of [0.6, -0.6]) {
      for (const sx of [1, -1]) {
        k.magazineRound('rack', 0, k.xf(sx * (R + rs * col), row * rs, zr), rs);
      }
    }
  }
  // dorsal PD aft (pd_0), sensors, mast
  k.pd(0, spineY, -L * 0.2);
  k.sensorCluster(0.3 * k.k, 0, spineY, L * 0.04);
  k.mast(k.mastH * 0.8, R * 0.28, spineY, -L * 0.125, 0.15);   // aft of the code
  k.greebles('-y', -deck - W * 0.025, [-R * 0.3, R * 0.3], [-L * 0.22, L * 0.04], 6, W * 0.04, W * 0.07);
  k.runningLights(R * 1.02, 0, L * 0.295);
  k.marking(0, spineY, codeZ, codeH, '+y');
  k.rcsStation(L * 0.235, R * 0.72, R * 0.46, k.rcsS, { diag: true });
  k.strobe(0, 0, L * 0.455, 0);
  k.formationLamp(0, spineY + W * 0.04, L * 0.12);
  k.formationLamp(0, -deck - W * 0.03, L * 0.0);
  const rad = k.radiatorSize(2, { maxSpan: W * 1.25, maxLen: L * 0.34 });
  const rz = -L * 0.335 + rad.len / 2;
  k.radiator(rad.span, rad.len, R, 0, rz, -Math.PI / 2);
  k.radiator(rad.span, rad.len, -R, 0, rz, Math.PI / 2);
  k.drive = { shape: 'torch', z: -L * 0.35, r: R };
  k.reactorPos.set(0, 0, -L * 0.42);
}

/** Frigate: escort - gun head with a sensor dome on a truss neck, ribbed drum. */
function hullFrigate(k) {
  const L = k.L, W = k.W;
  const hw = W * 0.62, hh = W * 0.56;
  // Sensor head carrying the nose battery (weapon_0)
  k.box('hull', hw, hh, L * 0.2, 0, 0, L * 0.33, { bevel: 0.3 });
  const domeZ = k.noseCap({ z0: L * 0.43, len: L * 0.06, w: hw, h: hh, tw: 0.78, th: 0.78, layers: 2, tip: 'dome' });
  k.frameBand('trim', hw * 1.02, hh * 1.02, L * 0.235, L * 0.012);
  k.opsBand(hw, hh, L * 0.41, W * 0.012);
  k.deckBand(hw, hh, L * 0.3, { top: false });
  k.box('trim', hw * 0.78, W * 0.05, L * 0.12, 0, hh / 2 + W * 0.02, L * 0.34, { bevel: 0.3 });
  k.turret('coilgun', 0, hh / 2 + W * 0.045, L * 0.35);
  k.mast(k.mastH * 0.75, hw * 0.4, hh / 2, L * 0.25, -0.2);
  // Truss neck
  k.truss('trim', W * 0.16, L * 0.15, L * 0.165);
  k.pipe('barrel', W * 0.03, L * 0.16, 0, -W * 0.1, L * 0.165, { clamps: 2 });
  // Ribbed drum body; hull codes on its flank flats
  const R = W * 0.36, ap = R * Math.cos(Math.PI / 8);
  k.drum('hull', R, L * 0.47, -L * 0.125);
  for (let i = 0; i < 4; i++) k.rib('trim', R * 1.04, L * 0.012, -L * 0.31 + i * L * 0.13);
  for (let i = 0; i < 3; i++) {
    const z = -L * 0.245 + i * L * 0.13;
    k.deckRing(R, z, { skipFaces: i === 2 ? [0, 4, 6] : [6] });
  }
  // Codes in the slot between the radiator cassette lids (front edge at
  // -0.014 L) and the fore rib (0.074 L); the floodlight sits on the rib
  // and shines aft, since the cassette walls the slot in behind
  k.markings2(ap, 0, L * 0.03, W * 0.09, { flood: 'fore' });
  // Comms boom where the old billboard fin stood
  k.boom(k.mastH * 1.5, 0, ap, -L * 0.02);
  k.spineStripe(ap, -L * 0.33, -L * 0.07, R * 0.5);
  // PD aft: dorsal pd_0, ventral pd_1
  k.pd(0, ap, -L * 0.2);
  k.pd(0, -ap, -L * 0.2, Math.PI);
  k.greebles('-y', -ap, [-W * 0.18, W * 0.18], [-L * 0.32, -L * 0.25], 4, W * 0.04, W * 0.08);
  k.greebles('-y', -ap, [-W * 0.18, W * 0.18], [-L * 0.14, L * 0.08], 5, W * 0.04, W * 0.08);
  k.sensorCluster(0.28 * k.k, 0, ap, L * 0.06);
  k.runningLights(hw / 2 + 0.01, 0, L * 0.4);
  k.rcsStation(L * 0.28, hw * 0.5, hh * 0.5, k.rcsS, { diag: true });
  // strobe on the crown of the sensor dome (radius 0.85 x the 0.39 hh tip)
  k.strobe(0, 0, domeZ + hh * 0.39 * 0.75 + k.lampS * 0.3, 0);
  k.formationLamp(0, -ap - W * 0.02, -L * 0.1);
  const rad = k.radiatorSize(2, { maxSpan: W * 1.4, maxLen: L * 0.32 });
  k.radiator(rad.span, rad.len, ap, 0, -L * 0.36 + rad.len / 2 + L * 0.01, -Math.PI / 2);
  k.radiator(rad.span, rad.len, -ap, 0, -L * 0.36 + rad.len / 2 + L * 0.01, Math.PI / 2);
  k.drive = { shape: 'torch', z: -L * 0.365, r: R };
  k.reactorPos.set(0, 0, -L * 0.42);
}

/** Destroyer: box sections, chisel bow, spinal coiler along the keel. */
function hullDestroyer(k) {
  const L = k.L, W = k.W;
  k.hasSpinal = true;
  const fw = W * 0.6, fh = W * 0.54, aw = W * 0.7, ah = W * 0.62;
  k.box('hull', fw, fh, L * 0.32, 0, 0, L * 0.2, { bevel: 0.22 });
  k.noseCap({ z0: L * 0.36, len: L * 0.14, w: fw, h: fh, tw: 0.84, th: 0.22, layers: 3, tip: null });
  k.box('hull', aw, ah, L * 0.4, 0, 0, -L * 0.16, { bevel: 0.22 });
  k.box('trim', aw * 1.03, ah * 1.03, L * 0.03, 0, 0, L * 0.04, { bevel: 0.2 }); // frame band
  // Spinal coilgun: rail + breech housing, muzzle proud of the chisel
  // (Projectiles' SPINAL default: y -0.31 W, z 0.56 L)
  k.box('accent', W * 0.07, W * 0.04, L * 0.78, 0, -W * 0.31, L * 0.05, { bevel: 0 });
  k.box('trim', W * 0.15, W * 0.1, L * 0.8, 0, -W * 0.29, L * 0.03, { bevel: 0.3 });
  k.drum('barrel', W * 0.06, L * 0.12, L * 0.5, { y: -W * 0.31, sides: 10 });
  k.drum('dark', W * 0.085, L * 0.03, L * 0.545, { y: -W * 0.31, sides: 10 });   // brake face = muzzle
  k.box('trim', W * 0.22, W * 0.14, L * 0.12, 0, -W * 0.3, -L * 0.3, { bevel: 0.3 });
  k.spinalMuzzle = new THREE.Vector3(0, -W * 0.31, L * 0.56);
  // flank armour + pipes
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.06, W * 0.4, L * 0.3, sx * W * 0.36, W * 0.02, -L * 0.2, { face: sx > 0 ? '+x' : '-x' });
    k.armorBelt(W * 0.05, W * 0.32, L * 0.26, sx * W * 0.31, 0, L * 0.18, { face: sx > 0 ? '+x' : '-x', mat: 'paint' });
    k.pipe('barrel', W * 0.02, L * 0.3, sx * W * 0.37, -W * 0.18, -L * 0.18, { clamps: 3 });
  }
  // deck rings: ops deck at the forward section, crew decks aft
  k.opsBand(fw, fh, L * 0.33, W * 0.012);
  k.deckBand(fw, fh, L * 0.29, { bottom: false });
  k.deckBand(fw, fh, L * 0.08, { bottom: false });
  k.deckBand(aw, ah, -L * 0.07, { bottom: false });
  k.deckBand(aw, ah, -L * 0.26, { bottom: false });
  // Sensors amidships, no tower
  k.sensorCluster(0.36 * k.k, -W * 0.14, ah / 2, -L * 0.04);
  k.mast(k.mastH, W * 0.2, ah / 2, -L * 0.12, 0.1);
  // weapon_1: dorsal battery behind the spinal mount (weapon_0)
  k.turret('coilgun', 0, fh / 2, L * 0.19);
  // PD aft: dorsal pd_0, ventral pd_1 (clear of the spinal rail)
  k.pd(0, ah / 2, -L * 0.24);
  k.pd(W * 0.2, -ah / 2, -L * 0.22, Math.PI);
  k.spineStripe(ah / 2, -L * 0.35, -L * 0.15, W * 0.22);
  k.greebles('+y', ah / 2, [W * 0.14, W * 0.3], [-L * 0.34, -L * 0.14], 5, W * 0.04, W * 0.08);
  k.greebles('+y', ah / 2, [-W * 0.3, -W * 0.14], [-L * 0.34, -L * 0.14], 5, W * 0.04, W * 0.08);
  k.greebles('+y', fh / 2, [-W * 0.25, -W * 0.08], [L * 0.05, L * 0.12], 3, W * 0.03, W * 0.06);
  k.runningLights(W * 0.405, W * 0.1, -L * 0.1);
  k.markings2(W * 0.335, 0, L * 0.15, W * 0.08);
  k.rcsStation(L * 0.3, fw * 0.49, fh * 0.49, k.rcsS, { diag: true });
  k.strobe(0, 0, L * 0.505, 0);
  k.formationLamp(W * 0.3, ah / 2 + 0.02, -L * 0.36);
  k.formationLamp(-W * 0.3, ah / 2 + 0.02, -L * 0.36);
  const rad = k.radiatorSize(2, { maxSpan: W * 1.1, maxLen: L * 0.3 });
  k.radiator(rad.span, rad.len, -W * 0.36, 0, -L * 0.36 + rad.len / 2 + L * 0.015, -Math.PI / 2);
  k.radiator(rad.span, rad.len, W * 0.36, 0, -L * 0.36 + rad.len / 2 + L * 0.015, Math.PI / 2);
  k.drive = { shape: 'box', w: aw * 1.02, h: ah * 1.02 };
  k.reactorPos.set(0, 0, -L * 0.3);
}

/** Cruiser: armoured collar, stepped octagonal bow, ribbed drum, reactor aft. */
function hullCruiser(k, opts = {}) {
  const L = k.L, W = k.W;
  const torpedo = !!opts.torpedo;
  k.hasSpinal = !torpedo;
  const Rc = W * 0.5, R = W * 0.44, ap = R * Math.cos(Math.PI / 8);
  k.drum('trim', Rc, L * 0.16, L * 0.28);
  k.livery(L * 0.285, L * 0.075, (zc, t) => k.rib('paint', Rc * 1.06, t, zc));
  k.opsDeck(Rc, L * 0.215);
  if (torpedo) k.noseCap({ z0: L * 0.36, len: L * 0.09, r: Rc, sides: 8, taper: 0.74, layers: 2 });
  else k.noseCap({ z0: L * 0.36, len: L * 0.14, r: Rc, sides: 8, taper: 0.5, layers: 3 });
  k.drum('hull', R, L * 0.5, -L * 0.02);
  for (let i = 0; i < 5; i++) k.rib('trim', R * 1.035, L * 0.014, -L * 0.24 + i * L * 0.11);
  // Hull codes sit on the flank flats of one deck ring (no windows there)
  const markRing = 1;
  for (let i = 0; i < 4; i++) {
    const skip = torpedo ? [0, 2, 4, 6] : [6];
    if (i === markRing) skip.push(0, 4);
    k.deckRing(R, -L * 0.185 + i * L * 0.11, { skipFaces: skip });
  }
  if (!torpedo) k.markings2(ap, 0, -L * 0.185 + markRing * L * 0.11, W * 0.075);
  // Reactor drum with warning ring and coolant manifolds
  k.drum('hull', W * 0.36, L * 0.14, -L * 0.32);
  k.drum('accent', W * 0.375, L * 0.015, -L * 0.32);
  k.rib('trim', W * 0.38, L * 0.02, -L * 0.26);
  k.rib('trim', W * 0.38, L * 0.02, -L * 0.38);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    k.pipe('barrel', W * 0.022, L * 0.18, Math.cos(a) * W * 0.4, Math.sin(a) * W * 0.4, -L * 0.3, { clamps: 2 });
  }
  if (!torpedo) {
    // Spinal: ventral rail, muzzle under the bow (SPINAL cruiser: -0.42 W, 0.5 L)
    k.box('accent', W * 0.06, W * 0.04, L * 0.7, 0, -W * 0.42, L * 0.05, { bevel: 0 });
    k.box('trim', W * 0.14, W * 0.1, L * 0.72, 0, -W * 0.4, L * 0.03, { bevel: 0.3 });
    k.drum('barrel', W * 0.055, L * 0.12, L * 0.44, { y: -W * 0.42, sides: 10 });
    k.drum('dark', W * 0.08, L * 0.03, L * 0.485, { y: -W * 0.42, sides: 10 });   // brake face = muzzle
    k.spinalMuzzle = new THREE.Vector3(0, -W * 0.42, L * 0.5);
    // weapon_1 port (+X), weapon_2 starboard (-X) batteries
    k.turret('coilgun', ap, 0, L * 0.15, -Math.PI / 2);
    k.turret('coilgun', -ap, 0, L * 0.15, Math.PI / 2);
    // PD aft: dorsal pd_0, ventral pd_1
    k.pd(0, ap, -L * 0.16);
    k.pd(0, -ap, -L * 0.16, Math.PI);
    k.spineStripe(ap, -L * 0.26, -L * 0.2, W * 0.2);
  }
  k.sensorCluster(0.4 * k.k, 0, ap, L * 0.06);
  k.mast(k.mastH * 1.1, W * 0.14, ap, -L * 0.03, -0.12);
  k.greebles('+y', ap, [-W * 0.2, W * 0.2], [-L * 0.24, -L * 0.2], 3, W * 0.035, W * 0.07);
  k.greebles('-y', -ap, [-W * 0.2, W * 0.2], [-L * 0.24, -L * 0.12], 6, W * 0.035, W * 0.07);
  // (the torpedo cruiser's flank banks cover the drum flats: its running
  // lights go on the collar, ahead of the livery band)
  if (torpedo) k.runningLights(Rc * Math.cos(Math.PI / 8) + 0.01, 0, L * 0.34);
  else k.runningLights(ap + 0.01, 0, L * 0.05);
  k.rcsStation(L * 0.172, W * 0.287, W * 0.287, k.rcsS, { diag: true });
  k.strobe(0, 0, L * (torpedo ? 0.452 : 0.502), 0);
  k.formationLamp(0, Rc + 0.02, L * 0.24);
  k.formationLamp(0, -Rc - 0.02, L * 0.24);
  // Four radiator wings in an X around the aft drum
  const rad = k.radiatorSize(4, { maxSpan: W * 0.6, maxLen: L * 0.2 });
  for (const a of [0.785, 2.356, -2.356, -0.785]) {
    k.radiator(rad.span, rad.len, Math.sin(a) * -W * 0.34, Math.cos(a) * W * 0.34, -L * 0.15, a);
  }
  k.drive = { shape: 'drum', r: W * 0.46 };
  k.reactorPos.set(0, 0, -L * 0.32);
}

/**
 * Torpedo cruiser: the cruiser drum wearing four raised VLS banks (one per
 * launcher) of 2 x 6 cells - 48 rounds, the fleet_ships magazine (4 x 12).
 * Each cell's hatch swings up as its round fires and stays open over the
 * dark, empty tube; its status lamp goes from the faction colour to dark.
 * The banks turn the drum's octagon into a cross: a different silhouette
 * from the gun cruiser.
 */
function hullTorpedoCruiser(k) {
  const L = k.L, W = k.W;
  hullCruiser(k, { torpedo: true });
  const R = W * 0.44, ap = R * Math.cos(Math.PI / 8);
  const cs = W * 0.17;                          // cell (hatch) size
  const bank = (z, angle, bi) => {
    const P = k.xf(-Math.sin(angle) * ap, Math.cos(angle) * ap, z, 0, 0, angle);
    const top = W * 0.2;
    k.box('trim', W * 0.62, W * 0.1, L * 0.36, 0, W * 0.03, 0, { bevel: 0.2, parent: P });
    k.box('hull', W * 0.5, W * 0.13, L * 0.34, 0, W * 0.13, 0, { bevel: 0.25, parent: P });
    // armoured chamfers along both long edges: the trapezoid section
    for (const ex of [-1, 1]) {
      k.box('hull', W * 0.09, W * 0.13, L * 0.33, ex * W * 0.27, W * 0.09, 0, { bevel: 0.3, rot: [0, 0, -ex * 0.55], parent: P });
    }
    // hazard-striped coaming round the cell field
    for (const ex of [-1, 1]) k.box('hazard', W * 0.025, W * 0.02, L * 0.33, ex * W * 0.215, top, 0, { bevel: 0, parent: P });
    for (const ez of [-1, 1]) k.box('hazard', W * 0.455, W * 0.02, W * 0.025, 0, top, ez * L * 0.163, { bevel: 0, parent: P });
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 2; col++) {
        const cx = (col - 0.5) * W * 0.2, cz = (2.5 - row) * L * 0.052;
        // the cell under the hatch: dark well, bare gunmetal bore ring
        // round a dark tube mouth (an empty cell reads dark, not lit)
        k.box('dark', cs * 0.92, W * 0.01, cs * 0.92, cx, top - W * 0.004, cz, { bevel: 0, parent: P });
        k.drum('barrel', cs * 0.32, W * 0.004, 0, { sides: 12, parent: k.xf(cx, top + W * 0.002, cz, Math.PI / 2, 0, 0, P) });
        k.drum('dark', cs * 0.24, W * 0.004, 0, { sides: 12, parent: k.xf(cx, top + W * 0.003, cz, Math.PI / 2, 0, 0, P) });
        // the hatch (instanced), turned so its hinge edge (local +X) is
        // outboard: the two columns open away from each other
        k.magazineRound('cell', bi, k.xf(cx, top + cs * 0.06, cz, 0, col ? 0 : Math.PI, 0, P), cs, { hinge: true });
      }
    }
  };
  // Dorsal/ventral banks forward, flank banks further aft: the corners
  // clear each other, and the drum's flank flats stay bare between the
  // flank banks (ending 0.11 L) and the collar rib (0.193 L)
  bank(L * 0.04, 0, 0);
  bank(L * 0.04, Math.PI, 1);
  bank(-L * 0.07, -Math.PI / 2, 2);
  bank(-L * 0.07, Math.PI / 2, 3);
  // Hull codes on those bare flats, lit from the collar step (the bank
  // walls the slot in behind them)
  k.markings2(ap, 0, L * 0.148, W * 0.075, { flood: 'fore' });
  // Four PD balls round the collar (pd_0..pd_3)
  const pr = W * 0.48 * Math.SQRT1_2;
  k.pd(pr, pr, L * 0.3, -Math.PI / 4);
  k.pd(-pr, pr, L * 0.3, Math.PI / 4);
  k.pd(pr, -pr, L * 0.3, -Math.PI * 0.75);
  k.pd(-pr, -pr, L * 0.3, Math.PI * 0.75);
}

/** Battlecruiser: stretched racer - knife bow, split hull, open waist truss. */
function hullBattlecruiser(k) {
  const L = k.L, W = k.W;
  k.hasSpinal = true;
  const fw = W * 0.5, fh = W * 0.44, aw = W * 0.62, ah = W * 0.54;
  k.box('hull', fw, fh, L * 0.38, 0, 0, L * 0.21, { bevel: 0.2 });
  k.noseCap({ z0: L * 0.4, len: L * 0.15, w: fw, h: fh, tw: 0.24, th: 0.72, layers: 3, tip: null });
  k.box('trim', fw * 1.04, fh * 1.04, L * 0.03, 0, 0, L * 0.03, { bevel: 0.2 });
  k.truss('trim', W * 0.17, L * 0.18, -L * 0.07);
  k.pipe('barrel', W * 0.03, L * 0.19, W * 0.08, -W * 0.06, -L * 0.07, { clamps: 2 });
  k.pipe('barrel', W * 0.03, L * 0.19, -W * 0.08, W * 0.06, -L * 0.07, { clamps: 2 });
  k.box('hull', aw, ah, L * 0.26, 0, 0, -L * 0.29, { bevel: 0.2 });
  k.box('trim', aw * 1.03, ah * 1.03, L * 0.03, 0, 0, -L * 0.16, { bevel: 0.2 });
  // Ops deck forward, crew decks along both sections
  k.opsBand(fw, fh, L * 0.37, W * 0.012);
  for (const z of [L * 0.33, L * 0.23, L * 0.11]) k.deckBand(fw, fh, z, { bottom: false });
  for (const z of [-L * 0.22, -L * 0.36]) k.deckBand(aw, ah, z, { bottom: false });
  // Sensors on the forward hull
  k.sensorCluster(0.36 * k.k, 0, fh / 2, L * 0.27);
  k.mast(k.mastH * 1.1, W * 0.12, fh / 2, L * 0.17, 0);
  // Dorsal spine + ventral spinal gun (SPINAL battlecruiser: -0.245 W, 0.56 L)
  k.box('trim', W * 0.07, W * 0.07, L * 0.7, 0, W * 0.26, -L * 0.02, { bevel: 0.3 });
  k.box('accent', W * 0.06, W * 0.04, L * 0.75, 0, -W * 0.245, L * 0.05, { bevel: 0 });
  k.box('trim', W * 0.13, W * 0.09, L * 0.76, 0, -W * 0.23, L * 0.04, { bevel: 0.3 });
  k.drum('barrel', W * 0.05, L * 0.12, L * 0.5, { y: -W * 0.245, sides: 10 });
  k.drum('dark', W * 0.07, L * 0.03, L * 0.545, { y: -W * 0.245, sides: 10 });   // brake face = muzzle
  k.spinalMuzzle = new THREE.Vector3(0, -W * 0.245, L * 0.56);
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.05, W * 0.3, L * 0.34, sx * W * 0.26, 0, L * 0.19, { face: sx > 0 ? '+x' : '-x', mat: 'paint' });
    k.armorBelt(W * 0.05, W * 0.36, L * 0.22, sx * W * 0.32, 0, -L * 0.3, { face: sx > 0 ? '+x' : '-x' });
  }
  // weapon_1 port / weapon_2 starboard batteries on the forward flanks
  k.turret('coilgun', W * 0.285, 0, L * 0.16, -Math.PI / 2);
  k.turret('coilgun', -W * 0.285, 0, L * 0.16, Math.PI / 2);
  // PD aft: dorsal pd_0, ventral pd_1
  k.pd(W * 0.12, ah / 2, -L * 0.26);
  k.pd(0, -ah / 2, -L * 0.26, Math.PI);
  k.spineStripe(ah / 2, -L * 0.41, -L * 0.17, W * 0.14, { x: -W * 0.14 });
  k.greebles('+y', ah / 2, [W * 0.02, W * 0.26], [-L * 0.41, -L * 0.32], 5, W * 0.035, W * 0.07);
  k.greebles('-y', -ah / 2, [-W * 0.2, W * 0.2], [-L * 0.42, -L * 0.32], 5, W * 0.035, W * 0.07);
  k.runningLights(fw / 2 + 0.01, W * 0.05, L * 0.37);
  k.markings2(W * 0.345, 0, -L * 0.3, W * 0.08);
  k.rcsStation(L * 0.3, fw * 0.49, fh * 0.49, k.rcsS, { diag: true });
  k.strobe(0, 0, L * 0.555, 0);
  k.formationLamp(0, W * 0.3, L * 0.02);
  k.formationLamp(0, W * 0.295 + 0.02, -L * 0.3);   // on the dorsal spine (top 0.295 W)
  const rad = k.radiatorSize(2, { maxSpan: W * 0.9, maxLen: L * 0.17 });
  k.radiator(rad.span, rad.len, 0, W * 0.18, -L * 0.07, 0);
  k.radiator(rad.span, rad.len, 0, -W * 0.18, -L * 0.07, Math.PI);
  k.drive = { shape: 'box', w: W * 0.9, h: W * 0.42 };
  k.reactorPos.set(0, 0, -L * 0.32);
}

/** Battleship: broad slab of layered armour, wide flat chisel, heavy batteries. */
function hullBattleship(k) {
  const L = k.L, W = k.W;
  k.hasSpinal = true;
  const sw = W * 0.95, sh = W * 0.48;
  k.box('hull', sw, sh, L * 0.64, 0, 0, -L * 0.04, { bevel: 0.18 });
  k.noseCap({ z0: L * 0.28, len: L * 0.14, w: sw, h: sh, tw: 0.78, th: 0.24, layers: 3, tip: null });
  // Layered applique armour slabs
  const top = W * 0.33;
  k.armorBelt(W * 0.8, W * 0.08, L * 0.5, 0, W * 0.27, -L * 0.04, { face: '+y' });
  k.armorBelt(W * 0.8, W * 0.08, L * 0.5, 0, -W * 0.27, -L * 0.06, { face: '-y' });
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.06, W * 0.36, L * 0.52, sx * W * 0.49, 0, -L * 0.05, { face: sx > 0 ? '+x' : '-x' });
    k.pipe('barrel', W * 0.02, L * 0.4, sx * W * 0.5, -W * 0.15, -L * 0.1, { clamps: 4 });
  }
  // Deck rings on the flanks (the armour belts leave a gap at each frame)
  k.opsBand(sw, sh, L * 0.255, W * 0.012);
  for (const z of [L * 0.22, L * 0.06, -L * 0.1, -L * 0.24]) {
    k.deckBand(sw, sh, z, { top: false, bottom: false });
  }
  // Heavy spinal rail (SPINAL battleship: -0.335 W, 0.5 L)
  k.box('accent', W * 0.06, W * 0.04, L * 0.66, 0, -W * 0.335, L * 0.08, { bevel: 0 });
  k.box('trim', W * 0.14, W * 0.1, L * 0.68, 0, -W * 0.32, L * 0.06, { bevel: 0.3 });
  k.drum('barrel', W * 0.055, L * 0.12, L * 0.44, { y: -W * 0.335, sides: 10 });
  k.drum('dark', W * 0.08, L * 0.03, L * 0.485, { y: -W * 0.335, sides: 10 });   // brake face = muzzle
  k.spinalMuzzle = new THREE.Vector3(0, -W * 0.335, L * 0.5);
  // Sensor farm between the batteries
  k.sensorCluster(0.5 * k.k, -W * 0.28, top, -L * 0.1);
  k.sensorCluster(0.4 * k.k, W * 0.28, top, -L * 0.13);
  k.mast(k.mastH * 1.2, -W * 0.3, top, -L * 0.2, 0.08);
  k.mast(k.mastH, W * 0.3, top, -L * 0.24, 0.5);
  // weapon_1 heavy dorsal fwd, weapon_2 heavy ventral fwd, weapon_3 heavy
  // mid dorsal, weapon_4 (standard) mid ventral - ventral mounts beside the rail
  k.turret('heavy', 0, top, L * 0.17);
  k.turret('heavy', W * 0.25, -top, L * 0.17, Math.PI);
  k.turret('heavy', 0, top, -L * 0.02);
  k.turret('coilgun', -W * 0.25, -top, -L * 0.02, Math.PI);
  // PD: dorsal aft pd_0, ventral aft pd_1, port midships pd_2
  k.pd(0, top, -L * 0.2);
  k.pd(W * 0.25, -top, -L * 0.2, Math.PI);
  k.pd(W * 0.52, 0, L * 0.12, -Math.PI / 2);
  k.spineStripe(top, -L * 0.285, -L * 0.08, W * 0.3);
  k.livery(L * 0.08, L * 0.05, (zc, t) => k.box('paint', W * 0.5, 0.012 * k.k, t, 0, top + 0.006 * k.k, zc, { bevel: 0.3, seg: 1 }));
  k.greebles('-y', -top, [-W * 0.38, -W * 0.12], [-L * 0.32, -L * 0.1], 6, W * 0.03, W * 0.06);
  k.greebles('-y', -top, [W * 0.12, W * 0.38], [-L * 0.32, -L * 0.28], 3, W * 0.03, W * 0.06);
  k.runningLights(W * 0.535, W * 0.1, L * 0.2);
  k.markings2(W * 0.52, W * 0.11, -L * 0.02, W * 0.07);
  k.rcsStation(L * 0.24, W * 0.46, W * 0.23, k.rcsS, { diag: true });
  k.strobe(0, 0, L * 0.425, 0);
  k.formationLamp(W * 0.52, W * 0.19, L * 0.24);
  k.formationLamp(-W * 0.52, W * 0.19, L * 0.24);
  const rad = k.radiatorSize(4, { maxSpan: W * 0.4, maxLen: L * 0.2 });
  const rz = -L * 0.36 + rad.len / 2 + L * 0.01;
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) k.radiator(rad.span, rad.len, sx * W * 0.52, sy * W * 0.1, rz, sx > 0 ? -Math.PI / 2 : Math.PI / 2);
  }
  k.drive = { shape: 'box', w: W * 0.92, h: W * 0.46 };
  k.reactorPos.set(0, 0, -L * 0.28);
}

/**
 * Dreadnought: Donnager-class quad-lobe fortress on a drive skirt, the
 * spinal coiler along the keel and its muzzle under a stepped citadel bow.
 * The siege variant carries the triple siege coiler instead.
 */
function hullDreadnought(k, opts = {}) {
  const L = k.L, W = k.W;
  k.hasSpinal = true;   // weapon_0 is the spinal (or siege) coiler
  const winS = k.winS;
  k.drum('hull', W * 0.3, L * 0.78, -L * 0.03);
  for (let i = 0; i < 6; i++) k.rib('trim', W * 0.315, L * 0.014, -L * 0.36 + i * L * 0.13);
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const px = sx * W * 0.36, py = sy * W * 0.36;
      // Armoured load paths tie the four sponsons into the pressure hull.
      k.box('trim', W * 0.24, W * 0.24, L * 0.57,
        sx * W * 0.23, sy * W * 0.23, -L * 0.07, { bevel: 0.15 });
      k.box('hull', W * 0.32, W * 0.32, L * 0.52, px, py, -L * 0.08, { bevel: 0.2 });
      k.box('trim', W * 0.34, W * 0.34, L * 0.03, px, py, -L * 0.2, { bevel: 0.2 });
      k.box('trim', W * 0.34, W * 0.34, L * 0.03, px, py, L * 0.05, { bevel: 0.2 });
      k.armorBelt(W * 0.34, W * 0.1, L * 0.3, px, py + sy * W * 0.16, -L * 0.04, { face: sy > 0 ? '+y' : '-y' });
      k.pipe('barrel', W * 0.02, L * 0.4, px + sx * W * 0.17, py, -L * 0.1, { clamps: 4 });
      // Sponson decks on the outboard faces: many small windows, the
      // dreadnought's scale read
      for (const z of [L * 0.12, L * 0.0, -L * 0.13, -L * 0.27]) {
        for (let j = 0; j < 4; j++) {
          const yy = py - sy * W * 0.105 + j * sy * W * 0.07;
          for (const dz of [-1, 1]) k._window(winS * 0.35, winS * 1.4, winS, px + sx * W * 0.161, yy, z + dz * winS * 2.2);
        }
      }
      k.box('trim', W * 0.28, W * 0.28, L * 0.05, px, py, L * 0.2, { bevel: 0.4 });
    }
  }
  // Bow citadel narrowing to a stepped glacis, ops deck band around it
  k.box('hull', W * 0.5, W * 0.5, L * 0.16, 0, 0, L * 0.42, { bevel: 0.25 });
  k.box('hull', W * 0.64, W * 0.64, L * 0.14, 0, 0, L * 0.3, { bevel: 0.25 });
  k.livery(L * 0.235, L * 0.04, (zc, t) => k.box('paint', W * 0.66, W * 0.66, t, 0, 0, zc, { bevel: 0.2 }));
  k.opsBand(W * 0.5, W * 0.5, L * 0.47, W * 0.012);
  k.deckBand(W * 0.5, W * 0.5, L * 0.44);
  k.deckBand(W * 0.64, W * 0.64, L * 0.33);
  k.noseCap({ z0: L * 0.5, len: L * 0.07, w: W * 0.5, h: W * 0.5, tw: 0.62, th: 0.62, layers: 3 });
  k.greebles('+y', W * 0.32, [-W * 0.26, -W * 0.1], [L * 0.24, L * 0.36], 3, W * 0.03, W * 0.05);
  if (!opts.siege) {
    // Spinal coiler along the keel, between the lower sponsons; the barrel
    // runs out under the glacis to the bow line. Its muzzle is the shared
    // default spinal muzzle (-0.31 W, 0.56 L, as on the destroyer), so
    // Projectiles needs no per-hull entry for it - hull_design.test.js
    // pins the two together.
    const ky = -W * 0.31;
    k.box('trim', W * 0.16, W * 0.15, L * 0.605, 0, ky, L * 0.1275, { bevel: 0.3 });
    k.box('accent', W * 0.03, W * 0.03, L * 0.58, 0, ky - W * 0.077, L * 0.13, { bevel: 0 });
    k.drum('barrel', W * 0.06, L * 0.13, L * 0.48, { y: ky, sides: 12 });
    for (const bz of [L * 0.45, L * 0.478, L * 0.506]) {
      k.drum('accent', W * 0.078, W * 0.02, bz, { y: ky, sides: 12 });
      k.drum('trim', W * 0.085, W * 0.035, bz + W * 0.03, { y: ky, sides: 12 });
    }
    k.drum('dark', W * 0.09, L * 0.03, L * 0.545, { y: ky, sides: 12 });
    k.spinalMuzzle = new THREE.Vector3(0, ky, L * 0.56);
  }
  // Sensor farms along the keel line instead of towers
  for (const [tz, sz] of [[L * 0.12, 0.6], [-L * 0.33, 0.5]]) {
    k.sensorCluster(sz * k.k, 0, W * 0.3, tz);
  }
  k.mast(k.mastH * 1.4, W * 0.1, W * 0.3, -L * 0.28, 0);
  // Batteries where fleet_ships mounts them: weapon_1 heavy dorsal fwd,
  // weapon_2 heavy ventral fwd (beside the keel spinal), weapon_3 heavy
  // port, weapon_4 heavy starboard, weapon_5 heavy mid dorsal, weapon_6
  // (standard) mid ventral
  k.turret('heavy', 0, W * 0.32, L * 0.3);
  k.turret('heavy', W * 0.2, -W * 0.32, L * 0.3, Math.PI);
  k.turret('heavy', W * 0.52, W * 0.36, L * 0.1, -Math.PI / 2);
  k.turret('heavy', -W * 0.52, W * 0.36, L * 0.1, Math.PI / 2);
  // PD galleries bridge the valleys between the armoured sponsons.
  for (const a of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
    k.box('hull', W * 0.52, W * 0.12, L * 0.2, 0, W * 0.54, -L * 0.1,
      { bevel: 0.25, parent: k.xf(0, 0, 0, 0, 0, a) });
  }
  k.turret('heavy', 0, W * 0.6, -L * 0.04);
  k.turret('coilgun', 0, -W * 0.6, -L * 0.04, Math.PI);
  // PD: dorsal aft pd_0, ventral aft pd_1, port pd_2, starboard pd_3
  k.pd(0, W * 0.6, -L * 0.16);
  k.pd(0, -W * 0.6, -L * 0.16, Math.PI);
  k.pd(W * 0.6, 0, -L * 0.03, -Math.PI / 2);
  k.pd(-W * 0.6, 0, -L * 0.03, Math.PI / 2);
  k.livery(-L * 0.1, L * 0.05, (zc, t) => {
    for (const a of [Math.PI / 2, -Math.PI / 2]) k.box('paint', W * 0.53, W * 0.125, t, 0, W * 0.54, zc, { bevel: 0.25, parent: k.xf(0, 0, 0, 0, 0, a) });
  });
  k.greebles('-y', -W * 0.3, [-W * 0.08, W * 0.08], [-L * 0.36, -L * 0.24], 4, W * 0.03, W * 0.05);
  k.runningLights(W * 0.505, W * 0.36, L * 0.2);
  k.markings2(W * 0.52, W * 0.43, -L * 0.1, W * 0.06);
  k.rcsStation(L * 0.27, W * 0.31, W * 0.31, k.rcsS, { diag: true });
  k.rcsStation(L * 0.38, W * 0.25, W * 0.25, k.rcsS, { diag: true });
  k.strobe(0, 0, L * 0.575, 0);
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) k.formationLamp(sx * W * 0.36, sy * W * 0.36, L * 0.228);
  if (opts.siege) {
    // Siege coiler: one long barrel through the glacis (SPINAL dreadnought_siege: -0.2 W, 0.8 L)
    k.drum('barrel', W * 0.07, L * 0.6, L * 0.5, { y: -W * 0.2, sides: 12, rTop: W * 0.06 });
    for (const bz of [L * 0.34, L * 0.6, L * 0.7]) {
      k.drum('accent', W * 0.11, W * 0.03, bz, { y: -W * 0.2, sides: 10 });
      k.drum('trim', W * 0.12, W * 0.06, bz + W * 0.05, { y: -W * 0.2, sides: 10 });
    }
    k.drum('dark', W * 0.1, L * 0.04, L * 0.78, { y: -W * 0.2, sides: 12 });   // brake face = muzzle
    k.spinalMuzzle = new THREE.Vector3(0, -W * 0.2, L * 0.8);
    // fire-control mast stands on the bow citadel (top at 0.32 W)
    k.mast(k.mastH * 1.6, W * 0.14, W * 0.32, L * 0.3, -0.3);
  }
  // Radiators on the outboard corners of the aft sponsons, deployed on
  // the diagonals where no sponson shadows them
  const rad = k.radiatorSize(4, { maxSpan: W * 0.35, maxLen: L * 0.16 });
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const a = Math.atan2(-sx, sy);   // local +Y -> (sx, sy) diagonal
      k.radiator(rad.span, rad.len, sx * W * 0.52, sy * W * 0.52, -L * 0.33 + rad.len / 2 + L * 0.005, a);
    }
  }
  k.drive = { shape: 'drum', r: W * 0.5 };
  k.reactorPos.set(0, 0, -L * 0.3);
}

const BUILDERS = {
  corvette: hullCorvette,
  frigate: hullFrigate,
  destroyer: hullDestroyer,
  cruiser: hullCruiser,
  cruiser_torpedo: hullTorpedoCruiser,
  battlecruiser: hullBattlecruiser,
  battleship: hullBattleship,
  dreadnought: hullDreadnought,
  dreadnought_siege: (k) => hullDreadnought(k, { siege: true })
};

/** Hull classes the kit knows (others fall back to the destroyer). */
export const HULL_CLASSES = Object.keys(BUILDERS);

/**
 * Lay out a class hull into a kit (parts, rigs, lamps, magazine...) without
 * building meshes - no materials or canvas needed, so tests run headless.
 */
export function assembleHull(shipType, size, engineConfig, seed, faction = 'alpha') {
  const kit = new HullKit(size, seed);
  kit.shipType = shipType;
  kit.faction = faction;
  (BUILDERS[shipType] || hullDestroyer)(kit);
  kit.driveSection(engineConfig);
  return kit;
}

/**
 * Build a hull into `group`. Returns the kit (for the drive section and
 * anything the caller wants to attach).
 */
export function buildHull(shipType, group, size, mats, engineConfig, seed) {
  const kit = assembleHull(shipType, size, engineConfig, seed, mats?.faction || 'alpha');
  kit.build(group, mats);
  return kit;
}

// ---------------------------------------------------------------------------
// Debris chunks: split the hull back into pieces along part boundaries
// ---------------------------------------------------------------------------

/**
 * @param {THREE.Group} shipGroup   built hull (userData.parts)
 * @param {number} chunkCount       target number of large chunks
 * @returns {Array<{mesh, center}>} chunk meshes in ship-local coordinates;
 *   each mesh's origin is at its own centroid so it tumbles naturally
 */
export function buildDebrisChunks(shipGroup, chunkCount, rnd = Math.random) {
  const parts = shipGroup.userData.parts || [];
  const mats = shipGroup.userData.materials;
  if (!parts.length) return [];
  const L = shipGroup.userData.size?.length || 1;
  const W = shipGroup.userData.size?.width || 0.25;

  // Bin along Z, then split fat bins laterally
  const sorted = parts.slice().sort((a, b) => a.center.z - b.center.z);
  const bins = [];
  const perBin = Math.max(1, Math.ceil(sorted.length / chunkCount));
  for (let i = 0; i < sorted.length; i += perBin) bins.push(sorted.slice(i, i + perBin));
  const clusters = [];
  for (const bin of bins) {
    if (bin.length > 6) {
      const left = bin.filter(p => p.center.x < -W * 0.12);
      const right = bin.filter(p => p.center.x > W * 0.12);
      const mid = bin.filter(p => Math.abs(p.center.x) <= W * 0.12);
      for (const c of [left, right, mid]) if (c.length) clusters.push(c);
    } else {
      clusters.push(bin);
    }
  }

  const chunks = [];
  for (const cluster of clusters) {
    const centroid = new THREE.Vector3();
    for (const p of cluster) centroid.add(p.center);
    centroid.divideScalar(cluster.length);
    // Drop a random third of small parts - the hull is blown apart, not
    // disassembled; missing pieces read as vaporised
    const keep = cluster.filter(p => cluster.length < 3 || rnd() > 0.3);
    if (!keep.length) continue;
    // Merge per material first, then merge the per-material pieces with
    // groups so the chunk is one mesh with one material per plating type
    const byMat = new Map();
    for (const p of keep) {
      const g = p.geom.clone();
      g.translate(-centroid.x, -centroid.y, -centroid.z);
      if (!byMat.has(p.mat)) byMat.set(p.mat, []);
      byMat.get(p.mat).push(g);
    }
    const geoms = [];
    const materials = [];
    for (const [k, gs] of byMat) {
      const g = BufferGeometryUtils.mergeGeometries(gs, false);
      if (!g) continue;
      geoms.push(g);
      const m = mats[k].clone();
      m.transparent = true;
      m.emissive = new THREE.Color(0xff7a30);
      m.emissiveIntensity = 0;
      if (m.emissiveMap) m.emissiveMap = null;
      materials.push(m);
    }
    const merged = geoms.length ? BufferGeometryUtils.mergeGeometries(geoms, true) : null;
    if (!merged) continue;
    const mesh = new THREE.Mesh(merged, materials);
    chunks.push({ mesh, center: centroid, size: Math.cbrt(keep.length) * L * 0.04 });
  }
  return chunks;
}
