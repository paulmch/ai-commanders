import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { getHullTextures, getRadiatorTextures, getGlowTexture, mulberry32 } from './Textures.js';

/**
 * Ship hull kit.
 *
 * Every hull is assembled from parts (bevelled plating, drums, trusses,
 * pipes, greebles, turrets...) that are baked into ship-local space with
 * UVs fitted to their physical size, then merged into one mesh per
 * material. A dreadnought is ~400 parts but ~6 draw calls. The part list
 * is kept on the group so the destruction sequence can split the hull
 * back into chunks along real part boundaries.
 *
 * Local frame: nose +Z, drive -Z, dorsal +Y. Sizes in scene km.
 */

const TEXEL = 1.35;   // plating tiles per scene unit

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
const _sharedMats = new Map();

function factionAccent(faction) {
  return faction === 'alpha' ? 0x2ad2ff : 0xff8a3c;
}

export function buildShipMaterials(faction) {
  const key = faction;
  if (!_sharedMats.has(key)) {
    const hullT = getHullTextures(faction, 'hull');
    const trimT = getHullTextures(faction, 'trim');
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
      dark: new THREE.MeshStandardMaterial({
        color: 0x1a1d22, roughness: 0.62, metalness: 0.85, envMapIntensity: 0.8
      }),
      barrel: new THREE.MeshStandardMaterial({
        color: 0x2a2d33, roughness: 0.45, metalness: 0.95, envMapIntensity: 1.0
      })
    });
  }
  const shared = _sharedMats.get(key);
  const accent = factionAccent(faction);
  // Per-ship emissive materials (damage flicker, dying dimming)
  return {
    ...shared,
    accent: new THREE.MeshStandardMaterial({
      color: accent, emissive: accent, emissiveIntensity: 1.6, roughness: 0.35, metalness: 0.1
    }),
    window: new THREE.MeshStandardMaterial({
      color: 0x0a0a0c, emissive: 0xffe4bd, emissiveIntensity: 2.6, roughness: 0.2, metalness: 0.2
    }),
    bellHeat: new THREE.MeshStandardMaterial({
      color: 0x1a1010, emissive: 0xff7a30, emissiveIntensity: 0.0, roughness: 0.8, metalness: 0.2,
      side: THREE.DoubleSide
    })
  };
}

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
function fitBoxUVs(g, w, h, d, rnd) {
  const sizes = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  const idx = g.index;
  // group vertex ranges: vertices are laid out face by face in BoxGeometry
  const perFace = g.attributes.uv.count / 6;
  for (let f = 0; f < 6; f++) {
    scaleUVRange(g, f * perFace, perFace, sizes[f][0] * TEXEL, sizes[f][1] * TEXEL);
    offsetUVs(g, rnd(), rnd(), f * perFace, perFace);
  }
  void idx;
}

function fitCylinderUVs(g, r, len, rnd) {
  const groups = g.groups.length ? g.groups : [{ start: 0, count: g.index.count }];
  // Torso vertices come first; caps after. Use the index groups to find
  // vertex ranges - torso is group 0.
  const uv = g.attributes.uv;
  const torsoVerts = groups.length > 1 ? Math.min(uv.count, indexRangeMaxVertex(g, groups[0]) + 1) : uv.count;
  scaleUVRange(g, 0, torsoVerts, 2 * Math.PI * r * TEXEL, len * TEXEL);
  if (torsoVerts < uv.count) scaleUVRange(g, torsoVerts, uv.count - torsoVerts, 2 * r * TEXEL, 2 * r * TEXEL);
  offsetUVs(g, rnd(), rnd());
}

function indexRangeMaxVertex(g, group) {
  const idx = g.index.array;
  let m = 0;
  for (let i = group.start; i < group.start + group.count; i++) if (idx[i] > m) m = idx[i];
  return m;
}

// ---------------------------------------------------------------------------
// Kit
// ---------------------------------------------------------------------------
export class HullKit {
  constructor(size, seed) {
    this.L = size.length;
    this.W = size.width;
    this.rnd = mulberry32(seed);
    this.parts = [];          // {geom, mat, center}
    this.radiators = [];      // {mesh} separate animated meshes
    this.navLights = [];      // {pos, color, period, phase}
    this.bellHeat = [];       // meshes whose emissive follows thrust
    this.turretRigs = [];     // {s,x,y,z,angle} moving turret parts built in build()
    this.reactorPos = new THREE.Vector3(0, 0, -this.L * 0.3);
    this._m = new THREE.Matrix4();
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
    fitBoxUVs(g, w, h, d, this.rnd);
    const m = this.xf(x, y, z, ...(rot || [0, 0, 0]), parent);
    return this.add(mat, g, m);
  }

  /** Drum with its axis along Z. sides=8 gives the octagonal hull look. */
  drum(mat, r, len, z, { x = 0, y = 0, sides = 8, rTop = null, open = false, parent = null, rot = null } = {}) {
    const g = new THREE.CylinderGeometry(rTop ?? r, r, len, sides, 1, open);
    if (sides <= 8) g.rotateY(Math.PI / sides);
    fitCylinderUVs(g, r, len, this.rnd);
    g.rotateX(Math.PI / 2);
    const m = this.xf(x, y, z, ...(rot || [0, 0, 0]), parent);
    return this.add(mat, g, m);
  }

  /** Thin ring rib around a drum (a slightly larger, very short drum). */
  rib(mat, r, thick, z, { sides = 8, x = 0, y = 0 } = {}) {
    return this.drum(mat, r, thick, z, { sides, x, y });
  }

  /** Four-sided wedge prow, apex at +Z. */
  prow(mat, w, h, len, z, { parent = null } = {}) {
    const g = new THREE.ConeGeometry(0.5, len, 4);
    g.rotateY(Math.PI / 4);
    g.rotateX(Math.PI / 2);
    g.scale(w * 1.4, h * 1.4, 1);
    fitCylinderUVs(g, (w + h) * 0.5, len, this.rnd);
    return this.add(mat, g, this.xf(0, 0, z, 0, 0, 0, parent));
  }

  /** Solid cone (radar dish, nose cap), axis along +Z. */
  cone(mat, r, len, x, y, z, { rot = null, sides = 12, parent = null } = {}) {
    const g = new THREE.ConeGeometry(r, len, sides);
    fitCylinderUVs(g, r, len, this.rnd);
    g.rotateX(Math.PI / 2);
    return this.add(mat, g, this.xf(x, y, z, ...(rot || [0, 0, 0]), parent));
  }

  sphere(mat, r, x, y, z, { parent = null, ws = 12, hs = 8 } = {}) {
    const g = new THREE.SphereGeometry(r, ws, hs);
    scaleUVRange(g, 0, g.attributes.uv.count, 2 * Math.PI * r * TEXEL, Math.PI * r * TEXEL);
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
  armorBelt(w, h, d, x, y, z, { face = '+y' } = {}) {
    const t = Math.min(w, h) ;
    this.box('trim', w * 1.06, h * 0.45, d * 1.04, x, y, z, { bevel: 0.2 });
    this.box('hull', w, h, d, x, y + (face === '+y' ? h * 0.3 : face === '-y' ? -h * 0.3 : 0), z, { bevel: 0.45 });
    void t;
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
   * Coilgun turret. Only the static ring is baked into the hull; the
   * housing (yaw) and barrels (elevation) are built as a rig in build()
   * so the guns can track a target. angle rotates the mount about Z.
   */
  turret(s, x, y, z, angle = 0) {
    const P = this.xf(x, y, z, 0, 0, angle);
    this.drum('dark', s * 0.62, s * 0.14, 0, { sides: 12, parent: this.xf(0, s * 0.07, 0, Math.PI / 2, 0, 0, P) });
    this.drum('trim', s * 0.5, s * 0.1, 0, { sides: 12, parent: this.xf(0, s * 0.17, 0, Math.PI / 2, 0, 0, P) });
    this.turretRigs.push({ s, x, y, z, angle });
  }

  /** PD laser blister: dome, trunnion and a slim barrel. */
  pd(s, x, y, z, angle = 0) {
    const P = this.xf(x, y, z, 0, 0, angle);
    this.drum('trim', s * 0.4, s * 0.1, 0, { sides: 10, parent: this.xf(0, s * 0.05, 0, Math.PI / 2, 0, 0, P) });
    const dome = new THREE.SphereGeometry(s * 0.32, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2);
    scaleUVRange(dome, 0, dome.attributes.uv.count, 2 * Math.PI * s * 0.32 * TEXEL, Math.PI * s * 0.16 * TEXEL);
    this.add('hull', dome, this.xf(0, s * 0.1, 0, 0, 0, 0, P));
    this.box('dark', s * 0.16, s * 0.16, s * 0.2, 0, s * 0.3, s * 0.12, { bevel: 0.3, parent: P });
    this.drum('barrel', s * 0.035, s * 0.55, 0, { sides: 6, parent: this.xf(0, s * 0.36, s * 0.32, -Math.PI / 3.2, 0, 0, P) });
    this.box('accent', s * 0.05, s * 0.05, s * 0.05, 0, s * 0.44, s * 0.46, { bevel: 0, parent: P });
  }

  /** Sensor mast with a lit tip and a couple of dishes. */
  mast(h, x, y, z, lean = 0) {
    const P = this.xf(x, y, z, 0, 0, lean);
    this.drum('dark', h * 0.035, h, 0, { sides: 6, parent: this.xf(0, h / 2, 0, Math.PI / 2, 0, 0, P) });
    this.box('trim', h * 0.16, h * 0.06, h * 0.16, 0, h * 0.55, 0, { bevel: 0.2, parent: P });
    this.cone('trim', h * 0.11, h * 0.05, 0, h * 0.78, h * 0.06, { rot: [-0.4, 0, 0], sides: 10, parent: P });
    this.box('accent', h * 0.05, h * 0.05, h * 0.05, 0, h * 1.02, 0, { bevel: 0, parent: P });
    this.navLights.push({ pos: new THREE.Vector3(x - Math.sin(lean) * h, y + Math.cos(lean) * h, z), color: 0xffffff, period: 1.8, phase: 0.0, strobe: true });
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

  /**
   * Deck windows ringing an octagonal drum at z: decks stack along the
   * thrust axis, so their windows run around the hull, not along it.
   * `perFace` windows per flat face, spread along the face.
   */
  deckRing(r, z, s, { sides = 8, perFace = 2, skipFaces = [] } = {}) {
    const apothem = r * Math.cos(Math.PI / sides);
    const faceW = 2 * r * Math.sin(Math.PI / sides);
    for (let k = 0; k < sides; k++) {
      if (skipFaces.includes(k)) continue;
      const a = k * (2 * Math.PI / sides);
      const dx = Math.cos(a), dy = Math.sin(a);
      const tx = -Math.sin(a), ty = Math.cos(a);
      for (let j = 0; j < perFace; j++) {
        const off = (j - (perFace - 1) / 2) * (faceW / (perFace + 0.6));
        const x = dx * apothem + tx * off;
        const y = dy * apothem + ty * off;
        this.box('window', s * 1.5, s * 0.35, s, x, y, z, { bevel: 0, rot: [0, 0, a - Math.PI / 2] });
      }
    }
  }

  /** Deck windows around a rectangular section (w wide, h tall) at z. */
  deckBand(w, h, z, s, nx, ny, { top = true, bottom = true } = {}) {
    for (let i = 0; i < nx; i++) {
      const x = -w / 2 + (i + 0.5) * (w / nx);
      if (top) this.box('window', s * 1.4, s * 0.35, s, x, h / 2, z, { bevel: 0 });
      if (bottom) this.box('window', s * 1.4, s * 0.35, s, x, -h / 2, z, { bevel: 0 });
    }
    for (let j = 0; j < ny; j++) {
      const y = -h / 2 + (j + 0.5) * (h / ny);
      this.box('window', s * 0.35, s * 1.4, s, w / 2, y, z, { bevel: 0 });
      this.box('window', s * 0.35, s * 1.4, s, -w / 2, y, z, { bevel: 0 });
    }
  }

  /** Thin lit accent band hugging a rectangular section (w x h) at z. */
  opsBand(w, h, z, t) {
    this.box('accent', w + t * 0.6, t, t, 0, h / 2, z, { bevel: 0 });
    this.box('accent', w + t * 0.6, t, t, 0, -h / 2, z, { bevel: 0 });
    this.box('accent', t, h + t * 0.6, t, w / 2, 0, z, { bevel: 0 });
    this.box('accent', t, h + t * 0.6, t, -w / 2, 0, z, { bevel: 0 });
  }

  /** Lit ops-deck band: a thin accent ring with a window ring beside it. */
  opsDeck(r, z, s, { sides = 8 } = {}) {
    this.rib('accent', r * 1.005, s * 0.5, z + s * 1.2, { sides });
    this.deckRing(r, z, s, { sides, perFace: 2 });
  }

  /** Row of lit windows. axis 'x' spaces along Z on an X-facing wall. */
  windows(n, x, y, z, spacing, s, axis = 'x') {
    for (let i = 0; i < n; i++) {
      if (axis === 'x') this.box('window', s * 0.3, s, s * 0.6, x, y, z + i * spacing, { bevel: 0 });
      else this.box('window', s * 0.6, s * 0.3, s, x, y, z + i * spacing, { bevel: 0 });
    }
  }

  /** Port (red) / starboard (green) running lights at a station. */
  runningLights(x, y, z) {
    this.navLights.push({ pos: new THREE.Vector3(-x, y, z), color: 0xff2a2a, period: 0, phase: 0 });
    this.navLights.push({ pos: new THREE.Vector3(x, y, z), color: 0x2aff55, period: 0, phase: 0 });
  }

  /**
   * Radiator wing anchored at its root; scale.y folds it against the hull
   * when retracted. Built as a separate mesh (animated), with coolant
   * manifold pipes at the root baked into the hull.
   */
  radiator(span, len, x, y, z, angle, thickness = null) {
    const t = thickness ?? Math.max(this.W * 0.018, 0.02);
    const tex = getRadiatorTextures();
    const mat = new THREE.MeshStandardMaterial({
      map: tex.map, emissiveMap: tex.emissiveMap, emissive: 0xffffff, emissiveIntensity: 0,
      roughness: 0.6, metalness: 0.5, side: THREE.DoubleSide
    });
    const g = new THREE.BoxGeometry(t, span, len, 1, 1, 1);
    // Panel faces (px/nx) carry the emissive gradient: v runs root->tip
    const uv = g.attributes.uv;
    const perFace = uv.count / 6;
    for (let f = 0; f < 6; f++) {
      for (let i = f * perFace; i < (f + 1) * perFace; i++) {
        if (f < 2) uv.setXY(i, uv.getX(i) * (len / span) * 1.0, 1 - uv.getY(i)); // px, nx
        else uv.setXY(i, 0.5, 0.98);   // edges: dark tip colour
      }
    }
    g.translate(0, span / 2, 0);
    const m = new THREE.Mesh(g, mat);
    m.position.set(x, y, z);
    m.rotation.z = angle;
    // Root manifold + hinge baked into the hull
    const P = this.xf(x, y, z, 0, 0, angle);
    this.box('trim', t * 4, span * 0.06, len * 1.05, 0, span * 0.02, 0, { bevel: 0.3, parent: P });
    this.drum('dark', t * 2.2, len * 0.9, 0, { sides: 8, parent: this.xf(0, span * 0.05, 0, 0, 0, 0, P) });
    this.radiators.push({ mesh: m, material: mat });
    return m;
  }

  /** Drive skirt, thrust frame and engine bells matching the plume layout. */
  driveSection(engineConfig) {
    const L = this.L, W = this.W;
    const n = engineConfig.count;
    const bellR = W * (n <= 2 ? 0.16 : n <= 4 ? 0.125 : 0.09);
    // skirt: flared drum with ribs
    this.drum('hull', W * 0.5, L * 0.1, -L * 0.44, { rTop: W * 0.4, sides: 8 });
    this.rib('trim', W * 0.52, L * 0.015, -L * 0.4);
    this.rib('trim', W * 0.51, L * 0.015, -L * 0.47);
    // thrust frame plate
    this.drum('trim', W * 0.46, L * 0.02, -L * 0.49, { sides: 8 });
    for (const p of engineConfig.positions) {
      // bell: open cone, dark, with a stiffening ring and throat collar
      const bell = new THREE.CylinderGeometry(bellR * 0.5, bellR, L * 0.065, 18, 1, true);
      fitCylinderUVs(bell, bellR, L * 0.065, this.rnd);
      bell.rotateX(Math.PI / 2);
      this.add('dark', bell, this.xf(p.x, p.y, -L * 0.515));
      this.drum('barrel', bellR * 1.02, L * 0.008, -L * 0.545, { x: p.x, y: p.y, sides: 18, open: true });
      this.drum('trim', bellR * 0.58, L * 0.02, -L * 0.485, { x: p.x, y: p.y, sides: 12 });
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
      const merged = BufferGeometryUtils.mergeGeometries(geoms, false);
      if (!merged) { console.error('hull merge failed for', key); continue; }
      const mesh = new THREE.Mesh(merged, mats[key]);
      mesh.name = `hull_${key}`;
      group.add(mesh);
    }
  }

  /** Movable turret: mount (static, on the hull) > yaw > pitch (barrels). */
  _buildTurretRig(r, mats) {
    const s = r.s;
    const mount = new THREE.Group();
    mount.position.set(r.x, r.y, r.z);
    mount.rotation.z = r.angle;
    const yaw = new THREE.Group();
    yaw.position.y = s * 0.2;
    mount.add(yaw);
    const yk = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    yk.box('trim', s * 0.9, s * 0.1, s * 0.6, 0, s * 0.05, -s * 0.1, { bevel: 0.2 });
    yk.box('hull', s * 0.82, s * 0.42, s * 0.9, 0, s * 0.28, -s * 0.05, { bevel: 0.35 });
    yk.box('trim', s * 0.3, s * 0.16, s * 0.3, s * 0.3, s * 0.55, -s * 0.2, { bevel: 0.3 });
    yk.box('accent', s * 0.08, s * 0.03, s * 0.2, s * 0.3, s * 0.64, -s * 0.2, { bevel: 0 });
    yk._mergeInto(yaw, mats);
    const pitch = new THREE.Group();
    pitch.position.set(0, s * 0.32, s * 0.15);
    yaw.add(pitch);
    const pk = new HullKit({ length: this.L, width: this.W }, this.rnd() * 1e6);
    pk.box('trim', s * 0.6, s * 0.3, s * 0.28, 0, 0, s * 0.12, { bevel: 0.25 });   // mantlet
    for (const bx of [-0.2, 0.2]) {
      pk.drum('barrel', s * 0.06, s * 1.25, s * 0.78, { x: bx * s, sides: 8 });
      pk.drum('dark', s * 0.085, s * 0.14, s * 1.33, { x: bx * s, sides: 8 });   // muzzle brake
      pk.drum('dark', s * 0.09, s * 0.12, s * 0.45, { x: bx * s, sides: 8 });    // recoil collar
    }
    pk._mergeInto(pitch, mats);
    return { mount, yaw, pitch, s, yawAngle: 0, pitchAngle: 0.12 };
  }

  build(group, mats) {
    this._mergeInto(group, mats);
    for (const r of this.radiators) group.add(r.mesh);
    for (const liner of this.bellHeat) {
      liner.material = mats.bellHeat;
      group.add(liner);
    }
    const turrets = [];
    for (const r of this.turretRigs) {
      const rig = this._buildTurretRig(r, mats);
      group.add(rig.mount);
      turrets.push(rig);
    }
    // Nav lights as small HDR sprites
    const glow = getGlowTexture();
    const lights = [];
    for (const nl of this.navLights) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glow, color: nl.color, transparent: true, opacity: 0.9,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      s.position.copy(nl.pos);
      s.scale.setScalar(this.W * 0.16);
      s.renderOrder = 15;
      group.add(s);
      lights.push({ sprite: s, ...nl });
    }
    group.userData.parts = this.parts;
    group.userData.radiators = this.radiators.map(r => r.mesh);
    group.userData.radiatorMaterials = this.radiators.map(r => r.material);
    group.userData.bellHeat = this.bellHeat;
    group.userData.navLights = lights;
    group.userData.turrets = turrets;
    group.userData.reactorPos = this.reactorPos;
    group.userData.materials = mats;
  }
}

// ---------------------------------------------------------------------------
// Class hulls
//
// Expanse rules: the ship is a tower stacked along its thrust axis. Decks
// are perpendicular to the drive, so windows ring the hull; there is no
// naval bridge superstructure - command spaces are inside, marked by a
// lit ops-deck band near the nose, and only sensors break the skin.
// ---------------------------------------------------------------------------

/** Corvette: compact stacked gunboat on a twin torch. Torpedo tubes ventral. */
function hullCorvette(k) {
  const L = k.L, W = k.W;
  k.box('hull', W * 0.56, W * 0.62, L * 0.55, 0, 0, -L * 0.05, { bevel: 0.25 });
  k.box('hull', W * 0.46, W * 0.16, L * 0.42, 0, W * 0.38, -L * 0.02, { bevel: 0.4 });
  k.box('hull', W * 0.34, W * 0.12, L * 0.3, 0, W * 0.5, 0, { bevel: 0.45 });
  k.prow('hull', W * 0.28, W * 0.31, L * 0.3, L * 0.36);
  k.cone('trim', W * 0.05, L * 0.06, 0, 0, L * 0.53, { sides: 8 });
  // flank armour and pipe runs
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.06, W * 0.36, L * 0.34, sx * W * 0.3, -W * 0.04, -L * 0.06, { face: sx > 0 ? '+x' : '-x' });
    k.pipe('barrel', W * 0.018, L * 0.36, sx * W * 0.31, W * 0.2, -L * 0.1, { clamps: 3 });
  }
  // deck rings: ops deck forward, crew decks aft
  k.opsBand(W * 0.56, W * 0.62, L * 0.2, W * 0.012);
  k.deckBand(W * 0.56, W * 0.62, L * 0.16, W * 0.03, 3, 3, { top: false });
  k.deckBand(W * 0.56, W * 0.62, -L * 0.02, W * 0.03, 3, 3, { top: false });
  k.deckBand(W * 0.56, W * 0.62, -L * 0.2, W * 0.03, 3, 3, { top: false });
  // ventral torpedo tubes + doors
  k.box('accent', W * 0.24, W * 0.02, L * 0.28, 0, -W * 0.325, L * 0.08, { bevel: 0 });
  k.box('trim', W * 0.3, W * 0.05, L * 0.32, 0, -W * 0.33, L * 0.08, { bevel: 0.3 });
  k.drum('dark', W * 0.05, L * 0.05, L * 0.25, { x: -W * 0.06, y: -W * 0.34, sides: 10 });
  k.drum('dark', W * 0.05, L * 0.05, L * 0.25, { x: W * 0.06, y: -W * 0.34, sides: 10 });
  // dorsal machinery, PD, sensors
  k.greebles('+y', W * 0.46, [-W * 0.17, W * 0.17], [-L * 0.24, -L * 0.1], 6, W * 0.04, W * 0.08);
  k.greebles('-y', -W * 0.31, [-W * 0.22, W * 0.22], [-L * 0.3, -L * 0.1], 6, W * 0.04, W * 0.09);
  k.pd(W * 0.28, 0, W * 0.56, -L * 0.14);
  k.sensorCluster(W * 0.22, 0, W * 0.56, L * 0.06);
  k.mast(W * 0.4, 0, W * 0.56, -L * 0.24, 0.15);
  k.runningLights(W * 0.3, W * 0.05, L * 0.1);
  k.radiator(W * 0.42, L * 0.2, -W * 0.28, 0, -L * 0.24, -Math.PI / 2);
  k.radiator(W * 0.42, L * 0.2, W * 0.28, 0, -L * 0.24, Math.PI / 2);
  k.reactorPos.set(0, 0, -L * 0.28);
}

/** Frigate: sensor head on a truss neck, ribbed drum body, tall dorsal fin. */
function hullFrigate(k) {
  const L = k.L, W = k.W;
  k.box('hull', W * 0.44, W * 0.44, L * 0.2, 0, 0, L * 0.36, { bevel: 0.3 });
  k.prow('trim', W * 0.22, W * 0.22, L * 0.14, L * 0.5);
  k.cone('dark', W * 0.1, L * 0.04, 0, W * 0.1, L * 0.47, { rot: [-0.5, 0, 0] }); // forward dish
  k.deckBand(W * 0.44, W * 0.44, L * 0.34, W * 0.03, 2, 2, { top: false });
  k.opsBand(W * 0.44, W * 0.44, L * 0.4, W * 0.012);
  k.truss('trim', W * 0.16, L * 0.14, L * 0.2);
  k.pipe('barrel', W * 0.03, L * 0.15, 0, -W * 0.1, L * 0.2, { clamps: 2 });
  k.drum('hull', W * 0.34, L * 0.52, -L * 0.08);
  for (let i = 0; i < 4; i++) k.rib('trim', W * 0.355, L * 0.014, -L * 0.3 + i * L * 0.14);
  for (let i = 0; i < 3; i++) k.deckRing(W * 0.34, -L * 0.23 + i * L * 0.14, W * 0.028, { perFace: 2, skipFaces: [2] });
  // Dorsal fin with lit leading edge
  k.box('hull', W * 0.05, W * 0.52, L * 0.24, 0, W * 0.5, -L * 0.06, { bevel: 0.15 });
  k.box('trim', W * 0.07, W * 0.5, L * 0.03, 0, W * 0.5, -L * 0.18, { bevel: 0.2 });
  k.box('accent', W * 0.02, W * 0.5, W * 0.04, 0, W * 0.5, L * 0.055, { bevel: 0 });
  k.greebles('+x', W * 0.025, [W * 0.3, W * 0.7], [-L * 0.16, L * 0.04], 5, W * 0.025, W * 0.05);
  k.greebles('-x', -W * 0.025, [W * 0.3, W * 0.7], [-L * 0.16, L * 0.04], 5, W * 0.025, W * 0.05);
  k.turret(W * 0.2, 0, W * 0.4, L * 0.32);
  k.pd(W * 0.24, 0, -W * 0.36, 0, Math.PI);
  k.pd(W * 0.2, W * 0.3, -W * 0.16, -L * 0.2, -Math.PI * 0.7);
  k.greebles('-y', -W * 0.32, [-W * 0.2, W * 0.2], [-L * 0.3, L * 0.1], 8, W * 0.04, W * 0.09);
  k.sensorCluster(W * 0.2, -W * 0.16, W * 0.3, L * 0.02);
  k.mast(W * 0.36, W * 0.12, W * 0.42, L * 0.42, -0.2);
  k.runningLights(W * 0.35, 0, L * 0.05);
  k.radiator(W * 0.5, L * 0.22, -W * 0.3, W * 0.12, -L * 0.26, -Math.PI / 2 + 0.35);
  k.radiator(W * 0.5, L * 0.22, W * 0.3, W * 0.12, -L * 0.26, Math.PI / 2 - 0.35);
  k.reactorPos.set(0, 0, -L * 0.28);
}

/** Destroyer: armoured prow, ventral spinal rail, stacked deck sections. */
function hullDestroyer(k) {
  const L = k.L, W = k.W;
  k.prow('hull', W * 0.36, W * 0.31, L * 0.26, L * 0.42);
  k.box('trim', W * 0.4, W * 0.34, L * 0.04, 0, 0, L * 0.3, { bevel: 0.3 });
  k.box('hull', W * 0.6, W * 0.54, L * 0.36, 0, 0, L * 0.12, { bevel: 0.22 });
  k.box('hull', W * 0.7, W * 0.62, L * 0.4, 0, 0, -L * 0.2, { bevel: 0.22 });
  k.box('trim', W * 0.72, W * 0.64, L * 0.03, 0, 0, -L * 0.02, { bevel: 0.2 }); // frame band
  // Spinal coilgun: rail + muzzle + breech housing
  k.box('accent', W * 0.07, W * 0.04, L * 0.78, 0, -W * 0.31, L * 0.05, { bevel: 0 });
  k.box('trim', W * 0.15, W * 0.1, L * 0.8, 0, -W * 0.29, L * 0.03, { bevel: 0.3 });
  k.drum('barrel', W * 0.06, L * 0.12, L * 0.5, { y: -W * 0.31, sides: 10 });
  k.drum('dark', W * 0.085, L * 0.03, L * 0.55, { y: -W * 0.31, sides: 10 });
  k.box('trim', W * 0.22, W * 0.14, L * 0.12, 0, -W * 0.3, -L * 0.3, { bevel: 0.3 });
  // flank armour + pipes
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.06, W * 0.4, L * 0.3, sx * W * 0.36, W * 0.02, -L * 0.2, { face: sx > 0 ? '+x' : '-x' });
    k.armorBelt(W * 0.05, W * 0.32, L * 0.26, sx * W * 0.31, 0, L * 0.12, { face: sx > 0 ? '+x' : '-x' });
    k.pipe('barrel', W * 0.02, L * 0.3, sx * W * 0.37, -W * 0.18, -L * 0.18, { clamps: 3 });
  }
  // deck rings: ops deck at the forward section, crew decks aft
  k.opsBand(W * 0.6, W * 0.54, L * 0.27, W * 0.012);
  k.deckBand(W * 0.6, W * 0.54, L * 0.24, W * 0.03, 3, 3, { bottom: false });
  k.deckBand(W * 0.6, W * 0.54, L * 0.04, W * 0.03, 3, 3, { bottom: false });
  k.deckBand(W * 0.7, W * 0.62, -L * 0.12, W * 0.03, 4, 3, { bottom: false });
  k.deckBand(W * 0.7, W * 0.62, -L * 0.3, W * 0.03, 4, 3, { bottom: false });
  // Sensors amidships, no tower
  k.sensorCluster(W * 0.3, 0, W * 0.27, -L * 0.06);
  k.mast(W * 0.4, W * 0.15, W * 0.27, -L * 0.14, 0.1);
  k.turret(W * 0.24, 0, W * 0.29, L * 0.2);
  k.turret(W * 0.24, 0, -W * 0.33, -L * 0.05, Math.PI);
  k.pd(W * 0.24, W * 0.33, W * 0.12, L * 0.05, -Math.PI / 2);
  k.pd(W * 0.24, -W * 0.33, W * 0.12, L * 0.05, Math.PI / 2);
  k.greebles('+y', W * 0.31, [-W * 0.3, W * 0.3], [-L * 0.38, -L * 0.18], 10, W * 0.04, W * 0.09);
  k.greebles('+y', W * 0.27, [-W * 0.25, -W * 0.05], [L * 0.02, L * 0.12], 4, W * 0.03, W * 0.07);
  k.runningLights(W * 0.36, W * 0.1, -L * 0.1);
  k.radiator(W * 0.55, L * 0.24, -W * 0.36, 0, -L * 0.28, -Math.PI / 2);
  k.radiator(W * 0.55, L * 0.24, W * 0.36, 0, -L * 0.28, Math.PI / 2);
  k.reactorPos.set(0, 0, -L * 0.3);
}

/** Cruiser: armoured collar, heavy ribbed drum, exposed reactor drum aft. */
function hullCruiser(k, opts = {}) {
  const L = k.L, W = k.W;
  k.prow('hull', W * 0.3, W * 0.3, L * 0.2, L * 0.44);
  k.drum('trim', W * 0.5, L * 0.16, L * 0.28);
  k.rib('hull', W * 0.53, L * 0.03, L * 0.24);
  k.rib('hull', W * 0.53, L * 0.03, L * 0.33);
  k.opsDeck(W * 0.5, L * 0.285, W * 0.03);
  k.drum('hull', W * 0.44, L * 0.5, -L * 0.02);
  for (let i = 0; i < 5; i++) k.rib('trim', W * 0.455, L * 0.014, -L * 0.24 + i * L * 0.11);
  for (let i = 0; i < 4; i++) k.deckRing(W * 0.44, -L * 0.185 + i * L * 0.11, W * 0.028, { perFace: 2, skipFaces: opts.noSpinal ? [] : [6] });
  // Reactor drum with warning ring and coolant manifolds
  k.drum('hull', W * 0.36, L * 0.14, -L * 0.32);
  k.drum('accent', W * 0.375, L * 0.015, -L * 0.32);
  k.rib('trim', W * 0.38, L * 0.02, -L * 0.26);
  k.rib('trim', W * 0.38, L * 0.02, -L * 0.38);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    k.pipe('barrel', W * 0.022, L * 0.18, Math.cos(a) * W * 0.4, Math.sin(a) * W * 0.4, -L * 0.3, { clamps: 2 });
  }
  if (!opts.noSpinal) {
    k.box('accent', W * 0.06, W * 0.04, L * 0.7, 0, -W * 0.42, L * 0.05, { bevel: 0 });
    k.box('trim', W * 0.14, W * 0.1, L * 0.72, 0, -W * 0.4, L * 0.03, { bevel: 0.3 });
    k.drum('barrel', W * 0.055, L * 0.12, L * 0.44, { y: -W * 0.42, sides: 10 });
    k.drum('dark', W * 0.08, L * 0.03, L * 0.49, { y: -W * 0.42, sides: 10 });
  }
  k.sensorCluster(W * 0.3, 0, W * 0.44, L * 0.06);
  k.mast(W * 0.42, W * 0.14, W * 0.44, -L * 0.02, -0.12);
  if (!opts.noTurrets) {
    k.turret(W * 0.26, W * 0.42, 0, L * 0.14, -Math.PI / 2);
    k.turret(W * 0.26, -W * 0.42, 0, L * 0.14, Math.PI / 2);
  }
  k.pd(W * 0.22, 0, W * 0.44, -L * 0.12);
  k.pd(W * 0.22, 0, -W * 0.44, -L * 0.12, Math.PI);
  k.greebles('+y', W * 0.44, [-W * 0.2, W * 0.2], [-L * 0.24, -L * 0.02], 8, W * 0.035, W * 0.08);
  k.greebles('-y', -W * 0.44, [-W * 0.2, W * 0.2], [-L * 0.24, L * 0.1], 8, W * 0.035, W * 0.08);
  k.runningLights(W * 0.46, 0, L * 0.05);
  // Four radiator wings in an X around the aft drum
  for (const a of [0.785, 2.356, -2.356, -0.785]) {
    k.radiator(W * 0.6, L * 0.22, Math.sin(a) * -W * 0.34, Math.cos(a) * W * 0.34, -L * 0.16, a);
  }
  k.reactorPos.set(0, 0, -L * 0.32);
}

/** Torpedo cruiser: cruiser drum with four VLS banks instead of guns. */
function hullTorpedoCruiser(k) {
  const L = k.L, W = k.W;
  hullCruiser(k, { noSpinal: true, noTurrets: true });
  const bank = (z, angle) => {
    const P = k.xf(Math.sin(angle) * -W * 0.45, Math.cos(angle) * W * 0.45, z, 0, 0, angle);
    k.box('trim', W * 0.34, W * 0.06, L * 0.28, 0, 0, 0, { bevel: 0.25, parent: P });
    k.box('hull', W * 0.36, W * 0.02, L * 0.3, 0, -W * 0.03, 0, { bevel: 0.2, parent: P });
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        k.box('dark', W * 0.11, W * 0.02, L * 0.062, (r - 0.5) * W * 0.15, W * 0.035, (c - 1) * L * 0.085, { bevel: 0.2, parent: P });
        k.box('accent', W * 0.09, W * 0.006, W * 0.012, (r - 0.5) * W * 0.15, W * 0.048, (c - 1) * L * 0.085 + L * 0.024, { bevel: 0, parent: P });
      }
    }
  };
  bank(L * 0.12, 0);
  bank(L * 0.12, Math.PI);
  bank(-L * 0.06, Math.PI / 2);
  bank(-L * 0.06, -Math.PI / 2);
  k.pd(W * 0.22, W * 0.34, W * 0.34, L * 0.3, -Math.PI / 4);
  k.pd(W * 0.22, -W * 0.34, W * 0.34, L * 0.3, Math.PI / 4);
  k.pd(W * 0.22, W * 0.34, -W * 0.34, L * 0.3, -Math.PI * 0.75);
  k.pd(W * 0.22, -W * 0.34, -W * 0.34, L * 0.3, Math.PI * 0.75);
}

/** Battlecruiser: stretched racer - split hull with an open waist truss. */
function hullBattlecruiser(k) {
  const L = k.L, W = k.W;
  k.prow('hull', W * 0.26, W * 0.24, L * 0.16, L * 0.46);
  k.box('hull', W * 0.5, W * 0.44, L * 0.42, 0, 0, L * 0.19, { bevel: 0.2 });
  k.box('trim', W * 0.52, W * 0.46, L * 0.03, 0, 0, L * 0.05, { bevel: 0.2 });
  k.truss('trim', W * 0.17, L * 0.16, -L * 0.08);
  k.pipe('barrel', W * 0.03, L * 0.17, W * 0.08, -W * 0.06, -L * 0.08, { clamps: 2 });
  k.pipe('barrel', W * 0.03, L * 0.17, -W * 0.08, W * 0.06, -L * 0.08, { clamps: 2 });
  k.box('hull', W * 0.62, W * 0.54, L * 0.28, 0, 0, -L * 0.3, { bevel: 0.2 });
  k.box('trim', W * 0.64, W * 0.56, L * 0.03, 0, 0, -L * 0.2, { bevel: 0.2 });
  // Ops deck forward, crew decks along both sections
  k.opsBand(W * 0.5, W * 0.44, L * 0.37, W * 0.012);
  k.deckBand(W * 0.5, W * 0.44, L * 0.33, W * 0.026, 3, 3, { bottom: false });
  k.deckBand(W * 0.5, W * 0.44, L * 0.21, W * 0.026, 3, 3, { bottom: false });
  k.deckBand(W * 0.5, W * 0.44, L * 0.09, W * 0.026, 3, 3, { bottom: false });
  k.deckBand(W * 0.62, W * 0.54, -L * 0.26, W * 0.026, 4, 3, { bottom: false });
  k.deckBand(W * 0.62, W * 0.54, -L * 0.36, W * 0.026, 4, 3, { bottom: false });
  // Sensors on the forward hull; the old bridge fin is gone
  k.sensorCluster(W * 0.26, 0, W * 0.22, L * 0.26);
  k.mast(W * 0.5, W * 0.12, W * 0.22, L * 0.16, 0);
  // Dorsal spine + ventral spinal gun
  k.box('trim', W * 0.07, W * 0.07, L * 0.7, 0, W * 0.26, -L * 0.02, { bevel: 0.3 });
  k.box('accent', W * 0.06, W * 0.04, L * 0.75, 0, -W * 0.245, L * 0.05, { bevel: 0 });
  k.box('trim', W * 0.13, W * 0.09, L * 0.76, 0, -W * 0.23, L * 0.04, { bevel: 0.3 });
  k.drum('barrel', W * 0.05, L * 0.12, L * 0.5, { y: -W * 0.245, sides: 10 });
  k.drum('dark', W * 0.07, L * 0.03, L * 0.55, { y: -W * 0.245, sides: 10 });
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.05, W * 0.3, L * 0.34, sx * W * 0.26, 0, L * 0.18, { face: sx > 0 ? '+x' : '-x' });
    k.armorBelt(W * 0.05, W * 0.36, L * 0.22, sx * W * 0.32, 0, -L * 0.3, { face: sx > 0 ? '+x' : '-x' });
  }
  k.turret(W * 0.22, 0, W * 0.27, -L * 0.28);
  k.turret(W * 0.2, 0, -W * 0.27, -L * 0.34, Math.PI);
  k.pd(W * 0.22, W * 0.28, 0, L * 0.1, -Math.PI / 2);
  k.pd(W * 0.22, -W * 0.28, 0, L * 0.1, Math.PI / 2);
  k.greebles('+y', W * 0.27, [-W * 0.22, W * 0.22], [-L * 0.42, -L * 0.2], 10, W * 0.035, W * 0.08);
  k.greebles('-y', -W * 0.27, [-W * 0.2, W * 0.2], [-L * 0.42, -L * 0.2], 6, W * 0.035, W * 0.08);
  k.runningLights(W * 0.25, W * 0.05, L * 0.25);
  k.radiator(W * 0.66, L * 0.14, 0, W * 0.18, -L * 0.08, 0);
  k.radiator(W * 0.66, L * 0.14, 0, -W * 0.18, -L * 0.08, Math.PI);
  k.reactorPos.set(0, 0, -L * 0.32);
}

/** Battleship: broad slab of layered armour with four heavy turrets. */
function hullBattleship(k) {
  const L = k.L, W = k.W;
  k.box('hull', W * 0.95, W * 0.48, L * 0.58, 0, 0, -L * 0.03, { bevel: 0.18 });
  k.prow('hull', W * 0.5, W * 0.26, L * 0.24, L * 0.38);
  k.box('trim', W * 0.6, W * 0.3, L * 0.04, 0, 0, L * 0.27, { bevel: 0.2 });
  // Layered applique armour slabs
  k.armorBelt(W * 0.8, W * 0.08, L * 0.46, 0, W * 0.27, -L * 0.01, { face: '+y' });
  k.armorBelt(W * 0.62, W * 0.08, L * 0.34, 0, W * 0.33, L * 0.03, { face: '+y' });
  k.armorBelt(W * 0.8, W * 0.08, L * 0.46, 0, -W * 0.27, -L * 0.05, { face: '-y' });
  for (const sx of [-1, 1]) {
    k.armorBelt(W * 0.06, W * 0.36, L * 0.5, sx * W * 0.49, 0, -L * 0.04, { face: sx > 0 ? '+x' : '-x' });
    k.pipe('barrel', W * 0.02, L * 0.4, sx * W * 0.5, -W * 0.15, -L * 0.1, { clamps: 4 });
  }
  // Deck rings on the flanks (the armour belts leave a gap at each frame)
  k.opsBand(W * 0.95, W * 0.48, L * 0.245, W * 0.012);
  for (const z of [L * 0.18, L * 0.04, -L * 0.1, -L * 0.24]) {
    k.deckBand(W * 0.95, W * 0.48, z, W * 0.024, 0, 3);
  }
  // Heavy spinal rail
  k.box('accent', W * 0.06, W * 0.04, L * 0.66, 0, -W * 0.335, L * 0.08, { bevel: 0 });
  k.box('trim', W * 0.14, W * 0.1, L * 0.68, 0, -W * 0.32, L * 0.06, { bevel: 0.3 });
  k.drum('barrel', W * 0.055, L * 0.12, L * 0.44, { y: -W * 0.335, sides: 10 });
  k.drum('dark', W * 0.08, L * 0.03, L * 0.49, { y: -W * 0.335, sides: 10 });
  // Sensor farm where the citadel used to be
  k.sensorCluster(W * 0.36, -W * 0.15, W * 0.3, -L * 0.08);
  k.sensorCluster(W * 0.26, W * 0.22, W * 0.3, -L * 0.2);
  k.mast(W * 0.45, 0, W * 0.3, -L * 0.1, 0.08);
  k.mast(W * 0.3, W * 0.2, W * 0.3, -L * 0.24, 0.5);
  k.turret(W * 0.3, 0, W * 0.34, L * 0.18);
  k.turret(W * 0.3, W * 0.36, W * 0.3, -L * 0.24, -0.5);
  k.turret(W * 0.3, 0, -W * 0.33, L * 0.1, Math.PI);
  k.turret(W * 0.26, -W * 0.36, -W * 0.3, -L * 0.24, Math.PI - 0.5);
  k.pd(W * 0.24, W * 0.5, 0, L * 0.15, -Math.PI / 2);
  k.pd(W * 0.24, -W * 0.5, 0, L * 0.15, Math.PI / 2);
  k.pd(W * 0.24, 0, W * 0.34, -L * 0.32);
  k.pd(W * 0.22, W * 0.3, W * 0.34, L * 0.02, 0);
  k.greebles('+y', W * 0.31, [-W * 0.35, -W * 0.15], [-L * 0.3, L * 0.1], 8, W * 0.035, W * 0.08);
  k.greebles('+y', W * 0.31, [W * 0.15, W * 0.35], [-L * 0.3, L * 0.1], 8, W * 0.035, W * 0.08);
  k.greebles('-y', -W * 0.31, [-W * 0.3, W * 0.3], [-L * 0.3, -L * 0.1], 10, W * 0.035, W * 0.08);
  k.runningLights(W * 0.48, W * 0.1, L * 0.1);
  k.radiator(W * 0.5, L * 0.2, -W * 0.5, W * 0.1, -L * 0.3, -Math.PI / 2);
  k.radiator(W * 0.5, L * 0.2, W * 0.5, W * 0.1, -L * 0.3, Math.PI / 2);
  k.radiator(W * 0.5, L * 0.2, -W * 0.5, -W * 0.1, -L * 0.3, -Math.PI / 2);
  k.radiator(W * 0.5, L * 0.2, W * 0.5, -W * 0.1, -L * 0.3, Math.PI / 2);
  k.reactorPos.set(0, 0, -L * 0.28);
}

/** Dreadnought: Donnager-class quad-lobe fortress on a drive skirt. */
function hullDreadnought(k, opts = {}) {
  const L = k.L, W = k.W;
  k.drum('hull', W * 0.3, L * 0.78, -L * 0.03);
  for (let i = 0; i < 6; i++) k.rib('trim', W * 0.315, L * 0.014, -L * 0.36 + i * L * 0.13);
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const px = sx * W * 0.36, py = sy * W * 0.36;
      k.box('hull', W * 0.32, W * 0.32, L * 0.52, px, py, -L * 0.08, { bevel: 0.2 });
      k.box('trim', W * 0.34, W * 0.34, L * 0.03, px, py, -L * 0.2, { bevel: 0.2 });
      k.box('trim', W * 0.34, W * 0.34, L * 0.03, px, py, L * 0.05, { bevel: 0.2 });
      k.armorBelt(W * 0.34, W * 0.1, L * 0.3, px, py + sy * W * 0.16, -L * 0.04, { face: sy > 0 ? '+y' : '-y' });
      k.pipe('barrel', W * 0.02, L * 0.4, px + sx * W * 0.17, py, -L * 0.1, { clamps: 4 });
      // sponson deck rings on the outboard faces
      for (const z of [L * 0.12, -L * 0.02, -L * 0.14, -L * 0.28]) {
        for (let j = 0; j < 3; j++) {
          const yy = py - sy * W * 0.1 + j * sy * W * 0.05;
          k.box('window', W * 0.008, W * 0.03, W * 0.022, px + sx * W * 0.165, yy, z, { bevel: 0 });
        }
      }
      k.box('trim', W * 0.28, W * 0.28, L * 0.05, px, py, L * 0.2, { bevel: 0.4 });
    }
  }
  // Bow citadel narrowing to the prow, ops deck band around it
  k.box('hull', W * 0.5, W * 0.5, L * 0.16, 0, 0, L * 0.42, { bevel: 0.25 });
  k.box('hull', W * 0.64, W * 0.64, L * 0.14, 0, 0, L * 0.3, { bevel: 0.25 });
  k.box('trim', W * 0.66, W * 0.66, L * 0.03, 0, 0, L * 0.235, { bevel: 0.2 });
  k.opsBand(W * 0.5, W * 0.5, L * 0.47, W * 0.012);
  k.deckBand(W * 0.5, W * 0.5, L * 0.44, W * 0.024, 3, 3);
  k.deckBand(W * 0.64, W * 0.64, L * 0.32, W * 0.024, 4, 4);
  k.prow('trim', W * 0.25, W * 0.25, L * 0.12, L * 0.54);
  k.box('accent', W * 0.36, W * 0.04, W * 0.04, 0, 0, L * 0.5, { bevel: 0 });
  k.greebles('+y', W * 0.32, [-W * 0.2, W * 0.2], [L * 0.24, L * 0.36], 6, W * 0.03, W * 0.06);
  // Sensor farms along the keel line instead of towers
  for (const [tz, sz] of [[L * 0.18, W * 0.3], [-L * 0.02, W * 0.36], [-L * 0.22, W * 0.3]]) {
    k.sensorCluster(sz, 0, W * 0.3, tz);
  }
  k.mast(W * 0.4, W * 0.1, W * 0.3, -L * 0.1, 0);
  k.turret(W * 0.26, W * 0.36, W * 0.56, L * 0.12);
  k.turret(W * 0.26, -W * 0.36, W * 0.56, L * 0.12);
  k.turret(W * 0.26, W * 0.36, -W * 0.56, L * 0.12, Math.PI);
  k.turret(W * 0.26, -W * 0.36, -W * 0.56, L * 0.12, Math.PI);
  k.turret(W * 0.22, W * 0.56, W * 0.36, -L * 0.2, -Math.PI / 2);
  k.turret(W * 0.22, -W * 0.56, W * 0.36, -L * 0.2, Math.PI / 2);
  k.pd(W * 0.2, W * 0.6, 0, -L * 0.05, -Math.PI / 2);
  k.pd(W * 0.2, -W * 0.6, 0, -L * 0.05, Math.PI / 2);
  k.pd(W * 0.2, 0, W * 0.6, -L * 0.15);
  k.pd(W * 0.2, 0, -W * 0.6, -L * 0.15, Math.PI);
  k.pd(W * 0.18, W * 0.36, W * 0.56, -L * 0.3);
  k.pd(W * 0.18, -W * 0.36, -W * 0.56, -L * 0.3, Math.PI);
  k.greebles('-y', -W * 0.3, [-W * 0.15, W * 0.15], [-L * 0.36, L * 0.15], 12, W * 0.03, W * 0.07);
  k.runningLights(W * 0.52, W * 0.36, L * 0.15);
  if (opts.siege) {
    k.drum('barrel', W * 0.07, L * 0.6, L * 0.5, { y: -W * 0.2, sides: 12, rTop: W * 0.06 });
    for (const bz of [L * 0.34, L * 0.52, L * 0.7]) {
      k.drum('accent', W * 0.11, W * 0.03, bz, { y: -W * 0.2, sides: 10 });
      k.drum('trim', W * 0.12, W * 0.06, bz + W * 0.05, { y: -W * 0.2, sides: 10 });
    }
    k.drum('dark', W * 0.1, L * 0.04, L * 0.8, { y: -W * 0.2, sides: 12 });
    k.mast(W * 0.5, W * 0.14, W * 0.5, L * 0.3, -0.3);
  }
  for (const a of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
    k.radiator(W * 0.55, L * 0.3, Math.sin(a) * -W * 0.32, Math.cos(a) * W * 0.32, -L * 0.24, a);
  }
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

/**
 * Build a hull into `group`. Returns the kit (for the drive section and
 * anything the caller wants to attach).
 */
export function buildHull(shipType, group, size, mats, engineConfig, seed) {
  const kit = new HullKit(size, seed);
  (BUILDERS[shipType] || hullDestroyer)(kit);
  kit.driveSection(engineConfig);
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
