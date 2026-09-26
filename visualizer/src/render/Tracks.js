import * as THREE from 'three';
import { Ribbon } from './Trails.js';

/**
 * Trajectory tracks: every live ship's predicted path to the next decision
 * checkpoint (TrajectoryPredictor), drawn as a thin faction-coloured dashed
 * line (HUD, deliberately unlike a beam or tracer) that fades with
 * time-to-go and near the camera, with a tick every 5 s of battle time
 * (ticks sit at absolute times, so they slide toward the hull as the clock
 * runs), and a checkpoint ghost: diamond + 1/2-sigma dispersion rings, a
 * T+Ns label and a faint hull ghost oriented along the predicted velocity.
 *
 * In replay the "actual" mode adds the recorded future path (thin dotted)
 * and a dotted error line from the predicted to the actual checkpoint.
 * In a close-up the focused ship's track is emphasised and the others dim
 * (trackEmphasis).
 *
 * Widths are in screen pixels (tracks are HUD, not matter). All buffers are
 * allocated once per ship and updated in place.
 */

const TICK_PERIOD_S = 5;
const MARKER_DIAMOND_PX = 7;
const PRED_INTENSITY = 0.75;
const PRED_LINE_PX = 1.15;
const ACTUAL_INTENSITY = 0.6;
const NEAR_FADE_MAX = 60;   // scene units

/**
 * Per-ship track weight. `viewDist` is the camera's orbit distance (scene
 * units): at fleet range every track reads equally, in a close-up the
 * focused ship's track thickens and the others dim to faint guides.
 * Writes {intensity, width} multipliers into `out` (no allocation).
 */
export function trackEmphasis(isFocus, viewDist, out = {}) {
  const t = Math.min(1, Math.max(0, ((viewDist ?? 1e9) - 40) / 180));
  const closeK = 1 - t * t * (3 - 2 * t);
  if (isFocus) {
    out.intensity = 1.1;
    out.width = 1.2 + 0.3 * closeK;
  } else {
    out.intensity = 1 - 0.75 * closeK;
    out.width = 1 - 0.15 * closeK;
  }
  return out;
}

const FACTION_COLOR = {
  alpha: [0.16, 0.8, 1.0],
  beta: [1.0, 0.5, 0.2]
};

// ---------------------------------------------------------------------------
// Screen-space ribbon along a sampled path
// ---------------------------------------------------------------------------
const ribbonVertex = /* glsl */`
  attribute float aSide;
  attribute float aAlong;
  attribute float aU;
  attribute vec3 aTangent;
  uniform float uHalfPx;
  uniform float uPixelWorld;
  varying float vSide;
  varying float vAlong;
  varying float vU;
  varying float vDepth;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 wp = (modelMatrix * vec4(position, 1.0)).xyz;
    vec3 view = normalize(cameraPosition - wp);
    vec3 side = cross(normalize(aTangent), view);
    float sl = length(side);
    side = sl > 1e-4 ? side / sl : vec3(0.0, 1.0, 0.0);
    float depth = max(0.01, -(viewMatrix * vec4(wp, 1.0)).z);
    wp += side * aSide * uHalfPx * uPixelWorld * depth;
    vSide = aSide;
    vAlong = aAlong;
    vU = aU;
    vDepth = depth;
    gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

// Tracks are HUD, not matter: a thin dashed (predicted) or dotted (actual)
// line, never a solid glowing rod, so they cannot be mistaken for a beam or
// a tracer. The dash period is snapped to power-of-two multiples of
// uDashBase so the on-screen spacing stays ~uDashPx at every zoom (the two
// nearest levels cross-fade, like mip levels), and dashes stay pinned to
// absolute battle time, sliding toward the hull as the clock runs.
const ribbonFragment = /* glsl */`
  uniform vec3 uColor;
  uniform float uIntensity;
  uniform float uFadeEnd;
  uniform float uHalo;
  uniform float uHalfPx;
  uniform float uLinePx;
  uniform float uTickPeriod;
  uniform float uTick;
  uniform float uDashBase;
  uniform float uDashPx;
  uniform float uDuty;
  uniform float uRound;
  uniform float uNear;
  varying float vSide;
  varying float vAlong;
  varying float vU;
  varying float vDepth;
  #include <logdepthbuf_pars_fragment>
  float pattern(float period, float fw, float xpx) {
    float spacing = period / fw;                   // px between dash starts
    float d = fract(vAlong / period) * spacing;    // px since the last start
    if (uRound > 0.5) {
      float r = uLinePx * 1.35;
      return 1.0 - smoothstep(r - 0.5, r + 0.7, length(vec2(abs(d - 0.5 * spacing), xpx)));
    }
    float on = spacing * uDuty;
    float gap = d < on ? 0.0 : min(d - on, spacing - d);
    return 1.0 - smoothstep(0.0, 1.0, gap);
  }
  void main() {
    #include <logdepthbuf_fragment>
    float xpx = abs(vSide) * uHalfPx;              // px from the centre line
    float fw = max(fwidth(vAlong), 1e-6);          // along-units per pixel
    float q = xpx / uLinePx;
    float body = exp(-q * q * 1.1) + uHalo * exp(-q * q * 0.12);
    if (uDashBase > 0.0) {
      float lod = max(0.0, log2(uDashPx * fw / uDashBase));
      float l0 = floor(lod);
      float p0 = uDashBase * exp2(l0);
      float m = mix(pattern(p0, fw, xpx), pattern(p0 * 2.0, fw, xpx), lod - l0);
      body = uRound > 0.5 ? m * 1.2 : body * m;
    }
    float tick = 0.0;
    if (uTickPeriod > 0.0) {
      float spacingPx = uTickPeriod / fw;
      float d = abs(fract(vAlong / uTickPeriod + 0.5) - 0.5) * spacingPx;
      tick = uTick * (1.0 - smoothstep(0.5, 1.3, d)) * (1.0 - smoothstep(uHalfPx - 1.2, uHalfPx, xpx))
           * smoothstep(5.0, 12.0, spacingPx);
    }
    float along = mix(1.0, uFadeEnd, vU);
    // Fade stretches that pass close to the camera: in a close-up they
    // would otherwise cut across the whole frame.
    float near = uNear > 0.0 ? smoothstep(uNear * 0.1, uNear * 0.55, vDepth) : 1.0;
    gl_FragColor = vec4(uColor * (body + tick) * along * near * uIntensity, 1.0);
  }
`;

class TrackRibbon {
  /**
   * @param {number} capacity - max points
   * @param {Object} o - halfPx (geometry), linePx (line core), color [r,g,b],
   *   intensity, fadeEnd, halo, tickPeriod, tick, dashBase, dashPx, duty,
   *   round (1 = dots instead of dashes)
   */
  constructor(capacity, o) {
    const n = capacity;
    this.cap = n;
    this.pos = new Float32Array(n * 2 * 3);
    this.tan = new Float32Array(n * 2 * 3);
    this.along = new Float32Array(n * 2);
    this.u = new Float32Array(n * 2);
    const side = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { side[i * 2] = -1; side[i * 2 + 1] = 1; }
    const index = [];
    for (let i = 0; i < n - 1; i++) {
      const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
      index.push(a, b, c, b, d, c);
    }
    const geom = new THREE.BufferGeometry();
    geom.setIndex(index);
    geom.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    geom.setAttribute('aTangent', new THREE.BufferAttribute(this.tan, 3).setUsage(THREE.DynamicDrawUsage));
    geom.setAttribute('aAlong', new THREE.BufferAttribute(this.along, 1).setUsage(THREE.DynamicDrawUsage));
    geom.setAttribute('aU', new THREE.BufferAttribute(this.u, 1).setUsage(THREE.DynamicDrawUsage));
    geom.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    geom.setDrawRange(0, 0);
    this.geom = geom;
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(...o.color) },
        uIntensity: { value: o.intensity ?? 1 },
        uFadeEnd: { value: o.fadeEnd ?? 1 },
        uHalo: { value: o.halo ?? 0 },
        uHalfPx: { value: o.halfPx ?? 4 },
        uLinePx: { value: o.linePx ?? 1.2 },
        uTickPeriod: { value: o.tickPeriod ?? 0 },
        uTick: { value: o.tick ?? 0 },
        uDashBase: { value: o.dashBase ?? 0 },
        uDashPx: { value: o.dashPx ?? 16 },
        uDuty: { value: o.duty ?? 0.55 },
        uRound: { value: o.round ?? 0 },
        uNear: { value: 0 },
        uPixelWorld: Ribbon.pixelWorld
      },
      vertexShader: ribbonVertex,
      fragmentShader: ribbonFragment,
      extensions: { derivatives: true },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquation: THREE.AddEquation
    });
    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = o.renderOrder ?? 21;
    this.mesh.visible = false;
  }

  /**
   * @param {Float64Array} pts - xyz in metres
   * @param {Float64Array|null} along - per-point pattern coordinate (battle
   *   time for tracks); null = cumulative scene distance
   * @param {number} n - point count
   * @param {number} scale - metres -> scene units
   */
  set(pts, along, n, scale) {
    n = Math.min(n, this.cap);
    if (n < 2) { this.hide(); return; }
    const P = this.pos, T = this.tan, A = this.along, U = this.u;
    let dist = 0;
    for (let i = 0; i < n; i++) {
      const x = pts[i * 3] * scale, y = pts[i * 3 + 1] * scale, z = pts[i * 3 + 2] * scale;
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      let tx = (pts[b * 3] - pts[a * 3]) * scale;
      let ty = (pts[b * 3 + 1] - pts[a * 3 + 1]) * scale;
      let tz = (pts[b * 3 + 2] - pts[a * 3 + 2]) * scale;
      const tl = Math.hypot(tx, ty, tz);
      if (tl < 1e-9) { tx = 0; ty = 0; tz = 1; } else { tx /= tl; ty /= tl; tz /= tl; }
      if (i > 0 && !along) {
        dist += Math.hypot(x - P[(i - 1) * 6], y - P[(i - 1) * 6 + 1], z - P[(i - 1) * 6 + 2]);
      }
      const al = along ? along[i] : dist;
      const u = i / (n - 1);
      for (let k = 0; k < 2; k++) {
        const v = (i * 2 + k) * 3;
        P[v] = x; P[v + 1] = y; P[v + 2] = z;
        T[v] = tx; T[v + 1] = ty; T[v + 2] = tz;
        A[i * 2 + k] = al;
        U[i * 2 + k] = u;
      }
    }
    const g = this.geom.attributes;
    g.position.needsUpdate = true;
    g.aTangent.needsUpdate = true;
    g.aAlong.needsUpdate = true;
    g.aU.needsUpdate = true;
    this.geom.setDrawRange(0, (n - 1) * 6);
    this.mesh.visible = true;
    this.length = dist;
  }

  hide() {
    this.mesh.visible = false;
  }

  dispose() {
    this.geom.dispose();
    this.material.dispose();
  }
}

// ---------------------------------------------------------------------------
// Checkpoint marker: diamond + dispersion rings, or a small hollow circle
// ---------------------------------------------------------------------------
const markerVertex = /* glsl */`
  attribute vec2 aCorner;
  uniform float uRingWorld;
  uniform float uDiamondPx;
  uniform float uCirclePx;
  uniform float uPixelWorld;
  varying vec2 vPx;
  varying float vRingPx;
  varying float vRing2Px;
  varying float vRingFade;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    float pw = uPixelWorld * max(0.01, -mv.z);
    // A sub-pixel sigma still gets a (dimmed) ring hugging the diamond so
    // "dispersion present" reads at fleet zoom; the 2-sigma ring only
    // appears once it clears the 1-sigma one.
    float truePx = uRingWorld / pw;
    float minPx = uDiamondPx * 1.8;
    float ringPx = uRingWorld > 0.0 ? clamp(truePx, minPx, 1500.0) : 0.0;
    vRingFade = uRingWorld > 0.0 ? clamp(truePx / minPx, 0.3, 1.0) : 0.0;
    vRing2Px = min(2.0 * truePx, 3000.0);
    float ext = max(max(uDiamondPx, uCirclePx) * 2.6 + 4.0, max(ringPx, vRing2Px) + 4.0);
    vPx = aCorner * ext;
    vRingPx = ringPx;
    mv.xy += vPx * pw;
    gl_Position = projectionMatrix * mv;
    #include <logdepthbuf_vertex>
  }
`;

const markerFragment = /* glsl */`
  uniform vec3 uColor;
  uniform float uIntensity;
  uniform float uDiamondPx;
  uniform float uCirclePx;
  uniform float uRingOpacity;
  varying vec2 vPx;
  varying float vRingPx;
  varying float vRing2Px;
  varying float vRingFade;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float r = length(vPx);
    float c = 0.0;
    if (uDiamondPx > 0.0) {
      float edge = abs(abs(vPx.x) + abs(vPx.y) - uDiamondPx) * 0.7071;
      c += 1.6 * (1.0 - smoothstep(0.6, 1.6, edge));
      c += 1.2 * (1.0 - smoothstep(1.0, 2.2, r));
      c += 0.25 * exp(-r * r / (uDiamondPx * uDiamondPx * 1.6));
    }
    if (uCirclePx > 0.0) {
      c += 1.0 * (1.0 - smoothstep(0.5, 1.4, abs(r - uCirclePx)));
      c += 0.9 * (1.0 - smoothstep(0.6, 1.6, r));
    }
    if (vRingPx > 0.0) {
      float a = atan(vPx.y, vPx.x);
      float dash = step(0.45, fract(a * 3.8197));   // 24 dashes around
      float two = smoothstep(vRingPx + 3.0, vRingPx + 10.0, vRing2Px);
      c += uRingOpacity * vRingFade * 0.9 * (1.0 - smoothstep(0.5, 1.5, abs(r - vRingPx)));
      c += uRingOpacity * two * 0.4 * dash * (1.0 - smoothstep(0.5, 1.5, abs(r - vRing2Px)));
      c += uRingOpacity * vRingFade * 0.08 * (1.0 - smoothstep(0.0, vRingPx * 1.05, r));
    }
    if (c < 0.003) discard;
    gl_FragColor = vec4(uColor * c * uIntensity, 1.0);
  }
`;

let _quad = null;
function quadGeometry() {
  if (_quad) return _quad;
  _quad = new THREE.BufferGeometry();
  _quad.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
  _quad.setAttribute('aCorner', new THREE.BufferAttribute(new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), 2));
  _quad.setIndex([0, 1, 2, 0, 2, 3]);
  return _quad;
}

function makeMarker(color, o) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(...color) },
      uIntensity: { value: o.intensity ?? 1 },
      uDiamondPx: { value: o.diamondPx ?? 0 },
      uCirclePx: { value: o.circlePx ?? 0 },
      uRingWorld: { value: 0 },
      uRingOpacity: { value: 1 },
      uPixelWorld: Ribbon.pixelWorld
    },
    vertexShader: markerVertex,
    fragmentShader: markerFragment,
    transparent: true,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendEquation: THREE.AddEquation
  });
  const mesh = new THREE.Mesh(quadGeometry(), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 26;
  mesh.visible = false;
  return mesh;
}

// ---------------------------------------------------------------------------
// Hull ghost: the ship's own hull geometry with a faint fresnel shell
// ---------------------------------------------------------------------------
const ghostVertex = /* glsl */`
  varying vec3 vN;
  varying vec3 vV;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = -mv.xyz;
    gl_Position = projectionMatrix * mv;
    #include <logdepthbuf_vertex>
  }
`;

const ghostFragment = /* glsl */`
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec3 vN;
  varying vec3 vV;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
    gl_FragColor = vec4(uColor * (0.012 + 0.24 * pow(f, 2.4)) * uOpacity, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Label: "T+12s" on a small canvas, redrawn only when the second changes
// ---------------------------------------------------------------------------
const LABEL_W = 128, LABEL_H = 32, LABEL_PX = 15;

function makeLabel(color) {
  const canvas = document.createElement('canvas');
  canvas.width = LABEL_W;
  canvas.height = LABEL_H;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, color: new THREE.Color(...color).lerp(new THREE.Color(1, 1, 1), 0.45),
    transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false
  }));
  sprite.renderOrder = 60;
  sprite.visible = false;
  return { sprite, canvas, ctx: canvas.getContext('2d'), texture, seconds: -1 };
}

function drawLabel(label, seconds) {
  if (label.seconds === seconds) return;
  label.seconds = seconds;
  const text = `T+${seconds}s`;
  const { ctx } = label;
  ctx.clearRect(0, 0, LABEL_W, LABEL_H);
  ctx.font = "600 22px 'Share Tech Mono', 'DejaVu Sans Mono', monospace";
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.9)';
  ctx.shadowBlur = 4;
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, 4, LABEL_H / 2 + 1);
  label.texture.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Per-ship bundle
// ---------------------------------------------------------------------------
class ShipTrack {
  constructor(root, faction, samples) {
    const col = FACTION_COLOR[faction] || FACTION_COLOR.alpha;
    const pale = col.map(c => c * 0.45 + 0.5);
    this.faction = faction;
    this.color = col;
    this.pred = new TrackRibbon(samples, {
      color: col, halfPx: 4.5, linePx: PRED_LINE_PX, halo: 0.1, intensity: PRED_INTENSITY, fadeEnd: 0.22,
      tickPeriod: TICK_PERIOD_S, tick: 0.9, dashBase: TICK_PERIOD_S / 16, dashPx: 15, duty: 0.55
    });
    this.actual = new TrackRibbon(samples, {
      color: pale, halfPx: 3, linePx: 1.0, intensity: ACTUAL_INTENSITY, fadeEnd: 0.7,
      dashBase: TICK_PERIOD_S / 16, dashPx: 8, round: 1, renderOrder: 20
    });
    this.error = new TrackRibbon(2, {
      color: [0.95, 0.9, 0.85], halfPx: 3, linePx: 0.9, intensity: 0.5, dashPx: 7, round: 1, renderOrder: 20
    });
    this.marker = makeMarker(col, { diamondPx: MARKER_DIAMOND_PX, intensity: 1.0 });
    this.actualMarker = makeMarker(pale, { circlePx: 4.5, intensity: 0.8 });
    this.label = makeLabel(col);
    this.ghost = null;
    this.ghostMaterial = null;
    root.add(this.pred.mesh, this.actual.mesh, this.error.mesh, this.marker, this.actualMarker, this.label.sprite);
  }

  hide() {
    this.pred.hide();
    this.actual.hide();
    this.error.hide();
    this.marker.visible = false;
    this.actualMarker.visible = false;
    this.label.sprite.visible = false;
    if (this.ghost) this.ghost.visible = false;
  }

  hideActual() {
    this.actual.hide();
    this.error.hide();
    this.actualMarker.visible = false;
  }

  /** Clone the ship's lit hull meshes (shared geometry) under one fresnel material. */
  buildGhost(root, shipGroup) {
    this.ghostMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(...this.color) },
        uOpacity: { value: 0 }
      },
      vertexShader: ghostVertex,
      fragmentShader: ghostFragment,
      transparent: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquation: THREE.AddEquation
    });
    const ghost = new THREE.Group();
    ghost.name = `ghost:${shipGroup.name}`;
    shipGroup.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(shipGroup.matrixWorld).invert();
    shipGroup.traverse(o => {
      if (!o.isMesh || o.isInstancedMesh || o.isSkinnedMesh || !o.geometry?.attributes?.normal) return;
      const m = Array.isArray(o.material) ? o.material[0] : o.material;
      if (!m || m.isShaderMaterial || m.transparent) return;
      const g = new THREE.Mesh(o.geometry, this.ghostMaterial);
      g.matrixAutoUpdate = false;
      g.matrix.multiplyMatrices(inv, o.matrixWorld);
      g.frustumCulled = false;
      g.renderOrder = 19;
      ghost.add(g);
    });
    ghost.visible = false;
    root.add(ghost);
    this.ghost = ghost;
    this.ghostLength = shipGroup.userData.size?.length || 8;
  }

  dispose() {
    this.pred.dispose();
    this.actual.dispose();
    this.error.dispose();
    this.marker.material.dispose();
    this.actualMarker.material.dispose();
    this.label.sprite.material.dispose();
    this.label.texture.dispose();
    this.ghostMaterial?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Public layer
// ---------------------------------------------------------------------------
export class Tracks {
  /**
   * @param {SceneManager} sceneManager - owns the THREE scene, camera, ships
   * @param {TrajectoryPredictor} predictor
   */
  constructor(sceneManager, predictor) {
    this.sm = sceneManager;
    this.predictor = predictor;
    this.scale = sceneManager.SCALE ?? 1 / 1000;
    this.root = new THREE.Group();
    this.root.name = 'trajectoryTracks';
    this.root.visible = false;
    sceneManager.scene.add(this.root);
    // SceneManager.updatePredictedPaths(null) clears through this
    sceneManager.tracks = this;
    this.visuals = new Map();
    this.showGhosts = true;
    this.showLabels = true;
    this._p = new THREE.Vector3();
    this._q = new THREE.Vector3();
  }

  /**
   * Per-frame entry point.
   * @param {number} time - playback time
   * @param {Object} ships - interpolated ship states (anchor positions)
   * @param {'off'|'predicted'|'actual'} mode
   * @param {string|null} [focusId] - selected / camera-followed ship; when
   *   absent, the ship the camera is orbiting (nearest the look-at point)
   */
  frame(time, ships, mode, focusId = null) {
    if (mode === 'off' || !this.predictor || !ships) {
      this.root.visible = false;
      return;
    }
    this.root.visible = true;
    const withActual = mode === 'actual';
    this.focusId = focusId || this._implicitFocus();
    this.update(this.predictor.predict(time, ships, withActual), time, withActual);
  }

  /** The ship under the orbit target, if the camera is looking at one. */
  _implicitFocus() {
    const target = this.sm.controls?.target;
    if (!target || !this.sm.ships) return null;
    const viewDist = this.sm.camera.position.distanceTo(target);
    let best = null, bd = Infinity;
    for (const [id, ship] of this.sm.ships) {
      if (!ship.visible) continue;
      const d = ship.position.distanceTo(target);
      const L = ship.userData.size?.length || 8;
      if (d < bd && d < L + 0.35 * viewDist) { bd = d; best = id; }
    }
    return best;
  }

  update(tracks, time, withActual) {
    const S = this.scale;
    const cam = this.sm.camera;
    const pw = Ribbon.pixelWorld.value;
    const viewDist = this.sm.controls ? cam.position.distanceTo(this.sm.controls.target) : 1e9;
    const emph = this._emph || (this._emph = {});
    // Near-camera fade scale: the orbit distance in a close-up, capped so
    // fleet-range views (where nothing passes the lens) keep every track
    const nearFade = Math.min(viewDist, NEAR_FADE_MAX);
    for (const tr of tracks.values()) {
      const id = tr.shipId;
      let v = this.visuals.get(id);
      if (!tr.active) {
        if (v) v.hide();
        continue;
      }
      if (!v) {
        v = new ShipTrack(this.root, tr.faction, this.predictor.samples);
        this.visuals.set(id, v);
      }
      trackEmphasis(id === this.focusId, viewDist, emph);
      const pu = v.pred.material.uniforms;
      pu.uIntensity.value = PRED_INTENSITY * emph.intensity;
      pu.uLinePx.value = PRED_LINE_PX * emph.width;
      pu.uNear.value = nearFade;
      v.actual.material.uniforms.uIntensity.value = ACTUAL_INTENSITY * emph.intensity;
      v.actual.material.uniforms.uNear.value = nearFade;
      v.error.material.uniforms.uIntensity.value = 0.5 * emph.intensity;

      v.pred.set(tr.path, tr.times, tr.n, S);

      const cp = this._p.set(tr.cp[0] * S, tr.cp[1] * S, tr.cp[2] * S);
      const togo = tr.tCp - time;
      const u = v.marker.material.uniforms;
      u.uRingWorld.value = tr.sigma * S;
      // Dim the marker as it folds into the hull at checkpoint arrival
      u.uIntensity.value = Math.min(1, Math.max(0.25, togo / 3)) * Math.min(1, emph.intensity);
      v.marker.position.copy(cp);
      v.marker.visible = true;
      v.actualMarker.material.uniforms.uIntensity.value = 0.8 * Math.min(1, emph.intensity);

      if (this.showLabels) {
        drawLabel(v.label, Math.max(0, Math.ceil(togo - 1e-6)));
        const sp = v.label.sprite;
        sp.position.copy(cp);
        sp.scale.set(LABEL_PX * (LABEL_W / LABEL_H) * pw, LABEL_PX * pw, 1);
        sp.material.opacity = Math.min(1, emph.intensity);
        // right of the diamond, clear of the 1-sigma ring
        const ringPx = tr.sigma > 0 ? Math.max(MARKER_DIAMOND_PX * 1.8, (tr.sigma * S) / (pw * Math.max(1e-6, cam.position.distanceTo(cp)))) : 0;
        sp.center.set(-Math.min(400, Math.max(MARKER_DIAMOND_PX + 5, ringPx + 5)) / (LABEL_PX * LABEL_W / LABEL_H), 0.5);
        // only when the track is long enough on screen to need a caption
        const distCp = Math.max(1e-6, cam.position.distanceTo(cp));
        const sepPx = Math.hypot(tr.path[0] * S - cp.x, tr.path[1] * S - cp.y, tr.path[2] * S - cp.z) / (pw * distCp);
        sp.visible = togo > 0.5 && sepPx > 48;
      } else {
        v.label.sprite.visible = false;
      }

      this._updateGhost(v, id, tr, cp, cam, pw, Math.min(1, emph.intensity));

      if (withActual && tr.actual.n >= 2) {
        v.actual.set(tr.actual.path, tr.actual.times, tr.actual.n, S);
        if (tr.actual.hasCp) {
          const a = tr.actual.cp;
          v.actualMarker.position.set(a[0] * S, a[1] * S, a[2] * S);
          v.actualMarker.visible = true;
          this._setErrorLine(v, tr);
        } else {
          v.actualMarker.visible = false;
          v.error.hide();
        }
      } else {
        v.hideActual();
      }
    }
    for (const [id, v] of this.visuals) {
      if (!tracks.has(id)) v.hide();
    }
  }

  _setErrorLine(v, tr) {
    const pts = this._err || (this._err = new Float64Array(6));
    const a = tr.actual.cp, p = tr.cp;
    pts[0] = p[0]; pts[1] = p[1]; pts[2] = p[2];
    pts[3] = a[0]; pts[4] = a[1]; pts[5] = a[2];
    const len = Math.hypot(a[0] - p[0], a[1] - p[1], a[2] - p[2]) * this.scale;
    if (len < 1e-4) { v.error.hide(); return; }
    // ~14 dots along the miss (fewer when that would crowd the screen)
    v.error.material.uniforms.uDashBase.value = len / 14;
    v.error.set(pts, null, 2, this.scale);
  }

  _updateGhost(v, id, tr, cp, cam, pw, weight = 1) {
    const ship = this.sm.ships?.get(id);
    if (!this.showGhosts || !ship) {
      if (v.ghost) v.ghost.visible = false;
      return;
    }
    if (!v.ghost) v.buildGhost(this.root, ship);
    const L = v.ghostLength;
    const dist = cam.position.distanceTo(cp);
    const px = L / Math.max(1e-6, pw * dist);
    // Only when the hull is big enough on screen to read, and not while the
    // ghost still overlaps the real hull
    const sep = ship.position.distanceTo(cp);
    const opacity = 0.75 * weight * THREE.MathUtils.smoothstep(px, 14, 40) * THREE.MathUtils.smoothstep(sep, L * 0.8, L * 2.0);
    if (opacity < 0.01) {
      v.ghost.visible = false;
      return;
    }
    v.ghostMaterial.uniforms.uOpacity.value = opacity;
    v.ghost.position.copy(cp);
    const d = tr.cpVel;
    const dl = Math.hypot(d[0], d[1], d[2]);
    if (dl > 1e-3) {
      v.ghost.lookAt(this._q.set(cp.x + d[0] / dl, cp.y + d[1] / dl, cp.z + d[2] / dl));
    } else {
      v.ghost.quaternion.copy(ship.quaternion);
    }
    v.ghost.visible = true;
  }

  /** Hide everything until the next frame() (mode changes, live end, reset). */
  clear() {
    for (const v of this.visuals.values()) v.hide();
    this.root.visible = false;
  }

  dispose() {
    for (const v of this.visuals.values()) v.dispose();
    this.visuals.clear();
    this.sm.scene.remove(this.root);
  }
}
