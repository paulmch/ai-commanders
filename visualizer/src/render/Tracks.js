import * as THREE from 'three';
import { Ribbon } from './Trails.js';
import { LOCUS_CAP } from '../TrajectoryPredictor.js';
import {
  hullExaggeration, magnifyRadius, magnifyOffset, formatCheckpointLabel, isReactiveMode,
  inTrackScope, GLYPH_RMAX_HULLS, ghostLook, capsuleGap2D
} from '../TrackDisplay.js';

/**
 * Trajectory tracks: every live ship's predicted path to the next decision
 * checkpoint (TrajectoryPredictor), drawn as a thin faction-coloured dashed
 * line (HUD, deliberately unlike a beam or tracer) that fades with
 * time-to-go and near the camera, with a tick every 5 s of battle time
 * (ticks sit at absolute times, so they slide toward the hull as the clock
 * runs), and a checkpoint ghost: diamond + 1/2-sigma dispersion rings and a
 * faint hull ghost along the predicted nose (plan.fwd, else the current one).
 *
 * The rings and a faint dotted "drift locus" (where this interval's earlier
 * frames predicted the same checkpoint) are drawn at the hull's own visual
 * exaggeration (TrackDisplay.js), so the evasion wiggle is as visible as the
 * ships are big; the focused ship's caption states the true numbers
 * ("T+25s ±180 m"). A plan traced before the checkpoint's new orders
 * applied (plan.stale) gets a hollow, desaturated diamond; a reactive
 * evasion mode (low confidence) a dashed ring and dashed locus. The
 * time-to-go lives in one HUD readout (TrackHud.js), not on every marker.
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
const LOCUS_INTENSITY = 1.0;
const NEAR_FADE_MAX = 60;   // scene units
const IMPLICIT_FOCUS_MAX = 220;   // orbit distance beyond which nothing is implicitly focused
// Marker and caption fade out within ~1.5 displayed hull lengths of the
// ship: for a slow hull the checkpoint sits on the hull itself
const HULL_CLEAR_LO = 1.2, HULL_CLEAR_HI = 1.8;

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
    out.intensity = 1 - 0.92 * closeK;
    out.width = 1 - 0.3 * closeK;
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
      // dots (optionally threaded on a faint line: uHalo) or dashes
      body = uRound > 0.5 ? m * 1.2 + uHalo * exp(-q * q * 1.1) : body * m;
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
  uniform float uRing2World;
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
    vRing2Px = min(uRing2World / pw, 3000.0);
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
  uniform float uHollow;
  uniform float uLowConf;
  varying vec2 vPx;
  varying float vRingPx;
  varying float vRing2Px;
  varying float vRingFade;
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float r = length(vPx);
    float a = atan(vPx.y, vPx.x);
    float c = 0.0;
    if (uDiamondPx > 0.0) {
      float edge = abs(abs(vPx.x) + abs(vPx.y) - uDiamondPx) * 0.7071;
      // provisional (orders pending): hollow, broken outline, no centre dot
      float broken = mix(1.0, 0.35 + 0.65 * step(0.42, fract(a * 1.2732 + 0.25)), uHollow);
      c += 1.6 * (1.0 - smoothstep(0.6, 1.6, edge)) * broken * (1.0 - 0.3 * uHollow);
      c += 1.2 * (1.0 - smoothstep(1.0, 2.2, r)) * (1.0 - uHollow);
      c += 0.25 * exp(-r * r / (uDiamondPx * uDiamondPx * 1.6)) * (1.0 - uHollow);
    }
    if (uCirclePx > 0.0) {
      c += 1.0 * (1.0 - smoothstep(0.5, 1.4, abs(r - uCirclePx)));
      c += 0.9 * (1.0 - smoothstep(0.6, 1.6, r));
    }
    if (vRingPx > 0.0) {
      float dash = step(0.45, fract(a * 3.8197));   // 24 dashes around
      // low confidence (reactive evasion): the 1-sigma ring breaks into 8 long dashes
      float dash1 = mix(1.0, step(0.38, fract(a * 1.2732)), uLowConf);
      float two = smoothstep(vRingPx + 3.0, vRingPx + 10.0, vRing2Px);
      // a screen-filling ring (big spread, close-up) stays a light outline
      float big = smoothstep(120.0, 360.0, vRingPx);
      c += uRingOpacity * vRingFade * 0.9 * (1.0 - 0.4 * big) * dash1 * (1.0 - smoothstep(0.5, 1.5, abs(r - vRingPx)));
      c += uRingOpacity * two * 0.4 * dash * (1.0 - smoothstep(0.5, 1.5, abs(r - vRing2Px)));
      c += uRingOpacity * vRingFade * (0.08 + 0.05 * uLowConf) * (1.0 - big) * (1.0 - smoothstep(0.0, vRingPx * 1.05, r));
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
      uRing2World: { value: 0 },
      uRingOpacity: { value: 1 },
      uHollow: { value: 0 },
      uLowConf: { value: 0 },
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
// Label: "T+12s ±180 m" on a small canvas, redrawn only when the text changes
// (only the focused ship carries one; the countdown lives in the HUD)
// ---------------------------------------------------------------------------
const LABEL_W = 384, LABEL_H = 32, LABEL_PX = 17;
const LABEL_NEAR_MAX_PX = 110;   // rings wider than this keep the caption by the diamond

function makeLabel(color) {
  const canvas = document.createElement('canvas');
  canvas.width = LABEL_W;
  canvas.height = LABEL_H;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  // The canvas is drawn at ~2x and minified through the mip chain: plain
  // linear sampling of the 32-texel strip at 15-25 px smeared the glyphs,
  // worst at gfx low where the pixel ratio is capped at 1.
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, color: new THREE.Color(...color).lerp(new THREE.Color(1, 1, 1), 0.45),
    transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false
  }));
  sprite.renderOrder = 60;
  sprite.visible = false;
  return { sprite, canvas, ctx: canvas.getContext('2d'), texture, text: null };
}

function drawLabel(label, text) {
  if (label.text === text) return;
  label.text = text;
  const { ctx } = label;
  ctx.clearRect(0, 0, LABEL_W, LABEL_H);
  ctx.font = "700 24px 'Share Tech Mono', 'DejaVu Sans Mono', monospace";
  ctx.textBaseline = 'middle';
  // A hard dark rim instead of a blurred shadow: it survives the 2x
  // minification and keeps the caption legible over a bright track.
  ctx.lineJoin = 'round';
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.strokeText(text, 4, LABEL_H / 2 + 1, LABEL_W - 8);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, 4, LABEL_H / 2 + 1, LABEL_W - 8);
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
    // Drift locus: one dot per earlier prediction, newest bright, oldest faint
    this.locus = new TrackRibbon(LOCUS_CAP, {
      color: pale, halfPx: 3.5, linePx: 1.15, halo: 0.22, intensity: LOCUS_INTENSITY, fadeEnd: 0.3,
      dashBase: 1, dashPx: 6, duty: 0.5, round: 1, renderOrder: 20
    });
    this.marker = makeMarker(col, { diamondPx: MARKER_DIAMOND_PX, intensity: 1.0 });
    // provisional (stale plan) markers are desaturated toward grey
    const lum = 0.3 * col[0] + 0.55 * col[1] + 0.15 * col[2];
    this.staleColor = new THREE.Color(...col.map(c => c * 0.3 + lum * 0.55 + 0.1));
    this.liveColor = new THREE.Color(...col);
    this.actualMarker = makeMarker(pale, { circlePx: 4.5, intensity: 0.8 });
    this.label = makeLabel(col);
    this.ghost = null;
    this.ghostMaterial = null;
    root.add(this.pred.mesh, this.actual.mesh, this.error.mesh, this.locus.mesh, this.marker, this.actualMarker, this.label.sprite);
  }

  hide() {
    this.pred.hide();
    this.actual.hide();
    this.error.hide();
    this.locus.hide();
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
    this.locus.dispose();
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
    this.hud = null;   // TrackHud (main.js): countdown, scope, legend
    this._p = new THREE.Vector3();
    this._q = new THREE.Vector3();
    this._d = new Float64Array(3);
    this._locusPts = new Float64Array(LOCUS_CAP * 3);
    this._ghostLook = { opacity: 0, scale: 1 };
    this._hud = { visible: true, togo: 0, stale: false, scope: 'all', focusId: null, exaggeration: [0, 0] };
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
      this.hud?.hide();
      return;
    }
    this.root.visible = true;
    const withActual = mode === 'actual';
    this.focusId = focusId || this._implicitFocus();
    this._states = ships;   // focused caption: maneuver + throttle
    this.update(this.predictor.predict(time, ships, withActual), time, withActual);
  }

  /**
   * The ship under the orbit target, if the camera is looking at one. Not
   * at fleet range: there the target is the fleet centroid, and a caption
   * (or the SEL scope) on whichever hull happens to sit near it would be noise.
   */
  _implicitFocus() {
    const target = this.sm.controls?.target;
    if (!target || !this.sm.ships) return null;
    const viewDist = this.sm.camera.position.distanceTo(target);
    if (viewDist > IMPLICIT_FOCUS_MAX) return null;
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
    const scope = this.hud?.scope || 'all';
    // Near-camera fade scale: the orbit distance in a close-up, capped so
    // fleet-range views (where nothing passes the lens) keep every track
    const nearFade = Math.min(viewDist, NEAR_FADE_MAX);
    let anyStale = false, focusCp = null, kMin = Infinity, kMax = 0;
    for (const tr of tracks.values()) {
      const id = tr.shipId;
      let v = this.visuals.get(id);
      if (!tr.active || !inTrackScope(scope, id, this.focusId)) {
        if (v) v.hide();
        continue;
      }
      if (!v) {
        v = new ShipTrack(this.root, tr.faction, this.predictor.samples);
        this.visuals.set(id, v);
      }
      const isFocus = id === this.focusId;
      const ship = this.sm.ships?.get(id);
      const L = ship?.userData.size?.length || 8;
      // rings and drift are drawn at the hull's own exaggeration, soft-capped
      const k = hullExaggeration(ship?.userData.shipType, L);
      const rMax = (GLYPH_RMAX_HULLS * L) / S;
      kMin = Math.min(kMin, k); kMax = Math.max(kMax, k);
      trackEmphasis(isFocus, viewDist, emph);
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
      if (isFocus) focusCp = tr.tCp;
      if (tr.stale) anyStale = true;
      const clear = ship ? THREE.MathUtils.smoothstep(ship.position.distanceTo(cp), HULL_CLEAR_LO * L, HULL_CLEAR_HI * L) : 1;
      const lowConf = isReactiveMode(tr.mode);
      const u = v.marker.material.uniforms;
      const ringWorld = magnifyRadius(tr.sigma, k, rMax) * S;
      u.uRingWorld.value = ringWorld;
      // Only the focused ship gets the 2-sigma ring and full-strength rings:
      // at fleet range sixteen overlapping ring sets fused into bright blobs
      u.uRing2World.value = isFocus ? magnifyRadius(2 * tr.sigma, k, rMax) * S : 0;
      u.uRingOpacity.value = isFocus ? 1 : 0.5;
      u.uHollow.value = tr.stale ? 1 : 0;
      u.uLowConf.value = lowConf ? 1 : 0;
      u.uColor.value.copy(tr.stale ? v.staleColor : v.liveColor);
      // Dim the marker as it folds into the hull at checkpoint arrival
      const markerW = Math.min(1, Math.max(0.25, togo / 3)) * Math.min(1, emph.intensity) * clear;
      u.uIntensity.value = markerW;
      v.marker.position.copy(cp);
      v.marker.visible = markerW > 0.005;
      v.actualMarker.material.uniforms.uIntensity.value = 0.8 * Math.min(1, emph.intensity);

      this._setLocus(v, tr, k, rMax, markerW * (isFocus ? 1 : 0.75), lowConf);

      if (this.showLabels && isFocus && clear > 0.01) {
        const st = this._states?.[id];
        drawLabel(v.label, formatCheckpointLabel({ togo, sigma: tr.sigma, stale: tr.stale, mode: tr.mode,
          maneuver: st?.maneuver, throttle: st?.thrust }));
        const sp = v.label.sprite;
        sp.position.copy(cp);
        sp.scale.set(LABEL_PX * (LABEL_W / LABEL_H) * pw, LABEL_PX * pw, 1);
        sp.material.opacity = clear;
        // right of the diamond, clear of the rings (the 2-sigma one once drawn)
        const distCp = Math.max(1e-6, cam.position.distanceTo(cp));
        const ringPx = ringWorld > 0 ? Math.max(MARKER_DIAMOND_PX * 1.8, ringWorld / (pw * distCp)) : 0;
        const ring2Px = u.uRing2World.value / (pw * distCp);
        const outerPx = ring2Px > ringPx + 6 ? ring2Px : ringPx;
        // a ring wider than the caption itself: the caption stays by the diamond
        const offPx = outerPx > LABEL_NEAR_MAX_PX ? MARKER_DIAMOND_PX + 6 : Math.max(MARKER_DIAMOND_PX + 5, outerPx + 5);
        sp.center.set(-offPx / (LABEL_PX * LABEL_W / LABEL_H), 0.5);
        // only when the track is long enough on screen to need a caption
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
    if (this.hud) {
      const h = this._hud;
      h.visible = true;
      h.togo = (focusCp ?? this.predictor.checkpointAfter(time)) - time;
      h.stale = anyStale;
      h.scope = scope;
      h.focusId = this.focusId;
      h.exaggeration[0] = Number.isFinite(kMin) ? kMin : 0;
      h.exaggeration[1] = kMax;
      this.hud.update(h);
    }
  }

  /**
   * Drift locus around the marker: earlier predictions of this checkpoint,
   * their offsets from the current one magnified like the rings.
   */
  _setLocus(v, tr, k, rMax, weight, lowConf) {
    const L = tr.locus;
    if (!L || L.n < 2 || weight < 0.01) { v.locus.hide(); return; }
    const pts = this._locusPts, d = this._d, c = tr.cp;
    let extent = 0;
    for (let i = 0; i < L.n; i++) {
      const o = i * 3;
      d[0] = L.pts[o] - c[0]; d[1] = L.pts[o + 1] - c[1]; d[2] = L.pts[o + 2] - c[2];
      magnifyOffset(d, k, rMax, pts, 0, o);
      extent = Math.max(extent, Math.abs(pts[o]) + Math.abs(pts[o + 1]) + Math.abs(pts[o + 2]));
      pts[o] += c[0]; pts[o + 1] += c[1]; pts[o + 2] += c[2];
    }
    if (extent < 1) { v.locus.hide(); return; }   // the prediction has not moved
    const lu = v.locus.material.uniforms;
    lu.uIntensity.value = LOCUS_INTENSITY * weight;
    lu.uRound.value = lowConf ? 0 : 1;
    lu.uDuty.value = lowConf ? 0.6 : 0.5;
    v.locus.set(pts, L.times, L.n, this.scale);
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
    const viewPx = 2 * Math.tan(THREE.MathUtils.degToRad(cam.fov || 55) / 2) / Math.max(1e-9, pw);
    // Only when the hull is big enough on screen to read, and not while the
    // ghost still overlaps the real hull
    const sep = ship.position.distanceTo(cp);
    let opacity = 0.75 * weight * THREE.MathUtils.smoothstep(sep, L * 0.8, L * 2.0);
    // Near the lens: shrink to the screen-size cap and fade out
    let look = ghostLook(px, viewPx, Infinity, this._ghostLook);
    if (opacity * look.opacity < 0.01) {
      v.ghost.visible = false;
      return;
    }
    // The predicted nose at the checkpoint when the plan carries it; else
    // the current attitude (a ship flies nose-first only while burning, so
    // the velocity was the wrong guess for a coasting or braking hull)
    const f = tr.cpFwd;
    v.ghost.position.copy(cp);
    if (tr.hasFwd) {
      v.ghost.lookAt(this._q.set(cp.x + f[0], cp.y + f[1], cp.z + f[2]));
    } else {
      v.ghost.quaternion.copy(ship.quaternion);
    }
    v.ghost.scale.setScalar(look.scale);
    // Never over the focused hull: fade a ghost in front of it as their
    // silhouettes close in
    const gap = this._ghostCoverGap(v.ghost, L * look.scale, (ship.userData.size?.width || L * 0.2) * look.scale, cam, pw);
    if (gap !== Infinity) look = ghostLook(px, viewPx, gap, this._ghostLook);
    opacity *= look.opacity;
    if (opacity < 0.01) {
      v.ghost.visible = false;
      return;
    }
    v.ghostMaterial.uniforms.uOpacity.value = opacity;
    v.ghost.visible = true;
  }

  /**
   * Screen clearance (px) between a ghost (placed and oriented, drawn
   * length/beam in scene units) and the focused ship when the ghost sits in
   * front of it; Infinity when there is no focused ship or the ghost is
   * behind it (the hull's depth then hides the overlap).
   */
  _ghostCoverGap(ghost, len, beam, cam, pw) {
    const focus = this.focusId ? this.sm.ships?.get(this.focusId) : null;
    if (!focus || !focus.visible) return Infinity;
    const fL = focus.userData.size?.length || 8, fW = focus.userData.size?.width || fL * 0.2;
    const V = cam.matrixWorldInverse;
    const P = this._proj || (this._proj = {
      c: new THREE.Vector3(), fwd: new THREE.Vector3(), g0: [0, 0], g1: [0, 0], f0: [0, 0], f1: [0, 0]
    });
    const gDepth = -P.c.copy(ghost.position).applyMatrix4(V).z;
    const fDepth = -P.c.copy(focus.position).applyMatrix4(V).z;
    if (!(gDepth > 0) || !(fDepth > 0) || gDepth > fDepth + 0.25 * fL) return Infinity;
    const near = cam.near || 0.1;
    this._spinePx(ghost, len / 2, V, pw, near, P.g0, P.g1);
    this._spinePx(focus, fL / 2, V, pw, near, P.f0, P.f1);
    return capsuleGap2D(P.g0, P.g1, beam / 2 / (gDepth * pw), P.f0, P.f1, fW / 2 / (fDepth * pw));
  }

  /** A hull's spine (centre +- half along its +Z) in screen pixels from the view centre. */
  _spinePx(obj, half, V, pw, near, s0, s1) {
    const P = this._proj;
    P.fwd.set(0, 0, half).applyQuaternion(obj.quaternion);
    for (let k = 0; k < 2; k++) {
      const out = k ? s1 : s0;
      const q = P.c.copy(obj.position).addScaledVector(P.fwd, k ? -1 : 1).applyMatrix4(V);
      const d = Math.max(near, -q.z);
      out[0] = q.x / (d * pw);
      out[1] = q.y / (d * pw);
    }
  }

  /** Hide everything until the next frame() (mode changes, live end, reset). */
  clear() {
    for (const v of this.visuals.values()) v.hide();
    this.root.visible = false;
    this.hud?.hide();
  }

  dispose() {
    for (const v of this.visuals.values()) v.dispose();
    this.visuals.clear();
    this.sm.scene.remove(this.root);
    this.hud?.hide();
  }
}
