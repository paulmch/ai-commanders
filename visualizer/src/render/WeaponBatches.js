import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Ribbon } from './Trails.js';
import { RCS_LAYOUT, ENGINE } from './TorpedoRcs.js';

const additive = {
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  blending: THREE.CustomBlending, blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
};

/**
 * Ribbon seed with the firing faction folded in: kinds that tint by side
 * (slug tracer halo and wake, torpedo wake, engagement line) decode it as
 * code = floor(w / 2) - 0 neutral, 1 alpha, 2 beta - and seed = w - 2 code.
 */
export function factionSeed(seed, faction) {
  const s = seed - Math.floor(seed);
  return s + (faction === 'alpha' ? 2 : faction === 'beta' ? 4 : 0);
}

// One draw for all streaks, curved trail segments, motor plumes, PD ribbons
// and torpedo RCS jets. Kinds: 0 slug tracer, 1 slug wake, 2 PD beam,
// 3 motor plume, 4 spark, 5 retarget dash, 6 torpedo wake, 7 RCS jet,
// 8 engagement line (fleet range: who is shooting at whom).
const ribbonVertex = /* glsl */`
  attribute vec3 aFrom;
  attribute vec3 aTo;
  attribute vec4 aStyle; // width, intensity, kind, seed
  attribute vec2 aAge;   // trail age at each endpoint
  uniform float uPixelWorld;
  varying vec2 vUv;
  varying vec4 vStyle;
  varying float vAge;
  varying float vCover;
  varying float vPx;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 from = (viewMatrix * vec4(aFrom, 1.0)).xyz;
    vec3 to = (viewMatrix * vec4(aTo, 1.0)).xyz;
    vec3 p = mix(from, to, uv.x);
    vec2 d = to.xy - from.xy;
    d = length(d) > 0.00001 ? normalize(d) : vec2(1.0, 0.0);
    float pixel = uPixelWorld * max(0.01, -p.z);
    float kind = aStyle.z;
    bool plume = kind > 2.5 && kind < 3.5;
    bool engage = kind > 7.5;
    bool jet = kind > 6.5 && !engage;
    // Half-width floor >= 1 px: a narrower strip rasterises as beads.
    float floorPx = kind < 0.5 ? 1.1 : kind < 1.5 ? 1.0 : kind < 2.5 ? 0.6 : plume ? 1.2 : engage ? 0.75 : jet ? 0.7 : 1.0;
    float capPx = plume ? 64.0 : engage ? 0.75 : jet ? 16.0 : kind > 5.5 ? 24.0 : kind > 1.5 && kind < 2.5 ? 2.2 : 3.0;
    float width = max(pixel * floorPx, min(aStyle.x, pixel * capPx));
    // Thin effects retain a controlled subpixel contribution at fleet scale.
    float minCover = kind < 0.5 ? 0.5 : kind < 1.5 ? 0.07 : kind < 2.5 ? 0.35 : engage ? 1.0 : jet ? 0.1 : kind > 5.5 ? 0.12 : 0.3;
    vCover = max(minCover, min(1.0, aStyle.x / width));
    // Half-width in pixels: the fragment keeps the bright core >= ~1.5 px
    // wide so a thin line never alternates hit/miss (beading) along its run.
    vPx = width / pixel;
    // Plumes flare out of the nozzle, then neck down slightly toward the tip.
    // RCS jets are tight at the nozzle and fan out as the gas expands.
    float taper = plume ? mix(0.3, 1.0, smoothstep(0.0, 0.3, uv.x)) * (1.0 - 0.35 * uv.x)
      : jet ? mix(0.3, 1.0, sqrt(uv.x)) : 1.0;
    p.xy += vec2(-d.y, d.x) * position.y * 2.0 * width * taper;
    gl_Position = projectionMatrix * vec4(p, 1.0);
    vUv = uv;
    vStyle = aStyle;
    vAge = mix(aAge.x, aAge.y, uv.x);
    #include <logdepthbuf_vertex>
  }
`;

const ribbonFragment = /* glsl */`
  uniform float uTime;
  varying vec2 vUv;
  varying vec4 vStyle;
  varying float vAge;
  varying float vCover;
  varying float vPx;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    float x = vUv.x; // hot head / nozzle -> tail
    float y = abs(vUv.y * 2.0 - 1.0);
    float aa = 0.5 * vPx * vPx;
    float core = exp(-y*y*min(30.0, aa));
    float halo = exp(-y*y*min(4.0, aa)) * (1.0-y*smoothstep(1.0, 3.0, vPx));
    float ends = smoothstep(0.0, 0.025, x) * (1.0-smoothstep(0.75, 1.0, x));
    vec3 color;
    float kind = vStyle.z;
    // firing side folded into the seed (factionSeed): 0 neutral, 1 alpha, 2 beta
    float side = floor(vStyle.w * 0.5 + 0.001);
    float seed = vStyle.w - 2.0 * side;
    if (kind < 0.5) {
      // White-hot core for every slug; the halo carries the firing side
      vec3 edge = side < 0.5 ? mix(vec3(1.6,0.36,0.055), vec3(0.06,0.6,0.85), smoothstep(0.2,1.0,x))
        : side < 1.5 ? mix(vec3(0.25,0.95,1.9), vec3(0.04,0.4,1.1), smoothstep(0.2,1.0,x))
        : mix(vec3(1.9,0.62,0.08), vec3(1.1,0.2,0.03), smoothstep(0.2,1.0,x));
      color = (vec3(3.4,3.05,2.65)*core + edge*halo*0.75) * pow(1.0-x,0.45) * ends;
    } else if (kind < 1.5) {
      // Slug wake: ionised track cooling from straw-white to the side's tint
      float cooling = pow(max(0.0,1.0-vAge),1.6);
      float wisps = 0.75+0.25*sin(x*13.0 + seed*51.0 - uTime*2.0);
      vec3 cold = side < 0.5 ? vec3(0.04,0.26,0.4) : side < 1.5 ? vec3(0.03,0.3,0.7) : vec3(0.55,0.13,0.02);
      color = mix(cold,vec3(1.0,0.72,0.4),cooling*cooling)
        * (core*0.6 + halo*0.5) * cooling * wisps;
    } else if (kind < 2.5) {
      // PD beam: warm white, chopped into energy packets marching out to
      // the round, so a beam never reads as a slug tracer
      float packets = 0.42+0.58*smoothstep(0.25,0.55,fract(x*26.0-uTime*7.0+seed*17.0));
      // Several turrets share a line of fire: keep pixel-floor beams slim.
      float slim = mix(0.6, 1.0, smoothstep(0.7, 2.0, vPx));
      // aAge fades a close-up's split beam ends out toward the cut (0 = full)
      color = (vec3(1.1,0.77,0.5)*core + vec3(0.5,0.13,0.035)*halo*0.3)*packets*slim*(1.0-vAge);
    } else if (kind < 3.5) {
      // Motor plume: broad white-hot core out of the nozzle with shock
      // cells, wrapped in a longer orange sheath that fades to the tip.
      float cells = pow(0.5+0.5*cos(x*47.0),10.0)*exp(-x*4.0);
      float ripple = 0.87+0.13*sin(x*67.0-uTime*19.0+seed*21.0);
      float hot = exp(-y*y*min(7.0, aa)) * exp(-x*3.8) * (1.0+cells);
      float sheath = exp(-y*y*2.2) * (1.0-y) * exp(-x*1.7);
      color = (vec3(2.6,2.2,1.7)*hot + vec3(1.35,0.45,0.12)*sheath)
        * smoothstep(0.0,0.03,x) * (1.0-smoothstep(0.55,1.0,x)) * ripple;
    } else if (kind < 4.5) {
      color = mix(vec3(2.5,2.15,1.65),vec3(1.0,0.16,0.015),vAge)*core*ends;
    } else if (kind < 5.5) {
      float dash = smoothstep(0.25,0.4,fract(x*32.0));
      color = vec3(0.08,0.55,0.5)*halo*dash;
    } else if (kind < 6.5) {
      // Torpedo exhaust wake: hot orange near the motor, cooling aft into
      // the launching side's tint
      float cooling = pow(max(0.0,1.0-vAge),1.4);
      float wisps = 0.7+0.3*sin(x*9.0 + seed*37.0 - uTime*1.3);
      vec3 cold = side < 0.5 ? vec3(0.22,0.07,0.03) : side < 1.5 ? vec3(0.05,0.2,0.48) : vec3(0.42,0.09,0.02);
      color = mix(cold,vec3(1.5,0.72,0.3),cooling*cooling)
        * (core*0.7 + halo*0.45) * cooling * wisps;
    } else if (kind > 7.5) {
      // Engagement line: a thin faction-coloured dash marching from the
      // shooter to its target; aAge carries the dash count for the length
      float dash = smoothstep(0.3, 0.42, fract(x * vAge - uTime * 1.2)) * (1.0 - smoothstep(0.8, 0.92, fract(x * vAge - uTime * 1.2)));
      vec3 tint = side < 1.5 ? vec3(0.16,0.82,1.0) : vec3(1.0,0.54,0.24);
      color = tint * core * (0.35 + 0.65 * dash) * smoothstep(0.0, 0.04, x) * (1.0 - smoothstep(0.9, 1.0, x));
    } else {
      // RCS jet: white-hot at the nozzle, cooling to pale blue as it thins
      float fall = exp(-x*2.2) * smoothstep(0.0,0.04,x) * (1.0-smoothstep(0.5,1.0,x));
      color = (vec3(2.6,2.8,3.2)*core*exp(-x*1.8) + vec3(0.45,0.75,1.6)*halo) * fall;
    }
    gl_FragColor = vec4(color * vStyle.y * vCover, 1.0);
  }
`;

const flashVertex = /* glsl */`
  attribute vec3 aCenter;
  attribute vec3 aAxis;
  attribute vec4 aStyle; // radius, intensity, kind, seed
  uniform float uPixelWorld;
  varying vec2 vUv;
  varying vec4 vStyle;
  varying float vCover;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 p = (viewMatrix * vec4(aCenter,1.0)).xyz;
    float pixel = uPixelWorld * max(0.01,-p.z);
    bool puff = aStyle.z > 4.5;
    float r = max(aStyle.x,pixel*(puff ? 1.0 : 2.0));
    // RCS puffs (kind 5) keep their energy when held at the pixel floor, so
    // they fade out with range instead of growing into dots.
    vCover = puff ? min(1.0, aStyle.x*aStyle.x/(r*r)) : 1.0;
    if (aStyle.z < 0.5) {
      vec3 n = normalize(aAxis);
      vec3 right = normalize(cross(n, abs(n.y) < 0.9 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0)));
      vec3 up = cross(n,right);
      p += mat3(viewMatrix) * (right*position.x + up*position.y)*2.0*r;
    } else {
      p.xy += position.xy*2.0*r;
    }
    gl_Position = projectionMatrix * vec4(p,1.0);
    vUv = uv;
    vStyle = aStyle;
    #include <logdepthbuf_vertex>
  }
`;

const flashFragment = /* glsl */`
  varying vec2 vUv;
  varying vec4 vStyle;
  varying float vCover;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  void main() {
    #include <logdepthbuf_fragment>
    vec2 p = vUv*2.0-1.0;
    float r = length(p);
    // White-hot kernel, warm bloom skirt; the skirt carries most of the
    // visible size so a flash reads at a glance, not only through bloom.
    float core = exp(-r*r*22.0);
    float halo = exp(-r*r*5.0)*(1.0-smoothstep(0.6,1.0,r));
    vec3 col = vec3(3.0,2.7,2.2)*core + vec3(1.3,0.5,0.16)*halo*0.55;
    if (vStyle.z < 0.5) {
      float angle = atan(p.y,p.x);
      float arc = pow(0.5+0.5*sin(angle*7.0+vStyle.w*73.0),3.0);
      float ring = exp(-pow((r-0.67)*24.0,2.0))*(0.3+0.7*arc);
      col = vec3(2.6,3.2,3.6)*exp(-r*r*45.0) + vec3(0.16,1.15,1.9)*ring + vec3(0.06,0.3,0.6)*halo*0.5;
    } else if (vStyle.z > 4.5) {
      // RCS nozzle glow / exhaust puff: white kernel in a cold blue skirt
      col = vec3(1.9,2.1,2.5)*exp(-r*r*10.0)
        + vec3(0.3,0.55,1.1)*exp(-r*r*3.2)*(1.0-smoothstep(0.55,1.0,r))*0.75;
    } else if (vStyle.z > 1.5) {
      float rays = pow(max(0.0,1.0-abs(p.x*p.y)*60.0),3.0)*exp(-r*5.0);
      col += vec3(1.6,0.95,0.4)*rays;
    }
    gl_FragColor = vec4(col*vStyle.y*vCover,1.0);
  }
`;

// Torpedo marker at fleet range: a filled faction arrowhead pointing along
// the round's screen-space heading, with a dark outline so it reads against
// bright nebula and explosions - unlike the Tracks checkpoint diamond.
const markerVertex = /* glsl */`
  attribute vec3 aCenter;
  attribute vec3 aAxis;
  attribute vec4 aStyle; // half-size px, intensity, side (0 alpha / 1 beta), unused
  uniform float uPixelWorld;
  varying vec2 vUv;
  varying vec4 vStyle;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 p = (viewMatrix * vec4(aCenter, 1.0)).xyz;
    float pixel = uPixelWorld * max(0.01, -p.z);
    vec3 ax = mat3(viewMatrix) * aAxis;
    vec2 d = length(ax.xy) > 1e-6 ? normalize(ax.xy) : vec2(0.0, 1.0);
    vec2 q = position.xy * 2.0 * aStyle.x * pixel;
    p.xy += q.x * vec2(d.y, -d.x) + q.y * d;
    gl_Position = projectionMatrix * vec4(p, 1.0);
    vUv = uv;
    vStyle = aStyle;
    #include <logdepthbuf_vertex>
  }
`;

const markerFragment = /* glsl */`
  varying vec2 vUv;
  varying vec4 vStyle;
  #include <common>
  #include <logdepthbuf_pars_fragment>
  float cross2(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }
  float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
  }
  bool inTri(vec2 p, vec2 a, vec2 b, vec2 c) {
    float s1 = cross2(b - a, p - a), s2 = cross2(c - b, p - b), s3 = cross2(a - c, p - c);
    return (s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0) || (s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0);
  }
  void main() {
    #include <logdepthbuf_fragment>
    vec2 p = vUv * 2.0 - 1.0;
    // dart: tip, right wing, notch, left wing (+y = heading)
    vec2 T = vec2(0.0, 0.92), R = vec2(0.62, -0.62), N = vec2(0.0, -0.22), L = vec2(-0.62, -0.62);
    float d = min(min(segDist(p, T, R), segDist(p, R, N)), min(segDist(p, N, L), segDist(p, L, T)));
    bool inside = inTri(p, T, R, N) || inTri(p, T, N, L);
    float px = 1.0 / max(1.0, vStyle.x);           // one pixel in quad units
    float fill = inside ? 1.0 : 1.0 - smoothstep(0.0, px, d);
    float outline = 1.0 - smoothstep(px * 1.0, px * 2.2, d);
    vec3 tint = vStyle.z < 0.5 ? vec3(0.16,0.82,1.0) : vec3(1.0,0.54,0.24);
    float a = max(fill, outline * 0.75) * vStyle.y;
    gl_FragColor = vec4(tint * 1.25 * fill * vStyle.y, a);
  }
`;

const premultiplied = {
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  blending: THREE.CustomBlending, blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor, blendEquation: THREE.AddEquation
};

/** Reusable instance buffers. Resizes only when a new peak exceeds capacity. */
class QuadBatch {
  constructor(scene, fields, vertexShader, fragmentShader, capacity, blending = additive) {
    this.fields = fields;
    this.capacity = capacity;
    this.count = 0;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uPixelWorld: Ribbon.pixelWorld },
      vertexShader, fragmentShader, ...blending
    });
    this.geometry = this.allocate(capacity);
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 24;
    scene.add(this.mesh);
  }

  allocate(capacity) {
    const plane = new THREE.PlaneGeometry(1, 1);
    const g = new THREE.InstancedBufferGeometry();
    g.index = plane.index;
    g.setAttribute('position', plane.attributes.position);
    g.setAttribute('uv', plane.attributes.uv);
    for (const name in this.fields) {
      g.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(capacity * this.fields[name]), this.fields[name]).setUsage(THREE.DynamicDrawUsage));
    }
    g.instanceCount = 0;
    return g;
  }

  reserve() {
    if (this.count < this.capacity) return;
    this.capacity *= 2;
    const next = this.allocate(this.capacity);
    for (const name in this.fields) next.attributes[name].array.set(this.geometry.attributes[name].array);
    this.geometry.dispose();
    this.geometry = next;
    this.mesh.geometry = next;
  }

  begin(time) { this.count = 0; this.material.uniforms.uTime.value = time; }
  end() {
    this.geometry.instanceCount = this.count;
    this.mesh.visible = this.count > 0;
    for (const name in this.fields) {
      const a = this.geometry.attributes[name];
      a.needsUpdate = true;
    }
  }
  dispose() { this.mesh.removeFromParent(); this.geometry.dispose(); this.material.dispose(); }
}

export class WeaponRibbons extends QuadBatch {
  constructor(scene) {
    super(scene, { aFrom: 3, aTo: 3, aStyle: 4, aAge: 2 }, ribbonVertex, ribbonFragment, 2048);
  }
  add(from, to, width, intensity, kind, seed = 0, age0 = 0, age1 = 0) {
    if (intensity <= 0 || from.distanceToSquared(to) < 1e-12) return;
    this.reserve();
    const i = this.count++, a = this.geometry.attributes;
    a.aFrom.setXYZ(i, from.x, from.y, from.z);
    a.aTo.setXYZ(i, to.x, to.y, to.z);
    a.aStyle.setXYZW(i, width, intensity, kind, seed);
    a.aAge.setXY(i, age0, age1);
  }
}

export class WeaponFlashes extends QuadBatch {
  constructor(scene) {
    super(scene, { aCenter: 3, aAxis: 3, aStyle: 4 }, flashVertex, flashFragment, 128);
    this.mesh.renderOrder = 25;
  }
  add(center, axis, radius, intensity, kind = 1, seed = 0) {
    if (intensity <= 0) return;
    this.reserve();
    const i = this.count++, a = this.geometry.attributes;
    a.aCenter.setXYZ(i, center.x, center.y, center.z);
    a.aAxis.setXYZ(i, axis.x, axis.y, axis.z);
    a.aStyle.setXYZW(i, radius, intensity, kind, seed);
  }
}

/** Fleet-range torpedo markers (arrowheads), drawn over the additive effects. */
export class WeaponMarkers extends QuadBatch {
  constructor(scene) {
    super(scene, { aCenter: 3, aAxis: 3, aStyle: 4 }, markerVertex, markerFragment, 64, premultiplied);
    this.mesh.renderOrder = 26;
  }
  /** @param {string} faction 'alpha' | 'beta'; halfPx marker half-size in pixels */
  add(center, axis, intensity, faction, halfPx = 7) {
    if (intensity <= 0) return;
    this.reserve();
    const i = this.count++, a = this.geometry.attributes;
    a.aCenter.setXYZ(i, center.x, center.y, center.z);
    a.aAxis.setXYZ(i, axis.x, axis.y, axis.z);
    a.aStyle.setXYZW(i, halfPx, Math.min(1, intensity), faction === 'beta' ? 1 : 0, 0);
  }
}

/** Vertex-coloured, non-indexed part list merged into one geometry. */
function partList() {
  const parts = [];
  const part = (geometry, color, x = 0, y = 0, z = 0) => {
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    if (g !== geometry) geometry.dispose();
    g.translate(x, y, z);
    const c = new THREE.Color(color), colors = new Float32Array(g.attributes.position.count * 3);
    for (let i = 0; i < colors.length; i += 3) { colors[i] = c.r; colors[i + 1] = c.g; colors[i + 2] = c.b; }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    parts.push(g);
  };
  const merge = () => {
    const merged = mergeGeometries(parts);
    for (const p of parts) p.dispose();
    return merged;
  };
  return { part, merge };
}

const TORPEDO_R = 0.13;

/**
 * Space torpedo, body frame +Z nose (display units, ~1.95 long with the
 * bell): no fins - there is no air to bite. Ogive penetrator nose with a
 * seeker window, a clean propellant tank, an aft thrust frame carrying the
 * gimbal mount, and two rings of four RCS quads (tiny nubs with nozzle
 * stubs at RCS_LAYOUT) that trim the course and slew the body. The engine
 * bell is a separate instanced mesh so it can gimbal.
 */
function torpedoGeometry() {
  const { part, merge } = partList();
  const R = TORPEDO_R;
  const cyl = (rTop, rBottom, len, seg = 24) =>
    new THREE.CylinderGeometry(rTop, rBottom, len, seg, 1, true).rotateX(Math.PI / 2);
  // Propellant tank (z -0.48..0.32) and the denser penetrator section ahead of it
  part(cyl(R, R, 0.8), 0xa7afb8, 0, 0, -0.08);
  part(cyl(R, R, 0.28), 0x6a737d, 0, 0, 0.46);
  // Ogive nose (z 0.60..1.0) closing onto the seeker window
  const ogive = [];
  for (let i = 0; i <= 12; i++) {
    const u = i / 12;
    ogive.push(new THREE.Vector2(Math.max(0.03, R * Math.pow(1 - u * u, 0.62)), 0.6 + 0.4 * u));
  }
  part(new THREE.LatheGeometry(ogive, 24).rotateX(Math.PI / 2), 0x5e6772);
  part(new THREE.SphereGeometry(0.036, 12, 8).scale(1, 1, 0.75), 0x12303f, 0, 0, 0.995);
  // Aft thrust frame, bulkhead and gimbal mount (bell hangs off ENGINE.pivot)
  part(cyl(R, 0.116, 0.16), 0x49515b, 0, 0, -0.56);
  part(new THREE.CircleGeometry(0.116, 24).rotateX(Math.PI), 0x262b31, 0, 0, -0.64);
  part(new THREE.CylinderGeometry(0.048, 0.058, 0.035, 16).rotateX(Math.PI / 2), 0x3a3f46, 0, 0, ENGINE.pivot + 0.0175);
  // Panel seams and one cable raceway along +X (the quads sit clear of it)
  for (const z of [-0.48, 0.32, 0.6]) part(new THREE.TorusGeometry(R + 0.001, 0.005, 4, 32), 0x7d858d, 0, 0, z);
  part(new THREE.BoxGeometry(0.022, 0.018, 0.74), 0xc2a978, R + 0.008, 0, -0.08);
  // RCS quads: a block on the skin plus an outward nozzle stub
  const out = RCS_LAYOUT.radius;
  for (const z of RCS_LAYOUT.rings) {
    for (let k = 0; k < 4; k++) {
      const phi = RCS_LAYOUT.phase + k * Math.PI / 2, c = Math.cos(phi), s = Math.sin(phi);
      part(new THREE.BoxGeometry(0.04, 0.05, 0.075).rotateZ(phi), 0x646c75, c * (R + 0.008), s * (R + 0.008), z);
      part(new THREE.CylinderGeometry(0.014, 0.009, out - R - 0.026, 8).rotateZ(phi - Math.PI / 2), 0x1d2126,
        c * (out + R + 0.026) / 2, s * (out + R + 0.026) / 2, z);
    }
  }
  return merge();
}

/** Engine bell hanging from its gimbal pivot at the origin, exit toward -Z. */
function bellGeometry() {
  const { part, merge } = partList();
  const L = ENGINE.bell, throat = 0.034, exit = 0.104;
  const pts = [];
  for (let i = 0; i <= 12; i++) {
    const u = i / 12;  // exit (u=0) -> throat (u=1): parabolic bell contour
    pts.push(new THREE.Vector2(Math.sqrt(exit * exit + (throat * throat - exit * exit) * u), -L * (1 - u)));
  }
  part(new THREE.LatheGeometry(pts, 24).rotateX(Math.PI / 2), 0x5e4d3f);
  part(new THREE.TorusGeometry(exit, 0.005, 4, 32), 0x8a7663, 0, 0, -L);
  part(new THREE.SphereGeometry(0.046, 12, 8), 0x3b3f45, 0, 0, -0.02);
  return merge();
}

const NEG_Z = new THREE.Vector3(0, 0, -1);
const UNIT = new THREE.Vector3(1, 1, 1);

export class TorpedoBodies {
  constructor(scene, capacity = 64) {
    this.scene = scene;
    this.geometry = torpedoGeometry();
    this.bellGeometry = bellGeometry();
    this.material = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.72, roughness: 0.36 });
    this.bellMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true, metalness: 0.6, roughness: 0.45, side: THREE.DoubleSide
    });
    this.capacity = capacity;
    this.mesh = this.instanced(this.geometry, this.material, capacity);
    this.bell = this.instanced(this.bellGeometry, this.bellMaterial, capacity);
    this.pose = new THREE.Matrix4();
    this.local = new THREE.Matrix4();
    this.pivot = new THREE.Vector3(0, 0, ENGINE.pivot);
    this.tilt = new THREE.Quaternion();
    this.v = new THREE.Vector3();
  }

  instanced(geometry, material, capacity, prev = null) {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.count = 0;
    if (prev) {
      mesh.instanceMatrix.array.set(prev.instanceMatrix.array);
      mesh.count = prev.count;
      prev.removeFromParent(); prev.dispose();
    }
    this.scene.add(mesh);
    return mesh;
  }

  begin() { this.mesh.count = 0; this.bell.count = 0; }

  /**
   * One torpedo: body pose (position + orientation quaternion, +Z nose) and
   * the gimbal deflection as body-local x/y components of the bell tilt
   * (radians; the bell exit swings toward that side).
   */
  add(position, quaternion, tiltX = 0, tiltY = 0) {
    if (this.mesh.count === this.capacity) {
      this.capacity *= 2;
      this.mesh = this.instanced(this.geometry, this.material, this.capacity, this.mesh);
      this.bell = this.instanced(this.bellGeometry, this.bellMaterial, this.capacity, this.bell);
    }
    this.pose.compose(position, quaternion, UNIT);
    const i = this.mesh.count++;
    this.mesh.setMatrixAt(i, this.pose);
    const angle = Math.hypot(tiltX, tiltY);
    if (angle > 1e-6) {
      const k = Math.sin(angle) / angle;
      this.tilt.setFromUnitVectors(NEG_Z, this.v.set(tiltX * k, tiltY * k, -Math.cos(angle)));
    } else {
      this.tilt.identity();
    }
    this.local.compose(this.pivot, this.tilt, UNIT).premultiply(this.pose);
    this.bell.setMatrixAt(i, this.local);
    this.bell.count = this.mesh.count;
  }

  end() {
    for (const m of [this.mesh, this.bell]) { m.visible = m.count > 0; m.instanceMatrix.needsUpdate = true; }
  }

  dispose() {
    for (const m of [this.mesh, this.bell]) { m.removeFromParent(); m.dispose(); }
    this.geometry.dispose(); this.bellGeometry.dispose();
    this.material.dispose(); this.bellMaterial.dispose();
  }
}
