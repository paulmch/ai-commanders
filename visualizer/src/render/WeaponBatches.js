import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Ribbon } from './Trails.js';

const additive = {
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  blending: THREE.CustomBlending, blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation
};

// One draw for all streaks, curved trail segments, motor plumes and PD ribbons.
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
    // Half-width floor >= 1 px: a narrower strip rasterises as beads.
    float floorPx = kind < 0.5 ? 1.1 : kind < 1.5 ? 1.0 : kind < 2.5 ? 0.6 : plume ? 1.2 : 1.0;
    float capPx = plume ? 64.0 : kind > 5.5 ? 24.0 : kind > 1.5 && kind < 2.5 ? 2.2 : 3.0;
    float width = max(pixel * floorPx, min(aStyle.x, pixel * capPx));
    // Thin effects retain a controlled subpixel contribution at fleet scale.
    float minCover = kind < 0.5 ? 0.5 : kind < 1.5 ? 0.07 : kind < 2.5 ? 0.35 : kind > 5.5 ? 0.12 : 0.3;
    vCover = max(minCover, min(1.0, aStyle.x / width));
    // Half-width in pixels: the fragment keeps the bright core >= ~1.5 px
    // wide so a thin line never alternates hit/miss (beading) along its run.
    vPx = width / pixel;
    // Plumes flare out of the nozzle, then neck down slightly toward the tip.
    float taper = plume ? mix(0.3, 1.0, smoothstep(0.0, 0.3, uv.x)) * (1.0 - 0.35 * uv.x) : 1.0;
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
    if (kind < 0.5) {
      vec3 edge = mix(vec3(1.6,0.36,0.055), vec3(0.06,0.6,0.85), smoothstep(0.2,1.0,x));
      color = (vec3(3.4,3.05,2.65)*core + edge*halo*0.75) * pow(1.0-x,0.45) * ends;
    } else if (kind < 1.5) {
      // Slug wake: ionised track cooling from straw-white to faint cyan
      float cooling = pow(max(0.0,1.0-vAge),1.6);
      float wisps = 0.75+0.25*sin(x*13.0 + vStyle.w*51.0 - uTime*2.0);
      color = mix(vec3(0.04,0.26,0.4),vec3(1.0,0.72,0.4),cooling*cooling)
        * (core*0.6 + halo*0.5) * cooling * wisps;
    } else if (kind < 2.5) {
      float packets = 0.82+0.18*sin(x*55.0-uTime*58.0+vStyle.w*17.0);
      // Several turrets share a line of fire: keep pixel-floor beams slim.
      float slim = mix(0.6, 1.0, smoothstep(0.7, 2.0, vPx));
      // aAge fades a close-up's split beam ends out toward the cut (0 = full)
      color = (vec3(1.1,0.77,0.5)*core + vec3(0.5,0.13,0.035)*halo*0.3)*packets*slim*(1.0-vAge);
    } else if (kind < 3.5) {
      // Motor plume: broad white-hot core out of the nozzle with shock
      // cells, wrapped in a longer orange sheath that fades to the tip.
      float cells = pow(0.5+0.5*cos(x*47.0),10.0)*exp(-x*4.0);
      float ripple = 0.87+0.13*sin(x*67.0-uTime*19.0+vStyle.w*21.0);
      float hot = exp(-y*y*min(7.0, aa)) * exp(-x*3.8) * (1.0+cells);
      float sheath = exp(-y*y*2.2) * (1.0-y) * exp(-x*1.7);
      color = (vec3(2.6,2.2,1.7)*hot + vec3(1.35,0.45,0.12)*sheath)
        * smoothstep(0.0,0.03,x) * (1.0-smoothstep(0.55,1.0,x)) * ripple;
    } else if (kind < 4.5) {
      color = mix(vec3(2.5,2.15,1.65),vec3(1.0,0.16,0.015),vAge)*core*ends;
    } else if (kind < 5.5) {
      float dash = smoothstep(0.25,0.4,fract(x*32.0));
      color = vec3(0.08,0.55,0.5)*halo*dash;
    } else {
      // Torpedo exhaust wake: hot orange near the motor, dull red-brown aft
      float cooling = pow(max(0.0,1.0-vAge),1.4);
      float wisps = 0.7+0.3*sin(x*9.0 + vStyle.w*37.0 - uTime*1.3);
      color = mix(vec3(0.22,0.07,0.03),vec3(1.5,0.72,0.3),cooling*cooling)
        * (core*0.7 + halo*0.45) * cooling * wisps;
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
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vec3 p = (viewMatrix * vec4(aCenter,1.0)).xyz;
    float pixel = uPixelWorld * max(0.01,-p.z);
    float r = max(aStyle.x,pixel*2.0);
    // Faction markers (kinds 3/4) hold a constant ~12 px screen size.
    if (aStyle.z > 2.5) r = pixel*6.0;
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
    } else if (vStyle.z > 2.5) {
      // Faction diamond so a torpedo stays identifiable when sub-pixel
      float d = abs(p.x) + abs(p.y);
      float edge = exp(-pow((d-0.62)*10.0,2.0));
      col = (vStyle.z < 3.5 ? vec3(0.16,0.82,1.0) : vec3(1.0,0.54,0.24)) * edge;
    } else if (vStyle.z > 1.5) {
      float rays = pow(max(0.0,1.0-abs(p.x*p.y)*60.0),3.0)*exp(-r*5.0);
      col += vec3(1.6,0.95,0.4)*rays;
    }
    gl_FragColor = vec4(col*vStyle.y,1.0);
  }
`;

/** Reusable instance buffers. Resizes only when a new peak exceeds capacity. */
class QuadBatch {
  constructor(scene, fields, vertexShader, fragmentShader, capacity) {
    this.fields = fields;
    this.capacity = capacity;
    this.count = 0;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uPixelWorld: Ribbon.pixelWorld },
      vertexShader, fragmentShader, ...additive
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

function torpedoGeometry() {
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
  part(new THREE.CylinderGeometry(0.12, 0.15, 1.15, 12).rotateX(Math.PI / 2), 0xa4acb6);
  part(new THREE.ConeGeometry(0.12, 0.42, 12).rotateX(Math.PI / 2), 0x626f80, 0, 0, 0.78);
  part(new THREE.CylinderGeometry(0.15, 0.1, 0.2, 12, 1, true).rotateX(Math.PI / 2), 0x303845, 0, 0, -0.67);
  for (const z of [-0.4, 0.34]) part(new THREE.TorusGeometry(0.145, 0.016, 4, 16), 0xced8df, 0, 0, z);
  for (let i = 0; i < 4; i++) {
    const a = i * Math.PI / 2;
    part(new THREE.BoxGeometry(0.025, 0.27, 0.3).rotateZ(a), 0x455362, -Math.sin(a)*0.19, Math.cos(a)*0.19, -0.45);
  }
  // Seeker window and longitudinal cable conduits.
  part(new THREE.SphereGeometry(0.06, 8, 6).scale(1, 1, 0.5), 0x19364a, 0, 0, 0.99);
  part(new THREE.BoxGeometry(0.028, 0.028, 0.85), 0xd0b787, 0.13, 0, 0);
  const merged = mergeGeometries(parts);
  for (const p of parts) p.dispose();
  return merged;
}

export class TorpedoBodies {
  constructor(scene, capacity = 64) {
    this.scene = scene;
    this.geometry = torpedoGeometry();
    this.material = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.72, roughness: 0.36 });
    this.capacity = capacity;
    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.pose = new THREE.Object3D();
    this.axis = new THREE.Vector3(0, 0, 1);
    scene.add(this.mesh);
  }
  begin() { this.mesh.count = 0; }
  add(position, direction, roll) {
    if (this.mesh.count === this.capacity) {
      const prev = this.mesh;
      this.capacity *= 2;
      this.mesh = new THREE.InstancedMesh(this.geometry, this.material, this.capacity);
      this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.mesh.instanceMatrix.array.set(prev.instanceMatrix.array);
      this.mesh.count = prev.count;
      this.mesh.frustumCulled = false;
      prev.removeFromParent(); prev.dispose();
      this.scene.add(this.mesh);
    }
    this.pose.position.copy(position);
    this.pose.quaternion.setFromUnitVectors(this.axis, direction);
    this.pose.rotateZ(roll);
    this.pose.updateMatrix();
    this.mesh.setMatrixAt(this.mesh.count++, this.pose.matrix);
  }
  end() { this.mesh.visible = this.mesh.count > 0; this.mesh.instanceMatrix.needsUpdate = true; }
  dispose() { this.mesh.removeFromParent(); this.mesh.dispose(); this.geometry.dispose(); this.material.dispose(); }
}
