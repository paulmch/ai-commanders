import * as THREE from 'three';
import { getNoiseTexture, getStarTexture, getGlowTexture, getPlanetTexture, mulberry32 } from './Textures.js';

/**
 * Space environment: nebula skydome, HDR star field, the sun (directional
 * key light + disc + screen-space lens flare), a ringed gas giant for scale
 * and planet-shine fill, and a PMREM environment map so the PBR hulls have
 * something to reflect.
 *
 * Distances are in km (scene units). The battle lives within a few hundred
 * km of the origin; everything here sits tens of thousands of km out so
 * camera moves never produce parallax against it.
 */

const LOG_DEPTH_V = /* glsl */`
  #include <common>
  #include <logdepthbuf_pars_vertex>
`;
const LOG_DEPTH_V_MAIN = /* glsl */`
  #include <logdepthbuf_vertex>
`;
const LOG_DEPTH_F = /* glsl */`
  #include <logdepthbuf_pars_fragment>
`;
const LOG_DEPTH_F_MAIN = /* glsl */`
  #include <logdepthbuf_fragment>
`;

export class Environment {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.sunDir = new THREE.Vector3(0.62, 0.42, 0.28).normalize();
    this.planetDir = new THREE.Vector3(0.55, -0.3, -1).normalize();
    this.planetDistance = 46000;
    this.planetRadius = 8800;
    this.time = 0;

    this.group = new THREE.Group();
    this.group.name = 'environment';
    scene.add(this.group);

    this._buildLights();
    this._buildSkydome();
    this._buildStars();
    this._buildSun();
    this._buildPlanet();
    this._buildFlare();
    this._buildEnvMap();
  }

  // -------------------------------------------------------------------------
  _buildLights() {
    // Sun: hard white key. Space has no atmosphere - shadows are black and
    // the only fill is starlight, planet-shine and the ship's own glow.
    this.sunLight = new THREE.DirectionalLight(0xfff4e4, 3.2);
    this.sunLight.position.copy(this.sunDir).multiplyScalar(100000);
    this.group.add(this.sunLight);

    // Planet-shine: warm bounce from the gas giant's lit face
    this.planetLight = new THREE.DirectionalLight(0xd9b48c, 0.55);
    this.planetLight.position.copy(this.planetDir).multiplyScalar(100000);
    this.group.add(this.planetLight);

    // Faint cool ambient so unlit faces aren't pure black on screen
    this.ambient = new THREE.HemisphereLight(0x2a3450, 0x0d0a10, 0.35);
    this.group.add(this.ambient);
  }

  // -------------------------------------------------------------------------
  _buildSkydome() {
    const geom = new THREE.SphereGeometry(300000, 48, 32);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uNoise: { value: getNoiseTexture() },
        uPlanetDir: { value: this.planetDir },
        uSunDir: { value: this.sunDir }
      },
      vertexShader: /* glsl */`
        varying vec3 vDir;
        ${LOG_DEPTH_V}
        void main() {
          vDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uNoise;
        uniform vec3 uPlanetDir;
        uniform vec3 uSunDir;
        varying vec3 vDir;
        ${LOG_DEPTH_F}

        // Triplanar-blended tileable noise so there is no polar pinch
        float noise3(vec3 p, float ch) {
          vec3 w = abs(normalize(p));
          w = w * w; w /= (w.x + w.y + w.z);
          vec4 a = texture2D(uNoise, p.yz);
          vec4 b = texture2D(uNoise, p.xz);
          vec4 c = texture2D(uNoise, p.xy);
          vec4 n = a * w.x + b * w.y + c * w.z;
          return ch < 0.5 ? n.r : (ch < 1.5 ? n.g : (ch < 2.5 ? n.b : n.a));
        }

        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 d = normalize(vDir);

          // Galactic band: a tilted great circle with dust lanes
          vec3 bandN = normalize(vec3(0.35, 0.82, -0.45));
          float lat = dot(d, bandN);
          float band = exp(-lat * lat * 26.0);
          float n1 = noise3(d * 2.2, 0.0);
          float n2 = noise3(d * 5.1 + 3.1, 1.0);
          float n3 = noise3(d * 11.0 + 7.7, 2.0);
          float wisps = pow(n1 * 0.45 + n2 * 0.35 + n3 * 0.2, 2.6);
          float dust = smoothstep(0.35, 0.75, noise3(d * 3.7 + 11.0, 3.0));
          vec3 milky = vec3(0.62, 0.66, 0.78) * band * wisps * 1.7 * (1.0 - dust * 0.85);

          // Two faint coloured nebulae far from the band
          vec3 nebA = normalize(vec3(-0.7, 0.2, 0.6));
          vec3 nebB = normalize(vec3(0.5, -0.6, 0.4));
          float fa = pow(max(0.0, dot(d, nebA)), 6.0) * pow(noise3(d * 3.0 + 21.0, 1.0), 1.8);
          float fb = pow(max(0.0, dot(d, nebB)), 5.0) * pow(noise3(d * 4.0 + 33.0, 2.0), 2.0);
          vec3 nebula = vec3(0.35, 0.18, 0.55) * fa * 1.4 + vec3(0.12, 0.35, 0.5) * fb * 1.2;

          // Background: near-black with the faintest blue toward the band
          vec3 col = vec3(0.004, 0.005, 0.010) + milky * 0.09 + nebula * 0.10;
          gl_FragColor = vec4(col, 1.0);
        }`,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false
    });
    this.skydome = new THREE.Mesh(geom, mat);
    this.skydome.renderOrder = -100;
    this.skydome.frustumCulled = false;
    this.group.add(this.skydome);
  }

  // -------------------------------------------------------------------------
  _buildStars() {
    const rnd = mulberry32(2024);
    const count = 7000;
    const R = 260000;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const size = new Float32Array(count);
    const bandN = new THREE.Vector3(0.35, 0.82, -0.45).normalize();
    const c = new THREE.Color();
    let i = 0;
    while (i < count) {
      // Isotropic direction, with a bias toward the galactic band
      const theta = rnd() * Math.PI * 2;
      const phi = Math.acos(2 * rnd() - 1);
      const d = new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi));
      const lat = d.dot(bandN);
      if (rnd() > 0.45 + 0.55 * Math.exp(-lat * lat * 12)) continue;

      pos[i * 3] = d.x * R; pos[i * 3 + 1] = d.y * R; pos[i * 3 + 2] = d.z * R;

      // Magnitude distribution: many faint, few bright
      const m = Math.pow(rnd(), 3.2);
      const bright = 0.25 + m * 1.3;
      size[i] = 1.3 + m * 2.6;
      const t = rnd();
      if (t < 0.10) c.setHSL(0.07, 0.55, 0.78);       // orange K/M
      else if (t < 0.22) c.setHSL(0.58, 0.55, 0.85);  // blue B/A
      else if (t < 0.4) c.setHSL(0.12, 0.25, 0.9);    // yellow G
      else c.setHSL(0.6, 0.08, 0.95);                 // white
      col[i * 3] = c.r * bright; col[i * 3 + 1] = c.g * bright; col[i * 3 + 2] = c.b * bright;
      i++;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    geom.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uMap: { value: getStarTexture() }, uPixelRatio: { value: 1 } },
      vertexShader: /* glsl */`
        attribute vec3 aColor;
        attribute float aSize;
        uniform float uPixelRatio;
        varying vec3 vColor;
        ${LOG_DEPTH_V}
        void main() {
          vColor = aColor;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = aSize * uPixelRatio;
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        varying vec3 vColor;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          float a = texture2D(uMap, gl_PointCoord).a;
          gl_FragColor = vec4(vColor * a, a);
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
    this.stars = new THREE.Points(geom, mat);
    this.stars.renderOrder = -99;
    this.stars.frustumCulled = false;
    this.group.add(this.stars);
  }

  // -------------------------------------------------------------------------
  _buildSun() {
    const dist = 240000;
    const pos = this.sunDir.clone().multiplyScalar(dist);
    // Disc: HDR white so bloom gives it a corona
    const disc = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: new THREE.Color(6, 5.6, 5.0),
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
    }));
    disc.position.copy(pos);
    disc.scale.setScalar(9000);
    disc.renderOrder = -98;
    this.group.add(disc);
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: new THREE.Color(0.9, 0.7, 0.5),
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
    }));
    halo.position.copy(pos);
    halo.scale.setScalar(60000);
    halo.renderOrder = -98;
    this.group.add(halo);
    this.sunPos = pos;
  }

  // -------------------------------------------------------------------------
  _buildPlanet() {
    const R = this.planetRadius;
    const pos = this.planetDir.clone().multiplyScalar(this.planetDistance);
    const geom = new THREE.SphereGeometry(R, 96, 64);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: getPlanetTexture() },
        uSunDir: { value: this.sunDir },
        uAtmo: { value: new THREE.Color(0.55, 0.7, 1.0) }
      },
      vertexShader: /* glsl */`
        varying vec3 vNormal;
        varying vec3 vView;
        varying vec2 vUv;
        ${LOG_DEPTH_V}
        void main() {
          vUv = uv;
          vNormal = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vView = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        uniform vec3 uSunDir;
        uniform vec3 uAtmo;
        varying vec3 vNormal;
        varying vec3 vView;
        varying vec2 vUv;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 n = normalize(vNormal);
          vec3 albedo = texture2D(uMap, vUv).rgb;
          float ndl = dot(n, uSunDir);
          float day = smoothstep(-0.08, 0.25, ndl);
          float fres = pow(1.0 - max(0.0, dot(n, normalize(vView))), 3.0);
          // Atmosphere: rim scattering on the lit side, faint on the night side
          vec3 atmo = uAtmo * fres * (0.25 + 0.9 * smoothstep(-0.3, 0.4, ndl));
          vec3 lit = albedo * (0.02 + day * 1.25 * max(0.0, ndl) + 0.03);
          // limb darkening
          lit *= 0.75 + 0.25 * max(0.0, dot(n, normalize(vView)));
          gl_FragColor = vec4(lit + atmo * 0.6, 1.0);
        }`
    });
    this.planet = new THREE.Mesh(geom, mat);
    this.planet.position.copy(pos);
    this.planet.rotation.z = 0.28;
    this.group.add(this.planet);

    // Atmosphere shell: additive fresnel halo just outside the limb
    const atmoGeom = new THREE.SphereGeometry(R * 1.035, 96, 64);
    const atmoMat = new THREE.ShaderMaterial({
      uniforms: { uSunDir: { value: this.sunDir }, uAtmo: { value: new THREE.Color(0.5, 0.68, 1.0) } },
      vertexShader: /* glsl */`
        varying vec3 vNormal;
        varying vec3 vView;
        ${LOG_DEPTH_V}
        void main() {
          vNormal = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vView = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uSunDir;
        uniform vec3 uAtmo;
        varying vec3 vNormal;
        varying vec3 vView;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 n = normalize(vNormal);
          float vdn = max(0.0, dot(n, normalize(vView)));
          float rim = pow(1.0 - vdn, 5.5) * smoothstep(0.0, 0.35, vdn + 0.35);
          float lit = 0.15 + 0.85 * smoothstep(-0.35, 0.3, dot(n, uSunDir));
          gl_FragColor = vec4(uAtmo * rim * lit * 1.6, 1.0);
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.FrontSide
    });
    const atmo = new THREE.Mesh(atmoGeom, atmoMat);
    atmo.position.copy(pos);
    this.group.add(atmo);

    // Rings: procedural radial bands, lit by the sun, shadowed by the planet
    const ringTex = this._makeRingTexture();
    const ringGeom = new THREE.RingGeometry(R * 1.35, R * 2.3, 180, 1);
    // Remap UV so u runs radially
    const uv = ringGeom.attributes.uv;
    const p = ringGeom.attributes.position;
    for (let i = 0; i < uv.count; i++) {
      const r = Math.hypot(p.getX(i), p.getY(i));
      uv.setXY(i, (r - R * 1.35) / (R * 0.95), 0.5);
    }
    const ringMat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: ringTex },
        uSunDir: { value: this.sunDir },
        uPlanetPos: { value: pos },
        uPlanetR: { value: R }
      },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        varying vec3 vWorld;
        ${LOG_DEPTH_V}
        void main() {
          vUv = uv;
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vWorld = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        uniform vec3 uSunDir;
        uniform vec3 uPlanetPos;
        uniform float uPlanetR;
        varying vec2 vUv;
        varying vec3 vWorld;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec4 t = texture2D(uMap, vUv);
          // Planet shadow: does the ray toward the sun pass through the planet?
          vec3 rel = vWorld - uPlanetPos;
          float along = dot(rel, uSunDir);
          vec3 perp = rel - uSunDir * along;
          float shadow = along < 0.0 ? smoothstep(uPlanetR * 0.97, uPlanetR * 1.03, length(perp)) : 1.0;
          vec3 col = t.rgb * (0.05 + 1.1 * shadow);
          gl_FragColor = vec4(col, t.a * 0.9);
        }`,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const rings = new THREE.Mesh(ringGeom, ringMat);
    rings.position.copy(pos);
    rings.rotation.set(-Math.PI / 2 + 0.42, 0.1, 0.28);
    this.group.add(rings);
    this.rings = rings;
  }

  _makeRingTexture() {
    const w = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = 4;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, 4);
    const rnd = mulberry32(77);
    // Random band structure: overlapping gaussian gaps and bright ringlets
    const bands = [];
    for (let i = 0; i < 40; i++) bands.push({ c: rnd(), w: 0.004 + rnd() * 0.03, a: (rnd() - 0.4) * 0.9 });
    for (let x = 0; x < w; x++) {
      const u = x / w;
      let a = 0.55 * Math.sin(u * Math.PI);
      for (const b of bands) a += b.a * Math.exp(-Math.pow((u - b.c) / b.w, 2));
      a *= smoothstep01(0, 0.03, u) * smoothstep01(1, 0.9, u);
      a = Math.max(0, Math.min(1, a));
      const tone = 0.75 + 0.25 * Math.sin(u * 40 + 1.0) * 0.5;
      for (let y = 0; y < 4; y++) {
        const i = (y * w + x) * 4;
        img.data[i] = 215 * tone; img.data[i + 1] = 200 * tone; img.data[i + 2] = 176 * tone;
        img.data[i + 3] = a * 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }

  // -------------------------------------------------------------------------
  /**
   * Screen-space lens flare without framebuffer readback: ghost sprites are
   * placed in the world along the ray through their screen position, sized
   * for constant apparent size, and faded as the sun leaves the frame.
   */
  _buildFlare() {
    const glow = getGlowTexture();
    const ringTex = this._makeHexGhost();
    this.flareGhosts = [];
    const spec = [
      { t: 0.0, s: 0.12, c: [1.0, 0.9, 0.7], a: 0.35, tex: glow },
      { t: 0.25, s: 0.035, c: [0.6, 0.8, 1.0], a: 0.3, tex: ringTex },
      { t: 0.45, s: 0.06, c: [1.0, 0.6, 0.4], a: 0.2, tex: ringTex },
      { t: 0.7, s: 0.025, c: [0.5, 1.0, 0.7], a: 0.35, tex: ringTex },
      { t: 1.0, s: 0.08, c: [0.9, 0.7, 1.0], a: 0.16, tex: ringTex },
      { t: 1.35, s: 0.05, c: [1.0, 0.85, 0.6], a: 0.25, tex: glow },
      { t: 1.7, s: 0.11, c: [0.6, 0.7, 1.0], a: 0.1, tex: ringTex }
    ];
    for (const g of spec) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: g.tex, color: new THREE.Color(...g.c), transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
      }));
      s.renderOrder = 1000;
      s.frustumCulled = false;
      this.group.add(s);
      this.flareGhosts.push({ sprite: s, ...g });
    }
    // Anamorphic streak through the sun
    const streak = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this._makeStreakGhost(), color: new THREE.Color(0.55, 0.7, 1.0), transparent: true,
      opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
    }));
    streak.renderOrder = 1000;
    streak.frustumCulled = false;
    this.group.add(streak);
    this.flareStreak = streak;
  }

  _makeHexGhost() {
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.translate(size / 2, size / 2);
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      ctx.lineTo(Math.cos(a) * 58, Math.sin(a) * 58);
    }
    ctx.closePath();
    const g = ctx.createRadialGradient(0, 0, 20, 0, 0, 58);
    g.addColorStop(0, 'rgba(255,255,255,0.08)');
    g.addColorStop(0.8, 'rgba(255,255,255,0.18)');
    g.addColorStop(1, 'rgba(255,255,255,0.5)');
    ctx.fillStyle = g;
    ctx.fill();
    const t = new THREE.CanvasTexture(canvas);
    return t;
  }

  _makeStreakGhost() {
    const w = 512, h = 32;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(0.5, 'rgba(255,255,255,1)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    const v = ctx.createLinearGradient(0, 0, 0, h);
    v.addColorStop(0, 'rgba(0,0,0,1)');
    v.addColorStop(0.5, 'rgba(0,0,0,0)');
    v.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = v;
    ctx.fillRect(0, 0, w, h);
    return new THREE.CanvasTexture(canvas);
  }

  // -------------------------------------------------------------------------
  /**
   * PMREM environment from a reduced copy of the sky: sun disc, planet,
   * dome. Gives the metallic hull plating real reflections.
   */
  _buildEnvMap() {
    const envScene = new THREE.Scene();
    envScene.add(this.skydome.clone());
    const sun = new THREE.Mesh(new THREE.SphereGeometry(9000, 16, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(40, 36, 30) }));
    sun.position.copy(this.sunPos);
    envScene.add(sun);
    const sunHalo = new THREE.Mesh(new THREE.SphereGeometry(30000, 16, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.6, 0.45, 0.3), transparent: true, opacity: 0.5 }));
    sunHalo.position.copy(this.sunPos);
    envScene.add(sunHalo);
    const planet = new THREE.Mesh(new THREE.SphereGeometry(this.planetRadius, 32, 24),
      new THREE.MeshLambertMaterial({ map: getPlanetTexture() }));
    planet.position.copy(this.planet.position);
    envScene.add(planet);
    const l = new THREE.DirectionalLight(0xffffff, 2.5);
    l.position.copy(this.sunDir).multiplyScalar(1000);
    envScene.add(l);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    const rt = pmrem.fromScene(envScene, 0.02, 10, 900000);
    this.scene.environment = rt.texture;
    pmrem.dispose();
  }

  // -------------------------------------------------------------------------
  update(camera, delta) {
    this.time += delta;
    // Keep the far backdrop centred on the camera so it never gets parallax
    this.skydome.position.copy(camera.position);
    this.stars.position.copy(camera.position);

    // Lens flare placement
    const sunNdc = this.sunPos.clone().project(camera);
    const viewZ = this.sunPos.clone().applyMatrix4(camera.matrixWorldInverse).z;
    const inFront = viewZ < 0;
    const sx = sunNdc.x, sy = sunNdc.y;
    const edge = Math.max(Math.abs(sx), Math.abs(sy));
    const vis = inFront ? THREE.MathUtils.smoothstep(1.25 - edge, 0, 0.45) : 0;
    // Aspect-corrected constant screen size: sprite world size at depth d
    const fovScale = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const depth = 50; // world units in front of the camera
    for (const g of this.flareGhosts) {
      const nx = sx * (1 - g.t) + 0 * g.t;   // toward screen centre and beyond
      const ny = sy * (1 - g.t);
      const wp = new THREE.Vector3(nx, ny, 0.5).unproject(camera);
      const dir = wp.sub(camera.position).normalize();
      g.sprite.position.copy(camera.position).addScaledVector(dir, depth);
      const worldSize = g.s * fovScale * depth;
      g.sprite.scale.set(worldSize, worldSize, 1);
      g.sprite.material.opacity = g.a * vis;
    }
    const swp = new THREE.Vector3(sx, sy, 0.5).unproject(camera);
    const sdir = swp.sub(camera.position).normalize();
    this.flareStreak.position.copy(camera.position).addScaledVector(sdir, depth);
    this.flareStreak.scale.set(1.9 * fovScale * depth, 0.06 * fovScale * depth, 1);
    this.flareStreak.material.opacity = 0.3 * vis;
  }
}

function smoothstep01(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
