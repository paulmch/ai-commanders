import * as THREE from 'three';
import { GLSL_NOISE, makeStarCatalogue, getGasGiantTexture, makeRingTexture, makeNoiseVolumeTexture } from './SkyTextures.js';
import { getGraphicsSettings } from './Quality.js';
import { DustMotes } from './DustMotes.js';

/**
 * Space environment:
 *  - Milky Way, dust lanes, emission nebulae and ~100k unresolved stars,
 *    rendered ONCE by a heavy procedural shader into a cube map (the old
 *    per-pixel skydome cost ~120 ms/frame in SwiftShader; the baked cube is
 *    a single lookup),
 *  - a black-body-coloured HDR star catalogue with diffraction spikes on
 *    the brightest stars,
 *  - the sun: limb-darkened disc + corona billboard on the key light's
 *    direction (lens flare / god rays live in PostFX, fed via
 *    scene.userData.sun),
 *  - a ringed gas giant with ring shadows and a scattering limb, placed to
 *    frame the tactical fleet view without sitting behind the battle,
 *  - lights: sun key, planet-shine, cool hemisphere, camera-relative rim,
 *  - a PMREM environment from the same sky (boosted, with a cool fill
 *    lobe opposite the sun) so the metallic hulls reflect something,
 *  - near-field dust motes (high quality) that streak with camera motion.
 *
 * Everything far is re-centred on the camera every frame: no parallax at
 * any zoom. Distances are km (scene units).
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

// CameraController.focusFleet looks along -(0, 0.6, 1): the backdrop is
// composed around that view.
const FLEET_FWD = new THREE.Vector3(0, -0.6, -1).normalize();

/** World direction that appears at (ndcX, ndcY) in the default fleet view. */
export function fleetViewDir(ndcX, ndcY, fov = 55, aspect = 16 / 9) {
  const right = new THREE.Vector3().crossVectors(FLEET_FWD, new THREE.Vector3(0, 1, 0)).normalize();
  const up = new THREE.Vector3().crossVectors(right, FLEET_FWD).normalize();
  const tv = Math.tan(THREE.MathUtils.degToRad(fov) / 2);
  return FLEET_FWD.clone()
    .addScaledVector(right, ndcX * tv * aspect)
    .addScaledVector(up, ndcY * tv)
    .normalize();
}

const SUN_DIST = 240000;
const SUN_ANGULAR_RADIUS = 0.0085;          // rad (~0.5 deg; dramatic, not huge)
const PLANET_DIST = 90000;
const PLANET_ANGULAR_RADIUS = 0.068;        // rad (~3.9 deg)
const STAR_MAX = 9000;
// Key light. With the shadow map the key:fill ratio carries the form: a hard
// sun ~4x the combined fill (hemisphere + rim + planet-shine + IBL).
export const SUN_INTENSITY = 4.4;

export class Environment {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.gfx = getGraphicsSettings(renderer);
    this.sunDir = new THREE.Vector3(0.62, 0.42, 0.28).normalize();
    this.sunColor = new THREE.Color(1.0, 0.93, 0.82);
    this.time = 0;

    // Galactic frame: band tilted ~16 deg off the fleet-view centre so the
    // Milky Way crosses the frame diagonally instead of sitting behind the
    // battle; the bulge lies ~40 deg along the band from there.
    const bandN = new THREE.Vector3(0.35, 0.82, -0.45).normalize().addScaledVector(FLEET_FWD, 0.33).normalize();
    const inPlane = FLEET_FWD.clone().addScaledVector(bandN, -FLEET_FWD.dot(bandN)).normalize();
    const e = new THREE.Vector3().crossVectors(bandN, inPlane);
    this.galN = bandN;
    this.galC = inPlane.clone().multiplyScalar(Math.cos(0.7)).addScaledVector(e, Math.sin(0.7)).normalize();
    this.nebulaDir = fleetViewDir(0.62, -0.5);
    this.nebulaDirB = fleetViewDir(0.1, 0.92);
    this.planetDir = fleetViewDir(0.74, 0.54);   // upper right: clear of the HUD panels
    this.planetRadius = PLANET_DIST * Math.sin(PLANET_ANGULAR_RADIUS);

    this.group = new THREE.Group();
    this.group.name = 'environment';
    scene.add(this.group);

    this._buildLights();
    this._bakeSky(this.gfx.preset.skySize);
    this._buildSkyMesh();
    this._buildStars();
    this._buildSun();
    this._buildPlanet();
    this._buildEnvMap();
    this.dust = new DustMotes(this.group);
    this.dust.setEnabled(this.gfx.preset.dust);

    // PostFX reads this for the lens flare / god rays
    scene.userData.sun = { dir: this.sunDir, angularRadius: SUN_ANGULAR_RADIUS, color: this.sunColor };

    this._offGfx = this.gfx.onChange(() => this.applyQuality());
    this.gfx.installControls();
  }

  /** Release the baked sky, env map and every backdrop mesh (reload). */
  dispose() {
    this._offGfx?.();
    this.scene.remove(this.group);
    if (this.scene.environment === this.envRT?.texture) this.scene.environment = null;
    delete this.scene.userData.sun;
    this.skyRT?.dispose();
    this.envRT?.dispose();
    this.noiseVol?.dispose();
    disposeTree(this.group);
  }

  applyQuality() {
    const p = this.gfx.preset;
    if (p.skySize !== this.skySize) {
      this._bakeSky(p.skySize);
      this.skyMesh.material.uniforms.uSky.value = this.skyRT.texture;
    }
    this.stars.geometry.setDrawRange(0, Math.min(p.starCount, STAR_MAX));
    this.dust.setEnabled(p.dust);
    this._setupShadow();
  }

  // -------------------------------------------------------------------------
  _buildLights() {
    // Sun: hard white key. Space has no atmosphere - shadows are black and
    // the only fill is starlight, planet-shine and the ship's own glow.
    this.sunLight = new THREE.DirectionalLight(0xfff1dc, SUN_INTENSITY);
    this.sunLight.position.copy(this.sunDir).multiplyScalar(100000);
    this.group.add(this.sunLight);
    this.group.add(this.sunLight.target);
    this._setupShadow();

    // Planet-shine: warm bounce from the gas giant's lit face
    this.planetLight = new THREE.DirectionalLight(0xe6a877, 0.55);
    this.planetLight.position.copy(this.planetDir).multiplyScalar(100000);
    this.group.add(this.planetLight);

    // Faint cool ambient so unlit faces aren't pure black on screen
    this.ambient = new THREE.HemisphereLight(0x2c3b60, 0x0c0a10, 0.26);
    this.group.add(this.ambient);

    // Rim: a cool light from beyond the subject, re-aimed every frame so
    // silhouettes separate from black space from any camera angle
    this.rimLight = new THREE.DirectionalLight(0x8fb2ff, 1.25);
    this.group.add(this.rimLight);
    this.group.add(this.rimLight.target);
  }

  // -------------------------------------------------------------------------
  // Sun shadow map (high tier). The scene is hundreds of km across and a
  // hull is a few units long, so one map cannot cover the battle: the
  // orthographic shadow box is re-fitted every frame to what the camera is
  // looking at - a square around the orbit focus sized to the view at focus
  // depth - which keeps about one shadow texel per screen pixel at any
  // zoom. Its size moves in quarter-octave steps and its centre snaps to
  // whole texels, so shadow edges do not crawl while orbiting or panning.
  // three renders BACK faces into the map for the (closed, front-sided)
  // hull parts: lit faces are never in the map, so there is no acne to
  // bias away and no peter-panning at contacts.
  _shadowSize() {
    const p = this.gfx.preset;
    if (!p.shadows) return 0;
    const cap = this.renderer.capabilities?.maxTextureSize || 2048;
    return Math.min(this.gfx.software ? Math.min(p.shadowMapSize, 2048) : p.shadowMapSize, cap);
  }

  _setupShadow() {
    const size = this._shadowSize();
    const light = this.sunLight;
    const sh = light.shadow;
    if (size && sh.mapSize.x !== size) {
      sh.mapSize.set(size, size);
      sh.map?.dispose();
      sh.map = null;
    }
    light.castShadow = size > 0;
    // light-space basis exactly as LightShadow.updateMatrices builds it
    // (lookAt with world up): needed for the texel snap
    this._shZ = this.sunDir.clone();
    this._shX = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), this._shZ).normalize();
    this._shY = new THREE.Vector3().crossVectors(this._shZ, this._shX).normalize();
  }

  _fitShadow(camera, focus, viewDist) {
    const light = this.sunLight;
    if (!light.castShadow || !focus) return;
    const sh = light.shadow;
    const size = sh.mapSize.x;
    const fovT = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    let R = Math.max(viewDist, 1) * fovT * Math.max(camera.aspect, 1) * 1.08;
    R = THREE.MathUtils.clamp(R, 4, 4000);
    R = Math.pow(2, Math.ceil(Math.log2(R) * 4) / 4);
    const texel = (2 * R) / size;
    const x = Math.round(focus.dot(this._shX) / texel) * texel;
    const y = Math.round(focus.dot(this._shY) / texel) * texel;
    const z = focus.dot(this._shZ);
    // casters up-sun of the box still shadow into it: keep a hull-length
    // margin on the depth range
    const margin = 60;
    const D = R + margin;
    light.target.position.set(0, 0, 0)
      .addScaledVector(this._shX, x).addScaledVector(this._shY, y).addScaledVector(this._shZ, z);
    light.position.copy(light.target.position).addScaledVector(this._shZ, D);
    const cam = sh.camera;
    if (cam.right !== R || cam.far !== 2 * D) {
      cam.left = -R; cam.right = R; cam.top = R; cam.bottom = -R;
      cam.near = 0.5; cam.far = 2 * D;
      cam.updateProjectionMatrix();
    }
    sh.normalBias = texel * 1.2;
    sh.bias = -texel * 0.6 / (cam.far - cam.near);
  }

  // -------------------------------------------------------------------------
  /** Render the procedural sky into a cube map (one-off; ~1 s on CPU GL). */
  _bakeSky(size) {
    const old = this.skyRT;
    const rt = new THREE.WebGLCubeRenderTarget(size, {
      type: THREE.UnsignedByteType,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter
    });
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uGalN: { value: this.galN },
        uGalC: { value: this.galC },
        uNebA: { value: this.nebulaDir },
        uNebB: { value: this.nebulaDirB },
        uTexel: { value: (Math.PI / 2) / size },
        uDustGain: { value: Math.min(1, size / 1024) },
        uNoiseVol: { value: this.noiseVol || (this.noiseVol = makeNoiseVolumeTexture(64, 8)) }
      },
      vertexShader: /* glsl */`
        varying vec3 vDir;
        ${LOG_DEPTH_V}
        void main() {
          vDir = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uGalN;
        uniform vec3 uGalC;
        uniform vec3 uNebA;
        uniform vec3 uNebB;
        uniform float uTexel;
        uniform float uDustGain;
        varying vec3 vDir;
        ${LOG_DEPTH_F}
        ${GLSL_NOISE}

        // Unresolved background stars: one candidate per 3D cell (2x2x2
        // neighbourhood), gaussian PSF ~a texel wide so the bake never aliases
        vec3 starDust(vec3 d, float density, float prob, float gain) {
          vec3 p = d * density;
          vec3 base = floor(p - 0.5);
          float sigma = max(uTexel * density * 0.55, 0.04);
          vec3 acc = vec3(0.0);
          for (int i = 0; i < 8; i++) {
            float fi = float(i);
            vec3 c = base + vec3(mod(fi, 2.0), mod(floor(fi * 0.5), 2.0), floor(fi * 0.25));
            vec3 h = hash33(c);
            if (h.x > prob) continue;
            vec3 off = p - (c + hash33(c + 7.31));
            off -= d * dot(off, d);
            float b = exp(-dot(off, off) / (sigma * sigma)) * pow(h.y, gain);
            acc += mix(vec3(1.0, 0.76, 0.55), vec3(0.7, 0.8, 1.0), h.z) * b;
          }
          return acc;
        }

        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 d = normalize(vDir);
          vec3 N = uGalN, C = uGalC, E = cross(N, C);

          // Domain warp gives the clouds a filamentary, sheared structure
          vec3 w = nz4(d * 2.1 + 1.7).xyz;
          vec3 dw = normalize(d + w * 0.07);
          float lat = asin(clamp(dot(dw, N), -1.0, 1.0));
          float lon = atan(dot(d, E), dot(d, C));

          // --- Milky Way ---------------------------------------------------
          float bulge = exp(-lon * lon / (2.0 * 0.5 * 0.5));
          float thick = 0.07 + 0.12 * bulge;
          float disk = exp(-lat * lat / (2.0 * thick * thick));
          float along = 0.25 + 0.75 * exp(-lon * lon / (2.0 * 1.3 * 1.3));
          float core = exp(-(lat * lat * 2.5 + lon * lon) / (2.0 * 0.17 * 0.17));
          float clouds = fbm(dw * 3.3, 6);
          float fine = fbm(dw * 12.0 + 5.0, 4);
          float light = disk * along * (0.25 + 1.3 * pow(clouds, 2.4)) * (0.7 + 0.6 * fine) + core * 1.6;

          // Dust lanes: ridged filaments hugging the mid-plane (a Great
          // Rift) plus patchy dark clouds, reddening what they don't block
          float laneBand = exp(-lat * lat / (2.0 * pow(thick * 0.5, 2.0)));
          float lanes = smoothstep(0.42, 0.8, ridged(dw * 4.0 + 2.0, 6)) * laneBand;
          float patches = smoothstep(0.55, 0.78, fbm(dw * 6.5 + 9.0, 5)) * exp(-lat * lat / (2.0 * thick * thick));
          float absorb = clamp(max(lanes, patches * 0.85), 0.0, 0.94);

          vec3 mwCol = mix(vec3(0.7, 0.78, 1.0), vec3(1.0, 0.86, 0.68), clamp(bulge * 0.8 + core, 0.0, 1.0));
          vec3 mw = mwCol * light * (1.0 - absorb);
          mw *= mix(vec3(1.0), vec3(1.0, 0.8, 0.62), clamp(absorb * 0.9, 0.0, 1.0));

          // HII regions (pink knots) and blue reflection wisps along the band
          float hii = smoothstep(0.64, 0.86, fbm(d * 7.5 + 21.0, 5)) * disk * along * (1.0 - absorb * 0.6);
          float refl = smoothstep(0.62, 0.86, fbm(dw * 5.0 + 31.0, 4)) * disk * (1.0 - absorb);
          vec3 emis = vec3(1.0, 0.26, 0.36) * hii * 1.4 + vec3(0.35, 0.55, 1.0) * refl * 0.5;

          // --- Off-band nebula complex: teal/magenta, with dark globules --
          float fa = max(0.0, dot(d, uNebA));
          float neb = pow(fa, 7.0);
          float shape = fbm(dw * 4.2 + 40.0, 6);
          float mask = neb * smoothstep(0.4, 0.82, shape);
          vec3 nebCol = mix(vec3(0.12, 0.55, 0.65), vec3(0.9, 0.24, 0.62),
                            smoothstep(0.42, 0.72, fbm(d * 3.0 + 55.0, 4)));
          vec3 nebula = nebCol * mask * 1.6;
          nebula *= 1.0 - 0.85 * smoothstep(0.62, 0.84, ridged(d * 8.0 + 60.0, 4)) * smoothstep(0.1, 0.5, neb);
          // a faint wide glow so the complex does not end in a hard cut
          nebula += nebCol * pow(fa, 4.0) * 0.06 * shape;

          // --- Second complex: a blue reflection nebula with a warm, dusty
          // heart, on the planet's side of the sky (depth behind close-ups)
          float fb = max(0.0, dot(d, uNebB));
          float shapeB = fbm(dw * 3.1 + 83.0, 6);
          float maskB = pow(fb, 5.0) * smoothstep(0.36, 0.8, shapeB);
          vec3 nebB = mix(vec3(0.22, 0.4, 1.0), vec3(1.0, 0.5, 0.26), pow(fb, 26.0) * shapeB * 1.6);
          nebula += nebB * maskB * 1.25 * (1.0 - 0.7 * smoothstep(0.55, 0.8, ridged(d * 6.0 + 17.0, 4)));

          // --- Galactic cirrus: faint all-sky wisps, so no part of the sky
          // is a featureless void
          float ci = fbm(dw * 2.3 + 70.0, 5);
          vec3 cirrus = mix(vec3(0.2, 0.3, 0.55), vec3(0.5, 0.33, 0.24), smoothstep(0.35, 0.7, shape))
                      * pow(ci, 3.0) * (0.35 + 0.65 * disk);

          // --- Stars too faint to resolve -----------------------------------
          float sd = 0.2 + 1.4 * disk * along * (1.0 - absorb * 0.9);
          vec3 stars = starDust(d, 120.0, 0.55, 3.0) * sd
                     + starDust(d + 0.37, 190.0, 0.7, 4.0) * sd * 0.8;

          vec3 col = vec3(0.0010, 0.0012, 0.0021)
                   + mw * 0.078 + emis * 0.04 + nebula * 0.046 + cirrus * 0.012
                   + stars * 0.115 * uDustGain;
          // gamma-2 encoding: 8-bit cube without banding in the blacks
          gl_FragColor = vec4(sqrt(max(col, vec3(0.0))), 1.0);
        }`,
      side: THREE.BackSide,
      depthTest: false,
      depthWrite: false
    });
    const bakeScene = new THREE.Scene();
    const box = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), mat);
    bakeScene.add(box);
    const cam = new THREE.CubeCamera(0.5, 50, rt);
    cam.update(this.renderer, bakeScene);
    box.geometry.dispose();
    mat.dispose();
    this.skyRT = rt;
    this.skySize = size;
    old?.dispose();
  }

  _skyMaterial(intensity) {
    return new THREE.ShaderMaterial({
      uniforms: {
        uSky: { value: this.skyRT.texture },
        uIntensity: { value: intensity },
        uFillDir: { value: this.sunDir.clone().negate().add(new THREE.Vector3(0, -0.25, 0)).normalize() },
        uFill: { value: new THREE.Color(0, 0, 0) }
      },
      vertexShader: /* glsl */`
        varying vec3 vDir;
        ${LOG_DEPTH_V}
        void main() {
          vDir = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform samplerCube uSky;
        uniform float uIntensity;
        uniform vec3 uFillDir;
        uniform vec3 uFill;
        varying vec3 vDir;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 d = normalize(vDir);
          vec3 c = textureCube(uSky, d).rgb;
          c = c * c * uIntensity;
          c += uFill * pow(max(0.0, dot(d, uFillDir)), 1.5);
          gl_FragColor = vec4(c, 1.0);
        }`,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false
    });
  }

  _buildSkyMesh() {
    this.skyMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this._skyMaterial(1.0));
    this.skyMesh.scale.setScalar(300000);
    this.skyMesh.renderOrder = -100;
    this.skyMesh.frustumCulled = false;
    this.group.add(this.skyMesh);
    this.skydome = this.skyMesh; // old name
  }

  // -------------------------------------------------------------------------
  _buildStars() {
    const cat = makeStarCatalogue(STAR_MAX, { seed: 2024, bandNormal: this.galN.toArray() });
    const R = 260000;
    const pos = new Float32Array(STAR_MAX * 3);
    for (let i = 0; i < STAR_MAX * 3; i++) pos[i] = cat.dir[i] * R;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('aColor', new THREE.BufferAttribute(cat.color, 3));
    geom.setAttribute('aSize', new THREE.BufferAttribute(cat.size, 1));
    geom.setAttribute('aSpike', new THREE.BufferAttribute(cat.spike, 1));
    geom.setDrawRange(0, Math.min(this.gfx.preset.starCount, STAR_MAX));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uPixelRatio: { value: 1 } },
      vertexShader: /* glsl */`
        attribute vec3 aColor;
        attribute float aSize;
        attribute float aSpike;
        uniform float uPixelRatio;
        varying vec3 vColor;
        varying float vSpike;
        varying float vPx;
        ${LOG_DEPTH_V}
        void main() {
          vColor = aColor;
          vSpike = aSpike;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float s = aSize * (1.0 + 2.2 * aSpike);
          gl_PointSize = s * uPixelRatio;
          vPx = 0.5 * s;                  // sprite radius in CSS px
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vColor;
        varying float vSpike;
        varying float vPx;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r = length(p);
          float rp = r * vPx;                         // distance in CSS px
          // crisp PSF: ~0.7 px gaussian core + faint exponential halo
          float a = exp(-rp * rp) + 0.05 * exp(-rp / 1.6);
          if (vSpike > 0.0) {
            vec2 q = abs(p) * vPx;
            float len = vPx * 0.3;
            float sx = exp(-q.y * q.y / 0.3) * exp(-q.x / len);
            float sy = exp(-q.x * q.x / 0.3) * exp(-q.y / len);
            a += (sx + sy) * 0.3 * vSpike;
          }
          a *= 1.0 - smoothstep(0.85, 1.0, r);
          gl_FragColor = vec4(vColor * a, 1.0);
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
    const discR = SUN_DIST * Math.tan(SUN_ANGULAR_RADIUS);
    const QUAD = '16.0'; // quad half-size in disc radii (corona + glare)
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: discR * 16 },
        uColor: { value: this.sunColor }
      },
      vertexShader: /* glsl */`
        uniform float uSize;
        varying vec2 vP;
        ${LOG_DEPTH_V}
        void main() {
          vP = position.xy;
          vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
          mv.xy += position.xy * uSize;
          gl_Position = projectionMatrix * mv;
          ${LOG_DEPTH_V_MAIN}
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uColor;
        varying vec2 vP;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          float r = length(vP) * ${QUAD};             // in disc radii
          float ang = atan(vP.y, vP.x);
          // limb-darkened photosphere
          float mu = sqrt(max(0.0, 1.0 - r * r));
          float disc = smoothstep(1.02, 0.96, r) * (0.42 + 0.58 * pow(mu, 0.55));
          // corona: soft streamers that thin with radius
          float st = 0.55 + 0.25 * sin(ang * 5.0 + 1.3) + 0.15 * sin(ang * 11.0 + 0.4) + 0.05 * sin(ang * 23.0 + 2.2);
          float rr = max(r - 1.0, 0.0);
          float corona = exp(-rr * 4.2) * (0.45 + 0.55 * st) * 1.8 + exp(-rr * 1.3) * 0.07 * st;
          float glare = exp(-rr * 0.4) * 0.012;
          float fade = 1.0 - smoothstep(0.7, 1.0, r / ${QUAD});
          vec3 col = uColor * disc * 48.0;
          col += vec3(1.0, 0.84, 0.66) * (corona * 2.0 + glare) * smoothstep(0.96, 1.0, r) * fade;
          gl_FragColor = vec4(col, 1.0);
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.sun = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    this.sun.frustumCulled = false;
    this.sun.renderOrder = -98;
    this.group.add(this.sun);
  }

  // -------------------------------------------------------------------------
  _buildPlanet() {
    const R = this.planetRadius;
    const ringIn = R * 1.32, ringOut = R * 2.25;
    // Ring plane seen ~22 deg open from the battle; poles along its normal
    const toPlanet = this.planetDir;
    const hint = new THREE.Vector3(0.15, 1, 0.1);
    const upPerp = hint.clone().addScaledVector(toPlanet, -hint.dot(toPlanet)).normalize();
    const ringN = upPerp.clone().multiplyScalar(Math.cos(0.38)).addScaledVector(toPlanet, -Math.sin(0.38)).normalize();
    this.ringN = ringN;
    this.planetGroup = new THREE.Group();
    this.planetGroup.name = 'planet';
    this.group.add(this.planetGroup);
    const center = new THREE.Vector3();
    this.planetCenter = center;
    this.ringTex = makeRingTexture();

    const common = () => ({
      uSunDir: { value: this.sunDir },
      uSunColor: { value: this.sunColor },
      uCenter: { value: center },
      uRingN: { value: ringN },
      uRingIn: { value: ringIn },
      uRingOut: { value: ringOut },
      uRingTex: { value: this.ringTex },
      uR: { value: R }
    });
    const worldVert = /* glsl */`
      varying vec3 vNormal;
      varying vec3 vWorld;
      varying vec2 vUv;
      ${LOG_DEPTH_V}
      void main() {
        vUv = uv;
        vNormal = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
        ${LOG_DEPTH_V_MAIN}
      }`;
    const planetMat = new THREE.ShaderMaterial({
      uniforms: {
        ...common(),
        uMap: { value: getGasGiantTexture() },
        uAtmo: { value: new THREE.Color(0.45, 0.62, 1.0) }
      },
      vertexShader: worldVert,
      fragmentShader: /* glsl */`
        uniform sampler2D uMap;
        uniform sampler2D uRingTex;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform vec3 uCenter;
        uniform vec3 uRingN;
        uniform float uRingIn;
        uniform float uRingOut;
        uniform vec3 uAtmo;
        varying vec3 vNormal;
        varying vec3 vWorld;
        varying vec2 vUv;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 n = normalize(vNormal);
          vec3 v = normalize(cameraPosition - vWorld);
          // Flow detail: the band lookup is warped by two finer scales of
          // the same map (v mirrored so the warp is continuous), which
          // breaks the straight belts into eddies and festoons
          vec2 q1 = vec2(vUv.x * 5.0 + 0.13, abs(fract(vUv.y * 2.0 + 0.2) * 2.0 - 1.0));
          vec2 q2 = vec2(vUv.x * 13.0 + 0.57, abs(fract(vUv.y * 5.5 + 0.4) * 2.0 - 1.0));
          vec3 w1 = texture2D(uMap, q1).rgb;
          vec3 w2 = texture2D(uMap, q2).rgb;
          float lum1 = dot(w1, vec3(0.5, 0.4, 0.1)), lum2 = dot(w2, vec3(0.5, 0.4, 0.1));
          vec2 uv = vUv + vec2((lum1 - 0.4) * 0.016, (w1.r - w1.b - 0.12) * 0.03 + (lum2 - 0.4) * 0.011);
          vec3 albedo = texture2D(uMap, uv).rgb;
          albedo *= 0.8 + 0.5 * lum2;
          // deeper, more saturated belts
          float al = dot(albedo, vec3(0.3, 0.5, 0.2));
          albedo = max(mix(vec3(al), albedo, 1.35), 0.0) * mix(0.78, 1.12, smoothstep(0.2, 0.6, al));
          float ndl = dot(n, uSunDir);
          float mu = max(dot(n, v), 0.0);
          // Thick atmosphere: Lambert with limb DARKENING (the disc must
          // read as a ball), a soft reddened terminator
          float lit = pow(max(ndl, 0.0), 0.9) * (0.38 + 0.62 * pow(mu, 0.55));
          float term = smoothstep(-0.06, 0.2, ndl);
          vec3 tint = mix(vec3(1.0, 0.5, 0.3), vec3(1.0), smoothstep(0.0, 0.38, ndl));
          // Ring shadow: where the ray toward the sun crosses the ring plane
          vec3 rel = vWorld - uCenter;
          float denom = dot(uSunDir, uRingN);
          float shadow = 1.0;
          if (abs(denom) > 1e-4) {
            float t = -dot(rel, uRingN) / denom;
            if (t > 0.0) {
              float u = (length(rel + uSunDir * t) - uRingIn) / (uRingOut - uRingIn);
              if (u > 0.0 && u < 1.0) shadow = 1.0 - 0.9 * texture2D(uRingTex, vec2(u, 0.5)).a;
            }
          }
          vec3 col = albedo * uSunColor * lit * term * tint * shadow * 0.78;
          col += albedo * vec3(0.5, 0.7, 1.0) * 0.004;        // night side: starlight
          // Rayleigh haze: a blue-white veil that thickens toward the limb
          // on the day side, and a thin forward-scattered arc that wraps a
          // little past the terminator
          float rimF = pow(1.0 - mu, 3.0);
          col += uAtmo * rimF * smoothstep(-0.18, 0.5, ndl) * 0.55;
          col += uAtmo * pow(1.0 - mu, 6.0) * smoothstep(-0.4, 0.05, ndl)
               * pow(max(dot(-v, uSunDir), 0.0), 3.0) * 1.6;
          gl_FragColor = vec4(col, 1.0);
        }`
    });
    this.planet = new THREE.Mesh(new THREE.SphereGeometry(R, 128, 64), planetMat);
    // Poles along the ring normal (SphereGeometry poles are +Y)
    this.planet.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), ringN);
    this.planetGroup.add(this.planet);

    // Limb halo: thin bright scattering layer just outside the disc, lit side only
    const haloMat = new THREE.ShaderMaterial({
      uniforms: { ...common(), uAtmo: { value: new THREE.Color(0.5, 0.68, 1.0) } },
      vertexShader: worldVert,
      fragmentShader: /* glsl */`
        uniform vec3 uSunDir;
        uniform vec3 uCenter;
        uniform float uR;
        uniform vec3 uAtmo;
        varying vec3 vWorld;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec3 rd = normalize(vWorld - cameraPosition);
          float tca = dot(uCenter - cameraPosition, rd);
          vec3 closest = cameraPosition + rd * tca;
          float b = length(closest - uCenter) / uR;      // impact parameter
          if (b < 1.0) discard;
          float h = b - 1.0;
          float halo = exp(-h / 0.006) * 0.9 + exp(-h / 0.03) * 0.18;
          float lit = smoothstep(-0.3, 0.4, dot(normalize(closest - uCenter), uSunDir));
          gl_FragColor = vec4(uAtmo * halo * lit, 1.0);
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.planetHalo = new THREE.Mesh(new THREE.SphereGeometry(R * 1.09, 96, 48), haloMat);
    this.planetGroup.add(this.planetHalo);

    // Rings: radial density texture, lit face vs back-lit face, planet shadow
    const ringGeom = new THREE.RingGeometry(ringIn, ringOut, 256, 1);
    const uv = ringGeom.attributes.uv;
    const p = ringGeom.attributes.position;
    for (let i = 0; i < uv.count; i++) {
      const r = Math.hypot(p.getX(i), p.getY(i));
      uv.setXY(i, (r - ringIn) / (ringOut - ringIn), 0.5);
    }
    const ringMat = new THREE.ShaderMaterial({
      uniforms: common(),
      vertexShader: worldVert,
      fragmentShader: /* glsl */`
        uniform sampler2D uRingTex;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform vec3 uCenter;
        uniform vec3 uRingN;
        uniform float uR;
        varying vec2 vUv;
        varying vec3 vWorld;
        ${LOG_DEPTH_F}
        void main() {
          ${LOG_DEPTH_F_MAIN}
          vec4 t = texture2D(uRingTex, vUv);
          vec3 rel = vWorld - uCenter;
          float along = dot(rel, uSunDir);
          vec3 perp = rel - uSunDir * along;
          // the planet's shadow across the far arc (soft penumbra)
          float shadow = along < 0.0 ? smoothstep(uR * 0.985, uR * 1.03, length(perp)) : 1.0;
          vec3 rd = normalize(vWorld - cameraPosition);
          float a = t.a;
          float sunSide = sign(dot(uSunDir, uRingN));
          float camSide = sign(dot(cameraPosition - uCenter, uRingN));
          float ndl = abs(dot(uSunDir, uRingN));
          // Lit face: dense bands reflect, thin ones are dim. Unlit face:
          // only light that made it THROUGH shows - the dense B ring goes
          // dark and the thin rings glow.
          float reflectF = (0.3 + 0.7 * a) * (0.4 + 0.6 * ndl);
          float transF = 4.0 * a * (1.0 - a) * 0.42 + (1.0 - a) * 0.1;
          float face = sunSide == camSide ? reflectF : transF;
          // forward scattering by fine ring dust when looking up-sun, and an
          // opposition surge with the sun behind the viewer
          float fwd = pow(max(dot(rd, uSunDir), 0.0), 5.0) * (1.0 - a) * 1.3;
          float opp = pow(max(dot(-rd, uSunDir), 0.0), 10.0) * 0.3;
          vec3 col = t.rgb * uSunColor * (0.012 + shadow * (face * (0.85 + opp) + fwd));
          gl_FragColor = vec4(col, a * 0.94);
        }`,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.rings = new THREE.Mesh(ringGeom, ringMat);
    this.rings.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), ringN);
    this.planetGroup.add(this.rings);
    for (const m of [this.planet, this.planetHalo, this.rings]) m.frustumCulled = false;
  }

  // -------------------------------------------------------------------------
  /**
   * PMREM environment from the baked sky (boosted, plus a cool fill lobe
   * opposite the sun), the sun and the planet - gives the metallic hull
   * plating reflections with some colour instead of pure black.
   */
  _buildEnvMap() {
    const envScene = new THREE.Scene();
    const sky = new THREE.Mesh(new THREE.BoxGeometry(2000, 2000, 2000), this._skyMaterial(2.6));
    sky.material.uniforms.uFill.value.setRGB(0.085, 0.125, 0.23);
    envScene.add(sky);
    const sun = new THREE.Mesh(new THREE.SphereGeometry(900 * Math.tan(0.03), 16, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(30, 27, 23) }));
    sun.position.copy(this.sunDir).multiplyScalar(900);
    envScene.add(sun);
    const planet = new THREE.Mesh(new THREE.SphereGeometry(900 * Math.sin(PLANET_ANGULAR_RADIUS), 32, 24),
      new THREE.MeshLambertMaterial({ map: getGasGiantTexture() }));
    planet.position.copy(this.planetDir).multiplyScalar(900);
    envScene.add(planet);
    const l = new THREE.DirectionalLight(0xffffff, 2.2);
    l.position.copy(this.sunDir).multiplyScalar(1000);
    envScene.add(l);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const rt = pmrem.fromScene(envScene, 0.02, 0.1, 5000);
    this.scene.environment = rt.texture;
    this.envRT = rt;
    pmrem.dispose();
    sky.geometry.dispose(); sky.material.dispose();
    sun.geometry.dispose(); sun.material.dispose();
    planet.geometry.dispose(); planet.material.dispose();
  }

  // -------------------------------------------------------------------------
  update(camera, delta, focus = null, viewDist = 0) {
    this.time += delta;
    this._fitShadow(camera, focus, viewDist || camera.userData.viewDist || 100);
    const cp = camera.position;
    // Keep the far backdrop centred on the camera so it never gets parallax
    this.skyMesh.position.copy(cp);
    this.stars.position.copy(cp);
    this.stars.material.uniforms.uPixelRatio.value = this.renderer.getPixelRatio();
    this.sun.position.copy(cp).addScaledVector(this.sunDir, SUN_DIST);
    this.planetGroup.position.copy(cp).addScaledVector(this.planetDir, PLANET_DIST);
    this.planetCenter.copy(this.planetGroup.position);

    // Rim light from beyond the subject, a little above the view axis
    camera.getWorldDirection(_fwd);
    _up.copy(camera.up).applyQuaternion(camera.quaternion);
    this.rimLight.position.copy(_fwd).addScaledVector(_up, 0.35).normalize().multiplyScalar(1000);

    this.dust.update(camera, delta);
  }
}

const _fwd = new THREE.Vector3();
const _up = new THREE.Vector3();

/**
 * Free the GPU resources of every mesh / points / line under `root`:
 * geometries, materials and their (non render-target) textures. Shared
 * cached textures are safe to dispose; three re-uploads them on next use.
 */
export function disposeTree(root) {
  const seen = new Set();
  const tex = (t) => {
    if (t?.isTexture && !t.isRenderTargetTexture && !seen.has(t)) { seen.add(t); t.dispose(); }
  };
  root.traverse(o => {
    if (o.geometry && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); }
    if (o.isInstancedMesh) o.dispose();
    for (const m of [].concat(o.material || [])) {
      if (seen.has(m)) continue;
      seen.add(m);
      for (const k in m) tex(m[k]);
      for (const k in m.uniforms || {}) tex(m.uniforms[k]?.value);
      m.dispose();
    }
  });
}
