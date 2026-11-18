import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';

// Effect objects bake small canvas textures: a canvas that accepts every 2D call is enough
const ctx2d = new Proxy({}, {
  get: (t, prop) => (prop === 'createImageData' || prop === 'getImageData'
    ? (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h })
    : prop === 'measureText' ? () => ({ width: 10 }) : prop === 'canvas' ? fakeCanvas() : () => ctx2d),
  set: () => true
});
function fakeCanvas() { return { width: 4, height: 4, style: {}, getContext: () => ctx2d }; }
globalThis.document ??= { createElement: () => fakeCanvas() };

const {
  JET_CORE_DEG, JET_MAX_DEG, JET_REACH_KM, JET_LEVEL, JET_FALL, JET_NEAR_KM, JET_LENS_CLEAR, JET_CAP, JET_SEGMENTS, JET_GAIN_SIN, TAN_CORE,
  jetAxial, jetWidth, jetColumn, jetCap, driveRadius
} = await import('../src/render/PlumeProfile.js');
const { DriveJets, jetPose, NOZZLE_AFT } = await import('../src/render/DriveJet.js');
const { createTorch, torchCentreline, torchCoreEnd, plumeIgnition } = await import('../src/render/Torch.js');
const { SceneManager } = await import('../src/SceneManager.js');
const { hullSize, HULL_CLASSES } = await import('../src/render/Hulls.js');
const { shipScale, trueK, trueLengthM } = await import('../src/render/Scale.js');
const { Interpolator } = await import('../src/Interpolator.js');
const { hullQuat } = await import('../src/HullFrame.js');

const FOV = 55;
const PX720 = 2 * Math.tan(FOV / 2 * Math.PI / 180) / 720;

/** A hull's bells and spear exactly as SceneManager.createShip builds them. */
function drive(type) {
  const size = hullSize(type);
  const cfg = SceneManager.prototype.getEngineConfig.call({}, type, size);
  const plumeR = size.width * (cfg.count <= 2 ? 0.2 : cfg.count <= 4 ? 0.155 : 0.11);
  const torch = createTorch({ radius: plumeR, length: size.length * cfg.plumeLength * 0.45, cluster: cfg.count, phase: 0.37 });
  return { size, cfg, plumeR, torch, driveR: driveRadius(cfg.positions, plumeR) };
}

/**
 * Side-on centreline radiance of the whole plume z km aft of the nozzle
 * plane: every bell's flame (drawn at scale k) plus the spear (on unless
 * cinematic, as SceneManager.syncCamera draws it).
 */
function plumeAt(z, d, p, k, jets, thr) {
  return d.cfg.count * torchCentreline(z / k, p) + (jets ? jetAxial(z, thr) : 0);
}

// ------------------------------------------------------------- the jet itself

test('the spear has the engine\'s geometry, starts at the nozzle and only fades', () => {
  const py = readFileSync(new URL('../../src/drive_plume.py', import.meta.url), 'utf8');
  const constant = (name) => Number(new RegExp(`^${name} = ([0-9.]+)`, 'm').exec(py)[1]);
  assert.equal(JET_CORE_DEG, constant('PLUME_CORE_HALF_ANGLE_DEG'));
  assert.equal(JET_MAX_DEG, constant('PLUME_MAX_HALF_ANGLE_DEG'));
  assert.ok(JET_REACH_KM <= constant('PLUME_MAX_RANGE_KM'));
  // no fade-in: full brightness at the bells (the old jet rose from zero over
  // 0.6 km and peaked ~8 km astern - the hole and the cloud)
  assert.equal(jetAxial(0), JET_LEVEL);
  let prev = Infinity;
  for (let z = 0; z <= JET_REACH_KM; z += 0.05) {
    const v = jetAxial(z);
    assert.ok(v < prev || (v === 0 && prev === 0), `strictly fading at ${z.toFixed(2)} km`);
    prev = v;
  }
  assert.equal(jetAxial(JET_REACH_KM), 0);
  // far field: about 1/z (the density of a 3 degree cone falls with the
  // square of the distance), a touch steeper so a hairline still fades
  assert.ok(JET_FALL >= 1 && JET_FALL < 1.3);
  assert.ok(Math.abs(jetAxial(20) / jetAxial(10) - Math.pow(10.5 / 20.5, JET_FALL)) < 1e-12);
  // and the line-of-sight gain is capped (a crossing at 20-35 degrees), the rule that keeps it off the vanishing point
  assert.ok(JET_GAIN_SIN > Math.sin(20 * Math.PI / 180) && JET_GAIN_SIN < Math.sin(35 * Math.PI / 180));
  // throttle scales it, the wash cone sets its width (3 degrees from the drive's drawn radius)
  assert.equal(jetAxial(3, 0.5), 0.5 * jetAxial(3, 1));
  assert.ok(Math.abs(jetWidth(10, 0.01) - (0.01 + 10 * Math.tan(3 * Math.PI / 180))) < 1e-12);
  // the engine's wash still has a visible spear where it heats a ship hard
  // (above 4 GJ/s out to ~20 km at full burn): readable side on at 6 km
  assert.ok(jetAxial(6) > 0.01);
});

// erf column vs a brute-force line integral of the same emission
function brute(ro, rd, w0) {
  let col = 0;
  const dt = 0.0005;
  for (let t = 0; t < 80; t += dt) {
    const x = ro[0] + rd[0] * t, y = ro[1] + rd[1] * t, z = -(ro[2] + rd[2] * t);
    if (z < 0 || z > JET_REACH_KM) continue;
    const w = jetWidth(z, w0);
    const rc = JET_LENS_CLEAR * w / Math.max(Math.hypot(rd[0], rd[1]), JET_GAIN_SIN);
    const lens = 1 - Math.exp(-(t * t) / (rc * rc));
    col += jetAxial(z) / (1.7724539 * w) * Math.exp(-(x * x + y * y) / (w * w)) * lens * dt;
  }
  return col;
}
const unit = (v) => { const l = Math.hypot(...v); return v.map(x => x / l); };
const toward = (cam, tgt) => unit(tgt.map((x, i) => x - cam[i]));

test('the closed-form column matches a brute-force integral of the beam', () => {
  const w0 = 0.008;
  const cases = [
    [[0.2, 0, 0.35], [0, 0, -0.1]], [[0.2, 0, 0.35], [0, 0, -3]], [[0.2, 0, 0.35], [0, 0, -30]],
    [[1, 0, 0.05], [0, 0, -0.3]], [[2, 0, -10], [0, 0, -10]], [[0.3, 0.4, 2], [0, 0, -6]]
  ];
  for (const [cam, tgt] of cases) {
    const rd = toward(cam, tgt), sinT = Math.hypot(rd[0], rd[1]);
    const b = brute(cam, rd, w0) * sinT / Math.max(sinT, JET_GAIN_SIN);   // the capped line-of-sight gain
    for (const segments of [JET_SEGMENTS.high, JET_SEGMENTS.low]) {
      const c = jetColumn(cam, rd, { w0 }, { segments });
      const tol = segments === JET_SEGMENTS.high ? 0.13 : 0.2;   // (about 10% typical)
      assert.ok(Math.abs(c - b) <= tol * b + 1e-4, `${cam} -> ${tgt} (${segments}): ${c} vs ${b}`);
    }
  }
  // straight down the axis the column is huge and the soft ceiling holds it
  for (const cam of [[0.01, 0, 5], [0, 0.003, 5], [0.05, 0.02, -2]]) {
    const rd = toward(cam, [0, 0, cam[2] > 0 ? -1 : 0]), sinT = Math.hypot(rd[0], rd[1]);
    const c = jetCap(jetColumn(cam, rd, { w0 })), b = jetCap(brute(cam, rd, w0) * sinT / Math.max(sinT, JET_GAIN_SIN));
    assert.ok(c <= JET_CAP && Math.abs(c - b) < 0.15 * b, `end-on ${cam}: ${c} vs ${b}`);
  }
});

/**
 * Screen brightness along the projected axis from a camera, as the shader
 * draws it: one ray per axis point z, kept when the point is inside the
 * camera's view cone around the ship.
 */
function screenLine(cam, w0, zs, opt) {
  const look = toward(cam, [0, 0, 0]);
  const px = opt.pixelWorld;
  const ang = (u, v) => Math.acos(Math.min(1, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]));
  const out = [];
  for (const z of zs) {
    const rd = toward(cam, [0, 0, -z]);
    if (ang(rd, look) * 180 / Math.PI > 40) continue;
    // points within four pixels of the nozzle on screen are the nozzle; and
    // the first kilometre of a beam thinner than a pixel is under the
    // flame's own pixel-floored streak (render/Torch.js)
    if (out.length === 0 && (ang(rd, look) < 4 * px
      || (z < 2 * JET_NEAR_KM && jetWidth(z, w0) < px * Math.hypot(cam[0], cam[1], cam[2] + z)))) continue;
    out.push([z, jetCap(jetColumn(cam, rd, { w0 }, opt))]);
  }
  return out;
}

test('from every angle the spear is brightest at the ship and fades into the distance: no cloud', () => {
  const zs = [];
  for (let z = 0.02; z < JET_REACH_KM; z *= 1.15) zs.push(z);
  const deg = Math.PI / 180;
  for (const type of ['corvette', 'frigate', 'battleship', 'dreadnought']) {
    const d = drive(type);
    const Lt = trueLengthM(type) / 1000;
    // the owner's framing (bow quarter, 3-5 lengths), from far ahead down the
    // axis (where the old jet showed a detached blob), abeam, astern-quarter
    const views = [
      [3, 25, 10], [4, 30, 15], [5, 35, 12], [4, 20, 20],
      [30, 6, 3], [100, 10, 4], [300, 8, 2],
      [10, 90, 5], [100, 90, 5], [1000, 90, 5],
      [6, 150, 10]
    ];
    for (const [hulls, az, el] of views) {
      const dist = hulls * Lt;
      for (const mode of ['true', 'auto']) {
        const k = shipScale(mode, type, dist, FOV);
        const w0 = d.driveR * k, aft = NOZZLE_AFT * hullSize(type).length * k;
        // camera around the hull centre; the jet frame's origin is the nozzle plane
        const a = az * deg, e = el * deg;
        const cam = [dist * Math.cos(e) * Math.sin(a), dist * Math.sin(e), dist * Math.cos(e) * Math.cos(a) + aft];
        const line = screenLine(cam, w0, zs, { pixelWorld: PX720 });
        if (line.length < 3) continue;
        const tag = `${type} ${mode} ${hulls}L az ${az}`;
        // non-increasing along the projected axis (4% slack for the sub-chord rule)
        for (let i = 1; i < line.length; i++) {
          assert.ok(line[i][1] <= line[i - 1][1] * 1.04 + 1e-4, `${tag}: rises at ${line[i][0].toFixed(2)} km (${line[i - 1][1].toFixed(3)} -> ${line[i][1].toFixed(3)})`);
        }
        // and the far half is a few percent of the root (the jet's own root:
        // the flame adds far more there) - the old jet's far cloud was
        // brighter than its root
        const root = line[0][1], far = Math.max(...line.filter(([z]) => z > JET_REACH_KM / 2).map(([, v]) => v), 0);
        // (from fleet range the root is a sub-pixel hairline, diluted by the floor: monotone only)
        if (hulls <= 300) assert.ok(far <= 0.1 * root + 1e-4, `${tag}: far ${far.toFixed(3)} vs root ${root.toFixed(3)}`);
      }
    }
  }
});

test('looking straight down the axis the jet is bounded and stays on the ship', () => {
  // a camera on the axis, far ahead and close ahead (through the hull): the
  // flame's own end-on disc is the picture; the jet adds a faint, bounded
  // ring a few degrees across around it and nothing out in space
  const w0 = 0.01;
  for (const camZ of [8, 1, 30]) {
    let peak = 0, wide = 0;
    for (let off = 0; off <= 0.4; off += 0.002) {
      const rd = unit([Math.sin(off), 0, -Math.cos(off)]);
      const v = jetCap(jetColumn([0, 0, camZ], rd, { w0 }, { pixelWorld: PX720 }));
      assert.ok(v <= JET_CAP + 1e-9);
      peak = Math.max(peak, v);
      if (off > 0.1) wide = Math.max(wide, v);
    }
    assert.ok(peak < 0.5 * JET_LEVEL, `${camZ} km ahead: a faint ring (${peak.toFixed(3)})`);
    assert.ok(wide < 0.05 * peak + 1e-4, `${camZ} km ahead: light outside 6 degrees (${wide.toFixed(4)} vs ${peak.toFixed(3)})`);
  }
});

test('a camera sitting in the beam (dead astern) sees no veil of its own surroundings', () => {
  // the old risk: inside the cone the jet is drawn at the lens, and an
  // optically thin beam around the camera would glow the same in every direction
  for (const type of ['corvette', 'battleship']) {
    const w0 = drive(type).driveR * trueK(type);
    for (const aft of [0.2, 0.5, 2, 8]) {
      const cam = [0.3 * jetWidth(aft, w0), 0, -aft];   // in the core
      // the view of a camera looking at the ship, but for the ship itself (the
      // bright root around the bells is the disc it should see); looking the
      // other way, down the jet, it sees the beam converge - also not a veil
      let veil = 0;
      for (let off = 0.3; off <= 1.75; off += 0.02) {
        for (const phi of [0, 1.3, 2.6]) {
          const rd = unit([Math.sin(off) * Math.cos(phi), Math.sin(off) * Math.sin(phi), Math.cos(off)]);
          veil = Math.max(veil, jetCap(jetColumn(cam, rd, { w0 }, { pixelWorld: PX720 })));
        }
      }
      assert.ok(veil < 0.05, `${type} ${aft} km astern: veil ${veil.toFixed(3)}`);
      // and toward the ship the root is still there
      const toShip = toward(cam, [0, 0, -0.01]);
      assert.ok(jetCap(jetColumn(cam, toShip, { w0 }, { pixelWorld: PX720 })) >= 0, 'finite');
    }
  }
});

// ------------------------------------------------- one plume: torch + spear

test('torch + spear: continuous from the bell, only fading past the shock diamonds, in every scale mode', () => {
  for (const type of HULL_CLASSES) {
    const d = drive(type);
    const Lt = trueLengthM(type) / 1000;
    const scales = [['true', trueK(type)], ['cinematic', 1]];
    for (const hulls of [3, 20, 200, 2000]) scales.push([`auto ${hulls}L`, shipScale('auto', type, hulls * Lt, FOV)]);
    for (const thr of [1, 0.6, 0.2]) {
      d.torch.update(thr, 0);
      const p = d.torch.profile();
      for (const [mode, k] of scales) {
        const jets = mode !== 'cinematic';
        const tag = `${type} ${mode} k=${k.toFixed(4)} throttle ${thr}`;
        const zCore = torchCoreEnd(p) * k, zEnd = jets ? 0.9 * JET_REACH_KM : p.len * k;
        // fine over the flame (4000 samples), then on to the spear's fade
        const zFlame = Math.min(zEnd, 1.2 * p.len * k);
        const zs = [];
        for (let i = 1; i <= 4000; i++) zs.push(zFlame * i / 4000);
        for (let i = 1; i <= 2000 && zFlame < zEnd; i++) zs.push(zFlame + (zEnd - zFlame) * i / 2000);
        let prev = plumeAt(0, d, p, k, jets, thr), peak = prev, maxJump = 0;
        const atCore = plumeAt(zCore, d, p, k, jets, thr);
        for (const z of zs) {
          const v = plumeAt(z, d, p, k, jets, thr);
          peak = Math.max(peak, v);
          maxJump = Math.max(maxJump, Math.abs(v - prev));
          if (z > zCore) assert.ok(v <= prev + 1e-9, `${tag}: brightens at ${z.toFixed(3)} km`);
          else if (z > p.r0 * k) assert.ok(v >= atCore - 1e-9, `${tag}: a hole inside the flame at ${z.toFixed(3)} km`);
          if (jets) assert.ok(v > 0, `${tag}: dark gap at ${z.toFixed(3)} km`);
          prev = v;
        }
        // continuous: no step anywhere at this resolution
        assert.ok(maxJump < 0.06 * peak, `${tag}: step of ${maxJump.toFixed(3)} (peak ${peak.toFixed(2)})`);
        // at full burn the shock-diamond zone (three cells) is the start of the flame,
        // not all of it (an idling torch is a short flicker of diamonds)
        if (thr === 1) assert.ok(zCore < 0.65 * p.len * k, `${tag}: diamonds cover ${(zCore / (p.len * k)).toFixed(2)} of the flame`);
      }
    }
    d.torch.dispose();
  }
});

// ------------------------------------------- rigid, instant, deterministic

function shipGroup(type, k, extra = {}) {
  const g = new THREE.Group();
  const size = hullSize(type);
  g.userData = {
    shipType: type, size, k, drawn: { length: size.length * k, width: size.width * k },
    driveRadius: drive(type).driveR, plumeThrottle: 1, ...extra
  };
  return g;
}

test('the spear leaves the drawn nozzle plane straight down the hull axis: a pure function of the attitude', () => {
  const pose = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), w0: 0, throttle: 0 };
  for (const k of [trueK('frigate'), trueK('frigate') * 7, 1]) {
    const g = shipGroup('frigate', k);
    g.position.set(120, -4, 9);
    // a ship turning hard: three attitudes a second apart; the turn rate is
    // not an input, so no history and no lag can bend the spear
    for (const yaw of [0, 0.35, 0.7]) {
      const fwd = [Math.sin(yaw), 0, Math.cos(yaw)];
      const q = hullQuat(fwd, [0, 1, 0]);
      g.quaternion.set(q[0], q[1], q[2], q[3]);
      g.userData.omega = [0, 0.35, 0];
      assert.ok(jetPose(g, pose));
      const axis = new THREE.Vector3(0, 0, -1).applyQuaternion(pose.quat);
      assert.ok(axis.distanceTo(new THREE.Vector3(...fwd).negate()) < 1e-9, 'axis = -forward');
      const want = g.position.clone().addScaledVector(new THREE.Vector3(...fwd), -NOZZLE_AFT * g.userData.drawn.length);
      assert.ok(pose.pos.distanceTo(want) < 1e-9, 'origin at the drawn nozzle plane');
      assert.ok(Math.abs(pose.w0 - g.userData.driveRadius * k) < 1e-12, 'drawn drive radius');
    }
  }
  // no jet for a hidden, idle or dying ship
  assert.equal(jetPose(Object.assign(shipGroup('frigate', 1), { visible: false }), pose), null);
  assert.equal(jetPose(shipGroup('frigate', 1, { plumeThrottle: 0.01 }), pose), null);
  assert.equal(jetPose(shipGroup('frigate', 1, { dyingNow: true }), pose), null);
  // the instanced draw: one jet per burning living hull, none when switched off (?jets=0)
  const scene = new THREE.Scene();
  const jets = new DriveJets(scene);
  const ships = new Map([['a', shipGroup('frigate', 0.02)], ['b', shipGroup('frigate', 0.02, { plumeThrottle: 0 })],
    ['c', shipGroup('battleship', 0.02, { dyingNow: true })], ['d', shipGroup('destroyer', 0.02, { plumeThrottle: 0.4 })]]);
  jets.update(ships, true);
  assert.equal(jets.mesh.count, 2);
  assert.equal(jets.jetAttr.array[2], JET_REACH_KM);
  assert.ok(Math.abs(jets.jetAttr.array[4] - 0.4) < 1e-6);
  const m = new THREE.Matrix4(), s = new THREE.Vector3();
  jets.mesh.getMatrixAt(0, m); m.decompose(new THREE.Vector3(), new THREE.Quaternion(), s);
  assert.ok(Math.abs(s.x - 1) < 1e-6 && Math.abs(s.z - 1) < 1e-6, 'never scaled: the spear is in scene km');
  jets.update(ships, false);
  assert.equal(jets.mesh.count, 0);
  assert.equal(jets.mesh.visible, false);
  jets.dispose();
  // the shader keeps the renderer contracts
  const src = readFileSync(new URL('../src/render/DriveJet.js', import.meta.url), 'utf8');
  for (const chunk of ['<common>', '<logdepthbuf_pars_vertex>', '<logdepthbuf_vertex>', '<logdepthbuf_pars_fragment>', '<logdepthbuf_fragment>']) {
    assert.ok(src.includes(`#include ${chunk}`), chunk);
  }
  assert.ok(!/uTime|uFlow|currentTime/.test(src), 'no clock: a steady lance');
  // shader constants are the JS model's
  assert.ok(src.includes('TAN_CORE') && Math.abs(TAN_CORE - Math.tan(3 * Math.PI / 180)) < 1e-15);
});

test('ignition and cutoff are instant along the whole plume, and a scrub shows the same plume', () => {
  // a recording: full burn; frame 4 carries the tick 3-4 at zero throttle (cut at
  // t = 3), frame 7 the tick 6-7 at full (relight at t = 6)
  const thr = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1];
  const trace = thr.map((th, t) => ({
    t, ships: { s: { pos: [t * 10, 0, 0], vel: [10, 0, 0], fwd: [Math.cos(t * 0.2), Math.sin(t * 0.2), 0], up: [0, 0, 1], thrust: th } }
  }));
  const frameAt = (time) => {
    let i = Math.min(trace.length - 2, Math.max(0, Math.floor(time)));
    if (trace[i + 1].t <= time && i + 2 < trace.length) i++;
    const a = (time - trace[i].t) / (trace[i + 1].t - trace[i].t);
    return Interpolator.interpolateShips(trace[i], trace[i + 1], a).s;
  };
  const { torch } = drive('frigate');
  const g = shipGroup('frigate', 0.02);
  const pose = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), w0: 0, throttle: 0 };
  // what the viewer computes for the plume at battle time t (wall clock fixed)
  const look = (time) => {
    const st = frameAt(time);
    const q = hullQuat(st.forward, st.up);
    g.quaternion.set(q[0], q[1], q[2], q[3]);
    g.position.set(...st.position.map(x => x / 1000));
    g.userData.plumeThrottle = st.throttle;
    torch.update(st.throttle, 42, 0, time, plumeIgnition(st.throttleFrom, st.throttle, st.throttleAge));
    const p = torch.profile(), j = jetPose(g, pose);
    return JSON.stringify([p, j && [j.pos.toArray(), j.quat.toArray(), j.w0, j.throttle]]);
  };
  // cut: the frame after t = 3 is dark along the whole length, the frame before at full length
  assert.equal(JSON.parse(look(2.99))[0].len > 0, true);
  assert.equal(JSON.parse(look(3.01))[0].len, 0);
  assert.equal(JSON.parse(look(3.01))[1], null);
  // relight at 6: full length on the first frame after it, with a light-off flash that dies away
  const lit = JSON.parse(look(6.02))[0], later = JSON.parse(look(6.6))[0];
  assert.equal(JSON.parse(look(5.99))[0].len, 0);
  assert.ok(Math.abs(lit.len - later.len) < 1e-9, 'whole length at once');
  assert.ok(lit.flare > later.flare * 1.5, 'light-off flash at the throat');
  // scrub determinism: forward at 60 fps, then the same instants after seeking around
  const times = [0.5, 2.25, 2.99, 3.01, 5.5, 6.02, 6.6, 8.9];
  const first = new Map();
  for (let t = 0; t <= 9; t += 1 / 60) {
    const v = look(t);
    for (const s of times) if (Math.abs(t - s) < 1 / 120) first.set(s, look(s)), void v;
  }
  for (const s of [...times].reverse()) { look(9 - s); assert.ok(first.has(s), `sampled ${s}`); assert.equal(look(s), first.get(s), `t = ${s}`); }
  torch.dispose();
});
