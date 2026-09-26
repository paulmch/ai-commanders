import * as THREE from 'three';
import { Ribbon } from './Trails.js';
import { WeaponRibbons, WeaponFlashes, TorpedoBodies } from './WeaponBatches.js';
import {
  WeaponTimeline, makeSample, sampleTrack, seedFor, tracerLength,
  missOpacity, beamPulse, torpedoPower, ION_LIFE, IMPACT_LIFE, OWNED_MOMENTS
} from './WeaponTimeline.js';

const Z = new THREE.Vector3(0, 0, 1);
const Y = new THREE.Vector3(0, 1, 0);
const ORIGIN = new THREE.Vector3();
const SCALE = 0.001;
// Spinal muzzle (fraction of hull width below the axis, of length forward)
const SPINAL = { cruiser: [-0.42, 0.5], battlecruiser: [-0.245, 0.56], battleship: [-0.335, 0.5], dreadnought_siege: [-0.2, 0.8] };
const MUZZLE_LIFE = 0.3;
const random = (seed, n) => { const x = Math.sin(seed * 127.1 + n * 311.7) * 43758.5453; return x - Math.floor(x); };

/**
 * In-flight weapons and the first 460 ms of an impact. All state is evaluated
 * from the recording at the requested time. Persistent hull damage stays in
 * SceneManager/the damage renderer. Three shared draws, no per-round lights.
 */
export class Projectiles {
  constructor(manager) {
    this.manager = manager;
    this.ribbons = new WeaponRibbons(manager.scene);
    this.flashes = new WeaponFlashes(manager.scene);
    this.bodies = new TorpedoBodies(manager.scene);
    this.a = new THREE.Vector3(); this.b = new THREE.Vector3();
    this.c = new THREE.Vector3(); this.d = new THREE.Vector3();
    this.dir = new THREE.Vector3(); this.vel = new THREE.Vector3();
    this.spark = new THREE.Vector3(); this.sparkEnd = new THREE.Vector3();
    this.axis = new THREE.Vector3(); this.q = new THREE.Quaternion();
    this.e = new THREE.Vector3(); this.u = new THREE.Vector3(); this.liftSample = makeSample();
    this.m4 = new THREE.Matrix4(); this.q2 = new THREE.Quaternion(); this.aim = new THREE.Vector3();
    this.s1 = new THREE.Vector3(); this.s2 = new THREE.Vector3(); this.s3 = new THREE.Vector3();
    this.sample = makeSample(); this.history = makeSample(); this.shipSample = makeSample(); this.probe = makeSample();
    this.timeline = null;
    this.time = 0;
  }

  prepare(loader) {
    if (this.trace === loader.simTrace && this.traceCount === loader.simTrace.length
      && this.events === loader.events && this.eventCount === loader.events.length) return;
    // Also rebuild on incremental live append. Geometry/buffers stay pooled.
    this.trace = loader.simTrace; this.traceCount = loader.simTrace.length;
    this.events = loader.events; this.eventCount = loader.events.length;
    this.timeline = new WeaponTimeline(loader);
    this.slugs = Array.from(this.timeline.slugs.values());
    this.torpedoes = Array.from(this.timeline.torpedoes.values());
    for (const tr of this.slugs) tr.position = new THREE.Vector3();
    for (const tr of this.torpedoes) {
      tr.position = new THREE.Vector3();
      const src = tr.samples[0]?.state.source || '';
      tr.source = src; tr.lift = true;
      const faction = this.manager.ships.get(src)?.userData.faction || src.split('_')[0];
      tr.marker = faction === 'alpha' ? 3 : 4;
    }
    this.beams = (loader.pdBeamSegments || []).map(beam => ({ ...beam,
      visualSeed: seedFor(beam.key), mountSeed: seedFor(`${beam.shipId}|${beam.turret}`)
    }));
  }

  update(loader, time) {
    this.prepare(loader);
    this.time = time;
    this.ribbons.begin(time); this.flashes.begin(time); this.bodies.begin();
    for (let i = 0; i < this.slugs.length; i++) this.slug(this.slugs[i], time);
    for (let i = 0; i < this.torpedoes.length; i++) this.torpedo(this.torpedoes[i], time);
    for (let i = 0; i < this.beams.length; i++) this.beam(this.beams[i], time);
    const moments = this.timeline.moments;
    // Binary-search the short transient window; a paused/direct-seek frame
    // gets exactly the same instances as continuous playback.
    let lo = 0, hi = moments.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (moments[mid].time < time - 4) lo = mid + 1; else hi = mid;
    }
    for (let i = lo; i < moments.length && moments[i].time <= time; i++) this.moment(moments[i], time);
    this.ribbons.end(); this.flashes.end(); this.bodies.end();
  }

  pixelAt(p) { return this.manager.camera.position.distanceTo(p) * Ribbon.pixelWorld.value; }
  point(out, sample) { return out.fromArray(sample.pos).multiplyScalar(SCALE); }
  direction(sample) {
    this.dir.fromArray(sample.vel);
    if (this.dir.lengthSq() < 1e-10) this.dir.set(0, 0, 1); else this.dir.normalize();
  }

  slug(tr, time) {
    tr.visible = time >= tr.start && time < tr.end;
    if (time < tr.start || time > tr.end + ION_LIFE) return;
    const sampleTime = Math.min(time, tr.end);
    this.timeline.sample(tr, sampleTime, this.sample);
    this.point(this.a, this.sample);
    this.muzzleLift(tr, sampleTime, this.a);
    tr.position.copy(this.a);
    this.direction(this.sample);
    this.vel.fromArray(this.sample.vel).multiplyScalar(SCALE);
    const speed = this.vel.length();
    const fade = missOpacity(time - tr.missAt);
    if (tr.visible) {
      const len = tracerLength(speed, this.pixelAt(this.a), time - tr.start);
      this.b.copy(this.a).addScaledVector(this.dir, -len);
      this.ribbons.add(this.a, this.b, 0.045, fade, 0, tr.seed);
      this.flashes.add(this.a, this.dir, Math.max(0.085, this.pixelAt(this.a) * 1.5), 0.7 * fade);
    }
    this.trail(tr, time, sampleTime, ION_LIFE, 12, fade, false);
  }

  /**
   * Viewer hulls are drawn ~8x oversize, so a fresh torpedo spends its first
   * seconds inside its own launcher's mesh. Push it out along its bearing
   * from the launcher by the displayed hull radius, fading the push to zero
   * by twice that range (monotonic, so the round never appears to stall).
   */
  lift(tr, t, p) {
    if (t - tr.start > 40) return p;
    const ship = this.manager.ships.get(tr.source);
    const track = this.timeline.ships.get(tr.source);
    const size = ship?.userData.size;
    if (!size || !track || !sampleTrack(track, t, this.liftSample)) return p;
    this.e.fromArray(this.liftSample.pos).multiplyScalar(SCALE);
    this.u.subVectors(p, this.e);
    const dist = this.u.length();
    this.axis.fromArray(this.liftSample.fwd);
    if (this.axis.lengthSq() < 1e-10) this.axis.copy(Z); else this.axis.normalize();
    if (dist > 1e-4) this.u.divideScalar(dist); else this.u.copy(this.axis);
    const c = this.u.dot(this.axis);
    const R = 1 / Math.sqrt((c / (size.length * 0.5)) ** 2 + (1 - c * c) / (size.width * 0.5) ** 2) + 0.15;
    if (dist >= 2 * R) return p;
    return p.copy(this.e).addScaledVector(this.u, dist + R * (1 - THREE.MathUtils.smoothstep(dist, 0, 2 * R)));
  }

  /**
   * Slugs are recorded leaving the (physical) ship centre, which sits deep
   * inside the ~8x oversize viewer hull. Offset a fresh round so it leaves
   * the firing gun's muzzle, blending the offset out by four displayed hull
   * half-lengths of flight. Pure function of battle time.
   */
  muzzleLift(tr, t, p) {
    if (!tr.shot || t - tr.start > 30) return p;
    const ship = this.manager.ships.get(tr.source);
    const track = this.timeline.ships.get(tr.source);
    const size = ship?.userData.size;
    if (!size || !track || !sampleTrack(track, t, this.liftSample)) return p;
    const mz = tr.muzzle || (tr.muzzle = this.slugMuzzle(tr, ship, track));
    if (!mz) return p;
    this.e.fromArray(this.liftSample.pos).multiplyScalar(SCALE);
    const reach = size.length * 2;
    const w = 1 - THREE.MathUtils.smoothstep(p.distanceTo(this.e), 0, reach);
    if (w <= 0) return p;
    this.hullQuaternion(this.liftSample.fwd, this.q2);
    return p.addScaledVector(this.u.copy(mz.pos).applyQuaternion(this.q2), w);
  }

  /** The recorded hull orientation (SceneManager.updateShip's lookAt). */
  hullQuaternion(fwd, out) {
    this.s1.fromArray(fwd);
    if (this.s1.lengthSq() < 1e-10) return out.identity();
    this.m4.lookAt(this.s1.normalize(), ORIGIN, Y);
    return out.setFromRotationMatrix(this.m4);
  }

  /** Hull-local muzzle point + barrel axis for a slug's firing gun (cached per round). */
  slugMuzzle(tr, ship, track) {
    const first = tr.samples[0];
    if (!first || !sampleTrack(track, tr.start, this.shipSample)) return null;
    // Barrel axis: the round's launch velocity relative to the shooter.
    this.aim.fromArray(first.state.vel || [0, 0, 1]).sub(this.s2.fromArray(this.shipSample.vel));
    this.hullQuaternion(this.shipSample.fwd, this.q2).invert();
    this.aim.applyQuaternion(this.q2);
    if (this.aim.lengthSq() < 1e-10) this.aim.copy(Z); else this.aim.normalize();
    const out = { pos: new THREE.Vector3(), dir: new THREE.Vector3() };
    return this.localMuzzle(ship, tr.shot, this.aim, out);
  }

  /**
   * Hull-local muzzle of a gun (turret rig by weapon slot, else the spinal
   * mount) aimed along hull-local `aim`: the turret's barrel-tip centre for
   * that yaw/pitch, from the rig's fixed mount (never its animated state).
   */
  localMuzzle(ship, shot, aim, out) {
    const ud = ship.userData;
    const W = ud.size.width, L = ud.size.length;
    const rigs = ud.gunTurrets?.length ? ud.gunTurrets : ud.turrets;
    const rig = !shot.spinal && rigs?.length
      ? rigs.find(r => r.slot && r.slot === shot.slot) || rigs[shot.index % rigs.length] : null;
    if (rig?.mount) {
      const s = rig.s || 0.5, rz = rig.mount.rotation.z;
      // aim in the mount frame, clamped to the rig's elevation limits
      const v = this.s3.copy(aim).applyAxisAngle(Z, -rz);
      const yaw = Math.atan2(v.x, v.z);
      const pitch = THREE.MathUtils.clamp(Math.atan2(v.y, Math.hypot(v.x, v.z)), -0.04, 1.45);
      out.dir.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
      // mount > yaw (0, .2s, 0) > pitch (0, .32s, .15s) > barrel tip 1.4s
      out.pos.set(0.15 * s * Math.sin(yaw), 0.52 * s, 0.15 * s * Math.cos(yaw)).addScaledVector(out.dir, 1.4 * s);
      out.pos.applyAxisAngle(Z, rz).add(rig.mount.position);
      out.dir.applyAxisAngle(Z, rz);
      out.yaw = yaw; out.pitch = pitch; out.rig = rig;
      return out;
    }
    const [y, z] = SPINAL[ud.shipType] || [-0.31, 0.56];
    out.pos.set(0, W * y, L * z);
    out.dir.copy(Z);
    out.rig = null;
    return out;
  }

  trail(tr, time, newest, life, segments, intensity, poweredOnly) {
    const oldest = Math.max(tr.start, time - life);
    if (newest <= oldest) return;
    this.timeline.sample(tr, newest, this.history);
    this.point(this.c, this.history);
    if (tr.lift) this.lift(tr, newest, this.c); else this.muzzleLift(tr, newest, this.c);
    let prev = newest;
    for (let i = 1; i <= segments; i++) {
      const t = newest + (oldest - newest) * i / segments;
      this.timeline.sample(tr, t, this.history);
      this.point(this.d, this.history);
      if (tr.lift) this.lift(tr, t, this.d); else this.muzzleLift(tr, t, this.d);
      const powered = !poweredOnly || (t < tr.disabledAt && !this.history.state.disabled
        && (this.history.state.dv_remaining - this.history.next.dv_remaining) > 0.002);
      if (powered) this.ribbons.add(this.c, this.d, poweredOnly ? 0.12 : 0.03,
        intensity * (poweredOnly ? 0.7 : 0.45), poweredOnly ? 6 : 1, tr.seed, (time - prev) / life, (time - t) / life);
      this.c.copy(this.d);
      prev = t;
    }
  }

  torpedo(tr, time) {
    tr.visible = time >= tr.start && time < tr.end;
    if (time < tr.start || time > tr.end + 1.3) return;
    this.timeline.sample(tr, Math.min(time, tr.end), this.sample);
    this.point(this.a, this.sample);
    this.lift(tr, Math.min(time, tr.end), this.a);
    tr.position.copy(this.a);
    this.direction(this.sample);
    const s = this.sample.state;
    const disabled = time >= tr.disabledAt || s.disabled;
    if (tr.visible) {
      let range = Infinity, closure = 0;
      const target = this.timeline.ships.get(s.target);
      if (target) {
        sampleTrack(target, time, this.shipSample);
        this.b.fromArray(this.shipSample.pos).sub(this.c.fromArray(this.sample.pos));
        range = this.b.length();
        this.vel.fromArray(this.sample.vel).sub(this.c.fromArray(this.shipSample.vel));
        closure = range > 0 ? this.vel.dot(this.b) / range : 0;
      }
      const power = torpedoPower(s, this.sample.next, this.sample.dt, range, closure, disabled);
      // Absolute orientation, including seeker-disabled roll; never integrate dt.
      this.bodies.add(this.a, this.dir, disabled ? (time - Math.min(time, tr.disabledAt)) * 1.1 + tr.seed * 6.28 : tr.seed * 6.28);
      if (power > 0) {
        const terminal = power > 0.9;
        const breathe = 0.96 + 0.04 * Math.sin(time * 11 + tr.seed * 41);
        this.b.copy(this.a).addScaledVector(this.dir, -0.8);
        this.c.copy(this.b).addScaledVector(this.dir, -(terminal ? 9 : 3.6) * breathe);
        this.ribbons.add(this.b, this.c, terminal ? 0.5 : 0.24, terminal ? 1.15 : 0.8, 3, tr.seed);
        // Nozzle glow: reads end-on and at ranges where the plume is a line.
        this.flashes.add(this.b, this.dir, terminal ? 0.6 : 0.28, terminal ? 1.2 : 0.6, 1);
      }
      // Small seeker glint keeps an unpowered body readable at mid range.
      this.b.copy(this.a).addScaledVector(this.dir, 0.97);
      this.flashes.add(this.b, this.dir, 0.045, disabled ? 0.06 : 0.18, 1);
      // Faction diamond at fleet range, gone once the body itself resolves.
      const bodyPx = 1.8 / this.pixelAt(this.a);
      const mark = 0.5 * (1 - THREE.MathUtils.smoothstep(bodyPx, 8, 24)) * (disabled ? 0.35 : 1);
      if (mark > 0) this.flashes.add(this.a, this.dir, 0, mark, tr.marker);
    }
    this.trail(tr, time, Math.min(time, tr.end), 1.3, 14, 1, true);
  }

  /** Loader torpedo moments rendered here instead of SceneManager flares. */
  ownsMoment(kind) { return OWNED_MOMENTS.has(kind); }

  /**
   * A round's displayed position at battle time `time`, straight from the
   * recording (no dependence on the last rendered frame): undefined when
   * this is not a round or nothing is loaded, null when not in flight.
   */
  positionAt(type, id, time, out) {
    if (!this.timeline || (type !== 'torpedo' && type !== 'slug' && type !== 'projectile')) return undefined;
    const tr = type === 'torpedo' ? this.timeline.torpedoes.get(id) : this.timeline.slugs.get(id);
    if (!tr || time < tr.start || time >= tr.end) return null;
    this.timeline.sample(tr, time, this.probe);
    this.point(out, this.probe);
    return tr.lift ? this.lift(tr, time, out) : this.muzzleLift(tr, time, out);
  }

  position(type, id) {
    if (type !== 'torpedo' && type !== 'slug' && type !== 'projectile') return null;
    const tr = type === 'torpedo' ? this.timeline?.torpedoes.get(id) : this.timeline?.slugs.get(id);
    return tr?.visible ? tr.position : null;
  }

  beam(seg, time) {
    const intensity = beamPulse(time, seg.start, seg.end, seg.visualSeed);
    if (intensity <= 0) return;
    const ship = this.manager.ships.get(seg.shipId);
    const target = this.position(seg.targetType, seg.targetId);
    if (!ship?.visible || !target) return;
    const W = ship.userData.size?.width || 2, L = ship.userData.size?.length || 8;
    const seed = seg.visualSeed;
    const torpedo = seg.targetType === 'torpedo';
    const tr = torpedo ? this.timeline.torpedoes.get(seg.targetId) : this.timeline.slugs.get(seg.targetId);
    const mount = this.pdMount(seg, ship);
    if (mount) {
      // From the firing PD ball's lens (the ball itself slews onto the round).
      this.a.copy(mount.pos).applyQuaternion(ship.quaternion).add(ship.position);
      this.dir.subVectors(target, this.a).normalize();
      this.a.addScaledVector(this.dir, mount.reach);
    } else {
      this.a.set((random(seg.mountSeed, 1) - 0.5) * W * 0.85, (random(seg.mountSeed, 2) - 0.5) * W * 0.85, (random(seg.mountSeed, 3) - 0.5) * L * 0.6);
      this.a.applyQuaternion(ship.quaternion).add(ship.position);
    }
    this.dir.subVectors(target, this.a).normalize();
    // Burn spot on the facing surface of the round, not hidden at its centre.
    this.surfaceOffset(tr, torpedo, time, this.d);
    this.b.copy(target).add(this.d);
    this.ribbons.add(this.a, this.b, 0.075, 0.85 * intensity, 2, seed);
    this.flashes.add(this.a, this.dir, 0.16, 0.38 * intensity);
    this.flashes.add(this.b, this.dir, 0.2, 0.75 * intensity, 2, seed);
    // Fixed birth cadence, sampled analytically; direct seeks retain spray.
    const cadence = 0.09, tick = Math.floor((time - seg.start) / cadence);
    this.dir.negate();
    for (let j = 0; j < 3; j++) {
      if (tick - j < 0) break;
      const age = time - (seg.start + (tick - j) * cadence);
      this.timeline.sample(tr, time - age, this.history);
      this.point(this.c, this.history).add(this.d);
      this.vel.fromArray(this.history.vel).multiplyScalar(SCALE);
      this.sparks(this.c, this.dir, this.vel, age, 0.28, 0.4, seed + (tick - j) * 0.013, 4, intensity);
    }
  }

  /**
   * Hull-local PD ball centre + lens reach for a dwell segment's turret
   * (hull `pdMounts` by name, else by turret index), or null for hulls
   * without PD mounts (then a seeded hull point is used). Cached per segment.
   */
  pdMount(seg, ship) {
    if (seg.mountShip === ship) return seg.mount;
    const ud = ship.userData;
    const list = ud.pdMounts;
    let mount = null;
    if (list?.length) {
      let i = list.findIndex(p => p.name === seg.turret);
      if (i < 0) {
        const n = Number(String(seg.turret ?? '').split('_').at(-1));
        if (Number.isFinite(n) && n >= 0) i = n % list.length;
      }
      if (i >= 0) {
        const rig = ud.pdTurrets?.find(r => r.name === list[i].name);
        mount = { pos: list[i].pos, reach: 0.72 * (rig?.s ?? 0) };
      }
    }
    seg.mount = mount;
    seg.mountShip = ship;
    return mount;
  }

  /** Offset from a round's track position to where a beam along this.dir
   * first meets the displayed body (torpedo ~ellipsoid 0.9 x 0.16 km). */
  surfaceOffset(tr, torpedo, time, out) {
    if (!torpedo) return out.copy(this.dir).multiplyScalar(-0.04);
    this.timeline.sample(tr, time, this.history);
    this.axis.fromArray(this.history.vel);
    if (this.axis.lengthSq() < 1e-10) this.axis.copy(Z); else this.axis.normalize();
    const c = this.axis.dot(this.dir);
    const r = 1 / Math.sqrt((c / 0.92) ** 2 + (1 - c * c) / 0.17 ** 2);
    return out.copy(this.axis).multiplyScalar(0.11).addScaledVector(this.dir, -r);
  }

  muzzle(m, age) {
    if (age > MUZZLE_LIFE) return;
    const ship = this.manager.ships.get(m.ship);
    if (!ship?.visible) return;
    // Hull kits can supply an exact mount without coupling this module to a
    // particular hull revision. Otherwise the firing turret (by weapon slot)
    // or the spinal mount, at the same muzzle the round is lifted from.
    const mounts = ship.userData.weaponMounts;
    const mount = mounts?.[m.data.weapon_slot];
    if (mount?.getWorldPosition) {
      mount.getWorldPosition(this.a); mount.getWorldQuaternion(this.q);
      this.dir.copy(Z).applyQuaternion(this.q);
    } else {
      const mz = this.shotMuzzle(m, ship);
      this.a.copy(mz.pos).applyQuaternion(ship.quaternion).add(ship.position);
      this.dir.copy(mz.dir).applyQuaternion(ship.quaternion);
      // The firing turret holds the shot's bearing through the flash; its
      // angles come from the recording, never from the rig's own slew.
      const rig = mz.rig;
      if (rig?.yaw && rig.pitch) {
        rig.yawAngle = mz.yaw; rig.pitchAngle = mz.pitch;
        rig.yaw.rotation.y = mz.yaw; rig.pitch.rotation.x = -mz.pitch;
      }
    }
    // Coil discharge: white-hot flash, an expanding plasma ring around the
    // barrel axis and a short forward jet. Spinal mounts are the big guns.
    const k = m.spinal ? 1.4 : 1.0, f = 1 - age / MUZZLE_LIFE;
    this.flashes.add(this.a, this.dir, k * (0.28 + age * 1.2), 2.2 * f * f * f, 1, m.seed);
    this.flashes.add(this.a, this.dir, k * (0.2 + age * 4), 1.3 * f * f, 0, m.seed);
    this.b.copy(this.a).addScaledVector(this.dir, k * (0.6 + age * 5));
    this.ribbons.add(this.a, this.b, k * 0.1, 1.4 * f * f, 0, m.seed);
  }

  /** Hull-local muzzle for a shot_fired moment (cached on the moment). */
  shotMuzzle(m, ship) {
    if (m.muzzleLocal && m.muzzleShip === ship) return m.muzzleLocal;
    const tr = m.track;
    const track = this.timeline.ships.get(m.ship);
    let mz = tr?.shot && track ? (tr.muzzle || (tr.muzzle = this.slugMuzzle(tr, ship, track))) : null;
    if (!mz) {
      // No recorded round: aim at the target's recorded position instead.
      this.aim.copy(Z);
      const target = this.timeline.ships.get(m.data.target_id);
      if (target && track && sampleTrack(target, m.time, this.shipSample) && sampleTrack(track, m.time, this.liftSample)) {
        this.aim.fromArray(this.shipSample.pos).sub(this.s2.fromArray(this.liftSample.pos));
        this.hullQuaternion(this.liftSample.fwd, this.q2).invert();
        this.aim.applyQuaternion(this.q2);
        if (this.aim.lengthSq() < 1e-10) this.aim.copy(Z); else this.aim.normalize();
      }
      mz = this.localMuzzle(ship, { slot: m.data.weapon_slot, index: m.slot, spinal: m.spinal }, this.aim,
        { pos: new THREE.Vector3(), dir: new THREE.Vector3() });
    }
    m.muzzleLocal = mz;
    m.muzzleShip = ship;
    return mz;
  }

  moment(m, time) {
    const age = time - m.time;
    if (m.kind === 'retarget') {
      const target = this.manager.ships.get(m.target);
      if (m.track.visible && target?.visible) {
        const fade = Math.max(0, 1 - age / 4);
        this.ribbons.add(m.track.position, target.position, 0.025, fade * 0.5, 5, m.seed);
      }
      return;
    }
    if (age >= IMPACT_LIFE) return;
    if (m.kind === 'muzzle') { this.muzzle(m, age); return; }
    this.timeline.sample(m.track, m.time, this.sample);
    this.point(this.a, this.sample);
    if (m.track.lift) this.lift(m.track, m.time, this.a); else this.muzzleLift(m.track, m.time, this.a);
    this.direction(this.sample);
    this.vel.fromArray(this.sample.vel).multiplyScalar(SCALE);
    if (m.kind === 'hit') {
      if (m.position) this.a.fromArray(m.position).multiplyScalar(SCALE);
      const target = this.manager.ships.get(m.target);
      const targetTrack = this.timeline.ships.get(m.target);
      if (target && targetTrack) {
        sampleTrack(targetTrack, m.time, this.shipSample);
        // Approach in the target's frame: fleet drift must not swing a
        // head-on hit round to the flank (absolute velocity if ~co-moving).
        this.c.fromArray(this.sample.vel).sub(this.d.fromArray(this.shipSample.vel));
        if (this.c.lengthSq() > 1) this.dir.copy(this.c).normalize();
        this.vel.fromArray(this.shipSample.vel).multiplyScalar(SCALE);
        // Recorded hits use physical hull dimensions, while viewer hulls are
        // exaggerated. Project onto the displayed target's incoming surface.
        this.point(this.b, this.shipSample);
        this.axis.fromArray(this.shipSample.fwd).normalize();
        const axial = Math.abs(this.dir.dot(this.axis));
        const radius = 1 / Math.sqrt((axial / (target.userData.size.length * 0.5)) ** 2
          + (1 - axial * axial) / (target.userData.size.width * 0.5) ** 2);
        this.a.copy(this.b).addScaledVector(this.dir, -radius);
      }
      this.dir.negate();
    }
    // Advect the flash with the struck hull / intercepted round.
    this.a.addScaledVector(this.vel, age);
    const k = m.kind, life = 1 - age / IMPACT_LIFE;
    this.axis.set(0, 0, 0); // centre already advected; sparks get no base velocity
    if (k === 'ignition' || k === 'burnout') {
      // Motor light-off / flame-out at the nozzle: a short stuttering flare.
      if (k === 'ignition') {
        // Launch: at the (lifted) tube exit so the hull does not hide it.
        this.flashes.add(this.a, this.dir, 0.5 + age * 2, 1.6 * Math.exp(-age * 6) * life, 1, m.seed);
        this.flashes.add(this.a, this.dir, 0.3 + age * 3, 0.9 * life * life, 0, m.seed);
        return;
      }
      this.a.addScaledVector(this.dir, -0.8);
      const stutter = 0.55 + 0.45 * Math.sin(age * 70 + m.seed * 9);
      this.flashes.add(this.a, this.dir, 0.3 + age * 1.6, 1.3 * Math.exp(-age * 7) * life * stutter, 1, m.seed);
      return;
    }
    if (k === 'seeker') {
      // Seeker burned out by PD: an electrical arc ring around the body.
      this.flashes.add(this.a, this.dir, 0.35 + age * 2.2, 1.6 * life * life, 0, m.seed);
      this.flashes.add(this.a, this.dir, 0.3, 1.4 * Math.exp(-age * 12) * life, 1, m.seed);
      this.sparks(this.a, this.dir, this.axis, age, IMPACT_LIFE, 0.5, m.seed, 14, 1);
      return;
    }
    const scale = k === 'hit' ? Math.min(2.1, 0.5 + Math.pow(Math.max(1, m.energy || 1), 0.3) * 0.3) : 0.9;
    const flash = Math.exp(-age * 9) * life;
    this.flashes.add(this.a, this.dir, scale * (0.55 + age * 2.5), flash * 2.2, 2, m.seed);
    this.flashes.add(this.a, this.dir, scale * 0.3, 2.5 * Math.exp(-age * 30), 1, m.seed);
    this.sparks(this.a, this.dir, this.axis, age, IMPACT_LIFE, scale, m.seed, 28, 1);
  }

  sparks(origin, normal, velocity, age, life, scale, seed, count, intensity) {
    for (let i = 0; i < count; i++) {
      const lifetime = life * (0.55 + random(seed, i * 5) * 0.45);
      if (age >= lifetime) continue;
      const z = random(seed, i * 5 + 1) * 2 - 1, phi = random(seed, i * 5 + 2) * Math.PI * 2;
      const r = Math.sqrt(1 - z * z);
      this.spark.set(r * Math.cos(phi), r * Math.sin(phi), z).addScaledVector(normal, 1.1).normalize();
      const speed = (1.5 + random(seed, i * 5 + 3) * 8) * scale;
      this.spark.multiplyScalar(speed);
      const travel = (1 - Math.exp(-age * 2)) * 0.5;
      this.sparkEnd.copy(origin).addScaledVector(velocity, age).addScaledVector(this.spark, travel);
      this.spark.multiplyScalar(-Math.min(age, 0.035) * Math.exp(-age * 2)).add(this.sparkEnd);
      this.ribbons.add(this.sparkEnd, this.spark, 0.02 * scale, Math.pow(1 - age / lifetime, 1.5) * intensity, 4, seed, age / lifetime, age / lifetime);
    }
  }

  clear() {
    this.trace = null; this.timeline = null;
    this.events = null; this.slugs = null; this.torpedoes = null; this.beams = null;
    this.ribbons.begin(0); this.ribbons.end();
    this.flashes.begin(0); this.flashes.end(); this.bodies.begin(); this.bodies.end();
  }
  dispose() { this.clear(); this.ribbons.dispose(); this.flashes.dispose(); this.bodies.dispose(); }
}
