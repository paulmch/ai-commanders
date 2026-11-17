import * as THREE from 'three';
import { impactPulse, PULLBACK_PRE_S, PULLBACK_POST_S } from './Director.js';

// ------------------------------------------------------- director framing
// Camera poses for the director's shot vocabulary. Every pose is a pure
// function of (shot, battle time, where the ships are at that time): no
// easing state, no wall clock. A cut is a snap; inside a shot the camera
// only moves through terms in (t - shot.t0) and with the ships it frames,
// so scrubbing to t and playing to t give the same frame.
//
// The battle lies near the scene's XY plane and the camera's up is +Y, so
// every shot keeps the camera on the +Z side of the plane: screen
// direction stays consistent across cuts (no jumping the line).

const FLEET_ELEVATION = THREE.MathUtils.degToRad(25);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);
const WIDE_SIDE = new THREE.Vector3(0, 0.3, 1).normalize();
/** Ages (s after the detonation) between which a pull-back has to contain the blast's transient ring. */
const RING_FROM_S = 1.0;
const RING_TO_S = 1.6;
/** The pull-out starts this long before the detonation (replay knows it is coming). */
const PULL_LEAD_S = 0.6;

const clamp01 = u => Math.min(1, Math.max(0, u));
const ease = u => { const c = clamp01(u); return c * c * (3 - 2 * c); };
const easeOut = u => 1 - (1 - clamp01(u)) ** 3;
const lerp = (a, b, u) => a + (b - a) * u;

/**
 * Camera framing for a set of points (scene units): orbit target at their
 * centre, viewed from 25 degrees up and perpendicular to the axis between
 * the two groups (both fleets side by side), far enough that the bounds
 * fill widthShare of the screen width.
 * @param {THREE.Vector3[]} points
 * @param {Object} opts - {fov, aspect, groupA, groupB (centroids or null), fallbackDir, widthShare, minRadius}
 * @returns {{target: THREE.Vector3, position: THREE.Vector3, radius: number}|null}
 */
export function frameGroups(points, { fov = 55, aspect = 16 / 9, groupA = null, groupB = null,
  fallbackDir = null, widthShare = 0.62, minRadius = 25, shift = 0 } = {}) {
  if (!points.length) return null;
  const bounds = new THREE.Box3();
  for (const p of points) bounds.expandByPoint(p);
  const target = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(minRadius, bounds.getSize(new THREE.Vector3()).length() / 2);
  const halfFov = THREE.MathUtils.degToRad(fov) / 2;
  const horizontal = Math.atan(Math.tan(halfFov) * aspect * widthShare);
  const distance = radius / Math.sin(Math.min(halfFov, horizontal)) * 1.05;
  let side = null;
  if (groupA && groupB) {
    const axis = groupB.clone().sub(groupA);
    axis.y = 0;
    if (axis.lengthSq() > 1e-6) side = new THREE.Vector3(-axis.z, 0, axis.x).normalize();
  }
  if (!side) {
    side = (fallbackDir ? fallbackDir.clone() : new THREE.Vector3(0, 0, 1));
    side.y = 0;
    if (side.lengthSq() < 1e-9) side.set(0, 0, 1);
    side.normalize();
  }
  // Keep the side the camera is already on (no flips across the battle)
  if (fallbackDir && side.dot(fallbackDir) < 0) side.negate();
  const dir = side.multiplyScalar(Math.cos(FLEET_ELEVATION)).add(new THREE.Vector3(0, Math.sin(FLEET_ELEVATION), 0));
  const position = target.clone().addScaledVector(dir, distance);
  return slideToFreeArea({ target, position, radius }, shift, fov, aspect);
}

/**
 * Centre a framing in the area the side panels leave free: slide camera
 * and target sideways so the target appears at NDC x = shift.
 */
export function slideToFreeArea(frame, shift, fov = 55, aspect = 16 / 9) {
  if (!shift) return frame;
  const view = frame.target.clone().sub(frame.position);
  const distance = view.length();
  const right = new THREE.Vector3().crossVectors(view, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-12) return frame;
  const halfWidth = distance * Math.tan(THREE.MathUtils.degToRad(fov) / 2) * aspect;
  const slide = right.normalize().multiplyScalar(-shift * halfWidth);
  frame.target.add(slide);
  frame.position.add(slide);
  return frame;
}

/** Distance at which a sphere of this radius fits the view (with margin). */
export function fitDistance(radius, fov = 55, aspect = 16 / 9, widthShare = 1, margin = 1.2) {
  const halfFov = THREE.MathUtils.degToRad(fov) / 2;
  const horizontal = Math.atan(Math.tan(halfFov) * aspect * widthShare);
  return radius * margin / Math.sin(Math.min(halfFov, horizontal));
}

/**
 * Things the camera must stay out of at this instant: every live hull (a
 * sphere) and its drive plume (a capsule trailing the stern, as long as
 * the throttle makes it).
 * @returns {Array<{a: THREE.Vector3, b: THREE.Vector3, r: number}>}
 */
export function obstaclesOf(world, t) {
  const out = [];
  for (const id of world.ids) {
    if (!(world.deathTime(id) > t)) continue;
    const p = world.pos(id);
    if (!p) continue;
    const L = world.length(id);
    out.push({ a: p, b: p, r: 0.62 * L });
    const f = world.fwd(id);
    const thrust = world.thrust(id);
    if (f && thrust > 0.02) {
      const stern = p.clone().addScaledVector(f, -0.5 * L);
      out.push({ a: stern, b: stern.clone().addScaledVector(f, -2.6 * L * (0.14 + 0.86 * thrust)), r: 0.55 * L });
    }
  }
  return out;
}

/**
 * Push a camera position out of any hull or plume it sits in (radially,
 * to just outside). Returns the same vector, moved.
 */
export function clearObstacles(position, obstacles, margin = 1.08) {
  const ab = new THREE.Vector3();
  const closest = new THREE.Vector3();
  const out = new THREE.Vector3();
  for (let pass = 0; pass < 2; pass++) {
    for (const o of obstacles) {
      ab.subVectors(o.b, o.a);
      const len2 = ab.lengthSq();
      const u = len2 > 1e-12 ? clamp01(out.subVectors(position, o.a).dot(ab) / len2) : 0;
      closest.copy(o.a).addScaledVector(ab, u);
      out.subVectors(position, closest);
      const d = out.length();
      if (d >= o.r) continue;
      // Dead centre on the axis: leave toward +Z, off the battle plane
      if (d < 1e-6) out.copy(Z);
      else out.divideScalar(d);
      position.copy(closest).addScaledVector(out, o.r * margin);
    }
  }
  return position;
}

function inPlaneUp(d) {
  // Perpendicular to the line of sight inside the battle plane, pointing
  // to the +Y half (screen up)
  const up = new THREE.Vector3().crossVectors(Z, d);
  if (up.lengthSq() < 1e-9) up.copy(Y);
  up.normalize();
  if (up.y < 0) up.negate();
  return up;
}

function centroid(points) {
  const c = new THREE.Vector3();
  for (const p of points) c.add(p);
  return points.length ? c.divideScalar(points.length) : null;
}

function directionTo(from, to, fallback) {
  const d = to && from ? to.clone().sub(from) : null;
  if (!d || d.lengthSq() < 1e-9) return fallback.clone();
  return d.normalize();
}

/**
 * A firing line for the camera: the line from -> to, fading to a fixed
 * direction as the two ships get close (inside ~25 hull lengths the line
 * swings faster than a camera should; at a close pass it would whip round).
 */
function firingLine(from, to, fixed, L) {
  if (!from || !to) return fixed.clone();
  const d = to.clone().sub(from);
  const range = d.length();
  if (range < 1e-6) return fixed.clone();
  const w = ease((range / L - 10) / 15);
  d.divideScalar(range).multiplyScalar(w).addScaledVector(fixed, 1 - w);
  return d.lengthSq() < 1e-6 ? fixed.clone() : d.normalize();
}

/** Centre of the other fleet: the ships that outlive the shot (a fixed cast, so it never jumps). */
function enemyCentre(world, id, shot, t) {
  const mine = world.faction(id);
  const until = Number.isFinite(shot.t1) ? shot.t1 : Math.max(t, shot.t0);
  const others = world.ids.filter(o => world.faction(o) !== mine);
  const cast = others.filter(o => world.deathTime(o) > until);
  return centroid((cast.length ? cast : others).map(o => world.pos(o)).filter(Boolean));
}

const scaled = (v, s) => new THREE.Vector3(v[0] * s, v[1] * s, v[2] * s);

/** Wide establishing shot: the fleets alive at the shot's start (or its cast). */
function wide(shot, t, world) {
  // The cast is fixed for the shot: ships that are still alive when it
  // ends (one dying half way would make the bounds jump)
  const until = Number.isFinite(shot.t1) ? shot.t1 : Math.max(t, shot.t0);
  let cast = shot.cast?.length ? shot.cast : world.ids.filter(id => world.deathTime(id) > until);
  if (!cast.length) cast = world.ids.filter(id => world.deathTime(id) > shot.t0);
  if (!cast.length) cast = world.ids;
  const points = [];
  // (floors tuned on the kit-size hulls, in the scale mode's units)
  const unit = world.unit ?? 1;
  let longest = 8 * unit;
  for (const id of cast) {
    const p = world.pos(id);
    if (!p) continue;
    points.push(p);
    longest = Math.max(longest, world.length(id));
  }
  // Always from the +Z side (no fleet axis: it spins as fleets pass each
  // other, and the camera would swing with it)
  const frame = frameGroups(points, {
    fov: world.fov, aspect: world.aspect, fallbackDir: WIDE_SIDE, widthShare: world.widthShare,
    minRadius: shot.cast?.length ? 3 * longest : 25 * unit
  });
  if (!frame) return null;
  // A slow push-in over the first seconds of the shot
  const k = 1.1 - 0.1 * ease((t - shot.t0) / 8);
  frame.position.sub(frame.target).multiplyScalar(k).add(frame.target);
  return frame;
}

/** Over the shoulder of a ship along its firing line (salvo, first fire, a gun kill). */
function shoulder(shot, t, world) {
  const S = world.pos(shot.subject);
  if (!S) return null;
  const L = world.length(shot.subject);
  const aim = (shot.target && world.pos(shot.target)) || enemyCentre(world, shot.subject, shot, t);
  const d = firingLine(S, aim, new THREE.Vector3(world.faction(shot.subject) === 'alpha' ? 1 : -1, 0, 0), L);
  const up = inPlaneUp(d);
  const k = (1.14 - 0.14 * ease((t - shot.t0) / 5)) * L;
  return {
    position: S.clone().addScaledVector(d, -1.7 * k).addScaledVector(Z, 0.95 * k).addScaledVector(up, 0.4 * k),
    target: S.clone().addScaledVector(d, 1.6 * L).addScaledVector(up, 0.15 * L)
  };
}

/**
 * The run-in: riding in on the target along the attack line, closing as
 * the torpedoes do, with a small push on every impact.
 */
function runin(shot, t, world) {
  const V = world.pos(shot.subject);
  if (!V) return null;
  const L = world.length(shot.subject);
  const from = (shot.attacker && world.pos(shot.attacker)) || enemyCentre(world, shot.subject, shot, t);
  const a = firingLine(from, V, new THREE.Vector3(world.faction(shot.subject) === 'alpha' ? -1 : 1, 0, 0), L);
  const up = inPlaneUp(a);
  const span = Math.max(0.5, shot.t - shot.t0);
  let D = lerp(4.8, 2.6, ease((t - shot.t0) / span)) * L;
  D *= 1 - 0.08 * impactPulse(shot.impacts, t);
  const dir = a.clone().multiplyScalar(-0.82).addScaledVector(Z, 0.5).addScaledVector(up, 0.28).normalize();
  return { position: V.clone().addScaledVector(dir, D), target: V };
}

/** Slow orbit of a dying hulk (or a ship whose magazine just burned). */
function orbit(shot, t, world) {
  const C = world.pos(shot.subject);
  if (!C) return null;
  const L = world.length(shot.subject);
  const close = shot.kind !== 'dying';
  const r = (close ? 3.3 : 4.2) * L * (1 - 0.07 * impactPulse(shot.impacts, t));
  const az = THREE.MathUtils.degToRad(20 + 7 * (t - shot.t0));
  const el = THREE.MathUtils.degToRad(14);
  const offset = new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
  return { position: C.clone().addScaledVector(offset, r), target: C };
}

/**
 * Pull-back on a detonation: close on the ship a second before it blows,
 * then back out (eased) to the distance that fits the blast radius - the
 * whole chain's, when several ships go together.
 */
function pullback(shot, t, world) {
  const kills = shot.kills?.length ? shot.kills : [{ id: shot.subject, t: shot.t, radiusKm: 15, position: null }];
  const first = kills[0];
  // Where the blast is: the wreck's position when it blew, coasting on
  // with it (the shell, fireball and debris do the same)
  const at = k => {
    if (!k.position) return world.pos(k.id);
    const p = scaled(k.position, world.scale);
    if (k.velocity) p.addScaledVector(scaled(k.velocity, world.scale), Math.max(0, t - (k.posT ?? k.t)));
    return p;
  };
  const firstAt = at(first);
  if (!firstAt) return null;
  const L = world.length(first.id);
  // What the camera has to fit, asked of the effect rather than guessed
  // (world.blastExtent -> Explosion.blastExtent): never less than the
  // recorded blast radius (the lethal shell). Two answers: the transient
  // ejecta ring while it is still bright when the camera arrives, and the
  // fireball that lasts for the rest of the shot.
  const ring = k => Math.max(k.radiusKm, world.blastExtent?.(k.id, RING_FROM_S, RING_TO_S) || 0);
  const body = k => Math.max(k.radiusKm, world.blastExtent?.(k.id, RING_TO_S + 0.4, PULLBACK_POST_S) || 0);
  // Frame what has gone off so far (a chain widens as it goes)
  const framing = extent => {
    let centre = firstAt.clone();
    let reach = extent(first);
    let widen = 0;
    for (const k of kills.slice(1)) {
      const p = at(k);
      if (!p) continue;
      const u = ease((t - (k.t - PULLBACK_PRE_S)) / 1.5);
      if (u <= 0) continue;
      const mid = firstAt.clone().add(p).multiplyScalar(0.5);
      const span = firstAt.distanceTo(p) / 2 + Math.max(extent(first), extent(k));
      if (span * u > widen) { widen = span * u; centre = firstAt.clone().lerp(mid, u); reach = lerp(extent(first), span, u); }
    }
    return { centre, reach };
  };
  const lasting = framing(body);
  const centre = lasting.centre;
  // Until it blows the ship is still under way: stay on the hull itself
  const live = world.pos(first.id);
  if (live && t < first.t) centre.lerp(live, 1 - ease((t - (first.t - PULLBACK_PRE_S)) / PULLBACK_PRE_S));
  // Where the shot starts. On the cinematic scale a hull is about as long
  // as its blast is wide, so one distance is both a close-up and outside
  // the shell. At real size the hull is a hundredth of the blast: start
  // on the hull (the hulk reads, molten and tumbling) and let the move
  // out, which begins before the flash, carry the camera clear of it.
  const near = world.realScale ? 2.4 * L : Math.max(2.4 * L, 1.3 * first.radiusKm);
  const fit = reach => Math.max(1.35 * near, fitDistance(reach, world.fov, world.aspect, world.widthShare, 1.15));
  // Out with the blast: the move starts from rest just before the flash
  // and has the ring in frame while it is still bright, then eases back
  // in on the fireball as the ring dies. The shock front of the first
  // half second outruns any camera: it sweeps past.
  const age = t - first.t;
  const out = ease((age + PULL_LEAD_S) / (PULL_LEAD_S + RING_FROM_S + 0.3));
  const settle = ease((age - RING_TO_S) / 2.5);
  const D = lerp(lerp(near, fit(framing(ring).reach), out), fit(lasting.reach), settle);
  // From the killer's side of the line of fire, off the battle plane
  const killer = first.killer ? world.pos(first.killer) : null;
  const k = directionTo(killer, firstAt, new THREE.Vector3(0, 0, 0));
  k.z = 0;
  const dir = new THREE.Vector3(0, 0.3, 0.8).addScaledVector(k, -0.5).normalize();
  dir.applyAxisAngle(Y, THREE.MathUtils.degToRad(2.5 * (t - shot.t0)));
  return { position: centre.clone().addScaledVector(dir, D), target: centre };
}

/** Chase: behind and beside a ship, looking down its heading. */
function chase(shot, t, world) {
  const P = world.pos(shot.subject);
  if (!P) return null;
  const L = world.length(shot.subject);
  const f = world.fwd(shot.subject) || new THREE.Vector3(1, 0, 0);
  const up = inPlaneUp(f);
  const k = (1.1 - 0.1 * ease((t - shot.t0) / 6)) * L;
  return {
    position: P.clone().addScaledVector(f, -1.9 * k).addScaledVector(Z, 1.3 * k).addScaledVector(up, 0.4 * k),
    target: P.clone().addScaledVector(f, 1.3 * L)
  };
}

/** Flyby: the camera slides from ahead of the bow to the stern quarter. */
function flyby(shot, t, world) {
  const P = world.pos(shot.subject);
  if (!P) return null;
  const L = world.length(shot.subject);
  const f = world.fwd(shot.subject) || new THREE.Vector3(1, 0, 0);
  const up = inPlaneUp(f);
  const span = Number.isFinite(shot.t1) ? shot.t1 - shot.t0 : 7;
  const u = clamp01((t - shot.t0) / span);
  const lead = Math.min(2.8, 0.4 * span);   // a short shot is a short pass, not a faster one
  return {
    position: P.clone().addScaledVector(f, lerp(lead, -lead / 2, u) * L).addScaledVector(Z, 1.5 * L).addScaledVector(up, -0.25 * L),
    target: P.clone().addScaledVector(f, lerp(0.5, -0.1, u) * L)
  };
}

const FRAMERS = { wide, shoulder, runin, orbit, pullback, chase, flyby };

/**
 * Camera pose for a shot at battle time t.
 * @param {Object} shot - from Director.buildShotList
 * @param {number} t - battle seconds
 * @param {Object} world - {ids, pos(id), fwd(id), thrust(id), length(id),
 *   deathTime(id), faction(id), fov, aspect, widthShare, shift, scale,
 *   unit, realScale}
 *   (positions in scene units; scale = scene units per recording metre;
 *   length(id) is the hull's FRAMING length - its real length in the auto
 *   and true scale modes, the kit length in cinematic - `unit` the scene
 *   km per kit unit for the kit-tuned floors and `realScale` whether hulls
 *   are drawn at real size up close: render/Scale.js)
 * @returns {{position: THREE.Vector3, target: THREE.Vector3}|null} null
 *   when the shot's subject cannot be placed (the caller falls back wide)
 */
export function shotPose(shot, t, world) {
  const frame = FRAMERS[shot?.type]?.(shot, t, world);
  if (!frame || !Number.isFinite(frame.position.x) || !Number.isFinite(frame.target.x)) return null;
  if (shot.type !== 'wide') clearObstacles(frame.position, obstaclesOf(world, t));
  return slideToFreeArea(frame, world.shift || 0, world.fov, world.aspect);
}
