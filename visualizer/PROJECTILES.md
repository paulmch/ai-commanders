# Projectile rendering handoff

`render/Projectiles.js` owns in-flight weapons and instantaneous effects
(<= 0.46 battle seconds): coilgun slugs (velocity-aligned tracer, 1 s cooling
wake, 6 s fading coast for misses), torpedoes (instanced body, phase-sized
motor plume + exhaust wake, faction diamond at fleet range), continuous PD
beams (burn spot on the facing surface of the round, ablation sparks), coil
muzzle discharge, torpedo ignition/burn-out/seeker-kill/intercept flashes and
the instant hull-impact flash + spark burst. `WeaponTimeline.js` indexes the
recording; `WeaponBatches.js` holds the three shared draws (ribbons, flashes,
torpedo bodies). Everything is evaluated from battle time (seek/scrub/pause
give identical buffers; see tests), no per-frame allocation, no PointLights.

## Merge hooks

- `SceneManager.init`: `this.weapons = new Projectiles(this)` before warmUp.
- `SceneManager.getEntityPosition`: consults `this.weapons.position()` first.
- `main.js` animation: `scene.weapons.update(loader, t)` replaces the old
  per-entity projectile/torpedo/PD-beam loops and the shot muzzle-flare loop.
  The torpedo-moment loop and the visual-impact loop are KEPT: torpedo
  detonations (`spawnTorpedoDetonation`), torpedo misses and coilgun hull hits
  (`spawnHitEffect`, damage stream's aftermath) still spawn their Blasts;
  moments in `OWNED_MOMENTS` (seeker kill, intercept, burnout) are skipped via
  `scene.weapons.ownsMoment(kind)` because this module draws them.
- `main.js` reset: `scene.weapons.clear()`.

The legacy `SceneManager.updateProjectile/updateTorpedo/updateContinuousBeams`
paths were removed in the polish pass; `getEntityPosition` answers rounds from
this module only.

## Rendering notes

- Ribbons keep their bright core >= ~1.5 px wide (`vPx` in the shader), so
  thin tracers and PD beams no longer bead at range.
- Viewer hulls are ~8x oversize; `Projectiles.lift()` pushes a fresh torpedo
  out of its launcher's displayed hull (fading to zero by 2x hull radius) so
  launches are visible instead of hidden inside the mesh for ~10 s.
- Muzzles: a hull kit may expose `ship.userData.weaponMounts[weapon_slot]`
  (Object3D); otherwise the turret rig in `userData.gunTurrets` whose `slot`
  matches the shot's `weapon_slot` (barrel tip computed from the rig's fixed
  mount and the round's launch bearing, never the animated rig state), else
  the spinal mount. `muzzleLift()` offsets a fresh slug from ship centre to
  that muzzle and blends the offset out over ~2 hull lengths of flight.
- PD beams start at the lens of the firing ball (`userData.pdMounts` by
  turret name, else by index; seeded hull point on hulls without mounts).
- Hit flashes are placed on the face met in the target's frame (relative
  velocity), so fleet drift cannot push a head-on hit to the flank.
- `positionAt(type, id, t)` gives a round's displayed position straight from
  the recording (HullRigs aims PD balls with it).
- Old recordings lack terminal-burn flags; the plume phase is inferred from
  fuel use, range and closure (explicit phase fields win when present).

## Verification

`cd visualizer && npm test`, `npm run build`. Before/after screenshots live in
`/home/plmch/aic-wt/_shots/projectiles/` (`before_*` = untouched viewer,
`a4_*`/`final_*` = this module; side-by-sides in `cmp/`).
