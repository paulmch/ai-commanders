# Projectile rendering handoff

`render/Projectiles.js` owns in-flight weapons and instantaneous effects
(<= 0.46 battle seconds): coilgun slugs (velocity-aligned tracer, 1 s cooling
wake, 6 s fading coast for misses), torpedoes (instanced finless body with a
gimbaled bell, throttle/phase-sized motor plume + exhaust wake, RCS jets and
puffs, faction arrowhead at fleet range pointing along its travel), continuous PD
beams (burn spot on the facing surface of the round, ablation sparks), coil
muzzle discharge, torpedo ignition/burn-out/seeker-kill/intercept flashes and
the instant hull-impact flash + spark burst (on the DamageFX crater of the
round: pierce = spall jet, graze = fan along the plating; duds skip off the
hull), and the fleet-range engagement lines (`TargetingLines.js`, ribbon kind
8). `WeaponTimeline.js` indexes the recording (plus `threatsAt` for inbound
torpedoes and the per-ship fire index); `WeaponBatches.js` holds the shared
draws (ribbons, flashes, fleet-range markers, torpedo bodies + bells);
`TorpedoRcs.js` is the torpedo thruster math. Slug and torpedo wakes carry the
firing side in their seed (`factionSeed`). Everything is evaluated from battle time (seek/scrub/pause
give identical buffers; see tests), no per-frame allocation, no PointLights.

## Merge hooks

- `SceneManager.init`: `this.weapons = new Projectiles(this)` before warmUp.
- `SceneManager.getEntityPosition`: consults `this.weapons.position()` first.
- `main.js` animation: `scene.weapons.update(loader, t)` replaces the old
  per-entity projectile/torpedo/PD-beam loops and the shot muzzle-flare loop.
  The torpedo-moment loop and the visual-impact loop are KEPT, but with a
  recording loaded `spawnTorpedoDetonation` / `spawnHitEffect` return at once
  (`damageFX.ownsImpacts()`): DamageFX spawns every persistent hit blast from
  the damage timeline (kineticStrike / breach / warhead for torpedoes by
  recorded yield and breach, impact / pierce / graze for slugs) at the mark's
  crater, carried with the struck ship. Torpedo misses still spawn their flare;
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
  the spinal mount: the kit's `userData.spinalMuzzle`, else the per-class
  `SPINAL_MUZZLE` table in `WeaponTimeline.js` (`spinalMuzzleInto`; DamageFX
  puts a wrecked spinal's scorch on the same point). `muzzleLift()` offsets a
  fresh slug from ship centre to that muzzle and blends the offset out over
  ~2 hull lengths of flight.
- PD beams start at the lens of the firing ball (`userData.pdMounts` by
  turret name, else by index; seeded hull point on hulls without mounts).
- Hit flashes sit on the crater `damageFX.surfaceFor(roundId)` resolves (the
  same spot as the persistent blast, spall and scar); a round's last
  sub-second flies into it (`surfaceImpact`, lazy). Without DamageFX they fall
  back to the face met in the target's frame (relative velocity), so fleet
  drift cannot push a head-on hit to the flank. Hit flashes keep a ~4 px
  floor so a hit still reads at fleet range.
- Duds: the engine rolls a torpedo's P(hit) at closest approach, so a failed
  roll at 100 m still passed deep inside the ~8x drawn hull. Every miss within
  `DUD_REACH_KM` is a dud candidate; `dud()` shows the ones whose recorded
  closest approach lies inside the displayed hull's radius toward the round
  (`dudSkims`; P(hit) >= `DUD_P` when no closest approach is recorded) as a
  spark skid off the plating, placed by `damageFX.hullSurface` (a ray cast in
  from outside onto the hull kit; the hull ellipse sat inside the box-like
  mesh and buried the flash). Misses that cleared the drawn hull show nothing.
  The loader's torpedo `miss` flare (main.js) still pops at the round's last
  sample as well.
- `positionAt(type, id, t)` gives a round's displayed position straight from
  the recording (HullRigs aims PD balls with it).
- Old recordings lack terminal-burn flags; the plume phase is inferred from
  fuel use, range and closure (explicit phase fields win when present).

## Torpedo thrusters

No fins: a space torpedo steers with its engine and its RCS. The sim
(`src/torpedo.py`) gimbals the main engine to thrust in any direction
(cruise closure burns, APN corrections up to 90 deg off the flight path, the
terminal NEZ burn) and routes only small precision trims through lateral RCS
at 5% of main thrust with the engine cold (`GuidanceCommand.use_rcs`, the APN
"correction" branch: above the 12 km/s closure floor, outside the terminal
burn, predicted miss > 50 m and a command inside RCS authority). That is
~0.6% of guided ticks in the auto-draft fleet showcase and ~11% of torpedo
frames against a jinking corvette (`generate_test_battle torpedo_strike`),
5-20 s before impact. The combined main+RCS command is never issued.

Recordings carry per torpedo per frame, for the tick that ENDED at that
frame: `thr` (main throttle 0..1), `fwd` (thrust = nose axis, unit, only
while the engine burns) and `rcs` (lateral RCS thrust, unit direction x
throttle, only while it fires). `TorpedoRcs.js` turns that into:

- Attitude: the nose is on each tick's thrust axis by the time that burn
  starts; with the engine cold the seeker holds the target (nose down the
  line of sight, which also makes the sim's LOS-perpendicular trims purely
  lateral), or the last attitude once the seeker is dead, the tanks are dry
  or the target is gone. Slews follow a trapezoidal rate profile: a spin-up
  couple, coast, a brake couple (`slewProfile`, `pulsesAt`). Engine lit,
  the gimbal takes the torque (the bell deflects up to 6 deg and the plume
  follows it) and only slews > 4 deg add RCS puffs; engine cold, the quads
  do it all.
- Quads: two rings of four radial nozzles (`RCS_LAYOUT`). `jetLevels` fires
  the quads facing away from a translation thrust on both rings (pure force)
  and opposite sides fore/aft for a couple (pure torque). Trims are pulse-
  width modulated at the recorded throttle (0.2 s period).
- Rendering: a firing quad draws a short white-blue ribbon (kind 7) plus a
  cool nozzle glow; every pulse leaves a small puff (flash kind 5) drifting
  off the nozzle axis for 0.35 s. Sized to the hardware and faded out
  between ~16 and ~5 px of body length, so fleet range shows only the
  faction diamond.
- Plume: on/off from `thr` (an RCS trim burns fuel but no longer lights the
  main plume), length and width scale with throttle.
- Old recordings (no `thr`): the body follows velocity as before and a
  trim is inferred only where the velocity turned while the main engine was
  cold (fuel use < 0.03 km/s per s): finite-differenced acceleration minus
  its along-axis part, above 0.5 m/s^2 of rounding noise (`legacyTrim`). On
  the showcase this finds all 11 recorded trims with the right direction,
  plus five 5-7%-throttle lateral main burns.

## Verification

`cd visualizer && npm test`, `npm run build`. Before/after screenshots live in
`/home/plmch/aic-wt/_shots/projectiles/` (`before_*` = untouched viewer,
`a4_*`/`final_*` = this module; side-by-sides in `cmp/`). Torpedo thruster
close-ups: `/home/plmch/aic-wt/_shots/torp-rcs/` (`before_*` / `after_*`,
iterations in `iter/`), from `showcase_torpedo_fleet.json`.
