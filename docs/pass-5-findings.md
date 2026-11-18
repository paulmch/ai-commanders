# Pass 5: graphics, damage, the drive plume, fairness and speed

Pass 5 (November 15, 2026) set out to improve the graphics and the simulation substantially. It ran
as parallel lanes in git worktrees: each lane audited its own area, built, verified, and was then
reviewed or sent back with specific criticism. This document records what changed, what was
measured, and what is still open.

**Caveats up front.** No LLM battle was run in this pass: every simulation number below comes from
free, seeded battles flown by the rule-based captains. How LLM commanders cope with the new
mechanics is untested. All viewer frames were rendered headless on software WebGL, so nothing was
checked on a real GPU: frame cost, the 4096-pixel shadow map and motion are unverified.

## Simulation

### The two sides were not treated alike

An earlier analysis had found that with equal fleets one side won more often. Mirror battles
(identical fleets, identical captains) confirmed it, and showed why an overall count hides it:

| Fleet, 200 points, both sides identical | Alpha wins | Beta wins | Draws |
|---|---|---|---|
| 3 corvettes, 2 torpedo cruisers, 2 frigates | 14 | 1 | 3 |
| 4 corvettes, 2 dreadnoughts, 2 frigates | 2 | 14 | 3 |
| 2 battleships, 3 corvettes, 3 destroyers | 3 | 7 | 1 |
| 5 cruisers, 3 destroyers | 6 | 3 | 3 |
| All 120 battles (budgets 150 and 200) | 51 | 51 | 18 |

The outcomes were nearly deterministic: the torpedo roster ended "beta eliminated at 352 s, alpha
keeps one ship" in seed after seed.

**Cause.** `CombatSimulation.step()` updated ships in place in dictionary order, alpha first. A
ship updated later steered on the already-moved position, velocity and throttle of every ship
updated before it. Range keeping reads exactly those values, so two mirrored fleets holding the
same order flew different courses from the first tick. Inserting beta's ships first flipped the
winner in every probe; rotating the battlefield 180 degrees changed nothing.

**Fix.** All ships now update from one start-of-tick snapshot (`_update_ships` /
`_advance_ship`, also used by the trajectory-prediction shadow). `tests/test_side_symmetry.py`
holds ten exact mirror tests, seven of which fail on the old engine, and `scripts/mirror_probe.py`
checks a full battle tick by tick.

**After the fix:** 838 fresh mirror battles on the merged engine (before the drive interlock, the
spent-torpedo cull and the final surrender rule were merged; an exact flat-RNG mirror check on the
final code, three fleet types, still ends 289.9 vs 289.9, 293.3 vs 293.3 and 320.0 vs 320.0) gave alpha 227, beta 234, draws
377 (alpha share of decisive battles 0.49, 95 % interval 0.45-0.54). No roster is one-sided
beyond chance (smallest per-roster p = 0.18 before any correction for eight rosters), and paired
survivors, damage, hits and torpedo hits show no side difference. With every random roll forced to
one value, whole 600 s battles stay exact mirrors on every tick. A true effect of a few points
cannot be excluded: the interval is about plus or minus five points. A second pass also staged
the death flags (a reactor meltdown inside the ship loop used to be seen at once by ships updated
later). Gun-roster results still depend on ship order through the shared battle RNG, which is
coupling, not bias.

### Damage has consequences (`src/battle_damage.py`)

Before pass 5, 70-75 % of gun hits did nothing but ablate armour, ships went from first module
damage to death in 0-10 s, and a death affected nothing around it.

- **Surface fittings.** Any hull hit can strike a gun mount, torpedo tube, point-defence turret or
  the drive. A damaged mount cycles up to twice as slowly, a wrecked one is silent, a wrecked tube
  loses its rounds, and the drive gives no thrust at 25 % health or below.
- **Reactor blast.** Every death detonates: 15 km radius for a cruiser, scaling with the square
  root of dry mass. It strips facing armour, adds heat (capped just below meltdown), shreds
  extended radiators, damages fittings, shortens dying hulks' fuses, and kills or blinds torpedoes.
- **Magazine cook-off.** A destroyed magazine burns half its rounds and damages neighbouring
  compartments, never the bridge or reactor.
- **Battle flow.** A fleet battle ends when one side can no longer deal damage. Wrecks keep
  coasting. Surrenders are recorded as events and the result says whether a fleet was eliminated,
  surrendered, or both.

Over six seeded battles, kills and winners were essentially unchanged (26 to 25 kills), while 77
fittings were destroyed and 31 of 68 survivors carried fitting damage. An independent review found
nine defects in the first version (an adrift hull delaying a whole formation's burn, a blast going
off one tick behind a moving hulk, blast heat able to cause a meltdown, and others); all are fixed
with regression tests.

### The drive plume is a hazard (`src/drive_plume.py`)

By the game's own constants a torch carries about 300 TW of jet power, and the engine ignored it:
ships flew directly astern of each other at full burn.

- A burning ship projects a narrow cone astern (3-degree core, hard edge 7.5 degrees, nothing
  beyond 100 km). A ship inside it takes heat, and inside about 22 km armour and fitting damage.
  The coupling is scaled down from the literal figure so the lethal range is tens of km.
- A turning ship's jet is sampled between ticks, so a sweep cannot skip over a ship.
- The crew accumulates a radiation dose, shielded by the armour on the facing presented. Impaired
  at 2 Sv (weapons cycle 1.5 times slower), incapacitated at 8 Sv (the ship coasts and cannot
  fight).
- **Drive interlock.** The engine caps any ship's throttle so its exhaust puts at most 0.5 GJ/s
  on an allied ship, whoever commands it; enemies never limit it. It is computed from the
  start-of-tick snapshot, is always on (also during torpedo evasion; the exemption is a switch,
  off), and commanders see "DRIVE INTERLOCK: throttle held at 3 % of the ordered 100 %". The
  tactical captains additionally side-slip a burn so the jet clears a friend.
- **Crew state reaches every commander.** Own-ship status, the weapon lines (combined cycle
  factor), the admiral's "needs a decision now" block, fleet-net datalink lines for fleet-mates,
  MCP state, and enemy ships ("NOT MANEUVERING, NOT FIRING"). An incapacitated crew gets no
  captain call and no admiral order; orders to it come back as not executable.

| km astern, full burn | Heat GJ/s | Armour cm/s | Dose Sv/s, destroyer bow |
|---|---|---|---|
| 5 | 40 | 1.08 | 0.26 |
| 10 | 16 | 0.36 | 0.11 |
| 20 | 4.7 | 0.02 | 0.031 |
| 50 | 0.79 | 0 | 0.005 |

Formations sat in friendly exhaust for 30 to 400 ship-seconds per battle before the mechanic
existed. In the scripted column demo a destroyer 11 km astern of a battleship takes 464 GJ, loses
its four bow fittings and ends with an impaired crew; with the interlock it takes at most
0.49 GJ/s and 0.07 Sv. With the interlock on, seven tactical battles kept identical outcomes;
four legacy-doctrine battles (close time-limit verdicts) all changed result, with ships held for
50-81 ship-seconds each, 0-10 of them while evading a torpedo. Friendly peaks of 0.64-0.85 GJ/s
still occur against the 0.5 threshold; the missed case was not traced.

Torpedoes in a plume are implemented but switched off (`PLUME_AFFECTS_TORPEDOES = False`). Four
seeded battles came out identical with it on, but they barely contained a tail chase, so that is
not evidence that it is balance-neutral.

### Rule-based captains that fight (`src/llm/heuristic_tactics.py`)

The free captains now plan per fleet: focus fire weighted by how hittable a target is, gun ships
closing nose-first under a closure cap, torpedo boats firing synchronised salvos by target
priority, crippled ships withdrawing on a heading the engine actually flies, a disarmed ship with
no armed friend surrendering. The pass-4 captain stays available as `--doctrine legacy` and is
bit-for-bit unchanged.

On equal fleets the new doctrine beat the old one 48 wins, 1 draw, 3 losses over 52 battles,
including seeds it was not tuned on, and the same totals held when re-measured after the
side-symmetry fix. After the torpedo reprice the battleship roster briefly fell to 5 wins, 2 draws, 5 losses: its
one or two remaining corvettes held their standoff in front of the slow battleships, outside the
fleet's point defence, and died alone. Torpedo boats now keep station behind their own gun line;
the final grid is 45 wins, 2 draws, 5 losses over 52 battles (battleship roster 9/2/1), with
several battleship wins decided on points. In mirror
self-play the torpedo-heavy rosters now mostly end in mutual surrender once both sides are dry. An independent review found eight defects in the first version (gun ships
stalling at zero closure between 81 and 245 km, a withdrawal order that was not flown under fire,
a disarmed ship fleeing until the time limit); all were addressed.

### Engine defects found by invariant checks

Thirteen seeded battles were checked against the engine's own rules. Six confirmed defects were
fixed (`tests/test_fixes_pass5_hunt.py`): the time-limit verdict valued a lost siege dreadnought
at 30 points instead of 105; agreed draws scored ships that were already lost; torpedoes vanished
from the recording at the battle edge; every gun hit named its weapon "unknown" (which is what
LLM captains read in their shot history); auto-draft left budget unspent at the eight-hull cap;
and superseded commands were logged up to 83 times per battle.

### Speed

The engine is 3.3 times faster on a golden set of six battles and a calibration slice (394 to 121
CPU seconds) with every hash identical: event streams, trace, end state and both RNG states.
Untraced gun battles gained about four times, the calibration harness 7.7 times. The main cost
was the 1 ms projectile micro-step loop, which built about 30 temporary vectors per step.
`scripts/perf_check.py` re-runs the golden set and must be re-recorded after any change that
alters battles.

### Rotation with real angular momentum

The engine kept only a scalar spin rate and re-pointed the rotation axis every tick, so a ship
could stop or reverse a spin instantly: a battleship was measured reversing at about 850 times
its thruster authority. Merged on November 16 after an independent review:

- **Spin is a vector.** Its change per tick is limited by the authority available (thrust
  vectoring or RCS); a spin too fast to stop overshoots and comes back; a tick with no attitude
  command brakes at RCS authority instead of zeroing. Ninety-degree turns from rest match the
  hull table within a tick on thrust vectoring and 1.5 % on RCS.
- **Two-stage stop.** The first version made ships under PADLOCK hunt across the 2-degree
  handover from thrust vectoring to RCS (a frigate crossed it about 355 times in 900 s and gained
  266 m/s holding a "no thrust" order). The turn is now planned so RCS can finish it; the
  reviewer's repro shows one crossing on every hull.
- **Evasion respects turn time.** RUN and PRESENT choose an attitude the hull can reach before
  the torpedo arrives. Capital ships now run for 4-20 s where they used to run 26-48 s; for a
  lone battleship or cruiser against a six-round salvo, hits, energy taken and deaths are
  unchanged.
- **Forecast inputs.** The aim forecast charges a burn the lead still extrapolates after a new
  order (in the regression case it promised 127 hits of 200 where none landed) and distinguishes
  a dodger that is already reacting from one that is calm when the round is fired.
- **Calibration set rebuilt on the new engine**, bucketed by base maneuver, every harness round
  kept. The eight LLM battles flown on the old physics moved to a historical file read by one pin.

**What it costs.**
- Eight calibration buckets are outside the 1.5x tolerance and pinned on their own two-sided
  bounds (`RESIDUALS` in `tests/test_fire_control_calibration.py`). The largest: a target dodging
  on PADLOCK under fire from several ships at 200-299 km is hit 2.8 times less than forecast
  (25 hits in 2,072 rounds). The doctrine now tells commanders so.
- Against maneuvering targets the nose of a ship on PADLOCK is more than one degree off about
  half the time (it was 19-28 %), because the line of sight often accelerates faster than RCS can
  follow. Spinal aim is honestly worse.
- The doctrine hit table moved, e.g. a dodging frigate 35 % to 19 % at 100 km. Gun hit rates per
  fleet are unchanged within two points; battle-line and gun-line fleets fire fewer rounds and
  score fewer kills (battle line 14 to 8 over 56 tournament battles).
- The rule-based captains' margin over the old ones fell from 8-0 to 6-2 on eight mirror
  battles; they have not been re-tuned for the new turning.
- About 8 % more CPU on the benchmark set.
- Two cases remain `xfail` with numbers: a burn begun from a coast, and a plain INTERCEPT re-aimed
  mid-flight, both of which burn while the nose swings.

The four Sonnet battles below were flown before this merge, on engine `7b8a7e4`.

### Weapon arcs, turret traverse and point-defence placement

A screenshot showed a point-defence laser firing through the ship's own spinal gun. Until then
every trainable weapon could fire in any direction at once, and the engine and the viewer did not
even share a roll: the trace recorded only the nose direction.

- **Masks from the real hull.** `visualizer/scripts/export_hull_mounts.mjs` ray-casts each mount's
  5-degree direction grid against the ship's own static geometry and writes the committed
  `data/hull_mounts.json`, keyed by hull class. Engine and viewer both read it; a node test fails
  when the hull kit changes and the file is stale.
- **Engine** (`src/weapon_arcs.py`). A battery or point-defence turret fires only when its line is
  in an open cell and the mount has slewed onto it (PD 120, light 45, battery 30, heavy 15 degrees
  per second). Hull roll is a separate, rate-limited angle about the thrust axis that the ship
  uses to unmask weapons (90 degrees takes 30 s on a corvette, 253 s on a dreadnought) and never
  changes motion. A masked battery whose order leaves the target free engages an enemy it does
  bear on. The hit roll and the lead are untouched. Trace frames carry `up` and per-mount `tur`.
- **Commanders** read how many mounts bear, slew time, what unmasks the rest, and the hull's
  blind cones on the thrust axis, which no roll can clear.
- **Point defence re-placed** on barbettes and shoulders, counts unchanged. Before, no turret bore
  on an incoming torpedo 76-85 % of the time; after, 2-6 %. The destroyer's dead-ahead hole and
  the torpedo cruiser's dead-astern hole are closed.
- **Gun depression 2 to 10 degrees.** The old limit was a leftover viewer clamp that put every
  in-plane target on the seam between dorsal and ventral mounts.
- **Viewer.** Hulls sit at the recorded roll, turrets traverse, and beams leave the real muzzle.
  In old recordings a beam whose line is blocked is shown from a mount with a clear line.

**Cost, measured on six seeded 150-point battles (final code).** In the three gun battles, fleets
fire about 9 % fewer rounds per second with arcs than without (0.89 against 0.98); before the
depression change and the fallback it was 19 %. About 57 % of ready gun-ticks are still held
masked. Effective point defence against one direction is roughly half the turret count, and
torpedoes blinded per launch against a screened torpedo fleet fell from 0.52 to 0.20 on the first
measurement, so arcs deepen the torpedo problem described under Balance. The rule-based captains
still beat the old ones 45 wins, 3 draws, 4 losses with arcs on; flying an off-nose aspect to
bear more guns gained only about 4 % rounds. An independent review confirmed the geometry on
40,000 random samples and found eight defects (a hunting roll goal, a double slew, masked guns
that never retargeted, a free turret snap), all fixed.

Not applied: the cruiser and battlecruiser carry one port and one starboard battery that can never
both bear in the battle plane. Arcs are on by default; `AIC_WEAPON_ARCS=0` switches them off.
Nine older point-defence tests still run with all-round fire, and the hit-probability calibration
harness runs with arcs off.

### Evade on a burning order keeps the burn

Measured on the engine before this change: a ship ordered to burn with the evade overlay
spent 2 % of its time under fire burning on its order (BRACE cut the drive, TANK halved it,
EVADE jinked off the line); without the overlay it was 81 %. The owner's rule: with evade
the engine burns the whole time and turns on thrust vectoring. A first version shipped and
was reviewed. It had two defects, fixed in a second round:
- **The armour pick held an attitude off the cone.** It kept the hull's current attitude as a
  candidate even 35-180 degrees off the burn line and burned there at the order's throttle.
  Under HOLD_RANGE that fed back into the range keeper: 34-43 km RMS range error.
- **Fire along the burn line was not dodged at all.** It dodged only rounds with more than
  7 s to run. A nose-on charger with the overlay was hit as often as one without (49 % vs
  50 %).

**The rule** (`src/simulation.py` module docstring):
- **Scope and throttle.** When the order itself burns this tick, the drive keeps the order's
  throttle. Evasion steers the thrust axis round a cone of 20 degrees (`EVADE_BURN_CONE_DEG`)
  about the burn line; 94 % of the push stays on the line at the cone edge.
- **When it swings.** Continuously while rounds are inbound or an armed enemy is within
  400 km (`EVADE_BURN_GUN_REACH_KM`). A coilgun round is led a whole flight time ahead, so what
  defeats it is being unpredictable for the whole flight.
- **How it swings.**
  - A fresh cone point every half edge-to-edge swing, at least 4 s apart: corvette 4 s,
    destroyer 6.8 s, battleship 12.2 s, dreadnought 16 degrees / 15 s. The hull is always
    turning and never settles on an attitude the shooter's lead can extrapolate.
  - Rounds from abeam give the side. Rounds within 30 degrees of the line, ahead or astern,
    get a uniformly random azimuth, from the keyed jink RNG.
- **Armour picks** are cone points only.
- **Off the cone**, the hull turns toward the cone at full authority and the drive burns at
  the order's throttle through the whole turn (owner's decision, round 3: "burn through the
  turn with the percentage given in command"; the round-2 idle beyond 90 degrees is gone).
  Range-control options, the captain's ENGINE ORDER CHECK and the trajectory plans include
  the turn impulse; HOLD_RANGE / standoff keepers size the correction they note by the
  cosine of the angle off (feed-forward, `_turn_then_burn(evade=True)`).
- **Unchanged:** engine-off orders and the EVADE maneuver, bit for bit (228 seeded engagements
  compared with the parent, 0 differ).

**Measured** (`scripts/evade_burn_measure.py`, 24 seeds per row; parent / round 1 / now):

| Ship burning under fire, dodge | Rounds that hit | No overlay | On order under fire |
|---|---|---|---|
| destroyer across 3 gun ships 120-200 km | 9.9 / 9.5 / 11.8 % | 38.5 % | 50 / 100 / 98 % |
| corvette, same | 5.7 / 5.2 / 4.6 % | 23.7 % | 46 / 100 / 97 % |
| battlecruiser, same | 17.2 / 21.6 / 21.7 % | 44.3 % | 45 / 99 / 98 % |
| dreadnought, same | 34.5 / 46.8 / 46.9 % | 51.4 % | 35 / 100 / 100 % |
| destroyer charging 2 gun ships nose-on (review's `noseon`) | 11.6 / 49.2 / 27.2 % | 49.7 % | - |
| frigate / cruiser / battleship / dreadnought charging | 16 / 13 / 20 / 34 -> 20 / 27 / 31 / 42 % | 35 / 51 / 47 / 51 % | - |
| destroyer escaping 2 gun ships from 150 km | 38.5 / 51.6 / 36.8 % | 46.5 % | 15 / 26 / 43 % |
| same, rounds fired while its drive is intact / damaged (now) | 14.0 % / 53.7 % | 39.0 % / 50.1 % | - |

| Escape, merge, range keeping (24 seeds) | Parent | Round 1 | Now | No overlay |
|---|---|---|---|---|
| destroyer escaping: lost | 2 | 19 | 16 | 24 |
| same: km gained after 120 s | 6 | 89 | 109 | 98 |
| destroyer cancelling 2 km/s closure on 1 destroyer: closest approach | 46 km | 67 km | 72 km | 67 km |
| same from 180 degrees off (flip): closing at 60 s | 2.0 km/s | 1.5 | 1.3 | 1.7 |
| flip: destroyers lost | 0 | 6 | 14 | 16 |
| HOLD_RANGE 150 km under torpedo + gun fire (review's `holdab`, 8 seeds x 3 hulls): RMS range error | 11-22 km | 34-43 km | 2.7-7.2 km | 1.8-5.1 km |

**Round 3 (owner: burn through the turn).** "Now" above is round 2, which idled beyond 90
degrees. With the drive burning at the order's throttle through the whole turn:

| 24 seeds | Idle (round 2) | Burn-through (now) | No overlay |
|---|---|---|---|
| flip, destroyer: closing at 60 s / lost of 24 | 1.31 km/s / 14 | 1.69 km/s / 4 | 1.69 km/s / 16 |
| flip, battlecruiser: closing at 60 s / lost of 24 | 1.63 km/s / 5 | 1.81 km/s / 1 | 1.90 km/s / 4 |

| HOLD_RANGE `holdab`, evade full, RMS range error | destroyer | cruiser | battleship |
|---|---|---|---|
| idle (round 2) | 2.7-2.8 km | 5.6-7.2 km | 3.2-4.4 km |
| burn-through, no feed-forward | 9.1-11.1 km | 13.4-13.6 km | 11.1-13.8 km |
| burn-through + feed-forward (now) | 2.0-2.1 km | 4.7-5.1 km | 3.1-3.9 km |
| no overlay | 1.8 km | 4.8-5.1 km | 3.5-4.1 km |

The feed-forward (`_turn_then_burn(evade=True)`) notes a keeper's correction off the cone
at throttle x cos(angle off), floored at `EVADE_BURN_MIN_THROTTLE`; it is a pure function
of the tick's state. Variants tried: no floor (the overlay then jinks at full throttle
while the share is under 10 %) 10-21 km; a floor at half the correction 5.7-11.7 km. The
cost: in the calibration harness (random geometry, the keeper correcting all the time) a
keeper is hit more while it turns at the low share (HOLD_RANGE dodge at 100-199 km: 29 %
of rounds without the feed-forward, 44 % with it); in `holdab` hits stay level (destroyer
15.6 per run with it, 16.2 without, 20.4 with no overlay). Fire control prices the keeper's
turn at its share plus half the rest (`KEEPER_TURN_DODGE_SHARE`); forecast/observed on 56
fresh harness seeds is 0.85-1.10 out to 400 km for HOLD_RANGE, 0.96-1.13 for the flipping
HEADING dodgers, 0.94-1.08 for INTERCEPT.

Predictions: range-control options (`turn_burn_segments`), the captain's ENGINE ORDER CHECK
and the admiral's HEADINGS rows add "evade under fire: the N s turn burns too -> ..." when
the nose is 90 degrees or more off; flown against stated, the end range agrees within 1.5 %
(cancel / open, destroyer to dreadnought; pinned). Trajectory plans already fly the engine
code: predicted vs flown position at the next checkpoint for 90 / 135 / 180 degree flips on
corvette, destroyer, battlecruiser and dreadnought (6 seeds, two windows, with evade) is
0.00-0.53 km mean, max 3.5 km (corvette 90 degrees, where the idle never applied) before,
and 0.00-0.53 km mean, max 3.5 km after; the 135 / 180 degree cases stay under 0.7 km. The
plan sigma is 0.16-0.77 km.

**What it costs.**
- **Escaping ships.** The older overlay saved ships that turned away from guns by turning the
  nose (5-6 times the tail's armour) back onto them and stopping the burn. The new one keeps
  the burn and shows the guns the tail. 30 % of tail hits strike the drive, and a damaged
  drive dodges far less. Escaping destroyers die 16 of 24 times against 2 of 24 before
  (24 of 24 without the overlay).
- **The flip.** Round 2 idled the drive through the far half of the turn and lost 14 of 24
  destroyers. Burning through the turn (round 3, the owner's decision) loses 4, closing at
  1.7 km/s after 60 s like a ship with no overlay (which loses 16). The range keeper's
  feed-forward keeps HOLD_RANGE on its range.
- **Along the line, the value depends on the hull and on the drive.**
  - With the drive intact, an escaping destroyer is hit by 14 % of rounds against 39 %
    without the overlay; an escaping battlecruiser by 35 % against 50 %.
  - A battlecruiser cancelling closure on a lone destroyer gains nothing: 50 % against 49 %.
  - Pooled over whole runs, an escaping battlecruiser is hit more with the overlay (61 %
    against 55 %), because the survivors get caught at short range.
- **Cone adherence.** The aim is always a cone point, but the nose overshoots on carried
  spin: up to 41 degrees, and past 30 degrees (outside the nose arc) on under 1 % of swing
  ticks. In the nose-on charge, 4 % (dreadnought) to 47 % (frigate) of the hits struck the
  flank.
- **Heavy hulls dodge less.** A dreadnought swings a narrower, slower cone and is hit 47 %
  against 34 % before and 51 % without the overlay.
- **Swing cost.** Under fire or near enemy guns the burn always loses about 6 % of its push to
  the swing.
- **Calibration.** Fire control prices a swinging fixed-line burn with one fraction
  (`SWING_ACCEL_FRACTION` 0.24, T^2 form, held out on odd seeds). Range keepers mostly jink
  at full throttle between swung corrections and are priced as engine-off dodgers. Five thin
  buckets beyond 300 km are pinned in `RESIDUALS` with bands that also hold on 56 fresh
  harness seeds; two of them, HOLD_RANGE under fleet fire beyond 300 km, are hit less than
  forecast in both seed sets.
- **CPU.** An overlay-heavy scenario (36 runs of the cross case) took 16.9 s against 14.7 s
  on the parent. The benchmark battles barely use the overlay.
- **Review probes, re-run.** Engine-off orders and EVADE: 228 engagements, 0 differ.
  The never-below monitor counts 0 ticks below the order's throttle in round 3 (scenario
  and draft sets; round 2 had 7,628 and 121, all from the idle beyond 90 degrees).
  `_order_burn` stays pure, and flat-RNG mirror battles show equal swing ticks per side.
  In draft battles (seeds 21-32), 317 of 1,366 swing ticks were off the cone (253 of 812 in
  round 1). Held attitudes went from 116 ticks to 0, and thrust against the order from 111
  throttle-seconds to 2.1 (token only).

### Recording files

New recordings are written as `.json.gz` (compact JSON, deterministic gzip header): 15-37 times
smaller (37.8 MB to 1.02 MB for the largest committed battle). Every tool and the viewer read both
formats through one reader. `run_draft_battle.py --no-recording` writes nothing, for sweeps that
only need outcomes. The committed `data/recordings` folder was converted with
`scripts/compress_recordings.py` (31 recordings, 263 MB to 17 MB, each copy loaded back and compared
before its original was deleted); the README, the capture commands and the notebooks' `source_battle`
paths name the `.json.gz` files.

## Viewer

- **Lighting.** A sun shadow map fitted to the hulls on screen (high tier), a higher key-to-fill
  ratio, image-based fill so the shadow side reads, a new planet with limb darkening, structured
  rings, a denser sky, an anamorphic streak and a print grade. Bloom is unchanged.
- **Hulls.** Armour is a hierarchy of large slabs on one hull-space lattice instead of a noisy
  mosaic. Capital ships have shouldered octagonal sections and wedge bows; the dreadnought is one
  cruciform hull. Engine bells are lathed with concave, ridged liners. Factions have their own
  plate colour and emblem. Triangle counts fell on most classes.
- **Effects.** The torch is a volumetric plume, one draw per nozzle, with shock cells and a turn
  sweep; every jet runs along the thrust axis. Coilgun fire differs by gun class. Point defence is
  a continuous green laser. Impacts, penetrations and torpedo strikes each read differently. A
  death shows a shell out to the engine's real blast radius. Dead mounts, dark bells, cook-offs
  and plume-washed plating are drawn from the recorded events.
- **HUD.** The registry shows a pip per gun, tube and point-defence turret and the drive state;
  the scoreboard counts working guns and tubes; kills carry what the blast caught; the result
  banner says why a battle ended. `admiral_directive_failed` events, a gap from pass 4, are shown.

- **Director.** Key A films a replay from a shot list built out of the recording (salvo over the
  shooter's shoulder, torpedo run-in, pull-back sized from the blast's real extent), with shake on
  battle time; `[` and `]` jump between highlights; `scripts/viewer_clip.py` renders a time range
  to MP4 or GIF.

- **True size.** Hulls were drawn about 60 times too large, so rounds passed through them and ships
  a few km apart overlapped. They are now drawn at their real 65-250 m when the camera is near
  (`render/Scale.js`; key S, `?scale=auto|true|cinematic`). In the default `auto` mode a distant
  hull is magnified only as far as needed to stay a readable silhouette, converging to exactly
  real size inside about 10 hull lengths; `cinematic` is the old fixed enlargement. The engine does
  not decide gun hits on the hull itself: it gates at 500 m around it and rolls a probability, and
  the recorded impact point is where the round entered that gate (median 49-101 m from the hull
  axis, up to 2.7 km; a quarter of the misses cross the real hull). The viewer therefore eases a
  round over its last 3 s onto the crater on the drawn hull, and a miss aside; recorded positions
  are not changed. At real size the torch flame is under 1.5 km long, so the exhaust cone that
  washes ships astern is drawn as a faint beam out to 60 km (`render/DriveJet.js`, `?jets=0`).
- **Point-defence beams** were cut into two 260 px stubs whenever a beam was longer than 520 px in
  a close-up; they are drawn whole. Ribbons are clipped in front of the near plane and widened
  across their on-screen direction.
- **Black squares on real GPUs.** GLSL `pow(x, y)` is undefined for a negative x: software GL
  returns a number, real GPUs return NaN, and an expired smoke puff was drawn as a black quad.
  Every shader in the viewer is now free of such calls, scene colour is guarded where it enters
  the bloom chain, and `tests/gpu_nan.test.js` checks every `pow`, `sqrt`, `log` and `normalize`
  in the shader sources. The fix was verified by emulating GPU behaviour on software GL, not on a
  GPU.

Viewer tests rose from 276 to 410; the Python suite from about 2,430 to 3,107, green on Python
3.11 and 3.12.

## Balance: prices cannot fix the torpedo problem

A new tournament tool (`scripts/draft_tournament.py`, 534 seeded battles) showed that a corvette
swarm wins every pairing at 100, 150 and 200 points, usually without losing a ship.

- Gun ships fire zero rounds at a pure torpedo fleet: salvos launched at 500 km land at about
  100 s and the last gun hull is dead by 140-175 s.
- Point defence destroyed no torpedo in any battle; blinded rounds still hit.
- It is a step function: two corvettes win nothing, three win nine of ten.
- A launched round destroys 2.2-6.1 points of gun hull and cost 0.72 points.

The cause is mechanical, not a matter of price. What landed: spent torpedoes leave the battle,
the time-limit verdict values hulls at their draft price, and the torpedo price weight went from
0.040 to 0.080 (corvette 18 to 24 points, torpedo cruiser 65 to 100, the dearest setting at which
every hull fits the default 100-point budget). A torpedo cruiser behind a frigate screen still
wins every battle at 150 and 200 points.

**The lever, parked on `wt5/balance-pd` (not green, 15 tests red).** Blinded rounds still hit:
70-77 % of hits taken came from rounds whose seeker was already dead. Lowering the seeker-kill
energy from 50 MJ to 29 MJ or below flips it: the screened torpedo fleet then loses to the two
turret-rich fleets and still beats the slow, turret-poor ones. But any value in that range lets a
lone two-turret hull blind a single round, which contradicts an explicit design guard ("a lone
ship does not stop a committed torpedo"), and it leaves the gun line and escort fleets at 89-95 %
of the field. That is an owner decision before it can be finished. Range also still costs a
torpedo nothing (a 500 km shot arrives faster, not weaker), with no small fix.

## First LLM battles on this engine: Sonnet 5.5 against Sonnet 5

The Anthropic API is now a native provider (`src/llm/anthropic_client.py`, official SDK; a bare
model id such as `claude-sonnet-5-5` goes to Anthropic, a `vendor/model` id to OpenRouter). Four
battles were run on November 15-16, 2026, engine `7b8a7e4`: 150-point draft, 20 checkpoints,
medium effort on both sides, admiral vision, server-side fallbacks off so a refused turn could
not be silently replayed by another model.

| Battle | Alpha | Beta | Winner | Score | Cost |
|---|---|---|---|---|---|
| a | Sonnet 5.5: 4 battlecruisers, 2 destroyers, 2 frigates | Sonnet 5: dreadnought, 2 battlecruisers, 2 destroyers, frigate | Sonnet 5.5 | 238.0 to 153.9 | $8.31 |
| b2 | Sonnet 5: 5 corvettes, 3 frigates | Sonnet 5.5: 4 battlecruisers, 2 destroyers, 2 frigates | Sonnet 5 | 368.8 to 58.6 | $6.06 |
| c | Sonnet 5.5: 3 battlecruisers, 2 destroyers, 3 frigates | Sonnet 5: 3 battleships, 3 frigates | Sonnet 5.5 | 276.0 to 123.7 | $7.49 |
| d | Sonnet 5: 6 battlecruisers | Sonnet 5.5: 4 battlecruisers, 2 destroyers, 2 frigates | Sonnet 5.5 | 252.4 to 127.4 | $7.13 |

Sonnet 5.5 won three of four. Four battles with different drafted fleets are too few to rank the
models, and the one Sonnet 5 win was decided in the draft: its five torpedo corvettes behind a
frigate screen killed six of eight ships (40 torpedoes, 25 impacts) without loss, the torpedo
dominance described under Balance, now found by an LLM. Sonnet 5.5 drafted nearly the same gun
fleet every time; in one aborted attempt (b, $0.07) it failed the draft three times in a row
(10 ships against a cap of 8, then 158 points against 150).

All 2,277 calls of the four battles completed: no failures, truncations, malformed arguments or
refusals, and about 70 % of prompt tokens were read from cache. Costs are computed from token
usage and the price table, not reconciled against the provider's invoice. Both models reasoned
about weapon arcs ("keep nose-on alignment during the burn so spinals stay unmasked"); neither
mentioned the plume, the interlock or crew state, and the recordings show why: every friendly
wash stayed under the interlock's 0.5 GJ/s limit. The recordings are in
`data/recordings/p5_sonnet_battles_20261116/` (65 MB, not committed).

## Open

- **First orders arrive at T+30 s.** Every hull idles for the first interval. Moving the first
  checkpoint to T+0 fails nine tests that pin the schedule and adds a paid round to LLM battles.
- **Plume.** An impaired crew's "slower response" is not built (only slower weapon cycling). The
  formation throttle matching does not know about the interlock, so a held lead falls behind
  unless its captain side-slips. The doctrine text does not mention the interlock.
- **Point defence and torpedo range** as described under Balance.
- **Seeded battles are bit-reproducible only within one Python minor version.** 3.11 and 3.12
  differ in the last bits; `scripts/perf_check_golden.json` was recorded on 3.12.
- **Ambient occlusion** is on for the high tier (hull pixels about 9 % darker in recesses on a
  shadow-side close-up); its frame cost is unmeasured.
- **Director.** Live mode, the highlight keys and surrender shots have unit tests only.
- **True size.** Not checked on a real GPU, in motion or in live mode. The far drive jet saturates
  when seen along its axis, track uncertainty rings follow the hull magnification, and
  `torpedo-salvo.png` and the hull line-up in the README are still cinematic-scale frames.
- **Not seen in a frame:** several torpedo effects, blast scorch on a victim, a dead tube.
- **Generated artifacts.** The fire-control fixture was not regenerated and its tests still pass;
  its heuristic battles must be regenerated with `--doctrine legacy`. Showcase recordings from
  seeded battles change with the symmetry fix.
