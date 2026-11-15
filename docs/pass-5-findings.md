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

**After the fix:** 180 mirror battles gave alpha 62, beta 52, draws 66 (alpha share of decisive
battles 0.54, 95 % interval 0.45-0.63). One sample of seeds still leaned to alpha and did not
repeat on fresh seeds.

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
- The tactical captains ease a lead ship's throttle when a friendly is astern. Captains, admirals
  and MCP clients see PLUME WASH and CREW DOSE status lines.

| km astern, full burn | Heat GJ/s | Armour cm/s | Dose Sv/s, destroyer bow |
|---|---|---|---|
| 5 | 40 | 1.08 | 0.26 |
| 10 | 16 | 0.36 | 0.11 |
| 20 | 4.7 | 0.02 | 0.031 |
| 50 | 0.79 | 0 | 0.005 |

Formations sat in friendly exhaust for 30 to 400 ship-seconds per battle before the mechanic
existed. No winner changed across seven seeded battles. In the scripted column demo a destroyer
11 km astern of a battleship takes 464 GJ, loses its four bow fittings and ends with an impaired
crew; with avoidance it takes nothing.

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
including seeds it was not tuned on. **That figure was measured before the side-symmetry fix** and
needs re-measuring. An independent review found eight defects in the first version (gun ships
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

Viewer tests rose from 276 to 334; the Python suite from about 2,430 to 2,745.

## Balance: prices cannot fix the torpedo problem

A new tournament tool (`scripts/draft_tournament.py`, 534 seeded battles) showed that a corvette
swarm wins every pairing at 100, 150 and 200 points, usually without losing a ship.

- Gun ships fire zero rounds at a pure torpedo fleet: salvos launched at 500 km land at about
  100 s and the last gun hull is dead by 140-175 s.
- Point defence destroyed no torpedo in any battle; blinded rounds still hit.
- It is a step function: two corvettes win nothing, three win nine of ten.
- A launched round destroys 2.2-6.1 points of gun hull and cost 0.72 points.

Raising the torpedo price weight helped at 150 points only and left the torpedo cruiser too
expensive for the default 100-point budget. The cause is mechanical, not a matter of price.

## Open

- **Rotation physics is parked** on branch `wt5/hunt-rotation`: honest angular momentum (ships can
  currently reverse a spin far faster than their thrusters allow), evasion that only starts a flip
  the hull can finish, and a refitted hit-probability model. Two calibration tests are still red:
  stale old-physics recordings in the fixture, and an ordering issue in how the aim forecast reads
  a target's orders on the checkpoint tick.
- **First orders arrive at T+30 s.** Every hull idles for the first interval. Moving the first
  checkpoint to T+0 fails nine tests that pin the schedule and adds a paid round to LLM battles.
- **Plume avoidance** exists only for the tactical rule-based captains, as a throttle cap on the
  lead ship. LLM commanders get warnings only. An impaired crew's "slower response" is not built.
- **Ambient occlusion** is implemented but not visible in the reference frames.
- **Not seen in a frame:** several torpedo effects, blast scorch on a victim, a dead tube.
- **Generated artifacts.** The fire-control fixture was not regenerated and its tests still pass;
  its heuristic battles must be regenerated with `--doctrine legacy`. Showcase recordings from
  seeded battles change with the symmetry fix.
