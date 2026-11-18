# Fleet Draft Mode

Admirals build their own fleets from a point budget, place them in a starting
formation, then command them through the battle - while cheap AI captains fly
the individual ships.

```bash
# Two LLM admirals draft and fight (heuristic captains, admiral vision on)
uv run python scripts/run_draft_battle.py \
    --alpha-admiral anthropic/claude-opus-5 \
    --beta-admiral google/gemini-3.5-pro --trace

# Fully offline smoke run - deterministic auto-drafts, zero LLM calls
uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace

# Cheap LLM captains instead of the rule-based AI
uv run python scripts/run_draft_battle.py --captain-model anthropic/claude-haiku-4.5
```

## Flow

1. **Selection** (`select_fleet` tool): each admiral sees the costed catalog
   and a point budget (default 100, `--budget`). Illegal drafts (over budget,
   unknown hull, too many ships) get the validation error fed back and are
   retried up to 3 times; total failure falls back to a deterministic
   auto-draft so a battle always starts.
2. **Formation** (`set_formation` tool): the admiral places each drafted ship
   as a km offset from its fleet anchor in its own frame (+x toward the
   enemy, y lateral, z vertical, clamped to +/-150 km, minimum 2 km
   separation enforced by nudging). Beta's frame is rotated 180 degrees - not
   mirrored - so "left flank" means the same thing to both sides.
3. **Battle**: the normal fleet battle loop. Admirals keep issuing checkpoint
   orders (with tactical-plot vision by default, `--no-vision` to disable);
   captains execute.

## The briefing admirals draft with

`build_catalog_text` + `_selection_prompt` give both admirals the same
tactical manual before they spend a point:

- **Catalog**: per-hull cost, acceleration, armor by facing, PD turret
  count, sim-truth torpedo magazines (read from the weapon entries the
  simulation actually loads), muzzle-rated gun energies, and the full
  designer role text.
- **Weapon effects** (calibrated from recorded battles): ~1.5 cm armor
  ablation per GJ; ALL kinetic impacts scale with closing speed (gun GJ
  ratings are muzzle energy - mutual closure multiplies, mutual recession
  bleeds off); torpedoes accelerate and steer all the way in, so their
  closure scaling is one-sided in the attacker's favor.
- **Torpedo flight profile**: 14 km/s at 12g, augmented proportional
  navigation, never brakes; hoards delta-v outside its no-escape zone,
  then commits everything above a ~0.5 km/s reserve into closing speed.
- **Counterplay**: kiting drains magazines and PD-farms seekers before the
  NEZ; a fleet only kites as fast as its slowest hull - commit to one
  doctrine, not half of each.

## Torpedo retargeting and the datalink (pass 5)

Rounds whose target dies (or becomes a dying hulk) are reassigned by the
fleet datalink (`src/torpedo_link.py`), as a GROUP: each reachable hull gets
the rounds it still needs (`need_for` + 1 margin: PD soak + rounds to kill on
the rounds' forecast impact, minus the live rounds that arrive there first),
the rest go to the least over-subscribed reachable hull. Reachable: the
engine's `can_intercept`, the steer flown as a finite burn, and at least
`0.25 + 0.02 * t_go` km/s (cap 1.12) left after it. Fresh seekers keep the
most fuel, PD-singed ones take the fastest intercept. A sprint round from a
launcher at rest reaches ~16 km abeam when its target is lost 200 km out; a
`reserve` round keeps ~6 km/s until its last 20 s (~106 km abeam), less from
a closing launcher (~52 km), and reserve rounds beyond what their target
needs are diverted before it dies. The captain's OVERKILL line and the
admiral's economy quote the same board. Measured in seeded rule-based
battles: 58% of rounds lost their target in flight, 69% of those were
reassigned, 56% of the reassigned hit; 73% of diverted reserve rounds hit.
Before pass 5 the per-round rule recovered about 39% of orphans (52% of those
hit); in one LLM battle (p5w5_b3) 8 of 28. Spacing units beyond an orphan's
steering envelope still turns enemy overkill back into waste.

## Point costs (src/llm/fleet_draft.py)

Costs are **computed from `data/fleet_ships.json`**, not hand-typed - change a
hull's armour or magazine and its price follows. A hull pays for:

| Term | Rate | What it buys |
|------|------|--------------|
| armour | 1 pt / 55 t | how hard it is to kill |
| PD turret | 1 pt each | active defence |
| gun mounts | 8 pts per GJ/s | sustained energy throughput |
| torpedo rounds | 0.080 pts per stowed GJ (1.44 per round) | **magazine depth x per-round yield** |

Gun throughput is `salvo_size * kinetic_energy_gj / (cooldown_s +
(salvo_size - 1) * intra_salvo_cooldown_s)`. TI Mk3 timing gives batteries
5 shots per 60s, spinals 5 per 72s, and siege coilers 3 per 60s, retaining
our per-shot damage. The Cost column below includes this timing update.

| Hull | Cost | 2026-11-15 r1 | 2026-09-27 | 2026-08-13 | Original | Notes |
|------|-----:|-----:|-----:|-----:|----:|-------|
| frigate | 7 | 7 | 7 | 7 | 8 | thin-skinned escort gunboat |
| destroyer | 18 | 18 | 18 | 18 | 16 | workhorse gun platform |
| corvette | 24 | 27 | 18 | 17 | 10 | 3g torpedo boat, 8 rounds (144 GJ) |
| battlecruiser | 25 | 25 | 25 | 24 | 22 | cruiser guns, lighter armor |
| cruiser | 29 | 29 | 29 | 28 | 26 | heavy combatant |
| battleship | 43 | 43 | 43 | 42 | 40 | line of battle |
| dreadnought | 55 | 55 | 55 | 53 | 55 | mobile fortress |
| dreadnought_siege | 56 | 56 | 56 | 54 | 58 | 7.25 GJ siege spinal |
| cruiser_torpedo | 100 | 119 | 65 | 58 | 30 | 4 launchers, 48 rounds (864 GJ) |

The Cost column is the 2026-11-15 (pass 5, round 2) table. Only the torpedo
dial moves the two torpedo hulls: 0.040 -> 0.102 in round 1 (priced on what a
magazine destroys), then back to 0.080, the dearest value at which every hull
can still be drafted at the default 100-point budget. The same points are also
what a hull is worth in a time-limit verdict (`ship_class_value` in
`src/llm/battle_runner.py`). The measurements are in
[2026-11-15: pass-5 tournament](#2026-11-15-pass-5-tournament-0040---0102)
and [round 2](#2026-11-15-round-2-mechanics-then-0102---0080) below; the
sections before them are the history of the dial.

### Why torpedoes got expensive (2026-08-13 rebalance)

`cruiser_torpedo` and `cruiser` are the **same hull** - identical mass, 241 cm
nose, 1.5g. The torpedo variant swaps guns for 4 launchers and doubles the PD,
and used to cost 4 points more. So 48 guided rounds were nearly free.

They should not have been. A Trident is a guided fusion-torch penetrator that
steers all the way in, and at the engine's enforced 12 km/s closure floor
(`MIN_CLOSING_SPEED_KPS`, `src/torpedo.py`) a 250 kg penetrator lands **18 GJ** -
4.2x a spinal coiler slug, 25x a coilgun round - rising past 40 GJ head-on.
Priced per GJ/s of sustained output, the old table read:

| Hull | pts per GJ/s (old) |
|------|-----:|
| cruiser_torpedo | 5.0 |
| corvette | 6.7 |
| dreadnought | 87.7 |

A 17.5x efficiency gap, which made saturation doctrine the only rational draft
at every budget. Two frontier admirals independently drafted 5x
`cruiser_torpedo` at 150 points in the same match; the one that later deviated
to a dreadnought line was swept 0-for-7 without killing a single ship.

Pricing the magazine on delivered energy closes the gap to under 10x. 100
points now buys a dreadnought plus a frigate, five destroyers, or **one**
torpedo cruiser - saturation is still buyable, but you pay for every round.

### Measured effect (offline A/B, heuristic captains, no LLM)

Five seeds per matchup, no admirals, fixed fleets - `scripts/` has no runner
for this, it was a throwaway harness driving `LLMBattleRunner` with
`client=None`. Record is from the torpedo side's point of view:

| torpedo side | vs gun side | points | record |
|--------------|-------------|-------:|--------|
| OLD 5x cruiser_torpedo | 5x cruiser | 290 v 140 | 5W-0L-0D |
| OLD 5x cruiser_torpedo | dreadnought + 3x cruiser + frigate | 290 v 144 | 5W-0L-0D |
| NEW 2x cruiser_torpedo + 4x frigate | 5x cruiser | 144 v 140 | 3W-1L-1D |
| NEW 2x cruiser_torpedo + 4x frigate | dreadnought + 3x cruiser + frigate | 144 v 144 | 0W-5L-0D |

The top two rows are the old distortion measured: the fleet 150 old points
bought costs **290** under the new model and beats both gun archetypes 10-0.

The bottom two are the new meta - rock-paper-scissors instead of dominance.
Torpedo doctrine still beats a massed medium-cruiser line (3-1-1), and a
dreadnought-anchored mix now hard-counters it (0-5): enough PD dwell and nose
armour to eat a 2-hull salvo and close. That counter did not exist before -
the 2026-08-11 note that "kiting is not available to battleship fleets" was
true because nothing survived the approach at any composition.

Caveat: 5 seeds per cell is a small sample, and both sides fly heuristic
captains. The 2026-09-26 recalibration below re-measured this with a proper
harness. Its old-engine baseline does not reproduce the rock-paper-scissors
result with corvette escorts.

### 2026-09-26 recalibration: 0.032 -> 0.040

The torpedo guidance fixes (final-second overshoot, terminal burn reserve,
intercept time, hit aspect, retarget resets, fuel-out accuracy) and the
cruiser_torpedo internals fix made torpedoes stronger. The dial was re-fit with
`scripts/calibrate_torpedo_cost.py`, a reusable harness. It fights zero-LLM
heuristic battles between fixed archetype fleets, each drafted greedily to the
same budget at a given dial value:

- **torpedo**: as many cruiser_torpedo as fit, then corvettes, then a frigate top-up
- **gunline**: as many cruisers as fit, then destroyers, then frigates
- **dread**: one dreadnought, then cruisers, then frigates
- **balanced**: auto-draft's battleship / destroyer / corvette cycle, then frigates

Each seed is fought twice with the sides swapped. The formation (staggered
wall / line abreast / screen) cycles with the seed. Results are cached per
battle, so re-runs and refinements are cheap.

```bash
uv run python scripts/calibrate_torpedo_cost.py --k 0.036,0.040,0.044 \
    --budget 100,150,200 --seeds 24 --out calib.jsonl
uv run python scripts/calibrate_torpedo_cost.py --engine-root ../other-checkout \
    --k 0.032 --seeds 36 --out base.jsonl      # any checkout's engine
```

Torpedo-side results, W-D-L and score = (W + D/2) / N. Most battles run to
the 25-checkpoint limit and are scored on tactical advantage.

Old engine (`main` @ 7b6bc4a), 0.032, 36 seeds x 2 sides:

| pts | torpedo fleet | vs gunline | vs dread | vs balanced | score |
|----:|---------------|-----------:|---------:|------------:|------:|
| 100 | CT + 2 CV + FF | 72-0-0 | 69-3-0 | 42-7-23 | 87% |
| 150 | 2 CT + 2 CV | 72-0-0 | 70-2-0 | 42-9-21 | 88% |
| 200 | 3 CT + CV + FF | 54-3-15 | 55-4-13 | 22-15-35 | 66% |

Current engine at 150 pts, 36 seeds x 2 sides (score per matchup):

| k | CT / CV cost | torpedo fleet | gunline | dread | balanced | score |
|--:|:------------:|---------------|--------:|------:|---------:|------:|
| 0.032 | 58 / 17 | 2 CT + 2 CV | 98% | 99% | 97% | 98% |
| 0.036 | 62 / 17 | 2 CT + CV + FF | 57% | 48% | 54% | 53% |
| **0.040** | **65 / 18** | **2 CT + CV** | **90%** | **57%** | **19%** | **56%** |
| 0.044 | 69 / 19 | 2 CT + FF | 4% | 10% | 3% | 6% |
| 0.048 | 72 / 19 | 2 CT | 3% | 57% | 15% | 25% |
| 0.052 | 76 / 20 | CT + 3 CV + 2 FF | 94% | 95% | 88% | 92% |

Current engine at 100 / 200 pts, 24 seeds x 2 sides:

| k | 100 pts fleet | score | 200 pts fleet | score |
|--:|---------------|------:|---------------|------:|
| 0.034 | CT + 2 CV | 92% | 3 CT + CV | 73% |
| 0.038 | CT + 2 CV | 92% | 3 CT + FF | 12% |
| **0.040** | CT + CV + 2 FF | 39% | 3 CT | 25% |

**Why 0.040.** Only the 0.034-0.041 range brings the 150-pt torpedo fleet near
even. Inside that range, 0.040 has the smallest spread across budgets
(39 / 56 / 25%). It is also never dominant, and at 150 pts it keeps a
rock-paper-scissors shape: it beats the cruiser line (65-0-7), splits with
the dreadnought mix (41-0-31) and loses to the battleship-heavy balanced
mix. The value sits inside the 0.0392-0.0415 band, where no calibration
fleet changes composition at any of the three budgets. The corvette's raw
cost there (17.996) is at the centre of its 18-pt step.

**What the dial cannot do.** No single value lands every budget in the 45-55%
band. The torpedo fleet's result tracks how many **corvettes** the leftover
budget buys, not the dial itself:

- 2+ corvettes: 73-98%
- one corvette: 53-56%
- none: 6-25%

A frigate escort is a net loss because it is killed almost every battle.
Heuristic torpedo cruisers launch only ~14 of their 48 rounds in a 750 s
battle, while corvettes empty their magazines. So per point, the corvette
is the strongest torpedo platform. Yet this dial moves its price one sixth
as fast as the cruiser_torpedo's (144 GJ vs 864 GJ stowed). Finer cost
granularity would only move the cliffs. The likely fix is structural: price
launch rate as well as stowed yield, so corvettes rise relative to
cruiser_torpedo. Re-fit with the same harness after any such change.

### 2026-09-27 re-check: salvo guns, heat ladder, range control

The next engine round changed three things:

- coilguns fire Terra Invicta salvos (5 shots per 60 s for batteries,
  5 per 72 s for spinals)
- a heat ladder (guns hold at 90%, meltdown at 150%)
- EVADE and torpedo hulls now hold range instead of merging

Gun hulls repriced by +1-2 pts from their own data. The torpedo hulls'
prices did not move, and neither did the dial's plateaus. Re-run at
viz-pass-3 @ bed65c5, 24 seeds x 2 sides:

| k | CT / CV | 150 pts fleet | gunline | dread | balanced | score |
|--:|:-------:|---------------|--------:|------:|---------:|------:|
| 0.032 | 58 / 17 | 2 CT + 2 CV | 96% | 98% | 94% | 96% |
| 0.035 | 61 / 17 | 2 CT + CV + FF | 46% | 46% | 67% | 53% |
| **0.040** | **65 / 18** | **2 CT + CV** | **67%** | **65%** | **88%** | **73%** |
| 0.044 | 69 / 19 | 2 CT + FF | 4% | 6% | 5% | 5% |
| 0.048 | 72 / 19 | 2 CT | 15% | 1% | 42% | 19% |

| k | 100 pts (gunline / dread / balanced) | 200 pts (gunline / dread / balanced) | mean |
|--:|------------------|------------------|-----:|
| 0.035 | 92% (96 / 99 / 81) | 68% (38 / 67 / 100) | 71% |
| **0.040** | 41% (75 / 0 / 48) | 26% (26 / 50 / 2) | 47% |

**0.040 stays.** Its 100 / 150 / 200 average (47%) is inside the target band.
It is above the band only at 150 pts. The 0.035 alternative is
torpedo-favoured at both 100 pts (92%) and 200 pts (68%). At 100 pts, 0.040
keeps rock-paper-scissors: it beats the cruiser line and cannot touch the
dreadnought mix. Against the previous engine it is better centred
(mean 47% vs 40%) but more budget-sensitive: the spread widened from 31 to
47 points, because 150 pts rose from 56% to 73%. The corvette cliff still
dominates:

- 2 corvettes: 92-96%
- 1 corvette: 41-73%
- none: 5-26%

The single dial can only choose the least-bad plateau. The launch-rate term
proposed above is now the gating change for balance that is robust across
budgets.

### 2026-11-15: pass-5 tournament (0.040 -> 0.102)

(Round 1. Measured before the captains' round-3 fixes were merged; the
engine, the prices and the conclusions about verdicts were revised in
[round 2](#2026-11-15-round-2-mechanics-then-0102---0080) below. The
diagnosis of the salvo and of point defence stands.)

Pass 5 changed the battlefield (fittings can be shot off, reactor blasts kill
torpedoes) and the free commanders: the tactical heuristic captains mass and
synchronise salvos and focus fire. The prices were re-measured with a new
instrument, `scripts/draft_tournament.py`: a seeded round robin over seven
fleet archetypes drafted to the same budget, every pairing fought in both side
assignments, heuristic captains on both sides, no admirals, no LLM. 500 km
start, 600 s limit, auto-draft's staggered wall for both fleets. Battles are
cached by (engine source, rosters, seed), so re-scoring a price table only
fights the pairings whose rosters it changed.

```bash
uv run python scripts/draft_tournament.py --budget 150,200 --seeds 2 --out t.jsonl
uv run python scripts/draft_tournament.py --set TORPEDO_POINTS_PER_GJ=0.04 \
    --budget 150 --seeds 2 --out t.jsonl               # any candidate price table
uv run python scripts/draft_tournament.py --rows torpedo,swarm --seed-start 3 --seeds 2 --out t.jsonl
uv run python scripts/draft_tournament.py --fleet three=corvette*3 \
    --archetypes three,gunline,battleline --out t.jsonl   # explicit rosters
```

Archetypes: `torpedo`, `gunline`, `combined`, `flagship` are the four
auto-draft doctrines (`auto_draft(doctrine=...)`); `battleline` (battleships,
then destroyers), `escort` (two battleships behind a frigate PD screen) and
`swarm` (corvettes to the hull cap) are explicit recipes.

Cells are the row fleet's score, (W + D/2) / N, followed by how those battles
ended: **A** one fleet annihilated, **T** time limit, decided on points.
"Tonnage margin" is the share of the enemy's dry mass destroyed minus the
share of the row fleet's own lost, averaged over its battles - it does not
depend on any price.

#### Before: 0.040 (corvette 18, cruiser_torpedo 65)

**150 pts**, seeds 1-2 on the torpedo and swarm rows, seed 1 on the gun-v-gun cells:
- **torpedo**: cruiser_torpedo, 4x corvette, frigate (144 pts)
- **gunline**: 4x destroyer, 2x cruiser, 2x frigate (144 pts)
- **combined**: 2x battleship, 2x destroyer, corvette (140 pts)
- **flagship**: 2x dreadnought, 3x frigate, corvette (149 pts)
- **battleline**: 3x battleship, destroyer (147 pts)
- **escort**: 2x battleship, cruiser, 5x frigate (150 pts)
- **swarm**: 8x corvette (144 pts)

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 100% A | 100% A | 100% A | 100% A | 0% A | 83% | +0.72 |
| **gunline** | 0% A | - | 100% T | 100% T | 100% T | 25% T | 0% A | 39% | -0.29 |
| **combined** | 0% A | 0% T | - | 50% T | 50% T | 0% T | 0% A | 12% | -0.52 |
| **flagship** | 0% A | 0% T | 50% T | - | 0% T | 0% T | 0% A | 6% | -0.68 |
| **battleline** | 0% A | 0% T | 50% T | 100% T | - | 0% T | 0% A | 19% | -0.52 |
| **escort** | 0% A | 75% T | 100% T | 100% T | 100% T | - | 0% A | 50% | -0.39 |
| **swarm** | 100% A | 100% A | 100% A | 100% A | 100% A | 100% A | - | 100% | +0.94 |

n per cell: [2, 4], ended {'annihilation': 44, 'time limit': 22}, side check {'alpha': 34, 'beta': 32}, missing 18

**200 pts**, seed 1:
- **torpedo**: 2x cruiser_torpedo, 3x corvette, 2x frigate (198 pts)
- **gunline**: 5x cruiser, 3x destroyer (199 pts)
- **combined**: 2x battleship, 3x destroyer, 3x corvette (194 pts)
- **flagship**: 2x dreadnought, 4x corvette, 2x frigate (196 pts)
- **battleline**: 4x battleship, destroyer, frigate (197 pts)
- **escort**: 4x battleship, 4x frigate (200 pts)
- **swarm**: cruiser_torpedo, 7x corvette (191 pts)

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 100% A | 100% A | 100% A | 100% A | 0% A | 83% | +0.52 |
| **gunline** | 0% A | - | 50% T | 50% T | 100% T | 50% T | 0% A | 42% | -0.30 |
| **combined** | 0% A | 50% T | - | 100% T | 100% I | 100% T | 0% A | 58% | +0.04 |
| **flagship** | 0% A | 50% T | 0% T | - | 100% IT | 100% T | 50% A | 50% | +0.24 |
| **battleline** | 0% A | 0% T | 0% I | 0% IT | - | 50% T | 0% A | 8% | -0.65 |
| **escort** | 0% A | 50% T | 0% T | 0% T | 50% T | - | 0% A | 17% | -0.53 |
| **swarm** | 100% A | 100% A | 100% A | 50% A | 100% A | 100% A | - | 92% | +0.68 |

n per cell: [2], ended {'annihilation': 22, 'time limit': 17, 'combat-ineffective': 3}, side check {'alpha': 18, 'beta': 24}, missing 0

Cell sizes: at 150 pts four battles per cell on the torpedo and swarm rows
(two seeds x two sides) and two on the gun-v-gun cells (seed 1; the 18
seed-2 battles were not fought, a few cells have four); at 200 pts two per
cell (seed 1). Almost every cell is the same both ways round and on every
seed: these battles are close to deterministic, so a cell is 0% or 100%, not
55%.

#### Why (from the battle events)

The torpedo fleets launch at the first checkpoint (t = 30 s, 500 km), their
first rounds arrive at t = 97-104 s and the last gun hull dies at t = 140-175 s.
Over the old-price battles of a pure torpedo fleet against a gun fleet (seeds
1-2, 150 pts):

| gun fleet (150 pts) | PD turrets | battles | rounds launched | hit | seeker blinded by PD | destroyed by PD | first hit | last kill | gun rounds fired | pts destroyed / lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| escort: 2 battleship + cruiser + 5 frigate | 18 | 4 | 64 | 31 (49%) | 12 (20%) | 0 | 101 s | 170 s | 0 | 150 / 0 |
| gunline: 4 destroyer + 2 cruiser + 2 frigate | 16 | 4 | 64 | 32 (51%) | 12 (19%) | 0 | 99 s | 162 s | 0 | 144 / 0 |
| battleline: 3 battleship + destroyer | 11 | 2 | 64 | 26 (41%) | 9 (14%) | 0 | 101 s | 173 s | 0 | 147 / 0 |

(8 corvettes, or a torpedo cruiser + 4 corvettes + frigate, against the three
gun fleets that carry no torpedo boat of their own. A hit delivers 26 GJ on
average; a battleship dies to the first wave, one second after it lands.)

- **The guns never fire.** Not one gun round was fired at a torpedo fleet in
  those battles. Against the smaller corvette forces below, which cannot kill
  everything, the battle runs to the time limit - still without a gun round,
  the fleets 290-355 km apart at the end. A 150-pt gun fleet spends its whole
  life under torpedo fire and none of it in gun range.
- **Point defence blinds rounds but does not stop them.** PD never destroyed
  a round in flight in any battle; it kills seekers. Against a thin salvo
  (two boats) it blinds every round and nothing dies; against a massed one it
  blinds a fraction, and rounds, blinded or not, still hit. An escort screen of 18
  turrets blinded 12-15 of a 64-round strike and the fleet was annihilated.
- **The salvo is a step function.** Corvettes against the 150-pt gun fleets,
  points destroyed (old prices) and the verdict:

  | corvettes | rounds | old price | escort (150) | combined (140) | flagship (149) | battleline (147) | gunline (144) | verdicts won |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 2 | 16 | 36 | 0 | 0 | 55 | 0 | 0 | 0 of 10 |
  | 3 | 24 | 54 | 101 | 122 | 135 | 147 | 76 | 9 of 10 |
  | 4 | 32 | 72 | 122 | 140 | 128 | 147 | 83 | 10 of 10 |
  | 5 | 40 | 90 | 129 | 140 | 149 | 147 | 137 | 10 of 10 |
  | 6 | 48 | 108 | 150 | 140 | 149 | 147 | 144 | 10 of 10 |
  | 8 | 64 | 144 | 150 | 140 | 149 | 147 | 144 | 10 of 10 |

  (points destroyed, mean of both side assignments; the fleet's size is in
  the header.)

  Two boats (6 rounds a wave) are blinded and beaten everywhere; three boats
  (9 a wave) destroy half to all of a 150-pt fleet. The same step sits at
  three boats against the 200-pt gun fleets. A round launched in a massed
  salvo destroys 2.2-6.1 pts of gun hull - 2.2 when the enemy is dead before
  the magazines are empty, 6.1 for three boats against a battle line, 3-4
  typically, at 150 and at 200 pts alike. At 0.040 it cost 0.72 pts, 2.25
  with its share of a corvette.
- **A dry torpedo fleet is never caught and rarely ruled out.** Corvettes pull
  3g; nothing but a frigate matches them. And the "fleet can no longer deal
  damage" rule seldom fires for them: blinded and fuel-exhausted rounds stay
  in the simulation as "in flight" (in the one battle inspected, 10 of 32 were
  still there at 600 s with every magazine empty), so the battle runs to the
  time limit and is scored on points - where killing a third of the enemy
  without loss wins.

#### The reprice

Only the magazine term moved: 0.040 -> **0.102** pts per stowed GJ (0.72 ->
1.84 per round). The rule, fixed before looking at any "after" result: a
torpedo hull should cost about what its magazine destroys, at the low end of
the measured range (gun fleets will close better once the captains' approach
is fixed, and a torpedo cruiser is one hull to lose), inside a band where no
price or roster flips.

| Hull | old | new | rounds | pts per round, all-in |
|------|----:|----:|-------:|----------------------:|
| corvette | 18 | 27 | 8 | 3.4 |
| cruiser_torpedo | 65 | 119 | 48 | 2.5 |

All gun hulls keep their prices. 150 pts now buy five corvettes (40 rounds)
instead of eight (64), or one torpedo cruiser and a corvette (56 rounds)
instead of a torpedo cruiser and four corvettes (80). The band is
0.0993-0.1060: inside it the corvette stays at 27 and the 150-pt torpedo
doctrine stays "torpedo cruiser + corvette".

#### After: 0.102 (corvette 27, cruiser_torpedo 119)

Cells between two fleets whose rosters the reprice did not change (`gunline`,
`battleline`, `escort` among themselves) are the same cached battles as in
the "before" tables, not new evidence.

**150 pts**, same seeds:
- **torpedo**: cruiser_torpedo, corvette (146 pts)
- **gunline**: 4x destroyer, 2x cruiser, 2x frigate (144 pts)
- **combined**: 2x battleship, 2x destroyer, corvette (149 pts)
- **flagship**: 2x dreadnought, frigate, corvette (144 pts)
- **battleline**: 3x battleship, destroyer (147 pts)
- **escort**: 2x battleship, cruiser, 5x frigate (150 pts)
- **swarm**: 5x corvette, 2x frigate (149 pts)

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 50% A | 0% A | 100% A | 100% A | 0% A | 58% | +0.35 |
| **gunline** | 0% A | - | 100% T | 100% T | 100% T | 25% T | 0% A | 39% | -0.29 |
| **combined** | 50% A | 0% T | - | 100% T | 50% T | 0% T | 0% A | 31% | -0.20 |
| **flagship** | 100% A | 0% T | 0% T | - | 0% T | 0% T | 0% A | 25% | -0.45 |
| **battleline** | 0% A | 0% T | 50% T | 100% T | - | 0% T | 0% A | 19% | -0.51 |
| **escort** | 0% A | 75% T | 100% T | 100% T | 100% T | - | 0% T | 50% | -0.30 |
| **swarm** | 100% A | 100% A | 100% A | 100% A | 100% A | 100% T | - | 100% | +0.87 |

n per cell: [2, 4], ended {'annihilation': 40, 'time limit': 26}, side check {'alpha': 31, 'beta': 31}, missing 18

**200 pts**, seed 1:
- **torpedo**: cruiser_torpedo, 2x corvette, 3x frigate (194 pts)
- **gunline**: 5x cruiser, 3x destroyer (199 pts)
- **combined**: 2x battleship, 3x destroyer, 2x corvette (194 pts)
- **flagship**: 2x dreadnought, 4x frigate, 2x corvette (192 pts)
- **battleline**: 4x battleship, destroyer, frigate (197 pts)
- **escort**: 4x battleship, 4x frigate (200 pts)
- **swarm**: 7x corvette, frigate (196 pts)

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 100% T | 100% T | 100% A | 100% A | 0% AT | 83% | +0.62 |
| **gunline** | 0% A | - | 100% T | 100% T | 100% T | 50% T | 0% A | 58% | -0.12 |
| **combined** | 0% T | 0% T | - | 100% T | 100% T | 0% T | 0% T | 33% | -0.12 |
| **flagship** | 0% T | 0% T | 0% T | - | 0% T | 0% T | 0% T | 0% | -0.61 |
| **battleline** | 0% A | 0% T | 0% T | 100% T | - | 50% T | 0% A | 25% | -0.36 |
| **escort** | 0% A | 50% T | 100% T | 100% T | 50% T | - | 0% T | 50% | -0.19 |
| **swarm** | 100% AT | 100% A | 100% T | 100% T | 100% A | 100% T | - | 100% | +0.79 |

n per cell: [2], ended {'annihilation': 11, 'time limit': 31}, side check {'alpha': 21, 'beta': 21}, missing 0

**Held-out seeds 3-4** (150 pts, torpedo and swarm rows; not looked at before the price was fixed):

Before:

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 100% A | 100% A | 100% A | 100% A | 0% A | 83% | +0.71 |
| **gunline** | 0% A | - | - | - | - | - | 0% AI | 0% | -0.98 |
| **combined** | 0% A | - | - | - | - | - | 0% A | 0% | -0.93 |
| **flagship** | 0% A | - | - | - | - | - | 0% A | 0% | -1.00 |
| **battleline** | 0% A | - | - | - | - | - | 0% AI | 0% | -0.93 |
| **escort** | 0% A | - | - | - | - | - | 0% AI | 0% | -0.99 |
| **swarm** | 100% A | 100% AI | 100% A | 100% A | 100% AI | 100% AI | - | 100% | +0.91 |

n per cell: [4], ended {'annihilation': 40, 'combat-ineffective': 4}, side check {'alpha': 22, 'beta': 22}, missing 0

After:

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 100% A | 38% A | 50% A | 100% A | 100% A | 0% A | 65% | +0.40 |
| **gunline** | 0% A | - | - | - | - | - | 0% A | 0% | -1.00 |
| **combined** | 62% A | - | - | - | - | - | 0% A | 31% | -0.45 |
| **flagship** | 50% A | - | - | - | - | - | 0% A | 25% | -0.48 |
| **battleline** | 0% A | - | - | - | - | - | 0% A | 0% | -1.00 |
| **escort** | 0% A | - | - | - | - | - | 0% T | 0% | -0.88 |
| **swarm** | 100% A | 100% A | 100% A | 100% A | 100% A | 100% T | - | 100% | +0.87 |

n per cell: [4], ended {'annihilation': 40, 'time limit': 4}, side check {'alpha': 21, 'beta': 20}, missing 0

#### What price did not fix, and cannot

The reprice helped at one budget out of three. At 150 pts the torpedo
doctrine fell from 83% of the field to 58% (65% on the held-out seeds) and now
loses to or splits with the mixed fleets that carry a corvette of their own;
150 pts no longer buy enough rounds to annihilate every gun fleet with rounds
to spare; and the fleet two LLM admirals drafted in August (five torpedo
cruisers at 150 pts) now costs 595. It does **not** meet the goal (no pairing
above ~65%, no fleet above ~60% of the field), and no price can on this
engine:

- **The corvette swarm still wins every pairing**, at 100, 150 and 200 pts,
  on the tuning and the held-out seeds. Against gun fleets three boats are
  enough to win and two never are (measured at 150 and 200 pts), so the
  verdict flips only when three no longer fit: above 50 pts per corvette at
  150 pts and 66 at 200 - past a battleship's 43, and a different price at
  every budget (by the same arithmetic 33 at 100 pts, not measured). A price
  moves the round count smoothly; the result is a step.
- **200 pts, the budget the price was not chosen on, did not improve.** The
  torpedo doctrine still fields 64 rounds (83% before and after) and the
  swarm 56 (92% -> 100%). What moved is the rest of the field: `combined`
  fell from 58% to 33% and `flagship` from 50% to 0%, because auto-draft now
  buys each of them two corvettes instead of three and four - below the
  three-boat step, so 54 pts of dead weight - while `gunline` rose from 42% to
  58%, `escort` from 17% to 50% and `battleline` from 8% to 25%. A corvette
  is now worth buying in threes or not at all; the mixed auto-draft doctrines
  have not been taught that.
- **100 pts, the default budget, did not improve either** (seeds 1-2, torpedo
  and swarm rows only). The old prices bought a torpedo cruiser + corvette +
  2 frigates (83%) or five corvettes + frigate (100%); the new ones buy three
  corvettes + 2 frigates under either doctrine, and that fleet wins all five
  pairings, four by annihilation. A torpedo cruiser (119) can no longer be
  drafted at 100 pts at all.
- **Cells are 0% or 100%.** The same archetype wins both side assignments on
  every seed in all but the torpedo-v-mixed cells, so "no pairing above 65%"
  is not something a price table can deliver; only the field score and the
  rock-paper-scissors shape are.
- **Gun fleets are not level among themselves either**, and were not
  repriced: at 150 pts every battle among the five fleets that are not
  torpedo fleets ran to the time limit, where (old prices, seed 1, those
  cells only) the frigate-screened `escort` took 100%, `gunline` 75%,
  `battleline` 38%, `combined` 25% and the dreadnought `flagship` 12% - two
  battles a cell, a lead, not a measurement. Time-limit verdicts
  are scored with `SHIP_CLASS_VALUES` (`src/llm/battle_runner.py`), which
  values a 7-pt frigate at 15 and a 55-pt dreadnought at 70.

What has to change is mechanical, not a price (none of it is in this file's
reach):

1. Point defence does not protect. It never destroys a round, and a blinded
   round still hits more often than not once the salvo is massed (18 turrets:
   20% of a 64-round strike blinded, 49% of it hit).
2. Gun ships never bring a torpedo boat to action: zero gun rounds in 600 s,
   290-355 km apart at the end.
3. A fleet with empty tubes stays "able to deal damage" while its duds drift
   (one battle inspected: 10 of 32 rounds still in the simulation at 600 s,
   magazines empty), so it is rarely ruled combat-ineffective - 10 of the
   534 battles fought for this section ended that way - and
   otherwise wins the time limit on points.
4. A full salvo launched at 500 km at the first checkpoint arrives intact;
   range costs a torpedo nothing.

Until then treat the torpedo prices as a floor, and the **A** cells above as
the robust part of the matrix: annihilation before the first gun round does
not depend on the captains' approach or on the scoring. The **T** cells do.

Re-run the tournament after any change to torpedo guidance, point defence,
the captains' approach or the time-limit scoring; the engine-source hash in
the cache key makes that automatic.

### 2026-11-15, round 2: mechanics, then 0.102 -> 0.080

Round 1 (above) was measured before the captains' round-3 fixes were merged:
gun ships no longer stall at zero closure, a withdrawal is actually flown, and
a disarmed ship that is alone surrenders. Round 2 re-ran the tournament on
that engine, changed two of the mechanics round 1 had named, and re-set the
dial under one new hard constraint.

**What changed in the engine (this lane):**

- **Spent rounds leave the battle** (`CombatSimulation._torpedo_is_spent`,
  `SPENT_TORPEDO_CULL_RANGE_M` = 300 km, `src/simulation.py`). A round that
  can no longer steer - seeker dead, tanks dry, or orphaned with an enemy
  still alive - is culled with the existing `left_battle` miss event once it
  is opening on every live hull beyond 300 km. It used to sit in
  `sim.torpedoes` for an hour of flight, which kept a fleet with empty tubes
  "able to deal damage".
- **The time-limit verdict values a hull at its draft price**
  (`ship_class_value`, `src/llm/battle_runner.py`). `SHIP_CLASS_VALUES`
  (frigate 15 for 7 pts, dreadnought 70 for 55) is now only the fallback for
  hulls the price table does not know. The tournament therefore keys a battle
  by the prices of the hulls in it.

**What did not change:** point defence and the cost of range. Both were
probed; point defence has a one-constant lever that was measured but not
applied, range has none (see "Still lopsided" below).

**The constraint:** every hull must be draftable at the default 100-point
budget. A torpedo cruiser at 119 was not. 0.080 is the dearest dial that keeps
it at 100 (band 0.0797-0.0808); the corvette lands on 24.

#### The slice after each step (150 pts, seed 1, both side assignments)

Torpedo and swarm rows against the five gun fleets; score of the row fleet and the number of battles.

| step | prices (CV / CT) | torpedo roster | torpedo score v gun fleets | swarm roster | swarm score v gun fleets |
|---|---|---|---:|---|---:|
| merged engine, before this round | 27 / 119 | CT + corvette | 20% of 10 | 5 corvettes + 2 frigates | 100% of 10 |
| + spent rounds culled | 27 / 119 | CT + corvette | 20% of 10 | 5 corvettes + 2 frigates | 100% of 10 |
| + verdict at draft price | 27 / 119 | CT + corvette | 20% of 10 | 5 corvettes + 2 frigates | 100% of 10 |
| + dial 0.080 | 24 / 100 | CT + 2 corvettes | 40% of 20 | 6 corvettes | 0% of 20 |

Neither mechanic moved a cell of this slice: 20 of its 22 battles end in
annihilation or surrender, which neither a dud round nor the scoring touches.
What decides it is something else, visible in explicit rosters fought on the
same engine (two battles a cell):

| torpedo fleet | gun hulls | score against the five gun fleets |
|---|---:|---:|
| 3 corvettes | 0 | 0% |
| 4 corvettes | 0 | 0% |
| 5 corvettes | 0 | 0% |
| 6 corvettes | 0 | 0% |
| torpedo cruiser + corvette | 0 | 20% |
| torpedo cruiser + 2 corvettes | 0 | 40% |
| 4 corvettes + 2 frigates | 2 | 80% |
| 4 corvettes + 4 frigates | 4 | 90% |
| 5 corvettes + 4 frigates | 4 | 100% |
| torpedo cruiser + 4 frigates | 4 | 100% |
| torpedo cruiser + corvette + 3 frigates | 3 | 100% |
| torpedo cruiser + 7 frigates | 7 | 100% |

A fleet of torpedo hulls alone now loses: at the checkpoint after its tubes
are empty every captain finds "all weapons out, no armed ship left to
protect" and surrenders - while the salvo is still in flight (six corvettes
v the gunline: 36 of 48 rounds in the air at the surrender, t = 120 s). Give
the same boats two frigates and the fleet stays in the fight, the rounds
land, and it wins. In round 1, before the round-3 captains, three bare
corvettes won 9 of 10. So the `torpedo` and `swarm` archetypes, whose recipes
buy no gun hull at these prices, now UNDERSTATE torpedoes, and the tournament
gained an eighth archetype that cannot: `screen`, one torpedo cruiser behind a
frigate screen.

#### Final matrix: 0.080 (corvette 24, cruiser_torpedo 100), 150 pts

**Tuning seeds 1-2** on the torpedo, swarm and screen rows (four battles a cell), seed 1 on the gun-v-gun cells (two):
- **torpedo**: cruiser_torpedo, 2x corvette (148 pts)
- **gunline**: 4x destroyer, 2x cruiser, 2x frigate (144 pts)
- **combined**: 2x battleship, 2x destroyer, corvette (146 pts)
- **flagship**: 2x dreadnought, 2x frigate, corvette (148 pts)
- **battleline**: 3x battleship, destroyer (147 pts)
- **escort**: 2x battleship, cruiser, 5x frigate (150 pts)
- **swarm**: 6x corvette (144 pts)
- **screen**: cruiser_torpedo, 7x frigate (149 pts)

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | screen | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 0% A | 0% A | 0% A | 100% AI | 100% A | 0% A | 0% A | 29% | -0.13 |
| **gunline** | 100% A | - | 100% T | 100% I | 100% T | 25% T | 100% A | 0% AI | 72% | +0.15 |
| **combined** | 100% A | 0% T | - | 100% T | 75% T | 25% T | 100% A | 0% A | 60% | +0.03 |
| **flagship** | 100% A | 0% I | 0% T | - | 0% AT | 0% IT | 100% A | 0% A | 40% | -0.16 |
| **battleline** | 0% AI | 0% T | 25% T | 100% AT | - | 0% T | 100% A | 0% A | 32% | -0.32 |
| **escort** | 0% A | 75% T | 75% T | 100% IT | 100% T | - | 100% A | 0% A | 55% | -0.14 |
| **swarm** | 100% A | 0% A | 0% A | 0% A | 0% A | 0% A | - | 0% A | 14% | -0.43 |
| **screen** | 100% A | 100% AI | 100% A | 100% A | 100% A | 100% A | 100% A | - | 100% | +0.88 |

n per cell: [2, 4], ended {'annihilation': 69, 'combat-ineffective': 7, 'time limit': 16}, side check {'beta': 43, 'alpha': 46}, missing 20

**Held-out seeds 3-4** (torpedo, swarm and screen rows):

| row vs column | torp | gunl | comb | flag | bline | esc | swarm | screen | field | tonnage margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **torpedo** | - | 0% A | 25% A | 0% A | 100% A | 100% A | 0% A | 0% A | 32% | -0.09 |
| **gunline** | 100% A | - | - | - | - | - | 100% A | 0% AIT | 67% | +0.02 |
| **combined** | 75% A | - | - | - | - | - | 100% A | 0% A | 58% | +0.05 |
| **flagship** | 100% A | - | - | - | - | - | 100% A | 0% A | 67% | +0.09 |
| **battleline** | 0% A | - | - | - | - | - | 100% A | 0% A | 33% | -0.57 |
| **escort** | 0% A | - | - | - | - | - | 100% A | 0% A | 33% | -0.41 |
| **swarm** | 100% A | 0% A | 0% A | 0% A | 0% A | 0% A | - | 0% A | 14% | -0.44 |
| **screen** | 100% A | 100% AIT | 100% A | 100% A | 100% A | 100% A | 100% A | - | 100% | +0.87 |

n per cell: [4], ended {'annihilation': 70, 'time limit': 1, 'combat-ineffective': 1}, side check {'beta': 35, 'alpha': 35}, missing 0

**Other budgets** (seed 1, torpedo / swarm / screen rows, two battles a cell; field score of the row fleet):

| budget | torpedo | swarm | screen |
|---:|---|---|---|
| 100 | cruiser_torpedo: 50% | 4 corvette: 43% | cruiser_torpedo: 50% |
| 200 | cruiser_torpedo + 3 corvette + 4 frigate: 100% | 8 corvette: 0% | cruiser_torpedo + 4 destroyer + 3 frigate: 86% |

#### Still lopsided, and why

The final table is **not balanced**, and the dial is not what keeps it so:

- **`screen` wins every battle**: one torpedo cruiser and seven frigates
  (149 pts) went 28-0-0 on the tuning seeds and 28-0-0 on the held-out ones,
  almost all by annihilation. At the round-1 price (119) the same idea still
  fits in 150 pts with four frigates and still won 10 of 10. No dial under
  the 100-point constraint, or above it, changes that.
- **It grows with the budget.** At 200 pts the torpedo doctrine itself
  (torpedo cruiser + 3 corvettes + 4 frigates) takes 100% of its cells and
  `screen` 86%; at the default 100 pts, where a torpedo cruiser is the whole
  fleet, it takes 50% and four corvettes 43% (seed 1, two battles a cell).
  The default budget is the one place the table is close to level.
- **`torpedo` and `swarm` look weak for the wrong reason.** Their recipes buy
  no gun hull at these prices, and a fleet of torpedo hulls alone strikes its
  colours the checkpoint after its tubes are empty - with its salvo still in
  the air. Six corvettes v the gunline: all 48 rounds launched by t = 90 s,
  the fleet surrendered at t = 120 s ("all weapons out, no armed ship left to
  protect", `src/llm/heuristic_captain.py`) with 36 of them still in flight
  and two cruisers dead; no corvette was hit. Any drafter - human or LLM -
  who adds two frigates gets the fleet in the table above. Read these rows
  as "torpedo hulls alone, under that surrender rule", not as "torpedoes".
- **Cells are still 0% or 100%**, the same in both side assignments and on
  held-out seeds.
- **Gun fleets among themselves** (seed 1, two battles a cell) are in the
  matrix above; the verdicts are now scored at draft prices.

Why the two mechanics that would matter were not changed in this round:

1. **Point defence.** With the 50 MJ seeker kill a blind comes too late to
   be a save. Against the `screen` fleet's 12-round waves the gunline's 16
   turrets blinded 33 of 48 rounds and the escort's 18 blinded 39 of 48 (one
   battle each) - and 19 and 20 of those blinded rounds hit anyway, 70-77% of
   all the hits those fleets took. Against five corvettes + two frigates 20
   of 40 were blinded and 15 of the 20 hit, nine within 2 s of the blind and
   six 5-7 s after it: a round blinded that close flies straight into a hull
   that has no time to leave its line. (Why hulls with 5-7 s do not leave it
   was not traced; the automatic evasion against a guided round is a steady
   burn, which is a guess at the cause, not a finding.)

   **The lever is the seeker-kill energy** (`TORPEDO_ELECTRONICS_THRESHOLD_J`,
   `src/pointdefense.py`). Probed by overriding it per round, one battle per
   cell, five corvettes + two frigates (150 pts) attacking:

   | seeker kill | v gunline (16 turrets): blinded / hit of 40, hulls lost, winner | v battleline (11 turrets): blinded / hit of 40, hulls lost, winner |
   |---:|---|---|
   | 50 MJ (now) | 20 / 26, 6 of 8, torpedo fleet | 11 / 23, 4 of 4, torpedo fleet |
   | 33 MJ | 39 / 17, 3 of 8, **gunline** | 13 / 22, 4 of 4, torpedo fleet |
   | 25 MJ | 40 / 6, 1 of 8, **gunline** | 26 / 23, 4 of 4, torpedo fleet |
   | 17 MJ | 40 / 2, 0 of 8, **gunline** | 36 / 24, 4 of 4, torpedo fleet |

   At 25 MJ ("one turret blinds ~10 km/s of closure", two turrets per round
   at the floor instead of three) every round is blinded 6-12 s out and a
   turret-rich fleet of nimble hulls survives the strike and wins, while a
   battle line - few turrets, 1 g - is still hit by the blinded rounds and
   dies. That is a counter that exists and a rock-paper-scissors shape, from
   one named constant. It was NOT changed here: the 50 MJ figure and its
   consequences ("~5 km/s per turret", "3 turrets per round", "<=46 of the
   50 MJ needed") are written into the captain and admiral doctrine and
   pinned by `tests/test_doctrine_accuracy.py` and the PD suites, and a
   re-pin of that size did not fit what was left of the round. It is the
   first thing to do next, followed by a re-fit of the dial.
   A structural kill by continued dwell stays out of reach either way: a
   turret delivers 17.8 MJ to a round crossing the 250 km envelope at
   14.5 km/s (21.5 MJ at the 12 km/s floor) and the kill is 1 GJ.
2. **Range.** A 500 km shot arrives at 13.8-15.6 km/s after 59-71 s; the
   26 GJ per hit is exactly 1/2 x 250 kg x v^2, the game's own rule. Burning
   all the way, a round keeps gaining speed out to 832 km (v^2 / 2a at 12 g
   and 14 km/s), and steering against a 1.5 g target for 70 s costs about
   1 km/s of the 14. Range is a bonus to a torpedo, not a cost. Changing that
   means changing the round (delta-v, acceleration: weapon data) or adding a
   seeker-range rule; neither is a named-constant change.

Until one of those lands, torpedo hulls are under-priced at any price that
keeps them draftable, and the honest reading of the cost table is: the gun
hulls are priced against each other, the torpedo hulls are capped.

## Cheap AI captains (src/llm/heuristic_captain.py)

Ships whose config model is the sentinel `"heuristic"` get a rule-based
captain (zero tokens). It executes decisions through the same
`_execute_tool` path as an LLM captain, so recordings and command validation
are identical. Behavior:

- Obeys the admiral's structured order fields (`suggested_target`,
  `torpedo_salvo`, `torpedo_target`) and simple order-text keywords
  (evade / brake / padlock / kite / intercept). A heading or structured
  maneuver order (standoff INTERCEPT, HOLD_RANGE, toward_point/ship) is flown
  by the runner and the captain keeps it; an order whose target did not
  resolve was not applied, and the captain flies its own maneuver instead.
- Evades incoming guided torpedoes (ETA < 75 s) at full throttle, unless the
  admiral's applied order carries the evade overlay (that already dodges).
- Gunships close to effective range, brake off a blind overshoot, and
  coast-track through the knife-fight; pure torpedo hulls keep a 250 km
  standoff and launch probing salvos while closing.
- Manages heat against the engine's ladder (guns hold at 90%, radiators
  forced out at 100%, reactor meltdown at 150%): extends radiators at 65%
  heat (35% when no enemy is within 800 km and no slug is inbound), buttons
  up under torpedo fire unless past 85%, and above 75% eases the burn below
  what its (possibly damaged) radiators can reject.

Drafted ships are named by class nickname (Dart, Ward, Falchion, Lancer,
Bastion, Harpoon, Sovereign, Colossus, Breaker) - "TIS Falchion-2" beats
"TIS Heuristic-2" in every log and replay.

`--captain-model <id>` swaps the heuristic for a real (cheap) LLM captain
per ship instead. `--alpha-captain-model` / `--beta-captain-model`
override per side - set each side's captains to its own admiral's model
and the admirals fly their fleets directly (the Kimi-vs-GPT and
DeepSeek-vs-Sonnet recordings use this).

## Outputs

- Normal battle recording (plus `--trace` for the 3D replay viewer).
- `<recording>.draft.json` sidecar: budgets, points spent, rosters,
  formation names/offsets and both admirals' rationales.
- Admiral-vision frames under `data/recordings/vision/<timestamp>/` when
  vision is active.
