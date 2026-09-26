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

## Torpedo retargeting

Rounds whose target dies mid-flight acquire a replacement on their own
(live seeker + delta-v above the terminal reserve). The round measures its
seeker health to pick an imperative: fresh seekers maximize fuel at
intercept (impact energy), PD-singed seekers race the fastest intercept
before going blind. Overkill concentration therefore chains down enemy
formations - Kimi K3's "survivors retarget onto the next destroyer
automatically" wave doctrine (2026-08-06 recording) killed three
destroyers with one 16-round salvo. Formation counterplay: spacing units
beyond an orphan's steering envelope turns enemy overkill back into waste.

## Point costs (src/llm/fleet_draft.py)

Costs are **computed from `data/fleet_ships.json`**, not hand-typed - change a
hull's armour or magazine and its price follows. A hull pays for:

| Term | Rate | What it buys |
|------|------|--------------|
| armour | 1 pt / 55 t | how hard it is to kill |
| PD turret | 1 pt each | active defence |
| gun mounts | 8 pts per GJ/s | sustained energy throughput |
| torpedo rounds | 0.040 pts per stowed GJ | **magazine depth x per-round yield** |

Gun throughput is `salvo_size * kinetic_energy_gj / (cooldown_s +
(salvo_size - 1) * intra_salvo_cooldown_s)`. TI Mk3 timing gives batteries
5 shots per 60s, spinals 5 per 72s, and siege coilers 3 per 60s, retaining
our per-shot damage. The Cost column below includes this timing update.

| Hull | Cost | 2026-08-13 | Original | Notes |
|------|-----:|-----:|----:|-------|
| frigate | 7 | 7 | 8 | thin-skinned escort gunboat |
| corvette | 18 | 17 | 10 | 3g torpedo boat, 8 rounds (144 GJ) |
| destroyer | 18 | 18 | 16 | workhorse gun platform |
| battlecruiser | 25 | 24 | 22 | cruiser guns, lighter armor |
| cruiser | 29 | 28 | 26 | heavy combatant |
| battleship | 43 | 42 | 40 | line of battle |
| dreadnought | 55 | 53 | 55 | mobile fortress |
| dreadnought_siege | 56 | 54 | 58 | 7.25 GJ siege spinal |
| cruiser_torpedo | 65 | 58 | 30 | 4 launchers, 48 rounds (864 GJ) |

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

## Cheap AI captains (src/llm/heuristic_captain.py)

Ships whose config model is the sentinel `"heuristic"` get a rule-based
captain (zero tokens). It executes decisions through the same
`_execute_tool` path as an LLM captain, so recordings and command validation
are identical. Behavior:

- Obeys the admiral's structured order fields (`suggested_target`,
  `torpedo_salvo`, `torpedo_target`) and simple order-text keywords
  (evade / brake / padlock / kite / intercept).
- Evades incoming guided torpedoes (ETA < 75 s) at full throttle.
- Gunships close to effective range, brake off a blind overshoot, and
  coast-track through the knife-fight; pure torpedo hulls keep a 250 km
  standoff and launch probing salvos while closing.
- Extends radiators above 65% heat, buttons up under torpedo fire.

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
