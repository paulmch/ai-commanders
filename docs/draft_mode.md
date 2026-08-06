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

## Point costs (src/llm/fleet_draft.py)

| Hull | Cost | Notes |
|------|-----:|-------|
| frigate | 8 | thin-skinned escort gunboat |
| corvette | 10 | 3g torpedo boat, 8 rounds |
| destroyer | 16 | workhorse gun platform |
| battlecruiser | 22 | cruiser guns, lighter armor |
| cruiser | 26 | heavy combatant |
| cruiser_torpedo | 30 | 4-launcher saturation striker |
| battleship | 40 | line of battle |
| dreadnought | 55 | mobile fortress |
| dreadnought_siege | 58 | 7.25 GJ siege spinal |

Hand-tuned, anchored on wet mass, adjusted for capability. 100 points buys a
dreadnought plus escorts, six destroyers, or a torpedo wolfpack.

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
per ship instead.

## Outputs

- Normal battle recording (plus `--trace` for the 3D replay viewer).
- `<recording>.draft.json` sidecar: budgets, points spent, rosters,
  formation names/offsets and both admirals' rationales.
- Admiral-vision frames under `data/recordings/vision/<timestamp>/` when
  vision is active.
