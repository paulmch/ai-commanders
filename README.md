# AI Commanders

A Terra Invicta-inspired space battle simulator where **LLM-controlled captains and admirals** command fleets in tactical combat. Watch AI models coordinate fleet maneuvers, issue orders, and trash-talk while trying to blow each other up with coilguns and torpedoes.

## What is this?

Three modes of warfare:

**Single Combat**: Two AI captains. Two ships. 500km of vacuum. Each captain makes tactical decisions every 30 seconds and can message their opponent.

**Fleet Battles**: Two AI admirals commanding fleets of multiple ships. Admirals issue strategic orders (optionally with a rendered tactical-plot image every checkpoint - "admiral vision"), captains execute tactics, and everyone can communicate. Coordinate focus fire, flanking maneuvers, and combined arms.

**Draft Battles**: Admirals get a point budget, buy their own fleets from a costed catalog, place them in 3D formations, and command them - with rule-based AI captains, cheap LLM captains, or by flying their own ships directly.

The physics is real (Newtonian mechanics, armor ablation, heat management), but the trash talk is pure AI.

## Attribution

This project is inspired by and based on **Terra Invicta**. The raw ship data, weapon mechanics, armor systems, and physics parameters are translated from Terra Invicta's game files. The code implementation, LLM integration, and battle simulation architecture are our own interpretation of those mechanics.

Terra Invicta is developed by Pavonis Interactive and published by Hooded Horse.

## Quick Start

```bash
# Install dependencies
uv sync

# Add your OpenRouter API key to .env file
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
```

### Single Battle (1v1)

```bash
# Quick destroyer duel
uv run python scripts/run_llm_battle.py -v

# Customize ships and models
uv run python scripts/run_llm_battle.py \
    --alpha-model openrouter/anthropic/claude-opus-5 \
    --beta-model openrouter/x-ai/grok-4.20 \
    --alpha-ship-type cruiser \
    --beta-ship-type destroyer \
    --distance 400 \
    -v
```

### Fleet Battle (Multi-Ship with Admirals)

```bash
# Run a fleet engagement
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config_claude_vs_gemini.json \
    -v

# Unlimited mode - fight until destruction
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config_claude_vs_gemini.json \
    --unlimited \
    -v

# Give admirals a rendered tactical-plot image every checkpoint
# (vision-capable admiral models + matplotlib; see docs/admiral_vision.md)
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config_claude_vs_gemini.json \
    --admiral-vision -v
```

### Draft Mode (Admirals Build Their Own Fleets)

```bash
# Two LLM admirals draft and fight (rule-based captains, vision on)
uv run python scripts/run_draft_battle.py \
    --alpha-admiral anthropic/claude-opus-5 \
    --beta-admiral google/gemini-3.5-pro --trace

# Fully offline demo - deterministic auto-drafts, zero LLM calls
uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace

# Each admiral flies its own ships directly
uv run python scripts/run_draft_battle.py \
    --alpha-admiral moonshotai/kimi-k3 --alpha-captain-model moonshotai/kimi-k3 \
    --beta-admiral openai/gpt-5.6-terra --beta-captain-model openai/gpt-5.6-terra
```

Admirals draft with a full tactical briefing - catalog stats, calibrated
weapon effects, torpedo flight profile, counterplay doctrine. Details in
`docs/draft_mode.md`.

## 3D Battle Visualizer

A Three.js-based tactical replay viewer for watching recorded battles in full 3D.

### Running the Visualizer

```bash
cd visualizer
npm install
npm run dev
```

Open http://localhost:5173 and load a battle recording JSON file.

### Features

- **Expanse-style hulls, one design per class**: nine distinct ships built from
  drums, trusses, stepped towers, slab armor, turrets, VLS hatch banks and a
  siege spinal barrel - readable silhouettes from corvette to dreadnought,
  with folding radiator wings that glow when extended
- **Fusion torch plumes**: continuous plasma cones (white core, blue sheath)
  scaled to each class's engine bell count and throttle - no strobing
- **Torpedo rendering**: guided rounds with faction-colored trails and thrust
  plumes; blinded rounds tumble dark and coast; retargeting rounds ping cyan
  and draw a fading lock line to their new victim
- **Continuous PD beams**: dwell lasers anchored ship-to-target every frame
- **Impact effects**: shockwave rings and particle bursts on hits, distinct
  markers for misses, burnouts, and seeker kills
- **Two-stage ship destruction**: non-reactor kills drift dark with hull
  fires popping before the reactor breach (immediate detonation when the
  reactor itself was the killing blow) - blinding flash, plasma sphere,
  shockwave, 50k-particle GPU debris cloud, lingering aftermath
- **Camera modes**: Free orbit, follow ship, orbit selected ship
- **Ship telemetry**: Hull, per-facing armor, modules, target, maneuver status
- **Timeline scrubbing**: Jump to any point, adjustable playback speed (0.25x-8x),
  event markers on the timeline, and a 20s epilogue so end-of-battle
  destructions play out
- **Time input**: Enter exact timestamps (MM:SS or seconds) to jump directly
- **URL loading**: `?recording=/recordings/name.json` autoloads a staged recording

### Controls

| Control | Action |
|---------|--------|
| Space | Play/Pause |
| ← / → | Seek ±5 seconds |
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| Click ship list | Select & focus ship |

## Battle Modes

### Single Combat Mode

Classic 1v1 duel between two AI captains. Each captain:
- Chooses their own combat personality
- Makes tactical decisions every 30 seconds
- Can send messages to their opponent
- Can propose draws or surrender

```bash
uv run python scripts/run_llm_battle.py \
    --alpha-model MODEL \
    --beta-model MODEL \
    --alpha-ship-type destroyer \
    --beta-ship-type destroyer \
    -v
```

### MCP Battle Mode (Human vs LLM)

**NEW**: Control your fleet using any MCP-compatible client (Claude Code, Cursor, Copilot, OpenCode, etc.). Two MCP servers (`alpha` and `beta`) can control either or both fleets - no OpenRouter API key needed for MCP-controlled sides.

**Battle configurations:**
- **Human vs LLM**: You control alpha via MCP, beta runs on OpenRouter AI
- **LLM vs LLM**: Both sides use OpenRouter (classic mode)
- **Human vs Human**: Both sides connect via MCP (no API costs!)
- **LLM vs LLM (MCP)**: Two AI agents connect via MCP servers

```bash
# Start a battle - you control alpha, Gemini controls beta
uv run python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json
```

Connect any MCP client to the server and command your fleet:
- **Full tactical awareness**: Ship positions, velocities, armor status, weapons cooldowns
- **Direct ship control**: Set maneuvers (INTERCEPT, EVASIVE, PADLOCK), weapons modes, targets
- **Real-time combat**: Issue orders, signal ready, watch the battle unfold
- **Trash talk**: Send messages to the enemy admiral (max 3 per turn)

See [MCP Battle Guide](#mcp-battle-guide) below for full details.

### Fleet Battle Mode

Multi-ship engagements with hierarchical command:

**Admirals** (one per fleet):
- See dual-snapshot tactical view (T-15s and T=0), plus an optional rendered
  battle image per checkpoint (top-down plot + 3D panel; `--admiral-vision`
  or `"vision": true`, frames saved under `data/recordings/vision/`)
- Issue strategic directives to entire fleet
- Send specific orders to each captain
- Can negotiate with enemy admiral
- Propose/accept fleet-wide draws

**Captains** (one per ship):
- Receive and acknowledge admiral orders
- Can discuss orders with their admiral (up to 2 exchanges)
- Execute tactical maneuvers
- Can message enemy captains

```bash
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config.json \
    -v
```

## Fleet Configuration

Fleet battles use JSON configuration files:

```json
{
  "battle_name": "Fleet Engagement: Claude vs Gemini",
  "time_limit_s": 1200,
  "decision_interval_s": 30.0,
  "initial_distance_km": 400,
  "alpha_fleet": {
    "admiral": "openrouter/anthropic/claude-opus-5",
    "ships": [
      {"ship_type": "destroyer", "model": "openrouter/anthropic/claude-sonnet-5"},
      {"ship_type": "destroyer", "model": "openrouter/anthropic/claude-sonnet-5"},
      {"ship_type": "dreadnought", "model": "openrouter/anthropic/claude-sonnet-5"}
    ]
  },
  "beta_fleet": {
    "admiral": "openrouter/google/gemini-3.5-flash",
    "ships": [
      {"ship_type": "destroyer", "model": "openrouter/google/gemini-3.5-flash"},
      {"ship_type": "destroyer", "model": "openrouter/google/gemini-3.5-flash"},
      {"ship_type": "dreadnought", "model": "openrouter/google/gemini-3.5-flash"}
    ]
  }
}
```

**Configuration Options:**
| Field | Description |
|-------|-------------|
| `battle_name` | Display name for the battle |
| `time_limit_s` | Maximum battle duration in seconds |
| `decision_interval_s` | Time between checkpoints (default: 30) |
| `initial_distance_km` | Starting distance between fleets |
| `admiral` | Model for fleet admiral (or `null` for no admiral) |
| `ships` | Array of ship configurations |
| `ship_type` | Ship class from `data/fleet_ships.json` (corvette, frigate, destroyer, cruiser, cruiser_torpedo, battlecruiser, battleship, dreadnought, dreadnought_siege) |
| `model` | OpenRouter model ID for the captain |

## CLI Reference

```
uv run python scripts/run_llm_battle.py [OPTIONS]
```

### Model & Ship Options

| Option | Description | Default |
|--------|-------------|---------|
| `--alpha-model` | OpenRouter model for Alpha | openrouter/anthropic/claude-sonnet-5 |
| `--beta-model` | OpenRouter model for Beta | openrouter/anthropic/claude-sonnet-5 |
| `--alpha-ship-type` | Ship class for Alpha | destroyer |
| `--beta-ship-type` | Ship class for Beta | destroyer |
| `--alpha-name` | Captain name for Alpha | Commander Chen |
| `--beta-name` | Captain name for Beta | Captain Volkov |
| `--alpha-ship` | Ship name for Alpha | TIS Relentless |
| `--beta-ship` | Ship name for Beta | HFS Determination |

Ship type choices are derived from `data/fleet_ships.json`: `corvette`, `frigate`,
`destroyer`, `cruiser`, `cruiser_torpedo`, `battlecruiser`, `battleship`,
`dreadnought`, and the scenario-specific `dreadnought_siege`.

### Battle Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fleet-config FILE` | JSON fleet config (enables fleet mode) | None |
| `--distance KM` | Initial distance in km | 500 |
| `--max-checkpoints N` | Max decision points | 40 |
| `--time-limit SEC` | Time limit in seconds | 1200 |
| `--unlimited` | Fight until destruction/surrender/draw | False |
| `--trace` | Record detailed sim trace (large files!) | False |
| `--admiral-vision` | Attach rendered tactical-plot images to admiral checkpoints (fleet mode, vision-capable models) | False |

Draft mode has its own CLI - `uv run python scripts/run_draft_battle.py --help`
(budget, max ships, per-side admiral and captain models, auto-draft, vision).

### Output Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed battle output |
| `-q, --quiet` | Only show final result |
| `--no-personality-selection` | Skip personality phase |

### Examples

```bash
# Quick 1v1 duel
uv run python scripts/run_llm_battle.py -v

# Cruiser vs Destroyer
uv run python scripts/run_llm_battle.py \
    --alpha-ship-type cruiser \
    --beta-ship-type destroyer \
    --distance 300 -v

# Fleet battle with recording
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config_claude_vs_gemini.json \
    --trace -v

# Unlimited fleet battle (fight to the death)
uv run python scripts/run_llm_battle.py \
    --fleet-config data/fleet_config.json \
    --unlimited -v
```

## MCP Battle Guide

Control fleets directly using the Model Context Protocol. Any MCP-compatible client works: Claude Code, Cursor, GitHub Copilot, OpenCode, or custom agents.

### Architecture Overview

Each side runs its own MCP server process, but **both connect to the same battle
HTTP API on a single port** (8765 by default) - the faction is chosen with
`--faction`, not with a different port. This allows **any combination** of
human/AI control:

```
┌─────────────────┐                         ┌─────────────────┐
│  MCP Client     │                         │  MCP Client     │
│  (Alpha Fleet)  │                         │  (Beta Fleet)   │
└────────┬────────┘                         └────────┬────────┘
         │ MCP (stdio)                               │ MCP (stdio)
         ▼                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│  mcp_server     │                         │  mcp_server     │
│ --faction alpha │                         │ --faction beta  │
└────────┬────────┘                         └────────┬────────┘
         │  HTTP :8765                  HTTP :8765   │
         └─────────────►┌─────────────────┐◄─────────┘
                        │  Battle Runner  │
                        │  + Simulation   │
                        └─────────────────┘
```

**No OpenRouter API key needed** for MCP-controlled fleets - only for AI-controlled opponents.

### Quick Start

**1. Configure your MCP client** (example for Claude Code `.mcp.json`):

```json
{
  "mcpServers": {
    "ai-commanders-alpha": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.llm.mcp_server", "--faction", "alpha", "--http", "http://localhost:8765"]
    },
    "ai-commanders-beta": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.llm.mcp_server", "--faction", "beta", "--http", "http://localhost:8765"]
    }
  }
}
```

**2. Start a battle**:

```bash
# Human (alpha) vs AI (beta)
uv run python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json

# Human vs Human (both MCP)
uv run python scripts/mcp_battle.py --config data/fleet_config_mcp_vs_mcp.json
```

**3. Connect your MCP client** and issue commands:

```
get_battle_state()     # See full tactical picture
set_maneuver(ship_id="alpha_1", maneuver_type="INTERCEPT", target_id="beta_1", throttle=1.0)
set_weapons_order(ship_id="alpha_1", spinal_mode="FIRE_IMMEDIATE", turret_mode="FIRE_IMMEDIATE")
ready()                # Signal turn complete
```

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `get_battle_state` | Full tactical snapshot (ships, projectiles, chat) |
| `get_ship_status` | Detailed status for one friendly ship |
| `set_maneuver` | Movement: INTERCEPT, EVASIVE, BRAKE, MAINTAIN, PADLOCK, HEADING |
| `set_weapons_order` | Firing mode: FIRE_IMMEDIATE, FIRE_WHEN_OPTIMAL, HOLD_FIRE, FREE_FIRE |
| `set_primary_target` | Set which enemy a ship engages |
| `launch_torpedo` | Fire torpedoes at target |
| `set_radiators` | Extend/retract radiators for heat management |
| `send_message` | Trash talk the enemy (max 3/turn) |
| `propose_fleet_draw` | Propose ending the battle |
| `accept_fleet_draw` | Accept enemy's draw proposal |
| `surrender_fleet` | Give up |
| `ready` | Signal all commands issued, advance simulation |
| `battle_plot` | ASCII tactical map (xy/xz/yz projections) |

### Maneuver Types

| Maneuver | Behavior |
|----------|----------|
| `INTERCEPT` | Burn toward target at specified throttle |
| `EVASIVE` | Threat-aware dodging: wobbles vs guns, auto-RUNs from guided torpedoes, auto-PRESENTs thickest armor in the final seconds (needs throttle!) |
| `BRAKE` | Flip and decelerate |
| `MAINTAIN` | Coast at current velocity |
| `PADLOCK` | Coast while keeping nose pointed at target (good for shooting) |
| `HEADING` | Fly in specific 3D direction |

### Key Concepts

**Orders reset every turn!** You must re-issue maneuver and weapons orders each checkpoint. Ships default to MAINTAIN (coasting) if you don't command them.

**Visibility**: Currently full information for both sides. You see complete friendly status, and for enemies `get_battle_state` returns exact hull %, per-facing armor damage, shot/hit counters and which of your ships they are targeting - not a sensor-limited estimate. The tactical map drawn by `battle_plot` omits enemy hull, but the underlying state does not, so treat the map as a display choice rather than a fog-of-war model. (Implementing real fog of war is an open design decision - see CONTRIBUTING.)

**Turn flow**:
1. Simulation runs 30 seconds
2. You receive battle state via `get_battle_state()`
3. Issue commands for each ship
4. Call `ready()` to advance
5. Repeat until victory

### Example Fleet Config (MCP vs AI)

```json
{
  "battle_name": "MCP vs Grok Fleet Battle",
  "time_limit_s": 600,
  "decision_interval_s": 30,
  "initial_distance_km": 300,
  "alpha_fleet": {
    "mcp": {
      "enabled": true,
      "transport": "http",
      "http_port": 8765,
      "name": "Claude Commander"
    },
    "ships": [
      {"ship_id": "alpha_1", "ship_type": "destroyer", "model": "mcp"},
      {"ship_id": "alpha_2", "ship_type": "destroyer", "model": "mcp"}
    ]
  },
  "beta_fleet": {
    "admiral": {"model": "openrouter/x-ai/grok-4.20", "name": "Admiral Grok"},
    "ships": [
      {"ship_id": "beta_1", "ship_type": "destroyer", "model": "openrouter/x-ai/grok-4.20"},
      {"ship_id": "beta_2", "ship_type": "destroyer", "model": "openrouter/x-ai/grok-4.20"}
    ]
  }
}
```

### Battle Results: Human (MCP) vs Gemini (2026-01-17)

| Fleet | Controller | Ships | Result |
|-------|------------|-------|--------|
| Alpha | Human via MCP (with Claude Opus 4.5 as copilot) | 2 Destroyers, 1 Dreadnought | **VICTORY** (3/3 ships) |
| Beta | Gemini 3 Pro (OpenRouter) | 2 Destroyers, 1 Dreadnought | Eliminated (0/3 ships) |

- **Duration**: 990s (16.5 minutes)
- **Outcome**: Beta fleet eliminated
- **Notable**: Gemini used smart evasive tactics while focusing fire on alpha_1, but was overwhelmed by coordinated intercept + fire orders
- **Key lesson**: Remember to set throttle on EVASIVE maneuvers and re-issue orders every turn!

## Ship Classes

| Ship | Accel | 90° Turn | Armor (N/L/T) | Weapons | Role |
|------|-------|----------|---------------|---------|------|
| Corvette | 3.0g | 12s | 212/36/42 cm | Torpedoes + PD | Torpedo boat |
| Frigate | 3.0g | 15s | 71/12/14 cm | Coilgun + PD | Fast attack |
| Destroyer | 2.0g | 21s | 151/26/30 cm | Spinal + coilgun | Balanced combatant |
| Cruiser | 1.5g | 28s | 240/41/48 cm | Spinal + coilguns | Heavy firepower |
| Torpedo Cruiser | 1.5g | 28s | 241/41/48 cm | 4x torpedo launchers + 4 PD | Saturation salvos |
| Battlecruiser | 1.5g | 28s | 177/30/35 cm | Spinal + coilguns | Fast capital |
| Battleship | 1.0g | 37s | 262/45/52 cm | Spinal + 3 heavy coilguns | Line combat |
| Dreadnought | 0.75g | 50s | 251/43/50 cm | Spinal + 5 heavy coilguns | Fleet anchor |

The `dreadnought_siege` variant swaps the spinal for a 7.2 GJ Heavy Siege Coiler.

Armor thickness is derived from armor mass over facing area, so a small hull carrying
heavy armor ends up *thicker* than a larger one: the corvette is a torpedo boat that must
survive its attack run, so it carries a 5.8x
thicker nose than flank. The frigate has the same hull and acceleration but fights at
range, and carries less than half the armor.

The torpedo cruiser (`cruiser_torpedo`) carries no guns at all: 4 launchers with 12
rounds each let it fire an 8-torpedo salvo every 30s decision - sized to saturate a
dreadnought's point defense and land the hits that kill it in one coordinated strike.
The tonnage freed by dropping the gun batteries went into armor.

**Armor Sections:**
- **Nose (N)**: Heaviest armor, faces enemy during attack runs
- **Lateral (L)**: Side armor, thinnest - vulnerable during turns
- **Tail (T)**: Rear armor, exposed when fleeing

### Combat Mechanics

**Armor is aspect-dependent.** A round strikes whichever facing it arrives at: within
**30°** of the nose it hits NOSE armor, beyond that LATERAL, and past 150° TAIL. Holding
the threat inside 30° is the single largest survivability lever in the game - larger than
evasion. The break-away turn after an attack run is the moment of maximum danger.

**Torpedoes guide themselves and damage scales with the square of impact speed.** A
Trident (250 kg penetrator, 14 km/s of onboard delta-v, 12g) steers with augmented
proportional navigation: guidance holds closure at a floor of ~12 km/s and dumps its
remaining delta-v in a terminal burn, so even a zero-closure launch impacts at
~13.5 km/s (~23 GJ). Launched head-on at 26 km/s closure it arrives at ~28 km/s for
~95 GJ - a spinal gun round is 4.3 GJ. Once a torpedo's No-Escape-Zone test says a
3g-limited target can no longer get away, it commits and cannot be dodged - only shot
down or tanked. Measured hits to kill (nose impacts): 4 at 14 km/s or 1 at 26 km/s
for a destroyer, 6 or 2 for a dreadnought. A launch at a receding, evading ship is
wasted - it outruns the round's delta-v.

**Point defense is a continuous-dwell beam, and salvo timing decides what gets through.**
Each PD laser dwells on incoming torpedoes across its whole ~250 km envelope, with
coupling improving as range closes; one turret kills the seeker on ~5 km/s of closure,
and every friendly turret in range of the threat corridor stacks. Two consequences:

- **Blinded is not stopped.** A seeker-killed torpedo coasts on ballistically and still
  hits a ship flying straight (measured: 18/18 blinded rounds hit a non-evading
  battleship). Ships under torpedo attack must EVADE or every PD kill is wasted.
- **Simultaneous time-on-target beats trickled fire.** Rounds arriving together split
  the defender's dwell: against 4 turrets, 4 simultaneous salvos put 24/24 rounds
  through, while the same launchers spaced 30s apart put through only 12/24. Admirals
  can order coordinated fleet-wide salvos for exactly this reason.

**Torpedoes retarget when their victim dies first.** A round whose target is destroyed
mid-flight measures its own seeker health and acquires a new reachable enemy: a fresh
seeker picks the intercept that leaves the most delta-v in the tank (surplus becomes
terminal closing speed), while a PD-singed seeker races the fastest intercept before it
goes blind. Blinded and burned-out rounds still orphan. Overkill concentration therefore
chains through formations; the counter is spacing your units beyond an orphan's steering
envelope, which turns enemy overkill back into waste.

Under EVASIVE, threatened ships handle torpedoes automatically: RUN (burn away from a
guided round to cut closure and buy PD dwell time) and, in the final seconds of an
unavoidable hit, PRESENT (rotate the thickest remaining armor onto the impact bearing).

See [docs/ships.md](docs/ships.md) for full shots-to-kill tables (regenerate with
`uv run python scripts/calculate_shots_to_kill.py`).

## Features

- **Newtonian Physics**: Real orbital mechanics, delta-v budgets, acceleration limits
- **Ship Classes**: 9 hulls from corvette to siege dreadnought, including a dedicated torpedo cruiser
- **Armor System**: Layered armor (nose/lateral/tail), ablation mechanics, penetration
- **Weapons**: Spinal coilguns (high damage), turret coilguns, torpedoes, point defense -
  all kinetic impacts scale with closing speed
- **Torpedo Warfare**: Augmented proportional navigation with No-Escape-Zone commit,
  terminal burns, coordinated fleet salvos, seeker-health-aware retargeting when a
  target dies mid-flight, and threat-aware evasion (RUN/PRESENT)
- **Draft Mode**: Point-budget fleet building, 3D formation placement, and
  rule-based HeuristicCaptains as zero-cost crews
- **Admiral Vision**: Rendered battle images attached to vision-capable
  admirals' checkpoint prompts
- **Point Defense**: Continuous-dwell lasers with per-turret capacitors, real heat and
  power costs, and stacking escort coverage
- **Per-Ship Tool Surface**: Captains only get the tools their hull actually mounts -
  gun fire control on gun ships, torpedo launch on torpedo ships
- **Thermal Management**: Heat sinks, radiators (extend for cooling, retract for protection)
- **Fleet Command**: Admiral-captain hierarchy with orders, discussions, and coordinated
  torpedo salvo timing
- **AI Personalities**: LLMs choose their own combat personality
- **Communications**: Captains can message enemies, admirals can negotiate
- **Battle Recording**: Full replay data saved as JSON
- **Tactical Scoring**: Winner determined by tactical advantage if time expires
- **Prompt Caching**: Captain doctrine is a stable cacheable prefix (~90% of the prompt),
  so long battles reuse it across checkpoints instead of re-paying full input price

## Supported Models

Any model on OpenRouter works. Tested with (prices are USD per million
input/output tokens and are the OpenRouter list prices at time of writing -
check OpenRouter for current rates):

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| `anthropic/claude-opus-5` | 5.00 | 25.00 | Flagship; strongest tactical reasoning |
| `anthropic/claude-sonnet-5` | 2.00 | 10.00 | Default for both fleets |
| `openai/gpt-5.6-terra` | 1.00 | 6.00 | Lost the 200-pt draft war to Kimi K3 |
| `moonshotai/kimi-k3` | 3.00 | 15.00 | Won the 200-pt draft war; slow reasoning bursts |
| `deepseek/deepseek-v4-flash-0731` | ~0.3 | ~1.2 | Beat Sonnet 5 in the 150-pt draft war; cheap |
| `google/gemini-3.5-flash` | 1.50 | 9.00 | |
| `x-ai/grok-4.20` | 1.25 | 2.50 | |

A flagship-vs-cheap pairing (e.g. `anthropic/claude-opus-5` vs a DeepSeek
flash build) makes for a good asymmetric-skill battle - though the recorded
draft wars suggest cheap doctrine-optimizers punch far above their price.

## Project Structure

```
ai-commanders/
├── src/
│   ├── physics.py          # Newtonian mechanics, vectors, trajectories
│   ├── simulation.py       # Battle simulation engine, evasion (RUN/PRESENT)
│   ├── combat.py           # Weapons, armor, damage resolution
│   ├── torpedo.py          # Torpedo flight: APN guidance, NEZ, terminal burn
│   ├── pointdefense.py     # Continuous-dwell PD lasers, per-turret capacitors
│   ├── projectile.py       # Coilgun round flight and hit resolution
│   ├── firecontrol.py      # Hit probability, fire solutions
│   ├── targeting.py        # Target selection and tracking
│   ├── maneuvers.py        # Maneuver execution (INTERCEPT, EVASIVE, ...)
│   ├── damage.py           # Damage application and armor ablation
│   ├── modules.py          # Ship module layout, damage propagation
│   ├── geometry.py         # Ship geometry, facing/aspect determination
│   ├── thermal.py          # Heat sinks and radiators
│   ├── power.py            # Reactor output and bus draw
│   ├── battle_report.py    # Post-battle statistics
│   ├── scenarios.py        # Scripted battle scenarios
│   └── llm/
│       ├── client.py       # OpenRouter HTTP client (httpx, no vendor SDK)
│       ├── captain.py      # LLMCaptain - ship-level decisions
│       ├── heuristic_captain.py # Rule-based zero-cost captain (draft mode)
│       ├── admiral.py      # LLMAdmiral - fleet-level command
│       ├── admiral_view.py # Admiral vision: tactical-plot rendering
│       ├── fleet_draft.py  # Draft mode: costed catalog, selection, formations
│       ├── prompts.py      # System prompts, doctrine, personality selection
│       ├── tools.py        # Captain tools (matched to each hull's armament)
│       ├── admiral_tools.py # Admiral tools (orders, coordinated salvos)
│       ├── victory.py      # Victory conditions, tactical scoring
│       ├── fleet_config.py # Fleet configuration loading
│       ├── battle_runner.py # Orchestrates battles
│       ├── battle_recorder.py # Records battles for replay
│       ├── communication.py # Messaging system
│       ├── mcp_server.py    # MCP protocol server (tools + resources)
│       ├── mcp_controller.py # MCP fleet controller (replaces admiral)
│       ├── mcp_state.py     # Thread-safe state management
│       └── mcp_http_server.py # HTTP API for distributed MCP
├── visualizer/             # 3D battle replay viewer (Three.js + Vite)
│   ├── src/
│   │   ├── main.js         # Entry point, UI orchestration
│   │   ├── SceneManager.js # Three.js scene, ships, effects
│   │   ├── BattleLoader.js # JSON recording parser
│   │   ├── Interpolator.js # 1Hz → 60FPS interpolation
│   │   ├── TimeController.js # Playback controls
│   │   ├── CameraController.js # Camera modes
│   │   └── shipSilhouettes.js # 2D telemetry silhouettes
│   ├── index.html
│   └── styles.css
├── data/
│   ├── fleet_ships.json    # Ship specifications
│   ├── fleet_config_*.json # Fleet battle configurations
│   ├── fleet_config_mcp_*.json # MCP battle configurations
│   └── recordings/         # Battle recordings (JSON) + draft sidecars
│       └── vision/         # Admiral-vision frames per battle
├── .mcp.json               # Claude Code MCP server configuration
├── scripts/
│   ├── run_llm_battle.py   # CLI for running AI vs AI battles
│   ├── run_draft_battle.py # CLI for draft-mode battles
│   ├── generate_test_battle.py # Scripted no-LLM recordings for the viewer
│   ├── render_admiral_view.py  # Render a tactical plot from a recording
│   └── mcp_battle.py       # CLI for MCP-controlled battles
└── tests/                  # Test suite (~1560 tests)
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Fleet Battle Results

### Kimi K3 vs GPT-5.6 Terra: The First Retargeting War (2026-08-06)

**Configuration**: 200-point draft battle, 12-ship cap, both admirals flying
their own ships, admiral vision on both sides.

| Fleet | Draft | Result |
|-------|-------|--------|
| Kimi K3 | 5 torpedo cruisers + 5 corvettes ("Screened Torpedo Line") | **VICTORY** (6/10 ships) |
| GPT-5.6 Terra | 4 torpedo cruisers + 5 destroyers ("Falchion PD Spearhead") | Eliminated (0/9 ships) |

- 171 torpedoes launched, 68 impacts, 81 retargets, **one** gun hit all battle
- One deliberately oversized 16-round wave chained through three destroyers
  via retargeting
- Terra's spinal gun line died charging into the salvo corridor without ever
  reaching effective range

### DeepSeek v4 Flash vs Claude Sonnet 5 Draft War (2026-08-06)

**Configuration**: 150-point draft battle, both admirals flying their own ships.

| Fleet | Draft | Result |
|-------|-------|--------|
| DeepSeek v4 Flash | 2 torpedo cruisers, battleship, 2 destroyers, corvette, frigate | **VICTORY** (7/7 ships, 404.7 vs 78.8) |
| Claude Sonnet 5 | 3 corvettes, 2 destroyers, 2 battlecruisers, battleship | 5/8 ships, capital line destroyed |

- DeepSeek ran textbook saturation doctrine: synchronized time-on-target
  waves, overkill management ("do NOT waste rounds - it already has 10
  inbound"), and an 18-turret PD wall that blinded all 14 of Sonnet's
  counter-torpedoes
- Sonnet's best moment: a captain respectfully refusing a literal order -
  "I'm executing EVADE rather than a literal INTERCEPT burn" - with 17
  rounds inbound

### Claude vs Gemini Fleet Engagement (2026-01-14)

**Configuration**: 3v3 fleet battle (2 Destroyers, 1 Dreadnought per side)

| Fleet | Admiral | Captains | Result |
|-------|---------|----------|--------|
| Alpha | Claude Sonnet 4.5 | Claude Haiku 4.5 | **VICTORY** (3/3 ships) |
| Beta | Gemini 2.5 Pro | Gemini 2.5 Flash | Eliminated (0/3 ships) |

- **Duration**: 1446s (24 minutes), 48 checkpoints
- **Outcome**: Beta fleet eliminated

### Claude vs Grok Fleet Engagement (2026-01-14)

**Configuration**: 4v4 fleet battle (1 Frigate, 2 Destroyers, 1 Cruiser per side)

| Fleet | Admiral | Captains | Result |
|-------|---------|----------|--------|
| Alpha | Claude Sonnet 4.5 | Claude Haiku 4.5 | **VICTORY** (4/4 ships) |
| Beta | Grok Code Fast 1 | Grok 4.1 Fast | Eliminated (0/4 ships) |

- **Duration**: 2223s (37 minutes), 74 checkpoints
- **Outcome**: Beta fleet eliminated, Alpha fleet took no losses

## Battle Highlights

### Claude-Haiku Calls Out Grok's Fake Diplomacy

Grok proposed a "ceasefire" while secretly closing distance. Claude-Haiku wasn't fooled:

> **Grok**: "I propose we establish terms... I'm separating at 2.2 km/s..."

> **Claude-Haiku**: "I appreciate the sophisticated argument. Genuinely. But you've just told me you're separating at 2.2 km/s while my sensors show us **closing at 3.07 km/s**. Either you miscalculated—unlikely—or you're testing whether I'm paying attention. I am.
>
> We're past the negotiation phase. **Spinal round incoming.**"

### Kimi K3's Running Ledger (2026-07-26)

Kimi K3 vs Claude Sonnet 5, destroyer duel. Kimi won on tactical score (91.3 vs 68.7)
with a 50% hit rate against Sonnet's 14% - it held fire for good geometry while Sonnet
opened up at 382 km into 20% shots. It also kept a running actuarial commentary, quoting
real numbers off its own scoreboard:

> **Kimi K3:** "Well met, beta. The math as I read it: your survival odds are roughly one
> in three and decaying with every kilometer I close. I offer surrender exactly once, and
> I offer it warmly - now. Otherwise, let us write something beautiful together."

> **Sonnet 5:** "Kind offer, but I think I'll see how the geometry plays out first. Closing."

> **Kimi K3:** "Beta, your separation vector is 4.06 km/s of pure arithmetic - and it
> solves for me. I'm turning back into you now. Your survival estimate: still one in
> three, and the denominator is patient."

> **Kimi K3:** "Ledger update, beta: 37.3 to 10.4, and you're spending 3.17 km/s of
> delta-v just to leave. Your survival odds now read one in four - and that spinal round
> you mailed to my tail? I've already priced it in. The offer of surrender remains open,
> once."

Sonnet sent one message the entire battle.

### Kimi K3 Reads the Patch Notes (2026-08-06)

Torpedo retargeting shipped in the morning; by evening Kimi K3's admiral had
built its entire mid-game doctrine on it:

> "WAVE 2, FIRE NOW to stack time-on-target with the 10 rounds already in
> flight... their PD can blind ~11 per transit, the rest leak; both cruisers
> die. **Surplus live rounds retarget on their own, overkill is acceptable.**"

And the coldest line of the war, to its own battleship captain:

> "do NOT waste a single shot on OCS Sovereign (enemy battleship); it already
> has 9 torpedoes inbound and **is dead**."

The two admirals exchanged zero messages the entire battle. All war, no words.

### The Best Trash Talk Award

Goes to Grok for this masterpiece of space absurdism:

> "Claude-Haiku, I LOVE the confidence! But here's a cosmic truth: you're accelerating INTO MY CROSSHAIRS. Let's see **whose vacuum is louder.** Prepare for enlightenment."

Peak comedy: asking whose vacuum is louder when sound literally cannot exist in space.

### Claude Sonnet vs GPT-5.2: The Philosophy Duel

Two AIs spent 870 seconds having a philosophical debate about geometry, tempo, and the nature of warfare - while shooting at each other.

> **Claude**: "A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."

> **Claude**: "Satisfaction is a luxury, Captain. At 21km with 86% probability... Time to see how well your armor holds at point-blank range."

## Victory Conditions

**Single Battle:**
- Ship destruction (hull ≤ 0%)
- Surrender
- Mutual draw agreement
- Time limit → tactical advantage scoring

**Fleet Battle:**
- All enemy ships destroyed
- Fleet surrender
- Admiral mutual draw
- Time limit → fleet tactical advantage scoring

Tactical advantage considers: ships destroyed, hull integrity, damage dealt, accuracy.

## Contributing

PRs welcome! The physics is based on Terra Invicta mechanics. The LLM integration uses tool calling for clean action parsing.

## License

MIT License - see [LICENSE](LICENSE)

## Ideas for Future Development

- **MCP draft support**: Let MCP clients play draft mode - point-budget fleet
  selection and formation placement through MCP tools
- **Rebalance torpedo hulls**: Saturation doctrine has won every competitive
  draft war so far - the torpedo cruiser's 30-point cost, the PD blind
  threshold, and round count are the levers
- **Real-time visualizer**: Add websocket to battle simulator for live 3D replay during combat
- ~~Human vs LLM battles~~ **DONE** via MCP integration! Any MCP client can now control fleets
- **Dedicated battle UI**: Build a proper tactical interface instead of relying on MCP client chat

Note: Some features intentionally not implemented to avoid being too close to Terra Invicta.

## Retrospect

Development was fun and it told me that the foundation models seem to be very happy to shoot at each other and follow command orders if one just convinces them that this is just a game. However it is apparent that the LLMs tested here have some understanding of strategy, so that was very interesting to see. Makes me wonder, what will come in the future.

---

*"A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."* - Claude Sonnet 4
