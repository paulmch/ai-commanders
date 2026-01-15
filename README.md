# AI Commanders

A Terra Invicta-inspired space battle simulator where **LLM-controlled captains and admirals** command fleets in tactical combat. Watch AI models coordinate fleet maneuvers, issue orders, and trash-talk while trying to blow each other up with coilguns and torpedoes.

## What is this?

Two modes of warfare:

**Single Combat**: Two AI captains. Two ships. 500km of vacuum. Each captain makes tactical decisions every 30 seconds and can message their opponent.

**Fleet Battles**: Two AI admirals commanding fleets of multiple ships. Admirals issue strategic orders, captains execute tactics, and everyone can communicate. Coordinate focus fire, flanking maneuvers, and combined arms.

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
    --alpha-model openrouter/anthropic/claude-sonnet-4 \
    --beta-model openrouter/x-ai/grok-3-fast \
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
```

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

- **Expanse-style ship designs**: Donnager-class capitals, Tachi-style corvettes
- **Multi-engine plumes**: Particle-based thrust visualization with bloom
- **Projectile trails**: Coilgun rounds with motion trails
- **PD laser beams**: Point defense engagements rendered as fading beams
- **Impact effects**: Dual-ring shockwaves with particle bursts
- **Ship destruction**: Multi-phase fusion reactor explosions
  - Hull breach explosions with point lights
  - Secondary detonations (munitions/fuel)
  - Blinding reactor breach flash
  - Expanding plasma sphere with shockwave
  - 100,000 particle debris cloud
  - Lingering plasma aftermath
- **Camera modes**: Free orbit, follow ship, orbit selected ship
- **Ship telemetry**: Hull, armor, modules, target, maneuver status
- **Timeline scrubbing**: Jump to any point, adjustable playback speed (0.25x-8x)
- **Time input**: Enter exact timestamps (MM:SS or seconds) to jump directly

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

### Fleet Battle Mode

Multi-ship engagements with hierarchical command:

**Admirals** (one per fleet):
- See dual-snapshot tactical view (T-15s and T=0)
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
    "admiral": "openrouter/anthropic/claude-sonnet-4",
    "ships": [
      {"ship_type": "destroyer", "model": "openrouter/anthropic/claude-haiku-4.5"},
      {"ship_type": "destroyer", "model": "openrouter/anthropic/claude-haiku-4.5"},
      {"ship_type": "dreadnought", "model": "openrouter/anthropic/claude-haiku-4.5"}
    ]
  },
  "beta_fleet": {
    "admiral": "openrouter/google/gemini-2.5-pro-preview",
    "ships": [
      {"ship_type": "destroyer", "model": "openrouter/google/gemini-2.5-flash-preview"},
      {"ship_type": "destroyer", "model": "openrouter/google/gemini-2.5-flash-preview"},
      {"ship_type": "dreadnought", "model": "openrouter/google/gemini-2.5-flash-preview"}
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
| `ship_type` | Ship class (frigate, destroyer, cruiser, etc.) |
| `model` | OpenRouter model ID for the captain |

## CLI Reference

```
uv run python scripts/run_llm_battle.py [OPTIONS]
```

### Model & Ship Options

| Option | Description | Default |
|--------|-------------|---------|
| `--alpha-model` | OpenRouter model for Alpha | claude-3.5-sonnet |
| `--beta-model` | OpenRouter model for Beta | claude-3.5-sonnet |
| `--alpha-ship-type` | Ship class for Alpha | destroyer |
| `--beta-ship-type` | Ship class for Beta | destroyer |
| `--alpha-name` | Captain name for Alpha | Commander Chen |
| `--beta-name` | Captain name for Beta | Captain Volkov |
| `--alpha-ship` | Ship name for Alpha | TIS Relentless |
| `--beta-ship` | Ship name for Beta | HFS Determination |

### Battle Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fleet-config FILE` | JSON fleet config (enables fleet mode) | None |
| `--distance KM` | Initial distance in km | 500 |
| `--max-checkpoints N` | Max decision points | 40 |
| `--time-limit SEC` | Time limit in seconds | 1200 |
| `--unlimited` | Fight until destruction/surrender/draw | False |
| `--trace` | Record detailed sim trace (large files!) | False |

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

## Ship Classes

| Ship | Accel | 90° Turn | Armor (N/L/T) | Role |
|------|-------|----------|---------------|------|
| Corvette | 3.0g | 12s | Light | Scout, harassment |
| Frigate | 3.0g | 15s | Light | Fast attack |
| Destroyer | 2.0g | 21s | 151/26/30 cm | Balanced combatant |
| Cruiser | 1.5g | 28s | Medium | Heavy firepower |
| Battlecruiser | 1.5g | 28s | Medium | Fast capital |
| Battleship | 1.0g | 37s | Heavy | Line combat |
| Dreadnought | 0.75g | 50s | 180/43/50 cm | Fleet anchor |

**Armor Sections:**
- **Nose (N)**: Heaviest armor, faces enemy during attack runs
- **Lateral (L)**: Side armor, thinnest - vulnerable during turns
- **Tail (T)**: Rear armor, exposed when fleeing

## Features

- **Newtonian Physics**: Real orbital mechanics, delta-v budgets, acceleration limits
- **Ship Classes**: 7 classes from corvette to dreadnought
- **Armor System**: Layered armor (nose/lateral/tail), ablation mechanics, penetration
- **Weapons**: Spinal coilguns (high damage), turret coilguns, torpedoes, point defense
- **Thermal Management**: Heat sinks, radiators (extend for cooling, retract for protection)
- **Fleet Command**: Admiral-captain hierarchy with orders and discussions
- **AI Personalities**: LLMs choose their own combat personality
- **Communications**: Captains can message enemies, admirals can negotiate
- **Battle Recording**: Full replay data saved as JSON
- **Tactical Scoring**: Winner determined by tactical advantage if time expires

## Supported Models

Any model on OpenRouter works. Tested with:
- `anthropic/claude-sonnet-4` / `claude-opus-4` / `claude-haiku-4.5`
- `openai/gpt-4o` / `gpt-4o-mini`
- `x-ai/grok-3-fast`
- `google/gemini-2.5-pro-preview` / `gemini-2.5-flash-preview`
- `deepseek/deepseek-chat`

## Project Structure

```
ai-commanders/
├── src/
│   ├── physics.py          # Newtonian mechanics, vectors, trajectories
│   ├── combat.py           # Weapons, armor, damage resolution
│   ├── simulation.py       # Battle simulation engine
│   ├── modules.py          # Ship module layout, damage propagation
│   └── llm/
│       ├── client.py       # LiteLLM wrapper for OpenRouter
│       ├── captain.py      # LLMCaptain - ship-level decisions
│       ├── admiral.py      # LLMAdmiral - fleet-level command
│       ├── prompts.py      # System prompts, personality selection
│       ├── tools.py        # Captain tool definitions
│       ├── admiral_tools.py # Admiral tool definitions
│       ├── fleet_config.py # Fleet configuration loading
│       ├── battle_runner.py # Orchestrates battles
│       ├── battle_recorder.py # Records battles for replay
│       └── communication.py # Messaging system
├── visualizer/             # 3D battle replay viewer (Three.js + Vite)
│   ├── src/
│   │   ├── main.js         # Entry point, UI orchestration
│   │   ├── SceneManager.js # Three.js scene, ships, effects
│   │   ├── BattleLoader.js # JSON recording parser
│   │   ├── Interpolator.js # 1Hz → 60FPS interpolation
│   │   ├── TimeController.js # Playback controls
│   │   └── CameraController.js # Camera modes
│   ├── index.html
│   └── styles.css
├── data/
│   ├── fleet_ships.json    # Ship specifications
│   ├── fleet_config_*.json # Fleet battle configurations
│   └── recordings/         # Battle recordings (JSON)
├── scripts/
│   └── run_llm_battle.py   # CLI for running battles
└── tests/                  # Test suite
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Fleet Battle Results

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

---

*"A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."* - Claude Sonnet 4
