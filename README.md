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

## MCP Battle Guide

Control fleets directly using the Model Context Protocol. Any MCP-compatible client works: Claude Code, Cursor, GitHub Copilot, OpenCode, or custom agents.

### Architecture Overview

Two independent MCP servers allow **any combination** of human/AI control:

```
┌─────────────────┐                         ┌─────────────────┐
│  MCP Client     │                         │  MCP Client     │
│  (Alpha Fleet)  │                         │  (Beta Fleet)   │
└────────┬────────┘                         └────────┬────────┘
         │ MCP                                       │ MCP
         ▼                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│  Alpha Server   │                         │  Beta Server    │
│  (port 8765)    │                         │  (port 8766)    │
└────────┬────────┘                         └────────┬────────┘
         │              ┌─────────────────┐          │
         └─────────────►│  Battle Runner  │◄─────────┘
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
      "args": ["run", "python", "-m", "src.llm.mcp_server", "--faction", "beta", "--http", "http://localhost:8766"]
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
| `EVASIVE` | Random dodging pattern (needs throttle!) |
| `BRAKE` | Flip and decelerate |
| `MAINTAIN` | Coast at current velocity |
| `PADLOCK` | Coast while keeping nose pointed at target (good for shooting) |
| `HEADING` | Fly in specific 3D direction |

### Key Concepts

**Orders reset every turn!** You must re-issue maneuver and weapons orders each checkpoint. Ships default to MAINTAIN (coasting) if you don't command them.

**Fog of war**: You see full status of friendly ships but only observable data for enemies (position, velocity, estimated hull %, hit chance).

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
    "admiral": {"model": "openrouter/x-ai/grok-3-fast", "name": "Admiral Grok"},
    "ships": [
      {"ship_id": "beta_1", "ship_type": "destroyer", "model": "openrouter/x-ai/grok-4.1-fast"},
      {"ship_id": "beta_2", "ship_type": "destroyer", "model": "openrouter/x-ai/grok-4.1-fast"}
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
│   │   └── CameraController.js # Camera modes
│   ├── index.html
│   └── styles.css
├── data/
│   ├── fleet_ships.json    # Ship specifications
│   ├── fleet_config_*.json # Fleet battle configurations
│   ├── fleet_config_mcp_*.json # MCP battle configurations
│   └── recordings/         # Battle recordings (JSON)
├── .mcp.json               # Claude Code MCP server configuration
├── scripts/
│   ├── run_llm_battle.py   # CLI for running AI vs AI battles
│   └── mcp_battle.py       # CLI for MCP-controlled battles
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

## Ideas for Future Development

- **Real-time visualizer**: Add websocket to battle simulator for live 3D replay during combat
- ~~Human vs LLM battles~~ **DONE** via MCP integration! Any MCP client can now control fleets
- **Dedicated battle UI**: Build a proper tactical interface instead of relying on MCP client chat

Note: Some features intentionally not implemented to avoid being too close to Terra Invicta.

## Retrospect

Devolpoment was fun and it told me that the foundation models seem to be very happy to shoot at each other and follow command orders if one just convinces them that this is just a gane. However it is apparent that the LLMS tested here have some understanding of strategy, so that was very intersting to see. Makes me wonder, what will come in the future. 

---

*"A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."* - Claude Sonnet 4
