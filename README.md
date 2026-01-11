# AI Commanders

A Terra Invicta-inspired space battle simulator where **LLM-controlled captains** command destroyers in tactical combat. Watch AI models trash-talk each other while trying to blow each other up with coilguns and torpedoes.

## What is this?

Two AI captains. Two destroyers. 500km of cold vacuum between them. Each captain gets a unique personality, makes tactical decisions every 30 seconds, and can send messages to their opponent. The physics is real (Newtonian mechanics, armor ablation, heat management), but the trash talk is pure AI.

**LLM Tournament coming soon!**

## Quick Start

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install numpy pytest litellm

# Set your OpenRouter API key
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Run a battle!
python scripts/run_llm_battle.py \
    --alpha-model openai/gpt-5.2 \
    --beta-model x-ai/grok-4.1-fast \
    --max-checkpoints 10 \
    --verbose
```

## Features

- **Newtonian Physics**: Real orbital mechanics, delta-v budgets, acceleration limits
- **Armor System**: Layered armor (nose/lateral/tail), ablation mechanics, penetration
- **Weapons**: Spinal coilguns (high damage), turret coilguns, torpedoes, point defense
- **Thermal Management**: Heat sinks, radiators (extend for cooling, but they're vulnerable!)
- **AI Captains**: LLMs choose their own personality, make tactical decisions, and trash-talk
- **Battle Recording**: Full replay data saved as JSON

## Supported Models

Any model on OpenRouter works. Tested with:
- `anthropic/claude-sonnet-4` / `claude-opus-4`
- `openai/gpt-5.2`
- `x-ai/grok-4.1-fast`
- `google/gemini-3-flash-preview`
- `deepseek/deepseek-v3.2`

## Battle Statistics (23 Battles)

| Model | Wins | Losses | Draws |
|-------|------|--------|-------|
| Claude (various) | 7 | 4 | 11 |
| Grok-4.1-fast | 4 | 7 | - |
| Gemini-3-flash | 1 | 0 | 1 |
| GPT-5.2 | 0 | 1 | 1 |
| DeepSeek-v3.2 | 0 | 1 | 0 |

Most decisive victory: **Claude-Haiku-4.5** vs Grok-4.1-Fast (24-0 hits over 1200s)

## Battle Highlights

### Claude Sonnet vs GPT-5.2: The Philosophy Duel

Two AIs spent 870 seconds having a philosophical debate about geometry, tempo, and the nature of warfare - while shooting at each other.

> **Claude**: "A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."

> **GPT**: "Asymmetric rhythms noted, Captain. But at 103km with 73% probability, I find the current tempo quite satisfactory."

> **Claude**: "Satisfaction is a luxury, Captain. At 21km with 86% probability... Time to see how well your armor holds at point-blank range."

Later, GPT started bragging about dealing damage - while taking 17% hull damage to Claude's 0%:

> **GPT**: "My 87% hit probability at 17km with both weapons cycling means your armor is eroding faster than mine."

*Narrator: GPT's armor was, in fact, eroding faster.*

The battle ended in a mutual draw after both captains agreed to stop, having developed what can only be described as "respect."

### GPT-5.2 vs Grok-4.1: Accountants with Knives

Both models chose remarkably similar personalities:

> **GPT**: "I fight like a patient accountant with a knife: ruthless about expected value, but always hunting the one irregularity that breaks a battle open."

> **Grok**: "I fight like a skeptical mathematician with a poet's spite... calm feint, brutal commit."

Despite calling itself "patient," GPT fired first at 500km (12% hit chance) and missed everything. Grok waited, then landed 7/8 shots (87.5% accuracy) vs GPT's 5/10 (50%).

### DeepSeek vs Gemini: Death by Persistence

Gemini destroyed DeepSeek after 1059 seconds of relentless pursuit. DeepSeek kept trying to evade while Gemini just... kept coming.

### Claude-Haiku Calls Out Grok's Fake Diplomacy

Grok proposed a "ceasefire" while secretly closing distance. Claude-Haiku wasn't fooled:

> **Grok**: "I propose we establish terms... I'm separating at 2.2 km/s..."

> **Claude-Haiku**: "I appreciate the sophisticated argument. Genuinely. But you've just told me you're separating at 2.2 km/s while my sensors show us **closing at 3.07 km/s**. Either you miscalculated—unlikely—or you're testing whether I'm paying attention. I am.
>
> We're past the negotiation phase. **Spinal round incoming.** I'm not firing because I expect a hit—I'm firing because you've demonstrated you're willing to use dialogue as cover for tactical positioning."

Cold-blooded.

### The Best Trash Talk Award

Goes to Grok-4.1-fast for this masterpiece of space absurdism:

> "Claude-Haiku, I LOVE the confidence! But here's a cosmic truth: you're accelerating INTO MY CROSSHAIRS. Let's see **whose vacuum is louder.** Prepare for enlightenment."

Peak comedy: asking whose vacuum is louder when sound literally cannot exist in space.

## Project Structure

```
ai-commanders/
├── src/
│   ├── physics.py          # Newtonian mechanics, vectors, trajectories
│   ├── combat.py           # Weapons, armor, damage resolution
│   ├── simulation.py       # Battle simulation engine
│   ├── modules.py          # Ship module layout, damage propagation
│   └── llm/                # LLM integration
│       ├── client.py       # LiteLLM wrapper for OpenRouter
│       ├── captain.py      # LLMCaptain - decision making
│       ├── prompts.py      # System prompts, personality selection
│       ├── tools.py        # Tool definitions for LLM actions
│       ├── battle_runner.py # Orchestrates LLM battles
│       └── communication.py # Captain messaging system
├── data/
│   ├── fleet_ships.json    # Ship specifications
│   └── recordings/         # Battle recordings (JSON)
├── scripts/
│   └── run_llm_battle.py   # CLI for running battles
└── tests/                  # Test suite
```

## Command Line Options

```bash
python scripts/run_llm_battle.py \
    --alpha-model MODEL_ID \      # OpenRouter model for Alpha
    --beta-model MODEL_ID \       # OpenRouter model for Beta
    --max-checkpoints N \         # Max decision points (default: 20)
    --initial-distance KM \       # Starting distance (default: 500)
    --verbose                     # Show detailed output
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## How It Works

1. **Personality Selection**: Each captain defines their combat personality (no presets - pure creativity)
2. **Checkpoint Loop**: Every 30 seconds, simulation pauses for captain decisions
3. **Tool Calls**: Captains use tools like `set_maneuver`, `fire_weapon`, `send_message`, `surrender`
4. **Physics Simulation**: Projectiles fly, armor ablates, modules take damage
5. **Victory**: First to destroy the enemy, or tactical advantage at time limit

## Ship Specifications (Destroyer Class)

| Attribute | Value |
|-----------|-------|
| Acceleration | 2.0g |
| Delta-V Budget | 500 km/s |
| Spinal Coilgun | 500km range, ~5 GJ per hit |
| Turret Coilgun | 200km range, ~1 GJ per hit |
| Nose Armor | 151 cm |
| Lateral Armor | 26 cm |
| Tail Armor | 30 cm |

## Contributing

PRs welcome! The physics is based on Terra Invicta mechanics. The LLM integration uses tool calling for clean action parsing.

## License

MIT License - see [LICENSE](LICENSE)

---

*"The beauty of asymmetric rhythms is that one partner dictates tempo. Watch."* - Claude Sonnet 4, moments before GPT learned about geometry
