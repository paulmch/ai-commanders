# AI Commanders - Development Guide

## Project Overview
Terra Invicta-inspired space battle simulator where LLM captains and admirals
command fleets: real Newtonian physics and combat simulation, with tactical
decisions made by LLMs (via OpenRouter) or humans (via MCP) every 30s checkpoint.

## Setup

```bash
# Use uv for Python package management
uv sync --locked --all-extras   # --all-extras pulls in the mcp extra (mcp, aiohttp)

# LLM battles need an OpenRouter key (MCP-controlled fleets don't)
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
```

## Running Tests

```bash
uv run pytest tests/ -v   # ~1500 tests, ~1-2 minutes
```

CI (`.github/workflows/tests.yml`) runs the same suite on Python 3.10-3.12 with
`uv sync --locked --all-extras`.

## Running Battles

```bash
uv run python scripts/run_llm_battle.py -v                                  # 1v1 duel
uv run python scripts/run_llm_battle.py --fleet-config data/fleet_config_claude_vs_gemini.json -v
uv run python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json  # human via MCP
```

## Project Structure

```
ai-commanders/
├── data/
│   ├── fleet_ships.json    # Ship specs, weapons, armor - source of truth for hulls
│   └── fleet_config_*.json # Fleet battle configurations
├── src/
│   ├── simulation.py       # Battle engine, threat-aware evasion (RUN/PRESENT)
│   ├── physics.py          # Newtonian mechanics, trajectories, rotation
│   ├── combat.py           # Weapons, armor, damage resolution
│   ├── torpedo.py          # APN guidance, No-Escape-Zone, terminal burn
│   ├── pointdefense.py     # Continuous-dwell PD lasers, per-turret capacitors
│   ├── projectile.py / firecontrol.py / targeting.py / maneuvers.py
│   ├── damage.py / modules.py / geometry.py / thermal.py / power.py
│   └── llm/                # OpenRouter client, captain/admiral agents, prompts,
│                           # tools, battle runner/recorder, MCP servers
├── tests/                  # pytest suite; test_fixes_*.py pin past bug fixes
├── scripts/
│   ├── run_llm_battle.py   # CLI for AI vs AI battles
│   ├── mcp_battle.py       # CLI for MCP-controlled battles
│   └── calculate_shots_to_kill.py  # Regenerates docs/ships.md combat tables
├── docs/ships.md           # Ship specs + simulated shots-to-kill tables
└── visualizer/             # Three.js 3D battle replay viewer
```

## Key Constants

- Exhaust velocity: 10,256 km/s
- Main thrust: 58.56 MN (Protium Converter Torch x6)
- Combat thrust vectoring: 1° deflection
- Target delta-v: 500 km/s
- Trident torpedo: 250 kg penetrator, 14 km/s delta-v, 12g; guidance holds a
  ~12 km/s closure floor and dumps remaining delta-v in the terminal burn
- PD laser: 250 km envelope, continuous dwell; one turret blinds ~5 km/s of closure

## Control Architecture

Hierarchical control:
1. **Admiral LLM** (fleet battles): two-phase orders every checkpoint - fleet
   directive plus per-captain orders, including coordinated torpedo salvos
2. **Captain LLM**: tool calls every 30s decision cycle (maneuver, weapons order,
   torpedo launch, radiators, messages). Tool surface is derived from the hull's
   actual armament - gun tools only on gun ships, torpedo tools only on torpedo ships
3. **Rule-based layer**: executes between checkpoints - fire control modes,
   automatic point defense, threat-aware evasion (wobble vs guns, RUN from guided
   torpedoes, PRESENT thickest armor when a hit is unavoidable)

MCP mode replaces the admiral+captains of a fleet with any MCP client
(`src/llm/mcp_server.py`, one battle HTTP API on port 8765, `--faction alpha|beta`).

## Ship Classes

| Ship | Accel | 90° Turn (TV) | 90° Turn (RCS) |
|------|-------|---------------|----------------|
| Corvette | 3.0g | 12.1s | 54.2s |
| Frigate | 3.0g | 15.1s | 83.3s |
| Destroyer | 2.0g | 20.6s | 127.6s |
| Cruiser | 1.5g | 28.2s | 206.3s |
| Torpedo Cruiser | 1.5g | 28.2s | 206.3s |
| Battlecruiser | 1.5g | 28.2s | 206.3s |
| Battleship | 1.0g | 36.9s | 288.7s |
| Dreadnought | 0.75g | 49.9s | 458.4s |

The corvette (1 launcher) and torpedo cruiser (4 launchers, 12 rounds each) are the
torpedo-armed hulls; all others fight with coilguns.

## Doc/Prompt Invariants

- Numbers quoted in captain/admiral doctrine (`src/llm/prompts.py`) are bound to
  engine constants or seeded runs by `tests/test_doctrine_accuracy.py` - change the
  engine and the tests will tell you which prompt claims drifted.
- After any armor or damage-model change, regenerate the combat tables in
  `docs/ships.md` with `uv run python scripts/calculate_shots_to_kill.py`.
