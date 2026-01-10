# AI Commanders - Development Guide

## Project Overview
Terra Invicta-inspired space battle simulator with LLM-controlled ships.

## Setup

```bash
# Use uv for Python package management
uv venv
source .venv/bin/activate
uv pip install numpy pytest
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
ai-commanders/
├── data/
│   └── fleet_ships.json    # Ship specs, weapons, armor, attitude control
├── src/
│   ├── physics.py          # Newtonian mechanics, trajectories, rotation
│   ├── combat.py           # Weapons, armor, damage resolution
│   └── command.py          # Hierarchical control architecture
├── tests/
│   ├── test_physics.py     # 129 tests - vectors, delta-v, thrust, rotation
│   ├── test_combat.py      # 35 tests - weapons, armor, penetration
│   ├── test_command.py     # 81 tests - commands, tactical/strategic control
│   └── test_scenarios.py   # Integration tests - battle scenarios
└── scripts/
    ├── ship_calculator_v2.py
    ├── armor_calculator.py
    └── attitude_control.py
```

## Key Constants

- Exhaust velocity: 10,256 km/s
- Main thrust: 58.56 MN (Protium Converter Torch x6)
- Combat thrust vectoring: 1° deflection
- Target delta-v: 500 km/s

## Control Architecture

Hierarchical control (Option 3):
1. **Strategic LLM**: Called every 30-60s, sets objectives and priorities
2. **Tactical Layer**: Rule-based, executes between LLM calls
   - Keep nose toward target
   - Fire when in range
   - Evade incoming torpedoes

## Ship Classes

| Ship | Accel | 90° Turn (TV) | 90° Turn (RCS) |
|------|-------|---------------|----------------|
| Corvette | 3.0g | 12.1s | 54.2s |
| Frigate | 3.0g | 15.1s | 83.3s |
| Destroyer | 2.0g | 20.6s | 127.6s |
| Cruiser | 1.5g | 28.2s | 206.3s |
| Battlecruiser | 1.5g | 28.2s | 206.3s |
| Battleship | 1.0g | 36.9s | 288.7s |
| Dreadnought | 0.75g | 49.9s | 458.4s |
