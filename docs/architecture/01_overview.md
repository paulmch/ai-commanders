# AI Commanders - Architecture Overview

## Project Vision

A realistic space combat simulator where LLM agents control ships in tactical engagements, inspired by Terra Invicta's space combat system.

## Core Concept

Two ships engage in Newtonian space combat near a station:
- **Defender**: Protect the station by destroying/disabling the attacker
- **Attacker**: Destroy the defender OR disengage to reach the station

## System Architecture

```
+------------------+     +------------------+
|   Ship Alpha     |     |   Ship Beta      |
|  (3 LLM Agents)  |     |  (3 LLM Agents)  |
|   - Captain      |     |   - Captain      |
|   - Weapons      |     |   - Weapons      |
|   - Helmsman     |     |   - Helmsman     |
+--------+---------+     +---------+--------+
         |                         |
         v                         v
+--------------------------------------------------+
|              GAME STATE MANAGER                   |
|  - Physics Engine (Newtonian)                    |
|  - Combat Resolution                             |
|  - Damage Model                                  |
|  - Heat/Power Systems                            |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|              VISUALIZATION (Optional)             |
|  - 3D Render (Panda3D/Plotly)                    |
|  - State Dashboard                               |
|  - Combat Log                                    |
+--------------------------------------------------+
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Runtime | Python 3.11+ with UV | Package management and execution |
| Physics | NumPy/SciPy | Vector math, physics calculations |
| LLM Framework | Claude API / CrewAI | Multi-agent orchestration |
| Visualization | Plotly 3D / Panda3D | Optional 3D rendering |
| Data | JSON/Dataclasses | State serialization |
| Logging | Python logging | Combat log, replay system |

## Key Design Principles

1. **Physics First**: All movement and combat follows Newtonian mechanics
2. **Information Asymmetry**: Agents see only what their sensors detect
3. **Turn-Based with Continuous Physics**: Discrete decision points, continuous simulation
4. **Modular Design**: Components can be swapped (LLM models, visualization)
5. **Reproducible**: Deterministic physics with seeded randomness for replay

## Document Index

| Document | Description |
|----------|-------------|
| [01_overview.md](01_overview.md) | This document - system overview |
| [02_physics_model.md](02_physics_model.md) | Newtonian physics implementation |
| [03_ship_systems.md](03_ship_systems.md) | Ship components and mechanics |
| [04_weapons_combat.md](04_weapons_combat.md) | Weapons and damage model |
| [05_agent_architecture.md](05_agent_architecture.md) | LLM agent design |
| [06_game_loop.md](06_game_loop.md) | Simulation loop and turn structure |
| [07_questions.md](07_questions.md) | Open questions for discussion |
