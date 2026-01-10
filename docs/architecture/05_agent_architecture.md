# LLM Agent Architecture

## Overview

Each ship is crewed by 3 LLM agents working together:
- **Captain**: Strategic decisions, orders to crew
- **Weapons Officer**: Targeting, firing, ammunition management
- **Helmsman**: Navigation, maneuvering, thrust control

## Hierarchical Structure

```
                    +----------+
                    | CAPTAIN  |
                    | (Leader) |
                    +----+-----+
                         |
              Orders/Status|
           +---------------+---------------+
           |                               |
     +-----+------+                 +------+-----+
     |  WEAPONS   |                 |  HELMSMAN  |
     |  OFFICER   |                 |            |
     +------------+                 +------------+
     - Targeting                    - Thrust control
     - Firing                       - Rotation
     - Ammo mgmt                    - Evasion
     - PD priority                  - Intercept
```

## Agent Framework Choice

Based on research, **CrewAI** is recommended for this use case:
- Role-based agent design fits our crew model
- Hierarchical task delegation
- Good performance
- Active development

Alternative: **AutoGen** for more conversational dynamics between agents.

## Communication Protocol

### Turn Structure

```
1. PERCEPTION PHASE
   - All agents receive game state
   - Filtered by their role's perspective

2. CAPTAIN DELIBERATION
   - Captain analyzes situation
   - Issues high-level orders

3. CREW EXECUTION
   - Weapons/Helmsman receive orders
   - Plan specific actions

4. ACTION SUBMISSION
   - Actions validated and queued
   - Physics simulation runs
```

### Message Format

```python
@dataclass
class AgentMessage:
    sender: str  # 'captain', 'weapons', 'helmsman'
    recipient: str
    message_type: str  # 'order', 'status', 'request'
    content: dict
    priority: int  # 1-5, 5 highest
    timestamp: float

# Examples
Order(
    sender='captain',
    recipient='weapons',
    message_type='order',
    content={
        'action': 'engage_target',
        'target': 'enemy_ship',
        'weapon': 'spinal',
        'priority': 'high'
    }
)

Status(
    sender='helmsman',
    recipient='captain',
    message_type='status',
    content={
        'current_heading': [1, 0, 0],
        'velocity': [500, 0, 0],
        'delta_v_remaining': 720000,
        'maneuver_in_progress': True
    }
)
```

## State Representation for Agents

### Common State (All Agents See)

```python
{
    "timestamp": 1234.5,
    "own_ship": {
        "position": [x, y, z],
        "velocity": [vx, vy, vz],
        "heading": [hx, hy, hz],
        "hull_status": {
            "forward": {"hp": 100, "max": 100},
            "central": {"hp": 120, "max": 150},
            "aft": {"hp": 50, "max": 100}
        },
        "heat_percentage": 45,
        "battery_percentage": 78,
        "delta_v_remaining": 720000
    },
    "enemy_ship": {
        "position": [x, y, z],  # Estimated
        "velocity": [vx, vy, vz],  # Estimated
        "heading": [hx, hy, hz],  # If visible
        "estimated_damage": "moderate",  # Based on observations
        "last_update": 1230.0  # Time of last sensor update
    },
    "threats": [
        {"type": "torpedo", "position": [...], "eta": 45.2},
        {"type": "projectile", "position": [...], "eta": 8.1}
    ]
}
```

### Captain-Specific State

```python
{
    "strategic_situation": {
        "engagement_type": "defender",
        "objective": "destroy_or_disable_attacker",
        "alternative": null
    },
    "crew_status": {
        "weapons_officer": "engaged",
        "helmsman": "maneuvering"
    },
    "tactical_assessment": {
        "range": 150000,  # meters
        "closing_rate": 500,  # m/s
        "weapon_envelope": ["turrets", "torpedoes"],
        "threat_level": "moderate"
    },
    "pending_orders": []
}
```

### Weapons Officer State

```python
{
    "weapons": {
        "turret_dorsal": {
            "status": "ready",
            "ammo": 145,
            "target": "enemy_ship",
            "tracking": true
        },
        "turret_ventral": {
            "status": "reloading",
            "ammo": 142,
            "ready_in": 3.2
        },
        "spinal": {
            "status": "charging",
            "charge": 0.75,
            "ammo": 18,
            "ready_in": 5.0
        },
        "pd_lasers": {
            "forward": {"status": "engaging", "target": "torpedo_1"},
            "aft": {"status": "standby"}
        },
        "torpedoes": {
            "loaded": 4,
            "total": 8
        }
    },
    "firing_solutions": {
        "enemy_ship": {
            "turret_hit_prob": 0.65,
            "spinal_hit_prob": 0.82,
            "torpedo_intercept_time": 120
        }
    }
}
```

### Helmsman State

```python
{
    "navigation": {
        "current_heading": [1, 0, 0],
        "target_heading": [0.9, 0.1, 0],
        "rotation_rate": 2.5,  # deg/s
        "time_to_align": 4.0
    },
    "propulsion": {
        "throttle": 0.5,
        "delta_v_remaining": 720000,
        "propellant_percentage": 85,
        "drive_status": "nominal"
    },
    "maneuvers": {
        "current": "intercept",
        "waypoints": [[x1,y1,z1], [x2,y2,z2]],
        "evasion_mode": false
    },
    "thermal": {
        "radiators_extended": true,
        "heat_percentage": 45,
        "heatsink_capacity": 0.65
    }
}
```

## Action Space

### Captain Actions

```python
CAPTAIN_ACTIONS = [
    # Strategic
    "set_engagement_strategy",  # aggressive, defensive, evasive
    "order_disengage",  # Attacker only
    "order_all_stop",  # Emergency

    # Orders to Weapons
    "order_weapons_hold_fire",
    "order_weapons_free",
    "order_engage_target",
    "order_launch_torpedoes",
    "order_point_defense_priority",

    # Orders to Helmsman
    "order_intercept",
    "order_evade",
    "order_maintain_range",
    "order_rotate_to_heading",
    "order_radiators_extend",
    "order_radiators_retract",
]
```

### Weapons Officer Actions

```python
WEAPONS_ACTIONS = [
    # Turrets
    "turret_target_ship",
    "turret_target_torpedo",
    "turret_hold_fire",
    "turret_fire_when_ready",

    # Spinal
    "spinal_begin_charge",
    "spinal_abort_charge",
    "spinal_fire",

    # Point Defense
    "pd_set_priority",  # missiles, projectiles, manual
    "pd_target_specific",
    "pd_hold",

    # Torpedoes
    "torpedo_load",
    "torpedo_set_target",
    "torpedo_launch",
    "torpedo_salvo",  # Launch multiple
]
```

### Helmsman Actions

```python
HELMSMAN_ACTIONS = [
    # Thrust
    "set_throttle",  # 0-100%
    "set_thrust_direction",  # vector

    # Rotation
    "rotate_to_heading",  # target vector
    "rotate_to_target",  # track target
    "maintain_orientation",

    # Maneuvers
    "execute_intercept",  # Calculate and execute
    "execute_evasion",  # Random jinking
    "execute_bracket",  # Circle target

    # Systems
    "extend_radiators",
    "retract_radiators",
    "emergency_thrust",  # Overdrive
]
```

## Agent Prompts

### Captain System Prompt

```
You are the Captain of a combat spacecraft in a tactical engagement.

YOUR ROLE:
- Make strategic decisions for ship survival and mission success
- Issue clear, specific orders to your Weapons Officer and Helmsman
- Coordinate crew actions for maximum effectiveness
- Monitor crew status and adjust plans as needed

YOUR MISSION:
{mission_description}

RULES OF ENGAGEMENT:
1. Preserve the ship when possible
2. Complete mission objectives
3. Communicate clearly with crew
4. React to changing tactical situations

When responding, provide:
1. ASSESSMENT: Brief tactical analysis
2. ORDERS: Specific orders for crew (use exact action names)
3. REASONING: Why these actions achieve objectives
```

### Weapons Officer System Prompt

```
You are the Weapons Officer of a combat spacecraft.

YOUR ROLE:
- Execute weapons-related orders from the Captain
- Manage targeting and firing solutions
- Control point defense priorities
- Report weapons status to Captain

YOUR WEAPONS:
- 2x Coilgun Turrets (6 tubes total, 200km range)
- 1x Spinal Coilgun (500km range, high damage, requires ship alignment)
- 2x Point Defense Lasers (100km range, anti-missile/torpedo)
- 2x Torpedo Launchers (8 torpedoes total, long range)

When responding, provide:
1. ACKNOWLEDGE: Confirm received orders
2. ACTIONS: Specific weapon actions to take
3. STATUS: Current weapons state and any concerns
```

### Helmsman System Prompt

```
You are the Helmsman of a combat spacecraft.

YOUR ROLE:
- Execute navigation orders from the Captain
- Control ship thrust and orientation
- Manage thermal systems (radiators, heatsinks)
- Execute tactical maneuvers

YOUR SYSTEMS:
- Main Drive: 4g max acceleration, 750 km/s delta-v budget
- RCS: Rotation control, can rotate ~5 deg/s
- Radiators: Extend for cooling, retract for combat
- Heatsink: Absorbs heat spikes

When responding, provide:
1. ACKNOWLEDGE: Confirm received orders
2. ACTIONS: Specific navigation/system actions
3. STATUS: Current propulsion and thermal state
```

## LLM Integration

### CrewAI Implementation

```python
from crewai import Agent, Task, Crew, Process

captain = Agent(
    role='Ship Captain',
    goal='Lead the ship to victory while preserving crew and vessel',
    backstory='Experienced combat commander...',
    llm=claude_model,
    allow_delegation=True,
    verbose=True
)

weapons_officer = Agent(
    role='Weapons Officer',
    goal='Effectively employ all weapon systems against designated targets',
    backstory='Expert in ship weapons systems...',
    llm=claude_model,
    allow_delegation=False,
    verbose=True
)

helmsman = Agent(
    role='Helmsman',
    goal='Navigate the ship safely and position for tactical advantage',
    backstory='Skilled pilot with combat experience...',
    llm=claude_model,
    allow_delegation=False,
    verbose=True
)

crew = Crew(
    agents=[captain, weapons_officer, helmsman],
    tasks=[combat_turn_task],
    process=Process.hierarchical,
    manager_agent=captain
)
```

### Token Efficiency

To minimize API costs and latency:

1. **Compressed state**: Send only relevant information
2. **Action templates**: Structured output formats
3. **Caching**: Remember previous decisions
4. **Batching**: Group similar decisions

```python
def prepare_state_for_agent(game_state, agent_role):
    """Create role-appropriate state summary"""

    compressed = {
        "turn": game_state.turn,
        "critical_info": extract_critical_info(game_state, agent_role),
        "recent_events": game_state.events[-5:],  # Last 5 events
    }

    if agent_role == 'captain':
        compressed["crew_status"] = get_crew_summary(game_state)
    elif agent_role == 'weapons':
        compressed["weapons_detail"] = get_weapons_detail(game_state)
    elif agent_role == 'helmsman':
        compressed["nav_detail"] = get_nav_detail(game_state)

    return compressed
```

## Different LLM Models

The system supports different LLM models per ship for comparison:

```python
SHIP_CONFIGS = {
    "alpha": {
        "captain": "claude-3-opus",
        "weapons": "claude-3-sonnet",
        "helmsman": "claude-3-sonnet"
    },
    "beta": {
        "captain": "gpt-4",
        "weapons": "gpt-4-turbo",
        "helmsman": "gpt-4-turbo"
    }
}
```

This allows us to:
- Compare model performance
- Test cost/quality tradeoffs
- Identify model-specific strengths
