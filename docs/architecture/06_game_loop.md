# Game Loop and Simulation

## Turn Structure

The simulation uses a **phased turn** system with continuous physics:

```
┌─────────────────────────────────────────────────────────┐
│                     COMBAT TURN                         │
├─────────────────────────────────────────────────────────┤
│  1. PERCEPTION (0.1s)                                   │
│     - Update sensor readings                            │
│     - Detect threats                                    │
│     - Generate state for agents                         │
├─────────────────────────────────────────────────────────┤
│  2. DECISION (1-5s real time)                           │
│     - Agents process state                              │
│     - Generate orders/actions                           │
│     - Validate actions                                  │
├─────────────────────────────────────────────────────────┤
│  3. EXECUTION (sim time varies)                         │
│     - Queue actions                                     │
│     - Run physics (multiple ticks)                      │
│     - Resolve combat                                    │
│     - Update game state                                 │
├─────────────────────────────────────────────────────────┤
│  4. RESOLUTION                                          │
│     - Check victory conditions                          │
│     - Generate combat log                               │
│     - Prepare next turn                                 │
└─────────────────────────────────────────────────────────┘
```

## Main Game Loop

```python
class CombatSimulation:
    def __init__(self, config):
        self.config = config
        self.game_state = GameState()
        self.ships = {}
        self.agents = {}
        self.combat_log = []
        self.turn = 0

    def run(self):
        """Main simulation loop"""
        self.initialize()

        while not self.is_game_over():
            self.turn += 1

            # Phase 1: Perception
            states = self.perception_phase()

            # Phase 2: Decision
            actions = self.decision_phase(states)

            # Phase 3: Execution
            self.execution_phase(actions)

            # Phase 4: Resolution
            self.resolution_phase()

        return self.get_results()

    def perception_phase(self):
        """Generate state for each agent"""
        states = {}

        for ship_id, ship in self.ships.items():
            # Get sensor data
            sensor_data = self.calculate_sensor_data(ship)

            # Generate per-agent states
            for role in ['captain', 'weapons', 'helmsman']:
                agent_id = f"{ship_id}_{role}"
                states[agent_id] = self.generate_agent_state(
                    ship, sensor_data, role
                )

        return states

    def decision_phase(self, states):
        """Get decisions from all agents"""
        actions = {}

        # Process both ships in parallel
        for ship_id in self.ships:
            ship_actions = self.process_ship_decisions(ship_id, states)
            actions[ship_id] = ship_actions

        return actions

    def execution_phase(self, actions):
        """Execute actions and run physics"""

        # Determine simulation time for this turn
        sim_duration = self.config.turn_duration  # e.g., 5 seconds

        # Queue actions with timing
        action_queue = self.build_action_queue(actions)

        # Run physics simulation
        current_time = 0
        physics_dt = self.config.physics_dt  # e.g., 0.1 seconds

        while current_time < sim_duration:
            # Execute any queued actions for this time
            self.execute_queued_actions(action_queue, current_time)

            # Physics tick
            self.physics_tick(physics_dt)

            # Combat resolution
            self.resolve_combat(current_time)

            # Torpedo/projectile updates
            self.update_projectiles(physics_dt)

            # Thermal update
            self.update_thermal(physics_dt)

            current_time += physics_dt

    def resolution_phase(self):
        """End of turn processing"""

        # Check victory conditions
        if self.check_victory():
            return

        # Log turn summary
        self.log_turn_summary()

        # Cleanup destroyed objects
        self.cleanup()
```

## Physics Tick

```python
def physics_tick(self, dt):
    """Single physics simulation step"""

    for ship in self.ships.values():
        # Apply thrust
        if ship.drive.throttle > 0:
            thrust = ship.drive.get_thrust()
            acceleration = thrust / ship.mass
            direction = ship.get_thrust_direction()
            ship.physics.velocity += direction * acceleration * dt

        # Apply RCS (rotation)
        if ship.rcs.active:
            torque = ship.rcs.get_torque()
            ship.physics.angular_velocity += torque * dt

        # Update position
        ship.physics.position += ship.physics.velocity * dt

        # Update orientation
        ship.physics.orientation = apply_rotation(
            ship.physics.orientation,
            ship.physics.angular_velocity,
            dt
        )

        # Consume propellant
        ship.consume_propellant(dt)

    # Update projectiles
    for projectile in self.projectiles:
        if projectile.is_guided:
            projectile.update_guidance(self.ships, dt)
        projectile.physics.position += projectile.physics.velocity * dt
```

## Combat Resolution

```python
def resolve_combat(self, current_time):
    """Check for hits and resolve damage"""

    # Check projectile impacts
    for projectile in list(self.projectiles):
        for ship in self.ships.values():
            if projectile.source_ship != ship:
                distance = np.linalg.norm(
                    projectile.physics.position - ship.physics.position
                )

                if distance < ship.collision_radius:
                    # Hit!
                    result = self.resolve_hit(projectile, ship)
                    self.combat_log.append(result)
                    self.projectiles.remove(projectile)
                    break

    # Check torpedo proximity detonation
    for torpedo in list(self.torpedoes):
        target = torpedo.target
        if target:
            distance = np.linalg.norm(
                torpedo.physics.position - target.physics.position
            )

            if distance < torpedo.detonation_range:
                result = self.resolve_torpedo_hit(torpedo, target)
                self.combat_log.append(result)
                self.torpedoes.remove(torpedo)

    # PD engagements (continuous)
    self.update_point_defense()
```

## Time Compression

For long-range approaches, use time compression:

```python
def calculate_time_warp(self):
    """Determine appropriate time compression"""

    min_range = self.get_minimum_range()
    threats_exist = len(self.threats) > 0
    weapons_active = self.any_weapons_firing()

    if weapons_active or threats_exist:
        return 1  # Real-time

    if min_range < 100_000:  # 100 km
        return 1  # Real-time
    elif min_range < 500_000:  # 500 km
        return 10  # 10x speed
    elif min_range < 1_000_000:  # 1000 km
        return 100  # 100x speed
    else:
        return 1000  # Max compression
```

## Victory Conditions

```python
def check_victory(self):
    """Check if game has ended"""

    defender = self.ships['defender']
    attacker = self.ships['attacker']

    # Destruction victories
    if defender.is_destroyed():
        return Victory(winner='attacker', reason='defender_destroyed')

    if attacker.is_destroyed():
        return Victory(winner='defender', reason='attacker_destroyed')

    # Disable victory
    if defender.is_disabled():
        return Victory(winner='attacker', reason='defender_disabled')

    if attacker.is_disabled():
        return Victory(winner='defender', reason='attacker_disabled')

    # Disengage victory (attacker only)
    if attacker.has_disengaged(self.station_position):
        return Victory(winner='attacker', reason='reached_station')

    # Time limit
    if self.turn >= self.config.max_turns:
        return Victory(winner='defender', reason='time_limit')

    return None


def is_ship_disabled(ship):
    """A ship is disabled if it can't effectively fight"""
    return (
        ship.drive.is_destroyed or
        (ship.turrets_operational == 0 and ship.torpedoes == 0) or
        ship.thermal.heat_percentage >= 100
    )
```

## Combat Log Format

```python
@dataclass
class CombatLogEntry:
    turn: int
    timestamp: float
    event_type: str
    details: dict

# Event types
EVENTS = [
    'weapon_fired',
    'projectile_hit',
    'projectile_miss',
    'torpedo_launched',
    'torpedo_destroyed',
    'torpedo_hit',
    'pd_engagement',
    'module_damaged',
    'module_destroyed',
    'hardpoint_disabled',
    'ship_destroyed',
    'ship_disabled',
    'heat_warning',
    'power_warning',
    'maneuver_started',
    'maneuver_completed',
]
```

## Configuration Options

```python
@dataclass
class SimulationConfig:
    # Timing
    physics_dt: float = 0.1  # Physics tick rate
    turn_duration: float = 5.0  # Seconds per turn
    max_turns: int = 1000

    # Decision timing
    agent_timeout: float = 30.0  # Max seconds for agent decision

    # Combat settings
    friendly_fire: bool = False
    instant_reload: bool = False  # Debug mode
    god_mode: bool = False  # Debug mode

    # Visualization
    enable_viz: bool = True
    viz_update_rate: float = 1.0  # Seconds

    # Logging
    log_level: str = 'INFO'
    save_replay: bool = True
    replay_path: str = './replays/'

    # LLM settings
    llm_model: str = 'claude-3-opus'
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000
```

## Replay System

```python
class ReplayRecorder:
    def __init__(self, filepath):
        self.filepath = filepath
        self.frames = []

    def record_frame(self, game_state, actions, events):
        frame = {
            'turn': game_state.turn,
            'timestamp': game_state.timestamp,
            'ships': self.serialize_ships(game_state.ships),
            'projectiles': self.serialize_projectiles(game_state.projectiles),
            'actions': actions,
            'events': events,
        }
        self.frames.append(frame)

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'frames': self.frames,
                'result': self.result
            }, f)


class ReplayPlayer:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        self.current_frame = 0

    def step(self):
        if self.current_frame < len(self.data['frames']):
            frame = self.data['frames'][self.current_frame]
            self.current_frame += 1
            return frame
        return None

    def seek(self, frame_num):
        self.current_frame = max(0, min(frame_num, len(self.data['frames'])))
```

## Performance Considerations

1. **Parallel agent processing**: Run both ships' agents concurrently
2. **Batch LLM calls**: Group non-interdependent decisions
3. **Physics optimization**: Use NumPy vectorization
4. **Early termination**: Skip detailed simulation if outcome certain
5. **State caching**: Don't recalculate unchanged values
