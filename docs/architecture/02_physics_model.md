# Physics Model

## Newtonian Movement in 3D Space

### Core Principles

1. **No friction**: Objects maintain velocity unless acted upon
2. **Conservation of momentum**: F = ma applies in all directions
3. **Vector thrust**: Ships can apply thrust in any direction
4. **Rotation**: Ships must rotate to aim weapons/drives

### State Representation

```python
@dataclass
class PhysicsState:
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    mass: float  # kg (changes with fuel consumption)
```

### Integration Method

We'll use **Velocity Verlet integration** for stability:

```python
def integrate(state, acceleration, dt):
    # Position update
    new_position = state.position + state.velocity * dt + 0.5 * acceleration * dt**2

    # Velocity update (requires acceleration at new position,
    # but for constant thrust it's the same)
    new_velocity = state.velocity + acceleration * dt

    return new_position, new_velocity
```

### Time Step Considerations

- **Physics tick**: 0.1 seconds (10 Hz)
- **Combat decisions**: 1-10 seconds (configurable)
- **Variable time warp**: Speed up non-combat phases

## Propulsion Systems

*Values based on Terra Invicta game data - see [08_terra_invicta_reference.md](08_terra_invicta_reference.md)*

### Main Drive (D-He3 Magnetic Confinement Fusion - "Helion Torus" Class)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Exhaust velocity | 690 km/s | D-He3 fusion torch |
| Max thrust | 8.9 MN | 4x thrusters for 600t ship @ 4g |
| Efficiency | 92.5% | Waste heat = 7.5% |
| Vector range | +/- 15 degrees | Off-axis thrust |
| Required reactor | 830 MW | Z-Pinch Fusion II or similar |

**Alternative Drive Options:**
| Drive Type | Thrust | Exhaust Vel | Use Case |
|------------|--------|-------------|----------|
| Neutron Flux Torch | 13 MN | 1700 km/s | High delta-v, self-powered |
| Firestar Fission Lantern | 5 MN | 50 km/s | High thrust, lower Isp |
| Pulsed Plasmoid | 2.2 kN | 425 km/s | Efficient cruise |

```python
def calculate_drive_acceleration(ship):
    thrust_direction = ship.orientation.rotate(ship.thrust_vector)
    thrust_magnitude = ship.drive.thrust * ship.throttle
    return thrust_direction * thrust_magnitude / ship.mass
```

### RCS Thrusters (Attitude Control)

| Parameter | Value |
|-----------|-------|
| Number of thrusters | 12-16 |
| Thrust per thruster | 50-200 kN |
| Response time | 0.1 seconds |
| Propellant | Stored separately (hydrazine or cold gas) |
| Rotation rate | ~5 deg/s max |

```python
def calculate_torque(ship, target_orientation):
    # PID controller for attitude
    error = quaternion_error(ship.orientation, target_orientation)
    torque = ship.rcs.kp * error + ship.rcs.kd * ship.angular_velocity
    return np.clip(torque, -ship.rcs.max_torque, ship.rcs.max_torque)
```

## Delta-V Budget

### Tsiolkovsky Rocket Equation

```
delta_v = v_exhaust * ln(m_initial / m_final)
```

**With Helion Torus Drive (690 km/s exhaust velocity):**

For 400 km/s delta-v:
```
400 = 690 * ln(m_i / m_f)
ln(m_i / m_f) = 0.58
m_i / m_f = 1.79

Mass ratio: 1.79:1
For 600t dry mass: 1074t total (474t propellant)
```

This is very reasonable! The high exhaust velocity makes the delta-v budget achievable.

**Combat Delta-V Allocation:**
| Maneuver Type | Delta-V Budget |
|---------------|----------------|
| Initial approach | 50 km/s |
| Combat maneuvering | 100-150 km/s |
| Evasion reserve | 50 km/s |
| Return/escape | 100-200 km/s |
| **Total** | 300-400 km/s |

### Fuel Consumption

```python
def consume_propellant(ship, thrust, dt):
    mass_flow = thrust / ship.drive.exhaust_velocity
    propellant_used = mass_flow * dt
    ship.propellant -= propellant_used
    ship.mass -= propellant_used
```

## Intercept and Trajectory Calculations

### Time to Intercept

For constant acceleration intercept:
```python
def time_to_intercept(pos1, vel1, pos2, vel2, accel):
    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1
    distance = np.linalg.norm(relative_pos)

    # Simplified: assuming head-on approach
    # Quadratic: d = v*t + 0.5*a*t^2
    # Solve for t
    a = 0.5 * accel
    b = np.dot(relative_vel, relative_pos) / distance
    c = -distance

    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
```

### Lead Calculation for Projectiles

```python
def calculate_lead(shooter_pos, target_pos, target_vel, projectile_vel):
    distance = np.linalg.norm(target_pos - shooter_pos)
    time_of_flight = distance / projectile_vel

    # Iterative refinement
    for _ in range(3):
        predicted_pos = target_pos + target_vel * time_of_flight
        distance = np.linalg.norm(predicted_pos - shooter_pos)
        time_of_flight = distance / projectile_vel

    return predicted_pos
```

## Reference Frames

### Absolute Frame
- Origin: Station position (0, 0, 0)
- Useful for overall positioning
- All physics calculated here

### Relative Frame (Combat)
- Origin: Own ship
- Target positions relative to self
- Used for weapon targeting

```python
def to_relative(absolute_pos, ship_pos, ship_orientation):
    relative = absolute_pos - ship_pos
    return ship_orientation.inverse().rotate(relative)
```

## Combat Range Envelopes

| Range | Characteristics |
|-------|-----------------|
| > 1000 km | Detection only, no effective weapons |
| 200-1000 km | Spinal weapon range, torpedoes effective |
| 50-200 km | Turret engagement range |
| 10-50 km | Point defense active, high hit probability |
| < 10 km | "Knife fight" - very short engagement |

## Physics Edge Cases

### Collision Detection
- Ships are ~100-200m long
- Collision = catastrophic destruction
- Check distance < combined radii

### Numerical Stability
- Use double precision for positions
- Limit maximum velocities (999 km/s)
- Clamp accelerations to physical limits
