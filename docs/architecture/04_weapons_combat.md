# Weapons and Combat System

*Values based on Terra Invicta game data - see [08_terra_invicta_reference.md](08_terra_invicta_reference.md)*

## Weapon Types Overview

| Weapon | Role | Range | Damage Type | Fire Pattern |
|--------|------|-------|-------------|--------------|
| Coilgun Battery | Anti-ship | 750 km | Baryonic (kinetic) | 5-shot salvo, 20s cooldown |
| Spinal Coilgun | Heavy strike | 1000 km | Baryonic (kinetic) | 5-shot salvo, 24s cooldown |
| PD Laser | Defense | 300 km | X-Ray | Continuous, 4s cooldown |
| Torpedo | Long range | Unlimited | Explosive | Single launch, 3-6s cooldown |

## Coilgun Turrets

### Configuration
- 2 turrets (dorsal + ventral)
- Burst fire with salvo system
- Independent targeting per turret

### Parameters (Coilgun Battery Mk3)

```python
@dataclass
class CoilgunTurret:
    name: str
    position: str  # 'dorsal' or 'ventral'

    # Weapon specs (from Terra Invicta)
    mass: float = 80_000  # 80 tons
    magazine_size: int = 1800  # rounds

    # Projectile specs
    projectile_mass: float = 20  # kg per round
    muzzle_velocity: float = 7200  # m/s (7.2 km/s)

    # Fire pattern
    salvo_size: int = 5  # shots per burst
    cooldown: float = 20  # seconds between salvos

    # Performance
    traverse_rate: float = 5  # degrees/second
    max_range: float = 750_000  # 750 km
    accuracy: float = 0.001  # 1 milliradian spread

    # Power/Heat (derived from muzzle energy)
    energy_per_shot: float = 518e6  # ~518 MJ (0.5 * m * v^2)
    heat_per_shot: float = 36e6  # 36 MJ waste heat

    # Status
    current_target: Optional[Entity] = None
    rounds_remaining: int = 1800
    cooldown_timer: float = 0
```

### Damage Calculation

**Critical: Relative Velocity Matters!**

Kinetic damage is based on *relative* velocity between projectile and target:
- Ships closing = additive velocity = MORE damage
- Ships separating = subtractive velocity = LESS damage

```python
def calculate_kinetic_damage(projectile, target, armor, impact_angle):
    """
    Calculate damage from kinetic impact with RELATIVE velocity.

    This is crucial for tactical depth:
    - Head-on approach: projectile 7.2 km/s + ship 4 km/s = 11.2 km/s impact
    - Fleeing target: projectile 7.2 km/s - ship 4 km/s = 3.2 km/s impact
    - KE scales with v^2, so closing doubles velocity = 4x damage!
    """
    # Calculate relative velocity (projectile frame vs target frame)
    relative_velocity = projectile.velocity - target.velocity

    # Impact speed is the component along the line of impact
    impact_direction = (target.position - projectile.position)
    impact_direction = impact_direction / np.linalg.norm(impact_direction)
    impact_speed = abs(np.dot(relative_velocity, impact_direction))

    # Kinetic energy with relative velocity
    kinetic_energy_mj = 0.5 * projectile.mass * impact_speed**2 / 1e6

    # Angle modifier (perpendicular = full damage)
    angle_factor = np.sin(np.radians(impact_angle))
    effective_energy = kinetic_energy_mj * angle_factor

    # Armor reduction (Baryonic damage type)
    armor_thickness = armor.thickness_cm
    half_value = armor.baryonic_half_value
    damage_factor = 0.5 ** (armor_thickness / half_value)

    penetrating_damage = effective_energy * damage_factor
    return penetrating_damage


# Example: Coilgun Battery Mk3 shot
# Muzzle velocity: 7.2 km/s
#
# Case 1: Head-on engagement (both closing at 4 km/s each)
#   Relative velocity = 7.2 + 4 + 4 = 15.2 km/s
#   KE = 0.5 * 20 * 15200^2 = 2.31 GJ per round (4.5x base!)
#
# Case 2: Pursuit (target fleeing at 4 km/s)
#   Relative velocity = 7.2 - 4 = 3.2 km/s
#   KE = 0.5 * 20 * 3200^2 = 102 MJ per round (0.2x base)
#
# Tactical implication: Never fly directly toward enemy guns!
```

### Accuracy and Hit Probability

```python
def calculate_hit_probability(turret, target, distance):
    """Calculate probability of hitting target"""

    # Base accuracy (milliradian spread)
    spread_radius = turret.accuracy * distance

    # Target cross-section
    target_radius = target.cross_section / 2

    # Tracking error (based on relative velocity)
    tracking_error = calculate_tracking_error(turret, target)

    # Combined accuracy
    total_spread = np.sqrt(spread_radius**2 + tracking_error**2)

    # Probability (simplified Gaussian)
    if target_radius >= total_spread:
        return 0.95
    else:
        ratio = target_radius / total_spread
        return min(0.95, ratio**2)
```

## Spinal Coilgun

### Parameters (Spinal Coilgun Mk3)

```python
@dataclass
class SpinalCoilgun:
    # Weapon specs (from Terra Invicta)
    mass: float = 200_000  # 200 tons
    magazine_size: int = 450  # rounds

    # Projectile specs
    projectile_mass: float = 100  # kg per round
    muzzle_velocity: float = 10800  # m/s (10.8 km/s)

    # Fire pattern
    salvo_size: int = 5  # shots per burst
    cooldown: float = 24  # seconds between salvos

    # Performance
    pivot_range: float = 30  # degrees off-axis
    max_range: float = 1_000_000  # 1000 km
    accuracy: float = 0.0005  # 0.5 milliradian

    # Power/Heat (derived from muzzle energy)
    # KE = 0.5 * 100 * 10800^2 = 5.83 GJ per shot
    energy_per_shot: float = 5.83e9  # 5.83 GJ
    heat_per_shot: float = 3.5e9  # ~60% waste heat

    # Status
    rounds_remaining: int = 450
    cooldown_timer: float = 0
```

### Firing Sequence

1. **Aim ship** (helmsman aligns within 30° pivot range)
2. **Acquire lock** (weapons officer confirms targeting)
3. **Fire salvo** (5 shots in rapid succession)
4. **Cooldown** (24 seconds, major heat event)

```python
def fire_spinal_salvo(spinal, ship, target, dt):
    """Fire a 5-shot spinal salvo"""
    if spinal.cooldown_timer > 0:
        return False  # Still cooling down

    if spinal.rounds_remaining < spinal.salvo_size:
        return False  # Insufficient ammo

    # Check alignment (must be within pivot range)
    aim_vector = target.physics.position - ship.physics.position
    aim_vector = aim_vector / np.linalg.norm(aim_vector)
    ship_forward = ship.get_forward_vector()
    angle = np.degrees(np.arccos(np.dot(aim_vector, ship_forward)))

    if angle > spinal.pivot_range:
        return False  # Target outside pivot arc

    # Fire salvo
    for _ in range(spinal.salvo_size):
        launch_projectile(spinal, ship, target)
        spinal.rounds_remaining -= 1
        ship.mass -= spinal.projectile_mass  # Ship gets lighter

    # Apply heat and start cooldown
    ship.thermal.add_heat(spinal.heat_per_shot * spinal.salvo_size)
    spinal.cooldown_timer = spinal.cooldown

    return True
```

## Point Defense Lasers

### Configuration
- 2 PDL units (forward and aft module)
- 180° hemisphere coverage each (360° total)
- Fast tracking, automatic engagement

### Parameters (PD Arc Laser Turret)

```python
@dataclass
class PointDefenseLaser:
    position: str  # 'forward' or 'aft'

    # Weapon specs (from Terra Invicta)
    mass: float = 20_000  # 20 tons
    cooldown: float = 4  # seconds between shots

    # Beam specs
    shot_power: float = 50e6  # 50 MJ per shot
    efficiency: float = 0.35  # 35% to target, 65% waste heat
    wavelength: float = 1080e-9  # 1080 nm (IR)
    mirror_radius: float = 0.30  # 30 cm

    # Performance
    max_range: float = 300_000  # 300 km effective
    pivot_range: float = 180  # degrees (hemisphere)
    tracking_rate: float = 50  # degrees/second (fast)

    # Power/Heat
    damage_per_shot: float = 17.5e6  # 17.5 MJ delivered (50 * 0.35)
    heat_per_shot: float = 32.5e6  # 32.5 MJ waste heat

    # Effectiveness (X-Ray damage type vs targets)
    attack_mode: bool = False  # Cannot target ships
    defense_mode: bool = True  # Can auto-engage missiles

    # Status
    current_target: Optional[Entity] = None
    cooldown_timer: float = 0
```

### Engagement Logic

```python
def pdl_engagement_priority(threats):
    """Sort threats by priority"""
    def priority_score(threat):
        time_to_impact = threat.distance / threat.closing_velocity
        damage_potential = threat.damage

        # Higher score = engage first
        return damage_potential / max(time_to_impact, 0.1)

    return sorted(threats, key=priority_score, reverse=True)


def update_pdl(pdl, ship, threats, dt):
    if not pdl.current_target:
        # Find new target
        sorted_threats = pdl_engagement_priority(threats)
        for threat in sorted_threats:
            if pdl.can_engage(threat):
                pdl.current_target = threat
                pdl.engagement_time = 0
                break

    if pdl.current_target:
        # Continue engaging
        pdl.engagement_time += dt

        # Check if destroyed
        required_time = pdl.dwell_time_missile if threat.is_missile else pdl.dwell_time_projectile

        if pdl.engagement_time >= required_time:
            pdl.current_target.destroyed = True
            pdl.current_target = None
```

### PD vs Projectile Types (Mass-Based Interception)

Point defense lasers destroy projectiles by **vaporizing** them. Heavier projectiles
require more energy (more shots) to ablate. This is why spinal slugs are so hard to stop.

```python
def calculate_pd_shots_needed(pdl, projectile):
    """
    Calculate how many PD shots needed to destroy a projectile.

    Lasers vaporize material. Heavier = more mass to ablate.
    Uses armor's heat_of_vaporization (MJ/kg) if available.
    """
    # Energy delivered per shot
    damage_per_shot = pdl.shot_power * pdl.efficiency  # 17.5 MJ for Arc Laser

    # Energy needed to vaporize projectile
    # Typical metal: ~10 MJ/kg to vaporize (heat + phase change)
    heat_of_vaporization = 10  # MJ/kg (simplified)
    energy_to_destroy = projectile.mass * heat_of_vaporization

    shots_needed = math.ceil(energy_to_destroy / damage_per_shot)
    return shots_needed
```

**Solid Slugs (must vaporize):**
| Target | Mass (kg) | Energy to Destroy | PD Shots Needed | Time @ 4s cooldown |
|--------|-----------|-------------------|-----------------|-------------------|
| Turret slug (20kg) | 20 | 200 MJ | 12 | 48 seconds |
| Heavy slug (40kg) | 40 | 400 MJ | 23 | 92 seconds |
| Spinal slug (100kg) | 100 | 1 GJ | 58 | 232 seconds |

**Torpedoes/Missiles (disable electronics OR cook-off warhead):**
| Target | Kill Method | Energy Needed | PD Shots | Result |
|--------|-------------|---------------|----------|--------|
| Missile/Torpedo | Fry electronics | 50-100 MJ | 3-6 | Inert slug (evadable) |
| Missile/Torpedo | Warhead cook-off | 200-500 MJ | 12-30 | Premature detonation |

**Tactical Implications:**
- PD lasers ARE effective against torpedoes - disable guidance, don't vaporize
- Disabled torpedo = inert 1600kg slug at 3-4 km/s → evade with small burn
- Turret slugs can be intercepted but require sustained fire
- Spinal slugs are effectively unstoppable by PD (solid metal, no electronics)
- Multiple PD turrets focusing on same target reduces time proportionally

### Realistic PD Engagement Model

```python
@dataclass
class Projectile:
    mass: float
    has_electronics: bool  # True for torpedoes/missiles
    has_warhead: bool      # True for torpedoes/missiles
    heat_absorbed: float = 0  # MJ accumulated from PD hits
    is_disabled: bool = False
    is_detonated: bool = False

    # Thermal thresholds
    electronics_failure_heat: float = 75  # MJ to fry guidance
    warhead_cookoff_heat: float = 350     # MJ to detonate warhead


def update_pdl(pdl, threats, dt):
    """PD engages threats - vaporizes slugs, disables guided weapons"""

    if not pdl.current_target or pdl.current_target.destroyed:
        # Prioritize: torpedoes first (can disable), then light slugs
        sorted_threats = sorted(threats, key=lambda t: (
            t.time_to_impact,
            not t.has_electronics,  # Guided weapons first (easier to kill)
            t.mass  # Lighter slugs next
        ))

        for threat in sorted_threats:
            # Skip heavy solid slugs - waste of shots
            if not threat.has_electronics and threat.mass > 50:
                continue
            if pdl.can_engage(threat):
                pdl.current_target = threat
                break

    if pdl.current_target and pdl.cooldown_timer <= 0:
        threat = pdl.current_target
        damage_mj = pdl.damage_per_shot

        if threat.has_electronics:
            # Heat accumulates - fry electronics or cook off warhead
            threat.heat_absorbed += damage_mj

            if threat.heat_absorbed >= threat.warhead_cookoff_heat:
                threat.is_detonated = True  # Premature detonation (safe)
                threat.destroyed = True
            elif threat.heat_absorbed >= threat.electronics_failure_heat:
                threat.is_disabled = True   # Now inert ballistic slug
                threat.has_electronics = False
                # Don't mark destroyed - still a (evadable) threat
                pdl.current_target = None   # Move to next target
        else:
            # Solid slug - must vaporize
            mass_ablated = damage_mj / 10  # 10 MJ/kg
            threat.mass -= mass_ablated
            if threat.mass <= 0:
                threat.destroyed = True
                pdl.current_target = None

        pdl.cooldown_timer = pdl.cooldown
```

## Torpedoes

### Parameters (Cobra-class Torpedo)

```python
@dataclass
class Torpedo:
    # Physical (from Terra Invicta)
    mass: float = 1600  # kg total
    warhead_mass: float = 200  # kg explosive

    # Propulsion
    max_acceleration: float = 146  # m/s^2 (14.9g)
    delta_v: float = 3170  # m/s (3.17 km/s)
    # Note: Burn time = delta_v / accel = ~22 seconds

    # Guidance
    guidance_type: str = 'inertial_terminal_ir'
    terminal_acquisition_range: float = 50_000  # 50 km
    turn_rate: float = 35  # degrees/second
    maneuver_angle: float = 45  # degrees max turn

    # Combat
    damage_mj: float = 540  # MJ explosive yield
    evasion_capability: float = 5  # g lateral during terminal

    # Launcher specs
    magazine_size: int = 16  # per launcher
    salvo_size: int = 6  # missiles per salvo
    cooldown: float = 4  # seconds between salvos

    # Status
    physics: PhysicsState
    fuel_remaining: float = 1.0
    is_terminal: bool = False
```

### Torpedo Mass Budget

Each torpedo launch reduces ship mass:
- Per torpedo: 1600 kg
- Full salvo (6): 9.6 tons
- Full magazine (16): 25.6 tons per launcher

### Torpedo Behavior

```python
def update_torpedo(torpedo, target, dt):
    if torpedo.fuel_remaining <= 0:
        # Ballistic - maintain course
        return

    distance = np.linalg.norm(target.physics.position - torpedo.physics.position)

    if distance < torpedo.terminal_acquisition_range:
        # Terminal homing
        torpedo.is_terminal = True

        # Proportional navigation
        los = target.physics.position - torpedo.physics.position
        los_rate = calculate_los_rate(torpedo, target)

        # Navigate to intercept
        accel_command = 3 * torpedo.max_acceleration * los_rate
        accel_command = np.clip(accel_command, -torpedo.max_acceleration, torpedo.max_acceleration)

    else:
        # Mid-course: head toward predicted intercept
        intercept_point = predict_intercept(torpedo, target)
        accel_command = steer_to_point(torpedo, intercept_point)

    # Apply thrust and consume fuel
    apply_acceleration(torpedo, accel_command, dt)
```

### Counter-Torpedo Options

1. **Point Defense**: Destroy with lasers
2. **Decoys**: Deploy to distract terminal guidance
3. **Jamming**: Disrupt guidance (if electronic)
4. **Maneuvering**: Out-accelerate the torpedo
5. **Kinetic intercept**: Shoot it with turrets

## Damage Model

### Damage Types (Terra Invicta System)

| Damage Type | Source | Armor Counter |
|-------------|--------|---------------|
| **Baryonic (Kinetic)** | Coilguns, Railguns, Missiles | High baryonicHalfValue |
| **X-Ray** | Lasers, Particle Beams | High xRayHalfValue |

### Armor Half-Value System

Damage is reduced exponentially by armor:
```
penetrating_damage = base_damage * 0.5^(thickness / half_value)
```

**Example materials:**
| Armor | Baryonic Half (cm) | X-Ray Half (cm) |
|-------|-------------------|-----------------|
| Adamantane | 115.8 | 18.0 |
| Nanotube | 155.4 | 19.9 |
| Composite | 72.0 | 15.3 |

### Damage Resolution Flow

```
Weapon Fired
    |
    v
Hit Check (accuracy vs distance/evasion)
    |
    v
Determine Hit Location (nose/lateral/tail)
    |
    v
Armor Reduction (half-value calculation)
    |
    v
Module Damage (apply MJ to HP)
    |
    v
Critical Check (based on damage ratio)
    |
    v
Overflow Check (to adjacent module?)
```

### Implementation

```python
def resolve_attack(weapon, target, hit_location):
    """Resolve a weapon hit with Terra Invicta armor model"""

    # Determine damage type
    if weapon.type in ['coilgun', 'railgun', 'torpedo']:
        damage_type = 'baryonic'
    else:  # laser
        damage_type = 'xray'

    # Get base damage (MJ)
    base_damage_mj = weapon.damage_mj

    # Armor reduction
    armor = target.get_armor(hit_location)
    if damage_type == 'baryonic':
        half_value = armor.baryonic_half_value
    else:
        half_value = armor.xray_half_value

    armor_factor = 0.5 ** (armor.thickness_cm / half_value)
    penetrating_damage = base_damage_mj * armor_factor

    # Degrade armor (ablation)
    armor.thickness_cm -= base_damage_mj * 0.001  # 0.1% ablation

    # Get module
    module = target.get_module(hit_location)

    # Apply damage (convert MJ to module HP)
    # Assume 10 MJ = 1 HP for game balance
    hp_damage = penetrating_damage / 10e6
    module.current_hp -= hp_damage

    # Critical hit check
    damage_ratio = hp_damage / module.max_hp
    critical_chance = min(0.5, damage_ratio)

    for hardpoint in module.hardpoints:
        if random.random() < critical_chance:
            hardpoint.disabled = True

    # Check overflow to adjacent module
    if module.current_hp < 0:
        overflow = abs(module.current_hp) * 10e6  # back to MJ
        module.current_hp = 0
        next_module = target.get_adjacent_module(module)
        if next_module:
            resolve_overflow(next_module, overflow, armor_factor)

    return DamageResult(
        damage_dealt_mj=penetrating_damage,
        critical_hits=[h for h in module.hardpoints if h.disabled],
        module_destroyed=module.is_destroyed
    )
```

### Critical Hit Modifiers

| Damage Ratio | Modifier |
|--------------|----------|
| < 25% HP | 0.5x |
| 25-50% HP | 1.0x |
| 50-75% HP | 1.5x |
| > 75% HP | 2.0x |

### Module Destruction Effects

| Module | Effect When Destroyed |
|--------|-----------------------|
| Forward | Blind (no sensors), no forward PDL |
| Central | Reactor damaged, weapons offline |
| Aft | Drive crippled, no rear PDL |

## Combat Log

All combat events should be logged for replay and analysis:

```python
@dataclass
class CombatEvent:
    timestamp: float
    event_type: str
    attacker: str
    target: str
    weapon: str
    hit_location: str
    damage: float
    result: str  # 'hit', 'miss', 'absorbed', 'critical'
    details: dict
```
